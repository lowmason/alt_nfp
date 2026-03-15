"""QCEW revision stability analysis by month-in-quarter (M1, M2, M3).

Reads the Hive-partitioned vintage store, computes growth-rate revisions
between each vintage and the final vintage, and reports stability metrics
stratified by month-in-quarter, quarter, era, and revision number.

Outputs (all written to ``output/qcew_revisions/``):
  - summary.parquet / summary.csv — main metrics table
  - revision_rmse_by_miq.png — RMSE bar chart (M1/M2/M3)
  - revision_decay_curves.png — RMSE by revision number per month tier
  - revision_bias.png — signed mean revision by month tier
  - revision_qq.png — QQ plots vs Normal and Student-t(5)
  - revision_era_comparison.png — pre-COVID vs post-COVID by month tier
  - revision_autocorrelation.png — serial correlation of revision errors
  - revision_by_quarter_miq.png — heatmap of RMSE by quarter × month

Usage::

    uv run python scripts/qcew_revision_stability.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from nfp_models.config import ERA_BREAKS, OUTPUT_DIR, QCEW_NU, STORE_DIR

OUT_DIR = OUTPUT_DIR / "qcew_revisions"

MIQ_LABELS = {1: "M1", 2: "M2", 3: "M3"}
ERA_LABELS = {0: "Pre-COVID\n(2017–2019)", 1: "Post-COVID\n(2022+)"}
ERA_CUTOFF = ERA_BREAKS[0]  # 2020-01-01

CALIBRATION_START = date(2017, 1, 12)
PRE_COVID_END = date(2019, 12, 12)
POST_COVID_START = date(2022, 1, 12)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_qcew_national() -> pl.DataFrame:
    """Read QCEW national-level data from the vintage store."""
    df = pl.read_parquet(
        STORE_DIR / "source=qcew" / "**" / "*.parquet",
        hive_partitioning=True,
    )
    return df.filter(
        pl.col("geographic_type") == "national",
        pl.col("employment").is_not_null(),
        pl.col("employment") > 0,
    )


def _compute_growth(df: pl.DataFrame) -> pl.DataFrame:
    """Compute log-growth within each (industry, revision) group."""
    group_key = ["industry_type", "industry_code", "revision"]
    return (
        df.sort(*group_key, "ref_date")
        .with_columns(
            pl.col("employment")
            .log()
            .diff()
            .over(group_key)
            .alias("growth")
        )
        .filter(pl.col("growth").is_not_null() & pl.col("growth").is_finite())
    )


def _build_revision_diffs(df: pl.DataFrame) -> pl.DataFrame:
    """Compute growth_rev - growth_final for every non-final observation.

    Returns a frame with columns: ref_date, industry_type, industry_code,
    revision, growth_rev, growth_final, revision_error, quarter,
    month_in_quarter, era.
    """
    df = df.with_columns(
        ((pl.col("ref_date").dt.month() - 1) // 3 + 1).alias("quarter"),
        (((pl.col("ref_date").dt.month() - 1) % 3) + 1).alias("month_in_quarter"),
    )

    series_key = ["industry_type", "industry_code", "ref_date"]

    # For each (series, ref_date), identify the final revision
    final_rev_per_obs = (
        df.group_by(series_key)
        .agg(pl.col("revision").max().alias("max_rev"))
    )

    df = df.join(final_rev_per_obs, on=series_key, how="left")

    # Only keep months where we have the theoretical final revision
    max_rev_map = {1: 4, 2: 3, 3: 2, 4: 1}
    df = df.with_columns(
        pl.col("quarter").replace_strict(max_rev_map, return_dtype=pl.UInt8)
        .alias("expected_final_rev")
    )
    has_final = df.filter(pl.col("max_rev") >= pl.col("expected_final_rev"))

    # Get final growth values
    final_growth = (
        has_final.filter(pl.col("revision") == pl.col("max_rev"))
        .select(*series_key, pl.col("growth").alias("growth_final"))
    )

    # Get non-final growth values
    non_final = has_final.filter(pl.col("revision") < pl.col("max_rev"))

    # Join to get revision error
    result = non_final.join(final_growth, on=series_key, how="inner").with_columns(
        (pl.col("growth") - pl.col("growth_final")).alias("revision_error"),
    )

    # Assign era
    result = result.with_columns(
        pl.when(
            (pl.col("ref_date") >= CALIBRATION_START)
            & (pl.col("ref_date") <= PRE_COVID_END)
        )
        .then(pl.lit(0))
        .when(pl.col("ref_date") >= POST_COVID_START)
        .then(pl.lit(1))
        .otherwise(pl.lit(-1))
        .alias("era")
    )

    return result.select(
        "ref_date",
        "industry_type",
        "industry_code",
        "revision",
        pl.col("growth").alias("growth_rev"),
        "growth_final",
        "revision_error",
        "quarter",
        "month_in_quarter",
        "era",
    )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def _compute_summary(diffs: pl.DataFrame) -> pl.DataFrame:
    """Compute RMSE, MAE, mean, kurtosis grouped by all relevant dimensions."""
    groups = ["quarter", "month_in_quarter", "era", "revision"]

    summary = (
        diffs.filter(pl.col("era") >= 0)
        .group_by(groups)
        .agg(
            pl.col("revision_error").count().alias("n_obs"),
            (pl.col("revision_error").pow(2).mean().sqrt()).alias("rmse"),
            pl.col("revision_error").abs().mean().alias("mae"),
            pl.col("revision_error").mean().alias("mean_revision"),
            pl.col("revision_error").std().alias("std_revision"),
        )
        .sort("era", "quarter", "month_in_quarter", "revision")
    )

    return summary


def _compute_overall_summary(diffs: pl.DataFrame) -> pl.DataFrame:
    """RMSE/MAE/mean collapsed across quarters for each (miq, era, rev)."""
    groups = ["month_in_quarter", "era", "revision"]
    return (
        diffs.filter(pl.col("era") >= 0)
        .group_by(groups)
        .agg(
            pl.col("revision_error").count().alias("n_obs"),
            (pl.col("revision_error").pow(2).mean().sqrt()).alias("rmse"),
            pl.col("revision_error").abs().mean().alias("mae"),
            pl.col("revision_error").mean().alias("mean_revision"),
            pl.col("revision_error").std().alias("std_revision"),
        )
        .sort("era", "month_in_quarter", "revision")
    )


def _compute_miq_overall(diffs: pl.DataFrame) -> pl.DataFrame:
    """Overall RMSE/MAE by month-in-quarter, pooling eras and revisions."""
    return (
        diffs.filter(pl.col("era") >= 0)
        .group_by("month_in_quarter")
        .agg(
            pl.col("revision_error").count().alias("n_obs"),
            (pl.col("revision_error").pow(2).mean().sqrt()).alias("rmse"),
            pl.col("revision_error").abs().mean().alias("mae"),
            pl.col("revision_error").mean().alias("mean_revision"),
            pl.col("revision_error").std().alias("std_revision"),
        )
        .sort("month_in_quarter")
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_rmse_by_miq(diffs: pl.DataFrame, out_dir: Path) -> None:
    """Bar chart: overall RMSE by M1/M2/M3."""
    miq_df = _compute_miq_overall(diffs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    labels = [MIQ_LABELS[m] for m in miq_df["month_in_quarter"].to_list()]
    colors = ["#d62728", "#2ca02c", "#d62728"]  # red for boundary, green for M2

    for ax, metric, title in zip(
        axes, ["rmse", "mae"], ["RMSE", "MAE"]
    ):
        vals = miq_df[metric].to_list()
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel(f"{title} (log growth-rate units)")
        ax.set_title(f"QCEW Revision {title} by Month-in-Quarter")
        ax.set_xlabel("Month in Quarter")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + max(vals) * 0.02,
                f"{v:.5f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_dir / "revision_rmse_by_miq.png", dpi=150)
    plt.close(fig)


def _plot_decay_curves(diffs: pl.DataFrame, out_dir: Path) -> None:
    """RMSE vs revision number, one line per M1/M2/M3."""
    overall = _compute_overall_summary(diffs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    miq_colors = {1: "#d62728", 2: "#2ca02c", 3: "#ff7f0e"}

    for ax, (era_val, era_label) in zip(axes, ERA_LABELS.items()):
        era_data = overall.filter(pl.col("era") == era_val)
        for miq in [1, 2, 3]:
            sub = era_data.filter(pl.col("month_in_quarter") == miq).sort("revision")
            if sub.is_empty():
                continue
            ax.plot(
                sub["revision"].to_list(),
                sub["rmse"].to_list(),
                marker="o",
                label=MIQ_LABELS[miq],
                color=miq_colors[miq],
                linewidth=2,
            )
        ax.set_xlabel("Revision Number")
        ax.set_ylabel("RMSE (log growth-rate)")
        ax.set_title(f"Revision Decay — {era_label}")
        ax.legend()
        ax.set_xticks(range(5))

    fig.tight_layout()
    fig.savefig(out_dir / "revision_decay_curves.png", dpi=150)
    plt.close(fig)


def _plot_bias(diffs: pl.DataFrame, out_dir: Path) -> None:
    """Signed mean revision by month-in-quarter and quarter."""
    era_filtered = diffs.filter(pl.col("era") >= 0)

    # By quarter × miq
    bias = (
        era_filtered.group_by("quarter", "month_in_quarter")
        .agg(
            pl.col("revision_error").mean().alias("mean"),
            pl.col("revision_error").std().alias("std"),
            pl.col("revision_error").count().alias("n"),
        )
        .sort("quarter", "month_in_quarter")
        .with_columns(
            (pl.col("std") / pl.col("n").sqrt()).alias("se")
        )
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = []
    x_labels = []
    colors = []
    means = []
    ses = []

    for i, row in enumerate(bias.iter_rows(named=True)):
        x_positions.append(i)
        x_labels.append(f"Q{row['quarter']}-{MIQ_LABELS[row['month_in_quarter']]}")
        colors.append("#2ca02c" if row["month_in_quarter"] == 2 else "#d62728")
        means.append(row["mean"])
        ses.append(row["se"] * 1.96)

    ax.bar(x_positions, means, yerr=ses, color=colors, edgecolor="black",
           linewidth=0.5, capsize=3, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Revision Error (log growth-rate)")
    ax.set_title("QCEW Revision Bias by Quarter × Month-in-Quarter")
    fig.tight_layout()
    fig.savefig(out_dir / "revision_bias.png", dpi=150)
    plt.close(fig)


def _plot_qq(diffs: pl.DataFrame, out_dir: Path) -> None:
    """QQ plots of revision errors vs Normal and Student-t(5) by month tier."""
    era_filtered = diffs.filter(pl.col("era") >= 0)

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    for row_idx, miq in enumerate([1, 2, 3]):
        errors = (
            era_filtered.filter(pl.col("month_in_quarter") == miq)["revision_error"]
            .to_numpy()
        )
        if len(errors) < 5:
            continue

        standardized = (errors - errors.mean()) / errors.std()

        # vs Normal
        ax_norm = axes[row_idx, 0]
        stats.probplot(standardized, dist="norm", plot=ax_norm)
        ax_norm.set_title(f"{MIQ_LABELS[miq]} vs Normal")
        ax_norm.get_lines()[0].set_markerfacecolor("#1f77b4")
        ax_norm.get_lines()[0].set_markersize(3)

        # vs Student-t(nu)
        ax_t = axes[row_idx, 1]
        stats.probplot(standardized, dist=stats.t, sparams=(QCEW_NU,), plot=ax_t)
        ax_t.set_title(f"{MIQ_LABELS[miq]} vs Student-t(ν={QCEW_NU})")
        ax_t.get_lines()[0].set_markerfacecolor("#ff7f0e")
        ax_t.get_lines()[0].set_markersize(3)

    fig.suptitle("QQ Plots of QCEW Revision Errors", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "revision_qq.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_era_comparison(diffs: pl.DataFrame, out_dir: Path) -> None:
    """Side-by-side RMSE comparison: pre-COVID vs post-COVID by month tier."""
    overall = _compute_overall_summary(diffs)

    # Pool across revisions within each (era, miq)
    pooled = (
        diffs.filter(pl.col("era") >= 0)
        .group_by("era", "month_in_quarter")
        .agg(
            (pl.col("revision_error").pow(2).mean().sqrt()).alias("rmse"),
            pl.col("revision_error").abs().mean().alias("mae"),
            pl.col("revision_error").count().alias("n_obs"),
        )
        .sort("era", "month_in_quarter")
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    bar_width = 0.35
    x = np.arange(3)

    for ax, metric, title in zip(axes, ["rmse", "mae"], ["RMSE", "MAE"]):
        for era_val, era_label, offset, color in [
            (0, "Pre-COVID (2017–19)", -bar_width / 2, "#1f77b4"),
            (1, "Post-COVID (2022+)", bar_width / 2, "#d62728"),
        ]:
            era_data = pooled.filter(pl.col("era") == era_val).sort("month_in_quarter")
            vals = era_data[metric].to_list()
            if len(vals) == 3:
                bars = ax.bar(x + offset, vals, bar_width, label=era_label,
                              color=color, edgecolor="black", linewidth=0.5)
                for bar, v in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        v + max(vals) * 0.02,
                        f"{v:.5f}",
                        ha="center", va="bottom", fontsize=8,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(["M1", "M2", "M3"])
        ax.set_ylabel(f"{title} (log growth-rate)")
        ax.set_title(f"QCEW Revision {title}: Pre- vs Post-COVID")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "revision_era_comparison.png", dpi=150)
    plt.close(fig)


def _plot_autocorrelation(diffs: pl.DataFrame, out_dir: Path) -> None:
    """Serial correlation of revision errors across consecutive quarters."""
    era_filtered = diffs.filter(pl.col("era") >= 0)

    # For national total (industry_code='00'), aggregate revision errors by ref_date
    nat_errors = (
        era_filtered.filter(pl.col("industry_code") == "00")
        .group_by("ref_date", "month_in_quarter")
        .agg(pl.col("revision_error").mean().alias("mean_error"))
        .sort("ref_date")
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, miq in zip(axes, [1, 2, 3]):
        errors = (
            nat_errors.filter(pl.col("month_in_quarter") == miq)
            .sort("ref_date")["mean_error"]
            .to_numpy()
        )
        if len(errors) < 6:
            ax.set_title(f"{MIQ_LABELS[miq]} — insufficient data")
            continue

        max_lag = min(8, len(errors) // 3)
        lags = range(1, max_lag + 1)
        acf_vals = [np.corrcoef(errors[:-lag], errors[lag:])[0, 1] for lag in lags]

        ax.bar(list(lags), acf_vals, color="#1f77b4", edgecolor="black", linewidth=0.5)
        # Approximate 95% CI under white noise
        ci = 1.96 / np.sqrt(len(errors))
        ax.axhline(ci, linestyle="--", color="gray", linewidth=0.8)
        ax.axhline(-ci, linestyle="--", color="gray", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Lag (quarters)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Revision Error ACF — {MIQ_LABELS[miq]}")
        ax.set_ylim(-0.6, 0.6)

    fig.tight_layout()
    fig.savefig(out_dir / "revision_autocorrelation.png", dpi=150)
    plt.close(fig)


def _plot_heatmap(diffs: pl.DataFrame, out_dir: Path) -> None:
    """Heatmap of RMSE by quarter × month-in-quarter."""
    era_filtered = diffs.filter(pl.col("era") >= 0)

    pivot = (
        era_filtered.group_by("quarter", "month_in_quarter")
        .agg((pl.col("revision_error").pow(2).mean().sqrt()).alias("rmse"))
        .sort("quarter", "month_in_quarter")
    )

    # Build 4×3 matrix (quarters × months)
    mat = np.full((4, 3), np.nan)
    for row in pivot.iter_rows(named=True):
        mat[row["quarter"] - 1, row["month_in_quarter"] - 1] = row["rmse"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(3))
    ax.set_xticklabels(["M1", "M2", "M3"])
    ax.set_yticks(range(4))
    ax.set_yticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("Month in Quarter")
    ax.set_ylabel("Quarter")
    ax.set_title("QCEW Revision RMSE: Quarter × Month-in-Quarter")

    for i in range(4):
        for j in range(3):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.5f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="RMSE (log growth-rate)")
    fig.tight_layout()
    fig.savefig(out_dir / "revision_by_quarter_miq.png", dpi=150)
    plt.close(fig)




# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------


def _print_report(diffs: pl.DataFrame) -> None:
    """Print a human-readable summary to stdout."""
    print("=" * 72)
    print("QCEW Revision Stability Analysis")
    print("=" * 72)

    miq_overall = _compute_miq_overall(diffs)
    print("\n--- Overall RMSE / MAE by Month-in-Quarter (pooled eras) ---")
    print(f"{'MIQ':>5} {'N':>6} {'RMSE':>10} {'MAE':>10} {'Mean':>10} {'Std':>10}")
    for row in miq_overall.iter_rows(named=True):
        print(
            f"{MIQ_LABELS[row['month_in_quarter']]:>5} "
            f"{row['n_obs']:>6} "
            f"{row['rmse']:>10.6f} "
            f"{row['mae']:>10.6f} "
            f"{row['mean_revision']:>10.6f} "
            f"{row['std_revision']:>10.6f}"
        )

    # Ratios
    rmse_vals = dict(
        zip(miq_overall["month_in_quarter"].to_list(), miq_overall["rmse"].to_list())
    )
    if rmse_vals.get(2, 0) > 0:
        print(f"\n  M1/M2 RMSE ratio: {rmse_vals.get(1, 0) / rmse_vals[2]:.2f}x")
        print(f"  M3/M2 RMSE ratio: {rmse_vals.get(3, 0) / rmse_vals[2]:.2f}x")

    # By quarter × miq
    print("\n--- RMSE by Quarter × Month-in-Quarter ---")
    qmiq = (
        diffs.filter(pl.col("era") >= 0)
        .group_by("quarter", "month_in_quarter")
        .agg(
            (pl.col("revision_error").pow(2).mean().sqrt()).alias("rmse"),
            pl.col("revision_error").count().alias("n"),
        )
        .sort("quarter", "month_in_quarter")
    )
    print(f"{'Q':>3} {'MIQ':>5} {'N':>6} {'RMSE':>10}")
    for row in qmiq.iter_rows(named=True):
        print(
            f"Q{row['quarter']:>2} "
            f"{MIQ_LABELS[row['month_in_quarter']]:>5} "
            f"{row['n']:>6} "
            f"{row['rmse']:>10.6f}"
        )

    # Era comparison
    print("\n--- Era Comparison (pooled across revisions) ---")
    era_miq = (
        diffs.filter(pl.col("era") >= 0)
        .group_by("era", "month_in_quarter")
        .agg(
            (pl.col("revision_error").pow(2).mean().sqrt()).alias("rmse"),
            pl.col("revision_error").count().alias("n"),
        )
        .sort("era", "month_in_quarter")
    )
    print(f"{'Era':>20} {'MIQ':>5} {'N':>6} {'RMSE':>10}")
    for row in era_miq.iter_rows(named=True):
        era_str = "Pre-COVID (2017-19)" if row["era"] == 0 else "Post-COVID (2022+)"
        print(
            f"{era_str:>20} "
            f"{MIQ_LABELS[row['month_in_quarter']]:>5} "
            f"{row['n']:>6} "
            f"{row['rmse']:>10.6f}"
        )

    # Post/Pre ratios
    pre = dict(
        zip(
            era_miq.filter(pl.col("era") == 0)["month_in_quarter"].to_list(),
            era_miq.filter(pl.col("era") == 0)["rmse"].to_list(),
        )
    )
    post = dict(
        zip(
            era_miq.filter(pl.col("era") == 1)["month_in_quarter"].to_list(),
            era_miq.filter(pl.col("era") == 1)["rmse"].to_list(),
        )
    )
    if pre and post:
        print("\n  Post-COVID / Pre-COVID RMSE ratios:")
        for miq in sorted(set(pre) & set(post)):
            ratio = post[miq] / pre[miq] if pre[miq] > 0 else float("inf")
            print(f"    {MIQ_LABELS[miq]}: {ratio:.2f}x")

    # Kurtosis by tier
    print("\n--- Excess Kurtosis by Month-in-Quarter ---")
    era_filtered = diffs.filter(pl.col("era") >= 0)
    for miq in [1, 2, 3]:
        errors = (
            era_filtered.filter(pl.col("month_in_quarter") == miq)["revision_error"]
            .to_numpy()
        )
        if len(errors) >= 5:
            kurt = float(stats.kurtosis(errors, fisher=True))
            print(f"  {MIQ_LABELS[miq]}: {kurt:.2f}  (Normal=0, t(5)=6)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading QCEW national data from vintage store...")
    raw = _load_qcew_national()
    print(f"  {len(raw):,} raw rows")

    print("Computing growth rates...")
    with_growth = _compute_growth(raw)
    print(f"  {len(with_growth):,} rows with growth")

    print("Building revision diffs...")
    diffs = _build_revision_diffs(with_growth)
    print(f"  {len(diffs):,} revision-error observations")

    # Summary table
    summary = _compute_summary(diffs)
    summary.write_parquet(OUT_DIR / "summary.parquet")
    summary.write_csv(OUT_DIR / "summary.csv")

    overall = _compute_overall_summary(diffs)
    overall.write_parquet(OUT_DIR / "overall_summary.parquet")
    overall.write_csv(OUT_DIR / "overall_summary.csv")

    _print_report(diffs)

    # Plots
    print("\nGenerating plots...")
    _plot_rmse_by_miq(diffs, OUT_DIR)
    _plot_decay_curves(diffs, OUT_DIR)
    _plot_bias(diffs, OUT_DIR)
    _plot_qq(diffs, OUT_DIR)
    _plot_era_comparison(diffs, OUT_DIR)
    _plot_autocorrelation(diffs, OUT_DIR)
    _plot_heatmap(diffs, OUT_DIR)
    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
