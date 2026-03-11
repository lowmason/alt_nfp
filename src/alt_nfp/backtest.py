"""Real-time nowcast backtest with vintage-aware censoring.

For each target month the backtest:

1. Sets ``as_of = target_date`` so that only data published by that date
   is available — CES, QCEW, and cyclical indicators are all censored
   according to their actual publication lags and vintage dates.
2. Fits the state-space model with lighter sampling
   (:data:`~alt_nfp.sampling.LIGHT_SAMPLER_KWARGS`).
3. Compares the latent-state nowcast to the final CES release.

Results are persisted per-run to a timestamped directory under
``output/backtest_runs/``.  Each iteration saves its InferenceData
(``.nc``), and the full run saves a summary parquet and plot.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .config import BASE_DIR, providers_from_settings
from .ingest import build_panel
from .model import build_model
from .panel_adapter import panel_to_model_data
from .sampling import sample_model
from .settings import NowcastConfig


# =========================================================================
# Vintage frontier diagnostic
# =========================================================================


def _vintage_frontier(data: dict) -> dict:
    """Summarise the information frontier for a censored data dict.

    Returns a dict with:
    - ``ces_latest``: last period with a finite CES SA observation (or None)
    - ``ces_n``: number of CES SA observations
    - ``qcew_latest``: last period with a finite QCEW observation (or None)
    - ``qcew_n``: number of QCEW observations
    - ``ces_revisions``: dict mapping the last few periods to their revision
      number (if the filtered panel is available)
    """
    dates = data["dates"]
    ces_sa_obs = data["ces_sa_obs"]
    qcew_obs = data["qcew_obs"]

    ces_latest = dates[ces_sa_obs[-1]] if len(ces_sa_obs) > 0 else None
    qcew_latest = dates[qcew_obs[-1]] if len(qcew_obs) > 0 else None

    ces_revisions: dict[date, int] = {}
    panel = data.get("panel")
    if panel is not None and "revision_number" in panel.columns:
        ces_rows = panel.filter(pl.col("source") == "ces_sa")
        if len(ces_rows) > 0:
            latest_per_period = (
                ces_rows.sort("revision_number", descending=True)
                .unique(subset=["period"], keep="first")
                .sort("period", descending=True)
                .head(5)
            )
            for row in latest_per_period.iter_rows(named=True):
                rev = row.get("revision_number")
                if rev is not None:
                    ces_revisions[row["period"]] = int(rev)

    return {
        "ces_latest": ces_latest,
        "ces_n": len(ces_sa_obs),
        "qcew_latest": qcew_latest,
        "qcew_n": len(qcew_obs),
        "ces_revisions": ces_revisions,
    }


def _print_frontier(frontier: dict, target_date: date) -> None:
    """Print a compact vintage frontier summary."""
    ces_str = str(frontier["ces_latest"]) if frontier["ces_latest"] else "none"
    qcew_str = str(frontier["qcew_latest"]) if frontier["qcew_latest"] else "none"
    parts = [
        f"  Vintage frontier (as_of={target_date}):",
        f"CES SA through {ces_str} ({frontier['ces_n']} obs)",
        f"QCEW through {qcew_str} ({frontier['qcew_n']} obs)",
    ]
    print("  ".join(parts))

    revs = frontier.get("ces_revisions", {})
    if revs:
        rev_strs = [f"{p}: rev {r}" for p, r in sorted(revs.items(), reverse=True)]
        print(f"  CES revisions (latest 5): {', '.join(rev_strs)}")

    ces_latest = frontier["ces_latest"]
    if ces_latest is not None:
        gap_months = (target_date.year - ces_latest.year) * 12 + (
            target_date.month - ces_latest.month
        )
        if gap_months > 2:
            print(
                f"  ** WARNING: CES frontier is {gap_months} months behind target. "
                f"Run the vintage pipeline (alt-nfp download && alt-nfp process "
                f"&& alt-nfp build) to capture intermediate vintages."
            )


# =========================================================================
# Main backtest
# =========================================================================


def _resolve_run_dir(output_dir: Path | None, default_output_dir: Path) -> Path:
    """Return the run directory, creating it if needed."""
    if output_dir is not None:
        run_dir = output_dir
    else:
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M")
        run_dir = default_output_dir / "backtest_runs" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_backtest(
    n_backtest: int = 24,
    *,
    start_date: date | None = None,
    output_dir: Path | None = None,
    use_era_specific: bool = True,
    cfg: NowcastConfig | None = None,
) -> pl.DataFrame:
    """Run the real-time nowcast backtest.

    For each target month T, the model sees only data published by date T
    (the 12th of the target month).  This means:

    - CES for T is not yet available (first print arrives ~first Friday of T+1)
    - CES for T-1 is revision 0, T-2 is revision 1, T-3 is revision 2
    - QCEW is missing for the most recent ~5-6 months
    - Cyclical indicators are censored per their publication lags

    The "actual" values come from the uncensored panel (best available
    revision, typically final/benchmark).

    Parameters
    ----------
    n_backtest : int
        Number of months to backtest (default 24).
    start_date : date, optional
        First target month.  When provided the backtest runs forward from
        ``start_date`` for ``n_backtest`` months (or to the panel end,
        whichever comes first).  When *None* (default) the backtest covers
        the last ``n_backtest`` months of the panel.
    output_dir : Path, optional
        Directory for all run artifacts (InferenceData ``.nc`` files,
        results parquet, plot).  Defaults to a timestamped subdirectory
        under ``output/backtest_runs/``.
    use_era_specific : bool, optional
        If True (default), use era-specific latent parameters when
        ``era_idx`` is in the data dict.

    Returns
    -------
    pl.DataFrame
        Per-month result records with actual vs nowcast metrics.
    """
    if cfg is None:
        cfg = NowcastConfig()
    resolved = cfg.resolve_paths(BASE_DIR)
    providers = providers_from_settings(cfg)
    sampler_kwargs = cfg.sampling.get_preset(cfg.backtest.sampling_preset).to_pymc_kwargs()

    run_dir = _resolve_run_dir(output_dir, resolved.output_dir)

    panel_full = build_panel()

    # Uncensored "truth" panel for actual values
    data_full = panel_to_model_data(panel_full, providers, cfg=cfg)

    dates = data_full["dates"]
    T = len(dates)
    g_ces_sa_actual = data_full["g_ces_sa"]
    levels = data_full["levels"]
    ces_sa_index = levels["ces_sa_index"].to_numpy().astype(float)
    base_index = float(ces_sa_index[0])

    base_row_idx = int(np.argmin(np.abs(ces_sa_index - 100.0)))
    ces_sa_base_level = levels["ces_sa_level"].to_numpy().astype(float)[base_row_idx]
    idx_to_level = ces_sa_base_level / 100.0

    # Resolve target indices from start_date / n_backtest
    if start_date is not None:
        first_idx = next(
            (i for i, d in enumerate(dates) if d >= start_date),
            T,
        )
        last_idx = min(first_idx + n_backtest, T)
        if first_idx >= T:
            raise ValueError(
                f"start_date {start_date} is beyond the panel end ({dates[-1]})"
            )
        target_indices = list(range(first_idx, last_idx))
    else:
        if T < n_backtest:
            raise ValueError(f"Need at least {n_backtest} months, got T={T}")
        target_indices = list(range(T - n_backtest, T))

    target_dates = [dates[i] for i in target_indices]
    n_runs = len(target_indices)

    results: list[dict] = []

    for run, (t_idx, target_date) in enumerate(zip(target_indices, target_dates)):
        print(f"\n--- Nowcast backtest {run + 1}/{n_runs}: {target_date} ---")

        panel = build_panel(as_of_ref=target_date)
        data = panel_to_model_data(panel, providers, as_of=target_date, cfg=cfg)
        frontier = _vintage_frontier(data)
        _print_frontier(frontier, target_date)

        if not use_era_specific and "era_idx" in data:
            data = {k: v for k, v in data.items() if k != "era_idx"}
        model = build_model(data, cfg=cfg)
        idata = sample_model(model, sampler_kwargs=sampler_kwargs)

        # Persist InferenceData for later analysis
        nc_path = run_dir / f"{target_date:%Y-%m}.nc"
        idata.to_netcdf(str(nc_path))
        print(f"  Saved: {nc_path}")

        g_sa_post = idata.posterior["g_total_sa"].values
        alpha_post = idata.posterior["alpha_ces"].values
        lambda_post = idata.posterior["lambda_ces"].values

        # Posterior predictive for CES SA: transform latent growth through
        # the CES observation equation so the nowcast is in the same space
        # as the actual CES print we compare against.
        g_ces_pred = alpha_post[:, :, None] + lambda_post[:, :, None] * g_sa_post
        g_sa_mean = np.nanmean(g_ces_pred, axis=(0, 1))

        # Map target_date to the censored model's index (the censored
        # model may have fewer time steps than the full model).
        censored_dates = data["dates"]
        if target_date in censored_dates:
            c_idx = censored_dates.index(target_date)
        else:
            c_idx = len(censored_dates) - 1
            print(
                f"  NOTE: Target {target_date} beyond censored horizon "
                f"({censored_dates[-1]}); using last latent state as proxy"
            )

        nowcast_index_series = np.empty(len(g_sa_mean) + 1, dtype=float)
        nowcast_index_series[0] = base_index
        for s in range(len(g_sa_mean)):
            nowcast_index_series[s + 1] = nowcast_index_series[s] * np.exp(g_sa_mean[s])
        nowcast_growth = g_sa_mean[c_idx]
        nowcast_index = nowcast_index_series[c_idx + 1]
        prev_index = nowcast_index_series[c_idx]

        actual_growth = g_ces_sa_actual[t_idx]
        actual_index = ces_sa_index[t_idx]
        prev_actual_index = ces_sa_index[t_idx - 1] if t_idx > 0 else np.nan

        actual_level = actual_index * idx_to_level
        prev_level = prev_actual_index * idx_to_level
        nowcast_level = nowcast_index * idx_to_level
        prev_nowcast_level = prev_index * idx_to_level

        actual_change_k = actual_level - prev_level
        nowcast_change_k = nowcast_level - prev_nowcast_level
        error_change_k = actual_change_k - nowcast_change_k

        err_growth_pp = (nowcast_growth - actual_growth) * 100
        err_level_k = actual_level - nowcast_level

        # Per-source availability flags for the target month
        has_ces = c_idx in data["ces_sa_obs"]
        has_qcew = c_idx in data["qcew_obs"]
        provider_flags: dict[str, bool] = {}
        sources: list[str] = []
        if has_ces:
            sources.append("CES")
        if has_qcew:
            sources.append("QCEW")
        for pp in data["pp_data"]:
            present = c_idx in pp["pp_obs"]
            provider_flags[pp["name"]] = present
            if present:
                sources.append(pp["name"])
        sources_str = "+".join(sources) if sources else "none"

        results.append(
            {
                "date": target_date,
                "actual_growth_pct": actual_growth * 100,
                "nowcast_growth_pct": nowcast_growth * 100,
                "error_growth_pp": err_growth_pp,
                "actual_change_k": actual_change_k,
                "nowcast_change_k": nowcast_change_k,
                "error_change_k": error_change_k,
                "actual_level_k": actual_level,
                "nowcast_level_k": nowcast_level,
                "error_level_k": err_level_k,
                "has_ces": has_ces,
                "has_qcew": has_qcew,
                **{f"has_{k}": v for k, v in provider_flags.items()},
                "sources": sources_str,
                "ces_latest": frontier["ces_latest"],
                "qcew_latest": frontier["qcew_latest"],
            }
        )
        print(
            f"  Jobs added (SA): actual {actual_change_k:+,.0f}k  "
            f"nowcast {nowcast_change_k:+,.0f}k  "
            f"error {error_change_k:+,.0f}k  [{sources_str}]"
        )

    results_df = pl.DataFrame(results)
    parquet_path = run_dir / "backtest_results.parquet"
    results_df.write_parquet(parquet_path)
    print(f"\nSaved: {parquet_path}")

    _print_results_table(results_df, n_runs)
    _plot_backtest(results_df, run_dir)

    return results_df


# =========================================================================
# Reporting
# =========================================================================


def _print_results_table(results: pl.DataFrame, n_backtest: int) -> None:
    """Print console summary table."""
    print("\n" + "=" * 120)
    print(
        f"NOWCAST BACKTEST (real-time vintage): {n_backtest} months "
        f"(jobs added = month-over-month change, SA)"
    )
    print("=" * 120)
    print(
        f"{'Date':>12}  {'Actual \u0394(k)':>11} {'Nowcast \u0394(k)':>11} "
        f"{'Error \u0394(k)':>10}  {'Actual %':>8} {'Nowcast %':>8} "
        f"{'Error (pp)':>9}  {'CES thru':>12} {'QCEW thru':>12}"
    )
    print("-" * 120)
    for row in results.iter_rows(named=True):
        ces_str = str(row["ces_latest"]) if row["ces_latest"] else "\u2014"
        qcew_str = str(row["qcew_latest"]) if row["qcew_latest"] else "\u2014"
        print(
            f'{str(row["date"]):>12}  {row["actual_change_k"]:>+10,.0f}  '
            f'{row["nowcast_change_k"]:>+10,.0f}  '
            f'{row["error_change_k"]:>+9,.0f}  '
            f'{row["actual_growth_pct"]:>+7.3f}  '
            f'{row["nowcast_growth_pct"]:>+7.3f}  '
            f'{row["error_growth_pp"]:>+8.3f}  '
            f'{ces_str:>12} {qcew_str:>12}'
        )

    errs_gr = results["error_growth_pp"].to_numpy()
    errs_chg = results["error_change_k"].to_numpy()

    mae_gr = float(np.mean(np.abs(errs_gr)))
    rmse_gr = float(np.sqrt(np.mean(errs_gr**2)))
    mae_chg = float(np.mean(np.abs(errs_chg)))
    rmse_chg = float(np.sqrt(np.mean(errs_chg**2)))

    print("-" * 120)
    print(
        f"Overall (n={len(results)}):  MAE jobs added = {mae_chg:,.0f} k   "
        f"RMSE = {rmse_chg:,.0f} k  |  MAE growth = {mae_gr:.3f} pp   "
        f"RMSE growth = {rmse_gr:.3f} pp"
    )


def _plot_backtest(results: pl.DataFrame, run_dir: Path) -> None:
    """Two-panel figure: jobs-added actual vs nowcast, and source availability."""
    x_dates = results["date"].to_list()
    actual_chg = results["actual_change_k"].to_numpy()
    nowcast_chg = results["nowcast_change_k"].to_numpy()
    n = len(x_dates)

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # --- Top panel: side-by-side jobs-added bars ---
    ax = axes[0]
    bar_width = 10
    x_num = mdates.date2num(x_dates)
    ax.bar(
        x_num - bar_width / 2, actual_chg, width=bar_width,
        color="darkorange", alpha=0.85, label="Actual",
    )
    ax.bar(
        x_num + bar_width / 2, nowcast_chg, width=bar_width,
        color="steelblue", alpha=0.85, label="Nowcast",
    )
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Jobs added (thousands, SA)")
    ax.set_title("Month-over-Month Jobs Added: Actual vs Real-Time Nowcast")
    ax.legend(loc="upper right")
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", alpha=0.3)

    mae_chg = float(np.mean(np.abs(actual_chg - nowcast_chg)))
    rmse_chg = float(np.sqrt(np.mean((actual_chg - nowcast_chg) ** 2)))
    ax.text(
        0.01, 0.02,
        f"MAE = {mae_chg:,.0f}k    RMSE = {rmse_chg:,.0f}k",
        transform=ax.transAxes, fontsize=9, color="gray",
        verticalalignment="bottom",
    )

    # --- Bottom panel: data-source availability strip ---
    ax = axes[1]

    source_cols = ["has_ces", "has_qcew"]
    source_labels = ["CES", "QCEW"]
    provider_cols = [c for c in results.columns if c.startswith("has_") and c not in source_cols]
    source_cols += provider_cols
    source_labels += [c.removeprefix("has_").upper() for c in provider_cols]

    n_sources = len(source_cols)
    for s_idx, (col, label) in enumerate(zip(source_cols, source_labels)):
        flags = results[col].to_numpy()
        for t_idx in range(n):
            color = "#2ca02c" if flags[t_idx] else "#d62728"
            marker = "o" if flags[t_idx] else "x"
            ax.plot(
                x_dates[t_idx], s_idx, marker=marker, color=color,
                ms=6, mew=1.5,
            )

    ax.set_yticks(range(n_sources))
    ax.set_yticklabels(source_labels)
    ax.set_ylim(-0.5, n_sources - 0.5)
    ax.invert_yaxis()
    ax.set_title("Data Source Availability at Target Month")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plot_path = run_dir / "nowcast_backtest.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")
