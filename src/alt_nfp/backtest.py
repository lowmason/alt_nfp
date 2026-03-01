"""Real-time nowcast backtest with vintage-aware censoring.

For each of the last *n* months the backtest:

1. Sets ``as_of = target_date`` so that only data published by that date
   is available — CES, QCEW, and cyclical indicators are all censored
   according to their actual publication lags and vintage dates.
2. Fits the state-space model with lighter sampling
   (:data:`~alt_nfp.sampling.LIGHT_SAMPLER_KWARGS`).
3. Compares the latent-state nowcast to the final CES release.

This produces a realistic evaluation of the model's real-time nowcasting
accuracy.  Results are reported both as growth-rate errors (percentage
points) and as jobs-added errors (thousands).
"""

from __future__ import annotations

from datetime import date

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .config import OUTPUT_DIR, PROVIDERS
from .ingest import build_panel
from .model import build_model
from .panel_adapter import panel_to_model_data
from .sampling import LIGHT_SAMPLER_KWARGS, sample_model


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

    # Try to extract per-period revision numbers from the filtered panel
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


def run_backtest(
    n_backtest: int = 24,
    *,
    use_era_specific: bool = True,
) -> list[dict]:
    """Run the real-time nowcast backtest over the last *n_backtest* months.

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
        Number of trailing months to backtest (default 24).
    use_era_specific : bool, optional
        If True (default), use era-specific latent parameters when
        ``era_idx`` is in the data dict.

    Returns
    -------
    list[dict]
        Per-month result records with actual vs nowcast metrics.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_panel()

    # Uncensored "truth" panel for actual values
    data_full = panel_to_model_data(panel, PROVIDERS)

    dates = data_full["dates"]
    T = len(dates)
    g_ces_sa_actual = data_full["g_ces_sa"]
    levels = data_full["levels"]
    ces_sa_index = levels["ces_sa_index"].to_numpy().astype(float)
    base_index = float(ces_sa_index[0])

    if T < n_backtest:
        raise ValueError(f"Need at least {n_backtest} months, got T={T}")

    target_indices = list(range(T - n_backtest, T))
    target_dates = [dates[i] for i in target_indices]

    results: list[dict] = []

    for run, (t_idx, target_date) in enumerate(zip(target_indices, target_dates)):
        print(f"\n--- Nowcast backtest {run + 1}/{n_backtest}: {target_date} ---")

        data = panel_to_model_data(panel, PROVIDERS, as_of=target_date)
        frontier = _vintage_frontier(data)
        _print_frontier(frontier, target_date)

        if not use_era_specific and "era_idx" in data:
            data = {k: v for k, v in data.items() if k != "era_idx"}
        model = build_model(data)
        idata = sample_model(model, sampler_kwargs=LIGHT_SAMPLER_KWARGS)

        g_sa_post = idata.posterior["g_total_sa"].values
        g_sa_mean = np.nanmean(g_sa_post, axis=(0, 1))

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
        actual_index = (
            ces_sa_index[t_idx + 1] if t_idx + 1 < len(ces_sa_index) else np.nan
        )

        actual_change_k = (actual_index - ces_sa_index[t_idx]) / 1000.0
        nowcast_change_k = (nowcast_index - prev_index) / 1000.0
        error_change_k = actual_change_k - nowcast_change_k

        err_growth_pp = (nowcast_growth - actual_growth) * 100
        err_level_k = (actual_index - nowcast_index) / 1000.0

        # Track which sources have data for the target month (use c_idx)
        sources: list[str] = []
        if c_idx in data["ces_sa_obs"]:
            sources.append("CES")
        if c_idx in data["qcew_obs"]:
            sources.append("QCEW")
        for pp in data["pp_data"]:
            if c_idx in pp["pp_obs"]:
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
                "actual_level_k": actual_index / 1000.0,
                "nowcast_level_k": nowcast_index / 1000.0,
                "error_level_k": err_level_k,
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

    _print_results_table(results, n_backtest)
    _plot_backtest(results)

    return results


# =========================================================================
# Reporting
# =========================================================================


def _print_results_table(results: list[dict], n_backtest: int) -> None:
    """Print console summary table."""
    print("\n" + "=" * 120)
    print(
        f"NOWCAST BACKTEST (real-time vintage): Last {n_backtest} months "
        f"(jobs added = month-over-month change, SA)"
    )
    print("=" * 120)
    print(
        f"{'Date':>12}  {'Actual \u0394(k)':>11} {'Nowcast \u0394(k)':>11} "
        f"{'Error \u0394(k)':>10}  {'Actual %':>8} {'Nowcast %':>8} "
        f"{'Error (pp)':>9}  {'CES thru':>12} {'QCEW thru':>12}"
    )
    print("-" * 120)
    for r in results:
        ces_str = str(r["ces_latest"]) if r["ces_latest"] else "—"
        qcew_str = str(r["qcew_latest"]) if r["qcew_latest"] else "—"
        print(
            f'{str(r["date"]):>12}  {r["actual_change_k"]:>+10,.0f}  '
            f'{r["nowcast_change_k"]:>+10,.0f}  '
            f'{r["error_change_k"]:>+9,.0f}  '
            f'{r["actual_growth_pct"]:>+7.3f}  '
            f'{r["nowcast_growth_pct"]:>+7.3f}  '
            f'{r["error_growth_pp"]:>+8.3f}  '
            f'{ces_str:>12} {qcew_str:>12}'
        )

    mae_gr, rmse_gr = _mae_rmse(results, "error_growth_pp")
    mae_chg, rmse_chg = _mae_rmse(results, "error_change_k")

    print("-" * 120)
    print(
        f"Overall (n={len(results)}):  MAE jobs added = {mae_chg:,.0f} k   "
        f"RMSE = {rmse_chg:,.0f} k  |  MAE growth = {mae_gr:.3f} pp   "
        f"RMSE growth = {rmse_gr:.3f} pp"
    )


def _plot_backtest(results: list[dict]) -> None:
    """Two-panel figure: growth actual vs nowcast, and jobs-added error."""
    x_dates = [r["date"] for r in results]
    actual_gr = [r["actual_growth_pct"] for r in results]
    nowcast_gr = [r["nowcast_growth_pct"] for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    ax.plot(x_dates, actual_gr, "o-", color="darkorange", label="Actual CES SA growth", ms=8)
    ax.plot(x_dates, nowcast_gr, "s--", color="steelblue", label="Nowcast (model)", ms=6)
    for i, (a, n) in enumerate(zip(actual_gr, nowcast_gr)):
        ax.vlines(x_dates[i], a, n, color="gray", alpha=0.6, lw=1)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Monthly growth (%)")
    ax.set_title("SA Total Employment Growth: Actual vs Real-Time Nowcast")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(
        x_dates,
        [r["error_change_k"] for r in results],
        color="steelblue",
        alpha=0.7,
        width=18,
        label="Error in jobs added",
    )
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Error (thousands)")
    ax.set_title(
        "Nowcast Error in Jobs Added "
        "(Actual \u2212 Nowcast; positive = we under-nowcast gain)"
    )
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "nowcast_backtest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_DIR / 'nowcast_backtest.png'}")


def _mae_rmse(lst: list[dict], key: str) -> tuple[float, float]:
    if not lst:
        return float("nan"), float("nan")
    errs = np.array([x[key] for x in lst])
    return float(np.mean(np.abs(errs))), float(np.sqrt(np.mean(errs**2)))
