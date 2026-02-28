"""Nowcast backtest via CES-censoring experiment.

For each of the last *n* months the backtest:

1. Censors CES observations from that month onward (simulating the
   real-time information set available before the BLS release).
2. Fits the state-space model with lighter sampling
   (:data:`~alt_nfp.sampling.LIGHT_SAMPLER_KWARGS`).
3. Compares the latent-state nowcast to the actual CES release.

This quantifies the operational forecasting value added by payroll-provider
data.  Results are reported both as growth-rate errors (percentage points)
and as jobs-added errors (thousands).
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_DIR, PROVIDERS
from .ingest import build_panel
from .model import build_model
from .panel_adapter import panel_to_model_data
from .sampling import LIGHT_SAMPLER_KWARGS, sample_model


def run_backtest(
    n_backtest: int = 24,
    *,
    use_era_specific: bool = True,
) -> list[dict]:
    """Run the nowcast backtest over the last *n_backtest* months.

    Parameters
    ----------
    n_backtest : int
        Number of trailing months to backtest (default 24).
    use_era_specific : bool, optional
        If True (default), use era-specific latent parameters when
        ``era_idx`` is in the data dict.  If False, pass data without
        ``era_idx`` for Phase 1 baseline comparison.

    Returns
    -------
    list[dict]
        Per-month result records with actual vs nowcast metrics.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_panel()
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

        data = panel_to_model_data(panel, PROVIDERS, censor_ces_from=target_date)
        if not use_era_specific and "era_idx" in data:
            data = {k: v for k, v in data.items() if k != "era_idx"}
        model = build_model(data)
        idata = sample_model(model, sampler_kwargs=LIGHT_SAMPLER_KWARGS)

        g_sa_post = idata.posterior["g_total_sa"].values
        g_sa_mean = g_sa_post.mean(axis=(0, 1))
        cum_sa = np.cumsum(g_sa_mean)
        nowcast_growth = g_sa_mean[t_idx]
        nowcast_index = base_index * np.exp(cum_sa[t_idx])

        actual_growth = g_ces_sa_actual[t_idx]
        actual_index = ces_sa_index[t_idx + 1] if t_idx + 1 < len(ces_sa_index) else np.nan

        actual_change_k = (actual_index - ces_sa_index[t_idx]) / 1000.0
        if t_idx > 0:
            prev_index = base_index * np.exp(cum_sa[t_idx - 1])
        else:
            prev_index = ces_sa_index[0]
        nowcast_change_k = (nowcast_index - prev_index) / 1000.0
        error_change_k = actual_change_k - nowcast_change_k

        err_growth_pp = (nowcast_growth - actual_growth) * 100
        err_level_k = (actual_index - nowcast_index) / 1000.0

        sources: list[str] = []
        if t_idx in data["qcew_obs"]:
            sources.append("QCEW")
        for pp in data["pp_data"]:
            if t_idx in pp["pp_obs"]:
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
    print("\n" + "=" * 95)
    print(
        f"NOWCAST BACKTEST: Last {n_backtest} months "
        f"(jobs added = month-over-month change, SA)"
    )
    print("=" * 95)
    print(
        f"{'Date':>12}  {'Actual \u0394(k)':>11} {'Nowcast \u0394(k)':>11} "
        f"{'Error \u0394(k)':>10}  {'Actual %':>8} {'Nowcast %':>8} "
        f"{'Error (pp)':>9}  {'Sources':>12}"
    )
    print("-" * 95)
    for r in results:
        print(
            f'{str(r["date"]):>12}  {r["actual_change_k"]:>+10,.0f}  '
            f'{r["nowcast_change_k"]:>+10,.0f}  '
            f'{r["error_change_k"]:>+9,.0f}  '
            f'{r["actual_growth_pct"]:>+7.3f}  '
            f'{r["nowcast_growth_pct"]:>+7.3f}  '
            f'{r["error_growth_pp"]:>+8.3f}  '
            f'{r["sources"]:>12}'
        )

    data_rich = [
        r for r in results if "QCEW" in r["sources"] or "PP1" in r["sources"]
    ]
    pp2_only = [r for r in results if r["sources"] == "PP2"]
    no_data = [r for r in results if r["sources"] == "none"]

    mae_gr, rmse_gr = _mae_rmse(results, "error_growth_pp")
    mae_chg, rmse_chg = _mae_rmse(results, "error_change_k")

    print("-" * 95)
    print(
        f"Overall (n={len(results)}):  MAE jobs added = {mae_chg:,.0f} k   "
        f"RMSE = {rmse_chg:,.0f} k  |  MAE growth = {mae_gr:.3f} pp"
    )
    if data_rich:
        mae_dr_chg, _ = _mae_rmse(data_rich, "error_change_k")
        mae_dr_gr, _ = _mae_rmse(data_rich, "error_growth_pp")
        dr_start = min(r["date"] for r in data_rich)
        dr_end = max(r["date"] for r in data_rich)
        print(
            f"Data-rich ({dr_start}\u2013{dr_end}, n={len(data_rich)}):  "
            f"MAE jobs added = {mae_dr_chg:,.0f} k   MAE growth = {mae_dr_gr:.3f} pp"
        )
    if pp2_only:
        mae_po_chg, _ = _mae_rmse(pp2_only, "error_change_k")
        mae_po_gr, _ = _mae_rmse(pp2_only, "error_growth_pp")
        po_start = min(r["date"] for r in pp2_only)
        po_end = max(r["date"] for r in pp2_only)
        print(
            f"PP2-only ({po_start}\u2013{po_end}, n={len(pp2_only)}):  "
            f"MAE jobs added = {mae_po_chg:,.0f} k   MAE growth = {mae_po_gr:.3f} pp"
        )
    if no_data:
        mae_nd_chg, _ = _mae_rmse(no_data, "error_change_k")
        mae_nd_gr, _ = _mae_rmse(no_data, "error_growth_pp")
        nd_start = min(r["date"] for r in no_data)
        nd_end = max(r["date"] for r in no_data)
        print(
            f"No alt data ({nd_start}\u2013{nd_end}, n={len(no_data)}):  "
            f"MAE jobs added = {mae_nd_chg:,.0f} k   MAE growth = {mae_nd_gr:.3f} pp"
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
    ax.set_title("SA Total Employment Growth: Actual vs Nowcast")
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
