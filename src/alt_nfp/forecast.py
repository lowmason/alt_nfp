# ---------------------------------------------------------------------------
# alt_nfp.forecast — Forward simulation and forecast plots
# ---------------------------------------------------------------------------
from __future__ import annotations

from datetime import date

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .config import BD_QCEW_LAG, OUTPUT_DIR


def forecast_and_plot(idata: az.InferenceData, data: dict) -> None:
    """Forecast SA and NSA indices to 2026-01-12 and plot."""
    dates = data["dates"]
    levels = data["levels"]
    pp_data = data["pp_data"]

    # --- Posterior parameter samples ---
    mu_g_post = idata.posterior["mu_g"].values
    phi_post = idata.posterior["phi"].values
    sigma_g_post = idata.posterior["sigma_g"].values
    seasonal_post = idata.posterior["seasonal"].values
    g_cont_post = idata.posterior["g_cont"].values
    g_sa_post = idata.posterior["g_total_sa"].values
    g_nsa_post = idata.posterior["g_total_nsa"].values

    # Structural BD parameters
    phi_0_post = idata.posterior["phi_0"].values
    phi_1_post = idata.posterior["phi_1"].values
    phi_2_post = idata.posterior["phi_2"].values
    sigma_bd_post = idata.posterior["sigma_bd"].values

    n_chains, n_draws, T_hist = g_cont_post.shape

    # --- Build forecast date grid ---
    last_date = dates[-1]
    forecast_end = date(2026, 1, 12)

    forecast_dates: list[date] = []
    d = last_date
    while True:
        yr, mo = d.year, d.month
        d = date(yr + 1, 1, 12) if mo == 12 else date(yr, mo + 1, 12)
        forecast_dates.append(d)
        if d >= forecast_end:
            break

    n_fwd = len(forecast_dates)
    forecast_month_idx = [d.month - 1 for d in forecast_dates]

    # --- BD forward-propagation covariates ---
    birth_rate = data["birth_rate"]
    birth_rate_mean = data["birth_rate_mean"]
    bd_proxy = data["bd_proxy"]
    bd_qcew_mean = data["bd_qcew_mean"]

    # Last observed birth rate (centred)
    finite_br = np.where(np.isfinite(birth_rate))[0]
    birth_rate_last_c = (birth_rate[finite_br[-1]] - birth_rate_mean) if len(finite_br) > 0 else 0.0

    # --- Simulate forward ---
    rng = np.random.default_rng(42)
    g_cont_fwd = np.zeros((n_chains, n_draws, n_fwd))
    g_sa_fwd = np.zeros((n_chains, n_draws, n_fwd))
    g_nsa_fwd = np.zeros((n_chains, n_draws, n_fwd))
    bd_fwd = np.zeros((n_chains, n_draws, n_fwd))

    for h in range(n_fwd):
        # AR(1) continuing-units growth
        eps = rng.standard_normal((n_chains, n_draws))
        g_prev = g_cont_post[:, :, -1] if h == 0 else g_cont_fwd[:, :, h - 1]
        g_cont_fwd[:, :, h] = mu_g_post + phi_post * (g_prev - mu_g_post) + sigma_g_post * eps

        # Structural BD
        # QCEW BD proxy: use t-L if still in historical window
        t_lag = T_hist + h - BD_QCEW_LAG
        if 0 <= t_lag < T_hist and np.isfinite(bd_proxy[t_lag]):
            fwd_bd_qcew_c = bd_proxy[t_lag] - bd_qcew_mean
        else:
            fwd_bd_qcew_c = 0.0

        eps_bd = rng.standard_normal((n_chains, n_draws))
        bd_fwd[:, :, h] = (
            phi_0_post
            + phi_1_post * birth_rate_last_c
            + phi_2_post * fwd_bd_qcew_c
            + sigma_bd_post * eps_bd
        )

        mi = forecast_month_idx[h]
        g_sa_fwd[:, :, h] = g_cont_fwd[:, :, h] + bd_fwd[:, :, h]
        g_nsa_fwd[:, :, h] = g_cont_fwd[:, :, h] + seasonal_post[:, :, mi] + bd_fwd[:, :, h]

    # --- Reconstruct index paths ---
    ces_sa_vals = levels["ces_sa_index"].to_numpy().astype(float)
    ces_nsa_vals = levels["ces_nsa_index"].to_numpy().astype(float)
    log_base_sa = np.log(ces_sa_vals[0])
    log_base_nsa = np.log(ces_nsa_vals[0])

    base_row_idx = np.argmin(np.abs(ces_sa_vals - 100.0))
    ces_sa_base_level = levels["ces_sa_level"].to_numpy().astype(float)[base_row_idx]
    ces_nsa_base_level = levels["ces_nsa_level"].to_numpy().astype(float)[base_row_idx]

    cum_sa_hist = log_base_sa + np.cumsum(g_sa_post, axis=2)
    cum_nsa_hist = log_base_nsa + np.cumsum(g_nsa_post, axis=2)
    cum_sa_fwd = cum_sa_hist[:, :, -1:] + np.cumsum(g_sa_fwd, axis=2)
    cum_nsa_fwd = cum_nsa_hist[:, :, -1:] + np.cumsum(g_nsa_fwd, axis=2)

    idx_sa_hist = np.exp(cum_sa_hist)
    idx_nsa_hist = np.exp(cum_nsa_hist)
    idx_sa_fwd = np.exp(cum_sa_fwd)
    idx_nsa_fwd = np.exp(cum_nsa_fwd)

    # --- Print forecast table ---
    _print_index_table(dates, forecast_dates, idx_sa_hist, idx_nsa_hist, idx_sa_fwd, idx_nsa_fwd)

    # --- Jobs added ---
    idx_to_sa = ces_sa_base_level / 100.0
    idx_to_nsa = ces_nsa_base_level / 100.0
    lvl_sa_hist = idx_sa_hist * idx_to_sa
    lvl_nsa_hist = idx_nsa_hist * idx_to_nsa
    lvl_sa_fwd = idx_sa_fwd * idx_to_sa
    lvl_nsa_fwd = idx_nsa_fwd * idx_to_nsa

    chg_sa_hist = (lvl_sa_hist[:, :, 1:] - lvl_sa_hist[:, :, :-1]) / 1000.0
    chg_nsa_hist = (lvl_nsa_hist[:, :, 1:] - lvl_nsa_hist[:, :, :-1]) / 1000.0
    chg_sa_fwd = np.zeros((n_chains, n_draws, n_fwd))
    chg_nsa_fwd = np.zeros((n_chains, n_draws, n_fwd))
    chg_sa_fwd[:, :, 0] = (lvl_sa_fwd[:, :, 0] - lvl_sa_hist[:, :, -1]) / 1000.0
    chg_nsa_fwd[:, :, 0] = (lvl_nsa_fwd[:, :, 0] - lvl_nsa_hist[:, :, -1]) / 1000.0
    for i in range(1, n_fwd):
        chg_sa_fwd[:, :, i] = (lvl_sa_fwd[:, :, i] - lvl_sa_fwd[:, :, i - 1]) / 1000.0
        chg_nsa_fwd[:, :, i] = (lvl_nsa_fwd[:, :, i] - lvl_nsa_fwd[:, :, i - 1]) / 1000.0

    _print_jobs_table(dates, forecast_dates, chg_sa_hist, chg_nsa_hist, chg_sa_fwd, chg_nsa_fwd)
    _print_growth_table(forecast_dates, g_sa_fwd, g_nsa_fwd)

    # --- Plot: index forecast ---
    _plot_index_forecast(
        dates, forecast_dates, levels, pp_data,
        idx_sa_hist, idx_nsa_hist, idx_sa_fwd, idx_nsa_fwd,
        ces_sa_vals, ces_nsa_vals,
    )

    # --- Plot: jobs-added forecast ---
    _plot_jobs_forecast(
        dates, forecast_dates, levels, T_hist,
        chg_sa_hist, chg_nsa_hist, chg_sa_fwd, chg_nsa_fwd,
    )


# =========================================================================
# Print helpers
# =========================================================================


def _print_index_table(dates, forecast_dates, sa_h, nsa_h, sa_f, nsa_f):
    print("=" * 72)
    print("FORECAST: SA & NSA Index to 2026-01-12")
    print("=" * 72)
    print(f"{'Date':>12}  {'SA Mean':>9} {'SA 80% HDI':>18}  "
          f"{'NSA Mean':>9} {'NSA 80% HDI':>18}")
    print("-" * 72)

    sa_last = sa_h[:, :, -1].flatten()
    nsa_last = nsa_h[:, :, -1].flatten()
    print(f"{str(dates[-1]):>12}  {sa_last.mean():9.2f} "
          f"[{np.percentile(sa_last, 10):7.2f}, {np.percentile(sa_last, 90):7.2f}]  "
          f"{nsa_last.mean():9.2f} "
          f"[{np.percentile(nsa_last, 10):7.2f}, {np.percentile(nsa_last, 90):7.2f}]  "
          f"\u2190 last obs")

    for i, fd in enumerate(forecast_dates):
        sa_v = sa_f[:, :, i].flatten()
        nsa_v = nsa_f[:, :, i].flatten()
        print(f"{str(fd):>12}  {sa_v.mean():9.2f} "
              f"[{np.percentile(sa_v, 10):7.2f}, {np.percentile(sa_v, 90):7.2f}]  "
              f"{nsa_v.mean():9.2f} "
              f"[{np.percentile(nsa_v, 10):7.2f}, {np.percentile(nsa_v, 90):7.2f}]  "
              f"\u2190 forecast")


def _print_jobs_table(dates, forecast_dates, c_sa_h, c_nsa_h, c_sa_f, c_nsa_f):
    print("\n" + "=" * 84)
    print("FORECAST: Jobs added (month-over-month change, thousands)")
    print("  Positive = jobs added, negative = jobs lost.  Base ref: CES at index 100.")
    print("=" * 84)
    print(f"{'Date':>12}  {'SA Mean':>10} {'SA 80% HDI':>22}  "
          f"{'NSA Mean':>10} {'NSA 80% HDI':>22}")
    print("-" * 84)

    last_sa = c_sa_h[:, :, -1].flatten()
    last_nsa = c_nsa_h[:, :, -1].flatten()
    print(f"{str(dates[-1]):>12}  {last_sa.mean():+10,.0f} "
          f"[{np.percentile(last_sa, 10):+10,.0f}, {np.percentile(last_sa, 90):+10,.0f}]  "
          f"{last_nsa.mean():+10,.0f} "
          f"[{np.percentile(last_nsa, 10):+10,.0f}, {np.percentile(last_nsa, 90):+10,.0f}]  "
          f"\u2190 last obs")

    for i, fd in enumerate(forecast_dates):
        sa = c_sa_f[:, :, i].flatten()
        nsa = c_nsa_f[:, :, i].flatten()
        print(f"{str(fd):>12}  {sa.mean():+10,.0f} "
              f"[{np.percentile(sa, 10):+10,.0f}, {np.percentile(sa, 90):+10,.0f}]  "
              f"{nsa.mean():+10,.0f} "
              f"[{np.percentile(nsa, 10):+10,.0f}, {np.percentile(nsa, 90):+10,.0f}]  "
              f"\u2190 forecast")


def _print_growth_table(forecast_dates, g_sa_fwd, g_nsa_fwd):
    print("\n" + "=" * 72)
    print("FORECAST: Monthly Growth Rates")
    print("=" * 72)
    for i, fd in enumerate(forecast_dates):
        gsa = g_sa_fwd[:, :, i].flatten() * 100
        gnsa = g_nsa_fwd[:, :, i].flatten() * 100
        print(f"{str(fd):>12}  "
              f"SA:  {gsa.mean():+.3f}%  "
              f"[{np.percentile(gsa, 10):+.3f}%, {np.percentile(gsa, 90):+.3f}%]   "
              f"NSA: {gnsa.mean():+.3f}%  "
              f"[{np.percentile(gnsa, 10):+.3f}%, {np.percentile(gnsa, 90):+.3f}%]")


# =========================================================================
# Plot helpers
# =========================================================================


def _plot_index_forecast(
    dates, forecast_dates, levels, pp_data,
    idx_sa_hist, idx_nsa_hist, idx_sa_fwd, idx_nsa_fwd,
    ces_sa_vals, ces_nsa_vals,
):
    base_dates = levels["ref_date"].to_list()
    connecting = [dates[-1]] + forecast_dates

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for panel_idx, (label, idx_h, idx_f, ces_vals, ls_nsa) in enumerate([
        ("SA", idx_sa_hist, idx_sa_fwd, ces_sa_vals, "-"),
        ("NSA", idx_nsa_hist, idx_nsa_fwd, ces_nsa_vals, "-"),
    ]):
        ax = axes[panel_idx]
        h_mean = idx_h.mean(axis=(0, 1))
        h_lo = np.percentile(idx_h, 10, axis=(0, 1))
        h_hi = np.percentile(idx_h, 90, axis=(0, 1))

        ax.fill_between(dates, h_lo, h_hi, alpha=0.2, color="steelblue")
        ax.plot(dates, h_mean, "steelblue", lw=2, label=f"Latent {label} (estimated)")

        f_mean = np.concatenate([[h_mean[-1]], idx_f.mean(axis=(0, 1))])
        f_lo = np.concatenate([[h_lo[-1]], np.percentile(idx_f, 10, axis=(0, 1))])
        f_hi = np.concatenate([[h_hi[-1]], np.percentile(idx_f, 90, axis=(0, 1))])

        ax.fill_between(connecting, f_lo, f_hi, alpha=0.3, color="coral")
        ax.plot(connecting, f_mean, "coral", lw=2, ls="--", label=f"Forecast {label}")

        ax.plot(base_dates[1:], ces_vals[1:], "darkorange", lw=1, alpha=0.7,
                label=f"CES {label} (observed)")

        bd_arr = np.array(base_dates)
        for pp in pp_data:
            try:
                vals = levels[pp["index_col"]].to_numpy().astype(float)
                mask = np.isfinite(vals)
                if mask.sum() > 1:
                    ax.plot(bd_arr[mask][1:], vals[mask][1:], color=pp["color"], lw=1,
                            alpha=0.5, label=f"{pp['name']} (NSA)")
            except Exception:
                pass

        if panel_idx == 1:
            qcew_rows = levels.filter(pl.col("qcew_nsa_index").is_not_null())
            ax.scatter(
                qcew_rows["ref_date"].to_list(),
                qcew_rows["qcew_nsa_index"].to_numpy(),
                s=20, c="red", marker="o", alpha=0.7, label="QCEW (NSA)", zorder=5,
            )

        ax.axvline(dates[-1], color="gray", ls=":", lw=1, alpha=0.7, label="Forecast start")
        ax.set_ylabel("Index (base \u2248 100)")
        ax.set_title(f"{label} Total Employment Index: Estimate + Forecast to 2026-01-12")
        ax.legend(fontsize=8, loc="upper left")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecast_sa_nsa.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'forecast_sa_nsa.png'}")


def _plot_jobs_forecast(
    dates, forecast_dates, levels, T_hist,
    chg_sa_hist, chg_nsa_hist, chg_sa_fwd, chg_nsa_fwd,
):
    obs_sa_levels = levels["ces_sa_level"].to_numpy().astype(float)
    obs_nsa_levels = levels["ces_nsa_level"].to_numpy().astype(float)
    base_dates = levels["ref_date"].to_list()
    obs_chg_sa = (obs_sa_levels[1:] - obs_sa_levels[:-1]) / 1000.0
    obs_chg_nsa = (obs_nsa_levels[1:] - obs_nsa_levels[:-1]) / 1000.0
    obs_chg_dates = base_dates[1:]

    n_recent = 24
    recent_start = max(0, T_hist - n_recent)
    recent_chg_start = max(0, recent_start - 1)
    recent_dates_chg = dates[recent_chg_start + 1: T_hist]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for panel_idx, (label, chg_hist, chg_fwd, obs_chg) in enumerate([
        ("SA", chg_sa_hist, chg_sa_fwd, obs_chg_sa),
        ("NSA", chg_nsa_hist, chg_nsa_fwd, obs_chg_nsa),
    ]):
        ax = axes[panel_idx]
        color_hist = "#2563eb"
        color_obs = "#ea580c"

        hist_mean = chg_hist[:, :, recent_chg_start:].mean(axis=(0, 1))
        hist_lo = np.percentile(chg_hist[:, :, recent_chg_start:], 10, axis=(0, 1))
        hist_hi = np.percentile(chg_hist[:, :, recent_chg_start:], 90, axis=(0, 1))

        ax.fill_between(recent_dates_chg, hist_lo, hist_hi, alpha=0.2, color=color_hist,
                        label="80% HDI")
        ax.plot(recent_dates_chg, hist_mean, color=color_hist, lw=2,
                label=f"Model {label} (jobs added)")

        fwd_mean = chg_fwd.mean(axis=(0, 1))
        fwd_lo = np.percentile(chg_fwd, 10, axis=(0, 1))
        fwd_hi = np.percentile(chg_fwd, 90, axis=(0, 1))

        ax.fill_between(forecast_dates, fwd_lo, fwd_hi, alpha=0.25, color="#ef4444",
                        label="Forecast 80% HDI")
        ax.plot(forecast_dates, fwd_mean, color="#ef4444", lw=2.5, ls="--", label="Forecast")

        for i, fd in enumerate(forecast_dates):
            fm = chg_fwd[:, :, i].flatten().mean()
            flo = np.percentile(chg_fwd[:, :, i].flatten(), 10)
            fhi = np.percentile(chg_fwd[:, :, i].flatten(), 90)
            ax.errorbar(fd, fm, yerr=[[fm - flo], [fhi - fm]], fmt="o", color="#ef4444",
                        markersize=8, capsize=6, capthick=2, elinewidth=2, zorder=10)
            ax.annotate(
                f"{fm:+,.0f}k", xy=(fd, fhi), xytext=(8, 8),
                textcoords="offset points", fontsize=9, fontweight="bold", color="#ef4444",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ef4444", alpha=0.9),
            )

        obs_idx = [i for i, d in enumerate(obs_chg_dates)
                   if recent_dates_chg[0] <= d <= dates[-1]]
        if obs_idx:
            ax.scatter([obs_chg_dates[i] for i in obs_idx],
                       [obs_chg[i] for i in obs_idx],
                       s=25, c=color_obs, alpha=0.8, zorder=5,
                       label=f"CES {label} (observed)")

        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axvline(dates[-1], color="#6b7280", ls=":", lw=1.2, alpha=0.8)
        ax.set_ylabel("Jobs added (thousands)", fontsize=11)
        ax.set_title(
            f"{label} Total Nonfarm Employment: Jobs Added (Month-over-Month) to 2026-01-12",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.tick_params(axis="x", labelsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+,.0f}"))
        ax.grid(axis="y", alpha=0.3, ls="--")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecast_levels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'forecast_levels.png'}")
