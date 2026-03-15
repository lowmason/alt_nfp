"""Core result plots: growth rates, seasonal patterns, index reconstruction,
and birth/death diagnostics.

Visualisations produced:

* :func:`plot_growth_and_seasonal` — four-panel figure: SA total growth,
  NSA total growth, continuing-units growth vs providers, and Fourier
  seasonal pattern comparison.
* :func:`plot_reconstructed_index` — latent SA/NSA index overlaid on
  observed CES, QCEW, and provider series.
* :func:`plot_bd_diagnostics` — structural BD offset time-series,
  covariate scatter-plots, and decomposition.
"""

from __future__ import annotations

import logging

import arviz as az

logger = logging.getLogger(__name__)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from nfp_lookups.paths import OUTPUT_DIR

from .settings import NowcastConfig


# =========================================================================
# Growth rates & seasonal pattern
# =========================================================================


def plot_growth_and_seasonal(idata: az.InferenceData, data: dict, cfg: NowcastConfig | None = None) -> None:
    """Latent growth rates vs observed data and seasonal pattern."""
    dates = data["dates"]
    g_ces_sa = data["g_ces_sa"]
    g_ces_nsa = data["g_ces_nsa"]
    pp_data = data["pp_data"]

    pct = lambda x: x * 100  # noqa: E731

    g_cont_post = idata.posterior["g_cont"].values
    g_total_sa_post = idata.posterior["g_total_sa"].values
    g_total_nsa_post = idata.posterior["g_total_nsa"].values

    g_cont_mean = g_cont_post.mean(axis=(0, 1))
    g_cont_lo = np.percentile(g_cont_post, 10, axis=(0, 1))
    g_cont_hi = np.percentile(g_cont_post, 90, axis=(0, 1))

    g_sa_mean = g_total_sa_post.mean(axis=(0, 1))
    g_sa_lo = np.percentile(g_total_sa_post, 10, axis=(0, 1))
    g_sa_hi = np.percentile(g_total_sa_post, 90, axis=(0, 1))

    g_nsa_mean = g_total_nsa_post.mean(axis=(0, 1))
    g_nsa_lo = np.percentile(g_total_nsa_post, 10, axis=(0, 1))
    g_nsa_hi = np.percentile(g_total_nsa_post, 90, axis=(0, 1))

    dates_arr = np.array(dates)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)

    # --- Panel 1: SA total growth vs CES SA ---
    ax = axes[0]
    ax.fill_between(dates, pct(g_sa_lo), pct(g_sa_hi), alpha=0.25, color="steelblue",
                    label="80% CI")
    ax.plot(dates, pct(g_sa_mean), "steelblue", lw=1.5, label="Latent total (SA)")
    ax.scatter(dates, pct(g_ces_sa), s=10, c="darkorange", alpha=0.7,
               label="CES SA (observed)", zorder=5)
    for pp in pp_data:
        m = np.isfinite(pp["g_pp"])
        ax.scatter(dates_arr[m], pct(pp["g_pp"][m]), s=6, c=pp["color"], alpha=0.4,
                   label=f"{pp['name']} (NSA)", zorder=3)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Monthly growth (%)")
    ax.set_title("SA Total Employment Growth: Latent State vs CES SA & PP")
    ax.legend(fontsize=8, loc="upper right")
    _year_axis(ax)

    # --- Panel 2: NSA total growth ---
    ax = axes[1]
    ax.fill_between(dates, pct(g_nsa_lo), pct(g_nsa_hi), alpha=0.25, color="steelblue",
                    label="80% CI")
    ax.plot(dates, pct(g_nsa_mean), "steelblue", lw=1.5, label="Latent total (NSA)")
    ax.scatter(dates, pct(g_ces_nsa), s=10, c="darkorange", alpha=0.7,
               label="CES NSA (observed)", zorder=5)
    for pp in pp_data:
        m = np.isfinite(pp["g_pp"])
        ax.scatter(dates_arr[m], pct(pp["g_pp"][m]), s=8, c=pp["color"], alpha=0.5,
                   label=f"{pp['name']} (NSA)", zorder=4)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Monthly growth (%)")
    ax.set_title("NSA Total Employment Growth: Latent State vs CES NSA & PP")
    ax.legend(fontsize=8, loc="upper right")
    _year_axis(ax)

    # --- Panel 3: Continuing-units growth vs PP ---
    ax = axes[2]
    ax.fill_between(dates, pct(g_cont_lo), pct(g_cont_hi), alpha=0.25, color="steelblue",
                    label="80% CI")
    ax.plot(dates, pct(g_cont_mean), "steelblue", lw=1.5, label="Latent cont. units (SA)")
    for pp in pp_data:
        m = np.isfinite(pp["g_pp"])
        ax.scatter(dates_arr[m], pct(pp["g_pp"][m]), s=10, c=pp["color"], alpha=0.6,
                   label=f"{pp['name']} (NSA)", zorder=5)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Monthly growth (%)")
    ax.set_title("SA Continuing-Units Growth vs PP (NSA)")
    ax.legend(fontsize=8, loc="upper right")
    _year_axis(ax)

    # --- Panel 4: Seasonal pattern (current-year Fourier evaluation) ---
    cfg = cfg or NowcastConfig()
    K = cfg.model.fourier.n_harmonics
    fourier_post = idata.posterior['fourier_coefs_det'].values  # (chains, draws, n_years, 2K)
    last_yr_coefs = fourier_post[:, :, -1, :]  # (chains, draws, 2K)

    k_vals = np.arange(1, K + 1)
    month_idx = np.arange(12)
    cos_basis_12 = np.cos(2 * np.pi * k_vals * month_idx[:, None] / 12)  # (12, K)
    sin_basis_12 = np.sin(2 * np.pi * k_vals * month_idx[:, None] / 12)  # (12, K)

    # Evaluate seasonal for each posterior draw at each of the 12 months
    A_last = last_yr_coefs[:, :, :K]   # (chains, draws, K)
    B_last = last_yr_coefs[:, :, K:]   # (chains, draws, K)
    s_12 = np.zeros((*A_last.shape[:2], 12))
    for m in range(12):
        for k in range(K):
            s_12[:, :, m] += A_last[:, :, k] * cos_basis_12[m, k] \
                + B_last[:, :, k] * sin_basis_12[m, k]

    s_mean = s_12.mean(axis=(0, 1))
    s_lo = np.percentile(s_12, 10, axis=(0, 1))
    s_hi = np.percentile(s_12, 90, axis=(0, 1))

    emp_seasonal = np.zeros(12)
    emp_counts = np.zeros(12)
    diff = g_ces_nsa - g_ces_sa
    for t, d in enumerate(dates):
        if np.isfinite(diff[t]):
            emp_seasonal[d.month - 1] += diff[t]
            emp_counts[d.month - 1] += 1
    emp_seasonal = np.where(emp_counts > 0, emp_seasonal / emp_counts, 0.0)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax = axes[3]
    x_pos = np.arange(12)
    width = 0.35
    ax.bar(x_pos - width / 2, pct(s_mean), width, color="steelblue", alpha=0.7,
           label="Model estimate (current year)")
    ax.bar(x_pos + width / 2, pct(emp_seasonal), width, color="darkorange", alpha=0.7,
           label="Empirical (CES NSA \u2212 SA)")
    ax.errorbar(x_pos - width / 2, pct(s_mean),
                yerr=[pct(s_mean - s_lo), pct(s_hi - s_mean)], fmt="none", c="k", capsize=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(month_labels)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Seasonal effect (%)")
    ax.set_title("Estimated vs Empirical Monthly Seasonal Pattern (Current Year Fourier)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "growth_and_seasonal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'growth_and_seasonal.png'}")


# =========================================================================
# Reconstructed index
# =========================================================================


def plot_reconstructed_index(idata: az.InferenceData, data: dict, cfg: NowcastConfig | None = None) -> None:
    """Reconstructed SA/NSA index vs observed series."""
    dates = data["dates"]
    levels = data["levels"]
    pp_data = data["pp_data"]

    g_total_mean = idata.posterior["g_total_sa"].values.mean(axis=(0, 1))
    g_nsa_mean = idata.posterior["g_total_nsa"].values.mean(axis=(0, 1))

    base_dates = levels["ref_date"].to_list()
    ces_sa_vals = levels["ces_sa_index"].to_numpy().astype(float)
    ces_nsa_vals = levels["ces_nsa_index"].to_numpy().astype(float)
    log_base_sa = np.log(ces_sa_vals[0])
    log_base_nsa = np.log(ces_nsa_vals[0])

    index_sa = np.exp(log_base_sa + np.cumsum(g_total_mean))
    index_nsa = np.exp(log_base_nsa + np.cumsum(g_nsa_mean))

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    ax.plot(dates, index_sa, "steelblue", lw=2, label="Latent total (SA)")
    ax.plot(dates, index_nsa, "steelblue", lw=1.5, ls="--", alpha=0.6,
            label="Latent total (NSA)")
    ax.plot(base_dates[1:], ces_sa_vals[1:], "darkorange", lw=1.5, alpha=0.8,
            label="CES SA (observed)")
    ax.plot(base_dates[1:], ces_nsa_vals[1:], "darkorange", lw=1, ls="--", alpha=0.5,
            label="CES NSA (observed)")

    pp_dates_arr = np.array(base_dates)
    for pp in pp_data:
        try:
            vals = levels[pp["emp_col"]].to_numpy().astype(float)
            mask = np.isfinite(vals)
            if mask.sum() > 1:
                ax.plot(pp_dates_arr[mask][1:], vals[mask][1:], color=pp["color"], lw=1,
                        alpha=0.6, label=f"{pp['name']} (NSA)")
        except KeyError:
            logger.debug("Provider column %s not in levels DataFrame", pp["emp_col"])

    # QCEW — small circles instead of bold diamonds
    qcew_rows = levels.filter(pl.col("qcew_nsa_index").is_not_null())
    ax.scatter(
        qcew_rows["ref_date"].to_list(),
        qcew_rows["qcew_nsa_index"].to_numpy(),
        s=20, c="red", marker="o", alpha=0.7, label="QCEW (NSA)", zorder=5,
    )

    ax.set_ylabel("Index (base \u2248 100)")
    ax.set_title("Reconstructed Latent Index vs Observed Series")
    ax.legend(fontsize=8)
    _year_axis(ax)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "reconstructed_index.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'reconstructed_index.png'}")


# =========================================================================
# BD diagnostic plot (new in v3)
# =========================================================================


def plot_bd_diagnostics(idata: az.InferenceData, data: dict, cfg: NowcastConfig | None = None) -> None:
    """Time-series of structural BD, covariate relationships, and decomposition."""
    dates = data["dates"]
    bd_post = idata.posterior["bd"].values  # (chains, draws, T)
    bd_mean = bd_post.mean(axis=(0, 1))
    bd_lo = np.percentile(bd_post, 10, axis=(0, 1))
    bd_hi = np.percentile(bd_post, 90, axis=(0, 1))

    bd_qcew_lagged = data["bd_qcew_lagged"]

    n_panels = 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels))

    # Panel 1: bd_t time series
    ax = axes[0]
    ax.fill_between(dates, bd_lo * 100, bd_hi * 100, alpha=0.25, color="steelblue",
                    label="80% CI")
    ax.plot(dates, bd_mean * 100, "steelblue", lw=1.5, label="bd_t (posterior mean)")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("BD offset (%/mo)")
    ax.set_title("Structural Birth/Death Offset Over Time")
    ax.legend(fontsize=8)
    _year_axis(ax)

    # Panel 2: bd_t vs lagged QCEW BD proxy
    ax = axes[1]
    bq_obs = np.isfinite(bd_qcew_lagged)
    if bq_obs.any():
        ax.scatter(bd_qcew_lagged[bq_obs] * 100, bd_mean[bq_obs] * 100, s=8,
                   c="steelblue", alpha=0.6)
        ax.set_xlabel("Lagged QCEW BD proxy (%)")
        ax.set_ylabel("bd_t (%/mo)")
        ax.set_title("BD Offset vs Lagged QCEW BD Proxy")
    else:
        ax.set_visible(False)

    # Panel 3: BD covariate decomposition
    ax = axes[2]
    phi_0_m = idata.posterior["phi_0"].values.flatten().mean()
    c_phi0 = np.full(len(dates), phi_0_m * 100)
    ax.plot(dates, c_phi0, lw=1.2, label="\u03c6\u2080 (intercept)", color="gray", ls="--")

    # Cyclical contributions (derived from cfg.indicators)
    cfg = cfg or NowcastConfig()

    _cyc_colors = ['#9467bd', '#ff7f0e', '#17becf', '#d62728', '#8c564b']
    cyc_contribs = np.zeros(len(dates))
    if "phi_3" in idata.posterior:
        phi_3_m = idata.posterior["phi_3"].values.mean(axis=(0, 1))  # (n_cyc,)
        cyc_i = 0
        for ci, spec in enumerate(cfg.indicators):
            key = f"{spec.name}_c"
            arr = data.get(key)
            if arr is not None and np.any(arr != 0.0):
                contrib = phi_3_m[cyc_i] * arr * 100
                clr = _cyc_colors[ci % len(_cyc_colors)]
                ax.plot(dates, contrib, lw=1.2,
                        label=f"\u03c6\u2083\u00b7{spec.name}", color=clr)
                cyc_contribs += contrib
                cyc_i += 1

    # Residual (innovation)
    xi_bd_m = bd_mean * 100 - c_phi0 - cyc_contribs
    ax.plot(dates, xi_bd_m, lw=0.8, alpha=0.5, label="\u03c3_bd\u00b7\u03be (residual)",
            color="lightgray")

    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Contribution (%/mo)")
    ax.set_title("BD Covariate Decomposition")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    _year_axis(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "bd_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'bd_diagnostics.png'}")


# =========================================================================
# Helpers
# =========================================================================


def _year_axis(ax) -> None:
    """Configure *ax* x-axis with yearly tick marks."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
