# ---------------------------------------------------------------------------
# alt_nfp.plots — Growth-rate, seasonal, reconstructed-index, and BD plots
# ---------------------------------------------------------------------------
from __future__ import annotations

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .config import OUTPUT_DIR


# =========================================================================
# Growth rates & seasonal pattern
# =========================================================================


def plot_growth_and_seasonal(idata: az.InferenceData, data: dict) -> None:
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

    # --- Panel 4: Seasonal pattern ---
    s_post = idata.posterior["seasonal"].values
    s_mean = s_post.mean(axis=(0, 1))
    s_lo = np.percentile(s_post, 10, axis=(0, 1))
    s_hi = np.percentile(s_post, 90, axis=(0, 1))

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
           label="Model estimate")
    ax.bar(x_pos + width / 2, pct(emp_seasonal), width, color="darkorange", alpha=0.7,
           label="Empirical (CES NSA \u2212 SA)")
    ax.errorbar(x_pos - width / 2, pct(s_mean),
                yerr=[pct(s_mean - s_lo), pct(s_hi - s_mean)], fmt="none", c="k", capsize=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(month_labels)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Seasonal effect (%)")
    ax.set_title("Estimated vs Empirical Monthly Seasonal Pattern")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "growth_and_seasonal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'growth_and_seasonal.png'}")


# =========================================================================
# Reconstructed index
# =========================================================================


def plot_reconstructed_index(idata: az.InferenceData, data: dict) -> None:
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
            vals = levels[pp["index_col"]].to_numpy().astype(float)
            mask = np.isfinite(vals)
            if mask.sum() > 1:
                ax.plot(pp_dates_arr[mask][1:], vals[mask][1:], color=pp["color"], lw=1,
                        alpha=0.6, label=f"{pp['name']} (NSA)")
        except Exception:
            pass

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


def plot_bd_diagnostics(idata: az.InferenceData, data: dict) -> None:
    """Time-series of structural BD and covariate relationships."""
    dates = data["dates"]
    dates_arr = np.array(dates)
    bd_post = idata.posterior["bd"].values  # (chains, draws, T)
    bd_mean = bd_post.mean(axis=(0, 1))
    bd_lo = np.percentile(bd_post, 10, axis=(0, 1))
    bd_hi = np.percentile(bd_post, 90, axis=(0, 1))

    birth_rate = data["birth_rate"]
    bd_qcew_lagged = data["bd_qcew_lagged"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

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

    # Panel 2: bd_t vs birth rate
    ax = axes[1]
    br_obs = np.isfinite(birth_rate)
    if br_obs.any():
        ax.scatter(birth_rate[br_obs] * 100, bd_mean[br_obs] * 100, s=8, c="steelblue",
                   alpha=0.6)
        ax.set_xlabel("PP birth rate (%)")
        ax.set_ylabel("bd_t (%/mo)")
        ax.set_title("BD Offset vs PP Birth Rate")
    else:
        ax.set_visible(False)

    # Panel 3: bd_t vs lagged QCEW BD proxy
    ax = axes[2]
    bq_obs = np.isfinite(bd_qcew_lagged)
    if bq_obs.any():
        ax.scatter(bd_qcew_lagged[bq_obs] * 100, bd_mean[bq_obs] * 100, s=8,
                   c="steelblue", alpha=0.6)
        ax.set_xlabel(f"Lagged QCEW BD proxy (%)")
        ax.set_ylabel("bd_t (%/mo)")
        ax.set_title("BD Offset vs Lagged QCEW BD Proxy")
    else:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "bd_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'bd_diagnostics.png'}")


# =========================================================================
# Helpers
# =========================================================================


def _year_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
