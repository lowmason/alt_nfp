# ---------------------------------------------------------------------------
# alt_nfp.residuals — Standardised residual plots
# ---------------------------------------------------------------------------
from __future__ import annotations

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_DIR


def plot_residuals(idata: az.InferenceData, data: dict) -> None:
    """Plot standardised residuals per data source over time.

    Residuals should be approximately iid N(0,1) if the model is
    well-specified.  Temporal patterns or heavy tails indicate misfit.
    """
    dates = data["dates"]
    dates_arr = np.array(dates)
    pp_data = data["pp_data"]

    # Posterior means of latent states
    g_cont = idata.posterior["g_cont"].values.mean(axis=(0, 1))
    g_total_sa = idata.posterior["g_total_sa"].values.mean(axis=(0, 1))
    g_total_nsa = idata.posterior["g_total_nsa"].values.mean(axis=(0, 1))
    seasonal = idata.posterior["seasonal"].values.mean(axis=(0, 1))  # (T,)
    g_cont_nsa = g_cont + seasonal

    alpha_ces = idata.posterior["alpha_ces"].values.flatten().mean()
    lambda_ces = idata.posterior["lambda_ces"].values.flatten().mean()
    sigma_ces_sa = idata.posterior["sigma_ces_sa"].values.mean(axis=(0, 1))   # (3,)
    sigma_ces_nsa = idata.posterior["sigma_ces_nsa"].values.mean(axis=(0, 1))  # (3,)

    # Count CES vintage panels (only vintages with data)
    vintage_labels = ['1st print', '2nd print', 'Final']
    vintage_colors = ['#ff7f0e', '#d62728', '#2ca02c']
    ces_panels: list[tuple[str, str, np.ndarray, np.ndarray, float, str]] = []
    for v in range(3):
        g_sa_v = data['g_ces_sa_by_vintage'][v]
        obs_sa = np.where(np.isfinite(g_sa_v))[0]
        if len(obs_sa) > 0:
            ces_panels.append((
                f'CES SA ({vintage_labels[v]})', 'sa', g_sa_v,
                obs_sa, sigma_ces_sa[v], vintage_colors[v],
            ))
        g_nsa_v = data['g_ces_nsa_by_vintage'][v]
        obs_nsa = np.where(np.isfinite(g_nsa_v))[0]
        if len(obs_nsa) > 0:
            ces_panels.append((
                f'CES NSA ({vintage_labels[v]})', 'nsa', g_nsa_v,
                obs_nsa, sigma_ces_nsa[v], vintage_colors[v],
            ))

    n_panels = len(ces_panels) + len(pp_data) + 1  # CES vintages + PPs + QCEW
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.2 * n_panels), sharex=True)

    # --- CES vintage panels ---
    for panel_i, (title, sa_or_nsa, g_v, obs_v, sig_v, clr) in enumerate(ces_panels):
        ax = axes[panel_i]
        if sa_or_nsa == 'sa':
            pred = alpha_ces + lambda_ces * g_total_sa[obs_v]
        else:
            pred = alpha_ces + lambda_ces * g_total_nsa[obs_v]
        resid = (g_v[obs_v] - pred) / sig_v
        ax.scatter(dates_arr[obs_v], resid, s=8, c=clr, alpha=0.6)
        _resid_lines(ax, title)

    n_ces_panels = len(ces_panels)

    # --- Per-provider PP ---
    for p_idx, pp in enumerate(pp_data):
        ax = axes[n_ces_panels + p_idx]
        name = pp["config"].name.lower()
        idx_obs = pp["pp_obs"]
        alp_p = idata.posterior[f"alpha_{name}"].values.flatten().mean()
        lam_p = idata.posterior[f"lam_{name}"].values.flatten().mean()
        sig_p = idata.posterior[f"sigma_{name}"].values.flatten().mean()

        mu_base = alp_p + lam_p * g_cont_nsa[idx_obs]

        if pp["config"].error_model == "ar1":
            rho_p = idata.posterior[f"rho_{name}"].values.flatten().mean()
            u = pp["g_pp"][idx_obs] - mu_base
            resid = np.zeros_like(u)
            resid[0] = u[0] * np.sqrt(1 - rho_p**2) / sig_p
            resid[1:] = (u[1:] - rho_p * u[:-1]) / sig_p
            title_suffix = " (AR(1)-filtered)"
        else:
            resid = (pp["g_pp"][idx_obs] - mu_base) / sig_p
            title_suffix = ""

        ax.scatter(dates_arr[idx_obs], resid, s=8, c=pp["color"], alpha=0.6)
        _resid_lines(ax, f"{pp['name']}{title_suffix}")

    # --- QCEW ---
    ax = axes[-1]
    idx_obs = data["qcew_obs"]
    pred = g_total_nsa[idx_obs]
    qcew_sigma = np.where(data["qcew_is_m3"], 0.0005, 0.0015)
    resid = (data["g_qcew"][idx_obs] - pred) / qcew_sigma
    m3 = data["qcew_is_m3"]
    ax.scatter(dates_arr[idx_obs][m3], resid[m3], s=12, c="darkred", alpha=0.7,
               label="M3 (quarter-end)")
    ax.scatter(dates_arr[idx_obs][~m3], resid[~m3], s=12, c="salmon", alpha=0.7,
               label="M1-2 (retrospective UI)")
    _resid_lines(ax, "QCEW")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(
        "Standardised Residuals by Source (should be approx. N(0,1))",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'residuals.png'}")


def _resid_lines(ax, title: str) -> None:
    """Add zero-line, ±2σ guides, and axis labels."""
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.axhline(2, color="red", lw=0.5, ls=":", alpha=0.5)
    ax.axhline(-2, color="red", lw=0.5, ls=":", alpha=0.5)
    ax.set_ylabel("Std. residual")
    ax.set_title(title)
