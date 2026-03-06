"""Standardised residual diagnostics by data source.

:func:`plot_residuals` computes and plots standardised residuals for every
data source (CES vintage-specific, each payroll provider, QCEW).  Under a
well-specified model, CES and provider residuals should be approximately
iid *N*(0, 1).  QCEW uses a Student-t likelihood (nu = ``QCEW_NU``), so
its standardised residuals follow *t*(nu) rather than *N*(0, 1) — slightly
heavier tails are expected.  Temporal patterns signal misfit for all sources.

For providers with AR(1) error structures the residuals are pre-whitened
(innovation residuals) so that serial correlation has been removed.
"""

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
    sigma_ces_sa = idata.posterior["sigma_ces_sa"].values.mean(axis=(0, 1))   # (n_ces_v,)
    sigma_ces_nsa = idata.posterior["sigma_ces_nsa"].values.mean(axis=(0, 1))  # (n_ces_v,)

    _orig_vintage_colors = {0: '#ff7f0e', 1: '#d62728', 2: '#2ca02c'}
    _orig_vintage_labels = {0: '1st', 1: '2nd', 2: 'Final'}
    vintage_map = data.get("ces_vintage_map", {0: 0, 1: 1, 2: 2})
    inv_map = {i: v for v, i in vintage_map.items()}
    vintage_colors = {i: _orig_vintage_colors[inv_map[i]] for i in inv_map}
    vintage_labels = {i: _orig_vintage_labels[inv_map[i]] for i in inv_map}

    n_panels = 2 + len(pp_data) + 1  # CES SA + CES NSA + PPs + QCEW
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.2 * n_panels), sharex=True)

    # --- CES SA ---
    ax = axes[0]
    ces_sa_obs = data["ces_sa_obs"]
    ces_sa_vidx = data["ces_sa_vintage_idx"]
    if len(ces_sa_obs) > 0:
        pred = alpha_ces + lambda_ces * g_total_sa[ces_sa_obs]
        sig = sigma_ces_sa[ces_sa_vidx]
        resid = (data["g_ces_sa"][ces_sa_obs] - pred) / sig
        for v in sorted(vintage_colors):
            mask = ces_sa_vidx == v
            if mask.any():
                ax.scatter(
                    dates_arr[ces_sa_obs[mask]], resid[mask], s=8,
                    c=vintage_colors[v], alpha=0.6, label=vintage_labels[v],
                )
        ax.legend(fontsize=7)
    _resid_lines(ax, "CES SA (best-available)")

    # --- CES NSA ---
    ax = axes[1]
    ces_nsa_obs = data["ces_nsa_obs"]
    ces_nsa_vidx = data["ces_nsa_vintage_idx"]
    if len(ces_nsa_obs) > 0:
        pred = alpha_ces + lambda_ces * g_total_nsa[ces_nsa_obs]
        sig = sigma_ces_nsa[ces_nsa_vidx]
        resid = (data["g_ces_nsa"][ces_nsa_obs] - pred) / sig
        for v in sorted(vintage_colors):
            mask = ces_nsa_vidx == v
            if mask.any():
                ax.scatter(
                    dates_arr[ces_nsa_obs[mask]], resid[mask], s=8,
                    c=vintage_colors[v], alpha=0.6, label=vintage_labels[v],
                )
        ax.legend(fontsize=7)
    _resid_lines(ax, "CES NSA (best-available)")

    n_ces_panels = 2

    # --- Per-provider PP ---
    for p_idx, pp in enumerate(pp_data):
        ax = axes[n_ces_panels + p_idx]
        name = pp["config"].name.lower()
        idx_obs = pp["pp_obs"]
        alp_p = idata.posterior[f"alpha_{name}"].values.flatten().mean()
        lam_p = idata.posterior[f"lam_{name}"].values.flatten().mean()
        sig_p = idata.posterior[f"sigma_pp_{name}"].values.flatten().mean()

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
    qcew_is_m2 = (
        np.asarray(data["qcew_is_m2"])
        if "qcew_is_m2" in data
        else np.zeros(len(idx_obs), dtype=bool)
    )
    if "sigma_qcew_mid" in idata.posterior and "qcew_noise_mult" in data:
        sigma_mid = float(idata.posterior["sigma_qcew_mid"].values.flatten().mean())
        sigma_boundary = float(
            idata.posterior["sigma_qcew_boundary"].values.flatten().mean()
        )
        qcew_noise_mult = np.asarray(data["qcew_noise_mult"], dtype=float)
        qcew_sigma = np.where(
            qcew_is_m2,
            sigma_mid * qcew_noise_mult,
            sigma_boundary * qcew_noise_mult,
        )
    else:
        qcew_sigma = np.full(len(idx_obs), 0.001)
    resid = (data["g_qcew"][idx_obs] - pred) / qcew_sigma
    ax.scatter(
        dates_arr[idx_obs][qcew_is_m2],
        resid[qcew_is_m2],
        s=12,
        c="darkred",
        alpha=0.7,
        label="M2 (mid-quarter)",
    )
    ax.scatter(
        dates_arr[idx_obs][~qcew_is_m2],
        resid[~qcew_is_m2],
        s=12,
        c="salmon",
        alpha=0.7,
        label="M3+M1 (boundary)",
    )
    _resid_lines(ax, "QCEW")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(
        "Standardised Residuals by Source (CES/PP: N(0,1); QCEW: t(nu))",
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
