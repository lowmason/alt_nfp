"""Sampling diagnostics, source contributions, and divergence visualisation.

After MCMC sampling this module provides:

* :func:`print_diagnostics` — structured console summary of convergence
  metrics (R-hat, ESS, divergences) and key parameter estimates.
* :func:`print_source_contributions` — precision-weighted information
  budget showing each data source's contribution to the posterior.
* :func:`plot_divergences` — bivariate scatter-plots highlighting
  divergent transitions for geometric-pathology diagnosis.
"""

from __future__ import annotations

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .config import OUTPUT_DIR, SIGMA_QCEW_M3, SIGMA_QCEW_M12

# =========================================================================
# Parameter summary & convergence
# =========================================================================


def print_diagnostics(idata: az.InferenceData, data: dict) -> None:
    """Print sampling diagnostics and structured parameter summary."""
    pp_data = data["pp_data"]

    print("=" * 72)
    print("SAMPLING DIAGNOSTICS")
    print("=" * 72)

    divs = int(idata.sample_stats.diverging.sum().values)
    print(f"Divergences: {divs}")
    try:
        max_td = int(idata.sample_stats.tree_depth.max().values)
        print(f"Max tree depth: {max_td}")
    except Exception:
        pass

    # Build var_names dynamically — era-specific or scalar latent params
    has_era = "mu_g_era" in idata.posterior
    if has_era:
        latent_vars = ["mu_g_era", "phi_raw_era", "sigma_g"]
    else:
        latent_vars = ["mu_g", "phi", "sigma_g"]

    var_names = [
        *latent_vars,
        "sigma_fourier",
        "phi_0", "sigma_bd",
        *(["phi_1"] if "phi_1" in idata.posterior else []),
        *(["phi_2"] if "phi_2" in idata.posterior else []),
        "alpha_ces", "lambda_ces", "sigma_ces_sa", "sigma_ces_nsa",
    ]
    # Cyclical indicator loadings (phi_3) if present
    if "phi_3" in idata.posterior:
        var_names.append("phi_3")
    for pp in pp_data:
        name = pp["config"].name.lower()
        var_names.extend([f"alpha_{name}", f"lam_{name}", f"sigma_pp_{name}"])
        if pp["config"].error_model == "ar1":
            var_names.append(f"rho_{name}")

    print("\n" + "=" * 72)
    print("PARAMETER SUMMARY")
    print("=" * 72)
    summary = az.summary(idata, var_names=var_names, hdi_prob=0.80)
    print(summary.to_string())

    rhat_bad = summary[summary["r_hat"] > 1.01]
    ess_bad = summary[summary["ess_bulk"] < 400]
    if len(rhat_bad) > 0:
        print("\n** WARNING: Parameters with R-hat > 1.01:")
        for pname, row in rhat_bad.iterrows():
            print(f"    {pname}: R-hat = {row['r_hat']:.4f}")
    if len(ess_bad) > 0:
        print("\n** WARNING: Parameters with ESS_bulk < 400:")
        for pname, row in ess_bad.iterrows():
            print(f"    {pname}: ESS = {row['ess_bulk']:.0f}")
    if len(rhat_bad) == 0 and len(ess_bad) == 0:
        print("\nAll parameters converged (R-hat <= 1.01, ESS_bulk >= 400)")

    # ----- Key outputs -----
    print("\n" + "=" * 72)
    print("KEY OUTPUTS")
    print("=" * 72)

    _print_bd_summary(idata)
    _print_ces_summary(idata)
    _print_pp_summary(idata, pp_data)


def _print_bd_summary(idata: az.InferenceData) -> None:
    """Print posterior summary of structural birth/death parameters."""
    phi_0 = idata.posterior["phi_0"].values.flatten()
    sigma_bd = idata.posterior["sigma_bd"].values.flatten()
    bd_mean = idata.posterior["bd"].values.mean(axis=(0, 1))

    print("\nStructural Birth/Death:")
    print(
        f"  \u03c6_0 (intercept):    {phi_0.mean() * 100:+.4f}%/mo  "
        f"[{np.percentile(phi_0, 10) * 100:+.4f}%, "
        f"{np.percentile(phi_0, 90) * 100:+.4f}%]"
    )
    if "phi_1" in idata.posterior:
        phi_1 = idata.posterior["phi_1"].values.flatten()
        print(
            f"  \u03c6_1 (birth rate):   {phi_1.mean():.3f}  "
            f"[{np.percentile(phi_1, 10):.3f}, {np.percentile(phi_1, 90):.3f}]"
        )
    if "phi_2" in idata.posterior:
        phi_2 = idata.posterior["phi_2"].values.flatten()
        print(
            f"  \u03c6_2 (QCEW BD lag):  {phi_2.mean():.3f}  "
            f"[{np.percentile(phi_2, 10):.3f}, {np.percentile(phi_2, 90):.3f}]"
        )
    print(f"  \u03c3_bd (innovation):  {sigma_bd.mean() * 100:.4f}%")

    if "phi_3" in idata.posterior:
        from .config import CYCLICAL_INDICATORS

        phi_3 = idata.posterior["phi_3"].values  # (chains, draws, n_cyclical)
        cyclical_labels = [spec['name'] for spec in CYCLICAL_INDICATORS]
        n_cyc = phi_3.shape[-1]
        for i in range(n_cyc):
            v = phi_3[:, :, i].flatten()
            lbl = cyclical_labels[i] if i < len(cyclical_labels) else f'cyc_{i}'
            print(
                f"  \u03c6_3[{lbl}]:  {v.mean():.3f}  "
                f"[{np.percentile(v, 10):.3f}, {np.percentile(v, 90):.3f}]"
            )

    print(
        f"  bd_t mean over sample: {bd_mean.mean() * 100:+.4f}%/mo "
        f"(annualised: {bd_mean.mean() * 12 * 100:+.2f}%)"
    )
    print(f"  bd_t range:         [{bd_mean.min() * 100:+.4f}%, {bd_mean.max() * 100:+.4f}%]")


def _print_ces_summary(idata: az.InferenceData) -> None:
    """Print posterior summary of CES observation parameters."""
    a = idata.posterior["alpha_ces"].values.flatten()
    l = idata.posterior["lambda_ces"].values.flatten()
    s_sa = idata.posterior["sigma_ces_sa"].values   # (chains, draws, 3)
    s_nsa = idata.posterior["sigma_ces_nsa"].values  # (chains, draws, 3)

    print("\nCES observation parameters (vs QCEW anchor):")
    print(f"  \u03b1_ces  = {a.mean() * 100:+.4f}%/mo  (bias relative to true growth)")
    print(
        f"  \u03bb_ces  = {l.mean():.4f}  "
        f"[{np.percentile(l, 10):.4f}, {np.percentile(l, 90):.4f}]"
    )
    vintage_labels = ['1st', '2nd', 'Final']
    for v in range(3):
        sa_v = s_sa[:, :, v].flatten()
        nsa_v = s_nsa[:, :, v].flatten()
        print(
            f"  \u03c3_ces_sa[{vintage_labels[v]}] = {sa_v.mean() * 100:.3f}%  "
            f"\u03c3_ces_nsa[{vintage_labels[v]}] = {nsa_v.mean() * 100:.3f}%"
        )


def _print_pp_summary(idata: az.InferenceData, pp_data: list[dict]) -> None:
    """Print posterior summary of payroll-provider observation parameters."""
    print("\nPP signal loadings (\u03bb), bias (\u03b1), and noise (\u03c3):")
    for pp in pp_data:
        name = pp["config"].name.lower()
        lam_p = idata.posterior[f"lam_{name}"].values.flatten()
        alp_p = idata.posterior[f"alpha_{name}"].values.flatten()
        sig_p = idata.posterior[f"sigma_pp_{name}"].values.flatten()
        extra = ""
        if pp["config"].error_model == "ar1":
            rho_p = idata.posterior[f"rho_{name}"].values.flatten()
            marg_sigma = sig_p.mean() / np.sqrt(1 - rho_p.mean() ** 2)
            extra = (
                f"  |  \u03c1 = {rho_p.mean():.3f} "
                f"[{np.percentile(rho_p, 10):.3f}, {np.percentile(rho_p, 90):.3f}]"
                f"  |  \u03c3_marg = {marg_sigma * 100:.3f}%"
            )
        print(
            f"  {pp['name']:5}: \u03bb = {lam_p.mean():.3f} "
            f"[{np.percentile(lam_p, 10):.3f}, {np.percentile(lam_p, 90):.3f}]"
            f"  |  \u03b1 = {alp_p.mean() * 100:+.4f}%"
            f"  |  \u03c3 = {sig_p.mean() * 100:.3f}%{extra}"
        )


# =========================================================================
# Source contribution (precision-weighted)
# =========================================================================


def print_source_contributions(idata: az.InferenceData, data: dict) -> None:
    """Quantify each source's precision-weighted information contribution."""
    pp_data = data["pp_data"]

    sigma_ces_sa = idata.posterior["sigma_ces_sa"].values   # (chains, draws, 3)
    sigma_ces_nsa = idata.posterior["sigma_ces_nsa"].values  # (chains, draws, 3)
    lambda_ces = idata.posterior["lambda_ces"].values.flatten().mean()

    prec_qcew_m3 = 1.0 / SIGMA_QCEW_M3**2
    prec_qcew_m12 = 1.0 / SIGMA_QCEW_M12**2

    qcew_is_m3 = data["qcew_is_m3"]
    n_qcew_m3 = int(qcew_is_m3.sum())
    n_qcew_m12 = len(data["qcew_obs"]) - n_qcew_m3
    total_prec_qcew = prec_qcew_m3 * n_qcew_m3 + prec_qcew_m12 * n_qcew_m12

    # CES vintage-specific precision
    vintage_labels = ['1st', '2nd', 'Final']
    ces_rows: list[tuple[str, int, float, float]] = []
    total_prec_ces = 0.0
    for v in range(3):
        g_sa_v = data['g_ces_sa_by_vintage'][v]
        g_nsa_v = data['g_ces_nsa_by_vintage'][v]
        n_sa = int(np.sum(np.isfinite(g_sa_v)))
        n_nsa = int(np.sum(np.isfinite(g_nsa_v)))

        sig_sa_v = sigma_ces_sa[:, :, v].flatten().mean()
        sig_nsa_v = sigma_ces_nsa[:, :, v].flatten().mean()

        if n_sa > 0:
            prec_sa = lambda_ces**2 / sig_sa_v**2
            total_sa = prec_sa * n_sa
            ces_rows.append((f"CES SA ({vintage_labels[v]})", n_sa, prec_sa, total_sa))
            total_prec_ces += total_sa
        if n_nsa > 0:
            prec_nsa = lambda_ces**2 / sig_nsa_v**2
            total_nsa = prec_nsa * n_nsa
            ces_rows.append((f"CES NSA ({vintage_labels[v]})", n_nsa, prec_nsa, total_nsa))
            total_prec_ces += total_nsa

    pp_rows: list[tuple[str, int, float, float]] = []
    total_prec_pp = 0.0
    for pp in pp_data:
        name = pp["config"].name.lower()
        sig_p = idata.posterior[f"sigma_pp_{name}"].values.flatten().mean()
        lam_p = idata.posterior[f"lam_{name}"].values.flatten().mean()
        n_obs = len(pp["pp_obs"])
        prec_p = lam_p**2 / sig_p**2
        if pp["config"].error_model == "ar1":
            rho_p = idata.posterior[f"rho_{name}"].values.flatten().mean()
            prec_p = lam_p**2 * (1 - rho_p**2) / sig_p**2
        total_p = prec_p * n_obs
        pp_rows.append((pp["name"], n_obs, prec_p, total_p))
        total_prec_pp += total_p

    total_all = total_prec_ces + total_prec_qcew + total_prec_pp

    print("\n" + "=" * 72)
    print("DATA SOURCE CONTRIBUTION (precision-weighted)")
    print("=" * 72)
    hdr = f"{'Source':<20} {'Obs':<8} {'Prec/obs':>12} {'Total prec':>14} {'Share':>8}"
    print(hdr)
    print("-" * 72)

    def _row(nm: str, n: int, pp_obs: float, total: float) -> None:
        pct = 100.0 * total / total_all
        print(f"{nm:<20} {n:<8} {pp_obs:>12,.0f} {total:>14,.0f} {pct:>7.1f}%")

    for nm, n, pp_per, pp_tot in ces_rows:
        _row(nm, n, pp_per, pp_tot)
    for nm, n, pp_per, pp_tot in pp_rows:
        _row(nm, n, pp_per, pp_tot)
    _row("QCEW (M3)", n_qcew_m3, prec_qcew_m3, prec_qcew_m3 * n_qcew_m3)
    _row("QCEW (M1-2)", n_qcew_m12, prec_qcew_m12, prec_qcew_m12 * n_qcew_m12)
    print("-" * 72)
    print(f"{'TOTAL':<16} {'':8} {'':12} {total_all:>14,.0f} {'100.0%':>8}")


def compute_precision_budget(idata: az.InferenceData, data: dict) -> pl.DataFrame:
    """Compute precision-weighted information budget as a structured DataFrame.

    Returns a DataFrame with columns:
        source, n_obs, precision_per_obs, total_precision, share,
        lambda_mean, alpha_mean
    """
    pp_data = data["pp_data"]

    sigma_ces_sa = idata.posterior["sigma_ces_sa"].values
    sigma_ces_nsa = idata.posterior["sigma_ces_nsa"].values
    lambda_ces = float(idata.posterior["lambda_ces"].values.flatten().mean())
    alpha_ces = float(idata.posterior["alpha_ces"].values.flatten().mean())

    prec_qcew_m3 = 1.0 / SIGMA_QCEW_M3**2
    prec_qcew_m12 = 1.0 / SIGMA_QCEW_M12**2

    qcew_is_m3 = data["qcew_is_m3"]
    n_qcew_m3 = int(qcew_is_m3.sum())
    n_qcew_m12 = len(data["qcew_obs"]) - n_qcew_m3

    rows: list[dict] = []

    # CES vintage-specific precision
    vintage_labels = ["1st", "2nd", "Final"]
    for v in range(3):
        g_sa_v = data["g_ces_sa_by_vintage"][v]
        g_nsa_v = data["g_ces_nsa_by_vintage"][v]
        n_sa = int(np.sum(np.isfinite(g_sa_v)))
        n_nsa = int(np.sum(np.isfinite(g_nsa_v)))

        sig_sa_v = float(sigma_ces_sa[:, :, v].flatten().mean())
        sig_nsa_v = float(sigma_ces_nsa[:, :, v].flatten().mean())

        if n_sa > 0:
            prec_sa = lambda_ces**2 / sig_sa_v**2
            rows.append({
                "source": f"CES SA ({vintage_labels[v]})",
                "n_obs": n_sa,
                "precision_per_obs": prec_sa,
                "total_precision": prec_sa * n_sa,
                "lambda_mean": lambda_ces,
                "alpha_mean": alpha_ces,
            })
        if n_nsa > 0:
            prec_nsa = lambda_ces**2 / sig_nsa_v**2
            rows.append({
                "source": f"CES NSA ({vintage_labels[v]})",
                "n_obs": n_nsa,
                "precision_per_obs": prec_nsa,
                "total_precision": prec_nsa * n_nsa,
                "lambda_mean": lambda_ces,
                "alpha_mean": alpha_ces,
            })

    # Provider precision
    for pp in pp_data:
        name = pp["config"].name.lower()
        sig_p = float(idata.posterior[f"sigma_pp_{name}"].values.flatten().mean())
        lam_p = float(idata.posterior[f"lam_{name}"].values.flatten().mean())
        alp_p = float(idata.posterior[f"alpha_{name}"].values.flatten().mean())
        n_obs = len(pp["pp_obs"])
        prec_p = lam_p**2 / sig_p**2
        if pp["config"].error_model == "ar1":
            rho_p = float(idata.posterior[f"rho_{name}"].values.flatten().mean())
            prec_p = lam_p**2 * (1 - rho_p**2) / sig_p**2
        rows.append({
            "source": pp["name"],
            "n_obs": n_obs,
            "precision_per_obs": prec_p,
            "total_precision": prec_p * n_obs,
            "lambda_mean": lam_p,
            "alpha_mean": alp_p,
        })

    # QCEW precision
    if n_qcew_m3 > 0:
        rows.append({
            "source": "QCEW (M3)",
            "n_obs": n_qcew_m3,
            "precision_per_obs": prec_qcew_m3,
            "total_precision": prec_qcew_m3 * n_qcew_m3,
            "lambda_mean": 1.0,
            "alpha_mean": 0.0,
        })
    if n_qcew_m12 > 0:
        rows.append({
            "source": "QCEW (M1-2)",
            "n_obs": n_qcew_m12,
            "precision_per_obs": prec_qcew_m12,
            "total_precision": prec_qcew_m12 * n_qcew_m12,
            "lambda_mean": 1.0,
            "alpha_mean": 0.0,
        })

    df = pl.DataFrame(rows)
    total = df["total_precision"].sum()
    df = df.with_columns((pl.col("total_precision") / total).alias("share"))
    return df


# =========================================================================
# Divergence scatter-plots
# =========================================================================


def plot_divergences(idata: az.InferenceData, data: dict) -> None:
    """Bivariate scatter-plots highlighting divergent transitions."""
    diverging = idata.sample_stats.diverging.values
    n_divs = int(diverging.sum())
    if n_divs == 0:
        print("\nNo divergent transitions \u2014 skipping divergence plot.")
        return

    print(f"\nPlotting {n_divs} divergent transitions\u2026")
    div_flat = diverging.flatten().astype(bool)

    # Build parameter pairs dynamically
    has_era = "mu_g_era" in idata.posterior
    pairs: list[tuple[str, str, str, str]] = [
        ("lambda_ces", "alpha_ces", "\u03bb_CES", "\u03b1_CES"),
    ]
    if "phi_1" in idata.posterior:
        pairs.append(("phi_0", "phi_1", "\u03c6_0 (BD)", "\u03c6_1 (birth rate)"))
    if not has_era:
        pairs.insert(0, ("phi", "sigma_g", "\u03c6", "\u03c3_g"))
    for pp in data["pp_data"]:
        if pp["config"].error_model == "ar1":
            n = pp["config"].name.lower()
            pairs.append(
                (f"rho_{n}", f"sigma_pp_{n}", f"\u03c1_{pp['name']}", f"\u03c3_{pp['name']}")
            )

    n_pairs = len(pairs)
    n_cols = 2
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes_flat = axes.flatten() if n_pairs > 2 else (axes.flatten() if n_pairs > 1 else [axes])

    for idx, (p1, p2, l1, l2) in enumerate(pairs):
        ax = axes_flat[idx]
        v1 = idata.posterior[p1].values.flatten()
        v2 = idata.posterior[p2].values.flatten()

        ax.scatter(v1[~div_flat], v2[~div_flat], s=2, alpha=0.1, c="steelblue", rasterized=True)
        ax.scatter(
            v1[div_flat],
            v2[div_flat],
            s=15,
            alpha=0.8,
            c="limegreen",
            edgecolors="darkgreen",
            lw=0.5,
            label=f"Divergent (n={n_divs})",
            zorder=10,
        )
        ax.set_xlabel(l1)
        ax.set_ylabel(l2)
        ax.legend(fontsize=8)

    for idx in range(n_pairs, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Divergence Diagnostics: Bivariate Scatterplots", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "divergences.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'divergences.png'}")
