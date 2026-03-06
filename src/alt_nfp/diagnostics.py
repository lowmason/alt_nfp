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

from .config import OUTPUT_DIR, QCEW_NU

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
        latent_vars = ["mu_g_era", "phi_raw", "tau"]
    else:
        latent_vars = ["mu_g", "phi_raw", "tau"]

    var_names = [
        *latent_vars,
        "sigma_fourier",
        "phi_0", "sigma_bd",
        "alpha_ces", "lambda_ces", "sigma_ces_sa", "sigma_ces_nsa",
        *(
            ["sigma_qcew_mid", "sigma_qcew_boundary"]
            if "sigma_qcew_mid" in idata.posterior
            else []
        ),
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
    _print_ces_summary(idata, data)
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


def _print_ces_summary(idata: az.InferenceData, data: dict | None = None) -> None:
    """Print posterior summary of CES observation parameters."""
    a = idata.posterior["alpha_ces"].values.flatten()
    l = idata.posterior["lambda_ces"].values.flatten()
    s_sa = idata.posterior["sigma_ces_sa"].values   # (chains, draws, n_ces_v)
    s_nsa = idata.posterior["sigma_ces_nsa"].values  # (chains, draws, n_ces_v)

    print("\nCES observation parameters (vs QCEW anchor):")
    print(f"  \u03b1_ces  = {a.mean() * 100:+.4f}%/mo  (bias relative to true growth)")
    print(
        f"  \u03bb_ces  = {l.mean():.4f}  "
        f"[{np.percentile(l, 10):.4f}, {np.percentile(l, 90):.4f}]"
    )
    _all_vintage_names = {0: "1st", 1: "2nd", 2: "Final"}
    vintage_map = data.get("ces_vintage_map") if data else None
    if vintage_map is None:
        vintage_map = {i: i for i in range(s_sa.shape[2])}
    inv_map = {i: v for v, i in vintage_map.items()}
    for i in range(s_sa.shape[2]):
        orig_v = inv_map.get(i, i)
        label = _all_vintage_names.get(orig_v, f"v{orig_v}")
        sa_v = s_sa[:, :, i].flatten()
        nsa_v = s_nsa[:, :, i].flatten()
        print(
            f"  \u03c3_ces_sa[{label}] = {sa_v.mean() * 100:.3f}%  "
            f"\u03c3_ces_nsa[{label}] = {nsa_v.mean() * 100:.3f}%"
        )
    print("  (best-available print per month; σ selected by vintage index)")


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


def _qcew_precision_by_tier(
    idata: az.InferenceData, data: dict
) -> tuple[float, float, int, int]:
    """Total QCEW precision and tier counts from posterior sigmas and multipliers."""
    sigma_mid = float(idata.posterior["sigma_qcew_mid"].values.flatten().mean())
    sigma_boundary = float(
        idata.posterior["sigma_qcew_boundary"].values.flatten().mean()
    )
    qcew_is_m2 = np.asarray(data["qcew_is_m2"])
    qcew_noise_mult = np.asarray(data["qcew_noise_mult"], dtype=float)
    n_obs = len(qcew_is_m2)
    sigma_per_obs = np.where(
        qcew_is_m2,
        sigma_mid * qcew_noise_mult,
        sigma_boundary * qcew_noise_mult,
    )
    # Student-t Fisher information: (nu+1)/((nu+3)*sigma^2) vs Normal 1/sigma^2
    studentt_factor = (QCEW_NU + 1) / (QCEW_NU + 3)
    prec_per_obs = studentt_factor / (sigma_per_obs**2)
    total_prec_qcew = float(np.sum(prec_per_obs))
    n_m2 = int(qcew_is_m2.sum())
    n_boundary = n_obs - n_m2
    prec_m2 = float(np.sum(prec_per_obs[qcew_is_m2])) if n_m2 > 0 else 0.0
    prec_boundary = (
        float(np.sum(prec_per_obs[~qcew_is_m2])) if n_boundary > 0 else 0.0
    )
    return total_prec_qcew, prec_m2, prec_boundary, n_m2, n_boundary


def _ces_precision_rows(
    idata: az.InferenceData, data: dict
) -> tuple[list[tuple[str, int, float, float]], float]:
    """CES precision by SA/NSA using per-obs vintage-indexed sigma."""
    sigma_ces_sa = idata.posterior["sigma_ces_sa"].values.mean(axis=(0, 1))  # (n_ces_v,)
    sigma_ces_nsa = idata.posterior["sigma_ces_nsa"].values.mean(axis=(0, 1))
    lambda_ces = float(idata.posterior["lambda_ces"].values.flatten().mean())

    rows: list[tuple[str, int, float, float]] = []
    total = 0.0
    for label, obs_key, vidx_key, sigma_arr in [
        ("CES SA", "ces_sa_obs", "ces_sa_vintage_idx", sigma_ces_sa),
        ("CES NSA", "ces_nsa_obs", "ces_nsa_vintage_idx", sigma_ces_nsa),
    ]:
        obs = data[obs_key]
        vidx = data[vidx_key]
        n = len(obs)
        if n == 0:
            continue
        sigma_per_obs = sigma_arr[vidx]
        prec_per_obs = lambda_ces**2 / sigma_per_obs**2
        total_prec = float(np.sum(prec_per_obs))
        avg_prec = total_prec / n
        rows.append((label, n, avg_prec, total_prec))
        total += total_prec
    return rows, total


def print_source_contributions(idata: az.InferenceData, data: dict) -> None:
    """Quantify each source's precision-weighted information contribution."""
    pp_data = data["pp_data"]

    (
        total_prec_qcew,
        prec_qcew_m2,
        prec_qcew_boundary,
        n_qcew_m2,
        n_qcew_boundary,
    ) = _qcew_precision_by_tier(idata, data)
    prec_per_m2 = prec_qcew_m2 / n_qcew_m2 if n_qcew_m2 > 0 else 0.0
    prec_per_boundary = (
        prec_qcew_boundary / n_qcew_boundary if n_qcew_boundary > 0 else 0.0
    )

    ces_rows, total_prec_ces = _ces_precision_rows(idata, data)

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
    if n_qcew_m2 > 0:
        _row("QCEW (M2)", n_qcew_m2, prec_per_m2, prec_qcew_m2)
    if n_qcew_boundary > 0:
        _row("QCEW (M3+M1)", n_qcew_boundary, prec_per_boundary, prec_qcew_boundary)
    print("-" * 72)
    print(f"{'TOTAL':<16} {'':8} {'':12} {total_all:>14,.0f} {'100.0%':>8}")


def compute_precision_budget(idata: az.InferenceData, data: dict) -> pl.DataFrame:
    """Compute precision-weighted information budget as a structured DataFrame.

    Returns a DataFrame with columns:
        source, n_obs, precision_per_obs, total_precision, share,
        lambda_mean, alpha_mean
    """
    pp_data = data["pp_data"]
    lambda_ces = float(idata.posterior["lambda_ces"].values.flatten().mean())
    alpha_ces = float(idata.posterior["alpha_ces"].values.flatten().mean())

    (
        _,
        prec_qcew_m2,
        prec_qcew_boundary,
        n_qcew_m2,
        n_qcew_boundary,
    ) = _qcew_precision_by_tier(idata, data)
    prec_per_m2 = prec_qcew_m2 / n_qcew_m2 if n_qcew_m2 > 0 else 0.0
    prec_per_boundary = (
        prec_qcew_boundary / n_qcew_boundary if n_qcew_boundary > 0 else 0.0
    )

    rows: list[dict] = []

    ces_prec_rows, _ = _ces_precision_rows(idata, data)
    for label, n, avg_prec, total_prec in ces_prec_rows:
        rows.append({
            "source": label,
            "n_obs": n,
            "precision_per_obs": avg_prec,
            "total_precision": total_prec,
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
    if n_qcew_m2 > 0:
        rows.append({
            "source": "QCEW (M2)",
            "n_obs": n_qcew_m2,
            "precision_per_obs": prec_per_m2,
            "total_precision": prec_qcew_m2,
            "lambda_mean": 1.0,
            "alpha_mean": 0.0,
        })
    if n_qcew_boundary > 0:
        rows.append({
            "source": "QCEW (M3+M1)",
            "n_obs": n_qcew_boundary,
            "precision_per_obs": prec_per_boundary,
            "total_precision": prec_qcew_boundary,
            "lambda_mean": 1.0,
            "alpha_mean": 0.0,
        })

    df = pl.DataFrame(rows)
    total = df["total_precision"].sum()
    df = df.with_columns((pl.col("total_precision") / total).alias("share"))
    return df


# =========================================================================
# Era-specific (windowed) precision budget
# =========================================================================

_ERA_LABELS_DIAG = [
    "Pre-COVID  (2012-01 → 2019-12)",
    "Post-COVID (2020-01 → present)",
]


def print_windowed_precision_budget(idata: az.InferenceData, data: dict) -> None:
    """Print precision-weighted information budget restricted to each era window.

    Shows per-era observation counts and precision shares so that sources
    (e.g. the G provider) that only exist in one era can be compared fairly.
    """
    pp_data = data["pp_data"]
    era_idx = data["era_idx"]  # (T,)
    dates = data["dates"]

    sigma_ces_sa = idata.posterior["sigma_ces_sa"].values.mean(axis=(0, 1))  # (n_ces_v,)
    sigma_ces_nsa = idata.posterior["sigma_ces_nsa"].values.mean(axis=(0, 1))
    lambda_ces = float(idata.posterior["lambda_ces"].values.flatten().mean())

    sigma_mid = float(idata.posterior["sigma_qcew_mid"].values.flatten().mean())
    sigma_boundary = float(
        idata.posterior["sigma_qcew_boundary"].values.flatten().mean()
    )
    qcew_obs = data["qcew_obs"]
    qcew_is_m2 = np.asarray(data["qcew_is_m2"])
    qcew_noise_mult = np.asarray(data["qcew_noise_mult"], dtype=float)

    n_eras = int(era_idx.max()) + 1 if len(era_idx) > 0 else 0

    for e in range(n_eras):
        era_t_idx = np.where(era_idx == e)[0]
        n_months = len(era_t_idx)
        if n_months == 0:
            continue

        start_d = dates[era_t_idx[0]]
        end_d = dates[era_t_idx[-1]]
        label = _ERA_LABELS_DIAG[e] if e < len(_ERA_LABELS_DIAG) else f"Era {e}"

        # CES precision in this era (per-obs vintage-indexed sigma)
        ces_rows: list[tuple[str, int, float, float]] = []
        for src_label, obs_key, vidx_key, sigma_arr in [
            ("CES SA", "ces_sa_obs", "ces_sa_vintage_idx", sigma_ces_sa),
            ("CES NSA", "ces_nsa_obs", "ces_nsa_vintage_idx", sigma_ces_nsa),
        ]:
            obs = data[obs_key]
            vidx = data[vidx_key]
            in_era_mask = np.isin(obs, era_t_idx)
            era_obs = obs[in_era_mask]
            era_vidx = vidx[in_era_mask]
            n = len(era_obs)
            if n == 0:
                continue
            sigma_per = sigma_arr[era_vidx]
            prec_per = lambda_ces**2 / sigma_per**2
            total_prec = float(np.sum(prec_per))
            avg_prec = total_prec / n
            ces_rows.append((src_label, n, avg_prec, total_prec))

        # Provider counts in this era
        pp_rows: list[tuple[str, int, float, float]] = []
        for pp in pp_data:
            name = pp["config"].name.lower()
            sig_p = float(idata.posterior[f"sigma_pp_{name}"].values.flatten().mean())
            lam_p = float(idata.posterior[f"lam_{name}"].values.flatten().mean())
            covered = np.intersect1d(pp["pp_obs"], era_t_idx)
            n_obs = len(covered)
            prec_p = lam_p**2 / sig_p**2
            if pp["config"].error_model == "ar1":
                rho_p = float(idata.posterior[f"rho_{name}"].values.flatten().mean())
                prec_p = lam_p**2 * (1 - rho_p**2) / sig_p**2
            pp_rows.append((pp["name"], n_obs, prec_p, prec_p * n_obs))

        # QCEW: per-obs sigma = base_tier * revision_mult
        in_era = np.isin(qcew_obs, era_t_idx)
        sigma_per = np.where(
            qcew_is_m2,
            sigma_mid * qcew_noise_mult,
            sigma_boundary * qcew_noise_mult,
        )
        studentt_factor = (QCEW_NU + 1) / (QCEW_NU + 3)
        prec_per = studentt_factor / (sigma_per**2)
        total_qcew = float(np.sum(prec_per[in_era]))
        n_qcew_m2 = int((in_era & qcew_is_m2).sum())
        n_qcew_boundary = int((in_era & ~qcew_is_m2).sum())
        prec_qcew_m2_era = float(np.sum(prec_per[in_era & qcew_is_m2]))
        prec_qcew_boundary_era = float(np.sum(prec_per[in_era & ~qcew_is_m2]))
        prec_per_m2 = prec_qcew_m2_era / n_qcew_m2 if n_qcew_m2 > 0 else 0.0
        prec_per_boundary = (
            prec_qcew_boundary_era / n_qcew_boundary
            if n_qcew_boundary > 0
            else 0.0
        )

        total_ces = sum(r[3] for r in ces_rows)
        total_pp = sum(r[3] for r in pp_rows)
        total_all = total_ces + total_pp + total_qcew
        if total_all == 0:
            continue

        print("\n" + "=" * 72)
        print("ERA-SPECIFIC PRECISION BUDGET")
        print("=" * 72)
        print(f"{label}: {start_d} \u2192 {end_d}  ({n_months} months)")
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
        if n_qcew_m2 > 0:
            _row("QCEW (M2)", n_qcew_m2, prec_per_m2, prec_qcew_m2_era)
        if n_qcew_boundary > 0:
            _row(
                "QCEW (M3+M1)",
                n_qcew_boundary,
                prec_per_boundary,
                prec_qcew_boundary_era,
            )
        print("-" * 72)
        print(f"{'TOTAL':<16} {'':8} {'':12} {total_all:>14,.0f} {'100.0%':>8}")


# =========================================================================
# Provider value-of-information (posterior g_cont width)
# =========================================================================


def print_provider_value_of_information(idata: az.InferenceData, data: dict) -> None:
    """Compare posterior g_cont uncertainty at provider-covered vs uncovered time steps.

    Within each era, reports mean 80% HDI width for g_cont at months where
    the provider has data vs months in the same era where it does not, to
    show whether the provider meaningfully tightens the posterior.
    """
    pp_data = data["pp_data"]
    era_idx = data["era_idx"]
    T = era_idx.shape[0]

    g_cont = idata.posterior["g_cont"].values  # (chains, draws, T)
    hdi_lo = np.percentile(g_cont, 10, axis=(0, 1))
    hdi_hi = np.percentile(g_cont, 90, axis=(0, 1))
    hdi_width = hdi_hi - hdi_lo  # (T,) in decimal; *100 for %

    print("\n" + "=" * 72)
    print("PROVIDER VALUE-OF-INFORMATION (posterior g_cont 80% HDI width)")
    print("=" * 72)

    for pp in pp_data:
        name = pp["config"].name.lower()
        covered = np.asarray(pp["pp_obs"])
        if len(covered) == 0:
            print(f"\n  {pp['name']}: no observations \u2014 skipping")
            continue

        # Era(s) where this provider has data
        eras_with_data = np.unique(era_idx[covered])
        for e in eras_with_data:
            era_t_idx = np.where(era_idx == e)[0]
            in_era_covered = np.intersect1d(covered, era_t_idx)
            in_era_uncovered = np.setdiff1d(era_t_idx, covered)
            if len(in_era_covered) == 0:
                continue

            label = _ERA_LABELS_DIAG[e] if e < len(_ERA_LABELS_DIAG) else f"Era {e}"
            width_with = np.mean(hdi_width[in_era_covered]) * 100
            n_with = len(in_era_covered)

            if len(in_era_uncovered) > 0:
                width_without = np.mean(hdi_width[in_era_uncovered]) * 100
                n_without = len(in_era_uncovered)
                pct_narrower = (1.0 - width_with / width_without) * 100.0
                print(
                    f"\n  {pp['name']} \u2014 {label}:"
                )
                print(
                    f"    With {pp['name']} data:    {width_with:.4f}%/mo mean HDI width (n={n_with})"
                )
                print(
                    f"    Without {pp['name']} data: {width_without:.4f}%/mo mean HDI width (n={n_without})"
                )
                print(f"    Narrowing:               {pct_narrower:+.1f}%")
            else:
                print(f"\n  {pp['name']} \u2014 {label}:")
                print(
                    f"    With {pp['name']} data:    {width_with:.4f}%/mo mean HDI width (n={n_with})"
                )
                print("    (all months in this era have provider data \u2014 no uncovered baseline)")


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

    phi_name = "phi_raw" if "phi_raw" in idata.posterior else "phi_raw_era"
    pairs: list[tuple[str, str, str, str]] = [
        (phi_name, "tau", "\u03c6", "\u03c4"),
        ("lambda_ces", "alpha_ces", "\u03bb_CES", "\u03b1_CES"),
    ]
    for pp in data["pp_data"]:
        if pp["config"].error_model == "ar1":
            n = pp["config"].name.lower()
            pairs.append(
                (f"rho_{n}", f"sigma_pp_{n}", f"\u03c1_{pp['name']}", f"\u03c3_{pp['name']}")
            )

    pairs = [(p1, p2, l1, l2) for p1, p2, l1, l2 in pairs
             if p1 in idata.posterior and p2 in idata.posterior]
    if not pairs:
        print("  No plottable parameter pairs found \u2014 skipping.")
        return

    n_pairs = len(pairs)
    n_cols = min(2, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

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


# =========================================================================
# Weight staleness reporting for QCEW-weighted compositing (§6.3)
# =========================================================================


def print_weight_staleness(providers: list) -> None:
    """Report QCEW weight staleness for cell-level providers.

    For each provider whose parquet contains cell-level data
    (``geographic_type='region'``), compute the QCEW-weighted composite
    staleness metadata and print a summary.

    Parameters
    ----------
    providers
        List of :class:`~alt_nfp.config.ProviderConfig` instances.
    """
    from .config import DATA_DIR, MIN_PSEUDO_ESTABS_PER_CELL, STORE_DIR
    from .ingest.compositing import compute_provider_composite
    from .ingest.payroll import _is_cell_level, read_provider_table

    found_any = False
    for cfg in providers:
        fpath = DATA_DIR / cfg.file
        raw = read_provider_table(fpath)
        if raw is None or not _is_cell_level(raw):
            continue

        _, staleness = compute_provider_composite(
            raw, STORE_DIR, MIN_PSEUDO_ESTABS_PER_CELL,
        )
        if staleness.is_empty():
            continue

        found_any = True
        s = staleness["weight_staleness_months"]
        n_current = int((s == 0).sum())
        n_forward = int((s > 0).sum())

        print(f"\n  {cfg.name}:")
        print(f"    Months with current QCEW weights:       {n_current}")
        print(f"    Months with carried-forward weights:     {n_forward}")
        if n_forward > 0:
            fwd = staleness.filter(pl.col("weight_staleness_months") > 0)
            fwd_s = fwd["weight_staleness_months"]
            print(
                f"    Staleness (carried-forward months):      "
                f"min={fwd_s.min()}, max={fwd_s.max()}, "
                f"mean={fwd_s.mean():.1f}"
            )

    if not found_any:
        print("  (no cell-level providers detected)")
