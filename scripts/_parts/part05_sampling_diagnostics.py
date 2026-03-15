# =============================================================================
# SECTION 4: MCMC SAMPLING
# =============================================================================
#
# Uses the nutpie sampler (fast Rust NUTS implementation) when available,
# falling back to PyMC's built-in NUTS.  nutpie is particularly fast on
# Apple Silicon.
#
# NUTS (No-U-Turn Sampler) is a self-tuning variant of Hamiltonian Monte
# Carlo (HMC).  It automatically selects the trajectory length, avoiding
# the manual tuning required by basic HMC.  The key tuning parameter is
# target_accept (0.95 here): higher values reduce divergences but slow
# down sampling.
#
# =============================================================================


def sample_model(model: pm.Model) -> az.InferenceData:
    """Sample the model using nutpie (preferred) or PyMC NUTS."""
    with model:
        warnings.filterwarnings(
            "ignore",
            message="Numba will use object mode",
            category=UserWarning,
            module=r"pytensor\.link\.numba",
        )
        try:
            idata = pm.sample(nuts_sampler="nutpie", **SAMPLER_KWARGS)
            sampler_used = "nutpie"
        except Exception as e:
            print(f"nutpie unavailable ({e}), falling back to PyMC NUTS")
            idata = pm.sample(**SAMPLER_KWARGS)
            sampler_used = "pymc"

    print(f"\nSampling complete ({sampler_used})")
    return idata


# =============================================================================
# SECTION 5: CONVERGENCE DIAGNOSTICS
# =============================================================================
#
# After sampling, we check:
#   1. Divergent transitions — indicate geometric pathology (funnels, ridges)
#   2. R-hat — should be < 1.01 for all parameters (chains have mixed)
#   3. ESS (effective sample size) — should be > 400 (enough independent samples)
#   4. Parameter summaries — posterior means, credible intervals
#   5. Precision budget — how much information each data source contributes
#
# =============================================================================


def print_diagnostics(idata: az.InferenceData, data: dict) -> None:
    """Print sampling diagnostics, parameter summary, and precision budget."""
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

    # Build variable list for summary
    var_names = [
        "mu_g_era", "phi_raw", "tau",
        "sigma_fourier",
        "phi_0", "sigma_bd",
        "alpha_ces", "lambda_ces", "sigma_ces_sa", "sigma_ces_nsa",
        "sigma_qcew_mid", "sigma_qcew_boundary",
    ]
    if "phi_3" in idata.posterior:
        var_names.append("phi_3")
    if len(data["provider_obs"]) > 0:
        var_names.extend(["alpha_g", "lam_g", "sigma_pp_g"])

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

    _print_key_outputs(idata, data)
    _print_precision_budget(idata, data)
    _print_era_summary(idata)


def _print_key_outputs(idata: az.InferenceData, data: dict) -> None:
    """Print posterior summaries of BD, CES, and provider parameters."""
    print("\n" + "=" * 72)
    print("KEY OUTPUTS")
    print("=" * 72)

    # Birth/death
    phi_0 = idata.posterior["phi_0"].values.flatten()
    sigma_bd = idata.posterior["sigma_bd"].values.flatten()
    bd_mean = idata.posterior["bd"].values.mean(axis=(0, 1))

    print("\nStructural Birth/Death:")
    print(
        f"  phi_0 (intercept):    {phi_0.mean() * 100:+.4f}%/mo  "
        f"[{np.percentile(phi_0, 10) * 100:+.4f}%, "
        f"{np.percentile(phi_0, 90) * 100:+.4f}%]"
    )
    print(f"  sigma_bd (innovation): {sigma_bd.mean() * 100:.4f}%")

    if "phi_3" in idata.posterior:
        phi_3 = idata.posterior["phi_3"].values
        cyc_labels = ["claims", "jolts"]
        n_cyc = phi_3.shape[-1]
        for i in range(n_cyc):
            v = phi_3[:, :, i].flatten()
            lbl = cyc_labels[i] if i < len(cyc_labels) else f"cyc_{i}"
            print(
                f"  phi_3[{lbl}]:  {v.mean():.3f}  "
                f"[{np.percentile(v, 10):.3f}, {np.percentile(v, 90):.3f}]"
            )

    print(
        f"  bd_t mean over sample: {bd_mean.mean() * 100:+.4f}%/mo "
        f"(annualised: {bd_mean.mean() * 12 * 100:+.2f}%)"
    )

    # CES
    a = idata.posterior["alpha_ces"].values.flatten()
    lam = idata.posterior["lambda_ces"].values.flatten()
    s_sa = idata.posterior["sigma_ces_sa"].values
    s_nsa = idata.posterior["sigma_ces_nsa"].values

    print("\nCES observation parameters:")
    print(f"  alpha_ces = {a.mean() * 100:+.4f}%/mo (bias vs latent truth)")
    print(
        f"  lambda_ces = {lam.mean():.4f}  "
        f"[{np.percentile(lam, 10):.4f}, {np.percentile(lam, 90):.4f}]"
    )

    vintage_names = {0: "1st", 1: "2nd", 2: "Final"}
    inv_map = {i: v for v, i in data["ces_vintage_map"].items()}
    for i in range(s_sa.shape[2]):
        orig_v = inv_map.get(i, i)
        label = vintage_names.get(orig_v, f"v{orig_v}")
        sa_v = s_sa[:, :, i].flatten()
        nsa_v = s_nsa[:, :, i].flatten()
        print(
            f"  sigma_ces_sa[{label}] = {sa_v.mean() * 100:.3f}%  "
            f"sigma_ces_nsa[{label}] = {nsa_v.mean() * 100:.3f}%"
        )

    # Provider
    if len(data["provider_obs"]) > 0:
        lam_g = idata.posterior["lam_g"].values.flatten()
        alp_g = idata.posterior["alpha_g"].values.flatten()
        sig_g = idata.posterior["sigma_pp_g"].values.flatten()
        print("\nProvider G:")
        print(
            f"  lam_g = {lam_g.mean():.3f}  "
            f"[{np.percentile(lam_g, 10):.3f}, {np.percentile(lam_g, 90):.3f}]"
        )
        print(f"  alpha_g = {alp_g.mean() * 100:+.4f}%/mo")
        print(f"  sigma_pp_g = {sig_g.mean() * 100:.3f}%")


def _print_precision_budget(idata: az.InferenceData, data: dict) -> None:
    """Precision-weighted information budget: how much each source contributes.

    Precision = (loading^2) / (sigma^2) per observation.
    Total precision = sum over all observations of that source.
    Share = source_precision / total_precision.

    For QCEW (Student-t), precision is reduced by factor (nu+1)/(nu+3)
    compared to Normal (heavier tails = less information per observation).
    """
    print("\n" + "=" * 72)
    print("DATA SOURCE CONTRIBUTION (precision-weighted)")
    print("=" * 72)

    lambda_ces = float(idata.posterior["lambda_ces"].values.flatten().mean())
    sigma_ces_sa = idata.posterior["sigma_ces_sa"].values.mean(axis=(0, 1))
    sigma_ces_nsa = idata.posterior["sigma_ces_nsa"].values.mean(axis=(0, 1))

    rows = []

    # CES precision
    for label, obs_key, vidx_key, sigma_arr in [
        ("CES SA", "ces_sa_obs", "ces_sa_vintage_idx", sigma_ces_sa),
        ("CES NSA", "ces_nsa_obs", "ces_nsa_vintage_idx", sigma_ces_nsa),
    ]:
        obs = data[obs_key]
        vidx = data[vidx_key]
        n = len(obs)
        if n == 0:
            continue
        sigma_per = sigma_arr[vidx]
        prec_per = lambda_ces**2 / sigma_per**2
        total_prec = float(np.sum(prec_per))
        rows.append((label, n, total_prec / n, total_prec))

    # QCEW precision (Student-t Fisher information)
    sigma_mid = float(idata.posterior["sigma_qcew_mid"].values.flatten().mean())
    sigma_boundary = float(
        idata.posterior["sigma_qcew_boundary"].values.flatten().mean()
    )
    qcew_is_m2 = np.asarray(data["qcew_is_m2"])
    qcew_mult = np.asarray(data["qcew_noise_mult"], dtype=float)
    sigma_per_qcew = np.where(
        qcew_is_m2,
        sigma_mid * qcew_mult,
        sigma_boundary * qcew_mult,
    )
    studentt_factor = (QCEW_NU + 1) / (QCEW_NU + 3)
    prec_per_qcew = studentt_factor / (sigma_per_qcew**2)

    n_m2 = int(qcew_is_m2.sum())
    n_boundary = len(data["qcew_obs"]) - n_m2
    prec_m2 = float(np.sum(prec_per_qcew[qcew_is_m2])) if n_m2 > 0 else 0.0
    prec_boundary = float(np.sum(prec_per_qcew[~qcew_is_m2])) if n_boundary > 0 else 0.0
    if n_m2 > 0:
        rows.append(("QCEW (M2)", n_m2, prec_m2 / n_m2, prec_m2))
    if n_boundary > 0:
        rows.append(("QCEW (M3+M1)", n_boundary, prec_boundary / n_boundary, prec_boundary))

    # Provider precision
    if len(data["provider_obs"]) > 0:
        sig_p = float(idata.posterior["sigma_pp_g"].values.flatten().mean())
        lam_p = float(idata.posterior["lam_g"].values.flatten().mean())
        n_obs = len(data["provider_obs"])
        prec_p = lam_p**2 / sig_p**2
        rows.append(("Provider G", n_obs, prec_p, prec_p * n_obs))

    total_all = sum(r[3] for r in rows)
    hdr = f"{'Source':<20} {'Obs':<8} {'Prec/obs':>12} {'Total prec':>14} {'Share':>8}"
    print(hdr)
    print("-" * 72)
    for nm, n, avg_p, tot_p in rows:
        pct = 100.0 * tot_p / total_all if total_all > 0 else 0.0
        print(f"{nm:<20} {n:<8} {avg_p:>12,.0f} {tot_p:>14,.0f} {pct:>7.1f}%")
    print("-" * 72)
    print(f"{'TOTAL':<16} {'':8} {'':12} {total_all:>14,.0f} {'100.0%':>8}")


def _print_era_summary(idata: az.InferenceData) -> None:
    """Print posterior summaries of era-specific mu_g and shared phi."""
    if "mu_g_era" not in idata.posterior:
        return

    mu_g = idata.posterior["mu_g_era"].values
    n_eras = mu_g.shape[-1]
    era_labels = [
        "Pre-COVID  (2012-2019)",
        "Post-COVID (2020+)   ",
    ]

    print("\n" + "=" * 72)
    print("ERA-SPECIFIC LATENT STATE PARAMETERS")
    print("=" * 72)
    for e in range(n_eras):
        v = mu_g[:, :, e].flatten()
        label = era_labels[e] if e < len(era_labels) else f"Era {e}"
        lo, hi = np.percentile(v, [10, 90])
        print(
            f"  {label:<24} mu_g = {v.mean() * 100:+.4f}%  "
            f"80% HDI [{lo * 100:+.4f}%, {hi * 100:+.4f}%]"
        )

    phi = idata.posterior["phi_raw"].values.flatten()
    lo, hi = np.percentile(phi, [10, 90])
    print(
        f"\n  phi (shared):            {phi.mean():.4f}  "
        f"80% HDI [{lo:.4f}, {hi:.4f}]"
    )


def plot_divergences(idata: az.InferenceData, data: dict) -> None:
    """Scatter plots highlighting divergent transitions for pathology diagnosis."""
    diverging = idata.sample_stats.diverging.values
    n_divs = int(diverging.sum())
    if n_divs == 0:
        print("\nNo divergent transitions — skipping divergence plot.")
        return

    print(f"\nPlotting {n_divs} divergent transitions...")
    div_flat = diverging.flatten().astype(bool)

    pairs = [
        ("phi_raw", "tau", "phi", "tau"),
        ("lambda_ces", "alpha_ces", "lambda_CES", "alpha_CES"),
    ]
    pairs = [
        (p1, p2, l1, l2) for p1, p2, l1, l2 in pairs
        if p1 in idata.posterior and p2 in idata.posterior
    ]
    if not pairs:
        return

    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (p1, p2, l1, l2) in enumerate(pairs):
        ax = axes_flat[idx]
        v1 = idata.posterior[p1].values.flatten()
        v2 = idata.posterior[p2].values.flatten()
        ax.scatter(
            v1[~div_flat], v2[~div_flat],
            s=2, alpha=0.1, c="steelblue", rasterized=True,
        )
        ax.scatter(
            v1[div_flat], v2[div_flat],
            s=15, alpha=0.8, c="limegreen", edgecolors="darkgreen", lw=0.5,
            label=f"Divergent (n={n_divs})", zorder=10,
        )
        ax.set_xlabel(l1)
        ax.set_ylabel(l2)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Divergence Diagnostics: Bivariate Scatterplots",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "divergences.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'divergences.png'}")

