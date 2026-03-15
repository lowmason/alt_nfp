# =============================================================================
# SECTION 6: PRIOR PREDICTIVE CHECKS
# =============================================================================
#
# Before fitting the model, we sample from the prior predictive distribution
# to verify that our priors generate data in a plausible range.  If the prior
# predictive is wildly different from the observed data, the priors need
# tightening.  If it's indistinguishable from the observed data, the priors
# are too informative (the data won't update them much).
#
# Good prior predictive: covers a range 2-10x wider than the observed data.
# (Gabry et al. 2019, "Visualization in Bayesian Workflow")
#
# =============================================================================


def _build_obs_sources(data: dict) -> dict:
    """Build {var_name: (label, observed_array)} for predictive checks."""
    sources = {}
    if len(data["ces_sa_obs"]) > 0:
        sources["obs_ces_sa"] = ("CES SA", data["g_ces_sa"][data["ces_sa_obs"]])
    if len(data["ces_nsa_obs"]) > 0:
        sources["obs_ces_nsa"] = ("CES NSA", data["g_ces_nsa"][data["ces_nsa_obs"]])
    sources["obs_qcew"] = ("QCEW", data["g_qcew"][data["qcew_obs"]])
    if len(data["provider_obs"]) > 0:
        sources["obs_g"] = ("Provider G", data["g_provider"][data["provider_obs"]])
    return sources


def run_prior_predictive(model: pm.Model, data: dict) -> az.InferenceData | None:
    """Sample from the prior predictive and visualise."""
    print("Sampling prior predictive...")
    try:
        with model:
            prior_idata = pm.sample_prior_predictive(samples=500)
    except Exception as e:
        print(f"Prior predictive sampling failed: {e}")
        return None

    dates = data["dates"]
    obs_sources = _build_obs_sources(data)
    n_sources = len(obs_sources)

    n_cols = 3
    n_rows = (n_sources + 1 + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes_flat = axes.flatten()

    for idx, (vn, (label, obs)) in enumerate(obs_sources.items()):
        ax = axes_flat[idx]
        pp = prior_idata.prior_predictive[vn].values.flatten()
        lo, hi = np.percentile(pp, [1, 99])
        pp_clip = pp[(pp >= lo) & (pp <= hi)]
        ax.hist(
            pp_clip * 100, bins=80, density=True, alpha=0.4,
            color="steelblue", label="Prior predictive",
        )
        ax.hist(
            obs * 100, bins=40, density=True, alpha=0.6,
            color="darkorange", label="Observed",
        )
        ax.set_xlabel("Monthly growth (%)")
        ax.set_title(label)
        ax.legend(fontsize=7)

    # Prior g_cont trajectories
    ax = axes_flat[n_sources]
    g_prior = prior_idata.prior["g_cont"].values
    n_show = min(20, g_prior.shape[1])
    for i in range(n_show):
        ax.plot(
            dates, g_prior[0, i, :] * 100,
            alpha=0.3, lw=0.8, color="steelblue",
        )
    ax.plot(
        dates, data["g_ces_sa"] * 100,
        "darkorange", lw=1.5, alpha=0.8, label="CES SA (observed)",
    )
    ax.set_ylabel("Growth (%/mo)")
    ax.set_title("Prior draws: latent g_cont")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for idx in range(n_sources + 1, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Prior Predictive Check: Do priors generate plausible data?",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "prior_predictive.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'prior_predictive.png'}")

    print("\nPrior predictive summary (growth %, monthly):")
    for vn, (label, obs) in obs_sources.items():
        pp = prior_idata.prior_predictive[vn].values.flatten() * 100
        lo5, hi95 = np.percentile(pp, [5, 95])
        print(
            f"  {label:10}: prior 90% [{lo5:+.2f}, {hi95:+.2f}]  "
            f"| obs mean: {obs.mean() * 100:+.4f}%"
        )

    return prior_idata


# =============================================================================
# SECTION 7: POSTERIOR PREDICTIVE CHECKS
# =============================================================================
#
# After fitting, we draw replicated data from the posterior predictive and
# compare to the observed data.  Two diagnostics:
#
# 1. DENSITY OVERLAYS — 100 replicated datasets (thin blue) vs observed (black).
#    If the model is well-specified, the observed should sit within the
#    replicated cloud.
#
# 2. TEST STATISTICS — skewness and lag-1 autocorrelation computed on each
#    replicated dataset.  The observed statistic (black line) should be
#    plausible under the replicated distribution.  P-values near 0 or 1
#    indicate systematic misfit.  We use statistics ORTHOGONAL to model
#    parameters: skewness tests tail behaviour (model assumes symmetric
#    errors), lag-1 ACF tests serial correlation (model assumes iid errors
#    conditional on the latent state).
#
# =============================================================================


def run_posterior_predictive(
    model: pm.Model, idata: az.InferenceData, data: dict,
) -> None:
    """Density overlays and test-statistics comparing replicated to observed."""
    print("Sampling posterior predictive...")
    with model:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=42)

    obs_sources = _build_obs_sources(data)
    rng = np.random.default_rng(42)

    # ---- Density overlays ----
    n_src = len(obs_sources)
    n_cols = 3
    n_rows = (n_src + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes_flat = axes.flatten()

    for idx, (vn, (label, obs)) in enumerate(obs_sources.items()):
        ax = axes_flat[idx]
        pp = idata.posterior_predictive[vn].values
        n_ch, n_dr, n_obs = pp.shape
        pp_flat = pp.reshape(-1, n_obs)
        n_total = pp_flat.shape[0]
        sub_idx = rng.choice(n_total, size=min(100, n_total), replace=False)

        obs_pct = obs * 100
        pad = 0.5 * np.ptp(obs_pct) if np.ptp(obs_pct) > 0 else 0.5
        bins = np.linspace(obs_pct.min() - pad, obs_pct.max() + pad, 60)

        for i in sub_idx:
            ax.hist(
                pp_flat[i] * 100, bins=bins, density=True,
                histtype="step", alpha=0.08, color="steelblue", lw=0.5,
            )
        ax.hist(
            obs_pct, bins=bins, density=True,
            histtype="step", color="black", lw=2, label="Observed",
        )
        ax.set_xlabel("Monthly growth (%)")
        ax.set_title(label)
        ax.legend(fontsize=7)

    for idx in range(n_src, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Posterior Predictive Check: Density Overlays",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "ppc_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'ppc_density.png'}")

    # ---- Test statistics ----
    def _lag1_acf(x):
        if len(x) < 3:
            return 0.0
        xc = x - x.mean()
        c0 = np.dot(xc, xc)
        return float(np.dot(xc[:-1], xc[1:]) / c0) if c0 > 0 else 0.0

    stat_fns = {"Skewness": lambda x: sp_stats.skew(x), "Lag-1 ACF": _lag1_acf}
    n_stat = len(stat_fns)
    fig, axes = plt.subplots(n_src, n_stat, figsize=(5 * n_stat, 2.5 * n_src))

    for row, (vn, (label, obs)) in enumerate(obs_sources.items()):
        pp = idata.posterior_predictive[vn].values
        pp_flat = pp.reshape(-1, pp.shape[-1])
        n_sub = min(500, pp_flat.shape[0])
        sub_idx = rng.choice(pp_flat.shape[0], size=n_sub, replace=False)

        for col, (sname, sfn) in enumerate(stat_fns.items()):
            ax = axes[row, col] if n_src > 1 else axes[col]
            rep_vals = np.array([sfn(pp_flat[i]) for i in sub_idx])
            obs_val = sfn(obs)
            ax.hist(rep_vals, bins=50, density=True, alpha=0.5, color="steelblue")
            ax.axvline(
                obs_val, color="black", lw=2, ls="--",
                label=f"Obs: {obs_val:.3f}",
            )
            p = np.mean(rep_vals >= obs_val)
            p = min(p, 1 - p)
            ax.set_title(f"{label}: {sname} (p={p:.3f})", fontsize=9)
            ax.legend(fontsize=6)

    fig.suptitle(
        "Posterior Predictive Test Statistics",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "ppc_test_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'ppc_test_stats.png'}")


# =============================================================================
# SECTION 8: LOO-CV (Leave-One-Out Cross-Validation)
# =============================================================================
#
# IMPORTANT: In a state-space model, LOO-CV does NOT measure forecast skill.
# Observations are temporally connected through the latent state, so LOO
# measures interpolation consistency (how well the model can reconstruct
# each observation from its neighbours), not prediction.
#
# LOO-CV is valuable here as a DATA QUALITY AUDIT:
#   - High k-hat (> 0.7): observation has outsized influence on the posterior.
#     For QCEW this is STRUCTURAL (high-precision obs dominate the latent state)
#     and does NOT indicate a bad model.
#   - Low pointwise ELPD: observation is poorly predicted even with all other
#     data — potential data anomaly or model misspecification at that point.
#
# For actual model evaluation, use the vintage-aware backtest (backtest.py).
#
# =============================================================================


def run_loo_cv(model: pm.Model, idata: az.InferenceData, data: dict) -> None:
    """LOO-CV per source with k-hat diagnostics and outlier tables."""
    if not hasattr(idata, "log_likelihood"):
        print("Computing log-likelihood for LOO-CV...")
        try:
            with model:
                idata.extend(pm.compute_log_likelihood(idata))
        except Exception as e:
            print(f"Could not compute log-likelihood: {e}")
            return

    obs_sources = _build_obs_sources(data)
    source_list = list(obs_sources.items())

    # Build time-index map for resolving observation dates
    idx_map = {}
    if len(data["ces_sa_obs"]) > 0:
        idx_map["obs_ces_sa"] = data["ces_sa_obs"]
    if len(data["ces_nsa_obs"]) > 0:
        idx_map["obs_ces_nsa"] = data["ces_nsa_obs"]
    idx_map["obs_qcew"] = data["qcew_obs"]
    if len(data["provider_obs"]) > 0:
        idx_map["obs_g"] = data["provider_obs"]

    all_dates = data["dates"]

    print("\n" + "=" * 72)
    print("LEAVE-ONE-OUT CROSS-VALIDATION (PSIS-LOO)")
    print("  Note: measures interpolation consistency, not forecast skill.")
    print("=" * 72)

    n_src = len(source_list)
    n_cols = 3
    n_rows = (n_src + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes_flat = axes.flatten()

    for idx, (var_name, (label, obs_vals)) in enumerate(source_list):
        ax = axes_flat[idx]
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Estimated shape parameter"
                )
                loo_result = az.loo(idata, var_name=var_name, pointwise=True)

            elpd = loo_result.elpd_loo
            se = loo_result.se
            p_loo = loo_result.p_loo
            khat = np.asarray(loo_result.pareto_k)
            elpd_i = np.asarray(loo_result.loo_i)

            n_high = int(np.sum(khat > 0.7))
            n_warn = int(np.sum((khat > 0.5) & (khat <= 0.7)))

            print(f"\n{label} ({var_name}):")
            print(f"  ELPD LOO: {elpd:.1f} +/- {se:.1f}")
            print(f"  p_loo:    {p_loo:.1f}")
            print(f"  k-hat > 0.7 (bad):  {n_high}")
            print(f"  k-hat > 0.5 (warn): {n_warn}")

            # Outlier table
            elpd_mean = float(np.mean(elpd_i))
            elpd_std = float(np.std(elpd_i))
            threshold = elpd_mean - 2.0 * elpd_std
            flagged = np.where((khat > 0.5) | (elpd_i < threshold))[0]
            time_idx = idx_map.get(var_name)

            if len(flagged) > 0:
                order = np.argsort(elpd_i[flagged])
                flagged = flagged[order][:10]
                print(f"  {'Date':>12}  {'k-hat':>7}  {'ELPD_i':>8}  {'g_obs':>10}  Flag")
                for j in flagged:
                    dt_str = (
                        str(all_dates[time_idx[j]]) if time_idx is not None
                        else f"idx {j}"
                    )
                    flag = (
                        "BAD" if khat[j] > 0.7
                        else ("WARN" if khat[j] > 0.5 else "ELPD")
                    )
                    print(
                        f"  {dt_str:>12}  {khat[j]:7.3f}  {elpd_i[j]:+8.1f}"
                        f"  {obs_vals[j]:+10.5f}  {flag}"
                    )

            # k-hat plot
            colors = np.where(
                khat > 0.7, "red",
                np.where(khat > 0.5, "orange", "steelblue"),
            )
            ax.scatter(range(len(khat)), khat, s=8, c=colors, alpha=0.6)
            ax.axhline(0.7, color="red", ls="--", lw=1, alpha=0.7, label="k-hat = 0.7")
            ax.axhline(0.5, color="orange", ls="--", lw=1, alpha=0.7, label="k-hat = 0.5")
            ax.set_xlabel("Observation index")
            ax.set_ylabel("k-hat")
            ax.set_title(f"{label}: PSIS-LOO k-hat")
            ax.legend(fontsize=7)

        except Exception as e:
            print(f"\n{label}: LOO-CV failed - {e}")
            ax.set_visible(False)

    for idx in range(n_src, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "LOO-CV k-hat Diagnostics (Pareto shape parameter)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "loo_khat.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_DIR / 'loo_khat.png'}")

