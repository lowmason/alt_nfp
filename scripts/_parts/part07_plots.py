# =============================================================================
# SECTION 9: RESULT PLOTS
# =============================================================================
#
# Four main visualisations:
#
# 1. GROWTH AND SEASONAL — latent growth rates vs observed data, plus
#    the estimated seasonal pattern compared to the empirical CES NSA-SA
#    difference.
#
# 2. RECONSTRUCTED INDEX — latent SA/NSA employment index overlaid on
#    CES, QCEW, and provider observed series.
#
# 3. BD DIAGNOSTICS — time series of the structural BD offset, its
#    covariate decomposition, and scatter vs the lagged QCEW BD proxy.
#
# 4. STANDARDISED RESIDUALS — per-source residuals that should be ~N(0,1)
#    for CES/provider and ~t(nu) for QCEW.  Temporal patterns signal misfit.
#
# =============================================================================


def _year_axis(ax) -> None:
    """Configure x-axis with yearly tick marks."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def plot_growth_and_seasonal(idata: az.InferenceData, data: dict) -> None:
    """Latent growth rates vs observed data and seasonal pattern."""
    dates = data["dates"]
    g_ces_sa = data["g_ces_sa"]
    g_ces_nsa = data["g_ces_nsa"]
    g_provider = data["g_provider"]
    provider_obs = data["provider_obs"]

    pct = lambda x: x * 100  # noqa: E731

    g_cont_post = idata.posterior["g_cont"].values
    g_sa_post = idata.posterior["g_total_sa"].values
    g_nsa_post = idata.posterior["g_total_nsa"].values

    g_cont_mean = g_cont_post.mean(axis=(0, 1))
    g_cont_lo = np.percentile(g_cont_post, 10, axis=(0, 1))
    g_cont_hi = np.percentile(g_cont_post, 90, axis=(0, 1))

    g_sa_mean = g_sa_post.mean(axis=(0, 1))
    g_sa_lo = np.percentile(g_sa_post, 10, axis=(0, 1))
    g_sa_hi = np.percentile(g_sa_post, 90, axis=(0, 1))

    g_nsa_mean = g_nsa_post.mean(axis=(0, 1))
    g_nsa_lo = np.percentile(g_nsa_post, 10, axis=(0, 1))
    g_nsa_hi = np.percentile(g_nsa_post, 90, axis=(0, 1))

    dates_arr = np.array(dates)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)

    # Panel 1: SA total growth
    ax = axes[0]
    ax.fill_between(dates, pct(g_sa_lo), pct(g_sa_hi), alpha=0.25, color="steelblue", label="80% CI")
    ax.plot(dates, pct(g_sa_mean), "steelblue", lw=1.5, label="Latent total (SA)")
    ax.scatter(dates, pct(g_ces_sa), s=10, c="darkorange", alpha=0.7, label="CES SA", zorder=5)
    m = np.isfinite(g_provider)
    ax.scatter(dates_arr[m], pct(g_provider[m]), s=6, c="#2ca02c", alpha=0.4, label="Provider (NSA)", zorder=3)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Monthly growth (%)")
    ax.set_title("SA Total Employment Growth: Latent vs CES SA & Provider")
    ax.legend(fontsize=8, loc="upper right")
    _year_axis(ax)

    # Panel 2: NSA total growth
    ax = axes[1]
    ax.fill_between(dates, pct(g_nsa_lo), pct(g_nsa_hi), alpha=0.25, color="steelblue", label="80% CI")
    ax.plot(dates, pct(g_nsa_mean), "steelblue", lw=1.5, label="Latent total (NSA)")
    ax.scatter(dates, pct(g_ces_nsa), s=10, c="darkorange", alpha=0.7, label="CES NSA", zorder=5)
    ax.scatter(dates_arr[m], pct(g_provider[m]), s=8, c="#2ca02c", alpha=0.5, label="Provider (NSA)", zorder=4)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Monthly growth (%)")
    ax.set_title("NSA Total Employment Growth: Latent vs CES NSA & Provider")
    ax.legend(fontsize=8, loc="upper right")
    _year_axis(ax)

    # Panel 3: Continuing-units growth
    ax = axes[2]
    ax.fill_between(dates, pct(g_cont_lo), pct(g_cont_hi), alpha=0.25, color="steelblue", label="80% CI")
    ax.plot(dates, pct(g_cont_mean), "steelblue", lw=1.5, label="Latent cont. units (SA)")
    ax.scatter(dates_arr[m], pct(g_provider[m]), s=10, c="#2ca02c", alpha=0.6, label="Provider (NSA)", zorder=5)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Monthly growth (%)")
    ax.set_title("SA Continuing-Units Growth vs Provider (NSA)")
    ax.legend(fontsize=8, loc="upper right")
    _year_axis(ax)

    # Panel 4: Seasonal pattern
    K = N_HARMONICS
    fourier_post = idata.posterior["fourier_coefs_det"].values
    last_yr_coefs = fourier_post[:, :, -1, :]

    k_vals = np.arange(1, K + 1)
    month_idx = np.arange(12)
    cos_basis_12 = np.cos(2 * np.pi * k_vals * month_idx[:, None] / 12)
    sin_basis_12 = np.sin(2 * np.pi * k_vals * month_idx[:, None] / 12)

    A_last = last_yr_coefs[:, :, :K]
    B_last = last_yr_coefs[:, :, K:]
    s_12 = np.zeros((*A_last.shape[:2], 12))
    for mo in range(12):
        for k in range(K):
            s_12[:, :, mo] += (
                A_last[:, :, k] * cos_basis_12[mo, k]
                + B_last[:, :, k] * sin_basis_12[mo, k]
            )

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
    ax.bar(x_pos - width / 2, pct(s_mean), width, color="steelblue", alpha=0.7, label="Model (current year)")
    ax.bar(x_pos + width / 2, pct(emp_seasonal), width, color="darkorange", alpha=0.7, label="Empirical (CES NSA-SA)")
    ax.errorbar(
        x_pos - width / 2, pct(s_mean),
        yerr=[pct(s_mean - s_lo), pct(s_hi - s_mean)],
        fmt="none", c="k", capsize=3,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("Seasonal effect (%)")
    ax.set_title("Estimated vs Empirical Monthly Seasonal Pattern")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "growth_and_seasonal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'growth_and_seasonal.png'}")


def plot_reconstructed_index(idata: az.InferenceData, data: dict) -> None:
    """Reconstructed latent SA/NSA index vs observed series."""
    dates = data["dates"]
    levels = data["levels"]

    g_total_mean = idata.posterior["g_total_sa"].values.mean(axis=(0, 1))
    g_nsa_mean = idata.posterior["g_total_nsa"].values.mean(axis=(0, 1))

    ces_sa_vals = levels["ces_sa_index"].to_numpy(dtype=float)
    ces_nsa_vals = levels["ces_nsa_index"].to_numpy(dtype=float)
    qcew_nsa_vals = levels["qcew_nsa_index"].to_numpy(dtype=float)
    provider_vals = levels["g_employment"].to_numpy(dtype=float)

    log_base_sa = np.log(ces_sa_vals[0])
    log_base_nsa = np.log(ces_nsa_vals[0])
    index_sa = np.exp(log_base_sa + np.cumsum(g_total_mean))
    index_nsa = np.exp(log_base_nsa + np.cumsum(g_nsa_mean))

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.plot(dates, index_sa, "steelblue", lw=2, label="Latent total (SA)")
    ax.plot(dates, index_nsa, "steelblue", lw=1.5, ls="--", alpha=0.6, label="Latent total (NSA)")
    ax.plot(dates[1:], ces_sa_vals[1:], "darkorange", lw=1.5, alpha=0.8, label="CES SA")
    ax.plot(dates[1:], ces_nsa_vals[1:], "darkorange", lw=1, ls="--", alpha=0.5, label="CES NSA")

    pp_mask = np.isfinite(provider_vals)
    if pp_mask.sum() > 1:
        pp_dates = np.array(dates)
        ax.plot(pp_dates[pp_mask][1:], provider_vals[pp_mask][1:], "#2ca02c", lw=1, alpha=0.6, label="Provider G (NSA)")

    qcew_mask = np.isfinite(qcew_nsa_vals)
    ax.scatter(np.array(dates)[qcew_mask], qcew_nsa_vals[qcew_mask], s=20, c="red", marker="o", alpha=0.7, label="QCEW (NSA)", zorder=5)

    ax.set_ylabel("Index (base ~ 100)")
    ax.set_title("Reconstructed Latent Index vs Observed Series")
    ax.legend(fontsize=8)
    _year_axis(ax)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "reconstructed_index.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'reconstructed_index.png'}")


def plot_bd_diagnostics(idata: az.InferenceData, data: dict) -> None:
    """BD offset time-series and covariate decomposition."""
    dates = data["dates"]
    bd_post = idata.posterior["bd"].values
    bd_mean = bd_post.mean(axis=(0, 1))
    bd_lo = np.percentile(bd_post, 10, axis=(0, 1))
    bd_hi = np.percentile(bd_post, 90, axis=(0, 1))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Panel 1: bd_t time series
    ax = axes[0]
    ax.fill_between(dates, bd_lo * 100, bd_hi * 100, alpha=0.25, color="steelblue", label="80% CI")
    ax.plot(dates, bd_mean * 100, "steelblue", lw=1.5, label="bd_t (posterior mean)")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("BD offset (%/mo)")
    ax.set_title("Structural Birth/Death Offset Over Time")
    ax.legend(fontsize=8)
    _year_axis(ax)

    # Panel 2: BD covariate decomposition
    ax = axes[1]
    phi_0_m = idata.posterior["phi_0"].values.flatten().mean()
    c_phi0 = np.full(len(dates), phi_0_m * 100)
    ax.plot(dates, c_phi0, lw=1.2, label="phi_0 (intercept)", color="gray", ls="--")

    cyc_colors = ["#9467bd", "#ff7f0e"]
    cyc_labels = ["claims", "jolts"]
    cyc_contribs = np.zeros(len(dates))
    if "phi_3" in idata.posterior:
        phi_3_m = idata.posterior["phi_3"].values.mean(axis=(0, 1))
        cyc_i = 0
        for ci, key in enumerate(["claims_c", "jolts_c"]):
            arr = data.get(key)
            if arr is not None and np.any(arr != 0.0):
                contrib = phi_3_m[cyc_i] * arr * 100
                clr = cyc_colors[ci % len(cyc_colors)]
                ax.plot(dates, contrib, lw=1.2, label=f"phi_3*{cyc_labels[ci]}", color=clr)
                cyc_contribs += contrib
                cyc_i += 1

    xi_bd_m = bd_mean * 100 - c_phi0 - cyc_contribs
    ax.plot(dates, xi_bd_m, lw=0.8, alpha=0.5, label="sigma_bd*xi (residual)", color="lightgray")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Contribution (%/mo)")
    ax.set_title("BD Covariate Decomposition")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    _year_axis(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "bd_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'bd_diagnostics.png'}")


def plot_residuals(idata: az.InferenceData, data: dict) -> None:
    """Standardised residuals per source over time.

    CES and provider residuals should be ~N(0,1).
    QCEW residuals follow t(nu) — slightly heavier tails are expected.
    """
    dates = data["dates"]
    dates_arr = np.array(dates)

    g_cont = idata.posterior["g_cont"].values.mean(axis=(0, 1))
    g_total_sa = idata.posterior["g_total_sa"].values.mean(axis=(0, 1))
    g_total_nsa = idata.posterior["g_total_nsa"].values.mean(axis=(0, 1))
    seasonal = idata.posterior["seasonal"].values.mean(axis=(0, 1))
    g_cont_nsa = g_cont + seasonal

    alpha_ces = idata.posterior["alpha_ces"].values.flatten().mean()
    lambda_ces = idata.posterior["lambda_ces"].values.flatten().mean()
    sigma_ces_sa = idata.posterior["sigma_ces_sa"].values.mean(axis=(0, 1))
    sigma_ces_nsa = idata.posterior["sigma_ces_nsa"].values.mean(axis=(0, 1))

    vintage_map = data["ces_vintage_map"]
    inv_map = {i: v for v, i in vintage_map.items()}
    orig_colors = {0: "#ff7f0e", 1: "#d62728", 2: "#2ca02c"}
    orig_labels = {0: "1st", 1: "2nd", 2: "Final"}
    vintage_colors = {i: orig_colors[inv_map[i]] for i in inv_map}
    vintage_labels = {i: orig_labels[inv_map[i]] for i in inv_map}

    has_provider = len(data["provider_obs"]) > 0
    n_panels = 2 + (1 if has_provider else 0) + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.2 * n_panels), sharex=True)

    def _resid_lines(ax, title):
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axhline(2, color="red", lw=0.5, ls=":", alpha=0.5)
        ax.axhline(-2, color="red", lw=0.5, ls=":", alpha=0.5)
        ax.set_ylabel("Std. residual")
        ax.set_title(title)

    # CES SA
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
                ax.scatter(dates_arr[ces_sa_obs[mask]], resid[mask], s=8, c=vintage_colors[v], alpha=0.6, label=vintage_labels[v])
        ax.legend(fontsize=7)
    _resid_lines(ax, "CES SA (best-available)")

    # CES NSA
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
                ax.scatter(dates_arr[ces_nsa_obs[mask]], resid[mask], s=8, c=vintage_colors[v], alpha=0.6, label=vintage_labels[v])
        ax.legend(fontsize=7)
    _resid_lines(ax, "CES NSA (best-available)")

    panel_idx = 2

    # Provider
    if has_provider:
        ax = axes[panel_idx]
        p_obs = data["provider_obs"]
        alp_g = idata.posterior["alpha_g"].values.flatten().mean()
        lam_g = idata.posterior["lam_g"].values.flatten().mean()
        sig_g = idata.posterior["sigma_pp_g"].values.flatten().mean()
        mu_base = alp_g + lam_g * g_cont_nsa[p_obs]
        resid = (data["g_provider"][p_obs] - mu_base) / sig_g
        ax.scatter(dates_arr[p_obs], resid, s=8, c="#2ca02c", alpha=0.6)
        _resid_lines(ax, "Provider G")
        panel_idx += 1

    # QCEW
    ax = axes[panel_idx]
    qcew_obs = data["qcew_obs"]
    pred = g_total_nsa[qcew_obs]
    qcew_is_m2 = np.asarray(data["qcew_is_m2"])
    sigma_mid = float(idata.posterior["sigma_qcew_mid"].values.flatten().mean())
    sigma_boundary = float(idata.posterior["sigma_qcew_boundary"].values.flatten().mean())
    qcew_mult = np.asarray(data["qcew_noise_mult"], dtype=float)
    qcew_sigma = np.where(qcew_is_m2, sigma_mid * qcew_mult, sigma_boundary * qcew_mult)
    resid = (data["g_qcew"][qcew_obs] - pred) / qcew_sigma
    ax.scatter(dates_arr[qcew_obs][qcew_is_m2], resid[qcew_is_m2], s=12, c="darkred", alpha=0.7, label="M2 (mid-quarter)")
    ax.scatter(dates_arr[qcew_obs][~qcew_is_m2], resid[~qcew_is_m2], s=12, c="salmon", alpha=0.7, label="M3+M1 (boundary)")
    _resid_lines(ax, "QCEW")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(
        "Standardised Residuals by Source (CES/Provider: N(0,1); QCEW: t(nu))",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'residuals.png'}")

