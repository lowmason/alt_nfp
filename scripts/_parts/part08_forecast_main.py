# =============================================================================
# SECTION 10: FORECAST
# =============================================================================
#
# Forward simulation from the posterior:
#   1. Propagate AR(1) g_cont forward using posterior (mu_g, phi, sigma_g)
#   2. Propagate structural BD (phi_0 + sigma_bd * innovation)
#   3. Evaluate Fourier seasonal using last year's coefficients
#   4. Reconstruct index levels from cumulative growth
#   5. Convert to jobs-added using the level-to-index ratio
#
# Uncertainty fans out because each forward step adds innovation noise.
# The forecast is conditional on the posterior — it integrates over all
# parameter uncertainty (unlike a point-estimate forecast).
#
# =============================================================================


def forecast_and_plot(idata: az.InferenceData, data: dict) -> None:
    """Forecast SA and NSA indices forward and plot."""
    dates = data["dates"]
    levels = data["levels"]

    # Posterior parameter samples
    mu_g_post = idata.posterior["mu_g_era"].values[:, :, -1]  # last era (Post-COVID)
    phi_post = idata.posterior["phi_raw"].values
    tau_post = idata.posterior["tau"].values
    sigma_g_post = tau_post * np.sqrt(1 - phi_post**2)
    fourier_post = idata.posterior["fourier_coefs_det"].values
    g_cont_post = idata.posterior["g_cont"].values
    g_sa_post = idata.posterior["g_total_sa"].values
    g_nsa_post = idata.posterior["g_total_nsa"].values

    phi_0_post = idata.posterior["phi_0"].values
    sigma_bd_post = idata.posterior["sigma_bd"].values

    n_chains, n_draws, T_hist = g_cont_post.shape

    # Build forecast date grid
    last_date = dates[-1]
    forecast_dates = []
    d = last_date
    while True:
        yr, mo = d.year, d.month
        d = date(yr + 1, 1, 12) if mo == 12 else date(yr, mo + 1, 12)
        forecast_dates.append(d)
        if d >= FORECAST_END:
            break

    n_fwd = len(forecast_dates)
    forecast_month_idx = [d.month - 1 for d in forecast_dates]

    # Simulate forward
    rng = np.random.default_rng(42)
    g_cont_fwd = np.zeros((n_chains, n_draws, n_fwd))
    g_sa_fwd = np.zeros((n_chains, n_draws, n_fwd))
    g_nsa_fwd = np.zeros((n_chains, n_draws, n_fwd))
    bd_fwd = np.zeros((n_chains, n_draws, n_fwd))

    K = N_HARMONICS

    for h in range(n_fwd):
        eps = rng.standard_normal((n_chains, n_draws))
        g_prev = g_cont_post[:, :, -1] if h == 0 else g_cont_fwd[:, :, h - 1]
        g_cont_fwd[:, :, h] = (
            mu_g_post + phi_post * (g_prev - mu_g_post) + sigma_g_post * eps
        )

        eps_bd = rng.standard_normal((n_chains, n_draws))
        bd_fwd[:, :, h] = phi_0_post + sigma_bd_post * eps_bd

        # Fourier seasonal for this forecast month
        mi = forecast_month_idx[h]
        k_vals = np.arange(1, K + 1)
        cos_val = np.cos(2 * np.pi * k_vals * mi / 12)
        sin_val = np.sin(2 * np.pi * k_vals * mi / 12)
        A_k = fourier_post[:, :, -1, :K]
        B_k = fourier_post[:, :, -1, K:]
        s_fwd = np.sum(A_k * cos_val + B_k * sin_val, axis=-1)

        g_sa_fwd[:, :, h] = g_cont_fwd[:, :, h] + bd_fwd[:, :, h]
        g_nsa_fwd[:, :, h] = g_cont_fwd[:, :, h] + s_fwd + bd_fwd[:, :, h]

    # Reconstruct index paths
    ces_sa_vals = levels["ces_sa_index"].to_numpy(dtype=float)
    ces_nsa_vals = levels["ces_nsa_index"].to_numpy(dtype=float)
    log_base_sa = np.log(ces_sa_vals[0])
    log_base_nsa = np.log(ces_nsa_vals[0])

    base_row_idx = np.argmin(np.abs(ces_sa_vals - 100.0))
    ces_sa_base_level = data["ces_sa_level"][base_row_idx]
    ces_nsa_base_level = data["ces_nsa_level"][base_row_idx]

    cum_sa_hist = log_base_sa + np.cumsum(g_sa_post, axis=2)
    cum_nsa_hist = log_base_nsa + np.cumsum(g_nsa_post, axis=2)
    cum_sa_fwd = cum_sa_hist[:, :, -1:] + np.cumsum(g_sa_fwd, axis=2)
    cum_nsa_fwd = cum_nsa_hist[:, :, -1:] + np.cumsum(g_nsa_fwd, axis=2)

    idx_sa_hist = np.exp(cum_sa_hist)
    idx_nsa_hist = np.exp(cum_nsa_hist)
    idx_sa_fwd = np.exp(cum_sa_fwd)
    idx_nsa_fwd = np.exp(cum_nsa_fwd)

    # Jobs added
    idx_to_sa = ces_sa_base_level / 100.0
    idx_to_nsa = ces_nsa_base_level / 100.0
    lvl_sa_fwd = idx_sa_fwd * idx_to_sa
    lvl_nsa_fwd = idx_nsa_fwd * idx_to_nsa
    lvl_sa_hist = idx_sa_hist * idx_to_sa
    lvl_nsa_hist = idx_nsa_hist * idx_to_nsa

    chg_sa_fwd = np.zeros((n_chains, n_draws, n_fwd))
    chg_nsa_fwd = np.zeros((n_chains, n_draws, n_fwd))
    chg_sa_fwd[:, :, 0] = lvl_sa_fwd[:, :, 0] - lvl_sa_hist[:, :, -1]
    chg_nsa_fwd[:, :, 0] = lvl_nsa_fwd[:, :, 0] - lvl_nsa_hist[:, :, -1]
    for i in range(1, n_fwd):
        chg_sa_fwd[:, :, i] = lvl_sa_fwd[:, :, i] - lvl_sa_fwd[:, :, i - 1]
        chg_nsa_fwd[:, :, i] = lvl_nsa_fwd[:, :, i] - lvl_nsa_fwd[:, :, i - 1]

    # Print forecast tables
    print("\n" + "=" * 72)
    print("FORECAST: SA & NSA Index")
    print("=" * 72)
    print(f"{'Date':>12}  {'SA Mean':>9} {'SA 80% HDI':>18}  {'NSA Mean':>9} {'NSA 80% HDI':>18}")
    print("-" * 72)

    sa_last = idx_sa_hist[:, :, -1].flatten()
    nsa_last = idx_nsa_hist[:, :, -1].flatten()
    print(
        f"{str(dates[-1]):>12}  {sa_last.mean():9.2f} "
        f"[{np.percentile(sa_last, 10):7.2f}, {np.percentile(sa_last, 90):7.2f}]  "
        f"{nsa_last.mean():9.2f} "
        f"[{np.percentile(nsa_last, 10):7.2f}, {np.percentile(nsa_last, 90):7.2f}]  "
        f"<- last obs"
    )
    for i, fd in enumerate(forecast_dates):
        sa_v = idx_sa_fwd[:, :, i].flatten()
        nsa_v = idx_nsa_fwd[:, :, i].flatten()
        print(
            f"{str(fd):>12}  {sa_v.mean():9.2f} "
            f"[{np.percentile(sa_v, 10):7.2f}, {np.percentile(sa_v, 90):7.2f}]  "
            f"{nsa_v.mean():9.2f} "
            f"[{np.percentile(nsa_v, 10):7.2f}, {np.percentile(nsa_v, 90):7.2f}]  "
            f"<- forecast"
        )

    print("\n" + "=" * 84)
    print("FORECAST: Jobs added (thousands, month-over-month)")
    print("=" * 84)
    for i, fd in enumerate(forecast_dates):
        sa = chg_sa_fwd[:, :, i].flatten()
        nsa = chg_nsa_fwd[:, :, i].flatten()
        print(
            f"{str(fd):>12}  SA: {sa.mean():+8,.0f}k "
            f"[{np.percentile(sa, 10):+8,.0f}, {np.percentile(sa, 90):+8,.0f}]  "
            f"NSA: {nsa.mean():+8,.0f}k "
            f"[{np.percentile(nsa, 10):+8,.0f}, {np.percentile(nsa, 90):+8,.0f}]"
        )

    print("\n" + "=" * 72)
    print("FORECAST: Monthly Growth Rates")
    print("=" * 72)
    for i, fd in enumerate(forecast_dates):
        gsa = g_sa_fwd[:, :, i].flatten() * 100
        gnsa = g_nsa_fwd[:, :, i].flatten() * 100
        print(
            f"{str(fd):>12}  SA: {gsa.mean():+.3f}% "
            f"[{np.percentile(gsa, 10):+.3f}%, {np.percentile(gsa, 90):+.3f}%]  "
            f"NSA: {gnsa.mean():+.3f}% "
            f"[{np.percentile(gnsa, 10):+.3f}%, {np.percentile(gnsa, 90):+.3f}%]"
        )

    # ---- Plot: index forecast ----
    connecting = [dates[-1]] + forecast_dates
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for panel_idx, (label, idx_h, idx_f, obs_vals) in enumerate([
        ("SA", idx_sa_hist, idx_sa_fwd, ces_sa_vals),
        ("NSA", idx_nsa_hist, idx_nsa_fwd, ces_nsa_vals),
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
        ax.plot(dates[1:], obs_vals[1:], "darkorange", lw=1, alpha=0.7, label=f"CES {label}")

        ax.axvline(dates[-1], color="gray", ls=":", lw=1, alpha=0.7, label="Forecast start")
        ax.set_ylabel("Index (base ~ 100)")
        ax.set_title(f"{label} Total Employment Index: Estimate + Forecast")
        ax.legend(fontsize=8, loc="upper left")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecast_sa_nsa.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'forecast_sa_nsa.png'}")

    # ---- Plot: jobs-added forecast ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    for panel_idx, (label, chg_fwd) in enumerate([
        ("SA", chg_sa_fwd),
        ("NSA", chg_nsa_fwd),
    ]):
        ax = axes[panel_idx]
        fwd_mean = chg_fwd.mean(axis=(0, 1))
        fwd_lo = np.percentile(chg_fwd, 10, axis=(0, 1))
        fwd_hi = np.percentile(chg_fwd, 90, axis=(0, 1))

        ax.fill_between(forecast_dates, fwd_lo, fwd_hi, alpha=0.25, color="#ef4444", label="80% HDI")
        ax.plot(forecast_dates, fwd_mean, color="#ef4444", lw=2.5, ls="--", label="Forecast")

        for i, fd in enumerate(forecast_dates):
            fm = chg_fwd[:, :, i].flatten().mean()
            flo = np.percentile(chg_fwd[:, :, i].flatten(), 10)
            fhi = np.percentile(chg_fwd[:, :, i].flatten(), 90)
            ax.errorbar(
                fd, fm, yerr=[[fm - flo], [fhi - fm]],
                fmt="o", color="#ef4444", markersize=8, capsize=6,
                capthick=2, elinewidth=2, zorder=10,
            )
            ax.annotate(
                f"{fm:+,.0f}k", xy=(fd, fhi), xytext=(8, 8),
                textcoords="offset points", fontsize=9, fontweight="bold",
                color="#ef4444",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ef4444", alpha=0.9),
            )

        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_ylabel("Jobs added (thousands)", fontsize=11)
        ax.set_title(f"{label} Total Nonfarm: Jobs Added (MoM) Forecast", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+,.0f}"))
        ax.grid(axis="y", alpha=0.3, ls="--")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecast_levels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'forecast_levels.png'}")


# =============================================================================
# SECTION 11: MAIN ENTRY POINT
# =============================================================================


def main() -> None:
    """Run the full estimation pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("=" * 80)
    print("alt_nfp STANDALONE ESTIMATION")
    print("=" * 80)
    data = load_data()

    # 2. Build model
    print("\nBuilding PyMC model...")
    model = build_model(data)
    print(f"  Free RVs: {[rv.name for rv in model.free_RVs]}")
    print(f"  Observed RVs: {[rv.name for rv in model.observed_RVs]}")

    # 3. Prior predictive checks
    print("\n" + "=" * 80)
    run_prior_predictive(model, data)

    # 4. Sample
    print("\n" + "=" * 80)
    print("MCMC SAMPLING")
    print("=" * 80)
    idata = sample_model(model)

    # 5. Diagnostics
    print_diagnostics(idata, data)
    plot_divergences(idata, data)

    # 6. Posterior predictive checks
    print("\n" + "=" * 80)
    run_posterior_predictive(model, idata, data)

    # 7. LOO-CV
    print("\n" + "=" * 80)
    run_loo_cv(model, idata, data)

    # 8. Plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    plot_growth_and_seasonal(idata, data)
    plot_reconstructed_index(idata, data)
    plot_bd_diagnostics(idata, data)
    plot_residuals(idata, data)

    # 9. Forecast
    print("\n" + "=" * 80)
    forecast_and_plot(idata, data)

    # 10. Save
    idata.to_netcdf(str(OUTPUT_DIR / "alt_nfp_standalone_idata.nc"))
    print(f"\nInferenceData saved to {OUTPUT_DIR / 'alt_nfp_standalone_idata.nc'}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
