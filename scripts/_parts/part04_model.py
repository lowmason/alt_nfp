# =============================================================================
# SECTION 3: PyMC MODEL CONSTRUCTION
# =============================================================================
#
# This function builds the full probabilistic model.  In MATLAB terms, this
# is where you would define your JAGS/Stan model block.  PyMC uses a
# "context manager" (the `with pm.Model() as model:` block) where each
# line declares a random variable or a deterministic transformation.
#
# The model has these blocks (in order):
#   A. QCEW observation noise (two-tier sigma with revision multipliers)
#   B. Latent AR(1) growth process (tau reparameterization)
#   C. Fourier seasonal (Gaussian random walk on harmonic coefficients)
#   D. Structural birth/death (intercept + cyclical indicators)
#   E. Composite growth signals
#   F. QCEW likelihood (Student-t, truth anchor)
#   G. CES likelihood (Normal, vintage-indexed sigma)
#   H. Provider likelihood (Normal, iid)
#
# =============================================================================


def build_model(data: dict) -> pm.Model:
    """Build the QCEW-anchored PyMC state-space model.

    Parameters
    ----------
    data : dict
        Output of load_data().

    Returns
    -------
    pm.Model
        Compiled PyMC model ready for sampling.
    """
    T = data["T"]
    month_of_year = data["month_of_year"]
    era_idx = data["era_idx"]

    with pm.Model() as model:

        # =============================================================
        # A. QCEW OBSERVATION NOISE
        # =============================================================
        # Two estimated base sigmas (LogNormal to avoid Neal's funnel):
        #   sigma_qcew_mid      — for M2 months (Feb, May, Aug, Nov)
        #   sigma_qcew_boundary — for M1+M3 months (boundary months)
        #
        # Per-observation sigma = base * revision_multiplier * era_mult
        # (the multipliers are pre-computed in the CSV and passed in
        # via data["qcew_noise_mult"]).
        #
        # WHY LOGNORMAL?  When sigma -> 0, QCEW observations become
        # infinitely precise and dominate the posterior, creating a
        # bimodal posterior.  LogNormal keeps sigma bounded away from
        # zero, preventing this pathology.
        # =============================================================

        sigma_qcew_mid = pm.LogNormal(
            "sigma_qcew_mid",
            mu=LOG_SIGMA_QCEW_MID_MU,
            sigma=LOG_SIGMA_QCEW_MID_SD,
        )
        sigma_qcew_boundary = pm.LogNormal(
            "sigma_qcew_boundary",
            mu=LOG_SIGMA_QCEW_BOUNDARY_MU,
            sigma=LOG_SIGMA_QCEW_BOUNDARY_SD,
        )

        # Select base sigma per QCEW observation based on M2 vs boundary
        qcew_is_m2_t = pt.as_tensor_variable(
            np.asarray(data["qcew_is_m2"], dtype=bool)
        )
        base_sigma = pt.switch(qcew_is_m2_t, sigma_qcew_mid, sigma_qcew_boundary)
        qcew_sigma = base_sigma * pt.as_tensor_variable(
            data["qcew_noise_mult"], dtype="float64"
        )

        # =============================================================
        # B. LATENT CONTINUING-UNITS GROWTH: AR(1)
        # =============================================================
        # g_cont(t) = mu_g + phi * (g_cont(t-1) - mu_g) + sigma_g * eps(t)
        #
        # REPARAMETERIZATION:
        #   tau = marginal SD of the stationary AR(1) process
        #   sigma_g = tau * sqrt(1 - phi^2)   [derived]
        #
        # This breaks the phi-sigma ridge that causes low ESS when phi
        # is near 1 (high persistence).  In the (phi, sigma_g) space,
        # the posterior is a thin banana shape that the sampler struggles
        # with.  In the (phi, tau) space, the posterior is approximately
        # elliptical — much easier for NUTS.
        #
        # mu_g is ERA-SPECIFIC: Pre-COVID and Post-COVID have different
        # mean growth rates, but share phi and tau.  We tried era-specific
        # phi but it was underpowered (~96 Pre-COVID vs ~60 Post-COVID
        # months) and produced pathological modes.
        # =============================================================

        tau = pm.LogNormal("tau", mu=LOG_TAU_MU, sigma=LOG_TAU_SD)

        # phi_raw ~ Beta(18, 2): mean = 0.9, concentrated near high persistence.
        # Capped at 0.99 to prevent the unit-root boundary (phi=1 makes the
        # process non-stationary and tau undefined).
        phi_raw = pm.Beta("phi_raw", alpha=18, beta=2)
        phi = pt.minimum(phi_raw, 0.99)

        # Derived innovation SD
        sigma_g = tau * pt.sqrt(1 - phi**2)

        # Standard normal innovations (non-centered parameterization)
        eps_g = pm.Normal("eps_g", 0, 1, shape=T)

        # Era-specific mean growth
        mu_g_era = pm.Normal("mu_g_era", mu=0.001, sigma=0.005, shape=N_ERAS)
        mu_g = mu_g_era[era_idx]  # (T,) — per-timestep mean

        # Initial value
        g0 = mu_g[0] + sigma_g * eps_g[0]

        # AR(1) recursion via pytensor.scan
        # (This is like a for-loop that PyMC can differentiate through)
        def ar1_step(e_t, mu_t, g_prev, _phi, _sig):
            """One step of the AR(1): mean-revert toward mu_t."""
            return mu_t + _phi * (g_prev - mu_t) + _sig * e_t

        g_rest, _ = pytensor.scan(
            fn=ar1_step,
            sequences=[eps_g[1:], mu_g[1:]],
            outputs_info=[g0],
            non_sequences=[phi, sigma_g],
            strict=True,
        )

        g_cont = pt.concatenate([g0.reshape((1,)), g_rest])
        pm.Deterministic("g_cont", g_cont)

        # =============================================================
        # C. FOURIER SEASONAL (Gaussian Random Walk on coefficients)
        # =============================================================
        # s(t) = sum_k [ A_k(year(t)) * cos(2*pi*k*month(t)/12)
        #              + B_k(year(t)) * sin(2*pi*k*month(t)/12) ]
        #
        # A_k and B_k evolve as Gaussian random walks across years,
        # allowing the seasonal pattern to slowly change over time
        # (e.g., less pronounced seasonality post-COVID).
        #
        # sigma_fourier has 1/k scaling: higher harmonics (k=2,3,4)
        # evolve more slowly than the fundamental (k=1).
        # =============================================================

        K = N_HARMONICS
        n_years = data["n_years"]
        year_of_obs = data["year_of_obs"]

        # Innovation SD per harmonic, decreasing with k
        sigma_fourier_mu = LOG_SIGMA_FOURIER_MU - np.log(np.arange(1, K + 1))
        sigma_fourier = pm.LogNormal(
            "sigma_fourier",
            mu=sigma_fourier_mu,
            sigma=LOG_SIGMA_FOURIER_SD,
            shape=K,
        )

        # GRW on Fourier coefficients: shape (2K, n_years)
        # Rows 0..K-1 = A_k (cosine), rows K..2K-1 = B_k (sine)
        sigma_vec = pt.tile(sigma_fourier, 2)  # (2K,)
        fourier_coefs = pm.GaussianRandomWalk(
            "fourier_coefs",
            sigma=sigma_vec,
            init_dist=pm.Normal.dist(0, 0.015),
            shape=(2 * K, n_years),
        )

        # Evaluate seasonal at each observation
        k_vals = np.arange(1, K + 1)
        cos_basis = pt.cos(2 * np.pi * k_vals * month_of_year[:, None] / 12)
        sin_basis = pt.sin(2 * np.pi * k_vals * month_of_year[:, None] / 12)

        A_t = fourier_coefs[:K, year_of_obs].T   # (T, K)
        B_t = fourier_coefs[K:, year_of_obs].T   # (T, K)

        s_t = pt.sum(A_t * cos_basis + B_t * sin_basis, axis=1)  # (T,)
        pm.Deterministic("seasonal", s_t)
        pm.Deterministic("fourier_coefs_det", fourier_coefs.T)  # (n_years, 2K)

        # =============================================================
        # D. STRUCTURAL BIRTH/DEATH OFFSET
        # =============================================================
        # bd(t) = phi_0 + phi_3 * X_cycle(t) + sigma_bd * xi(t)
        #
        # phi_0 is the average BD rate at mean cyclical conditions.
        # phi_3 loads cyclical demand indicators (initial claims, JOLTS)
        # onto BD.  When indicators are unavailable (early sample or
        # forecast), their centered value is 0 and bd(t) collapses to
        # phi_0 + sigma_bd * xi(t).
        # =============================================================

        phi_0 = pm.Normal("phi_0", mu=0.001, sigma=0.002)
        sigma_bd = pm.LogNormal(
            "sigma_bd", mu=LOG_SIGMA_BD_MU, sigma=LOG_SIGMA_BD_SD
        )
        xi_bd = pm.Normal("xi_bd", 0, 1, shape=T)

        bd_t = phi_0 + sigma_bd * xi_bd

        # Cyclical indicators: claims and JOLTS (centered, 0 where unavailable)
        cyclical_arrays = []
        cyclical_names = []
        for key in ["claims_c", "jolts_c"]:
            arr = data.get(key)
            if arr is not None and np.any(arr != 0.0):
                cyclical_arrays.append(arr)
                cyclical_names.append(key)

        n_cyclical = len(cyclical_arrays)
        if n_cyclical > 0:
            phi_3 = pm.Normal("phi_3", mu=0, sigma=0.3, shape=n_cyclical)
            for i, arr in enumerate(cyclical_arrays):
                bd_t = bd_t + phi_3[i] * pt.as_tensor_variable(arr)

        pm.Deterministic("bd", bd_t)

        # =============================================================
        # E. COMPOSITE GROWTH SIGNALS
        # =============================================================
        # g_total_sa(t)  = g_cont(t) + bd(t)           [seasonally adjusted]
        # g_total_nsa(t) = g_cont(t) + s(t) + bd(t)   [not seasonally adjusted]
        # g_cont_nsa(t)  = g_cont(t) + s(t)            [cont. units, NSA]
        # =============================================================

        g_cont_nsa = g_cont + s_t
        g_total_sa = g_cont + bd_t
        g_total_nsa = g_cont + s_t + bd_t

        pm.Deterministic("g_total_sa", g_total_sa)
        pm.Deterministic("g_total_nsa", g_total_nsa)

        # =============================================================
        # F. QCEW LIKELIHOOD — TRUTH ANCHOR
        # =============================================================
        # Student-t with nu=5 for robustness to NAICS reclassification
        # outliers.  QCEW observes total growth (NSA = continuing + seasonal + BD).
        # =============================================================

        pm.StudentT(
            "obs_qcew",
            nu=QCEW_NU,
            mu=g_total_nsa[data["qcew_obs"]],
            sigma=qcew_sigma,
            observed=data["g_qcew"][data["qcew_obs"]],
        )

        # =============================================================
        # G. CES LIKELIHOOD — best-available print, vintage-indexed sigma
        # =============================================================
        # One observation per month per SA/NSA using the highest available
        # revision (Final > 2nd > 1st print).  This eliminates the 3x
        # overcounting from treating correlated vintages (rho > 0.99) as
        # independent.
        #
        # CES observes: alpha_ces + lambda_ces * g_total_{sa,nsa}
        #   alpha_ces : small bias relative to latent truth
        #   lambda_ces: loading (truncated at 0.5 to prevent disconnection)
        #   sigma_ces : per-vintage noise (1st print noisier than Final)
        # =============================================================

        alpha_ces = pm.Normal("alpha_ces", 0, 0.005)
        lambda_ces = pm.TruncatedNormal(
            "lambda_ces", mu=1.0, sigma=0.1, lower=0.5
        )

        n_ces_v = data["n_ces_vintages"]
        sigma_ces_sa = pm.LogNormal(
            "sigma_ces_sa",
            mu=LOG_SIGMA_CES_MU,
            sigma=LOG_SIGMA_CES_SD,
            shape=n_ces_v,
        )
        sigma_ces_nsa = pm.LogNormal(
            "sigma_ces_nsa",
            mu=LOG_SIGMA_CES_MU,
            sigma=LOG_SIGMA_CES_SD,
            shape=n_ces_v,
        )

        ces_sa_obs = data["ces_sa_obs"]
        ces_nsa_obs = data["ces_nsa_obs"]
        ces_sa_vidx = data["ces_sa_vintage_idx"]
        ces_nsa_vidx = data["ces_nsa_vintage_idx"]

        if len(ces_sa_obs) > 0:
            pm.Normal(
                "obs_ces_sa",
                mu=alpha_ces + lambda_ces * g_total_sa[ces_sa_obs],
                sigma=sigma_ces_sa[ces_sa_vidx],
                observed=data["g_ces_sa"][ces_sa_obs],
            )

        if len(ces_nsa_obs) > 0:
            pm.Normal(
                "obs_ces_nsa",
                mu=alpha_ces + lambda_ces * g_total_nsa[ces_nsa_obs],
                sigma=sigma_ces_nsa[ces_nsa_vidx],
                observed=data["g_ces_nsa"][ces_nsa_obs],
            )

        # =============================================================
        # H. PROVIDER LIKELIHOOD — iid measurement error
        # =============================================================
        # Provider observes continuing-units growth (NSA, no BD):
        #   y_provider = alpha_g + lam_g * g_cont_nsa + eps
        #
        # InverseGamma(3, 0.004) for sigma: mean = beta/(alpha-1) = 0.002.
        # This is the one scale parameter using InverseGamma instead of
        # LogNormal because provider noise is well-separated from zero.
        # =============================================================

        provider_obs = data["provider_obs"]
        if len(provider_obs) > 0:
            y_obs_provider = data["g_provider"][provider_obs]

            alpha_g = pm.Normal("alpha_g", 0, 0.005)
            lam_g = pm.Normal("lam_g", 1.0, 0.15)
            sigma_pp_g = pm.InverseGamma("sigma_pp_g", alpha=3.0, beta=0.004)

            mu_base = alpha_g + lam_g * g_cont_nsa[provider_obs]

            pm.Normal(
                "obs_g",
                mu=mu_base,
                sigma=sigma_pp_g,
                observed=y_obs_provider,
            )

    return model

