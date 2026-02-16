# ---------------------------------------------------------------------------
# alt_nfp.model — PyMC model specification
# ---------------------------------------------------------------------------
"""QCEW-anchored state-space model with config-driven providers and
structural time-varying birth/death."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from .config import N_HARMONICS, SIGMA_QCEW_M3, SIGMA_QCEW_M12


def build_model(
    data: dict,
    sigma_qcew_m3: float | None = None,
    sigma_qcew_m12: float | None = None,
) -> pm.Model:
    """Build the QCEW-anchored PyMC model.

    Parameters
    ----------
    data : dict
        Output of :func:`alt_nfp.data.load_data`.
    sigma_qcew_m3 : float, optional
        Override QCEW quarter-end noise.  Defaults to config value.
    sigma_qcew_m12 : float, optional
        Override QCEW retrospective-UI noise.  Defaults to config value.

    Key changes from v2
    --------------------
    * Provider likelihoods are generated in a loop driven by
      ``data['pp_data']`` / ``ProviderConfig``.
    * Birth/death is time-varying:
      ``bd_t = φ₀ + φ₁·birth_rate_c + φ₂·bd_qcew_c + σ_bd·ξ_t``
    """
    if sigma_qcew_m3 is None:
        sigma_qcew_m3 = SIGMA_QCEW_M3
    if sigma_qcew_m12 is None:
        sigma_qcew_m12 = SIGMA_QCEW_M12

    T = data["T"]
    month_of_year = data["month_of_year"]
    pp_data = data["pp_data"]

    qcew_sigma_fixed = np.where(data["qcew_is_m3"], sigma_qcew_m3, sigma_qcew_m12)

    with pm.Model() as model:

        # =============================================================
        # Latent continuing-units growth: AR(1) with mean reversion
        # =============================================================

        mu_g = pm.Normal("mu_g", mu=0.001, sigma=0.005)
        phi = pm.Uniform("phi", lower=0.0, upper=0.99)
        sigma_g = pm.HalfNormal("sigma_g", sigma=0.005)

        eps_g = pm.Normal("eps_g", 0, 1, shape=T)
        g0 = mu_g + sigma_g * eps_g[0]

        def ar1_step(e_t, g_prev, _mu, _phi, _sig):
            return _mu + _phi * (g_prev - _mu) + _sig * e_t

        g_rest, _ = pytensor.scan(
            fn=ar1_step,
            sequences=[eps_g[1:]],
            outputs_info=[g0],
            non_sequences=[mu_g, phi, sigma_g],
            strict=True,
        )
        g_cont = pt.concatenate([g0.reshape((1,)), g_rest])
        pm.Deterministic("g_cont", g_cont)

        # =============================================================
        # Fourier seasonal with annually-evolving amplitudes (GRW)
        #
        #   s_t = Σ_k [ A_k(y(t)) cos(2πk m(t)/12)
        #             + B_k(y(t)) sin(2πk m(t)/12) ]
        #
        # A_k, B_k evolve as Gaussian random walks across years.
        # =============================================================

        K = N_HARMONICS
        n_years = data['n_years']
        year_of_obs = data['year_of_obs']

        # Innovation std per harmonic (decreasing with k)
        sigma_fourier = pm.HalfNormal(
            'sigma_fourier',
            sigma=0.005 / np.arange(1, K + 1),
            shape=K,
        )

        # GRW on Fourier coefficients: shape (n_years, 2*K)
        # Columns 0..K-1 are A_k, columns K..2K-1 are B_k
        sigma_vec = pt.tile(sigma_fourier, 2)  # (2K,)
        fourier_coefs = pm.GaussianRandomWalk(
            'fourier_coefs',
            sigma=sigma_vec,
            init_dist=pm.Normal.dist(0, 0.015),
            shape=(n_years, 2 * K),
        )

        # Evaluate seasonal at each observation t
        k_vals = np.arange(1, K + 1)  # (K,)
        cos_basis = pt.cos(2 * np.pi * k_vals * month_of_year[:, None] / 12)  # (T, K)
        sin_basis = pt.sin(2 * np.pi * k_vals * month_of_year[:, None] / 12)  # (T, K)

        A_t = fourier_coefs[year_of_obs, :K]   # (T, K)
        B_t = fourier_coefs[year_of_obs, K:]   # (T, K)

        s_t = pt.sum(A_t * cos_basis + B_t * sin_basis, axis=1)  # (T,)
        pm.Deterministic('seasonal', s_t)            # per-obs seasonal value
        pm.Deterministic('fourier_coefs_det', fourier_coefs)  # (n_years, 2K)

        # =============================================================
        # Structural birth/death offset
        #
        #   bd_t = φ_0 + φ_1·birth_rate_c + φ_2·bd_qcew_c
        #          + φ_3 · X^cycle + σ_bd · ξ_t
        #
        # Covariates are centred so φ_0 ≈ mean BD at average covariate
        # values.  Where a covariate is unavailable (early sample or
        # nowcast) the centred value is zeroed out and bd_t collapses
        # to  φ_0 + σ_bd·ξ_t  (similar to v2's constant bd).
        # =============================================================

        phi_0 = pm.Normal("phi_0", mu=0.001, sigma=0.002)
        phi_1 = pm.Normal("phi_1", mu=0.5, sigma=0.5)
        phi_2 = pm.Normal("phi_2", mu=0.3, sigma=0.3)
        sigma_bd = pm.HalfNormal("sigma_bd", sigma=0.001)

        xi_bd = pm.Normal("xi_bd", 0, 1, shape=T)

        bd_t = (
            phi_0
            + phi_1 * pt.as_tensor_variable(data["birth_rate_c"])
            + phi_2 * pt.as_tensor_variable(data["bd_qcew_c"])
            + sigma_bd * xi_bd
        )

        # Cyclical indicators (demand-side BD covariates)
        cyclical_keys = ['claims_c', 'nfci_c', 'biz_apps_c']
        cyclical_data = []
        cyclical_names = []
        for key in cyclical_keys:
            arr = data.get(key)
            if arr is not None and np.any(arr != 0.0):
                cyclical_data.append(arr)
                cyclical_names.append(key)

        n_cyclical = len(cyclical_data)
        if n_cyclical > 0:
            phi_3 = pm.Normal('phi_3', mu=0, sigma=0.3, shape=n_cyclical)
            for i, arr in enumerate(cyclical_data):
                bd_t = bd_t + phi_3[i] * pt.as_tensor_variable(arr)

        pm.Deterministic("bd", bd_t)

        # =============================================================
        # Composite growth signals
        # =============================================================

        g_cont_nsa = g_cont + s_t
        g_total_sa = g_cont + bd_t
        g_total_nsa = g_cont + s_t + bd_t

        pm.Deterministic("g_total_sa", g_total_sa)
        pm.Deterministic("g_total_nsa", g_total_nsa)

        # =============================================================
        # QCEW likelihood — TRUTH ANCHOR
        # =============================================================

        pm.Normal(
            "obs_qcew",
            mu=g_total_nsa[data["qcew_obs"]],
            sigma=qcew_sigma_fixed,
            observed=data["g_qcew"][data["qcew_obs"]],
        )

        # =============================================================
        # CES likelihood — vintage-specific noise
        #
        # Shared α_CES and λ_CES across vintages.  Separate σ per
        # vintage v ∈ {1, 2, 3} (first print, second print, final).
        # =============================================================

        alpha_ces = pm.Normal("alpha_ces", 0, 0.005)
        lambda_ces = pm.Normal("lambda_ces", 1.0, 0.15)
        sigma_ces_sa = pm.InverseGamma(
            "sigma_ces_sa", alpha=3.0, beta=0.004, shape=3
        )
        sigma_ces_nsa = pm.InverseGamma(
            "sigma_ces_nsa", alpha=3.0, beta=0.004, shape=3
        )

        for v in range(3):
            g_sa_v = data['g_ces_sa_by_vintage'][v]
            obs_v = np.where(np.isfinite(g_sa_v))[0]
            if len(obs_v) > 0:
                pm.Normal(
                    f'obs_ces_sa_v{v + 1}',
                    mu=alpha_ces + lambda_ces * g_total_sa[obs_v],
                    sigma=sigma_ces_sa[v],
                    observed=g_sa_v[obs_v],
                )

            g_nsa_v = data['g_ces_nsa_by_vintage'][v]
            obs_v_nsa = np.where(np.isfinite(g_nsa_v))[0]
            if len(obs_v_nsa) > 0:
                pm.Normal(
                    f'obs_ces_nsa_v{v + 1}',
                    mu=alpha_ces + lambda_ces * g_total_nsa[obs_v_nsa],
                    sigma=sigma_ces_nsa[v],
                    observed=g_nsa_v[obs_v_nsa],
                )

        # =============================================================
        # PP likelihoods — config-driven, per-provider
        # =============================================================

        for pp in pp_data:
            cfg = pp["config"]
            name = cfg.name.lower()
            obs_idx = pp["pp_obs"]
            y_obs = pp["g_pp"][obs_idx]

            # Per-provider measurement parameters
            alpha_p = pm.Normal(f"alpha_{name}", 0, 0.005)
            lam_p = pm.Normal(f"lam_{name}", 1.0, 0.15)
            sigma_p = pm.InverseGamma(f"sigma_{name}", alpha=3.0, beta=0.004)

            mu_base = alpha_p + lam_p * g_cont_nsa[obs_idx]

            if cfg.error_model == "ar1":
                # AR(1) measurement error (e.g. multi-establishment restructuring)
                rho_p = pm.Beta(f"rho_{name}", alpha=2, beta=3)

                mu_cond = pt.concatenate(
                    [
                        mu_base[:1],
                        mu_base[1:]
                        + rho_p * (pt.as_tensor_variable(y_obs[:-1]) - mu_base[:-1]),
                    ]
                )
                sigma_cond = pt.concatenate(
                    [
                        (sigma_p / pt.sqrt(1.0 - rho_p**2)).reshape((1,)),
                        pt.ones(len(obs_idx) - 1) * sigma_p,
                    ]
                )
                pm.Normal(f"obs_{name}", mu=mu_cond, sigma=sigma_cond, observed=y_obs)

            elif cfg.error_model == "iid":
                pm.Normal(f"obs_{name}", mu=mu_base, sigma=sigma_p, observed=y_obs)

    return model
