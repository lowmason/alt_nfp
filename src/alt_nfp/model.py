"""PyMC state-space model for employment growth nowcasting.

Builds a hierarchical Bayesian model with the following components:

1. **Latent continuing-units growth** — AR(1) process with mean reversion.
2. **Fourier seasonal** — annually-evolving harmonic amplitudes via
   Gaussian random walk.
3. **Structural birth/death** — time-varying offset driven by birth-rate,
   lagged QCEW BD proxy, and optional cyclical demand indicators.
4. **QCEW likelihood** — near-census truth anchor with estimated base noise
   by tier (M2 vs M3+M1) and per-observation revision multipliers.
5. **CES likelihood** — vintage-specific noise (1st print, 2nd print,
   final) with shared bias/loading parameters.
6. **Provider likelihoods** — config-driven loop supporting both iid and
   AR(1) measurement-error structures.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from .config import (
    CYCLICAL_INDICATORS,
    LOG_SIGMA_QCEW_BOUNDARY_MU,
    LOG_SIGMA_QCEW_BOUNDARY_SD,
    LOG_SIGMA_QCEW_MID_MU,
    LOG_SIGMA_QCEW_MID_SD,
    N_ERAS,
    N_HARMONICS,
)


def build_model(data: dict) -> pm.Model:
    """Build the QCEW-anchored PyMC model.

    Parameters
    ----------
    data : dict
        Output of :func:`alt_nfp.panel_adapter.panel_to_model_data`.

    Key changes from v2
    --------------------
    * Provider likelihoods are generated in a loop driven by
      ``data['pp_data']`` / ``ProviderConfig``.
    * Birth/death is time-varying:
      ``bd_t = φ₀ + φ₁·birth_rate_c + φ₂·bd_qcew_c + σ_bd·ξ_t``
    * QCEW observation noise: two estimated base sigmas (M2 vs M3+M1)
      times per-observation revision multiplier from ``data['qcew_noise_mult']``.
    """
    T = data["T"]
    month_of_year = data["month_of_year"]
    pp_data = data["pp_data"]

    with pm.Model() as model:

        # =============================================================
        # QCEW observation noise: estimated base by tier × revision mult
        # LogNormal avoids the funnel that HalfNormal creates when sigma
        # collapses toward zero (extreme QCEW precision → bimodality).
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
        qcew_is_m2 = pt.as_tensor_variable(
            np.asarray(data["qcew_is_m2"], dtype=bool)
        )
        base_sigma = pt.switch(
            qcew_is_m2, sigma_qcew_mid, sigma_qcew_boundary
        )
        qcew_sigma = base_sigma * pt.as_tensor_variable(
            data["qcew_noise_mult"], dtype=pt.dscalar
        )

        # =============================================================
        # Latent continuing-units growth: AR(1) with mean reversion
        # =============================================================

        era_idx = data.get("era_idx")

        sigma_g = pm.HalfNormal("sigma_g", sigma=0.005)
        eps_g = pm.Normal("eps_g", 0, 1, shape=T)

        if era_idx is not None:
            mu_g_era = pm.Normal("mu_g_era", mu=0.001, sigma=0.005, shape=N_ERAS)
            phi_raw_era = pm.Beta("phi_raw_era", alpha=18, beta=2, shape=N_ERAS)

            mu_g = mu_g_era[era_idx]       # (T,)
            phi = phi_raw_era[era_idx]     # (T,)

            g0 = mu_g[0] + sigma_g * eps_g[0]

            def ar1_step_era(e_t, mu_t, phi_t, g_prev, _sig):
                return mu_t + phi_t * (g_prev - mu_t) + _sig * e_t

            g_rest, _ = pytensor.scan(
                fn=ar1_step_era,
                sequences=[eps_g[1:], mu_g[1:], phi[1:]],
                outputs_info=[g0],
                non_sequences=[sigma_g],
                strict=True,
            )
        else:
            mu_g = pm.Normal("mu_g", mu=0.001, sigma=0.005)
            phi = pm.Uniform("phi", lower=0.0, upper=0.99)

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

        # GRW on Fourier coefficients: shape (2*K, n_years) so the walk is along
        # the year axis (PyMC GRW steps along the last dimension).
        # Rows 0..K-1 are A_k, rows K..2K-1 are B_k.
        sigma_vec = pt.tile(sigma_fourier, 2)  # (2K,)
        fourier_coefs = pm.GaussianRandomWalk(
            'fourier_coefs',
            sigma=sigma_vec,
            init_dist=pm.Normal.dist(0, 0.015),
            shape=(2 * K, n_years),
        )

        # Evaluate seasonal at each observation t
        k_vals = np.arange(1, K + 1)  # (K,)
        cos_basis = pt.cos(2 * np.pi * k_vals * month_of_year[:, None] / 12)  # (T, K)
        sin_basis = pt.sin(2 * np.pi * k_vals * month_of_year[:, None] / 12)  # (T, K)

        # fourier_coefs is (2*K, n_years); index by year then take .T for (T, K)
        A_t = fourier_coefs[:K, year_of_obs].T   # (T, K)
        B_t = fourier_coefs[K:, year_of_obs].T   # (T, K)

        s_t = pt.sum(A_t * cos_basis + B_t * sin_basis, axis=1)  # (T,)
        pm.Deterministic('seasonal', s_t)            # per-obs seasonal value
        pm.Deterministic('fourier_coefs_det', fourier_coefs.T)  # (n_years, 2K) for downstream

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
        sigma_bd = pm.HalfNormal("sigma_bd", sigma=0.001)

        xi_bd = pm.Normal("xi_bd", 0, 1, shape=T)

        bd_t = phi_0 + sigma_bd * xi_bd

        has_birth = np.any(data["birth_rate_c"] != 0.0)
        has_bd_qcew = np.any(data["bd_qcew_c"] != 0.0)
        if has_birth:
            phi_1 = pm.Normal("phi_1", mu=0.5, sigma=0.5)
            bd_t = bd_t + phi_1 * pt.as_tensor_variable(data["birth_rate_c"])
        if has_bd_qcew:
            phi_2 = pm.Normal("phi_2", mu=0.3, sigma=0.3)
            bd_t = bd_t + phi_2 * pt.as_tensor_variable(data["bd_qcew_c"])

        # Cyclical indicators (demand-side BD covariates)
        cyclical_keys = [f"{spec['name']}_c" for spec in CYCLICAL_INDICATORS]
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
            sigma=qcew_sigma,
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
            if len(obs_idx) == 0:
                continue  # skip providers with no observations (e.g. censored backtest)

            # Per-provider measurement parameters (sigma_pp_* to avoid collision with latent sigma_g)
            alpha_p = pm.Normal(f"alpha_{name}", 0, 0.005)
            lam_p = pm.Normal(f"lam_{name}", 1.0, 0.15)
            sigma_p = pm.InverseGamma(f"sigma_pp_{name}", alpha=3.0, beta=0.004)

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
