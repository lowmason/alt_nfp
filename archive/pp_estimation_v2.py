# -------------------------------------------------------------------------------------------------
# Growth-Rate State Space Model for NFP Nowcasting (v2)
#
# QCEW-anchored model:
#   - QCEW (near-census, monthly) is the truth reference (α=0, λ=1, fixed σ)
#     with quality-dependent σ: tight for quarter-end months (direct tax data),
#     looser for months 1-2 (retrospective UI tax records, not current-period)
#   - CES is a noisy observation with estimated α, λ, σ
#   - PP observes continuing-unit growth with estimated α, λ, σ
#   - Latent AR(1) process for SA continuing-units growth
#   - Explicit birth/death offset: identified by QCEW-vs-PP gap
#   - PP1 has AR(1) measurement error (multi-establishment clients)
#
# Designed for M4 Max: nutpie preferred, PyMC NUTS fallback
# Expected runtime: 2-5 minutes (4 chains, 4000 draws)
# -------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pymc as pm
import pytensor
import pytensor.tensor as pt
from scipy import stats as sp_stats

BASE_DIR = Path('/Users/lowell/Projects/pp')
OUTPUT_DIR = BASE_DIR / 'output'

N_PP = 2
PP_NAMES = ['PP1', 'PP2']


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    """Load and align CES (SA+NSA), QCEW, and PP data; compute growth rates."""

    ces = (
        pl.read_csv(f'{BASE_DIR}/data/ces_index.csv', try_parse_dates=True)
        .sort('ref_date')
    )
    qcew = (
        pl.read_csv(f'{BASE_DIR}/data/qcew_index.csv', try_parse_dates=True)
        .sort('ref_date')
    )
    pp1 = (
        pl.read_csv(f'{BASE_DIR}/data/pp_index_1.csv', try_parse_dates=True)
        .sort('ref_date')
    )
    pp2 = (
        pl.read_csv(f'{BASE_DIR}/data/pp_index_2.csv', try_parse_dates=True)
        .sort('ref_date')
    )

    # Monthly calendar spanning CES range
    cal = pl.DataFrame({
        'ref_date': pl.date_range(
            ces['ref_date'].min(),
            ces['ref_date'].max(),
            interval='1mo',
            eager=True,
        )
    })

    # Join all series onto calendar
    levels = (
        cal
        .join(ces, on='ref_date', how='left')
        .join(qcew, on='ref_date', how='left')
        .join(pp1, on='ref_date', how='left')
        .join(pp2, on='ref_date', how='left')
        .sort('ref_date')
    )

    # Monthly log growth rates (first-difference of log index)
    growth = (
        levels
        .with_columns([
            pl.col('ces_sa_index').log().diff().alias('g_ces_sa'),
            pl.col('ces_nsa_index').log().diff().alias('g_ces_nsa'),
            pl.col('qcew_nsa_index').log().diff().alias('g_qcew'),
            pl.col('pp_index_1').log().diff().alias('g_pp_1'),
            pl.col('pp_index_2_0').log().diff().alias('g_pp_2'),
        ])
        .slice(1)  # first row has no growth
    )

    dates = growth['ref_date'].to_list()
    T = len(dates)
    month_of_year = np.array([d.month - 1 for d in dates], dtype=int)

    # CES SA (monthly)
    g_ces_sa = growth['g_ces_sa'].to_numpy().astype(float)
    ces_sa_obs = np.where(np.isfinite(g_ces_sa))[0]

    # CES NSA (monthly)
    g_ces_nsa = growth['g_ces_nsa'].to_numpy().astype(float)
    ces_nsa_obs = np.where(np.isfinite(g_ces_nsa))[0]

    # PP (NSA, monthly: PP1 from provider 1, PP2 = pp_index_2_0 from provider 2)
    # PP2_1 and PP2_2 dropped: 0.999+ correlated with PP2_0, triple-counting info.
    pp_cols = ['g_pp_1', 'g_pp_2']
    g_pp = np.vstack([
        growth[c].to_numpy().astype(float) for c in pp_cols
    ])  # (2, T)
    pp_obs = [np.where(np.isfinite(g_pp[p]))[0] for p in range(N_PP)]

    # QCEW (NSA, monthly): anchor series
    # Month-3 of each quarter (Mar/Jun/Sep/Dec) is direct from current-period tax filings;
    # months 1-2 come from UI tax records reported retrospectively.
    g_qcew = growth['g_qcew'].to_numpy().astype(float)
    qcew_obs = np.where(np.isfinite(g_qcew))[0]
    # Boolean mask: True for quarter-end months (higher quality)
    qcew_is_m3 = np.array([dates[i].month in (3, 6, 9, 12) for i in qcew_obs])
    n_qcew_m3 = int(qcew_is_m3.sum())
    n_qcew_m12 = len(qcew_obs) - n_qcew_m3

    print(f'Growth-rate model: T = {T} months ({dates[0]} → {dates[-1]})')
    print(f'  CES SA:  {len(ces_sa_obs)} monthly obs')
    print(f'  CES NSA: {len(ces_nsa_obs)} monthly obs')
    print(f'  PP:      {[len(pp_obs[p]) for p in range(N_PP)]} obs per series '
          f'({", ".join(PP_NAMES)})')
    print(f'  QCEW:    {len(qcew_obs)} monthly obs (NSA) — '
          f'{n_qcew_m3} quarter-end (M3), {n_qcew_m12} retrospective UI (M1-2)')
    print(f'  CES SA  mean growth: {np.nanmean(g_ces_sa)*100:+.4f}%/mo')
    print(f'  CES NSA mean growth: {np.nanmean(g_ces_nsa)*100:+.4f}%/mo')
    print(f'  QCEW    mean growth: {np.nanmean(g_qcew)*100:+.4f}%/mo')
    for p in range(N_PP):
        print(f'  {PP_NAMES[p]:5} mean growth: '
              f'{np.nanmean(g_pp[p])*100:+.4f}%/mo')

    return dict(
        levels=levels,
        dates=dates,
        T=T,
        month_of_year=month_of_year,
        g_ces_sa=g_ces_sa,
        ces_sa_obs=ces_sa_obs,
        g_ces_nsa=g_ces_nsa,
        ces_nsa_obs=ces_nsa_obs,
        g_pp=g_pp,
        pp_obs=pp_obs,
        g_qcew=g_qcew,
        qcew_obs=qcew_obs,
        qcew_is_m3=qcew_is_m3,
    )


# =============================================================================
# Model specification and sampling
# =============================================================================

def build_model(data: dict) -> pm.Model:
    """Build the QCEW-anchored PyMC model."""

    T = data['T']
    month_of_year = data['month_of_year']
    g_ces_sa = data['g_ces_sa']
    ces_sa_obs = data['ces_sa_obs']
    g_ces_nsa = data['g_ces_nsa']
    ces_nsa_obs = data['ces_nsa_obs']
    g_pp = data['g_pp']
    pp_obs = data['pp_obs']
    g_qcew = data['g_qcew']
    qcew_obs = data['qcew_obs']
    qcew_is_m3 = data['qcew_is_m3']

    # QCEW fixed sigmas: quarter-end months are direct from current-period
    # tax filings (tight); months 1-2 come from retrospective UI tax records
    # (looser — only the current M3 is reported for the active reference period).
    SIGMA_QCEW_M3 = 0.0005   # ~0.05%/mo — near-census, current-period tax filing
    SIGMA_QCEW_M12 = 0.0015  # ~0.15%/mo — retrospective UI months

    qcew_sigma_fixed = np.where(qcew_is_m3, SIGMA_QCEW_M3, SIGMA_QCEW_M12)

    with pm.Model() as model:

        # =============================================================
        # Latent continuing-units growth: AR(1) with mean reversion
        # =============================================================

        mu_g = pm.Normal('mu_g', mu=0.001, sigma=0.005)
        phi = pm.Uniform('phi', lower=0.0, upper=0.99)
        sigma_g = pm.HalfNormal('sigma_g', sigma=0.008)

        eps_g = pm.Normal('eps_g', 0, 1, shape=T)
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
        pm.Deterministic('g_cont', g_cont)

        # =============================================================
        # Monthly seasonal (sum-to-zero, 11 free parameters)
        # =============================================================

        s_raw = pm.Normal('s_raw', 0, 0.015, shape=11)
        s_all = pt.concatenate([s_raw, (-pt.sum(s_raw)).reshape((1,))])
        s_t = s_all[month_of_year]
        pm.Deterministic('seasonal', s_all)

        # =============================================================
        # Birth/death offset — fixed (gold-standard specification)
        # Single scalar captures mean gap between total employment
        # (CES/QCEW) and continuing-units (PP). Time-varying bd
        # is available in extended variants but breaks nutpie.
        # =============================================================

        bd = pm.Normal('bd', mu=0.001, sigma=0.002)

        # =============================================================
        # Composite growth signals
        # =============================================================

        g_cont_nsa = g_cont + s_t
        g_total_sa = g_cont + bd       # bd is scalar (gold standard)
        g_total_nsa = g_cont + s_t + bd

        pm.Deterministic('g_total_sa', g_total_sa)
        pm.Deterministic('g_total_nsa', g_total_nsa)

        # =============================================================
        # QCEW likelihood — TRUTH ANCHOR
        # Monthly NSA total employment growth (α=0, λ=1, fixed σ).
        # σ varies by month-in-quarter: tight for M3 (current-period tax filings),
        # looser for M1-2 (retrospective UI tax records).
        # =============================================================

        pm.Normal(
            'obs_qcew',
            mu=g_total_nsa[qcew_obs],
            sigma=qcew_sigma_fixed,
            observed=g_qcew[qcew_obs],
        )

        # =============================================================
        # CES likelihood — noisy observation with estimated params
        # CES measures total employment (SA and NSA) with possible
        # bias (alpha_ces) and scale drift (lambda_ces).
        # =============================================================

        alpha_ces = pm.Normal('alpha_ces', 0, 0.005)
        lambda_ces = pm.Normal('lambda_ces', 1.0, 0.25)
        # Separate SA/NSA noise — InverseGamma(3, 0.004) has mode ≈ 0.001
        sigma_ces_sa = pm.InverseGamma('sigma_ces_sa', alpha=3.0, beta=0.004)
        sigma_ces_nsa = pm.InverseGamma('sigma_ces_nsa', alpha=3.0, beta=0.004)

        pm.Normal(
            'obs_ces_sa',
            mu=alpha_ces + lambda_ces * g_total_sa[ces_sa_obs],
            sigma=sigma_ces_sa,
            observed=g_ces_sa[ces_sa_obs],
        )

        pm.Normal(
            'obs_ces_nsa',
            mu=alpha_ces + lambda_ces * g_total_nsa[ces_nsa_obs],
            sigma=sigma_ces_nsa,
            observed=g_ces_nsa[ces_nsa_obs],
        )

        # =============================================================
        # PP likelihood (NSA continuing-units, independent providers)
        # No common factor: PP1 and PP2 are from different providers
        # with independent measurement systems.  Cross-provider
        # correlation is fully explained by the latent signal.
        # =============================================================

        alpha_pp = pm.Normal('alpha_pp', 0, 0.005, shape=N_PP)
        lam_pp = pm.Normal('lam_pp', 1.0, 0.25, shape=N_PP)
        # InverseGamma keeps sigma_pp away from zero (same funnel fix).
        # Mode ≈ 0.001, allows data to pull it larger for noisier series.
        sigma_pp = pm.InverseGamma('sigma_pp', alpha=3.0, beta=0.004, shape=N_PP)

        # --- PP1: AR(1) measurement error (gold standard: Gaussian) ---
        # PP1 clients are multi-establishment firms, so within-client
        # restructuring creates autocorrelated residuals:
        #   u[t] = rho * u[t-1] + eps[t],  eps ~ N(0, sigma_pp[0])
        rho_pp1 = pm.Uniform('rho_pp1', lower=0.0, upper=0.99)

        m_pp1 = pp_obs[0]
        mu_base_pp1 = alpha_pp[0] + lam_pp[0] * g_cont_nsa[m_pp1]
        y_pp1 = g_pp[0, m_pp1]  # observed (numpy)

        # Conditional mean:  mu_base[t] + rho * (y[t-1] - mu_base[t-1])
        mu_cond_pp1 = pt.concatenate([
            mu_base_pp1[:1],
            mu_base_pp1[1:] + rho_pp1 * (
                pt.as_tensor_variable(y_pp1[:-1]) - mu_base_pp1[:-1]
            ),
        ])
        # Conditional std: sigma/sqrt(1-rho²) for t=0, sigma for t>0
        sigma_cond_pp1 = pt.concatenate([
            (sigma_pp[0] / pt.sqrt(1.0 - rho_pp1 ** 2)).reshape((1,)),
            pt.ones(len(m_pp1) - 1) * sigma_pp[0],
        ])
        pm.Normal(
            f'obs_{PP_NAMES[0].lower()}',
            mu=mu_cond_pp1,
            sigma=sigma_cond_pp1,
            observed=y_pp1,
        )

        # --- PP2: iid measurement error ---
        m_pp2 = pp_obs[1]
        mu_pp2 = alpha_pp[1] + lam_pp[1] * g_cont_nsa[m_pp2]
        pm.Normal(
            'obs_pp2',
            mu=mu_pp2,
            sigma=sigma_pp[1],
            observed=g_pp[1, m_pp2],
        )

    return model


# =============================================================================
# Sampling
# =============================================================================

def sample_model(model: pm.Model) -> az.InferenceData:
    """Sample from the model using nutpie (preferred) or PyMC NUTS."""

    with model:
        sampler_kwargs = dict(
            draws=8000,
            tune=6000,
            chains=4,
            target_accept=0.97,
            return_inferencedata=True,
        )

        try:
            idata = pm.sample(
                nuts_sampler='nutpie', **sampler_kwargs
            )
            sampler_used = 'nutpie'
        except Exception as e:
            print(
                f'nutpie unavailable ({e}), '
                'falling back to PyMC NUTS'
            )
            idata = pm.sample(**sampler_kwargs)
            sampler_used = 'pymc'

    print(f'\nSampling complete ({sampler_used})')
    return idata


# =============================================================================
# Diagnostics
# =============================================================================

def print_diagnostics(idata: az.InferenceData, data: dict):
    """Print sampling diagnostics and parameter summary."""

    print('=' * 72)
    print('SAMPLING DIAGNOSTICS')
    print('=' * 72)

    divs = int(idata.sample_stats.diverging.sum().values)
    print(f'Divergences: {divs}')
    try:
        max_td = int(idata.sample_stats.tree_depth.max().values)
        print(f'Max tree depth: {max_td}')
    except Exception:
        pass

    var_names = [
        'mu_g', 'phi', 'sigma_g', 'bd',
        'alpha_ces', 'lambda_ces', 'sigma_ces_sa', 'sigma_ces_nsa',
        'rho_pp1',
        'alpha_pp', 'lam_pp', 'sigma_pp',
    ]

    print('\n' + '=' * 72)
    print('PARAMETER SUMMARY')
    print('=' * 72)
    summary = az.summary(idata, var_names=var_names, hdi_prob=0.80)
    print(summary.to_string())

    # --- Convergence warnings ---
    rhat_bad = summary[summary['r_hat'] > 1.01]
    ess_bad = summary[summary['ess_bulk'] < 400]
    if len(rhat_bad) > 0:
        print('\n** WARNING: Parameters with R-hat > 1.01:')
        for name, row in rhat_bad.iterrows():
            print(f'    {name}: R-hat = {row["r_hat"]:.4f}')
    if len(ess_bad) > 0:
        print('\n** WARNING: Parameters with ESS_bulk < 400:')
        for name, row in ess_bad.iterrows():
            print(
                f'    {name}: ESS = {row["ess_bulk"]:.0f}'
            )
    if len(rhat_bad) == 0 and len(ess_bad) == 0:
        print(
            '\nAll parameters converged '
            '(R-hat <= 1.01, ESS_bulk >= 400)'
        )

    # bd is scalar (gold standard): shape (chains, draws)
    bd_post = idata.posterior['bd'].values.flatten()
    lam_post = idata.posterior['lam_pp'].values
    alpha_post = idata.posterior['alpha_pp'].values

    print('\n' + '=' * 72)
    print('KEY OUTPUTS')
    print('=' * 72)

    print(f'\nBirth/Death offset (monthly SA):')
    print(f'  Mean:       {bd_post.mean() * 100:+.4f}%/mo')
    print(f'  80% HDI:    [{np.percentile(bd_post, 10) * 100:+.4f}%, '
          f'{np.percentile(bd_post, 90) * 100:+.4f}%]')
    print(f'  Annualized: {bd_post.mean() * 12 * 100:+.2f}%')

    # CES parameters
    alpha_ces_post = idata.posterior['alpha_ces'].values.flatten()
    lambda_ces_post = idata.posterior['lambda_ces'].values.flatten()
    sigma_ces_sa_post = idata.posterior['sigma_ces_sa'].values.flatten()
    sigma_ces_nsa_post = idata.posterior['sigma_ces_nsa'].values.flatten()
    print(f'\nCES observation parameters (vs QCEW anchor):')
    print(f'  α_ces  = {alpha_ces_post.mean() * 100:+.4f}%/mo  '
          f'(bias relative to true growth)')
    print(f'  λ_ces  = {lambda_ces_post.mean():.4f}  '
          f'[{np.percentile(lambda_ces_post, 10):.4f}, '
          f'{np.percentile(lambda_ces_post, 90):.4f}]')
    print(f'  σ_ces_sa  = {sigma_ces_sa_post.mean() * 100:.3f}%  '
          f'σ_ces_nsa = {sigma_ces_nsa_post.mean() * 100:.3f}%')

    rho_pp1_post = idata.posterior['rho_pp1'].values.flatten()
    print(f'\nPP1 AR(1) autocorrelation (ρ):')
    print(f'  Mean: {rho_pp1_post.mean():.3f}  '
          f'80% HDI: [{np.percentile(rho_pp1_post, 10):.3f}, '
          f'{np.percentile(rho_pp1_post, 90):.3f}]')

    print(f'\nPP signal loadings (λ) and bias (α):')
    for p in range(N_PP):
        lam_p = lam_post[:, :, p].flatten()
        alp_p = alpha_post[:, :, p].flatten()
        extra = ''
        if p == 0:
            # PP1: report marginal σ (accounting for AR(1))
            sigma_pp0 = idata.posterior['sigma_pp'].values[:, :, 0].flatten()
            marg_sigma = sigma_pp0.mean() / np.sqrt(1 - rho_pp1_post.mean() ** 2)
            extra = f'  |  σ_marg = {marg_sigma * 100:.3f}%'
        print(f'  {PP_NAMES[p]:5}: λ = {lam_p.mean():.3f} '
              f'[{np.percentile(lam_p, 10):.3f}, {np.percentile(lam_p, 90):.3f}]'
              f'  |  α = {alp_p.mean() * 100:+.4f}%{extra}')


# =============================================================================
# Source contribution analysis
# =============================================================================

def print_source_contributions(idata: az.InferenceData, data: dict):
    """Quantify how much each data source contributes to the latent estimate.

    For each observation, the model combines information weighted by
    *precision* (1 / σ²).  For CES and PP, the effective precision on
    the latent growth is  λ² / σ²  because they observe a scaled version.

    QCEW is the anchor (α=0, λ=1) with fixed σ that varies by
    quarter position — so its precision is 1/σ²_fixed.

    We compute the total precision each source injects across all its
    observations and express the relative share as a percentage.
    """

    sigma_ces_sa  = idata.posterior['sigma_ces_sa'].values.flatten().mean()
    sigma_ces_nsa = idata.posterior['sigma_ces_nsa'].values.flatten().mean()
    lambda_ces    = idata.posterior['lambda_ces'].values.flatten().mean()
    sigma_pp   = idata.posterior['sigma_pp'].values.mean(axis=(0, 1))   # (N_PP,)
    lam_pp     = idata.posterior['lam_pp'].values.mean(axis=(0, 1))     # (N_PP,)
    rho_pp1    = idata.posterior['rho_pp1'].values.flatten().mean()

    pp_obs   = data['pp_obs']       # list of integer index arrays
    qcew_obs = data['qcew_obs']
    qcew_is_m3 = data['qcew_is_m3']

    SIGMA_QCEW_M3 = 0.0005
    SIGMA_QCEW_M12 = 0.0015

    # ---- Per-observation precision ----
    # CES: effective precision on latent = λ_ces² / σ_ces²
    prec_ces_sa  = lambda_ces ** 2 / sigma_ces_sa ** 2
    prec_ces_nsa = lambda_ces ** 2 / sigma_ces_nsa ** 2

    # QCEW: anchor with fixed sigma — precision varies by month quality
    prec_qcew_m3  = 1.0 / SIGMA_QCEW_M3 ** 2
    prec_qcew_m12 = 1.0 / SIGMA_QCEW_M12 ** 2

    prec_pp = lam_pp ** 2 / sigma_pp ** 2  # per month, shape (N_PP,)
    # PP1: AR(1) autocorrelation reduces effective independent info per obs
    prec_pp[0] = lam_pp[0] ** 2 * (1 - rho_pp1 ** 2) / sigma_pp[0] ** 2

    # ---- Total precision budget (sum across all months) ----
    n_ces_sa  = len(data['ces_sa_obs'])
    n_ces_nsa = len(data['ces_nsa_obs'])
    n_qcew_m3  = int(qcew_is_m3.sum())
    n_qcew_m12 = len(qcew_obs) - n_qcew_m3

    total_prec_ces_sa  = prec_ces_sa  * n_ces_sa
    total_prec_ces_nsa = prec_ces_nsa * n_ces_nsa
    total_prec_qcew    = prec_qcew_m3 * n_qcew_m3 + prec_qcew_m12 * n_qcew_m12
    n_pp_obs = [len(pp_obs[p]) for p in range(N_PP)]
    total_prec_pp = np.array([prec_pp[p] * n_pp_obs[p] for p in range(N_PP)])

    total_all = (total_prec_ces_sa + total_prec_ces_nsa + total_prec_qcew
                 + total_prec_pp.sum())

    # ---- Display ----
    print('\n' + '=' * 72)
    print('DATA SOURCE CONTRIBUTION (precision-weighted)')
    print('=' * 72)
    print(f'{"Source":<16} {"Obs":<8} {"Prec/obs":>12} '
          f'{"Total prec":>14} {"Share":>8}')
    print('-' * 72)

    def _row(name, n, prec_per, total):
        pct = 100.0 * total / total_all
        print(f'{name:<16} {n:<8} {prec_per:>12,.0f} {total:>14,.0f} {pct:>7.1f}%')

    _row('CES SA',  n_ces_sa,  prec_ces_sa,  total_prec_ces_sa)
    _row('CES NSA', n_ces_nsa, prec_ces_nsa, total_prec_ces_nsa)

    for p in range(N_PP):
        _row(PP_NAMES[p], n_pp_obs[p], prec_pp[p], total_prec_pp[p])

    # QCEW split by quality tier
    _row('QCEW (M3)',  n_qcew_m3,  prec_qcew_m3,  prec_qcew_m3 * n_qcew_m3)
    _row('QCEW (M1-2)', n_qcew_m12, prec_qcew_m12, prec_qcew_m12 * n_qcew_m12)

    print('-' * 72)
    print(f'{"TOTAL":<16} {"":8} {"":12} {total_all:>14,.0f} {"100.0%":>8}')

    # ---- Per-month interpretation ----
    print(f'\nInterpretation (QCEW-anchored model):')
    print(f'  QCEW: fixed σ = {SIGMA_QCEW_M3*100:.2f}% (M3) / '
          f'{SIGMA_QCEW_M12*100:.2f}% (M1-2)  — truth anchor')
    print(f'  CES:  α = {idata.posterior["alpha_ces"].values.flatten().mean()*100:+.4f}%,  '
          f'λ = {lambda_ces:.4f},  '
          f'σ_SA = {sigma_ces_sa*100:.3f}%,  σ_NSA = {sigma_ces_nsa*100:.3f}%  (estimated)')
    print(f'  PP series contribute through λ²/σ² — a large loading λ ≈ '
          f'{lam_pp.mean():.2f}')
    print(f'  amplifies their information content despite higher noise.')
    print(f'  PP1 precision is discounted by (1-ρ²) for AR(1) autocorrelation'
          f' (ρ={rho_pp1:.3f}).')


# =============================================================================
# Prior predictive checks
# =============================================================================

def run_prior_predictive_checks(
    model: pm.Model, data: dict
) -> az.InferenceData | None:
    """Sample from the prior predictive and visualize.

    Validates that priors produce data in a plausible range before
    fitting (Gabry et al. 2019, Section 3).
    """

    print('Sampling prior predictive...')
    try:
        with model:
            prior_idata = pm.sample_prior_predictive(samples=500)
    except Exception as e:
        print(f'Prior predictive sampling failed: {e}')
        return None

    dates = data['dates']

    obs_sources = {
        'obs_ces_sa': (
            'CES SA', data['g_ces_sa'][data['ces_sa_obs']]
        ),
        'obs_ces_nsa': (
            'CES NSA', data['g_ces_nsa'][data['ces_nsa_obs']]
        ),
        'obs_qcew': (
            'QCEW', data['g_qcew'][data['qcew_obs']]
        ),
        f'obs_{PP_NAMES[0].lower()}': (
            'PP1', data['g_pp'][0, data['pp_obs'][0]]
        ),
        'obs_pp2': (
            'PP2', data['g_pp'][1, data['pp_obs'][1]]
        ),
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for idx, (vn, (label, obs)) in enumerate(
        obs_sources.items()
    ):
        ax = axes_flat[idx]
        pp = prior_idata.prior_predictive[vn].values.flatten()

        # Clip to 1st-99th percentile for readable histograms
        lo, hi = np.percentile(pp, [1, 99])
        pp_clip = pp[(pp >= lo) & (pp <= hi)]

        ax.hist(
            pp_clip * 100, bins=80, density=True,
            alpha=0.4, color='steelblue',
            label='Prior predictive',
        )
        ax.hist(
            obs * 100, bins=40, density=True,
            alpha=0.6, color='darkorange', label='Observed',
        )
        ax.set_xlabel('Monthly growth (%)')
        ax.set_title(label)
        ax.legend(fontsize=7)

    # Last panel: prior g_cont trajectories
    ax = axes_flat[5]
    g_prior = prior_idata.prior['g_cont'].values
    n_show = min(20, g_prior.shape[1])
    for i in range(n_show):
        ax.plot(
            dates, g_prior[0, i, :] * 100,
            alpha=0.3, lw=0.8, color='steelblue',
        )
    ax.plot(
        dates, data['g_ces_sa'] * 100,
        'darkorange', lw=1.5, alpha=0.8,
        label='CES SA (observed)',
    )
    ax.set_ylabel('Growth (%/mo)')
    ax.set_title('Prior draws: latent g_cont')
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.suptitle(
        'Prior Predictive Check: '
        'Do priors generate plausible data?',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    fig.savefig(
        OUTPUT_DIR / 'prior_predictive.png',
        dpi=150, bbox_inches='tight',
    )
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / "prior_predictive.png"}')

    # Summary statistics
    print('\nPrior predictive summary (growth %, monthly):')
    for vn, (label, obs) in obs_sources.items():
        pp = (
            prior_idata.prior_predictive[vn]
            .values.flatten() * 100
        )
        lo5 = np.percentile(pp, 5)
        hi95 = np.percentile(pp, 95)
        print(
            f'  {label:10}: prior 90% '
            f'[{lo5:+.2f}, {hi95:+.2f}]  '
            f'| obs mean: {obs.mean() * 100:+.4f}%'
        )

    return prior_idata


# =============================================================================
# Posterior predictive checks
# =============================================================================

def run_posterior_predictive_checks(
    model: pm.Model,
    idata: az.InferenceData,
    data: dict,
):
    """Posterior predictive checks with density overlays and
    test statistics.

    Generates replicated datasets from the fitted model and
    compares to observed data (Gabry et al. 2019, Section 5).
    """

    print('Sampling posterior predictive...')
    with model:
        pm.sample_posterior_predictive(
            idata, extend_inferencedata=True,
            random_seed=42,
        )

    obs_sources = {
        'obs_ces_sa': (
            'CES SA', data['g_ces_sa'][data['ces_sa_obs']]
        ),
        'obs_ces_nsa': (
            'CES NSA', data['g_ces_nsa'][data['ces_nsa_obs']]
        ),
        'obs_qcew': (
            'QCEW', data['g_qcew'][data['qcew_obs']]
        ),
        f'obs_{PP_NAMES[0].lower()}': (
            'PP1', data['g_pp'][0, data['pp_obs'][0]]
        ),
        'obs_pp2': (
            'PP2', data['g_pp'][1, data['pp_obs'][1]]
        ),
    }

    rng = np.random.default_rng(42)

    # ---- Plot 1: Density overlays (replicated vs observed) ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for idx, (vn, (label, obs)) in enumerate(
        obs_sources.items()
    ):
        ax = axes_flat[idx]
        pp = idata.posterior_predictive[vn].values
        n_ch, n_dr, n_obs = pp.shape
        pp_flat = pp.reshape(-1, n_obs)
        n_total = pp_flat.shape[0]

        # Subsample 100 replicated datasets for plotting
        sub_idx = rng.choice(
            n_total, size=min(100, n_total), replace=False
        )

        # Bin range from observed data with padding
        obs_pct = obs * 100
        pad = 0.5 * np.ptp(obs_pct)
        x_lo = obs_pct.min() - pad
        x_hi = obs_pct.max() + pad
        bins = np.linspace(x_lo, x_hi, 60)

        for i in sub_idx:
            ax.hist(
                pp_flat[i] * 100, bins=bins,
                density=True, histtype='step',
                alpha=0.08, color='steelblue', lw=0.5,
            )
        ax.hist(
            obs_pct, bins=bins, density=True,
            histtype='step', color='black', lw=2,
            label='Observed',
        )
        ax.set_xlabel('Monthly growth (%)')
        ax.set_title(label)
        ax.legend(fontsize=7)

    axes_flat[5].set_visible(False)
    fig.suptitle(
        'Posterior Predictive Check: Density Overlays',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    fig.savefig(
        OUTPUT_DIR / 'ppc_density.png',
        dpi=150, bbox_inches='tight',
    )
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / "ppc_density.png"}')

    # ---- Plot 2: Test statistics ----
    # Choose statistics orthogonal to Gaussian location-scale
    # parameters for informative checks.

    def _lag1_acf(x):
        """Lag-1 autocorrelation."""
        if len(x) < 3:
            return 0.0
        xc = x - x.mean()
        c0 = np.dot(xc, xc)
        return (
            np.dot(xc[:-1], xc[1:]) / c0 if c0 > 0 else 0.0
        )

    stat_fns = {
        'Skewness': lambda x: sp_stats.skew(x),
        'Lag-1 ACF': _lag1_acf,
    }

    n_src = len(obs_sources)
    n_stat = len(stat_fns)
    fig, axes = plt.subplots(
        n_src, n_stat,
        figsize=(5 * n_stat, 2.5 * n_src),
    )

    for row, (vn, (label, obs)) in enumerate(
        obs_sources.items()
    ):
        pp = idata.posterior_predictive[vn].values
        pp_flat = pp.reshape(-1, pp.shape[-1])
        # Subsample 500 draws for efficiency
        n_sub = min(500, pp_flat.shape[0])
        sub_idx = rng.choice(
            pp_flat.shape[0], size=n_sub, replace=False
        )

        for col, (sname, sfn) in enumerate(
            stat_fns.items()
        ):
            ax = axes[row, col]
            rep_vals = np.array(
                [sfn(pp_flat[i]) for i in sub_idx]
            )
            obs_val = sfn(obs)

            ax.hist(
                rep_vals, bins=50, density=True,
                alpha=0.5, color='steelblue',
            )
            ax.axvline(
                obs_val, color='black', lw=2, ls='--',
                label=f'Obs: {obs_val:.3f}',
            )
            # Two-sided Bayesian p-value
            p = np.mean(rep_vals >= obs_val)
            p = min(p, 1 - p)
            ax.set_title(
                f'{label}: {sname} (p={p:.3f})',
                fontsize=9,
            )
            ax.legend(fontsize=6)

    fig.suptitle(
        'Posterior Predictive Test Statistics',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    fig.savefig(
        OUTPUT_DIR / 'ppc_test_stats.png',
        dpi=150, bbox_inches='tight',
    )
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / "ppc_test_stats.png"}')


# =============================================================================
# LOO-CV diagnostics
# =============================================================================

def run_loo_cv(
    model: pm.Model,
    idata: az.InferenceData,
    data: dict,
):
    """Compute LOO-CV per observation source and visualize
    k-hat diagnostics.

    Large k-hat values flag influential observations where
    full-data and LOO posteriors diverge substantially.
    """

    # Ensure log_likelihood is available
    if not hasattr(idata, 'log_likelihood'):
        print('Computing log-likelihood for LOO-CV...')
        try:
            with model:
                idata.extend(
                    pm.compute_log_likelihood(idata)
                )
        except Exception as e:
            print(f'Could not compute log-likelihood: {e}')
            print('Skipping LOO-CV.')
            return

    obs_sources = [
        ('obs_ces_sa', 'CES SA'),
        ('obs_ces_nsa', 'CES NSA'),
        ('obs_qcew', 'QCEW'),
        (f'obs_{PP_NAMES[0].lower()}', 'PP1'),
        ('obs_pp2', 'PP2'),
    ]

    print('\n' + '=' * 72)
    print('LEAVE-ONE-OUT CROSS-VALIDATION (PSIS-LOO)')
    print('=' * 72)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for idx, (var_name, label) in enumerate(obs_sources):
        ax = axes_flat[idx]
        try:
            loo_result = az.loo(
                idata, var_name=var_name, pointwise=True
            )
            elpd = loo_result.elpd_loo
            se = loo_result.se
            p_loo = loo_result.p_loo
            khat = np.asarray(loo_result.pareto_k)

            n_high = int(np.sum(khat > 0.7))
            n_warn = int(
                np.sum((khat > 0.5) & (khat <= 0.7))
            )

            print(f'\n{label} ({var_name}):')
            print(f'  ELPD LOO: {elpd:.1f} +/- {se:.1f}')
            print(f'  p_loo:    {p_loo:.1f}')
            print(f'  k-hat > 0.7 (bad):  {n_high}')
            print(f'  k-hat > 0.5 (warn): {n_warn}')

            # Color code k-hat values
            colors = np.where(
                khat > 0.7, 'red',
                np.where(khat > 0.5, 'orange', 'steelblue')
            )
            ax.scatter(
                range(len(khat)), khat,
                s=8, c=colors, alpha=0.6,
            )
            ax.axhline(
                0.7, color='red', ls='--', lw=1,
                alpha=0.7, label='k-hat = 0.7',
            )
            ax.axhline(
                0.5, color='orange', ls='--', lw=1,
                alpha=0.7, label='k-hat = 0.5',
            )
            ax.set_xlabel('Observation index')
            ax.set_ylabel('k-hat')
            ax.set_title(f'{label}: PSIS-LOO k-hat')
            ax.legend(fontsize=7)

        except Exception as e:
            print(f'\n{label}: LOO-CV failed -- {e}')
            ax.set_visible(False)

    axes_flat[5].set_visible(False)
    fig.suptitle(
        'LOO-CV k-hat Diagnostics '
        '(Pareto shape parameter)',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    fig.savefig(
        OUTPUT_DIR / 'loo_khat.png',
        dpi=150, bbox_inches='tight',
    )
    plt.close(fig)
    print(f'\nSaved: {OUTPUT_DIR / "loo_khat.png"}')


# =============================================================================
# Divergence visualization
# =============================================================================

def plot_divergences(idata: az.InferenceData):
    """Visualize divergent transitions in parameter space.

    Per Gabry et al. (2019): concentrated divergences indicate
    geometric pathology; scattered divergences are likely false
    positives (increase adapt_delta).
    """

    diverging = idata.sample_stats.diverging.values
    n_divs = int(diverging.sum())

    if n_divs == 0:
        print(
            '\nNo divergent transitions -- '
            'skipping divergence plot.'
        )
        return

    print(f'\nPlotting {n_divs} divergent transitions...')
    div_flat = diverging.flatten().astype(bool)

    # Parameter pairs to check for geometric pathologies
    pairs = [
        ('phi', 'sigma_g', None, None),
        ('rho_pp1', 'sigma_pp', None, 0),
        ('lambda_ces', 'alpha_ces', None, None),
        ('bd', 'phi', None, None),
    ]
    pair_labels = [
        ('phi', 'sigma_g'),
        ('rho_PP1', 'sigma_PP[0]'),
        ('lambda_CES', 'alpha_CES'),
        ('bd', 'phi'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, ((p1, p2, i1, i2), (l1, l2)) in enumerate(
        zip(pairs, pair_labels)
    ):
        ax = axes.flatten()[idx]

        v1 = idata.posterior[p1].values
        if i1 is not None:
            v1 = v1[:, :, i1]
        v1 = v1.flatten()

        v2 = idata.posterior[p2].values
        if i2 is not None:
            v2 = v2[:, :, i2]
        v2 = v2.flatten()

        ax.scatter(
            v1[~div_flat], v2[~div_flat],
            s=2, alpha=0.1, c='steelblue', rasterized=True,
        )
        ax.scatter(
            v1[div_flat], v2[div_flat],
            s=15, alpha=0.8, c='limegreen',
            edgecolors='darkgreen', lw=0.5,
            label=f'Divergent (n={n_divs})', zorder=10,
        )
        ax.set_xlabel(l1)
        ax.set_ylabel(l2)
        ax.legend(fontsize=8)

    fig.suptitle(
        'Divergence Diagnostics: Bivariate Scatterplots',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    fig.savefig(
        OUTPUT_DIR / 'divergences.png',
        dpi=150, bbox_inches='tight',
    )
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / "divergences.png"}')


# =============================================================================
# Residual analysis
# =============================================================================

def plot_residuals(idata: az.InferenceData, data: dict):
    """Plot standardized residuals per data source over time.

    Residuals should be approximately iid N(0,1) if the model
    is well-specified. Temporal patterns or heavy tails indicate
    model misfit.
    """

    dates = data['dates']
    dates_arr = np.array(dates)

    # Posterior means of latent states and parameters
    g_cont = idata.posterior['g_cont'].values.mean(axis=(0, 1))
    g_total_sa = (
        idata.posterior['g_total_sa'].values.mean(axis=(0, 1))
    )
    g_total_nsa = (
        idata.posterior['g_total_nsa'].values.mean(axis=(0, 1))
    )
    seasonal = (
        idata.posterior['seasonal'].values.mean(axis=(0, 1))
    )
    month_of_year = data['month_of_year']
    g_cont_nsa = g_cont + seasonal[month_of_year]

    alpha_ces = (
        idata.posterior['alpha_ces'].values.flatten().mean()
    )
    lambda_ces = (
        idata.posterior['lambda_ces'].values.flatten().mean()
    )
    sigma_ces_sa = (
        idata.posterior['sigma_ces_sa'].values.flatten().mean()
    )
    sigma_ces_nsa = (
        idata.posterior['sigma_ces_nsa'].values.flatten().mean()
    )
    alpha_pp = (
        idata.posterior['alpha_pp'].values.mean(axis=(0, 1))
    )
    lam_pp = (
        idata.posterior['lam_pp'].values.mean(axis=(0, 1))
    )
    sigma_pp = (
        idata.posterior['sigma_pp'].values.mean(axis=(0, 1))
    )
    rho_pp1 = (
        idata.posterior['rho_pp1'].values.flatten().mean()
    )

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    # --- CES SA ---
    ax = axes[0]
    idx_obs = data['ces_sa_obs']
    pred = alpha_ces + lambda_ces * g_total_sa[idx_obs]
    resid = (data['g_ces_sa'][idx_obs] - pred) / sigma_ces_sa
    ax.scatter(
        dates_arr[idx_obs], resid,
        s=8, c='darkorange', alpha=0.6,
    )
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.axhline(2, color='red', lw=0.5, ls=':', alpha=0.5)
    ax.axhline(-2, color='red', lw=0.5, ls=':', alpha=0.5)
    ax.set_ylabel('Std. residual')
    ax.set_title('CES SA')

    # --- CES NSA ---
    ax = axes[1]
    idx_obs = data['ces_nsa_obs']
    pred = alpha_ces + lambda_ces * g_total_nsa[idx_obs]
    resid = (
        (data['g_ces_nsa'][idx_obs] - pred) / sigma_ces_nsa
    )
    ax.scatter(
        dates_arr[idx_obs], resid,
        s=8, c='darkorange', alpha=0.6,
    )
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.axhline(2, color='red', lw=0.5, ls=':', alpha=0.5)
    ax.axhline(-2, color='red', lw=0.5, ls=':', alpha=0.5)
    ax.set_ylabel('Std. residual')
    ax.set_title('CES NSA')

    # --- PP1 (AR(1)-filtered residuals) ---
    ax = axes[2]
    idx_obs = data['pp_obs'][0]
    mu_base = alpha_pp[0] + lam_pp[0] * g_cont_nsa[idx_obs]
    u = data['g_pp'][0, idx_obs] - mu_base
    resid_pp1 = np.zeros_like(u)
    # First obs: standardize by marginal std
    resid_pp1[0] = (
        u[0] * np.sqrt(1 - rho_pp1 ** 2) / sigma_pp[0]
    )
    # Subsequent: conditional residual
    resid_pp1[1:] = (
        (u[1:] - rho_pp1 * u[:-1]) / sigma_pp[0]
    )
    ax.scatter(
        dates_arr[idx_obs], resid_pp1,
        s=8, c='green', alpha=0.6,
    )
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.axhline(2, color='red', lw=0.5, ls=':', alpha=0.5)
    ax.axhline(-2, color='red', lw=0.5, ls=':', alpha=0.5)
    ax.set_ylabel('Std. residual')
    ax.set_title('PP1 (AR(1)-filtered)')

    # --- PP2 ---
    ax = axes[3]
    idx_obs = data['pp_obs'][1]
    pred = alpha_pp[1] + lam_pp[1] * g_cont_nsa[idx_obs]
    resid = (data['g_pp'][1, idx_obs] - pred) / sigma_pp[1]
    ax.scatter(
        dates_arr[idx_obs], resid,
        s=8, c='sienna', alpha=0.6,
    )
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.axhline(2, color='red', lw=0.5, ls=':', alpha=0.5)
    ax.axhline(-2, color='red', lw=0.5, ls=':', alpha=0.5)
    ax.set_ylabel('Std. residual')
    ax.set_title('PP2')

    # --- QCEW ---
    ax = axes[4]
    idx_obs = data['qcew_obs']
    pred = g_total_nsa[idx_obs]
    qcew_sigma = np.where(
        data['qcew_is_m3'], 0.0005, 0.0015
    )
    resid = (data['g_qcew'][idx_obs] - pred) / qcew_sigma
    m3 = data['qcew_is_m3']
    ax.scatter(
        dates_arr[idx_obs][m3], resid[m3],
        s=12, c='darkred', alpha=0.7,
        label='M3 (quarter-end)',
    )
    ax.scatter(
        dates_arr[idx_obs][~m3], resid[~m3],
        s=12, c='salmon', alpha=0.7,
        label='M1-2 (retrospective UI)',
    )
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.axhline(2, color='red', lw=0.5, ls=':', alpha=0.5)
    ax.axhline(-2, color='red', lw=0.5, ls=':', alpha=0.5)
    ax.set_ylabel('Std. residual')
    ax.set_title('QCEW')
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.suptitle(
        'Standardized Residuals by Source '
        '(should be approx. N(0,1))',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    fig.savefig(
        OUTPUT_DIR / 'residuals.png',
        dpi=150, bbox_inches='tight',
    )
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / "residuals.png"}')


# =============================================================================
# Plotting: growth rates and seasonal
# =============================================================================

def plot_growth_and_seasonal(idata: az.InferenceData, data: dict):
    """Plot latent growth rates vs observed data and seasonal pattern."""

    dates = data['dates']
    g_ces_sa = data['g_ces_sa']
    g_ces_nsa = data['g_ces_nsa']
    g_pp = data['g_pp']

    pct = lambda x: x * 100  # noqa: E731

    g_cont_post = idata.posterior['g_cont'].values
    g_total_sa_post = idata.posterior['g_total_sa'].values
    g_total_nsa_post = idata.posterior['g_total_nsa'].values

    g_cont_mean = g_cont_post.mean(axis=(0, 1))
    g_cont_lo = np.percentile(g_cont_post, 10, axis=(0, 1))
    g_cont_hi = np.percentile(g_cont_post, 90, axis=(0, 1))

    g_sa_mean = g_total_sa_post.mean(axis=(0, 1))
    g_sa_lo = np.percentile(g_total_sa_post, 10, axis=(0, 1))
    g_sa_hi = np.percentile(g_total_sa_post, 90, axis=(0, 1))

    g_nsa_mean = g_total_nsa_post.mean(axis=(0, 1))
    g_nsa_lo = np.percentile(g_total_nsa_post, 10, axis=(0, 1))
    g_nsa_hi = np.percentile(g_total_nsa_post, 90, axis=(0, 1))

    # PP masks for plotting
    dates_arr = np.array(dates)
    pp1_mask = np.isfinite(g_pp[0])
    pp2_mask = np.isfinite(g_pp[1])

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)

    # --- Panel 1: Latent SA total growth vs CES SA ---
    ax = axes[0]
    ax.fill_between(
        dates, pct(g_sa_lo), pct(g_sa_hi),
        alpha=0.25, color='steelblue', label='80% CI',
    )
    ax.plot(dates, pct(g_sa_mean), 'steelblue', lw=1.5, label='Latent total (SA)')
    ax.scatter(dates, pct(g_ces_sa), s=10, c='darkorange', alpha=0.7,
               label='CES SA (observed)', zorder=5)
    ax.scatter(dates_arr[pp1_mask], pct(g_pp[0, pp1_mask]), s=6,
               c='green', alpha=0.4, label='PP1 (NSA)', zorder=3)
    ax.scatter(dates_arr[pp2_mask], pct(g_pp[1, pp2_mask]), s=6,
               c='sienna', alpha=0.4, label='PP2 (NSA)', zorder=3)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_ylabel('Monthly growth (%)')
    ax.set_title('SA Total Employment Growth: Latent State vs CES SA & PP')
    ax.legend(fontsize=8, loc='upper right')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # --- Panel 2: Latent NSA total growth vs CES NSA & PP ---
    ax = axes[1]
    ax.fill_between(
        dates, pct(g_nsa_lo), pct(g_nsa_hi),
        alpha=0.25, color='steelblue', label='80% CI',
    )
    ax.plot(dates, pct(g_nsa_mean), 'steelblue', lw=1.5, label='Latent total (NSA)')
    ax.scatter(dates, pct(g_ces_nsa), s=10, c='darkorange', alpha=0.7,
               label='CES NSA (observed)', zorder=5)
    ax.scatter(dates_arr[pp1_mask], pct(g_pp[0, pp1_mask]), s=8,
               c='green', alpha=0.5, label='PP1 (NSA)', zorder=4)
    ax.scatter(dates_arr[pp2_mask], pct(g_pp[1, pp2_mask]), s=8,
               c='sienna', alpha=0.5, label='PP2 (NSA)', zorder=4)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_ylabel('Monthly growth (%)')
    ax.set_title('NSA Total Employment Growth: Latent State vs CES NSA & PP')
    ax.legend(fontsize=8, loc='upper right')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # --- Panel 3: Continuing-units growth vs PP ---
    ax = axes[2]
    ax.fill_between(
        dates, pct(g_cont_lo), pct(g_cont_hi),
        alpha=0.25, color='steelblue', label='80% CI',
    )
    ax.plot(dates, pct(g_cont_mean), 'steelblue', lw=1.5,
            label='Latent cont. units (SA)')
    ax.scatter(dates_arr[pp1_mask], pct(g_pp[0, pp1_mask]), s=10,
               c='green', alpha=0.6, label='PP1 (NSA)', zorder=5)
    ax.scatter(dates_arr[pp2_mask], pct(g_pp[1, pp2_mask]), s=10,
               c='sienna', alpha=0.6, label='PP2 (NSA)', zorder=5)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_ylabel('Monthly growth (%)')
    ax.set_title('SA Continuing-Units Growth vs PP (NSA)')
    ax.legend(fontsize=8, loc='upper right')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # --- Panel 4: Estimated seasonal pattern ---
    s_post = idata.posterior['seasonal'].values
    s_mean = s_post.mean(axis=(0, 1))
    s_lo = np.percentile(s_post, 10, axis=(0, 1))
    s_hi = np.percentile(s_post, 90, axis=(0, 1))

    # Empirical seasonal from CES SA-vs-NSA difference for comparison
    emp_seasonal = np.zeros(12)
    emp_counts = np.zeros(12)
    diff = g_ces_nsa - g_ces_sa
    for t, d in enumerate(dates):
        if np.isfinite(diff[t]):
            emp_seasonal[d.month - 1] += diff[t]
            emp_counts[d.month - 1] += 1
    emp_seasonal = np.where(emp_counts > 0, emp_seasonal / emp_counts, 0.0)

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax = axes[3]
    x_pos = np.arange(12)
    width = 0.35
    ax.bar(x_pos - width / 2, pct(s_mean), width,
           color='steelblue', alpha=0.7, label='Model estimate')
    ax.bar(x_pos + width / 2, pct(emp_seasonal), width,
           color='darkorange', alpha=0.7, label='Empirical (CES NSA − SA)')
    ax.errorbar(
        x_pos - width / 2, pct(s_mean),
        yerr=[pct(s_mean - s_lo), pct(s_hi - s_mean)],
        fmt='none', c='k', capsize=3,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(month_labels)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_ylabel('Seasonal effect (%)')
    ax.set_title('Estimated vs Empirical Monthly Seasonal Pattern')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'growth_and_seasonal.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / "growth_and_seasonal.png"}')


# =============================================================================
# Plotting: reconstructed index levels
# =============================================================================

def plot_reconstructed_index(idata: az.InferenceData, data: dict):
    """Plot reconstructed SA/NSA index vs observed series."""

    dates = data['dates']
    levels = data['levels']

    g_total_mean = idata.posterior['g_total_sa'].values.mean(axis=(0, 1))
    g_nsa_mean = idata.posterior['g_total_nsa'].values.mean(axis=(0, 1))

    base_dates = levels['ref_date'].to_list()
    ces_sa_vals = levels['ces_sa_index'].to_numpy().astype(float)
    ces_nsa_vals = levels['ces_nsa_index'].to_numpy().astype(float)
    log_base_sa = np.log(ces_sa_vals[0])
    log_base_nsa = np.log(ces_nsa_vals[0])

    index_sa = np.exp(log_base_sa + np.cumsum(g_total_mean))
    index_nsa = np.exp(log_base_nsa + np.cumsum(g_nsa_mean))

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    ax.plot(dates, index_sa, 'steelblue', lw=2, label='Latent total (SA)')
    ax.plot(dates, index_nsa, 'steelblue', lw=1.5, ls='--',
            alpha=0.6, label='Latent total (NSA)')

    # CES SA observed (skip first date since growth starts at t=1)
    ax.plot(base_dates[1:], ces_sa_vals[1:], 'darkorange', lw=1.5, alpha=0.8,
            label='CES SA (observed)')
    # CES NSA observed
    ax.plot(base_dates[1:], ces_nsa_vals[1:], 'darkorange', lw=1, ls='--',
            alpha=0.5, label='CES NSA (observed)')

    # PP1 (provider 1)
    pp1_vals = levels['pp_index_1'].to_numpy().astype(float)
    pp1_mask = np.isfinite(pp1_vals)
    pp_dates_arr = np.array(base_dates)
    ax.plot(pp_dates_arr[pp1_mask][1:], pp1_vals[pp1_mask][1:],
            'green', lw=1, alpha=0.6, label='PP1 (NSA)')

    # PP2 (provider 2, basic continuing units)
    pp2_vals = levels['pp_index_2_0'].to_numpy().astype(float)
    pp2_mask = np.isfinite(pp2_vals)
    ax.plot(pp_dates_arr[pp2_mask][1:], pp2_vals[pp2_mask][1:],
            'sienna', lw=1, alpha=0.6, label='PP2 (NSA)')

    # QCEW
    qcew_rows = levels.filter(pl.col('qcew_nsa_index').is_not_null())
    ax.scatter(
        qcew_rows['ref_date'].to_list(),
        qcew_rows['qcew_nsa_index'].to_numpy(),
        s=40, c='red', marker='D', alpha=0.7, label='QCEW (NSA)', zorder=5,
    )

    ax.set_ylabel('Index (base ≈ 100)')
    ax.set_title('Reconstructed Latent Index vs Observed Series')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'reconstructed_index.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / "reconstructed_index.png"}')


# =============================================================================
# Forecast
# =============================================================================

def forecast_and_plot(idata: az.InferenceData, data: dict):
    """Forecast SA and NSA indices out to 2026-01-12 and plot."""

    dates = data['dates']
    levels = data['levels']

    # --- Posterior parameter samples ---
    mu_g_post = idata.posterior['mu_g'].values
    phi_post = idata.posterior['phi'].values
    sigma_g_post = idata.posterior['sigma_g'].values
    bd_post = idata.posterior['bd'].values       # (chains, draws) — gold standard: scalar
    seasonal_post = idata.posterior['seasonal'].values
    g_cont_post = idata.posterior['g_cont'].values
    g_sa_post = idata.posterior['g_total_sa'].values
    g_nsa_post = idata.posterior['g_total_nsa'].values

    n_chains, n_draws, T_hist = g_cont_post.shape

    # --- Build forecast date grid ---
    last_date = dates[-1]
    forecast_end = date(2026, 1, 12)

    forecast_dates = []
    d = last_date
    while True:
        yr, mo = d.year, d.month
        if mo == 12:
            d = date(yr + 1, 1, 12)
        else:
            d = date(yr, mo + 1, 12)
        forecast_dates.append(d)
        if d >= forecast_end:
            break

    n_fwd = len(forecast_dates)
    forecast_month_idx = [d.month - 1 for d in forecast_dates]

    # --- Simulate AR(1) + random walk b/d forward ---
    rng = np.random.default_rng(42)

    g_cont_fwd = np.zeros((n_chains, n_draws, n_fwd))
    g_sa_fwd = np.zeros((n_chains, n_draws, n_fwd))
    g_nsa_fwd = np.zeros((n_chains, n_draws, n_fwd))
    bd_fwd = np.zeros((n_chains, n_draws, n_fwd))

    for h in range(n_fwd):
        eps = rng.standard_normal((n_chains, n_draws))
        g_prev = g_cont_post[:, :, -1] if h == 0 else g_cont_fwd[:, :, h - 1]

        g_cont_fwd[:, :, h] = (
            mu_g_post
            + phi_post * (g_prev - mu_g_post)
            + sigma_g_post * eps
        )

        # Fixed b/d (gold standard): same scalar for all forecast months
        bd_fwd[:, :, h] = bd_post

        mi = forecast_month_idx[h]
        g_sa_fwd[:, :, h] = g_cont_fwd[:, :, h] + bd_fwd[:, :, h]
        g_nsa_fwd[:, :, h] = (
            g_cont_fwd[:, :, h]
            + seasonal_post[:, :, mi]
            + bd_fwd[:, :, h]
        )

    # --- Reconstruct full index paths (historical + forecast) ---
    ces_sa_vals = levels['ces_sa_index'].to_numpy().astype(float)
    ces_nsa_vals = levels['ces_nsa_index'].to_numpy().astype(float)
    log_base_sa = np.log(ces_sa_vals[0])
    log_base_nsa = np.log(ces_nsa_vals[0])

    # Base employment level at the index base date (2020-03-12 where index = 100)
    # Find the row where the index is closest to 100
    base_row_idx = np.argmin(np.abs(ces_sa_vals - 100.0))
    ces_sa_base_level = levels['ces_sa_level'].to_numpy().astype(float)[base_row_idx]
    ces_nsa_base_level = levels['ces_nsa_level'].to_numpy().astype(float)[base_row_idx]
    base_date = levels['ref_date'].to_list()[base_row_idx]

    cum_sa_hist = log_base_sa + np.cumsum(g_sa_post, axis=2)
    cum_nsa_hist = log_base_nsa + np.cumsum(g_nsa_post, axis=2)

    cum_sa_fwd = cum_sa_hist[:, :, -1:] + np.cumsum(g_sa_fwd, axis=2)
    cum_nsa_fwd = cum_nsa_hist[:, :, -1:] + np.cumsum(g_nsa_fwd, axis=2)

    idx_sa_hist = np.exp(cum_sa_hist)
    idx_nsa_hist = np.exp(cum_nsa_hist)
    idx_sa_fwd = np.exp(cum_sa_fwd)
    idx_nsa_fwd = np.exp(cum_nsa_fwd)

    # --- Print forecast table ---
    print('=' * 72)
    print('FORECAST: SA & NSA Index to 2026-01-12')
    print('=' * 72)
    print(f'{"Date":>12}  {"SA Mean":>9} {"SA 80% HDI":>18}  '
          f'{"NSA Mean":>9} {"NSA 80% HDI":>18}')
    print('-' * 72)

    sa_last = idx_sa_hist[:, :, -1].flatten()
    nsa_last = idx_nsa_hist[:, :, -1].flatten()
    print(f'{str(dates[-1]):>12}  {sa_last.mean():9.2f} '
          f'[{np.percentile(sa_last, 10):7.2f}, {np.percentile(sa_last, 90):7.2f}]  '
          f'{nsa_last.mean():9.2f} '
          f'[{np.percentile(nsa_last, 10):7.2f}, {np.percentile(nsa_last, 90):7.2f}]  '
          f'← last obs')

    for i, fd in enumerate(forecast_dates):
        sa_v = idx_sa_fwd[:, :, i].flatten()
        nsa_v = idx_nsa_fwd[:, :, i].flatten()
        print(f'{str(fd):>12}  {sa_v.mean():9.2f} '
              f'[{np.percentile(sa_v, 10):7.2f}, {np.percentile(sa_v, 90):7.2f}]  '
              f'{nsa_v.mean():9.2f} '
              f'[{np.percentile(nsa_v, 10):7.2f}, {np.percentile(nsa_v, 90):7.2f}]  '
              f'← forecast')

    # --- Convert index to levels (people) for change calculation ---
    idx_to_sa_level = ces_sa_base_level / 100.0
    idx_to_nsa_level = ces_nsa_base_level / 100.0
    lvl_sa_hist = idx_sa_hist * idx_to_sa_level
    lvl_nsa_hist = idx_nsa_hist * idx_to_nsa_level
    lvl_sa_fwd = idx_sa_fwd * idx_to_sa_level
    lvl_nsa_fwd = idx_nsa_fwd * idx_to_nsa_level

    # Month-over-month change (jobs added) in thousands
    chg_sa_hist = (lvl_sa_hist[:, :, 1:] - lvl_sa_hist[:, :, :-1]) / 1000.0   # (chains, draws, T_hist-1)
    chg_nsa_hist = (lvl_nsa_hist[:, :, 1:] - lvl_nsa_hist[:, :, :-1]) / 1000.0
    chg_sa_fwd = np.zeros((n_chains, n_draws, n_fwd))
    chg_nsa_fwd = np.zeros((n_chains, n_draws, n_fwd))
    chg_sa_fwd[:, :, 0] = (lvl_sa_fwd[:, :, 0] - lvl_sa_hist[:, :, -1]) / 1000.0
    chg_nsa_fwd[:, :, 0] = (lvl_nsa_fwd[:, :, 0] - lvl_nsa_hist[:, :, -1]) / 1000.0
    for i in range(1, n_fwd):
        chg_sa_fwd[:, :, i] = (lvl_sa_fwd[:, :, i] - lvl_sa_fwd[:, :, i - 1]) / 1000.0
        chg_nsa_fwd[:, :, i] = (lvl_nsa_fwd[:, :, i] - lvl_nsa_fwd[:, :, i - 1]) / 1000.0

    # --- Print forecast: Jobs added (month-over-month change, thousands) ---
    print('\n' + '=' * 84)
    print('FORECAST: Jobs added (month-over-month change, thousands)')
    print('  Positive = jobs added, negative = jobs lost.  Base ref: CES at index 100.')
    print('=' * 84)
    print(f'{"Date":>12}  {"SA Mean":>10} {"SA 80% HDI":>22}  '
          f'{"NSA Mean":>10} {"NSA 80% HDI":>22}')
    print('-' * 84)

    # Last historical month's change (into dates[-1])
    last_chg_sa = chg_sa_hist[:, :, -1].flatten()
    last_chg_nsa = chg_nsa_hist[:, :, -1].flatten()
    print(f'{str(dates[-1]):>12}  {last_chg_sa.mean():+10,.0f} '
          f'[{np.percentile(last_chg_sa, 10):+10,.0f}, {np.percentile(last_chg_sa, 90):+10,.0f}]  '
          f'{last_chg_nsa.mean():+10,.0f} '
          f'[{np.percentile(last_chg_nsa, 10):+10,.0f}, {np.percentile(last_chg_nsa, 90):+10,.0f}]  '
          f'← last obs')

    for i, fd in enumerate(forecast_dates):
        c_sa = chg_sa_fwd[:, :, i].flatten()
        c_nsa = chg_nsa_fwd[:, :, i].flatten()
        print(f'{str(fd):>12}  {c_sa.mean():+10,.0f} '
              f'[{np.percentile(c_sa, 10):+10,.0f}, {np.percentile(c_sa, 90):+10,.0f}]  '
              f'{c_nsa.mean():+10,.0f} '
              f'[{np.percentile(c_nsa, 10):+10,.0f}, {np.percentile(c_nsa, 90):+10,.0f}]  '
              f'← forecast')

    print('\n' + '=' * 72)
    print('FORECAST: Monthly Growth Rates')
    print('=' * 72)
    for i, fd in enumerate(forecast_dates):
        gsa = g_sa_fwd[:, :, i].flatten() * 100
        gnsa = g_nsa_fwd[:, :, i].flatten() * 100
        print(f'{str(fd):>12}  '
              f'SA:  {gsa.mean():+.3f}%  '
              f'[{np.percentile(gsa, 10):+.3f}%, {np.percentile(gsa, 90):+.3f}%]   '
              f'NSA: {gnsa.mean():+.3f}%  '
              f'[{np.percentile(gnsa, 10):+.3f}%, {np.percentile(gnsa, 90):+.3f}%]')

    # =====================================================================
    # Plot: Historical + Forecast
    # =====================================================================

    base_dates = levels['ref_date'].to_list()
    connecting_dates = [dates[-1]] + forecast_dates

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # --- SA panel ---
    ax = axes[0]

    sa_hist_mean = idx_sa_hist.mean(axis=(0, 1))
    sa_hist_lo = np.percentile(idx_sa_hist, 10, axis=(0, 1))
    sa_hist_hi = np.percentile(idx_sa_hist, 90, axis=(0, 1))

    ax.fill_between(dates, sa_hist_lo, sa_hist_hi, alpha=0.2, color='steelblue')
    ax.plot(dates, sa_hist_mean, 'steelblue', lw=2, label='Latent SA (estimated)')

    sa_fwd_mean = np.concatenate([[sa_hist_mean[-1]], idx_sa_fwd.mean(axis=(0, 1))])
    sa_fwd_lo = np.concatenate([[sa_hist_lo[-1]], np.percentile(idx_sa_fwd, 10, axis=(0, 1))])
    sa_fwd_hi = np.concatenate([[sa_hist_hi[-1]], np.percentile(idx_sa_fwd, 90, axis=(0, 1))])

    ax.fill_between(connecting_dates, sa_fwd_lo, sa_fwd_hi, alpha=0.3, color='coral')
    ax.plot(connecting_dates, sa_fwd_mean, 'coral', lw=2, ls='--', label='Forecast SA')

    ax.plot(base_dates[1:], ces_sa_vals[1:], 'darkorange', lw=1, alpha=0.7,
            label='CES SA (observed)')

    # PP on SA panel (for reference — PP is NSA but useful comparison)
    pp1_idx_vals = levels['pp_index_1'].to_numpy().astype(float)
    pp1_idx_mask = np.isfinite(pp1_idx_vals)
    bd_arr = np.array(base_dates)
    ax.plot(bd_arr[pp1_idx_mask][1:], pp1_idx_vals[pp1_idx_mask][1:],
            'green', lw=1, alpha=0.5, label='PP1 (NSA)')
    pp2_idx_vals = levels['pp_index_2_0'].to_numpy().astype(float)
    pp2_idx_mask = np.isfinite(pp2_idx_vals)
    ax.plot(bd_arr[pp2_idx_mask][1:], pp2_idx_vals[pp2_idx_mask][1:],
            'sienna', lw=1, alpha=0.5, label='PP2 (NSA)')

    ax.axvline(dates[-1], color='gray', ls=':', lw=1, alpha=0.7, label='Forecast start')
    ax.set_ylabel('Index (base ≈ 100)')
    ax.set_title('SA Total Employment Index: Estimate + Forecast to 2026-01-12')
    ax.legend(fontsize=8, loc='upper left')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # --- NSA panel ---
    ax = axes[1]

    nsa_hist_mean = idx_nsa_hist.mean(axis=(0, 1))
    nsa_hist_lo = np.percentile(idx_nsa_hist, 10, axis=(0, 1))
    nsa_hist_hi = np.percentile(idx_nsa_hist, 90, axis=(0, 1))

    ax.fill_between(dates, nsa_hist_lo, nsa_hist_hi, alpha=0.2, color='steelblue')
    ax.plot(dates, nsa_hist_mean, 'steelblue', lw=2, label='Latent NSA (estimated)')

    nsa_fwd_mean = np.concatenate([[nsa_hist_mean[-1]], idx_nsa_fwd.mean(axis=(0, 1))])
    nsa_fwd_lo = np.concatenate([[nsa_hist_lo[-1]], np.percentile(idx_nsa_fwd, 10, axis=(0, 1))])
    nsa_fwd_hi = np.concatenate([[nsa_hist_hi[-1]], np.percentile(idx_nsa_fwd, 90, axis=(0, 1))])

    ax.fill_between(connecting_dates, nsa_fwd_lo, nsa_fwd_hi, alpha=0.3, color='coral')
    ax.plot(connecting_dates, nsa_fwd_mean, 'coral', lw=2, ls='--', label='Forecast NSA')

    # CES NSA observed
    ax.plot(base_dates[1:], ces_nsa_vals[1:], 'darkorange', lw=1, alpha=0.7,
            label='CES NSA (observed)')

    ax.plot(bd_arr[pp1_idx_mask][1:], pp1_idx_vals[pp1_idx_mask][1:],
            'green', lw=1, alpha=0.6, label='PP1 (NSA)')
    ax.plot(bd_arr[pp2_idx_mask][1:], pp2_idx_vals[pp2_idx_mask][1:],
            'sienna', lw=1, alpha=0.6, label='PP2 (NSA)')

    qcew_rows = levels.filter(pl.col('qcew_nsa_index').is_not_null())
    ax.scatter(
        qcew_rows['ref_date'].to_list(),
        qcew_rows['qcew_nsa_index'].to_numpy(),
        s=40, c='red', marker='D', alpha=0.7, label='QCEW (NSA)', zorder=5,
    )

    ax.axvline(dates[-1], color='gray', ls=':', lw=1, alpha=0.7, label='Forecast start')
    ax.set_ylabel('Index (base ≈ 100)')
    ax.set_title('NSA Total Employment Index: Estimate + Forecast to 2026-01-12')
    ax.legend(fontsize=8, loc='upper left')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'forecast_sa_nsa.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / "forecast_sa_nsa.png"}')

    # =====================================================================
    # Plot: Jobs added (month-over-month change, thousands) — recent + forecast
    # =====================================================================

    obs_sa_levels = levels['ces_sa_level'].to_numpy().astype(float)
    obs_nsa_levels = levels['ces_nsa_level'].to_numpy().astype(float)
    obs_chg_sa = (obs_sa_levels[1:] - obs_sa_levels[:-1]) / 1000.0
    obs_chg_nsa = (obs_nsa_levels[1:] - obs_nsa_levels[:-1]) / 1000.0
    obs_chg_dates = base_dates[1:]  # change into month i is at base_dates[i]

    n_recent = 24
    recent_start = max(0, T_hist - n_recent)
    # Change into month i is chg_*[:,:,i-1]; date for that is dates[i]
    recent_chg_start = max(0, recent_start - 1)  # so we have ~n_recent change points
    recent_dates_chg = dates[recent_chg_start + 1 : T_hist]  # dates for chg_hist[:,:,recent_chg_start:]
    connecting = [dates[-1]] + forecast_dates

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for panel_idx, (label, chg_hist, chg_fwd, obs_chg, color_hist, color_obs) in enumerate([
        ('SA', chg_sa_hist, chg_sa_fwd, obs_chg_sa, '#2563eb', '#ea580c'),
        ('NSA', chg_nsa_hist, chg_nsa_fwd, obs_chg_nsa, '#2563eb', '#ea580c'),
    ]):
        ax = axes[panel_idx]

        # Recent historical changes
        hist_mean = chg_hist[:, :, recent_chg_start:].mean(axis=(0, 1))
        hist_lo = np.percentile(chg_hist[:, :, recent_chg_start:], 10, axis=(0, 1))
        hist_hi = np.percentile(chg_hist[:, :, recent_chg_start:], 90, axis=(0, 1))

        ax.fill_between(recent_dates_chg, hist_lo, hist_hi,
                        alpha=0.2, color=color_hist, label='80% HDI')
        ax.plot(recent_dates_chg, hist_mean, color=color_hist, lw=2,
                label=f'Model {label} (jobs added)')

        # Forecast changes
        fwd_mean = chg_fwd.mean(axis=(0, 1))
        fwd_lo = np.percentile(chg_fwd, 10, axis=(0, 1))
        fwd_hi = np.percentile(chg_fwd, 90, axis=(0, 1))

        ax.fill_between(forecast_dates, fwd_lo, fwd_hi,
                        alpha=0.25, color='#ef4444', label='Forecast 80% HDI')
        ax.plot(forecast_dates, fwd_mean, color='#ef4444', lw=2.5, ls='--',
                label='Forecast')

        for i, fd in enumerate(forecast_dates):
            pt_mean = chg_fwd[:, :, i].flatten().mean()
            pt_lo = np.percentile(chg_fwd[:, :, i].flatten(), 10)
            pt_hi = np.percentile(chg_fwd[:, :, i].flatten(), 90)
            ax.errorbar(fd, pt_mean, yerr=[[pt_mean - pt_lo], [pt_hi - pt_mean]],
                        fmt='o', color='#ef4444', markersize=8, capsize=6,
                        capthick=2, elinewidth=2, zorder=10)
            ax.annotate(f'{pt_mean:+,.0f}k',
                        xy=(fd, pt_hi), xytext=(8, 8),
                        textcoords='offset points', fontsize=9, fontweight='bold',
                        color='#ef4444',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                  ec='#ef4444', alpha=0.9))

        # Observed CES jobs added (filter to recent window)
        obs_idx = [i for i, d in enumerate(obs_chg_dates) if recent_dates_chg[0] <= d <= dates[-1]]
        if obs_idx:
            ax.scatter(
                [obs_chg_dates[i] for i in obs_idx],
                [obs_chg[i] for i in obs_idx],
                s=25, c=color_obs, alpha=0.8, zorder=5, label=f'CES {label} (observed)',
            )

        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.axvline(dates[-1], color='#6b7280', ls=':', lw=1.2, alpha=0.8)
        ax.set_ylabel('Jobs added (thousands)', fontsize=11)
        ax.set_title(f'{label} Total Nonfarm Employment: Jobs Added (Month-over-Month) to 2026-01-12',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.tick_params(axis='x', labelsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:+,.0f}'))
        ax.grid(axis='y', alpha=0.3, ls='--')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'forecast_levels.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / "forecast_levels.png"}')


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Data loading ---
    print('Loading data...\n')
    data = load_data()

    # --- Model building ---
    print('\nBuilding model...\n')
    model = build_model(data)

    # --- Prior predictive check (before fitting) ---
    print('\nPrior predictive check...\n')
    run_prior_predictive_checks(model, data)

    # --- Posterior sampling ---
    print('\nSampling...\n')
    idata = sample_model(model)

    # --- Computational diagnostics ---
    print_diagnostics(idata, data)
    plot_divergences(idata)

    # --- Posterior predictive checks ---
    print('\nPosterior predictive checks...\n')
    run_posterior_predictive_checks(model, idata, data)

    # --- LOO cross-validation ---
    print('\nLOO cross-validation...\n')
    run_loo_cv(model, idata, data)

    # --- Residual analysis ---
    print('\nResidual analysis...\n')
    plot_residuals(idata, data)

    # --- Source contributions & existing plots ---
    print_source_contributions(idata, data)
    plot_growth_and_seasonal(idata, data)
    plot_reconstructed_index(idata, data)
    forecast_and_plot(idata, data)

    # Save inference data for later analysis
    idata.to_netcdf(OUTPUT_DIR / 'idata.nc')
    print(f'Saved: {OUTPUT_DIR / "idata.nc"}')

    print('\nDone.')


if __name__ == "__main__":
    main()
