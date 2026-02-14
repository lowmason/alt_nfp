# -------------------------------------------------------------------------------------------------
# QCEW Sigma Sensitivity (v3)
#
# Run the v2 model with three QCEW noise levels: 0.5x (tight), 1x (baseline), 2x (loose).
# Compare key parameters and precision budget to check whether calibrated sigmas drive conclusions.
# -------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pymc as pm
import pytensor
import pytensor.tensor as pt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'output'

N_PP = 2
PP_NAMES = ['PP1', 'PP2']

# Configs: (label, SIGMA_QCEW_M3, SIGMA_QCEW_M12)
QCEW_SIGMA_CONFIGS = [
    ('0.5x (tight)', 0.00025, 0.00075),
    ('1x (baseline)', 0.0005, 0.0015),
    ('2x (loose)', 0.001, 0.003),
]

SAMPLER_KWARGS = dict(
    draws=4000,
    tune=3000,
    chains=4,
    target_accept=0.95,
    return_inferencedata=True,
)


# =============================================================================
# Data loading (same as v2, no censoring)
# =============================================================================

def load_data():
    """Load and align CES, QCEW, and PP data; compute growth rates."""
    ces = (
        pl.read_csv(BASE_DIR / 'data/ces_index.csv', try_parse_dates=True)
        .sort('ref_date')
    )
    qcew = (
        pl.read_csv(BASE_DIR / 'data/qcew_index.csv', try_parse_dates=True)
        .sort('ref_date')
    )
    pp1 = (
        pl.read_csv(BASE_DIR / 'data/pp_index_1.csv', try_parse_dates=True)
        .sort('ref_date')
    )
    pp2 = (
        pl.read_csv(BASE_DIR / 'data/pp_index_2.csv', try_parse_dates=True)
        .sort('ref_date')
    )

    cal = pl.DataFrame({
        'ref_date': pl.date_range(
            ces['ref_date'].min(),
            ces['ref_date'].max(),
            interval='1mo',
            eager=True,
        )
    })

    levels = (
        cal
        .join(ces, on='ref_date', how='left')
        .join(qcew, on='ref_date', how='left')
        .join(pp1, on='ref_date', how='left')
        .join(pp2, on='ref_date', how='left')
        .sort('ref_date')
    )

    growth = (
        levels
        .with_columns([
            pl.col('ces_sa_index').log().diff().alias('g_ces_sa'),
            pl.col('ces_nsa_index').log().diff().alias('g_ces_nsa'),
            pl.col('qcew_nsa_index').log().diff().alias('g_qcew'),
            pl.col('pp_index_1').log().diff().alias('g_pp_1'),
            pl.col('pp_index_2_0').log().diff().alias('g_pp_2'),
        ])
        .slice(1)
    )

    dates = growth['ref_date'].to_list()
    T = len(dates)
    month_of_year = np.array([d.month - 1 for d in dates], dtype=int)

    g_ces_sa = growth['g_ces_sa'].to_numpy().astype(float)
    g_ces_nsa = growth['g_ces_nsa'].to_numpy().astype(float)
    ces_sa_obs = np.where(np.isfinite(g_ces_sa))[0]
    ces_nsa_obs = np.where(np.isfinite(g_ces_nsa))[0]

    pp_cols = ['g_pp_1', 'g_pp_2']
    g_pp = np.vstack([growth[c].to_numpy().astype(float) for c in pp_cols])
    pp_obs = [np.where(np.isfinite(g_pp[p]))[0] for p in range(N_PP)]

    g_qcew = growth['g_qcew'].to_numpy().astype(float)
    qcew_obs = np.where(np.isfinite(g_qcew))[0]
    qcew_is_m3 = np.array([dates[i].month in (3, 6, 9, 12) for i in qcew_obs])

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
# Model with parameterized QCEW sigmas
# =============================================================================

def build_and_sample(
    data: dict,
    sigma_qcew_m3: float,
    sigma_qcew_m12: float,
    sampler_kwargs: dict | None = None,
) -> az.InferenceData:
    if sampler_kwargs is None:
        sampler_kwargs = SAMPLER_KWARGS

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

    qcew_sigma_fixed = np.where(qcew_is_m3, sigma_qcew_m3, sigma_qcew_m12)

    with pm.Model() as model:
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

        s_raw = pm.Normal('s_raw', 0, 0.015, shape=11)
        s_all = pt.concatenate([s_raw, (-pt.sum(s_raw)).reshape((1,))])
        s_t = s_all[month_of_year]
        pm.Deterministic('seasonal', s_all)

        bd = pm.Normal('bd', mu=0.001, sigma=0.002)

        g_cont_nsa = g_cont + s_t
        g_total_sa = g_cont + bd
        g_total_nsa = g_cont + s_t + bd
        pm.Deterministic('g_total_sa', g_total_sa)
        pm.Deterministic('g_total_nsa', g_total_nsa)

        pm.Normal(
            'obs_qcew',
            mu=g_total_nsa[qcew_obs],
            sigma=qcew_sigma_fixed,
            observed=g_qcew[qcew_obs],
        )

        alpha_ces = pm.Normal('alpha_ces', 0, 0.005)
        lambda_ces = pm.Normal('lambda_ces', 1.0, 0.25)
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

        alpha_pp = pm.Normal('alpha_pp', 0, 0.005, shape=N_PP)
        lam_pp = pm.Normal('lam_pp', 1.0, 0.25, shape=N_PP)
        sigma_pp = pm.InverseGamma('sigma_pp', alpha=3.0, beta=0.004, shape=N_PP)
        rho_pp1 = pm.Uniform('rho_pp1', lower=0.0, upper=0.99)

        m_pp1 = pp_obs[0]
        mu_base_pp1 = alpha_pp[0] + lam_pp[0] * g_cont_nsa[m_pp1]
        y_pp1 = g_pp[0, m_pp1]
        mu_cond_pp1 = pt.concatenate([
            mu_base_pp1[:1],
            mu_base_pp1[1:] + rho_pp1 * (
                pt.as_tensor_variable(y_pp1[:-1]) - mu_base_pp1[:-1]
            ),
        ])
        sigma_cond_pp1 = pt.concatenate([
            (sigma_pp[0] / pt.sqrt(1.0 - rho_pp1 ** 2)).reshape((1,)),
            pt.ones(len(m_pp1) - 1) * sigma_pp[0],
        ])
        pm.Normal(
            'obs_pp1',
            mu=mu_cond_pp1,
            sigma=sigma_cond_pp1,
            observed=y_pp1,
        )

        m_pp2 = pp_obs[1]
        mu_pp2 = alpha_pp[1] + lam_pp[1] * g_cont_nsa[m_pp2]
        pm.Normal(
            'obs_pp2',
            mu=mu_pp2,
            sigma=sigma_pp[1],
            observed=g_pp[1, m_pp2],
        )

        try:
            idata = pm.sample(nuts_sampler='nutpie', **sampler_kwargs)
        except Exception:
            idata = pm.sample(**sampler_kwargs)

    return idata


# =============================================================================
# Precision budget (same logic as v2)
# =============================================================================

def precision_budget_shares(idata: az.InferenceData, data: dict, sigma_m3: float, sigma_m12: float):
    """Return dict of source -> share (fraction of total precision)."""
    sigma_ces_sa = idata.posterior['sigma_ces_sa'].values.flatten().mean()
    sigma_ces_nsa = idata.posterior['sigma_ces_nsa'].values.flatten().mean()
    lambda_ces = idata.posterior['lambda_ces'].values.flatten().mean()
    sigma_pp = idata.posterior['sigma_pp'].values.mean(axis=(0, 1))
    lam_pp = idata.posterior['lam_pp'].values.mean(axis=(0, 1))
    rho_pp1 = idata.posterior['rho_pp1'].values.flatten().mean()

    pp_obs = data['pp_obs']
    qcew_obs = data['qcew_obs']
    qcew_is_m3 = data['qcew_is_m3']
    n_qcew_m3 = int(qcew_is_m3.sum())
    n_qcew_m12 = len(qcew_obs) - n_qcew_m3

    prec_ces_sa = lambda_ces ** 2 / sigma_ces_sa ** 2
    prec_ces_nsa = lambda_ces ** 2 / sigma_ces_nsa ** 2
    prec_qcew_m3 = 1.0 / sigma_m3 ** 2
    prec_qcew_m12 = 1.0 / sigma_m12 ** 2
    prec_pp = lam_pp ** 2 / sigma_pp ** 2
    prec_pp[0] = lam_pp[0] ** 2 * (1 - rho_pp1 ** 2) / sigma_pp[0] ** 2

    n_ces_sa = len(data['ces_sa_obs'])
    n_ces_nsa = len(data['ces_nsa_obs'])
    n_pp_obs = [len(pp_obs[p]) for p in range(N_PP)]

    total_ces_sa = prec_ces_sa * n_ces_sa
    total_ces_nsa = prec_ces_nsa * n_ces_nsa
    total_qcew = prec_qcew_m3 * n_qcew_m3 + prec_qcew_m12 * n_qcew_m12
    total_pp = np.array([prec_pp[p] * n_pp_obs[p] for p in range(N_PP)])
    total_all = total_ces_sa + total_ces_nsa + total_qcew + total_pp.sum()

    return {
        'CES SA': total_ces_sa / total_all,
        'CES NSA': total_ces_nsa / total_all,
        'QCEW': total_qcew / total_all,
        'PP1': total_pp[0] / total_all,
        'PP2': total_pp[1] / total_all,
    }


# =============================================================================
# Run sensitivity and report
# =============================================================================

def run_sensitivity():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading data...')
    data = load_data()
    print(f'T = {data["T"]} months\n')

    # Parameters to compare (name, key, scale for display, format)
    PARAMS = [
        ('α_ces (%/mo)', 'alpha_ces', 100, '.4f'),
        ('λ_ces', 'lambda_ces', 1, '.4f'),
        ('σ_ces_sa (%)', 'sigma_ces_sa', 100, '.3f'),
        ('σ_ces_nsa (%)', 'sigma_ces_nsa', 100, '.3f'),
        ('bd (%/mo)', 'bd', 100, '.4f'),
        ('ρ_pp1', 'rho_pp1', 1, '.3f'),
        ('α_pp1 (%/mo)', 'alpha_pp', 100, '.4f'),  # index 0
        ('α_pp2 (%/mo)', 'alpha_pp', 100, '.4f'),  # index 1
        ('λ_pp1', 'lam_pp', 1, '.3f'),
        ('λ_pp2', 'lam_pp', 1, '.3f'),
        ('σ_pp1 (%)', 'sigma_pp', 100, '.3f'),
        ('σ_pp2 (%)', 'sigma_pp', 100, '.3f'),
    ]

    results = []
    idatas = []

    for label, sigma_m3, sigma_m12 in QCEW_SIGMA_CONFIGS:
        print(f'Running config: {label} (σ_M3={sigma_m3}, σ_M12={sigma_m12})...')
        idata = build_and_sample(data, sigma_m3, sigma_m12)
        idatas.append((label, idata))

        row = {'config': label}
        post = idata.posterior

        for pname, key, scale, _ in PARAMS:
            if key in ('alpha_pp', 'lam_pp', 'sigma_pp'):
                # shape (chain, draw, 2)
                vals = post[key].values
                if 'α_pp1' in pname or 'λ_pp1' in pname or 'σ_pp1' in pname:
                    v = vals[:, :, 0].flatten()
                else:
                    v = vals[:, :, 1].flatten()
            else:
                v = post[key].values.flatten()
            row[pname] = (v.mean() * scale, np.percentile(v, 10) * scale, np.percentile(v, 90) * scale)

        shares = precision_budget_shares(idata, data, sigma_m3, sigma_m12)
        row['precision_shares'] = shares
        results.append(row)
        print(f'  Done.\n')

    # Console table: parameter by config
    print('=' * 100)
    print('QCEW SIGMA SENSITIVITY: Parameter comparison (mean, 80% HDI)')
    print('=' * 100)
    print(f'{"Parameter":<18}', end='')
    for label, _, _ in QCEW_SIGMA_CONFIGS:
        print(f'  {label:>20}', end='')
    print()
    print('-' * 100)

    for pname, key, scale, fmt in PARAMS:
        line = f'{pname:<18}'
        for r in results:
            mean, lo, hi = r[pname]
            line += f'  {mean:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]'.rjust(24)
        print(line)

    print('\n' + '=' * 100)
    print('Precision budget shares (%)')
    print('=' * 100)
    print(f'{"Source":<12}', end='')
    for r in results:
        print(f'  {r["config"]:>18}', end='')
    print()
    print('-' * 100)
    for src in ['CES SA', 'CES NSA', 'QCEW', 'PP1', 'PP2']:
        line = f'{src:<12}'
        for r in results:
            pct = 100 * r['precision_shares'][src]
            line += f'  {pct:>17.1f}%'
        print(line)

    # Verdict
    print('\n' + '=' * 100)
    print('VERDICT')
    print('=' * 100)
    bd_means = [r['bd (%/mo)'][0] for r in results]
    alpha_ces_means = [r['α_ces (%/mo)'][0] for r in results]
    if max(bd_means) - min(bd_means) < 0.002 and max(alpha_ces_means) - min(alpha_ces_means) < 0.001:
        print('Key parameters (bd, α_ces) are stable across 0.5x–2x QCEW sigma. Calibration is not driving conclusions.')
    else:
        print('Key parameters shift meaningfully with QCEW sigma. Consider reporting sensitivity in the paper.')

    # Figure: grouped bar chart for a subset of parameters
    param_keys = ['α_ces (%/mo)', 'λ_ces', 'bd (%/mo)', 'ρ_pp1', 'λ_pp1', 'λ_pp2']
    n_params = len(param_keys)
    configs = [r['config'] for r in results]
    x = np.arange(n_params)
    width = 0.25

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i, cfg in enumerate(configs):
        means = [results[i][p][0] for p in param_keys]
        los = [results[i][p][1] for p in param_keys]
        his = [results[i][p][2] for p in param_keys]
        err_lo = [m - lo for m, lo in zip(means, los)]
        err_hi = [hi - m for m, hi in zip(means, his)]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width, label=cfg, yerr=[err_lo, err_hi], capsize=2)
        if i == 1:
            for b in bars:
                b.set_edgecolor('black')
                b.set_linewidth(1.2)

    ax.set_ylabel('Posterior mean (80% HDI)')
    ax.set_xticks(x)
    ax.set_xticklabels(param_keys, rotation=15, ha='right')
    ax.legend()
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_title('QCEW Sigma Sensitivity: Key Parameters by Configuration')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'sensitivity_qcew_sigma.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {OUTPUT_DIR / "sensitivity_qcew_sigma.png"}')

    return results


if __name__ == '__main__':
    run_sensitivity()
