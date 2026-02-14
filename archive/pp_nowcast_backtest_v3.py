# -------------------------------------------------------------------------------------------------
# Nowcast Backtest (v3)
#
# For each of the last 24 months: censor CES from that month onward, run the v2 model
# with lighter sampling, and compare the latent-state nowcast to the actual CES release.
# Quantifies PP's operational forecasting value.
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

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'output'

N_PP = 2
PP_NAMES = ['PP1', 'PP2']

# Lighter sampling for 24-run loop (~30 sec/run)
SAMPLER_KWARGS = dict(
    draws=2000,
    tune=2000,
    chains=2,
    target_accept=0.95,
    return_inferencedata=True,
)


# =============================================================================
# Data loading (with optional CES censoring)
# =============================================================================

def load_data(censor_ces_from: date | None = None):
    """Load and align CES (SA+NSA), QCEW, and PP data; compute growth rates.
    If censor_ces_from is set, CES SA/NSA are treated as missing from that month onward.
    """
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

    # Apply censoring: from censor_ces_from onward, treat CES as missing
    if censor_ces_from is not None:
        censor_idx = None
        for i, d in enumerate(dates):
            if d >= censor_ces_from:
                censor_idx = i
                break
        if censor_idx is not None:
            g_ces_sa = g_ces_sa.copy()
            g_ces_nsa = g_ces_nsa.copy()
            g_ces_sa[censor_idx:] = np.nan
            g_ces_nsa[censor_idx:] = np.nan

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
# Model and sampling (v2 logic with configurable sampler)
# =============================================================================

def build_and_sample(data: dict, sampler_kwargs: dict | None = None) -> az.InferenceData:
    """Build the QCEW-anchored model and sample."""
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

    SIGMA_QCEW_M3 = 0.001
    SIGMA_QCEW_M12 = 0.003
    qcew_sigma_fixed = np.where(qcew_is_m3, SIGMA_QCEW_M3, SIGMA_QCEW_M12)

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

        if len(ces_sa_obs) > 0:
            pm.Normal(
                'obs_ces_sa',
                mu=alpha_ces + lambda_ces * g_total_sa[ces_sa_obs],
                sigma=sigma_ces_sa,
                observed=g_ces_sa[ces_sa_obs],
            )
        if len(ces_nsa_obs) > 0:
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
# Backtest loop and reporting
# =============================================================================

def run_backtest():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Full data for actuals (no censoring)
    data_full = load_data(censor_ces_from=None)
    dates = data_full['dates']
    T = len(dates)
    g_ces_sa_actual = data_full['g_ces_sa']
    levels = data_full['levels']
    ces_sa_index = levels['ces_sa_index'].to_numpy().astype(float)
    ces_sa_level = levels['ces_sa_level'].to_numpy().astype(float)
    base_row_idx = np.argmin(np.abs(ces_sa_index - 100.0))
    ces_sa_base_level = float(ces_sa_level[base_row_idx])
    base_index = float(ces_sa_index[0])

    # Last 24 months
    n_backtest = 24
    if T < n_backtest:
        raise ValueError(f'Need at least {n_backtest} months, got T={T}')
    target_indices = list(range(T - n_backtest, T))
    target_dates = [dates[i] for i in target_indices]

    results = []
    for run, (t_idx, target_date) in enumerate(zip(target_indices, target_dates)):
        print(f'\n--- Nowcast backtest {run + 1}/{n_backtest}: {target_date} ---')
        data = load_data(censor_ces_from=target_date)
        idata = build_and_sample(data)

        g_sa_post = idata.posterior['g_total_sa'].values
        g_sa_mean = g_sa_post.mean(axis=(0, 1))
        cum_sa = np.cumsum(g_sa_mean)
        nowcast_growth = g_sa_mean[t_idx]
        nowcast_index = base_index * np.exp(cum_sa[t_idx])
        nowcast_level = nowcast_index * (ces_sa_base_level / 100.0)

        actual_growth = g_ces_sa_actual[t_idx]
        actual_index = ces_sa_index[t_idx + 1]
        actual_level = ces_sa_level[t_idx + 1]

        # Month-over-month change (jobs added) in thousands
        actual_change_k = (actual_level - ces_sa_level[t_idx]) / 1000.0
        if t_idx > 0:
            prev_level = base_index * np.exp(cum_sa[t_idx - 1]) * (ces_sa_base_level / 100.0)
        else:
            prev_level = ces_sa_level[0]
        nowcast_change_k = (nowcast_level - prev_level) / 1000.0
        error_change_k = actual_change_k - nowcast_change_k  # positive = we under-nowcast change

        err_growth_pp = (nowcast_growth - actual_growth) * 100  # positive = nowcast high
        err_level_k = (actual_level - nowcast_level) / 1000.0   # thousands; positive = under-nowcast

        # Data sources that have an observation at the target month
        has_qcew = t_idx in data['qcew_obs']
        has_pp1 = t_idx in data['pp_obs'][0]
        has_pp2 = t_idx in data['pp_obs'][1]
        sources = []
        if has_qcew:
            sources.append('QCEW')
        if has_pp1:
            sources.append('PP1')
        if has_pp2:
            sources.append('PP2')
        sources_str = '+'.join(sources) if sources else 'none'

        results.append({
            'date': target_date,
            'actual_growth_pct': actual_growth * 100,
            'nowcast_growth_pct': nowcast_growth * 100,
            'error_growth_pp': err_growth_pp,
            'actual_change_k': actual_change_k,
            'nowcast_change_k': nowcast_change_k,
            'error_change_k': error_change_k,
            'actual_level_k': actual_level / 1000.0,
            'nowcast_level_k': nowcast_level / 1000.0,
            'error_level_k': err_level_k,
            'sources': sources_str,
        })
        print(f'  Jobs added (SA): actual {actual_change_k:+,.0f}k  nowcast {nowcast_change_k:+,.0f}k  '
              f'error {error_change_k:+,.0f}k  [{sources_str}]')

    # Console table (lead with jobs added)
    print('\n' + '=' * 95)
    print(f'NOWCAST BACKTEST: Last {n_backtest} months (jobs added = month-over-month change, SA)')
    print('=' * 95)
    print(f'{"Date":>12}  {"Actual Δ(k)":>11} {"Nowcast Δ(k)":>11} {"Error Δ(k)":>10}  '
          f'{"Actual %":>8} {"Nowcast %":>8} {"Error (pp)":>9}  {"Sources":>12}')
    print('-' * 95)
    for r in results:
        print(f'{str(r["date"]):>12}  {r["actual_change_k"]:>+10,.0f}  {r["nowcast_change_k"]:>+10,.0f}  '
              f'{r["error_change_k"]:>+9,.0f}  {r["actual_growth_pct"]:>+7.3f}  {r["nowcast_growth_pct"]:>+7.3f}  '
              f'{r["error_growth_pp"]:>+8.3f}  {r["sources"]:>12}')

    # Split by data availability: data-rich (QCEW or PP1 present) vs PP2-only
    data_rich = [r for r in results if 'QCEW' in r['sources'] or 'PP1' in r['sources']]
    pp2_only = [r for r in results if r['sources'] == 'PP2']
    no_data = [r for r in results if r['sources'] == 'none']

    def mae_rmse(lst, key_err):
        if not lst:
            return np.nan, np.nan
        errs = [x[key_err] for x in lst]
        return np.mean(np.abs(errs)), np.sqrt(np.mean(np.array(errs) ** 2))

    mae_gr, rmse_gr = mae_rmse(results, 'error_growth_pp')
    mae_chg, rmse_chg = mae_rmse(results, 'error_change_k')
    mae_lv, rmse_lv = mae_rmse(results, 'error_level_k')
    print('-' * 95)
    print(f'Overall (n={len(results)}):  MAE jobs added = {mae_chg:,.0f} k   RMSE = {rmse_chg:,.0f} k  '
          f'|  MAE growth = {mae_gr:.3f} pp')
    if data_rich:
        mae_dr_chg, _ = mae_rmse(data_rich, 'error_change_k')
        mae_dr_gr, _ = mae_rmse(data_rich, 'error_growth_pp')
        dr_start = min(r['date'] for r in data_rich)
        dr_end = max(r['date'] for r in data_rich)
        print(f'Data-rich ({dr_start}–{dr_end}, n={len(data_rich)}):  '
              f'MAE jobs added = {mae_dr_chg:,.0f} k   MAE growth = {mae_dr_gr:.3f} pp')
    if pp2_only:
        mae_po_chg, _ = mae_rmse(pp2_only, 'error_change_k')
        mae_po_gr, _ = mae_rmse(pp2_only, 'error_growth_pp')
        po_start = min(r['date'] for r in pp2_only)
        po_end = max(r['date'] for r in pp2_only)
        print(f'PP2-only ({po_start}–{po_end}, n={len(pp2_only)}):  '
              f'MAE jobs added = {mae_po_chg:,.0f} k   MAE growth = {mae_po_gr:.3f} pp')
    if no_data:
        mae_nd_chg, _ = mae_rmse(no_data, 'error_change_k')
        mae_nd_gr, _ = mae_rmse(no_data, 'error_growth_pp')
        nd_start = min(r['date'] for r in no_data)
        nd_end = max(r['date'] for r in no_data)
        print(f'No alt data ({nd_start}–{nd_end}, n={len(no_data)}):  '
              f'MAE jobs added = {mae_nd_chg:,.0f} k   MAE growth = {mae_nd_gr:.3f} pp')

    # Figure: two panels
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    x_dates = [r['date'] for r in results]
    actual_gr = [r['actual_growth_pct'] for r in results]
    nowcast_gr = [r['nowcast_growth_pct'] for r in results]
    err_gr = [r['error_growth_pp'] for r in results]

    ax = axes[0]
    ax.plot(x_dates, actual_gr, 'o-', color='darkorange', label='Actual CES SA growth', ms=8)
    ax.plot(x_dates, nowcast_gr, 's--', color='steelblue', label='Nowcast (model)', ms=6)
    for i, (a, n) in enumerate(zip(actual_gr, nowcast_gr)):
        ax.vlines(x_dates[i], a, n, color='gray', alpha=0.6, lw=1)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_ylabel('Monthly growth (%)')
    ax.set_title('SA Total Employment Growth: Actual vs Nowcast')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.grid(axis='y', alpha=0.3)

    # Error in jobs added (actual − nowcast, thousands)
    ax = axes[1]
    ax.bar(x_dates, [r['error_change_k'] for r in results], color='steelblue', alpha=0.7, width=18,
           label='Error in jobs added')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_ylabel('Error (thousands)')
    ax.set_title('Nowcast Error in Jobs Added (Actual − Nowcast; positive = we under-nowcast gain)')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'nowcast_backtest.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {OUTPUT_DIR / "nowcast_backtest.png"}')

    return results


if __name__ == '__main__':
    run_backtest()
