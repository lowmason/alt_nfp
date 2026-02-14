# -------------------------------------------------------------------------------------------------
# Imports and parameters
# -------------------------------------------------------------------------------------------------

import marimo

__generated_with = "0.19.8"
app = marimo.App()


@app.cell
def _():
    # -------------------------------------------------------------------------------------------------
    # Imports and parameters
    # -------------------------------------------------------------------------------------------------

    import numpy as np
    import polars as pl

    import pymc as pm
    import pytensor
    import pytensor.tensor as pt

    BASE_DIR = '/Users/lowell/Projects/pp'
    return BASE_DIR, np, pl, pm, pt, pytensor


@app.cell
def _(BASE_DIR, np, pl):
    # -------------------------------------------------------------------------------------------------
    # Read index data
    # -------------------------------------------------------------------------------------------------

    def read_index(index_name: str) -> pl.DataFrame:

        return (
            pl.read_csv(
                f'{BASE_DIR}/data/{index_name}_index.csv'
            )
            .with_columns(
                ref_date=pl.col('ref_date')
                           .str.to_date('%m/%d/%Y')
            )
            .sort('ref_date')
        )

    qcew = read_index('qcew')
    ces  = read_index('ces')
    pp = read_index('pp')
    df = (
        ces
        .join(
            qcew, 
            on='ref_date', 
            how='full',
            coalesce=True
        )
        .join(
            pp,
            on='ref_date', 
            how='full',
            coalesce=True
        )
        .sort('ref_date')
    )

    cal = (
        pl
        .select(
            ref_date=pl.date_range(
                df['ref_date'].min(), 
                df['ref_date'].max(), 
                interval='1mo', 
                eager=True
            )
        )
    )

    df = (
        cal
        .join(
            df,
            on='ref_date',
            how='left'
        )
        .sort('ref_date')
        .select(
            ref_date=pl.col('ref_date'),
            ces=pl.col('ces_index').log(),
            qcew=pl.col('qcew_index').log(),
            pp1=pl.col('pp_index_1').log(),
            pp2=pl.col('pp_index_2').log(),
            pp3=pl.col('pp_index_3').log(),
            pp4=pl.col('pp_index_4').log()
        )
    )
    series_cols = ['ces', 'qcew', 'pp1', 'pp2', 'pp3', 'pp4']
    dates = df['ref_date'].to_list()

    _T = len(dates)
    _J = len(series_cols)

    y = np.vstack([
        df.select(pl.col(col)).to_numpy().astype(float).reshape(-1)  # (T,)
        for col in series_cols
    ])
    # Polars nulls -> nan already in float array; ensure:
    y = np.where(np.isfinite(y), y, np.nan)

    series_names = ['ces', 'qcew', 'pp1', 'pp2', 'pp3', 'pp4']

    print('T months:', _T)
    print('J series:', _J)
    print('Date range:', dates[0], '->', dates[-1])
    return series_names, y


@app.cell
def _(np, pm, pt, pytensor, series_names, y):
    obs_mask = ~np.isnan(y)
    J, T = y.shape

    # Indices for series (assumes consistent ordering)
    idx_ces = series_names.index('ces')
    idx_qcew = series_names.index('qcew')
    pp_idxs = [i for i, n in enumerate(series_names) if n.startswith('pp')]

    # Indices for 'non-CES' series to estimate alpha/lam
    other_idxs = [i for i in range(J) if i != idx_ces]
    with pm.Model() as model:

        # -----------------------------
        # Latent log growth: AR(1)
        # -----------------------------

        mu_g = pm.Normal('mu_g', mu=0.0, sigma=0.02)                         # mean monthly log growth
        phi = pm.Uniform('phi', lower=-0.99, upper=0.99)                     # AR(1) coefficient
        sigma_g = pm.HalfNormal('sigma_g', sigma=0.02)

        eps = pm.Normal('eps', 0.0, 1.0, shape=T)
        g0 = pm.Normal('g0', mu=mu_g, sigma=0.05)

        # Use scan to build AR(1) process efficiently
        def ar1_step(eps_t, g_prev, mu_g, phi, sigma_g):
            g_t = mu_g + phi * (g_prev - mu_g) + sigma_g * eps_t
            return g_t

        g_rest, _ = pytensor.scan(
            fn=ar1_step,
            sequences=[eps[1:]],
            outputs_info=[g0],
            non_sequences=[mu_g, phi, sigma_g],
            strict=True
        )

        g = pt.concatenate([pt.shape_padleft(g0), g_rest])
        pm.Deterministic('g', g)

        # -----------------------------
        # Latent log index level (integral of growth)
        # -----------------------------
        x0 = pm.Normal('x0', mu=np.log(100.0), sigma=0.2)  # base ~ log(100)
        x = x0 + pt.cumsum(g)

        pm.Deterministic('x', x)
        pm.Deterministic('index_latent', pt.exp(x))
        pm.Deterministic('growth_pct', 100.0 * (pt.exp(g) - 1.0))

        # -----------------------------
        # Measurement model (CES anchored)
        # -----------------------------
        # CES defines the scale: alpha_ces = 0, lam_ces = 1
        # Other series get alpha/lam relative to CES scale.
        alpha_other = pm.Normal('alpha_other', mu=0.0, sigma=0.05, shape=len(other_idxs))
        lam_other = pm.Normal('lam_other', mu=1.0, sigma=0.10, shape=len(other_idxs))

        # Build full alpha, lam vectors deterministically
        alpha = pt.zeros((J,), dtype='float64')
        lam   = pt.ones((J,), dtype='float64')  # start with 1s so CES is 1 by default

        # Insert the 'other' parameters into their positions
        alpha = pt.set_subtensor(alpha[other_idxs], alpha_other)
        lam   = pt.set_subtensor(lam[other_idxs],   lam_other)

        pm.Deterministic('alpha', alpha)
        pm.Deterministic('lam', lam)

        # -----------------------------
        # Differential noise priors
        # -----------------------------
        # QCEW tighter (levels anchor), PP looser, CES moderate
        sigma_y = pt.zeros((J,), dtype='float64')

        sigma_ces  = pm.HalfNormal('sigma_ces',  sigma=0.02)  # CES: moderate
        sigma_qcew = pm.HalfNormal('sigma_qcew', sigma=0.01)  # QCEW: tighter
        sigma_pp   = pm.HalfNormal('sigma_pp',   sigma=0.05, shape=len(pp_idxs))  # PP: looser

        sigma_y = pt.set_subtensor(sigma_y[idx_ces],  sigma_ces)
        sigma_y = pt.set_subtensor(sigma_y[idx_qcew], sigma_qcew)
        sigma_y = pt.set_subtensor(sigma_y[pp_idxs],  sigma_pp)

        pm.Deterministic('sigma_y', sigma_y)

        # -----------------------------
        # Likelihood with missing masks
        # -----------------------------
        for j in range(J):
            m = obs_mask[j, :]
            pm.Normal(
                f'y_{series_names[j]}',
                mu=alpha[j] + lam[j] * x[m],
                sigma=sigma_y[j],
                observed=y[j, m],
            )

        # Sample with PyMC (nutpie has compatibility issues with cumsum)
        idata = pm.sample(
            draws=2000, 
            tune=2000, 
            chains=4, 
            target_accept=0.95,
            return_inferencedata=True
        )

    return (idata,)


@app.cell
def _(idata):
    # Display sampling results
    import arviz as az

    print("\n" + "="*80)
    print("SAMPLING SUMMARY")
    print("="*80)
    print(az.summary(idata, var_names=['mu_g', 'phi', 'sigma_g', 'sigma_ces', 'sigma_qcew']))
    print("\n" + "="*80)
    print("DIAGNOSTICS")
    print("="*80)
    print(f"Divergences: {idata.sample_stats.diverging.sum().values.sum()}")
    print(f"Max tree depth: {idata.sample_stats.tree_depth.max().values.max()}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
