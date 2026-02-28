# TODO: National Model Phase 2 — Model Enhancements

## Context

Phase 2 implements targeted improvements to `build_model()` in
`src/alt_nfp/model.py`. Each enhancement is gated by a data-dict key so
the Phase 1 baseline model remains recoverable. Enhancements are
evaluated sequentially — implement → backtest → accept/reject — before
moving to the next. The evaluation protocol is the same for all:

1. Run `run_benchmark_backtest()` with and without the enhancement.
2. Run nowcast backtesting (one-step-ahead RMSE against CES first print).
3. Check sampler diagnostics: R-hat, ESS, divergences.
4. Accept if RMSE improves at target horizons without degrading others,
   and sampler diagnostics remain clean.

**Prerequisite:** Phase 1 evaluation infrastructure is complete (leakage-safe
backtesting, all four cyclical indicators loading, QCEW sensitivity and
precision budget diagnostics).

**Spec reference:** `national_model_spec.md` §3.


---


## Task 2a — Era-Specific Latent State Parameters

**Priority: High (nowcast). Spec §3.3.**

Mean growth `μ_g` and AR(1) persistence `φ` differ structurally across
macroeconomic eras. A single parameter set across 2003–present biases
dynamics toward an average that doesn't represent any actual era.

### 2a.1  Add era configuration to `config.py`

```python
from datetime import date

N_ERAS = 3
ERA_BREAKS: list[date] = [date(2009, 1, 1), date(2020, 1, 1)]
# Era 0: Pre-GFC    (2003-01 → 2008-12)
# Era 1: Post-GFC   (2009-01 → 2019-12)
# Era 2: Post-COVID  (2020-01 → present)
```

Files: `src/alt_nfp/config.py`

### 2a.2  Compute `era_idx` in `panel_to_model_data()`

Add a helper `_date_to_era(d: date) -> int` that maps a date to its era
index using `ERA_BREAKS`. Compute `era_idx` as a length-T int array and
add it to the data dict.

```python
def _date_to_era(d: date) -> int:
    for i, brk in enumerate(ERA_BREAKS):
        if d < brk:
            return i
    return len(ERA_BREAKS)

# In panel_to_model_data():
era_idx = np.array([_date_to_era(d) for d in dates])
# ... add to returned dict as 'era_idx'
```

Files: `src/alt_nfp/panel_adapter.py`

### 2a.3  Replace scalar `mu_g` and `phi` with era-indexed vectors in `build_model()`

Gate on `data.get('era_idx')` so the model falls back to scalar
parameters when era info is absent.

```python
era_idx = data.get('era_idx')

if era_idx is not None:
    mu_g_era = pm.Normal('mu_g_era', mu=0.001, sigma=0.005, shape=N_ERAS)
    phi_raw_era = pm.Beta('phi_raw_era', alpha=18, beta=2, shape=N_ERAS)
    mu_g = mu_g_era[era_idx]    # length-T
    phi = phi_raw_era[era_idx]  # length-T
else:
    mu_g = pm.Normal('mu_g', mu=0.001, sigma=0.005)
    phi = pm.Uniform('phi', lower=0.0, upper=0.99)
```

Modify the `pytensor.scan` AR(1) step function so that `mu_g` and `phi`
are indexed per-timestep. At era boundaries the dynamics parameters
switch discretely — the latent state `μ_{t-1}^cont` carries forward
continuously.

**Prior note:** Start with independent priors per era. If posteriors are
implausibly separated, move to hierarchical with shared hyperparameters.
The `phi` prior changes from `Uniform(0, 0.99)` to `Beta(18, 2)` (mode
≈ 0.94) — this is intentional and applies only when era-specific
parameters are active.

Adds 4 parameters (2 extra `mu_g_era` + 2 extra `phi_raw_era`).
Negligible runtime impact.

Files: `src/alt_nfp/model.py`

### 2a.4  Update diagnostics for era-specific posteriors

Update `checks.py` to report era-specific posterior summaries for
`mu_g_era` and `phi_raw_era`. Print a table showing posterior mean ± sd
per era.

Files: `src/alt_nfp/checks.py`

### 2a.5  Tests

Add test to `tests/test_model.py`:
- Construct model with `era_idx` in data dict, verify it builds and
  has `mu_g_era` / `phi_raw_era` variables with correct shapes.
- Construct model without `era_idx`, verify scalar `mu_g` / `phi`
  still work (backward compatibility).

### 2a.6  Evaluation criteria

- Nowcast RMSE (one-step-ahead vs CES first print) improves or is
  neutral relative to Phase 1 baseline.
- Era-specific posteriors separate meaningfully (non-overlapping or
  partially overlapping credible intervals). If all eras overlap,
  reject — adds complexity without value.
- Inspect boundary behavior at Jan 2009 and Jan 2020 for artifacts.
- Benchmark RMSE by horizon: no degradation.


---


## Task 2b — Cyclical Indicators Fully Wired + JOLTS

**Priority: High (nowcast). Spec §3.4. Depends on Phase 1 §2.3 data
loading being complete.**

The `build_model()` code already handles cyclical indicators via the
`cyclical_keys` loop. If Phase 1 delivered all four indicators with
correct loading and censoring, the remaining work is evaluation and
tuning — not model construction.

### 2b.1  Verify indicators flow through `build_model()` correctly

Confirm that all four cyclical indicators (initial claims, NFCI, BFS,
JOLTS) are present in the data dict as centered covariates with keys
matching the `cyclical_keys` loop. Run the model once and verify that
`φ_3` coefficients are estimated for each indicator.

Files: `src/alt_nfp/model.py`, `src/alt_nfp/panel_adapter.py`

### 2b.2  JOLTS series selection

Evaluate JOLTS openings (`JTSJOL`) vs hires (`JTSHIL`) as the BD
covariate. Run the model with each series separately. Pick the one with
stronger out-of-sample BD prediction (lower BD component RMSE against
annual benchmark revisions). Document the choice.

### 2b.3  Evaluate individual indicator contributions

Run the model with each indicator individually and all together. Compare
BD component accuracy across configurations. This determines whether all
four add value or whether a subset is sufficient.

### 2b.4  Consider informative priors on indicator loadings

Currently all cyclical loadings share `φ_3 ~ N(0, 0.3²)`. If some
indicators have known sign (higher claims → lower BD), consider
sign-constrained or informative priors. For example:

```python
# If claims have reliably negative BD loading:
phi_claims = pm.Normal('phi_claims', mu=-0.2, sigma=0.3)
```

Only apply if the unrestricted posterior strongly supports the sign.

Files: `src/alt_nfp/model.py` (if prior changes needed)

### 2b.5  Evaluation criteria

- BD component accuracy against annual benchmark revisions (primary
  metric for cyclical indicators).
- Does JOLTS improve BD accuracy beyond the three existing indicators?
- Nowcast RMSE improvement, especially during turning-point months
  (2008–2009, 2020, any recent slowdowns).


---


## Task 2c — National-Level Time-Varying Provider Bias

**Priority: Medium (nowcast). Spec §3.5.**

Provider composition drifts as clients are gained and lost. A static
`α_p` absorbs average bias but cannot track this drift.

### 2c.1  Replace static `alpha_p` with a Gaussian random walk

Use non-centered parameterization to avoid funnel geometry:

```python
sigma_alpha_drift_p = pm.HalfNormal(
    f'sigma_alpha_drift_{pp_name}', sigma=0.0005
)
alpha_p_innovations = pm.Normal(
    f'alpha_innovations_{pp_name}', 0, 1, shape=T
)
alpha_p_0 = pm.Normal(
    f'alpha_0_{pp_name}', mu=0, sigma=0.005
)
alpha_p = pm.Deterministic(
    f'alpha_{pp_name}',
    alpha_p_0 + pt.cumsum(sigma_alpha_drift_p * alpha_p_innovations)
)
```

Gate with a flag (e.g., `data.get('use_time_varying_bias', False)`) so
the static version remains recoverable.

For AR(1) error providers, the conditional mean uses `alpha_p[t]`
instead of scalar `alpha_p`. Index into the per-observation timesteps.

**Computational cost:** Adds `2 * P` parameters (`σ_α_drift` + `α_0`
per provider) plus `T * P` innovations. For 2 providers and T ≈ 270,
this is ~544 new parameters. Non-centered parameterization keeps
sampling efficient, but monitor ESS and runtime.

Files: `src/alt_nfp/model.py`

### 2c.2  Add `α_p(t)` trajectory diagnostics

Add a function in `checks.py` that plots the posterior mean and 90%
credible interval of `α_p(t)` over time for each provider. The
trajectory should be smooth (no high-frequency oscillation).

Files: `src/alt_nfp/checks.py`

### 2c.3  Tests

Add test to `tests/test_model.py`:
- Construct model with `use_time_varying_bias=True`, verify it builds
  and has `sigma_alpha_drift_*`, `alpha_innovations_*`, `alpha_0_*`,
  and `alpha_*` Deterministic with shape `(T,)`.
- Construct model with `use_time_varying_bias=False` (or absent),
  verify static `alpha_*` scalar still works.

### 2c.4  Acceptance criteria (strict — reject if any fail)

- Nowcast RMSE improves or is neutral.
- `α_p(t)` trajectories are smooth (no high-frequency oscillation).
- `σ_α_drift` posterior is bounded away from prior upper tail (data is
  informative about drift rate).
- No sampler pathologies (divergences, low ESS on drift parameters).

**Rejection rule:** If `σ_α_drift` posterior concentrates near zero for
all providers, reject the enhancement and keep static bias.


---


## Task 2d — BD Error-Correction Term

**Priority: High (benchmark). Spec §3.6. Phase 3a from
`specs/benchmark_spec.md` §5.2.**

Add an error-correction term to the structural BD equation that
conditions on the prior year's known benchmark revision.

### 2d.1  Add `prior_benchmark_revision` to data dict

In `panel_to_model_data()`, compute the monthly rate of the prior
benchmark revision and add it to the data dict. For benchmark year Y,
this is `BENCHMARK_REVISIONS[Y-1]`. For years without a prior revision
or COVID years, set to `0.0`.

```python
# For each benchmark year in the sample:
# prior_revision_rate_t = R_{Y-1} / (12 * L)
# where L is total employment level (~155M)
```

The term is constant within each benchmark year (April Y through March
Y+1).

Files: `src/alt_nfp/panel_adapter.py`

### 2d.2  Add error-correction to `build_model()`

Gate on `data.get('prior_benchmark_revision')`:

```python
if data.get('prior_benchmark_revision') is not None:
    phi_bm = pm.Normal('phi_bm', mu=0, sigma=0.5)
    prior_revision_rate = (
        data['prior_benchmark_revision'] / (12 * data['employment_level'])
    )
    bd_t = bd_t + phi_bm * prior_revision_rate
```

`φ_BM ~ N(0, 0.5)` centered at zero means no correction by default —
data determines whether BD drift persists from year to year.

Files: `src/alt_nfp/model.py`

### 2d.3  Tests

Add test to `tests/test_model.py`:
- Construct model with `prior_benchmark_revision` in data dict, verify
  `phi_bm` exists in model.
- Construct model without it, verify baseline BD equation unchanged.

### 2d.4  Evaluation criteria

Per `specs/benchmark_spec.md` §5.5:
- RMSE at T−12 improves by >5% relative to baseline.
- No degradation at T−1 (where QCEW dominates).
- Sampler diagnostics remain clean.
- Small backtest sample (limited benchmark revision history), so prior
  does heavy lifting. Evaluate primarily via posterior predictive checks.


---


## Task 2e — Preliminary Benchmark as Observation

**Priority: Medium (benchmark). Spec §3.7. Phase 3b from
`specs/benchmark_spec.md` §5.3.**

At T−5 and later horizons, treat the BLS preliminary benchmark estimate
as a biased, noisy observation of the model's implied revision.

### 2e.1  Add preliminary benchmark lookup

Add to `src/alt_nfp/lookups/benchmark_revisions.py`:

```python
PRELIMINARY_BENCHMARKS: dict[int, tuple[float, date]] = {
    # march_year: (preliminary_revision_000s, publication_date)
    2024: (-818, date(2024, 8, 21)),
    2025: (-911, date(2025, 8, 20)),
}
```

Files: `src/alt_nfp/lookups/benchmark_revisions.py`

### 2e.2  Feed preliminary data into data dict

In `panel_to_model_data()`, when `as_of >= preliminary_publication_date`,
add to data dict:
- `preliminary_benchmark` — the published revision (thousands)
- `benchmark_window_indices` — indices of months in the benchmark year
  (April Y through March Y+1)
- `cum_ces_nsa_benchmark` — cumulated CES NSA growth over the benchmark
  window
- `prior_benchmark_level` — employment level at start of benchmark
  window

Respect `as_of` censoring: only include preliminary data when it would
have been published.

Files: `src/alt_nfp/panel_adapter.py`

### 2e.3  Add preliminary benchmark observation to `build_model()`

Gate on `data.get('preliminary_benchmark')`:

```python
if data.get('preliminary_benchmark') is not None:
    window_idx = data['benchmark_window_indices']
    cum_latent = pt.sum(g_total_nsa[window_idx])
    cum_ces = data['cum_ces_nsa_benchmark']
    L_prior = data['prior_benchmark_level']
    implied_revision = L_prior * (pt.exp(cum_latent) - np.exp(cum_ces)) / 1000

    bias_prelim = pm.Normal('bias_prelim', mu=-150, sigma=75)
    sigma_prelim = pm.HalfNormal('sigma_prelim', sigma=100)

    pm.Normal(
        'obs_preliminary',
        mu=implied_revision + bias_prelim,
        sigma=sigma_prelim,
        observed=data['preliminary_benchmark'],
    )
```

`bias_prelim ~ N(-150, 75)` captures the systematic overshoot
(100–220K pattern in recent years). `σ_prelim ~ HalfNormal(100)`
captures idiosyncratic preliminary-to-final noise.

Files: `src/alt_nfp/model.py`

### 2e.4  Tests

Add test to `tests/test_model.py`:
- Construct model with preliminary benchmark data, verify
  `obs_preliminary`, `bias_prelim`, `sigma_prelim` exist.
- Construct model without it, verify baseline unchanged.

### 2e.5  Evaluation criteria

Per `specs/benchmark_spec.md` §5.5:
- RMSE at T−5 improves relative to baseline.
- No degradation at T−1.
- Very limited backtest history (2024–2025 only), so prior does most
  of the work. Evaluate primarily via posterior predictive checks on
  available years.


---


## Implementation Order

Tasks are sequential. Each must be evaluated and accepted/rejected
before starting the next.

| Step | Task | Target metric | New params |
|------|------|---------------|------------|
| 1 | 2a: Era-specific `μ_g` and `φ` | Nowcast RMSE | +4 |
| 2 | 2b: Cyclical indicators + JOLTS | BD accuracy, turning-point RMSE | 0 (evaluation only) |
| 3 | 2c: Time-varying provider bias | Nowcast RMSE | ~544 |
| 4 | 2d: BD error-correction | Benchmark RMSE at T−12 | +1 |
| 5 | 2e: Preliminary benchmark obs | Benchmark RMSE at T−5 | +2 |


## Files Modified (Summary)

| File | Tasks |
|------|-------|
| `src/alt_nfp/config.py` | 2a |
| `src/alt_nfp/panel_adapter.py` | 2a, 2d, 2e |
| `src/alt_nfp/model.py` | 2a, 2b (if priors change), 2c, 2d, 2e |
| `src/alt_nfp/checks.py` | 2a, 2c |
| `src/alt_nfp/lookups/benchmark_revisions.py` | 2e |
| `tests/test_model.py` | 2a, 2c, 2d, 2e |


## Risks

| Risk | Mitigation |
|------|------------|
| Era breakpoints misspecified → boundary artifacts | Inspect boundary behavior; consider soft transitions if artifacts appear |
| Time-varying bias absorbs latent state signal | Tight prior on `σ_α_drift`; reject if posterior → 0 |
| JOLTS revisions material but ignored | Empirical revision assessment done in Phase 1; add ALFRED vintages if needed |
| Small benchmark backtest sample (2d/2e) | Strong priors; LOO-CV within backtest; extend to pre-provider era if feasible |
| Stacking enhancements obscures individual contributions | Sequential evaluation with accept/reject gates |