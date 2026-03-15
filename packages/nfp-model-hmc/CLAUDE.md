# nfp-model-hmc

Bayesian state space model for nowcasting U.S. employment growth (NFP).

## Overview

PyMC model definition, sampling, diagnostics, backtesting, and forecasting. Provides:
- **Model** (`model.py`): PyMC state space model with AR(1) latent growth, structural BD, era-specific means
- **Sampling** (`sampling.py`): nutpie (preferred on Apple Silicon) / PyMC NUTS with preset configs
- **Panel adapter** (`panel_adapter.py`): panel → model data dict, cyclical indicators, as-of censoring
- **Diagnostics** (`diagnostics.py`): parameter summary, precision budget, divergences
- **Checks** (`checks.py`): prior/posterior predictive checks, LOO-CV, era summary
- **Residuals** (`residuals.py`): standardized residuals by source
- **Plots** (`plots.py`): growth/seasonal, reconstructed index, BD diagnostics
- **Forecast** (`forecast.py`): forward simulation with structural BD propagation
- **Backtest** (`backtest.py`): real-time vintage-aware nowcast backtest
- **Benchmark** (`benchmark.py`, `benchmark_backtest.py`, `benchmark_plots.py`): benchmark revision inference and evaluation
- **Sensitivity** (`sensitivity.py`): QCEW sigma sensitivity analysis
- **Config** (`config.py`): model hyperparameters, era breaks, prior constants

## Tech Stack

- **Language**: Python 3.12 (requires >= 3.10)
- **Core**: PyMC, PyTensor, ArviZ, nutpie
- **Data**: Polars, NumPy, Matplotlib
- **Config**: Pydantic, python-dotenv, tomli-w
- **Build**: hatchling
- **Internal deps**: `nfp-lookups` (schemas, industry, revision schedules, provider config), `nfp-ingest` (panel, vintage store), `nfp-vintages` (views)

## Key Commands

```bash
# Run the main v3 model pipeline
python alt_nfp_estimation_v3.py

# Run the nowcast backtest (real-time vintage-aware, 24 months)
python -c "from nfp_models.backtest import run_backtest; run_backtest()"

# Run backtest with custom start date and output directory
python -c "from datetime import date; from pathlib import Path; from nfp_models.backtest import run_backtest; run_backtest(n_backtest=12, start_date=date(2024,6,12), output_dir=Path('output/bt_2024h2'))"

# Run the benchmark revision diagnostic
python scripts/benchmark_diagnostic.py

# Run the benchmark backtest with as-of censoring
python -c "from nfp_models.benchmark_backtest import run_benchmark_backtest; run_benchmark_backtest()"

# Run the QCEW sigma sensitivity analysis
python -c "from nfp_models.sensitivity import run_sensitivity; run_sensitivity()"

# Run model tests
pytest tests/

# Lint
ruff check src/nfp_models/
```

## Package Structure

```
src/nfp_models/
├── __init__.py
├── config.py               # Model hyperparameters: N_ERAS, ERA_BREAKS, QCEW_NU, prior constants
│                           #   LOG_SIGMA_* priors, QCEW_POST_COVID_BOUNDARY_ERA_MULT, PP_COLORS
├── settings.py             # NowcastConfig (Pydantic), load_config(), save_config()
├── panel_adapter.py        # panel_to_model_data(), build_obs_sources(), cyclical indicator loading
├── model.py                # build_model() — PyMC state space model
├── sampling.py             # sample_model() — nutpie/PyMC with preset configs
├── diagnostics.py          # print_summary(), compute_precision_budget(), print_weight_staleness()
├── checks.py               # prior/posterior predictive checks, run_loo_cv(), print_era_summary()
├── residuals.py            # Standardized residuals by source
├── plots.py                # Growth/seasonal, reconstructed index, BD diagnostics
├── forecast.py             # Forward simulation with structural BD propagation
├── backtest.py             # run_backtest() — vintage-aware nowcast backtest
├── benchmark.py            # extract_benchmark_revision(), summarize_revision_posterior()
├── benchmark_backtest.py   # run_benchmark_backtest(), compute_backtest_metrics()
├── benchmark_plots.py      # Benchmark diagnostic visualizations
└── sensitivity.py          # run_sensitivity() — QCEW sigma sensitivity analysis
```

## Code Style

- **Formatter**: black (line length 100)
- **Linter**: ruff (line length 100, rules: E, W, F, I, B, C4, UP)
- Line length limit: 100 characters

## Key Patterns

- **Config** (`config.py`): model-only constants — `N_HARMONICS`, `N_ERAS`, `ERA_BREAKS`, `QCEW_NU`, all `LOG_SIGMA_*` prior hyperparameters, `QCEW_POST_COVID_BOUNDARY_ERA_MULT`, `PP_COLORS`. `ProviderConfig` and `CYCLICAL_INDICATORS` live in `nfp_lookups`.
- **Era-specific mean growth, shared persistence**: AR(1) latent growth uses era-indexed `mu_g_era` (Pre-COVID / Post-COVID). Persistence `phi_raw ~ Beta(18, 2)` and marginal SD `tau` shared across eras. `phi` capped at 0.99. `pytensor.scan` passes per-timestep `mu_g[t]`; `phi` and `sigma_g` are non-sequences.
- **Structural BD model**: `bd_t = phi_0 + phi_1*X^birth + phi_3*X^cycle + sigma_bd*xi_t` where `X^cycle = [claims, jolts]`. `phi_1` gated out when birth covariate is all-zero (backtest iterations without BD data).
- **CES best-available print**: one obs per month per SA/NSA using highest revision (2 > 1 > 0). Model uses `sigma_ces_sa[vintage_idx]` for per-obs noise. `_ces_best_available()` in `panel_adapter.py` selects; benchmark-revised rows never used as CES observations.
- **QCEW Student-t likelihood**: `pm.StudentT` with `QCEW_NU=5`. Per-obs sigma = `base_sigma_tier * revision_mult * era_mult`. Two estimated LogNormal base sigmas (M2 vs M3+M1). LogNormal priors prevent funnel geometry.
- **Precision budget** (`diagnostics.py`): Fisher information by source. CES uses vintage-indexed sigmas; QCEW uses Student-t adjusted precision; providers use signal loadings.
- **Nowcast backtest** (`backtest.py`): per target month T, builds censored panel via `build_panel(as_of_ref=T)` then `panel_to_model_data(panel, providers, as_of=T)`. Saves InferenceData + summary parquet.
- **LOO-CV** is a data quality audit, not model evaluation. High k-hat on QCEW is structural (high-precision dominates latent state).
- **Forecast** (`forecast.py`): expects `data["levels"]` as Polars DataFrame. Employment in thousands (BLS convention). Index starts at base=100 at first valid CES observation.

## Bayesian Workflow Reference

Bayesian workflow is iterative: model building → inference → checking → expansion.

### Workflow Checklist

1. EDA and model design
2. Set priors (weakly informative; run prior predictive checks)
3. Fake-data simulation (verify parameter recovery)
4. Fit the model (start fast, refine to full MCMC)
5. Computational diagnostics (R-hat < 1.01, ESS, divergences)
6. Posterior predictive checks
7. Cross-validation (LOO-CV)
8. Iterate based on findings
9. Compare models (LOO-CV, stacking)
10. Report with uncertainty

### Computational Diagnostics

- **Divergent transitions**: concentrated in small region → funnel pathology; no pattern → increase adapt_delta
- **Funnel fix**: non-centered parameterization
- **Folk Theorem**: computational problems often indicate model problems

### Key References

- Gelman et al. (2020). Bayesian Workflow. arXiv:2011.01808
- Gabry et al. (2019). Visualization in Bayesian Workflow. JRSS-A
- Vehtari, Gelman, & Gabry (2017). Practical Bayesian model evaluation using LOO-CV and WAIC
- Betancourt (2017). A Conceptual Introduction to Hamiltonian Monte Carlo

## Test Mapping

Tests from the monorepo `tests/` that belong here:
- `test_model.py` — model construction tests (era-specific + scalar fallback)
- `test_benchmark_backtest.py` — benchmark backtest infrastructure tests
- `test_backtesting_smoke.py` — integration smoke tests with real panel data
- `test_precision_budget.py` — precision budget DataFrame structure tests
- `test_sensitivity_smoke.py` — sensitivity analysis smoke tests
