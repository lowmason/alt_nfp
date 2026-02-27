# Quick Start

Run the full estimation pipeline in a few steps.

## 1. Prepare Data

Place your data files in the `data/` directory:

```
data/
├── ces_index.csv          # CES SA and NSA employment indices
├── qcew_index.csv         # QCEW NSA employment index
├── alt_nfp_index_1.csv    # Provider 1 index
├── alt_nfp_index_2.csv    # Provider 2 index
└── alt_nfp_births_2.csv   # Provider 2 birth-rate data (optional)
```

Each CSV must have a `ref_date` column (date) and the relevant index column(s).

## 2. Run the Full Pipeline

```bash
uv run python alt_nfp_estimation_v3.py
```

This executes the complete Bayesian workflow:

1. **Data loading** — reads CES, QCEW, and provider CSVs; computes
   growth rates and BD covariates.
2. **Model building** — constructs the PyMC state-space model.
3. **Prior predictive checks** — validates priors generate plausible data.
4. **MCMC sampling** — 8 000 draws × 4 chains via nutpie/NUTS.
5. **Convergence diagnostics** — R-hat, ESS, divergence checks.
6. **Posterior predictive checks** — density overlays and test statistics.
7. **LOO-CV** — leave-one-out cross-validation with k-hat diagnostics.
8. **Residual analysis** — standardised residuals by source.
9. **Plots** — growth rates, seasonal, reconstructed index, BD diagnostics.
10. **Forecast** — forward simulation to target date.
11. **Save** — NetCDF inference data and PNG plots to `output/`.

## 3. Run a Backtest

```python
from alt_nfp import run_backtest

results = run_backtest(n_backtest=24)
```

This censors CES data for each of the last 24 months and measures nowcast
accuracy.

## 4. Run Sensitivity Analysis

```python
from alt_nfp import run_sensitivity

results = run_sensitivity()
```

Compares key parameters across 0.5x, 1x, and 2x QCEW noise configurations.

## 5. Using the Panel API

For more control, build an observation panel and convert to model data:

```python
from alt_nfp import build_panel, PROVIDERS
from alt_nfp.panel_adapter import panel_to_model_data
from alt_nfp.model import build_model
from alt_nfp.sampling import sample_model

# Build unified observation panel
panel = build_panel(use_legacy=True)

# Convert to model dict
data = panel_to_model_data(panel, PROVIDERS)

# Build and sample
model = build_model(data)
idata = sample_model(model)
```

## Output

All artifacts are saved to `output/`:

| File | Description |
|---|---|
| `prior_predictive.png` | Prior predictive check |
| `ppc_density.png` | Posterior predictive density overlays |
| `ppc_test_stats.png` | Posterior predictive test statistics |
| `loo_khat.png` | LOO-CV k-hat diagnostics |
| `residuals.png` | Standardised residuals by source |
| `growth_and_seasonal.png` | Growth rates and seasonal pattern |
| `reconstructed_index.png` | Latent index vs observed series |
| `bd_diagnostics.png` | Birth/death decomposition |
| `forecast_sa_nsa.png` | Index forecast (SA/NSA) |
| `forecast_levels.png` | Jobs-added forecast |
| `divergences.png` | Divergence diagnostics |
| `sensitivity_qcew_sigma.png` | QCEW sigma sensitivity |
| `nowcast_backtest.png` | Backtest results |
