# CLAUDE.md

## Project Overview

Bayesian state space model for nowcasting U.S. employment growth (NFP) using multiple data sources (CES, QCEW, payroll provider data). Built with PyMC for Bayesian inference, Polars for data manipulation, and Marimo for interactive notebooks.

The current implementation (v3) lives in the `src/alt_nfp` Python package. Earlier monolithic scripts are preserved in `archive/`.

## Tech Stack

- **Language**: Python 3.12 (requires >= 3.10)
- **Package Manager**: uv
- **Build Backend**: hatchling
- **Core Libraries**: PyMC, PyTensor, ArviZ, Polars, NumPy, Matplotlib, Marimo
- **Sampler**: nutpie (preferred on Apple Silicon), falls back to PyMC NUTS

## Key Commands

```bash
# Install dependencies
uv sync

# Run the main v3 model pipeline
python alt_nfp_estimation_v3.py

# Run the nowcast backtest (24-month CES-censoring experiment)
python -c "from alt_nfp.backtest import run_backtest; run_backtest()"

# Run the QCEW sigma sensitivity analysis
python -c "from alt_nfp.sensitivity import run_sensitivity; run_sensitivity()"

# Format code
black . --line-length 100

# Lint
ruff check .

# Type check
mypy .

# Run tests
pytest tests/
```

## Project Structure

```
alt_nfp/
├── alt_nfp_estimation_v3.py  # Thin runner script for the full pipeline
├── src/alt_nfp/              # Core Python package
│   ├── __init__.py           # Package exports
│   ├── config.py             # ProviderConfig, PROVIDERS list, paths, constants
│   ├── data.py               # existing — legacy data loading
│   ├── model.py              # existing — PyMC model
│   ├── sampling.py           # sample_model() — nutpie/PyMC with preset configs
│   ├── diagnostics.py        # Parameter summary, source contributions, divergences
│   ├── checks.py             # Prior/posterior predictive checks, LOO-CV
│   ├── residuals.py          # Standardised residuals by source
│   ├── plots.py              # Growth/seasonal, reconstructed index, BD diagnostics
│   ├── forecast.py           # Forward simulation with structural BD propagation
│   ├── backtest.py           # Nowcast backtest (CES-censoring experiment)
│   ├── sensitivity.py        # QCEW sigma sensitivity analysis
│   ├── lookups/              # Static reference tables
│   │   ├── __init__.py
│   │   ├── industry.py       # NAICS → supersector → domain hierarchy + CES series ID map
│   │   ├── revision_schedules.py  # QCEW & CES vintage schedules + publication calendar
│   │   └── geography.py      # placeholder for future geo hierarchy
│   ├── ingest/               # Raw data → observation panel
│   │   ├── __init__.py
│   │   ├── base.py           # PANEL_SCHEMA, validate_panel
│   │   ├── qcew.py           # QCEW ingestion (BLS API + historical parquet)
│   │   ├── ces.py            # CES ingestion (BLS API + historical parquet)
│   │   ├── payroll.py        # provider index ingestion
│   │   └── panel.py          # build_panel, save/load
│   └── vintages/             # Vintage views & evaluation
│       ├── __init__.py
│       ├── views.py          # real_time_view, final_view
│       └── evaluation.py     # vintage_diff, noise multiplier builder
├── data/                     # Input CSV data (CES, QCEW, payroll provider indices)
│   └── raw/                  # Historical revision data
│       ├── vintages/         # Parquet files (QCEW/CES vintages, pub calendar)
│       └── providers/        # Raw provider data by provider
├── tests/                    # Test suite
│   ├── test_lookups.py       # Industry hierarchy & revision schedule tests
│   ├── test_ingest.py        # Panel validation & schema tests
│   └── test_vintages.py      # Vintage view & evaluation tests
├── docs/                     # Methodology and evaluation documentation
├── archive/                  # Earlier monolithic scripts (v1, v2)
├── output/                   # Generated results (InferenceData, plots)
├── pyproject.toml            # Project config, dependencies, tool settings
└── uv.lock                   # Dependency lock file
```

## Code Style

- **Formatter**: black (line length 100, targets py310-py312)
- **Linter**: ruff (line length 100, rules: E, W, F, I, B, C4, UP)
- **Type checking**: mypy (ignore_missing_imports=true)
- Line length limit: 100 characters

## Key Code Patterns

- **Config-driven providers**: adding a new payroll provider requires only a new `ProviderConfig` entry in `config.py`; data loading, model likelihood, diagnostics, plots, and forecasts adapt automatically.
- **Structural birth/death model**: `bd_t = φ₀ + φ₁·X^cycle + φ₂·BD^QCEW_{t-L} + σ_bd·ξ_t` replaces the v2 constant BD offset.
- **Provider-specific error structures**: each provider can have `iid` or `ar1` measurement error.
- `alt_nfp_estimation_v3.py` is a thin runner that orchestrates: data loading → model building → prior checks → sampling → diagnostics → PPC → LOO → plots → forecast → save.
- Data is loaded from CSV files in `data/` using Polars, transformed to growth rates.
- PyMC models are built declaratively; sampling uses nutpie when available.
- Output artifacts (NetCDF inference data, PNG plots) go to `output/`.

---

## Bayesian Workflow Reference

Bayesian workflow is iterative: model building → inference → checking → expansion. Expect to fit many models. Poor models and failures are informative steps, not dead ends.

### Workflow Checklist

1. **EDA and model design:** Explore data, choose initial model structure, build modularly
2. **Set priors:** Use weakly informative priors; run prior predictive checks
3. **Fake-data simulation:** Verify the model can recover known parameters
4. **Fit the model:** Start fast and approximate, refine to full MCMC
5. **Computational diagnostics:** Check R̂ < 1.01, ESS, divergences
6. **Posterior predictive checks:** Compare replicated data to observed data
7. **Cross-validation:** Assess predictive performance (LOO-CV), identify influential points
8. **Iterate:** Expand or modify model based on findings
9. **Compare models:** Use LOO-CV, stacking, pointwise diagnostics
10. **Report:** Present results with uncertainty, document workflow decisions

### Prior Selection

Ladder from least to most informative:
1. Flat / improper prior
2. Super-vague but proper
3. Very weakly informative
4. Generic weakly informative
5. Specific informative

**Weakly informative priors** should generate plausible (not indistinguishable-from-observed) data under the prior predictive distribution. As models grow more complex, priors generally need to become tighter.

### Computational Diagnostics

**The Folk Theorem:** When you have computational problems, often there's a problem with your model.

**HMC-specific:**
- **Divergent transitions:** Indicate regions of high curvature. Visualize in scatterplots or parallel coordinate plots.
  - Concentrated in small region → geometric pathology (e.g., funnel)
  - No obvious pattern → likely false positives (increase `adapt_delta`)
- **Funnel pathologies:** Fix with non-centered parameterization
- **Debugging:** Simplify failing model until something works; start simple and add features until problem appears

### Model Evaluation

**Posterior predictive checks:**
- Density overlays: Compare replicated data distribution to observed
- Test statistics: Use statistics orthogonal to model parameters (e.g., skewness for Gaussian models)
- Grouped checks: Evaluate within levels of grouping variables

**Cross-validation (LOO-CV):**
- LOO-PIT should be uniform for well-calibrated models
- "Frown" shape → predictive distributions too wide
- "Smile" shape → predictive distributions too narrow
- k̂ diagnostic flags influential observations

### Model Comparison

- Use LOO-CV (via PSIS-LOO) to compare predictive performance
- When comparison is uncertain, use **stacking** to combine model inferences
- Center model expansions on current model (weakly informative extensions)

### Key References

- Gelman et al. (2020). Bayesian Workflow. arXiv:2011.01808
- Gabry et al. (2019). Visualization in Bayesian Workflow. JRSS-A
- Vehtari, Gelman, & Gabry (2017). Practical Bayesian model evaluation using LOO-CV and WAIC
- Betancourt (2017). A Conceptual Introduction to Hamiltonian Monte Carlo
