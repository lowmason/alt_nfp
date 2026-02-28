# CLAUDE.md

## Project Overview

Bayesian state space model for nowcasting U.S. employment growth (NFP) using multiple data sources (CES, QCEW, payroll provider data). Built with PyMC for Bayesian inference, Polars for data manipulation, and Marimo for interactive notebooks.

The current implementation (v3) lives in the `src/alt_nfp` Python package. Earlier monolithic scripts are preserved in `archive/`.

## Tech Stack

- **Language**: Python 3.12 (requires >= 3.10)
- **Package Manager**: uv
- **Build Backend**: hatchling
- **Core Libraries**: PyMC, PyTensor, ArviZ, Polars, NumPy, Matplotlib, Marimo
- **Data Ingestion**: httpx (async HTTP with HTTP/2), BeautifulSoup4 + lxml (HTML/XML parsing)
- **Sampler**: nutpie (preferred on Apple Silicon), falls back to PyMC NUTS

## Key Commands

```bash
# Install dependencies
uv sync

# Run the main v3 model pipeline
python alt_nfp_estimation_v3.py

# Run the nowcast backtest (24-month CES-censoring experiment)
python -c "from alt_nfp.backtest import run_backtest; run_backtest()"

# Run the benchmark revision diagnostic (Phase 1)
python scripts/benchmark_diagnostic.py

# Run the benchmark backtest with as-of censoring (Phase 2)
python -c "from alt_nfp.benchmark_backtest import run_benchmark_backtest; run_benchmark_backtest()"

# Run the QCEW sigma sensitivity analysis
python -c "from alt_nfp.sensitivity import run_sensitivity; run_sensitivity()"

# Run the vintage pipeline CLI
python -m alt_nfp.vintages

# Update BLS publication dates (prints new entries for manual review)
python -m alt_nfp.lookups.update_schedule

# Build publication_calendar.parquet
python scripts/build_publication_calendar.py

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
│   ├── config.py             # ProviderConfig, PROVIDERS list, paths, cyclical indicators
│   ├── panel_adapter.py      # Panel → model data dict, cyclical indicators, as-of censoring
│   ├── model.py              # PyMC model definition
│   ├── sampling.py           # sample_model() — nutpie/PyMC with preset configs
│   ├── diagnostics.py        # Parameter summary, precision budget, divergences
│   ├── checks.py             # Prior/posterior predictive checks, LOO-CV
│   ├── residuals.py          # Standardised residuals by source
│   ├── plots.py              # Growth/seasonal, reconstructed index, BD diagnostics
│   ├── forecast.py           # Forward simulation with structural BD propagation
│   ├── backtest.py           # Nowcast backtest (CES-censoring experiment)
│   ├── benchmark.py          # Benchmark revision extraction & decomposition (Phase 1)
│   ├── benchmark_backtest.py # Benchmark backtest with as-of censoring (Phase 2)
│   ├── benchmark_plots.py    # Benchmark diagnostic visualizations
│   ├── sensitivity.py        # QCEW sigma sensitivity analysis
│   ├── lookups/              # Static reference tables
│   │   ├── industry.py       # NAICS → supersector → domain hierarchy + CES series ID map
│   │   ├── revision_schedules.py  # QCEW & CES vintage schedules + publication calendar
│   │   ├── publication_dates.py   # Hard-coded BLS release dates (CES, QCEW, SAE)
│   │   ├── benchmark_revisions.py # Historical actual BLS benchmark revisions
│   │   ├── update_schedule.py     # CLI to fetch new dates from BLS schedule pages
│   │   └── geography.py      # State/area geography hierarchy
│   ├── ingest/               # Raw data → observation panel
│   │   ├── base.py           # PANEL_SCHEMA, validate_panel
│   │   ├── ces_national.py   # CES national-level ingestion
│   │   ├── ces_state.py      # CES state-level ingestion
│   │   ├── qcew.py           # QCEW ingestion
│   │   ├── payroll.py        # Provider index ingestion
│   │   ├── panel.py          # build_panel, save/load
│   │   ├── aggregate.py      # Data aggregation utilities
│   │   ├── releases.py       # Release management
│   │   ├── tagger.py         # Data tagging (source/vintage metadata)
│   │   ├── vintage_store.py  # Vintage data storage/retrieval
│   │   ├── bls/              # BLS API client layer
│   │   │   ├── _http.py      # HTTP transport for BLS API
│   │   │   ├── _programs.py  # BLS program definitions (CES, QCEW, etc.)
│   │   │   ├── ces_national.py  # CES national series downloads
│   │   │   ├── ces_state.py    # CES state series downloads
│   │   │   └── qcew.py        # QCEW data downloads
│   │   └── release_dates/    # BLS publication schedule tracking
│   │       ├── config.py     # Release date configuration
│   │       ├── parser.py     # Schedule parser
│   │       ├── scraper.py    # Release date scraper
│   │       └── vintage_dates.py  # Vintage date management
│   └── vintages/             # Vintage data pipeline (runnable: python -m alt_nfp.vintages)
│       ├── __main__.py       # CLI entry point
│       ├── _client.py        # Vintage client utilities
│       ├── build_store.py    # Build vintage store from raw data
│       ├── views.py          # real_time_view, final_view
│       ├── evaluation.py     # vintage_diff, noise multiplier builder
│       ├── download/         # Vintage data downloaders
│       │   ├── ces.py        # CES vintage downloads
│       │   └── qcew.py       # QCEW vintage downloads
│       └── processing/       # Vintage data processing
│           ├── ces_national.py   # CES national vintage processing
│           ├── qcew.py           # QCEW vintage processing
│           ├── sae_states.py     # State and Area Employment processing
│           └── combine.py        # Combine vintage files
├── data/                     # All data assets
│   ├── store/                # Canonical Hive-partitioned vintage store
│   │   ├── source=ces/       #   partitioned by source × seasonally_adjusted
│   │   ├── source=qcew/
│   │   └── source=sae/
│   ├── providers/            # Payroll provider data (one dir per provider)
│   │   └── G/g_provider.parquet    # Same schema as vintage_store minus
│   │                               #   revision, benchmark_revision, vintage_date
│   ├── downloads/            # Raw inputs fetched from external sources
│   │   ├── ces/cesvinall/    # CES triangular revision CSVs
│   │   ├── qcew/             # QCEW bulk + revisions CSV
│   │   └── releases/         # Scraped BLS schedule HTML
│   ├── intermediate/         # Pipeline byproducts (derivable from downloads)
│   │   ├── ces_revisions.parquet
│   │   ├── qcew_revisions.parquet
│   │   ├── revisions.parquet # Combined CES+QCEW
│   │   ├── release_dates.parquet
│   │   └── vintage_dates.parquet
│   ├── reference/            # Static BLS crosswalks (CES/SAE industry + geography)
│   │   ├── industry_codes.csv
│   │   └── geographic_codes.csv
│   └── publication_calendar.parquet
├── tests/                    # Test suite
│   ├── test_lookups.py       # Industry hierarchy & revision schedule tests
│   ├── test_ingest.py        # Panel validation & schema tests
│   ├── test_new_ingest.py    # Tests for new ingest modules
│   ├── test_release_dates.py # Release date parsing/scraping tests
│   ├── test_vintage_store.py # Vintage store tests
│   ├── test_vintages.py      # Vintage view & evaluation tests
│   ├── test_publication_calendar.py  # Publication calendar build tests
│   ├── test_publication_dates.py     # Publication date lookup tests
│   ├── test_benchmark_backtest.py    # Benchmark backtest infrastructure tests
│   ├── test_backtesting_smoke.py     # Integration smoke tests with real panel data
│   ├── test_cyclical_indicators.py   # Cyclical indicator loading, centering, censoring
│   ├── test_precision_budget.py      # Precision budget DataFrame structure tests
│   ├── test_sensitivity_smoke.py     # Sensitivity analysis smoke tests
│   └── ingest/bls/           # BLS API client tests
│       ├── test_downloads.py
│       ├── test_http.py
│       └── test_programs.py
├── scripts/                  # One-off build/maintenance scripts
│   ├── build_publication_calendar.py  # Build publication_calendar.parquet
│   └── benchmark_diagnostic.py       # Benchmark revision diagnostic runner
├── specs/                    # Design specifications
│   └── vintage_pipeline_spec.md
├── archive/                  # Earlier monolithic scripts (v1, v2), old todos
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
- **Structural birth/death model**: `bd_t = φ₀ + φ₁·X^birth + φ₂·BD^QCEW_{t-L} + φ₃·X^cycle + σ_bd·ξ_t` where `X^cycle = [claims, nfci, biz_apps, jolts]` (centered cyclical indicators).
- **Cyclical indicators** (`config.CYCLICAL_INDICATORS`): claims (weekly), NFCI (weekly), business applications (monthly), JOLTS openings (monthly). Each has a publication lag defined in `panel_adapter._CYCLICAL_PUBLICATION_LAGS` used for as-of censoring.
- **Provider-specific error structures**: each provider can have `iid` or `ar1` measurement error.
- `alt_nfp_estimation_v3.py` is a thin runner: `build_panel()` → `panel_to_model_data()` → `build_model()` → prior checks → sampling → diagnostics → PPC → LOO → plots → forecast → save.
- **Data pipeline**: `build_panel()` (`ingest/panel.py`) reads the vintage store (`data/store/`) + provider parquet files (`data/providers/`) → `panel_to_model_data()` (`panel_adapter.py`) converts the panel to model arrays (growth rates, BD covariates, cyclical indicators). Provider files share the vintage store schema but omit `revision`, `benchmark_revision`, and `vintage_date` (no vintages for provider data; `ref_date` determines currency).
- PyMC models are built declaratively; sampling uses nutpie when available.
- Output artifacts (NetCDF inference data, PNG plots) go to `output/`.
- **BLS API client** (`ingest/bls/`): structured HTTP layer for downloading CES and QCEW data directly from BLS. `_programs.py` defines program metadata; `_http.py` handles request transport.
- **Vintage pipeline** (`vintages/`): download → process → store workflow for managing real-time data vintages. Runnable as `python -m alt_nfp.vintages`. The canonical output is `data/store/`, a Hive-partitioned parquet dataset holding all current and prior vintages of CES, QCEW, and SAE data, partitioned by `(source, seasonally_adjusted)`. Reader/writer utilities live in `ingest/vintage_store.py`.
- **Release date tracking** (`ingest/release_dates/`): scrapes and parses BLS publication schedules to determine data availability windows for real-time vintage construction.
- **Publication date lookups** (`lookups/publication_dates.py`): hard-coded BLS release dates for CES, QCEW, and SAE programs, scraped from BLS schedule pages. `update_schedule.py` fetches new dates and prints copy-paste-ready dict entries. `scripts/build_publication_calendar.py` merges historical + hard-coded dates into `publication_calendar.parquet`.
- **Benchmark revision inference** (`benchmark.py`): extracts March-level employment changes from the posterior, compares to actual BLS benchmark revisions (`lookups/benchmark_revisions.py`), decomposes into continuing-units divergence + BD accumulation.
- **Benchmark backtest** (`benchmark_backtest.py`): tests nowcasting accuracy at multiple horizons (T-12, T-9, T-6, T-3, T-1 months before March report). Uses `as_of` parameter in `panel_to_model_data()` to censor observations by vintage date, simulating real-time information sets. Computes RMSE, 90% coverage, and comparative baselines (naive zero, prior-year).
- **Precision budget** (`diagnostics.compute_precision_budget()`): quantifies information contribution by source as `share_i = precision_i / Σ(precision)`, accounting for CES vintage-specific sigmas, QCEW M3/M12 distinctions, and provider signal loadings with AR1 autocorrelation. Outputs to `output/precision_budget.parquet`.
- **`@pytest.mark.network`**: tests requiring network access are marked; deselect with `-m "not network"`.

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
