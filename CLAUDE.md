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

# Run the nowcast backtest (real-time vintage-aware, 24 months)
python -c "from alt_nfp.backtest import run_backtest; run_backtest()"

# Run backtest with custom start date and output directory
python -c "from datetime import date; from pathlib import Path; from alt_nfp.backtest import run_backtest; run_backtest(n_backtest=12, start_date=date(2024,6,12), output_dir=Path('output/bt_2024h2'))"

# Run the benchmark revision diagnostic (Phase 1)
python scripts/benchmark_diagnostic.py

# Run the benchmark backtest with as-of censoring (Phase 2)
python -c "from alt_nfp.benchmark_backtest import run_benchmark_backtest; run_benchmark_backtest()"

# Run the QCEW sigma sensitivity analysis
python -c "from alt_nfp.sensitivity import run_sensitivity; run_sensitivity()"

# Run the vintage pipeline CLI (or: python -m alt_nfp.vintages)
uv run alt-nfp                     # Run all steps
uv run alt-nfp download            # Download CES triangular + QCEW bulk
uv run alt-nfp download-indicators # Download FRED cyclical indicators
uv run alt-nfp process             # Scrape BLS calendar + process revisions
uv run alt-nfp current             # Fetch current BLS estimates (benchmark-revised)
uv run alt-nfp build               # Merge revisions + current → data/store/
uv run alt-nfp build --releases PATH

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
│   ├── config.py             # ProviderConfig (incl. birth_file), PROVIDERS list, paths, cyclical indicators, era breaks, MIN_PSEUDO_ESTABS_PER_CELL
│   ├── panel_adapter.py      # Panel → model data dict, cyclical indicators, as-of censoring
│   ├── model.py              # PyMC model definition
│   ├── sampling.py           # sample_model() — nutpie/PyMC with preset configs
│   ├── diagnostics.py        # Parameter summary, precision budget, divergences, weight staleness
│   ├── checks.py             # Prior/posterior predictive checks, LOO-CV, era summary
│   ├── residuals.py          # Standardised residuals by source
│   ├── plots.py              # Growth/seasonal, reconstructed index, BD diagnostics
│   ├── forecast.py           # Forward simulation with structural BD propagation
│   ├── backtest.py           # Nowcast backtest (real-time vintage-aware)
│   ├── benchmark.py          # Benchmark revision extraction & decomposition (Phase 1)
│   ├── benchmark_backtest.py # Benchmark backtest with as-of censoring (Phase 2)
│   ├── benchmark_plots.py    # Benchmark diagnostic visualizations
│   ├── sensitivity.py        # QCEW sigma sensitivity analysis
│   ├── lookups/              # Static reference tables
│   │   ├── industry.py       # NAICS → supersector → domain hierarchy + CES series ID map
│   │   │                     #   + NAICS3_TO_MFG_SECTOR, SINGLE_SECTOR_SUPERSECTORS,
│   │   │                     #   GOVT_OWNERSHIP_TO_SECTOR
│   │   ├── revision_schedules.py  # QCEW & CES vintage schedules + publication calendar
│   │   ├── benchmark_revisions.py # Historical actual BLS benchmark revisions
│   │   └── geography.py      # State/area geography hierarchy
│   ├── ingest/               # Raw data → observation panel
│   │   ├── base.py           # PANEL_SCHEMA, validate_panel
│   │   ├── fred.py           # FRED API client (fetch_fred_series)
│   │   ├── indicators.py     # Cyclical indicator store (download + read)
│   │   ├── ces_national.py   # CES national-level ingestion
│   │   ├── ces_state.py      # CES state-level ingestion
│   │   ├── qcew.py           # QCEW ingestion
│   │   ├── compositing.py    # QCEW-weighted national compositing for cell-level providers
│   │   ├── payroll.py        # Provider index ingestion (auto-detects cell-level → compositing)
│   │   ├── panel.py          # build_panel, save/load
│   │   ├── aggregate.py      # Data aggregation utilities
│   │   ├── releases.py       # Release management
│   │   ├── tagger.py         # Data tagging (source/vintage metadata)
│   │   ├── vintage_store.py  # Vintage data storage/retrieval + rank-based horizon censoring
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
│   └── vintages/             # Vintage data pipeline (CLI: alt-nfp or python -m alt_nfp.vintages)
│       ├── __main__.py       # CLI entry point
│       ├── _client.py        # Vintage client utilities
│       ├── build_store.py    # Build vintage store from raw data
│       ├── views.py          # real_time_view, final_view
│       ├── evaluation.py     # vintage_diff, noise multiplier builder
│       ├── download/         # Vintage data downloaders
│       │   ├── ces.py        # CES cesvinall.zip download + extraction
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
│   ├── indicators/           # Cyclical indicator parquets (FRED-sourced)
│   │   ├── claims.parquet    #   (ref_date, value) — ICNSA weekly claims
│   │   ├── nfci.parquet      #   (ref_date, value) — Chicago Fed NFCI
│   │   ├── biz_apps.parquet  #   (ref_date, value) — Census BFS applications
│   │   └── jolts.parquet     #   (ref_date, value) — JOLTS job openings
│   ├── providers/            # Payroll provider data (one dir per provider)
│   │   └── g/
│   │       ├── g_provider.parquet  # Cell-level (region × supersector) employment
│   │       └── g_births.parquet    # National-level birth rates (separate file)
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
├── tests/                    # Test suite
│   ├── test_lookups.py       # Industry hierarchy & revision schedule tests
│   ├── test_ingest.py        # Panel validation & schema tests
│   ├── test_new_ingest.py    # Tests for new ingest modules
│   ├── test_release_dates.py # Release date parsing/scraping tests
│   ├── test_vintage_store.py # Vintage store tests + rank-based censoring helpers + validation guards
│   ├── test_vintages.py      # Vintage view & evaluation tests
│   ├── test_benchmark_backtest.py    # Benchmark backtest infrastructure tests
│   ├── test_backtesting_smoke.py     # Integration smoke tests with real panel data
│   ├── test_cyclical_indicators.py   # Cyclical indicator loading, centering, censoring
│   ├── test_fred.py                  # FRED client, indicator store download/read tests
│   ├── test_precision_budget.py      # Precision budget DataFrame structure tests
│   ├── test_sensitivity_smoke.py     # Sensitivity analysis smoke tests
│   ├── test_compositing.py          # QCEW-weighted compositing: weights, redistribution, composite, integration
│   ├── test_model.py                # Model construction tests (era-specific + scalar fallback)
│   ├── test_store_coverage.py       # Store data-integrity tests + CES censored diagonal invariant
│   └── ingest/bls/           # BLS API client tests
│       ├── test_downloads.py
│       ├── test_http.py
│       └── test_programs.py
├── scripts/                  # One-off build/maintenance scripts
│   └── benchmark_diagnostic.py       # Benchmark revision diagnostic runner
├── specs/                    # Design specifications
│   ├── vintage_pipeline_spec.md
│   └── provider_spec.md      # Provider representativeness correction spec (v2)
├── archive/                  # Earlier monolithic scripts (v1, v2), old todos
├── output/                   # Generated results (InferenceData, plots)
│   └── backtest_runs/        # Per-run backtest artifacts (timestamped dirs with .nc + .parquet)
├── pyproject.toml            # Project config, dependencies, tool settings
└── uv.lock                   # Dependency lock file
```

## Code Style

- **Formatter**: black (line length 100, targets py310-py312)
- **Linter**: ruff (line length 100, rules: E, W, F, I, B, C4, UP)
- **Type checking**: mypy (ignore_missing_imports=true)
- Line length limit: 100 characters

## Key Code Patterns

- **Config-driven providers**: adding a new payroll provider requires only a new `ProviderConfig` entry in `config.py`; data loading, model likelihood, diagnostics, plots, and forecasts adapt automatically. Providers can be **national** (pre-aggregated) or **cell-level** (supersector × Census region); the loader auto-detects cell-level data and routes through QCEW-weighted compositing. `ProviderConfig.birth_file` allows birth-rate data at a different geographic/industry level from the main employment file (e.g. national birth rates alongside cell-level employment). Currently one active provider: `G` (cell-level with separate national birth rates).
- **Era-specific mean growth, shared persistence** (`config.N_ERAS`, `config.ERA_BREAKS`): the AR(1) latent growth process uses era-indexed `mu_g_era` (Pre-COVID / Post-COVID) when `era_idx` is present in the data dict (always, since `panel_to_model_data()` computes it). Persistence `phi_raw ~ Beta(18, 2)` and marginal SD `tau` are **shared across eras** — the eras differ in mean growth, not autocorrelation dynamics. `phi` is capped at 0.99 via `pt.minimum`. Gated in `build_model()` so removing `era_idx` from the data dict recovers the original scalar `mu_g` baseline. The `pytensor.scan` passes per-timestep `mu_g[t]` as a sequence; `phi` and `sigma_g` are non-sequences. `print_era_summary()` in `checks.py` reports per-era `mu_g` and shared phi. `forecast.py` uses the last era (Post-COVID) `mu_g` for forward simulation. Sample starts from 2012 (not 2003) to reduce scan length; the Pre-GFC era was dropped as it contributes nothing to the forecast.
- **Structural birth/death model**: `bd_t = φ₀ + φ₁·X^birth + φ₃·X^cycle + σ_bd·ξ_t` where `X^cycle = [claims, jolts]` (centered cyclical indicators). `φ₁` is gated out of the model graph when its covariate is all-zero (e.g. in backtest iterations where BD data is unavailable), avoiding unidentified parameters and divergences. Same pattern as `φ₃` / cyclical indicators. NFCI, biz_apps, and the QCEW BD lag (φ₂) were removed as their posteriors were indistinguishable from zero.
- **Cyclical indicators** (`config.CYCLICAL_INDICATORS`): claims (weekly, FRED `ICNSA`), JOLTS openings (monthly, FRED `JTSJOL`). Downloaded from FRED via `ingest/fred.py` into `data/indicators/<name>.parquet` (uniform `ref_date, value` schema). `_load_cyclical_indicators()` in `panel_adapter.py` reads parquet, aggregates weekly→monthly, joins to the model calendar via month-truncated keys (panel dates use day=12 BLS convention, indicators use day=1), and centers. Each has a publication lag in `panel_adapter._CYCLICAL_PUBLICATION_LAGS` used for as-of censoring. The model derives `cyclical_keys` dynamically from `CYCLICAL_INDICATORS` so new indicators are automatically picked up.
- **QCEW-weighted provider compositing** (`ingest/compositing.py`): cell-level provider parquets (schema: `geographic_type, geographic_code, industry_type, industry_code, ref_date, n_pseudo_estabs, employment`) are composited into a single national growth series. The canonical cell grid is 4 Census regions × 11 BLS supersectors (10, 20, 30, 40, 50, 55, 60, 65, 70, 80, 90) = 44 cells. `load_qcew_weights()` reads region × supersector employment from the vintage store (latest vintage per cell), computes shares summing to 1.0, and carries forward the last available QCEW weights for months beyond the QCEW publication frontier (tracked via `weight_staleness_months`). `redistribute_weights()` reallocates weight from uncovered cells: same supersector first, then same region, then uniform. `compute_provider_composite()` computes cell-level log-difference growth, applies `MIN_PSEUDO_ESTABS_PER_CELL` filtering, redistributes weights, and returns a synthetic national employment level (base=100) that feeds into the existing provider pipeline. `payroll.py` detects cell-level data via `_is_cell_level()` (checks `geographic_type='region'`) in both `load_provider_series()` and `ingest_provider()`. `diagnostics.print_weight_staleness()` reports per-provider staleness summary.
- **Provider-specific error structures**: each provider can have `iid` or `ar1` measurement error.
- `alt_nfp_estimation_v3.py` is a thin runner: `build_panel()` → `panel_to_model_data()` → `build_model()` → prior checks → sampling → diagnostics → PPC → LOO → plots → forecast → save.
- **Data pipeline**: `build_panel()` (`ingest/panel.py`) reads the vintage store (`data/store/`) + provider parquet files (`data/providers/`) → `panel_to_model_data()` (`panel_adapter.py`) converts the panel to model arrays (growth rates, BD covariates, cyclical indicators). Provider files share the vintage store schema but omit `revision`, `benchmark_revision`, and `vintage_date` (no vintages for provider data; `ref_date` determines currency). Cell-level provider parquets have a different schema (`geographic_type, geographic_code, industry_type, industry_code, ref_date, n_pseudo_estabs, employment`); the loader detects these and composites via `ingest/compositing.py` before entering the standard pipeline. `build_panel(as_of_ref=D)` activates rank-based horizon censoring for CES/QCEW (see below).
- **Rank-based horizon censoring** (`vintage_store.py`): `transform_to_panel(lf, as_of_ref=D)` applies two-layer censoring: (1) `vintage_date <= D` + `ref_date < D` filtering prevents lookahead; (2) rank-based selection picks the correct revision per recency rank. CES uses `_select_ces_at_horizon`: rank 1→rev-0 (1st print), rank 2→rev-1 (2nd print), rank 3+→rev-2 with `benchmark_revision=0` (actual 3rd print only — benchmark-revised rows are never selected). Benchmark-quality information enters the model exclusively through QCEW observations. QCEW uses `_select_qcew_at_horizon`: quarter-dependent revision rules matching BLS publication schedule (`_QCEW_MAX_REVISION = {Q1: 4, Q2: 3, Q3: 2, Q4: 1}`). Both helpers have fallback logic when prescribed revisions don't exist at the data frontier. `_validate_censored_selection` runs fail-fast checks (no duplicate ref_dates, consecutive months, no null/zero employment/growth, row count sanity) before data reaches the sampler. Growth is computed *before* rank selection to preserve per-vintage measurement error semantics. In `panel_adapter.py`, `_ces_best_available()` selects the highest revision in `[0, 1, 2]` per month; benchmark-revised rows (`revision_number=-1` or `benchmark_revision>0`) are never used as CES observations.
- PyMC models are built declaratively; sampling uses nutpie when available.
- Output artifacts (NetCDF inference data, PNG plots) go to `output/`.
- **FRED API client** (`ingest/fred.py`): `fetch_fred_series(series_id)` downloads a single FRED time series via the JSON API (`api.stlouisfed.org`). Uses `httpx` with exponential-backoff retry. Requires `FRED_API_KEY` env var.
- **Indicator store** (`ingest/indicators.py`): `download_indicators()` fetches all `CYCLICAL_INDICATORS` from FRED and writes to `data/indicators/`. `read_indicator(name)` loads a single parquet. CLI: `alt-nfp download-indicators`.
- **BLS API client** (`ingest/bls/`): structured HTTP layer for downloading CES and QCEW data directly from BLS. `_programs.py` defines program metadata; `_http.py` handles request transport.
- **Vintage pipeline** (`vintages/`): download → process → current → build workflow for managing real-time data vintages. CLI: `alt-nfp` (or `python -m alt_nfp.vintages`). Steps: `download` fetches `cesvinall.zip` (CES triangular revision CSVs) + QCEW bulk files; `process` scrapes the BLS release calendar (internally) then parses revisions into `revisions.parquet`; `current` fetches the latest BLS estimates (benchmark-revised) into `releases.parquet`; `build` merges both into `data/store/`, a Hive-partitioned parquet dataset partitioned by `(source, seasonally_adjusted)`. Reader/writer utilities live in `ingest/vintage_store.py`.
- **QCEW bulk download** (`vintages/download/qcew.py`): downloads yearly singlefile ZIPs from BLS, filters to `own_code ∈ {0,1,2,3,5}` (total, federal/state/local government, private) and `agglvl_code ∈ {10,11,14,15,50,51,54,55}` (national/state totals, by ownership, by 2-digit NAICS sector, by 3-digit NAICS subsector). Saves as `qcew_bulk.parquet`.
- **QCEW processing** (`vintages/processing/qcew.py`): four input streams from bulk data: (1) total all-ownership → `(national, '00')`; (2) private 2-digit NAICS sectors (excluding manufacturing); (3) government by ownership (`own_code` 1/2/3 → sectors 91/92/93 via `GOVT_OWNERSHIP_TO_SECTOR`); (4) manufacturing 3-digit NAICS subsectors → durable (sector 31) / nondurable (sector 32) via `NAICS3_TO_MFG_SECTOR`. Sectors aggregate → supersectors (including 90=Government) → domains (05, 06, 07, 08). Single-sector supersectors (20/50/80) naturally produce sector rows (23/51/81) from the NAICS mapping. QCEW national has 38 industry combos; CES national has 35 (difference: CES uses its own sector codes 41/42/43 for Wholesale/Retail/Transport vs NAICS 42/44/48, and QCEW has 3 extra NAICS sectors 23/51/81).
- **Release date tracking** (`ingest/release_dates/`): scrapes and parses BLS publication schedules to assign `vintage_date` to each CES/QCEW revision. Built automatically by the `process` step; intermediate outputs are `release_dates.parquet` and `vintage_dates.parquet`.
- **Benchmark revision inference** (`benchmark.py`): extracts March-level employment changes from the posterior, compares to actual BLS benchmark revisions (`lookups/benchmark_revisions.py`), decomposes into continuing-units divergence + BD accumulation.
- **Nowcast backtest** (`backtest.py`): real-time vintage-aware backtest with configurable window. `run_backtest(n_backtest=24, start_date=None, output_dir=None)` — when `start_date` is provided, runs forward from that date for `n_backtest` months; otherwise covers the last `n_backtest` months of the panel. For each target month T, builds a censored panel via `build_panel(as_of_ref=T)` (rank-based CES/QCEW selection), then applies `panel_to_model_data(panel, PROVIDERS, as_of=T)` for provider/cyclical indicator censoring. CES gets the revision that existed at each point (T-1 = rev 0, T-2 = rev 1, T-3 = rev 2), QCEW is naturally missing for the most recent ~5-6 months, cyclical indicators are censored by their publication lags. All per-iteration InferenceData is saved as `{run_dir}/{YYYY-MM}.nc`; summary metrics are saved as `{run_dir}/backtest_results.parquet`. Default `run_dir` is a timestamped subdirectory under `output/backtest_runs/`; override via `output_dir`. Returns a `pl.DataFrame` (not `list[dict]`). Plot shows side-by-side jobs-added bars (actual vs nowcast) and a data-source availability strip (CES/QCEW/provider per month). Requires the vintage store (`data/store/`) to have triangular revision history; warns when the CES frontier is stale.
- **Benchmark backtest** (`benchmark_backtest.py`): tests benchmark revision prediction at multiple horizons (T-12, T-9, T-6, T-3, T-1 months before March report). Uses `as_of` parameter in `panel_to_model_data()` to censor observations by vintage date, simulating real-time information sets. Computes RMSE, 90% coverage, and comparative baselines (naive zero, prior-year).
- **CES best-available print**: One observation per month per SA/NSA using the highest available revision (rev 2 > 1 > 0). Eliminates the overcounting from correlated vintages (ρ > 0.99). Data dict provides `ces_sa_vintage_idx`/`ces_nsa_vintage_idx` (values 0/1/2) alongside `g_ces_sa`/`g_ces_nsa`. Model uses `sigma_ces_sa[vintage_idx]` for per-obs noise. `_ces_best_available()` in `panel_adapter.py` selects the best print; `build_obs_sources()` returns `obs_ces_sa` and `obs_ces_nsa` (2 entries, not 6). The old `g_ces_sa_by_vintage`/`g_ces_nsa_by_vintage`/`has_ces_vintage` keys are removed.
- **QCEW observation noise** (post-overhaul): The QCEW likelihood uses a **Student-t** distribution (`pm.StudentT`) with fixed degrees of freedom `QCEW_NU` (default 5) in `config.py`, providing robustness to non-Gaussian QCEW error (NAICS reclassification, timing mismatches) and reducing per-observation precision by a factor of `(nu+1)/(nu+3)` vs Normal (0.75 at nu=5). Per-observation sigma is `sigma_qcew[i] = base_sigma_tier[i] * revision_multiplier[i] * era_multiplier[i]`. Two **estimated** LogNormal base sigmas: `sigma_qcew_mid` (M2 months: Feb, May, Aug, Nov) and `sigma_qcew_boundary` (M3 + M1, including January). LogNormal priors (instead of HalfNormal) prevent the funnel geometry that arises when sigma collapses toward zero and QCEW precision overwhelms all other sources. The M2 prior is tightened (`LOG_SIGMA_QCEW_MID_SD = 0.15`) to keep the posterior near the 0.05%/mo center and prevent extreme precision dominance. Revision multipliers come from `lookups/revision_schedules.py` via `get_noise_multiplier("qcew_Q1".."qcew_Q4", rev)`. Post-COVID boundary months (M1+M3) receive additional era multipliers from `QCEW_POST_COVID_BOUNDARY_ERA_MULT` in `config.py` ({0: 5.0, 1: 3.5, 2: 2.0}); M2 months are era-invariant. Panel adapter exposes `qcew_is_m2` and `qcew_noise_mult`. Priors: `LOG_SIGMA_QCEW_MID_MU`/`_SD`, `LOG_SIGMA_QCEW_BOUNDARY_MU`/`_SD` in `config.py`. `sigma_fourier` (Fourier seasonal GRW innovation) also uses LogNormal priors (`LOG_SIGMA_FOURIER_MU`/`_SD`) with 1/k scaling in log-space across harmonics. Empirical calibration of multipliers used 2017+ data only (excluding 2020–2021). M1/M2 QCEW are retrospective (not imputed); M3 can have payroll-timing quirks (e.g. monthly payers reporting 0 for M3).
- **Precision budget** (`diagnostics.compute_precision_budget()`): quantifies information contribution by source as `share_i = precision_i / Σ(precision)`, using CES per-obs vintage-indexed sigmas via `_ces_precision_rows()` (Normal Fisher information: `lambda^2/sigma[vidx]^2`), **QCEW (M2)** vs **QCEW (M3+M1)** with per-observation `qcew_noise_mult` (Student-t Fisher information: `(nu+1)/((nu+3)*sigma^2)`), and provider signal loadings (with AR1 where applicable). Outputs to `output/precision_budget.parquet`. `print_windowed_precision_budget()` reports precision shares by era window. `diagnostics.print_weight_staleness()` reports QCEW weight carry-forward for cell-level providers.
- **Provider data window**: Provider (e.g. G) data may start mid-sample (e.g. 2019-01). Birth-rate and provider growth means in `panel_adapter.py` are computed only over the calendar window where that provider has data, to avoid "Mean of empty slice" and to keep priors appropriate for the provider-covered period.
- **Forecast** (`forecast.py`): expects `data["levels"]` to be a Polars DataFrame with columns `ref_date`, `ces_sa_index`, `ces_nsa_index`, `ces_sa_level`, `ces_nsa_level`, and provider `emp_col` series. Levels are built in `panel_adapter.py`; `ces_sa_level`/`ces_nsa_level` come from national panel `employment_level` when present. **Employment level units**: CES is in thousands (BLS convention); QCEW is converted from persons to thousands in `vintages/processing/qcew.py`. The constructed model index (`ces_sa_index`) starts at base=100 at the first valid CES observation (~2012-01), not a BLS base period. Index-to-level conversion preserves the thousands unit; month-over-month level differences are directly in thousands (no further division needed). Backtest result columns (`actual_change_k`, `nowcast_change_k`, etc.) are also in thousands.
- **Era-specific diagnostics**: `print_windowed_precision_budget(idata, data)` reports precision shares per era (using `config.ERA_BREAKS`). `print_provider_value_of_information(idata, data)` compares posterior `g_cont` 80% HDI width at time steps with vs without provider data (within the same era, e.g. Post-COVID) to quantify provider value-of-information.
- **LOO-CV**: PSIS-LOO is used as a **data quality audit**, not for model evaluation — it measures interpolation consistency in a state-space model, not forecast skill. `run_loo_cv()` prints per-source outlier tables (flagged if k-hat > 0.5 or pointwise ELPD > 2 SD below source mean), sorted by worst ELPD, with calendar dates. High k-hat on QCEW is structural (high-precision observations dominate the latent state, making importance weights unstable). ArviZ Pareto warning is silenced; the code reports its own k-hat summary. For model evaluation, use the vintage-aware backtest (`backtest.py`).
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
