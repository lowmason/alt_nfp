# alt_nfp

**Bayesian state-space model for nowcasting U.S. nonfarm payroll employment.**

[Documentation](https://lowmason.github.io/alt_nfp/) · [API Reference](https://lowmason.github.io/alt_nfp/reference/)

------------------------------------------------------------------------

## Overview

`alt_nfp` fuses three families of employment data inside a single hierarchical Bayesian model to produce a real-time estimate of total nonfarm payroll (NFP) employment growth:

| Data Source | Role | Frequency |
|------------------------|------------------------|------------------------|
| **QCEW** (Quarterly Census of Employment and Wages) | Near-census truth anchor | Quarterly |
| **CES** (Current Employment Statistics) | High-frequency official estimate | Monthly |
| **Payroll providers** | Private real-time signals | Monthly |

The model is estimated with [PyMC](https://www.pymc.io/) using the NUTS sampler (via [nutpie](https://github.com/pymc-devs/nutpie) when available).

### Key Features

-   **Config-driven providers** — add a new payroll vendor with a single `ProviderConfig` entry. Model, diagnostics, plots, and forecasts adapt automatically.
-   **Structural birth/death model** — time-varying BD offset driven by birth-rate and lagged QCEW covariates.
-   **Vintage-tracked observation panel** — Hive-partitioned Parquet store manages real-time data revisions.
-   **Full Bayesian workflow** — prior/posterior predictive checks, LOO-CV, residual analysis, sensitivity sweeps, and backtests.

## Installation

Requires Python \>= 3.10 and [uv](https://docs.astral.sh/uv/).

``` bash
git clone https://github.com/lowmason/alt_nfp.git
cd alt_nfp
uv sync
```

## Quick Start

### Run the full estimation pipeline

``` bash
uv run python alt_nfp_estimation_v3.py
```

This executes data loading → model building → prior checks → MCMC sampling → diagnostics → posterior predictive checks → LOO-CV → residuals → plots → forecast → save.

### Run a nowcast backtest

``` python
from alt_nfp import run_backtest

results = run_backtest(n_backtest=24)
```

### Run QCEW sensitivity analysis

``` python
from alt_nfp import run_sensitivity

results = run_sensitivity()
```

### Use the panel API

``` python
from alt_nfp import build_panel, PROVIDERS
from alt_nfp.panel_adapter import panel_to_model_data
from alt_nfp.model import build_model
from alt_nfp.sampling import sample_model

panel = build_panel(use_legacy=True)
data = panel_to_model_data(panel, PROVIDERS)
model = build_model(data)
idata = sample_model(model)
```

## Project Structure

```         
src/alt_nfp/
├── config.py              # Paths, constants, provider registry
├── data.py                # Legacy CSV data loading
├── model.py               # PyMC state-space model
├── sampling.py            # MCMC sampling (nutpie / PyMC)
├── panel_adapter.py       # Observation panel → model dict
├── diagnostics.py         # Convergence, source contributions, divergences
├── checks.py              # Prior/posterior predictive, LOO-CV
├── residuals.py           # Standardised residual plots
├── plots.py               # Growth, seasonal, index, BD plots
├── forecast.py            # Forward simulation
├── backtest.py            # CES-censoring nowcast backtest
├── sensitivity.py         # QCEW sigma sensitivity sweep
├── lookups/               # Static reference tables
├── ingest/                # Raw data → observation panel
│   └── bls/               # BLS API client
└── vintages/              # Revision tracking pipeline
    ├── download/
    └── processing/
```

## Model

The state-space model decomposes total employment growth into:

1.  **Latent continuing-units growth** — AR(1) process with mean reversion.
2.  **Fourier seasonal** — annually-evolving harmonic amplitudes (GRW).
3.  **Structural birth/death** — time-varying offset with covariates.

Observed through QCEW (truth anchor, fixed noise), CES (vintage-specific noise), and payroll providers (per-provider bias, loading, and noise).

## Development

``` bash
# Install dev dependencies
uv sync --group dev

# Format
black . --line-length 100

# Lint
ruff check .

# Type check
mypy .

# Test (skip network-dependent tests)
pytest tests/ -m "not network"
```

## Documentation

Full documentation is hosted at [**lowmason.github.io/alt_nfp**](https://lowmason.github.io/alt_nfp/).

To build locally:

``` bash
uv sync --group docs
uv run mkdocs serve
```

## Tech Stack

| Component          | (Library                      |
|--------------------|-------------------------------|
| Bayesian inference | PyMC, PyTensor, ArviZ         |
| Sampler            | nutpie (preferred), PyMC NUTS |
| Data manipulation  | Polars                        |
| Visualisation      | Matplotlib                    |
| HTTP client        | httpx (HTTP/2)                |
| HTML parsing       | BeautifulSoup4, lxml          |
| Notebooks          | Marimo                        |
| Documentation      | MkDocs Material, mkdocstrings |

## License

MIT