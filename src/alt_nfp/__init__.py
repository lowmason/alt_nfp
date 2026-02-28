"""Bayesian state-space model for nowcasting U.S. nonfarm payroll employment.

``alt_nfp`` fuses three families of data source—BLS Current Employment
Statistics (CES), Quarterly Census of Employment and Wages (QCEW), and
private payroll-provider indices—inside a single PyMC state-space model.
QCEW serves as the near-census truth anchor; CES and payroll providers
contribute higher-frequency but noisier signals.

Key capabilities:

* **Config-driven providers** — add a new payroll vendor by appending a
  :class:`ProviderConfig` to :data:`PROVIDERS`.
* **Structural birth/death model** — time-varying BD offset driven by
  birth-rate and lagged QCEW covariates.
* **Vintage-tracked observation panel** — ``ingest`` and ``vintages``
  sub-packages manage real-time data revisions.
* **Bayesian workflow toolkit** — prior/posterior predictive checks,
  LOO-CV, residual diagnostics, sensitivity sweeps, and backtests.
"""

from .backtest import run_backtest
from .diagnostics import compute_precision_budget
from .config import (
    BASE_DIR,
    DATA_DIR,
    DOWNLOADS_DIR,
    INTERMEDIATE_DIR,
    OUTPUT_DIR,
    PROVIDERS,
    STORE_DIR,
    ProviderConfig,
)
from .model import build_model
from .panel_adapter import build_obs_sources, panel_to_model_data
from .sampling import (
    DEFAULT_SAMPLER_KWARGS,
    LIGHT_SAMPLER_KWARGS,
    MEDIUM_SAMPLER_KWARGS,
    sample_model,
)
from .sensitivity import run_sensitivity
from .benchmark import (
    decompose_benchmark_revision,
    extract_benchmark_revision,
    load_anchor_level_from_vintage_store,
    summarize_revision_posterior,
)
from .benchmark_backtest import (
    DEFAULT_YEARS,
    EXTENDED_YEARS,
    HORIZONS,
    build_comparative_benchmarks,
    compute_backtest_metrics,
    horizon_to_as_of,
    run_benchmark_backtest,
)

from .lookups import INDUSTRY_HIERARCHY, QCEW_REVISIONS, CES_REVISIONS
from .ingest import build_panel, validate_panel, PANEL_SCHEMA
from .vintages import real_time_view, final_view, vintage_diff

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "DOWNLOADS_DIR",
    "INTERMEDIATE_DIR",
    "OUTPUT_DIR",
    "STORE_DIR",
    "PROVIDERS",
    "ProviderConfig",
    "build_obs_sources",
    "build_model",
    "sample_model",
    "panel_to_model_data",
    "DEFAULT_SAMPLER_KWARGS",
    "LIGHT_SAMPLER_KWARGS",
    "MEDIUM_SAMPLER_KWARGS",
    "run_backtest",
    "run_sensitivity",
    "compute_precision_budget",
    # Benchmark prediction (Phase 1)
    "decompose_benchmark_revision",
    "extract_benchmark_revision",
    "load_anchor_level_from_vintage_store",
    "summarize_revision_posterior",
    # Benchmark backtest (Phase 2)
    "DEFAULT_YEARS",
    "EXTENDED_YEARS",
    "HORIZONS",
    "build_comparative_benchmarks",
    "compute_backtest_metrics",
    "horizon_to_as_of",
    "run_benchmark_backtest",
    # Data infrastructure
    "INDUSTRY_HIERARCHY",
    "QCEW_REVISIONS",
    "CES_REVISIONS",
    "build_panel",
    "validate_panel",
    "PANEL_SCHEMA",
    "real_time_view",
    "final_view",
    "vintage_diff",
]
