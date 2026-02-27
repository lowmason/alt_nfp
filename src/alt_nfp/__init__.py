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
from .config import BASE_DIR, DATA_DIR, OUTPUT_DIR, PROVIDERS, ProviderConfig
from .data import build_obs_sources, load_data
from .model import build_model
from .sampling import (
    DEFAULT_SAMPLER_KWARGS,
    LIGHT_SAMPLER_KWARGS,
    MEDIUM_SAMPLER_KWARGS,
    sample_model,
)
from .sensitivity import run_sensitivity
from .panel_adapter import panel_to_model_data

# New data infrastructure (v4)
from .lookups import INDUSTRY_HIERARCHY, QCEW_REVISIONS, CES_REVISIONS
from .ingest import build_panel, validate_panel, PANEL_SCHEMA
from .vintages import real_time_view, final_view, vintage_diff

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "OUTPUT_DIR",
    "PROVIDERS",
    "ProviderConfig",
    "build_obs_sources",
    "load_data",
    "build_model",
    "sample_model",
    "DEFAULT_SAMPLER_KWARGS",
    "LIGHT_SAMPLER_KWARGS",
    "MEDIUM_SAMPLER_KWARGS",
    "run_backtest",
    "run_sensitivity",
    "panel_to_model_data",
    # New data infrastructure
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
