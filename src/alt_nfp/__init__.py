# ---------------------------------------------------------------------------
# alt_nfp — Bayesian state-space model for U.S. employment growth
# ---------------------------------------------------------------------------
"""QCEW-anchored nowcasting model with config-driven providers, provider-
specific error structures, and a structural birth/death model."""

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
]
