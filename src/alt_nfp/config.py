"""Provider configuration, project paths, and model constants.

This module is the single source of truth for:

* **File-system paths** — ``BASE_DIR``, ``DATA_DIR``, ``STORE_DIR``,
  ``DOWNLOADS_DIR``, ``INTERMEDIATE_DIR``, ``OUTPUT_DIR``.
* **Model hyper-parameters** — QCEW noise floors, BD lag, Fourier harmonic
  count, cyclical indicator specs.
* **Provider registry** — the :data:`PROVIDERS` list of
  :class:`ProviderConfig` entries that drive the entire pipeline.

Adding a new payroll provider requires only a new ``ProviderConfig`` entry
in :data:`PROVIDERS`.  The model, diagnostics, and plots adapt automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]  # repo root (alt_nfp/)
DATA_DIR = BASE_DIR / "data"
STORE_DIR = DATA_DIR / "store"
DOWNLOADS_DIR = DATA_DIR / "downloads"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
OUTPUT_DIR = BASE_DIR / "output"

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

# QCEW fixed observation noise (growth-rate space)
SIGMA_QCEW_M3 = 0.0005  # ~0.05%/mo — near-census, current-period tax filing
SIGMA_QCEW_M12 = 0.0015  # ~0.15%/mo — retrospective UI months

# QCEW publication lag in months (for structural BD error-correction term)
BD_QCEW_LAG = 6

# Fourier seasonal expansion: number of harmonics (K)
N_HARMONICS = 4

# Cyclical indicators for structural BD model (demand-side covariates)
CYCLICAL_INDICATORS: list[dict] = [
    {'name': 'claims', 'file': 'claims_weekly.csv', 'col': 'claims', 'freq': 'weekly'},
    {'name': 'nfci', 'file': 'nfci.csv', 'col': 'nfci', 'freq': 'weekly'},
    {'name': 'biz_apps', 'file': 'business_applications.csv', 'col': 'applications',
     'freq': 'monthly'},
]

# Plot colours — one per provider, cycled if >7 providers
PP_COLORS = [
    "#2ca02c",  # green
    "#8c564b",  # brown
    "#9467bd",  # purple
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


# ---------------------------------------------------------------------------
# Provider specification
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    """Specification for a single payroll-provider series.

    Adding a new provider requires only a new entry in ``PROVIDERS``.
    The model, diagnostics, and plots adapt automatically.

    All provider files share a standard Parquet/CSV schema::

        ref_date, geography_type, geography_code,
        industry_type, industry_code, employment, [birth_rate]

    The ``employment`` column is log-differenced to compute growth rates.
    If ``birth_rate`` is present it is used as a BD covariate.

    Parameters
    ----------
    name : str
        Display name (e.g. ``'G'``).
    file : str
        Path relative to ``DATA_DIR`` (e.g. ``'providers/G/g_provider.parquet'``).
    error_model : ``'iid'`` | ``'ar1'``
        Measurement-error structure.
    industry_type : str
        Filter value for the ``industry_type`` column (default ``'national'``).
    industry_code : str
        Filter value for the ``industry_code`` column (default ``'00'``).
    geography_type : str
        Filter value for the ``geography_type`` column (default ``'national'``).
    geography_code : str
        Filter value for the ``geography_code`` column (default ``'00'``).
    """

    name: str
    file: str
    error_model: Literal["iid", "ar1"] = "iid"
    industry_type: str = "national"
    industry_code: str = "00"
    geography_type: str = "national"
    geography_code: str = "00"


# ---------------------------------------------------------------------------
# Active provider list — edit here to add/remove providers
# ---------------------------------------------------------------------------

PROVIDERS: list[ProviderConfig] = [
    ProviderConfig(
        name="G",
        file="providers/G/g_provider.parquet",
        error_model="iid",
    ),
]
