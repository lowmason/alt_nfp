# ---------------------------------------------------------------------------
# alt_nfp.config — Provider configuration and project constants
# ---------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]  # repo root (alt_nfp/)
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

# QCEW fixed observation noise (growth-rate space)
SIGMA_QCEW_M3 = 0.0005  # ~0.05%/mo — near-census, current-period tax filing
SIGMA_QCEW_M12 = 0.0015  # ~0.15%/mo — retrospective UI months

# QCEW publication lag in months (for structural BD error-correction term)
BD_QCEW_LAG = 6

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

    Parameters
    ----------
    name : str
        Display name (e.g. ``'PP1'``).
    file : str
        CSV filename under ``data/``.
    index_col : str
        Column name for the index level in *file*.
    error_model : ``'iid'`` | ``'ar1'``
        Measurement-error structure.
    births_file : str, optional
        CSV with a birth-rate column (for structural BD covariate).
    births_col : str, optional
        Column name for the birth rate inside *births_file*.
    """

    name: str
    file: str
    index_col: str
    error_model: Literal["iid", "ar1"] = "iid"
    births_file: str | None = None
    births_col: str | None = None


# ---------------------------------------------------------------------------
# Active provider list — edit here to add/remove providers
# ---------------------------------------------------------------------------

PROVIDERS: list[ProviderConfig] = [
    ProviderConfig(
        name="PP1",
        file="alt_nfp_index_1.csv",
        index_col="pp_index_1",
        error_model="ar1",  # multi-establishment restructuring → autocorrelated residuals
    ),
    ProviderConfig(
        name="PP2",
        file="alt_nfp_index_2.csv",
        index_col="pp_index_2_0",
        error_model="iid",
        births_file="alt_nfp_births_2.csv",
        births_col="pp2_births",
    ),
    # Future providers — just add entries here.  Example:
    # ProviderConfig(
    #     name="PP3",
    #     file="alt_nfp_index_3.csv",
    #     index_col="pp_index_3",
    #     error_model="iid",
    # ),
]
