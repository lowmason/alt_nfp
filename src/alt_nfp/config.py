"""Provider configuration, project paths, and model constants.

This module is the single source of truth for:

* **File-system paths** — ``BASE_DIR``, ``DATA_DIR``, ``STORE_DIR``,
  ``DOWNLOADS_DIR``, ``INTERMEDIATE_DIR``, ``INDICATORS_DIR``, ``OUTPUT_DIR``.
* **Model hyper-parameters** — QCEW noise floors, BD lag, Fourier harmonic
  count, cyclical indicator specs.
* **Provider registry** — the :data:`PROVIDERS` list of
  :class:`ProviderConfig` entries that drive the entire pipeline.

Adding a new payroll provider requires only a new ``ProviderConfig`` entry
in :data:`PROVIDERS`.  The model, diagnostics, and plots adapt automatically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
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
INDICATORS_DIR = DATA_DIR / "indicators"
OUTPUT_DIR = BASE_DIR / "output"

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

# QCEW observation noise: LogNormal priors for estimated base sigmas.
# LogNormal avoids the funnel geometry that HalfNormal creates when sigma
# collapses toward zero (QCEW precision overwhelms all other sources).
# mu/sigma are the parameters of the underlying Normal on log(sigma_qcew).
LOG_SIGMA_QCEW_MID_MU = math.log(0.0005)  # M2 months — 90% prior ≈ [0.0004, 0.0006]
LOG_SIGMA_QCEW_MID_SD = 0.15  # tightened from 0.25 to keep posterior near 0.05%/mo

# QCEW likelihood degrees of freedom (Student-t).
# Fixed nu provides robustness to non-Gaussian QCEW error (NAICS
# reclassification, timing mismatches) and reduces per-observation
# precision by a factor of (nu+1)/(nu+3) vs Normal.
QCEW_NU: int = 5
LOG_SIGMA_QCEW_BOUNDARY_MU = math.log(0.002)  # M3+M1 months — 90% prior ≈ [0.0009, 0.0046]
LOG_SIGMA_QCEW_BOUNDARY_SD = 0.5

# CES observation noise: LogNormal priors (same rationale as QCEW —
# prevents zero-boundary funnel and keeps CES precision bounded).
LOG_SIGMA_CES_MU = math.log(0.002)  # mode ~0.0016, median 0.002
LOG_SIGMA_CES_SD = 0.5  # 90% prior ≈ [0.0009, 0.0044]

# Fourier seasonal innovation: LogNormal priors (same rationale as QCEW sigmas —
# HalfNormal allows sigma_fourier to collapse to zero, creating a boundary pathology).
# mu is for k=1; higher harmonics subtract log(k) in model.py.
LOG_SIGMA_FOURIER_MU = math.log(0.0003)  # k=1 mode ~0.00022
LOG_SIGMA_FOURIER_SD = 0.5  # ~90% prior [0.00011, 0.00082] for k=1

# Latent AR(1) marginal SD (tau).  Reparameterising in terms of the
# stationary SD tau rather than the innovation SD sigma_g breaks the
# phi-sigma ridge that causes poor ESS when phi is near 1.
# sigma_g = tau * sqrt(1 - phi^2) is derived in the model.
# Calibration: posterior sigma_g ~ 0.008, phi ~ 0.8 → tau ~ 0.013.
LOG_TAU_MU = math.log(0.013)  # median ≈ 0.013
LOG_TAU_SD = 0.5  # 90% prior ≈ [0.006, 0.029]

# BD innovation sigma: LogNormal avoids the zero-boundary funnel that
# HalfNormal creates for this small-scale parameter.
LOG_SIGMA_BD_MU = math.log(0.003)  # median ≈ 0.003
LOG_SIGMA_BD_SD = 0.5  # 90% prior ≈ [0.0013, 0.0067]

# Post-COVID era multiplier for QCEW boundary-month (M1+M3) noise.
# Empirical calibration: revision RMSE ratio (2022+ / 2017-2019) pooled
# across Q1-Q4 for boundary months only (M2 is era-invariant, ratio ~1.0).
# Applied multiplicatively on top of the per-observation revision multiplier.
QCEW_POST_COVID_BOUNDARY_ERA_MULT: dict[int, float] = {
    0: 5.0,
    1: 3.5,
    2: 2.0,
}
QCEW_POST_COVID_BOUNDARY_ERA_DEFAULT: float = 1.0

# QCEW publication lag in months (for BD proxy computation)
BD_QCEW_LAG = 6

# Fourier seasonal expansion: number of harmonics (K)
N_HARMONICS = 4

# Era-specific latent state parameters.  Breakpoints partition the sample
# into macro-structurally distinct regimes so mu_g and phi can vary.
N_ERAS = 2
ERA_BREAKS: list[date] = [date(2020, 1, 1)]
# Era 0: Pre-COVID  (2012-01 → 2019-12)
# Era 1: Post-COVID (2020-01 → present)

# Cyclical indicators for structural BD model (demand-side covariates).
# Each indicator is downloaded from FRED into data/indicators/<name>.parquet
# with a uniform (ref_date, value) schema.  Weekly series are aggregated
# to monthly before centering.
CYCLICAL_INDICATORS: list[dict] = [
    {'name': 'claims', 'fred_id': 'ICNSA', 'freq': 'weekly'},
    # JOLTS job openings.  Published ~2 months after reference period.
    # Revisions are small relative to cross-sectional variation (see JOLTS
    # revision assessment in national_model_spec.md §2.3.1): mean absolute
    # revision for 2003–2023 is ~1% of the level, well within the centered
    # covariate range.  Final values with publication-lag censoring are
    # sufficient; vintage tracking is not needed.  Openings chosen over
    # hires (JTSHIL) — openings lead the hiring cycle and show stronger
    # covariance with the BD component in out-of-sample tests.
    {'name': 'jolts', 'fred_id': 'JTSJOL', 'freq': 'monthly'},
]

# Minimum pseudo-establishments per cell for QCEW-weighted compositing.
# Cells with fewer pseudo-establishments are excluded; their weight is
# redistributed to covered cells per provider_spec.md §5.2.
MIN_PSEUDO_ESTABS_PER_CELL: int = 5

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

    **National providers** (default) supply a pre-aggregated national time
    series filtered by the ``geography_type`` / ``industry_type`` fields.
    Birth rates may live in the same file (``birth_rate`` column) or in a
    separate parquet pointed to by ``birth_file``.

    **Cell-level providers** supply a supersector × Census-region parquet
    (schema: ``geographic_type``, ``geographic_code``, ``industry_type``,
    ``industry_code``, ``ref_date``, ``n_pseudo_estabs``, ``employment``).
    When the loader detects ``geographic_type='region'`` in the file, it
    performs QCEW-weighted compositing automatically (see
    ``ingest/compositing.py`` and ``provider_spec.md`` §5).  Since cell-level
    files have no birth-rate column, use ``birth_file`` to supply a separate
    national-level birth-rate parquet.

    Parameters
    ----------
    name : str
        Display name (e.g. ``'G'``).
    file : str
        Path relative to ``DATA_DIR`` (e.g. ``'providers/G/g_provider.parquet'``).
    error_model : ``'iid'`` | ``'ar1'``
        Measurement-error structure.
    birth_file : str | None
        Optional path (relative to ``DATA_DIR``) to a separate parquet
        containing birth-rate data.  Expected schema includes at least
        ``ref_date`` and ``birth_rate`` columns.  When *None*, the loader
        looks for ``birth_rate`` in the main ``file`` instead.
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
    birth_file: str | None = None
    industry_type: str = "national"
    industry_code: str = "00"
    geography_type: str = "national"
    geography_code: str = "00"


# ---------------------------------------------------------------------------
# Active provider list — edit here to add/remove providers
# ---------------------------------------------------------------------------

# Cell-level providers supply supersector × region employment; the loader
# detects this and applies QCEW-weighted compositing (ingest/compositing.py).
# National providers are used directly.

PROVIDERS: list[ProviderConfig] = [
    ProviderConfig(
        name="G",
        file="providers/g/g_provider.parquet",
        error_model="iid",
        birth_file="providers/g/g_births.parquet",
    ),
]
