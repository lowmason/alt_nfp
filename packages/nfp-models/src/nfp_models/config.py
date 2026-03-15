"""Provider configuration, project paths, and model constants.

**Backwards-compatibility shim.**  All authoritative definitions live in
:mod:`alt_nfp.settings`.  This module instantiates a default
:class:`~alt_nfp.settings.NowcastConfig` and re-exports the legacy
module-level constants so that existing ``from .config import …`` imports
continue to work during migration.
"""

from __future__ import annotations

from nfp_lookups.paths import (
    BASE_DIR,
    DATA_DIR,
    DOWNLOADS_DIR,
    INDICATORS_DIR,
    INTERMEDIATE_DIR,
    OUTPUT_DIR,
    STORE_DIR,
)
from nfp_lookups.provider_config import ProviderConfig

from .settings import NowcastConfig

# ---------------------------------------------------------------------------
# Default config instance — source of truth for all constants below
# ---------------------------------------------------------------------------

_DEFAULT = NowcastConfig()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Paths imported from nfp_lookups.paths

# ---------------------------------------------------------------------------
# Model constants — re-exported from _DEFAULT
# ---------------------------------------------------------------------------

# QCEW observation noise
LOG_SIGMA_QCEW_MID_MU = _DEFAULT.model.qcew.log_sigma_mid_mu
LOG_SIGMA_QCEW_MID_SD = _DEFAULT.model.qcew.log_sigma_mid_sd
LOG_SIGMA_QCEW_BOUNDARY_MU = _DEFAULT.model.qcew.log_sigma_boundary_mu
LOG_SIGMA_QCEW_BOUNDARY_SD = _DEFAULT.model.qcew.log_sigma_boundary_sd
QCEW_NU: int = _DEFAULT.model.qcew.nu

# Post-COVID era multiplier for QCEW boundary-month (M1+M3) noise
_mult, _dflt = _DEFAULT.model.qcew.post_covid_boundary_mult.as_dict()
QCEW_POST_COVID_BOUNDARY_ERA_MULT: dict[int, float] = _mult
QCEW_POST_COVID_BOUNDARY_ERA_DEFAULT: float = _dflt

# CES observation noise
LOG_SIGMA_CES_MU = _DEFAULT.model.ces.log_sigma_mu
LOG_SIGMA_CES_SD = _DEFAULT.model.ces.log_sigma_sd

# Fourier seasonal innovation
LOG_SIGMA_FOURIER_MU = _DEFAULT.model.fourier.log_sigma_mu
LOG_SIGMA_FOURIER_SD = _DEFAULT.model.fourier.log_sigma_sd

# Latent AR(1) marginal SD (tau)
LOG_TAU_MU = _DEFAULT.model.latent.log_tau_mu
LOG_TAU_SD = _DEFAULT.model.latent.log_tau_sd

# BD innovation sigma
LOG_SIGMA_BD_MU = _DEFAULT.model.birth_death.log_sigma_mu
LOG_SIGMA_BD_SD = _DEFAULT.model.birth_death.log_sigma_sd

# QCEW publication lag in months (for BD proxy computation)
BD_QCEW_LAG = _DEFAULT.model.birth_death.qcew_lag

# Fourier seasonal expansion: number of harmonics (K)
N_HARMONICS = _DEFAULT.model.fourier.n_harmonics

# Eras
N_ERAS = _DEFAULT.model.eras.n_eras
ERA_BREAKS = list(_DEFAULT.model.eras.breaks)

# Cyclical indicators — convert from IndicatorConfig list to legacy list[dict]
CYCLICAL_INDICATORS: list[dict] = [
    {"name": i.name, "fred_id": i.fred_id, "freq": i.freq}
    for i in _DEFAULT.indicators
]

# Compositing
MIN_PSEUDO_ESTABS_PER_CELL: int = _DEFAULT.model.compositing.min_pseudo_estabs_per_cell

# Plot colours — one per provider, cycled if >7 providers (not in TOML)
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
# ProviderConfig imported from nfp_lookups.provider_config
# Active provider list — derived from settings default
# ---------------------------------------------------------------------------

def providers_from_settings(cfg: NowcastConfig) -> list[ProviderConfig]:
    """Convert Pydantic provider settings to dataclass ProviderConfig list."""
    return [
        ProviderConfig(
            name=p.name,
            file=p.file,
            error_model=p.error_model,
            birth_file=p.birth_file,
            industry_type=p.industry_type,
            industry_code=p.industry_code,
            geography_type=p.geography_type,
            geography_code=p.geography_code,
        )
        for p in cfg.providers
    ]


PROVIDERS: list[ProviderConfig] = providers_from_settings(_DEFAULT)
