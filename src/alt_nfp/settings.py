"""Pydantic-validated TOML configuration for the nowcast pipeline.

This module defines the ``NowcastConfig`` root model and all sub-models.
A TOML file is optional — ``NowcastConfig()`` with no arguments reproduces
the legacy ``config.py`` defaults exactly.

Usage::

    from alt_nfp.settings import load_config, NowcastConfig

    # Pure defaults (identical to legacy config.py)
    cfg = NowcastConfig()

    # Load from TOML with partial overrides
    cfg = load_config(Path("experiments/tight_qcew.toml"))
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class PathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    data_dir: str = "data"
    output_dir: str = "output"


class LatentConfig(BaseModel):
    """AR(1) latent growth process."""

    model_config = ConfigDict(frozen=True)

    log_tau_mu: float = math.log(0.013)
    log_tau_sd: Annotated[float, Field(gt=0)] = 0.5


class CESConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    log_sigma_mu: float = math.log(0.002)
    log_sigma_sd: Annotated[float, Field(gt=0)] = 0.5


class QCEWPostCovidBoundaryMult(BaseModel):
    """Era multipliers keyed by revision number."""

    model_config = ConfigDict(frozen=True, extra="allow")

    default: float = 1.0

    def as_dict(self) -> tuple[dict[int, float], float]:
        """Return (int-keyed multiplier dict, default value)."""
        mult: dict[int, float] = {}
        if self.__pydantic_extra__:
            for k, v in self.__pydantic_extra__.items():
                mult[int(k)] = float(v)
        return mult, self.default


class QCEWConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    nu: Annotated[int, Field(ge=2)] = 5
    log_sigma_mid_mu: float = math.log(0.0005)
    log_sigma_mid_sd: Annotated[float, Field(gt=0)] = 0.15
    log_sigma_boundary_mu: float = math.log(0.002)
    log_sigma_boundary_sd: Annotated[float, Field(gt=0)] = 0.5
    post_covid_boundary_mult: QCEWPostCovidBoundaryMult = QCEWPostCovidBoundaryMult(
        **{"0": 5.0, "1": 3.5, "2": 2.0, "default": 1.0}
    )


class FourierConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_harmonics: Annotated[int, Field(ge=1, le=12)] = 4
    log_sigma_mu: float = math.log(0.0003)
    log_sigma_sd: Annotated[float, Field(gt=0)] = 0.5


class BirthDeathConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    qcew_lag: Annotated[int, Field(ge=1)] = 6
    log_sigma_mu: float = math.log(0.003)
    log_sigma_sd: Annotated[float, Field(gt=0)] = 0.5


class ErasConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    breaks: list[date] = [date(2020, 1, 1)]

    @field_validator("breaks")
    @classmethod
    def breaks_sorted(cls, v: list[date]) -> list[date]:
        if v != sorted(v):
            raise ValueError("era breaks must be in chronological order")
        return v

    @property
    def n_eras(self) -> int:
        return len(self.breaks) + 1


class CompositingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_pseudo_estabs_per_cell: Annotated[int, Field(ge=1)] = 5


class ModelConfig(BaseModel):
    """All model hyperparameters."""

    model_config = ConfigDict(frozen=True)

    latent: LatentConfig = LatentConfig()
    ces: CESConfig = CESConfig()
    qcew: QCEWConfig = QCEWConfig()
    fourier: FourierConfig = FourierConfig()
    birth_death: BirthDeathConfig = BirthDeathConfig()
    eras: ErasConfig = ErasConfig()
    compositing: CompositingConfig = CompositingConfig()


class ProviderSettingsConfig(BaseModel):
    """Provider specification (Pydantic version)."""

    model_config = ConfigDict(frozen=True)

    name: str
    file: str
    error_model: Literal["iid", "ar1"] = "iid"
    birth_file: str | None = None
    industry_type: str = "national"
    industry_code: str = "00"
    geography_type: str = "national"
    geography_code: str = "00"


class IndicatorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    fred_id: str
    freq: Literal["weekly", "monthly"]
    pub_lag: Annotated[int, Field(ge=0)]


class SamplingPreset(BaseModel):
    model_config = ConfigDict(frozen=True)

    draws: Annotated[int, Field(ge=100)] = 4000
    tune: Annotated[int, Field(ge=100)] = 3000
    chains: Annotated[int, Field(ge=1)] = 4
    target_accept: Annotated[float, Field(gt=0, lt=1)] = 0.95

    def to_pymc_kwargs(self) -> dict:
        return {
            "draws": self.draws,
            "tune": self.tune,
            "chains": self.chains,
            "target_accept": self.target_accept,
            "return_inferencedata": True,
        }


class SamplingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default: SamplingPreset = SamplingPreset()
    light: SamplingPreset = SamplingPreset(draws=2000, tune=2000, chains=2)
    medium: SamplingPreset = SamplingPreset()

    def get_preset(self, name: str) -> SamplingPreset:
        if not hasattr(self, name):
            raise ValueError(f"unknown sampling preset: {name!r}")
        return getattr(self, name)


class BacktestConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_months: Annotated[int, Field(ge=1)] = 24
    sampling_preset: str = "light"


class ForecastConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    end_date: date = date(2026, 1, 12)


class PublicationLagsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_weeks: Annotated[int, Field(ge=0)] = 3


# ---------------------------------------------------------------------------
# Resolved paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedPaths:
    base_dir: Path
    data_dir: Path
    output_dir: Path

    @property
    def store_dir(self) -> Path:
        return self.data_dir / "store"

    @property
    def downloads_dir(self) -> Path:
        return self.data_dir / "downloads"

    @property
    def intermediate_dir(self) -> Path:
        return self.data_dir / "intermediate"

    @property
    def indicators_dir(self) -> Path:
        return self.data_dir / "indicators"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class NowcastConfig(BaseModel):
    """Root configuration for the entire nowcast pipeline."""

    model_config = ConfigDict(frozen=True)

    paths: PathsConfig = PathsConfig()
    model: ModelConfig = ModelConfig()
    providers: list[ProviderSettingsConfig] = [
        ProviderSettingsConfig(
            name="G",
            file="providers/g/g_provider.parquet",
            error_model="iid",
            birth_file="providers/g/g_births.parquet",
        )
    ]
    indicators: list[IndicatorConfig] = [
        IndicatorConfig(name="claims", fred_id="ICNSA", freq="weekly", pub_lag=1),
        IndicatorConfig(name="jolts", fred_id="JTSJOL", freq="monthly", pub_lag=2),
    ]
    sampling: SamplingConfig = SamplingConfig()
    backtest: BacktestConfig = BacktestConfig()
    forecast: ForecastConfig = ForecastConfig()
    publication_lags: PublicationLagsConfig = PublicationLagsConfig()

    @model_validator(mode="after")
    def _check_backtest_preset_exists(self) -> NowcastConfig:
        self.sampling.get_preset(self.backtest.sampling_preset)
        return self

    def resolve_paths(self, base_dir: Path) -> ResolvedPaths:
        return ResolvedPaths(
            base_dir=base_dir,
            data_dir=base_dir / self.paths.data_dir,
            output_dir=base_dir / self.paths.output_dir,
        )


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------


def load_config(path: Path | None = None, *, base_dir: Path | None = None) -> NowcastConfig:
    """Load and validate a TOML config, falling back to defaults.

    Parameters
    ----------
    path
        Path to a TOML file.  *None* -> pure defaults (current behaviour).
    base_dir
        Repository root.  Defaults to the parent of ``src/alt_nfp``.
    """
    if path is not None:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        cfg = NowcastConfig.model_validate(raw)
    else:
        cfg = NowcastConfig()
    return cfg


def save_config(cfg: NowcastConfig, path: Path) -> None:
    """Write the resolved config as TOML (run receipt)."""
    import tomli_w

    data = cfg.model_dump(mode="python")
    # Convert date objects to ISO strings for TOML serialisation
    _convert_dates(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tomli_w.dumps(data).encode())


def _convert_dates(obj):
    """Recursively convert date objects to ISO strings in a nested dict."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, date):
                obj[k] = v.isoformat()
            elif isinstance(v, (dict, list)):
                _convert_dates(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, date):
                obj[i] = v.isoformat()
            elif isinstance(v, (dict, list)):
                _convert_dates(v)


# ---------------------------------------------------------------------------
# Sensitivity helpers
# ---------------------------------------------------------------------------


def build_sensitivity_configs(
    base: NowcastConfig,
    multipliers: list[tuple[str, float, float]],
) -> list[tuple[str, NowcastConfig]]:
    """Build variant configs for QCEW sigma sensitivity analysis.

    Parameters
    ----------
    base
        Baseline config.
    multipliers
        List of ``(label, mid_scale, boundary_scale)`` tuples.

    Returns
    -------
    list of (label, NowcastConfig) pairs.
    """
    configs: list[tuple[str, NowcastConfig]] = []
    for label, mid_mult, boundary_mult in multipliers:
        new_qcew = base.model.qcew.model_copy(
            update={
                "log_sigma_mid_mu": base.model.qcew.log_sigma_mid_mu + math.log(mid_mult),
                "log_sigma_boundary_mu": (
                    base.model.qcew.log_sigma_boundary_mu + math.log(boundary_mult)
                ),
            }
        )
        new_model = base.model.model_copy(update={"qcew": new_qcew})
        new_cfg = base.model_copy(update={"model": new_model})
        configs.append((label, new_cfg))
    return configs
