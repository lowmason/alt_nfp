"""Tests for alt_nfp.settings (TOML config system)."""

from __future__ import annotations

import math
import tempfile
from datetime import date
from pathlib import Path

import pytest

from alt_nfp.settings import (
    NowcastConfig,
    build_sensitivity_configs,
    load_config,
    save_config,
)


# ---------------------------------------------------------------------------
# Default config matches legacy config.py values
# ---------------------------------------------------------------------------


class TestDefaultMatchesLegacy:
    """Default NowcastConfig() must reproduce every config.py constant."""

    def setup_method(self):
        self.cfg = NowcastConfig()

    def test_qcew_nu(self):
        assert self.cfg.model.qcew.nu == 5

    def test_log_sigma_qcew_mid(self):
        assert self.cfg.model.qcew.log_sigma_mid_mu == pytest.approx(math.log(0.0005))
        assert self.cfg.model.qcew.log_sigma_mid_sd == 0.15

    def test_log_sigma_qcew_boundary(self):
        assert self.cfg.model.qcew.log_sigma_boundary_mu == pytest.approx(math.log(0.002))
        assert self.cfg.model.qcew.log_sigma_boundary_sd == 0.5

    def test_log_sigma_ces(self):
        assert self.cfg.model.ces.log_sigma_mu == pytest.approx(math.log(0.002))
        assert self.cfg.model.ces.log_sigma_sd == 0.5

    def test_log_tau(self):
        assert self.cfg.model.latent.log_tau_mu == pytest.approx(math.log(0.013))
        assert self.cfg.model.latent.log_tau_sd == 0.5

    def test_log_sigma_fourier(self):
        assert self.cfg.model.fourier.log_sigma_mu == pytest.approx(math.log(0.0003))
        assert self.cfg.model.fourier.log_sigma_sd == 0.5

    def test_n_harmonics(self):
        assert self.cfg.model.fourier.n_harmonics == 4

    def test_log_sigma_bd(self):
        assert self.cfg.model.birth_death.log_sigma_mu == pytest.approx(math.log(0.003))
        assert self.cfg.model.birth_death.log_sigma_sd == 0.5

    def test_bd_qcew_lag(self):
        assert self.cfg.model.birth_death.qcew_lag == 6

    def test_eras(self):
        assert self.cfg.model.eras.breaks == [date(2020, 1, 1)]
        assert self.cfg.model.eras.n_eras == 2

    def test_min_pseudo_estabs(self):
        assert self.cfg.model.compositing.min_pseudo_estabs_per_cell == 5

    def test_qcew_post_covid_boundary_mult(self):
        mult, default = self.cfg.model.qcew.post_covid_boundary_mult.as_dict()
        assert mult == {0: 5.0, 1: 3.5, 2: 2.0}
        assert default == 1.0

    def test_providers(self):
        assert len(self.cfg.providers) == 1
        p = self.cfg.providers[0]
        assert p.name == "G"
        assert p.file == "providers/g/g_provider.parquet"
        assert p.error_model == "iid"
        assert p.birth_file == "providers/g/g_births.parquet"

    def test_indicators(self):
        assert len(self.cfg.indicators) == 2
        names = [i.name for i in self.cfg.indicators]
        assert "claims" in names
        assert "jolts" in names
        claims = next(i for i in self.cfg.indicators if i.name == "claims")
        assert claims.fred_id == "ICNSA"
        assert claims.freq == "weekly"
        assert claims.pub_lag == 1

    def test_sampling_presets(self):
        default = self.cfg.sampling.default
        assert default.draws == 4000
        assert default.tune == 3000
        assert default.chains == 4
        assert default.target_accept == 0.95

        light = self.cfg.sampling.light
        assert light.draws == 2000
        assert light.chains == 2

    def test_backtest(self):
        assert self.cfg.backtest.n_months == 24
        assert self.cfg.backtest.sampling_preset == "light"

    def test_forecast_end_date(self):
        assert self.cfg.forecast.end_date == date(2026, 1, 12)

    def test_publication_lags(self):
        assert self.cfg.publication_lags.provider_weeks == 3


# ---------------------------------------------------------------------------
# Partial TOML override
# ---------------------------------------------------------------------------


class TestPartialOverride:
    """Only the specified fields should change; everything else stays default."""

    def test_override_qcew_nu(self, tmp_path):
        toml_path = tmp_path / "override.toml"
        toml_path.write_text('[model.qcew]\nnu = 10\n')
        cfg = load_config(toml_path)
        assert cfg.model.qcew.nu == 10
        # Other QCEW values unchanged
        assert cfg.model.qcew.log_sigma_mid_sd == 0.15

    def test_override_n_harmonics(self, tmp_path):
        toml_path = tmp_path / "override.toml"
        toml_path.write_text('[model.fourier]\nn_harmonics = 6\n')
        cfg = load_config(toml_path)
        assert cfg.model.fourier.n_harmonics == 6
        # Latent config unchanged
        assert cfg.model.latent.log_tau_sd == 0.5

    def test_override_sampling_preset(self, tmp_path):
        toml_path = tmp_path / "override.toml"
        toml_path.write_text('[sampling.default]\ndraws = 8000\nchains = 6\n')
        cfg = load_config(toml_path)
        assert cfg.sampling.default.draws == 8000
        assert cfg.sampling.default.chains == 6
        # Light preset unchanged
        assert cfg.sampling.light.draws == 2000


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_sd_rejected(self):
        with pytest.raises(Exception):
            NowcastConfig(model={"latent": {"log_tau_sd": -0.1}})

    def test_unsorted_era_breaks(self):
        with pytest.raises(Exception):
            NowcastConfig(
                model={"eras": {"breaks": ["2020-01-01", "2015-01-01"]}}
            )

    def test_invalid_sampling_preset_ref(self):
        with pytest.raises(Exception):
            NowcastConfig(backtest={"sampling_preset": "nonexistent"})

    def test_n_harmonics_zero(self):
        with pytest.raises(Exception):
            NowcastConfig(model={"fourier": {"n_harmonics": 0}})

    def test_qcew_nu_below_2(self):
        with pytest.raises(Exception):
            NowcastConfig(model={"qcew": {"nu": 1}})


# ---------------------------------------------------------------------------
# TOML round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_save_load_roundtrip(self, tmp_path):
        cfg = NowcastConfig()
        toml_path = tmp_path / "rt.toml"
        save_config(cfg, toml_path)
        cfg2 = load_config(toml_path)

        # Spot-check key values survive the round trip
        assert cfg2.model.qcew.nu == cfg.model.qcew.nu
        assert cfg2.model.qcew.log_sigma_mid_mu == pytest.approx(cfg.model.qcew.log_sigma_mid_mu)
        assert cfg2.model.fourier.n_harmonics == cfg.model.fourier.n_harmonics
        assert cfg2.model.eras.n_eras == cfg.model.eras.n_eras
        assert cfg2.sampling.default.draws == cfg.sampling.default.draws
        assert len(cfg2.providers) == len(cfg.providers)
        assert cfg2.providers[0].name == cfg.providers[0].name
        assert len(cfg2.indicators) == len(cfg.indicators)

    def test_custom_override_roundtrip(self, tmp_path):
        toml_path = tmp_path / "custom.toml"
        toml_path.write_text('[model.qcew]\nnu = 8\n')
        cfg = load_config(toml_path)
        assert cfg.model.qcew.nu == 8

        toml_path2 = tmp_path / "rt2.toml"
        save_config(cfg, toml_path2)
        cfg2 = load_config(toml_path2)
        assert cfg2.model.qcew.nu == 8


# ---------------------------------------------------------------------------
# ResolvedPaths
# ---------------------------------------------------------------------------


class TestResolvedPaths:
    def test_resolve_paths(self):
        cfg = NowcastConfig()
        base = Path("/tmp/test_project")
        resolved = cfg.resolve_paths(base)
        assert resolved.data_dir == base / "data"
        assert resolved.output_dir == base / "output"
        assert resolved.store_dir == base / "data" / "store"
        assert resolved.downloads_dir == base / "data" / "downloads"
        assert resolved.intermediate_dir == base / "data" / "intermediate"
        assert resolved.indicators_dir == base / "data" / "indicators"

    def test_custom_paths(self, tmp_path):
        toml_path = tmp_path / "paths.toml"
        toml_path.write_text(
            '[paths]\ndata_dir = "mydata"\noutput_dir = "myoutput"\n'
        )
        cfg = load_config(toml_path)
        base = Path("/tmp/proj")
        resolved = cfg.resolve_paths(base)
        assert resolved.data_dir == base / "mydata"
        assert resolved.output_dir == base / "myoutput"


# ---------------------------------------------------------------------------
# Sensitivity configs
# ---------------------------------------------------------------------------


class TestBuildSensitivityConfigs:
    def test_produces_correct_number(self):
        cfg = NowcastConfig()
        mults = [("0.5x", 0.5, 0.5), ("1x", 1.0, 1.0), ("2x", 2.0, 2.0)]
        variants = build_sensitivity_configs(cfg, mults)
        assert len(variants) == 3
        labels = [label for label, _ in variants]
        assert labels == ["0.5x", "1x", "2x"]

    def test_1x_matches_baseline(self):
        cfg = NowcastConfig()
        mults = [("1x", 1.0, 1.0)]
        [(_, variant)] = build_sensitivity_configs(cfg, mults)
        assert variant.model.qcew.log_sigma_mid_mu == pytest.approx(
            cfg.model.qcew.log_sigma_mid_mu
        )

    def test_2x_shifts_log_mu(self):
        cfg = NowcastConfig()
        mults = [("2x", 2.0, 2.0)]
        [(_, variant)] = build_sensitivity_configs(cfg, mults)
        expected = cfg.model.qcew.log_sigma_mid_mu + math.log(2.0)
        assert variant.model.qcew.log_sigma_mid_mu == pytest.approx(expected)

    def test_other_fields_unchanged(self):
        cfg = NowcastConfig()
        mults = [("2x", 2.0, 2.0)]
        [(_, variant)] = build_sensitivity_configs(cfg, mults)
        assert variant.model.ces.log_sigma_mu == cfg.model.ces.log_sigma_mu
        assert variant.model.fourier.n_harmonics == cfg.model.fourier.n_harmonics
        assert variant.model.qcew.nu == cfg.model.qcew.nu


# ---------------------------------------------------------------------------
# SamplingPreset
# ---------------------------------------------------------------------------


class TestSamplingPreset:
    def test_to_pymc_kwargs(self):
        cfg = NowcastConfig()
        kw = cfg.sampling.default.to_pymc_kwargs()
        assert kw["draws"] == 4000
        assert kw["tune"] == 3000
        assert kw["chains"] == 4
        assert kw["target_accept"] == 0.95
        assert kw["return_inferencedata"] is True

    def test_get_preset(self):
        cfg = NowcastConfig()
        light = cfg.sampling.get_preset("light")
        assert light.draws == 2000
        assert light.chains == 2

    def test_unknown_preset_raises(self):
        cfg = NowcastConfig()
        with pytest.raises(ValueError, match="unknown sampling preset"):
            cfg.sampling.get_preset("turbo")


# ---------------------------------------------------------------------------
# providers_from_settings
# ---------------------------------------------------------------------------


class TestProvidersFromSettings:
    def test_converts_to_dataclass(self):
        from alt_nfp.config import ProviderConfig, providers_from_settings

        cfg = NowcastConfig()
        providers = providers_from_settings(cfg)
        assert len(providers) == 1
        p = providers[0]
        assert isinstance(p, ProviderConfig)
        assert p.name == "G"
        assert p.error_model == "iid"
        assert p.birth_file == "providers/g/g_births.parquet"


# ---------------------------------------------------------------------------
# load_config with None path returns defaults
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_none_path_returns_defaults(self):
        cfg = load_config(None)
        default = NowcastConfig()
        assert cfg.model.qcew.nu == default.model.qcew.nu
        assert len(cfg.providers) == len(default.providers)
