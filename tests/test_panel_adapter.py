"""Tests for panel_to_model_data adapter (panel → model data dict)."""

from __future__ import annotations

import numpy as np
import pytest

from alt_nfp.config import PROVIDERS
from alt_nfp.ingest import build_panel
from alt_nfp.panel_adapter import build_obs_sources, panel_to_model_data


class TestPanelToModelData:
    """Adapter output has the expected structure for build_model."""

    @pytest.fixture(scope="class")
    def model_data(self):
        panel = build_panel()
        if len(panel) == 0:
            pytest.skip("No data available (vintage store or providers missing)")
        return panel_to_model_data(panel, PROVIDERS)

    def test_calendar_arrays(self, model_data):
        data = model_data
        assert data["T"] == len(data["dates"])
        assert data["month_of_year"].shape == (data["T"],)
        assert data["year_of_obs"].shape == (data["T"],)
        assert data["n_years"] == int(data["year_of_obs"].max()) + 1

    def test_growth_arrays(self, model_data):
        data = model_data
        T = data["T"]
        assert data["g_ces_sa"].shape == (T,)
        assert data["g_ces_nsa"].shape == (T,)
        assert data["g_qcew"].shape == (T,)

    def test_obs_index_arrays(self, model_data):
        data = model_data
        for key in ("ces_sa_obs", "ces_nsa_obs", "qcew_obs"):
            arr = data[key]
            assert arr.ndim == 1
            assert all(0 <= i < data["T"] for i in arr)

    def test_qcew_tier_and_mult(self, model_data):
        data = model_data
        n_qcew = len(data["qcew_obs"])
        assert data["qcew_is_m2"].shape == (n_qcew,)
        assert data["qcew_noise_mult"].shape == (n_qcew,)
        if n_qcew > 0:
            assert data["qcew_is_m2"].dtype == bool
            assert data["qcew_noise_mult"].dtype == float
            assert (data["qcew_noise_mult"] >= 1.0).all()

    def test_provider_data(self, model_data):
        data = model_data
        assert len(data["pp_data"]) == len(PROVIDERS)
        for pp in data["pp_data"]:
            assert "g_pp" in pp
            assert "pp_obs" in pp
            assert pp["g_pp"].shape == (data["T"],)

    def test_bd_covariates(self, model_data):
        data = model_data
        T = data["T"]
        assert data["birth_rate"].shape == (T,)
        assert data["birth_rate_c"].shape == (T,)
        assert data["bd_qcew_lagged"].shape == (T,)
        assert data["bd_qcew_c"].shape == (T,)

    def test_vintage_arrays(self, model_data):
        data = model_data
        T = data["T"]
        for v_list in (data["g_ces_sa_by_vintage"], data["g_ces_nsa_by_vintage"]):
            assert len(v_list) == 3
            for arr in v_list:
                assert arr.shape == (T,)

    def test_levels_dataframe(self, model_data):
        data = model_data
        levels = data["levels"]
        assert "ref_date" in levels.columns
        assert "ces_sa_index" in levels.columns


class TestBuildObsSources:
    def test_returns_dict(self):
        panel = build_panel()
        if len(panel) == 0:
            pytest.skip("No data available")
        data = panel_to_model_data(panel, PROVIDERS)
        sources = build_obs_sources(data)
        assert isinstance(sources, dict)
        assert "obs_qcew" in sources
        for key, (label, arr) in sources.items():
            assert isinstance(label, str)
            assert isinstance(arr, np.ndarray)
