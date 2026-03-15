"""Tests for build_model() — era-specific and scalar parameter paths."""

from __future__ import annotations

import numpy as np
import pytest

from nfp_models.config import N_ERAS, PROVIDERS
from nfp_ingest import build_panel
from nfp_models.model import build_model
from nfp_models.panel_adapter import panel_to_model_data


@pytest.fixture(scope="module")
def model_data():
    panel = build_panel()
    if len(panel) == 0:
        pytest.skip("No data available (vintage store or providers missing)")
    return panel_to_model_data(panel, PROVIDERS)


class TestEraSpecificModel:
    """Model builds correctly with era_idx in the data dict."""

    def test_era_params_exist(self, model_data):
        model = build_model(model_data)
        var_names = {v.name for v in model.free_RVs}
        assert "mu_g_era" in var_names
        assert "phi_raw" in var_names
        assert "mu_g" not in var_names

    def test_era_param_shapes(self, model_data):
        model = build_model(model_data)
        for rv in model.free_RVs:
            if rv.name == "mu_g_era":
                assert rv.eval().shape == (N_ERAS,)
            if rv.name == "phi_raw":
                assert rv.eval().shape == ()

    def test_era_idx_in_data(self, model_data):
        assert "era_idx" in model_data
        era_idx = model_data["era_idx"]
        assert era_idx.shape == (model_data["T"],)
        assert set(np.unique(era_idx)).issubset(set(range(N_ERAS)))


class TestScalarFallback:
    """Model builds correctly without era_idx (backward compatibility)."""

    def test_scalar_params_exist(self, model_data):
        data_no_era = {k: v for k, v in model_data.items() if k != "era_idx"}
        model = build_model(data_no_era)
        var_names = {v.name for v in model.free_RVs}
        assert "mu_g" in var_names
        assert "phi_raw" in var_names
        assert "mu_g_era" not in var_names
