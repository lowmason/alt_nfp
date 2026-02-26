"""Tests for panel_to_model_data adapter (panel → model data dict)."""

from __future__ import annotations

import numpy as np
import pytest

from alt_nfp.config import DATA_DIR, PROVIDERS
from alt_nfp.data import load_data
from alt_nfp.ingest import build_panel
from alt_nfp.panel_adapter import panel_to_model_data


def _model_data_keys():
    """Keys that load_data() returns (and adapter must match)."""
    data = load_data()
    return set(data.keys())


class TestPanelToModelDataShape:
    """Adapter output has the same structure as load_data()."""

    def test_adapter_returns_same_keys_as_load_data(self):
        """panel_to_model_data(panel, providers) returns the same keys as load_data()."""
        # Build minimal panel from legacy if CSVs exist
        panel = build_panel(use_legacy=True)
        if len(panel) == 0:
            pytest.skip("No legacy data (ces_index.csv / qcew_index.csv missing)")
        data = panel_to_model_data(panel, PROVIDERS)
        legacy = load_data()
        assert set(data.keys()) == set(legacy.keys()), "Adapter and load_data keys differ"

    def test_adapter_produces_valid_model_input(self):
        """Adapter output has correct types and shapes for build_model."""
        panel = build_panel(use_legacy=True)
        if len(panel) == 0:
            pytest.skip("No legacy data")
        data = panel_to_model_data(panel, PROVIDERS)
        assert data["T"] == len(data["dates"])
        assert data["month_of_year"].shape == (data["T"],)
        assert data["year_of_obs"].shape == (data["T"],)
        assert data["n_years"] == int(data["year_of_obs"].max()) + 1
        assert data["g_ces_sa"].shape == (data["T"],)
        assert data["g_qcew"].shape == (data["T"],)
        assert len(data["pp_data"]) == len(PROVIDERS)


class TestPanelAdapterEquivalenceToLegacy:
    """When panel is built from legacy CSVs, adapter output matches load_data()."""

    def test_legacy_panel_adapter_matches_load_data(self):
        """panel_to_model_data(legacy_panel) matches load_data() in T, n_years, and key arrays."""
        panel = build_panel(use_legacy=True)
        if len(panel) == 0:
            pytest.skip("No legacy data")
        data_panel = panel_to_model_data(panel, PROVIDERS)
        data_legacy = load_data()

        assert data_panel["T"] == data_legacy["T"], "T must match"
        assert data_panel["n_years"] == data_legacy["n_years"], "n_years must match"
        assert data_panel["dates"] == data_legacy["dates"], "dates must match"

        def vec_close(a: np.ndarray, b: np.ndarray) -> bool:
            return bool(np.allclose(a, b, equal_nan=True))

        assert vec_close(data_panel["g_ces_sa"], data_legacy["g_ces_sa"])
        assert vec_close(data_panel["g_ces_nsa"], data_legacy["g_ces_nsa"])
        assert vec_close(data_panel["g_qcew"], data_legacy["g_qcew"])
        assert np.array_equal(data_panel["ces_sa_obs"], data_legacy["ces_sa_obs"])
        assert np.array_equal(data_panel["ces_nsa_obs"], data_legacy["ces_nsa_obs"])
        assert np.array_equal(data_panel["qcew_obs"], data_legacy["qcew_obs"])
        assert np.array_equal(data_panel["qcew_is_m3"], data_legacy["qcew_is_m3"])

        for i, (pp_p, pp_l) in enumerate(zip(data_panel["pp_data"], data_legacy["pp_data"])):
            assert pp_p["name"] == pp_l["name"]
            assert vec_close(pp_p["g_pp"], pp_l["g_pp"])
            assert np.array_equal(pp_p["pp_obs"], pp_l["pp_obs"])

        assert vec_close(data_panel["birth_rate"], data_legacy["birth_rate"])
        assert vec_close(data_panel["bd_qcew_lagged"], data_legacy["bd_qcew_lagged"])
        assert np.isclose(data_panel["birth_rate_mean"], data_legacy["birth_rate_mean"])
        assert np.isclose(data_panel["bd_qcew_mean"], data_legacy["bd_qcew_mean"])
