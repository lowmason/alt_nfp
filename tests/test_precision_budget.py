"""Tests for compute_precision_budget (Step 1.6).

Verifies the structured DataFrame output of the precision budget
computation, including column structure and share normalization.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestComputePrecisionBudget:
    """Test compute_precision_budget returns well-formed DataFrame."""

    @pytest.fixture()
    def mock_idata_and_data(self):
        """Create minimal mock InferenceData and data dict for testing."""
        import xarray as xr

        n_chains, n_draws = 2, 50
        T = 24

        # Mock posterior
        posterior = xr.Dataset({
            "sigma_ces_sa": xr.DataArray(
                np.full((n_chains, n_draws, 3), 0.003),
                dims=["chain", "draw", "vintage"],
            ),
            "sigma_ces_nsa": xr.DataArray(
                np.full((n_chains, n_draws, 3), 0.004),
                dims=["chain", "draw", "vintage"],
            ),
            "lambda_ces": xr.DataArray(
                np.full((n_chains, n_draws), 0.95),
                dims=["chain", "draw"],
            ),
            "alpha_ces": xr.DataArray(
                np.full((n_chains, n_draws), 0.0001),
                dims=["chain", "draw"],
            ),
            "sigma_qcew_mid": xr.DataArray(
                np.full((n_chains, n_draws), 0.0005),
                dims=["chain", "draw"],
            ),
            "sigma_qcew_boundary": xr.DataArray(
                np.full((n_chains, n_draws), 0.002),
                dims=["chain", "draw"],
            ),
        })

        class MockIData:
            def __init__(self):
                self.posterior = posterior

        # Mock data dict
        g_ces_sa_v1 = np.full(T, np.nan)
        g_ces_sa_v1[:20] = 0.001
        g_ces_sa_v2 = np.full(T, np.nan)
        g_ces_sa_v2[:18] = 0.001
        g_ces_sa_v3 = np.full(T, np.nan)
        g_ces_sa_v3[:15] = 0.001

        data = {
            "T": T,
            "g_ces_sa_by_vintage": [g_ces_sa_v1, g_ces_sa_v2, g_ces_sa_v3],
            "g_ces_nsa_by_vintage": [
                np.full(T, np.nan),
                np.full(T, np.nan),
                np.full(T, np.nan),
            ],
            "qcew_obs": np.arange(20),
            "qcew_is_m2": np.array([i % 3 == 1 for i in range(20)]),
            "qcew_noise_mult": np.full(20, 1.0),
            "pp_data": [],
        }

        return MockIData(), data

    def test_returns_dataframe(self, mock_idata_and_data):
        from alt_nfp.diagnostics import compute_precision_budget

        idata, data = mock_idata_and_data
        df = compute_precision_budget(idata, data)
        assert hasattr(df, "columns")

    def test_has_required_columns(self, mock_idata_and_data):
        from alt_nfp.diagnostics import compute_precision_budget

        idata, data = mock_idata_and_data
        df = compute_precision_budget(idata, data)
        required = {"source", "n_obs", "precision_per_obs", "total_precision", "share"}
        assert required.issubset(set(df.columns))

    def test_shares_sum_to_one(self, mock_idata_and_data):
        from alt_nfp.diagnostics import compute_precision_budget

        idata, data = mock_idata_and_data
        df = compute_precision_budget(idata, data)
        total_share = df["share"].sum()
        assert abs(total_share - 1.0) < 1e-10

    def test_lambda_and_alpha_columns(self, mock_idata_and_data):
        from alt_nfp.diagnostics import compute_precision_budget

        idata, data = mock_idata_and_data
        df = compute_precision_budget(idata, data)
        assert "lambda_mean" in df.columns
        assert "alpha_mean" in df.columns

    def test_qcew_lambda_is_one(self, mock_idata_and_data):
        from alt_nfp.diagnostics import compute_precision_budget

        idata, data = mock_idata_and_data
        df = compute_precision_budget(idata, data)
        qcew_rows = df.filter(df["source"].str.starts_with("QCEW"))
        for lam in qcew_rows["lambda_mean"].to_list():
            assert lam == 1.0
