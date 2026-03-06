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
        n_ces_v = 2  # only 1st and Final observed

        # Mock posterior (sigma_ces shape matches n_ces_v, not hardcoded 3)
        posterior = xr.Dataset({
            "sigma_ces_sa": xr.DataArray(
                np.full((n_chains, n_draws, n_ces_v), 0.003),
                dims=["chain", "draw", "sigma_ces_sa_dim_0"],
            ),
            "sigma_ces_nsa": xr.DataArray(
                np.full((n_chains, n_draws, n_ces_v), 0.004),
                dims=["chain", "draw", "sigma_ces_nsa_dim_0"],
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

        # Mock data dict — best-available CES with remapped vintage index
        # Original vintages 0 (1st) and 2 (Final) → remapped to 0 and 1.
        ces_vintage_map = {0: 0, 2: 1}
        g_ces_sa = np.full(T, 0.001)
        ces_sa_obs = np.arange(T)
        ces_sa_vintage_idx = np.array([1] * 15 + [0] * 3 + [1] * 6)  # remapped

        data = {
            "T": T,
            "g_ces_sa": g_ces_sa,
            "ces_sa_obs": ces_sa_obs,
            "ces_sa_vintage_idx": ces_sa_vintage_idx,
            "g_ces_nsa": np.full(T, np.nan),
            "ces_nsa_obs": np.array([], dtype=int),
            "ces_nsa_vintage_idx": np.array([], dtype=int),
            "n_ces_vintages": n_ces_v,
            "ces_vintage_map": ces_vintage_map,
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
