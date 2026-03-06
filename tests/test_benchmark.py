"""Tests for alt_nfp.benchmark — benchmark revision extraction."""

from __future__ import annotations

from datetime import date

import arviz as az
import numpy as np
import polars as pl
import pytest
import xarray as xr

from alt_nfp.benchmark import (
    _find_benchmark_window,
    decompose_benchmark_revision,
    extract_benchmark_revision,
    summarize_revision_posterior,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dates(start_year: int = 2023, n_months: int = 36) -> list[date]:
    """Generate end-of-month dates starting from Jan of start_year."""
    import calendar

    dates = []
    y, m = start_year, 1
    for _ in range(n_months):
        last_day = calendar.monthrange(y, m)[1]
        dates.append(date(y, m, last_day))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


def _make_mock_idata(
    n_chains: int = 2,
    n_draws: int = 50,
    T: int = 36,
    g_cont_val: float = 0.002,
    seasonal_val: float = 0.0,
    bd_val: float = 0.001,
) -> az.InferenceData:
    """Create a mock InferenceData with constant posterior values."""
    g_cont = np.full((n_chains, n_draws, T), g_cont_val)
    seasonal = np.full((n_chains, n_draws, T), seasonal_val)
    bd = np.full((n_chains, n_draws, T), bd_val)
    g_total_nsa = g_cont + seasonal + bd

    posterior = xr.Dataset(
        {
            "g_cont": (["chain", "draw", "g_cont_dim_0"], g_cont),
            "seasonal": (["chain", "draw", "seasonal_dim_0"], seasonal),
            "bd": (["chain", "draw", "bd_dim_0"], bd),
            "g_total_nsa": (
                ["chain", "draw", "g_total_nsa_dim_0"],
                g_total_nsa,
            ),
        },
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
        },
    )
    return az.InferenceData(posterior=posterior)


def _make_mock_data(
    dates: list[date],
    g_ces_nsa_val: float = 0.0015,
    anchor_level: float = 130_000.0,
) -> dict:
    """Create a mock data dict with constant CES NSA growth."""
    T = len(dates)
    g_ces_nsa = np.full(T, g_ces_nsa_val)

    # Build a levels DataFrame with the anchor level at March dates
    levels = pl.DataFrame(
        {
            "ref_date": dates,
            "ces_nsa_index": [100.0] * T,
            "ces_sa_index": [100.0] * T,
            "ces_nsa_level": [anchor_level] * T,
            "ces_sa_level": [anchor_level] * T,
            "qcew_nsa_index": [100.0] * T,
        }
    )

    return {
        "dates": dates,
        "T": T,
        "g_ces_nsa": g_ces_nsa,
        "levels": levels,
    }


# ---------------------------------------------------------------------------
# Tests: _find_benchmark_window
# ---------------------------------------------------------------------------


class TestFindBenchmarkWindow:
    """Tests for benchmark window identification."""

    def test_window_2025(self):
        """April 2024 through March 2025 gives 12 indices."""
        dates = _make_dates(2023, 36)  # Jan 2023 – Dec 2025
        idx = _find_benchmark_window(dates, 2025)
        assert len(idx) == 12

        # First month should be April 2024
        assert dates[idx[0]].month == 4
        assert dates[idx[0]].year == 2024
        # Last month should be March 2025
        assert dates[idx[-1]].month == 3
        assert dates[idx[-1]].year == 2025

    def test_window_outside_range_raises(self):
        """March year outside the date range raises ValueError."""
        dates = _make_dates(2023, 12)  # only 2023
        with pytest.raises(ValueError, match="requires 12 months"):
            _find_benchmark_window(dates, 2020)


# ---------------------------------------------------------------------------
# Tests: extract_benchmark_revision
# ---------------------------------------------------------------------------


class TestExtractBenchmarkRevision:
    """Tests for the core extraction function."""

    def test_synthetic_known_revision(self):
        """Verify extraction with hand-computable constant growth values.

        With constant latent growth g=0.003/mo and CES growth g_ces=0.0015/mo
        over 12 months, the cumulative divergence is:
            L * (exp(12 * 0.003) - exp(12 * 0.0015))
        """
        g_latent = 0.003  # g_cont + seasonal + bd
        g_ces = 0.0015
        L = 130_000.0
        dates = _make_dates(2023, 36)
        idata = _make_mock_idata(T=36, g_cont_val=0.002, seasonal_val=0.0, bd_val=0.001)
        data = _make_mock_data(dates, g_ces_nsa_val=g_ces, anchor_level=L)

        samples = extract_benchmark_revision(idata, data, 2025)

        expected = L * (np.exp(12 * g_latent) - np.exp(12 * g_ces))
        # All draws should give the same value (constant posterior)
        np.testing.assert_allclose(samples, expected, rtol=1e-6)

    def test_explicit_anchor_level(self):
        """The anchor_level kwarg overrides auto-detection."""
        dates = _make_dates(2023, 36)
        idata = _make_mock_idata(T=36)
        data = _make_mock_data(dates, anchor_level=100_000.0)

        # Use a different anchor level
        samples_override = extract_benchmark_revision(idata, data, 2025, anchor_level=150_000.0)
        samples_default = extract_benchmark_revision(idata, data, 2025)

        # Ratio should match the anchor level ratio
        ratio = np.mean(samples_override) / np.mean(samples_default)
        np.testing.assert_allclose(ratio, 150_000.0 / 100_000.0, rtol=1e-6)

    def test_output_is_1d(self):
        """Output should be a flattened 1-D array."""
        dates = _make_dates(2023, 36)
        idata = _make_mock_idata(n_chains=2, n_draws=50, T=36)
        data = _make_mock_data(dates)

        samples = extract_benchmark_revision(idata, data, 2025)
        assert samples.ndim == 1
        assert len(samples) == 2 * 50  # chains * draws

    def test_outside_estimation_window_raises(self):
        """March year requiring dates outside estimation period raises."""
        dates = _make_dates(2024, 12)  # only Jan–Dec 2024
        idata = _make_mock_idata(T=12)
        data = _make_mock_data(dates)

        with pytest.raises(ValueError, match="requires 12 months"):
            extract_benchmark_revision(idata, data, 2020)


# ---------------------------------------------------------------------------
# Tests: decompose_benchmark_revision
# ---------------------------------------------------------------------------


class TestDecomposeBenchmarkRevision:
    """Tests for the revision decomposition."""

    def test_decomposition_approximately_additive(self):
        """total ≈ cont_divergence + bd_accumulation (up to cross term)."""
        dates = _make_dates(2023, 36)
        idata = _make_mock_idata(T=36, g_cont_val=0.002, seasonal_val=0.0005, bd_val=0.001)
        data = _make_mock_data(dates, g_ces_nsa_val=0.002)

        decomp = decompose_benchmark_revision(idata, data, 2025)

        total = decomp["total"]
        cont_plus_bd = decomp["cont_divergence"] + decomp["bd_accumulation"]

        # The cross term (from multiplicative → additive approximation)
        # should be small relative to the total
        cross_term = np.abs(total - cont_plus_bd)
        assert np.all(cross_term < 0.1 * np.abs(total) + 1.0)

    def test_bd_dominates_when_cont_matches_ces(self):
        """When g_cont + seasonal ≈ g_ces, BD should dominate."""
        dates = _make_dates(2023, 36)
        # Set cont + seasonal = 0.002, same as CES → cont_divergence ≈ 0
        idata = _make_mock_idata(T=36, g_cont_val=0.002, seasonal_val=0.0, bd_val=0.001)
        data = _make_mock_data(dates, g_ces_nsa_val=0.002)

        decomp = decompose_benchmark_revision(idata, data, 2025)

        # cont_divergence should be near zero
        assert np.abs(np.mean(decomp["cont_divergence"])) < 1.0
        # bd_accumulation should be positive (bd_val > 0)
        assert np.mean(decomp["bd_accumulation"]) > 0


# ---------------------------------------------------------------------------
# Tests: summarize_revision_posterior
# ---------------------------------------------------------------------------


class TestSummarizeRevisionPosterior:
    """Tests for posterior summary statistics."""

    def test_basic_statistics(self):
        """Verify mean, median, std are computed correctly."""
        rng = np.random.default_rng(42)
        samples = rng.normal(-500, 100, size=10_000)

        result = summarize_revision_posterior(samples)

        np.testing.assert_allclose(result["mean"], -500, atol=5)
        np.testing.assert_allclose(result["median"], -500, atol=5)
        np.testing.assert_allclose(result["std"], 100, atol=5)
        assert "hdi_5" in result
        assert "hdi_95" in result
        # 90% HDI should bracket the mean
        assert result["hdi_5"] < result["mean"] < result["hdi_95"]

    def test_with_actual(self):
        """When actual is provided, error = mean − actual."""
        samples = np.full(100, -600.0)

        result = summarize_revision_posterior(samples, actual=-598.0)

        assert result["actual"] == -598.0
        np.testing.assert_allclose(result["error"], -2.0, atol=0.01)

    def test_without_actual(self):
        """Without actual, 'actual' and 'error' keys are absent."""
        samples = np.full(100, -500.0)

        result = summarize_revision_posterior(samples)

        assert "actual" not in result
        assert "error" not in result
