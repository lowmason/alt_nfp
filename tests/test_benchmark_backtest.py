"""Tests for the Phase 2 benchmark backtest infrastructure.

Covers:
- horizon_to_as_of mapping
- as_of censoring in panel_to_model_data
- Evaluation metrics computation
- Comparative benchmarks
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from alt_nfp.benchmark_backtest import (
    DEFAULT_YEARS,
    EXTENDED_YEARS,
    HORIZONS,
    build_comparative_benchmarks,
    comparative_rmse,
    compute_backtest_metrics,
    horizon_to_as_of,
)
from alt_nfp.lookups.benchmark_revisions import BENCHMARK_REVISIONS


# ── horizon_to_as_of tests ──────────────────────────────────────────────

class TestHorizonToAsOf:
    def test_t12_is_april(self):
        d = horizon_to_as_of(2025, "T-12")
        assert d.month == 4
        assert d.year == 2025

    def test_t9_is_july(self):
        d = horizon_to_as_of(2025, "T-9")
        assert d.month == 7
        assert d.year == 2025

    def test_t6_is_october(self):
        d = horizon_to_as_of(2025, "T-6")
        assert d.month == 10
        assert d.year == 2025

    def test_t3_is_january_next(self):
        d = horizon_to_as_of(2025, "T-3")
        assert d.month == 1
        assert d.year == 2026

    def test_t1_is_february_next(self):
        d = horizon_to_as_of(2025, "T-1")
        assert d.month == 2
        assert d.year == 2026

    def test_horizons_are_monotonically_increasing(self):
        dates = [horizon_to_as_of(2025, h) for h in HORIZONS]
        for i in range(len(dates) - 1):
            assert dates[i] < dates[i + 1], (
                f"{HORIZONS[i]}={dates[i]} should be before "
                f"{HORIZONS[i + 1]}={dates[i + 1]}"
            )

    def test_unknown_horizon_raises(self):
        with pytest.raises(ValueError, match="Unknown horizon"):
            horizon_to_as_of(2025, "T-7")

    def test_works_for_all_extended_years(self):
        for year in EXTENDED_YEARS:
            for horizon in HORIZONS:
                d = horizon_to_as_of(year, horizon)
                assert isinstance(d, date)


# ── Evaluation metrics tests ─────────────────────────────────────────────

class TestComputeBacktestMetrics:
    @pytest.fixture()
    def sample_results(self) -> pl.DataFrame:
        rows = []
        for horizon in HORIZONS:
            for year in [2022, 2023, 2024, 2025]:
                actual = BENCHMARK_REVISIONS[year]
                assert actual is not None
                mean = actual + np.random.default_rng(year).normal(0, 50)
                rows.append({
                    "march_year": year,
                    "horizon": horizon,
                    "as_of_date": horizon_to_as_of(year, horizon),
                    "posterior_mean": mean,
                    "posterior_median": mean - 5,
                    "posterior_std": 200.0,
                    "hdi_5": mean - 350,
                    "hdi_95": mean + 350,
                    "actual": actual,
                    "error": mean - actual,
                    "squared_error": (mean - actual) ** 2,
                })
        return pl.DataFrame(rows)

    def test_returns_one_row_per_horizon(self, sample_results):
        metrics = compute_backtest_metrics(sample_results)
        assert metrics.height == len(HORIZONS)

    def test_rmse_is_positive(self, sample_results):
        metrics = compute_backtest_metrics(sample_results)
        for rmse in metrics["rmse"].to_list():
            assert rmse > 0

    def test_coverage_between_0_and_1(self, sample_results):
        metrics = compute_backtest_metrics(sample_results)
        for cov in metrics["coverage_90"].to_list():
            assert 0.0 <= cov <= 1.0

    def test_n_years_is_correct(self, sample_results):
        metrics = compute_backtest_metrics(sample_results)
        for n in metrics["n_years"].to_list():
            assert n == 4


# ── Comparative benchmarks tests ────────────────────────────────────────

class TestComparativeBenchmarks:
    def test_naive_zero_has_all_years(self):
        df = build_comparative_benchmarks(DEFAULT_YEARS)
        naive = df.filter(pl.col("benchmark") == "naive_zero")
        assert naive.height == len(DEFAULT_YEARS)

    def test_naive_zero_prediction_is_zero(self):
        df = build_comparative_benchmarks(DEFAULT_YEARS)
        naive = df.filter(pl.col("benchmark") == "naive_zero")
        assert all(p == 0.0 for p in naive["prediction"].to_list())

    def test_prior_year_uses_previous_revision(self):
        df = build_comparative_benchmarks([2025])
        prior = df.filter(pl.col("benchmark") == "prior_year")
        assert prior.height == 1
        assert prior["prediction"][0] == BENCHMARK_REVISIONS[2024]

    def test_skips_covid_years(self):
        df = build_comparative_benchmarks([2020, 2021])
        assert df.is_empty()

    def test_comparative_rmse_keys(self):
        df = build_comparative_benchmarks(DEFAULT_YEARS)
        rmse = comparative_rmse(df)
        assert "naive_zero" in rmse
        assert "prior_year" in rmse

    def test_naive_zero_rmse_matches_manual(self):
        years = [y for y in DEFAULT_YEARS if BENCHMARK_REVISIONS.get(y) is not None]
        actuals = [BENCHMARK_REVISIONS[y] for y in years]
        expected_rmse = float(np.sqrt(np.mean(np.array(actuals) ** 2)))

        df = build_comparative_benchmarks(years)
        rmse = comparative_rmse(df)
        assert abs(rmse["naive_zero"] - expected_rmse) < 0.01


# ── as_of censoring tests ───────────────────────────────────────────────

class TestAsOfCensoring:
    """Tests that as_of filtering in panel_to_model_data removes future data.

    These require a panel with vintage_date populated.  They construct
    synthetic panels to verify the filtering logic without needing the
    full vintage store.
    """

    def _make_panel(self, n_months: int = 36) -> pl.DataFrame:
        """Build a minimal synthetic panel for censoring tests."""
        from alt_nfp.ingest.base import PANEL_SCHEMA

        base = date(2023, 1, 1)
        rows: list[dict] = []
        for i in range(n_months):
            m = base.month + i
            year = base.year + (m - 1) // 12
            month = ((m - 1) % 12) + 1
            period = date(year, month, 1)
            # CES SA: published ~1 month later
            ces_pub = date(year + (month // 12), (month % 12) + 1, 7)
            rows.append({
                "period": period,
                "geographic_type": "national",
                "geographic_code": "US",
                "industry_code": "05",
                "industry_level": "supersector",
                "source": "ces_sa",
                "source_type": "official_sa",
                "growth": 0.001 * (1 + 0.1 * np.sin(i)),
                "employment_level": 150_000.0 + i * 100,
                "is_seasonally_adjusted": True,
                "vintage_date": ces_pub,
                "revision_number": 0,
                "is_final": False,
                "publication_lag_months": 1,
                "coverage_ratio": 1.0,
            })
            # QCEW: published ~5 months after quarter end
            q = (month - 1) // 3 + 1
            q_end_month = q * 3
            pub_offset = 5
            qcew_pub_month = q_end_month + pub_offset
            qcew_pub_year = year + (qcew_pub_month - 1) // 12
            qcew_pub_month = ((qcew_pub_month - 1) % 12) + 1
            qcew_pub = date(qcew_pub_year, qcew_pub_month, 15)
            rows.append({
                "period": period,
                "geographic_type": "national",
                "geographic_code": "US",
                "industry_code": "05",
                "industry_level": "supersector",
                "source": "qcew",
                "source_type": "census",
                "growth": 0.0008 * (1 + 0.05 * np.sin(i)),
                "employment_level": 150_000.0 + i * 100,
                "is_seasonally_adjusted": False,
                "vintage_date": qcew_pub,
                "revision_number": 0,
                "is_final": True,
                "publication_lag_months": 5,
                "coverage_ratio": 1.0,
            })

        return pl.DataFrame(rows, schema=PANEL_SCHEMA)

    def test_as_of_reduces_observations(self):
        panel = self._make_panel(36)
        from alt_nfp.panel_adapter import panel_to_model_data

        data_full = panel_to_model_data(panel, [])
        data_censored = panel_to_model_data(panel, [], as_of=date(2024, 6, 15))

        full_ces_obs = len(data_full["ces_sa_obs"])
        censored_ces_obs = len(data_censored["ces_sa_obs"])
        assert censored_ces_obs < full_ces_obs

    def test_as_of_masks_future_qcew(self):
        panel = self._make_panel(36)
        from alt_nfp.panel_adapter import panel_to_model_data

        data = panel_to_model_data(panel, [], as_of=date(2024, 1, 15))
        n_qcew = len(data["qcew_obs"])
        data_later = panel_to_model_data(panel, [], as_of=date(2025, 6, 15))
        n_qcew_later = len(data_later["qcew_obs"])

        assert n_qcew < n_qcew_later, (
            f"Earlier as_of should have fewer QCEW obs: {n_qcew} vs {n_qcew_later}"
        )

    def test_as_of_supersedes_censor_ces_from(self):
        panel = self._make_panel(36)
        from alt_nfp.panel_adapter import panel_to_model_data

        data_asof = panel_to_model_data(
            panel, [],
            censor_ces_from=date(2023, 6, 1),
            as_of=date(2025, 12, 15),
        )
        data_censor = panel_to_model_data(
            panel, [],
            censor_ces_from=date(2023, 6, 1),
        )

        assert len(data_asof["ces_sa_obs"]) > len(data_censor["ces_sa_obs"]), (
            "as_of should supersede censor_ces_from — later as_of keeps more obs"
        )
