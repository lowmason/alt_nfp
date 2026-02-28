"""Tests for cyclical indicator loading, centering, and censoring.

Covers:
- _load_cyclical_indicators() loading from CSV
- JOLTS addition to CYCLICAL_INDICATORS
- Publication-lag censoring for all indicators
- Data coverage verification (Step 1.4)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from alt_nfp.config import CYCLICAL_INDICATORS, DATA_DIR
from alt_nfp.panel_adapter import (
    _CYCLICAL_PUBLICATION_LAGS,
    _load_cyclical_indicators,
    _offset_month,
)


class TestCyclicalIndicatorsConfig:
    """Verify CYCLICAL_INDICATORS config entries."""

    def test_four_indicators_configured(self):
        assert len(CYCLICAL_INDICATORS) == 4

    def test_indicator_names(self):
        names = [spec["name"] for spec in CYCLICAL_INDICATORS]
        assert "claims" in names
        assert "nfci" in names
        assert "biz_apps" in names
        assert "jolts" in names

    def test_each_has_required_keys(self):
        for spec in CYCLICAL_INDICATORS:
            assert "name" in spec
            assert "file" in spec
            assert "col" in spec
            assert "freq" in spec
            assert spec["freq"] in ("weekly", "monthly")

    def test_jolts_config(self):
        jolts = [s for s in CYCLICAL_INDICATORS if s["name"] == "jolts"][0]
        assert jolts["file"] == "jolts_openings.csv"
        assert jolts["col"] == "openings"
        assert jolts["freq"] == "monthly"


class TestPublicationLags:
    """Verify publication lag values are correct."""

    def test_all_indicators_have_lags(self):
        for spec in CYCLICAL_INDICATORS:
            assert spec["name"] in _CYCLICAL_PUBLICATION_LAGS, (
                f"Missing publication lag for {spec['name']}"
            )

    def test_claims_lag(self):
        assert _CYCLICAL_PUBLICATION_LAGS["claims"] == 1

    def test_nfci_lag(self):
        assert _CYCLICAL_PUBLICATION_LAGS["nfci"] == 1

    def test_biz_apps_lag(self):
        assert _CYCLICAL_PUBLICATION_LAGS["biz_apps"] == 2

    def test_jolts_lag(self):
        assert _CYCLICAL_PUBLICATION_LAGS["jolts"] == 2


class TestLoadCyclicalIndicators:
    """Test _load_cyclical_indicators with synthetic CSV data."""

    def _write_monthly_csv(self, tmpdir: Path, filename: str, col: str, n: int = 60):
        """Write a synthetic monthly CSV for testing."""
        rows = []
        for i in range(n):
            m = 1 + i
            year = 2020 + (m - 1) // 12
            month = ((m - 1) % 12) + 1
            rows.append({"ref_date": date(year, month, 1), col: 100.0 + i * 0.5})
        df = pl.DataFrame(rows)
        df.write_csv(str(tmpdir / filename))
        return df

    def _write_weekly_csv(self, tmpdir: Path, filename: str, col: str, n: int = 260):
        """Write a synthetic weekly CSV for testing."""
        import datetime

        rows = []
        start = date(2020, 1, 3)
        for i in range(n):
            d = start + datetime.timedelta(weeks=i)
            rows.append({"ref_date": d, col: 200.0 + i * 0.1})
        df = pl.DataFrame(rows)
        df.write_csv(str(tmpdir / filename))
        return df

    def test_loads_monthly_csv(self, tmp_path, monkeypatch):
        """Monthly CSV should load and center correctly."""
        self._write_monthly_csv(tmp_path, "business_applications.csv", "applications")
        monkeypatch.setattr("alt_nfp.panel_adapter.DATA_DIR", tmp_path)

        dates = [date(2020 + (i // 12), (i % 12) + 1, 1) for i in range(60)]
        result = _load_cyclical_indicators(dates, len(dates))

        assert "biz_apps_c" in result
        arr = result["biz_apps_c"]
        assert arr is not None
        assert arr.shape == (60,)
        # Centered: mean of non-zero values should be ~0
        nonzero = arr[arr != 0.0]
        if len(nonzero) > 0:
            assert abs(nonzero.mean()) < 1.0

    def test_loads_weekly_csv(self, tmp_path, monkeypatch):
        """Weekly CSV should be aggregated to monthly and centered."""
        self._write_weekly_csv(tmp_path, "claims_weekly.csv", "claims")
        monkeypatch.setattr("alt_nfp.panel_adapter.DATA_DIR", tmp_path)

        dates = [date(2020 + (i // 12), (i % 12) + 1, 1) for i in range(60)]
        result = _load_cyclical_indicators(dates, len(dates))

        assert "claims_c" in result
        arr = result["claims_c"]
        assert arr is not None
        assert arr.shape == (60,)

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        """Missing CSV should set indicator to None."""
        monkeypatch.setattr("alt_nfp.panel_adapter.DATA_DIR", tmp_path)

        dates = [date(2020 + (i // 12), (i % 12) + 1, 1) for i in range(24)]
        result = _load_cyclical_indicators(dates, len(dates))

        for spec in CYCLICAL_INDICATORS:
            key = f"{spec['name']}_c"
            assert result[key] is None

    def test_jolts_loads_correctly(self, tmp_path, monkeypatch):
        """JOLTS CSV should load as monthly data."""
        self._write_monthly_csv(tmp_path, "jolts_openings.csv", "openings")
        monkeypatch.setattr("alt_nfp.panel_adapter.DATA_DIR", tmp_path)

        dates = [date(2020 + (i // 12), (i % 12) + 1, 1) for i in range(60)]
        result = _load_cyclical_indicators(dates, len(dates))

        assert "jolts_c" in result
        arr = result["jolts_c"]
        assert arr is not None
        assert arr.shape == (60,)

    def test_all_four_indicators_load(self, tmp_path, monkeypatch):
        """All four indicators should load when all files present."""
        self._write_weekly_csv(tmp_path, "claims_weekly.csv", "claims")
        self._write_weekly_csv(tmp_path, "nfci.csv", "nfci")
        self._write_monthly_csv(tmp_path, "business_applications.csv", "applications")
        self._write_monthly_csv(tmp_path, "jolts_openings.csv", "openings")
        monkeypatch.setattr("alt_nfp.panel_adapter.DATA_DIR", tmp_path)

        dates = [date(2020 + (i // 12), (i % 12) + 1, 1) for i in range(60)]
        result = _load_cyclical_indicators(dates, len(dates))

        for spec in CYCLICAL_INDICATORS:
            key = f"{spec['name']}_c"
            assert result[key] is not None, f"{key} should be loaded"
            assert result[key].shape == (60,)


class TestCyclicalCensoring:
    """Test that as_of correctly masks cyclical indicators."""

    def test_offset_month(self):
        assert _offset_month(date(2024, 1, 1), 1) == date(2024, 2, 1)
        assert _offset_month(date(2024, 11, 1), 2) == date(2025, 1, 1)
        assert _offset_month(date(2024, 12, 1), 1) == date(2025, 1, 1)

    def test_censoring_masks_future_periods(self, tmp_path, monkeypatch):
        """With as_of set, indicators should be zeroed for periods where
        ref_date + lag > as_of."""
        from alt_nfp.ingest.base import PANEL_SCHEMA
        from alt_nfp.panel_adapter import panel_to_model_data

        # Create indicator CSVs
        for spec in CYCLICAL_INDICATORS:
            n = 60
            rows = []
            for i in range(n):
                m = 1 + i
                year = 2020 + (m - 1) // 12
                month = ((m - 1) % 12) + 1
                rows.append({"ref_date": date(year, month, 1), spec["col"]: 100.0 + i})
            if spec["freq"] == "weekly":
                import datetime

                rows = []
                start = date(2020, 1, 3)
                for i in range(260):
                    d = start + datetime.timedelta(weeks=i)
                    rows.append({"ref_date": d, spec["col"]: 200.0 + i * 0.1})
            pl.DataFrame(rows).write_csv(str(tmp_path / spec["file"]))

        monkeypatch.setattr("alt_nfp.panel_adapter.DATA_DIR", tmp_path)

        # Build a minimal panel
        base = date(2020, 1, 1)
        n_months = 36
        panel_rows = []
        for i in range(n_months):
            m = base.month + i
            year = base.year + (m - 1) // 12
            month = ((m - 1) % 12) + 1
            period = date(year, month, 1)
            ces_pub = date(year + (month // 12), (month % 12) + 1, 7)
            panel_rows.append({
                "period": period,
                "geographic_type": "national",
                "geographic_code": "US",
                "industry_code": "05",
                "industry_level": "supersector",
                "source": "ces_sa",
                "source_type": "official_sa",
                "growth": 0.001,
                "employment_level": 150_000.0,
                "is_seasonally_adjusted": True,
                "vintage_date": ces_pub,
                "revision_number": 0,
                "is_final": False,
                "publication_lag_months": 1,
                "coverage_ratio": 1.0,
            })

        panel = pl.DataFrame(panel_rows, schema=PANEL_SCHEMA)

        # With as_of in the middle
        as_of = date(2021, 6, 15)
        data = panel_to_model_data(panel, [], as_of=as_of)

        # For each indicator with lag, periods after (as_of - lag) should be zero
        for spec in CYCLICAL_INDICATORS:
            key = f"{spec['name']}_c"
            arr = data.get(key)
            if arr is None:
                continue
            lag = _CYCLICAL_PUBLICATION_LAGS.get(spec["name"], 1)
            # Find the cutoff: first date where offset_month(d, lag) > as_of
            dates = data["dates"]
            for i, d in enumerate(dates):
                if _offset_month(d, lag) > as_of:
                    # Everything from here should be 0
                    assert np.all(arr[i:] == 0.0), (
                        f"{key}: values after index {i} should be 0 with "
                        f"as_of={as_of}, lag={lag}"
                    )
                    break


class TestDataCoverage:
    """Step 1.4 — Verify data coverage for existing indicator files.

    These tests check actual data files and are skipped if files are not present.
    """

    @pytest.mark.skipif(
        not (DATA_DIR / "claims_weekly.csv").exists(),
        reason="claims_weekly.csv not available",
    )
    def test_claims_coverage(self):
        df = pl.read_csv(str(DATA_DIR / "claims_weekly.csv"), try_parse_dates=True)
        min_date = df["ref_date"].min()
        assert min_date.year <= 2003, f"Claims should start by 2003, got {min_date}"

    @pytest.mark.skipif(
        not (DATA_DIR / "nfci.csv").exists(),
        reason="nfci.csv not available",
    )
    def test_nfci_coverage(self):
        df = pl.read_csv(str(DATA_DIR / "nfci.csv"), try_parse_dates=True)
        min_date = df["ref_date"].min()
        assert min_date.year <= 2003, f"NFCI should start by 2003, got {min_date}"

    @pytest.mark.skipif(
        not (DATA_DIR / "business_applications.csv").exists(),
        reason="business_applications.csv not available",
    )
    def test_bfs_coverage(self):
        df = pl.read_csv(
            str(DATA_DIR / "business_applications.csv"), try_parse_dates=True
        )
        min_date = df["ref_date"].min()
        # BFS begins July 2004 — partial coverage is acceptable
        assert min_date.year <= 2005, f"BFS should start by 2005, got {min_date}"

    @pytest.mark.skipif(
        not (DATA_DIR / "jolts_openings.csv").exists(),
        reason="jolts_openings.csv not available",
    )
    def test_jolts_coverage(self):
        df = pl.read_csv(
            str(DATA_DIR / "jolts_openings.csv"), try_parse_dates=True
        )
        min_date = df["ref_date"].min()
        # JOLTS begins December 2000
        assert min_date.year <= 2003, f"JOLTS should start by 2003, got {min_date}"
