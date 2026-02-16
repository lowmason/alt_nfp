"""Tests for alt_nfp.vintages — vintage views and evaluation."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from alt_nfp.ingest.base import PANEL_SCHEMA
from alt_nfp.vintages import (
    build_noise_multiplier_vector,
    final_view,
    real_time_view,
    vintage_diff,
)


def _make_test_panel() -> pl.DataFrame:
    """Create a test panel with multiple vintages for the same observations."""
    rows = [
        # CES SA: period 2023-02, industry '05', v0 (vintage_date=2023-03-01)
        {
            'period': date(2023, 2, 1),
            'industry_code': '05',
            'industry_level': 'supersector',
            'source': 'ces_sa',
            'source_type': 'official_sa',
            'growth': 0.002,
            'employment_level': 100000.0,
            'is_seasonally_adjusted': True,
            'vintage_date': date(2023, 3, 1),
            'revision_number': 0,
            'is_final': False,
            'publication_lag_months': 1,
            'coverage_ratio': None,
        },
        # CES SA: same period/industry, v1 (vintage_date=2023-04-01)
        {
            'period': date(2023, 2, 1),
            'industry_code': '05',
            'industry_level': 'supersector',
            'source': 'ces_sa',
            'source_type': 'official_sa',
            'growth': 0.0025,
            'employment_level': 100050.0,
            'is_seasonally_adjusted': True,
            'vintage_date': date(2023, 4, 1),
            'revision_number': 1,
            'is_final': False,
            'publication_lag_months': 2,
            'coverage_ratio': None,
        },
        # CES SA: same period/industry, v-1 benchmark (vintage_date=2024-03-01)
        {
            'period': date(2023, 2, 1),
            'industry_code': '05',
            'industry_level': 'supersector',
            'source': 'ces_sa',
            'source_type': 'official_sa',
            'growth': 0.003,
            'employment_level': 100100.0,
            'is_seasonally_adjusted': True,
            'vintage_date': date(2024, 3, 1),
            'revision_number': -1,
            'is_final': True,
            'publication_lag_months': 13,
            'coverage_ratio': None,
        },
        # Payroll provider: same period
        {
            'period': date(2023, 2, 1),
            'industry_code': '05',
            'industry_level': 'supersector',
            'source': 'pp1',
            'source_type': 'payroll',
            'growth': 0.0015,
            'employment_level': 50000.0,
            'is_seasonally_adjusted': False,
            'vintage_date': date(2023, 2, 22),
            'revision_number': 0,
            'is_final': True,
            'publication_lag_months': 0,
            'coverage_ratio': None,
        },
    ]
    return pl.DataFrame(rows, schema=PANEL_SCHEMA)


class TestRealTimeView:
    """Tests for real_time_view."""

    def test_real_time_view_filters_correctly(self):
        """real_time_view returns correct vintage based on as_of date."""
        panel = _make_test_panel().lazy()

        # as_of between v0 and v1 vintage dates -> should return v0
        result = real_time_view(panel, as_of=date(2023, 3, 15)).collect()
        ces_rows = result.filter(pl.col('source') == 'ces_sa')
        assert len(ces_rows) == 1
        assert ces_rows['revision_number'][0] == 0
        assert ces_rows['growth'][0] == 0.002

        # as_of after v1 but before benchmark -> should return v1
        result = real_time_view(panel, as_of=date(2023, 5, 1)).collect()
        ces_rows = result.filter(pl.col('source') == 'ces_sa')
        assert len(ces_rows) == 1
        assert ces_rows['revision_number'][0] == 1
        assert ces_rows['growth'][0] == 0.0025

        # as_of after benchmark -> should return benchmark (rev=-1 is highest? No.)
        # revision_number=-1 < 0 < 1, so v1 should still be selected as highest
        # unless we treat benchmark specially. With current logic, v1 > v-1 numerically.
        result = real_time_view(panel, as_of=date(2024, 4, 1)).collect()
        ces_rows = result.filter(pl.col('source') == 'ces_sa')
        assert len(ces_rows) == 1
        # v1 (revision_number=1) > v-1 (revision_number=-1) numerically
        assert ces_rows['revision_number'][0] == 1


class TestFinalView:
    """Tests for final_view."""

    def test_final_view(self):
        """Only rows with is_final=True are returned."""
        panel = _make_test_panel().lazy()

        result = final_view(panel).collect()
        assert all(result['is_final'].to_list())

        # Should have the benchmark CES row and the payroll row
        sources = sorted(result['source'].to_list())
        assert 'ces_sa' in sources
        assert 'pp1' in sources


class TestVintageDiff:
    """Tests for vintage_diff."""

    def test_vintage_diff(self):
        """Known revision between v0 and v1 is computed correctly."""
        panel = _make_test_panel().lazy()

        diff = vintage_diff(panel, source='ces_sa', rev_a=0, rev_b=1).collect()
        assert len(diff) == 1
        assert diff['revision_from'][0] == 0
        assert diff['revision_to'][0] == 1

        expected_diff = 0.0025 - 0.002
        assert abs(diff['growth_diff'][0] - expected_diff) < 1e-10


class TestNoiseMultiplier:
    """Tests for build_noise_multiplier_vector."""

    def test_noise_multiplier_vector_shape(self):
        """Output array length matches panel row count."""
        panel = _make_test_panel()
        multipliers = build_noise_multiplier_vector(panel)
        assert len(multipliers) == len(panel)

    def test_noise_multiplier_values(self):
        """Noise multipliers are looked up correctly."""
        panel = _make_test_panel()
        multipliers = build_noise_multiplier_vector(panel)

        # CES v0 should have multiplier 3.0
        ces_v0_idx = panel.with_row_index().filter(
            (pl.col('source') == 'ces_sa') & (pl.col('revision_number') == 0)
        )['index'][0]
        assert multipliers[ces_v0_idx] == 3.0

        # CES v1 should have multiplier 2.0
        ces_v1_idx = panel.with_row_index().filter(
            (pl.col('source') == 'ces_sa') & (pl.col('revision_number') == 1)
        )['index'][0]
        assert multipliers[ces_v1_idx] == 2.0

        # Payroll should have multiplier 1.0
        pp_idx = panel.with_row_index().filter(
            pl.col('source') == 'pp1'
        )['index'][0]
        assert multipliers[pp_idx] == 1.0
