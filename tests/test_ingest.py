"""Tests for alt_nfp.ingest — panel validation and schema conformance."""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from alt_nfp.ingest.base import (
    CES_VINTAGE_SCHEMA,
    PANEL_SCHEMA,
    QCEW_VINTAGE_SCHEMA,
    validate_panel,
)
from alt_nfp.ingest.panel import build_panel


def _make_panel_rows(n: int = 5, **overrides) -> list[dict]:
    """Generate n valid panel rows."""
    rows = []
    base_date = date(2023, 1, 1)
    for i in range(n):
        month = base_date.month + i
        year = base_date.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        period = date(year, month, 1)

        row = {
            'period': period,
            'industry_code': '05',
            'industry_level': 'supersector',
            'source': 'ces_sa',
            'source_type': 'official_sa',
            'growth': 0.001 * (i + 1),
            'employment_level': 100000.0 + i * 100,
            'is_seasonally_adjusted': True,
            'vintage_date': date(year, month + 1 if month < 12 else 1, 1),
            'revision_number': 0,
            'is_final': False,
            'publication_lag_months': 1,
            'coverage_ratio': None,
        }
        row.update(overrides)
        rows.append(row)
    return rows


def _make_panel_df(n: int = 5, **overrides) -> pl.DataFrame:
    """Generate a valid panel DataFrame."""
    return pl.DataFrame(_make_panel_rows(n, **overrides), schema=PANEL_SCHEMA)


class TestValidatePanel:
    """Tests for validate_panel."""

    def test_validate_panel_good(self):
        """A well-formed mini DataFrame passes validation."""
        df = _make_panel_df(5)
        result = validate_panel(df)
        assert len(result) == 5

    def test_validate_panel_duplicate(self):
        """Duplicate (period, source, industry_code, revision_number) raises ValueError."""
        rows = _make_panel_rows(3)
        # Create a duplicate of the first row
        dup_row = rows[0].copy()
        rows.append(dup_row)
        df = pl.DataFrame(rows, schema=PANEL_SCHEMA)

        with pytest.raises(ValueError, match='duplicate'):
            validate_panel(df)

    def test_validate_panel_missing_column(self):
        """Missing required column raises ValueError."""
        df = _make_panel_df(3)
        df = df.drop('growth')

        with pytest.raises(ValueError, match='Missing required columns'):
            validate_panel(df)

    def test_qcew_vintage_parquet_schema(self, tmp_path):
        """A sample QCEW vintage parquet with correct schema loads correctly."""
        # Create a minimal conforming parquet
        data = {
            'ref_year': [2023, 2023, 2023],
            'ref_quarter': [1, 1, 1],
            'ref_month': [1, 2, 3],
            'area_fips': ['US000', 'US000', 'US000'],
            'industry_code': ['31', '31', '31'],
            'own_code': [5, 5, 5],
            'employment': [12000000, 12010000, 12020000],
            'revision_number': [0, 0, 0],
            'vintage_date': [date(2023, 6, 1), date(2023, 6, 1), date(2023, 6, 1)],
        }
        df = pl.DataFrame(data, schema=QCEW_VINTAGE_SCHEMA)
        parquet_path = tmp_path / 'qcew_vintages.parquet'
        df.write_parquet(parquet_path)

        # Load and verify
        from alt_nfp.ingest.qcew import load_qcew_vintages

        result = load_qcew_vintages(parquet_path)
        assert len(result) > 0
        assert set(PANEL_SCHEMA.keys()).issubset(set(result.columns))

    def test_ces_vintage_parquet_schema(self, tmp_path):
        """A sample CES vintage parquet with correct schema loads correctly."""
        data = {
            'ref_date': [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)],
            'supersector_code': ['30', '30', '30'],
            'seasonal_adjustment': ['S', 'S', 'S'],
            'employment': [12800.0, 12810.0, 12820.0],
            'revision_number': [0, 0, 0],
            'vintage_date': [date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)],
            'bls_series_id': ['CES3000000001', 'CES3000000001', 'CES3000000001'],
        }
        df = pl.DataFrame(data, schema=CES_VINTAGE_SCHEMA)
        parquet_path = tmp_path / 'ces_vintages.parquet'
        df.write_parquet(parquet_path)

        from alt_nfp.ingest.ces import load_ces_vintages

        result = load_ces_vintages(parquet_path)
        assert len(result) > 0
        assert set(PANEL_SCHEMA.keys()).issubset(set(result.columns))

    def test_legacy_panel_builds(self):
        """build_panel(use_legacy=True) produces a valid panel from existing CSVs."""
        from alt_nfp.config import DATA_DIR

        ces_path = DATA_DIR / 'ces_index.csv'
        if not ces_path.exists():
            pytest.skip('Data files not present')

        panel = build_panel(use_legacy=True)
        assert len(panel) > 0
        assert set(PANEL_SCHEMA.keys()).issubset(set(panel.columns))
