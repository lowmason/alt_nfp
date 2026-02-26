"""QCEW data ingestion: BLS download → PANEL_SCHEMA.

QCEW (Quarterly Census of Employment and Wages) provides near-census
employment counts with a 5-month publication lag and asymmetric revision
schedule across quarters.

This module uses the ``bls/`` download layer instead of direct urllib or
eco-stats calls.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from ..lookups.industry import get_sector_codes, qcew_to_sector
from ..lookups.revision_schedules import QCEW_REVISIONS
from .base import PANEL_SCHEMA, QCEW_VINTAGE_SCHEMA
from .bls import BLSHttpClient, fetch_qcew as bls_fetch_qcew
from .bls.qcew import QCEW_INDUSTRY_CODES, fetch_qcew_with_geography

logger = logging.getLogger(__name__)

# Canonical mapping: QCEW API code / raw NAICS code -> simplified 2-digit sector code.
# Derived from lookups/industry.py; also includes range-notation NAICS codes that
# may appear in raw API responses.
_QCEW_SECTOR_MAP: dict[str, str] = {
    **qcew_to_sector(),
    '31-33': '31',
    '44-45': '44',
    '48-49': '48',
}


def _map_industry_code(code: str) -> str | None:
    """Map a QCEW industry code to our simplified sector code."""
    return _QCEW_SECTOR_MAP.get(code)


def fetch_qcew_current(
    start_year: int,
    end_year: int,
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    """Fetch current QCEW data from the BLS QCEW CSV API via bls/ layer.

    Parameters
    ----------
    start_year : int
        First year to fetch.
    end_year : int
        Last year to fetch (inclusive).
    client : BLSHttpClient or None
        Optional HTTP client.

    Returns
    -------
    pl.DataFrame
        Observation panel rows conforming to PANEL_SCHEMA.
    """
    years = list(range(start_year, end_year + 1))

    try:
        raw = bls_fetch_qcew(years=years, client=client)
    except Exception as e:
        logger.warning(f'Failed to fetch QCEW data via bls layer: {e}')
        return _empty_panel()

    if len(raw) == 0:
        return _empty_panel()

    rows: list[dict] = []
    today = date.today()

    # Extract industry_code from raw QCEW data and map to sector codes
    if 'industry_code' not in raw.columns:
        logger.warning('QCEW response missing industry_code column')
        return _empty_panel()

    # Ensure required columns exist
    for col in ['year', 'qtr', 'month1_emplvl', 'month2_emplvl', 'month3_emplvl']:
        if col not in raw.columns:
            logger.warning(f'QCEW response missing column: {col}')
            return _empty_panel()

    # Process each row: extract monthly employment levels
    all_industry_rows: list[dict] = []
    for row in raw.iter_rows(named=True):
        ind_code = str(row['industry_code'])
        sector = _map_industry_code(ind_code)
        if sector is None:
            continue

        year = int(row['year'])
        qtr = int(row['qtr'])

        for m_idx, m_col in enumerate(
            ['month1_emplvl', 'month2_emplvl', 'month3_emplvl'],
            start=1,
        ):
            if row.get(m_col) is None:
                continue
            try:
                emp = int(row[m_col])
            except (ValueError, TypeError):
                continue
            if emp <= 0:
                continue

            month_num = (qtr - 1) * 3 + m_idx
            ref_date = date(year, month_num, 1)
            all_industry_rows.append(
                {
                    'industry_code': sector,
                    'ref_date': ref_date,
                    'employment': emp,
                    'qtr': qtr,
                }
            )

    if not all_industry_rows:
        return _empty_panel()

    ind_df = pl.DataFrame(all_industry_rows).sort('industry_code', 'ref_date')

    for sector in ind_df['industry_code'].unique().to_list():
        sec_df = ind_df.filter(pl.col('industry_code') == sector).sort('ref_date')
        emp = sec_df['employment'].to_numpy().astype(float)
        ref_dates = sec_df['ref_date'].to_list()
        qtrs = sec_df['qtr'].to_list()

        for i in range(1, len(emp)):
            if emp[i] > 0 and emp[i - 1] > 0:
                growth = float(np.log(emp[i]) - np.log(emp[i - 1]))
                q_label = f'Q{qtrs[i]}'
                pub_lag = QCEW_REVISIONS[q_label][0].lag_months
                ref_month = ref_dates[i].month
                vdate_month = ref_month + pub_lag
                vdate_year = ref_dates[i].year + (vdate_month - 1) // 12
                vdate_month = ((vdate_month - 1) % 12) + 1
                vintage_dt = date(vdate_year, vdate_month, 1)

                lag_months = (
                    (vintage_dt.year - ref_dates[i].year) * 12
                    + vintage_dt.month
                    - ref_dates[i].month
                )

                rows.append(
                    {
                        'period': ref_dates[i],
                        'geographic_type': 'national',
                        'geographic_code': 'US',
                        'industry_code': sector,
                        'industry_level': 'sector',
                        'source': 'qcew',
                        'source_type': 'census',
                        'growth': growth,
                        'employment_level': float(emp[i]),
                        'is_seasonally_adjusted': False,
                        'vintage_date': vintage_dt,
                        'revision_number': 0,
                        'is_final': False,
                        'publication_lag_months': lag_months,
                        'coverage_ratio': None,
                    }
                )

    if not rows:
        return _empty_panel()

    return pl.DataFrame(rows, schema=PANEL_SCHEMA)


def fetch_qcew_current_with_geography(
    start_year: int,
    end_year: int,
    include_national: bool = True,
    include_states: bool = True,
    state_fips_list: list[str] | None = None,
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    """Fetch current QCEW data for national and/or state geographies.

    Parameters
    ----------
    start_year : int
        First year to fetch.
    end_year : int
        Last year to fetch (inclusive).
    include_national : bool
        Include national-level rows. Defaults to ``True``.
    include_states : bool
        Include state-level rows. Defaults to ``True``.
    state_fips_list : list[str] or None
        Specific 2-digit state FIPS codes. If ``None``, uses all from
        :data:`~alt_nfp.lookups.geography.STATES`.
    client : BLSHttpClient or None
        Optional HTTP client.

    Returns
    -------
    pl.DataFrame
        Observation panel rows conforming to PANEL_SCHEMA with
        ``geographic_type`` and ``geographic_code`` populated.
    """
    years = list(range(start_year, end_year + 1))

    try:
        raw = fetch_qcew_with_geography(
            years=years,
            include_national=include_national,
            include_states=include_states,
            state_fips_list=state_fips_list,
            client=client,
        )
    except Exception as e:
        logger.warning(f'Failed to fetch QCEW geographic data: {e}')
        return _empty_panel()

    if len(raw) == 0:
        return _empty_panel()

    # Validate required columns
    required = [
        'industry_code', 'year', 'qtr',
        'month1_emplvl', 'month2_emplvl', 'month3_emplvl',
        'geographic_type', 'geographic_code',
    ]
    for col in required:
        if col not in raw.columns:
            logger.warning(f'QCEW geographic response missing column: {col}')
            return _empty_panel()

    # Extract monthly employment rows with geographic info
    all_rows: list[dict] = []
    for row in raw.iter_rows(named=True):
        ind_code = str(row['industry_code'])
        sector = _map_industry_code(ind_code)
        if sector is None:
            continue

        year = int(row['year'])
        qtr = int(row['qtr'])
        geo_type = str(row['geographic_type'])
        geo_code = str(row['geographic_code'])

        for m_idx, m_col in enumerate(
            ['month1_emplvl', 'month2_emplvl', 'month3_emplvl'],
            start=1,
        ):
            if row.get(m_col) is None:
                continue
            try:
                emp = int(row[m_col])
            except (ValueError, TypeError):
                continue
            if emp <= 0:
                continue

            month_num = (qtr - 1) * 3 + m_idx
            ref_date = date(year, month_num, 1)
            all_rows.append(
                {
                    'industry_code': sector,
                    'ref_date': ref_date,
                    'employment': emp,
                    'qtr': qtr,
                    'geographic_type': geo_type,
                    'geographic_code': geo_code,
                }
            )

    if not all_rows:
        return _empty_panel()

    ind_df = pl.DataFrame(all_rows).sort(
        'geographic_type', 'geographic_code', 'industry_code', 'ref_date'
    )

    rows: list[dict] = []
    for (geo_type, geo_code, sector), grp in ind_df.group_by(
        ['geographic_type', 'geographic_code', 'industry_code'],
        maintain_order=True,
    ):
        grp = grp.sort('ref_date')
        emp = grp['employment'].to_numpy().astype(float)
        ref_dates = grp['ref_date'].to_list()
        qtrs = grp['qtr'].to_list()

        for i in range(1, len(emp)):
            if emp[i] > 0 and emp[i - 1] > 0:
                growth = float(np.log(emp[i]) - np.log(emp[i - 1]))
                q_label = f'Q{qtrs[i]}'
                pub_lag = QCEW_REVISIONS[q_label][0].lag_months
                ref_month = ref_dates[i].month
                vdate_month = ref_month + pub_lag
                vdate_year = ref_dates[i].year + (vdate_month - 1) // 12
                vdate_month = ((vdate_month - 1) % 12) + 1
                vintage_dt = date(vdate_year, vdate_month, 1)

                lag_months = (
                    (vintage_dt.year - ref_dates[i].year) * 12
                    + vintage_dt.month
                    - ref_dates[i].month
                )

                rows.append(
                    {
                        'period': ref_dates[i],
                        'geographic_type': str(geo_type),
                        'geographic_code': str(geo_code),
                        'industry_code': str(sector),
                        'industry_level': 'sector',
                        'source': 'qcew',
                        'source_type': 'census',
                        'growth': growth,
                        'employment_level': float(emp[i]),
                        'is_seasonally_adjusted': False,
                        'vintage_date': vintage_dt,
                        'revision_number': 0,
                        'is_final': False,
                        'publication_lag_months': lag_months,
                        'coverage_ratio': None,
                    }
                )

    if not rows:
        return _empty_panel()

    return pl.DataFrame(rows, schema=PANEL_SCHEMA)


def load_qcew_vintages(path: Path) -> pl.DataFrame:
    """Load historical QCEW revision data from a parquet file.

    Parameters
    ----------
    path : Path
        Path to qcew_vintages.parquet conforming to QCEW_VINTAGE_SCHEMA.

    Returns
    -------
    pl.DataFrame
        Observation panel rows conforming to PANEL_SCHEMA.
    """
    if not path.exists():
        logger.info(f'QCEW vintage file not found: {path}')
        return _empty_panel()

    raw = pl.read_parquet(path)

    # Validate schema
    for col, dtype in QCEW_VINTAGE_SCHEMA.items():
        if col not in raw.columns:
            raise ValueError(f'QCEW vintage parquet missing column: {col}')

    # Filter to national, private
    raw = raw.filter(
        (pl.col('area_fips') == 'US000') & (pl.col('own_code') == 5)
    )

    # Build period column: date from ref_year and month within quarter
    raw = raw.with_columns(
        pl.struct(['ref_year', 'ref_quarter', 'ref_month'])
        .map_elements(
            lambda s: date(s['ref_year'], (s['ref_quarter'] - 1) * 3 + s['ref_month'], 1),
            return_dtype=pl.Date,
        )
        .alias('period')
    )

    # Sort by (industry_code, period, revision_number) for growth computation
    raw = raw.sort('industry_code', 'period', 'revision_number')

    rows: list[dict] = []
    for (ind_code, rev_num), group in raw.group_by(
        ['industry_code', 'revision_number'], maintain_order=True
    ):
        group = group.sort('period')
        emp = group['employment'].to_numpy().astype(float)
        periods = group['period'].to_list()
        vintage_dates = group['vintage_date'].to_list()
        ref_quarters = group['ref_quarter'].to_list()

        # Determine if this is the final revision for this quarter
        q_label = f'Q{ref_quarters[0]}' if ref_quarters else 'Q1'
        max_rev = max(s.revision_number for s in QCEW_REVISIONS.get(q_label, []))
        is_final = rev_num == max_rev

        for i in range(1, len(emp)):
            if emp[i] > 0 and emp[i - 1] > 0:
                growth = float(np.log(emp[i]) - np.log(emp[i - 1]))
                vdate = vintage_dates[i]
                lag = (
                    (vdate.year - periods[i].year) * 12
                    + vdate.month
                    - periods[i].month
                )

                rows.append(
                    {
                        'period': periods[i],
                        'geographic_type': 'national',
                        'geographic_code': 'US',
                        'industry_code': str(ind_code),
                        'industry_level': 'sector',
                        'source': 'qcew',
                        'source_type': 'census',
                        'growth': growth,
                        'employment_level': float(emp[i]),
                        'is_seasonally_adjusted': False,
                        'vintage_date': vdate,
                        'revision_number': int(rev_num),
                        'is_final': is_final,
                        'publication_lag_months': lag,
                        'coverage_ratio': None,
                    }
                )

    if not rows:
        return _empty_panel()

    return pl.DataFrame(rows, schema=PANEL_SCHEMA)


def ingest_qcew(
    vintage_dir: Path | None = None,
    start_year: int = 2010,
    end_year: int | None = None,
    include_states: bool = False,
    state_fips_list: list[str] | None = None,
    client: BLSHttpClient | None = None,
) -> pl.DataFrame:
    """Orchestrate QCEW data ingestion from BLS and historical vintages.

    Parameters
    ----------
    vintage_dir : Path, optional
        Directory containing qcew_vintages.parquet.
    start_year : int
        First year for API fetch (default 2010).
    end_year : int, optional
        Last year for API fetch. Defaults to current year.
    include_states : bool
        If ``True``, also fetch state-level data via
        :func:`fetch_qcew_current_with_geography`. Defaults to ``False``.
    state_fips_list : list[str] or None
        Specific 2-digit state FIPS codes. If ``None`` and
        ``include_states`` is ``True``, uses all states.
    client : BLSHttpClient or None
        Optional HTTP client for BLS downloads.

    Returns
    -------
    pl.DataFrame
        Combined and deduplicated observation panel.
    """
    if end_year is None:
        end_year = date.today().year

    parts: list[pl.DataFrame] = []

    # Load historical vintages if available
    if vintage_dir is not None:
        vintage_path = vintage_dir / 'qcew_vintages.parquet'
        vintage_df = load_qcew_vintages(vintage_path)
        if len(vintage_df) > 0:
            parts.append(vintage_df)

    # Fetch current data via bls/ download layer
    if include_states:
        try:
            current_df = fetch_qcew_current_with_geography(
                start_year,
                end_year,
                include_national=True,
                include_states=True,
                state_fips_list=state_fips_list,
                client=client,
            )
            if len(current_df) > 0:
                parts.append(current_df)
        except Exception as e:
            logger.warning(f'Failed to fetch current QCEW geographic data: {e}')
    else:
        try:
            current_df = fetch_qcew_current(start_year, end_year, client=client)
            if len(current_df) > 0:
                parts.append(current_df)
        except Exception as e:
            logger.warning(f'Failed to fetch current QCEW data: {e}')

    if not parts:
        return _empty_panel()

    combined = pl.concat(parts)

    # Deduplicate: prefer vintage data (higher revision_number) over API data
    combined = (
        combined.sort('revision_number', descending=True)
        .unique(
            subset=[
                'period', 'geographic_type', 'geographic_code',
                'source', 'industry_code', 'revision_number',
            ],
            keep='first',
        )
        .sort('period', 'industry_code', 'revision_number')
    )

    return combined


def _empty_panel() -> pl.DataFrame:
    """Return an empty DataFrame with PANEL_SCHEMA columns."""
    return pl.DataFrame(schema=PANEL_SCHEMA)
