"""QCEW data ingestion: BLS API (current) and historical vintage parquet files.

QCEW (Quarterly Census of Employment and Wages) provides near-census
employment counts with a 5-month publication lag and asymmetric revision
schedule across quarters.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from ..lookups.industry import INDUSTRY_HIERARCHY, get_sector_codes
from ..lookups.revision_schedules import QCEW_REVISIONS
from .base import PANEL_SCHEMA, QCEW_VINTAGE_SCHEMA

logger = logging.getLogger(__name__)

# NAICS codes that map to simplified sector codes
_NAICS_TO_SECTOR: dict[str, str] = {
    '10': '21',  # NAICS 21 = Mining
    '1012': '21',
    '21': '21',
    '23': '23',
    '31-33': '31',
    '31': '31',
    '42': '42',
    '44-45': '44',
    '44': '44',
    '48-49': '48',
    '48': '48',
    '22': '22',
    '51': '51',
    '52': '52',
    '53': '53',
    '54': '54',
    '55': '55',
    '56': '56',
    '61': '61',
    '62': '62',
    '71': '71',
    '72': '72',
    '81': '81',
}


def _normalize_naics(code: str) -> str | None:
    """Map a NAICS industry code to a simplified sector code.

    Returns None if the code doesn't map to a known sector.
    """
    if code in _NAICS_TO_SECTOR:
        return _NAICS_TO_SECTOR[code]
    return None


def fetch_qcew_current(start_year: int, end_year: int) -> pl.DataFrame:
    """Fetch current QCEW data from the BLS QCEW open data API.

    Parameters
    ----------
    start_year : int
        First year to fetch.
    end_year : int
        Last year to fetch (inclusive).

    Returns
    -------
    pl.DataFrame
        Observation panel rows conforming to PANEL_SCHEMA.
    """
    try:
        from eco_stats import BLSClient  # noqa: F401
    except ImportError:
        logger.warning('eco-stats not installed; cannot fetch QCEW data from API')
        return _empty_panel()

    rows: list[dict] = []
    valid_sectors = set(get_sector_codes())
    today = date.today()

    for year in range(start_year, end_year + 1):
        for qtr in range(1, 5):
            # QCEW API endpoint for quarterly data by industry
            url = f'https://data.bls.gov/cew/data/api/{year}/{qtr}/industry/10.csv'
            try:
                import urllib.request

                # Fetch all 2-digit NAICS industries for this quarter
                all_industry_rows: list[dict] = []
                for sector in valid_sectors:
                    naics = sector
                    # BLS QCEW uses NAICS codes; map our simplified codes
                    if sector == '31':
                        naics = '31-33'
                    elif sector == '44':
                        naics = '44-45'
                    elif sector == '48':
                        naics = '48-49'

                    qcew_url = (
                        f'https://data.bls.gov/cew/data/api/{year}/{qtr}'
                        f'/industry/{naics}.csv'
                    )
                    try:
                        with urllib.request.urlopen(qcew_url, timeout=30) as resp:
                            csv_bytes = resp.read()
                        df = pl.read_csv(csv_bytes)

                        # Filter: national, private ownership
                        df = df.filter(
                            (pl.col('area_fips') == 'US000')
                            & (pl.col('own_code') == 5)
                        )

                        if len(df) == 0:
                            continue

                        for row in df.iter_rows(named=True):
                            for m_idx, m_col in enumerate(
                                ['month1_emplvl', 'month2_emplvl', 'month3_emplvl'],
                                start=1,
                            ):
                                if m_col not in row or row[m_col] is None:
                                    continue
                                month_num = (qtr - 1) * 3 + m_idx
                                ref_date = date(year, month_num, 1)
                                all_industry_rows.append(
                                    {
                                        'industry_code': sector,
                                        'ref_date': ref_date,
                                        'employment': int(row[m_col]),
                                    }
                                )
                    except Exception as e:
                        logger.debug(f'Failed to fetch QCEW {naics} {year}Q{qtr}: {e}')
                        continue

                if not all_industry_rows:
                    continue

                # Compute growth rates: sort by (industry, date), then log-diff
                ind_df = pl.DataFrame(all_industry_rows).sort('industry_code', 'ref_date')

                for sector in ind_df['industry_code'].unique().to_list():
                    sec_df = ind_df.filter(pl.col('industry_code') == sector).sort('ref_date')
                    emp = sec_df['employment'].to_numpy().astype(float)
                    ref_dates = sec_df['ref_date'].to_list()

                    for i in range(1, len(emp)):
                        if emp[i] > 0 and emp[i - 1] > 0:
                            growth = float(np.log(emp[i]) - np.log(emp[i - 1]))
                            # Approximate vintage date from revision schedule
                            q_label = f'Q{qtr}'
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
            except Exception as e:
                logger.warning(f'Failed to fetch QCEW {year}Q{qtr}: {e}')
                continue

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
    api_client: object | None = None,
    vintage_dir: Path | None = None,
    start_year: int = 2010,
    end_year: int | None = None,
) -> pl.DataFrame:
    """Orchestrate QCEW data ingestion from API and historical vintages.

    Parameters
    ----------
    api_client : object, optional
        BLS API client (unused currently; reserved for future eco-stats integration).
    vintage_dir : Path, optional
        Directory containing qcew_vintages.parquet.
    start_year : int
        First year for API fetch (default 2010).
    end_year : int, optional
        Last year for API fetch. Defaults to current year.

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

    # Fetch current data from API
    try:
        current_df = fetch_qcew_current(start_year, end_year)
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
        .unique(subset=['period', 'source', 'industry_code', 'revision_number'], keep='first')
        .sort('period', 'industry_code', 'revision_number')
    )

    return combined


def _empty_panel() -> pl.DataFrame:
    """Return an empty DataFrame with PANEL_SCHEMA columns."""
    return pl.DataFrame(schema=PANEL_SCHEMA)
