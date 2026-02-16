"""CES data ingestion: BLS API (current) and historical vintage parquet files.

CES (Current Employment Statistics) provides monthly employment estimates
from a survey of ~670,000 establishments, with a 3-revision cycle
(first print, second, third) plus annual benchmark revision.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from ..lookups.industry import CES_SERIES_MAP, get_supersector_codes
from ..lookups.revision_schedules import CES_REVISIONS
from .base import CES_VINTAGE_SCHEMA, PANEL_SCHEMA

logger = logging.getLogger(__name__)


def fetch_ces_current(start_year: int, end_year: int) -> pl.DataFrame:
    """Fetch current CES data from the BLS API via eco-stats.

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
        from eco_stats import BLSClient
    except ImportError:
        logger.warning('eco-stats not installed; cannot fetch CES data from API')
        return _empty_panel()

    client = BLSClient()

    # Build series IDs for all supersectors + total private, SA and NSA
    ss_codes = get_supersector_codes() + ['05']
    series_ids: list[str] = []
    series_meta: list[dict] = []

    for code in ss_codes:
        for sa in [True, False]:
            sid = CES_SERIES_MAP[(code, sa)]
            series_ids.append(sid)
            series_meta.append({
                'series_id': sid,
                'supersector_code': code,
                'sa': sa,
            })

    # BLS API v2 accepts up to 50 series per request
    rows: list[dict] = []
    batch_size = 50

    for batch_start in range(0, len(series_ids), batch_size):
        batch_ids = series_ids[batch_start : batch_start + batch_size]
        batch_meta = series_meta[batch_start : batch_start + batch_size]

        try:
            result = client.get_series(
                batch_ids,
                start_year=start_year,
                end_year=end_year,
            )
        except Exception as e:
            logger.warning(f'BLS API request failed for batch starting {batch_start}: {e}')
            continue

        # Parse API response: result is typically a dict of series_id -> list of observations
        for meta in batch_meta:
            sid = meta['series_id']
            if sid not in result:
                continue

            series_data = result[sid]
            # Sort by date
            obs_list = sorted(series_data, key=lambda x: (x.get('year', 0), x.get('period', '')))

            emp_values: list[tuple[date, float]] = []
            for obs in obs_list:
                year = int(obs.get('year', 0))
                period = obs.get('period', '')
                value = obs.get('value', '')

                # BLS periods are 'M01' through 'M12'
                if not period.startswith('M') or period == 'M13':
                    continue
                month = int(period[1:])

                try:
                    emp = float(str(value).replace(',', ''))
                except (ValueError, TypeError):
                    continue

                emp_values.append((date(year, month, 1), emp))

            # Compute log growth rates
            for i in range(1, len(emp_values)):
                ref_dt, emp_cur = emp_values[i]
                _, emp_prev = emp_values[i - 1]

                if emp_cur > 0 and emp_prev > 0:
                    growth = float(np.log(emp_cur) - np.log(emp_prev))

                    source = 'ces_sa' if meta['sa'] else 'ces_nsa'
                    source_type = 'official_sa' if meta['sa'] else 'official_nsa'

                    # Current API data = latest available revision
                    today = date.today()
                    months_ago = (today.year - ref_dt.year) * 12 + today.month - ref_dt.month
                    if months_ago <= 1:
                        rev_num = 0
                    elif months_ago <= 2:
                        rev_num = 1
                    elif months_ago <= 3:
                        rev_num = 2
                    else:
                        rev_num = -1  # benchmark

                    rows.append(
                        {
                            'period': ref_dt,
                            'industry_code': meta['supersector_code'],
                            'industry_level': 'supersector',
                            'source': source,
                            'source_type': source_type,
                            'growth': growth,
                            'employment_level': emp_cur,
                            'is_seasonally_adjusted': meta['sa'],
                            'vintage_date': today,
                            'revision_number': rev_num,
                            'is_final': months_ago > 13,
                            'publication_lag_months': months_ago,
                            'coverage_ratio': None,
                        }
                    )

    if not rows:
        return _empty_panel()

    return pl.DataFrame(rows, schema=PANEL_SCHEMA)


def load_ces_vintages(path: Path) -> pl.DataFrame:
    """Load historical CES revision data from a parquet file.

    Parameters
    ----------
    path : Path
        Path to ces_vintages.parquet conforming to CES_VINTAGE_SCHEMA.

    Returns
    -------
    pl.DataFrame
        Observation panel rows conforming to PANEL_SCHEMA.
    """
    if not path.exists():
        logger.info(f'CES vintage file not found: {path}')
        return _empty_panel()

    raw = pl.read_parquet(path)

    # Validate schema
    for col in CES_VINTAGE_SCHEMA:
        if col not in raw.columns:
            raise ValueError(f'CES vintage parquet missing column: {col}')

    rows: list[dict] = []

    # Group by (supersector_code, seasonal_adjustment, revision_number) for growth computation
    for (ss_code, sa_flag, rev_num), group in raw.group_by(
        ['supersector_code', 'seasonal_adjustment', 'revision_number'],
        maintain_order=True,
    ):
        group = group.sort('ref_date')
        emp = group['employment'].to_numpy().astype(float)
        ref_dates = group['ref_date'].to_list()
        vintage_dates = group['vintage_date'].to_list()

        is_sa = sa_flag == 'S'
        source = 'ces_sa' if is_sa else 'ces_nsa'
        source_type = 'official_sa' if is_sa else 'official_nsa'

        # CES benchmark is the final revision
        is_final = rev_num == -1

        for i in range(1, len(emp)):
            if emp[i] > 0 and emp[i - 1] > 0:
                growth = float(np.log(emp[i]) - np.log(emp[i - 1]))
                vdate = vintage_dates[i]
                lag = (
                    (vdate.year - ref_dates[i].year) * 12
                    + vdate.month
                    - ref_dates[i].month
                )

                rows.append(
                    {
                        'period': ref_dates[i],
                        'industry_code': str(ss_code),
                        'industry_level': 'supersector',
                        'source': source,
                        'source_type': source_type,
                        'growth': growth,
                        'employment_level': float(emp[i]),
                        'is_seasonally_adjusted': is_sa,
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


def ingest_ces(
    api_client: object | None = None,
    vintage_dir: Path | None = None,
    start_year: int = 2010,
    end_year: int | None = None,
) -> pl.DataFrame:
    """Orchestrate CES data ingestion from API and historical vintages.

    Parameters
    ----------
    api_client : object, optional
        BLS API client (unused currently; reserved for future eco-stats integration).
    vintage_dir : Path, optional
        Directory containing ces_vintages.parquet.
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
        vintage_path = vintage_dir / 'ces_vintages.parquet'
        vintage_df = load_ces_vintages(vintage_path)
        if len(vintage_df) > 0:
            parts.append(vintage_df)

    # Fetch current data from API
    try:
        current_df = fetch_ces_current(start_year, end_year)
        if len(current_df) > 0:
            parts.append(current_df)
    except Exception as e:
        logger.warning(f'Failed to fetch current CES data: {e}')

    if not parts:
        return _empty_panel()

    combined = pl.concat(parts)

    # Deduplicate: prefer vintage data (explicit revision tracking) over API data
    combined = (
        combined.sort('revision_number', descending=True)
        .unique(subset=['period', 'source', 'industry_code', 'revision_number'], keep='first')
        .sort('period', 'industry_code', 'source', 'revision_number')
    )

    return combined


def _empty_panel() -> pl.DataFrame:
    """Return an empty DataFrame with PANEL_SCHEMA columns."""
    return pl.DataFrame(schema=PANEL_SCHEMA)
