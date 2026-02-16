"""Payroll provider data ingestion.

Loads payroll provider index files (real-time, not revised) into the
unified observation panel format.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ..config import DATA_DIR, ProviderConfig
from .base import PANEL_SCHEMA

logger = logging.getLogger(__name__)


def ingest_provider(
    config: ProviderConfig,
    raw_dir: Path | None = None,
) -> pl.DataFrame:
    """Load a single payroll provider's index data into PANEL_SCHEMA format.

    Parameters
    ----------
    config : ProviderConfig
        Provider configuration from alt_nfp.config.
    raw_dir : Path, optional
        Directory containing raw provider files. Falls back to DATA_DIR
        if not provided or if file not found in raw_dir.

    Returns
    -------
    pl.DataFrame
        Observation panel rows conforming to PANEL_SCHEMA.
    """
    # Try raw_dir first, then fall back to DATA_DIR
    fpath = None
    if raw_dir is not None:
        candidate = raw_dir / config.file
        if candidate.exists():
            fpath = candidate

    if fpath is None:
        fpath = DATA_DIR / config.file
        if not fpath.exists():
            logger.warning(f'Provider file not found: {fpath}')
            return _empty_panel()

    try:
        raw = pl.read_csv(str(fpath), try_parse_dates=True).sort('ref_date')
    except Exception as e:
        logger.warning(f'Failed to read provider file {fpath}: {e}')
        return _empty_panel()

    if config.index_col not in raw.columns:
        logger.warning(
            f'Column {config.index_col!r} not found in {fpath}. '
            f'Available: {raw.columns}'
        )
        return _empty_panel()

    # Select relevant columns
    raw = raw.select(['ref_date', config.index_col]).drop_nulls()

    if len(raw) < 2:
        return _empty_panel()

    # Compute log growth rates
    ref_dates = raw['ref_date'].to_list()
    levels = raw[config.index_col].to_numpy().astype(float)

    rows: list[dict] = []
    source_name = config.name.lower()

    for i in range(1, len(levels)):
        if levels[i] > 0 and levels[i - 1] > 0 and np.isfinite(levels[i]) and np.isfinite(levels[i - 1]):
            growth = float(np.log(levels[i]) - np.log(levels[i - 1]))

            # Approximate vintage date as ref_date + 3 weeks
            ref_dt = ref_dates[i]
            vintage_dt = ref_dt + timedelta(weeks=3)

            lag = (
                (vintage_dt.year - ref_dt.year) * 12
                + vintage_dt.month
                - ref_dt.month
            )

            rows.append(
                {
                    'period': ref_dt,
                    'industry_code': '05',  # total private (national level)
                    'industry_level': 'supersector',
                    'source': source_name,
                    'source_type': 'payroll',
                    'growth': growth,
                    'employment_level': float(levels[i]),
                    'is_seasonally_adjusted': False,
                    'vintage_date': vintage_dt,
                    'revision_number': 0,
                    'is_final': True,
                    'publication_lag_months': lag,
                    'coverage_ratio': None,
                }
            )

    if not rows:
        return _empty_panel()

    return pl.DataFrame(rows, schema=PANEL_SCHEMA)


def _empty_panel() -> pl.DataFrame:
    """Return an empty DataFrame with PANEL_SCHEMA columns."""
    return pl.DataFrame(schema=PANEL_SCHEMA)
