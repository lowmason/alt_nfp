"""Build publication_calendar.parquet from historical + hard-coded dates.

Merges two sources:

1. **Historical dates** from ``release_dates.parquet`` (if available) —
   the scraper pipeline covers CES, SAE, and QCEW going back to ~2003.
2. **Forward-looking dates** from the hard-coded dicts in
   :mod:`alt_nfp.lookups.publication_dates`.

Output schema::

    source:           Utf8   ('ces', 'qcew')
    ref_period:       Utf8   ('2025-01' for monthly, '2025Q1' for quarterly)
    revision_number:  Int32  (0, 1, 2, -1 for benchmark)
    publication_date: Date

Usage::

    python scripts/build_publication_calendar.py
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl

from alt_nfp.config import DATA_DIR
from alt_nfp.ingest.release_dates.config import RELEASE_DATES_PATH
from alt_nfp.lookups.publication_dates import (
    CES_RELEASE_DATES,
    QCEW_RELEASE_DATES,
    # SAE_RELEASE_DATES,
)

log = logging.getLogger(__name__)

OUTPUT_PATH = DATA_DIR / 'publication_calendar.parquet'

SCHEMA = {
    'source': pl.Utf8,
    'ref_period': pl.Utf8,
    'revision_number': pl.Int32,
    'publication_date': pl.Date,
}


def _hardcoded_to_df() -> pl.DataFrame:
    """Convert hard-coded dicts to the output schema (revision 0 only)."""
    rows: list[dict] = []

    for ref, pub in CES_RELEASE_DATES.items():
        rows.append({
            'source': 'ces',
            'ref_period': ref.strftime('%Y-%m'),
            'revision_number': 0,
            'publication_date': pub,
        })

    # for ref, pub in SAE_RELEASE_DATES.items():
    #     rows.append({
    #         'source': 'sae',
    #         'ref_period': ref.strftime('%Y-%m'),
    #         'revision_number': 0,
    #         'publication_date': pub,
    #     })

    for (yr, q), pub in QCEW_RELEASE_DATES.items():
        rows.append({
            'source': 'qcew',
            'ref_period': f'{yr}Q{q}',
            'revision_number': 0,
            'publication_date': pub,
        })

    return pl.DataFrame(rows, schema=SCHEMA)


def _historical_to_df() -> pl.DataFrame | None:
    """Load release_dates.parquet and convert to the output schema.

    The scraper pipeline uses the 12th-of-month convention for
    ``ref_date``; this function normalises to ``YYYY-MM`` / ``YYYYQn``.

    Returns *None* if the parquet file does not exist.
    """
    if not RELEASE_DATES_PATH.exists():
        log.warning(
            'release_dates.parquet not found at %s; '
            'historical coverage will be limited to hard-coded dates.',
            RELEASE_DATES_PATH,
        )
        return None

    df = pl.read_parquet(RELEASE_DATES_PATH)
    # Columns: publication (Utf8), ref_date (Date), vintage_date (Date)

    rows: list[dict] = []
    for row in df.iter_rows(named=True):
        pub = row['publication']
        ref_d: date = row['ref_date']
        vint_d: date = row['vintage_date']

        if pub == 'ces':  # sae disabled
            ref_period = date(ref_d.year, ref_d.month, 1).strftime('%Y-%m')
        elif pub == 'qcew':
            q = (ref_d.month - 1) // 3 + 1
            ref_period = f'{ref_d.year}Q{q}'
        else:
            continue

        rows.append({
            'source': pub,
            'ref_period': ref_period,
            'revision_number': 0,
            'publication_date': vint_d,
        })

    return pl.DataFrame(rows, schema=SCHEMA)


def build() -> pl.DataFrame:
    """Build the merged publication calendar DataFrame.

    Hard-coded dates are preferred over historical dates for overlapping
    ``(source, ref_period, revision_number)`` keys.

    Returns
    -------
    pl.DataFrame
        Combined calendar in the output schema.
    """
    hardcoded = _hardcoded_to_df()
    historical = _historical_to_df()

    if historical is not None:
        # Hard-coded rows are appended last; keep='last' prefers them
        merged = pl.concat([historical, hardcoded])
        merged = merged.unique(
            subset=['source', 'ref_period', 'revision_number'],
            keep='last',
        )
    else:
        merged = hardcoded

    return merged.sort(['source', 'ref_period', 'revision_number'])


def main() -> None:
    """Build and write publication_calendar.parquet."""
    logging.basicConfig(level=logging.INFO)
    df = build()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUTPUT_PATH)
    print(f'Wrote {OUTPUT_PATH} ({df.height} rows)')


if __name__ == '__main__':
    main()
