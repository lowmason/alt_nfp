"""Combine estimate parquets into a unified releases.parquet.

Produces a levels-based, source-unified file suitable for revision analysis
and diagnostics. Unlike the growth-rate-oriented ``observation_panel.parquet``,
this file stores employment **levels** with full vintage/revision metadata.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from alt_nfp.config import DATA_DIR

RELEASES_PATH: Path = DATA_DIR / 'releases.parquet'

COMBINED_COLUMNS: list[str] = [
    'source',
    'seasonally_adjusted',
    'geographic_type',
    'geographic_code',
    'industry_type',
    'industry_code',
    'ref_date',
    'vintage_date',
    'revision',
    'benchmark_revision',
    'employment',
]

COMBINED_SCHEMA: dict[str, pl.DataType] = {
    'source': pl.Utf8,
    'seasonally_adjusted': pl.Boolean,
    'geographic_type': pl.Utf8,
    'geographic_code': pl.Utf8,
    'industry_type': pl.Utf8,
    'industry_code': pl.Utf8,
    'ref_date': pl.Date,
    'vintage_date': pl.Date,
    'revision': pl.UInt8,
    'benchmark_revision': pl.UInt8,
    'employment': pl.Float64,
}


def combine_estimates(
    *paths: Path,
    out_path: Path | None = None,
) -> pl.DataFrame:
    """Combine estimate parquets into a single releases DataFrame.

    Reads each estimate file (if present), keeps only :data:`COMBINED_COLUMNS`,
    casts to :data:`COMBINED_SCHEMA`, concatenates, and writes to ``out_path``.

    Parameters
    ----------
    *paths : Path
        Paths to estimate parquet files (QCEW, CES, SAE, etc.). Non-existent
        files are silently skipped.
    out_path : Path or None
        Output path. Defaults to ``data/releases.parquet``.

    Returns
    -------
    pl.DataFrame
        Combined DataFrame with :data:`COMBINED_SCHEMA`.
    """
    out_path = out_path or RELEASES_PATH

    frames: list[pl.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        if df.is_empty():
            continue

        # Select only columns that exist
        select_cols = [c for c in COMBINED_COLUMNS if c in df.columns]
        df = df.select(select_cols)

        # Fill missing vintage columns
        for c in COMBINED_COLUMNS:
            if c not in df.columns:
                if c == 'vintage_date':
                    df = df.with_columns(pl.lit(None).cast(pl.Date).alias(c))
                elif c in ('revision', 'benchmark_revision'):
                    df = df.with_columns(pl.lit(None).cast(pl.UInt8).alias(c))
                elif c == 'seasonally_adjusted':
                    df = df.with_columns(pl.lit(None).cast(pl.Boolean).alias(c))
                else:
                    raise ValueError(f'Missing required column {c} in {path}')

        df = df.select(COMBINED_COLUMNS).cast(COMBINED_SCHEMA)
        frames.append(df)

    if not frames:
        out_df = pl.DataFrame(schema=COMBINED_SCHEMA)
    else:
        out_df = pl.concat(frames, how='diagonal_relaxed')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path)
    return out_df
