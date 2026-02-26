"""Vintage store reader, panel transformer, and write utilities.

Bridges the Hive-partitioned vintage parquet store to the model-ready
observation panel (PANEL_SCHEMA).

Functions
---------
read_vintage_store : Lazy scan with Hive partition predicate pushdown.
transform_to_panel : Vintage store → PANEL_SCHEMA (growth, metadata).
append_to_vintage_store : Append new observations to the Hive store.
compact_partition : Merge small files within a single partition.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl

from ..config import DATA_DIR
from .base import PANEL_SCHEMA

logger = logging.getLogger(__name__)

VINTAGE_STORE_PATH = DATA_DIR / "raw" / "vintages" / "vintage_store"

VINTAGE_STORE_SCHEMA: dict[str, pl.DataType] = {
    "geographic_type": pl.Utf8,
    "geographic_code": pl.Utf8,
    "industry_type": pl.Utf8,
    "industry_code": pl.Utf8,
    "ref_date": pl.Date,
    "vintage_date": pl.Date,
    "revision": pl.UInt8,
    "benchmark_revision": pl.UInt8,
    "employment": pl.Float64,
    "source": pl.Utf8,
    "seasonally_adjusted": pl.Boolean,
}

# QCEW max revision number per quarter (asymmetric schedule).
# Q1 accumulates the most revisions; Q4 the fewest.
_QCEW_MAX_REVISION: dict[int, int] = {1: 4, 2: 3, 3: 2, 4: 1}



# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read_vintage_store(
    store_path: Path = VINTAGE_STORE_PATH,
    *,
    source: str | None = None,
    seasonally_adjusted: bool | None = None,
    geographic_type: str | None = None,
    geographic_code: str | None = None,
    industry_type: str | None = None,
    industry_code: str | None = None,
    ref_date_range: tuple[date, date] | None = None,
) -> pl.LazyFrame:
    """Read from the Hive-partitioned vintage store with predicate pushdown.

    Partition-level filters (``source``, ``seasonally_adjusted``) are pushed
    to the scan so only relevant parquet files are opened.  All other filters
    are applied lazily.

    Parameters
    ----------
    store_path : Path
        Root of the Hive-partitioned parquet store.
    source : {'ces', 'qcew', 'sae'} or None
        Filter to a specific source.
    seasonally_adjusted : bool or None
        Filter by seasonal adjustment flag.
    geographic_type : {'national', 'state', 'region', 'division'} or None
        Filter by geographic type.
    geographic_code : str or None
        Filter by geographic code (e.g. ``'00'`` for national).
    industry_type : {'national', 'domain', 'supersector', 'sector'} or None
        Filter by industry type.
    industry_code : str or None
        Filter by industry code (e.g. ``'05'`` for total private).
    ref_date_range : tuple[date, date] or None
        Inclusive date range filter on ``ref_date``.

    Returns
    -------
    pl.LazyFrame
        Lazy scan of the vintage store with filters applied.
    """
    lf = pl.scan_parquet(
        store_path / "**/*.parquet",
        hive_partitioning=True,
    )

    if source is not None:
        lf = lf.filter(pl.col("source") == source)
    if seasonally_adjusted is not None:
        lf = lf.filter(pl.col("seasonally_adjusted") == seasonally_adjusted)
    if geographic_type is not None:
        lf = lf.filter(pl.col("geographic_type") == geographic_type)
    if geographic_code is not None:
        lf = lf.filter(pl.col("geographic_code") == geographic_code)
    if industry_type is not None:
        lf = lf.filter(pl.col("industry_type") == industry_type)
    if industry_code is not None:
        lf = lf.filter(pl.col("industry_code") == industry_code)
    if ref_date_range is not None:
        lf = lf.filter(
            pl.col("ref_date").is_between(ref_date_range[0], ref_date_range[1])
        )

    return lf


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


def transform_to_panel(
    lf: pl.LazyFrame,
    *,
    geographic_scope: str | None = "national",
) -> pl.DataFrame:
    """Transform vintage store data into a model-ready observation panel.

    Pipeline stages:

    1. Filter by geographic scope; drop nulls and zero employment.
    2. Derive source tags (``ces_sa``, ``ces_nsa``, ``qcew``, ``sae_sa``,
       ``sae_nsa``) and ``source_type``.
    3. Compute log-growth within each consistent revision group.
    4. Map ``revision_number``: CES/SAE observations with publication lag
       > 10 months become ``-1`` (benchmark); others keep the store's
       ``revision`` value.
    5. Determine ``is_final`` flag.
    6. Deduplicate across benchmark generations (keep latest vintage).
    7. Project to ``PANEL_SCHEMA`` columns.

    Parameters
    ----------
    lf : pl.LazyFrame
        Lazy frame from :func:`read_vintage_store`.
    geographic_scope : str or None
        If provided, filter to this ``geographic_type`` before transforming.
        Defaults to ``'national'``.  Pass ``None`` to keep all geographies
        (the caller must then handle the missing geography columns in
        ``PANEL_SCHEMA``).

    Returns
    -------
    pl.DataFrame
        Observation panel conforming to ``PANEL_SCHEMA``.
    """
    # --- 1. Filter --------------------------------------------------------
    if geographic_scope is not None:
        lf = lf.filter(pl.col("geographic_type") == geographic_scope)

    lf = lf.filter(
        pl.col("vintage_date").is_not_null()
        & pl.col("industry_code").is_not_null()
        & pl.col("geographic_code").is_not_null()
        & pl.col("employment").is_not_null()
        & (pl.col("employment") > 0)
    )

    # --- 2. Source tags ----------------------------------------------------
    lf = _derive_source_tags(lf)

    # --- 3. Log-growth within each revision group -------------------------
    # Growth must be computed within a consistent vintage: same source tag,
    # geography, industry, and (revision, benchmark_revision) pair.
    growth_group = [
        "source_tag",
        "geographic_type",
        "geographic_code",
        "industry_type",
        "industry_code",
        "revision",
        "benchmark_revision",
    ]

    lf = (
        lf.sort(*growth_group, "ref_date")
        .with_columns(
            pl.col("employment")
            .log()
            .diff()
            .over(growth_group)
            .alias("growth")
        )
        .filter(pl.col("growth").is_not_null() & pl.col("growth").is_finite())
    )

    # --- 4. revision_number + publication_lag -----------------------------
    lf = lf.with_columns(
        (
            (pl.col("vintage_date").dt.year() - pl.col("ref_date").dt.year()) * 12
            + pl.col("vintage_date").dt.month()
            - pl.col("ref_date").dt.month()
        )
        .cast(pl.Int32)
        .alias("publication_lag_months"),
    )

    lf = lf.with_columns(
        pl.when(
            pl.col("source").is_in(["ces", "sae"])
            & (pl.col("benchmark_revision") > 0)
        )
        .then(pl.lit(-1).cast(pl.Int32))
        .otherwise(pl.col("revision").cast(pl.Int32))
        .alias("revision_number"),
    )

    # --- 5. is_final ------------------------------------------------------
    # CES/SAE: final once benchmarked.
    # QCEW: final at the maximum revision for the reference quarter.
    qtr_expr = (pl.col("ref_date").dt.month() - 1) // 3 + 1

    lf = lf.with_columns(
        pl.when(pl.col("source").is_in(["ces", "sae"]))
        .then(pl.col("revision_number") == -1)
        .when((pl.col("source") == "qcew") & (qtr_expr == 1))
        .then(pl.col("revision").cast(pl.Int32) == 4)
        .when((pl.col("source") == "qcew") & (qtr_expr == 2))
        .then(pl.col("revision").cast(pl.Int32) == 3)
        .when((pl.col("source") == "qcew") & (qtr_expr == 3))
        .then(pl.col("revision").cast(pl.Int32) == 2)
        .when((pl.col("source") == "qcew") & (qtr_expr == 4))
        .then(pl.col("revision").cast(pl.Int32) == 1)
        .otherwise(pl.lit(False))
        .alias("is_final"),
    )

    # --- 6. Deduplicate across benchmark generations ----------------------
    # Multiple benchmark_revision values can map to the same revision_number
    # (e.g. CES rev=2 pre-benchmark and post-benchmark both map to 2 or -1).
    # Keep the row with the latest vintage_date.
    dedup_key = [
        "ref_date",
        "source_tag",
        "industry_code",
        "geographic_type",
        "geographic_code",
        "revision_number",
    ]
    lf = lf.sort("vintage_date", descending=True).unique(
        subset=dedup_key, keep="first"
    )

    # --- 7. Project to PANEL_SCHEMA ---------------------------------------
    result = (
        lf.select(
            pl.col("ref_date").alias("period"),
            pl.col("industry_code"),
            pl.col("industry_type").alias("industry_level"),
            pl.col("source_tag").alias("source"),
            pl.col("source_type"),
            pl.col("growth"),
            pl.col("employment").alias("employment_level"),
            pl.col("seasonally_adjusted").alias("is_seasonally_adjusted"),
            pl.col("vintage_date"),
            pl.col("revision_number"),
            pl.col("is_final"),
            pl.col("publication_lag_months"),
            pl.lit(None).cast(pl.Float64).alias("coverage_ratio"),
        )
        .sort("period", "source", "industry_code", "revision_number")
        .collect()
    )

    logger.info("Panel built from vintage store: %d rows", len(result))
    return result


def _derive_source_tags(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add ``source_tag`` and ``source_type`` columns."""
    return lf.with_columns(
        pl.when((pl.col("source") == "ces") & pl.col("seasonally_adjusted"))
        .then(pl.lit("ces_sa"))
        .when((pl.col("source") == "ces") & ~pl.col("seasonally_adjusted"))
        .then(pl.lit("ces_nsa"))
        .when(pl.col("source") == "qcew")
        .then(pl.lit("qcew"))
        .when((pl.col("source") == "sae") & pl.col("seasonally_adjusted"))
        .then(pl.lit("sae_sa"))
        .when((pl.col("source") == "sae") & ~pl.col("seasonally_adjusted"))
        .then(pl.lit("sae_nsa"))
        .otherwise(pl.col("source"))
        .alias("source_tag"),
        #
        pl.when((pl.col("source") == "ces") & pl.col("seasonally_adjusted"))
        .then(pl.lit("official_sa"))
        .when((pl.col("source") == "ces") & ~pl.col("seasonally_adjusted"))
        .then(pl.lit("official_nsa"))
        .when(pl.col("source") == "qcew")
        .then(pl.lit("census"))
        .when((pl.col("source") == "sae") & pl.col("seasonally_adjusted"))
        .then(pl.lit("official_sa"))
        .when((pl.col("source") == "sae") & ~pl.col("seasonally_adjusted"))
        .then(pl.lit("official_nsa"))
        .otherwise(pl.lit("other"))
        .alias("source_type"),
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def append_to_vintage_store(
    new_rows: pl.DataFrame,
    store_path: Path = VINTAGE_STORE_PATH,
) -> int:
    """Append new vintage observations to the Hive-partitioned store.

    Rows are partitioned by ``(source, seasonally_adjusted)`` and written as
    new parquet files.  Rows whose uniqueness key already exists in the
    partition are silently skipped.

    Parameters
    ----------
    new_rows : pl.DataFrame
        Must include all columns in :data:`VINTAGE_STORE_SCHEMA`.
    store_path : Path
        Root of the Hive-partitioned parquet store.

    Returns
    -------
    int
        Number of new rows actually appended.
    """
    for col in VINTAGE_STORE_SCHEMA:
        if col not in new_rows.columns:
            raise ValueError(f"Missing required column: {col}")

    ukey = [
        "ref_date",
        "industry_type",
        "industry_code",
        "geographic_type",
        "geographic_code",
        "revision",
        "benchmark_revision",
    ]

    total_appended = 0

    for (source, sa), partition_df in new_rows.group_by(
        ["source", "seasonally_adjusted"], maintain_order=True
    ):
        sa_str = str(sa).lower()
        partition_dir = store_path / f"source={source}" / f"seasonally_adjusted={sa_str}"

        if partition_dir.exists():
            existing = pl.read_parquet(partition_dir / "*.parquet")
            partition_df = partition_df.join(
                existing.select(ukey).unique(),
                on=ukey,
                how="anti",
            )

        if len(partition_df) == 0:
            continue

        partition_dir.mkdir(parents=True, exist_ok=True)

        vmin = partition_df["vintage_date"].min()
        vmax = partition_df["vintage_date"].max()
        fname = f"v_{vmin}_{vmax}.parquet"

        partition_df.drop(["source", "seasonally_adjusted"]).write_parquet(
            partition_dir / fname
        )
        total_appended += len(partition_df)
        logger.info("Appended %d rows to %s", len(partition_df), partition_dir / fname)

    return total_appended


def compact_partition(
    store_path: Path,
    source: str,
    seasonally_adjusted: bool,
) -> None:
    """Merge all parquet files within a single partition into one file.

    Reads every file in the partition, deduplicates on the store uniqueness
    key (keeping the latest vintage), and writes a single consolidated file.
    Original fragment files are then removed.

    Parameters
    ----------
    store_path : Path
        Root of the Hive-partitioned parquet store.
    source : str
        Source partition key (``'ces'``, ``'qcew'``, ``'sae'``).
    seasonally_adjusted : bool
        Seasonal adjustment partition key.
    """
    sa_str = str(seasonally_adjusted).lower()
    partition_dir = store_path / f"source={source}" / f"seasonally_adjusted={sa_str}"

    if not partition_dir.exists():
        logger.warning("Partition directory does not exist: %s", partition_dir)
        return

    files = sorted(partition_dir.glob("*.parquet"))
    if len(files) <= 1:
        logger.info(
            "Partition already compact (%d file): %s", len(files), partition_dir
        )
        return

    ukey = [
        "ref_date",
        "industry_type",
        "industry_code",
        "geographic_type",
        "geographic_code",
        "revision",
        "benchmark_revision",
    ]

    combined = (
        pl.read_parquet(partition_dir / "*.parquet")
        .sort("vintage_date", descending=True)
        .unique(subset=ukey, keep="first")
        .sort("ref_date", "industry_code", "revision")
    )

    compacted_path = partition_dir / "compacted.parquet"
    combined.write_parquet(compacted_path)

    for f in files:
        if f != compacted_path:
            f.unlink()

    logger.info(
        "Compacted %d files → %d rows in %s",
        len(files),
        len(combined),
        compacted_path,
    )
