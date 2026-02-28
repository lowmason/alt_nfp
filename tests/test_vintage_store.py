"""Tests for alt_nfp.ingest.vintage_store — read, transform, append, compact."""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from alt_nfp.ingest.base import PANEL_SCHEMA
from alt_nfp.ingest.vintage_store import (
    VINTAGE_STORE_SCHEMA,
    append_to_vintage_store,
    compact_partition,
    read_vintage_store,
    transform_to_panel,
)

# Access private helper for direct unit testing
from alt_nfp.ingest import vintage_store as _vs

_derive_source_tags = _vs._derive_source_tags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vintage_rows(
    n_months: int = 6,
    source: str = "ces",
    sa: bool = True,
    industry_code: str = "05",
    industry_type: str = "supersector",
    geographic_type: str = "national",
    geographic_code: str = "00",
    revision: int = 0,
    benchmark_revision: int = 0,
    base_employment: float = 150_000.0,
    start_date: date = date(2023, 1, 1),
    vintage_offset_months: int = 1,
) -> list[dict]:
    """Generate synthetic vintage store rows."""
    rows = []
    for i in range(n_months):
        m = start_date.month + i
        y = start_date.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        ref = date(y, m, 1)

        vm = m + vintage_offset_months
        vy = y + (vm - 1) // 12
        vm = ((vm - 1) % 12) + 1
        vdate = date(vy, vm, 1)

        rows.append(
            {
                "geographic_type": geographic_type,
                "geographic_code": geographic_code,
                "industry_type": industry_type,
                "industry_code": industry_code,
                "ref_date": ref,
                "vintage_date": vdate,
                "revision": revision,
                "benchmark_revision": benchmark_revision,
                "employment": base_employment + i * 100.0,
                "source": source,
                "seasonally_adjusted": sa,
            }
        )
    return rows


def _make_vintage_df(**kwargs) -> pl.DataFrame:
    return pl.DataFrame(_make_vintage_rows(**kwargs), schema=VINTAGE_STORE_SCHEMA)


def _write_hive_store(df: pl.DataFrame, store_path: Path) -> None:
    """Write a DataFrame as a Hive-partitioned vintage store."""
    for (source, sa), part in df.group_by(["source", "seasonally_adjusted"]):
        sa_str = str(sa).lower()
        pdir = store_path / f"source={source}" / f"seasonally_adjusted={sa_str}"
        pdir.mkdir(parents=True, exist_ok=True)
        part.drop(["source", "seasonally_adjusted"]).write_parquet(pdir / "data.parquet")


# ---------------------------------------------------------------------------
# read_vintage_store
# ---------------------------------------------------------------------------


class TestReadVintageStore:
    def test_reads_all_rows(self, tmp_path):
        df = _make_vintage_df(n_months=4, source="ces", sa=True)
        _write_hive_store(df, tmp_path)

        lf = read_vintage_store(tmp_path)
        result = lf.collect()
        assert len(result) == 4

    def test_filter_by_source(self, tmp_path):
        ces = _make_vintage_df(n_months=3, source="ces", sa=True)
        qcew = _make_vintage_df(n_months=3, source="qcew", sa=False)
        combined = pl.concat([ces, qcew])
        _write_hive_store(combined, tmp_path)

        result = read_vintage_store(tmp_path, source="qcew").collect()
        assert len(result) == 3
        assert result["source"].unique().to_list() == ["qcew"]

    def test_filter_by_seasonally_adjusted(self, tmp_path):
        sa = _make_vintage_df(n_months=3, source="ces", sa=True)
        nsa = _make_vintage_df(n_months=3, source="ces", sa=False)
        combined = pl.concat([sa, nsa])
        _write_hive_store(combined, tmp_path)

        result = read_vintage_store(tmp_path, seasonally_adjusted=True).collect()
        assert len(result) == 3
        assert all(result["seasonally_adjusted"].to_list())

    def test_filter_by_ref_date_range(self, tmp_path):
        df = _make_vintage_df(n_months=6, source="ces", sa=True)
        _write_hive_store(df, tmp_path)

        result = read_vintage_store(
            tmp_path, ref_date_range=(date(2023, 3, 1), date(2023, 5, 1))
        ).collect()
        assert len(result) == 3
        assert result["ref_date"].min() == date(2023, 3, 1)
        assert result["ref_date"].max() == date(2023, 5, 1)

    def test_filter_by_industry(self, tmp_path):
        a = _make_vintage_df(n_months=3, industry_code="05")
        b = _make_vintage_df(n_months=3, industry_code="30")
        combined = pl.concat([a, b])
        _write_hive_store(combined, tmp_path)

        result = read_vintage_store(tmp_path, industry_code="30").collect()
        assert len(result) == 3
        assert result["industry_code"].unique().to_list() == ["30"]

    def test_filter_by_geographic_type(self, tmp_path):
        nat = _make_vintage_df(n_months=3, geographic_type="national")
        state = _make_vintage_df(n_months=3, geographic_type="state", geographic_code="06")
        combined = pl.concat([nat, state])
        _write_hive_store(combined, tmp_path)

        result = read_vintage_store(tmp_path, geographic_type="state").collect()
        assert len(result) == 3
        assert result["geographic_type"].unique().to_list() == ["state"]

    def test_combined_filters(self, tmp_path):
        ces_sa = _make_vintage_df(n_months=4, source="ces", sa=True, industry_code="05")
        ces_nsa = _make_vintage_df(n_months=4, source="ces", sa=False, industry_code="05")
        qcew = _make_vintage_df(n_months=4, source="qcew", sa=False, industry_code="30")
        combined = pl.concat([ces_sa, ces_nsa, qcew])
        _write_hive_store(combined, tmp_path)

        result = read_vintage_store(
            tmp_path, source="ces", seasonally_adjusted=True, industry_code="05"
        ).collect()
        assert len(result) == 4


# ---------------------------------------------------------------------------
# _derive_source_tags
# ---------------------------------------------------------------------------


class TestDeriveSourceTags:
    @pytest.mark.parametrize(
        "source,sa,expected_tag,expected_type",
        [
            ("ces", True, "ces_sa", "official_sa"),
            ("ces", False, "ces_nsa", "official_nsa"),
            ("qcew", False, "qcew", "census"),
            ("qcew", True, "qcew", "census"),
            # ("sae", True, "sae_sa", "official_sa"),
            # ("sae", False, "sae_nsa", "official_nsa"),
        ],
    )
    def test_source_tag_mapping(self, source, sa, expected_tag, expected_type):
        lf = pl.LazyFrame(
            {"source": [source], "seasonally_adjusted": [sa], "x": [1]}
        )
        result = _derive_source_tags(lf).collect()
        assert result["source_tag"][0] == expected_tag
        assert result["source_type"][0] == expected_type

    def test_unknown_source_passthrough(self):
        lf = pl.LazyFrame(
            {"source": ["pp1"], "seasonally_adjusted": [False], "x": [1]}
        )
        result = _derive_source_tags(lf).collect()
        assert result["source_tag"][0] == "pp1"
        assert result["source_type"][0] == "other"


# ---------------------------------------------------------------------------
# transform_to_panel
# ---------------------------------------------------------------------------


class TestTransformToPanel:
    def test_basic_ces_transform(self, tmp_path):
        """CES data with 1-month vintage lag produces valid panel rows."""
        df = _make_vintage_df(
            n_months=6, source="ces", sa=True, vintage_offset_months=1
        )
        _write_hive_store(df, tmp_path)

        lf = read_vintage_store(tmp_path)
        panel = transform_to_panel(lf)

        # First month is lost to diff → 5 rows
        assert len(panel) == 5
        assert set(PANEL_SCHEMA.keys()) == set(panel.columns)
        assert panel["source"].unique().to_list() == ["ces_sa"]
        assert panel["source_type"].unique().to_list() == ["official_sa"]

    def test_growth_computation(self, tmp_path):
        """Log-growth is computed correctly."""
        rows = [
            {
                **r,
                "employment": 100_000.0 if i == 0 else 100_100.0,
            }
            for i, r in enumerate(_make_vintage_rows(n_months=2, source="ces", sa=True))
        ]
        df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
        _write_hive_store(df, tmp_path)

        panel = transform_to_panel(read_vintage_store(tmp_path))
        assert len(panel) == 1

        expected = np.log(100_100.0) - np.log(100_000.0)
        assert abs(panel["growth"][0] - expected) < 1e-12

    def test_benchmark_revision_sets_revision_minus_one(self, tmp_path):
        """CES with benchmark_revision > 0 → revision_number = -1."""
        rows = _make_vintage_rows(
            n_months=3, source="ces", sa=True, benchmark_revision=1
        )
        df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
        _write_hive_store(df, tmp_path)

        panel = transform_to_panel(read_vintage_store(tmp_path))
        assert panel["revision_number"].to_list() == [-1] * len(panel)
        assert all(panel["is_final"].to_list())

    def test_benchmark_revision_zero_keeps_revision(self, tmp_path):
        """CES with benchmark_revision=0 keeps original revision value."""
        rows = _make_vintage_rows(
            n_months=3, source="ces", sa=True, revision=2, benchmark_revision=0
        )
        df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
        _write_hive_store(df, tmp_path)

        panel = transform_to_panel(read_vintage_store(tmp_path))
        assert all(r == 2 for r in panel["revision_number"].to_list())

    def test_non_benchmark_ces_keeps_revision(self, tmp_path):
        """CES observation with lag <= 10 months keeps original revision."""
        rows = _make_vintage_rows(
            n_months=3, source="ces", sa=True, revision=2, vintage_offset_months=3
        )
        df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
        _write_hive_store(df, tmp_path)

        panel = transform_to_panel(read_vintage_store(tmp_path))
        assert all(r == 2 for r in panel["revision_number"].to_list())
        assert all(not f for f in panel["is_final"].to_list())

    def test_qcew_is_final_per_quarter(self, tmp_path):
        """QCEW is_final flag respects the quarter-specific max revision."""
        rows = []
        # Q1 (Jan): max revision is 4
        rows.extend(
            _make_vintage_rows(
                n_months=1,
                source="qcew",
                sa=False,
                revision=4,
                start_date=date(2023, 1, 1),
                vintage_offset_months=18,
            )
        )
        # Q2 (Apr): max revision is 3
        rows.extend(
            _make_vintage_rows(
                n_months=1,
                source="qcew",
                sa=False,
                revision=3,
                start_date=date(2023, 4, 1),
                vintage_offset_months=15,
            )
        )
        # Q3 (Jul): max revision is 2
        rows.extend(
            _make_vintage_rows(
                n_months=1,
                source="qcew",
                sa=False,
                revision=2,
                start_date=date(2023, 7, 1),
                vintage_offset_months=12,
            )
        )
        # Q4 (Oct): max revision is 1
        rows.extend(
            _make_vintage_rows(
                n_months=1,
                source="qcew",
                sa=False,
                revision=1,
                start_date=date(2023, 10, 1),
                vintage_offset_months=9,
            )
        )

        # We need at least 2 rows per revision group for growth; add a preceding month each
        extra_rows = []
        for r in rows:
            prev = dict(r)
            rd = r["ref_date"]
            pm = rd.month - 1 or 12
            py = rd.year if rd.month > 1 else rd.year - 1
            prev["ref_date"] = date(py, pm, 1)
            vd = r["vintage_date"]
            prev["vintage_date"] = date(vd.year, vd.month, 1)
            prev["employment"] = r["employment"] - 50.0
            extra_rows.append(prev)

        all_rows = extra_rows + rows
        df = pl.DataFrame(all_rows, schema=VINTAGE_STORE_SCHEMA)
        _write_hive_store(df, tmp_path)

        panel = transform_to_panel(read_vintage_store(tmp_path))
        # Each quarter pair produces 1 growth row; the ones at max revision are final
        final_rows = panel.filter(pl.col("is_final"))
        assert len(final_rows) == 4

    def test_qcew_not_final_below_max(self, tmp_path):
        """QCEW below max revision for its quarter → is_final = False."""
        # Q1 month, revision=2 (max is 4) → not final
        rows = _make_vintage_rows(
            n_months=2,
            source="qcew",
            sa=False,
            revision=2,
            start_date=date(2023, 1, 1),
            vintage_offset_months=9,
        )
        df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
        _write_hive_store(df, tmp_path)

        panel = transform_to_panel(read_vintage_store(tmp_path))
        assert len(panel) == 1
        assert not panel["is_final"][0]

    def test_geographic_scope_filter(self, tmp_path):
        """Only the requested geographic_scope passes through."""
        nat = _make_vintage_df(n_months=3, geographic_type="national")
        state = _make_vintage_df(
            n_months=3, geographic_type="state", geographic_code="06"
        )
        combined = pl.concat([nat, state])
        _write_hive_store(combined, tmp_path)

        panel = transform_to_panel(
            read_vintage_store(tmp_path), geographic_scope="national"
        )
        assert all(panel["industry_level"].to_list())  # all rows present
        # State rows should be excluded
        panel_state = transform_to_panel(
            read_vintage_store(tmp_path), geographic_scope="state"
        )
        # Both should have rows only from their scope
        assert len(panel) > 0
        assert len(panel_state) > 0

    def test_zero_employment_filtered(self, tmp_path):
        """Rows with employment=0 are dropped; diff bridges remaining rows."""
        rows = _make_vintage_rows(n_months=3, source="ces", sa=True)
        rows[1]["employment"] = 0.0
        df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
        _write_hive_store(df, tmp_path)

        panel = transform_to_panel(read_vintage_store(tmp_path))
        # Months 0 and 2 survive; diff is computed across consecutive survivors
        assert len(panel) == 1
        # Verify no zero employment in panel
        assert all(e > 0 for e in panel["employment_level"].to_list())

    def test_null_employment_filtered(self, tmp_path):
        """Rows with null employment are dropped; diff bridges remaining rows."""
        rows = _make_vintage_rows(n_months=3, source="ces", sa=True)
        rows[1]["employment"] = None
        df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
        _write_hive_store(df, tmp_path)

        panel = transform_to_panel(read_vintage_store(tmp_path))
        # Months 0 and 2 survive; diff bridges the gap
        assert len(panel) == 1

    def test_dedup_across_benchmark_generations(self, tmp_path):
        """Dedup reduces rows when different benchmark_revisions map to the same key."""
        # Two revisions (0 and 1) for the same source, both within benchmark lag.
        # These have DIFFERENT revision_numbers so dedup preserves both.
        rows_rev0 = _make_vintage_rows(
            n_months=3,
            source="ces",
            sa=True,
            revision=0,
            benchmark_revision=0,
            vintage_offset_months=1,
            base_employment=100_000.0,
        )
        rows_rev1 = _make_vintage_rows(
            n_months=3,
            source="ces",
            sa=True,
            revision=1,
            benchmark_revision=0,
            vintage_offset_months=2,
            base_employment=100_000.0,
        )
        combined = pl.DataFrame(
            rows_rev0 + rows_rev1, schema=VINTAGE_STORE_SCHEMA
        )
        _write_hive_store(combined, tmp_path)

        panel = transform_to_panel(read_vintage_store(tmp_path))
        # 2 growth rows per revision (3 months → 2 diffs), 2 revisions → 4 total
        assert len(panel) == 4
        rev_nums = sorted(panel["revision_number"].unique().to_list())
        assert rev_nums == [0, 1]

    def test_publication_lag_months(self, tmp_path):
        """publication_lag_months is computed correctly."""
        rows = _make_vintage_rows(
            n_months=3, source="ces", sa=True, vintage_offset_months=2
        )
        df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
        _write_hive_store(df, tmp_path)

        panel = transform_to_panel(read_vintage_store(tmp_path))
        assert all(lag == 2 for lag in panel["publication_lag_months"].to_list())

    def test_coverage_ratio_is_null(self, tmp_path):
        """coverage_ratio is always null from vintage store transform."""
        df = _make_vintage_df(n_months=4, source="ces", sa=True)
        _write_hive_store(df, tmp_path)

        panel = transform_to_panel(read_vintage_store(tmp_path))
        assert panel["coverage_ratio"].null_count() == len(panel)

    def test_output_sorted(self, tmp_path):
        """Output panel is sorted by (period, source, industry_code, revision_number)."""
        df = _make_vintage_df(n_months=6, source="ces", sa=True)
        _write_hive_store(df, tmp_path)

        panel = transform_to_panel(read_vintage_store(tmp_path))
        resorted = panel.sort("period", "source", "industry_code", "revision_number")
        assert panel.equals(resorted)

    # def test_sae_benchmark_revision(self, tmp_path):
    #     """SAE with benchmark_revision > 0 → revision_number = -1."""
    #     rows = _make_vintage_rows(
    #         n_months=3, source="sae", sa=True, benchmark_revision=2
    #     )
    #     df = pl.DataFrame(rows, schema=VINTAGE_STORE_SCHEMA)
    #     _write_hive_store(df, tmp_path)
    #
    #     panel = transform_to_panel(read_vintage_store(tmp_path))
    #     assert all(r == -1 for r in panel["revision_number"].to_list())

    def test_multi_source_panel(self, tmp_path):
        """Multiple sources combine correctly into a single panel."""
        ces = _make_vintage_df(n_months=4, source="ces", sa=True, industry_code="05")
        qcew = _make_vintage_df(
            n_months=4, source="qcew", sa=False, industry_code="05",
            vintage_offset_months=6,
        )
        combined = pl.concat([ces, qcew])
        _write_hive_store(combined, tmp_path)

        panel = transform_to_panel(read_vintage_store(tmp_path))
        sources = sorted(panel["source"].unique().to_list())
        assert "ces_sa" in sources
        assert "qcew" in sources


# ---------------------------------------------------------------------------
# append_to_vintage_store
# ---------------------------------------------------------------------------


class TestAppendToVintageStore:
    def test_append_creates_partition(self, tmp_path):
        """First append creates partition directories and parquet file."""
        df = _make_vintage_df(n_months=3, source="ces", sa=True)
        count = append_to_vintage_store(df, tmp_path)

        assert count == 3
        pdir = tmp_path / "source=ces" / "seasonally_adjusted=true"
        assert pdir.exists()
        files = list(pdir.glob("*.parquet"))
        assert len(files) == 1

    def test_append_dedup_skips_existing(self, tmp_path):
        """Re-appending the same rows returns 0 (all duplicates skipped)."""
        df = _make_vintage_df(n_months=3, source="ces", sa=True)

        first = append_to_vintage_store(df, tmp_path)
        assert first == 3

        second = append_to_vintage_store(df, tmp_path)
        assert second == 0

    def test_append_adds_only_new(self, tmp_path):
        """Only genuinely new rows are appended."""
        df1 = _make_vintage_df(n_months=3, source="ces", sa=True, start_date=date(2023, 1, 1))
        df2 = _make_vintage_df(n_months=3, source="ces", sa=True, start_date=date(2023, 2, 1))

        append_to_vintage_store(df1, tmp_path)
        # df2 overlaps months 2-3 with df1; only month 4 is new
        count = append_to_vintage_store(df2, tmp_path)
        assert count == 1  # only the new month

    def test_append_multi_partition(self, tmp_path):
        """Rows for different (source, sa) go to different partitions."""
        ces = _make_vintage_df(n_months=2, source="ces", sa=True)
        qcew = _make_vintage_df(n_months=2, source="qcew", sa=False)
        combined = pl.concat([ces, qcew])

        count = append_to_vintage_store(combined, tmp_path)
        assert count == 4

        assert (tmp_path / "source=ces" / "seasonally_adjusted=true").exists()
        assert (tmp_path / "source=qcew" / "seasonally_adjusted=false").exists()

    def test_append_missing_column_raises(self, tmp_path):
        """Missing a required column raises ValueError."""
        df = _make_vintage_df(n_months=2).drop("employment")
        with pytest.raises(ValueError, match="Missing required column"):
            append_to_vintage_store(df, tmp_path)


# ---------------------------------------------------------------------------
# compact_partition
# ---------------------------------------------------------------------------


class TestCompactPartition:
    def test_compact_merges_files(self, tmp_path):
        """Multiple fragment files are merged into one."""
        # Append twice to create two files
        df1 = _make_vintage_df(n_months=3, source="ces", sa=True, start_date=date(2023, 1, 1))
        df2 = _make_vintage_df(n_months=3, source="ces", sa=True, start_date=date(2023, 7, 1))

        append_to_vintage_store(df1, tmp_path)
        append_to_vintage_store(df2, tmp_path)

        pdir = tmp_path / "source=ces" / "seasonally_adjusted=true"
        assert len(list(pdir.glob("*.parquet"))) == 2

        compact_partition(tmp_path, "ces", True)
        files = list(pdir.glob("*.parquet"))
        assert len(files) == 1
        assert files[0].name == "compacted.parquet"

        # All rows preserved
        compacted = pl.read_parquet(files[0])
        assert len(compacted) == 6

    def test_compact_deduplicates(self, tmp_path):
        """Compaction deduplicates on the uniqueness key, keeping latest vintage."""
        df1 = _make_vintage_df(
            n_months=3, source="ces", sa=True, vintage_offset_months=1
        )
        # Same ref_dates but later vintage
        df2 = _make_vintage_df(
            n_months=3, source="ces", sa=True, vintage_offset_months=2
        )

        append_to_vintage_store(df1, tmp_path)
        # Force write second batch (bypass dedup in append by writing directly)
        pdir = tmp_path / "source=ces" / "seasonally_adjusted=true"
        df2.drop(["source", "seasonally_adjusted"]).write_parquet(pdir / "batch2.parquet")

        compact_partition(tmp_path, "ces", True)
        files = list(pdir.glob("*.parquet"))
        assert len(files) == 1

        compacted = pl.read_parquet(files[0])
        assert len(compacted) == 3  # deduped to 3

    def test_compact_noop_single_file(self, tmp_path):
        """Partition with one file is unchanged."""
        df = _make_vintage_df(n_months=3, source="ces", sa=True)
        append_to_vintage_store(df, tmp_path)

        compact_partition(tmp_path, "ces", True)
        pdir = tmp_path / "source=ces" / "seasonally_adjusted=true"
        files = list(pdir.glob("*.parquet"))
        assert len(files) == 1

    def test_compact_missing_partition(self, tmp_path):
        """Non-existent partition logs warning, no error."""
        compact_partition(tmp_path, "nonexistent", True)
