"""Tests for alt_nfp.ingest.release_dates — parser and vintage_dates."""

import tempfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from alt_nfp.ingest.release_dates.config import PUBLICATIONS, Publication
from alt_nfp.ingest.release_dates.parser import (
    parse_ref_from_path,
    parse_vintage_date,
    ref_date_from_year_month,
)
from alt_nfp.ingest.release_dates.vintage_dates import (
    CES_MONTHLY_REVISIONS,
    # SAE_MONTHLY_REVISIONS,
    build_vintage_dates,
)


class TestPublications:
    """Tests for publication config."""

    def test_two_publications(self):
        assert len(PUBLICATIONS) == 2

    def test_publication_names(self):
        names = {p.name for p in PUBLICATIONS}
        assert names == {'ces', 'qcew'}

    def test_ces_is_monthly(self):
        ces = next(p for p in PUBLICATIONS if p.name == 'ces')
        assert ces.frequency == 'monthly'

    def test_qcew_is_quarterly(self):
        qcew = next(p for p in PUBLICATIONS if p.name == 'qcew')
        assert qcew.frequency == 'quarterly'


class TestParser:
    """Tests for vintage date parsing."""

    def test_parse_vintage_date_standard(self):
        html = '<p>Friday, January 10, 2025, at 8:30 A.M. Eastern Time</p>'
        assert parse_vintage_date(html) == date(2025, 1, 10)

    def test_parse_vintage_date_different_day(self):
        html = 'Tuesday, March 7, 2023, at 10:00 A.M.'
        assert parse_vintage_date(html) == date(2023, 3, 7)

    def test_parse_vintage_date_no_match(self):
        html = '<p>No date here</p>'
        assert parse_vintage_date(html) is None

    def test_parse_vintage_date_wrong_format(self):
        html = '01/10/2025'
        assert parse_vintage_date(html) is None

    def test_parse_ref_from_path_standard(self):
        p = Path('/data/releases/ces/ces_2010_03.htm')
        assert parse_ref_from_path(p) == (2010, 3)

    def test_parse_ref_from_path_december(self):
        p = Path('/data/releases/qcew/qcew_2023_12.htm')
        assert parse_ref_from_path(p) == (2023, 12)

    def test_parse_ref_from_path_invalid(self):
        p = Path('/data/releases/ces/ces_badname.htm')
        assert parse_ref_from_path(p) is None

    def test_parse_ref_from_path_too_few_parts(self):
        p = Path('/data/releases/ces/ces_2010.htm')
        assert parse_ref_from_path(p) is None

    def test_ref_date_from_year_month(self):
        assert ref_date_from_year_month(2023, 6) == date(2023, 6, 12)


class TestBuildVintageDates:
    """Tests for vintage_dates construction from release_dates."""

    @pytest.fixture()
    def release_dates_path(self, tmp_path: Path) -> Path:
        """Create a minimal release_dates.parquet for testing."""
        rows = [
            # CES: monthly releases for a few months
            ('ces', date(2020, 1, 12), date(2020, 2, 7)),
            ('ces', date(2020, 2, 12), date(2020, 3, 6)),
            ('ces', date(2020, 3, 12), date(2020, 4, 3)),
            ('ces', date(2020, 4, 12), date(2020, 5, 8)),
            # CES Jan 2021: triggers benchmark for all 2020 months
            ('ces', date(2021, 1, 12), date(2021, 2, 5)),
            # SAE release dates kept in fixture but SAE processing is disabled
            # ('sae', date(2020, 1, 12), date(2020, 3, 14)),
            # ('sae', date(2020, 2, 12), date(2020, 3, 25)),
            # ('sae', date(2020, 3, 12), date(2020, 4, 17)),
            # ('sae', date(2021, 3, 12), date(2021, 4, 16)),
            # ('sae', date(2022, 3, 12), date(2022, 4, 15)),
            # QCEW: quarterly releases
            ('qcew', date(2020, 3, 12), date(2020, 8, 19)),   # Q1
            ('qcew', date(2020, 6, 12), date(2020, 11, 25)),  # Q2
            ('qcew', date(2020, 9, 12), date(2021, 2, 24)),   # Q3
            ('qcew', date(2020, 12, 12), date(2021, 6, 9)),   # Q4
        ]
        df = pl.DataFrame(
            [{'publication': p, 'ref_date': r, 'vintage_date': v} for p, r, v in rows],
            schema={'publication': pl.Utf8, 'ref_date': pl.Date, 'vintage_date': pl.Date},
        )
        path = tmp_path / 'release_dates.parquet'
        df.write_parquet(path)
        return path

    def test_returns_dataframe(self, release_dates_path: Path):
        result = build_vintage_dates(release_dates_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height > 0

    def test_columns(self, release_dates_path: Path):
        result = build_vintage_dates(release_dates_path)
        expected_cols = {'publication', 'ref_date', 'vintage_date', 'revision', 'benchmark_revision'}
        assert set(result.columns) == expected_cols

    def test_ces_revisions_present(self, release_dates_path: Path):
        result = build_vintage_dates(release_dates_path)
        ces = result.filter(pl.col('publication') == 'ces')
        revisions = ces['revision'].unique().sort().to_list()
        # Should have revisions 0, 1, 2
        for rev in CES_MONTHLY_REVISIONS:
            assert rev in revisions, f'CES revision {rev} missing'

    # def test_sae_revisions_present(self, release_dates_path: Path):
    #     result = build_vintage_dates(release_dates_path)
    #     sae = result.filter(pl.col('publication') == 'sae')
    #     revisions = sae['revision'].unique().sort().to_list()
    #     for rev in SAE_MONTHLY_REVISIONS:
    #         assert rev in revisions, f'SAE revision {rev} missing'

    def test_qcew_q1_max_revision_is_4(self, release_dates_path: Path):
        result = build_vintage_dates(release_dates_path)
        qcew_q1 = result.filter(
            (pl.col('publication') == 'qcew')
            & (pl.col('ref_date') == date(2020, 3, 12))
        )
        max_rev = qcew_q1['revision'].max()
        assert max_rev == 4

    def test_qcew_q4_max_revision_is_1(self, release_dates_path: Path):
        result = build_vintage_dates(release_dates_path)
        qcew_q4 = result.filter(
            (pl.col('publication') == 'qcew')
            & (pl.col('ref_date') == date(2020, 12, 12))
        )
        max_rev = qcew_q4['revision'].max()
        assert max_rev == 1

    def test_ces_benchmark_rows_exist(self, release_dates_path: Path):
        result = build_vintage_dates(release_dates_path)
        benchmarks = result.filter(
            (pl.col('publication') == 'ces')
            & (pl.col('benchmark_revision') == 1)
        )
        # All 2020 ref_dates should have benchmark rows
        assert benchmarks.height > 0
        bench_refs = benchmarks['ref_date'].unique().sort().to_list()
        # 2020 months should be benchmarked
        assert date(2020, 1, 12) in bench_refs
        assert date(2020, 4, 12) in bench_refs

    # def test_sae_two_benchmark_generations(self, release_dates_path: Path):
    #     result = build_vintage_dates(release_dates_path)
    #     sae_bench = result.filter(
    #         (pl.col('publication') == 'sae')
    #         & (pl.col('benchmark_revision') > 0)
    #     )
    #     bench_revs = sae_bench['benchmark_revision'].unique().sort().to_list()
    #     assert 1 in bench_revs
    #     assert 2 in bench_revs

    def test_vintage_date_not_future(self, release_dates_path: Path):
        result = build_vintage_dates(release_dates_path)
        max_vintage = result['vintage_date'].max()
        assert max_vintage <= date.today()

    def test_sorted_output(self, release_dates_path: Path):
        result = build_vintage_dates(release_dates_path)
        # Check it's sorted by publication, ref_date, vintage_date
        resorted = result.sort(
            ['publication', 'ref_date', 'vintage_date', 'revision', 'benchmark_revision']
        )
        assert result.equals(resorted)
