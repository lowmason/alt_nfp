"""QCEW and CES revision schedules and publication calendar utilities.

Encodes the structural lag patterns and noise multipliers for each vintage
of QCEW and CES data. Provides helper functions to compute approximate
publication dates and lookup noise multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class RevisionSpec:
    """Specification for a single revision vintage.

    Parameters
    ----------
    revision_number : int
        0 = initial publication, 1+ = subsequent revisions, -1 = benchmark.
    lag_months : int
        Months from end of reference period to publication.
    noise_multiplier : float
        Relative noise scaling vs final vintage (final = 1.0).
    """

    revision_number: int
    lag_months: int
    noise_multiplier: float


# QCEW revision schedule by reference quarter.
# Q1 is revised 4 times (5 total vintages: initial + 4 revisions)
# Q2 is revised 3 times (4 total vintages)
# Q3 is revised 2 times (3 total vintages)
# Q4 is revised 1 time  (2 total vintages)
#
# This asymmetry arises from the annual file structure: each annual
# QCEW release revises all quarters of its reference year, so earlier
# quarters accumulate more revision opportunities.
QCEW_REVISIONS: dict[str, list[RevisionSpec]] = {
    'Q1': [
        RevisionSpec(revision_number=0, lag_months=5, noise_multiplier=2.0),
        RevisionSpec(revision_number=1, lag_months=8, noise_multiplier=1.6),
        RevisionSpec(revision_number=2, lag_months=14, noise_multiplier=1.3),
        RevisionSpec(revision_number=3, lag_months=20, noise_multiplier=1.1),
        RevisionSpec(revision_number=4, lag_months=26, noise_multiplier=1.0),
    ],
    'Q2': [
        RevisionSpec(revision_number=0, lag_months=5, noise_multiplier=2.0),
        RevisionSpec(revision_number=1, lag_months=11, noise_multiplier=1.5),
        RevisionSpec(revision_number=2, lag_months=17, noise_multiplier=1.2),
        RevisionSpec(revision_number=3, lag_months=23, noise_multiplier=1.0),
    ],
    'Q3': [
        RevisionSpec(revision_number=0, lag_months=5, noise_multiplier=2.0),
        RevisionSpec(revision_number=1, lag_months=14, noise_multiplier=1.4),
        RevisionSpec(revision_number=2, lag_months=20, noise_multiplier=1.0),
    ],
    'Q4': [
        RevisionSpec(revision_number=0, lag_months=8, noise_multiplier=1.5),
        RevisionSpec(revision_number=1, lag_months=17, noise_multiplier=1.0),
    ],
}


# CES revision schedule (same for all reference months).
# v0 (first print): published ~1 month after reference, noisiest
# v1 (second estimate): ~2 months after, revised
# v2 (third estimate): ~3 months after, revised again
# v-1 (benchmark): ~13 months after, annual benchmark revision
CES_REVISIONS: list[RevisionSpec] = [
    RevisionSpec(revision_number=0, lag_months=1, noise_multiplier=3.0),
    RevisionSpec(revision_number=1, lag_months=2, noise_multiplier=2.0),
    RevisionSpec(revision_number=2, lag_months=3, noise_multiplier=1.5),
    RevisionSpec(revision_number=-1, lag_months=13, noise_multiplier=1.0),
]


def get_qcew_vintage_date(ref_quarter: str, ref_year: int, revision: int) -> date:
    """Compute the approximate publication date for a specific QCEW vintage.

    Parameters
    ----------
    ref_quarter : str
        Reference quarter ('Q1' through 'Q4').
    ref_year : int
        Reference year.
    revision : int
        Revision number (0 = initial, 1+ = subsequent).

    Returns
    -------
    date
        Approximate publication date.

    Raises
    ------
    ValueError
        If ref_quarter is invalid or revision number exceeds available vintages.
    """
    if ref_quarter not in QCEW_REVISIONS:
        raise ValueError(f'Invalid ref_quarter: {ref_quarter}. Must be Q1-Q4.')

    specs = QCEW_REVISIONS[ref_quarter]
    matching = [s for s in specs if s.revision_number == revision]
    if not matching:
        max_rev = max(s.revision_number for s in specs)
        raise ValueError(
            f'Revision {revision} not available for {ref_quarter}. '
            f'Max revision is {max_rev}.'
        )

    lag = matching[0].lag_months

    # End of reference quarter
    quarter_num = int(ref_quarter[1])
    end_month = quarter_num * 3
    # Approximate: add lag_months to end of reference quarter
    pub_month = end_month + lag
    pub_year = ref_year + (pub_month - 1) // 12
    pub_month = ((pub_month - 1) % 12) + 1

    return date(pub_year, pub_month, 1)


def get_ces_vintage_date(ref_month: date, revision: int) -> date:
    """Compute the approximate publication date for a specific CES vintage.

    Parameters
    ----------
    ref_month : date
        First of the reference month.
    revision : int
        Revision number (0 = first print, 1 = second, 2 = third, -1 = benchmark).

    Returns
    -------
    date
        Approximate publication date.

    Raises
    ------
    ValueError
        If revision number is not in the CES revision schedule.
    """
    matching = [s for s in CES_REVISIONS if s.revision_number == revision]
    if not matching:
        valid = sorted(set(s.revision_number for s in CES_REVISIONS))
        raise ValueError(
            f'Revision {revision} not in CES schedule. Valid: {valid}.'
        )

    lag = matching[0].lag_months
    # Add lag months to end of reference month
    total_months = ref_month.month + lag
    pub_year = ref_month.year + (total_months - 1) // 12
    pub_month = ((total_months - 1) % 12) + 1

    return date(pub_year, pub_month, 1)


def get_noise_multiplier(source: str, revision_number: int) -> float:
    """Lookup noise multiplier for a given source and revision number.

    Parameters
    ----------
    source : str
        Source identifier. For QCEW, use 'qcew_Q1' through 'qcew_Q4'.
        For CES, use 'ces'. For payroll providers, returns 1.0.
    revision_number : int
        Revision number to look up.

    Returns
    -------
    float
        Noise multiplier (1.0 = final vintage noise level).
    """
    if source.startswith('qcew_Q'):
        quarter = source.replace('qcew_', '')
        if quarter in QCEW_REVISIONS:
            for spec in QCEW_REVISIONS[quarter]:
                if spec.revision_number == revision_number:
                    return spec.noise_multiplier
        return 1.0
    elif source.startswith('ces'):
        for spec in CES_REVISIONS:
            if spec.revision_number == revision_number:
                return spec.noise_multiplier
        return 1.0
    else:
        # Payroll providers are not revised
        return 1.0


@dataclass
class PublicationCalendar:
    """Actual BLS publication dates, loaded from a supplementary file.

    The structural lag_months in RevisionSpec gives approximate dates.
    This class provides exact dates when available, parsed from BLS
    schedule pages provided separately.

    Attributes
    ----------
    ces_release_dates : pl.DataFrame
        Columns: ref_month (Date), revision_number (Int32), pub_date (Date).
    qcew_release_dates : pl.DataFrame
        Columns: ref_quarter (Utf8), ref_year (Int32),
        revision_number (Int32), pub_date (Date).
    """

    ces_release_dates: pl.DataFrame
    qcew_release_dates: pl.DataFrame

    @classmethod
    def from_parquet(cls, path: Path) -> 'PublicationCalendar':
        """Load publication calendar from a parquet file.

        Parameters
        ----------
        path : Path
            Path to the publication_calendar.parquet file.

        Returns
        -------
        PublicationCalendar
            Loaded calendar with CES and QCEW release dates.
        """
        df = pl.read_parquet(path)

        ces_df = (
            df.filter(pl.col('source') == 'ces')
            .with_columns(
                pl.col('ref_period').str.to_date('%Y-%m').alias('ref_month'),
            )
            .select('ref_month', 'revision_number', pl.col('publication_date').alias('pub_date'))
        )

        qcew_df = (
            df.filter(pl.col('source') == 'qcew')
            .with_columns(
                pl.col('ref_period').str.slice(0, 4).cast(pl.Int32).alias('ref_year'),
                pl.col('ref_period').str.slice(4).alias('ref_quarter'),
            )
            .select(
                'ref_quarter',
                'ref_year',
                'revision_number',
                pl.col('publication_date').alias('pub_date'),
            )
        )

        return cls(ces_release_dates=ces_df, qcew_release_dates=qcew_df)
