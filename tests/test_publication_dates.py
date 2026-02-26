"""Tests for alt_nfp.lookups.publication_dates — hard-coded BLS release dates."""

from __future__ import annotations

import calendar as cal
from datetime import date

from alt_nfp.lookups.publication_dates import (
    CES_RELEASE_DATES,
    QCEW_RELEASE_DATES,
    SAE_RELEASE_DATES,
)


class TestPublicationDates:
    """Validate structural properties of the hard-coded release date dicts."""

    def test_ces_dates_are_after_ref_month(self):
        """Every CES release date is at least 20 days after its reference month."""
        for ref, pub in CES_RELEASE_DATES.items():
            delta = (pub - ref).days
            assert delta >= 20, (
                f'CES {ref}: release {pub} is only {delta} days after ref_month'
            )

    def test_sae_lags_ces(self):
        """For every month with both CES and SAE dates, SAE is on or after CES."""
        common = set(SAE_RELEASE_DATES.keys()) & set(CES_RELEASE_DATES.keys())
        assert len(common) > 0, 'No overlapping months between CES and SAE'
        for ref in sorted(common):
            assert SAE_RELEASE_DATES[ref] >= CES_RELEASE_DATES[ref], (
                f'{ref}: SAE {SAE_RELEASE_DATES[ref]} should be on or after '
                f'CES {CES_RELEASE_DATES[ref]}'
            )

    def test_qcew_lags_quarter_end(self):
        """Every QCEW release date is at least 120 days after end of quarter."""
        for (yr, q), pub in QCEW_RELEASE_DATES.items():
            end_month = q * 3
            last_day = cal.monthrange(yr, end_month)[1]
            quarter_end = date(yr, end_month, last_day)
            delta = (pub - quarter_end).days
            assert delta >= 120, (
                f'QCEW ({yr}, Q{q}): release {pub} is only {delta} days '
                f'after quarter end {quarter_end}'
            )

    def test_no_duplicate_ref_periods(self):
        """Dicts have the expected minimum number of entries (no truncation)."""
        assert len(CES_RELEASE_DATES) >= 10
        assert len(SAE_RELEASE_DATES) >= 10
        assert len(QCEW_RELEASE_DATES) >= 4

    def test_shutdown_entries_present(self):
        """Oct 2025 CES and SAE entries exist with known combined-release dates."""
        # CES Oct 2025: released Dec 16
        assert date(2025, 10, 1) in CES_RELEASE_DATES
        assert CES_RELEASE_DATES[date(2025, 10, 1)] == date(2025, 12, 16)

        # SAE Oct 2025: released Dec 16
        assert date(2025, 10, 1) in SAE_RELEASE_DATES
        assert SAE_RELEASE_DATES[date(2025, 10, 1)] == date(2025, 12, 16)
