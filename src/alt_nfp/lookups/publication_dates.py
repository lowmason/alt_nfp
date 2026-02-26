"""Hard-coded BLS publication dates for CES, QCEW, and SAE programs.

Release dates are keyed by reference period (1st of month for CES/SAE,
``(year, quarter)`` tuple for QCEW) and map to the first-print (revision 0)
publication date.  These were scraped from the BLS schedule pages listed in
:data:`BLS_SCHEDULE_URLS`.

The ref_month key convention uses ``date(YYYY, M, 1)`` — the 1st of the
reference month, *not* the 12th used internally by the scraper pipeline
in ``ingest.release_dates``.  The 12th convention is specific to the
scraper's ``ref_date`` column; here we use the 1st for cleaner semantics.
"""

from __future__ import annotations

from datetime import date

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ReleaseDateMap = dict[date, date]
QCEWReleaseDateMap = dict[tuple[int, int], date]

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

BLS_SCHEDULE_URLS: dict[str, str] = {
    'ces': 'https://www.bls.gov/schedule/news_release/empsit.htm',
    'qcew': 'https://www.bls.gov/schedule/news_release/cew.htm',
    'sae': 'https://www.bls.gov/schedule/news_release/laus.htm',
}

LAST_SCRAPED: date = date(2026, 2, 26)

# ---------------------------------------------------------------------------
# CES (Employment Situation) — first-print release dates
# ---------------------------------------------------------------------------
# All releases at 08:30 AM ET.
# Source: https://www.bls.gov/schedule/news_release/empsit.htm
# Key: ref_month (1st of month), Value: publication date

CES_RELEASE_DATES: ReleaseDateMap = {
    date(2024, 10, 1): date(2024, 11, 1),
    date(2024, 11, 1): date(2024, 12, 6),
    date(2024, 12, 1): date(2025, 1, 10),
    date(2025, 1, 1): date(2025, 2, 7),
    date(2025, 2, 1): date(2025, 3, 7),
    date(2025, 3, 1): date(2025, 4, 4),
    date(2025, 4, 1): date(2025, 5, 2),
    date(2025, 5, 1): date(2025, 6, 6),
    date(2025, 6, 1): date(2025, 7, 3),
    date(2025, 7, 1): date(2025, 8, 1),
    date(2025, 8, 1): date(2025, 9, 5),
    # Sep 2025: released Nov 20 (gov shutdown delayed Oct release)
    date(2025, 9, 1): date(2025, 11, 20),
    # Oct 2025: released with Nov on Dec 16 (gov shutdown)
    date(2025, 10, 1): date(2025, 12, 16),
    date(2025, 11, 1): date(2025, 12, 16),
    date(2025, 12, 1): date(2026, 1, 9),
}

# ---------------------------------------------------------------------------
# QCEW (County Employment and Wages) — release dates
# ---------------------------------------------------------------------------
# All releases at 10:00 AM ET.  QCEW is quarterly; key is (year, quarter).
# Source: https://www.bls.gov/schedule/news_release/cew.htm
# Key: (ref_year, ref_quarter as int 1-4), Value: publication date

QCEW_RELEASE_DATES: QCEWReleaseDateMap = {
    (2025, 2): date(2025, 12, 19),
    (2025, 3): date(2026, 3, 10),
    (2025, 4): date(2026, 6, 2),
    (2026, 1): date(2026, 8, 28),
    (2026, 2): date(2026, 12, 2),
}

# ---------------------------------------------------------------------------
# SAE (State Employment and Unemployment) — first-print release dates
# ---------------------------------------------------------------------------
# All releases at 10:00 AM ET.
# Source: https://www.bls.gov/schedule/news_release/laus.htm
# Key: ref_month (1st of month), Value: publication date

SAE_RELEASE_DATES: ReleaseDateMap = {
    date(2024, 10, 1): date(2024, 11, 19),
    date(2024, 11, 1): date(2024, 12, 20),
    date(2024, 12, 1): date(2025, 1, 28),
    date(2025, 1, 1): date(2025, 3, 17),
    date(2025, 2, 1): date(2025, 3, 28),
    date(2025, 3, 1): date(2025, 4, 18),
    date(2025, 4, 1): date(2025, 5, 21),
    date(2025, 5, 1): date(2025, 6, 24),
    date(2025, 6, 1): date(2025, 7, 18),
    date(2025, 7, 1): date(2025, 8, 19),
    date(2025, 8, 1): date(2025, 9, 19),
    # Sep 2025: released Dec 11 (gov shutdown delayed Oct release)
    date(2025, 9, 1): date(2025, 12, 11),
    # Oct 2025: released with Nov (gov shutdown)
    date(2025, 10, 1): date(2025, 12, 16),
}


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import calendar as cal

    # -- CES: most dates should be Fridays (weekday() == 4)
    non_friday_ces = [
        (ref, pub) for ref, pub in CES_RELEASE_DATES.items()
        if pub.weekday() != 4
    ]
    if non_friday_ces:
        print('CES non-Friday releases (likely holiday shifts or shutdowns):')
        for ref, pub in non_friday_ces:
            print(f'  {ref} -> {pub} ({pub.strftime("%A")})')
    else:
        print('All CES releases are on Fridays.')

    # -- QCEW: release >= end_of_quarter + 120 days
    for (yr, q), pub in QCEW_RELEASE_DATES.items():
        end_month = q * 3
        last_day = cal.monthrange(yr, end_month)[1]
        quarter_end = date(yr, end_month, last_day)
        delta = (pub - quarter_end).days
        assert delta >= 120, (
            f'QCEW ({yr}, Q{q}) release {pub} is only '
            f'{delta} days after quarter end {quarter_end}'
        )
    print('All QCEW dates pass the 120-day check.')

    # -- SAE lags CES for overlapping months (>= for shutdown combined releases)
    common_months = set(SAE_RELEASE_DATES.keys()) & set(CES_RELEASE_DATES.keys())
    for ref in sorted(common_months):
        assert SAE_RELEASE_DATES[ref] >= CES_RELEASE_DATES[ref], (
            f'{ref}: SAE {SAE_RELEASE_DATES[ref]} should be on or after '
            f'CES {CES_RELEASE_DATES[ref]}'
        )
    print(f'SAE lags CES for all {len(common_months)} overlapping months.')

    # -- No duplicate ref periods (dict keys are unique by definition,
    #    but verify data is not accidentally truncated)
    assert len(CES_RELEASE_DATES) == len(set(CES_RELEASE_DATES.keys()))
    assert len(QCEW_RELEASE_DATES) == len(set(QCEW_RELEASE_DATES.keys()))
    assert len(SAE_RELEASE_DATES) == len(set(SAE_RELEASE_DATES.keys()))
    print('No duplicate reference periods.')
