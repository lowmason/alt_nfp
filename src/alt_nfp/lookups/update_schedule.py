"""Fetch BLS schedule pages and print updated dict literals for publication_dates.py.

Usage::

    python -m alt_nfp.lookups.update_schedule

Fetches CES, QCEW, and SAE schedule HTML from the BLS website, parses the
release tables, and prints Python dict entries for any dates not already in
:mod:`alt_nfp.lookups.publication_dates`.  Output is copy-paste-ready for
manual review and insertion.

Uses :mod:`html.parser` from the stdlib — no BeautifulSoup dependency.
"""

from __future__ import annotations

import re
from datetime import date
from html.parser import HTMLParser

import httpx

from alt_nfp.lookups.publication_dates import (
    BLS_SCHEDULE_URLS,
    CES_RELEASE_DATES,
    QCEW_RELEASE_DATES,
    # SAE_RELEASE_DATES,
)


# ---------------------------------------------------------------------------
# HTML table parser
# ---------------------------------------------------------------------------


class _TableParser(HTMLParser):
    """Extract rows from the first ``<table>`` in an HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []
        self._cell_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == 'table':
            self.in_table = True
        elif self.in_table and tag == 'tr':
            self.in_row = True
            self.current_row = []
        elif self.in_row and tag in ('td', 'th'):
            self.in_cell = True
            self._cell_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == 'table':
            self.in_table = False
        elif tag == 'tr' and self.in_row:
            self.in_row = False
            if self.current_row:
                self.rows.append(self.current_row)
        elif tag in ('td', 'th') and self.in_cell:
            self.in_cell = False
            self.current_row.append(' '.join(self._cell_text).strip())

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self._cell_text.append(data.strip())


# ---------------------------------------------------------------------------
# Date parsing helpers
# ---------------------------------------------------------------------------

MONTH_NAMES: dict[str, int] = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12,
}

# "January 10, 2025" or "March 7, 2025"
_DATE_RE = re.compile(
    r'(January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
)

# "October 2024" (reference month for CES/SAE)
_REF_MONTH_RE = re.compile(
    r'(January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+(\d{4})',
)

# "2nd Quarter 2025" or "Second Quarter 2025"
_QUARTER_NUM: dict[str, int] = {
    '1st': 1, '2nd': 2, '3rd': 3, '4th': 4,
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4,
}


def _parse_release_date(text: str) -> date | None:
    """Parse a release date like ``January 10, 2025``."""
    m = _DATE_RE.search(text)
    if not m:
        return None
    month = MONTH_NAMES[m.group(1)]
    return date(int(m.group(3)), month, int(m.group(2)))


def _parse_ref_month(text: str) -> date | None:
    """Parse a reference month like ``October 2024`` → ``date(2024, 10, 1)``."""
    m = _REF_MONTH_RE.search(text)
    if not m:
        return None
    month = MONTH_NAMES[m.group(1)]
    return date(int(m.group(2)), month, 1)


def _parse_ref_quarter(text: str) -> tuple[int, int] | None:
    """Parse ``2nd Quarter 2025`` → ``(2025, 2)``."""
    for label, num in _QUARTER_NUM.items():
        if label.lower() in text.lower():
            m = re.search(r'(\d{4})', text)
            if m:
                return (int(m.group(1)), num)
    return None


def _fetch_schedule(url: str) -> str:
    """Fetch a BLS schedule page and return the HTML body."""
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
    }
    r = httpx.get(url, headers=headers, follow_redirects=True, timeout=30.0)
    r.raise_for_status()
    return r.text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Fetch BLS schedules and print new dict entries."""
    today = date.today()

    # -- CES --
    print(f'# CES -- fetched {today}')
    print('# New entries not in publication_dates.py:')
    html = _fetch_schedule(BLS_SCHEDULE_URLS['ces'])
    parser = _TableParser()
    parser.feed(html)
    new_ces = 0
    for row in parser.rows:
        if len(row) < 2:
            continue
        ref = _parse_ref_month(row[0])
        pub = _parse_release_date(row[1])
        if ref and pub and ref not in CES_RELEASE_DATES:
            print(f'    date({ref.year}, {ref.month}, 1): '
                  f'date({pub.year}, {pub.month}, {pub.day}),')
            new_ces += 1
    if new_ces == 0:
        print('    # (no new entries)')
    print()

    # # -- SAE -- (disabled)
    # print(f'# SAE -- fetched {today}')
    # print('# New entries not in publication_dates.py:')
    # html = _fetch_schedule(BLS_SCHEDULE_URLS['sae'])
    # parser = _TableParser()
    # parser.feed(html)
    # new_sae = 0
    # for row in parser.rows:
    #     if len(row) < 2:
    #         continue
    #     ref = _parse_ref_month(row[0])
    #     pub = _parse_release_date(row[1])
    #     if ref and pub and ref not in SAE_RELEASE_DATES:
    #         print(f'    date({ref.year}, {ref.month}, 1): '
    #               f'date({pub.year}, {pub.month}, {pub.day}),')
    #         new_sae += 1
    # if new_sae == 0:
    #     print('    # (no new entries)')
    # print()

    # -- QCEW --
    print(f'# QCEW -- fetched {today}')
    print('# New entries not in publication_dates.py:')
    html = _fetch_schedule(BLS_SCHEDULE_URLS['qcew'])
    parser = _TableParser()
    parser.feed(html)
    new_qcew = 0
    for row in parser.rows:
        if len(row) < 2:
            continue
        ref = _parse_ref_quarter(row[0])
        pub = _parse_release_date(row[1])
        if ref and pub and ref not in QCEW_RELEASE_DATES:
            yr, q = ref
            print(f'    ({yr}, {q}): '
                  f'date({pub.year}, {pub.month}, {pub.day}),')
            new_qcew += 1
    if new_qcew == 0:
        print('    # (no new entries)')


if __name__ == '__main__':
    main()
