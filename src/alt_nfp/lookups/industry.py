"""Canonical BLS NAICS -> supersector -> domain mapping for CES employment data.

Provides the industry hierarchy as a Polars LazyFrame, CES series ID mappings,
and index-builder functions for PyMC hierarchical indexing.
"""

from __future__ import annotations

import numpy as np
import polars as pl


# --- Industry hierarchy ---
# Each row maps a 2-digit NAICS sector to its BLS CES supersector and domain.
# Sector codes use simplified forms: '31' (not '31-33'), '44' (not '44-45'),
# '48' (not '48-49').

_HIERARCHY_ROWS = [
    # Goods-producing (G)
    ('21', 'Mining', '10', 'Mining and Logging', 'G', 'Goods-producing'),
    ('23', 'Construction', '20', 'Construction', 'G', 'Goods-producing'),
    ('31', 'Manufacturing', '30', 'Manufacturing', 'G', 'Goods-producing'),
    # Service-providing (S)
    ('42', 'Wholesale Trade', '40', 'Trade, Transportation, and Utilities', 'S', 'Service-providing'),
    ('44', 'Retail Trade', '40', 'Trade, Transportation, and Utilities', 'S', 'Service-providing'),
    ('48', 'Transportation and Warehousing', '40', 'Trade, Transportation, and Utilities', 'S', 'Service-providing'),
    ('22', 'Utilities', '40', 'Trade, Transportation, and Utilities', 'S', 'Service-providing'),
    ('51', 'Information', '50', 'Information', 'S', 'Service-providing'),
    ('52', 'Finance and Insurance', '55', 'Financial Activities', 'S', 'Service-providing'),
    ('53', 'Real Estate', '55', 'Financial Activities', 'S', 'Service-providing'),
    ('54', 'Professional and Technical Services', '60', 'Professional and Business Services', 'S', 'Service-providing'),
    ('55', 'Management of Companies', '60', 'Professional and Business Services', 'S', 'Service-providing'),
    ('56', 'Administrative and Waste Services', '60', 'Professional and Business Services', 'S', 'Service-providing'),
    ('61', 'Educational Services', '65', 'Private Education and Health Services', 'S', 'Service-providing'),
    ('62', 'Health Care and Social Assistance', '65', 'Private Education and Health Services', 'S', 'Service-providing'),
    ('71', 'Arts, Entertainment, and Recreation', '70', 'Leisure and Hospitality', 'S', 'Service-providing'),
    ('72', 'Accommodation and Food Services', '70', 'Leisure and Hospitality', 'S', 'Service-providing'),
    ('81', 'Other Services', '80', 'Other Services', 'S', 'Service-providing'),
]

INDUSTRY_HIERARCHY: pl.LazyFrame = pl.LazyFrame(
    {
        'sector_code': [r[0] for r in _HIERARCHY_ROWS],
        'sector_title': [r[1] for r in _HIERARCHY_ROWS],
        'supersector_code': [r[2] for r in _HIERARCHY_ROWS],
        'supersector_title': [r[3] for r in _HIERARCHY_ROWS],
        'domain_code': [r[4] for r in _HIERARCHY_ROWS],
        'domain_title': [r[5] for r in _HIERARCHY_ROWS],
    },
    schema={
        'sector_code': pl.Utf8,
        'sector_title': pl.Utf8,
        'supersector_code': pl.Utf8,
        'supersector_title': pl.Utf8,
        'domain_code': pl.Utf8,
        'domain_title': pl.Utf8,
    },
)


def _build_ces_series_id(supersector_code: str, sa: bool) -> str:
    """Build a BLS CES series ID for all-employees employment level.

    Parameters
    ----------
    supersector_code : str
        Two-digit BLS CES supersector code (e.g., '30').
    sa : bool
        True for seasonally adjusted, False for not seasonally adjusted.

    Returns
    -------
    str
        BLS series ID (e.g., 'CES3000000001' for Manufacturing SA).
    """
    prefix = 'S' if sa else 'U'
    return f'CE{prefix}{supersector_code}00000001'


# CES_SERIES_MAP: maps (supersector_code, sa) -> BLS series ID
# Includes all 10 supersectors plus total private ('05').
_SUPERSECTOR_CODES = sorted(
    set(r[2] for r in _HIERARCHY_ROWS)
)  # ['10', '20', '30', '40', '50', '55', '60', '65', '70', '80']

CES_SERIES_MAP: dict[tuple[str, bool], str] = {}
for _code in _SUPERSECTOR_CODES + ['05']:
    CES_SERIES_MAP[(_code, True)] = _build_ces_series_id(_code, sa=True)
    CES_SERIES_MAP[(_code, False)] = _build_ces_series_id(_code, sa=False)


def get_domain_codes() -> list[str]:
    """Return sorted unique domain codes.

    Returns
    -------
    list[str]
        ['G', 'S']
    """
    return sorted(set(r[4] for r in _HIERARCHY_ROWS))


def get_supersector_codes() -> list[str]:
    """Return sorted unique supersector codes.

    Returns
    -------
    list[str]
        ['10', '20', '30', '40', '50', '55', '60', '65', '70', '80']
    """
    return sorted(set(r[2] for r in _HIERARCHY_ROWS))


def get_sector_codes() -> list[str]:
    """Return sorted unique sector codes.

    Returns
    -------
    list[str]
        All 18 sector codes in sorted order.
    """
    return sorted(set(r[0] for r in _HIERARCHY_ROWS))


def supersector_to_domain_idx() -> np.ndarray:
    """Map each supersector to its parent domain index.

    Returns
    -------
    np.ndarray
        Integer array of length n_supersectors. Entry i is the domain index
        for the i-th supersector (sorted order).
    """
    domain_codes = get_domain_codes()
    domain_to_idx = {d: i for i, d in enumerate(domain_codes)}

    ss_codes = get_supersector_codes()
    # Build supersector -> domain lookup from hierarchy rows
    ss_to_domain: dict[str, str] = {}
    for r in _HIERARCHY_ROWS:
        ss_to_domain[r[2]] = r[4]

    return np.array([domain_to_idx[ss_to_domain[ss]] for ss in ss_codes], dtype=np.intp)


def sector_to_supersector_idx() -> np.ndarray:
    """Map each sector to its parent supersector index.

    Returns
    -------
    np.ndarray
        Integer array of length n_sectors. Entry i is the supersector index
        for the i-th sector (sorted order).
    """
    ss_codes = get_supersector_codes()
    ss_to_idx = {ss: i for i, ss in enumerate(ss_codes)}

    sec_codes = get_sector_codes()
    # Build sector -> supersector lookup from hierarchy rows
    sec_to_ss: dict[str, str] = {}
    for r in _HIERARCHY_ROWS:
        sec_to_ss[r[0]] = r[2]

    return np.array([ss_to_idx[sec_to_ss[sec]] for sec in sec_codes], dtype=np.intp)
