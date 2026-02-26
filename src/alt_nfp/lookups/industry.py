"""Canonical BLS NAICS -> supersector -> domain mapping for CES employment data.

Provides the industry hierarchy as a Polars LazyFrame, CES series ID mappings,
CES-to-QCEW industry cross-mapping (:class:`IndustryEntry`, :data:`INDUSTRY_MAP`),
EN (QCEW) series ID construction, and index-builder functions for PyMC
hierarchical indexing.
"""

from __future__ import annotations

from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# CES-to-QCEW industry cross-mapping
# ---------------------------------------------------------------------------

# CES domain/supersector/sector tuples: (ces_6digit, industry_code_2digit, industry_name)
_CES_DOMAIN = [
    ('000000', '00', 'Total Non-Farm'),
    ('050000', '05', 'Total Private'),
    ('060000', '06', 'Goods-Producing Industries'),
    ('070000', '07', 'Service-Providing Industries'),
    ('080000', '08', 'Private Service-Providing'),
]

_CES_SUPERSECTOR = [
    ('100000', '10', 'Natural Resources and Mining'),
    ('200000', '20', 'Construction'),
    ('300000', '30', 'Manufacturing'),
    ('400000', '40', 'Trade, Transportation, and Utilities'),
    ('500000', '50', 'Information'),
    ('550000', '55', 'Financial Activities'),
    ('600000', '60', 'Professional and Business Services'),
    ('650000', '65', 'Education and Health Services'),
    ('700000', '70', 'Leisure and Hospitality'),
    ('800000', '80', 'Other Services'),
    ('900000', '90', 'Government'),
]

_CES_SECTOR = [
    ('102100', '21', 'Mining, quarrying, and oil and gas extraction'),
    ('310000', '31', 'Durable goods'),
    ('320000', '32', 'Nondurable goods'),
    ('414200', '41', 'Wholesale trade'),
    ('420000', '42', 'Retail trade'),
    ('430000', '43', 'Transportation and warehousing'),
    ('442200', '22', 'Utilities'),
    ('555200', '52', 'Finance and insurance'),
    ('555300', '53', 'Real estate and rental and leasing'),
    ('605400', '54', 'Professional, scientific, and technical services'),
    ('605500', '55', 'Management of companies and enterprises'),
    ('605600', '56', 'Administrative and support and waste management'),
    ('656100', '61', 'Private educational services'),
    ('656200', '62', 'Health care and social assistance'),
    ('707100', '71', 'Arts, entertainment, and recreation'),
    ('707200', '72', 'Accommodation and food services'),
    ('909100', '91', 'Federal'),
    ('909200', '92', 'State government'),
    ('909300', '93', 'Local government'),
]

# QCEW CSV API industry slice codes by supersector (BLS NAICS-based hierarchy).
# '10' = total; '1011'-'1029' = supersectors.
_QCEW_NAICS_BY_SUPERSECTOR: dict[str, str] = {
    '10': '1011',   # Natural Resources and Mining
    '20': '1012',   # Construction
    '30': '1022',   # Manufacturing
    '40': '1023',   # Trade, Transportation, and Utilities
    '50': '1024',   # Information
    '55': '1025',   # Financial Activities
    '60': '1026',   # Professional and Business Services
    '65': '1027',   # Education and Health Services
    '70': '1028',   # Leisure and Hospitality
    '80': '1029',   # Other Services
    '90': '1029',   # Government (QCEW groups with Other; use 92 for gov)
}
_QCEW_NAICS_GOV = '92'

# Sector-level: QCEW uses 2-digit NAICS codes directly.
_QCEW_NAICS_BY_SECTOR: dict[str, str] = {
    '21': '21',     # Mining
    '31': '31',     # Durable goods (NAICS 31-33)
    '32': '32',     # Nondurable goods
    '41': '41',     # Wholesale trade
    '42': '42',     # Retail trade
    '43': '43',     # Transportation and warehousing
    '22': '22',     # Utilities
    '52': '52',     # Finance and insurance
    '53': '53',     # Real estate
    '54': '54',     # Professional, scientific, technical
    '55': '55',     # Management of companies
    '56': '56',     # Administrative and support
    '61': '61',     # Educational services
    '62': '62',     # Health care
    '71': '71',     # Arts, entertainment, recreation
    '72': '72',     # Accommodation and food services
    '91': '91',     # Federal government
    '92': '92',     # State government
    '93': '93',     # Local government
}


@dataclass(frozen=True)
class IndustryEntry:
    """Single industry mapping for cross-program consistency.

    Attributes
    ----------
    industry_code : str
        Unified 2-digit code (e.g. ``'00'``, ``'10'``).
    industry_type : str
        One of ``'domain'``, ``'supersector'``, ``'sector'``.
    industry_name : str
        Human-readable industry name.
    ces_code : str
        Six-digit CES industry code (e.g. ``'000000'``, ``'100000'``).
    qcew_naics : str
        QCEW NAICS code for the CSV slice API (e.g. ``'10'``, ``'1011'``).
    en_industry : str
        Six-digit industry code for EN series ID construction.
    """

    industry_code: str
    industry_type: str
    industry_name: str
    ces_code: str
    qcew_naics: str
    en_industry: str


def _build_industry_map() -> list[IndustryEntry]:
    """Build the canonical industry map across domain, supersector, and sector levels."""
    entries: list[IndustryEntry] = []

    # Domain: use CES code as EN industry; QCEW slice 10 for total, 101 for private, etc.
    domain_qcew = {'00': '10', '05': '101', '06': '101', '07': '102', '08': '102'}
    for ces_code, code, name in _CES_DOMAIN:
        qcew = domain_qcew.get(code, '10')
        entries.append(IndustryEntry(
            industry_code=code,
            industry_type='domain',
            industry_name=name,
            ces_code=ces_code,
            qcew_naics=qcew,
            en_industry=ces_code,
        ))

    # Supersector
    for ces_code, code, name in _CES_SUPERSECTOR:
        qcew = (
            _QCEW_NAICS_BY_SUPERSECTOR.get(code, '10')
            if code != '90'
            else _QCEW_NAICS_GOV
        )
        entries.append(IndustryEntry(
            industry_code=code,
            industry_type='supersector',
            industry_name=name,
            ces_code=ces_code,
            qcew_naics=qcew,
            en_industry=ces_code,
        ))

    # Sector
    for ces_code, code, name in _CES_SECTOR:
        qcew = _QCEW_NAICS_BY_SECTOR.get(code, code)
        entries.append(IndustryEntry(
            industry_code=code,
            industry_type='sector',
            industry_name=name,
            ces_code=ces_code,
            qcew_naics=qcew,
            en_industry=ces_code,
        ))

    return entries


INDUSTRY_MAP: list[IndustryEntry] = _build_industry_map()
"""Complete industry mapping table spanning domain, supersector, and sector levels."""


def qcew_to_sector() -> dict[str, str]:
    """Return a mapping from QCEW API industry codes to simplified 2-digit sector codes.

    Derived from :data:`INDUSTRY_MAP` sector-level entries plus the QCEW CSV API
    code scheme (``'1012'`` -> ``'21'``, etc.).

    Returns
    -------
    dict[str, str]
        QCEW API code -> simplified sector code (e.g. ``'1012'`` -> ``'21'``).
    """
    # Map from QCEW CSV API codes to our sector codes
    # These are the 4-digit API codes used in the QCEW CSV download URLs
    _api_code_to_sector: dict[str, str] = {
        '1012': '21',  # Mining
        '1013': '22',  # Utilities
        '1021': '23',  # Construction
        '1022': '31',  # Manufacturing (31-33)
        '1023': '42',  # Wholesale Trade
        '1024': '44',  # Retail Trade (44-45)
        '1025': '48',  # Transportation and Warehousing (48-49)
        '1026': '51',  # Information
        '1027': '52',  # Finance and Insurance
        '1028': '53',  # Real Estate
        '1029': '54',  # Professional and Technical Services
        '102A': '55',  # Management of Companies
        '102B': '56',  # Administrative and Waste Services
        '102C': '61',  # Educational Services
        '102D': '62',  # Health Care and Social Assistance
        '102E': '71',  # Arts, Entertainment, and Recreation
        '102F': '72',  # Accommodation and Food Services
        '102G': '81',  # Other Services
    }
    # Also include raw NAICS codes that may appear in API responses
    for entry in INDUSTRY_MAP:
        if entry.industry_type == 'sector':
            _api_code_to_sector[entry.qcew_naics] = entry.industry_code
    return _api_code_to_sector


def en_series_id(
    industry_entry: IndustryEntry,
    area: str = 'US000',
    ownership: str = '0',
) -> str:
    """Build an EN (QCEW) series ID for the given industry and area.

    Parameters
    ----------
    industry_entry : IndustryEntry
        Industry mapping entry (provides en_industry).
    area : str
        Area code: ``'US000'`` for national, or state FIPS + ``'000'``
        (e.g. ``'26000'`` for Michigan).
    ownership : str
        ``'0'`` = all ownerships, ``'5'`` = private.

    Returns
    -------
    str
        Full EN series ID string (e.g. ``'ENUUS0001010000001'``).
    """
    from ..ingest.bls import build_series_id

    return build_series_id(
        'EN',
        seasonal='N',
        area=area,
        data_type='1',
        size='0',
        ownership=ownership,
        industry=industry_entry.en_industry,
    )


def en_series_id_for_state(
    industry_entry: IndustryEntry,
    state_fips: str,
    ownership: str = '0',
) -> str:
    """Build an EN series ID for a state-level series.

    Parameters
    ----------
    industry_entry : IndustryEntry
        Industry mapping entry.
    state_fips : str
        Two-digit state FIPS code (e.g. ``'26'`` for Michigan).
    ownership : str
        ``'0'`` = all ownerships, ``'5'`` = private.

    Returns
    -------
    str
        Full EN series ID with area ``{state_fips}000``.
    """
    area = f'{state_fips}000'
    return en_series_id(industry_entry, area=area, ownership=ownership)
