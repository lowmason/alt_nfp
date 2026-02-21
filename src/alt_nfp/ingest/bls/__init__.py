'''
BLS data download infrastructure for alt-nfp.

Provides download functions for three BLS programs:
- QCEW (Quarterly Census of Employment and Wages)
- CES National (Current Employment Statistics, national)
- CES State (Current Employment Statistics, state and area)

Ported from eco-stats (https://github.com/lowmason/eco-stats),
retaining only the BLS functionality needed for NFP nowcasting.
'''

from alt_nfp.ingest.bls._http import BLSHttpClient
from alt_nfp.ingest.bls._programs import (
    BLSProgram,
    SeriesField,
    build_series_id,
    get_program,
    list_programs,
    parse_series_id,
)
from alt_nfp.ingest.bls.ces_national import fetch_ces_national
from alt_nfp.ingest.bls.ces_state import fetch_ces_state
from alt_nfp.ingest.bls.qcew import fetch_qcew

__all__ = [
    'BLSHttpClient',
    'BLSProgram',
    'SeriesField',
    'build_series_id',
    'fetch_ces_national',
    'fetch_ces_state',
    'fetch_qcew',
    'get_program',
    'list_programs',
    'parse_series_id',
]
