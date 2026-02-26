"""Download the QCEW revisions CSV (2017-present) from BLS.

Saves ``qcew-revisions.csv`` to ``DATA_DIR / 'raw' / 'qcew'``.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from alt_nfp.config import DATA_DIR
from alt_nfp.vintages._client import create_client, get_with_retry

QCEW_CSV_URL = 'https://www.bls.gov/cew/revisions/qcew-revisions.csv'
QCEW_FILENAME = 'qcew-revisions.csv'


def download_qcew(
    data_dir: Path | None = None,
    *,
    client: httpx.Client | None = None,
) -> None:
    """Download the QCEW revisions CSV.

    Parameters
    ----------
    data_dir : Path or None
        Root data directory. Defaults to ``DATA_DIR``.
    client : httpx.Client or None
        Optional pre-built client. A new one is created if not provided.
    """
    base = (data_dir or DATA_DIR) / 'raw'
    qcew_dir = base / 'qcew'
    qcew_dir.mkdir(parents=True, exist_ok=True)
    out_path = qcew_dir / QCEW_FILENAME

    own_client = client is None
    if client is None:
        client = create_client()

    try:
        r = get_with_retry(client, QCEW_CSV_URL)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        print(f'  saved {QCEW_FILENAME}')
    finally:
        if own_client:
            client.close()
