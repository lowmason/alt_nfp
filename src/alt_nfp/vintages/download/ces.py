"""Download CES vintage (triangular revision) data from BLS.

Scrapes the CES vintage-data page for ``cesvinall.zip`` and individual
``cesvin*.xlsx`` workbooks. The zip is extracted into
``DATA_DIR / 'downloads' / 'ces' / 'cesvinall'``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from alt_nfp.config import DATA_DIR
from alt_nfp.vintages._client import create_client, get_with_retry

CES_INDEX_URL = 'https://www.bls.gov/web/empsit/cesvindata.htm'
CES_BASE_URL = 'https://www.bls.gov/web/empsit/'


def _resolve_url(href: str, base: str = CES_BASE_URL) -> str:
    """Join *href* against *base* to produce an absolute URL."""
    return urljoin(base, href)


def _discover_links(html: str) -> list[str]:
    """Collect ``cesvinall.zip`` and ``cesvin*.xlsx`` URLs from the CES page."""
    soup = BeautifulSoup(html, 'html.parser')
    out: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if not href or href.startswith('#') or href.startswith('mailto:'):
            continue
        url = _resolve_url(href)
        if url in seen:
            continue
        low = href.lower()
        if 'cesvinall.zip' in low:
            seen.add(url)
            out.append(url)
        elif ('cesvin' in low or 'cesvin_template' in low) and (
            low.endswith('.xlsx') or low.endswith('.zip')
        ):
            seen.add(url)
            out.append(url)
    return out


def download_ces(
    data_dir: Path | None = None,
    *,
    client: httpx.Client | None = None,
) -> None:
    """Scrape the CES vintage-data page and download all linked files.

    Downloads ``cesvinall.zip`` (extracted into ``{data_dir}/downloads/ces/cesvinall/``)
    and individual ``cesvin*.xlsx`` workbooks into ``{data_dir}/downloads/ces/``.

    Parameters
    ----------
    data_dir : Path or None
        Root data directory. Defaults to ``DATA_DIR``.
    client : httpx.Client or None
        Optional pre-built client. A new one is created if not provided.
    """
    base = (data_dir or DATA_DIR) / 'downloads'
    ces_dir = base / 'ces'
    ces_dir.mkdir(parents=True, exist_ok=True)

    own_client = client is None
    if client is None:
        client = create_client()

    try:
        r = get_with_retry(client, CES_INDEX_URL)
        r.raise_for_status()
        links = _discover_links(r.text)
        if not links:
            raise RuntimeError('No CES data links found on index page')

        for url in links:
            parsed = urlparse(url)
            name = Path(parsed.path).name
            if not name:
                continue
            r = get_with_retry(client, url)
            r.raise_for_status()

            if name.lower().endswith('.zip'):
                extract_to = ces_dir / 'cesvinall'
                extract_to.mkdir(parents=True, exist_ok=True)
                zip_path = ces_dir / name
                zip_path.write_bytes(r.content)
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(extract_to)
                zip_path.unlink()
                print(f'  extracted {name} -> {extract_to}/')
            else:
                out_path = ces_dir / name
                out_path.write_bytes(r.content)
                print(f'  saved {name}')
    finally:
        if own_client:
            client.close()
