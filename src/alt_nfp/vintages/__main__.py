"""Unified CLI for the vintage-building pipeline.

Usage::

    python -m alt_nfp.vintages              # Run all steps
    python -m alt_nfp.vintages release      # Scrape release dates
    python -m alt_nfp.vintages download     # Download CES + QCEW revision files
    python -m alt_nfp.vintages process      # Process CES, SAE, QCEW revisions
    python -m alt_nfp.vintages build        # Combine + build vintage_store
"""

from __future__ import annotations

import sys


def cmd_release() -> None:
    """Scrape BLS news releases and build release_dates + vintage_dates Parquets."""
    import asyncio

    import httpx
    import polars as pl

    from alt_nfp.ingest.release_dates.config import (
        PUBLICATIONS,
        RELEASE_DATES_PATH,
        RELEASES_DIR,
        VINTAGE_DATES_PATH,
    )
    from alt_nfp.ingest.release_dates.parser import collect_release_dates
    from alt_nfp.ingest.release_dates.scraper import (
        download_all,
        fetch_index,
        parse_index_page,
    )
    from alt_nfp.ingest.release_dates.vintage_dates import (
        SUPPLEMENTAL_RELEASE_DATES,
        build_vintage_dates,
    )

    async def _download_all_publications() -> None:
        async with httpx.AsyncClient(
            http2=True, follow_redirects=True, timeout=30.0,
        ) as client:
            for pub in PUBLICATIONS:
                print(f'Fetching index for {pub.name}...')
                html = await fetch_index(client, pub.index_url)
                entries = parse_index_page(
                    html, pub.name, pub.series, pub.frequency,
                )
                print(f'  Found {len(entries)} releases for {pub.name}')
                paths = await download_all(entries, pub.name)
                print(f'  Downloaded {len(paths)} new files for {pub.name}')

    asyncio.run(_download_all_publications())

    # Build release_dates.parquet
    print('Building release_dates...')
    rows = []
    for pub in PUBLICATIONS:
        pub_dir = RELEASES_DIR / pub.name
        if not pub_dir.exists():
            continue
        for row in collect_release_dates(pub.name, pub_dir):
            rows.append(row)

    df = pl.DataFrame(
        rows,
        schema={'publication': pl.Utf8, 'ref_date': pl.Date, 'vintage_date': pl.Date},
        orient='row',
    )
    # Merge supplemental release dates
    supplemental = pl.DataFrame(
        [
            {'publication': p, 'ref_date': ref, 'vintage_date': vint}
            for p, ref, vint in SUPPLEMENTAL_RELEASE_DATES
        ],
        schema={'publication': pl.Utf8, 'ref_date': pl.Date, 'vintage_date': pl.Date},
    )
    existing_keys = df.select('publication', 'ref_date').unique()
    supplemental = supplemental.join(
        existing_keys, on=['publication', 'ref_date'], how='anti',
    )
    if supplemental.height > 0:
        df = pl.concat([df, supplemental])
    df = df.sort('publication', 'ref_date')

    RELEASE_DATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(RELEASE_DATES_PATH)
    print(f'Wrote {RELEASE_DATES_PATH} ({len(df)} rows)')

    # Build vintage_dates.parquet
    print('Building vintage_dates...')
    vdf = build_vintage_dates()
    VINTAGE_DATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    vdf.write_parquet(VINTAGE_DATES_PATH)
    print(f'Wrote {VINTAGE_DATES_PATH} ({len(vdf)} rows)')


def cmd_download() -> None:
    """Download CES and QCEW data files."""
    from alt_nfp.vintages.download import download_ces, download_qcew

    print('Downloading CES vintage data...')
    download_ces()
    print('Downloading QCEW revisions CSV...')
    download_qcew()
    print('Done.')


def cmd_process() -> None:
    """Run all processing steps: CES national, SAE states, QCEW, combine."""
    from alt_nfp.vintages.processing.ces_national import main as ces_national_main
    from alt_nfp.vintages.processing.combine import main as combine_main
    from alt_nfp.vintages.processing.qcew import main as qcew_main
    from alt_nfp.vintages.processing.sae_states import main as sae_states_main

    print('=== Processing CES national revisions ===')
    ces_national_main()
    print('\n=== Processing SAE state revisions ===')
    sae_states_main()
    print('\n=== Processing QCEW revisions ===')
    qcew_main()
    print('\n=== Combining revisions ===')
    combine_main()


def cmd_build() -> None:
    """Build the Hive-partitioned vintage store."""
    from alt_nfp.vintages.build_store import build_store

    releases_path = None
    args = sys.argv[2:]
    if '--releases' in args:
        idx = args.index('--releases')
        if idx + 1 < len(args):
            from pathlib import Path

            releases_path = Path(args[idx + 1])

    build_store(releases_path=releases_path)


def main() -> None:
    """Dispatch to a subcommand, or run all stages if none is given."""
    args = sys.argv[1:]

    if not args:
        cmd_release()
        cmd_download()
        cmd_process()
        cmd_build()
        return

    subcommand = args[0]
    dispatch = {
        'release': cmd_release,
        'download': cmd_download,
        'process': cmd_process,
        'build': cmd_build,
    }
    handler = dispatch.get(subcommand)
    if handler is None:
        print(f'Unknown subcommand: {subcommand}')
        print('Usage: python -m alt_nfp.vintages [release|download|process|build]')
        sys.exit(1)
    handler()


if __name__ == '__main__':
    main()
