"""Release-dates pipeline: scrape BLS release pages, build vintage_dates.

Use :func:`build_vintage_dates` to build vintage_dates from release_dates.parquet,
or run the full scrape-and-build pipeline via :func:`scrape_and_build`.
"""

from alt_nfp.ingest.release_dates.vintage_dates import build_vintage_dates

__all__ = ['build_vintage_dates']
