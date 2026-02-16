"""Observation panel builder: combines all data sources into a unified panel.

Provides three modes: legacy (existing CSVs), API (BLS fetch + vintages),
and offline (local parquet only).
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ..config import DATA_DIR, PROVIDERS, ProviderConfig
from .base import PANEL_SCHEMA, validate_panel
from .ces import ingest_ces
from .payroll import ingest_provider
from .qcew import ingest_qcew

logger = logging.getLogger(__name__)


def build_panel(
    raw_dir: Path | None = None,
    use_legacy: bool = False,
    use_api: bool = True,
    providers: list[ProviderConfig] | None = None,
    start_year: int = 2010,
    end_year: int | None = None,
) -> pl.DataFrame:
    """Build a unified observation panel from all data sources.

    Parameters
    ----------
    raw_dir : Path, optional
        Directory containing raw data files (vintages, provider CSVs).
        Defaults to DATA_DIR / 'raw'.
    use_legacy : bool
        If True, load from existing data/*.csv files with simplified
        single-vintage ingestion. Backward-compatible path.
    use_api : bool
        If True (default), fetch current data from BLS API via eco-stats.
    providers : list[ProviderConfig], optional
        Provider list. Defaults to PROVIDERS from config.
    start_year : int
        First year for data (default 2010).
    end_year : int, optional
        Last year for data. Defaults to current year.

    Returns
    -------
    pl.DataFrame
        Validated observation panel conforming to PANEL_SCHEMA.
    """
    if providers is None:
        providers = PROVIDERS
    if end_year is None:
        end_year = date.today().year
    if raw_dir is None:
        raw_dir = DATA_DIR / 'raw'

    vintage_dir = raw_dir / 'vintages' if raw_dir.exists() else None

    if use_legacy:
        return _build_legacy_panel(providers)

    parts: list[pl.DataFrame] = []

    # CES ingestion
    try:
        ces_df = ingest_ces(
            vintage_dir=vintage_dir,
            start_year=start_year,
            end_year=end_year,
        ) if use_api else ingest_ces(
            vintage_dir=vintage_dir,
            start_year=start_year,
            end_year=end_year,
        )
        if len(ces_df) > 0:
            parts.append(ces_df)
    except Exception as e:
        logger.warning(f'CES ingestion failed: {e}')

    # QCEW ingestion
    try:
        qcew_df = ingest_qcew(
            vintage_dir=vintage_dir,
            start_year=start_year,
            end_year=end_year,
        )
        if len(qcew_df) > 0:
            parts.append(qcew_df)
    except Exception as e:
        logger.warning(f'QCEW ingestion failed: {e}')

    # Payroll provider ingestion
    provider_dir = raw_dir / 'providers' if raw_dir.exists() else None
    for cfg in providers:
        try:
            pp_df = ingest_provider(cfg, raw_dir=provider_dir)
            if len(pp_df) > 0:
                parts.append(pp_df)
        except Exception as e:
            logger.warning(f'Provider {cfg.name} ingestion failed: {e}')

    if not parts:
        logger.warning('No data ingested from any source')
        return pl.DataFrame(schema=PANEL_SCHEMA)

    panel = pl.concat(parts).sort('period', 'source', 'industry_code', 'revision_number')

    return validate_panel(panel)


def _build_legacy_panel(providers: list[ProviderConfig]) -> pl.DataFrame:
    """Build panel from existing data/*.csv files (single vintage per source).

    This maintains backward compatibility with the existing flat CSV data.
    """
    rows: list[dict] = []
    today = date.today()

    # CES
    ces_path = DATA_DIR / 'ces_index.csv'
    if ces_path.exists():
        ces = pl.read_csv(str(ces_path), try_parse_dates=True).sort('ref_date')

        for sa_col, source, source_type, is_sa in [
            ('ces_sa_index', 'ces_sa', 'official_sa', True),
            ('ces_nsa_index', 'ces_nsa', 'official_nsa', False),
        ]:
            if sa_col not in ces.columns:
                continue

            ref_dates = ces['ref_date'].to_list()
            levels = ces[sa_col].to_numpy().astype(float)

            for i in range(1, len(levels)):
                if levels[i] > 0 and levels[i - 1] > 0 and np.isfinite(levels[i]) and np.isfinite(levels[i - 1]):
                    growth = float(np.log(levels[i]) - np.log(levels[i - 1]))
                    rows.append(
                        {
                            'period': ref_dates[i],
                            'industry_code': '05',
                            'industry_level': 'supersector',
                            'source': source,
                            'source_type': source_type,
                            'growth': growth,
                            'employment_level': float(levels[i]),
                            'is_seasonally_adjusted': is_sa,
                            'vintage_date': today,
                            'revision_number': -1,
                            'is_final': True,
                            'publication_lag_months': None,
                            'coverage_ratio': None,
                        }
                    )

    # QCEW
    qcew_path = DATA_DIR / 'qcew_index.csv'
    if qcew_path.exists():
        qcew = pl.read_csv(str(qcew_path), try_parse_dates=True).sort('ref_date')

        if 'qcew_nsa_index' in qcew.columns:
            ref_dates = qcew['ref_date'].to_list()
            levels = qcew['qcew_nsa_index'].to_numpy().astype(float)

            for i in range(1, len(levels)):
                if levels[i] > 0 and levels[i - 1] > 0 and np.isfinite(levels[i]) and np.isfinite(levels[i - 1]):
                    growth = float(np.log(levels[i]) - np.log(levels[i - 1]))
                    rows.append(
                        {
                            'period': ref_dates[i],
                            'industry_code': '05',
                            'industry_level': 'supersector',
                            'source': 'qcew',
                            'source_type': 'census',
                            'growth': growth,
                            'employment_level': float(levels[i]),
                            'is_seasonally_adjusted': False,
                            'vintage_date': today,
                            'revision_number': -1,
                            'is_final': True,
                            'publication_lag_months': None,
                            'coverage_ratio': None,
                        }
                    )

    # Payroll providers
    for cfg in providers:
        pp_df = ingest_provider(cfg)
        if len(pp_df) > 0:
            rows.extend(pp_df.to_dicts())

    if not rows:
        return pl.DataFrame(schema=PANEL_SCHEMA)

    panel = pl.DataFrame(rows, schema=PANEL_SCHEMA)
    panel = panel.sort('period', 'source', 'industry_code', 'revision_number')

    return validate_panel(panel)


def save_panel(panel: pl.DataFrame, output_dir: Path) -> None:
    """Save observation panel to parquet with a manifest file.

    Parameters
    ----------
    panel : pl.DataFrame
        Validated observation panel.
    output_dir : Path
        Directory to write observation_panel.parquet and panel_manifest.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    panel.write_parquet(output_dir / 'observation_panel.parquet')

    # Build manifest
    source_counts = (
        panel.group_by('source')
        .len()
        .sort('source')
        .to_dicts()
    )

    git_hash = None
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except Exception:
        pass

    manifest = {
        'build_timestamp': datetime.now().isoformat(),
        'row_count': len(panel),
        'source_counts': {d['source']: d['len'] for d in source_counts},
        'date_range': {
            'min': str(panel['period'].min()),
            'max': str(panel['period'].max()),
        },
        'git_hash': git_hash,
    }

    manifest_path = output_dir / 'panel_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f'Panel saved: {len(panel)} rows to {output_dir / "observation_panel.parquet"}'
    )


def load_panel(panel_dir: Path) -> pl.DataFrame:
    """Load a previously saved observation panel from parquet.

    Parameters
    ----------
    panel_dir : Path
        Directory containing observation_panel.parquet.

    Returns
    -------
    pl.DataFrame
        Observation panel.
    """
    parquet_path = panel_dir / 'observation_panel.parquet'
    if not parquet_path.exists():
        raise FileNotFoundError(f'Panel file not found: {parquet_path}')

    return pl.read_parquet(parquet_path)
