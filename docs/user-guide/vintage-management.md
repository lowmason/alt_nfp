# Vintage Management

Employment data undergoes revisions after initial publication.  `alt_nfp`
tracks these revisions through a vintage pipeline and observation panel
system.

## Why Vintages Matter

CES employment estimates are revised multiple times:

| Vintage | Timing | Typical Revision |
|---|---|---|
| First print (v1) | Release + 0 months | Baseline |
| Second print (v2) | Release + 1 month | ±15k jobs |
| Final (v3) | Release + 2 months | ±30k jobs |
| Benchmark (v-1) | Annual benchmark | ±100k+ jobs |

QCEW data is also revised as more UI tax records are processed.  Ignoring
revisions can introduce look-ahead bias in backtests and misspecify the
observation noise in the model.

## Observation Panel

All data sources are unified into a single observation panel with schema
`PANEL_SCHEMA`:

| Column | Type | Description |
|---|---|---|
| `period` | Date | Reference month |
| `geographic_type` | Utf8 | "national", "state", etc. |
| `geographic_code` | Utf8 | "US", FIPS code, etc. |
| `industry_code` | Utf8 | NAICS-based code |
| `industry_level` | Utf8 | "supersector" or "sector" |
| `source` | Utf8 | "ces_sa", "ces_nsa", "qcew", provider name |
| `source_type` | Utf8 | "official_sa", "official_nsa", "census", "payroll" |
| `growth` | Float64 | Log growth rate |
| `employment_level` | Float64 | Employment level |
| `is_seasonally_adjusted` | Boolean | SA flag |
| `vintage_date` | Date | When this estimate was published |
| `revision_number` | Int32 | 0=initial, 1+=subsequent, -1=benchmark |
| `is_final` | Boolean | Final vintage flag |
| `publication_lag_months` | Int32 | Months between ref period and publication |
| `coverage_ratio` | Float64 | Fraction of universe covered |

## Building the Panel

```python
from alt_nfp.ingest import build_panel

# From vintage store (primary path)
panel = build_panel()

# From legacy CSVs (backward compatible)
panel = build_panel(use_legacy=True)

# Save/load
from alt_nfp.ingest import save_panel, load_panel
save_panel(panel, output_dir=Path("data/processed"))
panel = load_panel(Path("data/processed"))
```

## Vintage Store

The Hive-partitioned Parquet vintage store provides efficient storage and
querying of revision history:

```
data/raw/vintages/vintage_store/
├── source=ces_sa/
│   ├── vintage_date=2024-01-05/
│   └── vintage_date=2024-02-02/
├── source=ces_nsa/
├── source=qcew/
└── source=sae/
```

### Reading the Store

```python
from alt_nfp.ingest import read_vintage_store, transform_to_panel

lf = read_vintage_store(store_path, ref_date_range=(start, end))
panel = transform_to_panel(lf, geographic_scope='national')
```

## Vintage Views

Lazy views filter the panel without materialising separate tables:

### Real-Time View

Reconstruct the information set available at a specific date:

```python
from alt_nfp.vintages import real_time_view

# What did we know on 2024-06-15?
rt = real_time_view(panel.lazy(), as_of=date(2024, 6, 15)).collect()
```

### Final View

Get only the most-revised estimates:

```python
from alt_nfp.vintages import final_view

final = final_view(panel.lazy()).collect()
```

### Revision Analysis

Compute growth differences between vintages:

```python
from alt_nfp.vintages import vintage_diff

# CES SA revision: first print vs final
revisions = vintage_diff(panel.lazy(), source="ces_sa", rev_a=0, rev_b=-1)
```

## Noise Multipliers

The model can scale observation noise based on revision stage:

```python
from alt_nfp.vintages import build_noise_multiplier_vector

multipliers = build_noise_multiplier_vector(rt_panel)
```

This uses the revision schedules in [`alt_nfp.lookups.revision_schedules`][alt_nfp.lookups.revision_schedules]
to assign appropriate noise scaling.

## Vintage Pipeline CLI

Download and process vintage data from BLS:

```bash
uv run python -m alt_nfp.vintages
```

This orchestrates:

1. **Download** CES and QCEW revision files from BLS.
2. **Process** raw files into tidy Parquet datasets.
3. **Build** the Hive-partitioned vintage store.
