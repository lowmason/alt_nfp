# Configuration

All configuration lives in [`alt_nfp.config`][alt_nfp.config].

## Paths

| Constant | Default | Description |
|---|---|---|
| `BASE_DIR` | Repository root | Resolved from package location |
| `DATA_DIR` | `BASE_DIR / "data"` | Input data directory |
| `OUTPUT_DIR` | `BASE_DIR / "output"` | Generated artifacts |

## Model Constants

### QCEW Observation Noise

| Constant | Value | Description |
|---|---|---|
| `SIGMA_QCEW_M3` | 0.0005 | Quarter-end (M3) noise — near-census filing |
| `SIGMA_QCEW_M12` | 0.0015 | Retrospective-UI months (M1–2) noise |

### Birth/Death Model

| Constant | Value | Description |
|---|---|---|
| `BD_QCEW_LAG` | 6 | QCEW publication lag in months |
| `N_HARMONICS` | 4 | Fourier seasonal harmonic count |

### Cyclical Indicators

Defined in `CYCLICAL_INDICATORS` as a list of dicts:

```python
CYCLICAL_INDICATORS = [
    {'name': 'claims', 'file': 'claims_weekly.csv', 'col': 'claims', 'freq': 'weekly'},
    {'name': 'nfci', 'file': 'nfci.csv', 'col': 'nfci', 'freq': 'weekly'},
    {'name': 'biz_apps', 'file': 'business_applications.csv',
     'col': 'applications', 'freq': 'monthly'},
]
```

## Provider Registry

The `PROVIDERS` list drives the entire pipeline.  To add a new vendor:

```python
from alt_nfp.config import ProviderConfig, PROVIDERS

PROVIDERS.append(
    ProviderConfig(
        name="Vendor",
        file="vendor_index.parquet",
        index_col="growth_index",
        error_model="iid",
        births_file="vendor_births.parquet",  # optional
        births_col="births",                   # optional
    )
)
```

The model, diagnostics, plots, and forecasts will automatically adapt to
include the new provider.

## Sampling Presets

Defined in [`alt_nfp.sampling`][alt_nfp.sampling]:

| Preset | Draws | Tune | Chains | Target Accept | Use Case |
|---|---|---|---|---|---|
| `DEFAULT_SAMPLER_KWARGS` | 8 000 | 6 000 | 4 | 0.97 | Production |
| `MEDIUM_SAMPLER_KWARGS` | 4 000 | 3 000 | 4 | 0.95 | Sensitivity sweeps |
| `LIGHT_SAMPLER_KWARGS` | 2 000 | 2 000 | 2 | 0.95 | Backtest loops |
