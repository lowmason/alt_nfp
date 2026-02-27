# Data Sources

`alt_nfp` integrates three families of employment data, each with distinct
statistical properties that the model exploits.

## QCEW — Quarterly Census of Employment and Wages

The QCEW is a near-census of all establishments covered by unemployment
insurance (UI), covering roughly 97 % of nonfarm payroll employment.

| Property | Detail |
|---|---|
| **Coverage** | ~97 % of nonfarm employment |
| **Frequency** | Quarterly (with monthly interpolation for model) |
| **Publication lag** | ~6 months |
| **Seasonal adjustment** | Not seasonally adjusted (NSA) |
| **Role in model** | Truth anchor — fixed observation noise |

**Noise calibration:** Quarter-end months (M3: March, June, September,
December) reflect current-period tax filings and receive tighter noise
(\(\sigma = 0.05\) %/month).  Retrospective-UI months (M1–2) are noisier
(\(\sigma = 0.15\) %/month).

## CES — Current Employment Statistics

The CES survey samples ~145,000 businesses and government agencies covering
approximately 697,000 worksites.

| Property | Detail |
|---|---|
| **Coverage** | ~40 % of nonfarm employment |
| **Frequency** | Monthly |
| **Publication lag** | ~3 weeks |
| **Seasonal adjustment** | Both SA and NSA versions used |
| **Vintages** | First print (v1), second print (v2), final/benchmark (v3) |
| **Role in model** | High-frequency signal with vintage-specific noise |

The model estimates shared CES bias (\(\alpha_{\text{CES}}\)) and loading
(\(\lambda_{\text{CES}}\)) parameters but allows each vintage to have its
own observation noise.

## Payroll Providers

Private payroll-processing companies provide employment indices based on
their client populations.

| Property | Detail |
|---|---|
| **Coverage** | Varies by vendor (typically 5–20 % of employment) |
| **Frequency** | Monthly |
| **Publication lag** | Days to weeks |
| **Seasonal adjustment** | Not seasonally adjusted |
| **Role in model** | Real-time nowcasting signals |

### Provider Configuration

Each provider is defined by a [`ProviderConfig`][alt_nfp.config.ProviderConfig]:

```python
ProviderConfig(
    name="PP1",
    file="alt_nfp_index_1.csv",
    index_col="pp_index_1",
    error_model="ar1",
)
```

**Error models:**

- `iid` — independent measurement errors (default).
- `ar1` — autocorrelated measurement errors, appropriate when provider
  data exhibits restructuring-related serial correlation.

### Birth-Rate Data

Providers that report business formation rates can supply a separate births
file, used as a covariate in the structural BD model:

```python
ProviderConfig(
    name="PP2",
    file="alt_nfp_index_2.csv",
    index_col="pp_index_2_0",
    error_model="iid",
    births_file="alt_nfp_births_2.csv",
    births_col="pp2_births",
)
```

## Cyclical Indicators

Optional demand-side covariates for the BD model:

| Indicator | File | Frequency | Column |
|---|---|---|---|
| Initial UI claims | `claims_weekly.csv` | Weekly → monthly avg | `claims` |
| Financial conditions (NFCI) | `nfci.csv` | Weekly → monthly avg | `nfci` |
| Business applications | `business_applications.csv` | Monthly | `applications` |

These are automatically loaded, aligned to the model calendar, and centred.
Missing files are gracefully skipped.

## File Layout

```
data/
├── ces_index.csv              # ref_date, ces_sa_index, ces_nsa_index, ...
├── qcew_index.csv             # ref_date, qcew_nsa_index
├── alt_nfp_index_1.csv        # ref_date, pp_index_1
├── alt_nfp_index_2.csv        # ref_date, pp_index_2_0
├── alt_nfp_births_2.csv       # ref_date, pp2_births
├── claims_weekly.csv          # ref_date, claims (optional)
├── nfci.csv                   # ref_date, nfci (optional)
├── business_applications.csv  # ref_date, applications (optional)
└── raw/
    ├── vintages/              # Parquet vintage files
    └── providers/             # Raw provider data by vendor
```
