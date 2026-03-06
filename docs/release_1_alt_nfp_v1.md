# National Accuracy Model: Bayesian State Space NFP Nowcasting

**Alt-NFP Nowcasting System — Release 1 Specification**

Version: 1.0 \| Date: 2026-03-02

------------------------------------------------------------------------

# Overview

This document describes the National Accuracy Model (Release 1) of the Bayesian state space NFP nowcasting system. The model produces a national NFP nowcast 2–3 weeks before the official CES release by fusing QCEW census data, CES survey data, and private payroll provider signals with structural birth/death modeling, Fourier seasonal adjustment, and vintage-aware CES measurement equations. The model's value is a better number, faster.

The National Accuracy Model is a fully functional, standalone system. It is designed as the foundation for subsequent releases that add industry and geographic decomposition without modifying the national model.

## Design Principles

1.  **Forecast accuracy before narrative:** Birth/death correction and QCEW anchoring are prioritized over decomposition. The system's central value proposition is detecting and correcting for birth/death drift in real time.

2.  **Multi-provider from the start:** The measurement error framework accommodates multiple providers as independent noisy observations of latent truth, with configurable provider-specific error structures. New providers require only a configuration entry.

3.  **Information-regime awareness:** The system explicitly tracks which data sources are available at each point in the publication cycle, producing estimates with uncertainty calibrated to the current information set.

4.  **Representativeness-corrected provider signals:** Provider national signals are pre-processed to remove first-order composition bias by reweighting cell-level continuing-units growth (constructed from pseudo-establishments at the supersector × Census region level) to national QCEW employment shares. This ensures the national model's single-latent-state assumption — that each provider measures the same underlying growth — is approximately satisfied regardless of provider portfolio composition.

## Data Sources

The system combines four classes of data, each with distinct timing, coverage, and measurement properties.

**QCEW (Quarterly Census of Employment and Wages).** Near-census of U.S. employment (\>95% coverage, \~97% of CES scope), based on mandatory quarterly UI tax filings. Published quarterly with \~5–6 month lag. Three monthly employment values per quarter: Month 3 (quarter-end) is contemporaneous at time of filing; Months 1–2 are retrospective. Revision history available from 2017Q1 at national and state levels.

**CES National (Current Employment Statistics).** Probability-based establishment survey (\~670,000 worksites). First print \~3 weeks after reference period; two subsequent monthly revisions. Annual benchmark in February re-anchors to QCEW. Vintage data from May 2003 at the national level by sector.

**Payroll Provider Data.** Continuing-units employment growth from private payroll processors. Available \~3 weeks before CES first print. Constructed from rotating, frozen measurement panels aligned to the CES reference week. Multiple providers supported with independent measurement parameters.

**Cyclical Indicators.** Demand-side covariates for the structural birth/death model. Four indicators are loaded from FRED, aggregated to monthly frequency, and centered (zero-mean, unit-variance) before entering the model:

| Indicator | Source | Frequency | Publication Lag |
|:----------------|:----------------|:----------------|:--------------------|
| Initial unemployment claims (ICNSA) | DOL via FRED | Weekly | \~1 month |
| Chicago Fed NFCI | Chicago Fed via FRED | Weekly | \~1 month |
| Census Business Formation Statistics (BABATOTALSAUS) | Census via FRED | Monthly | \~2 months |
| JOLTS Job Openings (JTSJOL) | BLS via FRED | Monthly | \~2 months |

JOLTS openings are preferred over hires — openings lead the hiring cycle and show stronger covariance with the birth/death component. JOLTS revisions are small relative to cross-sectional variation (mean absolute revision \~1% of level for 2003–2023); final values with publication-lag censoring are sufficient without vintage tracking.

### Training Window

The model trains on **2003–present**. QCEW data is treated as final for 2003–2016 (QCEW revision history is only available from 2017). This is a known limitation for backtesting over the pre-2017 period — it introduces mild QCEW lookahead that slightly overstates model performance. The tradeoff is worthwhile: the 2003–present window captures the 2008 recession, which is the canonical example of birth/death model failure and essential for identifying cyclical parameters.

### Publication Timeline

For a given reference month (e.g., December, reference date December 12):

| Event | Approximate Date | Information Available |
|:---------------------|:--------------------|:-----------------------------|
| Reference period | Dec 12 | --- |
| Provider data available | \~Jan 1 | Provider growth rates (national) |
| CES national 1st print | \~Jan 9 | National total nonfarm + supersectors (SA + NSA) |
| CES national 2nd print | \~Feb 6 | National revised |
| CES national 3rd (final) print | \~Mar 6 | National final sample-based |
| QCEW publication | \~Jun | Full quarter census data |

------------------------------------------------------------------------

# System Description

Multiple data sources observe the same latent national employment growth, each with distinct timing, coverage, and measurement properties. The model decomposes total employment growth into continuing-units growth, a structural birth/death contribution, and a seasonal component. QCEW serves as the truth anchor. CES provides timely but noisy official estimates at multiple vintages. Payroll providers measure continuing-units growth with provider-specific bias, signal loading, and error structure.

## Data Inputs

| Source | Frequency | Lag | Description |
|:-----------------|:-----------------|:-----------------|:-------------------|
| Payroll Provider *p* National Index | Monthly | Real-time (\~Jan 1 for prior month) | Representativeness- corrected national composite: QCEW-weighted average of cell-level continuing-units growth from pseudo- establishments at the supersector × Census region level. |
| CES National (SA) | Monthly | \~3 weeks (1st print), +1mo (2nd), +2mo (3rd) | BLS CES seasonally adjusted total nonfarm. Three vintages with declining noise. |
| CES National (NSA) | Monthly | Same vintage schedule | BLS CES not seasonally adjusted. |
| QCEW National | Quarterly (3 months) | \~5–6 months | Near-census total employment. M3 contemporaneous; M1–M2 retrospective. |
| Cyclical Indicators | Monthly | 1–2 months | Claims, NFCI, BFS, JOLTS (centered, standardized). |
| Provider Birth Rates | Monthly | Real-time | New-client formation rates from payroll providers. |
| QCEW Birth/Death Proxy | Quarterly | \~5–6 months | $g^{QCEW} - \bar{g}^{PP}$: QCEW growth minus mean provider growth (approximates actual BD). |

## Provider Series Construction: Rotating, Frozen Measurement Panel

Payroll-provider observations are explicitly interpreted as continuing-units (intensive-margin) employment change. Each provider's input series is constructed using a rotating, frozen measurement panel aligned to the CES reference week.

Let $k(t)$ denote the active panel at time $t$. The observed provider series $y_{p,t}$ is the panel-based continuing-units growth signal. Within the frozen window, month-to-month growth is computed using matched client observations only. Clients that administratively exit the provider during the window are removed from the panel and do not contribute negative change.

Panel mechanics: at fixed refresh intervals (e.g., quarterly), define an eligible set of clients that have satisfied a stabilization rule (e.g., $\geq K$ consecutive reference periods or change-point-based stabilization). Freeze this set for the panel window; do not add new clients mid-window even if they become eligible later.

This ensures that the provider likelihood loads on $\mu_{t}^{cont}$ and that administrative churn (client onboarding/offboarding) is excluded from the measurement equation rather than implicitly absorbed into the birth/death component.

## Pseudo-Establishment Construction and Representativeness Correction

### Motivation

A raw aggregate provider growth signal reflects the compositional mix of the provider's client base, which differs from the national employment distribution by industry and geography. When the composition of national growth shifts — e.g., healthcare surging while manufacturing contracts — providers with different client portfolios produce divergent aggregate signals. The national model's single latent state interprets this divergence as measurement noise rather than informative compositional heterogeneity. The result is attenuated signal extraction during periods of heterogeneous sectoral growth.

Representativeness correction addresses this by constructing a provider-specific national composite that reweights cell-level continuing-units growth to national employment shares, so that each provider's signal approximates what it would measure if its client base matched the national employment distribution.

### Pseudo-Establishment Construction

Payroll provider data are organized at the client level, where a client is a legal entity that may operate multiple establishments across different states and industries. Client-level NAICS codes and headquarters locations do not reliably identify where employment is situated, particularly for large multi-establishment firms. However, employee-level geocodes are available for all clients.

Pseudo-establishments are constructed by clustering employees within each client by geographic proximity. The spatial dispersion of a client's workforce identifies distinct worksites: a national retail chain with employees clustered in 50 metro areas produces 50 pseudo-establishments, each assigned to the appropriate state. Industry classification is inherited from the client's NAICS code, which is reliable at the supersector level — retail workers at a retail chain are in retail regardless of which store location they work at.

### Cell-Level Signal Construction

Pseudo-establishments are assigned to cells defined at the supersector × Census region level. The canonical cell grid consists of 11 supersectors × 4 Census regions = 44 cells. This aggregation is coarse enough that residual misclassification is rare.

Within each cell $c$, continuing-units growth is computed from the frozen measurement panel restricted to pseudo-establishments in that cell. The same panel mechanics apply: matched pseudo-establishment observations within the frozen window, exits removed without contributing negative change.

Cells with fewer than 5 pseudo-establishments (configurable via `MIN_PSEUDO_ESTABS_PER_CELL`) are excluded from the composite. Their weight is redistributed using a cascading priority: (1) same supersector, (2) same region, (3) uniform across all covered cells.

### Representativeness-Corrected National Composite

The provider's national signal entering the measurement equation is a weighted composite of cell-level signals:

$$y_{p,t}^{G} = \sum_{c} w_{c,t}^{QCEW} \cdot y_{p,c,t}^{G}$$

where the summation is over all cells $c$ with non-missing provider coverage. QCEW employment weights $w_{c,t}^{QCEW}$ are updated as new QCEW vintages become available, with carry-forward when provider months extend beyond the QCEW frontier. For cells where a provider has no coverage, the weight is redistributed proportionally across covered cells within the same supersector or region.

This composite removes the first-order composition bias: a provider concentrated in healthcare no longer overstates national growth when healthcare outperforms. The measurement equation retains its existing form — one signal per provider loading on a single national latent state — but the signal is now approximately representative of national employment dynamics rather than the provider's idiosyncratic client portfolio.

### Residual Composition Effects

Representativeness correction at the supersector × Census region level does not eliminate all composition effects. Within-cell heterogeneity (e.g., size class skew, sub-industry concentration) can still create provider-specific deviations from national growth. These residual effects are absorbed by the provider-specific bias $\alpha_{p}$ and noise $\sigma_{G,p}$ in the measurement equation. The correction reduces the time-varying component of composition bias — the part that matters for nowcasting accuracy — while leaving the time-invariant component to the existing parameter structure.

------------------------------------------------------------------------

# Complete Model Specification

## Latent State: Continuing-Units Growth

Continuing-units employment growth follows a mean-reverting AR(1) with era-specific parameters. Three structural eras partition the sample at distinct macroeconomic breakpoints:

| Era | Period | Interpretation |
|:----------------|:----------------|:-------------------------------------|
| 0 1 2 | 2003-01 → 2008-12 2009-01 → 2019-12 2020-01 → present | Pre-GFC Post-GFC expansion Post-COVID |

The latent state dynamics are:

$$\mu_{t}^{cont} = \mu_{g,e(t)} + \phi_{e(t)} \left( \mu_{t-1}^{cont} - \mu_{g,e(t)} \right) + \sigma_{\eta} \epsilon_{t}, \quad \epsilon_{t} \sim N(0,1)$$

where $e(t)$ is the era index for period $t$. Each era has its own mean growth rate $\mu_{g,e}$ and persistence $\phi_{e}$, while the innovation scale $\sigma_{\eta}$ is shared across eras.

Mean reversion is empirically supported: employment growth does not persist indefinitely in either direction. The era-specific specification allows the model to capture structurally different growth regimes — the post-GFC secular stagnation period had lower trend growth than the pre-GFC period, and the post-COVID recovery exhibited distinct dynamics.

When era-specific parameters are disabled (fallback mode), the model uses a single $\mu_g$ and $\phi$ across all periods, recovering the simpler AR(1) specification.

## Structural Birth/Death

Total employment growth decomposes into continuing-units growth and a birth/death offset:

$$\mu_{t} = \mu_{t}^{cont} + BD_{t}$$

The birth/death component captures net establishment entry minus exit — the systematic gap between what payroll providers measure (continuing units) and what official NFP reports (total employment including births and deaths).

The BD component is modeled as a time-varying structural process combining supply-side and demand-side information:

$$BD_{t} = \phi_{0} + \phi_{1} X_{t}^{birth} + \phi_{2} BD_{t-L}^{QCEW} + \phi_{3} \cdot X_{t}^{cycle} + \sigma_{BD} \xi_{t}, \quad \xi_{t} \sim N(0,1)$$

where:

-   $X_{t}^{birth}$ is the centered composite provider birth rate (averaged across providers that report it; centered so $\phi_{0}$ absorbs the mean)
-   $BD_{t-L}^{QCEW}$ is the centered QCEW-based BD proxy ($g^{QCEW} - \bar{g}^{PP}$), lagged by $L = 6$ months (QCEW publication delay)
-   $X_{t}^{cycle}$ is a vector of centered, standardized cyclical indicators (unemployment claims, NFCI, Census business applications, JOLTS openings)
-   $\phi_{3}$ is a vector of loadings, one per cyclical indicator
-   $\phi_{0}$ represents mean BD contribution at average covariate values

Where covariates are unavailable (early sample, nowcast horizon, missing provider birth rate data), centered covariate values are set to zero and $BD_{t}$ collapses gracefully to $\phi_{0} + \sigma_{BD} \xi_{t}$.

The provider birth rate captures supply-side formation dynamics observed in real time. The QCEW proxy anchors BD to census-quality data with a lag. Cyclical indicators capture demand-side forces that drive BD at turning points — the primary source of systematic CES error.

## Seasonality: Truncated Fourier Expansion with Time-Varying Amplitude

Employment seasonality is modeled as a truncated Fourier expansion with $K = 4$ harmonics and annually-evolving amplitudes:

$$s_{t} = \sum_{k=1}^{K} \left[ A_{k}(y(t)) \cos\left(\frac{2\pi k \, m(t)}{12}\right) + B_{k}(y(t)) \sin\left(\frac{2\pi k \, m(t)}{12}\right) \right]$$

where $m(t) \in \{0, 1, \ldots, 11\}$ is the month index and $y(t)$ is the year index.

The Fourier coefficients evolve annually via a Gaussian random walk. PyMC's `GaussianRandomWalk` steps along the year axis. The coefficients are organized as a $(2K, n_{years})$ matrix where rows $0 \ldots K-1$ are the $A_k$ coefficients and rows $K \ldots 2K-1$ are the $B_k$ coefficients:

$$A_{k}(y) = A_{k}(y-1) + \omega_{A,k}(y), \quad \omega_{A,k} \sim N(0, \sigma_{\omega,k}^{2})$$

$$B_{k}(y) = B_{k}(y-1) + \omega_{B,k}(y), \quad \omega_{B,k} \sim N(0, \sigma_{\omega,k}^{2})$$

The innovation variance $\sigma_{\omega,k}$ is shared across $A_k$ and $B_k$ for the same harmonic and decreases with harmonic number, reflecting the prior that broad seasonal swings (annual cycle) evolve more than fine-grained higher-frequency artifacts.

With $K = 4$, the expansion uses $2K = 8$ free parameters per year (versus 11 for unconstrained monthly effects) while capturing the dominant patterns: annual cycle (spring hiring, holiday retail), semi-annual, quarterly, and bi-monthly.

## Composite Growth Signals

The latent state produces three composite signals consumed by different measurement equations:

$$g_{t}^{cont,NSA} = \mu_{t}^{cont} + s_{t}$$

$$g_{t}^{total,SA} = \mu_{t}^{cont} + BD_{t}$$

$$g_{t}^{total,NSA} = \mu_{t}^{cont} + BD_{t} + s_{t}$$

## QCEW Measurement Equation — Truth Anchor

QCEW observations are modeled as high-precision measurements of total NSA growth, with estimated noise differentiated by filing timing and per-observation revision multipliers:

$$y_{t}^{QCEW} \sim N\left(g_{t}^{total,NSA}, \; \sigma_{QCEW}^{2}(t) \right)$$

$$\sigma_{QCEW}(t) = \sigma_{QCEW,tier(t)} \cdot m_{rev}(t)$$

where $\sigma_{QCEW,tier(t)}$ is the estimated base noise for the observation's filing tier and $m_{rev}(t) \geq 1$ is a revision-specific noise multiplier from the publication calendar.

Two tiers are distinguished based on within-quarter filing position:

-   **M2 (mid-quarter months: Feb, May, Aug, Nov):** Intermediate months reconstructed from recent payroll records. Base noise is $\sigma_{QCEW,mid}$.
-   **M3+M1 (boundary months: quarter-end and quarter-start):** Month 3 is filed contemporaneously; Month 1 is the most retrospective. Base noise is $\sigma_{QCEW,boundary}$.

The base noise parameters use LogNormal priors to avoid the funnel geometry that arises when HalfNormal collapses toward zero (QCEW precision can be extreme relative to other sources):

$$\log \sigma_{QCEW,mid} \sim N(\ln(0.0005), \; 0.3)$$

$$\log \sigma_{QCEW,boundary} \sim N(\ln(0.002), \; 0.5)$$

This gives approximate 90% prior intervals of $[0.0003, 0.0008]$ for mid-quarter months and $[0.0009, 0.0046]$ for boundary months.

The per-observation revision multiplier $m_{rev}(t)$ scales base noise upward for earlier QCEW vintages that are subject to revision. This is drawn from the QCEW revision history available from 2017Q1 and encoded in the revision schedule lookup. Later revisions receive multipliers closer to 1.0.

This is a **conditioning** approach (high-precision observation) rather than hard benchmarking (constraining the state to match QCEW exactly). Conditioning maintains probabilistic coherence while effectively pinning the latent state to QCEW at observed dates.

## CES Measurement Equations — Vintage-Specific

Each CES print (first, second, final) enters as a separate observation with vintage-specific noise, reflecting the declining sampling error as additional survey responses arrive. Both SA and NSA series are modeled with shared bias and signal loading but separate noise terms.

For vintage $v \in \{1, 2, 3\}$:

$$y_{t,v}^{CES,SA} \sim N\left(\alpha^{CES} + \lambda^{CES} \cdot g_{t}^{total,SA}, \; \sigma_{CES,SA,v}^{2} \right)$$

$$y_{t,v}^{CES,NSA} \sim N\left(\alpha^{CES} + \lambda^{CES} \cdot g_{t}^{total,NSA}, \; \sigma_{CES,NSA,v}^{2} \right)$$

where:

-   $\alpha^{CES}$ is CES-specific bias (absorbs scope differences, seasonal adjustment residuals)
-   $\lambda^{CES}$ is CES signal loading (should be near 1 if scales are aligned)
-   $\sigma_{CES,SA,v}$ and $\sigma_{CES,NSA,v}$ are vintage-specific observation noise

The first-print standard error is approximately 65,000 jobs for total nonfarm (\~0.04% in growth-rate space), declining with each subsequent closure. Vintage-specific noise parameters are calibrated from the CES revision data available from May 2003 at the national level by sector.

When the observation panel contains vintage-tagged CES data (revision numbers 0, 1, 2), each vintage enters as a separate likelihood term. When vintage data is unavailable, the final CES value is assigned to vintage 3 and vintages 1–2 are empty.

**Annual benchmark scope.** The February benchmark re-anchors the prior year's March employment level to QCEW-derived universe counts, with a wedge-back over the preceding 11 months. The total scope is 21 months of NSA data and 5 years of re-seasonalized SA data. Backtests using SA data must track re-seasonalization vintages, not just level revisions.

## Payroll Provider Measurement Equations — Config-Driven

Each provider $p$ gets independent measurement parameters with configurable error structure. Provider observations load on continuing-units NSA growth (not total growth), since payroll data excludes birth/death dynamics by construction. The input signal $y_{p,t}^{G}$ is the representativeness-corrected national composite described above — a QCEW-weighted average of cell-level continuing-units growth from pseudo-establishments at the supersector × Census region level.

**iid measurement error (default):**

$$y_{p,t}^{G} = \alpha_{p} + \lambda_{p} \cdot g_{t}^{cont,NSA} + \varepsilon_{p,t}, \quad \varepsilon_{p,t} \sim N(0, \sigma_{G,p}^{2})$$

**AR(1) measurement error** (for providers whose client base creates autocorrelated residuals, e.g., multi-establishment firms restructuring internally):

$$y_{p,t}^{G} \mid y_{p,t-1}^{G} \sim N\left(\mu_{p,t}^{base} + \rho_{p} \left(y_{p,t-1}^{G} - \mu_{p,t-1}^{base}\right), \; \sigma_{G,p}^{2} \right)$$

where $\mu_{p,t}^{base} = \alpha_{p} + \lambda_{p} \cdot g_{t}^{cont,NSA}$.

The marginal variance at $t = 0$ is $\sigma_{G,p}^{2} / (1 - \rho_{p}^{2})$, ensuring stationarity. The effective precision contribution of an AR(1) provider is reduced by the factor $(1 - \rho_{p}^{2})$ relative to an iid provider with the same noise level.

Because the input signal is representativeness-corrected, the bias parameter $\alpha_{p}$ primarily captures residual within-cell composition effects (size class skew, sub-industry concentration) rather than the provider's overall portfolio tilt. Similarly, the signal loading $\lambda_{p}$ reflects the provider's measurement fidelity rather than conflating fidelity with compositional similarity to the national distribution.

New providers require only a `ProviderConfig` entry specifying name, data file, error model (iid or AR(1)), and optional birth-rate file. The model, diagnostics, and plots adapt automatically.

------------------------------------------------------------------------

# Priors

## Latent State Dynamics (Era-Specific)

$$\mu_{g,e} \sim N(0.001, 0.005^{2}) \quad \text{for each era } e \in \{0, 1, 2\}$$

$$\phi_{e} \sim \text{Beta}(18, 2) \quad \text{for each era}$$

$$\sigma_{\eta} \sim \text{Half-N}(0, 0.005)$$

The $\text{Beta}(18, 2)$ prior on persistence concentrates mass near 0.9 (mode = 0.944, mean = 0.9), consistent with the near-unit-root dynamics reported in Cajner et al. (2019) ($\rho \approx 0.96$), while ruling out exact unit-root behavior. This is tighter than the Uniform$(0, 0.99)$ prior used in the non-era specification, reflecting the empirical regularity that employment growth is highly persistent.

When era-specific parameters are disabled, scalar priors apply:

$$\mu_{g} \sim N(0.001, 0.005^{2})$$

$$\phi \sim \text{Uniform}(0, 0.99)$$

## Birth/Death Parameters

$$\phi_{0} \sim N(0.001, 0.002^{2}) \quad \text{[mean BD contribution]}$$

$$\phi_{1} \sim N(0.5, 0.5^{2}) \quad \text{[birth rate loading; positive = procyclical]}$$

$$\phi_{2} \sim N(0.3, 0.3^{2}) \quad \text{[QCEW BD proxy persistence]}$$

$$\phi_{3} \sim N(0, 0.3^{2}) \quad \text{[cyclical indicator loadings; one per indicator]}$$

$$\sigma_{BD} \sim \text{Half-N}(0, 0.001)$$

The birth rate loading $\phi_1$ and QCEW BD proxy loading $\phi_2$ are only included when their respective covariates contain non-zero data. The cyclical indicator loadings $\phi_3$ are a vector of length $n_{cyclical}$ (up to 4), one per available indicator, also conditional on non-zero data.

## Seasonal Parameters

$$A_{k}(0), B_{k}(0) \sim N(0, 0.015^{2}) \quad \text{for } k = 1, \ldots, K$$

$$\sigma_{\omega,k} \sim \text{Half-N}(0, 0.005/k) \quad \text{[tighter for higher harmonics]}$$

The initial distributions are set by PyMC's `GaussianRandomWalk` `init_dist` parameter; the innovation standard deviations decrease as $1/k$ so that higher harmonics evolve more slowly.

## QCEW Noise Parameters

$$\log \sigma_{QCEW,mid} \sim N(\ln(0.0005), \; 0.3)$$

$$\log \sigma_{QCEW,boundary} \sim N(\ln(0.002), \; 0.5)$$

These LogNormal priors are parameterized to give approximate prior medians of 0.0005 and 0.002, with wider uncertainty on boundary months where the M3+M1 aggregation introduces more heterogeneity.

## CES Parameters

$$\alpha^{CES} \sim N(0, 0.005^{2})$$

$$\lambda^{CES} \sim N(1.0, 0.15^{2})$$

$$\sigma_{CES,SA,v} \sim \text{InverseGamma}(3, 0.004) \quad \text{for each vintage } v \in \{1, 2, 3\}$$

$$\sigma_{CES,NSA,v} \sim \text{InverseGamma}(3, 0.004) \quad \text{for each vintage } v \in \{1, 2, 3\}$$

The InverseGamma prior keeps $\sigma$ away from zero (mode $\approx 0.001$), preventing the Neal's funnel pathology while allowing data to pull toward larger values for noisier vintages.

## Per-Provider Parameters

$$\alpha_{p} \sim N(0, 0.005^{2})$$

$$\lambda_{p} \sim N(1.0, 0.15^{2})$$

$$\sigma_{G,p} \sim \text{InverseGamma}(3, 0.004)$$

$$\rho_{p} \sim \text{Beta}(2, 3) \quad \text{[AR(1) providers only; mode } \approx 0.25 \text{, allows up to } \sim 0.8 \text{]}$$

------------------------------------------------------------------------

# Data Pipeline

## Observation Panel Schema

All data sources are unified into a single observation panel with a standardized schema containing columns for period, source, geographic and industry identifiers, growth rates, employment levels, vintage dates, and revision numbers. The panel is the single interface between data ingestion and model building.

## Vintage Tracking and Censoring

The panel carries a `vintage_date` column for each observation, enabling leakage-safe backtesting. When `panel_to_model_data()` is called with an `as_of` date, observations whose `vintage_date` exceeds `as_of` are dropped before growth-rate extraction.

Provider data censoring applies a 3-week publication lag: observations for reference months whose publication date exceeds `as_of` are masked. Cyclical indicator censoring uses per-indicator publication lags (1 month for claims and NFCI, 2 months for BFS and JOLTS) to zero out unavailable observations.

Historical publication dates are sourced from a `vintage_dates.parquet` file built by scraping BLS release date archives. When exact dates are unavailable (pre-2017 QCEW, early CES), lag-based heuristics provide approximate dates.

## CES Vintage Series Construction

CES observations are split into three vintage series corresponding to the first print (revision 0), second print (revision 1), and final print (revision 2). Each vintage enters the model as a separate likelihood term with its own noise parameter.

When the panel contains vintage-tagged data, each vintage series is extracted independently. When vintage data is unavailable (e.g., legacy data without revision tracking), the final CES value is assigned to vintage 3 and vintages 1–2 are left empty.

## BD Covariate Construction

The birth rate covariate is constructed by averaging birth rates across all providers that report them, restricted to the provider-covered window to avoid NaN propagation. The QCEW BD proxy is computed as $g^{QCEW} - \bar{g}^{PP}$ and lagged by 6 months. Both covariates are centered (mean-subtracted) so their zero values (used for missing periods) represent "at average."

Cyclical indicators are loaded from per-indicator parquet files in `data/indicators/`, aggregated to monthly frequency for weekly series, and standardized (zero-mean, unit-variance). Missing files are gracefully skipped.

------------------------------------------------------------------------

# Forecast Production

1.  **Historical Estimation:** Run MCMC on all data through $T - 1$.
2.  **BD Forecast:** Estimate $BD_{T}$ using cyclical indicators, provider birth rates, and lagged QCEW BD proxy.
3.  **Continuing-Units Filtering:** Filter $\mu_{T}^{cont}$ from all payroll provider data for current month $T$.
4.  **Nowcast:** Posterior predictive distribution for $\mu_{T} = \mu_{T}^{cont} + BD_{T}$.

Forward simulation draws trajectories from the posterior, propagating the AR(1) continuing-units growth (using the last era's parameters), the structural BD offset (with QCEW lag error-correction), and last-year Fourier seasonal coefficients. Outputs include index-level forecasts (SA and NSA) with 80% HDI fan charts and month-over-month jobs-added estimates.

------------------------------------------------------------------------

# Benchmark Revision Prediction

The model's posterior over `g_total_nsa` implies a CES benchmark revision as a deterministic transformation — no additional MCMC is needed. The revision is the cumulative gap between the model's QCEW-anchored latent growth and observed CES NSA growth over the 12-month benchmark window (April Y−1 through March Y):

$$\hat{R}_Y = L_{Y-1}^{March} \cdot \left( \exp\left(\sum_{t \in W_Y} g_t^{total,NSA}\right) - \exp\left(\sum_{t \in W_Y} g_t^{CES,NSA}\right) \right)$$

where $L_{Y-1}^{March}$ is the CES NSA employment level at March Y−1 (in thousands) and $W_Y$ denotes the benchmark window indices.

The decomposition separates "CES sample is off on continuing-units growth" from "CES birth/death model is wrong":

-   **Continuing-units divergence:** $L \cdot (\exp(\sum g^{cont,NSA} + s) - \exp(\sum g^{CES,NSA}))$
-   **BD accumulation:** $L \cdot (\exp(\sum BD) - 1)$

The BD term should dominate; large continuing-units divergence may signal provider representativeness issues.

------------------------------------------------------------------------

# Output Specification

| Output | Description |
|:----------------------------|:------------------------------------------|
| `nfp_nowcast` | Point estimate (posterior mean of $\mu_{T}$) |
| `nfp_nowcast_std` | Uncertainty (posterior std of $\mu_{T}$) |
| `nfp_nowcast_ci` | 80% and 95% credible intervals |
| `g_cont` | Continuing-units growth trajectory |
| `bd` | Structural BD trajectory |
| `seasonal` | Fourier seasonal component |
| `g_total_sa` | Total SA growth trajectory |
| `g_total_nsa` | Total NSA growth trajectory |
| `provider_signal_quality` | Posterior mean of $\lambda_{p}$ for each provider |
| `provider_bias` | Posterior mean of $\alpha_{p}$ for each provider |
| `provider_weights` | Effective weight (precision share) of each provider in nowcast |
| `precision_budget` | Share of total precision from QCEW, CES (by vintage), each provider |
| `benchmark_revision` | Implied benchmark revision posterior (thousands of jobs) |

------------------------------------------------------------------------

# Diagnostics

-   Standardized residuals by source (QCEW, CES SA/NSA by vintage, each provider)
-   Posterior predictive checks: test statistics (skewness, lag-1 ACF) against replicated data
-   LOO-CV with Pareto $\hat{k}$ diagnostics flagging influential observations
-   QCEW sigma sensitivity: parameter stability across $0.5\times / 1\times / 2\times$ noise scales
-   Precision budget decomposition across noise scale configurations
-   Expanding-window out-of-sample backtest against NFP releases
-   Representativeness correction impact: comparison of provider residual dispersion with and without QCEW-weighted compositing, stratified by periods of high vs. low sectoral growth dispersion
-   Benchmark revision decomposition (continuing-units divergence vs. BD accumulation)
-   Sampler convergence: $\hat{R}$, ESS, divergences

# Validation Metrics

-   Out-of-sample RMSE against NFP releases (by CES vintage: first, second, final, benchmarked)
-   Coverage of credible intervals (80% and 95%)
-   Comparison to naive forecast (random walk on NFP)
-   Comparison to post-2026 BLS NBDM (with WLR-regression improvement)
-   Provider-specific signal quality rankings
-   BD component accuracy against annual benchmark revisions

------------------------------------------------------------------------

# Sampler Configuration

The default sampler is nutpie (Rust/C++ NUTS implementation, optimized for Apple Silicon). Falls back to PyMC's built-in NUTS sampler when nutpie is unavailable. Three configurations are available:

| Config  | Draws | Tune  | Chains | Target Accept | Use Case    |
|:--------|:------|:------|:-------|:--------------|:------------|
| DEFAULT | 8,000 | 6,000 | 4      | 0.97          | Production  |
| MEDIUM  | 4,000 | 3,000 | 4      | 0.95          | Sensitivity |
| LIGHT   | 2,000 | 2,000 | 2      | 0.95          | Backtesting |

# Config-Driven Provider Architecture

Adding a new provider requires only a `ProviderConfig` entry:

```         
ProviderConfig(
    name='PP2',
    file='providers/pp2/pp2_provider.parquet',
    error_model='ar1',  # or 'iid'
    birth_file='providers/pp2/pp2_births.parquet',
)
```

Cell-level providers supply a supersector × Census-region parquet; the loader detects this from the `geographic_type='region'` field and applies QCEW-weighted compositing automatically. National providers are used directly. The model, diagnostics, sensitivity analysis, and plots adapt automatically.

------------------------------------------------------------------------

# Limitations

-   No geographic or industry decomposition (addressed in Releases 2–3).
-   Cannot explain drivers of national forecast.
-   Static provider bias (no drift accommodation in Release 1).
-   Representativeness correction removes first-order composition bias but within-cell heterogeneity (size class skew, sub-industry concentration) can still create residual provider-specific deviations that vary with the business cycle.
-   QCEW data treated as final for 2003–2016 (pre-revision-history), introducing mild lookahead in backtesting over that period.

------------------------------------------------------------------------

# Appendix A: CES Birth/Death Model and System Value Proposition

The CES birth/death model (NBDM) operates in two steps: (1) excluding zero-employment reports from the matched sample, implicitly imputing dead establishments with continuers' growth rate; (2) adding an ARIMA-based forecast of the residual between actual and imputed birth/death dynamics, trained on 5 years of QCEW microdata from the Longitudinal Database. Updated quarterly.

The NBDM's fundamental vulnerability is inability to anticipate turning points — the ARIMA projects historical seasonal patterns forward, systematically overcounting employment during decelerations. Three consecutive benchmark misses totaling \~1.8 million jobs (−266K in 2023, −598K in 2024, −911K preliminary in 2025) demonstrate the failure mode.

Starting January 2026, BLS permanently incorporated the sample Weighted Link Relative (WLR) as a regression covariate in the NBDM, which internal research (Grieves, Mance, Witt 2023) showed nearly halved forecast RMSE. The `alt_nfp` system benchmarks its BD performance against this improved post-2026 BLS baseline — not the pre-2026 NBDM.

The system's comparative advantage lies in combining real-time provider-derived formation signals with cyclical indicators and QCEW anchoring to detect BD drift faster than the improved NBDM can self-correct. The structural BD model observes provider birth rates in real time (the NBDM does not use private payroll data) and conditions on QCEW as it becomes available (the NBDM updates quarterly but does not perform real-time Bayesian updating).

# Appendix B: Multi-Provider Design Rationale

## Why Multi-Provider from the Start?

1.  **No retrofitting:** Adding providers later requires restructuring; building it in from Release 1 is cleaner.

2.  **Graceful degradation:** With one provider, independent priors collapse to weakly informative; the framework works but doesn't overcomplicate.

3.  **Immediate benefits:** Even with one provider, the framework reveals provider-specific signal quality metrics and precision budget.

4.  **Provider comparison:** When multiple providers exist, the model automatically learns relative strengths by posterior precision weighting.

# Appendix C: Implementation Notes

## Non-Centered Parameterization

The latent state uses a non-centered parameterization via innovation variables $\epsilon_{t} \sim N(0,1)$:

```         
eps_g ~ N(0, 1, shape=T)
g_cont[t] = mu_g + phi * (g_cont[t-1] - mu_g) + sigma_g * eps_g[t]
```

This avoids the funnel pathology where the sampler struggles to simultaneously estimate the group mean, group variance, and individual effects. The same pattern applies to the BD innovations $\xi_t$.

## AR(1) via PyTensor Scan

The AR(1) latent state is implemented using PyTensor's `scan` operation, which compiles the sequential computation into an efficient loop. The era-specific version sequences over `eps_g[1:]`, `mu_g[1:]`, and `phi[1:]` simultaneously.

## Computational Scaling

With a single provider, the model has approximately 15 + 5×P parameters (P = number of providers) plus $2K \times n_{years}$ Fourier coefficients and $T$ latent state innovations. Typical runtime is minutes on Apple Silicon with nutpie.