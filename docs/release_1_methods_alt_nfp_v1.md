# Alt-NFP Model: Methods Reference

## 1. Overview

Alt-NFP is a Bayesian state space model for nowcasting U.S. total nonfarm employment growth. The model fuses three classes of data sources — the Quarterly Census of Employment and Wages (QCEW), the Current Employment Statistics survey (CES), and private payroll provider indices — into a unified latent growth process. Inference is performed via Hamiltonian Monte Carlo (NUTS) in PyMC.

All quantities are expressed as log-differences of employment levels (approximately month-over-month percent changes). The sample begins in January 2012.

------------------------------------------------------------------------

## 2. Data Sources

### 2.1 QCEW (truth anchor)

The Quarterly Census of Employment and Wages is a near-census of U.S. employers derived from unemployment insurance tax records. It covers approximately 95% of nonfarm employment. QCEW data are not seasonally adjusted (NSA only) and are published quarterly with a 5–8 month lag.

-   **Frequency**: Monthly observations published in quarterly batches.
-   **Revisions**: Quarter-dependent revision structure. Q1 data are revised up to 4 times; Q4 data are revised once. Each revision has an empirically calibrated noise multiplier (see Section 3.5).
-   **Role in model**: Serves as the primary truth anchor with the tightest observation noise among all sources.

### 2.2 CES (monthly survey)

The Current Employment Statistics survey is a monthly establishment survey covering roughly 145,000 businesses and government agencies representing approximately 700,000 individual worksites. Published by BLS one month after the reference period. The triangular revision archive (`cesvinall.zip`) provides data back to 2003; the model estimation window begins in January 2012.

-   **Frequency**: Monthly, both seasonally adjusted (SA) and not seasonally adjusted (NSA).
-   **Revisions**: Three sample-based vintages — first print (rev 0, +1 month lag), second estimate (rev 1, +2 months), third estimate (rev 2, +3 months). Benchmark-revised data (rev −1, incorporating QCEW universe counts) are never used as CES observations. Benchmark-quality information enters the model exclusively through QCEW. The `benchmark_revisions.py` lookup and `benchmark.py`/`benchmark_backtest.py` modules are post-hoc diagnostics only — they compare posterior inferences to actual BLS benchmarks but do not feed into the model.
-   **Best-available selection**: One observation per month per SA/NSA series, using the highest available revision number in {0, 1, 2}.

### 2.3 Private payroll providers

Payroll processor data provide high-frequency employment indices from administrative records. Currently one active provider (G), supplying employment at the national level.

-   **Compositing**: Cell-level data are aggregated to a national growth series via QCEW-weighted compositing. QCEW employment shares by cell serve as population weights; cells with fewer than 5 pseudo-establishments are excluded and their weight is redistributed (same supersector first, then same region, then uniform).
-   **Birth rates**: Provider-specific business birth rates are supplied in a separate national-level file and enter the structural birth/death component.
-   **Publication lag**: \~3 weeks after reference period.

### 2.4 Cyclical indicators

Two demand-side indicators from FRED are used as covariates in the birth/death component:

| Indicator                    | FRED ID | Frequency             | Publication lag |
|-----------------|-----------------|-----------------|----------------------|
| Initial jobless claims (NSA) | ICNSA   | Weekly → monthly mean | \~1 month       |
| JOLTS job openings           | JTSJOL  | Monthly               | \~2 months      |

Indicators are standardized (centered and scaled by standard deviation) before entering the model. Under as-of censoring (backtests), indicators are masked according to their publication lags.

------------------------------------------------------------------------

## 3. Model Specification

### 3.1 Latent continuing-units growth: AR(1) with era-specific means

The latent month-over-month continuing-establishment employment growth rate follows a stationary AR(1) process with era-indexed unconditional means:

$$
g_t^{cont} = \mu_{g,e(t)} + \phi \left( g_{t-1}^{cont} - \mu_{g,e(t)} \right) + \sigma_g \, \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,1)
$$

where $e(t) \in \{0, 1\}$ indexes two eras (Pre-COVID: 2012–2019, Post-COVID: 2020–present), with break at January 2020.

**Reparameterization**: The innovation standard deviation $\sigma_g$ is derived from the marginal (stationary) standard deviation $\tau$ via $\sigma_g = \tau \sqrt{1 - \phi^2}$. This breaks the $\phi$–$\sigma_g$ ridge that otherwise causes poor effective sample size.

**Priors**:

| Parameter | Prior | Notes |
|------------------------------|---------------------|---------------------|
| $\mu_{g,e}$ | $\mathcal{N}(0.001, 0.005)$ | Per-era, shape $(N_{eras},)$ |
| $\phi_{raw}$ | $\text{Beta}(18, 2)$ | Mode ≈ 0.89; $\phi = \min(\phi_{raw}, 0.99)$ |
| $\tau$ | $\text{LogNormal}(\ln 0.013, 0.5)$ | Marginal SD; median ≈ 0.013 |

### 3.2 Fourier seasonal component

Seasonality is captured by a truncated Fourier expansion with time-varying (annually-evolving) coefficients:

$$
s_t = \sum_{k=1}^{K} \left[ A_k(y_t) \cos\!\left(\frac{2\pi k \, m_t}{12}\right) + B_k(y_t) \sin\!\left(\frac{2\pi k \, m_t}{12}\right) \right]
$$

where $m_t$ is the zero-indexed month of observation $t$, $y_t$ is the year index, $K = 4$ harmonics, and the Fourier coefficients evolve as independent Gaussian random walks across years:

$$
A_k(y) = A_k(y-1) + \sigma_{F,k} \, \eta_k^A(y), \qquad B_k(y) = B_k(y-1) + \sigma_{F,k} \, \eta_k^B(y)
$$

**Priors**:

| Parameter | Prior |
|--------------------------------------------|----------------------------|
| $\sigma_{F,k}$ | $\text{LogNormal}(\ln 0.0003 - \ln k,\; 0.5)$ for $k = 1, \ldots, 4$ |
| $A_k(0), B_k(0)$ | $\mathcal{N}(0, 0.015)$ (GRW init) |

The $1/k$ scaling in log-space shrinks innovation variance for higher harmonics.

### 3.3 Structural birth/death component

The net birth/death contribution captures establishment entry and exit not reflected in continuing-units data:

$$
bd_t = \varphi_0 + \varphi_3^\top X_t^{cycle} + \sigma_{bd} \, \xi_t, \qquad \xi_t \sim \mathcal{N}(0,1)
$$

where $X_t^{cycle}$ is a vector of centered cyclical indicators (claims, JOLTS). When an indicator is unavailable (e.g., during real-time censoring), its centered value is set to zero and $bd_t$ collapses to $\varphi_0 + \sigma_{bd} \xi_t$.

**Priors**:

| Parameter     | Prior                                          |
|---------------|------------------------------------------------|
| $\varphi_0$   | $\mathcal{N}(0.001, 0.002)$                    |
| $\varphi_3$   | $\mathcal{N}(0, 0.3)$, shape $(n_{cyclical},)$ |
| $\sigma_{bd}$ | $\text{LogNormal}(\ln 0.003, 0.5)$             |

### 3.4 Composite growth signals

The model constructs three composite growth signals from the latent components:

| Signal | Definition | Used by |
|---------------------|----------------------------|-----------------------|
| $g_t^{cont,NSA} = g_t^{cont} + s_t$ | Continuing-units growth, NSA | Provider likelihoods |
| $g_t^{total,SA} = g_t^{cont} + bd_t$ | Total growth, SA | CES SA likelihood |
| $g_t^{total,NSA} = g_t^{cont} + s_t + bd_t$ | Total growth, NSA | QCEW & CES NSA likelihoods |

### 3.5 QCEW likelihood

$$
y_t^{QCEW} \sim \text{Student-}t\!\left(\nu,\; g_t^{total,NSA},\; \sigma_t^{QCEW}\right)
$$

with fixed degrees of freedom $\nu = 5$, providing robustness to non-Gaussian error from NAICS reclassification and payroll-timing mismatches. The Student-t reduces per-observation precision by a factor of $(\nu+1)/(\nu+3) = 0.75$ relative to a Normal.

**Per-observation noise**:

$$
\sigma_t^{QCEW} = \sigma^{base}_{tier(t)} \times m_t^{rev} \times m_t^{era}
$$

where:

-   **Base sigma by tier**: Two estimated parameters — $\sigma^{mid}$ for M2 months (Feb, May, Aug, Nov) and $\sigma^{boundary}$ for M3+M1 months.
-   **Revision multiplier** $m_t^{rev}$: Empirically calibrated per quarter × revision number (range 1.0–25.0; see revision schedule below).
-   **Era multiplier** $m_t^{era}$: Inflates boundary-month noise post-COVID (rev 0: ×5.0, rev 1: ×3.5, rev 2: ×2.0; M2 months are era-invariant).

**Priors**:

| Parameter           | Prior                                |
|---------------------|--------------------------------------|
| $\sigma^{mid}$      | $\text{LogNormal}(\ln 0.0005, 0.15)$ |
| $\sigma^{boundary}$ | $\text{LogNormal}(\ln 0.002, 0.5)$   |

The tightened M2 prior ($\text{SD} = 0.15$) keeps the posterior near the 0.05%/month center and prevents QCEW precision from dominating all other sources.

### 3.6 CES likelihood

One observation per month per SA/NSA series (best-available print). The two series share bias and loading but have separate, vintage-indexed observation noise:

$$
y_t^{CES,SA} \sim \mathcal{N}\!\left(\alpha_{CES} + \lambda_{CES} \cdot g_t^{total,SA},\; \sigma_{CES,v(t)}^{SA}\right)
$$ $$
y_t^{CES,NSA} \sim \mathcal{N}\!\left(\alpha_{CES} + \lambda_{CES} \cdot g_t^{total,NSA},\; \sigma_{CES,v(t)}^{NSA}\right)
$$

where $v(t) \in \{0, 1, 2\}$ is the revision number of the best-available print for month $t$ (first print, second estimate, or third estimate). Each SA/NSA series has its own sigma vector of shape $(n_{vintages},)$, allowing the model to assign higher noise to earlier prints. In the full (non-backtest) panel, most months have rev 2 available; only the most recent 1–2 months use rev 0 or rev 1.

**Priors**:

| Parameter | Prior |
|--------------------------------------------|----------------------------|
| $\alpha_{CES}$ | $\mathcal{N}(0, 0.005)$ |
| $\lambda_{CES}$ | $\text{TruncatedNormal}(1.0, 0.1, \text{lower}=0.5)$ |
| $\sigma_{CES,v}^{SA}$ | $\text{LogNormal}(\ln 0.002, 0.5)$, shape $(n_{vintages},)$ |
| $\sigma_{CES,v}^{NSA}$ | $\text{LogNormal}(\ln 0.002, 0.5)$, shape $(n_{vintages},)$ |

### 3.7 Provider likelihoods

For each provider $p$ with observations:

$$
y_t^{(p)} \sim \mathcal{N}\!\left(\mu_t^{(p)},\; \sigma_t^{(p)}\right)
$$

The signal mean depends on the error model:

**IID error model**: $$
\mu_t^{(p)} = \alpha_p + \lambda_p \cdot g_t^{cont,NSA}
$$

**AR(1) error model**: $$
\mu_t^{(p)} = \alpha_p + \lambda_p \cdot g_t^{cont,NSA} + \rho_p \left(y_{t-1}^{(p)} - \mu_{t-1}^{(p)}\right)
$$

with conditional variance $\sigma_p / \sqrt{1 - \rho_p^2}$ at $t=0$ and $\sigma_p$ thereafter.

Provider likelihoods load on continuing-units growth $g_t^{cont,NSA}$ (not total growth), since providers do not observe net business birth/death.

**Priors**:

| Parameter           | Prior                             |
|---------------------|-----------------------------------|
| $\alpha_p$          | $\mathcal{N}(0, 0.005)$           |
| $\lambda_p$         | $\mathcal{N}(1.0, 0.15)$          |
| $\sigma_p$          | $\text{InverseGamma}(3.0, 0.004)$ |
| $\rho_p$ (AR1 only) | $\text{Beta}(2, 3)$               |

------------------------------------------------------------------------

## 4. QCEW Revision Noise Schedule

Empirically calibrated from 2017+ data (excluding 2020–2021). Multipliers are applied to the base sigma for each observation.

| Quarter | Rev 0 | Rev 1 | Rev 2 | Rev 3 | Rev 4 |
|---------|-------|-------|-------|-------|-------|
| Q1      | 25.0  | 20.0  | 15.0  | 5.0   | 1.0   |
| Q2      | 17.0  | 4.0   | 1.2   | 1.0   | —     |
| Q3      | 17.0  | 4.0   | 1.0   | —     | —     |
| Q4      | 17.0  | 1.0   | —     | —     | —     |

Q1 shows larger revision RMSE through rev 3 due to January reporting patterns. The asymmetric revision count arises from the annual QCEW file structure.

------------------------------------------------------------------------

## 5. Inference

### 5.1 Sampler

The preferred sampler is **nutpie** (Rust/C++ NUTS implementation, performant on Apple Silicon). Falls back to PyMC's built-in NUTS when nutpie is unavailable.

### 5.2 Sampling configuration

| Setting       | Production | Backtest |
|---------------|------------|----------|
| Draws         | 4,000      | 2,000    |
| Tune          | 3,000      | 2,000    |
| Chains        | 4          | 2        |
| Target accept | 0.95       | 0.95     |

### 5.3 Computational diagnostics

Standard HMC diagnostics are applied: $\hat{R} < 1.01$, effective sample size, and divergent transition checks.

------------------------------------------------------------------------

## 6. Forecast

Forward simulation propagates the AR(1) latent growth, structural BD offset, and the most recent year's Fourier seasonal coefficients:

1.  **AR(1) growth**: $g_{T+h}^{cont} = \mu_{g,E} + \phi(g_{T+h-1}^{cont} - \mu_{g,E}) + \sigma_g \varepsilon_{T+h}$, using the Post-COVID era mean $\mu_{g,E}$.
2.  **Birth/death**: $bd_{T+h} = \varphi_0 + \sigma_{bd} \xi_{T+h}$ (cyclical indicators not projected forward).
3.  **Seasonal**: Evaluated at forecast months using the last estimated year's Fourier coefficients.
4.  **Composite**: $g_{T+h}^{SA} = g_{T+h}^{cont} + bd_{T+h}$; $g_{T+h}^{NSA} = g_{T+h}^{cont} + s_{T+h} + bd_{T+h}$.

Index levels are reconstructed by exponentiating cumulated log-differences. Jobs-added estimates are obtained by scaling the index by the CES employment level at the base period.

------------------------------------------------------------------------

## 7. Backtest

The nowcast backtest evaluates real-time predictive performance over a rolling window (default 24 months). For each target month $T$:

1.  **Panel construction**: `build_panel(as_of_ref=T)` applies rank-based horizon censoring:
    -   CES: rank 1 → rev 0, rank 2 → rev 1, rank 3+ → rev 2 (benchmark-revised rows are never selected).
    -   QCEW: quarter-dependent maximum revision ($Q1: 4, Q2: 3, Q3: 2, Q4: 1$).
2.  **Indicator censoring**: Cyclical indicators are masked beyond their publication lags relative to $T$.
3.  **Provider censoring**: Provider data with `ref_date + 3 weeks > T` are excluded.
4.  **Sampling**: Light configuration (2 chains, 2,000 draws) for computational tractability.

Output: per-iteration InferenceData (NetCDF) and summary metrics (Parquet).

------------------------------------------------------------------------

## 8. Precision Budget

The diagnostic precision budget quantifies each source's information contribution:

$$
\text{share}_i = \frac{\text{precision}_i}{\sum_j \text{precision}_j}
$$

Fisher information formulas used:

| Source | Precision formula |
|------------------------|------------------------------------------------|
| CES | $\lambda_{CES}^2 / \sigma_{CES,v}^2$ per-obs (Normal) |
| QCEW | $(\nu+1) / [(\nu+3) \cdot (\sigma_t^{QCEW})^2]$ (Student-t) |
| Provider | $\lambda_p^2 / \sigma_p^2$ (with AR1 adjustment where applicable) |

QCEW observations are split into M2 and M3+M1 tiers for reporting. Results are reported by era window using `ERA_BREAKS`.