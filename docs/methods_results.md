# QCEW-Anchored Growth-Rate Model: Methodology and Results (v2 — Gold Standard)

## 1. Overview

This document describes the econometric methodology and empirical results for the QCEW-anchored Bayesian state space model implemented in `pp_estimation_v2.py` (now in `archive/`; the current v3 implementation is in `src/alt_nfp/`). This is the **gold standard** specification against which all subsequent model variants (v3+) should be evaluated. The gold-standard run uses a **fixed** birth/death offset, **Gaussian** observation errors, the nutpie sampler (8k draws, 6k tune), and reports 0 divergences.

The model extracts a latent national employment growth signal from four data sources—monthly QCEW (truth anchor), CES (SA and NSA, noisy observations), and two payroll provider (PP) continuing-units series from independent providers—operating entirely in monthly log growth rates.

**Core design principles:**

1.  **QCEW is truth.** The near-census Quarterly Census of Employment and Wages defines the measurement scale ($\alpha = 0$, $\lambda = 1$, fixed $\sigma$). This choice is motivated by planned extension to state $\times$ supersector estimation, where CES precision degrades but QCEW coverage remains scale-invariant.

2.  **CES is a noisy observer.** CES measures total employment with estimated bias ($\alpha_{\text{CES}}$), loading ($\lambda_{\text{CES}}$), and noise ($\sigma_{\text{CES}}$). The model learns how CES relates to truth rather than assuming it.

3.  **PP providers are independent.** PP1 and PP2 come from different payroll companies with independent measurement systems. No common factor is imposed across providers; cross-provider correlation is fully explained by the shared latent signal.

4.  **PP1 has autocorrelated errors.** PP1's large multi-establishment clients create persistent measurement artifacts modeled as AR(1) errors.

The model is estimated on 119 months of data (February 2016 – December 2025). All indices are rebased to 100 at March 12, 2020.

**Presentation of results.** For NFP-style communication, the model reports **change** (jobs added or lost, thousands) as the primary output rather than employment levels. Forecasts and nowcasts are expressed as month-over-month change, e.g. “The economy added 138k jobs on a seasonally adjusted basis (−259k not seasonally adjusted).” Levels are available for reference but are secondary.

------------------------------------------------------------------------

## 2. Data Sources and Transformation

### 2.1 Data Sources

| Source | Type | Coverage | Frequency | Concept |
|---------------|---------------|---------------|---------------|---------------|
| CES SA | Survey | \~145,000 establishments | Monthly | Total private employment (seasonally adjusted) |
| CES NSA | Survey | \~145,000 establishments | Monthly | Total private employment (not seasonally adjusted) |
| QCEW | Administrative (UI tax records) | Near-census (\~10M establishments) | Monthly | Total private employment (NSA) |
| PP1 | Payroll microdata (Provider 1) | Large multi-establishment firms | Monthly | Continuing-units employment (NSA) |
| PP2 | Payroll microdata (Provider 2) | Small single-establishment firms | Monthly | Continuing-units employment (NSA) |

**QCEW data quality.** Employers file UI tax reports quarterly. BLS publishes monthly employment counts, but only the quarter-end month (March, June, September, December) is derived directly from current-period tax filings. Months 1 and 2 within each quarter come from UI tax records that are reported retrospectively — they are not BLS interpolations, but they are not available in real time for the active reference period. The model reflects this quality gradient through month-dependent fixed sigmas (Section 4.4).

**PP series selection.** Provider 2 originally supplied three series: PP2_0 (basic continuing-units), PP2_1, and PP2_2 (matched-panel variants). All three were 0.999+ correlated (Section 7.1), providing near-zero marginal information. Retaining all three triple-counted Provider 2's information and inflated its precision share to 44%. Only PP2_0 is retained, renamed PP2.

**No common provider factor.** A common factor $f_t$ was previously shared across all PP series. This was appropriate when multiple series from the same provider needed their shared artifacts absorbed. With two series from independent providers, a common factor is inappropriate: PP1 and PP2 use different payroll systems, different client bases, and different processing pipelines. Their measurement errors should be independent. Cross-provider correlation (empirically \~0.97) is fully explained by the shared latent employment signal through their respective loadings.

### 2.2 Growth-Rate Specification

All series enter the model as monthly log growth rates:

$$g_t^x = \ln(x_t) - \ln(x_{t-1})$$

The first observation is dropped since differencing consumes one period, yielding $T = 119$ monthly observations.

### 2.3 Sample Summary

| Series                    | Observations | Mean growth (%/mo) |
|---------------------------|:------------:|-------------------:|
| CES SA                    |     119      |             +0.098 |
| CES NSA                   |     119      |             +0.116 |
| QCEW (all monthly)        |     113      |             +0.116 |
| — Quarter-end (M3)        |      38      |                  — |
| — Retrospective UI (M1–2) |      75      |                  — |
| PP1                       |     113      |             +0.184 |
| PP2                       |      82      |             +0.313 |

CES NSA and QCEW mean growth agree at +0.116%/mo, confirming they measure the same concept. PP1 and PP2 show higher mean growth due to survivorship bias in continuing-units panels.

------------------------------------------------------------------------

## 3. Latent State: Continuing-Units Employment Growth

### 3.1 AR(1) Dynamics

The latent process is seasonally adjusted continuing-units employment growth:

$$g_t^{\text{cont}} = \mu_g + \phi\left(g_{t-1}^{\text{cont}} - \mu_g\right) + \sigma_g\,\varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, 1)$$

Initialized at $g_0^{\text{cont}} = \mu_g + \sigma_g\,\varepsilon_0$ and evolved via `pytensor.scan`. The non-centered parameterization (sampling $\varepsilon_t$ and scaling by $\sigma_g$) improves MCMC geometry.

### 3.2 Priors

| Parameter | Prior | Rationale |
|--------------------------|--------------------|--------------------------|
| $\mu_g$ | $\mathcal{N}(0.001, 0.005^2)$ | \~0.1%/mo (\~1.2% annualized); wide for recession/expansion |
| $\phi$ | $\text{Uniform}(0, 0.99)$ | Uninformative over the stationary region |
| $\sigma_g$ | $\text{Half-}\mathcal{N}(0, 0.008)$ | Scaled to observed monthly growth volatility |

------------------------------------------------------------------------

## 4. Measurement Equations

### 4.1 Seasonal Component

Deterministic monthly seasonal with 11 free parameters and a sum-to-zero constraint:

$$s_m \text{ for } m = 1, \ldots, 11, \qquad s_{12} = -\sum_{m=1}^{11} s_m$$

Prior: $s_m \sim \mathcal{N}(0, 0.015^2)$.

### 4.2 Birth/Death Offset

$$\text{bd} \sim \mathcal{N}(0.001, 0.002^2)$$

Time-invariant, capturing the systematic difference between continuing-units growth and total employment growth.

### 4.3 Composite Growth Signals

| Signal | Formula | Observed by |
|--------------------|----------------------|------------------------------|
| NSA continuing-units | $g_t^{\text{cont,nsa}} = g_t^{\text{cont}} + s_t$ | PP1, PP2 |
| SA total employment | $g_t^{\text{total,sa}} = g_t^{\text{cont}} + \text{bd}$ | CES SA |
| NSA total employment | $g_t^{\text{total,nsa}} = g_t^{\text{cont}} + s_t + \text{bd}$ | CES NSA, QCEW |

### 4.4 QCEW Likelihood — Truth Anchor

QCEW defines the measurement scale with $\alpha = 0$, $\lambda = 1$, and fixed noise that varies by month-in-quarter:

$$y_t^{\text{QCEW}} = g_t^{\text{total,nsa}} + \varepsilon_t^{\text{QCEW}}, \qquad \varepsilon_t^{\text{QCEW}} \sim \mathcal{N}(0, \sigma_{\text{QCEW},t}^2)$$

$$\sigma_{\text{QCEW},t} = \begin{cases} 0.0005 & \text{quarter-end month (M3: Mar, Jun, Sep, Dec)} \\ 0.0015 & \text{mid-quarter month (M1–2)} \end{cases}$$

**Why fixed sigma.** Estimating $\sigma_{\text{QCEW}}$ would allow the model to discount QCEW, undermining its anchor role. The fixed values are calibrated: 0.05%/mo for M3 reflects near-census quality ($\pm$\~6,500 jobs at current levels); 0.15%/mo for M1–2 reflects the additional uncertainty from retrospective UI reporting (these months are not from the current reference period's filing). These values were tightened from an earlier calibration (0.1% / 0.3%) after diagnostics revealed the looser values gave QCEW only \~4% of the precision budget—insufficient for the anchor to actually constrain the latent trajectory, which drifted systematically below QCEW observations as CES (95% precision share) dominated month-to-month filtering.

**Why QCEW over CES as anchor.** At the national level, CES is highly precise, but the architecture must scale to state $\times$ supersector where CES sample sizes shrink by 1–2 orders of magnitude. QCEW's administrative coverage is scale-invariant.

### 4.5 CES Likelihood — Noisy Observation

$$y_t^{\text{CES,SA}} = \alpha_{\text{CES}} + \lambda_{\text{CES}} \cdot g_t^{\text{total,sa}} + \varepsilon_t^{\text{SA}}, \qquad \varepsilon_t^{\text{SA}} \sim \mathcal{N}(0, \sigma_{\text{CES,SA}}^2)$$

$$y_t^{\text{CES,NSA}} = \alpha_{\text{CES}} + \lambda_{\text{CES}} \cdot g_t^{\text{total,nsa}} + \varepsilon_t^{\text{NSA}}, \qquad \varepsilon_t^{\text{NSA}} \sim \mathcal{N}(0, \sigma_{\text{CES,NSA}}^2)$$

SA and NSA share $\alpha_{\text{CES}}$ and $\lambda_{\text{CES}}$ but have separate noise parameters.

| Parameter | Prior | Rationale |
|--------------------------|--------------------|--------------------------|
| $\alpha_{\text{CES}}$ | $\mathcal{N}(0, 0.005^2)$ | Small bias allowed |
| $\lambda_{\text{CES}}$ | $\mathcal{N}(1.0, 0.25^2)$ | Centered on unit loading |
| $\sigma_{\text{CES,SA}}$ | $\text{InverseGamma}(3.0, 0.004)$ | Mode $\approx 0.001$; prevents zero-collapse |
| $\sigma_{\text{CES,NSA}}$ | $\text{InverseGamma}(3.0, 0.004)$ | Mode $\approx 0.001$; prevents zero-collapse |

### 4.6 PP Likelihood — Independent Providers

Each PP series observes NSA continuing-units growth with provider-specific parameters and independent measurement errors:

#### 4.6.1 PP1: AR(1) Measurement Error

PP1 clients are large multi-establishment firms. Within-client restructuring creates autocorrelated residuals:

$$u_t = \rho\, u_{t-1} + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, \sigma_1^2)$$

Implemented as a conditional likelihood:

$$y_{1,t}^{\text{PP}} \mid y_{1,t-1}^{\text{PP}} \sim \mathcal{N}\!\left(\mu_{1,t} + \rho\left(y_{1,t-1}^{\text{PP}} - \mu_{1,t-1}\right),\; \sigma_1^2\right)$$

where $\mu_{1,t} = \alpha_1 + \lambda_1 \, g_t^{\text{cont,nsa}}$, with marginal variance $\sigma_1^2 / (1 - \rho^2)$ at $t = 0$.

#### 4.6.2 PP2: iid Measurement Error

PP2 clients are predominantly single-establishment firms with clean client-establishment alignment:

$$y_{2,t}^{\text{PP}} = \alpha_2 + \lambda_2 \, g_t^{\text{cont,nsa}} + \varepsilon_{2,t}, \qquad \varepsilon_{2,t} \sim \mathcal{N}(0, \sigma_2^2)$$

#### 4.6.3 PP Priors

| Parameter | Prior | Rationale |
|--------------------------|--------------------|--------------------------|
| $\alpha_p$ | $\mathcal{N}(0, 0.005^2)$ | Small systematic bias allowed |
| $\lambda_p$ | $\mathcal{N}(1.0, 0.25^2)$ | Centered on unit loading |
| $\sigma_p$ | $\text{InverseGamma}(3.0, 0.004)$ | Mode $\approx 0.001$; prevents funnel geometry |
| $\rho$ (PP1 only) | $\text{Uniform}(0, 0.99)$ | Positive autocorrelation; bounded below 1 |

------------------------------------------------------------------------

## 5. Identification

1.  **Latent** $g_t^{\text{total,nsa}}$: QCEW observations with $\alpha = 0$, $\lambda = 1$ directly anchor NSA total employment growth.

2.  **CES calibration**: Given the QCEW-identified latent state, $\alpha_{\text{CES}}$ is the mean CES residual and $\lambda_{\text{CES}}$ is the CES-vs-latent slope.

3.  **Seasonal pattern**: CES SA + CES NSA jointly identify $s_t \approx g_t^{\text{CES,NSA}} - g_t^{\text{CES,SA}}$, crosschecked against the QCEW anchor which directly observes NSA total growth.

4.  **Birth/death offset**: The mean wedge between total growth (QCEW, CES) and continuing-units growth (PP).

5.  **PP parameters**: Given $g_t^{\text{cont,nsa}}$ (total minus BD), PP bias, loading, and noise are identified from PP residuals.

------------------------------------------------------------------------

## 6. Computational Implementation

| Setting           | Value                      |
|-------------------|----------------------------|
| Draws             | 8,000 per chain            |
| Tuning            | 6,000 per chain            |
| Chains            | 4                          |
| Target accept     | 0.97                       |
| Preferred sampler | nutpie (Rust-based NUTS)   |
| Fallback          | PyMC NUTS                  |
| Runtime           | \~2 minutes (Apple M4 Max) |

The higher draw count and target acceptance rate (vs. standard 4,000/0.95) are needed because the latent AR(1) dynamics create tight coupling between global parameters ($\mu_g$, $\phi$, $\sigma_g$) and the 119-element latent trajectory. These parameters require more exploration to achieve adequate effective sample sizes.

------------------------------------------------------------------------

## 7. Empirical Results

### 7.1 PP Series Redundancy Analysis

Provider 2 originally supplied three series. Pairwise analysis in the 80-month overlap:

| Pair           | Correlation | Diff std (%/mo) |
|----------------|:-----------:|----------------:|
| PP2_0 vs PP2_1 |   0.9988    |           0.236 |
| PP2_0 vs PP2_2 |   0.9995    |           0.114 |
| PP2_1 vs PP2_2 |   0.9993    |           0.184 |
| PP1 vs PP2_0   |   0.9705    |           0.985 |

Within-PP2 differences (0.11–0.24%) are an order of magnitude smaller than between-provider differences (0.99%). The three PP2 series share \>99.9% of their variation. Including all three inflated PP2's precision share to 44% and pushed PP2_2 alone to 32%—more than any other source. Retaining only PP2_0 eliminates the triple-counting.

### 7.2 Common Factor Analysis

A common provider factor $f_t$ was previously shared across all PP series. When reduced to two independent providers:

-   $\sigma_f$ dropped from 1.0% (4-series model) to 0.5% (2-series model), confirming most of what $f_t$ captured was within-PP2 redundancy
-   PP1 and PP2 come from different companies—no theoretical basis for shared measurement artifacts
-   Removing $f_t$ eliminated 119 latent parameters, cut runtime from 77s to 42s (subsequently increased to 91s with longer sampling), and produced cleaner parameter interpretation

The cross-provider correlation (\~0.97) is fully explained by the shared latent signal $g_t^{\text{cont,nsa}}$ through their respective loadings.

### 7.3 Sampling Diagnostics

| Diagnostic             | Value         |
|------------------------|---------------|
| Divergences            | 0             |
| R-hat (all parameters) | $\leq 1.01$   |
| Min ESS (bulk)         | 315 ($\mu_g$) |
| Median ESS (bulk)      | \~9,000       |
| Max ESS (bulk)         | \~27,000      |

Zero divergences. All R-hat values at 1.01 or below. The lowest ESS occurs for $\mu_g$ (315, just below the 400 threshold); all other parameters exceed 400. The tightened QCEW sigmas substantially improved convergence compared to the earlier calibration (where $\mu_g$ ESS was 53 and R-hat was 1.07), because the stronger QCEW anchor better identifies the latent trajectory and reduces the coupling between global parameters and the 119-element latent state.

### 7.4 Latent Dynamics

| Parameter | Posterior mean | 80% HDI | Interpretation |
|------------------|:----------------:|:----------------:|--------------------|
| $\mu_g$ | +0.001%/mo | \[−0.2%, +0.3%\] | Long-run continuing-units growth near zero |
| $\phi$ | 0.066 | \[0.00, 0.11\] | Minimal persistence; shocks dissipate quickly |
| $\sigma_g$ | 1.6%/mo | \[1.5%, 1.8%\] | Monthly innovation volatility |

The near-zero $\mu_g$ implies all trend employment growth comes from the birth/death offset. In a steady state, the continuing-units population is approximately constant (entries $\approx$ exits from the panel), and net growth is driven by business formation.

The low $\phi$ (0.07) indicates employment growth shocks are nearly transitory—a departure from random walk specifications common in the literature.

### 7.5 Birth/Death Offset

| Statistic  |              Value |
|------------|-------------------:|
| Mean       |         +0.014%/mo |
| 80% HDI    | \[−0.18%, +0.20%\] |
| Annualized |             +0.17% |

The BD offset is statistically indistinguishable from zero with wide uncertainty. This reflects genuine identification difficulty: BD is the difference between QCEW (total) and PP (continuing-units), both of which are noisy relative to this small quantity.

### 7.6 CES Observation Parameters

| Parameter | Estimate | Interpretation |
|----------------------|:------------------:|------------------------------|
| $\alpha_{\text{CES}}$ | +0.034%/mo | CES overstates growth by \~0.4%/year vs QCEW |
| $\lambda_{\text{CES}}$ | 0.986 \[0.98, 1.00\] | Near-unit loading; slight attenuation |
| $\sigma_{\text{CES,SA}}$ | 0.085%/mo | CES SA measurement noise |
| $\sigma_{\text{CES,NSA}}$ | 0.093%/mo | CES NSA measurement noise |

**The CES bias finding.** $\alpha_{\text{CES}} = +0.034\%$/month implies CES systematically overstates private employment growth by approximately 0.4%/year relative to the QCEW census, compounding to \~4% over a decade. This is consistent with the well-documented pattern of negative CES-to-QCEW benchmark revisions and may reflect assumptions in the CES birth/death model or declining survey response rates. Under the earlier, looser QCEW calibration (0.1% / 0.3%), the model estimated $\alpha_{\text{CES}} = +0.051\%$/month; the tighter QCEW anchor reduces the estimated bias because the latent trajectory tracks QCEW more closely, leaving less gap for $\alpha_{\text{CES}}$ to absorb.

**CES noise is higher than under the looser calibration.** $\sigma_{\text{CES,SA}} = 0.085\%$ and $\sigma_{\text{CES,NSA}} = 0.093\%$ are roughly double the earlier estimates (\~0.05%). With a tighter QCEW anchor, the latent is more tightly pinned to QCEW observations, and CES residuals relative to that anchor are larger. This is a more realistic estimate of CES survey noise for a sample of \~145,000 establishments.

$\lambda_{\text{CES}} \approx 0.99$ indicates CES has no meaningful scale distortion—it faithfully maps the magnitude of growth fluctuations. The systematic error is primarily a level drift (bias), not a slope problem.

### 7.7 PP Parameters

| Parameter                     |                 PP1 |                 PP2 |
|-------------------------------|--------------------:|--------------------:|
| $\alpha_p$ (%/mo)             |              +0.087 |              +0.224 |
| $\lambda_p$                   | 1.35 \[1.31, 1.38\] | 1.53 \[1.45, 1.60\] |
| $\sigma_p$ (%/mo)             |                 0.5 |                 1.1 |
| $\rho$                        | 0.29 \[0.17, 0.42\] |                   — |
| $\sigma_{\text{marg}}$ (%/mo) |                0.54 |                   — |

**PP loadings above unity.** Both PP series amplify the latent signal ($\lambda > 1$). This reflects survivorship bias: when weak establishments exit the continuing-units panel, the remaining sample's growth rate exceeds the population average. PP2 shows higher amplification ($\lambda = 1.53$) than PP1 ($\lambda = 1.35$), consistent with PP2's smaller client base experiencing more turnover.

**PP1 autocorrelation.** $\rho = 0.29$ with HDI \[0.16, 0.42\] confirms autocorrelated measurement errors. The multi-establishment structure of PP1 clients creates persistent artifacts as firms restructure across months. The innovation sigma (0.5%) measures new information per observation; the marginal sigma (0.55%) is the total residual volatility.

**PP2 noise.** PP2's higher noise ($\sigma = 1.1\%$) reflects its smaller, more volatile client base.

### 7.8 Precision Budget

| Source      | Obs | Prec/obs | Total prec |    Share |
|-------------|:---:|:--------:|:----------:|---------:|
| CES SA      | 119 |  1,345k  |    160M    |    32.8% |
| CES NSA     | 119 |  1,126k  |    134M    |    27.5% |
| PP1         | 113 |   62k    |     7M     |     1.4% |
| PP2         | 82  |   19k    |     2M     |     0.3% |
| QCEW (M3)   | 38  |  4,000k  |    152M    |    31.1% |
| QCEW (M1–2) | 75  |   444k   |    33M     |     6.8% |
| **Total**   |     |          |  **488M**  | **100%** |

**National-level interpretation.** CES contributes 60.3% and QCEW contributes 37.9%. The tightened QCEW sigmas (0.05% M3, 0.15% M1–2) give QCEW enough weight to actually anchor the latent trajectory month-to-month. Under the earlier, looser calibration (0.1% / 0.3%), QCEW had only 4% share and the latent index drifted systematically below QCEW observations as CES (95%) dominated filtering. CES noise is now estimated at \~0.09%—more realistic for a survey of 145,000 establishments—rather than the artificially tight \~0.05% that arose when QCEW lacked sufficient precision to constrain the latent. PP contributes 1.7% combined.

**This balance will invert at state** $\times$ supersector. CES noise scales as $\sim 1/\sqrt{n}$ where $n$ is the sample size. At a typical state-supersector cell, $\sigma_{\text{CES}}$ could be 0.5–2%, reducing CES per-observation precision by 30–500$\times$ and making QCEW the dominant information source.

### 7.9 Seasonal Pattern

The estimated seasonal is consistent with the empirical CES SA–NSA differential:

-   **January**: −2.0% (post-holiday layoffs)
-   **June**: +0.8% (summer hiring)
-   **September–October**: −0.8% (seasonal transitions)

The seasonal is well-identified from simultaneous CES SA and NSA data, crosschecked against monthly QCEW.

------------------------------------------------------------------------

## 8. Forecasting

Reported forecast output is **jobs added** (month-over-month change in thousands, SA and NSA) first; index and level are derived for consistency and optional use.

### 8.1 Method

Forecasts are generated by simulating the AR(1) forward from the last posterior state:

$$g_{T+h}^{\text{cont}} = \mu_g + \phi\left(g_{T+h-1}^{\text{cont}} - \mu_g\right) + \sigma_g\,\varepsilon_{T+h}$$

with $g_{T+h}^{\text{total,sa}} = g_{T+h}^{\text{cont}} + \text{bd}$ and $g_{T+h}^{\text{total,nsa}} = g_{T+h}^{\text{cont}} + s_{m(T+h)} + \text{bd}$.

Index levels are reconstructed by cumulating log growth rates; employment levels (thousands) by scaling against the base-period employment-to-index ratio. For NFP communication, **change** is primary: the model reports *jobs added* (month-over-month change in thousands, SA and NSA), e.g. “The economy added 138k jobs on a seasonally adjusted basis (−259k not seasonally adjusted).” The main forecast figure (saved as `forecast_levels.png`) plots **jobs added** (month-over-month change) for SA and NSA; console output leads with the same. Employment levels are available for reference.

### 8.2 January 2026 Forecast

| Metric                       |                   SA |                NSA |
|------------------------------|---------------------:|-------------------:|
| Jobs added (mean, thousands) |                +122k |            −2,572k |
| Jobs added (80% HDI)         | \[−2,641k, +2,905k\] | \[−5,291k, +162k\] |
| Monthly growth (mean)        |              +0.079% |            −1.990% |
| Index (mean)                 |               102.34 |             101.63 |
| Employment level (mean)      |             128,321k |           126,199k |

The wide intervals reflect $\sigma_g = 1.6\%$/mo innovation uncertainty. The strongly negative NSA forecast is driven by the January seasonal ($s_{\text{Jan}} \approx -2.0\%$).

### 8.3 Uncertainty Decomposition

Forecast uncertainty propagates three sources: parameter uncertainty (different MCMC draws yield different structural parameters), state uncertainty (the terminal filtered state has posterior spread), and innovation uncertainty (the future shock $\varepsilon_{T+h}$). Innovation uncertainty dominates at short horizons.

------------------------------------------------------------------------

## 9. The Nowcasting Window

### 9.1 Information Regime Shift

During the historical period, CES dominates (60% of precision) with QCEW providing 38%. In the nowcasting window—after PP data arrives but before CES is published—PP becomes the sole monthly signal. PP's relative contribution jumps from \~1.7% (in-sample) to dominant (nowcast month).

### 9.2 NFP Nowcast Construction

PP observes NSA continuing-units growth. Converting to an NFP-comparable estimate requires seasonal adjustment (subtract $s_{m(T+1)}$) and birth/death offset (add $\text{bd}$). The CES calibration parameters ($\alpha_{\text{CES}}$, $\lambda_{\text{CES}}$) indicate how the latent state maps to CES-reported growth. The **headline nowcast is jobs added** (month-over-month change in thousands, SA and NSA), e.g. “+138k SA, −259k NSA,” not the employment level.

### 9.3 Operational Workflow

1.  Append new PP index rows (CES/QCEW columns missing for nowcast month)
2.  Re-run `load_data()` — calendar extends automatically
3.  Re-run `build_and_sample()` — observation masks handle missing data
4.  Nowcast month is reported as **jobs added** (and growth); forecast horizon shifts forward

No code changes required. The model is data-availability-agnostic. Console and figure outputs lead with change (jobs added); levels are available if needed.

------------------------------------------------------------------------

## 10. Model Evolution Log

The gold standard specification was reached through systematic simplification, with each change validated by sampling diagnostics and parameter stability:

| Version | Change | Key effect |
|-----------------------|--------------------|-----------------------------|
| v2.0 | Initial: CES-anchored, 4 PP series, common factor, fixed $\sigma_{\text{CES}}$ | Baseline |
| v2.1 | QCEW monthly anchor (quality-dependent $\sigma$); CES as noisy obs ($\alpha$, $\lambda$, $\sigma$) | CES bias finding identified |
| v2.2 | Drop PP2_1, PP2_2 (0.999+ correlated with PP2_0) | Eliminated triple-counting; precision budget rebalanced |
| v2.3 | Remove common factor $f_t$ (PP1, PP2 are independent providers) | Cleaner interpretation; 119 fewer latent parameters |
| v2.4 | Longer sampling (8k draws, 6k tune, target_accept=0.97) | 0 divergences; observation params well identified |
| **v2.5 (gold)** | Tightened QCEW $\sigma$: 0.05% (M3), 0.15% (M1–2) | QCEW share 38%; all R-hat $\leq$ 1.01; latent tracks QCEW; CES bias +0.034%/mo |

The QCEW sigma tightening (v2.5) was motivated by a diagnostic finding: under the looser calibration, QCEW held only 4% of precision and the latent index drifted systematically below QCEW observations. CES (95% share) dominated month-to-month filtering, producing systematically positive QCEW residuals. Halving the QCEW sigmas gives the anchor 38% of precision, eliminates the latent–QCEW divergence, and dramatically improves convergence ($\mu_g$ ESS rose from 53 to 315, R-hat from 1.07 to 1.01).

All substantive parameters ($\lambda_{\text{CES}}$, PP loadings, $\rho_{\text{PP1}}$) were stable across all versions, confirming the simplifications did not alter the model's substantive conclusions. The CES bias ($\alpha_{\text{CES}}$) is sensitive to QCEW calibration: +0.051%/mo under the looser sigmas, +0.034%/mo under the tighter sigmas.

------------------------------------------------------------------------

## 11. Relationship to Planned Extensions

| Feature | v2 (Gold Standard) | Planned (State $\times$ Supersector) |
|-------------------|-------------------|-----------------------------------|
| Truth anchor | QCEW (monthly, fixed $\sigma$) | Same — scale-invariant |
| CES treatment | Noisy obs ($\alpha$, $\lambda$, $\sigma$ estimated) | Same, but $\sigma$ much larger |
| PP series | 2 national (PP1 + PP2, independent) | Disaggregated to cells |
| Latent dynamics | Single AR(1) | One per cell, hierarchical priors |
| Seasonal | Single pattern | Cell-specific, shrunk toward national |
| Birth/death | Single constant | Cell-specific, potentially time-varying |
| Cross-cell consistency | N/A | Reconciliation (cells sum to national) |
| Computation | \~2 min, single model | \~500 cells; joint hierarchical estimation |

**What transfers directly.** The model equation, QCEW anchoring with quality-dependent sigma, CES calibration parameters, PP measurement model (including AR(1) for PP1), and the observation-mask approach to mixed data availability.

**What requires extension.** Hierarchical priors across cells (partial pooling), PP disaggregation pipeline, computational infrastructure for \~500 cells, and potentially time-varying birth/death and seasonal components.

------------------------------------------------------------------------

## 12. Limitations

-   **Time-invariant birth/death.** Cannot capture cyclical variation in net business formation at turning points.
-   **Static seasonal.** Does not evolve across years (e.g., pandemic shifts).
-   **No time-varying provider bias.** PP bias ($\alpha_p$) is constant despite potential compositional drift.
-   **Calibrated QCEW sigmas.** The 0.05% (M3) / 0.15% (M1–2) values reflect judgment, not estimation. Sensitivity analysis (Section 13.2) shows key parameters ($\alpha_{\text{CES}}$, $\sigma_{\text{CES}}$) are sensitive to this choice while PP parameters are stable.
-   **QCEW LOO diagnostics.** The tight QCEW sigmas make individual QCEW observations highly influential (p_loo = 80, 21 high k-hat). PSIS-LOO is unreliable for QCEW; refitting-based LOO would be needed for formal assessment.
-   **National level only.** No geographic or industry decomposition.
-   **PP precision is low at national level.** PP contributes \~1.7% of total precision. Its value is primarily in the nowcasting window when CES is unavailable, and will grow substantially at state $\times$ supersector where CES precision degrades.

------------------------------------------------------------------------

## 13. v3 Experiments: Nowcast Backtest and QCEW Sigma Sensitivity

### 13.1 Operational value of payroll data (nowcast backtest)

The precision budget (Section 7.8) shows PP contributing \<1% of total information at the national level when CES and QCEW are available. That understates PP’s *operational* value: in real time, CES is released with a lag, while PP can be available sooner. To quantify how much PP helps when CES is missing, a **nowcast backtest** was run (`pp_nowcast_backtest_v3.py`).

**Design.** For each of the last 12 months, CES was censored from that month onward; the v2 model was run with lighter sampling (2000 draws, 2 chains) using only QCEW, PP1, and PP2 as available. The model’s posterior mean for SA *jobs added* (month-over-month change) and growth in that month was compared to the actual CES release. Months 2025-01–06 have QCEW, PP1, and PP2; months 2025-07–12 have only PP2 (QCEW and PP1 end in 2025-06 in the current data).

**Results.** Primary metric: error in jobs added (thousands). Growth (pp) is secondary.

| Setting                  | n   | MAE jobs added (k) | MAE growth (pp) |
|--------------------------|-----|--------------------|-----------------|
| Overall                  | 12  | 255                | 0.198           |
| Data-rich (QCEW+PP1+PP2) | 6   | 166                | 0.127           |
| PP2-only                 | 6   | 344                | 0.268           |

When only PP2 is available, nowcast error in jobs added is about twice as large (344k vs 166k) and growth error roughly twice as large (0.27 pp vs 0.13 pp). That directly quantifies PP’s value in the nowcasting window and supports the claim that PP matters most when CES (and at the national level, QCEW) is absent.

**Real-time nowcast.** The same procedure can be used operationally when CES for the latest month is delayed: run the backtest script with `censor_ces_from` set to that month’s reference date (e.g. the 12th). The script’s single-run output for that month is the model’s nowcast of SA jobs added and growth; the full 12-month backtest is only needed for evaluation, not for producing a single nowcast.

------------------------------------------------------------------------

### 13.2 QCEW sigma sensitivity and calibration

The QCEW truth anchor uses *fixed* observation noise: $\sigma_{\text{QCEW}}^{\text{M3}} = 0.05\%$ (quarter-end months) and $\sigma_{\text{QCEW}}^{\text{M1--2}} = 0.15\%$ (retrospective UI months). These values are calibrated judgments, not estimated. To see how much conclusions depend on them, the model was run with three QCEW noise levels (`pp_sensitivity_v3.py`): **0.5× (tight)**, **1× (baseline)**, and **2× (loose)**.

**Motivation for the current calibration.** An earlier version used looser QCEW sigmas (0.1% / 0.3%), which gave QCEW only ~4% of the precision budget. Diagnostics revealed the latent index drifted systematically below QCEW observations: CES (95% share) dominated month-to-month filtering, and the model's estimated CES bias ($\alpha_{\text{CES}} = +0.051\%$/mo) pulled the latent below QCEW-implied growth. The tightened calibration (0.05% / 0.15%) gives QCEW ~38% share, eliminates the latent--QCEW divergence, and substantially improves convergence diagnostics.

**Parameter stability.**

-   **Sensitive to QCEW sigma:** $\alpha_{\text{CES}}$ (bias) and $\sigma_{\text{CES}}$ (SA/NSA observation noise) move with the calibration; when QCEW is looser, the model attributes more noise to QCEW and estimates CES as relatively more precise.
-   **Stable:** $\lambda_{\text{CES}}$, $\rho_{\text{PP1}}$, and PP loadings ($\lambda_p$, $\sigma_p$) are nearly unchanged across the three configurations. Birth/death offset has very wide HDIs in all runs, so its point estimate is not emphasized.

**Precision budget.**

| Configuration  | QCEW sigma (M3 / M1-2) | QCEW share | CES share (SA+NSA) | PP share |
|----------------|-------------------------|------------|--------------------|----------|
| 0.5x (tight)   | 0.025% / 0.075%         | ---        | ---                | ---      |
| 1x (baseline)  | 0.05% / 0.15%           | 37.9%      | 60.3%              | 1.7%     |
| 2x (loose)     | 0.10% / 0.30%           | ~4%        | ~96%               | <1%      |

The 2x (loose) configuration corresponds to the earlier calibration and illustrates the problem: QCEW drops to ~4% share and CES dominates at 96%. The baseline (1x) keeps QCEW as a meaningful anchor (~38%) while CES still provides the majority of the total information (60%).

Reporting this sensitivity makes clear that conclusions are not driven solely by the sigma choice---readers can see how the precision budget and CES bias shift if they prefer tighter or looser QCEW noise.
