# Estimation Methodology: Growth-Rate Bayesian State Space Model for NFP Nowcasting (v2)

## 1. Overview

This document describes the econometric methodology implemented in `pp_estimation_v2.py` (now in `archive/`; the current v3 implementation is in `src/alt_nfp/`). The model is a Bayesian state space system that extracts a latent national employment growth signal from multiple noisy data sources: seasonally adjusted CES, not-seasonally-adjusted CES, four payroll provider (PP) continuing-units series, and quarterly QCEW observations. The specification operates entirely in monthly log growth rates, explicitly separates continuing-units dynamics from total employment via a birth/death offset, and models seasonal variation and correlated provider measurement errors within a unified framework.

This corresponds to Release 1 of the staged implementation plan described in the companion specification document, extended with elements of Release 2 (birth/death decomposition) adapted to the single-provider, multi-panel setting.

------------------------------------------------------------------------

## 2. Data Transformation and Alignment

### 2.1 Index Construction

All input series enter the model as index levels normalized to a common base period. Four data sources are aligned to a monthly calendar spanning the CES date range:

-   **CES SA and CES NSA:** Monthly employment indices (seasonally adjusted and not seasonally adjusted), published by BLS.
-   **Payroll Provider Panels (PP1–PP4):** Four continuing-units employment indices constructed from payroll microdata using rotating, frozen measurement panels. Each panel variant reflects a different stabilization or construction rule applied to the same underlying provider data.
-   **QCEW:** Quarterly employment index derived from the near-census Quarterly Census of Employment and Wages.

### 2.2 Growth-Rate Specification

The model operates on monthly log growth rates rather than levels. For each series $x$, the observed growth is:

$$g_t^x = \ln(x_t) - \ln(x_{t-1})$$

This transformation is applied to CES SA, CES NSA, and all four PP series to produce monthly growth observations. The first observation is dropped since differencing consumes one period.

For QCEW, which is observed quarterly, the model constructs quarterly log growth rates and maps them to sums of monthly latent growths (see Section 4.5).

### 2.3 Rationale for Growth Rates over Levels

The shift from a levels-based to a growth-rate specification addresses a fundamental identification problem. In levels, the accumulating wedge between continuing-units employment (what payroll providers measure) and total employment (what NFP captures) cannot be absorbed by a constant bias parameter—the gap grows without bound. In growth rates, the birth/death contribution to *growth* is approximately stationary, allowing a constant offset parameter to capture the systematic difference between continuing-units growth and total employment growth.

------------------------------------------------------------------------

## 3. Latent State: Continuing-Units Employment Growth

### 3.1 AR(1) Dynamics with Mean Reversion

The core latent process is seasonally adjusted continuing-units employment growth, modeled as a stationary AR(1) process with mean reversion:

$$g_t^{\text{cont}} = \mu_g + \phi \left(g_{t-1}^{\text{cont}} - \mu_g\right) + \sigma_g \, \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, 1)$$

where:

-   $\mu_g$ is the long-run mean monthly growth rate
-   $\phi \in [0, 0.99)$ is the persistence parameter
-   $\sigma_g$ is the innovation standard deviation
-   $\varepsilon_t$ are standard normal shocks (non-centered parameterization)

The AR(1) specification with $\phi < 1$ ensures stationarity and mean reversion, a deliberate departure from the random walk specification used in Release 1 of the companion document. This reflects the empirical observation that employment growth rates do not exhibit unit-root behavior: periods of above-average or below-average growth are transitory.

### 3.2 Implementation

The process is initialized at:

$$g_0^{\text{cont}} = \mu_g + \sigma_g \, \varepsilon_0$$

and evolved forward via a `pytensor.scan` loop implementing the AR(1) recursion. The non-centered parameterization (sampling standard normal innovations $\varepsilon_t$ and scaling by $\sigma_g$) improves MCMC geometry by decorrelating the innovation variance from the latent state trajectory.

### 3.3 Priors

| Parameter | Prior | Rationale |
|---------------------------|-------------------|---------------------------|
| $\mu_g$ | $\mathcal{N}(0.001, 0.005^2)$ | Centered on \~0.1%/month (\~1.2% annualized), consistent with trend employment growth; wide enough to accommodate recession/expansion |
| $\phi$ | $\text{Uniform}(0, 0.99)$ | Uninformative over the stationary region; upper bound prevents unit-root behavior |
| $\sigma_g$ | $\text{Half-}\mathcal{N}(0, 0.008)$ | Scaled to observed monthly employment growth volatility (\~0.2–0.5%/month) |

------------------------------------------------------------------------

## 4. Measurement Equations

The model combines four distinct data sources, each observing a different composite of the latent components. This section describes how each source maps to the latent state.

### 4.1 Seasonal Component

A deterministic monthly seasonal pattern is estimated with 11 free parameters subject to a sum-to-zero constraint:

$$s_m \text{ for } m = 1, \ldots, 11, \qquad s_{12} = -\sum_{m=1}^{11} s_m$$

The seasonal effect at time $t$ is $s_{t} = s_{m(t)}$ where $m(t) \in \{1, \ldots, 12\}$ is the calendar month. The prior $s_m \sim \mathcal{N}(0, 0.015^2)$ allows seasonal amplitudes of roughly ±1.5% per month, consistent with the strong seasonal patterns observed in NSA employment data.

### 4.2 Birth/Death Offset

The birth/death (BD) offset captures the systematic difference between continuing-units growth (what payroll providers measure) and total employment growth (which includes net business formation):

$$\text{bd} \sim \mathcal{N}(0.001, 0.002^2)$$

This is modeled as a time-invariant constant. The prior is centered on +0.1%/month (\~1.2% annualized), reflecting the historical positive contribution of net business births to employment growth. The tight prior reflects the relatively stable average BD contribution outside of deep recessions.

In this specification, BD is treated as constant rather than time-varying or cyclically responsive (as in Release 2 of the companion document). This simplification is appropriate for an initial implementation and can be extended in subsequent releases.

### 4.3 Composite Growth Signals

Three composite signals are constructed from the latent components:

| Signal | Formula | Interpretation |
|-------------------|--------------------|----------------------------------|
| NSA continuing-units growth | $g_t^{\text{cont,nsa}} = g_t^{\text{cont}} + s_t$ | What PP series measure |
| SA total employment growth | $g_t^{\text{total,sa}} = g_t^{\text{cont}} + \text{bd}$ | What CES SA measures |
| NSA total employment growth | $g_t^{\text{total,nsa}} = g_t^{\text{cont}} + s_t + \text{bd}$ | What CES NSA and QCEW measure |

This decomposition embodies the core conceptual framework: payroll providers observe continuing-units employment without seasonal adjustment, while official statistics capture total employment (including birth/death effects) with or without seasonal adjustment.

### 4.4 CES Likelihood (SA and NSA)

CES is treated as a high-precision benchmark with fixed measurement noise:

$$y_t^{\text{CES,SA}} = g_t^{\text{total,sa}} + \varepsilon_t^{\text{CES}}, \qquad \varepsilon_t^{\text{CES}} \sim \mathcal{N}(0, \sigma_{\text{CES}}^2)$$

$$y_t^{\text{CES,NSA}} = g_t^{\text{total,nsa}} + \varepsilon_t^{\text{CES}}, \qquad \varepsilon_t^{\text{CES}} \sim \mathcal{N}(0, \sigma_{\text{CES}}^2)$$

The CES measurement noise is **fixed** at $\sigma_{\text{CES}} = 0.001$ (0.1%/month) rather than estimated. This reflects two design decisions:

1.  **CES is a precise benchmark.** Monthly revisions are typically below 0.1%, and the survey methodology is well-established. Estimating $\sigma_{\text{CES}}$ would allow the model to discount CES observations, which is undesirable when CES serves as the primary anchor.

2.  **Sampling geometry.** Estimating a near-zero variance creates a funnel geometry in the posterior that severely degrades MCMC performance. Fixing it removes a problematic parameter while having negligible effect on the substantive posterior.

Both SA and NSA CES series are included simultaneously. The SA series identifies the latent continuing-units growth and BD offset, while the NSA series jointly identifies the seasonal pattern. Including both provides stronger identification than either alone.

The CES measurement equations use $\alpha = 0$ (no bias) and $\lambda = 1$ (unit loading). CES is the definitional anchor for the latent state—the model's objective is to produce estimates consistent with CES when CES is available and to extrapolate forward using PP signals when CES has not yet been released.

### 4.5 Payroll Provider Likelihood

Each PP series $p \in \{1, \ldots, 4\}$ observes NSA continuing-units growth with provider-specific bias, signal loading, and idiosyncratic noise, plus a common factor capturing correlated errors:

$$y_{p,t}^{\text{PP}} = \alpha_p + \lambda_p \, g_t^{\text{cont,nsa}} + f_t + \varepsilon_{p,t}, \qquad \varepsilon_{p,t} \sim \mathcal{N}(0, \sigma_p^2)$$

where:

-   $\alpha_p$ is provider-specific bias (systematic over/under-estimation of growth)
-   $\lambda_p$ is provider-specific signal loading (sensitivity to true growth)
-   $f_t$ is a common provider factor (see Section 4.5.1)
-   $\sigma_p^2$ is provider-specific idiosyncratic noise

**Key design choice:** PP series load on $g_t^{\text{cont,nsa}}$ (continuing-units growth including seasonality) rather than $g_t^{\text{total,nsa}}$. This reflects the fundamental distinction that payroll providers measure employment at continuing establishments and do not capture net business births/deaths.

#### 4.5.1 Correlated Provider Errors

The common factor $f_t$ captures correlation in measurement errors across the four PP panels:

$$f_t \sim \mathcal{N}(0, \sigma_f^2), \qquad \sigma_f \sim \text{Half-}\mathcal{N}(0, 0.008)$$

All four PP series share the same $f_t$ realization at each time period, while retaining independent idiosyncratic errors $\varepsilon_{p,t}$. This structure reflects that the four panels are derived from the same underlying payroll provider data and therefore share common measurement artifacts (e.g., payroll timing effects, provider-wide processing changes, reference-period misalignment) that are distinct from true employment dynamics.

Without the common factor, the model would treat four correlated signals as four independent signals, overstating the effective information content of the PP data and underestimating posterior uncertainty.

#### 4.5.2 PP Priors

| Parameter | Prior | Rationale |
|---------------------------|-------------------|---------------------------|
| $\alpha_p$ | $\mathcal{N}(0, 0.005^2)$ | Small systematic bias allowed; centered at zero |
| $\lambda_p$ | $\mathcal{N}(1.0, 0.25^2)$ | Centered on unit loading; ±0.5 covers attenuation/amplification |
| $\sigma_p$ | $\text{InverseGamma}(3.0, 0.004)$ | Mode ≈ 0.001, prevents collapse to zero; allows data to pull larger for noisier series |

The InverseGamma prior on $\sigma_p$ is a deliberate choice to avoid the same funnel problem as CES: it places a soft lower bound on the noise variance, preventing the sampler from exploring near-zero regions that create difficult posterior geometry.

### 4.6 QCEW Likelihood (Temporal Aggregation)

QCEW is observed quarterly with a publication lag of 5–6 months. Each quarterly growth observation is the sum of the constituent monthly NSA total employment growths:

$$y_q^{\text{QCEW}} = \sum_{t \in q} g_t^{\text{total,nsa}} + \varepsilon_q^{\text{QCEW}}, \qquad \varepsilon_q^{\text{QCEW}} \sim \mathcal{N}(0, \sigma_{\text{QCEW}}^2)$$

where the sum runs over the monthly indices belonging to quarter $q$, and the quarterly growth is computed as the log difference of the QCEW index across adjacent quarters.

This temporal aggregation constraint is critical: it forces the monthly latent states to be consistent with the quarterly near-census observation, providing an anchoring effect that prevents drift in the latent trajectory.

The QCEW noise prior is:

$$\sigma_{\text{QCEW}} \sim \text{InverseGamma}(3.0, 0.008)$$

with mode ≈ 0.002, reflecting that QCEW is a near-universe administrative dataset with very low measurement error. The InverseGamma parameterization again prevents collapse to zero.

------------------------------------------------------------------------

## 5. Identification

The model achieves identification through the following structure:

1.  **Level identification of** $g_t^{\text{cont}}$: CES SA observations with unit loading and no bias directly pin down $g_t^{\text{cont}} + \text{bd}$. The BD offset is then identified as the mean wedge between CES SA growth and PP growth (which observes $g_t^{\text{cont}}$ without BD).

2.  **Seasonal identification:** The simultaneous inclusion of CES SA (seasonally adjusted) and CES NSA (not seasonally adjusted) identifies the seasonal pattern as the systematic difference $s_t = g_t^{\text{total,nsa}} - g_t^{\text{total,sa}}$.

3.  **PP signal parameters:** Given identification of $g_t^{\text{cont}}$ from CES, the PP bias $\alpha_p$, loading $\lambda_p$, and noise $\sigma_p$ are identified from the cross-sectional and time-series variation in PP residuals.

4.  **Common factor:** The common factor $f_t$ is identified as the shared component of PP residuals after removing the fitted signal $\alpha_p + \lambda_p \, g_t^{\text{cont,nsa}}$.

5.  **QCEW anchoring:** The quarterly QCEW observations provide additional anchoring of the NSA total growth path, tightening posterior uncertainty on the seasonal pattern and BD offset.

------------------------------------------------------------------------

## 6. Forecasting

### 6.1 Posterior Predictive Simulation

Forecasts are generated by simulating the AR(1) latent process forward from the last historical posterior state. For each MCMC draw $(c, d)$:

$$g_{T+h}^{\text{cont}} = \mu_g^{(c,d)} + \phi^{(c,d)} \left(g_{T+h-1}^{\text{cont}} - \mu_g^{(c,d)}\right) + \sigma_g^{(c,d)} \, \varepsilon_{T+h}$$

where $\varepsilon_{T+h} \sim \mathcal{N}(0,1)$ are fresh draws. The forecast composite signals are:

$$g_{T+h}^{\text{total,sa}} = g_{T+h}^{\text{cont}} + \text{bd}^{(c,d)}$$

$$g_{T+h}^{\text{total,nsa}} = g_{T+h}^{\text{cont}} + s_{m(T+h)}^{(c,d)} + \text{bd}^{(c,d)}$$

### 6.2 Index and Level Reconstruction

Forecast index levels are reconstructed by cumulating growth rates from the base period:

$$I_{T+h}^{\text{SA}} = \exp\left(\ln I_0^{\text{SA}} + \sum_{t=1}^{T+h} g_t^{\text{total,sa}}\right)$$

Employment levels in thousands are obtained by scaling the index by the base-period employment-to-index ratio.

### 6.3 Uncertainty Quantification

Forecast uncertainty reflects three sources: parameter uncertainty (different MCMC draws have different $\mu_g, \phi, \sigma_g, \text{bd}, s_m$), state uncertainty (the filtered estimate of the terminal state $g_T^{\text{cont}}$ is uncertain), and innovation uncertainty (future shocks $\varepsilon_{T+h}$). All three are automatically propagated through the posterior predictive simulation, producing credible intervals that widen with forecast horizon.

------------------------------------------------------------------------

## 7. Source Contribution Analysis

The model quantifies the relative information contribution of each data source via precision weighting. For each source, the effective precision per observation is:

| Source  | Precision per observation  |
|---------|----------------------------|
| CES SA  | $1/\sigma_{\text{CES}}^2$  |
| CES NSA | $1/\sigma_{\text{CES}}^2$  |
| PP$_p$  | $\lambda_p^2 / \sigma_p^2$ |
| QCEW    | $1/\sigma_{\text{QCEW}}^2$ |

The PP precision formula $\lambda_p^2 / \sigma_p^2$ accounts for the fact that PP observes a scaled version of the latent signal: a larger loading $\lambda_p$ amplifies the signal content, while larger noise $\sigma_p$ dilutes it. Total precision is accumulated across all observations for each source and expressed as a share of the total precision budget.

This analysis reveals the effective weight each source carries in determining the posterior. Given the fixed $\sigma_{\text{CES}} = 0.001$, CES dominates in-sample estimation; the PP series become most valuable for the nowcasting window when CES has not yet been released.

------------------------------------------------------------------------

## 8. Computational Implementation

### 8.1 Sampler Configuration

The model is estimated via MCMC with the following configuration:

| Setting           | Value                                                 |
|----------------------------------------|--------------------------------|
| Draws             | 4,000 per chain                                       |
| Tuning            | 3,000 per chain                                       |
| Chains            | 4                                                     |
| Target accept     | 0.95                                                  |
| Preferred sampler | nutpie (Rust-based NUTS, optimized for Apple Silicon) |
| Fallback sampler  | PyMC NUTS                                             |

The high target acceptance rate (0.95) is chosen to handle the moderately complex posterior geometry arising from the scan-based AR(1) dynamics and the multiple observation equations.

### 8.2 Design Choices for Sampling Efficiency

Several implementation decisions are motivated by MCMC performance:

-   **Non-centered parameterization** of the AR(1) innovations decorrelates $\sigma_g$ from the latent trajectory.
-   **Fixed CES noise** eliminates a near-zero parameter that creates funnel geometry.
-   **InverseGamma priors** on PP and QCEW noise prevent variance parameters from collapsing to zero.
-   **pytensor.scan** for the AR(1) recursion enables efficient gradient computation through the time series.

### 8.3 Expected Runtime

On an Apple M4 Max, the model completes sampling in approximately 2–5 minutes with 4 chains and 4,000 draws per chain.

------------------------------------------------------------------------

## 9. Diagnostics and Validation

The implementation produces the following diagnostic outputs:

1.  **Sampling diagnostics:** Divergence count, maximum tree depth, and effective sample sizes for all parameters.
2.  **Parameter summary:** Posterior means, standard deviations, and 80% HDI for all structural parameters ($\mu_g, \phi, \sigma_g, \text{bd}, \alpha_p, \lambda_p, \sigma_p, \sigma_f, \sigma_{\text{QCEW}}$).
3.  **Growth rate plots:** Latent SA and NSA total growth against observed CES; latent continuing-units growth against PP average; estimated vs. empirical seasonal pattern.
4.  **Reconstructed index:** Cumulated latent growth overlaid on observed CES, PP, and QCEW index levels.
5.  **Forecast plots:** SA and NSA index and employment level forecasts with 80% credible intervals.

------------------------------------------------------------------------

## 10. Relationship to the Staged Implementation Plan

This implementation corresponds most closely to a hybrid of Releases 1 and 2 from the companion specification, adapted to the operational setting of a single payroll provider with four panel variants:

| Feature | Companion Spec | This Implementation |
|------------------|-----------------------|-------------------------------|
| Scope | National | National |
| Latent dynamics | Random walk | AR(1) with mean reversion |
| Birth/death | Release 2: cyclical model | Constant offset |
| Seasonal adjustment | Implicit (SA data only) | Explicit seasonal component with SA + NSA data |
| Provider structure | Multiple independent providers | Single provider, four panel variants with common factor |
| QCEW | Release 2: lagged observation | Temporal aggregation constraint |
| Hierarchical priors | Over providers | Not needed (single provider) |
| Geographic/industry decomposition | Releases 3–5 | Not yet implemented |

Key departures from the companion specification reflect lessons learned during implementation: the AR(1) specification proved more appropriate than a random walk for growth rates, the explicit seasonal component enables direct use of NSA data, and the common factor structure correctly handles the correlation among panel variants from the same provider.

------------------------------------------------------------------------

## 11. The Nowcasting Window

The model's primary operational value arises during the nowcasting window: the approximately three-week period after payroll provider data becomes available for a reference month but before BLS publishes the official NFP estimate. This section describes the mechanics of what happens when new PP observations arrive for a month that lacks CES data.

### 11.1 Information Regime Shift

During the historical estimation window, each month is typically observed by all six series: CES SA, CES NSA, four PP panels, and (for quarter-end months) QCEW. The fixed CES noise of $\sigma_{\text{CES}} = 0.001$ means that CES dominates the posterior for any month where it is available — the PP series provide supplementary information but cannot meaningfully shift the latent state away from the CES anchor.

In the nowcasting window, this regime inverts. With CES absent, the four PP panels become the **sole monthly signal** informing the current-month latent state. The relative contribution of PP to the posterior jumps from modest (in-sample) to dominant (nowcast month). This is the regime where the model's treatment of PP measurement parameters — bias, loading, noise, and the common factor — matters most.

### 11.2 From Prior Predictive to Filtered Estimate

Without PP data for the nowcast month, the model's estimate is a pure prior predictive simulation from the AR(1) dynamics:

$$g_{T+1}^{\text{cont}} \mid \text{no data} \;\sim\; \mu_g + \phi\left(g_T^{\text{cont}} - \mu_g\right) + \sigma_g\,\varepsilon_{T+1}$$

This forecast is pulled toward the long-run mean $\mu_g$ by mean reversion at rate $\phi$, with uncertainty governed by $\sigma_g$. The credible interval reflects parameter uncertainty (different MCMC draws yield different $\mu_g, \phi, \sigma_g$), state uncertainty (the terminal filtered state $g_T^{\text{cont}}$ is itself a posterior distribution), and innovation uncertainty (the unknown shock $\varepsilon_{T+1}$).

When PP data arrives, the January latent state becomes a **filtered** estimate. The posterior combines the AR(1) prior with the PP likelihood via precision weighting. Schematically, for a single PP series:

$$\text{posterior precision} \approx \underbrace{\frac{1}{\sigma_g^2}}_{\text{AR(1) prior}} + \underbrace{\frac{\lambda_p^2}{\sigma_p^2 + \sigma_f^2}}_{\text{PP signal}}$$

With four panels sharing a common factor, the effective PP contribution is less than four times a single panel but substantially more than one. The common factor $\sigma_f^2$ appears in the denominator because it represents irreducible shared noise that cannot be diversified away by adding more panels from the same provider.

### 11.3 Constructing the NFP Nowcast

The PP series observe NSA continuing-units growth. Converting the filtered estimate to an NFP-comparable quantity requires two adjustments:

1.  **Seasonal adjustment.** The January seasonal effect $s_{\text{Jan}}$ is added (or equivalently, removed from the NSA signal) to produce SA growth. This component is well-identified from the full history of CES SA and NSA observations, so it contributes minimal additional uncertainty.

2.  **Birth/death offset.** The constant BD offset is added to convert from continuing-units growth to total employment growth. In this implementation, BD is time-invariant and estimated from the historical CES-minus-PP wedge. It represents the **dominant source of structural uncertainty** in the nowcast: the model assumes that the average historical birth/death contribution persists into the nowcast month, which may not hold at cyclical turning points.

The complete nowcast for SA total employment growth is:

$$\hat{g}_{T+1}^{\text{total,sa}} = \hat{g}_{T+1}^{\text{cont}} + \widehat{\text{bd}}$$

and the corresponding employment level is obtained by exponentiating the cumulated log growth path.

### 11.4 Uncertainty Decomposition in the Nowcast

The nowcast credible interval can be decomposed into several components, listed in approximate order of importance:

| Source | Description | Reducible? |
|-------------------|----------------------------|--------------------------|
| Birth/death uncertainty | The constant BD offset has posterior uncertainty; the true BD contribution may also vary cyclically | Partially — cyclical BD model (Release 2) would help |
| PP measurement noise | Idiosyncratic noise $\sigma_p$ in each panel | Slowly — more history tightens $\sigma_p$ posteriors |
| Common factor | Shared provider noise $\sigma_f$ cannot be diversified across panels | Only by adding **independent** providers |
| Signal loading uncertainty | Uncertainty in $\lambda_p$ scales the PP signal | Slowly — more history helps |
| AR(1) innovation | The realized shock $\varepsilon_{T+1}$ | Resolved by the PP data |
| Seasonal uncertainty | Posterior uncertainty on $s_{\text{Jan}}$ | Negligible with multi-year history |

The key insight is that adding more panel variants from the same provider has diminishing returns due to the common factor. The most impactful improvements come from adding **independent** payroll providers (which would contribute information uncorrelated with the existing common factor) and from modeling the birth/death contribution as time-varying.

### 11.5 Operational Workflow

When new PP data arrives for a nowcast month, the operational workflow is:

1.  Append the new PP index rows to `pp_index.csv`. The four panel columns should contain the January 2026 index values; other columns (CES, QCEW) will be missing for this month.
2.  Re-run `load_data()`. The monthly calendar automatically extends to cover January. The `pp_obs` mask detects finite PP values; the `ces_sa_obs` and `ces_nsa_obs` masks remain false for January.
3.  Re-run `build_and_sample()`. The model includes January PP observations in the likelihood while leaving the January CES likelihood terms inactive. The latent state vector extends by one period.
4.  The forecast horizon shifts forward: January becomes a filtered estimate with reduced uncertainty, and the prior predictive forecast begins from January rather than December.

No code changes are required — the observation masks handle mixed data availability automatically. This is a deliberate design feature: the model specification is data-availability-agnostic, and the transition from forecast to nowcast to historical estimate occurs naturally as each data source publishes.

### 11.6 Expected Behavior

Based on the estimated model parameters, several qualitative predictions can be made about the nowcast update:

-   The posterior mean for January SA growth will be pulled away from the AR(1) prior toward the PP-implied growth, with the magnitude of the pull proportional to the PP signal-to-noise ratio $\lambda_p^2 / (\sigma_p^2 + \sigma_f^2)$.
-   If the PP series indicate growth meaningfully different from the AR(1) conditional mean $\mu_g + \phi(g_T - \mu_g)$, the nowcast will reflect this, tempered by the PP noise parameters.
-   The 80% credible interval on January SA growth should narrow substantially relative to the current pure forecast, with the residual width determined primarily by BD uncertainty and the common factor.
-   The employment level nowcast will inherit this reduced uncertainty, producing a tighter range for the headline NFP number.

Once BLS subsequently publishes the January NFP, re-running the model with CES data will further tighten the January estimate (the CES precision of $1/0.001^2 = 10^6$ will dominate), and the nowcasting window will shift forward to February.

------------------------------------------------------------------------

## 12. Limitations and Extensions

**Current limitations:**

-   The birth/death offset is time-invariant, so the model cannot capture cyclical variation in net business formation (important at turning points).
-   National-level only—no geographic or industry decomposition.
-   The seasonal pattern is static across years; it cannot capture evolving seasonal patterns (e.g., pandemic-era shifts).
-   No time-varying provider bias to accommodate compositional drift in the payroll sample.
-   QCEW enters as a contemporaneous constraint rather than a lagged anchor with forward propagation.

**Planned extensions (per the staged implementation plan):**

-   Cyclical birth/death model conditioned on macroeconomic indicators (Release 2).
-   Cell-level estimation with geographic × industry decomposition and MinT reconciliation (Release 3).
-   Nested hierarchical priors over geography and industry (Release 4).
-   Time-varying provider bias with QCEW error correction (Release 5).