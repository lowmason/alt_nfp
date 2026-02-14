# Review of CES / QCEW / Payroll-Provider Bayesian Nowcasting Model

## Executive Summary

This document reviews the **methodology**, **code implementation**, and **data usage** of the Bayesian state-space model implemented in `pp_estimation_v2.py` (now in `archive/`; the current v3 implementation is in `src/alt_nfp/`). The model integrates CES (SA and NSA), payroll-provider continuing-unit indices, and QCEW (including preliminary releases) to nowcast U.S. employment growth.

Special attention is given to:

-   The consequences of using **preliminary QCEW estimates** (2025-03-12, 2025-06-12)
-   The **CES benchmarking change effective with the 2026-01-12 release**
-   Whether excluding preliminary QCEW in the model approximates the new BLS methodology
-   How nowcasting should account for **anticipated benchmark revisions**

The short answer is:

> **Yes — excluding preliminary QCEW while using post-benchmark CES data closely approximates the new CES methodology, because the information content of those QCEW points has already been absorbed into CES.**

------------------------------------------------------------------------

## 1. Model Design Overview

### 1.1 Growth-Rate Formulation

The model is specified entirely in **monthly log growth rates**:

\[ g_t = \log(E_t) - \log(E\_{t-1})\]

This choice is foundational:

-   In **levels**, the gap between continuing-unit employment (payroll providers) and total employment (CES/QCEW) grows without bound.
-   In **growth rates**, the birth/death contribution becomes approximately stationary and can be modeled as a constant offset.

This resolves an identification failure present in level-based formulations.

------------------------------------------------------------------------

### 1.2 Latent State: Continuing-Unit Growth

The core latent process is **seasonally adjusted continuing-unit employment growth**, modeled as a stationary AR(1):

\[ g_t\^{\text{cont}} = \mu*g +\* \phi (g{t-1}\^{\text{cont}} - \mu\_g) + \sigma\_g \varepsilon\_t\]

Key properties:

-   Mean-reverting (no unit root)
-   Non-centered parameterization improves sampling
-   Appropriate for employment *growth* rather than employment *levels*

This latent process represents what payroll providers fundamentally measure.

------------------------------------------------------------------------

### 1.3 Seasonal Component

Seasonality is modeled explicitly:

-   11 free monthly seasonal effects
-   12th month defined by a sum-to-zero constraint

\[ g_t\^{\text{cont,nsa}} = g_t\^{\text{cont}} + s\_{m(t)}\]

This allows:

-   Joint identification using **CES SA and CES NSA**
-   Internal seasonal adjustment rather than reliance on external filters
-   Stable identification even during pandemic distortions

------------------------------------------------------------------------

### 1.4 Birth / Death Offset

A constant birth/death (BD) offset bridges continuing-unit growth to total employment growth:

\[ g_t\^{\text{total,sa}} = g_t\^{\text{cont}} + \text{bd}\]

Properties:

-   Time-invariant in this implementation
-   Prior centered around +0.1% per month
-   Identified primarily by the CES–PP wedge

This reflects the average net contribution of firm births and deaths to employment growth.

------------------------------------------------------------------------

## 2. Measurement Equations

### 2.1 CES (SA and NSA)

CES is treated as the **definitional anchor**:

\[ y_t\^{\text{CES}} = g_t\^{\text{total}} + \varepsilon\_t, \quad \varepsilon\_t \sim \mathcal{N}(0, 0.001\^2)\]

Design choices:

-   CES noise variance is **fixed**
-   Prevents sampler pathologies
-   Forces the model to match CES when CES is available

CES SA identifies trend growth + BD. CES NSA identifies seasonality.

------------------------------------------------------------------------

### 2.2 Payroll-Provider Panels

Each payroll-provider panel observes NSA continuing-unit growth:

\[ y\_{p,t} = \alpha\_p + \lambda*p g_t\^{\*\text{cont,nsa}} + f_t + \varepsilon{p,t}\]

Where:

-   $\alpha_p$: provider bias
-   $\lambda_p$: signal loading
-   $f_t$: **common provider factor**
-   $\varepsilon_{p,t}$: idiosyncratic noise

The **common factor** is critical:

-   Prevents four panels from being treated as four independent sources
-   Correctly inflates uncertainty
-   Reflects shared administrative artifacts

------------------------------------------------------------------------

### 2.3 QCEW (Quarterly Aggregation)

QCEW enters as a **temporal aggregation constraint**:

\[ y_q\^{\text{QCEW}} = \sum\_{t \in q} g_t\^{\text{total,nsa}} + \varepsilon\_q\]

This:

-   Anchors cumulative growth
-   Prevents drift
-   Identifies BD and seasonality more tightly

QCEW noise is small but non-zero, allowing for preliminary revisions.

------------------------------------------------------------------------

## 3. Code Implementation Review

### 3.1 Structure and Efficiency

The implementation in `pp_estimation_v2.py` is strong:

-   Clear separation of **data loading**, **model definition**, and **forecasting**
-   Vectorized likelihoods
-   Efficient use of `pytensor.scan` for AR(1)
-   Non-centered parameterization throughout
-   Inverse-Gamma priors prevent variance collapse

The design is intentionally MCMC-friendly.

------------------------------------------------------------------------

### 3.2 Observation Masking

A particularly good design choice:

-   Likelihood terms activate only when data exist

-   Same model handles:

    -   Historical estimation
    -   Nowcasting
    -   Forecasting

-   No branching logic required

This allows seamless transition from forecast → nowcast → historical revision.

------------------------------------------------------------------------

### 3.3 Common-Factor Treatment

The provider common factor is one of the strongest features:

-   Prevents overconfidence
-   Correctly models correlation across PP panels
-   Essential for honest uncertainty quantification

Without this, the model would materially understate uncertainty.

------------------------------------------------------------------------

## 4. Effects of Using Preliminary QCEW (2025-03-12, 2025-06-12)

### 4.1 What Including Preliminary QCEW Does

Including these points:

-   Anchors late-2024 and early-2025 growth earlier
-   Anticipates CES benchmark revisions
-   Improves nowcast accuracy for mid-2025 onward
-   Tightens posterior uncertainty

Even preliminary QCEW is vastly more informative than CES alone.

------------------------------------------------------------------------

### 4.2 Risk Assessment

Risks are limited:

-   Preliminary QCEW can revise, but errors are small relative to CES sampling error
-   Bayesian noise term absorbs modest revisions
-   Net effect is strongly positive

------------------------------------------------------------------------

## 5. CES Benchmarking Change (Effective 2026-01-12)

### 5.1 What Changed

Starting with the 2026-01-12 release:

-   CES incorporates **preliminary QCEW directly**
-   Benchmarking occurs earlier
-   Large annual level corrections are reduced

CES becomes closer to a rolling-benchmark series.

------------------------------------------------------------------------

### 5.2 Key Question: Should the Model Still Include Preliminary QCEW?

**Answer: Not necessarily.**

If CES already reflects preliminary QCEW, then including those same QCEW points separately would **double-count the information**.

------------------------------------------------------------------------

## 6. Excluding Preliminary QCEW as an Approximation

### 6.1 Core Result

> **Running the model *without* preliminary QCEW but *with post-benchmark CES* closely approximates running the old model *with* preliminary QCEW.**

Why?

-   The informational content of QCEW has been transferred into CES
-   CES growth rates already reflect those corrections
-   The model no longer needs an external anchor for those quarters

------------------------------------------------------------------------

### 6.2 When This Approximation Holds

This approximation is valid when:

-   CES has already incorporated the preliminary QCEW used for benchmarking
-   You are modeling post-revision data (e.g., after 2026-02-12)

It does **not** mean QCEW is unimportant — only that it is already embedded.

------------------------------------------------------------------------

## 7. Implications for Nowcasting and Forecasting

### 7.1 Should the Model Forecast Benchmark-Adjusted Estimates?

**Yes — and it already does.**

Because the model:

-   Anchors to QCEW when CES lags
-   Estimates BD explicitly
-   Works in growth rates

…it naturally forecasts toward where CES will end up post-benchmark, not where it is pre-benchmark.

------------------------------------------------------------------------

### 7.2 CES 2025: Final or Not?

After the 2026 benchmark:

-   CES 2025 should be treated as **effectively final**
-   Any further revisions will be minor
-   The large level correction has already occurred

Re-estimating the model with revised CES is essential.

------------------------------------------------------------------------

## 8. Recommendations

1.  **Post-benchmark runs**

    -   Use revised CES
    -   Exclude preliminary QCEW already used in benchmarking

2.  **Pre-benchmark runs**

    -   Include preliminary QCEW aggressively
    -   This anticipates the benchmark

3.  **Future extensions**

    -   Time-varying birth/death offset
    -   Explicit CES bias term for recent months
    -   Hierarchical extension across geography/industry

------------------------------------------------------------------------

## 9. Bottom Line

This model is:

-   Methodologically sound
-   Correctly specified for nowcasting
-   Robust to benchmark revisions
-   Already aligned with the new CES philosophy

The growth-rate formulation + QCEW anchoring effectively **internalizes benchmark logic** rather than reacting to it after the fact.

In short: **you built the right model for the world BLS is moving toward.**