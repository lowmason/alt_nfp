# Predicting CES Revisions with the QCEW-Anchored Model: Methodology

## 1. Motivation

The Current Employment Statistics (CES) survey publishes nonfarm payroll employment in three successive vintages: a **first print** (preliminary estimate, released \~3 weeks after the reference month), a **second print** (first revision, one month later), and a **final print** (second revision, two months later). Beyond these monthly revisions, CES undergoes an annual **benchmark revision** that reconciles estimates with the near-census Quarterly Census of Employment and Wages (QCEW). Revisions are economically meaningful: the March 2024 preliminary benchmark revision was −818,000 jobs, the largest downward adjustment since 2009.

The QCEW-anchored Bayesian state space model (v2, documented in `methods_results.md`) treats CES as a noisy, biased observer of QCEW-defined truth. This architecture implies that the model's posterior latent state—informed by QCEW, payroll provider (PP) data, and CES itself—should systematically differ from CES in the *direction of future revisions*. If CES first-print overstates growth (as the estimated $\alpha_{\text{CES}} = +0.034\%$/month suggests), the QCEW-anchored latent will sit below CES, implicitly predicting a downward revision.

This document describes a backtest framework for evaluating whether the model can predict CES second-print and final-print revisions, and how the quality of the QCEW anchor (preliminary vs. final) affects that prediction.

------------------------------------------------------------------------

## 2. Conceptual Framework

### 2.1 CES Revisions as Noise Reduction

In the model's measurement equation, CES observes the latent total employment growth with bias, scale, and noise:

$$y_t^{\text{CES,SA}} = \alpha_{\text{CES}} + \lambda_{\text{CES}} \cdot g_t^{\text{total,sa}} + \varepsilon_t^{\text{SA}}, \qquad \varepsilon_t^{\text{SA}} \sim \mathcal{N}(0, \sigma_{\text{CES,SA}}^2)$$

Each CES vintage can be understood as a draw from this measurement process with vintage-specific noise:

$$y_t^{(v)} = \alpha_{\text{CES}}^{(v)} + \lambda_{\text{CES}}^{(v)} \cdot g_t^{\text{total,sa}} + \varepsilon_t^{(v)}, \qquad v \in \{\text{first}, \text{second}, \text{final}\}$$

As BLS incorporates additional survey responses and corrects tabulation, the revision process reduces measurement error. We expect:

$$\sigma_{\text{CES}}^{(\text{first})} \geq \sigma_{\text{CES}}^{(\text{second})} \geq \sigma_{\text{CES}}^{(\text{final})}$$

and potentially a declining bias $|\alpha^{(v)}|$ as well (benchmark revisions correct systematic error).

The model does not need to know the revision structure explicitly. When fit with first-print CES data, the posterior latent state is pulled toward QCEW truth. The gap between the latent posterior and CES first-print is an *implied revision prediction*:

$$\widehat{\Delta}_t^{(v \to v')} = \mathbb{E}\left[g_t^{\text{total,sa}} \mid \mathcal{D}^{(v)}\right] - y_t^{(v)}$$

where $\mathcal{D}^{(v)}$ denotes the full dataset with CES at vintage $v$. The actual revision is:

$$\Delta_t^{(v \to v')} = y_t^{(v')} - y_t^{(v)}$$

### 2.2 Role of the QCEW Anchor

The model's ability to predict revisions depends critically on the QCEW anchor, since QCEW is what pulls the latent away from CES. QCEW itself undergoes revision: **preliminary** QCEW files reflect initial quarterly filings, while **final** QCEW incorporates corrections and late-filed reports.

In the model, QCEW observation noise is fixed by calibration:

$$\sigma_{\text{QCEW},t} = \begin{cases} 0.0005 & \text{quarter-end month (M3)} \\ 0.0015 & \text{mid-quarter month (M1–2)} \end{cases}$$

Preliminary QCEW data is noisier than final. This can be reflected by using wider fixed sigmas when the model is fit with preliminary QCEW:

$$\sigma_{\text{QCEW},t}^{(\text{prelim})} = \begin{cases} 0.001 & \text{M3 (quarter-end)} \\ 0.003 & \text{M1–2 (mid-quarter)} \end{cases}$$

These are twice the baseline values, reflecting the additional uncertainty before QCEW data is finalized. The exact calibration should be informed by empirical preliminary-to-final QCEW revision magnitudes.

### 2.3 The Revision Prediction Mechanism

The model predicts revisions through three channels:

1.  **QCEW anchor correction.** QCEW pulls the latent toward administrative truth. When CES first-print overstates growth, the latent sits below CES—predicting a downward revision. This is the dominant channel at the national level, where QCEW contributes \~38% of the precision budget.

2.  **PP triangulation.** Payroll provider data provides an independent read on continuing-units employment growth. If PP and QCEW agree on a growth rate that differs from CES, the model's confidence in the implied revision increases. PP's contribution is modest nationally (\~1.7%) but grows at finer geographic/industry cuts.

3.  **Estimated CES parameters.** The model's $\alpha_{\text{CES}}$ captures average CES bias. Even in months where QCEW is not available, the model discounts CES by $\alpha_{\text{CES}}$, predicting a revision of approximately $-\alpha_{\text{CES}}$ in the direction of the historical bias.

------------------------------------------------------------------------

## 3. Data Requirements

### 3.1 CES Vintage Files

Three CES files, each with the same schema as the current `ces_index.csv`:

| File | Description | Timing |
|------------------|----------------------------------|---------------------|
| `ces_index_first.csv` | Preliminary estimate | Released \~3 weeks after reference month |
| `ces_index_second.csv` | First revision | Released \~7 weeks after reference month |
| `ces_index_final.csv` | Second revision (pre-benchmark) | Released \~11 weeks after reference month |

Each file contains `ref_date`, `ces_sa_index`, `ces_nsa_index`, and level columns. For months within the backtest window (2024-10 through 2025-12), the values differ across vintages. For months before the window, the values are identical (historical data has already converged to final).

**Benchmark revisions.** The annual benchmark revision, which reconciles CES to QCEW levels, is a separate and larger process than the monthly first-to-final revision cycle. This backtest targets the monthly revision cycle. A separate analysis could use pre- and post-benchmark CES files to evaluate benchmark revision prediction, though that requires a longer time horizon and is better assessed annually.

### 3.2 QCEW Vintage Files

Two QCEW files with the same schema as the current `qcew_index.csv`:

| File | Description | Timing |
|------------------|----------------------------------|---------------------|
| `qcew_index_preliminary.csv` | Initial quarterly release | \~6 months after quarter end |
| `qcew_index_final.csv` | Revised (after corrections and late filings) | \~12 months after quarter end |

For quarters within the backtest window, preliminary and final values differ. For earlier quarters, values are identical.

### 3.3 PP Data

PP data does not undergo revision in this framework. The same `pp_index_1.csv` and `pp_index_2.csv` files are used in all runs.

### 3.4 Coverage Period

The proposed backtest window is **October 2024 – December 2025** (15 months). This requires:

-   CES first, second, and final prints for all 15 months
-   QCEW preliminary and final for the quarters covering this window (2024-Q4 through 2025-Q4, i.e., 5 quarters)
-   PP1 and PP2 for the same period (subject to availability; PP1 ends 2025-06, PP2 extends through 2025-12 in the current dataset)

------------------------------------------------------------------------

## 4. Backtest Design

### 4.1 Vintage Matrix

Each month $t$ in the backtest window is evaluated under four data configurations, forming a 2 × 2 matrix:

|                      | QCEW Preliminary | QCEW Final |
|----------------------|------------------|------------|
| **CES First Print**  | Run A            | Run B      |
| **CES Second Print** | Run C            | Run D      |

This yields four model fits per month, each producing a posterior latent state $\mathbb{E}[g_t^{\text{total,sa}} \mid \mathcal{D}]$ and associated uncertainty.

### 4.2 Revision Targets

For each month and configuration, we compute:

| Quantity | Definition | Interpretation |
|--------------------|----------------------|-------------------------------|
| Implied revision (first → final) | $\widehat{\Delta}_t^{(1 \to 3)} = \mathbb{E}[g_t^{\text{total,sa}} \mid \mathcal{D}^{(\text{first})}] - y_t^{(\text{first})}$ | Model's prediction of total CES revision |
| Implied revision (second → final) | $\widehat{\Delta}_t^{(2 \to 3)} = \mathbb{E}[g_t^{\text{total,sa}} \mid \mathcal{D}^{(\text{second})}] - y_t^{(\text{second})}$ | Model's prediction of remaining revision |
| Actual revision (first → second) | $\Delta_t^{(1 \to 2)} = y_t^{(\text{second})} - y_t^{(\text{first})}$ | Observed first revision |
| Actual revision (first → final) | $\Delta_t^{(1 \to 3)} = y_t^{(\text{final})} - y_t^{(\text{first})}$ | Observed total revision |
| Actual revision (second → final) | $\Delta_t^{(2 \to 3)} = y_t^{(\text{final})} - y_t^{(\text{second})}$ | Observed remaining revision |

All quantities are in monthly log growth rates. For reporting, they are also converted to jobs added (thousands) by applying the employment-to-index scaling factor.

### 4.3 Procedure

For each month $t$ in the backtest window:

1.  **Construct vintage-specific datasets.** Swap `ces_index.csv` with the appropriate vintage file. Swap `qcew_index.csv` with the appropriate vintage file. Adjust QCEW observation sigmas based on vintage (Section 2.2).

2.  **Fit the model.** Run the v2 model with lighter sampling parameters (2,000 draws, 2,000 tune, 2 chains, target_accept = 0.95) to keep the 60-run loop tractable (\~30 seconds per fit, \~30 minutes total).

3.  **Extract the implied revision.** From the posterior, compute $\mathbb{E}[g_t^{\text{total,sa}}]$ and the 80% HDI at the target month. Compute the implied revision as the difference from the CES vintage value.

4.  **Record revision prediction uncertainty.** The posterior HDI on the latent state at month $t$ provides a natural uncertainty band on the revision prediction:

    $$\widehat{\Delta}_t^{\text{lo}} = g_t^{\text{total,sa,10\%}} - y_t^{(v)}, \qquad \widehat{\Delta}_t^{\text{hi}} = g_t^{\text{total,sa,90\%}} - y_t^{(v)}$$

5.  **Compare to actual revision.** Score the implied revision against the observed revision.

### 4.4 No Censoring Required

Unlike the nowcast backtest (which censors CES from the target month onward to simulate CES unavailability), the revision backtest *includes* CES at the vintage being tested. The model uses CES data—it just doesn't trust it completely, because QCEW and PP provide independent information. This is the standard operating mode of the model.

------------------------------------------------------------------------

## 5. Evaluation Metrics

### 5.1 Primary Metrics

| Metric | Definition | Interpretation |
|------------------|----------------------|--------------------------------|
| MAE (jobs, thousands) | $\frac{1}{N}\sum_t \lvert \widehat{\Delta}_t - \Delta_t \rvert$ | Average absolute revision prediction error |
| RMSE (jobs, thousands) | $\sqrt{\frac{1}{N}\sum_t (\widehat{\Delta}_t - \Delta_t)^2}$ | Emphasizes large errors |
| Directional accuracy | $\frac{1}{N}\sum_t \mathbf{1}[\text{sign}(\widehat{\Delta}_t) = \text{sign}(\Delta_t)]$ | Does the model get the direction right? |
| Coverage | Fraction of actual revisions within the 80% HDI | Calibration of uncertainty |

### 5.2 Decomposition Axes

Results are reported along several dimensions:

1.  **CES vintage (first vs. second print).** First-to-final revisions are larger and harder to predict than second-to-final. The model should perform relatively better on the larger (first → final) revision because the signal-to-noise ratio is higher.

2.  **QCEW vintage (preliminary vs. final).** Final QCEW provides a tighter anchor, so revision predictions should improve when the model has access to final QCEW. The magnitude of improvement quantifies how much QCEW revision quality matters.

3.  **QCEW availability at the target month.** Months where QCEW is available at the target month $t$ (directly informing the latent at that point) vs. months where QCEW coverage stops before $t$ (latent at $t$ extrapolated from the AR(1) dynamics). The model's revision prediction should be sharper when QCEW directly observes the target month.

4.  **PP availability.** Months where PP1 is available (through 2025-06) vs. PP2-only (2025-07–2025-12). Tests the marginal contribution of the richer PP data.

### 5.3 Baselines

To contextualize model performance, compare against simple baselines:

-   **Naive (zero revision):** Predict no revision ($\widehat{\Delta} = 0$). This is the baseline if revisions are unpredictable.
-   **Historical mean revision:** Predict the average historical CES first-to-final revision magnitude (unconditional).
-   **AR(1) on past revisions:** Fit a simple AR(1) to the historical revision series and forecast.

If the model cannot beat the naive baseline, it has no revision-prediction value. If it beats the naive but not the historical-mean baseline, its value comes only from knowing the average revision pattern, not from conditioning on QCEW/PP data.

------------------------------------------------------------------------

## 6. Expected Results and Interpretation

### 6.1 What Success Looks Like

The model should:

-   **Predict the direction of first-to-final revisions more often than not** (directional accuracy \> 50%). Given the systematic positive bias ($\alpha_{\text{CES}} > 0$), the model should predict downward revisions on average, consistent with the historical pattern of negative CES benchmark revisions.

-   **Produce smaller revision prediction errors with final QCEW than preliminary QCEW.** The improvement quantifies the value of QCEW data quality for revision prediction.

-   **Show calibrated uncertainty.** The 80% HDI on implied revisions should contain \~80% of actual revisions. Overconfident intervals (low coverage) suggest the model underestimates CES measurement noise; underconfident intervals (high coverage) suggest the model is too conservative.

### 6.2 Interpreting the CES Vintage Dimension

The first-to-second revision reflects BLS incorporating late survey responses. The second-to-final revision reflects further corrections. The model's revision prediction should capture both, but the mechanisms differ:

-   **First → second:** Dominated by sample expansion. The model's prediction comes primarily from QCEW and PP data disagreeing with the first print.
-   **Second → final:** Smaller magnitude. The model's residual prediction comes from the same QCEW/PP triangulation but with less room for improvement.
-   **First → final (annual benchmark):** Not directly tested in this monthly backtest but is the largest and most policy-relevant revision. A separate analysis using pre- and post-benchmark files would address this.

### 6.3 Interpreting the QCEW Vintage Dimension

The QCEW preliminary-to-final comparison isolates the anchor quality effect:

-   If revision predictions are **substantially better with final QCEW**, the implication is that the model's revision-prediction power comes primarily from the QCEW anchor, and improving QCEW timeliness or quality would directly improve revision prediction.

-   If revision predictions are **similar across QCEW vintages**, the implication is that even preliminary QCEW provides a sufficiently strong anchor, and the model's revision-prediction value is robust to QCEW noise. This would be a strong result—it would mean the model works well with the QCEW data available in real time.

### 6.4 Limitations of This Backtest

-   **Small sample.** 15 months provides limited statistical power. Confidence intervals on MAE and directional accuracy will be wide. Results should be interpreted as indicative, not definitive.

-   **Monthly revisions are small.** CES first-to-final revisions are typically 10–40k jobs (0.01–0.03% growth). The model's posterior uncertainty at any given month ($\sigma_{\text{CES,SA}} \approx 0.085\%$) is larger than the typical revision magnitude. This means the model's *point prediction* of the revision may have low signal-to-noise, even if the *direction* is correct.

-   **Not a benchmark revision test.** This tests the monthly first/second/final print cycle, not the annual CES-to-QCEW benchmark reconciliation. The benchmark revision is economically more important but requires a multi-year evaluation horizon.

-   **Parameter stability assumption.** The model's CES parameters ($\alpha_{\text{CES}}$, $\lambda_{\text{CES}}$, $\sigma_{\text{CES}}$) are estimated as time-invariant. If the revision process has changed over the backtest window (e.g., due to declining CES response rates), the model may not capture the shift.

------------------------------------------------------------------------

## 7. Extensions

### 7.1 Vintage-Specific CES Parameters

The baseline approach uses the same model for all vintages—the only thing that changes is the CES data fed in. A more sophisticated approach would explicitly model vintage-specific CES noise:

$$y_t^{(v)} = \alpha_{\text{CES}}^{(v)} + \lambda_{\text{CES}} \cdot g_t^{\text{total,sa}} + \varepsilon_t^{(v)}, \qquad \sigma^{(\text{first})} > \sigma^{(\text{second})} > \sigma^{(\text{final})}$$

This could be implemented by feeding *all three vintages simultaneously* as separate observation likelihoods with ordered noise parameters. The model would then learn the revision structure directly. However, this requires substantially more data (all three vintages for every month in the training sample, not just the 15-month test window) and may not improve over the simpler approach.

### 7.2 Direct Revision Modeling

Rather than inferring revisions implicitly from the latent-vs-CES gap, one could model the revision process explicitly:

$$\Delta_t^{(1 \to 3)} = \beta_0 + \beta_1\left(\mathbb{E}[g_t^{\text{total,sa}}] - y_t^{(\text{first})}\right) + \beta_2 \cdot z_t + \eta_t$$

where $z_t$ includes predictors like the month's QCEW residual, PP-vs-CES disagreement, or CES response rate. This second-stage regression could improve calibration of the revision magnitude, but it requires a training sample of historical vintages.

### 7.3 State × Supersector Extension

At the state × supersector level, CES sample sizes shrink by 1–2 orders of magnitude, and $\sigma_{\text{CES}}$ could be 0.5–2%. Revisions at this level are correspondingly larger (both in growth rates and in absolute terms). The model's revision-prediction value should increase substantially at finer geographic and industry cuts, because:

-   QCEW's coverage advantage is even more pronounced (near-census vs. small CES samples)
-   CES revisions at the cell level are larger and more variable
-   PP data, once disaggregated, becomes a larger share of the precision budget

------------------------------------------------------------------------

## 8. Implementation Notes

### 8.1 Model Changes

The v2 model (`pp_estimation_v2.py`, now in `archive/`; see `src/alt_nfp/` for v3) requires no structural changes. The modifications are confined to data loading:

1.  `load_data()` accepts vintage-specific file paths for CES and QCEW.
2.  QCEW observation sigmas are parameterized by vintage (wider for preliminary).
3.  The backtest loop iterates over months and vintage configurations.

The model specification, priors, sampling, and all observation equations remain identical to the gold-standard v2.

### 8.2 Computational Budget

| Item                             | Count | Time/run | Total    |
|----------------------------------|-------|----------|----------|
| Months in window                 | 15    | —        | —        |
| Vintage configurations per month | 4     | —        | —        |
| Total model fits                 | 60    | \~30 sec | \~30 min |

Lighter sampling (2,000 draws, 2 chains) is used to keep the loop tractable. Diagnostics should be spot-checked on a subset of runs to confirm convergence is adequate.

### 8.3 Output Artifacts

| Artifact | Description |
|-------------------------------|----------------------------------------|
| Console table | Per-month revision predictions vs. actuals, broken out by vintage configuration |
| Summary statistics | MAE, RMSE, directional accuracy, coverage by decomposition axis |
| `revision_backtest.png` | Scatter plot: implied revision vs. actual revision, color-coded by configuration |
| `revision_by_vintage.png` | Bar chart: MAE by vintage configuration (2 × 2 matrix) |
| `revision_timeseries.png` | Time series: actual vs. predicted revisions over the backtest window |

------------------------------------------------------------------------

## 9. Relationship to Existing Backtests

| Dimension | Nowcast Backtest (v3) | Revision Backtest (this document) |
|----------------|----------------------|----------------------------------|
| **Question** | How well does the model predict CES *levels* when CES is missing? | How well does the model predict CES *revisions* when CES is present but preliminary? |
| **CES treatment** | Censored (removed from target month onward) | Included at specific vintage |
| **What varies** | CES availability (present/absent) | CES vintage (first/second/final) and QCEW vintage (preliminary/final) |
| **Primary metric** | MAE in jobs added (nowcast vs. actual release) | MAE in revision prediction (implied vs. actual revision) |
| **Value proposition** | PP data substitutes for missing CES | Model triangulation predicts where CES is heading |
| **Baseline** | No-change forecast | Zero-revision prediction |

The two backtests are complementary. The nowcast backtest measures the model's *replacement* value (filling in for missing CES). The revision backtest measures its *refinement* value (improving upon available CES by using QCEW and PP as cross-checks).

------------------------------------------------------------------------

## 10. Summary

The QCEW-anchored model is architecturally suited to predict CES revisions because its core function—extracting latent truth from multiple noisy sources—is precisely the operation that CES revisions perform over time. The model does it in one step using QCEW as truth; BLS does it iteratively through sample expansion and annual benchmarking.

The proposed backtest evaluates this capability over 15 months with a 2 × 2 vintage matrix (CES first/second × QCEW preliminary/final), producing 60 model fits. Success is measured by MAE in predicted revisions, directional accuracy, and calibration of uncertainty bands, compared against naive and historical-mean baselines.

If the model can reliably predict the direction and approximate magnitude of CES revisions, this has direct practical value: it provides real-time guidance on whether the latest CES release is likely to be revised up or down, and by roughly how much—information that is currently unavailable until the revision is published weeks or months later.