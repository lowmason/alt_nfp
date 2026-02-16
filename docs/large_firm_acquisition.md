# Acquiring Large-Firm Employment Microdata

## A Relationship-Based Recruitment Strategy

------------------------------------------------------------------------

## Abstract

Payroll providers' client bases are systematically skewed toward small and medium-sized establishments, leaving large firms poorly observed in high-frequency microdata. This paper proposes acquiring employment microdata directly from large firms through a structured relationship-based recruitment approach. This strategy is preferred over probability sampling because the universe of large firms is small and well-enumerated, the data request is high-friction, and the Bayesian state space nowcasting framework does not require representative inputs. The target is recurring microdata spanning January 2016 through the present, producing time series that enter the model alongside payroll provider aggregates.

------------------------------------------------------------------------

## 1. Introduction

The Bayesian NFP nowcasting system extracts a latent employment growth signal from multiple noisy observed series — payroll provider continuing-units aggregates, official NFP releases, and lagged QCEW observations. Each series enters through a measurement equation with estimated bias, loading, and noise parameters. The model does not require any input to be representative; it learns each series's mapping to the latent state and weights accordingly.

No major payroll provider has significant coverage of firms with 500+ employees. The largest national provider serves large employers but offers pre-aggregated data better suited to dynamic factor models than our establishment-level state space framework, and its large firms are idiosyncratic. Per SUSB, firms with 500+ employees account for over half of private-sector employment — too large a share to leave unobserved. Per the 2022 Economic Census, firms with 500+ employees account for 36% of private employment.

We propose recruiting individual large firms to share historical and ongoing payroll microdata, processed into time series that enter the model as additional observed series. A structured convenience approach leveraging existing business relationships is the optimal recruitment strategy.

------------------------------------------------------------------------

## 2. The Large-Firm Coverage Gap

### 2.1 Payroll Provider Size Distribution

Payroll processors primarily serve small and medium-sized businesses that outsource payroll functions. Our primary provider's client base is concentrated in the 1–49 employee range, with coverage falling well below QCEW benchmarks for establishments with 100+ employees. This is structural: large firms typically run in-house payroll or enterprise HR/payroll platforms outside conventional provider partnerships.

### 2.2 Quantifying the Gap

SUSB data provide the definitive firm-level size distribution. Firms with 500+ employees number approximately 20,000 but employ roughly 55 million workers. This segment has low entry/exit rates, relatively stable firm-level employment, and concentration in healthcare, finance, retail, and manufacturing. The stability means even lagged information is useful, but the employment share means ignoring the segment entirely introduces substantial nowcast uncertainty.

### 2.3 Implications for the Model

At the national level (Release 1–2), provider series that underrepresent large firms produce growth rates reflecting systematic compositional differences from true national growth. The measurement equation absorbs some of this as estimated bias, but time-varying compositional drift (e.g., a provider gaining small-firm market share) creates non-stationary bias that a fixed parameter cannot capture. At the disaggregated level (Release 3+), large-firm series populate region × industry cells where providers are weakest, with precision weighting giving them appropriate influence.

------------------------------------------------------------------------

## 3. Why Probability Sampling Is Infeasible

### 3.1 Small, Heterogeneous Universe

Approximately 20,000 firms have 500+ employees. Unlike household surveys with large, relatively homogeneous populations, large firms are highly heterogeneous — a 600-person regional hospital and a 200,000-person retailer share a size class but little else. Adequate stratification would require a sample approaching a substantial fraction of the universe.

### 3.2 High-Friction Data Request

We are requesting historical payroll microdata — establishment-level records from January 2016 through the present, delivered monthly with consistent definitions aligned to CES reference periods. This requires legal review, data governance approval, technical extraction, and ongoing operational commitment. Cold-outreach response rates for such a request would be negligibly low.

### 3.3 Nonresponse Bias Dominance

At a realistic 5–10% cold-outreach response rate, nonresponse bias would vastly exceed any gains from a probability design. Willing responders would be systematically unrepresentative — more data-forward, less proprietary, in industries where sharing is normalized. The result would be a convenience sample with extra steps and lower yield.

### 3.4 The Model Does Not Require Representative Inputs

The state space framework accepts heterogeneous, non-representative series. Each enters with its own measurement equation parameters. The model extracts a common signal without assuming any series is representative. The relevant criterion is information content, not representativeness.

------------------------------------------------------------------------

## 4. Structured Relationship-Based Recruitment

### 4.1 Core Principle

***A firm that shares data is infinitely more informative than a randomly selected firm that refuses.*** The binding constraint is willingness to participate. Relationship-based recruitment maximizes the probability of successful acquisition by approaching firms through established trust channels.

### 4.2 Why Relationships Matter

Existing business relationships provide pre-established trust, internal navigation to decision-makers, opportunity to iterate on scope and format, and sustainable ongoing delivery supported by an active partnership. For a data request of this magnitude — historical microdata with recurring delivery — these advantages are decisive.

### 4.3 Imposing Structure on Convenience

Using SUSB tabulations, we characterize the large-firm universe along three dimensions: industry (NAICS supersector/sector), geography (Census region/division), and size (500–999, 1,000–4,999, 5,000+). For each cell in the resulting matrix, we assess employment share, existing provider coverage, and relationship availability. This produces a prioritized recruitment target list directing relationship activation toward cells with the greatest marginal information value.

Recruitment priority is determined by:

1.  **Employment share:** Higher-employment cells contribute more to the national aggregate.
2.  **Provider coverage gap:** Cells with the weakest existing coverage offer the greatest marginal gain.
3.  **Feasibility:** Firms with existing relationships are approached first; cold outreach is deprioritized, not excluded.

### 4.4 Signal Diversity Over Representativeness

If all recruited series come from a single industry, they provide correlated signals that add less to latent state identification than diversified series would. The SUSB-guided framework targets recruitment across multiple cells, maximizing independent signal content — the correct optimization criterion for a measurement-error framework.

------------------------------------------------------------------------

## 5. Data Acquisition Requirements

### 5.1 Temporal Coverage

The target window is **January 2016 through the present**, with ongoing monthly delivery. The historical backfill is essential: the model requires sufficient series length to identify provider-specific parameters, and the full window spans pre-pandemic, pandemic, and recovery periods critical for calibrating shock response. A minimum of 36 months is required; the full window is preferred.

### 5.2 Data Granularity

The ideal delivery includes: establishment-level monthly payroll records, establishment identifiers or sufficient detail to construct pseudo-establishments, NAICS codes at the finest available level, establishment geography (state/county/ZIP), and pay period dates enabling alignment to the reference week containing the 12th of each month.

Where full microdata is infeasible, pre-aggregated monthly employment counts by establishment are acceptable. The minimum viable product is a monthly total firm employment series with consistent definition over the historical window.

### 5.3 Reference Period Alignment

Official statistics anchor to the pay period containing the 12th. Firm data must align to this directly or include sufficient pay period detail for post-hoc alignment. Even one-period misalignment introduces systematic timing artifacts absorbed as noise.

### 5.4 Consistency and Continuity

Changes in employment definition (headcount vs. FTE, treatment of temporary workers) create structural breaks. Firms should document definitional changes, system migrations, and corporate actions (mergers, divestitures) affecting the series.

------------------------------------------------------------------------

## 6. Series Construction and Model Integration

### 6.1 From Microdata to Observed Series

Firm-level microdata are processed using the same framework applied to payroll provider data. For each firm or group of firms, we construct a monthly continuing-units employment growth series using the rotating, frozen measurement panel methodology. This ensures definitional consistency with existing provider series.

### 6.2 Measurement Equation

Each large-firm series enters through the standard measurement equation:

> *y_s,t = α_s + λ_s μ_t + ε_s,t*

where *α_s* is series-specific bias, *λ_s* is signal loading, and *ε_s,t* is observation noise. Parameters are estimated alongside provider parameters with hierarchical priors enabling partial pooling.

### 6.3 Hierarchical Treatment

Large-firm series can be treated as additional "providers" pooled hierarchically with existing parameters, or at the disaggregated level they populate specific region × industry cells. In either case, precision weighting ensures noisy or idiosyncratic series receive appropriately discounted weight.

------------------------------------------------------------------------

## 7. Practical Considerations

### 7.1 Legal Framework

Each partnership requires a data sharing agreement covering permitted uses, security requirements, anonymization protections, retention schedules, and audit rights. The framework should be templated to reduce negotiation friction.

### 7.2 Operational Sustainability

The system requires ongoing monthly delivery. Sustainability depends on minimizing firm-side burden (ideally automated extraction), maintaining the supporting relationship, providing value back to participants (e.g., early access to findings, benchmarking), and contingency planning for firms that discontinue.

### 7.3 Minimum Viable Portfolio

We estimate that 10–15 diversified large-firm series spanning at least five supersectors would meaningfully reduce posterior uncertainty on the national latent state, with diminishing returns beyond approximately 25–30 series.

### 7.4 Firm-Level Idiosyncrasy

Individual large firms experience idiosyncratic shocks that may not reflect broader dynamics. The noise parameter absorbs transitory variation, but persistent firm-specific trends can bias loadings. For the most idiosyncratic firms, modeling a firm-specific drift component or restricting influence during known corporate events may be appropriate.

------------------------------------------------------------------------

## 8. Validation and Coverage Assessment

### 8.1 Ex Post Coverage Characterization

The recruited portfolio is characterized against the SUSB universe to document coverage gaps — transparently reported and updated as firms join or exit. This is diagnostic, not corrective.

### 8.2 Benchmarking with BDS

SUSB provides the level and cross-sectional distribution; BDS provides firm-level transition dynamics (entry, exit, expansion, contraction by size class). Together they validate birth/death rates, growth rates, and stability characteristics of the portfolio versus the universe.

### 8.3 Signal Quality Assessment

Each series is assessed via posterior estimates of *λ_s* and *σ_s*. Series with near-zero loadings or excessive noise provide little information and may be candidates for exclusion. This is performed as part of routine estimation.

------------------------------------------------------------------------

## 9. Implementation Brief

The preceding sections establish the methodological case. This section translates it into an operational plan.

### 9.1 Target Count

The initial target is **15 firms**, diversified across at least five supersectors. This is a recruitment goal calibrated to the point where additional series meaningfully reduce posterior uncertainty, not a fixed sample size. If early recruits cluster in a few industries, priority shifts to underrepresented sectors. Diminishing marginal returns are expected beyond 25–30 firms.

### 9.2 Industry Priorities

Recruitment should prioritize industries where large firms account for the greatest employment share and where existing payroll provider coverage is weakest. Initial focus areas:

-   **Healthcare and social assistance:** Large hospital systems and health networks; high concentration in 1,000+ employee firms.
-   **Retail trade:** National and regional chains; high employment share but thin provider coverage at scale.
-   **Manufacturing:** Large employers with stable, establishment-identifiable payroll structures.
-   **Finance and insurance:** Concentrated employment in large institutions; often data-sophisticated and potentially receptive.
-   **Professional and business services:** Includes large staffing firms and consultancies; existing provider data skews toward this sector's small firms.

Secondary priorities include transportation/warehousing, accommodation/food services, and information. The specific mix should be updated after the initial SUSB coverage gap analysis.

### 9.3 Geographic Considerations

Geography is a secondary priority at the national-level model stage but becomes important for Release 3+. Multi-establishment firms naturally provide geographic diversity through their footprints — a single national retailer may populate dozens of state-level cells. Recruitment should note each firm's establishment geography, not just headquarters location.

### 9.4 Recruitment Workflow

1.  **SUSB mapping:** Construct the industry × size matrix for firms with 500+ employees. Identify cells with the largest employment shares and weakest provider coverage.
2.  **Relationship inventory:** Catalog existing client relationships intersecting priority cells. Engage internal contacts to assess receptivity.
3.  **Exploratory conversations:** Initiate discussions through internal contacts, framing around the value proposition. Cross-organizational networks are a key resource for identifying and facilitating these conversations.
4.  **Technical scoping:** For receptive firms, identify the payroll platform in use (enterprise, mid-market, or in-house). Engage internal staff with platform expertise to specify the extraction path and minimize firm-side burden.
5.  **Legal and data governance:** Execute templated data sharing agreements.
6.  **Data onboarding:** Ingest historical backfill, validate against reference period and definitional requirements, construct the continuing-units growth series.
7.  **Ongoing delivery:** Establish automated or semi-automated monthly data pipelines.

### 9.5 Reducing Friction: The Payroll Platform Angle

Most large firms run payroll through Workday, SAP, or mid-market platforms like Paycor, Paylocity, or Paycom. Providing firm contacts with platform-specific extraction instructions — "here is exactly the report to run in Workday" — drops the operational burden substantially. Identifying internal staff with expertise in these platforms is a near-term priority. Existing leads in the payroll platform space (e.g., Paycon/Paycor) are a starting point for understanding mid-market extraction workflows.

### 9.6 Inducements to Participate

The primary value proposition is early access to macroeconomic findings derived in part from the firm's data — essentially a benchmarking service contextualizing their employment dynamics within broader trends. This must be calibrated carefully to avoid overpromising. Potential inducements:

-   Early access to nowcast estimates and labor market trend analysis.
-   Confidential benchmarking of firm employment growth against industry and size class.
-   Aggregated, non-identifying comparisons with peer firms.

### 9.7 Timeline

| Phase | Activity | Target |
|---------------------|-----------------------------|-----------------------|
| Month 1 | SUSB coverage gap analysis; relationship inventory | Priority cell list and contact map |
| Month 2–3 | Exploratory conversations with top-priority contacts | 5–8 firms engaged |
| Month 3–4 | Technical scoping and legal agreements | 3–5 firms committed |
| Month 4–6 | Historical data onboarding and series construction | First large-firm series in model |
| Ongoing | Expand recruitment; monthly data delivery | 10–15 firms by month 9–12 |

### 9.8 Fallback Options

If relationship-based recruitment yields fewer than 10 firms after six months, escalation options include: engaging a survey company for targeted cold outreach to firms in underrepresented cells, exploring data licensing with mid-market payroll platforms serving the 500–4,999 employee segment, and revisiting ADP pre-aggregated data for dynamic factor model integration as a complement to the state space framework.

------------------------------------------------------------------------

## 10. Conclusion

The large-firm coverage gap is the most significant data limitation facing the NFP nowcasting system. Firms with 500+ employees account for over half of private-sector employment but are largely absent from payroll provider microdata.

A structured relationship-based recruitment strategy is preferred over probability sampling: the universe is small and well-enumerated, the data request is high-friction with predictably low cold-outreach response rates, nonresponse bias would dominate any probability design, and the model does not require representative inputs. By mapping relationships against the SUSB distribution and prioritizing toward high-value cells, we convert convenience into a structured program with documented coverage properties.

The acquired series enter the model identically to provider aggregates — through measurement equations with estimated bias, loading, and noise — and are weighted by demonstrated informativeness. This ensures the data improves nowcast quality regardless of representativeness, provided it adds independent signal about the latent state of national employment.

------------------------------------------------------------------------

## References

*Cajner, T., Crane, L. D., Decker, R. A., Grigsby, J., Hamins-Puertolas, A., Hurst, E., Kurz, C., & Yildirmaz, A.* (2022). Improving the accuracy of economic measurement with multiple data sources: The case of payroll employment data. In Abraham, Jarmin, Moyer, & Shapiro (Eds.), *Big Data for Twenty-First-Century Economic Statistics*, NBER.

*U.S. Census Bureau.* Statistics of U.S. Businesses (SUSB). Annual data tables by enterprise size. Available at: https://www.census.gov/programs-surveys/susb.html

*U.S. Census Bureau.* Business Dynamics Statistics (BDS). Firm-level entry, exit, expansion, and contraction. Available at: https://www.census.gov/programs-surveys/bds.html

*U.S. Bureau of Labor Statistics.* Quarterly Census of Employment and Wages (QCEW). Available at: https://www.bls.gov/cew/