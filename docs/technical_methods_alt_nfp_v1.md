# Estimating Employment Using Payroll-Provider Microdata

## Technical Methodology

### Abstract

Private payroll providers offer rich, high-frequency microdata at the client-, employee-, and payment-level. These data can support timely and granular employment measurement, but they are not designed for the purpose of constructing macroeconomic timeseries and differ fundamentally from official employment universes. This paper details the technical methodology for producing official-style employment estimates from payroll microdata.

The approach is organized as a sequential pipeline: (1) define employment using payroll-period reference dates aligned to CES/QCEW concepts; (2) harmonize statistical units from client-level to establishment-level records; (3) exclude onboarding, offboarding, and administrative churn via measurement panels; (4) stratify to QCEW using geography × industry × size-class cells; (5) embed provider series in a Bayesian state-space model with continuing-units and birth/death decomposition, anchored to QCEW; and (6) monitor for business-process discontinuities.

The framework extends Cajner et al. (2022) by explicitly addressing PSP-specific issues related to statistical units, size-class distortions, onboarding dynamics, and business-process discontinuities.

------------------------------------------------------------------------

### Glossary

**Establishment.** A single physical location where business is conducted or where services or industrial operations are performed (BLS definition). The fundamental unit of observation in both CES and QCEW. A single firm may operate one or many establishments.

**Firm (or enterprise).** A business entity that may operate one or more establishments. We default to "firm" except where the multi-establishment nature of the entity is being emphasized, in which case we use "enterprise."

**Payroll-services provider (PSP).** A private company that processes payroll on behalf of client firms. We use the abbreviation PSP throughout.

**Client.** A firm that contracts with a PSP for payroll processing. A client may correspond to a single establishment, a multi-establishment enterprise, or an intermediate payroll account. The ambiguity in this mapping is a central challenge addressed in Section 2.

**Stratification.** The process of aligning the PSP's client base to the characteristics of the QCEW universe by partitioning both into comparable cells defined by geography, industry, and size class. "Benchmarking" denotes anchoring of estimates to QCEW employment levels.

**Continuing units.** Establishments (or PSP clients) present in both the current and prior month's data. Employment change over continuing units captures the intensive margin — hiring and separations at existing businesses — while excluding the contribution of business births and deaths.

**Birth/death (B/D).** The net employment contribution of new establishments opening (births) minus existing establishments closing (deaths). PSP data measure only continuing-units growth; the B/D contribution must be modeled separately.

**Supersector.** An aggregation of related NAICS industries used by BLS as a primary classification level in CES estimation. There are 13 private supersectors.

**QCEW (Quarterly Census of Employment and Wages).** A near-census derived from state UI tax records. Published quarterly with a 5–6 month lag. Covers approximately 95% of nonfarm payrolls and serves as the benchmark universe.

**CES (Current Employment Statistics).** The BLS monthly survey of approximately 631,000 worksites producing the headline nonfarm payroll number. Published approximately 3 weeks after the reference month.

------------------------------------------------------------------------

## 1. Data Inputs and Operational Employment Measurement

PSP microdata typically include client-, employee-, and payment-level records. Preserving this layered structure is critical for unit diagnostics, geographic assignment, and validation.

Employment is defined operationally using the payroll period containing a reference date aligned with the CES/QCEW reference week (the pay period including the 12th of each month). Two measures are constructed:

-   **Active employment**: employees on payroll during the period
-   **Qualified employment**: active employees receiving qualified pay (regular pay, overtime, paid sick leave, holiday pay, vacation pay, reported tips, and bonuses/commissions/severance that are regular in amount and schedule)

Let $i$ index employees and $p(t)$ denote the payroll period containing reference date $t$:

$$E_{t}^{\text{active}} = \sum_{i} \mathbf{1}\{ i \in \text{payroll in } p(t) \}$$

$$E_{t}^{\text{qual}} = \sum_{i} \mathbf{1}\{ i \in \text{payroll in } p(t),\ \text{qualified pay}_{i}(p(t)) > 0 \}$$

These definitions mirror CES concepts but remain sensitive to payroll timing, off-cycle payments, and provider-specific rules. The BLS faces similar timing issues when PSPs report CES data on employers' behalf, since pay period boundaries do not always align neatly with the reference week. When PSPs do report to CES, they are required to report for the specific pay period including the 12th of the month, mitigating some of this ambiguity.

Maintaining both measures enables robustness checks and diagnostics.

------------------------------------------------------------------------

## 2. Statistical Unit Harmonization

Official benchmarks are establishment-based, while PSP clients may represent single establishments, multi-establishment enterprises, or sub-establishment payroll accounts. To stratify PSP data to the QCEW universe and produce geographically meaningful estimates, the PSP unit of observation must approximate an establishment as closely as possible. If a multi-establishment enterprise enters the data as a single client, its employment will be assigned to one geography and one industry code even though its actual operations may span many states and industries, distorting both stratification and geographic decomposition.

The severity of this distortion depends critically on available location information. This section describes the full-information approach and then addresses progressively degraded information regimes.

### 2.1 Establishment-Likeness Diagnostics

Each client is assessed for establishment-likeness using employee geographic dispersion relative to the client address:

-   **Distance distributions**: percentile distances between employee locations and the client address.
-   **Multimodality detection**: mixture models fitted to employee location distributions to identify clusters suggestive of distinct worksites. Multimodality in the distribution suggests the client operates at multiple distinct physical sites; a unimodal distribution centered near the client address is consistent with a single establishment.
-   **Commuting priors**: industry- and metro-specific commuting distance thresholds (shorter for retail, longer for professional services) that distinguish "employees commuting to one worksite" from "employees spread across multiple worksites." Thresholds are calibrated using BLS and Census commuting data, providing a prior probability that an employee at a given distance is commuting to the client address vs. working at a remote site.

These diagnostics produce a latent classification: establishment-like, enterprise-like, or ambiguous.

### 2.2 Pseudo-Establishment Construction

For enterprise-like clients, pseudo-establishments are constructed by clustering employees into geographically coherent groups. For cluster $k$ within client $c$, pseudo-establishment employment is:

$$E_{c,k,t} = \sum_{i \in \mathcal{C}_{c,k}} \mathbf{1}\{ i \text{ employed in } p(t) \}$$

Each pseudo-establishment is assigned a geography via centroid or medoid and inherits (or optionally reassigns) industry. This converts enterprise clients into establishment-like units and improves coherence with QCEW.

### 2.3 Degraded Location Information

The diagnostics and pseudo-establishment construction above assume precise geocodes. In practice, PSPs may offer only coarse location information. Each step down the information ladder progressively undermines the ability to stratify the PSP's client base to QCEW characteristics: coverage ratios become less meaningful, shift-share decompositions absorb geographic misallocation alongside genuine compositional differences, and usability diagnostics are themselves contaminated by the same information gap.

Recognizing which information regime applies for a given provider is a prerequisite for selecting appropriate methods.

#### 2.3.1 Zip Codes Only

When employee-level zip codes (reflecting work location, not residential address) are available but precise geocodes are not, pseudo-establishment construction remains feasible with modest degradation. Zip code centroids — preferably population-weighted centroids published by the Census Bureau — provide sufficient resolution for geographic assignments at the supersector × Census region level. State-level assignment will be correct in the vast majority of cases (zip codes straddling state lines exist but are negligible in employment terms).

Identifying enterprise-like clients also remains effective for the most consequential cases: national retail chains, multi-location healthcare systems, and other large multi-establishment firms whose employees span many zip codes. The dispersion signal — employees in 50 zip codes versus one — is preserved.

The principal loss is in diagnostics for mid-size firms operating 2–3 worksites within the same metro area, where employees at distinct locations may share a zip code. The information loss for these cases is modest in aggregate: mid-size firms operating 2–3 nearby worksites in the same industry contribute similar growth signals regardless of whether they are classified as one establishment or split into two. The classification error matters mainly for geographic precision, which is secondary for the national estimates that are our initial focus.

For these cases, commuting-distance priors provide a partial substitute: if a zip centroid is within a reasonable commuting radius of the client address for the relevant industry, the client is treated as establishment-like; if not, the dispersion signals a remote worksite.

#### 2.3.2 No Employee or Client Location

When no location information is available, pseudo-establishment construction is impossible and geographic allocation must rely entirely on external benchmarks. The available information reduces to the client's NAICS code and employment count. The BLS would not face this particular problem even if a PSP reported on an employer's behalf, because CES requires establishment-level reporting regardless of payroll system structure.

For initial national-level estimates, geographic allocation of individual clients matters only insofar as it affects the stratification weights. Allocation errors partially cancel across the full client base. Geographic allocation becomes critical only when extending to state- or metro-level estimates.

The allocation uses QCEW employment shares by state × industry as a key. Let $s_{g,n}^{\text{QCEW}}$ denote the share of national employment in state $g$ and industry $n$ from the most recent QCEW vintage. For a client $c$ with industry code $n_c$ and employment $E_c$:

$$\widehat{E}_{c,g,t} = E_{c,t} \cdot s_{g,n_c}^{\text{QCEW}}$$

This treats every client as if its geographic footprint mirrors the national distribution for its industry. Allocation should be performed at the finest available NAICS level (6-digit) and aggregated upward, since finer industries carry more geographic specificity that is preserved through aggregation.

#### 2.3.3 Client State Only

When only the client's headquarters state is known, the data contain a geographic signal that is informative for small firms but systematically misleading for large ones (headquarters bias). The vast majority of small firms are single-establishment and operate only in their headquarters state. For large clients, the headquarters state may be uninformative about where employment actually sits, and total employment may exceed the QCEW total for that state × industry cell — not because of data error, but because the firm's establishments span many states.

A hybrid allocation treats small and large clients differently:

-   **Below a size threshold**: Accept the headquarters state assignment. These clients are overwhelmingly single-establishment.
-   **Above the size threshold**: Redistribute employment across states using QCEW state × industry shares as a prior. No special weight is given to the headquarters state, because for large multi-establishment firms the headquarters location is uninformative about where workers are actually located.

The size threshold should be calibrated using QCEW data on multi-establishment versus single-establishment firms by industry and size class. QCEW's Multiple Worksite Report (MWR) data show that only 4.7% of employers operate multiple establishments, but these account for 45% of national employment. The threshold varies by industry.

When the provider's data include some clients with full location information alongside others with only headquarters state, the observed relationship between firm size and geographic dispersion by industry for the located clients can be used to calibrate the redistribution for unlocated clients — a missing-data model where observed clients inform the allocation of unobserved ones.

#### 2.3.4 Coverage Ratios Exceeding Unity

Under degraded location information — particularly in the state-only regime — cell-level coverage ratios (PSP employment divided by QCEW employment) may exceed 1. This does not necessarily indicate data error but does indicate misclassification in the sense that employment has been assigned to the wrong geographic cell. Ratios exceeding 1 are the expected outcome for large multi-establishment firms under headquarters-state assignment and should be treated as a signal to redistribute, not to discard.

Several factors may contribute:

**Firm–establishment mismatch (primary cause).** The client's total firm employment exceeds the establishment-level QCEW total for the assigned cell because the firm's worksites span geographies not captured in the assignment. Remedied by the hybrid allocation in Section 2.3.3.

**NAICS misclassification.** At fine NAICS granularity (6-digit), some cells are genuinely small, and even a moderately-sized client with an incorrect industry code can exceed the cell total. Persistent ratios exceeding 1 at fine granularity after redistribution may serve as diagnostic flags for NAICS code auditing.

**Stale QCEW shares.** If the most recent QCEW vintage is from a prior year and the industry has grown rapidly in a particular state, the denominator may be too small.

**Remedies:**

**Aggregate to coarser cells.** At 6-digit NAICS × state, many cells are small enough that ratios above 1 are common. At supersector × Census region, cells are large enough that ratios above 1 become rare.

**Iterative proportional fitting (raking).** Treat the allocation as a two-dimensional margins problem with known provider employment totals by industry and known QCEW state × industry population totals. Raking iteratively rescales rows and columns until convergence and can incorporate upper-bound constraints preventing any cell from exceeding its QCEW total (or a specified multiple). This produces an internally consistent allocation respecting both the industry composition of the provider's client base and the geographic structure of official employment.

**Two-stage raking.** First, for clients below the size threshold, accept the state assignment and let them contribute directly to coverage ratios. Second, for clients above the threshold, redistribute using QCEW shares as a prior and then rake the combined result to QCEW margins. This preserves high-quality information from small single-establishment clients while preventing large multi-establishment firms from distorting cell-level coverage.

**Bayesian shrinkage.** Model the allocation probabilistically by placing a Dirichlet prior on the state distribution for each industry (informed by QCEW shares) and updating with observed data (headquarters state, any partial location information). Cells where the posterior allocation exceeds the QCEW total are automatically penalized because the likelihood of observing that much employment in a small cell is low. This integrates naturally with the Bayesian framework in Section 5.

#### 2.3.5 Handling PSPs with Zero Coverage Above a Size Threshold

When a PSP has zero coverage for firms above a certain size threshold (e.g., an SME-focused provider whose largest client has 200 employees), the provider's data contain no information about large-firm employment dynamics, which account for a disproportionate share of national employment (firms with 500+ employees account for roughly half of total private employment).

The framework handles this by construction: the Bayesian model estimates each provider's signal loading and bias conditional on the cells and size classes where that provider has coverage. A provider with no large-firm coverage simply contributes no information to the large-firm component; those cells are informed by other providers, CES vintages, and the QCEW anchor.

#### 2.3.6 Downstream Consequences for Representativeness Assessment

The degradation cascade affects not only geographic allocation but also the diagnostics used to assess representativeness. Coverage ratios computed under degraded location information reflect allocation assumptions rather than true coverage. The shift-share decomposition of growth divergence into composition and within-cell effects becomes unreliable because composition weights absorb geographic misallocation on top of genuine compositional differences. The cell-level reliability classification (reliable, marginal, insufficient) should be computed after any redistribution step, not before, to avoid marking headquarters states as reliable and branch-operation states as insufficient when the true coverage pattern may be the reverse.

The temporal stability of redistributed coverage ratios provides a useful cross-check. If QCEW-share-based allocation produces ratios stable over time, the allocation is likely reasonable. If ratios jump at business cycle frequencies, it may indicate that firm-level geographic footprints are shifting in ways that static QCEW shares do not capture, flagging cells for aggregation or closer investigation.

As a practical matter, when location information is limited to the state level, restricting geographic analysis to the subset of small clients where client ≈ establishment — which for SME-focused providers constitutes the core of the client base — may be preferable to applying allocation assumptions to the full sample. Large multi-establishment clients can still contribute to national industry-level signals without being forced into a geographic decomposition that the data cannot support.

------------------------------------------------------------------------

## 3. Sample Dynamics: Onboarding, Churn, and Administrative Artifacts

PSP data differ from CES in a critical respect: the set of continuing units is not conditionally static. Clients enter and exit the provider for administrative reasons (vendor switching, mergers, contract terminations) unrelated to true economic dynamics. Additionally, clients often do not onboard employees all at once — if migration to the new payroll system is a multi-period process, post-first-period onboarding will appear as economic growth rather than administrative change.

"Contract terminations" refers to the administrative event of a client ending its contract with a PSP, typically to switch to a competing provider or bring payroll in-house. Mergers at the establishment level are relatively rare and more of an issue for BLS (where two sampled establishments combining creates a matching problem) than for our framework (where the merged entity simply continues reporting, albeit potentially under a new client ID that must be linked).

### 3.1 Measurement Panels

To isolate within-firm employment change while preserving representativeness, we construct measurement panels refreshed periodically but held fixed within each interval. This structure is analogous to CES sample rotation, where new sample units are phased in periodically while the existing sample is held fixed between rotations.

At quarterly intervals, a measurement panel is defined consisting of all eligible clients as of the panel start date. This panel is then frozen for the quarter:

-   No new clients are added mid-quarter, even if they subsequently satisfy eligibility criteria.
-   Clients that stop reporting are removed without contributing to negative employment change. All mid-panel exits are treated as administrative departures rather than economic closures. This is a conservative choice: some exits may genuinely reflect business closures, but absent a reliable way to distinguish the two in real time, all exits are excluded from continuing-units growth measurement. The birth/death component (Section 5.3) is the mechanism through which business closures enter the estimate. Distinguishing true closures from PSP exits in real time is not possible with available data; however, for continuing-units growth measurement, the distinction is immaterial.

Within each panel, employment growth is computed using a CES-style continuing-unit estimator based solely on matched observations across consecutive periods.

The quarterly refresh introduces newly eligible clients and drops departures, ensuring the panel evolves with the provider's client base without allowing mid-quarter churn to contaminate growth measurement.

### 3.2 Client Stabilization

Slow onboarding generates spurious growth as employees migrate to the new payroll system over multiple periods. A client must have non-zero employment for $m$ consecutive months before inclusion.

Formally, for client $j$ at time $t$:

$$\mathbf{1}\{\text{stable}_{j,t}\} = \mathbf{1}\left\{ \sum_{s=0}^{5} \mathbf{1}\{ E_{j,t-s} > 0 \} = m \right\}$$

The $m$-month requirement is based on empirical analysis of onboarding patterns, which shows that most clients stabilize within 3–4 months; the six-month threshold provides a conservative buffer.

As a supplementary diagnostic, statistical change-point methods detect transitions from onboarding to steady-state regimes by estimating the posterior probability of a regime change at each point in a client's history. These can identify cases where the fixed tenure rule is either too conservative (a client that stabilized quickly) or not conservative enough (unusually slow ramp-up). In practice, the six-month rule serves as the primary gate, with change-point analysis informing periodic review and recalibration.

------------------------------------------------------------------------

## 4. Stratification and Size-Class Structure

Most PSP client bases are skewed toward small and medium-sized clients. Because size class affects growth dynamics, churn behavior, and B/D intensity, stratification by size alongside geography and industry is essential. Without this, the PSP-derived growth signal would disproportionately reflect SME dynamics.

Stratification is implemented over cells defined by:

$$\text{Cell} = (\text{state},\ \text{industry},\ \text{size class})$$

Let $w_j$ denote stratification weights for cell $j$. Conceptually:

$$\sum_{j \in \text{cell}} w_j\, E_{j,t}^{\text{PSP}} \approx E_{\text{cell}, t-l}^{\text{QCEW}}$$

where $l$ denotes the QCEW publication lag. Size-class stratification is essential to avoid systematic bias arising from overrepresentation of small firms.

When location information is degraded (Section 2.3), the cell definition may need adaptation. Under the state-only regime, cells for small clients (where client ≈ establishment) can retain the full state × industry × size class structure. For cells incorporating redistributed large-client employment, geographic stratification should be performed at coarser levels (e.g., Census region rather than state), because the geographic allocation of large clients is itself uncertain, and stratifying at a fine geographic level would propagate that allocation uncertainty into the weights.

------------------------------------------------------------------------

## 5. Latent-State Modeling and Benchmark Anchoring

The continuing-units series entering the state-space model is constructed using the measurement-panel methodology in Section 3. PSP observations measure only the intensive margin — hiring and separations at firms present in both the current and prior month. Client entry and exit dynamics are excluded from the measurement equation. The B/D contribution is modeled separately (Section 5.3) and added to produce a total employment change nowcast.

This separation improves identification of true continuing-unit growth and prevents administrative churn from being misinterpreted as economic signal. Entry and exit rates observed in PSP data may serve as covariates for time-varying measurement error or provider-specific drift, entering the model indirectly rather than through the primary measurement equation.

PSP continuing-unit microdata are embedded in a Bayesian state-space framework, summarized here and detailed in the companion estimation methods paper.

### 5.1 Relation to Cajner et al. (2022)

Cajner et al. model CES survey data and PSP-derived data as noisy observations of latent true employment:

$$y_t^{\text{source}} = E_t + \epsilon_t^{\text{source}}$$

Our framework extends this by:

-   **Decomposing employment change into continuing-units and B/D components.** PSP data directly measure continuing-units growth; total employment change additionally includes net business openings and closings, modeled separately because PSPs cannot observe it.
-   **Introducing hierarchical partial pooling over geography × industry × size.** Model parameters (bias, signal loading, noise variance) for sparse cells are regularized toward group-level means, improving estimates for cells with limited data while preserving genuine heterogeneity where the data support it.
-   **Accounting for onboarding artifacts.** These enter not in the measurement equation but through stabilization procedures (Section 3.2), which filter data before they reach the model. The measurement equation operates on stabilized, continuing-units data only.
-   **Treating QCEW as a near-census, lagged anchor.** The QCEW is modeled as an observation of true total employment change (continuing units plus B/D) with small measurement error, arriving with a lag. This allows the model to learn bias, loading, and noise parameters: the QCEW provides the "answer key" against which PSP signals are evaluated, with the lag meaning real-time nowcasts project the learned relationships forward.

### 5.2 Continuing Units and Birth/Death Decomposition

For each cell $j$:

$$\Delta E_{j,t}^{\text{true}} = \Delta E_{j,t}^{\text{cont}} + BD_{j,t}$$

PSP data provide information about the continuing-units component:

$$y_{j,t}^{\text{PSP}} = \lambda_j\, \Delta E_{j,t}^{\text{cont}} + \epsilon_{j,t}^{\text{PSP}}$$

### 5.3 Birth/Death Model

Birth/death contributions are modeled as:

$$BD_{j,t} = \alpha_j + \beta \cdot \text{cycle}_t + \gamma \cdot BD_{j,t-l}^{\text{QCEW}} + u_{j,t}$$

with hierarchical structure over:

-   Industry (nested NAICS)
-   Size class (small firms have higher turnover)
-   Geography (secondary)

### 5.4 QCEW as Benchmark Observation

QCEW is treated as a lagged observation of total employment change:

$$y_{j,t-l}^{\text{QCEW}} = \Delta E_{j,t-l}^{\text{true}} + \epsilon_{j,t-l}^{\text{QCEW}}$$

with small measurement error. These observations anchor bias, signal loadings, and B/D parameters and propagate forward to inform real-time nowcasts.

### 5.5 Hierarchical Partial Pooling

Cell-level parameters (bias, signal loading, B/D intensity) follow structured hierarchies:

-   Geography: region → division → state
-   Industry: domain → supersector → sector → subsector
-   Size class: explicit additive effects

This stabilizes sparse cells while preserving economically meaningful heterogeneity. Cells with abundant data have parameter estimates close to their cell-specific values, while cells with sparse data are pulled toward the group mean.

### 5.6 PSP Data Treated Separately, Not Pooled

Each PSP enters the model as a separate measurement equation with its own estimated bias, signal loading, and noise parameters. The Bayesian framework estimates these relationships jointly, weighting each provider's contribution by its informativeness (estimated signal-to-noise ratio) rather than requiring a prior decision about how to combine them. The model can learn which providers are most informative under which conditions without imposing assumptions about comparability.

### 5.7 Iterative Deployment

Given the complexity of the proposed methodology, we will repeatedly iterate from simpler models to the full model above. Each incremental complication will be evaluated against the simpler specification to demonstrate value before incorporation into the production version.

------------------------------------------------------------------------

## 6. Seasonality and Pay-Frequency Artifacts

Payroll series exhibit strong seasonal patterns driven by industry cycles, fiscal calendars, and payroll system behavior. Pay-frequency artifacts can induce mechanical fluctuations.

Seasonality is modeled explicitly in the latent state using additive seasonal components:

$$\Delta E_{j,t}^{\text{true}} = \mu_{j,t} + S_{j,t} + \eta_{j,t}$$

where $S_{j,t}$ follows a hierarchical seasonal structure by industry and geography, allowing seasonal amplitudes and patterns to evolve over time and differ systematically across sectors and regions.

------------------------------------------------------------------------

## 7. Quality, Governance, and Concept Drift

Because PSP data are generated by business processes, discontinuities are an inherent risk. Changes in provider operations, coding practices, or client mix can alter the effective measurement concept.

We adopt a statistical-agency-style quality framework emphasizing:

-   Accuracy vs. timeliness tradeoffs
-   Revision analysis (real-time vs. retrospective)
-   Coherence across geography and industry
-   Interpretability and transparency

Operational guardrails include monitoring of hours, pay types, and payment timing, drift-aware models, and rolling recalibration to QCEW.

------------------------------------------------------------------------

## 8. Benchmarking Frequency

We adopt annual benchmarking to the QCEW, consistent with CES practice. The QCEW enters the model as a lagged observation with small measurement error, anchoring the latent state each year. While QCEW is available quarterly, we do not benchmark quarterly for two reasons: first, to maintain maximum comparability with CES (which benchmarks annually to the March QCEW); and second, because using quarterly QCEW observations directly in the measurement equation would introduce QCEW-specific seasonal patterns into the estimates, which may differ from CES seasonality in ways that would confuse interpretation. The model does, however, use QCEW data from all quarters to inform parameter estimation in the backtesting framework.

------------------------------------------------------------------------

## References

Cajner, T., et al. (2022). *Improving the accuracy of economic measurement with multiple data sources: The case of payroll employment data.* In Abraham, Jarmin, Moyer, & Shapiro (Eds.), *Big Data for Twenty-First-Century Economic Statistics,* NBER.