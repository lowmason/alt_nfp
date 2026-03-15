# Estimating Employment Using Payroll-Provider Microdata

## Abstract

Private payroll providers offer rich, high-frequency microdata at the client-, employee-, and payment-level. These data can support timely and granular employment measurement, but they are not designed for the purpose of constructing macroeconomic timeseries and differ fundamentally from official employment universes. This paper proposes an integrated framework for producing official-style employment estimates from payroll microdata.

We organize the approach as a sequential pipeline:

1. Define "employment" as it can be measured in the private data sources, in as close conformity as possible to BLS concepts
2. Transform the private data, if necessary, so the unit of observation is as close as possible to the BLS concept of an "establishment"
3. Correct as much as possible distortions in the private data arising from onboarding, offboarding, and administrative churn
4. Address lack of representativeness in each provider's client base by stratifying to QCEW using available characteristics
5. Embed payroll-services-provider (PSP) series in a Bayesian state-space model anchored to the QCEW, maintaining consistency with BLS practice by focusing on the change in employment on a continuing-units basis and mimicking their approach to taking account of birth/death dynamics
6. Monitor for discontinuities arising from provider business-process changes that could alter the effective measurement concept

The framework builds on the model-based data fusion approach of Cajner et al. (2022) and extends it by explicitly addressing PSP-specific issues related to statistical units, size-class distortions, onboarding dynamics, and business-process discontinuities.


## Audience and Scope

This paper is intended for two audiences. The first several sections provide a high-level overview of the measurement challenges posed by PSP data and the logic of our approach, written for a general audience of economists and Bloomberg readers. The later sections, and certain technical subsections flagged throughout, provide detail for specialists and technicians who wish to understand or replicate the methodology.

### Scope of Estimates

Our initial objective is to produce national-level, monthly, not-seasonally-adjusted estimates of private nonfarm payroll employment change. These estimates will target the same concept as the BLS Current Employment Statistics (CES) survey: the month-over-month change in private nonfarm employment on a continuing-establishment basis, supplemented by estimates of the contribution of business births and deaths.

Specifically, we aim to produce estimates that are consistent with the CES measurement concept rather than an independent estimate of "ground truth." Bloomberg readers — and the markets more broadly — interpret employment data relative to the CES framework. An estimate that departs from CES concepts would require users to mentally translate between two different measurement frameworks, reducing its practical value. Where our methodology identifies a divergence between our estimate and the CES concept (for example, because our QCEW anchor suggests the CES birth/death model is biased), we will report that divergence transparently.

Future releases may extend to geographic and industry detail. Seasonal adjustment will be addressed separately once the not-seasonally-adjusted framework is validated.

### Strategy: Incremental Enhancement from a Proven Baseline

We adopt a disciplined, incremental approach, starting as close as possible to the Cajner et al. (2022) framework — a proven model for combining PSP microdata with CES and QCEW observations in a Bayesian state-space setting — and enhancing it step by step. Each enhancement (e.g., explicit birth/death modeling, size-class stratification, hierarchical partial pooling) will be added only after demonstrating measurable improvement in nowcast accuracy or benchmark revision prediction through rigorous, leakage-safe backtesting. We will not build toward the full model described in this paper all at once, but rather iterate through progressively more complex specifications, validating each increment before proceeding to the next.

This approach ensures that the complexity of the system is always justified by demonstrated value, and that the simpler specifications serve as baselines against which enhancements are evaluated.

### PSP Data Treated Separately, Not Pooled

Each PSP enters the model as a separate measurement equation with its own estimated bias, signal loading, and noise parameters. This reflects the fact that each provider's client base differs in composition, size distribution, and geographic coverage, meaning the relationship between their data and true employment varies provider by provider. The Bayesian framework estimates these relationships jointly, weighting each provider's contribution by its informativeness (estimated signal-to-noise ratio) rather than requiring a prior decision about how to combine them. This is preferable to pooling because the model can learn from the data which providers are most informative in which conditions, without imposing assumptions about comparability across providers.

### Benchmarking Frequency

We adopt annual benchmarking to the QCEW, consistent with CES practice. The QCEW enters the model as a lagged observation with small measurement error, anchoring the latent state each year. While the QCEW is available quarterly, we do not perform quarterly benchmarking for two reasons: first, to maintain maximum comparability with CES (which benchmarks annually to the March QCEW); and second, because using quarterly QCEW observations directly in the measurement equation would introduce QCEW-specific seasonal patterns into the estimates, which may differ from CES seasonality in ways that would confuse interpretation. The model does, however, use QCEW data from all quarters to inform parameter estimation in the backtesting framework.


## Glossary

The following terms are used throughout this paper with specific meanings. Where multiple synonyms exist in the literature, we adopt one term and use it consistently.

**Establishment.** A single physical location where business is conducted or where services or industrial operations are performed (BLS definition). The fundamental unit of observation in both the CES survey and the QCEW. A single firm may operate one or many establishments.

**Firm (or enterprise).** A business entity that may operate one or more establishments. In this paper, "firm" and "enterprise" are used interchangeably; both refer to the legal or organizational entity rather than the physical worksite. We default to "firm" except where the multi-establishment nature of the entity is being emphasized, in which case we use "enterprise."

**Payroll-services provider (PSP).** A private company that processes payroll on behalf of client firms. The source of the auxiliary microdata used in this framework. Examples include ADP, Paychex, and similar providers. We use the abbreviation PSP throughout.

**Client.** A firm that contracts with a PSP for payroll processing. A client may correspond to a single establishment, a multi-establishment enterprise, or an intermediate payroll account. The ambiguity in this mapping is a central challenge addressed in Section 3.

**Stratification.** The process of aligning the PSP's client base to the characteristics of the QCEW universe by partitioning both into comparable cells defined by geography, industry, and size class. Earlier versions of this paper used the terms "calibration," "benchmarking," and "representativeness correction" somewhat interchangeably; we now use "stratification" to denote the partitioning itself, "benchmarking" to denote the anchoring of estimates to QCEW employment levels, and avoid "calibration" to reduce ambiguity.

**Continuing units.** Establishments (or PSP clients) that are present in both the current and prior month's data. Employment change measured over continuing units captures the intensive margin — hiring and separations at existing businesses — while excluding the contribution of business births and deaths.

**Birth/death (B/D).** The net employment contribution of new establishments opening (births) minus existing establishments closing (deaths) in a given month. PSP data measure only continuing-units growth; the birth/death contribution must be modeled separately.

**Supersector.** An aggregation of related NAICS industries used by the BLS as a primary classification level in CES estimation. Examples include "Education and Health Services," "Professional and Business Services," and "Leisure and Hospitality." There are 13 private supersectors.

**QCEW (Quarterly Census of Employment and Wages).** A near-census of employment and wages derived from state unemployment insurance (UI) tax records. Published quarterly with a lag of approximately 5–6 months. Covers approximately 95% of all jobs on nonfarm payrolls and serves as the benchmark universe for both CES and our framework.

**CES (Current Employment Statistics).** The BLS monthly survey of approximately 631,000 worksites producing the headline nonfarm payroll employment number. Published approximately 3 weeks after the reference month.


## 1. Motivation and Organizing Principle

Traditional employment measurement relies on probability surveys (CES) and administrative near-universes (QCEW). Payroll microdata provide a complementary, high-frequency signal but are generated as a byproduct of business processes rather than statistical design.

As a result, payroll data pose several structural challenges:

- Non-random selection into the PSP's client base
- Ambiguity in the statistical unit (client vs. establishment)
- Onboarding to the PSP (as well as offboarding) that are difficult to distinguish from economic dynamics
- Discontinuities arising from provider business-process changes

These challenges motivate the organizing principle of this work:

> Payroll microdata should be treated as noisy, high-frequency measurements of latent true employment that must be anchored to a stable administrative benchmark (QCEW), rather than interpreted as a standalone census.

By "anchored," we mean that the QCEW — which covers approximately 95% of nonfarm employment and is derived from mandatory tax filings rather than voluntary survey response — serves as the ground truth against which PSP-derived signals are evaluated. The QCEW is available only with a lag of 5–6 months, so the PSP data provide timely information between QCEW releases, but the QCEW constrains the level and trend of the estimates over time.

This perspective follows Cajner et al. (2022), who model PSP and CES series as multiple noisy observations of latent employment. Our framework generalizes this approach to explicitly address PSP-specific unit mismatch, onboarding artifacts, size-class distortions, and business-process discontinuities.


## 2. Comparing Our Measurement Challenges with Those of the BLS

Before describing our methodology in detail, it is useful to compare the measurement problems we confront with those facing the BLS. This comparison clarifies where PSP data offer genuine advantages, where they introduce novel difficulties, and where the challenges overlap.

### 2.1 Problems We Confront That the BLS Does Not

**Churn among PSPs.** When a firm switches from one PSP to another, it disappears from the first provider's data and (eventually) appears in the second's. Neither event reflects an actual change in employment. The BLS does not face this problem because its CES sample frame is drawn from the QCEW universe, which is based on UI tax filings and is independent of any commercial relationship. We must explicitly identify and exclude PSP churn from our employment growth measures (Section 4).

**Client–establishment mismatch.** PSP data are organized by client (the contracting firm), which may or may not correspond to a single establishment. A national retail chain processed by a PSP may appear as a single client even though it operates hundreds of establishments across many states. The BLS collects data at the establishment level by design — even when a PSP files CES reports on behalf of a client, the BLS requires establishment-level detail. We must diagnose and, where possible, correct for this mismatch (Section 3).

**Slow onboarding.** When a new client begins using a PSP, employees may migrate to the new payroll system over multiple pay periods, creating the appearance of employment growth that is purely administrative. The BLS does not face this issue because CES sample additions are one-time events: an establishment is either in the sample or not. We require stabilization procedures to ensure that PSP data reflect steady-state employment before entering our measurement panels (Section 4.2).

### 2.2 Problems the BLS Confronts That We Do Not

**Non-response.** The CES relies on voluntary survey response, and response rates have declined from over 60% to below 43%. Non-response creates both sampling noise and potential bias. PSP data, by contrast, are a byproduct of payroll processing: if a firm is a PSP client, its data are observed with near certainty. There is no non-response problem in the traditional sense, though there are analogous issues (late-processing pay periods, clients with incomplete data migration).

**Birth/death model error.** The CES must estimate the net contribution of establishment births and deaths using an ARIMA model, because new establishments are not yet in the sampling frame. This model has been the dominant source of CES estimation error in recent years. While we must also model births and deaths, we can potentially do so with more information: the QCEW provides a lagged census of actual births and deaths, and cyclical indicators can inform the nowcast.

### 2.3 Problems Both We and the BLS Confront

**Stratifying to the QCEW.** Both CES and our framework must align their respective samples (CES survey respondents; PSP client bases) to the QCEW universe to correct for non-representativeness. CES does this through sampling weights based on QCEW-derived strata; we do this through cell-level stratification as described in Section 5.

**Benchmarking to the QCEW.** Both CES and our framework use the QCEW as the definitive measure of employment levels. CES benchmarks annually; our model anchors to the QCEW through a measurement equation with small observation error, also on an annual cycle.

**Seasonality.** Both CES and PSP-derived series exhibit strong seasonal patterns that must be modeled. An additional complication for our framework is that QCEW and CES exhibit somewhat different seasonal patterns; this is one reason we opt for annual rather than quarterly benchmarking (see "Benchmarking Frequency" above).


## 3. Data Inputs and Operational Employment Measurement

PSP microdata typically include client-, employee-, and payment-level records. Preserving this layered structure is critical for unit diagnostics, geographic assignment, and validation.

Employment is defined operationally using the payroll period containing a reference date (aligned with the CES/QCEW reference week, which includes the 12th of each month). We construct two measures:

- **Active employment**: employees on payroll during the period
- **Qualified employment**: active employees receiving qualified pay[^1]

Let $i$ index employees and $p(t)$ denote the payroll period containing reference date $t$. Then:

$$E_{t}^{\text{active}} = \sum_{i}\mathbf{1}\{ i \in \text{payroll in }p(t)\}$$

$$E_{t}^{\text{qual}} = \sum_{i}\mathbf{1}\{ i \in \text{payroll in }p(t),\; \text{qualified pay}_{i}(p(t)) > 0\}$$

These definitions mirror CES concepts but remain sensitive to payroll timing, off-cycle payments, and provider rules.[^2] Maintaining both measures enables robustness checks and diagnostics.


## 4. Statistical Unit Harmonization

Official benchmarks are establishment-based, while PSP clients may represent single establishments, multi-establishment enterprises, or sub-establishment payroll accounts.

**Why this matters.** The QCEW universe is organized by establishment — each record corresponds to a single physical worksite with a specific geographic location and NAICS industry code. To stratify PSP data to the QCEW universe (Section 5), and to produce geographically meaningful estimates in future releases, the PSP unit of observation must be as close as possible to an establishment. If a multi-establishment enterprise enters the data as a single client, its employment will be assigned to one geography and one industry code even though its actual operations may span many states and industries. This distorts both the stratification and any geographic decomposition.

Treating all clients as establishments can distort geography, industry, and stratification. The severity of this distortion depends critically on the location information available. This section describes the full-information approach to unit harmonization and then addresses progressively degraded information regimes, each of which requires different diagnostic and allocation methods.

### 4.1 Establishment-Likeness Diagnostics

We assess whether a client behaves like a single establishment using employee geographic dispersion relative to the client address. Diagnostics include:

- **Distance distributions**: percentile distances between employee locations and the client address
- **Multimodality detection**: fitting mixture models to employee location distributions to identify clusters suggestive of distinct worksites[^3]
- **Commuting priors**: industry- and metro-specific commuting distance thresholds (shorter for retail, longer for professional services) that distinguish "employees commuting to one worksite" from "employees spread across multiple worksites"[^4]

These diagnostics produce a latent classification of clients as establishment-like, enterprise-like, or ambiguous.

### 4.2 Pseudo-Establishment Construction

For enterprise-like clients, we construct pseudo-establishments by clustering employees into geographically coherent groups. For cluster $k$ within client $c$, pseudo-establishment employment is:

$$E_{c,k,t} = \sum_{i \in \mathcal{C}_{c,k}}\mathbf{1}\{ i \text{ employed in } p(t)\}$$

Each pseudo-establishment is assigned a geography via a centroid or medoid and inherits (or optionally reassigns) industry. This converts enterprise clients into establishment-like units and improves coherence with QCEW.

### 4.3 Degraded Location Information

The diagnostics and pseudo-establishment construction described above assume precise employee-level geocodes. In practice, PSPs may offer only coarse location information. The methods described in Sections 4.1 and 4.2 represent the best case. This section addresses a cascade of increasingly degraded information regimes and the adaptations required at each level.

The degradation is not merely a loss of precision. Each step down the information ladder progressively undermines our ability to stratify the PSP's client base to the characteristics of the QCEW.[^5]

Recognizing which information regime applies for a given provider is therefore a prerequisite for selecting appropriate methods.

#### 4.3.1 Zip Codes Only

When employee-level zip codes (reflecting the employee's work location, not residential address) are available but precise geocodes are not, pseudo-establishment construction remains feasible with modest degradation. Zip code centroids — preferably population-weighted centroids published by the Census Bureau, which sit where people actually are rather than at the geographic center of potentially large rural zip codes — provide sufficient resolution for the geographic assignments that matter most.

For the stratification targeting supersector × Census region cells, zip code centroids will almost never misassign a Census region, and state-level assignment will be correct in the vast majority of cases (zip codes straddling state lines exist but are negligible in employment terms). The process of identifying enterprise-like clients (i.e., clients that need to be disaggregated to entities plausibly closer to establishments) also remains effective for the cases that matter most: national retail chains, multi-location healthcare systems, and other large multi-establishment firms whose employees span many zip codes. The dispersion signal that distinguishes enterprise-like from establishment-like clients — employees in (say) 50 zip codes versus one — is preserved.

The principal loss is in the Section 4.1 diagnostics for mid-size firms operating 2–3 worksites within the same metro area, where employees at distinct locations may share a zip code.[^6] For these cases, commuting-distance priors provide a partial substitute: if a zip centroid is within a reasonable commuting radius of the client address for the relevant industry, the client is treated as establishment-like; if not, the dispersion signals a remote worksite. Industry-specific commuting norms (shorter for retail, longer for professional services) should inform the distance threshold.

#### 4.3.2 No Employee or Client Location

When no location information is available at the employee or client level, pseudo-establishment construction is impossible and geographic allocation must rely entirely on external benchmarks.[^7] The available information reduces to the client's NAICS code and its employment count. The question becomes how to distribute that employment geographically.

Note that for the initial national-level estimates that are our primary focus, geographic allocation of individual clients matters only insofar as it affects the stratification used in Section 5. If we are producing national aggregates only, the allocation errors partially cancel across the full client base. Geographic allocation becomes critical only when we extend to state- or metro-level estimates in future releases.

The natural approach when no location information is available is to use QCEW employment shares by state × industry as an allocation key. Let $s_{g,n}^{\text{QCEW}}$ denote the share of national employment in state $g$ and industry $n$ from the most recent QCEW vintage. For a client $c$ with industry code $n_c$ and employment $E_c$, allocated employment to state $g$ is:

$$\hat{E}_{c,g,t} = E_{c,t} \cdot s_{g,n_c}^{\text{QCEW}}$$

This treats every client as if its geographic footprint mirrors the national distribution for its industry — a strong assumption but the best available without location data. The allocation works well for geographically concentrated industries (mining, technology) where the QCEW shares correctly reflect where employment actually sits; it is less reliable for industries where the provider's specific clients may be concentrated in regions that differ from the national pattern.

Allocation should be performed at the finest available NAICS level (6-digit) and then aggregated upward, rather than allocating at the supersector level directly, because finer industries carry more geographic specificity that is preserved through aggregation.

#### 4.3.3 Client State Only

When only the state in which the client is headquartered is known, the data contain a geographic signal that is informative for small firms but systematically misleading for large ones. This is the headquarters bias problem: a retail chain headquartered in Arkansas has all its employment assigned to Arkansas regardless of where its stores operate.

For small clients (below a size threshold), the state assignment is likely correct. The vast majority of small firms are single-establishment and operate only in their headquarters state. For PSPs focused on small and medium-sized enterprises (SMEs), this means most of the client base can be directly assigned without significant distortion.

For large clients, the headquarters state may be uninformative about where employment actually sits, and the firm's total employment will often exceed the QCEW total for that state × industry cell — not because of data error, but because the firm is a multi-establishment enterprise whose employment — in real life — spans many states. A coverage ratio exceeding 1 in this context is the expected mechanical consequence of attributing firm-level employment to the headquarters state when the firm's establishments may be distributed across many states.

A hybrid allocation addresses this by treating small and large clients differently:

- **Below the size threshold**: Accept the headquarters state assignment. These clients are overwhelmingly single-establishment.
- **Above the size threshold**: Redistribute employment across states using QCEW state × industry shares as a prior. This is equivalent to allocating according to state × industry shares, with the headquarters state receiving its QCEW-proportional share like any other state. No special weight is given to the headquarters state, because for large multi-establishment firms the headquarters location is uninformative about where workers are actually located.

The size threshold itself should be calibrated using QCEW data on multi-establishment versus single-establishment firms by industry and size class. QCEW's Multiple Worksite Report (MWR) data show that only 4.7% of employers operate multiple establishments, but these account for 45% of national employment. The threshold varies by industry: in retail or food service, even modestly-sized firms are often multi-establishment; in professional services, larger firms may still operate from a single location.

When the provider's data include some clients with full location information (geocodes or zip codes) alongside others with only headquarters state, the observed relationship between firm size and geographic dispersion by industry for the located clients can be used to calibrate the redistribution for unlocated clients. This amounts to a missing-data model where observed clients inform the allocation of unobserved ones.

#### 4.3.4 Coverage Ratios Exceeding Unity

Under degraded location information — particularly in the state-only regime — cell-level coverage ratios (PSP employment divided by QCEW employment) may exceed 1. This does not necessarily indicate data error, though it does indicate misclassification in the sense that employment has been assigned to the wrong geographic cell. The distinction matters: a coverage ratio above 1 is a signal to redistribute, not to discard. When the PSP client is a firm rather than an establishment, its total employment may exceed the QCEW total for the headquarters state × industry cell simply because the firm operates establishments across many states while all its employment has been attributed to one.

Ratios exceeding 1 are the expected outcome for large multi-establishment firms under headquarters-state assignment and should be treated as the primary signal that redistribution is needed. Nevertheless, several factors may contribute, and the appropriate remedy depends on the cause:

**Firm–establishment mismatch (primary cause).** The client's total firm employment exceeds the establishment-level QCEW total for the assigned cell because the firm's worksites span geographies not captured in the assignment. This is remedied by the hybrid allocation described in Section 4.3.3.

**NAICS misclassification.** At fine NAICS granularity (6-digit), some cells are genuinely small, and even a moderately-sized client with an incorrect industry code can exceed the cell total. Moreover, this anomaly might persist even after geographic redistribution. Persistent ratios exceeding 1 at fine granularity after redistribution may serve as diagnostic flags for NAICS code auditing. (Note that NAICS codes are typically assigned by the PSP based on client self-reporting, and may not always match the codes assigned by BLS based on the establishment's primary activity.)

**Stale QCEW shares.** If the most recent QCEW vintage is from a prior year and the industry has grown rapidly in a particular state, the denominator may be too small. This is especially relevant for fast-growing industries or states and is addressed by using the most current QCEW vintage available and recognizing that allocation precision degrades as the QCEW anchor ages.

*The following remedies are more technical in nature and are included for specialist readers.*

**Aggregate to coarser cells.** At 6-digit NAICS × state, many cells are small enough that ratios above 1 are common. At supersector × Census region — the aggregation level targeted in the stratification procedure — cells are large enough that ratios above 1 become rare, resolving most cases mechanically.

**Iterative proportional fitting (raking).** Treat the allocation as a two-dimensional margins problem with known provider employment totals by industry (from NAICS codes) and known QCEW state × industry population totals. Raking iteratively rescales rows and columns until convergence and can incorporate upper-bound constraints that prevent any cell from exceeding its QCEW total (or a specified multiple reflecting the provider's national coverage rate). This produces an internally consistent allocation that respects both the industry composition of the provider's client base and the geographic structure of official employment.

**Two-stage raking.** The hybrid allocation motivates a two-stage approach: first, for clients below the size threshold, accept the state assignment and let them contribute directly to coverage ratios; second, for clients above the threshold, redistribute using QCEW shares as a prior and then rake the combined result to QCEW margins. This preserves the high-quality information from small single-establishment clients while preventing large multi-establishment firms from distorting cell-level coverage.

**Bayesian shrinkage.** Rather than hard allocation followed by capping, model the allocation probabilistically by placing a Dirichlet prior on the state distribution for each industry (informed by QCEW shares) and updating with observed data (headquarters state, any partial location information). Cells where the posterior allocation exceeds the QCEW total are automatically penalized because the likelihood of observing that much employment in a small cell is low. This approach integrates naturally with the Bayesian framework described in Section 7.

#### 4.3.5 Handling PSPs with Zero Coverage Above a Size Threshold

A practical challenge arises when a PSP has zero coverage for firms above a certain size threshold — for example, an SME-focused provider whose largest client has 200 employees. In this case, the provider's data contain no information about large-firm employment dynamics, which account for a disproportionate share of national employment (firms with 500+ employees account for roughly half of total private employment).

Our framework handles this by construction: the Bayesian model estimates each provider's signal loading and bias conditional on the cells and size classes where that provider has coverage. A provider with no large-firm coverage simply contributes no information to the large-firm component of the estimate; those cells are informed by other providers, CES vintage data, and the QCEW anchor.

#### 4.3.6 Downstream Consequences for Representativeness Assessment

*This subsection is technical and may be skipped by general readers.*

The degradation cascade affects not only geographic allocation but also the diagnostics used to assess representativeness. Coverage ratios computed under degraded location information reflect the allocation assumptions rather than true coverage. The shift-share decomposition of growth divergence into composition and within-cell effects (described in the companion evaluation methodology) becomes unreliable because the composition weights absorb geographic misallocation on top of genuine compositional differences. Similarly, the cell-level reliability classification (reliable, marginal, insufficient) should be computed after any redistribution step, not before, to avoid marking headquarters states as reliable and branch-operation states as insufficient when the true coverage pattern may be the reverse.

The temporal stability of redistributed coverage ratios provides a useful cross-check. If QCEW-share-based allocation produces coverage ratios that are stable over time, the allocation is likely reasonable. If ratios jump at business cycle frequencies, it may indicate that firm-level geographic footprints are shifting in ways that static QCEW shares do not capture, flagging cells for aggregation to coarser levels or for closer investigation.

As a practical matter, when location information is limited to the state level, restricting the geographic analysis to the subset of small clients where client ≈ establishment — which for SME-focused providers constitutes the core of the client base — may be preferable to applying allocation assumptions to the full sample. Large multi-establishment clients can still contribute to national industry-level signals without being forced into a geographic decomposition that the data cannot support.


## 5. Sample Dynamics: Onboarding, Churn, and Administrative Artifacts

PSP data differ from CES in a critical respect: the set of continuing units is not conditionally static. Clients inevitably enter and exit the provider for administrative reasons (vendor switching, mergers, contract terminations) that are unrelated to true economic hiring or separations.[^8]

Additionally, clients often do not onboard (or offboard) their employees with a new PSP all at once. If paying employees through the new PSP is a multi-period process, any onboarding after the first period will appear as economic growth rather than administrative change.

If untreated, client churn from one provider to another generates positive and negative employment changes that are purely administrative artifacts. In contrast to CES — where the sample is fixed by design — client entry and exit, and slow onboarding and offboarding, in PSP data must be explicitly excluded from employment growth measurement.

Accordingly, client churn is treated as measurement noise, not as employment dynamics.

### 5.1 Measurement Panels

To isolate within-firm employment change while preserving representativeness, we construct measurement panels that are refreshed periodically but held fixed within each interval.[^9] Clients become eligible for inclusion only after satisfying a tenure requirement that suppresses onboarding ramps (Section 5.2).

At quarterly intervals, a measurement panel is defined consisting of all eligible clients as of the panel start date. This panel is then frozen for the duration of the quarter:

- No new clients are added mid-quarter, even if they subsequently satisfy eligibility criteria.
- Clients that stop reporting to the PSP are removed from the panel without contributing to negative employment change. We treat all mid-panel exits as administrative departures from the PSP rather than economic closures. This is a conservative choice: some exits may genuinely reflect business closures, but absent a reliable way to distinguish the two in real time, we exclude all exits from continuing-units growth measurement. The birth/death component of the model (Section 7.3) is the mechanism through which business closures enter the estimate.[^10]

Within each panel, employment growth is computed using a CES-style continuing-unit estimator based solely on matched observations across consecutive periods. This ensures that measured growth reflects only within-firm employment change for surviving employers.

This design prevents artificial growth or contraction from client onboarding or offboarding, and prevents survivorship bias arising from permanently fixed cohorts by refreshing the panel quarterly.

The quarterly panel structure closely mirrors CES sample refresh procedures while respecting the realities of PSP data generation. Each quarterly refresh introduces newly eligible clients and drops clients that have left the provider, ensuring that the panel evolves with the provider's client base without allowing mid-quarter churn to contaminate growth measurement.

### 5.2 Client Stabilization

Slow onboarding can generate spurious growth as employees migrate to the new payroll system over multiple periods. Only after this initial ramp-up should a client be eligible for inclusion in a measurement panel.

We require a client to have non-zero employment registered through the PSP for six consecutive months before including data from that client in our estimates. Formally, for client $j$ at time $t$:

$$\mathbf{1}\{\text{stable}_{j,t}\} = \mathbf{1}\left\{ \sum_{s=0}^{5}\mathbf{1}\{ E_{j,t-s} > 0\} = 6 \right\}$$

The six-month requirement is based on empirical analysis of onboarding patterns across our PSP data, which shows that most clients stabilize within 3–4 months; the six-month threshold provides a conservative buffer.

As a supplementary diagnostic, we also employ statistical change-point methods to detect transitions from onboarding to steady-state regimes. These methods estimate the posterior probability of a regime change at each point in a client's history and can identify cases where the fixed tenure rule is either too conservative (a client that stabilized quickly) or not conservative enough (a client with an unusually slow ramp-up). In practice, the six-month tenure rule serves as the primary gate, with change-point analysis informing periodic review and recalibration of the threshold.


## 6. Stratification and Size-Class Structure

The client bases of most PSPs are skewed toward small and medium-sized clients. Because size class affects growth dynamics, churn behavior, and birth/death intensity, it is essential to stratify the PSP's client base by size alongside geography and industry. Without this stratification, the PSP-derived growth signal would disproportionately reflect SME dynamics, which may differ systematically from aggregate employment growth.

Stratification to QCEW is implemented over cells defined by:

$$\text{Cell} = (\text{state},\; \text{industry},\; \text{size class})$$

Let $w_j$ denote stratification weights for cell $j$. Conceptually:

$$\sum_{j \in \text{cell}} w_j \, E_{j,t}^{\text{PSP}} \approx E_{\text{cell}, t-l}^{\text{QCEW}}$$

where $l$ denotes the QCEW publication lag. Size-class stratification is essential to avoid systematic bias arising from the overrepresentation of small firms in PSP data.

When location information is degraded (Section 4.3), the stratification cell definition may need to be adapted. Under the state-only regime, cells for small clients (where client ≈ establishment) can retain the full state × industry × size class structure. For cells incorporating redistributed large-client employment, geographic stratification should be performed at coarser levels (e.g., Census region rather than state). The reason is that the geographic allocation of large clients is itself uncertain (Section 4.3.3), and stratifying at a fine geographic level would propagate that allocation uncertainty into the weights, potentially introducing more error than it corrects.


## 7. Latent-State Modeling and Benchmark Anchoring

The continuing-units series entering the state-space model is constructed using the measurement-panel methodology described in Section 5. As a result, PSP observations measure only the intensive margin of employment change — that is, hiring and separations at firms that were already present in both the current and prior month. Client entry and exit dynamics are excluded from the measurement equation. The contribution of business births and deaths to total employment change is modeled separately (Section 7.3) and added to the continuing-units estimate to produce a total employment change nowcast.

This separation improves identification of true continuing-unit growth and prevents administrative churn from being misinterpreted as economic signal. Entry and exit rates observed in the PSP data may also serve as useful covariates for informing time-varying measurement error or provider-specific drift, though these enter the model indirectly rather than through the primary measurement equation.

PSP continuing-unit microdata are embedded in a Bayesian state-space framework, summarized here and detailed in the separate estimation methods paper.

### 7.1 Relation to Cajner et al. (2022)

Cajner et al. model CES survey data and PSP-derived data as noisy observations of latent true employment:

$$y_t^{\text{source}} = E_t + \epsilon_t^{\text{source}}$$

Our framework extends this by:

- **Decomposing employment change into continuing-units and birth/death components.** This is not merely a labeling exercise: PSP data directly measure continuing-units growth, while total employment change additionally includes the net contribution of business openings and closings, which must be modeled separately because PSPs cannot observe it.
- **Introducing hierarchical partial pooling over geography × industry × size.** This means that model parameters (bias, signal loading, noise variance) for small or sparse cells are regularized toward group-level means, improving estimates for cells with limited data while preserving genuine heterogeneity where the data support it.[^11]
- **Accounting for onboarding artifacts.** These enter the model not in the primary measurement equation but through the stabilization procedures of Section 5.2, which filter the data before they reach the model. The measurement equation itself operates on stabilized, continuing-units data only.
- **Treating QCEW as a near-census, lagged anchor.** The QCEW is modeled as an observation of true total employment change (continuing units plus births/deaths) with small measurement error, arriving with a lag. This is what allows the model to learn bias, loading, and noise parameters: the QCEW provides the "answer key" against which PSP signals are evaluated, with the lag meaning that real-time nowcasts rely on projecting the learned relationships forward.

### 7.2 Continuing Units and Birth/Death Decomposition

For each cell $j$:

$$\Delta E_{j,t}^{\text{true}} = \Delta E_{j,t}^{\text{cont}} + BD_{j,t}$$

PSP-provided data provide information about the continuing-units component:

$$y_{j,t}^{\text{PSP}} = \lambda_j \, \Delta E_{j,t}^{\text{cont}} + \epsilon_{j,t}^{\text{PSP}}$$

### 7.3 Birth/Death Model

Birth/death contributions are modeled as:

$$BD_{j,t} = \alpha_j + \beta \cdot \text{cycle}_t + \gamma \cdot BD_{j,t-l}^{\text{QCEW}} + u_{j,t}$$

With hierarchical structure over:

- Industry (nested NAICS)
- Size class (small firms have higher turnover)
- Geography (secondary)

### 7.4 QCEW as Benchmark Observation

QCEW is treated as a lagged observation of total employment change:

$$y_{j,t-l}^{\text{QCEW}} = \Delta E_{j,t-l}^{\text{true}} + \epsilon_{j,t-l}^{\text{QCEW}}$$

with small measurement error. These observations anchor bias, signal loadings, and birth/death parameters and propagate forward to inform real-time nowcasts.

### 7.5 Hierarchical Partial Pooling

*This subsection provides technical detail for specialist readers.*

Cell-level parameters (bias, signal loading, birth/death intensity) follow structured hierarchies:

- Geography: region → division → state
- Industry: domain → supersector → sector → subsector
- Size class: explicit additive effects

This stabilizes sparse cells while preserving economically meaningful heterogeneity. In practice, it means that a cell with very few PSP clients borrows strength from related cells (same industry in neighboring states, for example) rather than producing unreliable estimates from limited data alone.

### 7.6 Iterative Deployment

Given the complexity of the proposed estimation methodology, we will repeatedly iterate from simpler models to the model presented above. Each incremental complication will be evaluated against the simpler specification to demonstrate that it adds value before being incorporated into the production version. This iterative approach is described in detail in the "Strategy" section at the beginning of this paper.


## 8. Seasonality and Pay-Frequency Artifacts

Payroll series exhibit strong seasonal patterns driven by industry cycles, fiscal calendars, and payroll system behavior. Pay-frequency artifacts can induce mechanical fluctuations.

Seasonality is modeled explicitly in the latent state using additive seasonal components:

$$\Delta E_{j,t}^{\text{true}} = \mu_{j,t} + S_{j,t} + \eta_{j,t}$$

where $S_{j,t}$ follows a hierarchical seasonal structure by industry and geography, allowing seasonal amplitudes and patterns to evolve over time and differ systematically across sectors and regions.


## 9. Quality, Governance, and Concept Drift

Because PSP data are generated by business processes, discontinuities are an inherent risk. Changes in provider operations, coding practices, or client mix can alter the effective measurement concept.

We adopt a statistical-agency-style quality framework emphasizing:

- Accuracy vs. timeliness tradeoffs
- Revision analysis (real-time vs. retrospective)
- Coherence across geography and industry
- Interpretability and transparency

Operational guardrails include monitoring of hours, pay types, and payment timing, drift-aware models, and rolling recalibration to QCEW. These practices help distinguish economic dynamics from business-process artifacts.


## 10. Conclusion

PSP microdata can materially improve the timeliness and granularity of employment measurement. However, credible official-style estimates require careful treatment of statistical units, effects generated by churn among PSPs, onboarding and offboarding, size-class distortions, and representativeness. By embedding PSP microdata in a hierarchical Bayesian state-space framework anchored to QCEW, and by explicitly modeling continuing-units measurement, birth/death dynamics, and business-process artifacts, PSP data can serve as a disciplined, high-frequency measurement system that complements official statistics.

This framework extends the model-based data fusion approach of Cajner et al. by incorporating PSP-specific features related to statistical units, size-class structure, and onboarding dynamics, providing a principled foundation for real-time employment nowcasting from payroll microdata.


## References

Cajner, T., et al. (2022). *Improving the accuracy of economic measurement with multiple data sources: The case of payroll employment data.* In Abraham, Jarmin, Moyer, & Shapiro (Eds.), *Big Data for Twenty-First-Century Economic Statistics,* NBER.


[^1]: Qualified pay includes payments made to employee in any of the following: regular pay; overtime; paid sick leave; holiday pay; vacation pay; tips reported on the employee's W-4 statement; and bonuses, commissions, and severance pay that are made in regular amounts and at regular intervals.

[^2]: The BLS would face similar timing issues if the PSP reported CES data on the employer's behalf, since pay period boundaries do not always align neatly with the reference week. In practice, when PSPs do report to the CES on behalf of clients, they are required to report employment for the specific pay period including the 12th of the month, mitigating some of this ambiguity.

[^3]: Multimodality in the employee location distribution suggests the client operates at multiple distinct physical sites. A unimodal distribution centered near the client address is consistent with a single establishment.

[^4]: These thresholds are calibrated using BLS and Census commuting data. They provide a prior probability that an employee at a given distance is commuting to the client address vs. working at a remote site.

[^5]: Specifically, coverage ratios become less meaningful, shift-share decompositions absorb geographic misallocation alongside genuine compositional differences, and usability diagnostics are themselves contaminated by the same information gap.

[^6]: The information loss for these cases is modest in aggregate. Mid-size firms operating 2–3 nearby worksites in the same industry will contribute similar growth signals regardless of whether they are classified as one establishment or split into two. The classification error matters mainly for geographic precision (assigning employment to the correct MSA or county), which is secondary for the national estimates that are our initial focus.

[^7]: The BLS would not face this particular problem even if a PSP reported on an employer's behalf, because the CES requires establishment-level reporting regardless of the payroll system structure. The PSP is responsible for providing data at the worksite level when reporting to BLS.

[^8]: "Contract terminations" (earlier referred to as "contract churn") refers to the administrative event of a client ending its contract with one PSP, typically to switch to a competing provider or to bring payroll processing in-house. Mergers at the establishment level are relatively rare and are more of an issue for BLS (where two sampled establishments combining into one creates a matching problem) than for our framework (where the merged entity simply continues reporting through the PSP, albeit potentially under a new client ID that we must link).

[^9]: This structure is analogous to the CES sample rotation, where new sample units are phased in periodically while the existing sample is held fixed between rotations. We use the term "measurement panel" rather than "rotating but internally static panel" for clarity.

[^10]: Distinguishing true business closures from PSP exits in real time is not possible with available data. However, for the purpose of measuring continuing-units growth — which is what PSP data are best suited to measure — the distinction is immaterial: both cases remove the unit from the continuing-units growth calculation.

[^11]: "Hierarchical partial pooling" is a standard Bayesian technique in which parameters for individual units (here, geographic-industry-size cells) are modeled as drawn from a common distribution. The degree of pooling is determined by the data: cells with abundant data will have parameter estimates close to their cell-specific values, while cells with sparse data will be pulled toward the group mean. This is described formally in the companion estimation methods paper.