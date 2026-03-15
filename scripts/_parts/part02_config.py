# =============================================================================
# SECTION 1: MODEL CONFIGURATION AND HYPERPARAMETERS
# =============================================================================
#
# All priors and structural choices are documented below.  The guiding
# principles are:
#
# 1. WEAKLY INFORMATIVE PRIORS — generate plausible but not
#    indistinguishable-from-observed data under prior predictive checks.
#
# 2. LOGNORMAL FOR SCALE PARAMETERS — Every sigma in this model uses a
#    LogNormal prior instead of HalfNormal.  Reason: when a scale parameter
#    can collapse toward zero, the posterior develops a "funnel" geometry
#    (Neal's funnel).  HalfNormal(0, s) has substantial mass near zero,
#    creating a region where the sampler must take very small steps (high
#    curvature) while simultaneously exploring a wide range for the
#    associated location parameter.  LogNormal pushes mass away from zero,
#    eliminating the funnel and producing much higher effective sample sizes.
#    In MATLAB terms: if you see low ESS or divergences with HalfNormal
#    priors on variance parameters, switching to LogNormal often fixes it.
#
# 3. TAU REPARAMETERIZATION — The AR(1) process is parameterized in terms
#    of its marginal (stationary) SD "tau" rather than the innovation SD
#    "sigma_g".  When phi is near 1 (high persistence), sigma_g = tau *
#    sqrt(1 - phi^2) becomes very small, creating a ridge in the
#    (phi, sigma_g) posterior.  The sampler struggles with this ridge
#    (low ESS, slow mixing).  Using tau directly breaks the ridge because
#    tau is roughly independent of phi in the posterior.
#
# 4. STUDENT-T FOR QCEW — QCEW data has occasional large outliers from
#    NAICS industry reclassifications and quarterly boundary effects.
#    A Normal likelihood would let these outliers dominate the posterior
#    (outlier pulls the latent state).  Student-t with nu=5 provides
#    "robustness": outliers have less influence because the t-distribution
#    has heavier tails.  The cost is slightly less precision per observation
#    by a factor of (nu+1)/(nu+3) = 0.75 vs Normal.
#
# =============================================================================

# ---- Eras ----
# The model allows era-specific mean growth (mu_g) to capture the structural
# break at COVID.  Pre-COVID (2012-2019) and Post-COVID (2020+) have different
# average growth rates but share the same persistence (phi) and volatility (tau).
N_ERAS = 2
ERA_BREAK = date(2020, 1, 1)  # everything before this is era 0

# ---- AR(1) latent growth process ----
# g_cont(t) = mu_g + phi * (g_cont(t-1) - mu_g) + sigma_g * eps(t)
#
# tau ~ LogNormal(log(0.013), 0.5)
#   Center: 0.013 = 1.3%/mo marginal SD.  This is the unconditional SD of
#   the AR(1) process.  Calibrated so that posterior sigma_g ~ 0.008,
#   phi ~ 0.8 implies tau ~ 0.013.
#
# phi_raw ~ Beta(18, 2)
#   Concentrated near 0.9 (mean = 18/20 = 0.9).  Economic intuition:
#   employment growth is highly persistent month-to-month.  The Beta(18,2)
#   puts ~95% mass above 0.75.  Capped at 0.99 to prevent unit root.
#
# sigma_g = tau * sqrt(1 - phi^2)   [derived, not sampled]
LOG_TAU_MU = math.log(0.013)
LOG_TAU_SD = 0.5

# ---- QCEW observation noise ----
# Two tiers of base noise:
#   M2 months (Feb, May, Aug, Nov) = quarter-interior, most reliable
#   M1+M3 months (boundary) = noisier due to NAICS reclassification, late filings
#
# sigma_qcew_mid ~ LogNormal(log(0.0005), 0.15)
#   Very tight prior: QCEW M2 observations are near-census quality.
#   The 0.15 SD (vs 0.5 for other LogNormals) prevents the posterior from
#   wandering too far, which would let QCEW precision overwhelm all other
#   sources and create a bimodal posterior (one mode where QCEW dominates,
#   another where it doesn't).
#
# sigma_qcew_boundary ~ LogNormal(log(0.002), 0.5)
#   Wider prior for M1+M3 months.
#
# Per-observation sigma = base_sigma * revision_multiplier * era_multiplier
#   revision_multiplier: early QCEW vintages are much noisier (25x for Q1 rev0)
#   era_multiplier: post-COVID boundary months are additionally inflated
#
# Student-t with nu=5 degrees of freedom provides robustness.
QCEW_NU = 5
LOG_SIGMA_QCEW_MID_MU = math.log(0.0005)
LOG_SIGMA_QCEW_MID_SD = 0.15
LOG_SIGMA_QCEW_BOUNDARY_MU = math.log(0.002)
LOG_SIGMA_QCEW_BOUNDARY_SD = 0.5

# Post-COVID era multipliers for boundary months, keyed by revision number.
# rev 0 is 5x noisier, rev 1 is 3.5x, rev 2 is 2x.  M2 months are unaffected.
QCEW_POST_COVID_BOUNDARY_ERA_MULT: dict[int, float] = {0: 5.0, 1: 3.5, 2: 2.0}
QCEW_POST_COVID_BOUNDARY_DEFAULT = 1.0

# ---- CES observation noise ----
# sigma_ces_sa, sigma_ces_nsa ~ LogNormal(log(0.002), 0.5)
#   One sigma per vintage tier (1st print, 2nd print, Final).
#
# alpha_ces ~ Normal(0, 0.005)
#   Bias of CES relative to the latent truth (typically small).
#
# lambda_ces ~ TruncatedNormal(1.0, 0.1, lower=0.5)
#   Loading/scaling of CES on the latent state.  Truncated at 0.5 because
#   CES measures the same payroll concept as QCEW — a loading below 0.5
#   is economically implausible and would create a degenerate mode where
#   CES disconnects from the latent state (lambda -> 0).
LOG_SIGMA_CES_MU = math.log(0.002)
LOG_SIGMA_CES_SD = 0.5

# ---- Fourier seasonal ----
# K=4 harmonics, evolving as Gaussian random walks across years.
# sigma_fourier ~ LogNormal(log(0.0003) - log(k), 0.5) for harmonic k.
# The 1/k scaling encodes the assumption that higher harmonics have
# smaller year-to-year changes (smoother seasonal evolution).
N_HARMONICS = 4
LOG_SIGMA_FOURIER_MU = math.log(0.0003)
LOG_SIGMA_FOURIER_SD = 0.5

# ---- Structural birth/death ----
# bd(t) = phi_0 + phi_3 * X_cycle + sigma_bd * xi(t)
#
# phi_0 ~ Normal(0.001, 0.002)
#   Mean BD offset at average cyclical conditions.  ~0.1%/mo ≈ ~1.2%/yr
#   net business formation.
#
# sigma_bd ~ LogNormal(log(0.003), 0.5)
#   Innovation SD for the BD process.
#
# phi_3 ~ Normal(0, 0.3)  [per cyclical indicator]
#   Loading of cyclical indicators (claims, JOLTS) on BD.
LOG_SIGMA_BD_MU = math.log(0.003)
LOG_SIGMA_BD_SD = 0.5

# ---- Provider (G) measurement model ----
# alpha_g ~ Normal(0, 0.005)       bias
# lam_g   ~ Normal(1.0, 0.15)      loading on latent continuing-units growth
# sigma_pp_g ~ InverseGamma(3, 0.004)   measurement noise
#   InverseGamma(3, 0.004) has mean = 0.002 and is proper with finite variance.
#   This is the one scale parameter that uses InverseGamma rather than LogNormal,
#   because the provider noise is well-separated from zero (no funnel risk).

# ---- Sampling configuration ----
SAMPLER_KWARGS = dict(
    draws=4000,
    tune=3000,
    chains=4,
    target_accept=0.95,
    return_inferencedata=True,
)

# ---- Forecast ----
FORECAST_END = date(2026, 1, 12)

