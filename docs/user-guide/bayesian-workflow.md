# Bayesian Workflow

`alt_nfp` implements the iterative Bayesian workflow described in
Gelman et al. (2020) and Gabry et al. (2019).  The workflow is:
build → check → fit → diagnose → evaluate → iterate.

## Workflow Checklist

| Step | Module | Function |
|---|---|---|
| 1. Load & explore data | `alt_nfp.data` | `load_data()` |
| 2. Build model | `alt_nfp.model` | `build_model()` |
| 3. Prior predictive checks | `alt_nfp.checks` | `run_prior_predictive_checks()` |
| 4. MCMC sampling | `alt_nfp.sampling` | `sample_model()` |
| 5. Convergence diagnostics | `alt_nfp.diagnostics` | `print_diagnostics()` |
| 6. Divergence analysis | `alt_nfp.diagnostics` | `plot_divergences()` |
| 7. Source contributions | `alt_nfp.diagnostics` | `print_source_contributions()` |
| 8. Posterior predictive checks | `alt_nfp.checks` | `run_posterior_predictive_checks()` |
| 9. LOO-CV | `alt_nfp.checks` | `run_loo_cv()` |
| 10. Residual analysis | `alt_nfp.residuals` | `plot_residuals()` |
| 11. Results plots | `alt_nfp.plots` | `plot_growth_and_seasonal()` etc. |
| 12. Forecast | `alt_nfp.forecast` | `forecast_and_plot()` |
| 13. Sensitivity | `alt_nfp.sensitivity` | `run_sensitivity()` |
| 14. Backtest | `alt_nfp.backtest` | `run_backtest()` |

## Prior Selection

Priors follow the weakly-informative strategy (Gelman et al. 2020):

!!! tip "Prior Predictive Check"
    Priors should generate data that is **plausible but not
    indistinguishable** from the observed data.  If the prior predictive
    distribution is too tight, priors are over-informative; if too wide,
    the sampler may struggle.

The prior predictive check (`run_prior_predictive_checks`) produces:

- Histograms comparing prior-predictive vs observed data for each source.
- Prior trajectory draws for the latent growth process.

## Computational Diagnostics

After sampling, check for:

| Diagnostic | Threshold | Module |
|---|---|---|
| R-hat | < 1.01 | `print_diagnostics()` |
| ESS bulk | > 400 | `print_diagnostics()` |
| Divergences | 0 (ideal) | `plot_divergences()` |

!!! warning "The Folk Theorem"
    Computational problems often indicate model problems.  If you see
    many divergences, consider reparameterisation (e.g., non-centred)
    before increasing `target_accept`.

## Posterior Predictive Checks

Two complementary diagnostics:

### Density Overlays

For each data source, 100 replicated datasets are drawn from the posterior
predictive and overlaid on the observed data histogram.  The observed
density should fall within the replicated envelope.

### Test Statistics

Statistics orthogonal to the model parameters (skewness, lag-1 ACF) are
computed for replicated data and compared to the observed value.  A
two-sided p-value < 0.05 suggests systematic misfit on that dimension.

## LOO-CV

Pareto-smoothed importance-sampling LOO cross-validation (PSIS-LOO) is
computed per data source.  Key diagnostics:

| k-hat | Interpretation |
|---|---|
| < 0.5 | Good — reliable PSIS estimate |
| 0.5–0.7 | Warning — some influential points |
| > 0.7 | Bad — PSIS unreliable, consider moment matching or resampling |

LOO-PIT uniformity:

- "Frown" shape → predictive distributions too wide.
- "Smile" shape → predictive distributions too narrow.

## Residual Analysis

Standardised residuals should be approximately iid \(\mathcal{N}(0, 1)\).
Look for:

- **Temporal patterns** → missing dynamics (e.g., regime change).
- **Heavy tails** → outliers or misspecified noise distribution.
- **Level shifts** → structural breaks.

For AR(1) providers, residuals are pre-whitened (innovation residuals).

## References

- Gelman, A. et al. (2020). *Bayesian Workflow*. arXiv:2011.01808.
- Gabry, J. et al. (2019). *Visualization in Bayesian Workflow*. JRSS-A.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). *Practical Bayesian Model
  Evaluation Using LOO-CV and WAIC*.
- Betancourt, M. (2017). *A Conceptual Introduction to Hamiltonian Monte
  Carlo*.
