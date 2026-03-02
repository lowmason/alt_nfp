#!/usr/bin/env python
# alt_nfp_estimation_v3.py — Thin runner for the alt_nfp package

'''
Growth-rate Bayesian state-space model for U.S. total nonfarm employment.

Usage:
    python alt_nfp_estimation_v3.py
'''


# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from alt_nfp.checks import (
    print_era_summary,
    run_loo_cv,
    run_posterior_predictive_checks,
    run_prior_predictive_checks,
)
from alt_nfp.config import OUTPUT_DIR, PROVIDERS
from alt_nfp.diagnostics import (
    compute_precision_budget,
    plot_divergences,
    print_diagnostics,
    print_provider_value_of_information,
    print_source_contributions,
    print_windowed_precision_budget,
)
from alt_nfp.forecast import forecast_and_plot
from alt_nfp.ingest import build_panel
from alt_nfp.model import build_model
from alt_nfp.panel_adapter import panel_to_model_data
from alt_nfp.plots import plot_bd_diagnostics, plot_growth_and_seasonal, plot_reconstructed_index
from alt_nfp.residuals import plot_residuals
from alt_nfp.sampling import sample_model

# -------------------------------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data -------------------------------------------------------------------------------------
    panel = build_panel()
    data = panel_to_model_data(panel, PROVIDERS)

    # 2. Build model ------------------------------------------------------------------------------
    model = build_model(data)

    # 3. Prior predictive checks ------------------------------------------------------------------
    run_prior_predictive_checks(model, data)

    # 4. Sample -----------------------------------------------------------------------------------
    idata = sample_model(model)

    # 5. Diagnostics ------------------------------------------------------------------------------
    print_diagnostics(idata, data)
    print_era_summary(idata)
    print_source_contributions(idata, data)
    print_windowed_precision_budget(idata, data)
    print_provider_value_of_information(idata, data)
    plot_divergences(idata, data)

    precision_df = compute_precision_budget(idata, data)
    precision_df.write_parquet(str(OUTPUT_DIR / 'precision_budget.parquet'))
    print(f"\nPrecision budget saved to {OUTPUT_DIR / 'precision_budget.parquet'}")
    print(precision_df)

    # 6. Posterior predictive checks --------------------------------------------------------------
    run_posterior_predictive_checks(model, idata, data)

    # 7. LOO-CV -----------------------------------------------------------------------------------
    run_loo_cv(model, idata, data)

    # 8. Plots ------------------------------------------------------------------------------------
    plot_growth_and_seasonal(idata, data)
    plot_reconstructed_index(idata, data)
    plot_bd_diagnostics(idata, data)
    plot_residuals(idata, data)

    # 9. Forecast ---------------------------------------------------------------------------------
    forecast_and_plot(idata, data)

    # 10. Save InferenceData ----------------------------------------------------------------------
    idata.to_netcdf(str(OUTPUT_DIR / 'alt_nfp_v3_idata.nc'))
    print(f"\nInferenceData saved to {OUTPUT_DIR / 'alt_nfp_v3_idata.nc'}")

    print('\n' + '=' * 80)
    print('alt_nfp v3 pipeline complete.')
    print('=' * 80)


# -------------------------------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
