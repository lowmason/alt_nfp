#!/usr/bin/env python
# alt_nfp_estimation_v3.py — Thin runner for the alt_nfp package

'''
Growth-rate Bayesian state-space model for U.S. total nonfarm employment.

Usage:
    python alt_nfp_estimation_v3.py                      # default (production)
    python alt_nfp_estimation_v3.py --fast               # light sampler, skip LOO-CV
    python alt_nfp_estimation_v3.py --config cfg.toml    # custom TOML config
'''


# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from alt_nfp.checks import (
    print_era_summary,
    run_loo_cv,
    run_posterior_predictive_checks,
    run_prior_predictive_checks,
)
from alt_nfp.config import BASE_DIR, providers_from_settings
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
from alt_nfp.settings import load_config, save_config

# -------------------------------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="alt_nfp v3 estimation pipeline")
    parser.add_argument(
        "--fast", action="store_true",
        help="Use light sampler (2k draws, 2 chains) and skip LOO-CV",
    )
    parser.add_argument(
        "--config", type=Path, default=None, metavar="PATH",
        help="Path to TOML config file (optional; defaults to built-in values)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    resolved = cfg.resolve_paths(BASE_DIR)
    output_dir = resolved.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    providers = providers_from_settings(cfg)

    # 1. Data -------------------------------------------------------------------------------------
    panel = build_panel()
    data = panel_to_model_data(panel, providers, cfg=cfg)

    # 2. Build model ------------------------------------------------------------------------------
    model = build_model(data, cfg=cfg)

    # 3. Prior predictive checks ------------------------------------------------------------------
    run_prior_predictive_checks(model, data, cfg=cfg)

    # 4. Sample -----------------------------------------------------------------------------------
    preset = "light" if args.fast else "default"
    sampler_kwargs = cfg.sampling.get_preset(preset).to_pymc_kwargs()
    idata = sample_model(model, sampler_kwargs=sampler_kwargs, cfg=cfg)

    # 5. Diagnostics ------------------------------------------------------------------------------
    print_diagnostics(idata, data, cfg=cfg)
    print_era_summary(idata)
    print_source_contributions(idata, data)
    print_windowed_precision_budget(idata, data)
    print_provider_value_of_information(idata, data)
    plot_divergences(idata, data, cfg=cfg)

    precision_df = compute_precision_budget(idata, data)
    precision_df.write_parquet(str(output_dir / 'precision_budget.parquet'))
    print(f"\nPrecision budget saved to {output_dir / 'precision_budget.parquet'}")
    print(precision_df)

    # 6. Posterior predictive checks --------------------------------------------------------------
    run_posterior_predictive_checks(model, idata, data, cfg=cfg)

    # 7. LOO-CV (skip in --fast mode) -------------------------------------------------------------
    if not args.fast:
        run_loo_cv(model, idata, data, cfg=cfg)

    # 8. Plots ------------------------------------------------------------------------------------
    plot_growth_and_seasonal(idata, data, cfg=cfg)
    plot_reconstructed_index(idata, data, cfg=cfg)
    plot_bd_diagnostics(idata, data, cfg=cfg)
    plot_residuals(idata, data, cfg=cfg)

    # 9. Forecast ---------------------------------------------------------------------------------
    forecast_and_plot(idata, data, cfg=cfg)

    # 10. Save ------------------------------------------------------------------------------------
    idata.to_netcdf(str(output_dir / 'alt_nfp_v3_idata.nc'))
    print(f"\nInferenceData saved to {output_dir / 'alt_nfp_v3_idata.nc'}")

    save_config(cfg, output_dir / 'config.toml')
    print(f"Config saved to {output_dir / 'config.toml'}")

    print('\n' + '=' * 80)
    print('alt_nfp v3 pipeline complete.')
    print('=' * 80)


# -------------------------------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
