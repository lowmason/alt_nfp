#!/usr/bin/env python
# ---------------------------------------------------------------------------
# alt_nfp_estimation_v3.py — Thin runner for the alt_nfp package
# ---------------------------------------------------------------------------
"""Growth-rate Bayesian state-space model for U.S. total nonfarm employment.

Usage:
    python alt_nfp_estimation_v3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from alt_nfp.checks import run_loo_cv, run_posterior_predictive_checks, run_prior_predictive_checks
from alt_nfp.config import OUTPUT_DIR
from alt_nfp.data import load_data
from alt_nfp.diagnostics import plot_divergences, print_diagnostics, print_source_contributions
from alt_nfp.forecast import forecast_and_plot
from alt_nfp.model import build_model
from alt_nfp.plots import plot_bd_diagnostics, plot_growth_and_seasonal, plot_reconstructed_index
from alt_nfp.residuals import plot_residuals
from alt_nfp.sampling import sample_model


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data ------------------------------------------------------------------
    data = load_data()

    # 2. Build model -----------------------------------------------------------
    model = build_model(data)

    # 3. Prior predictive checks -----------------------------------------------
    run_prior_predictive_checks(model, data)

    # 4. Sample ----------------------------------------------------------------
    idata = sample_model(model)

    # 5. Diagnostics -----------------------------------------------------------
    print_diagnostics(idata, data)
    print_source_contributions(idata, data)
    plot_divergences(idata, data)

    # 6. Posterior predictive checks -------------------------------------------
    run_posterior_predictive_checks(model, idata, data)

    # 7. LOO-CV ----------------------------------------------------------------
    run_loo_cv(model, idata, data)

    # 8. Plots -----------------------------------------------------------------
    plot_growth_and_seasonal(idata, data)
    plot_reconstructed_index(idata, data)
    plot_bd_diagnostics(idata, data)
    plot_residuals(idata, data)

    # 9. Forecast --------------------------------------------------------------
    forecast_and_plot(idata, data)

    # 10. Save InferenceData ---------------------------------------------------
    idata.to_netcdf(str(OUTPUT_DIR / "alt_nfp_v3_idata.nc"))
    print(f"\nInferenceData saved to {OUTPUT_DIR / 'alt_nfp_v3_idata.nc'}")

    print("\n" + "=" * 72)
    print("alt_nfp v3 pipeline complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
