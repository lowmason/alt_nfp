#!/usr/bin/env python
"""Reforecast January 2026 with and without provider G's January observation.

CES is censored from January 2026 onward so that January is truly a
nowcast (not observed via CES).  The two runs differ only in whether
G's early January signal is available:

  jan_0 — G January excluded, CES January censored → pure trend nowcast
  jan_1 — G January included, CES January censored → provider-informed nowcast

Outputs are saved to output/jan_0/ and output/jan_1/.
"""
from __future__ import annotations

import shutil
import sys
from copy import deepcopy
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
from alt_nfp.plots import (
    plot_bd_diagnostics,
    plot_growth_and_seasonal,
    plot_reconstructed_index,
)
from alt_nfp.residuals import plot_residuals
from alt_nfp.sampling import sample_model

JAN_2026 = date(2026, 1, 12)


def _censor_g_january(data: dict) -> dict:
    """Return a copy of data with G's last observation (Jan 2026) removed."""
    data = deepcopy(data)
    for pp in data["pp_data"]:
        if pp["config"].name.upper() != "G":
            continue
        obs = pp["pp_obs"]
        if len(obs) == 0:
            continue
        last_t = obs[-1]
        pp["g_pp"][last_t] = np.nan
        pp["pp_obs"] = obs[:-1]
        if pp["births"] is not None and len(pp["births_obs"]) > 0:
            last_b = pp["births_obs"][-1]
            pp["births"][last_b] = np.nan
            pp["births_obs"] = pp["births_obs"][:-1]
    return data


def _run_pipeline(data: dict, label: str, dest: Path) -> None:
    """Run the full estimation pipeline and copy outputs to dest."""
    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model(data)
    run_prior_predictive_checks(model, data)
    idata = sample_model(model)

    print_diagnostics(idata, data)
    print_era_summary(idata)
    print_source_contributions(idata, data)
    print_windowed_precision_budget(idata, data)
    print_provider_value_of_information(idata, data)
    plot_divergences(idata, data)

    precision_df = compute_precision_budget(idata, data)
    precision_df.write_parquet(str(OUTPUT_DIR / "precision_budget.parquet"))
    print(precision_df)

    run_posterior_predictive_checks(model, idata, data)
    run_loo_cv(model, idata, data)

    plot_growth_and_seasonal(idata, data)
    plot_reconstructed_index(idata, data)
    plot_bd_diagnostics(idata, data)
    plot_residuals(idata, data)
    forecast_and_plot(idata, data)

    idata.to_netcdf(str(OUTPUT_DIR / "alt_nfp_v3_idata.nc"))

    dest.mkdir(parents=True, exist_ok=True)
    for f in OUTPUT_DIR.glob("*"):
        if f.is_file() and f.suffix in (".png", ".parquet", ".nc"):
            shutil.copy2(f, dest / f.name)
    print(f"\nOutputs copied to {dest}")


def main() -> None:
    panel = build_panel()

    # CES censored from January 2026 + G January included
    data_with_g = panel_to_model_data(
        panel, PROVIDERS, censor_ces_from=JAN_2026
    )

    last_date = data_with_g["dates"][-1]
    n_ces_sa = len(data_with_g["ces_sa_obs"])
    n_ces_nsa = len(data_with_g["ces_nsa_obs"])
    print(f"Panel last date: {last_date}")
    print(f"CES censored from {JAN_2026}: {n_ces_sa} SA obs, {n_ces_nsa} NSA obs")
    for pp in data_with_g["pp_data"]:
        n = len(pp["pp_obs"])
        print(f"  {pp['name']}: {n} obs, last t-index={pp['pp_obs'][-1] if n else 'none'}")

    # jan_0: also remove G's January observation
    data_no_g = _censor_g_january(data_with_g)

    _run_pipeline(data_no_g, "jan_0: CES censored, G January EXCLUDED", OUTPUT_DIR / "jan_0")
    _run_pipeline(data_with_g, "jan_1: CES censored, G January INCLUDED", OUTPUT_DIR / "jan_1")

    print("\n" + "=" * 80)
    print("January 2026 reforecast complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
