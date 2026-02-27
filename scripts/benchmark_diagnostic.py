#!/usr/bin/env python
"""Benchmark revision diagnostic — Phase 1.

Runs the benchmark revision extraction on an existing posterior and prints
a summary table comparing posterior predictions against known actuals.

Usage:
    python scripts/benchmark_diagnostic.py
    python scripts/benchmark_diagnostic.py --idata output/alt_nfp_v3_idata.nc
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from alt_nfp.benchmark import (
    decompose_benchmark_revision,
    extract_benchmark_revision,
    load_anchor_level_from_vintage_store,
    summarize_revision_posterior,
)
from alt_nfp.config import OUTPUT_DIR, PROVIDERS
from alt_nfp.ingest import build_panel
from alt_nfp.lookups.benchmark_revisions import BENCHMARK_REVISIONS
from alt_nfp.panel_adapter import panel_to_model_data

COVID_YEARS = {2020, 2021}


def _load_data_and_panel() -> tuple[dict, bool]:
    """Load model data dict, preferring the panel path."""
    panel = build_panel(use_legacy=False)
    if len(panel) > 0:
        try:
            data = panel_to_model_data(panel, PROVIDERS)
            if len(data["qcew_obs"]) > 0:
                data["panel"] = panel
                return data, True
        except (ValueError, KeyError) as e:
            print(f"Panel adapter failed ({e}), trying legacy")

    from alt_nfp.data import load_data

    return load_data(), False


def _eligible_years(dates: list[date]) -> list[int]:
    """Return benchmark years whose full window is within the estimation period."""
    date_min, date_max = dates[0], dates[-1]
    years = []
    for y in sorted(BENCHMARK_REVISIONS.keys()):
        if y in COVID_YEARS:
            continue
        window_start = date(y - 1, 4, 1)
        window_end = date(y, 3, 31)
        if window_start >= date_min and window_end <= date_max:
            years.append(y)
    return years


def main(idata_path: Path | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load idata ---
    if idata_path is None:
        candidates = [
            OUTPUT_DIR / "idata.nc",
            OUTPUT_DIR / "alt_nfp_v3_idata.nc",
        ]
        for p in candidates:
            if p.exists():
                idata_path = p
                break
    if idata_path is None or not idata_path.exists():
        print("ERROR: No InferenceData file found. Run the model first.")
        sys.exit(1)

    print(f"Loading InferenceData from {idata_path}")
    idata = az.from_netcdf(str(idata_path))

    # --- Load data ---
    print("Loading data...")
    data, used_panel = _load_data_and_panel()
    print(f"  Data source: {'panel (vintage store)' if used_panel else 'legacy load_data()'}")
    print(f"  Estimation period: {data['dates'][0]} to {data['dates'][-1]}")
    print(f"  T = {data['T']} months")

    # --- Determine eligible years ---
    years = _eligible_years(data["dates"])
    if not years:
        print("ERROR: No eligible benchmark years within estimation window.")
        sys.exit(1)
    print(f"  Eligible benchmark years: {years}")

    # --- Pre-compute anchor levels ---
    anchor_levels: dict[int, float] = {}
    for y in years:
        try:
            anchor_levels[y] = load_anchor_level_from_vintage_store(y)
            print(f"  Anchor March {y - 1}: {anchor_levels[y]:,.0f}K")
        except (ValueError, FileNotFoundError):
            print(f"  WARNING: Could not load anchor for {y} from vintage store")

    # --- Extract revisions ---
    print()
    print("=" * 80)
    print("Benchmark Revision Diagnostic (Phase 1 — full-sample posterior)")
    print("=" * 80)
    header = (
        f"{'March Y':>8} | {'Post Mean':>10} | {'Std':>7} | "
        f"{'90% HDI':>20} | {'Actual':>8} | {'Error':>8}"
    )
    print(header)
    print("-" * len(header))

    results: dict[int, dict] = {}
    for y in years:
        actual = BENCHMARK_REVISIONS.get(y)
        anchor = anchor_levels.get(y)

        try:
            samples = extract_benchmark_revision(idata, data, y, anchor_level=anchor)
            summary = summarize_revision_posterior(samples, actual=actual)
            results[y] = summary

            hdi_str = f"[{summary['hdi_5']:+,.0f}, {summary['hdi_95']:+,.0f}]"
            actual_str = f"{actual:+,.0f}" if actual is not None else "N/A"
            error_str = f"{summary['error']:+,.0f}" if "error" in summary else "N/A"
            print(
                f"{y:>8} | {summary['mean']:>+10,.0f} | "
                f"{summary['std']:>7,.0f} | {hdi_str:>20} | "
                f"{actual_str:>8} | {error_str:>8}"
            )
        except (ValueError, KeyError) as e:
            print(f"{y:>8} | ERROR: {e}")

    # --- Decomposition for the most recent year ---
    latest = years[-1]
    anchor = anchor_levels.get(latest)
    try:
        decomp = decompose_benchmark_revision(idata, data, latest, anchor_level=anchor)
        total_mean = np.mean(decomp["total"])
        cont_mean = np.mean(decomp["cont_divergence"])
        bd_mean = np.mean(decomp["bd_accumulation"])

        print()
        print(f"Decomposition ({latest}):")
        pct_cont = 100 * cont_mean / total_mean if total_mean != 0 else 0
        pct_bd = 100 * bd_mean / total_mean if total_mean != 0 else 0
        print(f"  Continuing-units divergence: {cont_mean:+,.0f} ({pct_cont:+.0f}%)")
        print(f"  BD accumulation:             {bd_mean:+,.0f} ({pct_bd:+.0f}%)")
    except (ValueError, KeyError) as e:
        print(f"\nDecomposition for {latest} failed: {e}")

    # --- Plot: posterior density for latest two years with actuals ---
    plot_years = [y for y in years if y in results][-2:]
    if plot_years:
        fig, axes = plt.subplots(1, len(plot_years), figsize=(6 * len(plot_years), 4))
        if len(plot_years) == 1:
            axes = [axes]

        for ax, y in zip(axes, plot_years, strict=True):
            anchor = anchor_levels.get(y)
            samples = extract_benchmark_revision(idata, data, y, anchor_level=anchor)
            ax.hist(samples, bins=60, density=True, alpha=0.7, color="steelblue")
            ax.axvline(np.mean(samples), color="navy", ls="--", lw=1.5, label="Mean")
            actual = BENCHMARK_REVISIONS.get(y)
            if actual is not None:
                ax.axvline(actual, color="red", ls="-", lw=2, label=f"Actual ({actual:+,})")
            ax.set_title(f"Implied Benchmark Revision — March {y}")
            ax.set_xlabel("Revision (000s)")
            ax.legend(fontsize=8)

        fig.tight_layout()
        out_path = OUTPUT_DIR / "benchmark_diagnostic.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark revision diagnostic")
    parser.add_argument(
        "--idata",
        type=Path,
        default=None,
        help="Path to InferenceData NetCDF file",
    )
    args = parser.parse_args()
    main(idata_path=args.idata)
