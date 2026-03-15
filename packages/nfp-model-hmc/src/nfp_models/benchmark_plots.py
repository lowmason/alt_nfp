"""Benchmark backtest visualisations.

Phase 2 plots for the benchmark prediction system
(see ``specs/benchmark_spec.md`` §4.4.3).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .benchmark_backtest import (
    HORIZONS,
    LITERATURE_BENCHMARKS,
    build_comparative_benchmarks,
    comparative_rmse,
    compute_backtest_metrics,
)
from .config import OUTPUT_DIR

_BENCHMARK_OUTPUT = OUTPUT_DIR / "benchmark"


def plot_rmse_by_horizon(
    results: pl.DataFrame,
    output_dir: Path | None = None,
) -> None:
    """RMSE-by-horizon line plot with naive and literature benchmarks.

    Parameters
    ----------
    results : pl.DataFrame
        Output of :func:`~alt_nfp.benchmark_backtest.run_benchmark_backtest`.
    output_dir : Path, optional
        Directory to save the figure.  Defaults to ``output/benchmark/``.
    """
    out = output_dir or _BENCHMARK_OUTPUT
    out.mkdir(parents=True, exist_ok=True)

    metrics = compute_backtest_metrics(results)
    years = sorted(results["march_year"].unique().to_list())
    benchmarks = build_comparative_benchmarks(years)
    comp = comparative_rmse(benchmarks)

    horizon_order = [h for h in HORIZONS if h in metrics["horizon"].to_list()]
    rmse_vals = [
        float(metrics.filter(pl.col("horizon") == h)["rmse"][0])
        for h in horizon_order
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(horizon_order, rmse_vals, "o-", color="steelblue", lw=2.5, ms=9,
            label="Model", zorder=5)

    for name, rmse in comp.items():
        ax.axhline(rmse, ls="--", lw=1.2, alpha=0.7,
                    label=f"{name} (RMSE={rmse:,.0f}K)")

    for name, rmse in LITERATURE_BENCHMARKS.items():
        ax.axhline(rmse, ls=":", lw=1.2, alpha=0.6,
                    label=f"{name} (RMSE={rmse:,.0f}K)")

    ax.set_xlabel("Horizon (months before benchmark publication)")
    ax.set_ylabel("RMSE (000s jobs)")
    ax.set_title(
        f"Benchmark Revision RMSE by Horizon — {min(years)}–{max(years)}"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "rmse_by_horizon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'rmse_by_horizon.png'}")


def plot_revision_fan_chart(
    results: pl.DataFrame,
    march_year: int,
    output_dir: Path | None = None,
) -> None:
    """Fan chart of revision posterior across horizons for one year.

    Shows the posterior mean with 50% and 90% HDI bands, plus the
    actual revision as a horizontal reference line.

    Parameters
    ----------
    results : pl.DataFrame
        Output of :func:`~alt_nfp.benchmark_backtest.run_benchmark_backtest`.
    march_year : int
        Benchmark year to plot.
    output_dir : Path, optional
        Defaults to ``output/benchmark/``.
    """
    out = output_dir or _BENCHMARK_OUTPUT
    out.mkdir(parents=True, exist_ok=True)

    yr = results.filter(pl.col("march_year") == march_year).sort("horizon")
    if yr.is_empty():
        print(f"No results for {march_year}")
        return

    horizon_order = [h for h in HORIZONS if h in yr["horizon"].to_list()]
    yr_sorted = yr.join(
        pl.DataFrame({"horizon": horizon_order, "_order": range(len(horizon_order))}),
        on="horizon",
    ).sort("_order")

    x = np.arange(len(horizon_order))
    means = yr_sorted["posterior_mean"].to_numpy().astype(float)
    hdi_lo = yr_sorted["hdi_5"].to_numpy().astype(float)
    hdi_hi = yr_sorted["hdi_95"].to_numpy().astype(float)
    stds = yr_sorted["posterior_std"].to_numpy().astype(float)
    actual = float(yr_sorted["actual"][0])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(x, hdi_lo, hdi_hi, alpha=0.15, color="steelblue",
                     label="90% HDI")
    ax.fill_between(x, means - 0.675 * stds, means + 0.675 * stds,
                     alpha=0.3, color="steelblue", label="50% interval")
    ax.plot(x, means, "o-", color="steelblue", lw=2.5, ms=9,
            label="Posterior mean")
    ax.axhline(actual, color="darkorange", lw=2, ls="--",
               label=f"Actual ({actual:+,.0f}K)")
    ax.axhline(0, color="k", lw=0.5, ls=":")

    ax.set_xticks(x)
    ax.set_xticklabels(horizon_order)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Implied revision (000s jobs)")
    ax.set_title(f"Benchmark Revision Posterior — March {march_year}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"revision_fan_{march_year}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / f'revision_fan_{march_year}.png'}")


def plot_coverage_by_horizon(
    results: pl.DataFrame,
    output_dir: Path | None = None,
) -> None:
    """Bar chart of 90% HDI coverage rate by horizon.

    Parameters
    ----------
    results : pl.DataFrame
        Output of :func:`~alt_nfp.benchmark_backtest.run_benchmark_backtest`.
    output_dir : Path, optional
        Defaults to ``output/benchmark/``.
    """
    out = output_dir or _BENCHMARK_OUTPUT
    out.mkdir(parents=True, exist_ok=True)

    metrics = compute_backtest_metrics(results)
    horizon_order = [h for h in HORIZONS if h in metrics["horizon"].to_list()]
    coverage = [
        float(metrics.filter(pl.col("horizon") == h)["coverage_90"][0])
        for h in horizon_order
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(horizon_order, coverage, color="steelblue", alpha=0.8,
                  edgecolor="white", width=0.6)
    ax.axhline(0.9, color="darkorange", ls="--", lw=1.5,
               label="Nominal 90%")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Coverage (fraction)")
    ax.set_title("90% HDI Coverage by Horizon")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar, cov in zip(bars, coverage):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{cov:.0%}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(out / "coverage_by_horizon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'coverage_by_horizon.png'}")


def plot_all(results: pl.DataFrame, output_dir: Path | None = None) -> None:
    """Generate all benchmark plots.

    Parameters
    ----------
    results : pl.DataFrame
        Output of :func:`~alt_nfp.benchmark_backtest.run_benchmark_backtest`.
    output_dir : Path, optional
        Defaults to ``output/benchmark/``.
    """
    plot_rmse_by_horizon(results, output_dir)
    plot_coverage_by_horizon(results, output_dir)
    for year in sorted(results["march_year"].unique().to_list()):
        plot_revision_fan_chart(results, year, output_dir)
