#!/usr/bin/env python
"""Evaluation protocol §3.2 for Task 2a — Era-specific latent state parameters.

Runs the Phase 2 evaluation protocol (national_model_spec.md §3.2):

1. Benchmark backtest with and without era-specific parameters.
2. Nowcast backtest with and without era-specific parameters.
3. Sampler diagnostics (R-hat, ESS, divergences) for both variants.

Usage:
    python scripts/evaluate_2a.py
    python scripts/evaluate_2a.py --quick        # fewer runs
    python scripts/evaluate_2a.py --nowcast-only # only nowcast backtest (era vs baseline)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from alt_nfp.backtest import run_backtest
from alt_nfp.benchmark_backtest import (
    compute_backtest_metrics,
    run_benchmark_backtest,
)
from alt_nfp.config import OUTPUT_DIR, PROVIDERS
from alt_nfp.diagnostics import print_diagnostics
from alt_nfp.ingest import build_panel
from alt_nfp.model import build_model
from alt_nfp.panel_adapter import panel_to_model_data
from alt_nfp.sampling import LIGHT_SAMPLER_KWARGS, sample_model


def _nowcast_rmse(results: list[dict]) -> float:
    """RMSE of jobs-added error (thousands) from nowcast backtest results."""
    if not results:
        return float("nan")
    errs = np.array([r["error_change_k"] for r in results])
    return float(np.sqrt(np.mean(errs**2)))


def run_diagnostics_run(use_era_specific: bool) -> None:
    """One full-data fit and print R-hat, ESS, divergences."""
    label = "era_specific" if use_era_specific else "baseline"
    print(f"\n{'='*70}")
    print(f"SAMPLER DIAGNOSTICS — {label}")
    print("=" * 70)
    panel = build_panel()
    data = panel_to_model_data(panel, PROVIDERS, industry_code="00")
    if not use_era_specific and "era_idx" in data:
        data = {k: v for k, v in data.items() if k != "era_idx"}
    model = build_model(data)
    idata = sample_model(model, sampler_kwargs=LIGHT_SAMPLER_KWARGS)
    print_diagnostics(idata, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation protocol §3.2 for Task 2a")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run: 1 benchmark year, 2 nowcast months, skip second diagnostics",
    )
    parser.add_argument(
        "--nowcast-only",
        action="store_true",
        help="Run only the nowcast backtest (era vs baseline); skip benchmark and diagnostics",
    )
    args = parser.parse_args()

    nowcast_only = args.nowcast_only
    if args.quick:
        years = [2024]
        horizons = ["T-3", "T-1"]
        n_backtest = 2
    else:
        years = None  # DEFAULT_YEARS
        horizons = None  # HORIZONS
        n_backtest = 24

    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    benchmark_ok = True  # N/A when --nowcast-only

    if not nowcast_only:
        # ── 1. Benchmark backtest: with and without era-specific ─────────
        print("\n" + "=" * 70)
        print("BENCHMARK BACKTEST — with era-specific parameters")
        print("=" * 70)
        ckpt_era = out / "evaluate_2a_benchmark_era"
        df_era = run_benchmark_backtest(
            years=years,
            horizons=horizons,
            checkpoint_dir=ckpt_era,
            use_era_specific=True,
        )
        metrics_era = compute_backtest_metrics(df_era)
        df_era.write_parquet(out / "evaluate_2a_benchmark_era_results.parquet")

        print("\n" + "=" * 70)
        print("BENCHMARK BACKTEST — baseline (scalar mu_g, phi)")
        print("=" * 70)
        ckpt_baseline = out / "evaluate_2a_benchmark_baseline"
        df_baseline = run_benchmark_backtest(
            years=years,
            horizons=horizons,
            checkpoint_dir=ckpt_baseline,
            use_era_specific=False,
        )
        metrics_baseline = compute_backtest_metrics(df_baseline)
        df_baseline.write_parquet(out / "evaluate_2a_benchmark_baseline_results.parquet")

        # ── Benchmark comparison ────────────────────────────────────────
        print("\n" + "=" * 70)
        print("BENCHMARK RMSE BY HORIZON — era vs baseline")
        print("=" * 70)
        rmse_era = dict(zip(metrics_era["horizon"], metrics_era["rmse"]))
        rmse_base = dict(zip(metrics_baseline["horizon"], metrics_baseline["rmse"]))
        all_h = sorted(set(rmse_era) | set(rmse_base))
        print(f"{'Horizon':>8}  {'Era':>10}  {'Baseline':>10}  {'Diff':>10}  {'Verdict':>10}")
        print("-" * 50)
        for h in all_h:
            e, b = rmse_era.get(h, np.nan), rmse_base.get(h, np.nan)
            diff = e - b if np.isfinite(e) and np.isfinite(b) else np.nan
            if np.isfinite(diff):
                verdict = "OK" if diff <= 0 else "worse"
            else:
                verdict = "—"
            print(f"{h:>8}  {e:>10.0f}  {b:>10.0f}  {diff:>+10.0f}  {verdict:>10}")
        benchmark_ok = all(
            np.isfinite(rmse_era.get(h)) and np.isfinite(rmse_base.get(h))
            and rmse_era.get(h, np.inf) <= rmse_base.get(h, -np.inf) * 1.001
            for h in all_h
        )

    # ── 2. Nowcast backtest: with and without era-specific ───────────────
    print("\n" + "=" * 70)
    print("NOWCAST BACKTEST — with era-specific parameters")
    print("=" * 70)
    results_era = run_backtest(n_backtest=n_backtest, use_era_specific=True)
    rmse_nowcast_era = _nowcast_rmse(results_era)

    print("\n" + "=" * 70)
    print("NOWCAST BACKTEST — baseline (scalar mu_g, phi)")
    print("=" * 70)
    results_baseline = run_backtest(n_backtest=n_backtest, use_era_specific=False)
    rmse_nowcast_baseline = _nowcast_rmse(results_baseline)

    print("\n" + "=" * 70)
    print("NOWCAST RMSE (jobs-added error, thousands)")
    print("=" * 70)
    print(f"  Era-specific: {rmse_nowcast_era:,.0f} k")
    print(f"  Baseline:     {rmse_nowcast_baseline:,.0f} k")
    nowcast_ok = rmse_nowcast_era <= rmse_nowcast_baseline * 1.001

    # ── 3. Sampler diagnostics (skipped when --nowcast-only) ─────────────
    if not nowcast_only:
        run_diagnostics_run(use_era_specific=True)
        if not args.quick:
            run_diagnostics_run(use_era_specific=False)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION PROTOCOL §3.2 — SUMMARY (Task 2a)" + (" [nowcast only]" if nowcast_only else ""))
    print("=" * 70)
    if not nowcast_only:
        print(f"  Benchmark RMSE: no degradation at any horizon  →  {'PASS' if benchmark_ok else 'FAIL'}")
    print(f"  Nowcast RMSE:   improve or neutral             →  {'PASS' if nowcast_ok else 'FAIL'}")
    if not nowcast_only:
        print("  Sampler: check output above for R-hat, ESS, divergences.")
    print("=" * 70)


if __name__ == "__main__":
    main()
