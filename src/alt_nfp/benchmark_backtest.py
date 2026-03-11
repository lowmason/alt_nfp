"""Benchmark revision backtest across historical years and horizons.

Phase 2 of the benchmark prediction system (see ``specs/benchmark_spec.md`` §4).

Produces RMSE-by-horizon curves across benchmark years using only information
available at each horizon.  Data censoring via the ``as_of`` parameter in
:func:`~alt_nfp.panel_adapter.panel_to_model_data` prevents lookahead bias.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from .benchmark import (
    _find_benchmark_window,
    _get_anchor_level,
    extract_benchmark_revision,
    summarize_revision_posterior,
)
from .config import BASE_DIR, providers_from_settings
from .ingest import build_panel
from .lookups.benchmark_revisions import BENCHMARK_REVISIONS
from .model import build_model
from .panel_adapter import panel_to_model_data
from .sampling import sample_model
from .settings import NowcastConfig

# ── Year sets ────────────────────────────────────────────────────────────

DEFAULT_YEARS: list[int] = [2022, 2023, 2024, 2025]

EXTENDED_YEARS: list[int] = [
    2014, 2015, 2016, 2017, 2018, 2019,
    2022, 2023, 2024, 2025,
]

HORIZONS: list[str] = ["T-12", "T-9", "T-6", "T-3", "T-1"]

# ── Horizon mapping ─────────────────────────────────────────────────────

_HORIZON_SPEC: dict[str, tuple[int, int]] = {
    # horizon → (month, year_offset relative to march_year)
    "T-12": (4, 0),   # April Y
    "T-9":  (7, 0),   # July Y
    "T-6":  (10, 0),  # October Y
    "T-3":  (1, 1),   # January Y+1
    "T-1":  (2, 1),   # February Y+1
}


def horizon_to_as_of(march_year: int, horizon: str) -> date:
    """Map ``(march_year, horizon)`` to a concrete ``as_of`` date.

    The ``as_of`` date determines the real-time information set: only data
    published on or before this date is available to the model.

    Parameters
    ----------
    march_year : int
        March reference year for the benchmark (e.g. 2025).
    horizon : str
        One of ``'T-12'``, ``'T-9'``, ``'T-6'``, ``'T-3'``, ``'T-1'``.

    Returns
    -------
    date
        Mid-month date representing the information cutoff.

    Raises
    ------
    ValueError
        If *horizon* is not recognised.
    """
    if horizon not in _HORIZON_SPEC:
        raise ValueError(
            f"Unknown horizon {horizon!r}.  Must be one of {list(_HORIZON_SPEC)}."
        )
    month, year_offset = _HORIZON_SPEC[horizon]
    return date(march_year + year_offset, month, 15)


# ── Backtest orchestrator ────────────────────────────────────────────────

_CHECKPOINT_SCHEMA = {
    "march_year": pl.Int32,
    "horizon": pl.Utf8,
    "as_of_date": pl.Date,
    "posterior_mean": pl.Float64,
    "posterior_median": pl.Float64,
    "posterior_std": pl.Float64,
    "hdi_5": pl.Float64,
    "hdi_95": pl.Float64,
    "actual": pl.Float64,
    "error": pl.Float64,
    "squared_error": pl.Float64,
}


def run_benchmark_backtest(
    years: list[int] | None = None,
    horizons: list[str] | None = None,
    sampler_kwargs: dict | None = None,
    checkpoint_dir: Path | None = None,
    *,
    use_era_specific: bool = True,
    cfg: NowcastConfig | None = None,
) -> pl.DataFrame:
    """Run the benchmark revision backtest.

    For each ``(year, horizon)`` pair the procedure is:

    1. Compute ``as_of`` from :func:`horizon_to_as_of`.
    2. Build censored data via ``panel_to_model_data(as_of=...)``.
    3. Fit the model with light sampling.
    4. Extract the implied benchmark revision posterior.
    5. Summarise and record.

    Parameters
    ----------
    years : list[int], optional
        Benchmark years to evaluate.  Defaults to :data:`DEFAULT_YEARS`
        (post-pandemic: 2022–2025).
    horizons : list[str], optional
        Horizon labels.  Defaults to :data:`HORIZONS`.
    sampler_kwargs : dict, optional
        Override sampling kwargs.  Defaults to
        :data:`~alt_nfp.sampling.LIGHT_SAMPLER_KWARGS`.
    checkpoint_dir : Path, optional
        If provided, intermediate results are saved to a parquet file in
        this directory after each run, enabling resumption.
    use_era_specific : bool, optional
        If True (default), use era-specific latent parameters when
        ``era_idx`` is in the data dict.  If False, pass data without
        ``era_idx`` so the model uses scalar mu_g/phi (Phase 1 baseline).

    Returns
    -------
    pl.DataFrame
        One row per ``(year, horizon)`` with posterior summary statistics,
        actual revision, and error.
    """
    if cfg is None:
        cfg = NowcastConfig()
    providers = providers_from_settings(cfg)

    if years is None:
        years = DEFAULT_YEARS
    if horizons is None:
        horizons = HORIZONS
    if sampler_kwargs is None:
        sampler_kwargs = cfg.sampling.get_preset(cfg.backtest.sampling_preset).to_pymc_kwargs()

    ckpt_path = (
        checkpoint_dir / "benchmark_backtest_checkpoint.parquet"
        if checkpoint_dir is not None
        else None
    )

    results: list[dict] = _load_checkpoint(ckpt_path)
    completed = {(r["march_year"], r["horizon"]) for r in results}

    panel = build_panel()

    # Pre-compute anchor levels from the UNCENSORED panel so that
    # as_of censoring doesn't hide the March Y-1 employment level.
    uncensored_data = panel_to_model_data(panel, providers, industry_code="00", cfg=cfg)
    anchor_levels: dict[int, float | None] = {}
    for y in years:
        try:
            anchor_levels[y] = _get_anchor_level(uncensored_data, y)
        except (ValueError, KeyError):
            anchor_levels[y] = None

    total_runs = sum(
        1 for y in years
        if BENCHMARK_REVISIONS.get(y) is not None
        for _ in horizons
    )
    run_num = len(completed)

    for march_year in years:
        actual = BENCHMARK_REVISIONS.get(march_year)
        if actual is None:
            continue

        for horizon in horizons:
            if (march_year, horizon) in completed:
                continue

            run_num += 1
            as_of = horizon_to_as_of(march_year, horizon)
            print(
                f"\n{'=' * 70}\n"
                f"Benchmark backtest {run_num}/{total_runs}: "
                f"year={march_year}  horizon={horizon}  as_of={as_of}\n"
                f"{'=' * 70}"
            )

            data = panel_to_model_data(
                panel, providers, as_of=as_of, industry_code="00", cfg=cfg,
            )
            try:
                _find_benchmark_window(data["dates"], march_year)
            except ValueError:
                print(
                    f"  Skipping: benchmark window for {march_year} not fully "
                    "in estimation period."
                )
                continue

            if not use_era_specific and "era_idx" in data:
                data = {k: v for k, v in data.items() if k != "era_idx"}

            model = build_model(data, cfg=cfg)
            idata = sample_model(model, sampler_kwargs=sampler_kwargs, cfg=cfg)

            samples = extract_benchmark_revision(
                idata, data, march_year, anchor_level=anchor_levels.get(march_year),
            )
            summary = summarize_revision_posterior(samples, actual=actual)

            result = {
                "march_year": march_year,
                "horizon": horizon,
                "as_of_date": as_of,
                "posterior_mean": summary["mean"],
                "posterior_median": summary["median"],
                "posterior_std": summary["std"],
                "hdi_5": summary["hdi_5"],
                "hdi_95": summary["hdi_95"],
                "actual": actual,
                "error": summary.get("error", summary["mean"] - actual),
                "squared_error": (summary["mean"] - actual) ** 2,
            }
            results.append(result)
            _save_checkpoint(results, ckpt_path)

            _print_run_summary(result)

    df = pl.DataFrame(results, schema=_CHECKPOINT_SCHEMA)
    _print_backtest_summary(df)
    return df


# ── Evaluation metrics ───────────────────────────────────────────────────

def compute_backtest_metrics(results: pl.DataFrame) -> pl.DataFrame:
    """Compute per-horizon evaluation metrics.

    Parameters
    ----------
    results : pl.DataFrame
        Output of :func:`run_benchmark_backtest`.

    Returns
    -------
    pl.DataFrame
        One row per horizon with bias, MAE, RMSE, 90% HDI coverage,
        and average interval width.
    """
    metrics: list[dict] = []
    for horizon in HORIZONS:
        h = results.filter(pl.col("horizon") == horizon)
        if h.is_empty():
            continue

        errors = h["error"].to_numpy().astype(float)
        actuals = h["actual"].to_numpy().astype(float)
        hdi_lo = h["hdi_5"].to_numpy().astype(float)
        hdi_hi = h["hdi_95"].to_numpy().astype(float)

        metrics.append({
            "horizon": horizon,
            "n_years": len(h),
            "bias": float(np.mean(errors)),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "coverage_90": float(np.mean((actuals >= hdi_lo) & (actuals <= hdi_hi))),
            "avg_interval_width": float(np.mean(hdi_hi - hdi_lo)),
        })
    return pl.DataFrame(metrics)


# ── Comparative benchmarks ───────────────────────────────────────────────

LITERATURE_BENCHMARKS: dict[str, float] = {
    "Decker (2024, BED)": 229.0,
    "Cajner et al. (CES-ex-BD)": 243.0,
}


def build_comparative_benchmarks(years: list[int]) -> pl.DataFrame:
    """Build naive benchmark predictions for RMSE comparison.

    Two naive strategies:

    * **Naive zero:** predict no revision (market default).
    * **Prior-year revision:** predict ``R_Y = R_{Y-1}``.

    Parameters
    ----------
    years : list[int]
        Benchmark years to evaluate.

    Returns
    -------
    pl.DataFrame
        Columns: ``benchmark``, ``march_year``, ``prediction``,
        ``actual``, ``error``, ``squared_error``.
    """
    rows: list[dict] = []
    for year in years:
        actual = BENCHMARK_REVISIONS.get(year)
        if actual is None:
            continue

        rows.append({
            "benchmark": "naive_zero",
            "march_year": year,
            "prediction": 0.0,
            "actual": actual,
            "error": -actual,
            "squared_error": actual ** 2,
        })

        prior = BENCHMARK_REVISIONS.get(year - 1)
        if prior is not None:
            rows.append({
                "benchmark": "prior_year",
                "march_year": year,
                "prediction": prior,
                "actual": actual,
                "error": prior - actual,
                "squared_error": (prior - actual) ** 2,
            })

    return pl.DataFrame(rows)


def comparative_rmse(benchmarks: pl.DataFrame) -> dict[str, float]:
    """Compute RMSE for each naive benchmark strategy.

    Parameters
    ----------
    benchmarks : pl.DataFrame
        Output of :func:`build_comparative_benchmarks`.

    Returns
    -------
    dict
        ``{benchmark_name: rmse}``.
    """
    out: dict[str, float] = {}
    for name in benchmarks["benchmark"].unique().to_list():
        sub = benchmarks.filter(pl.col("benchmark") == name)
        se = sub["squared_error"].to_numpy().astype(float)
        out[name] = float(np.sqrt(np.mean(se)))
    return out


# ── Checkpoint I/O ───────────────────────────────────────────────────────

def _load_checkpoint(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    df = pl.read_parquet(path)
    return df.to_dicts()


def _save_checkpoint(results: list[dict], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(results, schema=_CHECKPOINT_SCHEMA).write_parquet(path)


# ── Console output ───────────────────────────────────────────────────────

def _print_run_summary(r: dict) -> None:
    print(
        f"  Posterior: mean={r['posterior_mean']:+,.0f}K  "
        f"std={r['posterior_std']:,.0f}K  "
        f"90% HDI=[{r['hdi_5']:+,.0f}, {r['hdi_95']:+,.0f}]"
    )
    print(
        f"  Actual: {r['actual']:+,.0f}K  "
        f"Error: {r['error']:+,.0f}K"
    )


def _print_backtest_summary(df: pl.DataFrame) -> None:
    """Print the full RMSE-by-horizon table to console."""
    metrics = compute_backtest_metrics(df)
    years = sorted(df["march_year"].unique().to_list())

    print(f"\n{'=' * 80}")
    print(f"BENCHMARK BACKTEST SUMMARY — years {years}")
    print(f"{'=' * 80}")
    print(
        f"{'Horizon':>8}  {'N':>3}  {'Bias':>8}  {'MAE':>8}  "
        f"{'RMSE':>8}  {'Cov90':>6}  {'Interval':>10}"
    )
    print("-" * 80)
    for row in metrics.iter_rows(named=True):
        print(
            f"{row['horizon']:>8}  {row['n_years']:>3}  "
            f"{row['bias']:>+8.0f}  {row['mae']:>8.0f}  "
            f"{row['rmse']:>8.0f}  {row['coverage_90']:>6.1%}  "
            f"{row['avg_interval_width']:>10.0f}"
        )

    benchmarks = build_comparative_benchmarks(years)
    comp_rmse = comparative_rmse(benchmarks)
    print("-" * 80)
    for name, rmse in comp_rmse.items():
        print(f"  Benchmark {name}: RMSE = {rmse:,.0f}K")
    for name, rmse in LITERATURE_BENCHMARKS.items():
        print(f"  Literature {name}: RMSE = {rmse:,.0f}K")
    print("=" * 80)
