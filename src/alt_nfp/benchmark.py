"""Benchmark revision extraction from existing posteriors.

Phase 1 of the benchmark prediction system (see ``specs/benchmark_spec.md``).
Extracts implied CES benchmark revisions as a deterministic transformation
of the posterior over ``g_total_nsa`` — no additional MCMC needed.

The benchmark revision is the difference between QCEW-anchored "true"
employment and CES-reported employment, accumulated over the 12-month
benchmark window (April Y-1 through March Y).
"""

from __future__ import annotations

import warnings
from datetime import date

import arviz as az
import numpy as np
import polars as pl

from .lookups.benchmark_revisions import BENCHMARK_REVISIONS  # noqa: F401 — re-exported


def _find_benchmark_window(
    dates: list[date],
    march_year: int,
) -> np.ndarray:
    """Return indices into *dates* for April Y-1 through March Y.

    The benchmark window contains 12 monthly observations.

    Raises
    ------
    ValueError
        If the window is not fully contained within *dates*.
    """
    start = date(march_year - 1, 4, 1)  # April Y-1
    end = date(march_year, 3, 31)  # end of March Y

    indices = []
    for i, d in enumerate(dates):
        if start <= d <= end:
            indices.append(i)

    if len(indices) != 12:
        date_min = dates[0] if dates else None
        date_max = dates[-1] if dates else None
        raise ValueError(
            f"Benchmark window April {march_year - 1} – March {march_year} "
            f"requires 12 months but found {len(indices)} within estimation "
            f"period ({date_min} to {date_max})."
        )

    return np.array(indices, dtype=int)


def _get_anchor_level(data: dict, march_year: int) -> float:
    """Look up CES NSA employment level at March Y-1 (thousands).

    Tries two sources in order:

    1. ``data['panel']`` — if a panel DataFrame is present, extract the
       ``employment_level`` for CES NSA at March Y-1.  This is the
       preferred path when data comes from the vintage store.
    2. ``data['levels']`` — fall back to the levels DataFrame from
       ``panel_to_model_data`` (base-100 reconstructed indices).

    Raises
    ------
    ValueError
        If March Y-1 level cannot be found from either source.
    """
    target_year = march_year - 1
    target_month = 3

    # Path 1: panel with raw employment levels (vintage store)
    panel = data.get("panel")
    if panel is not None:
        ces_nsa = panel.filter(
            (pl.col("source") == "ces_nsa")
            & (pl.col("period").dt.year() == target_year)
            & (pl.col("period").dt.month() == target_month)
            & (pl.col("industry_code").is_in(["00", "05"]))
            & (pl.col("geographic_type") == "national")
        )
        if not ces_nsa.is_empty():
            # Prefer benchmarked (is_final) vintage; fall back to latest
            final = ces_nsa.filter(pl.col("is_final") == True)  # noqa: E712
            row = final if not final.is_empty() else ces_nsa
            val = row.sort("vintage_date", descending=True)["employment_level"][0]
            if val is not None and np.isfinite(val):
                return float(val)

    # Path 2: levels DataFrame
    levels = data.get("levels")
    if levels is not None:
        mask = levels.filter(
            (pl.col("ref_date").dt.year() == target_year)
            & (pl.col("ref_date").dt.month() == target_month)
        )
        if not mask.is_empty():
            val = mask["ces_nsa_index"][0]
            if val is not None and np.isfinite(val):
                return float(val)

    raise ValueError(
        f"Cannot find CES NSA employment level at March {target_year} "
        f"for benchmark year {march_year}. Provide a panel with "
        f"employment_level in data['panel'], or pass anchor_level directly."
    )


def load_anchor_level_from_vintage_store(
    march_year: int,
    industry_code: str = "00",
) -> float:
    """Load CES NSA employment level at March Y-1 from the vintage store.

    Convenience function for the diagnostic script.  Returns the
    post-benchmark (final) employment level in thousands.

    Parameters
    ----------
    march_year : int
        Benchmark reference year (anchor is March Y-1).
    industry_code : str
        Industry code (default '00' = total nonfarm, all ownerships).

    Returns
    -------
    float
        Employment level in thousands.
    """
    from .ingest.vintage_store import read_vintage_store

    target_year = march_year - 1

    lf = read_vintage_store(
        source="ces",
        seasonally_adjusted=False,
        geographic_type="national",
        industry_code=industry_code,
    )
    df = (
        lf.filter(
            (pl.col("ref_date").dt.year() == target_year) & (pl.col("ref_date").dt.month() == 3)
        )
        .sort("vintage_date", descending=True)
        .collect()
    )

    if df.is_empty():
        raise ValueError(
            f"No CES NSA data for March {target_year} "
            f"(industry_code={industry_code!r}) in vintage store."
        )

    # Prefer benchmarked version (benchmark_revision > 0)
    benchmarked = df.filter(pl.col("benchmark_revision") > 0)
    row = benchmarked if not benchmarked.is_empty() else df
    return float(row["employment"][0])


def extract_benchmark_revision(
    idata: az.InferenceData,
    data: dict,
    march_year: int,
    *,
    anchor_level: float | None = None,
) -> np.ndarray:
    """Extract posterior samples of the implied benchmark revision.

    Follows spec §3.1: the revision is a deterministic transformation
    of the existing posterior over ``g_total_nsa``.

    Parameters
    ----------
    idata : az.InferenceData
        Posterior inference data from the fitted model.
    data : dict
        The model data dict (from ``panel_to_model_data()``), containing
        model dates, CES NSA growth rates, and level information.
    march_year : int
        The March reference year for the benchmark (e.g. 2025).
    anchor_level : float, optional
        CES NSA employment level at March Y-1 in thousands.  If not
        provided, looked up from ``data['panel']`` or ``data['levels']``.

    Returns
    -------
    np.ndarray
        1-D array of posterior samples of the implied revision in
        thousands of jobs.  Negative values indicate CES overstated
        employment.  Length = n_chains * n_draws.
    """
    if march_year in (2020, 2021):
        warnings.warn(
            f"Benchmark year {march_year} is a COVID year. " "Results may not be meaningful.",
            stacklevel=2,
        )

    # 1. Identify the benchmark window
    window_idx = _find_benchmark_window(data["dates"], march_year)

    # 2. Cumulate posterior latent growth over the window
    #    g_total_nsa shape: (chains, draws, T)
    g_total_nsa = idata.posterior["g_total_nsa"].values
    cum_latent = g_total_nsa[:, :, window_idx].sum(axis=2)  # (chains, draws)

    # 3. Cumulate observed CES NSA growth over the same window
    cum_ces = np.nansum(data["g_ces_nsa"][window_idx])  # scalar

    # 4. Convert to level-space revision
    #    L is CES NSA employment at March Y-1 in thousands (BLS convention).
    #    Result is in thousands of jobs.
    L = anchor_level if anchor_level is not None else _get_anchor_level(data, march_year)
    revision = L * (np.exp(cum_latent) - np.exp(cum_ces))  # (chains, draws)

    # 5. Flatten to 1-D
    return revision.reshape(-1)


def decompose_benchmark_revision(
    idata: az.InferenceData,
    data: dict,
    march_year: int,
    *,
    anchor_level: float | None = None,
) -> dict[str, np.ndarray]:
    """Decompose revision into continuing-units divergence and BD accumulation.

    The decomposition separates "CES sample is off on continuing-units
    growth" from "CES birth/death model is wrong."  Per the spec, the BD
    term should dominate; large continuing-units divergence may signal
    provider representativeness issues.

    Parameters
    ----------
    idata : az.InferenceData
        Posterior inference data from the fitted model.
    data : dict
        The loaded data dict.
    march_year : int
        The March reference year for the benchmark.
    anchor_level : float, optional
        CES NSA employment level at March Y-1 in thousands.  If not
        provided, looked up from ``data['panel']`` or ``data['levels']``.

    Returns
    -------
    dict
        Keys: ``'total'``, ``'cont_divergence'``, ``'bd_accumulation'``.
        Each value is a 1-D posterior sample array in thousands of jobs.
    """
    window_idx = _find_benchmark_window(data["dates"], march_year)
    L = anchor_level if anchor_level is not None else _get_anchor_level(data, march_year)

    # Posterior components over the benchmark window
    g_cont = idata.posterior["g_cont"].values[:, :, window_idx]  # (C, D, 12)
    seasonal = idata.posterior["seasonal"].values[:, :, window_idx]
    bd = idata.posterior["bd"].values[:, :, window_idx]

    # Continuing-units NSA growth = g_cont + seasonal
    cum_cont_nsa = (g_cont + seasonal).sum(axis=2)  # (C, D)

    # CES NSA growth (scalar)
    cum_ces = np.nansum(data["g_ces_nsa"][window_idx])

    # BD accumulation
    cum_bd = bd.sum(axis=2)  # (C, D)

    # Decomposition in level-space (thousands)
    cont_divergence = L * (np.exp(cum_cont_nsa) - np.exp(cum_ces))
    bd_accumulation = L * (np.exp(cum_bd) - 1.0)

    # Total (same as extract_benchmark_revision)
    g_total_nsa = idata.posterior["g_total_nsa"].values[:, :, window_idx]
    cum_latent = g_total_nsa.sum(axis=2)
    total = L * (np.exp(cum_latent) - np.exp(cum_ces))

    return {
        "total": total.reshape(-1),
        "cont_divergence": cont_divergence.reshape(-1),
        "bd_accumulation": bd_accumulation.reshape(-1),
    }


def summarize_revision_posterior(
    samples: np.ndarray,
    actual: float | None = None,
) -> dict[str, float]:
    """Compute summary statistics for the revision posterior.

    Parameters
    ----------
    samples : np.ndarray
        1-D array of posterior samples (in thousands of jobs).
    actual : float, optional
        Actual benchmark revision (thousands) for comparison.

    Returns
    -------
    dict
        Keys: ``'mean'``, ``'median'``, ``'std'``, ``'hdi_5'``, ``'hdi_95'``
        (90% HDI bounds).  If *actual* is provided, also ``'actual'`` and
        ``'error'`` (mean − actual).
    """
    hdi = az.hdi(samples, hdi_prob=0.90)

    result: dict[str, float] = {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "std": float(np.std(samples)),
        "hdi_5": float(hdi[0]),
        "hdi_95": float(hdi[1]),
    }

    if actual is not None:
        result["actual"] = actual
        result["error"] = result["mean"] - actual

    return result
