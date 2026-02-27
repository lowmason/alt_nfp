# ---------------------------------------------------------------------------
# alt_nfp.panel_adapter — Panel to model data dict
# ---------------------------------------------------------------------------
"""Convert a PANEL_SCHEMA observation panel into the dict consumed by
build_model(), diagnostics, and plots (same shape as load_data() output).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from .config import (
    BD_QCEW_LAG,
    CYCLICAL_INDICATORS,
    PP_COLORS,
    ProviderConfig,
)
from .data import _load_cyclical_indicators
from .ingest.payroll import load_provider_series


def panel_to_model_data(
    panel: pl.DataFrame,
    providers: list[ProviderConfig],
    censor_ces_from: date | None = None,
    *,
    geographic_code: str = "US",
    industry_code: str = "05",
) -> dict:
    """Convert an observation panel to the model data dict.

    Parameters
    ----------
    panel : pl.DataFrame
        Validated observation panel (PANEL_SCHEMA).
    providers : list[ProviderConfig]
        Provider list (e.g. from config.PROVIDERS).
    censor_ces_from : date, optional
        If set, treat CES SA/NSA as missing from this date onward (for backtests).
    geographic_code : str
        Geographic code to use for the main series (default 'US'; use '00' for
        vintage store convention).
    industry_code : str
        Industry code for the main series (default '05' = total private).
        Legacy path uses private only ('05'); vintage store may have '00' (total
        nonfarm, all ownerships). Use '05' to align with legacy; pass '00' only
        if you explicitly want all-ownerships.

    Returns
    -------
    dict
        Same structure as :func:`alt_nfp.data.load_data` for use with
        :func:`alt_nfp.model.build_model` and downstream steps.
    """
    # Restrict to national scope and chosen industry (no fallback 05→00 so we
    # stay private-only when matching legacy; vintage store often has '00' only)
    geo_filter = pl.col("geographic_code").is_in([geographic_code, "00", "US"])
    national = panel.filter(
        (pl.col("geographic_type") == "national") & geo_filter & (pl.col("industry_code") == industry_code)
    )
    if len(national) == 0:
        raise ValueError(
            f"No national observations for industry_code={industry_code!r}; "
            "panel may be empty or use a different code (e.g. '00' = all ownerships). "
            "Legacy uses '05' (private only)."
        )

    # Unique sorted periods = model calendar
    dates = sorted(national["period"].unique().to_list())
    T = len(dates)
    month_of_year = np.array([d.month - 1 for d in dates], dtype=int)
    year0 = dates[0].year
    year_of_obs = np.array([d.year - year0 for d in dates], dtype=int)
    n_years = int(year_of_obs.max()) + 1

    # Helper: one T-length array per source, filled from panel (final vintage per period)
    def _growth_series(source: str) -> np.ndarray:
        out = np.full(T, np.nan, dtype=float)
        sub = national.filter(pl.col("source") == source)
        if len(sub) == 0:
            return out
        # Prefer is_final; then highest revision (benchmark -1 before 0,1,2)
        by_period = (
            sub.with_columns(
                pl.when(pl.col("revision_number") == -1)
                .then(999)
                .otherwise(pl.col("revision_number"))
                .alias("_rev_sort")
            )
            .sort(pl.col("is_final").fill_null(False), "_rev_sort", descending=[True, True])
            .unique(subset=["period"], keep="first")
        )
        for row in by_period.iter_rows(named=True):
            period = row["period"]
            growth = row["growth"]
            if period in dates and growth is not None and np.isfinite(growth):
                out[dates.index(period)] = float(growth)
        return out

    g_ces_sa = _growth_series("ces_sa")
    g_ces_nsa = _growth_series("ces_nsa")
    g_qcew = _growth_series("qcew")

    if censor_ces_from is not None:
        for i, d in enumerate(dates):
            if d >= censor_ces_from:
                g_ces_sa[i:] = np.nan
                g_ces_nsa[i:] = np.nan
                break

    ces_sa_obs = np.where(np.isfinite(g_ces_sa))[0]
    ces_nsa_obs = np.where(np.isfinite(g_ces_nsa))[0]
    qcew_obs = np.where(np.isfinite(g_qcew))[0]
    qcew_is_m3 = np.array([dates[i].month in (3, 6, 9, 12) for i in qcew_obs])

    # CES by vintage: v1=rev 0, v2=rev 1, v3=rev 2 or -1
    def _vintage_series(source: str, rev_values: tuple[int, ...]) -> np.ndarray:
        out = np.full(T, np.nan, dtype=float)
        sub = national.filter(
            (pl.col("source") == source) & pl.col("revision_number").is_in(list(rev_values))
        )
        if len(sub) == 0:
            return out
        sub = sub.unique(subset=["period"], keep="first")
        for row in sub.iter_rows(named=True):
            period = row["period"]
            growth = row["growth"]
            if period in dates and growth is not None and np.isfinite(growth):
                out[dates.index(period)] = float(growth)
        return out

    g_ces_sa_v1 = _vintage_series("ces_sa", (0,))
    g_ces_sa_v2 = _vintage_series("ces_sa", (1,))
    g_ces_sa_v3 = _vintage_series("ces_sa", (2, -1))
    g_ces_nsa_v1 = _vintage_series("ces_nsa", (0,))
    g_ces_nsa_v2 = _vintage_series("ces_nsa", (1,))
    g_ces_nsa_v3 = _vintage_series("ces_nsa", (2, -1))

    has_ces_vintage = (
        np.any(np.isfinite(g_ces_sa_v1)) or np.any(np.isfinite(g_ces_sa_v2))
        or np.any(np.isfinite(g_ces_nsa_v1)) or np.any(np.isfinite(g_ces_nsa_v2))
    )
    g_ces_sa_by_vintage = [g_ces_sa_v1, g_ces_sa_v2, g_ces_sa_v3]
    g_ces_nsa_by_vintage = [g_ces_nsa_v1, g_ces_nsa_v2, g_ces_nsa_v3]
    if not has_ces_vintage:
        g_ces_sa_by_vintage = [
            np.full(T, np.nan),
            np.full(T, np.nan),
            g_ces_sa.copy(),
        ]
        g_ces_nsa_by_vintage = [
            np.full(T, np.nan),
            np.full(T, np.nan),
            g_ces_nsa.copy(),
        ]

    # Provider data
    pp_data: list[dict] = []
    for p, cfg in enumerate(providers):
        source_name = cfg.name.lower()
        emp_col = f"{cfg.name.lower()}_employment"
        g_pp = _growth_series(source_name)
        if not np.any(np.isfinite(g_pp)):
            g_pp = _growth_series(cfg.name)
        pp_obs = np.where(np.isfinite(g_pp))[0]
        entry: dict = {
            "name": cfg.name,
            "config": cfg,
            "g_pp": g_pp,
            "pp_obs": pp_obs,
            "emp_col": emp_col,
            "color": PP_COLORS[p % len(PP_COLORS)],
        }

        pp_series = load_provider_series(cfg)
        if pp_series is not None and "birth_rate" in pp_series.columns:
            births_df = pp_series.select(["ref_date", "birth_rate"])
            births_joined = pl.DataFrame({"ref_date": dates}).join(
                births_df, on="ref_date", how="left"
            )
            births_arr = births_joined["birth_rate"].to_numpy().astype(float)
            entry["births"] = births_arr
            entry["births_obs"] = np.where(np.isfinite(births_arr))[0]
        else:
            entry["births"] = None
            entry["births_obs"] = None
        pp_data.append(entry)

    # BD covariates
    birth_arrays = [pp["births"] for pp in pp_data if pp["births"] is not None]
    if birth_arrays:
        birth_rate = np.nanmean(np.vstack(birth_arrays), axis=0)
    else:
        birth_rate = np.full(T, np.nan)
    if pp_data:
        g_pp_stack = np.vstack([pp["g_pp"] for pp in pp_data])
        g_pp_avg = np.nanmean(g_pp_stack, axis=0)
    else:
        g_pp_avg = np.full(T, np.nan)
    bd_proxy = g_qcew - g_pp_avg
    bd_qcew_lagged = np.full(T, np.nan)
    for t in range(BD_QCEW_LAG, T):
        if np.isfinite(bd_proxy[t - BD_QCEW_LAG]):
            bd_qcew_lagged[t] = bd_proxy[t - BD_QCEW_LAG]
    birth_rate_mean = (
        float(np.nanmean(birth_rate)) if np.any(np.isfinite(birth_rate)) else 0.0
    )
    bd_qcew_mean = (
        float(np.nanmean(bd_qcew_lagged))
        if np.any(np.isfinite(bd_qcew_lagged))
        else 0.0
    )
    birth_rate_c = np.where(np.isfinite(birth_rate), birth_rate - birth_rate_mean, 0.0)
    bd_qcew_c = np.where(
        np.isfinite(bd_qcew_lagged), bd_qcew_lagged - bd_qcew_mean, 0.0
    )

    cyclical = _load_cyclical_indicators(dates, T)

    # Levels: ref_date + index columns (reconstruct from growth for compatibility)
    levels_df = _build_levels_from_growth(
        dates=dates,
        g_ces_sa=g_ces_sa,
        g_ces_nsa=g_ces_nsa,
        g_qcew=g_qcew,
        pp_data=pp_data,
    )

    n_qcew_m3 = int(qcew_is_m3.sum())
    n_qcew_m12 = len(qcew_obs) - n_qcew_m3
    n_birth = int(np.sum(np.isfinite(birth_rate)))
    n_bd_qcew = int(np.sum(np.isfinite(bd_qcew_lagged)))

    print(f"Growth-rate model (v3): T = {T} months ({dates[0]} → {dates[-1]})")
    print(f"  CES SA:  {len(ces_sa_obs)} monthly obs")
    print(f"  CES NSA: {len(ces_nsa_obs)} monthly obs")
    print(
        f"  QCEW:    {len(qcew_obs)} monthly obs (NSA) — "
        f"{n_qcew_m3} quarter-end (M3), {n_qcew_m12} retrospective UI (M1-2)"
    )
    for pp in pp_data:
        em = pp["config"].error_model
        n_obs = len(pp["pp_obs"])
        bstr = ""
        if pp["births"] is not None:
            bstr = f", births: {len(pp['births_obs'])} obs"
        print(f"  {pp['name']:5}: {n_obs} obs (error: {em}){bstr}")
    print(
        f"  BD covariates: birth_rate {n_birth} obs "
        f"(mean={birth_rate_mean * 100:.3f}%), "
        f"bd_qcew_lagged {n_bd_qcew} obs "
        f"(L={BD_QCEW_LAG}mo, mean={bd_qcew_mean * 100:.4f}%)"
    )
    for key in ["claims_c", "nfci_c", "biz_apps_c"]:
        arr = cyclical.get(key)
        if arr is not None:
            n_obs = int(np.sum(np.isfinite(arr)))
            print(f"  Cyclical {key}: {n_obs} obs")
        else:
            print(f"  Cyclical {key}: not available")
    print(f"  CES SA  mean growth: {np.nanmean(g_ces_sa) * 100:+.4f}%/mo")
    print(f"  CES NSA mean growth: {np.nanmean(g_ces_nsa) * 100:+.4f}%/mo")
    qcew_mean = float(np.nanmean(g_qcew)) if np.any(np.isfinite(g_qcew)) else np.nan
    print(f"  QCEW    mean growth: {qcew_mean * 100:+.4f}%/mo" if np.isfinite(qcew_mean) else "  QCEW    mean growth: (no obs)")
    for pp in pp_data:
        print(f"  {pp['name']:5} mean growth: {np.nanmean(pp['g_pp']) * 100:+.4f}%/mo")

    return dict(
        levels=levels_df,
        dates=dates,
        T=T,
        month_of_year=month_of_year,
        year_of_obs=year_of_obs,
        n_years=n_years,
        g_ces_sa=g_ces_sa,
        ces_sa_obs=ces_sa_obs,
        g_ces_nsa=g_ces_nsa,
        ces_nsa_obs=ces_nsa_obs,
        g_qcew=g_qcew,
        qcew_obs=qcew_obs,
        qcew_is_m3=qcew_is_m3,
        pp_data=pp_data,
        n_providers=len(providers),
        birth_rate=birth_rate,
        birth_rate_mean=birth_rate_mean,
        birth_rate_c=birth_rate_c,
        bd_proxy=bd_proxy,
        bd_qcew_lagged=bd_qcew_lagged,
        bd_qcew_mean=bd_qcew_mean,
        bd_qcew_c=bd_qcew_c,
        g_ces_sa_by_vintage=g_ces_sa_by_vintage,
        g_ces_nsa_by_vintage=g_ces_nsa_by_vintage,
        has_ces_vintage=has_ces_vintage,
        **cyclical,
    )


def _build_levels_from_growth(
    dates: list[date],
    g_ces_sa: np.ndarray,
    g_ces_nsa: np.ndarray,
    g_qcew: np.ndarray,
    pp_data: list[dict],
) -> pl.DataFrame:
    """Build a levels DataFrame from growth arrays (ref_date + index columns)."""
    T = len(dates)
    base = 100.0

    def cum_level(g: np.ndarray) -> np.ndarray:
        out = np.full(T, np.nan, dtype=float)
        log_level = np.nan
        for i in range(T):
            if np.isfinite(g[i]):
                if np.isnan(log_level):
                    log_level = np.log(base)
                log_level = log_level + g[i]
                out[i] = np.exp(log_level)
            elif not np.isnan(log_level):
                out[i] = np.exp(log_level)
        return out

    ces_sa_index = cum_level(g_ces_sa)
    ces_nsa_index = cum_level(g_ces_nsa)
    qcew_nsa_index = cum_level(g_qcew)

    d = {
        "ref_date": dates,
        "ces_sa_index": ces_sa_index,
        "ces_nsa_index": ces_nsa_index,
        "qcew_nsa_index": qcew_nsa_index,
    }
    for pp in pp_data:
        d[pp["emp_col"]] = cum_level(pp["g_pp"])

    return pl.DataFrame(d)
