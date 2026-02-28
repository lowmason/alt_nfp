# ---------------------------------------------------------------------------
# alt_nfp.panel_adapter — Panel to model data dict
# ---------------------------------------------------------------------------
"""Convert a PANEL_SCHEMA observation panel into the dict consumed by
build_model(), diagnostics, and plots.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import polars as pl

from .config import (
    BD_QCEW_LAG,
    CYCLICAL_INDICATORS,
    DATA_DIR,
    ERA_BREAKS,
    INDICATORS_DIR,
    PP_COLORS,
    ProviderConfig,
)
from .ingest.payroll import load_provider_series

# Provider data is typically available ~3 weeks after the reference period.
_PROVIDER_PUBLICATION_LAG_WEEKS = 3

_CYCLICAL_PUBLICATION_LAGS: dict[str, int] = {
    "claims": 1,      # Weekly initial jobless claims — ~1 week lag, rounded to 1 month
    "nfci": 1,         # Chicago Fed NFCI — ~1 week lag, rounded to 1 month
    "biz_apps": 2,     # Census Business Formation Statistics — ~2 month lag
    "jolts": 2,        # BLS JOLTS job openings — ~2 month lag
}


def _date_to_era(d: date) -> int:
    """Map a date to its era index using :data:`ERA_BREAKS`."""
    for i, brk in enumerate(ERA_BREAKS):
        if d < brk:
            return i
    return len(ERA_BREAKS)


def _offset_month(d: date, months: int) -> date:
    """Add *months* to a date, returning the 1st of the resulting month."""
    total = d.month + months
    year = d.year + (total - 1) // 12
    month = ((total - 1) % 12) + 1
    return date(year, month, 1)


def panel_to_model_data(
    panel: pl.DataFrame,
    providers: list[ProviderConfig],
    censor_ces_from: date | None = None,
    as_of: date | None = None,
    *,
    geographic_code: str = "US",
    industry_code: str = "00",
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
        Ignored when *as_of* is provided.
    as_of : date, optional
        Universal censoring cutoff.  When set, observations whose
        ``vintage_date`` exceeds *as_of* are dropped before growth-rate
        extraction, and cyclical indicators are masked using their
        respective publication lags.  Supersedes *censor_ces_from*.
    geographic_code : str
        Geographic code to use for the main series (default 'US'; use '00' for
        vintage store convention).
    industry_code : str
        Industry code for the main series (default '00' = total nonfarm, all
        ownerships).  This is the canonical code used by the vintage store.

    Returns
    -------
    dict
        Dict consumed by :func:`alt_nfp.model.build_model` and downstream
        diagnostics/plotting functions.
    """
    if as_of is not None and "vintage_date" in panel.columns:
        panel = panel.filter(
            pl.col("vintage_date").is_null() | (pl.col("vintage_date") <= as_of)
        )
    elif as_of is not None:
        warnings.warn(
            "Panel lacks vintage_date column; as_of censoring is incomplete.",
            stacklevel=2,
        )

    # Restrict to national scope and chosen industry (no fallback 05→00 so we
    # stay private-only when matching legacy; vintage store often has '00' only)
    geo_filter = pl.col("geographic_code").is_in([geographic_code, "00", "US"])
    national = panel.filter(
        (pl.col("geographic_type") == "national") & geo_filter & (pl.col("industry_code") == industry_code)
    )
    if len(national) == 0:
        raise ValueError(
            f"No national observations for industry_code={industry_code!r}; "
            "panel may be empty or use a different industry_code."
        )

    # Unique sorted periods = model calendar
    dates = sorted(national["period"].unique().to_list())
    T = len(dates)
    month_of_year = np.array([d.month - 1 for d in dates], dtype=int)
    year0 = dates[0].year
    year_of_obs = np.array([d.year - year0 for d in dates], dtype=int)
    n_years = int(year_of_obs.max()) + 1
    era_idx = np.array([_date_to_era(d) for d in dates], dtype=int)

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

    if censor_ces_from is not None and as_of is None:
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
            # Censor birth rate data not yet published as of the as_of date.
            # Provider data is available ~3 weeks after the reference period.
            if as_of is not None:
                lag = timedelta(weeks=_PROVIDER_PUBLICATION_LAG_WEEKS)
                for i, d in enumerate(dates):
                    if d + lag > as_of:
                        births_arr[i:] = np.nan
                        break
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

    if as_of is not None:
        for spec in CYCLICAL_INDICATORS:
            key = f"{spec['name']}_c"
            arr = cyclical.get(key)
            if arr is None:
                continue
            lag = _CYCLICAL_PUBLICATION_LAGS.get(spec["name"], 1)
            for i, d in enumerate(dates):
                if _offset_month(d, lag) > as_of:
                    arr[i:] = 0.0
                    break

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
    for spec in CYCLICAL_INDICATORS:
        key = f"{spec['name']}_c"
        arr = cyclical.get(key)
        if arr is not None:
            n_nonzero = int(np.sum(arr != 0.0))
            print(f"  Cyclical {key}: {n_nonzero} obs")
        else:
            print(f"  Cyclical {key}: not available")
    print(f"  CES SA  mean growth: {np.nanmean(g_ces_sa) * 100:+.4f}%/mo")
    print(f"  CES NSA mean growth: {np.nanmean(g_ces_nsa) * 100:+.4f}%/mo")
    qcew_mean = float(np.nanmean(g_qcew)) if np.any(np.isfinite(g_qcew)) else np.nan
    print(f"  QCEW    mean growth: {qcew_mean * 100:+.4f}%/mo" if np.isfinite(qcew_mean) else "  QCEW    mean growth: (no obs)")
    for pp in pp_data:
        print(f"  {pp['name']:5} mean growth: {np.nanmean(pp['g_pp']) * 100:+.4f}%/mo")

    return dict(
        panel=panel,
        levels=levels_df,
        dates=dates,
        T=T,
        month_of_year=month_of_year,
        year_of_obs=year_of_obs,
        n_years=n_years,
        era_idx=era_idx,
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


def build_obs_sources(data: dict) -> dict:
    """Build ``{var_name: (label, observed_array)}`` used by predictive checks."""
    sources: dict[str, tuple[str, np.ndarray]] = {}

    vintage_labels = ['1st print', '2nd print', 'Final']
    for v in range(3):
        g_sa_v = data['g_ces_sa_by_vintage'][v]
        obs_v = np.where(np.isfinite(g_sa_v))[0]
        if len(obs_v) > 0:
            sources[f'obs_ces_sa_v{v + 1}'] = (
                f'CES SA ({vintage_labels[v]})', g_sa_v[obs_v]
            )
        g_nsa_v = data['g_ces_nsa_by_vintage'][v]
        obs_v_nsa = np.where(np.isfinite(g_nsa_v))[0]
        if len(obs_v_nsa) > 0:
            sources[f'obs_ces_nsa_v{v + 1}'] = (
                f'CES NSA ({vintage_labels[v]})', g_nsa_v[obs_v_nsa]
            )

    sources["obs_qcew"] = ("QCEW", data["g_qcew"][data["qcew_obs"]])
    for pp in data["pp_data"]:
        name = pp["config"].name.lower()
        sources[f"obs_{name}"] = (pp["name"], pp["g_pp"][pp["pp_obs"]])
    return sources


def _load_cyclical_indicators(dates: list, T: int) -> dict:
    """Load cyclical indicators from parquet, align to model dates, and centre.

    Reads from ``data/indicators/<name>.parquet`` (schema: ``ref_date``,
    ``value``).  Weekly series are aggregated to monthly means before
    centering.

    Returns a dict with keys like ``'claims_c'``, ``'nfci_c'``,
    ``'biz_apps_c'``, ``'jolts_c'`` mapping to centred numpy arrays of
    length *T*.  Missing files are gracefully skipped (value set to
    ``None``).
    """
    result: dict = {}

    for spec in CYCLICAL_INDICATORS:
        key = f"{spec['name']}_c"
        fpath = INDICATORS_DIR / f"{spec['name']}.parquet"

        if not fpath.exists():
            result[key] = None
            continue

        try:
            raw = pl.read_parquet(fpath).sort('ref_date')
        except Exception:
            result[key] = None
            continue

        if 'value' not in raw.columns:
            result[key] = None
            continue

        if spec['freq'] == 'weekly':
            raw = raw.with_columns(
                pl.col('ref_date').dt.truncate('1mo').alias('month')
            )
            monthly = raw.group_by('month').agg(
                pl.col('value').mean().alias('value')
            ).sort('month').rename({'month': 'ref_date'})
        else:
            monthly = raw.select(['ref_date', 'value'])

        cal = pl.DataFrame({'ref_date': dates}).with_columns(
            pl.col('ref_date').dt.truncate('1mo').alias('month')
        )
        monthly = monthly.with_columns(
            pl.col('ref_date').dt.truncate('1mo').alias('month')
        )
        joined = cal.join(monthly.select(['month', 'value']), on='month', how='left')
        arr = joined['value'].to_numpy().astype(float)

        if np.any(np.isfinite(arr)):
            mean_val = float(np.nanmean(arr))
            arr_c = np.where(np.isfinite(arr), arr - mean_val, 0.0)
            result[key] = arr_c
        else:
            result[key] = None

    return result


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
