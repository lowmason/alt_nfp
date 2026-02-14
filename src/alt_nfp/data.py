# ---------------------------------------------------------------------------
# alt_nfp.data — Data loading and alignment
# ---------------------------------------------------------------------------
"""Load CES, QCEW, and payroll-provider data; compute growth rates and
birth/death covariates for the structural BD model."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from .config import (
    BD_QCEW_LAG,
    DATA_DIR,
    PP_COLORS,
    PROVIDERS,
    ProviderConfig,
)


def load_data(
    providers: list[ProviderConfig] | None = None,
    censor_ces_from: date | None = None,
) -> dict:
    """Load and align all data sources; compute growth rates and BD covariates.

    Parameters
    ----------
    providers : list[ProviderConfig], optional
        Provider list.  Defaults to :pydata:`PROVIDERS` from config.
    censor_ces_from : date, optional
        If set, treat CES SA and NSA observations as missing from this date
        onward.  Used by the nowcast backtest to simulate real-time
        information sets.

    Returns
    -------
    dict
        Keyed arrays consumed by :func:`alt_nfp.model.build_model` and the
        plotting / diagnostics functions.
    """
    if providers is None:
        providers = PROVIDERS

    # ------------------------------------------------------------------
    # CES and QCEW (fixed data sources)
    # ------------------------------------------------------------------
    ces = pl.read_csv(str(DATA_DIR / "ces_index.csv"), try_parse_dates=True).sort("ref_date")
    qcew = pl.read_csv(str(DATA_DIR / "qcew_index.csv"), try_parse_dates=True).sort("ref_date")

    # Monthly calendar spanning CES range
    cal = pl.DataFrame(
        {
            "ref_date": pl.date_range(
                ces["ref_date"].min(),
                ces["ref_date"].max(),
                interval="1mo",
                eager=True,
            )
        }
    )

    # Start building *levels* by joining CES and QCEW
    levels = cal.join(ces, on="ref_date", how="left").join(qcew, on="ref_date", how="left")

    # ------------------------------------------------------------------
    # Load each provider's index and join onto calendar
    # ------------------------------------------------------------------
    for cfg in providers:
        pp_df = (
            pl.read_csv(str(DATA_DIR / cfg.file), try_parse_dates=True)
            .sort("ref_date")
            .select(["ref_date", cfg.index_col])
        )
        levels = levels.join(pp_df, on="ref_date", how="left")

    levels = levels.sort("ref_date")

    # ------------------------------------------------------------------
    # Growth rates (first-difference of log index)
    # ------------------------------------------------------------------
    growth_exprs = [
        pl.col("ces_sa_index").log().diff().alias("g_ces_sa"),
        pl.col("ces_nsa_index").log().diff().alias("g_ces_nsa"),
        pl.col("qcew_nsa_index").log().diff().alias("g_qcew"),
    ]
    for cfg in providers:
        growth_exprs.append(
            pl.col(cfg.index_col).log().diff().alias(f"g_{cfg.name.lower()}")
        )

    growth = levels.with_columns(growth_exprs).slice(1)  # first row has no growth

    dates = growth["ref_date"].to_list()
    T = len(dates)
    month_of_year = np.array([d.month - 1 for d in dates], dtype=int)

    # CES
    g_ces_sa = growth["g_ces_sa"].to_numpy().astype(float)
    g_ces_nsa = growth["g_ces_nsa"].to_numpy().astype(float)

    # Apply CES censoring for backtest experiments
    if censor_ces_from is not None:
        g_ces_sa = g_ces_sa.copy()
        g_ces_nsa = g_ces_nsa.copy()
        for i, d in enumerate(dates):
            if d >= censor_ces_from:
                g_ces_sa[i:] = np.nan
                g_ces_nsa[i:] = np.nan
                break

    ces_sa_obs = np.where(np.isfinite(g_ces_sa))[0]
    ces_nsa_obs = np.where(np.isfinite(g_ces_nsa))[0]

    # QCEW
    g_qcew = growth["g_qcew"].to_numpy().astype(float)
    qcew_obs = np.where(np.isfinite(g_qcew))[0]
    qcew_is_m3 = np.array([dates[i].month in (3, 6, 9, 12) for i in qcew_obs])

    # ------------------------------------------------------------------
    # Per-provider data
    # ------------------------------------------------------------------
    pp_data: list[dict] = []
    for p, cfg in enumerate(providers):
        g_col = f"g_{cfg.name.lower()}"
        g_pp_p = growth[g_col].to_numpy().astype(float)
        pp_obs_p = np.where(np.isfinite(g_pp_p))[0]

        entry: dict = {
            "name": cfg.name,
            "config": cfg,
            "g_pp": g_pp_p,
            "pp_obs": pp_obs_p,
            "index_col": cfg.index_col,
            "color": PP_COLORS[p % len(PP_COLORS)],
        }

        # Load birth-rate data if specified
        if cfg.births_file and cfg.births_col:
            births_df = (
                pl.read_csv(str(DATA_DIR / cfg.births_file), try_parse_dates=True)
                .sort("ref_date")
                .select(["ref_date", cfg.births_col])
            )
            births_joined = pl.DataFrame({"ref_date": dates}).join(
                births_df, on="ref_date", how="left"
            )
            births_arr = births_joined[cfg.births_col].to_numpy().astype(float)
            entry["births"] = births_arr
            entry["births_obs"] = np.where(np.isfinite(births_arr))[0]
        else:
            entry["births"] = None
            entry["births_obs"] = None

        pp_data.append(entry)

    # ------------------------------------------------------------------
    # BD covariates for structural birth/death model
    # ------------------------------------------------------------------

    # Composite birth rate: average across providers that report it
    birth_arrays = [pp["births"] for pp in pp_data if pp["births"] is not None]
    if birth_arrays:
        birth_rate = np.nanmean(np.vstack(birth_arrays), axis=0)
    else:
        birth_rate = np.full(T, np.nan)

    # BD QCEW proxy:  g_qcew_nsa − mean(g_pp)  ≈  BD contribution
    g_pp_stack = np.vstack([pp["g_pp"] for pp in pp_data])
    g_pp_avg = np.nanmean(g_pp_stack, axis=0)
    bd_proxy = g_qcew - g_pp_avg  # NaN where either source missing

    # Lag by L months (QCEW publication delay)
    bd_qcew_lagged = np.full(T, np.nan)
    for t in range(BD_QCEW_LAG, T):
        if np.isfinite(bd_proxy[t - BD_QCEW_LAG]):
            bd_qcew_lagged[t] = bd_proxy[t - BD_QCEW_LAG]

    # Centre covariates so that φ_0 = mean BD at average covariate values
    birth_rate_mean = float(np.nanmean(birth_rate)) if np.any(np.isfinite(birth_rate)) else 0.0
    bd_qcew_mean = (
        float(np.nanmean(bd_qcew_lagged)) if np.any(np.isfinite(bd_qcew_lagged)) else 0.0
    )

    birth_rate_c = np.where(np.isfinite(birth_rate), birth_rate - birth_rate_mean, 0.0)
    bd_qcew_c = np.where(np.isfinite(bd_qcew_lagged), bd_qcew_lagged - bd_qcew_mean, 0.0)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_qcew_m3 = int(qcew_is_m3.sum())
    n_qcew_m12 = len(qcew_obs) - n_qcew_m3
    n_birth = int(np.sum(np.isfinite(birth_rate)))
    n_bd_qcew = int(np.sum(np.isfinite(bd_qcew_lagged)))

    print(f"Growth-rate model (v3): T = {T} months ({dates[0]} \u2192 {dates[-1]})")
    print(f"  CES SA:  {len(ces_sa_obs)} monthly obs")
    print(f"  CES NSA: {len(ces_nsa_obs)} monthly obs")
    print(
        f"  QCEW:    {len(qcew_obs)} monthly obs (NSA) \u2014 "
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
    print(f"  CES SA  mean growth: {np.nanmean(g_ces_sa) * 100:+.4f}%/mo")
    print(f"  CES NSA mean growth: {np.nanmean(g_ces_nsa) * 100:+.4f}%/mo")
    print(f"  QCEW    mean growth: {np.nanmean(g_qcew) * 100:+.4f}%/mo")
    for pp in pp_data:
        print(f"  {pp['name']:5} mean growth: {np.nanmean(pp['g_pp']) * 100:+.4f}%/mo")

    return dict(
        levels=levels,
        dates=dates,
        T=T,
        month_of_year=month_of_year,
        g_ces_sa=g_ces_sa,
        ces_sa_obs=ces_sa_obs,
        g_ces_nsa=g_ces_nsa,
        ces_nsa_obs=ces_nsa_obs,
        g_qcew=g_qcew,
        qcew_obs=qcew_obs,
        qcew_is_m3=qcew_is_m3,
        pp_data=pp_data,
        n_providers=len(providers),
        # BD covariates
        birth_rate=birth_rate,
        birth_rate_mean=birth_rate_mean,
        birth_rate_c=birth_rate_c,
        bd_proxy=bd_proxy,
        bd_qcew_lagged=bd_qcew_lagged,
        bd_qcew_mean=bd_qcew_mean,
        bd_qcew_c=bd_qcew_c,
    )


def build_obs_sources(data: dict) -> dict:
    """Build ``{var_name: (label, observed_array)}`` used by predictive checks."""
    sources: dict[str, tuple[str, np.ndarray]] = {
        "obs_ces_sa": ("CES SA", data["g_ces_sa"][data["ces_sa_obs"]]),
        "obs_ces_nsa": ("CES NSA", data["g_ces_nsa"][data["ces_nsa_obs"]]),
        "obs_qcew": ("QCEW", data["g_qcew"][data["qcew_obs"]]),
    }
    for pp in data["pp_data"]:
        name = pp["config"].name.lower()
        sources[f"obs_{name}"] = (pp["name"], pp["g_pp"][pp["pp_obs"]])
    return sources
