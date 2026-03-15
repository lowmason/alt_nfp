"""Backtest using the ~Feb 9 v2 model specification against the current vintage store.

Reimplements the model from archive/pp_estimation_v2.py (last modified Feb 11)
and runs it through the same vintage-aware backtest infrastructure as the
current model, enabling an apples-to-apples comparison.

v2 model differences from the current (v3) model:
  - AR(1): Uniform phi, HalfNormal sigma_g (no tau reparameterization)
  - No eras: single scalar mu_g
  - Seasonal: sum-to-zero 11 free parameters (not Fourier GRW)
  - Birth/death: fixed scalar (no cyclical indicators)
  - QCEW: Normal likelihood with fixed sigmas (not Student-t, not estimated)
  - CES: InverseGamma sigma, Normal lambda prior (no truncation, no vintage indexing)
  - Provider: AR(1) measurement error for provider G (mapped from PP1)

Usage:
    python scripts/backtest_v2_model.py
    python scripts/backtest_v2_model.py --n-backtest 12
    python scripts/backtest_v2_model.py --start-date 2024-06-12
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pymc as pm
import pytensor
import pytensor.tensor as pt

from nfp_models.config import OUTPUT_DIR, PROVIDERS
from nfp_ingest import build_panel
from nfp_models.panel_adapter import panel_to_model_data
from nfp_models.sampling import LIGHT_SAMPLER_KWARGS, sample_model

# v2 model constants (from archive/pp_estimation_v2.py)
SIGMA_QCEW_M3 = 0.0005
SIGMA_QCEW_M12 = 0.0015


# =========================================================================
# Data adapter: v3 data dict → v2 data dict
# =========================================================================


def _to_v2_data(data: dict) -> dict:
    """Down-convert a v3 panel_to_model_data() dict to the simpler v2 format.

    Drops era indexing, revision multipliers, vintage-indexed CES sigma,
    and cyclical indicators.  Recomputes qcew_is_m3 (quarter-end months)
    from dates.  Maps the first provider (G) to g_pp / pp_obs.
    """
    dates = data["dates"]
    T = data["T"]
    qcew_obs = data["qcew_obs"]

    qcew_is_m3 = np.array(
        [dates[i].month in (3, 6, 9, 12) for i in qcew_obs], dtype=bool
    )

    # Map provider G → PP1 (AR(1) error)
    pp_data_v3 = data["pp_data"]
    if pp_data_v3:
        pp = pp_data_v3[0]
        g_pp = pp["g_pp"].reshape(1, T)
        pp_obs = [pp["pp_obs"]]
        n_pp = 1
    else:
        g_pp = np.full((1, T), np.nan)
        pp_obs = [np.array([], dtype=int)]
        n_pp = 0

    return {
        "dates": dates,
        "T": T,
        "month_of_year": data["month_of_year"],
        "g_ces_sa": data["g_ces_sa"],
        "ces_sa_obs": data["ces_sa_obs"],
        "g_ces_nsa": data["g_ces_nsa"],
        "ces_nsa_obs": data["ces_nsa_obs"],
        "g_qcew": data["g_qcew"],
        "qcew_obs": qcew_obs,
        "qcew_is_m3": qcew_is_m3,
        "g_pp": g_pp,
        "pp_obs": pp_obs,
        "n_pp": n_pp,
        "levels": data["levels"],
    }


# =========================================================================
# v2 model definition (faithful to archive/pp_estimation_v2.py lines 163-333)
# =========================================================================


def build_v2_model(data: dict) -> pm.Model:
    """Build the ~Feb 9 v2 QCEW-anchored PyMC model."""
    T = data["T"]
    month_of_year = data["month_of_year"]
    g_ces_sa = data["g_ces_sa"]
    ces_sa_obs = data["ces_sa_obs"]
    g_ces_nsa = data["g_ces_nsa"]
    ces_nsa_obs = data["ces_nsa_obs"]
    g_pp = data["g_pp"]
    pp_obs = data["pp_obs"]
    g_qcew = data["g_qcew"]
    qcew_obs = data["qcew_obs"]
    qcew_is_m3 = data["qcew_is_m3"]
    n_pp = data["n_pp"]

    qcew_sigma_fixed = np.where(qcew_is_m3, SIGMA_QCEW_M3, SIGMA_QCEW_M12)

    with pm.Model() as model:

        # AR(1) latent continuing-units growth
        mu_g = pm.Normal("mu_g", mu=0.001, sigma=0.005)
        phi = pm.Uniform("phi", lower=0.0, upper=0.99)
        sigma_g = pm.HalfNormal("sigma_g", sigma=0.008)

        eps_g = pm.Normal("eps_g", 0, 1, shape=T)
        g0 = mu_g + sigma_g * eps_g[0]

        def ar1_step(e_t, g_prev, _mu, _phi, _sig):
            return _mu + _phi * (g_prev - _mu) + _sig * e_t

        g_rest, _ = pytensor.scan(
            fn=ar1_step,
            sequences=[eps_g[1:]],
            outputs_info=[g0],
            non_sequences=[mu_g, phi, sigma_g],
            strict=True,
        )
        g_cont = pt.concatenate([g0.reshape((1,)), g_rest])
        pm.Deterministic("g_cont", g_cont)

        # Sum-to-zero seasonal (11 free parameters)
        s_raw = pm.Normal("s_raw", 0, 0.015, shape=11)
        s_all = pt.concatenate([s_raw, (-pt.sum(s_raw)).reshape((1,))])
        s_t = s_all[month_of_year]
        pm.Deterministic("seasonal", s_all)

        # Scalar birth/death offset
        bd = pm.Normal("bd", mu=0.001, sigma=0.002)

        # Composite growth signals
        g_cont_nsa = g_cont + s_t
        g_total_sa = g_cont + bd
        g_total_nsa = g_cont + s_t + bd

        pm.Deterministic("g_total_sa", g_total_sa)
        pm.Deterministic("g_total_nsa", g_total_nsa)

        # QCEW — truth anchor, Normal with fixed sigma
        pm.Normal(
            "obs_qcew",
            mu=g_total_nsa[qcew_obs],
            sigma=qcew_sigma_fixed,
            observed=g_qcew[qcew_obs],
        )

        # CES — noisy observation with estimated bias/loading/noise
        alpha_ces = pm.Normal("alpha_ces", 0, 0.005)
        lambda_ces = pm.Normal("lambda_ces", 1.0, 0.25)
        sigma_ces_sa = pm.InverseGamma("sigma_ces_sa", alpha=3.0, beta=0.004)
        sigma_ces_nsa = pm.InverseGamma("sigma_ces_nsa", alpha=3.0, beta=0.004)

        pm.Normal(
            "obs_ces_sa",
            mu=alpha_ces + lambda_ces * g_total_sa[ces_sa_obs],
            sigma=sigma_ces_sa,
            observed=g_ces_sa[ces_sa_obs],
        )
        pm.Normal(
            "obs_ces_nsa",
            mu=alpha_ces + lambda_ces * g_total_nsa[ces_nsa_obs],
            sigma=sigma_ces_nsa,
            observed=g_ces_nsa[ces_nsa_obs],
        )

        # Provider G → PP1 with AR(1) measurement error
        if n_pp > 0 and len(pp_obs[0]) > 0:
            alpha_pp = pm.Normal("alpha_pp", 0, 0.005)
            lam_pp = pm.Normal("lam_pp", 1.0, 0.25)
            sigma_pp = pm.InverseGamma("sigma_pp", alpha=3.0, beta=0.004)
            rho_pp = pm.Uniform("rho_pp", lower=0.0, upper=0.99)

            m_pp = pp_obs[0]
            mu_base_pp = alpha_pp + lam_pp * g_cont_nsa[m_pp]
            y_pp = g_pp[0, m_pp]

            mu_cond_pp = pt.concatenate([
                mu_base_pp[:1],
                mu_base_pp[1:] + rho_pp * (
                    pt.as_tensor_variable(y_pp[:-1]) - mu_base_pp[:-1]
                ),
            ])
            sigma_cond_pp = pt.concatenate([
                (sigma_pp / pt.sqrt(1.0 - rho_pp ** 2)).reshape((1,)),
                pt.ones(len(m_pp) - 1) * sigma_pp,
            ])
            pm.Normal(
                "obs_provider",
                mu=mu_cond_pp,
                sigma=sigma_cond_pp,
                observed=y_pp,
            )

    return model


# =========================================================================
# Backtest loop
# =========================================================================


def run_v2_backtest(
    n_backtest: int = 24,
    *,
    start_date: date | None = None,
    output_dir: Path | None = None,
) -> pl.DataFrame:
    """Run the vintage-aware backtest using the v2 model specification."""
    if output_dir is not None:
        run_dir = output_dir
    else:
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M")
        run_dir = OUTPUT_DIR / "backtest_runs" / f"v2_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    panel_full = build_panel()
    data_full = panel_to_model_data(panel_full, PROVIDERS)
    data_full_v2 = _to_v2_data(data_full)

    dates = data_full_v2["dates"]
    T_full = len(dates)
    g_ces_sa_actual = data_full_v2["g_ces_sa"]
    levels = data_full_v2["levels"]
    ces_sa_index = levels["ces_sa_index"].to_numpy().astype(float)
    base_index = float(ces_sa_index[0])

    base_row_idx = int(np.argmin(np.abs(ces_sa_index - 100.0)))
    ces_sa_base_level = levels["ces_sa_level"].to_numpy().astype(float)[base_row_idx]
    idx_to_level = ces_sa_base_level / 100.0

    if start_date is not None:
        first_idx = next(
            (i for i, d in enumerate(dates) if d >= start_date), T_full
        )
        last_idx = min(first_idx + n_backtest, T_full)
        if first_idx >= T_full:
            raise ValueError(
                f"start_date {start_date} is beyond the panel end ({dates[-1]})"
            )
        target_indices = list(range(first_idx, last_idx))
    else:
        if T_full < n_backtest:
            raise ValueError(f"Need at least {n_backtest} months, got T={T_full}")
        target_indices = list(range(T_full - n_backtest, T_full))

    target_dates = [dates[i] for i in target_indices]
    n_runs = len(target_indices)
    results: list[dict] = []

    for run, (t_idx, target_date) in enumerate(zip(target_indices, target_dates, strict=True)):
        print(f"\n--- v2 backtest {run + 1}/{n_runs}: {target_date} ---")

        panel = build_panel(as_of_ref=target_date)
        data_v3 = panel_to_model_data(panel, PROVIDERS, as_of=target_date)
        data = _to_v2_data(data_v3)

        ces_n = len(data["ces_sa_obs"])
        qcew_n = len(data["qcew_obs"])
        pp_n = len(data["pp_obs"][0]) if data["n_pp"] > 0 else 0
        print(f"  CES SA: {ces_n} obs, QCEW: {qcew_n} obs, Provider: {pp_n} obs")

        model = build_v2_model(data)
        idata = sample_model(model, sampler_kwargs=LIGHT_SAMPLER_KWARGS)

        nc_path = run_dir / f"{target_date:%Y-%m}.nc"
        idata.to_netcdf(str(nc_path))
        print(f"  Saved: {nc_path}")

        g_sa_post = idata.posterior["g_total_sa"].values
        alpha_post = idata.posterior["alpha_ces"].values
        lambda_post = idata.posterior["lambda_ces"].values

        g_ces_pred = alpha_post[:, :, None] + lambda_post[:, :, None] * g_sa_post
        g_sa_mean = np.nanmean(g_ces_pred, axis=(0, 1))

        censored_dates = data["dates"]
        if target_date in censored_dates:
            c_idx = censored_dates.index(target_date)
        else:
            c_idx = len(censored_dates) - 1
            print(
                f"  NOTE: Target {target_date} beyond censored horizon "
                f"({censored_dates[-1]}); using last latent state as proxy"
            )

        nowcast_index_series = np.empty(len(g_sa_mean) + 1, dtype=float)
        nowcast_index_series[0] = base_index
        for s in range(len(g_sa_mean)):
            nowcast_index_series[s + 1] = nowcast_index_series[s] * np.exp(g_sa_mean[s])
        nowcast_growth = g_sa_mean[c_idx]
        nowcast_index = nowcast_index_series[c_idx + 1]
        prev_index = nowcast_index_series[c_idx]

        actual_growth = g_ces_sa_actual[t_idx]
        actual_index = ces_sa_index[t_idx]
        prev_actual_index = ces_sa_index[t_idx - 1] if t_idx > 0 else np.nan

        actual_level = actual_index * idx_to_level
        prev_level = prev_actual_index * idx_to_level
        nowcast_level = nowcast_index * idx_to_level
        prev_nowcast_level = prev_index * idx_to_level

        actual_change_k = actual_level - prev_level
        nowcast_change_k = nowcast_level - prev_nowcast_level
        error_change_k = actual_change_k - nowcast_change_k
        err_growth_pp = (nowcast_growth - actual_growth) * 100
        err_level_k = actual_level - nowcast_level

        has_ces = c_idx in data["ces_sa_obs"]
        has_qcew = c_idx in data["qcew_obs"]
        has_provider = (
            data["n_pp"] > 0 and c_idx in data["pp_obs"][0]
        )
        sources = []
        if has_ces:
            sources.append("CES")
        if has_qcew:
            sources.append("QCEW")
        if has_provider:
            sources.append("G")
        sources_str = "+".join(sources) if sources else "none"

        results.append(
            {
                "date": target_date,
                "actual_growth_pct": actual_growth * 100,
                "nowcast_growth_pct": nowcast_growth * 100,
                "error_growth_pp": err_growth_pp,
                "actual_change_k": actual_change_k,
                "nowcast_change_k": nowcast_change_k,
                "error_change_k": error_change_k,
                "actual_level_k": actual_level,
                "nowcast_level_k": nowcast_level,
                "error_level_k": err_level_k,
                "has_ces": has_ces,
                "has_qcew": has_qcew,
                "has_G": has_provider,
                "sources": sources_str,
            }
        )
        print(
            f"  Jobs added (SA): actual {actual_change_k:+,.0f}k  "
            f"nowcast {nowcast_change_k:+,.0f}k  "
            f"error {error_change_k:+,.0f}k  [{sources_str}]"
        )

    results_df = pl.DataFrame(results)
    parquet_path = run_dir / "backtest_results.parquet"
    results_df.write_parquet(parquet_path)
    print(f"\nSaved: {parquet_path}")

    _print_results_table(results_df, n_runs)
    _plot_backtest(results_df, run_dir)

    return results_df


# =========================================================================
# Reporting (adapted from backtest.py)
# =========================================================================


def _print_results_table(results: pl.DataFrame, n_backtest: int) -> None:
    print("\n" + "=" * 100)
    print(
        f"v2 MODEL BACKTEST (vintage-aware): {n_backtest} months "
        f"(jobs added = MoM change, SA)"
    )
    print("=" * 100)
    print(
        f"{'Date':>12}  {'Actual Δ(k)':>11} {'Nowcast Δ(k)':>11} "
        f"{'Error Δ(k)':>10}  {'Actual %':>8} {'Nowcast %':>8} "
        f"{'Error (pp)':>9}  {'Sources':>15}"
    )
    print("-" * 100)
    for row in results.iter_rows(named=True):
        print(
            f'{str(row["date"]):>12}  {row["actual_change_k"]:>+10,.0f}  '
            f'{row["nowcast_change_k"]:>+10,.0f}  '
            f'{row["error_change_k"]:>+9,.0f}  '
            f'{row["actual_growth_pct"]:>+7.3f}  '
            f'{row["nowcast_growth_pct"]:>+7.3f}  '
            f'{row["error_growth_pp"]:>+8.3f}  '
            f'{row["sources"]:>15}'
        )

    errs_gr = results["error_growth_pp"].to_numpy()
    errs_chg = results["error_change_k"].to_numpy()
    mae_gr = float(np.mean(np.abs(errs_gr)))
    rmse_gr = float(np.sqrt(np.mean(errs_gr**2)))
    mae_chg = float(np.mean(np.abs(errs_chg)))
    rmse_chg = float(np.sqrt(np.mean(errs_chg**2)))

    print("-" * 100)
    print(
        f"Overall (n={len(results)}):  MAE jobs added = {mae_chg:,.0f} k   "
        f"RMSE = {rmse_chg:,.0f} k  |  MAE growth = {mae_gr:.3f} pp   "
        f"RMSE growth = {rmse_gr:.3f} pp"
    )


def _plot_backtest(results: pl.DataFrame, run_dir: Path) -> None:
    x_dates = results["date"].to_list()
    actual_chg = results["actual_change_k"].to_numpy()
    nowcast_chg = results["nowcast_change_k"].to_numpy()
    n = len(x_dates)

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax = axes[0]
    bar_width = 10
    x_num = mdates.date2num(x_dates)
    ax.bar(
        x_num - bar_width / 2, actual_chg, width=bar_width,
        color="darkorange", alpha=0.85, label="Actual",
    )
    ax.bar(
        x_num + bar_width / 2, nowcast_chg, width=bar_width,
        color="steelblue", alpha=0.85, label="Nowcast (v2)",
    )
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Jobs added (thousands, SA)")
    ax.set_title("v2 Model: Month-over-Month Jobs Added — Actual vs Real-Time Nowcast")
    ax.legend(loc="upper right")
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", alpha=0.3)

    mae_chg = float(np.mean(np.abs(actual_chg - nowcast_chg)))
    rmse_chg = float(np.sqrt(np.mean((actual_chg - nowcast_chg) ** 2)))
    ax.text(
        0.01, 0.02,
        f"MAE = {mae_chg:,.0f}k    RMSE = {rmse_chg:,.0f}k",
        transform=ax.transAxes, fontsize=9, color="gray",
        verticalalignment="bottom",
    )

    ax = axes[1]
    source_cols = ["has_ces", "has_qcew", "has_G"]
    source_labels = ["CES", "QCEW", "G"]
    n_sources = len(source_cols)
    for s_idx, (col, _label) in enumerate(zip(source_cols, source_labels, strict=True)):
        flags = results[col].to_numpy()
        for t_idx in range(n):
            color = "#2ca02c" if flags[t_idx] else "#d62728"
            marker = "o" if flags[t_idx] else "x"
            ax.plot(
                x_dates[t_idx], s_idx, marker=marker, color=color,
                ms=6, mew=1.5,
            )

    ax.set_yticks(range(n_sources))
    ax.set_yticklabels(source_labels)
    ax.set_ylim(-0.5, n_sources - 0.5)
    ax.invert_yaxis()
    ax.set_title("Data Source Availability at Target Month")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plot_path = run_dir / "nowcast_backtest_v2.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")


# =========================================================================
# CLI
# =========================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest using the ~Feb 9 v2 model specification"
    )
    parser.add_argument(
        "--n-backtest", type=int, default=24,
        help="Number of months to backtest (default: 24)",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="First target month (YYYY-MM-DD), e.g. 2024-06-12",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for run artifacts",
    )
    args = parser.parse_args()

    sd = None
    if args.start_date:
        sd = date.fromisoformat(args.start_date)
    od = Path(args.output_dir) if args.output_dir else None

    run_v2_backtest(n_backtest=args.n_backtest, start_date=sd, output_dir=od)
