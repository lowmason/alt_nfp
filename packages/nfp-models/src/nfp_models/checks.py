"""Prior/posterior predictive checks and LOO-CV diagnostics.

Implements the predictive-check steps of the Bayesian workflow
(Gabry et al. 2019):

* :func:`run_prior_predictive_checks` — validates that priors produce
  data in a plausible range before fitting.
* :func:`run_posterior_predictive_checks` — density overlays and
  test-statistic comparisons (skewness, lag-1 ACF) between replicated
  and observed data.
* :func:`run_loo_cv` — Pareto-smoothed importance-sampling LOO
  cross-validation with per-source k-hat diagnostics.
"""

from __future__ import annotations

import warnings

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy import stats as sp_stats

from .config import OUTPUT_DIR
from .panel_adapter import build_obs_sources
from .settings import NowcastConfig


# =========================================================================
# Prior predictive
# =========================================================================


def run_prior_predictive_checks(
    model: pm.Model, data: dict, cfg: NowcastConfig | None = None,
) -> az.InferenceData | None:
    """Sample from the prior predictive and visualise.

    Validates that priors produce data in a plausible range before
    fitting (Gabry et al. 2019, Section 3).
    """
    print("Sampling prior predictive\u2026")
    try:
        with model:
            prior_idata = pm.sample_prior_predictive(samples=500)
    except Exception as e:
        print(f"Prior predictive sampling failed: {e}")
        return None

    dates = data["dates"]
    obs_sources = build_obs_sources(data)
    n_sources = len(obs_sources)

    n_cols = 3
    n_rows = (n_sources + 1 + n_cols - 1) // n_cols  # +1 for g_cont panel
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes_flat = axes.flatten()

    for idx, (vn, (label, obs)) in enumerate(obs_sources.items()):
        ax = axes_flat[idx]
        pp = prior_idata.prior_predictive[vn].values.flatten()
        lo, hi = np.percentile(pp, [1, 99])
        pp_clip = pp[(pp >= lo) & (pp <= hi)]

        ax.hist(pp_clip * 100, bins=80, density=True, alpha=0.4, color="steelblue",
                label="Prior predictive")
        ax.hist(obs * 100, bins=40, density=True, alpha=0.6, color="darkorange",
                label="Observed")
        ax.set_xlabel("Monthly growth (%)")
        ax.set_title(label)
        ax.legend(fontsize=7)

    # Last content panel: prior g_cont trajectories
    ax = axes_flat[n_sources]
    g_prior = prior_idata.prior["g_cont"].values
    n_show = min(20, g_prior.shape[1])
    for i in range(n_show):
        ax.plot(dates, g_prior[0, i, :] * 100, alpha=0.3, lw=0.8, color="steelblue")
    ax.plot(dates, data["g_ces_sa"] * 100, "darkorange", lw=1.5, alpha=0.8,
            label="CES SA (observed)")
    ax.set_ylabel("Growth (%/mo)")
    ax.set_title("Prior draws: latent g_cont")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for idx in range(n_sources + 1, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Prior Predictive Check: Do priors generate plausible data?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "prior_predictive.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'prior_predictive.png'}")

    # Summary statistics
    print("\nPrior predictive summary (growth %, monthly):")
    for vn, (label, obs) in obs_sources.items():
        pp = prior_idata.prior_predictive[vn].values.flatten() * 100
        lo5, hi95 = np.percentile(pp, [5, 95])
        print(f"  {label:10}: prior 90% [{lo5:+.2f}, {hi95:+.2f}]  "
              f"| obs mean: {obs.mean() * 100:+.4f}%")

    return prior_idata


# =========================================================================
# Posterior predictive
# =========================================================================


def run_posterior_predictive_checks(
    model: pm.Model, idata: az.InferenceData, data: dict, cfg: NowcastConfig | None = None,
) -> None:
    """Density overlays and test-statistics comparing replicated to observed."""
    print("Sampling posterior predictive\u2026")
    with model:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=42)

    obs_sources = build_obs_sources(data)
    rng = np.random.default_rng(42)

    # ---- Density overlays ----
    n_src = len(obs_sources)
    n_cols = 3
    n_rows = (n_src + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes_flat = axes.flatten()

    for idx, (vn, (label, obs)) in enumerate(obs_sources.items()):
        ax = axes_flat[idx]
        pp = idata.posterior_predictive[vn].values
        n_ch, n_dr, n_obs = pp.shape
        pp_flat = pp.reshape(-1, n_obs)
        n_total = pp_flat.shape[0]
        sub_idx = rng.choice(n_total, size=min(100, n_total), replace=False)

        obs_pct = obs * 100
        pad = 0.5 * np.ptp(obs_pct)
        bins = np.linspace(obs_pct.min() - pad, obs_pct.max() + pad, 60)

        for i in sub_idx:
            ax.hist(pp_flat[i] * 100, bins=bins, density=True, histtype="step",
                    alpha=0.08, color="steelblue", lw=0.5)
        ax.hist(obs_pct, bins=bins, density=True, histtype="step", color="black", lw=2,
                label="Observed")
        ax.set_xlabel("Monthly growth (%)")
        ax.set_title(label)
        ax.legend(fontsize=7)

    for idx in range(n_src, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Posterior Predictive Check: Density Overlays", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "ppc_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'ppc_density.png'}")

    # ---- Test statistics ----
    def _lag1_acf(x: np.ndarray) -> float:
        if len(x) < 3:
            return 0.0
        xc = x - x.mean()
        c0 = np.dot(xc, xc)
        return np.dot(xc[:-1], xc[1:]) / c0 if c0 > 0 else 0.0

    stat_fns = {"Skewness": lambda x: sp_stats.skew(x), "Lag-1 ACF": _lag1_acf}
    n_stat = len(stat_fns)
    fig, axes = plt.subplots(n_src, n_stat, figsize=(5 * n_stat, 2.5 * n_src))

    for row, (vn, (label, obs)) in enumerate(obs_sources.items()):
        pp = idata.posterior_predictive[vn].values
        pp_flat = pp.reshape(-1, pp.shape[-1])
        n_sub = min(500, pp_flat.shape[0])
        sub_idx = rng.choice(pp_flat.shape[0], size=n_sub, replace=False)

        for col, (sname, sfn) in enumerate(stat_fns.items()):
            ax = axes[row, col] if n_src > 1 else axes[col]
            rep_vals = np.array([sfn(pp_flat[i]) for i in sub_idx])
            obs_val = sfn(obs)
            ax.hist(rep_vals, bins=50, density=True, alpha=0.5, color="steelblue")
            ax.axvline(obs_val, color="black", lw=2, ls="--", label=f"Obs: {obs_val:.3f}")
            p = np.mean(rep_vals >= obs_val)
            p = min(p, 1 - p)
            ax.set_title(f"{label}: {sname} (p={p:.3f})", fontsize=9)
            ax.legend(fontsize=6)

    fig.suptitle("Posterior Predictive Test Statistics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "ppc_test_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'ppc_test_stats.png'}")


# =========================================================================
# Era-specific posterior summary
# =========================================================================

_ERA_LABELS = [
    "Pre-COVID  (2012-2019)",
    "Post-COVID (2020+)   ",
]


def print_era_summary(idata: az.InferenceData) -> None:
    """Print posterior summaries of era-specific mu_g and shared phi."""
    if "mu_g_era" not in idata.posterior:
        return

    mu_g = idata.posterior["mu_g_era"].values       # (chains, draws, N_ERAS)
    n_eras = mu_g.shape[-1]

    print("\n" + "=" * 72)
    print("ERA-SPECIFIC LATENT STATE PARAMETERS")
    print("=" * 72)
    hdr = f"{'Era':<26} {'mu_g mean':>10} {'mu_g sd':>9} {'mu_g 80% HDI':>20}"
    print(hdr)
    print("-" * 72)
    for e in range(n_eras):
        v = mu_g[:, :, e].flatten()
        label = _ERA_LABELS[e] if e < len(_ERA_LABELS) else f"Era {e}"
        lo, hi = np.percentile(v, [10, 90])
        print(
            f"  {label:<24} {v.mean() * 100:+.4f}%  {v.std() * 100:.4f}%  "
            f"[{lo * 100:+.4f}%, {hi * 100:+.4f}%]"
        )

    phi = idata.posterior["phi_raw"].values.flatten()
    lo, hi = np.percentile(phi, [10, 90])
    print(
        f"\n  phi (shared):            {phi.mean():.4f}    {phi.std():.4f}    "
        f"[{lo:.4f}, {hi:.4f}]"
    )


# =========================================================================
# LOO-CV
# =========================================================================


def _obs_time_indices(data: dict) -> dict[str, np.ndarray]:
    """Map likelihood var_name → array of time-step indices into data["dates"]."""
    idx_map: dict[str, np.ndarray] = {}
    if len(data["ces_sa_obs"]) > 0:
        idx_map["obs_ces_sa"] = data["ces_sa_obs"]
    if len(data["ces_nsa_obs"]) > 0:
        idx_map["obs_ces_nsa"] = data["ces_nsa_obs"]
    idx_map["obs_qcew"] = data["qcew_obs"]
    for pp in data["pp_data"]:
        if len(pp["pp_obs"]) > 0:
            idx_map[f"obs_{pp['config'].name.lower()}"] = pp["pp_obs"]
    return idx_map


def _print_loo_outliers(
    label: str,
    khat: np.ndarray,
    elpd_i: np.ndarray,
    dates: list,
    time_idx: np.ndarray | None,
    obs_vals: np.ndarray,
    *,
    max_rows: int = 10,
) -> None:
    """Print a table of LOO outlier observations sorted by pointwise ELPD."""
    elpd_mean = float(np.mean(elpd_i))
    elpd_std = float(np.std(elpd_i))
    threshold = elpd_mean - 2.0 * elpd_std

    flagged = np.where((khat > 0.5) | (elpd_i < threshold))[0]
    if len(flagged) == 0:
        return

    order = np.argsort(elpd_i[flagged])
    flagged = flagged[order][:max_rows]

    print(f"  {'Date':>12}  {'k-hat':>7}  {'ELPD_i':>8}  {'g_obs':>10}  Flag")
    for j in flagged:
        dt_str = str(dates[time_idx[j]]) if time_idx is not None else f"idx {j}"
        flag = "BAD" if khat[j] > 0.7 else ("WARN" if khat[j] > 0.5 else "ELPD")
        print(
            f"  {dt_str:>12}  {khat[j]:7.3f}  {elpd_i[j]:+8.1f}"
            f"  {obs_vals[j]:+10.5f}  {flag}"
        )
    print(f"  (mean ELPD_i = {elpd_mean:+.1f}, flagged if k-hat > 0.5 "
          f"or ELPD_i < {threshold:+.1f})")


def run_loo_cv(model: pm.Model, idata: az.InferenceData, data: dict, cfg: NowcastConfig | None = None) -> None:
    """Compute LOO-CV per source, list outliers, and visualise k-hat diagnostics.

    For each source, prints summary statistics and a table of flagged
    observations (k-hat > 0.5 or pointwise ELPD more than 2 SD below
    the source mean), sorted by worst ELPD first.  This is the primary
    diagnostic value of LOO in a state-space model — identifying
    observations where the noise model is most strained.
    """
    if not hasattr(idata, "log_likelihood"):
        print("Computing log-likelihood for LOO-CV\u2026")
        try:
            with model:
                idata.extend(pm.compute_log_likelihood(idata))
        except Exception as e:
            print(f"Could not compute log-likelihood: {e}")
            print("Skipping LOO-CV.")
            return

    obs_sources = build_obs_sources(data)
    source_list = list(obs_sources.items())
    time_idx_map = _obs_time_indices(data)
    all_dates = data["dates"]

    print("\n" + "=" * 72)
    print("LEAVE-ONE-OUT CROSS-VALIDATION (PSIS-LOO)")
    print("  Note: LOO measures interpolation consistency, not forecast")
    print("  skill.  Outlier table flags potential data anomalies.")
    print("=" * 72)

    n_src = len(source_list)
    n_cols = 3
    n_rows = (n_src + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes_flat = axes.flatten()

    for idx, (var_name, (label, obs_vals)) in enumerate(source_list):
        ax = axes_flat[idx]
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Estimated shape parameter")
                loo_result = az.loo(idata, var_name=var_name, pointwise=True)
            elpd = loo_result.elpd_loo
            se = loo_result.se
            p_loo = loo_result.p_loo
            khat = np.asarray(loo_result.pareto_k)
            elpd_i = np.asarray(loo_result.loo_i)

            n_high = int(np.sum(khat > 0.7))
            n_warn = int(np.sum((khat > 0.5) & (khat <= 0.7)))

            print(f"\n{label} ({var_name}):")
            print(f"  ELPD LOO: {elpd:.1f} +/- {se:.1f}")
            print(f"  p_loo:    {p_loo:.1f}")
            print(f"  k-hat > 0.7 (bad):  {n_high}")
            print(f"  k-hat > 0.5 (warn): {n_warn}")

            _print_loo_outliers(
                label, khat, elpd_i, all_dates,
                time_idx_map.get(var_name), obs_vals,
            )

            colors = np.where(khat > 0.7, "red", np.where(khat > 0.5, "orange", "steelblue"))
            ax.scatter(range(len(khat)), khat, s=8, c=colors, alpha=0.6)
            ax.axhline(0.7, color="red", ls="--", lw=1, alpha=0.7, label="k-hat = 0.7")
            ax.axhline(0.5, color="orange", ls="--", lw=1, alpha=0.7, label="k-hat = 0.5")
            ax.set_xlabel("Observation index")
            ax.set_ylabel("k-hat")
            ax.set_title(f"{label}: PSIS-LOO k-hat")
            ax.legend(fontsize=7)
        except Exception as e:
            print(f"\n{label}: LOO-CV failed \u2014 {e}")
            ax.set_visible(False)

    for idx in range(n_src, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("LOO-CV k-hat Diagnostics (Pareto shape parameter)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "loo_khat.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_DIR / 'loo_khat.png'}")
