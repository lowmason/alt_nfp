# ---------------------------------------------------------------------------
# alt_nfp.checks — Prior / posterior predictive checks, LOO-CV
# ---------------------------------------------------------------------------
from __future__ import annotations

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy import stats as sp_stats

from .config import OUTPUT_DIR
from .data import build_obs_sources


# =========================================================================
# Prior predictive
# =========================================================================


def run_prior_predictive_checks(
    model: pm.Model, data: dict
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
    model: pm.Model, idata: az.InferenceData, data: dict,
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
# LOO-CV
# =========================================================================


def run_loo_cv(model: pm.Model, idata: az.InferenceData, data: dict) -> None:
    """Compute LOO-CV per source and visualise k-hat diagnostics."""
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

    print("\n" + "=" * 72)
    print("LEAVE-ONE-OUT CROSS-VALIDATION (PSIS-LOO)")
    print("=" * 72)

    n_src = len(source_list)
    n_cols = 3
    n_rows = (n_src + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes_flat = axes.flatten()

    for idx, (var_name, (label, _obs)) in enumerate(source_list):
        ax = axes_flat[idx]
        try:
            loo_result = az.loo(idata, var_name=var_name, pointwise=True)
            elpd = loo_result.elpd_loo
            se = loo_result.se
            p_loo = loo_result.p_loo
            khat = np.asarray(loo_result.pareto_k)

            n_high = int(np.sum(khat > 0.7))
            n_warn = int(np.sum((khat > 0.5) & (khat <= 0.7)))

            print(f"\n{label} ({var_name}):")
            print(f"  ELPD LOO: {elpd:.1f} +/- {se:.1f}")
            print(f"  p_loo:    {p_loo:.1f}")
            print(f"  k-hat > 0.7 (bad):  {n_high}")
            print(f"  k-hat > 0.5 (warn): {n_warn}")

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
