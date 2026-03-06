"""QCEW observation-noise sensitivity analysis.

Runs the state-space model under multiple QCEW noise configurations
(default: 0.5x, 1x, 2x baseline) using
:data:`~alt_nfp.sampling.MEDIUM_SAMPLER_KWARGS`.  For each configuration
the sweep records:

* Posterior summaries of all key parameters (BD, CES, provider-specific).
* Precision-weighted information budgets showing how the QCEW share shifts.
* A stability verdict: if key parameters (phi_0, alpha_ces) remain within
  tolerance the calibrated sigmas are not driving conclusions.

Outputs include a grouped bar-chart comparing parameters across configs.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from . import config as _config
from .config import (
    LOG_SIGMA_QCEW_BOUNDARY_MU,
    LOG_SIGMA_QCEW_MID_MU,
    OUTPUT_DIR,
    PROVIDERS,
)
from .diagnostics import _qcew_precision_by_tier, print_source_contributions
from .ingest import build_panel
from .model import build_model
from .panel_adapter import panel_to_model_data
from .sampling import MEDIUM_SAMPLER_KWARGS, sample_model

# Configs: (label, scale_mid, scale_boundary)
# Scaling shifts the LogNormal log-mu by log(scale), so "2x" doubles the
# prior center for sigma_qcew while preserving the log-space spread.
QCEW_SIGMA_CONFIGS: list[tuple[str, float, float]] = [
    ("0.5x (tight)", 0.5, 0.5),
    ("1x (baseline)", 1.0, 1.0),
    ("2x (loose)", 2.0, 2.0),
]


def run_sensitivity(
    configs: list[tuple[str, float, float]] | None = None,
) -> list[dict]:
    """Run QCEW prior-scale sensitivity sweep.

    Parameters
    ----------
    configs : list of (label, scale_mid, scale_boundary), optional
        Override the default 0.5x / 1x / 2x grid.  Each scale factor shifts
        the LogNormal log-mu by ``log(scale)``, effectively multiplying the
        prior center for sigma_qcew by ``scale``.

    Returns
    -------
    list[dict]
        Per-config result records with parameter summaries and precision shares.
    """
    if configs is None:
        configs = QCEW_SIGMA_CONFIGS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data\u2026")
    panel = build_panel()
    data = panel_to_model_data(panel, PROVIDERS)
    print()

    param_specs = _build_param_specs(data)

    orig_mid_mu = _config.LOG_SIGMA_QCEW_MID_MU
    orig_boundary_mu = _config.LOG_SIGMA_QCEW_BOUNDARY_MU
    results: list[dict] = []

    try:
        for label, scale_mid, scale_boundary in configs:
            print(
                f"Running config: {label} "
                f"(prior scale mid={scale_mid}, boundary={scale_boundary})\u2026"
            )
            _config.LOG_SIGMA_QCEW_MID_MU = orig_mid_mu + math.log(scale_mid)
            _config.LOG_SIGMA_QCEW_BOUNDARY_MU = orig_boundary_mu + math.log(scale_boundary)
            model = build_model(data)
            idata = sample_model(model, sampler_kwargs=MEDIUM_SAMPLER_KWARGS)

            row: dict = {"config": label}
            post = idata.posterior

            for pname, key, idx, scale, _ in param_specs:
                vals = post[key].values
                if idx is not None:
                    vals = vals[:, :, idx]
                v = vals.flatten()
                row[pname] = (
                    v.mean() * scale,
                    float(np.percentile(v, 10)) * scale,
                    float(np.percentile(v, 90)) * scale,
                )

            shares = _precision_shares(idata, data)
            row["precision_shares"] = shares
            results.append(row)

            print_source_contributions(idata, data)
            print(f"  Done.\n")
    finally:
        _config.LOG_SIGMA_QCEW_MID_MU = orig_mid_mu
        _config.LOG_SIGMA_QCEW_BOUNDARY_MU = orig_boundary_mu

    _print_comparison_table(results, param_specs, configs)
    _print_precision_table(results)
    _print_verdict(results)
    _plot_sensitivity(results, param_specs, configs)

    return results


# =========================================================================
# Parameter spec builder (adapts to configured providers)
# =========================================================================


def _build_param_specs(data: dict) -> list[tuple[str, str, int | None, float, str]]:
    """Build (display_name, key, index, scale, fmt) for each parameter."""
    specs: list[tuple[str, str, int | None, float, str]] = [
        ("\u03b1_ces (%/mo)", "alpha_ces", None, 100, ".4f"),
        ("\u03bb_ces", "lambda_ces", None, 1, ".4f"),
    ]
    # CES vintage-specific sigmas
    vintage_labels = ['v1', 'v2', 'v3']
    for v in range(3):
        specs.append(
            (f"\u03c3_ces_sa_{vintage_labels[v]} (%)", "sigma_ces_sa", v, 100, ".3f")
        )
        specs.append(
            (f"\u03c3_ces_nsa_{vintage_labels[v]} (%)", "sigma_ces_nsa", v, 100, ".3f")
        )
    specs.append(("\u03c6_0 BD (%/mo)", "phi_0", None, 100, ".4f"))
    specs.append(("\u03c3_bd (%)", "sigma_bd", None, 100, ".4f"))
    # Cyclical indicator loadings (phi_3) if available
    from .config import CYCLICAL_INDICATORS

    cyclical_labels = [spec['name'] for spec in CYCLICAL_INDICATORS]
    cyclical_keys = [f"{spec['name']}_c" for spec in CYCLICAL_INDICATORS]
    n_cyc = sum(
        1 for k in cyclical_keys
        if data.get(k) is not None and np.any(data[k] != 0.0)
    )
    for i in range(n_cyc):
        lbl = cyclical_labels[i] if i < len(cyclical_labels) else f'cyc_{i}'
        specs.append((f"\u03c6_3[{lbl}]", "phi_3", i, 1, ".3f"))
    for pp in data["pp_data"]:
        name = pp["config"].name.lower()
        label = pp["name"]
        specs.append((f"\u03b1_{label} (%/mo)", f"alpha_{name}", None, 100, ".4f"))
        specs.append((f"\u03bb_{label}", f"lam_{name}", None, 1, ".3f"))
        specs.append((f"\u03c3_{label} (%)", f"sigma_pp_{name}", None, 100, ".3f"))
        if pp["config"].error_model == "ar1":
            specs.append((f"\u03c1_{label}", f"rho_{name}", None, 1, ".3f"))
    return specs


# =========================================================================
# Precision budget (returns dict, parallels diagnostics.print_source_contributions)
# =========================================================================


def _precision_shares(idata, data: dict) -> dict[str, float]:
    """Return ``{source: share}`` of total precision."""
    from .diagnostics import _ces_precision_rows

    total_qcew, _, _, _, _ = _qcew_precision_by_tier(idata, data)

    shares: dict[str, float] = {}
    ces_rows, total_ces = _ces_precision_rows(idata, data)
    for label, _n, _avg, total_prec in ces_rows:
        shares[label] = total_prec

    total_pp = 0.0
    post = idata.posterior
    for pp in data["pp_data"]:
        name = pp["config"].name.lower()
        sig_p = post[f"sigma_pp_{name}"].values.flatten().mean()
        lam_p = post[f"lam_{name}"].values.flatten().mean()
        n_obs = len(pp["pp_obs"])
        prec_p = lam_p**2 / sig_p**2
        if pp["config"].error_model == "ar1":
            rho_p = post[f"rho_{name}"].values.flatten().mean()
            prec_p = lam_p**2 * (1 - rho_p**2) / sig_p**2
        pp_total = prec_p * n_obs
        shares[pp["name"]] = pp_total
        total_pp += pp_total

    total_all = total_ces + total_qcew + total_pp
    shares["QCEW"] = total_qcew
    for key in shares:
        shares[key] = shares[key] / total_all

    return shares


# =========================================================================
# Reporting
# =========================================================================


def _print_comparison_table(results, param_specs, configs) -> None:
    print("=" * 100)
    print("QCEW SIGMA SENSITIVITY: Parameter comparison (mean, 80% HDI)")
    print("=" * 100)
    print(f"{'Parameter':<20}", end="")
    for label, _s1, _s2 in configs:
        print(f"  {label:>24}", end="")
    print()
    print("-" * 100)

    for pname, _key, _idx, _scale, fmt in param_specs:
        line = f"{pname:<20}"
        for r in results:
            if pname in r:
                mean, lo, hi = r[pname]
                cell = f"{mean:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"
                line += f"  {cell:>24}"
            else:
                line += f"  {'n/a':>24}"
        print(line)


def _print_precision_table(results) -> None:
    print("\n" + "=" * 100)
    print("Precision budget shares (%)")
    print("=" * 100)
    print(f"{'Source':<12}", end="")
    for r in results:
        print(f"  {r['config']:>18}", end="")
    print()
    print("-" * 100)

    all_sources = list(results[0]["precision_shares"].keys())
    for src in all_sources:
        line = f"{src:<12}"
        for r in results:
            pct = 100 * r["precision_shares"].get(src, 0.0)
            line += f"  {pct:>17.1f}%"
        print(line)


def _print_verdict(results) -> None:
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    bd_key = "\u03c6_0 BD (%/mo)"
    ces_key = "\u03b1_ces (%/mo)"
    bd_means = [r[bd_key][0] for r in results if bd_key in r]
    ces_means = [r[ces_key][0] for r in results if ces_key in r]
    if bd_means and ces_means:
        bd_range = max(bd_means) - min(bd_means)
        ces_range = max(ces_means) - min(ces_means)
        if bd_range < 0.002 and ces_range < 0.001:
            print(
                "Key parameters (\u03c6_0, \u03b1_ces) are stable across QCEW sigma "
                "configurations. Calibration is not driving conclusions."
            )
        else:
            print(
                "Key parameters shift meaningfully with QCEW sigma. "
                "Consider reporting sensitivity in the paper."
            )


def _plot_sensitivity(results, param_specs, configs) -> None:
    # Select a subset of parameters for the bar chart
    plot_keys = [
        "\u03b1_ces (%/mo)", "\u03bb_ces",
        "\u03c6_0 BD (%/mo)", "\u03c6_1 (birth)", "\u03c6_2 (QCEW lag)",
    ]
    # Add per-provider lambda
    for pname, *_ in param_specs:
        if pname.startswith("\u03bb_PP") or pname.startswith("\u03bb_pp"):
            plot_keys.append(pname)
    # Also add rho if present
    for pname, *_ in param_specs:
        if pname.startswith("\u03c1_"):
            plot_keys.append(pname)

    # Filter to keys that exist in results
    plot_keys = [k for k in plot_keys if k in results[0]]
    n_params = len(plot_keys)

    x = np.arange(n_params)
    width = 0.25

    fig, ax = plt.subplots(1, 1, figsize=(max(12, 2.5 * n_params), 6))
    for i, r in enumerate(results):
        means = [r[p][0] for p in plot_keys]
        los = [r[p][1] for p in plot_keys]
        his = [r[p][2] for p in plot_keys]
        err_lo = [m - lo for m, lo in zip(means, los)]
        err_hi = [hi - m for m, hi in zip(means, his)]
        offset = (i - (len(results) - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width, label=r["config"],
            yerr=[err_lo, err_hi], capsize=2,
        )
        if len(results) >= 2 and i == (len(results) - 1) // 2:
            for b in bars:
                b.set_edgecolor("black")
                b.set_linewidth(1.2)

    ax.set_ylabel("Posterior mean (80% HDI)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_keys, rotation=15, ha="right")
    ax.legend()
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_title("QCEW Sigma Sensitivity: Key Parameters by Configuration")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "sensitivity_qcew_sigma.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_DIR / 'sensitivity_qcew_sigma.png'}")
