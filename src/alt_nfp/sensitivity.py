# ---------------------------------------------------------------------------
# alt_nfp.sensitivity — QCEW sigma sensitivity analysis
# ---------------------------------------------------------------------------
"""Run the v3 model with several QCEW noise levels (0.5x, 1x, 2x).
Compare key parameters and precision budgets to check whether calibrated
sigmas drive conclusions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_DIR, SIGMA_QCEW_M3, SIGMA_QCEW_M12
from .data import load_data
from .diagnostics import print_source_contributions
from .model import build_model
from .sampling import MEDIUM_SAMPLER_KWARGS, sample_model

# Configs: (label, sigma_m3, sigma_m12)
QCEW_SIGMA_CONFIGS: list[tuple[str, float, float]] = [
    ("0.5x (tight)", SIGMA_QCEW_M3 * 0.5, SIGMA_QCEW_M12 * 0.5),
    ("1x (baseline)", SIGMA_QCEW_M3, SIGMA_QCEW_M12),
    ("2x (loose)", SIGMA_QCEW_M3 * 2.0, SIGMA_QCEW_M12 * 2.0),
]


def run_sensitivity(
    configs: list[tuple[str, float, float]] | None = None,
) -> list[dict]:
    """Run QCEW sigma sensitivity sweep.

    Parameters
    ----------
    configs : list of (label, sigma_m3, sigma_m12), optional
        Override the default 0.5x / 1x / 2x grid.

    Returns
    -------
    list[dict]
        Per-config result records with parameter summaries and precision shares.
    """
    if configs is None:
        configs = QCEW_SIGMA_CONFIGS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data\u2026")
    data = load_data()
    print()

    # Parameters to compare — (display_name, posterior_key, pp_index_or_None, scale, fmt)
    param_specs = _build_param_specs(data)

    results: list[dict] = []

    for label, sigma_m3, sigma_m12 in configs:
        print(f"Running config: {label} (\u03c3_M3={sigma_m3}, \u03c3_M12={sigma_m12})\u2026")
        model = build_model(data, sigma_qcew_m3=sigma_m3, sigma_qcew_m12=sigma_m12)
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

        shares = _precision_shares(idata, data, sigma_m3, sigma_m12)
        row["precision_shares"] = shares
        results.append(row)

        print_source_contributions(idata, data)
        print(f"  Done.\n")

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
        ("\u03c3_ces_sa (%)", "sigma_ces_sa", None, 100, ".3f"),
        ("\u03c3_ces_nsa (%)", "sigma_ces_nsa", None, 100, ".3f"),
        ("\u03c6_0 BD (%/mo)", "phi_0", None, 100, ".4f"),
        ("\u03c6_1 (birth)", "phi_1", None, 1, ".3f"),
        ("\u03c6_2 (QCEW lag)", "phi_2", None, 1, ".3f"),
        ("\u03c3_bd (%)", "sigma_bd", None, 100, ".4f"),
    ]
    for pp in data["pp_data"]:
        name = pp["config"].name.lower()
        label = pp["name"]
        specs.append((f"\u03b1_{label} (%/mo)", f"alpha_{name}", None, 100, ".4f"))
        specs.append((f"\u03bb_{label}", f"lam_{name}", None, 1, ".3f"))
        specs.append((f"\u03c3_{label} (%)", f"sigma_{name}", None, 100, ".3f"))
        if pp["config"].error_model == "ar1":
            specs.append((f"\u03c1_{label}", f"rho_{name}", None, 1, ".3f"))
    return specs


# =========================================================================
# Precision budget (returns dict, parallels diagnostics.print_source_contributions)
# =========================================================================


def _precision_shares(
    idata, data: dict, sigma_m3: float, sigma_m12: float,
) -> dict[str, float]:
    """Return ``{source: share}`` of total precision."""
    post = idata.posterior
    lam_ces = post["lambda_ces"].values.flatten().mean()
    sig_ces_sa = post["sigma_ces_sa"].values.flatten().mean()
    sig_ces_nsa = post["sigma_ces_nsa"].values.flatten().mean()

    prec_ces_sa = lam_ces**2 / sig_ces_sa**2
    prec_ces_nsa = lam_ces**2 / sig_ces_nsa**2
    prec_qcew_m3 = 1.0 / sigma_m3**2
    prec_qcew_m12 = 1.0 / sigma_m12**2

    n_ces_sa = len(data["ces_sa_obs"])
    n_ces_nsa = len(data["ces_nsa_obs"])
    n_qcew_m3 = int(data["qcew_is_m3"].sum())
    n_qcew_m12 = len(data["qcew_obs"]) - n_qcew_m3

    total_ces_sa = prec_ces_sa * n_ces_sa
    total_ces_nsa = prec_ces_nsa * n_ces_nsa
    total_qcew = prec_qcew_m3 * n_qcew_m3 + prec_qcew_m12 * n_qcew_m12

    shares: dict[str, float] = {}
    total_pp = 0.0
    for pp in data["pp_data"]:
        name = pp["config"].name.lower()
        sig_p = post[f"sigma_{name}"].values.flatten().mean()
        lam_p = post[f"lam_{name}"].values.flatten().mean()
        n_obs = len(pp["pp_obs"])
        prec_p = lam_p**2 / sig_p**2
        if pp["config"].error_model == "ar1":
            rho_p = post[f"rho_{name}"].values.flatten().mean()
            prec_p = lam_p**2 * (1 - rho_p**2) / sig_p**2
        pp_total = prec_p * n_obs
        shares[pp["name"]] = pp_total
        total_pp += pp_total

    total_all = total_ces_sa + total_ces_nsa + total_qcew + total_pp
    shares["CES SA"] = total_ces_sa / total_all
    shares["CES NSA"] = total_ces_nsa / total_all
    shares["QCEW"] = total_qcew / total_all
    for pp in data["pp_data"]:
        shares[pp["name"]] = shares[pp["name"]] / total_all

    return shares


# =========================================================================
# Reporting
# =========================================================================


def _print_comparison_table(results, param_specs, configs) -> None:
    print("=" * 100)
    print("QCEW SIGMA SENSITIVITY: Parameter comparison (mean, 80% HDI)")
    print("=" * 100)
    print(f"{'Parameter':<20}", end="")
    for label, _, _ in configs:
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
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, means, width, label=r["config"],
            yerr=[err_lo, err_hi], capsize=2,
        )
        if i == 1:
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
