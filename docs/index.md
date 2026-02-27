---
hide:
  - navigation
---

# alt_nfp

**Bayesian state-space model for nowcasting U.S. nonfarm payroll employment.**

---

## What is alt_nfp?

`alt_nfp` fuses three families of employment data inside a single hierarchical
Bayesian model to produce a real-time estimate of total nonfarm payroll (NFP)
employment growth:

| Data Source | Role | Frequency |
|---|---|---|
| **QCEW** (Quarterly Census of Employment and Wages) | Near-census truth anchor | Quarterly (with monthly interpolation) |
| **CES** (Current Employment Statistics) | High-frequency official estimate | Monthly |
| **Payroll providers** | Private real-time signals | Monthly |

The model is estimated with [PyMC](https://www.pymc.io/) using the NUTS
sampler (via [nutpie](https://github.com/pymc-devs/nutpie) when available) and
produces:

- **Latent employment growth** estimates with full posterior uncertainty.
- **Forward forecasts** of SA and NSA employment indices and jobs added.
- **Nowcast backtests** quantifying the operational value of provider data.
- **Bayesian diagnostic suite** — prior/posterior predictive checks, LOO-CV,
  residual analysis, and sensitivity sweeps.

## Key Features

!!! info "Config-Driven Providers"
    Adding a new payroll vendor requires only a single `ProviderConfig` entry.
    The model, diagnostics, plots, and forecasts adapt automatically.

!!! info "Structural Birth/Death Model"
    Time-varying BD offset:
    \[
    \text{bd}_t = \varphi_0 + \varphi_1 \cdot \text{birth\_rate}_c
    + \varphi_2 \cdot \text{bd}^{\text{QCEW}}_{t-L}
    + \varphi_3 \cdot X^{\text{cycle}} + \sigma_{\text{bd}} \cdot \xi_t
    \]

!!! info "Vintage-Tracked Data Pipeline"
    Real-time data revisions are tracked through a Hive-partitioned Parquet
    vintage store, enabling faithful reconstruction of historical information
    sets.

## Quick Links

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Install the package and run your first nowcast.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Learn about the model, data sources, and Bayesian workflow.

    [:octicons-arrow-right-24: Model Overview](user-guide/model-overview.md)

-   :material-sitemap:{ .lg .middle } **Architecture**

    ---

    Understand the system design and data flow.

    [:octicons-arrow-right-24: System Design](architecture/system-design.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Auto-generated reference from docstrings.

    [:octicons-arrow-right-24: API Reference](reference/index.md)

</div>
