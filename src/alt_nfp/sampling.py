"""MCMC sampling utilities.

Provides :func:`sample_model` which attempts sampling with the *nutpie*
backend (fast C++/Rust NUTS implementation preferred on Apple Silicon) and
falls back to PyMC's built-in NUTS sampler when *nutpie* is unavailable.

Three preset configurations are exported for different use cases:

* :data:`DEFAULT_SAMPLER_KWARGS` — full production run (8 000 draws).
* :data:`MEDIUM_SAMPLER_KWARGS` — sensitivity sweeps (4 000 draws).
* :data:`LIGHT_SAMPLER_KWARGS` — backtest loops (~30 s / run).
"""

from __future__ import annotations

import arviz as az
import pymc as pm

# Default sampling configuration (full production run)
DEFAULT_SAMPLER_KWARGS: dict = dict(
    draws=8000,
    tune=6000,
    chains=4,
    target_accept=0.97,
    return_inferencedata=True,
)

# Lighter configuration for backtest loops (~30 sec/run)
LIGHT_SAMPLER_KWARGS: dict = dict(
    draws=2000,
    tune=2000,
    chains=2,
    target_accept=0.95,
    return_inferencedata=True,
)

# Medium configuration for sensitivity sweeps
MEDIUM_SAMPLER_KWARGS: dict = dict(
    draws=4000,
    tune=3000,
    chains=4,
    target_accept=0.95,
    return_inferencedata=True,
)


def sample_model(
    model: pm.Model,
    sampler_kwargs: dict | None = None,
) -> az.InferenceData:
    """Sample the model using nutpie (preferred) or PyMC NUTS.

    Parameters
    ----------
    model : pm.Model
        Compiled PyMC model.
    sampler_kwargs : dict, optional
        Override the default sampling configuration.  Use
        ``LIGHT_SAMPLER_KWARGS`` for backtest loops or
        ``MEDIUM_SAMPLER_KWARGS`` for sensitivity sweeps.
    """
    if sampler_kwargs is None:
        sampler_kwargs = DEFAULT_SAMPLER_KWARGS

    with model:
        try:
            idata = pm.sample(nuts_sampler="nutpie", **sampler_kwargs)
            sampler_used = "nutpie"
        except Exception as e:
            print(f"nutpie unavailable ({e}), falling back to PyMC NUTS")
            idata = pm.sample(**sampler_kwargs)
            sampler_used = "pymc"

    print(f"\nSampling complete ({sampler_used})")
    return idata
