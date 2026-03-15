"""MCMC sampling utilities.

Provides :func:`sample_model` which attempts sampling with the *nutpie*
backend (fast C++/Rust NUTS implementation preferred on Apple Silicon) and
falls back to PyMC's built-in NUTS sampler when *nutpie* is unavailable.

Three preset configurations are exported for different use cases:

* :data:`DEFAULT_SAMPLER_KWARGS` — production run (4 000 draws).
* :data:`MEDIUM_SAMPLER_KWARGS` — sensitivity sweeps (4 000 draws).
* :data:`LIGHT_SAMPLER_KWARGS` — backtest loops / fast iteration (~30 s).
"""

from __future__ import annotations

import warnings

import arviz as az
import pymc as pm

from .settings import NowcastConfig

# Legacy sampling dicts — kept for backwards compatibility during migration.
# New code should use ``cfg.sampling.get_preset(name).to_pymc_kwargs()``.
_DEFAULT = NowcastConfig()
DEFAULT_SAMPLER_KWARGS: dict = _DEFAULT.sampling.default.to_pymc_kwargs()
LIGHT_SAMPLER_KWARGS: dict = _DEFAULT.sampling.light.to_pymc_kwargs()
MEDIUM_SAMPLER_KWARGS: dict = _DEFAULT.sampling.medium.to_pymc_kwargs()


def sample_model(
    model: pm.Model,
    sampler_kwargs: dict | None = None,
    *,
    cfg: NowcastConfig | None = None,
    preset: str | None = None,
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
    cfg : NowcastConfig, optional
        Pipeline configuration.  Used to look up the *preset* when
        *sampler_kwargs* is not provided.
    preset : str, optional
        Sampling preset name (e.g. ``"default"``, ``"light"``,
        ``"medium"``).  Ignored when *sampler_kwargs* is given.
    """
    if sampler_kwargs is None:
        if cfg is not None and preset is not None:
            sampler_kwargs = cfg.sampling.get_preset(preset).to_pymc_kwargs()
        else:
            sampler_kwargs = DEFAULT_SAMPLER_KWARGS

    with model:
        warnings.filterwarnings(
            "ignore",
            message="Numba will use object mode",
            category=UserWarning,
            module=r"pytensor\.link\.numba",
        )
        try:
            idata = pm.sample(nuts_sampler="nutpie", **sampler_kwargs)
            sampler_used = "nutpie"
        except Exception as e:
            print(f"nutpie unavailable ({e}), falling back to PyMC NUTS")
            idata = pm.sample(**sampler_kwargs)
            sampler_used = "pymc"

    print(f"\nSampling complete ({sampler_used})")
    return idata
