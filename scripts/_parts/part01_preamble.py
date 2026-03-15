#!/usr/bin/env python
# =============================================================================
# alt_nfp_estimation.py — Standalone Bayesian State-Space Estimation
# =============================================================================
#
# Self-contained script for estimating U.S. total nonfarm employment growth
# using a Bayesian state-space model.  No imports from the alt_nfp package —
# only standard scientific Python (numpy, pandas, pymc, arviz, matplotlib).
#
# DATA:  Reads scripts/alt_nfp_data.csv (produced by scripts/export_data.py).
# OUTPUT: Plots and InferenceData saved to scripts/output/.
#
# For a Bayesian practitioner more familiar with MATLAB than Python:
#   - PyMC is analogous to JAGS/Stan: you declare a probabilistic model,
#     then the library runs MCMC (specifically NUTS, a variant of HMC).
#   - ArviZ is the diagnostics library (like mcmcdiag in MATLAB).
#   - pytensor.scan is like a for-loop that PyMC can differentiate through
#     (needed for the AR(1) latent process).
#
# MODEL OVERVIEW
# ==============
# The model decomposes total employment growth g(t) into three components:
#
#   g_total_nsa(t) = g_cont(t) + s(t) + bd(t)
#   g_total_sa(t)  = g_cont(t) + bd(t)
#
# where:
#   g_cont(t) : Latent continuing-units growth — AR(1) process
#   s(t)      : Fourier seasonal pattern — annually-evolving harmonics
#   bd(t)     : Structural birth/death offset — intercept + cyclical indicators
#
# Four data sources observe these latent quantities:
#   1. QCEW  — near-census "truth anchor" (NSA, Student-t likelihood)
#   2. CES SA  — BLS survey, seasonally adjusted (Normal likelihood)
#   3. CES NSA — BLS survey, not seasonally adjusted (Normal likelihood)
#   4. Provider — private payroll provider (Normal, iid errors)
#
# =============================================================================

from __future__ import annotations

import math
import warnings
from datetime import date
from pathlib import Path

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
from scipy import stats as sp_stats

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATA_PATH = Path(__file__).resolve().parent / "alt_nfp_data.csv"

