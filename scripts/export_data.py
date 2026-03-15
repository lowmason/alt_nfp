#!/usr/bin/env python
"""Export model-ready data to CSV for the standalone estimation script.

Reads from the alt_nfp vintage store, provider data, and cyclical indicators,
then writes scripts/alt_nfp_data.csv with both raw employment levels and
pre-processed model-ready arrays.

Usage:
    python scripts/export_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfp_models.config import BASE_DIR, providers_from_settings
from nfp_ingest import build_panel
from nfp_models.panel_adapter import panel_to_model_data
from nfp_models.settings import NowcastConfig


def main() -> None:
    cfg = NowcastConfig()
    providers = providers_from_settings(cfg)

    print("Building panel from vintage store + providers...")
    panel = build_panel()
    data = panel_to_model_data(panel, providers, cfg=cfg)

    T = data["T"]
    dates = data["dates"]
    levels = data["levels"]

    # Build per-month arrays, T-length (NaN where missing)
    g_ces_sa = data["g_ces_sa"]
    g_ces_nsa = data["g_ces_nsa"]
    g_qcew = data["g_qcew"]

    # CES vintage indices (T-length, -1 where missing)
    ces_sa_vintage_full = np.full(T, -1, dtype=int)
    ces_nsa_vintage_full = np.full(T, -1, dtype=int)
    inv_map = {i: v for v, i in data["ces_vintage_map"].items()}
    for j, t in enumerate(data["ces_sa_obs"]):
        ces_sa_vintage_full[t] = inv_map[data["ces_sa_vintage_idx"][j]]
    for j, t in enumerate(data["ces_nsa_obs"]):
        ces_nsa_vintage_full[t] = inv_map[data["ces_nsa_vintage_idx"][j]]

    # QCEW noise multiplier and M2 flag (T-length, NaN where no QCEW obs)
    qcew_noise_mult_full = np.full(T, np.nan)
    qcew_is_m2_full = np.full(T, np.nan)
    for j, t in enumerate(data["qcew_obs"]):
        qcew_noise_mult_full[t] = data["qcew_noise_mult"][j]
        qcew_is_m2_full[t] = float(data["qcew_is_m2"][j])

    # Provider growth
    pp_growth = np.full(T, np.nan)
    pp_level = np.full(T, np.nan)
    if data["pp_data"]:
        pp = data["pp_data"][0]
        pp_growth = pp["g_pp"].copy()
        try:
            pp_level = levels[pp["emp_col"]].to_numpy().astype(float)
        except Exception:
            pass

    # Cyclical indicators
    claims_c = data.get("claims_c")
    if claims_c is None:
        claims_c = np.zeros(T)
    jolts_c = data.get("jolts_c")
    if jolts_c is None:
        jolts_c = np.zeros(T)

    # Employment levels from the levels DataFrame
    ces_sa_level = levels["ces_sa_level"].to_numpy().astype(float)
    ces_nsa_level = levels["ces_nsa_level"].to_numpy().astype(float)
    ces_sa_index = levels["ces_sa_index"].to_numpy().astype(float)
    ces_nsa_index = levels["ces_nsa_index"].to_numpy().astype(float)
    qcew_nsa_index = levels["qcew_nsa_index"].to_numpy().astype(float)

    df = pd.DataFrame({
        "ref_date": [d.isoformat() for d in dates],
        "ces_sa_level": ces_sa_level,
        "ces_nsa_level": ces_nsa_level,
        "ces_sa_index": ces_sa_index,
        "ces_nsa_index": ces_nsa_index,
        "qcew_nsa_index": qcew_nsa_index,
        "provider_g_level": pp_level,
        "ces_sa_growth": g_ces_sa,
        "ces_nsa_growth": g_ces_nsa,
        "qcew_growth": g_qcew,
        "provider_g_growth": pp_growth,
        "ces_sa_vintage": ces_sa_vintage_full,
        "ces_nsa_vintage": ces_nsa_vintage_full,
        "qcew_noise_mult": qcew_noise_mult_full,
        "qcew_is_m2": qcew_is_m2_full,
        "claims_c": claims_c,
        "jolts_c": jolts_c,
    })

    out_path = Path(__file__).resolve().parent / "alt_nfp_data.csv"
    df.to_csv(out_path, index=False, float_format="%.10g")
    print(f"\nExported {len(df)} rows to {out_path}")
    print(f"  Date range: {dates[0]} -> {dates[-1]}")
    print(f"  CES SA obs:  {int(np.isfinite(g_ces_sa).sum())}")
    print(f"  CES NSA obs: {int(np.isfinite(g_ces_nsa).sum())}")
    print(f"  QCEW obs:    {int(np.isfinite(g_qcew).sum())}")
    print(f"  Provider obs: {int(np.isfinite(pp_growth).sum())}")


if __name__ == "__main__":
    main()
