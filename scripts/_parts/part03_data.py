# =============================================================================
# SECTION 2: DATA LOADING
# =============================================================================
#
# Reads the CSV exported by export_data.py.  The CSV has one row per month
# with both raw employment levels and pre-processed model-ready arrays.
#
# The model operates on LOG-DIFFERENCES (growth rates), not levels.
# Growth rates are already computed in the CSV by the export script, which
# used the best-available CES print (highest revision per month: rev 2 >
# 1 > 0) and the latest QCEW vintage.
#
# Key arrays built here:
#   g_ces_sa[t]   : CES SA monthly log-difference (NaN where missing)
#   g_ces_nsa[t]  : CES NSA monthly log-difference
#   g_qcew[t]     : QCEW NSA monthly log-difference
#   g_provider[t] : Provider monthly log-difference
#   qcew_obs      : integer indices where QCEW has data
#   ces_sa_obs    : integer indices where CES SA has data
#   etc.
#
# =============================================================================


def load_data(csv_path: Path = DATA_PATH) -> dict:
    """Load CSV and build the model data dictionary.

    Returns a dict with all arrays needed by build_model() and downstream
    diagnostics/plotting functions.  This is the standalone equivalent of
    panel_to_model_data() in the alt_nfp package.
    """
    df = pd.read_csv(csv_path, parse_dates=["ref_date"])
    dates = [d.date() for d in df["ref_date"]]
    T = len(dates)

    # ---- Calendar arrays ----
    month_of_year = np.array([d.month - 1 for d in dates], dtype=int)
    year0 = dates[0].year
    year_of_obs = np.array([d.year - year0 for d in dates], dtype=int)
    n_years = int(year_of_obs.max()) + 1

    # Era index: 0 = Pre-COVID, 1 = Post-COVID
    era_idx = np.array([0 if d < ERA_BREAK else 1 for d in dates], dtype=int)

    # ---- Growth rate arrays (T-length, NaN where missing) ----
    g_ces_sa = df["ces_sa_growth"].to_numpy(dtype=float)
    g_ces_nsa = df["ces_nsa_growth"].to_numpy(dtype=float)
    g_qcew = df["qcew_growth"].to_numpy(dtype=float)
    g_provider = df["provider_g_growth"].to_numpy(dtype=float)

    # ---- Observation indices (where each source has valid data) ----
    ces_sa_obs = np.where(np.isfinite(g_ces_sa))[0]
    ces_nsa_obs = np.where(np.isfinite(g_ces_nsa))[0]
    qcew_obs = np.where(np.isfinite(g_qcew))[0]
    provider_obs = np.where(np.isfinite(g_provider))[0]

    # ---- CES vintage indices ----
    # The CSV has per-month vintage number (0=1st print, 1=2nd, 2=Final, -1=missing).
    # We remap to a contiguous 0-based range so sigma_ces has only as many
    # parameters as there are observed vintage tiers (avoids "ghost" parameters).
    ces_sa_vintage_raw = df["ces_sa_vintage"].to_numpy(dtype=int)
    ces_nsa_vintage_raw = df["ces_nsa_vintage"].to_numpy(dtype=int)
    ces_sa_vidx_at_obs = ces_sa_vintage_raw[ces_sa_obs]
    ces_nsa_vidx_at_obs = ces_nsa_vintage_raw[ces_nsa_obs]

    all_vintages = sorted(
        set(ces_sa_vidx_at_obs.tolist()) | set(ces_nsa_vidx_at_obs.tolist())
    )
    if not all_vintages:
        all_vintages = [2]
    ces_vintage_map = {v: i for i, v in enumerate(all_vintages)}
    n_ces_vintages = len(all_vintages)

    ces_sa_vintage_idx = np.array(
        [ces_vintage_map[v] for v in ces_sa_vidx_at_obs], dtype=int
    )
    ces_nsa_vintage_idx = np.array(
        [ces_vintage_map[v] for v in ces_nsa_vidx_at_obs], dtype=int
    )

    # ---- QCEW noise multipliers (pre-computed in CSV) ----
    # qcew_noise_mult already incorporates revision multipliers AND
    # post-COVID era multipliers for boundary months.
    qcew_noise_mult_full = df["qcew_noise_mult"].to_numpy(dtype=float)
    qcew_noise_mult = qcew_noise_mult_full[qcew_obs]

    # QCEW M2 flag: which QCEW obs are M2 (mid-quarter) vs M1+M3 (boundary)
    qcew_is_m2_full = df["qcew_is_m2"].to_numpy(dtype=float)
    qcew_is_m2 = np.array(
        [bool(qcew_is_m2_full[t]) if np.isfinite(qcew_is_m2_full[t]) else False
         for t in qcew_obs]
    )

    # ---- Birth rate and cyclical indicators ----
    birth_rate = df["birth_rate"].to_numpy(dtype=float)
    claims_c = df["claims_c"].to_numpy(dtype=float)
    jolts_c = df["jolts_c"].to_numpy(dtype=float)

    # ---- Employment levels (for forecast) ----
    ces_sa_level = df["ces_sa_level"].to_numpy(dtype=float)
    ces_nsa_level = df["ces_nsa_level"].to_numpy(dtype=float)
    ces_sa_index = df["ces_sa_index"].to_numpy(dtype=float)
    ces_nsa_index = df["ces_nsa_index"].to_numpy(dtype=float)
    qcew_nsa_index = df["qcew_nsa_index"].to_numpy(dtype=float)
    provider_level = df["provider_g_level"].to_numpy(dtype=float)

    # ---- Build levels DataFrame (for forecast plotting) ----
    levels = pd.DataFrame({
        "ref_date": dates,
        "ces_sa_index": ces_sa_index,
        "ces_nsa_index": ces_nsa_index,
        "qcew_nsa_index": qcew_nsa_index,
        "g_employment": provider_level,
        "ces_sa_level": ces_sa_level,
        "ces_nsa_level": ces_nsa_level,
    })

    # ---- Summary ----
    vintage_names = {0: "1st", 1: "2nd", 2: "Final"}
    inv_vintage_map = {i: v for v, i in ces_vintage_map.items()}

    def _vdist(vidx_raw):
        counts = {}
        for v in vidx_raw:
            counts[int(v)] = counts.get(int(v), 0) + 1
        return ", ".join(
            f"{counts[k]} {vintage_names.get(k, f'v{k}')}"
            for k in sorted(counts) if counts[k] > 0
        ) or "none"

    n_qcew_m2 = int(qcew_is_m2.sum())
    n_qcew_boundary = len(qcew_obs) - n_qcew_m2

    print(f"Data loaded: T = {T} months ({dates[0]} -> {dates[-1]})")
    print(f"  CES SA:  {len(ces_sa_obs)} obs ({_vdist(ces_sa_vidx_at_obs)})")
    print(f"  CES NSA: {len(ces_nsa_obs)} obs ({_vdist(ces_nsa_vidx_at_obs)})")
    print(f"  QCEW:    {len(qcew_obs)} obs — {n_qcew_m2} M2, {n_qcew_boundary} M3+M1")
    print(f"  Provider: {len(provider_obs)} obs")
    print(f"  CES vintage tiers: {n_ces_vintages} ({all_vintages})")

    return dict(
        dates=dates,
        T=T,
        month_of_year=month_of_year,
        year_of_obs=year_of_obs,
        n_years=n_years,
        era_idx=era_idx,
        g_ces_sa=g_ces_sa,
        ces_sa_obs=ces_sa_obs,
        ces_sa_vintage_idx=ces_sa_vintage_idx,
        g_ces_nsa=g_ces_nsa,
        ces_nsa_obs=ces_nsa_obs,
        ces_nsa_vintage_idx=ces_nsa_vintage_idx,
        n_ces_vintages=n_ces_vintages,
        ces_vintage_map=ces_vintage_map,
        g_qcew=g_qcew,
        qcew_obs=qcew_obs,
        qcew_is_m2=qcew_is_m2,
        qcew_noise_mult=qcew_noise_mult,
        g_provider=g_provider,
        provider_obs=provider_obs,
        birth_rate=birth_rate,
        claims_c=claims_c,
        jolts_c=jolts_c,
        levels=levels,
        ces_sa_level=ces_sa_level,
        ces_nsa_level=ces_nsa_level,
    )

