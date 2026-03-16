"""
Precompute calibrated CI for the interactive Streamlit plot.

Runs once locally, writes a small flat CSV (~300 KB) that Streamlit reads
directly — no MC data or model weights needed at runtime.

Key optimisation — quantile equivariance
-----------------------------------------
The calibration transform is a linear rescaling around the sector mean:

    y_cal = mean_s + T_s * (y_mc - mean_s)

Since quantiles are equivariant under monotone linear transforms:

    quantile(y_cal, p) = mean_s + T_s * (quantile(y_mc, p) - mean_s)

For per-country, per-sector CI we therefore compute the raw quantiles
once and apply T analytically — no iteration over 10 000 MC samples.

For the EU27 total (sum of sectors × sum of countries) the quantile must
be taken *after* summing across sectors and countries, so we cannot reduce
to per-sector quantiles.  We handle this with a vectorised numpy pivot
(n_mc × n_geo matrix per sector) — still no Python loop over samples.

Output columns (one row per geo × year)
-----------------------------------------
  geo, year,
  mean_total_Mt,  p05_cal_total_Mt,  p95_cal_total_Mt,
  mean_total_tCO2_cap, p05_cal_total_tCO2_cap, p95_cal_total_tCO2_cap,
  <sector>_mean_Mt, <sector>_p05_cal_Mt, <sector>_p95_cal_Mt,
  <sector>_mean_tCO2_cap, <sector>_p05_cal_tCO2_cap, <sector>_p95_cal_tCO2_cap

Usage:
    python -m scripts.visualization.precompute_interactive_ci

Outputs:
    data/calibration/interactive_ci.csv   (upload this to GDrive)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.utils import load_dataset

# =============================================================================
# Paths & constants
# =============================================================================

MC_PROJECTIONS_PATH = Path("data/projections/mc_projections.csv")
CALIBRATION_TEMPS_PATH = Path("data/calibration/calibration_temperatures.csv")
POPULATION_HIST_PATH = Path("data/full_timeseries/population.csv")
POPULATION_PROJ_PATH = Path("data/full_timeseries/projections/population.csv")
DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")
OUTPUT_PATH = Path("data/calibration/interactive_ci.csv")

OUTPUT_SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]

EU27_COUNTRIES = [
    "AT",
    "BE",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "EL",
    "FI",
    "FR",
    "DE",
    "HU",
    "IE",
    "IT",
    "LV",
    "LT",
    "LU",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
]

PROJECTION_YEARS = list(range(2025, 2031))
P_LO, P_HI = 5, 95


# =============================================================================
# Loading helpers
# =============================================================================


def load_temperatures() -> dict[tuple[str, str], float]:
    df = pd.read_csv(CALIBRATION_TEMPS_PATH)
    return {(row["geo"], row["sector"]): float(row["T"]) for _, row in df.iterrows()}


def load_population() -> pd.DataFrame:
    pop = pd.concat(
        [
            pd.read_csv(POPULATION_HIST_PATH),
            pd.read_csv(POPULATION_PROJ_PATH),
        ],
        ignore_index=True,
    )
    pop["population"] = pop["population:POP_NC"].astype(float)
    return (
        pop[["geo", "year", "population"]]
        .groupby(["geo", "year"], as_index=False)["population"]
        .mean()
    )


def unnorm_mc(df_mc: pd.DataFrame, dataset) -> pd.DataFrame:
    """Add {sector}_phys columns (kg CO2 / hab) to the MC DataFrame."""
    df = df_mc.copy()
    for s in OUTPUT_SECTORS:
        m = dataset.precomputed_scaling_params[s]["mean"]
        sd = dataset.precomputed_scaling_params[s]["std"]
        df[f"{s}_phys"] = (df[f"emissions_{s}"] * sd + m).clip(lower=0)
    return df


# =============================================================================
# Per-country CI  (quantile equivariance — no sample loop)
# =============================================================================


def compute_country_ci(
    yr_df: pd.DataFrame,
    geo: str,
    pop: float,
    temperatures: dict[tuple[str, str], float],
) -> dict:
    """
    For a single (geo, year) compute calibrated CI using quantile equivariance.

    Per-sector:
        q_cal(p) = mean_s + T_s * (q_raw(p) - mean_s)

    For the total we still need to keep the full sample arrays because
    we must take quantile(sum_sectors) not sum(quantile_sectors):
        total_cal_i = sum_s [ mean_s + T_s * (y_i_s - mean_s) ]
    which is O(n_sectors * n_samples) additions — fast with numpy.
    """

    def to_Mt(v_kg_hab):
        return v_kg_hab * pop / 1e6

    def to_tCap(v_kg_hab):
        return v_kg_hab / 1e3

    result: dict = {}
    total_raw = np.zeros(len(yr_df))
    total_cal = np.zeros(len(yr_df))

    for s in OUTPUT_SECTORS:
        vals = yr_df[f"{s}_phys"].values  # (n_mc,) kg/hab
        mean_s = vals.mean()
        T = temperatures.get((geo, s), 1.0)

        # Raw quantiles, then shift analytically — no sample loop
        q_lo_raw = np.percentile(vals, P_LO)
        q_hi_raw = np.percentile(vals, P_HI)
        q_lo_cal = mean_s + T * (q_lo_raw - mean_s)
        q_hi_cal = mean_s + T * (q_hi_raw - mean_s)

        result[f"{s}_mean_Mt"] = round(to_Mt(mean_s), 4)
        result[f"{s}_p05_cal_Mt"] = round(to_Mt(q_lo_cal), 4)
        result[f"{s}_p95_cal_Mt"] = round(to_Mt(q_hi_cal), 4)
        result[f"{s}_mean_tCO2_cap"] = round(to_tCap(mean_s), 6)
        result[f"{s}_p05_cal_tCO2_cap"] = round(to_tCap(q_lo_cal), 6)
        result[f"{s}_p95_cal_tCO2_cap"] = round(to_tCap(q_hi_cal), 6)

        total_raw += vals
        total_cal += mean_s + T * (vals - mean_s)  # calibrated array, kept for total

    result["mean_total_Mt"] = round(to_Mt(total_raw.mean()), 4)
    result["p05_cal_total_Mt"] = round(to_Mt(np.percentile(total_cal, P_LO)), 4)
    result["p95_cal_total_Mt"] = round(to_Mt(np.percentile(total_cal, P_HI)), 4)
    result["mean_total_tCO2_cap"] = round(to_tCap(total_raw.mean()), 6)
    result["p05_cal_total_tCO2_cap"] = round(to_tCap(np.percentile(total_cal, P_LO)), 6)
    result["p95_cal_total_tCO2_cap"] = round(to_tCap(np.percentile(total_cal, P_HI)), 6)

    return result


# =============================================================================
# EU27 aggregate CI  (vectorised pivot — no Python sample loop)
# =============================================================================


def compute_eu27_ci(
    yr_df: pd.DataFrame,
    population_df: pd.DataFrame,
    temperatures: dict[tuple[str, str], float],
    year: int,
) -> dict:
    """
    EU27 aggregate for a single year using a vectorised pivot.

    For each sector s we build a (n_mc × n_geo) matrix, apply T per column,
    multiply by population, and sum across countries.  Summing the per-sector
    results gives the (n_mc,) array of EU27 total calibrated emissions, whose
    percentiles are then taken.  No Python loop over MC samples.
    """
    eu_df = yr_df[yr_df["geo"].isin(EU27_COUNTRIES)]
    pop_yr = population_df[population_df["year"] == year].set_index("geo")["population"]
    total_pop = pop_yr.reindex(EU27_COUNTRIES).sum()

    def to_Mt(kg):
        return kg / 1e6

    def to_tCap(kg):
        return kg / total_pop / 1e3

    result: dict = {}
    eu_total_raw = None
    eu_total_cal = None
    sector_raw_eu: dict[str, np.ndarray] = {}
    sector_cal_eu: dict[str, np.ndarray] = {}

    for s in OUTPUT_SECTORS:
        # Pivot: rows = mc_sample, cols = geo  →  (n_mc × n_geo)
        pivot = eu_df.pivot(
            index="mc_sample", columns="geo", values=f"{s}_phys"
        ).reindex(columns=EU27_COUNTRIES)

        vals_mat = pivot.values  # (n_mc, n_geo)
        means = vals_mat.mean(axis=0)  # (n_geo,)
        Ts = np.array(
            [temperatures.get((g, s), 1.0) for g in EU27_COUNTRIES]
        )  # (n_geo,)
        pops = pop_yr.reindex(EU27_COUNTRIES).values  # (n_geo,)

        # Calibrate: broadcast over n_mc rows
        cal_mat = means + Ts * (vals_mat - means)  # (n_mc, n_geo)

        # Weighted sum across countries → (n_mc,)
        s_raw = (vals_mat * pops).sum(axis=1)
        s_cal = (cal_mat * pops).sum(axis=1)

        sector_raw_eu[s] = s_raw
        sector_cal_eu[s] = s_cal

        eu_total_raw = s_raw if eu_total_raw is None else eu_total_raw + s_raw
        eu_total_cal = s_cal if eu_total_cal is None else eu_total_cal + s_cal

    # Total CI
    result["mean_total_Mt"] = round(to_Mt(eu_total_raw.mean()), 4)
    result["p05_cal_total_Mt"] = round(to_Mt(np.percentile(eu_total_cal, P_LO)), 4)
    result["p95_cal_total_Mt"] = round(to_Mt(np.percentile(eu_total_cal, P_HI)), 4)
    result["mean_total_tCO2_cap"] = round(to_tCap(eu_total_raw.mean()), 6)
    result["p05_cal_total_tCO2_cap"] = round(
        to_tCap(np.percentile(eu_total_cal, P_LO)), 6
    )
    result["p95_cal_total_tCO2_cap"] = round(
        to_tCap(np.percentile(eu_total_cal, P_HI)), 6
    )

    # Per-sector CI
    for s in OUTPUT_SECTORS:
        raw = sector_raw_eu[s]
        cal = sector_cal_eu[s]
        result[f"{s}_mean_Mt"] = round(to_Mt(raw.mean()), 4)
        result[f"{s}_p05_cal_Mt"] = round(to_Mt(np.percentile(cal, P_LO)), 4)
        result[f"{s}_p95_cal_Mt"] = round(to_Mt(np.percentile(cal, P_HI)), 4)
        result[f"{s}_mean_tCO2_cap"] = round(to_tCap(raw.mean()), 6)
        result[f"{s}_p05_cal_tCO2_cap"] = round(to_tCap(np.percentile(cal, P_LO)), 6)
        result[f"{s}_p95_cal_tCO2_cap"] = round(to_tCap(np.percentile(cal, P_HI)), 6)

    return result


# =============================================================================
# Main
# =============================================================================


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("Precomputing calibrated CI for interactive plot")
    print("=" * 60)

    print("\n[1/5] Loading dataset scaling parameters...")
    dataset = load_dataset(DATASET_PATH)
    print("      Done.")

    print("[2/5] Loading calibration temperatures...")
    temperatures = load_temperatures()
    print(f"      {len(temperatures)} (geo, sector) scalars loaded.")

    print("[3/5] Loading population...")
    population_df = load_population()
    print("      Done.")

    print(f"[4/5] Loading MC projections ({MC_PROJECTIONS_PATH})...")
    df_mc = pd.read_csv(MC_PROJECTIONS_PATH)
    df_mc["geo"] = df_mc["geo"].astype(str)
    df_mc = unnorm_mc(df_mc, dataset)
    df_mc = df_mc.merge(population_df, on=["geo", "year"], how="left")
    n_samples = df_mc["mc_sample"].nunique()
    print(
        f"      {len(df_mc):,} rows | {n_samples:,} MC samples | "
        f"{df_mc['year'].nunique()} years | {df_mc['geo'].nunique()} geos"
    )

    df_proj = df_mc[df_mc["year"].isin(PROJECTION_YEARS)]

    # ── Per-country (fast — equivariance, no sample loop) ─────────────────
    print(
        f"\n[5/5] Computing CI for {len(EU27_COUNTRIES)} countries × "
        f"{len(PROJECTION_YEARS)} years + EU27 aggregate..."
    )

    n_cells = len(EU27_COUNTRIES) * len(PROJECTION_YEARS) + len(PROJECTION_YEARS)
    rows = []

    with tqdm(total=n_cells, desc="Progress", unit="cell", ncols=70) as pbar:
        for geo in EU27_COUNTRIES:
            geo_df = df_proj[df_proj["geo"] == geo]
            pop_geo = population_df[population_df["geo"] == geo].set_index("year")[
                "population"
            ]

            for year in PROJECTION_YEARS:
                yr_df = geo_df[geo_df["year"] == year]
                pop = pop_geo.get(year, np.nan)

                pbar.set_postfix(geo=geo, year=year, refresh=False)

                if yr_df.empty or np.isnan(pop):
                    pbar.update(1)
                    continue

                ci = compute_country_ci(yr_df, geo, pop, temperatures)
                rows.append({"geo": geo, "year": year, **ci})
                pbar.update(1)

        # ── EU27 aggregate (vectorised pivot) ─────────────────────────────
        for year in PROJECTION_YEARS:
            pbar.set_postfix(geo="EU27", year=year, refresh=False)
            yr_df = df_proj[df_proj["year"] == year]
            ci = compute_eu27_ci(yr_df, population_df, temperatures, year)
            rows.append({"geo": "EU27", "year": year, **ci})
            pbar.update(1)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_PATH, index=False)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Saved : {OUTPUT_PATH}")
    print(f"Shape : {result_df.shape}")
    print(f"Size  : {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    print(f"Time  : {elapsed:.1f} s")
    print(f"{'=' * 60}")
    print("\nSample (DE, 2030):")
    s = result_df[(result_df["geo"] == "DE") & (result_df["year"] == 2030)]
    if not s.empty:
        cols = ["geo", "year", "mean_total_Mt", "p05_cal_total_Mt", "p95_cal_total_Mt"]
        print(s[cols].to_string(index=False))


if __name__ == "__main__":
    main()
