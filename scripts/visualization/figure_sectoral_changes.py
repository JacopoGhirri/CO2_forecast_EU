"""
Figure 2: Sectoral Emission Changes Heatmap (% change, bivariate palette).

FONT SIZING RATIONALE:
  LaTeX full-width figures display at ~50% native size.
  Cell annotation text must be large enough to survive this scaling.

  font.size base:       18 pt  → ~9 pt in paper
  cell annotation text: 14 pt  → ~7 pt in paper  (min legible)
  axis tick labels:     16 pt  → ~8 pt in paper
  legend axis labels:   14 pt  → ~7 pt in paper
  figure title:         20 pt  → ~10 pt in paper

Usage:
    python -m scripts.visualization.figure_sectoral_changes

Outputs:
    - outputs/figures/fig2_sectoral_heatmap.[pdf|png|svg]
    - outputs/tables/fig2_sectoral_change_data.csv
"""

import pickle
from pathlib import Path
from matplotlib.colors import to_rgba

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")
MC_PROJECTIONS_PATH = Path("data/projections/mc_projections.csv")
POPULATION_HIST_PATH = Path("data/full_timeseries/population.csv")
POPULATION_PROJ_PATH = Path("data/full_timeseries/projections/population.csv")
CALIBRATION_PATH = Path("data/calibration/calibration_temperatures.csv")

OUTPUT_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")

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

COUNTRY_NAMES = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "HR": "Croatia",
    "CY": "Cyprus",
    "CZ": "Czechia",
    "DK": "Denmark",
    "EE": "Estonia",
    "FI": "Finland",
    "FR": "France",
    "DE": "Germany",
    "EL": "Greece",
    "HU": "Hungary",
    "IE": "Ireland",
    "IT": "Italy",
    "LV": "Latvia",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "MT": "Malta",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "SK": "Slovakia",
    "SI": "Slovenia",
    "ES": "Spain",
    "SE": "Sweden",
    "EU27": "EU27",
}

SECTOR_DISPLAY = {
    "Overall": "Overall",
    "HeatingCooling": "Heating &\nCooling",
    "Industry": "Industry",
    "Land": "Land Use",
    "Mobility": "Mobility",
    "Power": "Power",
    "Other": "Other",
}

SECTOR_ORDER = ["HeatingCooling", "Industry", "Land", "Mobility", "Power", "Other"]

BIVARIATE_COLORS = {
    (-1, 2): "#08519c",
    (-1, 1): "#6baed6",
    (-1, 0): "#bdd7e7",
    (0, 2): "#737373",
    (0, 1): "#bdbdbd",
    (0, 0): "#e0e0e0",
    (1, 2): "#a50f15",
    (1, 1): "#fb6a4a",
    (1, 0): "#fcbba1",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# Data loading
# =============================================================================


def load_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_population_data():
    pop_hist = pd.read_csv(POPULATION_HIST_PATH)
    pop_proj = pd.read_csv(POPULATION_PROJ_PATH)
    population_df = pd.concat([pop_hist, pop_proj], ignore_index=True)
    population_df["population"] = population_df["population:POP_NC"].astype(float)
    population_df = population_df[["geo", "year", "population"]]
    return population_df.groupby(["geo", "year"], as_index=False)["population"].mean()


def compute_sector_data(dataset, population_df):
    df_mc = pd.read_csv(MC_PROJECTIONS_PATH)
    df_mc["geo"] = df_mc["geo"].astype(str)
    for s in OUTPUT_SECTORS:
        m = dataset.precomputed_scaling_params[s]["mean"]
        sd = dataset.precomputed_scaling_params[s]["std"]
        df_mc[f"{s}_unnorm"] = np.clip(df_mc[f"emissions_{s}"] * sd + m, 0, None)
    df_mc = df_mc.merge(population_df, on=["geo", "year"], how="left")
    for s in OUTPUT_SECTORS:
        df_mc[f"{s}_total"] = df_mc[f"{s}_unnorm"] * df_mc["population"]

    df_2024 = df_mc[df_mc["year"] == 2024].copy()
    hist_2024 = df_2024.groupby("geo", as_index=False).agg(
        {f"{s}_total": "mean" for s in OUTPUT_SECTORS}
    )
    df_2030 = df_mc[df_mc["year"] == 2030].copy()

    records = []
    for geo in EU27_COUNTRIES:
        g24 = hist_2024[hist_2024["geo"] == geo]
        g30 = df_2030[df_2030["geo"] == geo]
        if g24.empty or g30.empty:
            continue
        pop_2030 = g30["population"].iloc[0]
        for s in OUTPUT_SECTORS:
            e2024 = g24[f"{s}_total"].values[0]
            e2030_mean = g30[f"{s}_total"].mean()
            pct = ((e2030_mean - e2024) / abs(e2024)) * 100 if abs(e2024) > 0 else 0.0
            unc_col = f"uncertainty_{s}"
            mean_unc = g30[unc_col].mean() if unc_col in g30.columns else np.nan
            records.append(
                {
                    "geo": geo,
                    "sector": s,
                    "pct_change": pct,
                    "mean_uncertainty": mean_unc,
                    "emissions_2024": e2024,
                    "emissions_2030": e2030_mean,
                    "population_2030": pop_2030,
                }
            )

    df = pd.DataFrame(records)

    for s in OUTPUT_SECTORS:
        e2024_eu = hist_2024[hist_2024["geo"].isin(EU27_COUNTRIES)][f"{s}_total"].sum()
        eu_mc = df_2030[df_2030["geo"].isin(EU27_COUNTRIES)]
        e2030_eu = eu_mc.groupby("mc_sample")[f"{s}_total"].sum().mean()
        pct_eu = (
            ((e2030_eu - e2024_eu) / abs(e2024_eu)) * 100 if abs(e2024_eu) > 0 else 0
        )
        records.append(
            {
                "geo": "EU27",
                "sector": s,
                "pct_change": pct_eu,
                "mean_uncertainty": np.nan,
                "emissions_2024": e2024_eu,
                "emissions_2030": e2030_eu,
                "population_2030": np.nan,
            }
        )
    return pd.DataFrame(records)


def compute_eu_uncertainty(df, population_weighted):
    countries = df[df["geo"].isin(EU27_COUNTRIES)].copy()
    eu_unc = {}
    for s in OUTPUT_SECTORS:
        sec = countries[countries["sector"] == s].dropna(subset=["mean_uncertainty"])
        if sec.empty:
            eu_unc[s] = np.nan
            continue
        if population_weighted:
            total_pop = sec["population_2030"].sum()
            eu_unc[s] = (
                (sec["mean_uncertainty"] * sec["population_2030"]).sum() / total_pop
                if total_pop > 0
                else sec["mean_uncertainty"].mean()
            )
        else:
            eu_unc[s] = sec["mean_uncertainty"].mean()
    return eu_unc


def classify_change(pct, thresholds):
    lo, hi = thresholds
    return -1 if pct <= lo else (1 if pct >= hi else 0)


def classify_confidence(unc_value, cutoffs):
    if pd.isna(unc_value):
        return 1
    return 2 if unc_value <= cutoffs[0] else (1 if unc_value <= cutoffs[1] else 0)


# =============================================================================
# Figure
# =============================================================================


def create_heatmap(changes_df, change_thresholds, confidence_cutoffs):
    setup_style()

    pct_pivot = changes_df.pivot(index="sector", columns="geo", values="pct_change")
    unc_pivot = changes_df.pivot(
        index="sector", columns="geo", values="mean_uncertainty"
    )

    total_2024 = changes_df.groupby("geo")["emissions_2024"].sum()
    total_2030 = changes_df.groupby("geo")["emissions_2030"].sum()
    overall_pct = (total_2030 - total_2024) / total_2024.abs() * 100
    pct_pivot.loc["Overall"] = overall_pct
    unc_pivot.loc["Overall"] = changes_df.groupby("geo")["mean_uncertainty"].mean()

    row_order = ["Overall"] + SECTOR_ORDER
    pct_pivot = pct_pivot.loc[row_order]
    unc_pivot = unc_pivot.loc[row_order]

    country_alpha = sorted([c for c in pct_pivot.columns if c in EU27_COUNTRIES])
    col_order = [c for c in (["EU27"] + country_alpha) if c in pct_pivot.columns]
    pct_pivot = pct_pivot[col_order]
    unc_pivot = unc_pivot[col_order]

    n_rows = len(row_order)
    n_cols = len(col_order)

    # Build colour matrix
    color_matrix = np.empty((n_rows, n_cols), dtype=object)
    for i, sector in enumerate(row_order):
        for j, geo in enumerate(col_order):
            chg_level = classify_change(pct_pivot.iloc[i, j], change_thresholds)
            conf_level = classify_confidence(unc_pivot.iloc[i, j], confidence_cutoffs)
            color_matrix[i, j] = BIVARIATE_COLORS[(chg_level, conf_level)]

    # Larger figure for a heatmap with 28 columns and 7 rows + big fonts
    fig, ax = plt.subplots(figsize=(26, 9))

    for i in range(n_rows):
        for j in range(n_cols):
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                facecolor=color_matrix[i, j],
                edgecolor="white",
                linewidth=1.1,
            )
            ax.add_patch(rect)

            val = pct_pivot.iloc[i, j]
            if pd.notna(val):
                fmt = f"{val:+.0f}" if abs(val) >= 1 else f"{val:+.1f}"
                rgb = to_rgba(color_matrix[i, j])[:3]
                lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                txt_color = "white" if lum < 0.55 else "black"
                weight = "bold" if (i == 0 or j == 0) else "regular"
                size = 14 if (i == 0 or j == 0) else 13
                ax.text(
                    j,
                    i,
                    fmt,
                    ha="center",
                    va="center",
                    fontsize=size,
                    color=txt_color,
                    fontweight=weight,
                )

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    # Bold separator lines
    ax.axhline(0.5, color="#333333", linewidth=2.0)
    ax.axvline(0.5, color="#333333", linewidth=2.0)

    # Country labels on top, 45° — large font
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [COUNTRY_NAMES.get(g, g) for g in col_order],
        rotation=45,
        ha="left",
        fontsize=15,
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([SECTOR_DISPLAY.get(s, s) for s in row_order], fontsize=16)

    for label in ax.get_xticklabels():
        if label.get_text() == "EU27":
            label.set_fontweight("bold")
            label.set_fontsize(16)
    for label in ax.get_yticklabels():
        if label.get_text() == "Overall":
            label.set_fontweight("bold")

    ax.tick_params(axis="both", length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#333333")

    ax.set_title(
        "Change in emissions, 2024–2030 (%)",
        fontsize=20,
        pad=8,
        fontweight="normal",
        color="#333333",
    )

    # Leave extra room at bottom for the legend
    plt.subplots_adjust(bottom=0.18, top=0.82, left=0.08, right=0.97)

    # ── Bivariate legend — bottom-right, below the heatmap ───────────────
    # Position in figure coords: right-aligned, just below the heatmap
    legend_ax = fig.add_axes([0.40, 0.01, 0.21, 0.15])
    legend_ax.set_xlim(-0.5, 2.5)
    legend_ax.set_ylim(-0.5, 2.5)

    for ci, conf in enumerate([0, 1, 2]):
        for ri, chg in enumerate([1, 0, -1]):
            rect = plt.Rectangle(
                (ci - 0.5, ri - 0.5),
                1,
                1,
                facecolor=BIVARIATE_COLORS[(chg, conf)],
                edgecolor="white",
                linewidth=0.8,
            )
            legend_ax.add_patch(rect)

    legend_ax.set_xticks([0, 1, 2])
    legend_ax.set_xticklabels(["Low", "Med", "High"], fontsize=13)
    legend_ax.set_xlabel("Model confidence", fontsize=14, labelpad=4)
    legend_ax.xaxis.set_label_position("bottom")

    legend_ax.set_yticks([0, 1, 2])
    legend_ax.set_yticklabels(["Increasing", "Near zero", "Decreasing"], fontsize=13)
    legend_ax.set_ylabel("Emission\nchange", fontsize=14, labelpad=4)

    legend_ax.tick_params(length=0, pad=3)
    for spine in legend_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#666666")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(
            OUTPUT_DIR / f"fig2_sectoral_heatmap.{fmt}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR}/fig2_sectoral_heatmap.[png|pdf|svg]")


# =============================================================================
# Calibrated CI helpers
# =============================================================================


def _load_temperatures() -> dict[tuple[str, str], float]:
    """Load (geo, sector) -> T from calibration_temperatures.csv."""
    if not CALIBRATION_PATH.exists():
        print(f"[WARNING] {CALIBRATION_PATH} not found — using T=1 (no calibration).")
        return {}
    df = pd.read_csv(CALIBRATION_PATH)
    return {(row["geo"], row["sector"]): float(row["T"]) for _, row in df.iterrows()}


def _calibrated_sector_samples(
    df_mc: pd.DataFrame,
    dataset,
    geos: list,
    year: int,
    temperatures: dict,
    sector: str,
) -> np.ndarray:
    """
    Return a (n_mc,) array of calibrated per-sector total CO2 (Mt) for a
    group of countries at a given year.

    Calibration:  y_cal = mean_s + T_{geo,s} * (y_mc - mean_s)
    Physical units: kg CO2 / hab → multiplied by population → Mt.
    """
    df_year = df_mc[(df_mc["year"] == year) & (df_mc["geo"].isin(geos))].copy()
    if df_year.empty:
        return np.array([])

    mc_samples = sorted(df_year["mc_sample"].unique())

    m = dataset.precomputed_scaling_params[sector]["mean"]
    sd = dataset.precomputed_scaling_params[sector]["std"]
    df_year[f"{sector}_phys"] = (df_year[f"emissions_{sector}"] * sd + m).clip(lower=0)

    pivot = df_year.pivot(
        index="mc_sample", columns="geo", values=f"{sector}_phys"
    ).reindex(index=mc_samples, columns=geos)
    vals = pivot.values  # (n_mc, n_geo)
    means = np.nanmean(vals, axis=0)
    Ts = np.array([temperatures.get((g, sector), 1.0) for g in geos])
    pops = np.array(
        [
            df_year.loc[df_year["geo"] == g, "population"].iloc[0]
            if (df_year["geo"] == g).any()
            else 0.0
            for g in geos
        ]
    )

    cal = means + Ts * (vals - means)  # (n_mc, n_geo)
    totals = (cal * pops).sum(axis=1)  # kg CO2 total
    return totals / 1e6  # → Mt


def print_calibrated_ci(dataset, population_df):
    """
    Append 90 % calibrated CI for all sector × country/EU27 combinations
    shown in Figure 2.

    For each sector s and each member state / EU27 aggregate:
      - load MC samples
      - apply calibration:  y_cal = mean_s + T_{geo,s} * (y_mc - mean_s)
      - report [5th, 95th] percentile of the calibrated distribution
      - also express as % change from 2024 baseline

    Additionally reports the calibrated CI on the *overall* (all-sector)
    total for EU27 and per country, matching the "Overall" row in the heatmap.
    """
    print("\n" + "=" * 75)
    print("90 % CALIBRATED CI — FIGURE 2 SECTORAL CHANGES (Mt CO2, 2030)")
    print("=" * 75)
    print("  Calibration: y_cal = mean_s + T_{geo,s} * (y_mc - mean_s)")
    print("  Interval: [5th percentile, 95th percentile]")
    print()

    temperatures = _load_temperatures()
    df_mc = pd.read_csv(MC_PROJECTIONS_PATH)
    df_mc["geo"] = df_mc["geo"].astype(str)
    df_mc = df_mc.merge(population_df, on=["geo", "year"], how="left")

    # Pre-compute 2024 baselines per (geo, sector)
    df_2024 = df_mc[df_mc["year"] == 2024].copy()
    for s in OUTPUT_SECTORS:
        m = dataset.precomputed_scaling_params[s]["mean"]
        sd = dataset.precomputed_scaling_params[s]["std"]
        df_2024[f"{s}_phys"] = (df_2024[f"emissions_{s}"] * sd + m).clip(lower=0)
        df_2024[f"{s}_total"] = df_2024[f"{s}_phys"] * df_2024["population"]

    # ── Per-sector × per-country table ────────────────────────────────────
    for s in OUTPUT_SECTORS:
        label = SECTOR_DISPLAY.get(s, s).replace("\n", " ")
        print(f"  Sector: {label}")
        print(
            f"  {'Country':<16s} {'ISO2':>4s}  {'mean Mt':>8s}  "
            f"{'p05 (5%)':>10s}  {'p95 (95%)':>10s}  "
            f"{'Δ mean %':>9s}  {'Δ p05 %':>9s}  {'Δ p95 %':>9s}"
        )
        print("  " + "-" * 80)

        # EU27 aggregate for this sector
        eu_samp = _calibrated_sector_samples(
            df_mc, dataset, EU27_COUNTRIES, 2030, temperatures, s
        )
        eu_2024_val = (
            df_2024[df_2024["geo"].isin(EU27_COUNTRIES)][f"{s}_total"].sum() / 1e6
        )
        if eu_samp.size and eu_2024_val > 0:
            lo, hi = np.percentile(eu_samp, 5), np.percentile(eu_samp, 95)
            m_pct = (eu_samp.mean() - eu_2024_val) / eu_2024_val * 100
            lo_pct = (lo - eu_2024_val) / eu_2024_val * 100
            hi_pct = (hi - eu_2024_val) / eu_2024_val * 100
            print(
                f"  {'EU27':<16s} {'':>4s}  {eu_samp.mean():8.1f}  "
                f"{lo:10.1f}  {hi:10.1f}  "
                f"{m_pct:+9.1f}%  {lo_pct:+9.1f}%  {hi_pct:+9.1f}%"
            )

        for geo in sorted(EU27_COUNTRIES):
            samp = _calibrated_sector_samples(
                df_mc, dataset, [geo], 2030, temperatures, s
            )
            base_val = df_2024[df_2024["geo"] == geo][f"{s}_total"].sum() / 1e6
            if samp.size == 0 or base_val <= 0:
                continue
            lo, hi = np.percentile(samp, 5), np.percentile(samp, 95)
            m_pct = (samp.mean() - base_val) / base_val * 100
            lo_pct = (lo - base_val) / base_val * 100
            hi_pct = (hi - base_val) / base_val * 100
            name = COUNTRY_NAMES.get(geo, geo)
            print(
                f"  {name:<16s} {geo:>4s}  {samp.mean():8.1f}  "
                f"{lo:10.1f}  {hi:10.1f}  "
                f"{m_pct:+9.1f}%  {lo_pct:+9.1f}%  {hi_pct:+9.1f}%"
            )
        print()

    # ── Overall (all sectors summed) CI ───────────────────────────────────
    print(f"  Overall (all sectors combined):")
    print(
        f"  {'Country':<16s} {'ISO2':>4s}  {'mean Mt':>8s}  "
        f"{'p05 (5%)':>10s}  {'p95 (95%)':>10s}  "
        f"{'Δ mean %':>9s}  {'Δ p05 %':>9s}  {'Δ p95 %':>9s}"
    )
    print("  " + "-" * 80)

    # EU27 overall
    eu_total_samp = np.zeros(0)
    for s in OUTPUT_SECTORS:
        samp = _calibrated_sector_samples(
            df_mc, dataset, EU27_COUNTRIES, 2030, temperatures, s
        )
        if eu_total_samp.size == 0:
            eu_total_samp = samp
        elif samp.size == eu_total_samp.size:
            eu_total_samp = eu_total_samp + samp

    eu_2024_total = (
        df_2024[df_2024["geo"].isin(EU27_COUNTRIES)][
            [f"{s}_total" for s in OUTPUT_SECTORS]
        ]
        .sum()
        .sum()
        / 1e6
    )
    if eu_total_samp.size and eu_2024_total > 0:
        lo, hi = np.percentile(eu_total_samp, 5), np.percentile(eu_total_samp, 95)
        m_pct = (eu_total_samp.mean() - eu_2024_total) / eu_2024_total * 100
        lo_pct = (lo - eu_2024_total) / eu_2024_total * 100
        hi_pct = (hi - eu_2024_total) / eu_2024_total * 100
        print(
            f"  {'EU27':<16s} {'':>4s}  {eu_total_samp.mean():8.1f}  "
            f"{lo:10.1f}  {hi:10.1f}  "
            f"{m_pct:+9.1f}%  {lo_pct:+9.1f}%  {hi_pct:+9.1f}%"
        )

    for geo in sorted(EU27_COUNTRIES):
        geo_total = np.zeros(0)
        for s in OUTPUT_SECTORS:
            samp = _calibrated_sector_samples(
                df_mc, dataset, [geo], 2030, temperatures, s
            )
            if geo_total.size == 0:
                geo_total = samp
            elif samp.size == geo_total.size:
                geo_total = geo_total + samp

        geo_2024_total = (
            df_2024[df_2024["geo"] == geo][[f"{s}_total" for s in OUTPUT_SECTORS]]
            .sum()
            .sum()
            / 1e6
        )
        if geo_total.size == 0 or geo_2024_total <= 0:
            continue
        lo, hi = np.percentile(geo_total, 5), np.percentile(geo_total, 95)
        m_pct = (geo_total.mean() - geo_2024_total) / geo_2024_total * 100
        lo_pct = (lo - geo_2024_total) / geo_2024_total * 100
        hi_pct = (hi - geo_2024_total) / geo_2024_total * 100
        name = COUNTRY_NAMES.get(geo, geo)
        print(
            f"  {name:<16s} {geo:>4s}  {geo_total.mean():8.1f}  "
            f"{lo:10.1f}  {hi:10.1f}  "
            f"{m_pct:+9.1f}%  {lo_pct:+9.1f}%  {hi_pct:+9.1f}%"
        )

    print()


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("GENERATING FIGURE 2: SECTORAL HEATMAP")
    print("=" * 70)

    CONFIDENCE_QUANTILE_CUTOFFS = [0.34, 0.65]
    CHANGE_THRESHOLDS = (-5.0, 5.0)
    EU_CONFIDENCE_POPULATION_WEIGHTED = True

    dataset = load_dataset(DATASET_PATH)
    population_df = load_population_data()

    changes_df = compute_sector_data(dataset, population_df)
    eu_unc = compute_eu_uncertainty(changes_df, EU_CONFIDENCE_POPULATION_WEIGHTED)
    for s in OUTPUT_SECTORS:
        mask = (changes_df["geo"] == "EU27") & (changes_df["sector"] == s)
        changes_df.loc[mask, "mean_uncertainty"] = eu_unc.get(s, np.nan)

    all_unc = changes_df[changes_df["geo"].isin(EU27_COUNTRIES)][
        "mean_uncertainty"
    ].dropna()
    confidence_cutoffs = [
        np.quantile(all_unc, CONFIDENCE_QUANTILE_CUTOFFS[0]),
        np.quantile(all_unc, CONFIDENCE_QUANTILE_CUTOFFS[1]),
    ]
    print(f"Confidence cutoffs: {confidence_cutoffs}")

    create_heatmap(changes_df, CHANGE_THRESHOLDS, confidence_cutoffs)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY (% change, 2024 ->2030)")
    print(f"{'=' * 70}")
    print(f"\n{'Sector':<20s} {'EU27':>8s}  {'Mean':>8s}  {'Min':>8s}  {'Max':>8s}")
    print("-" * 56)
    for s in OUTPUT_SECTORS:
        sd = changes_df[changes_df["sector"] == s]
        eu = sd[sd["geo"] == "EU27"]["pct_change"].values
        eu_v = eu[0] if len(eu) > 0 else np.nan
        cd = sd[sd["geo"].isin(EU27_COUNTRIES)]["pct_change"].dropna()
        print(
            f"{SECTOR_DISPLAY.get(s, s).replace(chr(10), ' '):<20s} "
            f"{eu_v:>+7.0f}%  {cd.mean():>+7.0f}%  "
            f"{cd.min():>+7.0f}%  {cd.max():>+7.0f}%"
        )

    # ── Numbers for paper Section: "Uneven progress" (heatmap paragraph) ──
    print("\n" + "=" * 60)
    print("NUMBERS FOR PAPER — Section: Sectoral heatmap paragraph")
    print("=" * 60)

    country_data = changes_df[changes_df["geo"].isin(EU27_COUNTRIES)]

    # Power sector: how many countries exceed various reduction thresholds
    power = country_data[country_data["sector"] == "Power"]["pct_change"]
    for thresh in [-30, -40, -50]:
        n = (power < thresh).sum()
        print(f"  Power: countries with >{abs(thresh)}% reduction: {n}")

    # Mobility: how many show projected increases
    mob = country_data[country_data["sector"] == "Mobility"]["pct_change"]
    n_mob_increase = (mob > 0).sum()
    print(f"\n  Mobility: countries with projected INCREASE: {n_mob_increase}")

    # Heating & Cooling: range
    hc = country_data[country_data["sector"] == "HeatingCooling"]["pct_change"]
    print(f"\n  Heating & Cooling % change range: {hc.min():+.1f}% to {hc.max():+.1f}%")

    # Overall EU27 row
    eu27_data = changes_df[changes_df["geo"] == "EU27"]
    print("\n  EU27 sector % changes (2024→2030):")
    for s in OUTPUT_SECTORS:
        row = eu27_data[eu27_data["sector"] == s]
        if not row.empty:
            print(f"    {s:<16s}: {row['pct_change'].values[0]:+.1f}%")

    # ── 90 % calibrated CI for all sector × country combinations ──────
    print_calibrated_ci(dataset, population_df)

    print("Done!")


if __name__ == "__main__":
    main()
