"""
Figure 2: Sectoral Emission Changes Heatmap (% change, bivariate palette).

Wide heatmap (countries as columns, sectors as rows) showing percentage
change in emissions 2023 → 2030 with a bivariate colour scheme:

  - Hue (red–grey–blue): direction of emission change
      Red = increasing, Grey = near zero, Blue = decreasing
  - Saturation: model confidence (from learned uncertainty)
      High saturation = high confidence, Low saturation = low confidence

The model's per-sector uncertainty score (from the emission predictor's
learned heteroscedastic variance) is averaged across MC samples for each
country–sector pair at 2030. For the EU-27 aggregate, confidence can be
optionally population-weighted.

Confidence is discretised into 3 levels (high/medium/low) based on
user-specified quantile cutoffs, and emission change into 3 levels
(decreasing/near-zero/increasing), giving a 3×3 bivariate colour grid.

Usage:
    python -m scripts.visualization.figure_sectoral_changes

Outputs:
    - outputs/figures/fig2_sectoral_heatmap.pdf
    - outputs/figures/fig2_sectoral_heatmap.png
    - outputs/figures/fig2_sectoral_heatmap.svg
    - outputs/tables/fig2_sectoral_change_data.csv
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")
MC_PROJECTIONS_PATH = Path("data/projections/mc_projections.csv")
POPULATION_HIST_PATH = Path("data/full_timeseries/population.csv")
POPULATION_PROJ_PATH = Path("data/full_timeseries/projections/population.csv")

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


# =============================================================================
# Bivariate colour palette: 3×3
#
#   Rows: emission change direction (decreasing / near-zero / increasing)
#   Cols: model confidence (high / medium / low)
#
# Convention: high saturation = high confidence
# Blue tones for decreasing, grey for near-zero, red for increasing
# =============================================================================

BIVARIATE_COLORS = {
    # (change_level, confidence_level) → hex colour
    # change: -1 = decreasing, 0 = near-zero, +1 = increasing
    # confidence: 2 = high, 1 = medium, 0 = low
    (-1, 2): "#08519c",  # strong decrease, high confidence — deep blue
    (-1, 1): "#6baed6",  # strong decrease, medium confidence — medium blue
    (-1, 0): "#bdd7e7",  # strong decrease, low confidence — pale blue
    (0, 2): "#737373",  # near zero, high confidence — dark grey
    (0, 1): "#bdbdbd",  # near zero, medium confidence — medium grey
    (0, 0): "#e0e0e0",  # near zero, low confidence — light grey
    (1, 2): "#a50f15",  # strong increase, high confidence — deep red
    (1, 1): "#fb6a4a",  # strong increase, medium confidence — medium red
    (1, 0): "#fcbba1",  # strong increase, low confidence — pale red/salmon
}


# =============================================================================
# Style
# =============================================================================


def setup_nature_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.4,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
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
    """
    Compute per-country per-sector:
      - percentage change 2023 → 2030
      - mean model uncertainty at 2030

    Returns DataFrame with columns:
        geo, sector, pct_change, mean_uncertainty
    """
    # --- Historical 2023 ---
    keys = dataset.keys
    emi = pd.DataFrame(dataset.emi_df.cpu().numpy(), columns=OUTPUT_SECTORS)
    hist = pd.concat([keys, emi], axis=1)

    for s in OUTPUT_SECTORS:
        m = dataset.precomputed_scaling_params[s]["mean"]
        sd = dataset.precomputed_scaling_params[s]["std"]
        hist[s] = hist[s] * sd + m

    hist = hist.merge(population_df, on=["geo", "year"], how="left")
    for s in OUTPUT_SECTORS:
        hist[f"{s}_total"] = hist[s] * hist["population"]  # tonnes

    hist_2023 = hist[hist["year"] == 2023].copy()

    # --- MC 2030 ---
    df_mc = pd.read_csv(MC_PROJECTIONS_PATH)
    df_mc["geo"] = df_mc["geo"].astype(str)

    for s in OUTPUT_SECTORS:
        m = dataset.precomputed_scaling_params[s]["mean"]
        sd = dataset.precomputed_scaling_params[s]["std"]
        df_mc[f"{s}_unnorm"] = np.clip(df_mc[f"emissions_{s}"] * sd + m, 0, None)

    df_mc = df_mc.merge(population_df, on=["geo", "year"], how="left")
    for s in OUTPUT_SECTORS:
        df_mc[f"{s}_total"] = df_mc[f"{s}_unnorm"] * df_mc["population"]  # tonnes

    df_2030 = df_mc[df_mc["year"] == 2030].copy()

    # --- Per country, per sector ---
    records = []
    for geo in EU27_COUNTRIES:
        g23 = hist_2023[hist_2023["geo"] == geo]
        g30 = df_2030[df_2030["geo"] == geo]
        if g23.empty or g30.empty:
            continue

        pop_2030 = g30["population"].iloc[0]  # thousands

        for s in OUTPUT_SECTORS:
            e2023 = g23[f"{s}_total"].values[0]
            e2030_mean = g30[f"{s}_total"].mean()

            # % change
            if abs(e2023) > 0:
                pct = ((e2030_mean - e2023) / abs(e2023)) * 100
            else:
                pct = 0.0

            # Mean learned uncertainty (averaged across MC samples)
            unc_col = f"uncertainty_{s}"
            if unc_col in g30.columns:
                mean_unc = g30[unc_col].mean()
            else:
                mean_unc = np.nan

            records.append(
                {
                    "geo": geo,
                    "sector": s,
                    "pct_change": pct,
                    "mean_uncertainty": mean_unc,
                    "emissions_2023": e2023,
                    "emissions_2030": e2030_mean,
                    "population_2030": pop_2030,
                }
            )

    df = pd.DataFrame(records)

    # --- EU27 aggregates ---
    for s in OUTPUT_SECTORS:
        # Sum emissions across countries
        e2023_eu = hist_2023[hist_2023["geo"].isin(EU27_COUNTRIES)][f"{s}_total"].sum()
        eu_mc = df_2030[df_2030["geo"].isin(EU27_COUNTRIES)]
        e2030_eu = eu_mc.groupby("mc_sample")[f"{s}_total"].sum().mean()

        pct_eu = (
            ((e2030_eu - e2023_eu) / abs(e2023_eu)) * 100 if abs(e2023_eu) > 0 else 0
        )

        # Uncertainty for EU: placeholder — will be filled in main()
        # depending on weighting scheme
        records.append(
            {
                "geo": "EU27",
                "sector": s,
                "pct_change": pct_eu,
                "mean_uncertainty": np.nan,  # computed below
                "emissions_2023": e2023_eu,
                "emissions_2030": e2030_eu,
                "population_2030": np.nan,
            }
        )

    return pd.DataFrame(records)


def compute_eu_uncertainty(df, population_weighted):
    """
    Compute EU-27 aggregate uncertainty per sector, optionally
    population-weighted.

    Args:
        df: DataFrame with country-level data (geo, sector, mean_uncertainty,
            population_2030).
        population_weighted: If True, weight each country's uncertainty
            by its 2030 population share.

    Returns:
        dict: {sector: eu27_mean_uncertainty}
    """
    countries = df[df["geo"].isin(EU27_COUNTRIES)].copy()

    eu_unc = {}
    for s in OUTPUT_SECTORS:
        sec = countries[countries["sector"] == s].dropna(subset=["mean_uncertainty"])
        if sec.empty:
            eu_unc[s] = np.nan
            continue

        if population_weighted:
            total_pop = sec["population_2030"].sum()
            if total_pop > 0:
                eu_unc[s] = (
                    sec["mean_uncertainty"] * sec["population_2030"]
                ).sum() / total_pop
            else:
                eu_unc[s] = sec["mean_uncertainty"].mean()
        else:
            eu_unc[s] = sec["mean_uncertainty"].mean()

    return eu_unc


# =============================================================================
# Bivariate classification
# =============================================================================


def classify_change(pct, thresholds):
    """
    Classify % change into -1 (decreasing), 0 (near-zero), +1 (increasing).

    thresholds: (lo, hi) — values between lo and hi are 'near-zero'.
    """
    lo, hi = thresholds
    if pct <= lo:
        return -1
    elif pct >= hi:
        return 1
    else:
        return 0


def classify_confidence(unc_value, cutoffs):
    """
    Classify uncertainty into confidence levels.
    Lower uncertainty = higher confidence.

    cutoffs: [q_low, q_high] — quantile boundaries.
    unc_value <= q_low  → high confidence (2)
    q_low < unc_value <= q_high → medium confidence (1)
    unc_value > q_high → low confidence (0)
    """
    if pd.isna(unc_value):
        return 1  # default to medium if missing
    if unc_value <= cutoffs[0]:
        return 2
    elif unc_value <= cutoffs[1]:
        return 1
    else:
        return 0


# =============================================================================
# Figure
# =============================================================================


def create_heatmap(changes_df, change_thresholds, confidence_cutoffs):
    """
    Wide heatmap: sectors as rows, countries as columns.
    Bivariate colour palette encoding change direction + model confidence.
    """
    setup_nature_style()

    # Pivot tables
    pct_pivot = changes_df.pivot(index="sector", columns="geo", values="pct_change")
    unc_pivot = changes_df.pivot(
        index="sector", columns="geo", values="mean_uncertainty"
    )

    # Add "Overall" row
    overall_pct = changes_df.groupby("geo")["pct_change"].mean()
    overall_unc = changes_df.groupby("geo")["mean_uncertainty"].mean()
    pct_pivot.loc["Overall"] = overall_pct
    unc_pivot.loc["Overall"] = overall_unc

    # Row order: Overall first, then sectors
    row_order = ["Overall"] + SECTOR_ORDER
    pct_pivot = pct_pivot.loc[row_order]
    unc_pivot = unc_pivot.loc[row_order]

    # Column order: EU27 first, then countries sorted alphabetically
    country_alpha = sorted([c for c in pct_pivot.columns if c in EU27_COUNTRIES])
    col_order = ["EU27"] + country_alpha
    col_order = [c for c in col_order if c in pct_pivot.columns]
    pct_pivot = pct_pivot[col_order]
    unc_pivot = unc_pivot[col_order]

    n_rows = len(row_order)
    n_cols = len(col_order)

    # --- Build colour matrix ---
    color_matrix = np.empty((n_rows, n_cols), dtype=object)
    for i, sector in enumerate(row_order):
        for j, geo in enumerate(col_order):
            pct_val = pct_pivot.iloc[i, j]
            unc_val = unc_pivot.iloc[i, j]

            chg_level = classify_change(pct_val, change_thresholds)
            conf_level = classify_confidence(unc_val, confidence_cutoffs)
            color_matrix[i, j] = BIVARIATE_COLORS[(chg_level, conf_level)]

    # --- Plot ---
    # fig_width = max(16, n_cols * 0.42 + 2.5)
    # fig_height = n_rows * 0.55 + 2.0
    # fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig, ax = plt.subplots(figsize=(16, 6))

    # Draw coloured cells as rectangles
    for i in range(n_rows):
        for j in range(n_cols):
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                facecolor=color_matrix[i, j],
                edgecolor="white",
                linewidth=0.8,
            )
            ax.add_patch(rect)

            # Text annotation
            val = pct_pivot.iloc[i, j]
            if pd.notna(val):
                fmt = f"{val:+.0f}" if abs(val) >= 1 else f"{val:+.1f}"
                # Text colour: white on dark cells, black on light
                from matplotlib.colors import to_rgba

                rgb = to_rgba(color_matrix[i, j])[:3]
                lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                txt_color = "white" if lum < 0.55 else "black"
                weight = "bold" if (i == 0 or j == 0) else "regular"
                size = 6 if (i == 0 or j == 0) else 5.5

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

    # Separator lines after "Overall" row and after "EU27" column
    ax.axhline(0.5, color="#333333", linewidth=1.2)
    ax.axvline(0.5, color="#333333", linewidth=1.2)

    # Tick labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [COUNTRY_NAMES.get(g, g) for g in col_order],
        rotation=55,
        ha="right",
        fontsize=6,
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([SECTOR_DISPLAY.get(s, s) for s in row_order], fontsize=7)

    # Bold EU27 and Overall labels
    for label in ax.get_xticklabels():
        if label.get_text() == "EU27":
            label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        if label.get_text() == "Overall":
            label.set_fontweight("bold")

    ax.tick_params(axis="both", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_color("#333333")

    # =====================================================================
    # Bivariate legend (3×3 grid)
    # =====================================================================
    # Position: below the heatmap
    legend_ax = fig.add_axes([0.86, 0.35, 0.10, 0.20])
    legend_ax.set_xlim(-0.5, 2.5)
    legend_ax.set_ylim(-0.5, 2.5)

    change_labels = ["Increasing", "Near zero", "Decreasing"]
    conf_labels = ["Low", "Med", "High"]

    for ci, conf in enumerate([0, 1, 2]):  # x: confidence (low→high)
        for ri, chg in enumerate([1, 0, -1]):  # y: change (increasing at bottom)
            color = BIVARIATE_COLORS[(chg, conf)]
            rect = plt.Rectangle(
                (ci - 0.5, ri - 0.5),
                1,
                1,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
            )
            legend_ax.add_patch(rect)

    legend_ax.set_xticks([0, 1, 2])
    legend_ax.set_xticklabels(conf_labels, fontsize=5)
    legend_ax.set_xlabel("Model confidence", fontsize=5.5, labelpad=2)
    legend_ax.xaxis.set_label_position("bottom")

    legend_ax.set_yticks([0, 1, 2])
    legend_ax.set_yticklabels(change_labels, fontsize=5)
    legend_ax.set_ylabel("Emission change", fontsize=5.5, labelpad=2)

    legend_ax.tick_params(length=0, pad=2)
    for spine in legend_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.3)
        spine.set_color("#666666")

    # Title annotation

    # Main title annotation (values are %)
    ax.set_title(
        "Change in emissions, 2023\u20132030 (%)",
        fontsize=8,
        pad=40,
        fontweight="normal",
        color="#333333",
    )

    plt.subplots_adjust(bottom=0.12, top=0.85, left=0.10, right=0.82)

    # Save
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
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("GENERATING FIGURE 2: SECTORAL HEATMAP (bivariate palette)")
    print("=" * 70)

    # =================================================================
    # USER-CONFIGURABLE PARAMETERS
    # =================================================================

    # Quantile cutoffs for model confidence classification.
    # uncertainty <= q[0] → high confidence
    # q[0] < uncertainty <= q[1] → medium confidence
    # uncertainty > q[1] → low confidence
    CONFIDENCE_QUANTILE_CUTOFFS = [0.34, 0.65]

    # Thresholds for emission change classification (%).
    # change <= lo → decreasing (-1)
    # lo < change < hi → near-zero (0)
    # change >= hi → increasing (+1)
    CHANGE_THRESHOLDS = (-5.0, 5.0)

    # Should EU-wide average confidence be population-weighted?
    EU_CONFIDENCE_POPULATION_WEIGHTED = True

    # =================================================================

    # Load data
    dataset = load_dataset(DATASET_PATH)
    population_df = load_population_data()

    print("Computing sector-level changes and uncertainty...")
    changes_df = compute_sector_data(dataset, population_df)

    # Compute EU27 uncertainty
    eu_unc = compute_eu_uncertainty(changes_df, EU_CONFIDENCE_POPULATION_WEIGHTED)
    weight_label = (
        "population-weighted" if EU_CONFIDENCE_POPULATION_WEIGHTED else "unweighted"
    )
    print(f"EU27 confidence aggregation: {weight_label}")

    # Fill in EU27 uncertainty values
    for s in OUTPUT_SECTORS:
        mask = (changes_df["geo"] == "EU27") & (changes_df["sector"] == s)
        changes_df.loc[mask, "mean_uncertainty"] = eu_unc.get(s, np.nan)

    # Compute confidence cutoffs from the actual uncertainty distribution
    all_unc = changes_df[changes_df["geo"].isin(EU27_COUNTRIES)][
        "mean_uncertainty"
    ].dropna()
    confidence_cutoffs = [
        np.quantile(all_unc, CONFIDENCE_QUANTILE_CUTOFFS[0]),
        np.quantile(all_unc, CONFIDENCE_QUANTILE_CUTOFFS[1]),
    ]
    print(
        f"Confidence cutoffs (quantiles {CONFIDENCE_QUANTILE_CUTOFFS}): {confidence_cutoffs}"
    )
    print(f"Change thresholds: {CHANGE_THRESHOLDS}")

    # Create figure
    print("\nCreating heatmap...")
    create_heatmap(changes_df, CHANGE_THRESHOLDS, confidence_cutoffs)

    # Export data
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    changes_df["country_name"] = changes_df["geo"].map(COUNTRY_NAMES)
    changes_df.to_csv(TABLE_DIR / "fig2_sectoral_change_data.csv", index=False)
    print(f"Exported: {TABLE_DIR}/fig2_sectoral_change_data.csv")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY (% change, 2023 → 2030)")
    print(f"{'=' * 70}")
    print(f"\n{'Sector':<20s} {'EU27':>8s}  {'Mean':>8s}  {'Min':>8s}  {'Max':>8s}")
    print("-" * 56)
    for s in OUTPUT_SECTORS:
        sd = changes_df[changes_df["sector"] == s]
        eu = sd[sd["geo"] == "EU27"]["pct_change"].values
        eu_val = eu[0] if len(eu) > 0 else np.nan
        cd = sd[sd["geo"].isin(EU27_COUNTRIES)]["pct_change"].dropna()
        print(
            f"{SECTOR_DISPLAY.get(s, s).replace(chr(10), ' '):<20s} "
            f"{eu_val:>+7.0f}%  {cd.mean():>+7.0f}%  {cd.min():>+7.0f}%  {cd.max():>+7.0f}%"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
