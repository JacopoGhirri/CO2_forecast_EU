"""
Figure 2: Sectoral Emission Changes Heatmap.

Visualizes the projected percentage change in emissions by sector for each
EU27 member state from 2023 to 2030. Displays both a stacked bar chart
and a heatmap showing country-sector decomposition of emission trajectories.

Usage:
    python -m scripts.visualization.figure_sectoral_changes

Outputs:
    - outputs/figures/fig2_sectoral_change_heatmap.pdf
    - outputs/figures/fig2_sectoral_change_heatmap.png
    - outputs/figures/fig2_sectoral_change_stacked_bar.pdf
    - outputs/tables/fig2_sectoral_change_data.csv

Reference:
    Figure 2 in the paper shows sectoral emission changes across EU27.
"""

import pickle
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

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

SECTOR_CONFIG = {
    "HeatingCooling": {"name": "Heating & Cooling", "color": "#E69F00"},
    "Industry": {"name": "Industry", "color": "#56B4E9"},
    "Land": {"name": "Land Use", "color": "#009E73"},
    "Mobility": {"name": "Mobility", "color": "#F0E442"},
    "Other": {"name": "Other", "color": "#0072B2"},
    "Power": {"name": "Power", "color": "#D55E00"},
}


def setup_nature_style():
    """Configure matplotlib for Nature Climate Change publication style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.4,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_dataset(path: Path):
    """Load pickled dataset."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_population_data() -> pd.DataFrame:
    """Load and combine historical and projected population data."""
    pop_hist = pd.read_csv(POPULATION_HIST_PATH)
    pop_proj = pd.read_csv(POPULATION_PROJ_PATH)
    population_df = pd.concat([pop_hist, pop_proj], ignore_index=True)
    population_df["population"] = population_df["population:POP_NC"].astype(float)
    population_df = population_df[["geo", "year", "population"]]
    population_df = population_df.groupby(["geo", "year"], as_index=False)[
        "population"
    ].mean()
    return population_df


def compute_sector_changes(dataset, population_df: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage change in emissions by sector from 2023 to 2030."""
    keys = dataset.keys
    emi_dataset = pd.DataFrame(dataset.emi_df.cpu().numpy(), columns=OUTPUT_SECTORS)
    historical = pd.concat([keys, emi_dataset], axis=1)

    for s in OUTPUT_SECTORS:
        mean_ = dataset.precomputed_scaling_params[s]["mean"]
        std_ = dataset.precomputed_scaling_params[s]["std"]
        historical[s] = historical[s] * std_ + mean_

    historical = historical.merge(population_df, on=["geo", "year"], how="left")
    for s in OUTPUT_SECTORS:
        historical[f"{s}_total"] = historical[s] * historical["population"]

    df_mc = pd.read_csv(MC_PROJECTIONS_PATH)
    df_mc["geo"] = df_mc["geo"].astype(str)

    for s in OUTPUT_SECTORS:
        mean_ = dataset.precomputed_scaling_params[s]["mean"]
        std_ = dataset.precomputed_scaling_params[s]["std"]
        df_mc[f"{s}_unnorm"] = np.clip(df_mc[f"emissions_{s}"] * std_ + mean_, 0, None)

    df_mc = df_mc.merge(population_df, on=["geo", "year"], how="left")
    for s in OUTPUT_SECTORS:
        df_mc[f"{s}_total"] = df_mc[f"{s}_unnorm"] * df_mc["population"]

    hist_2023 = historical[historical["year"] == 2023].copy()
    df_2030 = df_mc[df_mc["year"] == 2030].copy()

    sector_changes = []
    for geo in EU27_COUNTRIES:
        geo_2023 = hist_2023[hist_2023["geo"] == geo]
        geo_2030 = df_2030[df_2030["geo"] == geo]
        if len(geo_2023) == 0 or len(geo_2030) == 0:
            continue
        for s in OUTPUT_SECTORS:
            e2023 = geo_2023[f"{s}_total"].values[0]
            e2030_mean = geo_2030[f"{s}_total"].mean()
            e2030_low = geo_2030[f"{s}_total"].quantile(0.05)
            e2030_high = geo_2030[f"{s}_total"].quantile(0.95)
            if e2023 > 0:
                pct = ((e2030_mean - e2023) / e2023) * 100
                pct_low = ((e2030_low - e2023) / e2023) * 100
                pct_high = ((e2030_high - e2023) / e2023) * 100
            else:
                pct = pct_low = pct_high = np.nan
            sector_changes.append(
                {
                    "geo": geo,
                    "sector": s,
                    "emissions_2023": e2023,
                    "emissions_2030": e2030_mean,
                    "pct_change": pct,
                    "pct_change_low": pct_low,
                    "pct_change_high": pct_high,
                }
            )

    hist_eu = hist_2023[hist_2023["geo"].isin(EU27_COUNTRIES)]
    fcast_eu = df_2030[df_2030["geo"].isin(EU27_COUNTRIES)]
    for s in OUTPUT_SECTORS:
        e2023 = hist_eu[f"{s}_total"].sum()
        eu_by_mc = fcast_eu.groupby("mc_sample")[f"{s}_total"].sum()
        e2030_mean = eu_by_mc.mean()
        e2030_low = eu_by_mc.quantile(0.05)
        e2030_high = eu_by_mc.quantile(0.95)
        if e2023 > 0:
            pct = ((e2030_mean - e2023) / e2023) * 100
            pct_low = ((e2030_low - e2023) / e2023) * 100
            pct_high = ((e2030_high - e2023) / e2023) * 100
        else:
            pct = pct_low = pct_high = np.nan
        sector_changes.append(
            {
                "geo": "EU27",
                "sector": s,
                "emissions_2023": e2023,
                "emissions_2030": e2030_mean,
                "pct_change": pct,
                "pct_change_low": pct_low,
                "pct_change_high": pct_high,
            }
        )

    return pd.DataFrame(sector_changes)


def create_heatmap(changes_df: pd.DataFrame) -> None:
    """Create the sectoral changes heatmap."""
    setup_nature_style()
    sector_order = ["HeatingCooling", "Industry", "Land", "Mobility", "Power", "Other"]
    sector_display = {
        "Overall": "Overall",
        "HeatingCooling": "Heating &\nCooling",
        "Industry": "Industry",
        "Land": "Land Use",
        "Mobility": "Mobility",
        "Power": "Power",
        "Other": "Other",
    }

    heatmap_data = changes_df.pivot(index="geo", columns="sector", values="pct_change")
    overall_changes = []
    for geo in heatmap_data.index:
        geo_rows = changes_df[changes_df["geo"] == geo]
        t2023 = geo_rows["emissions_2023"].sum()
        t2030 = geo_rows["emissions_2030"].sum()
        overall = ((t2030 - t2023) / t2023) * 100 if t2023 > 0 else np.nan
        overall_changes.append({"geo": geo, "Overall": overall})
    overall_df = pd.DataFrame(overall_changes).set_index("geo")
    heatmap_data = pd.concat([overall_df, heatmap_data], axis=1)
    column_order = ["Overall"] + sector_order
    heatmap_data = heatmap_data[column_order]
    country_codes_alpha = sorted([c for c in heatmap_data.index if c != "EU27"])
    row_order = ["EU27"] + country_codes_alpha
    heatmap_data = heatmap_data.loc[row_order]

    fig, ax = plt.subplots(figsize=(4.5, 7.0))
    vabs = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    im = ax.imshow(heatmap_data.values, cmap=plt.cm.RdBu_r, norm=norm, aspect="auto")

    for i in range(len(row_order)):
        for j in range(len(column_order)):
            val = heatmap_data.iloc[i, j]
            if pd.notna(val):
                norm_val = norm(val)
                color = "white" if (norm_val < 0.25 or norm_val > 0.75) else "black"
                weight = "bold" if (j == 0 or i == 0) else "regular"
                size = 5.5 if (j == 0 or i == 0) else 5
                ax.text(
                    j,
                    i,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=size,
                    color=color,
                    fontweight=weight,
                )

    ax.axvline(x=0.5, color="white", linewidth=1.5)
    ax.axhline(y=0.5, color="white", linewidth=1.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xticks(range(len(column_order)))
    ax.set_xticklabels([sector_display.get(s, s) for s in column_order], fontsize=6.5)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([COUNTRY_NAMES.get(g, g) for g in row_order], fontsize=6.5)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.tick_params(axis="both", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_color("#333333")

    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Change in emissions, 2023–2030 (%)", fontsize=7, labelpad=4)
    cbar.ax.tick_params(labelsize=6, length=2, width=0.4)
    cbar.outline.set_linewidth(0.4)
    cbar.set_ticks([-60, -40, -20, 0, 20, 40, 60])
    plt.subplots_adjust(bottom=0.12, top=0.92, left=0.22, right=0.98)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ["png", "pdf", "svg"]:
        plt.savefig(
            OUTPUT_DIR / f"fig2_sectoral_change_heatmap.{fmt}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/fig2_sectoral_change_heatmap.[png|pdf|svg]")


def create_stacked_bar(changes_df: pd.DataFrame) -> None:
    """Create the stacked bar chart of sectoral changes."""
    setup_nature_style()
    pivot = changes_df.pivot(index="geo", columns="sector", values="pct_change")
    overall = pivot.sum(axis=1)
    country_order = ["EU27"] + overall.drop("EU27").sort_values().index.tolist()
    pivot = pivot.loc[country_order]

    fig, ax = plt.subplots(figsize=(6.0, 7.5))
    ax.axvspan(-200, 0, color="#eafaf1", alpha=0.7, zorder=0)
    ax.axvspan(0, 100, color="#fdedec", alpha=0.7, zorder=0)

    bar_height = 0.7
    for i, geo in enumerate(country_order):
        left_neg, left_pos = 0, 0
        for sector in OUTPUT_SECTORS:
            val = pivot.loc[geo, sector]
            if pd.isna(val):
                continue
            color = SECTOR_CONFIG[sector]["color"]
            if val < 0:
                ax.barh(
                    i,
                    val,
                    height=bar_height,
                    left=left_neg,
                    color=color,
                    edgecolor="white",
                    linewidth=0.3,
                    zorder=3,
                )
                left_neg += val
            else:
                ax.barh(
                    i,
                    val,
                    height=bar_height,
                    left=left_pos,
                    color=color,
                    edgecolor="white",
                    linewidth=0.3,
                    zorder=3,
                )
                left_pos += val

    ax.axvline(0, color="#2c3e50", linewidth=1.5, zorder=10)
    ax.set_yticks(range(len(country_order)))
    ax.set_yticklabels([COUNTRY_NAMES.get(g, g) for g in country_order])
    for label in ax.get_yticklabels():
        if label.get_text() == "EU27":
            label.set_fontweight("bold")
            label.set_fontsize(8)
    ax.set_xlim(-120, 80)
    ax.set_xlabel("Cumulative change in emissions 2023→2030 (%)")
    ax.text(
        -60,
        -1.5,
        "← Net decrease",
        fontsize=6.5,
        ha="center",
        color="#1e8449",
        fontstyle="italic",
    )
    ax.text(
        40,
        -1.5,
        "Net increase →",
        fontsize=6.5,
        ha="center",
        color="#922b21",
        fontstyle="italic",
    )
    ax.xaxis.grid(True, linestyle="-", alpha=0.2, color="#bdc3c7", linewidth=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    legend_elements = [
        mpatches.Patch(
            facecolor=SECTOR_CONFIG[s]["color"],
            edgecolor="white",
            linewidth=0.3,
            label=SECTOR_CONFIG[s]["name"],
        )
        for s in OUTPUT_SECTORS
    ]
    leg = ax.legend(
        handles=legend_elements,
        loc="lower right",
        frameon=True,
        fontsize=6,
        framealpha=0.95,
        edgecolor="#bdc3c7",
        handletextpad=0.4,
        labelspacing=0.35,
        borderpad=0.5,
        title="Sector",
        title_fontsize=7,
    )
    leg.get_frame().set_linewidth(0.4)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color("#5d6d7e")
    plt.tight_layout()

    for fmt in ["png", "pdf", "svg"]:
        plt.savefig(
            OUTPUT_DIR / f"fig2_sectoral_change_stacked_bar.{fmt}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/fig2_sectoral_change_stacked_bar.[png|pdf|svg]")


def main():
    """Generate Figure 2: Sectoral Changes Visualizations."""
    print("=" * 70)
    print("GENERATING FIGURE 2: SECTORAL EMISSION CHANGES")
    print("=" * 70)

    print("\nLoading data...")
    dataset = load_dataset(DATASET_PATH)
    population_df = load_population_data()

    print("Computing sector-wise changes...")
    changes_df = compute_sector_changes(dataset, population_df)

    print("\nCreating visualizations...")
    create_heatmap(changes_df)
    create_stacked_bar(changes_df)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    changes_df["country_name"] = changes_df["geo"].map(COUNTRY_NAMES)
    changes_df["sector_name"] = changes_df["sector"].map(
        lambda x: SECTOR_CONFIG[x]["name"]
    )
    export_cols = [
        "geo",
        "country_name",
        "sector",
        "sector_name",
        "emissions_2023",
        "emissions_2030",
        "pct_change",
        "pct_change_low",
        "pct_change_high",
    ]
    changes_df[export_cols].to_csv(
        TABLE_DIR / "fig2_sectoral_change_data.csv", index=False
    )
    print(f"Exported: {TABLE_DIR}/fig2_sectoral_change_data.csv")

    print("\n" + "=" * 70)
    print("SUMMARY (2023 → 2030)")
    print("=" * 70)
    print(f"\n{'Sector':<20} {'EU27':>10} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    for sector in OUTPUT_SECTORS:
        sd = changes_df[changes_df["sector"] == sector]
        eu = sd[sd["geo"] == "EU27"]["pct_change"].values[0]
        cd = sd[sd["geo"] != "EU27"]["pct_change"].dropna()
        print(
            f"{SECTOR_CONFIG[sector]['name']:<20} {eu:>+9.1f}% {cd.mean():>+9.1f}% "
            f"{cd.min():>+9.1f}% {cd.max():>+9.1f}%"
        )
    print("\nFigure 2 generation complete!")


if __name__ == "__main__":
    main()
