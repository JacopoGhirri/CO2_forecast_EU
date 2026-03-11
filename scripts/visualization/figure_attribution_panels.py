"""
Figure 3 (Revised): Attribution Panels + 2030 Boxplots.

Four-panel figure:
  (a) Historical + projected stacked area by country/region (unchanged)
  (b) Historical + projected stacked area by sector (unchanged)
  (c) NEW: Boxplots of 2030 MC distribution by country/region (Gt CO2)
  (d) NEW: Boxplots of 2030 MC distribution by sector (Gt CO2)

Panels (c) and (d) show the full Monte Carlo uncertainty for the 2030
projected emissions, decomposed by the same groupings as panels (a)/(b).

Usage:
    python figure_attribution_panels_revised.py

Changes from original:
    - Figure layout: 2x2 grid instead of 1x2
    - Bottom row: boxplots with MC uncertainty for 2030
"""

import pickle
from pathlib import Path

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

OUTPUT_DIR = Path("outputs/figures")

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

MAJOR_COUNTRIES = ["DE", "FR", "IT", "ES", "PL"]
EAST_EUROPE = ["BG", "HR", "CZ", "EE", "EL", "HU", "LV", "LT", "RO", "SK", "SI"]
WEST_EUROPE = [
    c for c in EU27_COUNTRIES if c not in MAJOR_COUNTRIES and c not in EAST_EUROPE
]

COUNTRY_LABELS = {
    "DE": "Germany",
    "FR": "France",
    "IT": "Italy",
    "ES": "Spain",
    "PL": "Poland",
    "East Europe": "Other Central &\nEastern Europe",
    "West Europe": "Other Western\nEurope",
}

SECTOR_LABELS = {
    "Power": "Power",
    "Industry": "Industry",
    "Mobility": "Mobility",
    "HeatingCooling": "Heating & Cooling",
    "Land": "Land Use",
    "Other": "Other",
}

COLORS_COUNTRY = {
    "DE": "#332288",
    "FR": "#88CCEE",
    "IT": "#44AA99",
    "ES": "#117733",
    "PL": "#999933",
    "East Europe": "#DDCC77",
    "West Europe": "#CC6677",
}

COLORS_SECTOR = {
    "Power": "#CC6677",
    "Industry": "#332288",
    "Mobility": "#117733",
    "HeatingCooling": "#DDCC77",
    "Land": "#88CCEE",
    "Other": "#AA4499",
}


def setup_nature_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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


def prepare_data(dataset, population_df):
    """Prepare historical data and per-MC-sample forecast data."""
    # Historical
    keys = dataset.keys
    emi_dataset = pd.DataFrame(dataset.emi_df.cpu().numpy(), columns=OUTPUT_SECTORS)
    historical = pd.concat([keys, emi_dataset], axis=1)
    for s in OUTPUT_SECTORS:
        mean_ = dataset.precomputed_scaling_params[s]["mean"]
        std_ = dataset.precomputed_scaling_params[s]["std"]
        historical[s] = historical[s] * std_ + mean_
    historical = historical.merge(population_df, on=["geo", "year"], how="left")
    historical["total_CO2"] = (
        historical[OUTPUT_SECTORS].sum(axis=1) * historical["population"]
    )
    for s in OUTPUT_SECTORS:
        historical[f"{s}_total"] = historical[s] * historical["population"]

    # MC forecasts (keep per-sample for boxplots)
    df_mc = pd.read_csv(MC_PROJECTIONS_PATH)
    df_mc["geo"] = df_mc["geo"].astype(str)
    for s in OUTPUT_SECTORS:
        mean_ = dataset.precomputed_scaling_params[s]["mean"]
        std_ = dataset.precomputed_scaling_params[s]["std"]
        df_mc[f"{s}_unnorm"] = np.clip(df_mc[f"emissions_{s}"] * std_ + mean_, 0, None)
    df_mc = df_mc.merge(population_df, on=["geo", "year"], how="left")
    df_mc["total_CO2"] = (
        df_mc[[f"{s}_unnorm" for s in OUTPUT_SECTORS]].sum(axis=1) * df_mc["population"]
    )
    for s in OUTPUT_SECTORS:
        df_mc[f"{s}_total"] = df_mc[f"{s}_unnorm"] * df_mc["population"]

    # Mean forecast for area plots
    agg_dict = {"total_CO2": "mean"}
    for s in OUTPUT_SECTORS:
        agg_dict[f"{s}_total"] = "mean"
    forecast_summary = df_mc.groupby(["geo", "year"]).agg(agg_dict).reset_index()

    return historical, forecast_summary, df_mc


def build_country_data(historical, forecast_summary):
    country_groups = ["DE", "FR", "IT", "ES", "PL", "East Europe", "West Europe"]

    def _aggregate(df, year_col="year"):
        rows = []
        for year in sorted(df[year_col].unique()):
            row = {year_col: year}
            for c in MAJOR_COUNTRIES:
                cd = df[(df["geo"] == c) & (df[year_col] == year)]
                row[c] = cd["total_CO2"].values[0] if not cd.empty else 0
            row["East Europe"] = df[
                (df["geo"].isin(EAST_EUROPE)) & (df[year_col] == year)
            ]["total_CO2"].sum()
            row["West Europe"] = df[
                (df["geo"].isin(WEST_EUROPE)) & (df[year_col] == year)
            ]["total_CO2"].sum()
            rows.append(row)
        return pd.DataFrame(rows)

    hist_df = _aggregate(historical)
    fcast_df = _aggregate(forecast_summary)
    combined = pd.concat([hist_df, fcast_df], ignore_index=True).sort_values("year")
    combined = combined.drop_duplicates(subset=["year"], keep="first")
    return combined, country_groups


def build_sector_data(historical, forecast_summary):
    hist_eu = historical[historical["geo"].isin(EU27_COUNTRIES)]
    hist_sectors = hist_eu.groupby("year", as_index=False).agg(
        {f"{s}_total": "sum" for s in OUTPUT_SECTORS}
    )
    for s in OUTPUT_SECTORS:
        hist_sectors[s] = hist_sectors[f"{s}_total"]
        hist_sectors = hist_sectors.drop(columns=[f"{s}_total"])

    fcast_eu = forecast_summary[forecast_summary["geo"].isin(EU27_COUNTRIES)]
    fcast_sectors = fcast_eu.groupby("year", as_index=False).agg(
        {f"{s}_total": "sum" for s in OUTPUT_SECTORS}
    )
    for s in OUTPUT_SECTORS:
        fcast_sectors[s] = fcast_sectors[f"{s}_total"]
        fcast_sectors = fcast_sectors.drop(columns=[f"{s}_total"])

    combined = pd.concat([hist_sectors, fcast_sectors], ignore_index=True).sort_values(
        "year"
    )
    combined = combined.drop_duplicates(subset=["year"], keep="first")
    return combined


def build_mc_2030_country(df_mc):
    """Build per-MC-sample 2030 emissions by country group (in Gt)."""
    df_2030 = df_mc[df_mc["year"] == 2030].copy()
    country_groups = ["DE", "FR", "IT", "ES", "PL", "East Europe", "West Europe"]

    records = []
    for mc in df_2030["mc_sample"].unique():
        mc_slice = df_2030[df_2030["mc_sample"] == mc]
        row = {"mc_sample": mc}
        for c in MAJOR_COUNTRIES:
            cd = mc_slice[mc_slice["geo"] == c]
            row[c] = cd["total_CO2"].values[0] / 1e9 if not cd.empty else 0
        row["East Europe"] = (
            mc_slice[mc_slice["geo"].isin(EAST_EUROPE)]["total_CO2"].sum() / 1e9
        )
        row["West Europe"] = (
            mc_slice[mc_slice["geo"].isin(WEST_EUROPE)]["total_CO2"].sum() / 1e9
        )
        records.append(row)

    return pd.DataFrame(records), country_groups


def build_mc_2030_sector(df_mc):
    """Build per-MC-sample 2030 emissions by sector (in Gt)."""
    df_2030 = df_mc[
        (df_mc["year"] == 2030) & (df_mc["geo"].isin(EU27_COUNTRIES))
    ].copy()

    records = []
    for mc in df_2030["mc_sample"].unique():
        mc_slice = df_2030[df_2030["mc_sample"] == mc]
        row = {"mc_sample": mc}
        for s in OUTPUT_SECTORS:
            row[s] = mc_slice[f"{s}_total"].sum() / 1e9
        records.append(row)

    return pd.DataFrame(records)


def create_panel_figure(
    combined_country,
    country_groups,
    combined_sector,
    mc_country_df,
    mc_sector_df,
    hist_max_year,
):
    setup_nature_style()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 6.0),
        gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.35, "wspace": 0.15},
    )
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # Convert to Gt for area plots
    plot_country = combined_country.copy()
    plot_sector = combined_sector.copy()
    for g in country_groups:
        plot_country[g] = plot_country[g] / 1e9
    for s in OUTPUT_SECTORS:
        plot_sector[s] = plot_sector[s] / 1e9

    ylabel = "CO2 emissions (Gt)"

    # Order by 2030 emissions
    data_2030 = combined_country[combined_country["year"] == 2030]
    emissions_2030 = {g: data_2030[g].values[0] for g in country_groups}
    ordered_countries = sorted(
        MAJOR_COUNTRIES, key=lambda g: emissions_2030[g], reverse=True
    )
    ordered_country_groups = ordered_countries + ["East Europe", "West Europe"]

    sector_2030 = plot_sector[plot_sector["year"] == 2030][OUTPUT_SECTORS].iloc[0]
    ordered_sectors = sector_2030.sort_values(ascending=False).index.tolist()

    # ========== Panel (a): Country stacked area ==========
    ax1.stackplot(
        plot_country["year"],
        *[plot_country[g] for g in ordered_country_groups],
        labels=[COUNTRY_LABELS[g] for g in ordered_country_groups],
        colors=[COLORS_COUNTRY[g] for g in ordered_country_groups],
        alpha=0.9,
        linewidth=0,
    )
    cumsum = np.zeros(len(plot_country))
    for g in ordered_country_groups:
        cumsum = cumsum + plot_country[g].values
        ax1.plot(plot_country["year"], cumsum, color="white", linewidth=0.4, alpha=0.6)
    ax1.axvspan(hist_max_year, 2030, alpha=0.04, color="#000000", zorder=0)
    ax1.axvline(
        hist_max_year, color="#2c3e50", linestyle="--", linewidth=0.7, alpha=0.6
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(2010, 2030)
    ax1.set_xticks([2010, 2015, 2020, 2025, 2030])
    ax1.set_ylim(0, None)
    ax1.yaxis.grid(True, linestyle="-", alpha=0.15, color="#666666", linewidth=0.3)
    ax1.set_axisbelow(True)
    leg1 = ax1.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=5.5,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.4,
        handlelength=1.2,
    )
    leg1.get_frame().set_linewidth(0.3)
    ax1.text(
        -0.12,
        1.05,
        "a",
        transform=ax1.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # ========== Panel (b): Sector stacked area ==========
    ax2.stackplot(
        plot_sector["year"],
        *[plot_sector[s] for s in ordered_sectors],
        labels=[SECTOR_LABELS[s] for s in ordered_sectors],
        colors=[COLORS_SECTOR[s] for s in ordered_sectors],
        alpha=0.9,
        linewidth=0,
    )
    cumsum = np.zeros(len(plot_sector))
    for s in ordered_sectors:
        cumsum = cumsum + plot_sector[s].values
        ax2.plot(plot_sector["year"], cumsum, color="white", linewidth=0.4, alpha=0.6)
    ax2.axvspan(hist_max_year, 2030, alpha=0.04, color="#000000", zorder=0)
    ax2.axvline(
        hist_max_year, color="#2c3e50", linestyle="--", linewidth=0.7, alpha=0.6
    )
    ax2.set_xlabel("Year")
    ax2.set_xlim(2010, 2030)
    ax2.set_xticks([2010, 2015, 2020, 2025, 2030])
    ax2.yaxis.grid(True, linestyle="-", alpha=0.15, color="#666666", linewidth=0.3)
    ax2.set_axisbelow(True)
    leg2 = ax2.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=5.5,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.4,
        handlelength=1.2,
    )
    leg2.get_frame().set_linewidth(0.3)
    ax2.text(
        -0.05,
        1.05,
        "b",
        transform=ax2.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # ========== Panel (c): Country boxplots for 2030 ==========
    bp_data_country = [mc_country_df[g].values for g in ordered_country_groups]
    bp_labels_country = [
        COUNTRY_LABELS[g].replace("\n", " ") for g in ordered_country_groups
    ]

    bp = ax3.boxplot(
        bp_data_country,
        vert=True,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(linewidth=0.6, color="#555555"),
        capprops=dict(linewidth=0.6, color="#555555"),
        flierprops=dict(
            marker=".",
            markersize=1.5,
            markerfacecolor="#aaaaaa",
            markeredgecolor="none",
        ),
    )
    for patch, g in zip(bp["boxes"], ordered_country_groups):
        patch.set_facecolor(COLORS_COUNTRY[g])
        patch.set_alpha(0.85)
        patch.set_linewidth(0.5)

    ax3.set_xticklabels(bp_labels_country, rotation=30, ha="right", fontsize=6)
    ax3.set_ylabel("2030 projected CO₂ (Gt)")
    ax3.yaxis.grid(True, linestyle="-", alpha=0.15, color="#666666", linewidth=0.3)
    ax3.set_axisbelow(True)
    ax3.text(
        -0.12,
        1.05,
        "c",
        transform=ax3.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # ========== Panel (d): Sector boxplots for 2030 ==========
    bp_data_sector = [mc_sector_df[s].values for s in ordered_sectors]
    bp_labels_sector = [SECTOR_LABELS[s] for s in ordered_sectors]

    bp2 = ax4.boxplot(
        bp_data_sector,
        vert=True,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(linewidth=0.6, color="#555555"),
        capprops=dict(linewidth=0.6, color="#555555"),
        flierprops=dict(
            marker=".",
            markersize=1.5,
            markerfacecolor="#aaaaaa",
            markeredgecolor="none",
        ),
    )
    for patch, s in zip(bp2["boxes"], ordered_sectors):
        patch.set_facecolor(COLORS_SECTOR[s])
        patch.set_alpha(0.85)
        patch.set_linewidth(0.5)

    ax4.set_xticklabels(bp_labels_sector, rotation=30, ha="right", fontsize=6)
    ax4.yaxis.grid(True, linestyle="-", alpha=0.15, color="#666666", linewidth=0.3)
    ax4.set_axisbelow(True)
    ax4.text(
        -0.05,
        1.05,
        "d",
        transform=ax4.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Final styling
    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.tick_params(axis="both", which="both", length=3, width=0.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ["png", "pdf", "svg"]:
        plt.savefig(
            OUTPUT_DIR / f"fig3_panel_attribution_with_boxplots.{fmt}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/fig3_panel_attribution_with_boxplots.[png|pdf|svg]")


def main():
    print("=" * 70)
    print("GENERATING FIGURE 3 (REVISED): ATTRIBUTION PANELS + BOXPLOTS")
    print("=" * 70)

    dataset = load_dataset(DATASET_PATH)
    population_df = load_population_data()

    print("Preparing data...")
    historical, forecast_summary, df_mc = prepare_data(dataset, population_df)
    combined_country, country_groups = build_country_data(historical, forecast_summary)
    combined_sector = build_sector_data(historical, forecast_summary)

    print("Building MC boxplot data...")
    mc_country_df, _ = build_mc_2030_country(df_mc)
    mc_sector_df = build_mc_2030_sector(df_mc)

    hist_max_year = historical["year"].max()

    print("Creating 4-panel figure...")
    create_panel_figure(
        combined_country,
        country_groups,
        combined_sector,
        mc_country_df,
        mc_sector_df,
        hist_max_year,
    )

    # Summary
    print("\n" + "=" * 60)
    print("2030 MC UNCERTAINTY SUMMARY (Gt CO2)")
    print("=" * 60)
    print("\nBy Country/Region:")
    for g in country_groups:
        vals = mc_country_df[g]
        label = COUNTRY_LABELS[g].replace("\n", " ")
        print(
            f"  {label:<30s} median={vals.median():.3f}  IQR=[{vals.quantile(0.25):.3f}, {vals.quantile(0.75):.3f}]  "
            f"90%=[{vals.quantile(0.05):.3f}, {vals.quantile(0.95):.3f}]"
        )

    print("\nBy Sector:")
    for s in OUTPUT_SECTORS:
        vals = mc_sector_df[s]
        print(
            f"  {SECTOR_LABELS[s]:<20s} median={vals.median():.3f}  IQR=[{vals.quantile(0.25):.3f}, {vals.quantile(0.75):.3f}]  "
            f"90%=[{vals.quantile(0.05):.3f}, {vals.quantile(0.95):.3f}]"
        )

    print("\nFigure 3 (revised) generation complete!")


if __name__ == "__main__":
    main()
