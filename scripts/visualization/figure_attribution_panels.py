"""
Figure 3: Attribution Panel Figure.

Generates a two-panel figure showing EU27 emission trajectories from 2010 to 2030:
  (a) Attribution by country/region
  (b) Attribution by sector

This shows how emissions are distributed across major emitters and sectors,
with historical data (2010-2023) and projections (2024-2030).

Usage:
    python -m scripts.visualization.figure_attribution_panels

Outputs:
    - outputs/figures/fig3_panel_attribution_absolute.pdf
    - outputs/figures/fig3_panel_attribution_absolute.png

Reference:
    Figure 3 in the paper decomposes EU27 emissions by country and sector.
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

# Paul Tol colorblind-safe palette
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
    """Configure matplotlib for Nature Climate Change publication style."""
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


def prepare_data(dataset, population_df: pd.DataFrame):
    """Prepare historical and forecast data for visualization."""
    # Historical data
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

    # MC forecasts
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

    # Aggregate forecasts
    agg_dict = {"total_CO2": "mean"}
    for s in OUTPUT_SECTORS:
        agg_dict[f"{s}_total"] = "mean"
    forecast_summary = df_mc.groupby(["geo", "year"]).agg(agg_dict).reset_index()

    return historical, forecast_summary


def build_country_data(historical: pd.DataFrame, forecast_summary: pd.DataFrame):
    """Build time series data by country/region."""
    country_groups = ["DE", "FR", "IT", "ES", "PL", "East Europe", "West Europe"]

    # Historical
    hist_country = []
    for year in sorted(historical["year"].unique()):
        year_data = {"year": year}
        for country in MAJOR_COUNTRIES:
            cd = historical[
                (historical["geo"] == country) & (historical["year"] == year)
            ]
            year_data[country] = cd["total_CO2"].values[0] if not cd.empty else 0
        year_data["East Europe"] = historical[
            (historical["geo"].isin(EAST_EUROPE)) & (historical["year"] == year)
        ]["total_CO2"].sum()
        year_data["West Europe"] = historical[
            (historical["geo"].isin(WEST_EUROPE)) & (historical["year"] == year)
        ]["total_CO2"].sum()
        hist_country.append(year_data)
    hist_country_df = pd.DataFrame(hist_country)

    # Forecast
    fcast_country = []
    for year in sorted(forecast_summary["year"].unique()):
        year_data = {"year": year}
        for country in MAJOR_COUNTRIES:
            cd = forecast_summary[
                (forecast_summary["geo"] == country)
                & (forecast_summary["year"] == year)
            ]
            year_data[country] = cd["total_CO2"].values[0] if not cd.empty else 0
        year_data["East Europe"] = forecast_summary[
            (forecast_summary["geo"].isin(EAST_EUROPE))
            & (forecast_summary["year"] == year)
        ]["total_CO2"].sum()
        year_data["West Europe"] = forecast_summary[
            (forecast_summary["geo"].isin(WEST_EUROPE))
            & (forecast_summary["year"] == year)
        ]["total_CO2"].sum()
        fcast_country.append(year_data)
    fcast_country_df = pd.DataFrame(fcast_country)

    combined = pd.concat(
        [hist_country_df, fcast_country_df], ignore_index=True
    ).sort_values("year")
    combined = combined.drop_duplicates(subset=["year"], keep="first")

    return combined, country_groups


def build_sector_data(historical: pd.DataFrame, forecast_summary: pd.DataFrame):
    """Build time series data by sector."""
    hist_eu = historical[historical["geo"].isin(EU27_COUNTRIES)].copy()
    hist_sectors = hist_eu.groupby("year", as_index=False).agg(
        {f"{s}_total": "sum" for s in OUTPUT_SECTORS}
    )
    for s in OUTPUT_SECTORS:
        hist_sectors[s] = hist_sectors[f"{s}_total"]
        hist_sectors = hist_sectors.drop(columns=[f"{s}_total"])

    fcast_eu = forecast_summary[forecast_summary["geo"].isin(EU27_COUNTRIES)].copy()
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


def create_panel_figure(
    combined_country: pd.DataFrame,
    country_groups: list,
    combined_sector: pd.DataFrame,
    hist_max_year: int,
):
    """Create the two-panel attribution figure."""
    setup_nature_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)

    # Convert to Gt
    plot_country = combined_country.copy()
    plot_sector = combined_sector.copy()
    for g in country_groups:
        plot_country[g] = plot_country[g] / 1e9
    for s in OUTPUT_SECTORS:
        plot_sector[s] = plot_sector[s] / 1e9

    ylabel = "CO2 emissions (Gt)"

    # Order by 2030 emissions (largest at bottom)
    data_2030 = combined_country[combined_country["year"] == 2030]
    emissions_2030 = {g: data_2030[g].values[0] for g in country_groups}
    individual = ["DE", "FR", "IT", "ES", "PL"]
    ordered_countries = sorted(
        individual, key=lambda g: emissions_2030[g], reverse=True
    )
    ordered_country_groups = ordered_countries + ["East Europe", "West Europe"]

    sector_2030 = plot_sector[plot_sector["year"] == 2030][OUTPUT_SECTORS].iloc[0]
    ordered_sectors = sector_2030.sort_values(ascending=False).index.tolist()

    # Panel (a): Country attribution
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

    # Panel (b): Sector attribution
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

    # Final styling
    for ax in [ax1, ax2]:
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.tick_params(axis="both", which="both", length=3, width=0.5)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.08)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ["png", "pdf", "svg"]:
        plt.savefig(
            OUTPUT_DIR / f"fig3_panel_attribution_absolute.{fmt}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close()

    print(f"Saved: {OUTPUT_DIR}/fig3_panel_attribution_absolute.[png|pdf|svg]")

    return combined_country, combined_sector, ordered_country_groups, ordered_sectors


def main():
    """Generate Figure 3: Attribution Panel Figure."""
    print("=" * 70)
    print("GENERATING FIGURE 3: ATTRIBUTION PANELS")
    print("=" * 70)

    print("\nLoading data...")
    dataset = load_dataset(DATASET_PATH)
    population_df = load_population_data()

    print("Preparing data...")
    historical, forecast_summary = prepare_data(dataset, population_df)
    combined_country, country_groups = build_country_data(historical, forecast_summary)
    combined_sector = build_sector_data(historical, forecast_summary)

    hist_max_year = historical["year"].max()

    print("\nCreating visualization...")
    combined_country, combined_sector, ordered_groups, ordered_sectors = (
        create_panel_figure(
            combined_country, country_groups, combined_sector, hist_max_year
        )
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PANEL FIGURE SUMMARY")
    print("=" * 60)

    total_2010 = (
        combined_country[combined_country["year"] == 2010][country_groups]
        .sum(axis=1)
        .values[0]
    )
    total_2023 = (
        combined_country[combined_country["year"] == 2023][country_groups]
        .sum(axis=1)
        .values[0]
    )
    total_2030 = (
        combined_country[combined_country["year"] == 2030][country_groups]
        .sum(axis=1)
        .values[0]
    )

    print("\nEU27 Total Emissions:")
    print(f"  2010: {total_2010 / 1e9:.2f} Gt")
    print(f"  2023: {total_2023 / 1e9:.2f} Gt")
    print(f"  2030: {total_2030 / 1e9:.2f} Gt")
    print(f"  Change 2010-2030: {(total_2030 / total_2010 - 1) * 100:+.1f}%")

    print("\n--- By Country/Region (2030) ---")
    for g in ordered_groups:
        val = combined_country[combined_country["year"] == 2030][g].values[0]
        pct = val / total_2030 * 100
        label = COUNTRY_LABELS[g].replace("\n", " ")
        print(f"  {label}: {val / 1e9:.2f} Gt ({pct:.1f}%)")

    print("\n--- By Sector (2030) ---")
    sector_total = (
        combined_sector[combined_sector["year"] == 2030][OUTPUT_SECTORS]
        .sum(axis=1)
        .values[0]
    )
    for s in ordered_sectors:
        val = combined_sector[combined_sector["year"] == 2030][s].values[0]
        pct = val / sector_total * 100
        print(f"  {SECTOR_LABELS[s]}: {val / 1e9:.2f} Gt ({pct:.1f}%)")

    print("\nFigure 3 generation complete!")


if __name__ == "__main__":
    main()
