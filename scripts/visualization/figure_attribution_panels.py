"""
Figure 3 (Revised): Attribution Panels + 2030 Boxplots.

Four-panel figure:
  (a) Historical + projected stacked area by country/region
  (b) Historical + projected stacked area by sector
  (c) Boxplots of 2030 MC distribution by country/region (Gt CO2)
  (d) Boxplots of 2030 MC distribution by sector (Gt CO2)

NOTE: The MC projections now use the 2024-anchored file
      (mc_projections.csv).  Year 2024 is therefore present in
      both the historical dataset AND the MC file (as the anchor point).
      To avoid a discontinuity in the stacked-area panels we:
        1. Keep all historical data up to and including 2024 from the
           observed (denormalised dataset) source.
        2. Use MC mean values from year 2025 onward for the forecast
           continuation, dropping the 2024 MC row since it is identical
           to the historical anchor.
      hist_max_year is set to 2024 so the dashed "forecast starts here"
      line falls at the correct position.

Usage:
    python -m scripts.visualization.figure_attribution_panels

Outputs:
    - outputs/figures/fig3_panel_attribution_with_boxplots.[png|pdf|svg]
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

BASELINE_YEAR = 2024  # first year in MC file; also last observed year
PROJECTION_YEAR = 2030

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
    """
    Prepare historical and forecast data.

    Historical series: drawn from the dataset object (years 2010–2023) plus
    the 2024 observed anchor extracted from the MC projections file. The
    training dataset only covers up to 2023, but the MC file contains
    identical 2024 values across all samples (the observed anchor), so we
    average across MC samples and append them to the historical series.

    Forecast series: MC mean from year 2025 onward. Year 2024 is excluded
    from the forecast continuation to avoid double-counting in the
    stacked-area plots.
    """
    # ---- Historical (from dataset, up to 2023) ----
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

    # ---- MC forecasts ----
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

    # ---- Append 2024 anchor to historical ----
    # The MC file year==2024 is the observed anchor (identical across samples).
    # Average across MC samples and append so the stacked area is continuous.
    agg_cols = {"total_CO2": "mean", "population": "first"}
    for s in OUTPUT_SECTORS:
        agg_cols[s + "_unnorm"] = "mean"
        agg_cols[f"{s}_total"] = "mean"
    anchor_2024 = (
        df_mc[df_mc["year"] == BASELINE_YEAR]
        .groupby(["geo", "year"])
        .agg(agg_cols)
        .reset_index()
    )
    # Rename unnorm columns to match historical schema
    for s in OUTPUT_SECTORS:
        anchor_2024[s] = anchor_2024[f"{s}_unnorm"]
    anchor_2024 = anchor_2024[
        ["geo", "year", "population", "total_CO2"]
        + OUTPUT_SECTORS
        + [f"{s}_total" for s in OUTPUT_SECTORS]
    ]
    historical = pd.concat([historical, anchor_2024], ignore_index=True)

    # Exclude year==BASELINE_YEAR from the forecast continuation:
    # it is already covered by the historical series.
    df_mc_forecast = df_mc[df_mc["year"] > BASELINE_YEAR].copy()

    # Summary (mean across mc_samples) for area plots
    agg_dict = {"total_CO2": "mean"}
    for s in OUTPUT_SECTORS:
        agg_dict[f"{s}_total"] = "mean"
    forecast_summary = (
        df_mc_forecast.groupby(["geo", "year"]).agg(agg_dict).reset_index()
    )

    return historical, forecast_summary, df_mc  # df_mc keeps all years for boxplots


def build_country_data(historical, forecast_summary):
    country_groups = ["DE", "FR", "IT", "ES", "PL", "East Europe", "West Europe"]

    def _aggregate(df):
        rows = []
        for year in sorted(df["year"].unique()):
            row = {"year": year}
            for c in MAJOR_COUNTRIES:
                cd = df[(df["geo"] == c) & (df["year"] == year)]
                row[c] = cd["total_CO2"].values[0] if not cd.empty else 0
            row["East Europe"] = df[
                (df["geo"].isin(EAST_EUROPE)) & (df["year"] == year)
            ]["total_CO2"].sum()
            row["West Europe"] = df[
                (df["geo"].isin(WEST_EUROPE)) & (df["year"] == year)
            ]["total_CO2"].sum()
            rows.append(row)
        return pd.DataFrame(rows)

    hist_df = _aggregate(historical)
    fcast_df = _aggregate(forecast_summary)

    # Concatenate: historical covers up to BASELINE_YEAR; forecast from BASELINE_YEAR+1
    combined = (
        pd.concat([hist_df, fcast_df], ignore_index=True)
        .sort_values("year")
        .drop_duplicates(subset=["year"], keep="first")
    )
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

    combined = (
        pd.concat([hist_sectors, fcast_sectors], ignore_index=True)
        .sort_values("year")
        .drop_duplicates(subset=["year"], keep="first")
    )
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
        (df_mc["year"] == PROJECTION_YEAR) & (df_mc["geo"].isin(EU27_COUNTRIES))
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

    # Convert to Gt
    plot_country = combined_country.copy()
    plot_sector = combined_sector.copy()
    for g in country_groups:
        plot_country[g] = plot_country[g] / 1e9
    for s in OUTPUT_SECTORS:
        plot_sector[s] = plot_sector[s] / 1e9

    ylabel = "CO2 emissions (Gt)"

    # Order by 2030 emissions
    data_2030 = combined_country[combined_country["year"] == PROJECTION_YEAR]
    ordered_countries = sorted(
        MAJOR_COUNTRIES, key=lambda g: data_2030[g].values[0], reverse=True
    )
    ordered_country_groups = ordered_countries + ["East Europe", "West Europe"]

    sector_2030 = plot_sector[plot_sector["year"] == PROJECTION_YEAR][
        OUTPUT_SECTORS
    ].iloc[0]
    ordered_sectors = sector_2030.sort_values(ascending=False).index.tolist()

    def _stacked_area(ax, df_plot, groups, colors_map, labels_map):
        ax.stackplot(
            df_plot["year"],
            *[df_plot[g] for g in groups],
            labels=[labels_map[g] for g in groups],
            colors=[colors_map[g] for g in groups],
            alpha=0.9,
            linewidth=0,
        )
        cumsum = np.zeros(len(df_plot))
        for g in groups:
            cumsum += df_plot[g].values
            ax.plot(df_plot["year"], cumsum, color="white", linewidth=0.4, alpha=0.6)
        ax.axvspan(
            hist_max_year, PROJECTION_YEAR, alpha=0.04, color="#000000", zorder=0
        )
        ax.axvline(
            hist_max_year, color="#2c3e50", linestyle="--", linewidth=0.7, alpha=0.6
        )
        ax.set_xlabel("Year")
        ax.set_xlim(2010, PROJECTION_YEAR)
        ax.set_xticks([2010, 2015, 2020, 2025, 2030])
        ax.set_ylim(0, None)
        ax.yaxis.grid(True, linestyle="-", alpha=0.15, color="#666666", linewidth=0.3)
        ax.set_axisbelow(True)

    # ---- (a) Country stacked area ----
    _stacked_area(
        ax1, plot_country, ordered_country_groups, COLORS_COUNTRY, COUNTRY_LABELS
    )
    ax1.set_ylabel(ylabel)
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

    # ---- (b) Sector stacked area ----
    _stacked_area(ax2, plot_sector, ordered_sectors, COLORS_SECTOR, SECTOR_LABELS)
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

    # ---- (c) Country boxplots ----
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
    ax3.set_ylabel(f"{PROJECTION_YEAR} projected CO₂ (Gt)")
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

    # ---- (d) Sector boxplots ----
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
    print(f"  MC projections: {MC_PROJECTIONS_PATH}")
    print(f"  Baseline year : {BASELINE_YEAR}")
    print("=" * 70)

    dataset = load_dataset(DATASET_PATH)
    population_df = load_population_data()

    print("Preparing data...")
    historical, forecast_summary, df_mc = prepare_data(dataset, population_df)
    combined_country, country_groups = build_country_data(historical, forecast_summary)
    combined_sector = build_sector_data(historical, forecast_summary)

    # hist_max_year: last year present in the historical (observed) dataset.
    # With 2024 data included in training, this will be 2024; otherwise 2023.
    hist_max_year = int(historical["year"].max())
    print(f"  Historical data ends at: {hist_max_year}")
    print(f"  Forecast continuation:   {hist_max_year + 1} → {PROJECTION_YEAR}")

    print("Building MC boxplot data...")
    mc_country_df, _ = build_mc_2030_country(df_mc)
    mc_sector_df = build_mc_2030_sector(df_mc)

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
    print(f"\n{'=' * 60}")
    print(f"{PROJECTION_YEAR} MC UNCERTAINTY SUMMARY (Gt CO₂)")
    print(f"{'=' * 60}")
    print("\nBy Country/Region:")
    for g in country_groups:
        vals = mc_country_df[g]
        label = COUNTRY_LABELS[g].replace("\n", " ")
        print(
            f"  {label:<30s} median={vals.median():.3f}  "
            f"IQR=[{vals.quantile(0.25):.3f}, {vals.quantile(0.75):.3f}]  "
            f"90%=[{vals.quantile(0.05):.3f}, {vals.quantile(0.95):.3f}]"
        )
    print("\nBy Sector:")
    for s in OUTPUT_SECTORS:
        vals = mc_sector_df[s]
        print(
            f"  {SECTOR_LABELS[s]:<20s} median={vals.median():.3f}  "
            f"IQR=[{vals.quantile(0.25):.3f}, {vals.quantile(0.75):.3f}]  "
            f"90%=[{vals.quantile(0.05):.3f}, {vals.quantile(0.95):.3f}]"
        )
    print("\nFigure 3 (revised) generation complete!")


if __name__ == "__main__":
    main()