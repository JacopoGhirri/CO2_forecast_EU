"""
Figure 3 (Revised): Attribution Panels + 2030 Boxplots.

FONT SIZING RATIONALE:
  When embedded in a LaTeX two-column paper at \textwidth, figures are
  displayed at roughly 45-50% of their native pixel size.  All font sizes
  below are therefore set ~2–2.5× larger than what looks comfortable on
  screen, so they render legibly on a printed/PDF page.

  Target minimum readable size in paper: ~7 pt
  → font.size base: 18 pt  (renders to ~8–9 pt in paper)
  → axis labels:    20 pt
  → tick labels:    16 pt
  → legend:         15 pt
  → panel letters:  24 pt bold

Usage:
    python -m scripts.visualization.figure_attribution_panels

Outputs:
    - outputs/figures/fig3_panel_attribution_with_boxplots.[png|pdf|svg]
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

BASELINE_YEAR = 2024
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
    "East Europe": "Other C. & E. Europe",
    "West Europe": "Other W. Europe",
}

SECTOR_LABELS = {
    "Power": "Power",
    "Industry": "Industry",
    "Mobility": "Mobility",
    "HeatingCooling": "Heating & Cooling",
    "Land": "Land Use",
    "Other": "Other",
}

# ── Intuitive country colours ──────────────────────────────────────────────
COLORS_COUNTRY = {
    "DE": "#2c2c54",
    "FR": "#1a5276",
    "IT": "#5dade2",
    "ES": "#f4d03f",
    "PL": "#c0392b",
    "East Europe": "#a04000",
    "West Europe": "#229954",
}

# ── Intuitive sector colours ──────────────────────────────────────────────
COLORS_SECTOR = {
    "Mobility": "#2980b9",
    "Industry": "#717d7e",
    "Power": "#e67e22",
    "HeatingCooling": "#e74c3c",
    "Land": "#27ae60",
    "Other": "#8e44ad",
}


def setup_style():
    """
    Font sizes are set ~2× larger than screen-comfortable values so the
    figure remains legible when scaled down inside a LaTeX document.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            # ── Core sizes (will render ~half this in paper) ──
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            # ── Lines & ticks ──
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # ── Backgrounds ──
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# Data helpers (unchanged logic)
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


def prepare_data(dataset, population_df):
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
    for s in OUTPUT_SECTORS:
        anchor_2024[s] = anchor_2024[f"{s}_unnorm"]
    anchor_2024 = anchor_2024[
        ["geo", "year", "population", "total_CO2"]
        + OUTPUT_SECTORS
        + [f"{s}_total" for s in OUTPUT_SECTORS]
    ]
    historical = pd.concat([historical, anchor_2024], ignore_index=True)
    df_mc_forecast = df_mc[df_mc["year"] > BASELINE_YEAR].copy()

    agg_dict = {"total_CO2": "mean"}
    for s in OUTPUT_SECTORS:
        agg_dict[f"{s}_total"] = "mean"
    forecast_summary = (
        df_mc_forecast.groupby(["geo", "year"]).agg(agg_dict).reset_index()
    )
    return historical, forecast_summary, df_mc


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


def _place_legend_above(ax, groups, colors_map, labels_map, ncol=4):
    """Frameless colour-patch legend placed just above the axes."""
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=colors_map[g], edgecolor="none", alpha=0.88
        )
        for g in groups
    ]
    labels = [labels_map[g] for g in groups]
    leg = ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.03),
        ncol=ncol,
        frameon=False,
        fontsize=15,  # large enough to survive paper scaling
        handlelength=0.9,
        handleheight=0.75,
        handletextpad=0.35,
        columnspacing=0.7,
        borderpad=0,
    )
    return leg


# =============================================================================
# Main figure
# =============================================================================


def create_panel_figure(
    combined_country,
    country_groups,
    combined_sector,
    mc_country_df,
    mc_sector_df,
    hist_max_year,
):
    setup_style()

    # Large native size — will be scaled down in LaTeX
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.45, 1],
        hspace=0.45,  # tighter vertical gap between rows
        wspace=0.30,
        top=0.91,
        bottom=0.10,
        left=0.09,
        right=0.98,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Convert to Gt
    plot_country = combined_country.copy()
    plot_sector = combined_sector.copy()
    for g in country_groups:
        plot_country[g] = plot_country[g] / 1e9
    for s in OUTPUT_SECTORS:
        plot_sector[s] = plot_sector[s] / 1e9

    ylabel = "CO2 emissions (Gt)"

    data_2030 = combined_country[combined_country["year"] == PROJECTION_YEAR]
    ordered_countries = sorted(
        MAJOR_COUNTRIES, key=lambda g: data_2030[g].values[0], reverse=True
    )
    ordered_country_groups = ordered_countries + ["East Europe", "West Europe"]

    sector_2030 = plot_sector[plot_sector["year"] == PROJECTION_YEAR][
        OUTPUT_SECTORS
    ].iloc[0]
    ordered_sectors = sector_2030.sort_values(ascending=False).index.tolist()

    def _stacked_area(ax, df_plot, groups, colors_map):
        ax.stackplot(
            df_plot["year"],
            *[df_plot[g] for g in groups],
            colors=[colors_map[g] for g in groups],
            alpha=0.88,
            linewidth=0,
        )
        cumsum = np.zeros(len(df_plot))
        for g in groups:
            cumsum += df_plot[g].values
            ax.plot(df_plot["year"], cumsum, color="white", linewidth=0.7, alpha=0.7)
        ax.axvspan(
            hist_max_year, PROJECTION_YEAR, alpha=0.05, color="#000000", zorder=0
        )
        ax.axvline(
            hist_max_year, color="#444444", linestyle="--", linewidth=1.2, alpha=0.7
        )
        ax.set_xlabel("Year", labelpad=4)
        ax.set_xlim(2010, PROJECTION_YEAR)
        ax.set_xticks([2010, 2015, 2020, 2025, 2030])
        ax.set_ylim(0, None)
        ax.yaxis.grid(True, linestyle="-", alpha=0.15, color="#666666", linewidth=0.5)
        ax.set_axisbelow(True)

    # ── (a) Country stacked area ──────────────────────────────────────────
    _stacked_area(ax1, plot_country, ordered_country_groups, COLORS_COUNTRY)
    ax1.set_ylabel(ylabel)
    _place_legend_above(
        ax1, ordered_country_groups, COLORS_COUNTRY, COUNTRY_LABELS, ncol=4
    )

    # ── (b) Sector stacked area ───────────────────────────────────────────
    _stacked_area(ax2, plot_sector, ordered_sectors, COLORS_SECTOR)
    _place_legend_above(ax2, ordered_sectors, COLORS_SECTOR, SECTOR_LABELS, ncol=3)

    # ── (c) Country boxplots ──────────────────────────────────────────────
    bp_data_country = [mc_country_df[g].values for g in ordered_country_groups]
    bp_labels_country = [COUNTRY_LABELS[g] for g in ordered_country_groups]
    bp = ax3.boxplot(
        bp_data_country,
        vert=True,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.1, color="#555555"),
        capprops=dict(linewidth=1.1, color="#555555"),
        flierprops=dict(
            marker=".", markersize=3, markerfacecolor="#aaaaaa", markeredgecolor="none"
        ),
    )
    for patch, g in zip(bp["boxes"], ordered_country_groups):
        patch.set_facecolor(COLORS_COUNTRY[g])
        patch.set_alpha(0.85)
        patch.set_linewidth(0.8)
    ax3.set_xticklabels(bp_labels_country, rotation=35, ha="right", fontsize=15)
    ax3.set_ylabel(f"{PROJECTION_YEAR} projected CO2 (Gt)")
    ax3.yaxis.grid(True, linestyle="-", alpha=0.15, color="#666666", linewidth=0.5)
    ax3.set_axisbelow(True)

    # ── (d) Sector boxplots ───────────────────────────────────────────────
    bp_data_sector = [mc_sector_df[s].values for s in ordered_sectors]
    bp_labels_sector = [SECTOR_LABELS[s] for s in ordered_sectors]
    bp2 = ax4.boxplot(
        bp_data_sector,
        vert=True,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.1, color="#555555"),
        capprops=dict(linewidth=1.1, color="#555555"),
        flierprops=dict(
            marker=".", markersize=3, markerfacecolor="#aaaaaa", markeredgecolor="none"
        ),
    )
    for patch, s in zip(bp2["boxes"], ordered_sectors):
        patch.set_facecolor(COLORS_SECTOR[s])
        patch.set_alpha(0.85)
        patch.set_linewidth(0.8)
    ax4.set_xticklabels(bp_labels_sector, rotation=35, ha="right", fontsize=15)
    ax4.yaxis.grid(True, linestyle="-", alpha=0.15, color="#666666", linewidth=0.5)
    ax4.set_axisbelow(True)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines["bottom"].set_linewidth(1.0)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.tick_params(axis="both", which="both", length=5, width=1.0)

    # ── Panel letters in figure coordinates — guaranteed consistent alignment ──
    # Render first so bbox positions are computed, then read axes corners.
    fig.canvas.draw()

    def _panel_letter(ax, letter, x_offset=-0.055):
        """Place panel letter just above-left of the axes top-left corner,
        using figure coordinates so all four labels sit at the same relative
        position regardless of legend height."""
        bbox = ax.get_position()  # fraction of figure
        fig.text(
            bbox.x0 + x_offset * bbox.width,
            bbox.y1 + 0.012,  # small fixed gap above axes top
            letter,
            fontsize=24,
            fontweight="bold",
            va="bottom",
            ha="left",
            transform=fig.transFigure,
        )

    _panel_letter(ax1, "a", x_offset=-0.055)
    _panel_letter(ax2, "b", x_offset=-0.040)
    _panel_letter(ax3, "c", x_offset=-0.055)
    _panel_letter(ax4, "d", x_offset=-0.040)

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
    print("GENERATING FIGURE 3: ATTRIBUTION PANELS + BOXPLOTS")
    print("=" * 70)

    dataset = load_dataset(DATASET_PATH)
    population_df = load_population_data()

    historical, forecast_summary, df_mc = prepare_data(dataset, population_df)
    combined_country, country_groups = build_country_data(historical, forecast_summary)
    combined_sector = build_sector_data(historical, forecast_summary)

    hist_max_year = int(historical["year"].max())
    mc_country_df, _ = build_mc_2030_country(df_mc)
    mc_sector_df = build_mc_2030_sector(df_mc)

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

    # ── Numbers for paper Section: "Uneven progress" ───────────────────────
    print("\n" + "=" * 60)
    print("NUMBERS FOR PAPER — Section: Uneven progress")
    print("=" * 60)

    eu27_hist = historical[historical["geo"].isin(EU27_COUNTRIES)]

    # Aggregate EU27 totals by year
    eu_year = eu27_hist.groupby("year")["total_CO2"].sum()
    eu_2010 = eu_year.get(2010, float("nan"))
    eu_2024 = eu_year.get(2024, float("nan"))
    hist_pct = (eu_2024 - eu_2010) / eu_2010 * 100

    # 2030 from MC
    eu_mc_2030 = (
        mc_country_df[["DE", "FR", "IT", "ES", "PL", "East Europe", "West Europe"]]
        .sum(axis=1)
        .mean()
        * 1e9
    )
    eu_2030_total = eu_mc_2030  # Gt → tonnes already (*1e9 above)
    total_pct_from_2010 = (eu_mc_2030 - eu_2010) / eu_2010 * 100

    print(f"  EU27 total CO2 2010: {eu_2010 / 1e9:.2f} Gt")
    print(f"  EU27 total CO2 2024: {eu_2024 / 1e9:.2f} Gt")
    print(f"  Historical reduction 2010→2024: {hist_pct:.1f}%")
    print(f"  EU27 projected 2030: {eu_mc_2030 / 1e9:.2f} Gt")
    print(f"  Total reduction 2010→2030: {total_pct_from_2010:.1f}%")

    # Sector-level from mc_sector_df (Gt)
    print("\n  Sector medians (Gt CO2, 2030):")
    total_2030_gt = mc_sector_df[OUTPUT_SECTORS].median().sum()
    for s in OUTPUT_SECTORS:
        val = mc_sector_df[s].median()
        share = val / total_2030_gt * 100
        print(f"    {s:<16s}: {val:.3f} Gt  ({share:.1f}% of total)")

    # Sector historical (2010) from stacked area data
    hist_sector_2010 = eu27_hist[eu27_hist["year"] == 2010]
    hist_sector_2024 = eu27_hist[eu27_hist["year"] == 2024]
    print("\n  Sector % change 2010→2030 (vs 2010 historical total):")
    for s in OUTPUT_SECTORS:
        s_col = f"{s}_total"
        if s_col in eu27_hist.columns:
            v2010 = hist_sector_2010[s_col].sum()
            v2030 = mc_sector_df[s].median() * 1e9
            pct = (v2030 - v2010) / v2010 * 100 if v2010 > 0 else float("nan")
            print(
                f"    {s:<16s}: 2010={v2010 / 1e9:.3f} Gt  2030≈{v2030 / 1e9:.3f} Gt  Δ={pct:+.1f}%"
            )
    print("\nDone!")


if __name__ == "__main__":
    main()
