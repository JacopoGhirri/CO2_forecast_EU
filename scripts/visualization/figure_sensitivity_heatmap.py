"""
Figure 4: Sobol Sensitivity Analysis Heatmap.

Generates a publication-quality single-panel figure showing total-order
Sobol indices (ST) across emission sectors and input variables. Total-order
indices (ST) capture each variable's full contribution to output variance
including all interaction effects with other variables.

Variable Mode:
    - "full": Includes both input variables and context variables

Layout:
    - Rows: emission sectors (y-axis)
    - Columns: input variables ordered thematically (x-axis)
    - No panel letter (standalone figure)

Usage:
    python -m scripts.visualization.figure_sensitivity_heatmap

Outputs:
    - outputs/figures/fig4_sobol_sensitivity_heatmap_full.pdf
    - outputs/figures/fig4_sobol_sensitivity_heatmap_full.png

Reference:
    Figure 4 in the paper shows Sobol sensitivity analysis results.
    Section 4 "Methods" discusses sensitivity analysis methodology.
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# Configuration
# =============================================================================

# Input paths
SENSITIVITY_DIR = Path("data/sensitivity")
OUTPUT_DIR = Path("outputs/figures")

# Colorbar ceiling — values above this are clipped to the darkest red
VMAX_SOBOL = 0.60

# Number of top variables to select per sector when building the variable list
TOP_N_PER_SECTOR = 5

# Sector display order (row order, top to bottom)
SECTOR_ORDER = [
    "Mobility",
    "Industry",
    "Power",
    "HeatingCooling",
    "Land",
    "Other",
]

# Sector tick labels matching SECTOR_ORDER
SECTOR_LABELS = [
    "Mobility",
    "Industry",
    "Power",
    "Heating &\nCooling",
    "Land Use",
    "Other",
]

# Human-readable variable name mapping applied before any sorting or selection
NAME_MAP = {
    "Monthly_electricity_statistics:Net Electricity Production:Solar": "Solar power",
    "Monthly_electricity_statistics:Net Electricity Production:Total Combustible Fuels": "Fossil fuels power",
    "Monthly_electricity_statistics:Net Electricity Production:Wind": "Wind power",
    "Monthly_electricity_statistics:Used for pumped storage:Electricity": "Pumped storage",
    "Monthly_oil_price_statistics:Diesel (unit/litre):Total price:US dollars": "Diesel price",
    "Monthly_oil_price_statistics:Domestic heating oil (unit/litre):Total price:US dollars": "Heating oil price",
    "Monthly_oil_price_statistics:Gasoline (unit/litre):Total price:US dollars": "Gasoline price",
    "CONTEXT::gdp_quarterly:MillionEUR": "GDP",
    "CONTEXT::climate:rainfall:POP": "Precipitation (pop.)",
    "CONTEXT::climate:temperature:POP": "Temperature (pop.)",
    "CONTEXT::climate:temperature:AREA": "Temperature (area)",
    "CONTEXT::climate:temperature_variability:POP": "Temp. variability (pop.)",
    "CONTEXT::climate:temperature_variability:AREA": "Temp. variability (area)",
    "CONTEXT::climate:rainfall:AREA": "Precipitation (area)",
    "CONTEXT::population:POP_NC": "Population",
    "carbon_price:EU_ETS": "ETS price",
    "heat_pumps:GWH": "Heat pump capacity",
    "land_use:Cropland:Area:1000_ha": "Cropland area",
    "land_use:Forest_land:Area:1000_ha": "Forest area",
    "modal_split_transport:AIR": "Modal split (air)",
    "modal_split_transport:RAIL": "Modal split (rail)",
    "modal_split_transport:ROAD": "Modal split (road)",
    "modal_split_transport:SEA": "Modal split (sea)",
    "energy_taxes:MIOEUR": "Energy taxes",
    "EV_data:EV sales:Cars:BEV:Vehicles": "EV car sales (BEV)",
    "EV_data:EV stock share:Cars:EV:percent": "EV car stock share",
    "crops_livestock:Wheat:Production:t": "Wheat production",
    "crops_livestock:Meat,_Total:Production:t": "Meat production (total)",
    "trade:import_volume_index:Raw_materials:WORLD": "Import vol. (raw mat.)",
    "trade:import_volume_index:Mineral_fuels_lubrificants:WORLD": "Import vol. (fuels)",
    "train_performance:Passenger_trains:Total:Diesel:THS_train_mk": "Passenger trains (diesel)",
    "train_performance:Passenger_trains:Total:Electricity:THS_train_mk": "Passenger trains (elec.)",
    "Monthly_electricity_statistics:Distribution Losses:Electricity": "Electricity losses",
    "Monthly_electricity_statistics:Net Electricity Production:Hydro": "Hydro power",
    "Monthly_electricity_statistics:Net Electricity Production:Total Renewables (Hydro, Geo, Solar, Wind, Other)": "Renewables (total)",
    "Monthly_electricity_statistics:Total Exports:Electricity": "Electricity exports",
    "Monthly_electricity_statistics:Total Imports:Electricity": "Electricity imports",
    "solar_thermal_surface:THS_M2": "Solar thermal surface",
    "energy_consumption:FC_E:GJ_HAB": "Energy cons. (total)",
    "energy_consumption:FC_IND_E:GJ_HAB": "Energy cons. (industry)",
    "energy_consumption:FC_TRA_E:GJ_HAB": "Energy cons. (transport)",
}

# Thematic column order for the x-axis.
# Variables are placed in this exact left-to-right sequence. Any selected
# variable not present in this list is appended alphabetically at the end.
# The grouping logic (separators, readable blocks) follows this order:
#   Economic macro → Energy prices → Electricity mix → Transport → Land/Agri
THEMATIC_ORDER = [
    # ── Fuel / energy prices ──────────────────────────────────────────────
    "Diesel price",
    "Gasoline price",
    "Heating oil price",
    # ── Economic / macro ──────────────────────────────────────────────────
    "GDP",
    "Temperature (pop.)",
    # ── Electricity generation mix ────────────────────────────────────────
    "Fossil fuels power",
    "Solar power",
    "Wind power",
    "Hydro power",
    "Renewables (total)",
    "Pumped storage",
    "Electricity exports",
    "Electricity imports",
    "Electricity losses",
    # ── Heating technology ────────────────────────────────────────────────
    "Heat pump capacity",
    "Solar thermal surface",
    # ── Transport ─────────────────────────────────────────────────────────
    "EV car sales (BEV)",
    "EV car stock share",
    "Modal split (road)",
    "Modal split (rail)",
    "Modal split (air)",
    "Passenger trains (diesel)",
    "Passenger trains (elec.)",
    # ── Land use / agriculture ────────────────────────────────────────────
    "Cropland area",
    "Forest area",
    "Wheat production",
    "Meat production (total)",
    # ── Trade ─────────────────────────────────────────────────────────────
    "Import vol. (fuels)",
    "Import vol. (raw mat.)",
    # ── Energy demand ─────────────────────────────────────────────────────
    "Energy cons. (total)",
    "Energy cons. (industry)",
    "Energy cons. (transport)",
    # ── Climate ───────────────────────────────────────────────────────────
    "Precipitation (pop.)",
    "Temp. variability (pop.)",
    "ETS price",
    "Energy taxes",
    "Population",
]

THEMATIC_SEPARATORS_AFTER = {
    "Heating oil price",       # after fuel prices
    "Temperature (pop.)",      # after GDP + temperature
}


# =============================================================================
# Style Setup
# =============================================================================


def setup_nature_style():
    """
    Configure matplotlib for Nature Climate Change publication style.

    Base font sizes are set slightly above strict Nature minimums (7 pt) so
    the figure remains legible when printed at single-column width (88 mm).
    For a wide two-column figure (180 mm) these sizes are comfortable.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "pdf.fonttype": 42,   # TrueType — text remains editable in Illustrator
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# Variable Selection and Ordering
# =============================================================================


def select_variables_top_per_sector(
    pivot: pd.DataFrame,
    top_n: int,
) -> set:
    """
    Select the union of the top N variables per sector by ST index.

    For each sector column, variables are ranked by their ST value and the
    top N are retained. Taking the union across all sectors ensures that
    any variable that is highly influential for at least one sector appears
    in the figure.

    Args:
        pivot: DataFrame with variables as rows and sectors as columns,
            values are ST indices.
        top_n: Number of top variables to select per sector.

    Returns:
        Set of selected variable names (human-readable, post NAME_MAP).
    """
    selected = set()
    for sector in pivot.columns:
        top_in_sector = pivot[sector].sort_values(ascending=False).head(top_n).index
        selected.update(top_in_sector)
    return selected


def order_variables_thematically(variables: set) -> list:
    """
    Order the selected variable set according to THEMATIC_ORDER.

    Variables present in THEMATIC_ORDER are placed first in that sequence.
    Any remaining variables (not listed) are appended in alphabetical order
    at the end so no selected variable is silently dropped.

    Args:
        variables: Set of selected human-readable variable names.

    Returns:
        Ordered list of variable names for the x-axis.
    """
    ordered = [v for v in THEMATIC_ORDER if v in variables]
    remainder = sorted(variables - set(ordered))
    return ordered + remainder


def get_separator_positions(ordered_vars: list) -> list:
    """
    Compute x-axis positions for thematic group separator lines.

    A separator is drawn at x = i + 0.5 whenever the variable at index i
    appears in THEMATIC_SEPARATORS_AFTER and the variable is not the last
    one in the list.

    Args:
        ordered_vars: Ordered list of variable names on the x-axis.

    Returns:
        List of float x-positions where vertical separators should be drawn.
    """
    positions = []
    for i, var in enumerate(ordered_vars[:-1]):   # skip the last variable
        if var in THEMATIC_SEPARATORS_AFTER:
            positions.append(i + 0.5)
    return positions


# =============================================================================
# Main Figure
# =============================================================================


def create_sobol_heatmap():
    """
    Generate a publication-quality Sobol sensitivity heatmap.

    Layout:
        - Rows (y-axis): emission sectors in SECTOR_ORDER
        - Columns (x-axis): input variables in thematic order
        - Cell colour: total-order Sobol index (ST), white → dark red
        - Cell text: ST value rounded to 2 decimal places;
                     bold for indices above 0.10 that exceed 2× their
                     confidence interval (i.e. robustly non-zero)

    The colorbar is placed above the heatmap as a horizontal strip.
    Vertical separator lines delineate thematic variable groups.
    No panel letter is included (standalone figure).

    Saves outputs as both PDF and PNG in OUTPUT_DIR.
    """
    setup_nature_style()

    print("=" * 70)
    print("GENERATING FIGURE 4: SOBOL SENSITIVITY HEATMAP (full mode)")
    print("=" * 70)

    # ── Load Sobol results ────────────────────────────────────────────────
    sobol_csv = SENSITIVITY_DIR / "sobol_results_full.csv"

    if not sobol_csv.exists():
        print(f"Error: Sobol results not found at {sobol_csv}")
        print("Run scripts.analysis.sobol_analysis first.")
        return

    df_sobol = pd.read_csv(sobol_csv)

    # Apply human-readable variable names before any further processing
    df_sobol["var_clean"] = df_sobol["var"].map(NAME_MAP).fillna(df_sobol["var"])

    # Build pivot tables: rows = variables, columns = sectors
    pivot_st   = df_sobol.pivot(index="var_clean", columns="sector", values="ST")
    pivot_conf = df_sobol.pivot(index="var_clean", columns="sector", values="ST_conf")

    # Restrict to sectors in the specified display order
    available_sectors = [s for s in SECTOR_ORDER if s in pivot_st.columns]
    pivot_st   = pivot_st[available_sectors]
    pivot_conf = pivot_conf[available_sectors]

    # ── Variable selection and ordering ───────────────────────────────────
    print(f"\nSelecting top {TOP_N_PER_SECTOR} variables per sector...")

    selected_vars = select_variables_top_per_sector(pivot_st, TOP_N_PER_SECTOR)
    ordered_vars  = order_variables_thematically(selected_vars)
    n_vars    = len(ordered_vars)
    n_sectors = len(available_sectors)

    print(f"Selected {n_vars} variables across {n_sectors} sectors")
    for i, v in enumerate(ordered_vars):
        print(f"  {i + 1:2d}. {v}")

    # Filter and reorder pivot rows to match the thematic column order
    pivot_st_plot   = pivot_st.loc[ordered_vars]
    pivot_conf_plot = pivot_conf.loc[ordered_vars]

    # ── Colormap ──────────────────────────────────────────────────────────
    # Sequential warm palette: white → cream → orange → dark red.
    # Matches the original dual-panel figure palette.
    colors_sobol = [
        "#FFFFFF",
        "#FFF7EC",
        "#FEE8C8",
        "#FDD49E",
        "#FDBB84",
        "#FC8D59",
        "#EF6548",
        "#D7301F",
        "#990000",
    ]
    cmap_sobol = LinearSegmentedColormap.from_list("warm", colors_sobol, N=256)

    # ── Figure dimensions ─────────────────────────────────────────────────
    # Target a wide landscape figure suitable for a Nature two-column layout
    # (max 180 mm ≈ 7.09 in). Each variable column is ~0.38 in; each sector
    # row is ~0.45 in. Extra height accommodates the colorbar strip.
    fig_width  = max(9.0, 0.38 * n_vars + 2.0)
    fig_height = max(3.2, 0.45 * n_sectors + 1.8)

    fig = plt.figure(figsize=(fig_width, fig_height))

    # GridSpec: top row = colorbar, bottom row = heatmap.
    # The colorbar row is kept slim; heatmap takes the bulk of the height.
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[0.7, 10],
        hspace=0.10,
    )

    ax_cbar    = fig.add_subplot(gs[0, 0])   # horizontal colorbar strip
    ax_heatmap = fig.add_subplot(gs[1, 0])   # main heatmap

    # ── Heatmap data ──────────────────────────────────────────────────────
    # Transpose so rows = sectors (y-axis) and columns = variables (x-axis)
    data_st   = pivot_st_plot.values.T      # shape: (n_sectors, n_vars)
    data_conf = pivot_conf_plot.values.T    # shape: (n_sectors, n_vars)

    im = ax_heatmap.imshow(
        data_st,
        cmap=cmap_sobol,
        aspect="auto",
        vmin=0,
        vmax=VMAX_SOBOL,
    )

    # ── Cell annotations ──────────────────────────────────────────────────
    # Print the ST value in each cell. Use white text on dark backgrounds
    # (ST > 0.40) and bold for robustly non-zero indices (ST > 0.10 and
    # ST > 2 × confidence interval).
    for i in range(n_sectors):
        for j in range(n_vars):
            val  = data_st[i, j]
            conf = data_conf[i, j] if not np.isnan(data_conf[i, j]) else 0.0

            if np.isnan(val):
                continue

            text_color = "white" if val > 0.40 else "#333333"
            weight = "bold" if val > 0.10 and val > 2 * conf else "normal"

            ax_heatmap.text(
                j, i,
                f"{val:.2f}",
                ha="center", va="center",
                fontsize=7.5,
                color=text_color,
                fontweight=weight,
            )

    # ── y-axis: sector labels ─────────────────────────────────────────────
    ax_heatmap.set_yticks(range(n_sectors))
    ax_heatmap.set_yticklabels(
        [SECTOR_LABELS[SECTOR_ORDER.index(s)] for s in available_sectors],
        fontsize=9.5,
    )

    # ── x-axis: variable labels, rotated 45° ─────────────────────────────
    ax_heatmap.set_xticks(range(n_vars))
    ax_heatmap.set_xticklabels(
        ordered_vars,
        rotation=45, ha="right",
        fontsize=8.5,
    )

    ax_heatmap.set_xlabel("Input variable", fontsize=10, labelpad=6)
    ax_heatmap.set_ylabel("Emission sector", fontsize=10, labelpad=6)

    # Remove tick marks (the heatmap grid provides sufficient visual structure)
    ax_heatmap.tick_params(axis="both", which="both", length=0)

    # ── White cell separators ─────────────────────────────────────────────
    for i in range(n_sectors + 1):
        ax_heatmap.axhline(i - 0.5, color="white", linewidth=0.7)
    for j in range(n_vars + 1):
        ax_heatmap.axvline(j - 0.5, color="white", linewidth=0.7)

    # ── Thematic group separators (darker vertical lines) ─────────────────
    # Drawn on top of cell separators to visually group related variables.
    for pos in get_separator_positions(ordered_vars):
        ax_heatmap.axvline(pos, color="#333333", linewidth=1.4)

    # ── Spines ────────────────────────────────────────────────────────────
    for spine in ax_heatmap.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#333333")

    # ── Colorbar ──────────────────────────────────────────────────────────
    # Horizontal strip above the heatmap; label and ticks on top edge.
    cbar = plt.colorbar(im, cax=ax_cbar, orientation="horizontal")
    cbar.set_label("Total-order Sobol index (ST)", fontsize=9.5, labelpad=4)
    cbar.ax.tick_params(labelsize=8.5, length=3, width=0.5, pad=2)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.outline.set_linewidth(0.5)
    cbar.set_ticks([0, 0.15, 0.30, 0.45, 0.60])

    # ── Save ──────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for fmt in ["pdf", "png"]:
        out_path = OUTPUT_DIR / f"fig4_sobol_sensitivity_heatmap_full.{fmt}"
        plt.savefig(
            out_path,
            bbox_inches="tight",
            dpi=300,
            facecolor="white",
            edgecolor="none",
        )
        print(f"Saved: {out_path}")

    plt.close()


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """
    Generate Figure 4: Sobol Sensitivity Heatmap (full variable mode).

    Loads precomputed Sobol results from data/sensitivity/sobol_results_full.csv,
    selects the top variables per sector, orders them thematically, and
    produces a wide publication-quality heatmap with sectors on the y-axis
    and variables on the x-axis.
    """
    create_sobol_heatmap()
    print("\nFigure 4 generation complete.")


if __name__ == "__main__":
    main()