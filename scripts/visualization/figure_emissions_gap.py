"""
Figure 1: Emissions Gap Scatter Plot.

Visualizes the deviation of projected 2030 emissions from national climate targets
for each EU member state, comparing this study's projections against OECD, EEA,
and PyPSA scenarios.

This script generates a lollipop chart showing how far each country is from
meeting its Fit for 55 / ESR burden-sharing target, with multiple projection
sources for comparison.

Usage:
    python -m scripts.visualization.figure_emissions_gap

Outputs:
    - outputs/figures/fig1_emissions_gap.pdf
    - outputs/figures/fig1_emissions_gap.png
    - outputs/figures/fig1_emissions_gap.svg
    - outputs/tables/fig1_summary_table.csv

Reference:
    Figure 1 in the paper shows deviation from 2030 targets across EU27.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# =============================================================================
# Configuration
# =============================================================================

# Paths
DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")
MC_PROJECTIONS_PATH = Path("data/projections/mc_projections.csv")
POPULATION_HIST_PATH = Path("data/full_timeseries/population.csv")
POPULATION_PROJ_PATH = Path("data/full_timeseries/projections/population.csv")

# External data paths
OECD_PATH = Path("data/external/oecd_projections.csv")
EEA_PATH = Path("data/external/eea_projections.xlsx")
PYPSA_PATH = Path("data/external/pypsa_projections.csv")

# Output paths
OUTPUT_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")

# Emission sectors
OUTPUT_SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]

# EU27 country codes
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

# Country code to full name mapping
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

# PyPSA country code mapping (Greece GR -> EL)
PYPSA_COUNTRY_MAPPING = {"GR": "EL"}

# ISO3 to ISO2 mapping for OECD data
ISO3_TO_ISO2 = {
    "AUT": "AT",
    "BEL": "BE",
    "BGR": "BG",
    "HRV": "HR",
    "CYP": "CY",
    "CZE": "CZ",
    "DNK": "DK",
    "EST": "EE",
    "FIN": "FI",
    "FRA": "FR",
    "DEU": "DE",
    "HUN": "HU",
    "IRL": "IE",
    "ITA": "IT",
    "LVA": "LV",
    "LTU": "LT",
    "LUX": "LU",
    "MLT": "MT",
    "NLD": "NL",
    "POL": "PL",
    "PRT": "PT",
    "ROU": "RO",
    "SVK": "SK",
    "SVN": "SI",
    "ESP": "ES",
    "SWE": "SE",
    "GRC": "EL",
    "EU27": "EU27",
}

# ESR 2030 reduction targets (% change from 2005 baseline)
ESR_TARGETS_2030 = {
    "AT": -48.0,
    "BE": -47.0,
    "BG": -10.0,
    "HR": -16.7,
    "CY": -32.0,
    "CZ": -26.0,
    "DK": -50.0,
    "EE": -24.0,
    "FI": -50.0,
    "FR": -47.5,
    "DE": -50.0,
    "EL": -22.7,
    "HU": -18.7,
    "IE": -42.0,
    "IT": -43.7,
    "LV": -17.0,
    "LT": -21.0,
    "LU": -50.0,
    "MT": -19.0,
    "NL": -48.0,
    "PL": -17.7,
    "PT": -28.7,
    "RO": -12.7,
    "SK": -22.7,
    "SI": -27.0,
    "ES": -37.7,
    "SE": -50.0,
}


# =============================================================================
# Nature Climate Change Style Settings
# =============================================================================


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
            "pdf.fonttype": 42,  # TrueType fonts (Nature requirement)
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# Data Loading Functions
# =============================================================================


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


def load_mc_projections(dataset, population_df: pd.DataFrame) -> pd.DataFrame:
    """Load and process Monte Carlo projections."""
    df_mc = pd.read_csv(MC_PROJECTIONS_PATH)
    df_mc["geo"] = df_mc["geo"].astype(str)

    # Unnormalize emissions
    for s in OUTPUT_SECTORS:
        mean_ = dataset.precomputed_scaling_params[s]["mean"]
        std_ = dataset.precomputed_scaling_params[s]["std"]
        df_mc[f"{s}_unnorm"] = np.clip(df_mc[f"emissions_{s}"] * std_ + mean_, 0, None)

    # Merge population and compute total emissions
    df_mc = df_mc.merge(population_df, on=["geo", "year"], how="left")
    df_mc["total_CO2"] = (
        df_mc[[f"{s}_unnorm" for s in OUTPUT_SECTORS]].sum(axis=1) * df_mc["population"]
    )

    # Aggregate by country and year
    forecast_summary = (
        df_mc.groupby(["geo", "year"])["total_CO2"]
        .agg(["mean", lambda x: np.quantile(x, 0.05), lambda x: np.quantile(x, 0.95)])
        .reset_index()
    )
    forecast_summary.columns = [
        "geo",
        "year",
        "total_CO2_mean",
        "total_CO2_low",
        "total_CO2_high",
    ]

    # Add EU27 aggregate
    df_mc_eu = forecast_summary[forecast_summary["geo"].isin(EU27_COUNTRIES)].copy()
    forecast_eu = df_mc_eu.groupby("year", as_index=False).agg(
        {
            "total_CO2_mean": "sum",
            "total_CO2_low": "sum",
            "total_CO2_high": "sum",
        }
    )
    forecast_eu["geo"] = "EU27"
    forecast_summary = pd.concat([forecast_summary, forecast_eu], ignore_index=True)

    return forecast_summary


def load_oecd_projections() -> pd.DataFrame:
    """Load OECD emission projections."""
    if not OECD_PATH.exists():
        print(f"Warning: OECD data not found at {OECD_PATH}")
        return pd.DataFrame()

    oecd_df = pd.read_csv(OECD_PATH)
    oecd_df = oecd_df[["REF_AREA", "SCENARIO", "TIME_PERIOD", "OBS_VALUE"]].copy()
    oecd_df = oecd_df.rename(
        columns={
            "REF_AREA": "geo_iso3",
            "SCENARIO": "scenario",
            "TIME_PERIOD": "year",
            "OBS_VALUE": "value",
        }
    )
    oecd_df["value"] = pd.to_numeric(oecd_df["value"], errors="coerce")
    oecd_df = oecd_df.dropna(subset=["value"])
    oecd_df["geo"] = oecd_df["geo_iso3"].map(ISO3_TO_ISO2)

    # Filter to 2030 and convert to tonnes
    oecd_2030 = oecd_df[oecd_df["year"] == 2030].copy()
    oecd_2030["value_tonnes"] = oecd_2030["value"] * 1e6

    return oecd_2030


def load_eea_projections() -> pd.DataFrame:
    """Load EEA emission projections."""
    if not EEA_PATH.exists():
        print(f"Warning: EEA data not found at {EEA_PATH}")
        return pd.DataFrame()

    eea_df = pd.read_excel(EEA_PATH, sheet_name="Database")
    eea_2030 = eea_df[
        (eea_df["Year"] == 2030)
        & (eea_df["Category"] == "Total excluding LULUCF")
        & (eea_df["Gas"] == "ESR emissions (ktCO2e)")
    ].copy()
    eea_2030 = eea_2030.rename(columns={"CountryCode": "geo", "Scenario": "scenario"})
    eea_2030["value_tonnes"] = eea_2030["Gapfilled"] * 1000
    eea_2030 = eea_2030[["geo", "scenario", "value_tonnes"]]

    return eea_2030


def load_pypsa_projections() -> pd.DataFrame:
    """Load PyPSA emission projections."""
    if not PYPSA_PATH.exists():
        print(f"Warning: PyPSA data not found at {PYPSA_PATH}")
        return pd.DataFrame()

    pypsa_df = pd.read_csv(PYPSA_PATH)
    pypsa_df["country"] = pypsa_df["country"].replace(PYPSA_COUNTRY_MAPPING)
    pypsa_totals = pypsa_df.groupby(["country", "scenario"], as_index=False)[
        "value"
    ].sum()
    pypsa_totals.columns = ["geo", "scenario", "total_CO2"]

    return pypsa_totals


def compute_targets(oecd_df: pd.DataFrame) -> pd.DataFrame:
    """Compute corrected 2030 targets based on ESR burden sharing."""
    # Build ESR targets dataframe
    esr_targets = pd.DataFrame(
        [
            {"geo": geo, "target_pct": pct / 100.0}
            for geo, pct in ESR_TARGETS_2030.items()
        ]
    )

    # Get 2005 baseline emissions
    baseline_2005 = (
        oecd_df[oecd_df["year"] == 2005].groupby("geo", as_index=False)["value"].mean()
    )
    baseline_2005 = baseline_2005.rename(columns={"value": "emissions_2005_Mt"})

    # Add Cyprus and Malta baselines (from Eurostat, not in OECD)
    cy_mt = pd.DataFrame({"geo": ["CY", "MT"], "emissions_2005_Mt": [7.56, 2.56]})
    baseline_2005 = pd.concat([baseline_2005, cy_mt], ignore_index=True)

    # Merge and compute raw ESR targets
    esr_targets = esr_targets.merge(baseline_2005, on="geo", how="left")
    esr_targets["esr_target_2030_Mt"] = esr_targets["emissions_2005_Mt"] * (
        1 + esr_targets["target_pct"]
    )

    # Correct to match EU27 FF55 aggregate (-55% from 1990)
    eu27_countries_with_targets = esr_targets[esr_targets["geo"].isin(EU27_COUNTRIES)]
    eu27_esr_aggregate = eu27_countries_with_targets["esr_target_2030_Mt"].sum()

    # Get 1990 EU27 baseline
    eu27_1990 = oecd_df[(oecd_df["year"] == 1990) & (oecd_df["geo"] == "EU27")]
    if not eu27_1990.empty:
        eu27_1990_emissions = eu27_1990.iloc[0]["value"]
        eu27_ff55_target = eu27_1990_emissions * 0.45  # 55% reduction = 45% remaining
        correction_ratio = eu27_ff55_target / eu27_esr_aggregate
    else:
        correction_ratio = 1.0

    # Apply correction
    esr_targets["corrected_target_2030_Mt"] = (
        esr_targets["esr_target_2030_Mt"] * correction_ratio
    )
    esr_targets["corrected_target_2030_tonnes"] = (
        esr_targets["corrected_target_2030_Mt"] * 1e6
    )

    # Add EU27 aggregate target
    eu27_target = pd.DataFrame(
        {
            "geo": ["EU27"],
            "corrected_target_2030_tonnes": (
                [eu27_ff55_target * 1e6] if not eu27_1990.empty else [0]
            ),
        }
    )
    esr_targets = pd.concat([esr_targets, eu27_target], ignore_index=True)

    return esr_targets


# =============================================================================
# Visualization Functions
# =============================================================================


def create_emissions_gap_plot(comparison_df: pd.DataFrame) -> None:
    """Create the emissions gap lollipop chart."""
    setup_nature_style()

    # Color palette (Paul Tol colorblind-friendly)
    COLORS = {
        "mc": "#332288",  # Dark indigo - This study
        "oecd_bau": "#AA4499",  # Purple - OECD BAU
        "oecd_et": "#CC6677",  # Dusty rose - OECD ET
        "eea_wam": "#117733",  # Dark green - EEA WAM
        "eea_wem": "#44AA99",  # Teal - EEA WEM
        "pypsa_base": "#DDCC77",  # Sand - PyPSA Baseline
        "pypsa_ff55": "#88CCEE",  # Light cyan - PyPSA FF55
        "target": "#2c3e50",  # Dark gray for target line
        "stem": "#aab7b8",  # Light gray for stems
        "bg_pos": "#fdedec",  # Very light red bg
        "bg_neg": "#eafaf1",  # Very light green bg
    }

    fig, ax = plt.subplots(figsize=(6.0, 7.5))

    n_countries = len(comparison_df)
    y_pos = np.arange(n_countries)

    # Background panels
    ax.axvspan(-100, 0, color=COLORS["bg_neg"], alpha=0.7, zorder=0)
    ax.axvspan(0, 300, color=COLORS["bg_pos"], alpha=0.7, zorder=0)

    # Y-offset jitter for each projection type
    jitter = {
        "eea_wam_pct": 0.36,
        "eea_wem_pct": 0.24,
        "pypsa_ff55_pct": 0.12,
        "mc_pct": 0.0,
        "pypsa_base_pct": -0.12,
        "oecd_et1_pct": -0.24,
        "oecd_bau1_pct": -0.36,
    }

    # Draw stems (lollipop style)
    for i, row in comparison_df.iterrows():
        if pd.isna(row["mc_pct"]):
            continue
        ax.hlines(
            y=i,
            xmin=0,
            xmax=row["mc_pct"],
            color=COLORS["stem"],
            linewidth=1.0,
            zorder=2,
        )

    # Plot MC results (main)
    mc_mask = ~comparison_df["mc_pct"].isna()
    eu_mask = comparison_df["geo"] == "EU27"

    # Regular countries
    regular_mask = mc_mask & ~eu_mask
    ax.scatter(
        comparison_df.loc[regular_mask, "mc_pct"],
        y_pos[regular_mask] + jitter["mc_pct"],
        s=80,
        c=COLORS["mc"],
        marker="o",
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
        label="This study",
    )

    # EU27 (larger)
    eu27_mask = mc_mask & eu_mask
    ax.scatter(
        comparison_df.loc[eu27_mask, "mc_pct"],
        y_pos[eu27_mask] + jitter["mc_pct"],
        s=160,
        c=COLORS["mc"],
        marker="o",
        edgecolors="white",
        linewidths=1.0,
        zorder=7,
    )

    # Other projection sources
    marker_styles = {
        "oecd_bau1_pct": {
            "color": COLORS["oecd_bau"],
            "marker": "D",
            "s": 20,
            "s_eu": 40,
            "label": "OECD Business as Usual",
        },
        "oecd_et1_pct": {
            "color": COLORS["oecd_et"],
            "marker": "D",
            "s": 20,
            "s_eu": 40,
            "label": "OECD Energy Transition",
        },
        "eea_wam_pct": {
            "color": COLORS["eea_wam"],
            "marker": "^",
            "s": 24,
            "s_eu": 48,
            "label": "EEA With Additional Measures",
        },
        "eea_wem_pct": {
            "color": COLORS["eea_wem"],
            "marker": "^",
            "s": 24,
            "s_eu": 48,
            "label": "EEA With Existing Measures",
        },
        "pypsa_base_pct": {
            "color": COLORS["pypsa_base"],
            "marker": "s",
            "s": 20,
            "s_eu": 40,
            "label": "PyPSA Baseline",
        },
        "pypsa_ff55_pct": {
            "color": COLORS["pypsa_ff55"],
            "marker": "s",
            "s": 20,
            "s_eu": 40,
            "label": "PyPSA Fit for 55",
        },
    }

    for col, style in marker_styles.items():
        if col not in comparison_df.columns:
            continue
        mask = ~comparison_df[col].isna()
        if mask.any():
            eu_in_mask = comparison_df.loc[mask, "geo"] == "EU27"
            sizes = [style["s_eu"] if eu else style["s"] for eu in eu_in_mask]
            ax.scatter(
                comparison_df.loc[mask, col],
                y_pos[mask] + jitter[col],
                c=style["color"],
                marker=style["marker"],
                s=sizes,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.3,
                zorder=5,
                label=style["label"],
            )

    # Target line at 0
    ax.axvline(0, color=COLORS["target"], linewidth=1.8, zorder=3)
    ax.text(
        0,
        n_countries + 0.6,
        "2030 TARGET",
        ha="center",
        va="bottom",
        fontsize=6.5,
        fontweight="bold",
        color=COLORS["target"],
    )

    # Y-axis labels
    ax.set_yticks(y_pos)
    y_labels = [COUNTRY_NAMES.get(geo, geo) for geo in comparison_df["geo"].tolist()]
    ax.set_yticklabels(y_labels)

    # Bold EU27
    for label in ax.get_yticklabels():
        if label.get_text() == "EU27":
            label.set_fontweight("bold")
            label.set_fontsize(8)

    # X-axis
    ax.set_xlim(-80, 280)
    ax.set_xlabel("Deviation from 2030 target (Fit for 55, ESR burden sharing) (%)")

    # Directional labels
    ax.text(
        -40,
        -1.3,
        "← Meeting target",
        fontsize=6.5,
        ha="center",
        color="#1e8449",
        fontstyle="italic",
    )
    ax.text(
        140,
        -1.3,
        "Exceeding emissions →",
        fontsize=6.5,
        ha="center",
        color="#922b21",
        fontstyle="italic",
    )

    # Grid
    ax.xaxis.grid(True, linestyle="-", alpha=0.25, color="#bdc3c7", linewidth=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["mc"],
            markersize=7.5,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label="This study",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=COLORS["pypsa_base"],
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.2,
            label="PyPSA Baseline",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=COLORS["pypsa_ff55"],
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.2,
            label="PyPSA Fit for 55",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=COLORS["oecd_bau"],
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.2,
            label="OECD Business as Usual",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=COLORS["oecd_et"],
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.2,
            label="OECD Energy Transition",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=COLORS["eea_wam"],
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.2,
            label="EEA With Additional Measures",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=COLORS["eea_wem"],
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.2,
            label="EEA With Existing Measures",
        ),
    ]

    leg = ax.legend(
        handles=legend_elements,
        loc="lower right",
        frameon=True,
        fontsize=5.5,
        framealpha=0.95,
        edgecolor="#bdc3c7",
        handletextpad=0.5,
        labelspacing=0.4,
        borderpad=0.5,
    )
    leg.get_frame().set_linewidth(0.4)

    # Spines
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color("#5d6d7e")

    plt.tight_layout()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ["png", "pdf", "svg"]:
        plt.savefig(
            OUTPUT_DIR / f"fig1_emissions_gap.{fmt}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close()

    print(f"Saved: {OUTPUT_DIR}/fig1_emissions_gap.[png|pdf|svg]")


# =============================================================================
# Main Function
# =============================================================================


def main():
    """Generate Figure 1: Emissions Gap Scatter Plot."""
    print("=" * 70)
    print("GENERATING FIGURE 1: EMISSIONS GAP SCATTER PLOT")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    dataset = load_dataset(DATASET_PATH)
    population_df = load_population_data()
    forecast_summary = load_mc_projections(dataset, population_df)

    # Load external projections
    oecd_2030 = load_oecd_projections()
    eea_2030 = load_eea_projections()
    pypsa_2030 = load_pypsa_projections()

    # Compute targets
    print("Computing targets...")
    esr_targets = compute_targets(oecd_2030 if not oecd_2030.empty else pd.DataFrame())

    # Build comparison dataframe
    print("Building comparison data...")
    comparison_data = []

    for geo in esr_targets["geo"].unique():
        target_row = esr_targets[esr_targets["geo"] == geo]
        if target_row.empty:
            continue

        target_value = target_row.iloc[0]["corrected_target_2030_tonnes"]
        data = {"geo": geo, "target": target_value}

        # This study (MC projections)
        mc_2030 = forecast_summary[
            (forecast_summary["geo"] == geo) & (forecast_summary["year"] == 2030)
        ]
        if not mc_2030.empty:
            data["mc_pct"] = (
                (mc_2030.iloc[0]["total_CO2_mean"] - target_value) / target_value
            ) * 100
        else:
            data["mc_pct"] = np.nan

        # OECD scenarios
        if not oecd_2030.empty:
            for scenario, col in [("BAU1", "oecd_bau1_pct"), ("ET1", "oecd_et1_pct")]:
                oecd_row = oecd_2030[
                    (oecd_2030["geo"] == geo) & (oecd_2030["scenario"] == scenario)
                ]
                data[col] = (
                    ((oecd_row.iloc[0]["value_tonnes"] - target_value) / target_value)
                    * 100
                    if not oecd_row.empty
                    else np.nan
                )

        # EEA scenarios
        if not eea_2030.empty:
            for scenario, col in [("WAM", "eea_wam_pct"), ("WEM", "eea_wem_pct")]:
                eea_row = eea_2030[
                    (eea_2030["geo"] == geo) & (eea_2030["scenario"] == scenario)
                ]
                data[col] = (
                    ((eea_row.iloc[0]["value_tonnes"] - target_value) / target_value)
                    * 100
                    if not eea_row.empty
                    else np.nan
                )

        # PyPSA scenarios
        if not pypsa_2030.empty:
            pypsa_base = pypsa_2030[
                (pypsa_2030["geo"] == geo) & (pypsa_2030["scenario"] == "base")
            ]
            data["pypsa_base_pct"] = (
                ((pypsa_base.iloc[0]["total_CO2"] - target_value) / target_value) * 100
                if not pypsa_base.empty
                else np.nan
            )

            pypsa_ff55 = pypsa_2030[
                (pypsa_2030["geo"] == geo)
                & (pypsa_2030["scenario"].isin(["policy", "FF55", "ff55"]))
            ]
            data["pypsa_ff55_pct"] = (
                ((pypsa_ff55.iloc[0]["total_CO2"] - target_value) / target_value) * 100
                if not pypsa_ff55.empty
                else np.nan
            )

        comparison_data.append(data)

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by MC percentage
    has_mc = ~comparison_df["mc_pct"].isna()
    df_with_mc = comparison_df[has_mc].sort_values("mc_pct").reset_index(drop=True)
    df_without_mc = comparison_df[~has_mc].reset_index(drop=True)
    comparison_df = pd.concat([df_with_mc, df_without_mc], ignore_index=True)

    # Create plot
    print("\nCreating visualization...")
    create_emissions_gap_plot(comparison_df)

    # Export summary table
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df["country_name"] = comparison_df["geo"].map(COUNTRY_NAMES)
    comparison_df.to_csv(TABLE_DIR / "fig1_summary_table.csv", index=False)
    print(f"Exported: {TABLE_DIR}/fig1_summary_table.csv")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    mc_valid = comparison_df[comparison_df["geo"] != "EU27"]["mc_pct"].dropna()
    on_track = (mc_valid < 0).sum()
    print(f"\nCountries on track (this study): {on_track}/{len(mc_valid)}")

    eu27_row = comparison_df[comparison_df["geo"] == "EU27"]
    if not eu27_row.empty:
        print(f"EU27 aggregate deviation: {eu27_row.iloc[0]['mc_pct']:+.1f}%")

    print("\nFigure 1 generation complete!")


if __name__ == "__main__":
    main()
