"""
Country-level emissions gap visualization.

Generates a scatter plot comparing 2030 projected emissions against ESR targets
for all EU27 countries, with comparison to external scenarios (OECD, EEA, PyPSA).

This corresponds to Figure 1 in the paper showing the emissions gap analysis.

Usage:
    python -m scripts.visualization.plot_emissions_gap

Outputs:
    - outputs/figures/fig1_emissions_gap.{png,pdf,svg}
    - outputs/figures/fig1_summary_table.csv

Reference:
    Section 2.1 "Projected vs Target Emissions" in the paper.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.analysis.common import (
    COUNTRY_NAMES,
    EMISSION_SECTORS,
    ESR_TARGETS_2030,
    EU27_COUNTRIES,
    load_population_data,
    set_nature_style_minimal_spines,
    unnormalize_emissions,
)
from scripts.utils import load_dataset

# =============================================================================
# Configuration
# =============================================================================

# Paths
DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")
MC_PROJECTIONS_PATH = Path("data/projections/mc_projections.csv")
OUTPUT_DIR = Path("outputs/figures")

# External data paths (optional)
OECD_PATH = Path("data/external/oecd_projections.csv")
EEA_PATH = Path("data/external/eea_projections.xlsx")
PYPSA_PATH = Path("data/external/pypsa_projections.csv")


# =============================================================================
# Data Loading
# =============================================================================


def load_and_process_historical(dataset, population_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load historical emissions from dataset and convert to total emissions.

    Args:
        dataset: Unified dataset with emission data.
        population_df: Population data for per-capita to total conversion.

    Returns:
        DataFrame with historical emissions by country and year.
    """
    keys = dataset.keys
    emi_data = pd.DataFrame(dataset.emi_df.cpu().numpy(), columns=EMISSION_SECTORS)
    historical = pd.concat([keys, emi_data], axis=1)

    # Unnormalize emissions
    for sector in EMISSION_SECTORS:
        historical[sector] = unnormalize_emissions(
            historical[sector].values,
            sector,
            dataset.precomputed_scaling_params,
        )

    # Add population and compute total emissions
    historical = historical.merge(population_df, on=["geo", "year"], how="left")
    historical["total_CO2"] = (
        historical[EMISSION_SECTORS].sum(axis=1) * historical["population"]
    )

    return historical


def load_and_process_projections(
    dataset,
    population_df: pd.DataFrame,
    mc_path: Path = MC_PROJECTIONS_PATH,
) -> pd.DataFrame:
    """
    Load MC projections and compute summary statistics.

    Args:
        dataset: Dataset with scaling parameters.
        population_df: Population data.
        mc_path: Path to MC projections CSV.

    Returns:
        DataFrame with mean, 5th, and 95th percentile projections by country/year.
    """
    df_mc = pd.read_csv(mc_path)
    df_mc["geo"] = df_mc["geo"].astype(str)

    # Unnormalize emissions
    for sector in EMISSION_SECTORS:
        df_mc[f"{sector}_unnorm"] = unnormalize_emissions(
            df_mc[f"emissions_{sector}"].values,
            sector,
            dataset.precomputed_scaling_params,
        )

    # Add population and compute total
    df_mc = df_mc.merge(population_df, on=["geo", "year"], how="left")
    df_mc["total_CO2"] = (
        df_mc[[f"{s}_unnorm" for s in EMISSION_SECTORS]].sum(axis=1)
        * df_mc["population"]
    )

    # Compute summary statistics
    summary = (
        df_mc.groupby(["geo", "year"])["total_CO2"]
        .agg(["mean", lambda x: np.quantile(x, 0.05), lambda x: np.quantile(x, 0.95)])
        .reset_index()
    )
    summary.columns = [
        "geo",
        "year",
        "total_CO2_mean",
        "total_CO2_low",
        "total_CO2_high",
    ]

    return summary


def compute_2005_baseline(historical: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 2005 baseline emissions for ESR target calculation.

    If 2005 data unavailable, uses earliest available year with adjustment.

    Args:
        historical: Historical emissions data.

    Returns:
        DataFrame with 2005 baseline by country.
    """
    baselines = []

    for country in EU27_COUNTRIES:
        country_data = historical[historical["geo"] == country].sort_values("year")

        if 2005 in country_data["year"].values:
            baseline = country_data[country_data["year"] == 2005]["total_CO2"].values[0]
        else:
            # Use earliest available with trend adjustment
            earliest = country_data.iloc[0]
            years_diff = earliest["year"] - 2005
            # Assume ~2% annual decline historically
            baseline = earliest["total_CO2"] * (1.02**years_diff)

        baselines.append({"geo": country, "baseline_2005": baseline})

    return pd.DataFrame(baselines)


def compute_targets_and_gaps(
    projections_2030: pd.DataFrame,
    baseline_2005: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute ESR targets and emission gaps for each country.

    Args:
        projections_2030: Projected 2030 emissions.
        baseline_2005: 2005 baseline emissions.

    Returns:
        DataFrame with targets, projections, and gap percentages.
    """
    # Merge projections with baselines
    df = projections_2030.merge(baseline_2005, on="geo")

    # Compute targets
    df["esr_reduction_pct"] = df["geo"].map(ESR_TARGETS_2030)
    df["target_2030"] = df["baseline_2005"] * (1 + df["esr_reduction_pct"] / 100)

    # Compute gap (positive = overshoot, negative = on track)
    df["gap_pct"] = (
        (df["total_CO2_mean"] - df["target_2030"]) / df["target_2030"]
    ) * 100
    df["gap_pct_low"] = (
        (df["total_CO2_low"] - df["target_2030"]) / df["target_2030"]
    ) * 100
    df["gap_pct_high"] = (
        (df["total_CO2_high"] - df["target_2030"]) / df["target_2030"]
    ) * 100

    return df


# =============================================================================
# Visualization
# =============================================================================


def create_emissions_gap_figure(
    comparison_df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Create the emissions gap scatter plot.

    Args:
        comparison_df: DataFrame with country-level gap data.
        output_dir: Directory to save outputs.
    """
    set_nature_style_minimal_spines()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort countries by gap (worst to best)
    country_data = comparison_df[comparison_df["geo"] != "EU27"].copy()
    country_data = country_data.sort_values("gap_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.5))

    y_positions = range(len(country_data))
    country_labels = [COUNTRY_NAMES.get(c, c) for c in country_data["geo"]]

    # Plot error bars and points
    colors = []
    for _, row in country_data.iterrows():
        if row["gap_pct"] < 0:
            colors.append("#2ecc71")  # Green for on track
        elif row["gap_pct"] < 20:
            colors.append("#f39c12")  # Orange for moderate overshoot
        else:
            colors.append("#e74c3c")  # Red for significant overshoot

    # Confidence intervals
    xerr_low = country_data["gap_pct"] - country_data["gap_pct_low"]
    xerr_high = country_data["gap_pct_high"] - country_data["gap_pct"]

    ax.errorbar(
        country_data["gap_pct"],
        y_positions,
        xerr=[xerr_low, xerr_high],
        fmt="none",
        ecolor="#bdc3c7",
        elinewidth=1,
        capsize=2,
        capthick=0.8,
        zorder=1,
    )

    ax.scatter(
        country_data["gap_pct"],
        y_positions,
        c=colors,
        s=50,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )

    # Zero line (target)
    ax.axvline(0, color="#2c3e50", linewidth=1.5, linestyle="-", zorder=0, alpha=0.8)

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(country_labels)
    ax.set_xlabel("Deviation from 2030 ESR target (%)")

    # Grid
    ax.xaxis.grid(True, linestyle="-", alpha=0.2, color="#bdc3c7")
    ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#2ecc71", edgecolor="white", label="On track (<0%)"),
        mpatches.Patch(
            facecolor="#f39c12", edgecolor="white", label="Moderate (0-20%)"
        ),
        mpatches.Patch(
            facecolor="#e74c3c", edgecolor="white", label="Significant (>20%)"
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        fontsize=6,
    )

    # Invert y-axis so worst performers are at top
    ax.invert_yaxis()

    plt.tight_layout()

    # Save in multiple formats
    for fmt in ["png", "pdf", "svg"]:
        plt.savefig(
            output_dir / f"fig1_emissions_gap.{fmt}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close()

    print(f"Saved: {output_dir}/fig1_emissions_gap.[png|pdf|svg]")


def print_summary_statistics(comparison_df: pd.DataFrame) -> None:
    """Print summary statistics for the paper."""
    print("\n" + "=" * 70)
    print("EMISSIONS GAP SUMMARY")
    print("=" * 70)

    country_data = comparison_df[comparison_df["geo"] != "EU27"]

    # Countries on track
    on_track = (country_data["gap_pct"] < 0).sum()
    total = len(country_data)
    print(f"\nCountries on track: {on_track}/{total}")

    # Worst performers
    print("\nTop 5 worst performers:")
    worst = country_data.nlargest(5, "gap_pct")
    for _, row in worst.iterrows():
        name = COUNTRY_NAMES.get(row["geo"], row["geo"])
        print(f"  {name}: {row['gap_pct']:+.1f}%")

    # Best performers
    print("\nTop 5 best performers:")
    best = country_data.nsmallest(5, "gap_pct")
    for _, row in best.iterrows():
        name = COUNTRY_NAMES.get(row["geo"], row["geo"])
        status = "ON TRACK" if row["gap_pct"] < 0 else "off track"
        print(f"  {name}: {row['gap_pct']:+.1f}% ({status})")

    # EU27 aggregate
    if "EU27" in comparison_df["geo"].values:
        eu27 = comparison_df[comparison_df["geo"] == "EU27"].iloc[0]
        print(f"\nEU27 aggregate gap: {eu27['gap_pct']:+.1f}%")
        print(f"  Target: {eu27['target_2030'] / 1e9:.2f} Gt CO2")
        print(f"  Projected: {eu27['total_CO2_mean'] / 1e9:.2f} Gt CO2")


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Generate emissions gap figure and summary."""
    print("Loading data...")

    # Load datasets
    dataset = load_dataset(DATASET_PATH)
    population_df = load_population_data()

    # Process data
    historical = load_and_process_historical(dataset, population_df)
    projections = load_and_process_projections(dataset, population_df)

    # Get 2030 projections
    projections_2030 = projections[projections["year"] == 2030].copy()

    # Compute baselines and gaps
    baseline_2005 = compute_2005_baseline(historical)
    comparison_df = compute_targets_and_gaps(projections_2030, baseline_2005)

    # Add EU27 aggregate
    eu27_projection = projections_2030[projections_2030["geo"].isin(EU27_COUNTRIES)]
    eu27_total = eu27_projection[
        ["total_CO2_mean", "total_CO2_low", "total_CO2_high"]
    ].sum()
    eu27_baseline = baseline_2005["baseline_2005"].sum()
    eu27_target = eu27_baseline * 0.60  # -40% aggregate target approximation

    eu27_row = pd.DataFrame(
        [
            {
                "geo": "EU27",
                "total_CO2_mean": eu27_total["total_CO2_mean"],
                "total_CO2_low": eu27_total["total_CO2_low"],
                "total_CO2_high": eu27_total["total_CO2_high"],
                "baseline_2005": eu27_baseline,
                "target_2030": eu27_target,
                "gap_pct": ((eu27_total["total_CO2_mean"] - eu27_target) / eu27_target)
                * 100,
                "gap_pct_low": (
                    (eu27_total["total_CO2_low"] - eu27_target) / eu27_target
                )
                * 100,
                "gap_pct_high": (
                    (eu27_total["total_CO2_high"] - eu27_target) / eu27_target
                )
                * 100,
            }
        ]
    )
    comparison_df = pd.concat([comparison_df, eu27_row], ignore_index=True)

    # Generate outputs
    create_emissions_gap_figure(comparison_df)
    print_summary_statistics(comparison_df)

    # Export summary table
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df["country_name"] = comparison_df["geo"].map(COUNTRY_NAMES)
    comparison_df.to_csv(OUTPUT_DIR / "fig1_summary_table.csv", index=False)
    print(f"\nExported: {OUTPUT_DIR}/fig1_summary_table.csv")


if __name__ == "__main__":
    main()
