"""
Common constants, utilities, and styling for analysis and visualization.

This module provides:
- EU27 country codes and name mappings
- Emission sector definitions
- Nature Climate Change publication styling
- Data loading and unnormalization utilities
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Country Definitions
# =============================================================================

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
    "EL": "Greece",
    "FI": "Finland",
    "FR": "France",
    "DE": "Germany",
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

# Country groupings for regional analysis
MAJOR_EMITTERS = ["DE", "FR", "IT", "ES", "PL"]
EASTERN_EUROPE = ["BG", "HR", "CZ", "EE", "EL", "HU", "LV", "LT", "RO", "SK", "SI"]
WESTERN_EUROPE = [
    c for c in EU27_COUNTRIES if c not in MAJOR_EMITTERS and c not in EASTERN_EUROPE
]

# ISO3 to ISO2 mapping for external data sources
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

# PyPSA uses GR for Greece instead of EL
PYPSA_COUNTRY_MAPPING = {"GR": "EL"}

# =============================================================================
# Emission Sector Definitions
# =============================================================================

EMISSION_SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]

SECTOR_LABELS = {
    "HeatingCooling": "Heating & Cooling",
    "Industry": "Industry",
    "Land": "Land Use",
    "Mobility": "Mobility",
    "Other": "Other",
    "Power": "Power",
}

SECTOR_LABELS_MULTILINE = {
    "HeatingCooling": "Heating &\nCooling",
    "Industry": "Industry",
    "Land": "Land Use",
    "Mobility": "Mobility",
    "Other": "Other",
    "Power": "Power",
}

# =============================================================================
# ESR 2030 Targets (% reduction from 2005 baseline)
# =============================================================================

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
# Color Palettes (Paul Tol colorblind-safe)
# =============================================================================

# Colors for country/region stacked plots
COLORS_COUNTRY = {
    "DE": "#332288",  # Dark blue/indigo - Germany (largest)
    "FR": "#88CCEE",  # Light cyan - France
    "IT": "#44AA99",  # Teal - Italy
    "ES": "#117733",  # Dark green - Spain
    "PL": "#999933",  # Olive - Poland
    "East Europe": "#DDCC77",  # Sand/wheat - Other CEE
    "West Europe": "#CC6677",  # Dusty rose - Other Western
}

# Colors for sector stacked plots
COLORS_SECTOR = {
    "Power": "#CC6677",  # Dusty rose
    "Industry": "#332288",  # Dark indigo
    "Mobility": "#117733",  # Dark green
    "HeatingCooling": "#DDCC77",  # Sand/wheat
    "Land": "#88CCEE",  # Light cyan
    "Other": "#AA4499",  # Purple
}

# Alternative colorblind-friendly sector colors
COLORS_SECTOR_ALT = {
    "HeatingCooling": "#E69F00",  # Orange
    "Industry": "#56B4E9",  # Sky blue
    "Land": "#009E73",  # Bluish green
    "Mobility": "#F0E442",  # Yellow
    "Other": "#0072B2",  # Blue
    "Power": "#D55E00",  # Vermillion
}

# =============================================================================
# Nature Climate Change Publication Styling
# =============================================================================


def set_nature_style() -> None:
    """
    Configure matplotlib for Nature Climate Change publication standards.

    Sets font family, sizes, line widths, and other parameters to match
    Nature's style guidelines. TrueType fonts are used for PDF compatibility.
    """
    plt.rcParams.update(
        {
            # Font settings
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
            # Line widths
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Colors
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            # Output
            "savefig.dpi": 300,
            "pdf.fonttype": 42,  # TrueType fonts (Nature requirement)
            "ps.fonttype": 42,
        }
    )


def set_nature_style_minimal_spines() -> None:
    """
    Nature style with minimal spines (no left spine).

    Useful for scatter plots and bar charts where the y-axis line
    is not needed.
    """
    set_nature_style()
    plt.rcParams.update(
        {
            "axes.spines.left": False,
            "ytick.major.size": 0,
        }
    )


# =============================================================================
# Data Loading and Processing Utilities
# =============================================================================


def unnormalize_emissions(
    values: np.ndarray | pd.Series,
    sector: str,
    scaling_params: dict[str, dict[str, float]],
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Convert normalized emission values back to original scale.

    Args:
        values: Normalized values to convert.
        sector: Emission sector name (key in scaling_params).
        scaling_params: Dict with 'mean' and 'std' for each sector.
        clip_negative: If True, clip negative values to 0.

    Returns:
        Unnormalized values in original units.
    """
    mean = scaling_params[sector]["mean"]
    std = scaling_params[sector]["std"]
    result = values * std + mean

    if clip_negative:
        result = np.clip(result, 0, None)

    return result


def load_population_data(
    historical_path: str | Path = "data/full_timeseries/population.csv",
    projections_path: str | Path = "data/full_timeseries/projections/population.csv",
) -> pd.DataFrame:
    """
    Load and combine historical and projected population data.

    Args:
        historical_path: Path to historical population CSV.
        projections_path: Path to projected population CSV.

    Returns:
        DataFrame with columns ['geo', 'year', 'population'].
    """
    pop_hist = pd.read_csv(historical_path)
    pop_proj = pd.read_csv(projections_path)

    population_df = pd.concat([pop_hist, pop_proj], ignore_index=True)
    population_df["population"] = population_df["population:POP_NC"].astype(float)
    population_df = population_df[["geo", "year", "population"]]
    population_df = population_df.groupby(["geo", "year"], as_index=False)[
        "population"
    ].mean()

    return population_df


def load_mc_projections(
    mc_path: str | Path = "data/projections/mc_projections.csv",
) -> pd.DataFrame:
    """
    Load Monte Carlo projection results.

    Args:
        mc_path: Path to MC projections CSV.

    Returns:
        DataFrame with MC samples, including unnormalized emissions.
    """
    df = pd.read_csv(mc_path)
    df["geo"] = df["geo"].astype(str)
    return df


# =============================================================================
# Variable Name Mappings for Sensitivity Analysis
# =============================================================================

VARIABLE_NAME_MAP = {
    # Electricity production
    "Monthly_electricity_statistics:Net Electricity Production:Solar": "Solar power",
    "Monthly_electricity_statistics:Net Electricity Production:Total Combustible Fuels": "Fossil fuels power",
    "Monthly_electricity_statistics:Net Electricity Production:Wind": "Wind power",
    "Monthly_electricity_statistics:Used for pumped storage:Electricity": "Pumped storage",
    "Monthly_electricity_statistics:Distribution Losses:Electricity": "Electricity losses",
    "Monthly_electricity_statistics:Net Electricity Production:Hydro": "Hydro power",
    "Monthly_electricity_statistics:Net Electricity Production:Total Renewables (Hydro, Geo, Solar, Wind, Other)": "Renewables (total)",
    "Monthly_electricity_statistics:Total Exports:Electricity": "Electricity exports",
    "Monthly_electricity_statistics:Total Imports:Electricity": "Electricity imports",
    # Energy prices
    "Monthly_oil_price_statistics:Diesel (unit/litre):Total price:US dollars": "Diesel price",
    "Monthly_oil_price_statistics:Domestic heating oil (unit/litre):Total price:US dollars": "Heating oil price",
    "Monthly_oil_price_statistics:Gasoline (unit/litre):Total price:US dollars": "Gasoline price",
    # Context variables
    "CONTEXT::gdp_quarterly:MillionEUR": "GDP",
    "CONTEXT::climate:rainfall:POP": "Precipitation (pop.)",
    "CONTEXT::climate:temperature:POP": "Temperature (pop.)",
    "CONTEXT::climate:temperature:AREA": "Temperature (area)",
    "CONTEXT::climate:temperature_variability:POP": "Temp. variability (pop.)",
    "CONTEXT::climate:temperature_variability:AREA": "Temp. variability (area)",
    "CONTEXT::climate:rainfall:AREA": "Precipitation (area)",
    "CONTEXT::population:POP_NC": "Population",
    # Other inputs
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
    "EV_data:EV sales:Buses:BEV:Vehicles": "EV bus sales (BEV)",
    "EV_data:EV stock share:Cars:EV:percent": "EV car stock share",
    "EV_data:EV stock share:Buses:EV:percent": "EV bus stock share",
    "crops_livestock:Wheat:Production:t": "Wheat production",
    "crops_livestock:Meat,_Total:Production:t": "Meat production (total)",
    "crops_livestock:Meat,_Poultry:Production:t": "Poultry production",
    "crops_livestock:Fruit_Primary:Production:t": "Fruit production",
    "crops_livestock:Vegetables_Primary:Production:t": "Vegetable production",
    "trade:import_volume_index:Raw_materials:WORLD": "Import vol. (raw mat.)",
    "trade:import_volume_index:Food_drinks_tobacco:WORLD": "Import vol. (food)",
    "trade:import_volume_index:Machinery_transportequipment:WORLD": "Import vol. (machinery)",
    "trade:import_volume_index:Mineral_fuels_lubrificants:WORLD": "Import vol. (fuels)",
    "trade:import_volume_index:Chemicals:WORLD": "Import vol. (chemicals)",
    "trade:export_volume_index:Raw_materials:WORLD": "Export vol. (raw mat.)",
    "trade:export_volume_index:Mineral_fuels_lubrificants:WORLD": "Export vol. (fuels)",
    "trade:export_volume_index:Machinery_transportequipment:WORLD": "Export vol. (machinery)",
    "trade:export_volume_index:Chemicals:WORLD": "Export vol. (chemicals)",
    "train_performance:Passenger_trains:Total:Diesel:THS_train_mk": "Passenger trains (diesel)",
    "train_performance:Passenger_trains:Total:Electricity:THS_train_mk": "Passenger trains (elec.)",
    "train_performance:Goods_trains:Total:Diesel:THS_train_mk": "Goods trains (diesel)",
    "train_performance:Goods_trains:Total:Electricity:THS_train_mk": "Goods trains (elec.)",
    "solar_thermal_surface:THS_M2": "Solar thermal surface",
    "energy_consumption:FC_E:GJ_HAB": "Energy cons. (total)",
    "energy_consumption:FC_IND_E:GJ_HAB": "Energy cons. (industry)",
    "energy_consumption:FC_TRA_E:GJ_HAB": "Energy cons. (transport)",
}

# Variable categories for grouping in visualizations
VARIABLE_CATEGORIES = {
    "Temperature (pop.)": ("Climate", 0),
    "Temperature (area)": ("Climate", 1),
    "Precipitation (pop.)": ("Climate", 2),
    "Precipitation (area)": ("Climate", 3),
    "Temp. variability (pop.)": ("Climate", 4),
    "Temp. variability (area)": ("Climate", 5),
    "GDP": ("Economic", 10),
    "Population": ("Economic", 11),
    "ETS price": ("Economic", 12),
    "Energy taxes": ("Economic", 13),
    "Diesel price": ("Energy prices", 20),
    "Gasoline price": ("Energy prices", 21),
    "Heating oil price": ("Energy prices", 22),
    "Wind power": ("Electricity", 30),
    "Solar power": ("Electricity", 31),
    "Hydro power": ("Electricity", 32),
    "Fossil fuels power": ("Electricity", 33),
    "Renewables (total)": ("Electricity", 34),
    "Pumped storage": ("Electricity", 35),
    "Electricity exports": ("Electricity", 36),
    "Electricity imports": ("Electricity", 37),
    "Electricity losses": ("Electricity", 38),
    "Heat pump capacity": ("Heating", 40),
    "Solar thermal surface": ("Heating", 41),
    "Energy cons. (total)": ("Energy consumption", 50),
    "Energy cons. (industry)": ("Energy consumption", 51),
    "Energy cons. (transport)": ("Energy consumption", 52),
    "Modal split (road)": ("Transport", 60),
    "Modal split (rail)": ("Transport", 61),
    "Modal split (air)": ("Transport", 62),
    "Modal split (sea)": ("Transport", 63),
    "EV car sales (BEV)": ("Transport", 64),
    "EV car stock share": ("Transport", 65),
    "Cropland area": ("Land use", 70),
    "Forest area": ("Land use", 71),
    "Wheat production": ("Agriculture", 80),
    "Meat production (total)": ("Agriculture", 81),
}


def get_clean_variable_name(raw_name: str) -> str:
    """
    Convert raw variable name to clean display label.

    Args:
        raw_name: Original variable name from dataset.

    Returns:
        Clean display name, or original if no mapping exists.
    """
    return VARIABLE_NAME_MAP.get(raw_name, raw_name)
