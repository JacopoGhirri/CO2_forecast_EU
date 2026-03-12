"""
Interactive Streamlit dashboard for EU CO2 emission projections.

Displays historical emissions and Monte Carlo projections for EU27
countries, with comparison against OECD, EEA, and PyPSA scenarios.
The EU-27 Fit-for-55 target (55% reduction from 1990) is shown as a
reference when viewing the EU27 aggregate.

Adapted to the publishable repository structure:
    - Dataset pickle: data/pytorch_datasets/unified_dataset.pkl --CPU version--
    - MC projections:  data/projections/mc_projections.csv
    - Population:      data/full_timeseries/population.csv
                       data/full_timeseries/projections/population.csv
    - OECD:            data/external/oecd_projections.csv
    - EEA:             data/external/eea_projections.xlsx
    - PyPSA:           data/external/pypsa_projections.csv

The MC projections file uses a 2024 observed anchor: year 2024 rows
contain real emission data (identical across MC samples), while years
2025-2030 are model forecasts. The training dataset only covers up to
2023, so the 2024 baseline for visualisation is extracted from the MC
file rather than from the dataset pickle.

Usage:
    streamlit run scripts/visualization/interactive_plot.py
"""

import os
import pickle

import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# Data download
# =============================================================================

GDRIVE_FILES = {
    "data/pytorch_datasets/unified_dataset.pkl": "1MgQj0Vocz_eiPJGcYz9a0QtSGNOT2KyK",
    "data/projections/mc_projections.csv": "1Xc5B1ZVeURZty4Y1Pe5sNLq1hGM91sKV",
    "data/full_timeseries/population.csv": "17p2WHNCVnUNlp1C-8R5k-jfhZ5mNsFIW",
    "data/full_timeseries/projections/population.csv": "1swlcyxov_mcH1dH-woUXzfBpNQkTuriN",
    "data/external/oecd_projections.csv": "13m0bZLjSF85LXt8mwgnjVu1LK1JjdIsU",
    "data/external/eea_projections.xlsx": "17lqxJ1HKn7gX8kv8ndkl_O3iiOo8-noW",
    "data/external/pypsa_projections.csv": "1YTQwdaxZtUgKRY-59vpC4dmD9OmYLRa4",
}


@st.cache_resource
def download_data_if_needed():
    """Download data files from Google Drive if not already present."""
    downloaded = []
    errors = []

    for filepath, file_id in GDRIVE_FILES.items():
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not os.path.exists(filepath):
            try:
                result = gdown.download(
                    f"https://drive.google.com/uc?id={file_id}",
                    filepath,
                    quiet=True,
                )
                if result is None:
                    errors.append(filepath)
                else:
                    downloaded.append(filepath)
            except Exception as e:
                errors.append(f"{filepath}: {e}")

    return downloaded, errors


downloaded, errors = download_data_if_needed()

if errors:
    for err in errors:
        st.error(f"Failed to download: {err}")
    st.stop()
if downloaded:
    st.toast(f"Downloaded {len(downloaded)} data files", icon="\U0001f4e5")


# =============================================================================
# Configuration
# =============================================================================

DATASET_PATH = "data/pytorch_datasets/unified_dataset.pkl"
MC_PROJECTIONS_PATH = "data/projections/mc_projections.csv"
POPULATION_HIST_PATH = "data/full_timeseries/population.csv"
POPULATION_PROJ_PATH = "data/full_timeseries/projections/population.csv"
OECD_PATH = "data/external/oecd_projections.csv"
EEA_PATH = "data/external/eea_projections.xlsx"
PYPSA_PATH = "data/external/pypsa_projections.csv"

OUTPUT_SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]

BASELINE_YEAR = 2024

PYPSA_COUNTRY_MAPPING = {"GR": "EL"}

EU27 = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "EL", "FI", "FR",
    "DE", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT",
    "RO", "SK", "SI", "ES", "SE",
]

COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "HR": "Croatia",
    "CY": "Cyprus", "CZ": "Czechia", "DK": "Denmark", "EE": "Estonia",
    "FI": "Finland", "FR": "France", "DE": "Germany", "EL": "Greece",
    "HU": "Hungary", "IE": "Ireland", "IT": "Italy", "LV": "Latvia",
    "LT": "Lithuania", "LU": "Luxembourg", "MT": "Malta", "NL": "Netherlands",
    "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SK": "Slovakia",
    "SI": "Slovenia", "ES": "Spain", "SE": "Sweden", "EU27": "EU-27",
}

ISO3_TO_ISO2 = {
    "AUT": "AT", "BEL": "BE", "BGR": "BG", "HRV": "HR", "CYP": "CY",
    "CZE": "CZ", "DNK": "DK", "EST": "EE", "FIN": "FI", "FRA": "FR",
    "DEU": "DE", "HUN": "HU", "IRL": "IE", "ITA": "IT", "LVA": "LV",
    "LTU": "LT", "LUX": "LU", "MLT": "MT", "NLD": "NL", "POL": "PL",
    "PRT": "PT", "ROU": "RO", "SVK": "SK", "SVN": "SI", "ESP": "ES",
    "SWE": "SE", "GRC": "EL", "EU27": "EU27",
}

SCENARIOS = {
    "OECD_BAU": {
        "name": "OECD Business as Usual", "short": "OECD BAU",
        "marker": "o", "color": "#c0392b", "size": 120,
    },
    "OECD_ET": {
        "name": "OECD Energy Transition", "short": "OECD ET",
        "marker": "^", "color": "#27ae60", "size": 120,
    },
    "EEA_WEM": {
        "name": "EEA With Existing Measures", "short": "EEA WEM",
        "marker": "s", "color": "#8e44ad", "size": 100,
    },
    "EEA_WAM": {
        "name": "EEA With Additional Measures", "short": "EEA WAM",
        "marker": "D", "color": "#2980b9", "size": 100,
    },
    "PYPSA_BASE": {
        "name": "PyPSA Baseline", "short": "PyPSA Base",
        "marker": "p", "color": "#DDCC77", "size": 120,
    },
    "PYPSA_FF55": {
        "name": "PyPSA Fit for 55", "short": "PyPSA FF55",
        "marker": "h", "color": "#88CCEE", "size": 120,
    },
    "TARGET": {
        "name": "FF55 Target (EU-27)", "short": "Target",
        "marker": "*", "color": "#2c3e50", "size": 200,
    },
}

COLORS = {
    "historical": "#1a1a1a",
    "mc_mean": "#e67e22",
    "mc_ci": "#f39c12",
    "grid": "#ecf0f1",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# =============================================================================
# Data loading
# =============================================================================


@st.cache_resource
def load_pickle(path):
    """Load a pickle that may reference project modules not on sys.path.

    The dataset pickle was saved with class references like
    ``scripts.elements.datasets.DatasetUnified``. On Streamlit Cloud the
    repo root is not on sys.path, so unpickling fails with
    ModuleNotFoundError. We add the repo root temporarily so that
    ``import scripts.elements.datasets`` resolves correctly.
    """
    import sys
    from pathlib import Path

    repo_root = str(Path(__file__).resolve().parents[3])
    added = repo_root not in sys.path
    if added:
        sys.path.insert(0, repo_root)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        if added:
            sys.path.remove(repo_root)


@st.cache_data
def load_and_process_data():
    """Load all data sources and prepare historical + forecast DataFrames.

    Historical emissions come from the training dataset (up to 2023).
    The 2024 observed anchor is extracted from the MC projections file
    and appended, since the training dataset does not contain 2024.
    Forecast data covers 2025-2030 (MC mean and quantiles).
    """
    full_dataset = load_pickle(DATASET_PATH)

    keys = full_dataset.keys
    emi_dataset = pd.DataFrame(
        full_dataset.emi_df.cpu().numpy(), columns=OUTPUT_SECTORS
    )
    historical = pd.concat([keys, emi_dataset], axis=1)
    for s in OUTPUT_SECTORS:
        mean_ = full_dataset.precomputed_scaling_params[s]["mean"]
        std_ = full_dataset.precomputed_scaling_params[s]["std"]
        historical[s] = historical[s] * std_ + mean_

    pop_hist = pd.read_csv(POPULATION_HIST_PATH)
    pop_proj = pd.read_csv(POPULATION_PROJ_PATH)
    population_df = pd.concat([pop_hist, pop_proj], ignore_index=True)
    population_df["population"] = population_df["population:POP_NC"].astype(float)
    population_df = population_df[["geo", "year", "population"]]
    population_df = (
        population_df.groupby(["geo", "year"], as_index=False)["population"].mean()
    )

    historical = historical.merge(population_df, on=["geo", "year"], how="left")
    historical["total_CO2"] = (
        historical[OUTPUT_SECTORS].sum(axis=1) * historical["population"]
    )

    df_mc = pd.read_csv(MC_PROJECTIONS_PATH)
    df_mc["geo"] = df_mc["geo"].astype(str)

    for s in OUTPUT_SECTORS:
        m = full_dataset.precomputed_scaling_params[s]["mean"]
        sd = full_dataset.precomputed_scaling_params[s]["std"]
        df_mc[f"{s}_unnorm"] = np.clip(df_mc[f"emissions_{s}"] * sd + m, 0, None)

    df_mc = df_mc.merge(population_df, on=["geo", "year"], how="left")
    df_mc["total_CO2"] = (
        df_mc[[f"{s}_unnorm" for s in OUTPUT_SECTORS]].sum(axis=1) * df_mc["population"]
    )

    # Append 2024 anchor to historical
    mc_2024 = df_mc[df_mc["year"] == BASELINE_YEAR].copy()
    anchor_2024 = mc_2024.groupby("geo", as_index=False).agg(
        {"population": "first", "total_CO2": "mean",
         **{f"{s}_unnorm": "mean" for s in OUTPUT_SECTORS}}
    )
    anchor_2024["year"] = BASELINE_YEAR
    for s in OUTPUT_SECTORS:
        anchor_2024[s] = anchor_2024[f"{s}_unnorm"]
    anchor_2024 = anchor_2024[
        ["geo", "year", "population", "total_CO2"] + OUTPUT_SECTORS
    ]
    historical = pd.concat([historical, anchor_2024], ignore_index=True)

    # Forecast: 2025-2030
    df_mc_forecast = df_mc[df_mc["year"] > BASELINE_YEAR].copy()
    forecast_summary = (
        df_mc_forecast.groupby(["geo", "year"])
        .agg({
            "total_CO2": [
                "mean",
                lambda x: np.quantile(x, 0.05),
                lambda x: np.quantile(x, 0.95),
            ],
            "population": "mean",
            **{f"{s}_unnorm": [
                "mean",
                lambda x: np.quantile(x, 0.05),
                lambda x: np.quantile(x, 0.95),
            ] for s in OUTPUT_SECTORS},
        })
        .reset_index()
    )
    forecast_summary.columns = (
        ["geo", "year", "total_CO2_mean", "total_CO2_low", "total_CO2_high",
         "population"]
        + [f"{s}_{stat}" for s in OUTPUT_SECTORS for stat in ["mean", "low", "high"]]
    )

    # OECD
    oecd_df = pd.DataFrame()
    if os.path.exists(OECD_PATH):
        oecd_df = pd.read_csv(OECD_PATH)
        oecd_df = oecd_df.rename(columns={
            "REF_AREA": "iso3", "SCENARIO": "scenario",
            "TIME_PERIOD": "year", "OBS_VALUE": "value",
        })
        oecd_df["value"] = pd.to_numeric(oecd_df["value"], errors="coerce")
        oecd_df = oecd_df.dropna(subset=["value"])
        oecd_df["geo"] = oecd_df["iso3"].map(ISO3_TO_ISO2)
        oecd_df = oecd_df[oecd_df["geo"].notna()]

    # EEA
    eea_df = None
    if os.path.exists(EEA_PATH):
        eea_raw = pd.read_excel(EEA_PATH, sheet_name="Database")
        eea_df = eea_raw[
            (eea_raw["Year"] == 2030)
            & (eea_raw["Category"] == "Total excluding LULUCF")
            & (eea_raw["Gas"] == "ESR emissions (ktCO2e)")
        ].copy()
        eea_df = eea_df.rename(columns={"CountryCode": "geo", "Scenario": "scenario"})
        eea_df["value_Mt"] = eea_df["Gapfilled"] / 1000
        eea_df = eea_df[["geo", "scenario", "value_Mt"]]

    # PyPSA
    pypsa_df = None
    if os.path.exists(PYPSA_PATH):
        pypsa_raw = pd.read_csv(PYPSA_PATH)
        pypsa_raw["country"] = pypsa_raw["country"].replace(PYPSA_COUNTRY_MAPPING)
        pypsa_df = (
            pypsa_raw.groupby(["country", "scenario"], as_index=False)["value"].sum()
        )
        pypsa_df["value_Mt"] = pypsa_df["value"] / 1e6
        pypsa_df = pypsa_df.rename(columns={"country": "geo"})
        pypsa_df = pypsa_df[["geo", "scenario", "value_Mt"]]

    # EU-27 FF55 target: 45% of 1990 emissions
    eu27_ff55_mt = np.nan
    if not oecd_df.empty:
        eu27_1990 = oecd_df[
            (oecd_df["year"] == 1990) & (oecd_df["geo"] == "EU27")
        ]
        if not eu27_1990.empty:
            eu27_ff55_mt = eu27_1990["value"].mean() * 0.45

    return (historical, forecast_summary, oecd_df, eea_df, pypsa_df,
            population_df, eu27_ff55_mt)


@st.cache_data
def compute_eu_aggregates(historical, forecast_summary, population_df):
    """Aggregate country-level data to EU27 totals."""
    eu_hist = historical[historical["geo"].isin(EU27)].copy()
    eu_fcast = forecast_summary[forecast_summary["geo"].isin(EU27)].copy()

    def hist_year_agg(g):
        total_pop = g["population"].sum()
        total_co2 = g["total_CO2"].sum()
        sector_pc = {
            s: (g[s] * g["population"]).sum() / total_pop if total_pop > 0 else 0.0
            for s in OUTPUT_SECTORS
        }
        row = {"population": total_pop, "total_CO2": total_co2}
        row.update(sector_pc)
        return pd.Series(row)

    hist_agg = (
        eu_hist.groupby("year", as_index=False)
        .apply(hist_year_agg)
        .reset_index(drop=True)
    )

    def fcast_year_agg(g):
        total_pop = g["population"].sum()
        row = {
            "population": total_pop,
            "total_CO2_mean": g["total_CO2_mean"].sum(),
            "total_CO2_low": g["total_CO2_low"].sum(),
            "total_CO2_high": g["total_CO2_high"].sum(),
        }
        for s in OUTPUT_SECTORS:
            for stat in ["mean", "low", "high"]:
                col = f"{s}_{stat}"
                if col in g.columns:
                    row[col] = (g[col] * g["population"]).sum() / total_pop
                else:
                    row[col] = 0.0
        return pd.Series(row)

    fcast_agg = (
        eu_fcast.groupby("year", as_index=False)
        .apply(fcast_year_agg)
        .reset_index(drop=True)
    )

    return hist_agg, fcast_agg


# =============================================================================
# Streamlit app
# =============================================================================

st.set_page_config(layout="wide", page_title="EU CO\u2082 Emissions Explorer")
st.title("\U0001f30d EU CO\u2082 Emissions Projections")
st.markdown(
    "*Comparing model projections with OECD, EEA, and PyPSA scenarios "
    "against the EU-27 Fit-for-55 target*"
)

with st.spinner("Loading data..."):
    (historical, forecast_summary, oecd_df, eea_df, pypsa_df,
     population_df, eu27_ff55_mt) = load_and_process_data()

st.sidebar.header("\U0001f4ca Plot Settings")

country_options_unsorted = [c for c in historical["geo"].unique() if c in EU27]
country_options_sorted = sorted(
    country_options_unsorted, key=lambda x: COUNTRY_NAMES.get(x, x)
)
country_options = ["EU27"] + country_options_sorted
country_display = {c: COUNTRY_NAMES.get(c, c) for c in country_options}

selected_country = st.sidebar.selectbox(
    "Country/Region:", country_options,
    format_func=lambda x: country_display[x],
)
metric_type = st.sidebar.radio("Metric:", ["Total Emissions", "Per Capita Emissions"])
sector_option = st.sidebar.radio(
    "Sectors:", ["All sectors combined", "Individual sector"]
)
selected_sector = None
if sector_option == "Individual sector":
    selected_sector = st.sidebar.selectbox("Select sector:", OUTPUT_SECTORS)

st.sidebar.markdown("---")
st.sidebar.subheader("Display Options")
show_ci = st.sidebar.checkbox(
    "Show latent space uncertainty (90% Monte Carlo confidence intervals)",
    value=True,
)
show_oecd = st.sidebar.checkbox("Show OECD scenarios", value=True)
show_eea = st.sidebar.checkbox("Show EEA scenarios", value=(eea_df is not None))
show_pypsa = st.sidebar.checkbox("Show PyPSA scenarios", value=(pypsa_df is not None))
show_target = st.sidebar.checkbox(
    "Show FF55 target (EU-27 only)", value=(selected_country == "EU27")
)

if selected_country in ["CY", "MT"] and show_oecd:
    st.sidebar.warning(
        f"OECD data not available for {COUNTRY_NAMES[selected_country]}"
    )

# Prepare country data
if selected_country == "EU27":
    hist_data, fcast_data = compute_eu_aggregates(
        historical, forecast_summary, population_df
    )
    eu_pop = (
        population_df[population_df["geo"].isin(EU27)]
        .groupby("year")["population"].sum().reset_index()
    )
    hist_data["geo"] = "EU27"
    fcast_data["geo"] = "EU27"
else:
    eu_pop = None
    hist_data = (
        historical[historical["geo"] == selected_country].sort_values("year").copy()
    )
    fcast_data = forecast_summary[
        forecast_summary["geo"] == selected_country
    ].copy()

# Compute emissions metric
if sector_option == "All sectors combined":
    if metric_type == "Per Capita Emissions":
        hist_data["emissions"] = hist_data["total_CO2"] / hist_data["population"]
        fcast_data["emissions_mean"] = (
            fcast_data["total_CO2_mean"] / fcast_data["population"]
        )
        if show_ci:
            fcast_data["emissions_low"] = (
                fcast_data["total_CO2_low"] / fcast_data["population"]
            )
            fcast_data["emissions_high"] = (
                fcast_data["total_CO2_high"] / fcast_data["population"]
            )
    else:
        hist_data["emissions"] = hist_data["total_CO2"]
        fcast_data["emissions_mean"] = fcast_data["total_CO2_mean"]
        if show_ci:
            fcast_data["emissions_low"] = fcast_data["total_CO2_low"]
            fcast_data["emissions_high"] = fcast_data["total_CO2_high"]
else:
    hist_data["emissions"] = hist_data[selected_sector] * hist_data["population"]
    fcast_data["emissions_mean"] = (
        fcast_data[f"{selected_sector}_mean"] * fcast_data["population"]
    )
    if metric_type == "Per Capita Emissions":
        hist_data["emissions"] = hist_data["emissions"] / hist_data["population"]
        fcast_data["emissions_mean"] = (
            fcast_data["emissions_mean"] / fcast_data["population"]
        )
        if show_ci:
            fcast_data["emissions_low"] = fcast_data[f"{selected_sector}_low"]
            fcast_data["emissions_high"] = fcast_data[f"{selected_sector}_high"]
    else:
        if show_ci:
            fcast_data["emissions_low"] = (
                fcast_data[f"{selected_sector}_low"] * fcast_data["population"]
            )
            fcast_data["emissions_high"] = (
                fcast_data[f"{selected_sector}_high"] * fcast_data["population"]
            )

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

if metric_type == "Total Emissions":
    scale = 1e6
    unit = "MtCO\u2082"
else:
    scale = 1000
    unit = "tCO\u2082/capita"

if not hist_data.empty and "emissions" in hist_data.columns:
    ax.plot(
        hist_data["year"], hist_data["emissions"] / scale,
        color=COLORS["historical"], linewidth=2.5, label="Historical", zorder=4,
    )

if not fcast_data.empty and "emissions_mean" in fcast_data.columns:
    ax.plot(
        fcast_data["year"], fcast_data["emissions_mean"] / scale,
        color=COLORS["mc_mean"], linewidth=2.5, label="This study (mean)", zorder=4,
    )
    if show_ci and "emissions_low" in fcast_data.columns:
        ax.fill_between(
            fcast_data["year"],
            fcast_data["emissions_low"] / scale,
            fcast_data["emissions_high"] / scale,
            color=COLORS["mc_ci"], alpha=0.3, label="MC CI", zorder=2,
        )

pop_2030 = None
if metric_type == "Per Capita Emissions":
    if selected_country == "EU27" and eu_pop is not None:
        p = eu_pop[eu_pop["year"] == 2030]
        pop_2030 = p["population"].values[0] if len(p) > 0 else None
    else:
        p = population_df[
            (population_df["geo"] == selected_country)
            & (population_df["year"] == 2030)
        ]
        pop_2030 = p["population"].iloc[0] if not p.empty else None

# OECD
if (show_oecd and sector_option == "All sectors combined"
        and selected_country not in ["CY", "MT"] and not oecd_df.empty):
    oecd_2030 = oecd_df[
        (oecd_df["geo"] == selected_country) & (oecd_df["year"] == 2030)
    ]
    for key, scen in [("OECD_BAU", "BAU1"), ("OECD_ET", "ET1")]:
        row = oecd_2030[oecd_2030["scenario"] == scen]
        if not row.empty:
            val_mt = row["value"].values[0]
            val = (val_mt * 1e6) / (pop_2030 * 1000) if (metric_type == "Per Capita Emissions" and pop_2030) else val_mt
            cfg = SCENARIOS[key]
            ax.scatter(
                2030, val, color=cfg["color"], marker=cfg["marker"],
                s=cfg["size"], edgecolors="white", linewidths=1.5,
                label=cfg["name"], zorder=5,
            )

# EEA
if show_eea and eea_df is not None and sector_option == "All sectors combined":
    eea_c = eea_df[eea_df["geo"] == selected_country]
    for key, scen in [("EEA_WEM", "WEM"), ("EEA_WAM", "WAM")]:
        row = eea_c[eea_c["scenario"] == scen]
        if not row.empty:
            val_mt = row["value_Mt"].values[0]
            val = (val_mt * 1e6) / (pop_2030 * 1000) if (metric_type == "Per Capita Emissions" and pop_2030) else val_mt
            cfg = SCENARIOS[key]
            ax.scatter(
                2030, val, color=cfg["color"], marker=cfg["marker"],
                s=cfg["size"], edgecolors="white", linewidths=1.5,
                label=cfg["name"], zorder=5,
            )

# PyPSA
if show_pypsa and pypsa_df is not None and sector_option == "All sectors combined":
    pypsa_c = pypsa_df[pypsa_df["geo"] == selected_country]
    for key, scen_names in [
        ("PYPSA_BASE", ["base"]),
        ("PYPSA_FF55", ["policy", "FF55", "ff55"]),
    ]:
        matched = pd.DataFrame()
        for sn in scen_names:
            matched = pypsa_c[pypsa_c["scenario"] == sn]
            if not matched.empty:
                break
        if not matched.empty:
            val_mt = matched["value_Mt"].values[0]
            val = (val_mt * 1e6) / (pop_2030 * 1000) if (metric_type == "Per Capita Emissions" and pop_2030) else val_mt
            cfg = SCENARIOS[key]
            ax.scatter(
                2030, val, color=cfg["color"], marker=cfg["marker"],
                s=cfg["size"], edgecolors="white", linewidths=1.5,
                label=cfg["name"], zorder=5,
            )

# FF55 target — EU-27 only
if (show_target and selected_country == "EU27"
        and sector_option == "All sectors combined"
        and not np.isnan(eu27_ff55_mt)):
    target_val = (eu27_ff55_mt * 1e6) / (pop_2030 * 1000) if (metric_type == "Per Capita Emissions" and pop_2030) else eu27_ff55_mt
    cfg = SCENARIOS["TARGET"]
    ax.scatter(
        2030, target_val, color=cfg["color"], marker=cfg["marker"],
        s=cfg["size"], edgecolors="white", linewidths=1.5,
        label="FF55 target (55% from 1990)", zorder=6,
    )

country_name = COUNTRY_NAMES.get(selected_country, selected_country)
sector_label = selected_sector if selected_sector else "All sectors"
ax.set_title(
    f"{country_name} \u2013 {metric_type} ({sector_label})",
    fontsize=14, fontweight="bold",
)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel(unit, fontsize=11)
ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, color=COLORS["grid"])
ax.set_axisbelow(True)
ax.legend(loc="upper right", framealpha=0.95, edgecolor="#bdc3c7",
          fontsize=8, frameon=True)
ax.set_xticks(range(2010, 2031, 2))
ax.set_xlim(2009, 2031)
plt.tight_layout()

st.pyplot(fig)

# Info panel
col1, col2, col3 = st.columns(3)

with col1:
    plt.savefig("temp_plot.png", dpi=300, bbox_inches="tight", facecolor="white")
    with open("temp_plot.png", "rb") as f:
        st.download_button(
            label="\U0001f4e5 Download Plot (PNG)", data=f,
            file_name=f"{selected_country}_{metric_type.replace(' ', '_')}.png",
            mime="image/png",
        )

with col2:
    if selected_country == "EU27" and not np.isnan(eu27_ff55_mt):
        st.metric(
            "2030 FF55 Target",
            f"{eu27_ff55_mt:.0f} Mt",
            help="55% reduction from 1990 CO\u2082 levels (EU-27 aggregate)",
        )

with col3:
    has_oecd = selected_country not in ["CY", "MT"] and not oecd_df.empty
    has_eea = eea_df is not None and selected_country in eea_df["geo"].values
    has_pypsa = pypsa_df is not None and selected_country in pypsa_df["geo"].values
    st.markdown("**Data availability:**")
    st.markdown(f"- OECD: {'yes' if has_oecd else 'no'}")
    st.markdown(f"- EEA: {'yes' if has_eea else 'no'}")
    st.markdown(f"- PyPSA: {'yes' if has_pypsa else 'no'}")

with st.expander("\U0001f4cb View underlying data"):
    tab1, tab2 = st.tabs(["Historical", "Forecast"])

    with tab1:
        if "emissions" in hist_data.columns:
            display_hist = hist_data[["year", "emissions"]].copy()
            display_hist["emissions"] = display_hist["emissions"] / scale
            display_hist = display_hist.rename(
                columns={"emissions": f"Emissions ({unit})"}
            )
            st.dataframe(display_hist.round(2))

    with tab2:
        if "emissions_mean" in fcast_data.columns:
            cols = ["year", "emissions_mean"]
            if "emissions_low" in fcast_data.columns:
                cols.extend(["emissions_low", "emissions_high"])
            display_fcast = fcast_data[cols].copy()
            for c in cols[1:]:
                display_fcast[c] = display_fcast[c] / scale
            display_fcast = display_fcast.rename(columns={
                "emissions_mean": f"Mean ({unit})",
                "emissions_low": f"MC 5th percentile ({unit})",
                "emissions_high": f"MC 95th percentile ({unit})",
            })
            st.dataframe(display_fcast.round(2))