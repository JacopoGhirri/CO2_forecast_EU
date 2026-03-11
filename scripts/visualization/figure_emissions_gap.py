"""
Figure 1: Projected Emission Change from 2023 Baseline (paper-ready).

Three-panel figure:
  (a) EU27 aggregate — % change by model/source (bar chart)
  (b) EU27 aggregate — Mt CO2 change by model/source (bar chart)
  (c) Country-level — Mt CO2 change (symlog y-axis, countries on x-axis)

FF55 target shown as horizontal dashed line in (a) and (b).
Single shared legend on the right side of the figure.

UNIT CONVENTIONS:
  - Model sectors after denormalization: kg CO2 per inhabitant
  - Population (Eurostat POP_NC): thousands of persons
  - total_CO2 = sum(kg/hab) * population(thousands) = tonnes
  - total_CO2 / 1e6 = Mt CO2
  - OECD 'value': Mt CO2
  - EEA 'Gapfilled': kt CO2e -> /1000 = Mt
  - PyPSA 'value': tonnes -> /1e6 = Mt

Usage:
    python -m scripts.visualization.figure_emissions_change_from_2023

Outputs:
    - outputs/figures/fig1_emissions_change.pdf
    - outputs/figures/fig1_emissions_change.png
    - outputs/figures/fig1_emissions_change.svg
    - outputs/tables/fig1_change_from_2023_summary.csv
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

OECD_PATH = Path("data/external/oecd_projections.csv")
EEA_PATH = Path("data/external/eea_projections.xlsx")
PYPSA_PATH = Path("data/external/pypsa_projections.csv")

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

PYPSA_COUNTRY_MAPPING = {"GR": "EL"}

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

# =============================================================================
# Source styling — order matters for the bar charts
# =============================================================================

SOURCES = {
    "mc": {"label": "This study", "color": "#332288", "marker": "o"},
    "oecd_bau1": {"label": "OECD BAU", "color": "#AA4499", "marker": "D"},
    "oecd_et1": {"label": "OECD Energy Transition", "color": "#CC6677", "marker": "D"},
    "eea_wem": {"label": "EEA With Exist. Measures", "color": "#44AA99", "marker": "^"},
    "eea_wam": {"label": "EEA With Add. Measures", "color": "#117733", "marker": "^"},
    "pypsa_base": {"label": "PyPSA Baseline", "color": "#DDCC77", "marker": "s"},
    "pypsa_ff55": {"label": "PyPSA Fit for 55", "color": "#88CCEE", "marker": "s"},
}

# Additional jitter offsets for panel (c) scatter
JITTER_Y = {
    "eea_wam": 0.30,
    "eea_wem": 0.20,
    "pypsa_ff55": 0.10,
    "mc": 0.0,
    "pypsa_base": -0.10,
    "oecd_et1": -0.20,
    "oecd_bau1": -0.30,
}


def setup_nature_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
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


# =============================================================================
# Data loading — all outputs in Mt CO2
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


def get_2030_mc_mt(dataset, population_df):
    """MC 2030 mean total CO2 per country in Mt."""
    df_mc = pd.read_csv(MC_PROJECTIONS_PATH)
    df_mc["geo"] = df_mc["geo"].astype(str)
    for s in OUTPUT_SECTORS:
        m = dataset.precomputed_scaling_params[s]["mean"]
        sd = dataset.precomputed_scaling_params[s]["std"]
        df_mc[f"{s}_unnorm"] = np.clip(df_mc[f"emissions_{s}"] * sd + m, 0, None)
    df_mc = df_mc.merge(population_df, on=["geo", "year"], how="left")
    df_mc["total_CO2"] = (
        df_mc[[f"{s}_unnorm" for s in OUTPUT_SECTORS]].sum(axis=1) * df_mc["population"]
    )

    mc_2030 = df_mc[df_mc["year"] == 2030].groupby("geo")["total_CO2"].mean() / 1e6
    eu = df_mc[(df_mc["year"] == 2030) & (df_mc["geo"].isin(EU27_COUNTRIES))]
    eu27_mt = eu.groupby("mc_sample")["total_CO2"].sum().mean() / 1e6

    result = mc_2030.to_dict()
    result["EU27"] = eu27_mt
    return result


def get_2023_baseline_mt(dataset, population_df):
    """2023 historical total CO2 per country in Mt."""
    keys = dataset.keys
    emi = pd.DataFrame(dataset.emi_df.cpu().numpy(), columns=OUTPUT_SECTORS)
    hist = pd.concat([keys, emi], axis=1)
    for s in OUTPUT_SECTORS:
        m = dataset.precomputed_scaling_params[s]["mean"]
        sd = dataset.precomputed_scaling_params[s]["std"]
        hist[s] = hist[s] * sd + m
    hist = hist.merge(population_df, on=["geo", "year"], how="left")
    hist["total_CO2"] = hist[OUTPUT_SECTORS].sum(axis=1) * hist["population"]

    h23 = hist[hist["year"] == 2023]
    baseline = {row["geo"]: row["total_CO2"] / 1e6 for _, row in h23.iterrows()}
    baseline["EU27"] = sum(v for k, v in baseline.items() if k in EU27_COUNTRIES)
    return baseline


def load_oecd_2030_mt():
    if not OECD_PATH.exists():
        return {}
    df = pd.read_csv(OECD_PATH)
    df = df.rename(
        columns={
            "REF_AREA": "iso3",
            "SCENARIO": "scenario",
            "TIME_PERIOD": "year",
            "OBS_VALUE": "value",
        }
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["geo"] = df["iso3"].map(ISO3_TO_ISO2)
    df = df[df["geo"].notna() & (df["year"] == 2030)]
    return {(r["geo"], r["scenario"]): r["value"] for _, r in df.iterrows()}


def load_oecd_full():
    if not OECD_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(OECD_PATH)
    df = df.rename(
        columns={
            "REF_AREA": "iso3",
            "SCENARIO": "scenario",
            "TIME_PERIOD": "year",
            "OBS_VALUE": "value",
        }
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["geo"] = df["iso3"].map(ISO3_TO_ISO2)
    return df[df["geo"].notna()]


def load_eea_2030_mt():
    if not EEA_PATH.exists():
        return {}
    eea = pd.read_excel(EEA_PATH, sheet_name="Database")
    eea = eea[
        (eea["Year"] == 2030)
        & (eea["Category"] == "Total excluding LULUCF")
        & (eea["Gas"] == "ESR emissions (ktCO2e)")
    ].copy()
    return {
        (r["CountryCode"], r["Scenario"]): r["Gapfilled"] / 1000
        for _, r in eea.iterrows()
    }


def load_pypsa_2030_mt():
    if not PYPSA_PATH.exists():
        return {}
    df = pd.read_csv(PYPSA_PATH)
    df["country"] = df["country"].replace(PYPSA_COUNTRY_MAPPING)
    totals = df.groupby(["country", "scenario"], as_index=False)["value"].sum()
    return {
        (r["country"], r["scenario"]): r["value"] / 1e6 for _, r in totals.iterrows()
    }


# =============================================================================
# Build comparison table
# =============================================================================


def build_comparison(baseline_mt, mc_2030_mt, oecd_2030, eea_2030, pypsa_2030):
    """Build DataFrame with _pct and _mt columns for every source and geo."""
    rows = []
    for geo in EU27_COUNTRIES + ["EU27"]:
        base = baseline_mt.get(geo)
        if not base or base <= 0:
            continue
        d = {"geo": geo, "baseline_mt": base}

        # This study
        mc = mc_2030_mt.get(geo)
        if mc is not None:
            d["mc_pct"] = ((mc - base) / base) * 100
            d["mc_mt"] = mc - base
        else:
            d["mc_pct"] = d["mc_mt"] = np.nan

        # OECD
        for scen, key in [("BAU1", "oecd_bau1"), ("ET1", "oecd_et1")]:
            v = oecd_2030.get((geo, scen))
            if v is not None:
                d[f"{key}_pct"] = ((v - base) / base) * 100
                d[f"{key}_mt"] = v - base
            else:
                d[f"{key}_pct"] = d[f"{key}_mt"] = np.nan

        # EEA
        for scen, key in [("WAM", "eea_wam"), ("WEM", "eea_wem")]:
            v = eea_2030.get((geo, scen))
            if v is not None:
                d[f"{key}_pct"] = ((v - base) / base) * 100
                d[f"{key}_mt"] = v - base
            else:
                d[f"{key}_pct"] = d[f"{key}_mt"] = np.nan

        # PyPSA
        v = pypsa_2030.get((geo, "base"))
        if v is not None:
            d["pypsa_base_pct"] = ((v - base) / base) * 100
            d["pypsa_base_mt"] = v - base
        else:
            d["pypsa_base_pct"] = d["pypsa_base_mt"] = np.nan

        pv = None
        for sn in ["policy", "FF55", "ff55"]:
            pv = pypsa_2030.get((geo, sn))
            if pv is not None:
                break
        if pv is not None:
            d["pypsa_ff55_pct"] = ((pv - base) / base) * 100
            d["pypsa_ff55_mt"] = pv - base
        else:
            d["pypsa_ff55_pct"] = d["pypsa_ff55_mt"] = np.nan

        rows.append(d)
    return pd.DataFrame(rows)


# =============================================================================
# Figure
# =============================================================================


def create_figure(df, ff55_pct, ff55_mt):
    setup_nature_style()

    eu = df[df["geo"] == "EU27"].iloc[0]
    countries = df[df["geo"] != "EU27"].copy()
    countries = countries.sort_values("mc_mt", ascending=False).reset_index(drop=True)

    # Source keys in display order
    src_keys = list(SOURCES.keys())

    # --- Layout: 2 columns, top row 2 panels, bottom row spans both ---
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(
        2, 2, figure=fig, height_ratios=[1, 2.2], hspace=0.35, wspace=0.35
    )

    ax_a = fig.add_subplot(gs[0, 0])  # EU27 %
    ax_b = fig.add_subplot(gs[0, 1])  # EU27 Mt
    ax_c = fig.add_subplot(gs[1, :])  # Country-level Mt

    # =====================================================================
    # Panel (a): EU27 % change — grouped bars
    # =====================================================================
    vals_pct = []
    colors_pct = []
    labels_pct = []
    for key in src_keys:
        v = eu.get(f"{key}_pct", np.nan)
        vals_pct.append(v if pd.notna(v) else 0)
        colors_pct.append(SOURCES[key]["color"])
        labels_pct.append(SOURCES[key]["label"])

    x_a = np.arange(len(src_keys))
    bars_a = ax_a.bar(
        x_a,
        vals_pct,
        color=colors_pct,
        width=0.65,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    # Grey out missing bars
    for i, key in enumerate(src_keys):
        if pd.isna(eu.get(f"{key}_pct", np.nan)):
            bars_a[i].set_alpha(0.15)

    # FF55 target line
    ax_a.axhline(ff55_pct, color="black", linestyle="--", linewidth=1.0, zorder=4)
    ax_a.text(
        len(src_keys) - 0.5,
        ff55_pct + 1.5,
        "FF55 target",
        fontsize=6,
        ha="right",
        va="bottom",
        color="black",
        fontstyle="italic",
    )

    ax_a.axhline(0, color="#2c3e50", linewidth=0.8, zorder=2)
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(
        [SOURCES[k]["label"] for k in src_keys], rotation=40, ha="right", fontsize=5.5
    )
    ax_a.set_ylabel("Change from 2023 (%)")
    ax_a.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.3)
    ax_a.set_axisbelow(True)
    ax_a.set_title("EU-27", fontsize=9, fontweight="bold", pad=8)
    ax_a.text(
        -0.15,
        1.08,
        "a",
        transform=ax_a.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # =====================================================================
    # Panel (b): EU27 Mt change — grouped bars
    # =====================================================================
    vals_mt = []
    colors_mt = []
    for key in src_keys:
        v = eu.get(f"{key}_mt", np.nan)
        vals_mt.append(v if pd.notna(v) else 0)
        colors_mt.append(SOURCES[key]["color"])

    x_b = np.arange(len(src_keys))
    bars_b = ax_b.bar(
        x_b,
        vals_mt,
        color=colors_mt,
        width=0.65,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    for i, key in enumerate(src_keys):
        if pd.isna(eu.get(f"{key}_mt", np.nan)):
            bars_b[i].set_alpha(0.15)

    ax_b.axhline(ff55_mt, color="black", linestyle="--", linewidth=1.0, zorder=4)
    ax_b.text(
        len(src_keys) - 0.5,
        ff55_mt + 15,
        "FF55 target",
        fontsize=6,
        ha="right",
        va="bottom",
        color="black",
        fontstyle="italic",
    )

    ax_b.axhline(0, color="#2c3e50", linewidth=0.8, zorder=2)
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(
        [SOURCES[k]["label"] for k in src_keys], rotation=40, ha="right", fontsize=5.5
    )
    ax_b.set_ylabel("Change from 2023 (Mt CO2)")
    ax_b.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.3)
    ax_b.set_axisbelow(True)
    ax_b.set_title("EU-27", fontsize=9, fontweight="bold", pad=8)
    ax_b.text(
        -0.15,
        1.08,
        "b",
        transform=ax_b.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # =====================================================================
    # Panel (c): Country-level Mt CO2 — symlog y, countries on x
    # =====================================================================
    n_c = len(countries)
    x_c = np.arange(n_c)

    # Symlog scale
    ax_c.set_yscale("symlog", linthresh=5)

    # Background bands
    y_all = []
    for key in src_keys:
        col = f"{key}_mt"
        if col in countries.columns:
            y_all.extend(countries[col].dropna().tolist())
    if y_all:
        y_lo = min(y_all) * 1.2
        y_hi = max(max(y_all) * 1.2, 5)
    else:
        y_lo, y_hi = -200, 50

    ax_c.axhspan(y_lo * 2, 0, color="#eafaf1", alpha=0.6, zorder=0)
    ax_c.axhspan(0, y_hi * 2, color="#fdedec", alpha=0.6, zorder=0)

    # Zero line
    ax_c.axhline(0, color="#2c3e50", linewidth=1.0, zorder=2)

    # Stems for MC
    for i, (_, row) in enumerate(countries.iterrows()):
        mc_v = row.get("mc_mt", np.nan)
        if pd.notna(mc_v):
            ax_c.vlines(
                x=x_c[i], ymin=0, ymax=mc_v, color="#aab7b8", linewidth=0.8, zorder=2
            )

    # Plot each source
    for key in src_keys:
        col = f"{key}_mt"
        if col not in countries.columns:
            continue
        style = SOURCES[key]
        mask = countries[col].notna()
        if not mask.any():
            continue

        # Small horizontal jitter per source
        jitter = JITTER_Y.get(key, 0) * 0.8
        ax_c.scatter(
            x_c[mask.values] + jitter,
            countries.loc[mask, col],
            c=style["color"],
            marker=style["marker"],
            s=22 if key != "mc" else 45,
            edgecolors="white",
            linewidths=0.3,
            alpha=0.9,
            zorder=5 if key != "mc" else 6,
        )

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(
        [COUNTRY_NAMES.get(g, g) for g in countries["geo"]],
        rotation=55,
        ha="right",
        fontsize=5.5,
    )
    ax_c.set_ylabel("Change from 2023 (Mt CO2, symlog scale)")
    ax_c.set_xlim(-0.8, n_c - 0.2)
    ax_c.set_ylim(y_lo, y_hi)
    ax_c.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.3)
    ax_c.set_axisbelow(True)
    ax_c.text(
        -0.04,
        1.03,
        "c",
        transform=ax_c.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Spines for all panels
    for ax in [ax_a, ax_b, ax_c]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)

    # =====================================================================
    # Shared legend — right side of figure
    # =====================================================================
    from matplotlib.lines import Line2D

    handles = []
    for key in src_keys:
        s = SOURCES[key]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=s["marker"],
                color="w",
                markerfacecolor=s["color"],
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.3,
                label=s["label"],
            )
        )
    # FF55 dashed line entry
    handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="FF55 target (EU-27)",
        )
    )

    fig.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        framealpha=0.95,
        edgecolor="#bdc3c7",
        fontsize=6.5,
        handletextpad=0.5,
        labelspacing=0.6,
        borderpad=0.6,
    ).get_frame().set_linewidth(0.4)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ["png", "pdf", "svg"]:
        fig.savefig(
            OUTPUT_DIR / f"fig1_emissions_change.{fmt}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR}/fig1_emissions_change.[png|pdf|svg]")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("GENERATING FIGURE 1: EMISSION CHANGE FROM 2023")
    print("=" * 70)

    dataset = load_dataset(DATASET_PATH)
    population_df = load_population_data()

    baseline_mt = get_2023_baseline_mt(dataset, population_df)
    mc_2030_mt = get_2030_mc_mt(dataset, population_df)
    oecd_2030 = load_oecd_2030_mt()
    eea_2030 = load_eea_2030_mt()
    pypsa_2030 = load_pypsa_2030_mt()

    # FF55 target
    oecd_full = load_oecd_full()
    eu27_1990 = oecd_full[(oecd_full["year"] == 1990) & (oecd_full["geo"] == "EU27")]
    eu_base = baseline_mt["EU27"]
    eu_1990_mt = eu27_1990.iloc[0]["value"]
    ff55_mt_abs = eu_1990_mt * 0.45
    ff55_pct = ((ff55_mt_abs - eu_base) / eu_base) * 100
    ff55_mt = ff55_mt_abs - eu_base

    print(f"EU27 2023: {eu_base:.1f} Mt | 1990: {eu_1990_mt:.1f} Mt")
    print(
        f"FF55 target: {ff55_mt_abs:.1f} Mt -> {ff55_pct:+.1f}% / {ff55_mt:+.1f} Mt vs 2023"
    )

    # Build table
    df = build_comparison(baseline_mt, mc_2030_mt, oecd_2030, eea_2030, pypsa_2030)

    # Create figure
    print("\nCreating figure...")
    create_figure(df, ff55_pct, ff55_mt)

    # Export
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df["country_name"] = df["geo"].map(COUNTRY_NAMES)
    df.to_csv(TABLE_DIR / "fig1_change_from_2023_summary.csv", index=False)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY (This study, 2023 -> 2030)")
    print(f"{'=' * 60}")
    print(f"  {'Country':<15s} {'%':>8s} {'Mt CO2':>10s}  {'2023':>8s}  {'2030':>8s}")
    print("  " + "-" * 52)
    for _, row in df.sort_values("mc_mt").iterrows():
        if pd.isna(row["mc_pct"]):
            continue
        name = COUNTRY_NAMES.get(row["geo"], row["geo"])
        base = row["baseline_mt"]
        proj = base + row["mc_mt"]
        print(
            f"  {name:<15s} {row['mc_pct']:>+7.1f}% {row['mc_mt']:>+9.1f}  {base:>7.1f}  {proj:>7.1f}"
        )
    print("\nDone!")


if __name__ == "__main__":
    main()
