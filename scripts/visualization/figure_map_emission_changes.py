"""
Figure S_MAP: Choropleth maps of projected emission changes (2024-2030)
by country, for total emissions and each sector individually.

Layout   : one large total map on the left, 2×3 grid of sector maps on the right.
Style    : Nature Climate Change submission guidelines
           - Double-column width: 180 mm (7.09 in)
           - Font: Arial, 6 pt axis labels / ticks, 7 pt panel titles, 8 pt panel letters
           - Vector PDF (pdf.fonttype 42) + 300 dpi PNG

Usage:
    python -m scripts.visualization.figure_map_emission_changes

Outputs:
    outputs/figures/supplementary/fig_SI_map_emission_changes.pdf
    outputs/figures/supplementary/fig_SI_map_emission_changes.png
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

# =============================================================================
# Paths
# =============================================================================

MC_PROJECTIONS_PATH = Path("data/projections/mc_projections.csv")
POPULATION_HIST_PATH = Path("data/full_timeseries/population.csv")
POPULATION_PROJ_PATH = Path("data/full_timeseries/projections/population.csv")
DATASET_PATH = Path("data/pytorch_datasets/unified_dataset.pkl")
OUTPUT_DIR = Path("outputs/figures/supplementary")

# =============================================================================
# Constants
# =============================================================================

OUTPUT_SECTORS = ["HeatingCooling", "Industry", "Land", "Mobility", "Other", "Power"]

SECTOR_LABELS = {
    "HeatingCooling": "Heating & Cooling",
    "Industry": "Industry",
    "Land": "Land Use",
    "Mobility": "Mobility",
    "Other": "Other",
    "Power": "Power",
}

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

ISO2_TO_ISO3 = {
    "AT": "AUT",
    "BE": "BEL",
    "BG": "BGR",
    "HR": "HRV",
    "CY": "CYP",
    "CZ": "CZE",
    "DK": "DNK",
    "EE": "EST",
    "EL": "GRC",
    "FI": "FIN",
    "FR": "FRA",
    "DE": "DEU",
    "HU": "HUN",
    "IE": "IRL",
    "IT": "ITA",
    "LV": "LVA",
    "LT": "LTU",
    "LU": "LUX",
    "MT": "MLT",
    "NL": "NLD",
    "PL": "POL",
    "PT": "PRT",
    "RO": "ROU",
    "SK": "SVK",
    "SI": "SVN",
    "ES": "ESP",
    "SE": "SWE",
}

CMAP_NAME = "RdBu_r"
XLIM = (-25, 35)
YLIM = (34, 72)
EDGE_COLOR = "#555555"
MISSING_COLOR = "#dddddd"

# Nature CC typography
FONT_PANEL_LETTER = 8
FONT_TITLE = 7
FONT_CBAR_LABEL = 6
FONT_CBAR_TICK = 5

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 6,
        "axes.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# =============================================================================
# Data loading & processing
# =============================================================================


def load_population() -> pd.DataFrame:
    pop = pd.concat(
        [
            pd.read_csv(POPULATION_HIST_PATH),
            pd.read_csv(POPULATION_PROJ_PATH),
        ],
        ignore_index=True,
    )
    pop["population"] = pop["population:POP_NC"].astype(float)
    return (
        pop[["geo", "year", "population"]]
        .groupby(["geo", "year"], as_index=False)["population"]
        .mean()
    )


def compute_pct_changes(dataset, population_df: pd.DataFrame) -> pd.DataFrame:
    df_mc = pd.read_csv(MC_PROJECTIONS_PATH)
    df_mc["geo"] = df_mc["geo"].astype(str)

    for s in OUTPUT_SECTORS:
        m = dataset.precomputed_scaling_params[s]["mean"]
        sd = dataset.precomputed_scaling_params[s]["std"]
        df_mc[f"{s}_phys"] = (df_mc[f"emissions_{s}"] * sd + m).clip(lower=0)

    df_mc = df_mc.merge(population_df, on=["geo", "year"], how="left")
    for s in OUTPUT_SECTORS:
        df_mc[f"{s}_total"] = df_mc[f"{s}_phys"] * df_mc["population"]
    df_mc["total_CO2"] = sum(df_mc[f"{s}_total"] for s in OUTPUT_SECTORS)

    def agg_year(year):
        return (
            df_mc[df_mc["year"] == year]
            .groupby("geo")
            .agg(
                total=("total_CO2", "mean"),
                **{s: (f"{s}_total", "mean") for s in OUTPUT_SECTORS},
            )
            .add_suffix(f"_{year}")
            .reset_index()
        )

    merged = agg_year(2024).merge(agg_year(2030), on="geo")
    merged = merged[merged["geo"].isin(EU27_COUNTRIES)]

    results = pd.DataFrame({"geo": merged["geo"]})
    for col in ["total"] + OUTPUT_SECTORS:
        b = merged[f"{col}_2024"]
        p = merged[f"{col}_2030"]
        results[col] = (p - b) / b.abs() * 100

    print("\nProjected changes (%):")
    print(results.set_index("geo").round(1).to_string())
    return results


def load_eu_geodataframe() -> gpd.GeoDataFrame:
    import io, urllib.request, zipfile

    cache_dir = Path("data/geodata")
    cache_path = cache_dir / "ne_110m_admin_0_countries.shp"

    if not cache_path.exists():
        print("  Downloading Natural Earth 110m cultural shapefile...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = (
            "https://naciscdn.org/naturalearth/110m/cultural/"
            "ne_110m_admin_0_countries.zip"
        )
        with urllib.request.urlopen(url) as r:
            zip_data = io.BytesIO(r.read())
        with zipfile.ZipFile(zip_data) as z:
            z.extractall(cache_dir)

    world = gpd.read_file(cache_path)
    iso3_to_iso2 = {v: k for k, v in ISO2_TO_ISO3.items()}

    iso_col = None
    for candidate in ["ISO_A3", "ISO_A3_EH", "iso_a3", "ADM0_A3", "GU_A3"]:
        if candidate in world.columns:
            if world[candidate].map(iso3_to_iso2).notna().sum() > 20:
                iso_col = candidate
                break

    if iso_col is None:
        raise ValueError(f"No ISO3 column found. Available: {list(world.columns)}")

    world["iso2"] = world[iso_col].map(iso3_to_iso2)

    if "FR" not in set(world["iso2"].dropna()):
        mask = world["NAME"] == "France"
        if mask.any():
            world.loc[mask, "iso2"] = "FR"

    eu = world[world["iso2"].isin(EU27_COUNTRIES)].copy()
    print(f"  Loaded {len(eu)} EU27 country geometries")
    return eu


# =============================================================================
# Colormap / norm helpers
# =============================================================================


def symmetric_vmax(values: pd.Series, pad: float = 1.05, minimum: float = 5.0) -> float:
    """Symmetric vmax so the full colormap midpoint (white) sits at zero."""
    clean = values.dropna()
    vmax = max(abs(float(clean.min())), abs(float(clean.max())))
    return max(vmax * pad, minimum)


def data_range(values: pd.Series) -> tuple[float, float]:
    """
    Colorbar display range: actual data min/max, extended to touch 0
    at whichever edge it belongs.  All-negative → (data_min, 0).
    All-positive → (0, data_max).  Mixed → (data_min, data_max).
    """
    clean = values.dropna()
    return min(float(clean.min()), 0), max(float(clean.max()), 0)


def truncated_cmap(cmap, vmax: float, cbar_min: float, cbar_max: float):
    """
    Return a new ListedColormap that is the slice of `cmap` corresponding
    to [cbar_min, cbar_max] within the symmetric range [-vmax, +vmax].

    The map is always drawn with the full symmetric norm so country colours
    are correct. The colorbar uses this truncated cmap with a plain
    Normalize(cbar_min, cbar_max), so the gradient it displays only covers
    the actual data window — no wasted red on an all-blue panel.
    """
    lo = (cbar_min + vmax) / (2 * vmax)  # fraction of full cmap at cbar_min
    hi = (cbar_max + vmax) / (2 * vmax)  # fraction of full cmap at cbar_max
    lo = float(np.clip(lo, 0, 1))
    hi = float(np.clip(hi, 0, 1))
    colors = cmap(np.linspace(lo, hi, 512))
    return mcolors.ListedColormap(colors)


def data_ticks(cbar_min: float, cbar_max: float, n: int = 5) -> np.ndarray:
    """n evenly-spaced integer ticks across [cbar_min, cbar_max]."""
    return np.unique(np.round(np.linspace(cbar_min, cbar_max, n)).astype(int))


# =============================================================================
# Drawing
# =============================================================================


def draw_map(ax, gdf, col, cmap, vmax, title, letter):
    """Choropleth using the full symmetric norm so colours are consistent."""
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mask = gdf[col].notna()
        if mask.any():
            gdf[mask].plot(
                column=col,
                ax=ax,
                cmap=cmap,
                norm=norm,
                linewidth=0.25,
                edgecolor=EDGE_COLOR,
            )
        if (~mask).any():
            gdf[~mask].plot(
                ax=ax,
                color=MISSING_COLOR,
                linewidth=0.25,
                edgecolor=EDGE_COLOR,
            )
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axis("off")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="regular", pad=3, loc="center")
    ax.text(
        -0.02,
        1.02,
        letter,
        transform=ax.transAxes,
        fontsize=FONT_PANEL_LETTER,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def draw_colorbar(cax, fig, cmap_full, vmax, cbar_min, cbar_max, ticks, label):
    """
    Colorbar that shows only the [cbar_min, cbar_max] slice of the gradient.

    Strategy: build a truncated colormap covering exactly that slice, then
    attach a plain Normalize(cbar_min, cbar_max) mappable to the colorbar.
    No set_clim tricks — the truncated cmap IS the correct gradient.
    """
    cmap_trunc = truncated_cmap(cmap_full, vmax, cbar_min, cbar_max)
    norm_cbar = mcolors.Normalize(vmin=cbar_min, vmax=cbar_max)
    sm = cm.ScalarMappable(cmap=cmap_trunc, norm=norm_cbar)
    sm.set_array(np.linspace(cbar_min, cbar_max, 512))

    tick_labels = ["0%" if t == 0 else f"{t:+d}%" for t in ticks]

    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    cbar.set_label(label, fontsize=FONT_CBAR_LABEL, labelpad=1.5)
    cbar.ax.tick_params(length=2, width=0.4, pad=1.5, labelsize=FONT_CBAR_TICK)
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor("#888888")


# =============================================================================
# Figure assembly
# =============================================================================


def create_map_figure(pct_df: pd.DataFrame, eu_gdf: gpd.GeoDataFrame) -> None:
    eu_gdf = eu_gdf.merge(pct_df, left_on="iso2", right_on="geo", how="left")
    cmap = plt.get_cmap(CMAP_NAME)

    vmax_total = symmetric_vmax(pct_df["total"])
    vmaxes = [symmetric_vmax(pct_df[s]) for s in OUTPUT_SECTORS]

    cmin_total, cmax_total = data_range(pct_df["total"])
    cranges = [data_range(pct_df[s]) for s in OUTPUT_SECTORS]

    ticks_total = data_ticks(cmin_total, cmax_total)
    ticks_list = [data_ticks(cmin, cmax) for cmin, cmax in cranges]

    # Nature CC double-column = 180 mm
    fig_w_in = 180 / 25.4
    fig_h_in = fig_w_in * 0.50
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        left=0.005,
        right=0.995,
        bottom=0.005,
        top=0.965,
        wspace=0.04,
        width_ratios=[1, 2.2],
    )

    left_gs = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer[0],
        height_ratios=[22, 1],
        hspace=0.06,
    )
    ax_total = fig.add_subplot(left_gs[0])
    # cax_total is placed manually after drawing (see below)

    right_gs = gridspec.GridSpecFromSubplotSpec(
        4,
        3,
        subplot_spec=outer[1],
        height_ratios=[22, 1, 22, 1],
        hspace=0.55,
        wspace=0.18,
    )
    sector_axes, sector_caxs = [], []
    for i in range(len(OUTPUT_SECTORS)):
        r, c = divmod(i, 3)
        sector_axes.append(fig.add_subplot(right_gs[r * 2, c]))
        sector_caxs.append(fig.add_subplot(right_gs[r * 2 + 1, c]))

    # Maps
    draw_map(
        ax_total,
        eu_gdf,
        "total",
        cmap,
        vmax_total,
        title=r"Total CO$_2$ emissions",
        letter="a",
    )
    for i, (sector, ax) in enumerate(zip(OUTPUT_SECTORS, sector_axes)):
        draw_map(
            ax,
            eu_gdf,
            sector,
            cmap,
            vmaxes[i],
            title=SECTOR_LABELS[sector],
            letter="bcdefg"[i],
        )

    # Place cax_total just below the actual map content.
    # We use the renderer to get the tight bbox of ax_total so the colorbar
    # sits snug under the geographic content rather than the full axes box.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    map_bbox = ax_total.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    cbar_h = 0.013  # colorbar height in figure fraction
    cbar_gap = 0.008  # gap below map content
    cbar_x_pad = 0.08  # horizontal inset as fraction of map width
    cax_total = fig.add_axes(
        [
            map_bbox.x0 + cbar_x_pad * map_bbox.width,
            map_bbox.y0 - cbar_gap - cbar_h,
            map_bbox.width * (1 - 2 * cbar_x_pad),
            cbar_h,
        ]
    )

    # Colorbars
    draw_colorbar(
        cax_total,
        fig,
        cmap,
        vmax_total,
        cmin_total,
        cmax_total,
        ticks_total,
        label="Change 2024–2030 (%)",
    )
    for cax, vmax, (cmin, cmax), ticks in zip(sector_caxs, vmaxes, cranges, ticks_list):
        draw_colorbar(cax, fig, cmap, vmax, cmin, cmax, ticks, label="Change (%)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        out = OUTPUT_DIR / f"fig_SI_map_emission_changes.{fmt}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")

    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================


def main():
    print("=" * 60)
    print("GENERATING SI MAP FIGURE: EMISSION CHANGES BY COUNTRY")
    print("=" * 60)

    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    population_df = load_population()
    pct_df = compute_pct_changes(dataset, population_df)
    eu_gdf = load_eu_geodataframe()

    missing = set(EU27_COUNTRIES) - set(eu_gdf["iso2"].dropna())
    if missing:
        print(f"  Warning: missing geometries for {missing}")

    create_map_figure(pct_df, eu_gdf)
    print("\nDone.")


if __name__ == "__main__":
    main()
