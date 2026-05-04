"""
Figure S_MAP v4: Dot map of projected CO2 emission changes (2024-2030).

Design:
  - Country polygons drawn in neutral light gray (geography only)
  - One circle per country, placed at the representative point
  - Dot COLOR  = % change 2024→2030  (RdBu_r)
  - Dot SIZE   = 2024 per-capita emissions (kg CO2/person), LINEAR scale
                 (removes country-size bias; small emitters per capita
                  appear small even if their absolute total is large)
  - Per-panel size legend in kg CO2/person
  - Single shared colorbar per panel

Layout   : one large total map on the left, 2×3 grid of sector maps on the right.
Style    : Nature Climate Change submission guidelines

Usage:
    python scripts/visualization/figure_map_emission_changes_v4.py

Outputs:
    outputs/figures/supplementary/fig_SI_map_emission_changes_v4.pdf
    outputs/figures/supplementary/fig_SI_map_emission_changes_v4.png
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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

XLIM = (-25, 35)
YLIM = (34, 72)

GEO_FACECOLOR = "#e8e8e8"
GEO_EDGECOLOR = "#aaaaaa"
GEO_LINEWIDTH = 0.3
OCEAN_COLOR = "white"

CMAP_NAME = "RdBu_r"
DOT_EDGE_COLOR = "#333333"
DOT_EDGE_WIDTH = 0.3
DOT_ALPHA = 0.88

# Dot size range: scatter s parameter (marker area in points^2)
# Linear scaling of per-capita values
MAX_DOT_PT = 40.0
MIN_DOT_PT = 1.5

FONT_PANEL_LETTER = 8
FONT_TITLE = 7
FONT_CBAR_LABEL = 6
FONT_CBAR_TICK = 5
FONT_LEGEND_TITLE = 5.5
FONT_LEGEND_VAL = 5.0

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
# Data loading
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


def compute_pct_and_percap(dataset, population_df: pd.DataFrame):
    """
    Returns:
      pct_df    — % change 2024→2030  (geo, total, HeatingCooling, ...)
      percap_df — 2024 per-capita emissions in kg CO2/person
                  (geo, total, HeatingCooling, ...)
    """
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

    def agg_year(yr):
        return (
            df_mc[df_mc["year"] == yr]
            .groupby("geo")
            .agg(
                total=("total_CO2", "mean"),
                population=("population", "first"),
                **{s: (f"{s}_total", "mean") for s in OUTPUT_SECTORS},
            )
            .add_suffix(f"_{yr}")
            .reset_index()
        )

    merged = agg_year(2024).merge(agg_year(2030), on="geo")
    merged = merged[merged["geo"].isin(EU27_COUNTRIES)]

    pct = pd.DataFrame({"geo": merged["geo"]})
    percap = pd.DataFrame({"geo": merged["geo"]})

    for col in ["total"] + OUTPUT_SECTORS:
        b = merged[f"{col}_2024"]
        p = merged[f"{col}_2030"]
        pct[col] = (p - b) / b.abs() * 100

    # Per-capita: kg CO2/person = total_CO2 (kg*person ... wait,
    # total_CO2 = phys_kg_per_person * population, so divide back out)
    pop_2024 = merged["population_2024"]
    percap["total"] = merged["total_2024"] / pop_2024  # kg CO2/person
    for s in OUTPUT_SECTORS:
        percap[s] = merged[f"{s}_2024"] / pop_2024  # kg CO2/person

    return pct, percap


def load_eu_geodataframe() -> gpd.GeoDataFrame:
    import io, urllib.request, zipfile
    from shapely.geometry import Point

    cache_dir = Path("data/geodata")
    cache_path = cache_dir / "ne_110m_admin_0_countries.shp"

    if not cache_path.exists():
        print("  Downloading Natural Earth 110m shapefile...")
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
        raise ValueError(f"No usable ISO3 column. Available: {list(world.columns)}")

    world["iso2"] = world[iso_col].map(iso3_to_iso2)
    if "FR" not in set(world["iso2"].dropna()):
        mask = world["NAME"] == "France"
        if mask.any():
            world.loc[mask, "iso2"] = "FR"

    eu = world[world["iso2"].isin(EU27_COUNTRIES)].copy()

    if "MT" not in set(eu["iso2"].dropna()):
        malta_lon, malta_lat = 14.375, 35.937
        malta_row = gpd.GeoDataFrame(
            {"iso2": ["MT"], "geometry": [Point(malta_lon, malta_lat).buffer(0.15)]},
            crs=eu.crs,
        )
        for col in eu.columns:
            if col not in malta_row.columns:
                malta_row[col] = np.nan
        eu = pd.concat([eu, malta_row], ignore_index=True)
        print("  Added Malta (MT) as synthetic point geometry")

    pts = eu.geometry.representative_point()
    eu["dot_x"] = pts.x
    eu["dot_y"] = pts.y
    print(f"  Loaded {len(eu)} EU27 geometries")
    return eu


# =============================================================================
# Dot sizing — LINEAR scale on per-capita values
# =============================================================================


def linear_areas(vals: pd.Series, ref_vals: pd.Series | None = None) -> pd.Series:
    """
    Map per-capita emission values → scatter s (marker area in points^2).
    Linear scaling between MIN_DOT_PT and MAX_DOT_PT.

    Using linear (not log) scaling because per-capita values already
    remove the country-size effect, so the full dynamic range is meaningful
    and should be preserved without compression.

    ref_vals: reference distribution for computing min/max (pass the full
              panel series so legend circles use the same scale as map dots).
    """
    if ref_vals is None:
        ref_vals = vals
    ref_clean = ref_vals.clip(lower=0).fillna(0)
    v_min = ref_clean.min()
    v_max = ref_clean.max()

    v = vals.clip(lower=0).fillna(0)
    if v_max == v_min:
        return pd.Series(MAX_DOT_PT, index=vals.index)
    return MIN_DOT_PT + (v - v_min) / (v_max - v_min) * (MAX_DOT_PT - MIN_DOT_PT)


def _val_to_area_linear(v: float, ref_vals: pd.Series) -> float:
    """Return scatter s for a single kg/person value using the same linear mapping."""
    ref_clean = ref_vals.clip(lower=0).fillna(0)
    v_min = float(ref_clean.min())
    v_max = float(ref_clean.max())
    if v_max == v_min:
        return MAX_DOT_PT
    r_norm = np.clip((v - v_min) / (v_max - v_min), 0, 1)
    return MIN_DOT_PT + r_norm * (MAX_DOT_PT - MIN_DOT_PT)


# =============================================================================
# Per-panel dot-size legend  (values in kg CO2/person)
# =============================================================================


def _nice_legend_values_linear(percap_vals: pd.Series) -> list[float]:
    """
    Return [min, arithmetic_mid, max] of the per-capita distribution,
    rounded to 2 significant figures.
    Arithmetic (not geometric) midpoint because the scale is now linear.
    """
    clean = percap_vals.clip(lower=0).dropna()
    v_min = float(clean.min())
    v_max = float(clean.max())
    v_mid = (v_min + v_max) / 2.0

    def round_2sf(x):
        if x <= 0:
            return x
        mag = 10 ** (np.floor(np.log10(x + 1e-9)) - 1)
        return round(x / mag) * mag

    return [round_2sf(v_min), round_2sf(v_mid), round_2sf(v_max)]


def draw_dot_legend_axes(fig, percap_vals: pd.Series, legend_rect: list):
    """
    Draw a size legend whose circles are pixel-identical to the map dots.
    Values shown in kg CO2/person.
    legend_rect = [left, bottom, width, height] in figure-fraction coords.
    """
    if percap_vals.dropna().empty or percap_vals.max() <= 0:
        return

    ref_vals = _nice_legend_values_linear(percap_vals)
    ref_areas = [_val_to_area_linear(v, percap_vals) for v in ref_vals]

    ax = fig.add_axes(legend_rect)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axis("off")

    map_w = XLIM[1] - XLIM[0]
    map_h = YLIM[1] - YLIM[0]
    y_circ = YLIM[0] + map_h * 0.55
    y_label = YLIM[0] + map_h * 0.15
    y_title = YLIM[0] + map_h * 0.88
    n = len(ref_vals)
    xs = [XLIM[0] + map_w * (0.15 + i * 0.35) for i in range(n)]

    ax.text(
        XLIM[0] + map_w * 0.5,
        y_title,
        "2024 emissions (tCO2/hab)",
        ha="center",
        va="center",
        fontsize=FONT_LEGEND_TITLE,
        color="#444444",
        style="italic",
    )
    ax.scatter(
        xs,
        [y_circ] * n,
        s=ref_areas,
        facecolors="#cccccc",
        edgecolors=DOT_EDGE_COLOR,
        linewidths=DOT_EDGE_WIDTH * 1.5,
        alpha=0.92,
        zorder=3,
        clip_on=False,
    )
    for x, val in zip(xs, ref_vals):
        # Format: show as integer kg if ≥ 10, one decimal if < 10
        label = (
            f"{val / 1e3:.0f}"
            if val / 1e3 >= 10
            else (f"{val / 1e3:.1f}" if val / 1e3 >= 0.1 else f"{val / 1e3:.2f}")
        )
        ax.text(
            x,
            y_label,
            label,
            ha="center",
            va="center",
            fontsize=FONT_LEGEND_VAL,
            color="#333333",
        )


# =============================================================================
# Colormap helpers  (unchanged from v3)
# =============================================================================


def symmetric_vmax(values: pd.Series, pad: float = 1.05, minimum: float = 5.0) -> float:
    clean = values.dropna()
    vmax = max(abs(float(clean.min())), abs(float(clean.max())))
    return max(vmax * pad, minimum)


def data_range(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna()
    return min(float(clean.min()), 0), max(float(clean.max()), 0)


def data_ticks(cbar_min: float, cbar_max: float, n: int = 5) -> np.ndarray:
    return np.unique(np.round(np.linspace(cbar_min, cbar_max, n)).astype(int))


def truncated_cmap(cmap, vmax: float, cbar_min: float, cbar_max: float):
    lo = float(np.clip((cbar_min + vmax) / (2 * vmax), 0, 1))
    hi = float(np.clip((cbar_max + vmax) / (2 * vmax), 0, 1))
    return mcolors.ListedColormap(cmap(np.linspace(lo, hi, 512)))


# =============================================================================
# Drawing
# =============================================================================


def draw_panel(
    ax,
    gdf: gpd.GeoDataFrame,
    pct_col: str,
    percap_col: str,
    norm: mcolors.Normalize,
    cmap,
    title: str,
    letter: str,
):
    ax.set_facecolor(OCEAN_COLOR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gdf.plot(
            ax=ax,
            color=GEO_FACECOLOR,
            edgecolor=GEO_EDGECOLOR,
            linewidth=GEO_LINEWIDTH,
        )

    valid = gdf[gdf[percap_col].notna() & gdf[pct_col].notna()].copy()
    ref_series = gdf[percap_col].dropna()
    areas = linear_areas(valid[percap_col], ref_vals=ref_series)
    colors = [cmap(norm(v)) for v in valid[pct_col]]

    ax.scatter(
        valid["dot_x"],
        valid["dot_y"],
        s=areas,
        c=colors,
        edgecolors=DOT_EDGE_COLOR,
        linewidths=DOT_EDGE_WIDTH,
        alpha=DOT_ALPHA,
        zorder=4,
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


def create_map_figure(
    pct_df: pd.DataFrame,
    percap_df: pd.DataFrame,
    eu_gdf: gpd.GeoDataFrame,
) -> None:
    eu_gdf = eu_gdf.merge(pct_df, left_on="iso2", right_on="geo", how="left")
    percap_renamed = percap_df.rename(
        columns={c: f"{c}_percap" for c in percap_df.columns if c != "geo"}
    )
    eu_gdf = eu_gdf.merge(percap_renamed, left_on="iso2", right_on="geo", how="left")

    cmap = plt.get_cmap(CMAP_NAME)

    vmax_total = symmetric_vmax(pct_df["total"])
    vmaxes = [symmetric_vmax(pct_df[s]) for s in OUTPUT_SECTORS]
    norm_total = mcolors.Normalize(vmin=-vmax_total, vmax=vmax_total)
    norms = [mcolors.Normalize(vmin=-v, vmax=v) for v in vmaxes]

    cmin_total, cmax_total = data_range(pct_df["total"])
    cranges = [data_range(pct_df[s]) for s in OUTPUT_SECTORS]
    ticks_total = data_ticks(cmin_total, cmax_total)
    ticks_list = [data_ticks(cmin, cmax) for cmin, cmax in cranges]

    fig_w_in = 180 / 25.4
    fig_h_in = fig_w_in * 0.52
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

    draw_panel(
        ax_total,
        eu_gdf,
        pct_col="total",
        percap_col="total_percap",
        norm=norm_total,
        cmap=cmap,
        title=r"Total CO$_2$ emissions",
        letter="a",
    )
    for i, (sector, ax) in enumerate(zip(OUTPUT_SECTORS, sector_axes)):
        draw_panel(
            ax,
            eu_gdf,
            pct_col=sector,
            percap_col=f"{sector}_percap",
            norm=norms[i],
            cmap=cmap,
            title=SECTOR_LABELS[sector],
            letter="bcdefg"[i],
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    T = fig.transFigure.inverted()

    cbar_h = 0.013
    dot_leg_h = 0.055
    gap_map_leg = 0.006
    gap_leg_cbar = 0.005
    cx_pad = 0.08

    def _strip_below(ax_bbox, strip_h, gap_above):
        return [
            ax_bbox.x0,
            ax_bbox.y0 - gap_above - strip_h,
            ax_bbox.width,
            strip_h,
        ]

    bbox_total = ax_total.get_tightbbox(renderer).transformed(T)
    draw_dot_legend_axes(
        fig,
        eu_gdf["total_percap"].dropna(),
        _strip_below(bbox_total, dot_leg_h, gap_map_leg),
    )
    cax_total = fig.add_axes(
        [
            bbox_total.x0 + cx_pad * bbox_total.width,
            bbox_total.y0 - gap_map_leg - dot_leg_h - gap_leg_cbar - cbar_h,
            bbox_total.width * (1 - 2 * cx_pad),
            cbar_h,
        ]
    )
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

    for i, (sector, ax, vmax, (cmin, cmax), ticks) in enumerate(
        zip(OUTPUT_SECTORS, sector_axes, vmaxes, cranges, ticks_list)
    ):
        bbox = ax.get_tightbbox(renderer).transformed(T)
        draw_dot_legend_axes(
            fig,
            eu_gdf[f"{sector}_percap"].dropna(),
            _strip_below(bbox, dot_leg_h, gap_map_leg),
        )
        sector_caxs[i].remove()
        cax = fig.add_axes(
            [
                bbox.x0 + cx_pad * bbox.width,
                bbox.y0 - gap_map_leg - dot_leg_h - gap_leg_cbar - cbar_h,
                bbox.width * (1 - 2 * cx_pad),
                cbar_h,
            ]
        )
        draw_colorbar(cax, fig, cmap, vmax, cmin, cmax, ticks, label="Change (%)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        out = OUTPUT_DIR / f"fig_SI_map_emission_changes_v4.{fmt}"
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
    print("GENERATING SI MAP v4 — DOT MAP (color=%, size=per capita linear)")
    print("=" * 60)

    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    population_df = load_population()
    pct_df, percap_df = compute_pct_and_percap(dataset, population_df)
    eu_gdf = load_eu_geodataframe()

    missing = set(EU27_COUNTRIES) - set(eu_gdf["iso2"].dropna())
    if missing:
        print(f"  Warning: missing geometries for {missing}")

    print("\n2024 total per-capita emissions (kg CO2/person), descending:")
    print(
        percap_df[["geo", "total"]]
        .sort_values("total", ascending=False)
        .to_string(index=False)
    )

    create_map_figure(pct_df, percap_df, eu_gdf)
    print("\nDone.")


if __name__ == "__main__":
    main()
