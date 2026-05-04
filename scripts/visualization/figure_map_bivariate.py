"""
Figure S_MAP bivariate: choropleth maps of projected CO2 emission changes
(2024-2030) with a bivariate color scheme:
  - Hue   = % change direction/magnitude (blue=decrease, red=increase)
  - Value = 2024 absolute emissions      (light=small, dark=large)

Bins are purely data-driven (quantiles), controlled by two parameters:
  N_CHANGE_BINS  — granularity of the change axis (odd recommended)
  N_ABS_BINS     — granularity of the absolute emission axis

Thresholds are computed per-panel from that panel's own distribution.

Usage:
    python scripts/visualization/figure_map_bivariate.py
    python -m scripts.visualization.figure_map_bivariate

Outputs:
    outputs/figures/supplementary/fig_SI_map_bivariate.pdf
    outputs/figures/supplementary/fig_SI_map_bivariate.png
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
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
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
# TUNABLE — only these two numbers control granularity
# =============================================================================
N_CHANGE_BINS = 5  # change axis bins  (try 3, 5, 7)
N_ABS_BINS = 3  # abs emission bins (try 2, 3, 4)

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
GEO_EDGE_COLOR = "#888888"
GEO_EDGE_WIDTH = 0.25
MISSING_COLOR = "#dddddd"

FONT_PANEL_LETTER = 8
FONT_TITLE = 7
FONT_LEGEND = 5.0

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
# Color grid
# =============================================================================


def _blend_to_white(hex_color: str, t: float) -> str:
    import matplotlib.colors as mc

    r, g, b = mc.to_rgb(hex_color)
    return mc.to_hex((1 - t * (1 - r), 1 - t * (1 - g), 1 - t * (1 - b)))


def build_color_grid(n_change: int, n_abs: int) -> np.ndarray:
    """
    (n_abs, n_change) array of hex colors.
    Row 0 = smallest emitter (lightest), Row n_abs-1 = largest (darkest).
    Col 0 = largest decrease (blue), Col n_change-1 = largest increase (red).
    """
    import matplotlib.colors as mc

    anchors = ["#1a3f6f", "#6baed6", "#f7f7f7", "#fc8d59", "#b30000"]
    anchor_xs = np.linspace(0, 1, len(anchors))
    target_xs = np.linspace(0, 1, n_change)
    src_rgb = np.array([mc.to_rgb(c) for c in anchors])
    change_colors = [
        mc.to_hex(
            np.clip([np.interp(x, anchor_xs, src_rgb[:, ch]) for ch in range(3)], 0, 1)
        )
        for x in target_xs
    ]

    lightness = np.linspace(0.35, 1.0, n_abs)
    grid = np.empty((n_abs, n_change), dtype=object)
    for ai, lt in enumerate(lightness):
        for ci, base in enumerate(change_colors):
            grid[ai, ci] = _blend_to_white(base, lt)
    return grid


# =============================================================================
# Binning
# =============================================================================


def change_thresholds_zero_anchored(vals: pd.Series, n_bins: int) -> list[float]:
    """
    Compute change bin thresholds that always include 0 as a boundary.
    Quantiles are applied separately within the negative and positive halves,
    so the decrease/increase split is always honoured regardless of distribution.

    n_bins should be odd so 0 is a true midpoint boundary.
    With n_bins=5: 2 decrease bins | 0 boundary | 2 increase bins.
    With n_bins=3: 1 decrease bin  | 0 boundary | 1 increase bin.
    """
    clean = vals.dropna()
    neg = clean[clean < 0]
    pos = clean[clean > 0]

    n_half = n_bins // 2  # bins on each side of zero

    # Inner thresholds for the negative side (exclude 0 and -inf)
    if len(neg) > 0 and n_half > 1:
        qs_neg = np.linspace(0, 100, n_half + 1)[1:-1]
        neg_thresh = sorted([float(np.nanpercentile(neg, q)) for q in qs_neg])
    else:
        neg_thresh = []

    # Inner thresholds for the positive side
    if len(pos) > 0 and n_half > 1:
        qs_pos = np.linspace(0, 100, n_half + 1)[1:-1]
        pos_thresh = sorted([float(np.nanpercentile(pos, q)) for q in qs_pos])
    else:
        pos_thresh = []

    return neg_thresh + [0.0] + pos_thresh


def quantile_thresholds(vals: pd.Series, n_bins: int) -> list[float]:
    """Simple quantile thresholds — used for the abs emission axis."""
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    return [float(np.nanpercentile(vals.dropna(), q)) for q in qs]


def assign_bins(vals: pd.Series, thresholds: list[float]) -> pd.Series:
    return (
        pd.cut(
            vals,
            bins=[-np.inf] + sorted(thresholds) + [np.inf],
            labels=False,
            right=True,
        )
        .fillna(0)
        .astype(int)
    )
    return (
        pd.cut(
            vals,
            bins=[-np.inf] + sorted(thresholds) + [np.inf],
            labels=False,
            right=True,
        )
        .fillna(0)
        .astype(int)
    )


def fmt_pct(v: float) -> str:
    return f"{v:+.0f}%" if abs(v) >= 1 else f"{v:+.1f}%"


def fmt_abs(v: float) -> str:
    return f"{v:.0f}" if v >= 1 else f"{v:.1f}"


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


def compute_pct_and_abs(dataset, population_df: pd.DataFrame):
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
                **{s: (f"{s}_total", "mean") for s in OUTPUT_SECTORS},
            )
            .add_suffix(f"_{yr}")
            .reset_index()
        )

    merged = agg_year(2024).merge(agg_year(2030), on="geo")
    merged = merged[merged["geo"].isin(EU27_COUNTRIES)]
    pct = pd.DataFrame({"geo": merged["geo"]})
    ab = pd.DataFrame({"geo": merged["geo"]})
    for col in ["total"] + OUTPUT_SECTORS:
        b = merged[f"{col}_2024"]
        p = merged[f"{col}_2030"]
        pct[col] = (p - b) / b.abs() * 100
        ab[col] = b / 1e6
    return pct, ab


def load_eu_geodataframe() -> gpd.GeoDataFrame:
    import io, urllib.request, zipfile
    from shapely.geometry import Point

    cache_dir = Path("data/geodata")
    cache_path = cache_dir / "ne_110m_admin_0_countries.shp"
    if not cache_path.exists():
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
        raise ValueError("No usable ISO3 column.")
    world["iso2"] = world[iso_col].map(iso3_to_iso2)
    if "FR" not in set(world["iso2"].dropna()):
        mask = world["NAME"] == "France"
        if mask.any():
            world.loc[mask, "iso2"] = "FR"
    eu = world[world["iso2"].isin(EU27_COUNTRIES)].copy()
    if "MT" not in set(eu["iso2"].dropna()):
        malta_row = gpd.GeoDataFrame(
            {"iso2": ["MT"], "geometry": [Point(14.375, 35.937).buffer(0.15)]},
            crs=eu.crs,
        )
        for col in eu.columns:
            if col not in malta_row.columns:
                malta_row[col] = np.nan
        eu = pd.concat([eu, malta_row], ignore_index=True)
    print(f"  Loaded {len(eu)} EU27 geometries")
    return eu


# =============================================================================
# Drawing
# =============================================================================


def draw_panel(
    ax, gdf, pct_col, abs_col, color_grid, change_thresh, abs_thresh, title, letter
):
    ax.set_facecolor("white")
    gdf = gdf.copy()
    gdf["cb"] = assign_bins(gdf[pct_col], change_thresh)
    gdf["ab"] = assign_bins(gdf[abs_col], abs_thresh)

    # Build a color column, then plot each unique color group at once
    def get_color(row):
        if pd.isna(row.get(pct_col)) or pd.isna(row.get(abs_col)):
            return MISSING_COLOR
        return color_grid[int(row["ab"]), int(row["cb"])]

    gdf["_color"] = gdf.apply(get_color, axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for color, group in gdf.groupby("_color", sort=False):
            group.plot(
                ax=ax, color=color, edgecolor=GEO_EDGE_COLOR, linewidth=GEO_EDGE_WIDTH
            )

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axis("off")
    ax.set_title(title, fontsize=FONT_TITLE, pad=2, loc="center")
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


def draw_legend(fig, rect, color_grid, change_thresh, abs_thresh, n_change, n_abs):
    """
    Bivariate legend grid.
    rect = [left, bottom, width, height] in figure fraction.
    Rows = abs emission level (bottom=low, top=high).
    Cols = % change (left=largest decrease, right=largest increase).
    """
    ax = fig.add_axes(rect)
    ax.set_xlim(-0.8, n_change)  # left margin for y-labels
    ax.set_ylim(-1.2, n_abs + 0.6)  # bottom margin for x-labels, top for title
    ax.axis("off")

    # Color cells
    for ai in range(n_abs):
        for ci in range(n_change):
            ax.add_patch(
                mpatches.Rectangle(
                    (ci, ai),
                    1,
                    1,
                    facecolor=color_grid[ai, ci],
                    edgecolor="white",
                    linewidth=0.6,
                )
            )

    # x-axis labels: show only the boundaries (shorter than bin ranges)
    # Label format: show left edge of each bin except first (<) and last (>)
    xtick_positions = [0] + [i + 1 for i in range(len(change_thresh))] + [n_change]
    xtick_labels = (
        [fmt_pct(change_thresh[0])]  # leftmost boundary
        + [fmt_pct(t) for t in change_thresh]  # interior boundaries
        + [fmt_pct(change_thresh[-1])]  # rightmost boundary (unused, skip)
    )
    # simpler: just label each boundary line
    boundaries = change_thresh  # n_change-1 values
    for i, t in enumerate(boundaries):
        ax.text(
            i + 1,
            -0.15,
            fmt_pct(t),
            ha="center",
            va="top",
            fontsize=FONT_LEGEND,
            color="#333333",
        )
    # outer labels
    ax.text(
        0.5,
        -0.15,
        f"< {fmt_pct(change_thresh[0])}",
        ha="center",
        va="top",
        fontsize=FONT_LEGEND - 0.5,
        color="#666666",
    )
    ax.text(
        n_change - 0.5,
        -0.15,
        f"> {fmt_pct(change_thresh[-1])}",
        ha="center",
        va="top",
        fontsize=FONT_LEGEND - 0.5,
        color="#666666",
    )

    # y-axis labels: one per row, on left
    ylabels = (
        [f"< {fmt_abs(abs_thresh[0])}"]
        + [
            f"{fmt_abs(abs_thresh[i])}–{fmt_abs(abs_thresh[i + 1])}"
            for i in range(len(abs_thresh) - 1)
        ]
        + [f"> {fmt_abs(abs_thresh[-1])}"]
    )
    for ai, lbl in enumerate(ylabels):
        ax.text(
            -0.08,
            ai + 0.5,
            lbl,
            ha="right",
            va="center",
            fontsize=FONT_LEGEND,
            color="#333333",
        )

    # Axis titles
    ax.text(
        n_change / 2,
        -0.85,
        "Change 2024–2030 (%)",
        ha="center",
        va="top",
        fontsize=FONT_LEGEND + 0.5,
        color="#333333",
    )
    ax.text(
        -0.65,
        n_abs / 2,
        "MtCO2\n(2024)",
        ha="center",
        va="center",
        fontsize=FONT_LEGEND + 0.5,
        color="#333333",
        rotation=90,
    )


# =============================================================================
# Figure assembly
# =============================================================================


def create_figure(pct_df, abs_df, eu_gdf):
    eu_gdf = eu_gdf.merge(pct_df, left_on="iso2", right_on="geo", how="left")
    abs_r = abs_df.rename(columns={c: f"{c}_abs" for c in abs_df.columns if c != "geo"})
    eu_gdf = eu_gdf.merge(abs_r, left_on="iso2", right_on="geo", how="left")

    color_grid = build_color_grid(N_CHANGE_BINS, N_ABS_BINS)

    fig_w_in = 180 / 25.4
    fig_h_in = fig_w_in * 0.68
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        left=0.01,
        right=0.99,
        top=0.97,
        bottom=0.28,
        wspace=0.04,
        width_ratios=[1, 2.2],
    )
    ax_total = fig.add_subplot(outer[0])
    right_gs = gridspec.GridSpecFromSubplotSpec(
        2,
        3,
        subplot_spec=outer[1],
        hspace=0.10,
        wspace=0.05,
    )
    sector_axes = [fig.add_subplot(right_gs[r, c]) for r in range(2) for c in range(3)]

    # ── Total panel ───────────────────────────────────────────────────────
    ct = change_thresholds_zero_anchored(pct_df["total"], N_CHANGE_BINS)
    at = quantile_thresholds(abs_df["total"], N_ABS_BINS)
    draw_panel(
        ax_total,
        eu_gdf,
        "total",
        "total_abs",
        color_grid,
        ct,
        at,
        r"Total CO$_2$ emissions",
        "a",
    )

    # ── Sector panels ─────────────────────────────────────────────────────
    sector_thresh = []
    for i, (sector, ax) in enumerate(zip(OUTPUT_SECTORS, sector_axes)):
        ct_s = change_thresholds_zero_anchored(pct_df[sector], N_CHANGE_BINS)
        at_s = quantile_thresholds(abs_df[sector], N_ABS_BINS)
        sector_thresh.append((ct_s, at_s))
        draw_panel(
            ax,
            eu_gdf,
            sector,
            f"{sector}_abs",
            color_grid,
            ct_s,
            at_s,
            SECTOR_LABELS[sector],
            "bcdefg"[i],
        )

    # ── Legends — one per panel, centered below each map ─────────────────
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    T = fig.transFigure.inverted()

    all_axes = [ax_total] + sector_axes
    all_thresh = [(ct, at)] + sector_thresh
    leg_cell = 0.020  # figure-fraction per legend cell — larger for readability

    for ax, (ct_s, at_s) in zip(all_axes, all_thresh):
        bbox = ax.get_tightbbox(renderer).transformed(T)
        leg_w = leg_cell * N_CHANGE_BINS
        leg_h = leg_cell * N_ABS_BINS
        leg_l = bbox.x0 + (bbox.width - leg_w) / 2
        leg_b = 0.06

        draw_legend(
            fig,
            rect=[leg_l, leg_b, leg_w, leg_h],
            color_grid=color_grid,
            change_thresh=ct_s,
            abs_thresh=at_s,
            n_change=N_CHANGE_BINS,
            n_abs=N_ABS_BINS,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        out = OUTPUT_DIR / f"fig_SI_map_bivariate.{fmt}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================


def main():
    print("=" * 60)
    print("GENERATING BIVARIATE MAP")
    print(f"  Grid: {N_CHANGE_BINS} change bins x {N_ABS_BINS} abs bins")
    print("=" * 60)

    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    population_df = load_population()
    pct_df, abs_df = compute_pct_and_abs(dataset, population_df)
    eu_gdf = load_eu_geodataframe()

    missing = set(EU27_COUNTRIES) - set(eu_gdf["iso2"].dropna())
    if missing:
        print(f"  Warning: missing geometries for {missing}")

    create_figure(pct_df, abs_df, eu_gdf)
    print("\nDone.")


if __name__ == "__main__":
    main()
