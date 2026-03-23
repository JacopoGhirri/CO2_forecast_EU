"""
Figure: Gradient flow Sankey diagrams for emission and uncertainty attribution.

Generates publication-quality Sankey diagrams showing the gradient flow from
raw input variables through latent dimensions to emission sector outputs.
Ribbons are stacked (not overlapping): each ribbon occupies a vertical slice
of its source and destination nodes, and ribbon widths sum exactly to node
heights.

Three-column layout:
    Left:   Input variables (raw socioeconomic indicators)
    Centre: Latent dimensions (VAE bottleneck)
    Right:  Emission sectors (predictor outputs)

Flow normalisation:
    1. Every sector receives the same total inflow (equal visual weight).
    2. Each latent node's left-side inflow equals its right-side outflow.
    3. Input and latent node heights follow from the flows passing through.

Monthly/quarterly variable suffixes are aggregated by summing absolute
activations back to the parent variable.

Filtering (set exactly one, leave the other None):
    - TOP_K_PER_DEST: union of the K strongest links per destination node.
    - TOP_K_TOTAL: keep the K nodes with the highest total flow in each
      column (inputs and latents are filtered independently).

Toggles:
    - SHOW_CONTEXT: include or hide context variables (GDP, population,
      climate) in the latent→sector links.
    - SHOW_UNCERTAINTY: show both panels or emissions only.

Prerequisites:
    - data/explainability/predictor_gradient_attributions.csv
    - data/explainability/encoder_gradient_attributions.csv

Usage:
    python -m scripts.visualization.figure_gradient_sankey

Outputs:
    - outputs/figures/fig_gradient_sankey[_emissions_only].pdf
    - outputs/figures/fig_gradient_sankey[_emissions_only].png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path as MplPath

# =============================================================================
# Configuration
# =============================================================================

PREDICTOR_CSV = Path("data/explainability/predictor_gradient_attributions.csv")
ENCODER_CSV = Path("data/explainability/encoder_gradient_attributions.csv")
OUTPUT_DIR = Path("outputs/figures")

# ── Filtering (set exactly one, leave the other None) ────────────────────────
TOP_K_PER_DEST: int | None = 5
TOP_K_TOTAL: int | None = None

# ── Toggles ──────────────────────────────────────────────────────────────────
SHOW_CONTEXT: bool = True
SHOW_UNCERTAINTY: bool = False

# When True, sector node heights are proportional to their projected 2030
# emissions (median).  When False, all sectors receive equal visual weight.
SCALE_SECTORS_BY_EMISSIONS: bool = True

# Projected 2030 median sectoral emissions (Gt CO2).
# Used only when SCALE_SECTORS_BY_EMISSIONS is True.
SECTOR_EMISSIONS_2030 = {
    "HeatingCooling": 0.238,
    "Industry":       0.643,
    "Land":           0.123,
    "Mobility":       0.869,
    "Other":          0.135,
    "Power":          0.354,
}

# ── Visual tuning ────────────────────────────────────────────────────────────
LINK_ALPHA: float = 0.35
NODE_WIDTH: float = 0.022
NODE_GAP: float = 0.018
COLOR_LINK = "#999999"         # gray ribbons

# ── Sector colours ───────────────────────────────────────────────────────────
COLORS_SECTOR = {
    "Mobility": "#2980b9",
    "Industry": "#717d7e",
    "Power": "#e67e22",
    "HeatingCooling": "#e74c3c",
    "Land": "#27ae60",
    "Other": "#8e44ad",
}

# ── Centre column colours (latent / socioeconomic / climate) ─────────────────
COLOR_LATENT = "#2C3E50"
COLOR_SOCIOECONOMIC = "#D4AC0D"    # warm gold for GDP / Population
COLOR_CLIMATE = "#1ABC9C"          # teal for climate variables

# Mapping: human-readable context name → colour category
CENTRE_COLOR_MAP = {
    "GDP": COLOR_SOCIOECONOMIC,
    "Population": COLOR_SOCIOECONOMIC,
    "Precipitation (pop.)": COLOR_CLIMATE,
    "Temperature (pop.)": COLOR_CLIMATE,
    "Temperature (area)": COLOR_CLIMATE,
    "Temp. Variability (pop.)": COLOR_CLIMATE,
    "Temp. Variability (area)": COLOR_CLIMATE,
    "Precipitation (area)": COLOR_CLIMATE,
}

# ── Input variable grouping with colours ─────────────────────────────────────
# Maps human-readable variable names to thematic groups.
# Variables not listed here fall into "Other" with a default colour.
# Change this dictionary to adjust the colour-coding.
INPUT_GROUPS = {
    "Fuel Prices": {
        "color": "#717d7e",
        "members": [
            "Diesel Price", "Gasoline Price", "Light Fuel Oil Price",
        ],
    },
    "Electricity Mix": {
        "color": "#E67E22",
        "members": [
            "Fossil Fuels Power", "Solar Power",
            "Wind Power", "Hydro Power", "Renewables (total)",
            "Pumped Electricity Storage", "Electricity Exports",
            "Electricity Imports", "Electricity Losses",
        ],
    },
    "Heating Tech.": {
        "color": "#e74c3c",
        "members": [
            "Heat Pump Capacity", "Solar Thermal Surface",
        ],
    },
    "Transport": {
        "color": "#2980B9",
        "members": [
            "Sales of BEV Cars", "Sales of BEV Buses",
            "EV Car Stock Share", "EV Bus Stock Share",
            "Modal Split (road)", "Modal Split (rail)",
            "Modal Split (air)", "Modal Split (sea)",
            "Passenger Trains (diesel)", "Passenger Trains (elec.)",
            "Goods Trains (diesel)", "Goods Trains (elec.)",
        ],
    },
    "Land & Agri.": {
        "color": "#27AE60",
        "members": [
            "Cropland Area", "Forest Area", "Wheat Production",
            "Meat Production", "Fruit Production",
            "Poultry Production", "Vegetable Production",
        ],
    },
    "Trade": {
        "color": "#8E44AD",
        "members": [
            "Import Vol. (fuels)", "Import Vol. (raw mat.)",
            "Import Vol. (chemicals)", "Import Vol. (food)",
            "Import Vol. (machinery)", "Export Vol. (fuels)",
            "Export Vol. (raw mat.)", "Export Vol. (chemicals)",
            "Export Vol. (machinery)",
        ],
    },
    "Energy Demand": {
        "color": "#D35400",
        "members": [
            "Energy Cons. (total)", "Industrial Energy Consumption",
            "Transportation Energy Consumption",
        ],
    },
    "Policy": {
        "color": "#717D7E",
        "members": [
            "ETS Carbon Price", "Energy Taxes",
        ],
    },
}
INPUT_DEFAULT_COLOR = "#95A5A6"    # gray for ungrouped variables


def _input_color(human_name: str) -> str:
    """Look up the colour for an input variable by its human-readable name."""
    for group in INPUT_GROUPS.values():
        if human_name in group["members"]:
            return group["color"]
    return INPUT_DEFAULT_COLOR


def _centre_color(human_name: str) -> str:
    """Look up the colour for a centre-column node."""
    if human_name in CENTRE_COLOR_MAP:
        return CENTRE_COLOR_MAP[human_name]
    return COLOR_LATENT

SECTOR_ORDER = [
    "Mobility", "Industry", "Power", "HeatingCooling", "Land", "Other",
]
SECTOR_LABELS = {s: s for s in SECTOR_ORDER}

CONTEXT_PREFIXES = ["gdp_quarterly", "population", "climate"]
MONTHLY_SUFFIXES = {f"_{i}" for i in range(1, 13)}

NAME_MAP = {
    "Monthly_electricity_statistics:Net Electricity Production:Solar": "Solar Power",
    "Monthly_electricity_statistics:Net Electricity Production:Total Combustible Fuels": "Fossil Fuels Power",
    "Monthly_electricity_statistics:Net Electricity Production:Wind": "Wind Power",
    "Monthly_electricity_statistics:Used for pumped storage:Electricity": "Pumped Electricity Storage",
    "Monthly_oil_price_statistics:Diesel (unit/litre):Total price:US dollars": "Diesel Price",
    "Monthly_oil_price_statistics:Domestic heating oil (unit/litre):Total price:US dollars": "Light Fuel Oil Price",
    "Monthly_oil_price_statistics:Gasoline (unit/litre):Total price:US dollars": "Gasoline Price",
    "gdp_quarterly:MillionEUR": "GDP",
    "climate:rainfall:POP": "Precipitation (pop.)",
    "climate:temperature:POP": "Temperature (pop.)",
    "climate:temperature:AREA": "Temperature (area)",
    "climate:temperature_variability:POP": "Temp. Variability (pop.)",
    "climate:temperature_variability:AREA": "Temp. Variability (area)",
    "climate:rainfall:AREA": "Precipitation (area)",
    "population:POP_NC": "Population",
    "carbon_price:EU_ETS": "ETS Carbon Price",
    "heat_pumps:GWH": "Heat Pump Capacity",
    "land_use:Cropland:Area:1000_ha": "Cropland Area",
    "land_use:Forest_land:Area:1000_ha": "Forest Area",
    "modal_split_transport:AIR": "Modal Split (air)",
    "modal_split_transport:RAIL": "Modal Split (rail)",
    "modal_split_transport:ROAD": "Modal Split (road)",
    "modal_split_transport:SEA": "Modal Split (sea)",
    "energy_taxes:MIOEUR": "Energy Taxes",
    "EV_data:EV sales:Cars:BEV:Vehicles": "Sales of BEV Cars",
    "EV_data:EV stock share:Cars:EV:percent": "EV Car Stock Share",
    "EV_data:EV sales:Buses:BEV:Vehicles": "Sales of BEV Buses",
    "EV_data:EV stock share:Buses:EV:percent": "EV Bus Stock Share",
    "crops_livestock:Wheat:Production:t": "Wheat Production",
    "crops_livestock:Meat,_Total:Production:t": "Meat Production",
    "crops_livestock:Fruit_Primary:Production:t": "Fruit Production",
    "crops_livestock:Meat,_Poultry:Production:t": "Poultry Production",
    "crops_livestock:Vegetables_Primary:Production:t": "Vegetable Production",
    "trade:import_volume_index:Raw_materials:WORLD": "Import Vol. (raw mat.)",
    "trade:import_volume_index:Mineral_fuels_lubrificants:WORLD": "Import Vol. (fuels)",
    "trade:import_volume_index:Chemicals:WORLD": "Import Vol. (chemicals)",
    "trade:import_volume_index:Food_drinks_tobacco:WORLD": "Import Vol. (food)",
    "trade:import_volume_index:Machinery_transportequipment:WORLD": "Import Vol. (machinery)",
    "trade:export_volume_index:Raw_materials:WORLD": "Export Vol. (raw mat.)",
    "trade:export_volume_index:Mineral_fuels_lubrificants:WORLD": "Export Vol. (fuels)",
    "trade:export_volume_index:Chemicals:WORLD": "Export Vol. (chemicals)",
    "trade:export_volume_index:Machinery_transportequipment:WORLD": "Export Vol. (machinery)",
    "train_performance:Passenger_trains:Total:Diesel:THS_train_mk": "Passenger Trains (diesel)",
    "train_performance:Passenger_trains:Total:Electricity:THS_train_mk": "Passenger Trains (elec.)",
    "train_performance:Goods_trains:Total:Diesel:THS_train_mk": "Goods Trains (diesel)",
    "train_performance:Goods_trains:Total:Electricity:THS_train_mk": "Goods Trains (elec.)",
    "Monthly_electricity_statistics:Distribution Losses:Electricity": "Electricity Losses",
    "Monthly_electricity_statistics:Net Electricity Production:Hydro": "Hydro Power",
    "Monthly_electricity_statistics:Net Electricity Production:Total Renewables (Hydro, Geo, Solar, Wind, Other)": "Renewables (total)",
    "Monthly_electricity_statistics:Total Exports:Electricity": "Electricity Exports",
    "Monthly_electricity_statistics:Total Imports:Electricity": "Electricity Imports",
    "solar_thermal_surface:THS_M2": "Solar Thermal Surface",
    "energy_consumption:FC_E:GJ_HAB": "Energy Cons. (total)",
    "energy_consumption:FC_IND_E:GJ_HAB": "Industrial Energy Consumption",
    "energy_consumption:FC_TRA_E:GJ_HAB": "Transportation Energy Consumption",
}


# =============================================================================
# Helpers
# =============================================================================


def setup_style():
    """Configure matplotlib for publication output."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": 0.5,
        "figure.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def strip_monthly_suffix(name: str) -> str:
    """Strip ``_<month>`` suffixes, protecting ``latent_<N>`` names."""
    if name.startswith("latent_"):
        return name
    for sfx in sorted(MONTHLY_SUFFIXES, key=len, reverse=True):
        if name.endswith(sfx):
            base = name[: -len(sfx)]
            if base and not base[-1].isdigit():
                return base
    return name


def humanise(name: str) -> str:
    """Map internal variable name to human-readable label."""
    if name in NAME_MAP:
        return NAME_MAP[name]
    if name.startswith("latent_"):
        return name.replace("_", " ").title()
    return name.replace("_", " ")


def is_context(name: str) -> bool:
    """True for GDP / population / climate variables."""
    return any(name.startswith(p) for p in CONTEXT_PREFIXES)


def _lsort(name: str) -> tuple[int, str]:
    """Sort key: latent dims first (numeric), then context alphabetically."""
    if name.startswith("latent_"):
        try:
            return (0, f"{int(name.split('_')[1]):04d}")
        except (IndexError, ValueError):
            return (0, name)
    return (1, name)


# =============================================================================
# Data Loading
# =============================================================================


def _aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Sum monthly/quarterly column variants back to parent variable."""
    groups: dict[str, list[str]] = {}
    for col in df.columns:
        groups.setdefault(strip_monthly_suffix(col), []).append(col)
    return pd.DataFrame(
        {base: df[cols].sum(axis=1) for base, cols in groups.items()},
        index=df.index,
    )


def load_predictor(csv_path: Path, show_ctx: bool):
    """Load predictor gradient attributions → (emission_df, uncertainty_df)."""
    raw = pd.read_csv(csv_path, index_col=0)
    df = raw[[c for c in raw.columns if c.endswith("_absolute")]].copy()
    df.columns = [c.replace("_absolute", "") for c in df.columns]
    df = _aggregate_monthly(df)
    if not show_ctx:
        df = df[[c for c in df.columns if not is_context(c)]]

    emi = df.loc[[i for i in df.index if i.startswith("emission_delta_")]].copy()
    unc = df.loc[[i for i in df.index if i.startswith("uncertainty_")]].copy()
    emi.index = [r.replace("emission_delta_", "") for r in emi.index]
    unc.index = [r.replace("uncertainty_", "") for r in unc.index]
    return emi, unc


def load_encoder(csv_path: Path):
    """Load encoder gradient attributions (latent means ← input features)."""
    raw = pd.read_csv(csv_path, index_col=0)
    df = raw[[c for c in raw.columns if c.endswith("_absolute")]].copy()
    df.columns = [c.replace("_absolute", "") for c in df.columns]
    return _aggregate_monthly(df)


# =============================================================================
# Link Building & Filtering
# =============================================================================


def build_right(pred_df: pd.DataFrame):
    """Build (latent/context, sector, weight) links from predictor attributions."""
    return [(var, sec, float(pred_df.loc[sec, var]))
            for sec in pred_df.index for var in pred_df.columns
            if pred_df.loc[sec, var] > 0]


def build_left(enc_df: pd.DataFrame, needed: set[str]):
    """Build (input_feature, latent, weight) links from encoder attributions."""
    links = []
    for row in enc_df.index:
        short = row.replace("latent_mean_", "latent_")
        if short not in needed:
            continue
        for feat in enc_df.columns:
            w = float(enc_df.loc[row, feat])
            if w > 0:
                links.append((feat, short, w))
    return links


def filt_topk_dest(links, k):
    """Keep the union of the K strongest links per destination."""
    by_d: dict[str, list] = {}
    for lnk in links:
        by_d.setdefault(lnk[1], []).append(lnk)
    out = set()
    for dl in by_d.values():
        for lnk in sorted(dl, key=lambda x: x[2], reverse=True)[:k]:
            out.add(lnk)
    return list(out)


def filt_topk_total(links, k, key=0):
    """Keep all links connected to the K nodes with highest total flow."""
    tots: dict[str, float] = {}
    for lnk in links:
        tots[lnk[key]] = tots.get(lnk[key], 0.0) + lnk[2]
    top = set(sorted(tots, key=tots.get, reverse=True)[:k])
    return [lnk for lnk in links if lnk[key] in top]


def apply_filt(links, kd, kt, key=0):
    """Apply the active filtering strategy."""
    if kd is not None:
        return filt_topk_dest(links, kd)
    if kt is not None:
        return filt_topk_total(links, kt, key=key)
    return links


# =============================================================================
# Flow Normalisation
# =============================================================================


def normalise(right, left, scale_by_emissions=False):
    """
    Normalise so latent left = latent right and sector totals are consistent.

    When ``scale_by_emissions`` is False, every sector receives total = 1.0
    (equal visual weight).  When True, each sector's total is proportional
    to its projected 2030 emissions from SECTOR_EMISSIONS_2030, so that
    larger-emitting sectors appear visually larger.

    Returns (right_norm, left_norm).
    """
    # Determine target total per sector
    if scale_by_emissions:
        # Normalise emission shares so they sum to n_sectors (same total
        # budget as the equal-weight case, preserving overall figure scale)
        sectors_present = list({d for _, d, _ in right})
        raw_shares = {s: SECTOR_EMISSIONS_2030.get(s, 0.1) for s in sectors_present}
        share_sum = sum(raw_shares.values())
        n = len(sectors_present)
        sector_target = {s: v / share_sum * n for s, v in raw_shares.items()}
    else:
        sector_target = {d: 1.0 for _, d, _ in right}

    # Step 1: per-sector normalisation to target
    stot: dict[str, float] = {}
    for s, d, w in right:
        stot[d] = stot.get(d, 0) + w
    rn = [(s, d, w / stot[d] * sector_target.get(d, 1.0) if stot[d] > 0 else 0)
          for s, d, w in right]

    # Per-latent right-side total
    lr: dict[str, float] = {}
    for s, _, w in rn:
        if s.startswith("latent_"):
            lr[s] = lr.get(s, 0) + w

    # Step 2: per-latent left normalisation
    ll: dict[str, float] = {}
    for _, d, w in left:
        ll[d] = ll.get(d, 0) + w

    ln = [(s, d, w / ll[d] * lr.get(d, 0)
           if ll[d] > 0 and lr.get(d, 0) > 0 else 0)
          for s, d, w in left]

    return rn, ln


# =============================================================================
# Sankey Drawing Engine
# =============================================================================


def _draw_ribbon(ax, x0, y0t, y0b, x1, y1t, y1b, color, alpha):
    """Filled Bézier ribbon between two vertical slots."""
    dx = x1 - x0
    verts = [
        (x0, y0t), (x0 + .4*dx, y0t), (x1 - .4*dx, y1t), (x1, y1t),
        (x1, y1b), (x1 - .4*dx, y1b), (x0 + .4*dx, y0b), (x0, y0b),
        (x0, y0t),
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(mpatches.PathPatch(
        MplPath(verts, codes), fc=color, ec="none", alpha=alpha, lw=0))


def _stack(names, raw_heights, gap):
    """
    Stack nodes vertically (top-to-bottom) with fixed gaps.

    Returns (positions, scale) where:
        positions: dict name → (y_bottom, visual_height)
        scale: single float so that  raw_weight × scale = visual_height
    """
    n = len(names)
    if n == 0:
        return {}, 1.0
    total_raw = sum(raw_heights.get(nm, 1e-12) for nm in names)
    usable = 1.0 - (n + 1) * gap
    scale = usable / total_raw if total_raw > 0 else 1.0
    pos = {}
    cursor = 1.0 - gap
    for nm in names:
        h = raw_heights.get(nm, 1e-12) * scale
        pos[nm] = (cursor - h, h)
        cursor = cursor - h - gap
    return pos, scale


def draw_panel(ax, right_links, left_links, title):
    """
    Draw one Sankey panel with properly stacked, non-overlapping ribbons.

    Ordering:
        - Inputs: sorted by total flow (largest at top).
        - Centre: latent dims first (largest at top), then context vars
          (largest at top), separated by a visual gap.
        - Sectors: canonical SECTOR_ORDER.

    Colour-coding:
        - Inputs: by thematic group (INPUT_GROUPS dictionary).
        - Centre: latent dims in dark slate, socioeconomic in gold,
          climate in teal.
        - Sectors: per COLORS_SECTOR.
        - Ribbons: gray.
    """
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    # ── Identify nodes ──────────────────────────────────────────────────────
    sectors = [s for s in SECTOR_ORDER if any(d == s for _, d, _ in right_links)]
    cset = {s for s, _, _ in right_links} | {d for _, d, _ in left_links}
    input_set = {s for s, _, _ in left_links}

    # ── Compute raw heights = total flow ────────────────────────────────────
    def _tally(lks, k):
        t = {}
        for lnk in lks:
            t[lnk[k]] = t.get(lnk[k], 0) + lnk[2]
        return t

    sec_h = _tally(right_links, 1)
    cen_r = _tally(right_links, 0)
    cen_l = _tally(left_links, 1)
    inp_h = _tally(left_links, 0)
    cen_h = {nm: max(cen_r.get(nm, 0), cen_l.get(nm, 0), 1e-12) for nm in cset}
    for s in sectors:
        sec_h.setdefault(s, 1e-12)
    for nm in input_set:
        inp_h.setdefault(nm, 1e-12)

    # ── Order inputs by total flow (largest first = top) ────────────────────
    inputs = sorted(input_set, key=lambda nm: -inp_h.get(nm, 0))

    # ── Order centre: latents first (by flow desc), then context (by flow desc)
    latent_nodes = sorted(
        [nm for nm in cset if nm.startswith("latent_")],
        key=lambda nm: -cen_h.get(nm, 0),
    )
    context_nodes = sorted(
        [nm for nm in cset if not nm.startswith("latent_")],
        key=lambda nm: -cen_h.get(nm, 0),
    )
    centres = latent_nodes + context_nodes

    # ── Stack positions (one scale per column) ──────────────────────────────
    x_L, x_C, x_R = -0.02, 0.50, 1.02
    pos_in, sc_in = _stack(inputs, inp_h, NODE_GAP)
    pos_cn, sc_cn = _stack(centres, cen_h, NODE_GAP)
    pos_sc, sc_sc = _stack(sectors, sec_h, NODE_GAP)

    # ── Slot cursors ────────────────────────────────────────────────────────
    cur_in_R = {nm: yb + h for nm, (yb, h) in pos_in.items()}
    cur_cn_L = {nm: yb + h for nm, (yb, h) in pos_cn.items()}
    cur_cn_R = {nm: yb + h for nm, (yb, h) in pos_cn.items()}
    cur_sc_L = {nm: yb + h for nm, (yb, h) in pos_sc.items()}

    # ── Right ribbons (centre → sector) ─────────────────────────────────────
    for src, dst, w in sorted(right_links, key=lambda t: (
        SECTOR_ORDER.index(t[1]) if t[1] in SECTOR_ORDER else 99, _lsort(t[0]))):
        if w <= 0 or src not in pos_cn or dst not in pos_sc:
            continue
        h_s = w * sc_cn
        h_d = w * sc_sc
        y0t = cur_cn_R[src]; y0b = y0t - h_s; cur_cn_R[src] = y0b
        y1t = cur_sc_L[dst]; y1b = y1t - h_d; cur_sc_L[dst] = y1b
        _draw_ribbon(ax, x_C + NODE_WIDTH/2, y0t, y0b,
                     x_R - NODE_WIDTH/2, y1t, y1b, COLOR_LINK, LINK_ALPHA)

    # ── Left ribbons (input → latent) ───────────────────────────────────────
    for src, dst, w in sorted(left_links, key=lambda t: (_lsort(t[1]), t[0])):
        if w <= 0 or src not in pos_in or dst not in pos_cn:
            continue
        h_s = w * sc_in
        h_d = w * sc_cn
        y0t = cur_in_R[src]; y0b = y0t - h_s; cur_in_R[src] = y0b
        y1t = cur_cn_L[dst]; y1b = y1t - h_d; cur_cn_L[dst] = y1b
        _draw_ribbon(ax, x_L + NODE_WIDTH/2, y0t, y0b,
                     x_C - NODE_WIDTH/2, y1t, y1b, COLOR_LINK, LINK_ALPHA)

    # ── Node bars ───────────────────────────────────────────────────────────
    NW = NODE_WIDTH
    TEXT_PAD = 0.012  # horizontal gap between node edge and label

    # Input nodes — coloured by thematic group
    for nm, (yb, h) in pos_in.items():
        hname = humanise(nm)
        ax.add_patch(plt.Rectangle((x_L - NW/2, yb), NW, h,
                                    fc=_input_color(hname), ec="none"))
        ax.text(x_L - NW/2 - TEXT_PAD, yb + h/2, hname,
                ha="right", va="center", fontsize=8.5, color="#222")

    # Centre nodes — latent / socioeconomic / climate colours
    for nm, (yb, h) in pos_cn.items():
        hname = humanise(nm)
        ax.add_patch(plt.Rectangle((x_C - NW/2, yb), NW, h,
                                    fc=_centre_color(hname), ec="none"))
        ax.text(x_C - NW/2 - TEXT_PAD, yb + h/2, hname,
                ha="right", va="center", fontsize=8.5, color="#222")

    # Sector nodes — per-sector colours
    for nm, (yb, h) in pos_sc.items():
        ax.add_patch(plt.Rectangle((x_R - NW/2, yb), NW, h,
                                    fc=COLORS_SECTOR.get(nm, "#95A5A6"),
                                    ec="none"))
        ax.text(x_R + NW/2 + TEXT_PAD, yb + h/2, SECTOR_LABELS.get(nm, nm),
                ha="left", va="center", fontsize=9.5, color="#222",
                fontweight="medium")

    #ax.set_title(title, fontsize=11, pad=10)


# =============================================================================
# Main Figure
# =============================================================================


def create_sankey_figure(
    predictor_csv: Path = PREDICTOR_CSV,
    encoder_csv: Path = ENCODER_CSV,
    output_dir: Path = OUTPUT_DIR,
    top_k_per_dest: int | None = TOP_K_PER_DEST,
    top_k_total: int | None = TOP_K_TOTAL,
    show_context: bool = SHOW_CONTEXT,
    show_uncertainty: bool = SHOW_UNCERTAINTY,
    scale_by_emissions: bool = SCALE_SECTORS_BY_EMISSIONS,
):
    """
    Create the gradient-flow Sankey figure.

    Args:
        predictor_csv: Predictor gradient attributions CSV.
        encoder_csv: Encoder gradient attributions CSV.
        output_dir: Output directory.
        top_k_per_dest: Union of K strongest per destination (or None).
        top_k_total: K nodes with highest total flow (or None).
        show_context: Include context variables.
        show_uncertainty: Show both panels or emissions only.
        scale_by_emissions: If True, sector node heights are proportional
            to their projected 2030 emissions.
    """
    setup_style()
    print("=" * 70)
    print("GENERATING GRADIENT FLOW SANKEY DIAGRAM")
    print("=" * 70)

    emi_df, unc_df = load_predictor(predictor_csv, show_context)
    enc_df = load_encoder(encoder_csv)
    print(f"  Predictor (emi): {emi_df.shape}, (unc): {unc_df.shape}")
    print(f"  Encoder:         {enc_df.shape}")
    print(f"  Scale sectors by emissions: {scale_by_emissions}")

    specs = [("emission", emi_df, "(a) Total emissions gradient flow")]
    if show_uncertainty:
        specs.append(("uncertainty", unc_df, "(b) Uncertainty gradient flow"))

    panels = []
    for name, pdf, title in specs:
        rl = apply_filt(build_right(pdf), top_k_per_dest, top_k_total, key=0)
        needed = {s for s, _, _ in rl if s.startswith("latent_")}
        ll = apply_filt(build_left(enc_df, needed),
                        top_k_per_dest, top_k_total, key=0)
        rn, ln = normalise(rl, ll, scale_by_emissions=scale_by_emissions)
        print(f"  Panel '{name}': {len(rl)} right links, {len(ll)} left links")
        panels.append((rn, ln, title))

    # ── Figure layout ───────────────────────────────────────────────────────
    n = len(panels)
    if n == 1:
        fig, ax = plt.subplots(1, 1, figsize=(16, 7.5))
        axes = [ax]
        panels[0] = (panels[0][0], panels[0][1],
                      "Total emissions gradient flow")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(28, 7.5))
        axes = list(axes)

    for ax, (rl, ll, ttl) in zip(axes, panels):
        draw_panel(ax, rl, ll, ttl)

    plt.tight_layout(pad=1.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    sfx = "" if show_uncertainty else "_emissions_only"
    for fmt in ["pdf", "png"]:
        p = output_dir / f"fig_gradient_sankey{sfx}.{fmt}"
        plt.savefig(p, bbox_inches="tight", dpi=300,
                    facecolor="white", edgecolor="none")
        print(f"Saved: {p}")
    plt.close()


def main():
    """Entry point — reads module-level configuration constants."""
    create_sankey_figure(
        top_k_per_dest=TOP_K_PER_DEST,
        top_k_total=TOP_K_TOTAL,
        show_context=SHOW_CONTEXT,
        show_uncertainty=SHOW_UNCERTAINTY,
        scale_by_emissions=SCALE_SECTORS_BY_EMISSIONS,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()