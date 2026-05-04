"""
Visualise historical + projected decoded values for a single input variable
and country.

For monthly/quarterly variables (columns with _1 … _12 or _1 _4 _7 _10
suffixes), all sub-series are plotted as ONE continuous line on a decimal-year
time axis, giving a wiggly series that shows both trend and seasonal pattern.

Usage (command-line):
    python scripts/visualization/plot_decoded_variable.py \
        --variable "Monthly_electricity_statistics:Net Electricity Production:Wind" \
        --country DE

Usage (hardcoded — set CLI_MODE = False):
    Edit VARIABLE and COUNTRY below, then run the script.
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# ── Hardcoded fallback (used when CLI_MODE = False) ──────────────────────────
# =============================================================================

CLI_MODE = False

VARIABLE = "Monthly_oil_price_statistics:Gasoline (unit/litre):Total price:US dollars"
COUNTRY  = "IT"

# =============================================================================
# Paths
# =============================================================================

DATASET_PATH        = Path("data/pytorch_datasets/unified_dataset.pkl")
DECODED_PROJECTIONS = Path("data/projections/decoded_projections.csv")

EU27_COUNTRIES = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "EL", "FI",
    "FR", "DE", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE",
]

COUNTRY_NAMES = {
    "AT": "Austria",    "BE": "Belgium",     "BG": "Bulgaria",
    "HR": "Croatia",    "CY": "Cyprus",      "CZ": "Czechia",
    "DK": "Denmark",    "EE": "Estonia",     "EL": "Greece",
    "FI": "Finland",    "FR": "France",      "DE": "Germany",
    "HU": "Hungary",    "IE": "Ireland",     "IT": "Italy",
    "LV": "Latvia",     "LT": "Lithuania",   "LU": "Luxembourg",
    "MT": "Malta",      "NL": "Netherlands", "PL": "Poland",
    "PT": "Portugal",   "RO": "Romania",     "SK": "Slovakia",
    "SI": "Slovenia",   "ES": "Spain",       "SE": "Sweden",
}

# Last year that is purely historical (projections start at PROJ_START_YEAR)
LAST_HISTORICAL_YEAR = 2023
PROJ_START_YEAR      = 2024

# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    if not CLI_MODE:
        class NS:
            variable = VARIABLE
            country  = COUNTRY
        return NS()

    parser = argparse.ArgumentParser(
        description=(
            "Plot historical + decoded projected values for one variable / country."
        )
    )
    parser.add_argument(
        "--variable", "-v", default=None,
        help=(
            "Base variable name, e.g. "
            "'Monthly_electricity_statistics:Net Electricity Production:Wind' "
            "or 'gdp_quarterly:MillionEUR'. "
            "Do NOT include the month/quarter suffix (_1, _4, …)."
        ),
    )
    parser.add_argument(
        "--country", "-c", default=None,
        help="ISO-2 country code, e.g. DE, FR, IT.",
    )
    args = parser.parse_args()

    if args.variable is None:
        args.variable = VARIABLE
    if args.country is None:
        args.country = COUNTRY

    return args


# =============================================================================
# Dataset loading
# =============================================================================

def load_dataset_obj():
    repo_root = str(Path(__file__).resolve().parents[2])
    added = repo_root not in sys.path
    if added:
        sys.path.insert(0, repo_root)
    try:
        with open(DATASET_PATH, "rb") as f:
            return pickle.load(f)
    finally:
        if added:
            sys.path.remove(repo_root)


# =============================================================================
# Column matching
# =============================================================================

def find_matching_columns(base_name: str, all_columns: list) -> list:
    """
    Return all column names whose base (stripped of trailing _<digits>)
    matches base_name, sorted by their numeric suffix.

    Examples
    --------
    base = "Monthly_electricity_statistics:Net Electricity Production:Wind"
    → [...:Wind_1, ...:Wind_2, ..., ...:Wind_12]

    base = "gdp_quarterly:MillionEUR"
    → [gdp_quarterly:MillionEUR_1, ..._4, ..._7, ..._10]

    base = "heat_pumps:GWH"   (annual)
    → [heat_pumps:GWH]
    """
    suffix_re = re.compile(r"^(.+?)(?:_(\d+))?$")

    matched = []
    for col in all_columns:
        m = suffix_re.match(col)
        if m is None:
            continue
        col_base   = m.group(1)
        col_suffix = m.group(2)
        if col_base == base_name:
            matched.append((col, int(col_suffix) if col_suffix else -1))

    if not matched:
        return []

    matched.sort(key=lambda x: x[1])
    return [col for col, _ in matched]


def suffix_to_decimal(suffix: int) -> float:
    """
    Convert a month number (1–12) to a decimal year offset in [0, 1).
    Annual variables (suffix == -1) map to 0.0.

    Month  1 → 0.000
    Month  4 → 0.250
    Month  7 → 0.500
    Month 12 → 0.917
    """
    if suffix == -1:
        return 0.0
    return (suffix - 1) / 12.0


# =============================================================================
# Historical data extraction
# =============================================================================

def extract_historical(dataset, country: str, columns: list) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with one row per (year, sub-column) observation,
    sorted by decimal_year so all months/quarters form one continuous series.
    """
    suffix_re = re.compile(r"^(.+?)(?:_(\d+))?$")

    rows = []
    for col in columns:
        m          = suffix_re.match(col)
        suffix     = int(m.group(2)) if m.group(2) else -1
        dec_offset = suffix_to_decimal(suffix)
        col_idx    = list(dataset.input_variable_names).index(col)

        for i, (_, row) in enumerate(dataset.keys.iterrows()):
            if row["geo"] != country:
                continue
            year       = int(row["year"])
            scaled_val = float(dataset.input_df[i, col_idx].cpu())

            params = dataset.precomputed_scaling_params.get(col)
            if params is not None and dataset.scaling_type == "normalization":
                phys_val = scaled_val * params["std"] + params["mean"]
            elif params is not None and dataset.scaling_type == "maxmin":
                phys_val = (
                    scaled_val * (params["max"] - params["min"]) + params["min"]
                )
            else:
                phys_val = scaled_val

            rows.append({
                "decimal_year":   year + dec_offset,
                "year":           year,
                "suffix":         suffix,
                "col":            col,
                "value_scaled":   scaled_val,
                "value_physical": phys_val,
            })

    return pd.DataFrame(rows).sort_values("decimal_year").reset_index(drop=True)


# =============================================================================
# Projected data extraction
# =============================================================================

def extract_projected(country: str, columns: list) -> pd.DataFrame:
    """
    Reads decoded_projections.csv and returns mean + 5/95 percentile MC bands
    for the requested columns and country, sorted by decimal_year so all
    months/quarters form one continuous series.
    """
    if not DECODED_PROJECTIONS.exists():
        print(
            f"[WARNING] {DECODED_PROJECTIONS} not found — "
            "projected data will not be shown.\n"
            "Run scripts/inference/generate_decoded_projections.py first."
        )
        return pd.DataFrame()

    suffix_re = re.compile(r"^(.+?)(?:_(\d+))?$")

    # Check which columns are actually present in the CSV
    header = pd.read_csv(DECODED_PROJECTIONS, nrows=0).columns.tolist()
    missing = [c for c in columns if c not in header]
    if missing:
        print(f"[WARNING] These columns not found in decoded CSV: {missing}")
        columns = [c for c in columns if c in header]
    if not columns:
        return pd.DataFrame()

    df = pd.read_csv(
        DECODED_PROJECTIONS,
        usecols=["mc_sample", "geo", "year"] + columns,
    )
    df = df[df["geo"] == country].copy()

    if df.empty:
        print(f"[WARNING] No projected data found for country {country}.")
        return pd.DataFrame()

    rows = []
    for col in columns:
        m          = suffix_re.match(col)
        suffix     = int(m.group(2)) if m.group(2) else -1
        dec_offset = suffix_to_decimal(suffix)

        for year, group_vals in df.groupby("year")[col]:
            vals = group_vals.dropna().values
            if len(vals) == 0:
                continue
            rows.append({
                "decimal_year": year + dec_offset,
                "year":         year,
                "suffix":       suffix,
                "col":          col,
                "mean":         np.mean(vals),
                "p05":          np.percentile(vals, 5),
                "p95":          np.percentile(vals, 95),
            })

    return pd.DataFrame(rows).sort_values("decimal_year").reset_index(drop=True)


# =============================================================================
# Plot
# =============================================================================

def make_plot(
    base_name: str,
    country: str,
    hist_df: pd.DataFrame,
    proj_df: pd.DataFrame,
) -> None:

    country_label  = COUNTRY_NAMES.get(country, country)
    color          = "#2980b9"
    separator_year = float(PROJ_START_YEAR)   # vertical line at 2024.0

    fig, ax = plt.subplots(figsize=(13, 5))

    # ── Historical — one continuous line ──────────────────────────────────
    if not hist_df.empty:
        ax.plot(
            hist_df["decimal_year"],
            hist_df["value_physical"],
            color=color, linewidth=1.6, zorder=3, label="Historical",
        )
        ax.scatter(
            hist_df["decimal_year"],
            hist_df["value_physical"],
            color=color, s=14, zorder=4, linewidths=0,
        )

    # ── Projected — one continuous mean line + uncertainty band ───────────
    if not proj_df.empty:
        ax.axvline(
            separator_year,
            color="#555555", linestyle="--", linewidth=1.0, alpha=0.7, zorder=2,
        )
        ax.axvspan(
            separator_year, proj_df["decimal_year"].max() + 0.1,
            alpha=0.04, color="#000000", zorder=0,
        )
        ax.plot(
            proj_df["decimal_year"],
            proj_df["mean"],
            color=color, linewidth=1.6, zorder=3, label="Projected (mean)",
        )
        ax.fill_between(
            proj_df["decimal_year"],
            proj_df["p05"],
            proj_df["p95"],
            color=color, alpha=0.20, zorder=1, label="90 % MC band",
        )

    ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Value (physical units)", fontsize=11)
    ax.set_title(
        f"{base_name}\n{country_label} ({country})",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


# =============================================================================
# Entry point
# =============================================================================

def main():
    args      = parse_args()
    base_name = args.variable
    country   = args.country.upper()

    if country not in EU27_COUNTRIES:
        sys.exit(
            f"[ERROR] '{country}' is not a recognised EU27 country code.\n"
            f"Valid codes: {', '.join(sorted(EU27_COUNTRIES))}"
        )

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset_obj()

    input_variable_names = list(dataset.input_variable_names)

    columns = find_matching_columns(base_name, input_variable_names)
    if not columns:
        candidates = sorted({
            re.sub(r"_\d+$", "", c) for c in input_variable_names
        })
        close = [c for c in candidates if base_name.lower() in c.lower()]
        hint = (
            "\nDid you mean one of these?\n  " + "\n  ".join(close[:10])
            if close else
            "\nAvailable base names (first 20):\n  " +
            "\n  ".join(candidates[:20])
        )
        sys.exit(
            f"[ERROR] No columns found matching base variable:\n  '{base_name}'"
            + hint
        )

    print(f"Found {len(columns)} column(s) for '{base_name}':")
    for c in columns:
        print(f"  {c}")

    # ── Historical ────────────────────────────────────────────────────────
    print(f"\nExtracting historical data for {country}...")
    hist_df = extract_historical(dataset, country, columns)
    print(f"  {len(hist_df)} historical data points.")

    # ── Projected ─────────────────────────────────────────────────────────
    print("Loading projected (decoded) data...")
    proj_df = extract_projected(country, columns)
    print(f"  {len(proj_df)} projected summary rows.")

    # ── Plot ──────────────────────────────────────────────────────────────
    make_plot(base_name, country, hist_df, proj_df)


if __name__ == "__main__":
    main()