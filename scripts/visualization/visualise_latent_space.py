"""
Latent space visualization via dimensionality reduction (UMAP / t-SNE).

This script extracts the VAE latent representations for all country–year
observations and produces a two-panel figure comparing the structure of
the original high-dimensional input space with the learned latent space.
Country trajectories are shown as lines with flag overlays at the first
observed year.

The figure is intended for inclusion in a Nature-format publication and
follows their style guidelines (vector PDF, minimal chartjunk, ≤ 180 mm
width for two-column figures, legible at print size).

Prerequisites:
    - Trained VAE model  (data/pytorch_models/vae_model.pth)
    - Cached dataset     (data/pytorch_datasets/unified_dataset.pkl)

Usage:
    python -m scripts.analysis.visualise_latent_space [--no-flags]

Outputs:
    - outputs/figures/supplementary/latent_space_umap.pdf
    - outputs/figures/supplementary/latent_space_tsne.pdf

Reference:
    Section 4.2.1 "Variational Autoencoder" in the paper (Figure 3).
"""

from __future__ import annotations

import argparse
import random
from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageOps
from scipy.spatial import procrustes
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

from config.data.output_configs import output_configs
from scripts.elements.datasets import DatasetUnified
from scripts.elements.models import Decoder, Encoder, VAEModel
from scripts.utils import load_config, load_dataset, save_dataset

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Nature-style matplotlib defaults
# ---------------------------------------------------------------------------
# Nature figures: max 180 mm (two-column), 88 mm (single-column).
# Font: sans-serif, 5–7 pt for labels, 8 pt for panel titles.
# All text must be legible after reduction to final column width.
NATURE_RC = {
    # --- Font ---
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    # --- Axes ---
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "axes.labelsize": 7,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # --- Ticks ---
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    # --- Lines ---
    "lines.linewidth": 0.6,
    "lines.markersize": 2,
    # --- Legend ---
    "legend.fontsize": 5.5,
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 0.8,
    # --- Figure ---
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # --- PDF ---
    "pdf.fonttype": 42,  # TrueType (editable text in Illustrator)
    "ps.fonttype": 42,
}
mpl.rcParams.update(NATURE_RC)

# ---------------------------------------------------------------------------
# Country colour palette  (muted, colourblind-friendly where possible)
# ---------------------------------------------------------------------------
COUNTRY_COLOURS: dict[str, str] = {
    "AT": "#C8102E",
    "BE": "#FFB81C",
    "BG": "#00966E",
    "HR": "#003DA5",
    "CY": "#D47600",
    "CZ": "#1D428A",
    "DK": "#C8102E",
    "EE": "#0072CE",
    "EL": "#0D5EAF",
    "FI": "#003580",
    "FR": "#0055A4",
    "DE": "#333333",
    "HU": "#477050",
    "IE": "#169B62",
    "IT": "#008C45",
    "LV": "#9E3039",
    "LT": "#FDB913",
    "LU": "#00A1DE",
    "MT": "#CF2030",
    "NL": "#21468B",
    "PL": "#DC143C",
    "PT": "#006600",
    "RO": "#002B7F",
    "SK": "#0B4EA2",
    "SI": "#005DA5",
    "ES": "#AA151B",
    "SE": "#006AA7",
}

EU27_COUNTRIES = sorted(COUNTRY_COLOURS.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def get_flag_image(country_code: str) -> Image.Image:
    """
    Downloads a small flag PNG for a given ISO-3166 alpha-2 country code.

    Args:
        country_code: Two-letter code (e.g. "DE").  Greek "EL" is
            automatically mapped to "GR" for the flag CDN.

    Returns:
        PIL Image with a thin black border.
    """
    code = "gr" if country_code == "EL" else country_code.lower()
    url = f"https://flagcdn.com/w80/{code}.png"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return ImageOps.expand(img, border=1, fill="black")


def extract_latent_means(
    vae_model: VAEModel,
    dataset: DatasetUnified,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Passes every sample through the encoder and returns latent means.

    Args:
        vae_model: Trained VAE in eval mode (on CUDA).
        dataset: The unified dataset whose ``input_df`` is already on CUDA.

    Returns:
        Tuple of (latent_means array [N, latent_dim], keys DataFrame [geo, year]).
    """
    vae_model.eval()
    means = []
    with torch.inference_mode():
        for i in range(len(dataset)):
            x = dataset.input_df[i].unsqueeze(0)
            mean, _ = vae_model.encoder(x)
            means.append(mean.squeeze(0).cpu().numpy())
    return np.stack(means), dataset.keys.copy()


def compute_reduction(
    data: np.ndarray,
    method: str = "umap",
    **kwargs,
) -> np.ndarray:
    """
    Reduces *data* to 2-D via UMAP or t-SNE.

    Args:
        data: Array of shape (N, D).
        method: ``"umap"`` or ``"tsne"``.
        **kwargs: Forwarded to the reducer constructor.

    Returns:
        Array of shape (N, 2).
    """
    if method == "umap":
        import umap

        reducer = umap.UMAP(
            n_components=2,
            random_state=SEED,
            n_neighbors=kwargs.get("n_neighbors", 50),
        )
    elif method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=2,
            random_state=SEED,
            early_exaggeration=kwargs.get("early_exaggeration", 50),
        )
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'umap' or 'tsne'.")

    return reducer.fit_transform(data)


# ═══════════════════════════════════════════════════════════════════════════
# Similarity metrics
# ═══════════════════════════════════════════════════════════════════════════


def rv_coefficient(X: np.ndarray, Y: np.ndarray) -> float:
    """
    RV coefficient — a multivariate generalisation of R².

    Measures the similarity between two sets of points after removing mean.
    Value of 1 indicates identical configuration (up to rotation/scaling).
    """
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    Sx = Xc.T @ Xc
    Sy = Yc.T @ Yc
    return float(np.trace(Sx @ Sy) / np.sqrt(np.trace(Sx @ Sx) * np.trace(Sy @ Sy)))


def distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Distance correlation (Székely et al., 2007).

    Captures both linear and non-linear association between two
    multivariate samples.  Returns 0 only under full independence.
    """
    def _centred(A: np.ndarray) -> np.ndarray:
        D = pairwise_distances(A)
        return D - D.mean(axis=1, keepdims=True) - D.mean(axis=0, keepdims=True) + D.mean()

    A, B = _centred(X), _centred(Y)
    dCov = np.sqrt(np.mean(A * B))
    dVarX = np.sqrt(np.mean(A * A))
    dVarY = np.sqrt(np.mean(B * B))
    return float(dCov / np.sqrt(dVarX * dVarY))


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════


def _plot_panel(
    ax: mpl.axes.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    method_label: str,
    show_flags: bool = True,
) -> None:
    """
    Draws a single dimensionality-reduction panel.

    Each country is rendered as a trajectory (line connecting successive
    years) with small scatter dots.  A flag icon is placed at the earliest
    year for each country.

    Args:
        ax: Matplotlib Axes to draw on.
        df: DataFrame with columns [Country, Year, x_col, y_col].
        x_col: Column name for the first component.
        y_col: Column name for the second component.
        title: Panel title.
        method_label: Display name of the reduction method (e.g. "UMAP").
        show_flags: If True, download and overlay flag images.
    """
    plot_df = df[df["Year"] < 2023].copy()

    # --- Country trajectories ---
    for country, grp in plot_df.groupby("Country"):
        grp = grp.sort_values("Year")
        colour = COUNTRY_COLOURS.get(country, "#888888")

        ax.plot(
            grp[x_col], grp[y_col],
            color=colour, alpha=0.50, zorder=1, linewidth=0.6,
        )
        ax.scatter(
            grp[x_col], grp[y_col],
            color=colour, s=4, alpha=0.75, zorder=2, linewidths=0,
        )

    # --- Flag overlays at earliest year ---
    if show_flags:
        first = plot_df.sort_values("Year").drop_duplicates("Country", keep="first")
        for _, row in first.iterrows():
            try:
                flag = get_flag_image(row["Country"])
                imagebox = OffsetImage(flag, zoom=0.15)
                ab = AnnotationBbox(
                    imagebox,
                    (row[x_col], row[y_col]),
                    frameon=False,
                    zorder=3,
                )
                ax.add_artist(ab)
            except Exception as exc:
                print(f"  ⚠ Flag download failed for {row['Country']}: {exc}")

    # --- Axis formatting ---
    ax.set_title(title, pad=6)
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    ax.set_xticks([])
    ax.set_yticks([])


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def _generate_figure(
    method: str,
    input_data: np.ndarray,
    latent_means: np.ndarray,
    keys: pd.DataFrame,
    output_dir: Path,
    show_flags: bool = True,
) -> None:
    """
    Run one dimensionality-reduction method and save the two-panel PDF.

    Args:
        method: ``"umap"`` or ``"tsne"``.
        input_data: Raw input features, shape (N, D_input).
        latent_means: Latent means, shape (N, D_latent).
        keys: DataFrame with geo / year columns.
        output_dir: Directory for the output PDF.
        show_flags: Whether to overlay flag images.
    """
    method_label = "UMAP" if method == "umap" else "t-SNE"
    output_pdf = output_dir / f"latent_space_{method}.pdf"

    # --- Dimensionality reduction -----------------------------------------
    print(f"\nRunning {method_label} on input space...")
    proj_input = compute_reduction(input_data, method=method)

    print(f"Running {method_label} on latent space...")
    if method == "umap":
        proj_latent = compute_reduction(latent_means, method=method, n_neighbors=50)
    else:
        proj_latent = compute_reduction(
            latent_means, method=method, early_exaggeration=24,
        )

    # --- Assemble DataFrame -----------------------------------------------
    df = pd.DataFrame({
        "Country": keys.iloc[:, 0].values,
        "Year": keys.iloc[:, 1].astype(int).values,
        "input_1": proj_input[:, 0],
        "input_2": proj_input[:, 1],
        "latent_1": proj_latent[:, 0],
        "latent_2": proj_latent[:, 1],
    })

    # --- Figure — two-panel, Nature two-column width (180 mm ≈ 7.09 in) --
    fig_width_in = 7.09
    fig_height_in = 3.0
    fig, axes = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in))

    _plot_panel(
        axes[0], df, "input_1", "input_2",
        title=f"{method_label} of Original Input Space",
        method_label=method_label,
        show_flags=show_flags,
    )
    _plot_panel(
        axes[1], df, "latent_1", "latent_2",
        title=f"{method_label} of VAE Latent Space",
        method_label=method_label,
        show_flags=show_flags,
    )

    for ax, label in zip(axes, ["a", "b"]):
        ax.text(
            -0.02, 1.05, f"({label})",
            transform=ax.transAxes,
            fontsize=9, fontweight="bold",
            va="bottom", ha="right",
        )

    plt.subplots_adjust(wspace=0.25)
    fig.savefig(output_pdf)
    print(f"Figure saved → {output_pdf}")
    plt.close(fig)

    # --- Similarity metrics -----------------------------------------------
    X1 = StandardScaler().fit_transform(df[["input_1", "input_2"]].values)
    X2 = StandardScaler().fit_transform(df[["latent_1", "latent_2"]].values)

    _, _, disparity = procrustes(X1, X2)
    rv = rv_coefficient(X1, X2)
    dcor = distance_correlation(X1, X2)

    print(f"\n— Similarity Metrics [{method_label}] —")
    print(f"  Procrustes disparity : {disparity:.4f}")
    print(f"  RV coefficient       : {rv:.4f}")
    print(f"  Distance correlation : {dcor:.4f}")


def main() -> None:
    """
    Entry point: extract latent means, then produce two-panel PDFs for
    both UMAP and t-SNE dimensionality reductions.
    """
    parser = argparse.ArgumentParser(
        description="Visualise VAE latent space via UMAP and t-SNE.",
    )
    parser.add_argument(
        "--no-flags",
        action="store_true",
        help="Skip flag downloads (useful offline).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    vae_config_path = "config/models/vae_config.yaml"
    vae_model_path = Path("data/pytorch_models/vae_model.pth")
    dataset_path = Path("data/pytorch_datasets/unified_dataset.pkl")
    variable_file = Path("config/data/variable_selection.txt")
    output_dir = Path("outputs/figures/supplementary")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    with open(variable_file) as f:
        nested_variables = [line.strip() for line in f if line.strip()]

    if dataset_path.exists():
        print(f"Loading cached dataset from {dataset_path}")
        full_dataset = load_dataset(dataset_path)
    else:
        print("Creating dataset (no cache found)...")
        full_dataset = DatasetUnified(
            path_csvs="data/full_timeseries/",
            output_configs=output_configs,
            select_years=np.arange(2010, 2023 + 1),
            select_geo=EU27_COUNTRIES,
            nested_variables=nested_variables,
            with_cuda=True,
            scaling_type="normalization",
        )
        save_dataset(full_dataset, dataset_path)

    # ------------------------------------------------------------------
    # Build and load VAE
    # ------------------------------------------------------------------
    config = load_config(vae_config_path)
    input_dim = len(full_dataset.input_variable_names)

    encoder = Encoder(
        input_dim=input_dim,
        latent_dim=config.vae_latent_dim,
        num_blocks=config.vae_num_blocks,
        dim_blocks=config.vae_dim_blocks,
        activation=config.vae_activation,
        normalization=config.vae_normalization,
        dropout=config.vae_dropouts,
        input_dropout=config.vae_input_dropouts,
    ).cuda()

    decoder = Decoder(
        input_dim=input_dim,
        latent_dim=config.vae_latent_dim,
        num_blocks=config.vae_num_blocks,
        dim_blocks=config.vae_dim_blocks,
        activation=config.vae_activation,
        normalization=config.vae_normalization,
        dropout=config.vae_dropouts,
    ).cuda()

    vae_model = VAEModel(encoder, decoder).cuda()
    vae_model.load_state_dict(torch.load(vae_model_path))
    vae_model.eval()
    print(f"VAE loaded — latent dim = {config.vae_latent_dim}")

    # ------------------------------------------------------------------
    # Extract latent means (shared across both methods)
    # ------------------------------------------------------------------
    latent_means, keys = extract_latent_means(vae_model, full_dataset)
    input_data = full_dataset.input_df.cpu().numpy()

    print(f"Input data shape:  {input_data.shape}")
    print(f"Latent means shape: {latent_means.shape}")

    # ------------------------------------------------------------------
    # Generate figures for both methods
    # ------------------------------------------------------------------
    for method in ("umap", "tsne"):
        _generate_figure(
            method=method,
            input_data=input_data,
            latent_means=latent_means,
            keys=keys,
            output_dir=output_dir,
            show_flags=not args.no_flags,
        )

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()