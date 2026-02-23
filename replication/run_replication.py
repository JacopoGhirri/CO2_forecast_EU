#!/usr/bin/env python3
"""
Main Replication Script for EU Emissions Forecast.

This script orchestrates the complete replication pipeline:
1. Data download and verification
2. Model training (VAE → Predictor → Forecaster)
3. Inference (Monte Carlo projections)
4. Sensitivity analysis (Sobol + Perturbation)
5. Cross-validation model comparison (Table 3)
6. Figure generation

Usage:
    python run_replication.py                  # Run full pipeline
    python run_replication.py train            # Run only training stages
    python run_replication.py sobol            # Run only Sobol analysis
    python run_replication.py --skip download  # Skip data download
    python run_replication.py --gpu 0 train    # Train on specific GPU
    python run_replication.py --dry-run full   # Show what would run

Requirements:
    See pyproject.toml for full dependencies
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent

# Pipeline stages with their scripts and descriptions
PIPELINE_STAGES = {
    "download": {
        "script": SCRIPT_DIR / "download_data.py",
        "description": "Download data from Google Drive",
        "args": [],
        "module": False,
    },
    "verify": {
        "script": SCRIPT_DIR / "download_data.py",
        "description": "Verify all required data files exist",
        "args": ["--verify"],
        "module": False,
    },
    "train_vae": {
        "script": "scripts.training.train_vae",
        "description": "Train Variational Autoencoder",
        "args": [],
        "module": True,
        "gpu_required": True,
    },
    "train_predictor": {
        "script": "scripts.training.train_predictor",
        "description": "Train Emission Predictor",
        "args": [],
        "module": True,
        "gpu_required": True,
    },
    "train_forecaster": {
        "script": "scripts.training.train_forecaster",
        "description": "Train Latent Forecaster",
        "args": [],
        "module": True,
        "gpu_required": True,
    },
    "inference": {
        "script": "scripts.inference.generate_projections",
        "description": "Generate Monte Carlo projections",
        "args": [],
        "module": True,
        "gpu_required": True,
    },
    "sobol": {
        "script": "scripts.analysis.sobol_analysis",
        "description": "Run Sobol sensitivity analysis",
        "args": [],
        "module": True,
        "gpu_required": True,
    },
    "perturbation": {
        "script": "scripts.analysis.perturbation_analysis",
        "description": "Run perturbation sensitivity analysis",
        "args": [],
        "module": True,
        "gpu_required": True,
    },
    "cv_comparison": {
        "script": "scripts.analysis.cross_validation_comparison",
        "description": "Cross-validation comparison of model variants (Table 3)",
        "args": [],
        "module": True,
        "gpu_required": True,
    },
    "fig1": {
        "script": "scripts.visualization.figure_emissions_gap",
        "description": "Generate Figure 1: Emissions Gap",
        "args": [],
        "module": True,
    },
    "fig2": {
        "script": "scripts.visualization.figure_sectoral_changes",
        "description": "Generate Figure 2: Sectoral Changes",
        "args": [],
        "module": True,
    },
    "fig3": {
        "script": "scripts.visualization.figure_attribution_panels",
        "description": "Generate Figure 3: Attribution Panels",
        "args": [],
        "module": True,
    },
    "fig4": {
        "script": "scripts.visualization.figure_sensitivity_heatmap",
        "description": "Generate Figure 4: Sensitivity Heatmap",
        "args": [],
        "module": True,
    },
    "fig_latent": {
        "script": "scripts.visualization.visualise_latent_space",
        "description": "Generate supplementary latent space figures (UMAP + t-SNE)",
        "args": [],
        "module": True,
        "gpu_required": True,
    },
}

# Stage groups for convenience
STAGE_GROUPS = {
    "all": list(PIPELINE_STAGES.keys()),
    "data": ["download", "verify"],
    "train": ["train_vae", "train_predictor", "train_forecaster"],
    "inference": ["inference"],
    "analysis": ["sobol", "perturbation", "cv_comparison"],
    "figures": ["fig1", "fig2", "fig3", "fig4", "fig_latent"],
    "full": [
        "download",
        "verify",
        "train_vae",
        "train_predictor",
        "train_forecaster",
        "inference",
        "sobol",
        "perturbation",
        "cv_comparison",
        "fig1",
        "fig2",
        "fig3",
        "fig4",
        "fig_latent",
    ],
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, level: str = "INFO"):
    """Print a log message with timestamp."""
    print(f"[{get_timestamp()}] [{level}] {message}")


def check_gpu_availability() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def run_stage(
    stage_name: str, stage_config: dict, gpu_id: int = None, dry_run: bool = False
) -> bool:
    """
    Run a single pipeline stage.

    Args:
        stage_name: Name of the stage
        stage_config: Stage configuration dictionary
        gpu_id: GPU ID to use (None for CPU or default)
        dry_run: If True, only print what would be executed

    Returns:
        True if successful, False otherwise
    """
    log(f"Starting stage: {stage_name}")
    log(f"  Description: {stage_config['description']}")

    # Build command
    if stage_config.get("module", False):
        cmd = [sys.executable, "-m", stage_config["script"]]
    else:
        cmd = [sys.executable, str(stage_config["script"])]

    cmd.extend(stage_config.get("args", []))

    # Set environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log(f"  Command: {' '.join(cmd)}")

    if dry_run:
        log("  [DRY RUN] Would execute command")
        return True

    # Execute
    start_time = time.time()

    try:
        subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            check=True,
        )
        elapsed = time.time() - start_time
        log(f"  ✓ Completed in {elapsed:.1f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        log(
            f"  ✗ Failed after {elapsed:.1f} seconds (exit code {e.returncode})",
            "ERROR",
        )
        return False

    except Exception as e:
        log(f"  ✗ Error: {e}", "ERROR")
        return False


def expand_stages(stage_specs: list) -> list:
    """
    Expand stage specifications into individual stages.

    Args:
        stage_specs: List of stage names or group names

    Returns:
        Ordered list of individual stage names
    """
    stages = []

    for spec in stage_specs:
        if spec in STAGE_GROUPS:
            # Expand group
            for stage in STAGE_GROUPS[spec]:
                if stage not in stages:
                    stages.append(stage)
        elif spec in PIPELINE_STAGES:
            if spec not in stages:
                stages.append(spec)
        else:
            log(f"Unknown stage or group: {spec}", "WARNING")

    return stages


# =============================================================================
# Main Pipeline
# =============================================================================


def run_pipeline(
    stages: list,
    gpu_id: int = None,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> dict:
    """
    Run the replication pipeline.

    Args:
        stages: List of stages to run
        gpu_id: GPU ID to use
        dry_run: If True, only print what would be executed
        continue_on_error: If True, continue to next stage on failure

    Returns:
        Dictionary with execution statistics
    """
    stats = {
        "total": len(stages),
        "completed": 0,
        "failed": 0,
        "skipped": 0,
    }

    # Check GPU if needed
    gpu_available = check_gpu_availability()
    gpu_stages = [
        s for s in stages if PIPELINE_STAGES.get(s, {}).get("gpu_required", False)
    ]

    if gpu_stages and not gpu_available:
        log("WARNING: Some stages require GPU but CUDA is not available", "WARNING")
        log(f"  Affected stages: {', '.join(gpu_stages)}", "WARNING")
        if not continue_on_error:
            log("Use --continue-on-error to run CPU-only stages", "WARNING")

    log(f"\n{'=' * 70}")
    log("EU EMISSIONS FORECAST REPLICATION PIPELINE")
    log(f"{'=' * 70}")
    log(f"Stages to run: {', '.join(stages)}")
    log(
        f"GPU: {'cuda:' + str(gpu_id) if gpu_id is not None else ('available' if gpu_available else 'not available')}"
    )
    log(f"Dry run: {dry_run}")
    log(f"{'=' * 70}\n")

    # Execute stages
    failed_stages = []

    for i, stage_name in enumerate(stages, 1):
        log(f"\n[{i}/{len(stages)}] Stage: {stage_name}")
        log("-" * 50)

        config = PIPELINE_STAGES.get(stage_name)
        if not config:
            log(f"Unknown stage: {stage_name}", "ERROR")
            stats["failed"] += 1
            failed_stages.append(stage_name)
            continue

        # Check GPU requirement
        if config.get("gpu_required", False) and not gpu_available:
            log(f"Skipping {stage_name}: requires GPU", "WARNING")
            stats["skipped"] += 1
            continue

        success = run_stage(stage_name, config, gpu_id=gpu_id, dry_run=dry_run)

        if success:
            stats["completed"] += 1
        else:
            stats["failed"] += 1
            failed_stages.append(stage_name)

            if not continue_on_error:
                log(
                    f"\nPipeline stopped due to failure in stage: {stage_name}", "ERROR"
                )
                break

    # Print summary
    log(f"\n{'=' * 70}")
    log("PIPELINE SUMMARY")
    log(f"{'=' * 70}")
    log(f"Total stages: {stats['total']}")
    log(f"  Completed: {stats['completed']}")
    log(f"  Failed: {stats['failed']}")
    log(f"  Skipped: {stats['skipped']}")

    if failed_stages:
        log(f"\nFailed stages: {', '.join(failed_stages)}", "ERROR")

    return stats


# =============================================================================
# Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run EU Emissions Forecast replication pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stage Groups:
    all        - All stages
    data       - download, verify
    train      - train_vae, train_predictor, train_forecaster
    inference  - inference
    analysis   - sobol, perturbation, cv_comparison
    figures    - fig1, fig2, fig3, fig4, fig_latent
    full       - Complete pipeline in order

Individual Stages:
    download         - Download data from Google Drive
    verify           - Verify data files exist
    train_vae        - Train VAE model
    train_predictor  - Train emission predictor
    train_forecaster - Train latent forecaster
    inference        - Generate Monte Carlo projections
    sobol            - Sobol sensitivity analysis
    perturbation     - Perturbation sensitivity analysis
    cv_comparison    - Cross-validation model comparison (Table 3)
    fig1             - Generate Figure 1
    fig2             - Generate Figure 2
    fig3             - Generate Figure 3
    fig4             - Generate Figure 4
    fig_latent       - Generate supplementary latent space figures (UMAP + t-SNE)

Examples:
    python run_replication.py                      # Run full pipeline
    python run_replication.py train                # Run training stages only
    python run_replication.py figures              # Generate all figures
    python run_replication.py fig1 fig2            # Generate specific figures
    python run_replication.py fig_latent           # Generate latent space figures only
    python run_replication.py cv_comparison        # Run Table 3 cross-validation only
    python run_replication.py --skip download      # Skip data download
    python run_replication.py --gpu 0 train        # Train on GPU 0
    python run_replication.py --dry-run full       # Show what would run
        """,
    )

    parser.add_argument(
        "stages",
        nargs="*",
        default=["full"],
        help="Stages or stage groups to run (default: full)",
    )

    parser.add_argument("--skip", nargs="+", default=[], help="Stages to skip")

    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID to use (default: use all available)",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be executed without running",
    )

    parser.add_argument(
        "--continue-on-error",
        "-c",
        action="store_true",
        help="Continue to next stage if a stage fails",
    )

    parser.add_argument(
        "--list", "-l", action="store_true", help="List available stages and exit"
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("\nAvailable Stages:")
        print("-" * 50)
        for name, config in PIPELINE_STAGES.items():
            gpu = " [GPU]" if config.get("gpu_required", False) else ""
            print(f"  {name:<20} {config['description']}{gpu}")

        print("\nStage Groups:")
        print("-" * 50)
        for name, stages in STAGE_GROUPS.items():
            print(f"  {name:<12} {', '.join(stages)}")
        return

    # Expand stage specifications
    stages = expand_stages(args.stages)

    # Remove skipped stages
    if args.skip:
        skip_stages = expand_stages(args.skip)
        stages = [s for s in stages if s not in skip_stages]

    if not stages:
        print("No stages to run.")
        return

    # Run pipeline
    stats = run_pipeline(
        stages,
        gpu_id=args.gpu,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )

    # Exit with error code if any stage failed
    sys.exit(1 if stats["failed"] > 0 else 0)


if __name__ == "__main__":
    main()
