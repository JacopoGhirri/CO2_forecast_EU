"""
Training energy & CO2 tracker.

Wraps each training script with CodeCarbon and prints a minimal summary
suitable for reporting in the paper.

Prerequisites:
    pip install codecarbon

Usage:
    python scripts/training/track_training_emissions.py
    python scripts/training/track_training_emissions.py --stage vae
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

STAGES = {
    "vae":        ("scripts.training.train_vae",        5000),
    "predictor":  ("scripts.training.train_predictor",  5000),
    "forecaster": ("scripts.training.train_forecaster", 5001),
}

LABELS = {
    "vae":        "VAE",
    "predictor":  "Emission Predictor",
    "forecaster": "Latent Forecaster",
}


def run_stage(stage: str) -> None:
    from codecarbon import EmissionsTracker

    module_path, n_epochs = STAGES[stage]
    label = LABELS[stage]

    tracker = EmissionsTracker(
        project_name=f"eu_emissions_{stage}",
        save_to_file=False,
        log_level="error",
        measure_power_secs=15,
        tracking_mode="process",
    )

    mod = importlib.import_module(module_path)

    tracker.start()
    try:
        mod.main()
    finally:
        tracker.stop()

    energy_wh  = tracker.final_emissions_data.energy_consumed * 1000
    co2_g      = tracker.final_emissions_data.emissions * 1000
    wh_epoch   = energy_wh / n_epochs
    co2_epoch  = co2_g / n_epochs

    print(f"\n{label} ({n_epochs:,} epochs)")
    print(f"  Energy : {energy_wh:.2f} Wh  ({wh_epoch * 1000:.2f} mWh/epoch)")
    print(f"  CO2    : {co2_g:.2f} g CO2eq  ({co2_epoch:.4f} g/epoch)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=list(STAGES.keys()),
        default=list(STAGES.keys()),
    )
    args = parser.parse_args()

    for stage in args.stage:
        run_stage(stage)


if __name__ == "__main__":
    main()