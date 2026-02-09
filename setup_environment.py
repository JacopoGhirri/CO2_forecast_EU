#!/usr/bin/env python3
"""
Environment Setup Script for EU Emissions Forecast Replication.

This script:
1. Checks Python version compatibility
2. Installs required packages
3. Verifies CUDA/GPU availability
4. Creates necessary directories
5. Validates the repository structure

Usage:
    python setup_environment.py
    python setup_environment.py --check-only  # Check without installing
"""

import argparse
import subprocess
import sys
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent

MIN_PYTHON_VERSION = (3, 10)

# Core dependencies
CORE_PACKAGES = [
    "torch>=2.0",
    "numpy>=1.24",
    "pandas>=2.0",
    "pyyaml>=6.0",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "tqdm>=4.65",
]

# Optional dependencies
OPTIONAL_PACKAGES = {
    "data_download": ["gdown>=4.7"],
    "sensitivity": ["SALib>=1.4"],
    "excel": ["openpyxl>=3.1"],
}

# Required directories
REQUIRED_DIRS = [
    "data/full_timeseries",
    "data/full_timeseries/projections",
    "data/external",
    "data/pytorch_datasets",
    "data/pytorch_models",
    "data/projections",
    "data/sensitivity",  # Created by sensitivity analysis scripts
    "outputs/figures",
    "outputs/tables",
    "config/data",
    "config/models",
]

# Required files (that should exist in repo)
REQUIRED_FILES = [
    "config/data/output_configs.py",
    "config/data/variable_selection.txt",
    "config/models/vae_config.yaml",
    "config/models/co2_predictor_config.yaml",
    "config/models/latent_forecaster_config.yaml",
    "scripts/elements/datasets.py",
    "scripts/elements/models.py",
    "scripts/utils.py",
]


# =============================================================================
# Helper Functions
# =============================================================================


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info[:2]
    if version >= MIN_PYTHON_VERSION:
        print(f"✓ Python version: {sys.version.split()[0]}")
        return True
    else:
        print(f"✗ Python version {sys.version.split()[0]} is too old")
        print(f"  Minimum required: {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}")
        return False


def check_cuda() -> dict:
    """Check CUDA and GPU availability."""
    cuda_info = {
        "available": False,
        "version": None,
        "devices": [],
    }

    try:
        import torch

        cuda_info["available"] = torch.cuda.is_available()
        if cuda_info["available"]:
            cuda_info["version"] = torch.version.cuda
            cuda_info["devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    if cuda_info["available"]:
        print(f"✓ CUDA available: {cuda_info['version']}")
        for i, device in enumerate(cuda_info["devices"]):
            print(f"  GPU {i}: {device}")
    else:
        print("○ CUDA not available (training will be slower on CPU)")

    return cuda_info


def install_packages(packages: list, upgrade: bool = False) -> bool:
    """Install packages using pip."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(packages)

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e.stderr.decode()}")
        return False


def check_package_installed(package: str) -> bool:
    """Check if a package is installed."""
    package_name = package.split(">=")[0].split("==")[0].split("<")[0]
    try:
        __import__(package_name.replace("-", "_"))
        return True
    except ImportError:
        return False


def create_directories(repo_root: Path) -> int:
    """Create required directories."""
    created = 0
    for dir_path in REQUIRED_DIRS:
        full_path = repo_root / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {dir_path}")
            created += 1
    return created


def check_required_files(repo_root: Path) -> list:
    """Check for required repository files."""
    missing = []
    for file_path in REQUIRED_FILES:
        full_path = repo_root / file_path
        if not full_path.exists():
            missing.append(file_path)
    return missing


# =============================================================================
# Main Setup
# =============================================================================


def setup_environment(check_only: bool = False, install_optional: bool = True) -> bool:
    """
    Set up the replication environment.

    Args:
        check_only: If True, only check without installing
        install_optional: If True, install optional packages

    Returns:
        True if setup successful
    """
    print("=" * 70)
    print("EU EMISSIONS FORECAST - ENVIRONMENT SETUP")
    print("=" * 70)
    print()

    all_ok = True

    # 1. Check Python version
    print("1. Checking Python version...")
    if not check_python_version():
        return False
    print()

    # 2. Check/install core packages
    print("2. Checking core packages...")
    missing_core = [p for p in CORE_PACKAGES if not check_package_installed(p)]

    if missing_core:
        if check_only:
            print(f"  Missing packages: {', '.join(missing_core)}")
            all_ok = False
        else:
            print(f"  Installing: {', '.join(missing_core)}")
            if not install_packages(missing_core):
                print("  ✗ Failed to install some packages")
                all_ok = False
            else:
                print("  ✓ Core packages installed")
    else:
        print("  ✓ All core packages installed")
    print()

    # 3. Check/install optional packages
    print("3. Checking optional packages...")
    for group, packages in OPTIONAL_PACKAGES.items():
        missing = [p for p in packages if not check_package_installed(p)]
        if missing:
            if check_only or not install_optional:
                print(f"  {group}: {', '.join(missing)} (not installed)")
            else:
                print(f"  Installing {group}: {', '.join(missing)}")
                install_packages(missing)
        else:
            print(f"  ✓ {group}: all installed")
    print()

    # 4. Check CUDA
    print("4. Checking CUDA/GPU...")
    check_cuda()
    print()

    # 5. Create directories
    print("5. Creating directories...")
    if check_only:
        missing_dirs = [d for d in REQUIRED_DIRS if not (REPO_ROOT / d).exists()]
        if missing_dirs:
            print(f"  Would create {len(missing_dirs)} directories")
        else:
            print("  ✓ All directories exist")
    else:
        created = create_directories(REPO_ROOT)
        if created > 0:
            print(f"  Created {created} directories")
        else:
            print("  ✓ All directories exist")
    print()

    # 6. Check required files
    print("6. Checking repository files...")
    missing_files = check_required_files(REPO_ROOT)
    if missing_files:
        print(f"  ⚠ Missing {len(missing_files)} required files:")
        for f in missing_files[:5]:
            print(f"    - {f}")
        if len(missing_files) > 5:
            print(f"    ... and {len(missing_files) - 5} more")
        all_ok = False
    else:
        print("  ✓ All required files present")
    print()

    # Summary
    print("=" * 70)
    if all_ok:
        print("✓ Environment setup complete!")
        print()
        print("Next steps:")
        print("  1. Edit replication/data_manifest.yaml with Google Drive IDs")
        print("  2. Run: python replication/download_data.py")
        print("  3. Run: python replication/run_replication.py")
    else:
        print("⚠ Some issues were found. Please resolve them before proceeding.")
    print("=" * 70)

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Set up environment for EU Emissions Forecast replication"
    )

    parser.add_argument(
        "--check-only",
        "-c",
        action="store_true",
        help="Only check environment without installing packages",
    )

    parser.add_argument(
        "--no-optional",
        action="store_true",
        help="Skip installation of optional packages",
    )

    args = parser.parse_args()

    success = setup_environment(
        check_only=args.check_only, install_optional=not args.no_optional
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
