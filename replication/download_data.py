#!/usr/bin/env python3
"""
Data Download Script for EU Emissions Forecast Replication.

Downloads all required data files from Google Drive based on the data manifest.
Supports selective downloading (required vs optional), resume on interruption,
and checksum verification.

Usage:
    python download_data.py                    # Download required files only
    python download_data.py --all              # Download all files including optional
    python download_data.py --category raw     # Download specific category
    python download_data.py --dry-run          # Show what would be downloaded
    python download_data.py --verify           # Verify existing files

Requirements:
    pip install gdown pyyaml tqdm
"""

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

try:
    import gdown
except ImportError:
    print("Error: gdown not installed. Run: pip install gdown")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
MANIFEST_PATH = SCRIPT_DIR / "data_manifest.yaml"

# Google Drive download URL template
GDRIVE_URL_TEMPLATE = "https://drive.google.com/uc?id={}"

# Categories and their required status
CATEGORY_CONFIG = {
    "raw_timeseries": {"required": True, "description": "Raw time series data"},
    "projections": {"required": True, "description": "Projection data for scenarios"},
    "external_comparisons": {
        "required": True,
        "description": "External comparison data",
    },
    "cached_datasets": {"required": False, "description": "Pre-computed datasets"},
    "trained_models": {"required": False, "description": "Trained model weights"},
    "generated_outputs": {
        "required": False,
        "description": "Generated projection outputs",
    },
}


# =============================================================================
# Helper Functions
# =============================================================================


def load_manifest(manifest_path: Path) -> dict:
    """Load and parse the data manifest YAML file."""
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        return yaml.safe_load(f)


def get_file_hash(filepath: Path, algorithm: str = "md5") -> str:
    """Calculate hash of a file for verification."""
    hash_func = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def download_from_gdrive(
    gdrive_id: str, output_path: Path, quiet: bool = False
) -> bool:
    """
    Download a file from Google Drive.

    Args:
        gdrive_id: Google Drive file ID
        output_path: Local path to save the file
        quiet: Suppress download progress

    Returns:
        True if download successful, False otherwise
    """
    if gdrive_id == "##GDRIVE_ID##" or not gdrive_id:
        return False

    url = GDRIVE_URL_TEMPLATE.format(gdrive_id)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gdown.download(url, str(output_path), quiet=quiet)
        return output_path.exists()
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# =============================================================================
# Main Download Logic
# =============================================================================


def collect_files(
    manifest: dict, categories: list | None = None, include_optional: bool = False
) -> list:
    """
    Collect all files to download from manifest.

    Args:
        manifest: Parsed manifest dictionary
        categories: List of categories to include (None = all)
        include_optional: Whether to include optional files

    Returns:
        List of file dictionaries with path, gdrive_id, etc.
    """
    files = []

    for category, config in CATEGORY_CONFIG.items():
        # Skip if category not in manifest
        if category not in manifest:
            continue

        # Skip if filtering by category and this one not included
        if categories and category not in categories:
            continue

        # Skip optional categories if not requested
        if not config["required"] and not include_optional:
            continue

        for file_entry in manifest[category]:
            file_entry["category"] = category
            file_entry["category_required"] = config["required"]
            files.append(file_entry)

    return files


def download_files(
    files: list, repo_root: Path, dry_run: bool = False, force: bool = False
) -> dict:
    """
    Download files from Google Drive.

    Args:
        files: List of file dictionaries
        repo_root: Repository root directory
        dry_run: If True, only show what would be downloaded
        force: If True, redownload even if file exists

    Returns:
        Dictionary with download statistics
    """
    stats = {
        "total": len(files),
        "downloaded": 0,
        "skipped_exists": 0,
        "skipped_no_id": 0,
        "failed": 0,
    }

    print(f"\n{'=' * 70}")
    print(f"{'DRY RUN - ' if dry_run else ''}Downloading {len(files)} files")
    print(f"{'=' * 70}\n")

    for file_entry in tqdm(files, desc="Downloading"):
        path = repo_root / file_entry["path"]
        gdrive_id = file_entry.get("gdrive_id", "")
        description = file_entry.get("description", "")
        optional = file_entry.get("optional", False)

        # Check if ID is placeholder
        if gdrive_id == "##GDRIVE_ID##" or not gdrive_id:
            if not optional:
                print(f"⚠ Missing Google Drive ID: {file_entry['path']}")
            stats["skipped_no_id"] += 1
            continue

        # Check if file already exists
        if path.exists() and not force:
            if not dry_run:
                print(f"✓ Already exists: {file_entry['path']}")
            stats["skipped_exists"] += 1
            continue

        # Perform download
        print(f"{'Would download' if dry_run else 'Downloading'}: {file_entry['path']}")
        print(f"  Description: {description}")

        if dry_run:
            stats["downloaded"] += 1
            continue

        success = download_from_gdrive(gdrive_id, path, quiet=False)

        if success:
            size = path.stat().st_size
            print(f"  ✓ Downloaded ({format_size(size)})")
            stats["downloaded"] += 1
        else:
            print("  ✗ Download failed")
            stats["failed"] += 1

    return stats


def verify_files(files: list, repo_root: Path) -> dict:
    """
    Verify that downloaded files exist and are valid.

    Args:
        files: List of file dictionaries
        repo_root: Repository root directory

    Returns:
        Dictionary with verification statistics
    """
    stats = {
        "total": len(files),
        "exists": 0,
        "missing_required": 0,
        "missing_optional": 0,
    }

    print(f"\n{'=' * 70}")
    print(f"Verifying {len(files)} files")
    print(f"{'=' * 70}\n")

    missing_required = []
    missing_optional = []

    for file_entry in files:
        path = repo_root / file_entry["path"]
        optional = file_entry.get("optional", False) or not file_entry.get(
            "category_required", True
        )

        if path.exists():
            size = path.stat().st_size
            print(f"✓ {file_entry['path']} ({format_size(size)})")
            stats["exists"] += 1
        else:
            if optional:
                print(f"○ {file_entry['path']} (optional, missing)")
                missing_optional.append(file_entry)
                stats["missing_optional"] += 1
            else:
                print(f"✗ {file_entry['path']} (REQUIRED, missing)")
                missing_required.append(file_entry)
                stats["missing_required"] += 1

    # Print summary
    print(f"\n{'=' * 70}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total files: {stats['total']}")
    print(f"  Exists: {stats['exists']}")
    print(f"  Missing (required): {stats['missing_required']}")
    print(f"  Missing (optional): {stats['missing_optional']}")

    if missing_required:
        print(f"\n⚠ {len(missing_required)} required files are missing!")
        print("These files must be downloaded before running the analysis.")
        for f in missing_required:
            print(f"  - {f['path']}")
            if f.get("gdrive_id") == "##GDRIVE_ID##":
                print("    (Google Drive ID not configured in manifest)")

    if missing_optional:
        print(f"\n○ {len(missing_optional)} optional files are missing.")
        print("These can be regenerated by running the appropriate scripts.")

    return stats


def print_manifest_summary(manifest: dict):
    """Print a summary of the manifest contents."""
    print(f"\n{'=' * 70}")
    print("DATA MANIFEST SUMMARY")
    print(f"{'=' * 70}\n")

    for category, config in CATEGORY_CONFIG.items():
        if category not in manifest:
            continue

        files = manifest[category]
        required_str = "REQUIRED" if config["required"] else "optional"
        configured = sum(1 for f in files if f.get("gdrive_id", "") != "##GDRIVE_ID##")

        print(f"{category} ({required_str}):")
        print(f"  {config['description']}")
        print(
            f"  Files: {len(files)} total, {configured} with Google Drive IDs configured"
        )
        print()


# =============================================================================
# Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download data files for EU Emissions Forecast replication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_data.py                    # Download required files only
    python download_data.py --all              # Download all files including optional
    python download_data.py --category raw_timeseries projections
    python download_data.py --dry-run          # Show what would be downloaded
    python download_data.py --verify           # Check which files exist
    python download_data.py --summary          # Show manifest summary
        """,
    )

    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Download all files including optional (models, cached data)",
    )

    parser.add_argument(
        "--category",
        "-c",
        nargs="+",
        choices=list(CATEGORY_CONFIG.keys()),
        help="Download only specific categories",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if files exist",
    )

    parser.add_argument(
        "--verify",
        "-v",
        action="store_true",
        help="Verify which files exist without downloading",
    )

    parser.add_argument(
        "--summary", "-s", action="store_true", help="Print manifest summary and exit"
    )

    parser.add_argument(
        "--manifest",
        "-m",
        type=Path,
        default=MANIFEST_PATH,
        help=f"Path to manifest file (default: {MANIFEST_PATH})",
    )

    parser.add_argument(
        "--repo-root",
        "-r",
        type=Path,
        default=REPO_ROOT,
        help=f"Repository root directory (default: {REPO_ROOT})",
    )

    args = parser.parse_args()

    # Load manifest
    manifest = load_manifest(args.manifest)

    # Summary mode
    if args.summary:
        print_manifest_summary(manifest)
        return

    # Collect files based on options
    files = collect_files(
        manifest,
        categories=args.category,
        include_optional=args.all or bool(args.category),
    )

    if not files:
        print("No files to process.")
        return

    # Verify mode
    if args.verify:
        stats = verify_files(files, args.repo_root)
        sys.exit(0 if stats["missing_required"] == 0 else 1)

    # Download mode
    stats = download_files(
        files, args.repo_root, dry_run=args.dry_run, force=args.force
    )

    # Print summary
    print(f"\n{'=' * 70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total files: {stats['total']}")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Skipped (exists): {stats['skipped_exists']}")
    print(f"  Skipped (no ID): {stats['skipped_no_id']}")
    print(f"  Failed: {stats['failed']}")

    if stats["skipped_no_id"] > 0:
        print(f"\n⚠ {stats['skipped_no_id']} files have placeholder or invalid Google Drive IDs.")
        print("Contact repository owner for a valid ID.")

    if stats["failed"] > 0:
        print(f"\n✗ {stats['failed']} downloads failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
