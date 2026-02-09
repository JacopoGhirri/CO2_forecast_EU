#!/bin/bash
# =============================================================================
# EU Emissions Forecast - Replication Script
# =============================================================================
#
# Quick replication wrapper script.
#
# Usage:
#   ./replicate.sh              # Run full pipeline
#   ./replicate.sh setup        # Set up environment only
#   ./replicate.sh download     # Download data only
#   ./replicate.sh train        # Train models only
#   ./replicate.sh figures      # Generate figures only
#   ./replicate.sh help         # Show help
#
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "EU Emissions Forecast - Replication Script"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  setup       Set up environment (check dependencies, create directories)"
    echo "  download    Download data from Google Drive"
    echo "  verify      Verify all required data files exist"
    echo "  train       Train all models (VAE, Predictor, Forecaster)"
    echo "  inference   Generate Monte Carlo projections"
    echo "  analysis    Run sensitivity analyses (Sobol + Perturbation)"
    echo "  figures     Generate all figures"
    echo "  full        Run complete pipeline (default)"
    echo "  help        Show this help message"
    echo ""
    echo "Options:"
    echo "  --gpu N     Use specific GPU (e.g., --gpu 0)"
    echo "  --dry-run   Show what would be executed without running"
    echo "  --continue  Continue on error"
    echo ""
    echo "Examples:"
    echo "  $0                      # Run full pipeline"
    echo "  $0 setup download       # Set up and download data"
    echo "  $0 train --gpu 0        # Train models on GPU 0"
    echo "  $0 figures              # Generate all figures"
    echo ""
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.10+."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    log_info "Python version: $PYTHON_VERSION"
}

run_setup() {
    log_info "Setting up environment..."
    cd "$REPO_ROOT"
    python3 "$SCRIPT_DIR/setup_environment.py" "$@"
}

run_download() {
    log_info "Downloading data..."
    cd "$REPO_ROOT"
    python3 "$SCRIPT_DIR/download_data.py" "$@"
}

run_verify() {
    log_info "Verifying data files..."
    cd "$REPO_ROOT"
    python3 "$SCRIPT_DIR/download_data.py" --verify "$@"
}

run_pipeline() {
    log_info "Running pipeline: $*"
    cd "$REPO_ROOT"
    python3 "$SCRIPT_DIR/run_replication.py" "$@"
}

# Main entry point
main() {
    check_python

    if [ $# -eq 0 ]; then
        # Default: run full pipeline
        run_pipeline full
        exit 0
    fi

    COMMAND=$1
    shift

    case $COMMAND in
        setup)
            run_setup "$@"
            ;;
        download)
            run_download "$@"
            ;;
        verify)
            run_verify "$@"
            ;;
        train)
            run_pipeline train "$@"
            ;;
        inference)
            run_pipeline inference "$@"
            ;;
        analysis)
            run_pipeline analysis "$@"
            ;;
        figures)
            run_pipeline figures "$@"
            ;;
        full)
            run_pipeline full "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            # Pass through to run_replication.py
            run_pipeline "$COMMAND" "$@"
            ;;
    esac
}

main "$@"