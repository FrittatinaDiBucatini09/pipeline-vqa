#!/bin/bash
# ==============================================================================
# Pipeline Orchestrator Launcher
# ==============================================================================
# Creates a lightweight virtual environment and launches the orchestrator CLI.
# Usage: ./run_orchestrator.sh [--dry-run]
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.orchestrator_env"

# Cleanup on exit
cleanup() {
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating orchestrator virtual environment..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet rich inquirer
    echo "[INFO] Environment ready."
fi

# Activate and run
source "$VENV_DIR/bin/activate"
python3 "$SCRIPT_DIR/orchestrator/orchestrator.py" "$@"
