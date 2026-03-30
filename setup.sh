#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3.10}"
EXTRAS="${1:-video}"

echo "Creating venv with $PYTHON..."
uv venv --python "$PYTHON"

echo "Installing xdof-sim[$EXTRAS]..."
uv pip install -e ".[$EXTRAS]"

echo ""
echo "Done! Activate with:"
echo "  source .venv/bin/activate"
