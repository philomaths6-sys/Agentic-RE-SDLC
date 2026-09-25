#!/usr/bin/env bash

# scripts/setup_venv.sh
# Create and activate a Python virtual environment, then install project dependencies.

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
VENV_DIR="$PROJECT_ROOT/venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# Upgrade pip and install requirements
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install --break-system-packages -r "$PROJECT_ROOT/requirements.txt"

echo "[INFO] Virtual environment setup complete. To activate later, run: source $VENV_DIR/bin/activate"
