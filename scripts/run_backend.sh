#!/usr/bin/env bash

# scripts/run_backend.sh
# Activate virtual environment and start FastAPI backend.

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
VENV_DIR="$PROJECT_ROOT/venv"

source "$VENV_DIR/bin/activate"

# Ensure backend module is importable
export PYTHONPATH="$PROJECT_ROOT"

uvicorn backend.main:app --host 0.0.0.0 --port 8000
