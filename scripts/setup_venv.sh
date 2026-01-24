#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN not found."
  echo "Install Python 3.11 (pyenv install 3.11.7) or set PYTHON_BIN=python3.12."
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if major != 3 or minor not in (11, 12):
    raise SystemExit(1)
PY
then
  echo "ERROR: $PYTHON_BIN must be Python 3.11 or 3.12."
  echo "Run: pyenv install 3.11.7 && pyenv local 3.11.7"
  exit 1
fi

if [ ! -f "requirements.txt" ]; then
  echo "ERROR: requirements.txt not found in repo root."
  exit 1
fi

echo "Using $PYTHON_BIN: $($PYTHON_BIN --version)"

if [ -d ".venv" ]; then
  echo "Removing existing .venv"
  rm -rf .venv
fi

"$PYTHON_BIN" -m venv .venv
VENV_PY=".venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: venv creation failed."
  exit 1
fi

"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements.txt
"$VENV_PY" -c "import torch; print(torch.__version__)"

echo "Setup complete."
echo "Activate with: source .venv/bin/activate"
