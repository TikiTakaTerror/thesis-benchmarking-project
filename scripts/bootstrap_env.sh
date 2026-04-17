#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Python binary: ${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"[ERROR] Expected Python 3.10.x, got {sys.version.split()[0]}"
    )

print(f"[OK] Python version: {sys.version.split()[0]}")
PY

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
pip install -r "${PROJECT_ROOT}/requirements-dev.txt"
python "${PROJECT_ROOT}/scripts/check_environment.py"

echo "[OK] Bootstrap complete."

