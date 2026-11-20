#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_DIR="${1:-${SCRIPT_DIR}/.venv}"
PACKAGE_ROOT="${BAGEL_PACKAGE_ROOT:-${SCRIPT_DIR}/.bagel_source}"
if [[ ! -f "${PACKAGE_ROOT}/pyproject.toml" ]]; then
  echo "Install this adapter using scripts/install_adapter.py, or set BAGEL_PACKAGE_ROOT to the research checkout." >&2
  exit 1
fi
"${PYTHON_BIN:-python3.10}" -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip
"${ENV_DIR}/bin/python" -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
"${ENV_DIR}/bin/python" -m pip install -r "${PACKAGE_ROOT}/requirements-inference.txt" \
  -c "${SCRIPT_DIR}/requirements-transport.txt" \
  -e "${PACKAGE_ROOT}" -e "${XPL_ROOT}"
"${ENV_DIR}/bin/python" -m pip check
echo "Policy environment: ${ENV_DIR}"
