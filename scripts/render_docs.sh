#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC_NOTEBOOK_DIR="${ROOT_DIR}/docs/notebooks"

export QUARTO_PYTHON="${ROOT_DIR}/.venv/bin/python"
export JAX_PLATFORMS="cpu"

copy_notebooks() {
  cp "${ROOT_DIR}/notebooks/example.ipynb" "${DOC_NOTEBOOK_DIR}/example.ipynb"
  cp "${ROOT_DIR}/notebooks/banana.ipynb" "${DOC_NOTEBOOK_DIR}/banana.ipynb"
  cp "${ROOT_DIR}/notebooks/gmm.ipynb" "${DOC_NOTEBOOK_DIR}/gmm.ipynb"
  cp "${ROOT_DIR}/notebooks/autodiff_mnl.ipynb" "${DOC_NOTEBOOK_DIR}/autodiff_mnl.ipynb"
}

trap copy_notebooks EXIT
copy_notebooks

"${ROOT_DIR}/.venv/bin/python" -m quartodoc build --config "${ROOT_DIR}/docs/_quarto.yml"

exec quarto render "${ROOT_DIR}/docs"
