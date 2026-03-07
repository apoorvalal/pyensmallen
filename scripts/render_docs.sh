#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export QUARTO_PYTHON="${ROOT_DIR}/.venv/bin/python"
export JAX_PLATFORMS="cpu"

exec quarto render "${ROOT_DIR}/docs"
