#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${PROJECT_DIRECTORY}/.venv/bin:${PATH}"
mkdir -p "${PROJECT_DIRECTORY}/logs"
cd "${PROJECT_DIRECTORY}"
dvc repro >> "${PROJECT_DIRECTORY}/logs/pipeline.log" 2>&1
