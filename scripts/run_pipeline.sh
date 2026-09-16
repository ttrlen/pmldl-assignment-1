#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${PROJECT_DIRECTORY}/.venv/bin:${PATH}"
mkdir -p "${PROJECT_DIRECTORY}/logs"
cd "${PROJECT_DIRECTORY}"
LOG_PATH="${PROJECT_DIRECTORY}/logs/pipeline.log"
printf '%s Pipeline started\n' "$(date -Is)" >> "${LOG_PATH}"
dvc repro >> "${LOG_PATH}" 2>&1
printf '%s Pipeline finished successfully\n' "$(date -Is)" >> "${LOG_PATH}"
