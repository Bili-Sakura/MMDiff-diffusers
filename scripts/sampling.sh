#!/usr/bin/env bash
# End-to-end MMDiff sampling through the native Diffusers pipeline.
# Usage:
#   bash scripts/sampling.sh
#   PROMPT="a ship near the coast" bash scripts/sampling.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/MMDiff}"
PROMPT="${PROMPT:-There is a ship in the blue water on the shore.}"
SCENE="${SCENE:-ship}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/result}"

cd "${PROJECT_ROOT}"

"${PYTHON}" "${PROJECT_ROOT}/scripts/infer.py" \
  --model_path "${MODEL_PATH}" \
  --prompt "${PROMPT}" \
  --scene "${SCENE}" \
  --device "${DEVICE}" \
  --output_dir "${OUTPUT_DIR}" \
  --name "${SCENE}" \
  "$@"
