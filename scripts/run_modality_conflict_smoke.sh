#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/root/Datasets}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/modality_conflict_smoke/${STAMP}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"

COMMON_ARGS=(
  --rounds 10
  --local_epochs 1
  --batch_size 2
  --grad_accum 4
  --num_workers 0
  --lora_r 1
  --lora_alpha 1
  --lora_dropout 0.0
  --lr 1e-4
  --data_path "${DATA_ROOT}"
)

run_trace() {
  local profile="$1"
  shift

  local output_dir="${OUTPUT_ROOT}/${profile}"
  local cmd=(
    "${PYTHON_BIN}"
    "${ROOT_DIR}/experiments/modality_conflict/run_janus_round_trace.py"
    --dataset_profile "${profile}"
    --output_dir "${output_dir}"
    "${COMMON_ARGS[@]}"
    "$@"
  )

  if [[ -n "${MODEL_NAME_OR_PATH}" ]]; then
    cmd+=(--model_name_or_path "${MODEL_NAME_OR_PATH}")
  fi

  echo
  echo "[smoke] profile=${profile}"
  echo "[smoke] output=${output_dir}"
  printf '[smoke] command='
  printf ' %q' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

mkdir -p "${OUTPUT_ROOT}"

run_trace vqav2 \
  --max_samples 20 \
  --eval_max_samples 10

run_trace cc3m \
  --cc3m_dir "${DATA_ROOT}/cc3m" \
  --max_samples 20 \
  --eval_max_samples 10

run_trace instruct \
  --dataset_name "imthanhlv/instructpix2pix-clip-filtered-10k" \
  --max_samples 20 \
  --eval_max_samples 10

run_trace text \
  --max_samples 20 \
  --eval_max_samples 10 \


echo
echo "[smoke] all profiles finished"
echo "[smoke] outputs saved under ${OUTPUT_ROOT}"
