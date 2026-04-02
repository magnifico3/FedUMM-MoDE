#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/nvflare_januspro/bin/python}"
DATA_ROOT="${DATA_ROOT:-/root/Datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/modality_conflict_R16}"

run_trace() {
  local profile="$1"
  shift

  local cmd=(
    "${PYTHON_BIN}"
    "${ROOT_DIR}/experiments/modality_conflict/run_janus_round_trace.py"
    --dataset_profile "${profile}"
    --data_path "${DATA_ROOT}"
    "$@"
  )

  echo
  echo "[run] profile=${profile}"
  printf '[run] command='
  printf ' %q' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

mkdir -p "${OUTPUT_ROOT}"

run_trace vqav2 \
  --output_dir "${OUTPUT_ROOT}/vqav2_s2000_e500_lr3e5" \
  --max_samples 2000 \
  --eval_max_samples 500 \
  --local_epochs 1\
  --batch_size 8\
  --grad_accum 1\
  --rounds 10\
  --lora_r 16\
  --lr 3e-5

run_trace cc3m \
  --cc3m_dir "${DATA_ROOT}/cc3m" \
  --output_dir "${OUTPUT_ROOT}/cc3m_s1000_e250_lr3e4" \
  --max_samples 1000 \
  --eval_max_samples 250 \
  --local_epochs 1\
  --batch_size 1\
  --grad_accum 1\
  --rounds 10\
  --lora_r 16\
  --lr 1e-4

run_trace instruct \
  --dataset_name "imthanhlv/instructpix2pix-clip-filtered-10k" \
  --output_dir "${OUTPUT_ROOT}/instruct_s1000_e250_lr3e4" \
  --max_samples 1000 \
  --eval_max_samples 250 \
  --local_epochs 1\
  --batch_size 8\
  --grad_accum 1\
  --rounds 10\
  --lora_r 16\
  --lr 6e-4

run_trace text \
  --output_dir "${OUTPUT_ROOT}/text_s4000_e1000_lr2e4" \
  --max_samples 4000 \
  --eval_max_samples 1000 \
  --local_epochs 1\
  --batch_size 8\
  --grad_accum 1\
  --rounds 10\
  --lora_r 16\
  --lr 2e-4\
  

echo
echo "[run] all profiles finished"
echo "[run] outputs saved under ${OUTPUT_ROOT}"
