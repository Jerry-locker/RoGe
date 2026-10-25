#!/usr/bin/env bash
# DL3DV benchmark eval (high-res recipe: 480x832 + joint-train LoRA).
#
# Usage:
#   bash evaluation/dl3dv/dl3dv_eval.sh \
#     --checkpoint_path /path/to/step-500.safetensors \
#     --output_dir outputs/dl3dv_eval
#
# Optional env: LAGERNVS_CKPT, MLP_WORKER_NUM / MLP_ROLE_INDEX / MLP_WORKER_GPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export DIFFSYNTH_MODEL_BASE_PATH="${REPO_ROOT}/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

CHECKPOINT_PATH=""
OUTPUT_DIR="${REPO_ROOT}/outputs/dl3dv_eval"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash evaluation/dl3dv/dl3dv_eval.sh --checkpoint_path PATH [--output_dir PATH]" >&2
      echo "  default --output_dir: ${REPO_ROOT}/outputs/dl3dv_eval" >&2
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      echo "Usage: bash evaluation/dl3dv/dl3dv_eval.sh --checkpoint_path PATH [--output_dir PATH]" >&2
      exit 1
      ;;
  esac
done
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "error: --checkpoint_path is required" >&2
  exit 1
fi

DATA_ROOT="${REPO_ROOT}/data/processed_dl3dv/test"
LAGERNVS_CKPT="${LAGERNVS_CKPT:-${REPO_ROOT}/models/lagernvs/model.pt}"

NUM_MACHINES="${MLP_WORKER_NUM:-1}"
MACHINE_RANK="${MLP_ROLE_INDEX:-0}"
GPUS_PER_MACHINE="${MLP_WORKER_GPU:-8}"
GPUS="$(seq -s, 0 $((GPUS_PER_MACHINE - 1)))"

echo "DL3DV eval: checkpoint=${CHECKPOINT_PATH}"
echo "output_dir=${OUTPUT_DIR} data_root=${DATA_ROOT}"
echo "Multi-node: num_machines=${NUM_MACHINES}, machine_rank=${MACHINE_RANK}, gpus=${GPUS}"

python "${SCRIPT_DIR}/dl3dv_benchmark_infer.py" \
  --scene_root "${DATA_ROOT}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --height 480 --width 832 \
  --chunk_size 81 \
  --gpus "${GPUS}" \
  --launch_interval 10 \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --hybrid_vae_encoding \
  --lagernvs_ckpt "${LAGERNVS_CKPT}" \
  --lagernvs_channels 384 \
  --seed 42 \
  --num_inference_steps 40 \
  --overwrite \
  --raymap_pe \
  --geom_pe \
  --joint_train \
  --train_reconstructor_lora \
  --lagernvs_lora_rank 8
