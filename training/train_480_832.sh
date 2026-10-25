#!/usr/bin/env bash
# RoGe LoRA fine-tuning @ 480x832 (high-res stage).
#
# Usage:
#   bash training/train_480_832.sh \
#     --output_path outputs/exp_highres \
#     --dataset_cache_dir data/cache_re10k_dl3dv \
#     --resume_ckpt_path outputs/exp_base/step-15000.safetensors
#
# Optional env: LAGERNVS_CKPT, USE_DEEPSPEED_ZERO2=0, MLP_* (multi-node)
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_PATH="${REPO_ROOT}/outputs/exp_highres"
DATASET_CACHE_DIR="${REPO_ROOT}/data/cache_re10k_dl3dv"
RESUME_CKPT_PATH="${REPO_ROOT}/outputs/exp_base/step-15000.safetensors"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_path) OUTPUT_PATH="$2"; shift 2 ;;
    --dataset_cache_dir) DATASET_CACHE_DIR="$2"; shift 2 ;;
    --resume_ckpt_path) RESUME_CKPT_PATH="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash training/train_480_832.sh [--output_path PATH] [--dataset_cache_dir PATH] [--resume_ckpt_path PATH]" >&2
      echo "  default --output_path: ${REPO_ROOT}/outputs/exp_highres" >&2
      echo "  default --dataset_cache_dir: ${REPO_ROOT}/data/cache_re10k_dl3dv" >&2
      echo "  default --resume_ckpt_path: ${REPO_ROOT}/outputs/exp_base/step-15000.safetensors" >&2
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      echo "Usage: bash training/train_480_832.sh [--output_path PATH] [--dataset_cache_dir PATH] [--resume_ckpt_path PATH]" >&2
      exit 1
      ;;
  esac
done

LAGERNVS_CKPT="${LAGERNVS_CKPT:-${REPO_ROOT}/models/lagernvs/model.pt}"
MIXED_CONFIG="${SCRIPT_DIR}/dataset.yaml"

USE_DEEPSPEED_ZERO2="${USE_DEEPSPEED_ZERO2:-1}"
if [[ "${USE_DEEPSPEED_ZERO2}" == "1" ]]; then
  ACCELERATE_CONFIG="training/deepspeed_config.yaml"
  # DeepSpeed reads nvcc at import. A wheel install has no CUDA toolkit.
  if [[ ! -x "${CUDA_HOME:-}/bin/nvcc" ]] && ! command -v nvcc >/dev/null 2>&1; then
    cuda_ver="$(python -c 'import torch; print(torch.version.cuda)')"
    mkdir -p "${HOME}/fake_cuda/bin"
    cat > "${HOME}/fake_cuda/bin/nvcc" << EOF
#!/bin/bash
echo "Cuda compilation tools, release ${cuda_ver}, V${cuda_ver}.0"
EOF
    chmod +x "${HOME}/fake_cuda/bin/nvcc"
    export CUDA_HOME="${HOME}/fake_cuda"
    export PATH="${CUDA_HOME}/bin:${PATH}"
  fi
else
  ACCELERATE_CONFIG="training/ddp_config.yaml"
fi

EXTRA_LAUNCH_ARGS=()
if [[ "${USE_DEEPSPEED_ZERO2}" == "1" && "${MLP_WORKER_NUM:-1}" -gt 1 ]]; then
  EXTRA_LAUNCH_ARGS+=(--deepspeed_multinode_launcher standard)
fi

NUM_MACHINES="${MLP_WORKER_NUM:-1}"
GPUS_PER_MACHINE="${MLP_WORKER_GPU:-8}"
NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_MACHINE))
NUM_EPOCHS=600
STEPS_PER_EPOCH_BASE=10000
STEPS_PER_EPOCH=$((STEPS_PER_EPOCH_BASE / NUM_PROCESSES))
COS_LR_TOTAL_STEPS=$((NUM_EPOCHS * STEPS_PER_EPOCH))
MAX_STEPS=500
MACHINE_RANK="${MLP_ROLE_INDEX:-0}"
MAIN_PROCESS_IP="${MLP_WORKER_0_HOST:-127.0.0.1}"
MAIN_PROCESS_PORT="${MLP_WORKER_0_PORT:-29500}"

echo "train_480_832: epochs=${NUM_EPOCHS} gpus=${NUM_PROCESSES} steps/epoch=${STEPS_PER_EPOCH} max_steps=${MAX_STEPS} cos_lr_total_steps=${COS_LR_TOTAL_STEPS}"
echo "output_path=${OUTPUT_PATH}"
echo "dataset_cache_dir=${DATASET_CACHE_DIR}"
echo "resume_ckpt_path=${RESUME_CKPT_PATH}"

export DIFFSYNTH_MODEL_BASE_PATH="${REPO_ROOT}/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export WANDB_MODE=disabled
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
if [[ -n "${MLP_IFNAME:-}" ]]; then
  export NCCL_SOCKET_IFNAME="${MLP_IFNAME}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_IFNAME}}"
fi

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_machines "${NUM_MACHINES}" \
  --num_processes "${NUM_PROCESSES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${EXTRA_LAUNCH_ARGS[@]}" \
  training/train.py \
  --dataset_type mixed \
  --mixed_dataset_config "${MIXED_CONFIG}" \
  --dataset_cache_dir "${DATASET_CACHE_DIR}" \
  --dataset_metadata_path "" \
  --height 480 \
  --width 832 \
  --num_context_frames 6 \
  --num_target_frames 81 \
  --dataset_repeat 1 \
  --dataset_num_workers 12 \
  --distributed_timeout_mins 1440 \
  --num_epochs "${NUM_EPOCHS}" \
  --steps_per_epoch_base "${STEPS_PER_EPOCH_BASE}" \
  --max_steps "${MAX_STEPS}" \
  --cos_lr_total_steps "${COS_LR_TOTAL_STEPS}" \
  --save_steps 1000 \
  --save_keep_first 999999 \
  --vis_steps 5000 \
  --vis_inference_steps 10 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth" \
  --skip_clip \
  --unload_text_encoder \
  --learning_rate 1e-4 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 200 \
  --target_inverse \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --resume_ckpt_path "${RESUME_CKPT_PATH}" \
  --extra_inputs "input_image" \
  --modify_channels \
  --gradient_accumulation_steps 1 \
  --initialize_model_on_cpu \
  --hybrid_vae_encoding \
  --lagernvs_ckpt "${LAGERNVS_CKPT}" \
  --lagernvs_channels 384 \
  --seed 42 \
  --use_gradient_checkpointing \
  --raymap_pe \
  --geom_pe \
  --load_frozen_lagernvs_lora \
  --lagernvs_lora_rank 8 \
  --mixed_train \
  --cos_lr
