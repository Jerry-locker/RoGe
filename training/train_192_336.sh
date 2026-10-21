#!/usr/bin/env bash
# RoGe LoRA fine-tuning @ 192x336 (base stage).
#
# Usage:
#   bash training/train_192_336.sh \
#     --output_path /path/exp \
#     --dataset_cache_dir /path/cache
#
# Optional env: LAGERNVS_CKPT, USE_DEEPSPEED_ZERO2=0, MLP_* (multi-node)
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_PATH=""
DATASET_CACHE_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_path) OUTPUT_PATH="$2"; shift 2 ;;
    --dataset_cache_dir) DATASET_CACHE_DIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash training/train_192_336.sh --output_path PATH --dataset_cache_dir PATH" >&2
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      echo "Usage: bash training/train_192_336.sh --output_path PATH --dataset_cache_dir PATH" >&2
      exit 1
      ;;
  esac
done
if [[ -z "${OUTPUT_PATH}" || -z "${DATASET_CACHE_DIR}" ]]; then
  echo "error: --output_path and --dataset_cache_dir are required" >&2
  exit 1
fi

LAGERNVS_CKPT="${LAGERNVS_CKPT:-${REPO_ROOT}/models/lagernvs/model.pt}"
MIXED_CONFIG="${SCRIPT_DIR}/dataset.yaml"

USE_DEEPSPEED_ZERO2="${USE_DEEPSPEED_ZERO2:-1}"
if [[ "${USE_DEEPSPEED_ZERO2}" == "1" ]]; then
  ACCELERATE_CONFIG="training/deepspeed_config.yaml"
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
MAX_STEPS=15000
MACHINE_RANK="${MLP_ROLE_INDEX:-0}"
MAIN_PROCESS_IP="${MLP_WORKER_0_HOST:-127.0.0.1}"
MAIN_PROCESS_PORT="${MLP_WORKER_0_PORT:-29500}"

echo "train_192_336: epochs=${NUM_EPOCHS} gpus=${NUM_PROCESSES} steps/epoch=${STEPS_PER_EPOCH} max_steps=${MAX_STEPS} cos_lr_total_steps=${COS_LR_TOTAL_STEPS}"
echo "output_path=${OUTPUT_PATH}"
echo "dataset_cache_dir=${DATASET_CACHE_DIR}"

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
  --height 192 \
  --width 336 \
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
  --lr_warmup_steps 1000 \
  --target_inverse \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
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
  --from_wan \
  --joint_train \
  --nvs_loss_weight 0.1 \
  --lagernvs_lora_rank 8 \
  --train_reconstructor_lora \
  --mixed_train \
  --cos_lr
