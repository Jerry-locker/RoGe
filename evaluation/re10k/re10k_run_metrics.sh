#!/usr/bin/env bash
# Run unified RE10K metrics on aft_postprocess outputs under EXP_ROOT.
# APE is separate: bash evaluation/re10k/re10k_run_ape.sh EXP_ROOT
# MEt3R is on by default; disable with ENABLE_MET3R=0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

EXP_ROOT="${1:-}"
if [[ -z "${EXP_ROOT}" ]]; then
  echo "usage: bash evaluation/re10k/re10k_run_metrics.sh EXP_ROOT" >&2
  exit 1
fi

I3D_PATH="${I3D_PATH:-${REPO_ROOT}/models/fvd/i3d_torchscript.pt}"
MET3R_WEIGHTS_ROOT="${MET3R_WEIGHTS_ROOT:-${REPO_ROOT}/models/met3r}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/processed_re10k/benchmark}"
ENABLE_MET3R="${ENABLE_MET3R:-1}"
DREAMSIM_WEIGHTS="${DREAMSIM_WEIGHTS:-${REPO_ROOT}/models/dreamsim}"

TORCH_HUB_CKPT_DIR="${TORCH_HUB_CKPT_DIR:-/root/.cache/torch/hub/checkpoints}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-${REPO_ROOT}/models}"

mkdir -p "${TORCH_HUB_CKPT_DIR}"
for ckpt in \
  vgg16-397923af.pth \
  alexnet-owt-7be5be79.pth \
  vgg.pth \
  inception_v3_google-0cc3c7bd.pth
do
  src="${PRETRAIN_ROOT}/torch_hub/checkpoints/${ckpt}"
  if [[ "${ckpt}" == "inception_v3_google-0cc3c7bd.pth" ]]; then
    src="${PRETRAIN_ROOT}/inception_v3_google-0cc3c7bd.pth"
  fi
  dst="${TORCH_HUB_CKPT_DIR}/${ckpt}"
  if [[ ! -f "${dst}" && -f "${src}" ]]; then
    cp "${src}" "${dst}"
  fi
done

MET3R_ARGS=()
if [[ "${ENABLE_MET3R}" == "0" ]]; then
  MET3R_ARGS+=(--skip_met3r)
fi

echo "RE10K metrics: exp_root=${EXP_ROOT} enable_met3r=${ENABLE_MET3R}"

python "${SCRIPT_DIR}/unified_metrics_re10k.py" \
  --data_root "${DATA_ROOT}" \
  --exp_root "${EXP_ROOT}" \
  --i3d_path "${I3D_PATH}" \
  --dreamsim_weights "${DREAMSIM_WEIGHTS}" \
  --met3r_weights_root "${MET3R_WEIGHTS_ROOT}" \
  "${MET3R_ARGS[@]}" \
  "${@:2}"
