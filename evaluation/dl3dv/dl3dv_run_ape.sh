#!/usr/bin/env bash
# Run VGGT-Omega APE on DL3DV aft_postprocess outputs under EXP_ROOT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

EXP_ROOT="${1:-}"
if [[ -z "${EXP_ROOT}" ]]; then
  echo "usage: bash evaluation/dl3dv/dl3dv_run_ape.sh EXP_ROOT [extra unified_ape_dl3dv.py args...]" >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/processed_dl3dv/test}"
VGGT_OMEGA_CHECKPOINT="${VGGT_OMEGA_CHECKPOINT:-${REPO_ROOT}/models/vggt-omega/vggt_omega_1b_512.pt}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
VGGT_OMEGA_RESOLUTION="${VGGT_OMEGA_RESOLUTION:-512}"

echo "DL3DV APE: exp_root=${EXP_ROOT} gpus=${GPUS}"

python "${SCRIPT_DIR}/unified_ape_dl3dv.py" \
  --data_root "${DATA_ROOT}" \
  --exp_root "${EXP_ROOT}" \
  --vggt_omega_checkpoint "${VGGT_OMEGA_CHECKPOINT}" \
  --vggt_omega_resolution "${VGGT_OMEGA_RESOLUTION}" \
  --gpus "${GPUS}" \
  "${@:2}"
