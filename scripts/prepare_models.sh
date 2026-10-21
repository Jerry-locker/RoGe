#!/usr/bin/env bash
# Download weights used by RoGe training, inference, and evaluation into
# ${REPO_ROOT}/models. Existing files are left in place.
#
# Layout:
#   models/Wan-AI/Wan2.1-I2V-14B-480P/     DiT; T5/VAE/CLIP .pth on Hugging Face
#   models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/
#                                          T5, VAE, CLIP (ModelScope default)
#   models/Wan-AI/Wan2.1-T2V-1.3B/         umT5 tokenizer
#   models/lagernvs/model.pt               LagerNVS (gated Hugging Face)
#   models/vggt-omega/vggt_omega_1b_512.pt VGGT-Omega (gated Hugging Face)
#   models/fvd/i3d_torchscript.pt          FVD I3D
#   models/dreamsim/                       DreamSim ensemble cache
#   models/met3r/                          MEt3R offline weights
#   models/torch_hub/checkpoints/          LPIPS / torchvision hubs
#   models/inception_v3_google-0cc3c7bd.pth
#
# Wan uses ModelScope unless DIFFSYNTH_DOWNLOAD_SOURCE=huggingface.
# LagerNVS and VGGT-Omega require a Hugging Face token (HF_TOKEN) and
# accepted access on the model pages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS="${REPO_ROOT}/models"

fetch() {
  local url="$1" dest="$2" min_bytes="${3:-1000}"
  if [[ -f "${dest}" ]]; then
    local size
    size="$(stat -c%s "${dest}")"
    if [[ "${size}" -ge "${min_bytes}" ]]; then
      echo "exists ${dest}"
      return 0
    fi
  fi
  mkdir -p "$(dirname "${dest}")"
  echo "download ${url}"
  curl -fL --retry 3 --retry-delay 2 -o "${dest}.partial" "${url}"
  mv "${dest}.partial" "${dest}"
}

clone_repo() {
  local url="$1" dest="$2"
  if [[ -d "${dest}" && -n "$(ls -A "${dest}" 2>/dev/null || true)" ]]; then
    echo "exists ${dest}"
    return 0
  fi
  mkdir -p "$(dirname "${dest}")"
  git clone --depth 1 "${url}" "${dest}"
}

echo "models -> ${MODELS}"
mkdir -p \
  "${MODELS}/fvd" \
  "${MODELS}/dreamsim" \
  "${MODELS}/lagernvs" \
  "${MODELS}/vggt-omega" \
  "${MODELS}/torch_hub/checkpoints" \
  "${MODELS}/met3r/mast3r" \
  "${MODELS}/met3r/torch_hub_checkpoints" \
  "${MODELS}/met3r/torch_hub_repos"

fetch \
  "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1" \
  "${MODELS}/fvd/i3d_torchscript.pt" \
  1000000

fetch \
  "https://download.pytorch.org/models/vgg16-397923af.pth" \
  "${MODELS}/torch_hub/checkpoints/vgg16-397923af.pth" \
  100000000
fetch \
  "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth" \
  "${MODELS}/torch_hub/checkpoints/alexnet-owt-7be5be79.pth" \
  100000000
fetch \
  "https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/vgg.pth" \
  "${MODELS}/torch_hub/checkpoints/vgg.pth" \
  1000000
fetch \
  "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth" \
  "${MODELS}/inception_v3_google-0cc3c7bd.pth" \
  100000000

if [[ ! -d "${MODELS}/dreamsim/ensemble_lora" ]]; then
  fetch \
    "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_ensemble_checkpoint.zip" \
    "${MODELS}/dreamsim/dreamsim_ensemble_checkpoint.zip" \
  1000000
  unzip -q -o "${MODELS}/dreamsim/dreamsim_ensemble_checkpoint.zip" -d "${MODELS}/dreamsim"
fi

fetch \
  "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
  "${MODELS}/met3r/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
  100000000
fetch \
  "https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/dino16_jbu_stack_cocostuff.ckpt" \
  "${MODELS}/met3r/torch_hub_checkpoints/dino16_jbu_stack_cocostuff.ckpt" \
  1000000
fetch \
  "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain.pth" \
  "${MODELS}/met3r/torch_hub_checkpoints/dino_deitsmall16_pretrain.pth" \
  10000000

clone_repo \
  "https://github.com/mhamilton723/FeatUp.git" \
  "${MODELS}/met3r/torch_hub_repos/mhamilton723_FeatUp_main"
clone_repo \
  "https://github.com/facebookresearch/dino.git" \
  "${MODELS}/met3r/torch_hub_repos/facebookresearch_dino_main"

python3 - "${MODELS}" <<'PY'
import os
import sys

models = sys.argv[1]
source = os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE", "modelscope").lower()
i2v = os.path.join(models, "Wan-AI/Wan2.1-I2V-14B-480P")
t2v = os.path.join(models, "Wan-AI/Wan2.1-T2V-1.3B")
converted = os.path.join(models, "DiffSynth-Studio/Wan-Series-Converted-Safetensors")
i2v_patterns = [
    "diffusion_pytorch_model*.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "config.json",
    "google/umt5-xxl/*",
    "xlm-roberta-large/*",
]
t2v_patterns = ["google/umt5-xxl/*"]
# Default ModelScope load redirects T5/VAE/CLIP .pth onto these files.
converted_patterns = [
    "models_t5_umt5-xxl-enc-bf16.safetensors",
    "Wan2.1_VAE.safetensors",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
]
print(f"Wan download source={source}")
if source == "huggingface":
    from huggingface_hub import snapshot_download
    snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir=i2v, allow_patterns=i2v_patterns)
    snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir=t2v, allow_patterns=t2v_patterns)
else:
    from modelscope import snapshot_download
    snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir=i2v, allow_file_pattern=i2v_patterns)
    snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir=t2v, allow_file_pattern=t2v_patterns)
    snapshot_download(
        "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
        local_dir=converted,
        allow_file_pattern=converted_patterns,
    )
PY

python3 - "${MODELS}" <<'PY'
import sys
from huggingface_hub import hf_hub_download

models = sys.argv[1]
jobs = [
    ("facebook/lagernvs_general_512", "model.pt", f"{models}/lagernvs"),
    ("facebook/VGGT-Omega", "vggt_omega_1b_512.pt", f"{models}/vggt-omega"),
]
for repo_id, filename, dest in jobs:
    print(f"Hugging Face {repo_id}/{filename}")
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest)
PY

echo "done: ${MODELS}"
