#!/usr/bin/env python3
"""Unified DL3DV metrics on aft_postprocess outputs.

For each scene under ``--exp_root`` (e.g. outputs/evals/dl3dv):

* PSNR / SSIM / LPIPS(Alex) on ``target_idx`` frames (per-scene, then mean)
* DreamSim ensemble distance on ``target_idx`` frames (per-scene, then mean)
* TSED (thresholded SED) on pred + cameras, split into 81-frame videos
  (per-scene mean over chunks, then mean over scenes)
* MEt3R (pose-free multi-view consistency) on adjacent pred pairs within each
  81-frame video (per-scene mean over chunks, then mean over scenes; lower is better;
  on by default, skip with ``--skip_met3r``)
* FID on all frames except ``context_idx``; FVD on the same frames but split
  into 81-frame videos per scene before I3D clip extraction:
  features are pooled across all scenes into one GT / Pred distribution,
  then a single Frechet distance is computed (standard dataset-level FID/FVD)

Splits come from ``evaluation/dl3dv/splits``.
Cameras for TSED come from ``transforms.json`` under the same scene data dir.
Expects ``<exp_root>/<scene>/aft_postprocess/{gt,pred}/<frame_idx:06d>.png``.

MEt3R requires the vendored package under ``third_party/met3r`` (or ``pip install``).
APE is evaluated separately via ``unified_ape_dl3dv.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from evaluation.paths import ensure_eval_paths

ensure_eval_paths()


import argparse
import csv
import json
import math
import multiprocessing
import os
import queue
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg

from dl3dv_eval_common import (
    DATA_ROOT,
    DEFAULT_EXP_ROOT,
    DEFAULT_ORIG_H,
    DEFAULT_ORIG_W,
    DEFAULT_VIDEO_CHUNK_SIZE,
    SceneSplit,
    chunk_sequence,
    collect_video_frame_paths,
    discover_exp_scenes,
    discover_split_index,
    list_frame_indices,
    load_eval_cameras,
    load_rgb,
    mean_or_nan,
    resolve_frame_path,
    write_summary_metrics_csv,
)

DEFAULT_DREAMSIM_BATCH_SIZE = 16

I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
FVD_CLIP_LEN = 16

# Vendored MEt3R (https://github.com/mohammadasim98/met3r).
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"
DEFAULT_DREAMSIM_WEIGHTS = MODELS_ROOT / "dreamsim"
I3D_CACHE = MODELS_ROOT / "fvd" / "i3d_torchscript.pt"
DEFAULT_MET3R_WEIGHTS_ROOT = MODELS_ROOT / "met3r"
MET3R_ROOT = REPO_ROOT / "third_party" / "met3r"
DEFAULT_MET3R_IMG_SIZE = 256
DEFAULT_MET3R_BATCH_SIZE = 4
DEFAULT_MET3R_PAIR_STRIDE = 1

MET3R_BACKBONE_PTH = {
    "mast3r": "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    "dust3r": "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
}
MET3R_HF_CACHE_NAME = {
    "mast3r": "models--naver--MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    "dust3r": "models--naver--DUSt3R_ViTLarge_BaseDecoder_512_dpt",
}

# Default aft_postprocess geometry (matches unified_postprocess.py).
DEFAULT_TSED_T_MATCHES = 10
DEFAULT_TSED_T_ERROR = 1.5  # scaled from Photoconsistent-NVS 2.0 @ 256² → 192²
SIFT_RATIO = 0.75


def skew_symmetric(t: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]],
        dtype=np.float64,
    )


def fundamental_from_poses(
    c2w1: np.ndarray, c2w2: np.ndarray, K1: np.ndarray, K2: np.ndarray
) -> np.ndarray:
    """F such that x2^T F x1 = 0 for corresponding pixels (OpenCV coords)."""
    w2c1 = np.linalg.inv(c2w1)
    w2c2 = np.linalg.inv(c2w2)
    T_1to2 = w2c2 @ c2w1
    R = T_1to2[:3, :3]
    t = T_1to2[:3, 3]
    E = skew_symmetric(t) @ R
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    return K2_inv.T @ E @ K1_inv


def point_line_distances(points_h: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """points_h / lines: [N,3]; return perpendicular distances in pixels."""
    num = np.abs(np.sum(lines * points_h, axis=1))
    den = np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2) + 1e-12
    return num / den


def symmetric_epipolar_distances(
    pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray
) -> np.ndarray:
    """pts*: [N,2] pixel coords -> SED per match."""
    n = pts1.shape[0]
    p1 = np.concatenate([pts1, np.ones((n, 1), dtype=np.float64)], axis=1)
    p2 = np.concatenate([pts2, np.ones((n, 1), dtype=np.float64)], axis=1)
    l2 = (F @ p1.T).T
    l1 = (F.T @ p2.T).T
    return 0.5 * (point_line_distances(p2, l2) + point_line_distances(p1, l1))


def sift_match_points(
    img1_rgb: np.ndarray, img2_rgb: np.ndarray, ratio: float = SIFT_RATIO
) -> tuple[np.ndarray, np.ndarray]:
    """Return matched point arrays [N,2], [N,2] (may be empty)."""
    g1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des1, des2, k=2)
    pts1, pts2 = [], []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    if not pts1:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    return np.asarray(pts1, dtype=np.float64), np.asarray(pts2, dtype=np.float64)


@dataclass(frozen=True)
class GeometryPairStat:
    n_matches: int
    median_sed: float


def compute_geometry_pair_stats(
    frame_paths: list[Path],
    Ks: list[np.ndarray],
    c2ws: list[np.ndarray],
) -> list[GeometryPairStat]:
    """SIFT matches + median SED for each adjacent frame pair."""
    if len(frame_paths) < 2:
        raise RuntimeError("geometry metrics need >=2 frames")
    if not (len(frame_paths) == len(Ks) == len(c2ws)):
        raise ValueError("frame_paths / Ks / c2ws length mismatch")

    pair_stats: list[GeometryPairStat] = []
    for i in range(len(frame_paths) - 1):
        img1 = load_rgb(frame_paths[i])
        img2 = load_rgb(frame_paths[i + 1])
        pts1, pts2 = sift_match_points(img1, img2)
        F = fundamental_from_poses(c2ws[i], c2ws[i + 1], Ks[i], Ks[i + 1])
        n_matches = int(pts1.shape[0])
        if n_matches == 0:
            median_sed = 1e9
        else:
            seds = symmetric_epipolar_distances(pts1, pts2, F)
            median = float(np.median(seds))
            median_sed = median if np.isfinite(median) else 1e9
        pair_stats.append(
            GeometryPairStat(
                n_matches=n_matches,
                median_sed=median_sed,
            )
        )
    return pair_stats


def tsed_from_pair_stats(
    pair_stats: list[GeometryPairStat],
    *,
    t_matches: int = DEFAULT_TSED_T_MATCHES,
    t_error: float = DEFAULT_TSED_T_ERROR,
) -> dict[str, float]:
    """Photoconsistent-NVS TSED from precomputed pair stats (higher better)."""
    n_pairs = len(pair_stats)
    n_consistent = sum(
        1
        for stat in pair_stats
        if stat.n_matches >= t_matches and stat.median_sed < t_error
    )
    tsed = float(n_consistent / n_pairs) if n_pairs else float("nan")
    return {
        "tsed": tsed,
        "tsed_n_pairs": float(n_pairs),
        "tsed_n_consistent": float(n_consistent),
    }


def compute_tsed(
    frame_paths: list[Path],
    Ks: list[np.ndarray],
    c2ws: list[np.ndarray],
    *,
    t_matches: int = DEFAULT_TSED_T_MATCHES,
    t_error: float = DEFAULT_TSED_T_ERROR,
) -> dict[str, float]:
    """Photoconsistent-NVS TSED: fraction of consistent adjacent pairs (higher better)."""
    return tsed_from_pair_stats(
        compute_geometry_pair_stats(frame_paths, Ks, c2ws),
        t_matches=t_matches,
        t_error=t_error,
    )


def compute_pose_geometry_metrics(
    frame_paths: list[Path],
    Ks: list[np.ndarray],
    c2ws: list[np.ndarray],
    *,
    t_matches: int = DEFAULT_TSED_T_MATCHES,
    t_error: float = DEFAULT_TSED_T_ERROR,
) -> dict[str, float]:
    """TSED from a single SIFT pass over adjacent pairs."""
    return tsed_from_pair_stats(
        compute_geometry_pair_stats(frame_paths, Ks, c2ws),
        t_matches=t_matches,
        t_error=t_error,
    )


def _ensure_met3r_on_path() -> None:
    """Prefer a complete vendored ``third_party/met3r`` (with mast3r/dust3r).

    If the vendored tree is incomplete (common after a shallow copy without
    ``git submodule update --init --recursive``), do **not** put it on
    ``sys.path`` — otherwise it shadows a working site-packages install.
    """
    mast3r_lib = MET3R_ROOT / "mast3r" / "mast3r"
    dust3r_lib = MET3R_ROOT / "mast3r" / "dust3r" / "dust3r"
    if mast3r_lib.is_dir() and dust3r_lib.is_dir():
        root = str(MET3R_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        return
    # Incomplete vendored tree: leave path alone so installed met3r can load.
    if MET3R_ROOT.is_dir():
        print(
            f"[warn] vendored MEt3R incomplete (missing mast3r/dust3r under {MET3R_ROOT}). "
            "Init submodules or rely on site-packages met3r:\n"
            f"  git -C {MET3R_ROOT} submodule update --init --recursive",
            flush=True,
        )


def configure_met3r_weights_root(weights_root: Path) -> Path:
    """Point HF + torch.hub caches at an offline ``met3r`` weights directory.

    Expected layout under ``models/met3r``::

        weights_root/
          mast3r/models--naver--MASt3R_...
          torch_hub_checkpoints/{dino_*.pth,dino16_*.ckpt}
          torch_hub_repos/{mhamilton723_FeatUp_main,facebookresearch_dino_main}

    Returns the runtime torch.hub directory used.
    """
    weights_root = weights_root.expanduser().resolve()
    mast3r_hub = weights_root / "mast3r"
    ckpt_dir = weights_root / "torch_hub_checkpoints"
    repos_dir = weights_root / "torch_hub_repos"
    if not mast3r_hub.is_dir():
        raise FileNotFoundError(
            f"MEt3R weights root missing mast3r/: {mast3r_hub}"
        )
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"MEt3R weights root missing torch_hub_checkpoints/: {ckpt_dir}"
        )
    if not repos_dir.is_dir():
        raise FileNotFoundError(
            f"MEt3R weights root missing torch_hub_repos/: {repos_dir}"
        )

    # HuggingFace: MASt3R / DUSt3R live under HF_HUB_CACHE/models--...
    os.environ["HF_HUB_CACHE"] = str(mast3r_hub)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(mast3r_hub)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # torch.hub expects <hub_dir>/{checkpoints,<repo_dirs>}
    hub_dir = weights_root / "_torch_hub_runtime"
    hub_ckpt = hub_dir / "checkpoints"
    hub_ckpt.mkdir(parents=True, exist_ok=True)
    for src in ckpt_dir.iterdir():
        if not src.is_file():
            continue
        dst = hub_ckpt / src.name
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src.resolve())
    for src in repos_dir.iterdir():
        if not src.is_dir():
            continue
        dst = hub_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src.resolve())

    torch.hub.set_dir(str(hub_dir))
    print(
        f"[met3r] offline weights_root={weights_root}\n"
        f"        HF_HUB_CACHE={mast3r_hub}\n"
        f"        torch.hub dir={hub_dir}",
        flush=True,
    )
    return hub_dir


def _find_hf_snapshot(cache_model_dir: Path) -> Path | None:
    refs_main = cache_model_dir / "refs" / "main"
    if refs_main.is_file():
        rev = refs_main.read_text(encoding="utf-8").strip()
        snap = cache_model_dir / "snapshots" / rev
        if snap.is_dir():
            return snap
    snaps = cache_model_dir / "snapshots"
    if snaps.is_dir():
        children = sorted(
            (p for p in snaps.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if children:
            return children[0]
    return None


def _hf_snapshot_complete(snapshot: Path) -> bool:
    if not (snapshot / "config.json").exists():
        return False
    return (snapshot / "model.safetensors").exists() or (
        snapshot / "pytorch_model.bin"
    ).exists()


def resolve_met3r_backbone_ckpt(weights_root: Path, backbone: str) -> Path | None:
    """Resolve a local MASt3R/DUSt3R checkpoint (.pth) under ``weights_root``."""
    if backbone not in MET3R_BACKBONE_PTH:
        return None
    weights_root = weights_root.expanduser().resolve()
    pth_name = MET3R_BACKBONE_PTH[backbone]
    for base in (
        weights_root,
        weights_root / "mast3r",
        weights_root / "torch_hub_checkpoints",
    ):
        candidate = base / pth_name
        if candidate.is_file() and candidate.stat().st_size > 100_000_000:
            return candidate
    return None


def _parse_mast3r_hf_config(config_path: Path) -> dict:
    """Parse MASt3R HF ``config.json`` (uses bare ``Infinity`` tokens)."""
    text = config_path.read_text(encoding="utf-8")
    text = (
        text.replace("-Infinity", "-float('inf')")
        .replace("Infinity", "float('inf')")
        .replace("true", "True")
        .replace("false", "False")
    )
    config = eval(text, {"__builtins__": {}}, {"float": float})
    img_size = config.get("img_size")
    if isinstance(img_size, int):
        config["img_size"] = (img_size, img_size)
    elif isinstance(img_size, list):
        config["img_size"] = tuple(img_size)
    return config


def load_met3r_backbone_from_hf_snapshot(snapshot_dir: Path, backbone: str):
    """Load MASt3R/DUSt3R directly from an HF cache snapshot (no hub network)."""
    snapshot_dir = snapshot_dir.expanduser().resolve()
    config = _parse_mast3r_hf_config(snapshot_dir / "config.json")
    if backbone == "mast3r":
        from mast3r.model import AsymmetricMASt3R as model_cls
    elif backbone == "dust3r":
        from dust3r.model import AsymmetricCroCo3DStereo as model_cls
    else:
        raise ValueError(f"Unsupported MEt3R backbone: {backbone}")

    model = model_cls(**config)
    safetensors_path = snapshot_dir / "model.safetensors"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path))
    else:
        bin_path = snapshot_dir / "pytorch_model.bin"
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    return model


def validate_met3r_hf_cache(weights_root: Path, backbone: str) -> Path:
    """Ensure the offline HF cache for ``backbone`` exists and looks complete."""
    if backbone not in MET3R_HF_CACHE_NAME:
        raise ValueError(f"Unknown MEt3R backbone: {backbone}")
    weights_root = weights_root.expanduser().resolve()
    cache_dir = weights_root / "mast3r" / MET3R_HF_CACHE_NAME[backbone]
    snapshot = _find_hf_snapshot(cache_dir)
    if snapshot is None or not _hf_snapshot_complete(snapshot):
        raise FileNotFoundError(
            f"MEt3R offline HF cache incomplete for {backbone}: {cache_dir}\n"
            "Expected layout:\n"
            f"  {cache_dir}/snapshots/<rev>/config.json\n"
            f"  {cache_dir}/snapshots/<rev>/model.safetensors\n"
            "Fix options:\n"
            f"  1) Place {MET3R_BACKBONE_PTH[backbone]} under "
            f"{weights_root}/mast3r/\n"
            "  2) Copy a complete mast3r/ HF cache tree from a machine with network "
            "(see scripts/prepare_models.sh)\n"
            "  3) Pass --skip_met3r to skip this metric"
        )
    config = _parse_mast3r_hf_config(snapshot / "config.json")
    img_size = config.get("img_size")
    if not isinstance(img_size, (tuple, list)) or len(img_size) != 2:
        raise RuntimeError(
            f"MEt3R HF config has invalid img_size={img_size!r} in {snapshot / 'config.json'}\n"
            "Re-copy the mast3r HF cache from a working machine, or use a .pth checkpoint."
        )
    weights_path = snapshot / "model.safetensors"
    if weights_path.is_file():
        min_size = 2_000_000_000
        if weights_path.stat().st_size < min_size:
            raise RuntimeError(
                f"MEt3R model.safetensors looks truncated "
                f"({weights_path.stat().st_size} bytes): {weights_path}"
            )
    return snapshot


def _apply_met3r_offline_backbone(
    weights_root: Path | None, backbone: str
) -> None:
    """Force MEt3R backbone loading without HuggingFace network access."""
    import met3r.met3r as met3r_mod

    offline_snapshot: Path | None = None
    if weights_root is not None:
        local_pth = resolve_met3r_backbone_ckpt(weights_root, backbone)
        if local_pth is not None:
            met3r_mod.backbone_to_weights[backbone] = str(local_pth)
            print(
                f"[met3r] offline backbone={backbone} -> {local_pth}",
                flush=True,
            )
            return
        offline_snapshot = validate_met3r_hf_cache(weights_root, backbone)
        met3r_mod.backbone_to_weights[backbone] = str(offline_snapshot)
        print(
            f"[met3r] offline backbone={backbone} -> HF snapshot {offline_snapshot}",
            flush=True,
        )

    if backbone == "mast3r":
        from mast3r.model import AsymmetricMASt3R as model_cls
    elif backbone == "dust3r":
        from dust3r.model import AsymmetricCroCo3DStereo as model_cls
    else:
        return

    original = model_cls.from_pretrained.__func__
    snapshot_for_backbone = offline_snapshot

    @classmethod
    def _from_pretrained_offline(cls, pretrained_model_name_or_path, **kw):
        path = Path(str(pretrained_model_name_or_path))
        if path.is_file():
            return original(cls, str(path), **kw)
        if path.is_dir() and (path / "config.json").is_file():
            return load_met3r_backbone_from_hf_snapshot(path, backbone)
        if snapshot_for_backbone is not None:
            return load_met3r_backbone_from_hf_snapshot(
                snapshot_for_backbone, backbone
            )
        kw.setdefault("local_files_only", True)
        return original(cls, pretrained_model_name_or_path, **kw)

    model_cls.from_pretrained = _from_pretrained_offline


class MEt3RScorer:
    """Thin wrapper around official MEt3R (pose-free multi-view consistency).

    Score is feature dissimilarity after 3D warping: **lower is better**.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        img_size: int = DEFAULT_MET3R_IMG_SIZE,
        backbone: str = "mast3r",
        batch_size: int = DEFAULT_MET3R_BATCH_SIZE,
        weights_root: Path | None = None,
    ):
        if weights_root is not None:
            configure_met3r_weights_root(weights_root)
        _ensure_met3r_on_path()
        try:
            from met3r import MEt3R
        except ImportError as exc:
            raise ImportError(
                "MEt3R is required for this metric.\n"
                "Install a complete vendored copy (with submodules):\n"
                f"  git -C {MET3R_ROOT} submodule update --init --recursive\n"
                f"  pip install {MET3R_ROOT} --no-deps --no-build-isolation\n"
                "Or pass --skip_met3r to skip."
            ) from exc

        _apply_met3r_offline_backbone(weights_root, backbone)

        if device.type != "cuda":
            raise RuntimeError("MEt3R currently expects a CUDA device.")

        self.device = device
        self.img_size = int(img_size)
        self.batch_size = max(1, int(batch_size))
        self.metric = MEt3R(
            img_size=self.img_size,
            use_norm=True,
            backbone=backbone,
            feature_backbone="dino16",
            feature_backbone_weights="mhamilton723/FeatUp",
            upsampler="featup",
            distance="cosine",
            freeze=True,
        ).to(device)
        self.metric.eval()

    def release_gpu(self) -> None:
        """Move MEt3R off GPU so another CUDA subprocess can run safely."""
        if self.device.type == "cuda":
            self.metric.cpu()
            torch.cuda.empty_cache()

    def ensure_gpu(self) -> None:
        if self.device.type == "cuda":
            self.metric.to(self.device)

    def _paths_to_batch(self, paths_a: list[Path], paths_b: list[Path]) -> torch.Tensor:
        """Load RGB pairs -> [B, 2, 3, H, W] in [-1, 1], resized to img_size."""
        assert len(paths_a) == len(paths_b)
        views = []
        for pa, pb in zip(paths_a, paths_b):
            pair = []
            for path in (pa, pb):
                rgb = load_rgb(path).astype(np.float32) / 255.0  # HWC [0,1]
                ten = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # CHW
                if ten.shape[-2:] != (self.img_size, self.img_size):
                    ten = F.interpolate(
                        ten.unsqueeze(0),
                        size=(self.img_size, self.img_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                pair.append(ten * 2.0 - 1.0)
            views.append(torch.stack(pair, dim=0))  # [2,3,H,W]
        return torch.stack(views, dim=0).to(self.device)

    @torch.no_grad()
    def score_frame_paths(
        self,
        frame_paths: list[Path],
        *,
        pair_stride: int = DEFAULT_MET3R_PAIR_STRIDE,
    ) -> dict[str, float]:
        """Mean MEt3R over adjacent (optionally strided) frame pairs."""
        self.ensure_gpu()
        if len(frame_paths) < 2:
            raise RuntimeError("MEt3R needs >=2 frames")
        stride = max(1, int(pair_stride))
        pair_starts = list(range(0, len(frame_paths) - 1, stride))
        pairs_a = [frame_paths[i] for i in pair_starts]
        pairs_b = [frame_paths[i + 1] for i in pair_starts]
        n = len(pairs_a)
        if n == 0:
            raise RuntimeError("MEt3R produced no pairs after applying pair_stride")

        scores: list[float] = []
        for i in range(0, n, self.batch_size):
            batch = self._paths_to_batch(pairs_a[i : i + self.batch_size], pairs_b[i : i + self.batch_size])
            score, *_ = self.metric(
                images=batch,
                return_overlap_mask=False,
                return_score_map=False,
                return_projections=False,
            )
            # score: [B] or [B, ...] — take per-sample mean then collect
            score = score.detach().float().reshape(score.shape[0], -1).mean(dim=1)
            scores.extend(float(x) for x in score.cpu().tolist())

        return {
            "met3r": float(np.mean(scores)),
            "met3r_n_pairs": float(len(scores)),
        }

    @torch.no_grad()
    def score_frame_paths_chunked(
        self,
        frame_paths: list[Path],
        chunk_size: int,
        *,
        pair_stride: int = DEFAULT_MET3R_PAIR_STRIDE,
        drop_incomplete: bool = False,
    ) -> dict[str, float]:
        """Score multiple fixed-length videos and average per metric."""
        chunks = chunk_sequence(
            frame_paths, chunk_size, drop_incomplete=drop_incomplete
        )
        chunk_metrics: list[dict[str, float]] = []
        for chunk_idx, chunk_paths in enumerate(chunks):
            if len(chunk_paths) < 2:
                print(
                    f"[warn] MEt3R skip chunk {chunk_idx}: "
                    f"only {len(chunk_paths)} frame(s)"
                )
                continue
            try:
                chunk_metrics.append(
                    self.score_frame_paths(chunk_paths, pair_stride=pair_stride)
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] MEt3R chunk {chunk_idx} failed: {exc}")
        if not chunk_metrics:
            return {"met3r": float("nan"), "met3r_n_pairs": 0.0}
        return {
            "met3r": mean_or_nan([m["met3r"] for m in chunk_metrics]),
            "met3r_n_pairs": float(
                sum(float(m["met3r_n_pairs"]) for m in chunk_metrics)
            ),
        }


def _met3r_worker_process(
    device_str: str,
    img_size: int,
    backbone: str,
    batch_size: int,
    weights_root: str | None,
    jobs_queue: multiprocessing.Queue,
    results_queue: multiprocessing.Queue,
) -> None:
    """Run MEt3R in a child process so native SIGFPE cannot kill the parent."""
    device = torch.device(device_str)
    weights = Path(weights_root) if weights_root else None
    scorer = MEt3RScorer(
        device,
        img_size=img_size,
        backbone=backbone,
        batch_size=batch_size,
        weights_root=weights,
    )
    while True:
        job = jobs_queue.get()
        if job is None:
            break
        job_id, paths, chunk_size, pair_stride, drop_incomplete = job
        try:
            metrics = scorer.score_frame_paths_chunked(
                [Path(p) for p in paths],
                chunk_size,
                pair_stride=pair_stride,
                drop_incomplete=drop_incomplete,
            )
            results_queue.put((job_id, metrics, None))
        except Exception as exc:  # noqa: BLE001
            results_queue.put(
                (
                    job_id,
                    {"met3r": float("nan"), "met3r_n_pairs": 0.0},
                    str(exc),
                )
            )


class MEt3RSubprocessRunner:
    """Isolate MEt3R in a subprocess (mast3r can SIGFPE without a Python traceback)."""

    def __init__(
        self,
        device: torch.device,
        *,
        img_size: int = DEFAULT_MET3R_IMG_SIZE,
        backbone: str = "mast3r",
        batch_size: int = DEFAULT_MET3R_BATCH_SIZE,
        weights_root: Path | None = None,
        timeout_s: float = 3600.0,
    ) -> None:
        self._device_str = str(device)
        self._img_size = int(img_size)
        self._backbone = backbone
        self._batch_size = max(1, int(batch_size))
        self._weights_root = str(weights_root) if weights_root is not None else None
        self._timeout_s = float(timeout_s)
        self._ctx = multiprocessing.get_context("spawn")
        self._jobs: multiprocessing.Queue = self._ctx.Queue()
        self._results: multiprocessing.Queue = self._ctx.Queue()
        self._proc: multiprocessing.Process | None = None

    def _start_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._proc = self._ctx.Process(
            target=_met3r_worker_process,
            args=(
                self._device_str,
                self._img_size,
                self._backbone,
                self._batch_size,
                self._weights_root,
                self._jobs,
                self._results,
            ),
            daemon=True,
        )
        self._proc.start()
        print("[met3r] started isolated worker process", flush=True)

    def score_frame_paths(
        self,
        frame_paths: list[Path],
        *,
        pair_stride: int = DEFAULT_MET3R_PAIR_STRIDE,
    ) -> dict[str, float]:
        """Score the full sequence as one video (no 81-frame chunking)."""
        if not frame_paths:
            return {"met3r": float("nan"), "met3r_n_pairs": 0.0}
        return self.score_frame_paths_chunked(
            frame_paths,
            max(len(frame_paths), 1),
            pair_stride=pair_stride,
        )

    def score_frame_paths_chunked(
        self,
        frame_paths: list[Path],
        chunk_size: int,
        *,
        pair_stride: int = DEFAULT_MET3R_PAIR_STRIDE,
        drop_incomplete: bool = False,
    ) -> dict[str, float]:
        if not frame_paths:
            return {"met3r": float("nan"), "met3r_n_pairs": 0.0}
        self._start_worker()
        job_id = uuid.uuid4().hex
        self._jobs.put(
            (
                job_id,
                [str(p) for p in frame_paths],
                int(chunk_size),
                int(pair_stride),
                bool(drop_incomplete),
            )
        )
        deadline = time.monotonic() + self._timeout_s
        while time.monotonic() < deadline:
            if self._proc is None or not self._proc.is_alive():
                exit_code = self._proc.exitcode if self._proc is not None else None
                print(
                    f"[warn] MEt3R worker died (exit={exit_code}); "
                    "returning NaN for this scene",
                    flush=True,
                )
                self._proc = None
                return {"met3r": float("nan"), "met3r_n_pairs": 0.0}
            try:
                rid, metrics, err = self._results.get(timeout=1.0)
            except queue.Empty:
                continue
            if rid != job_id:
                print(f"[warn] MEt3R unexpected job id {rid!r}", flush=True)
                continue
            if err:
                print(f"[warn] MEt3R worker error: {err}", flush=True)
            return metrics
        print("[warn] MEt3R worker timed out; returning NaN for this scene", flush=True)
        return {"met3r": float("nan"), "met3r_n_pairs": 0.0}

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._jobs.put(None)
        except Exception:  # noqa: BLE001
            pass
        self._proc.join(timeout=30)
        if self._proc.is_alive():
            self._proc.kill()
        self._proc = None


def psnr_torch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred - gt) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim_torch(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> float:
    # pred/gt: [1,3,H,W] in [0,1]
    channels = pred.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, pred.device, pred.dtype)
    padding = window_size // 2
    mu1 = F.conv2d(pred, window, padding=padding, groups=channels)
    mu2 = F.conv2d(gt, window, padding=padding, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=padding, groups=channels) - mu1_mu2
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean().item())


class LPIPSAlex:
    def __init__(self, device: torch.device):
        import lpips

        self.device = device
        self.model = lpips.LPIPS(net="alex").to(device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        # lpips expects [-1, 1]
        return float(self.model(pred * 2 - 1, gt * 2 - 1).item())


class DreamSimScorer:
    """Official DreamSim ensemble distance (lower is better)."""

    def __init__(
        self,
        device: torch.device,
        *,
        weights_dir: Path = DEFAULT_DREAMSIM_WEIGHTS,
        batch_size: int = DEFAULT_DREAMSIM_BATCH_SIZE,
    ):
        try:
            from dreamsim import dreamsim
        except ImportError as exc:
            raise ImportError(
                "DreamSim is required for this metric. Install dreamsim-main.zip "
                "and its pip dependencies in the evaluation environment."
            ) from exc

        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.weights_dir = weights_dir.expanduser().resolve()
        if not self.weights_dir.is_dir():
            raise FileNotFoundError(f"DreamSim weights directory not found: {self.weights_dir}")
        self.model, self.preprocess = dreamsim(
            pretrained=True,
            device=str(device),
            cache_dir=str(self.weights_dir),
            normalize_embeds=True,
            dreamsim_type="ensemble",
        )

    @torch.no_grad()
    def score_paths(self, pred_paths: list[Path], gt_paths: list[Path]) -> list[float]:
        if len(pred_paths) != len(gt_paths):
            raise ValueError("DreamSim pred/GT path count mismatch")
        scores: list[float] = []
        for start in range(0, len(pred_paths), self.batch_size):
            pred = torch.cat(
                [self._preprocess_path(p) for p in pred_paths[start : start + self.batch_size]],
                dim=0,
            ).to(self.device)
            gt = torch.cat(
                [self._preprocess_path(p) for p in gt_paths[start : start + self.batch_size]],
                dim=0,
            ).to(self.device)
            batch_scores = self.model(pred, gt).detach().float().reshape(-1)
            scores.extend(float(x) for x in batch_scores.cpu().tolist())
        return scores

    def _preprocess_path(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return self.preprocess(image)


def frechet_distance(
    feats_a: np.ndarray, feats_b: np.ndarray, eps: float = 1e-6
) -> float:
    """Frechet distance between two sets of features [N, D]."""
    mu1 = np.mean(feats_a, axis=0)
    mu2 = np.mean(feats_b, axis=0)
    sigma1 = np.atleast_2d(np.cov(feats_a, rowvar=False))
    sigma2 = np.atleast_2d(np.cov(feats_b, rowvar=False))
    if feats_a.shape[0] == 1:
        sigma1 = np.zeros((feats_a.shape[1], feats_a.shape[1]), dtype=np.float64)
    if feats_b.shape[0] == 1:
        sigma2 = np.zeros((feats_b.shape[1], feats_b.shape[1]), dtype=np.float64)

    diff = mu1 - mu2
    # Stabilize singular covariances (few samples / high feature dim).
    offset = np.eye(sigma1.shape[0], dtype=np.float64) * eps
    covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm(
            (sigma1 + 10 * offset).dot(sigma2 + 10 * offset), disp=False
        )
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(
                f"Imaginary component in covmean: {np.max(np.abs(covmean.imag))}"
            )
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean
    )


class InceptionFIDFeatureExtractor(nn.Module):
    """Inception-v3 pool features for FID (torchvision weights)."""

    def __init__(self, device: torch.device):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights

        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = nn.Identity()
        model.aux_logits = False
        self.model = model.eval().to(device)
        self.device = device
        self.input_size = 299

    @torch.no_grad()
    def extract(self, images_uint8: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        # images_uint8: [N,3,H,W] uint8
        feats = []
        for i in range(0, images_uint8.shape[0], batch_size):
            batch = images_uint8[i : i + batch_size].float() / 255.0
            batch = F.interpolate(
                batch,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
            # torchvision inception normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            batch = (batch.to(self.device) - mean) / std
            out = self.model(batch)
            if isinstance(out, tuple):
                out = out[0]
            feats.append(out.detach().cpu().numpy())
        return np.concatenate(feats, axis=0)


def ensure_i3d(cache_path: Path = I3D_CACHE) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 1_000_000:
        return cache_path
    print(f"[download] I3D weights -> {cache_path}")
    urllib.request.urlretrieve(I3D_URL, cache_path)
    if cache_path.stat().st_size < 1_000_000:
        raise RuntimeError(f"Downloaded I3D weights look invalid: {cache_path}")
    return cache_path


class I3DFVDExtractor:
    def __init__(self, device: torch.device, cache_path: Path = I3D_CACHE):
        path = ensure_i3d(cache_path)
        self.device = device
        self.model = torch.jit.load(str(path), map_location=device)
        self.model.eval()

    @torch.no_grad()
    def extract_video_clips(
        self,
        frames_uint8: torch.Tensor,
        clip_len: int = FVD_CLIP_LEN,
        clip_stride: int = 1,
        batch_size: int = 8,
    ) -> np.ndarray:
        """frames_uint8: [T,3,H,W] uint8 -> clip features [N_clips, 400]."""
        t = frames_uint8.shape[0]
        if t < clip_len:
            pad = frames_uint8[-1:].repeat(clip_len - t, 1, 1, 1)
            frames_uint8 = torch.cat([frames_uint8, pad], dim=0)
            t = frames_uint8.shape[0]

        starts = list(range(0, t - clip_len + 1, clip_stride))
        if not starts:
            starts = [0]
        clips = torch.stack(
            [frames_uint8[s : s + clip_len] for s in starts], dim=0
        )  # [N,T,3,H,W]
        clips = clips.permute(0, 2, 1, 3, 4).contiguous().float()  # [N,3,T,H,W]

        feats = []
        for i in range(0, clips.shape[0], batch_size):
            batch = clips[i : i + batch_size].to(self.device)
            out = self.model(batch, rescale=True, resize=True, return_features=True)
            feats.append(out.detach().cpu().numpy())
        return np.concatenate(feats, axis=0)


def images_to_tensor_uint8(paths: list[Path]) -> torch.Tensor:
    tensors = []
    for path in paths:
        arr = load_rgb(path)  # HWC RGB uint8
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
        tensors.append(ten)
    return torch.stack(tensors, dim=0)


def to_float01(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_target_metrics(
    gt_dir: Path,
    pred_dir: Path,
    target_indices: list[int],
    lpips_fn: LPIPSAlex,
    dreamsim_scorer: DreamSimScorer | None = None,
    device: torch.device | None = None,
) -> dict[str, float]:
    if device is None:
        raise TypeError("compute_target_metrics requires device")
    psnrs, ssims, lpipss = [], [], []
    gt_paths: list[Path] = []
    pred_paths: list[Path] = []
    missing = []
    for idx in target_indices:
        gt_path = resolve_frame_path(gt_dir, idx)
        pred_path = resolve_frame_path(pred_dir, idx)
        if gt_path is None or pred_path is None:
            missing.append(idx)
            continue
        gt = to_float01(gt_path, device)
        pred = to_float01(pred_path, device)
        if gt.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch at frame {idx}: gt={tuple(gt.shape)} pred={tuple(pred.shape)}"
            )
        psnrs.append(psnr_torch(pred, gt))
        ssims.append(ssim_torch(pred, gt))
        lpipss.append(lpips_fn(pred, gt))
        gt_paths.append(gt_path)
        pred_paths.append(pred_path)

    if not psnrs:
        raise RuntimeError(
            f"No target frames found in {pred_dir} (missing e.g. {missing[:5]})"
        )
    if missing:
        print(f"[warn] missing {len(missing)} target frames, e.g. {missing[:5]}")

    out: dict[str, float] = {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpipss)),
        "num_target": float(len(psnrs)),
    }
    if dreamsim_scorer is not None:
        dreamsim_scores = dreamsim_scorer.score_paths(pred_paths, gt_paths)
        out["dreamsim"] = float(np.mean(dreamsim_scores))
    return out


def compute_dreamsim_target_metric(
    gt_dir: Path,
    pred_dir: Path,
    target_indices: list[int],
    dreamsim_scorer: DreamSimScorer,
) -> dict[str, float]:
    gt_paths: list[Path] = []
    pred_paths: list[Path] = []
    missing: list[int] = []
    for idx in target_indices:
        gt_path = resolve_frame_path(gt_dir, idx)
        pred_path = resolve_frame_path(pred_dir, idx)
        if gt_path is None or pred_path is None:
            missing.append(idx)
            continue
        gt_paths.append(gt_path)
        pred_paths.append(pred_path)
    if not pred_paths:
        raise RuntimeError(
            f"No target frames found in {pred_dir} (missing e.g. {missing[:5]})"
        )
    if missing:
        print(f"[warn] missing {len(missing)} target frames, e.g. {missing[:5]}")
    scores = dreamsim_scorer.score_paths(pred_paths, gt_paths)
    return {"dreamsim": float(np.mean(scores)), "num_target": float(len(scores))}


def aggregate_tsed_metrics(chunk_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not chunk_metrics:
        return {
            "tsed": float("nan"),
            "tsed_n_pairs": 0.0,
            "tsed_n_consistent": 0.0,
        }
    return {
        "tsed": mean_or_nan([float(m["tsed"]) for m in chunk_metrics]),
        "tsed_n_pairs": float(sum(float(m["tsed_n_pairs"]) for m in chunk_metrics)),
        "tsed_n_consistent": float(
            sum(float(m["tsed_n_consistent"]) for m in chunk_metrics)
        ),
    }


def extract_fvd_clips_chunked(
    fvd_extractor: I3DFVDExtractor,
    frames_uint8: torch.Tensor,
    *,
    clip_stride: int,
    chunk_size: int,
    drop_incomplete: bool = False,
) -> np.ndarray:
    """Extract I3D clip features from fixed-length video chunks."""
    if chunk_size <= 0 or frames_uint8.shape[0] <= chunk_size:
        if (
            drop_incomplete
            and chunk_size > 0
            and frames_uint8.shape[0] < chunk_size
        ):
            return np.zeros((0, 400), dtype=np.float32)
        return fvd_extractor.extract_video_clips(frames_uint8, clip_stride=clip_stride)

    feats: list[np.ndarray] = []
    for start in range(0, frames_uint8.shape[0], chunk_size):
        chunk = frames_uint8[start : start + chunk_size]
        if drop_incomplete and chunk.shape[0] < chunk_size:
            continue
        feats.append(fvd_extractor.extract_video_clips(chunk, clip_stride=clip_stride))
    if not feats:
        return np.zeros((0, 400), dtype=np.float32)
    return np.concatenate(feats, axis=0)


def evaluate_scene(
    scene_dir: Path,
    split: SceneSplit,
    *,
    lpips_fn: LPIPSAlex,
    dreamsim_scorer: DreamSimScorer | None,
    fid_extractor: InceptionFIDFeatureExtractor,
    fvd_extractor: I3DFVDExtractor,
    met3r_scorer: MEt3RScorer | MEt3RSubprocessRunner | None,
    device: torch.device,
    clip_stride: int,
    video_chunk_size: int,
    orig_hw: tuple[int, int],
    tsed_t_matches: int,
    tsed_t_error: float,
    skip_tsed: bool,
    met3r_pair_stride: int,
) -> tuple[dict[str, float | str], dict[str, np.ndarray | float]]:
    gt_dir = scene_dir / "aft_postprocess" / "gt"
    pred_dir = scene_dir / "aft_postprocess" / "pred"

    present = list_frame_indices(pred_dir)
    context_set = set(split.context_indices)
    if present:
        video_indices = [i for i in present if i not in context_set]
    else:
        # Fallback: all indices implied by target ∪ denser listing unavailable
        video_indices = [
            i for i in range(max(split.target_indices) + 1) if i not in context_set
        ]

    target_metrics = compute_target_metrics(
        gt_dir, pred_dir, split.target_indices, lpips_fn, dreamsim_scorer, device
    )

    gt_paths, pred_paths, loaded_indices, missing = collect_video_frame_paths(
        gt_dir, pred_dir, video_indices
    )
    if missing:
        print(f"[warn] missing {len(missing)} video frames, e.g. {missing[:5]}")

    if len(gt_paths) < 2:
        raise RuntimeError(
            f"Need >=2 non-context frames for FID/FVD, got {len(gt_paths)} "
            f"(missing e.g. {missing[:5]})"
        )

    gt_u8 = images_to_tensor_uint8(gt_paths)
    pred_u8 = images_to_tensor_uint8(pred_paths)
    gt_fid = fid_extractor.extract(gt_u8)
    pred_fid = fid_extractor.extract(pred_u8)
    gt_fvd = extract_fvd_clips_chunked(
        fvd_extractor,
        gt_u8,
        clip_stride=clip_stride,
        chunk_size=video_chunk_size
        
    )
    pred_fvd = extract_fvd_clips_chunked(
        fvd_extractor,
        pred_u8,
        clip_stride=clip_stride,
        chunk_size=video_chunk_size
        
    )
    video_chunks = chunk_sequence(
        pred_paths, video_chunk_size
    )
    num_video_chunks = float(len(video_chunks))

    # TSED (pred + known cameras), 81-frame chunks.
    epipolar_metrics: dict[str, float] = {}
    if not skip_tsed:
        if split.transforms_path is None:
            print(
                f"[warn] no transforms.json for {split.scene_hash}; skip TSED"
            )
        else:
            sample = load_rgb(pred_paths[0])
            image_hw = (sample.shape[0], sample.shape[1])
            tsed_chunks: list[dict[str, float]] = []
            for chunk_idx, (chunk_paths, chunk_indices) in enumerate(
                zip(
                    chunk_sequence(
                        pred_paths, video_chunk_size
                    ),
                    chunk_sequence(
                        loaded_indices, video_chunk_size
                    ),
                )
            ):
                if len(chunk_paths) < 2:
                    print(
                        f"[warn] pose-geometry skip chunk {chunk_idx}: "
                        f"only {len(chunk_paths)} frame(s)"
                    )
                    continue
                try:
                    Ks, c2ws = load_eval_cameras(
                        split.transforms_path,
                        chunk_indices,
                        image_hw=image_hw,
                        orig_hw=orig_hw,
                    )
                    geom = compute_pose_geometry_metrics(
                        chunk_paths,
                        Ks,
                        c2ws,
                        t_matches=tsed_t_matches,
                        t_error=tsed_t_error,
                    )
                    tsed_chunks.append(
                        {
                            "tsed": geom["tsed"],
                            "tsed_n_pairs": geom["tsed_n_pairs"],
                            "tsed_n_consistent": geom["tsed_n_consistent"],
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] pose-geometry chunk {chunk_idx} failed: {exc}")
            epipolar_metrics = aggregate_tsed_metrics(tsed_chunks)

    # MEt3R (pred only; pose-free adjacent-pair consistency, lower better).
    met3r_metrics: dict[str, float] = {}
    if met3r_scorer is not None:
        try:
            met3r_metrics = met3r_scorer.score_frame_paths_chunked(
                pred_paths,
                video_chunk_size,
                pair_stride=met3r_pair_stride,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] MEt3R (chunked) failed: {exc}")
        if isinstance(met3r_scorer, MEt3RScorer):
            met3r_scorer.release_gpu()

    row: dict[str, float | str] = {
        "scene": split.scene_hash,
        **target_metrics,
        "num_video_frames": float(len(gt_paths)),
        "num_video_chunks": num_video_chunks,
        "num_fvd_clips": float(gt_fvd.shape[0]),
        # FID/FVD are dataset-level; filled only on the pooled summary row.
        "fid": "",
        "fvd": "",
    }
    if epipolar_metrics:
        row.update(
            {
                "tsed": float(epipolar_metrics["tsed"]),
                "tsed_n_pairs": float(epipolar_metrics["tsed_n_pairs"]),
                "tsed_n_consistent": float(epipolar_metrics["tsed_n_consistent"]),
            }
        )
    if met3r_metrics:
        row["met3r"] = float(met3r_metrics["met3r"])
        row["met3r_n_pairs"] = float(met3r_metrics["met3r_n_pairs"])
    feats: dict[str, np.ndarray | float] = {
        "gt_fid": gt_fid,
        "pred_fid": pred_fid,
        "gt_fvd": gt_fvd,
        "pred_fvd": pred_fvd,
        "num_video_frames": float(len(gt_paths)),
        "num_video_chunks": num_video_chunks,
        "num_fvd_clips": float(gt_fvd.shape[0]),
    }
    return row, feats


def metric_csv_fieldnames(
    *,
    include_dreamsim: bool,
    include_tsed: bool,
    include_met3r: bool,
) -> list[str]:
    fieldnames = ["scene", "psnr", "ssim", "lpips"]
    if include_dreamsim:
        fieldnames.append("dreamsim")
    if include_tsed:
        fieldnames.extend(
            [
                "tsed",
                "tsed_n_pairs",
                "tsed_n_consistent",
            ]
        )
    if include_met3r:
        fieldnames.extend(["met3r", "met3r_n_pairs"])
    fieldnames.extend(
        [
            "fid",
            "fvd",
            "num_target",
            "num_video_frames",
            "num_video_chunks",
            "num_fvd_clips",
        ]
    )
    return fieldnames


def write_csv(
    rows: list[dict[str, float | str]],
    csv_path: Path,
    *,
    fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None:
        fieldnames = metric_csv_fieldnames(
            include_dreamsim=True,
            include_tsed=True,
            include_met3r=True,
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_dreamsim_backfill(
    scene_rows: list[dict[str, float | str]], csv_path: Path
) -> None:
    """Update DreamSim in the compact summary CSV (and detail JSON if present)."""
    average = mean_or_nan([float(row["dreamsim"]) for row in scene_rows])
    write_summary_metrics_csv(csv_path, {"dreamsim": float(average)}, merge=True)

    detail_json = csv_path.with_name(csv_path.stem + "_detail.json")
    if not detail_json.is_file():
        detail_json = csv_path.with_suffix(".json")
    if detail_json.is_file():
        updates = {str(row["scene"]): row for row in scene_rows}
        updates["average"] = {"scene": "average", "dreamsim": average}
        json_rows = json.loads(detail_json.read_text())
        if isinstance(json_rows, list):
            for row in json_rows:
                scene = str(row.get("scene", ""))
                if scene in updates:
                    row["dreamsim"] = updates[scene]["dreamsim"]
                elif scene == "dataset_pooled":
                    row["dreamsim"] = ""
            detail_json.write_text(json.dumps(json_rows, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified DL3DV metrics on aft_postprocess gt/pred"
    )
    parser.add_argument("--data_root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--exp_root",
        type=Path,
        default=DEFAULT_EXP_ROOT,
        help="Method output root containing per-scene aft_postprocess/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Results CSV path (default: <exp_root>/unified_metrics.csv)",
    )
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene filter")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--dreamsim_weights",
        type=Path,
        default=DEFAULT_DREAMSIM_WEIGHTS,
        help="Directory containing the official DreamSim ensemble weights",
    )
    parser.add_argument(
        "--dreamsim_batch_size",
        type=int,
        default=DEFAULT_DREAMSIM_BATCH_SIZE,
        help="Batch size for paired DreamSim target-frame evaluation",
    )
    parser.add_argument(
        "--skip_dreamsim",
        action="store_true",
        help="Skip DreamSim computation",
    )
    parser.add_argument(
        "--dreamsim_only",
        action="store_true",
        help="Only compute DreamSim and merge it into an existing output CSV",
    )
    parser.add_argument(
        "--clip_stride",
        type=int,
        default=4,
        help="Stride for FVD 16-frame sliding windows",
    )
    parser.add_argument(
        "--video_chunk_size",
        type=int,
        default=DEFAULT_VIDEO_CHUNK_SIZE,
        help=(
            "Split each scene's non-context frames into videos of this many frames "
            "for MEt3R / FVD (<=0 disables chunking)"
        ),
    )
    parser.add_argument(
        "--i3d_path",
        type=Path,
        default=I3D_CACHE,
        help="Path to i3d_torchscript.pt (auto-download if missing)",
    )
    parser.add_argument("--orig_h", type=int, default=DEFAULT_ORIG_H)
    parser.add_argument("--orig_w", type=int, default=DEFAULT_ORIG_W)
    parser.add_argument(
        "--tsed_t_matches",
        type=int,
        default=DEFAULT_TSED_T_MATCHES,
        help="TSED min SIFT matches (Photoconsistent-NVS default 10)",
    )
    parser.add_argument(
        "--tsed_t_error",
        type=float,
        default=DEFAULT_TSED_T_ERROR,
        help="TSED max median SED in pixels (default 1.5 for 192² eval)",
    )
    parser.add_argument(
        "--skip_tsed",
        action="store_true",
        help="Skip TSED computation",
    )
    parser.add_argument(
        "--met3r_subprocess",
        action="store_true",
        help="Run MEt3R in an isolated subprocess (default: in-process)",
    )
    parser.add_argument(
        "--skip_met3r",
        action="store_true",
        help="Skip MEt3R multi-view consistency (on by default).",
    )
    parser.add_argument(
        "--met3r_img_size",
        type=int,
        default=DEFAULT_MET3R_IMG_SIZE,
        help="Resize frames to this square size before MEt3R (default 256)",
    )
    parser.add_argument(
        "--met3r_backbone",
        type=str,
        default="mast3r",
        choices=["mast3r", "dust3r", "raft"],
        help="MEt3R warping backbone (paper default: mast3r)",
    )
    parser.add_argument(
        "--met3r_batch_size",
        type=int,
        default=DEFAULT_MET3R_BATCH_SIZE,
        help="Batch size for MEt3R image pairs",
    )
    parser.add_argument(
        "--met3r_pair_stride",
        type=int,
        default=DEFAULT_MET3R_PAIR_STRIDE,
        help="Stride over adjacent pred pairs for MEt3R (1 = every adjacent pair)",
    )
    parser.add_argument(
        "--met3r_weights_root",
        type=Path,
        default=DEFAULT_MET3R_WEIGHTS_ROOT,
        help=(
            "Offline MEt3R weights directory (default: models/met3r). "
            "Must contain mast3r/, torch_hub_checkpoints/, torch_hub_repos/. "
            "Sets HF_HUB_CACHE + torch.hub dir and forces offline mode."
        ),
    )
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    split_index = discover_split_index(args.data_root)
    scene_dirs = discover_exp_scenes(args.exp_root)
    if args.scenes:
        wanted = set(args.scenes)
        scene_dirs = [d for d in scene_dirs if d.name in wanted]
        if not scene_dirs:
            raise SystemExit(f"No matching experiment scenes for: {args.scenes}")

    print(
        f"Scenes={len(scene_dirs)}  device={device}  clip_stride={args.clip_stride}  "
        f"video_chunk_size={args.video_chunk_size}  "
        f"data_root={args.data_root}  exp_root={args.exp_root}"
    )
    print(
        "FID/FVD: pool features across all scenes, then compute one Frechet distance "
        "(dataset-level)."
    )
    print(
        "TSED: per-scene then mean over scenes "
        f"(T_matches={args.tsed_t_matches}, T_error={args.tsed_t_error})."
    )
    if args.skip_met3r:
        print("MEt3R: skipped (--skip_met3r).")
    else:
        print(
            "MEt3R: enabled "
            f"(stride={args.met3r_pair_stride}, backbone={args.met3r_backbone})."
        )

    dreamsim_scorer: DreamSimScorer | None = None
    if not args.skip_dreamsim:
        dreamsim_scorer = DreamSimScorer(
            device,
            weights_dir=args.dreamsim_weights,
            batch_size=args.dreamsim_batch_size,
        )

    if args.dreamsim_only:
        if dreamsim_scorer is None:
            raise SystemExit("--dreamsim_only cannot be combined with --skip_dreamsim")
        dreamsim_rows: list[dict[str, float | str]] = []
        for scene_dir in scene_dirs:
            scene_hash = scene_dir.name
            if scene_hash not in split_index:
                print(f"[skip] {scene_hash}: no split in data_root")
                continue
            metrics = compute_dreamsim_target_metric(
                scene_dir / "aft_postprocess" / "gt",
                scene_dir / "aft_postprocess" / "pred",
                split_index[scene_hash].target_indices,
                dreamsim_scorer,
            )
            row = {"scene": scene_hash, **metrics}
            dreamsim_rows.append(row)
            print(
                f"{scene_hash}: DreamSim={metrics['dreamsim']:.6f} "
                f"target={int(metrics['num_target'])}",
                flush=True,
            )
        if not dreamsim_rows:
            raise SystemExit("No scenes evaluated.")
        dreamsim_average = mean_or_nan(
            [float(row["dreamsim"]) for row in dreamsim_rows]
        )
        out_csv = args.output or (args.exp_root / "unified_metrics.csv")
        write_dreamsim_backfill(dreamsim_rows, out_csv)
        print(
            f"[average over {len(dreamsim_rows)} scenes] "
            f"DreamSim={dreamsim_average:.6f}"
        )
        print(f"Wrote {out_csv}")
        return 0

    lpips_fn = LPIPSAlex(device)
    fid_extractor = InceptionFIDFeatureExtractor(device)
    fvd_extractor = I3DFVDExtractor(device, cache_path=args.i3d_path)

    include_dreamsim = not args.skip_dreamsim
    include_tsed = not args.skip_tsed
    include_met3r = not args.skip_met3r

    met3r_scorer: MEt3RScorer | MEt3RSubprocessRunner | None = None
    if include_met3r:
        if args.met3r_subprocess:
            met3r_scorer = MEt3RSubprocessRunner(
                device,
                img_size=args.met3r_img_size,
                backbone=args.met3r_backbone,
                batch_size=args.met3r_batch_size,
                weights_root=args.met3r_weights_root,
            )
        else:
            met3r_scorer = MEt3RScorer(
                device,
                img_size=args.met3r_img_size,
                backbone=args.met3r_backbone,
                batch_size=args.met3r_batch_size,
                weights_root=args.met3r_weights_root,
            )

    rows: list[dict[str, float | str]] = []
    gt_fid_all: list[np.ndarray] = []
    pred_fid_all: list[np.ndarray] = []
    gt_fvd_all: list[np.ndarray] = []
    pred_fvd_all: list[np.ndarray] = []
    total_video_frames = 0.0
    total_fvd_clips = 0.0

    try:
        for scene_dir in scene_dirs:
            scene_hash = scene_dir.name
            if scene_hash not in split_index:
                print(f"[skip] {scene_hash}: no split in data_root")
                continue
            print(f"\n=== {scene_hash} ===", flush=True)
            row, feats = evaluate_scene(
                scene_dir,
                split_index[scene_hash],
                lpips_fn=lpips_fn,
                dreamsim_scorer=dreamsim_scorer,
                fid_extractor=fid_extractor,
                fvd_extractor=fvd_extractor,
                met3r_scorer=met3r_scorer,
                device=device,
                clip_stride=args.clip_stride,
                video_chunk_size=args.video_chunk_size,
                orig_hw=(args.orig_h, args.orig_w),
                tsed_t_matches=args.tsed_t_matches,
                tsed_t_error=args.tsed_t_error,
                skip_tsed=args.skip_tsed,
                met3r_pair_stride=args.met3r_pair_stride,
            )
            parts = [
                f"PSNR={row['psnr']:.4f}",
                f"SSIM={row['ssim']:.4f}",
                f"LPIPS={row['lpips']:.4f}",
            ]
            if include_dreamsim:
                parts.append(f"DreamSim={float(row['dreamsim']):.4f}")
            if include_tsed:
                parts.append(f"TSED={float(row['tsed']):.4f}")
            if include_met3r:
                parts.append(f"MEt3R={float(row['met3r']):.4f}")
            parts.extend(
                [
                    f"target={int(row['num_target'])}",
                    f"video_frames={int(row['num_video_frames'])}",
                    f"video_chunks={int(row['num_video_chunks'])}",
                    f"fvd_clips={int(row['num_fvd_clips'])}",
                ]
            )
            print("  " + "  ".join(parts), flush=True)
            rows.append(row)
            gt_fid_all.append(feats["gt_fid"])  # type: ignore[arg-type]
            pred_fid_all.append(feats["pred_fid"])  # type: ignore[arg-type]
            gt_fvd_all.append(feats["gt_fvd"])  # type: ignore[arg-type]
            pred_fvd_all.append(feats["pred_fvd"])  # type: ignore[arg-type]
            total_video_frames += float(feats["num_video_frames"])
            total_fvd_clips += float(feats["num_fvd_clips"])
    finally:
        if isinstance(met3r_scorer, MEt3RSubprocessRunner):
            met3r_scorer.close()

    if not rows:
        raise SystemExit("No scenes evaluated.")

    gt_fid = np.concatenate(gt_fid_all, axis=0)
    pred_fid = np.concatenate(pred_fid_all, axis=0)
    gt_fvd = np.concatenate(gt_fvd_all, axis=0)
    pred_fvd = np.concatenate(pred_fvd_all, axis=0)
    dataset_fid = frechet_distance(gt_fid, pred_fid)
    dataset_fvd = frechet_distance(gt_fvd, pred_fvd)

    average: dict[str, float | str] = {
        "scene": "average",
        "psnr": mean_or_nan([float(r["psnr"]) for r in rows]),
        "ssim": mean_or_nan([float(r["ssim"]) for r in rows]),
        "lpips": mean_or_nan([float(r["lpips"]) for r in rows]),
        "fid": "",
        "fvd": "",
        "num_target": mean_or_nan([float(r["num_target"]) for r in rows]),
        "num_video_frames": mean_or_nan([float(r["num_video_frames"]) for r in rows]),
        "num_video_chunks": mean_or_nan([float(r["num_video_chunks"]) for r in rows]),
        "num_fvd_clips": mean_or_nan([float(r["num_fvd_clips"]) for r in rows]),
    }
    if include_dreamsim:
        average["dreamsim"] = mean_or_nan([float(r["dreamsim"]) for r in rows])
    if include_tsed:
        average.update(
            {
                "tsed": mean_or_nan([float(r["tsed"]) for r in rows]),
                "tsed_n_pairs": mean_or_nan([float(r["tsed_n_pairs"]) for r in rows]),
                "tsed_n_consistent": mean_or_nan(
                    [float(r["tsed_n_consistent"]) for r in rows]
                ),
            }
        )
    if include_met3r:
        average.update(
            {
                "met3r": mean_or_nan([float(r["met3r"]) for r in rows]),
                "met3r_n_pairs": mean_or_nan(
                    [float(r["met3r_n_pairs"]) for r in rows]
                ),
            }
        )
    pooled: dict[str, float | str] = {
        "scene": "dataset_pooled",
        "psnr": "",
        "ssim": "",
        "lpips": "",
        "fid": float(dataset_fid),
        "fvd": float(dataset_fvd),
        "num_target": "",
        "num_video_frames": float(total_video_frames),
        "num_video_chunks": "",
        "num_fvd_clips": float(total_fvd_clips),
    }
    if include_dreamsim:
        pooled["dreamsim"] = ""
    if include_tsed:
        for key in ("tsed", "tsed_n_pairs", "tsed_n_consistent"):
            pooled[key] = ""
    if include_met3r:
        for key in ("met3r", "met3r_n_pairs"):
            pooled[key] = ""

    rows_out = rows + [average, pooled]

    out_csv = args.output or (args.exp_root / "unified_metrics.csv")
    detail_json = out_csv.with_name(out_csv.stem + "_detail.json")
    summary: dict[str, float] = {
        "psnr": float(average["psnr"]),
        "ssim": float(average["ssim"]),
        "lpips": float(average["lpips"]),
        "fid": float(dataset_fid),
        "fvd": float(dataset_fvd),
    }
    if include_dreamsim:
        summary["dreamsim"] = float(average["dreamsim"])
    if include_tsed:
        summary["tsed"] = float(average["tsed"])
    if include_met3r:
        summary["met3r"] = float(average["met3r"])
    write_summary_metrics_csv(out_csv, summary, merge=True)
    detail_json.write_text(json.dumps(rows_out, indent=2))

    avg_parts = [
        f"PSNR={float(average['psnr']):.4f}",
        f"SSIM={float(average['ssim']):.4f}",
        f"LPIPS={float(average['lpips']):.4f}",
    ]
    if include_dreamsim:
        avg_parts.append(f"DreamSim={float(average['dreamsim']):.4f}")
    if include_tsed:
        avg_parts.append(f"TSED={float(average['tsed']):.4f}")
    if include_met3r:
        avg_parts.append(f"MEt3R={float(average['met3r']):.4f}")
    print(f"\n[average over {len(rows)} scenes] " + "  ".join(avg_parts))
    print(
        f"[dataset-pooled FID/FVD] "
        f"FID={dataset_fid:.4f}  "
        f"FVD={dataset_fvd:.4f}  "
        f"frames={int(total_video_frames)}  "
        f"clips={int(total_fvd_clips)} "
        f"(gt_fid={gt_fid.shape[0]}, pred_fid={pred_fid.shape[0]}, "
        f"gt_fvd={gt_fvd.shape[0]}, pred_fvd={pred_fvd.shape[0]})"
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {detail_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
