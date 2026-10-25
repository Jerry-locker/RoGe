#!/usr/bin/env python3
"""Roaming demo: 6 posed context views to an 81-frame video.

The bundled example lives in ``assets/demo/``. ``context/`` holds six
undistorted RGB frames and ``cameras.npz`` (OpenCV w2c, same order as
``000.png`` ... ``005.png``). ``trajectory.npz`` holds the 81 target w2c
poses. Both camera files share one intrinsic at the RGB resolution.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.model import RoGe
from diffsynth.utils.data import save_video

NUM_CONTEXT = 6
NUM_TARGET = 81
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _load_rgb_dir(context_dir: Path) -> list[Image.Image]:
    paths = sorted(
        path
        for path in context_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(paths) != NUM_CONTEXT:
        raise ValueError(
            f"Expected {NUM_CONTEXT} context images in {context_dir}, found {len(paths)}"
        )
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def _load_cameras(path: Path, num_poses: int, label: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} cameras: {path}")
    data = np.load(path)
    w2c = np.asarray(data["w2c"], dtype=np.float32)
    intrinsic = np.asarray(data["intrinsic"], dtype=np.float32)
    if w2c.shape != (num_poses, 4, 4):
        raise ValueError(f"{label} w2c shape {w2c.shape} != {(num_poses, 4, 4)}")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"{label} intrinsic shape {intrinsic.shape} != (3, 3)")
    width = int(data["width"])
    height = int(data["height"])
    return w2c, intrinsic, (width, height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an 81-frame roaming video from 6 context views and a camera trajectory."
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=REPO_ROOT / "assets" / "demo" / "context",
        help="Directory of 6 context RGB images and cameras.npz.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=REPO_ROOT / "assets" / "demo" / "trajectory.npz",
        help="81-frame OpenCV w2c trajectory.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=REPO_ROOT / "models" / "roge" / "highres_step-500.safetensors",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "roaming.mp4",
    )
    parser.add_argument(
        "--lagernvs_ckpt",
        type=Path,
        default=REPO_ROOT / "models" / "lagernvs" / "model.pt",
    )
    parser.add_argument(
        "--base_model_dir",
        type=Path,
        default=REPO_ROOT / "models",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vram_limit", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(args.base_model_dir))
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

    context_dir = args.context.expanduser().resolve()
    images = _load_rgb_dir(context_dir)
    context_w2c, context_k, context_wh = _load_cameras(
        context_dir / "cameras.npz", NUM_CONTEXT, "context"
    )
    target_w2c, target_k, target_wh = _load_cameras(
        args.trajectory.expanduser().resolve(), NUM_TARGET, "trajectory"
    )
    if context_wh != target_wh:
        raise ValueError(f"Context image size {context_wh} != trajectory size {target_wh}")
    if not np.allclose(context_k, target_k):
        raise ValueError("Context and trajectory intrinsics differ")
    image_wh = (images[0].width, images[0].height)
    if image_wh != context_wh:
        raise ValueError(f"Context RGB size {image_wh} != cameras size {context_wh}")

    w2c = np.concatenate([context_w2c, target_w2c], axis=0)
    intrinsics = np.stack([context_k] * (NUM_CONTEXT + NUM_TARGET), axis=0)

    model = RoGe(
        checkpoint_path=str(args.checkpoint_path.expanduser().resolve()),
        device=args.device,
        vram_limit=args.vram_limit,
        base_model_dir=str(args.base_model_dir.expanduser().resolve()),
        hybrid_vae_encoding=True,
        raymap_pe=True,
        geom_pe=True,
        lagernvs_ckpt=str(args.lagernvs_ckpt.expanduser().resolve()),
        lagernvs_channels=384,
        joint_train=True,
        lagernvs_lora_rank=8,
        train_reconstructor_lora=True,
    )
    video = model.generate(
        images=images,
        w2c_poses=w2c,
        intrinsics=intrinsics,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        resize_mode="crop",
        orig_wh=image_wh,
    )
    if len(video) != NUM_CONTEXT + NUM_TARGET:
        raise RuntimeError(
            f"Model returned {len(video)} frames, expected {NUM_CONTEXT + NUM_TARGET}"
        )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_video(video[NUM_CONTEXT:], str(output), fps=args.fps)
    print(f"Saved {NUM_TARGET} frames to {output}")


if __name__ == "__main__":
    main()
