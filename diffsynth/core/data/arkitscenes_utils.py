"""Shared helpers for CUT3R-preprocessed ARKitScenes layouts."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image


def frame_timestamp(frame_name: str) -> float:
    """Extract timestamp from ``{scene_id}_{ts}.png`` metadata names."""
    stem = frame_name.rsplit(".", 1)[0]
    return float(stem.split("_")[-1])


def sort_indices_by_timestamp(images: list[str], indices: list[int]) -> list[int]:
    return sorted(indices, key=lambda i: frame_timestamp(images[i]))


def tuple_intrinsics_to_matrix(intrin) -> np.ndarray:
    """Convert CUT3R ARKit tuple ``(w, h, fx, fy, cx, cy)`` to a 3x3 K matrix."""
    w, h, fx, fy, cx, cy = intrin
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def resolve_image_path(images_dir: str, frame_name: str) -> str:
    """Resolve RGB under ``vga_wide/`` (metadata may use ``.png``, disk uses ``.jpg``)."""
    candidates = [frame_name]
    if frame_name.endswith(".png"):
        candidates.append(frame_name[:-4] + ".jpg")
    stem = frame_name.rsplit(".", 1)[0]
    for base in candidates:
        for ext in ("", ".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            path = os.path.join(images_dir, base if ext == "" else stem + ext)
            if os.path.isfile(path):
                return path
    raise FileNotFoundError(
        f"RGB not found for frame '{frame_name}' under {images_dir}"
    )


def load_rgb_image(images_dir: str, frame_name: str) -> Image.Image:
    return Image.open(resolve_image_path(images_dir, frame_name)).convert("RGB")
