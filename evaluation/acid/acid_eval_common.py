"""Shared ACID (LagerNVS layout) scene discovery — ``processed_acid/benchmark``.

Layout::

    processed_acid/benchmark/
    ├── images/<seq_id>/*.jpg
    ├── metadata/<seq_id>.json                 # fxfycxcy (px) + w2c
    └── full_list.txt

Context and target indices are ``evaluation/acid/splits/<seq_id>.json``
(9 context views, stride-8 targets). Same camera conventions as RE10K;
image suffix is typically ``.jpg``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from evaluation.paths import ensure_eval_paths

ensure_eval_paths()


import json
from pathlib import Path

import cv2

import re10k_eval_common as _re10k

# Re-export helpers used by metrics / APE adapters.
IMAGE_SUFFIXES = _re10k.IMAGE_SUFFIXES
SUMMARY_METRIC_ORDER = (
    "psnr",
    "ssim",
    "lpips",
    "dreamsim",
    "tsed",
    "met3r",
    "fvd",
    "transape",
    "rotape",
    "fid",
)
SceneSplit = _re10k.SceneSplit
SceneLayout = _re10k.SceneLayout
load_rgb = _re10k.load_rgb
resolve_frame_path = _re10k.resolve_frame_path
find_metadata_path = _re10k.find_metadata_path
discover_exp_scenes = _re10k.discover_exp_scenes
list_frame_indices = _re10k.list_frame_indices
compute_common_crop = _re10k.compute_common_crop
load_eval_cameras = _re10k.load_eval_cameras
chunk_sequence = _re10k.chunk_sequence
mean_or_nan = _re10k.mean_or_nan
collect_video_frame_paths = _re10k.collect_video_frame_paths
write_summary_metrics_csv = _re10k.write_summary_metrics_csv
read_summary_metrics_csv = _re10k.read_summary_metrics_csv

DATA_ROOT = _REPO_ROOT / "data" / "processed_acid" / "benchmark"
SPLITS_DIR = Path(__file__).resolve().parent / "splits"
DEFAULT_EXP_ROOT = _REPO_ROOT / "outputs" / "evals" / "acid"
DEFAULT_ORIG_H = 360
DEFAULT_ORIG_W = 640
DEFAULT_VIDEO_CHUNK_SIZE = 81


def _first_image_hw(images_dir: Path) -> tuple[int, int]:
    for name in (
        "00000.jpg",
        "00000.jpeg",
        "00000.png",
        "0000.jpg",
        "0000.png",
        "0.jpg",
        "0.png",
    ):
        path = images_dir / name
        if path.is_file():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                return h, w
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        files = sorted(images_dir.glob(pattern))
        if not files:
            continue
        img = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        return h, w
    raise FileNotFoundError(f"No images under {images_dir}")


def _hw_from_metadata(metadata_path: Path) -> tuple[int, int] | None:
    try:
        data = json.loads(metadata_path.read_text())
        w = int(data.get("width") or 0)
        h = int(data.get("height") or 0)
        if w > 0 and h > 0:
            return h, w
    except Exception:
        return None
    return _re10k._infer_hw_from_metadata(metadata_path)


def split_file(scene_hash: str) -> Path:
    return SPLITS_DIR / f"{scene_hash}.json"


def load_scene_split(scene_hash: str) -> dict:
    path = split_file(scene_hash)
    if not path.is_file():
        raise FileNotFoundError(f"Missing eval split for {scene_hash}: {path}")
    return json.loads(path.read_text())


def resolve_scene_layout(scene_dir: Path) -> SceneLayout:
    """Resolve a scene directory whose id has a shipped split."""
    scene_dir = scene_dir.resolve()
    context_target_path = split_file(scene_dir.name)
    if not context_target_path.is_file():
        raise FileNotFoundError(f"Missing eval split for {scene_dir.name}: {context_target_path}")

    scene_hash = scene_dir.name
    metadata_path = find_metadata_path(scene_dir, scene_hash)
    if metadata_path is None:
        raise FileNotFoundError(f"Cannot locate metadata JSON for {scene_dir}")

    if (scene_dir / "images").is_dir():
        images_dir = scene_dir / "images"
    else:
        images_dir = scene_dir

    try:
        orig_hw = _first_image_hw(images_dir)
    except FileNotFoundError:
        inferred = _hw_from_metadata(metadata_path)
        if inferred is None:
            raise
        orig_hw = inferred

    return SceneLayout(
        scene_hash=scene_hash,
        scene_dir=scene_dir,
        images_dir=images_dir,
        metadata_path=metadata_path,
        context_target_path=context_target_path,
        orig_hw=orig_hw,
    )


def discover_scenes(data_root: Path) -> list[SceneLayout]:
    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")

    candidates: list[Path] = []
    images_root = data_root / "images"
    if images_root.is_dir():
        for entry in sorted(images_root.iterdir()):
            if entry.is_dir() and split_file(entry.name).is_file():
                candidates.append(entry)
    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir() or entry.name in {"images", "metadata"}:
            continue
        if split_file(entry.name).is_file():
            candidates.append(entry)
            continue
        for sub in sorted(entry.iterdir()):
            if sub.is_dir() and split_file(sub.name).is_file():
                candidates.append(sub)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)

    scenes: list[SceneLayout] = []
    for entry in unique:
        try:
            scenes.append(resolve_scene_layout(entry))
        except FileNotFoundError:
            continue
    if not scenes:
        raise FileNotFoundError(
            f"No ACID scenes with a split in {SPLITS_DIR} under {data_root}"
        )
    return scenes


def discover_split_index(data_root: Path) -> dict[str, SceneSplit]:
    index: dict[str, SceneSplit] = {}
    for layout in discover_scenes(data_root):
        split = json.loads(layout.context_target_path.read_text())
        index[layout.scene_hash] = SceneSplit(
            scene_hash=layout.scene_hash,
            context_indices=[int(x) for x in split["context_idx"]],
            target_indices=[int(x) for x in split["target_idx"]],
            data_dir=layout.scene_dir,
            transforms_path=layout.metadata_path,
            images_dir=layout.images_dir,
            orig_hw=layout.orig_hw,
        )
    return index
