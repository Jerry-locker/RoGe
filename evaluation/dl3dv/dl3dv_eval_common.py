"""Shared DL3DV scene discovery, frame paths, and camera loading."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = _REPO_ROOT / "data" / "processed_dl3dv" / "test"
SPLITS_DIR = Path(__file__).resolve().parent / "splits"
DEFAULT_EXP_ROOT = _REPO_ROOT / "outputs" / "evals" / "dl3dv"
DEFAULT_ORIG_H = 540
DEFAULT_ORIG_W = 960
DEFAULT_VIDEO_CHUNK_SIZE = 81

# Compact summary CSV: one metric per row (average / dataset-pooled values only).
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


def read_summary_metrics_csv(csv_path: Path) -> dict[str, float]:
    if not csv_path.is_file():
        return {}
    values: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        # New format: metric,value
        if "metric" in fieldnames and "value" in fieldnames:
            for row in reader:
                key = str(row.get("metric", "")).strip()
                raw = str(row.get("value", "")).strip()
                if not key or raw == "":
                    continue
                try:
                    values[key] = float(raw)
                except ValueError:
                    continue
            return values
        # Legacy wide format: use average / dataset_pooled rows if present.
        rows = list(reader)
    average = next((r for r in rows if r.get("scene") == "average"), None)
    pooled = next((r for r in rows if r.get("scene") == "dataset_pooled"), None)
    mapping = {
        "psnr": ("average", "psnr"),
        "ssim": ("average", "ssim"),
        "lpips": ("average", "lpips"),
        "dreamsim": ("average", "dreamsim"),
        "tsed": ("average", "tsed"),
        "met3r": ("average", "met3r"),
        "fvd": ("pooled", "fvd"),
        "fid": ("pooled", "fid"),
        "transape": ("average", "ape_rmse"),
        "rotape": ("average", "rotation_ape"),
    }
    for out_key, (which, src_key) in mapping.items():
        row = average if which == "average" else pooled
        if not row:
            continue
        raw = str(row.get(src_key, "")).strip()
        if raw == "":
            continue
        try:
            values[out_key] = float(raw)
        except ValueError:
            continue
    return values


def write_summary_metrics_csv(
    csv_path: Path,
    values: dict[str, float],
    *,
    merge: bool = True,
) -> None:
    """Write ``metric,value`` CSV with one row per metric.

    Only metrics present in ``values`` (and known in ``SUMMARY_METRIC_ORDER``)
    are written — skipped metrics are omitted entirely.
    """
    merged = read_summary_metrics_csv(csv_path) if merge else {}
    for key, val in values.items():
        if key not in SUMMARY_METRIC_ORDER:
            continue
        if isinstance(val, (float, int, np.floating, np.integer)):
            merged[key] = float(val)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in SUMMARY_METRIC_ORDER:
            if key not in merged:
                continue
            val = merged[key]
            if isinstance(val, float) and math.isnan(val):
                continue
            writer.writerow({"metric": key, "value": f"{float(val):.8g}"})


@dataclass(frozen=True)
class SceneSplit:
    scene_hash: str
    context_indices: list[int]
    target_indices: list[int]
    data_dir: Path
    transforms_path: Path | None


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resolve_frame_path(directory: Path, frame_idx: int) -> Path | None:
    candidates = [
        directory / f"{frame_idx:06d}.png",
        directory / f"{frame_idx:05d}.png",
        directory / f"{frame_idx:04d}.png",
        directory / f"frame_{frame_idx + 1:05d}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    for path in candidates:
        for suffix in IMAGE_SUFFIXES:
            alt = path.with_suffix(suffix)
            if alt.exists():
                return alt
    return None


def find_transforms_path(scene_dir: Path) -> Path | None:
    candidates = [
        scene_dir / "transforms.json",
        scene_dir / "gaussian_splat" / "transforms.json",
        scene_dir / "gs" / "transforms.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def split_file(scene_hash: str) -> Path:
    return SPLITS_DIR / f"{scene_hash}.json"


def load_scene_split(scene_hash: str) -> dict:
    path = split_file(scene_hash)
    if not path.is_file():
        raise FileNotFoundError(f"Missing eval split for {scene_hash}: {path}")
    return json.loads(path.read_text())


def _scene_dirs(data_root: Path) -> list[Path]:
    """Scene directories under ``data_root`` that have a shipped split."""
    found: list[Path] = []

    def consider(path: Path) -> None:
        if find_transforms_path(path) is None or not split_file(path.name).is_file():
            return
        found.append(path)

    consider(data_root)
    if not data_root.is_dir():
        return found
    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue
        if find_transforms_path(entry) is not None:
            consider(entry)
            continue
        for sub in sorted(entry.iterdir()):
            if sub.is_dir():
                consider(sub)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def discover_split_index(data_root: Path) -> dict[str, SceneSplit]:
    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")

    index: dict[str, SceneSplit] = {}
    for scene_dir in _scene_dirs(data_root):
        split = load_scene_split(scene_dir.name)
        index[scene_dir.name] = SceneSplit(
            scene_hash=scene_dir.name,
            context_indices=[int(x) for x in split["context_idx"]],
            target_indices=[int(x) for x in split["target_idx"]],
            data_dir=scene_dir,
            transforms_path=find_transforms_path(scene_dir),
        )
    if not index:
        raise FileNotFoundError(
            f"No DL3DV scenes with a split in {SPLITS_DIR} under {data_root}"
        )
    return index


def discover_exp_scenes(exp_root: Path) -> list[Path]:
    exp_root = exp_root.resolve()
    if not exp_root.is_dir():
        raise FileNotFoundError(f"Experiment root not found: {exp_root}")
    scenes: list[Path] = []
    for entry in sorted(exp_root.iterdir()):
        if entry.name.startswith("_"):
            continue
        if not entry.is_dir():
            continue
        gt = entry / "aft_postprocess" / "gt"
        pred = entry / "aft_postprocess" / "pred"
        if gt.is_dir() and pred.is_dir():
            scenes.append(entry)
    if not scenes:
        raise FileNotFoundError(
            f"No scenes with aft_postprocess/{{gt,pred}} under {exp_root}"
        )
    return scenes


def list_frame_indices(directory: Path) -> list[int]:
    indices: list[int] = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = path.stem
        if stem.startswith("frame_"):
            try:
                indices.append(int(stem.split("_", 1)[1]) - 1)
            except ValueError:
                continue
        else:
            try:
                indices.append(int(stem))
            except ValueError:
                continue
    return sorted(set(indices))


def compute_common_crop(orig_h: int, orig_w: int) -> tuple[int, int, int, int]:
    if orig_w >= orig_h:
        crop_h = orig_h
        crop_w = orig_h
        y0 = 0
        x0 = (orig_w - crop_w) // 2
    else:
        crop_w = orig_w
        crop_h = orig_w
        x0 = 0
        y0 = (orig_h - crop_h) // 2
    return y0, x0, crop_h, crop_w


def load_eval_cameras(
    transforms_path: Path,
    frame_indices: list[int],
    *,
    image_hw: tuple[int, int],
    orig_hw: tuple[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load K and OpenCV-convention c2w for aft_postprocess frames."""
    transforms = json.loads(transforms_path.read_text())
    frames = transforms["frames"]
    src_w = float(transforms.get("w", orig_hw[1]))
    src_h = float(transforms.get("h", orig_hw[0]))
    sx0 = orig_hw[1] / src_w
    sy0 = orig_hw[0] / src_h
    fx = float(transforms["fl_x"]) * sx0
    fy = float(transforms["fl_y"]) * sy0
    cx = float(transforms["cx"]) * sx0
    cy = float(transforms["cy"]) * sy0

    y0, x0, crop_h, crop_w = compute_common_crop(orig_hw[0], orig_hw[1])
    cx = cx - x0
    cy = cy - y0
    eval_h, eval_w = image_hw
    sx = eval_w / float(crop_w)
    sy = eval_h / float(crop_h)
    fx *= sx
    fy *= sy
    cx *= sx
    cy *= sy
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    Ks: list[np.ndarray] = []
    c2ws: list[np.ndarray] = []
    for idx in frame_indices:
        if idx < 0 or idx >= len(frames):
            raise IndexError(
                f"Frame index {idx} out of range for transforms ({len(frames)} frames) "
                f"in {transforms_path}"
            )
        c2w = np.array(frames[idx]["transform_matrix"], dtype=np.float64)
        c2w = c2w.copy()
        c2w[:3, 1:3] *= -1.0
        Ks.append(K.copy())
        c2ws.append(c2w)
    return Ks, c2ws


def chunk_sequence(
    items: list, chunk_size: int, *, drop_incomplete: bool = False
) -> list[list]:
    """Split ``items`` into contiguous chunks of ``chunk_size``.

    If ``drop_incomplete`` is True and the final chunk is shorter than
    ``chunk_size``, it is discarded (no-op when ``chunk_size <= 0``).
    """
    if chunk_size <= 0:
        return [items]
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
    if drop_incomplete and chunks and len(chunks[-1]) < chunk_size:
        chunks = chunks[:-1]
    return chunks


def mean_or_nan(values: list[float]) -> float:
    vals = [v for v in values if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def collect_video_frame_paths(
    gt_dir: Path,
    pred_dir: Path,
    video_indices: list[int],
) -> tuple[list[Path], list[Path], list[int], list[int]]:
    gt_paths, pred_paths, loaded, missing = [], [], [], []
    for idx in video_indices:
        gt_path = resolve_frame_path(gt_dir, idx)
        pred_path = resolve_frame_path(pred_dir, idx)
        if gt_path is None or pred_path is None:
            missing.append(idx)
            continue
        gt_paths.append(gt_path)
        pred_paths.append(pred_path)
        loaded.append(idx)
    return gt_paths, pred_paths, loaded, missing
