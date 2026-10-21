"""Shared RE10K (LagerNVS layout) scene discovery, frame paths, and camera loading.

Expects ``--data_root`` like::

    processed_re10k/benchmark/
    ├── images/<seq_id>/*.png
    ├── images/<seq_id>/context_target_6.json
    ├── metadata/<seq_id>.json   # frames: fxfycxcy (px) + w2c (4x4)
    └── full_list.txt

Also accepts a flat layout ``<data_root>/<seq_id>/{images,metadata.json,context_target_6.json}``.
"""

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
DATA_ROOT = _REPO_ROOT / "data" / "processed_re10k" / "benchmark"
DEFAULT_EXP_ROOT = _REPO_ROOT / "outputs" / "evals" / "re10k"
# Most common RE10K resolution in mutou/DavidYan exports (fallback only).
DEFAULT_ORIG_H = 360
DEFAULT_ORIG_W = 640
DEFAULT_VIDEO_CHUNK_SIZE = 81

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
    transforms_path: Path | None  # metadata/<seq>.json
    images_dir: Path
    orig_hw: tuple[int, int]


@dataclass(frozen=True)
class SceneLayout:
    scene_hash: str
    scene_dir: Path  # directory containing context_target_6.json (usually images/<seq>)
    images_dir: Path
    metadata_path: Path
    context_target_path: Path
    orig_hw: tuple[int, int]


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


def _first_image_hw(images_dir: Path) -> tuple[int, int]:
    for name in ("00000.png", "0000.png", "0.png"):
        path = images_dir / name
        if path.is_file():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                return h, w
    pngs = sorted(images_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No png images under {images_dir}")
    img = cv2.imread(str(pngs[0]), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read {pngs[0]}")
    h, w = img.shape[:2]
    return h, w


def _infer_hw_from_metadata(metadata_path: Path) -> tuple[int, int] | None:
    """Fallback: RE10K principal point is image center → W=2*cx, H=2*cy."""
    try:
        data = json.loads(metadata_path.read_text())
        frames = data.get("frames") or []
        if not frames:
            return None
        fx, fy, cx, cy = [float(x) for x in frames[0]["fxfycxcy"]]
        w = int(round(2.0 * cx))
        h = int(round(2.0 * cy))
        if w > 0 and h > 0:
            return h, w
    except Exception:
        return None
    return None


def find_metadata_path(scene_dir: Path, scene_hash: str | None = None) -> Path | None:
    scene_hash = scene_hash or scene_dir.name
    candidates = [
        scene_dir / "metadata.json",
        scene_dir / f"{scene_hash}.json",
        scene_dir.parent.parent / "metadata" / f"{scene_hash}.json",
        scene_dir.parent / "metadata" / f"{scene_hash}.json",
    ]
    # images/<seq> → benchmark/metadata/<seq>.json
    if scene_dir.parent.name == "images":
        candidates.insert(
            0, scene_dir.parent.parent / "metadata" / f"{scene_hash}.json"
        )
    for path in candidates:
        if path.is_file():
            return path
    return None


def resolve_scene_layout(scene_dir: Path) -> SceneLayout:
    """Resolve a scene directory that contains ``context_target_6.json``."""
    scene_dir = scene_dir.resolve()
    context_target_path = scene_dir / "context_target_6.json"
    if not context_target_path.is_file():
        raise FileNotFoundError(f"Missing context_target_6.json in {scene_dir}")

    scene_hash = scene_dir.name
    metadata_path = find_metadata_path(scene_dir, scene_hash)
    if metadata_path is None:
        raise FileNotFoundError(f"Cannot locate metadata JSON for {scene_dir}")

    if (scene_dir / "images").is_dir():
        images_dir = scene_dir / "images"
    else:
        # context_target lives inside images/<seq>
        images_dir = scene_dir

    try:
        orig_hw = _first_image_hw(images_dir)
    except FileNotFoundError:
        inferred = _infer_hw_from_metadata(metadata_path)
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
            if entry.is_dir() and (entry / "context_target_6.json").is_file():
                candidates.append(entry)
    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir() or entry.name in {"images", "metadata"}:
            continue
        if (entry / "context_target_6.json").is_file():
            candidates.append(entry)
            continue
        for sub in sorted(entry.iterdir()):
            if sub.is_dir() and (sub / "context_target_6.json").is_file():
                candidates.append(sub)

    # Deduplicate by resolved path.
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
            f"No RE10K scenes with context_target_6.json found under {data_root}"
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
    """Load K and OpenCV c2w for aft_postprocess frames from RE10K metadata.

    Metadata stores pixel ``fxfycxcy`` + ``w2c`` at the original image
    resolution. Cameras are mapped to the RoGe eval FOV via the same
    square common-crop + resize used for DL3DV aft_postprocess.
    """
    data = json.loads(transforms_path.read_text())
    frames = data["frames"]
    if not frames:
        raise ValueError(f"Empty frames in {transforms_path}")

    fx0, fy0, cx0, cy0 = [float(x) for x in frames[0]["fxfycxcy"]]
    # Prefer explicit orig_hw; fall back to principal-point inference.
    orig_h, orig_w = orig_hw
    if orig_h <= 0 or orig_w <= 0:
        orig_w = int(round(2.0 * cx0))
        orig_h = int(round(2.0 * cy0))

    y0, x0, crop_h, crop_w = compute_common_crop(orig_h, orig_w)
    eval_h, eval_w = image_hw
    sx = eval_w / float(crop_w)
    sy = eval_h / float(crop_h)

    Ks: list[np.ndarray] = []
    c2ws: list[np.ndarray] = []
    for idx in frame_indices:
        if idx < 0 or idx >= len(frames):
            raise IndexError(
                f"Frame index {idx} out of range for metadata ({len(frames)} frames) "
                f"in {transforms_path}"
            )
        fx, fy, cx, cy = [float(x) for x in frames[idx]["fxfycxcy"]]
        cx = cx - x0
        cy = cy - y0
        K = np.array(
            [
                [fx * sx, 0.0, cx * sx],
                [0.0, fy * sy, cy * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        w2c = np.array(frames[idx]["w2c"], dtype=np.float64)
        c2w = np.linalg.inv(w2c)
        Ks.append(K)
        c2ws.append(c2w)
    return Ks, c2ws


def chunk_sequence(
    items: list, chunk_size: int, *, drop_incomplete: bool = False
) -> list[list]:
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
