#!/usr/bin/env python3
"""DL3DV-10K benchmark batch inference for RoGe experiments.

Pipeline per scene:
1. Undistort all frames (540x960).
2. Cover-crop to 192x336 (+ matching intrinsics).
3. Generate all non-context frames in ordered chunks (context from evaluation/dl3dv/splits).
4. Save bef_postprocess (192x336) and aft_postprocess (192x192 column-crop eval FOV).
5. No metrics — output layout matches dl3dv_3dgs_exp.py for fair comparison.
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
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from diffsynth.model import RoGe, preprocess_images_and_intrinsics, resize_and_center_crop
from dl3dv_eval_common import SPLITS_DIR, load_scene_split, split_file


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

DATA_ROOT_DEFAULT = str(REPO_ROOT / "data" / "processed_dl3dv" / "test")

ORIG_H, ORIG_W = 540, 960
TRAIN_H, TRAIN_W = 192, 336
EVAL_H, EVAL_W = 192, 192


@dataclass(frozen=True)
class FrameRecord:
    idx: int
    label: str
    image_path: Path
    w2c: np.ndarray
    intrinsic: np.ndarray


@dataclass(frozen=True)
class SceneResult:
    scene: str
    scene_dir: str
    output_dir: str
    num_saved_frames: int


def compute_common_crop(orig_h: int, orig_w: int) -> tuple[int, int, int, int]:
    if orig_w >= orig_h:
        crop_h = crop_w = orig_h
        y0 = 0
        x0 = (orig_w - crop_w) // 2
    else:
        crop_h = crop_w = orig_w
        x0 = 0
        y0 = (orig_h - crop_h) // 2
    return y0, x0, crop_h, crop_w


def _fc_resize_params(
    orig_h: int, orig_w: int, tgt_h: int, tgt_w: int
) -> tuple[int, int, int, int]:
    tgt_aspect = tgt_w / tgt_h
    src_aspect = orig_w / orig_h
    if src_aspect > tgt_aspect:
        resize_h = tgt_h
        resize_w = int(round(orig_w * tgt_h / orig_h))
    else:
        resize_w = tgt_w
        resize_h = int(round(orig_h * tgt_w / orig_w))
    offset_x = (resize_w - tgt_w) // 2
    offset_y = (resize_h - tgt_h) // 2
    return resize_h, resize_w, offset_x, offset_y


def compute_roge_col_range(
    *,
    orig_h: int = ORIG_H,
    orig_w: int = ORIG_W,
    roge_tgt_h: int = TRAIN_H,
    roge_tgt_w: int = TRAIN_W,
) -> tuple[int, int]:
    """Map common-crop x-range on original image to columns in RoGe 192x336 pred."""
    _y0, x0, _crop_h, crop_w = compute_common_crop(orig_h, orig_w)
    _resize_h, resize_w, offset_x, _offset_y = _fc_resize_params(
        orig_h, orig_w, roge_tgt_h, roge_tgt_w
    )
    sx = resize_w / orig_w
    col0 = int(round(x0 * sx - offset_x))
    col1 = int(round((x0 + crop_w) * sx - offset_x))
    col0 = max(0, col0)
    col1 = min(roge_tgt_w, max(col0 + 1, col1))
    return col0, col1


def postprocess_roge(
    img: np.ndarray,
    *,
    roge_tgt_h: int = TRAIN_H,
    roge_tgt_w: int = TRAIN_W,
) -> np.ndarray:
    """Column-crop an RoGe frame to eval FOV, then resize to 192x192."""
    col0, col1 = compute_roge_col_range(
        roge_tgt_h=roge_tgt_h, roge_tgt_w=roge_tgt_w
    )
    ih, iw = img.shape[:2]
    if (ih, iw) != (roge_tgt_h, roge_tgt_w):
        raise ValueError(
            f"RoGe frame {(ih, iw)} != expected {(roge_tgt_h, roge_tgt_w)}"
        )
    if col1 > iw:
        raise ValueError(f"RoGe column crop [{col0}:{col1}] exceeds width {iw}")
    cropped = img[:, col0:col1]
    ch, cw = cropped.shape[:2]
    upsample = (ch < EVAL_H) or (cw < EVAL_W)
    interp = cv2.INTER_LINEAR if upsample else cv2.INTER_AREA
    return cv2.resize(cropped, (EVAL_W, EVAL_H), interpolation=interp)


def _expand_paths(values: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                paths.append(Path(part).expanduser().resolve())
    return paths


def _discover_scenes(inputs: list[Path]) -> list[Path]:
    scenes: list[Path] = []
    for path in inputs:
        if (path / "gaussian_splat" / "transforms.json").is_file():
            if split_file(path.name).is_file():
                scenes.append(path)
            continue
        if not path.is_dir():
            raise NotADirectoryError(f"Scene root is not a directory: {path}")
        for child in sorted(path.iterdir()):
            if (
                child.is_dir()
                and (child / "gaussian_splat" / "transforms.json").is_file()
                and split_file(child.name).is_file()
            ):
                scenes.append(child)
    if not scenes:
        raise FileNotFoundError(
            "No DL3DV benchmark scenes found. Expected <scene>/gaussian_splat/transforms.json "
            f"for a split under {SPLITS_DIR}"
        )
    return scenes


def _safe_prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir = output_dir.resolve()
    if output_dir in (Path("/"), Path.home()):
        raise ValueError(f"Refusing unsafe output_dir: {output_dir}")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _prepare_output_dir_multi_node(
    output_dir: Path,
    overwrite: bool,
    *,
    machine_rank: int,
    num_machines: int,
    wait_timeout_sec: float = 600.0,
) -> None:
    if num_machines <= 1:
        _safe_prepare_output_dir(output_dir, overwrite)
        return

    ready_marker = output_dir / ".multi_node_ready"
    if machine_rank == 0:
        _safe_prepare_output_dir(output_dir, overwrite)
        ready_marker.touch()
        return

    deadline = time.time() + wait_timeout_sec
    while not ready_marker.exists():
        if time.time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for multi-node ready marker from machine_rank=0: {ready_marker}"
            )
        time.sleep(1.0)


def _frame_image_path(scene_dir: Path, file_path: str) -> Path:
    gs_dir = scene_dir / "gaussian_splat"
    src = Path(file_path)
    candidates = []
    if src.name:
        candidates.append(gs_dir / "images_4" / src.name)
    candidates.append(gs_dir / str(src).replace("images/", "images_4/", 1))
    candidates.append(gs_dir / "images_4" / src)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    suffixes = ", ".join(IMAGE_SUFFIXES)
    raise FileNotFoundError(
        f"Missing downsampled image for {file_path!r}. Looked under "
        f"{gs_dir / 'images_4'} for {src.name or '*'} ({suffixes})."
    )


def _base_intrinsic(transforms: dict, image_w: int, image_h: int) -> np.ndarray:
    src_w = float(transforms.get("w", image_w))
    src_h = float(transforms.get("h", image_h))
    sx = image_w / src_w
    sy = image_h / src_h
    return np.array(
        [
            [float(transforms["fl_x"]) * sx, 0.0, float(transforms["cx"]) * sx],
            [0.0, float(transforms["fl_y"]) * sy, float(transforms["cy"]) * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _distortion(transforms: dict) -> np.ndarray:
    return np.array(
        [
            float(transforms.get("k1", 0.0)),
            float(transforms.get("k2", 0.0)),
            float(transforms.get("p1", 0.0)),
            float(transforms.get("p2", 0.0)),
            float(transforms.get("k3", 0.0)),
        ],
        dtype=np.float32,
    )


def _load_undistorted_image(path: Path, k: np.ndarray, dist: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    h, w = bgr.shape[:2]
    new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist, (w, h), alpha=0.0, newImgSize=(w, h))
    undistorted = cv2.undistort(bgr, k, dist, None, new_k)
    rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), new_k.astype(np.float32)


def _cover_crop_image(image: Image.Image, height: int, width: int) -> Image.Image:
    """Cover-resize + center-crop undistorted RGB to model resolution."""
    img_np = np.asarray(image.convert("RGB"))
    cropped, _ = resize_and_center_crop(img_np, height, width)
    return Image.fromarray(cropped)


def _w2c_opencv_from_transforms_c2w(c2w: np.ndarray, *, opengl_to_opencv: bool) -> np.ndarray:
    w2c = np.linalg.inv(c2w.astype(np.float64)).astype(np.float32)
    if opengl_to_opencv:
        w2c[[1, 2], :] *= -1.0
    return w2c


def _load_scene_frames(scene_dir: Path, *, opengl_to_opencv: bool) -> list[FrameRecord]:
    transforms_path = scene_dir / "gaussian_splat" / "transforms.json"
    with transforms_path.open("r", encoding="utf-8") as f:
        transforms = json.load(f)
    raw_frames = transforms.get("frames", [])
    if not raw_frames:
        raise ValueError(f"No frames in {transforms_path}")

    first_image = _frame_image_path(scene_dir, raw_frames[0]["file_path"])
    with Image.open(first_image) as img:
        image_w, image_h = img.size
    k_scaled = _base_intrinsic(transforms, image_w, image_h)
    dist = _distortion(transforms)

    frames: list[FrameRecord] = []
    for idx, frame in enumerate(raw_frames):
        image_path = _frame_image_path(scene_dir, frame["file_path"])
        _, undistorted_k = _load_undistorted_image(image_path, k_scaled, dist)
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
        if c2w.shape != (4, 4):
            raise ValueError(f"Bad transform_matrix shape for frame {idx}: {c2w.shape}")
        w2c = _w2c_opencv_from_transforms_c2w(c2w, opengl_to_opencv=opengl_to_opencv)
        frames.append(
            FrameRecord(
                idx=idx,
                label=Path(frame["file_path"]).stem or f"frame_{idx:06d}",
                image_path=image_path,
                w2c=w2c,
                intrinsic=undistorted_k,
            )
        )
    return frames


def _read_context_indices(scene_dir: Path) -> list[int]:
    data = load_scene_split(scene_dir.name)
    return [int(x) for x in data["context_idx"]]


def _ensure_indices(indices: list[int], n: int, label: str) -> None:
    bad = [idx for idx in indices if idx < 0 or idx >= n]
    if bad:
        raise IndexError(f"{label} contains out-of-range indices for {n} frames: {bad[:10]}")


def _save_pil(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def _save_rgb_np(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise IOError(f"Failed to write image: {path}")


def _chunks(values: list[int], chunk_size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _parse_gpus(gpus: str | None) -> list[str]:
    if not gpus:
        return []
    parsed: list[str] = []
    for part in gpus.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            step = 1 if end >= start else -1
            parsed.extend(str(i) for i in range(start, end + step, step))
        else:
            parsed.append(part)
    return parsed


def _tail_text(path: Path, max_lines: int = 80) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        return f"(failed to read log {path}: {exc})"
    if not lines:
        return "(worker log is empty)"
    return "".join(lines[-max_lines:]).rstrip()


def _scene_done(scene_out: Path, n_expected: int) -> bool:
    pred = scene_out / "aft_postprocess" / "pred"
    if not pred.is_dir():
        return False
    return len(list(pred.glob("*.png"))) >= n_expected


def _inference_done_path(output_dir: Path, machine_rank: int) -> Path:
    return output_dir / f".inference_rank{machine_rank}.done"


def _mark_inference_done(output_dir: Path, machine_rank: int) -> None:
    _inference_done_path(output_dir, machine_rank).touch()


def _wait_all_inference_done(
    output_dir: Path,
    num_machines: int,
    *,
    wait_timeout_sec: float = 172800.0,
) -> None:
    deadline = time.time() + wait_timeout_sec
    while True:
        missing = [
            rank
            for rank in range(num_machines)
            if not _inference_done_path(output_dir, rank).exists()
        ]
        if not missing:
            return
        if time.time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for inference done markers under {output_dir}; "
                f"still missing ranks={missing}"
            )
        time.sleep(5.0)


def _run_post_inference_metrics(output_dir: Path, *, skip_ape: bool = False) -> None:
    metrics_script = SCRIPT_DIR / "dl3dv_run_metrics.sh"
    if not metrics_script.is_file():
        raise FileNotFoundError(f"Metrics runner not found: {metrics_script}")
    print(f"\n=== Running DL3DV unified metrics on {output_dir} ===", flush=True)
    subprocess.run(
        ["bash", str(metrics_script), str(output_dir)],
        cwd=str(REPO_ROOT),
        check=True,
    )

    if skip_ape:
        print("Skipping unified DL3DV APE (--skip_ape).")
        return

    ape_script = SCRIPT_DIR / "dl3dv_run_ape.sh"
    if not ape_script.is_file():
        raise FileNotFoundError(f"APE runner not found: {ape_script}")
    print(f"\n=== Running DL3DV unified APE on {output_dir} ===", flush=True)
    subprocess.run(
        ["bash", str(ape_script), str(output_dir)],
        cwd=str(REPO_ROOT),
        check=True,
    )


def run_scene(args, model: RoGe) -> SceneResult:
    scene_dir = args._scene_dir
    scene_name = scene_dir.name
    scene_out = Path(args.output_dir) / scene_name

    frames = _load_scene_frames(scene_dir, opengl_to_opencv=not args.no_opengl_to_opencv)
    context_idx = _read_context_indices(scene_dir)
    _ensure_indices(context_idx, len(frames), "context_idx")

    context_set = set(context_idx)
    all_target_idx = [idx for idx in range(len(frames)) if idx not in context_set]

    n_expected = len(all_target_idx)
    if not args.overwrite and _scene_done(scene_out, n_expected):
        print(f"[skip] {scene_name}: already complete ({n_expected} frames)")
        return SceneResult(
            scene=scene_name,
            scene_dir=str(scene_dir),
            output_dir=str(scene_out),
            num_saved_frames=n_expected,
        )

    scene_out.mkdir(parents=True, exist_ok=True)
    bef_pred_dir = scene_out / "bef_postprocess" / "pred"
    bef_gt_dir = scene_out / "bef_postprocess" / "gt"
    aft_pred_dir = scene_out / "aft_postprocess" / "pred"
    aft_gt_dir = scene_out / "aft_postprocess" / "gt"

    print(
        f"\n=== {scene_name}: context={len(context_idx)}, "
        f"non_context={len(all_target_idx)} ==="
    )

    transforms_path = scene_dir / "gaussian_splat" / "transforms.json"
    with transforms_path.open("r", encoding="utf-8") as f:
        transforms = json.load(f)
    first_image = frames[0].image_path
    with Image.open(first_image) as img:
        image_w, image_h = img.size
    k_scaled = _base_intrinsic(transforms, image_w, image_h)
    dist = _distortion(transforms)
    undistorted_cache: dict[int, Image.Image] = {}

    def load_undistorted(frame_idx: int) -> tuple[Image.Image, np.ndarray]:
        rec = frames[frame_idx]
        if frame_idx not in undistorted_cache:
            img, _ = _load_undistorted_image(rec.image_path, k_scaled, dist)
            undistorted_cache[frame_idx] = img
        return undistorted_cache[frame_idx], rec.intrinsic

    context_images_raw = [load_undistorted(idx)[0] for idx in context_idx]
    context_w2c = np.stack([frames[idx].w2c for idx in context_idx], axis=0).astype(np.float32)
    context_k = np.stack([frames[idx].intrinsic for idx in context_idx], axis=0).astype(np.float32)

    saved_count = 0

    for chunk_id, target_chunk in enumerate(_chunks(all_target_idx, args.chunk_size)):
        real_target_chunk = list(target_chunk)
        if not real_target_chunk:
            continue
        pad_count = max(0, args.chunk_size - len(real_target_chunk))
        inference_target_chunk = real_target_chunk + [real_target_chunk[-1]] * pad_count
        target_w2c = np.stack([frames[idx].w2c for idx in inference_target_chunk], axis=0).astype(np.float32)
        target_k = np.stack([frames[idx].intrinsic for idx in inference_target_chunk], axis=0).astype(np.float32)
        w2c = np.concatenate([context_w2c, target_w2c], axis=0)
        intrinsics = np.concatenate([context_k, target_k], axis=0)

        context_images = [img.copy() for img in context_images_raw]
        context_images, intrinsics_model = preprocess_images_and_intrinsics(
            context_images,
            intrinsics,
            (args.orig_w, args.orig_h),
            args.height,
            args.width,
            "crop",
        )
        generate_kwargs = dict(
            images=context_images,
            w2c_poses=w2c,
            intrinsics=intrinsics_model,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            cfg_scale=args.cfg_scale,
            resize_mode=None,
            pose_gauge=args.pose_gauge,
        )
        if getattr(model, "lagernvs_condition", None) is not None:
            generate_kwargs["lagernvs_context_images"] = [img.copy() for img in context_images_raw]
            generate_kwargs["lagernvs_intrinsics"] = intrinsics
        video = model.generate(**generate_kwargs)
        pred_frames = list(video)

        expected_frames = len(context_idx) + len(inference_target_chunk)
        if len(pred_frames) != expected_frames:
            raise RuntimeError(
                f"Model returned {len(pred_frames)} frames, expected {expected_frames} "
                f"(context={len(context_idx)}, target={len(inference_target_chunk)}, "
                f"real_target={len(real_target_chunk)}, pad={pad_count})"
            )

        for local_i, frame_idx in enumerate(real_target_chunk):
            pred = pred_frames[len(context_idx) + local_i]
            gt_full, _ = load_undistorted(frame_idx)
            gt_bef = _cover_crop_image(gt_full, args.height, args.width)
            name = f"{frame_idx:06d}.png"

            _save_pil(bef_pred_dir / name, pred)
            _save_pil(bef_gt_dir / name, gt_bef)

            pred_aft = postprocess_roge(
                np.asarray(pred.convert("RGB")),
                roge_tgt_h=args.height,
                roge_tgt_w=args.width,
            )
            gt_aft = postprocess_roge(
                np.asarray(gt_bef.convert("RGB")),
                roge_tgt_h=args.height,
                roge_tgt_w=args.width,
            )
            _save_rgb_np(aft_pred_dir / name, pred_aft)
            _save_rgb_np(aft_gt_dir / name, gt_aft)
            saved_count += 1

    if saved_count != n_expected:
        raise RuntimeError(
            f"Expected to save {n_expected} non-context frames for {scene_name}, got {saved_count}"
        )

    manifest = {
        "scene": scene_name,
        "scene_dir": str(scene_dir),
        "context_idx": context_idx,
        "num_saved_frames": saved_count,
    }
    _save_json(scene_out / "manifest.json", manifest)
    print(f"[done] {scene_name}: saved {saved_count} non-context frames")

    return SceneResult(
        scene=scene_name,
        scene_dir=str(scene_dir),
        output_dir=str(scene_out),
        num_saved_frames=saved_count,
    )


def _append_common_worker_args(cmd: list[str], args) -> None:
    cmd.extend(
        [
            "--output_dir",
            str(args.output_dir),
            "--checkpoint_path",
            str(args.checkpoint_path),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--orig_h",
            str(args.orig_h),
            "--orig_w",
            str(args.orig_w),
            "--chunk_size",
            str(args.chunk_size),
            "--num_inference_steps",
            str(args.num_inference_steps),
            "--seed",
            str(args.seed),
            "--cfg_scale",
            str(args.cfg_scale),
            "--device",
            "cuda" if "cuda" in str(args.device) else str(args.device),
        ]
    )
    if args.overwrite:
        cmd.append("--overwrite")
    if args.vram_limit is not None:
        cmd.extend(["--vram_limit", str(args.vram_limit)])
    if args.base_model_dir is not None:
        cmd.extend(["--base_model_dir", str(args.base_model_dir)])
    if args.lagernvs_ckpt is not None:
        cmd.extend(["--lagernvs_ckpt", str(args.lagernvs_ckpt)])
    if getattr(args, "lagernvs_channels", None) is not None:
        cmd.extend(["--lagernvs_channels", str(args.lagernvs_channels)])
    if args.hybrid_vae_encoding:
        cmd.append("--hybrid_vae_encoding")
    if getattr(args, "raymap_pe", False):
        cmd.append("--raymap_pe")
    if getattr(args, "geom_pe", False):
        cmd.append("--geom_pe")
    if getattr(args, "joint_train", False):
        cmd.append("--joint_train")
    if getattr(args, "train_reconstructor_lora", False):
        cmd.append("--train_reconstructor_lora")
    if getattr(args, "lagernvs_lora_rank", None) is not None:
        cmd.extend(["--lagernvs_lora_rank", str(args.lagernvs_lora_rank)])
    if args.no_opengl_to_opencv:
        cmd.append("--no_opengl_to_opencv")


def _roge_from_args(args) -> RoGe:
    return RoGe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        vram_limit=args.vram_limit,
        base_model_dir=args.base_model_dir,
        hybrid_vae_encoding=args.hybrid_vae_encoding,
        raymap_pe=getattr(args, "raymap_pe", False),
        geom_pe=getattr(args, "geom_pe", False),
        lagernvs_ckpt=args.lagernvs_ckpt,
        lagernvs_channels=args.lagernvs_channels,
        joint_train=getattr(args, "joint_train", False),
        lagernvs_lora_rank=getattr(args, "lagernvs_lora_rank", 8),
        train_reconstructor_lora=getattr(args, "train_reconstructor_lora", False),
    )


def _run_worker_scene(args, scene_dir: Path, result_path: Path) -> SceneResult:
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    args._scene_dir = scene_dir
    model = _roge_from_args(args)
    result = run_scene(args, model)
    _save_json(result_path, result.__dict__)
    return result


def run_parallel(args, scenes: list[Path], gpus: list[str]) -> list[SceneResult]:
    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    result_dir = output_dir / ".worker_results"
    logs_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    pending = list(scenes)
    running: dict[subprocess.Popen, tuple[Path, str, Path, object]] = {}
    free_gpus = list(gpus)
    completed: list[SceneResult] = []

    print(f"Parallel mode: {len(scenes)} scene(s), GPUs={','.join(gpus)}")
    while pending or running:
        while pending and free_gpus:
            scene_dir = pending.pop(0)
            gpu = free_gpus.pop(0)
            result_path = result_dir / f"{scene_dir.name}.json"
            log_path = logs_dir / f"{scene_dir.name}.gpu{gpu}.log"
            cmd = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--worker_scene",
                str(scene_dir),
                "--worker_result",
                str(result_path),
            ]
            _append_common_worker_args(cmd, args)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
            env.setdefault("PYTHONUNBUFFERED", "1")
            log_f = log_path.open("w", encoding="utf-8")
            print(f"[launch] gpu={gpu} scene={scene_dir.name} log={log_path}")
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[proc] = (scene_dir, gpu, result_path, log_f)
            if args.launch_interval > 0 and (pending and free_gpus):
                time.sleep(float(args.launch_interval))

        finished = []
        for proc, item in running.items():
            if proc.poll() is not None:
                finished.append((proc, item))

        if not finished:
            time.sleep(5.0)
            continue

        for proc, (scene_dir, gpu, result_path, log_f) in finished:
            log_f.close()
            del running[proc]
            free_gpus.append(gpu)
            if proc.returncode != 0:
                for other_proc, (_, _, _, other_log) in list(running.items()):
                    other_proc.terminate()
                    other_log.close()
                tail = _tail_text(logs_dir / f"{scene_dir.name}.gpu{gpu}.log")
                raise RuntimeError(
                    f"Worker failed for scene={scene_dir.name} on gpu={gpu} "
                    f"(returncode={proc.returncode}); "
                    f"see {logs_dir / f'{scene_dir.name}.gpu{gpu}.log'}\n\n"
                    f"--- worker log tail ---\n{tail}\n--- end worker log tail ---"
                )
            with result_path.open("r", encoding="utf-8") as f:
                completed.append(SceneResult(**json.load(f)))
            print(f"[done] gpu={gpu} scene={scene_dir.name}")

    order = {scene.name: i for i, scene in enumerate(scenes)}
    completed.sort(key=lambda r: order.get(r.scene, 10**9))
    return completed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch RoGe inference for DL3DV-10K benchmark gaussian_splat scenes."
    )
    parser.add_argument(
        "--scene_root",
        nargs="+",
        default=[DATA_ROOT_DEFAULT],
        help=f"Scene dirs or roots containing scene dirs (default: {DATA_ROOT_DEFAULT}).",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--height", type=int, default=TRAIN_H)
    parser.add_argument("--width", type=int, default=TRAIN_W)
    parser.add_argument("--orig_h", type=int, default=ORIG_H)
    parser.add_argument("--orig_w", type=int, default=ORIG_W)
    parser.add_argument("--chunk_size", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vram_limit", type=float, default=None)
    parser.add_argument("--base_model_dir", type=str, default=None)
    parser.add_argument("--lagernvs_ckpt", type=str, required=True)
    parser.add_argument(
        "--lagernvs_channels",
        "--lagernvs_geom_channels",
        type=int,
        default=384,
        dest="lagernvs_channels",
    )
    parser.add_argument(
        "--joint_train",
        action="store_true",
        help="Load LagerNVS renderer LoRA from checkpoint "
             "(match training --joint_train).",
    )
    parser.add_argument(
        "--train_reconstructor_lora",
        action="store_true",
        help="With --joint_train: also inject/load reconstructor LoRA "
             "(match training --train_reconstructor_lora).",
    )
    parser.add_argument(
        "--lagernvs_lora_rank",
        type=int,
        default=8,
        help="LoRA rank for LagerNVS renderer/reconstructor under --joint_train "
             "(default 8).",
    )
    parser.add_argument("--hybrid_vae_encoding", action="store_true")
    parser.add_argument("--raymap_pe", action="store_true")
    parser.add_argument("--geom_pe", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip unified_metrics_dl3dv.py and unified_ape_dl3dv.py after inference "
        "(default: run on rank 0).",
    )
    parser.add_argument(
        "--skip_ape",
        action="store_true",
        help="Skip unified_ape_dl3dv.py after metrics (default: run on rank 0).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids for parallel scene inference, e.g. 0,1,2,3 or 0-7.",
    )
    parser.add_argument(
        "--launch_interval",
        type=float,
        default=0.0,
        help="Seconds to wait between launching parallel workers.",
    )
    parser.add_argument(
        "--num_machines",
        type=int,
        default=1,
        help="Number of machines for multi-node scene sharding (Volcano: MLP_WORKER_NUM).",
    )
    parser.add_argument(
        "--machine_rank",
        type=int,
        default=0,
        help="This machine's rank in [0, num_machines) (Volcano: MLP_ROLE_INDEX).",
    )
    parser.add_argument("--worker_scene", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_result", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no_opengl_to_opencv",
        action="store_true",
        help="Treat transform_matrix as OpenCV c2w already. Default converts Nerfstudio/OpenGL c2w to OpenCV.",
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk_size must be positive")
    if args.num_machines < 1:
        parser.error("--num_machines must be >= 1")
    if args.machine_rank < 0 or args.machine_rank >= args.num_machines:
        parser.error(f"--machine_rank must be in [0, {args.num_machines})")
    if args.worker_scene is None and not args.scene_root:
        parser.error("--scene_root is required")
    if args.worker_scene is not None and args.worker_result is None:
        parser.error("--worker_result is required with --worker_scene")
    if getattr(args, "train_reconstructor_lora", False) and not getattr(
        args, "joint_train", False
    ):
        parser.error("--train_reconstructor_lora requires --joint_train")
    args.pose_gauge = "cam_first_lagernvs"
    return args


def main() -> None:
    args = parse_args()

    if args.worker_scene is not None:
        _run_worker_scene(
            args,
            args.worker_scene.expanduser().resolve(),
            args.worker_result.expanduser().resolve(),
        )
        return

    output_dir = Path(args.output_dir).expanduser().resolve()
    _prepare_output_dir_multi_node(
        output_dir,
        args.overwrite,
        machine_rank=args.machine_rank,
        num_machines=args.num_machines,
    )
    args.output_dir = str(output_dir)

    scene_inputs = _expand_paths(args.scene_root)
    scenes = _discover_scenes(scene_inputs)
    print(f"Discovered {len(scenes)} scene(s) from {scene_inputs[0]}")
    if args.num_machines > 1:
        scenes = scenes[args.machine_rank :: args.num_machines]
        print(
            f"Multi-node shard: machine_rank={args.machine_rank}/{args.num_machines}, "
            f"local_scenes={len(scenes)}"
        )

    gpus = _parse_gpus(args.gpus)
    if gpus:
        results = run_parallel(args, scenes, gpus)
    else:
        model = _roge_from_args(args)
        results = []
        for scene_dir in scenes:
            args._scene_dir = scene_dir
            results.append(run_scene(args, model))

    total_frames = sum(r.num_saved_frames for r in results)
    print(
        f"\nFinished {len(results)} scene(s), {total_frames} non-context frames saved under {output_dir}"
    )

    _mark_inference_done(output_dir, args.machine_rank)
    if args.num_machines > 1:
        print(
            f"Machine {args.machine_rank}/{args.num_machines}: inference done; "
            "waiting for other ranks...",
            flush=True,
        )
        _wait_all_inference_done(output_dir, args.num_machines)

    if args.skip_metrics:
        if args.machine_rank == 0:
            print("Skipping unified DL3DV metrics/APE (--skip_metrics).")
        return

    if args.machine_rank == 0:
        _run_post_inference_metrics(output_dir, skip_ape=args.skip_ape)
    elif args.num_machines > 1:
        print(
            f"Machine {args.machine_rank}: skipping metrics/APE; rank 0 owns "
            f"{output_dir / 'unified_metrics.csv'} and "
            f"{output_dir / 'unified_ape.csv'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
