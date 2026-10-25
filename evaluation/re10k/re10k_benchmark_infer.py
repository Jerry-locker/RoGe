#!/usr/bin/env python3
"""RE10K benchmark batch inference for RoGe (w/ LagerNVS).

Pipeline per scene (LagerNVS-layout RE10K under processed_re10k/benchmark):
1. Load frames + metadata (fxfycxcy / w2c; no distortion).
2. Cover-crop to 192x336 (+ matching intrinsics) for RoGe.
3. Generate all non-context frames in ordered chunks (context from
   evaluation/re10k/splits); LagerNVS condition uses native-resolution context.
4. Column-crop 192x336 -> 192x192 eval FOV using per-scene native resolution.
5. Rank 0 runs unified_metrics_re10k.py + unified_ape_re10k.py.
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

from diffsynth.model import RoGe
from re10k_eval_common import (
    DATA_ROOT,
    SPLITS_DIR,
    discover_scenes,
    resolve_frame_path,
    resolve_scene_layout,
    split_file,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT_DEFAULT = str(DATA_ROOT)

TRAIN_H, TRAIN_W = 192, 336
EVAL_H, EVAL_W = 192, 192


@dataclass(frozen=True)
class SceneResult:
    scene: str
    scene_dir: str
    output_dir: str
    num_saved_frames: int


def compute_cover_crop_params(
    orig_w: int, orig_h: int, tgt_w: int, tgt_h: int
) -> dict[str, float | int]:
    scale = max(tgt_w / orig_w, tgt_h / orig_h)
    resize_w = int(round(orig_w * scale))
    resize_h = int(round(orig_h * scale))
    crop_x = (resize_w - tgt_w) / 2.0
    crop_y = (resize_h - tgt_h) / 2.0
    return {
        "resize_w": resize_w,
        "resize_h": resize_h,
        "offset_x": int(round(crop_x)),
        "offset_y": int(round(crop_y)),
        "crop_x": crop_x,
        "crop_y": crop_y,
    }


def resize_cover_center_crop(rgb: np.ndarray, tgt_h: int, tgt_w: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h == tgt_h and w == tgt_w:
        return np.copy(rgb)
    params = compute_cover_crop_params(w, h, tgt_w, tgt_h)
    resized = cv2.resize(
        rgb,
        (int(params["resize_w"]), int(params["resize_h"])),
        interpolation=cv2.INTER_AREA,
    )
    y0, x0 = int(params["offset_y"]), int(params["offset_x"])
    return resized[y0 : y0 + tgt_h, x0 : x0 + tgt_w]


def scale_intrinsic_cover_crop(
    k: np.ndarray, orig_w: int, orig_h: int, tgt_w: int, tgt_h: int
) -> np.ndarray:
    params = compute_cover_crop_params(orig_w, orig_h, tgt_w, tgt_h)
    scale = float(params["resize_w"]) / orig_w
    # cover uses uniform scale; keep fy consistent with resize_h/orig_h
    sy = float(params["resize_h"]) / orig_h
    k_out = np.asarray(k, dtype=np.float64).copy()
    k_out[0, 0] = k[0, 0] * scale
    k_out[1, 1] = k[1, 1] * sy
    k_out[0, 2] = k[0, 2] * scale - float(params["crop_x"])
    k_out[1, 2] = k[1, 2] * sy - float(params["crop_y"])
    return k_out


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


def compute_roge_col_range(
    *,
    orig_h: int,
    orig_w: int,
    roge_tgt_h: int = TRAIN_H,
    roge_tgt_w: int = TRAIN_W,
) -> tuple[int, int]:
    _y0, x0, _crop_h, crop_w = compute_common_crop(orig_h, orig_w)
    params = compute_cover_crop_params(orig_w, orig_h, roge_tgt_w, roge_tgt_h)
    resize_w = int(params["resize_w"])
    offset_x = int(params["offset_x"])
    sx = resize_w / orig_w
    col0 = int(round(x0 * sx - offset_x))
    col1 = int(round((x0 + crop_w) * sx - offset_x))
    col0 = max(0, col0)
    col1 = min(roge_tgt_w, max(col0 + 1, col1))
    return col0, col1


def postprocess_roge(
    img: np.ndarray,
    *,
    orig_h: int,
    orig_w: int,
    roge_tgt_h: int = TRAIN_H,
    roge_tgt_w: int = TRAIN_W,
) -> np.ndarray:
    col0, col1 = compute_roge_col_range(
        orig_h=orig_h,
        orig_w=orig_w,
        roge_tgt_h=roge_tgt_h,
        roge_tgt_w=roge_tgt_w,
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


def _k_from_fxfycxcy(fxfycxcy) -> np.ndarray:
    fx, fy, cx, cy = [float(x) for x in fxfycxcy]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


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
        if path.is_dir() and split_file(path.name).is_file():
            scenes.append(path)
            continue
        if not path.is_dir():
            raise NotADirectoryError(f"Scene root is not a directory: {path}")
        for layout in discover_scenes(path):
            scenes.append(layout.scene_dir)
    if not scenes:
        raise FileNotFoundError(
            "No RE10K benchmark scenes found. Expected images/<seq>/ under "
            f"processed_re10k/benchmark for a split in {SPLITS_DIR}"
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


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


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
    metrics_script = SCRIPT_DIR / "re10k_run_metrics.sh"
    if not metrics_script.is_file():
        raise FileNotFoundError(f"Metrics runner not found: {metrics_script}")
    print(f"\n=== Running RE10K unified metrics on {output_dir} ===", flush=True)
    subprocess.run(
        ["bash", str(metrics_script), str(output_dir)],
        cwd=str(REPO_ROOT),
        check=True,
    )

    if skip_ape:
        print("Skipping unified RE10K APE (--skip_ape).")
        return

    ape_script = SCRIPT_DIR / "re10k_run_ape.sh"
    if not ape_script.is_file():
        raise FileNotFoundError(f"APE runner not found: {ape_script}")
    print(f"\n=== Running RE10K unified APE on {output_dir} ===", flush=True)
    subprocess.run(
        ["bash", str(ape_script), str(output_dir)],
        cwd=str(REPO_ROOT),
        check=True,
    )


def run_scene(args, model: RoGe) -> SceneResult:
    scene_dir = args._scene_dir
    layout = resolve_scene_layout(scene_dir)
    scene_name = layout.scene_hash
    scene_out = Path(args.output_dir) / scene_name

    split = json.loads(layout.context_target_path.read_text())
    context_idx = [int(x) for x in split["context_idx"]]
    meta = json.loads(layout.metadata_path.read_text())
    frames = meta["frames"]
    n_frames = len(frames)
    _ensure_indices(context_idx, n_frames, "context_idx")

    context_set = set(context_idx)
    all_target_idx = [idx for idx in range(n_frames) if idx not in context_set]
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

    first_path = resolve_frame_path(layout.images_dir, 0)
    if first_path is None:
        raise FileNotFoundError(f"Missing frame 0 under {layout.images_dir}")
    sample = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
    if sample is None:
        raise FileNotFoundError(f"Missing sample image for {scene_name}: {first_path}")
    orig_h, orig_w = sample.shape[:2]
    layout_h, layout_w = layout.orig_hw
    if (orig_h, orig_w) != (layout_h, layout_w):
        print(
            f"[warn] {scene_name}: image size {(orig_h, orig_w)} != layout.orig_hw "
            f"{(layout_h, layout_w)}; using image size",
            flush=True,
        )

    src_intr = _k_from_fxfycxcy(frames[0]["fxfycxcy"])
    infer_intr = scale_intrinsic_cover_crop(
        src_intr, orig_w, orig_h, args.width, args.height
    ).astype(np.float32)

    w2cs = np.zeros((n_frames, 4, 4), dtype=np.float32)
    native_k_by_idx: dict[int, np.ndarray] = {}
    cropped_by_idx: dict[int, np.ndarray] = {}
    native_rgb_by_idx: dict[int, np.ndarray] = {}

    for frame_idx, frame in enumerate(frames):
        src = resolve_frame_path(layout.images_dir, frame_idx)
        if src is None:
            raise FileNotFoundError(
                f"Missing image for frame {frame_idx} under {layout.images_dir}"
            )
        rgb = _load_rgb(src)
        native_rgb_by_idx[frame_idx] = rgb
        cropped_by_idx[frame_idx] = resize_cover_center_crop(
            rgb, args.height, args.width
        )
        native_k_by_idx[frame_idx] = _k_from_fxfycxcy(frame["fxfycxcy"])
        w2cs[frame_idx] = np.asarray(frame["w2c"], dtype=np.float32)

    context_w2c = np.stack([w2cs[idx] for idx in context_idx], axis=0)
    context_images = [Image.fromarray(cropped_by_idx[idx]) for idx in context_idx]
    lagernvs_context_images = [
        Image.fromarray(native_rgb_by_idx[idx]) for idx in context_idx
    ]

    saved_count = 0

    for chunk_id, target_chunk in enumerate(_chunks(all_target_idx, args.chunk_size)):
        real_target_chunk = list(target_chunk)
        if not real_target_chunk:
            continue
        pad_count = max(0, args.chunk_size - len(real_target_chunk))
        inference_target_chunk = real_target_chunk + [real_target_chunk[-1]] * pad_count

        target_w2c = np.stack(
            [w2cs[idx] for idx in inference_target_chunk], axis=0
        ).astype(np.float32)
        w2c = np.concatenate([context_w2c, target_w2c], axis=0)
        n_poses = len(context_idx) + len(inference_target_chunk)
        intrinsics = np.stack([infer_intr] * n_poses, axis=0)

        generate_kwargs = dict(
            images=[img.copy() for img in context_images],
            w2c_poses=w2c,
            intrinsics=intrinsics,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            cfg_scale=args.cfg_scale,
            resize_mode=None,
            pose_gauge=args.pose_gauge,
        )
        if getattr(model, "lagernvs_condition", None) is not None:
            pose_indices = list(context_idx) + list(inference_target_chunk)
            full_intr = np.stack(
                [native_k_by_idx[i] for i in pose_indices],
                axis=0,
            )
            generate_kwargs["lagernvs_context_images"] = [
                img.copy() for img in lagernvs_context_images
            ]
            generate_kwargs["lagernvs_intrinsics"] = full_intr

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
            gt_bef = Image.fromarray(cropped_by_idx[frame_idx])
            name = f"{frame_idx:06d}.png"

            _save_pil(bef_pred_dir / name, pred)
            _save_pil(bef_gt_dir / name, gt_bef)

            pred_aft = postprocess_roge(
                np.asarray(pred.convert("RGB")),
                orig_h=orig_h,
                orig_w=orig_w,
                roge_tgt_h=args.height,
                roge_tgt_w=args.width,
            )
            gt_aft = postprocess_roge(
                cropped_by_idx[frame_idx],
                orig_h=orig_h,
                orig_w=orig_w,
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
        "orig_h": orig_h,
        "orig_w": orig_w,
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
        description="Batch RoGe inference for RE10K benchmark (w/ LagerNVS)."
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
        help="With --joint_train: also inject/load reconstructor LoRA.",
    )
    parser.add_argument(
        "--lagernvs_lora_rank",
        type=int,
        default=8,
        help="LoRA rank for LagerNVS under --joint_train (default 8).",
    )
    parser.add_argument("--hybrid_vae_encoding", action="store_true")
    parser.add_argument("--raymap_pe", action="store_true")
    parser.add_argument("--geom_pe", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip unified_metrics_re10k.py and unified_ape_re10k.py after inference.",
    )
    parser.add_argument(
        "--skip_ape",
        action="store_true",
        help="Skip unified_ape_re10k.py after metrics.",
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
            print("Skipping unified RE10K metrics/APE (--skip_metrics).")
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
