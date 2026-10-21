#!/usr/bin/env python3
"""VGGT-Omega pose APE on DL3DV aft_postprocess outputs.

For each scene under ``--exp_root``:

* Feed-forward VGGT-Omega poses on pred frames excluding ``context_idx``
* evo translational + rotational APE RMSE (deg) vs ``transforms.json`` GT
* GT translations normalized by GT trajectory length (sum of consecutive
  camera-center distances), then Sim(3) Umeyama (``align + correct_scale``)
* Per-scene chunk mean (81-frame videos by default), then dataset mean

Splits come from ``context_target_6.json`` under ``--data_root``.
Expects ``<exp_root>/<scene>/aft_postprocess/{gt,pred}/<frame_idx:06d>.png``.

Multi-GPU: pass ``--gpus 0,1,2,3`` (or rely on auto-detected visible GPUs) to
evaluate scenes in parallel, one subprocess per GPU with round-robin assignment.

Requires vendored ``third_party/vggt-omega/``, a local VGGT-Omega checkpoint, and ``evo``.
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
import faulthandler
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

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
    write_summary_metrics_csv,
)
from evaluation.vggt_omega_ape import (
    DEFAULT_IMAGE_RESOLUTION,
    DEFAULT_PREPROCESS_MODE,
    DEFAULT_VGGT_OMEGA_CHECKPOINT,
    VggtOmegaApeResult,
    VggtOmegaApeSubprocessRunner,
    VggtOmegaPoseEstimator,
    aggregate_vggt_omega_chunk_results,
    evaluate_vggt_omega_ape,
    nan_ape_result,
)

PoseBackendRunner = VggtOmegaApeSubprocessRunner | VggtOmegaPoseEstimator

CSV_FIELDNAMES = [
    "scene",
    "ape_rmse",
    "rotation_ape",
    "ape_n_images",
    "ape_n_poses",
    "ape_n_aligned",
    "ape_n_chunks",
    "ape_n_nan_chunks",
    "num_video_frames",
    "num_video_chunks",
]


def collect_scene_video_frames(
    scene_dir: Path,
    split: SceneSplit,
) -> tuple[list[Path], list[Path], list[int], list[int]]:
    gt_dir = scene_dir / "aft_postprocess" / "gt"
    pred_dir = scene_dir / "aft_postprocess" / "pred"
    present = list_frame_indices(pred_dir)
    context_set = set(split.context_indices)
    if present:
        video_indices = [i for i in present if i not in context_set]
    else:
        video_indices = [
            i for i in range(max(split.target_indices) + 1) if i not in context_set
        ]
    gt_paths, pred_paths, loaded_indices, missing = collect_video_frame_paths(
        gt_dir, pred_dir, video_indices
    )
    return gt_paths, pred_paths, loaded_indices, missing


def _is_subprocess_runner(runner: PoseBackendRunner) -> bool:
    return isinstance(runner, VggtOmegaApeSubprocessRunner)

def _evaluate_chunk_specs(
    runner: PoseBackendRunner,
    chunk_specs: list[tuple[list[str], list[int], list[list[float]]]],
    *,
    backend: str,
) -> VggtOmegaApeResult:
    if isinstance(runner, VggtOmegaApeSubprocessRunner):
        return runner.evaluate_chunks(chunk_specs)

    ape_chunks: list[VggtOmegaApeResult] = []
    for chunk_idx, (paths_str, chunk_indices, gt_lists) in enumerate(chunk_specs):
        paths = [Path(p) for p in paths_str]
        gt_c2ws = [np.asarray(x, dtype=np.float64) for x in gt_lists]
        print(
            f"[{backend}] chunk {chunk_idx}: {len(paths)} frame(s)",
            flush=True,
        )
        ape_chunks.append(
            evaluate_vggt_omega_ape(runner, paths, chunk_indices, gt_c2ws)
        )
    return aggregate_vggt_omega_chunk_results(ape_chunks)


def run_scene_ape_evaluation(
    pred_paths: list[Path],
    loaded_indices: list[int],
    split: SceneSplit,
    *,
    ape_runner: PoseBackendRunner,
    pose_backend: str,
    video_chunk_size: int,
    orig_hw: tuple[int, int],
    close_subprocess: bool = True,
) -> VggtOmegaApeResult:
    ape_result = nan_ape_result(len(pred_paths))
    if split.transforms_path is None:
        print(f"[warn] no transforms.json for {split.scene_hash}; skip APE")
        return ape_result
    if len(pred_paths) < 2:
        print(
            f"[warn] need >=2 non-context frames for APE, got {len(pred_paths)} "
            f"({split.scene_hash})"
        )
        return ape_result

    print(f"[stage] {pose_backend} APE", flush=True)
    sample = load_rgb(pred_paths[0])
    image_hw = (sample.shape[0], sample.shape[1])
    path_chunks = chunk_sequence(pred_paths, video_chunk_size)
    index_chunks = chunk_sequence(loaded_indices, video_chunk_size)

    chunk_specs: list[tuple[list[str], list[int], list[list[float]]]] = []
    for chunk_idx, (chunk_paths, chunk_indices) in enumerate(
        zip(path_chunks, index_chunks)
    ):
        if len(chunk_paths) < 2:
            print(
                f"[warn] APE skip chunk {chunk_idx}: only {len(chunk_paths)} frame(s)"
            )
            continue
        try:
            _, gt_c2ws = load_eval_cameras(
                split.transforms_path,
                chunk_indices,
                image_hw=image_hw,
                orig_hw=orig_hw,
            )
            chunk_specs.append(
                (
                    [str(p) for p in chunk_paths],
                    list(chunk_indices),
                    [c2w.tolist() for c2w in gt_c2ws],
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] APE chunk {chunk_idx} camera load failed: {exc}")

    if not chunk_specs:
        print(
            f"[warn] no APE chunks for {split.scene_hash}; skip",
            flush=True,
        )
        return ape_result

    print(
        f"[{pose_backend}] evaluating {len(chunk_specs)} chunk(s), "
        f"{sum(len(spec[0]) for spec in chunk_specs)} frame(s) total",
        flush=True,
    )
    try:
        ape_result = _evaluate_chunk_specs(
            ape_runner, chunk_specs, backend=pose_backend
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {pose_backend} APE (chunked) failed: {exc}")
        ape_result = nan_ape_result(len(pred_paths))

    if close_subprocess and _is_subprocess_runner(ape_runner):
        ape_runner.close()  # type: ignore[union-attr]
    return ape_result


def ape_metrics_row(
    scene_hash: str,
    ape_result: VggtOmegaApeResult,
    *,
    num_video_frames: int,
    num_video_chunks: int,
) -> dict[str, float | str]:
    return {
        "scene": scene_hash,
        "ape_rmse": float(ape_result.ape_rmse),
        "rotation_ape": float(ape_result.rotation_ape),
        "ape_n_images": float(ape_result.n_images),
        "ape_n_poses": float(ape_result.n_poses),
        "ape_n_aligned": float(ape_result.n_ape_poses),
        "ape_n_chunks": float(ape_result.n_chunks),
        "ape_n_nan_chunks": float(ape_result.n_nan_chunks),
        "num_video_frames": float(num_video_frames),
        "num_video_chunks": float(num_video_chunks),
    }


def write_csv(rows: list[dict[str, float | str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDNAMES})


def visible_gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip():
        return [part.strip() for part in visible.split(",") if part.strip()]
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return [
            str(index)
            for index, line in enumerate(output.splitlines())
            if line.startswith("GPU ")
        ]
    except Exception:
        return ["0"]


def parse_gpu_ids(gpus: str | None) -> list[str]:
    if gpus:
        return [part.strip() for part in gpus.split(",") if part.strip()]
    return visible_gpu_ids()


def append_average_row(
    ape_rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    if not ape_rows:
        return ape_rows
    ape_rmse_mean = mean_or_nan([float(r["ape_rmse"]) for r in ape_rows])
    rotation_ape_mean = mean_or_nan([float(r["rotation_ape"]) for r in ape_rows])
    total_ape_nan_chunks = sum(int(float(r["ape_n_nan_chunks"])) for r in ape_rows)
    total_ape_chunks = sum(int(float(r["ape_n_chunks"])) for r in ape_rows)
    ape_rows.append(
        {
            "scene": "average",
            "ape_rmse": ape_rmse_mean,
            "rotation_ape": rotation_ape_mean,
            "ape_n_images": mean_or_nan([float(r["ape_n_images"]) for r in ape_rows]),
            "ape_n_poses": mean_or_nan([float(r["ape_n_poses"]) for r in ape_rows]),
            "ape_n_aligned": mean_or_nan(
                [float(r["ape_n_aligned"]) for r in ape_rows]
            ),
            "ape_n_chunks": float(total_ape_chunks),
            "ape_n_nan_chunks": float(total_ape_nan_chunks),
            "num_video_frames": mean_or_nan(
                [float(r["num_video_frames"]) for r in ape_rows]
            ),
            "num_video_chunks": mean_or_nan(
                [float(r["num_video_chunks"]) for r in ape_rows]
            ),
        }
    )
    return ape_rows


def evaluate_scenes(
    scene_dirs: list[Path],
    *,
    data_root: Path,
    video_chunk_size: int,
    orig_hw: tuple[int, int],
    ape_runner: PoseBackendRunner,
    pose_backend: str,
) -> list[dict[str, float | str]]:
    split_index = discover_split_index(data_root)
    ape_rows: list[dict[str, float | str]] = []
    for scene_dir in scene_dirs:
        scene_hash = scene_dir.name
        if scene_hash not in split_index:
            print(f"[skip] {scene_hash}: no split in data_root")
            continue
        split = split_index[scene_hash]
        print(f"\n=== {scene_hash} ===", flush=True)
        _gt_paths, pred_paths, loaded_indices, missing = collect_scene_video_frames(
            scene_dir, split
        )
        if missing:
            print(f"[warn] missing {len(missing)} video frames, e.g. {missing[:5]}")
        used_chunks = chunk_sequence(
            pred_paths, video_chunk_size
        )
        num_video_chunks = len(used_chunks)
        num_video_frames = sum(len(c) for c in used_chunks)
        ape_result = run_scene_ape_evaluation(
            pred_paths,
            loaded_indices,
            split,
            ape_runner=ape_runner,
            pose_backend=pose_backend,
            video_chunk_size=video_chunk_size,
            orig_hw=orig_hw,
            close_subprocess=False,
        )
        row = ape_metrics_row(
            scene_hash,
            ape_result,
            num_video_frames=num_video_frames,
            num_video_chunks=num_video_chunks,
        )
        ape_rows.append(row)
        print(
            f"  TransAPE={row['ape_rmse']}  "
            f"RotAPE={row['rotation_ape']}  "
            f"APE_nan_chunks={int(row['ape_n_nan_chunks'])}/"
            f"{int(row['ape_n_chunks'])}  "
            f"video_frames={int(row['num_video_frames'])}  "
            f"video_chunks={int(row['num_video_chunks'])}",
            flush=True,
        )
    return ape_rows


def build_ape_runner(
    device: torch.device,
    args: argparse.Namespace,
) -> PoseBackendRunner:
    use_subprocess = bool(args.ape_subprocess)
    if use_subprocess:
        return VggtOmegaApeSubprocessRunner(
            device,
            args.vggt_omega_checkpoint,
            image_resolution=args.vggt_omega_resolution,
            preprocess_mode=args.vggt_omega_preprocess,
        )
    print("[vggt-omega] loading model in-process", flush=True)
    return VggtOmegaPoseEstimator(
        device,
        args.vggt_omega_checkpoint,
        image_resolution=args.vggt_omega_resolution,
        preprocess_mode=args.vggt_omega_preprocess,
    )


def worker_entry(scene_paths: list[str], args: argparse.Namespace) -> None:
    scene_dirs = [Path(path) for path in scene_paths]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        f"[worker] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"device={device} scenes={len(scene_dirs)}",
        flush=True,
    )
    ape_runner = build_ape_runner(device, args)
    try:
        ape_rows = evaluate_scenes(
            scene_dirs,
            data_root=args.data_root,
            video_chunk_size=args.video_chunk_size,
            orig_hw=(args.orig_h, args.orig_w),
            ape_runner=ape_runner,
            pose_backend="vggt_omega",
        )
    finally:
        if _is_subprocess_runner(ape_runner):
            ape_runner.close()  # type: ignore[union-attr]
    if not ape_rows:
        raise RuntimeError("No scenes evaluated in worker")
    if args.output is None:
        raise RuntimeError("Worker mode requires --output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ape_rows, indent=2))


def build_worker_command(
    scene_paths: list[str],
    args: argparse.Namespace,
    *,
    output_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--data_root",
        str(args.data_root),
        "--exp_root",
        str(args.exp_root),
        "--output",
        str(output_path),
        "--orig_h",
        str(args.orig_h),
        "--orig_w",
        str(args.orig_w),
        "--video_chunk_size",
        str(args.video_chunk_size),
        "--vggt_omega_checkpoint",
        str(args.vggt_omega_checkpoint),
        "--vggt_omega_resolution",
        str(args.vggt_omega_resolution),
        "--vggt_omega_preprocess",
        args.vggt_omega_preprocess,
        "--scene_paths",
        *scene_paths,
    ]
    if args.ape_subprocess:
        command.append("--ape_subprocess")
    return command


def launch_multi_gpu(
    scene_dirs: list[Path],
    gpu_ids: list[str],
    args: argparse.Namespace,
) -> list[dict[str, float | str]]:
    """Round-robin scenes across GPUs; each worker writes a partial JSON."""
    if not gpu_ids:
        raise ValueError("At least one GPU is required")
    buckets: list[list[str]] = [[] for _ in gpu_ids]
    for index, scene_dir in enumerate(scene_dirs):
        buckets[index % len(gpu_ids)].append(str(scene_dir))

    with tempfile.TemporaryDirectory(prefix="unified_ape_dl3dv_") as tmpdir:
        tmp_path = Path(tmpdir)
        processes: list[tuple[str, subprocess.Popen]] = []
        for gpu_id, bucket in zip(gpu_ids, buckets):
            if not bucket:
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["PYTHONUNBUFFERED"] = "1"
            partial_output = tmp_path / f"worker_{gpu_id}.json"
            command = build_worker_command(
                bucket,
                args,
                output_path=partial_output,
            )
            print(f"[launch] GPU {gpu_id}: {len(bucket)} scene(s)", flush=True)
            processes.append((gpu_id, subprocess.Popen(command, env=env)))

        return_codes = [(gpu_id, process.wait()) for gpu_id, process in processes]
        failed = [item for item in return_codes if item[1] != 0]
        if failed:
            raise SystemExit(f"One or more GPU workers failed: {failed}")

        ape_rows: list[dict[str, float | str]] = []
        for partial_output in sorted(tmp_path.glob("worker_*.json")):
            ape_rows.extend(json.loads(partial_output.read_text()))
        return ape_rows


def default_ape_output_paths(exp_root: Path) -> tuple[Path, Path]:
    """Return ``(csv_path, detail_json_path)``."""
    return (
        exp_root / "unified_ape.csv",
        exp_root / "unified_ape_detail.json",
    )


def detail_json_for_csv(csv_path: Path) -> Path:
    """Map APE CSV name to the companion detail JSON name."""
    if csv_path.name == "unified_ape.csv":
        return csv_path.with_name("unified_ape_detail.json")
    return csv_path.with_name(csv_path.stem + "_detail.json")



def ape_summary_metrics(average_row: dict[str, float | str]) -> dict[str, float]:
    """Map scene-average APE columns to ``unified_metrics.csv`` keys."""
    return {
        "transape": float(average_row["ape_rmse"]),
        "rotape": float(average_row["rotation_ape"]),
    }


def write_results(
    ape_rows: list[dict[str, float | str]],
    out_csv: Path,
    detail_json: Path | None = None,
) -> None:
    ape_rows = append_average_row(list(ape_rows))
    detail_path = detail_json or detail_json_for_csv(out_csv)
    if out_csv.name.startswith("unified_ape") and out_csv.suffix == ".csv":
        # Prefer merging APE into the compact unified_metrics.csv summary.
        summary_csv = out_csv.with_name("unified_metrics.csv")
    else:
        summary_csv = out_csv
    write_csv(ape_rows, out_csv)  # keep per-scene APE table for debugging
    detail_path.write_text(json.dumps(ape_rows, indent=2))
    average_row = ape_rows[-1]
    write_summary_metrics_csv(
        summary_csv,
        ape_summary_metrics(average_row),
        merge=True,
    )
    print(
        f"\n[average over {len(ape_rows) - 1} scenes] "
        f"TransAPE={average_row['ape_rmse']}  "
        f"RotAPE_deg={average_row['rotation_ape']}  "
        f"nan_chunks={int(float(average_row['ape_n_nan_chunks']))}/"
        f"{int(float(average_row['ape_n_chunks']))}"
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {detail_path}")
    print(f"Updated summary {summary_csv}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VGGT-Omega pose APE on DL3DV aft_postprocess frames"
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
        help="Results CSV path (default: <exp_root>/unified_ape.csv)",
    )
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene filter")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated physical GPU ids (default: visible GPUs when multi-GPU)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--orig_h", type=int, default=DEFAULT_ORIG_H)
    parser.add_argument("--orig_w", type=int, default=DEFAULT_ORIG_W)
    parser.add_argument(
        "--video_chunk_size",
        type=int,
        default=DEFAULT_VIDEO_CHUNK_SIZE,
        help="Split each scene's non-context frames into videos of this many frames",
    )
    parser.add_argument(
        "--ape_subprocess",
        action="store_true",
        help="Run APE in an isolated subprocess (default: in-process)",
    )
    parser.add_argument(
        "--vggt_ape_subprocess",
        action="store_true",
        help=argparse.SUPPRESS,  # backward-compatible alias for --ape_subprocess
    )
    parser.add_argument(
        "--vggt_omega_checkpoint",
        type=Path,
        default=DEFAULT_VGGT_OMEGA_CHECKPOINT,
        help="Local VGGT-Omega checkpoint (.pt)",
    )
    parser.add_argument(
        "--vggt_omega_resolution",
        type=int,
        default=DEFAULT_IMAGE_RESOLUTION,
        help="VGGT-Omega input resolution (512 for vggt_omega_1b_512.pt)",
    )
    parser.add_argument(
        "--vggt_omega_preprocess",
        type=str,
        default=DEFAULT_PREPROCESS_MODE,
        choices=["balanced", "max_size"],
        help="VGGT-Omega image preprocessing mode",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--scene_paths", nargs="*", default=None, help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.vggt_ape_subprocess:
        args.ape_subprocess = True

    faulthandler.enable()

    if args.worker:
        if not args.scene_paths:
            parser.error("--worker requires --scene_paths")
        worker_entry(args.scene_paths, args)
        return 0

    scene_dirs = discover_exp_scenes(args.exp_root)
    if args.scenes:
        wanted = set(args.scenes)
        scene_dirs = [d for d in scene_dirs if d.name in wanted]
        if not scene_dirs:
            raise SystemExit(f"No matching experiment scenes for: {args.scenes}")

    gpu_ids = parse_gpu_ids(args.gpus)
    use_multi_gpu = len(gpu_ids) > 1 and len(scene_dirs) > 1
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(
        f"Scenes={len(scene_dirs)}  device={device}  gpus={gpu_ids}  "
        f"multi_gpu={use_multi_gpu}  video_chunk_size={args.video_chunk_size}  "
        f"data_root={args.data_root}  exp_root={args.exp_root}"
    )
    print(
        "VGGT-Omega APE: pred excl. context; GT traj-length-normalized TransAPE + "
        f"RotAPE (deg) with Sim(3) align; chunk mean per scene, then scene mean "
        f"(checkpoint={args.vggt_omega_checkpoint}, "
        f"resolution={args.vggt_omega_resolution}, "
        f"preprocess={args.vggt_omega_preprocess})."
    )

    default_csv, default_detail = default_ape_output_paths(args.exp_root)
    out_csv = args.output or default_csv
    detail_json = (
        detail_json_for_csv(out_csv) if args.output is not None else default_detail
    )

    if use_multi_gpu:
        ape_rows = launch_multi_gpu(scene_dirs, gpu_ids, args)
        if not ape_rows:
            raise SystemExit("No scenes evaluated.")
        write_results(ape_rows, out_csv, detail_json=detail_json)
        return 0

    ape_runner = build_ape_runner(device, args)
    try:
        ape_rows = evaluate_scenes(
            scene_dirs,
            data_root=args.data_root,
            video_chunk_size=args.video_chunk_size,
            orig_hw=(args.orig_h, args.orig_w),
            ape_runner=ape_runner,
            pose_backend="vggt_omega",
        )
    finally:
        if _is_subprocess_runner(ape_runner):
            ape_runner.close()  # type: ignore[union-attr]

    if not ape_rows:
        raise SystemExit("No scenes evaluated.")

    write_results(ape_rows, out_csv, detail_json=detail_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
