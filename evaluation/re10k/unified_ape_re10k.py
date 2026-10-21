#!/usr/bin/env python3
"""Pose APE on RE10K aft_postprocess outputs.

Thin adapter over ``unified_ape_dl3dv.py`` for the LagerNVS-layout RE10K
benchmark::

    .../processed_re10k/benchmark/{images,metadata}/

Expects ``<exp_root>/<scene>/aft_postprocess/{gt,pred}/<frame_idx:06d>.png``.
Uses per-scene native resolution from metadata/images when mapping GT cameras.

Important: multi-GPU workers must re-exec **this** script (not the DL3DV
entry), otherwise camera loading falls back to ``transforms.json`` and APE is
skipped / crashes.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from evaluation.paths import ensure_eval_paths

ensure_eval_paths()


from pathlib import Path

import re10k_eval_common as re10k_common

import dl3dv_eval_common as dl3dv_common

dl3dv_common.DATA_ROOT = re10k_common.DATA_ROOT
dl3dv_common.DEFAULT_EXP_ROOT = re10k_common.DEFAULT_EXP_ROOT
dl3dv_common.DEFAULT_ORIG_H = re10k_common.DEFAULT_ORIG_H
dl3dv_common.DEFAULT_ORIG_W = re10k_common.DEFAULT_ORIG_W
dl3dv_common.DEFAULT_VIDEO_CHUNK_SIZE = re10k_common.DEFAULT_VIDEO_CHUNK_SIZE
dl3dv_common.SUMMARY_METRIC_ORDER = re10k_common.SUMMARY_METRIC_ORDER
dl3dv_common.SceneSplit = re10k_common.SceneSplit
dl3dv_common.discover_split_index = re10k_common.discover_split_index
dl3dv_common.discover_exp_scenes = re10k_common.discover_exp_scenes
dl3dv_common.load_eval_cameras = re10k_common.load_eval_cameras
dl3dv_common.load_rgb = re10k_common.load_rgb
dl3dv_common.resolve_frame_path = re10k_common.resolve_frame_path
dl3dv_common.list_frame_indices = re10k_common.list_frame_indices
dl3dv_common.chunk_sequence = re10k_common.chunk_sequence
dl3dv_common.mean_or_nan = re10k_common.mean_or_nan
dl3dv_common.collect_video_frame_paths = re10k_common.collect_video_frame_paths
dl3dv_common.write_summary_metrics_csv = re10k_common.write_summary_metrics_csv
dl3dv_common.read_summary_metrics_csv = re10k_common.read_summary_metrics_csv
dl3dv_common.compute_common_crop = re10k_common.compute_common_crop

import unified_ape_dl3dv as ape  # noqa: E402

RE10K_SCRIPT = Path(__file__).resolve()


def _run_scene_ape_with_scene_hw(
    pred_paths,
    loaded_indices,
    split,
    *,
    ape_runner,
    pose_backend,
    video_chunk_size,
    orig_hw,
    close_subprocess=True,
):
    scene_hw = getattr(split, "orig_hw", None) or orig_hw
    if split.transforms_path is None:
        print(
            f"[warn] no RE10K metadata for {split.scene_hash}; skip APE "
            f"(expected metadata/<seq>.json)",
            flush=True,
        )
    return _orig_run_scene_ape(
        pred_paths,
        loaded_indices,
        split,
        ape_runner=ape_runner,
        pose_backend=pose_backend,
        video_chunk_size=video_chunk_size,
        orig_hw=scene_hw,
        close_subprocess=close_subprocess,
    )


def _build_worker_command_re10k(scene_paths, args, *, output_path):
    """Same as DL3DV builder, but re-exec this RE10K entrypoint."""
    command = _orig_build_worker_command(
        scene_paths, args, output_path=output_path
    )
    for i, part in enumerate(command):
        if str(part).endswith("unified_ape_dl3dv.py"):
            command[i] = str(RE10K_SCRIPT)
            break
    else:
        # Fallback: replace argv[1] if it looks like a python script path.
        if len(command) >= 2 and str(command[1]).endswith(".py"):
            command[1] = str(RE10K_SCRIPT)
    return command


ape.DATA_ROOT = re10k_common.DATA_ROOT
ape.DEFAULT_EXP_ROOT = re10k_common.DEFAULT_EXP_ROOT
ape.DEFAULT_ORIG_H = re10k_common.DEFAULT_ORIG_H
ape.DEFAULT_ORIG_W = re10k_common.DEFAULT_ORIG_W
ape.DEFAULT_VIDEO_CHUNK_SIZE = re10k_common.DEFAULT_VIDEO_CHUNK_SIZE
ape.discover_split_index = re10k_common.discover_split_index
ape.discover_exp_scenes = re10k_common.discover_exp_scenes
ape.load_eval_cameras = re10k_common.load_eval_cameras
ape.load_rgb = re10k_common.load_rgb
ape.list_frame_indices = re10k_common.list_frame_indices
ape.chunk_sequence = re10k_common.chunk_sequence
ape.mean_or_nan = re10k_common.mean_or_nan
ape.collect_video_frame_paths = re10k_common.collect_video_frame_paths
ape.write_summary_metrics_csv = re10k_common.write_summary_metrics_csv

_orig_run_scene_ape = ape.run_scene_ape_evaluation
ape.run_scene_ape_evaluation = _run_scene_ape_with_scene_hw

_orig_build_worker_command = ape.build_worker_command
ape.build_worker_command = _build_worker_command_re10k


def main() -> int:
    return ape.main()


if __name__ == "__main__":
    raise SystemExit(main())
