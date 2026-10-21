#!/usr/bin/env python3
"""Pose APE on ACID aft_postprocess outputs (benchmark).

Thin adapter over ``unified_ape_dl3dv.py`` for LagerNVS-layout ACID::

    .../processed_acid/benchmark/{images,metadata}/
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

import acid_eval_common as acid_common
import dl3dv_eval_common as dl3dv_common

dl3dv_common.DATA_ROOT = acid_common.DATA_ROOT
dl3dv_common.DEFAULT_EXP_ROOT = acid_common.DEFAULT_EXP_ROOT
dl3dv_common.DEFAULT_ORIG_H = acid_common.DEFAULT_ORIG_H
dl3dv_common.DEFAULT_ORIG_W = acid_common.DEFAULT_ORIG_W
dl3dv_common.DEFAULT_VIDEO_CHUNK_SIZE = acid_common.DEFAULT_VIDEO_CHUNK_SIZE
dl3dv_common.SUMMARY_METRIC_ORDER = acid_common.SUMMARY_METRIC_ORDER
dl3dv_common.SceneSplit = acid_common.SceneSplit
dl3dv_common.discover_split_index = acid_common.discover_split_index
dl3dv_common.discover_exp_scenes = acid_common.discover_exp_scenes
dl3dv_common.load_eval_cameras = acid_common.load_eval_cameras
dl3dv_common.load_rgb = acid_common.load_rgb
dl3dv_common.resolve_frame_path = acid_common.resolve_frame_path
dl3dv_common.list_frame_indices = acid_common.list_frame_indices
dl3dv_common.chunk_sequence = acid_common.chunk_sequence
dl3dv_common.mean_or_nan = acid_common.mean_or_nan
dl3dv_common.collect_video_frame_paths = acid_common.collect_video_frame_paths
dl3dv_common.write_summary_metrics_csv = acid_common.write_summary_metrics_csv
dl3dv_common.read_summary_metrics_csv = acid_common.read_summary_metrics_csv
dl3dv_common.compute_common_crop = acid_common.compute_common_crop

import unified_ape_dl3dv as ape  # noqa: E402

ACID_SCRIPT = Path(__file__).resolve()


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
            f"[warn] no ACID metadata for {split.scene_hash}; skip APE "
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


def _build_worker_command_acid(scene_paths, args, *, output_path):
    command = _orig_build_worker_command(
        scene_paths, args, output_path=output_path
    )
    for i, part in enumerate(command):
        if str(part).endswith("unified_ape_dl3dv.py"):
            command[i] = str(ACID_SCRIPT)
            break
    else:
        if len(command) >= 2 and str(command[1]).endswith(".py"):
            command[1] = str(ACID_SCRIPT)
    return command


ape.DATA_ROOT = acid_common.DATA_ROOT
ape.DEFAULT_EXP_ROOT = acid_common.DEFAULT_EXP_ROOT
ape.DEFAULT_ORIG_H = acid_common.DEFAULT_ORIG_H
ape.DEFAULT_ORIG_W = acid_common.DEFAULT_ORIG_W
ape.DEFAULT_VIDEO_CHUNK_SIZE = acid_common.DEFAULT_VIDEO_CHUNK_SIZE
ape.discover_split_index = acid_common.discover_split_index
ape.discover_exp_scenes = acid_common.discover_exp_scenes
ape.load_eval_cameras = acid_common.load_eval_cameras
ape.load_rgb = acid_common.load_rgb
ape.list_frame_indices = acid_common.list_frame_indices
ape.chunk_sequence = acid_common.chunk_sequence
ape.mean_or_nan = acid_common.mean_or_nan
ape.collect_video_frame_paths = acid_common.collect_video_frame_paths
ape.write_summary_metrics_csv = acid_common.write_summary_metrics_csv

_orig_run_scene_ape = ape.run_scene_ape_evaluation
ape.run_scene_ape_evaluation = _run_scene_ape_with_scene_hw

_orig_build_worker_command = ape.build_worker_command
ape.build_worker_command = _build_worker_command_acid


def main() -> int:
    return ape.main()


if __name__ == "__main__":
    raise SystemExit(main())
