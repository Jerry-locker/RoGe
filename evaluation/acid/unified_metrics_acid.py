#!/usr/bin/env python3
"""Unified ACID metrics on aft_postprocess outputs.

Thin adapter over ``unified_metrics_dl3dv.py`` for the LagerNVS-layout ACID
benchmark::

    .../processed_acid/benchmark/{images,metadata}/

Expects ``<exp_root>/<scene>/aft_postprocess/{gt,pred}/<frame_idx:06d>.png``.
APE is evaluated separately via ``unified_ape_acid.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from evaluation.paths import ensure_eval_paths

ensure_eval_paths()


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

import unified_metrics_dl3dv as metrics  # noqa: E402

metrics.DATA_ROOT = acid_common.DATA_ROOT
metrics.DEFAULT_EXP_ROOT = acid_common.DEFAULT_EXP_ROOT
metrics.DEFAULT_ORIG_H = acid_common.DEFAULT_ORIG_H
metrics.DEFAULT_ORIG_W = acid_common.DEFAULT_ORIG_W
metrics.DEFAULT_VIDEO_CHUNK_SIZE = acid_common.DEFAULT_VIDEO_CHUNK_SIZE
metrics.discover_split_index = acid_common.discover_split_index
metrics.discover_exp_scenes = acid_common.discover_exp_scenes
metrics.load_eval_cameras = acid_common.load_eval_cameras
metrics.load_rgb = acid_common.load_rgb
metrics.resolve_frame_path = acid_common.resolve_frame_path
metrics.list_frame_indices = acid_common.list_frame_indices
metrics.chunk_sequence = acid_common.chunk_sequence
metrics.mean_or_nan = acid_common.mean_or_nan
metrics.collect_video_frame_paths = acid_common.collect_video_frame_paths
metrics.write_summary_metrics_csv = acid_common.write_summary_metrics_csv

_orig_evaluate_scene = metrics.evaluate_scene


def _evaluate_scene_with_scene_hw(scene_dir, split, *, orig_hw, **kwargs):
    scene_hw = getattr(split, "orig_hw", None) or orig_hw
    return _orig_evaluate_scene(scene_dir, split, orig_hw=scene_hw, **kwargs)


metrics.evaluate_scene = _evaluate_scene_with_scene_hw


def main() -> int:
    return metrics.main()


if __name__ == "__main__":
    raise SystemExit(main())
