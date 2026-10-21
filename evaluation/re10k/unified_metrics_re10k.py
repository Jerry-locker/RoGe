#!/usr/bin/env python3
"""Unified RE10K metrics on aft_postprocess outputs.

Thin adapter over ``unified_metrics_dl3dv.py`` for the LagerNVS-layout RE10K
benchmark::

    .../processed_re10k/benchmark/{images,metadata}/

Expects ``<exp_root>/<scene>/aft_postprocess/{gt,pred}/<frame_idx:06d>.png``.
APE is evaluated separately via ``unified_ape_re10k.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from evaluation.paths import ensure_eval_paths

ensure_eval_paths()


import re10k_eval_common as re10k_common

# Patch shared helpers before importing the DL3DV metrics module body usages.
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

import unified_metrics_dl3dv as metrics  # noqa: E402

metrics.DATA_ROOT = re10k_common.DATA_ROOT
metrics.DEFAULT_EXP_ROOT = re10k_common.DEFAULT_EXP_ROOT
metrics.DEFAULT_ORIG_H = re10k_common.DEFAULT_ORIG_H
metrics.DEFAULT_ORIG_W = re10k_common.DEFAULT_ORIG_W
metrics.DEFAULT_VIDEO_CHUNK_SIZE = re10k_common.DEFAULT_VIDEO_CHUNK_SIZE
metrics.discover_split_index = re10k_common.discover_split_index
metrics.discover_exp_scenes = re10k_common.discover_exp_scenes
metrics.load_eval_cameras = re10k_common.load_eval_cameras
metrics.load_rgb = re10k_common.load_rgb
metrics.resolve_frame_path = re10k_common.resolve_frame_path
metrics.list_frame_indices = re10k_common.list_frame_indices
metrics.chunk_sequence = re10k_common.chunk_sequence
metrics.mean_or_nan = re10k_common.mean_or_nan
metrics.collect_video_frame_paths = re10k_common.collect_video_frame_paths
metrics.write_summary_metrics_csv = re10k_common.write_summary_metrics_csv

_orig_evaluate_scene = metrics.evaluate_scene


def _evaluate_scene_with_scene_hw(scene_dir, split, *, orig_hw, **kwargs):
    scene_hw = getattr(split, "orig_hw", None) or orig_hw
    return _orig_evaluate_scene(scene_dir, split, orig_hw=scene_hw, **kwargs)


metrics.evaluate_scene = _evaluate_scene_with_scene_hw


def main() -> int:
    return metrics.main()


if __name__ == "__main__":
    raise SystemExit(main())
