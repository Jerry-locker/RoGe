#!/usr/bin/env python3
"""ACID benchmark batch inference for RoGe (w/ LagerNVS).

Same pipeline as ``re10k_benchmark_infer.py`` (LagerNVS-layout images +
metadata + context_target_6.json). Uses ``processed_acid/benchmark``;
context has 9 k-means views; all non-context frames are generated as targets.

After inference, rank 0 runs ``unified_metrics_acid.py`` + ``unified_ape_acid.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from evaluation.paths import ensure_eval_paths

ensure_eval_paths()


import subprocess
from pathlib import Path

import acid_eval_common as acid
import re10k_eval_common as re10k

# Patch RE10K helpers *before* importing the shared infer entrypoint so that
# ``from re10k_eval_common import ...`` binds ACID-aware symbols.
re10k.DATA_ROOT = acid.DATA_ROOT
re10k.DEFAULT_EXP_ROOT = acid.DEFAULT_EXP_ROOT
re10k.DEFAULT_ORIG_H = acid.DEFAULT_ORIG_H
re10k.DEFAULT_ORIG_W = acid.DEFAULT_ORIG_W
re10k.DEFAULT_VIDEO_CHUNK_SIZE = acid.DEFAULT_VIDEO_CHUNK_SIZE
re10k.SUMMARY_METRIC_ORDER = acid.SUMMARY_METRIC_ORDER
re10k.SceneSplit = acid.SceneSplit
re10k.SceneLayout = acid.SceneLayout
re10k.resolve_scene_layout = acid.resolve_scene_layout
re10k.discover_scenes = acid.discover_scenes
re10k.discover_split_index = acid.discover_split_index
re10k.resolve_frame_path = acid.resolve_frame_path
re10k.find_metadata_path = acid.find_metadata_path
re10k.load_eval_cameras = acid.load_eval_cameras
re10k.load_rgb = acid.load_rgb
re10k.list_frame_indices = acid.list_frame_indices
re10k.chunk_sequence = acid.chunk_sequence
re10k.mean_or_nan = acid.mean_or_nan
re10k.collect_video_frame_paths = acid.collect_video_frame_paths
re10k.write_summary_metrics_csv = acid.write_summary_metrics_csv
re10k.read_summary_metrics_csv = acid.read_summary_metrics_csv
re10k.compute_common_crop = acid.compute_common_crop

import re10k_benchmark_infer as infer  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
infer.DATA_ROOT = acid.DATA_ROOT
infer.DATA_ROOT_DEFAULT = str(acid.DATA_ROOT)
infer.resolve_scene_layout = acid.resolve_scene_layout
infer.discover_scenes = acid.discover_scenes
infer.resolve_frame_path = acid.resolve_frame_path


def _run_post_inference_metrics(output_dir: Path, *, skip_ape: bool = False) -> None:
    metrics_script = SCRIPT_DIR / "acid_run_metrics.sh"
    if not metrics_script.is_file():
        raise FileNotFoundError(f"Metrics runner not found: {metrics_script}")
    print(f"\n=== Running ACID unified metrics on {output_dir} ===", flush=True)
    subprocess.run(
        ["bash", str(metrics_script), str(output_dir)],
        cwd=str(REPO_ROOT),
        check=True,
    )

    if skip_ape:
        print("Skipping unified ACID APE (--skip_ape).")
        return

    ape_script = SCRIPT_DIR / "acid_run_ape.sh"
    if not ape_script.is_file():
        raise FileNotFoundError(f"APE runner not found: {ape_script}")
    print(f"\n=== Running ACID unified APE on {output_dir} ===", flush=True)
    subprocess.run(
        ["bash", str(ape_script), str(output_dir)],
        cwd=str(REPO_ROOT),
        check=True,
    )


infer._run_post_inference_metrics = _run_post_inference_metrics


def main() -> None:
    infer.main()


if __name__ == "__main__":
    main()
