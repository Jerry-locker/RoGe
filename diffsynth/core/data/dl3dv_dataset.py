"""RoGe training dataset for CUT3R-preprocessed DL3DV scenes.

Expected layout (CUT3R ``processed_dl3dv`` release)::

    <base_path>/
        {1K|2K|...|7K}/
            {scene_hash}/
                dense/
                    rgb/           # *.png
                    cam/           # *.npz (intrinsic, pose)
                    depth/         # *.npy, unused here
                    sky_mask/      # unused here
                    outlier_mask/  # unused here
        DL3DV-10K-Benchmark/       # scene hashes listed here are excluded from 1K–7K

Each sample picks a scene from the configured train splits (default 1K–7K),
excluding any scene hash indexed under ``DL3DV-10K-Benchmark/`` (same protocol
as VGGT). Before sampling, a random contiguous trajectory Q is selected based
on the context count M: 2/5 of the stream for M=3, 8/15 for M=4, 5/6 for M=5,
and the full stream for M>=6. Q always contains at least M + target frames.
Sampling on Q is then a fixed 50/50 mixture:

* **Context-first mode (50%)**
  - Context: random contiguous segment covering [4/5, 1] of Q,
    then K-means for ``num_context_frames`` views (sorted by time).
  - Target: ``num_target_frames`` consecutive entries from the ordered
    complement of context (may skip context positions on the original timeline).

* **Target-first mode (50%)**
  - Target: one of six ordered sampling schemes on Q —
    consecutive frames, fixed strides in {2, 3, 4, 5}, or a random gap
    (``max_interval=5``).
  - Context: K-means ``num_context_frames`` from the remaining positions in Q.

When ``mixed_train=True``, each sample keeps ``num_target_frames`` but draws
``num_context_frames`` uniformly from ``[3, 9]`` (rank-synced via fetch step).
Scene-length gating uses the min context (3); larger M is clamped to what the
stream can host at sample time.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from diffsynth.lagernvs_condition import lagernvs_infer_cover_crop

from .sampling_seed import (
    ResampleSample,
    clamp_mixed_context_frames,
    mixed_train_filter_context_frames,
    resolve_num_context_frames,
    sample_rng,
)
from .dataset import (
    ImageCropAndResize,
    get_plucker_rays,
    iter_directory_names,
    limit_dataset_entries,
    logger,
    apply_pose_gauge,
    resolve_discovery_limit,
)
from .dataset_cache import build_scene_list_with_cache
from .frame_sampling import (
    sample_kmeans_context_complement_target_positions,
    sample_target_first_kmeans_context_positions,
    stream_supports_kmeans_sampling,
)

DL3DV_DEFAULT_TRAIN_SPLITS = ("1K", "2K", "3K", "4K", "5K", "6K", "7K")
DL3DV_BENCHMARK_TIER = "DL3DV-10K-Benchmark"
DL3DV_CONTEXT_SEGMENT_MIN_RATIO = 4 / 5
DL3DV_TRAJECTORY_RATIO_BY_CONTEXT = {
    3: (3 / 6) * (4 / 5),
    4: (4 / 6) * (4 / 5),
    5: 5 / 6,
}


def list_dl3dv_benchmark_scene_hashes(
    base_path: str,
    benchmark_tier: str = DL3DV_BENCHMARK_TIER,
) -> set[str]:
    """Collect scene hashes under the DL3DV benchmark tier for training exclusion."""
    benchmark_dir = os.path.join(base_path, benchmark_tier)
    if not os.path.isdir(benchmark_dir):
        logger.info(
            "DL3DV benchmark tier not found (skip hash exclusion): %s",
            benchmark_dir,
        )
        return set()

    scene_hashes = set()
    for name in os.listdir(benchmark_dir):
        if name.startswith("."):
            continue
        path = os.path.join(benchmark_dir, name)
        if os.path.isdir(path):
            scene_hashes.add(name)

    if scene_hashes:
        logger.info(
            "DL3DV benchmark tier %s: collected %d scene hashes for exclusion",
            benchmark_tier,
            len(scene_hashes),
        )
    else:
        logger.warning(
            "DL3DV benchmark tier %s exists but no scene-hash subdirectories "
            "found under %s",
            benchmark_tier,
            benchmark_dir,
        )
    return scene_hashes


def _discover_scenes(
    base_path: str,
    splits: tuple[str, ...],
    *,
    excluded_scene_hashes: set[str] | None = None,
    max_discover: int | None = None,
    seed: int = 42,
) -> list[dict]:
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"DL3DV base_path not found: {base_path}")

    excluded_hashes = excluded_scene_hashes or set()
    scenes: list[dict] = []
    for split in splits:
        split_dir = os.path.join(base_path, split)
        if not os.path.isdir(split_dir):
            logger.warning(f"DL3DV split not found, skipping: {split_dir}")
            continue

        remaining = None if max_discover is None else max_discover - len(scenes)
        if remaining is not None and remaining <= 0:
            break

        for entry in iter_directory_names(
            split_dir, max_discover=remaining, seed=seed
        ):
            if entry in excluded_hashes:
                continue
            dense_dir = os.path.join(split_dir, entry, "dense")
            rgb_dir = os.path.join(dense_dir, "rgb")
            cam_dir = os.path.join(dense_dir, "cam")
            if not (os.path.isdir(rgb_dir) and os.path.isdir(cam_dir)):
                continue
            scenes.append(
                {
                    "split": split,
                    "scene_hash": entry,
                    "scene_id": f"{split}/{entry}",
                    "dense_dir": dense_dir,
                    "rgb_dir": rgb_dir,
                    "cam_dir": cam_dir,
                }
            )
            if max_discover is not None and len(scenes) >= max_discover:
                return scenes
    return scenes


class DL3DVRoGeDataset(Dataset):
    """Sparse context + consecutive target dataset for DL3DV fine-tuning."""

    def __init__(
        self,
        base_path: str,
        repeat: int = 1,
        num_context_frames: int = 16,
        num_target_frames: int = 81,
        height: int = 480,
        width: int = 832,
        height_division_factor: int = 8,
        width_division_factor: int = 8,
        no_pixel_unshuffle: bool = False,
        raymap_full_resolution: bool = False,
        num_dataset_samples: int | None = None,
        random_subset: bool = False,
        pose_gauge: str = "cam_last_unit",
        context_shuffle: bool = False,
        target_inverse: bool = False,
        mixed_train: bool = False,
        dataset_cache_dir: str | None = None,
        update_dataset_cache: bool = False,
        splits: tuple[str, ...] | list[str] | None = None,
        benchmark_tier: str | None = DL3DV_BENCHMARK_TIER,
        excluded_scene_hashes: tuple[str, ...] | list[str] | None = None,
    ):
        self.base_path = base_path
        self.repeat = repeat
        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.no_pixel_unshuffle = no_pixel_unshuffle
        self.raymap_full_resolution = raymap_full_resolution
        self.pose_gauge = pose_gauge
        self.context_shuffle = context_shuffle
        self.target_inverse = target_inverse
        self.mixed_train = mixed_train
        self.splits = tuple(splits) if splits is not None else DL3DV_DEFAULT_TRAIN_SPLITS
        self.benchmark_tier = benchmark_tier
        self.excluded_scene_hashes = set(excluded_scene_hashes or ())
        if self.benchmark_tier is not None:
            self.excluded_scene_hashes |= list_dl3dv_benchmark_scene_hashes(
                base_path, self.benchmark_tier
            )

        if num_context_frames <= 0:
            raise ValueError(f"num_context_frames must be > 0, got {num_context_frames}")
        if num_target_frames <= 0:
            raise ValueError(f"num_target_frames must be > 0, got {num_target_frames}")

        max_discover = resolve_discovery_limit(num_dataset_samples, random_subset)

        def build_scenes():
            scene_entries = _discover_scenes(
                base_path,
                self.splits,
                excluded_scene_hashes=self.excluded_scene_hashes,
                max_discover=max_discover,
            )
            scene_entries = limit_dataset_entries(
                scene_entries, num_dataset_samples, random_subset
            )
            if not scene_entries:
                raise FileNotFoundError(
                    f"No DL3DV scenes found under {base_path} "
                    f"(expected <split>/<hash>/dense/{{rgb,cam}}/ for splits {self.splits})"
                )

            scenes = []
            for entry in scene_entries:
                info = self._load_scene_info(entry)
                if info is not None:
                    scenes.append(info)
            filter_ctx = mixed_train_filter_context_frames(
                self.mixed_train, num_context_frames
            )
            if not scenes:
                raise RuntimeError(
                    f"No usable DL3DV scenes "
                    f"(need enough frames for context={filter_ctx}, "
                    f"target={num_target_frames}) "
                    f"under {base_path}"
                )
            return scenes

        self.scenes, from_cache = build_scene_list_with_cache(
            dataset_name="dl3dv_no_benchmark_mmin",
            base_path=base_path,
            cache_dir=dataset_cache_dir,
            build_fn=build_scenes,
            update_cache=update_dataset_cache,
        )

        gauge = pose_gauge
        cache_note = ", from cache" if from_cache else ""
        benchmark_note = (
            f", excluded_benchmark={self.benchmark_tier} "
            f"({len(self.excluded_scene_hashes)} hashes)"
            if self.benchmark_tier is not None
            else ""
        )
        if self.mixed_train:
            layout_note = (
                f"mixed_train context=[3,9], target={num_target_frames}"
            )
        else:
            layout_note = (
                f"context={num_context_frames}, target={num_target_frames}"
            )
        print(
            f"DL3DV dataset (splits={','.join(self.splits)}): {len(self.scenes)} scenes"
            f"{cache_note}{benchmark_note}, "
            f"{layout_note}, "
            f"sampling=50/50 "
            f"(M-dependent trajectory Q; "
            f"[{DL3DV_CONTEXT_SEGMENT_MIN_RATIO:g},1]-context-first / target-first), "
            f"pose_gauge={gauge}"
        )

    def _layout_is_usable(
        self, stream_len: int, num_context: int, num_target: int
    ) -> bool:
        if stream_len < num_context + num_target:
            return False
        if stream_supports_kmeans_sampling(
            stream_len,
            num_context,
            num_target,
            min_ratio=DL3DV_CONTEXT_SEGMENT_MIN_RATIO,
        ):
            return True
        return stream_len - num_target >= num_context

    def _stream_is_usable(self, stream_len: int) -> bool:
        filter_ctx = mixed_train_filter_context_frames(
            self.mixed_train, self.num_context_frames
        )
        return self._layout_is_usable(
            stream_len, filter_ctx, self.num_target_frames
        )

    def _resolve_layout(self, stream_len: int) -> tuple[int, int]:
        num_context = resolve_num_context_frames(
            self.num_context_frames, self.mixed_train
        )
        num_context = clamp_mixed_context_frames(
            num_context,
            stream_len,
            self.num_target_frames,
            mixed_train=self.mixed_train,
        )
        return num_context, self.num_target_frames

    def _frame_exists(self, rgb_dir: str, cam_dir: str, image_name: str) -> bool:
        stem = os.path.splitext(image_name)[0]
        rgb_path = os.path.join(rgb_dir, image_name)
        cam_path = os.path.join(cam_dir, f"{stem}.npz")
        return os.path.isfile(rgb_path) and os.path.isfile(cam_path)

    def _load_scene_info(self, entry: dict):
        rgb_dir = entry["rgb_dir"]
        cam_dir = entry["cam_dir"]
        images = sorted(
            name for name in os.listdir(rgb_dir) if name.lower().endswith(".png")
        )
        stream = [
            i for i, name in enumerate(images) if self._frame_exists(rgb_dir, cam_dir, name)
        ]
        if not self._stream_is_usable(len(stream)):
            logger.warning(f"Skipping {entry['scene_id']}: stream too short for sampling")
            return None

        return {
            **entry,
            "images": images,
            "stream": stream,
        }

    def _load_cam(self, cam_dir: str, stem: str) -> tuple[np.ndarray, np.ndarray]:
        path = os.path.join(cam_dir, f"{stem}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing camera for {stem} under {cam_dir}")
        with np.load(path) as data:
            if "intrinsic" in data:
                intrinsics = data["intrinsic"].astype(np.float32)
            elif "intrinsics" in data:
                intrinsics = data["intrinsics"].astype(np.float32)
            else:
                raise KeyError(f"{path} must contain 'intrinsic' or 'intrinsics'")
            pose = data["pose"].astype(np.float32)
        return intrinsics, pose

    def _stream_w2cs(self, scene: dict, stream: list[int]) -> np.ndarray:
        w2cs = []
        for i in stream:
            stem = os.path.splitext(scene["images"][i])[0]
            _, pose = self._load_cam(scene["cam_dir"], stem)
            w2cs.append(np.linalg.inv(pose.astype(np.float64)))
        return np.stack(w2cs, axis=0)

    def _sample_context_first_mode(
        self,
        stream_w2cs: np.ndarray,
        num_context: int,
        num_target: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        return sample_kmeans_context_complement_target_positions(
            len(stream_w2cs),
            stream_w2cs,
            num_context,
            num_target,
            min_ratio=DL3DV_CONTEXT_SEGMENT_MIN_RATIO,
        )

    def _sample_target_first_mode(
        self,
        stream_w2cs: np.ndarray,
        num_context: int,
        num_target: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        stream_len = len(stream_w2cs)
        intervals: list[int | None] = []
        if stream_len >= num_target:
            intervals.append(1)
        for interval in (2, 3, 4, 5):
            if 1 + (num_target - 1) * interval <= stream_len:
                intervals.append(interval)
        intervals.append(None)
        if not intervals:
            return None
        interval = sample_rng(salt=16).choice(intervals)
        return sample_target_first_kmeans_context_positions(
            stream_len,
            stream_w2cs,
            num_context,
            num_target,
            interval=interval,
            max_interval=5,
        )

    def _sample_frame_positions(
        self,
        scene: dict,
        stream: list[int],
        num_context: int,
        num_target: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        ratio = DL3DV_TRAJECTORY_RATIO_BY_CONTEXT.get(num_context)
        q_start = 0
        q_stream = stream
        if ratio is not None:
            q_len = min(
                len(stream),
                max(num_context + num_target, math.ceil(len(stream) * ratio)),
            )
            q_start = sample_rng(salt=23).randint(0, len(stream) - q_len)
            q_stream = stream[q_start : q_start + q_len]

        stream_w2cs = self._stream_w2cs(scene, q_stream)
        if sample_rng(salt=14).random() < 0.5:
            samplers = (
                self._sample_context_first_mode,
                self._sample_target_first_mode,
            )
        else:
            samplers = (
                self._sample_target_first_mode,
                self._sample_context_first_mode,
            )
        for sampler in samplers:
            sampled = sampler(stream_w2cs, num_context, num_target)
            if sampled is not None:
                context_positions, target_positions = sampled
                return context_positions + q_start, target_positions + q_start
        return None

    def _sample_frame_indices(
        self, scene: dict, stream: list[int]
    ) -> tuple[list[int], int, int] | None:
        num_context, num_target = self._resolve_layout(len(stream))
        sampled = self._sample_frame_positions(
            scene, stream, num_context, num_target
        )

        if sampled is None:
            return None

        context_positions, target_positions = sampled
        context_indices = [stream[pos] for pos in context_positions]
        if self.context_shuffle:
            sample_rng(salt=13).shuffle(context_indices)
        target_indices = [stream[pos] for pos in target_positions]
        if self.target_inverse and sample_rng(salt=19).random() < 0.5:
            target_indices = target_indices[::-1]
        return context_indices + target_indices, num_context, num_target

    def __len__(self):
        return len(self.scenes) * self.repeat

    def _scale_intrinsics(self, k: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        scale = max(self.width / orig_w, self.height / orig_h)
        resized_w = round(orig_w * scale)
        resized_h = round(orig_h * scale)
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        fx_s, fy_s = fx * scale, fy * scale
        cx_s, cy_s = cx * scale, cy * scale
        crop_x = (resized_w - self.width) / 2.0
        crop_y = (resized_h - self.height) / 2.0
        return np.array(
            [
                [fx_s, 0, cx_s - crop_x],
                [0, fy_s, cy_s - crop_y],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    def _load_frame(self, rgb_dir: str, image_name: str, process: ImageCropAndResize):
        path = os.path.join(rgb_dir, image_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing RGB for {image_name} under {rgb_dir}")
        return process(Image.open(path).convert("RGB"))

    def __getitem__(self, idx):
        scene = self.scenes[idx % len(self.scenes)]
        stream = scene["stream"]
        sampled = self._sample_frame_indices(scene, stream)
        if sampled is None:
            raise ResampleSample("sample retry")
        frame_indices, num_context, num_output = sampled

        process = ImageCropAndResize(
            height=self.height,
            width=self.width,
            max_pixels=1920 * 1080,
            height_division_factor=self.height_division_factor,
            width_division_factor=self.width_division_factor,
        )

        all_frames = []
        lagernvs_context_images = []
        lagernvs_target_crops = []
        intrinsics_list = []
        native_intrinsics_list = []
        trajectories = []
        for i, frame_i in enumerate(frame_indices):
            image_name = scene["images"][frame_i]
            stem = os.path.splitext(image_name)[0]
            rgb_path = os.path.join(scene["rgb_dir"], image_name)
            try:
                with Image.open(rgb_path) as img:
                    img = img.convert("RGB")
                    ow, oh = img.size
                    all_frames.append(process(img))
                    if i < num_context:
                        lagernvs_context_images.append(img.copy())
                    else:
                        # Targets are only ever NVS-loss GT at LagerNVS
                        # infer res, so crop here instead of shipping a
                        # full-resolution PIL to the training process.
                        lagernvs_target_crops.append(
                            lagernvs_infer_cover_crop(
                                img, self.height, self.width
                            )
                        )
                intrinsics, pose = self._load_cam(scene["cam_dir"], stem)
            except (FileNotFoundError, OSError, KeyError):
                raise ResampleSample("sample retry")
            native_intrinsics_list.append(np.asarray(intrinsics, dtype=np.float32).copy())
            intrinsics_list.append(self._scale_intrinsics(intrinsics, ow, oh))
            trajectories.append(pose)

        input_images = all_frames[:num_context]
        target_images = all_frames

        intrinsics = np.stack(intrinsics_list, axis=0)
        w2cs = torch.from_numpy(np.linalg.inv(np.stack(trajectories, axis=0))).float()
        context_w2c = w2cs[:num_context].numpy()
        target_w2c = w2cs[num_context:].numpy()
        context_intrinsics = intrinsics[:num_context]
        target_intrinsics = intrinsics[num_context:]
        _, c2w_norm, _ = apply_pose_gauge(w2cs, self.pose_gauge, num_context=num_context)
        raymap = get_plucker_rays(
            c2w_norm,
            torch.from_numpy(intrinsics).float(),
            height=self.height,
            width=self.width,
            no_pixel_unshuffle=self.no_pixel_unshuffle,
            full_resolution=self.raymap_full_resolution,
        )

        sample = {
            "input_images": input_images,
            "target_images": target_images,
            "raymap": raymap,
            "context_w2c": context_w2c,
            "target_w2c": target_w2c,
            "context_intrinsics": context_intrinsics,
            "target_intrinsics": target_intrinsics,
            "scene_name": scene["scene_id"],
            "prompt": "",
            "num_output_frames": num_output,
        }
        native_k = np.stack(native_intrinsics_list, axis=0)
        sample["lagernvs_context_images"] = lagernvs_context_images
        sample["lagernvs_context_intrinsics"] = native_k[:num_context]
        sample["lagernvs_target_intrinsics"] = native_k[num_context:]
        # (V, 288, 504, 3) uint8, already cover-cropped for the NVS loss.
        sample["lagernvs_target_images"] = torch.from_numpy(
            np.stack(lagernvs_target_crops, axis=0)
        )
        return sample
