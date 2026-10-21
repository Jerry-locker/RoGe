"""RoGe training dataset for CUT3R-preprocessed ARKitScenes.

Expected layout (``preprocess_arkitscenes.py`` + ``generate_set_arkitscenes.py``)::

    <base_path>/
        Training/
            {scene_id}/
                vga_wide/            # RGB (.jpg)
                lowres_depth/        # depth (.png), unused here
                new_scene_metadata.npz

Each sample picks a scene from ``Training`` only (Test/Validation are never
loaded) and uses the single temporal stream (sorted by frame timestamp).

Sampling is a fixed 50/50 mixture (not controlled by
``data_sampling_strategy``):

* **Context-first mode (50%)**
  - Context: random contiguous segment covering [4/5, 1] of the stream,
    then K-means for ``num_context_frames`` views (sorted by time).
  - Target: ``num_target_frames`` consecutive entries from the ordered
    complement of context (may skip context positions on the original timeline).

* **Target-first mode (50%)**
  - Target: one of six ordered sampling schemes on the full stream —
    consecutive frames, fixed strides in {2, 3, 4, 5}, or a random gap
    (``max_interval=5``).
  - Context: K-means ``num_context_frames`` from the remaining stream positions.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .arkitscenes_utils import (
    resolve_image_path,
    sort_indices_by_timestamp,
    tuple_intrinsics_to_matrix,
)
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

ARKITSCENES_TRAIN_SPLIT = "Training"
ARKITSCENES_CONTEXT_SEGMENT_MIN_RATIO = 4 / 5


def _train_split_dir(base_path: str) -> str:
    split_dir = os.path.join(base_path, ARKITSCENES_TRAIN_SPLIT)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"ARKitScenes Training split not found: {split_dir}"
        )
    return split_dir


def _discover_scenes(
    data_root: str,
    metadata_name: str,
    *,
    max_discover: int | None = None,
    seed: int = 42,
) -> list[str]:
    scenes = []
    for entry in iter_directory_names(data_root, max_discover=max_discover, seed=seed):
        scene_dir = os.path.join(data_root, entry)
        if not os.path.isdir(scene_dir):
            continue
        meta = os.path.join(scene_dir, metadata_name)
        images_dir = os.path.join(scene_dir, "vga_wide")
        if os.path.isfile(meta) and os.path.isdir(images_dir):
            scenes.append(scene_dir)
            if max_discover is not None and len(scenes) >= max_discover:
                break
    return scenes


class ARKitScenesRoGeDataset(Dataset):
    """Sparse context + consecutive target dataset for ARKitScenes fine-tuning."""

    def __init__(
        self,
        base_path: str,
        repeat: int = 1,
        num_context_frames: int = 16,
        num_target_frames: int = 5,
        height: int = 480,
        width: int = 832,
        height_division_factor: int = 8,
        width_division_factor: int = 8,
        metadata_name: str = "new_scene_metadata.npz",
        no_pixel_unshuffle: bool = False,
        raymap_full_resolution: bool = False,
        num_dataset_samples: int | None = None,
        random_subset: bool = False,
        pose_gauge: str = "cam_last_unit",
        mixed_train: bool = False,
        dataset_label: str = "ARKitScenes",
        dataset_cache_dir: str | None = None,
        update_dataset_cache: bool = False,
    ):
        self.base_path = base_path
        self.data_root = _train_split_dir(base_path)
        self.repeat = repeat
        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.metadata_name = metadata_name
        self.no_pixel_unshuffle = no_pixel_unshuffle
        self.raymap_full_resolution = raymap_full_resolution
        self.pose_gauge = pose_gauge
        self.mixed_train = mixed_train
        self.dataset_label = dataset_label

        if num_context_frames <= 0:
            raise ValueError(f"num_context_frames must be > 0, got {num_context_frames}")
        if num_target_frames <= 0:
            raise ValueError(f"num_target_frames must be > 0, got {num_target_frames}")

        max_discover = resolve_discovery_limit(num_dataset_samples, random_subset)
        filter_ctx = mixed_train_filter_context_frames(mixed_train, num_context_frames)

        def build_scenes():
            scene_dirs = _discover_scenes(
                self.data_root, metadata_name, max_discover=max_discover
            )
            scene_dirs = limit_dataset_entries(
                scene_dirs, num_dataset_samples, random_subset
            )
            if not scene_dirs:
                raise FileNotFoundError(
                    f"No ARKitScenes scenes found under {self.data_root} "
                    f"(expected <scene>/{metadata_name} and <scene>/vga_wide/)"
                )
            scenes = []
            for scene_dir in scene_dirs:
                info = self._load_scene_info(scene_dir)
                if info is not None:
                    scenes.append(info)
            if not scenes:
                raise RuntimeError(
                    f"No usable ARKitScenes scenes "
                    f"(>= {filter_ctx + num_target_frames} frames for "
                    f"50/50 context-target sampling) under {self.data_root}"
                )
            return scenes

        cache_dataset_name = dataset_label.lower().replace(" ", "_") + "_mmin"
        self.scenes, from_cache = build_scene_list_with_cache(
            dataset_name=cache_dataset_name,
            base_path=base_path,
            cache_dir=dataset_cache_dir,
            build_fn=build_scenes,
            update_cache=update_dataset_cache,
        )

        gauge = pose_gauge
        cache_note = ", from cache" if from_cache else ""
        if mixed_train:
            layout_note = f"mixed_train context=[3,9], target={num_target_frames}"
        else:
            layout_note = f"context={num_context_frames}, target={num_target_frames}"
        print(
            f"{self.dataset_label} dataset ({ARKITSCENES_TRAIN_SPLIT}): {len(self.scenes)} scenes{cache_note}, "
            f"{layout_note}, "
            f"sampling=50/50 "
            f"([{ARKITSCENES_CONTEXT_SEGMENT_MIN_RATIO:g},1]-context-first / target-first), "
            f"pose_gauge={gauge}"
        )

    def _stream_is_usable(self, stream_len: int) -> bool:
        num_context = mixed_train_filter_context_frames(
            self.mixed_train, self.num_context_frames
        )
        num_target = self.num_target_frames
        if stream_len < num_context + num_target:
            return False
        if stream_supports_kmeans_sampling(
            stream_len,
            num_context,
            num_target,
            min_ratio=ARKITSCENES_CONTEXT_SEGMENT_MIN_RATIO,
        ):
            return True
        return stream_len - num_target >= num_context

    def _load_scene_info(self, scene_dir: str):
        meta_path = os.path.join(scene_dir, self.metadata_name)
        images_dir = os.path.join(scene_dir, "vga_wide")
        with np.load(meta_path, allow_pickle=True) as data:
            images = list(data["images"])
            intrinsics = data["intrinsics"]
            trajectories = data["trajectories"].astype(np.float32)

        stream = []
        for i, name in enumerate(images):
            try:
                resolve_image_path(images_dir, name)
                stream.append(i)
            except FileNotFoundError:
                continue

        stream = sort_indices_by_timestamp(images, stream)
        if not self._stream_is_usable(len(stream)):
            logger.warning(f"Skipping {scene_dir}: stream too short for sampling")
            return None

        return {
            "scene_dir": scene_dir,
            "images_dir": images_dir,
            "images": images,
            "intrinsics": intrinsics,
            "trajectories": trajectories,
            "stream": stream,
        }

    def _stream_w2cs(self, scene: dict, stream: list[int]) -> np.ndarray:
        c2w = scene["trajectories"][stream].astype(np.float64)
        return np.linalg.inv(c2w)

    def _sample_context_first_mode(
        self,
        stream_w2cs: np.ndarray,
        num_context: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        return sample_kmeans_context_complement_target_positions(
            len(stream_w2cs),
            stream_w2cs,
            num_context,
            self.num_target_frames,
            min_ratio=ARKITSCENES_CONTEXT_SEGMENT_MIN_RATIO,
        )

    def _sample_target_first_mode(
        self,
        stream_w2cs: np.ndarray,
        num_context: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        stream_len = len(stream_w2cs)
        num_target = self.num_target_frames
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

    def _sample_frame_indices(
        self, scene: dict, stream: list[int]
    ) -> tuple[list[int], int] | None:
        num_context = resolve_num_context_frames(
            self.num_context_frames, self.mixed_train
        )
        num_context = clamp_mixed_context_frames(
            num_context,
            len(stream),
            self.num_target_frames,
            mixed_train=self.mixed_train,
        )
        stream_w2cs = self._stream_w2cs(scene, stream)
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

        sampled = None
        for sampler in samplers:
            sampled = sampler(stream_w2cs, num_context)
            if sampled is not None:
                break
        if sampled is None:
            return None

        context_positions, target_positions = sampled
        context_indices = [stream[pos] for pos in context_positions]
        target_indices = [stream[pos] for pos in target_positions]
        return context_indices + target_indices, num_context

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

    def _load_frame(self, images_dir: str, frame_name: str, process: ImageCropAndResize):
        path = resolve_image_path(images_dir, frame_name)
        return process(Image.open(path).convert("RGB"))

    def __getitem__(self, idx):
        scene = self.scenes[idx % len(self.scenes)]
        stream = scene["stream"]
        sampled = self._sample_frame_indices(scene, stream)
        if sampled is None:
            raise ResampleSample("sample retry")
        frame_indices, num_context = sampled
        num_output = self.num_target_frames

        process = ImageCropAndResize(
            height=self.height,
            width=self.width,
            max_pixels=1920 * 1080,
            height_division_factor=self.height_division_factor,
            width_division_factor=self.width_division_factor,
        )

        all_frames = []
        lagernvs_context_images = []
        lagernvs_target_images = []
        intrinsics_list = []
        native_intrinsics_list = []
        for i, frame_i in enumerate(frame_indices):
            frame_name = scene["images"][frame_i]
            try:
                path = resolve_image_path(scene["images_dir"], frame_name)
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    ow, oh = img.size
                    all_frames.append(process(img))
                    if i < num_context:
                        lagernvs_context_images.append(img.copy())
                    else:
                        lagernvs_target_images.append(img.copy())
                k = tuple_intrinsics_to_matrix(scene["intrinsics"][frame_i]).astype(
                    np.float32
                )
            except (FileNotFoundError, OSError, KeyError, ValueError):
                raise ResampleSample("sample retry")
            native_intrinsics_list.append(np.asarray(k, dtype=np.float32).copy())
            intrinsics_list.append(self._scale_intrinsics(k, ow, oh))

        input_images = all_frames[:num_context]
        target_images = all_frames

        intrinsics = np.stack(intrinsics_list, axis=0)
        w2cs = torch.from_numpy(np.linalg.inv(scene["trajectories"][frame_indices])).float()
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
            "scene_name": os.path.basename(scene["scene_dir"]),
            "prompt": "",
            "num_output_frames": num_output,
        }
        native_k = np.stack(native_intrinsics_list, axis=0)
        sample["lagernvs_context_images"] = lagernvs_context_images
        sample["lagernvs_context_intrinsics"] = native_k[:num_context]
        sample["lagernvs_target_intrinsics"] = native_k[num_context:]
        sample["lagernvs_target_images"] = lagernvs_target_images
        return sample
