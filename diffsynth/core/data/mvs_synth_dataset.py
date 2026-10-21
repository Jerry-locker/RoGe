"""RoGe training dataset for CUT3R-preprocessed MVS-Synth scenes.

Expected layout (``CUT3R/datasets_preprocess/preprocess_mvs_synth.py``)::

    <base_path>/                         # e.g. processed_mvs_synth/
        <seq_id>/
            rgb/        0000.jpg ...
            depth/      0000.npy ...      # unused here
            cam/        0000.npz ...      # intrinsics (3x3), pose (4x4 c2w)

Each sample picks a train scene. Sampling is a fixed 60/40 mixture (not
controlled by ``data_sampling_strategy``):

* **Long-segment mode (60%)**
  - Context: random contiguous segment covering [3/5, 1] of the stream, then
    K-means for ``num_context_frames`` views (sorted by time).
  - Target: ``num_target_frames`` ordered frames from the remaining stream
    positions, with random gaps (``max_interval=5``) between consecutive picks.

* **Equal-partition mode (40%)**
  - Context: divide the stream into ``num_context_frames`` equal parts and pick
    one random frame from each (sorted by time).
  - Target: same complement interval sampling as long-segment mode.

Returns the same keys as other RoGe sparse-context datasets:
  ``input_images``, ``target_images``, ``raymap``, ``context_w2c``,
  ``context_intrinsics``, ``scene_name``, ``prompt``, ``num_output_frames``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

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
    sample_equal_partition_context_positions,
    sample_interval_target_from_complement,
    sample_kmeans_context_in_segment,
    stream_supports_kmeans_sampling,
)

MVSSYNTH_SEGMENT_MIN_RATIO = 3 / 5
MVSSYNTH_LONG_SEGMENT_MODE_PROB = 0.6
MVSSYNTH_TARGET_MAX_INTERVAL = 5


@dataclass(frozen=True)
class _MVSSynthScene:
    scene_dir: str
    rgb_dir: str
    cam_dir: str
    frame_ids: list[str]
    trajectories: np.ndarray


def _frame_sort_key(name: str):
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)


def _list_available_frame_ids(rgb_dir: str, cam_dir: str) -> list[str]:
    """Basenames that have both rgb/<id>.jpg and cam/<id>.npz."""
    ids: list[str] = []
    if not os.path.isdir(rgb_dir) or not os.path.isdir(cam_dir):
        return ids
    for name in os.listdir(rgb_dir):
        if not name.lower().endswith(".jpg"):
            continue
        stem = os.path.splitext(name)[0]
        if os.path.isfile(os.path.join(cam_dir, f"{stem}.npz")):
            ids.append(stem)
    return sorted(ids, key=_frame_sort_key)


def _discover_scene_dirs(
    base_path: str,
    *,
    max_discover: int | None = None,
    seed: int = 42,
) -> list[str]:
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"MVS-Synth base_path not found: {base_path}")

    scene_dirs: list[str] = []
    for seq_name in iter_directory_names(base_path, max_discover=max_discover, seed=seed):
        scene_dir = os.path.join(base_path, seq_name)
        if not os.path.isdir(scene_dir):
            continue
        rgb_dir = os.path.join(scene_dir, "rgb")
        cam_dir = os.path.join(scene_dir, "cam")
        if os.path.isdir(rgb_dir) and os.path.isdir(cam_dir):
            scene_dirs.append(scene_dir)
            if max_discover is not None and len(scene_dirs) >= max_discover:
                break
    return scene_dirs


class MVSSynthRoGeDataset(Dataset):
    """Context/target dataset for MVS-Synth with 60/40 sampling."""

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
        no_pixel_unshuffle: bool = False,
        raymap_full_resolution: bool = False,
        num_dataset_samples: int | None = None,
        random_subset: bool = False,
        pose_gauge: str = "cam_last_unit",
        mixed_train: bool = False,
        dataset_cache_dir: str | None = None,
        update_dataset_cache: bool = False,
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
        self.mixed_train = mixed_train

        if num_context_frames <= 0:
            raise ValueError(f"num_context_frames must be > 0, got {num_context_frames}")
        if num_target_frames <= 0:
            raise ValueError(f"num_target_frames must be > 0, got {num_target_frames}")

        max_discover = resolve_discovery_limit(num_dataset_samples, random_subset)
        filter_ctx = mixed_train_filter_context_frames(mixed_train, num_context_frames)

        def build_scenes():
            scene_dirs = _discover_scene_dirs(base_path, max_discover=max_discover)
            scene_dirs = limit_dataset_entries(
                scene_dirs, num_dataset_samples, random_subset
            )
            if not scene_dirs:
                raise FileNotFoundError(
                    f"No MVS-Synth scenes found under {base_path} "
                    f"(expected <seq>/rgb/ and <seq>/cam/)"
                )
            scenes: list[_MVSSynthScene] = []
            for scene_dir in scene_dirs:
                info = self._load_scene_info(scene_dir)
                if info is not None:
                    scenes.append(info)
            if not scenes:
                raise RuntimeError(
                    f"No usable MVS-Synth scenes "
                    f"(>= {filter_ctx + num_target_frames} frames for "
                    f"60/40 context-target sampling) under {base_path}"
                )
            return scenes

        self.scenes, from_cache = build_scene_list_with_cache(
            dataset_name="mvssynth_mmin",
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
            f"MVS-Synth dataset: {len(self.scenes)} scenes{cache_note}, "
            f"{layout_note}, "
            f"sampling=60/40 "
            f"([{MVSSYNTH_SEGMENT_MIN_RATIO:g},1] k-means / equal-partition; "
            f"complement interval target max={MVSSYNTH_TARGET_MAX_INTERVAL}), "
            f"pose_gauge={gauge}"
        )

    def _stream_is_usable(self, stream_len: int) -> bool:
        filter_ctx = mixed_train_filter_context_frames(
            self.mixed_train, self.num_context_frames
        )
        return stream_len >= filter_ctx + self.num_target_frames

    def _load_cam(self, cam_dir: str, frame_id: str) -> tuple[np.ndarray, np.ndarray]:
        path = os.path.join(cam_dir, f"{frame_id}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing camera for {frame_id} under {cam_dir}")
        with np.load(path) as data:
            intrinsics = data["intrinsics"].astype(np.float32)
            c2w = data["pose"].astype(np.float32)
        return intrinsics, c2w

    def _load_scene_info(self, scene_dir: str) -> _MVSSynthScene | None:
        rgb_dir = os.path.join(scene_dir, "rgb")
        cam_dir = os.path.join(scene_dir, "cam")
        frame_ids = _list_available_frame_ids(rgb_dir, cam_dir)

        if not self._stream_is_usable(len(frame_ids)):
            logger.warning(f"Skipping {scene_dir}: stream too short for sampling")
            return None

        trajectories: list[np.ndarray] = []
        for frame_id in frame_ids:
            try:
                _, c2w = self._load_cam(cam_dir, frame_id)
            except (FileNotFoundError, OSError, KeyError):
                logger.warning(f"Skipping {scene_dir}: missing camera for frame {frame_id}")
                return None
            trajectories.append(c2w)

        return _MVSSynthScene(
            scene_dir=scene_dir,
            rgb_dir=rgb_dir,
            cam_dir=cam_dir,
            frame_ids=frame_ids,
            trajectories=np.stack(trajectories, axis=0).astype(np.float32),
        )

    def _stream_w2cs(self, scene: _MVSSynthScene) -> np.ndarray:
        return np.linalg.inv(scene.trajectories.astype(np.float64))

    def _complement_positions(
        self,
        stream_len: int,
        context_positions: np.ndarray,
    ) -> np.ndarray:
        context_set = set(context_positions.tolist())
        return np.asarray(
            [pos for pos in range(stream_len) if pos not in context_set],
            dtype=np.int64,
        )

    def _sample_target_from_complement(
        self,
        stream_len: int,
        context_positions: np.ndarray,
    ) -> np.ndarray | None:
        remaining = self._complement_positions(stream_len, context_positions)
        return sample_interval_target_from_complement(
            remaining,
            self.num_target_frames,
            max_interval=MVSSYNTH_TARGET_MAX_INTERVAL,
        )

    def _sample_long_segment_mode(
        self,
        stream_w2cs: np.ndarray,
        num_context: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        stream_len = len(stream_w2cs)
        if not stream_supports_kmeans_sampling(
            stream_len,
            num_context,
            self.num_target_frames,
            min_ratio=MVSSYNTH_SEGMENT_MIN_RATIO,
        ):
            return None
        context_positions = sample_kmeans_context_in_segment(
            stream_len,
            stream_w2cs,
            num_context,
            min_ratio=MVSSYNTH_SEGMENT_MIN_RATIO,
        )
        if context_positions is None:
            return None
        target_positions = self._sample_target_from_complement(
            stream_len, context_positions
        )
        if target_positions is None:
            return None
        return context_positions, target_positions

    def _sample_equal_partition_mode(
        self,
        stream_w2cs: np.ndarray,
        num_context: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        stream_len = len(stream_w2cs)
        context_positions = sample_equal_partition_context_positions(
            stream_len,
            num_context,
        )
        if context_positions is None:
            return None
        target_positions = self._sample_target_from_complement(
            stream_len, context_positions
        )
        if target_positions is None:
            return None
        return context_positions, target_positions

    def _sample_frame_ids(self, scene: _MVSSynthScene) -> tuple[list[str], int] | None:
        num_context = resolve_num_context_frames(
            self.num_context_frames, self.mixed_train
        )
        stream_w2cs = self._stream_w2cs(scene)
        num_context = clamp_mixed_context_frames(
            num_context,
            len(stream_w2cs),
            self.num_target_frames,
            mixed_train=self.mixed_train,
        )
        if sample_rng(salt=14).random() < MVSSYNTH_LONG_SEGMENT_MODE_PROB:
            samplers = (
                self._sample_long_segment_mode,
                self._sample_equal_partition_mode,
            )
        else:
            samplers = (
                self._sample_equal_partition_mode,
                self._sample_long_segment_mode,
            )

        sampled = None
        for sampler in samplers:
            sampled = sampler(stream_w2cs, num_context)
            if sampled is not None:
                break
        if sampled is None:
            return None

        context_positions, target_positions = sampled
        frame_ids = scene.frame_ids
        context_ids = [frame_ids[pos] for pos in context_positions]
        target_ids = [frame_ids[pos] for pos in target_positions]
        return context_ids + target_ids, num_context

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

    def _load_rgb(self, rgb_dir: str, frame_id: str, process: ImageCropAndResize):
        path = os.path.join(rgb_dir, f"{frame_id}.jpg")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing RGB for {frame_id} under {rgb_dir}")
        return process(Image.open(path).convert("RGB"))

    def __getitem__(self, idx):
        scene = self.scenes[idx % len(self.scenes)]
        sampled = self._sample_frame_ids(scene)
        if sampled is None:
            raise ResampleSample("sample retry")
        frame_ids, num_context = sampled
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
        intrinsics_list: list[np.ndarray] = []
        native_intrinsics_list: list[np.ndarray] = []
        c2ws: list[np.ndarray] = []
        for i, fid in enumerate(frame_ids):
            rgb_path = os.path.join(scene.rgb_dir, f"{fid}.jpg")
            try:
                with Image.open(rgb_path) as img:
                    img = img.convert("RGB")
                    ow, oh = img.size
                    all_frames.append(process(img))
                    if i < num_context:
                        lagernvs_context_images.append(img.copy())
                    else:
                        lagernvs_target_images.append(img.copy())
                k, c2w = self._load_cam(scene.cam_dir, fid)
            except (FileNotFoundError, OSError, KeyError):
                raise ResampleSample("sample retry")
            native_intrinsics_list.append(np.asarray(k, dtype=np.float32).copy())
            intrinsics_list.append(self._scale_intrinsics(k, ow, oh))
            c2ws.append(c2w)

        input_images = all_frames[:num_context]
        target_images = all_frames

        intrinsics = np.stack(intrinsics_list, axis=0)
        w2cs = torch.from_numpy(np.linalg.inv(np.stack(c2ws, axis=0))).float()
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

        scene_name = os.path.relpath(scene.scene_dir, self.base_path).replace(os.sep, "/")

        sample = {
            "input_images": input_images,
            "target_images": target_images,
            "raymap": raymap,
            "context_w2c": context_w2c,
            "target_w2c": target_w2c,
            "context_intrinsics": context_intrinsics,
            "target_intrinsics": target_intrinsics,
            "scene_name": scene_name,
            "prompt": "",
            "num_output_frames": num_output,
        }
        native_k = np.stack(native_intrinsics_list, axis=0)
        sample["lagernvs_context_images"] = lagernvs_context_images
        sample["lagernvs_context_intrinsics"] = native_k[:num_context]
        sample["lagernvs_target_intrinsics"] = native_k[num_context:]
        sample["lagernvs_target_images"] = lagernvs_target_images
        return sample
