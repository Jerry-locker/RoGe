"""RoGe training dataset for CUT3R-preprocessed BlendedMVS scenes.

Expected layout (``preprocess_blendedmvs.py``)::

    <base_path>/
        new_overlap.h5                  # optional overlap index from CUT3R
        <scene_id_24hex>/               # e.g. 5a3b2c1d0000000000000001
            00000001.jpg
            00000001.npz                # intrinsics, R_cam2world, t_cam2world
            00000002.jpg
            00000002.npz
            ...

Each sample picks a scene and uses views sorted by numeric basename as a
pseudo-temporal stream. Sampling is a single fixed-segment strategy (not
controlled by ``data_sampling_strategy``):

* Context: random contiguous ``segment_len``-frame segment (default
  ``num_context_frames + num_target_frames``, i.e. 87 for 6+81), then K-means
  for ``num_context_frames`` views (sorted by time).
* Target: the remaining ``num_target_frames`` frames inside that segment, in
  temporal order (81 consecutive complement entries when ``segment_len`` equals
  ``num_context_frames + num_target_frames``).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .sampling_seed import (
    ResampleSample,
    clamp_mixed_context_frames,
    mixed_train_filter_context_frames,
    resolve_num_context_frames,
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
    stream_supports_fixed_segment_sampling,
)

BLENDEDMVS_OVERLAP_INDEX = "new_overlap.h5"


def _decode_basename(name) -> str:
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def _view_sort_key(name: str) -> int:
    try:
        return int(name)
    except ValueError:
        return name.__hash__()


def _view_exists(scene_dir: str, basename: str) -> bool:
    image_path = os.path.join(scene_dir, f"{basename}.jpg")
    meta_path = os.path.join(scene_dir, f"{basename}.npz")
    return os.path.isfile(image_path) and os.path.isfile(meta_path)


def _list_views(scene_dir: str) -> list[str]:
    views = []
    for entry in os.listdir(scene_dir):
        if not entry.endswith(".jpg"):
            continue
        basename = entry[:-4]
        if _view_exists(scene_dir, basename):
            views.append(basename)
    return sorted(views, key=_view_sort_key)


def _discover_scenes_from_overlap(
    base_path: str,
    *,
    max_discover: int | None = None,
    seed: int = 42,
) -> list[str] | None:
    overlap_path = os.path.join(base_path, BLENDEDMVS_OVERLAP_INDEX)
    if not os.path.isfile(overlap_path):
        return None
    try:
        import h5py
    except ImportError:
        logger.warning(
            f"{overlap_path} exists but h5py is not installed; "
            "falling back to filesystem scene discovery"
        )
        return None

    scenes = []
    with h5py.File(overlap_path, "r") as f:
        scene_ids = list(f.keys())
        if max_discover is not None:
            random.Random(seed).shuffle(scene_ids)
        else:
            scene_ids.sort()
        for scene_id in scene_ids:
            scene_dir = os.path.join(base_path, scene_id)
            if not os.path.isdir(scene_dir):
                continue
            basenames = [_decode_basename(x) for x in f[scene_id]["basenames"][:]]
            if any(_view_exists(scene_dir, name) for name in basenames):
                scenes.append(scene_dir)
                if max_discover is not None and len(scenes) >= max_discover:
                    break
    return scenes


def _discover_scenes(
    base_path: str,
    *,
    max_discover: int | None = None,
    seed: int = 42,
) -> list[str]:
    scenes = _discover_scenes_from_overlap(
        base_path, max_discover=max_discover, seed=seed
    )
    if scenes is not None:
        return scenes

    scenes = []
    for entry in iter_directory_names(base_path, max_discover=max_discover, seed=seed):
        if len(entry) != 24:
            continue
        scene_dir = os.path.join(base_path, entry)
        if not os.path.isdir(scene_dir):
            continue
        if _list_views(scene_dir):
            scenes.append(scene_dir)
            if max_discover is not None and len(scenes) >= max_discover:
                break
    return scenes


class BlendedMVSRoGeDataset(Dataset):
    """Sparse context + consecutive target dataset for BlendedMVS fine-tuning."""

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
        segment_len: int | None = None,
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
        filter_ctx = mixed_train_filter_context_frames(mixed_train, num_context_frames)
        if segment_len is None:
            segment_len = filter_ctx + num_target_frames
        self.segment_len = segment_len

        if num_context_frames <= 0:
            raise ValueError(f"num_context_frames must be > 0, got {num_context_frames}")
        if num_target_frames <= 0:
            raise ValueError(f"num_target_frames must be > 0, got {num_target_frames}")
        if segment_len < filter_ctx + num_target_frames:
            raise ValueError(
                f"segment_len must be >= filter_context + num_target_frames "
                f"({filter_ctx + num_target_frames}), got {segment_len}"
            )

        max_discover = resolve_discovery_limit(num_dataset_samples, random_subset)

        def build_scenes():
            scene_dirs = _discover_scenes(base_path, max_discover=max_discover)
            scene_dirs = limit_dataset_entries(
                scene_dirs, num_dataset_samples, random_subset
            )
            if not scene_dirs:
                raise FileNotFoundError(
                    f"No BlendedMVS scenes found under {base_path} "
                    f"(expected 24-char scene folders with matching .jpg/.npz pairs)"
                )
            scenes = []
            for scene_dir in scene_dirs:
                info = self._load_scene_info(scene_dir)
                if info is not None:
                    scenes.append(info)
            if not scenes:
                raise RuntimeError(
                    f"No usable BlendedMVS scenes "
                    f"(>= {self.segment_len} views for fixed-segment K-means sampling) "
                    f"under {base_path}"
                )
            return scenes

        self.scenes, from_cache = build_scene_list_with_cache(
            dataset_name="blendedmvs_mmin",
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
            f"BlendedMVS dataset: {len(self.scenes)} scenes{cache_note}, "
            f"{layout_note}, "
            f"sampling=fixed-segment-(M+N) "
            f"(K-means context + remaining target frames), "
            f"pose_gauge={gauge}"
        )

    def _stream_is_usable(self, stream_len: int) -> bool:
        filter_ctx = mixed_train_filter_context_frames(
            self.mixed_train, self.num_context_frames
        )
        min_segment = filter_ctx + self.num_target_frames
        return stream_supports_fixed_segment_sampling(
            stream_len,
            max(self.segment_len, min_segment),
            filter_ctx,
            self.num_target_frames,
        )

    def _load_scene_info(self, scene_dir: str):
        stream = _list_views(scene_dir)
        if not self._stream_is_usable(len(stream)):
            logger.warning(f"Skipping {scene_dir}: stream too short for sampling")
            return None

        return {
            "scene_dir": scene_dir,
            "scene_name": os.path.basename(scene_dir),
            "stream": stream,
        }

    def _stream_w2cs(self, scene_dir: str, stream: list[str]) -> np.ndarray:
        w2cs = []
        for basename in stream:
            _, c2w = self._load_cam(scene_dir, basename)
            w2cs.append(np.linalg.inv(c2w.astype(np.float64)))
        return np.stack(w2cs, axis=0)

    def _sample_frame_indices(self, scene: dict) -> tuple[list[str], int] | None:
        num_context = resolve_num_context_frames(
            self.num_context_frames, self.mixed_train
        )
        stream = scene["stream"]
        num_context = clamp_mixed_context_frames(
            num_context,
            len(stream),
            self.num_target_frames,
            mixed_train=self.mixed_train,
        )
        segment_len = num_context + self.num_target_frames
        sampled = sample_kmeans_context_complement_target_positions(
            len(stream),
            self._stream_w2cs(scene["scene_dir"], stream),
            num_context,
            self.num_target_frames,
            segment_len=segment_len,
            target_within_segment=True,
        )
        if sampled is None:
            return None

        context_positions, target_positions = sampled
        context_views = [stream[pos] for pos in context_positions]
        target_views = [stream[pos] for pos in target_positions]
        return context_views + target_views, num_context

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

    def _load_frame(self, scene_dir: str, basename: str, process: ImageCropAndResize):
        path = os.path.join(scene_dir, f"{basename}.jpg")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing RGB for {basename} under {scene_dir}")
        return process(Image.open(path).convert("RGB"))

    def _load_cam(self, scene_dir: str, basename: str) -> tuple[np.ndarray, np.ndarray]:
        path = os.path.join(scene_dir, f"{basename}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing camera for {basename} under {scene_dir}")
        with np.load(path) as data:
            intrinsics = data["intrinsics"].astype(np.float32)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = data["R_cam2world"].astype(np.float32)
            c2w[:3, 3] = data["t_cam2world"].astype(np.float32)
        return intrinsics, c2w

    def __getitem__(self, idx):
        scene = self.scenes[idx % len(self.scenes)]
        sampled = self._sample_frame_indices(scene)
        if sampled is None:
            raise ResampleSample("sample retry")
        view_names, num_context = sampled
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
        trajectories = []
        for i, basename in enumerate(view_names):
            image_path = os.path.join(scene["scene_dir"], f"{basename}.jpg")
            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    ow, oh = img.size
                    all_frames.append(process(img))
                    if i < num_context:
                        lagernvs_context_images.append(img.copy())
                    else:
                        lagernvs_target_images.append(img.copy())
                intrinsics, pose = self._load_cam(scene["scene_dir"], basename)
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
            "scene_name": scene["scene_name"],
            "prompt": "",
            "num_output_frames": num_output,
        }
        native_k = np.stack(native_intrinsics_list, axis=0)
        sample["lagernvs_context_images"] = lagernvs_context_images
        sample["lagernvs_context_intrinsics"] = native_k[:num_context]
        sample["lagernvs_target_intrinsics"] = native_k[num_context:]
        sample["lagernvs_target_images"] = lagernvs_target_images
        return sample
