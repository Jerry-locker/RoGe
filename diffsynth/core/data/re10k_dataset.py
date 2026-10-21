"""RoGe training dataset for LagerNVS-layout RealEstate10K (RE10K).

Expected layout (``process_re10k_mutou.py`` / LagerNVS-style)::

    <base_path>/
        train/
            images/<seq_id>/00000.png ...
            metadata/<seq_id>.json
            full_list.txt                  # optional
        test/                              # optional; excluded unless listed in splits

Each ``metadata/<seq_id>.json``::

    {
      "id": "<seq_id>",
      "width": W,
      "height": H,
      "frames": [
        {"timestamp": int, "fxfycxcy": [fx, fy, cx, cy], "w2c": 4x4},
        ...
      ]
    }

``fxfycxcy`` are pixel intrinsics; ``w2c`` is OpenCV world-to-camera.
Internally poses are converted to c2w so the rest of the pipeline matches
DL3DV (``inv(c2w) → w2c`` for raymaps / returned extrinsics).

Sampling mirrors ``DL3DVRoGeDataset`` (50/50 context-first /
target-first; optional ``mixed_train`` randomizes context count in ``[3, 9]``,
gates scenes by min context, and clamps M to stream capacity at sample time).
"""

from __future__ import annotations

import json
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

RE10K_DEFAULT_TRAIN_SPLITS = ("train",)
RE10K_CONTEXT_SEGMENT_MIN_RATIO = 9 / 10
RE10K_CONTEXT_KMEANS_PROBABILITY = 0.7


def _fxfycxcy_to_K(fxfycxcy) -> np.ndarray:
    fx, fy, cx, cy = [float(v) for v in fxfycxcy]
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    w2c = np.asarray(w2c, dtype=np.float64)
    if w2c.shape != (4, 4):
        raise ValueError(f"w2c must be 4x4, got {w2c.shape}")
    return np.linalg.inv(w2c).astype(np.float32)


def _resolve_split_root(base_path: str, split: str) -> str | None:
    """Return directory that contains ``images/`` + ``metadata/`` for a split."""
    candidates = [
        os.path.join(base_path, split),
        base_path if split == "train" else "",
    ]
    for root in candidates:
        if not root:
            continue
        if os.path.isdir(os.path.join(root, "images")) and os.path.isdir(
            os.path.join(root, "metadata")
        ):
            return root
    return None


def _discover_scenes(
    base_path: str,
    splits: tuple[str, ...],
    *,
    excluded_scene_ids: set[str] | None = None,
    max_discover: int | None = None,
    seed: int = 42,
) -> list[dict]:
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"RE10K base_path not found: {base_path}")

    excluded = excluded_scene_ids or set()
    scenes: list[dict] = []
    for split in splits:
        split_root = _resolve_split_root(base_path, split)
        if split_root is None:
            logger.warning(
                "RE10K split not found or missing images/metadata, skipping: "
                f"{os.path.join(base_path, split)}"
            )
            continue

        images_root = os.path.join(split_root, "images")
        meta_root = os.path.join(split_root, "metadata")
        remaining = None if max_discover is None else max_discover - len(scenes)
        if remaining is not None and remaining <= 0:
            break

        for seq_id in iter_directory_names(
            images_root, max_discover=remaining, seed=seed
        ):
            if seq_id in excluded:
                continue
            rgb_dir = os.path.join(images_root, seq_id)
            meta_path = os.path.join(meta_root, f"{seq_id}.json")
            if not (os.path.isdir(rgb_dir) and os.path.isfile(meta_path)):
                continue
            scenes.append(
                {
                    "split": split,
                    "scene_id": f"{split}/{seq_id}" if split != "." else seq_id,
                    "seq_id": seq_id,
                    "rgb_dir": rgb_dir,
                    "meta_path": meta_path,
                }
            )
            if max_discover is not None and len(scenes) >= max_discover:
                return scenes
    return scenes


class RE10KRoGeDataset(Dataset):
    """Sparse context + consecutive target dataset for RE10K fine-tuning."""

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
        mixed_train: bool = False,
        dataset_cache_dir: str | None = None,
        update_dataset_cache: bool = False,
        splits: tuple[str, ...] | list[str] | None = None,
        excluded_scene_ids: tuple[str, ...] | list[str] | None = None,
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
        self.mixed_train = mixed_train
        self.splits = tuple(splits) if splits is not None else RE10K_DEFAULT_TRAIN_SPLITS
        self.excluded_scene_ids = set(excluded_scene_ids or ())

        if num_context_frames <= 0:
            raise ValueError(f"num_context_frames must be > 0, got {num_context_frames}")
        if num_target_frames <= 0:
            raise ValueError(f"num_target_frames must be > 0, got {num_target_frames}")

        max_discover = resolve_discovery_limit(num_dataset_samples, random_subset)

        def build_scenes():
            scene_entries = _discover_scenes(
                base_path,
                self.splits,
                excluded_scene_ids=self.excluded_scene_ids,
                max_discover=max_discover,
            )
            scene_entries = limit_dataset_entries(
                scene_entries, num_dataset_samples, random_subset
            )
            if not scene_entries:
                raise FileNotFoundError(
                    f"No RE10K scenes found under {base_path} "
                    f"(expected <split>/images/<seq> + metadata/<seq>.json "
                    f"for splits {self.splits})"
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
                    f"No usable RE10K scenes "
                    f"(need enough frames for context={filter_ctx}, "
                    f"target={num_target_frames}) "
                    f"under {base_path}"
                )
            return scenes

        self.scenes, from_cache = build_scene_list_with_cache(
            dataset_name="re10k_mmin",
            base_path=base_path,
            cache_dir=dataset_cache_dir,
            build_fn=build_scenes,
            update_cache=update_dataset_cache,
        )

        cache_note = ", from cache" if from_cache else ""
        if self.mixed_train:
            layout_note = (
                f"mixed_train context=[3,9], target={num_target_frames}"
            )
        else:
            layout_note = (
                f"context={num_context_frames}, target={num_target_frames}"
            )
        print(
            f"RE10K dataset (splits={','.join(self.splits)}): {len(self.scenes)} scenes"
            f"{cache_note}, "
            f"{layout_note}, "
            f"sampling=50/50 "
            f"([{RE10K_CONTEXT_SEGMENT_MIN_RATIO:g},1]-context-first / target-first; "
            f"context={RE10K_CONTEXT_KMEANS_PROBABILITY:g} k-means / "
            f"{1 - RE10K_CONTEXT_KMEANS_PROBABILITY:g} uniform), "
            f"pose_gauge={pose_gauge}"
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
            min_ratio=RE10K_CONTEXT_SEGMENT_MIN_RATIO,
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

    def _load_scene_info(self, entry: dict):
        rgb_dir = entry["rgb_dir"]
        meta_path = entry["meta_path"]
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Skipping {entry['scene_id']}: bad metadata ({exc})")
            return None

        frames = meta.get("frames")
        if not isinstance(frames, list) or not frames:
            logger.warning(f"Skipping {entry['scene_id']}: empty frames")
            return None

        images = sorted(
            name for name in os.listdir(rgb_dir) if name.lower().endswith(".png")
        )
        if len(images) != len(frames):
            n = min(len(images), len(frames))
            images = images[:n]
            frames = frames[:n]
            if n == 0:
                return None

        valid_images = []
        valid_K = []
        valid_c2w = []
        for image_name, frame in zip(images, frames):
            rgb_path = os.path.join(rgb_dir, image_name)
            if not os.path.isfile(rgb_path):
                continue
            try:
                K = _fxfycxcy_to_K(frame["fxfycxcy"])
                c2w = _w2c_to_c2w(np.asarray(frame["w2c"], dtype=np.float64))
            except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
                logger.warning(
                    f"Skipping frame {entry['scene_id']}/{image_name}: {exc}"
                )
                continue
            valid_images.append(image_name)
            valid_K.append(K)
            valid_c2w.append(c2w)

        stream = list(range(len(valid_images)))
        if not self._stream_is_usable(len(stream)):
            logger.warning(
                f"Skipping {entry['scene_id']}: stream too short for sampling "
                f"(n={len(stream)})"
            )
            return None

        return {
            **entry,
            "images": valid_images,
            "stream": stream,
            "intrinsics": np.stack(valid_K, axis=0).astype(np.float32),
            "c2ws": np.stack(valid_c2w, axis=0).astype(np.float32),
        }

    def _stream_w2cs(self, scene: dict, stream: list[int]) -> np.ndarray:
        c2ws = scene["c2ws"][stream]
        w2cs = np.linalg.inv(c2ws.astype(np.float64))
        return w2cs

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
            min_ratio=RE10K_CONTEXT_SEGMENT_MIN_RATIO,
            kmeans_probability=RE10K_CONTEXT_KMEANS_PROBABILITY,
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
            kmeans_probability=RE10K_CONTEXT_KMEANS_PROBABILITY,
        )

    def _sample_frame_positions(
        self,
        scene: dict,
        stream: list[int],
        num_context: int,
        num_target: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
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
        for sampler in samplers:
            sampled = sampler(stream_w2cs, num_context, num_target)
            if sampled is not None:
                return sampled
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
            rgb_path = os.path.join(scene["rgb_dir"], image_name)
            try:
                with Image.open(rgb_path) as img:
                    img = img.convert("RGB")
                    ow, oh = img.size
                    all_frames.append(process(img))
                    if i < num_context:
                        lagernvs_context_images.append(img.copy())
                    else:
                        lagernvs_target_crops.append(
                            lagernvs_infer_cover_crop(
                                img, self.height, self.width
                            )
                        )
            except (FileNotFoundError, OSError):
                raise ResampleSample("sample retry")

            intrinsics = scene["intrinsics"][frame_i]
            pose = scene["c2ws"][frame_i]
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
        sample["lagernvs_target_images"] = torch.from_numpy(
            np.stack(lagernvs_target_crops, axis=0)
        )
        return sample
