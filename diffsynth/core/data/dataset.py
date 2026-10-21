"""Shared camera and image utilities for RoGe training datasets."""

import logging
import os
import random

# Standard logging: datasets are built before Accelerator() in train.py.
logger = logging.getLogger("trainer")

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch import Tensor

from diffsynth.camera_utils import normalize_w2c_make_cam_first_origin_lagernvs


def normalize_num_dataset_samples(num_dataset_samples: int | None) -> int | None:
    """Return None for unlimited; positive int caps discovery/selection."""
    if num_dataset_samples is None or num_dataset_samples < 0:
        return None
    if num_dataset_samples == 0:
        raise ValueError(
            f"num_dataset_samples must be > 0 or -1 (no limit), got {num_dataset_samples}"
        )
    return num_dataset_samples


def select_dataset_entries(
    entries,
    num_dataset_samples: int | None = None,
    random_subset: bool = False,
    seed: int = 42,
):
    """Return a subset of an already-enumerated entry list."""
    num_dataset_samples = normalize_num_dataset_samples(num_dataset_samples)
    if num_dataset_samples is None:
        return entries
    if len(entries) <= num_dataset_samples:
        return entries
    if random_subset:
        rng = random.Random(seed)
        return rng.sample(entries, num_dataset_samples)
    return entries[:num_dataset_samples]


def resolve_discovery_limit(
    num_dataset_samples: int | None,
    random_subset: bool,
) -> int | None:
    """When ``random_subset`` is set (--dataset_check), stop discovery early."""
    num_dataset_samples = normalize_num_dataset_samples(num_dataset_samples)
    if random_subset and num_dataset_samples is not None:
        return num_dataset_samples
    return None


def iter_directory_names(
    path: str,
    *,
    max_discover: int | None = None,
    seed: int = 42,
):
    """Iterate directory entries; shuffle when doing early-stop discovery."""
    names = os.listdir(path)
    if max_discover is not None:
        random.Random(seed).shuffle(names)
    else:
        names.sort()
    return names


def limit_dataset_entries(
    entries,
    num_dataset_samples: int | None,
    random_subset: bool,
    seed: int = 42,
):
    """Apply sample limits after discovery (no-op when already early-stopped)."""
    if random_subset:
        return entries
    return select_dataset_entries(
        entries, num_dataset_samples, random_subset=False, seed=seed
    )


def _camera_to_raymap(Ks: Tensor, camtoworlds: Tensor, height: int, width: int):
    dtype = Ks.dtype
    x, y = torch.meshgrid(
        torch.arange(width, device=Ks.device),
        torch.arange(height, device=Ks.device),
        indexing="xy",
    )
    coords = torch.stack([x + 0.5, y + 0.5, torch.ones_like(x)], dim=-1).to(dtype)
    dirs = torch.einsum("...ij,...hwj->...hwi", Ks.float().inverse().to(dtype), coords)
    dirs = torch.einsum("...ij,...hwj->...hwi", camtoworlds[..., :3, :3], dirs)
    dirs = F.normalize(dirs, p=2, dim=-1)
    origins = torch.broadcast_to(camtoworlds[..., None, None, :3, -1], dirs.shape)
    return torch.cat([origins, dirs], dim=-1)


def _raymap_to_plucker(raymap: Tensor) -> Tensor:
    ray_origins, ray_directions = torch.split(raymap, [3, 3], dim=-1)
    ray_directions = F.normalize(ray_directions, p=2, dim=-1)
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
    return torch.cat([ray_directions, plucker_normal], dim=-1)


def get_plucker_rays(
    pose,
    intrinsic,
    height,
    width,
    no_pixel_unshuffle=False,
    downsample_factor=8,
    full_resolution=False,
):
    """Compute Plucker features for ``pose`` (c2w).

    By default returns PixelUnshuffle(downsample_factor) features
    ``(F, 384, H/8, W/8)``. With ``full_resolution=True``, keep ``(F, 6, H, W)``
    for hybrid-VAE raymap mappers in the pipeline.
    """
    raymap = _camera_to_raymap(intrinsic, pose, height=height, width=width)
    plucker = _raymap_to_plucker(raymap).permute(0, 3, 1, 2)
    if full_resolution:
        return plucker
    if not no_pixel_unshuffle:
        plucker = torch.nn.PixelUnshuffle(downscale_factor=downsample_factor)(plucker)
    else:
        plucker = F.interpolate(plucker, scale_factor=1 / 8, mode="bilinear", align_corners=False)
    return plucker


@torch.no_grad()
def normalize_w2c_make_cam_last_origin(w2c: torch.Tensor):
    """Rotate / translate / uniformly scale ``w2c`` so the LAST camera is at
    the world origin and the mean camera-center distance is 1.

    Returns ``(w2c_norm, c2w_norm, scale)``.
    """
    assert w2c.ndim == 3 and w2c.shape[-2:] == (4, 4)
    device, dtype = w2c.device, w2c.dtype

    c2w = torch.linalg.inv(w2c)
    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3]

    R0 = R[-1]
    t0 = t[-1]

    R_align = R0.transpose(0, 1)
    t_shift = t - t0
    t_rot = (R_align @ t_shift.unsqueeze(-1)).squeeze(-1)
    R_rot = R_align @ R

    dists = t_rot.norm(dim=-1)
    scale = dists.mean().clamp_min(1e-12)
    t_norm = t_rot / scale

    c2w_norm = torch.zeros_like(c2w)
    c2w_norm[:, :3, :3] = R_rot
    c2w_norm[:, :3, 3] = t_norm
    c2w_norm[:, 3, :] = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)

    w2c_norm = torch.linalg.inv(c2w_norm)
    return w2c_norm, c2w_norm, scale


@torch.no_grad()
def normalize_w2c_make_cam_first_origin(w2c: torch.Tensor):
    """Align w2c poses so the FIRST camera is at the origin (metric scale)."""
    assert w2c.ndim == 3 and w2c.shape[-2:] == (4, 4)
    device, dtype = w2c.device, w2c.dtype

    c2w = torch.linalg.inv(w2c)
    c2w_ref_inv = torch.linalg.inv(c2w[0])
    c2w_norm = c2w_ref_inv @ c2w
    w2c_norm = torch.linalg.inv(c2w_norm)
    scale = torch.tensor(1.0, device=device, dtype=dtype)
    return w2c_norm, c2w_norm, scale


@torch.no_grad()
def normalize_w2c_make_cam_first_origin_unit(
    w2c: torch.Tensor,
    num_context: int | None = None,
):
    """Align w2c poses so the FIRST camera is at the origin, then uniformly
    scale so the mean camera-center distance among context cameras is 1.

    Args:
        w2c: (N, 4, 4) world-to-camera extrinsics.
        num_context: Number of leading context cameras used to estimate scale.
            If None, all cameras are used.

    Returns:
        w2c_norm, c2w_norm, scale
    """
    assert w2c.ndim == 3 and w2c.shape[-2:] == (4, 4)
    device, dtype = w2c.device, w2c.dtype

    c2w = torch.linalg.inv(w2c)
    c2w_ref_inv = torch.linalg.inv(c2w[0])
    c2w_aligned = c2w_ref_inv @ c2w

    t = c2w_aligned[:, :3, 3]
    t_for_scale = t[:num_context] if num_context is not None else t
    scale = t_for_scale.norm(dim=-1).mean().clamp_min(1e-12)

    c2w_norm = c2w_aligned.clone()
    c2w_norm[:, :3, 3] = t / scale
    c2w_norm[:, 3, :] = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)

    w2c_norm = torch.linalg.inv(c2w_norm)
    return w2c_norm, c2w_norm, scale


def apply_pose_gauge(
    w2c: torch.Tensor,
    pose_gauge: str,
    num_context: int | None = None,
):
    """Apply pose gauge normalization for Plucker raymap construction."""
    if pose_gauge == "cam_first_metric":
        return normalize_w2c_make_cam_first_origin(w2c)
    if pose_gauge == "cam_first_unit":
        return normalize_w2c_make_cam_first_origin_unit(w2c, num_context=num_context)
    if pose_gauge == "cam_first_lagernvs":
        return normalize_w2c_make_cam_first_origin_lagernvs(w2c, num_context=num_context)
    if pose_gauge == "cam_last_unit":
        return normalize_w2c_make_cam_last_origin(w2c)
    raise ValueError(f"Unknown pose_gauge: {pose_gauge!r}")


class ImageCropAndResize:
    """Resize-to-cover + center-crop a PIL image to (height, width)."""

    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def _crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def _get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def __call__(self, data: Image.Image):
        return self._crop_and_resize(data, *self._get_height_width(data))
