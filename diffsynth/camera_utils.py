"""
Camera utilities for Wan2.1 Novel View Synthesis.

All poses are expected in **w2c (world-to-camera) OpenCV convention**:
  - x-right, y-down, z-forward
  - This is the native COLMAP / OpenCV coordinate system.

The pipeline internally (default):
  1. Normalises w2c poses so the *last* camera is at the origin with unit mean scale.
  2. Converts to c2w for raymap construction.
  3. Computes Plucker ray coordinates and pixel-unshuffles to 1/8 resolution.

With ``pose_gauge="cam_first_metric"``:
  1. Aligns w2c poses so the *first* context camera is at the origin.
  2. Preserves metric scale (no uniform scene scaling).

With ``pose_gauge="cam_first_unit"``:
  1. Aligns w2c poses so the *first* context camera is at the origin.
  2. Uniformly scales so the mean context camera-center distance is 1.

With ``pose_gauge="cam_first_lagernvs"``:
  1. Aligns w2c poses so the *first* context camera is at the origin.
  2. Scales by ``1.35 * max`` context camera-center distance (LagerNVS).
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Core raymap functions (from DiffSynth-Studio dataset code)
# ──────────────────────────────────────────────────────────────────────────────

def camera_to_raymap(
    Ks: Tensor,
    camtoworlds: Tensor,
    height: int,
    width: int,
    downscale: float = 1.0,
    include_ups: bool = False,
):
    """Construct the raymap from camera intrinsics and c2w extrinsics.

    Expects OpenCV camera coordinates.

    Args:
        Ks: Camera intrinsics, shape (..., 3, 3).
        camtoworlds: Camera-to-world extrinsics, shape (..., 4, 4).
        height: Image height corresponding to intrinsics.
        width: Image width corresponding to intrinsics.
        downscale: Downscale factor for the raymap.
        include_ups: Whether to include the up direction.

    Returns:
        Raymap tensor, shape (..., H, W, 6) or (..., H, W, 9) if include_ups.
    """
    assert Ks.shape[-2:] == (3, 3)
    assert camtoworlds.shape[-2:] == (4, 4)

    dtype = Ks.dtype
    Ks = torch.stack(
        [Ks[..., 0, :] * downscale, Ks[..., 1, :] * downscale, Ks[..., 2, :]],
        dim=-2,
    )
    width = int(width * downscale)
    height = int(height * downscale)

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

    if include_ups:
        ups = torch.broadcast_to(camtoworlds[..., None, None, :3, 1], dirs.shape)
        ups = F.normalize(ups, p=2, dim=-1)
        return torch.cat([origins, dirs, ups], dim=-1)
    else:
        return torch.cat([origins, dirs], dim=-1)


def raymap_to_plucker(raymap: Tensor) -> Tensor:
    """Convert raymap to Plucker coordinates.

    Args:
        raymap: Shape (..., H, W, 6).

    Returns:
        Plucker coordinates, shape (..., H, W, 6).
    """
    assert raymap.shape[-1] == 6
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
    """Compute Plucker ray maps from c2w poses and intrinsics.

    Args:
        pose: (N, 4, 4) camera-to-world poses.
        intrinsic: (N, 3, 3) intrinsic matrices.
        height: Image height.
        width: Image width.
        no_pixel_unshuffle: If True, use bilinear downsampling instead of PixelUnshuffle.
        downsample_factor: Spatial downsample factor (default 8, matching VAE).
        full_resolution: Keep ``(N, 6, H, W)`` Plucker rays for hybrid-VAE
            packed raymap conditioning.

    Returns:
        Plucker ray features, shape (N, C, H/factor, W/factor), or
        (N, 6, H, W) when ``full_resolution=True``.
    """
    downscale_factor = 1
    raymap = camera_to_raymap(intrinsic, pose, height=height, width=width, downscale=downscale_factor)
    plucker_ray_map = raymap_to_plucker(raymap)
    plucker_ray_map_permuted = plucker_ray_map.permute(0, 3, 1, 2)
    if full_resolution:
        return plucker_ray_map_permuted
    if not no_pixel_unshuffle:
        pixel_unshuffle = torch.nn.PixelUnshuffle(downscale_factor=downsample_factor)
        plucker_ray_map_permuted_unshuffled = pixel_unshuffle(plucker_ray_map_permuted)
    else:
        plucker_ray_map_permuted_unshuffled = F.interpolate(
            plucker_ray_map_permuted, scale_factor=1 / 8, mode='bilinear', align_corners=False
        )
    return plucker_ray_map_permuted_unshuffled


# ──────────────────────────────────────────────────────────────────────────────
# Pose normalisation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def normalize_w2c_make_cam_last_origin(w2c: torch.Tensor):
    """Normalise w2c poses so the *last* camera sits at the world origin.

    Steps:
      1. Convert w2c -> c2w.
      2. Align so that the last camera has identity rotation and zero translation.
      3. Scale so that the mean camera-center distance from the origin is 1.
      4. Convert back to w2c.

    Args:
        w2c: (N, 4, 4) world-to-camera extrinsics (OpenCV convention).

    Returns:
        w2c_norm: (N, 4, 4) normalised w2c.
        c2w_norm: (N, 4, 4) normalised c2w (inverse of w2c_norm).
        scale: Scalar used for the uniform scaling.
    """
    assert w2c.ndim == 3 and w2c.shape[-2:] == (4, 4)

    device, dtype = w2c.device, w2c.dtype
    N = w2c.shape[0]

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
    """Align w2c poses so the *first* camera sits at the world origin.

    Unlike ``normalize_w2c_make_cam_last_origin``, no uniform scale is applied;
    translations keep their metric magnitude in scene units.

    Args:
        w2c: (N, 4, 4) world-to-camera extrinsics (OpenCV convention).

    Returns:
        w2c_norm: (N, 4, 4) aligned w2c.
        c2w_norm: (N, 4, 4) aligned c2w.
        scale: Scalar tensor fixed to 1.0 (API compatibility).
    """
    assert w2c.ndim == 3 and w2c.shape[-2:] == (4, 4)

    device, dtype = w2c.device, w2c.dtype
    c2w = torch.linalg.inv(w2c)
    c2w_ref_inv = torch.linalg.inv(c2w[0])
    c2w_norm = c2w_ref_inv @ c2w
    w2c_norm = torch.linalg.inv(c2w_norm)
    scale = torch.tensor(1.0, device=device, dtype=dtype)
    return w2c_norm, c2w_norm, scale


CAMERA_SCALE_MULTIPLIER = 1.35


@torch.no_grad()
def normalize_w2c_make_cam_first_origin_lagernvs(
    w2c: torch.Tensor,
    num_context: int | None = None,
):
    """First camera at origin; scale by 1.35 * max context translation norm (LagerNVS)."""
    assert w2c.ndim == 3 and w2c.shape[-2:] == (4, 4)
    device, dtype = w2c.device, w2c.dtype

    c2w = torch.linalg.inv(w2c)
    c2w_ref_inv = torch.linalg.inv(c2w[0])
    c2w_aligned = c2w_ref_inv @ c2w

    t = c2w_aligned[:, :3, 3]
    t_for_scale = t[:num_context] if num_context is not None else t
    if num_context == 1:
        scene_scale = torch.tensor(1.0, device=device, dtype=dtype)
    else:
        scene_scale = (
            CAMERA_SCALE_MULTIPLIER * t_for_scale.norm(dim=-1).max()
        ).clamp_min(1e-12)

    c2w_norm = c2w_aligned.clone()
    c2w_norm[:, :3, 3] = t / scene_scale
    c2w_norm[:, 3, :] = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)

    w2c_norm = torch.linalg.inv(c2w_norm)
    return w2c_norm, c2w_norm, scene_scale


@torch.no_grad()
def normalize_w2c_make_cam_first_origin_unit(
    w2c: torch.Tensor,
    num_context: int | None = None,
):
    """Align w2c poses so the FIRST camera is at the origin, then uniformly
    scale so the mean camera-center distance among context cameras is 1."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def prepare_raymap(
    w2c_poses_np,
    intrinsics_np,
    height,
    width,
    pose_gauge: str = "cam_last_unit",
    num_context: int | None = None,
    cam_first_metric: bool = False,
    full_resolution: bool = False,
):
    """Compute plucker ray features from w2c poses (OpenCV convention).

    This is the main entry point used by ``model.RoGe``.

    Args:
        w2c_poses_np: (N, 4, 4) numpy array -- w2c poses in OpenCV convention.
        intrinsics_np: (N, 3, 3) numpy array -- intrinsics at model resolution.
        height: Model input height.
        width: Model input width.
        pose_gauge: One of ``cam_last_unit``, ``cam_first_metric``,
            ``cam_first_unit``, ``cam_first_lagernvs``.
        num_context: Number of context cameras for unit / LagerNVS scale.
        cam_first_metric: Deprecated alias for ``pose_gauge="cam_first_metric"``.
        full_resolution: Keep ``(N, 6, H, W)`` Plucker rays for hybrid VAE.

    Returns:
        raymap: (N, C, H/8, W/8) tensor, or (N, 6, H, W) when
            ``full_resolution=True`` (float32 on CPU).
    """
    if cam_first_metric and pose_gauge == "cam_last_unit":
        pose_gauge = "cam_first_metric"

    w2cs = torch.from_numpy(w2c_poses_np).float()

    if pose_gauge == "cam_first_metric":
        _, c2w_norm, _ = normalize_w2c_make_cam_first_origin(w2cs)
    elif pose_gauge == "cam_first_unit":
        _, c2w_norm, _ = normalize_w2c_make_cam_first_origin_unit(
            w2cs, num_context=num_context
        )
    elif pose_gauge == "cam_first_lagernvs":
        _, c2w_norm, _ = normalize_w2c_make_cam_first_origin_lagernvs(
            w2cs, num_context=num_context
        )
    elif pose_gauge == "cam_last_unit":
        _, c2w_norm, _ = normalize_w2c_make_cam_last_origin(w2cs)
    else:
        raise ValueError(f"Unknown pose_gauge: {pose_gauge!r}")

    raymap = get_plucker_rays(
        c2w_norm,
        torch.from_numpy(intrinsics_np).float(),
        height=height,
        width=width,
        full_resolution=full_resolution,
    )
    return raymap
