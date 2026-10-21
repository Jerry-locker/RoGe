"""LagerNVS ray-token geom conditioning for RoGe.

Run all context+target Plucker rays through the LagerNVS decoder (12-layer
cross-attn with ``rec_tokens``; stock LagerNVS only feeds target rays —
context decode is an intentional extension), project **ray** tokens to latent
resolution via a 1x1 + bilinear projector, then VAE-style temporal **pack**
(same expand + channel-pack as hybrid raymaps; no ``TargetFeatureTemporalMapper``),
and inject as ``vggt_geom``. Concat channels are ``geom_channels * pack_group``.
Views are decoded in one forward for up to 7 context frames. At 8 or more,
renderer views are chunked to bound the checkpoint-recompute memory peak.
Training stops after the 12 attn layers; vis/aux RGB adds FinalLayer readout
on the same forward (``cache_aux_rgb=True``).

By default the backbone is frozen and only the geom projector trains.
``--joint_train`` injects LoRA into the LagerNVS **renderer**, opens the token
graph for geom conditioning, and adds an MSE NVS auxiliary loss on target-view
RGB. The reconstructor stays frozen unless ``--train_reconstructor_lora``.

Inference preprocess normally matches ``run_dl3dv.py --test2``: RoGe
cover-crop from native-resolution context images to 288x504 (same FOV as RoGe
192x336). For RoGe 480x832, it first applies the RoGe cover-crop and then directly
resizes that result to 288x504 so both branches see the same crop.

Callers should pass native-resolution context images and matching intrinsics
(``lagernvs_context_*``). RoGe-resolution inputs are accepted but lose the
high-frequency detail of a true native cover-crop.
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Literal, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# RoGe native resolution (H, W).
ROGE_HEIGHT, ROGE_WIDTH = 192, 336
ROGE_480P_HEIGHT, ROGE_480P_WIDTH = 480, 832
# LagerNVS inference resolution: same FOV as RoGe, 1.5x scale (--test2).
LAGERNVS_INFER_HEIGHT, LAGERNVS_INFER_WIDTH = 288, 504
LAGERNVS_PATCH_SIZE = 8
LAGERNVS_TOKEN_DIM = 768
LAGERNVS_CHANNELS = 384
LAGERNVS_PACK_GROUP = 4
# Default LoRA targets: Attention Linear projections in the renderer.
LAGERNVS_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "proj")
# Reconstructor (VGGT) Attention uses fused qkv + out proj; regex avoids
# matching patch_embed.proj (Conv2d).
LAGERNVS_RECONSTRUCTOR_LORA_TARGET_MODULES = r".*\.attn\.(qkv|proj)$"
LAGERNVS_LORA_RANK = 8
# Larger context sets make every renderer view cross-attend substantially more
# reconstruction tokens. Chunk only those expensive steps to preserve the
# existing fast path for the common M<=7 case.
LAGERNVS_RENDERER_CHUNK_CONTEXT_THRESHOLD = 7
LAGERNVS_RENDERER_VIEW_CHUNK_SIZE = 32

# Top-level packages inside the LagerNVS checkout. Inserting that repo on
# ``sys.path`` would otherwise permanently shadow RoGe / site-packages
# modules with the same names (notably ``models``, ``data``, ``vggt``, ``utils``).
_LAGERNVS_TOPLEVEL_PACKAGES = ("models", "data", "vggt", "utils")
_LAGERNVS_PRIVATE_PREFIX = "_roge_lagernvs"
_LAGERNVS_API_CACHE: dict[str, SimpleNamespace] = {}


def bundled_lagernvs_repo_path() -> str:
    """Vendored LagerNVS checkout at ``<repo>/third_party/lagernvs``."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "third_party", "lagernvs")


def lagernvs_packed_channels(
    geom_channels: int = LAGERNVS_CHANNELS,
    pack_group: int = LAGERNVS_PACK_GROUP,
) -> int:
    """DiT concat channels after VAE-style temporal pack of projected features.

    ``--lagernvs_channels`` is the pre-pack projector width; pack expands by
    ``pack_group`` (default 4), matching hybrid raymap temporal packing.
    """
    return int(geom_channels) * int(pack_group)


def lagernvs_in_dim(
    geom_channels: int = LAGERNVS_CHANNELS,
    raymap_pe: bool = False,
    geom_pe: bool = False,
) -> int:
    """DiT in_dim with packed LagerNVS geom (concat unless ``geom_pe``)."""
    from diffsynth.vggt_condition import base_in_dim

    dim = base_in_dim(raymap_pe=raymap_pe)
    if not geom_pe:
        dim += lagernvs_packed_channels(geom_channels)
    return dim


def _is_lagernvs_toplevel_module(name: str) -> bool:
    return any(
        name == root or name.startswith(root + ".")
        for root in _LAGERNVS_TOPLEVEL_PACKAGES
    )


def _import_lagernvs_api(repo_path: str) -> SimpleNamespace:
    """Import LagerNVS symbols without permanently shadowing public package names.

    Temporarily puts ``repo_path`` on ``sys.path``, imports the needed API, then:
    - restores any pre-existing ``models`` / ``data`` / ``vggt`` / ``utils`` modules
    - re-registers LagerNVS modules under ``_roge_lagernvs.*``
    - removes ``repo_path`` from ``sys.path``

    Callers must use the returned callables/classes (cached), not re-import
    ``models.*`` / ``data.*`` under the public names.
    """
    repo = os.path.abspath(repo_path)
    cached = _LAGERNVS_API_CACHE.get(repo)
    if cached is not None:
        return cached

    if not os.path.isdir(repo):
        raise FileNotFoundError(f"LagerNVS repo path not found: {repo}")

    preexisting = {
        name: module
        for name, module in list(sys.modules.items())
        if _is_lagernvs_toplevel_module(name)
    }
    for name in preexisting:
        del sys.modules[name]

    inserted = False
    if repo not in sys.path:
        sys.path.insert(0, repo)
        inserted = True
    try:
        from models.encoder_decoder import EncDec_VitB8
        from data.normalization import build_cam_cond, normalize_extrinsics
        from vggt.utils.pose_enc import extri_intri_to_pose_encoding
    except Exception:
        # Roll back any partial LagerNVS imports, then restore prior modules.
        for name in list(sys.modules):
            if _is_lagernvs_toplevel_module(name) and name not in preexisting:
                del sys.modules[name]
        sys.modules.update(preexisting)
        raise
    finally:
        if inserted:
            try:
                sys.path.remove(repo)
            except ValueError:
                pass

    lagernvs_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if _is_lagernvs_toplevel_module(name)
    }
    for name in lagernvs_modules:
        del sys.modules[name]
    sys.modules.update(preexisting)
    for name, module in lagernvs_modules.items():
        sys.modules[f"{_LAGERNVS_PRIVATE_PREFIX}.{name}"] = module

    api = SimpleNamespace(
        EncDec_VitB8=EncDec_VitB8,
        build_cam_cond=build_cam_cond,
        normalize_extrinsics=normalize_extrinsics,
        extri_intri_to_pose_encoding=extri_intri_to_pose_encoding,
    )
    _LAGERNVS_API_CACHE[repo] = api
    return api


def _is_accelerate_deepspeed() -> bool:
    try:
        from accelerate.state import AcceleratorState
        from accelerate.utils import DistributedType

        return AcceleratorState().distributed_type == DistributedType.DEEPSPEED
    except Exception:
        return False


def _cast_patch_grids_for_projector(
    patch_grids: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Match VGGT projector dtype policy (DeepSpeed bf16 vs DDP float32)."""
    if _is_accelerate_deepspeed():
        return patch_grids.to(device=weight.device, dtype=weight.dtype)
    return patch_grids.to(device=weight.device, dtype=torch.float32)


class LagerNVSGeometryProjector(nn.Module):
    """1x1 conv + bilinear resize: LagerNVS tokens -> RoGe latent spatial grid."""

    def __init__(
        self,
        in_dim: int = LAGERNVS_TOKEN_DIM,
        out_channels: int = LAGERNVS_CHANNELS,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.proj = nn.Conv2d(in_dim, out_channels, kernel_size=1)

    def forward(
        self,
        patch_grids: torch.Tensor,
        latent_h: int,
        latent_w: int,
    ) -> torch.Tensor:
        """Args:
            patch_grids: (M, in_dim, ph, pw)
        Returns:
            (M, out_channels, latent_h, latent_w)
        """
        patch_grids = _cast_patch_grids_for_projector(patch_grids, self.proj.weight)
        x = self.proj(patch_grids)
        return F.interpolate(
            x, size=(latent_h, latent_w), mode="bilinear", align_corners=False
        )


def _pil_to_rgb_uint8(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def _compute_cover_crop_params(orig_w: int, orig_h: int, tgt_w: int, tgt_h: int) -> dict:
    """RoGe cover-resize + center-crop geometry (matches ``run_dl3dv.py``)."""
    scale = max(tgt_w / orig_w, tgt_h / orig_h)
    resize_w = int(round(orig_w * scale))
    resize_h = int(round(orig_h * scale))
    crop_x = (resize_w - tgt_w) / 2.0
    crop_y = (resize_h - tgt_h) / 2.0
    return {
        "scale": scale,
        "resize_w": resize_w,
        "resize_h": resize_h,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "offset_x": int(round(crop_x)),
        "offset_y": int(round(crop_y)),
    }


def _scale_intrinsic_cover_crop(
    k: np.ndarray,
    orig_w: int,
    orig_h: int,
    tgt_w: int,
    tgt_h: int,
) -> np.ndarray:
    """Adjust K for RoGe cover-resize + center-crop."""
    params = _compute_cover_crop_params(orig_w, orig_h, tgt_w, tgt_h)
    scale = params["scale"]
    k_out = np.asarray(k, dtype=np.float32).copy()
    k_out[0, 0] = k[0, 0] * scale
    k_out[1, 1] = k[1, 1] * scale
    k_out[0, 2] = k[0, 2] * scale - params["crop_x"]
    k_out[1, 2] = k[1, 2] * scale - params["crop_y"]
    return k_out


def _resize_cover_center_crop(rgb: np.ndarray, tgt_h: int, tgt_w: int) -> np.ndarray:
    """rgb: HxWx3 uint8 -> tgt_h x tgt_w uint8 (RoGe cover-crop)."""
    h, w = rgb.shape[:2]
    if h == tgt_h and w == tgt_w:
        return np.copy(rgb)
    params = _compute_cover_crop_params(w, h, tgt_w, tgt_h)
    resized = cv2.resize(
        rgb, (params["resize_w"], params["resize_h"]), interpolation=cv2.INTER_AREA
    )
    y0, x0 = params["offset_y"], params["offset_x"]
    return resized[y0 : y0 + tgt_h, x0 : x0 + tgt_w]


def lagernvs_infer_cover_crop(
    image: Image.Image,
    roge_height: int = ROGE_HEIGHT,
    roge_width: int = ROGE_WIDTH,
) -> np.ndarray:
    """Cover-crop one PIL to LagerNVS infer res; return ``(H, W, 3)`` uint8.

    Datasets call this in DataLoader workers so NVS-loss GT crosses the
    worker -> training-process boundary as a 288x504 uint8 array instead of a
    full-resolution PIL. Byte-identical to the crop
    ``_gt_to_infer_rgb_tensor`` would have applied in the training step.
    """
    rgb = _pil_to_rgb_uint8(image)
    if (roge_height, roge_width) == (ROGE_480P_HEIGHT, ROGE_480P_WIDTH):
        rgb = _resize_cover_center_crop(rgb, roge_height, roge_width)
        return cv2.resize(
            rgb,
            (LAGERNVS_INFER_WIDTH, LAGERNVS_INFER_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
    return _resize_cover_center_crop(
        rgb, LAGERNVS_INFER_HEIGHT, LAGERNVS_INFER_WIDTH
    )


def _w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    w2c = np.asarray(w2c, dtype=np.float32)
    if w2c.shape == (3, 4):
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :4] = w2c
        w2c = mat
    return np.linalg.inv(w2c).astype(np.float32)


def _fxfycxcy_from_k(k: np.ndarray) -> np.ndarray:
    return np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], dtype=np.float32)


# Pixel-center grid depends only on (H, W), so build it once with the same
# ``torch.linspace`` calls as LagerNVS ``get_uv_hom`` and cache it per device.
# Seeding the GPU path from the CPU grid keeps ``uv`` bit-identical instead of
# relying on CUDA ``linspace`` matching.
_UV_HOM_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def _uv_hom_on_device(height: int, width: int, device: torch.device) -> torch.Tensor:
    """LagerNVS ``get_uv_hom`` output ``(H, W, 3)``, cached on ``device``."""
    key = (int(height), int(width), str(device))
    cached = _UV_HOM_CACHE.get(key)
    if cached is not None:
        return cached
    pixel_centers_x = torch.linspace(0.5, width - 0.5, width)[None, :]
    pixel_centers_y = torch.linspace(0.5, height - 0.5, height)[:, None]
    pixel_centers_x = pixel_centers_x.expand((height, -1))
    pixel_centers_y = pixel_centers_y.expand((-1, width))
    ones = torch.ones_like(pixel_centers_x, dtype=pixel_centers_x.dtype)
    uv_hom = torch.stack([pixel_centers_x, pixel_centers_y, ones], axis=2).float()
    uv_hom = uv_hom.to(device)
    _UV_HOM_CACHE[key] = uv_hom
    return uv_hom


def _plucker_rays_on_device(
    c2w: torch.Tensor,
    fxfycxcy: torch.Tensor,
    tgt_hw: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """LagerNVS ``compute_plucker_rays`` evaluated on ``device``: ``(V, 6, H, W)``.

    Mirrors ``data/camera_utils.compute_plucker_rays`` op for op (including the
    vectorised equivalent of ``get_K_matrices``). Building the rays on the GPU
    keeps a 300 MB fp32 tensor (87 views at 288x504) off the training process
    CPU and off the PCIe bus. The rays only feed ``renderer.tgt_embedder``, a
    Conv2d under bf16 autocast, so fp32 tail-bit differences between CPU LAPACK
    and cuSOLVER ``inv`` are cast away before they reach any weight.
    """
    h, w = int(tgt_hw[0]), int(tgt_hw[1])
    c2w = c2w.to(device=device, dtype=torch.float32)
    fxfycxcy = fxfycxcy.to(device=device, dtype=torch.float32)
    v = c2w.shape[0]

    ks = torch.zeros(v, 3, 3, device=device, dtype=torch.float32)
    ks[:, 0, 0] = fxfycxcy[:, 0]
    ks[:, 1, 1] = fxfycxcy[:, 1]
    ks[:, 0, 2] = fxfycxcy[:, 2]
    ks[:, 1, 2] = fxfycxcy[:, 3]
    ks[:, 2, 2] = 1.0

    uv = _uv_hom_on_device(h, w, device)
    uv = uv[None, ...].expand(v, h, w, 3).view(v, -1, 3)

    k_inv = torch.linalg.inv(ks).float()
    dirs_local = torch.bmm(k_inv, uv.transpose(-2, -1)).transpose(-2, -1)
    # Out-of-place, matching the reference: an in-place divide would hand bmm a
    # differently strided operand and let cuBLAS pick another kernel.
    dirs_local = dirs_local / torch.linalg.norm(dirs_local, dim=-1, keepdim=True)
    dirs_global = torch.bmm(
        c2w[:, :3, :3], dirs_local.transpose(-2, -1)
    ).transpose(-2, -1)
    del dirs_local
    dirs_global = dirs_global.view(v, h, w, 3)
    ray_o = c2w[:, :3, 3][:, None, None, :].expand_as(dirs_global)
    moment = torch.cross(ray_o, dirs_global, dim=-1)
    plucker_rays = torch.cat([moment, dirs_global], dim=-1)
    return plucker_rays.permute(0, 3, 1, 2)


def _build_cam_cond_tokens(
    api: SimpleNamespace,
    c2w_poses: torch.Tensor,
    intrinsics_fxfycxcy_px: torch.Tensor,
    num_cond_views: int,
    tgt_hw: tuple[int, int],
    camera_scale,
    split: str = "test",
) -> torch.Tensor:
    """Token half of LagerNVS ``build_cam_cond``, with the ray half split out.

    Same ops on the same CPU tensors as ``build_cam_cond`` called with
    ``zero_out_cam_cond_p=0.0`` (which makes its zero-out branch unreachable),
    so the tokens stay bit-identical. Only the Plucker rays move to the GPU via
    ``_plucker_rays_on_device``; the tokens are 13 floats per view, so there is
    nothing to gain from moving them.
    """
    cam_cond_token = api.extri_intri_to_pose_encoding(
        c2w_poses.unsqueeze(0),
        intrinsics_fxfycxcy_px.unsqueeze(0),
        image_size_hw=tgt_hw,
    ).squeeze(0)

    if num_cond_views == 1 and split == "test":
        camera_scale = 0.0
        world_points_scale = 1.0
    else:
        world_points_scale = 0.0

    scene_scale_tokens = (
        torch.tensor([camera_scale, world_points_scale])
        .unsqueeze(0)
        .expand(cam_cond_token.shape[0], -1)
    )
    return torch.cat([cam_cond_token, scene_scale_tokens], dim=-1)


class LagerNVSConditioningModule(nn.Module):
    """LagerNVS encoder-decoder + trainable ray-token geom projector.

    By default the backbone is frozen. With ``joint_train=True``, renderer LoRA
    is injected and registered so optimizer / checkpointing can see it.
    Optional ``train_reconstructor_lora`` also adapts the reconstructor.
    ``load_frozen_lora`` injects both adapters for checkpoint loading but keeps
    them frozen and disables the auxiliary training loss.
    """

    def __init__(
        self,
        lagernvs_ckpt: str,
        offload_per_step: bool = False,
        geom_channels: int = LAGERNVS_CHANNELS,
        device: str | torch.device = "cuda",
        joint_train: bool = False,
        train_reconstructor_lora: bool = False,
        load_frozen_lora: bool = False,
        lora_rank: int = LAGERNVS_LORA_RANK,
        lora_alpha: int | None = None,
        lora_target_modules: Sequence[str] = LAGERNVS_LORA_TARGET_MODULES,
        reconstructor_lora_target_modules: str = LAGERNVS_RECONSTRUCTOR_LORA_TARGET_MODULES,
    ):
        super().__init__()
        self.lagernvs_repo_path = os.path.abspath(bundled_lagernvs_repo_path())
        self.lagernvs_ckpt = lagernvs_ckpt
        self.joint_train = bool(joint_train)
        self.train_reconstructor_lora = bool(train_reconstructor_lora)
        self.load_frozen_lora = bool(load_frozen_lora)
        if self.joint_train and self.load_frozen_lora:
            raise ValueError("joint_train and load_frozen_lora are mutually exclusive")
        if self.train_reconstructor_lora and not self.joint_train:
            raise ValueError("train_reconstructor_lora requires joint_train=True")
        # CPU/GPU offload breaks the joint-train autograd graph.
        self.offload_per_step = bool(offload_per_step) and not self.joint_train
        self.geom_channels = int(geom_channels)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha) if lora_alpha is not None else self.lora_rank
        self.lora_target_modules = tuple(lora_target_modules)
        self.reconstructor_lora_target_modules = str(reconstructor_lora_target_modules)
        self._init_device = torch.device(device)
        # Tiny buffer so accelerator.prepare / .to() moves this module to the
        # training GPU even when --initialize_model_on_cpu built it on CPU.
        # compute_device follows this anchor (same idea as VGGT's projector).
        self.register_buffer(
            "_device_anchor", torch.empty(0, device=self._init_device), persistent=False
        )
        # Plain frozen: lazy attribute. LoRA modes: registered ``backbone``.
        self._model = None
        self._api: SimpleNamespace | None = None
        # Cached all-view RGB from the same decoder pass as build_lagernvs_geom.
        self.last_pred_all_views: list[Image.Image] | None = None
        self.last_nvs_loss: torch.Tensor | None = None
        # Temporal alignment uses pack_hybrid_condition_map (no mapper).
        self.projector = LagerNVSGeometryProjector(
            LAGERNVS_TOKEN_DIM, self.geom_channels
        ).to(self._init_device)
        if self.joint_train or self.load_frozen_lora:
            # Eager load so LoRA params exist before accelerator.prepare / optimizer.
            self.ensure_model_loaded()

    @property
    def compute_device(self) -> torch.device:
        return self.projector.proj.weight.device

    def train(self, mode: bool = True):
        super().train(mode)
        if self.load_frozen_lora:
            backbone = self._get_backbone()
            if backbone is not None:
                backbone.eval()
        return self

    def sync_device(self, device: str | torch.device) -> None:
        """Align the device anchor (and loaded weights) with ``pipe.device``."""
        device = torch.device(device)
        if self._device_anchor.device != device:
            self.register_buffer(
                "_device_anchor", torch.empty(0, device=device), persistent=False
            )
        self._init_device = device
        self.projector.to(device)
        model = self._get_backbone()
        if model is not None and next(model.parameters()).device != device:
            model.to(device)

    def _get_api(self) -> SimpleNamespace:
        if self._api is None:
            self._api = _import_lagernvs_api(self.lagernvs_repo_path)
        return self._api

    def _get_backbone(self) -> nn.Module | None:
        """Return the loaded LagerNVS EncDec (registered or lazy)."""
        backbone = getattr(self, "backbone", None)
        if backbone is not None:
            return backbone
        return self._model

    def _onload(self) -> None:
        model = self._get_backbone()
        if model is None:
            return
        device = self.compute_device
        if next(model.parameters()).device != device:
            model.to(device)

    def _offload(self) -> None:
        if not self.offload_per_step:
            return
        model = self._get_backbone()
        if model is None:
            return
        if next(model.parameters()).device.type == "cuda":
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _inject_joint_lora(self, model: nn.Module) -> nn.Module:
        """Inject peft LoRA into renderer (and optionally reconstructor)."""
        from peft import LoraConfig, inject_adapter_in_model

        for param in model.parameters():
            param.requires_grad = False

        renderer_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=list(self.lora_target_modules),
            bias="none",
        )
        model.renderer = inject_adapter_in_model(renderer_config, model.renderer)

        if self.train_reconstructor_lora:
            recon_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=self.reconstructor_lora_target_modules,
                bias="none",
            )
            model.reconstructor = inject_adapter_in_model(
                recon_config, model.reconstructor
            )
            # EncDec defaults to freeze_vggt=True which no_grad+detachs VGGT.
            model.reconstructor.freeze_vggt = False
            model.reconstructor.train()
        else:
            model.reconstructor.eval()

        n_lora = 0
        n_renderer = 0
        n_recon = 0
        for name, param in model.named_parameters():
            is_lora = "lora_" in name
            param.requires_grad = is_lora
            if is_lora:
                n_lora += 1
                if name.startswith("renderer."):
                    n_renderer += 1
                elif name.startswith("reconstructor."):
                    n_recon += 1
        if n_renderer == 0:
            raise RuntimeError(
                "LagerNVS joint_train LoRA injection matched 0 renderer parameters; "
                f"check target_modules={self.lora_target_modules}"
            )
        if self.train_reconstructor_lora and n_recon == 0:
            raise RuntimeError(
                "LagerNVS train_reconstructor_lora matched 0 parameters; "
                f"check target_modules={self.reconstructor_lora_target_modules!r}"
            )
        # renderer train() for LoRA dropout (default 0).
        model.renderer.train()
        sides = "renderer"
        if self.train_reconstructor_lora:
            sides = "renderer+reconstructor"
        print(
            f"LagerNVS joint_train: injected {sides} LoRA "
            f"(rank={self.lora_rank}, alpha={self.lora_alpha}, "
            f"renderer_targets={list(self.lora_target_modules)}, "
            f"recon_targets={self.reconstructor_lora_target_modules!r}, "
            f"trainable_tensors={n_lora} "
            f"[renderer={n_renderer}, reconstructor={n_recon}])"
        )
        return model

    def _inject_frozen_lora(self, model: nn.Module) -> nn.Module:
        """Inject renderer+reconstructor LoRA modules, then freeze all weights."""
        from peft import LoraConfig, inject_adapter_in_model

        for param in model.parameters():
            param.requires_grad = False

        renderer_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=list(self.lora_target_modules),
            bias="none",
        )
        model.renderer = inject_adapter_in_model(renderer_config, model.renderer)

        recon_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.reconstructor_lora_target_modules,
            bias="none",
        )
        model.reconstructor = inject_adapter_in_model(
            recon_config, model.reconstructor
        )
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        print(
            "LagerNVS frozen LoRA: injected renderer+reconstructor adapters "
            f"(rank={self.lora_rank}) for checkpoint loading"
        )
        return model

    def _store_backbone(self, model: nn.Module) -> nn.Module:
        """Keep a reference usable by both frozen (lazy) and joint (registered) paths."""
        if self.joint_train or self.load_frozen_lora:
            self.add_module("backbone", model)
            self._model = model
        else:
            self._model = model
            if hasattr(self, "backbone"):
                delattr(self, "backbone")
        return model

    def ensure_model_loaded(self) -> nn.Module:
        """Load (and optionally LoRA-adapt) LagerNVS; safe to call repeatedly."""
        return self._load_model()

    def _load_model(self):
        existing = self._get_backbone()
        if existing is not None:
            self._onload()
            return existing
        api = self._get_api()

        ckpt_path = os.path.expanduser(self.lagernvs_ckpt)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"LagerNVS checkpoint not found: {ckpt_path}")

        model = api.EncDec_VitB8(
            pretrained_vggt=False,
            pretrained_patch_embed=False,
            attention_to_features_type="bidirectional_cross_attention",
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)

        if self.joint_train:
            model = self._inject_joint_lora(model)
        elif self.load_frozen_lora:
            model = self._inject_frozen_lora(model)
        else:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        device = self.compute_device
        if device.type != "cuda" and torch.cuda.is_available() and not self.joint_train:
            # Fallback if prepare() did not move the empty module yet.
            # Prefer the process-local CUDA device (LOCAL_RANK), not cuda:0.
            # Joint-train stays on the init device so accelerator.prepare can move it.
            device = torch.device("cuda", torch.cuda.current_device())
            self.sync_device(device)
        model.to(device)
        self._store_backbone(model)
        mode = "joint_train LoRA" if self.joint_train else "frozen"
        if self.joint_train and self.train_reconstructor_lora:
            mode = "joint_train LoRA (renderer+reconstructor)"
        elif self.load_frozen_lora:
            mode = "frozen renderer+reconstructor LoRA"
        print(f"Loaded {mode} LagerNVS from {ckpt_path} on {device}")
        return model

    def _prepare_lagernvs_inputs(
        self,
        context_images: Sequence[Image.Image],
        context_w2c: np.ndarray,
        target_w2c: np.ndarray,
        context_intrinsics: np.ndarray,
        target_intrinsics: np.ndarray,
        roge_height: int = ROGE_HEIGHT,
        roge_width: int = ROGE_WIDTH,
    ):
        """Prepare native RGB/K at LagerNVS inference resolution.

        RoGe 480x832 uses native -> cover-crop 480x832 -> direct resize 288x504.
        Other RoGe sizes retain the original native -> cover-crop 288x504 path.
        """
        num_context = len(context_images)
        infer_h, infer_w = LAGERNVS_INFER_HEIGHT, LAGERNVS_INFER_WIDTH
        resize_from_roge = (roge_height, roge_width) == (
            ROGE_480P_HEIGHT,
            ROGE_480P_WIDTH,
        )

        all_w2c = np.concatenate(
            [np.asarray(context_w2c), np.asarray(target_w2c)], axis=0
        ).astype(np.float32)
        all_k = np.concatenate(
            [np.asarray(context_intrinsics), np.asarray(target_intrinsics)], axis=0
        ).astype(np.float32)
        all_c2w = np.stack([_w2c_to_c2w(w) for w in all_w2c], axis=0)

        images_infer = []
        k_infer = []
        for i, pil_img in enumerate(context_images):
            rgb = _pil_to_rgb_uint8(pil_img)
            orig_h, orig_w = rgb.shape[:2]
            if resize_from_roge:
                rgb_roge = _resize_cover_center_crop(rgb, roge_height, roge_width)
                rgb_inf = cv2.resize(
                    rgb_roge, (infer_w, infer_h), interpolation=cv2.INTER_AREA
                )
                k_inf = _scale_intrinsic_cover_crop(
                    all_k[i], orig_w, orig_h, roge_width, roge_height
                )
                k_inf[[0, 0], [0, 2]] *= infer_w / roge_width
                k_inf[[1, 1], [1, 2]] *= infer_h / roge_height
            else:
                rgb_inf = _resize_cover_center_crop(rgb, infer_h, infer_w)
                k_inf = _scale_intrinsic_cover_crop(
                    all_k[i], orig_w, orig_h, infer_w, infer_h
                )
            images_infer.append(rgb_inf)
            k_infer.append(k_inf)

        # Targets have no RGB; scale K using context frame-0 geometry as the
        # image size reference when target K is already at the same native res.
        # Prefer explicit size from the matching context if available, else
        # assume target intrinsics share the native resolution of context[0].
        ref_w, ref_h = context_images[0].size
        for j in range(num_context, len(all_w2c)):
            if resize_from_roge:
                k_inf = _scale_intrinsic_cover_crop(
                    all_k[j], ref_w, ref_h, roge_width, roge_height
                )
                k_inf[[0, 0], [0, 2]] *= infer_w / roge_width
                k_inf[[1, 1], [1, 2]] *= infer_h / roge_height
            else:
                k_inf = _scale_intrinsic_cover_crop(
                    all_k[j], ref_w, ref_h, infer_w, infer_h
                )
            k_infer.append(k_inf)

        images_t = torch.from_numpy(np.stack(images_infer).astype(np.float32))
        images_t = images_t.permute(0, 3, 1, 2).div(255.0)
        c2w = torch.from_numpy(all_c2w).float()
        intrinsics = torch.from_numpy(
            np.stack([_fxfycxcy_from_k(k) for k in k_infer]).astype(np.float32)
        )
        return images_t, c2w, intrinsics, (infer_h, infer_w)

    def _renderer_decode_from_rays(
        self,
        model,
        rec_tokens_bv: torch.Tensor,
        rays_bv: torch.Tensor,
        *,
        outputs: Literal["tokens", "rgb", "both"] = "tokens",
        rgb_num_context: int | None = None,
    ):
        """Run LagerNVS renderer on arbitrary rays (context and/or target).

        Args:
            rec_tokens_bv: (B*V, N_rec, C)
            rays_bv: (B, V, 6, H, W)
            outputs:
                * ``tokens``: stop before FinalLayer -> (B, C, V, H/ps, W/ps)
                * ``rgb``: FinalLayer+sigmoid -> (B, V_rgb, 3, H, W)
                * ``both``: one ``renderer_core`` pass, then both of the above
            rgb_num_context: If set, FinalLayer only runs on views after this
                index (target-only RGB). Tokens still cover all V views.
                ``None`` means RGB for every view (vis path).
        """
        import einops

        renderer = model.renderer
        b, v, _, h_tgt, w_tgt = rays_bv.shape
        rays_flat = einops.rearrange(rays_bv, "b v c h w -> (b v) c h w")
        target_tokens = renderer.tgt_embedder(rays_flat)
        target_tokens = renderer.tgt_norm(target_tokens)
        register_tokens = einops.repeat(
            renderer.per_view_register_tokens,
            "n p c -> (n b1) p c",
            b1=b * v,
        )
        x = torch.cat([register_tokens, target_tokens], dim=1)
        x = renderer.renderer_core(x, rec_tokens_bv)
        x = x[:, renderer.patch_start_idx :, :]
        ph = h_tgt // renderer.patch_size
        pw = w_tgt // renderer.patch_size

        tokens = None
        if outputs in ("tokens", "both"):
            tokens = einops.rearrange(
                x, "(b v) (h w) c -> b c v h w", b=b, v=v, h=ph, w=pw
            )
        rgb = None
        if outputs in ("rgb", "both"):
            x_for_rgb = x
            v_rgb = v
            if rgb_num_context is not None:
                if not (0 <= rgb_num_context < v):
                    raise ValueError(
                        f"rgb_num_context={rgb_num_context} out of range for V={v}"
                    )
                x_views = einops.rearrange(x, "(b v) n c -> b v n c", b=b, v=v)
                x_for_rgb = einops.rearrange(
                    x_views[:, rgb_num_context:], "b v n c -> (b v) n c"
                )
                v_rgb = v - rgb_num_context
            x_rgb = renderer.output_act(renderer.final_layer(x_for_rgb))
            rgb = einops.rearrange(
                x_rgb,
                "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                b=b,
                v=v_rgb,
                h=ph,
                w=pw,
                p1=renderer.patch_size,
                p2=renderer.patch_size,
                c=3,
            )
        if outputs == "both":
            return tokens, rgb
        return tokens if outputs == "tokens" else rgb

    def _chunked_renderer_over_views(
        self,
        model,
        rec_tokens: torch.Tensor,
        rays: torch.Tensor,
        *,
        num_context: int,
        outputs: Literal["tokens", "rgb", "both"] = "tokens",
        rgb_num_context: int | None = None,
    ):
        """Apply renderer, chunking views only for large context sets."""
        import einops

        v = rays.shape[1]
        if num_context <= LAGERNVS_RENDERER_CHUNK_CONTEXT_THRESHOLD:
            rec_rep = einops.repeat(rec_tokens, "b np d -> (b v) np d", v=v)
            return self._renderer_decode_from_rays(
                model,
                rec_rep,
                rays,
                outputs=outputs,
                rgb_num_context=rgb_num_context,
            )

        token_chunks = []
        rgb_chunks = []
        for start in range(0, v, LAGERNVS_RENDERER_VIEW_CHUNK_SIZE):
            end = min(start + LAGERNVS_RENDERER_VIEW_CHUNK_SIZE, v)
            chunk_v = end - start
            rays_chunk = rays[:, start:end]
            rec_rep = einops.repeat(
                rec_tokens, "b np d -> (b v) np d", v=chunk_v
            )

            # ``rgb_num_context`` is global. A chunk before the boundary needs
            # tokens but no RGB; crossing/target chunks use a local boundary.
            if outputs == "both" and rgb_num_context is not None:
                local_context = max(0, min(end, rgb_num_context) - start)
                if local_context == chunk_v:
                    tokens = self._renderer_decode_from_rays(
                        model, rec_rep, rays_chunk, outputs="tokens"
                    )
                    token_chunks.append(tokens)
                    continue
                local_rgb_context = local_context if local_context > 0 else None
                tokens, rgb = self._renderer_decode_from_rays(
                    model,
                    rec_rep,
                    rays_chunk,
                    outputs="both",
                    rgb_num_context=local_rgb_context,
                )
                token_chunks.append(tokens)
                rgb_chunks.append(rgb)
                continue

            decoded = self._renderer_decode_from_rays(
                model,
                rec_rep,
                rays_chunk,
                outputs=outputs,
                rgb_num_context=None,
            )
            if outputs == "both":
                tokens, rgb = decoded
                token_chunks.append(tokens)
                rgb_chunks.append(rgb)
            elif outputs == "tokens":
                token_chunks.append(decoded)
            else:
                rgb_chunks.append(decoded)

        tokens = torch.cat(token_chunks, dim=2) if token_chunks else None
        rgb = torch.cat(rgb_chunks, dim=1) if rgb_chunks else None
        if outputs == "both":
            return tokens, rgb
        return tokens if outputs == "tokens" else rgb

    @staticmethod
    def _rgb_batch_to_pils(
        pred: torch.Tensor,
        roge_height: int,
        roge_width: int,
    ) -> list[Image.Image]:
        """Convert ``(V, 3, H, W)`` float RGB in [0, 1] to RoGe-resolution PILs."""
        pred = pred.float().cpu()
        out: list[Image.Image] = []
        for i in range(pred.shape[0]):
            rgb_inf = (
                pred[i].permute(1, 2, 0).numpy() * 255.0
            ).clip(0, 255).astype(np.uint8)
            rgb_roge = cv2.resize(
                rgb_inf, (roge_width, roge_height), interpolation=cv2.INTER_AREA
            )
            out.append(Image.fromarray(rgb_roge, mode="RGB"))
        return out

    def _prepare_all_view_renderer_inputs(
        self,
        context_images: Sequence[Image.Image],
        context_w2c: np.ndarray,
        target_w2c: np.ndarray,
        context_intrinsics: np.ndarray,
        target_intrinsics: np.ndarray,
        roge_height: int = ROGE_HEIGHT,
        roge_width: int = ROGE_WIDTH,
    ):
        """Shared LagerNVS preprocess + reconstructor setup for all-view decode."""
        import einops

        model = self._load_model()
        api = self._get_api()

        num_context = len(context_images)
        images_t, c2w_all, intrinsics_all, infer_hw = self._prepare_lagernvs_inputs(
            context_images,
            context_w2c,
            target_w2c,
            context_intrinsics,
            target_intrinsics,
            roge_height=roge_height,
            roge_width=roge_width,
        )
        num_total = c2w_all.shape[0]
        c2w_all, camera_scale, _ = api.normalize_extrinsics(
            c2w_all, num_cond_views=num_context
        )

        device = self.compute_device
        if device.type != "cuda":
            raise RuntimeError(
                f"LagerNVS requires CUDA for xformers attention, got device={device}."
            )
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability(device)[0] >= 8
            else torch.float16
        )

        # Split ``build_cam_cond``: tokens stay on CPU (13 floats per view),
        # rays are built straight on the GPU (300 MB at 87 views x 288x504).
        cam_tokens = _build_cam_cond_tokens(
            api,
            c2w_poses=c2w_all,
            intrinsics_fxfycxcy_px=intrinsics_all,
            num_cond_views=num_context,
            tgt_hw=infer_hw,
            camera_scale=camera_scale,
        )
        rays = _plucker_rays_on_device(c2w_all, intrinsics_all, infer_hw, device)

        cond_images = images_t[:num_context].unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        cam_cond = cam_tokens[:num_context].unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        rays = rays.unsqueeze(0)
        return model, cond_images, cam_cond, rays, dtype, num_context, num_total, einops

    def extract_all_view_tokens(
        self,
        context_images: Sequence[Image.Image],
        context_w2c: np.ndarray,
        target_w2c: np.ndarray,
        context_intrinsics: np.ndarray,
        target_intrinsics: np.ndarray,
        *,
        also_rgb: bool = False,
        enable_grad: bool = False,
        rgb_target_only: bool = False,
        roge_height: int = ROGE_HEIGHT,
        roge_width: int = ROGE_WIDTH,
    ):
        """Decoder tokens for every context+target ray: (1, 768, V, 36, 63).

        Note: stock LagerNVS only feeds target rays to the renderer; this path
        intentionally also decodes context rays.

        If ``also_rgb``, also returns RoGe-resolution RGB PILs from the same
        ``renderer_core`` pass (FinalLayer only; no second attention) when
        ``enable_grad`` is False. With ``enable_grad=True`` and ``also_rgb``,
        returns the live RGB tensor instead (for NVS loss): full
        ``(1, V, 3, H, W)`` by default, or target-only ``(1, V_tgt, 3, H, W)``
        when ``rgb_target_only=True`` (skips FinalLayer on context).

        Reconstructor runs under ``no_grad`` unless ``train_reconstructor_lora``
        and ``enable_grad``. Renderer grads are enabled only when
        ``enable_grad=True`` (joint_train).
        """
        model, cond_images, cam_cond, rays, dtype, num_context, _, einops = (
            self._prepare_all_view_renderer_inputs(
                context_images,
                context_w2c,
                target_w2c,
                context_intrinsics,
                target_intrinsics,
                roge_height=roge_height,
                roge_width=roge_width,
            )
        )

        open_recon_grad = bool(
            enable_grad and self.train_reconstructor_lora and self.joint_train
        )
        rgb_num_context = num_context if (also_rgb and rgb_target_only) else None
        grad_ctx = nullcontext() if enable_grad else torch.no_grad()
        with grad_ctx:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                if open_recon_grad:
                    rec_tokens = model.reconstructor(cond_images, cam_cond)
                    rec_tokens = einops.rearrange(
                        rec_tokens, "b v p c -> b (v p) c"
                    )
                else:
                    with torch.no_grad():
                        rec_tokens = model.reconstructor(cond_images, cam_cond)
                    rec_tokens = einops.rearrange(
                        rec_tokens.detach(), "b v p c -> b (v p) c"
                    )
                if also_rgb:
                    tokens, pred = self._chunked_renderer_over_views(
                        model,
                        rec_tokens,
                        rays,
                        num_context=num_context,
                        outputs="both",
                        rgb_num_context=rgb_num_context,
                    )
                else:
                    tokens = self._chunked_renderer_over_views(
                        model,
                        rec_tokens,
                        rays,
                        num_context=num_context,
                        outputs="tokens",
                    )
                    pred = None

        self._offload()
        if also_rgb:
            if enable_grad:
                return tokens, pred
            rgb_pils = self._rgb_batch_to_pils(pred[0], roge_height, roge_width)
            return tokens, rgb_pils
        return tokens  # (1, 768, V, 36, 63)

    @torch.no_grad()
    def predict_all_view_rgb(
        self,
        context_images: Sequence[Image.Image],
        context_w2c: np.ndarray,
        target_w2c: np.ndarray,
        context_intrinsics: np.ndarray,
        target_intrinsics: np.ndarray,
        roge_height: int = ROGE_HEIGHT,
        roge_width: int = ROGE_WIDTH,
    ) -> list[Image.Image]:
        """Render RGB for every context+target ray; resize to RoGe resolution.

        Prefer ``build_lagernvs_geom(..., cache_aux_rgb=True)`` then
        ``last_pred_all_views`` for vis. This method is the fallback (full
        renderer decode: 12 attn + FinalLayer).
        """
        model, cond_images, cam_cond, rays, dtype, num_context, _, einops = (
            self._prepare_all_view_renderer_inputs(
                context_images,
                context_w2c,
                target_w2c,
                context_intrinsics,
                target_intrinsics,
                roge_height=roge_height,
                roge_width=roge_width,
            )
        )

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            rec_tokens = model.reconstructor(cond_images, cam_cond)
            rec_tokens = einops.rearrange(rec_tokens, "b v p c -> b (v p) c")
            pred = self._chunked_renderer_over_views(
                model,
                rec_tokens,
                rays,
                num_context=num_context,
                outputs="rgb",
            )

        out = self._rgb_batch_to_pils(pred[0], roge_height, roge_width)
        self.last_pred_all_views = out
        self._offload()
        return out

    def _gt_to_infer_rgb_tensor(
        self,
        images: Sequence[Image.Image] | torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        roge_height: int = ROGE_HEIGHT,
        roge_width: int = ROGE_WIDTH,
    ) -> torch.Tensor:
        """GT frames as ``(1, V, 3, H, W)`` in [0, 1] at LagerNVS infer res.

        Accepts PILs at any resolution (cover-cropped here) or a
        ``(V, H, W, 3)`` uint8 tensor already cropped by
        ``lagernvs_infer_cover_crop`` in a DataLoader worker. The uint8 form
        transfers ~4x less over PCIe and keeps the resize off this process.
        """
        if isinstance(images, torch.Tensor):
            frames_u8 = images
        else:
            frames = [
                lagernvs_infer_cover_crop(pil_img, roge_height, roge_width)
                for pil_img in images
            ]
            if not frames:
                raise ValueError("expected at least one GT image for NVS loss")
            frames_u8 = torch.from_numpy(np.stack(frames))
        if frames_u8.shape[0] == 0:
            raise ValueError("expected at least one GT image for NVS loss")
        t = frames_u8.to(device=device, dtype=torch.float32)
        t = t.permute(0, 3, 1, 2).div(255.0).unsqueeze(0)
        return t.to(dtype=dtype)

    def compute_nvs_aux_loss(
        self,
        pred_rgb: torch.Tensor,
        target_gt_images: Sequence[Image.Image] | torch.Tensor,
        num_context: int,
        *,
        pred_target_only: bool = False,
        roge_height: int = ROGE_HEIGHT,
        roge_width: int = ROGE_WIDTH,
    ) -> torch.Tensor:
        """MSE between predicted target-view RGB and GT cover-cropped frames.

        Args:
            pred_rgb: ``(1, V, 3, H, W)`` in [0, 1]. Full context+target when
                ``pred_target_only=False``; target-only when True.
            target_gt_images: either all frames (context+target) or target-only,
                as PILs or a worker-cropped ``(V, H, W, 3)`` uint8 tensor.
            num_context: number of leading context views (used to slice full
                preds / all-frame GT lists).
            pred_target_only: If True, ``pred_rgb`` already excludes context.
        """
        if pred_rgb.dim() != 5 or pred_rgb.shape[0] != 1:
            raise ValueError(f"pred_rgb expected (1, V, 3, H, W), got {tuple(pred_rgb.shape)}")
        if pred_target_only:
            pred_tgt = pred_rgb
            n_tgt = pred_tgt.shape[1]
            if n_tgt <= 0:
                raise ValueError("NVS loss needs target views; got empty target-only pred")
            # All-frames GT length is context + targets.
            num_total = num_context + n_tgt
        else:
            num_total = pred_rgb.shape[1]
            if num_total <= num_context:
                raise ValueError(
                    f"NVS loss needs target views; got V={num_total}, num_context={num_context}"
                )
            pred_tgt = pred_rgb[:, num_context:]
            n_tgt = pred_tgt.shape[1]

        # Slice (not list()) so a worker-cropped uint8 tensor stays a tensor.
        if len(target_gt_images) == num_total:
            gt_src = target_gt_images[num_context:]
        elif len(target_gt_images) == n_tgt:
            gt_src = target_gt_images
        else:
            raise ValueError(
                "target_gt_images length must be all-frames "
                f"({num_total}) or target-only ({n_tgt}), got {len(target_gt_images)}"
            )

        gt = self._gt_to_infer_rgb_tensor(
            gt_src,
            device=pred_tgt.device,
            dtype=pred_tgt.dtype,
            roge_height=roge_height,
            roge_width=roge_width,
        )
        if gt.shape != pred_tgt.shape:
            raise ValueError(
                f"NVS GT shape {tuple(gt.shape)} != pred target shape {tuple(pred_tgt.shape)}"
            )
        return F.mse_loss(pred_tgt.float(), gt.float())

    def build_lagernvs_geom(
        self,
        context_images: Sequence[Image.Image],
        context_w2c: np.ndarray,
        target_w2c: np.ndarray,
        context_intrinsics: np.ndarray,
        target_intrinsics: np.ndarray,
        num_context: int,
        num_target_latent_frames: int,
        roge_height: int = ROGE_HEIGHT,
        roge_width: int = ROGE_WIDTH,
        dtype: torch.dtype = torch.bfloat16,
        pipe=None,
        cache_aux_rgb: bool = False,
        target_gt_images: Sequence[Image.Image] | None = None,
        compute_nvs_loss: bool = False,
    ):
        """Build geom ``(F_lat, C*4, H/8, W/8)`` for ``vggt_geom`` injection.

        ``cache_aux_rgb`` runs FinalLayer RGB readout in the same decoder pass
        (for vis). Keep ``False`` on the training loss path unless joint_train
        needs NVS aux loss via ``compute_nvs_loss``.

        When ``compute_nvs_loss=True`` (joint_train), returns
        ``(geom, nvs_loss)``; otherwise returns ``geom`` only.

        Pipeline:
          1. All V=context+target rays -> decoder **ray** tokens (1, 768, V, ...)
             (12 attn layers; vis/NVS optionally adds FinalLayer readout).
             Training NVS (no vis) runs FinalLayer on **target views only**.
          2. Project -> (1, C, V, H/8, W/8) where C=``geom_channels`` (pre-pack)
          3. VAE-style temporal pack (expand + channel-pack group=4) ->
             (F_lat, C*4, H/8, W/8)
        """
        if pipe is not None and getattr(pipe, "device", None) is not None:
            self.sync_device(pipe.device)

        if compute_nvs_loss and not self.joint_train:
            raise ValueError("compute_nvs_loss requires joint_train=True")
        if compute_nvs_loss and target_gt_images is None:
            raise ValueError("compute_nvs_loss requires target_gt_images")

        latent_h, latent_w = roge_height // 8, roge_width // 8
        # Open renderer graph whenever joint_train is active and grads are enabled
        # (training). Vis runs under ``@torch.no_grad()`` and stays on the PIL path.
        enable_grad = bool(self.joint_train) and torch.is_grad_enabled()
        need_rgb = bool(cache_aux_rgb or compute_nvs_loss)
        # Skip FinalLayer on context unless vis needs all-view RGB PILs.
        rgb_target_only = bool(compute_nvs_loss and not cache_aux_rgb)

        decoded = self.extract_all_view_tokens(
            context_images,
            context_w2c,
            target_w2c,
            context_intrinsics,
            target_intrinsics,
            also_rgb=need_rgb,
            enable_grad=enable_grad,
            rgb_target_only=rgb_target_only,
            roge_height=roge_height,
            roge_width=roge_width,
        )
        nvs_loss = None
        if need_rgb:
            tokens, rgb_or_pils = decoded
            if enable_grad:
                if compute_nvs_loss:
                    nvs_loss = self.compute_nvs_aux_loss(
                        rgb_or_pils,
                        target_gt_images,
                        num_context=num_context,
                        pred_target_only=rgb_target_only,
                        roge_height=roge_height,
                        roge_width=roge_width,
                    )
                    self.last_nvs_loss = nvs_loss.detach()
                else:
                    self.last_nvs_loss = None
                if cache_aux_rgb:
                    self.last_pred_all_views = self._rgb_batch_to_pils(
                        rgb_or_pils[0].detach(), roge_height, roge_width
                    )
                else:
                    self.last_pred_all_views = None
            else:
                self.last_pred_all_views = rgb_or_pils if cache_aux_rgb else None
                self.last_nvs_loss = None
        else:
            tokens = decoded
            self.last_pred_all_views = None
            self.last_nvs_loss = None

        # (1, 768, V, ph, pw) -> (V, 768, ph, pw)
        patch_grids = tokens[0].permute(1, 0, 2, 3).contiguous()
        patch_grids = _cast_patch_grids_for_projector(
            patch_grids, self.projector.proj.weight
        )
        projected = self.projector(patch_grids, latent_h, latent_w)
        # (V, C, lh, lw) -> (1, C, V, lh, lw)
        projected = projected.permute(1, 0, 2, 3).unsqueeze(0).contiguous()

        num_total = projected.shape[2]
        if num_total < num_context:
            raise ValueError(
                f"LagerNVS geom tokens have {num_total} views but num_context={num_context}"
            )
        num_target_video_frames = num_total - num_context
        if num_target_video_frames <= 0:
            raise ValueError(
                "LagerNVS geom pack requires target video frames after context, "
                f"got num_total={num_total}, num_context={num_context}"
            )
        # Already at latent spatial resolution: temporal pack only (no PixelUnshuffle).
        from diffsynth.pipelines.wan_video import pack_hybrid_condition_map

        packed = pack_hybrid_condition_map(
            projected,
            num_context_frames=num_context,
            num_target_latent_frames=num_target_latent_frames,
            num_target_video_frames=num_target_video_frames,
            group=LAGERNVS_PACK_GROUP,
        )
        # (1, C*4, F_lat, H, W) -> (F_lat, C*4, H, W)
        geom = packed[0].permute(1, 0, 2, 3).contiguous().to(dtype=dtype)

        if compute_nvs_loss:
            if nvs_loss is None:
                raise RuntimeError(
                    "compute_nvs_loss=True but NVS loss was not computed "
                    "(grads disabled?). Enable grad mode for joint_train."
                )
            return geom, nvs_loss
        return geom

    def _load_submodule_state(self, state_dict: dict, submodule_name: str) -> int:
        """Load one trainable head by exact key prefixes; report missing/unexpected."""
        prefixes = (
            f"lagernvs_condition.{submodule_name}.",
            f"{submodule_name}.",
        )
        sub_state = {}
        for key, value in state_dict.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    sub_state[key[len(prefix):]] = value
                    break
        submodule = getattr(self, submodule_name, None)
        if not sub_state or submodule is None:
            return 0
        load_result = submodule.load_state_dict(sub_state, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                f"Resume warning: lagernvs_condition.{submodule_name} "
                f"missing={list(load_result.missing_keys)} "
                f"unexpected={list(load_result.unexpected_keys)}"
            )
        return len(sub_state)

    def load_trainable_state(self, state_dict: dict):
        """Load projector / joint-train LoRA weights from a training checkpoint."""
        loaded_proj = self._load_submodule_state(state_dict, "projector")
        loaded_lora = 0
        if self.joint_train or self.load_frozen_lora:
            loaded_lora = self._load_backbone_lora_state(state_dict)
        if self.load_frozen_lora and loaded_lora == 0:
            raise ValueError(
                "Requested frozen LagerNVS LoRA, but the resume checkpoint "
                "contains no lagernvs_condition.backbone LoRA tensors"
            )
        if loaded_proj or loaded_lora:
            print(
                f"Loaded LagerNVS heads "
                f"(projector={loaded_proj}, backbone_lora={loaded_lora} keys)"
            )
        else:
            print(
                "Resume warning: no lagernvs_condition.projector"
                + (" / backbone LoRA" if self.joint_train else "")
                + " weights found in checkpoint"
            )

    def _load_backbone_lora_state(self, state_dict: dict) -> int:
        """Load peft LoRA tensors into the registered ``backbone`` submodule."""
        model = self._get_backbone()
        if model is None:
            return 0
        prefixes = (
            "lagernvs_condition.backbone.",
            "backbone.",
        )
        sub_state = {}
        for key, value in state_dict.items():
            if "lora_" not in key:
                continue
            for prefix in prefixes:
                if key.startswith(prefix):
                    sub_state[key[len(prefix):]] = value
                    break
        if not sub_state:
            return 0
        load_result = model.load_state_dict(sub_state, strict=False)
        # Only complain about unexpected LoRA keys; base weights are absent by design.
        unexpected_lora = [k for k in load_result.unexpected_keys if "lora_" in k]
        if unexpected_lora:
            print(
                f"Resume warning: lagernvs_condition.backbone LoRA "
                f"unexpected={unexpected_lora}"
            )
        return len(sub_state)

    @torch.no_grad()
    def build_vis_aux_frames(
        self,
        context_images: Sequence[Image.Image],
        context_w2c: np.ndarray,
        target_w2c: np.ndarray,
        context_intrinsics: np.ndarray,
        target_intrinsics: np.ndarray,
        num_total_frames: int,
        roge_height: int = ROGE_HEIGHT,
        roge_width: int = ROGE_WIDTH,
    ) -> list[Image.Image]:
        """First gt_vs_pred row: LagerNVS RGB for all context+target views.

        Prefer frames cached by ``build_lagernvs_geom(..., cache_aux_rgb=True)``.
        """
        black = Image.new("RGB", (roge_width, roge_height), (0, 0, 0))
        if self.last_pred_all_views is not None:
            frames = list(self.last_pred_all_views)
        else:
            frames = self.predict_all_view_rgb(
                context_images,
                context_w2c,
                target_w2c,
                context_intrinsics,
                target_intrinsics,
                roge_height=roge_height,
                roge_width=roge_width,
            )
        while len(frames) < num_total_frames:
            frames.append(black.copy())
        return frames[:num_total_frames]
