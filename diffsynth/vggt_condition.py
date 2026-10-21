"""DiT input-channel helpers shared by RoGe raymap / geom conditioning.

Concat budget: noise(16) + I2V y(20) + packed raymap(1536) = 1572.
``--raymap_pe`` drops the raymap slice from concat (base 36) and injects it by add.
"""

from __future__ import annotations

import torch

# DiT concat budget: noise(16) + I2V y(mask4+vae16=20) + packed raymap(1536)
LATENT_CHANNELS = 16
Y_CHANNELS = 20
LATENT_Y_CHANNELS = LATENT_CHANNELS + Y_CHANNELS  # 36; pe-only base when --raymap_pe
# VAE-style temporal pack (6x4=24ch) then PixelUnshuffle(8) -> 1536
RAYMAP_CHANNELS = 24 * 8 * 8
BASE_IN_DIM = LATENT_Y_CHANNELS + RAYMAP_CHANNELS  # 1572
# --raymap_pe: Unshuffle(2) packs 1536 -> 6144 before a separate 1x1 pe
RAYMAP_PE_IN_CHANNELS = RAYMAP_CHANNELS * 2 * 2  # 6144
# Legacy non-pack single-frame Plücker: PixelUnshuffle(8) on 6ch -> 384
RAYMAP_CHANNELS_LEGACY = 6 * 8 * 8
BASE_IN_DIM_LEGACY = LATENT_Y_CHANNELS + RAYMAP_CHANNELS_LEGACY  # 420


def raymap_channels() -> int:
    """Raymap channels after hybrid pack prep (concat path, or raymap_pe input)."""
    return RAYMAP_CHANNELS


def raymap_pe_in_channels() -> int:
    """Channels into ``raymap_pe`` after PixelUnshuffle(2) on packed raymap."""
    return RAYMAP_PE_IN_CHANNELS


def base_in_dim(raymap_pe: bool = False) -> int:
    """DiT ``patch_embedding`` in_dim.

    Default: latent + y + packed raymap (1572).
    With ``raymap_pe=True``: latent + y only (36); raymap is injected via add.
    """
    if raymap_pe:
        return LATENT_Y_CHANNELS
    return BASE_IN_DIM


def _unwrap_patch_embedding(dit_model):
    pe = dit_model.patch_embedding
    return pe.module if hasattr(pe, "module") else pe


def load_patch_embedding_from_checkpoint(dit_model, ckpt_sd: dict, ckpt_path: str = ""):
    """Load patch_embedding weights, with safe partial-copy when ``in_dim`` differs.

    Concat layout: ``[latent 16 | y 20 | raymap 1536 | geom?]``.
    ``--raymap_pe`` layout: ``[latent 16 | y 20 | geom?]`` (raymap injected via add).
    ``--geom_pe`` drops geom from concat (raymap may still be concat or pe).

    When ``in_dim`` differs (legacy pack, concat<->pe, etc.), only the shared
    prefix ``latent+y`` (36 channels) is copied so raymap slices are not pasted
    onto the wrong budget. Matching shapes load the full weight (including geom).
    """
    loaded_w = None
    loaded_b = None
    for key, value in ckpt_sd.items():
        if key.endswith("patch_embedding.weight"):
            loaded_w = value
        elif key.endswith("patch_embedding.bias"):
            loaded_b = value

    if loaded_w is None:
        print(f"Resume warning: no patch_embedding.weight in {ckpt_path}")
        return

    pe = _unwrap_patch_embedding(dit_model)
    target_w = pe.weight

    if loaded_w.shape != target_w.shape:
        shared_out = min(loaded_w.shape[0], target_w.shape[0])
        # Prefer latent+y only when migrating into/out of packed raymap dims.
        pack_mismatch = (
            loaded_w.shape[1] != target_w.shape[1]
            and (
                loaded_w.shape[1] >= BASE_IN_DIM
                or target_w.shape[1] >= BASE_IN_DIM
                or loaded_w.shape[1] == BASE_IN_DIM_LEGACY
                or target_w.shape[1] == BASE_IN_DIM_LEGACY
            )
        )
        if pack_mismatch:
            shared_in = min(
                LATENT_CHANNELS + Y_CHANNELS,
                loaded_w.shape[1],
                target_w.shape[1],
            )
        else:
            shared_in = min(loaded_w.shape[1], target_w.shape[1])
        with torch.no_grad():
            target_w[:shared_out, :shared_in].copy_(
                loaded_w[:shared_out, :shared_in].to(
                    device=target_w.device, dtype=target_w.dtype
                )
            )
        print(
            f"Resume: partial patch_embedding.weight from {ckpt_path} "
            f"({shared_out} out x {shared_in} in copied"
            f"{'; latent+y only (pack dim change)' if pack_mismatch else ''}; "
            f"target {tuple(target_w.shape)}, ckpt {tuple(loaded_w.shape)})"
        )
    else:
        with torch.no_grad():
            target_w.copy_(
                loaded_w.to(device=target_w.device, dtype=target_w.dtype)
            )
        print(f"Resume: loaded patch_embedding.weight from {ckpt_path}")

    if loaded_b is not None and pe.bias is not None:
        if loaded_b.shape == pe.bias.shape:
            with torch.no_grad():
                pe.bias.copy_(
                    loaded_b.to(device=pe.bias.device, dtype=pe.bias.dtype)
                )
        else:
            print(
                f"Resume: skipped patch_embedding.bias (shape {tuple(loaded_b.shape)} "
                f"vs {tuple(pe.bias.shape)})"
            )
