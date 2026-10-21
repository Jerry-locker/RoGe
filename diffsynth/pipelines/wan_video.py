import torch, types
import torch.nn as nn
import numpy as np
from PIL import Image
from einops import rearrange
from typing import Optional, Union
from tqdm import tqdm
from typing_extensions import Literal

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, RaymapPatchEmbed, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder


def enable_raymap_pe(
    dit: WanModel,
    raymap_channels: int = 1536,
    zero_init: bool = False,
) -> RaymapPatchEmbed:
    """Attach ``RaymapPatchEmbed`` for residual raymap injection (``--raymap_pe``).

    Packed raymap ``(B, raymap_channels, T, H/8, W/8)`` is PixelUnshuffled(2)
    then projected to ``dit.dim`` and residual-added after ``patch_embedding``
    (``x = patch_embedding(...) + raymap_pe(raymap)``).
    """
    pe_ref = dit.patch_embedding
    if hasattr(pe_ref, "module"):
        pe_ref = pe_ref.module
    pe_device = pe_ref.weight.device
    pe_dtype = pe_ref.weight.dtype
    module = RaymapPatchEmbed(
        in_channels=int(raymap_channels),
        out_dim=int(dit.dim),
        unshuffle=2,
        zero_init=bool(zero_init),
    ).to(device=pe_device, dtype=pe_dtype)
    dit.raymap_pe = module
    dit.use_raymap_pe = True
    print(
        f"Enabled raymap_pe: PixelUnshuffle(2)+Conv3d(1x1) "
        f"{raymap_channels}*4={raymap_channels * 4} -> dim={dit.dim}"
        f" (zero_init={bool(zero_init)})"
    )
    return module


def enable_geom_pe(
    dit: WanModel,
    geom_channels: int,
    zero_init: bool = False,
) -> RaymapPatchEmbed:
    """Attach residual PE for VGGT / LagerNVS ``vggt_geom`` (``--geom_pe``).

    Same architecture as ``raymap_pe``: PixelUnshuffle(2) + 1x1 Conv3d to
    ``dit.dim``, residual-added after ``patch_embedding``.
    ``geom_channels`` is the concat width (VGGT ``vggt_channels``, or LagerNVS
    packed ``lagernvs_channels * 4``).
    """
    pe_ref = dit.patch_embedding
    if hasattr(pe_ref, "module"):
        pe_ref = pe_ref.module
    pe_device = pe_ref.weight.device
    pe_dtype = pe_ref.weight.dtype
    module = RaymapPatchEmbed(
        in_channels=int(geom_channels),
        out_dim=int(dit.dim),
        unshuffle=2,
        zero_init=bool(zero_init),
    ).to(device=pe_device, dtype=pe_dtype)
    dit.geom_pe = module
    dit.use_geom_pe = True
    print(
        f"Enabled geom_pe: PixelUnshuffle(2)+Conv3d(1x1) "
        f"{geom_channels}*4={geom_channels * 4} -> dim={dit.dim}"
        f" (zero_init={bool(zero_init)})"
    )
    return module


def _load_named_pe_from_checkpoint(
    dit: WanModel,
    ckpt_sd: dict,
    attr_name: str,
    ckpt_path: str = "",
) -> None:
    """Load ``{attr_name}.*`` (or ``*.{attr_name}.*``) weights into ``dit.{attr_name}``."""
    module = getattr(dit, attr_name, None)
    if module is None:
        return
    if hasattr(module, "module"):
        module = module.module
    prefix = f"{attr_name}."
    infix = f".{attr_name}."
    loaded = {}
    for key, value in ckpt_sd.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith(prefix):
            loaded[key[len(prefix):]] = value
        elif infix in key:
            loaded[key.split(infix, 1)[1]] = value
    if not loaded:
        print(f"Resume: no {attr_name} weights in {ckpt_path}")
        return
    remapped = {}
    for key, value in loaded.items():
        if key.startswith("module."):
            remapped[key[len("module."):]] = value
        else:
            remapped[key] = value
    missing, unexpected = module.load_state_dict(remapped, strict=False)
    print(
        f"Resume: loaded {attr_name} from {ckpt_path} "
        f"({len(remapped)} tensors; missing={list(missing)[:5]}, unexpected={list(unexpected)[:5]})"
    )


def load_raymap_pe_from_checkpoint(dit: WanModel, ckpt_sd: dict, ckpt_path: str = "") -> None:
    """Load ``raymap_pe.*`` weights from a training checkpoint when present."""
    _load_named_pe_from_checkpoint(dit, ckpt_sd, "raymap_pe", ckpt_path)


def load_geom_pe_from_checkpoint(dit: WanModel, ckpt_sd: dict, ckpt_path: str = "") -> None:
    """Load ``geom_pe.*`` weights from a training checkpoint when present."""
    _load_named_pe_from_checkpoint(dit, ckpt_sd, "geom_pe", ckpt_path)


def enable_modality_tokens(
    dit: WanModel,
    vggt_channels: int | None = None,
    raymap_channels: int = 1536,
) -> None:
    """Enable learnable modality tokens (channel-wise biases) for conditioning streams.

    This repo concatenates heterogeneous inputs along the channel axis before
    patchification (Conv3d). A close analogue to "modality tokens" is therefore
    a per-stream learnable bias broadcast to (B, C, F, H, W) and added before
    concatenation.
    """
    setattr(dit, "modality_tokens_enabled", True)

    if not hasattr(dit, "modality_token_x"):
        # Latent channels are fixed by the backbone out_dim (typically 16).
        dit.modality_token_x = nn.Parameter(torch.zeros(int(dit.out_dim)))
    if not hasattr(dit, "modality_token_y"):
        # y is concatenated as [msk(4ch), vae_latents(out_dim ch)].
        dit.modality_token_y = nn.Parameter(torch.zeros(int(dit.out_dim) + 4))
    if not hasattr(dit, "modality_token_raymap"):
        # Hybrid pack: PixelUnshuffle(24ch, 8) -> 1536.
        dit.modality_token_raymap = nn.Parameter(torch.zeros(int(raymap_channels)))
    if vggt_channels is not None and not hasattr(dit, "modality_token_vggt"):
        dit.modality_token_vggt = nn.Parameter(torch.zeros(int(vggt_channels)))


def load_modality_tokens_from_checkpoint(dit: WanModel, ckpt_sd: dict, ckpt_path: str = "") -> None:
    """Load modality token parameters saved in a training checkpoint."""
    loaded = 0
    for name in ("modality_token_x", "modality_token_y", "modality_token_raymap", "modality_token_vggt"):
        param = getattr(dit, name, None)
        if not isinstance(param, nn.Parameter):
            continue
        loaded_w = None
        for key, value in ckpt_sd.items():
            if key == name or key.endswith(f".{name}"):
                loaded_w = value
                break
        if loaded_w is None:
            continue
        if loaded_w.shape != param.shape:
            print(
                f"Resume warning: skipped {name} (shape {tuple(loaded_w.shape)} "
                f"vs {tuple(param.shape)})"
            )
            continue
        with torch.no_grad():
            param.copy_(loaded_w.to(device=param.device, dtype=param.dtype))
        loaded += 1
    if loaded:
        print(f"Resume: loaded {loaded} modality token(s) from {ckpt_path}")


def _maybe_apply_modality_token(x: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
    # token: (C,) broadcast to (B, C, F, H, W)
    return x + token.view(1, -1, 1, 1, 1).to(device=x.device, dtype=x.dtype)


def wan_native_num_latent_frames(num_frames: int) -> int:
    return (num_frames - 1) // 4 + 1


def wan_native_slot_end_indices(num_frames: int, num_latent_frames: int | None = None) -> list[int]:
    if num_latent_frames is None:
        num_latent_frames = wan_native_num_latent_frames(num_frames)
    return [0 if i == 0 else min(4 * i, num_frames - 1) for i in range(num_latent_frames)]


def _as_optional_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return int(value.flatten()[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _as_optional_int(value[0])
    return int(value)


def _build_rope_grid(dit: WanModel, f: int, h: int, w: int, temporal_freqs: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        temporal_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1)


def _build_split_rope_freqs(
    dit: WanModel,
    f: int,
    h: int,
    w: int,
    num_output_frames=None,
    disable_context_temporal_rope: bool = False,
) -> torch.Tensor:
    if not getattr(dit, "use_temporal_rope", False):
        return _build_rope_grid(dit, f, h, w, torch.ones_like(dit.freqs[0][:f]))

    num_output = _as_optional_int(num_output_frames)
    if num_output is None or num_output <= 0:
        raise ValueError("split temporal RoPE requires positive num_output_frames")
    if num_output >= f:
        return _build_rope_grid(dit, f, h, w, dit.freqs[0][:f])

    num_context = f - num_output
    if disable_context_temporal_rope:
        context_temporal = torch.ones_like(dit.freqs[0][:num_context])
    else:
        context_temporal = dit.freqs[0][:num_context]
    context_freqs = _build_rope_grid(dit, num_context, h, w, context_temporal)
    target_freqs = _build_rope_grid(dit, num_output, h, w, dit.freqs[0][:num_output])
    return torch.cat([context_freqs, target_freqs], dim=0)


def _vae_encode_video(pipe: "WanVideoPipeline", video: torch.Tensor, tiled, tile_size, tile_stride) -> torch.Tensor:
    encoded = pipe.vae.encode(
        video.to(dtype=pipe.torch_dtype, device=pipe.device),
        device=pipe.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    if isinstance(encoded, (list, tuple)):
        encoded = encoded[0]
    return encoded.to(dtype=pipe.torch_dtype, device=pipe.device)


def _as_bcfhw_raymap(raymap: torch.Tensor, in_channels: int = 6) -> torch.Tensor:
    if raymap.dim() == 4:
        raymap = raymap.permute(1, 0, 2, 3).unsqueeze(0)
    elif raymap.dim() == 5:
        if raymap.shape[1] != in_channels and raymap.shape[2] == in_channels:
            raymap = raymap.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Expected raymap with 4 or 5 dims, got {tuple(raymap.shape)}")

    if raymap.shape[1] != in_channels:
        raise ValueError(
            "raymap condition expects full-resolution Plucker rays "
            f"with {in_channels} channels, got shape {tuple(raymap.shape)}"
        )
    return raymap


def _spatial_pixel_unshuffle_raymap(raymap: torch.Tensor, downscale_factor: int = 8) -> torch.Tensor:
    b, c, f, h, w = raymap.shape
    if h % downscale_factor != 0 or w % downscale_factor != 0:
        raise ValueError(
            f"Raymap spatial size {(h, w)} must be divisible by {downscale_factor}"
        )
    raymap_2d = raymap.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
    raymap_2d = torch.nn.functional.pixel_unshuffle(raymap_2d, downscale_factor)
    h_ds, w_ds = raymap_2d.shape[-2:]
    out_channels = c * downscale_factor * downscale_factor
    return raymap_2d.view(b, f, out_channels, h_ds, w_ds).permute(0, 2, 1, 3, 4).contiguous()


def _channel_pack_raymap_frames(raymap: torch.Tensor, group: int = 4) -> torch.Tensor:
    """Pack every ``group`` temporal frames into channels (Fun Control-Camera style).

    ``(B, C, F, H, W) -> (B, C*group, F//group, H, W)``.
    """
    b, c, f, h, w = raymap.shape
    if f % group != 0:
        raise ValueError(f"temporal length {f} is not divisible by pack group {group}")
    x = raymap.transpose(1, 2).contiguous()  # (B, F, C, H, W)
    x = x.view(b, f // group, group, c, h, w).transpose(2, 3).contiguous()
    x = x.view(b, f // group, c * group, h, w).transpose(1, 2).contiguous()
    return x


def _expand_hybrid_vae_temporal_frames(
    feat: torch.Tensor,
    num_context_frames: int,
    num_target_latent_frames: int,
    num_target_video_frames: int,
) -> torch.Tensor:
    """Expand hybrid video features to a 4x VAE-aligned temporal grid.

    ``feat`` is ``(B, C, Nc+Nt, H, W)``. Each context frame and the first target
    frame are repeated 4x; remaining target frames are kept (padded if needed).
    Returns ``(B, C, 4*(Nc+Nt_lat), H, W)``.
    """
    if feat.dim() != 5:
        raise ValueError(f"Expected (B,C,F,H,W) feature map, got {tuple(feat.shape)}")
    if num_context_frames <= 0:
        raise ValueError("hybrid temporal expand requires positive num_context_frames")
    if num_target_latent_frames <= 0:
        raise ValueError("hybrid temporal expand requires positive num_target_latent_frames")
    if num_target_video_frames <= 0:
        raise ValueError("hybrid temporal expand requires positive num_target_video_frames")
    if feat.shape[2] < num_context_frames:
        raise ValueError(
            "hybrid temporal expand received too few context frames: "
            f"need {num_context_frames}, got {feat.shape[2]}"
        )

    target_end = num_context_frames + num_target_video_frames
    if feat.shape[2] < target_end:
        raise ValueError(
            "hybrid temporal expand received too few frames: "
            f"need at least {target_end}, got {feat.shape[2]}"
        )

    context = feat[:, :, :num_context_frames]
    target = feat[:, :, num_context_frames:target_end]

    parts: list[torch.Tensor] = []
    for i in range(num_context_frames):
        parts.append(context[:, :, i : i + 1].repeat(1, 1, 4, 1, 1))

    parts.append(target[:, :, :1].repeat(1, 1, 4, 1, 1))
    if num_target_latent_frames > 1:
        required_rest = 4 * (num_target_latent_frames - 1)
        rest = target[:, :, 1:]
        if rest.shape[2] <= 0:
            rest = target[:, :, :1]
        if rest.shape[2] < required_rest:
            rest = torch.cat(
                [rest, rest[:, :, -1:].expand(-1, -1, required_rest - rest.shape[2], -1, -1)],
                dim=2,
            )
        parts.append(rest[:, :, :required_rest])

    expanded = torch.cat(parts, dim=2)
    expected_len = 4 * (num_context_frames + num_target_latent_frames)
    if expanded.shape[2] != expected_len:
        raise ValueError(
            f"hybrid temporal expand length {expanded.shape[2]} != expected {expected_len}"
        )
    return expanded


def pack_hybrid_condition_map(
    feat: torch.Tensor,
    num_context_frames: int,
    num_target_latent_frames: int,
    num_target_video_frames: int,
    group: int = 4,
) -> torch.Tensor:
    """VAE-style temporal pack for hybrid raymaps.

    Accepts ``(T, C, H, W)`` or ``(B, C, T, H, W)``. Returns the same layout with
    ``T' = Nc + Nt_lat`` and ``C' = C * group`` (no spatial unshuffle).
    VGGT geom is not packed; callers align it separately to latent length.
    """
    squeeze_batch = False
    if feat.dim() == 4:
        feat = feat.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
        squeeze_batch = True
    elif feat.dim() != 5:
        raise ValueError(f"Expected (T,C,H,W) or (B,C,T,H,W), got {tuple(feat.shape)}")

    expanded = _expand_hybrid_vae_temporal_frames(
        feat,
        num_context_frames=num_context_frames,
        num_target_latent_frames=num_target_latent_frames,
        num_target_video_frames=num_target_video_frames,
    )
    packed = _channel_pack_raymap_frames(expanded, group=group)
    expected_slots = num_context_frames + num_target_latent_frames
    if packed.shape[2] != expected_slots:
        raise ValueError(
            f"packed hybrid condition produced {packed.shape[2]} slots, expected {expected_slots}"
        )
    if squeeze_batch:
        return packed[0].permute(1, 0, 2, 3).contiguous()  # (T', C', H, W)
    return packed


def _build_context_raymap_condition(
    raymap: torch.Tensor,
    num_context_frames: int,
    num_target_latent_frames: int,
    num_target_video_frames: int | None = None,
    downscale_factor: int = 8,
) -> torch.Tensor:
    """Hybrid raymap via VAE-style temporal pack + PixelUnshuffle.

    Full-res Plücker ``(B, 6, Nc+Nt, H, W)`` is expanded / channel-packed to
    ``(B, 24, Nc+Nt_lat, H, W)`` then PixelUnshuffled to
    ``(B, 1536, Nc+Nt_lat, H/8, W/8)``.
    """
    if num_target_video_frames is None or num_target_video_frames <= 0:
        raise ValueError("hybrid packed raymap requires positive num_target_video_frames")
    raymap = _as_bcfhw_raymap(raymap)
    packed = pack_hybrid_condition_map(
        raymap,
        num_context_frames=num_context_frames,
        num_target_latent_frames=num_target_latent_frames,
        num_target_video_frames=num_target_video_frames,
        group=4,
    )
    return _spatial_pixel_unshuffle_raymap(packed, downscale_factor=downscale_factor)


class WanVideoPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.in_iteration_models = ("dit",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_InputVideoEmbedderMultiple(),
            WanVideoUnit_ImageEmbedderVAE_IndividualEncoding(),
            WanVideoUnit_TargetRaymapInput(),
            WanVideoUnit_ImageEmbedderCLIP(),
        ]
        self.post_units = []
        self.model_fn = model_fn_wan_video


    def enable_usp(self):
        from ..utils.xfuser import get_sequence_parallel_world_size, usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        self.sp_size = get_sequence_parallel_world_size()


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
                "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
                "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]
        
        if use_usp:
            from ..utils.xfuser import initialize_usp
            initialize_usp(device)
            import torch.distributed as dist
            from ..core.device.npu_compatible_device import get_device_name
            if dist.is_available() and dist.is_initialized():
                device = get_device_name()

        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        pipe.dit = model_pool.fetch_model("wan_video_dit")
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')
        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    def _prompt_embedder_unit(self) -> "WanVideoUnit_PromptEmbedder":
        for unit in self.units:
            if isinstance(unit, WanVideoUnit_PromptEmbedder):
                return unit
        raise RuntimeError("WanVideoPipeline is missing WanVideoUnit_PromptEmbedder")

    @torch.no_grad()
    def prebake_and_unload_text_encoder(self, prompts: tuple[str, ...] = ("",)) -> None:
        """Encode ``prompts`` once, keep CPU copies, then free UMT5 weights.

        RoGe training/vis uses a constant empty prompt, so the 4.7B
        text encoder only needs to run at init. Call this after
        ``from_pretrained`` (ideally while still on CPU) and before
        ``accelerator.prepare``.
        """
        import gc

        if self.text_encoder is None:
            raise RuntimeError(
                "prebake_and_unload_text_encoder requires a loaded text_encoder"
            )
        if self.tokenizer is None:
            raise RuntimeError(
                "prebake_and_unload_text_encoder requires a loaded tokenizer"
            )
        unit = self._prompt_embedder_unit()
        self.load_models_to_device(("text_encoder",))
        for prompt in prompts:
            emb = unit.encode_prompt(self, prompt)
            unit.prebake_prompt(prompt, emb)
            print(
                f"Prebaked text embedding for prompt={prompt!r}: "
                f"shape={tuple(emb.shape)} dtype={emb.dtype}"
            )
        self.text_encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Unloaded text_encoder after prebake (embeddings kept on CPU).")


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Reference
        vace_reference_image: Optional[Image.Image] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # progress_bar
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
        raymap: Optional[torch.Tensor] = None,
        vggt_geom: Optional[torch.Tensor] = None,
        num_latent_frames: Optional[int] = None,
        num_output_frames: Optional[int] = None,
        num_context_frames: Optional[int] = None,
        num_target_video_frames: Optional[int] = None,
        hybrid_vae_encoding: bool = False,
        decode_target_only: bool = False,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "vace_reference_image": vace_reference_image,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "sigma_shift": sigma_shift,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "raymap": raymap,
            "vggt_geom": vggt_geom,
            "num_latent_frames": num_latent_frames,
            "num_output_frames": num_output_frames,
            "num_context_frames": num_context_frames,
            "num_target_video_frames": num_target_video_frames,
            "hybrid_vae_encoding": hybrid_vae_encoding,
            "decode_target_only": decode_target_only,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
        
        # Strip reference frames if present
        if vace_reference_image is not None:
            if isinstance(vace_reference_image, list):
                f = len(vace_reference_image)
            else:
                f = 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]

        # Decode
        self.load_models_to_device(['vae'])
        latents_to_decode = inputs_shared["latents"]
        if decode_target_only:
            num_output = _as_optional_int(inputs_shared.get("num_output_frames"))
            if num_output is None or num_output <= 0:
                raise ValueError("decode_target_only requires a positive num_output_frames")
            latents_to_decode = latents_to_decode[:, :, -num_output:]
        if self.dit.individual_encoding and not hybrid_vae_encoding:
            video = []
            for i in range(latents_to_decode.shape[2]):
                video.append(self.vae.decode(latents_to_decode[:, :, i:i+1], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride))
            video = torch.concat(video, dim=2)
        else:
            video = self.vae.decode(latents_to_decode, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if decode_target_only:
            num_target_video = _as_optional_int(inputs_shared.get("num_target_video_frames"))
            if num_target_video is not None and num_target_video > 0:
                video = video[:, :, :num_target_video]
        
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image", "num_latent_frames"),
            output_params=("noise",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image, num_latent_frames=None):
        if num_latent_frames is not None:
            length = num_latent_frames
        else:
            length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )
        # The text encoder is frozen, so its output for a given prompt never
        # changes. RoGe training uses a constant empty prompt on every
        # step, which would otherwise re-run the 4.7B UMT5-xxl over 512 padded
        # tokens each step. Cache the embedding keyed by (prompt, device, dtype).
        self._prompt_cache = {}
        # Device-agnostic prebaked embeddings (CPU). Survives text_encoder unload
        # and pipe CPU→GPU moves; copied to pipe.device on each use.
        self._prebaked_prompts: dict[str, torch.Tensor] = {}

    def prebake_prompt(self, prompt: str, emb: torch.Tensor) -> None:
        self._prebaked_prompts[prompt] = emb.detach().to(device="cpu").contiguous()

    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        if pipe.text_encoder is None:
            raise RuntimeError(
                "text_encoder is None; cannot encode prompt. "
                "Prebake embeddings before unload, or keep the text encoder loaded."
            )
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        cache_key = (prompt, str(pipe.device), pipe.torch_dtype)
        cached = self._prompt_cache.get(cache_key)
        if cached is not None:
            return {"context": cached}
        if prompt in self._prebaked_prompts:
            # Promote CPU prebake to device once; reuse via _prompt_cache.
            emb = self._prebaked_prompts[prompt].to(
                device=pipe.device, dtype=pipe.torch_dtype
            )
            self._prompt_cache[cache_key] = emb
            return {"context": emb}
        if pipe.text_encoder is None:
            raise RuntimeError(
                f"No prebaked embedding for prompt={prompt!r} and text_encoder "
                "was unloaded. Pass --unload_text_encoder only for prompts that "
                "were prebaked (training uses empty string)."
            )
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        self._prompt_cache[cache_key] = prompt_emb
        return {"context": prompt_emb}



class WanVideoUnit_InputVideoEmbedderMultiple(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_video", "noise", "tiled", "tile_size", "tile_stride",
                "num_context_frames", "num_target_video_frames", "num_output_frames",
                "hybrid_vae_encoding",
            ),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_video,
        noise,
        tiled,
        tile_size,
        tile_stride,
        num_context_frames=None,
        num_target_video_frames=None,
        num_output_frames=None,
        hybrid_vae_encoding=False,
    ):
        if input_video is None:
            return {"latents": noise}
        if not pipe.dit.individual_encoding and not hybrid_vae_encoding:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)

        with torch.no_grad():
            if hybrid_vae_encoding:
                num_context = _as_optional_int(num_context_frames)
                num_target_video = _as_optional_int(num_target_video_frames)
                num_target_latents = _as_optional_int(num_output_frames)
                if num_context is None or num_context <= 0:
                    raise ValueError("hybrid_vae_encoding requires positive num_context_frames")
                if num_target_video is None or num_target_video <= 0:
                    raise ValueError("hybrid_vae_encoding requires positive num_target_video_frames")
                if num_target_latents is None or num_target_latents <= 0:
                    raise ValueError("hybrid_vae_encoding requires positive num_output_frames")
                if input_video.shape[2] < num_context + num_target_video:
                    raise ValueError(
                        "hybrid_vae_encoding received too few input frames: "
                        f"need {num_context + num_target_video}, got {input_video.shape[2]}"
                    )

                context_latents = []
                for i in range(num_context):
                    context_latents.append(
                        pipe.vae.encode(
                            input_video[:, :, i:i + 1],
                            device=pipe.device,
                            tiled=tiled,
                            tile_size=tile_size,
                            tile_stride=tile_stride,
                        ).to(dtype=pipe.torch_dtype, device=pipe.device)
                    )
                context_latents = torch.concat(context_latents, dim=2)
                target_video = input_video[:, :, num_context:num_context + num_target_video]
                target_latents = _vae_encode_video(pipe, target_video, tiled, tile_size, tile_stride)
                if target_latents.shape[2] != num_target_latents:
                    raise ValueError(
                        f"hybrid_vae_encoding produced {target_latents.shape[2]} target latent frames, "
                        f"expected {num_target_latents}"
                    )
                input_latents = torch.cat([context_latents, target_latents], dim=2)
            else:
                input_latents = []
                for i in range(input_video.shape[2]):
                    input_latents.append(
                        pipe.vae.encode(
                            input_video[:, :, i:i + 1],
                            device=pipe.device,
                            tiled=tiled,
                            tile_size=tile_size,
                            tile_stride=tile_stride,
                        ).to(dtype=pipe.torch_dtype, device=pipe.device)
                    )
                input_latents = torch.concat(input_latents, dim=2)

        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents, "mask_loss": True}
        latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
        # Keep input_latents so ImageEmbedderVAE can reuse context encodes (hybrid path).
        return {"latents": latents, "input_latents": input_latents}



class WanVideoUnit_ImageEmbedderVAE_IndividualEncoding(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_image", "end_image", "num_frames", "height", "width",
                "tiled", "tile_size", "tile_stride", "num_latent_frames",
                "num_context_frames", "num_output_frames", "hybrid_vae_encoding",
                "input_latents",
            ),
            output_params=("y",),
            onload_model_names=("vae",)
        )

    @staticmethod
    def _build_context_mask(num_input, num_total, height, width, device):
        msk = torch.ones(1, num_total, height // 8, width // 8, device=device)
        msk[:, num_input:] = 0
        msk = torch.repeat_interleave(msk, repeats=4, dim=1)
        msk = msk.view(1, num_total, 4, height // 8, width // 8)
        return msk.transpose(1, 2)[0]  # [4, num_total, h//8, w//8]

    def _y_from_context_latents(self, context_latents, num_output, height, width):
        """Build I2V ``y`` = [mask | context_latents | target_zeros] without re-encoding."""
        num_input = context_latents.shape[2]
        num_total = num_input + num_output
        msk = self._build_context_mask(
            num_input, num_total, height, width, context_latents.device
        ).to(dtype=context_latents.dtype)
        if num_output > 0:
            target_zeros = torch.zeros(
                context_latents.shape[0],
                context_latents.shape[1],
                num_output,
                context_latents.shape[3],
                context_latents.shape[4],
                device=context_latents.device,
                dtype=context_latents.dtype,
            )
            latents_y = torch.cat([context_latents, target_zeros], dim=2)
        else:
            latents_y = context_latents
        msk_b = msk.unsqueeze(0).expand(latents_y.shape[0], -1, -1, -1, -1)
        y = torch.concat([msk_b, latents_y], dim=1)
        return {"y": y}

    def _process_individual_context(
        self,
        pipe,
        input_image,
        end_image,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        num_total,
        input_latents=None,
    ):
        if end_image is not None:
            raise ValueError("end_image is not supported for hybrid_vae_encoding")

        num_input = len(input_image)
        if num_total < num_input:
            raise ValueError(f"num_total={num_total} is smaller than num_input={num_input}")
        num_output = num_total - num_input

        # Reuse context VAE latents already computed by InputVideoEmbedderMultiple.
        if input_latents is not None:
            if input_latents.ndim != 5:
                raise ValueError(
                    f"hybrid_vae_encoding expected input_latents [B,C,T,H,W], got shape {tuple(input_latents.shape)}"
                )
            if input_latents.shape[2] < num_input:
                raise ValueError(
                    "hybrid_vae_encoding input_latents has fewer temporal frames than context: "
                    f"got {input_latents.shape[2]}, need >= {num_input}"
                )
            context_latents = input_latents[:, :, :num_input].to(
                dtype=pipe.torch_dtype, device=pipe.device
            )
            return self._y_from_context_latents(context_latents, num_output, height, width)

        pipe.load_models_to_device(self.onload_model_names)
        image = []
        for i in range(len(input_image)):
            image.append(pipe.preprocess_image(input_image[i].resize((width, height))).to(pipe.device))
        image = torch.concat(image, dim=0)

        msk = self._build_context_mask(num_input, num_total, height, width, pipe.device)

        target_zeros = torch.zeros(3, num_output, height, width, device=image.device, dtype=image.dtype)
        vae_input = torch.concat([image.transpose(0, 1), target_zeros], dim=1)

        ys = []
        with torch.no_grad():
            for i in range(num_input):
                ys.append(
                    pipe.vae.encode(
                        [vae_input[:, i:i + 1].to(dtype=pipe.torch_dtype, device=pipe.device)],
                        device=pipe.device,
                        tiled=tiled,
                        tile_size=tile_size,
                        tile_stride=tile_stride,
                    )[0]
                )
        if num_output > 0:
            target_latents = torch.zeros(
                ys[0].shape[0], num_output, ys[0].shape[2], ys[0].shape[3],
                device=ys[0].device, dtype=ys[0].dtype,
            )
            ys.append(target_latents)
        y = torch.concat(ys, dim=1).to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        return {"y": y.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)}

    def process(
        self,
        pipe: WanVideoPipeline,
        input_image,
        end_image,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        num_latent_frames=None,
        num_context_frames=None,
        num_output_frames=None,
        hybrid_vae_encoding=False,
        input_latents=None,
    ):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        if not isinstance(input_image, list):
            input_image = [input_image]
        if hybrid_vae_encoding:
            num_context = _as_optional_int(num_context_frames) or len(input_image)
            num_target_latents = _as_optional_int(num_output_frames)
            if num_context != len(input_image):
                raise ValueError(
                    "hybrid_vae_encoding expects num_context_frames to match input_image length, "
                    f"got {num_context} and {len(input_image)}"
                )
            if num_target_latents is None or num_target_latents <= 0:
                raise ValueError("hybrid_vae_encoding requires positive num_output_frames")
            return self._process_individual_context(
                pipe, input_image, end_image, height, width, tiled, tile_size, tile_stride,
                num_context + num_target_latents,
                input_latents=input_latents,
            )
        if not pipe.dit.individual_encoding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = []
        for i in range(len(input_image)):
            image.append(pipe.preprocess_image(input_image[i].resize((width, height))).to(pipe.device))
        image = torch.concat(image, dim=0)
        
        num_input = len(input_image)
        num_total = num_latent_frames if num_latent_frames is not None else (num_input + 1)
        num_output = num_total - num_input
        
        msk = torch.ones(1, num_total, height//8, width//8, device=pipe.device)
        msk[:, num_input:] = 0
        
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            target_zeros = torch.zeros(3, num_output, height, width).to(image.device)
            vae_input = torch.concat([image.transpose(0, 1), target_zeros], dim=1)

        msk = torch.repeat_interleave(msk, repeats=4, dim=1)
        msk = msk.view(1, num_total, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]  # [4, num_total, h//8, w//8]
        
        ys = []
        for i in range(num_total):
            ys.append(pipe.vae.encode([vae_input[:, i:i+1].to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0])
        y = torch.concat(ys, dim=1)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}



class WanVideoUnit_TargetRaymapInput(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "raymap", "hybrid_vae_encoding", "num_context_frames",
                "num_target_video_frames", "num_output_frames",
            ),
            output_params=("raymap",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        raymap,
        hybrid_vae_encoding=False,
        num_context_frames=None,
        num_target_video_frames=None,
        num_output_frames=None,
    ):
        if raymap is None or not hybrid_vae_encoding:
            return {}
        num_context = _as_optional_int(num_context_frames)
        num_target_video = _as_optional_int(num_target_video_frames)
        num_target_latents = _as_optional_int(num_output_frames)
        if num_context is None or num_context <= 0:
            raise ValueError("hybrid raymap requires positive num_context_frames")
        if num_target_video is None or num_target_video <= 0:
            raise ValueError("hybrid raymap requires positive num_target_video_frames")
        if num_target_latents is None or num_target_latents <= 0:
            raise ValueError("hybrid raymap requires positive num_output_frames")
        raymap = raymap.to(device=pipe.device, dtype=pipe.torch_dtype)
        raymap_condition = _build_context_raymap_condition(
            raymap,
            num_context,
            num_target_latents,
            num_target_video_frames=num_target_video,
        )
        return {"raymap": raymap_condition}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "height", "width"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        if pipe.dit.individual_encoding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}



def model_fn_wan_video(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    raymap: Optional[torch.Tensor] = None,
    vggt_geom: Optional[torch.Tensor] = None,
    num_output_frames: Optional[int] = None,
    hybrid_vae_encoding: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    # Timestep
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)

    x = latents
    if getattr(dit, "modality_tokens_enabled", False) and hasattr(dit, "modality_token_x"):
        x = _maybe_apply_modality_token(x, dit.modality_token_x)
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        if getattr(dit, "modality_tokens_enabled", False) and hasattr(dit, "modality_token_y"):
            y = _maybe_apply_modality_token(y, dit.modality_token_y)
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    use_raymap_pe = (
        getattr(dit, "use_raymap_pe", False)
        and getattr(dit, "raymap_pe", None) is not None
    )
    use_geom_pe = (
        getattr(dit, "use_geom_pe", False)
        and getattr(dit, "geom_pe", None) is not None
    )
    if raymap is not None:
        if raymap.dim() == 4:
            raymap = raymap.permute(1, 0, 2, 3).unsqueeze(0)
        if getattr(dit, "modality_tokens_enabled", False) and hasattr(dit, "modality_token_raymap"):
            raymap = _maybe_apply_modality_token(raymap, dit.modality_token_raymap)
        if not use_raymap_pe:
            x = torch.cat([x, raymap], dim=1)
    if vggt_geom is not None:
        if vggt_geom.dim() == 4:
            vggt_geom = vggt_geom.permute(1, 0, 2, 3).unsqueeze(0)
        if getattr(dit, "modality_tokens_enabled", False):
            if not hasattr(dit, "modality_token_vggt"):
                # Inference-only fallback; for training this should be created up-front.
                dit.modality_token_vggt = nn.Parameter(torch.zeros(int(vggt_geom.shape[1]), device=vggt_geom.device))
            vggt_geom = _maybe_apply_modality_token(vggt_geom, dit.modality_token_vggt)
        if not use_geom_pe:
            x = torch.cat([x, vggt_geom], dim=1)

    # Patchify
    x = dit.patchify(x)
    if use_raymap_pe and raymap is not None:
        y_camera = dit.raymap_pe(raymap)
        if y_camera.shape[0] != x.shape[0]:
            y_camera = torch.concat([y_camera] * x.shape[0], dim=0)
        if y_camera.shape != x.shape:
            raise ValueError(
                f"raymap_pe output shape {tuple(y_camera.shape)} must match "
                f"patch_embedding output {tuple(x.shape)}"
            )
        x = x + y_camera
    if use_geom_pe and vggt_geom is not None:
        y_geom = dit.geom_pe(vggt_geom)
        if y_geom.shape[0] != x.shape[0]:
            y_geom = torch.concat([y_geom] * x.shape[0], dim=0)
        if y_geom.shape != x.shape:
            raise ValueError(
                f"geom_pe output shape {tuple(y_geom.shape)} must match "
                f"patch_embedding output {tuple(x.shape)}"
            )
        x = x + y_geom

    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    freqs = _build_split_rope_freqs(
        dit, f, h, w, num_output_frames,
        disable_context_temporal_rope=hybrid_vae_encoding,
    ).to(x.device)

    for block in dit.blocks:
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                block,
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x
