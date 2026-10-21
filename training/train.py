"""LoRA fine-tuning entry point for RoGe.

Trains a Wan2.1-I2V-14B backbone for permutation-invariant novel view
synthesis: each frame is encoded independently, the Plucker raymap is
channel-concatenated into the DiT input, and the flow-matching SFT loss is
applied to the last N target frames of an M+N latent sequence.
"""

import argparse
import json
import os
import sys
from datetime import timedelta

# Ensure repo root is importable when launched as ``training/train.py``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import accelerate
import torch
import torch.nn as nn

from diffsynth.core.data.arkitscenes_dataset import ARKitScenesRoGeDataset
from diffsynth.core.data.blendedmvs_dataset import BlendedMVSRoGeDataset
from diffsynth.core.data.mvs_synth_dataset import MVSSynthRoGeDataset
from diffsynth.core.data.wildrgbd_dataset import WildRGBDRoGeDataset
from diffsynth.core.data.dl3dv_dataset import DL3DVRoGeDataset
from diffsynth.core.data.re10k_dataset import RE10KRoGeDataset
from diffsynth.core.data.dataset import normalize_num_dataset_samples
from diffsynth.core.data.composed_dataset import MixedRoGeDataset
from diffsynth.diffusion import (
    DiffusionTrainingModule,
    FlowMatchSFTLoss,
    ModelLogger,
    add_general_config,
    add_video_size_config,
    launch_training_task,
)
from diffsynth.pipelines.wan_video import (
    ModelConfig,
    WanVideoPipeline,
    load_modality_tokens_from_checkpoint,
    enable_geom_pe,
    enable_raymap_pe,
    load_geom_pe_from_checkpoint,
    load_raymap_pe_from_checkpoint,
    wan_native_num_latent_frames,
)
from diffsynth.vggt_condition import (
    base_in_dim,
    load_patch_embedding_from_checkpoint,
    raymap_channels,
)
from diffsynth.lagernvs_condition import (
    LAGERNVS_CHANNELS,
    LAGERNVS_LORA_RANK,
    LagerNVSConditioningModule,
    lagernvs_in_dim,
    lagernvs_packed_channels,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _is_wan_clip_model_config(config: ModelConfig) -> bool:
    """True for Wan OpenCLIP image-encoder weight configs."""
    blob = f"{config.origin_file_pattern or ''} {config.path or ''}".lower()
    return (
        "models_clip_open-clip" in blob
        or "wan_video_image_encoder" in blob
    )


def _filter_skip_clip_configs(model_configs: list) -> list:
    kept = []
    for config in model_configs:
        if _is_wan_clip_model_config(config):
            label = config.origin_file_pattern or config.path or repr(config)
            print(f"skip_clip: omitting CLIP config {label}")
            continue
        kept.append(config)
    return kept


def resolve_resume_ckpt_path(resume_ckpt_path, resume_checkpoint):
    """``--resume_ckpt_path`` takes precedence over legacy ``--resume_checkpoint``."""
    if resume_ckpt_path is not None:
        return resume_ckpt_path
    return resume_checkpoint


def normalize_pretrained_state_dict(state_dict):
    """Unwrap common checkpoint containers (.pt training ckpts, etc.)."""
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state dict, got {type(state_dict)}")
    tensor_keys = [k for k, v in state_dict.items() if isinstance(v, torch.Tensor)]
    if any(
        "lora_A" in k or "lora_B" in k or "patch_embedding" in k or "vggt_condition" in k
        or "lagernvs_condition" in k
        for k in tensor_keys
    ):
        return state_dict
    for wrapper in ("model", "state_dict", "module", "model_state"):
        inner = state_dict.get(wrapper)
        if isinstance(inner, dict) and inner:
            return normalize_pretrained_state_dict(inner)
    return state_dict


class WanTrainingModule(DiffusionTrainingModule):
    """Single-DiT LoRA SFT module for RoGe."""

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        device="cpu",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        modify_channels=False,
        new_in_dim=None,
        individual_encoding=False,
        hybrid_vae_encoding=False,
        resume_ckpt_path=None,
        resume_checkpoint=None,
        modality_tokens: bool = False,
        raymap_pe: bool = False,
        raymap_pe_zero_init: bool = False,
        geom_pe: bool = False,
        geom_pe_zero_init: bool = False,
        from_wan: bool = False,
        lagernvs_ckpt: str | None = None,
        lagernvs_offload_per_step: bool = False,
        lagernvs_channels: int = LAGERNVS_CHANNELS,
        joint_train: bool = False,
        nvs_loss_weight: float = 0.1,
        lagernvs_lora_rank: int = LAGERNVS_LORA_RANK,
        train_reconstructor_lora: bool = False,
        load_frozen_lagernvs_lora: bool = False,
        skip_clip: bool = False,
        unload_text_encoder: bool = False,
    ):
        super().__init__()

        # ---- pipeline -----------------------------------------------------
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, device=device)
        if skip_clip:
            model_configs = _filter_skip_clip_configs(model_configs)
        tokenizer_config = (
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
            if tokenizer_path is None
            else ModelConfig(path=tokenizer_path)
        )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )

        if skip_clip:
            self.pipe.image_encoder = None
            if self.pipe.dit is not None:
                self.pipe.dit.require_clip_embedding = False
            print("skip_clip: image_encoder=None, dit.require_clip_embedding=False")

        if unload_text_encoder:
            # Bake empty prompt (train + vis both use "") then free UMT5 VRAM.
            self.pipe.prebake_and_unload_text_encoder(prompts=("",))

        self.individual_encoding = individual_encoding or hybrid_vae_encoding or modify_channels
        if hybrid_vae_encoding and not individual_encoding and not modify_channels:
            print("hybrid_vae_encoding: enabling individual_encoding for context frames")
        if self.individual_encoding:
            self.pipe.dit.individual_encoding = True
        self.hybrid_vae_encoding = hybrid_vae_encoding
        self.raymap_pe = bool(raymap_pe)
        self.raymap_pe_zero_init = bool(raymap_pe_zero_init)
        self.geom_pe = bool(geom_pe)
        self.geom_pe_zero_init = bool(geom_pe_zero_init)
        self.from_wan = bool(from_wan)
        if not hybrid_vae_encoding:
            raise ValueError("RoGe training requires --hybrid_vae_encoding")
        if lagernvs_ckpt is None:
            raise ValueError("--lagernvs_ckpt is required")
        if self.raymap_pe and not hybrid_vae_encoding:
            raise ValueError("--raymap_pe requires --hybrid_vae_encoding")
        if self.from_wan:
            if not (self.raymap_pe and self.geom_pe):
                raise ValueError(
                    "--from_wan requires both --raymap_pe and --geom_pe "
                    "(keeps Wan I2V in_dim=36; raymap/geom via residual pe)"
                )
            if resolve_resume_ckpt_path(resume_ckpt_path, resume_checkpoint) is not None:
                raise ValueError(
                    "--from_wan cannot be combined with --resume_ckpt_path/--resume_checkpoint "
                    "(Wan base weights come from model_id / model_paths only)"
                )
            if lora_checkpoint is not None:
                raise ValueError(
                    "--from_wan cannot be combined with --lora_checkpoint "
                    "(LoRA starts from scratch on Wan)"
                )
            print(
                "--from_wan: start from Wan-I2V-14B-480P (full base incl. 36ch "
                "patch_embedding); skip RoGe resume; train LoRA + "
                "patch_embedding + raymap_pe + geom_pe (+ geom projector)"
            )
        if hybrid_vae_encoding:
            self.pipe.dit.hybrid_vae_encoding = True
            self.pipe.dit.use_temporal_rope = True
            self.pipe.dit.pack_raymap = True
            print(
                "Enabled hybrid VAE encoding: context per-frame + target continuous, "
                "split temporal RoPE, VAE-style raymap pack "
                f"(raymap ch={raymap_channels()}, "
                f"in_dim={base_in_dim(raymap_pe=self.raymap_pe)}"
                f"{', raymap_pe=add' if self.raymap_pe else ', raymap=concat'}"
                f"{', geom_pe=add' if self.geom_pe else ''})"
            )

        self.lagernvs_channels = lagernvs_channels
        self.joint_train = bool(joint_train)
        self.nvs_loss_weight = float(nvs_loss_weight)
        self.lagernvs_lora_rank = int(lagernvs_lora_rank)
        self.train_reconstructor_lora = bool(train_reconstructor_lora)
        self.load_frozen_lagernvs_lora = bool(load_frozen_lagernvs_lora)
        self._pending_nvs_loss = None
        # Detached scalars for TensorBoard / wandb (set each forward).
        self.last_fm_loss = None
        self.last_nvs_loss = None
        self.last_nvs_loss_weighted = None
        if self.train_reconstructor_lora and not self.joint_train:
            raise ValueError("--train_reconstructor_lora requires --joint_train")
        if self.load_frozen_lagernvs_lora and self.joint_train:
            raise ValueError(
                "--load_frozen_lagernvs_lora and --joint_train are mutually exclusive"
            )
        if self.load_frozen_lagernvs_lora and resolve_resume_ckpt_path(
            resume_ckpt_path, resume_checkpoint
        ) is None:
            raise ValueError(
                "--load_frozen_lagernvs_lora requires "
                "--resume_ckpt_path/--resume_checkpoint"
            )
        if self.joint_train and lagernvs_offload_per_step:
            print(
                "Warning: --lagernvs_offload_per_step is ignored under --joint_train"
            )
        expected_in_dim = lagernvs_in_dim(
            lagernvs_channels, raymap_pe=self.raymap_pe, geom_pe=self.geom_pe
        )
        if new_in_dim is not None and new_in_dim != expected_in_dim:
            print(
                f"Warning: --new_in_dim={new_in_dim} differs from "
                f"expected {expected_in_dim}; using {expected_in_dim}"
            )
        new_in_dim = expected_in_dim
        modify_channels = True

        self.lagernvs_condition = LagerNVSConditioningModule(
            lagernvs_ckpt=lagernvs_ckpt,
            offload_per_step=lagernvs_offload_per_step and not self.joint_train,
            geom_channels=lagernvs_channels,
            device=device,
            joint_train=self.joint_train,
            train_reconstructor_lora=self.train_reconstructor_lora,
            load_frozen_lora=self.load_frozen_lagernvs_lora,
            lora_rank=self.lagernvs_lora_rank,
        )
        for param in self.lagernvs_condition.projector.parameters():
            param.requires_grad = True
        if self.joint_train:
            sides = (
                "renderer+reconstructor LoRA"
                if self.train_reconstructor_lora
                else "renderer LoRA"
            )
            print(
                f"LagerNVS --joint_train: {sides} + NVS aux loss "
                f"(weight={self.nvs_loss_weight}, "
                f"lora_rank={self.lagernvs_lora_rank})"
            )
        elif self.load_frozen_lagernvs_lora:
            print(
                "LagerNVS frozen LoRA: loading renderer+reconstructor "
                f"adapters at rank={self.lagernvs_lora_rank}; no NVS aux loss"
            )

        if self.from_wan and new_in_dim is not None:
            wan_in_dim = int(self.pipe.dit.in_dim)
            if int(new_in_dim) != wan_in_dim:
                raise ValueError(
                    f"--from_wan expects patch_embedding in_dim={wan_in_dim} "
                    f"(Wan I2V), got expected_in_dim={new_in_dim}; "
                    "use --raymap_pe --geom_pe so raymap/geom leave concat"
                )

        if modify_channels and new_in_dim is not None:
            self._modify_model_channels(self.pipe.dit, new_in_dim, device)

        if self.raymap_pe:
            enable_raymap_pe(
                self.pipe.dit,
                raymap_channels=raymap_channels(),
                zero_init=self.raymap_pe_zero_init,
            )
        if self.geom_pe:
            geom_pe_channels = lagernvs_packed_channels(self.lagernvs_channels)
            enable_geom_pe(
                self.pipe.dit,
                geom_channels=geom_pe_channels,
                zero_init=self.geom_pe_zero_init,
            )

        # ---- LoRA + freezing ---------------------------------------------
        # --from_wan: never warm-start from RoGe / LoRA ckpts.
        effective_resume_ckpt = (
            None
            if self.from_wan
            else resolve_resume_ckpt_path(resume_ckpt_path, resume_checkpoint)
        )
        effective_lora_checkpoint = (
            None
            if self.from_wan
            else (
                effective_resume_ckpt
                if effective_resume_ckpt is not None
                else lora_checkpoint
            )
        )
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models=trainable_models,
            lora_base_model=lora_base_model,
            lora_target_modules=lora_target_modules,
            lora_rank=lora_rank,
            lora_checkpoint=effective_lora_checkpoint,
        )

        # Optional: channel-wise modality tokens for x / raymap / vggt_geom.
        self.modality_tokens = modality_tokens
        if self.modality_tokens:
            from diffsynth.pipelines.wan_video import enable_modality_tokens
            enable_modality_tokens(
                self.pipe.dit,
                vggt_channels=lagernvs_packed_channels(self.lagernvs_channels),
                raymap_channels=raymap_channels(),
            )
            # Ensure tokens remain trainable (switch_pipe_to_training_mode may freeze modules).
            for name in ("modality_token_x", "modality_token_y", "modality_token_raymap", "modality_token_vggt"):
                p = getattr(self.pipe.dit, name, None)
                if isinstance(p, torch.nn.Parameter):
                    p.requires_grad = True

        # Keep patch_embedding trainable (rebuilt, or Wan 36ch kept under --from_wan).
        # LoRA only touches q/k/v/o/ffn; pe / residual pe are trained fully.
        if (
            (modify_channels and new_in_dim is not None) or self.from_wan
        ) and lora_base_model is not None:
            self._unfreeze_patch_embedding(self.pipe, lora_base_model)
        if self.raymap_pe and lora_base_model is not None:
            self._unfreeze_raymap_pe(self.pipe, lora_base_model)
        if self.geom_pe and lora_base_model is not None:
            self._unfreeze_geom_pe(self.pipe, lora_base_model)

        if effective_resume_ckpt is not None:
            self._load_pretrained_checkpoint(effective_resume_ckpt, lora_base_model)

        # ---- training-time misc ------------------------------------------
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def checkpoint_param_names(self):
        names = super().checkpoint_param_names()
        if self.load_frozen_lagernvs_lora:
            names.update(
                name
                for name, _ in self.named_parameters()
                if name.startswith("lagernvs_condition.backbone.")
                and "lora_" in name
            )
        return names

    # ----------------------------------------------------------------------
    # Channel surgery
    # ----------------------------------------------------------------------

    def _modify_model_channels(self, model, new_in_dim, device):
        """Replace ``patch_embedding`` so the DiT accepts ``new_in_dim``
        channels (16 latent + 4 mask + 4*100 vae context + raymap, etc.)
        instead of the stock 36. Weights of the new Conv3d are random;
        they are unfrozen for training below.

        When ``new_in_dim`` already matches (e.g. ``--from_wan`` with
        ``--raymap_pe --geom_pe`` → 36), keep the loaded Wan patch_embedding.
        """
        if model is None:
            return

        old_in_dim = int(model.in_dim)
        new_in_dim = int(new_in_dim)
        # Encode every frame independently: one temporal latent per video frame.
        model.individual_encoding = True
        if old_in_dim == new_in_dim:
            print(
                f"Keeping DiT patch_embedding in_dim={old_in_dim} "
                f"(no rebuild; pretrained weights preserved)"
            )
            return

        print(f"Modifying DiT input channels: {old_in_dim} -> {new_in_dim}")

        old_pe = model.patch_embedding
        pe_device = next(old_pe.parameters()).device
        pe_dtype = next(old_pe.parameters()).dtype
        new_pe = nn.Conv3d(
            new_in_dim, model.dim,
            kernel_size=model.patch_size, stride=model.patch_size,
        ).to(device=pe_device, dtype=pe_dtype)
        model.patch_embedding = new_pe
        del old_pe

        model.in_dim = new_in_dim
        print("DiT channels modified successfully")

    def _unfreeze_patch_embedding(self, pipe, lora_base_model):
        model = getattr(pipe, lora_base_model, None)
        if model is None or not hasattr(model, "patch_embedding"):
            return
        for param in model.patch_embedding.parameters():
            param.requires_grad = True
        print(f"Unfroze patch_embedding in {lora_base_model} for full training")

    def _unfreeze_raymap_pe(self, pipe, lora_base_model):
        model = getattr(pipe, lora_base_model, None)
        raymap_pe = getattr(model, "raymap_pe", None) if model is not None else None
        if raymap_pe is None:
            return
        for param in raymap_pe.parameters():
            param.requires_grad = True
        print(f"Unfroze raymap_pe in {lora_base_model} for full training")

    def _unfreeze_geom_pe(self, pipe, lora_base_model):
        model = getattr(pipe, lora_base_model, None)
        geom_pe = getattr(model, "geom_pe", None) if model is not None else None
        if geom_pe is None:
            return
        for param in geom_pe.parameters():
            param.requires_grad = True
        print(f"Unfroze geom_pe in {lora_base_model} for full training")

    def _load_pretrained_checkpoint(self, ckpt_path, lora_base_model):
        """Load patch_embedding / LagerNVS projector weights not covered by LoRA init."""
        from diffsynth.core import load_state_dict

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

        ckpt_sd = normalize_pretrained_state_dict(load_state_dict(ckpt_path))
        print(f"Loading pretrained weights from {ckpt_path} ({len(ckpt_sd)} keys)")

        if lora_base_model is not None:
            dit_model = getattr(self.pipe, lora_base_model, None)
            if dit_model is not None:
                load_patch_embedding_from_checkpoint(dit_model, ckpt_sd, ckpt_path)
                if self.raymap_pe:
                    load_raymap_pe_from_checkpoint(dit_model, ckpt_sd, ckpt_path)
                if self.geom_pe:
                    load_geom_pe_from_checkpoint(dit_model, ckpt_sd, ckpt_path)

        self.lagernvs_condition.load_trainable_state(ckpt_sd)

        if self.modality_tokens and lora_base_model is not None:
            dit_model = getattr(self.pipe, lora_base_model, None)
            if dit_model is not None:
                load_modality_tokens_from_checkpoint(dit_model, ckpt_sd, ckpt_path)

    # ----------------------------------------------------------------------
    # Forward / loss
    # ----------------------------------------------------------------------

    def _parse_extra_inputs(self, data, inputs_shared):
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["input_images"]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def _lagernvs_context_inputs(self, data):
        """Prefer native-resolution context (for --test2 cover-crop); fall back to RoGe."""
        if data.get("target_w2c") is None:
            raise ValueError("LagerNVS conditioning requires target_w2c")
        height = data["input_images"][0].size[1]
        width = data["input_images"][0].size[0]
        if (height, width) == (480, 832):
            # At 480P, LagerNVS must consume a direct 480x832 -> 288x504
            # resize of the exact RoGe cover-crop, with matching K.
            return (
                data["input_images"],
                data["context_intrinsics"],
                data["target_intrinsics"],
            )
        context_images = data.get("lagernvs_context_images", data["input_images"])
        context_intrinsics = data.get(
            "lagernvs_context_intrinsics", data["context_intrinsics"]
        )
        target_intrinsics = data.get(
            "lagernvs_target_intrinsics", data.get("target_intrinsics")
        )
        if target_intrinsics is None:
            raise ValueError(
                "LagerNVS conditioning requires target_intrinsics "
                "(or lagernvs_target_intrinsics)"
            )
        return context_images, context_intrinsics, target_intrinsics

    def _build_lagernvs_geom(
        self,
        data,
        num_output_frames: int,
        *,
        cache_aux_rgb: bool = False,
        compute_nvs_loss: bool = False,
    ):
        height = data["input_images"][0].size[1]
        width = data["input_images"][0].size[0]
        context_images, context_intrinsics, target_intrinsics = (
            self._lagernvs_context_inputs(data)
        )
        # Prefer native target RGB when the dataset provides it; else RoGe frames.
        target_gt_images = None
        if compute_nvs_loss:
            if (height, width) == (480, 832):
                target_gt_images = data["target_images"]
            else:
                target_gt_images = data.get("lagernvs_target_images")
                if target_gt_images is None:
                    target_gt_images = data["target_images"]
        return self.lagernvs_condition.build_lagernvs_geom(
            context_images=context_images,
            context_w2c=data["context_w2c"],
            target_w2c=data["target_w2c"],
            context_intrinsics=context_intrinsics,
            target_intrinsics=target_intrinsics,
            num_context=len(data["input_images"]),
            num_target_latent_frames=num_output_frames,
            roge_height=height,
            roge_width=width,
            dtype=self.pipe.torch_dtype,
            pipe=self.pipe,
            cache_aux_rgb=cache_aux_rgb,
            target_gt_images=target_gt_images,
            compute_nvs_loss=compute_nvs_loss,
        )

    def _attach_lagernvs_inputs(self, inputs_shared, data, num_output_frames):
        if self.joint_train:
            geom, nvs_loss = self._build_lagernvs_geom(
                data,
                num_output_frames=num_output_frames,
                compute_nvs_loss=True,
            )
            self._pending_nvs_loss = nvs_loss
            inputs_shared["vggt_geom"] = geom
            return inputs_shared
        geom = self._build_lagernvs_geom(data, num_output_frames=num_output_frames)
        inputs_shared["vggt_geom"] = geom
        return inputs_shared

    def _latent_layout(self, num_frames, num_context):
        num_target_video_frames = num_frames - num_context
        if num_target_video_frames <= 0:
            raise ValueError(
                f"Expected target frames after {num_context} context frames, got num_frames={num_frames}"
            )
        if self.hybrid_vae_encoding:
            num_output_frames = wan_native_num_latent_frames(num_target_video_frames)
            num_latent_frames = num_context + num_output_frames
        else:
            num_latent_frames = num_frames
            num_output_frames = num_target_video_frames
        return num_latent_frames, num_output_frames, num_target_video_frames

    def _get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        num_frames = len(data["target_images"])
        num_context = len(data["input_images"])
        num_latent_frames, num_output_frames, num_target_video_frames = self._latent_layout(
            num_frames, num_context
        )

        inputs_shared = {
            "input_image": data["input_images"],
            "input_video": data["target_images"],
            "raymap": data["raymap"],
            "height": data["input_images"][0].size[1],
            "width": data["input_images"][0].size[0],
            "num_frames": num_frames,
            "num_output_frames": num_output_frames,
            "num_latent_frames": num_latent_frames,
            "num_context_frames": num_context,
            "num_target_video_frames": num_target_video_frames,
            "hybrid_vae_encoding": self.hybrid_vae_encoding,
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self._attach_lagernvs_inputs(
            inputs_shared, data, num_output_frames
        )
        inputs_shared = self._parse_extra_inputs(data, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        self._pending_nvs_loss = None
        self.last_fm_loss = None
        self.last_nvs_loss = None
        self.last_nvs_loss_weighted = None
        if inputs is None:
            inputs = self._get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs_shared, inputs_posi, _ = inputs
        fm_loss = FlowMatchSFTLoss(self.pipe, **inputs_shared, **inputs_posi)
        self.last_fm_loss = fm_loss.detach()
        if self.joint_train and self._pending_nvs_loss is not None:
            nvs_loss = self._pending_nvs_loss
            self._pending_nvs_loss = None
            nvs_weighted = self.nvs_loss_weight * nvs_loss
            self.last_nvs_loss = nvs_loss.detach()
            self.last_nvs_loss_weighted = nvs_weighted.detach()
            return fm_loss + nvs_weighted
        return fm_loss

    @torch.no_grad()
    def predict_video(self, data, num_inference_steps: int = 20, seed: int = 42):
        """Run full inference on one training sample for visualization."""
        input_images = data["input_images"]
        target_images = data["target_images"]
        num_context = len(input_images)
        height = input_images[0].size[1]
        width = input_images[0].size[0]

        raymap = data["raymap"]
        if isinstance(raymap, torch.Tensor):
            raymap = raymap.to(device=self.pipe.device, dtype=self.pipe.torch_dtype)

        num_latent_frames, num_output_frames, num_target_video_frames = self._latent_layout(
            len(target_images), num_context
        )

        pipe_kwargs = {}
        context_images_l, context_intrinsics_l, target_intrinsics_l = (
            self._lagernvs_context_inputs(data)
        )
        lagernvs_geom = self._build_lagernvs_geom(
            data,
            num_output_frames=num_output_frames,
            cache_aux_rgb=True,
        )
        pipe_kwargs["vggt_geom"] = lagernvs_geom.to(
            device=self.pipe.device, dtype=self.pipe.torch_dtype
        )
        aux_row_frames = self.lagernvs_condition.build_vis_aux_frames(
            context_images=context_images_l,
            context_w2c=data["context_w2c"],
            target_w2c=data["target_w2c"],
            context_intrinsics=context_intrinsics_l,
            target_intrinsics=target_intrinsics_l,
            num_total_frames=len(target_images),
            roge_height=height,
            roge_width=width,
        )

        dit = self.pipe.dit
        was_training = dit.training
        dit.eval()
        try:
            pred_frames = self.pipe(
                prompt="",
                negative_prompt="",
                input_image=input_images,
                raymap=raymap,
                height=height,
                width=width,
                **pipe_kwargs,
                num_frames=len(target_images),
                num_latent_frames=num_latent_frames,
                num_output_frames=num_output_frames,
                num_context_frames=num_context,
                num_target_video_frames=num_target_video_frames,
                hybrid_vae_encoding=self.hybrid_vae_encoding,
                decode_target_only=self.hybrid_vae_encoding,
                cfg_scale=1.0,
                num_inference_steps=num_inference_steps,
                seed=seed,
                tiled=True,
                progress_bar_cmd=lambda x: x,
            )
            if self.hybrid_vae_encoding:
                pred_frames = list(input_images) + pred_frames
        finally:
            if was_training:
                dit.train()
            # ``pipe.__call__`` sets scheduler to inference mode (training=False,
            # 50 timesteps). Restore training scheduler or the next forward pass
            # will not produce ``input_latents``.
            self.pipe.scheduler.set_timesteps(1000, training=True)

        return target_images, pred_frames, num_context, aux_row_frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def wan_parser():
    parser = argparse.ArgumentParser(description="RoGe LoRA training.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional override for the T5 tokenizer path.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max sampling boundary (fraction of timesteps).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min sampling boundary (fraction of timesteps).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Build the pipeline on CPU before accelerator.prepare moves it.")
    parser.add_argument("--modify_channels", default=False, action="store_true", help="Rebuild patch_embedding with --new_in_dim input channels.")
    parser.add_argument("--new_in_dim", type=int, default=None, help="New input channels for patch_embedding (required with --modify_channels).")
    parser.add_argument(
        "--hybrid_vae_encoding",
        default=False,
        action="store_true",
        help="Context frames use per-frame VAE; target frames use Wan continuous VAE. "
             "Implies per-frame encoding for context. Enables split temporal RoPE and "
             f"VAE-style packed raymaps (raymap {raymap_channels()}ch, in_dim={base_in_dim()}).",
    )
    parser.add_argument(
        "--raymap_pe",
        default=False,
        action="store_true",
        help="Inject packed raymap via PixelUnshuffle(2) + 1x1 pe residual-add "
             "(requires --hybrid_vae_encoding; patch_embedding in_dim drops "
             f"raymap channels to {base_in_dim(raymap_pe=True)} plus optional geom).",
    )
    parser.add_argument(
        "--geom_pe",
        default=False,
        action="store_true",
        help="Inject LagerNVS geom via PixelUnshuffle(2) + 1x1 pe residual-add "
             "(patch_embedding in_dim drops geom channels).",
    )
    parser.add_argument(
        "--from_wan",
        default=False,
        action="store_true",
        help="Start from Wan-I2V-14B-480P only (keep full 36ch patch_embedding). "
             "Requires --raymap_pe and --geom_pe. Incompatible with "
             "--resume_ckpt_path/--resume_checkpoint/--lora_checkpoint. "
             "Trainable set matches non-from_wan: LoRA + patch_embedding + "
             "raymap_pe + geom_pe (+ geom projector).",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
        help="Path to pretrained RoGe weights (.safetensors or .pt). "
             "Restores LoRA, patch_embedding, and LagerNVS projector (if present).",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Alias of --resume_ckpt_path (legacy). Ignored when --resume_ckpt_path is set.",
    )
    parser.add_argument(
        "--num_dataset_samples",
        type=int,
        default=-1,
        help="Max scenes to load (sorted). -1 means no limit.",
    )
    parser.add_argument(
        "--dataset_check",
        default=False,
        action="store_true",
        help="Quick debug mode for mixed training: each sub-dataset stops discovery "
             "after 10 valid candidates (no full index/directory scan), then loads "
             "only those samples.",
    )
    parser.add_argument("--dataset_type", type=str, default="mixed",
                        choices=["arkitscenes", "blendedmvs", "dl3dv", "re10k",
                                 "mvssynth", "wildrgbd", "mixed"],
                        help="Training dataset backend. Use 'mixed' with --mixed_dataset_config.")
    parser.add_argument(
        "--mixed_dataset_config",
        type=str,
        default=None,
        help="YAML/JSON config listing sub-datasets for mixed training "
             "(see training/dataset.yaml).",
    )
    parser.add_argument(
        "--inside_random",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Sample uniformly across all sub-datasets in mixed mode. "
             "Defaults to the config value, or True for mixed training.",
    )
    parser.add_argument(
        "--dataloader_shuffle",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Shuffle DataLoader indices. Auto-disabled when inside_random is on.",
    )
    parser.add_argument(
        "--distributed_timeout_mins",
        type=int,
        default=1440,
        help="PyTorch distributed process-group timeout in minutes (VGGT distributed.timeout_mins). "
             "Increase when mixed-dataset scanning or checkpoint/vis collectives are slow.",
    )
    parser.add_argument("--scene_metadata", type=str, default="new_scene_metadata.npz",
                        help="Metadata npz filename inside each ARKitScenes scene folder "
                             "(default: new_scene_metadata.npz).")
    parser.add_argument("--num_context_frames", type=int, default=16,
                        help="Number of context frames M per sample.")
    parser.add_argument("--context_shuffle", default=False, action="store_true",
                        help="Randomly shuffle sampled DL3DV/RE10K context frames.")
    parser.add_argument(
        "--target_inverse",
        default=False,
        action="store_true",
        help="DL3DV only: after sampling, reverse target frames with 50% probability.",
    )
    parser.add_argument(
        "--mixed_train",
        default=False,
        action="store_true",
        help="All datasets: keep --num_target_frames fixed, but draw num_context_frames "
             "uniformly from [3, 9] each step (same M across all DDP ranks; "
             "DataLoader shuffling is disabled automatically).",
    )
    parser.add_argument("--num_target_frames", type=int, default=5,
                        help="Number of consecutive target frames N per sample.")
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        required=True,
        help="Directory for dataset scene-index caches. Created if missing. "
             "Each sub-dataset stores one pickle keyed by (dataset_name, base_path).",
    )
    parser.add_argument(
        "--update_dataset_cache",
        default=False,
        action="store_true",
        help="Rescan all datasets and overwrite existing cache files under "
             "--dataset_cache_dir.",
    )
    parser.add_argument(
        "--lagernvs_ckpt",
        type=str,
        required=True,
        help="LagerNVS checkpoint (.pt).",
    )
    parser.add_argument(
        "--lagernvs_offload_per_step",
        default=False,
        action="store_true",
        help="Offload frozen LagerNVS to CPU after each conditioning forward.",
    )
    parser.add_argument(
        "--lagernvs_channels",
        "--lagernvs_geom_channels",
        type=int,
        default=LAGERNVS_CHANNELS,
        dest="lagernvs_channels",
        help="Pre-pack projected LagerNVS feature channels "
             "(DiT concat = this value * 4 after pack; in_dim auto-aligns to "
             "base_in_dim()+packed channels; default 384). "
             "--lagernvs_geom_channels is a deprecated alias.",
    )
    parser.add_argument(
        "--joint_train",
        default=False,
        action="store_true",
        help="Light joint adaptation: inject LoRA into the LagerNVS renderer, "
             "open the token autograd graph for geom conditioning, and add an "
             "NVS aux MSE loss on target-view RGB.",
    )
    parser.add_argument(
        "--train_reconstructor_lora",
        default=False,
        action="store_true",
        help="With --joint_train: also inject LoRA into the LagerNVS "
             "reconstructor (VGGT attn qkv/proj), disable its freeze_vggt "
             "detach, and open the reconstructor autograd graph. "
             "Requires --joint_train.",
    )
    parser.add_argument(
        "--load_frozen_lagernvs_lora",
        default=False,
        action="store_true",
        help="Load renderer and reconstructor LoRA from the resume checkpoint, "
             "but freeze both and disable joint/NVS auxiliary training. "
             "Mutually exclusive with --joint_train.",
    )
    parser.add_argument(
        "--nvs_loss_weight",
        type=float,
        default=0.1,
        help="Weight for the NVS auxiliary MSE loss under --joint_train.",
    )
    parser.add_argument(
        "--lagernvs_lora_rank",
        type=int,
        default=LAGERNVS_LORA_RANK,
        help="LoRA rank for LagerNVS renderer/reconstructor under --joint_train "
             f"(default {LAGERNVS_LORA_RANK}).",
    )
    parser.add_argument(
        "--skip_clip",
        default=False,
        action="store_true",
        help="Do not load the Wan OpenCLIP image encoder. Safe for "
             "RoGe NVS training (CLIP is unused).",
    )
    parser.add_argument(
        "--unload_text_encoder",
        default=False,
        action="store_true",
        help="After init, encode the empty prompt once, keep a CPU copy, then "
             "delete UMT5 to free VRAM. Train/vis must keep using prompt=\"\".",
    )
    return parser


DATASET_CHECK_SAMPLES = 10


def _resolve_num_dataset_samples(args, num_dataset_samples=None):
    if args.dataset_check:
        return DATASET_CHECK_SAMPLES
    if num_dataset_samples is None:
        num_dataset_samples = args.num_dataset_samples
    return normalize_num_dataset_samples(num_dataset_samples)


def _common_dataset_kwargs(args, *, num_dataset_samples=None, random_subset=None):
    if args.dataset_check:
        random_subset = True
    elif random_subset is None:
        random_subset = False
    return dict(
        repeat=args.dataset_repeat,
        num_context_frames=args.num_context_frames,
        num_target_frames=args.num_target_frames,
        height=args.height,
        width=args.width,
        height_division_factor=8,
        width_division_factor=8,
        no_pixel_unshuffle=False,
        num_dataset_samples=_resolve_num_dataset_samples(args, num_dataset_samples),
        random_subset=random_subset,
        pose_gauge="cam_first_lagernvs",
        dataset_cache_dir=args.dataset_cache_dir,
        update_dataset_cache=args.update_dataset_cache,
        raymap_full_resolution=args.hybrid_vae_encoding,
        mixed_train=bool(getattr(args, "mixed_train", False)),
    )


def _load_mixed_dataset_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required for YAML mixed-dataset configs. "
                    "Install with: pip install pyyaml"
                ) from exc
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Mixed dataset config must be a mapping, got {type(config)}")
    datasets = config.get("datasets")
    if not datasets:
        raise ValueError(f"Mixed dataset config {path} must define a non-empty 'datasets' list")
    return config


def build_single_dataset(dataset_type: str, base_path: str, args, **overrides):
    num_dataset_samples = overrides.pop("num_dataset_samples", args.num_dataset_samples)
    metadata_name = overrides.pop("metadata_name", None)
    splits = overrides.pop("splits", None)
    benchmark_tier = overrides.pop("benchmark_tier", None)
    excluded_scene_hashes = overrides.pop("excluded_scene_hashes", None)
    excluded_scene_ids = overrides.pop("excluded_scene_ids", None)

    common = _common_dataset_kwargs(args, num_dataset_samples=num_dataset_samples)
    common.update(overrides)

    if dataset_type == "blendedmvs":
        return BlendedMVSRoGeDataset(base_path=base_path, **common)
    if dataset_type == "arkitscenes":
        return ARKitScenesRoGeDataset(
            base_path=base_path,
            metadata_name=metadata_name or args.scene_metadata,
            **common,
        )
    if dataset_type == "wildrgbd":
        return WildRGBDRoGeDataset(base_path=base_path, **common)
    if dataset_type == "mvssynth":
        return MVSSynthRoGeDataset(base_path=base_path, **common)
    if dataset_type == "dl3dv":
        dl3dv_kwargs = {
            "context_shuffle": args.context_shuffle,
            "target_inverse": args.target_inverse,
        }
        if splits is not None:
            dl3dv_kwargs["splits"] = splits
        if benchmark_tier is not None:
            dl3dv_kwargs["benchmark_tier"] = benchmark_tier
        if excluded_scene_hashes is not None:
            dl3dv_kwargs["excluded_scene_hashes"] = excluded_scene_hashes
        return DL3DVRoGeDataset(base_path=base_path, **common, **dl3dv_kwargs)
    if dataset_type == "re10k":
        re10k_kwargs = {
            "context_shuffle": args.context_shuffle,
        }
        if splits is not None:
            re10k_kwargs["splits"] = splits
        if excluded_scene_ids is not None:
            re10k_kwargs["excluded_scene_ids"] = excluded_scene_ids
        return RE10KRoGeDataset(base_path=base_path, **common, **re10k_kwargs)
    raise ValueError(f"Unknown dataset_type: {dataset_type}")


def build_mixed_dataset(args):
    if args.mixed_dataset_config is None:
        raise ValueError("--mixed_dataset_config is required when --dataset_type mixed")

    if args.dataset_check:
        print(
            f"Dataset check mode: each sub-dataset discovers at most "
            f"{DATASET_CHECK_SAMPLES} candidates (early stop, no full scan)."
        )

    config = _load_mixed_dataset_config(args.mixed_dataset_config)
    inside_random = config.get("inside_random", True)
    if args.inside_random is not None:
        inside_random = args.inside_random

    datasets = []
    names = []
    len_trains = []
    for entry in config["datasets"]:
        if not isinstance(entry, dict):
            raise ValueError("Each mixed-dataset entry must be a mapping")
        dataset_type = entry.get("type")
        base_path = entry.get("base_path")
        if not dataset_type or not base_path:
            raise ValueError("Each mixed-dataset entry requires 'type' and 'base_path'")
        if not os.path.isabs(base_path):
            base_path = os.path.join(_REPO_ROOT, base_path)

        overrides = {}
        if "metadata_name" in entry:
            overrides["metadata_name"] = entry["metadata_name"]
        if not args.dataset_check and "num_dataset_samples" in entry:
            overrides["num_dataset_samples"] = entry["num_dataset_samples"]
        if "repeat" in entry:
            overrides["repeat"] = entry["repeat"]
        if "splits" in entry:
            overrides["splits"] = entry["splits"]
        if "benchmark_tier" in entry:
            overrides["benchmark_tier"] = entry["benchmark_tier"]
        if "excluded_scene_hashes" in entry:
            overrides["excluded_scene_hashes"] = entry["excluded_scene_hashes"]
        if "excluded_scene_ids" in entry:
            overrides["excluded_scene_ids"] = entry["excluded_scene_ids"]

        datasets.append(build_single_dataset(dataset_type, base_path, args, **overrides))
        names.append(dataset_type)
        actual_len = len(datasets[-1])
        if args.dataset_check:
            len_trains.append(min(DATASET_CHECK_SAMPLES, actual_len))
        else:
            len_trains.append(entry.get("len_train", actual_len))

    return MixedRoGeDataset(
        datasets=datasets,
        names=names,
        len_trains=len_trains,
        inside_random=inside_random,
    )


def build_dataset(args):
    if args.dataset_check and args.dataset_type != "mixed":
        raise ValueError("--dataset_check is only supported with --dataset_type mixed")
    if args.dataset_type == "mixed":
        return build_mixed_dataset(args)
    if not args.dataset_base_path:
        raise ValueError("--dataset_base_path is required unless --dataset_type mixed")
    return build_single_dataset(args.dataset_type, args.dataset_base_path, args)


def resolve_training_schedule(args, world_size: int) -> None:
    """Derive steps_per_epoch and max_steps from epoch schedule (mutates args)."""
    if args.steps_per_epoch is None and args.steps_per_epoch_base is not None:
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        args.steps_per_epoch = args.steps_per_epoch_base // world_size

    if args.steps_per_epoch is not None and args.max_steps is None:
        args.max_steps = args.num_epochs * args.steps_per_epoch


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    if getattr(args, "train_reconstructor_lora", False) and not getattr(
        args, "joint_train", False
    ):
        parser.error("--train_reconstructor_lora requires --joint_train")
    if getattr(args, "load_frozen_lagernvs_lora", False):
        if args.joint_train:
            parser.error(
                "--load_frozen_lagernvs_lora and --joint_train are mutually exclusive"
            )
    limit = normalize_num_dataset_samples(args.num_dataset_samples)
    limit_msg = "unlimited" if limit is None else str(limit)
    print(f"num_dataset_samples={args.num_dataset_samples} (effective per-dataset limit: {limit_msg})")
    from diffsynth.core.data.dataset_cache import resolve_cache_dir

    cache_dir = resolve_cache_dir(args.dataset_cache_dir)
    cache_action = "update (rescan)" if args.update_dataset_cache else "load or create"
    print(f"Dataset scene-index cache: {cache_action} -> {cache_dir}")
    # Scan datasets before initializing distributed; avoids NCCL timeout while ranks
    # walk large mixed-dataset roots.
    dataset = build_dataset(args)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.InitProcessGroupKwargs(
                timeout=timedelta(minutes=args.distributed_timeout_mins)
            ),
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            ),
        ],
    )
    resolve_training_schedule(args, accelerator.num_processes)
    if accelerator.is_main_process:
        schedule_parts = [f"num_epochs={args.num_epochs}"]
        if args.steps_per_epoch is not None:
            base_note = (
                f" (steps_per_epoch_base={args.steps_per_epoch_base}"
                f" / {accelerator.num_processes} GPUs)"
                if args.steps_per_epoch_base is not None
                else ""
            )
            schedule_parts.append(f"steps_per_epoch={args.steps_per_epoch}{base_note}")
        if args.max_steps is not None:
            schedule_parts.append(f"max_steps={args.max_steps}")
        schedule_parts.append(f"log_interval={args.log_interval}")
        print("Training schedule: " + ", ".join(schedule_parts))
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        modify_channels=args.modify_channels,
        new_in_dim=args.new_in_dim,
        hybrid_vae_encoding=args.hybrid_vae_encoding,
        resume_ckpt_path=args.resume_ckpt_path,
        resume_checkpoint=args.resume_checkpoint,
        raymap_pe=args.raymap_pe,
        geom_pe=args.geom_pe,
        from_wan=args.from_wan,
        lagernvs_ckpt=args.lagernvs_ckpt,
        lagernvs_offload_per_step=args.lagernvs_offload_per_step,
        lagernvs_channels=args.lagernvs_channels,
        joint_train=args.joint_train,
        nvs_loss_weight=args.nvs_loss_weight,
        lagernvs_lora_rank=args.lagernvs_lora_rank,
        train_reconstructor_lora=args.train_reconstructor_lora,
        load_frozen_lagernvs_lora=args.load_frozen_lagernvs_lora,
        skip_clip=args.skip_clip,
        unload_text_encoder=args.unload_text_encoder,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        save_keep_first=args.save_keep_first,
    )
    launch_training_task(accelerator, dataset, model, model_logger, args=args)
