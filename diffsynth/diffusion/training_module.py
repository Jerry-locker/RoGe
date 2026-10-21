import torch, json, os
from ..core import ModelConfig, load_state_dict
from ..utils.lora import adapt_lora_rank
from peft import LoraConfig, inject_adapter_in_model


def get_unwrapped_module(model):
    """Peel DDP / FSDP ``.module`` wrappers without ``accelerator.unwrap_model``.

    ``accelerator.unwrap_model`` may import DeepSpeed even under plain DDP,
    which fails on machines without ``nvcc`` in ``CUDA_HOME``.
    """
    m = model
    while hasattr(m, "module"):
        m = m.module
    return m


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self


    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules


    @staticmethod
    def _is_patch_embedding_param(name: str) -> bool:
        """True for DiT ``...patch_embedding.*`` params (exact path segment)."""
        return "patch_embedding" in name.split(".")


    def trainable_param_groups(
        self,
        learning_rate: float,
        patch_embedding_lr: float | None = None,
    ):
        """Build AdamW param groups, optionally with a separate ``patch_embedding`` LR.

        Args:
            learning_rate: Default LR for LoRA / raymap_pe / geom_pe / projector / etc.
            patch_embedding_lr: If ``None``, ``patch_embedding`` shares ``learning_rate``.
                If ``0``, freeze ``patch_embedding`` (exclude from optimizer).
                Otherwise use this LR for ``patch_embedding`` only.

        Returns:
            List of ``{"params", "lr", "name"}`` dicts suitable for ``AdamW``.
        """
        pe_lr = learning_rate if patch_embedding_lr is None else float(patch_embedding_lr)
        freeze_pe = patch_embedding_lr is not None and float(patch_embedding_lr) == 0.0

        pe_params = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if self._is_patch_embedding_param(name):
                if freeze_pe:
                    param.requires_grad = False
                    continue
                pe_params.append(param)
            else:
                other_params.append(param)

        groups = []
        if other_params:
            groups.append(
                {"params": other_params, "lr": float(learning_rate), "name": "default"}
            )
        if pe_params:
            groups.append(
                {
                    "params": pe_params,
                    "lr": pe_lr,
                    "name": "patch_embedding",
                }
            )
        if not groups:
            raise ValueError("No trainable parameters found for the optimizer")
        return groups


    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names

    def checkpoint_param_names(self):
        """Parameter names persisted by lightweight training checkpoints."""
        return self.trainable_param_names()


    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        if isinstance(target_modules, list) and len(target_modules) == 1:
            target_modules = target_modules[0]
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        checkpoint_param_names = self.checkpoint_param_names()
        state_dict = {
            name: param
            for name, param in state_dict.items()
            if name in checkpoint_param_names
        }
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict


    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        if data is None:
            return data
        elif isinstance(data, torch.Tensor):
            data = data.to(device)
            if torch_float_dtype is not None and data.dtype in [torch.float, torch.float16, torch.bfloat16]:
                data = data.to(torch_float_dtype)
            return data
        elif isinstance(data, tuple):
            data = tuple(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, list):
            data = list(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, dict):
            data = {i: self.transfer_data_to_device(data[i], device, torch_float_dtype) for i in data}
            return data
        else:
            return data


    def parse_model_configs(self, model_paths, model_id_with_origin_paths, device="cpu"):
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            for path in model_paths:
                model_configs.append(ModelConfig(path=path))
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            for model_id_with_origin_path in model_id_with_origin_paths:
                config = self.parse_path_or_model_id(model_id_with_origin_path)
                model_configs.append(ModelConfig(model_id=config.model_id, origin_file_pattern=config.origin_file_pattern))
        return model_configs


    def parse_path_or_model_id(self, model_id_with_origin_path, default_value=None):
        if model_id_with_origin_path is None:
            return default_value
        elif os.path.exists(model_id_with_origin_path):
            return ModelConfig(path=model_id_with_origin_path)
        else:
            if ":" not in model_id_with_origin_path:
                raise ValueError(f"Failed to parse model config: {model_id_with_origin_path}. This is neither a valid path nor in the format of `model_id/origin_file_pattern`.")
            split_id = model_id_with_origin_path.rfind(":")
            model_id = model_id_with_origin_path[:split_id]
            origin_file_pattern = model_id_with_origin_path[split_id + 1:]
            return ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern)


    def parse_lora_target_modules(self, lora_target_modules):
        if lora_target_modules == "":
            raise ValueError("--lora_target_modules must be provided (e.g. 'q,k,v,o,ffn.0,ffn.2').")
        return lora_target_modules.split(",")


    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
    ):
        pipe.scheduler.set_timesteps(1000, training=True)

        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))

        if lora_base_model is not None:
            if (not hasattr(pipe, lora_base_model)) or getattr(pipe, lora_base_model) is None:
                print(f"No {lora_base_model} models in the pipeline. We cannot patch LoRA on the model.")
                return
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=self.parse_lora_target_modules(lora_target_modules),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                # Joint-train ckpts also store LagerNVS/VGGT LoRA (often a
                # different rank). Only adapt/load DiT LoRA here; condition
                # heads are restored later via _load_pretrained_checkpoint.
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if "lagernvs_condition." not in k and "vggt_condition." not in k
                }
                state_dict = adapt_lora_rank(state_dict, lora_rank)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)


