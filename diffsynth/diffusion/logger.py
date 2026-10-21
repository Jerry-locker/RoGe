import os

import torch
from accelerate import Accelerator
from safetensors.torch import save_file

from .training_module import get_unwrapped_module


def _get_trainable_state_dict(model: torch.nn.Module, remove_prefix: str | None):
    """Collect trainable weights without ``accelerator.get_state_dict``.

    ``accelerator.get_state_dict`` / ``unwrap_model`` may import DeepSpeed even
    under plain DDP, which fails when ``nvcc`` is unavailable.
    """
    unwrapped = get_unwrapped_module(model)
    return unwrapped.export_trainable_state_dict(
        unwrapped.state_dict(), remove_prefix=remove_prefix
    )


class ModelLogger:
    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x: x,
        save_keep_first: int = 5,
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.save_keep_first = save_keep_first
        self.num_steps = 0

    def _maybe_delete_previous_rolling(self, step: int, save_steps: int):
        """Drop the previous rolling ckpt once we are past the first N saves."""
        save_index = step // save_steps
        if save_index <= self.save_keep_first:
            return
        prev_step = step - save_steps
        if prev_step > self.save_keep_first * save_steps:
            prev_path = os.path.join(self.output_path, f"step-{prev_step}.safetensors")
            if os.path.isfile(prev_path):
                os.remove(prev_path)
                print(f"Removed old rolling checkpoint: {prev_path}")

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(
                accelerator, model, f"step-{self.num_steps}.safetensors", save_steps=save_steps
            )

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = self.state_dict_converter(
                _get_trainable_state_dict(model, self.remove_prefix_in_ckpt)
            )
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            save_file(state_dict, path)
            print(f"Saved checkpoint: {path}")

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(
                accelerator, model, f"step-{self.num_steps}.safetensors", save_steps=save_steps
            )

    def save_model(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        file_name,
        save_steps: int | None = None,
    ):
        # Training with variable-size samples leaves the CUDA caching allocator
        # at its high-water mark. Return unused cached blocks before NCCL's
        # checkpoint barrier so the backend can allocate its own workspace.
        # Every rank reaches this method, so cleanup happens symmetrically.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if save_steps is not None:
                self._maybe_delete_previous_rolling(self.num_steps, save_steps)
            state_dict = self.state_dict_converter(
                _get_trainable_state_dict(model, self.remove_prefix_in_ckpt)
            )
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            save_file(state_dict, path)
            print(f"Saved checkpoint: {path}")
