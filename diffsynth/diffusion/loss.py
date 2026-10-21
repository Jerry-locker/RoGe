import torch
from .base_pipeline import BasePipeline


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    """Supervised-finetuning flow-matching loss.

    Adds noise to ``inputs['input_latents']`` at a sampled timestep, runs the
    DiT once via ``pipe.model_fn``, and computes a weighted MSE against the
    flow-matching training target. When ``inputs['mask_loss']`` is set the
    loss is restricted to the last ``num_output_frames`` latent frames.
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    num_out = inputs.get("num_output_frames", 1)
    if "mask_loss" in inputs:
        noise_pred = noise_pred[:, :, -num_out:]
        training_target = training_target[:, :, -num_out:]

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss
