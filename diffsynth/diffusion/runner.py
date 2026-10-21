import os

import torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule, get_unwrapped_module
from .logger import ModelLogger


def _loss_log_extras(model) -> dict[str, float]:
    """Optional per-term losses cached on the training module after forward."""
    unwrapped = get_unwrapped_module(model)
    extras: dict[str, float] = {}
    fm = getattr(unwrapped, "last_fm_loss", None)
    if fm is not None:
        extras["fm_loss"] = float(fm.item())
    nvs = getattr(unwrapped, "last_nvs_loss", None)
    if nvs is not None:
        extras["nvs_loss"] = float(nvs.item())
    nvs_w = getattr(unwrapped, "last_nvs_loss_weighted", None)
    if nvs_w is not None:
        extras["nvs_loss_weighted"] = float(nvs_w.item())
    return extras
from diffsynth.core.data.reproducible_dataset import (
    ReproducibleTrainingDataset,
    wrap_for_reproducible_training,
)
from diffsynth.core.data.sampling_seed import build_worker_init_fn, set_training_seeds


def _format_training_progress(
    epoch_id: int,
    num_epochs: int,
    step_in_epoch: int | None,
    steps_per_epoch: int | None,
    global_step: int,
    max_steps: int | None,
    loss: float,
    learning_rate: float,
    learning_rates: list[float] | None = None,
) -> str:
    parts = [f"[train] epoch {epoch_id + 1}/{num_epochs}"]
    if step_in_epoch is not None and steps_per_epoch is not None:
        parts.append(f"step {step_in_epoch}/{steps_per_epoch}")
    if max_steps is not None:
        parts.append(f"global {global_step}/{max_steps}")
    else:
        parts.append(f"global {global_step}")
    parts.append(f"loss={loss:.6f} lr={learning_rate:.2e}")
    if learning_rates is not None and len(learning_rates) > 1:
        parts.append(
            "lrs=[" + ", ".join(f"{lr:.2e}" for lr in learning_rates) + "]"
        )
    return " ".join(parts)


def _maybe_log_training_progress(
    accelerator,
    *,
    log_interval: int,
    epoch_id: int,
    num_epochs: int,
    step_in_epoch: int | None,
    steps_per_epoch: int | None,
    global_step: int,
    max_steps: int | None,
    loss: float,
    learning_rate: float,
    learning_rates: list[float] | None = None,
):
    if log_interval <= 0 or not accelerator.is_main_process:
        return
    if global_step % log_interval != 0:
        return
    print(
        _format_training_progress(
            epoch_id,
            num_epochs,
            step_in_epoch,
            steps_per_epoch,
            global_step,
            max_steps,
            loss,
            learning_rate,
            learning_rates=learning_rates,
        ),
        flush=True,
    )


def _training_step(
    accelerator,
    model,
    optimizer,
    scheduler,
    data,
    model_logger,
    save_steps,
    use_wandb,
    epoch_id,
    global_step,
    tensorboard_tracker=None,
    vis_steps=None,
    vis_inference_steps=20,
    vis_seed=42,
    vis_output_dir=None,
):
    with accelerator.accumulate(model):
        optimizer.zero_grad()
        loss = model(data)
        accelerator.backward(loss)
        optimizer.step()
        # Release gradient buffers as soon as the synchronized optimizer step
        # finishes. In particular, checkpoint barriers should not have to
        # allocate NCCL workspace while the just-consumed gradients remain live.
        if accelerator.sync_gradients:
            optimizer.zero_grad()
        model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
        scheduler.step()

    global_step += 1
    if accelerator.is_main_process:
        last_lrs = scheduler.get_last_lr()
        loss_extras = _loss_log_extras(model)
        if tensorboard_tracker is not None:
            tensorboard_tracker.log_scalars(
                global_step,
                loss.item(),
                last_lrs[0],
                epoch_id,
                fm_loss=loss_extras.get("fm_loss"),
                nvs_loss=loss_extras.get("nvs_loss"),
                nvs_loss_weighted=loss_extras.get("nvs_loss_weighted"),
            )
            if len(last_lrs) > 1:
                for i, lr in enumerate(last_lrs):
                    tensorboard_tracker.writer.add_scalar(
                        f"train/learning_rate_group_{i}", lr, global_step
                    )
                # Common layout: group0=default, group1=patch_embedding
                if len(optimizer.param_groups) > 1:
                    for group, lr in zip(optimizer.param_groups, last_lrs):
                        name = group.get("name")
                        if name:
                            tensorboard_tracker.writer.add_scalar(
                                f"train/learning_rate_{name}", lr, global_step
                            )
                    tensorboard_tracker.writer.flush()
        if use_wandb:
            import wandb
            log_payload = {
                "train/loss": loss.item(),
                "train/epoch": epoch_id,
                "train/learning_rate": last_lrs[0],
            }
            if "fm_loss" in loss_extras:
                log_payload["train/fm_loss"] = loss_extras["fm_loss"]
            if "nvs_loss" in loss_extras:
                log_payload["train/nvs_loss"] = loss_extras["nvs_loss"]
            if "nvs_loss_weighted" in loss_extras:
                log_payload["train/nvs_loss_weighted"] = loss_extras["nvs_loss_weighted"]
            for group, lr in zip(optimizer.param_groups, last_lrs):
                name = group.get("name")
                if name:
                    log_payload[f"train/learning_rate_{name}"] = lr
            wandb.log(log_payload, step=global_step)

    from .training_vis import maybe_log_training_visuals

    maybe_log_training_visuals(
        accelerator,
        model,
        data,
        global_step,
        tensorboard_tracker,
        vis_steps,
        vis_inference_steps,
        vis_seed,
        vis_output_dir,
    )
    return global_step, loss.item()


def _set_dataloader_epoch(dataloader, epoch_id: int):
    """Advance the epoch counter each epoch for reshuffling / resampling.

    ``accelerate.prepare`` wraps the loader in a ``DataLoaderShard`` whose
    ``__iter__`` calls ``self.set_epoch(self.iteration)`` with its own internal
    ``iteration`` counter (initialised to 0). That call forwards to
    ``dataset.set_epoch(...)``, so setting the epoch on the dataset directly is
    silently overwritten back to 0 on the next ``iter(dataloader)`` -- every
    epoch would then draw the identical reproducible sample sequence. The
    shard's own ``set_epoch`` both advances ``iteration`` and forwards to the
    sampler/dataset, so it must be preferred.
    """
    if hasattr(dataloader, "set_epoch"):
        dataloader.set_epoch(epoch_id)
        return
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch_id)
        return
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None:
        sampler = getattr(batch_sampler, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch_id)


def _fetch_batch(dataloader, data_iter):
    """Return the next batch, restarting the dataloader iterator if exhausted."""
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(dataloader)
        return next(data_iter), data_iter


def _begin_epoch(dataloader, dataset, epoch_id: int):
    _set_dataloader_epoch(dataloader, epoch_id)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch_id)
    elif hasattr(dataset, "current_epoch"):
        dataset.current_epoch = epoch_id


def _run_epoch_step_training(
    accelerator,
    dataloader,
    dataset,
    model,
    optimizer,
    scheduler,
    model_logger,
    *,
    num_epochs: int,
    steps_per_epoch: int,
    max_steps: int,
    save_steps,
    use_wandb,
    global_step: int,
    log_interval: int = 10,
    tensorboard_tracker=None,
    vis_steps=None,
    vis_inference_steps=20,
    vis_seed=42,
    vis_output_dir=None,
) -> int:
    """Train for ``num_epochs`` epochs with a fixed ``steps_per_epoch`` each."""
    progress = tqdm(
        total=max_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc=f"epoch 1/{num_epochs}",
        mininterval=5,
    )
    for epoch_id in range(num_epochs):
        if global_step >= max_steps:
            break
        if accelerator.is_main_process:
            print(
                f"[train] === epoch {epoch_id + 1}/{num_epochs} started "
                f"({steps_per_epoch} steps/epoch) ===",
                flush=True,
            )
        _begin_epoch(dataloader, dataset, epoch_id)
        data_iter = iter(dataloader)
        for step_in_epoch in range(1, steps_per_epoch + 1):
            if global_step >= max_steps:
                break
            data, data_iter = _fetch_batch(dataloader, data_iter)
            global_step, loss_value = _training_step(
                accelerator,
                model,
                optimizer,
                scheduler,
                data,
                model_logger,
                save_steps,
                use_wandb,
                epoch_id,
                global_step,
                tensorboard_tracker=tensorboard_tracker,
                vis_steps=vis_steps,
                vis_inference_steps=vis_inference_steps,
                vis_seed=vis_seed,
                vis_output_dir=vis_output_dir,
            )
            _maybe_log_training_progress(
                accelerator,
                log_interval=log_interval,
                epoch_id=epoch_id,
                num_epochs=num_epochs,
                step_in_epoch=step_in_epoch,
                steps_per_epoch=steps_per_epoch,
                global_step=global_step,
                max_steps=max_steps,
                loss=loss_value,
                learning_rate=scheduler.get_last_lr()[0],
                learning_rates=scheduler.get_last_lr(),
            )
            if accelerator.is_local_main_process:
                progress.set_description(f"epoch {epoch_id + 1}/{num_epochs}")
                progress.set_postfix(
                    step=f"{step_in_epoch}/{steps_per_epoch}",
                    loss=f"{loss_value:.4f}",
                    refresh=False,
                )
            progress.update(1)
    progress.close()
    return global_step


class CosineLRController:
    """Cosine LR decay driven by real optimizer steps.

    Writes the LR straight into ``optimizer.param_groups`` instead of going
    through ``torch.optim.lr_scheduler``: ``accelerator.prepare`` wraps torch
    schedulers in ``AcceleratedScheduler``, which advances them
    ``num_processes`` times per ``step()`` call (and DeepSpeed only adopts its
    own schedule classes), so a prepared ``CosineAnnealingLR(T_max=max_steps)``
    would finish decaying ``num_processes``x too early. Direct
    ``param_groups['lr']`` writes work identically under DDP and ZeRO-2.
    Exposes ``step()`` / ``get_last_lr()`` like a torch scheduler.

    Multi-group: each param group's current ``lr`` at construction is treated as
    that group's base LR. ``eta_min`` scales per group by
    ``group_base / base_lr`` so relative LRs (e.g. pe 1e-5 vs default 1e-4)
    stay constant through warmup + cosine.
    """

    def __init__(self, optimizer, base_lr: float, eta_min: float, total_steps: int, warmup_steps: int = 0):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.eta_min = float(eta_min)
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        if self.warmup_steps >= self.total_steps:
            raise ValueError(
                f"lr_warmup_steps ({self.warmup_steps}) must be < total_steps ({self.total_steps})"
            )
        self.group_base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        scale = (self.eta_min / self.base_lr) if self.base_lr > 0 else 0.0
        self.group_eta_mins = [blr * scale for blr in self.group_base_lrs]
        self.num_steps = 0
        self._apply()

    def _lr_at(self, step: int, base_lr: float, eta_min: float) -> float:
        import math

        # Linear warmup: optimizer step k (1-indexed) runs at base_lr * k / warmup_steps,
        # reaching full base_lr exactly at step == warmup_steps (continuous with cosine).
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return base_lr * (step + 1) / self.warmup_steps
        span = max(1, self.total_steps - self.warmup_steps)
        t = (min(step, self.total_steps) - self.warmup_steps) / span
        return eta_min + (base_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * t))

    def _apply(self):
        for group, blr, emin in zip(
            self.optimizer.param_groups, self.group_base_lrs, self.group_eta_mins
        ):
            group["lr"] = self._lr_at(self.num_steps, blr, emin)

    def step(self):
        self.num_steps += 1
        self._apply()

    def get_last_lr(self):
        return [
            self._lr_at(self.num_steps, blr, emin)
            for blr, emin in zip(self.group_base_lrs, self.group_eta_mins)
        ]


class WarmupConstantLRController:
    """Linear warmup then constant LR, driven by real optimizer steps.

    Same ``param_groups['lr']`` write path as ``CosineLRController`` so DDP /
    ZeRO-2 do not multiply-step a prepared torch scheduler. Multi-group base
    LRs are preserved after warmup.
    """

    def __init__(self, optimizer, base_lr: float, warmup_steps: int = 0):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup_steps = max(0, int(warmup_steps))
        self.group_base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        self.num_steps = 0
        self._apply()

    def _lr_at(self, step: int, base_lr: float) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return base_lr * (step + 1) / self.warmup_steps
        return base_lr

    def _apply(self):
        for group, blr in zip(self.optimizer.param_groups, self.group_base_lrs):
            group["lr"] = self._lr_at(self.num_steps, blr)

    def step(self):
        self.num_steps += 1
        self._apply()

    def get_last_lr(self):
        return [self._lr_at(self.num_steps, blr) for blr in self.group_base_lrs]


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 100,
    steps_per_epoch: int = None,
    max_steps: int = None,
    args=None,
):
    """Flow-matching SFT training loop with optional W&B logging."""
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        steps_per_epoch = getattr(args, "steps_per_epoch", None)
        max_steps = args.max_steps

    training_seed = int(getattr(args, "seed", 42) if args is not None else 42)
    if not isinstance(dataset, ReproducibleTrainingDataset):
        dataset = wrap_for_reproducible_training(dataset, training_seed)
    set_training_seeds(training_seed, accelerator.process_index)
    mixed_train_enabled = bool(
        args is not None and getattr(args, "mixed_train", False)
    )
    dataset.set_training_context(
        accelerator.process_index,
        accelerator.num_processes,
        synchronize_mixed_steps=mixed_train_enabled,
    )

    use_wandb = args is not None and getattr(args, "wandb_project", None) is not None
    if use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    tensorboard_tracker = None
    vis_output_dir = None
    vis_steps = None
    vis_inference_steps = 20
    vis_seed = 42
    log_interval = 10
    if args is not None:
        log_interval = getattr(args, "log_interval", 10)
    if args is not None and not getattr(args, "no_tensorboard", False):
        tb_dir = getattr(args, "tensorboard_dir", None) or os.path.join(
            args.output_path, "tensorboard"
        )
        from .training_vis import TensorboardTracker

        if accelerator.is_main_process:
            tensorboard_tracker = TensorboardTracker(tb_dir)
        vis_output_dir = os.path.join(args.output_path, "vis")
        vis_steps = getattr(args, "vis_steps", None)
        if vis_steps is not None and vis_steps <= 0:
            vis_steps = None
        vis_inference_steps = getattr(args, "vis_inference_steps", 20)
        vis_seed = getattr(args, "vis_seed", 42)

    patch_embedding_lr = (
        getattr(args, "patch_embedding_lr", None) if args is not None else None
    )
    param_groups = model.trainable_param_groups(
        learning_rate, patch_embedding_lr=patch_embedding_lr
    )
    if accelerator.is_main_process:
        for group in param_groups:
            n_params = sum(p.numel() for p in group["params"])
            print(
                f"Optimizer group '{group.get('name', '?')}': "
                f"lr={group['lr']:.2e}, params={n_params:,}"
            )
            if group.get("name") == "patch_embedding" and patch_embedding_lr is None:
                print(
                    "  (patch_embedding uses --learning_rate; "
                    "pass --patch_embedding_lr to override, or 0 to freeze)"
                )
        if patch_embedding_lr is not None and float(patch_embedding_lr) == 0.0:
            print("patch_embedding: frozen (--patch_embedding_lr 0)")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    lr_scheduler_type = getattr(args, "lr_scheduler", "constant") if args is not None else "constant"
    cos_lr_enabled = bool(getattr(args, "cos_lr", False)) if args is not None else False
    use_cosine_lr = cos_lr_enabled or lr_scheduler_type == "cosine"
    lr_warmup_steps = int(getattr(args, "lr_warmup_steps", 0) or 0) if args is not None else 0
    lr_min = None
    cos_lr_total_steps = None
    if use_cosine_lr:
        cos_lr_total_steps = getattr(args, "cos_lr_total_steps", None) if args is not None else None
        if cos_lr_total_steps is None:
            cos_lr_total_steps = max_steps
        if cos_lr_total_steps is None:
            raise ValueError("--cos_lr requires --cos_lr_total_steps or a resolved max_steps")
        cos_lr_total_steps = int(cos_lr_total_steps)
        lr_min = getattr(args, "lr_min", None)
        if lr_min is None:
            lr_min = learning_rate * 0.1
        if accelerator.is_main_process:
            warmup_note = f", linear warmup {lr_warmup_steps} steps" if lr_warmup_steps > 0 else ""
            group_note = ""
            if len(param_groups) > 1:
                group_note = "; per-group base LRs scale eta_min proportionally"
            stop_note = (
                f", stop at max_steps={max_steps}"
                if max_steps is not None and int(max_steps) != cos_lr_total_steps
                else ""
            )
            print(
                f"LR scheduler: cosine {learning_rate:.2e} -> {lr_min:.2e} "
                f"over {cos_lr_total_steps} steps (real-step controller"
                f"{warmup_note}{group_note}{stop_note})"
            )
        scheduler = None  # built after accelerator.prepare on the live optimizer
    elif lr_scheduler_type == "constant_with_warmup":
        if accelerator.is_main_process:
            warmup_note = (
                f"linear warmup {lr_warmup_steps} steps then hold"
                if lr_warmup_steps > 0
                else "no warmup (constant)"
            )
            print(
                f"LR scheduler: constant_with_warmup {learning_rate:.2e} "
                f"({warmup_note}, real-step controller)"
            )
        scheduler = None  # built after accelerator.prepare on the live optimizer
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    dataloader_shuffle = True
    if mixed_train_enabled:
        # ReproducibleTrainingDataset maps sequential logical indices to a
        # rank-shared mixed step. Worker scheduling is deliberately irrelevant.
        dataloader_shuffle = False
        if (
            args is not None
            and getattr(args, "dataloader_shuffle", None) is True
            and accelerator.is_main_process
        ):
            print(
                "mixed_train: ignoring --dataloader_shuffle; synchronized M "
                "requires deterministic sequential index sharding"
            )
    elif args is not None and getattr(args, "dataloader_shuffle", None) is not None:
        dataloader_shuffle = args.dataloader_shuffle
    elif getattr(dataset, "inside_random", False):
        # VGGT-style mixed sampling: randomness lives in the dataset.
        dataloader_shuffle = False

    dataloader_generator = None
    if dataloader_shuffle:
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(
            training_seed + accelerator.process_index * 1_000_003
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=dataloader_shuffle,
        collate_fn=lambda x: x[0],
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=dataloader_generator,
        worker_init_fn=(
            build_worker_init_fn(
                training_seed,
                num_workers,
                accelerator.process_index,
                epoch=0,
            )
            if num_workers > 0
            else None
        ),
    )

    if use_cosine_lr or lr_scheduler_type == "constant_with_warmup":
        # Keep the LR controller outside accelerator.prepare (see CosineLRController);
        # build it on the prepared optimizer so param_group writes hit the live one.
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        if use_cosine_lr:
            scheduler = CosineLRController(
                optimizer, base_lr=learning_rate, eta_min=lr_min,
                total_steps=cos_lr_total_steps, warmup_steps=lr_warmup_steps,
            )
        else:
            scheduler = WarmupConstantLRController(
                optimizer, base_lr=learning_rate, warmup_steps=lr_warmup_steps,
            )
    else:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )

    dataset.set_epoch(0)

    global_step = 0
    if hasattr(dataset, "num_epochs"):
        dataset.num_epochs = num_epochs

    use_epoch_steps = steps_per_epoch is not None and max_steps is not None
    if use_epoch_steps:
        global_step = _run_epoch_step_training(
            accelerator,
            dataloader,
            dataset,
            model,
            optimizer,
            scheduler,
            model_logger,
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            max_steps=max_steps,
            save_steps=save_steps,
            use_wandb=use_wandb,
            global_step=global_step,
            log_interval=log_interval,
            tensorboard_tracker=tensorboard_tracker,
            vis_steps=vis_steps,
            vis_inference_steps=vis_inference_steps,
            vis_seed=vis_seed,
            vis_output_dir=vis_output_dir,
        )
        if save_steps is None or global_step % save_steps != 0:
            model_logger.save_model(
                accelerator, model, f"step-{global_step}.safetensors", save_steps=save_steps
            )
    elif max_steps is not None:
        epoch_id = 0
        step_in_epoch = 0
        progress = tqdm(
            total=max_steps,
            disable=not accelerator.is_local_main_process,
            desc=f"epoch 1/{num_epochs}",
            mininterval=5,
        )
        while global_step < max_steps:
            step_in_epoch = 0
            if accelerator.is_main_process:
                print(
                    f"[train] === epoch {epoch_id + 1}/{num_epochs} started ===",
                    flush=True,
                )
            _begin_epoch(dataloader, dataset, epoch_id)
            data_iter = iter(dataloader)
            while global_step < max_steps:
                step_in_epoch += 1
                data, data_iter = _fetch_batch(dataloader, data_iter)
                global_step, loss_value = _training_step(
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    data,
                    model_logger,
                    save_steps,
                    use_wandb,
                    epoch_id,
                    global_step,
                    tensorboard_tracker=tensorboard_tracker,
                    vis_steps=vis_steps,
                    vis_inference_steps=vis_inference_steps,
                    vis_seed=vis_seed,
                    vis_output_dir=vis_output_dir,
                )
                _maybe_log_training_progress(
                    accelerator,
                    log_interval=log_interval,
                    epoch_id=epoch_id,
                    num_epochs=num_epochs,
                    step_in_epoch=step_in_epoch,
                    steps_per_epoch=None,
                    global_step=global_step,
                    max_steps=max_steps,
                    loss=loss_value,
                    learning_rate=scheduler.get_last_lr()[0],
                    learning_rates=scheduler.get_last_lr(),
                )
                if accelerator.is_local_main_process:
                    progress.set_description(f"epoch {epoch_id + 1}/{num_epochs}")
                    progress.set_postfix(
                        step=str(step_in_epoch),
                        loss=f"{loss_value:.4f}",
                        refresh=False,
                    )
                progress.update(1)
            epoch_id += 1
        progress.close()
        if save_steps is None or global_step % save_steps != 0:
            model_logger.save_model(
                accelerator, model, f"step-{global_step}.safetensors", save_steps=save_steps
            )
    else:
        for epoch_id in range(num_epochs):
            if accelerator.is_main_process:
                print(
                    f"[train] === epoch {epoch_id + 1}/{num_epochs} started ===",
                    flush=True,
                )
            _begin_epoch(dataloader, dataset, epoch_id)
            step_in_epoch = 0
            epoch_bar = tqdm(
                dataloader,
                disable=not accelerator.is_local_main_process,
                desc=f"epoch {epoch_id + 1}/{num_epochs}",
                mininterval=5,
            )
            for data in epoch_bar:
                step_in_epoch += 1
                global_step, loss_value = _training_step(
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    data,
                    model_logger,
                    save_steps,
                    use_wandb,
                    epoch_id,
                    global_step,
                    tensorboard_tracker=tensorboard_tracker,
                    vis_steps=vis_steps,
                    vis_inference_steps=vis_inference_steps,
                    vis_seed=vis_seed,
                    vis_output_dir=vis_output_dir,
                )
                _maybe_log_training_progress(
                    accelerator,
                    log_interval=log_interval,
                    epoch_id=epoch_id,
                    num_epochs=num_epochs,
                    step_in_epoch=step_in_epoch,
                    steps_per_epoch=len(dataloader),
                    global_step=global_step,
                    max_steps=None,
                    loss=loss_value,
                    learning_rate=scheduler.get_last_lr()[0],
                    learning_rates=scheduler.get_last_lr(),
                )
                if accelerator.is_local_main_process:
                    epoch_bar.set_postfix(loss=f"{loss_value:.4f}", refresh=False)

            if save_steps is None:
                model_logger.on_epoch_end(accelerator, model, epoch_id)
        model_logger.on_training_end(accelerator, model, save_steps)

    if use_wandb and accelerator.is_main_process:
        wandb.finish()
    if tensorboard_tracker is not None and accelerator.is_main_process:
        tensorboard_tracker.close()
