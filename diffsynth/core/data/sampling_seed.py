"""Deterministic per-step dataset sampling for reproducible training."""

from __future__ import annotations

import contextvars
import random
from contextlib import contextmanager
from functools import partial
from typing import Iterator

import numpy as np
import torch

MAX_SAMPLE_RETRIES = 32

# When --mixed_train is on, every dataset draws M uniformly from this range.
# M is keyed by a rank-independent mixed step carried by the dataset index, so
# worker scheduling and prefetch cannot change which M a training step receives.
MIXED_TRAIN_CONTEXT_MIN = 3
MIXED_TRAIN_CONTEXT_MAX = 9
MIXED_TRAIN_M_SALT = 9041


class ResampleSample(Exception):
    """Signal that sampling failed and should retry with the next retry salt."""

_training_ctx: contextvars.ContextVar[tuple[int, int, int, int] | None] = contextvars.ContextVar(
    "training_sample_ctx", default=None
)
_retry_ctx: contextvars.ContextVar[int] = contextvars.ContextVar("training_sample_retry", default=0)
_mixed_step_ctx: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "training_mixed_step", default=None
)


def derive_sample_seed(
    base_seed: int,
    rank: int,
    epoch: int,
    idx: int,
    *,
    salt: int = 0,
    retry: int = 0,
) -> int:
    """Stable 31-bit seed from training coordinates and optional salt/retry."""
    value = int(base_seed) & 0x7FFFFFFF
    for part in (rank, epoch, idx, salt, retry):
        value = (value * 1_000_003 + int(part)) & 0x7FFFFFFF
    return value or 1


def sample_rng(*, salt: int = 0) -> random.Random:
    """RNG for the current dataloader index (requires active sample context)."""
    ctx = _training_ctx.get()
    if ctx is None:
        raise RuntimeError(
            "sample_rng() called without an active training sample context. "
            "Wrap the dataset with ReproducibleTrainingDataset during training."
        )
    base_seed, rank, epoch, idx = ctx
    retry = _retry_ctx.get()
    return random.Random(
        derive_sample_seed(base_seed, rank, epoch, idx, salt=salt, retry=retry)
    )


def sample_random_state(*, salt: int = 0) -> int:
    """Integer seed suitable for sklearn KMeans(random_state=...)."""
    ctx = _training_ctx.get()
    if ctx is None:
        raise RuntimeError(
            "sample_random_state() called without an active training sample context."
        )
    base_seed, rank, epoch, idx = ctx
    retry = _retry_ctx.get()
    return derive_sample_seed(base_seed, rank, epoch, idx, salt=salt, retry=retry)


@contextmanager
def sample_seed_context(
    base_seed: int,
    rank: int,
    epoch: int,
    idx: int,
) -> Iterator[None]:
    ctx_token = _training_ctx.set((int(base_seed), int(rank), int(epoch), int(idx)))
    retry_token = _retry_ctx.set(0)
    try:
        yield
    finally:
        _retry_ctx.reset(retry_token)
        _training_ctx.reset(ctx_token)


@contextmanager
def sample_mixed_step_context(step: int) -> Iterator[None]:
    """Bind the rank-shared mixed-train step for the current sample."""
    token = _mixed_step_ctx.set(int(step))
    try:
        yield
    finally:
        _mixed_step_ctx.reset(token)


@contextmanager
def sample_retry_scope(retry: int) -> Iterator[None]:
    token = _retry_ctx.set(int(retry))
    try:
        yield
    finally:
        _retry_ctx.reset(token)


def mixed_train_max_context_frames(mixed_train: bool, fixed: int) -> int:
    """Upper bound of the mixed_train context draw range (``MIXED_TRAIN_CONTEXT_MAX``)."""
    if mixed_train:
        return MIXED_TRAIN_CONTEXT_MAX
    return int(fixed)


def mixed_train_filter_context_frames(mixed_train: bool, fixed: int) -> int:
    """Context count used for scene-length gating.

    Under ``mixed_train``, scenes only need to support ``MIXED_TRAIN_CONTEXT_MIN``
    (+ targets). Larger drawn M values are clamped per-scene at sample time.
    """
    if mixed_train:
        return MIXED_TRAIN_CONTEXT_MIN
    return int(fixed)


def clamp_mixed_context_frames(
    num_context: int,
    stream_len: int,
    num_target: int,
    *,
    mixed_train: bool,
) -> int:
    """Clamp drawn M so ``M + num_target`` fits ``stream_len`` (mixed_train only)."""
    if not mixed_train:
        return int(num_context)
    max_fit = int(stream_len) - int(num_target)
    if max_fit < MIXED_TRAIN_CONTEXT_MIN:
        return int(num_context)
    return min(int(num_context), max_fit)


def resolve_num_context_frames(fixed: int, mixed_train: bool) -> int:
    """Return M for this sample.

    With ``mixed_train``, M ~ Uniform{3..9} derived from
    ``(seed, epoch, mixed_step)`` with rank forced to 0, so every DDP rank draws
    the same M for the same training step. Without an active mixed-step context
    (e.g. dataset check),
    falls back to ``fixed``. Callers should pass the result through
    ``clamp_mixed_context_frames`` once ``stream_len`` is known.
    """
    if not mixed_train:
        return int(fixed)
    ctx = _training_ctx.get()
    mixed_step = _mixed_step_ctx.get()
    if ctx is None or mixed_step is None:
        return int(fixed)
    base_seed, _rank, epoch, _idx = ctx
    seed = derive_sample_seed(
        base_seed,
        0,
        epoch,
        mixed_step,
        salt=MIXED_TRAIN_M_SALT,
        retry=0,
    )
    return random.Random(seed).randint(MIXED_TRAIN_CONTEXT_MIN, MIXED_TRAIN_CONTEXT_MAX)


def set_training_seeds(base_seed: int, rank: int, *, max_epochs: int = 10_000) -> None:
    """Initialize global RNGs once per process (model/noise reproducibility helper)."""
    seed = derive_sample_seed(base_seed, rank, 0, 0, salt=max_epochs)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int, num_workers: int, base_seed: int, rank: int, epoch: int) -> None:
    """Seed a DataLoader worker; sampling still uses per-index context in __getitem__."""
    worker_seed = derive_sample_seed(base_seed, rank, epoch, worker_id, salt=num_workers * 997)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    # Datasets resize with cv2; its thread pool would otherwise oversubscribe
    # the box num_workers-fold. Area/linear resize is thread-count invariant.
    try:
        import cv2

        cv2.setNumThreads(0)
    except ImportError:
        pass


def build_worker_init_fn(base_seed: int, num_workers: int, rank: int, epoch: int):
    return partial(
        worker_init_fn,
        num_workers=num_workers,
        base_seed=base_seed,
        rank=rank,
        epoch=epoch,
    )
