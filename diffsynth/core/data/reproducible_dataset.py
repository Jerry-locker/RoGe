"""Wrap training datasets so each (rank, epoch, dataloader idx) maps to fixed samples."""

from __future__ import annotations

from torch.utils.data import Dataset

from .sampling_seed import (
    MAX_SAMPLE_RETRIES,
    ResampleSample,
    sample_mixed_step_context,
    sample_retry_scope,
    sample_seed_context,
)


class ReproducibleTrainingDataset(Dataset):
    """Bind dataset randomness to (seed, rank, epoch, DataLoader index).

    For synchronized mixed training, the exposed length is padded to a multiple
    of the world size. With an unshuffled Accelerate DataLoader, rank ``r`` gets
    logical index ``step * world_size + r``. Therefore ``idx // world_size`` is
    a stable rank-shared step independent of worker execution and prefetch order.
    """

    def __init__(self, dataset: Dataset, seed: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.rank = 0
        self.num_processes = 1
        self.epoch = 0
        self.synchronize_mixed_steps = False

    def set_training_context(
        self,
        rank: int,
        num_processes: int = 1,
        *,
        synchronize_mixed_steps: bool = False,
    ) -> None:
        self.rank = int(rank)
        self.num_processes = max(int(num_processes), 1)
        self.synchronize_mixed_steps = bool(synchronize_mixed_steps)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        if hasattr(self.dataset, "current_epoch"):
            self.dataset.current_epoch = epoch

    def __len__(self):
        length = len(self.dataset)
        if not self.synchronize_mixed_steps:
            return length
        # Avoid Accelerate padding the last DDP step with repeated low indices,
        # whose idx // world_size would otherwise disagree with the other ranks.
        return ((length + self.num_processes - 1) // self.num_processes) * self.num_processes

    def __getitem__(self, idx: int):
        idx = int(idx)
        dataset_idx = idx % len(self.dataset)
        mixed_step = idx // self.num_processes
        last_error: Exception | None = None
        for retry in range(MAX_SAMPLE_RETRIES):
            with sample_seed_context(self.seed, self.rank, self.epoch, idx):
                with sample_mixed_step_context(mixed_step):
                    with sample_retry_scope(retry):
                        try:
                            return self.dataset[dataset_idx]
                        except ResampleSample as exc:
                            last_error = exc
                            continue
        if last_error is not None:
            raise last_error
        raise RuntimeError(
            f"Failed to draw a training sample for idx={idx} after "
            f"{MAX_SAMPLE_RETRIES} retries"
        )

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)


def wrap_for_reproducible_training(dataset: Dataset, seed: int) -> Dataset:
    if isinstance(dataset, ReproducibleTrainingDataset):
        return dataset
    return ReproducibleTrainingDataset(dataset, seed=seed)
