"""Mixed multi-dataset training (VGGT-style ComposedDataset / TupleConcatDataset).

Concatenates multiple RoGe datasets with optional per-dataset virtual
lengths (``len_train``). When ``inside_random`` is enabled, each sample is drawn
uniformly from the union of all sub-datasets — use ``shuffle=False`` on the
DataLoader to avoid duplicating large scene lists in worker memory.
"""

from __future__ import annotations

import bisect
from typing import Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset

from .sampling_seed import sample_rng


class SizedSubDataset(Dataset):
    """Wrap a dataset with a virtual length (``len_train``)."""

    def __init__(self, dataset: Dataset, len_train: int | None = None, name: str = ""):
        self.dataset = dataset
        self.name = name
        actual_len = len(dataset)
        if len_train is None:
            len_train = actual_len
        if len_train <= 0:
            raise ValueError(f"len_train must be > 0 for dataset {name!r}, got {len_train}")
        self.len_train = len_train
        self.actual_len = actual_len

    def __len__(self) -> int:
        return self.len_train

    def __getitem__(self, idx: int):
        if self.actual_len <= 0:
            raise IndexError(f"Dataset {self.name!r} has no samples")
        return self.dataset[idx % self.actual_len]


class MixedRoGeDataset(Dataset):
    """Concatenate multiple RoGe datasets for mixed training."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        names: Sequence[str] | None = None,
        len_trains: Sequence[int] | None = None,
        inside_random: bool = False,
    ):
        if not datasets:
            raise ValueError("MixedRoGeDataset requires at least one sub-dataset")

        if names is None:
            names = [f"dataset_{i}" for i in range(len(datasets))]
        if len(names) != len(datasets):
            raise ValueError("names must match the number of sub-datasets")

        if len_trains is None:
            len_trains = [len(d) for d in datasets]
        if len(len_trains) != len(datasets):
            raise ValueError("len_trains must match the number of sub-datasets")

        self.inside_random = inside_random
        self.dataset_names = list(names)
        self.sub_datasets = [
            SizedSubDataset(dataset, len_train=len_train, name=name)
            for dataset, len_train, name in zip(datasets, len_trains, names)
        ]
        self._concat = ConcatDataset(self.sub_datasets)
        self.cumulative_sizes = list(self._concat.cumulative_sizes)

        print(
            "Mixed dataset: "
            + ", ".join(
                f"{name} (virtual={len_train}, actual={sub.actual_len})"
                for name, len_train, sub in zip(names, len_trains, self.sub_datasets)
            )
            + f"; inside_random={inside_random}; total_virtual={len(self)}"
        )

    def __len__(self) -> int:
        return len(self._concat)

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        if self.inside_random:
            idx = sample_rng(salt=0).randint(0, len(self) - 1)

        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def __getitem__(self, idx: int):
        dataset_idx, sample_idx = self._resolve_index(idx)
        sample = self.sub_datasets[dataset_idx][sample_idx]
        if not isinstance(sample, dict):
            return sample

        sample = dict(sample)
        dataset_name = self.dataset_names[dataset_idx]
        scene_name = sample.get("scene_name", "")
        sample["dataset_name"] = dataset_name
        if scene_name and not str(scene_name).startswith(f"{dataset_name}_"):
            sample["scene_name"] = f"{dataset_name}_{scene_name}"
        return sample
