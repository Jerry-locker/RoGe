"""Shared context/target frame sampling for indoor trajectory datasets.

Context: pick a random contiguous segment covering [19/20, 1] of the trajectory,
then K-means on camera pose (position + viewing direction) to select context frames.
Target: pick a random contiguous run from the remaining (non-context) frames.
"""

from __future__ import annotations

import math
import random

import numpy as np

from .sampling_seed import sample_random_state, sample_rng

CONTEXT_SEGMENT_MIN_RATIO = 19 / 20
CONTEXT_SEGMENT_MAX_RATIO = 1.0
# Randomly pick among the nearest ``top_k`` frames per cluster (1 = argmin only).
KMEANS_TOP_K_JITTER = 3


def _as_w2c_4x4(extrinsics_w2c: np.ndarray) -> np.ndarray:
    """Ensure w2c poses are (S, 4, 4)."""
    w2c = np.asarray(extrinsics_w2c, dtype=np.float64)
    if w2c.ndim != 3:
        raise ValueError(f"Expected (S, 3/4, 4) w2c, got shape {w2c.shape}")
    if w2c.shape[-2:] == (3, 4):
        bottom = np.zeros((w2c.shape[0], 1, 4), dtype=w2c.dtype)
        bottom[:, :, 3] = 1.0
        w2c = np.concatenate([w2c, bottom], axis=1)
    elif w2c.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (S, 3/4, 4) w2c, got shape {w2c.shape}")
    return w2c


def camera_pose_features_from_w2c(extrinsics_w2c: np.ndarray) -> np.ndarray:
    """6D pose features for K-means (SEVA / Long-LRM / VGGT demo_viser protocol)."""
    w2c = _as_w2c_4x4(extrinsics_w2c)
    cam_to_world = np.linalg.inv(w2c)
    positions = cam_to_world[:, :3, 3]
    directions = cam_to_world[:, :3, 2]
    directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
    return np.concatenate([positions, directions], axis=1)


def kmeans_frame_indices(
    extrinsics_w2c: np.ndarray,
    num: int,
    random_state: int = 42,
    top_k_jitter: int = KMEANS_TOP_K_JITTER,
) -> np.ndarray:
    """Pick ``num`` frame indices via K-means on 6D camera pose features.

    Each cluster center maps to one frame. When ``top_k_jitter > 1``, the frame
    is chosen uniformly from the ``top_k_jitter`` nearest candidates (still near
    the center, but not always the single argmin).
    """
    num_frames = extrinsics_w2c.shape[0]
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if num <= 0:
        raise ValueError(f"num must be positive, got {num}")
    if top_k_jitter <= 0:
        raise ValueError(f"top_k_jitter must be > 0, got {top_k_jitter}")
    if num >= num_frames:
        return np.arange(num_frames, dtype=np.int64)

    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError(
            "K-means frame sampling requires scikit-learn. "
            "Install with: pip install scikit-learn"
        ) from exc

    features = camera_pose_features_from_w2c(extrinsics_w2c)
    kmeans = KMeans(n_clusters=num, random_state=random_state)
    kmeans.fit(features)
    centers = kmeans.cluster_centers_

    ids: list[int] = []
    used: set[int] = set()
    for cluster_idx, center in enumerate(centers):
        distances = np.linalg.norm(features - center, axis=1)
        for used_idx in used:
            distances[used_idx] = np.inf

        if top_k_jitter <= 1:
            pick = int(np.argmin(distances))
        else:
            k = min(top_k_jitter, num_frames - len(used))
            nearest_k = np.argpartition(distances, k - 1)[:k]
            pick = int(
                random.Random(random_state + cluster_idx * 1009).choice(
                    nearest_k.tolist()
                )
            )

        ids.append(pick)
        used.add(pick)

    return np.sort(np.array(ids, dtype=np.int64))


def _resolve_kmeans_seed(random_state: int | None, *, salt: int) -> int:
    if random_state is not None:
        return int(random_state)
    return sample_random_state(salt=salt)


def _sample_segment_length(
    stream_len: int,
    num_context: int,
    *,
    min_ratio: float = CONTEXT_SEGMENT_MIN_RATIO,
    max_ratio: float = CONTEXT_SEGMENT_MAX_RATIO,
    rng: random.Random | None = None,
) -> int | None:
    min_len = max(num_context, math.ceil(stream_len * min_ratio))
    max_len = min(stream_len, math.ceil(stream_len * max_ratio))
    if min_len > max_len:
        return None
    if rng is None:
        rng = sample_rng(salt=1)
    return rng.randint(min_len, max_len)


def _valid_target_starts(
    stream_len: int,
    context_positions: np.ndarray,
    num_target: int,
) -> list[int]:
    context_set = set(context_positions.tolist())
    return [
        start
        for start in range(0, stream_len - num_target + 1)
        if all(pos not in context_set for pos in range(start, start + num_target))
    ]


def stream_supports_kmeans_sampling(
    stream_len: int,
    num_context: int,
    num_target: int,
    *,
    min_ratio: float = CONTEXT_SEGMENT_MIN_RATIO,
    max_ratio: float = CONTEXT_SEGMENT_MAX_RATIO,
) -> bool:
    """Conservative pre-filter: stream length and segment budget are sufficient."""
    if stream_len < num_context + num_target:
        return False
    min_len = max(num_context, math.ceil(stream_len * min_ratio))
    max_len = min(stream_len, math.ceil(stream_len * max_ratio))
    return min_len <= max_len


def sample_kmeans_context_positions(
    stream_len: int,
    extrinsics_w2c: np.ndarray,
    num_context: int,
    random_state: int | None = None,
) -> np.ndarray | None:
    """K-means context frame positions on the full trajectory."""
    if stream_len < num_context:
        return None
    if extrinsics_w2c.shape[0] != stream_len:
        raise ValueError(
            f"extrinsics_w2c length ({extrinsics_w2c.shape[0]}) != stream_len ({stream_len})"
        )
    kmeans_seed = _resolve_kmeans_seed(random_state, salt=3)
    return kmeans_frame_indices(extrinsics_w2c, num_context, random_state=kmeans_seed)


def sample_kmeans_context_in_segment(
    stream_len: int,
    extrinsics_w2c: np.ndarray,
    num_context: int,
    *,
    min_ratio: float = CONTEXT_SEGMENT_MIN_RATIO,
    max_ratio: float = CONTEXT_SEGMENT_MAX_RATIO,
    random_state: int | None = None,
) -> np.ndarray | None:
    """K-means context frames inside a random contiguous trajectory segment."""
    if stream_len <= 0:
        return None
    if extrinsics_w2c.shape[0] != stream_len:
        raise ValueError(
            f"extrinsics_w2c length ({extrinsics_w2c.shape[0]}) != stream_len ({stream_len})"
        )
    min_len = max(num_context, math.ceil(stream_len * min_ratio))
    max_len = min(stream_len, math.ceil(stream_len * max_ratio))
    if min_len > max_len:
        return None

    rng = sample_rng(salt=0) if random_state is None else random.Random(random_state)
    seg_len = rng.randint(min_len, max_len)
    seg_start = rng.randint(0, stream_len - seg_len)
    segment_w2c = extrinsics_w2c[seg_start : seg_start + seg_len]

    kmeans_seed = _resolve_kmeans_seed(random_state, salt=3)
    local_context = kmeans_frame_indices(
        segment_w2c, num_context, random_state=kmeans_seed
    )
    return seg_start + local_context


def sample_equal_partition_context_positions(
    stream_len: int,
    num_context: int,
    random_state: int | None = None,
) -> np.ndarray | None:
    """Pick one random frame from each of ``num_context`` equal time partitions."""
    if stream_len < num_context:
        return None

    rng = sample_rng(salt=20) if random_state is None else random.Random(random_state)
    positions: list[int] = []
    for part_idx in range(num_context):
        part_start = (part_idx * stream_len) // num_context
        part_end = ((part_idx + 1) * stream_len) // num_context
        if part_end <= part_start:
            return None
        positions.append(rng.randrange(part_start, part_end))
    return np.sort(np.asarray(positions, dtype=np.int64))


def sample_uniform_context_positions(
    stream_len: int,
    num_context: int,
    random_state: int | None = None,
) -> np.ndarray | None:
    """Pick uniformly spaced frames with a random valid temporal offset."""
    if stream_len < num_context or num_context <= 0:
        return None

    rng = sample_rng(salt=22) if random_state is None else random.Random(random_state)
    interval = stream_len // num_context
    max_start = stream_len - 1 - (num_context - 1) * interval
    start = rng.randint(0, max_start)
    return start + np.arange(num_context, dtype=np.int64) * interval


def sample_interval_target_from_complement(
    remaining_positions: np.ndarray,
    num_target: int,
    *,
    max_interval: int = 5,
    random_state: int | None = None,
) -> np.ndarray | None:
    """
    Pick ordered target frames from complement indices with random gaps
    (1..max_interval) between consecutive picks in complement index space.
    """
    pool_len = len(remaining_positions)
    if pool_len < num_target or num_target <= 0:
        return None
    if max_interval <= 0:
        raise ValueError(f"max_interval must be > 0, got {max_interval}")

    rng = sample_rng(salt=21) if random_state is None else random.Random(random_state)
    offsets = [0]
    remaining_budget = pool_len - 1
    for gap_idx in range(num_target - 1):
        remaining_gaps = num_target - gap_idx - 2
        max_gap = min(max_interval, remaining_budget - remaining_gaps)
        if max_gap < 1:
            return None
        gap = rng.randint(1, max_gap)
        offsets.append(offsets[-1] + gap)
        remaining_budget -= gap
    offsets_arr = np.asarray(offsets, dtype=np.int64)

    span = int(offsets_arr[-1]) + 1
    if span > pool_len:
        return None
    start = rng.randint(0, pool_len - span)
    return remaining_positions[start + offsets_arr]


def sample_consecutive_target_positions(
    stream_len: int,
    num_target: int,
    rng: random.Random | None = None,
) -> np.ndarray | None:
    """Random contiguous target frame positions along a stream."""
    if stream_len < num_target:
        return None
    if rng is None:
        rng = sample_rng(salt=4)
    target_start = rng.randint(0, stream_len - num_target)
    return np.arange(target_start, target_start + num_target, dtype=np.int64)


def sample_interval_target_positions(
    stream_len: int,
    num_target: int,
    *,
    interval: int | None = 1,
    max_interval: int = 5,
    rng: random.Random | None = None,
) -> np.ndarray | None:
    """Sample an ordered target sequence with fixed or random frame intervals."""
    if stream_len <= 0 or num_target <= 0:
        return None
    if rng is None:
        rng = sample_rng(salt=9)

    if interval is not None:
        if interval <= 0:
            raise ValueError(f"interval must be > 0, got {interval}")
        offsets = np.arange(num_target, dtype=np.int64) * interval
    else:
        if max_interval <= 0:
            raise ValueError(f"max_interval must be > 0, got {max_interval}")
        if stream_len < num_target:
            return None
        offsets = [0]
        remaining_budget = stream_len - 1
        for gap_idx in range(num_target - 1):
            remaining_gaps = num_target - gap_idx - 2
            max_gap = min(max_interval, remaining_budget - remaining_gaps)
            if max_gap < 1:
                return None
            gap = rng.randint(1, max_gap)
            offsets.append(offsets[-1] + gap)
            remaining_budget -= gap
        offsets = np.asarray(offsets, dtype=np.int64)

    span = int(offsets[-1]) + 1
    if span > stream_len:
        return None
    start = rng.randint(0, stream_len - span)
    return start + offsets


def sample_target_first_kmeans_context_positions(
    stream_len: int,
    extrinsics_w2c: np.ndarray,
    num_context: int,
    num_target: int,
    *,
    interval: int | None = 1,
    max_interval: int = 5,
    kmeans_probability: float = 1.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Sample targets first, then K-means or uniform context from the complement."""
    if extrinsics_w2c.shape[0] != stream_len:
        raise ValueError(
            f"extrinsics_w2c length ({extrinsics_w2c.shape[0]}) != stream_len ({stream_len})"
        )
    if not 0.0 <= kmeans_probability <= 1.0:
        raise ValueError(
            f"kmeans_probability must be in [0, 1], got {kmeans_probability}"
        )
    if stream_len < num_context + num_target:
        return None

    rng = sample_rng(salt=10) if random_state is None else random.Random(random_state)
    target_positions = sample_interval_target_positions(
        stream_len,
        num_target,
        interval=interval,
        max_interval=max_interval,
        rng=rng,
    )
    if target_positions is None:
        return None

    target_set = set(target_positions.tolist())
    remaining_positions = np.asarray(
        [pos for pos in range(stream_len) if pos not in target_set],
        dtype=np.int64,
    )
    if len(remaining_positions) < num_context:
        return None

    context_rng = (
        sample_rng(salt=12) if random_state is None else random.Random(random_state + 1)
    )
    if context_rng.random() < kmeans_probability:
        kmeans_seed = _resolve_kmeans_seed(random_state, salt=11)
        local_context = kmeans_frame_indices(
            extrinsics_w2c[remaining_positions],
            num_context,
            random_state=kmeans_seed,
        )
    else:
        uniform_seed = None if random_state is None else random_state + 2
        local_context = sample_uniform_context_positions(
            len(remaining_positions),
            num_context,
            random_state=uniform_seed,
        )
        if local_context is None:
            return None
    return remaining_positions[local_context], target_positions


def sample_kmeans_context_target_positions(
    stream_len: int,
    extrinsics_w2c: np.ndarray,
    num_context: int,
    num_target: int,
    *,
    min_ratio: float = CONTEXT_SEGMENT_MIN_RATIO,
    max_ratio: float = CONTEXT_SEGMENT_MAX_RATIO,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Sample context and target positions along a temporal stream.

    Returns:
        (context_positions, target_positions) as ascending int arrays in [0, stream_len),
        or None when sampling fails.
    """
    if stream_len <= 0:
        return None
    if extrinsics_w2c.shape[0] != stream_len:
        raise ValueError(
            f"extrinsics_w2c length ({extrinsics_w2c.shape[0]}) != stream_len ({stream_len})"
        )
    if not stream_supports_kmeans_sampling(
        stream_len, num_context, num_target, min_ratio=min_ratio, max_ratio=max_ratio
    ):
        return None

    rng = sample_rng(salt=0) if random_state is None else random.Random(random_state)
    seg_len = _sample_segment_length(
        stream_len, num_context, min_ratio=min_ratio, max_ratio=max_ratio, rng=rng
    )
    if seg_len is None:
        return None

    seg_start = rng.randint(0, stream_len - seg_len)
    seg_end = seg_start + seg_len
    segment_w2c = extrinsics_w2c[seg_start:seg_end]

    kmeans_seed = _resolve_kmeans_seed(random_state, salt=3)
    local_context = kmeans_frame_indices(segment_w2c, num_context, random_state=kmeans_seed)
    context_positions = seg_start + local_context

    valid_starts = _valid_target_starts(stream_len, context_positions, num_target)
    if not valid_starts:
        return None

    target_start = rng.choice(valid_starts)
    target_positions = np.arange(target_start, target_start + num_target, dtype=np.int64)
    return context_positions, target_positions


def sample_kmeans_context_complement_target_positions(
    stream_len: int,
    extrinsics_w2c: np.ndarray,
    num_context: int,
    num_target: int,
    *,
    segment_len: int | None = None,
    min_ratio: float = CONTEXT_SEGMENT_MIN_RATIO,
    max_ratio: float = CONTEXT_SEGMENT_MAX_RATIO,
    target_within_segment: bool = False,
    kmeans_probability: float = 1.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    K-means or uniform context in a random segment, then slice complement targets.

    "Consecutive" targets here are consecutive entries in the ordered complement,
    so their original frame indices may skip context positions.
    """
    if stream_len <= 0:
        return None
    if extrinsics_w2c.shape[0] != stream_len:
        raise ValueError(
            f"extrinsics_w2c length ({extrinsics_w2c.shape[0]}) != stream_len ({stream_len})"
        )
    if not 0.0 <= kmeans_probability <= 1.0:
        raise ValueError(
            f"kmeans_probability must be in [0, 1], got {kmeans_probability}"
        )
    rng = sample_rng(salt=17) if random_state is None else random.Random(random_state)

    if segment_len is None:
        seg_len = _sample_segment_length(
            stream_len,
            num_context,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            rng=rng,
        )
    else:
        seg_len = segment_len if 0 < segment_len <= stream_len else None
    if seg_len is None or seg_len < num_context:
        return None

    seg_start = rng.randint(0, stream_len - seg_len)
    segment_w2c = extrinsics_w2c[seg_start : seg_start + seg_len]
    context_rng = (
        sample_rng(salt=19) if random_state is None else random.Random(random_state + 1)
    )
    if context_rng.random() < kmeans_probability:
        kmeans_seed = _resolve_kmeans_seed(random_state, salt=18)
        local_context = kmeans_frame_indices(
            segment_w2c,
            num_context,
            random_state=kmeans_seed,
        )
    else:
        uniform_seed = None if random_state is None else random_state + 2
        local_context = sample_uniform_context_positions(
            seg_len,
            num_context,
            random_state=uniform_seed,
        )
        if local_context is None:
            return None
    context_positions = seg_start + local_context

    if target_within_segment:
        target_pool = np.arange(seg_start, seg_start + seg_len, dtype=np.int64)
    else:
        target_pool = np.arange(stream_len, dtype=np.int64)
    context_set = set(context_positions.tolist())
    remaining_positions = np.asarray(
        [pos for pos in target_pool if pos not in context_set],
        dtype=np.int64,
    )
    if len(remaining_positions) < num_target:
        return None

    target_start = rng.randint(0, len(remaining_positions) - num_target)
    target_positions = remaining_positions[target_start : target_start + num_target]
    return context_positions, target_positions


def stream_supports_fixed_segment_sampling(
    stream_len: int,
    segment_len: int,
    num_context: int,
    num_target: int,
) -> bool:
    """Pre-filter: stream can host a fixed-length segment with context + target."""
    if segment_len < num_context + num_target:
        return False
    return stream_len >= segment_len


def sample_fixed_segment_kmeans_context_target_positions(
    stream_len: int,
    extrinsics_w2c: np.ndarray,
    num_context: int,
    num_target: int,
    segment_len: int,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Sample within a random contiguous segment of fixed length.

    Context: K-means ``num_context`` frames inside the segment (sorted by time).
    Target: ``num_target`` consecutive frames from the remaining segment positions.
    """
    if stream_len <= 0:
        return None
    if extrinsics_w2c.shape[0] != stream_len:
        raise ValueError(
            f"extrinsics_w2c length ({extrinsics_w2c.shape[0]}) != stream_len ({stream_len})"
        )
    if not stream_supports_fixed_segment_sampling(
        stream_len, segment_len, num_context, num_target
    ):
        return None

    rng = sample_rng(salt=7) if random_state is None else random.Random(random_state)
    seg_start = rng.randint(0, stream_len - segment_len)
    segment_w2c = extrinsics_w2c[seg_start : seg_start + segment_len]

    kmeans_seed = _resolve_kmeans_seed(random_state, salt=8)
    local_context = kmeans_frame_indices(segment_w2c, num_context, random_state=kmeans_seed)
    context_positions = seg_start + local_context

    valid_starts = _valid_target_starts(segment_len, local_context, num_target)
    if not valid_starts:
        return None

    target_start_local = rng.choice(valid_starts)
    target_positions = seg_start + np.arange(
        target_start_local,
        target_start_local + num_target,
        dtype=np.int64,
    )
    return context_positions, target_positions


def sample_head_tail_target_kmeans_context_positions(
    stream_len: int,
    extrinsics_w2c: np.ndarray,
    num_context: int,
    num_target: int,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Target: first or last ``num_target`` consecutive frames (50/50).
    Context: K-means ``num_context`` frames from the remaining stream positions.
    """
    if stream_len < num_context + num_target:
        return None
    if extrinsics_w2c.shape[0] != stream_len:
        raise ValueError(
            f"extrinsics_w2c length ({extrinsics_w2c.shape[0]}) != stream_len ({stream_len})"
        )

    rng = sample_rng(salt=5) if random_state is None else random.Random(random_state)
    if rng.random() < 0.5:
        target_positions = np.arange(0, num_target, dtype=np.int64)
    else:
        target_positions = np.arange(stream_len - num_target, stream_len, dtype=np.int64)

    target_set = set(target_positions.tolist())
    remaining_positions = np.array(
        [pos for pos in range(stream_len) if pos not in target_set],
        dtype=np.int64,
    )
    if len(remaining_positions) < num_context:
        return None

    kmeans_seed = _resolve_kmeans_seed(random_state, salt=6)
    local_context = kmeans_frame_indices(
        extrinsics_w2c[remaining_positions],
        num_context,
        random_state=kmeans_seed,
    )
    context_positions = remaining_positions[local_context]
    return context_positions, target_positions
