"""Disk cache for dataset discovery / scene-index builds.

Cache files are keyed only by ``(dataset_name, base_path)``. Use
``--update_dataset_cache`` to force a rescan and overwrite existing files.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger("trainer")

T = TypeVar("T")


def resolve_cache_dir(cache_dir: str) -> str:
    if not cache_dir or not str(cache_dir).strip():
        raise ValueError("--dataset_cache_dir is required and must be non-empty")
    return os.path.abspath(os.path.expanduser(cache_dir.strip()))


def _base_path_digest(base_path: str) -> str:
    return hashlib.sha256(os.path.abspath(base_path).encode()).hexdigest()[:16]


def cache_file_path(
    dataset_name: str,
    base_path: str,
    *,
    cache_dir: str,
) -> str:
    digest = _base_path_digest(base_path)
    return os.path.join(resolve_cache_dir(cache_dir), f"{dataset_name}_{digest}.pkl")


def _load_payload(path: str) -> list[Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError, ValueError) as exc:
        logger.warning(f"Dataset cache read failed ({path}): {exc}")
        return None

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        scenes = payload.get("scenes")
        if isinstance(scenes, list):
            return scenes
    logger.warning(f"Dataset cache has unexpected format: {path}")
    return None


def _save_payload(path: str, scenes: list[Any]) -> None:
    """Atomically write ``scenes`` to ``path``.

    Multi-node launches share one ``--dataset_cache_dir`` on NFS. ``fcntl``
    locks are unreliable there, so several ranks may build and save the same
    key. Treat replace races as success when another writer already left a
    readable cache.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(scenes, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        if _load_payload(path) is not None:
            logger.warning(
                f"Dataset cache save raced ({path}): {exc}; using existing cache"
            )
            return
        raise
    finally:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _acquire_build_lock(lock_path: str, *, timeout_sec: float = 7200.0):
    import fcntl

    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lock_file = open(lock_path, "a+", encoding="utf-8")
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                lock_file.close()
                raise TimeoutError(
                    f"Timed out waiting for dataset cache lock: {lock_path}"
                )
            time.sleep(0.5)
    return lock_file


def _release_build_lock(lock_file) -> None:
    import fcntl

    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def build_scene_list_with_cache(
    *,
    dataset_name: str,
    base_path: str,
    cache_dir: str,
    build_fn: Callable[[], list[T]],
    update_cache: bool = False,
) -> tuple[list[T], bool]:
    """Return ``(scenes, loaded_from_cache)``."""
    cache_path = cache_file_path(dataset_name, base_path, cache_dir=cache_dir)
    lock_path = f"{cache_path}.lock"

    if update_cache and os.path.isfile(cache_path):
        try:
            os.remove(cache_path)
        except OSError as exc:
            logger.warning(f"Failed to remove dataset cache {cache_path}: {exc}")

    if not update_cache:
        scenes = _load_payload(cache_path)
        if scenes is not None:
            logger.info(
                f"{dataset_name}: loaded {len(scenes)} scenes from cache ({cache_path})"
            )
            return scenes, True

    lock_file = _acquire_build_lock(lock_path)
    try:
        if not update_cache:
            scenes = _load_payload(cache_path)
            if scenes is not None:
                logger.info(
                    f"{dataset_name}: loaded {len(scenes)} scenes from cache "
                    f"after waiting ({cache_path})"
                )
                return scenes, True

        action = "updating cache" if update_cache else "cache miss"
        logger.info(f"{dataset_name}: building scene index ({action}: {cache_path})")
        scenes = build_fn()
        # Only rank 0 is required to persist; other ranks may race-save as a
        # best effort (torchrun sets RANK before Accelerator is created).
        rank = int(os.environ.get("RANK", "0"))
        try:
            if rank == 0 or not os.path.isfile(cache_path):
                _save_payload(cache_path, scenes)
                logger.info(
                    f"{dataset_name}: saved {len(scenes)} scenes to cache ({cache_path})"
                )
            else:
                logger.info(
                    f"{dataset_name}: skip cache save on rank {rank}; "
                    f"file already present ({cache_path})"
                )
        except OSError as exc:
            cached = _load_payload(cache_path)
            if cached is not None:
                logger.warning(
                    f"{dataset_name}: cache save failed ({exc}); "
                    f"using {len(cached)} scenes written by another rank"
                )
                return cached, True
            raise
        return scenes, False
    finally:
        _release_build_lock(lock_file)
