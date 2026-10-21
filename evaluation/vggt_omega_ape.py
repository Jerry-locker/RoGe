"""VGGT-Omega pose estimation + evo APE RMSE vs GT cameras.

Runs VGGT-Omega on pred frames, decodes camera-from-world extrinsics to
OpenCV c2w, and compares trajectories against ``transforms.json`` GT via
``colmap_ape.compute_ape_sim3`` (GT path-length normalization, then evo
translational + rotational APE with Sim(3) Umeyama ``align + correct_scale``).
"""

from __future__ import annotations

import multiprocessing
import queue
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from evaluation.colmap_ape import compute_ape_sim3
from evaluation.paths import REPO_ROOT

VGGT_OMEGA_ROOT = REPO_ROOT / "third_party" / "vggt-omega"
DEFAULT_IMAGE_RESOLUTION = 512
DEFAULT_PREPROCESS_MODE = "balanced"
DEFAULT_VGGT_OMEGA_CHECKPOINT = (
    REPO_ROOT / "models" / "vggt-omega" / "vggt_omega_1b_512.pt"
)


def ensure_vggt_omega_on_path() -> None:
    """Prefer vendored ``third_party/vggt-omega/`` on ``sys.path``."""
    if not VGGT_OMEGA_ROOT.is_dir():
        return
    root = str(VGGT_OMEGA_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass(frozen=True)
class VggtOmegaApeResult:
    ape_rmse: float
    rotation_ape: float
    n_images: int
    n_poses: int
    n_ape_poses: int
    n_chunks: int = 0
    n_nan_chunks: int = 0


def is_nan_ape_chunk(result: VggtOmegaApeResult) -> bool:
    return float(result.ape_rmse) != float(result.ape_rmse)


def nan_ape_result(n_images: int = 0, *, n_poses: int = 0) -> VggtOmegaApeResult:
    return VggtOmegaApeResult(
        ape_rmse=float("nan"),
        rotation_ape=float("nan"),
        n_images=n_images,
        n_poses=n_poses,
        n_ape_poses=0,
    )


def w2c_3x4_to_c2w(w2c_3x4: np.ndarray) -> np.ndarray:
    """OpenCV cam-from-world 3x4 -> camera-to-world 4x4."""
    w2c = np.asarray(w2c_3x4, dtype=np.float64)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


class VggtOmegaPoseEstimator:
    """Load VGGT-Omega once and estimate c2w poses for image sequences."""

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: Path,
        *,
        image_resolution: int = DEFAULT_IMAGE_RESOLUTION,
        preprocess_mode: str = DEFAULT_PREPROCESS_MODE,
        enable_alignment: bool = False,
    ) -> None:
        if device.type != "cuda":
            raise RuntimeError("VGGT-Omega APE currently expects a CUDA device.")
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint_path}")
        if preprocess_mode not in {"balanced", "max_size"}:
            raise ValueError(f"Unknown preprocess mode: {preprocess_mode}")

        ensure_vggt_omega_on_path()
        try:
            from vggt_omega.models import VGGTOmega
            from vggt_omega.utils.load_fn import load_and_preprocess_images
            from vggt_omega.utils.pose_enc import encoding_to_camera
        except ImportError as exc:
            raise ImportError(
                "vggt-omega is required for APE pose estimation.\n"
                f"Install vendored copy: pip install {VGGT_OMEGA_ROOT}\n"
                "Or pass --skip_ape to skip."
            ) from exc

        self.device = device
        self.image_resolution = int(image_resolution)
        self.preprocess_mode = preprocess_mode
        self._load_and_preprocess_images = load_and_preprocess_images
        self._encoding_to_camera = encoding_to_camera

        model = VGGTOmega(
            enable_camera=True,
            enable_depth=False,
            enable_alignment=enable_alignment,
        ).eval()
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(
                f"[vggt-omega] ignored {len(unexpected)} unexpected checkpoint key(s)",
                flush=True,
            )
        if missing:
            print(
                f"[vggt-omega] missing {len(missing)} checkpoint key(s) "
                f"(expected when enable_depth=False)",
                flush=True,
            )
        self.model = model.to(device)
        print(
            f"[vggt-omega] model loaded on {device} "
            f"(resolution={self.image_resolution}, preprocess={self.preprocess_mode})",
            flush=True,
        )

    def release_gpu(self) -> None:
        """Move pose model off GPU so CD (or another process) can use the device."""
        if self.device.type == "cuda":
            self.model.cpu()
            torch.cuda.empty_cache()

    def ensure_gpu(self) -> None:
        if self.device.type == "cuda":
            self.model.to(self.device)

    @torch.inference_mode()
    def _forward_pose_only(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run aggregator + camera head only (skip depth head)."""
        if images.ndim == 4:
            images = images.unsqueeze(0)
        images = images.float()
        aggregated_tokens_list, patch_token_start = self.model.aggregator(images)
        pose_enc = self.model.camera_head(
            aggregated_tokens_list,
            patch_token_start=patch_token_start,
        )
        return {"pose_enc": pose_enc, "images": images}

    @torch.no_grad()
    def estimate_c2ws(self, image_paths: list[Path]) -> list[np.ndarray]:
        """Return OpenCV c2w 4x4 poses in the same order as ``image_paths``."""
        if not image_paths:
            return []
        self.ensure_gpu()
        print(
            f"[vggt-omega] preprocessing {len(image_paths)} image(s)...",
            flush=True,
        )
        images = self._load_and_preprocess_images(
            [str(path) for path in image_paths],
            mode=self.preprocess_mode,
            image_resolution=self.image_resolution,
        ).to(self.device)
        print(
            f"[vggt-omega] forward on shape={tuple(images.shape)} dtype={images.dtype}",
            flush=True,
        )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            sdp_ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=True,
                enable_math=True,
            )
        else:
            sdp_ctx = nullcontext()
        with sdp_ctx:
            predictions = self._forward_pose_only(images)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        print("[vggt-omega] forward done, decoding poses", flush=True)
        extrinsic, _ = self._encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )
        ext = extrinsic.detach().float().cpu().numpy()
        if ext.ndim == 4:
            ext = ext[0]
        return [w2c_3x4_to_c2w(ext[i]) for i in range(ext.shape[0])]


def evaluate_vggt_omega_ape(
    estimator: VggtOmegaPoseEstimator,
    pred_paths: list[Path],
    frame_indices: list[int],
    gt_c2ws_list: list[np.ndarray],
) -> VggtOmegaApeResult:
    """Estimate poses with VGGT-Omega and compute evo APE RMSE vs GT c2ws."""
    n_images = len(pred_paths)
    if n_images < 2:
        return nan_ape_result(n_images)
    if len(gt_c2ws_list) != n_images:
        raise ValueError("gt_c2ws_list must align with pred_paths")
    if len(frame_indices) != n_images:
        raise ValueError("frame_indices must align with pred_paths")

    gt_by_idx = {
        int(idx): np.asarray(c2w, dtype=np.float64)
        for idx, c2w in zip(frame_indices, gt_c2ws_list)
    }

    try:
        est_c2ws = estimator.estimate_c2ws(pred_paths)
    except Exception as exc:  # noqa: BLE001 — per-chunk soft fail
        print(f"[warn] VGGT-Omega inference failed: {exc}", flush=True)
        return nan_ape_result(n_images)

    if len(est_c2ws) != n_images:
        print(
            f"[warn] VGGT-Omega pose count mismatch: got {len(est_c2ws)} expected {n_images}",
            flush=True,
        )
        return nan_ape_result(n_images, n_poses=len(est_c2ws))

    est_by_idx = {
        int(idx): c2w for idx, c2w in zip(frame_indices, est_c2ws)
    }

    try:
        trans_rmse, rot_rmse, n_ape = compute_ape_sim3(gt_by_idx, est_by_idx)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] APE computation failed: {exc}", flush=True)
        return VggtOmegaApeResult(
            ape_rmse=float("nan"),
            rotation_ape=float("nan"),
            n_images=n_images,
            n_poses=len(est_c2ws),
            n_ape_poses=0,
        )

    return VggtOmegaApeResult(
        ape_rmse=float(trans_rmse),
        rotation_ape=float(rot_rmse),
        n_images=n_images,
        n_poses=len(est_c2ws),
        n_ape_poses=n_ape,
    )


def aggregate_vggt_omega_chunk_results(
    chunk_results: list[VggtOmegaApeResult],
) -> VggtOmegaApeResult:
    """Arithmetic mean of per-chunk APE over all evaluated chunks in a scene."""
    if not chunk_results:
        return nan_ape_result()

    ape_vals = np.asarray([float(result.ape_rmse) for result in chunk_results], dtype=np.float64)
    rot_vals = np.asarray(
        [float(result.rotation_ape) for result in chunk_results], dtype=np.float64
    )
    n_nan_chunks = sum(1 for result in chunk_results if is_nan_ape_chunk(result))
    return VggtOmegaApeResult(
        ape_rmse=float(np.nanmean(ape_vals)),
        rotation_ape=float(np.nanmean(rot_vals)),
        n_images=int(sum(result.n_images for result in chunk_results)),
        n_poses=int(sum(result.n_poses for result in chunk_results)),
        n_ape_poses=int(sum(result.n_ape_poses for result in chunk_results)),
        n_chunks=len(chunk_results),
        n_nan_chunks=n_nan_chunks,
    )


def _vggt_ape_worker_process(
    device_str: str,
    checkpoint_path: str,
    image_resolution: int,
    preprocess_mode: str,
    jobs_queue: multiprocessing.Queue,
    results_queue: multiprocessing.Queue,
) -> None:
    """Run VGGT-Omega APE in a child process (keeps the 1B model off the parent GPU)."""
    import faulthandler

    faulthandler.enable()
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    try:
        print("[vggt-omega] worker loading checkpoint...", flush=True)
        estimator = VggtOmegaPoseEstimator(
            device,
            Path(checkpoint_path),
            image_resolution=image_resolution,
            preprocess_mode=preprocess_mode,
        )
        print("[vggt-omega] worker model ready", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[vggt-omega] worker init failed: {exc}", flush=True)
        while True:
            job = jobs_queue.get()
            if job is None:
                break
            job_id, chunk_specs = job
            n_images = sum(len(spec[0]) for spec in chunk_specs)
            results_queue.put((job_id, nan_ape_result(n_images), str(exc)))
        return

    while True:
        job = jobs_queue.get()
        if job is None:
            break
        job_id, chunk_specs = job
        try:
            chunk_results: list[VggtOmegaApeResult] = []
            for chunk_idx, (paths_str, frame_indices, gt_c2ws_lists) in enumerate(
                chunk_specs
            ):
                paths = [Path(p) for p in paths_str]
                gt_c2ws = [np.asarray(c2w, dtype=np.float64) for c2w in gt_c2ws_lists]
                print(
                    f"[vggt-omega] worker chunk {chunk_idx}: {len(paths)} frame(s)",
                    flush=True,
                )
                chunk_results.append(
                    evaluate_vggt_omega_ape(estimator, paths, frame_indices, gt_c2ws)
                )
            results_queue.put(
                (job_id, aggregate_vggt_omega_chunk_results(chunk_results), None)
            )
        except Exception as exc:  # noqa: BLE001
            n_images = sum(len(spec[0]) for spec in chunk_specs)
            print(f"[vggt-omega] worker job failed: {exc}", flush=True)
            results_queue.put((job_id, nan_ape_result(n_images), str(exc)))


class VggtOmegaApeSubprocessRunner:
    """Isolate VGGT-Omega APE in a subprocess so it does not share the GPU with MEt3R."""

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: Path,
        *,
        image_resolution: int = DEFAULT_IMAGE_RESOLUTION,
        preprocess_mode: str = DEFAULT_PREPROCESS_MODE,
        timeout_s: float = 3600.0,
    ) -> None:
        if device.type != "cuda":
            raise RuntimeError("VGGT-Omega APE currently expects a CUDA device.")
        self._device_str = str(device)
        self._checkpoint_path = str(checkpoint_path)
        self._image_resolution = int(image_resolution)
        self._preprocess_mode = preprocess_mode
        self._timeout_s = float(timeout_s)
        self._ctx = multiprocessing.get_context("spawn")
        self._jobs: multiprocessing.Queue = self._ctx.Queue()
        self._results: multiprocessing.Queue = self._ctx.Queue()
        self._proc: multiprocessing.Process | None = None

    def _start_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._proc = self._ctx.Process(
            target=_vggt_ape_worker_process,
            args=(
                self._device_str,
                self._checkpoint_path,
                self._image_resolution,
                self._preprocess_mode,
                self._jobs,
                self._results,
            ),
            daemon=True,
        )
        self._proc.start()
        print("[vggt-omega] started isolated APE worker process", flush=True)

    def evaluate_chunks(
        self,
        chunk_specs: list[tuple[list[str], list[int], list[list[float]]]],
    ) -> VggtOmegaApeResult:
        """Evaluate APE for all chunks in one scene."""
        if not chunk_specs:
            return nan_ape_result()
        self._start_worker()
        job_id = uuid.uuid4().hex
        self._jobs.put((job_id, chunk_specs))
        deadline = time.monotonic() + self._timeout_s
        while time.monotonic() < deadline:
            if self._proc is None or not self._proc.is_alive():
                exit_code = self._proc.exitcode if self._proc is not None else None
                print(
                    f"[warn] VGGT-Omega worker died (exit={exit_code}); "
                    "returning NaN APE for this scene",
                    flush=True,
                )
                self._proc = None
                n_images = sum(len(spec[0]) for spec in chunk_specs)
                return nan_ape_result(n_images)
            try:
                rid, result, err = self._results.get(timeout=1.0)
            except queue.Empty:
                continue
            if rid != job_id:
                print(f"[warn] VGGT-Omega unexpected job id {rid!r}", flush=True)
                continue
            if err:
                print(f"[warn] VGGT-Omega worker error: {err}", flush=True)
            return result
        print("[warn] VGGT-Omega worker timed out; returning NaN APE", flush=True)
        n_images = sum(len(spec[0]) for spec in chunk_specs)
        return nan_ape_result(n_images)

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._jobs.put(None)
        except Exception:  # noqa: BLE001
            pass
        self._proc.join(timeout=60)
        if self._proc.is_alive():
            self._proc.kill()
        self._proc = None
