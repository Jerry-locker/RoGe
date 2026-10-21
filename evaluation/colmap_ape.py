"""pycolmap SfM on pred frames + evo APE RMSE vs GT cameras.

Runs pycolmap on non-context pred frames, takes the largest reconstruction, and
compares camera-to-world trajectories against ``transforms.json`` GT using
evo APE (translational) with Sim(3) Umeyama alignment (``align + correct_scale``).

Before APE, GT camera translations are divided by the GT trajectory length
(sum of consecutive camera-center distances) so TransAPE is scale-free across
monocular / arbitrary-scale COLMAP scenes. RotAPE is unchanged (degrees).

A sequence is **successful** iff COLMAP registers all input frames
(``n_registered == n_images``; default ``min_ratio=1.0``). APE is only reported
for successful sequences.
"""

from __future__ import annotations

import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation.paths import REPO_ROOT

DEFAULT_MIN_REGISTERED = 3
DEFAULT_MIN_RATIO = 1.0  # require every frame registered
DEFAULT_MATCHER = "sequential"  # or "exhaustive"
# Reject near-static GT trajectories before dividing by path length.
DEFAULT_TRAJ_LENGTH_EPS = 1e-8


def trajectory_length(centers: np.ndarray) -> float:
    """Sum of consecutive camera-center distances. ``centers`` is ``[N, 3]``."""
    pts = np.asarray(centers, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"centers expected [N,3], got {pts.shape}")
    if pts.shape[0] < 2:
        return 0.0
    if not np.isfinite(pts).all():
        return float("nan")
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def scale_c2w_translations(
    c2ws: list[np.ndarray], scale: float
) -> list[np.ndarray]:
    """Return c2w copies with translation ``t' = scale * t`` (rotation unchanged)."""
    out: list[np.ndarray] = []
    for c2w in c2ws:
        pose = np.asarray(c2w, dtype=np.float64).copy()
        pose[:3, 3] *= float(scale)
        out.append(pose)
    return out


def ensure_evo_on_path() -> None:
    """Prefer installed ``evo``; fall back to vendored ``third_party/evo``."""
    try:
        import evo  # noqa: F401

        return
    except ImportError:
        pass
    vendor = REPO_ROOT / "third_party"
    vendor_s = str(vendor)
    if vendor_s not in sys.path:
        sys.path.insert(0, vendor_s)
    try:
        import evo  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "evo is required for COLMAP APE RMSE.\n"
            "Install with:  pip install evo\n"
            "Or ensure vendored copy exists at third_party/evo.\n"
            "Or pass --skip_colmap_ape to skip."
        ) from exc


def import_pycolmap():
    try:
        import pycolmap

        return pycolmap
    except ImportError:
        pass
    # Optional local staging dir (e.g. copied wheel contents); not committed.
    for extra in (REPO_ROOT / ".python_packages", REPO_ROOT / "third_party"):
        extra_s = str(extra)
        if extra.is_dir() and extra_s not in sys.path:
            sys.path.insert(0, extra_s)
        try:
            import pycolmap

            return pycolmap
        except ImportError:
            continue
    raise ImportError(
        "pycolmap is required for COLMAP APE RMSE.\n"
        "Install with:  pip install pycolmap\n"
        "Or pass --skip_colmap_ape to skip."
    )

@dataclass(frozen=True)
class ColmapApeResult:
    success: bool
    ape_rmse: float
    rotation_ape: float
    n_images: int
    n_registered: int
    n_ape_poses: int
    model_id: str
    message: str = ""


def prepare_colmap_images(
    pred_paths: list[Path],
    frame_indices: list[int],
    images_dir: Path,
) -> list[str]:
    """Stage pred frames as COLMAP-readable ``{idx:06d}.jpg``.

    OpenCV/Pillow can read the experiment PNGs, but pycolmap's FreeImage/libpng
    often fails on them with ``libpng error: bad parameters to zlib`` /
    ``BITMAP_ERROR``. Re-encode to 8-bit RGB JPEG instead of symlink/copy.
    """
    import cv2

    if len(pred_paths) != len(frame_indices):
        raise ValueError("pred_paths / frame_indices length mismatch")
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for path, idx in zip(pred_paths, frame_indices):
        name = f"{idx:06d}.jpg"
        dst = images_dir / name
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read image for COLMAP staging: {path}")
        if not cv2.imwrite(str(dst), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise ValueError(f"Failed to write COLMAP image: {dst}")
        names.append(name)
    return names


def _device(pycolmap, use_gpu: bool):
    if use_gpu:
        return pycolmap.Device.auto
    return pycolmap.Device.cpu


def _w2c_matrix_to_c2w(w2c_3x4: np.ndarray) -> np.ndarray:
    """COLMAP cam_from_world 3x4 → OpenCV camera-to-world 4x4."""
    w2c = np.asarray(w2c_3x4, dtype=np.float64)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def _image_w2c_matrix(image) -> np.ndarray:
    """Support pycolmap 3.10 (property) and 3.11+ (method)."""
    cfw = image.cam_from_world
    if callable(cfw):
        cfw = cfw()
    return np.asarray(cfw.matrix(), dtype=np.float64)


def _image_is_registered(image) -> bool:
    if hasattr(image, "has_pose"):
        return bool(image.has_pose)
    if hasattr(image, "registered"):
        return bool(image.registered)
    return True


def reconstruction_to_c2ws(reconstruction) -> dict[str, np.ndarray]:
    """``{image_name: c2w_4x4}`` for registered images with poses."""
    poses: dict[str, np.ndarray] = {}
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.images[image_id]
        if not _image_is_registered(image):
            continue
        poses[str(image.name)] = _w2c_matrix_to_c2w(_image_w2c_matrix(image))
    return poses


def _match_features(pycolmap, database_path: Path, *, matcher: str, device) -> None:
    if matcher == "exhaustive":
        pycolmap.match_exhaustive(database_path=str(database_path), device=device)
        return

    # sequential — option class renamed across pycolmap versions
    if hasattr(pycolmap, "SequentialPairingOptions"):
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 10
        pairing.quadratic_overlap = True
        pairing.loop_detection = False
        pycolmap.match_sequential(
            database_path=str(database_path),
            pairing_options=pairing,
            device=device,
        )
    else:
        matching = pycolmap.SequentialMatchingOptions()
        matching.overlap = 10
        matching.quadratic_overlap = True
        matching.loop_detection = False
        pycolmap.match_sequential(
            database_path=str(database_path),
            matching_options=matching,
            device=device,
        )


def run_pycolmap_sfm(
    images_dir: Path,
    work_dir: Path,
    *,
    matcher: str = DEFAULT_MATCHER,
    use_gpu: bool = True,
    min_model_size: int = 3,
) -> tuple[dict[str, np.ndarray], str]:
    """Run pycolmap feature → match → incremental mapping.

    Returns ``(c2ws_by_name, model_id)`` for the largest reconstruction.
    """
    pycolmap = import_pycolmap()
    work_dir.mkdir(parents=True, exist_ok=True)
    database_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    if database_path.exists():
        database_path.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    matcher = matcher.lower().strip()
    if matcher not in {"exhaustive", "sequential"}:
        raise ValueError(f"Unknown COLMAP matcher: {matcher}")

    pipeline_opts = pycolmap.IncrementalPipelineOptions()
    pipeline_opts.min_model_size = max(3, int(min_model_size))
    pipeline_opts.ba_global_max_num_iterations = 30

    def _extract_and_match(device) -> None:
        if database_path.exists():
            database_path.unlink()
        pycolmap.extract_features(
            database_path=str(database_path),
            image_path=str(images_dir),
            camera_mode=pycolmap.CameraMode.SINGLE,
            device=device,
        )
        _match_features(pycolmap, database_path, matcher=matcher, device=device)

    device = _device(pycolmap, use_gpu)
    try:
        _extract_and_match(device)
    except Exception as exc:  # noqa: BLE001
        if use_gpu and device != pycolmap.Device.cpu:
            print(f"[warn] pycolmap GPU/auto failed ({exc}); retrying on CPU", flush=True)
            _extract_and_match(pycolmap.Device.cpu)
        else:
            raise

    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(images_dir),
        output_path=str(sparse_dir),
        options=pipeline_opts,
    )
    if not reconstructions:
        return {}, ""

    best_id, best_rec = max(
        reconstructions.items(),
        key=lambda kv: int(kv[1].num_reg_images()),
    )
    # Persist for optional --colmap_keep_work inspection.
    model_out = sparse_dir / str(best_id)
    model_out.mkdir(parents=True, exist_ok=True)
    best_rec.write(str(model_out))
    return reconstruction_to_c2ws(best_rec), str(best_id)


def frame_index_from_name(name: str) -> int | None:
    stem = Path(name).stem
    try:
        return int(stem)
    except ValueError:
        return None


def compute_ape_sim3(
    gt_c2ws: dict[int, np.ndarray],
    est_c2ws: dict[int, np.ndarray],
    *,
    traj_length_eps: float = DEFAULT_TRAJ_LENGTH_EPS,
) -> tuple[float, float, int]:
    """evo APE RMSE after GT path-length normalization + Sim(3) alignment.

    GT translations are divided by the sum of consecutive GT camera-center
    distances so TransAPE is comparable across arbitrary-scale scenes.
    Estimated poses are left unscaled; Umeyama ``correct_scale`` maps them onto
    the normalized GT.

    Returns ``(translation_rmse, rotation_rmse_deg, n_poses)``.
    """
    ensure_evo_on_path()
    from evo.core.metrics import APE, PoseRelation, StatisticsType
    from evo.core.trajectory import PoseTrajectory3D

    common = sorted(set(gt_c2ws) & set(est_c2ws))
    if len(common) < 2:
        raise RuntimeError(f"Need >=2 shared poses for APE, got {len(common)}")

    timestamps = np.asarray(common, dtype=np.float64)
    gt_poses = [np.asarray(gt_c2ws[i], dtype=np.float64) for i in common]
    est_poses = [np.asarray(est_c2ws[i], dtype=np.float64) for i in common]

    gt_centers = np.stack([pose[:3, 3] for pose in gt_poses], axis=0)
    path_len = trajectory_length(gt_centers)
    if not np.isfinite(path_len) or path_len <= float(traj_length_eps):
        raise RuntimeError(
            f"GT trajectory length too small for TransAPE normalization: {path_len}"
        )
    gt_poses = scale_c2w_translations(gt_poses, 1.0 / path_len)

    traj_ref = PoseTrajectory3D(poses_se3=gt_poses, timestamps=timestamps.copy())
    traj_est = PoseTrajectory3D(poses_se3=est_poses, timestamps=timestamps.copy())
    # Mutates traj_est in-place (Umeyama Sim3).
    traj_est.align(traj_ref, correct_scale=True)

    ape_trans = APE(PoseRelation.translation_part)
    ape_trans.process_data((traj_ref, traj_est))
    trans_rmse = float(ape_trans.get_statistic(StatisticsType.rmse))

    ape_rot = APE(PoseRelation.rotation_angle_deg)
    ape_rot.process_data((traj_ref, traj_est))
    rot_rmse = float(ape_rot.get_statistic(StatisticsType.rmse))

    return trans_rmse, rot_rmse, len(common)


def compute_ape_rmse_sim3(
    gt_c2ws: dict[int, np.ndarray],
    est_c2ws: dict[int, np.ndarray],
) -> tuple[float, int]:
    """evo translational APE RMSE after Sim(3) alignment. Returns (rmse, n_poses)."""
    trans_rmse, _, n_poses = compute_ape_sim3(gt_c2ws, est_c2ws)
    return trans_rmse, n_poses


def evaluate_colmap_ape(
    pred_paths: list[Path],
    frame_indices: list[int],
    gt_c2ws_list: list[np.ndarray],
    work_dir: Path,
    *,
    matcher: str = DEFAULT_MATCHER,
    use_gpu: bool = True,
    min_registered: int = DEFAULT_MIN_REGISTERED,
    min_ratio: float = DEFAULT_MIN_RATIO,
) -> ColmapApeResult:
    """Run pycolmap on pred frames and compute evo APE RMSE vs GT c2ws."""
    n_images = len(pred_paths)
    if n_images < 2:
        return ColmapApeResult(
            success=False,
            ape_rmse=float("nan"),
            rotation_ape=float("nan"),
            n_images=n_images,
            n_registered=0,
            n_ape_poses=0,
            model_id="",
            message="need >=2 pred frames",
        )
    if len(gt_c2ws_list) != n_images:
        raise ValueError("gt_c2ws_list must align with pred_paths")

    gt_by_idx = {
        int(idx): np.asarray(c2w, dtype=np.float64)
        for idx, c2w in zip(frame_indices, gt_c2ws_list)
    }

    images_dir = work_dir / "images"
    try:
        prepare_colmap_images(pred_paths, frame_indices, images_dir)
        est_by_name, model_id = run_pycolmap_sfm(
            images_dir,
            work_dir,
            matcher=matcher,
            use_gpu=use_gpu,
            min_model_size=min(10, max(3, n_images // 2)),
        )
    except Exception as exc:  # noqa: BLE001 — per-scene soft fail
        return ColmapApeResult(
            success=False,
            ape_rmse=float("nan"),
            rotation_ape=float("nan"),
            n_images=n_images,
            n_registered=0,
            n_ape_poses=0,
            model_id="",
            message=str(exc),
        )

    est_by_idx: dict[int, np.ndarray] = {}
    for name, c2w in est_by_name.items():
        idx = frame_index_from_name(name)
        if idx is None:
            continue
        est_by_idx[idx] = c2w

    n_registered = len(est_by_idx)
    need = max(int(min_registered), int(math.ceil(float(min_ratio) * n_images)))
    if n_registered < need:
        return ColmapApeResult(
            success=False,
            ape_rmse=float("nan"),
            rotation_ape=float("nan"),
            n_images=n_images,
            n_registered=n_registered,
            n_ape_poses=0,
            model_id=model_id,
            message=f"registered {n_registered}/{n_images} < required {need}",
        )

    try:
        trans_rmse, rot_rmse, n_ape = compute_ape_sim3(gt_by_idx, est_by_idx)
    except Exception as exc:  # noqa: BLE001
        return ColmapApeResult(
            success=False,
            ape_rmse=float("nan"),
            rotation_ape=float("nan"),
            n_images=n_images,
            n_registered=n_registered,
            n_ape_poses=0,
            model_id=model_id,
            message=f"APE failed: {exc}",
        )

    return ColmapApeResult(
        success=True,
        ape_rmse=float(trans_rmse),
        rotation_ape=float(rot_rmse),
        n_images=n_images,
        n_registered=n_registered,
        n_ape_poses=n_ape,
        model_id=model_id,
        message="ok",
    )
