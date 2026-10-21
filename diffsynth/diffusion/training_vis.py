"""TensorBoard logging and gt-vs-pred visualization for RoGe training."""

from __future__ import annotations

import os
import time
import traceback
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .training_module import get_unwrapped_module

if TYPE_CHECKING:
    from accelerate import Accelerator


def border_image(image: Image.Image, color: tuple[int, int, int], width: int = 4) -> Image.Image:
    """Draw a solid outline on ``image``."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(width):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)
    return img


def _vis_label_font(size: int = 22) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_scene_label(image: Image.Image, scene_name: str) -> Image.Image:
    """Draw scene name on the top-left corner with a dark backing plate."""
    if not scene_name:
        return image
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = _vis_label_font()
    padding = 6
    margin = 8
    bbox = draw.textbbox((0, 0), scene_name, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x0, y0 = margin, margin
    draw.rectangle(
        [x0 - padding, y0 - padding, x0 + text_w + padding, y0 + text_h + padding],
        fill=(0, 0, 0),
    )
    draw.text((x0, y0), scene_name, fill=(255, 255, 255), font=font)
    return img


def concat_horizontal(images: list[Image.Image]) -> Image.Image:
    w, h = images[0].size
    canvas = Image.new("RGB", (w * len(images), h))
    for col, img in enumerate(images):
        canvas.paste(img, (col * w, 0))
    return canvas


def make_gt_vs_pred_grid(
    gt_frames: list[Image.Image],
    pred_frames: list[Image.Image],
    num_context: int,
    border_width: int = 4,
    scene_name: str | None = None,
    vggt_depth_frames: list[Image.Image] | None = None,
    aux_row_frames: list[Image.Image] | None = None,
) -> Image.Image:
    """Build a gt-vs-pred grid.

    Without VGGT depth (default): 2 rows — GT, pred.
    With ``vggt_depth_frames`` or ``aux_row_frames``: 3 rows — auxiliary
    conditioning preview (row 1), GT (row 2), pred (row 3).

    Context columns get green borders; target columns get red borders.
    """
    if len(gt_frames) != len(pred_frames):
        raise ValueError(
            f"GT ({len(gt_frames)}) and pred ({len(pred_frames)}) must have the same length"
        )
    num_frames = len(gt_frames)
    if not (0 < num_context < num_frames):
        raise ValueError(f"num_context must be in [1, {num_frames - 1}], got {num_context}")
    aux_frames = aux_row_frames if aux_row_frames is not None else vggt_depth_frames
    if aux_frames is not None and len(aux_frames) != num_frames:
        raise ValueError(
            f"aux row frames ({len(aux_frames)}) must match num_frames ({num_frames})"
        )

    green = (0, 255, 0)
    red = (255, 0, 0)

    def _bordered_row(frame_list: list[Image.Image]) -> Image.Image:
        row = []
        for i, frame in enumerate(frame_list):
            color = green if i < num_context else red
            row.append(border_image(frame, color, border_width))
        return concat_horizontal(row)

    rows = []
    if aux_frames is not None:
        rows.append(_bordered_row(aux_frames))
    rows.append(_bordered_row(gt_frames))
    rows.append(_bordered_row(pred_frames))

    row_h = rows[0].height
    grid = Image.new("RGB", (rows[0].width, row_h * len(rows)))
    for row_idx, row_img in enumerate(rows):
        grid.paste(row_img, (0, row_idx * row_h))
    if scene_name:
        grid = draw_scene_label(grid, scene_name)
    return grid


class TensorboardTracker:
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to {log_dir}")

    def log_scalars(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        epoch: int,
        *,
        fm_loss: float | None = None,
        nvs_loss: float | None = None,
        nvs_loss_weighted: float | None = None,
    ):
        self.writer.add_scalar("train/loss", loss, step)
        if fm_loss is not None:
            self.writer.add_scalar("train/fm_loss", fm_loss, step)
        if nvs_loss is not None:
            self.writer.add_scalar("train/nvs_loss", nvs_loss, step)
        if nvs_loss_weighted is not None:
            self.writer.add_scalar("train/nvs_loss_weighted", nvs_loss_weighted, step)
        self.writer.add_scalar("train/learning_rate", learning_rate, step)
        self.writer.add_scalar("train/epoch", epoch, step)
        self.writer.flush()

    def log_gt_vs_pred(self, step: int, grid: Image.Image, tag: str = "gt_vs_pred"):
        arr = np.asarray(grid)
        self.writer.add_image(tag, arr, step, dataformats="HWC")
        self.writer.flush()

    def close(self):
        self.writer.close()


def _vis_flag_path(flag_dir: str, global_step: int) -> str:
    return os.path.join(flag_dir, f"step_{global_step:06d}.done")


def _wait_for_rank0_vis(flag_path: str, global_step: int, poll_interval: float = 2.0, max_wait_sec: float = 7200.0):
    """Block without NCCL collectives while rank 0 runs visualization."""
    start = time.monotonic()
    while not os.path.isfile(flag_path):
        if time.monotonic() - start > max_wait_sec:
            raise TimeoutError(
                f"[vis] step {global_step}: timed out after {max_wait_sec:.0f}s waiting for {flag_path}"
            )
        time.sleep(poll_interval)


def maybe_log_training_visuals(
    accelerator: Accelerator,
    model,
    data: dict,
    global_step: int,
    tracker: TensorboardTracker | None,
    vis_steps: int | None,
    vis_inference_steps: int,
    vis_seed: int,
    vis_output_dir: str | None,
):
    """Run inference on the current batch and log gt-vs-pred to TensorBoard.

    Rank 0 performs inference; other ranks poll a filesystem flag instead of
    calling ``accelerator.wait_for_everyone()``.  NCCL barriers during long
    rank-0-only inference were the root cause of collective timeouts.
    """
    if tracker is None or vis_steps is None or vis_steps <= 0 or global_step <= 0 or global_step % vis_steps != 0:
        return

    flag_dir = vis_output_dir or os.path.join(os.getcwd(), ".roge_vis_flags")
    flag_path = _vis_flag_path(flag_dir, global_step)

    if not accelerator.is_main_process:
        _wait_for_rank0_vis(flag_path, global_step)
        return

    os.makedirs(flag_dir, exist_ok=True)
    if os.path.isfile(flag_path):
        os.remove(flag_path)

    try:
        module = get_unwrapped_module(model)
        if not hasattr(module, "predict_video"):
            print(f"[vis] step {global_step}: model has no predict_video(), skipping")
        else:
            print(f"[vis] step {global_step}: running {vis_inference_steps}-step inference for gt_vs_pred ...")
            vis_result = module.predict_video(
                data,
                num_inference_steps=vis_inference_steps,
                seed=vis_seed + global_step,
            )
            aux_row_frames = None
            if len(vis_result) == 4:
                gt_frames, pred_frames, num_context, aux_row_frames = vis_result
            else:
                gt_frames, pred_frames, num_context = vis_result
            scene_name = data.get("scene_name", "")
            grid = make_gt_vs_pred_grid(
                gt_frames,
                pred_frames,
                num_context,
                scene_name=scene_name,
                aux_row_frames=aux_row_frames,
            )
            tracker.log_gt_vs_pred(global_step, grid)

            if vis_output_dir is not None:
                os.makedirs(vis_output_dir, exist_ok=True)
                out_path = os.path.join(vis_output_dir, f"step_{global_step:06d}.png")
                grid.save(out_path)
                print(f"[vis] saved {out_path}")
    except Exception as exc:
        print(f"[vis] step {global_step} failed: {exc}")
        traceback.print_exc()
    finally:
        with open(flag_path, "w", encoding="utf-8") as f:
            f.write("done\n")
