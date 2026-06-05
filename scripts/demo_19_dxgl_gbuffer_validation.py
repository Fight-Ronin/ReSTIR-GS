from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.eval.gbuffer_validation import (
    binary_mask_metrics,
    depth_metrics,
    masked_rgb_metrics,
    normal_display_metrics,
)
from restir_gs.render.dxgl_asset import load_dxgl_aligned_asset, load_dxgl_frame_modalities, scale_camera
from restir_gs.render.gbuffer import GBuffer, make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.render.scene_normalization import scene_normalization_to_dict
from scripts.demo_17_dxgl_aligned_intake import parse_int_list
from scripts.download_dxgl_apple import DEFAULT_EXTRACT_DIR, find_dxgl_dataset_root, validate_dxgl_dataset_root
from scripts.download_dxgl_apple_splat import DEFAULT_SPLAT_PATH, validate_dxgl_splat_file


DEFAULT_OUTPUT_DIR = Path("outputs/aligned_gbuffer")
DEFAULT_FRAME_INDICES = (0, 49, 98, 147)


def to_u8_rgb(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)


def to_u8_scalar(values: torch.Tensor, mask: torch.Tensor | None = None) -> np.ndarray:
    data = values.detach().cpu().float()
    valid = torch.isfinite(data)
    if mask is not None:
        valid = valid & mask.detach().cpu().to(torch.bool)
    valid = valid & (data > 0.0)
    out = torch.zeros_like(data)
    if bool(valid.any()):
        selected = data[valid]
        lo = selected.min()
        hi = selected.max()
        denom = hi - lo if float(hi - lo) > 1e-8 else torch.tensor(1.0)
        out[valid] = (selected - lo) / denom
    return (out.numpy() * 255.0).astype(np.uint8)


def to_u8_normal(normal_cam: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    data = ((normal_cam.detach().cpu() * 0.5) + 0.5).clamp(0.0, 1.0)
    out = torch.zeros_like(data)
    valid = mask.detach().cpu().to(torch.bool)
    out[valid] = data[valid]
    return (out.numpy() * 255.0).astype(np.uint8)


def make_error_image(estimate: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    error = torch.abs(estimate.detach().cpu().float() - reference.detach().cpu().float())
    if error.ndim == 3:
        error = error.mean(dim=-1)
    valid = mask.detach().cpu().to(torch.bool) & torch.isfinite(error)
    out = torch.zeros_like(error)
    if bool(valid.any()):
        selected = error[valid]
        hi = torch.clamp(selected.max(), min=1e-8)
        out[valid] = selected / hi
    return (out.numpy() * 255.0).astype(np.uint8)


def make_gbuffer_contact_sheet(records: list[dict[str, Any]], output_path: str | Path) -> None:
    cell_w = 168
    cell_h = 138
    label_w = 184
    header_h = 36
    labels = ["Ref RGB", "Render RGB", "Alpha", "Ref Depth", "Render Depth", "Ref Normal", "Pseudo Normal", "Depth Error"]
    rows = max(len(records), 1)
    sheet = Image.new("RGB", (label_w + len(labels) * cell_w, header_h + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), "DXGL Apple aligned G-buffer validation", fill=(0, 0, 0))
    for row, record in enumerate(records):
        y = header_h + row * cell_h
        draw.text(
            (8, y + 10),
            f"frame {record['frame_index']}\n"
            f"rgb MAE={record['rgb_metrics']['mae']:.4f}\n"
            f"alpha IoU={record['alpha_metrics']['iou']:.3f}",
            fill=(0, 0, 0),
        )
        for col, label in enumerate(labels):
            x = label_w + col * cell_w
            draw.text((x + 6, y + 6), label, fill=(0, 0, 0))
            image = Image.open(record["images"][label]).convert("RGB")
            _paste_thumbnail(sheet, draw, image, (x + 6, y + 24), (154, 104))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _paste_thumbnail(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    xy: tuple[int, int],
    size: tuple[int, int],
) -> None:
    x, y = xy
    width, height = size
    thumb = image.copy()
    thumb.thumbnail(size)
    sheet.paste(thumb, (x + (width - thumb.width) // 2, y + (height - thumb.height) // 2))
    draw.rectangle((x, y, x + width, y + height), outline=(180, 180, 180))


def save_frame_outputs(frame_index: int, gbuffer: GBuffer, modalities, output_dir: Path) -> dict[str, Any]:
    prefix = f"dxgl_apple_gbuffer_frame_{frame_index:03d}"
    render_mask = gbuffer.valid_mask.detach().cpu()
    normal_mask = gbuffer.normal_mask.detach().cpu()
    ref_mask = modalities.mask.detach().cpu()
    common_mask = ref_mask & render_mask
    normal_common_mask = common_mask & normal_mask

    paths = {
        "Ref RGB": output_dir / f"{prefix}_reference_rgb.png",
        "Render RGB": output_dir / f"{prefix}_render_rgb.png",
        "Alpha": output_dir / f"{prefix}_alpha.png",
        "Ref Depth": output_dir / f"{prefix}_reference_depth.png",
        "Render Depth": output_dir / f"{prefix}_render_depth.png",
        "Ref Normal": output_dir / f"{prefix}_reference_normal.png",
        "Pseudo Normal": output_dir / f"{prefix}_pseudo_normal.png",
        "Depth Error": output_dir / f"{prefix}_depth_error.png",
    }
    imageio.imwrite(paths["Ref RGB"], to_u8_rgb(modalities.rgb))
    imageio.imwrite(paths["Render RGB"], to_u8_rgb(gbuffer.rgb))
    imageio.imwrite(paths["Alpha"], (gbuffer.alpha.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8))

    if modalities.depth_normalized is not None:
        imageio.imwrite(paths["Ref Depth"], to_u8_scalar(modalities.depth_normalized, ref_mask))
        depth_error = make_error_image(gbuffer.depth.detach().cpu(), modalities.depth_normalized, common_mask)
    else:
        imageio.imwrite(paths["Ref Depth"], np.zeros_like(to_u8_scalar(gbuffer.depth, render_mask)))
        depth_error = np.zeros_like(to_u8_scalar(gbuffer.depth, render_mask))
    imageio.imwrite(paths["Render Depth"], to_u8_scalar(gbuffer.depth, render_mask))
    imageio.imwrite(paths["Depth Error"], depth_error)

    if modalities.normal_rgb is not None:
        imageio.imwrite(paths["Ref Normal"], to_u8_rgb(modalities.normal_rgb))
        normal_metrics = normal_display_metrics(
            torch.tensor(to_u8_normal(gbuffer.normal_cam, normal_mask), dtype=torch.float32) / 255.0,
            modalities.normal_rgb,
            normal_common_mask,
        )
    else:
        imageio.imwrite(paths["Ref Normal"], np.zeros((*gbuffer.depth.shape, 3), dtype=np.uint8))
        normal_metrics = {"valid_pixels": 0, "mae": 0.0, "rmse": 0.0, "psnr": 0.0}
    imageio.imwrite(paths["Pseudo Normal"], to_u8_normal(gbuffer.normal_cam, normal_mask))

    rgb_metrics = masked_rgb_metrics(gbuffer.rgb, modalities.rgb, ref_mask)
    alpha_metrics = binary_mask_metrics(render_mask, ref_mask)
    depth_metric_payload = (
        depth_metrics(gbuffer.depth.detach().cpu(), modalities.depth_normalized, common_mask)
        if modalities.depth_normalized is not None
        else {"valid_pixels": 0, "mae": 0.0, "rmse": 0.0, "abs_rel": 0.0}
    )
    return {
        "frame_index": frame_index,
        "images": {key: str(path) for key, path in paths.items()},
        "rgb_metrics": rgb_metrics,
        "alpha_metrics": alpha_metrics,
        "depth_metrics": depth_metric_payload,
        "normal_display_metrics": normal_metrics,
        "normal_metric_note": "display-space diagnostic only; DXGL normal semantic space is not assumed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DXGL Apple render buffers and pseudo G-buffer against aligned modalities.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--splat", type=Path, default=DEFAULT_SPLAT_PATH)
    parser.add_argument("--frame-indices", type=parse_int_list, default=list(DEFAULT_FRAME_INDICES))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--depth-unit-scale", type=float, default=10000.0)
    parser.add_argument("--max-gaussians", type=int, default=0)
    parser.add_argument("--normalization-bbox-percentile", type=float, default=0.98)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    dataset_root = find_dxgl_dataset_root(args.dataset_root, required=True)
    validate_dxgl_dataset_root(dataset_root, raise_on_missing=True)
    splat_validation = validate_dxgl_splat_file(args.splat)
    device = torch.device(args.device)
    max_gaussians = None if args.max_gaussians <= 0 else args.max_gaussians
    asset = load_dxgl_aligned_asset(
        dataset_root,
        args.splat,
        device=device,
        max_gaussians=max_gaussians,
        normalization_bbox_percentile=args.normalization_bbox_percentile,
    )

    selected_indices = list(args.frame_indices)
    if min(selected_indices) < 0 or max(selected_indices) >= asset.transforms.frame_count:
        raise ValueError(f"Selected frame indices {selected_indices} exceed frame count {asset.transforms.frame_count}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for frame_index in selected_indices:
        frame = asset.transforms.frames[frame_index]
        camera = scale_camera(frame.camera, args.width, args.height)
        render_buffers = render_rgbd(asset.loaded.scene, camera)
        gbuffer = make_pseudo_gbuffer(render_buffers, camera)
        modalities = load_dxgl_frame_modalities(
            frame,
            args.width,
            args.height,
            depth_unit_scale=args.depth_unit_scale,
            scene_scale=asset.normalization.scale if asset.normalization is not None else None,
        )
        records.append(save_frame_outputs(frame_index, gbuffer, modalities, args.output_dir))

    contact_path = args.output_dir / "dxgl_apple_gbuffer_contact.png"
    make_gbuffer_contact_sheet(records, contact_path)
    summary = {
        "version": 1,
        "dataset": "dxgl_polyhaven_10_apple",
        "dataset_root": str(dataset_root),
        "splat_path": str(args.splat),
        "splat_validation": splat_validation,
        "render": {
            "width": args.width,
            "height": args.height,
            "device": str(device),
            "max_gaussians": max_gaussians,
            "loaded_count": asset.loaded.stats.loaded_count,
            "original_count": asset.loaded.stats.original_count,
        },
        "camera_normalization": scene_normalization_to_dict(asset.normalization) if asset.normalization is not None else None,
        "depth_reference": {
            "source": "depth_16bit",
            "raw_depth_unit_scale": args.depth_unit_scale,
            "normalized_by_scene_scale": asset.normalization.scale if asset.normalization is not None else None,
            "note": "Depth comparison assumes DXGL depth_16bit is raw camera z-depth divided by raw_depth_unit_scale.",
        },
        "selected_frame_indices": selected_indices,
        "mean_metrics": {
            "rgb_mae": float(np.mean([record["rgb_metrics"]["mae"] for record in records])),
            "alpha_iou": float(np.mean([record["alpha_metrics"]["iou"] for record in records])),
            "depth_mae": float(np.mean([record["depth_metrics"]["mae"] for record in records])),
            "normal_display_mae": float(np.mean([record["normal_display_metrics"]["mae"] for record in records])),
        },
        "frames": records,
        "contact_sheet": str(contact_path),
    }
    summary_path = args.output_dir / "dxgl_apple_gbuffer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"splat:      {args.splat}")
    print(f"frames:     {selected_indices}")
    print(f"rgb MAE:    {summary['mean_metrics']['rgb_mae']:.6f}")
    print(f"alpha IoU:  {summary['mean_metrics']['alpha_iou']:.6f}")
    print(f"depth MAE:  {summary['mean_metrics']['depth_mae']:.6f}")
    print(f"wrote:      {contact_path.resolve()}")
    print(f"wrote:      {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
