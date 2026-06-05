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

from restir_gs.render.gsplat_renderer import RenderBuffers, render_rgbd
from restir_gs.render.ply_loader import load_gaussian_ply_with_stats
from restir_gs.render.scene_normalization import (
    infer_scene_normalization_from_plys,
    scene_normalization_to_dict,
)
from restir_gs.render.synthetic_scene import PinholeCamera
from restir_gs.render.transforms_loader import ImportedTransformFrame, load_nerfstudio_transforms
from scripts.demo_17_dxgl_aligned_intake import parse_int_list
from scripts.download_dxgl_apple import DEFAULT_EXTRACT_DIR, find_dxgl_dataset_root, validate_dxgl_dataset_root
from scripts.download_dxgl_apple_splat import DEFAULT_SPLAT_PATH, validate_dxgl_splat_file


DEFAULT_OUTPUT_DIR = Path("outputs/aligned_fidelity")
DEFAULT_FRAME_INDICES = (0, 49, 98, 147)


def scale_camera(camera: PinholeCamera, width: int, height: int) -> PinholeCamera:
    if width <= 0 or height <= 0:
        raise ValueError(f"Expected positive output size, got {width}x{height}")
    sx = float(width) / float(camera.width)
    sy = float(height) / float(camera.height)
    intrinsics = camera.intrinsics.clone()
    intrinsics[:, 0, :] *= sx
    intrinsics[:, 1, :] *= sy
    return PinholeCamera(viewmats=camera.viewmats.clone(), intrinsics=intrinsics, width=width, height=height)


def load_reference_rgb_and_mask(frame: ImportedTransformFrame, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    rgb_image = Image.open(frame.image_path).convert("RGBA").resize((width, height), Image.Resampling.BILINEAR)
    rgb_array = np.asarray(rgb_image, dtype=np.float32) / 255.0
    rgb = torch.tensor(rgb_array[..., :3], dtype=torch.float32)
    alpha_mask = torch.tensor(rgb_array[..., 3] > 0.5, dtype=torch.bool)
    if frame.mask_path is not None and frame.mask_path.exists():
        mask_image = Image.open(frame.mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
        mask = torch.tensor(np.asarray(mask_image) > 127, dtype=torch.bool)
    else:
        mask = alpha_mask
    return rgb, mask


def compute_masked_rgb_metrics(estimate: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    if estimate.shape != reference.shape:
        raise ValueError(f"Estimate/reference shape mismatch: {tuple(estimate.shape)} vs {tuple(reference.shape)}")
    if mask.shape != estimate.shape[:2]:
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match RGB shape {tuple(estimate.shape)}")
    valid_count = int(mask.sum().item())
    if valid_count <= 0:
        return {"valid_pixels": 0, "mae": 0.0, "rmse": 0.0, "psnr": 0.0}
    diff = estimate.detach().cpu()[mask] - reference.detach().cpu()[mask]
    mse = torch.mean(diff.square())
    mae = torch.mean(diff.abs())
    rmse = torch.sqrt(mse)
    psnr = 20.0 * torch.log10(torch.tensor(1.0) / torch.clamp(rmse, min=1e-8))
    return {
        "valid_pixels": valid_count,
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "psnr": float(psnr.item()),
    }


def to_u8_rgb(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def to_u8_error(estimate: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    error = torch.mean(torch.abs(estimate.detach().cpu() - reference.detach().cpu()), dim=-1)
    out = torch.zeros_like(error)
    if bool(mask.any()):
        values = error[mask]
        hi = torch.clamp(values.max(), min=1e-8)
        out[mask] = values / hi
    return (out.numpy() * 255.0).astype(np.uint8)


def make_fidelity_contact_sheet(records: list[dict[str, Any]], output_path: str | Path) -> None:
    cell_w = 192
    cell_h = 156
    label_w = 176
    header_h = 36
    cols = 4
    rows = max(len(records), 1)
    sheet = Image.new("RGB", (label_w + cols * cell_w, header_h + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), "DXGL Apple splat fidelity: reference / render / alpha / abs error", fill=(0, 0, 0))
    for row, record in enumerate(records):
        y = header_h + row * cell_h
        metric = record["metrics"]
        draw.text(
            (8, y + 12),
            f"frame {record['frame_index']}\nMAE={metric['mae']:.4f}\nPSNR={metric['psnr']:.2f}",
            fill=(0, 0, 0),
        )
        for col, key in enumerate(["reference_path", "render_path", "alpha_path", "error_path"]):
            label = ["Reference", "Render", "Alpha", "Error"][col]
            x = label_w + col * cell_w
            draw.text((x + 8, y + 8), label, fill=(0, 0, 0))
            image = Image.open(record[key]).convert("RGB")
            _paste_thumbnail(sheet, draw, image, (x + 8, y + 28), (176, 116))
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
    paste_x = x + (width - thumb.width) // 2
    paste_y = y + (height - thumb.height) // 2
    sheet.paste(thumb, (paste_x, paste_y))
    draw.rectangle((x, y, x + width, y + height), outline=(180, 180, 180))


def _save_frame_outputs(
    frame: ImportedTransformFrame,
    buffers: RenderBuffers,
    reference_rgb: torch.Tensor,
    mask: torch.Tensor,
    output_dir: Path,
) -> dict[str, Any]:
    prefix = f"dxgl_apple_splat_frame_{frame.index:03d}"
    render_path = output_dir / f"{prefix}_render.png"
    reference_path = output_dir / f"{prefix}_reference.png"
    alpha_path = output_dir / f"{prefix}_alpha.png"
    error_path = output_dir / f"{prefix}_abs_error.png"
    imageio.imwrite(render_path, to_u8_rgb(buffers.rgb))
    imageio.imwrite(reference_path, to_u8_rgb(reference_rgb))
    imageio.imwrite(alpha_path, (buffers.alpha.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8))
    imageio.imwrite(error_path, to_u8_error(buffers.rgb, reference_rgb, mask))
    metrics = compute_masked_rgb_metrics(buffers.rgb, reference_rgb, mask)
    render_valid = int((buffers.alpha.detach().cpu() > 1e-4).sum().item())
    return {
        "frame_index": frame.index,
        "file_path": frame.file_path,
        "reference_path": str(reference_path),
        "render_path": str(render_path),
        "alpha_path": str(alpha_path),
        "error_path": str(error_path),
        "metrics": metrics,
        "render_valid_pixels": render_valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render DXGL Apple pretrained splat from aligned transforms cameras.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--splat", type=Path, default=DEFAULT_SPLAT_PATH)
    parser.add_argument("--frame-indices", type=parse_int_list, default=list(DEFAULT_FRAME_INDICES))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-gaussians", type=int, default=0)
    parser.add_argument(
        "--camera-normalization",
        choices=["inferred_from_points3d", "none"],
        default="inferred_from_points3d",
    )
    parser.add_argument(
        "--normalization-rotation",
        choices=["identity", "raw_y_to_z_up", "raw_y_to_minus_z_up"],
        default="raw_y_to_z_up",
    )
    parser.add_argument("--normalization-bbox-percentile", type=float, default=0.98)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    dataset_root = find_dxgl_dataset_root(args.dataset_root, required=True)
    validate_dxgl_dataset_root(dataset_root, raise_on_missing=True)
    splat_validation = validate_dxgl_splat_file(args.splat)
    normalization = None
    if args.camera_normalization == "inferred_from_points3d":
        normalization = infer_scene_normalization_from_plys(
            dataset_root / "points3D.ply",
            args.splat,
            bbox_percentile=args.normalization_bbox_percentile,
            raw_to_target_rotation=args.normalization_rotation,
        )
    max_gaussians = None if args.max_gaussians <= 0 else args.max_gaussians
    loaded = load_gaussian_ply_with_stats(args.splat, device=device, max_gaussians=max_gaussians)
    imported = load_nerfstudio_transforms(
        dataset_root / "transforms.json",
        dataset_root=dataset_root,
        device=device,
        camera_normalization=normalization,
    )

    selected_indices = list(args.frame_indices)
    if max(selected_indices) >= imported.frame_count or min(selected_indices) < 0:
        raise ValueError(f"Selected frame indices {selected_indices} exceed frame count {imported.frame_count}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for frame_index in selected_indices:
        frame = imported.frames[frame_index]
        camera = scale_camera(frame.camera, args.width, args.height)
        buffers = render_rgbd(loaded.scene, camera)
        reference_rgb, mask = load_reference_rgb_and_mask(frame, args.width, args.height)
        records.append(_save_frame_outputs(frame, buffers, reference_rgb, mask, args.output_dir))

    contact_path = args.output_dir / "dxgl_apple_splat_contact.png"
    make_fidelity_contact_sheet(records, contact_path)
    mean_mae = float(np.mean([record["metrics"]["mae"] for record in records]))
    mean_rmse = float(np.mean([record["metrics"]["rmse"] for record in records]))
    mean_psnr = float(np.mean([record["metrics"]["psnr"] for record in records]))
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
            "loaded_count": loaded.stats.loaded_count,
            "original_count": loaded.stats.original_count,
        },
        "camera_normalization": {
            "mode": args.camera_normalization,
            "rotation_mode": args.normalization_rotation if normalization is not None else None,
            "normalization": scene_normalization_to_dict(normalization) if normalization is not None else None,
        },
        "selected_frame_indices": selected_indices,
        "mean_metrics": {"mae": mean_mae, "rmse": mean_rmse, "psnr": mean_psnr},
        "frames": records,
        "contact_sheet": str(contact_path),
    }
    summary_path = args.output_dir / "dxgl_apple_splat_fidelity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"splat:        {args.splat}")
    print(f"gaussians:    {loaded.stats.loaded_count} / {loaded.stats.original_count}")
    if normalization is not None:
        print(f"normalization scale: {normalization.scale:.6f}")
    print(f"frames:       {selected_indices}")
    print(f"mean MAE:     {mean_mae:.6f}")
    print(f"mean RMSE:    {mean_rmse:.6f}")
    print(f"mean PSNR:    {mean_psnr:.3f}")
    print(f"wrote:        {contact_path.resolve()}")
    print(f"wrote:        {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
