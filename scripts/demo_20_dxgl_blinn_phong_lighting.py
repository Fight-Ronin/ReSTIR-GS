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

from restir_gs.lighting.asset_lights import make_asset_scaled_point_lights
from restir_gs.lighting.deferred import shade_deferred_blinn_phong, shade_deferred_lambertian
from restir_gs.render.dxgl_asset import load_dxgl_aligned_asset, scale_camera
from restir_gs.render.gbuffer import make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.render.scene_normalization import scene_normalization_to_dict
from scripts.demo_17_dxgl_aligned_intake import parse_int_list
from scripts.demo_19_dxgl_gbuffer_validation import to_u8_normal
from scripts.download_dxgl_apple import DEFAULT_EXTRACT_DIR, find_dxgl_dataset_root, validate_dxgl_dataset_root
from scripts.download_dxgl_apple_splat import DEFAULT_SPLAT_PATH, validate_dxgl_splat_file


DEFAULT_OUTPUT_DIR = Path("outputs/aligned_lighting")
DEFAULT_FRAME_INDICES = (0, 49, 98, 147)


def to_u8_rgb(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)


def to_u8_normalized_rgb(rgb: torch.Tensor, valid_mask: torch.Tensor) -> np.ndarray:
    data = rgb.detach().cpu().float()
    valid = valid_mask.detach().cpu().to(torch.bool)
    out = torch.zeros_like(data)
    if bool(valid.any()):
        selected = data[valid]
        hi = torch.clamp(selected.max(), min=1e-8)
        out[valid] = (selected / hi).clamp(0.0, 1.0)
    return (out.numpy() * 255.0).astype(np.uint8)


def make_lighting_contact_sheet(records: list[dict[str, Any]], output_path: str | Path) -> None:
    cell_w = 178
    cell_h = 142
    label_w = 190
    header_h = 36
    labels = ["Base RGB", "Lambertian", "Blinn-Phong", "Specular", "Pseudo Normal"]
    sheet = Image.new("RGB", (label_w + len(labels) * cell_w, header_h + max(len(records), 1) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), "DXGL Apple dataset-agnostic deferred lighting validation", fill=(0, 0, 0))
    for row, record in enumerate(records):
        y = header_h + row * cell_h
        draw.text(
            (8, y + 10),
            f"frame {record['frame_index']}\n"
            f"valid={record['valid_pixels']}\n"
            f"spec mean={record['specular_mean']:.4f}",
            fill=(0, 0, 0),
        )
        for col, label in enumerate(labels):
            x = label_w + col * cell_w
            draw.text((x + 6, y + 6), label, fill=(0, 0, 0))
            image = Image.open(record["images"][label]).convert("RGB")
            _paste_thumbnail(sheet, draw, image, (x + 6, y + 24), (162, 108))
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dataset-agnostic Blinn-Phong lighting on aligned DXGL Apple.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--splat", type=Path, default=DEFAULT_SPLAT_PATH)
    parser.add_argument("--frame-indices", type=parse_int_list, default=list(DEFAULT_FRAME_INDICES))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-gaussians", type=int, default=0)
    parser.add_argument("--num-lights", type=int, default=128)
    parser.add_argument("--light-seed", type=int, default=2027)
    parser.add_argument("--ambient", type=float, default=0.2)
    parser.add_argument("--specular-strength", type=float, default=0.15)
    parser.add_argument("--shininess", type=float, default=24.0)
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
    asset = load_dxgl_aligned_asset(dataset_root, args.splat, device=device, max_gaussians=max_gaussians)

    selected_indices = list(args.frame_indices)
    if min(selected_indices) < 0 or max(selected_indices) >= asset.transforms.frame_count:
        raise ValueError(f"Selected frame indices {selected_indices} exceed frame count {asset.transforms.frame_count}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    light_info: dict[str, object] | None = None
    for frame_index in selected_indices:
        frame = asset.transforms.frames[frame_index]
        camera = scale_camera(frame.camera, args.width, args.height)
        render_buffers = render_rgbd(asset.loaded.scene, camera)
        gbuffer = make_pseudo_gbuffer(render_buffers, camera)
        lights, light_info = make_asset_scaled_point_lights(gbuffer, count=args.num_lights, seed=args.light_seed, device=device)
        lambertian = shade_deferred_lambertian(gbuffer, lights, ambient=args.ambient)
        blinn = shade_deferred_blinn_phong(
            gbuffer,
            lights,
            ambient=args.ambient,
            specular_strength=args.specular_strength,
            shininess=args.shininess,
        )

        prefix = f"dxgl_lighting_frame_{frame_index:03d}"
        paths = {
            "Base RGB": args.output_dir / f"{prefix}_base_rgb.png",
            "Lambertian": args.output_dir / f"{prefix}_lambertian.png",
            "Blinn-Phong": args.output_dir / f"{prefix}_blinn_phong.png",
            "Specular": args.output_dir / f"{prefix}_specular.png",
            "Pseudo Normal": args.output_dir / f"{prefix}_normal.png",
        }
        imageio.imwrite(paths["Base RGB"], to_u8_rgb(gbuffer.rgb))
        imageio.imwrite(paths["Lambertian"], to_u8_rgb(lambertian.composite_rgb))
        imageio.imwrite(paths["Blinn-Phong"], to_u8_rgb(blinn.composite_rgb))
        imageio.imwrite(paths["Specular"], to_u8_normalized_rgb(blinn.specular_rgb, blinn.valid_mask))
        imageio.imwrite(paths["Pseudo Normal"], to_u8_normal(gbuffer.normal_cam, blinn.valid_mask))

        valid_pixels = int(blinn.valid_mask.sum().detach().cpu())
        specular_mean = float(blinn.specular_rgb[blinn.valid_mask].mean().detach().cpu()) if valid_pixels > 0 else 0.0
        records.append(
            {
                "frame_index": frame_index,
                "valid_pixels": valid_pixels,
                "images": {key: str(path) for key, path in paths.items()},
                "specular_mean": specular_mean,
                "lambertian_mean": float(lambertian.composite_rgb[lambertian.valid_mask].mean().detach().cpu()) if valid_pixels > 0 else 0.0,
                "blinn_phong_mean": float(blinn.composite_rgb[blinn.valid_mask].mean().detach().cpu()) if valid_pixels > 0 else 0.0,
            }
        )

    contact_path = args.output_dir / "dxgl_blinn_phong_lighting_contact.png"
    make_lighting_contact_sheet(records, contact_path)
    summary = {
        "version": 1,
        "dataset": "dxgl_polyhaven_10_apple",
        "dataset_role": "validation demo only; shader is dataset-agnostic",
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
        "lights": light_info,
        "shader": {
            "ambient": args.ambient,
            "specular_strength": args.specular_strength,
            "shininess": args.shininess,
            "two_sided": True,
        },
        "selected_frame_indices": selected_indices,
        "frames": records,
        "contact_sheet": str(contact_path),
    }
    summary_path = args.output_dir / "dxgl_blinn_phong_lighting_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"splat:       {args.splat}")
    print(f"frames:      {selected_indices}")
    print(f"lights:      {args.num_lights}")
    print(f"specular:    strength={args.specular_strength}, shininess={args.shininess}")
    print(f"wrote:       {contact_path.resolve()}")
    print(f"wrote:       {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
