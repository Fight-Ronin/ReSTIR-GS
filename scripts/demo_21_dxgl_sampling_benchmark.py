from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.eval.dxgl_sampling_benchmark import (
    expected_sampling_row_count,
    frame_alignment_metrics,
    parse_k_values,
    run_sampling_benchmark_for_frame,
    select_evenly_spaced_frames,
    summarize_sampling_rows,
)
from restir_gs.lighting.asset_lights import make_asset_scaled_point_lights
from restir_gs.lighting.deferred import shade_deferred_blinn_phong, shade_deferred_lambertian
from restir_gs.render.dxgl_asset import load_dxgl_aligned_asset, load_dxgl_frame_modalities, scale_camera
from restir_gs.render.gbuffer import make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.render.scene_normalization import scene_normalization_to_dict
from scripts.demo_17_dxgl_aligned_intake import parse_int_list
from scripts.download_dxgl_apple import DEFAULT_EXTRACT_DIR, find_dxgl_dataset_root, validate_dxgl_dataset_root
from scripts.download_dxgl_apple_splat import DEFAULT_SPLAT_PATH, validate_dxgl_splat_file


DEFAULT_OUTPUT_DIR = Path("outputs/aligned_sampling")
DEFAULT_VIEW_COUNT = 8
DEFAULT_K_VALUES = "1,2,4,8,16,32"


def to_u8_rgb(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)


def to_u8_alpha(alpha: torch.Tensor) -> np.ndarray:
    data = alpha.detach().cpu().clamp(0.0, 1.0).numpy()
    return (data * 255.0).astype(np.uint8)


def make_sampling_contact_sheet(records: list[dict[str, Any]], output_path: str | Path) -> None:
    cell_w = 172
    cell_h = 136
    label_w = 178
    header_h = 36
    labels = ["Reference RGB", "Render RGB", "Alpha", "Lambertian", "Blinn-Phong"]
    sheet = Image.new("RGB", (label_w + len(labels) * cell_w, header_h + max(len(records), 1) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), "DXGL Apple aligned sampling benchmark", fill=(0, 0, 0))
    for row, record in enumerate(records):
        y = header_h + row * cell_h
        draw.text(
            (8, y + 10),
            f"frame {record['frame_index']}\n"
            f"valid={record['valid_pixels']}\n"
            f"rgb MAE={record['rgb_mae_to_reference']:.4f}\n"
            f"alpha IoU={record['alpha_iou']:.3f}",
            fill=(0, 0, 0),
        )
        images = record["images"]
        for col, label in enumerate(labels):
            x = label_w + col * cell_w
            draw.text((x + 6, y + 6), label, fill=(0, 0, 0))
            _paste_thumbnail(sheet, draw, images[label], (x + 6, y + 24), (158, 104))
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
    thumb = image.copy().convert("RGB")
    thumb.thumbnail(size)
    sheet.paste(thumb, (x + (width - thumb.width) // 2, y + (height - thumb.height) // 2))
    draw.rectangle((x, y, x + width, y + height), outline=(180, 180, 180))


def _write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("Cannot write empty sampling benchmark CSV.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DXGL Apple aligned multi-frame MC/RIS sampling benchmark.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--splat", type=Path, default=DEFAULT_SPLAT_PATH)
    parser.add_argument("--frame-indices", type=parse_int_list, default=None)
    parser.add_argument("--view-count", type=int, default=DEFAULT_VIEW_COUNT)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-gaussians", type=int, default=0)
    parser.add_argument("--num-lights", type=int, default=128)
    parser.add_argument("--light-seed", type=int, default=2027)
    parser.add_argument("--k-values", type=parse_k_values, default=parse_k_values(DEFAULT_K_VALUES))
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--candidate-seed-base", type=int, default=15100)
    parser.add_argument("--selection-seed-base", type=int, default=16100)
    parser.add_argument("--ambient", type=float, default=0.2)
    parser.add_argument("--specular-strength", type=float, default=0.15)
    parser.add_argument("--shininess", type=float, default=24.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if args.seed_count <= 0:
        raise ValueError(f"Expected positive seed_count, got {args.seed_count}")

    dataset_root = find_dxgl_dataset_root(args.dataset_root, required=True)
    validate_dxgl_dataset_root(dataset_root, raise_on_missing=True)
    splat_validation = validate_dxgl_splat_file(args.splat)
    device = torch.device(args.device)
    max_gaussians = None if args.max_gaussians <= 0 else args.max_gaussians
    asset = load_dxgl_aligned_asset(dataset_root, args.splat, device=device, max_gaussians=max_gaussians)

    selected_indices = list(args.frame_indices) if args.frame_indices is not None else select_evenly_spaced_frames(
        asset.transforms.frame_count,
        args.view_count,
    )
    if min(selected_indices) < 0 or max(selected_indices) >= asset.transforms.frame_count:
        raise ValueError(f"Selected frame indices {selected_indices} exceed frame count {asset.transforms.frame_count}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, int | float | str]] = []
    frame_records: list[dict[str, Any]] = []
    contact_records: list[dict[str, Any]] = []
    light_info: dict[str, object] | None = None

    for frame_index in selected_indices:
        frame = asset.transforms.frames[frame_index]
        camera = scale_camera(frame.camera, args.width, args.height)
        modalities = load_dxgl_frame_modalities(
            frame,
            args.width,
            args.height,
            scene_scale=asset.normalization.scale if asset.normalization is not None else None,
        )
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
        alignment = frame_alignment_metrics(gbuffer.rgb, gbuffer.alpha, modalities.rgb, modalities.mask)
        rows = run_sampling_benchmark_for_frame(
            gbuffer,
            lights,
            lambertian,
            blinn,
            frame_index=frame_index,
            k_values=args.k_values,
            seed_count=args.seed_count,
            candidate_seed_base=args.candidate_seed_base,
            selection_seed_base=args.selection_seed_base,
            ambient=args.ambient,
            specular_strength=args.specular_strength,
            shininess=args.shininess,
            rgb_mae_to_reference=alignment["rgb_mae_to_reference"],
            alpha_iou=alignment["alpha_iou"],
        )
        all_rows.extend(rows)

        valid_pixels = int((gbuffer.valid_mask & gbuffer.normal_mask).sum().detach().cpu())
        frame_records.append(
            {
                "frame_index": frame_index,
                "valid_pixels": valid_pixels,
                "rgb_mae_to_reference": alignment["rgb_mae_to_reference"],
                "alpha_iou": alignment["alpha_iou"],
                "row_count": len(rows),
            }
        )
        contact_records.append(
            {
                "frame_index": frame_index,
                "valid_pixels": valid_pixels,
                "rgb_mae_to_reference": alignment["rgb_mae_to_reference"],
                "alpha_iou": alignment["alpha_iou"],
                "images": {
                    "Reference RGB": Image.fromarray(to_u8_rgb(modalities.rgb if modalities.rgb is not None else gbuffer.rgb)),
                    "Render RGB": Image.fromarray(to_u8_rgb(gbuffer.rgb)),
                    "Alpha": Image.fromarray(to_u8_alpha(gbuffer.alpha)).convert("RGB"),
                    "Lambertian": Image.fromarray(to_u8_rgb(lambertian.composite_rgb)),
                    "Blinn-Phong": Image.fromarray(to_u8_rgb(blinn.composite_rgb)),
                },
            }
        )

    expected_rows = expected_sampling_row_count(len(selected_indices), args.k_values, args.seed_count)
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} sampling rows, got {len(all_rows)}")

    csv_path = args.output_dir / "dxgl_sampling_rows.csv"
    json_path = args.output_dir / "dxgl_sampling_summary.json"
    contact_path = args.output_dir / "dxgl_sampling_contact.png"
    _write_csv(csv_path, all_rows)
    make_sampling_contact_sheet(contact_records, contact_path)

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
        "lights": light_info,
        "sampling": {
            "target_modes": ["diffuse", "blinn_phong"],
            "proposals": ["uniform", "geometric"],
            "estimators": ["mc", "ris"],
            "k_values": list(args.k_values),
            "seed_count": args.seed_count,
            "candidate_seed_base": args.candidate_seed_base,
            "selection_seed_base": args.selection_seed_base,
            "ambient": args.ambient,
            "specular_strength": args.specular_strength,
            "shininess": args.shininess,
            "row_count": len(all_rows),
            "expected_row_count": expected_rows,
        },
        "selected_frame_indices": selected_indices,
        "frames": frame_records,
        "summary": summarize_sampling_rows(all_rows),
        "outputs": {
            "csv": str(csv_path),
            "contact_sheet": str(contact_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"splat:       {args.splat}")
    print(f"frames:      {selected_indices}")
    print(f"row_count:   {len(all_rows)}")
    print(f"wrote:       {csv_path.resolve()}")
    print(f"wrote:       {json_path.resolve()}")
    print(f"wrote:       {contact_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
