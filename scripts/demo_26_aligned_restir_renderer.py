from __future__ import annotations

import argparse
import csv
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

from restir_gs.lighting.asset_lights import make_asset_scaled_world_lights
from restir_gs.render.aligned_asset_registry import (
    DEFAULT_MANIFEST_PATH,
    get_aligned_asset_spec,
    load_aligned_asset_manifest,
    load_registered_aligned_asset,
    resolve_aligned_asset_paths,
    resolve_requested_asset_ids,
)
from restir_gs.render.dxgl_asset import scale_camera
from restir_gs.restir.renderer import (
    RestirHistory,
    RestirRenderSettings,
    all_numeric_finite,
    make_restir_metric_rows,
    render_restir_frame,
    summarize_restir_rows,
)


DEFAULT_OUTPUT_DIR = Path("outputs/aligned_restir")


def parse_asset_ids(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"Expected at least one asset id, got {text!r}")
    return values


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"Expected at least one frame index, got {text!r}")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError(f"Expected non-negative frame indices, got {values}")
    return values


def to_u8_rgb(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)


def to_u8_mask(mask: torch.Tensor) -> np.ndarray:
    return (mask.detach().cpu().to(torch.uint8).numpy() * 255).astype(np.uint8)


def to_u8_scalar(values: torch.Tensor, mask: torch.Tensor | None = None) -> np.ndarray:
    data = values.detach().cpu().float()
    valid = torch.isfinite(data)
    if mask is not None:
        valid = valid & mask.detach().cpu().to(torch.bool)
    out = torch.zeros_like(data)
    if bool(valid.any()):
        selected = data[valid]
        lo = selected.min()
        hi = selected.max()
        denom = hi - lo if float(hi - lo) > 1e-8 else torch.tensor(1.0)
        out[valid] = (selected - lo) / denom
    return (out.numpy() * 255.0).astype(np.uint8)


def make_abs_error_image(estimate: torch.Tensor, reference: torch.Tensor, valid_mask: torch.Tensor) -> np.ndarray:
    error = torch.abs(estimate.detach().cpu() - reference.detach().cpu()).mean(dim=-1)
    return to_u8_scalar(error, valid_mask.detach().cpu())


def write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    if not rows:
        raise RuntimeError("Cannot write empty aligned ReSTIR renderer CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_final_previews(output_dir: Path, result) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "reference": output_dir / "final_reference.png",
        "initial_ris": output_dir / "final_initial_ris.png",
        "temporal_ris": output_dir / "final_temporal_ris.png",
        "reuse_mask": output_dir / "final_reuse_mask.png",
        "motion_magnitude": output_dir / "final_motion_magnitude.png",
        "initial_abs_error": output_dir / "final_initial_abs_error.png",
        "temporal_abs_error": output_dir / "final_temporal_abs_error.png",
    }
    imageio.imwrite(paths["reference"], to_u8_rgb(result.reference.composite_rgb))
    imageio.imwrite(paths["initial_ris"], to_u8_rgb(result.initial.composite_rgb))
    imageio.imwrite(paths["temporal_ris"], to_u8_rgb(result.temporal.composite_rgb))
    imageio.imwrite(paths["reuse_mask"], to_u8_mask(result.lookup.valid_mask))
    imageio.imwrite(paths["motion_magnitude"], to_u8_scalar(torch.linalg.norm(result.lookup.motion_pixels, dim=-1), result.lookup.valid_mask))
    imageio.imwrite(paths["initial_abs_error"], make_abs_error_image(result.initial.contribution_rgb, result.reference.diffuse_rgb, result.reference.valid_mask))
    imageio.imwrite(paths["temporal_abs_error"], make_abs_error_image(result.temporal.contribution_rgb, result.reference.diffuse_rgb, result.reference.valid_mask))
    return {key: str(path) for key, path in paths.items()}


def make_contact_sheet(records: list[dict[str, Any]], output_path: Path) -> None:
    cell_w = 160
    cell_h = 132
    label_w = 182
    header_h = 34
    labels = ["Reference", "Initial RIS", "Temporal RIS", "Reuse Mask", "Initial Error", "Temporal Error"]
    sheet = Image.new("RGB", (label_w + len(labels) * cell_w, header_h + max(len(records), 1) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 9), "Aligned ReSTIR renderer path", fill=(0, 0, 0))
    for row, record in enumerate(records):
        y = header_h + row * cell_h
        draw.text(
            (8, y + 8),
            f"{record['asset_id']}\nframe {record['frame_index']}\n"
            f"reuse={record['reuse_fraction']:.3f}\n"
            f"init={record['initial_mae']:.4f}\n"
            f"temp={record['temporal_mae']:.4f}",
            fill=(0, 0, 0),
        )
        for col, label in enumerate(labels):
            x = label_w + col * cell_w
            draw.text((x + 6, y + 6), label, fill=(0, 0, 0))
            _paste_thumbnail(sheet, draw, record["images"][label], (x + 6, y + 24), (146, 96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _paste_thumbnail(sheet: Image.Image, draw: ImageDraw.ImageDraw, image: Image.Image, xy: tuple[int, int], size: tuple[int, int]) -> None:
    x, y = xy
    width, height = size
    thumb = image.copy().convert("RGB")
    thumb.thumbnail(size)
    sheet.paste(thumb, (x + (width - thumb.width) // 2, y + (height - thumb.height) // 2))
    draw.rectangle((x, y, x + width, y + height), outline=(180, 180, 180))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the registry-driven aligned ReSTIR renderer path.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--asset-ids", type=parse_asset_ids, default=None)
    parser.add_argument("--asset-set", default=None)
    parser.add_argument("--frame-indices", type=parse_int_list, default=None)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-gaussians", type=int, default=None)
    parser.add_argument("--num-lights", type=int, default=128)
    parser.add_argument("--light-seed", type=int, default=2027)
    parser.add_argument("--light-bbox-percentile", type=float, default=0.98)
    parser.add_argument("--light-radius-scale", type=float, default=1.25)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--candidate-seed-base", type=int, default=31100)
    parser.add_argument("--initial-selection-seed-base", type=int, default=32100)
    parser.add_argument("--temporal-selection-seed-base", type=int, default=33100)
    parser.add_argument("--depth-tolerance", type=float, default=0.05)
    parser.add_argument("--ambient", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError(f"Expected positive render size, got {args.width}x{args.height}")

    manifest = load_aligned_asset_manifest(args.manifest)
    asset_ids = resolve_requested_asset_ids(manifest, asset_ids=args.asset_ids, asset_set=args.asset_set)
    device = torch.device(args.device)
    settings = RestirRenderSettings(
        candidate_count=args.candidate_count,
        candidate_seed_base=args.candidate_seed_base,
        initial_selection_seed_base=args.initial_selection_seed_base,
        temporal_selection_seed_base=args.temporal_selection_seed_base,
        depth_tolerance=args.depth_tolerance,
        ambient=args.ambient,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, int | float | str]] = []
    asset_summaries: list[dict[str, Any]] = []
    for asset_id in asset_ids:
        spec = get_aligned_asset_spec(manifest, asset_id)
        resolved = resolve_aligned_asset_paths(spec, repo_root=manifest.repo_root)
        asset = load_registered_aligned_asset(resolved, device=device, max_gaussians_override=args.max_gaussians)
        frame_indices = list(args.frame_indices) if args.frame_indices is not None else list(spec.temporal_window)
        if min(frame_indices) < 0 or max(frame_indices) >= asset.transforms.frame_count:
            raise ValueError(f"{asset_id} frame indices {frame_indices} exceed frame count {asset.transforms.frame_count}")
        world_lights, light_info = make_asset_scaled_world_lights(
            asset.loaded.scene.means,
            count=args.num_lights,
            seed=args.light_seed,
            bbox_percentile=args.light_bbox_percentile,
            radius_scale=args.light_radius_scale,
            device=device,
        )

        asset_output_dir = args.output_dir / asset_id
        previous: RestirHistory | None = None
        contact_records: list[dict[str, Any]] = []
        frame_records: list[dict[str, Any]] = []
        first_frame_temporal_equals_initial = False
        last_result = None
        for offset, frame_index in enumerate(frame_indices):
            frame = asset.transforms.frames[frame_index]
            camera = scale_camera(frame.camera, args.width, args.height)
            result = render_restir_frame(
                asset.loaded.scene,
                camera,
                world_lights,
                frame_index=frame_index,
                settings=settings,
                previous_history=previous,
            )
            frame_rows = make_restir_metric_rows(asset_id, result, settings)
            rows.extend(frame_rows)
            contribution_rows = {
                str(row["estimator"]): row
                for row in frame_rows
                if row["reference_quantity"] == "contribution_rgb"
            }
            reuse_fraction = float(contribution_rows["temporal_ris"]["reuse_fraction"])
            if offset == 0:
                first_frame_temporal_equals_initial = bool(
                    torch.allclose(result.temporal.contribution_rgb, result.initial.contribution_rgb)
                    and torch.allclose(result.temporal.composite_rgb, result.initial.composite_rgb)
                )
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "valid_pixels": int(result.reference.valid_mask.sum().detach().cpu()),
                    "reuse_pixels": int(result.lookup.valid_mask.sum().detach().cpu()),
                    "reuse_fraction": reuse_fraction,
                    "initial_contribution_mae": contribution_rows["initial_ris"]["mae"],
                    "temporal_contribution_mae": contribution_rows["temporal_ris"]["mae"],
                    "temporal_m_mean": contribution_rows["temporal_ris"]["reservoir_m_mean"],
                }
            )
            contact_records.append(
                {
                    "asset_id": asset_id,
                    "frame_index": frame_index,
                    "reuse_fraction": reuse_fraction,
                    "initial_mae": float(contribution_rows["initial_ris"]["mae"]),
                    "temporal_mae": float(contribution_rows["temporal_ris"]["mae"]),
                    "images": {
                        "Reference": Image.fromarray(to_u8_rgb(result.reference.composite_rgb)),
                        "Initial RIS": Image.fromarray(to_u8_rgb(result.initial.composite_rgb)),
                        "Temporal RIS": Image.fromarray(to_u8_rgb(result.temporal.composite_rgb)),
                        "Reuse Mask": Image.fromarray(to_u8_mask(result.lookup.valid_mask)).convert("RGB"),
                        "Initial Error": Image.fromarray(
                            make_abs_error_image(result.initial.contribution_rgb, result.reference.diffuse_rgb, result.reference.valid_mask)
                        ).convert("RGB"),
                        "Temporal Error": Image.fromarray(
                            make_abs_error_image(result.temporal.contribution_rgb, result.reference.diffuse_rgb, result.reference.valid_mask)
                        ).convert("RGB"),
                    },
                }
            )
            previous = result.history
            last_result = result

        if last_result is None:
            raise RuntimeError(f"{asset_id} produced no ReSTIR frames.")
        preview_paths = save_final_previews(asset_output_dir, last_result)
        contact_path = asset_output_dir / "contact.png"
        make_contact_sheet(contact_records, contact_path)
        asset_summaries.append(
            {
                "asset_id": asset_id,
                "dataset_type": spec.dataset_type,
                "dataset_root": str(resolved.dataset_root),
                "splat_path": str(resolved.splat_path),
                "loaded_count": asset.loaded.stats.loaded_count,
                "original_count": asset.loaded.stats.original_count,
                "frame_indices": frame_indices,
                "first_frame_temporal_equals_initial": first_frame_temporal_equals_initial,
                "light_info": light_info,
                "frames": frame_records,
                "outputs": {
                    "contact_sheet": str(contact_path),
                    "final_previews": preview_paths,
                },
            }
        )

    csv_path = args.output_dir / "restir_renderer_rows.csv"
    summary_path = args.output_dir / "restir_renderer_summary.json"
    write_csv(csv_path, rows)
    summary = {
        "version": 1,
        "manifest": str(args.manifest),
        "asset_ids": asset_ids,
        "render": {"width": args.width, "height": args.height, "device": str(device)},
        "settings": {
            "num_lights": args.num_lights,
            "light_seed": args.light_seed,
            "light_space": "world",
            "light_policy": "asset_scaled_spherical_shell",
            "light_bbox_percentile": args.light_bbox_percentile,
            "light_radius_scale": args.light_radius_scale,
            "candidate_count": args.candidate_count,
            "candidate_seed_base": args.candidate_seed_base,
            "initial_selection_seed_base": args.initial_selection_seed_base,
            "temporal_selection_seed_base": args.temporal_selection_seed_base,
            "depth_tolerance": args.depth_tolerance,
            "ambient": args.ambient,
            "target_mode": "diffuse",
            "proposal": "geometric",
        },
        "row_count": len(rows),
        "all_numeric_finite": all_numeric_finite(rows),
        "assets": asset_summaries,
        "summary": summarize_restir_rows(rows),
        "outputs": {"csv": str(csv_path), "summary": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"assets:    {asset_ids}")
    print(f"rows:      {len(rows)}")
    print(f"finite:    {summary['all_numeric_finite']}")
    print(f"wrote:     {csv_path.resolve()}")
    print(f"wrote:     {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
