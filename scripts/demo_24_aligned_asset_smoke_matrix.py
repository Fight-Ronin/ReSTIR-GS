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

from restir_gs.eval.dxgl_sampling_benchmark import frame_alignment_metrics, run_sampling_benchmark_for_frame
from restir_gs.eval.gbuffer_validation import depth_metrics
from restir_gs.lighting.asset_lights import make_asset_scaled_world_lights, world_lights_to_camera_lights
from restir_gs.lighting.deferred import shade_deferred_blinn_phong, shade_deferred_lambertian
from restir_gs.render.aligned_asset_registry import (
    DEFAULT_MANIFEST_PATH,
    get_aligned_asset_spec,
    load_aligned_asset_manifest,
    load_registered_aligned_asset,
    resolve_aligned_asset_paths,
    resolve_requested_asset_ids,
)
from restir_gs.render.dxgl_asset import load_dxgl_frame_modalities, scale_camera
from restir_gs.render.gbuffer import GBuffer, make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.restir.initial import estimate_ris_initial_lighting
from restir_gs.restir.proposal import compute_geometric_proposal_distribution, sample_light_candidates_from_distribution
from restir_gs.restir.temporal import (
    TemporalReservoirState,
    combine_temporal_reservoirs,
    reproject_current_to_previous,
    temporal_reservoir_from_initial,
)


DEFAULT_OUTPUT_DIR = Path("outputs/aligned_smoke")


def parse_asset_ids(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected at least one asset id, got {text!r}")
    return values


def make_smoke_row(
    asset_id: str,
    dataset_type: str,
    stage: str,
    metric_name: str,
    metric_value: float,
    frame_index: int = -1,
    loaded_count: int = 0,
    original_count: int = 0,
    valid_pixels: int = 0,
) -> dict[str, int | float | str]:
    finite = bool(np.isfinite(float(metric_value)))
    return {
        "asset_id": asset_id,
        "dataset_type": dataset_type,
        "stage": stage,
        "frame_index": frame_index,
        "loaded_count": loaded_count,
        "original_count": original_count,
        "valid_pixels": valid_pixels,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "finite": str(finite).lower(),
    }


def to_u8_rgb(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)


def to_u8_alpha(alpha: torch.Tensor) -> np.ndarray:
    return (alpha.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)


def make_contact_sheet(records: list[dict[str, Any]], output_path: Path) -> None:
    cell_w = 160
    cell_h = 132
    label_w = 172
    header_h = 34
    labels = ["Reference", "Render", "Alpha", "Lambertian"]
    sheet = Image.new("RGB", (label_w + len(labels) * cell_w, header_h + max(len(records), 1) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 9), "Aligned asset smoke matrix", fill=(0, 0, 0))
    for row, record in enumerate(records):
        y = header_h + row * cell_h
        draw.text(
            (8, y + 8),
            f"{record['asset_id']}\nframe {record['frame_index']}\n"
            f"rgb={record['rgb_mae']:.4f}\nalpha={record['alpha_iou']:.3f}",
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


def write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    if not rows:
        raise RuntimeError("Cannot write empty aligned smoke CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small manifest-driven aligned asset smoke matrix.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--asset-ids", type=parse_asset_ids, default=None)
    parser.add_argument("--asset-set", default=None)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--max-gaussians", type=int, default=None)
    parser.add_argument("--num-lights", type=int, default=64)
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if args.candidate_count <= 0:
        raise ValueError(f"Expected positive candidate_count, got {args.candidate_count}")

    manifest = load_aligned_asset_manifest(args.manifest)
    asset_ids = resolve_requested_asset_ids(manifest, asset_ids=args.asset_ids, asset_set=args.asset_set)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, int | float | str]] = []
    summary_assets: list[dict[str, Any]] = []
    for asset_id in asset_ids:
        spec = get_aligned_asset_spec(manifest, asset_id)
        resolved = resolve_aligned_asset_paths(spec, repo_root=manifest.repo_root)
        asset = load_registered_aligned_asset(resolved, device=device, max_gaussians_override=args.max_gaussians)
        loaded_count = asset.loaded.stats.loaded_count
        original_count = asset.loaded.stats.original_count
        world_lights, light_info = make_asset_scaled_world_lights(asset.loaded.scene.means, args.num_lights, device=device)

        asset_output_dir = args.output_dir / asset_id
        asset_output_dir.mkdir(parents=True, exist_ok=True)
        frame_records: list[dict[str, Any]] = []
        gbuffer_cache: dict[int, tuple[GBuffer, Any, Any]] = {}
        for frame_index in spec.default_frames:
            frame = asset.transforms.frames[frame_index]
            camera = scale_camera(frame.camera, args.width, args.height)
            render_buffers = render_rgbd(asset.loaded.scene, camera)
            gbuffer = make_pseudo_gbuffer(render_buffers, camera)
            lights = world_lights_to_camera_lights(world_lights, camera)
            lambertian = shade_deferred_lambertian(gbuffer, lights)
            modalities = load_dxgl_frame_modalities(
                frame,
                args.width,
                args.height,
                scene_scale=asset.normalization.scale if asset.normalization is not None else None,
            )
            alignment = frame_alignment_metrics(gbuffer.rgb, gbuffer.alpha, modalities.rgb, modalities.mask)
            render_mask = gbuffer.valid_mask.detach().cpu()
            ref_mask = modalities.mask.detach().cpu()
            common_mask = render_mask & ref_mask
            depth_mae = 0.0
            if modalities.depth_normalized is not None:
                depth_mae = float(depth_metrics(gbuffer.depth.detach().cpu(), modalities.depth_normalized, common_mask)["mae"])
            valid_pixels = int((gbuffer.valid_mask & gbuffer.normal_mask).sum().detach().cpu())
            rows.extend(
                [
                    make_smoke_row(asset_id, spec.dataset_type, "gbuffer", "rgb_mae", alignment["rgb_mae_to_reference"], frame_index, loaded_count, original_count, valid_pixels),
                    make_smoke_row(asset_id, spec.dataset_type, "gbuffer", "alpha_iou", alignment["alpha_iou"], frame_index, loaded_count, original_count, valid_pixels),
                    make_smoke_row(asset_id, spec.dataset_type, "gbuffer", "depth_mae", depth_mae, frame_index, loaded_count, original_count, valid_pixels),
                    make_smoke_row(asset_id, spec.dataset_type, "lighting", "lambertian_mean", float(lambertian.composite_rgb[lambertian.valid_mask].mean().detach().cpu()) if valid_pixels > 0 else 0.0, frame_index, loaded_count, original_count, valid_pixels),
                ]
            )
            gbuffer_cache[frame_index] = (gbuffer, camera, lambertian)
            frame_records.append(
                {
                    "asset_id": asset_id,
                    "frame_index": frame_index,
                    "rgb_mae": alignment["rgb_mae_to_reference"],
                    "alpha_iou": alignment["alpha_iou"],
                    "images": {
                        "Reference": Image.fromarray(to_u8_rgb(modalities.rgb if modalities.rgb is not None else gbuffer.rgb)),
                        "Render": Image.fromarray(to_u8_rgb(gbuffer.rgb)),
                        "Alpha": Image.fromarray(to_u8_alpha(gbuffer.alpha)).convert("RGB"),
                        "Lambertian": Image.fromarray(to_u8_rgb(lambertian.composite_rgb)),
                    },
                }
            )

        sample_frame = spec.default_frames[0]
        sample_gbuffer, sample_camera, sample_lambertian = gbuffer_cache[sample_frame]
        sample_lights = world_lights_to_camera_lights(world_lights, sample_camera)
        sample_blinn = shade_deferred_blinn_phong(sample_gbuffer, sample_lights)
        sampling_rows = run_sampling_benchmark_for_frame(
            sample_gbuffer,
            sample_lights,
            sample_lambertian,
            sample_blinn,
            frame_index=sample_frame,
            k_values=[1],
            seed_count=1,
        )
        sampling_finite = all(np.isfinite(float(row["mae"])) and np.isfinite(float(row["rmse"])) for row in sampling_rows)
        rows.append(
            make_smoke_row(
                asset_id,
                spec.dataset_type,
                "sampling",
                "sampling_rows_finite",
                1.0 if sampling_finite else 0.0,
                sample_frame,
                loaded_count,
                original_count,
                int((sample_gbuffer.valid_mask & sample_gbuffer.normal_mask).sum().detach().cpu()),
            )
        )

        temporal_metrics = _run_tiny_temporal_smoke(asset, spec.temporal_window, world_lights, args, device)
        for name, value in temporal_metrics.items():
            rows.append(make_smoke_row(asset_id, spec.dataset_type, "temporal", name, value, -1, loaded_count, original_count, 0))

        contact_path = asset_output_dir / "contact.png"
        make_contact_sheet(frame_records, contact_path)
        summary_assets.append(
            {
                "asset_id": asset_id,
                "dataset_type": spec.dataset_type,
                "dataset_root": str(resolved.dataset_root),
                "splat_path": str(resolved.splat_path),
                "loaded_count": loaded_count,
                "original_count": original_count,
                "default_frames": spec.default_frames,
                "temporal_window": spec.temporal_window,
                "light_info": light_info,
                "contact_sheet": str(contact_path),
            }
        )

    csv_path = args.output_dir / "aligned_asset_smoke_rows.csv"
    summary_path = args.output_dir / "aligned_asset_smoke_summary.json"
    write_csv(csv_path, rows)
    summary = {
        "version": 1,
        "manifest": str(args.manifest),
        "asset_ids": asset_ids,
        "render": {"width": args.width, "height": args.height, "device": str(device)},
        "row_count": len(rows),
        "all_numeric_finite": all(row["finite"] == "true" for row in rows),
        "assets": summary_assets,
        "outputs": {"csv": str(csv_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"assets:    {asset_ids}")
    print(f"rows:      {len(rows)}")
    print(f"finite:    {summary['all_numeric_finite']}")
    print(f"wrote:     {csv_path.resolve()}")
    print(f"wrote:     {summary_path.resolve()}")
    return 0


def _run_tiny_temporal_smoke(asset, frame_indices: list[int], world_lights, args, device: torch.device) -> dict[str, float]:
    prev_gbuffer = None
    prev_camera = None
    prev_reservoir: TemporalReservoirState | None = None
    first_frame_equal = 0.0
    reuse_fractions: list[float] = []
    for offset, frame_index in enumerate(frame_indices[:3]):
        frame = asset.transforms.frames[frame_index]
        camera = scale_camera(frame.camera, args.width, args.height)
        render_buffers = render_rgbd(asset.loaded.scene, camera)
        gbuffer = make_pseudo_gbuffer(render_buffers, camera)
        lights = world_lights_to_camera_lights(world_lights, camera)
        proposal = compute_geometric_proposal_distribution(gbuffer, lights)
        samples = sample_light_candidates_from_distribution(proposal, args.candidate_count, seed=28000 + frame_index, device=device)
        initial, initial_reservoir = estimate_ris_initial_lighting(
            gbuffer,
            lights,
            samples.light_indices,
            selection_seed=29000 + frame_index,
            proposal_probs=samples.proposal_probs,
        )
        if prev_gbuffer is None or prev_camera is None or prev_reservoir is None:
            temporal = initial
            reservoir = temporal_reservoir_from_initial(initial_reservoir)
            first_frame_equal = 1.0 if torch.allclose(temporal.contribution_rgb, initial.contribution_rgb) else 0.0
            reuse_fractions.append(0.0)
        else:
            lookup = reproject_current_to_previous(gbuffer, camera, prev_gbuffer, prev_camera)
            temporal, reservoir = combine_temporal_reservoirs(
                gbuffer,
                lights,
                initial,
                initial_reservoir,
                prev_reservoir,
                lookup,
                selection_seed=30000 + frame_index,
            )
            valid_pixels = int((gbuffer.valid_mask & gbuffer.normal_mask).sum().detach().cpu())
            reuse_fractions.append(float(lookup.valid_mask.sum().detach().cpu()) / float(max(valid_pixels, 1)))
        prev_gbuffer = gbuffer
        prev_camera = camera
        prev_reservoir = reservoir
    later = reuse_fractions[1:] if len(reuse_fractions) > 1 else []
    return {
        "first_frame_equal_initial": first_frame_equal,
        "reuse_fraction_mean": float(np.mean(later)) if later else 0.0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
