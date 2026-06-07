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

from restir_gs.eval.ris_ablation import compute_error_metrics
from restir_gs.lighting.asset_lights import make_asset_scaled_world_lights, world_lights_to_camera_lights
from restir_gs.lighting.deferred import LightingBuffers, shade_deferred_lambertian
from restir_gs.render.dxgl_asset import load_dxgl_aligned_asset, scale_camera
from restir_gs.render.gbuffer import GBuffer, make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.render.scene_normalization import scene_normalization_to_dict
from restir_gs.render.synthetic_scene import PinholeCamera
from restir_gs.restir.initial import LightingEstimatorBuffers, estimate_ris_initial_lighting
from restir_gs.restir.proposal import compute_geometric_proposal_distribution, sample_light_candidates_from_distribution
from restir_gs.restir.temporal import (
    TemporalLookup,
    TemporalReservoirState,
    combine_temporal_reservoirs,
    reproject_current_to_previous,
    temporal_reservoir_from_initial,
)
from scripts.demo_17_dxgl_aligned_intake import parse_int_list
from scripts.download_dxgl_apple import DEFAULT_EXTRACT_DIR, find_dxgl_dataset_root, validate_dxgl_dataset_root
from scripts.download_dxgl_apple_splat import DEFAULT_SPLAT_PATH, validate_dxgl_splat_file


DEFAULT_OUTPUT_DIR = Path("outputs/aligned_temporal")
DEFAULT_FRAME_INDICES = "45,46,47,48,49,50,51,52,53"


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


def make_temporal_contact_sheet(records: list[dict[str, Any]], output_path: Path) -> None:
    cell_w = 170
    cell_h = 136
    label_w = 184
    header_h = 36
    labels = ["Reference", "Initial RIS", "Temporal RIS", "Reuse Mask", "Motion"]
    sheet = Image.new("RGB", (label_w + len(labels) * cell_w, header_h + max(len(records), 1) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), "DXGL Apple aligned temporal reuse smoke", fill=(0, 0, 0))
    for row, record in enumerate(records):
        y = header_h + row * cell_h
        draw.text(
            (8, y + 10),
            f"frame {record['frame_index']}\n"
            f"reuse={record['reuse_fraction']:.3f}\n"
            f"init MAE={record['initial_contribution_mae']:.4f}\n"
            f"temp MAE={record['temporal_contribution_mae']:.4f}",
            fill=(0, 0, 0),
        )
        for col, label in enumerate(labels):
            x = label_w + col * cell_w
            draw.text((x + 6, y + 6), label, fill=(0, 0, 0))
            image = record["images"][label]
            _paste_thumbnail(sheet, draw, image, (x + 6, y + 24), (156, 104))
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
        raise RuntimeError("Cannot write empty temporal metrics CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_metric_rows(
    frame_index: int,
    initial: LightingEstimatorBuffers,
    temporal: LightingEstimatorBuffers,
    reference: LightingBuffers,
    lookup: TemporalLookup,
    temporal_reservoir: TemporalReservoirState,
) -> list[dict[str, int | float | str]]:
    valid_mask = reference.valid_mask
    valid_pixels = int(valid_mask.sum().detach().cpu())
    reuse_pixels = int(lookup.valid_mask.sum().detach().cpu())
    reuse_fraction = reuse_pixels / float(max(valid_pixels, 1))
    mean_depth_error = _masked_mean(lookup.relative_depth_error, lookup.valid_mask)
    motion_magnitude = torch.linalg.norm(lookup.motion_pixels, dim=-1)
    mean_motion = _masked_mean(motion_magnitude, lookup.valid_mask)
    m_valid = temporal_reservoir.valid_mask
    m_values = temporal_reservoir.M.detach().cpu()[m_valid.detach().cpu()]
    m_mean = float(m_values.float().mean()) if m_values.numel() > 0 else 0.0
    m_max = int(m_values.max()) if m_values.numel() > 0 else 0

    rows: list[dict[str, int | float | str]] = []
    for estimator, buffers in (("initial_ris", initial), ("temporal_ris", temporal)):
        for quantity, estimate, target in (
            ("contribution_rgb", buffers.contribution_rgb, reference.diffuse_rgb),
            ("composite_rgb", buffers.composite_rgb, reference.composite_rgb),
        ):
            row: dict[str, int | float | str] = {
                "frame_index": frame_index,
                "light_space": "world",
                "light_policy": "asset_scaled_spherical_shell",
                "estimator": estimator,
                "reference_quantity": quantity,
                "valid_pixels": valid_pixels,
                "reuse_pixels": reuse_pixels,
                "reuse_fraction": reuse_fraction,
                "mean_relative_depth_error": mean_depth_error,
                "mean_motion_pixels": mean_motion,
                "reservoir_m_mean": m_mean,
                "reservoir_m_max": m_max,
            }
            row.update(compute_error_metrics(estimate, target, valid_mask))
            rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, int | float | str]]) -> list[dict[str, int | float | str]]:
    groups: dict[tuple[str, str], list[dict[str, int | float | str]]] = {}
    for row in rows:
        key = (str(row["estimator"]), str(row["reference_quantity"]))
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, int | float | str]] = []
    for key in sorted(groups):
        estimator, quantity = key
        group = groups[key]
        mae = [float(row["mae"]) for row in group]
        rmse = [float(row["rmse"]) for row in group]
        summary.append(
            {
                "estimator": estimator,
                "reference_quantity": quantity,
                "frame_count": len(group),
                "mae_mean": float(np.mean(mae)),
                "rmse_mean": float(np.mean(rmse)),
            }
        )
    return summary


def save_final_previews(
    output_dir: Path,
    reference: LightingBuffers,
    initial: LightingEstimatorBuffers,
    temporal: LightingEstimatorBuffers,
    lookup: TemporalLookup,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "reference": output_dir / "dxgl_temporal_final_reference.png",
        "initial_ris": output_dir / "dxgl_temporal_final_initial_ris.png",
        "temporal_ris": output_dir / "dxgl_temporal_final_temporal_ris.png",
        "reuse_mask": output_dir / "dxgl_temporal_final_reuse_mask.png",
        "motion_magnitude": output_dir / "dxgl_temporal_final_motion_magnitude.png",
        "initial_abs_error": output_dir / "dxgl_temporal_final_initial_abs_error.png",
        "temporal_abs_error": output_dir / "dxgl_temporal_final_temporal_abs_error.png",
    }
    imageio.imwrite(paths["reference"], to_u8_rgb(reference.composite_rgb))
    imageio.imwrite(paths["initial_ris"], to_u8_rgb(initial.composite_rgb))
    imageio.imwrite(paths["temporal_ris"], to_u8_rgb(temporal.composite_rgb))
    imageio.imwrite(paths["reuse_mask"], to_u8_mask(lookup.valid_mask))
    imageio.imwrite(paths["motion_magnitude"], to_u8_scalar(torch.linalg.norm(lookup.motion_pixels, dim=-1), lookup.valid_mask))
    imageio.imwrite(paths["initial_abs_error"], make_abs_error_image(initial.contribution_rgb, reference.diffuse_rgb, reference.valid_mask))
    imageio.imwrite(paths["temporal_abs_error"], make_abs_error_image(temporal.contribution_rgb, reference.diffuse_rgb, reference.valid_mask))
    return {key: str(path) for key, path in paths.items()}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.detach().cpu().to(torch.bool) & torch.isfinite(values.detach().cpu())
    if not bool(valid.any()):
        return 0.0
    return float(values.detach().cpu()[valid].float().mean())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run aligned DXGL temporal reprojection and reservoir reuse smoke.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--splat", type=Path, default=DEFAULT_SPLAT_PATH)
    parser.add_argument("--frame-indices", type=parse_int_list, default=parse_int_list(DEFAULT_FRAME_INDICES))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-gaussians", type=int, default=0)
    parser.add_argument("--num-lights", type=int, default=128)
    parser.add_argument("--light-seed", type=int, default=2027)
    parser.add_argument("--light-bbox-percentile", type=float, default=0.98)
    parser.add_argument("--light-radius-scale", type=float, default=1.25)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--candidate-seed-base", type=int, default=17100)
    parser.add_argument("--initial-selection-seed-base", type=int, default=18100)
    parser.add_argument("--temporal-selection-seed-base", type=int, default=19100)
    parser.add_argument("--depth-tolerance", type=float, default=0.05)
    parser.add_argument("--ambient", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if args.candidate_count <= 0:
        raise ValueError(f"Expected positive candidate_count, got {args.candidate_count}")

    dataset_root = find_dxgl_dataset_root(args.dataset_root, required=True)
    validate_dxgl_dataset_root(dataset_root, raise_on_missing=True)
    splat_validation = validate_dxgl_splat_file(args.splat)
    device = torch.device(args.device)
    max_gaussians = None if args.max_gaussians <= 0 else args.max_gaussians
    asset = load_dxgl_aligned_asset(dataset_root, args.splat, device=device, max_gaussians=max_gaussians)
    frame_indices = list(args.frame_indices)
    if min(frame_indices) < 0 or max(frame_indices) >= asset.transforms.frame_count:
        raise ValueError(f"Selected frame indices {frame_indices} exceed frame count {asset.transforms.frame_count}")
    world_lights, light_info = make_asset_scaled_world_lights(
        asset.loaded.scene.means,
        count=args.num_lights,
        seed=args.light_seed,
        bbox_percentile=args.light_bbox_percentile,
        radius_scale=args.light_radius_scale,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | float | str]] = []
    frame_records: list[dict[str, Any]] = []
    contact_records: list[dict[str, Any]] = []
    prev_gbuffer: GBuffer | None = None
    prev_camera: PinholeCamera | None = None
    prev_temporal_reservoir: TemporalReservoirState | None = None
    last_reference: LightingBuffers | None = None
    last_initial: LightingEstimatorBuffers | None = None
    last_temporal: LightingEstimatorBuffers | None = None
    last_lookup: TemporalLookup | None = None
    first_frame_temporal_equals_initial = False

    for frame_index in frame_indices:
        frame = asset.transforms.frames[frame_index]
        camera = scale_camera(frame.camera, args.width, args.height)
        render_buffers = render_rgbd(asset.loaded.scene, camera)
        gbuffer = make_pseudo_gbuffer(render_buffers, camera)
        lights = world_lights_to_camera_lights(world_lights, camera)
        reference = shade_deferred_lambertian(gbuffer, lights, ambient=args.ambient)
        proposal_distribution = compute_geometric_proposal_distribution(gbuffer, lights)
        samples = sample_light_candidates_from_distribution(
            proposal_distribution,
            args.candidate_count,
            seed=args.candidate_seed_base + frame_index,
            device=device,
        )
        initial, initial_reservoir = estimate_ris_initial_lighting(
            gbuffer,
            lights,
            samples.light_indices,
            selection_seed=args.initial_selection_seed_base + frame_index,
            ambient=args.ambient,
            proposal_probs=samples.proposal_probs,
            target_mode="diffuse",
        )

        if prev_gbuffer is None or prev_camera is None or prev_temporal_reservoir is None:
            lookup = _empty_lookup(gbuffer)
            temporal = initial
            temporal_reservoir = temporal_reservoir_from_initial(initial_reservoir)
            first_frame_temporal_equals_initial = bool(
                torch.allclose(temporal.contribution_rgb, initial.contribution_rgb)
                and torch.allclose(temporal.composite_rgb, initial.composite_rgb)
            )
        else:
            lookup = reproject_current_to_previous(
                gbuffer,
                camera,
                prev_gbuffer,
                prev_camera,
                depth_tolerance=args.depth_tolerance,
            )
            temporal, temporal_reservoir = combine_temporal_reservoirs(
                gbuffer,
                lights,
                initial,
                initial_reservoir,
                prev_temporal_reservoir,
                lookup,
                selection_seed=args.temporal_selection_seed_base + frame_index,
                ambient=args.ambient,
                target_mode="diffuse",
            )

        frame_rows = make_metric_rows(frame_index, initial, temporal, reference, lookup, temporal_reservoir)
        rows.extend(frame_rows)
        contribution_rows = {str(row["estimator"]): row for row in frame_rows if row["reference_quantity"] == "contribution_rgb"}
        valid_pixels = int(reference.valid_mask.sum().detach().cpu())
        reuse_pixels = int(lookup.valid_mask.sum().detach().cpu())
        frame_records.append(
            {
                "frame_index": frame_index,
                "valid_pixels": valid_pixels,
                "reuse_pixels": reuse_pixels,
                "reuse_fraction": reuse_pixels / float(max(valid_pixels, 1)),
                "initial_contribution_mae": contribution_rows["initial_ris"]["mae"],
                "temporal_contribution_mae": contribution_rows["temporal_ris"]["mae"],
                "temporal_m_mean": next(row["reservoir_m_mean"] for row in frame_rows if row["estimator"] == "temporal_ris"),
            }
        )
        contact_records.append(
            {
                "frame_index": frame_index,
                "reuse_fraction": reuse_pixels / float(max(valid_pixels, 1)),
                "initial_contribution_mae": contribution_rows["initial_ris"]["mae"],
                "temporal_contribution_mae": contribution_rows["temporal_ris"]["mae"],
                "images": {
                    "Reference": Image.fromarray(to_u8_rgb(reference.composite_rgb)),
                    "Initial RIS": Image.fromarray(to_u8_rgb(initial.composite_rgb)),
                    "Temporal RIS": Image.fromarray(to_u8_rgb(temporal.composite_rgb)),
                    "Reuse Mask": Image.fromarray(to_u8_mask(lookup.valid_mask)).convert("RGB"),
                    "Motion": Image.fromarray(to_u8_scalar(torch.linalg.norm(lookup.motion_pixels, dim=-1), lookup.valid_mask)).convert("RGB"),
                },
            }
        )

        prev_gbuffer = gbuffer
        prev_camera = camera
        prev_temporal_reservoir = temporal_reservoir
        last_reference = reference
        last_initial = initial
        last_temporal = temporal
        last_lookup = lookup

    if last_reference is None or last_initial is None or last_temporal is None or last_lookup is None:
        raise RuntimeError("Temporal demo produced no frames.")

    csv_path = args.output_dir / "dxgl_temporal_reuse_metrics.csv"
    summary_path = args.output_dir / "dxgl_temporal_reuse_summary.json"
    contact_path = args.output_dir / "dxgl_temporal_reuse_contact.png"
    preview_paths = save_final_previews(args.output_dir, last_reference, last_initial, last_temporal, last_lookup)
    write_csv(csv_path, rows)
    make_temporal_contact_sheet(contact_records, contact_path)

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
        "settings": {
            "frame_indices": frame_indices,
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
        },
        "light_info": light_info,
        "first_frame_temporal_equals_initial": first_frame_temporal_equals_initial,
        "frames": frame_records,
        "summary": summarize_rows(rows),
        "outputs": {
            "csv": str(csv_path),
            "contact_sheet": str(contact_path),
            "final_previews": preview_paths,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"splat:      {args.splat}")
    print(f"frames:     {frame_indices}")
    print(f"rows:       {len(rows)}")
    print(f"first eq:   {first_frame_temporal_equals_initial}")
    print(f"wrote:      {csv_path.resolve()}")
    print(f"wrote:      {summary_path.resolve()}")
    print(f"wrote:      {contact_path.resolve()}")
    return 0


def _empty_lookup(gbuffer: GBuffer) -> TemporalLookup:
    height, width = gbuffer.depth.shape
    return TemporalLookup(
        prev_pixels=torch.zeros((height, width, 2), dtype=torch.long, device=gbuffer.rgb.device),
        valid_mask=torch.zeros((height, width), dtype=torch.bool, device=gbuffer.rgb.device),
        pre_gate_mask=torch.zeros((height, width), dtype=torch.bool, device=gbuffer.rgb.device),
        relative_depth_error=torch.full((height, width), float("inf"), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
        normal_dot=torch.zeros((height, width), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
        rgb_distance=torch.full((height, width), float("inf"), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
        motion_pixels=torch.zeros((height, width, 2), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
    )


if __name__ == "__main__":
    raise SystemExit(main())
