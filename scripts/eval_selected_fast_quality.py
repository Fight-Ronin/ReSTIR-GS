from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import imageio.v2 as imageio
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.lighting.asset_lights import make_asset_scaled_world_lights
from restir_gs.lighting.visibility import make_shadow_map_bundle
from restir_gs.metrics import compute_rgb_error_metrics
from restir_gs.render.aligned_asset_registry import (
    DEFAULT_MANIFEST_PATH,
    get_aligned_asset_spec,
    load_aligned_asset_manifest,
    load_registered_aligned_asset,
    resolve_aligned_asset_paths,
    resolve_requested_asset_ids,
)
from restir_gs.render.dxgl_asset import scale_camera
from restir_gs.restir.renderer import RestirHistory, RestirRenderSettings, render_restir_frame
from scripts.bench_realtime_display_fps import render_selected_visibility_frame_timed, run_with_gpu_timing
from scripts.demo_26_aligned_restir_renderer import (
    make_abs_error_image,
    parse_asset_ids,
    parse_int_list,
    parse_optional_float,
    parse_optional_int,
    to_u8_mask,
    to_u8_rgb,
    to_u8_scalar,
)


DEFAULT_OUTPUT_DIR = Path("outputs/selected_fast_quality")


def write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    if not rows:
        raise RuntimeError("Cannot write empty selected-fast quality CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all().detach().cpu()):
        raise RuntimeError(f"{name} contains non-finite values.")


def summarize_numeric_fields(rows: list[dict[str, int | float | str]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    if not rows:
        return summary
    keys = rows[0].keys()
    for key in keys:
        values: list[float] = []
        for row in rows:
            value = row[key]
            if isinstance(value, (int, float)):
                values.append(float(value))
        if not values:
            continue
        sorted_values = sorted(values)
        mid = len(sorted_values) // 2
        median = sorted_values[mid] if len(sorted_values) % 2 else 0.5 * (sorted_values[mid - 1] + sorted_values[mid])
        p90_index = min(len(sorted_values) - 1, int(0.9 * (len(sorted_values) - 1)))
        summary[key] = {
            "count": len(values),
            "mean": sum(values) / float(len(values)),
            "median": median,
            "p90": sorted_values[p90_index],
            "min": sorted_values[0],
            "max": sorted_values[-1],
        }
    return summary


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def temporal_delta_metrics(
    current: torch.Tensor,
    previous: torch.Tensor | None,
    valid_mask: torch.Tensor,
) -> dict[str, float | int]:
    if previous is None:
        return {"mean_abs_frame_delta": 0.0, "valid_pixel_frame_delta": 0}
    if current.shape != previous.shape:
        raise ValueError(f"Expected temporal delta tensors to match, got {tuple(current.shape)} and {tuple(previous.shape)}")
    valid = valid_mask.to(device=current.device, dtype=torch.bool)
    if not bool(valid.any()):
        return {"mean_abs_frame_delta": 0.0, "valid_pixel_frame_delta": 0}
    delta = torch.abs(current - previous).mean(dim=-1)
    selected = delta[valid]
    return {
        "mean_abs_frame_delta": float(selected.mean().detach().cpu()),
        "valid_pixel_frame_delta": int(valid.sum().detach().cpu()),
    }


def make_abs_diff_image(a: torch.Tensor, b: torch.Tensor, valid_mask: torch.Tensor):
    diff = torch.abs(a.detach().cpu() - b.detach().cpu()).mean(dim=-1)
    return to_u8_scalar(diff, valid_mask.detach().cpu())


def image_record(
    asset_id: str,
    frame_index: int,
    dense_mae: float,
    selected_mae: float,
    paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "frame_index": frame_index,
        "dense_mae": dense_mae,
        "selected_mae": selected_mae,
        "images": {
            "Reference": Image.open(paths["reference"]).convert("RGB"),
            "Dense": Image.open(paths["dense_temporal_filtered"]).convert("RGB"),
            "Selected": Image.open(paths["selected_fast_temporal_filtered"]).convert("RGB"),
            "Dense Error": Image.open(paths["dense_abs_error"]).convert("RGB"),
            "Selected Error": Image.open(paths["selected_fast_abs_error"]).convert("RGB"),
            "Selected-Dense Diff": Image.open(paths["selected_vs_dense_abs_diff"]).convert("RGB"),
        },
    }


def make_contact_sheet(records: list[dict[str, Any]], output_path: Path) -> None:
    cell_w = 160
    cell_h = 132
    label_w = 190
    header_h = 34
    labels = ["Reference", "Dense", "Selected", "Dense Error", "Selected Error", "Selected-Dense Diff"]
    sheet = Image.new("RGB", (label_w + len(labels) * cell_w, header_h + max(len(records), 1) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 9), "Selected-fast quality A/B", fill=(0, 0, 0))
    for row, record in enumerate(records):
        y = header_h + row * cell_h
        draw.text(
            (8, y + 8),
            f"{record['asset_id']}\nframe {record['frame_index']}\n"
            f"dense={record['dense_mae']:.4f}\n"
            f"selected={record['selected_mae']:.4f}",
            fill=(0, 0, 0),
        )
        for col, label in enumerate(labels):
            x = label_w + col * cell_w
            draw.text((x + 6, y + 6), label, fill=(0, 0, 0))
            paste_thumbnail(sheet, draw, record["images"][label], (x + 6, y + 24), (146, 96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def paste_thumbnail(sheet: Image.Image, draw: ImageDraw.ImageDraw, image: Image.Image, xy: tuple[int, int], size: tuple[int, int]) -> None:
    x, y = xy
    width, height = size
    thumb = image.copy().convert("RGB")
    thumb.thumbnail(size)
    sheet.paste(thumb, (x + (width - thumb.width) // 2, y + (height - thumb.height) // 2))
    draw.rectangle((x, y, x + width, y + height), outline=(180, 180, 180))


def add_quality_rows(
    rows: list[dict[str, int | float | str]],
    *,
    asset_id: str,
    frame_index: int,
    path_name: str,
    result,
    reference,
    selected_candidate_visibility_gpu_ms: float,
    previous_temporal_filtered: torch.Tensor | None,
) -> None:
    estimators = {
        "initial": result.initial,
        "temporal": result.temporal,
        "temporal_filtered": result.temporal_filtered,
    }
    for estimator_name, buffers in estimators.items():
        finite_tensor(f"{path_name}.{estimator_name}.contribution_rgb", buffers.contribution_rgb)
        finite_tensor(f"{path_name}.{estimator_name}.composite_rgb", buffers.composite_rgb)
        contribution_metrics = compute_rgb_error_metrics(
            buffers.contribution_rgb,
            reference.diffuse_rgb,
            reference.valid_mask,
        )
        composite_metrics = compute_rgb_error_metrics(
            buffers.composite_rgb,
            reference.composite_rgb,
            reference.valid_mask,
        )
        delta = temporal_delta_metrics(
            buffers.composite_rgb,
            previous_temporal_filtered if estimator_name == "temporal_filtered" else None,
            reference.valid_mask,
        )
        timings = result.timings
        row: dict[str, int | float | str] = {
            "asset_id": asset_id,
            "frame_index": int(frame_index),
            "path": path_name,
            "estimator": estimator_name,
            "valid_pixels": int(reference.valid_mask.sum().detach().cpu()),
            "reuse_pixels": int(result.lookup.valid_mask.sum().detach().cpu()),
            "reuse_fraction": int(result.lookup.valid_mask.sum().detach().cpu()) / float(max(int(reference.valid_mask.sum().detach().cpu()), 1)),
            "selected_candidate_visibility_gpu_ms": float(selected_candidate_visibility_gpu_ms),
            "frame_gpu_ms": float(timings.frame_gpu_ms),
            "frame_wall_ms": float(timings.frame_wall_ms),
            "visibility_cache_gpu_ms": float(timings.reference_lighting_gpu_ms) if path_name == "dense_cache" else 0.0,
        }
        row.update(prefix_metrics("contribution", contribution_metrics))
        row.update(prefix_metrics("composite", composite_metrics))
        row.update(delta)
        rows.append(row)


def save_frame_images(output_dir: Path, frame_index: int, dense, selected) -> dict[str, Path]:
    frame_prefix = output_dir / f"frame_{frame_index:04d}"
    paths = {
        "reference": frame_prefix.with_name(f"{frame_prefix.name}_reference.png"),
        "dense_temporal_filtered": frame_prefix.with_name(f"{frame_prefix.name}_dense_temporal_filtered.png"),
        "selected_fast_temporal_filtered": frame_prefix.with_name(f"{frame_prefix.name}_selected_fast_temporal_filtered.png"),
        "dense_abs_error": frame_prefix.with_name(f"{frame_prefix.name}_dense_abs_error.png"),
        "selected_fast_abs_error": frame_prefix.with_name(f"{frame_prefix.name}_selected_fast_abs_error.png"),
        "selected_vs_dense_abs_diff": frame_prefix.with_name(f"{frame_prefix.name}_selected_vs_dense_abs_diff.png"),
        "reuse_mask": frame_prefix.with_name(f"{frame_prefix.name}_selected_fast_reuse_mask.png"),
        "temporal_filter_alpha": frame_prefix.with_name(f"{frame_prefix.name}_selected_fast_temporal_filter_alpha.png"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(paths["reference"], to_u8_rgb(dense.reference.composite_rgb))
    imageio.imwrite(paths["dense_temporal_filtered"], to_u8_rgb(dense.temporal_filtered.composite_rgb))
    imageio.imwrite(paths["selected_fast_temporal_filtered"], to_u8_rgb(selected.temporal_filtered.composite_rgb))
    imageio.imwrite(paths["dense_abs_error"], make_abs_error_image(dense.temporal_filtered.contribution_rgb, dense.reference.diffuse_rgb, dense.reference.valid_mask))
    imageio.imwrite(
        paths["selected_fast_abs_error"],
        make_abs_error_image(selected.temporal_filtered.contribution_rgb, dense.reference.diffuse_rgb, dense.reference.valid_mask),
    )
    imageio.imwrite(
        paths["selected_vs_dense_abs_diff"],
        make_abs_diff_image(selected.temporal_filtered.composite_rgb, dense.temporal_filtered.composite_rgb, dense.reference.valid_mask),
    )
    imageio.imwrite(paths["reuse_mask"], to_u8_mask(selected.lookup.valid_mask))
    imageio.imwrite(paths["temporal_filter_alpha"], to_u8_scalar(selected.temporal_filter_stats.alpha, dense.reference.valid_mask))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate selected-fast visibility quality against the dense-cache ReSTIR path.")
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
    parser.add_argument("--temporal-normal-threshold", type=parse_optional_float, default=0.85)
    parser.add_argument("--temporal-rgb-threshold", type=parse_optional_float, default=0.20)
    parser.add_argument("--temporal-max-motion-pixels", type=parse_optional_float, default=32.0)
    parser.add_argument("--temporal-reprojection-search-radius", type=int, default=1)
    parser.add_argument("--temporal-history-m-cap", type=parse_optional_int, default=1)
    parser.add_argument("--temporal-filter-blend-max", type=float, default=0.15)
    parser.add_argument("--temporal-filter-clamp-scale", type=float, default=0.50)
    parser.add_argument("--temporal-filter-clamp-min", type=float, default=1e-5)
    parser.add_argument("--visibility-shadow-resolution", type=int, default=128)
    parser.add_argument("--visibility-shadow-bias-scale", type=float, default=0.02)
    parser.add_argument("--visibility-shadow-alpha-threshold", type=float, default=1e-4)
    parser.add_argument("--visibility-shadow-pcf-radius", type=int, default=1)
    parser.add_argument("--ambient", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError(f"Expected positive render size, got {args.width}x{args.height}")
    if args.num_lights <= 0:
        raise ValueError(f"Expected positive --num-lights, got {args.num_lights}")
    if args.candidate_count <= 0:
        raise ValueError(f"Expected positive --candidate-count, got {args.candidate_count}")

    manifest = load_aligned_asset_manifest(args.manifest)
    asset_ids = resolve_requested_asset_ids(manifest, asset_ids=args.asset_ids, asset_set=args.asset_set)
    device = torch.device(args.device)
    settings = RestirRenderSettings(
        target_mode="visibility",
        candidate_count=args.candidate_count,
        candidate_seed_base=args.candidate_seed_base,
        initial_selection_seed_base=args.initial_selection_seed_base,
        temporal_selection_seed_base=args.temporal_selection_seed_base,
        depth_tolerance=args.depth_tolerance,
        temporal_normal_threshold=args.temporal_normal_threshold,
        temporal_rgb_threshold=args.temporal_rgb_threshold,
        temporal_max_motion_pixels=args.temporal_max_motion_pixels,
        temporal_reprojection_search_radius=args.temporal_reprojection_search_radius,
        temporal_history_m_cap=args.temporal_history_m_cap,
        temporal_filter_blend_max=args.temporal_filter_blend_max,
        temporal_filter_clamp_scale=args.temporal_filter_clamp_scale,
        temporal_filter_clamp_min=args.temporal_filter_clamp_min,
        ambient=args.ambient,
        include_mc_baseline=False,
        visibility_shadow_resolution=args.visibility_shadow_resolution,
        visibility_shadow_bias_scale=args.visibility_shadow_bias_scale,
        visibility_shadow_alpha_threshold=args.visibility_shadow_alpha_threshold,
        visibility_shadow_pcf_radius=args.visibility_shadow_pcf_radius,
    )

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
        target_world = torch.tensor(light_info["center"], dtype=torch.float32, device=device)
        light_indices = torch.arange(args.num_lights, dtype=torch.long, device=device)
        shadow_bundle, shadow_bundle_asset_gpu_ms = run_with_gpu_timing(
            device,
            lambda: make_shadow_map_bundle(
                asset.loaded.scene,
                world_lights.positions_world,
                light_indices,
                target_world,
                scene_radius=float(light_info["radius"]),
                resolution=args.visibility_shadow_resolution,
                shadow_bias_scale=args.visibility_shadow_bias_scale,
            ),
        )

        asset_output_dir = args.output_dir / asset_id / f"{args.num_lights}l_k{args.candidate_count}"
        dense_previous: RestirHistory | None = None
        selected_previous: RestirHistory | None = None
        dense_previous_filtered: torch.Tensor | None = None
        selected_previous_filtered: torch.Tensor | None = None
        contact_records: list[dict[str, Any]] = []
        for frame_index in frame_indices:
            camera = scale_camera(asset.transforms.frames[frame_index].camera, args.width, args.height)
            dense = render_restir_frame(
                asset.loaded.scene,
                camera,
                world_lights,
                frame_index=frame_index,
                settings=settings,
                previous_history=dense_previous,
                shadow_bundle=shadow_bundle,
            )
            selected, selected_extras = render_selected_visibility_frame_timed(
                asset.loaded.scene,
                camera,
                world_lights,
                frame_index=frame_index,
                settings=settings,
                previous_history=selected_previous,
                shadow_bundle=shadow_bundle,
                selected_visibility_impl="fast",
            )
            finite_tensor("reference.diffuse_rgb", dense.reference.diffuse_rgb)
            finite_tensor("reference.composite_rgb", dense.reference.composite_rgb)
            paths = save_frame_images(asset_output_dir, frame_index, dense, selected)
            add_quality_rows(
                rows,
                asset_id=asset_id,
                frame_index=frame_index,
                path_name="dense_cache",
                result=dense,
                reference=dense.reference,
                selected_candidate_visibility_gpu_ms=0.0,
                previous_temporal_filtered=dense_previous_filtered,
            )
            add_quality_rows(
                rows,
                asset_id=asset_id,
                frame_index=frame_index,
                path_name="selected_fast",
                result=selected,
                reference=dense.reference,
                selected_candidate_visibility_gpu_ms=float(selected_extras.get("selected_candidate_visibility_gpu_ms", 0.0)),
                previous_temporal_filtered=selected_previous_filtered,
            )
            dense_metrics = compute_rgb_error_metrics(dense.temporal_filtered.contribution_rgb, dense.reference.diffuse_rgb, dense.reference.valid_mask)
            selected_metrics = compute_rgb_error_metrics(selected.temporal_filtered.contribution_rgb, dense.reference.diffuse_rgb, dense.reference.valid_mask)
            contact_records.append(
                image_record(
                    asset_id,
                    frame_index,
                    float(dense_metrics["mae"]),
                    float(selected_metrics["mae"]),
                    paths,
                )
            )
            dense_previous = dense.history
            selected_previous = selected.history
            dense_previous_filtered = dense.temporal_filtered.composite_rgb.detach()
            selected_previous_filtered = selected.temporal_filtered.composite_rgb.detach()

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
                "light_info": light_info,
                "shadow_bundle_asset_gpu_ms": shadow_bundle_asset_gpu_ms,
                "outputs": {"contact_sheet": str(contact_path), "asset_output_dir": str(asset_output_dir)},
            }
        )

    csv_path = args.output_dir / "selected_fast_quality_rows.csv"
    summary_path = args.output_dir / "selected_fast_quality_summary.json"
    write_csv(csv_path, rows)
    summary = {
        "version": 1,
        "benchmark": "phase54_selected_fast_quality",
        "manifest": str(args.manifest),
        "asset_ids": asset_ids,
        "render": {"width": args.width, "height": args.height, "device": str(device)},
        "settings": {
            "target_mode": "visibility",
            "dense_path": "visibility_cache",
            "selected_path": "selected_fast",
            "selected_visibility_impl": "fast",
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
            "visibility_shadow_resolution": args.visibility_shadow_resolution,
            "visibility_shadow_bias_scale": args.visibility_shadow_bias_scale,
            "visibility_shadow_alpha_threshold": args.visibility_shadow_alpha_threshold,
            "visibility_shadow_pcf_radius": args.visibility_shadow_pcf_radius,
        },
        "quality_summary": summarize_numeric_fields(rows),
        "assets": asset_summaries,
        "outputs": {"rows_csv": str(csv_path), "summary_json": str(summary_path)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary": str(summary_path), "csv": str(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
