from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.lighting.asset_lights import make_asset_scaled_world_lights, world_lights_to_camera_lights
from restir_gs.lighting.visibility import make_shadow_map_bundle
from restir_gs.render.aligned_asset_registry import (
    DEFAULT_MANIFEST_PATH,
    get_aligned_asset_spec,
    load_aligned_asset_manifest,
    load_registered_aligned_asset,
    resolve_aligned_asset_paths,
    resolve_requested_asset_ids,
)
from restir_gs.render.dxgl_asset import scale_camera
from restir_gs.render.gbuffer import make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.restir.renderer import (
    RESTIR_TIMING_FIELDS,
    RestirDisplayFrameResult,
    RestirHistory,
    RestirRenderSettings,
    _FrameStageTimer,
    evaluate_restir_display_frame_from_gbuffer,
)


DEFAULT_OUTPUT_DIR = Path("outputs/realtime_display_fps")


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


def parse_optional_float(text: str) -> float | None:
    if text.strip().lower() == "none":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a float or 'none', got {text!r}") from exc


def parse_optional_int(text: str) -> int | None:
    if text.strip().lower() == "none":
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an int or 'none', got {text!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive int or 'none', got {text!r}")
    return value


def run_with_gpu_timing(device: torch.device, fn: Callable[[], Any]) -> tuple[Any, float]:
    if device.type != "cuda":
        return fn(), 0.0
    with torch.cuda.device(device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
    torch.cuda.synchronize(device)
    return result, float(start.elapsed_time(end))


def render_display_frame_timed(
    scene,
    camera,
    world_lights,
    frame_index: int,
    settings: RestirRenderSettings,
    previous_history: RestirHistory | None,
    shadow_bundle,
) -> RestirDisplayFrameResult:
    wall_start = time.perf_counter()
    timer = _FrameStageTimer(scene.means.device)
    with torch.no_grad():
        timer.mark("start")
        render_buffers = render_rgbd(scene, camera)
        timer.mark("after_render_rgbd")
        gbuffer = make_pseudo_gbuffer(render_buffers, camera)
        timer.mark("after_gbuffer")
        lights = world_lights_to_camera_lights(world_lights, camera)
        timer.mark("after_world_lights_to_camera")
        result = evaluate_restir_display_frame_from_gbuffer(
            gbuffer,
            camera,
            lights,
            frame_index=frame_index,
            settings=settings,
            previous_history=previous_history,
            shadow_bundle=shadow_bundle,
            _timer=timer,
        )
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return replace(result, timings=replace(result.timings, frame_wall_ms=wall_ms))


def make_frame_row(
    asset_id: str,
    dataset_type: str,
    repeat_index: int,
    frame_index: int,
    width: int,
    height: int,
    num_lights: int,
    settings: RestirRenderSettings,
    shadow_bundle_asset_gpu_ms: float,
    result: RestirDisplayFrameResult,
) -> dict[str, int | float | str]:
    timing_fields = result.timings.as_row_fields(shadow_bundle_asset_gpu_ms=shadow_bundle_asset_gpu_ms)
    valid_pixels = int(result.initial.valid_mask.sum().detach().cpu())
    reuse_pixels = int(result.lookup.valid_mask.sum().detach().cpu())
    frame_gpu_ms = float(timing_fields["frame_gpu_ms"])
    frame_wall_ms = float(timing_fields["frame_wall_ms"])
    row: dict[str, int | float | str] = {
        "asset_id": asset_id,
        "dataset_type": dataset_type,
        "repeat_index": int(repeat_index),
        "frame_index": int(frame_index),
        "width": int(width),
        "height": int(height),
        "target_mode": settings.target_mode,
        "num_lights": int(num_lights),
        "candidate_count": int(settings.candidate_count),
        "valid_pixels": valid_pixels,
        "reuse_pixels": reuse_pixels,
        "reuse_fraction": reuse_pixels / float(max(valid_pixels, 1)),
        "estimated_gpu_fps": 1000.0 / frame_gpu_ms if frame_gpu_ms > 0.0 else 0.0,
        "estimated_wall_fps": 1000.0 / frame_wall_ms if frame_wall_ms > 0.0 else 0.0,
        "visibility_cache_gpu_ms": float(timing_fields["reference_lighting_gpu_ms"])
        if settings.target_mode == "visibility"
        else 0.0,
    }
    row.update(timing_fields)
    return row


def write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    if not rows:
        raise RuntimeError("Cannot write an empty real-time display benchmark CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_numeric_fields(
    rows: list[dict[str, int | float | str]],
    fields: list[str],
) -> dict[str, dict[str, float | int]]:
    return {field: summarize_values([float(row[field]) for row in rows if field in row]) for field in fields}


def summarize_values(values: list[float]) -> dict[str, float | int]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(finite),
        "mean": sum(finite) / float(len(finite)),
        "median": percentile(finite, 0.50),
        "p90": percentile(finite, 0.90),
        "min": finite[0],
        "max": finite[-1],
    }


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * float(len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    t = pos - float(lo)
    return sorted_values[lo] * (1.0 - t) + sorted_values[hi] * t


def assert_display_frame_finite(result: RestirDisplayFrameResult, asset_id: str, frame_index: int) -> None:
    tensors = {
        "initial.composite_rgb": result.initial.composite_rgb,
        "temporal.composite_rgb": result.temporal.composite_rgb,
        "temporal_filtered.composite_rgb": result.temporal_filtered.composite_rgb,
        "temporal_reservoir.W": result.temporal_reservoir.W,
    }
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all().detach().cpu()):
            raise RuntimeError(f"{asset_id} frame {frame_index} produced non-finite values in {name}.")


def finite_rows(rows: list[dict[str, int | float | str]]) -> bool:
    for row in rows:
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the real-time display ReSTIR renderer path.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--asset-ids", type=parse_asset_ids, default=None)
    parser.add_argument("--asset-set", default="smoke")
    parser.add_argument("--frame-indices", type=parse_int_list, default=None)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-gaussians", type=int, default=None)
    parser.add_argument("--num-lights", type=int, default=128)
    parser.add_argument("--target-mode", choices=("diffuse", "visibility"), default="visibility")
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
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--repeat-iters", type=int, default=3)
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
    if args.warmup_iters < 0 or args.repeat_iters <= 0:
        raise ValueError("Expected --warmup-iters >= 0 and --repeat-iters > 0")

    manifest = load_aligned_asset_manifest(args.manifest)
    asset_ids = resolve_requested_asset_ids(manifest, asset_ids=args.asset_ids, asset_set=args.asset_set)
    device = torch.device(args.device)
    settings = RestirRenderSettings(
        target_mode=args.target_mode,
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
        shadow_bundle = None
        shadow_bundle_asset_gpu_ms = 0.0
        if args.target_mode == "visibility":
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

        for _ in range(args.warmup_iters):
            previous: RestirHistory | None = None
            for frame_index in frame_indices:
                camera = scale_camera(asset.transforms.frames[frame_index].camera, args.width, args.height)
                result = render_display_frame_timed(
                    asset.loaded.scene,
                    camera,
                    world_lights,
                    frame_index=frame_index,
                    settings=settings,
                    previous_history=previous,
                    shadow_bundle=shadow_bundle,
                )
                previous = result.history

        asset_rows: list[dict[str, int | float | str]] = []
        for repeat_index in range(args.repeat_iters):
            previous = None
            for frame_index in frame_indices:
                camera = scale_camera(asset.transforms.frames[frame_index].camera, args.width, args.height)
                result = render_display_frame_timed(
                    asset.loaded.scene,
                    camera,
                    world_lights,
                    frame_index=frame_index,
                    settings=settings,
                    previous_history=previous,
                    shadow_bundle=shadow_bundle,
                )
                assert_display_frame_finite(result, asset_id, frame_index)
                row = make_frame_row(
                    asset_id,
                    spec.dataset_type,
                    repeat_index,
                    frame_index,
                    args.width,
                    args.height,
                    args.num_lights,
                    settings,
                    shadow_bundle_asset_gpu_ms,
                    result,
                )
                rows.append(row)
                asset_rows.append(row)
                previous = result.history

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
                "timing_summary": summarize_numeric_fields(asset_rows, summary_fields()),
            }
        )

    if not finite_rows(rows):
        raise RuntimeError("Benchmark produced non-finite numeric row values.")

    csv_path = args.output_dir / "restir_display_fps_rows.csv"
    summary_path = args.output_dir / "restir_display_fps_summary.json"
    write_csv(csv_path, rows)
    summary = {
        "version": 1,
        "benchmark": "phase51_realtime_display_fps",
        "note": (
            "This measures the display path without an all-lights reference. "
            "In visibility mode, reference_lighting_gpu_ms is the shadow visibility cache stage."
        ),
        "manifest": str(args.manifest),
        "asset_ids": asset_ids,
        "render": {"width": args.width, "height": args.height, "device": str(device)},
        "settings": {
            "target_mode": args.target_mode,
            "num_lights": args.num_lights,
            "light_seed": args.light_seed,
            "light_space": "world",
            "light_policy": "asset_scaled_spherical_shell",
            "light_bbox_percentile": args.light_bbox_percentile,
            "light_radius_scale": args.light_radius_scale,
            "candidate_count": args.candidate_count,
            "warmup_iters": args.warmup_iters,
            "repeat_iters": args.repeat_iters,
            "visibility_shadow_resolution": args.visibility_shadow_resolution,
            "visibility_shadow_bias_scale": args.visibility_shadow_bias_scale,
            "visibility_shadow_alpha_threshold": args.visibility_shadow_alpha_threshold,
            "visibility_shadow_pcf_radius": args.visibility_shadow_pcf_radius,
        },
        "timing_summary": summarize_numeric_fields(rows, summary_fields()),
        "assets": asset_summaries,
        "outputs": {"rows_csv": str(csv_path), "summary_json": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary": str(summary_path), "csv": str(csv_path)}, indent=2))
    return 0


def summary_fields() -> list[str]:
    return [
        "estimated_gpu_fps",
        "estimated_wall_fps",
        "visibility_cache_gpu_ms",
        *list(RESTIR_TIMING_FIELDS),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
