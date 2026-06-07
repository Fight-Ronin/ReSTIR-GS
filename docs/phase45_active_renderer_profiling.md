# Phase 45: Active Renderer GPU Profiling

## Purpose

Phase 45 adds profiling to the active aligned renderer without changing renderer behavior.

The active baseline remains:

```text
target_mode = visibility
proposal = visibility_geometric
preferred output = temporal_filtered_ris
lights = scene-stable world lights
temporal = local-repaired reprojection + confidence-clamped filter
```

## Timing Policy

GPU stage timings use `torch.cuda.Event(enable_timing=True)`. These are the primary performance numbers.

`frame_wall_ms` uses `time.perf_counter()` and is only auxiliary context. It can include Python orchestration, synchronization, file-independent setup, and other CPU overhead.

CPU runs do not create CUDA events. They report finite `0.0` GPU timings so unit tests remain deterministic.

## Renderer Row Fields

Every active renderer CSV row includes:

```text
render_rgbd_gpu_ms
gbuffer_gpu_ms
world_lights_to_camera_gpu_ms
reference_lighting_gpu_ms
proposal_gpu_ms
initial_ris_gpu_ms
temporal_lookup_gpu_ms
temporal_ris_gpu_ms
temporal_filter_gpu_ms
frame_gpu_ms
frame_wall_ms
shadow_bundle_asset_gpu_ms
```

`shadow_bundle_asset_gpu_ms` is measured once per asset because the active visibility renderer builds one shadow bundle per asset and reuses it across the frame window. The value is repeated on per-frame rows for CSV convenience.

## Summary Output

`outputs/aligned_restir/restir_renderer_summary.json` now includes:

```text
timing_summary
asset_timing
```

Each timing entry reports `mean`, `max`, and `count`.

## Run

```powershell
scripts\run_active_validation_windows.bat
```

Expected active renderer invariants remain:

```text
row_count = 72
target_mode = visibility
proposal = visibility_geometric
all_numeric_finite = true
```

On CUDA, meaningful stages such as `render_rgbd_gpu_ms`, `reference_lighting_gpu_ms`, `proposal_gpu_ms`, `initial_ris_gpu_ms`, or `frame_gpu_ms` should be nonzero.
