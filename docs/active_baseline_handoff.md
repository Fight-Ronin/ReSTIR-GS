# Active Baseline Handoff

## Baseline Definition

The current active ReSTIR-GS baseline is:

```text
aligned asset registry
-> generic compatible 3DGS PLY loader
-> aligned camera render with gsplat
-> pseudo G-buffer
-> scene-stable world-space lights
-> visibility-aware direct-lighting target
-> visibility-geometric proposal
-> initial RIS
-> previous-frame temporal reservoir debug path
-> confidence-clamped temporal filtered output
```

The preferred displayed output is `temporal_filtered_ris`.

The active renderer policy is:

```text
target_mode = visibility
proposal = visibility_geometric
lights = scene-stable world-space lights
temporal_history_m_cap = 1
temporal_reprojection_search_radius = 1
visibility_shadow_pcf_radius = 1
```

`initial_ris` is the fresh current-frame estimate. `temporal_ris` is retained as a reservoir-combine debug path. Diffuse mode is retained for compatibility/debugging, but it is not the preferred active output.

## Run The Baseline

Download aligned testing assets first if needed:

```powershell
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

Run the full active baseline demo:

```powershell
scripts\run_active_baseline_demo_windows.bat
```

This runs:

```text
scripts/run_active_validation_windows.bat
scripts/run_active_demo_snapshot_windows.bat
```

## Expected Outputs

```text
outputs/aligned_smoke/aligned_asset_smoke_rows.csv
outputs/aligned_smoke/aligned_asset_smoke_summary.json
outputs/aligned_restir/restir_renderer_rows.csv
outputs/aligned_restir/restir_renderer_summary.json
outputs/aligned_restir/<asset_id>/contact.png
outputs/active_demo/active_renderer_snapshot_contact.png
outputs/active_demo/active_renderer_snapshot_summary.json
```

Expected active validation:

```text
aligned smoke matrix rows = 76
active renderer rows = 72
target_mode = visibility
proposal = visibility_geometric
all_numeric_finite = true
timing_summary exists
asset_timing exists
```

## Interactive Inspection

Use the viewer for manual inspection:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```

Viewer display and evaluation are deliberately separate:

```powershell
$env:RESTIRGS_VIEWER_EXTRA_ARGS="--save-and-exit --save-visibility"
scripts\run_interactive_viewer_windows.bat
```

This saves the visibility RIS display output. Add `--save-visibility-reference` only when all-lights reference and error images are needed.

Useful keys:

```text
W/S            move forward / backward
A/D            move left / right
Shift/Ctrl     move up / down
left drag      orbit yaw/pitch
shift+left     pan camera target
wheel          dolly focus distance
[ / ]          previous / next aligned frame
1-6            RGB / Alpha / Depth / Normal / Lambertian / Blinn-Phong
Ctrl+S         save current camera and previews
q              quit
```

## Current Performance Readout

Performance timing is reported with CUDA events in:

```text
outputs/aligned_restir/restir_renderer_summary.json
outputs/active_demo/active_renderer_snapshot_summary.json
```

`frame_wall_ms` is auxiliary wall-clock context. GPU timing fields are the primary performance numbers.

Current profiling shows the main GPU cost is in lighting/proposal/RIS evaluation:

```text
initial_ris_gpu_ms
proposal_gpu_ms
reference_lighting_gpu_ms
temporal_ris_gpu_ms
```

The main cost is not `gsplat` RGB+D rendering, pseudo G-buffer generation, reprojection, or the temporal image-space filter.

## Known Limitations

- Visibility is a shadow-map proxy, not physically exact visibility.
- Pseudo normals come from screen-space expected-depth positions.
- The active renderer uses a small DXGL testing asset set, not a broad production benchmark.
- `temporal_filtered_ris` is a conservative real-time stabilization layer, not a proof that temporal reservoir reuse beats initial RIS in every case.
- The interactive viewer is an inspection tool, not a production real-time renderer.
- The viewer display path does not compute all-lights reference/error outputs unless explicitly requested.
- Current performance is still dominated by unfused lighting/proposal/RIS tensor evaluation.

## Recommended Future Work

The next meaningful work should be a larger step, not more small parameter tuning:

```text
1. GPU performance engineering for visibility proposal and RIS light evaluation.
2. Fused or batched kernels for direct-light contribution evaluation.
3. Better visibility/shadow semantics.
4. Larger aligned asset coverage.
5. Production-quality interactive renderer path.
```

Do not spend more time tuning temporal alpha, spatial reuse, or small ablations on the current 10-asset testing set unless a larger research question changes the objective.
