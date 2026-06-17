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

## Published Main Workflow

Download aligned testing assets first if needed:

```powershell
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

Inspect a registered aligned asset:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```

The full validation and test runners are retained on the `dev` branch. The
published `main` branch keeps the deliverable inspection surface only.

## Viewer Outputs

```text
outputs/interactive_viewer/current_camera.json
outputs/interactive_viewer/current_rgb.png
outputs/interactive_viewer/current_alpha.png
outputs/interactive_viewer/current_normal.png
outputs/interactive_viewer/current_blinn_phong.png
outputs/interactive_viewer/current_visibility_ris.png
outputs/interactive_viewer/current_visibility_reference.png
outputs/interactive_viewer/current_visibility_error.png
outputs/interactive_viewer/interactive_viewer_save_summary.json
```

The visibility reference and error images are written only when
`--save-visibility-reference` is requested.

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
1-6            RGB / Alpha / Depth / Normal / Lambertian / Blinn-Phong diagnostics
Ctrl+S         save current camera and previews
q              quit
```

## Development Validation Readout

Development validation on the `dev` branch reports performance timing with CUDA
events in renderer summary artifacts.

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

The project is currently in wrap-up mode. If work continues, use the `dev`
branch and keep it narrowly focused on realtime visibility performance. Validate
on a small asset matrix before changing renderer defaults:

```text
1. GPU performance engineering for visibility proposal and RIS light evaluation.
2. Fused or batched kernels for visible direct-light contribution evaluation.
3. Small asset-matrix quality checks before adding or keeping new renderer knobs.
```

Do not spend more time tuning temporal alpha, spatial reuse, material objectives,
or cheap proposal mixtures on the current testing set unless a larger research
question changes the objective.
