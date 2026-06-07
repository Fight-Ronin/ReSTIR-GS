# ReSTIR-GS

Windows-native prototype for visibility-aware ReSTIR over aligned 3D Gaussian Splatting assets.

The current codebase is organized around one active baseline: registered aligned assets are rendered with `gsplat`, converted into pseudo G-buffers, evaluated with visibility-aware direct lighting, sampled with a visibility-geometric proposal, and stabilized with a conservative previous-frame temporal path. Older synthetic demos, Voxel51 intake, single-view PLY experiments, standalone ablations, and spatial MIS diagnostics are no longer active source surfaces.

## Current Active Path

```text
configs/aligned_assets.json
-> aligned asset registry
-> DXGL dataset adapter
-> compatible 3DGS PLY loader
-> gsplat RGB+expected-depth render
-> pseudo G-buffer
-> scene-stable world-space lights
-> PCF shadow-map visibility target
-> visibility-geometric light proposal
-> initial RIS
-> previous-frame temporal reprojection
-> confidence-clamped temporal filtered output
-> CSV/JSON metrics, contact sheets, and GPU timing summaries
```

The preferred displayed output is `temporal_filtered_ris`. `initial_ris` is the current-frame estimate, and `temporal_ris` remains a reservoir-combine debug row. Diffuse mode still exists for compatibility tests, but the active renderer target is visibility-aware direct lighting.

Active defaults:

```text
target_mode = visibility
proposal = visibility_geometric
visibility_shadow_pcf_radius = 1
temporal_reprojection_search_radius = 1
temporal_history_m_cap = 1
```

## Repository Layout

```text
configs/       aligned asset manifest
restir_gs/     renderer, lighting, ReSTIR, metrics, and eval helpers
interactive/   interactive viewer package
gs_gen/        standalone local Gaussian asset generation helper
scripts/       Windows runners and active demos
docs/          architecture notes, handoff docs, and phase records
tests/         unit and smoke coverage for the active path
outputs/       ignored assets, renders, metrics, and snapshots
```

Important implementation files:

```text
restir_gs/render/aligned_asset_registry.py
restir_gs/render/dxgl_asset.py
restir_gs/render/ply_loader.py
restir_gs/render/gbuffer.py
restir_gs/lighting/visibility.py
restir_gs/restir/proposal.py
restir_gs/restir/visibility.py
restir_gs/restir/temporal.py
restir_gs/restir/renderer.py
interactive/launcher.py
interactive/rendering.py
interactive/session.py
interactive/viewer.py
interactive/web_server.py
```

## Environment

Use the existing `restirgs` conda environment. On Windows, prefer the `.bat` runners for CUDA and `gsplat` work because they set the Visual Studio, CUDA, torch extension cache, and `gsplat` compatibility patch environment.

Useful checks:

```powershell
conda activate restirgs
python -m pytest -q
python -m compileall restir_gs scripts interactive gs_gen
python -m pip check
```

## Download Aligned Assets

The active manifest is `configs/aligned_assets.json`. The default testing set is:

```text
dxgl_apple
dxgl_cash_register
dxgl_drill
dxgl_fire_extinguisher
```

Dry-run first:

```powershell
python scripts/download_aligned_asset.py --asset-set testing --dry-run
python scripts/download_aligned_splat.py --asset-set testing --dry-run
```

Then download:

```powershell
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

Downloaded assets and generated renderer outputs stay under ignored `outputs/`.

## Run The Active Baseline

Preferred closeout command:

```powershell
scripts\run_active_baseline_demo_windows.bat
```

This runs active validation and then builds the demo/performance snapshot.

Run active validation only:

```powershell
scripts\run_active_validation_windows.bat
```

Build the active demo snapshot only:

```powershell
scripts\run_active_demo_snapshot_windows.bat
```

Expected high-level outputs:

```text
outputs/aligned_smoke/aligned_asset_smoke_rows.csv
outputs/aligned_smoke/aligned_asset_smoke_summary.json
outputs/aligned_restir/restir_renderer_rows.csv
outputs/aligned_restir/restir_renderer_summary.json
outputs/aligned_restir/<asset_id>/contact.png
outputs/active_demo/active_renderer_snapshot_contact.png
outputs/active_demo/active_renderer_snapshot_summary.json
```

The active summary should report finite numeric metrics, `target_mode=visibility`, `proposal=visibility_geometric`, and GPU-event timing summaries. Treat GPU timing fields as the performance source of truth; `frame_wall_ms` is auxiliary wall-clock context.

## Individual Commands

Aligned asset smoke matrix:

```powershell
scripts\run_aligned_asset_smoke_matrix_windows.bat
```

Aligned ReSTIR renderer:

```powershell
scripts\run_aligned_restir_renderer_windows.bat
```

Interactive viewer:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```

Browser WebUI prototype:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_web_viewer_windows.bat
```

Then open:

```text
http://127.0.0.1:8765
```

The WebUI runner defaults to `1024x1024`. Override `RESTIRGS_VIEWER_WIDTH` and `RESTIRGS_VIEWER_HEIGHT` to trade quality for responsiveness.

Viewer timing smoke:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
$env:RESTIRGS_VIEWER_EXTRA_ARGS="--save-and-exit --save-visibility"
scripts\run_interactive_viewer_windows.bat
```

`--save-visibility` saves the visibility RIS display output without computing an all-lights reference. For explicit debug/reference output, use:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
$env:RESTIRGS_VIEWER_EXTRA_ARGS="--save-and-exit --save-visibility-reference"
scripts\run_interactive_viewer_windows.bat
```

The viewer computes buffers lazily by view layer, records GPU-stage timings, and supports free camera movement. Its ordinary display path is separate from the evaluation/reference path used by CSV metrics and error maps. Useful keys:

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

The viewer can also inspect a generic compatible 3DGS PLY:

```powershell
python -m interactive.launcher --ply path\to\asset.ply --device cuda
```

The browser prototype uses the same backend session and renderer:

```powershell
python -m interactive.web_server --ply path\to\asset.ply --device cuda
```

## Local Asset Generation Helper

`gs_gen/` is a standalone helper for planning, validating, and staging local Gaussian Splatting assets. It does not modify `configs/aligned_assets.json` and is not part of the active renderer pipeline yet.

Local videos and third-party reconstruction tools are intentionally not committed. After cloning, put local capture videos under `gs_gen/asset/` or another ignored data directory, and extract the Windows COLMAP bundle expected by the helper scripts to:

```text
gs_gen/tools/colmap-4.0.4-nocuda/
```

The compatibility wrapper expects:

```text
gs_gen/tools/colmap-4.0.4-nocuda/bin/colmap.exe
```

Example flow:

```powershell
python -m gs_gen probe-source --images data\room_capture\my_room\images
python -m gs_gen plan --asset-id my_room --images data\room_capture\my_room\images
python -m gs_gen validate --dataset-root outputs\gsgen\my_room\processed --splat outputs\gsgen\my_room\export\splat.ply
python -m gs_gen stage --asset-id my_room --dataset-root outputs\gsgen\my_room\processed --splat outputs\gsgen\my_room\export\splat.ply --copy-images
```

For video sources:

```powershell
python -m gs_gen extract-frames --video data\room_capture\my_room\walkthrough.mp4 --output-dir outputs\gsgen\my_room\source_images --target-fps 5
python -m gs_gen plan --asset-id my_room --images outputs\gsgen\my_room\source_images
```

`gs_gen` prints Nerfstudio/Splatfacto commands but does not install or run those external training tools.

## Architecture Boundaries

This prototype currently assumes:

- aligned assets from the manifest are the active source of truth;
- visibility is a shadow-map proxy, not exact physical visibility;
- pseudo normals are reconstructed from expected-depth positions;
- temporal reuse is previous-frame only;
- `temporal_filtered_ris` is a conservative stabilization layer, not proof that reservoir reuse wins for every frame;
- the interactive viewer is an inspection tool, not a production real-time renderer;
- viewer display outputs do not imply all-lights reference/error evaluation unless explicitly requested.

The next meaningful work is likely GPU performance engineering for visibility/proposal/RIS evaluation, better visibility semantics, broader aligned asset coverage, or a production-quality viewer path. More small parameter tuning on the current four-asset testing set is not the main priority.

## Documentation

Start here:

```text
docs/active_workflow.md
docs/active_baseline_handoff.md
docs/current_architecture.md
docs/current_milestone_snapshot.md
docs/phase37_visibility_active_renderer.md
docs/phase44_temporal_reprojection_repair.md
docs/phase45_active_renderer_profiling.md
docs/phase46_active_demo_snapshot.md
docs/phase48_interactive_viewer_timing.md
scripts/README.md
gs_gen/README.md
```
