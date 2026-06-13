# Active Workflow

The active workflow is registry-driven and aligned-asset focused.

## 1. Download Assets

```powershell
python scripts/download_aligned_asset.py --asset-set testing --dry-run
python scripts/download_aligned_splat.py --asset-set testing --dry-run
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

The manifest is `configs/aligned_assets.json`. `asset_sets.testing` contains the 10 DXGL Polyhaven assets and resolves local data under `data/dxgl/`.

## 2. Run Active Baseline Demo

```powershell
scripts\run_active_baseline_demo_windows.bat
```

This is the preferred closeout command. It runs active validation and then builds the active demo/performance snapshot.

## 3. Run Active Validation Only

```powershell
scripts\run_active_validation_windows.bat
```

This runs:

```text
scripts/demo_24_aligned_asset_smoke_matrix.py
scripts/demo_26_aligned_restir_renderer.py
```

Expected outputs:

```text
outputs/aligned_smoke/
outputs/aligned_restir/
```

## 4. Build Demo Snapshot Only

```powershell
scripts\run_active_demo_snapshot_windows.bat
```

This consumes `outputs/aligned_restir/` and writes:

```text
outputs/active_demo/active_renderer_snapshot_contact.png
outputs/active_demo/active_renderer_snapshot_summary.json
```

## 5. Inspect Interactively

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```

For a non-interactive viewer save smoke:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
$env:RESTIRGS_VIEWER_EXTRA_ARGS="--save-and-exit --save-visibility"
scripts\run_interactive_viewer_windows.bat
```

This saves the visibility RIS display output without computing an all-lights reference. Use `--save-visibility-reference` only when reference/error images are needed for debugging.

Useful viewer keys:

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

## Current Active Renderer

```text
target_mode = visibility
proposal = visibility_geometric
lights = scene-stable world-space lights
shadow filtering = 3x3 PCF
temporal reuse = previous frame only
temporal correspondence = depth + unoriented normal + RGB + motion gate
temporal repair = local 3x3 reprojection candidate search
preferred output = temporal_filtered_ris
```

The active renderer records GPU-event stage timings in `restir_renderer_rows.csv` and summarizes them in `restir_renderer_summary.json`. Treat GPU timing fields as the performance source of truth; `frame_wall_ms` is auxiliary orchestration context.

Diffuse, standalone ablation, Voxel51, and single-view PLY workflows are no longer active source surfaces.

## Display Versus Evaluation

The active renderer/evaluator path writes CSV rows, all-lights references, and error images. The interactive viewer uses a lighter display path for ordinary inspection and only enters the reference/evaluation path when explicitly asked for reference/error output.
