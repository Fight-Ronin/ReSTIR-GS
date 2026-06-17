# Active Workflow

The published workflow is registry-driven and aligned-asset focused. It keeps
the deliverable inspection surface on `main`; full validation and test runners
live on the `dev` branch.

## 1. Download Assets

```powershell
python scripts/download_aligned_asset.py --asset-set testing --dry-run
python scripts/download_aligned_splat.py --asset-set testing --dry-run
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

The manifest is `configs/aligned_assets.json`. `asset_sets.testing` contains
the 10 DXGL Polyhaven assets and resolves local data under `data/dxgl/`.

## 2. Inspect Interactively

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```

For a non-interactive viewer save:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
$env:RESTIRGS_VIEWER_EXTRA_ARGS="--save-and-exit --save-visibility"
scripts\run_interactive_viewer_windows.bat
```

This saves the visibility RIS display output without computing an all-lights
reference. Use `--save-visibility-reference` only when reference/error images
are needed for debugging.

Viewer saves write ignored local artifacts under `outputs/interactive_viewer/`.

Useful viewer keys:

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

The development validation runners on `dev` record GPU-event stage timings in
renderer CSV and summary files. Treat GPU timing fields as the performance
source of truth; `frame_wall_ms` is auxiliary orchestration context.

Diffuse/material diagnostics, standalone ablation, Voxel51, single-view PLY
workflows, and closeout experiment scripts are no longer active source surfaces.

## Display Versus Evaluation

The interactive viewer uses a lighter display path for ordinary inspection and
only enters the reference/evaluation path when explicitly asked for reference
and error output. Published `main` exposes that inspection path; the broader
evaluation workflow is retained on `dev`.
