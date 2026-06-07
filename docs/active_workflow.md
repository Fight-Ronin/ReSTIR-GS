# Active Workflow

The active workflow is registry-driven and aligned-asset focused.

## 1. Download Assets

```powershell
python scripts/download_aligned_asset.py --asset-set testing --dry-run
python scripts/download_aligned_splat.py --asset-set testing --dry-run
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

The manifest is `configs/aligned_assets.json`. `asset_sets.testing` currently contains four DXGL assets.

## 2. Run Active Validation

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

## 3. Inspect Interactively

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```

Useful viewer keys:

```text
left drag      orbit
shift+left     pan
wheel          dolly
[ / ]          previous / next aligned frame
4              ReSTIR inspection
5              visibility inspection
s              save current camera and previews
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

Diffuse, standalone ablation, Voxel51, and single-view PLY workflows are no longer active source surfaces.
