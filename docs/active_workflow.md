# Active Workflow

This is the short path to use for current development. Historical phase scripts remain available, but new work should start here.

## 1. Environment

```powershell
conda activate restirgs
python scripts/patch_gsplat_windows.py --check
```

If the `gsplat` patch check fails, run:

```powershell
python scripts/patch_gsplat_windows.py
```

## 2. Aligned Assets

The active assets are registered in:

```text
configs/aligned_assets.json
```

The default testing set is:

```text
dxgl_apple
dxgl_cash_register
dxgl_drill
dxgl_fire_extinguisher
```

Download or dry-run the full testing set:

```powershell
python scripts/download_aligned_asset.py --asset-set testing --dry-run
python scripts/download_aligned_splat.py --asset-set testing --dry-run
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

## 3. Active Validation

Run the one-command active validation wrapper:

```powershell
scripts\run_active_validation_windows.bat
```

It runs:

```text
aligned asset smoke matrix
aligned ReSTIR renderer path
```

The runner performs the Windows Visual Studio/CUDA/`gsplat` preflight once through `scripts\_setup_windows_cuda_env.bat`.

For a faster single-asset pass:

```powershell
$env:RESTIRGS_ALIGNED_ASSET_SET="smoke"
scripts\run_active_validation_windows.bat
```

## 4. Main Outputs

The active smoke matrix writes:

```text
outputs/aligned_smoke/aligned_asset_smoke_rows.csv
outputs/aligned_smoke/aligned_asset_smoke_summary.json
outputs/aligned_smoke/<asset_id>/contact.png
```

The active ReSTIR renderer writes:

```text
outputs/aligned_restir/restir_renderer_rows.csv
outputs/aligned_restir/restir_renderer_summary.json
outputs/aligned_restir/<asset_id>/contact.png
outputs/aligned_restir/<asset_id>/final_*.png
```

## 5. Interactive Inspection

Open the viewer on a registered asset:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```

Useful viewer modes:

```text
1: RGB
2: G-buffer
3: lighting
4: single-frame ReSTIR inspection
```

## Current Boundaries

- Active dataset path: aligned DXGL assets through the manifest registry.
- Active renderer path: diffuse target, geometric proposal, world-space lights, initial RIS, previous-frame temporal reuse.
- Historical Voxel51, single-view PLY, broad ablation, and spatial diagnostic scripts are retained for reproducibility but are not the current expansion surface.
