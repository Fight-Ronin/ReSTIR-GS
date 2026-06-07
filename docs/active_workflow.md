# Active Workflow

This is the short path to use for current development. Historical phase scripts remain available, but new work should start here.

For the current stable baseline, retained compatibility path, and viewer inspection readout, see `docs/current_milestone_snapshot.md`.

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
aligned ReSTIR renderer path with visibility target
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

The active renderer summary should record:

```text
target_mode=visibility
proposal=visibility_geometric
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
5: optional visibility target inspection
```

## Retained Diffuse Compatibility

The old diffuse renderer remains available for debugging:

```powershell
$env:RESTIRGS_RESTIR_TARGET_MODE="diffuse"
$env:RESTIRGS_RESTIR_NUM_LIGHTS="128"
$env:RESTIRGS_RESTIR_WIDTH="256"
$env:RESTIRGS_RESTIR_HEIGHT="256"
$env:RESTIRGS_RESTIR_FRAME_INDICES="manifest"
$env:RESTIRGS_RESTIR_OUTPUT_DIR="outputs\aligned_restir_diffuse"
scripts\run_aligned_restir_renderer_windows.bat
```

## Optional Visibility Diagnostics

These deeper visibility diagnostics are optional because active validation already uses the visibility renderer:

```powershell
scripts\run_aligned_visibility_smoke_windows.bat
scripts\run_aligned_visibility_ris_smoke_windows.bat
scripts\run_aligned_visibility_smoke_matrix_windows.bat
scripts\run_visibility_validation_windows.bat
```

They write under:

```text
outputs/aligned_visibility/
outputs/aligned_visibility_ris/
outputs/aligned_visibility_matrix/
```

## Current Boundaries

- Active dataset path: aligned DXGL assets through the manifest registry.
- Active renderer path: visibility target, visibility-geometric proposal, world-space lights, initial RIS, compatibility-gated previous-frame temporal reuse.
- Historical Voxel51, single-view PLY, broad ablation, and spatial diagnostic scripts are retained for reproducibility but are not the current expansion surface.
