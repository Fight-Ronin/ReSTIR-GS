# Phase 30: Aligned Testing Asset Set Expansion

Phase 30 expands the registry-driven aligned workflow from one smoke asset to a small testing set. It does not change rendering, lighting, sampling, temporal reuse, or Gaussian loading logic.

## Asset Facts And Asset Sets

The manifest now separates registered asset facts from run selection:

```text
assets      = dataset URL, splat URL, local paths, frames, normalization
asset_sets  = named groups used by download and smoke commands
```

The default sets are:

```text
smoke   = dxgl_apple
testing = all 10 DXGL Polyhaven assets
```

This keeps future expansion clean: adding an object starts with a manifest entry, then optionally adding it to a set. The Gaussian splat still loads through `load_gaussian_asset(...)`; no object-specific loader is introduced.

## Testing Assets

The current testing set uses DXGL Polyhaven 10 assets. DXGL lists these datasets as CC0, with 196 views, RGB/depth/normals/masks/camera poses, and pretrained splats: https://dx.gl/datasets/polyhaven-10

Chosen coverage:

```text
dxgl_apple              organic baseline
dxgl_cash_register      boxy electronics / hard edges
dxgl_drill              elongated tool / local geometry complexity
dxgl_fire_extinguisher  metallic cylinder / stronger highlight pressure
dxgl_led_lightbulb      glass / transmissive-looking material pressure
dxgl_measuring_tape     small tool with fine markings
dxgl_modern_arm_chair   larger furniture silhouette
dxgl_multi_cleaner_5l   product container with labels
dxgl_potted_plant       organic structure and thin leaves
dxgl_wet_floor_sign     bright plastic planar object
```

Each asset uses:

```text
default_frames  = [0, 49, 98, 147]
temporal_window = [45, 46, 47, 48, 49, 50, 51, 52, 53]
gaussian_schema = auto
max_gaussians   = 0
normalization   = inferred_from_points3d, raw_y_to_z_up, bbox_percentile 0.98
```

## Commands

Dry-run the full testing set:

```powershell
python scripts/download_aligned_asset.py --asset-set testing --dry-run
python scripts/download_aligned_splat.py --asset-set testing --dry-run
```

Download the full testing set:

```powershell
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

Run the Windows CUDA smoke matrix:

```powershell
scripts\run_aligned_asset_smoke_matrix_windows.bat
```

For targeted debugging, override the set:

```powershell
$env:RESTIRGS_ALIGNED_ASSET_IDS="dxgl_apple,dxgl_drill"
scripts\run_aligned_asset_smoke_matrix_windows.bat
```

## Expected Output

The smoke matrix writes:

```text
outputs/aligned_smoke/aligned_asset_smoke_rows.csv
outputs/aligned_smoke/aligned_asset_smoke_summary.json
outputs/aligned_smoke/<asset_id>/contact.png
```

When all 10 testing assets are present, the expected row count is:

```text
10 assets * 19 rows per asset = 190 rows
```

`all_numeric_finite=true` is required for the testing set to pass.
