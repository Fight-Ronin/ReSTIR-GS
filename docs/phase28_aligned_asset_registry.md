# Phase 28: Aligned Asset Registry

Phase 28 moves the active aligned workflow from Apple-specific script defaults toward a manifest-driven aligned Gaussian asset path.

The goal is not a new benchmark result. The goal is an interface boundary:

```text
dataset adapter: cameras, RGB/depth/masks/normals, normalization
Gaussian loader: dataset-agnostic compatible 3DGS asset loading
```

## Manifest

The default manifest is:

```text
configs/aligned_assets.json
```

It now separates asset facts from run selection. Asset entries record dataset and splat facts; `asset_sets` records named groups such as `smoke` and `testing`.

```text
asset_id = dxgl_apple
dataset_type = dxgl
dataset_root = data/dxgl/apple
splat_path = data/dxgl/apple_splat/apple.ply
```

Adding a new aligned object should start by adding another manifest entry with explicit dataset and splat URLs, then optionally adding it to an asset set. The code should not grow another object-specific downloader script.

## Generic Gaussian Loading

Aligned dataset loading uses the dataset adapter for camera and modality semantics, but splats are loaded through:

```text
load_gaussian_asset(path, schema="auto")
```

V1 supports compatible GraphDECO/Nerfstudio-style 3DGS PLY only. Dataset-specific normalization stays outside the Gaussian loader.

## Generic Intake Commands

Dry-run the manifest-registered dataset and splat downloads:

```powershell
python scripts/download_aligned_asset.py --asset-set testing --dry-run
python scripts/download_aligned_splat.py --asset-set testing --dry-run
```

Download when needed:

```powershell
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

Legacy Apple-specific commands remain available, but the generic manifest commands are the active workflow.

## Smoke Matrix

Run the small aligned smoke matrix:

```powershell
python scripts/demo_24_aligned_asset_smoke_matrix.py --asset-set testing --device cuda
```

On Windows, run this from an x64 Visual Studio developer shell or after applying the same `vcvars64.bat`, `TORCH_CUDA_ARCH_LIST`, and `TORCH_EXTENSIONS_DIR` setup used by the existing `gsplat` runners.

Outputs:

```text
outputs/aligned_smoke/aligned_asset_smoke_rows.csv
outputs/aligned_smoke/aligned_asset_smoke_summary.json
outputs/aligned_smoke/<asset_id>/contact.png
```

Rows are normalized by `asset_id`, `stage`, and `metric_name`. This is intentionally not a full research benchmark; it checks that render, G-buffer, lighting, tiny sampling, and tiny temporal world-light smoke can run from a manifest entry.

## Limits

This phase does not add new datasets, new estimators, visibility, shadows, temporal optimization, or a new splat format. More assets should be added only after this one-asset manifest path is stable.
