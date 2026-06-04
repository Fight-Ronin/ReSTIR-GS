# Phase 17: Public Asset Intake And Multi-Scene Benchmark Expansion

## Goal

Phase 17 expands the Phase 16 benchmark from a one-scene smoke test into a four-scene public asset smoke benchmark.

This phase does not change the renderer, proposal estimators, or defensive spatial MIS. It only adds repeatable asset intake and updates the benchmark manifest.

## Public Assets

Default source:

```text
https://huggingface.co/datasets/Voxel51/gaussian_splatting
```

The current Hugging Face file tree stores the 7000-iteration PLYs as:

```text
FO_dataset/<scene>/point_cloud/iteration_7000/point_cloud.ply
```

The dataset README also describes an older `point_cloud_folder/reconstruction_7000.ply` layout. The download script uses the current file tree because that is the directly resolvable path.

Phase 17 downloads only these four 7000-iteration assets:

```text
drjohnson
playroom
train
truck
```

Local output paths:

```text
outputs/assets/voxel51_drjohnson_iteration_7000_point_cloud.ply
outputs/assets/voxel51_playroom_iteration_7000_point_cloud.ply
outputs/assets/voxel51_train_iteration_7000_point_cloud.ply
outputs/assets/voxel51_truck_iteration_7000_point_cloud.ply
```

Large `.ply` files stay under ignored `outputs/assets/` and should not be committed.

## Download

Preview the exact URLs and local paths:

```powershell
conda activate restirgs
python scripts/download_voxel51_assets.py --dry-run
```

Download missing assets:

```powershell
python scripts/download_voxel51_assets.py
```

The script:

```text
skips existing non-empty files
streams downloads with Python stdlib only
writes through a .part file before replacing the final asset
fails if a downloaded file is missing or empty
```

If network access is unavailable, use the dry-run output to manually place each PLY at the printed path.

## Benchmark Manifest

Default path:

```text
configs/real_asset_benchmark.json
```

The manifest now lists all four Voxel51 scenes and keeps the Phase 16 benchmark defaults:

```text
view_count: 3
width: 128
height: 128
max_gaussians: 200000
num_lights: 128
k_values: [1, 2, 4, 8, 16, 32]
seed_count: 8
```

## Run

After the assets exist:

```powershell
scripts\run_real_asset_benchmark_windows.bat
```

Expected default row count:

```text
4 scenes * 3 views * (384 proposal rows + 6 spatial_mis rows) = 4680 rows
```

Outputs:

```text
outputs/benchmark/real_asset_benchmark_rows.csv
outputs/benchmark/real_asset_benchmark_summary.json
outputs/benchmark/<scene_id>/view_<index>/preview_rgb.png
outputs/benchmark/<scene_id>/view_<index>/camera.json
```

## Interpretation

This phase gives the project a real multi-scene smoke benchmark. It is still not the final research readout.

Use Phase 17 results to check:

```text
whether geometric proposal improves with larger K across scenes
whether defensive spatial MIS remains useful beyond one scene/view
whether any scene has camera, G-buffer, or loader failure modes
```

Do not tune estimator behavior to a single Voxel51 view after this phase. If one scene fails, fix the intake/camera/rendering issue first; if the full benchmark runs, use the aggregate CSV/JSON for the next research decision.
