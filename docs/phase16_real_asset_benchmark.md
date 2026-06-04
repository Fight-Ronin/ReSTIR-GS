# Phase 16: Multi-Scene / Multi-View Real-Asset Benchmark

## Goal

Phase 16 turns the current real-asset pipeline into a benchmark harness. It runs camera probing, proposal ablation, and defensive spatial MIS over every scene listed in a manifest.

This phase is infrastructure, not a new estimator. The default manifest contains the current Voxel51 playroom asset as a smoke test. Research conclusions require adding more scenes.

## Manifest

Default path:

```text
configs/real_asset_benchmark.json
```

Example:

```json
{
  "scenes": [
    {
      "scene_id": "voxel51_playroom",
      "ply": "outputs/assets/voxel51_playroom_iteration_7000_point_cloud.ply",
      "max_gaussians": 200000
    }
  ],
  "defaults": {
    "view_count": 3,
    "width": 128,
    "height": 128,
    "num_lights": 128,
    "k_values": [1, 2, 4, 8, 16, 32],
    "seed_count": 8
  }
}
```

Large `.ply` files should stay under ignored local paths such as `outputs/assets/`; do not commit real assets.

## Benchmark Flow

For each scene:

```text
load PLY -> render camera probe grid -> select top views
-> build G-buffer per selected view
-> asset-scaled point lights
-> Phase 10 proposal ablation
-> Phase 15 defensive spatial MIS
```

The camera probe uses the existing 45-candidate grid:

```text
yaw: [-30, -15, 0, 15, 30]
pitch: [-10, 0, 10]
radius scale: [0.9, 1.1, 1.3]
```

Top views are selected by the existing camera score without diversity filtering.

## Run

```powershell
conda activate restirgs
scripts\run_real_asset_benchmark_windows.bat
```

For non-runner use, call `scripts/demo_15_real_asset_benchmark.py` from an environment where the conda CUDA toolkit paths and `TORCH_EXTENSIONS_DIR` are already set.

Outputs:

```text
outputs/benchmark/real_asset_benchmark_rows.csv
outputs/benchmark/real_asset_benchmark_summary.json
outputs/benchmark/<scene_id>/view_<index>/preview_rgb.png
outputs/benchmark/<scene_id>/view_<index>/camera.json
```

Rows include:

```text
scene_id
view_id
method_family
camera_score
valid_pixels
```

`method_family="proposal_ablation"` contains Phase 10 uniform/geometric MC/RIS rows. `method_family="spatial_mis"` contains Phase 15 defensive spatial MIS rows.

## Interpretation

The default one-scene benchmark is only a smoke test. Use it to verify the harness and output schema.

For research decisions, add at least two more scenes to the manifest and compare aggregate behavior across scenes/views:

```text
Does geometric proposal improve with K across scenes?
Does RIS variance behave similarly across scenes?
Does defensive spatial MIS remain helpful beyond one view?
```

Do not use this phase to tune thresholds on a single scene.
