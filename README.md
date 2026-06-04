# ReSTIR-GS

Minimal Windows-native prototype setup for ReSTIR-GS. Phase 1 uses deterministic synthetic Gaussians to verify `gsplat` RGB/expected-depth/alpha rendering before adding `.ply` loading, normal estimation, lighting, or ReSTIR.

## Environment

Create the conda environment and install Python dependencies:

```powershell
conda env create -f environment.yml
conda activate restirgs
pip install -r requirements.txt
```

The current verified stack is Python 3.10, CUDA toolkit 12.4, PyTorch `2.5.1+cu124`, and `gsplat==1.5.3`.

## Windows gsplat Patch

`gsplat==1.5.3` passes the GCC-only flag `-Wno-attributes` to MSVC during JIT extension builds. Apply the local compatibility patch after installing dependencies:

```powershell
python scripts/patch_gsplat_windows.py
python scripts/patch_gsplat_windows.py --check
```

The patch is idempotent and only changes the installed package inside the active Python environment.

## Smoke Demo

Run the repo-native synthetic RGB+ED render from Windows:

```powershell
scripts\run_smoke_windows.bat
```

It writes:

```text
outputs/synthetic_rgb.png
outputs/synthetic_depth.png
outputs/synthetic_alpha.png
```

## Pseudo G-buffer Demo

After the smoke demo passes, run the synthetic pseudo G-buffer demo:

```powershell
scripts\run_gbuffer_windows.bat
```

It writes:

```text
outputs/gbuffer_rgb.png
outputs/gbuffer_depth.png
outputs/gbuffer_alpha.png
outputs/gbuffer_position.png
outputs/gbuffer_normal.png
```

See `docs/phase2_pseudo_gbuffer.md` for the expected-depth, unprojection, and normal-estimation details. This is the gate before adding deferred lighting or real Gaussian `.ply` assets.

## Proposal Ablation

Run the per-pixel geometric proposal ablation:

```powershell
scripts\run_proposal_ablation_windows.bat
```

It writes:

```text
outputs/proposal_ablation.csv
outputs/proposal_ablation_summary.json
```

See `docs/phase6_per_pixel_proposal.md` for the proposal formula, estimator equations, and limitations.

## PLY Asset Baseline

For a new real 3DGS `.ply`, first probe a stable camera:

```powershell
$env:RESTIRGS_PLY="C:\path\to\point_cloud.ply"
scripts\run_camera_probe_windows.bat
```

Then replay the selected camera in the single-frame baseline:

```powershell
$env:RESTIRGS_PLY="C:\path\to\point_cloud.ply"
$env:RESTIRGS_CAMERA_CONFIG="outputs\camera_probe_selected_camera.json"
scripts\run_ply_asset_windows.bat
```

It writes PLY render/G-buffer/lighting images plus:

```text
outputs/ply_asset_summary.json
```

See `docs/phase8_ply_asset_baseline.md` for the supported PLY schema, field conversions, robust auto-camera controls, and asset-scaled lighting assumptions.
See `docs/phase9_asset_camera_probe.md` for the camera probe grid, scoring formula, and selected camera JSON schema.

## Real-Asset Proposal Ablation

After selecting a camera, run the real-asset single-frame proposal ablation:

```powershell
$env:RESTIRGS_PLY="C:\path\to\point_cloud.ply"
$env:RESTIRGS_CAMERA_CONFIG="outputs\camera_probe_selected_camera.json"
scripts\run_real_asset_proposal_ablation_windows.bat
```

It writes:

```text
outputs/real_asset_proposal_ablation.csv
outputs/real_asset_proposal_ablation_summary.json
```

See `docs/phase10_real_asset_proposal_ablation.md` for the real-scene sweep settings and limitations.

## Defensive Spatial MIS Reuse

Run the verified defensive spatial MIS candidate reuse on the selected real view:

```powershell
$env:RESTIRGS_PLY="C:\path\to\point_cloud.ply"
$env:RESTIRGS_CAMERA_CONFIG="outputs\camera_probe_selected_camera.json"
scripts\run_spatial_mis_reuse_windows.bat
```

It writes:

```text
outputs/spatial_mis_ablation.csv
outputs/spatial_mis_ablation_summary.json
outputs/spatial_mis_best_composite.png
outputs/spatial_mis_best_abs_error.png
outputs/spatial_mis_initial_abs_error.png
```

See `docs/phase15_spatial_mis_reuse.md` for the defensive mixture proposal, MIS equation, and interpretation.

## Real-Asset Benchmark

Download the public Voxel51 benchmark assets:

```powershell
python scripts/download_voxel51_assets.py --dry-run
python scripts/download_voxel51_assets.py
```

Run the multi-scene/multi-view benchmark harness:

```powershell
scripts\run_real_asset_benchmark_windows.bat
```

It writes:

```text
outputs/benchmark/real_asset_benchmark_rows.csv
outputs/benchmark/real_asset_benchmark_summary.json
outputs/benchmark/<scene_id>/view_<index>/preview_rgb.png
outputs/benchmark/<scene_id>/view_<index>/camera.json
```

See `docs/phase16_real_asset_benchmark.md` for the manifest format and benchmark interpretation.
See `docs/phase17_public_asset_intake.md` for public asset download, four-scene manifest setup, and expected benchmark row counts.
