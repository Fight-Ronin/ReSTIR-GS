# ReSTIR-GS

Windows-native ReSTIR-GS prototype. The current active path is manifest-driven aligned 3DGS assets, pseudo G-buffers, scene-stable world lights, initial RIS, and previous-frame temporal reservoir reuse.

For the concise current workflow, see `docs/active_workflow.md`. For the module map and historical boundaries, see `docs/current_architecture.md`.

## Environment

Create the conda environment and install Python dependencies:

```powershell
conda env create -f environment.yml
conda activate restirgs
pip install -r requirements.txt
```

The current verified stack is Python 3.10, CUDA toolkit 12.4, PyTorch `2.5.1+cu124`, and `gsplat==1.5.3`.

## Current Active Workflow

Download aligned testing assets:

```powershell
python scripts/download_aligned_asset.py --asset-set testing --dry-run
python scripts/download_aligned_splat.py --asset-set testing --dry-run
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

Run the active validation path:

```powershell
scripts\run_active_validation_windows.bat
```

Inspect an aligned asset interactively:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```

Main active outputs:

```text
outputs/aligned_smoke/
outputs/aligned_restir/
outputs/interactive_viewer/
```

Older synthetic, Voxel51, single-view PLY, and broad ablation sections below are retained for reproduction and context; they are not the preferred expansion surface for new aligned work.

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

## Aligned Dataset Intake

Download the manifest-registered aligned testing set:

```powershell
python scripts/download_aligned_asset.py --asset-set testing --dry-run
python scripts/download_aligned_splat.py --asset-set testing --dry-run
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

Run the small manifest-driven smoke matrix:

```powershell
scripts\run_aligned_asset_smoke_matrix_windows.bat
```

The Windows runner calls `vcvars64.bat`, checks the local `gsplat` patch, and uses the manifest-driven path. It accepts:

```powershell
$env:RESTIRGS_ALIGNED_MANIFEST="configs\aligned_assets.json"
$env:RESTIRGS_ALIGNED_ASSET_SET="testing"
$env:RESTIRGS_ALIGNED_SMOKE_EXTRA_ARGS="--width 128 --height 128"
```

For targeted debugging, `RESTIRGS_ALIGNED_ASSET_IDS` overrides `RESTIRGS_ALIGNED_ASSET_SET`:

```powershell
$env:RESTIRGS_ALIGNED_ASSET_IDS="dxgl_apple,dxgl_drill"
```

Direct Python execution remains available from an x64 Visual Studio developer shell or equivalent `vcvars64.bat` environment:

```powershell
python scripts/demo_24_aligned_asset_smoke_matrix.py --asset-set testing --device cuda
```

It writes:

```text
outputs/aligned_smoke/aligned_asset_smoke_rows.csv
outputs/aligned_smoke/aligned_asset_smoke_summary.json
outputs/aligned_smoke/<asset_id>/contact.png
```

See `docs/phase28_aligned_asset_registry.md` for the manifest format and dataset-agnostic Gaussian loading boundary.
See `docs/phase30_aligned_testing_assets.md` for the current aligned testing set.

The older Apple-specific intake commands remain available:

```powershell
python scripts/download_dxgl_apple.py --dry-run
python scripts/download_dxgl_apple.py
```

Inspect aligned RGB/mask/depth/normal frames and imported cameras:

```powershell
python scripts/demo_17_dxgl_aligned_intake.py
```

It writes:

```text
outputs/aligned_fidelity/dxgl_apple_contact.png
outputs/aligned_fidelity/dxgl_apple_intake_summary.json
outputs/aligned_fidelity/dxgl_apple_frame_<index>_camera.json
```

See `docs/phase19_aligned_dataset_intake.md` for the DXGL dataset validation rules, camera convention conversion, and current `points3D.ply` compatibility probe.

## DXGL Splat Fidelity Smoke

Download the DXGL Apple pretrained splat:

```powershell
python scripts/download_dxgl_apple_splat.py --dry-run
python scripts/download_dxgl_apple_splat.py
```

Render it from the aligned `transforms.json` cameras:

```powershell
scripts\run_dxgl_splat_fidelity_windows.bat
```

It writes:

```text
outputs/aligned_fidelity/dxgl_apple_splat_contact.png
outputs/aligned_fidelity/dxgl_apple_splat_fidelity_summary.json
```

See `docs/phase20_dxgl_splat_fidelity.md` for the splat validation, camera scaling, masked RGB metrics, and interpretation.
See `docs/phase21_dxgl_camera_normalization.md` for the raw `transforms.json` to normalized splat-space camera fix.

## DXGL G-buffer Validation

Validate the aligned real-asset RGB/depth/alpha/pseudo-normal buffers:

```powershell
scripts\run_dxgl_gbuffer_validation_windows.bat
```

It writes:

```text
outputs/aligned_gbuffer/dxgl_apple_gbuffer_contact.png
outputs/aligned_gbuffer/dxgl_apple_gbuffer_summary.json
```

See `docs/phase22_dxgl_gbuffer_validation.md` for the modality comparisons and geometry-buffer gate.

## Blinn-Phong Deferred Lighting

Run the dataset-agnostic Blinn-Phong shader on the aligned DXGL Apple validation asset:

```powershell
scripts\run_dxgl_blinn_phong_lighting_windows.bat
```

It writes:

```text
outputs/aligned_lighting/dxgl_blinn_phong_lighting_contact.png
outputs/aligned_lighting/dxgl_blinn_phong_lighting_summary.json
```

See `docs/phase23_blinn_phong_lighting.md` for the generic Gaussian asset loader, shader details, and the opt-in Blinn-Phong RIS target probe.

## DXGL Aligned Sampling Benchmark

Run the multi-frame aligned MC/RIS benchmark comparing diffuse and Blinn-Phong targets:

```powershell
scripts\run_dxgl_sampling_benchmark_windows.bat
```

It writes:

```text
outputs/aligned_sampling/dxgl_sampling_rows.csv
outputs/aligned_sampling/dxgl_sampling_summary.json
outputs/aligned_sampling/dxgl_sampling_contact.png
```

See `docs/phase24_dxgl_aligned_sampling_benchmark.md` for the benchmark setup and interpretation.

## Interactive Viewer

Open the default aligned DXGL Apple debug viewer:

```powershell
scripts\run_interactive_viewer_windows.bat
```

Or view any compatible 3DGS PLY through the generic loader:

```powershell
$env:RESTIRGS_VIEWER_PLY="C:\path\to\splat.ply"
scripts\run_interactive_viewer_windows.bat
```

The runner calls the Visual Studio x64 setup needed by `gsplat` CUDA JIT. Direct `python ... --device cuda` is only safe from a VS x64 dev shell or after calling `vcvars64.bat`.

Use it to orbit/pan/dolly the active splat, inspect RGB/G-buffer/lighting panels, and save a replayable camera config under `outputs/interactive_viewer/`. See `docs/phase25_interactive_viewer.md` for controls, generic PLY mode, and saved-output details.

## DXGL Aligned Temporal Reuse Smoke

Run the aligned consecutive-frame temporal reprojection and reservoir reuse smoke with scene-stable world-space lights:

```powershell
scripts\run_dxgl_temporal_reuse_windows.bat
```

It writes CSV/JSON metrics, a contact sheet, and final-frame debug previews under `outputs/aligned_temporal/`.

See `docs/phase27_scene_stable_lights.md` for why the temporal path now uses fixed world-space light identities before converting to per-frame camera-space `PointLights`.

## Aligned ReSTIR Renderer Path

Run the registry-driven aligned ReSTIR renderer path on the active testing asset set:

```powershell
scripts\run_aligned_asset_smoke_matrix_windows.bat
scripts\run_aligned_restir_renderer_windows.bat
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```

It writes CSV/JSON metrics and per-asset contact sheets under `outputs/aligned_restir/`. In the interactive viewer, press `4` to inspect the single-frame ReSTIR debug panel. See `docs/phase31_aligned_restir_renderer.md` for the renderer contract and interpretation.

## Legacy Voxel51 Benchmark

Voxel51 benchmark code and docs remain in the repo for historical diagnostics, but Voxel51 is no longer the active dataset path because the available assets are not camera-aligned for photometric comparison. See `docs/phase16_real_asset_benchmark.md`, `docs/phase17_public_asset_intake.md`, and `docs/phase18_reference_fidelity_triage.md` if you need to reproduce those smoke tests.
