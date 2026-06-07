# Current Architecture

## Active Data Flow

```text
configs/aligned_assets.json
-> restir_gs.render.aligned_asset_registry
-> dataset adapter: restir_gs.render.dxgl_asset
-> generic Gaussian loader: restir_gs.render.ply_loader.load_gaussian_asset
-> gsplat renderer: restir_gs.render.gsplat_renderer.render_rgbd
-> pseudo G-buffer: restir_gs.render.gbuffer
-> scene-stable world lights: restir_gs.lighting.asset_lights
-> visibility target: restir_gs.lighting.visibility
-> proposal / RIS: restir_gs.restir.proposal + restir_gs.restir.initial
-> temporal reuse/filter: restir_gs.restir.temporal + restir_gs.restir.renderer
```

## Active Scripts

```text
scripts/download_aligned_asset.py
scripts/download_aligned_splat.py
scripts/demo_24_aligned_asset_smoke_matrix.py
scripts/demo_26_aligned_restir_renderer.py
scripts/demo_22_interactive_viewer.py
scripts/run_active_validation_windows.bat
scripts/run_aligned_asset_smoke_matrix_windows.bat
scripts/run_aligned_restir_renderer_windows.bat
scripts/run_interactive_viewer_windows.bat
```

## Core Packages

`restir_gs.render`

- `aligned_asset_registry.py`: manifest parsing and registered asset loading.
- `dxgl_asset.py`: DXGL aligned camera/modality adapter.
- `ply_loader.py`: dataset-agnostic compatible 3DGS PLY loader.
- `gbuffer.py`: expected-depth unprojection and pseudo normal generation.
- `orbit_camera.py`: dataset-agnostic orbit camera math for the viewer.

`restir_gs.lighting`

- `asset_lights.py`: scene-stable world lights and camera-space conversion.
- `deferred.py`: Lambertian and Blinn-Phong deferred lighting helpers.
- `visibility.py`: shadow-map proxy and PCF-filtered visibility-aware direct lighting.

`restir_gs.restir`

- `proposal.py`: geometric and visibility-geometric proposal distributions.
- `initial.py`: MC/RIS estimators and reservoir state.
- `temporal.py`: reprojection lookup, compatibility diagnostics, and reservoir combine.
- `visibility.py`: visibility-aware MC/RIS estimator wrappers.
- `renderer.py`: active renderer composition layer and metric rows.

`restir_gs.eval`

- `gbuffer_validation.py`: aligned modality comparison helpers used by smoke matrix.
- `dxgl_sampling_benchmark.py`: small sampling smoke helper used by smoke matrix.

## Removed Legacy Surface

Synthetic demos, Voxel51 intake, single-view PLY camera probe, real-asset ablations, spatial MIS diagnostics, standalone visibility smoke scripts, and Apple-specific download scripts have been removed from the active source tree. The retained workflow starts from the aligned asset manifest.
