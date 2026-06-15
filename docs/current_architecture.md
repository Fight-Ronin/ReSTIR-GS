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
-> visibility target: restir_gs.lighting.shadow_* + restir_gs.lighting.visible_lighting
-> proposal / RIS: restir_gs.restir.proposal + restir_gs.restir.initial
-> temporal reuse/filter: restir_gs.restir.temporal + restir_gs.restir.temporal_filter
-> renderer orchestration/metrics facade: restir_gs.restir.renderer
```

## Active Scripts

```text
scripts/download_aligned_asset.py
scripts/download_aligned_splat.py
scripts/demo_24_aligned_asset_smoke_matrix.py
scripts/demo_26_aligned_restir_renderer.py
scripts/demo_28_active_renderer_snapshot.py
scripts/run_active_baseline_demo_windows.bat
scripts/run_active_validation_windows.bat
scripts/run_aligned_asset_smoke_matrix_windows.bat
scripts/run_aligned_restir_renderer_windows.bat
scripts/run_active_demo_snapshot_windows.bat
scripts/run_interactive_viewer_windows.bat
scripts/run_interactive_web_viewer_windows.bat
```

`interactive.launcher` is the matplotlib viewer launcher and data adapter. `interactive.web_server` serves the browser prototype over the same session/rendering boundary.

## Core Packages

`restir_gs.render`

- `aligned_asset_registry.py`: manifest parsing and registered asset loading.
- `dxgl_asset.py`: DXGL aligned camera/modality adapter.
- `ply_loader.py`: dataset-agnostic compatible 3DGS PLY loader.
- `gbuffer.py`: expected-depth unprojection and pseudo normal generation.
- `orbit_camera.py`: dataset-agnostic orbit camera math for the viewer.

`interactive`

- `launcher.py`: interactive viewer CLI, registered-asset/generic-PLY adapters, and runtime preflight.
- `camera.py`: free-camera movement helpers that still materialize backend-compatible `PinholeCamera` poses.
- `layers.py`: shared view-layer registry, hotkeys, and render requirement mapping.
- `rendering.py`: backend renderer adapter, viewer render result types, output image conversion, and save helpers.
- `session.py`: data-agnostic interactive state for frame, camera, active layer, and render refresh policy.
- `viewer.py`: matplotlib inspector client wired through the active ReSTIR renderer backend.
- `web_server.py`: FastAPI server for the browser prototype; it exposes snapshots, PNG renders, layer triggers, camera actions, and save.
- `web/`: static browser UI for the WebUI prototype.

`restir_gs.lighting`

- `asset_lights.py`: scene-stable world lights and camera-space conversion.
- `deferred.py`: diffuse and Blinn-Phong diagnostic helpers retained for compatibility; they are not the current optimization target.
- `shadow_maps.py`: shadow-map camera setup and expected-depth shadow-map bundle rendering.
- `shadow_visibility.py`: dense/cache/selected shadow-map visibility evaluation.
- `visible_lighting.py`: visibility-aware direct diffuse lighting wrappers.
- `visibility.py`: compatibility facade that re-exports the visibility API.

`restir_gs.restir`

- `proposal.py`: geometric and visibility-geometric proposal distributions.
- `initial.py`: MC/RIS estimators and reservoir state.
- `temporal.py`: reprojection lookup, compatibility diagnostics, and reservoir combine.
- `temporal_filter.py`: confidence-weighted temporal RGB filter and empty temporal lookup/stat helpers.
- `visibility.py`: visibility-aware MC/RIS estimator wrappers.
- `types.py`: renderer settings, frame result dataclasses, and GPU-event timing field definitions.
- `metrics.py`: ReSTIR metric rows, timing summaries, finite-value checks, and small metric helpers.
- `renderer.py`: active renderer composition layer and compatibility facade. It exposes a reference-free display frame path for interactive inspection and an evaluation frame path for all-lights references.

`restir_gs.eval`

- `gbuffer_validation.py`: aligned modality comparison helpers used by smoke matrix.
- `dxgl_sampling_benchmark.py`: small sampling smoke helper used by smoke matrix.
- `active_renderer_snapshot.py`: active renderer demo/performance snapshot helpers.

## Removed Legacy Surface

Historical phase notes, closeout experiment scripts, synthetic demos, Voxel51 intake, single-view PLY camera probe, real-asset ablations, spatial MIS diagnostics, standalone visibility smoke scripts, and Apple-specific download scripts have been removed from the deliverable surface. The retained workflow starts from the aligned asset manifest.

## Display And Evaluation Boundary

Interactive inspection should use display-oriented outputs unless reference/error images are explicitly requested. The active validation and demo snapshot remain evaluation-oriented because they need all-lights references, CSV metrics, and error maps.
