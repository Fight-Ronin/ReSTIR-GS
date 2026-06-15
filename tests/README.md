# Test Map

The test suite is split between the active renderer package and the standalone
`gs_gen` helper.

## Commands

These commands assume the `restirgs` conda environment is active.

Run everything:

```powershell
python -m pytest -q
```

Run the active renderer core checks:

```powershell
python -m pytest tests/test_lighting.py tests/test_visibility_lighting.py tests/test_visibility_restir.py tests/test_restir_renderer.py -q
```

Run viewer and web viewer checks:

```powershell
python -m pytest tests/test_interactive_viewer.py tests/test_interactive_web_server.py -q
```

Run local asset-generation helper checks:

```powershell
python -m pytest gs_gen/tests -q
```

## Top-Level Tests

- `test_aligned_asset_registry.py`: manifest parsing, asset-set resolution,
  download dry runs, and registered loader routing.
- `test_ply_loader.py`, `test_scene_normalization.py`,
  `test_dxgl_gbuffer_validation.py`: asset and camera ingestion helpers.
- `test_gbuffer.py`, `test_asset_lights.py`, `test_lighting.py`,
  `test_visibility_lighting.py`: G-buffer, scene-stable lights, deferred
  diagnostics, and shadow-map visibility.
- `test_proposal.py`, `test_restir_initial.py`, `test_visibility_restir.py`,
  `test_temporal_reuse.py`, `test_restir_renderer.py`: proposal sampling,
  initial RIS, visibility-aware RIS, temporal reuse/filtering, and renderer
  orchestration.
- `test_active_baseline_handoff.py`, `test_active_renderer_snapshot.py`,
  `test_dxgl_sampling_benchmark.py`: active baseline and evaluation helpers.
- `test_interactive_viewer.py`, `test_interactive_web_server.py`,
  `test_orbit_camera.py`: interactive inspection surfaces.

## `gs_gen` Tests

`gs_gen/tests` covers config loading, source probing, video frame extraction
planning, Nerfstudio command planning, transform validation, PLY compatibility,
and staging. These tests should stay independent from the active renderer
manifest.

## Cleanup Rules

- Keep tests focused on maintained behavior or deliberately retained diagnostic
  surfaces.
- Do not remove Lambertian/Blinn-Phong diagnostic tests unless the viewer and
  smoke paths no longer expose those modes.
- Keep selected-fast tests tied to explicit experiment flags and quality/FPS
  harnesses; do not make selected-fast look like the default renderer policy.
