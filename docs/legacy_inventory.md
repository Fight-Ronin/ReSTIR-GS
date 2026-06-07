# Legacy Inventory

This repo keeps historical scripts and docs for reproducibility, but new work should start from `docs/active_workflow.md` and the registry-driven aligned asset path.

## Active Surface

Use these for current work:

```text
configs/aligned_assets.json
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

Optional visibility diagnostics:

```text
scripts/demo_27_aligned_visibility_smoke.py
scripts/demo_28_aligned_visibility_ris_smoke.py
scripts/demo_29_aligned_visibility_smoke_matrix.py
scripts/run_visibility_validation_windows.bat
```

## Retained Historical Surface

These remain available because tests or phase reproduction still reference them:

```text
configs/real_asset_benchmark.json
scripts/download_voxel51_assets.py
scripts/download_voxel51_references.py
scripts/demo_07_ply_asset_baseline.py
scripts/demo_08_asset_camera_probe.py
scripts/demo_09_real_asset_proposal_ablation.py
scripts/demo_14_spatial_mis_reuse.py
scripts/demo_15_real_asset_benchmark.py
scripts/demo_16_render_fidelity_triage.py
scripts/demo_17_dxgl_aligned_intake.py
scripts/demo_18_dxgl_splat_fidelity.py
scripts/demo_19_dxgl_gbuffer_validation.py
scripts/demo_20_dxgl_blinn_phong_lighting.py
scripts/demo_21_dxgl_sampling_benchmark.py
scripts/demo_23_dxgl_temporal_reuse.py
```

They are not the preferred expansion surface. Do not add new features to these files unless the goal is to reproduce or repair an earlier phase.

## Cleanup Rule

Before moving or deleting a historical file, first update the tests that import it and verify whether the phase output is still needed. Until then, retaining the file is less risky than breaking reproducibility.
