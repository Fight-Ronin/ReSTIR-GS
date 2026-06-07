# ReSTIR-GS

Windows-native ReSTIR-GS prototype for aligned 3D Gaussian Splatting assets.

The active path is intentionally small:

```text
configs/aligned_assets.json
-> generic aligned asset download
-> generic 3DGS PLY loading
-> aligned cameras + pseudo G-buffer
-> PCF visibility-aware direct lighting
-> visibility-geometric proposal
-> initial RIS
-> local-repaired temporal reprojection
-> confidence-clamped temporal filtered output
```

Older synthetic, Voxel51, single-view PLY, standalone ablation, and diagnostic-only scripts have been removed from the active source tree. The current codebase is organized around the registry-driven aligned workflow.

## Environment

Use the existing `restirgs` conda environment. On Windows, prefer the provided runners because they set the Visual Studio/CUDA/torch-extension environment before touching `gsplat`.

Useful checks:

```powershell
conda activate restirgs
python -m pytest -q
python -m compileall restir_gs scripts
python -m pip check
```

## Download Aligned Assets

Dry-run first:

```powershell
python scripts/download_aligned_asset.py --asset-set testing --dry-run
python scripts/download_aligned_splat.py --asset-set testing --dry-run
```

Download the DXGL testing set:

```powershell
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

The default testing set is:

```text
dxgl_apple
dxgl_cash_register
dxgl_drill
dxgl_fire_extinguisher
```

Assets and generated outputs stay under ignored `outputs/`.

## Active Validation

Run the active smoke matrix plus active renderer:

```powershell
scripts\run_active_validation_windows.bat
```

Expected active outputs:

```text
outputs/aligned_smoke/aligned_asset_smoke_rows.csv
outputs/aligned_smoke/aligned_asset_smoke_summary.json
outputs/aligned_restir/restir_renderer_rows.csv
outputs/aligned_restir/restir_renderer_summary.json
outputs/aligned_restir/<asset_id>/contact.png
```

The active renderer should report:

```text
target_mode = visibility
proposal = visibility_geometric
visibility_shadow_pcf_radius = 1
temporal_reprojection_search_radius = 1
temporal_history_m_cap = 1
```

`temporal_filtered_ris` is the preferred temporal output. `temporal_ris` remains a reservoir-combine debug row.

## Individual Active Commands

Aligned asset smoke matrix:

```powershell
scripts\run_aligned_asset_smoke_matrix_windows.bat
```

Aligned ReSTIR renderer:

```powershell
scripts\run_aligned_restir_renderer_windows.bat
```

Interactive viewer:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```

The viewer also supports a generic compatible 3DGS PLY:

```powershell
python scripts/demo_22_interactive_viewer.py --ply path\to\asset.ply --device cuda
```

## Important Files

```text
configs/aligned_assets.json
restir_gs/render/aligned_asset_registry.py
restir_gs/render/ply_loader.py
restir_gs/render/gbuffer.py
restir_gs/lighting/visibility.py
restir_gs/restir/renderer.py
restir_gs/restir/temporal.py
scripts/demo_24_aligned_asset_smoke_matrix.py
scripts/demo_26_aligned_restir_renderer.py
scripts/demo_22_interactive_viewer.py
```

See:

```text
docs/active_workflow.md
docs/current_architecture.md
docs/current_milestone_snapshot.md
docs/phase44_temporal_reprojection_repair.md
```
