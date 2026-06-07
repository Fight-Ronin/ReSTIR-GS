# Scripts

Use this file to choose the right entrypoint. The repo keeps older phase scripts for reproducibility, but new development should use the active aligned workflow first.

## Active Entry Points

```text
download_aligned_asset.py
download_aligned_splat.py
demo_24_aligned_asset_smoke_matrix.py
demo_26_aligned_restir_renderer.py
demo_22_interactive_viewer.py
run_active_validation_windows.bat
run_aligned_asset_smoke_matrix_windows.bat
run_aligned_restir_renderer_windows.bat
run_interactive_viewer_windows.bat
```

Optional visibility diagnostics:

```text
demo_27_aligned_visibility_smoke.py
demo_28_aligned_visibility_ris_smoke.py
demo_29_aligned_visibility_smoke_matrix.py
run_visibility_validation_windows.bat
```

## Active Workflow

```powershell
python scripts/download_aligned_asset.py --asset-set testing --dry-run
python scripts/download_aligned_splat.py --asset-set testing --dry-run
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
scripts\run_active_validation_windows.bat
```

`run_aligned_restir_renderer_windows.bat` defaults to the visibility-aware active renderer target. Set `RESTIRGS_RESTIR_TARGET_MODE=diffuse` for the retained diffuse compatibility baseline.

For interactive inspection:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```

## Support Scripts

```text
_setup_windows_cuda_env.bat
patch_gsplat_windows.py
```

`_setup_windows_cuda_env.bat` is a shared Windows runner preflight. It sets the Visual Studio x64 environment, conda CUDA paths, torch extension cache, matplotlib cache, and checks the local `gsplat` Windows patch.

## Historical / Compatibility Scripts

The remaining `demo_*`, `download_voxel51_*`, Apple-specific DXGL intake, single-view PLY, Voxel51, and older ablation runners are retained for reproducing earlier phases. They are not the preferred surface for new aligned ReSTIR work. See `docs/legacy_inventory.md` for the retained historical surface.
