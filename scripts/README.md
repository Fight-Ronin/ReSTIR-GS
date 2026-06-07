# Scripts

Current active entrypoints:

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
patch_gsplat_windows.py
_setup_windows_cuda_env.bat
```

Use Windows runners for CUDA/`gsplat` commands. They set the VS x64/CUDA environment, torch extension cache, and run the installed `gsplat` compatibility patch check.

Generic downloads:

```powershell
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

Active validation:

```powershell
scripts\run_active_validation_windows.bat
```

Interactive viewer:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```
