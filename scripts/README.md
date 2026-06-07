# Scripts

Current active entrypoints:

```text
download_aligned_asset.py
download_aligned_splat.py
demo_24_aligned_asset_smoke_matrix.py
demo_26_aligned_restir_renderer.py
demo_28_active_renderer_snapshot.py
run_active_baseline_demo_windows.bat
run_active_validation_windows.bat
run_active_demo_snapshot_windows.bat
run_aligned_asset_smoke_matrix_windows.bat
run_aligned_restir_renderer_windows.bat
run_interactive_viewer_windows.bat
run_interactive_web_viewer_windows.bat
patch_gsplat_windows.py
_setup_windows_cuda_env.bat
```

Use Windows runners for CUDA/`gsplat` commands. They set the VS x64/CUDA environment, torch extension cache, and run the installed `gsplat` compatibility patch check.

Generic downloads:

```powershell
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

Active baseline demo:

```powershell
scripts\run_active_baseline_demo_windows.bat
```

Active validation only:

```powershell
scripts\run_active_validation_windows.bat
```

Interactive viewer:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```

Browser WebUI prototype:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_web_viewer_windows.bat
```

The server listens on `http://127.0.0.1:8765` by default. Override with `RESTIRGS_WEB_HOST` and `RESTIRGS_WEB_PORT`.
The WebUI runner defaults to `1024x1024`; override `RESTIRGS_VIEWER_WIDTH` and `RESTIRGS_VIEWER_HEIGHT` for faster lower-resolution interaction.

Display-side visibility save:

```powershell
$env:RESTIRGS_VIEWER_EXTRA_ARGS="--save-and-exit --save-visibility"
scripts\run_interactive_viewer_windows.bat
```

Reference/error visibility save:

```powershell
$env:RESTIRGS_VIEWER_EXTRA_ARGS="--save-and-exit --save-visibility-reference"
scripts\run_interactive_viewer_windows.bat
```

The interactive viewer implementation lives in `interactive/`: `launcher.py` handles the matplotlib CLI, `web_server.py` handles the browser prototype, `rendering.py` adapts the backend renderer, `session.py` owns interactive state, `camera.py` owns free-camera movement, `layers.py` owns view-layer triggers, and `viewer.py` is the matplotlib client. The Windows runners call those package entrypoints.
