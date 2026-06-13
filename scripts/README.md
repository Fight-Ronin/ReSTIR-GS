# Scripts

Windows runners and small utility scripts for the current ReSTIR-GS implementation.

Use the `.bat` runners for CUDA and `gsplat` commands. They call `scripts\_setup_windows_cuda_env.bat`, which sets the `restirgs` Python path, CUDA variables, torch extension cache, and `gsplat` Windows patch check. Override `RESTIRGS_ENV` if your conda environment is not at `%USERPROFILE%\miniconda3\envs\restirgs`.

## Active Workflow

Download aligned assets:

```powershell
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

Run the full active baseline:

```powershell
scripts\run_active_baseline_demo_windows.bat
```

This calls:

```text
scripts\run_active_validation_windows.bat
scripts\run_active_demo_snapshot_windows.bat
```

Run validation only:

```powershell
scripts\run_active_validation_windows.bat
```

Build the demo snapshot from existing renderer outputs:

```powershell
scripts\run_active_demo_snapshot_windows.bat
```

## Individual Renderer Commands

Aligned asset smoke matrix:

```powershell
scripts\run_aligned_asset_smoke_matrix_windows.bat
```

Aligned ReSTIR renderer:

```powershell
scripts\run_aligned_restir_renderer_windows.bat
```

The aligned renderer runner defaults to:

```text
RESTIRGS_RESTIR_TARGET_MODE=visibility
RESTIRGS_RESTIR_NUM_LIGHTS=16
RESTIRGS_RESTIR_WIDTH=128
RESTIRGS_RESTIR_HEIGHT=128
RESTIRGS_RESTIR_FRAME_INDICES=45,46,47
RESTIRGS_RESTIR_OUTPUT_DIR=outputs\aligned_restir
```

Use `RESTIRGS_ALIGNED_ASSET_SET` or `RESTIRGS_ALIGNED_ASSET_IDS` to choose assets.

## Interactive Viewers

Matplotlib viewer:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```

Browser viewer:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_web_viewer_windows.bat
```

The browser server listens on `http://127.0.0.1:8765` by default. Override with `RESTIRGS_WEB_HOST` and `RESTIRGS_WEB_PORT`.

Default viewer sizes:

```text
matplotlib runner = 768x768
browser runner    = 1024x1024
```

Override with `RESTIRGS_VIEWER_WIDTH` and `RESTIRGS_VIEWER_HEIGHT`.

Save display-side visibility output:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
$env:RESTIRGS_VIEWER_EXTRA_ARGS="--save-and-exit --save-visibility"
scripts\run_interactive_viewer_windows.bat
```

Save reference/error visibility output:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
$env:RESTIRGS_VIEWER_EXTRA_ARGS="--save-and-exit --save-visibility-reference"
scripts\run_interactive_viewer_windows.bat
```

Inspect a generic compatible 3DGS PLY:

```powershell
$env:RESTIRGS_VIEWER_PLY="path\to\asset.ply"
scripts\run_interactive_viewer_windows.bat
```

Optionally set `RESTIRGS_VIEWER_CAMERA_CONFIG` for generic PLY mode.

## Python Utilities

```text
download_aligned_asset.py              download registered aligned datasets
download_aligned_splat.py              download registered aligned splats
demo_24_aligned_asset_smoke_matrix.py  validate aligned assets and sampling basics
demo_26_aligned_restir_renderer.py     run the active renderer/evaluator path
demo_28_active_renderer_snapshot.py    build the compact demo/performance snapshot
patch_gsplat_windows.py                verify or patch gsplat 1.5.x MSVC JIT flags
```

The interactive implementation lives in `interactive/`:

```text
launcher.py     matplotlib viewer CLI
web_server.py   browser viewer server
rendering.py    backend renderer adapter and save helpers
session.py      frame, camera, layer, and render state
camera.py       free-camera movement
layers.py       shared view-layer registry
viewer.py       matplotlib client
web/            browser UI assets
```
