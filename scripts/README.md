# Scripts

Published helper scripts for the current ReSTIR-GS deliverable.

The published `main` branch keeps the asset download utilities and interactive
viewer runners. Full validation, test, and baseline runner scripts are retained
on the `dev` branch.

Use the `.bat` viewer runners for CUDA and `gsplat` inspection. They call
`scripts\_setup_windows_cuda_env.bat`, which sets the `restirgs` Python path,
CUDA variables, torch extension cache, and `gsplat` Windows patch check.
Override `RESTIRGS_ENV` if your conda environment is not at
`%USERPROFILE%\miniconda3\envs\restirgs`.

## Download Assets

Preview and download the aligned testing set:

```powershell
python scripts/download_aligned_asset.py --asset-set testing --dry-run
python scripts/download_aligned_splat.py --asset-set testing --dry-run
python scripts/download_aligned_asset.py --asset-set testing
python scripts/download_aligned_splat.py --asset-set testing
```

The active manifest is `configs/aligned_assets.json`. Use `--asset-id` or
`--asset-set` to choose individual assets or named sets.

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
