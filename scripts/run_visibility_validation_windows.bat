@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%" || exit /b 1

call scripts\_setup_windows_cuda_env.bat
if errorlevel 1 (
  set "STATUS=%ERRORLEVEL%"
  goto done
)

if not defined RESTIRGS_ALIGNED_MANIFEST set "RESTIRGS_ALIGNED_MANIFEST=configs\aligned_assets.json"
if not defined RESTIRGS_VISIBILITY_ASSET_SET set "RESTIRGS_VISIBILITY_ASSET_SET=testing"
if not defined RESTIRGS_VISIBILITY_RENDERER_FRAMES set "RESTIRGS_VISIBILITY_RENDERER_FRAMES=45,46,47"
if not defined RESTIRGS_VISIBILITY_RENDERER_OUTPUT set "RESTIRGS_VISIBILITY_RENDERER_OUTPUT=outputs\aligned_restir_visibility"

echo [1/2] Running visibility target smoke matrix...
if defined RESTIRGS_VISIBILITY_ASSET_IDS (
  "%RESTIRGS_ENV%\python.exe" scripts\demo_29_aligned_visibility_smoke_matrix.py --manifest "%RESTIRGS_ALIGNED_MANIFEST%" --asset-ids "%RESTIRGS_VISIBILITY_ASSET_IDS%" --device cuda %RESTIRGS_VISIBILITY_MATRIX_EXTRA_ARGS%
) else (
  "%RESTIRGS_ENV%\python.exe" scripts\demo_29_aligned_visibility_smoke_matrix.py --manifest "%RESTIRGS_ALIGNED_MANIFEST%" --asset-set "%RESTIRGS_VISIBILITY_ASSET_SET%" --device cuda %RESTIRGS_VISIBILITY_MATRIX_EXTRA_ARGS%
)
if errorlevel 1 (
  set "STATUS=%ERRORLEVEL%"
  goto done
)

echo [2/2] Running aligned renderer with visibility target...
if defined RESTIRGS_VISIBILITY_ASSET_IDS (
  "%RESTIRGS_ENV%\python.exe" scripts\demo_26_aligned_restir_renderer.py --manifest "%RESTIRGS_ALIGNED_MANIFEST%" --asset-ids "%RESTIRGS_VISIBILITY_ASSET_IDS%" --target-mode visibility --num-lights 16 --frame-indices "%RESTIRGS_VISIBILITY_RENDERER_FRAMES%" --width 128 --height 128 --output-dir "%RESTIRGS_VISIBILITY_RENDERER_OUTPUT%" --device cuda %RESTIRGS_VISIBILITY_RENDERER_EXTRA_ARGS%
) else (
  "%RESTIRGS_ENV%\python.exe" scripts\demo_26_aligned_restir_renderer.py --manifest "%RESTIRGS_ALIGNED_MANIFEST%" --asset-set "%RESTIRGS_VISIBILITY_ASSET_SET%" --target-mode visibility --num-lights 16 --frame-indices "%RESTIRGS_VISIBILITY_RENDERER_FRAMES%" --width 128 --height 128 --output-dir "%RESTIRGS_VISIBILITY_RENDERER_OUTPUT%" --device cuda %RESTIRGS_VISIBILITY_RENDERER_EXTRA_ARGS%
)
set "STATUS=%ERRORLEVEL%"

:done
popd
endlocal & exit /b %STATUS%
