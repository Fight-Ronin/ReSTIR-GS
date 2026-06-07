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
if not defined RESTIRGS_ALIGNED_ASSET_SET set "RESTIRGS_ALIGNED_ASSET_SET=testing"

echo [1/2] Running aligned asset smoke matrix...
if defined RESTIRGS_ALIGNED_ASSET_IDS goto smoke_asset_ids
"%RESTIRGS_ENV%\python.exe" scripts\demo_24_aligned_asset_smoke_matrix.py --manifest "%RESTIRGS_ALIGNED_MANIFEST%" --asset-set "%RESTIRGS_ALIGNED_ASSET_SET%" --device cuda %RESTIRGS_ALIGNED_SMOKE_EXTRA_ARGS%
goto after_smoke

:smoke_asset_ids
"%RESTIRGS_ENV%\python.exe" scripts\demo_24_aligned_asset_smoke_matrix.py --manifest "%RESTIRGS_ALIGNED_MANIFEST%" --asset-ids "%RESTIRGS_ALIGNED_ASSET_IDS%" --device cuda %RESTIRGS_ALIGNED_SMOKE_EXTRA_ARGS%

:after_smoke
if errorlevel 1 (
  set "STATUS=%ERRORLEVEL%"
  goto done
)

echo [2/2] Running aligned ReSTIR renderer path...
set "RESTIRGS_SKIP_WINDOWS_PREFLIGHT=1"
call scripts\run_aligned_restir_renderer_windows.bat
set "STATUS=%ERRORLEVEL%"
goto done

:done
popd
endlocal & exit /b %STATUS%
