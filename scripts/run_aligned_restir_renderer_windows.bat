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

if defined RESTIRGS_ALIGNED_ASSET_IDS goto run_asset_ids

"%RESTIRGS_ENV%\python.exe" scripts\demo_26_aligned_restir_renderer.py --manifest "%RESTIRGS_ALIGNED_MANIFEST%" --asset-set "%RESTIRGS_ALIGNED_ASSET_SET%" --device cuda %RESTIRGS_RESTIR_EXTRA_ARGS%
set "STATUS=%ERRORLEVEL%"
goto done

:run_asset_ids
"%RESTIRGS_ENV%\python.exe" scripts\demo_26_aligned_restir_renderer.py --manifest "%RESTIRGS_ALIGNED_MANIFEST%" --asset-ids "%RESTIRGS_ALIGNED_ASSET_IDS%" --device cuda %RESTIRGS_RESTIR_EXTRA_ARGS%
set "STATUS=%ERRORLEVEL%"

:done
popd
endlocal & exit /b %STATUS%
