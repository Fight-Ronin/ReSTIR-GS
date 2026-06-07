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
if not defined RESTIRGS_VISIBILITY_RIS_ASSET_ID set "RESTIRGS_VISIBILITY_RIS_ASSET_ID=dxgl_apple"

"%RESTIRGS_ENV%\python.exe" scripts\demo_28_aligned_visibility_ris_smoke.py --manifest "%RESTIRGS_ALIGNED_MANIFEST%" --asset-id "%RESTIRGS_VISIBILITY_RIS_ASSET_ID%" --device cuda %RESTIRGS_VISIBILITY_RIS_EXTRA_ARGS%
set "STATUS=%ERRORLEVEL%"

:done
popd
endlocal & exit /b %STATUS%
