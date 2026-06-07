@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%" || exit /b 1

call scripts\_setup_windows_cuda_env.bat
if errorlevel 1 (
  set "STATUS=%ERRORLEVEL%"
  goto done
)

set "RESTIRGS_SKIP_CL_WARNING=1"
if not defined RESTIRGS_ALIGNED_MANIFEST set "RESTIRGS_ALIGNED_MANIFEST=configs\aligned_assets.json"
if not defined RESTIRGS_VIEWER_WIDTH set "RESTIRGS_VIEWER_WIDTH=768"
if not defined RESTIRGS_VIEWER_HEIGHT set "RESTIRGS_VIEWER_HEIGHT=768"

if defined RESTIRGS_VIEWER_PLY goto generic_ply
if defined RESTIRGS_VIEWER_ASSET_ID goto registered_asset

"%RESTIRGS_ENV%\python.exe" -m interactive.launcher --width "%RESTIRGS_VIEWER_WIDTH%" --height "%RESTIRGS_VIEWER_HEIGHT%" --output-dir outputs\interactive_viewer --frame-index 49 %RESTIRGS_VIEWER_EXTRA_ARGS%
set "STATUS=%ERRORLEVEL%"
goto done

:registered_asset
"%RESTIRGS_ENV%\python.exe" -m interactive.launcher --width "%RESTIRGS_VIEWER_WIDTH%" --height "%RESTIRGS_VIEWER_HEIGHT%" --output-dir outputs\interactive_viewer --manifest "%RESTIRGS_ALIGNED_MANIFEST%" --asset-id "%RESTIRGS_VIEWER_ASSET_ID%" --frame-index 49 %RESTIRGS_VIEWER_EXTRA_ARGS%
set "STATUS=%ERRORLEVEL%"
goto done

:generic_ply
if defined RESTIRGS_VIEWER_CAMERA_CONFIG goto generic_ply_with_camera

"%RESTIRGS_ENV%\python.exe" -m interactive.launcher --width "%RESTIRGS_VIEWER_WIDTH%" --height "%RESTIRGS_VIEWER_HEIGHT%" --output-dir outputs\interactive_viewer --ply "%RESTIRGS_VIEWER_PLY%" %RESTIRGS_VIEWER_EXTRA_ARGS%
set "STATUS=%ERRORLEVEL%"
goto done

:generic_ply_with_camera
"%RESTIRGS_ENV%\python.exe" -m interactive.launcher --width "%RESTIRGS_VIEWER_WIDTH%" --height "%RESTIRGS_VIEWER_HEIGHT%" --output-dir outputs\interactive_viewer --ply "%RESTIRGS_VIEWER_PLY%" --camera-config "%RESTIRGS_VIEWER_CAMERA_CONFIG%" %RESTIRGS_VIEWER_EXTRA_ARGS%
set "STATUS=%ERRORLEVEL%"

:done
popd
endlocal & exit /b %STATUS%
