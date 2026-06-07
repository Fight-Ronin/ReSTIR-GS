@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%" || exit /b 1

if not defined RESTIRGS_SKIP_WINDOWS_PREFLIGHT (
  call scripts\_setup_windows_cuda_env.bat
  if errorlevel 1 (
    set "STATUS=%ERRORLEVEL%"
    goto done
  )
)

if not defined RESTIRGS_ALIGNED_MANIFEST set "RESTIRGS_ALIGNED_MANIFEST=configs\aligned_assets.json"
if not defined RESTIRGS_ALIGNED_ASSET_SET set "RESTIRGS_ALIGNED_ASSET_SET=testing"
if not defined RESTIRGS_RESTIR_TARGET_MODE set "RESTIRGS_RESTIR_TARGET_MODE=visibility"
if not defined RESTIRGS_RESTIR_NUM_LIGHTS set "RESTIRGS_RESTIR_NUM_LIGHTS=16"
if not defined RESTIRGS_RESTIR_WIDTH set "RESTIRGS_RESTIR_WIDTH=128"
if not defined RESTIRGS_RESTIR_HEIGHT set "RESTIRGS_RESTIR_HEIGHT=128"
if not defined RESTIRGS_RESTIR_FRAME_INDICES set "RESTIRGS_RESTIR_FRAME_INDICES=45,46,47"
if not defined RESTIRGS_RESTIR_OUTPUT_DIR set "RESTIRGS_RESTIR_OUTPUT_DIR=outputs\aligned_restir"

set "RESTIRGS_RESTIR_FRAME_ARGS=--frame-indices %RESTIRGS_RESTIR_FRAME_INDICES%"
if "%RESTIRGS_RESTIR_FRAME_INDICES%"=="" set "RESTIRGS_RESTIR_FRAME_ARGS="
if /I "%RESTIRGS_RESTIR_FRAME_INDICES%"=="manifest" set "RESTIRGS_RESTIR_FRAME_ARGS="

if defined RESTIRGS_ALIGNED_ASSET_IDS goto run_asset_ids

"%RESTIRGS_ENV%\python.exe" scripts\demo_26_aligned_restir_renderer.py --manifest "%RESTIRGS_ALIGNED_MANIFEST%" --asset-set "%RESTIRGS_ALIGNED_ASSET_SET%" --target-mode "%RESTIRGS_RESTIR_TARGET_MODE%" --num-lights "%RESTIRGS_RESTIR_NUM_LIGHTS%" --width "%RESTIRGS_RESTIR_WIDTH%" --height "%RESTIRGS_RESTIR_HEIGHT%" %RESTIRGS_RESTIR_FRAME_ARGS% --output-dir "%RESTIRGS_RESTIR_OUTPUT_DIR%" --device cuda %RESTIRGS_RESTIR_EXTRA_ARGS%
set "STATUS=%ERRORLEVEL%"
goto done

:run_asset_ids
"%RESTIRGS_ENV%\python.exe" scripts\demo_26_aligned_restir_renderer.py --manifest "%RESTIRGS_ALIGNED_MANIFEST%" --asset-ids "%RESTIRGS_ALIGNED_ASSET_IDS%" --target-mode "%RESTIRGS_RESTIR_TARGET_MODE%" --num-lights "%RESTIRGS_RESTIR_NUM_LIGHTS%" --width "%RESTIRGS_RESTIR_WIDTH%" --height "%RESTIRGS_RESTIR_HEIGHT%" %RESTIRGS_RESTIR_FRAME_ARGS% --output-dir "%RESTIRGS_RESTIR_OUTPUT_DIR%" --device cuda %RESTIRGS_RESTIR_EXTRA_ARGS%
set "STATUS=%ERRORLEVEL%"

:done
popd
endlocal & exit /b %STATUS%
