@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%" || exit /b 1

call scripts\_setup_windows_cuda_env.bat
if errorlevel 1 (
  set "STATUS=%ERRORLEVEL%"
  goto done
)

if "%RESTIRGS_RENDERER_OUTPUT_DIR%"=="" set "RESTIRGS_RENDERER_OUTPUT_DIR=outputs\aligned_restir"
if "%RESTIRGS_ACTIVE_DEMO_OUTPUT_DIR%"=="" set "RESTIRGS_ACTIVE_DEMO_OUTPUT_DIR=outputs\active_demo"

"%RESTIRGS_ENV%\python.exe" scripts\demo_28_active_renderer_snapshot.py ^
  --renderer-output-dir "%RESTIRGS_RENDERER_OUTPUT_DIR%" ^
  --output-dir "%RESTIRGS_ACTIVE_DEMO_OUTPUT_DIR%"

set "STATUS=%ERRORLEVEL%"

:done
popd
endlocal & exit /b %STATUS%
