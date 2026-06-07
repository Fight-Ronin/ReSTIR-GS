@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%" || exit /b 1

echo [1/2] Running active validation...
call scripts\run_active_validation_windows.bat
if errorlevel 1 (
  set "STATUS=%ERRORLEVEL%"
  goto done
)

echo [2/2] Building active demo snapshot...
call scripts\run_active_demo_snapshot_windows.bat
set "STATUS=%ERRORLEVEL%"

:done
popd
endlocal & exit /b %STATUS%
