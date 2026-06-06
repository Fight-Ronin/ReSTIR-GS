@echo off

if not defined VS_VCVARS64 set "VS_VCVARS64=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VS_VCVARS64%" goto have_vs
echo Missing Visual Studio vcvars64.bat: %VS_VCVARS64%
exit /b 1

:have_vs
call "%VS_VCVARS64%" || exit /b 1

if not defined RESTIRGS_ENV set "RESTIRGS_ENV=%USERPROFILE%\miniconda3\envs\restirgs"
if exist "%RESTIRGS_ENV%\python.exe" goto have_python
echo Missing restirgs Python: %RESTIRGS_ENV%\python.exe
exit /b 1

:have_python

set "CUDA_HOME=%RESTIRGS_ENV%"
set "CUDA_PATH=%RESTIRGS_ENV%"
set "PATH=%RESTIRGS_ENV%;%RESTIRGS_ENV%\Library\bin;%RESTIRGS_ENV%\bin;%RESTIRGS_ENV%\Scripts;%PATH%"
set "TORCH_CUDA_ARCH_LIST=8.9"
set "MAX_JOBS=4"
if not defined TORCH_EXTENSIONS_DIR set "TORCH_EXTENSIONS_DIR=C:\tmp\torch_extensions_restirgs_cu124_patched"
if not defined MPLCONFIGDIR set "MPLCONFIGDIR=%CD%\outputs\matplotlib_cache"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"

"%RESTIRGS_ENV%\python.exe" scripts\patch_gsplat_windows.py --check
if not errorlevel 1 goto patch_ok
echo gsplat Windows patch is missing. Run: "%RESTIRGS_ENV%\python.exe" scripts\patch_gsplat_windows.py
exit /b 1

:patch_ok
exit /b 0
