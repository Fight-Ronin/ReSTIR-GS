@echo off
setlocal

set "REPO_ROOT=%~dp0..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

if "%CONDA_PREFIX%"=="" set "CONDA_PREFIX=C:\Users\yinha\miniconda3\envs\gs_gen"

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %ERRORLEVEL%

set "PYTHONIOENCODING=utf-8"
set "CUDA_HOME=%CONDA_PREFIX%"
set "CUDA_PATH=%CONDA_PREFIX%"
set "TORCH_HOME=%REPO_ROOT%\outputs\gsgen\torch_cache"
set "TORCH_EXTENSIONS_DIR=%REPO_ROOT%\outputs\gsgen\torch_extensions"
set "NVCC_PREPEND_FLAGS=-allow-unsupported-compiler"

%*
exit /b %ERRORLEVEL%
