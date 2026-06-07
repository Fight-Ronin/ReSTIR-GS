@echo off
setlocal enabledelayedexpansion

set "COLMAP_EXE=%~dp0colmap-4.0.4-nocuda\bin\colmap.exe"
set "ARGS="

:next_arg
if "%~1"=="" goto run_colmap
set "ARG=%~1"

if "!ARG!"=="--SiftExtraction.use_gpu" set "ARG=--FeatureExtraction.use_gpu"
if "!ARG!"=="--SiftExtraction.gpu_index" set "ARG=--FeatureExtraction.gpu_index"
if "!ARG!"=="--SiftMatching.use_gpu" set "ARG=--FeatureMatching.use_gpu"
if "!ARG!"=="--SiftMatching.gpu_index" set "ARG=--FeatureMatching.gpu_index"

set "ARGS=!ARGS! "!ARG!""
shift
goto next_arg

:run_colmap
"%COLMAP_EXE%" %ARGS%
exit /b %ERRORLEVEL%
