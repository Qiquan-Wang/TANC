@echo off
REM Launch the TANC Visual Builder with the project's virtual-env Python.
REM   web\serve.bat
setlocal
set "HERE=%~dp0"
set "REPO=%HERE%.."
set "PY="
if exist "%REPO%\.venv\Scripts\python.exe" set "PY=%REPO%\.venv\Scripts\python.exe"
if not defined PY if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "PY=%VIRTUAL_ENV%\Scripts\python.exe"
if not defined PY if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PY=%CONDA_PREFIX%\python.exe"
if not defined PY set "PY=python"
echo launching with: %PY%
"%PY%" "%HERE%server.py"
