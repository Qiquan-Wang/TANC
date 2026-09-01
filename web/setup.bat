@echo off
REM One-time setup: create <repo>\.venv with every dependency, then run web\serve.bat
setlocal
set "HERE=%~dp0"
set "REPO=%HERE%.."
set "VENV=%REPO%\.venv"
if "%PYTHON%"=="" set "PYTHON=python"

echo creating virtual-env at %VENV%  (base: %PYTHON%)
"%PYTHON%" -m venv "%VENV%"
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip wheel
echo installing tanc + all dependencies from pyproject.toml (this can take a few minutes)...
"%VENV%\Scripts\python.exe" -m pip install -e "%REPO%"

echo.
echo Done. Launch the builder with:   web\serve.bat
echo   then open http://localhost:8000
