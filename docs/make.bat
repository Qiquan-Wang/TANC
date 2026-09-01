@ECHO OFF
REM Windows equivalent of docs/Makefile. Usage:  make.bat html
pushd %~dp0

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help
if "%1" == "clean" goto clean

%SPHINXBUILD% -b %1 %SOURCEDIR% %BUILDDIR%\%1 %SPHINXOPTS%
echo.
echo Build finished. Output is in %BUILDDIR%\%1.
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:clean
if exist %BUILDDIR% rmdir /S /Q %BUILDDIR%
if exist _generated rmdir /S /Q _generated
goto end

:end
popd