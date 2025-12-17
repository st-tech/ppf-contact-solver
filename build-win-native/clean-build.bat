@echo off
REM File: clean-build.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0

setlocal

echo === Cleaning build files ===

REM Get the directory where this script is located
set BUILD_WIN=%~dp0
set BUILD_WIN=%BUILD_WIN:~0,-1%
for %%I in ("%BUILD_WIN%\..") do set SRC=%%~fI

echo.
echo Removing MSBuild intermediate files...
if exist "%BUILD_WIN%\simbackend_cuda" rmdir /S /Q "%BUILD_WIN%\simbackend_cuda"

echo Removing C++/CUDA build output...
if exist "%SRC%\src\cpp\build" rmdir /S /Q "%SRC%\src\cpp\build"

echo Removing Rust build output...
if exist "%SRC%\target" rmdir /S /Q "%SRC%\target"

echo Removing distribution folder...
if exist "%BUILD_WIN%\dist" rmdir /S /Q "%BUILD_WIN%\dist"

echo Removing build log...
if exist "%BUILD_WIN%\build.log" del /Q "%BUILD_WIN%\build.log"

echo Removing launcher scripts...
if exist "%BUILD_WIN%\start.bat" del /Q "%BUILD_WIN%\start.bat"
if exist "%BUILD_WIN%\start-jupyterlab.pyw" del /Q "%BUILD_WIN%\start-jupyterlab.pyw"

echo.
echo === Clean complete ===

endlocal

REM Skip pause if /nopause argument is provided (for automation)
echo %* | find /i "/nopause" >nul
if errorlevel 1 (
    echo Press any key to exit...
    pause >nul
)
