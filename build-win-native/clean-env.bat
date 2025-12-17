@echo off
REM File: clean-env.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0

setlocal

echo === Cleaning environment files ===

REM Get the directory where this script is located
set BUILD_WIN=%~dp0
set BUILD_WIN=%BUILD_WIN:~0,-1%

echo.
echo Removing embedded Python...
if exist "%BUILD_WIN%\python" rmdir /S /Q "%BUILD_WIN%\python"

echo Removing local Rust installation...
if exist "%BUILD_WIN%\rust" rmdir /S /Q "%BUILD_WIN%\rust"

echo Removing dependencies...
if exist "%BUILD_WIN%\deps" rmdir /S /Q "%BUILD_WIN%\deps"

echo Removing downloads...
if exist "%BUILD_WIN%\downloads" rmdir /S /Q "%BUILD_WIN%\downloads"

echo Removing simulation data...
if exist "%BUILD_WIN%\ppf-cts" rmdir /S /Q "%BUILD_WIN%\ppf-cts"

echo Removing log files...
if exist "%BUILD_WIN%\warmup.log" del /Q "%BUILD_WIN%\warmup.log"
if exist "%BUILD_WIN%\build.log" del /Q "%BUILD_WIN%\build.log"

echo.
echo === Clean complete ===

endlocal

REM Skip pause if /nopause argument is provided (for automation)
echo %* | find /i "/nopause" >nul
if errorlevel 1 (
    echo Press any key to exit...
    pause >nul
)
