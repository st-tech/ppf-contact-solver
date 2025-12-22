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

echo Removing portable MSVC...
if exist "%BUILD_WIN%\msvc" rmdir /S /Q "%BUILD_WIN%\msvc"

echo Removing portable CUDA...
if exist "%BUILD_WIN%\cuda" rmdir /S /Q "%BUILD_WIN%\cuda"
if exist "%BUILD_WIN%\cuda_temp" rmdir /S /Q "%BUILD_WIN%\cuda_temp"

echo Removing portable 7-Zip...
if exist "%BUILD_WIN%\7zip" rmdir /S /Q "%BUILD_WIN%\7zip"

echo Removing MinGit...
if exist "%BUILD_WIN%\mingit" rmdir /S /Q "%BUILD_WIN%\mingit"

echo Removing MSYS2...
if exist "%BUILD_WIN%\msys64" rmdir /S /Q "%BUILD_WIN%\msys64"

echo Removing ffmpeg...
if exist "%BUILD_WIN%\ffmpeg" rmdir /S /Q "%BUILD_WIN%\ffmpeg"
if exist "%BUILD_WIN%\temp_ffmpeg" rmdir /S /Q "%BUILD_WIN%\temp_ffmpeg"

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
