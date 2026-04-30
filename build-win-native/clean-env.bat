@echo off
REM File: clean-env.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0

setlocal enabledelayedexpansion

echo === Cleaning environment files ===

REM Get the directory where this script is located
set BUILD_WIN=%~dp0
set BUILD_WIN=%BUILD_WIN:~0,-1%

set HAS_ERROR=0

REM Kill any process whose executable lives under %BUILD_WIN% so rmdir
REM does not silently fail with "Access is denied" on locked files
REM (e.g. orphaned git.exe from a prior interrupted git pull holding
REM mingit\cmd\git.exe and friends open).
echo.
echo Releasing file locks under %BUILD_WIN%...
powershell -NoProfile -Command "$base='%BUILD_WIN%'; Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath -like ($base + '\*') } | ForEach-Object { Write-Host ('  Killing PID ' + $_.ProcessId + ' (' + $_.Name + '): ' + $_.ExecutablePath); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

call :remove "embedded Python"        "%BUILD_WIN%\python"
call :remove "local Rust installation" "%BUILD_WIN%\rust"
call :remove "portable MSVC"          "%BUILD_WIN%\msvc"
call :remove "portable CUDA"          "%BUILD_WIN%\cuda"
call :remove "portable CUDA temp"     "%BUILD_WIN%\cuda_temp"
call :remove "portable 7-Zip"         "%BUILD_WIN%\7zip"
call :remove "MinGit"                 "%BUILD_WIN%\mingit"
call :remove "MSYS2"                  "%BUILD_WIN%\msys64"
call :remove "ffmpeg"                 "%BUILD_WIN%\ffmpeg"
call :remove "ffmpeg temp"            "%BUILD_WIN%\temp_ffmpeg"
call :remove "dependencies"           "%BUILD_WIN%\deps"
call :remove "downloads"              "%BUILD_WIN%\downloads"
call :remove "simulation data"        "%BUILD_WIN%\ppf-cts"

echo.
echo Removing log files...
if exist "%BUILD_WIN%\warmup.log" del /Q "%BUILD_WIN%\warmup.log" 2>nul
if exist "%BUILD_WIN%\build.log"  del /Q "%BUILD_WIN%\build.log"  2>nul

echo.
if "!HAS_ERROR!"=="1" (
    echo === [FAIL] Some directories could not be removed ===
    echo Inspect the [FAIL] lines above. The most common cause is a process
    echo still holding files open ^(check tasklist for git.exe, python.exe,
    echo cargo.exe, etc. running from %BUILD_WIN%^).
    set EXIT_CODE=1
) else (
    echo === Clean complete ===
    set EXIT_CODE=0
)

REM Skip pause if /nopause argument is provided (for automation)
echo %* | find /i "/nopause" >nul
if errorlevel 1 (
    echo Press any key to exit...
    pause >nul
)

endlocal & exit /b %EXIT_CODE%

:remove
REM %~1 = label, %~2 = path
if not exist "%~2" (
    echo [SKIP] %~1 not present
    exit /b 0
)
echo Removing %~1...
rmdir /S /Q "%~2" 2>nul
if exist "%~2" (
    echo   [FAIL] %~2 still present ^(likely locked by another process^)
    set HAS_ERROR=1
)
exit /b 0
