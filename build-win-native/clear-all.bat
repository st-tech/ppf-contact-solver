@echo off
REM File: clear-all.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0
REM
REM Full cleanup including all ppf-cts directories.
REM Used before creating release bundles or Docker images.

setlocal enabledelayedexpansion

REM Check for /nopause argument
set NOPAUSE=0
echo %* | find /i "/nopause" >nul
if not errorlevel 1 set NOPAUSE=1

echo === Clearing All (Full Cleanup) ===
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

REM Track errors
set HAS_ERROR=0

REM Detect if running from source tree or bundled distribution
for %%I in ("%SCRIPT_DIR%\..") do set PARENT=%%~fI

if exist "%PARENT%\frontend" (
    REM Source tree - script is in build-win-native/, project root is PARENT
    set PROJ_ROOT=%PARENT%
    set FAST_CHECK_DIR=%SCRIPT_DIR%\fast_check
) else (
    REM Bundled distribution - script is in dist root
    set PROJ_ROOT=%SCRIPT_DIR%
    set FAST_CHECK_DIR=%SCRIPT_DIR%\fast_check
)

REM Clear fast_check directory
if exist "%FAST_CHECK_DIR%" (
    echo Removing fast_check directory...
    rmdir /s /q "%FAST_CHECK_DIR%"
    if exist "%FAST_CHECK_DIR%" (
        echo   [FAIL] Could not remove %FAST_CHECK_DIR%
        set HAS_ERROR=1
    ) else (
        echo   [OK] Removed %FAST_CHECK_DIR%
    )
) else (
    echo   [SKIP] fast_check directory not found
)

REM Clear project-relative cache directory (cache/ppf-cts)
set CACHE_DIR=!PROJ_ROOT!\cache\ppf-cts
if exist "!CACHE_DIR!" (
    echo Removing cache directory...
    rmdir /s /q "!CACHE_DIR!"
    if exist "!CACHE_DIR!" (
        echo   [FAIL] Could not remove !CACHE_DIR!
        set HAS_ERROR=1
    ) else (
        echo   [OK] Removed !CACHE_DIR!
    )
) else (
    echo   [SKIP] Cache directory not found
)

REM Clear project-relative local directory (local/share/ppf-cts)
set LOCAL_DIR=!PROJ_ROOT!\local\share\ppf-cts
if exist "!LOCAL_DIR!" (
    echo Removing local session data...
    rmdir /s /q "!LOCAL_DIR!"
    if exist "!LOCAL_DIR!" (
        echo   [FAIL] Could not remove !LOCAL_DIR!
        set HAS_ERROR=1
    ) else (
        echo   [OK] Removed !LOCAL_DIR!
    )
) else (
    echo   [SKIP] Local session data not found
)

REM Clear jupyter directory (in build-win-native or dist)
set JUPYTER_DIR=%SCRIPT_DIR%\jupyter
if exist "!JUPYTER_DIR!" (
    echo Removing Jupyter config/data...
    rmdir /s /q "!JUPYTER_DIR!"
    if exist "!JUPYTER_DIR!" (
        echo   [FAIL] Could not remove !JUPYTER_DIR!
        set HAS_ERROR=1
    ) else (
        echo   [OK] Removed !JUPYTER_DIR!
    )
) else (
    echo   [SKIP] Jupyter directory not found
)

echo.
if %HAS_ERROR%==1 (
    echo === [FAIL] Some directories could not be cleared ===
) else (
    echo === [SUCCESS] Full Cleanup Complete ===
)

if "%NOPAUSE%"=="0" (
    echo.
    echo Press any key to exit...
    pause >nul
)

endlocal
