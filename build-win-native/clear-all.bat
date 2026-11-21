@echo off
REM File: clear-all.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0
REM
REM Full cleanup including all ppf-cts directories.
REM Used before creating release bundles or Docker images.
REM
REM THE FOLDER THIS RUNS FROM MAY BE NAMED WITH ANY CHARACTER A WINDOWS FILE NAME
REM CAN HOLD, such as a space, parentheses, an ampersand, an apostrophe, a
REM percent sign, a caret or an exclamation mark. So delayed expansion stays off,
REM since it would remove an exclamation mark from the path; every path is set
REM with the quoted assignment form and used inside quotes, where cmd takes those
REM characters literally; and each folder is handled through a FOR variable,
REM which cmd never expands a second time, so a percent sign in it survives.

setlocal DisableDelayedExpansion

REM Check for /nopause argument
set "NOPAUSE=0"
echo %* | find /i "/nopause" >nul
if not errorlevel 1 set "NOPAUSE=1"

echo === Clearing All (Full Cleanup) ===
echo.

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "HAS_ERROR=0"

REM Detect if running from source tree or bundled distribution: in a source
REM tree this script is in build-win-native\ and the project root is its
REM parent, which holds frontend\; in a bundle the project root is this folder.
set "PROJ_ROOT=%SCRIPT_DIR%"
for %%I in ("%SCRIPT_DIR%\..") do if exist "%%~fI\frontend" set "PROJ_ROOT=%%~fI"

REM Clear fast_check directory (created by fast-check-all.bat)
for %%T in ("%SCRIPT_DIR%\fast_check") do (
    if exist "%%~T" (
        echo Removing fast_check directory...
        rmdir /s /q "%%~T"
        if exist "%%~T" (
            echo   [FAIL] Could not remove "%%~T"
            set "HAS_ERROR=1"
        ) else (
            echo   [OK] Removed "%%~T"
        )
    ) else (
        echo   [SKIP] fast_check directory not found
    )
)

REM Clear project-relative cache directory
for %%T in ("%PROJ_ROOT%\cache\ppf-cts") do (
    if exist "%%~T" (
        echo Removing cache directory...
        rmdir /s /q "%%~T"
        if exist "%%~T" (
            echo   [FAIL] Could not remove "%%~T"
            set "HAS_ERROR=1"
        ) else (
            echo   [OK] Removed "%%~T"
        )
    ) else (
        echo   [SKIP] Cache directory not found
    )
)

REM Clear project-relative local directory (local/share/ppf-cts)
for %%T in ("%PROJ_ROOT%\local\share\ppf-cts") do (
    if exist "%%~T" (
        echo Removing local session data...
        rmdir /s /q "%%~T"
        if exist "%%~T" (
            echo   [FAIL] Could not remove "%%~T"
            set "HAS_ERROR=1"
        ) else (
            echo   [OK] Removed "%%~T"
        )
    ) else (
        echo   [SKIP] Local session data not found
    )
)

REM Clear jupyter directory (in build-win-native or dist)
for %%T in ("%SCRIPT_DIR%\jupyter") do (
    if exist "%%~T" (
        echo Removing Jupyter config/data...
        rmdir /s /q "%%~T"
        if exist "%%~T" (
            echo   [FAIL] Could not remove "%%~T"
            set "HAS_ERROR=1"
        ) else (
            echo   [OK] Removed "%%~T"
        )
    ) else (
        echo   [SKIP] Jupyter directory not found
    )
)

echo.
if "%HAS_ERROR%"=="1" (
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
