@echo off
REM File: scripts/check-downloads.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0
REM
REM Verifies every URL_* in scripts\downloads.txt is reachable. Run before
REM warmup.bat to catch broken upstream pointers; if a URL fails, edit the
REM offending line in scripts\downloads.txt and re-run this script.

setlocal enabledelayedexpansion

REM Check for /nopause argument
set NOPAUSE=0
echo %* | find /i "/nopause" >nul
if not errorlevel 1 set NOPAUSE=1

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "MANIFEST=%SCRIPT_DIR%\downloads.txt"

if not exist "%MANIFEST%" (
    echo ERROR: Manifest not found: %MANIFEST%
    exit /b 1
)

REM Retry policy for the probes below. A single attempt is not a sound
REM reachability verdict: a rotted pointer and a dropped SYN are
REM indistinguishable from one sample, and warmup.bat makes this script's
REM verdict fatal before the first download, so one transient drop discards a
REM ~30 minute multi-GB setup. --connect-timeout bounds each attempt well
REM below the Windows SYN-retransmit budget (measured at 21.1 s on Server
REM 2025), which --max-time never reaches because no connection is ever
REM established. --retry alone already covers that timeout class, and
REM --retry-connrefused adds the RST that a rate-limiting host sends. Neither
REM retries an HTTP 4xx, so a genuinely dead pointer still fails on its first
REM attempt (measured: 1.1 s) rather than being retried into a slow abort.
set "RETRY=--retry 2 --retry-delay 1 --retry-connrefused --connect-timeout 8"

set HAS_ERROR=0
echo === Checking download URLs in %MANIFEST% ===
echo.

for /f "usebackq eol=# tokens=1* delims==" %%K in ("%MANIFEST%") do (
    set "name=%%K"
    set "value=%%L"
    if "!name:~0,4!"=="URL_" call :probe "!name!" "!value!"
)

echo.
if "!HAS_ERROR!"=="1" (
    echo === [FAIL] One or more URLs are not reachable ===
    echo Update the offending entries in %MANIFEST% and re-run.
    set EXIT_CODE=1
) else (
    echo === [OK] All download URLs reachable ===
    set EXIT_CODE=0
)

if "%NOPAUSE%"=="0" (
    echo.
    echo Press any key to exit...
    pause >nul
)

endlocal & exit /b %EXIT_CODE%

:probe
REM %~1 = name, %~2 = url
REM Try HEAD first; some hosts (S3, signed URLs, certain CDNs) reject HEAD,
REM so fall back to a one-byte ranged GET that pulls almost no data.
echo Checking %~1
curl.exe -fsSLI %RETRY% --max-time 30 -o NUL -w "  HTTP %%{http_code}  %~2\n" "%~2" 2>nul
if not errorlevel 1 exit /b 0
curl.exe -fsSL %RETRY% --max-time 30 -r 0-0 -o NUL -w "  HTTP %%{http_code}  %~2\n" "%~2"
if errorlevel 1 (
    REM Report curl's own exit code, not just "unreachable". It is the only
    REM thing that distinguishes a dead pointer (22, an HTTP 4xx) from a
    REM transport failure (28 timeout, 7 refused, 6 DNS), and without it this
    REM verdict has to be diagnosed by inference from the surrounding log.
    echo   [FAIL] %~1 unreachable ^(curl exit !ERRORLEVEL!^): %~2
    set HAS_ERROR=1
)
exit /b 0
