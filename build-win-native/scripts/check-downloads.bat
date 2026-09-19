@echo off
REM File: scripts/check-downloads.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0
REM
REM Verifies URL_* entries in scripts\downloads.txt are reachable. Run before
REM warmup.bat to catch broken upstream pointers; if a URL fails, edit the
REM offending line in scripts\downloads.txt and re-run this script.
REM
REM     scripts\check-downloads.bat                         every URL_* in the manifest
REM     scripts\check-downloads.bat URL_MINGIT_ARM64 URL_ROCM_SDK
REM                                                         only the named entries
REM
REM Naming entries is how warmup.bat probes only what it is about to fetch, for this
REM architecture and these backends: an entry whose tool is already installed, or
REM whose file is already in downloads\, needs no network. With no names every entry
REM is probed, every architecture's included, which is the check to run after
REM editing the manifest. /nopause may appear anywhere among the arguments.

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

REM The manifest's own values, so a named key can be checked against it.
call "%SCRIPT_DIR%\load-downloads.bat"
if errorlevel 1 exit /b 1

set "KEYS="
for %%A in (%*) do (
    if /i not "%%~A"=="/nopause" set "KEYS=!KEYS! %%~A"
)
if not defined KEYS (
    for /f "usebackq eol=# tokens=1 delims==" %%K in ("%MANIFEST%") do (
        set "name=%%K"
        if "!name:~0,4!"=="URL_" set "KEYS=!KEYS! %%K"
    )
)

set HAS_ERROR=0
set CHECKED=0
echo === Checking download URLs in %MANIFEST% ===
echo.

for %%K in (!KEYS!) do (
    set "name=%%K"
    if not "!name:~0,4!"=="URL_" (
        echo   [FAIL] %%K is not a URL_* key
        set HAS_ERROR=1
    ) else if not defined %%K (
        echo   [FAIL] %%K is not defined in %MANIFEST%
        set HAS_ERROR=1
    ) else (
        set /a CHECKED+=1
        REM BY NAME, for the reason scripts\fetch-download.bat gives: `call`
        REM expands its arguments again, which rewrites a URL carrying `%%2B`.
        call :probe %%K
    )
)

echo.
if "!CHECKED!"=="0" if "!HAS_ERROR!"=="0" (
    REM Nothing to probe would pass vacuously, which reads as a clean check and
    REM certifies nothing. warmup.bat does not call this when it needs no network.
    echo === [FAIL] no URL_* entries were named or defined, so nothing was checked ===
    set HAS_ERROR=1
)
if "!HAS_ERROR!"=="1" (
    echo === [FAIL] One or more URLs are not reachable ===
    echo Update the offending entries in %MANIFEST% and re-run.
    set EXIT_CODE=1
) else (
    echo === [OK] All !CHECKED! download URLs reachable ===
    set EXIT_CODE=0
)

if "%NOPAUSE%"=="0" (
    echo.
    echo Press any key to exit...
    pause >nul
)

endlocal & exit /b %EXIT_CODE%

:probe
REM %~1 = the manifest key. The URL is read here, by name.
REM Try HEAD first; some hosts (S3, signed URLs, certain CDNs) reject HEAD,
REM so fall back to a one-byte ranged GET that pulls almost no data.
set "url=!%~1!"
echo Checking %~1
echo   !url!
curl.exe -fsSLI %RETRY% --max-time 30 -o NUL -w "  HTTP %%{http_code}\n" "!url!" 2>nul
if not errorlevel 1 exit /b 0
curl.exe -fsSL %RETRY% --max-time 30 -r 0-0 -o NUL -w "  HTTP %%{http_code}\n" "!url!"
if errorlevel 1 (
    REM Report curl's own exit code, not just "unreachable". It is the only
    REM thing that distinguishes a dead pointer (22, an HTTP 4xx) from a
    REM transport failure (28 timeout, 7 refused, 6 DNS), and without it this
    REM verdict has to be diagnosed by inference from the surrounding log.
    echo   [FAIL] %~1 unreachable ^(curl exit !ERRORLEVEL!^): !url!
    set HAS_ERROR=1
)
exit /b 0
