@echo off
REM File: scripts/fetch-download.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0
REM
REM Places one manifest entry's file in downloads\ and checks it. Use as:
REM
REM     call "%BUILD_WIN%\scripts\fetch-download.bat" PYTHON
REM     if errorlevel 1 exit /b 1
REM
REM The argument is an entry's BASE name: URL_<BASE> and FILE_<BASE> must be set
REM (load-downloads.bat and platform.bat set them), and SHA256_<BASE> is checked
REM whenever it is set. After the call DOWNLOADED_FILE names the file.
REM
REM THE ENTRY IS PASSED BY NAME, NEVER BY VALUE, and that is load-bearing. `call`
REM percent-expands its argument list a second time, so a URL passed as an argument
REM loses every `%` escape it carries: python-build-standalone names its archives
REM with a `+`, which the manifest spells `%2B`, and a second expansion turns `%2`
REM into the second argument of whatever script is running. The URL would change
REM with nothing reporting it. Reading the variable here, by name, involves no
REM second expansion.
REM
REM A FILE ALREADY IN downloads\ IS CHECKED, NOT TRUSTED. With a digest it must
REM match, and a mismatch deletes it and fetches again rather than installing
REM whatever an interrupted run left. The fetch writes to a .part file and renames
REM only after the digest matches, so a partial download is never taken for a
REM complete one. An entry with no digest is used as it is when present.
REM
REM NO LABELS. cmd finds a `goto` or `call :label` target by scanning the file,
REM and in a file checked out with LF line endings that scan can miss a label that
REM exists (build-cuda.bat records the same hazard). The digest is therefore
REM compared by one PowerShell expression whose exit code is the verdict, and the
REM already-present case is a flag rather than a jump.

setlocal enabledelayedexpansion

set "_BASE=%~1"
if not defined _BASE (
    echo ERROR: scripts\fetch-download.bat needs an entry name, for example PYTHON
    exit /b 1
)
set "_URL=!URL_%_BASE%!"
set "_FILE=!FILE_%_BASE%!"
set "_SHA=!SHA256_%_BASE%!"
if not defined _URL (
    echo ERROR: URL_%_BASE% is not set. Load scripts\downloads.txt and call platform.bat first.
    exit /b 1
)
if not defined _FILE (
    echo ERROR: FILE_%_BASE% is not set. Load scripts\downloads.txt and call platform.bat first.
    exit /b 1
)
set "_DIR=%~dp0..\downloads"
for %%I in ("!_DIR!") do set "_DIR=%%~fI"
if not exist "!_DIR!" mkdir "!_DIR!"
set "_PATH=!_DIR!\!_FILE!"
REM The comparison, reused for the file in place and for a fresh download. Exit 0
REM when the SHA256 equals _SHA, compared without case; exit 1 otherwise, printing
REM the digest it computed. PowerShell's Get-FileHash rather than certutil, whose
REM output format has changed between Windows releases.
set "_CHECK=$h = (Get-FileHash -Algorithm SHA256 -LiteralPath $env:_CHECK_FILE).Hash; if ($h -ieq $env:_SHA) { exit 0 }; Write-Output ('  SHA256 of ' + $env:_CHECK_FILE + ' is ' + $h); exit 1"

set "_HAVE="
if exist "!_PATH!" (
    if not defined _SHA (
        echo   !_FILE! is already in downloads\
        set "_HAVE=1"
    ) else (
        set "_CHECK_FILE=!_PATH!"
        powershell -NoProfile -Command "!_CHECK!"
        if errorlevel 1 (
            echo   !_FILE! in downloads\ does not match SHA256_%_BASE%; fetching it again
            del /q "!_PATH!"
        ) else (
            echo   !_FILE! is already in downloads\ and matches its SHA256
            set "_HAVE=1"
        )
    )
)

if not defined _HAVE (
    echo   Downloading !_FILE!
    echo     from !_URL!
    if exist "!_PATH!.part" del /q "!_PATH!.part"
    curl.exe -fL --retry 3 --retry-delay 2 --connect-timeout 15 -o "!_PATH!.part" "!_URL!"
    if errorlevel 1 (
        echo ERROR: could not download !_FILE! from URL_%_BASE%
        if exist "!_PATH!.part" del /q "!_PATH!.part"
        exit /b 1
    )
    if defined _SHA (
        set "_CHECK_FILE=!_PATH!.part"
        powershell -NoProfile -Command "!_CHECK!"
        if errorlevel 1 (
            echo ERROR: the download of !_FILE! does not match SHA256_%_BASE%.
            echo        expected !_SHA!
            echo        The file is left at !_PATH!.part for inspection and is not used.
            exit /b 1
        )
    )
    move /y "!_PATH!.part" "!_PATH!" >nul
    if errorlevel 1 (
        echo ERROR: could not rename !_PATH!.part into place
        exit /b 1
    )
    echo   [OK] !_FILE!
)

endlocal & set "DOWNLOADED_FILE=%_PATH%"
exit /b 0
