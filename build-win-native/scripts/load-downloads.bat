@echo off
REM File: scripts/load-downloads.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0
REM
REM Reads scripts\downloads.txt and exports every KEY=VALUE pair into the
REM caller's environment. Use as:
REM
REM     call "%BUILD_WIN%\scripts\load-downloads.bat"
REM
REM After the call, URL_*/FILE_* variables are available to the caller.

set "_LOAD_DOWNLOADS_TXT=%~dp0downloads.txt"
if not exist "%_LOAD_DOWNLOADS_TXT%" (
    echo ERROR: Manifest not found: %_LOAD_DOWNLOADS_TXT%
    set "_LOAD_DOWNLOADS_TXT="
    exit /b 1
)

for /f "usebackq eol=# tokens=1* delims==" %%K in ("%_LOAD_DOWNLOADS_TXT%") do (
    if not "%%K"=="" set "%%K=%%L"
)

set "_LOAD_DOWNLOADS_TXT="
exit /b 0
