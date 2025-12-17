@echo off
REM File: enable_long_path.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0

echo === Enabling Windows Long Path Support ===
echo.

REM Check if running as administrator
net session >nul 2>&1
if errorlevel 1 (
    echo ERROR: This script requires administrator privileges.
    echo Please right-click and select "Run as administrator".
    exit /b 1
)

REM Enable Long Path support
powershell -Command "Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1"
if errorlevel 1 (
    echo ERROR: Failed to set registry value.
    exit /b 1
)

REM Verify it was set
for /f %%i in ('powershell -Command "Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled | Select-Object -ExpandProperty LongPathsEnabled"') do set LONGPATH_VALUE=%%i

if "%LONGPATH_VALUE%"=="1" (
    echo SUCCESS: Windows Long Path support has been enabled.
    echo.
    echo IMPORTANT: You must REBOOT your system for this change to take effect.
) else (
    echo ERROR: Failed to enable Long Path support.
    REM Skip pause if /nopause argument is provided (for automation)
    echo %* | find /i "/nopause" >nul
    if errorlevel 1 (
        echo Press any key to exit...
        pause >nul
    )
    exit /b 1
)

REM Skip pause if /nopause argument is provided (for automation)
echo %* | find /i "/nopause" >nul
if errorlevel 1 (
    echo Press any key to exit...
    pause >nul
)
