@echo off
REM File: fast-check-all.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0
REM
REM Runs fast_check_all.py, which sits beside this file, with the embedded Python
REM beside it too: build-win-native\python in a source tree, python\ in a
REM distribution. That script runs the unit tests, the fail-examples and every
REM example in fast-check mode, and writes fast-check-results.log here.
REM
REM THIS FILE ONLY STARTS IT, because the folder may be named with any character
REM a Windows file name can hold, such as a space, parentheses, an ampersand, an
REM apostrophe, a percent sign, a caret or an exclamation mark, and the batch
REM version mishandled several of them. Here the path appears only through the
REM script's own location, inside quotes, with delayed expansion off, where cmd
REM takes every one of those characters literally.

setlocal DisableDelayedExpansion

if not exist "%~dp0python\python.exe" (
    echo ERROR: Embedded Python not found at "%~dp0python\python.exe"
    echo Please run warmup.bat first to set up the environment.
    echo %* | find /i "/nopause" >nul || pause
    exit /b 1
)
"%~dp0python\python.exe" "%~dp0fast_check_all.py" %*
exit /b %ERRORLEVEL%
