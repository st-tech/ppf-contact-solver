@echo off
REM File: fast-check-all.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0

setlocal enabledelayedexpansion

REM Check for /nopause argument
set NOPAUSE=0
echo %* | find /i "/nopause" >nul
if not errorlevel 1 set NOPAUSE=1

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

REM Detect if running from source tree or bundled distribution
REM In source: script is in build-win-native/, parent has frontend/ folder
REM In bundle: script is in dist root, examples/ is in same directory
set BUILD_WIN=%SCRIPT_DIR%
for %%I in ("%BUILD_WIN%\..") do set SRC=%%~fI

if exist "%SRC%\frontend" (
    REM Source tree - script is in build-win-native/
    set PYTHON_DIR=%BUILD_WIN%\python
    set EXAMPLES_DIR=%SRC%\examples
    set FAST_CHECK_DIR=%BUILD_WIN%\fast_check
    set INJECT_SCRIPT=%BUILD_WIN%\inject_fast_check.py
    set BIN_DIR=%SRC%\src\cpp\build\lib
    set MINGIT_DIR=%BUILD_WIN%\mingit\cmd
    set PYTHONPATH=%SRC%
    REM Find examples.txt - first try source location, then local directory
    set EXAMPLES_TXT=%SRC%\.github\workflows\scripts\examples.txt
    if not exist "!EXAMPLES_TXT!" (
        set EXAMPLES_TXT=%BUILD_WIN%\examples.txt
    )
) else (
    REM Bundled distribution - script is in dist root
    set DIST=%SCRIPT_DIR%
    set PYTHON_DIR=!DIST!\python
    set EXAMPLES_DIR=!DIST!\examples
    set EXAMPLES_TXT=!DIST!\examples.txt
    set FAST_CHECK_DIR=!DIST!\fast_check
    set INJECT_SCRIPT=!DIST!\inject_fast_check.py
    set BIN_DIR=!DIST!\bin
    set MINGIT_DIR=!DIST!\mingit\cmd
    set PYTHONPATH=!DIST!
)

set PYTHON=%PYTHON_DIR%\python.exe
REM Use local CUDA (in source tree: build-win-native\cuda, in bundle: bin\ has the DLLs)
if exist "%BUILD_WIN%\cuda\bin" (
    set CUDA_PATH=%BUILD_WIN%\cuda
) else (
    REM In bundled dist, CUDA DLLs are in bin\
    set CUDA_PATH=
)
set PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%BIN_DIR%;%MINGIT_DIR%;%CUDA_PATH%\bin;%PATH%

REM Clear caches before running tests
call "%SCRIPT_DIR%\clear-cache.bat" /nopause
echo.

REM Verify examples.txt exists
if not exist "%EXAMPLES_TXT%" (
    echo ERROR: examples.txt not found at %EXAMPLES_TXT%
    goto :error
)

REM Check if Python exists
if not exist "%PYTHON%" (
    echo ERROR: Embedded Python not found at %PYTHON%
    echo Please run warmup.bat first to set up the environment.
    goto :error
)

REM ============================================================
REM Run Unit Tests (equivalent to warmup.py run_tests)
REM ============================================================
echo === Running Unit Tests ===
echo.
"%PYTHON%" -c "from frontend.tests._runner_ import run_all_tests; import sys; sys.exit(0 if run_all_tests() else 1)"
if errorlevel 1 (
    echo.
    echo === UNIT TESTS FAILED ===
    goto :error
)
echo.
echo === Unit Tests Passed ===
echo.

echo === Fast Check All Examples ===
echo Using: %EXAMPLES_TXT%
echo.

REM Create fast_check directory
if exist "%FAST_CHECK_DIR%" rmdir /s /q "%FAST_CHECK_DIR%"
mkdir "%FAST_CHECK_DIR%"

REM Track results
set PASSED=0
set FAILED=0
set FAILED_LIST=
set PASSED_LIST=
set TOTAL=0
set CURRENT=0

REM Count total tests first
for /f "usebackq tokens=* eol=#" %%n in ("%EXAMPLES_TXT%") do (
    set NOTEBOOK=%%n
    if not "!NOTEBOOK!"=="" (
        set /a TOTAL+=1
    )
)

REM Set up log file
set LOG_FILE=%SCRIPT_DIR%\fast-check-results.log
echo === Fast Check Results === > "%LOG_FILE%"
echo Started: %DATE% %TIME% >> "%LOG_FILE%"
echo Total tests: !TOTAL! >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

REM Convert and run each notebook from examples.txt
echo Converting notebooks to Python and running tests...
echo Total tests to run: !TOTAL!
echo Log file: %LOG_FILE%
echo.

for /f "usebackq tokens=* eol=#" %%n in ("%EXAMPLES_TXT%") do (
    set NOTEBOOK=%%n
    REM Skip empty lines
    if not "!NOTEBOOK!"=="" (
        set /a CURRENT+=1
        set NOTEBOOK_PATH=%EXAMPLES_DIR%\!NOTEBOOK!.ipynb

        if exist "!NOTEBOOK_PATH!" (
            echo [TEST !CURRENT!/!TOTAL!] !NOTEBOOK!
            echo [TEST !CURRENT!/!TOTAL!] !NOTEBOOK! >> "%LOG_FILE%"

            REM Convert notebook to Python (use nbconvert directly, not via jupyter, to avoid hardcoded shebang paths)
            "%PYTHON%" -m nbconvert --to script "!NOTEBOOK_PATH!" --output-dir "%FAST_CHECK_DIR%"
            set CONVERT_ERR=!ERRORLEVEL!

            if !CONVERT_ERR! NEQ 0 (
                echo.
                echo === FAILED: !NOTEBOOK! [nbconvert error code: !CONVERT_ERR!] ===
                goto :error
            )

            if exist "%FAST_CHECK_DIR%\!NOTEBOOK!.py" (
                REM Insert App.set_fast_check() after "app = App" line
                "%PYTHON%" "%INJECT_SCRIPT%" "%FAST_CHECK_DIR%\!NOTEBOOK!.py"

                REM Run the Python script with output to terminal
                cd /d "%EXAMPLES_DIR%"
                "%PYTHON%" "%FAST_CHECK_DIR%\!NOTEBOOK!.py"

                if !ERRORLEVEL! EQU 0 (
                    echo        PASSED
                    echo        PASSED >> "%LOG_FILE%"
                    set /a PASSED+=1
                    set "PASSED_LIST=!PASSED_LIST! !NOTEBOOK!"
                ) else (
                    echo.
                    echo === FAILED: !NOTEBOOK! ===
                    echo        FAILED >> "%LOG_FILE%"
                    goto :error
                )
            ) else (
                echo.
                echo === FAILED: !NOTEBOOK! [conversion error] ===
                goto :error
            )
        ) else (
            echo [SKIP] !NOTEBOOK! - notebook not found
        )
    )
)

REM All tests passed - cleanup
echo.
echo === Cleaning up ===
if exist "%FAST_CHECK_DIR%" rmdir /s /q "%FAST_CHECK_DIR%"
call "%SCRIPT_DIR%\clear-cache.bat" /nopause

echo.
echo ============================================
echo === ALL !PASSED!/!TOTAL! TESTS PASSED ===
echo ============================================
echo.
echo Passed tests:
echo. >> "%LOG_FILE%"
echo ============================================ >> "%LOG_FILE%"
echo === ALL !PASSED!/!TOTAL! TESTS PASSED === >> "%LOG_FILE%"
echo ============================================ >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"
echo Passed tests: >> "%LOG_FILE%"
set TEST_NUM=0
for %%t in (!PASSED_LIST!) do (
    set /a TEST_NUM+=1
    echo   !TEST_NUM!. %%t
    echo   !TEST_NUM!. %%t >> "%LOG_FILE%"
)
echo.
echo Completed: %DATE% %TIME% >> "%LOG_FILE%"
echo Log saved to: %LOG_FILE%
echo.

if "%NOPAUSE%"=="0" (
    echo Press any key to exit...
    pause >nul
)

endlocal & exit /b 0

:error
REM Cleanup on error too
if exist "%FAST_CHECK_DIR%" rmdir /s /q "%FAST_CHECK_DIR%"
call "%SCRIPT_DIR%\clear-cache.bat" /nopause
if "%NOPAUSE%"=="0" (
    echo.
    echo Press any key to exit...
    pause >nul
)
endlocal & exit /b 1
