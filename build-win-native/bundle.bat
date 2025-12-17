@echo off
REM File: bundle.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0

setlocal enabledelayedexpansion

REM Check for /nopause argument
set NOPAUSE=0
echo %* | find /i "/nopause" >nul
if not errorlevel 1 set NOPAUSE=1

echo ============================================================
echo   ZOZO's Contact Solver - Bundle for Distribution
echo ============================================================
echo.

REM Get the directory where this script is located
set BUILD_WIN=%~dp0
set BUILD_WIN=%BUILD_WIN:~0,-1%

REM Get parent directory (SRC_DIR)
for %%I in ("%BUILD_WIN%\..") do set SRC_DIR=%%~fI

set DIST_DIR=%BUILD_WIN%\dist
set BIN_DIR=%DIST_DIR%\bin
set TARGET_DIR=%DIST_DIR%\target\release

if not defined CUDA_PATH set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

REM ============================================================
REM Verify build exists
REM ============================================================
echo [1/9] Verifying build...

set RUST_EXE=%SRC_DIR%\target\release\ppf-contact-solver.exe
set CUDA_DLL=%SRC_DIR%\src\cpp\build\lib\libsimbackend_cuda.dll

if not exist "%RUST_EXE%" (
    echo ERROR: ppf-contact-solver.exe not found. Please run build.bat first.
    exit /b 1
)
if not exist "%CUDA_DLL%" (
    echo ERROR: libsimbackend_cuda.dll not found. Please run build.bat first.
    exit /b 1
)
if not exist "%BUILD_WIN%\python\python.exe" (
    echo ERROR: Embedded Python not found. Please run warmup.bat first.
    exit /b 1
)
echo   [OK] Build verified

REM ============================================================
REM Create dist directory
REM ============================================================
echo.
echo [2/9] Creating dist directory...

if exist "%DIST_DIR%" (
    echo   Removing old dist folder...
    rmdir /s /q "%DIST_DIR%"
)
mkdir "%DIST_DIR%"
mkdir "%BIN_DIR%"
mkdir "%DIST_DIR%\target"
mkdir "%TARGET_DIR%"
echo   [OK] Created %DIST_DIR%

REM ============================================================
REM Copy application binaries
REM ============================================================
echo.
echo [3/9] Copying application binaries...

REM Copy Rust exe to target/release/ (where frontend expects it)
copy "%RUST_EXE%" "%TARGET_DIR%\" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy ppf-contact-solver.exe
    exit /b 1
)
echo   Copied ppf-contact-solver.exe to target/release/

REM Copy CUDA backend DLL to bin/ (loaded via PATH)
copy "%CUDA_DLL%" "%BIN_DIR%\" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy libsimbackend_cuda.dll
    exit /b 1
)
echo   Copied libsimbackend_cuda.dll to bin/

REM ============================================================
REM Copy CUDA redistributable DLLs
REM ============================================================
echo.
echo [4/9] Copying CUDA runtime libraries...

set CUDA_BIN=%CUDA_PATH%\bin

REM List of redistributable CUDA DLLs (only cudart needed)
set CUDA_DLLS=cudart64_12.dll

for %%D in (%CUDA_DLLS%) do (
    if exist "%CUDA_BIN%\%%D" (
        copy "%CUDA_BIN%\%%D" "%BIN_DIR%\" >nul
        if errorlevel 1 (
            echo ERROR: Failed to copy %%D
            exit /b 1
        )
        echo   Copied %%D
    ) else (
        echo   WARNING: %%D not found in %CUDA_BIN%
    )
)

REM ============================================================
REM Copy Python environment (exclude caches)
REM ============================================================
echo.
echo [5/9] Copying Python environment...

REM Use robocopy to exclude __pycache__, .pyc files, backup files, and galata (test framework)
REM Note: Cannot exclude 'tests' as numpy._core.tests is required at runtime
REM Note: Cannot exclude '.pyi' as skimage uses stub files for lazy loading
robocopy "%BUILD_WIN%\python" "%DIST_DIR%\python" /E /XD __pycache__ third_party galata /XF *.pyc *.pyo *.orig /NJH /NJS /NDL /NC /NS /NP >nul
REM robocopy returns 0-7 for success, 8+ for errors
if errorlevel 8 (
    echo ERROR: Failed to copy Python environment
    exit /b 1
)
echo   [OK] Copied Python environment

REM ============================================================
REM Shorten webpack chunk IDs to reduce path length
REM ============================================================
echo.
echo [5.5/9] Shortening webpack chunk IDs...

"%BUILD_WIN%\python\python.exe" "%BUILD_WIN%\shorten_webpack_chunks.py" "%DIST_DIR%"
if errorlevel 1 (
    echo WARNING: Chunk shortening failed, continuing anyway...
)
echo   [OK] Chunk IDs shortened

REM ============================================================
REM Copy examples (exclude caches and checkpoints)
REM ============================================================
echo.
echo [6/9] Copying examples...

if exist "%SRC_DIR%\examples" (
    robocopy "%SRC_DIR%\examples" "%DIST_DIR%\examples" /E /XD __pycache__ .ipynb_checkpoints /XF *.pyc *.pyo /NJH /NJS /NDL /NC /NS /NP >nul
    if errorlevel 8 (
        echo ERROR: Failed to copy examples
        exit /b 1
    )
    echo   [OK] Copied examples
) else (
    echo   WARNING: examples folder not found, skipping
)

REM ============================================================
REM Copy frontend module (exclude caches)
REM ============================================================
echo.
echo [7/9] Copying frontend module...

if exist "%SRC_DIR%\frontend" (
    robocopy "%SRC_DIR%\frontend" "%DIST_DIR%\frontend" /E /XD __pycache__ /XF *.pyc *.pyo /NJH /NJS /NDL /NC /NS /NP >nul
    if errorlevel 8 (
        echo ERROR: Failed to copy frontend
        exit /b 1
    )
    echo   [OK] Copied frontend
) else (
    echo   WARNING: frontend folder not found, skipping
)

REM ============================================================
REM Copy src Python module (exclude cpp build artifacts and caches)
REM ============================================================
echo.
echo [8/9] Copying src module...

if exist "%SRC_DIR%\src" (
    REM Copy src directory excluding build artifacts and caches
    robocopy "%SRC_DIR%\src" "%DIST_DIR%\src" /E /XD __pycache__ build obj tests /XF *.pyc *.pyo *.obj *.lib *.exp /NJH /NJS /NDL /NC /NS /NP >nul
    if errorlevel 8 (
        echo ERROR: Failed to copy src
        exit /b 1
    )
    echo   [OK] Copied src
) else (
    echo   WARNING: src folder not found, skipping
)

REM ============================================================
REM Generate launcher and documentation
REM ============================================================
echo.
echo [9/9] Generating launcher and documentation...

REM Generate start.bat (self-contained launcher)
(
echo @echo off
echo setlocal
echo.
echo REM Get the directory where this script is located
echo set DIST=%%~dp0
echo set DIST=%%DIST:~0,-1%%
echo.
echo REM Load configuration
echo call "%%DIST%%\config.bat"
echo.
echo REM Set PATH to include bundled binaries
echo set PATH=%%DIST%%\bin;%%PATH%%
echo set PYTHONPATH=%%DIST%%;%%PYTHONPATH%%
echo.
echo REM Start JupyterLab - auto-increments port if taken
echo "%%DIST%%\python\python.exe" -m jupyterlab --no-browser --port=%%PORT%% --ServerApp.port_retries=50 --ServerApp.token="" --notebook-dir="%%DIST%%\examples"
echo.
echo REM Kill any remaining ppf-contact-solver processes when JupyterLab exits
echo taskkill /F /IM ppf-contact-solver.exe 2^>nul
echo endlocal
) > "%DIST_DIR%\start.bat"
echo   Created start.bat

REM Generate start-jupyterlab.pyw (GUI launcher)
(
echo import subprocess
echo import sys
echo import os
echo import webbrowser
echo import time
echo.
echo script_dir = os.path.dirname^(os.path.abspath^(__file__^)^)
echo python_exe = os.path.join^(script_dir, "python", "pythonw.exe"^)
echo bin_dir = os.path.join^(script_dir, "bin"^)
echo.
echo os.environ["PATH"] = bin_dir + ";" + os.environ.get^("PATH", ""^)
echo os.environ["PYTHONPATH"] = script_dir + ";" + os.environ.get^("PYTHONPATH", ""^)
echo.
echo proc = subprocess.Popen^([
echo     python_exe, "-m", "jupyterlab",
echo     "--no-browser", "--port=8080",
echo     "--ServerApp.token=",
echo     "--notebook-dir=" + os.path.join^(script_dir, "examples"^)
echo ], env=os.environ^)
echo.
echo time.sleep^(3^)
echo webbrowser.open^("http://localhost:8080"^)
) > "%DIST_DIR%\start-jupyterlab.pyw"
echo   Created start-jupyterlab.pyw

REM Generate headless.bat (run headless example without JupyterLab)
(
echo @echo off
echo setlocal
echo.
echo REM Check for /nopause argument
echo set NOPAUSE=0
echo echo %%* ^| find /i "/nopause" ^>nul
echo if not errorlevel 1 set NOPAUSE=1
echo.
echo REM Get the directory where this script is located
echo set DIST=%%~dp0
echo set DIST=%%DIST:~0,-1%%
echo.
echo REM Set PATH to include bundled binaries
echo set PATH=%%DIST%%\bin;%%PATH%%
echo set PYTHONPATH=%%DIST%%;%%PYTHONPATH%%
echo.
echo REM Run headless.py
echo "%%DIST%%\python\python.exe" "%%DIST%%\examples\headless.py"
echo set EXITCODE=%%ERRORLEVEL%%
echo.
echo if "%%NOPAUSE%%"=="0" ^(
echo     echo.
echo     echo Press any key to exit...
echo     pause ^>nul
echo ^)
echo.
echo endlocal ^& exit /b %%EXITCODE%%
) > "%DIST_DIR%\headless.bat"
echo   Created headless.bat

REM Update Python path configuration for dist
(
echo python311.zip
echo .
echo Lib\site-packages
echo ..
echo import site
) > "%DIST_DIR%\python\python311._pth"

REM Generate THIRD_PARTY_LICENSES.txt
> "%DIST_DIR%\THIRD_PARTY_LICENSES.txt" (
echo ============================================================
echo THIRD PARTY LICENSES
echo ============================================================
echo.
echo ZOZO's Contact Solver
echo -------------------
echo Copyright 2025 Ryoichi Ando ^(ZOZO, Inc.^)
echo Licensed under the Apache License, Version 2.0
echo.
echo.
echo NVIDIA CUDA Libraries
echo ---------------------
echo This software contains source code provided by NVIDIA Corporation.
echo.
echo CUDA Runtime Libraries ^(cudart, cublas, cublasLt, cusparse^) are
echo redistributed under the NVIDIA CUDA Toolkit End User License Agreement.
echo.
echo See: https://docs.nvidia.com/cuda/eula/index.html
echo.
echo.
echo Eigen
echo -----
echo Eigen is a C++ template library for linear algebra.
echo Licensed under the Mozilla Public License 2.0 ^(MPL2^).
echo.
echo See: https://eigen.tuxfamily.org/
echo.
echo.
echo Python
echo ------
echo Python is distributed under the Python Software Foundation License.
echo.
echo See: https://docs.python.org/3/license.html
echo.
)
echo   Created THIRD_PARTY_LICENSES.txt

REM Generate README.txt
> "%DIST_DIR%\README.txt" (
echo ============================================================
echo ZOZO's Contact Solver - Standalone Distribution
echo ============================================================
echo.
echo QUICK START
echo -----------
echo 1. Double-click "start.bat" to launch JupyterLab
echo 2. Open your browser to http://localhost:8080
echo 3. Navigate to the examples folder and run a notebook
echo.
echo.
echo CONFIGURATION
echo -------------
echo Edit "config.bat" to change the default port:
echo.
echo   set PORT=8080
echo.
echo If the port is already in use, JupyterLab will automatically
echo try the next available port.
echo.
echo.
echo HEADLESS MODE
echo -------------
echo Run the headless example without JupyterLab:
echo.
echo   headless.bat
echo.
echo.
echo REQUIREMENTS
echo ------------
echo - Windows 10/11 ^(64-bit^)
echo - NVIDIA GPU with CUDA support
echo - NVIDIA GPU driver installed
echo.
echo.
echo CONTENTS
echo --------
echo bin/           - Solver binaries and CUDA libraries
echo python/        - Embedded Python environment
echo examples/      - Example Jupyter notebooks
echo config.bat     - Port configuration
echo start.bat      - JupyterLab launcher
echo start-jupyterlab.pyw - GUI launcher
echo headless.bat   - Run examples without JupyterLab
echo.
echo.
echo LICENSE
echo -------
echo See THIRD_PARTY_LICENSES.txt for license information.
echo.
)
echo   Created README.txt

REM Copy config.bat configuration file
copy "%BUILD_WIN%\config.bat" "%DIST_DIR%\config.bat" >nul
echo   Copied config.bat

REM ============================================================
REM Summary
REM ============================================================
echo.
echo ============================================================
echo   BUNDLE COMPLETE!
echo ============================================================
echo.
echo Distribution created at: %DIST_DIR%
echo.
echo Contents:
echo   target/release/
for %%F in ("%TARGET_DIR%\*") do echo     %%~nxF
echo   bin/
for %%F in ("%BIN_DIR%\*") do echo     %%~nxF
echo   python/
echo   frontend/
echo   src/
echo   examples/
echo   start.bat
echo   start-jupyterlab.pyw
echo   headless.bat
echo   config.bat
echo   THIRD_PARTY_LICENSES.txt
echo   README.txt
echo.

REM Calculate total size
for /f "tokens=3" %%S in ('dir /s "%DIST_DIR%" 2^>nul ^| findstr /c:"File(s)"') do set TOTAL_SIZE=%%S
echo Total size: %TOTAL_SIZE% bytes
echo.
echo Ready for distribution!
echo.

if "%NOPAUSE%"=="0" pause
endlocal
