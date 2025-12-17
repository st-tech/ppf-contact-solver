@echo off
REM File: build.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0

setlocal enabledelayedexpansion

REM Check for /nopause argument early (before re-launch)
REM Only check if NOPAUSE is not already set (from re-launch environment)
if not defined NOPAUSE (
    set NOPAUSE=0
    echo %* | find /i "/nopause" >nul
    if not errorlevel 1 set NOPAUSE=1
)

REM Get the directory where this script is located
set BUILD_WIN=%~dp0
REM Remove trailing backslash
set BUILD_WIN=%BUILD_WIN:~0,-1%
set LOGFILE=%BUILD_WIN%\build.log

REM If not already being logged, restart with logging
if "%BUILD_LOGGING%"=="" (
    set BUILD_LOGGING=1
    echo Logging to %LOGFILE%
    powershell -Command "& { cmd /c 'set BUILD_LOGGING=1&& set NOPAUSE=!NOPAUSE!&& \"%~f0\"' 2>&1 | Tee-Object -FilePath '%LOGFILE%' }"
    exit /b %ERRORLEVEL%
)

echo ============================================================
echo   ZOZO's Contact Solver - Windows Build Script
echo ============================================================
echo.
REM Get parent directory (SRC_DIR)
for %%I in ("%BUILD_WIN%\..") do set SRC_DIR=%%~fI

set CPP_DIR=%SRC_DIR%\src\cpp
set OUT_DIR=%CPP_DIR%\build
set LIB_DIR=%OUT_DIR%\lib
set DEPS=%BUILD_WIN%\deps
set DOWNLOADS=%BUILD_WIN%\downloads
set RUST_DIR=%BUILD_WIN%\rust

if not defined CUDA_PATH set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

REM Use local Rust if installed by warmup.bat
if exist "%RUST_DIR%\bin\cargo.exe" (
    echo Using local Rust from %RUST_DIR%
    set "PATH=%RUST_DIR%\bin;%PATH%"
    set "RUSTUP_HOME=%RUST_DIR%\rustup"
    set "CARGO_HOME=%RUST_DIR%"
) else (
    where cargo >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Rust not found. Please run warmup.bat first to install Rust.
        exit /b 1
    )
)

REM ============================================================
REM Download Eigen if not present
REM ============================================================
if not exist "%DEPS%\eigen-3.4.0" (
    echo [0/3] Downloading Eigen 3.4.0...
    if not exist "%DOWNLOADS%" mkdir "%DOWNLOADS%"
    if not exist "%DEPS%" mkdir "%DEPS%"

    set EIGEN_URL=https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    set EIGEN_ZIP=%DOWNLOADS%\eigen-3.4.0.zip

    if not exist "!EIGEN_ZIP!" (
        powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '!EIGEN_URL!' -OutFile '!EIGEN_ZIP!' -UseBasicParsing"
        if errorlevel 1 (
            echo ERROR: Failed to download Eigen
            exit /b 1
        )
    )

    echo Extracting Eigen...
    powershell -Command "Expand-Archive -Path '!EIGEN_ZIP!' -DestinationPath '%DEPS%' -Force"
    if errorlevel 1 (
        echo ERROR: Failed to extract Eigen
        exit /b 1
    )
    echo   [DONE] Eigen ready
    echo.
)

REM Setup Visual Studio environment
echo [1/4] Setting up Visual Studio environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
    echo ERROR: Failed to setup Visual Studio environment
    exit /b 1
)

REM Create output directories
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
if not exist "%OUT_DIR%\obj" mkdir "%OUT_DIR%\obj"
if not exist "%LIB_DIR%" mkdir "%LIB_DIR%"

echo.
echo ============================================================
echo [2/4] Building CUDA Library with MSBuild
echo ============================================================
echo.

REM Build using MSBuild - incremental build (only recompiles changed files)
msbuild "%BUILD_WIN%\simbackend_cuda.vcxproj" /p:Configuration=Release /p:Platform=x64 "/p:SolutionDir=%BUILD_WIN%\\" /m /v:minimal
if errorlevel 1 (
    echo ERROR: MSBuild failed
    exit /b 1
)
echo   [DONE] libsimbackend_cuda.dll created

echo.
echo ============================================================
echo [3/4] Building Rust
echo ============================================================
echo.

REM Build Rust
echo Building Rust project...
cd /d "%SRC_DIR%"
cargo build --release
if errorlevel 1 (
    echo ERROR: Rust build failed
    exit /b 1
)
echo   [DONE] Rust build complete

echo.
echo ============================================================
echo [4/4] Creating Launcher Scripts
echo ============================================================
echo.

REM Create launcher script that sets up PATH to reference binaries directly
(
echo @echo off
echo setlocal
echo.
echo REM Get the directory where this script is located
echo set BUILD_WIN=%%~dp0
echo set BUILD_WIN=%%BUILD_WIN:~0,-1%%
echo for %%%%I in ^("%%BUILD_WIN%%\.."^) do set SRC=%%%%~fI
echo.
echo if not defined CUDA_PATH set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
echo.
echo REM Set PATH to include binaries from their source locations
echo set PATH=%%SRC%%\target\release;%%SRC%%\src\cpp\build\lib;%%CUDA_PATH%%\bin;%%PATH%%
echo set PYTHONPATH=%%SRC%%;%%PYTHONPATH%%
echo.
echo REM Start JupyterLab
echo "%%BUILD_WIN%%\python\python.exe" -m jupyterlab --no-browser --port=8080 --ServerApp.token="" --notebook-dir="%%SRC%%\examples"
echo.
echo REM Kill any remaining ppf-contact-solver processes when JupyterLab exits
echo taskkill /F /IM ppf-contact-solver.exe 2^>nul
echo endlocal
) > "%BUILD_WIN%\start.bat"

REM Create Python launcher
(
echo import subprocess
echo import sys
echo import os
echo import webbrowser
echo import time
echo.
echo script_dir = os.path.dirname^(os.path.abspath^(__file__^)^)
echo src = os.path.dirname^(script_dir^)
echo.
echo python_exe = os.path.join^(script_dir, "python", "pythonw.exe"^)
echo bin_dir = os.path.join^(src, "target", "release"^)
echo lib_dir = os.path.join^(src, "src", "cpp", "build", "lib"^)
echo cuda_path = os.environ.get^("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"^)
echo cuda_bin = os.path.join^(cuda_path, "bin"^)
echo.
echo os.environ["PATH"] = bin_dir + ";" + lib_dir + ";" + cuda_bin + ";" + os.environ.get^("PATH", ""^)
echo os.environ["PYTHONPATH"] = src + ";" + os.environ.get^("PYTHONPATH", ""^)
echo.
echo proc = subprocess.Popen^([
echo     python_exe, "-m", "jupyterlab",
echo     "--no-browser", "--port=8080",
echo     "--ServerApp.token=",
echo     "--notebook-dir=" + os.path.join^(src, "examples"^)
echo ], env=os.environ^)
echo.
echo time.sleep^(3^)
echo webbrowser.open^("http://localhost:8080"^)
) > "%BUILD_WIN%\start-jupyterlab.pyw"

REM Update Python path configuration (embedded Python uses .pth file)
(
echo python311.zip
echo .
echo Lib\site-packages
echo %SRC_DIR%
echo import site
) > "%BUILD_WIN%\python\python311._pth"

echo   [DONE] Launcher scripts created

echo.
echo ============================================================
echo   BUILD COMPLETE!
echo ============================================================
echo.
echo To start JupyterLab, run: %BUILD_WIN%\start.bat
echo.

REM Skip pause if /nopause argument is provided (for automation)
if "%NOPAUSE%"=="0" (
    echo Press any key to exit...
    pause >nul
)

endlocal
