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

REM Use local CUDA installed by warmup.bat (required)
set CUDA_DIR=%BUILD_WIN%\cuda
set CUDA_PATH=%CUDA_DIR%
if not exist "%CUDA_DIR%\bin\nvcc.exe" (
    echo ERROR: Local CUDA not found at %CUDA_DIR%
    echo Please run warmup.bat first to install CUDA locally.
    exit /b 1
)
echo Using local CUDA from %CUDA_PATH%

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

REM Use local MinGit if installed by warmup.bat
set MINGIT_DIR=%BUILD_WIN%\mingit
if exist "%MINGIT_DIR%\cmd\git.exe" (
    echo Using local MinGit from %MINGIT_DIR%
    set "PATH=%MINGIT_DIR%\cmd;%PATH%"
) else (
    where git >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Git not found. Please run warmup.bat first to install MinGit.
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
        curl.exe -L -o "!EIGEN_ZIP!" "!EIGEN_URL!"
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

REM Setup portable MSVC environment (required)
set MSVC_DIR=%BUILD_WIN%\msvc
echo [1/4] Setting up Visual Studio environment...
if exist "%MSVC_DIR%\setup_x64.bat" (
    echo Using portable MSVC from %MSVC_DIR%
    call "%MSVC_DIR%\setup_x64.bat"
) else if exist "%MSVC_DIR%\setup.bat" (
    echo Using portable MSVC from %MSVC_DIR%
    call "%MSVC_DIR%\setup.bat"
) else (
    echo ERROR: Portable MSVC not found at %MSVC_DIR%
    echo Please run warmup.bat first to install MSVC locally.
    exit /b 1
)

REM Create output directories
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
if not exist "%OUT_DIR%\obj" mkdir "%OUT_DIR%\obj"
if not exist "%LIB_DIR%" mkdir "%LIB_DIR%"

echo.
echo ============================================================
echo [2/4] Building CUDA Library with nvcc
echo ============================================================
echo.

set NVCC="%CUDA_PATH%\bin\nvcc.exe"
set EIGEN_DIR=%DEPS%\eigen-3.4.0

REM Source files
set CPP_SRCS=%CPP_DIR%\simplelog\SimpleLog.cpp %CPP_DIR%\stub.cpp
set CU_SRCS=%CPP_DIR%\buffer\buffer.cu %CPP_DIR%\main\main.cu %CPP_DIR%\utility\utility.cu %CPP_DIR%\utility\dispatcher.cu %CPP_DIR%\csrmat\csrmat.cu %CPP_DIR%\contact\contact.cu %CPP_DIR%\energy\energy.cu %CPP_DIR%\eigenanalysis\eigenanalysis.cu %CPP_DIR%\barrier\barrier.cu %CPP_DIR%\strainlimiting\strainlimiting.cu %CPP_DIR%\solver\solver.cu %CPP_DIR%\kernels\reduce.cu %CPP_DIR%\kernels\exclusive_scan.cu %CPP_DIR%\kernels\vec_ops.cu

REM Compiler flags
set NVCC_FLAGS=-std=c++17 --expt-relaxed-constexpr --extended-lambda -O3 -rdc=true -shared -Wno-deprecated-gpu-targets
set NVCC_DEFINES=-DWIN32 -DNDEBUG -D_WINDOWS -D_USRDLL -D__NVCC__ -DEIGEN_WARNINGS_DISABLED -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT
set NVCC_INCLUDES=-I"%EIGEN_DIR%"
set NVCC_XCOMPILER=-Xcompiler "/EHsc /W0 /MD /O2"
set NVCC_SUPPRESS=--diag-suppress=1222,2527,2529,2651,2653,2668,2669,2670,2671,2735,2737,2739,20012,20011,20014,177,940,1394

echo Building with nvcc...
%NVCC% %NVCC_FLAGS% %NVCC_DEFINES% %NVCC_INCLUDES% %NVCC_XCOMPILER% %NVCC_SUPPRESS% -o "%LIB_DIR%\libsimbackend_cuda.dll" %CPP_SRCS% %CU_SRCS% -lcudart
if errorlevel 1 (
    echo ERROR: nvcc build failed
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
echo set CUDA_PATH=%%BUILD_WIN%%\cuda
echo.
echo REM Set PATH to include binaries from their source locations
echo set PATH=%%BUILD_WIN%%\python;%%BUILD_WIN%%\python\Scripts;%%SRC%%\target\release;%%SRC%%\src\cpp\build\lib;%%CUDA_PATH%%\bin;%%PATH%%
echo set PYTHONPATH=%%SRC%%;%%PYTHONPATH%%
echo.
echo REM Set Jupyter/IPython config to build-win-native relative paths
echo set JUPYTER_CONFIG_DIR=%%BUILD_WIN%%\jupyter\config
echo set JUPYTER_DATA_DIR=%%BUILD_WIN%%\jupyter\data
echo set IPYTHONDIR=%%BUILD_WIN%%\jupyter\ipython
echo.
echo REM Set dark theme if not already configured
echo set THEME_DIR=%%JUPYTER_CONFIG_DIR%%\lab\user-settings\@jupyterlab\apputils-extension
echo if not exist "%%THEME_DIR%%" mkdir "%%THEME_DIR%%"
echo if not exist "%%THEME_DIR%%\themes.jupyterlab-settings" ^(
echo     echo {"theme": "JupyterLab Dark"} ^> "%%THEME_DIR%%\themes.jupyterlab-settings"
echo ^)
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
echo cuda_path = os.path.join^(script_dir, "cuda"^)
echo cuda_bin = os.path.join^(cuda_path, "bin"^)
echo.
echo os.environ["PATH"] = bin_dir + ";" + lib_dir + ";" + cuda_bin + ";" + os.environ.get^("PATH", ""^)
echo os.environ["PYTHONPATH"] = src + ";" + os.environ.get^("PYTHONPATH", ""^)
echo.
echo # Set Jupyter/IPython config to build-win-native relative paths
echo os.environ["JUPYTER_CONFIG_DIR"] = os.path.join^(script_dir, "jupyter", "config"^)
echo os.environ["JUPYTER_DATA_DIR"] = os.path.join^(script_dir, "jupyter", "data"^)
echo os.environ["IPYTHONDIR"] = os.path.join^(script_dir, "jupyter", "ipython"^)
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
