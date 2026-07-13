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
    REM `exit $LASTEXITCODE` inside the -Command is REQUIRED: a PowerShell
    REM pipeline ending in the Tee-Object cmdlet exits 0 regardless of the
    REM inner cmd's failure, so without it a failed build (or the MSVC-not-
    REM found guard below) is masked as success. This forwards the real code.
    powershell -Command "& { cmd /c 'set BUILD_LOGGING=1&& set NOPAUSE=!NOPAUSE!&& \"%~f0\"' 2>&1 | Tee-Object -FilePath '%LOGFILE%'; exit $LASTEXITCODE }"
    exit /b %ERRORLEVEL%
)

echo ============================================================
echo   ZOZO's Contact Solver - Windows Build Script
echo ============================================================
echo.
REM Get parent directory (SRC_DIR)
for %%I in ("%BUILD_WIN%\..") do set SRC_DIR=%%~fI

REM CUDA driver lives at crates\ppf-cts-solver\; its src\cpp tree
REM (and the build\ output produced by nvcc) sits alongside the Rust
REM source.
set CPP_DIR=%SRC_DIR%\crates\ppf-cts-solver\src\cpp
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

REM Embedded Python from warmup.bat. Runs JupyterLab via the launcher
REM scripts written below. The PyO3 extension (_ppf_cts_py.dll) is built
REM directly into target\release by `cargo build --release` and loaded
REM by frontend/__init__.py from there, so no wheel is installed here.
set PYTHON_EXE=%BUILD_WIN%\python\python.exe
if not exist "%PYTHON_EXE%" (
    echo ERROR: Embedded Python not found at %PYTHON_EXE%
    echo Please run warmup.bat first.
    exit /b 1
)

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
call "%BUILD_WIN%\scripts\load-downloads.bat"
if errorlevel 1 (
    echo ERROR: Failed to load download manifest
    exit /b 1
)

REM Eigen extracts to a directory whose name matches the archive stem
for %%I in ("%FILE_EIGEN%") do set EIGEN_STEM=%%~nI
if not exist "%DEPS%\%EIGEN_STEM%" (
    echo [0/3] Downloading Eigen...
    if not exist "%DOWNLOADS%" mkdir "%DOWNLOADS%"
    if not exist "%DEPS%" mkdir "%DEPS%"

    set "EIGEN_ZIP=%DOWNLOADS%\%FILE_EIGEN%"

    if not exist "!EIGEN_ZIP!" (
        curl.exe -fL -o "!EIGEN_ZIP!" "%URL_EIGEN%"
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
set EIGEN_DIR=%DEPS%\%EIGEN_STEM%

REM Source files
set CPP_SRCS=%CPP_DIR%\simplelog\SimpleLog.cpp %CPP_DIR%\stub.cpp
REM Keep this list in sync with BASE_DIRS in crates/ppf-cts-solver/src/cpp/Makefile
REM (the Linux/macOS build). schwarz/schwarz.cu defines schwarz::build/apply that
REM solver.cu references; omitting it here fails the link with unresolved externals.
set CU_SRCS=%CPP_DIR%\buffer\buffer.cu %CPP_DIR%\main\main.cu %CPP_DIR%\utility\utility.cu %CPP_DIR%\utility\dispatcher.cu %CPP_DIR%\csrmat\csrmat.cu %CPP_DIR%\contact\contact.cu %CPP_DIR%\energy\energy.cu %CPP_DIR%\eigenanalysis\eigenanalysis.cu %CPP_DIR%\barrier\barrier.cu %CPP_DIR%\strainlimiting\strainlimiting.cu %CPP_DIR%\solver\solver.cu %CPP_DIR%\schwarz\schwarz.cu %CPP_DIR%\kernels\reduce.cu %CPP_DIR%\kernels\exclusive_scan.cu %CPP_DIR%\kernels\vec_ops.cu %CPP_DIR%\kernels\radix_sort.cu %CPP_DIR%\lbvh\lbvh.cu %CPP_DIR%\plasticity\plasticity.cu

REM Compiler flags. Device link-time optimization (LTO), matching the Linux
REM Makefile: compile each TU to an LTO intermediate (code=lto_86), then device-
REM link with -dlto so cross-TU device callees (notably barrier::compute_stiffness)
REM inline into the contact-Hessian embed kernels, roughly halving contact
REM matrix-assembly cost. The LTO IR is compute_86, so sm_86 (Ampere) is the floor.
REM IMPORTANT: device LTO CANNOT embed a JIT-able PTX (the -dlto link lowers its IR
REM straight to SASS; a code=compute_XX request is silently dropped), so the .dll
REM is frozen to the exact cubins we ship, no forward-JIT fallback. We therefore
REM emit one native SASS cubin per supported arch at the link (see the -gencode
REM list below) and add a new entry, then rebuild, when a new GPU generation ships.
REM nvcc forbids -dlto beside -gencode at compile (hence code=lto_86), but needs
REM the full -gencode list at the link.
set NVCC_COMMON=-std=c++17 --expt-relaxed-constexpr --extended-lambda -O3 -Wno-deprecated-gpu-targets
set NVCC_DEFINES=-DWIN32 -DNDEBUG -D_WINDOWS -D_USRDLL -D__NVCC__ -DEIGEN_WARNINGS_DISABLED -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT
set NVCC_INCLUDES=-I"%EIGEN_DIR%"
set NVCC_XCOMPILER=-Xcompiler "/EHsc /W0 /MD /O2"
set NVCC_SUPPRESS=--diag-suppress=1222,2527,2529,2651,2653,2668,2669,2670,2671,2735,2737,2739,20012,20011,20014,177,940,1394

set OBJ_DIR=%OUT_DIR%\obj
if not exist "%OBJ_DIR%" mkdir "%OBJ_DIR%"

echo Compiling CUDA TUs to LTO intermediates (code=lto_86)...
set OBJS=
for %%f in (%CU_SRCS%) do (
    %NVCC% -dc -gencode arch=compute_86,code=lto_86 %NVCC_COMMON% %NVCC_DEFINES% %NVCC_INCLUDES% %NVCC_XCOMPILER% %NVCC_SUPPRESS% "%%f" -o "%OBJ_DIR%\%%~nf.obj" || exit /b 1
    set OBJS=!OBJS! "%OBJ_DIR%\%%~nf.obj"
)

echo Compiling host C++ TUs...
for %%f in (%CPP_SRCS%) do (
    %NVCC% -c %NVCC_COMMON% %NVCC_DEFINES% %NVCC_INCLUDES% %NVCC_XCOMPILER% %NVCC_SUPPRESS% "%%f" -o "%OBJ_DIR%\%%~nf.obj" || exit /b 1
    set OBJS=!OBJS! "%OBJ_DIR%\%%~nf.obj"
)

echo Device-linking with LTO (-dlto: native SASS cubins sm_86/89/90/100/120, no PTX)...
REM SASS is forward-compatible within a major version: sm_86 covers sm_87 (Orin)
REM and sm_89 (Ada: RTX 40, L40S); sm_90 Hopper; sm_100 Blackwell DC (B200);
REM sm_120 Blackwell consumer (RTX 50-series). Add a cubin + rebuild for new archs.
REM Keep in sync with SUPPORTED_SM in crates\ppf-cts-core\src\utils.rs (launch gate).
%NVCC% -shared -dlto -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=sm_89 -gencode arch=compute_86,code=sm_90 -gencode arch=compute_86,code=sm_100 -gencode arch=compute_86,code=sm_120 -Xcompiler "/MD" !OBJS! -lcudart -o "%LIB_DIR%\libsimbackend_cuda.dll"
if errorlevel 1 (
    echo ERROR: nvcc device-link failed
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

REM PyO3 links _ppf_cts_py.dll against python3.lib (the abi3 stable lib).
REM The embedded python\ ships no libs\, so point PyO3 at the full NuGet
REM CPython (python_full\, provisioned by warmup.bat) for the cdylib link.
REM Unlike macOS/Linux (which resolve Python symbols at load time), Windows
REM must link the import library at build time.
set "PYTHON_FULL_EXE=%BUILD_WIN%\python_full\python.exe"
if not exist "%PYTHON_FULL_EXE%" (
    echo ERROR: Full Python not found at %PYTHON_FULL_EXE%
    echo Please run warmup.bat first ^(it installs python_full alongside the embedded python^).
    exit /b 1
)
set "PYO3_PYTHON=%PYTHON_FULL_EXE%"

cargo build --release
if errorlevel 1 (
    echo ERROR: Rust build failed
    exit /b 1
)

REM `cargo build --release` above already builds ppf-cts-server (it is a
REM workspace default-member alongside ppf-cts-solver and ppf-cts-py). The
REM Blender addon's Windows Native launcher
REM (blender_addon/core/connection.py:spawn_win_native_server) spawns
REM target\release\ppf-cts-server.exe whenever the user picks "Windows
REM Native" mode in the addon UI, so rebuild explicitly as a guard and
REM verify it exists below so the bundle always ships both binaries.
echo Building ppf-cts-server...
cargo build --release -p ppf-cts-server
if errorlevel 1 (
    echo ERROR: ppf-cts-server build failed
    exit /b 1
)
if not exist "%SRC_DIR%\target\release\ppf-cts-server.exe" (
    echo ERROR: target\release\ppf-cts-server.exe not found after build
    exit /b 1
)
echo   [DONE] Rust build complete

REM `cargo build --release` above also builds the ppf-cts-py crate
REM (a workspace default-member) into target\release\_ppf_cts_py.dll.
REM frontend/__init__.py loads that cdylib directly by absolute path, so
REM no wheel install into the embedded Python is needed. The launcher
REM scripts below put %SRC%\target\release on PATH and %SRC% on
REM PYTHONPATH, which is all frontend needs to find it.
if not exist "%SRC_DIR%\target\release\_ppf_cts_py.dll" (
    echo ERROR: target\release\_ppf_cts_py.dll not found after build
    exit /b 1
)
echo   [DONE] _ppf_cts_py.dll built

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
echo set PATH=%%BUILD_WIN%%\python;%%BUILD_WIN%%\python\Scripts;%%SRC%%\target\release;%%SRC%%\crates\ppf-cts-solver\src\cpp\build\lib;%%CUDA_PATH%%\bin;%%PATH%%
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
echo lib_dir = os.path.join^(src, "crates", "ppf-cts-solver", "src", "cpp", "build", "lib"^)
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
