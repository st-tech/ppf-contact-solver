@echo off
REM File: warmup.bat
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
set BUILD_WIN=%BUILD_WIN:~0,-1%
set LOGFILE=%BUILD_WIN%\warmup.log

REM If not already being logged, restart with logging
if "%WARMUP_LOGGING%"=="" (
    set WARMUP_LOGGING=1
    echo Logging to %LOGFILE%
    powershell -Command "& { cmd /c 'set WARMUP_LOGGING=1&& set NOPAUSE=!NOPAUSE!&& \"%~f0\"' 2>&1 | Tee-Object -FilePath '%LOGFILE%' }"
    exit /b %ERRORLEVEL%
)

echo === ZOZO's Contact Solver Native Windows Environment Setup ===

for %%I in ("%BUILD_WIN%\..") do set SRC=%%~fI

set PYTHON_DIR=%BUILD_WIN%\python
set PYTHON=%PYTHON_DIR%\python.exe
set DOWNLOADS=%BUILD_WIN%\downloads
set RUST_DIR=%BUILD_WIN%\rust
set CARGO=%RUST_DIR%\bin\cargo.exe

echo.
echo Build directory: %BUILD_WIN%
echo Source directory: %SRC%
echo Python: %PYTHON%
echo Log file: %LOGFILE%
echo.

REM Create directories if needed
if not exist "%DOWNLOADS%" mkdir "%DOWNLOADS%"

REM ============================================================
REM Download 7-Zip portable (needed for CUDA extraction)
REM ============================================================
set SEVENZIP_DIR=%BUILD_WIN%\7zip
set SEVENZIP=%SEVENZIP_DIR%\7z.exe

if not exist "%SEVENZIP%" (
    echo === Downloading 7-Zip Portable ===

    set SEVENZIP_URL=https://www.7-zip.org/a/7z2408-x64.exe
    set SEVENZIP_EXE=%DOWNLOADS%\7z2408-x64.exe

    if not exist "!SEVENZIP_EXE!" (
        echo Downloading 7-Zip...
        curl.exe -L -o "!SEVENZIP_EXE!" "!SEVENZIP_URL!"
        if errorlevel 1 (
            echo ERROR: Failed to download 7-Zip
            exit /b 1
        )
    )

    echo Extracting 7-Zip...
    if not exist "%SEVENZIP_DIR%" mkdir "%SEVENZIP_DIR%"
    "!SEVENZIP_EXE!" /S /D=%SEVENZIP_DIR%
    if errorlevel 1 (
        echo ERROR: Failed to extract 7-Zip
        exit /b 1
    )

    echo 7-Zip setup complete!
) else (
    echo 7-Zip already installed
)

REM ============================================================
REM Download and setup MinGit (used instead of full Git installation)
REM ============================================================
set MINGIT_DIR=%BUILD_WIN%\mingit
set MINGIT_EXE=%MINGIT_DIR%\cmd\git.exe

if not exist "%MINGIT_EXE%" (
    echo === Downloading MinGit ===

    set MINGIT_VERSION=2.47.1
    set MINGIT_URL=https://github.com/git-for-windows/git/releases/download/v!MINGIT_VERSION!.windows.1/MinGit-!MINGIT_VERSION!-64-bit.zip
    set MINGIT_ZIP=%DOWNLOADS%\MinGit-!MINGIT_VERSION!-64-bit.zip

    if not exist "!MINGIT_ZIP!" (
        echo Downloading MinGit !MINGIT_VERSION!...
        curl.exe -L -o "!MINGIT_ZIP!" "!MINGIT_URL!"
        if errorlevel 1 (
            echo ERROR: Failed to download MinGit
            exit /b 1
        )
    )

    echo Extracting MinGit...
    if not exist "%MINGIT_DIR%" mkdir "%MINGIT_DIR%"
    powershell -Command "Expand-Archive -Path '!MINGIT_ZIP!' -DestinationPath '%MINGIT_DIR%' -Force"
    if errorlevel 1 (
        echo ERROR: Failed to extract MinGit
        exit /b 1
    )

    echo MinGit setup complete!
) else (
    echo MinGit already installed
)

REM Add MinGit to PATH for current session
set "PATH=%MINGIT_DIR%\cmd;%PATH%"

REM ============================================================
REM Download and extract CUDA Toolkit (no admin required)
REM ============================================================
set CUDA_DIR=%BUILD_WIN%\cuda
set NVCC=%CUDA_DIR%\bin\nvcc.exe

if not exist "%NVCC%" (
    echo === Downloading CUDA Toolkit 12.8 ===

    set CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_572.61_windows.exe
    set CUDA_EXE=%DOWNLOADS%\cuda_12.8.1_572.61_windows.exe

    if not exist "!CUDA_EXE!" (
        echo Downloading CUDA 12.8.1 (about 3GB, please wait^)...
        curl.exe -L -o "!CUDA_EXE!" "!CUDA_URL!"
        if errorlevel 1 (
            echo ERROR: Failed to download CUDA Toolkit
            exit /b 1
        )
    )

    echo Extracting CUDA (this takes several minutes^)...
    "%SEVENZIP%" x "!CUDA_EXE!" -o"%CUDA_DIR%_temp" -y
    if errorlevel 1 (
        echo ERROR: Failed to extract CUDA Toolkit
        exit /b 1
    )

    REM Create CUDA directory and merge required components
    if not exist "%CUDA_DIR%" mkdir "%CUDA_DIR%"

    REM Copy nvcc (compiler)
    echo   Copying nvcc...
    robocopy "%CUDA_DIR%_temp\cuda_nvcc\nvcc" "%CUDA_DIR%" /E /NFL /NDL /NJH /NJS /NC /NS /NP

    REM Copy cudart (runtime)
    echo   Copying cudart...
    robocopy "%CUDA_DIR%_temp\cuda_cudart\cudart" "%CUDA_DIR%" /E /NFL /NDL /NJH /NJS /NC /NS /NP

    REM Copy cccl (thrust, cub headers)
    echo   Copying cccl headers...
    robocopy "%CUDA_DIR%_temp\cuda_cccl\thrust" "%CUDA_DIR%" /E /NFL /NDL /NJH /NJS /NC /NS /NP

    REM Copy nvrtc (runtime compilation)
    echo   Copying nvrtc...
    robocopy "%CUDA_DIR%_temp\cuda_nvrtc\nvrtc" "%CUDA_DIR%" /E /NFL /NDL /NJH /NJS /NC /NS /NP
    robocopy "%CUDA_DIR%_temp\cuda_nvrtc\nvrtc_dev" "%CUDA_DIR%" /E /NFL /NDL /NJH /NJS /NC /NS /NP

    REM Copy profiler API
    echo   Copying profiler API...
    robocopy "%CUDA_DIR%_temp\cuda_profiler_api\cuda_profiler_api" "%CUDA_DIR%" /E /NFL /NDL /NJH /NJS /NC /NS /NP

    REM Cleanup temp directory
    rmdir /s /q "%CUDA_DIR%_temp"

    echo CUDA Toolkit extracted successfully!
) else (
    echo CUDA Toolkit already installed
)
set CUDA_PATH=%CUDA_DIR%

REM ============================================================
REM Install Rust locally if not available
REM ============================================================
where cargo >nul 2>&1
if errorlevel 1 (
    if not exist "%CARGO%" (
        echo === Installing Rust locally ===

        set RUSTUP_INIT=%DOWNLOADS%\rustup-init.exe
        if not exist "!RUSTUP_INIT!" (
            echo Downloading rustup-init.exe...
            curl.exe -L -o "!RUSTUP_INIT!" "https://win.rustup.rs/x86_64"
            if errorlevel 1 (
                echo ERROR: Failed to download rustup-init.exe
                exit /b 1
            )
        )

        echo Installing Rust to %RUST_DIR%...
        set RUSTUP_HOME=%RUST_DIR%\rustup
        set CARGO_HOME=%RUST_DIR%
        "!RUSTUP_INIT!" -y --no-modify-path --default-toolchain stable
        if errorlevel 1 (
            echo ERROR: Failed to install Rust
            exit /b 1
        )

        echo Rust installed successfully!
    ) else (
        echo Rust found at %CARGO%
    )
) else (
    echo Rust already available in PATH
)

REM ============================================================
REM Download and setup embedded Python if not present
REM ============================================================
if not exist "%PYTHON%" (
    echo === Downloading Embedded Python ===

    set PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip
    set PYTHON_ZIP=%DOWNLOADS%\python-3.11.9-embed-amd64.zip

    if not exist "!PYTHON_ZIP!" (
        echo Downloading Python 3.11.9 embedded...
        curl.exe -L -o "!PYTHON_ZIP!" "!PYTHON_URL!"
        if errorlevel 1 (
            echo ERROR: Failed to download Python
            exit /b 1
        )
    )

    echo Extracting Python...
    if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"
    powershell -Command "Expand-Archive -Path '!PYTHON_ZIP!' -DestinationPath '%PYTHON_DIR%' -Force"
    if errorlevel 1 (
        echo ERROR: Failed to extract Python
        exit /b 1
    )

    REM Enable pip by modifying python311._pth
    REM Also add source directory so 'frontend' module can be imported
    echo Enabling pip support and adding source path...
    echo python311.zip> "%PYTHON_DIR%\python311._pth"
    echo .>> "%PYTHON_DIR%\python311._pth"
    echo Lib\site-packages>> "%PYTHON_DIR%\python311._pth"
    echo %SRC%>> "%PYTHON_DIR%\python311._pth"
    echo import site>> "%PYTHON_DIR%\python311._pth"

    REM Download and install pip
    echo Downloading get-pip.py...
    set GET_PIP=%PYTHON_DIR%\get-pip.py
    curl.exe -L -o "!GET_PIP!" "https://bootstrap.pypa.io/get-pip.py"
    if errorlevel 1 (
        echo ERROR: Failed to download get-pip.py
        exit /b 1
    )

    echo Installing pip...
    "%PYTHON%" "!GET_PIP!"
    if errorlevel 1 (
        echo ERROR: Failed to install pip
        exit /b 1
    )

    echo Embedded Python setup complete!
)

REM Check if Python exists
if not exist "%PYTHON%" (
    echo ERROR: Embedded Python not found at %PYTHON%
    echo Please ensure the python directory exists with an embedded Python installation.
    exit /b 1
)

echo === Checking Python ===
"%PYTHON%" --version
if errorlevel 1 (
    echo ERROR: Python check failed
    exit /b 1
)

REM ============================================================
REM Install Portable MSVC (no admin required)
REM Uses: https://gist.github.com/mmozeiko/7f3162ec2988e81e56d5c4e22cde9977
REM ============================================================
set MSVC_DIR=%BUILD_WIN%\msvc

REM Check for setup.bat or setup_x64.bat (mmozeiko script creates setup_x64.bat)
set MSVC_SETUP=%MSVC_DIR%\setup.bat
if not exist "%MSVC_SETUP%" set MSVC_SETUP=%MSVC_DIR%\setup_x64.bat

if not exist "%MSVC_SETUP%" (
    echo.
    echo === Downloading Portable MSVC ===

    set PBT_SCRIPT=%DOWNLOADS%\portable-msvc.py
    if not exist "!PBT_SCRIPT!" (
        echo Downloading portable-msvc.py...
        curl.exe -L -o "!PBT_SCRIPT!" "https://gist.github.com/mmozeiko/7f3162ec2988e81e56d5c4e22cde9977/raw/portable-msvc.py"
        if errorlevel 1 (
            echo ERROR: Failed to download portable-msvc.py
            exit /b 1
        )
    )

    echo Installing MSVC to %MSVC_DIR% (this takes a while^)...
    pushd "%BUILD_WIN%"
    "%PYTHON%" "!PBT_SCRIPT!" --accept-license --vs 2022
    popd
    if errorlevel 1 (
        echo ERROR: Failed to install Portable MSVC
        exit /b 1
    )

    echo Portable MSVC installed successfully!
) else (
    echo Portable MSVC already installed
)

REM ============================================================
REM Install Python packages (skip if already installed)
REM ============================================================
REM Check if jupyterlab is installed as proxy for "packages installed"
"%PYTHON%" -c "import jupyterlab" >nul 2>&1
if errorlevel 1 (
    echo.
    echo === Upgrading pip ===
    "%PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 (
        echo WARNING: pip upgrade failed, continuing anyway...
    )

    echo.
    echo === Installing Python packages ===

    REM Core packages from warmup.py python_packages()
    set PACKAGES=numpy numba plyfile requests gdown trimesh pywavefront matplotlib tqdm pythreejs ipywidgets fast-simplification tabulate triangle

    REM Development tools
    set DEV_PACKAGES=ruff black isort

    REM JupyterLab (LSP disabled on Windows due to embedded Python subprocess issues)
    REM nbconvert is needed for fast-check-all.bat to convert notebooks to Python scripts
    set JUPYTER_PACKAGES=jupyterlab jupyterlab-code-formatter nbconvert

    echo.
    echo Installing core packages...
    "%PYTHON%" -m pip install --no-warn-script-location !PACKAGES!
    if errorlevel 1 (
        echo WARNING: Some core packages failed to install
    )

    echo.
    echo Installing pytetwild (fTetWild wrapper^)...
    "%PYTHON%" -m pip install --no-warn-script-location pytetwild
    if errorlevel 1 (
        echo WARNING: pytetwild failed to install
    )

    echo.
    echo Installing development tools...
    "%PYTHON%" -m pip install --no-warn-script-location !DEV_PACKAGES!
    if errorlevel 1 (
        echo WARNING: Some development tools failed to install
    )

    echo.
    echo Installing JupyterLab packages...
    "%PYTHON%" -m pip install --no-warn-script-location !JUPYTER_PACKAGES!
    if errorlevel 1 (
        echo WARNING: Some JupyterLab packages failed to install
    )

    echo.
    echo === Disabling LSP for Windows (embedded Python compatibility) ===
    if not exist "%PYTHON_DIR%\share\jupyter\lab\settings" mkdir "%PYTHON_DIR%\share\jupyter\lab\settings"
    (
    echo {
    echo   "@jupyterlab/lsp-extension:plugin": {
    echo     "languageServers": {}
    echo   }
    echo }
    ) > "%PYTHON_DIR%\share\jupyter\lab\settings\overrides.json"

    echo.
    echo === Verifying installation ===
    "%PYTHON%" -m pip list
) else (
    echo Python packages already installed
)

REM ============================================================
REM Build slim FFmpeg (for video export)
REM ============================================================
set FFMPEG_DIR=%BUILD_WIN%\ffmpeg
if not exist "%FFMPEG_DIR%\ffmpeg.exe" (
    echo.
    echo === Building slim FFmpeg ===
    call "%BUILD_WIN%\make-slim-ffmpeg.bat"
    if errorlevel 1 (
        echo WARNING: FFmpeg build failed
    )
) else (
    echo FFmpeg already installed
)

echo.
echo === Setup complete! ===
echo.
echo Next step: Run build.bat to build the solver.

REM Skip pause if /nopause argument is provided (for automation)
if "%NOPAUSE%"=="0" (
    echo Press any key to exit...
    pause >nul
)

endlocal
