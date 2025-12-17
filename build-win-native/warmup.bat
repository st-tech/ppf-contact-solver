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

REM Check if Long Path support is enabled
for /f %%i in ('powershell -Command "Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled 2>$null | Select-Object -ExpandProperty LongPathsEnabled"') do set LONGPATH_VALUE=%%i
if not "%LONGPATH_VALUE%"=="1" (
    echo ERROR: Windows Long Path support is not enabled.
    echo.
    echo Please run enable_long_path.bat as Administrator to enable it,
    echo then REBOOT your system before running this script.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

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
REM Install Chocolatey if not available
REM ============================================================
where choco >nul 2>&1
if errorlevel 1 (
    echo === Installing Chocolatey ===
    powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
    if errorlevel 1 (
        echo ERROR: Failed to install Chocolatey
        exit /b 1
    )
    echo Chocolatey installed successfully!
) else (
    echo Chocolatey already installed
)

REM ============================================================
REM Install Git via Chocolatey if not available
REM ============================================================
where git >nul 2>&1
if errorlevel 1 (
    echo === Installing Git ===
    C:\ProgramData\chocolatey\bin\choco.exe install git -y
    if errorlevel 1 (
        echo ERROR: Failed to install Git
        exit /b 1
    )
    echo Git installed successfully!
    REM Add Git to PATH for current session
    set "PATH=C:\Program Files\Git\cmd;%PATH%"
) else (
    echo Git already installed
)

REM ============================================================
REM Install Visual Studio 2022 Build Tools if not available
REM ============================================================
if not exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    echo === Installing Visual Studio 2022 Build Tools ===
    echo This may take a while...
    C:\ProgramData\chocolatey\bin\choco.exe install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" -y
    if errorlevel 1 (
        echo ERROR: Failed to install Visual Studio 2022 Build Tools
        exit /b 1
    )
    echo Visual Studio 2022 Build Tools installed successfully!
) else (
    echo Visual Studio 2022 Build Tools already installed
)

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
            powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://win.rustup.rs/x86_64' -OutFile '!RUSTUP_INIT!' -UseBasicParsing"
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
        powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '!PYTHON_URL!' -OutFile '!PYTHON_ZIP!' -UseBasicParsing"
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
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '!GET_PIP!' -UseBasicParsing"
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

echo.
echo === Upgrading pip ===
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: pip upgrade failed, continuing anyway...
)

echo.
echo === Installing Python packages ===

REM Core packages from warmup.py python_packages()
set PACKAGES=numpy pandas libigl plyfile requests gdown trimesh pywavefront matplotlib tqdm pythreejs ipywidgets open3d gpytoolbox tabulate tetgen triangle

REM Development tools
set DEV_PACKAGES=ruff black isort

REM JupyterLab (LSP disabled on Windows due to embedded Python subprocess issues)
set JUPYTER_PACKAGES=jupyterlab jupyterlab-code-formatter

echo.
echo Installing core packages...
"%PYTHON%" -m pip install --no-warn-script-location %PACKAGES%
if errorlevel 1 (
    echo WARNING: Some core packages failed to install
)

echo.
echo Installing development tools...
"%PYTHON%" -m pip install --no-warn-script-location %DEV_PACKAGES%
if errorlevel 1 (
    echo WARNING: Some development tools failed to install
)

echo.
echo Installing JupyterLab packages...
"%PYTHON%" -m pip install --no-warn-script-location %JUPYTER_PACKAGES%
if errorlevel 1 (
    echo WARNING: Some JupyterLab packages failed to install
)

echo.
echo === Installing sdf package from GitHub ===
"%PYTHON%" -m pip install --no-warn-script-location git+https://github.com/fogleman/sdf.git
if errorlevel 1 (
    echo WARNING: sdf package failed to install
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
