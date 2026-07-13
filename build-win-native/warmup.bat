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
    REM `exit $LASTEXITCODE` inside the -Command is REQUIRED: a PowerShell
    REM pipeline ending in the Tee-Object cmdlet exits 0 regardless of the
    REM inner cmd's failure, so without it every warmup failure (bad MSVC
    REM install, missing dep) is masked as success and only surfaces steps
    REM later. This forwards the re-launched warmup's real exit code.
    powershell -Command "& { cmd /c 'set WARMUP_LOGGING=1&& set NOPAUSE=!NOPAUSE!&& \"%~f0\"' 2>&1 | Tee-Object -FilePath '%LOGFILE%'; exit $LASTEXITCODE }"
    exit /b %ERRORLEVEL%
)

echo === ZOZO's Contact Solver Native Windows Environment Setup ===

for %%I in ("%BUILD_WIN%\..") do set SRC=%%~fI

set PYTHON_DIR=%BUILD_WIN%\python
set PYTHON=%PYTHON_DIR%\python.exe
set DOWNLOADS=%BUILD_WIN%\downloads
set RUST_DIR=%BUILD_WIN%\rust
set CARGO=%RUST_DIR%\bin\cargo.exe

REM Load the URL/FILE manifest (single source of truth, scripts\downloads.txt)
call "%BUILD_WIN%\scripts\load-downloads.bat"
if errorlevel 1 (
    echo ERROR: Failed to load download manifest
    exit /b 1
)

REM Pre-flight: every URL in the manifest must be reachable before we start
REM downloading. Catches upstream rot at the manifest, not 3GB into a fetch.
call "%BUILD_WIN%\scripts\check-downloads.bat" /nopause
if errorlevel 1 (
    echo ERROR: One or more download URLs are unreachable.
    echo Fix the offending pointer^(s^) in scripts\downloads.txt and re-run.
    exit /b 1
)

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

    set "SEVENZIP_EXE=%DOWNLOADS%\%FILE_7ZIP%"

    if not exist "!SEVENZIP_EXE!" (
        echo Downloading 7-Zip...
        curl.exe -fL -o "!SEVENZIP_EXE!" "%URL_7ZIP%"
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

    set "MINGIT_ZIP=%DOWNLOADS%\%FILE_MINGIT%"

    if not exist "!MINGIT_ZIP!" (
        echo Downloading MinGit...
        curl.exe -fL -o "!MINGIT_ZIP!" "%URL_MINGIT%"
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
    echo === Downloading CUDA Toolkit ===

    set "CUDA_EXE=%DOWNLOADS%\%FILE_CUDA%"

    if not exist "!CUDA_EXE!" (
        echo Downloading CUDA (about 3GB, please wait^)...
        curl.exe -fL -o "!CUDA_EXE!" "%URL_CUDA%"
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

        set "RUSTUP_INIT=%DOWNLOADS%\%FILE_RUSTUP%"
        if not exist "!RUSTUP_INIT!" (
            echo Downloading rustup-init.exe...
            curl.exe -fL -o "!RUSTUP_INIT!" "%URL_RUSTUP%"
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

    set "PYTHON_ZIP=%DOWNLOADS%\%FILE_PYTHON%"

    if not exist "!PYTHON_ZIP!" (
        echo Downloading embedded Python...
        curl.exe -fL -o "!PYTHON_ZIP!" "%URL_PYTHON%"
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
    set "GET_PIP=%PYTHON_DIR%\%FILE_GET_PIP%"
    curl.exe -fL -o "!GET_PIP!" "%URL_GET_PIP%"
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
REM Install Full Python (with libs\python3.lib + include\) via NuGet
REM ============================================================
REM
REM The embedded Python distribution from python.org strips libs/ and
REM include/, so it can't be used as the PyO3 build interpreter for the
REM _ppf_cts_py.dll cdylib: PyO3's Windows link step needs python3.lib,
REM which only ships with the dev-headers payload of a regular CPython
REM install. build.bat points PYO3_PYTHON at this python_full for
REM `cargo build --release`.
REM
REM Pull a fully portable CPython via the official NuGet `python`
REM package (https://www.nuget.org/packages/python). Unlike the
REM python.org MSI installer, NuGet's package writes nothing to the
REM Windows registry, py launcher, or %LOCALAPPDATA%\Programs\Python
REM — it's a self-contained zip that lands at <out>\python\tools\
REM with python.exe + libs\python3.lib + include\Python.h alongside.
REM We move tools\ to python_full\ and it's purely a build-time tool
REM here; bundle.bat does NOT include python_full\ in dist (only the
REM stripped embedded python\ ships with releases).
set PYTHON_FULL_DIR=%BUILD_WIN%\python_full
set PYTHON_FULL=%PYTHON_FULL_DIR%\python.exe
if not exist "%PYTHON_FULL%" (
    echo === Installing Full Python 3.11.9 to %PYTHON_FULL_DIR% ^(NuGet, portable, no system pollution^) ===
    set "NUGET_EXE=%DOWNLOADS%\nuget.exe"
    if not exist "!NUGET_EXE!" (
        echo Downloading nuget.exe...
        curl.exe -fL -o "!NUGET_EXE!" "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe"
        if errorlevel 1 (
            echo ERROR: Failed to download nuget.exe
            exit /b 1
        )
    )
    REM Stage NuGet output under nuget_tmp\ so we don't collide with
    REM the existing embedded `python\` directory (NuGet defaults to
    REM <out>\python\ when -ExcludeVersion is set).
    set "NUGET_TMP=%BUILD_WIN%\nuget_tmp"
    if exist "!NUGET_TMP!" rmdir /s /q "!NUGET_TMP!"
    echo Running NuGet install python -Version 3.11.9...
    "!NUGET_EXE!" install python -Version 3.11.9 -OutputDirectory "!NUGET_TMP!" -ExcludeVersion -Verbosity quiet
    if errorlevel 1 (
        echo ERROR: NuGet python install failed
        exit /b 1
    )
    if not exist "!NUGET_TMP!\python\tools\python.exe" (
        echo ERROR: NuGet package layout unexpected; tools\python.exe missing
        exit /b 1
    )
    move "!NUGET_TMP!\python\tools" "%PYTHON_FULL_DIR%" >nul
    if errorlevel 1 (
        echo ERROR: Failed to move NuGet python\tools to %PYTHON_FULL_DIR%
        exit /b 1
    )
    rmdir /s /q "!NUGET_TMP!"
    if not exist "%PYTHON_FULL%" (
        echo ERROR: NuGet python install completed but %PYTHON_FULL% not present
        exit /b 1
    )
    echo Full Python installed ^(NuGet portable^).
)
"%PYTHON_FULL%" --version
if errorlevel 1 (
    echo ERROR: Full Python check failed
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

    set "PBT_SCRIPT=%DOWNLOADS%\%FILE_PORTABLE_MSVC%"
    if not exist "!PBT_SCRIPT!" (
        echo Downloading portable-msvc.py...
        curl.exe -fL -o "!PBT_SCRIPT!" "%URL_PORTABLE_MSVC%"
        if errorlevel 1 (
            echo ERROR: Failed to download portable-msvc.py
            exit /b 1
        )
    )

    echo Installing MSVC to %MSVC_DIR% (this takes a while^)...
    pushd "%BUILD_WIN%"
    "%PYTHON%" "!PBT_SCRIPT!" --accept-license --vs 2022
    REM Capture the installer's exit code BEFORE popd: popd can reset
    REM errorlevel and hide a failed MSVC install.
    set MSVC_RC=!errorlevel!
    popd
    if not "!MSVC_RC!"=="0" (
        echo ERROR: Failed to install Portable MSVC ^(exit !MSVC_RC!^)
        exit /b 1
    )

    REM Belt-and-suspenders: portable-msvc.py is fetched from a mutable gist
    REM and could exit 0 yet leave no setup script (an upstream regression
    REM once did exactly this). Assert the toolchain env script the rest of
    REM the build calls actually exists, so a silent no-op fails here loudly
    REM instead of two build steps later.
    if not exist "%MSVC_DIR%\setup_x64.bat" if not exist "%MSVC_DIR%\setup.bat" (
        echo ERROR: portable-msvc.py finished but no setup_x64.bat/setup.bat under %MSVC_DIR%
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

    REM Core packages from warmup.py python_packages(). certifi is
    REM listed explicitly so the bundled Python ships a CA bundle even
    REM if a future warmup pass drops requests as a transitive carrier.
    REM Without a CA bundle, the verify-on-clean-instance fast-check
    REM run trips SSLCertVerificationError when an example downloads a
    REM mesh asset over HTTPS.
    REM cbor2: required by frontend/_cbor_bridge_.py for the CBOR
    REM envelope codec the addon now uses to encode meshes/params.
    REM psutil: pulled in by frontend (BUILD_WORKER) and by the response
    REM builder's runtime utilization probe (CPU%/RAM%) on machines
    REM without nvidia-smi.
    REM scipy: REQUIRED by frontend/_decoder_.py. The two-stage Poisson pin
    REM diffusion for partially-pinned SOLID objects (_build_solid_pin_fields,
    REM _build_harmonic_interior_operator) uses scipy.sparse solves. It is
    REM imported inside a try/except that silently returns None when scipy is
    REM absent, so a Windows build WITHOUT scipy does not crash: it quietly
    REM takes a different (surface-only) fallback pin path than the Linux build,
    REM producing a DIFFERENT driven-vertex set (e.g. 861 vs 845 on a partial-
    REM pin SOLID) and hence a divergent simulation for the same scene. scipy is
    REM present transitively on Linux but was never installed here, so it must
    REM be listed explicitly to keep Windows and Linux on the same code path.
    REM PIN the ABI-coupled / native-extension deps to a verified-good set.
    REM Left unpinned, every warmup bundles whatever was latest that day.
    REM scipy is built against a specific numpy ABI; when pip pairs a scipy
    REM with a numpy it was not built against, the
    REM SuperLU solve in the partial-pin SOLID harmonic extension crashed the
    REM build worker NATIVELY (no Python exception, so the graceful scipy
    REM fallback never ran and no ERROR line was emitted, the add-on only
    REM showed "build worker exited with code 1" at ~10%). Pinning numpy +
    REM scipy + numba (numpy-coupled) and the native tetra stack keeps the
    REM bundle reproducible so a good combo cannot silently drift into a bad
    REM one. Bump these together, and re-verify a partial-pin SOLID build,
    REM whenever the bundled Python minor changes. Pure-Python deps stay
    REM unpinned (they cannot cause an ABI crash).
    set PACKAGES=numpy==2.4.4 scipy==1.17.1 numba==0.65.1 plyfile requests certifi gdown trimesh==4.12.2 pywavefront matplotlib tqdm pythreejs ipywidgets fast-simplification==0.1.13 tabulate triangle==20250106 cbor2==6.0.1 psutil==7.2.2

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
    echo Installing pytetwild (fTetWild^) + tetgen (TetGen^) + pyvista...
    rem pyvista is imported at the top of pytetwild._accessor but is not
    rem declared as a hard dependency, so install it explicitly.
    "%PYTHON%" -m pip install --no-warn-script-location pytetwild==0.2.3 tetgen==0.8.4 pyvista==0.48.4
    if errorlevel 1 (
        echo WARNING: pytetwild/tetgen failed to install
    )

    echo.
    echo === Verifying critical frontend dependencies ===
    REM The pip installs above only WARN on failure and continue, so a partial
    REM failure (observed: pytetwild installed but tetgen/pyvista silently
    REM missing; or scipy absent) would otherwise ship a bundle whose build
    REM worker cannot run a SOLID simulation. tetgen/pytetwild/pyvista missing =
    REM ModuleNotFoundError at build time; scipy missing is worse, it does not
    REM crash but makes the frontend take a different pin-diffusion path so the
    REM Windows result silently diverges from Linux. Hard-fail here so a broken
    REM bundle can never be built or published; re-run warmup on a working pip.
    "%PYTHON%" -c "import importlib.util as u, sys; req=['numpy','scipy','cbor2','tetgen','pytetwild','pyvista']; miss=[m for m in req if u.find_spec(m) is None]; sys.stderr.write('missing critical frontend deps: '+', '.join(miss)+'\n') if miss else sys.stdout.write('all critical frontend deps present\n'); sys.exit(1 if miss else 0)"
    if errorlevel 1 (
        echo ERROR: critical frontend dependencies are missing after install.
        echo A bundle without these cannot run a SOLID simulation. Re-run
        echo warmup.bat with a working network/pip index and check the log above.
        exit /b 1
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
