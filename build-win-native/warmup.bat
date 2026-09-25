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
    REM
    REM The `!ERRORLEVEL!` below is equally REQUIRED, for the same reason one
    REM level up. cmd.exe percent-expands a parenthesized block once, when it
    REM parses the whole block, so `%ERRORLEVEL%` here would be substituted
    REM with the value from BEFORE the powershell line ran (0, since the
    REM preceding statements are successful SETs) and would discard the exit
    REM code the line above went to such lengths to forward. Delayed
    REM expansion is evaluated per line at execution time, so it sees the
    REM real code. Verified on Windows Server 2025: with `%ERRORLEVEL%` a
    REM child exiting 1 reports success, with `!ERRORLEVEL!` it reports 1.
    REM
    REM THE ENVIRONMENT IS SET BY POWERSHELL, NOT BY A `set ... &&` CHAIN
    REM INSIDE THE cmd STRING, and that is load-bearing rather than tidier.
    REM Passing the script as `\"%~f0\"` inside the -Command string requires
    REM escaped quotes, and that spelling loses the child's exit code: the
    REM pipeline reports 0 however the child exited, so `exit $LASTEXITCODE`
    REM and `!ERRORLEVEL!` both faithfully forward a zero and a failed warmup
    REM is reported to CI as success. It then surfaces two steps later as
    REM build.bat's "Portable MSVC not found", which names neither the failure
    REM nor the step that had it. Measured on Windows Server 2025, A/B on one
    REM host: with the escaped-quote form a child exiting 1 gives 0, and with
    REM this form it gives 1. Every layer propagates correctly when tested
    REM alone (the Tee pipeline, the `set ... &&` chain, `exit /b` from a
    REM nested block), so the fault appears only in composition and cannot be
    REM found by reading any single line.
    REM
    REM `cmd /c call`, NOT `cmd /c`, for a failure signaled INSIDE A BLOCK. The
    REM relaunched script runs as `cmd /c` runs a batch file, and without `call`
    REM an `exit /b 1` executed inside a parenthesized block leaves cmd exiting
    REM 0, while one at the top level still gives 1, which is why the A/B above
    REM passed. Measured on Windows, invoked as GitHub's `shell: cmd` does:
    REM a nested `exit /b 1` gave step exit code 0 through `cmd /c '%~f0'` and 1
    REM through `cmd /c call '%~f0'`. The windows-11-arm release run showed what
    REM it costs: the tetgen wheel failed inside the ARM64 block, warmup.bat
    REM printed ERROR and exited 1, and the step passed, so the failure surfaced
    REM in bundle.bat as a missing source record.
    powershell -NoProfile -Command "& { $env:WARMUP_LOGGING='1'; $env:NOPAUSE='!NOPAUSE!'; cmd /c call '%~f0' 2>&1 | Tee-Object -FilePath '%LOGFILE%'; exit $LASTEXITCODE }"
    exit /b !ERRORLEVEL!
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

REM The architecture and the backends this run provisions for. Everything below
REM that differs between them reads what this sets, and a combination that cannot
REM be built is refused there by name before anything is downloaded.
call "%BUILD_WIN%\scripts\platform.bat"
if errorlevel 1 (
    echo ERROR: this host and PPF_WIN_BACKENDS do not describe a build this directory can make
    exit /b 1
)

set SEVENZIP_DIR=%BUILD_WIN%\7zip
set SEVENZIP=%SEVENZIP_DIR%\7z.exe
set MINGIT_DIR=%BUILD_WIN%\mingit
set MINGIT_EXE=%MINGIT_DIR%\cmd\git.exe
set CUDA_DIR=%BUILD_WIN%\cuda
set NVCC=%CUDA_DIR%\bin\nvcc.exe
set ROCM_DIR=%BUILD_WIN%\rocm
set WHEELS_DIR=%BUILD_WIN%\wheels
set MSVC_DIR=%BUILD_WIN%\msvc
REM portable-msvc.py names its environment script after the target. The x64
REM build also accepts the setup.bat an older revision of that script wrote.
set "MSVC_SETUP=%MSVC_DIR%\setup_%PPF_WIN_MSVC_ARCH%.bat"
if "%PPF_WIN_ARCH%"=="x64" (
    set "MSVC_SETUP=%MSVC_DIR%\setup.bat"
    if not exist "%MSVC_DIR%\setup.bat" set "MSVC_SETUP=%MSVC_DIR%\setup_x64.bat"
)
REM Windows' own bsdtar, named by path: an MSYS2 or Git tar earlier on PATH reads
REM a drive letter as a remote host.
set "TAR=%SystemRoot%\System32\tar.exe"

REM Pre-flight: every URL this run is about to fetch must be reachable before
REM the first download starts. Catches upstream rot at the manifest, not 3GB
REM into a fetch.
REM
REM ONLY WHAT THIS RUN FETCHES IS PROBED, which is the other architecture's and
REM the other backends' entries left out, and every tool already installed or
REM already in downloads\ left out too. A cached file with a digest is still
REM checked when it is used, by scripts\fetch-download.bat. The probe set on a
REM fresh machine is therefore every download that machine needs, as before, and
REM a machine that holds all of them provisions without any network.
set "PROBE="
REM Membership, not equality: a distribution can carry CUDA and ROCm together,
REM and then both toolkits are provisioned and both sets of URLs are probed.
set "HAS_CUDA=0"
set "HAS_ROCM=0"
for %%B in (!PPF_WIN_GPU_BACKENDS!) do (
    if "%%B"=="cuda" set "HAS_CUDA=1"
    if "%%B"=="rocm" set "HAS_ROCM=1"
)
if "!HAS_CUDA!"=="1" (
    if not exist "%SEVENZIP%" if not exist "%DOWNLOADS%\%FILE_7ZIP%" set "PROBE=!PROBE! URL_7ZIP"
    if not exist "%NVCC%" if not exist "%DOWNLOADS%\%FILE_CUDA%" set "PROBE=!PROBE! URL_CUDA"
    REM build.bat fetches Eigen, and it is probed here so that a rotted pointer
    REM stops the warmup rather than the build.
    for %%I in ("%FILE_EIGEN%") do if not exist "%BUILD_WIN%\deps\%%~nI" if not exist "%DOWNLOADS%\%FILE_EIGEN%" set "PROBE=!PROBE! URL_EIGEN"
)
if "!HAS_ROCM!"=="1" (
    if not exist "%ROCM_DIR%\bin\hipcc.exe" if not exist "%DOWNLOADS%\%FILE_ROCM_SDK%" set "PROBE=!PROBE! URL_ROCM_SDK"
)
if not exist "%MINGIT_EXE%" if not exist "%DOWNLOADS%\%FILE_MINGIT%" set "PROBE=!PROBE! URL_MINGIT_%PPF_WIN_ARCH_KEY%"
where cargo >nul 2>&1
if errorlevel 1 if not exist "%CARGO%" if not exist "%DOWNLOADS%\%FILE_RUSTUP%" set "PROBE=!PROBE! URL_RUSTUP_%PPF_WIN_ARCH_KEY%"
if not exist "%PYTHON%" if not exist "%DOWNLOADS%\%FILE_PYTHON%" set "PROBE=!PROBE! URL_PYTHON_%PPF_WIN_ARCH_KEY%"
if "%PPF_WIN_ARCH%"=="x64" if not exist "%PYTHON%" set "PROBE=!PROBE! URL_GET_PIP"
if not exist "%MSVC_SETUP%" if not exist "%DOWNLOADS%\%FILE_PORTABLE_MSVC%" set "PROBE=!PROBE! URL_PORTABLE_MSVC"
if "%PPF_WIN_ARCH%"=="arm64" (
    if not exist "%WHEELS_DIR%\triangle-*.whl" if not exist "%DOWNLOADS%\src\triangle\.git" set "PROBE=!PROBE! URL_TRIANGLE_GIT URL_TRIANGLE_C_GIT"
    if not exist "%WHEELS_DIR%\tetgen-*.whl" if not exist "%DOWNLOADS%\src\tetgen\.git" set "PROBE=!PROBE! URL_TETGEN_GIT"
)
if not exist "%BUILD_WIN%\ffmpeg\ffmpeg.exe" (
    if not exist "%BUILD_WIN%\msys64\usr\bin\bash.exe" if not exist "%DOWNLOADS%\%FILE_MSYS2%" set "PROBE=!PROBE! URL_MSYS2"
    set "PROBE=!PROBE! URL_FFMPEG_GIT URL_X264_GIT"
)
if defined PROBE (
    call "%BUILD_WIN%\scripts\check-downloads.bat" /nopause !PROBE!
    if errorlevel 1 (
        echo ERROR: One or more download URLs are unreachable.
        echo Fix the offending pointer^(s^) in scripts\downloads.txt and re-run.
        exit /b 1
    )
) else (
    echo Every download this run needs is already installed or in downloads\, so nothing is probed.
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
REM ONLY WHERE CUDA IS BUILT. Nothing else in this directory opens a 7-Zip
REM archive: the ROCm SDK and the ARM64 interpreter are gzip tarballs, which
REM Windows' own tar reads.
if "!HAS_CUDA!"=="1" (
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
)

REM ============================================================
REM Download and setup MinGit (used instead of full Git installation)
REM ============================================================
if not exist "%MINGIT_EXE%" (
    echo === Downloading MinGit ===

    call "%BUILD_WIN%\scripts\fetch-download.bat" MINGIT
    if errorlevel 1 exit /b 1
    set "MINGIT_ZIP=!DOWNLOADED_FILE!"

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
if "!HAS_CUDA!"=="1" (
    if not exist "%NVCC%" (
        echo === Downloading CUDA Toolkit ===

        set "CUDA_EXE=%DOWNLOADS%\%FILE_CUDA%"

        if not exist "!CUDA_EXE!" (
            echo Downloading CUDA ^(about 3GB, please wait^)...
            curl.exe -fL -o "!CUDA_EXE!" "%URL_CUDA%"
            if errorlevel 1 (
                echo ERROR: Failed to download CUDA Toolkit
                exit /b 1
            )
        )

        echo Extracting CUDA ^(this takes several minutes^)...
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
)

REM ============================================================
REM Unpack the ROCm SDK (no admin required)
REM ============================================================
REM ONLY WHERE ROCm IS BUILT. TheRock's self-contained Windows tarball carries
REM hipcc and the offload toolchain the backend library is compiled with, the
REM import library the solver links, and the HIP runtime DLL bundle.bat
REM redistributes, so nothing is installed and nothing touches the registry.
REM
REM SIX PATHS ARE LEFT IN THE TARBALL, and each is a math library this project
REM neither compiles against nor ships: MIOpen's DLLs and its device kernel
REM archives, rocBLAS and hipBLASLt's kernel directories, and the .kpack
REM payloads those libraries load. They are most of the SDK's size, which is
REM the reason to name them, since a CI runner's disk has to hold the whole
REM extraction beside the tarball. Anything not named is extracted, so a file the
REM toolchain turns out to need is present rather than filtered out by a guess.
if "!HAS_ROCM!"=="1" (
    if not exist "%ROCM_DIR%\bin\hipcc.exe" (
        echo === Unpacking the ROCm SDK ===
        call "%BUILD_WIN%\scripts\fetch-download.bat" ROCM_SDK
        if errorlevel 1 exit /b 1
        if exist "%ROCM_DIR%_temp" rmdir /s /q "%ROCM_DIR%_temp"
        mkdir "%ROCM_DIR%_temp"
        echo Extracting !DOWNLOADED_FILE! ^(this takes several minutes^)...
        "%TAR%" -xzf "!DOWNLOADED_FILE!" -C "%ROCM_DIR%_temp" --exclude "./.kpack" --exclude "./bin/MIOpen*" --exclude "./bin/rocblas" --exclude "./bin/hipblaslt" --exclude "./lib/device_*_operations.lib"
        if errorlevel 1 (
            echo ERROR: Failed to extract the ROCm SDK
            exit /b 1
        )
        REM Renamed into place only once the extraction finished, so an
        REM interrupted one is never taken for an installed SDK.
        move "%ROCM_DIR%_temp" "%ROCM_DIR%" >nul
        if errorlevel 1 (
            echo ERROR: could not move %ROCM_DIR%_temp to %ROCM_DIR%
            exit /b 1
        )
        echo ROCm SDK unpacked successfully!
    ) else (
        echo ROCm SDK already installed
    )
    REM CHECKED BY CONTENT, NOT BY THE hipcc THAT DECIDED WHETHER TO UNPACK. These
    REM are the files build.bat and bundle.bat use by name, so an SDK that lacks
    REM one fails here, where the cause is still visible.
    for %%F in (bin\hipcc.exe bin\amdhip64_7.dll lib\amdhip64.lib include\hip\hip_runtime.h lib\llvm\bin\clang.exe lib\llvm\bin\clang-offload-bundler.exe lib\llvm\bin\llvm-objdump.exe lib\llvm\bin\llvm-objcopy.exe share\doc\rocm-core\LICENSE.md share\hip\version share\therock\therock_manifest.json .info\version) do (
        if not exist "%ROCM_DIR%\%%F" (
            echo ERROR: the ROCm SDK at %ROCM_DIR% has no %%F
            echo        Remove %ROCM_DIR% and re-run warmup.bat.
            exit /b 1
        )
    )
    set "ROCM_SDK_VERSION="
    for /f "usebackq delims=" %%V in ("%ROCM_DIR%\.info\version") do set "ROCM_SDK_VERSION=%%V"
    echo ROCm SDK version: !ROCM_SDK_VERSION!
)

REM ============================================================
REM Install Rust locally if not available
REM ============================================================
where cargo >nul 2>&1
if errorlevel 1 (
    if not exist "%CARGO%" (
        echo === Installing Rust locally ===

        call "%BUILD_WIN%\scripts\fetch-download.bat" RUSTUP
        if errorlevel 1 exit /b 1
        set "RUSTUP_INIT=!DOWNLOADED_FILE!"

        echo Installing Rust to %RUST_DIR%...
        set RUSTUP_HOME=%RUST_DIR%\rustup
        set CARGO_HOME=%RUST_DIR%
        REM --default-host is stated rather than left to the installer's guess,
        REM which is the installer's own architecture.
        "!RUSTUP_INIT!" -y --no-modify-path --default-host !PPF_WIN_RUST_HOST! --default-toolchain stable
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

REM THE TOOLCHAIN MUST BE THIS HOST'S. build.bat prefers the local installation and
REM falls back to a cargo on PATH, and either can be for the wrong architecture: a
REM rustup started under emulation installs an x64 toolchain on Windows on ARM, and
REM that toolchain builds an x64 solver that passes every later check. So the rustc
REM build.bat will use is asked which host it is. Rust is Tier 1 on
REM aarch64-pc-windows-msvc, so the ARM64 answer is an ordinary installation.
if exist "%CARGO%" (
    set "RUSTUP_HOME=%RUST_DIR%\rustup"
    set "CARGO_HOME=%RUST_DIR%"
    set "RUSTC_CHECK=%RUST_DIR%\bin\rustc.exe"
) else (
    set "RUSTC_CHECK=rustc"
)
set "RUST_HOST_SEEN="
for /f "tokens=1,2" %%A in ('"%RUSTC_CHECK%" -vV') do if "%%A"=="host:" set "RUST_HOST_SEEN=%%B"
if not "!RUST_HOST_SEEN!"=="!PPF_WIN_RUST_HOST!" (
    echo ERROR: !RUSTC_CHECK! reports host "!RUST_HOST_SEEN!", and this is a !PPF_WIN_RUST_HOST! build.
    echo        Remove the toolchain it belongs to, or put a native one first on PATH,
    echo        and re-run warmup.bat.
    exit /b 1
)
echo Rust host: !RUST_HOST_SEEN!

REM ============================================================
REM Download and setup the bundled Python if not present
REM ============================================================
if not exist "%PYTHON%" (
    if "!PPF_WIN_ARCH!"=="arm64" (
        REM python-build-standalone's install_only tree: a full CPython layout
        REM with Lib\, DLLs\, libs\python3.lib, include\ and pip already in
        REM site-packages, all under one top-level python\ directory.
        echo === Downloading python-build-standalone CPython ===

        call "%BUILD_WIN%\scripts\fetch-download.bat" PYTHON
        if errorlevel 1 exit /b 1

        echo Extracting Python...
        if exist "%PYTHON_DIR%_temp" rmdir /s /q "%PYTHON_DIR%_temp"
        mkdir "%PYTHON_DIR%_temp"
        "%TAR%" -xzf "!DOWNLOADED_FILE!" -C "%PYTHON_DIR%_temp"
        if errorlevel 1 (
            echo ERROR: Failed to extract Python
            exit /b 1
        )
        if not exist "%PYTHON_DIR%_temp\python\python.exe" (
            echo ERROR: the python-build-standalone archive holds no python\python.exe
            exit /b 1
        )
        move "%PYTHON_DIR%_temp\python" "%PYTHON_DIR%" >nul
        if errorlevel 1 (
            echo ERROR: could not move the extracted interpreter to %PYTHON_DIR%
            exit /b 1
        )
        rmdir /s /q "%PYTHON_DIR%_temp"

        echo python-build-standalone CPython setup complete!
    ) else (
        echo === Downloading Embedded Python ===

        call "%BUILD_WIN%\scripts\fetch-download.bat" PYTHON
        if errorlevel 1 exit /b 1
        set "PYTHON_ZIP=!DOWNLOADED_FILE!"

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
)

REM Check if Python exists
if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    echo Please ensure the python directory exists with a Python installation.
    exit /b 1
)

echo === Checking Python ===
"%PYTHON%" --version
if errorlevel 1 (
    echo ERROR: Python check failed
    exit /b 1
)
REM THE INTERPRETER MUST BE THIS HOST'S, for the reason the Rust check above
REM gives: every wheel pip installs into it takes its architecture from it.
REM platform.machine() reports the architecture the interpreter was built for,
REM spelled as PROCESSOR_ARCHITECTURE spells it.
"%PYTHON%" -c "import platform, sys; m = platform.machine(); print('Interpreter architecture: ' + m); sys.exit(0 if m.upper() == '%PROCESSOR_ARCHITECTURE%'.upper() else 1)"
if errorlevel 1 (
    echo ERROR: %PYTHON% is not a %PROCESSOR_ARCHITECTURE% interpreter. Remove %PYTHON_DIR% and re-run warmup.bat.
    exit /b 1
)
if "!PPF_WIN_ARCH!"=="arm64" (
    REM What PyO3 links _ppf_cts_py.dll against and what a source-built wheel
    REM compiles against, both in the one tree on ARM64.
    for %%F in (python3.dll libs\python3.lib include\Python.h) do (
        if not exist "%PYTHON_DIR%\%%F" (
            echo ERROR: %PYTHON_DIR% has no %%F. Remove %PYTHON_DIR% and re-run warmup.bat.
            exit /b 1
        )
    )
    "%PYTHON%" -m pip --version
    if errorlevel 1 (
        echo ERROR: the interpreter at %PYTHON% carries no pip
        exit /b 1
    )
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
REM Windows registry, py launcher, or %LOCALAPPDATA%\Programs\Python;
REM it is a self-contained zip that lands at <out>\python\tools\
REM with python.exe + libs\python3.lib + include\Python.h alongside.
REM We move tools\ to python_full\ and it's purely a build-time tool
REM here; bundle.bat does NOT include python_full\ in dist (only the
REM stripped embedded python\ ships with releases).
REM
REM x64 ONLY. The ARM64 interpreter above is a full CPython tree that already
REM carries libs\python3.lib, so it is the PyO3 build interpreter as well.
set PYTHON_FULL_DIR=%BUILD_WIN%\python_full
set PYTHON_FULL=%PYTHON_FULL_DIR%\python.exe
if "!PPF_WIN_ARCH!"=="x64" (
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
)

REM ============================================================
REM Install Portable MSVC (no admin required)
REM Uses: https://gist.github.com/mmozeiko/7f3162ec2988e81e56d5c4e22cde9977
REM ============================================================
REM THE HOST AND TARGET ARE THIS ARCHITECTURE'S. portable-msvc.py defaults both
REM to x64, which on Windows on ARM installs a compiler that runs under
REM emulation and emits x64 objects, so both are stated for every architecture.
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

    echo Installing MSVC to %MSVC_DIR% ^(this takes a while^)...
    pushd "%BUILD_WIN%"
    REM The compiler and SDK versions come from the manifest, not from
    REM portable-msvc.py's `max()` over whatever Microsoft publishes today.
    if not defined MSVC_VERSION (
        echo ERROR: MSVC_VERSION is not set by scripts\downloads.txt
        exit /b 1
    )
    if not defined WINDOWS_SDK_VERSION (
        echo ERROR: WINDOWS_SDK_VERSION is not set by scripts\downloads.txt
        exit /b 1
    )
    echo Pinned toolchain: MSVC !MSVC_VERSION!, Windows SDK !WINDOWS_SDK_VERSION!, host and target !PPF_WIN_MSVC_ARCH!
    "%PYTHON%" "!PBT_SCRIPT!" --accept-license --vs 2022 --msvc-version !MSVC_VERSION! --sdk-version !WINDOWS_SDK_VERSION! --host !PPF_WIN_MSVC_ARCH! --target !PPF_WIN_MSVC_ARCH!
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
    if "!PPF_WIN_ARCH!"=="x64" (
        if not exist "%MSVC_DIR%\setup_x64.bat" if not exist "%MSVC_DIR%\setup.bat" (
            echo ERROR: portable-msvc.py finished but no setup_x64.bat/setup.bat under %MSVC_DIR%
            exit /b 1
        )
    ) else (
        if not exist "%MSVC_DIR%\setup_!PPF_WIN_MSVC_ARCH!.bat" (
            echo ERROR: portable-msvc.py finished but no setup_!PPF_WIN_MSVC_ARCH!.bat under %MSVC_DIR%
            exit /b 1
        )
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
    REM _build_harmonic_interior_operator) uses scipy.sparse solves, and a
    REM frontend without scipy refuses to build such a scene, naming scipy.
    REM scipy is present transitively on Linux but not here, so it must be
    REM listed explicitly.
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
    REM
    REM pillow: frontend/_rasterizer_.py imports PIL at top level, so without it
    REM `import frontend` fails. matplotlib and pyvista both depend on it, which
    REM is how the x64 set always had it, so a set without those two needs it
    REM named. Both sets name it, pinned at the version the x64 set was already
    REM shipping, so neither leans on a dependency of a package the frontend
    REM does not import.
    REM
    REM THE ARM64 SET IS THE SAME SET LESS WHAT HAS NO win_arm64 WHEEL AND IS
    REM IMPORTED BY NOTHING: numba, matplotlib and fast-simplification. triangle
    REM has no win_arm64 wheel either and IS imported, so it is built from its
    REM pinned upstream source below rather than dropped. Read on PyPI on
    REM 2026-09-15: every pinned native package that remains publishes a cp312 or
    REM abi3 win_arm64 wheel at its pinned version, pillow 12.3.0 included.
    if "!PPF_WIN_ARCH!"=="arm64" (
        set PACKAGES=numpy==2.4.4 scipy==1.17.1 pillow==12.3.0 plyfile requests certifi gdown trimesh==4.12.2 pywavefront tqdm pythreejs ipywidgets tabulate cbor2==6.0.1 psutil==7.2.2
    ) else (
        set PACKAGES=numpy==2.4.4 scipy==1.17.1 pillow==12.3.0 numba==0.65.1 plyfile requests certifi gdown trimesh==4.12.2 pywavefront matplotlib tqdm pythreejs ipywidgets fast-simplification==0.1.13 tabulate triangle==20250106 cbor2==6.0.1 psutil==7.2.2
    )

    REM Development tools
    set DEV_PACKAGES=ruff black isort

    REM JupyterLab (LSP disabled on Windows due to embedded Python subprocess issues)
    REM nbconvert is needed for fast-check-all.bat to convert notebooks to Python scripts
    set JUPYTER_PACKAGES=jupyterlab jupyterlab-code-formatter nbconvert

    echo.
    echo Installing core packages...
    "%PYTHON%" -m pip install --no-warn-script-location !PACKAGES!
    if errorlevel 1 (
        echo ERROR: the core packages did not install ^(see pip's output above^)
        exit /b 1
    )

    if "!PPF_WIN_ARCH!"=="arm64" (
        REM THE TETRAHEDRALIZERS, AND WHAT ARM64 DOES WITHOUT ONE OF THEM.
        REM TetGen and triangle publish no win_arm64 wheel, so each is built
        REM from its pinned upstream commit by scripts\source-wheel.py, which also
        REM counts fused multiply-add instructions in the result, since both carry
        REM exact geometric predicates. The compiler is this architecture's
        REM portable MSVC, whose environment is loaded for the build only.
        REM
        REM pytetwild publishes no win_arm64 wheel, and its source build needs
        REM fTetWild, geogram, oneTBB and GMP under MSVC for ARM64, which nothing
        REM in this directory builds. So the ARM64 distribution ships WITHOUT
        REM fTetWild, and frontend/_mesh_.py refuses backend="ftetwild" there by
        REM name rather than switching to TetGen: the two meshers produce
        REM different meshes, so a quiet switch would be a different scene.
        echo.
        echo Building triangle and tetgen from pinned upstream source...
        setlocal
        call "%MSVC_SETUP%" >nul
        for %%W in (triangle tetgen) do (
            "%PYTHON%" -u "%BUILD_WIN%\scripts\source-wheel.py" %%W "%WHEELS_DIR%"
            if errorlevel 1 (
                echo ERROR: the %%W wheel could not be built from its pinned source ^(see above^)
                exit /b 1
            )
        )
        endlocal
        REM --no-deps: each wheel's one runtime dependency, numpy, is installed
        REM above at its pin, and --no-index keeps the install to the wheels just
        REM verified.
        "%PYTHON%" -m pip install --no-warn-script-location --no-index --no-deps --find-links "%WHEELS_DIR%" triangle tetgen
        if errorlevel 1 (
            echo ERROR: the source-built triangle and tetgen wheels did not install
            exit /b 1
        )
    ) else (
        echo.
        echo Installing pytetwild ^(fTetWild^) + tetgen ^(TetGen^) + pyvista...
        rem pyvista rides along so the three provisioning paths carry one
        rem dependency set; nothing in this repository imports it, and neither
        rem tetrahedralizer needs it for the numpy entry points the frontend
        rem calls. Dropping it means dropping it from warmup.py and
        rem .github/workflows/blender.yml in the same change, or the paths
        rem disagree about what a provisioned host holds.
        rem pytetwild is pinned to the version every bundle ships, so a scene
        rem tetrahedralizes the same way on each. 0.4.2 publishes a
        rem cp311-cp311-win_amd64 wheel, which is this interpreter's.
        "%PYTHON%" -m pip install --no-warn-script-location pytetwild==0.4.2 tetgen==0.8.4 pyvista==0.48.4
        if errorlevel 1 (
            echo ERROR: pytetwild, tetgen and pyvista did not install ^(see pip's output above^)
            exit /b 1
        )
    )

    echo.
    echo === Verifying critical frontend dependencies ===
    REM pip can exit 0 and still leave a package out, through a partial index
    REM or a resolver backtrack (observed: pytetwild installed but tetgen/pyvista
    REM missing; or scipy absent), and that would ship a bundle whose build
    REM worker cannot run a SOLID simulation. tetgen/pytetwild missing =
    REM ModuleNotFoundError at build time; scipy missing is worse, it does not
    REM crash but makes the frontend take a different pin-diffusion path so the
    REM Windows result silently diverges from Linux. pyvista is checked
    REM because this list is what the install above promises, not because
    REM anything imports it. Hard-fail here so a broken
    REM bundle can never be built or published; re-run warmup on a working pip.
    REM The ARM64 list is what the ARM64 install promises, which is triangle and
    REM tetgen and not pytetwild or pyvista, for the reasons given there.
    if "!PPF_WIN_ARCH!"=="arm64" (
        set "CRITICAL_DEPS='numpy','scipy','PIL','cbor2','tetgen','triangle'"
    ) else (
        set "CRITICAL_DEPS='numpy','scipy','PIL','cbor2','tetgen','pytetwild','pyvista'"
    )
    "%PYTHON%" -c "import importlib.util as u, sys; req=[!CRITICAL_DEPS!]; miss=[m for m in req if u.find_spec(m) is None]; sys.stderr.write('missing critical frontend deps: '+', '.join(miss)+'\n') if miss else sys.stdout.write('all critical frontend deps present\n'); sys.exit(1 if miss else 0)"
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
        echo ERROR: the development tools did not install ^(see pip's output above^)
        exit /b 1
    )

    echo.
    echo Installing JupyterLab packages...
    "%PYTHON%" -m pip install --no-warn-script-location !JUPYTER_PACKAGES!
    if errorlevel 1 (
        echo ERROR: JupyterLab did not install ^(see pip's output above^). The distribution's launcher starts it.
        exit /b 1
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
        echo ERROR: the slim FFmpeg build failed ^(see above^). The distribution's video export needs it.
        exit /b 1
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
