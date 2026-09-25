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
    REM and `!ERRORLEVEL!` both faithfully forward a zero and the failure is
    REM masked. Measured on Windows Server 2025, A/B on one host: with the
    REM escaped-quote form a child exiting 1 gives 0, and with this form it
    REM gives 1. Every layer here propagates correctly when tested alone (the
    REM Tee pipeline, the `set ... &&` chain, `exit /b` from a nested block),
    REM so the fault appears only in composition and cannot be found by
    REM reading any single line.
    REM
    REM `cmd /c call`, NOT `cmd /c`, for a failure signaled INSIDE A BLOCK:
    REM without `call` an `exit /b 1` executed inside a parenthesized block
    REM leaves cmd exiting 0, while a top-level one still gives 1. Measured on
    REM Windows, invoked as GitHub's `shell: cmd` does: step exit code 0
    REM through `cmd /c '%~f0'`, 1 through `cmd /c call '%~f0'`. warmup.bat
    REM records the release run that found it.
    powershell -NoProfile -Command "& { $env:BUILD_LOGGING='1'; $env:NOPAUSE='!NOPAUSE!'; cmd /c call '%~f0' 2>&1 | Tee-Object -FilePath '%LOGFILE%'; exit $LASTEXITCODE }"
    exit /b !ERRORLEVEL!
)

echo ============================================================
echo   ZOZO's Contact Solver - Windows Build Script
echo ============================================================
echo.
REM Get parent directory (SRC_DIR)
for %%I in ("%BUILD_WIN%\..") do set SRC_DIR=%%~fI

REM The CUDA backend's directories. scripts\build-cuda.bat, which builds that
REM library, describes them; they are set here because the Rust step and the
REM launchers below name them too.
set CPP_DIR=%SRC_DIR%\crates\ppf-cts-compute\cuda
set KERNEL_DIR=%SRC_DIR%\crates\ppf-cts-solver\src\kernels
set KERNELGEN=%SRC_DIR%\crates\ppf-cts-compute\seam\kernelgen.py
set OUT_DIR=%CPP_DIR%\build
set KERNELGEN_DIR=%OUT_DIR%\kernelgen
set LIB_DIR=%OUT_DIR%\lib
REM The ROCm backend's. crates\ppf-cts-solver\build.rs links libppfbe_rocm out of
REM ROCM_LIB_DIR on Windows and builds nothing there itself, so the library has to
REM exist before cargo runs, exactly as the CUDA one does.
set ROCM_SRC_DIR=%SRC_DIR%\crates\ppf-cts-compute\rocm
set ROCM_OUT_DIR=%ROCM_SRC_DIR%\build
set ROCM_LIB_DIR=%ROCM_OUT_DIR%\lib
set ROCM_DIR=%BUILD_WIN%\rocm
set DEPS=%BUILD_WIN%\deps
set DOWNLOADS=%BUILD_WIN%\downloads
set RUST_DIR=%BUILD_WIN%\rust
set CUDA_DIR=%BUILD_WIN%\cuda

call "%BUILD_WIN%\scripts\load-downloads.bat"
if errorlevel 1 (
    echo ERROR: Failed to load download manifest
    exit /b 1
)
REM The architecture and the backends this build makes, decided for warmup.bat,
REM this script and bundle.bat in one place.
call "%BUILD_WIN%\scripts\platform.bat"
if errorlevel 1 (
    echo ERROR: this host and PPF_WIN_BACKENDS do not describe a build this directory can make
    exit /b 1
)

REM ============================================================
REM The ROCm platform
REM ============================================================
REM ONE HIP SOURCE BUILDS FOR TWO PLATFORMS, and they are different libraries.
REM The AMD platform is what ships: hipcc compiles it for every target in
REM rocm_arch.txt, and nothing in this fleet can execute it. The NVIDIA platform
REM compiles the same source through nvcc and runs on an NVIDIA GPU, which makes it
REM the only way the Windows ROCm build can be EXECUTED without AMD hardware; it
REM is a verification lever and never a distribution, which is why bundle.bat
REM refuses a library built for it. A green NVIDIA-platform build establishes
REM that this HIP source compiles and runs, and nothing about the code hipcc
REM generates for the AMD targets.
REM
REM   PPF_WIN_ROCM_PLATFORM      amd (the default) or nvidia
REM   PPF_WIN_HIP_ROOT           nvidia only: a directory whose include\hip holds
REM                              nvidia_detail, which TheRock 10.0.0 does not
REM                              carry and ROCm 7.2.4's hip-dev package does
REM   PPF_WIN_ROCM_NVIDIA_ARCH   nvidia only: the one compute capability the staging
REM                              library is compiled for, sm_89 unless given
REM Membership, not equality: this build can carry CUDA and ROCm together.
set "HAS_CUDA=0"
set "HAS_ROCM=0"
for %%B in (!PPF_WIN_GPU_BACKENDS!) do (
    if "%%B"=="cuda" set "HAS_CUDA=1"
    if "%%B"=="rocm" set "HAS_ROCM=1"
)
set "ROCM_PLATFORM="
if defined PPF_WIN_ROCM_PLATFORM if "!HAS_ROCM!"=="0" (
    echo ERROR: PPF_WIN_ROCM_PLATFORM is set, and rocm is not one of the backends ^(!PPF_WIN_BACKENDS!^).
    echo        Set PPF_WIN_BACKENDS to a set naming rocm, or clear PPF_WIN_ROCM_PLATFORM.
    exit /b 1
)
if "!HAS_ROCM!"=="1" (
    set "ROCM_PLATFORM=amd"
    if defined PPF_WIN_ROCM_PLATFORM set "ROCM_PLATFORM=!PPF_WIN_ROCM_PLATFORM!"
)
if "!ROCM_PLATFORM!"=="amd" (
    if not exist "%ROCM_DIR%\bin\hipcc.exe" (
        echo ERROR: the ROCm SDK is not installed at %ROCM_DIR%
        echo        Run warmup.bat with PPF_WIN_BACKENDS=rocm cpu first.
        exit /b 1
    )
    set "HIP_ROOT=%ROCM_DIR%"
) else if "!ROCM_PLATFORM!"=="nvidia" (
    if not defined PPF_WIN_HIP_ROOT (
        echo ERROR: PPF_WIN_ROCM_PLATFORM=nvidia needs PPF_WIN_HIP_ROOT, a directory whose
        echo        include\hip holds nvidia_detail. See the note above this check.
        exit /b 1
    )
    if not exist "!PPF_WIN_HIP_ROOT!\include\hip\nvidia_detail\nvidia_hip_runtime.h" (
        echo ERROR: PPF_WIN_HIP_ROOT=!PPF_WIN_HIP_ROOT! has no include\hip\nvidia_detail\nvidia_hip_runtime.h
        exit /b 1
    )
    set "HIP_ROOT=!PPF_WIN_HIP_ROOT!"
    set "ROCM_NVIDIA_ARCH=sm_89"
    if defined PPF_WIN_ROCM_NVIDIA_ARCH set "ROCM_NVIDIA_ARCH=!PPF_WIN_ROCM_NVIDIA_ARCH!"
) else if defined ROCM_PLATFORM (
    echo ERROR: PPF_WIN_ROCM_PLATFORM names "!ROCM_PLATFORM!", and the platforms are amd and nvidia.
    exit /b 1
)

REM Use local CUDA installed by warmup.bat, wherever something here compiles with
REM nvcc: the CUDA backend, and the ROCm backend's NVIDIA staging platform.
set "NEED_CUDA=0"
if "!HAS_CUDA!"=="1" set "NEED_CUDA=1"
if "!ROCM_PLATFORM!"=="nvidia" set "NEED_CUDA=1"
if "!NEED_CUDA!"=="1" (
    set "CUDA_PATH=%CUDA_DIR%"
    if not exist "%CUDA_DIR%\bin\nvcc.exe" (
        echo ERROR: Local CUDA not found at %CUDA_DIR%
        echo Please run warmup.bat first to install CUDA locally.
        exit /b 1
    )
    echo Using local CUDA from %CUDA_DIR%
)

REM The bundled Python from warmup.bat. Runs JupyterLab via the launcher
REM scripts written below. The PyO3 extension (_ppf_cts_py.dll) is built into
REM each backend's target\<backend>\release by that backend's `cargo build
REM --release` and loaded by frontend/__init__.py from the directory it
REM resolves, so no wheel is installed here.
set PYTHON_EXE=%BUILD_WIN%\python\python.exe
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python not found at %PYTHON_EXE%
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
REM THE TOOLCHAIN MUST BE THIS HOST'S, which warmup.bat checks when it installs one
REM and is checked again here because a cargo on PATH can change between the two.
set "RUST_HOST_SEEN="
for /f "tokens=1,2" %%A in ('rustc -vV') do if "%%A"=="host:" set "RUST_HOST_SEEN=%%B"
if not "!RUST_HOST_SEEN!"=="!PPF_WIN_RUST_HOST!" (
    echo ERROR: rustc reports host "!RUST_HOST_SEEN!", and this is a !PPF_WIN_RUST_HOST! build.
    echo        A toolchain for another architecture builds a solver that runs only under
    echo        emulation and passes every check here. Re-run warmup.bat.
    exit /b 1
)
echo Rust host: !RUST_HOST_SEEN!

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
REM Download Eigen if not present (the CUDA build only)
REM ============================================================
REM Eigen extracts to a directory whose name matches the archive stem
for %%I in ("%FILE_EIGEN%") do set EIGEN_STEM=%%~nI
if "!HAS_CUDA!"=="1" (
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
)

REM Setup portable MSVC environment (required)
set MSVC_DIR=%BUILD_WIN%\msvc
echo [1/4] Setting up Visual Studio environment...
if "!PPF_WIN_ARCH!"=="x64" (
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
) else (
    if exist "%MSVC_DIR%\setup_!PPF_WIN_MSVC_ARCH!.bat" (
        echo Using portable MSVC from %MSVC_DIR%
        call "%MSVC_DIR%\setup_!PPF_WIN_MSVC_ARCH!.bat"
    ) else (
        echo ERROR: Portable MSVC for !PPF_WIN_MSVC_ARCH! not found at %MSVC_DIR%\setup_!PPF_WIN_MSVC_ARCH!.bat
        echo Please run warmup.bat first to install MSVC locally.
        exit /b 1
    )
)
REM THE COMPILER MUST TARGET THIS HOST. cl.exe names its target at the end of its
REM banner, "for x64" or "for ARM64", and a setup script for another target would
REM put that compiler first on PATH with nothing else changing.
REM Percent expansion, not delayed: each side of a pipe runs in a child cmd.exe
REM with delayed expansion off, where `!name!` would reach findstr as text.
cl 2>&1 | findstr /i /c:"for %PPF_WIN_MSVC_ARCH%" >nul
if errorlevel 1 (
    echo ERROR: cl.exe on PATH does not target !PPF_WIN_MSVC_ARCH!. Its banner says:
    cl 2>&1 | findstr /i /c:"compiler"
    exit /b 1
)
echo cl.exe targets !PPF_WIN_MSVC_ARCH!

REM ============================================================
REM [2/4] The GPU backend library
REM ============================================================
REM EACH LIBRARY IS BUILT ON ITS OWN TERMS, and a build carrying both builds both:
REM they are separate files with separate toolchains, and neither is an
REM alternative to the other.
if "!HAS_CUDA!"=="1" (
    call "%BUILD_WIN%\scripts\build-cuda.bat"
    if errorlevel 1 (
        echo ERROR: the CUDA backend library did not build ^(see above^)
        exit /b 1
    )
)
if "!HAS_ROCM!"=="1" (
    echo.
    echo ============================================================
    echo [2/4] Building the ROCm backend library for the !ROCM_PLATFORM! platform
    echo ============================================================
    echo.
    REM hipcc drives the SDK's own clang and the offload tools beside it. -u
    REM because the script's progress lines otherwise reach build.log only when
    REM the build ends: Python buffers a pipe, and the logging relaunch is one.
    if "!ROCM_PLATFORM!"=="amd" (
        set "PATH=%ROCM_DIR%\bin;%ROCM_DIR%\lib\llvm\bin;!PATH!"
        "%PYTHON_EXE%" -u "%BUILD_WIN%\scripts\build_rocm.py" --platform amd --kernel-root "%KERNEL_DIR%" --rocm-dir "%ROCM_SRC_DIR%" --cuda-prologue-dir "%CPP_DIR%" --kernelgen "%KERNELGEN%" --gen-def "%BUILD_WIN%\scripts\gen_def.py" --out-dir "%ROCM_OUT_DIR%" --hip-include "%ROCM_DIR%\include" --rocm-path "%ROCM_DIR%"
    ) else (
        "%PYTHON_EXE%" -u "%BUILD_WIN%\scripts\build_rocm.py" --platform nvidia --kernel-root "%KERNEL_DIR%" --rocm-dir "%ROCM_SRC_DIR%" --cuda-prologue-dir "%CPP_DIR%" --kernelgen "%KERNELGEN%" --gen-def "%BUILD_WIN%\scripts\gen_def.py" --out-dir "%ROCM_OUT_DIR%" --hip-include "!HIP_ROOT!\include" --cuda-path "%CUDA_DIR%" --arch !ROCM_NVIDIA_ARCH!
    )
    if errorlevel 1 (
        echo ERROR: the ROCm backend library did not build ^(see above^)
        exit /b 1
    )
    REM ASK THE LIBRARY WHAT IT IS. It must export the whole C ABI the module
    REM definition lists, since the solver imports every be_* by name, and it must
    REM import its own platform's device runtime and not the other's: the two
    REM platforms write the same file name, so the name says nothing about which
    REM one is on disk.
    if "!ROCM_PLATFORM!"=="amd" (
        "%PYTHON_EXE%" "%BUILD_WIN%\scripts\pe-audit.py" backend --arch !PPF_WIN_ARCH! --dll "%ROCM_LIB_DIR%\libppfbe_rocm.dll" --def "%ROCM_OUT_DIR%\backend-exports.def" --must-import amdhip64_7.dll --must-not-import cudart64_12.dll
    ) else (
        "%PYTHON_EXE%" "%BUILD_WIN%\scripts\pe-audit.py" backend --arch !PPF_WIN_ARCH! --dll "%ROCM_LIB_DIR%\libppfbe_rocm.dll" --def "%ROCM_OUT_DIR%\backend-exports.def" --must-import cudart64_12.dll --must-not-import amdhip64_7.dll
    )
    if errorlevel 1 exit /b 1
    REM AND ASK THE DEVICE IMAGE WHICH TARGETS IT CARRIES, on the platform that
    REM ships. Every target in rocm_arch.txt must be in the linked library, each
    REM must disassemble to real instructions, and none may carry FP64: the same
    REM gate the Linux build and .github/workflows/rocm.yml run.
    if "!ROCM_PLATFORM!"=="amd" (
        "%PYTHON_EXE%" "%SRC_DIR%\.github\workflows\scripts\check-rocm-code-objects.py" --library "%ROCM_LIB_DIR%\libppfbe_rocm.dll" --arch-file "%ROCM_SRC_DIR%\rocm_arch.txt" --rocm-path "%ROCM_DIR%\lib"
        if errorlevel 1 (
            echo ERROR: the ROCm device image does not carry what rocm_arch.txt names ^(see above^)
            exit /b 1
        )
    )
)
if not defined PPF_WIN_GPU_BACKENDS (
    echo.
    echo [2/4] No GPU backend library: this build is !PPF_WIN_BACKENDS! on !PPF_WIN_ARCH!.
)

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
REM must link the import library at build time. On ARM64 the bundled
REM interpreter is a full CPython tree carrying libs\python3.lib, so it is
REM the build interpreter too.
if "!PPF_WIN_ARCH!"=="x64" (
    set "PYTHON_FULL_EXE=%BUILD_WIN%\python_full\python.exe"
) else (
    set "PYTHON_FULL_EXE=%PYTHON_EXE%"
)
if not exist "!PYTHON_FULL_EXE!" (
    echo ERROR: Full Python not found at !PYTHON_FULL_EXE!
    echo Please run warmup.bat first ^(it installs the interpreter PyO3 links against^).
    exit /b 1
)
set "PYO3_PYTHON=!PYTHON_FULL_EXE!"

REM The Rust build scripts render the transcompiler by invoking `python3`
REM (Command::new("python3"), the name that exists on Linux and macOS). The
REM embedded interpreter ships only python.exe, so on Windows a bare `python3`
REM resolves to the App-execution-alias store stub and the render panics with
REM "Python was not found". Provide a real python3.exe (a copy of the embedded
REM interpreter, which already renders the CUDA side in step 2) and put its
REM directory first on PATH so it wins over the stub.
copy /y "%BUILD_WIN%\python\python.exe" "%BUILD_WIN%\python\python3.exe" >nul
set "PATH=%BUILD_WIN%\python;%PATH%"

REM EVERY BACKEND BUILDS INTO ITS OWN target\<backend>, which is what lets one
REM distribution carry several: they link the same executable name, so
REM crates\ppf-cts-solver\build.rs refuses to put two in one directory. Each
REM build sets CARGO_TARGET_DIR for itself, and an inherited value is cleared
REM first rather than obeyed, since it would send one backend's artifacts where
REM another's checks look.
set "CARGO_TARGET_DIR="

if "!HAS_CUDA!"=="1" (
    set "CARGO_TARGET_DIR=%SRC_DIR%\target\cuda"
    REM ppf-cts-compute's build script (a build-dependency here) emits a link against
    REM libsimbackend_cuda for its dependents' BUILD SCRIPTS on Windows, so those
    REM build-script executables import be_* from the backend DLL. Windows resolves
    REM a static import at executable LOAD time even when the function is never
    REM called, so the build script cannot start unless the DLL is on the search
    REM path when cargo runs it. Put the freshly built lib directory on PATH; cudart,
    REM the DLL's own dependency, is already there from the CUDA setup above.
    set "PATH=%LIB_DIR%;!PATH!"

    cargo build --release --features cuda
    if errorlevel 1 (
        echo ERROR: Rust build failed
        set "CARGO_TARGET_DIR="
        exit /b 1
    )

    REM ONE WORKSPACE BUILD, NOT A SECOND `-p ppf-cts-server` ONE, which is the
    REM rule the ROCm arm below states for its own reason. The build above
    REM already produces ppf-cts-server, a workspace default-member beside
    REM ppf-cts-solver and ppf-cts-py, and the Blender addon's Windows Native
    REM launcher (blender_addon/core/connection.py:spawn_win_native_server)
    REM spawns ppf-cts-server.exe out of whichever backend directory it
    REM resolves. A second `-p ppf-cts-server --features cuda` build is not a
    REM guard but an ERROR: the backend features are declared on
    REM ppf-cts-solver, while ppf-cts-server declares only `cpu`, so cargo
    REM refuses with "the package 'ppf-cts-server' does not contain this
    REM feature: cuda". The guard is the existence check below, which fails the
    REM build when either binary is missing.
    set "CARGO_TARGET_DIR="
)
if "!HAS_ROCM!"=="1" (
    set "CARGO_TARGET_DIR=%SRC_DIR%\target\rocm"
    REM THE SAME LOAD-TIME RULE, TWO LEVELS DEEP. The build scripts import be_*
    REM from libppfbe_rocm.dll, and that DLL imports its platform's runtime, so both
    REM directories go on PATH: the SDK's bin, which holds amdhip64_7.dll, for the
    REM AMD platform, and the CUDA toolkit's for the staging one.
    REM ppf-cts-core compiles the check_gpu probe against the SDK named by
    REM ROCM_PATH, for the platform HIP_PLATFORM names.
    set "ROCM_PATH=!HIP_ROOT!"
    set "HIP_PATH=!HIP_ROOT!"
    set "HIP_PLATFORM=!ROCM_PLATFORM!"
    if "!ROCM_PLATFORM!"=="amd" (
        set "PATH=%ROCM_LIB_DIR%;%ROCM_DIR%\bin;!PATH!"
    ) else (
        set "PATH=%ROCM_LIB_DIR%;%CUDA_DIR%\bin;!PATH!"
    )

    REM ONE WORKSPACE BUILD, NOT A SECOND `-p ppf-cts-server` ONE. The server and
    REM the cdylib take ppf-cts-core's rocm feature from the solver through feature
    REM unification in this invocation; a separate server build without the
    REM feature would replace target\rocm\release\ppf-cts-server.exe with one
    REM whose check_gpu asks for an NVIDIA device.
    cargo build --release --features rocm
    if errorlevel 1 (
        echo ERROR: Rust build failed
        set "CARGO_TARGET_DIR="
        exit /b 1
    )
    set "CARGO_TARGET_DIR="
)

REM EACH BACKEND'S THREE ARTIFACTS, in its own directory. The addon spawns the
REM chosen directory's own ppf-cts-server, and the build worker it starts loads
REM the cdylib beside it, so a missing one of the three is a backend that cannot
REM be selected. frontend/__init__.py loads that cdylib by absolute path, which
REM is why no wheel is installed into the bundled Python.
for %%B in (!PPF_WIN_GPU_BACKENDS!) do (
    for %%F in (ppf-contact-solver.exe ppf-cts-server.exe _ppf_cts_py.dll) do (
        if not exist "%SRC_DIR%\target\%%B\release\%%F" (
            echo ERROR: target\%%B\release\%%F not found after build
            exit /b 1
        )
    )
    echo   [DONE] the %%B backend built into target\%%B\release
)

REM ASK THE BINARY WHETHER IT ACTUALLY IMPORTS THE GPU LIBRARY. Naming the
REM library on the link line does NOT mean the binary uses it: the linker drops
REM an unreferenced import, so a build whose dispatch path resolved to the host
REM renderings links clean, runs, computes the right numbers and never touches
REM the GPU. That shipped. `crates\ppf-cts-solver\build.rs` had emitted
REM `abi_backend_linked` on every platform but Windows, so `driver/launch.rs`
REM resolved `Backend` to `HostDevice` here, and the whole solve ran on the CPU
REM under a binary that answers `--backend cuda`.
REM
REM MEASURED ON THE UNFIXED TREE, on one box: `dumpbin /dependents` listed only
REM system DLLs, with neither libsimbackend_cuda.dll nor cudart present;
REM nvidia-smi read 210 MHz and 0 percent utilization in all 27 samples of a
REM solve; and examples/headless.py took about 6.9 sec per step against 84 msec
REM for the same commit on Linux. Every CI gate was green throughout.
REM
REM So this checks the ARTIFACT rather than the intent: ask what a thing IS,
REM never infer it from the path or the command that produced it. build.rs now
REM fails such a build outright; this is the second, independent witness, and it
REM catches any future cause rather than only a missing cfg. The ROCm build is
REM asked the same question about its own library.
for %%B in (!PPF_WIN_GPU_BACKENDS!) do (
    set "GPU_LIBRARY="
    if "%%B"=="cuda" set "GPU_LIBRARY=libsimbackend_cuda.dll"
    if "%%B"=="rocm" set "GPU_LIBRARY=libppfbe_rocm.dll"
    dumpbin /dependents "%SRC_DIR%\target\%%B\release\ppf-contact-solver.exe" | findstr /i "!GPU_LIBRARY!" >nul
    if errorlevel 1 (
        echo ERROR: target\%%B\release\ppf-contact-solver.exe does not import !GPU_LIBRARY!.
        echo        The binary links no GPU backend, so the solve would run on the CPU
        echo        while still reporting --backend %%B. See the note above this check.
        exit /b 1
    )
    echo   [DONE] the %%B solver imports !GPU_LIBRARY!
)

REM AND THE ROCm SOLVER IS ASKED WHICH PLATFORM IT LOADED. `--backend` prints what
REM build.rs selected and then, when the loaded library answers differently, a
REM `linked:` line: the AMD library answers `rocm` and the staging library
REM `hip-nvidia`. The platform this build was asked for must be the one that
REM loads, since the two write the same file.
if "!HAS_ROCM!"=="1" (
    set "GOT_BACKEND="
    set "GOT_LINKED="
    for /f "tokens=1,2" %%A in ('"%SRC_DIR%\target\rocm\release\ppf-contact-solver.exe" --backend') do (
        if "%%A"=="linked:" (
            set "GOT_LINKED=%%B"
        ) else (
            set "GOT_BACKEND=%%A"
        )
    )
    set "WANT_LINKED="
    if "!ROCM_PLATFORM!"=="nvidia" set "WANT_LINKED=hip-nvidia"
    if not "!GOT_BACKEND!"=="rocm" (
        echo ERROR: target\rocm\release\ppf-contact-solver.exe --backend answers "!GOT_BACKEND!", expected "rocm"
        exit /b 1
    )
    if not "!GOT_LINKED!"=="!WANT_LINKED!" (
        echo ERROR: the ROCm solver loaded a library for another platform than !ROCM_PLATFORM!:
        echo        --backend reports linked "!GOT_LINKED!", and this build expects "!WANT_LINKED!"
        exit /b 1
    )
    echo   [DONE] the solver answers --backend rocm and loads the !ROCM_PLATFORM! platform library
)

REM ============================================================
REM The CPU backend, into its OWN target directory
REM ============================================================
REM TWO BACKENDS CANNOT SHARE ONE DIRECTORY. Every backend links the same
REM executable name, so crates\ppf-cts-solver\build.rs refuses to put a second
REM one where another already sits: --features cpu against target\cuda would
REM replace the CUDA solver in place with one roughly 30x slower, leaving
REM nothing to say it had changed. CARGO_TARGET_DIR is the variable that file
REM names in its own refusal, and the frontend reads it too, so the cdylib
REM lands beside the binary it belongs to.
REM
REM WHY THE BUNDLE CARRIES IT. The Blender addon offers a GPU/CPU choice and a
REM device is available exactly when a build for it exists under the selected
REM root, so without this the choice can never be satisfied by a download: the
REM artist is told to run a cargo command, which is the one thing someone
REM running an unzipped release cannot do.
REM
REM THE SAME DIRECTORY WHATEVER ELSE WAS BUILT. On ARM64, and on x64 with
REM PPF_WIN_BACKENDS=cpu, no GPU backend is built at all, and the CPU build still
REM goes to target\cpu, so the addon's device table, the frontend and bundle.bat
REM read one layout whatever else sits beside it.
REM
REM MEASURED RATHER THAN ASSUMED. .github\workflows\unit-tests.yml recorded
REM that the CPU backend's entrypoints shim had never been compiled with MSVC
REM and that enabling it was "a change that has to be tried rather than
REM assumed". It was tried on a Windows Server 2025 machine, the same family
REM as the CI image: the build completes and the binary answers --backend cpu.
echo.
echo Building the CPU backend into target\cpu...
set "CARGO_TARGET_DIR=%SRC_DIR%\target\cpu"
cargo build --release --features cpu
if errorlevel 1 (
    echo ERROR: CPU backend build failed
    set "CARGO_TARGET_DIR="
    exit /b 1
)
REM CLEARED IMMEDIATELY, and on the failure path above as well. Every step after
REM this one names the directory it means, so a value left set would silently
REM point one of them at the CPU build.
set "CARGO_TARGET_DIR="

if not exist "%SRC_DIR%\target\cpu\release\ppf-contact-solver.exe" (
    echo ERROR: target\cpu\release\ppf-contact-solver.exe not found after build
    exit /b 1
)
if not exist "%SRC_DIR%\target\cpu\release\ppf-cts-server.exe" (
    echo ERROR: target\cpu\release\ppf-cts-server.exe not found after build
    exit /b 1
)
REM ALL THREE TRAVEL, not just the solver. The addon spawns the chosen
REM directory's own ppf-cts-server, and the build worker that server starts
REM loads the cdylib beside it, which is what makes the session script name the
REM CPU solver rather than the CUDA one.
if not exist "%SRC_DIR%\target\cpu\release\_ppf_cts_py.dll" (
    echo ERROR: target\cpu\release\_ppf_cts_py.dll not found after build
    exit /b 1
)
echo   [DONE] CPU backend built

echo.
echo ============================================================
echo [4/4] Creating Launcher Scripts
echo ============================================================
echo.

REM WHAT THE LAUNCHERS PUT ON PATH IS EVERY BUILT BACKEND'S LIBRARY DIRECTORY,
REM because which backend a run uses is decided when the run starts
REM (App.get_backend) rather than here: each solver imports its own backend
REM library, so all of them have to be findable, and the solver directories
REM themselves go on PATH for the same reason.
REM
REM NO TARGET DIRECTORY IS PINNED where a GPU backend was built. The frontend
REM searches target\<backend> itself and resolves the choice; a CARGO_TARGET_DIR
REM here would make that choice for every run instead. A CPU-only build is the
REM one case with nothing to resolve, and naming it keeps the frontend from
REM searching for a GPU build that was never made.
REM
REM The lines are held in variables with the launcher's own %SRC% and %BUILD_WIN%
REM written literally, and echoed through delayed expansion, which inserts them
REM after cmd.exe has parsed the block, so no character in them needs escaping.
set "LAUNCH_ENV="
set "LAUNCH_PATH=%%SRC%%\target\cpu\release"
set "PYW_DIRS=[os.path.join(src, 'target', 'cpu', 'release')"
set "PYW_TARGET=# the frontend searches this tree's target\<backend> and resolves the backend itself"
if "!HAS_CUDA!"=="1" (
    set "LAUNCH_ENV=set CUDA_PATH=%%BUILD_WIN%%\cuda"
    set "LAUNCH_PATH=%%SRC%%\target\cuda\release;%%SRC%%\crates\ppf-cts-compute\cuda\build\lib;%%CUDA_PATH%%\bin;!LAUNCH_PATH!"
    set "PYW_DIRS=!PYW_DIRS!, os.path.join(src, 'target', 'cuda', 'release'), os.path.join(src, 'crates', 'ppf-cts-compute', 'cuda', 'build', 'lib'), os.path.join(script_dir, 'cuda', 'bin')"
)
if "!HAS_ROCM!"=="1" (
    set "LAUNCH_PATH=%%SRC%%\target\rocm\release;%%SRC%%\crates\ppf-cts-compute\rocm\build\lib;%%BUILD_WIN%%\rocm\bin;!LAUNCH_PATH!"
    set "PYW_DIRS=!PYW_DIRS!, os.path.join(src, 'target', 'rocm', 'release'), os.path.join(src, 'crates', 'ppf-cts-compute', 'rocm', 'build', 'lib'), os.path.join(script_dir, 'rocm', 'bin')"
)
set "PYW_DIRS=!PYW_DIRS!]"
if not defined PPF_WIN_GPU_BACKENDS (
    set "LAUNCH_ENV=set CARGO_TARGET_DIR=%%SRC%%\target\cpu"
    set "PYW_TARGET=os.environ['CARGO_TARGET_DIR'] = os.path.join(src, 'target', 'cpu')"
)

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
echo !LAUNCH_ENV!
echo.
echo REM Set PATH to include binaries from their source locations
echo set PATH=%%BUILD_WIN%%\python;%%BUILD_WIN%%\python\Scripts;!LAUNCH_PATH!;%%PATH%%
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
echo dll_dirs = !PYW_DIRS!
echo.
echo os.environ["PATH"] = ";".join^(dll_dirs^) + ";" + os.environ.get^("PATH", ""^)
echo os.environ["PYTHONPATH"] = src + ";" + os.environ.get^("PYTHONPATH", ""^)
echo !PYW_TARGET!
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

REM Update Python path configuration (embedded Python uses .pth file). x64 only:
REM the python.org embeddable interpreter reads its search path from this file
REM and nothing else. The ARM64 interpreter is a full CPython installation that
REM honors PYTHONPATH, which the launchers above set, and it is given no ._pth in
REM the development tree because an isolated base interpreter also isolates every
REM venv made from it, which scripts\source-wheel.py builds wheels in.
if "!PPF_WIN_ARCH!"=="x64" (
    (
    echo python311.zip
    echo .
    echo Lib\site-packages
    echo %SRC_DIR%
    echo import site
    ) > "%BUILD_WIN%\python\python311._pth"
)

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
