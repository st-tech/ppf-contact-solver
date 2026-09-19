@echo off
REM File: scripts/platform.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0
REM
REM The host architecture and the backends built for it, decided in one place for
REM warmup.bat, build.bat and bundle.bat, as build-linux-native/scripts/platform.sh
REM decides them on Linux. CALL it after the manifest is loaded, from a script that
REM has already run `setlocal enabledelayedexpansion`:
REM
REM     call "%BUILD_WIN%\scripts\load-downloads.bat"
REM     if errorlevel 1 exit /b 1
REM     call "%BUILD_WIN%\scripts\platform.bat"
REM     if errorlevel 1 exit /b 1
REM
REM It has no setlocal of its own, so what it sets lands in the caller:
REM   PPF_WIN_ARCH          x64 or arm64, the architecture of the operating system
REM   PPF_WIN_ARCH_KEY      X64 or ARM64, the suffix this architecture's entries
REM                         carry in scripts\downloads.txt
REM   PPF_WIN_BACKENDS      the backends built and shipped, always in this order:
REM                         "cuda rocm cpu", "cuda cpu", "rocm cpu" or "cpu"
REM   PPF_WIN_GPU_BACKENDS  the GPU backends among them, space separated, and
REM                         empty where only the CPU backend ships
REM   PPF_WIN_MSVC_ARCH     the host and target portable-msvc.py installs; its
REM                         environment script is msvc\setup_<this>.bat
REM   PPF_WIN_RUST_HOST     the host triple `rustc -vV` must report
REM   PPF_WIN_PE_MACHINE    the PE Machine field of every binary that ships, in hex
REM   PPF_WIN_DIST_SUFFIX   the suffix of the distribution's archive name
REM   URL_<BASE>, FILE_<BASE>, SHA256_<BASE>
REM                         for each per-architecture download below, copied from
REM                         this architecture's URL_<BASE>_<ARCH_KEY> and siblings
REM
REM THE ARCHITECTURE IS THE OPERATING SYSTEM'S, NEVER A SETTING. Everything here
REM builds for the machine it runs on: the bundled interpreter, the wheels pip
REM installs into it, MinGit and the slim ffmpeg are all the host's, so a build for
REM another architecture would be a distribution whose pieces do not run together.
REM Nothing cross-compiles.
REM
REM AND THE PROCESS HAS TO AGREE WITH IT. Windows on ARM runs x64 programs under
REM emulation, and an emulated process sees PROCESSOR_ARCHITECTURE=AMD64. A build
REM started from one would provision the x64 interpreter, compiler and Rust
REM toolchain and produce an x64 distribution that passes every later check,
REM because an emulated binary answers --backend exactly as a native one does. So
REM the operating system is asked through WMI, whose provider host runs natively
REM whatever the caller runs as, and the process's own variable is compared with
REM that answer. Win32_Processor.Architecture is 9 on x64 and 12 on ARM64.
REM
REM THE BACKENDS DEFAULT PER ARCHITECTURE, AND x64 BUILDS EVERY ONE IT CAN: CUDA,
REM ROCm and the CPU backend, each into its own target\<backend>, so one x64
REM distribution runs on an NVIDIA machine and on an AMD one and the choice is
REM made when a run starts. PPF_WIN_BACKENDS narrows that set. ARM64 builds the
REM CPU backend alone. Every other combination is refused below by name, each
REM with the fact that rules it out, rather than quietly narrowed to what can be
REM built: a distribution that silently lost a GPU backend is a different product
REM under the same name.

if not "!!"=="" (
    echo ERROR: scripts\platform.bat was called without delayed expansion.
    echo        Its caller must run `setlocal enabledelayedexpansion` first, because
    echo        this script reads the variables it sets inside its own blocks.
    exit /b 1
)

set "PPF_WIN_OS_ARCH_CODE="
for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Select-Object -First 1).Architecture"`) do set "PPF_WIN_OS_ARCH_CODE=%%A"
if "!PPF_WIN_OS_ARCH_CODE!"=="9" (
    set "PPF_WIN_ARCH=x64"
    set "PPF_WIN_ARCH_KEY=X64"
    set "PPF_WIN_MSVC_ARCH=x64"
    set "PPF_WIN_RUST_HOST=x86_64-pc-windows-msvc"
    set "PPF_WIN_PE_MACHINE=8664"
    set "_PPF_NATIVE_PROCESS=AMD64"
) else if "!PPF_WIN_OS_ARCH_CODE!"=="12" (
    set "PPF_WIN_ARCH=arm64"
    set "PPF_WIN_ARCH_KEY=ARM64"
    set "PPF_WIN_MSVC_ARCH=arm64"
    set "PPF_WIN_RUST_HOST=aarch64-pc-windows-msvc"
    set "PPF_WIN_PE_MACHINE=AA64"
    set "_PPF_NATIVE_PROCESS=ARM64"
) else (
    echo ERROR: this build supports x64 and ARM64 Windows, and WMI reports processor
    echo        architecture "!PPF_WIN_OS_ARCH_CODE!" ^(9 is x64, 12 is ARM64^).
    exit /b 1
)
echo Operating system architecture: !PPF_WIN_ARCH! ^(Win32_Processor.Architecture !PPF_WIN_OS_ARCH_CODE!^)
echo This process: PROCESSOR_ARCHITECTURE=%PROCESSOR_ARCHITECTURE% PROCESSOR_ARCHITEW6432=%PROCESSOR_ARCHITEW6432%

if defined PROCESSOR_ARCHITEW6432 (
    echo ERROR: this command prompt is a 32-bit process on 64-bit Windows.
    echo        Every tool it starts would run as 32-bit or emulated code. Run the build
    echo        from a native !_PPF_NATIVE_PROCESS! command prompt.
    exit /b 1
)
if /i not "%PROCESSOR_ARCHITECTURE%"=="!_PPF_NATIVE_PROCESS!" (
    echo ERROR: this command prompt runs as %PROCESSOR_ARCHITECTURE% on !PPF_WIN_ARCH! Windows.
    echo        A build started here would provision and compile under emulation and
    echo        produce a %PROCESSOR_ARCHITECTURE% distribution that passes every check.
    echo        Run the build from a native !_PPF_NATIVE_PROCESS! command prompt.
    exit /b 1
)

set "_PPF_REQUESTED=%PPF_WIN_BACKENDS%"
if not defined _PPF_REQUESTED (
    if "!PPF_WIN_ARCH!"=="x64" (
        set "_PPF_REQUESTED=cuda rocm cpu"
    ) else (
        set "_PPF_REQUESTED=cpu"
    )
)
set "_PPF_WANT_CUDA=0"
set "_PPF_WANT_ROCM=0"
set "_PPF_WANT_CPU=0"
for %%B in (!_PPF_REQUESTED!) do (
    if /i "%%B"=="cuda" (
        set "_PPF_WANT_CUDA=1"
    ) else if /i "%%B"=="rocm" (
        set "_PPF_WANT_ROCM=1"
    ) else if /i "%%B"=="cpu" (
        set "_PPF_WANT_CPU=1"
    ) else (
        echo ERROR: PPF_WIN_BACKENDS names "%%B", and the backends are cuda, rocm and cpu.
        exit /b 1
    )
)

if "!_PPF_WANT_CPU!"=="0" (
    echo ERROR: PPF_WIN_BACKENDS="!_PPF_REQUESTED!" leaves out cpu.
    echo        Every distribution ships the CPU backend, which is what runs on a machine
    echo        with no supported GPU. Name cpu, or leave the setting empty.
    exit /b 1
)
REM NVIDIA PUBLISHES NO CUDA TOOLKIT FOR WINDOWS ON ARM. Read on PyPI, which carries
REM NVIDIA's own redistributable toolkit components: nvidia-cuda-nvcc,
REM nvidia-cuda-runtime and nvidia-cuda-crt publish manylinux x86_64, manylinux
REM aarch64 and win_amd64 builds in every release from 12.8 to 13.4, and no win_arm64
REM build in any of them. The installer this directory extracts is x86_64 as well.
if "!_PPF_WANT_CUDA!"=="1" if "!PPF_WIN_ARCH!"=="arm64" (
    echo ERROR: PPF_WIN_BACKENDS asks for cuda, and this host is ARM64 Windows.
    echo        NVIDIA publishes no CUDA toolkit for Windows on ARM: its nvcc, runtime
    echo        and CRT components exist for x86_64 Windows and for x86_64 and aarch64
    echo        Linux only. Leave the setting empty for the CPU backend.
    exit /b 1
)
REM AMD PUBLISHES NO ROCm FOR WINDOWS ON ARM. TheRock's tarball index, the channel
REM scripts\downloads.txt pins the SDK from, lists Windows distributions for x86_64
REM only, and the HIP runtime a ROCm distribution ships is one of them.
if "!_PPF_WANT_ROCM!"=="1" if "!PPF_WIN_ARCH!"=="arm64" (
    echo ERROR: PPF_WIN_BACKENDS asks for rocm, and this host is ARM64 Windows.
    echo        AMD publishes no ROCm for Windows on ARM: the SDK channel this build
    echo        pins lists Windows distributions for x86_64 only. Leave the setting
    echo        empty for the CPU backend.
    exit /b 1
)

REM Built additively and in a fixed order, so every consumer sees the same list
REM whatever order the caller named them in, and the GPU backends come before the
REM CPU one as they do everywhere else.
set "PPF_WIN_GPU_BACKENDS="
if "!_PPF_WANT_CUDA!"=="1" set "PPF_WIN_GPU_BACKENDS=cuda"
if "!_PPF_WANT_ROCM!"=="1" (
    if defined PPF_WIN_GPU_BACKENDS (
        set "PPF_WIN_GPU_BACKENDS=!PPF_WIN_GPU_BACKENDS! rocm"
    ) else (
        set "PPF_WIN_GPU_BACKENDS=rocm"
    )
)
if defined PPF_WIN_GPU_BACKENDS (
    set "PPF_WIN_BACKENDS=!PPF_WIN_GPU_BACKENDS! cpu"
) else (
    set "PPF_WIN_BACKENDS=cpu"
)

REM The archive name. ONE x64 ARCHIVE CARRIES EVERY GPU BACKEND BUILT, so the name
REM does not say which one: it keeps the name it has always been published
REM under, and a link to it does not change meaning. An x64 build narrowed to the
REM CPU backend says so, because that is a different product.
if "!PPF_WIN_ARCH!"=="arm64" (
    set "PPF_WIN_DIST_SUFFIX=windows-arm64"
) else if defined PPF_WIN_GPU_BACKENDS (
    set "PPF_WIN_DIST_SUFFIX=win64"
) else (
    set "PPF_WIN_DIST_SUFFIX=win64-cpu"
)

REM THE PER-ARCHITECTURE DOWNLOADS. A missing entry is an error rather than a fall
REM back to another architecture's file, which would install something this host
REM cannot run natively and would then pass, for the reason the header gives.
REM
REM MinGit and the interpreter are pinned by content, so each carries a SHA256
REM that scripts\fetch-download.bat checks. rustup-init is served from a moving
REM pointer that has no stable digest, so it carries none, and what stands behind
REM it instead is the host triple warmup.bat and build.bat read back from rustc.
for %%D in (MINGIT PYTHON) do (
    for %%P in (URL FILE SHA256) do (
        if not defined %%P_%%D_%PPF_WIN_ARCH_KEY% (
            echo ERROR: scripts\downloads.txt defines no %%P_%%D_%PPF_WIN_ARCH_KEY%, so %PPF_WIN_ARCH% has no %%D entry.
            exit /b 1
        )
        set "%%P_%%D=!%%P_%%D_%PPF_WIN_ARCH_KEY%!"
    )
)
for %%P in (URL FILE) do (
    if not defined %%P_RUSTUP_%PPF_WIN_ARCH_KEY% (
        echo ERROR: scripts\downloads.txt defines no %%P_RUSTUP_%PPF_WIN_ARCH_KEY%, so %PPF_WIN_ARCH% has no RUSTUP entry.
        exit /b 1
    )
    set "%%P_RUSTUP=!%%P_RUSTUP_%PPF_WIN_ARCH_KEY%!"
)

echo Backends: !PPF_WIN_BACKENDS!   Distribution suffix: !PPF_WIN_DIST_SUFFIX!
set "_PPF_REQUESTED="
set "_PPF_WANT_CUDA="
set "_PPF_WANT_ROCM="
set "_PPF_WANT_CPU="
set "_PPF_NATIVE_PROCESS="
exit /b 0
