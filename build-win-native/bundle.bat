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

REM Full cleanup before bundling
echo [0/10] Performing full cleanup before bundling...
call "%BUILD_WIN%\clear-all.bat" /nopause
echo.

REM Get parent directory (SRC_DIR)
for %%I in ("%BUILD_WIN%\..") do set SRC_DIR=%%~fI

call "%BUILD_WIN%\scripts\load-downloads.bat"
if errorlevel 1 (
    echo ERROR: Failed to load download manifest
    exit /b 1
)
REM WHICH DIRECTORIES SHIP IS THE BUILD'S CONFIGURATION, decided by the same
REM scripts\platform.bat that warmup.bat and build.bat read: the CPU backend
REM always, and one target\<backend> for every GPU backend in PPF_WIN_BACKENDS.
REM WHICH BACKEND A DIRECTORY HOLDS is still asked of the binary below rather than
REM taken from the directory's name, and a disagreement between the two stops the
REM bundle, as build-linux-native/bundle.sh does.
call "%BUILD_WIN%\scripts\platform.bat"
if errorlevel 1 (
    echo ERROR: this host and PPF_WIN_BACKENDS do not describe a distribution this directory can make
    exit /b 1
)

set DIST_DIR=%BUILD_WIN%\dist
set BIN_DIR=%DIST_DIR%\bin
set CPU_TARGET_DIR=%DIST_DIR%\target\cpu\release
set LICENSES_DIR=%DIST_DIR%\licenses
REM The interpreter the bundle's own checks run with. It is the build tree's, not
REM the copy in dist, so a check cannot write caches into what ships.
set BUILD_PYTHON=%BUILD_WIN%\python\python.exe
set ROCM_DIR=%BUILD_WIN%\rocm

REM Use local CUDA from build-win-native
set CUDA_PATH=%BUILD_WIN%\cuda

REM ============================================================
REM Verify build exists
REM ============================================================
echo [1/10] Verifying build...

REM ONE DIRECTORY PER BACKEND, in the source tree and reproduced inside the
REM bundle. Every backend links the same executable name and
REM crates\ppf-cts-solver\build.rs refuses to put two in one directory, so the
REM directories are what keep them apart; that layout is not an internal detail
REM of the build, it is what the addon's selector probes and what
REM frontend.get_backend resolves among.
set CPU_RUST_EXE=%SRC_DIR%\target\cpu\release\ppf-contact-solver.exe
set CPU_SERVER_EXE=%SRC_DIR%\target\cpu\release\ppf-cts-server.exe
set CPU_PYO3_DLL=%SRC_DIR%\target\cpu\release\_ppf_cts_py.dll
set CUDA_DLL=%SRC_DIR%\crates\ppf-cts-compute\cuda\build\lib\libsimbackend_cuda.dll
set ROCM_DLL=%SRC_DIR%\crates\ppf-cts-compute\rocm\build\lib\libppfbe_rocm.dll

set "HAS_CUDA=0"
set "HAS_ROCM=0"
for %%B in (!PPF_WIN_GPU_BACKENDS!) do (
    if "%%B"=="cuda" set "HAS_CUDA=1"
    if "%%B"=="rocm" set "HAS_ROCM=1"
)

for %%B in (!PPF_WIN_GPU_BACKENDS!) do (
    for %%F in (ppf-contact-solver.exe ppf-cts-server.exe _ppf_cts_py.dll) do (
        if not exist "%SRC_DIR%\target\%%B\release\%%F" (
            echo ERROR: target\%%B\release\%%F not found. Please run build.bat first.
            exit /b 1
        )
    )
)
if "!HAS_CUDA!"=="1" (
    if not exist "%CUDA_DLL%" (
        echo ERROR: %CUDA_DLL% not found. Please run build.bat first.
        exit /b 1
    )
)
if "!HAS_ROCM!"=="1" (
    if not exist "%ROCM_DLL%" (
        echo ERROR: %ROCM_DLL% not found. Please run build.bat first.
        exit /b 1
    )
)
for %%F in ("%CPU_RUST_EXE%" "%CPU_SERVER_EXE%" "%CPU_PYO3_DLL%") do (
    if not exist %%F (
        echo ERROR: %%~F not found. Please run build.bat first.
        exit /b 1
    )
)
if "!HAS_ROCM!"=="1" (
    for %%F in (bin\amdhip64_7.dll share\doc\rocm-core\LICENSE.md share\hip\version share\therock\therock_manifest.json .info\version) do (
        if not exist "%ROCM_DIR%\%%F" (
            echo ERROR: the ROCm SDK at %ROCM_DIR% has no %%F. Please run warmup.bat first.
            exit /b 1
        )
    )
)
if not exist "%BUILD_WIN%\python\python.exe" (
    echo ERROR: Bundled Python not found. Please run warmup.bat first.
    exit /b 1
)
if not exist "%BUILD_WIN%\mingit\cmd\git.exe" (
    echo ERROR: MinGit not found. Please run warmup.bat first.
    exit /b 1
)

REM WHICH BACKEND IS IN EACH target\<backend>\release IS ASKED OF THE BINARY,
REM NOT CONFIGURED.
REM `--backend` prints what build.rs compiled in and, when the library it loads
REM answers differently, a second `linked:` line. That is the rule this build
REM follows for a backend: ask an artifact what it IS rather than inferring it
REM from its path. The configuration above is a second statement of the same
REM fact, free to disagree with the build, so the two are compared.
REM
REM The solver imports its backend library, and that library its device runtime,
REM so both directories are put on PATH for the question and taken off after it.
REM The ROCm staging library, built through nvcc to run on an NVIDIA GPU, answers
REM `linked: hip-nvidia`, and it is refused here: it is a way to execute the ROCm
REM build without AMD hardware, and nothing a user downloads should carry it.
for %%B in (!PPF_WIN_GPU_BACKENDS!) do (
    set "SAVED_PATH=!PATH!"
    if "%%B"=="cuda" (
        set "PATH=%SRC_DIR%\crates\ppf-cts-compute\cuda\build\lib;%CUDA_PATH%\bin;!PATH!"
    ) else (
        set "PATH=%SRC_DIR%\crates\ppf-cts-compute\rocm\build\lib;%ROCM_DIR%\bin;!PATH!"
    )
    set "GOT_BACKEND="
    set "GOT_LINKED="
    for /f "tokens=1,2" %%A in ('"%SRC_DIR%\target\%%B\release\ppf-contact-solver.exe" --backend') do (
        if "%%A"=="linked:" (
            set "GOT_LINKED=%%B"
        ) else (
            set "GOT_BACKEND=%%A"
        )
    )
    set "PATH=!SAVED_PATH!"
    if not "!GOT_BACKEND!"=="%%B" (
        echo ERROR: target\%%B\release\ppf-contact-solver.exe reports backend "!GOT_BACKEND!", and this directory ships %%B.
        echo        The binary is the authority on what it is. Rebuild target\%%B for
        echo        that backend, or narrow PPF_WIN_BACKENDS to what was built.
        exit /b 1
    )
    if "!GOT_LINKED!"=="hip-nvidia" (
        echo ERROR: target\%%B loads the ROCm library built for the NVIDIA staging platform.
        echo        That build exists to run the ROCm source on an NVIDIA GPU and does not
        echo        ship. Rebuild with PPF_WIN_ROCM_PLATFORM unset, which builds for AMD.
        exit /b 1
    )
    if defined GOT_LINKED (
        echo ERROR: target\%%B\release\ppf-contact-solver.exe loads a library that answers "!GOT_LINKED!",
        echo        not the %%B backend it was built for.
        exit /b 1
    )
    echo   [OK] target\%%B\release answers --backend !GOT_BACKEND!
)
echo   [OK] Build verified

REM ============================================================
REM Create dist directory
REM ============================================================
echo.
echo [2/10] Creating dist directory...

if exist "%DIST_DIR%" (
    echo   Removing old dist folder...
    rmdir /s /q "%DIST_DIR%"
)
mkdir "%DIST_DIR%"
mkdir "%BIN_DIR%"
mkdir "%DIST_DIR%\target"
for %%B in (!PPF_WIN_GPU_BACKENDS!) do (
    mkdir "%DIST_DIR%\target\%%B"
    mkdir "%DIST_DIR%\target\%%B\release"
)
mkdir "%DIST_DIR%\target\cpu"
mkdir "%CPU_TARGET_DIR%"
echo   [OK] Created %DIST_DIR%

REM ============================================================
REM Copy application binaries
REM ============================================================
echo.
echo [3/10] Copying application binaries...

REM Copy each GPU backend's three artifacts into its own target\<backend>\release.
REM ppf-contact-solver.exe is the solver driver; ppf-cts-server.exe is the
REM tokio-based solver host that the Blender addon's Windows Native launcher
REM (blender_addon/core/connection.py) spawns when users select "Windows Native"
REM mode in the addon UI; _ppf_cts_py.dll is the extension frontend/__init__.py
REM loads by absolute path, which is why no wheel is installed into the bundled
REM Python. All three travel per backend, because the addon spawns the chosen
REM directory's own server and the build worker it starts loads the cdylib
REM beside it.
for %%B in (!PPF_WIN_GPU_BACKENDS!) do (
    for %%F in (ppf-contact-solver.exe ppf-cts-server.exe _ppf_cts_py.dll) do (
        copy "%SRC_DIR%\target\%%B\release\%%F" "%DIST_DIR%\target\%%B\release\" >nul
        if errorlevel 1 (
            echo ERROR: Failed to copy %%F of the %%B backend
            exit /b 1
        )
    )
    echo   Copied the %%B solver, server and cdylib to target/%%B/release/
)

REM ============================================================
REM The CPU backend's own three artifacts
REM ============================================================
REM All three, for the reason build.bat states where it builds them: the addon
REM spawns the chosen directory's own ppf-cts-server, and the build worker that
REM server starts loads the cdylib beside it, which is what makes the session
REM script name the CPU solver. Ship two of the three and a CPU run quietly
REM executes the CUDA solver, the two halves of one run coming from different
REM builds.
copy "%CPU_RUST_EXE%" "%CPU_TARGET_DIR%\" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy the CPU ppf-contact-solver.exe
    exit /b 1
)
copy "%CPU_SERVER_EXE%" "%CPU_TARGET_DIR%\" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy the CPU ppf-cts-server.exe
    exit /b 1
)
copy "%CPU_PYO3_DLL%" "%CPU_TARGET_DIR%\" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy the CPU _ppf_cts_py.dll
    exit /b 1
)
echo   Copied the CPU solver, server and cdylib to target/cpu/release/

REM ASK EACH ARTIFACT WHAT IT IS RATHER THAN INFERRING IT FROM ITS PATH, which
REM is why build.rs writes a marker at all: the executable name cannot carry
REM the backend, so the directory name is a convention and the marker is
REM evidence. frontend.backend_of reads exactly this file, and without it
REM frontend.solver_dir("cpu") answers None inside a bundle that ships a CPU
REM build. Written here rather than copied, because it should state what
REM LANDED, not whatever the build directory happened to hold.
REM NO SPACE BEFORE THE REDIRECT: `echo cuda >file` writes "cuda " with the
REM space included. The readers strip surrounding whitespace, so the trailing
REM CRLF is harmless, but a trailing space inside the name is not worth relying
REM on them for. The `echo|set /p=` form that writes no newline at all is NOT
REM used here: with a pipe the redirect binds to the wrong side of it.
REM Each GPU directory's marker names that backend, which step 1 established is
REM what its binary answers: the loop there refused any directory whose solver
REM reported something else, so the name and the binary already agree.
for %%B in (!PPF_WIN_GPU_BACKENDS!) do echo %%B>"%DIST_DIR%\target\%%B\release\.ppf-backend"
echo cpu>"%CPU_TARGET_DIR%\.ppf-backend"
echo   Wrote the backend markers

REM THE MARKER THAT SAYS THIS TREE KEEPS ITS OWN STATE. On Windows the frontend
REM already roots the data directory and the cache at the tree unconditionally,
REM so this changes nothing here and is written anyway: it is the same
REM statement the macOS bundle makes, read by the same function, and a bundle
REM that omitted it would be relying on the OS test rather than saying what it
REM is. datamodel::app::is_selfcontained reads this file.
REM The content is not read by anything; is_selfcontained tests for the FILE.
REM A word rather than a version, because this script has no version variable
REM and inventing one here would be a second place for it to drift from.
echo selfcontained>"%DIST_DIR%\.ppf-selfcontained"
echo   Wrote the self-contained marker

REM AND THE CPU MARKER IS CHECKED AGAINST THE ARTIFACT, not just written. The
REM exe answers --backend before its argument parser and before any backend is
REM opened, so this costs milliseconds and proves the directory holds what the
REM marker claims. It links no backend DLL, so it loads with nothing extra on
REM PATH.
for /f "delims=" %%B in ('"%CPU_TARGET_DIR%\ppf-contact-solver.exe" --backend') do set "GOT_CPU_BACKEND=%%B"
if not "%GOT_CPU_BACKEND%"=="cpu" (
    echo ERROR: target\cpu\release\ppf-contact-solver.exe reports backend "%GOT_CPU_BACKEND%", expected "cpu"
    echo        The two builds each own a target directory and must not be mixed.
    exit /b 1
)
echo   Verified the CPU solver reports backend cpu

REM Copy each GPU backend's library to bin/ (loaded via PATH). They are separate
REM files with separate names, so a distribution carrying both ships both.
for %%B in (!PPF_WIN_GPU_BACKENDS!) do (
    set "GPU_DLL="
    if "%%B"=="cuda" set "GPU_DLL=%CUDA_DLL%"
    if "%%B"=="rocm" set "GPU_DLL=%ROCM_DLL%"
    copy "!GPU_DLL!" "%BIN_DIR%\" >nul
    if errorlevel 1 (
        echo ERROR: Failed to copy !GPU_DLL!
        exit /b 1
    )
    for %%G in ("!GPU_DLL!") do echo   Copied %%~nxG to bin/
)

REM ============================================================
REM Copy the GPU backend's runtime libraries
REM ============================================================
echo.
echo [4/10] Copying the GPU backend's runtime libraries...

if "!HAS_CUDA!"=="1" (
    set CUDA_BIN=%CUDA_PATH%\bin

    REM List of redistributable CUDA DLLs (only cudart needed)
    set CUDA_DLLS=cudart64_12.dll

    for %%D in (!CUDA_DLLS!) do (
        if exist "!CUDA_BIN!\%%D" (
            copy "!CUDA_BIN!\%%D" "%BIN_DIR%\" >nul
            if errorlevel 1 (
                echo ERROR: Failed to copy %%D
                exit /b 1
            )
            echo   Copied %%D
        ) else (
            echo   WARNING: %%D not found in !CUDA_BIN!
        )
    )
)
if "!HAS_ROCM!"=="1" (
    REM THE ROCm RUNTIME IS DERIVED FROM WHAT THE SHIPPED FILES IMPORT, NOT LISTED.
    REM The backend library imports the HIP runtime, and ppf-cts-core's check_gpu
    REM probe puts the same import into the solver, the server and the cdylib, so
    REM all four are walked, through bin\ and then the SDK's bin directory, and
    REM every DLL the walk reaches outside the declared Windows system set is
    REM copied. A DLL found nowhere fails the bundle by name, since a
    REM distribution missing one does not start at all on the user's machine and
    REM the loader's message would name a file rather than this step.
    REM The distribution carries the SDK's runtime closure itself rather than
    REM asking the user to install one, and the Windows closure is its own, with
    REM no hsa-runtime64, because Windows reaches the GPU through the driver
    REM rather than through ROCr.
    set "CLOSURE_LIST=%BUILD_WIN%\rocm-runtime-closure.txt"
    "%BUILD_PYTHON%" "%BUILD_WIN%\scripts\pe-audit.py" closure --file "%BIN_DIR%\libppfbe_rocm.dll" --file "%DIST_DIR%\target\rocm\release\ppf-contact-solver.exe" --file "%DIST_DIR%\target\rocm\release\ppf-cts-server.exe" --file "%DIST_DIR%\target\rocm\release\_ppf_cts_py.dll" --search "%BIN_DIR%" --search "%ROCM_DIR%\bin" --present "%BUILD_WIN%\python" > "!CLOSURE_LIST!"
    if errorlevel 1 (
        echo ERROR: the ROCm runtime closure does not resolve ^(see above^)
        exit /b 1
    )
    for /f "usebackq delims=" %%L in ("!CLOSURE_LIST!") do (
        if /i not "%%~dpL"=="%BIN_DIR%\" (
            copy /y "%%L" "%BIN_DIR%\" >nul
            if errorlevel 1 (
                echo ERROR: Failed to copy %%L
                exit /b 1
            )
            echo   Copied %%~nxL from the ROCm SDK
        )
    )
    if not exist "%BIN_DIR%\amdhip64_7.dll" (
        echo ERROR: the runtime closure did not reach amdhip64_7.dll, which the ROCm backend imports.
        exit /b 1
    )
) else (
    echo   [SKIP] no GPU backend ships in this distribution
)

REM Copy ffmpeg for video export
echo.
echo Copying ffmpeg for video export...
if exist "%BUILD_WIN%\ffmpeg\ffmpeg.exe" (
    echo   Source: %BUILD_WIN%\ffmpeg\ffmpeg.exe
    echo   Target: %BIN_DIR%\ffmpeg.exe
    copy "%BUILD_WIN%\ffmpeg\ffmpeg.exe" "%BIN_DIR%\" >nul
    if errorlevel 1 (
        echo   [FAILED] Could not copy ffmpeg.exe
    ) else (
        echo   [OK] Copied ffmpeg.exe to bin/
    )
) else (
    echo   [SKIP] ffmpeg.exe not found at %BUILD_WIN%\ffmpeg\ffmpeg.exe
)

REM ============================================================
REM Copy Python environment (exclude caches)
REM ============================================================
echo.
echo [5/10] Copying Python environment...

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
REM Copy MinGit (for sparse_clone functionality)
REM ============================================================
echo.
echo [5.3/10] Copying MinGit...

robocopy "%BUILD_WIN%\mingit" "%DIST_DIR%\mingit" /E /NJH /NJS /NDL /NC /NS /NP >nul
if errorlevel 8 (
    echo ERROR: Failed to copy MinGit
    exit /b 1
)
echo   [OK] Copied MinGit

REM ============================================================
REM Shorten webpack chunk IDs to reduce path length
REM ============================================================
echo.
echo [5.5/10] Shortening webpack chunk IDs...

"%BUILD_WIN%\python\python.exe" "%BUILD_WIN%\shorten_webpack_chunks.py" "%DIST_DIR%"
if errorlevel 1 (
    echo WARNING: Chunk shortening failed, continuing anyway...
)
echo   [OK] Chunk IDs shortened

REM ============================================================
REM Copy examples (.ipynb and .py files)
REM ============================================================
echo.
echo [6/10] Copying examples...
echo   Source: %SRC_DIR%\examples
echo   Target: %DIST_DIR%\examples
echo   Filter: *.ipynb and *.py

if exist "%SRC_DIR%\examples" (
    echo   Copying notebook and Python files...
    robocopy "%SRC_DIR%\examples" "%DIST_DIR%\examples" *.ipynb *.py /S /XD .* /XF .*
    if errorlevel 8 (
        echo   [FAILED] Could not copy examples
        exit /b 1
    )
    echo   [OK] Copied examples
) else (
    echo   [SKIP] examples folder not found
)

REM ============================================================
REM Copy frontend module (exclude caches)
REM ============================================================
echo.
echo [7/10] Copying frontend module...

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
REM Copy crate source roots used by frontend log harvesting
REM ============================================================
REM frontend/_session_inspect_.py:_harvest_log_docstrings walks
REM crates/ppf-cts-solver/src and crates/ppf-cts-core/src to discover
REM log channel names from `// Name:` docstrings. Without these in the
REM bundle, session.get.log.names() returns [] and notebooks fail at
REM `assert "time-per-frame" in logs`. Exclude cpp/build (CUDA build
REM artifacts; the resulting DLL is copied separately into bin\).
echo.
echo [8/10] Copying crate source roots...

if exist "%SRC_DIR%\crates\ppf-cts-solver\src" (
    robocopy "%SRC_DIR%\crates\ppf-cts-solver\src" "%DIST_DIR%\crates\ppf-cts-solver\src" /E /XD __pycache__ build obj tests /XF *.pyc *.pyo *.obj *.lib *.exp /NJH /NJS /NDL /NC /NS /NP >nul
    if errorlevel 8 (
        echo ERROR: Failed to copy crates\ppf-cts-solver\src
        exit /b 1
    )
    echo   [OK] Copied crates\ppf-cts-solver\src
) else (
    echo ERROR: crates\ppf-cts-solver\src not found at %SRC_DIR%\crates\ppf-cts-solver\src
    exit /b 1
)

if exist "%SRC_DIR%\crates\ppf-cts-core\src" (
    robocopy "%SRC_DIR%\crates\ppf-cts-core\src" "%DIST_DIR%\crates\ppf-cts-core\src" /E /XD __pycache__ /XF *.pyc *.pyo /NJH /NJS /NDL /NC /NS /NP >nul
    if errorlevel 8 (
        echo ERROR: Failed to copy crates\ppf-cts-core\src
        exit /b 1
    )
    echo   [OK] Copied crates\ppf-cts-core\src
) else (
    echo ERROR: crates\ppf-cts-core\src not found at %SRC_DIR%\crates\ppf-cts-core\src
    exit /b 1
)

REM ============================================================
REM Generate launcher and documentation
REM ============================================================
echo.
echo [9/10] Generating launcher and documentation...

REM NO LAUNCHER PINS A TARGET DIRECTORY WHERE A GPU BACKEND SHIPS: each backend
REM has its own target\<backend>, the frontend searches them, and which one a run
REM uses is resolved when it starts (frontend.get_backend). A CPU-only
REM distribution is the one case with nothing to resolve, and its launchers name
REM target\cpu so the frontend does not search for a GPU build that never
REM shipped. The line is held in a variable and echoed through delayed expansion,
REM which inserts it after cmd.exe has parsed the generator block, so nothing in
REM it needs escaping; and it is never empty, since `echo` with nothing to print
REM writes "ECHO is on." into the file.
if defined PPF_WIN_GPU_BACKENDS (
    set "DIST_TARGET_LINE=REM the frontend searches target\^<backend^> and resolves which backend a run uses"
    set "PYW_TARGET_LINE=# the frontend searches target\<backend> and resolves which backend a run uses"
) else (
    set "DIST_TARGET_LINE=set CARGO_TARGET_DIR=%%DIST%%\target\cpu"
    set "PYW_TARGET_LINE=os.environ['CARGO_TARGET_DIR'] = os.path.join(script_dir, 'target', 'cpu')"
)

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
echo REM Set PATH to include bundled binaries and MinGit
echo set PATH=%%DIST%%\python;%%DIST%%\python\Scripts;%%DIST%%\bin;%%DIST%%\mingit\cmd;%%PATH%%
echo set PYTHONPATH=%%DIST%%;%%PYTHONPATH%%
echo !DIST_TARGET_LINE!
echo.
echo REM Point Python at certifi's CA bundle so HTTPS works in the
echo REM bundled embedded distribution. The unparenthesized phrasing
echo REM here is deliberate: parens inside this generator's outer
echo REM redirect block confuse cmd.exe's parser.
echo set SSL_CERT_FILE=%%DIST%%\python\Lib\site-packages\certifi\cacert.pem
echo set REQUESTS_CA_BUNDLE=%%DIST%%\python\Lib\site-packages\certifi\cacert.pem
echo.
echo REM Set Jupyter/IPython config to dist-relative paths
echo set JUPYTER_CONFIG_DIR=%%DIST%%\jupyter\config
echo set JUPYTER_DATA_DIR=%%DIST%%\jupyter\data
echo set IPYTHONDIR=%%DIST%%\jupyter\ipython
echo.
echo REM Set dark theme if not already configured
echo set THEME_DIR=%%JUPYTER_CONFIG_DIR%%\lab\user-settings\@jupyterlab\apputils-extension
echo if not exist "%%THEME_DIR%%" mkdir "%%THEME_DIR%%"
echo if not exist "%%THEME_DIR%%\themes.jupyterlab-settings" ^(
echo     echo {"theme": "JupyterLab Dark"} ^> "%%THEME_DIR%%\themes.jupyterlab-settings"
echo ^)
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
echo mingit_dir = os.path.join^(script_dir, "mingit", "cmd"^)
echo.
echo os.environ["PATH"] = bin_dir + ";" + mingit_dir + ";" + os.environ.get^("PATH", ""^)
echo os.environ["PYTHONPATH"] = script_dir + ";" + os.environ.get^("PYTHONPATH", ""^)
echo !PYW_TARGET_LINE!
echo.
echo # Set Jupyter/IPython config to project-relative paths
echo os.environ["JUPYTER_CONFIG_DIR"] = os.path.join^(script_dir, "jupyter", "config"^)
echo os.environ["JUPYTER_DATA_DIR"] = os.path.join^(script_dir, "jupyter", "data"^)
echo os.environ["IPYTHONDIR"] = os.path.join^(script_dir, "jupyter", "ipython"^)
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
echo REM Set PATH to include bundled binaries and MinGit
echo set PATH=%%DIST%%\python;%%DIST%%\python\Scripts;%%DIST%%\bin;%%DIST%%\mingit\cmd;%%PATH%%
echo set PYTHONPATH=%%DIST%%;%%PYTHONPATH%%
echo !DIST_TARGET_LINE!
echo.
echo REM Point Python at certifi's CA bundle - see start.bat for context.
echo set SSL_CERT_FILE=%%DIST%%\python\Lib\site-packages\certifi\cacert.pem
echo set REQUESTS_CA_BUNDLE=%%DIST%%\python\Lib\site-packages\certifi\cacert.pem
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

REM Update Python path configuration for dist. A ._pth file beside the
REM interpreter's DLL puts CPython in isolated mode: sys.path is exactly its lines,
REM and PYTHONHOME and PYTHONPATH are ignored, so the distribution runs from its own
REM files whatever the machine's environment holds. `..` is the distribution root,
REM which holds frontend\.
REM
REM THE TWO INTERPRETERS NEED DIFFERENT LINES. The x64 embeddable interpreter keeps
REM its standard library in python311.zip and its extension modules beside
REM python.exe. The ARM64 interpreter is a full CPython tree, with the standard
REM library in Lib\ and extension modules in DLLs\, and the file is named for
REM whichever pythonXY.dll it carries.
if "!PPF_WIN_ARCH!"=="x64" (
    (
    echo python311.zip
    echo .
    echo Lib\site-packages
    echo ..
    echo import site
    ) > "%DIST_DIR%\python\python311._pth"
) else (
    set "PY_DLL_STEM="
    for %%F in ("%DIST_DIR%\python\python3*.dll") do (
        if /i not "%%~nF"=="python3" set "PY_DLL_STEM=%%~nF"
    )
    if not defined PY_DLL_STEM (
        echo ERROR: no pythonXY.dll under %DIST_DIR%\python, so the ._pth file has no name
        exit /b 1
    )
    (
    echo Lib
    echo DLLs
    echo Lib\site-packages
    echo ..
    echo import site
    ) > "%DIST_DIR%\python\!PY_DLL_STEM!._pth"
)

REM THE LICENSE TEXTS OF WHAT IS REDISTRIBUTED, gathered from what they cover. The
REM ROCm runtime files carry theirs from the SDK they were copied out of, so the
REM text matches the binaries: TheRock 10.0.0's Windows tarball holds no HIP
REM license file of its own, and its share\doc\rocm-core\LICENSE.md is the MIT text
REM that ROCm 7.2.4's hip-runtime-amd package ships as share/doc/hip/LICENSE.md,
REM identical line for line when the two are compared. Its therock_manifest.json
REM and share\hip\version name the commits the runtime was built from.
mkdir "%LICENSES_DIR%"
copy "%SRC_DIR%\LICENSE" "%LICENSES_DIR%\ZOZO-Contact-Solver-LICENSE.txt" >nul
if errorlevel 1 (
    echo ERROR: could not copy the project's LICENSE into %LICENSES_DIR%
    exit /b 1
)
if "!HAS_ROCM!"=="1" (
    for %%C in ("share\doc\rocm-core\LICENSE.md=ROCm-LICENSE.md" "share\therock\therock_manifest.json=ROCm-therock_manifest.json" "share\hip\version=ROCm-hip-version.txt" ".info\version=ROCm-distribution-version.txt") do (
        for /f "tokens=1,2 delims==" %%A in (%%C) do (
            copy "%ROCM_DIR%\%%A" "%LICENSES_DIR%\%%B" >nul
            if errorlevel 1 (
                echo ERROR: could not copy %ROCM_DIR%\%%A into %LICENSES_DIR%
                exit /b 1
            )
        )
    )
)
REM The source record of every wheel warmup.bat built from pinned upstream source.
if "!PPF_WIN_ARCH!"=="arm64" (
    for %%W in (triangle tetgen) do (
        set "WHEEL_RECORD="
        for %%R in ("%BUILD_WIN%\wheels\%%W-*.sources.txt") do set "WHEEL_RECORD=%%~fR"
        if not defined WHEEL_RECORD (
            echo ERROR: no source record for the %%W wheel in %BUILD_WIN%\wheels. Re-run warmup.bat.
            exit /b 1
        )
        copy "!WHEEL_RECORD!" "%LICENSES_DIR%\%%W-sources.txt" >nul
        if errorlevel 1 (
            echo ERROR: could not copy the %%W source record into %LICENSES_DIR%
            exit /b 1
        )
    )
)

REM Generate THIRD_PARTY_LICENSES.txt, one section per component that ships.
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
)
if "!HAS_CUDA!"=="1" (
    >> "%DIST_DIR%\THIRD_PARTY_LICENSES.txt" (
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
    )
)
if "!HAS_ROCM!"=="1" (
    >> "%DIST_DIR%\THIRD_PARTY_LICENSES.txt" (
    echo AMD ROCm runtime
    echo ----------------
    echo bin\libppfbe_rocm.dll is the ROCm backend. It loads the HIP runtime
    echo dynamically, so these AMD ROCm files are REDISTRIBUTED here:
    REM LISTED FROM THE CLOSURE THIS BUILD WALKED, not from everything in bin\:
    REM that directory also holds the CUDA backend library and its runtime where
    REM this distribution carries CUDA too, and naming those here would say AMD
    REM redistributes NVIDIA's files.
    for /f "usebackq delims=" %%L in ("%BUILD_WIN%\rocm-runtime-closure.txt") do (
        for %%N in ("%%L") do echo   bin\%%~nxN
    )
    echo They come from AMD's TheRock ROCm distribution, version
    type "%ROCM_DIR%\.info\version"
    echo and are licensed under the MIT license text in licenses\ROCm-LICENSE.md.
    echo licenses\ROCm-therock_manifest.json and licenses\ROCm-hip-version.txt
    echo name the sources they were built from. The AMD GPU driver is not
    echo redistributed and must be installed on the system.
    echo.
    echo.
    )
)
if "!PPF_WIN_ARCH!"=="arm64" (
    >> "%DIST_DIR%\THIRD_PARTY_LICENSES.txt" (
    echo Packages built from source
    echo --------------------------
    echo No binary wheel of triangle or tetgen exists for Windows on ARM, so each
    echo was built from its upstream repository at a pinned commit, unmodified.
    echo The repositories, commits and compiler settings are in
    echo licenses\triangle-sources.txt and licenses\tetgen-sources.txt, and each
    echo package's own license files are in its dist-info directory under
    echo python\Lib\site-packages. TetGen is licensed under the GNU Affero General
    echo Public License version 3, and Triangle's C core under its author's terms,
    echo which permit redistribution without charge with its notices intact.
    echo.
    echo.
    )
)
>> "%DIST_DIR%\THIRD_PARTY_LICENSES.txt" (
echo Python
echo ------
echo Python is distributed under the Python Software Foundation License.
echo.
echo See: https://docs.python.org/3/license.html
echo.
echo.
echo Git ^(MinGit^)
echo ------------
echo Git is distributed under the GNU General Public License v2.0 ^(GPLv2^).
echo.
echo MinGit is a minimal distribution of Git for Windows.
echo Source code is available at: https://github.com/git-for-windows/git
echo.
echo See: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
echo.
)
echo   Created THIRD_PARTY_LICENSES.txt

REM Generate README.txt, whose requirements and contents depend on what ships.
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
echo BLENDER INTEGRATION
echo -------------------
echo The Blender addon is distributed separately - this bundle only
echo ships bin\ppf-cts-server.exe, the backend the addon connects to.
echo.
echo   1. Download and install the ppf-contact-solver Blender addon
echo      following the addon's own install instructions.
echo   2. In Blender, select "Windows Native" from the server type dropdown
echo   3. Set the "Solver Path" to this directory
echo   4. Click Connect
echo.
echo.
echo REQUIREMENTS
echo ------------
)
REM ANY GPU BACKEND TAKES THIS TEXT, which names each one that shipped. Keyed on
REM CUDA alone, a ROCm-only x64 distribution fell through to the CPU-only text
REM below and told its reader no GPU is used.
if defined PPF_WIN_GPU_BACKENDS (
    >> "%DIST_DIR%\README.txt" (
    echo - Windows 10/11 ^(64-bit^)
    if "!HAS_CUDA!"=="1" (
        echo - For the CUDA backend: an NVIDIA GPU and its driver. The CUDA toolkit
        echo   is not required, since the runtime is bundled.
    )
    if "!HAS_ROCM!"=="1" (
        echo - For the ROCm backend: an AMD Radeon GPU of a family this build carries
        echo   code for, and the AMD GPU driver. No ROCm installation is needed: the
        echo   HIP runtime ships in bin\, and one that is installed is not used.
    )
    echo - Which backend a run uses is chosen when it starts: the one GPU build
    echo   here, or the first whose solver reports a usable device. In Blender,
    echo   the GPU Backend dropdown names one explicitly.
    echo - Without a supported GPU the CPU backend in target\cpu\release still
    echo   runs, much more slowly. In Blender, set Compute Device to CPU.
    echo.
    echo.
    echo CONTENTS
    echo --------
    echo bin/           - The backend library of each GPU backend, the runtime each loads, and ffmpeg
    )
) else (
    >> "%DIST_DIR%\README.txt" (
    if "!PPF_WIN_ARCH!"=="arm64" (
        echo - Windows 11 on ARM ^(ARM64^)
    ) else (
        echo - Windows 10/11 ^(64-bit^)
    )
    echo - No GPU is used. This distribution carries the CPU backend only, in
    echo   target\cpu\release. In Blender, set Compute Device to CPU.
    if "!PPF_WIN_ARCH!"=="arm64" (
        echo - The fTetWild tetrahedralizer is not included: its Python package
        echo   publishes no Windows on ARM build. A scene that tetrahedralizes with
        echo   the default backend="ftetwild" is refused with that reason, and
        echo   backend="tetgen" tetrahedralizes with TetGen, which is included.
    )
    echo.
    echo.
    echo CONTENTS
    echo --------
    echo bin/           - ffmpeg
    )
)
>> "%DIST_DIR%\README.txt" (
echo target/        - The solver, server and Python extension, one directory per backend
echo python/        - Bundled Python environment
echo mingit/        - Embedded Git for repository cloning
echo frontend/      - Python frontend package
echo examples/      - Example Jupyter notebooks
echo licenses/      - License texts of what this distribution redistributes
echo config.bat     - Port configuration
echo start.bat      - JupyterLab launcher
echo start-jupyterlab.pyw - GUI launcher
echo headless.bat   - Run examples without JupyterLab
echo fast-check-all.bat - Run all example notebooks as tests
echo clear-cache.bat - Clear cache directories
echo clear-all.bat  - Full cleanup including session data
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

REM Copy fast-check and utility files
copy "%BUILD_WIN%\fast-check-all.bat" "%DIST_DIR%\fast-check-all.bat" >nul
copy "%BUILD_WIN%\inject_fast_check.py" "%DIST_DIR%\inject_fast_check.py" >nul
copy "%BUILD_WIN%\clear-cache.bat" "%DIST_DIR%\clear-cache.bat" >nul
copy "%BUILD_WIN%\clear-all.bat" "%DIST_DIR%\clear-all.bat" >nul
copy "%SRC_DIR%\.github\workflows\scripts\examples.txt" "%DIST_DIR%\examples.txt" >nul
echo   Copied fast-check and utility files

REM ============================================================
REM The distribution's gates
REM ============================================================
echo.
echo [10/10] Checking the distribution...

REM GATE: EVERY PE IMAGE IS THIS ARCHITECTURE'S. scripts\pe-audit.py states why, and
REM the exemptions below name each file of another architecture that ships and
REM the reason it may. An exemption that matches no such file fails the gate, so
REM this list cannot outlive what it excuses.
REM
REM x64, measured over an x64 bundle of this interpreter and MinGit:
REM 411 PE images, of which 44 are IL-only .NET assemblies with no machine code,
REM 348 are x64, and the 19 below are x86 or ARM64. pip's vendored distlib and
REM setuptools carry script launcher templates for every architecture they can
REM write a script for, and copy the one matching the interpreter when they do.
REM debugpy's none-any wheel carries the helpers for attaching to a process of each
REM architecture. MinGit ships Git Credential Manager as a 32-bit application with
REM its native libraries, which Windows runs under WOW64, and a 32-bit helper that
REM exists to reach 32-bit processes.
REM
REM ARM64, measured on the windows-11-arm runner (Build Windows 34992900145): 474
REM PE images, of which 44 are IL-only, 324 are ARM64, and 105 are x64 or x86. The
REM gate named the 95 of them the debugpy and distlib entries did not already excuse,
REM and each family below is one of those:
REM   - MinGit's ARM64 build carries Git for Windows' MSYS2 POSIX layer as x64 only,
REM     which ARM64 Windows runs under x64 emulation: 82 files in usr\bin, the
REM     three ssh helpers in usr\lib\ssh, and both getprocaddr helpers.
REM   - It carries the same 32-bit Git Credential Manager as the x64 build, under
REM     clangarm64\bin.
REM   - python-build-standalone's ARM64 interpreter directory holds
REM     vcruntime140_1.dll as x64, a runtime DLL no ARM64 process can load.
REM   - debugpy's none-any wheel carries two x64 Cython speedups built for CPython
REM     3.9, which the 3.12 interpreter never imports.
set "PE_EXEMPT=--exempt "python/Lib/site-packages/pip/_vendor/distlib/*.exe=pip's distlib launcher templates for every architecture, of which pip copies the interpreter's own""
if "!PPF_WIN_ARCH!"=="x64" (
    set PE_EXEMPT=!PE_EXEMPT! --exempt "python/Lib/site-packages/setuptools/*.exe=setuptools launcher templates for every architecture, of which it copies the interpreter's own"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "python/Lib/site-packages/debugpy/_vendored/pydevd/pydevd_attach_to_process/*_x86.*=debugpy helpers for attaching to a 32-bit process, which the bundled interpreter is not"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "mingit/mingw64/bin/git-credential-manager.exe=Git Credential Manager, which MinGit ships as a 32-bit application run under WOW64"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "mingit/mingw64/bin/msalruntime_x86.dll=a native library of the 32-bit Git Credential Manager"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "mingit/mingw64/bin/libSkiaSharp.dll=a native library of the 32-bit Git Credential Manager"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "mingit/mingw64/bin/libHarfBuzzSharp.dll=a native library of the 32-bit Git Credential Manager"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "mingit/mingw64/bin/av_libglesv2.dll=a native library of the 32-bit Git Credential Manager"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "mingit/usr/libexec/getprocaddr32.exe=Git for Windows' helper for reaching 32-bit processes, 32-bit by purpose"
) else (
    set PE_EXEMPT=!PE_EXEMPT! --exempt "python/Lib/site-packages/debugpy/_vendored/pydevd/pydevd_attach_to_process/*_x86.*=debugpy helpers for attaching to a 32-bit x86 process, which the bundled interpreter is not"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "python/Lib/site-packages/debugpy/_vendored/pydevd/pydevd_attach_to_process/*_amd64.*=debugpy helpers for attaching to an x64 process, which the bundled interpreter is not"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "python/Lib/site-packages/debugpy/_vendored/pydevd/_pydevd_*/*.cp39-win_amd64.pyd=debugpy's x64 Cython speedups for CPython 3.9, which this interpreter never imports"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "python/vcruntime140_1.dll=an x64-only runtime DLL python-build-standalone ships beside its ARM64 interpreter, which no ARM64 process can load"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "mingit/usr/bin/*=Git for Windows' MSYS2 POSIX layer, which MinGit's ARM64 build ships as x64 and Windows runs under x64 emulation"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "mingit/usr/lib/ssh/*=the ssh helpers of Git for Windows' x64 MSYS2 layer, run under x64 emulation"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "mingit/usr/libexec/getprocaddr*.exe=Git for Windows' helpers for reaching 32-bit and 64-bit x86 processes, one of each by purpose"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "mingit/clangarm64/bin/git-credential-manager.exe=Git Credential Manager, which MinGit ships as a 32-bit application in its ARM64 build too"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "mingit/clangarm64/bin/msalruntime_x86.dll=a native library of the 32-bit Git Credential Manager"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "mingit/clangarm64/bin/libSkiaSharp.dll=a native library of the 32-bit Git Credential Manager"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "mingit/clangarm64/bin/libHarfBuzzSharp.dll=a native library of the 32-bit Git Credential Manager"
    set PE_EXEMPT=!PE_EXEMPT! --exempt "mingit/clangarm64/bin/av_libglesv2.dll=a native library of the 32-bit Git Credential Manager"
)
"%BUILD_PYTHON%" "%BUILD_WIN%\scripts\pe-audit.py" machine --arch !PPF_WIN_ARCH! --root "%DIST_DIR%" !PE_EXEMPT!
if errorlevel 1 (
    echo ERROR: the distribution carries a binary for another architecture ^(see above^)
    exit /b 1
)

REM GATE: THE DISTRIBUTION IMPORTS ITS OWN FRONTEND AND LOADS ITS OWN BUILD. The
REM bundled interpreter is started with PYTHONHOME and PYTHONPATH pointing at
REM directories that do not exist and a PATH holding nothing but the distribution
REM and Windows, which is what build-mac-native/bundle.sh does to its payload.
REM An interpreter the ._pth file does not isolate fails to start; one that finds
REM the frontend or the cdylib anywhere but the distribution fails the comparison.
REM Importing the frontend makes it search this payload's target\<backend>
REM directories and load one of their cdylibs, which also proves the runtime DLLs
REM in bin\ resolve. Nothing is pinned where a GPU backend ships, exactly as the
REM launchers leave it.
set "SMOKE_PATH=%DIST_DIR%\python;%BIN_DIR%;%SystemRoot%\System32;%SystemRoot%"
set "SMOKE_TARGET="
if not defined PPF_WIN_GPU_BACKENDS set "SMOKE_TARGET=%DIST_DIR%\target\cpu"
setlocal
set "PYTHONHOME=%DIST_DIR%\no-such-pythonhome"
set "PYTHONPATH=%DIST_DIR%\no-such-pythonpath"
set "PATH=%SMOKE_PATH%"
set "CARGO_TARGET_DIR=%SMOKE_TARGET%"
REM The payload is already compiled to bytecode, and this import must not write
REM a __pycache__ entry into it.
set "PYTHONDONTWRITEBYTECODE=1"
"%DIST_DIR%\python\python.exe" -c "import os, sys; root = os.path.normcase(os.path.abspath(sys.argv[1])); import frontend; f = os.path.normcase(os.path.abspath(frontend.__file__)); a = os.path.normcase(os.path.abspath(frontend.artifact_dir())); print('  frontend  ' + f); print('  artifacts ' + a); sys.exit(0 if f.startswith(root) and a.startswith(root) else 3)" "%DIST_DIR%"
set SMOKE_RC=%ERRORLEVEL%
endlocal & set SMOKE_RC=%SMOKE_RC%
if not "%SMOKE_RC%"=="0" (
    echo ERROR: the distribution's interpreter did not import its own frontend and build ^(exit %SMOKE_RC%^)
    exit /b 1
)
echo   [OK] the bundled interpreter imports the distribution's frontend and loads its build

REM ============================================================
REM Summary
REM ============================================================
echo.
echo ============================================================
echo   BUNDLE COMPLETE!
echo ============================================================
echo.
echo Distribution created at: %DIST_DIR%
echo Architecture !PPF_WIN_ARCH!, backends !PPF_WIN_BACKENDS!, archive suffix !PPF_WIN_DIST_SUFFIX!
echo.
echo Contents:
for %%B in (!PPF_WIN_GPU_BACKENDS!) do (
    echo   target/%%B/release/
    for %%F in ("%DIST_DIR%\target\%%B\release\*") do echo     %%~nxF
)
echo   target/cpu/release/
for %%F in ("%CPU_TARGET_DIR%\*") do echo     %%~nxF
echo   bin/
for %%F in ("%BIN_DIR%\*") do echo     %%~nxF
echo   licenses/
for %%F in ("%LICENSES_DIR%\*") do echo     %%~nxF
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
