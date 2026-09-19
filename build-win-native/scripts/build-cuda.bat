@echo off
REM File: scripts/build-cuda.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0
REM
REM The CUDA backend library, libsimbackend_cuda.dll. build.bat CALLs this where
REM CUDA is a backend, with the portable MSVC environment already loaded and these
REM set: BUILD_WIN, SRC_DIR, CPP_DIR, KERNEL_DIR, KERNELGEN, OUT_DIR,
REM KERNELGEN_DIR, LIB_DIR, DEPS, EIGEN_STEM, CUDA_PATH and PYTHON_EXE.
REM
REM IT IS A FILE OF ITS OWN BECAUSE cmd.exe CANNOT SKIP IT ANY OTHER WAY. build.bat
REM builds this library for one backend of three, and the two ways to make a run of
REM statements conditional in a batch file both fail here. A parenthesized block
REM percent-expands its whole body when it is parsed, before any line in it runs,
REM and this recipe sets and then reads a dozen variables with %% expansion, so
REM inside a block every one of them would be substituted empty. A goto over it
REM depends on cmd.exe finding the label, and in a file with LF line endings, which
REM these files are, the label search reads in fixed-size chunks and can miss a
REM label depending on where it falls. A CALL of a separate file does neither.
REM
REM THE CUDA BUILD MIRRORS THE LINUX MAKEFILE (crates/ppf-cts-compute/cuda/Makefile).
REM It renders the neutral `*.kernel.cpp` bodies with the transcompiler
REM (scripts/gen_cuda.py drives seam/kernelgen.py) into the .kernel.cu / .args.cuh
REM the ABI translation unit includes and the .entry.cu the entry-declaring kernels
REM need, assembles the kernel table, then compiles backend/backend.cu, the two
REM mechanism TUs (arena, diagnostics), the generated entries, SimpleLog and the
REM print_rust stub, and device-links them into libsimbackend_cuda.dll.
REM
REM CPP_DIR is ppf-cts-compute/cuda: the CUDA MECHANISM (arena, diagnostic
REM transport, dispatch) and the ABI backend TU, none of it simulation. KERNEL_DIR
REM is the neutral kernel tree both platforms compile against, in ppf-cts-solver;
REM the Linux recipe takes it as `KERNEL_ROOT`. There is no LOGIC_DIR any more,
REM because there is no third directory holding simulation: every kernel is a
REM neutral *.kernel.cpp the transcompiler renders.

setlocal enabledelayedexpansion

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

REM Source files. The CUDA backend library is, exactly as the Linux Makefile
REM builds it, the one ABI translation unit backend\backend.cu, the two mechanism
REM TUs arena\arena.cu and diagnostics\diagnostics.cu, the host C++ SimpleLog and
REM print_rust stub, and one object per generated entry point. There is no
REM hand-listed "logic" set any more, because no directory holds simulation
REM written against CUDA: every kernel is a neutral *.kernel.cpp the
REM transcompiler renders below.
set CPP_SRCS=%KERNEL_DIR%\simplelog\SimpleLog.cpp %KERNEL_DIR%\stub.cpp
set BACKEND_SRC=%CPP_DIR%\backend\backend.cu
set MECH_SRCS=%CPP_DIR%\arena\arena.cu %CPP_DIR%\diagnostics\diagnostics.cu

REM CUDA architectures, read from the same manifest the Linux Makefile reads
REM (crates\ppf-cts-compute\cuda\cuda_arch.txt). Neither build spells the list
REM out, so the two cannot come to ship different architectures, and neither can
REM disagree with the SUPPORTED_SM gate that is generated from the same file.
REM `for /f` skips blank lines and ';' comments by default, which is why the
REM manifest comments with ';'. The floor is read in its own pass so the file
REM does not have to list it before the cubins.
set CUDA_ARCH_FILE=%CPP_DIR%\cuda_arch.txt
if not exist "%CUDA_ARCH_FILE%" (
    echo ERROR: CUDA architecture manifest not found: %CUDA_ARCH_FILE%
    exit /b 1
)
set ARCH_FLOOR=
set GENCODE=
set ARCH_LIST_STR=
for /f "tokens=1,2" %%A in (%CUDA_ARCH_FILE%) do (
    if "%%A"=="floor" set ARCH_FLOOR=%%B
)
if not defined ARCH_FLOOR (
    echo ERROR: no 'floor' line in %CUDA_ARCH_FILE%
    exit /b 1
)
for /f "tokens=1,2" %%A in (%CUDA_ARCH_FILE%) do (
    if "%%A"=="cubin" (
        set "GENCODE=!GENCODE! -gencode arch=compute_!ARCH_FLOOR!,code=sm_%%B"
        if defined ARCH_LIST_STR (
            set "ARCH_LIST_STR=!ARCH_LIST_STR!, sm_%%B"
        ) else (
            set "ARCH_LIST_STR=sm_%%B"
        )
    )
)
REM Stop here rather than link a DLL with no device code. That DLL builds and
REM installs cleanly and then rejects every GPU at run time, and the run-time
REM failure names the device rather than the manifest that went missing.
if not defined GENCODE (
    echo ERROR: no 'cubin' lines in %CUDA_ARCH_FILE%
    exit /b 1
)
echo CUDA architectures from manifest: floor compute_!ARCH_FLOOR!, cubins !ARCH_LIST_STR!

REM Compiler flags. Device link-time optimization (LTO), matching the Linux
REM Makefile: compile each TU to an LTO intermediate (code=lto_<floor>), then
REM device-link with -dlto so cross-TU device callees (notably
REM barrier::compute_stiffness) inline into the contact-Hessian embed kernels,
REM roughly halving contact matrix-assembly cost.
REM IMPORTANT: device LTO CANNOT embed a JIT-able PTX (the -dlto link lowers its IR
REM straight to SASS; a code=compute_XX request is silently dropped), so the .dll
REM is frozen to the exact cubins we ship, no forward-JIT fallback. One native
REM SASS cubin per supported arch is therefore emitted at the link, from the
REM manifest above. nvcc forbids -dlto beside -gencode at compile (hence
REM code=lto_<floor>), but needs the full -gencode list at the link.
set NVCC_COMMON=-std=c++17 --expt-relaxed-constexpr --extended-lambda -O3 -Wno-deprecated-gpu-targets --diag-error=2803
REM -Werror=attributes (Linux NVCC_FLAGS) is a GCC flag MSVC rejects (D8021), so it is
REM not passed here; the Linux build and CI enforce that attribute check.
set NVCC_DEFINES=-DWIN32 -DNDEBUG -D_WINDOWS -D_USRDLL -D__NVCC__ -DEIGEN_WARNINGS_DISABLED -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT -DSHIPPED_ARCH_STR="\"!ARCH_LIST_STR!\""
set NVCC_INCLUDES=-I"%EIGEN_DIR%" -I"%KERNELGEN_DIR%" -I"%CPP_DIR%" -I"%KERNEL_DIR%"
set NVCC_XCOMPILER=-Xcompiler "/EHsc /W0 /MD /O2"
set NVCC_SUPPRESS=--diag-suppress=1222,2527,2529,2651,2653,2668,2669,2670,2671,2735,2737,2739,20012,20011,20014,177,940,1394

set OBJ_DIR=%OUT_DIR%\obj
if not exist "%OBJ_DIR%" mkdir "%OBJ_DIR%"
if not exist "%KERNELGEN_DIR%" mkdir "%KERNELGEN_DIR%"
set ENTRY_OBJ_DIR=%OUT_DIR%\cuda-entry
if not exist "%ENTRY_OBJ_DIR%" mkdir "%ENTRY_OBJ_DIR%"

REM Transcompiler. Render every neutral *.kernel.cpp into the .kernel.cu the ABI
REM TU includes and the .args.cuh / kernel table it needs, and one .entry.cu per
REM kernel that declares an entry. gen_cuda.py walks the tree in the same sorted
REM order the Linux `find | sort` does, so the kernel ids (a table INDEX) match,
REM and prints the entry .cu files this build must compile.
echo Rendering neutral kernels with the transcompiler...
set ENTRY_LIST=%OUT_DIR%\entry-cu-list.txt
"%PYTHON_EXE%" "%BUILD_WIN%\scripts\gen_cuda.py" --kernel-root "%KERNEL_DIR%" --kernelgen "%KERNELGEN%" --gen-dir "%KERNELGEN_DIR%" > "%ENTRY_LIST%"
if errorlevel 1 (
    echo ERROR: transcompiler generation failed
    exit /b 1
)

REM The entry-context headers, matching the Linux ENTRY_CONTEXT_FLAGS: a generated
REM entry point names Vec3f / Mat3x3f and the eigen entry points with no include of
REM its own, so the vocabulary is force-included ahead of it.
set ENTRY_CONTEXT_FLAGS=-include data.hpp -include linalg/eigsolve.hpp

echo Compiling the ABI TU (backend.cu) with the vocabulary headers...
set OBJS=
REM THE OBJECT LIST GOES TO A RESPONSE FILE, NOT ONTO THE COMMAND LINE.
REM
REM cmd.exe caps a command line at 8191 characters and TRUNCATES rather than
REM failing loudly. Every object here is an absolute path, so the list grows
REM with both the entry count and the length of the install root, and the port
REM branch crossed that cap: measured, ~85 objects under
REM `C:\ppf-contact-solver` (the CI checkout) reach about 8220 characters
REM against about 7100 for the same tree under a short root like `C:\port`.
REM That is why this link succeeded on a developer box and failed every time
REM in CI, with the same two lines in build.log each run.
REM
REM `-optf` is nvcc's own options-file flag, so the list is passed as a file
REM and the command line stays short whatever the root is called or how many
REM kernels declare an entry.
set OBJS_RSP=%OUT_DIR%\link-objects.rsp
if exist "!OBJS_RSP!" del /q "!OBJS_RSP!"

REM EVERY COMPILE BELOW IS INDEPENDENT, AND THEY RUN N AT A TIME. Each loop
REM used to run its nvcc inline, one after another: measured on the release
REM builder, the CUDA library took about ten minutes that way with all but one
REM core idle, while the ROCm build beside it (build_rocm.py) compiles through
REM a pool. So the loops now only WRITE the command lines they would have run,
REM one per line as `<label>|<command>`, and scripts\run_jobs.py runs the file
REM NUMBER_OF_PROCESSORS at a time (PPF_WIN_BUILD_JOBS overrides), printing
REM each command's output in job order and stopping at the first failure with
REM that command's output and exit code. The device link after them is one
REM nvcc invocation and is left as it was. `^|` is the escaped separator: a bare
REM `|` would make cmd pipe the echo.
set COMPILE_JOBS=%OUT_DIR%\compile-jobs.txt
if exist "!COMPILE_JOBS!" del /q "!COMPILE_JOBS!"
>>"!COMPILE_JOBS!" echo backend.cu^|%NVCC% -dc -gencode arch=compute_!ARCH_FLOOR!,code=lto_!ARCH_FLOOR! %NVCC_COMMON% %NVCC_DEFINES% %NVCC_INCLUDES% %ENTRY_CONTEXT_FLAGS% %NVCC_XCOMPILER% %NVCC_SUPPRESS% "%BACKEND_SRC%" -o "%OBJ_DIR%\backend.obj"
set OBJS=!OBJS! "%OBJ_DIR%\backend.obj"
>>"!OBJS_RSP!" echo "%OBJ_DIR%\backend.obj"

echo Compiling the mechanism TUs (arena, diagnostics)...
for %%f in (%MECH_SRCS%) do (
    >>"!COMPILE_JOBS!" echo %%~nxf^|%NVCC% -dc -gencode arch=compute_!ARCH_FLOOR!,code=lto_!ARCH_FLOOR! %NVCC_COMMON% %NVCC_DEFINES% %NVCC_INCLUDES% %NVCC_XCOMPILER% %NVCC_SUPPRESS% "%%f" -o "%OBJ_DIR%\%%~nf.obj"
    set OBJS=!OBJS! "%OBJ_DIR%\%%~nf.obj"
    >>"!OBJS_RSP!" echo "%OBJ_DIR%\%%~nf.obj"
)

echo Compiling the generated entry points...
for /f "usebackq delims=" %%f in ("%ENTRY_LIST%") do (
    >>"!COMPILE_JOBS!" echo entry %%~nxf^|%NVCC% -dc -gencode arch=compute_!ARCH_FLOOR!,code=lto_!ARCH_FLOOR! %NVCC_COMMON% %NVCC_DEFINES% %NVCC_INCLUDES% %ENTRY_CONTEXT_FLAGS% %NVCC_XCOMPILER% %NVCC_SUPPRESS% "%%f" -o "%ENTRY_OBJ_DIR%\%%~nf.obj"
    set OBJS=!OBJS! "%ENTRY_OBJ_DIR%\%%~nf.obj"
    >>"!OBJS_RSP!" echo "%ENTRY_OBJ_DIR%\%%~nf.obj"
)

echo Compiling host C++ TUs...
for %%f in (%CPP_SRCS%) do (
    >>"!COMPILE_JOBS!" echo %%~nxf^|%NVCC% -c %NVCC_COMMON% %NVCC_DEFINES% %NVCC_INCLUDES% %NVCC_XCOMPILER% %NVCC_SUPPRESS% "%%f" -o "%OBJ_DIR%\%%~nf.obj"
    set OBJS=!OBJS! "%OBJ_DIR%\%%~nf.obj"
    >>"!OBJS_RSP!" echo "%OBJ_DIR%\%%~nf.obj"
)

set COMPILE_JOBS_N=%NUMBER_OF_PROCESSORS%
if defined PPF_WIN_BUILD_JOBS set COMPILE_JOBS_N=%PPF_WIN_BUILD_JOBS%
echo Running the compiles, !COMPILE_JOBS_N! at a time...
"%PYTHON_EXE%" "%BUILD_WIN%\scripts\run_jobs.py" "!COMPILE_JOBS!" --jobs !COMPILE_JOBS_N!
if errorlevel 1 (
    echo ERROR: a CUDA compile failed; the failing command and its output are above
    exit /b 1
)

REM The ABI functions (be_*) are declared extern "C" with no export attribute,
REM because on Linux and macOS a shared library exports every default-visibility
REM symbol and the reference build needs nothing. MSVC exports nothing from a DLL
REM unless told to, and has no export-all-symbols equivalent, so the device-link
REM is handed an explicit EXPORTS list. gen_def.py reads it from the ABI header,
REM the single source of truth, so the list cannot drift from the declarations.
set ABI_HEADER=%KERNEL_DIR%\seam\backend_abi.h
set DEF_FILE=%OUT_DIR%\backend-exports.def
"%PYTHON_EXE%" "%BUILD_WIN%\scripts\gen_def.py" "%ABI_HEADER%" > "%DEF_FILE%"
if errorlevel 1 (
    echo ERROR: gen_def.py failed to produce the export list
    exit /b 1
)

echo Device-linking with LTO (-dlto: native SASS cubins !ARCH_LIST_STR!, no PTX)...
REM Delete any prior DLL first so the existence check below cannot pass on a
REM stale artifact. nvcc's -dlto -shared device-link has been observed to exit
REM 0 on Windows even when the host link.exe step fails with a fatal LNK error
REM (measured: LNK1120 unresolved externals, nvcc rc=0), so the errorlevel
REM check alone is not sufficient. Requiring the DLL to actually exist afterward
REM makes a failed link stop the build loudly rather than falling through to a
REM Rust link that fails obscurely on the missing backend symbols.
del /q "%LIB_DIR%\libsimbackend_cuda.dll" 2>nul
REM -Wno-deprecated-gpu-targets matches NVCC_COMMON: the -gencode list lands at
REM the device link, and nvcc 12.8 warns per deprecated target now that the
REM floor and the sm_61 cubin are both below 75.
REM THE DEVICE LINK GETS ITS OWN LOG, AND IT IS PRINTED ON FAILURE.
REM The guard below has always said "see LNK errors above", and on a CI runner
REM that was not true: the self-relaunch at the top of build.bat tees
REM `cmd /c <self> 2>&1`, and nvcc's host link.exe step still reached the
REM console without passing through that pipe, so build.log ended at the ERROR
REM line with nothing above it to see. A failure was therefore reduced to
REM "device-link produced no DLL" for anyone who could not watch the console
REM live, which on a disposable EC2 builder is everyone.
REM
REM Redirecting here captures it at the source, and typing the file back out
REM puts it INTO build.log, which is the artifact the workflow uploads. The
REM redirect is on the nvcc line rather than around the whole script because
REM only this command's diagnostics were escaping.
REM THE RESPONSE FILE MUST HOLD EVERY OBJECT THAT WAS COMPILED, and this
REM asserts it rather than trusting the loops above. A short list is exactly
REM what the 8191-character truncation used to produce, and its symptom was a
REM link that "succeeded" into unresolved externals, so counting here turns a
REM silent wrong-answer into a named failure.
set /a OBJ_COUNT=0
for /f %%C in ('type "!OBJS_RSP!" ^| find /c /v ""') do set /a OBJ_COUNT=%%C
echo Linking !OBJ_COUNT! objects from the response file.
if !OBJ_COUNT! LSS 4 (
    echo ERROR: the link response file holds only !OBJ_COUNT! objects.
    echo        Expected the ABI TU, two mechanism TUs, the generated entries
    echo        and the host C++ TUs. Something above failed to record its object.
    exit /b 1
)

set DEVLINK_LOG=%OUT_DIR%\device-link.log
%NVCC% -shared -dlto -t 0 -Wno-deprecated-gpu-targets !GENCODE! -Xcompiler "/MD" -optf "!OBJS_RSP!" -lcudart -Xlinker "/DEF:%DEF_FILE%" -o "%LIB_DIR%\libsimbackend_cuda.dll" > "!DEVLINK_LOG!" 2>&1
set DEVLINK_RC=!ERRORLEVEL!
if exist "!DEVLINK_LOG!" (
    echo --- device-link output ---
    type "!DEVLINK_LOG!"
    echo --- end device-link output ---
)
if not "!DEVLINK_RC!"=="0" (
    echo ERROR: nvcc device-link failed
    exit /b 1
)
if not exist "%LIB_DIR%\libsimbackend_cuda.dll" (
    echo ERROR: device-link produced no DLL ^(host link.exe step failed; see LNK errors above^)
    exit /b 1
)
echo   [DONE] libsimbackend_cuda.dll created
endlocal
exit /b 0
