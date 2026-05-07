@echo off
REM File: build-emul.bat
REM Code: Claude Code and Codex
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0
REM
REM CUDA-free Windows build of the test-rig emulator. Produces a
REM static archive libsimbackend_cpu.lib that the Rust binary links
REM directly. Static linkage sidesteps the Windows DLL search-order
REM issue we hit when the rig server spawns the solver via tokio.

setlocal enabledelayedexpansion

set "BUILD_WIN=%~dp0"
set "BUILD_WIN=%BUILD_WIN:~0,-1%"
for %%I in ("%BUILD_WIN%\..") do set "SRC_DIR=%%~fI"

set "CPP_DIR=%SRC_DIR%\crates\ppf-cts-solver\src\cpp"
set "EMUL_DIR=%SRC_DIR%\crates\ppf-cts-solver\src\cpp_emul"
set "OUT_DIR=%EMUL_DIR%\build"
set "LIB_DIR=%OUT_DIR%\lib"
set "DEPS=%BUILD_WIN%\deps"

call "%BUILD_WIN%\scripts\load-downloads.bat"
if errorlevel 1 (
    echo ERROR: Failed to load download manifest
    exit /b 1
)
for %%I in ("%FILE_EIGEN%") do set "EIGEN_STEM=%%~nI"
if not exist "%DEPS%\%EIGEN_STEM%" (
    echo ERROR: Eigen not present at %DEPS%\%EIGEN_STEM%
    echo Please run build.bat or warmup.bat first to download Eigen.
    exit /b 1
)
set "EIGEN_DIR=%DEPS%\%EIGEN_STEM%"

set "MSVC_DIR=%BUILD_WIN%\msvc"
echo [1/2] Setting up Visual Studio environment...
if exist "%MSVC_DIR%\setup_x64.bat" (
    call "%MSVC_DIR%\setup_x64.bat"
    goto :msvc_ready
)
if exist "%MSVC_DIR%\setup.bat" (
    call "%MSVC_DIR%\setup.bat"
    goto :msvc_ready
)
echo ERROR: Portable MSVC not found at %MSVC_DIR%
echo Please run warmup.bat first to install MSVC locally.
exit /b 1
:msvc_ready

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
if not exist "%LIB_DIR%" mkdir "%LIB_DIR%"

echo.
echo ============================================================
echo [2/2] Building libsimbackend_cpu.lib (static) with cl + lib
echo ============================================================
echo.

set "CL_FLAGS=/nologo /std:c++17 /EHsc /MD /O2 /W0 /c"
set "CL_DEFINES=/DWIN32 /DNDEBUG /D_WINDOWS /DEIGEN_WARNINGS_DISABLED"

cd /d "%EMUL_DIR%"
cl.exe %CL_FLAGS% %CL_DEFINES% /I"%EIGEN_DIR%" main.cpp /Fo:"%OUT_DIR%\\main.obj"
if errorlevel 1 (
    echo ERROR: cl.exe compile failed
    exit /b 1
)

if exist "%LIB_DIR%\libsimbackend_cpu.lib" del /q "%LIB_DIR%\libsimbackend_cpu.lib"
lib.exe /nologo /OUT:"%LIB_DIR%\libsimbackend_cpu.lib" "%OUT_DIR%\main.obj"
if errorlevel 1 (
    echo ERROR: lib.exe failed
    exit /b 1
)

REM Stale DLL from prior dynamic-link builds is harmless but confusing.
if exist "%LIB_DIR%\libsimbackend_cpu.dll" del /q "%LIB_DIR%\libsimbackend_cpu.dll"
if exist "%LIB_DIR%\libsimbackend_cpu.exp" del /q "%LIB_DIR%\libsimbackend_cpu.exp"

echo   [DONE] %LIB_DIR%\libsimbackend_cpu.lib (static)

endlocal
