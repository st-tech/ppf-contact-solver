@echo off
REM File: make-slim-ffmpeg.bat
REM Code: Claude Code
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0
REM
REM Build slim ffmpeg for Windows using MSYS2/MinGW-w64

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set BUILD_WIN=%~dp0
set BUILD_WIN=%BUILD_WIN:~0,-1%

set FFMPEG_DIR=%BUILD_WIN%\ffmpeg
set MSYS2_DIR=%BUILD_WIN%\msys64
set DOWNLOADS=%BUILD_WIN%\downloads

REM Check if ffmpeg already exists
if exist "%FFMPEG_DIR%\ffmpeg.exe" (
    echo ffmpeg already exists at %FFMPEG_DIR%\ffmpeg.exe
    "%FFMPEG_DIR%\ffmpeg.exe" -version | findstr /C:"ffmpeg version"
    goto :end
)

REM Create downloads directory if needed
if not exist "%DOWNLOADS%" mkdir "%DOWNLOADS%"

REM Download and install MSYS2 if not present
if not exist "%MSYS2_DIR%\usr\bin\bash.exe" (
    echo === Downloading MSYS2 ===

    set MSYS2_VERSION=20241208
    set MSYS2_URL=https://github.com/msys2/msys2-installer/releases/download/2024-12-08/msys2-base-x86_64-!MSYS2_VERSION!.sfx.exe
    set MSYS2_SFX=%DOWNLOADS%\msys2-base-x86_64-!MSYS2_VERSION!.sfx.exe

    if not exist "!MSYS2_SFX!" (
        echo Downloading MSYS2...
        curl.exe -L -o "!MSYS2_SFX!" "!MSYS2_URL!"
        if errorlevel 1 (
            echo ERROR: Failed to download MSYS2
            exit /b 1
        )
    )

    echo Extracting MSYS2 to %BUILD_WIN%...
    pushd "%BUILD_WIN%"
    "!MSYS2_SFX!" -y
    popd
    if errorlevel 1 (
        echo ERROR: Failed to extract MSYS2
        exit /b 1
    )

    echo Creating tmp directory for MSYS2...
    if not exist "%MSYS2_DIR%\tmp" mkdir "%MSYS2_DIR%\tmp"

    echo Initializing MSYS2...
    "%MSYS2_DIR%\usr\bin\bash.exe" --login -c "exit"

    echo Installing required packages...
    "%MSYS2_DIR%\usr\bin\bash.exe" --login -c "pacman -Syu --noconfirm"
    "%MSYS2_DIR%\usr\bin\bash.exe" --login -c "pacman -S --noconfirm --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-make mingw-w64-x86_64-pkg-config make nasm git diffutils"

    echo MSYS2 setup complete!
)

echo ============================================================
echo   Building Slim FFmpeg for Windows
echo ============================================================
echo.

REM Run the build script in MSYS2 MinGW64 environment
echo Running build in MSYS2 MinGW64 environment...
REM Convert Windows path to MSYS2 path (C:\foo\bar -> /c/foo/bar)
REM Extract drive letter and convert to lowercase
set DRIVE_LETTER=%BUILD_WIN:~0,1%
for %%a in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do call set DRIVE_LETTER=%%DRIVE_LETTER:%%a=%%a%%
set REST_PATH=%BUILD_WIN:~2%
set REST_PATH=%REST_PATH:\=/%
set MSYS_PATH=/%DRIVE_LETTER%%REST_PATH%
"%MSYS2_DIR%\usr\bin\bash.exe" --login -c "cd '%MSYS_PATH%' && bash make-slim-ffmpeg.sh"

REM Verify
if exist "%FFMPEG_DIR%\ffmpeg.exe" (
    echo.
    echo ============================================================
    echo   SUCCESS
    echo ============================================================
    echo.
    "%FFMPEG_DIR%\ffmpeg.exe" -version | findstr /C:"ffmpeg version"
    echo.
    dir "%FFMPEG_DIR%\ffmpeg.exe"
) else (
    echo ERROR: Build failed - ffmpeg.exe not found
    exit /b 1
)

:end
endlocal
