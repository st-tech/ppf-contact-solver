@echo off
setlocal

echo === Cleaning build files ===

REM Get the directory where this script is located
set BUILD_WIN=%~dp0
set BUILD_WIN=%BUILD_WIN:~0,-1%
for %%I in ("%BUILD_WIN%\..") do set SRC=%%~fI

echo.
echo Removing MSBuild intermediate files...
if exist "%BUILD_WIN%\simbackend_cuda" rmdir /S /Q "%BUILD_WIN%\simbackend_cuda"

echo Removing C++/CUDA build output...
if exist "%SRC%\src\cpp\build" rmdir /S /Q "%SRC%\src\cpp\build"

echo Removing Rust build output...
if exist "%SRC%\target" rmdir /S /Q "%SRC%\target"

echo Removing distribution folder...
if exist "%BUILD_WIN%\dist" rmdir /S /Q "%BUILD_WIN%\dist"

echo.
echo === Clean complete ===

endlocal
