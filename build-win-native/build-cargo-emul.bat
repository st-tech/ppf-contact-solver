@echo off
REM File: build-cargo-emul.bat
REM Code: Claude Code and Codex
REM Review: Ryoichi Ando (ryoichi.ando@zozo.com)
REM License: Apache v2.0
REM
REM Wrap the cargo --features emulated build with the local Rust +
REM MSVC environment that warmup.bat provisioned. Run after
REM build-emul.bat has produced libsimbackend_cpu.dll. The build.rs
REM only links to the prebuilt DLL, no compiler needed beyond cargo.

setlocal enabledelayedexpansion

set "BUILD_WIN=%~dp0"
set "BUILD_WIN=%BUILD_WIN:~0,-1%"
for %%I in ("%BUILD_WIN%\..") do set "SRC_DIR=%%~fI"

set "RUST_DIR=%BUILD_WIN%\rust"
set "RUSTUP_HOME=%RUST_DIR%\rustup"
set "CARGO_HOME=%RUST_DIR%"
set "PATH=%RUST_DIR%\bin;%PATH%"

if exist "%BUILD_WIN%\msvc\setup_x64.bat" call "%BUILD_WIN%\msvc\setup_x64.bat"

cd /d "%SRC_DIR%"
cargo +stable build --release --features emulated --no-default-features
endlocal
