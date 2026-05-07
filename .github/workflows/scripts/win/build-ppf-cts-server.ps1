# File: build-ppf-cts-server.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Build the Rust ppf-cts-server binary (workspace member) and smoke
# check it. The launcher in blender_addon/core/connection.py always
# launches this .exe at
# C:\ppf-contact-solver\target\release\ppf-cts-server.exe. The Windows
# build.bat step only builds the root `ppf-contact-solver` crate
# (cargo build --release at the workspace root, no -p), so the server
# binary is not produced by default; this script runs the missing
# build.
#
# Reuses the portable MSVC and Rust toolchain that build.bat already
# prepared under build-win-native\, the same pattern that
# install-ppf-cts-py.ps1 follows.

$ErrorActionPreference = "Stop"

$SrcDir   = "C:\ppf-contact-solver"
$BuildWin = Join-Path $SrcDir "build-win-native"
$RustDir  = Join-Path $BuildWin "rust"
$MsvcDir  = Join-Path $BuildWin "msvc"

if (-not (Test-Path $RustDir)) {
    Write-Error "Local Rust toolchain missing at $RustDir (run warmup.bat first)"
    exit 1
}

# Load portable MSVC env so cl.exe / link.exe are on PATH for cargo.
$MsvcSetup = Join-Path $MsvcDir "setup_x64.bat"
if (-not (Test-Path $MsvcSetup)) {
    $MsvcSetup = Join-Path $MsvcDir "setup.bat"
}
if (-not (Test-Path $MsvcSetup)) {
    Write-Error "Portable MSVC setup script missing under $MsvcDir"
    exit 1
}

$envDump = & cmd /c "`"$MsvcSetup`" >nul 2>&1 && set"
foreach ($line in $envDump) {
    if ($line -match "^([^=]+)=(.*)$") {
        Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2]
    }
}

$env:PATH        = (Join-Path $RustDir "bin") + ";" + $env:PATH
$env:RUSTUP_HOME = Join-Path $RustDir "rustup"
$env:CARGO_HOME  = $RustDir

Set-Location $SrcDir
& cargo build --release -p ppf-cts-server
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$exe = Join-Path $SrcDir "target\release\ppf-cts-server.exe"
if (-not (Test-Path $exe)) {
    Write-Error "ppf-cts-server.exe not found at $exe after build"
    exit 1
}
Write-Host "Built: $exe"

# Smoke test: --help prints version + usage and exits 0. Proves the
# binary loaded its deps (tokio, log4rs, ppf-cts-formats, ...) and
# parses CLI args. We discard stdout but surface the exit code.
& $exe --help | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "ppf-cts-server.exe --help exited $LASTEXITCODE"
    exit $LASTEXITCODE
}
Write-Host "ppf-cts-server.exe --help OK"
