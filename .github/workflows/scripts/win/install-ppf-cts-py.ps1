# File: install-ppf-cts-py.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Build the PyO3 extension `_ppf_cts_py` via maturin and install it
# into the embedded Python that build-win-native dropped at
# C:\ppf-contact-solver\build-win-native\python\python.exe. Mirrors
# the maturin develop step in blender.yml; the frontend hard-imports
# `_ppf_cts_py` so without this step the rig fails at import time.
#
# build.bat already installed Rust under build-win-native\rust and
# the portable MSVC under build-win-native\msvc. We reuse both so
# maturin can drive cl.exe / link.exe and rustc without going through
# Visual Studio's Developer PowerShell.

$ErrorActionPreference = "Stop"

$BuildWin   = "C:\ppf-contact-solver\build-win-native"
$Python     = Join-Path $BuildWin "python\python.exe"
$PythonFull = Join-Path $BuildWin "python_full\python.exe"
$RustDir    = Join-Path $BuildWin "rust"
$MsvcDir    = Join-Path $BuildWin "msvc"

if (-not (Test-Path $Python)) {
    Write-Error "Embedded python.exe missing at $Python (run warmup.bat first)"
    exit 1
}
# python_full is the regular CPython install with libs\python3.lib +
# include\, used as the maturin --interpreter because the embedded
# Python strips libs/ and the abi3 link step needs python3.lib.
if (-not (Test-Path $PythonFull)) {
    Write-Error "Full python.exe missing at $PythonFull (run warmup.bat first)"
    exit 1
}
if (-not (Test-Path $RustDir)) {
    Write-Error "Local Rust toolchain missing at $RustDir (run warmup.bat first)"
    exit 1
}

# Load portable MSVC env so cl.exe / link.exe are on PATH for cargo's
# build script. setup_x64.bat is the mmozeiko portable MSVC script
# build.bat already calls; setup.bat is the older single-arch variant.
$MsvcSetup = Join-Path $MsvcDir "setup_x64.bat"
if (-not (Test-Path $MsvcSetup)) {
    $MsvcSetup = Join-Path $MsvcDir "setup.bat"
}
if (-not (Test-Path $MsvcSetup)) {
    Write-Error "Portable MSVC setup script missing under $MsvcDir"
    exit 1
}

# Run the .bat in cmd and re-export its env into PowerShell. Use `set`
# rather than `set | sort` so values with `=` (e.g. some path lists)
# round-trip cleanly.
$envDump = & cmd /c "`"$MsvcSetup`" >nul 2>&1 && set"
foreach ($line in $envDump) {
    if ($line -match "^([^=]+)=(.*)$") {
        Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2]
    }
}

# Put the local Rust toolchain ahead of anything user-installed so
# we use the same rustc that build.bat already exercised.
$env:PATH        = (Join-Path $RustDir "bin") + ";" + $env:PATH
$env:RUSTUP_HOME = Join-Path $RustDir "rustup"
$env:CARGO_HOME  = $RustDir

# Install pip + maturin into python_full (the build interpreter).
# Mirrors build.bat step [4/5]: maturin must run against a Python that
# has libs\python3.lib for the abi3 link step.
& $PythonFull -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonFull -m pip install --no-warn-script-location maturin
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Build the abi3 wheel using python_full as both runner and
# --interpreter. PyO3's build script picks an interpreter from PATH
# if PYO3_PYTHON isn't set, and `--interpreter` alone doesn't override
# that, so prepend python_full to PATH and pin PYO3_PYTHON explicitly
# (same workaround build.bat applies).
$env:PATH        = (Split-Path $PythonFull -Parent) + ";" + $env:PATH
$env:PYO3_PYTHON = $PythonFull
Set-Location C:\ppf-contact-solver\crates\ppf-cts-py
& $PythonFull -m maturin build --release --interpreter $PythonFull --out C:\ppf-cts-py-dist
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$wheel = Get-ChildItem -Path "C:\ppf-cts-py-dist" -Filter "ppf_cts_py-*.whl" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $wheel) {
    Write-Error "maturin build produced no wheel under C:\ppf-cts-py-dist"
    exit 1
}
Write-Host "Built wheel: $($wheel.FullName)"

# Install the abi3 wheel into the embedded Python. abi3-py38 means the
# wheel is forward-compatible across CPython 3.8+, so a wheel built
# against python_full loads in the embedded Python at runtime.
& $Python -m pip install --force-reinstall --no-deps $wheel.FullName
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Verify the extension imports and exposes the schema/process surface
# the frontend dispatchers query.
& $Python -c "import _ppf_cts_py as m; print('schema_version', m.schema_version()); print('process_name', m.process_name())"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "_ppf_cts_py installed into embedded Python at $Python"
