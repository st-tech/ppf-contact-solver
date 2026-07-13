# File: install-addon.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Cold-start wrapper that installs the Blender addon on the disposable
# CI instance. Kept as a standalone -File script (not an inline
# `powershell -Command "..."`) because the instance's SSH DefaultShell
# is PowerShell: an inline command string is parsed by the outer shell
# first, so `$env:PPF_BLENDER_BIN=...` gets expanded (to empty) before
# the inner powershell.exe ever sees it. Running a real .ps1 with -File
# avoids that double-evaluation.
#
# Reads the Blender binary path written by download-blender.ps1 and
# points the addon installer's wheel-fetch at the embedded Python.

$ErrorActionPreference = "Stop"

$Src = "C:\ppf-contact-solver"

if (-not (Test-Path "C:\blender_bin.txt")) {
    Write-Error "C:\blender_bin.txt missing (run download-blender.ps1 first)"
    exit 1
}
$env:PPF_BLENDER_BIN = (Get-Content "C:\blender_bin.txt" -Raw).Trim()
$env:PPF_BUILD_PYTHON = "$Src\build-win-native\python\python.exe"
Write-Host "PPF_BLENDER_BIN=$($env:PPF_BLENDER_BIN)"
Write-Host "PPF_BUILD_PYTHON=$($env:PPF_BUILD_PYTHON)"

Set-Location $Src
& powershell -ExecutionPolicy Bypass -File "$Src\install-blender-addon.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Error "install-blender-addon.ps1 failed (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}
Write-Host "Addon installed."
