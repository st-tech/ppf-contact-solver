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

# paramiko INTO BLENDER'S OWN PYTHON, for the same reason the Linux GPU leg
# installs it: `bl_mcp_connection_refusals` is declared for the real backend on
# this branch, and its connect-family check drives `connect_ssh`, whose operator
# polls on `module_exists(["paramiko"])`. A failed poll surfaces as "Operator
# bpy.ops.ssh.run_command.poll() failed, context is incorrect", which reads like
# a UI-context bug rather than a missing module, and the scenario's own install
# call is asynchronous so the check would otherwise race a pip download.
#
# Blender's bundled interpreter is the one that matters here, NOT
# PPF_BUILD_PYTHON: the operator polls inside Blender. Mirrors the addon's
# install_module(): ensurepip, then pip --target the addon modules dir, which is
# on Blender's sys.path.
Write-Host "=== install paramiko into Blender's Python ==="
& "$env:PPF_BLENDER_BIN" -b --factory-startup --python-expr "import bpy,sys,subprocess; t=bpy.utils.user_resource('SCRIPTS', path='addons/modules', create=True); subprocess.run([sys.executable,'-m','ensurepip','--upgrade'], check=False); subprocess.check_call([sys.executable,'-m','pip','install','--target',t,'paramiko']); print('PARAMIKO_TARGET', t)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "paramiko install failed (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}
& "$env:PPF_BLENDER_BIN" -b --factory-startup --python-expr "import bpy,sys; sys.path.insert(0, bpy.utils.user_resource('SCRIPTS', path='addons/modules')); import paramiko; print('PARAMIKO_OK', paramiko.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "paramiko not importable inside Blender (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}
