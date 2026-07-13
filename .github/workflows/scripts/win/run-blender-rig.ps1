# File: run-blender-rig.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Runs the headless Blender debug rig against the REAL Windows-native build,
# directly over SSH (Session 0). PPF_BLENDER_HEADLESS=1 makes the rig launch
# Blender with --background, which needs NO OpenGL/WGL context and no desktop,
# so it runs fine in a Session-0 SSH shell. This sidesteps the L4 coming up
# in TCC (compute-only) mode, which provides no WGL/OpenGL and made a GUI
# Blender launch block forever. The rig driver holds the main thread and
# drains its own PC2 frames synchronously, so each scenario completes within
# a single --python run with no event loop (blender_harness' bootstrap calls
# _start() directly when bpy.app.background). Verified: the whole real subset
# passes headless with no display.
#
# The script exits with the rig's exit code so the invoking SSH command
# reflects pass/fail; it also mirrors output to C:\ci\rig.log and the code to
# C:\ci\rig_exit.txt for artifact collection.

param(
    [double]$Timeout = 420.0,
    [int]$Parallel = 1
)

$Src = "C:\ppf-contact-solver"
$Py = "$Src\build-win-native\python\python.exe"
$CiDir = "C:\ci"

New-Item -ItemType Directory -Path $CiDir -Force | Out-Null
$Log = "$CiDir\rig.log"
Remove-Item "$CiDir\rig_exit.txt" -ErrorAction SilentlyContinue

function Log($msg) { $msg | Tee-Object -FilePath $Log -Append }

$exitCode = 1
try {
    Log "=== Blender rig (real backend, headless) starting $(Get-Date -Format o) ==="

    # Environment, mirroring the example runner (start.bat semantics): local
    # CUDA + embedded Python + the built solver on PATH so the real
    # ppf-cts-server.exe the rig spawns can load its CUDA backend lib.
    $env:PATH = "$Src\target\release;" +
                "$Src\crates\ppf-cts-solver\src\cpp\build\lib;" +
                "$Src\build-win-native\cuda\bin;" +
                "$Src\build-win-native\python;" +
                "$Src\build-win-native\python\Scripts;" +
                "$Src\build-win-native\mingit\cmd;" + $env:PATH
    $env:CUDA_PATH = "$Src\build-win-native\cuda"
    $env:PYTHONPATH = "$Src;" + $env:PYTHONPATH
    $env:PYTHONUNBUFFERED = "1"

    # Blender binary path (written by download-blender.ps1).
    if (-not (Test-Path "C:\blender_bin.txt")) {
        throw "C:\blender_bin.txt missing (run download-blender.ps1 first)"
    }
    $env:PPF_BLENDER_BIN = (Get-Content "C:\blender_bin.txt" -Raw).Trim()
    Log "PPF_BLENDER_BIN=$($env:PPF_BLENDER_BIN)"

    # Point the rig-spawned real server's Python build worker at the embedded
    # interpreter (it has frontend's deps incl. scipy/tetgen).
    $env:PPF_CTS_BUILD_PYTHON = $Py
    # Worker dirs (report.json + per-worker logs) land here; collected after.
    $env:PPF_DEBUG_ROOT = "C:\ppf-debug"
    # Run Blender headless (--background): no OpenGL/desktop required.
    $env:PPF_BLENDER_HEADLESS = "1"

    Log "Running rig: main.py runtests --backend real (headless, timeout=$Timeout parallel=$Parallel)"
    & $Py "$Src\blender_addon\debug\main.py" runtests `
        --backend real `
        --timeout $Timeout `
        --parallel $Parallel `
        --report "$CiDir\report.json" *>&1 | Tee-Object -FilePath $Log -Append
    $exitCode = $LASTEXITCODE
    Log "rig exit code: $exitCode"
}
catch {
    Log "EXCEPTION: $_"
    $exitCode = 1
}
finally {
    Set-Content -Path "$CiDir\rig_exit.txt" -Value "$exitCode" -Encoding ASCII -NoNewline
    Log "=== Blender rig finished (exit=$exitCode) $(Get-Date -Format o) ==="
    exit $exitCode
}
