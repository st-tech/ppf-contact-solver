# File: run-blender-rig.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Runs the headless Blender debug rig against the REAL Windows-native build,
# directly over SSH (Session 0). PPF_BLENDER_HEADLESS=1 makes the rig launch
# Blender with --background, which needs NO OpenGL/WGL context and no desktop,
# so it runs fine in a Session-0 SSH shell whatever driver model the GPU comes
# up in. An L4 comes up in TCC (compute-only) mode, which provides no
# WGL/OpenGL and made a GUI Blender launch block forever. The rig driver holds
# the main thread and drains its own PC2 frames synchronously, so each
# scenario completes within a single --python run with no event loop
# (blender_harness' bootstrap calls _start() directly when bpy.app.background).
# Verified on an L4: the whole real subset passes headless with no display.
#
# The script exits with the rig's exit code so the invoking SSH command
# reflects pass/fail; it also mirrors output to C:\ci\rig.log and the code to
# C:\ci\rig_exit.txt for artifact collection.

param(
    [double]$Timeout = 420.0,
    [int]$Parallel = 1,
    # Space-separated scenario names; empty runs the whole set.
    [string]$Scenarios = "",
    # I/N: this host's share of the selection; empty runs all of it.
    [string]$Shard = "",
    # The tree the rig runs from. CI unpacks the build payload here; a
    # rehearsal on a box that already holds that path can point elsewhere.
    [string]$Src = "C:\ppf-contact-solver",
    # THE DISPATCH TEST: print this rig's share of the scenarios (`runtests
    # --list`, after the shard) to C:\ci\shard_list.txt and exit, without
    # Blender, a build or any package: the source tree and an interpreter are
    # enough. Everything else about the launch (the detachment, the shard
    # argument, the exit file, the log) is the real path, which is the point.
    [switch]$ListOnly
)

$Py = "$Src\build-win-native\python\python.exe"
$CiDir = "C:\ci"

New-Item -ItemType Directory -Path $CiDir -Force | Out-Null
$Log = "$CiDir\rig.log"
Remove-Item "$CiDir\rig_exit.txt" -ErrorAction SilentlyContinue

function Log($msg) { $msg | Tee-Object -FilePath $Log -Append }

$exitCode = 1
try {
    Log "=== Blender rig (real backend, headless) starting $(Get-Date -Format o) ==="

    # Environment, mirroring the launcher build.bat writes (start.bat): for
    # EVERY GPU backend the tree carries, its solver directory, the library
    # the backend was built into, and the toolkit's bin, plus the embedded
    # Python. The rig's orchestrator and the addon each resolve which
    # directory's server to spawn through the .ppf-backend markers under
    # target\<backend>, not through PATH. PATH is for the DLLs the solvers
    # import, and BOTH backends need theirs, because resolving the backend
    # runs `ppf-contact-solver.exe --probe` in every GPU build present, and
    # a probe whose DLLs are missing does not answer "unusable", it fails to
    # start (exit 0xC0000135, STATUS_DLL_NOT_FOUND). Measured on run
    # 35088365107: target\rocm\release\ppf-contact-solver.exe --probe died
    # that way before slot 00, with only the CUDA directories on PATH.
    $env:PATH = "$Src\target\cuda\release;" +
                "$Src\crates\ppf-cts-compute\cuda\build\lib;" +
                "$Src\build-win-native\cuda\bin;" +
                "$Src\target\rocm\release;" +
                "$Src\crates\ppf-cts-compute\rocm\build\lib;" +
                "$Src\build-win-native\rocm\bin;" +
                "$Src\build-win-native\python;" +
                "$Src\build-win-native\python\Scripts;" +
                "$Src\build-win-native\mingit\cmd;" + $env:PATH
    $env:CUDA_PATH = "$Src\build-win-native\cuda"
    $env:PYTHONPATH = "$Src;" + $env:PYTHONPATH
    $env:PYTHONUNBUFFERED = "1"

    # Blender binary path (written by download-blender.ps1).
    $names = @($Scenarios -split '\s+' | Where-Object { $_ })
    # NOT `$shard`: PowerShell variables are case-insensitive, so that name IS
    # the `$Shard` parameter, and assigning `@()` to it emptied the parameter
    # before the next line read it. Every rig of the first sharded runs
    # (35180161718 to 35193244970) ran the whole list for that reason, and the
    # `Running rig:` line below, which carries no `--shard`, is where it shows.
    $shardArgs = @()
    if ($Shard.Trim()) { $shardArgs = @("--shard", $Shard.Trim()) }
    if ($ListOnly) {
        # BEFORE THE BLENDER REQUIREMENT: a listing rig has no Blender (the
        # dispatch test installs none), and the first dispatch test failed
        # on the check below, which the rehearsal box passed by having one.
        Log "Running rig (list only): main.py runtests $($names -join ' ') $($shardArgs -join ' ') --backend real --list"
        & $Py "$Src\blender_addon\debug\main.py" runtests @names @shardArgs --backend real --list 2>>$Log |
            Out-File -FilePath "$CiDir\shard_list.txt" -Encoding ascii
        $exitCode = $LASTEXITCODE
        $count = @(Get-Content "$CiDir\shard_list.txt" | Where-Object { $_.Trim() }).Count
        Log "list only: shard '$Shard' holds $count scenario(s); exit code: $exitCode"
        exit $exitCode
    }
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

    Log "Running rig: main.py runtests $($names -join ' ') $($shardArgs -join ' ') --backend real (headless, timeout=$Timeout parallel=$Parallel)"
    & $Py "$Src\blender_addon\debug\main.py" runtests @names @shardArgs `
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
