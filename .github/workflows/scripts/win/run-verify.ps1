# File: run-verify.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The clean-environment verification of a Windows x64 distribution, run
# DETACHED on the verification instance by `launch-verify.ps1`: unpack the
# archive as a user would, run its `headless.bat` smoke and then
# `fast-check-all.bat`, and leave the verdict where `poll-verify.ps1` reads it.
#
# WHAT THIS PROVES. The instance carries the NVIDIA driver and nothing else:
# no CUDA toolkit, no compiler, no Python. Every example that passes here
# passed against the distribution's own binaries and its own interpreter,
# which is the only evidence that the archive has no dependency on the
# machine that built it. It is the one test the build instance cannot
# perform on itself, which is why the same checks are not also run there.
#
# THE VERDICT IS TWO FILES, WRITTEN IN A `finally`. `C:\ci\verify_exit.txt`
# holds the exit code and appears only when this script is done, whether it
# passed, failed or threw, so its presence is the finish signal the poll
# waits for. `C:\ci\verify.log` holds every line the batch files printed;
# the batches' output is redirected there by cmd itself, because a process
# created through WMI has no console for a transcript to capture.
#
# THE BATCH FILES ARE RUN THROUGH `call`. `cmd /c` without it loses an
# `exit /b 1` fired inside a parenthesized block, which is exactly how
# `fast-check-all.bat` reports a failed example.

param(
    [string]$Zip = "C:\bundle.zip",
    [string]$Dir = "C:\bundle"
)

$ErrorActionPreference = "Continue"
$CiDir = "C:\ci"
New-Item -ItemType Directory -Path $CiDir -Force | Out-Null
$ExitFile = "$CiDir\verify_exit.txt"
$Log = "$CiDir\verify.log"
Remove-Item $ExitFile -ErrorAction SilentlyContinue

function Log([string]$line) {
    Add-Content -Path $Log -Value ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $line)
}

$code = 1
try {
    Log "verification starting: $Zip -> $Dir"
    if (-not (Test-Path $Zip)) { throw "the archive $Zip is not here" }
    if (Test-Path $Dir) { Remove-Item -Recurse -Force $Dir }
    New-Item -ItemType Directory -Path $Dir -Force | Out-Null
    Expand-Archive -Path $Zip -DestinationPath $Dir -Force
    Log "unpacked"

    # PPF_DIAG_SELFTEST makes every backend open prove, on this GPU, that a
    # failed device check reaches the host: the backend fires one deliberately
    # failing check from a real kernel and refuses to open unless the record
    # comes back. It is off by default, because a run should not pay a kernel
    # launch for the proof and the two batch files here also ship to users; it
    # is set for this verification, which is the pass that asks whether the
    # archive reports what it is supposed to report. A channel that is
    # unattached or drained privately leaves every gate green while the
    # guarantee-class checks report nothing.
    $env:PPF_DIAG_SELFTEST = "1"

    foreach ($bat in @("headless.bat", "fast-check-all.bat")) {
        if (-not (Test-Path "$Dir\$bat")) { throw "$bat is not in the archive" }
        Log "=== $bat ==="
        # `>>` inside the cmd string, so the batch's own stdout and stderr
        # land in the log as they are produced and the poll can read the
        # newest [TEST n/N] line while the run is in flight.
        & $env:ComSpec /c "cd /d `"$Dir`" && call $bat /nopause >> `"$Log`" 2>&1"
        $rc = $LASTEXITCODE
        Log "=== $bat exit $rc ==="
        if ($rc -ne 0) {
            $code = $rc
            throw "$bat failed with exit $rc"
        }
    }
    $code = 0
    Log "verification PASSED"
} catch {
    Log "ERROR: $_"
    if ($code -eq 0) { $code = 1 }
} finally {
    Set-Content -Path $ExitFile -Value $code
}
