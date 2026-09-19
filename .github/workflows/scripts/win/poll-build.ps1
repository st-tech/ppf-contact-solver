# File: poll-build.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Reports the detached build's state in one line, the shape wait-rig.sh
# reads: `STATE|<exit code or RUNNING>|<newest phase or output line>`.
#
# THE PROGRESS FIELD IS THE NEWEST LINE THAT SAYS WHERE THE BUILD IS. The
# build's phases print `[n/4]` markers and its long stretches (the
# transcompiler render, nvcc, cargo) print little for minutes, which is why
# the last line of the log alone reads like a stall; the newest `[n/4]`
# marker is kept alongside it so a poll always says which phase it is in.

$CiDir = "C:\ci"
$exit = "RUNNING"
$progress = ""
try {
    if (Test-Path "$CiDir\build_exit.txt") {
        $exit = (Get-Content "$CiDir\build_exit.txt" -Raw).Trim()
        if (-not $exit) { $exit = "RUNNING" }
    }
} catch { $exit = "RUNNING" }
try {
    if (Test-Path "$CiDir\build.log") {
        $lines = Get-Content "$CiDir\build.log" -Tail 400 -ErrorAction SilentlyContinue
        $phase = ($lines | Where-Object { $_ -match '^\[[0-9]/[0-9]\]|^=== ' } | Select-Object -Last 1)
        $last = ($lines | Where-Object { $_.Trim() } | Select-Object -Last 1)
        $progress = (@($phase, $last) | Where-Object { $_ }) -join ' :: '
        if ($progress.Length -gt 200) { $progress = $progress.Substring(0, 200) }
    }
} catch { $progress = "" }
Write-Output ("STATE|" + $exit + "|" + $progress)
