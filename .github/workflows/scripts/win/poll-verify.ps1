# File: poll-verify.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Reports the detached verification's state in ONE line, so the waiting step
# in `release.yml` needs a single short ssh per poll and no nested quoting:
#
#   STATE|<exit code>|<the newest progress line>
#
# with the exit field reading `RUNNING` while `verify_exit.txt` is absent.
# `run-verify.ps1` writes that file in a `finally`, so its appearance is the
# finish signal whether the verification passed, failed or threw.
#
# THE PROGRESS FIELD IS WHAT PROVES THE RUN ADVANCED. A detached run writes
# nothing to the CI log, and a live process is not progress: a verification
# wedged on one example and one working through them look identical from the
# waiting side. Echoing `fast-check-all.bat`'s newest `[TEST n/N] <example>`
# line, or the section marker before the examples start, per poll makes a
# stall visible as a REPEATED line rather than as a wait that runs out.

$CiDir = "C:\ci"
$exit = "RUNNING"
$progress = ""

try {
    if (Test-Path "$CiDir\verify_exit.txt") {
        $exit = (Get-Content "$CiDir\verify_exit.txt" -Raw).Trim()
        if (-not $exit) { $exit = "RUNNING" }
    }
} catch {
    # A read racing the `finally` that writes the file reports RUNNING and
    # is resolved by the next poll.
    $exit = "RUNNING"
}

try {
    if (Test-Path "$CiDir\verify.log") {
        # The log is open for append by the batch under test, so this read
        # can lose a sharing race. Progress is advisory, so a miss costs one
        # line.
        $m = Select-String -Path "$CiDir\verify.log" `
            -Pattern '\[TEST \d+/\d+\]|=== [a-z-]+\.bat|PASSED|FAILED|ERROR' |
            Select-Object -Last 1
        if ($m) { $progress = $m.Line.Trim() }
    }
} catch {
    $progress = ""
}

Write-Output ("STATE|" + $exit + "|" + $progress)
