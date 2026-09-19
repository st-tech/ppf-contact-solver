# File: poll-rig.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Reports the detached rig's state in ONE line, so the waiting step in
# `blender.yml` needs a single short ssh per poll and no nested quoting:
#
#   STATE|<exit code>|<the rig's most recent progress line>
#
# with the exit field reading `RUNNING` while `rig_exit.txt` is absent.
# `run-blender-rig.ps1` writes that file in a `finally`, so its appearance is
# the finish signal whether the rig passed, failed or threw.
#
# THE PROGRESS FIELD IS WHAT PROVES THE RUN ADVANCED. A detached rig writes
# nothing to the CI log, and a live process is not progress: a rig wedged on
# one scenario and a rig working through them look identical from the waiting
# side. Echoing the orchestrator's newest `slot N -> <scenario>` line per poll
# makes a stall visible as a REPEATED line rather than as a wait that simply
# runs out.
#
# THE SLOT NUMBER IS AN IDENTIFIER, NOT A MONOTONIC COUNTER, so do not read a
# backwards jump as a restart. The orchestrator defers every
# `NOT_PARALLELIZABLE` scenario to the end of the run while it keeps the slot
# number it was assigned in the requested order, so the tail of a healthy run
# descends: measured on a 211 scenario Windows sweep, the last nine were slots
# 3, 5, 79, 80, 92, 132, 206, 207, 208, and all nine declare that flag. What
# signals a stall is the line not CHANGING, which holds whatever the number
# does.

$CiDir = "C:\ci"
$exit = "RUNNING"
$progress = ""

try {
    if (Test-Path "$CiDir\rig_exit.txt") {
        $exit = (Get-Content "$CiDir\rig_exit.txt" -Raw).Trim()
        if (-not $exit) { $exit = "RUNNING" }
    }
} catch {
    # A read racing the `finally` that writes the file reports RUNNING and is
    # resolved by the next poll.
    $exit = "RUNNING"
}

try {
    if (Test-Path "$CiDir\rig.log") {
        # The log is open for append by the rig's Tee-Object, so this read can
        # lose a sharing race. Progress is advisory, so a miss costs one line.
        $m = Select-String -Path "$CiDir\rig.log" -Pattern 'slot \d+ ->' |
            Select-Object -Last 1
        if ($m) { $progress = $m.Line.Trim() }
    }
} catch {
    $progress = ""
}

Write-Output ("STATE|" + $exit + "|" + $progress)
