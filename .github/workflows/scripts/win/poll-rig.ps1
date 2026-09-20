# File: poll-rig.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Reports the detached rig's state to the waiting step in `blender.yml`, in
# one short ssh with no nested quoting:
#
#     poll-rig.ps1 -Since <orchestrator lines the waiter has already printed>
#
#   STATE|<exit code>|<the rig's newest orchestrator line>
#   LINE|<an orchestrator line the waiter has not printed yet>
#   LINE|...
#
# with the exit field reading `RUNNING` while `rig_exit.txt` is absent.
# `run-blender-rig.ps1` writes that file in a `finally`, so its appearance is
# the finish signal whether the rig passed, failed or threw.
#
# THE `LINE|` RECORDS ARE WHAT MAKE A PASS OR A FAILURE VISIBLE WHILE THE RUN
# IS STILL GOING. A detached rig writes nothing to the CI log, so the waiting
# step is the only place its orchestrator output appears before the run ends.
# Echoing one newest line per poll would drop every
# `slot N <- <scenario> <status> (<seconds>)` that falls between two polls, so
# the CI log would name the scenario each rig had STARTED and never how any of
# them ended. A Linux rig runs inside its own step and shows both arrows, and
# these records are what give the Windows leg the same reading. `-Since`
# counts the matching lines the waiter has already printed, so a poll returns
# what is new, in order, and one that never arrives is re-sent by the next
# poll rather than lost. At most $MaxLines leave per poll, which bounds one
# ssh answer; a rig further ahead than that catches up over the polls that
# follow, and `C:\ci\rig.log` is collected whole as an artifact either way.
#
# THE PROGRESS FIELD IS WHAT PROVES THE RUN ADVANCED when a poll carries no
# new line. A live process is not progress: a rig wedged on one scenario and a
# rig working through them look identical from the waiting side, so the waiter
# repeats this field and a stall reads as a REPEATED line rather than as a
# wait that simply runs out.
#
# THE SLOT NUMBER IS AN IDENTIFIER, NOT A MONOTONIC COUNTER, so do not read a
# backwards jump as a restart. The orchestrator defers every
# `NOT_PARALLELIZABLE` scenario to the end of the run while it keeps the slot
# number it was assigned in the requested order, so the tail of a healthy run
# descends: measured on a 211 scenario Windows sweep, the last nine were slots
# 3, 5, 79, 80, 92, 132, 206, 207, 208, and all nine declare that flag. What
# signals a stall is the line not CHANGING, which holds whatever the number
# does.

param(
    [int]$Since = 0
)

$CiDir = "C:\ci"
# Both arrows, and the one line that explains a run which stops on its own.
$Pattern = '^\s*\[orchestrator\] (slot \d+ (->|<-)|ABORTING RUN)'
$MaxLines = 400
$exit = "RUNNING"
$progress = ""
$lines = @()

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
        # lose a sharing race. A miss costs nothing lasting: the waiter
        # advances its cursor by the records it receives, so the next poll
        # sends the same lines again.
        $m = @(Select-String -Path "$CiDir\rig.log" -Pattern $Pattern)
        if ($m.Count -gt 0) {
            $progress = $m[-1].Line.Trim()
            $lines = @($m | Select-Object -Skip $Since -First $MaxLines |
                ForEach-Object { $_.Line.Trim() })
        }
    }
} catch {
    $progress = ""
    $lines = @()
}

Write-Output ("STATE|" + $exit + "|" + $progress)
foreach ($line in $lines) { Write-Output ("LINE|" + $line) }
