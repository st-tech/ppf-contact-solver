# File: launch-rig.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Starts `run-blender-rig.ps1` DETACHED and returns at once, so the rig's
# lifetime is not the ssh session's.
#
# WHY THE RIG MAY NOT RUN SYNCHRONOUSLY OVER ONE SSH. The CI runner reaches
# this instance through `aws ec2-instance-connect open-tunnel`, whose websocket
# has a ONE HOUR session cap, and the Windows rig's real subset needs slightly
# more than that: measured, 204 of 211 scenarios had reported, all passing,
# when the websocket closed with "Connection exceeded session timeout" and ssh
# returned 255 at exactly 60 minutes. Being a couple of minutes over the cap is
# the worst place to sit, because the run looks healthy right up to a transport
# failure that reads as whichever scenario was in flight. Keepalives cannot
# help: the cap is on the session, not on idleness.
#
# THE DETACHMENT MECHANISM IS MEASURED, NOT ASSUMED, AND THE OBVIOUS ONE DOES
# NOT WORK. Windows OpenSSH reaps the session's descendants when the client
# disconnects, so a `Start-Process` child dies with the ssh that launched it.
# All three were measured on a Windows Server 2025 box by starting a child that
# sleeps 90 s and writes a marker, disconnecting at once, and looking for the
# marker 130 s later with no ssh held:
#
#   Start-Process -WindowStyle Hidden        KILLED
#   Invoke-CimMethod Win32_Process Create    SURVIVED
#   schtasks /ru SYSTEM                      SURVIVED
#
# The WMI route is taken because the process is created by the WMI provider
# service, so it is no descendant of sshd, while still running as the invoking
# user: a scheduled task survives equally but would run as SYSTEM, against a
# different profile from every other step in the job, and would leave a task to
# clean up. The rig's own children (Blender, `ppf-cts-server.exe`) descend from
# the detached process and survive with it.
#
# This is the ONE place the mechanism lives, so the waiting side in
# `blender.yml` is unaffected by how the process is started.

param(
    [double]$Timeout = 420.0,
    [int]$Parallel = 1,
    # Space-separated scenario names; empty runs the whole set.
    [string]$Scenarios = "",
    # I/N: this host's share of the selection (the orchestrator's --shard);
    # empty runs all of it.
    [string]$Shard = "",
    # The tree the rig runs from; passed through to run-blender-rig.ps1.
    [string]$Src = "",
    # Launch phony-rig.ps1 instead of the rig: blender.yml's pipeline_test.
    [switch]$Phony,
    # With -Phony: make the stand-in report a failure.
    [switch]$PhonyFail,
    # The real runner, but it only lists its share and exits (the dispatch
    # test): no Blender, no build, no packages needed on the rig.
    [switch]$ListOnly
)

$CiDir = "C:\ci"
New-Item -ItemType Directory -Path $CiDir -Force | Out-Null

# Both files are this run's only evidence and a leftover would be read as its
# result, so they go before anything starts. `run-blender-rig.ps1` clears
# `rig_exit.txt` itself; the log it only appends to.
Remove-Item "$CiDir\rig_exit.txt" -ErrorAction SilentlyContinue
Remove-Item "$CiDir\rig.log" -ErrorAction SilentlyContinue

# An absolute interpreter path: a WMI-created process does not inherit this
# session's PATH.
if ($Phony) {
    # The stand-in takes the same detachment and leaves the same files, so
    # everything downstream of this launch runs unchanged against it.
    $Command = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" +
               " -ExecutionPolicy Bypass -File C:/phony_rig.ps1 -Seconds 60"
    if ($PhonyFail) { $Command += " -Fail" }
} else {
    $Command = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" +
               " -ExecutionPolicy Bypass -File C:/run_blender_rig.ps1" +
               " -Timeout $Timeout -Parallel $Parallel"
}
if (-not $Phony -and $Scenarios.Trim()) {
    # Names are [a-z0-9_] and spaces, so double quotes are enough to keep
    # them one argument through Win32_Process.Create's command line.
    $Command += " -Scenarios `"$($Scenarios.Trim())`""
}
if (-not $Phony -and $Shard.Trim()) {
    $Command += " -Shard $($Shard.Trim())"
}
if (-not $Phony -and $Src.Trim()) {
    $Command += " -Src `"$($Src.Trim())`""
}
if (-not $Phony -and $ListOnly) {
    $Command += " -ListOnly"
}

$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
    -Arguments @{ CommandLine = $Command }

# Fail here rather than leaving the waiting side to time out on a rig that was
# never started.
if ($null -eq $result -or $result.ReturnValue -ne 0) {
    $rv = if ($result) { $result.ReturnValue } else { "no result" }
    throw "Win32_Process.Create did not start the rig (ReturnValue=$rv)"
}

Write-Output "$(if ($Phony) { "PHONY rig" } else { "rig" }) launched detached, pid=$($result.ProcessId)"
Write-Output "command: $Command"
