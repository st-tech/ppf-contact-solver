# File: launch-verify.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Starts `run-verify.ps1` DETACHED on the verification instance and returns
# at once, so the verification's lifetime is not the ssh session's.
#
# WHY DETACHED. The runner reaches the instance through an
# `aws ec2-instance-connect open-tunnel` websocket with a one hour session
# cap, and the credential that opens it lasts an hour too. The verification
# itself is well inside that, but detaching it is what lets the runner do
# something else while it runs: `release.yml` downloads the 400 MB release
# archive from the build instance, a transfer measured at 17 minutes through
# the same kind of tunnel, WHILE the examples run here, instead of after
# them. The two were serial before, and that is the time this buys.
#
# THE DETACHMENT MECHANISM IS THE ONE `launch-rig.ps1` MEASURED, and its
# header records the measurement: Windows OpenSSH reaps the session's
# descendants when the client disconnects, so a `Start-Process` child dies
# with the ssh that started it, while a process created through the WMI
# provider service is no descendant of sshd and survives, still running as
# the invoking user. The two launchers are kept in step by hand; if that
# measurement is ever revised there, revise it here.

$CiDir = "C:\ci"
New-Item -ItemType Directory -Path $CiDir -Force | Out-Null

# Both files are this run's only evidence and a leftover would be read as
# its result, so they go before anything starts.
Remove-Item "$CiDir\verify_exit.txt" -ErrorAction SilentlyContinue
Remove-Item "$CiDir\verify.log" -ErrorAction SilentlyContinue

# An absolute interpreter path: a WMI-created process does not inherit this
# session's PATH.
$Command = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" +
           " -ExecutionPolicy Bypass -File C:/run_verify.ps1"

$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
    -Arguments @{ CommandLine = $Command }

# Fail here rather than leaving the waiting side to time out on a
# verification that was never started.
if ($null -eq $result -or $result.ReturnValue -ne 0) {
    $rv = if ($result) { $result.ReturnValue } else { "no result" }
    throw "Win32_Process.Create did not start the verification (ReturnValue=$rv)"
}

Write-Output "verification launched detached, pid=$($result.ProcessId)"
Write-Output "command: $Command"
