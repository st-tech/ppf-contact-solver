# File: launch-build.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Starts run-build.ps1 DETACHED on the Blender CI build instance and returns
# at once. The detachment is the one launch-rig.ps1 measured (Win32_Process
# through the WMI provider service, which is no descendant of sshd and so
# survives the ssh that started it); its header holds the measurement, and
# the two are kept in step by hand.

param(
    [string]$Backends = "cuda rocm cpu"
)

$CiDir = "C:\ci"
New-Item -ItemType Directory -Path $CiDir -Force | Out-Null
Remove-Item "$CiDir\build_exit.txt" -ErrorAction SilentlyContinue
Remove-Item "$CiDir\build.log" -ErrorAction SilentlyContinue

$Command = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" +
           " -ExecutionPolicy Bypass -File C:/run_build.ps1 -Backends `"$Backends`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
    -Arguments @{ CommandLine = $Command }
if ($null -eq $result -or $result.ReturnValue -ne 0) {
    $rv = if ($result) { $result.ReturnValue } else { "no result" }
    throw "Win32_Process.Create did not start the build (ReturnValue=$rv)"
}
Write-Output "build launched detached, pid=$($result.ProcessId), backends: $Backends"
