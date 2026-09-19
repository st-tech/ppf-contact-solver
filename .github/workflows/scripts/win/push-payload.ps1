# File: push-payload.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Runs on the Blender CI BUILD instance: copies one file to every rig
# instance over the VPC, then deletes the key it used. push-archive.ps1 is
# the one-target version of this, for the release workflow; its header
# records why the key's ACL is cut down first (Windows OpenSSH refuses a key
# file other accounts can read) and why the build instance sends.
#
# Every target is attempted; the exit code is the number that failed, so
# one unreachable rig does not hide the others in the log. The key is
# removed in every exit path.

param(
    [Parameter(Mandatory = $true)][string]$Archive,
    [Parameter(Mandatory = $true)][string]$Key,
    # Space-separated private addresses.
    [Parameter(Mandatory = $true)][string]$Targets,
    [string]$Destination = "C:/rig-payload.tar"
)

$ErrorActionPreference = "Continue"
$failed = 0
try {
    if (-not (Test-Path $Archive)) { throw "the archive is not here: $Archive" }
    if (-not (Test-Path $Key)) { throw "the key is not here: $Key" }
    & icacls $Key /inheritance:r /grant:r "$($env:USERNAME):F" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "icacls on $Key failed with exit $LASTEXITCODE" }
    $size = (Get-Item $Archive).Length
    foreach ($target in @($Targets -split '\s+' | Where-Object { $_ })) {
        $t = Get-Date
        & scp.exe -i $Key -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL `
            -o BatchMode=yes -o ConnectTimeout=30 $Archive "Administrator@${target}:$Destination"
        $rc = $LASTEXITCODE
        $secs = [int]((Get-Date) - $t).TotalSeconds
        Write-Output ("scp exit {0}: {1:N0} MB to {2} in {3}s" -f $rc, ($size / 1MB), $target, $secs)
        if ($rc -ne 0) { $failed++ }
    }
} catch {
    Write-Output "ERROR: $_"
    $failed = 1
} finally {
    Remove-Item $Key -Force -ErrorAction SilentlyContinue
}
exit $failed
