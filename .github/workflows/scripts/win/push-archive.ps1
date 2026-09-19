# File: push-archive.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Runs on the BUILD instance: copies the release archive straight to the
# verification instance over the VPC, then deletes the key it used.
#
# WHY THE BUILD INSTANCE SENDS IT. Both instances sit in the same subnet of
# the CI VPC, whose security group admits TCP 22, so this copy runs at VPC
# speed. What it replaces is two serial hops through ec2-instance-connect
# tunnels: 17 minutes measured to bring the 400 MB archive down to the
# runner, and about as long to push it back up to the verifier. The runner
# still downloads the archive for the publish job, but `release.yml` does
# that WHILE the verification runs, so that copy is no longer on the path.
#
# THE KEY IS THE VERIFIER'S OWN, PUSHED HERE FOR THIS STEP ONLY. The
# verification instance authorizes the key pair it was launched with (its
# user-data reads the public half from instance metadata), so the private
# half is what lets this host in. Windows OpenSSH refuses a key file that
# other accounts can read, and a file scp'd from the runner inherits the
# directory's ACL, so the ACL is cut down to the invoking user first;
# measured on a Windows Server 2025 box, that icacls line is what makes
# scp.exe accept a pushed key. The key is removed in every exit path.

param(
    [Parameter(Mandatory = $true)][string]$Archive,
    [Parameter(Mandatory = $true)][string]$Key,
    [Parameter(Mandatory = $true)][string]$Target,
    [string]$Destination = "C:/bundle.zip"
)

$ErrorActionPreference = "Continue"
$rc = 1
try {
    if (-not (Test-Path $Archive)) { throw "the archive is not here: $Archive" }
    if (-not (Test-Path $Key)) { throw "the key is not here: $Key" }

    & icacls $Key /inheritance:r /grant:r "$($env:USERNAME):F" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "icacls on $Key failed with exit $LASTEXITCODE" }

    $size = (Get-Item $Archive).Length
    $t = Get-Date
    & scp.exe -i $Key -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL `
        -o BatchMode=yes -o ConnectTimeout=30 $Archive "Administrator@${Target}:$Destination"
    $rc = $LASTEXITCODE
    $secs = [int]((Get-Date) - $t).TotalSeconds
    Write-Output ("scp exit {0}: {1} bytes to {2} in {3}s" -f $rc, $size, $Target, $secs)
} catch {
    Write-Output "ERROR: $_"
    $rc = 1
} finally {
    Remove-Item $Key -Force -ErrorAction SilentlyContinue
}
exit $rc
