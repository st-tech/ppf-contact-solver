# File: download-blender.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Download + extract the official Blender Windows portable build so the
# debug rig can launch it. Mirrors the "Download Blender" steps in
# blender.yml (Linux/macOS) for the Windows-native GPU job.
#
# Extracts to C:\blender\blender-<ver>-windows-x64\blender.exe and writes
# that path to C:\blender_bin.txt so later steps (install-blender-addon,
# run-blender-rig) can read it without re-deriving the folder name.
#
# Usage (over SSH):
#   powershell -ExecutionPolicy Bypass -File C:/download_blender.ps1 -Version 5.2.0

param(
    [string]$Version = "5.2.0"
)

$ErrorActionPreference = "Stop"

# Minor series dir on the mirror, e.g. 5.2.0 -> Blender5.2
$minor = ($Version -split '\.')[0..1] -join '.'
$url = "https://download.blender.org/release/Blender$minor/blender-$Version-windows-x64.zip"
$zip = "C:\blender.zip"
$dest = "C:\blender"

Write-Host "Downloading Blender $Version from $url ..."

# curl rather than WebClient, for two reasons that both cost a CI run when
# they are missing. It retries a dropped transfer itself, which a 400 MB
# one-shot download from a single origin needs; and its exit code names the
# cause, distinguishing a dead pointer (22, an HTTP 4xx) from a transport
# failure (28 timeout, 56 recv error). A WebClient exception surfaces only
# "An exception occurred during a WebClient request", which says nothing
# about which of those happened. curl.exe is in System32 on Windows Server
# and is what warmup.bat already downloads with. TLS needs no arranging
# here: curl negotiates it through Schannel.
#
# Do NOT add --retry-all-errors. Plain --retry already covers the transient
# class (timeouts, 5xx, dropped connections) and deliberately leaves 4xx
# alone, so a stale version pointer still fails on its first attempt instead
# of being retried into a slow abort; with --retry-all-errors a 404 is
# reported six times over half a minute. This matches the retry policy
# scripts\check-downloads.bat settled on for the same reason.
if (Test-Path $zip) { Remove-Item -Force $zip }

# PowerShell converts a native command's stderr into error records, and with
# ErrorActionPreference=Stop that aborts on curl's own output before
# $LASTEXITCODE can be read. Let the exit code be what decides.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$curlOut = & curl.exe --silent --show-error --fail --location `
    --retry 5 --retry-delay 5 --retry-connrefused `
    --connect-timeout 20 `
    --output $zip $url 2>&1
$curlRc = $LASTEXITCODE
$ErrorActionPreference = $prevEAP

if ($curlRc -ne 0) {
    Write-Host "curl exit code: $curlRc"
    if ($curlOut) { Write-Host "curl: $curlOut" }
    Write-Error "Failed to download Blender from $url (curl exit $curlRc)"
    exit 1
}

# The Windows Blender zip is ~350 MB; a truncated CDN response would
# unzip to a broken tree and only fail minutes later at launch. Reject
# anything implausibly small so the caller can retry.
$size = (Get-Item $zip).Length
Write-Host "Downloaded $size bytes"
if ($size -lt 100 * 1024 * 1024) {
    Write-Error "Blender download is $size bytes, below the 100 MiB floor (truncated download or stale URL)."
    exit 1
}

if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
New-Item -ItemType Directory -Path $dest -Force | Out-Null
Write-Host "Extracting to $dest ..."
Expand-Archive -Path $zip -DestinationPath $dest -Force
Remove-Item $zip

$exe = Get-ChildItem -Path $dest -Recurse -Filter "blender.exe" |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $exe) {
    Write-Error "blender.exe not found under $dest after extraction"
    exit 1
}

Write-Host "Blender binary: $exe"
& $exe --version | Select-Object -First 3
Set-Content -Path "C:\blender_bin.txt" -Value $exe -Encoding ASCII -NoNewline
Write-Host "Wrote path to C:\blender_bin.txt"
