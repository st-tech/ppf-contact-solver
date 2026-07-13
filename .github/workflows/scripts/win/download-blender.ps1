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
#   powershell -ExecutionPolicy Bypass -File C:/download_blender.ps1 -Version 5.1.1

param(
    [string]$Version = "5.1.1"
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# Minor series dir on the mirror, e.g. 5.1.1 -> Blender5.1
$minor = ($Version -split '\.')[0..1] -join '.'
$url = "https://download.blender.org/release/Blender$minor/blender-$Version-windows-x64.zip"
$zip = "C:\blender.zip"
$dest = "C:\blender"

Write-Host "Downloading Blender $Version from $url ..."
$wc = New-Object System.Net.WebClient
$wc.DownloadFile($url, $zip)

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
