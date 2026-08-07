# File: install-nvidia-driver.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

$ErrorActionPreference = "Continue"

Write-Host "=== Installing NVIDIA Driver Only (No CUDA Toolkit) ==="

# For AWS G6e instances with L4 GPUs, use the Windows Server data center
# driver. NVIDIA splits client (win10/win11) and Server builds; the EC2
# AMI is Server, so use the winserver-2022-2025 build.
$driverUrl = "https://us.download.nvidia.com/tesla/580.88/580.88-data-center-tesla-desktop-winserver-2022-2025-dch-international.exe"
$driverInstaller = "C:\nvidia_driver.exe"

Write-Host "Downloading NVIDIA driver installer..."

# curl rather than WebClient. It retries a dropped transfer itself, which
# a 700 MB download from a single CDN needs, and its exit code names the
# cause: a dead pointer (22, an HTTP 4xx) reads differently from a
# transport failure (28 timeout, 18 partial, 56 recv error). A WebClient
# exception surfaces only "An exception occurred during a WebClient
# request", which distinguishes none of them. curl.exe is in System32 on
# Windows Server and negotiates TLS through Schannel, so no protocol needs
# arranging here.
#
# Do NOT add --retry-all-errors: plain --retry covers the transient class
# and leaves 4xx alone, so a retired driver URL fails on its first attempt
# rather than being retried into a slow abort.
#
# Removing any earlier file first matters beyond tidiness. This script runs
# with ErrorActionPreference=Continue, so a failed download does not stop
# it; a stale installer left by a previous attempt would then satisfy the
# size floor below and be installed as though it had just been fetched.
if (Test-Path $driverInstaller) { Remove-Item -Force $driverInstaller }

$curlOut = & curl.exe --silent --show-error --fail --location `
    --retry 5 --retry-delay 5 --retry-connrefused `
    --connect-timeout 20 `
    --output $driverInstaller $driverUrl 2>&1
$curlRc = $LASTEXITCODE
if ($curlRc -ne 0) {
    Write-Host "ERROR: curl exit code $curlRc downloading $driverUrl"
    if ($curlOut) { Write-Host "curl: $curlOut" }
    exit 1
}

# Sanity-check the size before invoking the installer. curl reports a
# truncated transfer as exit 18 and an HTTP error as 22, so what is left
# for this check is an artifact that arrived intact and small: a CDN error
# page served with 200, or a stub. Either would run through the silent
# installer and only surface minutes later at the nvidia-smi verify step,
# after EC2 minutes have been billed. The real installer is ~700 MB.
$downloadedSize = (Get-Item $driverInstaller).Length
Write-Host "Download complete. File size: $downloadedSize bytes"
$minSize = 200 * 1024 * 1024  # 200 MiB
if ($downloadedSize -lt $minSize) {
    Write-Host "ERROR: downloaded file is $downloadedSize bytes, below the $minSize byte minimum."
    Write-Host "curl reported success, so this is not a truncation (18) or an HTTP error (22):"
    Write-Host "us.download.nvidia.com served a small artifact with 200, such as an error page."
    exit 1
}

Write-Host "Installing NVIDIA driver silently (this takes a few minutes)..."
# -s for silent, -noreboot to prevent automatic reboot
Start-Process -FilePath $driverInstaller -ArgumentList "-s", "-noreboot" -Wait -NoNewWindow

Write-Host "Verifying NVIDIA driver installation..."
$nvidiaSmiPath = "C:\Windows\System32\nvidia-smi.exe"
if (Test-Path $nvidiaSmiPath) {
    Write-Host "NVIDIA driver installed successfully!"
    & $nvidiaSmiPath
    "DRIVER_READY" | Out-File -FilePath "C:\driver_ready.txt"
} else {
    Write-Host "ERROR: nvidia-smi.exe not found"
    Write-Host "Checking for driver files..."
    Get-ChildItem "C:\Windows\System32\nv*.dll" -ErrorAction SilentlyContinue | Select-Object Name
    exit 1
}

Write-Host ""
Write-Host "=== NVIDIA Driver Installation Complete ==="
