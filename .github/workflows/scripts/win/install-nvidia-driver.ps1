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
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$webClient = New-Object System.Net.WebClient
$webClient.DownloadFile($driverUrl, $driverInstaller)

# Sanity-check the download size before invoking the installer.
# WebClient.DownloadFile silently returns whatever bytes arrived if the
# CDN drops the connection mid-transfer, so a partial 2 MiB stub will
# happily run through the silent installer and only get caught minutes
# later by the nvidia-smi verify step (after EC2 minutes have already
# been billed). The real installer is ~700 MB; reject anything under
# 200 MiB and fail fast so the dispatcher can retry.
$downloadedSize = (Get-Item $driverInstaller).Length
Write-Host "Download complete. File size: $downloadedSize bytes"
$minSize = 200 * 1024 * 1024  # 200 MiB
if ($downloadedSize -lt $minSize) {
    Write-Host "ERROR: downloaded file is $downloadedSize bytes, below the $minSize byte minimum."
    Write-Host "Likely a truncated download from us.download.nvidia.com (CDN flake) or a stale URL."
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
