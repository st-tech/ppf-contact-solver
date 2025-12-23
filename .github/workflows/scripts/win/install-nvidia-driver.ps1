# File: install-nvidia-driver.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

$ErrorActionPreference = "Continue"

Write-Host "=== Installing NVIDIA Driver Only (No CUDA Toolkit) ==="

# For AWS G6e instances with L4 GPUs, use the data center driver
# Using NVIDIA Data Center Driver for Linux/Windows (L4 GPU)
$driverUrl = "https://us.download.nvidia.com/tesla/572.83/572.83-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe"
$driverInstaller = "C:\nvidia_driver.exe"

Write-Host "Downloading NVIDIA driver installer..."
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$webClient = New-Object System.Net.WebClient
$webClient.DownloadFile($driverUrl, $driverInstaller)

Write-Host "Download complete. File size:"
(Get-Item $driverInstaller).Length

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
