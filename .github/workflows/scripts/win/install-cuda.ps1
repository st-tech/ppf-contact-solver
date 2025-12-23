# File: install-cuda.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

$ErrorActionPreference = "Continue"

Write-Host "=== Installing CUDA 12.8 ==="

$cudaUrl = "https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe"
$cudaInstaller = "C:\cuda_installer.exe"

Write-Host "Downloading CUDA installer (about 3GB, this will take a while)..."
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$webClient = New-Object System.Net.WebClient
$webClient.DownloadFile($cudaUrl, $cudaInstaller)

Write-Host "Download complete. File size:"
(Get-Item $cudaInstaller).Length

Write-Host "Installing CUDA silently (this takes 5-10 minutes)..."
Start-Process -FilePath $cudaInstaller -ArgumentList "-s" -Wait -NoNewWindow

Write-Host "Verifying CUDA installation..."
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if (Test-Path "$cudaPath\bin\nvcc.exe") {
    Write-Host "CUDA installed successfully!"
    & "$cudaPath\bin\nvcc.exe" --version
    "CUDA_READY" | Out-File -FilePath "C:\cuda_ready.txt"
} else {
    Write-Host "ERROR: nvcc.exe not found at $cudaPath\bin\nvcc.exe"
    exit 1
}

Write-Host ""
Write-Host "=== CUDA Installation Complete ==="
