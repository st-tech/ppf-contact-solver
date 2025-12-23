<powershell>
# File: user-data.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

$ErrorActionPreference = "Continue"
Start-Transcript -Path "C:\build-setup.log" -Append

Write-Host "=== Starting SSH Setup ==="

$SSHPort = SSH_PORT_PLACEHOLDER
Write-Host "Configuring SSH on port $SSHPort"

# Disable Windows Firewall - AWS security groups provide network-level protection
Write-Host "Disabling Windows Firewall (relying on AWS security groups)..."
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False

# Enable Long Path support
Write-Host "Enabling Long Path support..."
Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1

# Check if OpenSSH Server is installed (pre-installed on Windows Server 2025)
Write-Host "Checking OpenSSH Server status..."
$sshCapability = Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
Write-Host "OpenSSH Server state: $($sshCapability.State)"

if ($sshCapability.State -ne "Installed") {
    Write-Host "Installing OpenSSH Server..."
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
}

# Ensure C:\ProgramData\ssh directory exists
$sshDir = "C:\ProgramData\ssh"
if (-not (Test-Path $sshDir)) {
    Write-Host "Creating SSH directory..."
    New-Item -ItemType Directory -Path $sshDir -Force
}

# Get public key from EC2 instance metadata (IMDSv2)
Write-Host "Getting public key from EC2 metadata..."
try {
    $ImdsToken = (Invoke-WebRequest -Uri "http://169.254.169.254/latest/api/token" -Method "PUT" -Headers @{"X-aws-ec2-metadata-token-ttl-seconds" = "21600"} -UseBasicParsing -TimeoutSec 10).Content
    $ImdsHeaders = @{"X-aws-ec2-metadata-token" = $ImdsToken}
    $PublicKey = (Invoke-WebRequest -Uri "http://169.254.169.254/latest/meta-data/public-keys/0/openssh-key" -Headers $ImdsHeaders -UseBasicParsing -TimeoutSec 10).Content
    Write-Host "Retrieved public key: $PublicKey"
} catch {
    Write-Host "ERROR getting public key: $_"
}

# Create administrators_authorized_keys file
$AuthorizedKeysPath = "C:\ProgramData\ssh\administrators_authorized_keys"
Write-Host "Writing authorized_keys to $AuthorizedKeysPath"
Set-Content -Path $AuthorizedKeysPath -Value $PublicKey -Force

# Set proper permissions (critical for admin users)
Write-Host "Setting permissions on authorized_keys..."
icacls $AuthorizedKeysPath /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"

# Set default shell to PowerShell
Write-Host "Setting default shell to PowerShell..."
if (-not (Test-Path "HKLM:\SOFTWARE\OpenSSH")) {
    New-Item -Path "HKLM:\SOFTWARE\OpenSSH" -Force
}
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -PropertyType String -Force

# Configure SSH to use custom port
Write-Host "Configuring SSH to use port $SSHPort..."
$sshdConfigPath = "C:\ProgramData\ssh\sshd_config"

# First, start SSH service to generate default config files
Write-Host "Starting SSH service to generate default config..."
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd -ErrorAction SilentlyContinue
Start-Sleep -Seconds 5

# Now wait for sshd_config to be created
$maxWait = 60
$waited = 0
while (-not (Test-Path $sshdConfigPath) -and $waited -lt $maxWait) {
    Write-Host "Waiting for sshd_config to be created..."
    Start-Sleep -Seconds 2
    $waited += 2
}

if (Test-Path $sshdConfigPath) {
    $sshdConfig = Get-Content $sshdConfigPath
    # Replace or add Port directive
    $portFound = $false
    $newConfig = @()
    foreach ($line in $sshdConfig) {
        if ($line -match "^#?Port ") {
            $newConfig += "Port $SSHPort"
            $portFound = $true
        } else {
            $newConfig += $line
        }
    }
    if (-not $portFound) {
        $newConfig = @("Port $SSHPort") + $newConfig
    }
    Set-Content -Path $sshdConfigPath -Value $newConfig
    Write-Host "sshd_config updated with Port $SSHPort"
} else {
    Write-Host "WARNING: sshd_config not found, creating complete config..."
    $defaultConfig = @"
Port $SSHPort
PubkeyAuthentication yes
PasswordAuthentication no
AuthorizedKeysFile __PROGRAMDATA__/ssh/administrators_authorized_keys
Subsystem sftp sftp-server.exe
Match Group administrators
    AuthorizedKeysFile __PROGRAMDATA__/ssh/administrators_authorized_keys
"@
    Set-Content -Path $sshdConfigPath -Value $defaultConfig -Encoding UTF8
    Write-Host "Created complete sshd_config"
}

# Restart SSH service with new config
Write-Host "Restarting SSH service with new config..."
Stop-Service sshd -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Start-Service sshd

# Wait for service to start
Start-Sleep -Seconds 3
$sshStatus = Get-Service sshd -ErrorAction SilentlyContinue
Write-Host "SSH Service Status: $($sshStatus.Status)"

# Show status
Write-Host ""
Write-Host "=== Status Check ==="
Write-Host "SSH Service:"
Get-Service sshd | Format-Table Name, Status, StartType

Write-Host "Listening on port $SSHPort :"
netstat -an | findstr ":$SSHPort.*LISTENING"

Write-Host "Firewall status:"
Get-NetFirewallProfile | Format-Table Name, Enabled

Write-Host "authorized_keys content:"
Get-Content $AuthorizedKeysPath

Write-Host "sshd_config Port setting:"
Get-Content $sshdConfigPath | Select-String "^Port"

Write-Host ""
Write-Host "=== SSH Setup Complete on Port $SSHPort ==="
"SSH_READY" | Out-File -FilePath "C:\ssh_ready.txt"

Stop-Transcript
</powershell>
<persist>true</persist>
