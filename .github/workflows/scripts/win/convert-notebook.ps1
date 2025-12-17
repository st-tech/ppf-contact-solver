# File: convert-notebook.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

$ErrorActionPreference = "Stop"

$Example = "EXAMPLE_PLACEHOLDER"

# Set up Python path
$env:PATH = "C:\ppf-contact-solver\build-win-native\python;C:\ppf-contact-solver\build-win-native\python\Scripts;" + $env:PATH

# Create CI temp directory
New-Item -ItemType Directory -Path "$env:TEMP\ci" -Force | Out-Null

# Convert notebook to Python script
Write-Host "Converting $Example.ipynb to Python script..."
& C:\ppf-contact-solver\build-win-native\python\python.exe -m jupyter nbconvert --to python "C:\ppf-contact-solver\examples\$Example.ipynb" --output "$env:TEMP\ci\${Example}_base.py"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to convert notebook"
    exit 1
}

# Create wrapper script with path setup
$wrapper = @"
import sys
import os
sys.path.insert(0, 'C:\\ppf-contact-solver')
sys.path.insert(0, 'C:\\ppf-contact-solver\\frontend')
os.environ['PYTHONPATH'] = 'C:\\ppf-contact-solver;C:\\ppf-contact-solver\\frontend;' + os.environ.get('PYTHONPATH', '')
"@

$base = Get-Content "$env:TEMP\ci\${Example}_base.py" -Raw
Set-Content -Path "$env:TEMP\ci\$Example.py" -Value ($wrapper + "`n" + $base)

Write-Host "Script prepared at $env:TEMP\ci\$Example.py"
