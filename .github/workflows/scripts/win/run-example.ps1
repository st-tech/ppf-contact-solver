# File: run-example.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

$ErrorActionPreference = "Continue"

$Example = "EXAMPLE_PLACEHOLDER"

# Set up environment like start.bat does
$env:PATH = "C:\ppf-contact-solver\target\release;C:\ppf-contact-solver\src\cpp\build\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;" + $env:PATH
$env:PYTHONPATH = "C:\ppf-contact-solver;" + $env:PYTHONPATH

cd C:\ppf-contact-solver

# Set CI marker (use ASCII to avoid UTF-8 BOM which corrupts the path)
Set-Content -Path "frontend\.CI" -Value $Example -Encoding ASCII -NoNewline

# Convert notebook to Python script
Write-Host "Converting $Example.ipynb to Python script..."
& C:\ppf-contact-solver\build-win-native\python\python.exe -m jupyter nbconvert --to python "examples\$Example.ipynb" --output "C:\ci\${Example}_base.py"

# Create wrapper script with path setup
$wrapperContent = @"
import sys
import os
sys.path.insert(0, 'C:\\ppf-contact-solver')
sys.path.insert(0, 'C:\\ppf-contact-solver\\frontend')
os.environ['PYTHONPATH'] = 'C:\\ppf-contact-solver;C:\\ppf-contact-solver\\frontend;' + os.environ.get('PYTHONPATH', '')
"@

$baseScript = Get-Content "C:\ci\${Example}_base.py" -Raw
$fullScript = $wrapperContent + "`n" + $baseScript
Set-Content -Path "C:\ci\$Example.py" -Value $fullScript

# Run the example
Write-Host "Running $Example..."
& C:\ppf-contact-solver\build-win-native\python\python.exe "C:\ci\$Example.py" 2>&1 | Tee-Object -FilePath "C:\ci\$Example.log"

# Propagate Python exit code
exit $LASTEXITCODE
