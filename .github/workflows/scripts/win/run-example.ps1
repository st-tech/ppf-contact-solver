# File: run-example.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

$ErrorActionPreference = "Continue"

$Example = "EXAMPLE_PLACEHOLDER"

# Set up environment like start.bat does (use local CUDA from build-win-native)
$env:PATH = "C:\ppf-contact-solver\target\release;C:\ppf-contact-solver\src\cpp\build\lib;C:\ppf-contact-solver\build-win-native\cuda\bin;C:\ppf-contact-solver\build-win-native\python;C:\ppf-contact-solver\build-win-native\python\Scripts;C:\ppf-contact-solver\build-win-native\mingit\cmd;" + $env:PATH
$env:PYTHONPATH = "C:\ppf-contact-solver;" + $env:PYTHONPATH

cd C:\ppf-contact-solver

# Create output directory for this example
$ExampleDir = "C:\ci\$Example"
New-Item -ItemType Directory -Path $ExampleDir -Force | Out-Null

# Set CI marker (use ASCII to avoid UTF-8 BOM which corrupts the path)
Set-Content -Path "frontend\.CI" -Value $Example -Encoding ASCII -NoNewline

# Convert notebook to Python script
Write-Host "Converting $Example.ipynb to Python script..."
& C:\ppf-contact-solver\build-win-native\python\python.exe -m jupyter nbconvert --to python "examples\$Example.ipynb" --output "$ExampleDir\${Example}_base.py"

# Create wrapper script with path setup
$wrapperContent = @"
import sys
import os
sys.path.insert(0, 'C:\\ppf-contact-solver')
sys.path.insert(0, 'C:\\ppf-contact-solver\\frontend')
os.environ['PYTHONPATH'] = 'C:\\ppf-contact-solver;C:\\ppf-contact-solver\\frontend;' + os.environ.get('PYTHONPATH', '')
"@

$baseScript = Get-Content "$ExampleDir\${Example}_base.py" -Raw
$fullScript = $wrapperContent + "`n" + $baseScript
Set-Content -Path "$ExampleDir\$Example.py" -Value $fullScript

# Run the example
Write-Host "Running $Example..."
& C:\ppf-contact-solver\build-win-native\python\python.exe "$ExampleDir\$Example.py" 2>&1 | Tee-Object -FilePath "$ExampleDir\$Example.log"

# Propagate Python exit code
exit $LASTEXITCODE
