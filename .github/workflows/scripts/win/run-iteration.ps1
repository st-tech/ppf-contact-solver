# File: run-iteration.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

$ErrorActionPreference = "Continue"

$Example = "EXAMPLE_PLACEHOLDER"
$Iteration = "ITERATION_PLACEHOLDER"
$IterationNum = "ITERATION_NUM_PLACEHOLDER"

# Set up environment like start.bat does
$env:PATH = "C:\ppf-contact-solver\target\release;C:\ppf-contact-solver\src\cpp\build\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;C:\ppf-contact-solver\build-win-native\python;C:\ppf-contact-solver\build-win-native\python\Scripts;" + $env:PATH
$env:PYTHONPATH = "C:\ppf-contact-solver;" + $env:PYTHONPATH

cd C:\ppf-contact-solver

# Set CI marker
Set-Content -Path "frontend\.CI" -Value $Iteration -Encoding ASCII -NoNewline

# Run the example
Write-Host "Running $Iteration iteration of $Example..."
& C:\ppf-contact-solver\build-win-native\python\python.exe "$env:TEMP\ci\$Example.py" 2>&1 | Tee-Object -FilePath "$env:TEMP\ci\run_$IterationNum.log"

# Propagate Python exit code
exit $LASTEXITCODE
