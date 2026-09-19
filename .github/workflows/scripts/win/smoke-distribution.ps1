# File: smoke-distribution.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Runs example scenes from an unpacked Windows distribution on its CPU backend and
# judges each by the frames it wrote and the motion in them. It is for hosts with
# no GPU the distribution can use: the x64 ROCm distribution's CPU backend on a
# machine without an AMD GPU, and the ARM64 distribution, whose only backend is
# the CPU one. No workflow runs it yet.
#
#     powershell -File smoke-distribution.ps1 -Dist C:\path\to\dist -Scenes drape
#
# WHY THE VERDICT IS NOT THE EXIT CODE. An example's exit status is not evidence
# that it simulated: a solver that aborts part way can leave
# a notebook exiting 0, and a stale solver process makes the frontend refuse to
# start while the script still exits 0. So the scene's own output is read: at
# least two frames written by this run, and a mean vertex displacement between the
# first and the last that is not zero, read as float32 triples in numeric frame
# order, which is how examples/run_suite.py reads them.
#
# THE DISTRIBUTION IS RUN AS A USER RUNS IT: its own interpreter, its own bin\ and
# MinGit on PATH, and CARGO_TARGET_DIR naming target\cpu, which is what the
# distribution's launchers set where the CPU backend is the one asked for.
# Every scene runs in fast-check mode, one frame, through the same
# inject_fast_check.py that fast-check-all.bat uses.
param(
    [Parameter(Mandatory = $true)][string]$Dist,
    [string[]]$Scenes = @('drape')
)

$ErrorActionPreference = 'Stop'

function Fail([string]$message) {
    Write-Host "ERROR: $message"
    exit 1
}

$Dist = (Resolve-Path $Dist).Path
$python = Join-Path $Dist 'python\python.exe'
$cpuSolver = Join-Path $Dist 'target\cpu\release\ppf-contact-solver.exe'
foreach ($required in @($python, $cpuSolver, (Join-Path $Dist 'inject_fast_check.py'))) {
    if (-not (Test-Path $required)) { Fail "$required is not in the distribution" }
}

$env:PATH = "$Dist\python;$Dist\python\Scripts;$Dist\bin;$Dist\mingit\cmd;" + $env:PATH
$env:CARGO_TARGET_DIR = Join-Path $Dist 'target\cpu'
$env:SSL_CERT_FILE = Join-Path $Dist 'python\Lib\site-packages\certifi\cacert.pem'
$env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE

$backend = (& $cpuSolver --backend | Select-Object -First 1)
if ($backend -ne 'cpu') { Fail "$cpuSolver answers --backend '$backend', expected 'cpu'" }
Write-Host "target\cpu\release\ppf-contact-solver.exe answers --backend cpu"

$work = Join-Path $Dist 'smoke'
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Force -Path $work | Out-Null
$dataRoot = Join-Path $Dist 'local\share\ppf-cts'

$judge = @'
import os, re, sys
import numpy as np
directory = sys.argv[1]
frames = []
for name in os.listdir(directory):
    m = re.match(r"vert_(\d+)\.bin$", name)
    if m:
        frames.append((int(m.group(1)), os.path.join(directory, name)))
frames.sort()
arrays = [np.fromfile(p, dtype=np.float32) for _, p in frames]
arrays = [a.reshape(-1, 3) for a in arrays if a.size and a.size % 3 == 0]
if len(arrays) < 2:
    print(f"  {len(arrays)} frame(s) in {directory}, and motion needs two")
    sys.exit(1)
if arrays[0].shape != arrays[-1].shape:
    print(f"  the first and last frames differ in size: {arrays[0].shape} and {arrays[-1].shape}")
    sys.exit(1)
disp = np.linalg.norm(arrays[-1] - arrays[0], axis=1)
finite = bool(np.isfinite(disp).all())
print(f"  frames={len(arrays)} vertices={arrays[0].shape[0]} disp_mean={disp.mean():.6e} "
      f"disp_max={disp.max():.6e} finite={finite}")
sys.exit(0 if finite and disp.mean() > 0.0 else 1)
'@
$judgeScript = Join-Path $work 'judge.py'
Set-Content -Path $judgeScript -Value $judge -Encoding ascii

$failed = @()
foreach ($scene in $Scenes) {
    Write-Host "=== $scene ==="
    $notebook = Join-Path $Dist "examples\$scene.ipynb"
    if (-not (Test-Path $notebook)) { Fail "$notebook is not in the distribution" }
    & $python -m nbconvert --to script $notebook --output-dir $work
    if ($LASTEXITCODE -ne 0) { Fail "nbconvert failed on $scene" }
    $script = Join-Path $work "$scene.py"
    & $python (Join-Path $Dist 'inject_fast_check.py') $script
    if ($LASTEXITCODE -ne 0) { Fail "inject_fast_check.py failed on $scene" }

    $started = Get-Date
    Push-Location (Join-Path $Dist 'examples')
    try {
        & $python $script
        $rc = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    Write-Host "  exit code $rc"

    # THE OUTPUT DIRECTORY IS RESOLVED BY WHAT THIS RUN WROTE, not by a guessed
    # path: a session directory's name comes from the notebook and the tree's
    # branch, neither of which this script should restate.
    $written = @()
    if (Test-Path $dataRoot) {
        $written = Get-ChildItem -Path $dataRoot -Recurse -File -Filter 'vert_*.bin' |
            Where-Object { $_.LastWriteTime -ge $started }
    }
    if ($written.Count -eq 0) {
        Write-Host "  no frame was written under $dataRoot by this run"
        $failed += $scene
        continue
    }
    $outputDir = ($written | Group-Object DirectoryName | Sort-Object Count -Descending | Select-Object -First 1).Name
    Write-Host "  output $outputDir"
    & $python $judgeScript $outputDir
    if ($LASTEXITCODE -ne 0 -or $rc -ne 0) { $failed += $scene }
}

if ($failed.Count -gt 0) { Fail ("scenes that did not simulate: " + ($failed -join ', ')) }
Write-Host "[OK] every scene wrote frames that move: $($Scenes -join ', ')"
