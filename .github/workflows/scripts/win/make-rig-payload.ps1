# File: make-rig-payload.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Runs on the Blender CI BUILD instance after build.bat: packs everything a
# rig instance needs to run the rig from `C:\ppf-contact-solver`, and nothing
# it does not, into one tar the builder then pushes to every rig over the VPC.
#
# WHAT A RIG READS, taken from run-blender-rig.ps1 and install-addon.ps1: the
# source tree (blender_addon, frontend, examples, install-blender-addon.ps1),
# each backend's three artifacts and marker in target\<backend>\release, the
# backend libraries in crates\ppf-cts-compute\<backend>\build\lib, and from
# the toolchain only what runs at run time: the embedded Python with its
# packages, cuda\bin (cudart), rocm\bin (the HIP runtime DLLs) and mingit.
#
# WHAT IT LEAVES BEHIND, because a rig compiles nothing: the toolchain's
# compilers and SDKs (msvc, msys64, rust, the rest of cuda and rocm), the
# downloaded installers, cargo's intermediates under target (deps, build,
# incremental, .fingerprint), the CUDA and ROCm object directories, and .git.
# Measured, that is the difference between a tree of tens of gigabytes and a
# payload of a few.
#
# THE EMBEDDED INTERPRETER PINS ITS TREE. python311._pth lists the checkout
# as a sys.path entry and puts CPython in isolated mode, so a Python warmed
# under one path imports THAT tree's frontend wherever the directory is later
# copied. The payload therefore must land at the same path it was built
# under, and this script refuses to pack an
# interpreter whose ._pth names anything else.

param(
    [string]$Src = "C:\ppf-contact-solver",
    [string]$Out = "C:\rig-payload.tar",
    # Space-separated GPU backends whose builds go in (cpu always does).
    [string]$GpuBackends = "cuda rocm"
)

$ErrorActionPreference = "Stop"
$Stage = "C:\rig-payload"
$Dest = Join-Path $Stage "ppf-contact-solver"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Path $Dest -Force | Out-Null

$pth = Get-ChildItem -Path "$Src\build-win-native\python" -Filter "python*._pth" | Select-Object -First 1
if (-not $pth) { throw "no python*._pth under $Src\build-win-native\python; was warmup.bat run?" }
$pinned = Get-Content $pth.FullName | Where-Object { $_ -match '^[A-Za-z]:\\' }
foreach ($line in $pinned) {
    if (-not $line.StartsWith($Src, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "the embedded interpreter's $($pth.Name) names '$line', not a path under $Src; a rig would import that tree"
    }
}

function Copy-Tree([string]$from, [string]$to, [string[]]$excludeDirs) {
    New-Item -ItemType Directory -Path $to -Force | Out-Null
    $args = @($from, $to, "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/R:2", "/W:2")
    if ($excludeDirs.Count) { $args += "/XD"; $args += $excludeDirs }
    & robocopy @args | Out-Null
    # robocopy exit codes below 8 are successes.
    if ($LASTEXITCODE -ge 8) { throw "robocopy $from -> $to failed with exit $LASTEXITCODE" }
}

# 1. The source tree, without .git, target, the toolchain and the object dirs;
#    those parts that ARE needed are copied back below, narrowly.
$excl = @(
    (Join-Path $Src ".git"),
    (Join-Path $Src "target"),
    (Join-Path $Src "build-win-native"),
    (Join-Path $Src "crates\ppf-cts-compute\cuda\build"),
    (Join-Path $Src "crates\ppf-cts-compute\rocm\build"),
    # Outputs and caches a build or a smoke run leaves in the tree: measured
    # 572 MB of `export` alone on a tree that had run an example.
    (Join-Path $Src "export"),
    (Join-Path $Src "cache"),
    (Join-Path $Src "local"),
    (Join-Path $Src ".pytest_cache"),
    (Join-Path $Src ".ruff_cache")
)
Copy-Tree $Src $Dest $excl

# 1b. The cbor2 wheels the add-on installer needs, fetched HERE so that a rig
#     reads them from the payload and never from PyPI. install-blender-addon.ps1
#     runs wheels\fetch.py, which downloads any wheel not already on disk and
#     is a no-op for one whose hash matches; on run 35180161718 that download
#     failed on all four rig instances at once (exit 1, the reason not
#     captured) while this builder had fetched the same files in earlier
#     runs, so the network step is done where it is known to work and the
#     rigs take the idempotent path.
$py = Join-Path $Src "build-win-native\python\python.exe"
if (-not (Test-Path $py)) { throw "$py is missing; was warmup.bat run?" }
& $py (Join-Path $Src "blender_addon\wheels\fetch.py")
if ($LASTEXITCODE -ne 0) { throw "blender_addon\wheels\fetch.py exited $LASTEXITCODE on the builder" }
$wheels = @(Get-ChildItem -Path (Join-Path $Src "blender_addon\wheels") -Filter "*.whl")
if ($wheels.Count -eq 0) { throw "fetch.py left no wheel under blender_addon\wheels" }
Copy-Tree (Join-Path $Src "blender_addon\wheels") (Join-Path $Dest "blender_addon\wheels") @()
Write-Output ("cbor2 wheels in the payload: {0}" -f $wheels.Count)

# 2. Each backend's artifacts and marker.
$backends = @("cpu") + @($GpuBackends -split '\s+' | Where-Object { $_ })
foreach ($b in $backends) {
    $rel = "target\$b\release"
    $from = Join-Path $Src $rel
    if (-not (Test-Path $from)) { throw "no build under $from (backend $b)" }
    $to = Join-Path $Dest $rel
    New-Item -ItemType Directory -Path $to -Force | Out-Null
    foreach ($f in @("ppf-contact-solver.exe", "ppf-cts-server.exe", "_ppf_cts_py.dll", ".ppf-backend")) {
        $p = Join-Path $from $f
        if (-not (Test-Path $p)) { throw "$p is missing; build.bat did not finish that backend" }
        Copy-Item $p $to
    }
    if ($b -ne "cpu") {
        $lib = Join-Path $Src "crates\ppf-cts-compute\$b\build\lib"
        if (-not (Test-Path $lib)) { throw "$lib is missing; the $b backend library was not built" }
        Copy-Tree $lib (Join-Path $Dest "crates\ppf-cts-compute\$b\build\lib") @()
    }
}

# 3. The toolchain's run-time pieces only.
$bw = Join-Path $Src "build-win-native"
$bwDest = Join-Path $Dest "build-win-native"
Copy-Tree (Join-Path $bw "python") (Join-Path $bwDest "python") @()
Copy-Tree (Join-Path $bw "mingit") (Join-Path $bwDest "mingit") @()
Copy-Tree (Join-Path $bw "cuda\bin") (Join-Path $bwDest "cuda\bin") @()
if ($backends -contains "rocm") {
    Copy-Tree (Join-Path $bw "rocm\bin") (Join-Path $bwDest "rocm\bin") @()
}
# The batch files beside them, which build.bat's launchers and config read.
Get-ChildItem -Path $bw -Filter "*.bat" | Copy-Item -Destination $bwDest
if (Test-Path (Join-Path $bw "scripts")) { Copy-Tree (Join-Path $bw "scripts") (Join-Path $bwDest "scripts") @() }

# 4. No reparse points in the payload. Where the checkout carries a symbolic
#    link, robocopy stages it as a link and Windows tar then fails on it with
#    `Cannot stat: Invalid argument` (measured). None is read by the rig, so
#    they are dropped rather than resolved.
$links = @(Get-ChildItem -Recurse -Force -Path $Dest -Attributes ReparsePoint -ErrorAction SilentlyContinue)
foreach ($l in $links) { Remove-Item -Force -Recurse $l.FullName -ErrorAction SilentlyContinue }
Write-Output ("dropped {0} reparse point(s)" -f $links.Count)

# 5. One tar, uncompressed: the bulk is already-compressed DLLs and wheels,
#    and the copy is in-VPC, so time goes to disk, not to the wire.
if (Test-Path $Out) { Remove-Item -Force $Out }
& tar -cf $Out -C $Stage "ppf-contact-solver"
if ($LASTEXITCODE -ne 0) { throw "tar exited $LASTEXITCODE" }
$size = (Get-Item $Out).Length
$files = (Get-ChildItem -Recurse -File $Dest | Measure-Object).Count
Write-Output ("rig payload: {0} ({1:N0} MB, {2} files, backends {3})" -f $Out, ($size / 1MB), $files, ($backends -join " "))
Remove-Item -Recurse -Force $Stage
