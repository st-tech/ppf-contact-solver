# File: run-build.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Runs DETACHED on the Blender CI build instance (launch-build.ps1 starts it):
# warmup.bat then build.bat, both through `call`, without which `cmd /c`
# loses an `exit /b` fired inside a parenthesized block, for the backends
# named, with every line they print in C:\ci\build.log and the
# exit code in C:\ci\build_exit.txt, written last, in a `finally`, so its
# presence is the finish signal poll-build.ps1 reports.
#
# WHY DETACHED. A blocking ssh through an ec2-instance-connect tunnel dies at
# 60 minutes, and warmup plus a three-backend build sat near that line. More
# to the point, while this runs the workflow prepares the rig instances
# (driver, Blender) in parallel, which a blocking step could not allow.

param(
    [string]$Backends = "cuda rocm cpu",
    [string]$Src = "C:\ppf-contact-solver"
)

$CiDir = "C:\ci"
New-Item -ItemType Directory -Path $CiDir -Force | Out-Null
$Log = "$CiDir\build.log"
$ExitFile = "$CiDir\build_exit.txt"
Remove-Item $ExitFile -ErrorAction SilentlyContinue
$env:PPF_WIN_BACKENDS = $Backends

$code = 1
try {
    Add-Content $Log ("[{0}] warmup.bat + build.bat for backends: {1}" -f (Get-Date -Format o), $Backends)
    # cmd runs the chain; a failed warmup stops it and its exit code is the
    # chain's. The batch files' own output is appended by cmd's redirection.
    & $env:ComSpec /c "cd /d `"$Src\build-win-native`" && call warmup.bat /nopause >> `"$Log`" 2>&1 && call build.bat /nopause >> `"$Log`" 2>&1"
    $code = $LASTEXITCODE
    Add-Content $Log ("[{0}] build chain exit {1}" -f (Get-Date -Format o), $code)
} catch {
    Add-Content $Log "EXCEPTION: $_"
    $code = 1
} finally {
    Set-Content -Path $ExitFile -Value "$code" -Encoding ASCII -NoNewline
}
exit $code
