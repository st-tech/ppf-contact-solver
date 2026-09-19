# File: phony-rig.ps1
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A stand-in for `run-blender-rig.ps1` that runs no Blender, no server and no
# solver, and produces every file the workflow's waiting, collecting and
# gating steps read, in the shapes they read them:
#
#   C:\ci\rig.log         progress lines `slot N -> <scenario>` (poll-rig.ps1)
#   C:\ci\report.json     the orchestrator's report shape (the artifact)
#   C:\ppf-debug\...      a worker directory (Collect results copies it)
#   C:\ci\rig_exit.txt    the exit code, written last, in a finally
#
# WHY. The Windows leg takes about an hour to reach its rig and up to two
# more to wait for it, and the steps AFTER the rig (the wait windows, the
# collect, the gate) failed twice at the three hour mark before they had
# ever run against a rig that reported. `pipeline_test` in blender.yml
# skips the driver, the warmup, the build and Blender, and launches THIS
# through the same launch-rig.ps1, WMI detachment, poll-rig.ps1 and
# wait-rig.sh, so the whole chain after the rig is exercised in minutes.
# It proves the PIPELINE, and nothing about the solver: a green
# pipeline_test run says only that a rig which reports is collected and
# gated correctly.
#
# `-Fail` makes it report a failure, so the gate's red path can be tested
# as well as its green one.

param(
    [int]$Seconds = 60,
    [switch]$Fail
)

$CiDir = "C:\ci"
New-Item -ItemType Directory -Path $CiDir -Force | Out-Null
$Log = "$CiDir\rig.log"
Remove-Item "$CiDir\rig_exit.txt" -ErrorAction SilentlyContinue
function Log($msg) { $msg | Tee-Object -FilePath $Log -Append }

$exitCode = 1
try {
    Log "=== PHONY rig (pipeline test, no Blender, no solver) starting $(Get-Date -Format o) ==="
    $worker = "C:\ppf-debug\phony\worker-00"
    New-Item -ItemType Directory -Path $worker -Force | Out-Null
    $scenarios = @("phony_pipeline_a", "phony_pipeline_b", "phony_pipeline_c")
    $per = [Math]::Max(1, [int]($Seconds / $scenarios.Count))
    $slot = 0
    foreach ($name in $scenarios) {
        Log "[orchestrator] slot $slot -> $name"
        Start-Sleep -Seconds $per
        Add-Content -Path "$worker\scenario.log" -Value "${name}: pass (phony)"
        Log "[orchestrator] slot $slot <- $name pass (${per}s)"
        $slot++
    }
    $failed = if ($Fail) { 1 } else { 0 }
    $report = @{
        run_id = "phony-pipeline-test"
        passed = $scenarios.Count - $failed
        failed = $failed
        total = $scenarios.Count
        unrunnable = 0
        phony = $true
    } | ConvertTo-Json
    Set-Content -Path "$CiDir\report.json" -Value $report -Encoding ASCII
    $exitCode = $failed
    Log "rig exit code: $exitCode"
}
catch {
    Log "EXCEPTION: $_"
    $exitCode = 1
}
finally {
    Set-Content -Path "$CiDir\rig_exit.txt" -Value "$exitCode" -Encoding ASCII -NoNewline
    Log "=== PHONY rig finished (exit=$exitCode) $(Get-Date -Format o) ==="
    exit $exitCode
}
