# Install / uninstall the addon as a Blender 5+ extension on Windows.
#
# Drops a directory junction at:
#   <BlenderBase>\<version>\extensions\user_default\ppf_contact_solver
# pointing at this repo's blender_addon\ directory. The manifest inside
# blender_addon\blender_manifest.toml is what makes Blender treat it
# as an extension; from there it loads under the module name
# bl_ext.user_default.ppf_contact_solver.
#
# Usage:
#   Install:   .\install-blender-addon.ps1
#   Uninstall: .\install-blender-addon.ps1 -Uninstall
#                 prompts (interactively, default No) to also wipe
#                 third-party deps the addon pip-installed into
#                 Blender's user scripts\addons\modules\ dir.
#
# Cold-start (e.g. fresh runner where Blender hasn't booted yet):
#   $env:PPF_BLENDER_BIN="C:\path\to\blender.exe"; .\install-blender-addon.ps1
# The version dir is derived from `<bin> --version` and created.

param(
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AddonSource = Join-Path $ScriptDir "blender_addon"
$AddonName = "ppf_contact_solver"
# Stale dirname from very old installs that used a hyphen. Python module
# names cannot contain hyphens, so this variant could never have worked
# under addon_enable; we sweep it on install/uninstall.
$LegacyAddonName = "ppf-contact-solver"

$BlenderBase = Join-Path $env:APPDATA "Blender Foundation\Blender"

# Resolve a Blender version directory. Prefer an existing one; otherwise
# ask a Blender binary (PPF_BLENDER_BIN) to print its version and create
# the dir from that. The latter path is what CI hits when no Blender
# has booted to seed the user prefs tree yet.
function Get-VersionFromExisting {
    if (-not (Test-Path $BlenderBase)) { return $null }
    return Get-ChildItem -Path $BlenderBase -Directory |
        Where-Object { $_.Name -match '^\d+\.\d+$' } |
        Sort-Object { [version]$_.Name } |
        Select-Object -Last 1 -ExpandProperty Name
}

function Get-VersionFromBinary($BinPath) {
    if (-not (Test-Path $BinPath)) { return $null }
    $out = & $BinPath --version 2>$null | Select-Object -First 1
    if ($out -match '^Blender (\d+)\.(\d+)') {
        return "$($Matches[1]).$($Matches[2])"
    }
    return $null
}

$BlenderVersion = Get-VersionFromExisting
if (-not $BlenderVersion -and $env:PPF_BLENDER_BIN) {
    $BlenderVersion = Get-VersionFromBinary $env:PPF_BLENDER_BIN
    if (-not $BlenderVersion) {
        Write-Error "Could not parse version from PPF_BLENDER_BIN=$env:PPF_BLENDER_BIN"
        exit 1
    }
    Write-Host "Cold-start: derived Blender version $BlenderVersion from $env:PPF_BLENDER_BIN"
}

if (-not $BlenderVersion) {
    Write-Error "No Blender version directory found in $BlenderBase. Either launch Blender once to materialize it, or set `$env:PPF_BLENDER_BIN and re-run."
    exit 1
}

# Refuse to install for pre-5 Blender. Manifest declares min 5.0.0.
$Major = [int]($BlenderVersion -split '\.')[0]
if ($Major -lt 5) {
    Write-Error "Blender $BlenderVersion is not supported. This addon requires Blender 5.0 or later (extensions system)."
    exit 1
}

$ExtDir = Join-Path $BlenderBase "$BlenderVersion\extensions\user_default"
$ExtLink = Join-Path $ExtDir $AddonName

# Legacy locations we sweep on every install run so an old layout cannot
# shadow the canonical extension junction.
$LegacyAddonsDir = Join-Path $BlenderBase "$BlenderVersion\scripts\addons"
$LegacyLink = Join-Path $LegacyAddonsDir $AddonName
$LegacyHyphenLink = Join-Path $LegacyAddonsDir $LegacyAddonName

Write-Host "Blender version: $BlenderVersion"
Write-Host "Extensions directory: $ExtDir"

if ($Uninstall) {
    $removed = $false
    foreach ($link in @($ExtLink, $LegacyLink, $LegacyHyphenLink)) {
        if (Test-Path $link) {
            Remove-Item -Path $link -Recurse -Force
            Write-Host "Removed: $link"
            $removed = $true
        }
    }
    if (-not $removed) {
        Write-Host "Addon not installed: $ExtLink"
    }

    # Offer to wipe third-party Python deps that the addon's Install
    # operators (paramiko, docker) and the test rig's orchestrator
    # (cbor2) pip-install into Blender's user modules dir. The dir is
    # bpy.utils.user_resource('SCRIPTS', path='addons/modules'); see
    # blender_addon\core\module.py:get_install_target. Default is No;
    # non-tty stdin (CI) skips the prompt and leaves them in place.
    $ModulesDir = Join-Path $BlenderBase "$BlenderVersion\scripts\addons\modules"
    if (Test-Path $ModulesDir) {
        $entries = Get-ChildItem -Path $ModulesDir -Force
        if ($entries) {
            Write-Host ""
            Write-Host "Third-party packages in ${ModulesDir}:"
            foreach ($entry in $entries) {
                Write-Host "  $($entry.Name)"
            }
            if (-not [Console]::IsInputRedirected) {
                $answer = Read-Host "Also remove these? [y/N]"
                if ($answer -match '^(y|Y|yes|YES)$') {
                    Remove-Item -Path $ModulesDir -Recurse -Force
                    Write-Host "Removed: $ModulesDir"
                } else {
                    Write-Host "Kept: $ModulesDir"
                }
            } else {
                Write-Host "(non-interactive stdin; left in place)"
            }
        }
    }
    exit 0
}

# Fetch the cbor2 wheels declared in blender_manifest.toml. They are
# gitignored; without them Blender will refuse to enable the extension
# because the wheel paths in the manifest won't resolve. fetch.py is
# idempotent: re-runs are no-ops when the local files already match
# the pinned sha256 digests.
# Resolve a real Python interpreter explicitly instead of trusting bare
# `python` from PATH. On Windows, `python` commonly resolves to the
# Microsoft Store App-execution-alias stub: Get-Command reports it as
# present, but running it prints "Python was not found..." and exits
# 9009, which (under $ErrorActionPreference="Stop") aborts this script
# before the junction is ever created. So we (1) honor an explicit
# override, (2) prefer the embedded interpreter shipped under
# build-win-native, and only then (3) fall back to PATH candidates --
# and in every case we accept a candidate only if it actually prints a
# version, which rejects the Store stub.
function Test-PythonRuns($bin) {
    if (-not $bin) { return $false }
    try {
        $v = & $bin --version 2>&1
        return ($LASTEXITCODE -eq 0 -and "$v" -match 'Python \d')
    } catch {
        return $false
    }
}

$PythonCandidates = @(
    $env:PPF_BUILD_PYTHON,      # explicit override (matches the rig's build-python knobs)
    $env:PPF_PYTHON_BIN,        # legacy override name kept for back-compat
    (Join-Path $ScriptDir "build-win-native\dist\python\python.exe"),  # embedded interpreter
    "python",
    "python3",
    "py"
)
$PythonBin = $null
foreach ($cand in $PythonCandidates) {
    if (Test-PythonRuns $cand) { $PythonBin = $cand; break }
}
if (-not $PythonBin) {
    Write-Error ("No working Python found for blender_addon\wheels\fetch.py " +
        "(tried PPF_BUILD_PYTHON, PPF_PYTHON_BIN, the embedded interpreter, " +
        "python, python3, py). Set PPF_BUILD_PYTHON to a python.exe.")
    exit 1
}
Write-Host "Using Python for fetch.py: $PythonBin"
& $PythonBin (Join-Path $AddonSource "wheels\fetch.py")
if ($LASTEXITCODE -ne 0) {
    Write-Error "blender_addon\wheels\fetch.py failed (exit $LASTEXITCODE)"
    exit 1
}

if (-not (Test-Path $ExtDir)) {
    New-Item -ItemType Directory -Path $ExtDir -Force | Out-Null
}

# Sweep legacy installs unconditionally so the new extension junction
# isn't shadowed by an old scripts/addons copy.
foreach ($legacy in @($LegacyLink, $LegacyHyphenLink)) {
    if (Test-Path $legacy) {
        Remove-Item -Path $legacy -Recurse -Force
        Write-Host "Removed legacy install: $legacy"
    }
}

# Idempotent rebind. Junctions can't be atomically swung like POSIX
# symlinks, so we delete + recreate.
if (Test-Path $ExtLink) {
    Remove-Item -Path $ExtLink -Recurse -Force
}
cmd /c mklink /J "$ExtLink" "$AddonSource" | Out-Null
Write-Host "Created junction: $ExtLink -> $AddonSource"
Write-Host ""
Write-Host "Enable in Blender with:"
Write-Host "    blender --addons bl_ext.user_default.$AddonName"
