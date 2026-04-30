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
    exit 0
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
