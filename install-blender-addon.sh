#!/bin/bash
#
# Install / uninstall the addon as a Blender 5+ extension.
#
# Drops a symlink at:
#   <BLENDER_BASE>/<version>/extensions/user_default/ppf_contact_solver
# pointing at this repo's blender_addon/ directory. The manifest inside
# blender_addon/blender_manifest.toml is what makes Blender treat it
# as an extension; from there it loads under the module name
# bl_ext.user_default.ppf_contact_solver.
#
# Usage:
#   Install:   ./install-blender-addon.sh
#   Uninstall: ./install-blender-addon.sh --uninstall
#
# Cold-start (e.g. fresh CI runner where Blender hasn't booted yet):
#   PPF_BLENDER_BIN=/opt/blender/blender ./install-blender-addon.sh
# The version dir is derived from `<bin> --version` and created.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADDON_SOURCE="$SCRIPT_DIR/blender_addon"
ADDON_NAME="ppf_contact_solver"
UNINSTALL=false

if [ "$1" = "--uninstall" ]; then
  UNINSTALL=true
fi

case "$(uname -s)" in
Darwin)
  BLENDER_BASE="$HOME/Library/Application Support/Blender"
  ;;
Linux)
  BLENDER_BASE="$HOME/.config/blender"
  ;;
MINGW* | MSYS* | CYGWIN*)
  BLENDER_BASE="$APPDATA/Blender Foundation/Blender"
  ;;
*)
  echo "Unsupported OS: $(uname -s)"
  exit 1
  ;;
esac

# Resolve a Blender version directory. If one already exists we use the
# newest; otherwise we ask a Blender binary (PPF_BLENDER_BIN) to tell us
# its major.minor and create the dir. The latter path is what CI hits
# because no Blender has booted yet to seed the user prefs tree.
detect_version_from_existing() {
  if [ ! -d "$BLENDER_BASE" ]; then
    return 1
  fi
  ls -1 "$BLENDER_BASE" 2>/dev/null \
    | grep -E '^[0-9]+\.[0-9]+$' \
    | sort -V \
    | tail -n1
}

detect_version_from_binary() {
  local bin="$1"
  [ -x "$bin" ] || return 1
  # `blender --version` prints lines like "Blender 5.1.1\n  build date: ...".
  # We want just "5.1" (major.minor); that's the user-prefs dirname.
  "$bin" --version 2>/dev/null \
    | head -n1 \
    | sed -nE 's/^Blender ([0-9]+)\.([0-9]+).*/\1.\2/p'
}

BLENDER_VERSION="$(detect_version_from_existing || true)"
if [ -z "$BLENDER_VERSION" ] && [ -n "${PPF_BLENDER_BIN:-}" ]; then
  BLENDER_VERSION="$(detect_version_from_binary "$PPF_BLENDER_BIN" || true)"
  if [ -z "$BLENDER_VERSION" ]; then
    echo "Could not parse version from PPF_BLENDER_BIN=$PPF_BLENDER_BIN"
    exit 1
  fi
  echo "Cold-start: derived Blender version $BLENDER_VERSION from $PPF_BLENDER_BIN"
fi

if [ -z "$BLENDER_VERSION" ]; then
  echo "No Blender version directory found in $BLENDER_BASE."
  echo "Either launch Blender once to materialize it, or set "
  echo "PPF_BLENDER_BIN=<path/to/blender> and re-run."
  exit 1
fi

# Refuse to install for pre-5 Blender. The manifest declares
# blender_version_min = 5.0.0 and the runtime expects the extension
# loader's bl_ext.user_default.<id> module name.
major="${BLENDER_VERSION%%.*}"
if [ "$major" -lt 5 ]; then
  echo "Blender $BLENDER_VERSION is not supported."
  echo "This addon requires Blender 5.0 or later (extensions system)."
  exit 1
fi

EXT_DIR="$BLENDER_BASE/$BLENDER_VERSION/extensions/user_default"
EXT_LINK="$EXT_DIR/$ADDON_NAME"

# Legacy locations we sweep on every run so an old install can't shadow
# the canonical extension symlink. Two flavors of legacy:
#   1. scripts/addons/<id>            -- pre-extension layout (5.0 still loads it)
#   2. scripts/addons/ppf-contact-solver -- hyphenated dirname from very old installs
LEGACY_ADDONS_DIR="$BLENDER_BASE/$BLENDER_VERSION/scripts/addons"
LEGACY_LINK="$LEGACY_ADDONS_DIR/$ADDON_NAME"
LEGACY_HYPHEN_LINK="$LEGACY_ADDONS_DIR/ppf-contact-solver"

echo "Blender version: $BLENDER_VERSION"
echo "Extensions directory: $EXT_DIR"

# Handle uninstall
if [ "$UNINSTALL" = true ]; then
  removed=false
  for link in "$EXT_LINK" "$LEGACY_LINK" "$LEGACY_HYPHEN_LINK"; do
    if [ -e "$link" ] || [ -L "$link" ]; then
      rm -rf "$link"
      echo "Removed: $link"
      removed=true
    fi
  done
  if [ "$removed" = false ]; then
    echo "Addon not installed: $EXT_LINK"
  fi
  exit 0
fi

mkdir -p "$EXT_DIR"

# Sweep legacy installs unconditionally. Either the user is upgrading
# from the old layout or never had one; in both cases an existing
# scripts/addons/<id> symlink would have Blender load both the
# extension and the legacy copy, conflicting on operator names.
for legacy in "$LEGACY_LINK" "$LEGACY_HYPHEN_LINK"; do
  if [ -e "$legacy" ] || [ -L "$legacy" ]; then
    rm -rf "$legacy"
    echo "Removed legacy install: $legacy"
  fi
done

# Idempotent replacement of the canonical symlink. ``ln -sfn`` swings
# the link atomically when the destination already exists, so a re-run
# is a no-op if nothing changed and a clean rebind otherwise.
if [ -L "$EXT_LINK" ] && [ "$(readlink "$EXT_LINK")" = "$ADDON_SOURCE" ]; then
  echo "Symlink already correct: $EXT_LINK -> $ADDON_SOURCE"
else
  ln -sfn "$ADDON_SOURCE" "$EXT_LINK"
  echo "Created symlink: $EXT_LINK -> $ADDON_SOURCE"
fi

echo
echo "Enable in Blender with:"
echo "    blender --addons bl_ext.user_default.$ADDON_NAME"
