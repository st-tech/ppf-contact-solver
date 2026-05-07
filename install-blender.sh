#!/bin/bash
#
# Install Blender 5.1.1 + the Xvfb virtual framebuffer (Linux only)
# from the Blender Foundation's official download mirror.
#
# Mirrors the "Linux runtime deps + Xvfb" / "Download Blender" steps
# in `.github/workflows/blender.yml` so a dev host can reproduce the
# exact graphics-stack the CI runs against.
#
# Usage:
#   Install:    ./install-blender.sh
#   Print env:  ./install-blender.sh --env
#   Uninstall:  ./install-blender.sh --uninstall
#
# What goes where (Linux):
#   ~/.local/opt/blender-5.1.1-linux-x64/  -- extracted release tree
#   ~/.local/bin/blender                  -- symlink to the binary
#   /usr/bin/Xvfb                          -- via apt, requires sudo
#
# What goes where (macOS):
#   /Applications/Blender.app              -- copied from the official .dmg
#
# Used together with `install-blender-addon.sh`:
#   ./install-blender.sh             # one-time per host
#   ./install-blender-addon.sh       # per repo checkout, links the
#                                    # addon into Blender's user-prefs
#
# After install, source the env block to run the rig:
#   eval "$(./install-blender.sh --env)"
#   python blender_addon/debug/main.py runtests <scenario>

set -euo pipefail

# Global tempdir so the EXIT trap can clean up after the function
# scope where it was created has unwound. Empty by default; set when
# we actually download something.
BLENDER_TMP=""
trap 'if [ -n "${BLENDER_TMP:-}" ] && [ -d "$BLENDER_TMP" ]; then rm -rf "$BLENDER_TMP"; fi' EXIT

BLENDER_VERSION_DEFAULT="5.1.1"
BLENDER_VERSION="${PPF_BLENDER_VERSION:-$BLENDER_VERSION_DEFAULT}"

OS="$(uname -s)"
case "$OS" in
Darwin)
  PLATFORM=macos
  BLENDER_DIR=/Applications/Blender.app
  BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender
  ;;
Linux)
  PLATFORM=linux
  BLENDER_OPT="$HOME/.local/opt/blender-${BLENDER_VERSION}-linux-x64"
  BLENDER_BIN="$BLENDER_OPT/blender"
  BLENDER_LINK="$HOME/.local/bin/blender"
  ;;
*)
  echo "Unsupported OS: $OS" >&2
  exit 1
  ;;
esac

ACTION="install"
case "${1:-}" in
--uninstall) ACTION="uninstall" ;;
--env) ACTION="env" ;;
"") ACTION="install" ;;
*)
  echo "Usage: $0 [--uninstall|--env]" >&2
  exit 2
  ;;
esac

detect_existing_blender() {
  # If a Blender of the right version is already on PATH (e.g.
  # `/usr/local/bin/blender` from a manual install), prefer it over
  # downloading our own copy. Returns 0 + sets BLENDER_BIN when a
  # matching install is found.
  local bin
  bin="$(command -v blender 2>/dev/null || true)"
  [ -n "$bin" ] && [ -x "$bin" ] || return 1
  local installed
  installed="$("$bin" --version 2>/dev/null | head -n1 \
               | sed -nE 's/^Blender ([0-9.]+).*/\1/p')"
  [ "$installed" = "$BLENDER_VERSION" ] || return 1
  BLENDER_BIN="$bin"
  echo "==> reusing existing Blender $BLENDER_VERSION at $BLENDER_BIN"
}

linux_install() {
  # Runtime libs Blender's glib/X11/openal stack needs to start
  # headless under Xvfb. Mirrors the apt-get list in blender.yml's
  # "Linux runtime deps + Xvfb" step.
  if command -v apt-get >/dev/null 2>&1; then
    local pkgs=(
      xvfb x11-xserver-utils
      libxi6 libxxf86vm1 libxfixes3 libxrender1
      libgl1 libglu1-mesa libxkbcommon0 libsm6
      libdbus-1-3 libsndfile1
    )
    local missing=()
    for p in "${pkgs[@]}"; do
      if ! dpkg -s "$p" >/dev/null 2>&1; then
        missing+=("$p")
      fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
      echo "==> apt install: ${missing[*]}"
      # Containers running as root may have no `sudo` installed; the
      # apt commands work directly. Outside containers we always need
      # sudo because the dev hosts run as a regular user.
      local sudo_cmd=""
      if [ "$(id -u)" -ne 0 ]; then
        sudo_cmd="sudo "
      fi
      # Don't fail the whole script if `apt-get update` errors on a
      # third-party repo (e.g. AWS hosts pin a Radeon repo that may
      # change its Origin/Label between releases). The install step
      # still works against the cached package metadata.
      ${sudo_cmd}apt-get update -qq || echo "warn: apt-get update partial; proceeding"
      if ! ${sudo_cmd}apt-get install -y --no-install-recommends "${missing[@]}"; then
        echo "warn: apt install failed for ${missing[*]}; an existing Blender may still work"
      fi
    else
      echo "==> apt deps already present"
    fi
  else
    echo "warn: apt-get not found; skipping system deps"
  fi

  # Reuse an already-installed Blender of the right version if present.
  if detect_existing_blender; then
    return
  fi

  if [ -x "$BLENDER_BIN" ]; then
    local installed
    installed="$("$BLENDER_BIN" --version 2>/dev/null | head -n1 \
                 | sed -nE 's/^Blender ([0-9.]+).*/\1/p')"
    if [ "$installed" = "$BLENDER_VERSION" ]; then
      echo "==> Blender $BLENDER_VERSION already at $BLENDER_BIN"
    else
      echo "==> Blender $installed found but want $BLENDER_VERSION; reinstalling"
      rm -rf "$BLENDER_OPT"
    fi
  fi

  if [ ! -x "$BLENDER_BIN" ]; then
    local minor="${BLENDER_VERSION%.*}"
    local url="https://download.blender.org/release/Blender${minor}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
    BLENDER_TMP="$(mktemp -d)"
    echo "==> downloading $url"
    curl -fL -o "$BLENDER_TMP/blender.tar.xz" "$url"
    mkdir -p "$BLENDER_OPT"
    echo "==> extracting to $BLENDER_OPT"
    tar -xf "$BLENDER_TMP/blender.tar.xz" -C "$BLENDER_OPT" --strip-components=1
  fi

  mkdir -p "$(dirname "$BLENDER_LINK")"
  if [ -L "$BLENDER_LINK" ] && [ "$(readlink "$BLENDER_LINK")" = "$BLENDER_BIN" ]; then
    echo "==> symlink already points at $BLENDER_BIN"
  else
    ln -sfn "$BLENDER_BIN" "$BLENDER_LINK"
    echo "==> linked $BLENDER_LINK -> $BLENDER_BIN"
  fi

  echo
  echo "Done. Source the env block before running the test rig:"
  echo "    eval \"\$($PWD/install-blender.sh --env)\""
}

macos_install() {
  if [ -x "$BLENDER_BIN" ]; then
    local installed
    installed="$("$BLENDER_BIN" --version 2>/dev/null | head -n1 \
                 | sed -nE 's/^Blender ([0-9.]+).*/\1/p')"
    if [ "$installed" = "$BLENDER_VERSION" ]; then
      echo "==> Blender $BLENDER_VERSION already at $BLENDER_BIN"
      return
    fi
    echo "==> Blender $installed found but want $BLENDER_VERSION; reinstalling"
    rm -rf "$BLENDER_DIR"
  fi
  local arch
  arch="$(uname -m)"
  local minor="${BLENDER_VERSION%.*}"
  local url="https://download.blender.org/release/Blender${minor}/blender-${BLENDER_VERSION}-macos-${arch/x86_64/x64}.dmg"
  BLENDER_TMP="$(mktemp -d)"
  echo "==> downloading $url"
  curl -fL -o "$BLENDER_TMP/blender.dmg" "$url"
  local mount
  # hdiutil prints a table; the volume mountpoint is the last column on
  # the row that mentions /Volumes/. Capture instead of hard-coding so
  # a renamed volume label doesn't break the install silently.
  mount="$(hdiutil attach -nobrowse "$BLENDER_TMP/blender.dmg" \
           | awk '/\/Volumes\// {print $NF; exit}')"
  cp -R "$mount/Blender.app" /Applications/
  hdiutil detach "$mount" >/dev/null
  echo "==> installed to /Applications/Blender.app"
}

linux_uninstall() {
  if [ -L "$BLENDER_LINK" ]; then rm -f "$BLENDER_LINK"; echo "removed $BLENDER_LINK"; fi
  if [ -d "$BLENDER_OPT" ]; then rm -rf "$BLENDER_OPT"; echo "removed $BLENDER_OPT"; fi
}

macos_uninstall() {
  if [ -d "$BLENDER_DIR" ]; then rm -rf "$BLENDER_DIR"; echo "removed $BLENDER_DIR"; fi
}

# Print the env block needed to run the headless rig. On Linux we
# also start Xvfb on :99 if no DISPLAY is set yet. The block is
# eval-friendly: every line is a `export FOO=bar` statement.
print_env() {
  # Reuse the same lookup logic as install: prefer an in-PATH Blender
  # of the right version, fall back to our managed install.
  if [ "$PLATFORM" = linux ]; then
    detect_existing_blender >/dev/null 2>&1 || true
  fi
  if [ ! -x "$BLENDER_BIN" ]; then
    echo "echo 'install-blender.sh: Blender not installed at $BLENDER_BIN' >&2; exit 1"
    return
  fi
  echo "export PPF_BLENDER_BIN=$BLENDER_BIN"
  if [ "$PLATFORM" = linux ]; then
    if ! pgrep -x Xvfb >/dev/null 2>&1; then
      # Spawn detached so the eval'ing shell doesn't inherit a child.
      # Output goes to /tmp/xvfb.log per the CI convention.
      echo "(Xvfb :99 -screen 0 1280x720x24 -ac >/tmp/xvfb.log 2>&1 &)"
      echo "for _ in 1 2 3 4 5; do xset -q -display :99 >/dev/null 2>&1 && break || sleep 1; done"
    fi
    echo "export DISPLAY=:99"
  fi
}

case "$ACTION" in
install)
  case "$PLATFORM" in
    linux) linux_install ;;
    macos) macos_install ;;
  esac
  ;;
uninstall)
  case "$PLATFORM" in
    linux) linux_uninstall ;;
    macos) macos_uninstall ;;
  esac
  ;;
env)
  print_env
  ;;
esac
