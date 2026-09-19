#!/usr/bin/env bash
# File: build-mac-native/warmup.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# One-time provisioning for a native macOS build of the solver, the
# counterpart of build-win-native/warmup.bat. Run it once per machine, then
# ./build.sh.
#
# It is safe to run twice: every step checks for what it would install and
# skips the install while still verifying the result, so a second run is a
# check rather than a reinstall.
#
# WHAT IT PROVISIONS
#   - Rust, into build-mac-native/rust, and only when the host has no cargo.
#   - A Python environment carrying the frontend dependencies, built from the
#     canonical list in the repository's own warmup.py so this file cannot
#     drift from it. It builds that environment with a Python the host already
#     has, and does not download one for it.
#   - A relocatable CPython in build-mac-native/python, carrying that same
#     package list, for bundle.sh to ship inside the distribution so a user who
#     downloads a release needs no Python of their own. It is a second install
#     target for one list rather than a replacement for the environment above,
#     which has its own location and is what build.sh and the Metal fixture
#     target use.
#
# WHAT IT DELIBERATELY DOES NOT PROVISION
#   - The Metal Toolchain component, which supplies `xcrun metal`. build.sh
#     pre-compiles the shader libraries with it when a machine has it and
#     continues without them when it does not, so it is an optional speed-up
#     rather than a build requirement, and downloading it here would make it
#     one for everybody. The Xcode command line tools carry the macOS SDK,
#     whose Metal.framework is the only thing the backend links.
#   - A C++ toolchain, git or curl. The command line tools carry all three.
#   - ffmpeg. See scripts/downloads.txt for why there is no pointer here.

set -euo pipefail

BUILD_MAC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$BUILD_MAC/.." && pwd)"
LOGFILE="$BUILD_MAC/warmup.log"
DOWNLOADS="$BUILD_MAC/downloads"

# Re-run under tee so the whole provision is logged. The exit status has to be
# read out of PIPESTATUS: the status of a pipeline is its LAST command's, which
# here is tee's, and tee succeeds whatever the script did, so without this
# every failure would report success to the caller.
if [ -z "${PPF_MAC_WARMUP_LOGGING:-}" ]; then
    printf 'Logging to %s\n' "$LOGFILE"
    set +e
    PPF_MAC_WARMUP_LOGGING=1 "${BASH_SOURCE[0]}" "$@" 2>&1 | tee "$LOGFILE"
    rc=${PIPESTATUS[0]}
    set -e
    exit "$rc"
fi

die() {
    printf 'ERROR: %s\n' "$1" >&2
    shift
    for line in "$@"; do
        printf '       %s\n' "$line" >&2
    done
    exit 1
}

step() { printf '\n=== %s ===\n' "$1"; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die \
        "required tool not found on PATH: $1" \
        "$2"
}

CLT_HINT="Install the Xcode command line tools:  xcode-select --install"

printf "=== ZOZO's Contact Solver, native macOS environment setup ===\n"
printf 'Build directory:  %s\n' "$BUILD_MAC"
printf 'Source directory: %s\n' "$SRC_DIR"
printf 'Log file:         %s\n' "$LOGFILE"

# ---------------------------------------------------------------------------
# Host preconditions
# ---------------------------------------------------------------------------
step "Checking the host"

[ "$(uname -s)" = "Darwin" ] || die \
    "this script provisions a macOS host, and uname reports $(uname -s)" \
    "On Linux or in the container use warmup.py at the repository root." \
    "On Windows use build-win-native/warmup.bat."

# Apple silicon only. The Metal backend links -arch arm64
# (crates/ppf-cts-compute/metal/Makefile) and the port scopes Intel
# Macs out, so an x86_64 host would provision a toolchain that cannot produce
# a runnable solver.
[ "$(uname -m)" = "arm64" ] || die \
    "this build targets Apple silicon and uname -m reports $(uname -m)" \
    "Under Rosetta a native shell reports arm64; if this is an Intel Mac," \
    "there is no supported build."

require_cmd curl "$CLT_HINT"
require_cmd git "$CLT_HINT"
require_cmd make "$CLT_HINT"
require_cmd clang++ "$CLT_HINT"
require_cmd xcrun "$CLT_HINT"
require_cmd rsync "rsync ships with macOS; a host without it has a broken base system."
# Both belong to the base system rather than to the command line tools, so the
# hint that fits them is rsync's rather than CLT_HINT: installing the tools
# would not supply either one.
require_cmd tar "tar ships with macOS; a host without it has a broken base system."
require_cmd shasum "shasum ships with macOS; a host without it has a broken base system."

xcode-select -p >/dev/null 2>&1 || die \
    "xcode-select reports no developer directory" \
    "$CLT_HINT"

SDK_PATH="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
[ -n "$SDK_PATH" ] || die \
    "xcrun could not report a macOS SDK path" \
    "$CLT_HINT"
[ -d "$SDK_PATH/System/Library/Frameworks/Metal.framework" ] || die \
    "the macOS SDK at $SDK_PATH carries no Metal.framework" \
    "The Metal backend links that framework, and crates/ppf-cts-solver/build.rs" \
    "makes the same check, so a build would fail here anyway." \
    "$CLT_HINT"

printf 'macOS %s on %s\n' "$(sw_vers -productVersion)" "$(uname -m)"
printf 'SDK: %s\n' "$SDK_PATH"
printf 'Metal.framework: present\n'

# Report the shader toolchain question rather than leaving a reader to wonder
# whether its absence is a problem. It is not: the backend never invokes it.
if xcrun --find metal >/dev/null 2>&1; then
    printf 'xcrun metal: present, and unused. Shaders compile at run time.\n'
else
    printf 'xcrun metal: absent, which is expected. Shaders compile at run time\n'
    printf '             through newLibraryWithSource, so no Metal Toolchain\n'
    printf '             component is provisioned or needed.\n'
fi

# ---------------------------------------------------------------------------
# Download preflight: every pointer, before the first fetch
# ---------------------------------------------------------------------------
step "Checking download pointers"

"$BUILD_MAC/scripts/check-downloads.sh" || die \
    "one or more download URLs are unreachable" \
    "Fix the offending pointer in build-mac-native/scripts/downloads.txt and re-run." \
    "This check runs before any download precisely so a rotted pointer costs" \
    "seconds instead of a whole provision."

# shellcheck source=build-mac-native/scripts/load-downloads.sh
. "$BUILD_MAC/scripts/load-downloads.sh"
load_downloads "$BUILD_MAC/scripts/downloads.txt"

# shellcheck source=build-mac-native/config.sh
. "$BUILD_MAC/config.sh"

# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------
step "Rust"

RUST_DIR="$BUILD_MAC/rust"
CARGO=""
if command -v cargo >/dev/null 2>&1; then
    CARGO="$(command -v cargo)"
    printf 'cargo already on PATH: %s\n' "$CARGO"
elif [ -x "$HOME/.cargo/bin/cargo" ]; then
    CARGO="$HOME/.cargo/bin/cargo"
    printf 'cargo found at %s (not on PATH; build.sh adds it)\n' "$CARGO"
elif [ -x "$RUST_DIR/bin/cargo" ]; then
    CARGO="$RUST_DIR/bin/cargo"
    printf 'cargo found in this directory: %s\n' "$CARGO"
else
    printf 'No cargo on this host. Installing rustup into %s\n' "$RUST_DIR"
    mkdir -p "$DOWNLOADS"
    RUSTUP_INIT="$DOWNLOADS/$FILE_RUSTUP"
    if [ ! -f "$RUSTUP_INIT" ]; then
        if ! curl -fL --retry 2 --retry-delay 1 --connect-timeout 8 \
            -o "$RUSTUP_INIT" "$URL_RUSTUP"; then
            # curl -o leaves whatever arrived in place, and the existence test
            # above would read that partial file as a finished download on the
            # next run and hand it to sh below. A truncated shell script does
            # not fail to parse: it runs the part that arrived. There is no
            # checksum for this entry either, and scripts/downloads.txt says
            # why it carries none, so deleting the partial here is the whole of
            # the protection.
            rm -f "$RUSTUP_INIT"
            die "failed to download rustup from $URL_RUSTUP"
        fi
    fi
    # --no-modify-path plus the two HOME overrides keep the install inside
    # this directory: nothing is written to ~/.cargo, ~/.rustup or any shell
    # profile, so removing build-mac-native/rust removes the toolchain.
    CARGO_HOME="$RUST_DIR" RUSTUP_HOME="$RUST_DIR/rustup" \
        sh "$RUSTUP_INIT" -y --no-modify-path --profile minimal \
        --default-toolchain stable || die "rustup install failed"
    CARGO="$RUST_DIR/bin/cargo"
    [ -x "$CARGO" ] || die \
        "rustup reported success but there is no cargo at $CARGO"
fi

# Assert it runs. An installer that exits 0 and leaves an unusable toolchain
# is a failure this script must surface, not one build.sh discovers.
if [ "$CARGO" = "$RUST_DIR/bin/cargo" ]; then
    CARGO_HOME="$RUST_DIR" RUSTUP_HOME="$RUST_DIR/rustup" "$CARGO" --version \
        || die "$CARGO does not run"
else
    "$CARGO" --version || die "$CARGO does not run"
fi

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
step "Python interpreter"

PY=""
if [ -n "${PPF_CTS_PYTHON:-}" ]; then
    [ -x "$PPF_CTS_PYTHON" ] || die \
        "PPF_CTS_PYTHON is set to $PPF_CTS_PYTHON, which is not executable"
    PY="$PPF_CTS_PYTHON"
else
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
                >/dev/null 2>&1; then
                PY="$(command -v "$candidate")"
                break
            fi
        fi
    done
fi

[ -n "$PY" ] || die \
    "no Python 3.10 or newer found on PATH" \
    "The frontend uses PEP 604 unions, which 3.9 does not parse, and macOS" \
    "ships 3.9. Install one of these and re-run, or point PPF_CTS_PYTHON at" \
    "an interpreter you already have (see config.sh):" \
    "  brew install python@3.11" \
    "  uv python install 3.11   (then PPF_CTS_PYTHON=\"\$(uv python find 3.11)\")" \
    "  the python.org macOS installer" \
    "This script does not download an interpreter: it would be one more" \
    "pinned pointer to keep current, and macOS has several supported ways to" \
    "get one already."

printf 'Using %s (%s)\n' "$PY" "$("$PY" -c 'import platform; print(platform.python_version())')"

step "Frontend environment"

# The venv location comes from warmup.py's own get_venv_path() unless config.sh
# overrides it, so this file does not carry a second copy of that path. It is
# also what the Metal fixture target defaults FIXTURE_PYTHON to.
if [ -z "${PPF_CTS_VENV:-}" ]; then
    PPF_CTS_VENV="$("$PY" - "$SRC_DIR/warmup.py" <<'PY'
import runpy
import sys

ns = runpy.run_path(sys.argv[1])
print(ns["get_venv_path"]())
PY
)"
fi
[ -n "$PPF_CTS_VENV" ] || die "could not determine a venv path"
printf 'Environment: %s\n' "$PPF_CTS_VENV"

if [ ! -x "$PPF_CTS_VENV/bin/python" ]; then
    if [ -e "$PPF_CTS_VENV" ]; then
        die "$PPF_CTS_VENV exists but holds no bin/python" \
            "Remove it and re-run, or point PPF_CTS_VENV somewhere else."
    fi
    printf 'Creating the environment...\n'
    "$PY" -m venv "$PPF_CTS_VENV" || die "python -m venv failed"
fi
VENV_PY="$PPF_CTS_VENV/bin/python"
[ -x "$VENV_PY" ] || die "no interpreter at $VENV_PY after venv creation"

# The dependency set is read out of warmup.py rather than restated here. That
# file is the canonical list for every provisioning path, and it explains at
# each entry why the frontend needs it; a copy in this script is a second list
# that can silently disagree with it.
PACKAGES=()
while IFS= read -r pkg; do
    if [ -n "$pkg" ]; then
        PACKAGES+=("$pkg")
    fi
done < <("$PY" - "$SRC_DIR/warmup.py" <<'PY'
import runpy
import sys

ns = runpy.run_path(sys.argv[1])
for name in list(ns["python_packages"]()) + list(ns["tetra_packages"]()):
    print(name)
PY
)

[ "${#PACKAGES[@]}" -gt 0 ] || die \
    "warmup.py yielded no package list" \
    "python_packages() and tetra_packages() are what this reads; if either" \
    "was renamed, update this script in the same change."

printf 'Installing %d packages from warmup.py...\n' "${#PACKAGES[@]}"
"$VENV_PY" -m pip install --upgrade pip || die "pip upgrade failed"
"$VENV_PY" -m pip install "${PACKAGES[@]}" || die \
    "pip install failed" \
    "Re-run on a working network and package index. Nothing downstream can" \
    "work around a missing frontend dependency: a build worker without scipy" \
    "silently takes a different pin-diffusion path, and one without a" \
    "tetrahedralizer cannot build a SOLID scene at all."

# Verify against warmup.py's REQUIRED_PACKAGES, in the environment that was
# just built. pip can report success overall and still leave a package out.
"$VENV_PY" - "$SRC_DIR/warmup.py" <<'PY' || die "the frontend environment is incomplete"
import importlib.util as util
import runpy
import sys

ns = runpy.run_path(sys.argv[1])
required = list(ns["REQUIRED_PACKAGES"])
missing = [name for name in required if util.find_spec(name) is None]
if missing:
    sys.stderr.write("missing required packages: " + ", ".join(missing) + "\n")
    sys.exit(1)
print("verified %d required packages" % len(required))
PY

# ---------------------------------------------------------------------------
# The interpreter bundle.sh ships inside the distribution
# ---------------------------------------------------------------------------
step "Bundled interpreter"

# This is what makes a downloaded release runnable on a Mac carrying no Python
# at all: bundle.sh copies this tree into the distribution and the launcher
# runs it. It
# is a second install target for the package list the environment above was
# built from, not a replacement for that environment, which has its own
# location and is what build.sh's start.sh and the Metal fixture target use.
#
# Packages install INTO the extracted tree rather than into a venv over it. A
# venv records an absolute path in its pyvenv.cfg and its bin/python names the
# base interpreter by absolute path, so a venv does not survive being copied
# anywhere; a python-build-standalone install_only tree is relocatable as an
# installation, and only as one.

PY_ROOT="$BUILD_MAC/python"
# bin/python3 is the canonical entry point of such a tree. It is a relative
# symlink to the versioned binary, so it survives every copy the build makes,
# and naming it costs no knowledge of the minor version. bin/python may not
# exist at all.
PY_BUNDLED="$PY_ROOT/bin/python3"

# The version is read out of the pinned filename rather than written down a
# second time. FILE_PYTHON is cpython-<version>+<build>-<triple>.tar.gz, so
# dropping the prefix and everything from the + leaves the version alone.
PY_BUNDLED_VERSION="${FILE_PYTHON#cpython-}"
PY_BUNDLED_VERSION="${PY_BUNDLED_VERSION%%+*}"
case "$PY_BUNDLED_VERSION" in
    '' | *[!0-9.]*)
        die "could not read a version out of FILE_PYTHON=$FILE_PYTHON" \
            "This step expects cpython-<version>+<build>-<triple>.tar.gz, the" \
            "name python-build-standalone publishes. If the pointer in" \
            "scripts/downloads.txt now names something else, update this" \
            "parse in the same change rather than carrying the version twice."
        ;;
esac

# Verifying a tree that is already here and verifying one just installed ask
# the same question, so both call this.
verify_bundled_python() {
    local jlab_version
    "$PY_BUNDLED" - "$SRC_DIR/warmup.py" <<'PY' || return 1
import importlib.util as util
import runpy
import sys

ns = runpy.run_path(sys.argv[1])
required = list(ns["REQUIRED_PACKAGES"])
# jupyterlab is asked for by name because REQUIRED_PACKAGES does not carry it.
# That tuple is what a host cannot run a solve without, and jupyterlab is not
# one of those; for the interpreter that ships in the distribution it is the whole
# point, the launcher starting it with -m jupyterlab. warmup.py's
# python_packages() does install it, so this checks the install rather than
# adding a package to it.
required.append("jupyterlab")
missing = [name for name in required if util.find_spec(name) is None]
if missing:
    sys.stderr.write("missing packages: " + ", ".join(missing) + "\n")
    sys.exit(1)
print("verified %d packages, jupyterlab among them" % len(required))
PY

    # Then ask for the entry point the distribution itself uses. The launcher starts the
    # server with `python -m jupyterlab`, which resolves jupyterlab/__main__.py
    # and imports the whole server stack behind it, and the find_spec above
    # answers for neither: a package can be importable with no __main__, and a
    # server dependency that is present but broken surfaces only when something
    # imports it. Running it here costs about a second on the developer's
    # machine, where a failure has a terminal to print to. The same failure
    # inside a shipped distribution is a launcher that starts nothing.
    jlab_version="$("$PY_BUNDLED" -m jupyterlab --version)" || return 1
    printf 'python -m jupyterlab reports %s\n' "$jlab_version"
}

if [ "${PPF_MAC_PYTHON:-1}" = "0" ]; then
    PY_BUNDLED_STATUS="skipped this run, PPF_MAC_PYTHON=0"
    printf 'PPF_MAC_PYTHON=0, so no interpreter is provisioned for the distribution.\n'
    printf 'That skips a 24 MB download for someone who builds and runs from\n'
    printf 'this tree and never bundles. bundle.sh REFUSES to build a distribution when\n'
    printf 'it finds no interpreter here, and names this variable when it does:\n'
    printf 'an app that quietly fell back to whatever Python the user happens to\n'
    printf 'have is the failure a shipped interpreter exists to remove.\n'
elif [ -x "$PY_BUNDLED" ]; then
    # Verify rather than reinstall, which is the contract every other step in
    # this file keeps. Nothing here wipes the tree and starts over: a
    # half-provisioned tree quietly replaced hides what went wrong with it, so
    # any failed check stops the run and names the repair.
    printf 'Found %s. Verifying rather than reinstalling.\n' "$PY_BUNDLED"
    PY_REPAIR="rm -rf $PY_ROOT && $BUILD_MAC/warmup.sh"

    PY_HAVE="$("$PY_BUNDLED" -c 'import platform; print(platform.python_version())')" || die \
        "the interpreter at $PY_BUNDLED does not run" \
        "Repair with:  $PY_REPAIR"
    [ "$PY_HAVE" = "$PY_BUNDLED_VERSION" ] || die \
        "$PY_BUNDLED reports Python $PY_HAVE and the pin names $PY_BUNDLED_VERSION" \
        "This tree was provisioned from a different pointer than the one" \
        "scripts/downloads.txt carries now." \
        "Repair with:  $PY_REPAIR"
    verify_bundled_python || die \
        "the bundled interpreter is missing packages the distribution needs" \
        "Repair with:  $PY_REPAIR"

    PY_BUNDLED_STATUS="verified at $PY_ROOT"
    printf 'Verified Python %s at %s\n' "$PY_HAVE" "$PY_ROOT"
else
    # A directory with no runnable bin/python3 is a half-extracted or
    # hand-edited tree rather than a provisioned one, and the mv below would
    # move a fresh extraction INSIDE it and nest one interpreter in another.
    # Say so instead, which is how the venv step above treats the same shape.
    if [ -e "$PY_ROOT" ]; then
        die "$PY_ROOT exists and carries no runnable bin/python3" \
            "Remove it and re-run:  rm -rf $PY_ROOT"
    fi

    printf 'Provisioning %s\n' "$PY_ROOT"
    # What a cached download does and does not buy, said plainly because it is
    # easy to expect more of it: check-downloads.sh has already run by now and
    # needs the network, so a fully offline machine cannot run this script at
    # all. What the cache serves is a machine that reaches the network for that
    # preflight and has had this one file placed in downloads/ by hand, which
    # is how the file arrives on a fleet with no egress of its own.
    mkdir -p "$DOWNLOADS"
    PY_TARBALL="$DOWNLOADS/$FILE_PYTHON"
    if [ -f "$PY_TARBALL" ]; then
        printf 'Using the cached download at %s\n' "$PY_TARBALL"
    else
        printf 'Downloading %s\n' "$FILE_PYTHON"
        if ! curl -fL --retry 2 --retry-delay 1 --connect-timeout 8 \
            -o "$PY_TARBALL" "$URL_PYTHON"; then
            # curl -o leaves whatever arrived in place, and a later run would
            # read that partial file as a hand-relayed copy. The checksum below
            # would still catch it; removing it here keeps the failure at the
            # step that produced it.
            rm -f "$PY_TARBALL"
            die "failed to download the interpreter from $URL_PYTHON" \
                "On a machine that cannot reach that host, copy the file to" \
                "$PY_TARBALL by hand and re-run: this step uses a cached" \
                "download when it finds one."
        fi
    fi

    # Checked on every run that provisions from the archive, whether this run
    # downloaded the file or found a relayed copy already in downloads/. A
    # later run finds the provisioned tree at the top of this step and verifies
    # that instead, reading the archive not at all. The relayed copy is the
    # case this exists for: a truncated transfer extracts into a tree that
    # mostly works, and the user of a shipped distribution has no way to repair what it
    # left out.
    printf 'Verifying the checksum...\n'
    PY_SUM="$(shasum -a 256 "$PY_TARBALL")" || die \
        "shasum could not read $PY_TARBALL"
    PY_SUM="${PY_SUM%% *}"
    [ "$PY_SUM" = "$SHA256_PYTHON" ] || die \
        "checksum mismatch on $PY_TARBALL" \
        "expected  $SHA256_PYTHON" \
        "measured  $PY_SUM" \
        "Delete the file and re-run so it is fetched again:" \
        "  rm -f $PY_TARBALL"

    # Extract into a scratch directory and move the result into place.
    # Extracting straight into build-mac-native would merge a partial tree into
    # whatever is already there and leave something that looks provisioned. The
    # name is dot-prefixed because .gitignore already ignores
    # build-mac-native/.*
    PY_EXTRACT="$BUILD_MAC/.python-extract"
    printf 'Extracting...\n'
    rm -rf "$PY_EXTRACT"
    mkdir -p "$PY_EXTRACT"
    tar -xzf "$PY_TARBALL" -C "$PY_EXTRACT" || die \
        "tar could not extract $PY_TARBALL" \
        "Delete the file and re-run so it is fetched again:" \
        "  rm -f $PY_TARBALL"
    [ -x "$PY_EXTRACT/python/bin/python3" ] || die \
        "the archive carries no python/bin/python3" \
        "It extracted: $(ls "$PY_EXTRACT" | tr '\n' ' ')" \
        "This step expects a python-build-standalone install_only build, whose" \
        "archive holds a single top-level directory named python."
    mv "$PY_EXTRACT/python" "$PY_ROOT"
    rmdir "$PY_EXTRACT"

    PY_HAVE="$("$PY_BUNDLED" -c 'import platform; print(platform.python_version())')" || die \
        "the extracted interpreter at $PY_BUNDLED does not run"
    [ "$PY_HAVE" = "$PY_BUNDLED_VERSION" ] || die \
        "the extracted interpreter reports Python $PY_HAVE, and $FILE_PYTHON" \
        "names $PY_BUNDLED_VERSION"
    printf 'Python %s at %s\n' "$PY_HAVE" "$PY_ROOT"

    if ! "$PY_BUNDLED" -m pip --version >/dev/null 2>&1; then
        # ensurepip installs from a wheel inside the standard library, so it
        # needs no network of its own.
        printf 'No pip in the extracted tree. Running ensurepip...\n'
        "$PY_BUNDLED" -m ensurepip --upgrade || die \
            "ensurepip failed in $PY_ROOT"
        "$PY_BUNDLED" -m pip --version >/dev/null 2>&1 || die \
            "ensurepip reported success and $PY_ROOT still carries no pip"
    fi

    # PIP STAMPS THIS BUILD TREE'S PATH INTO EVERY CONSOLE SCRIPT IT WRITES.
    # Both commands below generate scripts in $PY_ROOT/bin, the upgrade
    # rewriting pip's own and the install adding jupyter, jupyter-lab,
    # pygmentize and the rest, and each one carries the absolute path of the
    # interpreter it was installed for as its first line, which is a path under
    # this tree.
    #
    # Nothing here rewrites them, deliberately. This tree is the developer's
    # and is run from where it sits, so the paths are correct for it; rewriting
    # them would leave the tree that ships differing from the tree these checks
    # verified. bundle.sh is what neutralizes them in the COPY it places inside
    # the distribution, and its build-tree scan (gate C) is what fails the build if that
    # step is ever dropped. Do not answer such a failure by relaxing the gate:
    # a shebang naming this machine is dead on a user's Mac, and it could not
    # work there in any case, because the distribution path carries a space and the
    # kernel splits a shebang line on whitespace.
    #
    # Nothing inside the distribution invokes a console script. The launcher starts
    # the server with `python -m jupyterlab`, and jupyter_client rewrites the
    # kernelspec's argv[0] to sys.executable before it launches a kernel.
    printf 'Installing %d packages from warmup.py...\n' "${#PACKAGES[@]}"
    "$PY_BUNDLED" -m pip install --upgrade pip || die \
        "pip upgrade failed for the bundled interpreter"
    "$PY_BUNDLED" -m pip install "${PACKAGES[@]}" || die \
        "pip install failed for the bundled interpreter" \
        "Re-run on a working network and package index. Nothing downstream can" \
        "work around a missing frontend dependency: a build worker without" \
        "scipy silently takes a different pin-diffusion path, and one without" \
        "a tetrahedralizer cannot build a SOLID scene at all. This tree ships" \
        "to a user who cannot repair it."

    verify_bundled_python || die \
        "the bundled interpreter is incomplete after the install" \
        "pip can report success overall and still leave a package out, and" \
        "whatever this one is missing ships inside the distribution."

    PY_BUNDLED_STATUS="installed at $PY_ROOT"
fi

# ---------------------------------------------------------------------------
# ffmpeg, which is optional and is not provisioned here
# ---------------------------------------------------------------------------
step "ffmpeg (optional)"

if command -v ffmpeg >/dev/null 2>&1; then
    printf 'ffmpeg on PATH: %s\n' "$(command -v ffmpeg)"
else
    printf 'No ffmpeg on PATH. Video export from a session is skipped without\n'
    printf 'one (frontend/_session_inspect_.py falls back to shutil.which and\n'
    printf 'returns early); everything else works. This directory provisions\n'
    printf 'none, and if it ever does it takes the pinned pointers from\n'
    printf 'build-win-native/scripts/downloads.txt rather than adding a second\n'
    printf 'copy. See build-mac-native/scripts/downloads.txt.\n'
fi

# ---------------------------------------------------------------------------
step "Setup complete"
printf 'Frontend environment: %s\n' "$PPF_CTS_VENV"
printf 'Bundled interpreter:  %s\n' "$PY_BUNDLED_STATUS"
printf 'Next: %s/build.sh\n' "$BUILD_MAC"
