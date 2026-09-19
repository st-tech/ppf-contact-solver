#!/usr/bin/env bash
# File: build-mac-native/bundle.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Packages a built tree into a plain macOS distribution directory under
# build-mac-native/dist, the counterpart of build-win-native/bundle.bat. Run
# warmup.sh and build.sh first.
#
# It is safe to run twice: dist is removed and rebuilt from the current build.
#
# WHAT IT PRODUCES
#   dist/<name>/                one directory, holding a launcher and its
#                               payload. <name> is `ppf-contact-solver` unless
#                               PPF_MAC_DIST_NAME says otherwise, which the
#                               release workflow sets to the archive's stem so
#                               an unzipped release carries its version
#                               payload. Nothing else lands in dist: no
#                               archive, no disk image, no installer. A person
#                               who receives this directory opens Terminal,
#                               changes into it, and runs ./ppf-contact-solver.
#
# THE LAYOUT, AND WHY THE DIRECTORY IS THE TREE ROOT
#   ppf-contact-solver          the launcher, a bash script, which starts
#                               JupyterLab in the foreground
#   config.sh                   the port, and the developer override
#   README.txt                  what a person needs to run it
#   THIRD_PARTY_LICENSES.txt
#   bin/                        libppfbe_metal.dylib, the generated entry
#                               library ppf_entries.metallib that the backend
#                               refuses to open without, and any pre-compiled
#                               shader cache libraries
#   python/                     a relocatable CPython carrying the frontend
#                               dependencies and jupyterlab, provisioned by
#                               warmup.sh
#   frontend/  examples/  crates/
#   target/release/             ppf-contact-solver, ppf-cts-server and
#                               lib_ppf_cts_py.dylib
#
#   THE DISTRIBUTION DIRECTORY IS THE TREE ROOT. frontend/__init__.py resolves
#   the root from its own __file__ as dirname(__file__)/.., so this directory
#   is what it computes and <root>/target/release/ is where it finds the
#   cdylib. Every path this script writes is relative to that one directory.
#
#   TWO FILES IN IT CARRY THE BASENAME ppf-contact-solver, AT DIFFERENT PATHS.
#   The one at the top is the launcher, a shell script, and is what a person
#   runs. target/release/ppf-contact-solver is the solver, a Mach-O the
#   frontend runs per scene. Nothing in the launcher runs the solver directly.
#
# WHAT IT SHIPS AND WHAT IT DOES NOT
#   It ships an interpreter, so a machine running this needs no Python of its
#   own. It does not ship ffmpeg, which nothing here builds and which the
#   frontend degrades to skipping video export without, and it does not ship
#   git.
#
#   NOTHING THE LAUNCHER ITSELF DOES RUNS git, AND THAT IS LOAD BEARING RATHER
#   THAN INCIDENTAL. On a Mac without the command line tools /usr/bin/git is
#   the developer-tools shim, so running it presents an "install the command
#   line developer tools" dialog, which is precisely what this distribution
#   promises never to do. The one code path that would reach it is
#   data_dirpath_for in crates/ppf-cts-core/src/datamodel/app.rs, which every
#   App.create() reaches through frontend/_app_.py, and step 4 stamps the
#   branch file it reads first so that it never falls through to the
#   subprocess.
#
#   Four shipped example notebooks are a separate matter: large-fluffy,
#   large-animals, fitting and trapped-919539a call sparse_clone,
#   which does run git, and several examples fetch preset meshes over the
#   network into cache/ppf-cts inside the distribution. That is a notebook the user chose to open,
#   not the launcher, and README.txt draws that distinction rather than
#   claiming the payload never reaches the network.
#
# THE DISTRIBUTION NEVER TOUCHES THE USER'S macOS
#   Nothing here installs a library or a tool, writes outside the user's home,
#   registers a LaunchAgent or a login item, edits a PATH or a shell profile,
#   or downloads anything when the launcher starts. Removing it is deleting
#   the directory, and that is now literally true: sessions, the asset cache,
#   the Jupyter and IPython state and the notebooks are all under this folder,
#   which is what the `.ppf-selfcontained` marker written in step 3 tells the
#   frontend. The signed payload itself is still never written to: step 9
#   pre-compiles every .pyc at build time and the launcher exports
#   PYTHONDONTWRITEBYTECODE, so what the program creates are new directories
#   beside the signed files rather than edits to them.
#
# SIGNING IS THE LAST THING THAT TOUCHES THE CONTENTS
#   Every write to a signed Mach-O invalidates its signature, so the order
#   here is generate, prune, thin, verify, smoke test, and only then sign. A
#   payload that fails a gate therefore costs seconds rather than the minute
#   or two the signing loop takes, and nothing that was signed can have
#   changed afterwards.
#
#   Every Mach-O file is signed, one at a time, ad-hoc by default, because
#   rewriting a load command invalidates the signature and an arm64 binary
#   with an invalid signature does not run at all. An ad-hoc signature is
#   enough for the machine that built it and is rejected by Gatekeeper
#   anywhere else. Set MAC_CODESIGN_IDENTITY in config.sh to sign with a real
#   identity.
#
#   WHAT CARRIES NO SIGNATURE, STATED BECAUSE THE OMISSION LOOKS LIKE AN
#   OVERSIGHT: the .metallib files, the Python sources, README.txt,
#   config.sh and the launcher itself. A signature on a file that is not a
#   Mach-O is stored in an extended attribute rather than in the file, nothing
#   on the running path verifies it, and an extended attribute does not
#   survive every way a person may unpack an archive. What makes the arm64
#   binaries loadable is their own embedded signature, and that is what the
#   loop in step 12 puts there. This is a property of a plain directory rather
#   than a defect to route around: do not add a checksum manifest to
#   compensate, and do not sign anything a loader will not verify.
#
#   THIS SCRIPT DOES NOT NOTARIZE. The LAUNCHER it generates clears the
#   quarantine mark on its own folder at startup, which is what makes a
#   browser download runnable without notarization; this script clears
#   nothing, because what it packages was never marked. It says at the end
#   what the missing notarization still costs.
#
# THINNING TO arm64
#   This distribution is arm64 by construction. A handful of wheels in the
#   interpreter tree ship universal Mach-O files carrying an x86_64 slice that
#   cannot run here, so step 7 thins them and then asserts that no Mach-O in
#   the payload reports any other architecture.
#
# SWITCHES
#   PPF_MAC_PYTHON    warmup.sh's switch, not this script's. A tree
#                     provisioned with PPF_MAC_PYTHON=0 has no interpreter to
#                     ship and step 1 refuses rather than producing a
#                     distribution that quietly needs the user's own Python.
set -euo pipefail

# THE C LOCALE IS A CORRECTNESS SETTING HERE, NOT A PREFERENCE. This script
# runs sed and grep over the payload, and the payload is mostly not text: the
# interpreter tree carries test fixtures whose names and contents are not valid
# UTF-8. Under a UTF-8 locale sed answers such a byte with "RE error: illegal
# byte sequence" and EXITS, which closes the pipe feeding it and truncates
# whatever it was filtering. That is how a Mach-O discovery silently returns a
# short list, so the gates walk part of the payload and the signing step signs
# part of it, both reporting success. In the C locale a byte is a byte and
# nothing aborts.
export LC_ALL=C

BUILD_MAC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$BUILD_MAC/.." && pwd)"
LOGFILE="$BUILD_MAC/bundle.log"

if [ -z "${PPF_MAC_BUNDLE_LOGGING:-}" ]; then
    printf 'Logging to %s\n' "$LOGFILE"
    set +e
    PPF_MAC_BUNDLE_LOGGING=1 "${BASH_SOURCE[0]}" "$@" 2>&1 | tee "$LOGFILE"
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

step() { printf '\n============================================================\n%s\n============================================================\n\n' "$1"; }

[ "$(uname -s)" = "Darwin" ] || die "this packages a macOS build, and uname reports $(uname -s)"
[ "$(uname -m)" = "arm64" ] || die "this build targets Apple silicon and uname -m reports $(uname -m)"
# file, xargs and strings find and classify the Mach-O files the gates walk;
# lipo reads and thins their architectures in step 7; stat records a file's
# size and mode across that rewrite; plutil validates the entitlements
# generated in step 12.
for tool in otool install_name_tool codesign rsync file xargs strings lipo stat plutil; do
    command -v "$tool" >/dev/null 2>&1 || die \
        "required tool not found on PATH: $tool" \
        "Install the Xcode command line tools:  xcode-select --install"
done

# shellcheck source=build-mac-native/config.sh
. "$BUILD_MAC/config.sh"

# The manifest, for the one value this script needs out of it: the interpreter
# archive's filename, which THIRD_PARTY_LICENSES.txt names so a reader can see
# exactly which build is redistributed. No URL, filename or version is spelled
# a second time here.
# shellcheck source=build-mac-native/scripts/load-downloads.sh
. "$BUILD_MAC/scripts/load-downloads.sh"
load_downloads "$BUILD_MAC/scripts/downloads.txt" || die \
    "could not read build-mac-native/scripts/downloads.txt"

DIST="$BUILD_MAC/dist"
TARGET_REL="$SRC_DIR/target/release"
PY_ROOT="$BUILD_MAC/python"

# The distribution directory, which is both what ships and the tree root the
# frontend computes. No path this script builds carries a space. Every
# expansion of these is quoted anyway, without exception, and every file walk
# below is -print0 or `while IFS= read -r`, because a USER's path may hold one:
# the launcher resolves its own directory at run time from wherever the person
# who received it put the folder.
#
# THE DIRECTORY'S NAME IS WHAT A PERSON SEES AFTER UNZIPPING, so a release
# stamps it and a local build does not. `ditto --keepParent` puts this name at
# the top of the archive, so an unstamped one expands to `ppf-contact-solver`
# every time and each download lands on the previous one; the release workflow
# therefore passes the archive's own stem here, and the folder and the zip then
# carry the same name by construction rather than by two places agreeing.
#
# The default stays unstamped on purpose. A developer runs this repeatedly and
# `dist/ppf-contact-solver` is the path their shell history, their notes and
# this README all name; a timestamped one would change under them every build
# for no gain, since nothing local unzips anything.
PPF_MAC_DIST_NAME="${PPF_MAC_DIST_NAME:-ppf-contact-solver}"
case "$PPF_MAC_DIST_NAME" in
    "" | */* | .* )
        die "PPF_MAC_DIST_NAME is not a usable directory name" \
            "  got: ${PPF_MAC_DIST_NAME:-<empty>}" \
            "It names ONE directory directly under dist, so it cannot be" \
            "empty, cannot contain a slash, and cannot begin with a dot. The" \
            "release workflow passes the archive's stem, for example" \
            "ppf-contact-solver-2026-09-08-17-47-macos-arm64."
        ;;
esac
PKG="$DIST/$PPF_MAC_DIST_NAME"
PKG_PY="$PKG/python/bin/python3"

printf '============================================================\n'
printf "  ZOZO's Contact Solver, macOS distribution\n"
printf '============================================================\n'

# ---------------------------------------------------------------------------
step "[1/13] Verifying the build"
# ---------------------------------------------------------------------------

SOLVER_BIN="$TARGET_REL/ppf-contact-solver"
SERVER_BIN="$TARGET_REL/ppf-cts-server"
PY_DYLIB="$TARGET_REL/lib_ppf_cts_py.dylib"

# THE SECOND BACKEND. `build.sh` builds the portable Rust CPU solver into its
# own target directory, because every backend links the same executable name
# and `crates/ppf-cts-solver/build.rs` refuses to put two in one directory.
# That layout is not an internal detail of the build: it is the layout the
# add-on's GPU/CPU selector probes and the one `frontend.solver_dir` resolves,
# so it is reproduced inside the distribution rather than flattened.
#
# ALL THREE TRAVEL, not just the solver. The add-on spawns the chosen
# directory's own ppf-cts-server, and the build worker that server starts loads
# the cdylib beside it, which is what makes the session script name the CPU
# solver. Ship two of the three and a CPU run quietly executes the Metal
# solver, the two halves of one run coming from different builds.
CPU_REL="$SRC_DIR/target/cpu/release"
CPU_SOLVER_BIN="$CPU_REL/ppf-contact-solver"
CPU_SERVER_BIN="$CPU_REL/ppf-cts-server"
CPU_PY_DYLIB="$CPU_REL/lib_ppf_cts_py.dylib"

for artifact in "$SOLVER_BIN" "$SERVER_BIN" "$PY_DYLIB" \
    "$CPU_SOLVER_BIN" "$CPU_SERVER_BIN" "$CPU_PY_DYLIB"; do
    [ -f "$artifact" ] || die \
        "$artifact not found" \
        "Run build.sh first."
done

# ASK EACH ARTIFACT WHAT IT IS RATHER THAN INFERRING IT FROM ITS PATH, which
# is the whole reason build.rs writes a marker at all: the executable name
# cannot carry the backend, so the directory name is a convention and this is
# evidence. Answered before the argument parser and before any backend is
# opened, so it costs nothing and says which target each was compiled for.
for expected in "$SOLVER_BIN:metal" "$CPU_SOLVER_BIN:cpu"; do
    binary="${expected%:*}"
    want="${expected##*:}"
    got="$("$binary" --backend)" || die \
        "$binary --backend failed" \
        "Re-run build.sh. A binary that will not run here cannot be packaged."
    [ "$got" = "$want" ] || die \
        "$binary is the $got backend and this step expects $want" \
        "The two builds each own a target directory and must not be mixed." \
        "Remove target and target/cpu and re-run build.sh."
done
printf '  [OK] metal and cpu solvers, each in its own target directory\n'

# Take the backend from the load command the solver actually carries, not from
# a glob over target/release/build: several stale build-script output
# directories can coexist there and only one of them is loaded.
. "$BUILD_MAC/scripts/backend-path.sh"
BACKEND_SRC="$(resolve_backend "$SOLVER_BIN")" || die \
    "could not resolve the Metal backend $SOLVER_BIN loads (see above)" \
    "Either that binary was not built against the Metal backend, or the" \
    "directory its rpath names is gone; re-run build.sh, since a cargo clean" \
    "removes that directory."
printf '  [OK] backend %s\n' "$BACKEND_SRC"

# The interpreter warmup.sh provisioned. A distribution without one is the
# exact thing this packaging exists to prevent, so its absence is fatal rather
# than a payload that quietly reverts to needing the user's own Python.
[ -x "$PY_ROOT/bin/python3" ] || die \
    "no bundled interpreter at $PY_ROOT/bin/python3" \
    "Run ./warmup.sh, which downloads a relocatable CPython and installs the" \
    "frontend packages into it. If that run was made with PPF_MAC_PYTHON=0," \
    "re-run it without that switch: a distribution that ships no interpreter" \
    "cannot start JupyterLab on a machine that has no Python, which is the" \
    "whole point of packaging one."
"$PY_ROOT/bin/python3" -c 'pass' >/dev/null 2>&1 || die \
    "$PY_ROOT/bin/python3 exists but does not run" \
    "Remove build-mac-native/python and re-run ./warmup.sh."
printf '  [OK] interpreter %s (%s)\n' "$PY_ROOT/bin/python3" \
    "$("$PY_ROOT/bin/python3" -c 'import platform; print(platform.python_version())')"

# The version, asked of the artifact rather than written down here. Read now,
# while the build tree's rpath still resolves: after step 6 the copy in the
# payload resolves the backend only through @rpath, and a version read from a
# binary
# that cannot load its dylib would be a failure with nothing to do with the
# version itself. `#[clap(author, version, ...)]` in args.rs makes this print
# "ppf-contact-solver <version>", so the last field is the number.
VERSION_LINE="$("$SOLVER_BIN" --version 2>/dev/null)" || die \
    "$SOLVER_BIN --version failed" \
    "The version comes from the binary, so a binary that will not run" \
    "here cannot be packaged. Re-run build.sh."
APP_VERSION="${VERSION_LINE##* }"
printf '%s' "$APP_VERSION" | grep -Eq '^[0-9][0-9.]*$' || die \
    "could not read a version out of: $VERSION_LINE" \
    "The command was:  $SOLVER_BIN --version" \
    "Its last whitespace-separated field is taken as the version and has to" \
    "look like 1.2.3. Nothing here invents one."
printf '  [OK] version %s\n' "$APP_VERSION"

# ---------------------------------------------------------------------------
step "[2/13] Creating the distribution directory"
# ---------------------------------------------------------------------------

rm -rf "$DIST"
mkdir -p "$PKG/bin" "$PKG/target/release" "$PKG/target/cpu/release"
printf '  [OK] %s\n' "$PKG"

# ---------------------------------------------------------------------------
step "[3/13] Copying binaries"
# ---------------------------------------------------------------------------

cp "$SOLVER_BIN" "$PKG/target/release/"
cp "$SERVER_BIN" "$PKG/target/release/"
cp "$PY_DYLIB" "$PKG/target/release/"
cp "$BACKEND_SRC" "$PKG/bin/"
BACKEND_DST="$PKG/bin/libppfbe_metal.dylib"
[ -f "$BACKEND_DST" ] || die "the backend did not land at $BACKEND_DST"
chmod u+w "$BACKEND_DST" "$PKG/target/release/"*

cp "$CPU_SOLVER_BIN" "$PKG/target/cpu/release/"
cp "$CPU_SERVER_BIN" "$PKG/target/cpu/release/"
cp "$CPU_PY_DYLIB" "$PKG/target/cpu/release/"
chmod u+w "$PKG/target/cpu/release/"*
# The CPU solver links no backend dylib: that backend is Rust and its kernels
# are compiled into the binary, so there is no second library to place beside
# it and nothing to rewrite in step 6.

# THE MARKER TRAVELS WITH THE BINARIES, and it is not decoration. It is the
# only thing that says which backend a directory holds, and `frontend.backend_of`
# reads exactly this file: without it `frontend.solver_dir(CPU)` answers None
# inside a distribution that ships a CPU build, so a notebook could not select
# the backend the payload carries. Written here rather than copied, because
# `build.rs` writes it into the profile directory of a build tree and what
# should ship is a statement about what LANDED, not whatever that directory
# happened to hold.
printf 'metal' > "$PKG/target/release/.ppf-backend"
printf 'cpu' > "$PKG/target/cpu/release/.ppf-backend"

# THE MARKER THAT SAYS THIS TREE KEEPS ITS OWN STATE. Without it the frontend
# resolves a session data root and an asset cache under $HOME, which is right
# for a developer checkout and wrong for a folder someone downloaded: removing
# the folder would then leave gigabytes behind with nothing naming where.
#
# IT IS A FILE RATHER THAN AN ENVIRONMENT VARIABLE because the launcher is not
# the only way into this distribution. The Blender add-on spawns
# target/release/ppf-cts-server directly, and a variable exported by the
# launcher would not reach it, so one distribution would keep state in two
# places depending on how it was started. `datamodel::app::is_selfcontained`
# reads this file.
printf '%s\n' "$APP_VERSION" > "$PKG/.ppf-selfcontained"
printf '  [OK] solver, server, cdylib, backend, and the CPU build beside them\n'

# The .metallib files that travel with the dylib. They sit beside it because
# that is where the backend looks: library_directory() in
# crates/ppf-cts-compute/metal/metal_context.mm resolves the directory through
# dladdr on its own loaded code, so a moved distribution reads its own
# neighbors and never the build tree's. The Metal Makefile puts both kinds in
# $(OUT_DIR)/lib beside the dylib, so one loop copies both.
#
# THE TWO KINDS ARE NOT INTERCHANGEABLE, AND ONLY ONE OF THEM IS OPTIONAL. That
# distinction is the whole reason this block counts them apart:
#
#   ppf_entries.metallib  the generated entry library, MANDATORY. `be_open` in
#                         crates/ppf-cts-compute/metal/backend/backend.mm builds
#                         <library_dir>/ppf_entries.metallib and fails to open
#                         the backend when that load fails, so a distribution
#                         shipped without it would package cleanly here and fail
#                         at the user's first run with a backend that will not
#                         open. Asserted by name below.
#
#   <hash>.metallib       the pre-compiled shader cache, OPTIONAL. Each is named
#                         by a hash over the shader text it was compiled from, so
#                         one that does not match the shipped dylib is simply not
#                         asked for. The Metal Toolchain that produces them is a
#                         separate downloadable component, and a distribution
#                         built without it runs the same way at a slower first
#                         start.
ENTRY_METALLIB_NAME="ppf_entries.metallib"
SHADER_CACHE_COUNT=0
while IFS= read -r metallib; do
    [ -n "$metallib" ] || continue
    cp "$metallib" "$PKG/bin/"
    metallib_leaf="$(basename "$metallib")"
    chmod u+w "$PKG/bin/$metallib_leaf"
    if [ "$metallib_leaf" != "$ENTRY_METALLIB_NAME" ]; then
        SHADER_CACHE_COUNT=$((SHADER_CACHE_COUNT + 1))
    fi
done <<EOF
$(find "$(dirname "$BACKEND_SRC")" -maxdepth 1 -name '*.metallib' | sort)
EOF

# The mandatory half, by name and separately from the count above, so the
# "none present" message below can only ever be about the optional cache. This
# is a hard failure rather than a warning because nothing downstream catches
# it: the gates in step 10 walk references and the smoke test in step 11 asks
# the solver which backend it was compiled for, which is answered before any
# backend is opened.
[ -f "$PKG/bin/$ENTRY_METALLIB_NAME" ] || die \
    "the generated entry library is missing: bin/$ENTRY_METALLIB_NAME" \
    "It is built beside the backend dylib, and this run resolved that" \
    "directory as:  $(dirname "$BACKEND_SRC")" \
    "The Metal backend refuses to open without that library, so a" \
    "distribution packaged without it builds here and fails on the user's" \
    "first run." \
    "Re-run build.sh. The Metal Makefile's abi target names this library as a" \
    "hard prerequisite of the dylib, so a directory holding one and not the" \
    "other is from a build that did not run to completion."
printf '  [OK] %s\n' "$ENTRY_METALLIB_NAME"

if [ "$SHADER_CACHE_COUNT" -gt 0 ]; then
    printf '  [OK] %s pre-compiled shader cache librar(ies)\n' "$SHADER_CACHE_COUNT"
else
    printf '  [--] no pre-compiled shader cache libraries; the payload will\n'
    printf '       compile its pipelines at run time (build.sh step 4 says why)\n'
fi

# ---------------------------------------------------------------------------
step "[4/13] Copying the Python side"
# ---------------------------------------------------------------------------

copy_tree() {
    # $1 = source dir, $2 = destination dir, remaining = rsync excludes
    local src="$1" dst="$2"
    shift 2
    [ -d "$src" ] || die "$src not found"
    mkdir -p "$dst"
    local args=(-a)
    local pattern
    for pattern in "$@"; do
        args+=(--exclude "$pattern")
    done
    rsync "${args[@]}" "$src/" "$dst/"
}

copy_tree "$SRC_DIR/frontend" "$PKG/frontend" '__pycache__' '*.pyc' '*.pyo'
printf '  [OK] frontend\n'

# Examples ship as sources only, the way the Windows distribution does: notebooks
# and scripts, none of the assets a run downloads or writes.
mkdir -p "$PKG/examples"
# -m prunes the directories the include of '*/' would otherwise create empty.
rsync -a -m --include '*/' --include '*.ipynb' --include '*.py' --exclude '*' \
    "$SRC_DIR/examples/" "$PKG/examples/"
printf '  [OK] examples\n'

# frontend/_session_inspect_.py harvests log-channel names by walking these
# two trees for `// Name:` docstrings. Without them, session.get.log.names()
# returns an empty list and a notebook asserting on a channel name fails.
copy_tree "$SRC_DIR/crates/ppf-cts-solver/src" "$PKG/crates/ppf-cts-solver/src" \
    '__pycache__' 'build' 'build-tests' 'obj' 'tests' '*.pyc' '*.o' '*.d'
copy_tree "$SRC_DIR/crates/ppf-cts-core/src" "$PKG/crates/ppf-cts-core/src" \
    '__pycache__' '*.pyc'
printf '  [OK] crate source roots\n'

# THE BRANCH STAMP, WHICH IS WHAT KEEPS THE PKG FROM RUNNING git.
#
# data_dirpath_for in crates/ppf-cts-core/src/datamodel/app.rs reads
# <tree root>/.git/branch_name.txt and falls through to
# `git branch --show-current` in that directory when the file is absent or
# empty. Every App.create() reaches it through get_data_dirpath in
# frontend/_app_.py, so the fall-through fires on the FIRST CELL OF THE FIRST
# NOTEBOOK. The distribution directory is the tree root and carries no
# repository, so without this file every run spawns git.
#
# On a Mac with no command line tools /usr/bin/git is the developer-tools
# shim, and invoking it presents the "install the command line developer
# tools" dialog. That is xcode-select --install by another name, which this
# distribution promises never to require, so this is a constraint
# violation rather than a slow path. app.rs's own comment names this file as
# the one release packaging writes.
#
# THE EMPTINESS IS TESTED, NOT THE EXIT STATUS. `git branch --show-current`
# succeeds with EMPTY output on a detached HEAD, and app.rs treats an empty
# file as absent, so `|| echo unknown` alone would write a blank file and the
# fall-through would fire anyway.
#
# app.rs is separately being taught to return "unknown" without spawning when
# the directory holds no .git at all. The two are belt and braces on purpose:
# this one gives a session directory a real name instead of git-unknown, and
# that one holds for any payload assembled by something other than this script.
BUILD_BRANCH="$(git -C "$SRC_DIR" branch --show-current 2>/dev/null || true)"
[ -n "$BUILD_BRANCH" ] || BUILD_BRANCH="unknown"
mkdir -p "$PKG/.git"
printf '%s\n' "$BUILD_BRANCH" > "$PKG/.git/branch_name.txt"
[ -s "$PKG/.git/branch_name.txt" ] || die \
    "the branch stamp at .git/branch_name.txt is empty" \
    "An empty file is what app.rs treats as absent, so the frontend would" \
    "spawn git on the first notebook cell. Nothing here invents a branch name; the" \
    "value is whatever this build tree reports, or the literal 'unknown'."
printf '  [OK] branch stamp .git/branch_name.txt (%s)\n' "$BUILD_BRANCH"

# ---------------------------------------------------------------------------
step "[5/13] Copying the bundled interpreter and relocating its scripts"
# ---------------------------------------------------------------------------

# -a and NOT -aL. A python-build-standalone install tree is full of relative
# symlinks, bin/python3 among them, and dereferencing them would both bloat the
# payload and break the layout the interpreter computes its own home from.
rsync -a "$PY_ROOT/" "$PKG/python/"
[ -x "$PKG_PY" ] || die \
    "no interpreter at $PKG_PY after the copy" \
    "The canonical path is <python root>/bin/python3, a relative symlink to" \
    "the versioned binary that survives any copy that preserves symlinks."
printf '  [OK] interpreter copied\n'

# THE CONSOLE SCRIPTS' SHEBANGS, WHICH NAME THIS BUILD TREE UNTIL THIS RUNS.
#
# pip writes the absolute path of the interpreter it installed for into every
# console script it generates, and warmup.sh installs into
# build-mac-native/python, so roughly sixty files under python/bin open with
# #!<this build tree>/python/bin/python3. jupyter, jupyter-lab, jupyter-server,
# f2py, debugpy, pygmentize and pip itself are all in that set.
#
# TWO SEPARATE DEFECTS, AND EITHER ALONE WOULD JUSTIFY THIS LOOP:
#
#   The path names this build tree, so gate C in step 10 lists every one of
#   them and the run dies with nothing signed. Gate C's own prose names an
#   absolute shebang under python/bin as a thing it catches, and until now
#   nothing in warmup.sh or here produced anything else.
#
#   The path is dead on the user's Mac whatever it says, because it names
#   this build tree, and no absolute shebang written here could be right on
#   any other machine either: where the distribution folder lands is the
#   user's choice, and the kernel takes the whole #! line as one literal path.
#
# THE REPLACEMENT IS TWO LINES THAT ARE A VALID SHELL SCRIPT AND A VALID PYTHON
# MODULE AT ONCE. /bin/sh reads the opening word as `exec`, so line 2 execs the
# interpreter sitting beside the script, on the script itself; the interpreter
# then reads lines 2 and 3 as one triple-quoted string and evaluates nothing.
# Resolution is relative to the script, so it follows the distribution wherever
# the user puts it, and every expansion is quoted, so a space in the path the
# user chose costs nothing.
#
# THE SCRIPTS ARE REWRITTEN RATHER THAN DELETED. Nothing in the distribution
# invokes one: the launcher runs `python -m jupyterlab`, and jupyter_client
# rewrites a kernelspec's argv[0] to sys.executable before launching a kernel.
# Deleting them would therefore satisfy the gates just as well. They are kept
# because a developer who opens a terminal in the distribution reasonably
# expects python/bin/jupyter to work, and a trampoline that resolves its own
# neighbor is the form that keeps working wherever the folder is put.
printf 'Rewriting console script shebangs...\n'
# The count is kept beside the list rather than read off it. macOS ships bash
# 3.2, where expanding an empty array under set -u is an unbound-variable
# error, so the list is expanded only inside the test below and never when it
# is empty.
TRAMPOLINED=()
TRAMPOLINE_COUNT=0
for script in "$PKG/python/bin"/*; do
    [ -f "$script" ] || continue
    # Two bytes first, so the interpreter's own Mach-O is dismissed without
    # reading a line of binary into a variable.
    [ "$(head -c 2 "$script" 2>/dev/null)" = '#!' ] || continue
    shebang="$(head -1 "$script")"
    case "$shebang" in
        '#!'*python*) ;;
        # python3.12-config and its siblings are shell scripts. Left alone.
        *) continue ;;
    esac

    # What remains after an optional /usr/bin/env must name an interpreter and
    # nothing else. The trampoline passes no interpreter arguments, so an
    # argument here would be dropped, and a silent loss inside a release
    # is the class of defect this script exists to prevent: it stops the build
    # and names the file instead.
    #
    # A SPACE IS NOT THE TEST, because it cannot tell an argument from a
    # directory name that contains one, and a developer who cloned this tree
    # into a path with a space would otherwise be refused with a message about
    # arguments that are not there. What settles it is whether the whole
    # remainder resolves to a file: the shebang names the build tree's own
    # interpreter, which is still on disk at this point in the run.
    interp="${shebang#\#!}"
    case "$interp" in
        "/usr/bin/env "*) interp="${interp#/usr/bin/env }" ;;
    esac
    if [ ! -e "$interp" ]; then
        case "$interp" in
            *" "*) die \
                "a script under python/bin has a shebang that is not just an interpreter" \
                "  file:    ${script#"$PKG/"}" \
                "  shebang: $shebang" \
                "The remainder after the interpreter does not resolve to a file," \
                "so it reads as an interpreter followed by an argument. The" \
                "trampoline written here passes no interpreter arguments, so that" \
                "argument would be dropped silently. Decide deliberately what it" \
                "becomes before shipping." ;;
        esac
    fi

    # The body is parked outside the payload: a temporary file inside python/bin
    # would survive a failure between the two writes and ship.
    body="$(mktemp "${TMPDIR:-/tmp}/ppf-trampoline.XXXXXX")"
    tail -n +2 "$script" > "$body"
    {
        printf '%s\n' '#!/bin/sh'
        printf '%s\n' "'''exec' \"\$(dirname \"\$0\")/python3\" \"\$0\" \"\$@\"" "' '''"
        cat "$body"
    } > "$script"
    rm -f "$body"
    chmod 755 "$script"
    TRAMPOLINED+=("$script")
    TRAMPOLINE_COUNT=$((TRAMPOLINE_COUNT + 1))
done
printf '  [OK] %d console scripts now resolve the interpreter beside them\n' \
    "$TRAMPOLINE_COUNT"

# The post-condition, asserted rather than assumed, because this loop is the
# only thing between gate C and a build-tree path in sixty files.
for script in "$PKG/python/bin"/*; do
    [ -f "$script" ] || continue
    [ "$(head -c 2 "$script" 2>/dev/null)" = '#!' ] || continue
    case "$(head -1 "$script")" in
        '#!'*python*) die \
            "a shebang under python/bin still names an interpreter directly" \
            "  file: ${script#"$PKG/"}" \
            "The loop above was supposed to have rewritten it, so this is a" \
            "defect in that loop rather than in the payload. A shebang that" \
            "survives here is an absolute path on the user's machine." ;;
    esac
done

# And the other half of the polyglot, which no later step would exercise: the
# two lines have to remain a valid Python module, or every one of these scripts
# fails at its first use with a SyntaxError. Parsed from bytes rather than from
# decoded text, so this sees exactly what the interpreter will see, including a
# coding declaration that the two inserted lines pushed past line 2.
if [ "$TRAMPOLINE_COUNT" -gt 0 ]; then
    "$PKG_PY" - "${TRAMPOLINED[@]}" <<'PY' || die \
        "a trampolined console script no longer parses as Python" \
        "The traceback above names the file and the line. The trampoline is a" \
        "polyglot: its second line must read as the word exec to /bin/sh and" \
        "as the start of a triple-quoted string to Python, and this says one" \
        "of those two is no longer true."
import ast
import sys

for path in sys.argv[1:]:
    with open(path, "rb") as handle:
        ast.parse(handle.read(), filename=path)
print("  [OK] %d trampolined scripts still parse as Python" % (len(sys.argv) - 1))
PY
fi

# ---------------------------------------------------------------------------
step "[6/13] Rewriting load commands"
# ---------------------------------------------------------------------------

# A copied binary still records the ABSOLUTE path the backend had in the build
# tree, in its own load command and in an LC_RPATH. Left alone, the payload
# either fails to load on a machine without that tree or, worse, loads the
# build tree's dylib on the machine that built it, so the thing that ships is
# never the thing that was tested. Both are rewritten to forms relative to the
# loading file, and step 10 asserts nothing absolute survived.
#
# THE INTERPRETER'S MACH-O FILES ARE NOT REWRITTEN. They are relocatable
# already, they reach their neighbors through @executable_path and
# @loader_path, and rewriting a load command this project did not create is a
# change with no defect behind it. They ARE walked by the gates below and they
# ARE signed.

MACHO_FILES=(
    "$PKG/target/release/ppf-contact-solver"
    "$PKG/target/release/ppf-cts-server"
    "$PKG/target/release/lib_ppf_cts_py.dylib"
    "$PKG/target/cpu/release/ppf-contact-solver"
    "$PKG/target/cpu/release/ppf-cts-server"
    "$PKG/target/cpu/release/lib_ppf_cts_py.dylib"
)

install_name_tool -id "@rpath/libppfbe_metal.dylib" "$BACKEND_DST"
printf '  [OK] id of libppfbe_metal.dylib\n'

# A dylib also carries its OWN name, LC_ID_DYLIB, and cargo writes that as the
# absolute path the library was linked at, which is inside the build tree.
# Nothing in the distribution RESOLVES the cdylib through that name, since
# frontend/__init__.py opens it by absolute path through importlib, but it is
# still a recorded reference to a directory the payload does not ship, and that
# is what the self-containment gate at the end refuses. Rewrite it for every
# dylib copied in, so the rule is the one the backend above already follows and
# a dylib added later needs no edit here.
for macho in "${MACHO_FILES[@]}"; do
    case "$macho" in
    *.dylib) ;;
    *) continue ;;
    esac
    # `otool -D` prints a filename header first and the install name second.
    own_id="$(otool -D "$macho" | awk 'NR > 1 { print $1 }')"
    case "$own_id" in
    "" | @*) continue ;;
    esac
    leaf="$(basename "$macho")"
    install_name_tool -id "@rpath/$leaf" "$macho"
    printf '  %s: id %s -> @rpath/%s\n' "$leaf" "$own_id" "$leaf"
done

rpaths_of() {
    # LC_RPATH entries, one per line. The path is on the line after the cmd.
    otool -l "$1" | awk '/^ *cmd LC_RPATH$/ {found = 1} found && /^ *path / {print $2; found = 0}'
}

count_occurrences() {
    # How many times the fixed string $1 occurs in the text on stdin. grep -c
    # counts the LINES that match, which is 1 for a binary's string table read
    # back as a single run however many times the needle is in it, so the
    # occurrences are counted directly. Fixed-string throughout: the needle is
    # a filesystem path and is never a pattern.
    awk -v needle="$1" '
        {
            line = $0
            while ((p = index(line, needle)) > 0) {
                n++
                line = substr(line, p + length(needle))
            }
        }
        END { print n + 0 }
    '
}

ids_of() {
    # The file's own install name, LC_ID_DYLIB, one line per architecture. A
    # dylib has one and an executable or a loadable module has none, so this
    # printing nothing is the normal case rather than a failure. otool -D puts
    # the file's own path, and for a fat file an "(architecture ...)" header,
    # on a line ending in a colon, which is what separates a header from the
    # name itself.
    otool -D "$1" | awk '!/:$/ && NF { print $1 }'
}

deps_of() {
    # The dependency names in LC_LOAD_DYLIB and friends, one per line.
    #
    # A DYLIB'S OWN INSTALL NAME IS TAB-INDENTED IN otool -L's OUTPUT TOO, AND
    # IT IS NOT A DEPENDENCY. Reading it as one is not a cosmetic error: a
    # wheel published by a project that pads its install names, or that lipos
    # two thin builds together, carries an id naming a path on ITS build
    # machine, and a gate that treats that as a dependency reports a working
    # payload as broken. Pillow's .dylibs are the worked case, every one of them
    # named /DLC/PIL/... by delocate. The id is asked for separately and
    # subtracted here, so what this returns is what dyld must actually resolve.
    local own
    own="$(ids_of "$1")"
    # THE IDS REACH awk THROUGH THE ENVIRONMENT RATHER THAN THROUGH -v. A fat
    # binary carries one install name PER ARCHITECTURE, so this value
    # legitimately contains a newline, and BSD awk refuses a newline inside a
    # -v assignment: "awk: newline in string ... at source line 1", which
    # aborts the walk on the first such file. debugpy's attach.dylib is two
    # thin builds lipoed together and is exactly that case. ENVIRON carries the
    # value whatever is in it. The assignment sits on awk rather than at the
    # head of the pipeline, where it would reach otool instead.
    otool -L "$1" | PPF_MACHO_OWN_IDS="$own" awk '
        BEGIN {
            n = split(ENVIRON["PPF_MACHO_OWN_IDS"], a, "\n")
            for (i = 1; i <= n; i++) if (a[i] != "") is_own[a[i]] = 1
        }
        NR > 1 && /^\t/ && !($1 in is_own) { print $1 }
    '
}

for macho in "${MACHO_FILES[@]}"; do
    name="$(basename "$macho")"
    references_backend=0

    # Point every reference to the backend at the copy in bin/. Only the
    # solver binary carries one today: ppf-cts-server and the PyO3 cdylib
    # depend on ppf-cts-core, not on the solver crate whose build script emits
    # the link arguments. The loop does not assume that, so a new binary that
    # starts linking the backend is handled without an edit here.
    while IFS= read -r ref; do
        [ -n "$ref" ] || continue
        install_name_tool -change "$ref" "@rpath/libppfbe_metal.dylib" "$macho"
        references_backend=1
        printf '  %s: %s -> @rpath/libppfbe_metal.dylib\n' "$name" "$ref"
    done <<EOF
$(otool -L "$macho" | awk '/libppfbe_metal\.dylib/ {print $1}')
EOF

    # Drop every absolute rpath. The ones cargo baked in point into the build
    # tree; a system path would be a load-command entry, not an rpath.
    while IFS= read -r rp; do
        [ -n "$rp" ] || continue
        case "$rp" in
            @*) continue ;;
        esac
        install_name_tool -delete_rpath "$rp" "$macho"
        printf '  %s: dropped rpath %s\n' "$name" "$rp"
    done <<EOF
$(rpaths_of "$macho")
EOF

    # target/release/<file> to bin is two levels up: from the file, one level
    # to target/release's parent target, one more to the distribution
    # directory, and then down into bin. An executable resolves
    # @executable_path against itself; a dylib loaded by an interpreter must
    # use @loader_path, because the executable is python. Added only where a
    # reference was rewritten, so nothing carries a search path it has no use
    # for.
    if [ "$references_backend" -eq 1 ]; then
        case "$macho" in
            *.dylib) install_name_tool -add_rpath "@loader_path/../../bin" "$macho" ;;
            *) install_name_tool -add_rpath "@executable_path/../../bin" "$macho" ;;
        esac
        printf '  %s: rpath -> the bin directory\n' "$name"
    fi
done
printf '  [OK] load commands rewritten\n'

# ---------------------------------------------------------------------------
step "[7/13] Thinning the payload to arm64"
# ---------------------------------------------------------------------------

# THE PRUNE HAPPENS BEFORE THE WALK, AND THAT ORDER IS LOAD-BEARING TWICE.
# Everything below reads the Mach-O list this step builds: the thinning itself,
# the deployment target stamped into the launcher, and the gates. A file that is
# deleted later must therefore not be in it. The deployment target is the sharp
# case: it is the MAXIMUM over the payload's own binaries, so one Mach-O under
# the stdlib's test package can raise the minimum macOS the launcher accepts
# above what the shipped set actually requires, and refuse a user whose machine
# is fine. Pruning first makes one list serve every reader, which is also why
# the thinning pass does not spend time on files that do not ship.
#
# Resolved from the interpreter rather than from a literal 3.12, so a version
# bump in the manifest needs no edit here. purelib is not guaranteed to sit
# under stdlib, so both are named.
STDLIB="$("$PKG_PY" -c 'import sysconfig; print(sysconfig.get_paths()["stdlib"])')" || die \
    "the bundled interpreter could not report its stdlib path"
SITE="$("$PKG_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" || die \
    "the bundled interpreter could not report its site-packages path"
[ -d "$STDLIB" ] || die "the interpreter reports a stdlib at $STDLIB, which is not there"

# The only prune. It is certainly unused in production, it saves tens of
# megabytes, and it removes the badsyntax_*.py files that would otherwise make
# compileall report a failure in step 9.
#
# DO NOT EXTEND THIS LIST. Pruning tkinter, idlelib or config-* each saves a
# little and each risks an import failure that appears only at run time on a
# user's machine, which is the class of defect this packaging is built to avoid.
if [ -d "$STDLIB/test" ]; then
    rm -rf "$STDLIB/test"
    printf '  [OK] pruned the stdlib test package\n'
fi


# WHERE THIS SITS IN THE ORDER, AND WHY IT IS HERE RATHER THAN ANYWHERE ELSE.
#
#   It precedes signing, because rewriting a file invalidates its signature.
#   Thinning after step 12 would leave every touched file with a signature
#   that does not match its bytes, which on Apple silicon is a binary that
#   does not load at all.
#
#   It precedes the deployment-target read at the top of step 8. `otool -l`
#   on a universal file prints the load commands of EVERY slice, so a maximum
#   taken over a fat payload can include an architecture that does not ship.
#   Taken here, the number is a property of what ships.
#
#   Step 9's prune is the only later deletion, and it removes the stdlib test
#   package, which is not where the universal files are, so nothing thinned
#   here is thrown away.

discover_macho() {
    # Every Mach-O file in the payload, one path per line. No name filter: a
    # filter on *.dylib, *.so or the executable bit is faster and can miss one,
    # and missing one is the silent failure the gates exist to catch. The sed
    # cuts at ": Mach-O", which is safe because no path in the payload carries
    # that substring.
    find "$PKG" -type f -print0 \
        | xargs -0 file -- \
        | sed -n 's/: *Mach-O .*$//p'
}

assert_walk_complete() {
    # $1 = the discovered list, one path per line.
    #
    # A LOWER BOUND ON THE WALK, BECAUSE A TRUNCATED DISCOVERY REPORTS
    # SUCCESS. The discovery asks `file` what every payload file is, which is
    # the only way to catch a Mach-O whose name says nothing. A pipeline that
    # stops early comes back SHORT rather than empty, and the emptiness test
    # at each call site only catches empty, so the thinning pass would leave a
    # fat file untouched, the gates would walk part of the payload and step 12
    # would sign part of it, all printing [OK]. Every *.dylib and *.so in the
    # payload is certainly Mach-O, so those are a set the walk must contain.
    # This does not replace the walk, which legitimately finds more than these;
    # it establishes that the walk reached the end of the payload.
    #
    # BOTH WALKS CALL IT. Step 7 thins what it is given and step 10 gates and
    # step 12 signs what it is given, and a short list is silent in all three,
    # so each list is checked where it is built rather than once.
    local list="$1"
    local named
    local short_walk=0
    local short_reported=0
    while IFS= read -r named; do
        [ -n "$named" ] || continue
        # A HERESTRING RATHER THAN A PIPE, BECAUSE grep -q PLUS pipefail
        # INVERTS THIS TEST. grep -q exits at its first match, the writer
        # feeding it then dies with EPIPE, and `set -o pipefail` reports that
        # writer's failure as the pipeline's status, so a file that IS in the
        # walk reads as missing and every one of them is reported. A herestring
        # has no pipeline and no writer to kill.
        grep -qxF -- "$named" <<<"$list" && continue
        short_walk=$((short_walk + 1))
        # Capped, because a truncation misses everything past the cut and the
        # wall of names that produces buries the count, which is the part that
        # says how bad it is. The first few name where the payload was still
        # being read.
        if [ "$short_reported" -lt 20 ]; then
            printf 'ERROR: %s is a library the discovery did not report\n' \
                "${named#"$PKG/"}" >&2
            short_reported=$((short_reported + 1))
        fi
    done < <(find "$PKG" -type f \( -name '*.dylib' -o -name '*.so' \) -print)
    if [ "$short_walk" -gt "$short_reported" ]; then
        printf 'ERROR: ... and %s more not listed\n' \
            "$((short_walk - short_reported))" >&2
    fi
    [ "$short_walk" -eq 0 ] || die \
        "the Mach-O discovery came back short: $short_walk libraries missing" \
        "The libraries named above are in the payload and are certainly" \
        "Mach-O, and the walk did not report them, so it stopped before the" \
        "end of the payload. Everything past that point would go unthinned," \
        "ungated and UNSIGNED while this script printed success, which is the" \
        "failure the walk exists to prevent. Fix the discovery; do not lower" \
        "this check."
}

assert_arm64_only() {
    # $1 = the list to check, one path per line.
    #
    # Every Mach-O in the payload reports arm64 and nothing else. This is the
    # statement the thinning pass exists to make true, and it is checked
    # rather than assumed: lipo is asked file by file, and one answer that is
    # not exactly arm64 fails the build. It runs at the end of step 7, so a
    # thinning defect is named where it happened, and again in step 10 over
    # the list rebuilt after the prune, which is the set that actually ships.
    local list="$1"
    local checked
    local found
    local bad=0
    while IFS= read -r checked; do
        [ -n "$checked" ] || continue
        found="$(lipo -archs "$checked" 2>/dev/null)" || found=""
        found="$(printf '%s' "$found" | tr -s ' ' ' ' | sed 's/^ *//; s/ *$//')"
        if [ "$found" != "arm64" ]; then
            printf 'ERROR: %s reports architectures: %s\n' \
                "${checked#"$PKG/"}" "${found:-<lipo said nothing>}" >&2
            bad=$((bad + 1))
        fi
    done <<EOF
$list
EOF
    [ "$bad" -eq 0 ] || die \
        "$bad Mach-O files in the payload are not arm64 alone" \
        "This distribution is arm64 by construction and the thinning pass is" \
        "what makes that true of every file. A file listed above either was" \
        "not reached by the walk or was not thinned, and either way what" \
        "ships carries code that cannot run on it."
}

# The list this pass walks. Discovered now, before anything is thinned, so a
# file rewritten below was seen by the same walk that gates it.
THIN_MACHO=()
while IFS= read -r found; do
    [ -n "$found" ] || continue
    THIN_MACHO+=("$found")
done < <(discover_macho)
[ "${#THIN_MACHO[@]}" -gt 0 ] || die \
    "the Mach-O discovery found nothing in the payload" \
    "That is a failure of the discovery itself, not a clean payload: the" \
    "distribution certainly contains a solver, a backend and an interpreter." \
    "A pass reporting clean over nothing is the mistake this refuses to make."
THIN_MACHO_LIST="$(printf '%s\n' "${THIN_MACHO[@]}")"
assert_walk_complete "$THIN_MACHO_LIST"
printf 'Walking %d Mach-O files\n' "${#THIN_MACHO[@]}"

THINNED=0
ALREADY_THIN=0
SAVED_BYTES=0
for macho in "${THIN_MACHO[@]}"; do
    # lipo -archs answers for a thin file as well as a fat one, which is what
    # makes this a read rather than a try-and-recover: `lipo -thin` on a file
    # that is already one architecture is an error, so the already-thin case is
    # skipped rather than attempted and its failure swallowed.
    archs="$(lipo -archs "$macho" 2>/dev/null)" || die \
        "lipo could not read the architectures of a Mach-O file in the payload" \
        "  file: ${macho#"$PKG/"}" \
        "Every file the walk reports is a Mach-O, so a file lipo cannot read" \
        "is one this build cannot make a statement about. Decide deliberately" \
        "what it is before shipping it."
    archs="$(printf '%s' "$archs" | tr -s ' ' ' ' | sed 's/^ *//; s/ *$//')"
    case " $archs " in
        " arm64 ")
            ALREADY_THIN=$((ALREADY_THIN + 1))
            continue
            ;;
        *" arm64 "*) ;;
        *) die \
            "a Mach-O file in the payload carries no arm64 code" \
            "  file:          ${macho#"$PKG/"}" \
            "  architectures: $archs" \
            "This distribution is arm64 by construction, so a file with no" \
            "arm64 slice cannot run in it. Nothing here invents one." ;;
    esac

    before="$(stat -f %z "$macho")"
    # The permission bits in octal, in the BSD spelling, and ALL FOUR DIGITS.
    # Several files arrive from rsync -a with the publisher's mode, which is
    # not necessarily writable, so the write below needs u+w and the recorded
    # mode puts back exactly what was there. `%OLp` alone is the low three
    # digits, which silently drops a set-user-ID, set-group-ID or sticky bit;
    # nothing in a python-build-standalone tree carries one today, and reading
    # three digits is the one way this pass could quietly change a file it was
    # meant to leave alone.
    mode="$(stat -f '%OMp%OLp' "$macho")"
    # The thinned body is parked OUTSIDE the payload, for the same reason the
    # trampoline loop in step 5 gives: a temporary file inside the payload
    # survives a failure between the two writes and ships.
    thin="$(mktemp "${TMPDIR:-/tmp}/ppf-lipo.XXXXXX")"
    lipo -thin arm64 "$macho" -output "$thin" || die \
        "lipo -thin arm64 failed" \
        "  file: ${macho#"$PKG/"}" \
        "lipo's own message is above."
    # WRITTEN BACK THROUGH THE EXISTING FILE RATHER THAN MOVED OVER IT. That
    # preserves the inode, so a hard link the interpreter tree may hold to the
    # same file sees the new bytes rather than being left on the old ones, and
    # it means the mode is disturbed only by the chmod u+w this needs.
    chmod u+w "$macho"
    cat "$thin" > "$macho"
    rm -f "$thin"
    chmod "$mode" "$macho"
    after="$(stat -f %z "$macho")"
    SAVED_BYTES=$((SAVED_BYTES + before - after))
    THINNED=$((THINNED + 1))
    printf '  %s: %s -> arm64\n' "${macho#"$PKG/"}" "$archs"
done

printf '  [OK] thinned %d universal files to arm64, saving %d bytes\n' \
    "$THINNED" "$SAVED_BYTES"
printf '  [OK] %d files were already arm64 alone\n' "$ALREADY_THIN"
assert_arm64_only "$THIN_MACHO_LIST"
printf '  [OK] every Mach-O in the payload reports arm64 and nothing else\n'

# ---------------------------------------------------------------------------
step "[8/13] Generating the launcher, config.sh and the documentation"
# ---------------------------------------------------------------------------

minos_of() {
    # The deployment target a Mach-O records: LC_BUILD_VERSION's `minos` on
    # anything current, LC_VERSION_MIN_MACOSX's `version` on an older one. Both
    # spellings are read because the interpreter tree carries wheels built over
    # a range of toolchains.
    otool -l "$1" 2>/dev/null | awk '
        $1 == "cmd" && $2 == "LC_BUILD_VERSION"      { want = "minos";   next }
        $1 == "cmd" && $2 == "LC_VERSION_MIN_MACOSX" { want = "version"; next }
        want != "" && $1 == want                     { print $2; want = "" }
    '
}

# THE MINIMUM macOS THIS DISTRIBUTION RUNS ON, AND WHY IT IS COMPUTED RATHER
# THAN WRITTEN DOWN. It is the maximum over the payload's own binaries: the
# code cannot run below what it itself requires, so that is the only correct
# source. The launcher below refuses on an older macOS with both numbers named,
# and README.txt states the requirement, which is what makes the computation
# worth keeping: a directory has no property list for the system to enforce, so
# without this check a person on an older macOS gets a dyld diagnostic naming a
# symbol, which reads as a corrupt download.
#
# It is read off the payload AFTER step 7, so every file it reads carries one
# architecture and the maximum is over what ships.
printf 'Reading the deployment target off the payload...\n'
MIN_OS_LIST="$(
    while IFS= read -r macho; do
        [ -n "$macho" ] || continue
        minos_of "$macho"
    done <<EOF
$THIN_MACHO_LIST
EOF
)"
# sort -V is not something to rest a release on: its presence in BSD sort is
# not a given. A numeric sort on the three dotted fields orders versions the
# same way for anything a deployment target looks like.
MIN_OS="$(printf '%s\n' "$MIN_OS_LIST" | grep -E '^[0-9]' | sort -t. -k1,1n -k2,2n -k3,3n | tail -1 || true)"
[ -n "$MIN_OS" ] || die \
    "no Mach-O file in the payload reported a deployment target" \
    "Either the Mach-O discovery found nothing, which means the payload is" \
    "not what step 3 to step 5 copied, or otool printed neither" \
    "LC_BUILD_VERSION nor LC_VERSION_MIN_MACOSX. Guessing a number here would" \
    "put a claim in the launcher and in README.txt that no artifact supports."
printf '%s' "$MIN_OS" | grep -Eq '^[0-9][0-9.]*$' || die \
    "the deployment target read off the payload is not a version: $MIN_OS"
printf '  [OK] minimum macOS %s\n' "$MIN_OS"

# ---------------------------------------------------------------------------
# The launcher. This is the whole product: the file a person runs, at the top
# of the distribution directory, sharing its basename with the solver Mach-O
# down in target/release and being an entirely different program.
#
# THE BODY IS A SINGLE-QUOTED HEREDOC, SO NOTHING IN IT EXPANDS HERE. The two
# build-time values it needs are printed above it as assignments instead. Both
# were validated against ^[0-9][0-9.]*$ earlier in this run, so neither can
# carry a quote, a space or a newline, which is what lets them be spliced in
# unquoted.
# ---------------------------------------------------------------------------
{
    printf '%s\n' '#!/bin/bash'
    printf '%s\n' '# Generated by build-mac-native/bundle.sh. Edits are lost on the next'
    printf '%s\n' '# bundle; change bundle.sh instead.'
    printf '%s\n' '#'
    printf '%s\n' '# It starts JupyterLab IN THE FOREGROUND. The server output is this'
    printf '%s\n' "# terminal's and Ctrl+C stops it. There is no lock file, no pid file, no"
    printf '%s\n' '# log file and no separate stopper: the terminal is the log, and two'
    printf '%s\n' "# copies started at once are two servers, which Jupyter's own port retry"
    printf '%s\n' '# resolves.'
    printf '%s\n' '#'
    printf '%s\n' '# IT WRITES EVERYTHING INSIDE THE DISTRIBUTION DIRECTORY, and nothing'
    printf '%s\n' '# outside it. Sessions, the asset cache, the Jupyter and IPython state'
    printf '%s\n' '# and the notebooks all sit under this folder, so removing the folder is'
    printf '%s\n' '# a complete uninstall. That is the property the Windows distribution has'
    printf '%s\n' '# always had, and it is why this folder has to be somewhere you can'
    printf '%s\n' '# write.'
    printf '%s\n' '#'
    printf '%s\n' '# The two values below are stamped in when this file is written. Both are'
    printf '%s\n' '# read off the payload rather than written down: the version comes from'
    printf '%s\n' '# ppf-contact-solver --version, and the minimum is the highest deployment'
    printf '%s\n' '# target over every Mach-O file that ships here.'
    printf 'PPF_DIST_VERSION=%s\n' "$APP_VERSION"
    printf 'PPF_MIN_MACOS=%s\n' "$MIN_OS"
    cat <<'LAUNCHER_EOF'
set -euo pipefail

die() {
    printf 'ERROR: %s\n' "$1" >&2
    shift
    for line in "$@"; do
        printf '       %s\n' "$line" >&2
    done
    exit 1
}

# PATH IS SET FIRST, BEFORE ANYTHING BELOW RUNS A COMMAND. This program is
# started from the user's own shell, so it inherits whatever PATH that shell
# has, and everything from the uname below to the rsync that seeds the
# notebooks resolves through it. A PATH that carries none of the system
# directories would otherwise fail at the first external command with a bare
# "command not found", before this file has said anything useful.
#
# THE SYSTEM DIRECTORIES GO FIRST AND THE USER'S PATH IS KEPT AFTER THEM.
# This file does resolve tools through PATH: dirname, readlink, sleep, mkdir,
# rsync, uname, sw_vers, grep, awk and cp are all external. The interpreter and
# every binary of ours are named by absolute path, so the ordering does not
# decide which SOLVER runs, but it does decide which `cp` this script gets, and
# a shim earlier in a user's PATH should not be able to change that. Keeping
# the user's entries after the system ones is what lets a NOTEBOOK still find
# the git that four of them clone with, and the ffmpeg that video export needs,
# both of which are the user's to supply.
# ${PATH:+...} so that set -u is satisfied when PATH is unset.
export PATH="/usr/bin:/bin:/usr/sbin:/sbin${PATH:+:$PATH}"

# WHERE THIS FILE LIVES, RESOLVED BY HAND. It may be started through a symlink,
# from any working directory, and from a path holding a space.
#
# The chain is walked one hop at a time because `readlink -f` and `realpath`
# are not in the base system on every supported macOS, while `readlink` with no
# flag is. The walk is bounded because an unbounded one over a symlink cycle
# hangs with nothing printed, which is exactly the failure this program exists
# to make legible. `pwd -P` resolves a symlinked parent, so one name for this
# directory appears in every message below and in PYTHONPATH.
SELF="${BASH_SOURCE[0]}"
hops=0
while [ -L "$SELF" ]; do
    hops=$((hops + 1))
    [ "$hops" -le 32 ] || die \
        "the path this program was started from is a symlink loop" \
        "Started as: ${BASH_SOURCE[0]}" \
        "Run the launcher at its real location."
    link="$(readlink "$SELF")"
    case "$link" in
        /*) SELF="$link" ;;
        *)  SELF="$(dirname "$SELF")/$link" ;;
    esac
done
ROOT="$(cd "$(dirname "$SELF")" && pwd -P)" || die \
    "this program could not resolve the directory it lives in" \
    "Started as: ${BASH_SOURCE[0]}" \
    "This copy is incomplete. Unpack the download again."

[ "$(uname -m)" = "arm64" ] || die \
    "this program runs on Apple silicon and this Mac reports $(uname -m)" \
    "The solver and its Metal backend are built for arm64 and there is no" \
    "Intel build."

# THE PAYLOAD, NAMED PIECE BY PIECE. One message per missing piece, each
# naming the path, because "something is missing" sends a reader looking and a
# path tells them what they have. The interpreter is not checked here:
# PPF_CTS_VENV may replace it, so it is settled below once that is known.
for needed in \
    "$ROOT/config.sh" \
    "$ROOT/target/release/ppf-contact-solver" \
    "$ROOT/target/release/ppf-cts-server" \
    "$ROOT/target/release/lib_ppf_cts_py.dylib" \
    "$ROOT/bin/libppfbe_metal.dylib" \
    "$ROOT/bin/ppf_entries.metallib"; do
    [ -f "$needed" ] || die \
        "a file this program needs is missing" \
        "  $needed" \
        "The download is incomplete. Unpack it again."
done
for needed in "$ROOT/frontend" "$ROOT/examples"; do
    [ -d "$needed" ] || die \
        "a folder this program needs is missing" \
        "  $needed" \
        "The download is incomplete. Unpack it again."
done

# GATEKEEPER QUARANTINE, CLEARED HERE RATHER THAN ASKED OF THE USER.
#
# A browser marks what it downloads with com.apple.quarantine, and the
# unarchiver carries that mark onto every entry it extracts, the folder
# included. This payload is signed but not notarized, so Gatekeeper refuses a
# marked copy: the first Mach-O file that runs is refused, and a person who
# wanted to open a notebook meets a dialog about a program they downloaded on
# purpose.
#
# THIS FILE CAN CLEAR THE MARK BECAUSE A SHELL SCRIPT IS NOT ASSESSED.
# Gatekeeper acts on the exec of a Mach-O file, and this one is read by
# /bin/bash, which is the system's own. So it runs while marked, and it can
# clear the folder before anything here starts a binary. Both halves were
# measured on macOS 26.6: a marked script runs, and an ad-hoc signed binary
# whose mark was cleared runs.
#
# THE WHOLE FOLDER, NOT THE FEW BINARIES THIS FILE STARTS. Every Mach-O here
# is assessed when it is loaded, not only when it is started, and the
# interpreter alone brings hundreds of extension modules that are loaded.
# Clearing only what this script execs would get past this file and then fail
# at the first import.
#
# -s IS LOAD BEARING TWICE. Without it xattr follows a symbolic link, which
# leaves the link's own mark in place and reaches whatever it points at, and
# a link pointing nowhere is an error that would end this script under set -e.
# Measured on a tree carrying both: with -s the walk is silent, exits 0,
# clears the links themselves and touches nothing outside this folder.
#
# THE WALK IS WHAT DECIDES WHETHER TO SAY ANYTHING, so a folder with no mark
# on it costs one pass over the folder and prints nothing. If find cannot
# answer, nothing is cleared and nothing is claimed: the interpreter check
# further down is the backstop, and it names both ways through by hand.
#
# WHAT THIS DOES NOT DO is decide for someone who has not decided. It runs
# because they started this program, it clears the mark on THIS folder only,
# and it says so.
if [ -n "$(find "$ROOT" -xattrname com.apple.quarantine -print -quit 2>/dev/null)" ]; then
    printf 'This copy was downloaded through a browser, so macOS marked it\n'
    printf 'quarantined. Clearing that mark on this folder so it can start.\n'
    xattr -s -d -r com.apple.quarantine "$ROOT" 2>/dev/null || true
    if [ -n "$(find "$ROOT" -xattrname com.apple.quarantine -print -quit 2>/dev/null)" ]; then
        printf 'Part of the mark could not be cleared, so this may still refuse to\n' >&2
        printf 'start. A folder owned by another user, or on a read-only volume, is\n' >&2
        printf 'the usual cause. Move it somewhere you own and run it again.\n' >&2
    fi
fi

# shellcheck source=/dev/null
. "$ROOT/config.sh" || die \
    "the settings file would not load" \
    "  $ROOT/config.sh" \
    "If you edited it, an unbalanced quote is the usual cause. Delete your" \
    "edit, or unpack the download again to get the shipped file back."

# PORT is config.sh's. An edited file that dropped the line would otherwise end
# this process without a word, because set -u treats the unset name as fatal.
[ -n "${PORT:-}" ] || die \
    "the settings file does not set a port" \
    "PORT is missing or empty in $ROOT/config.sh." \
    "Give it a number, for example:  PORT=8080"
printf '%s' "$PORT" | grep -Eq '^[0-9]+$' || die \
    "the settings file sets a port that is not a number: $PORT" \
    "PORT is read from $ROOT/config.sh and handed to the server." \
    "Give it a number, for example:  PORT=8080"

usage() {
    # THE DISTRIBUTION'S OWN PATH IS PART OF THE HELP, and it is what the
    # build's smoke test reads back: this file may be reached through a
    # symlink to itself or to the folder above it, so the one place that can
    # say where the payload really is is the program that just resolved it.
    printf '%s\n' "ZOZO's Contact Solver $PPF_DIST_VERSION"
    printf '%s\n' "$ROOT"
    printf '\n'
    printf '%s\n' 'Usage:'
    printf '%s\n' '    ./ppf-contact-solver          start JupyterLab and serve the examples'
    printf '%s\n' '    ./ppf-contact-solver --help   this message'
    printf '\n'
    printf '%s\n' "It runs in the foreground. JupyterLab's log appears in this terminal, and"
    printf '%s\n' 'Ctrl+C stops it.'
    printf '\n'
    printf '%s\n' 'Settings, both optional:'
    printf '%s\n' "    config.sh      PORT is the port JupyterLab is asked for (currently $PORT)."
    printf '%s\n' '                   If that port is taken, the server takes the next free one.'
    printf '%s\n' '    PPF_CTS_VENV   an alternative Python environment, for a developer who'
    printf '%s\n' '                   wants their own packages instead of the ones in python/.'
    printf '\n'
    printf '%s\n' 'Notebooks are served from examples/ in this folder and save in place. Sessions,'
    printf '%s\n' 'cached assets, the Metal pipeline cache and the Jupyter state are kept in this'
    printf '%s\n' 'folder too, under local/ and cache/, so removing the folder removes all of them.'
}

# --help IS THE CHEAP PROBE AND MUST STAY CHEAP. It returns before any
# interpreter is started, which is what lets the build and CI run this file on
# a machine that will not serve notebooks and still learn that it parses, finds
# its own directory and reads its settings. What it costs is one walk of this
# folder, from the quarantine check above, which is deliberately on this side
# of the argument parsing: a person whose first move is to ask for help gets a
# folder that works afterward, and CI can gate the clearing without starting a
# server. It still writes nothing.
if [ "$#" -gt 0 ]; then
    case "$1" in
        --help | -h)
            usage
            exit 0
            ;;
        *)
            printf 'ERROR: this program takes no arguments, and got: %s\n' "$1" >&2
            printf '       The port is set in %s.\n' "$ROOT/config.sh" >&2
            printf '       An alternative Python environment is set in PPF_CTS_VENV.\n' >&2
            printf '       Run  ./ppf-contact-solver --help  for both.\n' >&2
            exit 2
            ;;
    esac
fi

version_ge() {
    # Is $1 at least $2, both dotted version numbers? Field by field, numeric,
    # with a missing field read as zero. `sort -V` is not something to rest
    # this on: its presence in BSD sort is not a given.
    awk -v have="$1" -v want="$2" '
        BEGIN {
            n = split(have, a, "."); m = split(want, b, ".")
            k = (n > m ? n : m)
            for (i = 1; i <= k; i++) {
                x = (i <= n ? a[i] + 0 : 0); y = (i <= m ? b[i] + 0 : 0)
                if (x > y) exit 0
                if (x < y) exit 1
            }
            exit 0
        }'
}

# THE macOS VERSION, CHECKED HERE BECAUSE NOTHING ELSE CHECKS IT. A folder has
# no property list for the system to read a minimum out of, so an older macOS
# would otherwise reach dyld and fail with a message naming a missing symbol,
# which reads as a corrupt download rather than as an old operating system.
#
# A CHECK THAT CANNOT BE MADE DOES NOT STOP A MACHINE THAT WOULD HAVE WORKED.
# If sw_vers answers with something that does not start with a digit, this says
# so in one line and carries on.
HAVE_MACOS="$(sw_vers -productVersion 2>/dev/null || true)"
case "$HAVE_MACOS" in
    [0-9]*)
        version_ge "$HAVE_MACOS" "$PPF_MIN_MACOS" || die \
            "this Mac runs macOS $HAVE_MACOS and this program needs $PPF_MIN_MACOS or newer" \
            "The minimum is read off the binaries in this folder, so it is what" \
            "they were built to require rather than a preference."
        ;;
    *)
        printf 'Could not read this macOS version, so the minimum (%s) was not checked.\n' \
            "$PPF_MIN_MACOS"
        ;;
esac

# THE INTERPRETER. The override wins and the shipped one is the default.
# config.sh sets PPF_CTS_VENV to the empty string when it is unset, so the test
# has to be -n: a ${VAR+x} test would take the override branch with an empty
# value and end up at /bin/python.
#
# The two paths differ in spelling for a reason. A venv always has bin/python;
# a python-build-standalone install tree is addressed through bin/python3, a
# relative symlink to the versioned binary.
if [ -n "${PPF_CTS_VENV:-}" ]; then
    PY="$PPF_CTS_VENV/bin/python"
    PY_NOTE="  (PPF_CTS_VENV override)"
    [ -x "$PY" ] || die \
        "PPF_CTS_VENV names an environment with no interpreter in it" \
        "  PPF_CTS_VENV: $PPF_CTS_VENV" \
        "  expected:     $PY" \
        "This refuses rather than falling back to the interpreter in python/," \
        "because a silent fallback would run a different environment than the" \
        "one you asked for. Unset PPF_CTS_VENV to use the shipped one."
else
    PY="$ROOT/python/bin/python3"
    PY_NOTE=""
    [ -x "$PY" ] || die \
        "this distribution has no Python interpreter" \
        "  expected: $PY" \
        "It ships one, so this copy is incomplete. Unpack the download again."
fi
"$PY" -c 'pass' >/dev/null 2>&1 || die \
    "the Python interpreter would not start" \
    "  $PY" \
    "This program clears the macOS download mark on its own folder before it" \
    "reaches here, so Gatekeeper is an unlikely cause unless that clearing" \
    "printed a warning above. If it did, part of this folder is still marked:" \
    "open System Settings > Privacy and Security, find the message naming" \
    "this program, and click Open Anyway; or clear it by hand with:" \
    "  xattr -s -d -r com.apple.quarantine \"$ROOT\"" \
    "If it did not, this copy is incomplete or was built for another system."

# EVERYTHING THIS PROGRAM WRITES LIVES INSIDE THE FOLDER IT WAS UNPACKED
# INTO, which is what makes removing that folder a complete uninstall. It is
# the property the Windows distribution has always had, and the reason is the
# same: a person who downloaded a folder should not have to learn where else a
# program put things in order to be rid of it.
#
# THE MARKER IS WHAT MAKES THE OTHER HALF AGREE. The launcher can set the
# Jupyter and IPython directories itself, but the SESSION data root and the
# asset cache are resolved inside the frontend, and the Blender add-on spawns
# this distribution's ppf-cts-server directly without going through this
# script. A file at the tree root is the same answer to every entry point,
# which an environment variable set here would not be.
STATE="$ROOT/local/share/ppf-cts"
NBDIR="$ROOT/examples"
mkdir -p "$STATE" || die \
    "could not create the state folder" \
    "  $STATE" \
    "This program keeps its state inside its own folder, so it needs that" \
    "folder to be writable. A distribution unpacked into /Applications or" \
    "another read-only location lands here; move it somewhere you own, such" \
    "as your home directory or Downloads, and run it again."

# unset PYTHONHOME       a relocatable interpreter computes its home from its
#                        own path, and an inherited PYTHONHOME overrides that
#                        and hides the stdlib
# unset CARGO_TARGET_DIR frontend/__init__.py's _target_dirs() honors it and
#                        searches ONLY it, never <tree root>/target, so a
#                        developer who exports it in their shell would have the
#                        frontend look for the cdylib somewhere else and either
#                        fail or load a foreign build. This program runs in the
#                        user's own shell, so that variable arrives here.
# unset PYTHONOPTIMIZE   it redirects every import to a .opt-N.pyc, which the
#                        build did not write, so with PYTHONDONTWRITEBYTECODE
#                        set the whole import graph recompiles on every start
#                        and nothing is stored. It presents as a program that
#                        is simply slow.
# PYTHONNOUSERSITE       otherwise ~/.local/lib/python3.X/site-packages joins
#                        sys.path and a user's own numpy shadows the pinned one
# PYTHONDONTWRITEBYTECODE  this program writes nothing inside the distribution,
#                        and the build pre-compiled every .pyc so nothing is
#                        lost
# PYTHONPATH             assigned, not appended. A PYTHONPATH shadowing
#                        frontend, numpy or scipy changes what the solver
#                        computes, and this project treats a frontend package
#                        difference as a physics variable rather than a
#                        preference. Someone who wants their own packages uses
#                        PPF_CTS_VENV.
# PPF_CTS_BUILD_PYTHON   the build worker's interpreter. This is checked ahead
#                        of VIRTUAL_ENV, and without it the worker falls
#                        through to a bare python3 on PATH, which on a stock
#                        Mac is 3.9 and cannot parse the frontend at all.
# JUPYTER_PATH           searched before every other location, and pointed at
#                        the prefix of the interpreter that will run, so the
#                        kernel found is that one rather than a stray
#                        kernelspec on this Mac naming another Python.
# PATH                   set at the top of this file rather than here, because
#                        the checks above resolve commands through it. The
#                        reasoning is at that line.
unset PYTHONHOME
unset CARGO_TARGET_DIR
unset PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT"
export PPF_CTS_BUILD_PYTHON="$PY"
export JUPYTER_CONFIG_DIR="$STATE/jupyter/config"
export JUPYTER_DATA_DIR="$STATE/jupyter/data"
export JUPYTER_RUNTIME_DIR="$STATE/jupyter/runtime"
export IPYTHONDIR="$STATE/jupyter/ipython"
if [ -n "${PPF_CTS_VENV:-}" ]; then
    export JUPYTER_PATH="$PPF_CTS_VENV/share/jupyter"
else
    export JUPYTER_PATH="$ROOT/python/share/jupyter"
fi
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" \
    "$IPYTHONDIR" || die \
    "could not create the Jupyter folders" \
    "  $STATE/jupyter" \
    "This program keeps its state inside its own folder, so it needs that" \
    "folder to be writable and to have room."

# THE NOTEBOOKS ARE SERVED IN PLACE, from the distribution's own examples
# folder, which is what the Windows launcher does with --notebook-dir.
#
# COPYING THEM OUT TO A DIRECTORY UNDER $HOME WOULD BUY NOTHING HERE. The
# reason to copy is that a distribution folder "may sit somewhere that is not
# writable" and a notebook saves itself when it runs, and the check above
# answers that reason: this program needs its own folder writable for every
# other thing it stores, so a read-only location fails there, with a message
# saying so, instead of quietly working while leaving state behind in two
# places.
#
# AND A COPY COSTS TWO PROPERTIES. A seeded directory would never be re-seeded,
# so a user who kept one across upgrades would run the OLD notebooks against a
# NEW solver with nothing saying so; and edits made there would survive
# deleting the distribution, which is the opposite of the property this layout
# exists to give.
[ -d "$NBDIR" ] || die \
    "the example notebooks are missing from this distribution" \
    "  $NBDIR" \
    "The download is incomplete. Unpack it again."

printf '\n'
printf "ZOZO's Contact Solver %s\n" "$PPF_DIST_VERSION"
printf '  distribution  %s\n' "$ROOT"
printf '  interpreter   %s%s\n' "$PY" "$PY_NOTE"
printf '  notebooks     %s\n' "$NBDIR"
printf '  state         %s\n' "$STATE"
printf '  URL           http://localhost:%s/lab\n' "$PORT"
printf '\n'
printf 'Starting JupyterLab. Press Ctrl+C to stop it.\n'
printf 'If port %s is taken the server takes the next free one, and the exact URL\n' "$PORT"
printf 'is in the log below.\n'
printf '\n'

SERVER_PID=""
SHUTDOWN_REQUESTED=0

stop_server() {
    # IDEMPOTENT, AND INSTALLED ON TWO TRAPS. A signal runs it, and the EXIT
    # trap runs it again on the way out; the second call returns at once. One
    # shutdown path, reached from two places.
    [ -n "$SERVER_PID" ] || return 0
    kill -0 "$SERVER_PID" 2>/dev/null || return 0
    # SIGTERM FIRST. jupyter_server handles it by shutting its kernels down
    # through cleanup_kernels() and then exiting, and that is what ends a
    # kernel: NOT the group form below.
    #
    # THE GROUP FORM DOES NOT REACH A KERNEL, AND IT IS A MISTAKE TO THINK IT
    # DOES. jupyter_client launches every kernel with start_new_session=True,
    # which puts it in a session and a process group of its own before it runs,
    # so a signal to the server's group reaches the server and nothing else.
    # The group form is kept because it costs nothing and covers a non-jupyter
    # child that stays in the group; it is defensive, not the mechanism.
    kill -s TERM -- "-$SERVER_PID" 2>/dev/null \
        || kill -s TERM "$SERVER_PID" 2>/dev/null || true
    # THE GRACE PERIOD IS HAND-ROLLED BECAUSE macOS SHIPS NO GNU timeout.
    # Wrapping the wait in one exits 127, which reads as the wrapped command
    # failing rather than as a missing utility.
    local n=0
    while [ "$n" -lt 40 ]; do
        kill -0 "$SERVER_PID" 2>/dev/null || return 0
        sleep 0.25
        n=$((n + 1))
    done
    # A CHILD THAT IGNORES THE FIRST SIGNAL IS KILLED AFTER TEN SECONDS, AND
    # THIS IS THE ONE PATH THAT CAN LEAVE A KERNEL BEHIND. SIGKILL cannot be
    # handled, so the server does not get to run cleanup_kernels(), and the
    # kernels are in their own process groups where neither signal reaches
    # them. What reclaims an ipykernel is its own ParentPollerUnix, which
    # notices the pid in JPY_PARENT_PID has gone and exits; a kernel that is
    # not ipykernel has no such poller and is left running. This program ends
    # only what it started, and anything left is a plain process that can be
    # ended from Activity Monitor or with kill.
    printf 'JupyterLab did not exit within 10 seconds. Killing it.\n' >&2
    kill -s KILL -- "-$SERVER_PID" 2>/dev/null \
        || kill -s KILL "$SERVER_PID" 2>/dev/null || true
}

on_signal() {
    SHUTDOWN_REQUESTED=1
    printf '\nStopping JupyterLab...\n'
    stop_server
}

# INSTALLED BEFORE THE LAUNCH. SERVER_PID is empty until the launch returns and
# stop_server returns at once on an empty pid, so installing early costs
# nothing and covers a signal that arrives during startup. HUP is trapped
# alongside INT and TERM so that closing the terminal window stops the server
# rather than leaving it running.
trap on_signal INT TERM HUP
trap stop_server EXIT

# JOB CONTROL IS ON SO THE SERVER GETS A PROCESS GROUP OF ITS OWN, which is
# what lets one signal reach the server AND the kernels it started. Without it
# the server and its kernels sit in this shell's own group, there is no group
# to signal, and a killed server leaves its kernels behind.
#
# THE LAUNCHER SIGNALS RATHER THAN RELYING ON THE TERMINAL. A background child
# of a non-interactive shell has SIGINT ignored, and with job control on the
# child is not in the foreground process group at all, so the tty's Ctrl+C
# reaches this script and nothing else. Sending SIGTERM from here is therefore
# the only path, and it is the same path a kill from another terminal takes.
#
# Bash may print a job notice of its own, a line reading [1] and a number,
# around the launch or the kill. That is bash talking, it is cosmetic, and
# nothing suppresses it.
#
# --ServerApp.port rather than --port so Jupyter's own retry applies: a taken
# port becomes the next free one, and the real URL is in the log below.
# --ServerApp.root_dir rather than --notebook-dir so that every flag here is a
# trait name, which is what lets the build check all of them in one place
# against ServerApp.class_traits() and LabApp.class_traits().
# --ServerApp.answer_yes so that a SIGINT which does reach the server shuts it
# down rather than prompting on a stdin the server does not own.
# --ServerApp.open_browser hands the opening to JupyterLab, which logs a
# failure and keeps serving, so opening a browser is a convenience and never a
# requirement. Opening it here instead would need a readiness probe, a JSON
# parse of the server's runtime record and a poll loop, all to learn a URL the
# server prints to this terminal anyway.
#
# THE TWO LabApp FLAGS ARE WHAT MAKE THE OFFLINE CLAIM TRUE. Left at their
# defaults, the Lab interface fetches two things of its own as soon as it
# loads, with no notebook involved: a news feed from jupyterlab.github.io, and
# a version check against pypi.org. Neither is anything this distribution
# needs, and a machine with no route out waits on both. news_url is
# allow_none and the interface tests it with `is None`, so an empty string
# does not disable it and the literal None does; traitlets maps the word on
# the command line to the value. The update check is a class, and JupyterLab
# ships NeverCheckForUpdate for exactly this.
set -m
"$PY" -m jupyterlab \
    --ServerApp.port="$PORT" \
    --ServerApp.port_retries=50 \
    --ServerApp.token="" \
    --ServerApp.root_dir="$NBDIR" \
    --ServerApp.open_browser=True \
    --ServerApp.answer_yes=True \
    --LabApp.news_url=None \
    --LabApp.check_for_updates_class=jupyterlab.handlers.announcements.NeverCheckForUpdate &
SERVER_PID=$!
# A SIGNAL BETWEEN THE trap AND THIS ASSIGNMENT WOULD LEAVE A SERVER NOBODY
# ASKED TO STOP. `on_signal` runs, sets SHUTDOWN_REQUESTED and returns from
# `stop_server` at once because the pid is still empty; execution then resumes
# here and blocks in `wait` forever. The window is microseconds and this closes
# it: if the request already arrived, honor it now that there is a pid to act on.
[ "$SHUTDOWN_REQUESTED" -eq 0 ] || stop_server

rc=0
wait "$SERVER_PID" || rc=$?
if [ "$SHUTDOWN_REQUESTED" -eq 1 ]; then
    # A TRAP INTERRUPTING wait MAKES IT RETURN 128 PLUS THE SIGNAL rather than
    # the child's status, and the handler has already asked the server to stop,
    # so the real status is collected by waiting again. A stop the user asked
    # for is not a failure, so this exits 0; a server that fails on its own
    # passes its status through below.
    wait "$SERVER_PID" 2>/dev/null || true
    printf 'JupyterLab stopped.\n'
    exit 0
fi
if [ "$rc" -ne 0 ]; then
    printf 'ERROR: JupyterLab exited with status %s. Its output is above.\n' "$rc" >&2
    exit "$rc"
fi
printf 'JupyterLab stopped.\n'
exit 0
LAUNCHER_EOF
} > "$PKG/ppf-contact-solver"
chmod 755 "$PKG/ppf-contact-solver"

# A SYNTAX ERROR IN THE ONE THING THE USER RUNS MUST FAIL THIS BUILD RATHER
# THAN THEIR FIRST RUN, so the generated file is parsed immediately.
bash -n "$PKG/ppf-contact-solver" || die \
    "the generated launcher does not parse" \
    "bash -n's own message is above, and it names the line in" \
    "$PKG/ppf-contact-solver. That file is generated by this script, so the" \
    "defect is in the heredoc here rather than in the payload."
printf '  [OK] launcher\n'

# THE SHIPPED config.sh IS GENERATED, NOT COPIED, AND THAT IS THE FIX RATHER
# THAN A STYLE CHOICE.
#
# build-mac-native/config.sh exists to be edited locally, and its own header
# says so. Four of its settings name this machine: PPF_CTS_VENV and
# PPF_CTS_PYTHON are absolute paths to a developer's interpreter, and
# MAC_CODESIGN_IDENTITY and MAC_NOTARY_KEYCHAIN_PROFILE are keychain names.
# Copying that file verbatim ships every one of them.
#
# The consequence is worst for PPF_CTS_VENV, because the launcher acts on it: a
# builder who pointed it at their own environment would ship a distribution
# that takes the override branch on the user's Mac and refuses, naming a
# directory that does not exist there. GATE C CANNOT SEE THAT, because such a
# path need not live under this build tree, so no check downstream would catch
# it.
#
# Generating the file removes the class rather than testing for it. What the
# launcher reads is exactly two settings, so those two are what it carries:
# PORT and PPF_CTS_VENV. The signing settings are build-time only and nothing
# in the distribution looks at them.
#
# The generated file and the launcher that sources it are written by this one
# step, so a setting added to one moves with the other.
printf '%s' "$PORT" | grep -Eq '^[0-9]+$' || die \
    "PORT is not a number: $PORT" \
    "It is written into the shipped config.sh and handed to the server as" \
    "--ServerApp.port, so a value that is not a port would fail on the user's" \
    "machine rather than here. It comes from config.sh or from the" \
    "environment of this run."
cat > "$PKG/config.sh" <<EOF
#!/usr/bin/env bash
# Generated by build-mac-native/bundle.sh. This is the distribution's copy and
# is not the build tree's config.sh, which carries settings that name the
# machine this was built on and have no meaning here.
#
# Source it, do not execute it. There is no \`set -euo pipefail\`, because
# those options would leak into the shell of whoever sources it.
#
# Each setting reads an already-exported value first, so either can be
# overridden for a single run without editing this file:
#
#     PORT=9000 ./ppf-contact-solver

# The port JupyterLab is asked to listen on. If it is already taken on this
# machine the server takes the next free one and prints the real URL, so this
# is a preference rather than a requirement.
PORT="\${PORT:-$PORT}"

# An alternative Python environment, for a developer who wants to run the
# frontend from their own packages instead of the ones in python/. Empty means
# the interpreter that ships here is used, which is what an ordinary install
# wants. The launcher honors it, and refuses rather than falling back when it
# names an environment with no interpreter in it.
PPF_CTS_VENV="\${PPF_CTS_VENV:-}"
EOF

# Nothing above interpolates a path, so this can only fail if the file was
# edited to interpolate one. It is asserted anyway, because the whole reason
# this file is generated is that gate C cannot see a path outside the build
# tree, and an assertion here is the only place that can.
if grep -Eq '^[^#]*=[^#]*(/Users/|/home/|/Volumes/)' "$PKG/config.sh"; then
    grep -En '^[^#]*=[^#]*(/Users/|/home/|/Volumes/)' "$PKG/config.sh" >&2
    die \
        "the generated config.sh assigns an absolute path" \
        "The lines above name a directory on the machine that built this," \
        "which will not exist on the machine that runs it. Gate C sees only" \
        "paths under the build tree, so this is the only check that would" \
        "catch it."
fi
printf '  [OK] config.sh (PORT %s)\n' "$PORT"

cat > "$PKG/README.txt" <<EOF
============================================================
ZOZO's Contact Solver, macOS distribution
============================================================

QUICK START
-----------
Open Terminal, change into this folder, and run the launcher:

    cd /path/to/ppf-contact-solver
    ./ppf-contact-solver

Dragging the folder from Finder onto the Terminal window after typing "cd "
fills the path in for you.

JupyterLab starts and its log appears in that terminal. A browser opens at
the URL; if it does not, the URL is in the log. Pick a notebook and run it.
Ctrl+C in that terminal stops the server.

The first run of a copy downloaded through a browser needs one extra step,
described under IF IT WILL NOT START below.

REQUIREMENTS
------------
- A Mac with Apple silicon. The solver and its Metal backend are built for
  arm64 and there is no Intel build.
- macOS $MIN_OS or newer. That number is read off the binaries in this folder,
  so it is what they were built to require rather than a preference, and the
  launcher checks it and refuses by name on an older one.
- Nothing else to start it. This folder carries its own Python interpreter
  and every package the frontend needs, so no Python, no Homebrew, no Xcode
  and no command line tools are needed to run it and open a notebook. It
  installs nothing on your Mac, adds nothing to your PATH, and reaches the
  network for nothing of its own. The two things the JupyterLab interface
  would otherwise fetch as it loads, a news feed and a check for a newer
  version, are turned off at the launch line, so a Mac with no route out
  waits on neither.

IF IT WILL NOT START
--------------------
"Permission denied" means the executable bit did not survive the way this
folder was unpacked. Restore it:

    chmod +x ppf-contact-solver

Gatekeeper quarantine needs nothing from you. macOS marks what a browser
downloads, and this program clears that mark on its own folder the first time
you run it, saying so in one line before it starts. It clears this folder and
nothing else.

If that clearing prints a warning, part of the folder stayed marked and the
program may still refuse to start. That happens on a folder owned by another
user or on a read-only volume. Move it somewhere you own and run it again, or
go through Gatekeeper by hand:

  - System Settings > Privacy and Security, find the message naming this
    program, and click Open Anyway. Later runs work normally.
  - Or clear the mark yourself:

        xattr -s -d -r com.apple.quarantine "path/to/this/folder"

The arm64 binaries in this folder each carry their own signature. The rest,
the .metallib files, the Python sources, this file, config.sh and the
launcher, carry none: a plain folder has nothing to hold a signature over its
whole contents, and a signature on a file that is not a binary is stored in
an extended attribute that does not survive every way a folder is unpacked.

CONFIGURATION
-------------
config.sh sets the port JupyterLab is asked for:

    PORT="\${PORT:-$PORT}"

If that port is taken, the server takes the next free one and the real URL
is in the log. To change it for one run without editing the file:

    PORT=9000 ./ppf-contact-solver

PPF_CTS_VENV names an alternative Python environment, for a developer who
wants their own packages instead of the ones in python/:

    PPF_CTS_VENV=/path/to/venv ./ppf-contact-solver

Set to an environment with no interpreter in it, the launcher refuses rather
than quietly using the one in python/.

STOPPING IT
-----------
Ctrl+C in the terminal running it. That asks JupyterLab to shut down, which
ends its kernels too, waits ten seconds, and then kills it if it has not
gone. A kernel outlives that last case: a killed server does not get to shut
its kernels down, and a Python kernel notices on its own and exits shortly
after, while any other kind is left for you to end from Activity Monitor.

Ctrl+Z suspends the launcher and not JupyterLab, which keeps running and
keeps writing to the terminal. Use Ctrl+C to stop it.

File > Shut Down inside JupyterLab also ends the server, and the launcher
exits with it.

Closing the browser tab does not stop the server: the tab is a client and
the terminal is the program.

WHAT SOME EXAMPLES NEED, WHICH THIS PROGRAM DOES NOT
---------------------------------------------------
This program and the notebooks shipped with it are separate questions, and
this is the one place the answers differ.

Several examples fetch the mesh they simulate the first time you run them,
into cache/ppf-cts inside this folder. Four of them clone a repository as well, and need the
git command to do it: large-fluffy, large-animals, fitting and
trapped-919539a. macOS supplies git with the Xcode command line tools, so
running one of those four on a Mac that does not have them shows the
"install the command line developer tools" dialog. fishingknot also fetches
its mesh, over https with urllib rather than with git, so it needs neither. Those notebooks are the
only thing here that can raise it; the launcher itself never runs git, and
the other examples run offline.

Video export in a notebook uses ffmpeg, which is not shipped here either.
Without it the frontend skips the export and carries on.

BLENDER INTEGRATION
-------------------
The Blender addon is distributed separately. This folder ships the server
the addon connects to.

  1. Download and install the ppf-contact-solver Blender addon, following
     the addon's own install instructions.
  2. In Blender, select "macOS Native" from the server type dropdown.
  3. Set "Solver Path" to THIS folder, the one holding target/release.
  4. Click Connect.

WHAT IT WRITES, AND HOW TO REMOVE IT
------------------------------------
Everything it writes is inside this folder:

    local/share/ppf-cts/   sessions, and the Jupyter and IPython state
    cache/ppf-cts/         meshes and tetrahedralizations the examples fetch,
                           and the Metal pipeline cache, which is compiled for
                           this Mac's GPU on the first run
    examples/              the notebooks, which save in place

Removing this folder removes all of it. Nothing is installed anywhere else: no
system files, no LaunchAgent, no login item, and no change to your PATH or
shell profile.

CONTENTS
--------
ppf-contact-solver   the launcher, which is what you run
config.sh            the port, and the PPF_CTS_VENV override
README.txt           this file
THIRD_PARTY_LICENSES.txt
bin/                 the Metal backend (libppfbe_metal.dylib), the entry
                     library it loads (ppf_entries.metallib), and any
                     pre-compiled shader libraries
target/release/      the solver, the server, and the Python extension module
python/              the bundled interpreter and its packages
frontend/            the Python frontend package
examples/            the example notebooks seeded on the first run
crates/              crate source roots the frontend reads log-channel names
                     out of
.git/                one file, branch_name.txt, naming the branch this was
                     built from. It is what names your session folder, and it
                     is there so nothing here has to run git to find out. It
                     is not a repository and nothing clones it.

Two files here carry the name ppf-contact-solver and they are different
programs. The one at the top of this folder is the launcher, a shell script.
target/release/ppf-contact-solver is the solver itself, which the frontend
runs once per scene.

LICENSE
-------
See THIRD_PARTY_LICENSES.txt.
EOF
printf '  [OK] README.txt\n'

cat > "$PKG/THIRD_PARTY_LICENSES.txt" <<EOF
============================================================
THIRD PARTY LICENSES
============================================================

ZOZO's Contact Solver
---------------------
Copyright 2025 Ryoichi Ando (ZOZO, Inc.)
Licensed under the Apache License, Version 2.0


Apple frameworks
----------------
The backend links Metal.framework and Foundation.framework from the macOS
SDK. Neither is redistributed here: both are part of the operating system
and are resolved on the machine that runs this.


CPython
-------
python/ is a redistributed CPython, licensed under the Python Software
Foundation License Agreement. It is a python-build-standalone install_only
build, taken unmodified from:

  $FILE_PYTHON

That project's own license terms and the licenses of the third party
libraries compiled into that interpreter are in
python/lib/python*/site-packages and in the interpreter tree's own
documentation.


Python packages
---------------
INCOMPLETE. python/lib/python*/site-packages carries the frontend's
dependencies and JupyterLab, each under its own license. That set is
enumerated by

  python/bin/python3 -m pip list

and is not transcribed into this file yet. It must be before this
distribution is published outside the project.


Rust dependencies
-----------------
INCOMPLETE. The binaries in target/release statically link the crates in
the repository's Cargo.lock, each under its own license. That set is not
enumerated in this file yet, and it must be before this distribution is
published outside the project.
EOF
printf '  [OK] THIRD_PARTY_LICENSES.txt\n'

# ---------------------------------------------------------------------------
step "[9/13] Pre-compiling bytecode"
# ---------------------------------------------------------------------------

# The stdlib test package is already gone, pruned in step 7 so that the Mach-O
# walk and the deployment target it feeds see only what ships. That prune is
# also what removes the badsyntax_*.py files, which would otherwise make the
# compile below report a failure.
#
# Three things this does, each load-bearing:
#
#   -f with --invalidation-mode unchecked-hash makes every .pyc valid whatever
#   the source's mtime is. The default timestamp mode records the source's
#   mtime and size, and any extraction path that does not preserve mtime would
#   silently invalidate the whole cache, leaving the payload recompiling its
#   entire import graph on every run with PYTHONDONTWRITEBYTECODE set and
#   nowhere to put the result. That presents as a program that is simply slow.
#
#   -s "$PKG" strips the build-tree prefix from the path recorded inside
#   each .pyc. Without it every .pyc records an absolute path under this build
#   tree and gate C below reports thousands of hits that are not what it is
#   looking for. The recorded paths are relative afterwards and no longer
#   resolve; Python never opens them for import, and a traceback still recovers
#   source through the module's loader.
#
#   Together with PYTHONDONTWRITEBYTECODE in the launcher, nothing is written
#   inside the distribution, which is what keeps every Mach-O signature valid
#   wherever the user keeps the folder.
printf 'Pre-compiling bytecode...\n'

# COMPILED IN TWO PASSES SO A WARNING STILL MEANS SOMETHING. Byte-compiling
# reports a SyntaxWarning for every questionable construct it meets, and the
# vendored trees contain constructs this project did not write and cannot fix:
# a regex spelled "([^\.\/\\]+)\.py" in vtkmodules/web/testing.py and another
# in an ipywidgets test raise "invalid escape sequence" on every build. Those
# are true reports about someone else's source, they will not change until
# those projects release a fix, and left in the log they train a reader to
# scroll past the one place a real warning about OUR code would appear.
#
# So the vendored code is compiled with SyntaxWarning suppressed and the
# frontend is compiled without, in a second pass. THE SUPPRESSION IS SCOPED BY
# WHAT IT COVERS RATHER THAN BY WHICH WARNING IS CURRENTLY NOISY: a new warning
# in frontend/ still fails to be quiet, which is the property worth keeping.
compile_tree() {
    # $1 is a -W action for the interpreter, the rest are roots.
    local action="$1"
    shift
    "$PKG_PY" -W "$action" -m compileall -q -f \
        --invalidation-mode unchecked-hash -s "$PKG" "$@" || die \
        "compileall failed" \
        "It names the file it could not compile above. Add an -x for that one" \
        "file to the command in bundle.sh, with a comment saying why it cannot" \
        "compile. Do not drop the exit-status check: a payload that ships a" \
        "half-compiled cache recompiles the rest on every run with nowhere to" \
        "store the result."
}
compile_tree ignore::SyntaxWarning "$STDLIB" "$SITE"
compile_tree default "$PKG/frontend"
printf '  [OK] bytecode compiled\n'

# ---------------------------------------------------------------------------
step "[10/13] Verifying the distribution is self-contained"
# ---------------------------------------------------------------------------

# The Mach-O list, discovered again now that the prune has run, so nothing on
# it has since been deleted. It is what the gates below walk and what step 12
# signs, and it is the set that actually ships.
ALL_MACHO=()
while IFS= read -r found; do
    [ -n "$found" ] || continue
    ALL_MACHO+=("$found")
done < <(discover_macho)
[ "${#ALL_MACHO[@]}" -gt 0 ] || die \
    "the Mach-O discovery found nothing in the payload" \
    "That is a failure of the discovery itself, not a clean payload: the" \
    "distribution certainly contains a solver, a backend and an interpreter." \
    "A gate reporting clean over nothing is the mistake this refuses to make."
printf 'Walking %d Mach-O files\n' "${#ALL_MACHO[@]}"
ALL_MACHO_LIST="$(printf '%s\n' "${ALL_MACHO[@]}")"
assert_walk_complete "$ALL_MACHO_LIST"

# GATE D. Every Mach-O reports arm64 and nothing else. Step 7 made that true
# and this asks again over the final list, because this is the set that ships
# and a guarantee is worth making where the payload stops changing. A file that
# was added after the thinning pass, or one the earlier walk did not reach,
# fails here.
assert_arm64_only "$ALL_MACHO_LIST"
printf '  [OK] [gate D] every Mach-O reports arm64 and nothing else\n'

# GATE A. No load command and no rpath may name the tree this was built from. A
# reference that survives makes the payload load the build tree's library on
# this machine, so it would pass every test here and fail on the first machine
# that is not this one.
#
# GATE B. No absolute DEPENDENCY outside the operating system. @rpath,
# @loader_path and @executable_path are fine; an absolute path is fine only
# under /usr/lib or /System/Library. This is the gate that catches "works on
# the machine that built it, fails on the user's": a wheel that linked the
# builder's Homebrew libgfortran passes gate A and fails on every other Mac. A
# third allowed prefix gets added deliberately, with a comment saying why that
# path is on every supported macOS.
#
# GATE B ASKS ONLY ABOUT DEPENDENCIES, AND THE TWO THINGS IT DELIBERATELY DOES
# NOT ASK ABOUT ARE WHERE IT WOULD OTHERWISE CONDEMN WORKING WHEELS.
#   An INSTALL NAME is the file's own LC_ID_DYLIB. dyld loads a library from
#   the path it was found at, so an id naming the publisher's build machine is
#   inert. Pillow ships every one of its bundled dylibs with a /DLC/PIL/ id,
#   which is delocate reserving room to rewrite the load commands later, and
#   debugpy's attach.dylib is two thin builds lipoed together so it carries one
#   id per slice from debugpy's own CI. All of them work on every Mac.
#   An RPATH is a place to look, not a thing to find. A stale absolute rpath is
#   skipped by dyld when it does not exist, so it costs nothing.
# Both are still read, because gate A must see them: an id or an rpath naming
# THIS build tree is fatal exactly as a dependency would be, and that is a
# different question from whether the path is outside the OS.
escaped=0
for macho in "${ALL_MACHO[@]}"; do
    name="${macho#"$PKG/"}"
    macho_deps="$(deps_of "$macho")"
    macho_ids="$(ids_of "$macho")"
    macho_rpaths="$(rpaths_of "$macho")"

    # GATE A, over every path the file names in any load command.
    while IFS= read -r ref; do
        [ -n "$ref" ] || continue
        case "$ref" in
            "$SRC_DIR"*)
                printf 'ERROR: [gate A] %s references the build tree: %s\n' "$name" "$ref" >&2
                escaped=1
                ;;
        esac
    done <<EOF
$macho_deps
$macho_ids
$macho_rpaths
EOF

    # GATE B, over the dependencies alone.
    while IFS= read -r ref; do
        [ -n "$ref" ] || continue
        case "$ref" in
            @rpath/* | @loader_path/* | @executable_path/*) ;;
            /usr/lib | /usr/lib/* | /System/Library | /System/Library/*) ;;
            /*)
                printf 'ERROR: [gate B] %s depends on a path outside the OS: %s\n' "$name" "$ref" >&2
                escaped=1
                ;;
        esac
    done <<EOF
$macho_deps
EOF
done

# GATE C. No file anywhere in the distribution names the build tree. This
# catches what a
# load-command walk cannot: an absolute shebang under python/bin, a baked path
# in a kernel.json, a direct_url.json, a stray config, and a path compiled into
# a binary and resolved at run time rather than by the loader. It takes tens of
# seconds over a payload this size and that cost is accepted deliberately.
#
# Step 9's -s is what keeps this quiet on .pyc files, so there is no exclusion
# for them: if it fires on a .pyc, that -s was dropped.
#
# ONE FILE IS ALLOWED, BY NAME AND FOR ONE STRING. crates/ppf-cts-py/src/lib.rs
# publishes env!("CARGO_MANIFEST_DIR") as __build_manifest_dir__, a provenance
# stamp. Nothing resolves it as a path: frontend/__init__.py says in its own
# comment that comparing it would wrongly reject a packaged tree, and it
# does not compare it. The allowance is per file and per string rather than a
# pattern, so the file is admitted only while that stamp is the ONLY build-tree
# string in it, and any second one fails here by name.
# TWO FILES, ONE PER BACKEND, because the distribution ships a cdylib beside
# each solver and both carry the same stamp from the same crate. Still a LIST
# OF EXACT PATHS rather than a pattern: a glob over target/*/release would
# admit whatever else landed there, and the point of naming them is that a
# third file carrying a build-tree string fails here rather than being covered
# by a rule written for these two.
GATE_C_ALLOWED=(
    "$PKG/target/release/lib_ppf_cts_py.dylib"
    "$PKG/target/cpu/release/lib_ppf_cts_py.dylib"
)
GATE_C_STAMP="$SRC_DIR/crates/ppf-cts-py"

# THE SCAN IS RUN ONCE AND ITS EXIT STATUS IS READ, RATHER THAN PIPED THROUGH
# `|| true`. grep exits 1 for "no match" and 2 for "I could not read that", and
# collapsing the two makes an unreadable payload look like a clean one. This is
# the one gate that catches an absolute shebang or a baked path, so a gate that
# went blind here would certify nothing while printing [OK]. Its own stderr is
# captured separately for the same reason: a warning line mixed into the hit
# list would be read as a filename.
printf 'Scanning every file for the build-tree path...\n'
GATE_C_ERR="$(mktemp "${TMPDIR:-/tmp}/ppf-gate-c.XXXXXX")"
gate_c_rc=0
GATE_C_HITS="$(grep -rlF -- "$SRC_DIR" "$PKG" 2>"$GATE_C_ERR")" || gate_c_rc=$?
if [ "$gate_c_rc" -gt 1 ] || [ -s "$GATE_C_ERR" ]; then
    sed 's/^/       /' "$GATE_C_ERR" >&2
    rm -f "$GATE_C_ERR"
    die \
        "gate C could not read the whole distribution (grep exited $gate_c_rc)" \
        "grep's own output is above. Whatever it could not read was not" \
        "scanned, so this run has established nothing about the payload and" \
        "nothing was signed." \
        "Fix what grep is complaining about. Silencing it would turn this" \
        "gate into one that passes over the files it cannot see, which is" \
        "worse than the red build it replaces."
fi
rm -f "$GATE_C_ERR"

while IFS= read -r hit; do
    [ -n "$hit" ] || continue
    hit_is_allowed=0
    for allowed in "${GATE_C_ALLOWED[@]}"; do
        [ "$hit" = "$allowed" ] && hit_is_allowed=1 && break
    done
    if [ "$hit_is_allowed" -eq 1 ]; then
        # Admitted only if every build-tree string it carries is the stamp.
        #
        # THE COMPARISON COUNTS OCCURRENCES RATHER THAN MATCHING WHOLE LINES.
        # A Rust binary lays its string literals down end to end with no
        # separator between them, so `strings` returns one long run with the
        # manifest directory glued to whatever the compiler placed beside it,
        # and no line of that output is ever EQUAL to the stamp. A line
        # compare therefore rejects the one file the allowance was written to
        # admit. What is checked instead is that the build tree occurs exactly
        # as often as the ppf-cts-py manifest directory occurs, which states
        # the allowance exactly: that manifest directory is the ONLY
        # build-tree path in the file.
        #
        # The count is checked the same way the scan above is: this file
        # reached the list because it CONTAINS the build-tree path, so a zero
        # count means strings could not see what grep saw, and admitting the
        # file on that basis would be the allowance granting itself.
        hit_strings="$(strings -a "$hit")"
        total_refs="$(printf '%s\n' "$hit_strings" | count_occurrences "$SRC_DIR")"
        stamp_refs="$(printf '%s\n' "$hit_strings" | count_occurrences "$GATE_C_STAMP")"
        if [ "$total_refs" -eq 0 ]; then
            printf 'ERROR: [gate C] %s matched the build-tree path but no string\n' \
                "${hit#"$PKG/"}" >&2
            printf '       could be extracted from it. The allowance admits this\n' >&2
            printf '       file only for one known string, and that string cannot be\n' >&2
            printf '       confirmed, so it is refused.\n' >&2
            escaped=1
            continue
        fi
        if [ "$total_refs" -ne "$stamp_refs" ]; then
            printf 'ERROR: [gate C] %s names the build tree %s times but only %s of\n' \
                "${hit#"$PKG/"}" "$total_refs" "$stamp_refs" >&2
            printf '       those are the __build_manifest_dir__ provenance stamp\n' >&2
            printf '       (%s). The allowance covers that one string and\n' "$GATE_C_STAMP" >&2
            printf '       nothing else, so the rest are real references and this is\n' >&2
            printf '       refused. The runs carrying them:\n' >&2
            printf '%s\n' "$hit_strings" | grep -F -- "$SRC_DIR" | cut -c 1-200 \
                | sed 's/^/         /' >&2 || true
            escaped=1
            continue
        fi
        printf '  [--] %s carries the __build_manifest_dir__ provenance stamp\n' \
            "${hit#"$PKG/"}"
        printf '       %s times and no other build-tree string, and nothing\n' "$stamp_refs"
        printf '       resolves it as a path. Allowed by name.\n'
        continue
    fi
    printf 'ERROR: [gate C] %s names the build tree\n' "${hit#"$PKG/"}" >&2
    escaped=1
done <<EOF
$GATE_C_HITS
EOF

if [ "$escaped" -ne 0 ]; then
    # The one offender this tree is known to produce, named here so the failure
    # is a diagnosis rather than a list. It is a real defect and not a gate to
    # relax: what ships would load the build tree's entry library on this
    # machine and fail on every other one.
    if grep -qF -- "$SRC_DIR" "$PKG/target/release/ppf-contact-solver" 2>/dev/null; then
        printf '\n' >&2
        printf 'NOTE: ppf-contact-solver carries PPF_BACKEND_LIBRARY_DIR, an absolute\n' >&2
        printf '      path to this build tree that crates/ppf-cts-solver/build.rs emits\n' >&2
        printf '      and src/driver/launch.rs reads with option_env! at compile time.\n' >&2
        printf '      It is handed to the backend as OpenConfig.library_dir, and\n' >&2
        printf '      be_open loads <library_dir>/ppf_entries.metallib from it and\n' >&2
        printf '      refuses to open without one. The library itself ships, beside the\n' >&2
        printf '      backend in bin/, so what is missing is a way to\n' >&2
        printf '      resolve that directory at RUN time rather than at compile time.\n' >&2
        printf '\n' >&2
        printf '      The mechanism is already in this tree and simply unused on this\n' >&2
        printf '      path: library_directory() in metal_context.mm resolves the\n' >&2
        printf '      directory of the loaded dylib through dladdr, which is how the\n' >&2
        printf '      shader cache is already found, and it is exposed as\n' >&2
        printf '      context_prebuilt_library_dir(). The fix is for be_open to fall\n' >&2
        printf '      back to that when the caller names no directory, and for build.rs\n' >&2
        printf '      and launch.rs to stop emitting and reading\n' >&2
        printf '      PPF_BACKEND_LIBRARY_DIR, so that no absolute string is left in\n' >&2
        printf '      the binary at all. The abi_device test reads it too and moves in\n' >&2
        printf '      the same change. Until that lands, this distribution cannot start\n' >&2
        printf '      on a machine that is not this one.\n' >&2
        printf '\n' >&2
        printf '      DO NOT WIDEN THIS GATE TO GET PAST IT. A gate that goes quiet by\n' >&2
        printf '      being widened is worse than a red build here: it would ship a\n' >&2
        printf '      program that cannot open its backend, and the first person to find\n' >&2
        printf '      out would be whoever downloaded it.\n' >&2
    fi
    die \
        "the distribution is not self-contained" \
        "The references above point back into $SRC_DIR." \
        "Nothing was signed and dist was left in place for inspection."
fi

# The backend must be reachable through the rpath that was added, so check the
# file is where the rewritten reference says it is.
[ -f "$PKG/bin/libppfbe_metal.dylib" ] || die \
    "no backend at bin/libppfbe_metal.dylib after packaging"
printf '  [OK] no reference escapes the distribution\n'

# A BROKEN SYMLINK IS A PIECE THAT DID NOT TRAVEL, AND NOTHING ELSE HERE SEES
# ONE. The Mach-O walk is `find -type f`, so a dangling link is not on the list
# and is neither gated nor signed; codesign says nothing about it either.
# Whatever it names is simply absent from the payload, and the first thing to
# ask for it is on the user's Mac. Naming the link here costs one find and
# turns a run-time "image not found" into a build-time diagnosis.
printf 'Checking for broken symlinks...\n'
BROKEN_LINKS="$(find "$PKG" -type l ! -exec test -e {} \; -print)"
[ -z "$BROKEN_LINKS" ] || die \
    "the payload carries a symlink that points at nothing" \
    "$BROKEN_LINKS" \
    "Whatever put the link there did not bring its target: prune the link if" \
    "nothing needs it, or copy the target in beside it."
printf '  [OK] every symlink resolves\n'

# ---------------------------------------------------------------------------
step "[11/13] Smoke test"
# ---------------------------------------------------------------------------

# The payload's own interpreter, in the environment the launcher builds. This
# is what catches a copy that lost the cdylib, a load command that resolves
# nowhere, a pruned stdlib that broke an import and a rewritten rpath pointing
# at nothing, before any of it is signed.
#
# THE ENVIRONMENT IS POISONED FIRST, WITH THE FOUR VARIABLES A DEVELOPER'S
# SHELL PLAUSIBLY CARRIES. The launcher runs in the user's own shell, so these
# reach it, and each one breaks or silently redirects the import:
#   CARGO_TARGET_DIR  frontend/__init__.py searches ONLY it, never <tree
#                     root>/target, so the cdylib is looked for somewhere else
#   PYTHONHOME        overrides where a relocatable interpreter computes its
#                     home from and hides the stdlib
#   PYTHONPATH        shadows frontend, numpy or scipy
#   PYTHONOPTIMIZE    redirects every import to a .opt-N.pyc nothing wrote
# The subshell below then applies EXACTLY the unsets and assignments the
# launcher applies, so a missing unset fails this build rather than the user's
# first run. THAT DUPLICATION IS DELIBERATE: this block and the launcher's own
# environment block are edited together, and it is named at both sites.
#
# The Jupyter directories are exported at the paths the launcher uses and are
# NOT created here: nothing in this test starts a server, so nothing writes
# there, and a build has no business creating the user's run-time state.
#
# PYTHONDONTWRITEBYTECODE is set in that environment, so this test cannot
# write a __pycache__ into the payload and invalidate a signature one step
# later.
# The launcher's own state root, which is inside the distribution. Named the
# same way here as there, because this block and the launcher's environment
# block are edited together and the duplication is deliberate.
SMOKE_STATE="$PKG/local/share/ppf-cts"
(
    export CARGO_TARGET_DIR=/nonexistent/ppf-poison
    export PYTHONHOME=/nonexistent/ppf-poison
    export PYTHONOPTIMIZE=2
    export PYTHONPATH=/nonexistent/ppf-poison

    unset PYTHONHOME
    unset CARGO_TARGET_DIR
    unset PYTHONOPTIMIZE
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="$PKG"
    export PPF_CTS_BUILD_PYTHON="$PKG_PY"
    export JUPYTER_PATH="$PKG/python/share/jupyter"
    export JUPYTER_CONFIG_DIR="$SMOKE_STATE/jupyter/config"
    export JUPYTER_DATA_DIR="$SMOKE_STATE/jupyter/data"
    export JUPYTER_RUNTIME_DIR="$SMOKE_STATE/jupyter/runtime"
    export IPYTHONDIR="$SMOKE_STATE/jupyter/ipython"
    "$PKG_PY" -c 'import frontend, jupyterlab; print("  frontend  " + frontend.__file__); print("  jupyterlab " + jupyterlab.__version__)'
) || die \
    "the bundled interpreter could not import frontend and jupyterlab" \
    "The traceback above names which one failed. A missing frontend import" \
    "usually means the cdylib did not travel or cannot load; a missing" \
    "jupyterlab means warmup.sh installed into a different interpreter." \
    "This test runs with CARGO_TARGET_DIR, PYTHONHOME, PYTHONPATH and" \
    "PYTHONOPTIMIZE poisoned and then cleared the way the launcher clears" \
    "them, so a failure can also mean the launcher's environment block and" \
    "the one in this step have drifted apart."

# THE JUPYTER FLAGS THE LAUNCHER PASSES, CHECKED AGAINST THE SERVER THAT WILL
# READ THEM. Every one of them is a trait name rather than a command line
# alias, which is what makes this one lookup possible, and this is the only
# check between a renamed trait upstream and a distribution that refuses to
# start on the user's Mac: nothing here starts a server, and the launcher does
# not parse them itself.
"$PKG_PY" - <<'PY' || die \
    "the launcher passes a setting this build's Jupyter does not have" \
    "The names are printed above. Either something was renamed upstream, in" \
    "which case the launcher heredoc in bundle.sh moves with it, or the" \
    "warmup pinned a jupyter_server or jupyterlab that predates it."
import sys

from jupyter_server.serverapp import ServerApp

traits = ServerApp.class_traits()
wanted = ("port", "port_retries", "token", "root_dir", "open_browser", "answer_yes")
missing = [name for name in wanted if name not in traits]
if missing:
    print("  missing ServerApp traits: " + ", ".join(missing))
    sys.exit(1)
print("  jupyter    ServerApp carries all %d traits the launcher passes" % len(wanted))

# THE TWO LabApp NAMES ARE CHECKED HERE TOO, FOR THE SAME REASON: the launcher
# passes them, so an upstream rename must fail this build rather than a user's
# first run. The class is imported rather than named as a string, because a
# module path that no longer resolves is exactly the failure this catches.
from jupyterlab.labapp import LabApp

lab_traits = LabApp.class_traits()
lab_wanted = ("news_url", "check_for_updates_class")
lab_missing = [name for name in lab_wanted if name not in lab_traits]
if lab_missing:
    print("  missing LabApp traits: " + ", ".join(lab_missing))
    sys.exit(1)
if not lab_traits["news_url"].allow_none:
    print("  LabApp.news_url no longer allows None, so the launcher cannot disable the news fetch")
    sys.exit(1)
try:
    from jupyterlab.handlers.announcements import NeverCheckForUpdate  # noqa: F401
except ImportError as exc:
    print("  jupyterlab.handlers.announcements.NeverCheckForUpdate is gone: %s" % exc)
    sys.exit(1)
print("  jupyter    LabApp carries both names the launcher passes to keep it offline")
PY

# WHAT THIS DOES NOT REACH: --backend is answered before the argument parser
# and before any backend is opened, so it proves the binary runs and says which
# target it was compiled for, and says nothing about whether be_open can find
# its entry library. Gate C above is what speaks to that.
BACKEND_NAME="$("$PKG/target/release/ppf-contact-solver" --backend)" || die \
    "the packaged solver would not run" \
    "$PKG/target/release/ppf-contact-solver --backend failed. It answers that" \
    "flag before the argument parser and before any backend is opened, so a" \
    "failure here is the binary itself: a load command resolving nowhere, or a" \
    "backend dylib that did not travel."
printf '  backend    %s\n' "$BACKEND_NAME"

# THE LAUNCHER, THROUGH TWO SYMLINKS WHOSE NAMES CARRY A SPACE. --help returns
# before any interpreter is started and before any state directory is created,
# so this writes nothing, and what it proves is the part of the launcher no
# later step reaches: that it parses, resolves the directory it lives in, reads
# config.sh and prints the real path. It also walks this folder once, for the
# quarantine check, which is a fraction of a second and finds nothing here:
# what bundle.sh assembled was never downloaded, so it was never marked.
#
# The two links are different questions. A link to the FILE exercises the
# readlink loop; a link to the DIRECTORY leaves BASH_SOURCE unlinked and
# exercises `pwd -P` on a symlinked parent. Both are named with a space, and so
# is the directory holding them, because a person's own path may carry one even
# though nothing this script builds does.
#
# The links live outside the payload, so a failure between the two leaves
# nothing shipped.
SMOKE_LINKDIR="$(mktemp -d "${TMPDIR:-/tmp}/ppf smoke.XXXXXX")" || die \
    "could not create a temporary directory for the launcher check"
# The path the launcher must report. Resolved with pwd -P here for the same
# reason the launcher uses it there: if this build tree sits under a symlink,
# the logical path and the physical one differ and only one of them is what
# the launcher prints.
PKG_REAL="$(cd "$PKG" && pwd -P)"

ln -s "$PKG/ppf-contact-solver" "$SMOKE_LINKDIR/launcher link" || die \
    "could not create the launcher symlink under $SMOKE_LINKDIR"
ln -s "$PKG" "$SMOKE_LINKDIR/dist link" || die \
    "could not create the distribution symlink under $SMOKE_LINKDIR"

for probe in "$SMOKE_LINKDIR/launcher link" "$SMOKE_LINKDIR/dist link/ppf-contact-solver"; do
    probe_out="$("$probe" --help)" || { rm -rf "$SMOKE_LINKDIR"; die \
        "the launcher's --help failed when it was reached through a symlink" \
        "  started as: $probe" \
        "Its own message is above. --help returns before any interpreter is" \
        "started, so a failure here is the launcher resolving its own" \
        "directory, reading config.sh, or parsing at all."; }
    case "$probe_out" in
        *"$PKG_REAL"*) ;;
        *) rm -rf "$SMOKE_LINKDIR"
           die \
            "the launcher reached through a symlink did not report the real distribution path" \
            "  started as: $probe" \
            "  expected:   $PKG_REAL" \
            "It resolved somewhere else, so every path it builds at run time" \
            "would be wrong. The symlink walk and the pwd -P in the launcher" \
            "are what this covers." ;;
    esac
done
rm -rf "$SMOKE_LINKDIR"
printf '  launcher   --help answers through a file symlink and a directory symlink\n'
printf '  [OK] smoke test\n'

# ---------------------------------------------------------------------------
step "[12/13] Signing"
# ---------------------------------------------------------------------------

ENTITLEMENTS=""
cleanup_entitlements() {
    [ -n "$ENTITLEMENTS" ] && rm -f "$ENTITLEMENTS"
    return 0
}
trap cleanup_entitlements EXIT

if [ -n "${MAC_CODESIGN_IDENTITY:-}" ]; then
    # Matched with a case rather than a pipe into grep -q: grep exits on its
    # first match, and with pipefail a SIGPIPE upstream would turn a found
    # identity into a failed pipeline.
    IDENTITIES="$(security find-identity -v -p codesigning || true)"
    case "$IDENTITIES" in
        *"$MAC_CODESIGN_IDENTITY"*) ;;
        *) die \
            "no codesigning identity matching '$MAC_CODESIGN_IDENTITY' in the keychain" \
            "List what is available with:  security find-identity -v -p codesigning" \
            "MAC_CODESIGN_IDENTITY is set in config.sh or in the environment." ;;
    esac
    printf 'Signing with identity: %s\n' "$MAC_CODESIGN_IDENTITY"

    # Entitlements, and only in the identity case: they are meaningful only
    # alongside --options runtime. numba is in warmup.py's package list and it
    # JIT-compiles, so under the hardened runtime llvmlite's executable memory
    # is refused without these and a Developer-ID-signed release would break at
    # the first @njit while the ad-hoc developer build works. That is a defect
    # that appears only in the artifact nobody tests.
    #
    # Written to a temporary file rather than into this tracked directory,
    # which would need a .gitignore entry nobody owns.
    ENTITLEMENTS="$(mktemp "${TMPDIR:-/tmp}/ppf-entitlements.XXXXXX")"
    cat > "$ENTITLEMENTS" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.cs.allow-jit</key>
    <true/>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
</dict>
</plist>
EOF
    plutil -lint "$ENTITLEMENTS" >/dev/null || die "the generated entitlements do not lint"
else
    printf 'No MAC_CODESIGN_IDENTITY set. Signing ad-hoc.\n'
    printf 'An ad-hoc signature is what makes each arm64 binary loadable on the\n'
    printf 'machine that built it, and Gatekeeper rejects it on a copy that\n'
    printf 'arrives anywhere else quarantined. See config.sh.\n'
fi

sign_file() {
    local file="$1"
    if [ -n "${MAC_CODESIGN_IDENTITY:-}" ]; then
        # --options runtime is the hardened runtime, which notarization
        # requires; --timestamp needs network and fails loudly without it.
        codesign --force --timestamp --options runtime \
            --entitlements "$ENTITLEMENTS" \
            --sign "$MAC_CODESIGN_IDENTITY" "$file" || die "codesign failed on $file"
    else
        codesign --force --sign - "$file" || die "ad-hoc codesign failed on $file"
    fi
    codesign --verify --strict "$file" || die "the signature on $file does not verify"
}

# ONE FILE AT A TIME, AND NOTHING ELSE. Never --deep, which Apple deprecated
# for signing and which does not do what the name suggests; the explicit loop
# is the supported way.
#
# The interpreter's binaries arrive already ad-hoc signed and are re-signed
# anyway, with --force, so that every Mach-O that ships carries a signature
# made by this run over the bytes this run produced. Step 7 rewrote some of
# them and step 6 rewrote others, and a signature made before either is invalid.
#
# WHAT IS DELIBERATELY NOT SIGNED, BECAUSE THE OMISSION LOOKS LIKE AN
# OVERSIGHT: the launcher, config.sh, README.txt, THIRD_PARTY_LICENSES.txt,
# the Python sources and the .metallib files. `codesign` on a file that is
# not a Mach-O stores the signature in an extended attribute rather than in the
# file, nothing on the running path verifies it, and an extended attribute does
# not survive every way a person may unpack an archive. What makes the arm64
# binaries loadable at all is their own embedded signature, and that is what
# this loop puts there. Do not add a checksum manifest to compensate, and do
# not sign anything a loader will not verify.
printf 'Signing %d Mach-O files...\n' "${#ALL_MACHO[@]}"
SIGNED=0
for macho in "${ALL_MACHO[@]}"; do
    sign_file "$macho"
    SIGNED=$((SIGNED + 1))
    if [ "$((SIGNED % 100))" -eq 0 ]; then
        printf '  ... %d of %d\n' "$SIGNED" "${#ALL_MACHO[@]}"
    fi
done
printf '  [OK] %d Mach-O files signed\n' "$SIGNED"

# ---------------------------------------------------------------------------
step "[13/13] Verifying the signatures"
# ---------------------------------------------------------------------------

# A SECOND PASS OVER THE WHOLE LIST, TAKEN AFTER THE LOOP. sign_file verifies
# each file at the moment it writes it, which answers for that file at that
# moment; this answers for the payload as a SET, after the last thing that
# writes to it has run. A file signed early and then modified by something
# later in the loop is what the two readings differ about.
printf 'Verifying %d signatures...\n' "${#ALL_MACHO[@]}"
VERIFIED=0
for macho in "${ALL_MACHO[@]}"; do
    codesign --verify --strict "$macho" || die \
        "the signature on a payload file does not verify" \
        "  file: ${macho#"$PKG/"}" \
        "codesign's own message is above. Something wrote to this file after" \
        "it was signed, which is the ordering step 12 exists to prevent."
    VERIFIED=$((VERIFIED + 1))
done
printf '  [OK] %d signatures verify\n' "$VERIFIED"

# WHICH ppf-contact-solver IS AT THE TOP. Two files in this distribution carry
# that basename, and only the launcher belongs at the root: a Mach-O found here
# would be the solver copied to the wrong place, which presents to a person as
# a program that starts a solver with no arguments and says nothing useful.
[ -x "$PKG/ppf-contact-solver" ] || die \
    "the launcher at the top of the distribution is missing or not executable" \
    "  $PKG/ppf-contact-solver" \
    "It is what a person runs, so it has to be there and it has to have its" \
    "executable bit."
[ "$(head -c 2 "$PKG/ppf-contact-solver")" = '#!' ] || die \
    "the file at the top of the distribution is not the launcher" \
    "  $PKG/ppf-contact-solver" \
    "It does not begin with a shebang, so it is not the shell script this" \
    "script generates. The solver Mach-O of the same name belongs at" \
    "target/release/ppf-contact-solver and nowhere else."
# Cheap, and step 12 is the last thing that writes, so this is asked again
# rather than trusted from step 8.
bash -n "$PKG/ppf-contact-solver" || die \
    "the launcher does not parse" \
    "bash -n's own message is above."
printf '  [OK] the launcher is at the top, is executable, and parses\n'

# NOTHING HERE ASKS spctl FOR A VERDICT. Gatekeeper's answer is about a
# quarantined copy on someone else's machine, and this script has none: what it
# holds is the directory it just wrote, which carries no quarantine attribute.
# A verdict taken here would therefore be about a different question than the
# one a reader has, and it could only be reported rather than acted on. The
# NOTARIZATION section printed at the end of this run answers that question
# directly instead.

# ---------------------------------------------------------------------------
step "DISTRIBUTION COMPLETE"
# ---------------------------------------------------------------------------
printf 'Distribution: %s\n' "$PKG"
# Roughly a gigabyte, most of it the interpreter's packages. Said here rather
# than left to surprise a reader of the output.
du -sh "$PKG" 2>/dev/null || true
printf 'Version:      %s\n' "$APP_VERSION"
printf 'Minimum:      macOS %s\n' "$MIN_OS"
printf 'Thinning:     %d universal files, %d bytes saved\n' "$THINNED" "$SAVED_BYTES"
printf 'Signed:       %d Mach-O files\n' "$SIGNED"
printf '\n'
printf 'To run it:\n'
printf '  cd "%s" && ./ppf-contact-solver\n' "$PKG"

cat <<EOF

NOTARIZATION
------------
This script does not notarize, and nothing above pretends to. TWO things are
missing, and only one of them is a credential.

Credentials. Notarization needs an Apple Developer account. Supply them at
the single point config.sh describes: MAC_CODESIGN_IDENTITY for signing, and
for notarytool either a keychain profile created with 'xcrun notarytool
store-credentials', or the triple Apple ID, team ID and an app-specific
password. Never put that password in config.sh, which is tracked in git.

A container. 'xcrun stapler staple' takes an .app, a .dmg or a .pkg, and
this script produces a directory, so there is nothing here to staple a
ticket to. A distributor who needs a stapled ticket wraps this directory in
a container of their own and notarizes that; nothing here produces one.

What is present today, with a Developer ID identity set, is that every
Mach-O in the distribution carries a Developer ID signature with the
hardened runtime, a secure timestamp and the three entitlements above.

What the missing notarization costs is narrower than it sounds, and worth
stating exactly, because the wrong reading is that this cannot leave the
machine that built it.

Gatekeeper acts on the com.apple.quarantine attribute, which a DOWNLOADING
application sets: a browser, or Mail. It is not set by the filesystem and not
by the folder arriving on another Mac. So:

  * Fetched with curl, scp, git or rsync, or copied from a volume, nothing
    here carries a quarantine attribute and it runs. The ad-hoc signature is
    what makes each binary executable at all, which on Apple silicon is
    required of every arm64 Mach-O, and it is enough.
  * Downloaded through a browser, the payload is quarantined, and the first
    exec of the bundled interpreter is what Gatekeeper would refuse. THE
    LAUNCHER CLEARS THE MARK ON ITS OWN FOLDER BEFORE IT REACHES THAT EXEC,
    so this does not surface. It can, because Gatekeeper assesses the exec of
    a Mach-O file and the launcher is a shell script read by /bin/bash: it
    runs while marked. It walks the whole folder rather than the binaries it
    starts, since every Mach-O is assessed when it is LOADED and the
    interpreter brings hundreds of extension modules; and it walks with
    xattr -s, which clears a symbolic link itself instead of following it
    somewhere outside the folder. It prints one line saying what it cleared,
    and if anything stays marked it says that too and the interpreter check
    then names Open Anyway and the by-hand command.

      xattr -s -d -r com.apple.quarantine "$PKG"

    Nothing in THIS script clears anything: what it packages was never
    marked.

Every half of that was measured rather than reasoned about, on macOS 26.6.
spctl assesses an ad-hoc signature as rejected and reports no TeamIdentifier,
which is the refusal a marked copy meets. A quarantined shell script run from
a terminal executes normally, which is what lets the launcher act at all. An
ad-hoc signed Mach-O whose mark was cleared execs, though spctl still assesses
it as rejected, so spctl's verdict is not the gate. And a real ditto archive,
marked and re-extracted, carries the mark onto every entry, after which the
generated launcher clears all of them and the payload binaries run.

The release workflow's macOS leg exercises that directly: its verification job
marks the extracted distribution itself and asserts the launcher clears it,
which is a Gatekeeper risk no build-time step can reach.
Notarization would still buy something this does not, a copy that satisfies
Gatekeeper without any file being modified, and it remains the better answer
wherever an Apple Developer account is available.

TO HAND THIS TO SOMEONE
-----------------------
One archive whose top-level entry is the distribution directory itself, so
whoever unpacks it gets a folder with the launcher inside rather than a
loose dist wrapper:

  ditto -c -k --keepParent "$PKG" \\
      "$DIST/ppf-contact-solver-$APP_VERSION-macos-arm64.zip"
EOF
