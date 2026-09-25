#!/usr/bin/env bash
# File: build-linux-native/bundle.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Packages a built tree into a self-contained Linux distribution directory under
# build-linux-native/dist, the counterpart of build-mac-native/bundle.sh and
# build-win-native/bundle.bat. Run warmup.sh and build.sh first.
#
# It is safe to run twice: dist is removed and rebuilt from the current build.
#
# WHO RUNS WHAT IT PRODUCES: a Linux user on the architecture it was built on,
# with a WORKING GRAPHICS DRIVER where the distribution carries a GPU backend,
# and NOTHING ELSE. No CUDA toolkit, no ROCm installation, no Python, no Rust, no
# compiler. Every library a shipped binary loads that is not part of glibc,
# libstdc++ or the driver travels in the directory, and the gates in step 10 are
# what make that a checked property rather than a hope.
#
# WHETHER A GPU BACKEND SHIPS IS THE BUILD'S, from scripts/platform.sh: one does
# on x86_64, and none on aarch64 or when config.sh sets PPF_LINUX_BACKENDS=cpu,
# where the CPU backend ships alone. What is said below about the GPU backend and
# its library applies to a distribution that carries one.
#
# WHICH GPU BACKEND IT PACKAGES IS ASKED OF THE BINARY, not configured: step 1
# reads `target/release/ppf-contact-solver --backend` and the whole script
# follows that answer, so one script packages a CUDA distribution and a ROCm one
# and neither can disagree with what was actually built.
#
# THE TWO BACKENDS DIFFER IN EXACTLY ONE STRUCTURAL WAY, and it is worth knowing
# before reading the steps. CUDA links its runtime STATICALLY, so no toolkit
# library ships: libppfbe_cuda.so loads only the driver's libcuda.so.1, which
# build.sh step 3 checks on the artifact. ROCm's runtime is DYNAMIC, so the HIP
# libraries libppfbe_rocm.so loads are files that have to travel, and step 1
# finds them by walking the library's own search path rather than by naming
# them. Everything else, the RPATH rewriting, the gates, the poison test, is the
# same work over a different file list.
#
# WHAT IT PRODUCES
#   dist/<name>/    one directory. <name> is `ppf-contact-solver` unless
#                   PPF_LINUX_DIST_NAME says otherwise, which the release
#                   workflow sets to the archive's stem. Nothing else lands in
#                   dist: the release workflow makes the archive.
#
# THE LAYOUT, AND WHY THE DIRECTORY IS THE TREE ROOT
#   ppf-contact-solver          the launcher, a bash script
#   config.sh  README.txt  THIRD_PARTY_LICENSES.txt  licenses/
#   bin/                        ffmpeg, and with a GPU backend libppfbe_<backend>.so
#                               and whatever it loads
#   target/<backend>/release/   that backend's solver, server and Python
#                               extension, one directory per backend shipped
#                               (`cuda`, `rocm`, `cpu`), each with the
#                               `.ppf-backend` marker step 3 writes
#   python/                     a relocatable CPython with the frontend packages
#   frontend/  examples/  crates/
#
#   THERE IS NO `target/release` IN A DISTRIBUTION. Every backend keeps the
#   directory it was built in, which is what lets one distribution carry
#   several, and `backend_target_dir` in scripts/platform.sh is the one place
#   that spelling lives. `frontend/__init__.py` resolves the root from its own
#   __file__ and `frontend._backends_` then reads the markers to decide which
#   of those directories a run uses, so nothing needs a fixed one. The Blender
#   add-on's Linux Native connection searches the same set.
#
# THE SOLVER FINDS ITS BACKEND THROUGH DT_RPATH, NOT DT_RUNPATH. The glibc loader
#   searches LD_LIBRARY_PATH BEFORE an object's DT_RUNPATH, so a shell that
#   exports a directory holding another libppfbe_<backend>.so, a developer's
#   build tree for instance, would have the shipped solver load that instead
#   of its own. DT_RPATH is searched before LD_LIBRARY_PATH, and it also applies
#   to what the loaded library loads in turn. Step 6 writes it with patchelf, and
#   step 11 proves the precedence against a poisoned LD_LIBRARY_PATH.
#
#   The launcher leaves LD_LIBRARY_PATH alone, so a machine that locates its
#   driver's libcuda.so.1 or libamdhip64's dependencies through it keeps working.
#
# THE DISTRIBUTION WRITES NOTHING OUTSIDE ITSELF
#   `.ppf-selfcontained` at the root roots the session data and the asset cache
#   inside the directory, for the launcher and for anything else that starts the
#   server. The launcher points Jupyter, IPython, matplotlib and numba there too.
#   Removing the directory removes everything the program wrote.
#
# SWITCHES
#   PPF_LINUX_DIST_NAME   the directory name under dist
#   PPF_LINUX_MAX_GLIBC   refuse a payload that needs a newer glibc than this.
#                         The release workflow sets it to the floor it targets.
set -euo pipefail

# THE C LOCALE IS A CORRECTNESS SETTING HERE. This script runs grep and sed over
# a payload that is mostly not text, and a multibyte locale can make a filter
# stop at a byte that is not valid in it, truncating whatever it was reading.
export LC_ALL=C

BUILD_LINUX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$BUILD_LINUX/.." && pwd)"
LOGFILE="$BUILD_LINUX/bundle.log"

if [ -z "${PPF_LINUX_BUNDLE_LOGGING:-}" ]; then
    printf 'Logging to %s\n' "$LOGFILE"
    set +e
    PPF_LINUX_BUNDLE_LOGGING=1 "${BASH_SOURCE[0]}" "$@" 2>&1 | tee "$LOGFILE"
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

[ "$(uname -s)" = "Linux" ] || die "this packages a Linux build, and uname reports $(uname -s)"

# GATE C SEARCHES EVERY SHIPPED FILE FOR THIS TREE'S PATH AS PLAIN TEXT, so the
# path has to be one that unrelated text does not contain. A single component
# such as /src occurs inside any source path ("crates/x/src/y.rs",
# ".cargo/registry/src/"), and a release built there reported 275 files, nearly
# all of them that coincidence, which hides a real hit among them.
case "$SRC_DIR" in
    /*/*) ;;
    *) die "the build tree $SRC_DIR is too short a path for the self-containment check" \
           "Gate C looks for this path as text in every shipped file, and a single" \
           "component occurs inside unrelated source paths. Build from a deeper" \
           "directory, for example /opt/ppf-release-build/ppf-contact-solver." ;;
esac
for tool in readelf strings rsync python3 grep sed awk; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool not found on PATH: $tool" \
        "Run warmup.sh, which names what the host must provide."
done

# shellcheck source=build-linux-native/config.sh
. "$BUILD_LINUX/config.sh"
# shellcheck source=build-linux-native/scripts/platform.sh
. "$BUILD_LINUX/scripts/platform.sh"
resolve_platform || die "this host, or PPF_LINUX_BACKENDS, cannot be packaged (see above)"
# shellcheck source=build-linux-native/scripts/load-downloads.sh
. "$BUILD_LINUX/scripts/load-downloads.sh"
load_downloads "$BUILD_LINUX/scripts/downloads.txt" || die \
    "could not read build-linux-native/scripts/downloads.txt"
select_arch_downloads PYTHON PATCHELF || die \
    "scripts/downloads.txt has no complete $PPF_LINUX_ARCH entry (see above)"
# shellcheck source=build-linux-native/scripts/backend-path.sh
. "$BUILD_LINUX/scripts/backend-path.sh"

DIST="$BUILD_LINUX/dist"
CPU_REL="$SRC_DIR/$(backend_target_dir cpu)/release"
CUDA_ROOT="$BUILD_LINUX/cuda"
PY_ROOT="$BUILD_LINUX/python"
PATCHELF="$BUILD_LINUX/patchelf/bin/patchelf"
FFMPEG_DIR="$BUILD_LINUX/ffmpeg"

PPF_LINUX_DIST_NAME="${PPF_LINUX_DIST_NAME:-ppf-contact-solver}"
case "$PPF_LINUX_DIST_NAME" in
    "" | */* | .* | *" "*)
        die "PPF_LINUX_DIST_NAME is not a usable directory name" \
            "  got: ${PPF_LINUX_DIST_NAME:-<empty>}" \
            "It names ONE directory directly under dist: not empty, no slash, no" \
            "leading dot, no space." ;;
esac
PKG="$DIST/$PPF_LINUX_DIST_NAME"
PKG_PY="$PKG/python/bin/python3"

# EVERY INTERPRETER THIS SCRIPT STARTS SEES ONLY ITS OWN TREE, as the launcher's
# will. A user site-packages directory is on sys.path by default, so a question
# asked of the copy (where certifi is, which Jupyter settings exist) would
# otherwise be answered by a package in ~/.local that does not ship.
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

printf '============================================================\n'
printf "  ZOZO's Contact Solver, Linux distribution\n"
printf '============================================================\n'

# ---------------------------------------------------------------------------
step "[1/11] Verifying the build and the provisioned inputs"
# ---------------------------------------------------------------------------

# The release directories this distribution ships, one per backend, each as
# "<directory under the tree>:<backend>". Every later step that copies, rewrites
# or starts a solver walks this list rather than naming a directory.
#
# WHICH BACKENDS SHIP IS THE BUILD'S CONFIGURATION, decided in
# scripts/platform.sh: every backend this architecture can build by default,
# narrowed by PPF_LINUX_BACKENDS, and each one built into its own
# target/<backend> directory. WHICH BACKEND A DIRECTORY HOLDS IS STILL ASKED OF
# THE BINARY, never read off the directory's name, which is the rule this build
# follows for a backend: ask an artifact what it IS rather than inferring it from
# its path. A tree built for one backend and packaged as
# another stops the build here rather than shipping with a stamp, licences and
# an audit flag that describe something else.
#
# GPU_BACKENDS is empty where the CPU backend ships alone, and every later step
# walks it rather than testing the configuration.
GPU_BACKENDS=()
declare -A BACKEND_LIBS=()
RUNTIME_LIBS=()
SHIPPED=()
for backend in $PPF_LINUX_BACKENDS; do
    release_rel="$(backend_target_dir "$backend")/release"
    for artifact in ppf-contact-solver ppf-cts-server lib_ppf_cts_py.so; do
        [ -f "$SRC_DIR/$release_rel/$artifact" ] || die \
            "$SRC_DIR/$release_rel/$artifact not found" "Run build.sh first."
    done
    # build_tree_run rather than a bare run: these are the BUILD TREE's binaries,
    # and a ROCm one names the HIP runtime directly, reaching it only through the
    # SDK until this script stages that runtime into bin/. The packaged binaries
    # are asked the same question later, from the payload alone and with no
    # LD_LIBRARY_PATH, which is the property worth proving.
    got="$(build_tree_run "$backend" "$SRC_DIR/$release_rel/ppf-contact-solver" --backend)" || die \
        "$SRC_DIR/$release_rel/ppf-contact-solver --backend failed" "Re-run build.sh."
    [ "$got" = "$backend" ] || die \
        "$release_rel holds the $got backend and this step packages it as $backend" \
        "The binary is the authority on what it is, so this is not resolved by" \
        "preferring one of them. Remove $release_rel and re-run build.sh."
    SHIPPED+=("$release_rel:$backend")
    if [ "$backend" != cpu ]; then
        GPU_BACKENDS+=("$backend")
        BACKEND_LIBS["$backend"]="libppfbe_${backend}.so"
    fi
done
DIST_BACKENDS="$PPF_LINUX_BACKENDS"
printf '  architecture %s, backends %s\n' "$PPF_LINUX_ARCH" "$DIST_BACKENDS"
printf '  [OK] %s, each in its own target directory\n' "$DIST_BACKENDS"

declare -A BACKEND_SRCS=()
for backend in ${GPU_BACKENDS[@]+"${GPU_BACKENDS[@]}"}; do
    backend_lib="${BACKEND_LIBS[$backend]}"
    backend_rel="$(backend_target_dir "$backend")/release"
    # The backend library, from the search path the solver itself records.
    BACKEND_SRC="$(resolve_needed "$SRC_DIR/$backend_rel/ppf-contact-solver" "$backend_lib")" || die \
        "could not resolve the $backend backend the solver loads (see above)" "Re-run build.sh."
    BACKEND_SRCS["$backend"]="$BACKEND_SRC"
    printf '  [OK] backend %s\n' "$BACKEND_SRC"

    # WHAT THE BACKEND LOADS IN TURN, AND WHY THE TWO BACKENDS DIFFER HERE. The CUDA
    # backend links the CUDA runtime STATICALLY, so it loads nothing of the toolkit's
    # and this closure is EMPTY; that is the property the header states and build.sh
    # step 3 checks. ROCm's runtime is DYNAMIC, so the HIP libraries are files that
    # have to travel, and they are found by walking the library's own search path
    # rather than by naming them here (scripts/backend-path.sh says why).
    #
    # The machine supplies the same families it supplies for CUDA, plus the four a
    # ROCm payload adds, which elf-audit.py's ROCM_SYSTEM lists and measured on the
    # packages themselves. Both lists are stated once, here and there, and gate B is
    # what makes a disagreement fail rather than ship.
    RUNTIME_SYSTEM=(
        libc.so.6 libm.so.6 libdl.so.2 librt.so.1 libpthread.so.0 libutil.so.1
        libresolv.so.2 "$PPF_LINUX_LOADER" libstdc++.so.6 libgcc_s.so.1 libz.so.1
        libelf.so.1 libdrm.so.2 libdrm_amdgpu.so.1 libnuma.so.1
        libcuda.so.1
    )
    # CAPTURED WITH `$(...)` SO THE EXIT STATUS IS ACTUALLY CHECKED. The obvious
    # spelling, `while read ... done < <(runtime_closure ...) || die`, tests the
    # WHILE LOOP's status and never the walk's: a process substitution's exit code is
    # not reported to the construct it feeds. Measured on a chain with one library
    # hidden, that form collected a PARTIAL closure, returned success, and would have
    # shipped a distribution missing a file with nothing failing until a user ran it.
    RUNTIME_CLOSURE_OUT="$(runtime_closure "$BACKEND_SRC" "${RUNTIME_SYSTEM[@]}")" || die \
        "could not resolve everything $backend_lib loads (see above)" \
        "A name reported there is either a library that did not travel or a system" \
        "library this script does not declare. Neither is fixed by dropping the check."
    backend_runtime=0
    # ONE LIST FOR EVERY GPU BACKEND, WITH NO DUPLICATES. bin/ is shared, so two
    # backends that load the same library ship one copy of it, and the RPATH and
    # licence steps below walk each file once.
    while IFS= read -r runtime_lib; do
        [ -n "$runtime_lib" ] || continue
        backend_runtime=$((backend_runtime + 1))
        for seen in ${RUNTIME_LIBS[@]+"${RUNTIME_LIBS[@]}"}; do
            [ "$seen" = "$runtime_lib" ] && continue 2
        done
        RUNTIME_LIBS+=("$runtime_lib")
        printf '         %s\n' "$runtime_lib"
    done <<< "$RUNTIME_CLOSURE_OUT"
    if [ "$backend_runtime" -eq 0 ]; then
        printf '  [OK] %s loads no toolkit library of its own\n' "$backend_lib"
    else
        printf '  [OK] %s also loads %d library/libraries, which travel (listed above)\n' \
            "$backend_lib" "$backend_runtime"
    fi
done

[ -x "$PY_ROOT/bin/python3" ] || die \
    "no bundled interpreter at $PY_ROOT/bin/python3" \
    "Run warmup.sh without PPF_LINUX_PYTHON=0. A distribution that ships no" \
    "interpreter cannot start on a machine with no Python, which is the point."
"$PY_ROOT/bin/python3" -c 'pass' >/dev/null 2>&1 || die \
    "$PY_ROOT/bin/python3 exists but does not run" "Remove $PY_ROOT and re-run warmup.sh."

[ -x "$FFMPEG_DIR/ffmpeg" ] && [ -s "$FFMPEG_DIR/ffmpeg.sources.txt" ] && [ -d "$FFMPEG_DIR/licenses" ] || die \
    "no complete slim ffmpeg at $FFMPEG_DIR" \
    "Run warmup.sh without PPF_LINUX_FFMPEG=0. The binary ships with its source" \
    "revisions and license texts, and all three are required."

[ -x "$PATCHELF" ] || die "no patchelf at $PATCHELF" "Run warmup.sh."
if has_backend cuda; then
    [ -f "$CUDA_ROOT/LICENSE" ] || die \
        "no $CUDA_ROOT/LICENSE, the terms the statically linked CUDA runtime ships under" \
        "Re-run warmup.sh."
fi

# The CPU solver is asked because every distribution ships it; each build
# reports the same version.
VERSION_LINE="$(env -u LD_LIBRARY_PATH "$CPU_REL/ppf-contact-solver" --version 2>/dev/null)" || die \
    "$CPU_REL/ppf-contact-solver --version failed"
APP_VERSION="${VERSION_LINE##* }"
printf '%s' "$APP_VERSION" | grep -Eq '^[0-9][0-9.]*$' || die \
    "could not read a version out of: $VERSION_LINE"
printf '  [OK] version %s\n' "$APP_VERSION"

# ---------------------------------------------------------------------------
step "[2/11] Creating the distribution directory"
# ---------------------------------------------------------------------------

rm -rf "$DIST"
mkdir -p "$PKG/bin" "$PKG/licenses"
for entry in "${SHIPPED[@]}"; do
    mkdir -p "$PKG/${entry%:*}"
done
printf '  [OK] %s\n' "$PKG"

# ---------------------------------------------------------------------------
step "[3/11] Copying binaries"
# ---------------------------------------------------------------------------

for entry in "${SHIPPED[@]}"; do
    cp "$SRC_DIR/${entry%:*}/ppf-contact-solver" "$SRC_DIR/${entry%:*}/ppf-cts-server" \
        "$SRC_DIR/${entry%:*}/lib_ppf_cts_py.so" "$PKG/${entry%:*}/"
    chmod u+w "$PKG/${entry%:*}/"*
    # The backend marker, written rather than copied: it states what LANDED, and
    # frontend.backend_of reads exactly this file.
    printf '%s' "${entry##*:}" > "$PKG/${entry%:*}/.ppf-backend"
done
for backend in ${GPU_BACKENDS[@]+"${GPU_BACKENDS[@]}"}; do
    cp "${BACKEND_SRCS[$backend]}" "$PKG/bin/"
done
# Each by its SONAME, which is what the loader will look for: a build tree can
# hold the file under a versioned name with symlinks beside it, and a symlink
# copied into bin/ would point outside the distribution.
for runtime_lib in ${RUNTIME_LIBS[@]+"${RUNTIME_LIBS[@]}"}; do
    cp -L "$runtime_lib" "$PKG/bin/$(basename "$runtime_lib")"
done
cp "$FFMPEG_DIR/ffmpeg" "$PKG/bin/"
chmod u+w "$PKG/bin/"*

# The marker that roots every session and the asset cache inside this directory
# (datamodel::app::is_selfcontained). A file rather than a variable, because the
# launcher is not the only way to start this tree's server.
printf '%s\n' "$APP_VERSION" > "$PKG/.ppf-selfcontained"
printf '  [OK] %s, ffmpeg, the markers%s\n' "$DIST_BACKENDS" \
    "${GPU_BACKENDS[*]:+, and the backend library of each of ${GPU_BACKENDS[*]}}"

# ---------------------------------------------------------------------------
step "[4/11] Copying the Python side"
# ---------------------------------------------------------------------------

copy_tree() {
    # $1 = source dir, $2 = destination dir, remaining = rsync excludes
    local src="$1" dst="$2" pattern
    shift 2
    [ -d "$src" ] || die "$src not found"
    mkdir -p "$dst"
    local args=(-a)
    for pattern in "$@"; do
        args+=(--exclude "$pattern")
    done
    rsync "${args[@]}" "$src/" "$dst/"
}

copy_tree "$SRC_DIR/frontend" "$PKG/frontend" '__pycache__' '*.pyc' '*.pyo'
mkdir -p "$PKG/examples"
rsync -a -m --include '*/' --include '*.ipynb' --include '*.py' --exclude '*' \
    "$SRC_DIR/examples/" "$PKG/examples/"
# The roots ppf-cts-server harvests log-channel names from
# (crates/ppf-cts-server/src/main.rs), which probes crates/ppf-cts-compute whole
# and recurses, so shipping the backend this distribution actually carries is
# what that root wants. MEASURED: no backend directory declares a channel today,
# `SimpleLog logging(` appearing only in ppf-cts-core/src and
# ppf-cts-solver/src/driver, so a distribution's channels come from the other two
# roots and neither backend's presence changes them. It ships as the reference
# source for the backend that shipped, which is why a ROCm distribution carries
# rocm/ rather than a CUDA tree it does not use. Build directories are output,
# not source.
copy_tree "$SRC_DIR/crates/ppf-cts-solver/src" "$PKG/crates/ppf-cts-solver/src" \
    '__pycache__' 'build' 'build-tests' 'obj' 'tests' '*.pyc' '*.o' '*.d'
for backend in ${GPU_BACKENDS[@]+"${GPU_BACKENDS[@]}"}; do
    copy_tree "$SRC_DIR/crates/ppf-cts-compute/$backend" \
        "$PKG/crates/ppf-cts-compute/$backend" 'build' '*.o' '*.d'
done
copy_tree "$SRC_DIR/crates/ppf-cts-core/src" "$PKG/crates/ppf-cts-core/src" \
    '__pycache__' '*.pyc'
printf '  [OK] frontend, examples, crate source roots\n'

# THE BRANCH STAMP, WHICH KEEPS THE FRONTEND FROM RUNNING git. data_dirpath_for
# reads .git/branch_name.txt before it would ask git, and every App.create()
# reaches it. The test is on emptiness, because an empty file is treated as
# absent.
BUILD_BRANCH="$(git -C "$SRC_DIR" branch --show-current 2>/dev/null || true)"
[ -n "$BUILD_BRANCH" ] || BUILD_BRANCH="unknown"
mkdir -p "$PKG/.git"
printf '%s\n' "$BUILD_BRANCH" > "$PKG/.git/branch_name.txt"
printf '  [OK] branch stamp .git/branch_name.txt (%s)\n' "$BUILD_BRANCH"

# ---------------------------------------------------------------------------
step "[5/11] Copying the bundled interpreter and relocating its scripts"
# ---------------------------------------------------------------------------

# -a and NOT -aL: the interpreter tree is full of relative symlinks.
rsync -a "$PY_ROOT/" "$PKG/python/"
[ -x "$PKG_PY" ] || die "no interpreter at $PKG_PY after the copy"

# THE CONSOLE SCRIPTS' SHEBANGS NAME THIS BUILD TREE UNTIL THIS RUNS. pip writes
# the absolute path of the interpreter it installed for into every console
# script, which is dead on any other machine and which gate C would refuse. The
# replacement is two lines that are a valid shell script and a valid Python
# module at once: /bin/sh reads the second line as `exec` of the interpreter
# beside the script; Python reads lines two and three as one string.
TRAMPOLINED=()
for script in "$PKG/python/bin"/*; do
    [ -f "$script" ] && [ ! -L "$script" ] || continue
    [ "$(head -c 2 "$script" 2>/dev/null)" = '#!' ] || continue
    shebang="$(head -1 "$script")"
    case "$shebang" in
        '#!'*python*) ;;
        *) continue ;;
    esac
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
                "The trampoline passes no interpreter arguments, so that argument" \
                "would be dropped silently. Decide deliberately what it becomes." ;;
        esac
    fi
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
done
printf '  [OK] %d console scripts now resolve the interpreter beside them\n' "${#TRAMPOLINED[@]}"
for script in "$PKG/python/bin"/*; do
    [ -f "$script" ] && [ ! -L "$script" ] || continue
    [ "$(head -c 2 "$script" 2>/dev/null)" = '#!' ] || continue
    case "$(head -1 "$script")" in
        '#!'*python*) die "a shebang under python/bin still names an interpreter directly" \
            "  file: ${script#"$PKG/"}" ;;
    esac
done
if [ "${#TRAMPOLINED[@]}" -gt 0 ]; then
    "$PKG_PY" - "${TRAMPOLINED[@]}" <<'PY' || die "a trampolined console script no longer parses as Python"
import ast
import sys

for path in sys.argv[1:]:
    with open(path, "rb") as handle:
        ast.parse(handle.read(), filename=path)
print("  [OK] %d trampolined scripts still parse as Python" % (len(sys.argv) - 1))
PY
fi

# ---------------------------------------------------------------------------
step "[6/11] Rewriting library search paths"
# ---------------------------------------------------------------------------

# Every file this project built gets DT_RPATH naming bin/ when it needs a
# library there, and no search path at all when it does not, so nothing points
# back into the build tree and nothing carries a path it has no use for. The
# interpreter's files and ffmpeg are not rewritten: they arrive relocatable or
# static, and the gates walk them all the same.
set_search_path() {
    # $1 = file, $2 = the DT_RPATH to write when the file needs a library in bin
    local file="$1" rpath="$2" soname needs_bin=0 recorded
    while IFS= read -r soname; do
        [ -n "$soname" ] || continue
        if [ -e "$PKG/bin/$soname" ]; then
            needs_bin=1
        fi
    done < <(elf_needed "$file")
    if [ "$needs_bin" -eq 1 ]; then
        "$PATCHELF" --force-rpath --set-rpath "$rpath" "$file"
    else
        "$PATCHELF" --remove-rpath "$file"
    fi
    recorded="$(readelf -d "$file" | awk '/\((RPATH|RUNPATH)\)/ { print $2, $NF }')"
    if [ "$needs_bin" -eq 1 ]; then
        [ "$recorded" = "(RPATH) [$rpath]" ] || die \
            "${file#"$PKG/"} records '$recorded' after patchelf, not (RPATH) [$rpath]"
    else
        [ -z "$recorded" ] || die \
            "${file#"$PKG/"} still records a search path after patchelf: $recorded"
    fi
    printf '  %-40s %s\n' "${file#"$PKG/"}" "${recorded:-no search path}"
}

# EVERY BACKEND SITS AT THE SAME DEPTH, target/<backend>/release, so one
# relative path reaches bin/ from all of them.
for entry in "${SHIPPED[@]}"; do
    for name in ppf-contact-solver ppf-cts-server lib_ppf_cts_py.so; do
        # shellcheck disable=SC2016 # $ORIGIN is for the loader, not for this shell.
        set_search_path "$PKG/${entry%:*}/$name" '$ORIGIN/../../../bin'
    done
done
for backend in ${GPU_BACKENDS[@]+"${GPU_BACKENDS[@]}"}; do
    backend_lib="${BACKEND_LIBS[$backend]}"
    # shellcheck disable=SC2016
    set_search_path "$PKG/bin/$backend_lib" '$ORIGIN'

    # The one binary each GPU path rests on has to be one that searches bin/,
    # whatever the loop above decided file by file.
    elf_needed "$PKG/$(backend_target_dir "$backend")/release/ppf-contact-solver" \
        | grep -qxF "$backend_lib" || die \
        "target/$backend/release/ppf-contact-solver does not list $backend_lib as NEEDED" \
        "A $backend solver that does not load its backend library runs on the host."
done
# EVERYTHING THE BACKENDS LOAD IS POINTED AT ITS OWN DIRECTORY TOO. For CUDA that
# list is empty. For ROCm the HIP libraries reference each other, so each needs
# $ORIGIN of its own: without it the first one the loader opens would search the
# BUILD host's SDK path, which is absent on the user's machine and is a
# build-tree path gate A refuses anyway.
for runtime_lib in ${RUNTIME_LIBS[@]+"${RUNTIME_LIBS[@]}"}; do
    # shellcheck disable=SC2016
    set_search_path "$PKG/bin/$(basename "$runtime_lib")" '$ORIGIN'
done
printf '  [OK] search paths rewritten\n'

# ---------------------------------------------------------------------------
step "[7/11] Pruning and reading the glibc floor"
# ---------------------------------------------------------------------------

STDLIB="$("$PKG_PY" -c 'import sysconfig; print(sysconfig.get_paths()["stdlib"])')" || die \
    "the bundled interpreter could not report its stdlib path"
SITE="$("$PKG_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" || die \
    "the bundled interpreter could not report its site-packages path"
[ -d "$STDLIB" ] || die "the interpreter reports a stdlib at $STDLIB, which is not there"
# The only prune, and the list is not to be extended: each further removal risks
# an import failure that appears only on a user's machine. It also removes the
# badsyntax_*.py files that would make compileall report a failure.
if [ -d "$STDLIB/test" ]; then
    rm -rf "$STDLIB/test"
    printf '  [OK] pruned the stdlib test package\n'
fi

# Modules that name a library the payload deliberately does not carry. Each is a
# statement about one file and one library, with its reason, and elf-audit.py
# refuses an entry that matches nothing, so a stale one fails rather than
# lingering to cover something it was never written for.
#   numba's OpenMP threading layer needs a libgomp.so.1.0.0 soname that no
#     distribution provides, so numba falls back to another layer on every host.
#   numba's TBB threading layer needs the tbb package, which is not installed.
#     numba builds that layer for x86_64 only: its 0.67.0 aarch64 wheel carries
#     omppool and no tbbpool, so the exemption is stated where the file exists.
#   _crypt is the deprecated crypt module, which nothing here imports; the
#     library it names is absent on distributions that ship only libcrypt.so.2.
ELF_EXEMPTIONS=(
    --exempt 'python/lib/python3*/site-packages/numba/np/ufunc/omppool.*.so:libgomp.so.1.0.0'
    --exempt 'python/lib/python3*/lib-dynload/_crypt.*.so:libcrypt.so.1'
)
if [ "$PPF_LINUX_ARCH" = x86_64 ]; then
    ELF_EXEMPTIONS+=(
        --exempt 'python/lib/python3*/site-packages/numba/np/ufunc/tbbpool.*.so:libtbb.so.12'
    )
fi

# Files left out of the glibc and libstdc++ floors, each named with its reason.
# The floor is what the launcher tells a system it needs, so only a file the
# program does not load in normal use belongs here, and elf-audit.py refuses an
# entry that matches no file.
#   debugpy's attach helper is a prebuilt library that debugpy's
#     py2.py3-none-any wheel ships built against glibc 2.34; pip installs that
#     wheel on any glibc because it claims no platform. pydevd's in-process
#     loader returns before loading it on any Python newer than 3.11, which the
#     bundled interpreter is, so JupyterLab's debugger never loads it. The one
#     remaining use is attaching a debugger to an already running process by
#     PID through gdb, which is what fails on a glibc older than 2.34.
FLOOR_EXEMPTIONS=(
    --floor-exempt 'python/lib/python3*/site-packages/debugpy/_vendored/pydevd/pydevd_attach_to_process/attach_linux_amd64.so'
)

# Files gate E leaves alone: an object for another architecture that the payload
# carries and nothing loads.
#   debugpy's attach helper again. Its py2.py3-none-any wheel ships one Linux
#     build of it, for x86_64, so on any other architecture it is a foreign
#     object. Attaching a debugger to a running process by PID, its one use,
#     cannot work there with or without the file.
MACHINE_EXEMPTIONS=()
if [ "$PPF_LINUX_ARCH" != x86_64 ]; then
    MACHINE_EXEMPTIONS+=(
        --machine-exempt 'python/lib/python3*/site-packages/debugpy/_vendored/pydevd/pydevd_attach_to_process/attach_linux_amd64.so'
    )
fi

AUDIT_LOG="$(mktemp "${TMPDIR:-/tmp}/ppf-elf-audit.XXXXXX")"
# --rocm ONLY for a ROCm payload, which is the point of the flag rather than a
# convenience: it admits libdrm, libdrm_amdgpu, libelf and libnuma as machine
# libraries, and a CUDA distribution that somehow acquired a dependency on one of
# them must still fail gate B rather than inherit an allowance written for
# another backend.
AUDIT_BACKEND_FLAG=()
if has_backend rocm; then
    # THE FILES THE ALLOWANCE APPLIES TO ARE NAMED, because this payload can
    # carry CUDA beside ROCm: the ROCm target directory, the ROCm backend
    # library, and the runtime closure step 1 walked. A CUDA file that acquired
    # one of those dependencies is outside every glob and still fails gate B.
    AUDIT_BACKEND_FLAG=(--rocm
        --rocm-scope "$(backend_target_dir rocm)/release/*"
        --rocm-scope 'bin/libppfbe_rocm.so')
    for runtime_lib in ${RUNTIME_LIBS[@]+"${RUNTIME_LIBS[@]}"}; do
        AUDIT_BACKEND_FLAG+=(--rocm-scope "bin/$(basename "$runtime_lib")")
    done
fi
run_audit() {
    python3 "$BUILD_LINUX/scripts/elf-audit.py" "$PKG" --build-tree "$SRC_DIR" \
        --arch "$PPF_LINUX_ARCH" \
        ${AUDIT_BACKEND_FLAG[@]+"${AUDIT_BACKEND_FLAG[@]}"} \
        "${ELF_EXEMPTIONS[@]}" "${FLOOR_EXEMPTIONS[@]}" \
        ${MACHINE_EXEMPTIONS[@]+"${MACHINE_EXEMPTIONS[@]}"} \
        > "$AUDIT_LOG" 2>&1
}
# Read for the floor here, and again as the gate in step 10 over the final
# payload. A failure here is reported in full at step 10; the floor is only
# taken from what the audit printed.
run_audit || true
MIN_GLIBC="$(awk '$1 == "FLOOR" && $2 == "GLIBC" { print $3 }' "$AUDIT_LOG")"
MIN_GLIBC_FILE="$(awk '$1 == "FLOOR" && $2 == "GLIBC" { print $4 }' "$AUDIT_LOG")"
MIN_GLIBCXX="$(awk '$1 == "FLOOR" && $2 == "GLIBCXX" { print $3 }' "$AUDIT_LOG")"
MIN_GLIBCXX_FILE="$(awk '$1 == "FLOOR" && $2 == "GLIBCXX" { print $4 }' "$AUDIT_LOG")"
printf '%s' "$MIN_GLIBC" | grep -Eq '^[0-9]+\.[0-9]+$' || {
    cat "$AUDIT_LOG"
    die "no glibc floor could be read off the payload" \
        "elf-audit.py's output is above. A floor that cannot be read is not guessed."
}
printf '%s' "$MIN_GLIBCXX" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' || MIN_GLIBCXX="none"
printf '  [OK] minimum glibc %s, raised by %s\n' "$MIN_GLIBC" "$MIN_GLIBC_FILE"
printf '  [OK] minimum libstdc++ symbol version GLIBCXX_%s, raised by %s\n' \
    "$MIN_GLIBCXX" "${MIN_GLIBCXX_FILE:-nothing}"
while IFS= read -r exempt_line; do
    printf '  [--] left out of the floor by name: %s\n' "${exempt_line#FLOOR-EXEMPT }"
done < <(grep '^FLOOR-EXEMPT ' "$AUDIT_LOG" || true)

version_ge() {
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

if [ -n "${PPF_LINUX_MAX_GLIBC:-}" ]; then
    version_ge "$PPF_LINUX_MAX_GLIBC" "$MIN_GLIBC" || die \
        "the payload needs glibc $MIN_GLIBC and PPF_LINUX_MAX_GLIBC is $PPF_LINUX_MAX_GLIBC" \
        "The file that raises it is $MIN_GLIBC_FILE." \
        "A release targets the glibc it names, so a newer requirement is a decision" \
        "about who can run it, not something to wave through here."
    printf '  [OK] within PPF_LINUX_MAX_GLIBC=%s\n' "$PPF_LINUX_MAX_GLIBC"
fi
# The same ceiling for libstdc++, which a distribution upgrades with its compiler
# rather than with its C library, so a glibc that is old enough does not imply a
# libstdc++ that is.
if [ -n "${PPF_LINUX_MAX_GLIBCXX:-}" ] && [ "$MIN_GLIBCXX" != "none" ]; then
    version_ge "$PPF_LINUX_MAX_GLIBCXX" "$MIN_GLIBCXX" || die \
        "the payload needs GLIBCXX_$MIN_GLIBCXX and PPF_LINUX_MAX_GLIBCXX is $PPF_LINUX_MAX_GLIBCXX" \
        "The file that raises it is $MIN_GLIBCXX_FILE." \
        "Build with a compiler whose libstdc++ links the newer symbols statically," \
        "as the gcc-toolset packages of the release container do."
    printf '  [OK] within PPF_LINUX_MAX_GLIBCXX=%s\n' "$PPF_LINUX_MAX_GLIBCXX"
fi

# The CA bundle the launcher points SSL_CERT_FILE at, as a path relative to the
# distribution, asked of the interpreter rather than spelled with a version.
CA_BUNDLE_REL="$("$PKG_PY" -c 'import certifi; print(certifi.where())')" || die \
    "the bundled interpreter could not locate certifi's CA bundle"
CA_BUNDLE_REL="${CA_BUNDLE_REL#"$PKG/"}"
case "$CA_BUNDLE_REL" in
    python/*) [ -f "$PKG/$CA_BUNDLE_REL" ] || die "no CA bundle at $PKG/$CA_BUNDLE_REL" ;;
    *) die "certifi's CA bundle is not inside the distribution: $CA_BUNDLE_REL" ;;
esac
case "$CA_BUNDLE_REL" in
    *[!A-Za-z0-9._/-]*) die "the CA bundle path carries a character the launcher cannot splice: $CA_BUNDLE_REL" ;;
esac

# ---------------------------------------------------------------------------
step "[8/11] Generating the launcher, config.sh and the documentation"
# ---------------------------------------------------------------------------

# The body is a single-quoted heredoc, so nothing in it expands here. The four
# build-time values it needs are printed above it as assignments; each was
# validated against a pattern that admits no quote, space or newline.
{
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' '# Generated by build-linux-native/bundle.sh. Edits are lost on the next'
    printf '%s\n' '# bundle; change bundle.sh instead.'
    printf '%s\n' '#'
    printf '%s\n' '# It starts JupyterLab IN THE FOREGROUND. The server output is this'
    printf '%s\n' "# terminal's and Ctrl+C stops it."
    printf '%s\n' '#'
    printf '%s\n' '# IT WRITES EVERYTHING INSIDE THE DISTRIBUTION DIRECTORY, and nothing outside'
    printf '%s\n' '# it, so removing the directory is a complete uninstall.'
    printf '%s\n' '#'
    printf '%s\n' '# The values below are stamped in when this file is written, read off the'
    printf '%s\n' '# payload: the version from ppf-contact-solver --version, the glibc floor'
    printf '%s\n' '# from the highest version any binary here requires.'
    printf 'PPF_DIST_VERSION=%s\n' "$APP_VERSION"
    printf 'PPF_MIN_GLIBC=%s\n' "$MIN_GLIBC"
    printf 'PPF_MIN_GLIBCXX=%s\n' "$MIN_GLIBCXX"
    printf 'PPF_CA_BUNDLE_REL=%s\n' "$CA_BUNDLE_REL"
    # The architecture from scripts/platform.sh's fixed set, and the backends as
    # step 1 found them in the build, joined with a comma, so neither carries a
    # quote, a space or a newline either.
    printf 'PPF_DIST_ARCH=%s\n' "$PPF_LINUX_ARCH"
    printf 'PPF_DIST_BACKENDS=%s\n' "$(printf '%s' "$DIST_BACKENDS" | tr ' ' ',')"
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

# PATH IS SET FIRST. The system directories go first, so a shim earlier in the
# user's PATH cannot change which cp or grep this script gets, and the user's
# entries are kept after them, so a NOTEBOOK still finds the git that four
# examples clone with. The interpreter and every binary of ours are named by
# absolute path, so PATH never decides which solver runs.
export PATH="/usr/bin:/bin:/usr/sbin:/sbin${PATH:+:$PATH}"

# WHERE THIS FILE LIVES, resolved by a bounded walk of any symlink chain, so it
# runs from any working directory, through a symlink, and from a path holding a
# space. `pwd -P` resolves a symlinked parent.
SELF="${BASH_SOURCE[0]}"
hops=0
while [ -L "$SELF" ]; do
    hops=$((hops + 1))
    [ "$hops" -le 32 ] || die \
        "the path this program was started from is a symlink loop" \
        "Started as: ${BASH_SOURCE[0]}"
    link="$(readlink "$SELF")"
    case "$link" in
        /*) SELF="$link" ;;
        *)  SELF="$(dirname "$SELF")/$link" ;;
    esac
done
ROOT="$(cd "$(dirname "$SELF")" && pwd -P)" || die \
    "this program could not resolve the directory it lives in" \
    "Started as: ${BASH_SOURCE[0]}"

[ "$(uname -s)" = "Linux" ] || die \
    "this program runs on Linux and this machine reports $(uname -s)"
[ "$(uname -m)" = "$PPF_DIST_ARCH" ] || die \
    "this program runs on $PPF_DIST_ARCH and this machine reports $(uname -m)" \
    "Every binary in this folder is built for $PPF_DIST_ARCH. Use the distribution" \
    "built for $(uname -m), where one is published."

# ppf_has_backend NAME: true when this distribution ships backend NAME.
ppf_has_backend() {
    case ",$PPF_DIST_BACKENDS," in
        *",$1,"*) return 0 ;;
    esac
    return 1
}

# WHICH GPU BACKENDS SHIP, IF ANY. That they do is the PPF_DIST_BACKENDS stamp
# above. That each directory holds what the stamp says is READ, NOT ASSUMED:
# .ppf-backend beside the binary states what landed, it is what
# frontend._backends_ reads to choose among them, and bundle.sh writes it from
# the binary's own --backend. Empty where the CPU backend ships alone.
GPU_BACKENDS=""
for candidate in cuda rocm; do
    ppf_has_backend "$candidate" || continue
    recorded="$(cat "$ROOT/target/$candidate/release/.ppf-backend" 2>/dev/null || true)"
    [ "$recorded" = "$candidate" ] || die \
        "this program cannot tell which GPU backend a folder of it ships" \
        "  $ROOT/target/$candidate/release/.ppf-backend says: ${recorded:-<nothing>}" \
        "The download is incomplete. Unpack it again."
    GPU_BACKENDS="${GPU_BACKENDS:+$GPU_BACKENDS }$candidate"
done

NEEDED_FILES=(
    "$ROOT/config.sh"
    "$ROOT/target/cpu/release/ppf-contact-solver"
    "$ROOT/target/cpu/release/ppf-cts-server"
    "$ROOT/target/cpu/release/lib_ppf_cts_py.so"
    "$ROOT/bin/ffmpeg"
)
for backend in $GPU_BACKENDS; do
    NEEDED_FILES+=(
        "$ROOT/target/$backend/release/ppf-contact-solver"
        "$ROOT/target/$backend/release/ppf-cts-server"
        "$ROOT/target/$backend/release/lib_ppf_cts_py.so"
        "$ROOT/bin/libppfbe_$backend.so"
    )
done
for needed in "${NEEDED_FILES[@]}"; do
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

# shellcheck source=/dev/null
. "$ROOT/config.sh" || die \
    "the settings file would not load" \
    "  $ROOT/config.sh" \
    "If you edited it, an unbalanced quote is the usual cause."
[ -n "${PORT:-}" ] || die \
    "the settings file does not set a port" \
    "PORT is missing or empty in $ROOT/config.sh. Give it a number, for example PORT=8080"
printf '%s' "$PORT" | grep -Eq '^[0-9]+$' || die \
    "the settings file sets a port that is not a number: $PORT" \
    "Give PORT in $ROOT/config.sh a number, for example PORT=8080"

usage() {
    printf '%s\n' "ZOZO's Contact Solver $PPF_DIST_VERSION"
    printf '%s\n' "$ROOT"
    printf '\n'
    printf '%s\n' 'Usage:'
    printf '%s\n' '    ./ppf-contact-solver                        start JupyterLab and serve the examples'
    printf '%s\n' '    ./ppf-contact-solver python FILE [ARGS...]  run a Python script with this folder'"'"'s'
    printf '%s\n' '                                                interpreter and environment, no JupyterLab'
    printf '%s\n' '    ./ppf-contact-solver --help                 this message'
    printf '\n'
    printf '%s\n' "It runs in the foreground. JupyterLab's log appears in this terminal, and"
    printf '%s\n' 'Ctrl+C stops it.'
    printf '\n'
    printf '%s\n' 'Settings, all optional:'
    printf '%s\n' "    config.sh         PORT is the port JupyterLab is asked for (currently $PORT)."
    printf '%s\n' '                      If that port is taken, the server takes the next free one.'
    printf '%s\n' '    PPF_CTS_VENV      an alternative Python environment, for a developer who'
    printf '%s\n' '                      wants their own packages instead of the ones in python/.'
    printf '%s\n' "    CARGO_TARGET_DIR  set to <this folder>/target/<backend> to run that backend's"
    printf '%s\n' "                      solver, where <backend> is one of ${PPF_DIST_BACKENDS//,/, }."
    printf '%s\n' '                      Left unset, the backend is chosen when a run starts: the one'
    printf '%s\n' '                      GPU build here, or the first whose solver reports a usable'
    printf '%s\n' '                      device. A value naming anywhere outside this folder is'
    printf '%s\n' '                      ignored.'
    printf '\n'
    printf '%s\n' 'Notebooks are served from examples/ in this folder. Sessions, cached assets'
    printf '%s\n' 'and the Jupyter state are kept under local/ and cache/ in this folder too.'
}

# --help returns before any interpreter or binary starts, which is what lets the
# build and CI run it on a machine that will not serve notebooks. `python` runs
# a script in exactly the environment JupyterLab would get, which is how a
# headless script, and the release verification, use this folder.
RUN_PYTHON=0
if [ "$#" -gt 0 ]; then
    case "$1" in
        --help | -h)
            usage
            exit 0
            ;;
        python)
            RUN_PYTHON=1
            shift
            ;;
        *)
            printf 'ERROR: unknown argument: %s\n' "$1" >&2
            printf '       Run  ./ppf-contact-solver --help  for the usage.\n' >&2
            exit 2
            ;;
    esac
fi

version_ge() {
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

# THE C LIBRARY, CHECKED HERE BECAUSE A LOADER MESSAGE WOULD NAME A SYMBOL. The
# floor is the highest glibc version a binary in this folder requires, so it is
# what they were built to need rather than a preference.
HAVE_LIBC="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
case "$HAVE_LIBC" in
    "glibc "[0-9]*)
        version_ge "${HAVE_LIBC#glibc }" "$PPF_MIN_GLIBC" || die \
            "this system has ${HAVE_LIBC} and this program needs glibc $PPF_MIN_GLIBC or newer" \
            "The minimum is read off the binaries in this folder. A newer distribution" \
            "release is the way to run it."
        ;;
    *)
        die "this system's C library is not glibc (getconf reports: ${HAVE_LIBC:-nothing})" \
            "The binaries in this folder are built against glibc and do not run on" \
            "another C library, such as the musl of Alpine Linux."
        ;;
esac

# THE INHERITED PYTHON SETTINGS ARE CLEARED BEFORE ANY INTERPRETER STARTS, the
# start check below included. A PYTHONHOME left in the shell points the
# interpreter at another installation's standard library, and it then does not
# start at all, which would read as a damaged download. The environment block
# further down says what each of these does.
unset PYTHONHOME PYTHONPATH PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1

# THE INTERPRETER. The override wins and the shipped one is the default.
if [ -n "${PPF_CTS_VENV:-}" ]; then
    PY="$PPF_CTS_VENV/bin/python"
    PY_NOTE="  (PPF_CTS_VENV override)"
    [ -x "$PY" ] || die \
        "PPF_CTS_VENV names an environment with no interpreter in it" \
        "  PPF_CTS_VENV: $PPF_CTS_VENV" \
        "  expected:     $PY" \
        "This refuses rather than falling back to the interpreter in python/." \
        "Unset PPF_CTS_VENV to use the shipped one."
else
    PY="$ROOT/python/bin/python3"
    PY_NOTE=""
    [ -x "$PY" ] || die \
        "this distribution has no Python interpreter" \
        "  expected: $PY" \
        "The download is incomplete. Unpack it again."
fi
"$PY" -c 'pass' >/dev/null 2>&1 || die \
    "the Python interpreter would not start" \
    "  $PY" \
    "The download is incomplete, or was unpacked without its symbolic links."

# THE SOLVERS LOAD ON THIS SYSTEM, ASKED BEFORE ANY SERVER STARTS. --backend is
# answered before any device is opened, so it needs no GPU; what it does need is
# every library the binary links, which is where a libstdc++ older than the one
# these binaries were built against surfaces. The loader's own words are kept.
PROBE_SOLVERS=("$ROOT/target/cpu/release/ppf-contact-solver")
for backend in $GPU_BACKENDS; do
    PROBE_SOLVERS=("$ROOT/target/$backend/release/ppf-contact-solver" "${PROBE_SOLVERS[@]}")
done
for solver in "${PROBE_SOLVERS[@]}"; do
    if ! probe_err="$("$solver" --backend 2>&1 >/dev/null)"; then
        case "$probe_err" in
            *GLIBCXX_* | *CXXABI_*)
                die "this system's libstdc++ is older than this program needs" \
                    "The binaries require libstdc++ symbol version GLIBCXX_$PPF_MIN_GLIBCXX." \
                    "The loader reported: $probe_err" ;;
            *)
                die "a solver in this folder does not start on this system" \
                    "  $solver" \
                    "It reported: $probe_err" ;;
        esac
    fi
done

# THE SOLVER BUILD, CHOSEN ONLY AMONG THIS FOLDER'S OWN. Each backend ships in
# its own target/<backend>, and NOTHING IS PINNED HERE: the frontend searches
# them and resolves which one a run uses (App.get_backend), which is the one
# GPU build present or, where several are, the first whose solver reports a
# usable device. CARGO_TARGET_DIR names one explicitly, and a value naming
# anywhere outside this folder would load a build this folder did not ship, so it
# is dropped with a note rather than obeyed.
TARGET_NOTE="chosen at run time among ${PPF_DIST_BACKENDS//,/, }"
if [ -n "${CARGO_TARGET_DIR:-}" ]; then
    wanted_target="$CARGO_TARGET_DIR"
    if resolved_target="$(cd "$wanted_target" 2>/dev/null && pwd -P)" \
        && [ "$resolved_target" != "$ROOT/target" ] \
        && case "$resolved_target" in "$ROOT/target/"*) true ;; *) false ;; esac \
        && [ -f "$resolved_target/release/lib_ppf_cts_py.so" ]; then
        export CARGO_TARGET_DIR="$resolved_target"
        TARGET_NOTE="${resolved_target#"$ROOT/"}/release  (CARGO_TARGET_DIR)"
    else
        unset CARGO_TARGET_DIR
        printf 'Note: CARGO_TARGET_DIR=%s is not a build inside this folder, so it is ignored.\n' "$wanted_target" >&2
        printf '      Set it to %s/target/cpu for the CPU solver, or unset it.\n' "$ROOT" >&2
    fi
fi

# EVERYTHING THIS PROGRAM WRITES LIVES INSIDE THIS FOLDER. The session data and
# the asset cache are rooted here by the .ppf-selfcontained marker, which every
# entry point reads; the rest is pointed here below.
STATE="$ROOT/local/share/ppf-cts"
NBDIR="$ROOT/examples"
mkdir -p "$STATE" || die \
    "could not create the state folder" \
    "  $STATE" \
    "This program keeps its state inside its own folder, so that folder has to be" \
    "writable. Move it somewhere you own, such as your home directory, and run it again."

# unset PYTHONHOME       a relocatable interpreter computes its home from its own
#                        path, and an inherited PYTHONHOME hides the stdlib
# CARGO_TARGET_DIR       already settled above: kept only when it names a build
#                        inside this folder
# unset PYTHONOPTIMIZE   it redirects every import to a .opt-N.pyc nothing wrote,
#                        and with nowhere to write one every start recompiles
# PYTHONNOUSERSITE       keeps ~/.local site-packages, a user's own numpy among
#                        them, off sys.path
# PYTHONDONTWRITEBYTECODE  the build pre-compiled every .pyc
# PYTHONPATH             assigned, not appended: a PYTHONPATH shadowing frontend,
#                        numpy or scipy changes what the solver computes
# PPF_CTS_BUILD_PYTHON   the build worker's interpreter, or it takes whatever
#                        python3 is on PATH
# JUPYTER_*, IPYTHONDIR  Jupyter's and IPython's state, inside this folder
# MPLCONFIGDIR           matplotlib's configuration and font cache, which would
#                        otherwise land in ~/.cache or ~/.config
# NUMBA_CACHE_DIR        numba's compiled-function cache, which would otherwise
#                        land beside the sources or under the home directory
# SSL_CERT_FILE          the CA bundle shipped here, for the examples that fetch
#                        over HTTPS; the interpreter's OpenSSL is not guaranteed
#                        to find every distribution's own store. A value the
#                        user already set is kept.
# PYTHONHOME, PYTHONOPTIMIZE and PYTHONNOUSERSITE are settled before the
# interpreter check above, and are listed here with the rest.
# LD_LIBRARY_PATH is deliberately left alone. The solver finds its backend
# through the search path recorded in the binary, which the loader consults
# before LD_LIBRARY_PATH, and some systems locate the NVIDIA driver's own
# library through it.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT"
export PPF_CTS_BUILD_PYTHON="$PY"
export JUPYTER_CONFIG_DIR="$STATE/jupyter/config"
export JUPYTER_DATA_DIR="$STATE/jupyter/data"
export JUPYTER_RUNTIME_DIR="$STATE/jupyter/runtime"
export IPYTHONDIR="$STATE/jupyter/ipython"
export MPLCONFIGDIR="$STATE/matplotlib"
export NUMBA_CACHE_DIR="$STATE/numba"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$ROOT/$PPF_CA_BUNDLE_REL}"
if [ -n "${PPF_CTS_VENV:-}" ]; then
    export JUPYTER_PATH="$PPF_CTS_VENV/share/jupyter"
else
    export JUPYTER_PATH="$ROOT/python/share/jupyter"
fi
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" \
    "$IPYTHONDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" || die \
    "could not create the state folders under $STATE"

# `python` ends here, in the environment above and the caller's own directory.
if [ "$RUN_PYTHON" -eq 1 ]; then
    exec "$PY" "$@"
fi

# THE GPU, REPORTED AND NEVER REQUIRED. What this folder needs from the system is
# a working driver, and nothing else; the CPU backend runs without one, and the
# solver refuses by name when a scene asks for a GPU it does not have.
#
# EACH BACKEND IS ASKED IN THE TERMS ITS OWN DRIVER ANSWERS IN, and for ROCm that
# is deliberately NOT rocm-smi: that is a ROCm PACKAGE, and requiring it would put
# an installation back in front of a user this distribution exists to spare one.
# The kernel side is what matters and what is always there when the driver is:
# /dev/kfd is the compute device amdgpu exposes, and it is exactly what the
# solver's own preflight names when it refuses.
backend_note() {
    # $1 = backend name. Each backend is asked in the terms its own driver
    # answers in; the line is a report and never a requirement.
    case "$1" in
        rocm)
            if [ -e /dev/kfd ] && [ -d /sys/class/kfd ]; then
                if [ -r /dev/kfd ]; then
                    note="amdgpu is loaded and /dev/kfd is readable"
                    # Present only with a ROCm install, so it is a bonus and
                    # never a requirement.
                    if command -v rocm-smi >/dev/null 2>&1; then
                        ROCM_LINE="$(timeout 20 rocm-smi --showproductname --csv 2>/dev/null | sed -n '2p')"
                        [ -n "$ROCM_LINE" ] && note="$note ($ROCM_LINE)"
                    fi
                else
                    note="/dev/kfd exists and this user cannot read it, so this backend will refuse; add this user to the render or video group"
                fi
            else
                note="no amdgpu compute device (/dev/kfd is absent), so this backend will refuse"
            fi
            ;;
        *)
            if command -v nvidia-smi >/dev/null 2>&1; then
                # nvidia-smi prints its own failure to stdout, so its status
                # decides whether the line is a GPU or an error.
                if GPU_LINE="$(timeout 20 nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1)"; then
                    note="$(printf '%s\n' "$GPU_LINE" | head -1)"
                    [ -n "$note" ] || note="nvidia-smi is installed and reported no GPU"
                else
                    note="nvidia-smi could not reach a driver, so this backend will refuse: $(printf '%s\n' "$GPU_LINE" | head -1)"
                fi
            else
                note="no NVIDIA driver found (nvidia-smi is not on PATH), so this backend will refuse"
            fi
            ;;
    esac
    printf '%s' "$note"
}

if [ -z "$GPU_BACKENDS" ]; then
    GPU_NOTE="not used: this distribution carries the CPU backend only"
else
    GPU_NOTE=""
    for backend in $GPU_BACKENDS; do
        GPU_NOTE="${GPU_NOTE:+$GPU_NOTE
              }$backend: $(backend_note "$backend")"
    done
fi

# A BROWSER IS OPENED ONLY WHERE THERE IS A DISPLAY. Without one, Python's
# webbrowser module can pick a console browser such as lynx or w3m, which takes
# over this terminal while the server log is printing to it. A remote GPU server
# is the common case, so the URL and an SSH forwarding line are printed instead.
if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    OPEN_BROWSER=True
    BROWSER_NOTE="a browser opens at the URL"
else
    OPEN_BROWSER=False
    BROWSER_NOTE="no display, so no browser is opened; from another machine forward the port with
                ssh -L $PORT:localhost:$PORT <this machine>   and open the URL there"
fi

printf '\n'
printf "ZOZO's Contact Solver %s\n" "$PPF_DIST_VERSION"
printf '  distribution  %s\n' "$ROOT"
printf '  interpreter   %s%s\n' "$PY" "$PY_NOTE"
printf '  solver        %s\n' "$TARGET_NOTE"
printf '  GPU           %s\n' "$GPU_NOTE"
printf '  notebooks     %s\n' "$NBDIR"
printf '  state         %s\n' "$STATE"
printf '  URL           http://localhost:%s/lab\n' "$PORT"
printf '  browser       %s\n' "$BROWSER_NOTE"
printf '\n'
printf 'Starting JupyterLab. Press Ctrl+C to stop it.\n'
printf 'If port %s is taken the server takes the next free one, and the exact URL\n' "$PORT"
printf 'is in the log below.\n'
printf '\n'

SERVER_PID=""
SHUTDOWN_REQUESTED=0

stop_server() {
    # Idempotent: a signal runs it and the EXIT trap runs it again.
    [ -n "$SERVER_PID" ] || return 0
    kill -0 "$SERVER_PID" 2>/dev/null || return 0
    # SIGTERM first: jupyter_server shuts its kernels down on it and exits. The
    # group form also reaches any non-kernel child left in the server's group;
    # a kernel runs in a session of its own and is ended by the server.
    kill -s TERM -- "-$SERVER_PID" 2>/dev/null \
        || kill -s TERM "$SERVER_PID" 2>/dev/null || true
    local n=0
    while [ "$n" -lt 40 ]; do
        kill -0 "$SERVER_PID" 2>/dev/null || return 0
        sleep 0.25
        n=$((n + 1))
    done
    printf 'JupyterLab did not exit within 10 seconds. Killing it.\n' >&2
    kill -s KILL -- "-$SERVER_PID" 2>/dev/null \
        || kill -s KILL "$SERVER_PID" 2>/dev/null || true
}

on_signal() {
    SHUTDOWN_REQUESTED=1
    printf '\nStopping JupyterLab...\n'
    stop_server
}

trap on_signal INT TERM HUP
trap stop_server EXIT

# Job control gives the server a process group of its own, so one signal from
# here reaches it and any child in its group. The two LabApp settings keep the
# interface from fetching a news feed and a version check as it loads.
set -m
"$PY" -m jupyterlab \
    --ServerApp.ip=localhost \
    --ServerApp.port="$PORT" \
    --ServerApp.port_retries=50 \
    --ServerApp.token="" \
    --ServerApp.root_dir="$NBDIR" \
    --ServerApp.open_browser="$OPEN_BROWSER" \
    --ServerApp.answer_yes=True \
    --LabApp.news_url=None \
    --LabApp.check_for_updates_class=jupyterlab.handlers.announcements.NeverCheckForUpdate &
SERVER_PID=$!
[ "$SHUTDOWN_REQUESTED" -eq 0 ] || stop_server

rc=0
wait "$SERVER_PID" || rc=$?
if [ "$SHUTDOWN_REQUESTED" -eq 1 ]; then
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
bash -n "$PKG/ppf-contact-solver" || die \
    "the generated launcher does not parse" \
    "The defect is in the heredoc in bundle.sh, not in the payload."
printf '  [OK] launcher\n'

# THE SHIPPED config.sh IS GENERATED, NOT COPIED: the build tree's config.sh can
# carry settings that name this machine, and the launcher would act on them.
printf '%s' "$PORT" | grep -Eq '^[0-9]+$' || die "PORT is not a number: $PORT"
cat > "$PKG/config.sh" <<EOF
#!/usr/bin/env bash
# Generated by build-linux-native/bundle.sh. This is the distribution's copy.
#
# Source it, do not execute it. Each setting reads an already-exported value
# first, so either can be overridden for one run without editing this file:
#
#     PORT=9000 ./ppf-contact-solver

# The port JupyterLab is asked to listen on. If it is taken, the server takes
# the next free one and prints the real URL.
PORT="\${PORT:-$PORT}"

# An alternative Python environment, for a developer who wants their own
# packages instead of the ones in python/. Empty means the shipped interpreter.
PPF_CTS_VENV="\${PPF_CTS_VENV:-}"
EOF
if grep -Eq '^[^#]*=[^#]*(/home/|/root/|/tmp/)' "$PKG/config.sh"; then
    die "the generated config.sh assigns an absolute path from this machine"
fi
printf '  [OK] config.sh (PORT %s)\n' "$PORT"

# The license texts, gathered from what they cover rather than restated.
if has_backend cuda; then
    cp "$CUDA_ROOT/LICENSE" "$PKG/licenses/NVIDIA-CUDA-LICENSE.txt"
fi
if has_backend rocm; then
    # THE ROCm RUNTIME IS REDISTRIBUTED AS FILES, so its terms travel with them
    # rather than covering something linked in. Taken from the SDK the libraries
    # themselves came from, so the text matches the binaries: a license copied
    # from anywhere else is a statement about a different build.
    #
    # An empty closure here would mean a ROCm backend that loads no HIP library,
    # which is not a thing that happens; said plainly rather than left to `set -u`
    # to report as an unbound variable ten lines from the cause.
    [ "${#RUNTIME_LIBS[@]}" -gt 0 ] || die \
        "the ROCm backend library loads no HIP runtime library" \
        "Step 1 walked libppfbe_rocm.so and found nothing outside the system set," \
        "which for a ROCm build means the walk is wrong or the library is not" \
        "the one that was built. Neither is fixed by skipping the licenses."
    # THE SDK ROOT IS FOUND FROM THE FIRST SHIPPED LIBRARY, by walking up to the
    # directory that holds share/doc: TheRock keeps its runtime libraries in
    # <root>/lib and its bundled system libraries one level deeper, under
    # lib/rocm_sysdeps, so a fixed number of dirname calls lands in the wrong
    # place for one of them.
    #
    # EVERY LICENSE TEXT THE SDK CARRIES IS COPIED, not a list of components. The
    # closure spans HIP, the HSA runtime, comgr, the profiler hook, the kpack
    # loader and the bundled libdrm, libelf and libnuma, and a list written here
    # would silently miss the next component a TheRock release adds. The copies
    # are named by the directory each text documents, so two texts with one file
    # name cannot overwrite each other.
    rocm_license_root=""
    rocm_probe="$(dirname "${RUNTIME_LIBS[0]}")"
    while [ "$rocm_probe" != "/" ]; do
        if [ -d "$rocm_probe/share/doc" ]; then
            rocm_license_root="$rocm_probe"
            break
        fi
        rocm_probe="$(dirname "$rocm_probe")"
    done
    [ -n "$rocm_license_root" ] || die \
        "no directory above ${RUNTIME_LIBS[0]} holds share/doc" \
        "The license texts are located from the SDK the runtime libraries came from."
    rocm_licenses_found=0
    while IFS= read -r rocm_license; do
        [ -n "$rocm_license" ] || continue
        rocm_rel="${rocm_license#"$rocm_license_root/"}"
        rocm_name="ROCm-$(printf '%s' "${rocm_rel%/*}" | sed 's|^share/doc/||; s|/share/doc/|-|g; s|/|-|g')-${rocm_license##*/}"
        cp "$rocm_license" "$PKG/licenses/$rocm_name"
        rocm_licenses_found=$((rocm_licenses_found + 1))
    done <<< "$(find "$rocm_license_root" -path '*/share/doc/*' -type f \
        \( -iname 'LICENSE*' -o -iname 'COPYING*' -o -iname 'NOTICE*' \) | LC_ALL=C sort)"
    [ "$rocm_licenses_found" -gt 0 ] || die \
        "no ROCm license text found under $rocm_license_root/share/doc" \
        "The HIP runtime libraries are REDISTRIBUTED by this distribution, so" \
        "their terms have to travel with them. Do not ship the libraries without them."
    printf '  [OK] %s ROCm license texts from %s\n' "$rocm_licenses_found" "$rocm_license_root"
fi
# The source record of every wheel warmup.sh built from pinned upstream source:
# the repositories, commits and compiler flags the installed package came from.
PKG_PY_TAG="$("$PKG_PY" -c 'import sys; print("cp%d%d" % sys.version_info[:2])')" || die \
    "the bundled interpreter could not report its version"
for pkg in $PPF_LINUX_SOURCE_WHEELS; do
    sources="$(find "$BUILD_LINUX/wheels" -maxdepth 1 \
        -name "$pkg-*-$PKG_PY_TAG-$PKG_PY_TAG-linux_$PPF_LINUX_ARCH.sources.txt" 2>/dev/null | head -1 || true)"
    [ -n "$sources" ] && [ -s "$sources" ] || die \
        "no source record for the $pkg wheel in $BUILD_LINUX/wheels" \
        "warmup.sh builds $pkg from pinned source on $PPF_LINUX_ARCH and writes the" \
        "record beside the wheel. Re-run warmup.sh."
    cp "$sources" "$PKG/licenses/$pkg-sources.txt"
done
cp "$STDLIB/LICENSE.txt" "$PKG/licenses/CPython-LICENSE.txt"
cp "$FFMPEG_DIR/licenses/"* "$PKG/licenses/"
cp "$FFMPEG_DIR/ffmpeg.sources.txt" "$PKG/licenses/ffmpeg-sources.txt"
cp "$SRC_DIR/LICENSE" "$PKG/licenses/ZOZO-Contact-Solver-LICENSE.txt"
printf '  [OK] licenses/\n'

# The README's lines that depend on what ships, composed here so the text below
# stays one heredoc.
README_GPU_REQUIREMENT=""
# `if`, NOT `cond && cmd`, BECAUSE THIS IS THE LAST COMMAND IN THE SUBSTITUTION.
# Under `set -e` a command substitution hands its exit status to the assignment,
# so `has_backend rocm && printf ...` aborted the whole script wherever ROCm is
# absent: the `&&` short-circuits, the substitution ends 1, and the assignment
# fails. It was silent, because an errexit abort prints nothing. Measured in CI
# on 2026-09-16: the aarch64 job, which builds `cuda cpu`, died immediately
# after `[OK] licenses/` with no message, while x86_64 passed the same line
# because it ships ROCm and the printf ran. `if cond; then cmd; fi` is 0 either
# way.
README_BINARIES="bin/                 ffmpeg$(
    for backend in ${GPU_BACKENDS[@]+"${GPU_BACKENDS[@]}"}; do
        printf ', the %s backend library' "$backend"
    done
    if has_backend rocm; then printf ' and the HIP runtime it loads'; fi
)"
for entry in "${SHIPPED[@]}"; do
    README_BINARIES="$README_BINARIES
$(printf '%-20s' "${entry%:*}/")the ${entry##*:} solver, the server, and the Python extension"
done
# The solver a reader is pointed at is the launcher's own choice, which is why
# the line names the rule rather than a directory.
README_SOLVER="the solver under target/<backend>/release chosen when a run starts"
if has_backend cuda; then
    README_GPU_REQUIREMENT="- For the CUDA backend: an NVIDIA GPU and its driver. Nothing else. The CUDA
  runtime is built into the solver's backend library in bin/, so no CUDA
  toolkit is needed, and one that is installed is not used."
fi
if has_backend rocm; then
    # THE ROCm STORY IS THE SAME SHAPE AND THE FILES ARE NOT. The user supplies a
    # driver and nothing else, as with CUDA; what differs is that the HIP runtime
    # is SHIPPED here rather than linked in, so bin/ holds more than one library
    # and the text says so rather than implying a single file.
    README_GPU_REQUIREMENT="${README_GPU_REQUIREMENT:+$README_GPU_REQUIREMENT
}- For the ROCm backend: an AMD GPU the amdgpu kernel driver supports, and that
  driver. Nothing else. No ROCm installation is needed: the HIP runtime ships
  in bin/, and a ROCm installation that is present is not used.
- Your user account needs read access to /dev/kfd, which on most distributions
  means membership of the render or video group."
fi
if [ -n "$README_GPU_REQUIREMENT" ]; then
    README_GPU_REQUIREMENT="$README_GPU_REQUIREMENT
- Which backend a run uses is chosen when it starts: the one GPU build here, or
  the first whose solver reports a usable device. Without a supported GPU the
  CPU backend still runs, much more slowly."
else
    README_GPU_REQUIREMENT="- No GPU. This distribution carries the CPU backend only."
fi

cat > "$PKG/README.txt" <<EOF
============================================================
ZOZO's Contact Solver, Linux distribution
============================================================

QUICK START
-----------
In a terminal, change into this folder and run the launcher:

    cd /path/to/this/folder
    ./ppf-contact-solver

JupyterLab starts and its log appears in that terminal. Open the URL it
prints, pick a notebook and run it. Ctrl+C in that terminal stops the server.

On a machine without a display, such as a remote GPU server, no browser is
opened. Forward the port from your own computer and open the URL there:

    ssh -L 8080:localhost:8080 user@gpu-server

REQUIREMENTS
------------
- Linux on $PPF_LINUX_ARCH, with glibc $MIN_GLIBC or newer. That number is read off the
  binaries in this folder, and the launcher checks it and refuses by name on
  an older system.
$README_GPU_REQUIREMENT
- Nothing else to start it. This folder carries its own Python interpreter and
  every package the frontend needs. It installs nothing on your system, adds
  nothing to your PATH, and reaches the network for nothing of its own.

IF IT WILL NOT START
--------------------
"Permission denied" means the executable bit did not survive the way this
folder was unpacked. Restore it with:  chmod +x ppf-contact-solver

The launcher names what is wrong in every other case: an older glibc, a missing
file, or a solver that does not load on this system.

CONFIGURATION
-------------
config.sh sets the port JupyterLab is asked for:

    PORT="\${PORT:-$PORT}"

To change it for one run:  PORT=9000 ./ppf-contact-solver

PPF_CTS_VENV names an alternative Python environment, for a developer who wants
their own packages instead of the ones in python/.

STOPPING IT
-----------
Ctrl+C in the terminal running it. That asks JupyterLab to shut down, which
ends its kernels too, waits ten seconds, and then kills it if it has not gone.
File > Shut Down inside JupyterLab also ends the server.

WHAT SOME EXAMPLES NEED, WHICH THIS PROGRAM DOES NOT
---------------------------------------------------
Several examples fetch the mesh they simulate the first time they run. Four of
them also clone a repository and need the git command: large-fluffy,
large-animals, fitting and trapped-919539a. Install git with your package
manager to run those. The other examples need, at most, a network connection
on their first run.

Video export uses the ffmpeg in bin/.

WHAT IT WRITES, AND HOW TO REMOVE IT
------------------------------------
Everything it writes is inside this folder:

    local/share/ppf-cts/   sessions, and the Jupyter, IPython, matplotlib and
                           numba state
    cache/ppf-cts/         meshes and tetrahedralizations the examples fetch,
                           and the NVIDIA driver's compute cache
    examples/              the notebooks, which save in place

Removing this folder removes all of it. Nothing is installed anywhere else, and
no PATH, shell profile or service is changed.

CONTENTS
--------
ppf-contact-solver   the launcher, which is what you run
config.sh            the port, and the PPF_CTS_VENV override
README.txt           this file
THIRD_PARTY_LICENSES.txt, licenses/
$README_BINARIES
python/              the bundled interpreter and its packages
frontend/            the Python frontend package
examples/            the example notebooks
crates/              source roots the server reads log-channel names from
.git/                one file, branch_name.txt, naming the branch this was
                     built from. It is not a repository.

Two files carry the name ppf-contact-solver. The one at the top of this folder
is the launcher. $README_SOLVER is the solver itself.

LICENSE
-------
See THIRD_PARTY_LICENSES.txt and licenses/.
EOF
printf '  [OK] README.txt\n'

# THE GPU RUNTIME SECTION IS BUILT HERE, because what this distribution
# redistributes differs by backend and a license file is not a place to be
# approximately right. CUDA is LINKED IN and no toolkit file ships; ROCm's
# libraries are SHIPPED AS FILES, which is a redistribution and names them. A
# distribution that carries no GPU backend has no such section.
#
# The sections that depend on what ships are composed here so the text below
# stays one heredoc. Each ends in the two blank lines that separate sections.
THIRD_PARTY_GPU=""
if has_backend cuda; then
    THIRD_PARTY_GPU="NVIDIA CUDA runtime
-------------------
bin/libppfbe_cuda.so links the CUDA runtime statically, under the NVIDIA license
that ships with the CUDA Toolkit components it was built from. See
licenses/NVIDIA-CUDA-LICENSE.txt. The NVIDIA driver is not redistributed and
must be installed on the system.


"
fi
if has_backend rocm; then
    THIRD_PARTY_GPU="${THIRD_PARTY_GPU}AMD ROCm runtime
----------------
bin/libppfbe_rocm.so is the ROCm backend, and unlike the CUDA one it loads the HIP
runtime dynamically, so these files are REDISTRIBUTED here:
$(for runtime_lib in ${RUNTIME_LIBS[@]+"${RUNTIME_LIBS[@]}"}; do
    printf '  bin/%s\n' "$(basename "$runtime_lib")"
done)
They are AMD ROCm components, under the license texts in licenses/ that begin
ROCm-. The AMD kernel driver (amdgpu) and its user-mode firmware are NOT
redistributed and must be present on the system.


"
fi
THIRD_PARTY_SOURCE_WHEELS=""
if [ -n "$PPF_LINUX_SOURCE_WHEELS" ]; then
    THIRD_PARTY_SOURCE_WHEELS="Packages built from source
--------------------------
No binary wheel of $PPF_LINUX_SOURCE_WHEELS exists for $PPF_LINUX_ARCH, so each was built
from its upstream repository at a pinned commit, unmodified. The repositories,
commits and compiler flags are in licenses/<package>-sources.txt, and each
package's own license travels in its dist-info directory under python/.


"
fi

cat > "$PKG/THIRD_PARTY_LICENSES.txt" <<EOF
============================================================
THIRD PARTY LICENSES
============================================================

ZOZO's Contact Solver
---------------------
Copyright 2025 Ryoichi Ando (ZOZO, Inc.)
Licensed under the Apache License, Version 2.0.
See licenses/ZOZO-Contact-Solver-LICENSE.txt.


${THIRD_PARTY_GPU}CPython
-------
python/ is a redistributed CPython, a python-build-standalone install_only
build taken unmodified from:

  $FILE_PYTHON

See licenses/CPython-LICENSE.txt, and the interpreter tree's own license files
for the libraries compiled into it.


ffmpeg, x264 and zlib
---------------------
bin/ffmpeg is built with --enable-gpl --enable-libx264, so it is distributed
under the GNU General Public License, version 2 or later. It links x264 (GPL)
and zlib (zlib license) statically. The exact source revisions are recorded in
licenses/ffmpeg-sources.txt, and the license texts are in licenses/.


${THIRD_PARTY_SOURCE_WHEELS}Python packages
---------------
INCOMPLETE. python/lib/python*/site-packages carries the frontend's dependencies
and JupyterLab, each under its own license, enumerated by

  python/bin/python3 -m pip list

and not transcribed into this file yet. It must be before this distribution is
published outside the project.


Rust dependencies
-----------------
INCOMPLETE. The binaries in target/ statically link the crates in the
repository's Cargo.lock, each under its own license. That set is not enumerated
in this file yet, and it must be before this distribution is published outside
the project.
EOF
printf '  [OK] THIRD_PARTY_LICENSES.txt\n'

# ---------------------------------------------------------------------------
step "[9/11] Pre-compiling bytecode"
# ---------------------------------------------------------------------------

# unchecked-hash makes every .pyc valid whatever mtime an unpacker gives the
# sources; -s strips the build path from what each .pyc records, which is what
# keeps gate C quiet on them. Vendored code is compiled with SyntaxWarning
# suppressed and the frontend without, so a warning about this project's own
# code still shows.
compile_tree() {
    local action="$1"
    shift
    "$PKG_PY" -W "$action" -m compileall -q -f \
        --invalidation-mode unchecked-hash -s "$PKG" "$@" || die \
        "compileall failed" \
        "It names the file it could not compile above."
}
compile_tree ignore::SyntaxWarning "$STDLIB" "$SITE"
compile_tree default "$PKG/frontend"
printf '  [OK] bytecode compiled\n'

# ---------------------------------------------------------------------------
step "[10/11] Verifying the distribution is self-contained"
# ---------------------------------------------------------------------------

# GATES A, B AND D, AND THE WALK, over the final payload. elf-audit.py says what
# each one checks; its own count line comes first.
run_audit || {
    cat "$AUDIT_LOG"
    rm -f "$AUDIT_LOG"
    die "the distribution's libraries are not self-contained" \
        "The findings are above. Gate B's allowlist and the exemptions above are" \
        "deliberate statements; a finding is a library that did not travel or a" \
        "decision to make, never a list entry to add so the build passes."
}
grep -E '^(elf-audit|FLOOR)' "$AUDIT_LOG" | sed 's/^/  /'
rm -f "$AUDIT_LOG"

# GATE C. No file anywhere names the build tree. This catches what a dynamic
# section cannot show: an absolute shebang, a baked path in a config file, and a
# path compiled into a binary and resolved at run time.
#
# Two files are admitted, by name and for one string: each cdylib publishes
# env!("CARGO_MANIFEST_DIR") as __build_manifest_dir__, a provenance stamp that
# nothing resolves as a path. A file is admitted only while that stamp is the
# ONLY build-tree string it carries.
GATE_C_ALLOWED=()
for entry in "${SHIPPED[@]}"; do
    GATE_C_ALLOWED+=("$PKG/${entry%:*}/lib_ppf_cts_py.so")
done
GATE_C_STAMP="$SRC_DIR/crates/ppf-cts-py"

count_occurrences() {
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

printf 'Scanning every file for the build-tree path...\n'
GATE_C_ERR="$(mktemp "${TMPDIR:-/tmp}/ppf-gate-c.XXXXXX")"
gate_c_rc=0
GATE_C_HITS="$(grep -rlF -- "$SRC_DIR" "$PKG" 2>"$GATE_C_ERR")" || gate_c_rc=$?
# grep exits 1 for no match and 2 for a file it could not read; collapsing the
# two would make an unreadable payload look clean.
if [ "$gate_c_rc" -gt 1 ] || [ -s "$GATE_C_ERR" ]; then
    sed 's/^/       /' "$GATE_C_ERR" >&2
    rm -f "$GATE_C_ERR"
    die "gate C could not read the whole distribution (grep exited $gate_c_rc)"
fi
rm -f "$GATE_C_ERR"

escaped=0
while IFS= read -r hit; do
    [ -n "$hit" ] || continue
    hit_is_allowed=0
    for allowed in "${GATE_C_ALLOWED[@]}"; do
        [ "$hit" = "$allowed" ] && hit_is_allowed=1 && break
    done
    if [ "$hit_is_allowed" -eq 1 ]; then
        hit_strings="$(strings -a "$hit")"
        total_refs="$(printf '%s\n' "$hit_strings" | count_occurrences "$SRC_DIR")"
        stamp_refs="$(printf '%s\n' "$hit_strings" | count_occurrences "$GATE_C_STAMP")"
        if [ "$total_refs" -eq 0 ] || [ "$total_refs" -ne "$stamp_refs" ]; then
            printf 'ERROR: [gate C] %s names the build tree %s times, %s of them the provenance stamp\n' \
                "${hit#"$PKG/"}" "$total_refs" "$stamp_refs" >&2
            printf '%s\n' "$hit_strings" | grep -F -- "$SRC_DIR" | cut -c 1-200 \
                | sed 's/^/         /' >&2 || true
            escaped=1
            continue
        fi
        printf '  [--] %s carries only the __build_manifest_dir__ stamp (%s times). Allowed by name.\n' \
            "${hit#"$PKG/"}" "$stamp_refs"
        continue
    fi
    printf 'ERROR: [gate C] %s names the build tree %s times, for example:\n' \
        "${hit#"$PKG/"}" "$(strings -a "$hit" | count_occurrences "$SRC_DIR")" >&2
    strings -a "$hit" | grep -F -- "$SRC_DIR" | sort -u | head -5 | cut -c 1-200 \
        | sed 's/^/         /' >&2 || true
    escaped=1
done <<EOF
$GATE_C_HITS
EOF
[ "$escaped" -eq 0 ] || die \
    "the distribution is not self-contained" \
    "The files above name $SRC_DIR." \
    "Do not widen gate C to get past a hit: a path compiled into a binary is" \
    "dead on every other machine, and one that is only a diagnostic still names" \
    "the machine that built it."
printf '  [OK] [gate C] no file names the build tree beyond the named stamp\n'

printf 'Checking for broken symlinks...\n'
BROKEN_LINKS="$(find "$PKG" -type l ! -exec test -e {} \; -print)"
[ -z "$BROKEN_LINKS" ] || die \
    "the payload carries a symlink that points at nothing" \
    "$BROKEN_LINKS"
printf '  [OK] every symlink resolves\n'

# ---------------------------------------------------------------------------
step "[11/11] Smoke test"
# ---------------------------------------------------------------------------

# THE ENVIRONMENT IS POISONED FIRST, then cleared exactly as the launcher clears
# it, so a missing unset fails here rather than on a user's machine. This block
# and the launcher's environment block are edited together.
SMOKE_STATE="$PKG/local/share/ppf-cts"
(
    export CARGO_TARGET_DIR=/nonexistent/ppf-poison
    export PYTHONHOME=/nonexistent/ppf-poison
    export PYTHONOPTIMIZE=2
    export PYTHONPATH=/nonexistent/ppf-poison

    unset PYTHONHOME
    # CARGO_TARGET_DIR IS LEFT UNSET ON PURPOSE, which is what the launcher does
    # too: the frontend searches this payload's target/<backend> directories and
    # resolves which one a run uses, so the import below exercises that search
    # rather than a directory this script picked.
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
    export MPLCONFIGDIR="$SMOKE_STATE/matplotlib"
    export NUMBA_CACHE_DIR="$SMOKE_STATE/numba"
    "$PKG_PY" -c 'import frontend, jupyterlab; print("  frontend   " + frontend.__file__); print("  jupyterlab " + jupyterlab.__version__)'
) || die \
    "the bundled interpreter could not import frontend and jupyterlab" \
    "The traceback above names which one failed."

"$PKG_PY" - <<'PY' || die "the launcher passes a setting this build's Jupyter does not have"
import sys

from jupyter_server.serverapp import ServerApp
from jupyterlab.labapp import LabApp

traits = ServerApp.class_traits()
wanted = ("ip", "port", "port_retries", "token", "root_dir", "open_browser", "answer_yes")
missing = [name for name in wanted if name not in traits]
lab_traits = LabApp.class_traits()
missing += ["LabApp." + name for name in ("news_url", "check_for_updates_class") if name not in lab_traits]
if missing:
    print("  missing traits: " + ", ".join(missing))
    sys.exit(1)
if not lab_traits["news_url"].allow_none:
    print("  LabApp.news_url no longer allows None")
    sys.exit(1)
from jupyterlab.handlers.announcements import NeverCheckForUpdate  # noqa: F401
print("  jupyter    every setting the launcher passes exists")
PY

# THE SOLVERS START FROM THE PAYLOAD ALONE, with no LD_LIBRARY_PATH from this
# script. Then, where a GPU backend ships, its solver is started with
# LD_LIBRARY_PATH naming a directory that holds a broken copy of its backend
# library: the loader resolves NEEDED entries when the binary starts, before
# --backend is answered, so a start proves the search path step 6 recorded is
# searched ahead of LD_LIBRARY_PATH, and the loader's own report says which file
# it took.
for entry in "${SHIPPED[@]}"; do
    solver="$PKG/${entry%:*}/ppf-contact-solver"
    got="$(env -u LD_LIBRARY_PATH "$solver" --backend)" || die \
        "${solver#"$PKG/"} does not start from the payload alone"
    [ "$got" = "${entry##*:}" ] || die \
        "${solver#"$PKG/"} reports the $got backend, and its directory ships ${entry##*:}"
    printf '  backend    %s: %s\n' "${solver#"$PKG/"}" "$got"
done
for backend in ${GPU_BACKENDS[@]+"${GPU_BACKENDS[@]}"}; do
    backend_lib="${BACKEND_LIBS[$backend]}"
    backend_solver="$PKG/$(backend_target_dir "$backend")/release/ppf-contact-solver"
    POISON_LIB_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ppf-poison-lib.XXXXXX")"
    : > "$POISON_LIB_DIR/$backend_lib"
    LD_LIBRARY_PATH="$POISON_LIB_DIR" "$backend_solver" --backend >/dev/null || {
        rm -rf "$POISON_LIB_DIR"
        die "the $backend solver loads a $backend_lib from LD_LIBRARY_PATH ahead of its own" \
            "Step 6 records DT_RPATH so the shipped library wins; this says it did not."
    }
    loaded="$(LD_DEBUG=libs LD_LIBRARY_PATH="$POISON_LIB_DIR" "$backend_solver" --backend 2>&1 >/dev/null \
        | sed -n "s|^.*calling init: \(.*${backend_lib%.so}[^/]*\)$|\1|p" | head -1)"
    rm -rf "$POISON_LIB_DIR"
    case "$loaded" in
        "$PKG/target/$backend/release/../../../bin/"* | "$PKG/bin/"*)
            printf '  loader     %s taken from bin/, ahead of a poisoned LD_LIBRARY_PATH\n' "$backend_lib" ;;
        *) die "the loader reports $backend_lib from '${loaded:-nowhere}', not from bin/" ;;
    esac
done
for entry in "${SHIPPED[@]}"; do
    server="$PKG/${entry%:*}/ppf-cts-server"
    env -u LD_LIBRARY_PATH "$server" --help >/dev/null || die "${server#"$PKG/"} --help failed"
done
printf '  servers    every shipped server answers --help\n'
"$PKG/bin/ffmpeg" -hide_banner -version >/dev/null || die "bin/ffmpeg does not run"
printf '  ffmpeg     runs\n'

# NOTHING IS WRITTEN OUTSIDE THE FOLDER. The frontend is asked for the paths it
# would use, under an empty HOME, and the HOME is then checked for anything that
# appeared. Both answers must be inside the distribution.
FAKE_HOME="$(mktemp -d "${TMPDIR:-/tmp}/ppf-fake-home.XXXXXX")"
PROBE='from frontend import App, get_cache_dir; print("data:", App.get_data_dirpath()); print("cache:", get_cache_dir())'
# NO CARGO_TARGET_DIR IS PASSED BACK IN. `env -i` clears it, which is the state
# the launcher leaves it in: the frontend searches this payload's
# target/<backend> directories itself.
PROBE_TARGET=()
PROBE_OUT="$(cd "$PKG" && env -i HOME="$FAKE_HOME" PATH=/usr/bin:/bin PYTHONPATH="$PKG" \
    "${PROBE_TARGET[@]}" \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$SMOKE_STATE/matplotlib" \
    NUMBA_CACHE_DIR="$SMOKE_STATE/numba" "$PKG_PY" -c "$PROBE")" || {
    rm -rf "$FAKE_HOME"
    die "the frontend could not report its data and cache paths under a fresh HOME"
}
while IFS= read -r line; do
    [ -n "$line" ] || continue
    case "${line#*: }" in
        "$PKG"/*) printf '  %s\n' "${line/"$PKG"/<dist>}" ;;
        *) rm -rf "$FAKE_HOME"; die "the frontend resolves a path outside the distribution: $line" ;;
    esac
done <<EOF
$PROBE_OUT
EOF
if [ -n "$(find "$FAKE_HOME" -mindepth 1 -print -quit)" ]; then
    find "$FAKE_HOME" -mindepth 1 | head -10 >&2
    rm -rf "$FAKE_HOME"
    die "importing the frontend wrote into a fresh HOME"
fi
rm -rf "$FAKE_HOME"
printf '  state      nothing appeared under a fresh HOME\n'

# The launcher, through a file symlink and a directory symlink whose names carry
# a space. --help returns before any interpreter starts, so this proves the
# launcher parses, resolves its own directory and reads config.sh.
SMOKE_LINKDIR="$(mktemp -d "${TMPDIR:-/tmp}/ppf smoke.XXXXXX")"
PKG_REAL="$(cd "$PKG" && pwd -P)"
ln -s "$PKG/ppf-contact-solver" "$SMOKE_LINKDIR/launcher link"
ln -s "$PKG" "$SMOKE_LINKDIR/dist link"
for probe in "$SMOKE_LINKDIR/launcher link" "$SMOKE_LINKDIR/dist link/ppf-contact-solver"; do
    probe_out="$("$probe" --help)" || { rm -rf "$SMOKE_LINKDIR"; die \
        "the launcher's --help failed through a symlink" "  started as: $probe"; }
    case "$probe_out" in
        *"$PKG_REAL"*) ;;
        *) rm -rf "$SMOKE_LINKDIR"; die \
            "the launcher reached through a symlink did not report the real distribution path" \
            "  started as: $probe" "  expected:   $PKG_REAL" ;;
    esac
done
rm -rf "$SMOKE_LINKDIR"
printf '  launcher   --help answers through a file symlink and a directory symlink\n'

# `ppf-contact-solver python` under a poisoned environment: it must run the
# folder's own interpreter and frontend, take the build CARGO_TARGET_DIR names
# when it names one of this folder's own, and drop a CARGO_TARGET_DIR naming
# anywhere else.
#
# WITH NO CARGO_TARGET_DIR THE EXPECTED DIRECTORY IS THE ONE THE FRONTEND
# SEARCHES FIRST, which is `frontend._load_dirs`'s order: the CPU build, whose
# extension module loads no GPU runtime. Which backend a RUN uses is a separate
# question, answered by `App.get_backend` when the run starts, and
# verify-distribution.sh is what checks that answer against each build.
PYMODE_PROBE='import sys, frontend; print(sys.executable); print(frontend.__file__); print(frontend.artifact_dir())'
DEFAULT_RELEASE="$PKG_REAL/target/cpu/release"
for case_spec in "unset:$DEFAULT_RELEASE" "$PKG/target/cpu:$PKG_REAL/target/cpu/release" \
    "/nonexistent/ppf-poison:$DEFAULT_RELEASE"; do
    target_in="${case_spec%%:*}"
    want_dir="${case_spec#*:}"
    # PPF_CTS_VENV is cleared because the launcher obeys it by design, and what
    # this checks is the interpreter the folder ships: a builder with a developer
    # environment exported would otherwise fail here over a setting, not a defect.
    if [ "$target_in" = "unset" ]; then
        pymode_out="$(cd / && env -u CARGO_TARGET_DIR -u PPF_CTS_VENV PYTHONHOME=/nonexistent/ppf-poison \
            PYTHONPATH=/nonexistent/ppf-poison PYTHONOPTIMIZE=2 \
            "$PKG/ppf-contact-solver" python -c "$PYMODE_PROBE" 2>&1)"
    else
        pymode_out="$(cd / && env -u PPF_CTS_VENV CARGO_TARGET_DIR="$target_in" PYTHONHOME=/nonexistent/ppf-poison \
            PYTHONPATH=/nonexistent/ppf-poison PYTHONOPTIMIZE=2 \
            "$PKG/ppf-contact-solver" python -c "$PYMODE_PROBE" 2>&1)"
    fi || die "ppf-contact-solver python failed with CARGO_TARGET_DIR=$target_in" "$pymode_out"
    pymode_exe="$(printf '%s\n' "$pymode_out" | tail -3 | sed -n 1p)"
    pymode_frontend="$(printf '%s\n' "$pymode_out" | tail -3 | sed -n 2p)"
    pymode_dir="$(printf '%s\n' "$pymode_out" | tail -3 | sed -n 3p)"
    case "$pymode_exe" in "$PKG_REAL/python/"*) ;; *) die \
        "ppf-contact-solver python ran an interpreter outside the folder: $pymode_exe" ;; esac
    case "$pymode_frontend" in "$PKG_REAL/frontend/"*) ;; *) die \
        "ppf-contact-solver python imported a frontend outside the folder: $pymode_frontend" ;; esac
    [ "$(cd "$pymode_dir" && pwd -P)" = "$want_dir" ] || die \
        "with CARGO_TARGET_DIR=$target_in the frontend uses $pymode_dir" "  expected: $want_dir"
done
printf '  launcher   python mode runs the folder'"'"'s interpreter and picks target/cpu only from inside it\n'

# The smoke test created state directories inside the payload; a distribution
# ships without them.
rm -rf "$PKG/local" "$PKG/cache"
printf '  [OK] smoke test\n'

# ---------------------------------------------------------------------------
step "DISTRIBUTION COMPLETE"
# ---------------------------------------------------------------------------
printf 'Distribution:  %s\n' "$PKG"
du -sh "$PKG" 2>/dev/null || true
printf 'Version:       %s\n' "$APP_VERSION"
printf 'Architecture:  %s, backends: %s\n' "$PPF_LINUX_ARCH" "$DIST_BACKENDS"
printf 'Minimum glibc: %s\n' "$MIN_GLIBC"
printf '\nTo run it:\n'
printf '  cd "%s" && ./ppf-contact-solver\n' "$PKG"
printf '\nTo hand it to someone, as one archive whose top entry is this directory:\n'
printf '  tar -C "%s" -czf "%s/%s.tar.gz" "%s"\n' "$DIST" "$DIST" "$PPF_LINUX_DIST_NAME" "$PPF_LINUX_DIST_NAME"
