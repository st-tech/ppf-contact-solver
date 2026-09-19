#!/usr/bin/env bash
# File: build-linux-native/scripts/source-wheel.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Builds one frontend package's wheel from its upstream source, at the commit
# scripts/downloads.txt pins, for one Python interpreter, and checks what it built.
# warmup.sh runs it for each package scripts/platform.sh names, once for the
# developer environment and once for the bundled interpreter.
#
#     scripts/source-wheel.sh PYTHON PACKAGE OUT_DIR
#
# It is safe to run twice: a wheel already in OUT_DIR for this interpreter whose
# source record matches the pins is checked again rather than rebuilt.
#
# THE SOURCE IS FETCHED, NEVER CARRIED. This repository holds no third-party code.
# The package is fetched at its pinned commit into downloads/src/<package>, where
# a host with no route to the repository can be given a copy, and is built from
# there unmodified. Anything a build needs to differ is a compiler flag or an
# environment variable below, never an edit to the source.
#
# FLOATING-POINT CONTRACTION IS OFF, AND CHECKED ON THE RESULT. The packages built
# here carry exact geometric predicates, which are exact only when every product
# is rounded on its own; a fused multiply-add breaks the error-free transformation
# they rest on. GCC and clang both fuse by default on aarch64, where FMA is in the
# base instruction set. A flag the build system dropped would still build
# cleanly, so the built extension's instructions are counted and a single fused
# one fails the build.
#
# THE BUILD TOOLS ARE PINNED BY HASH. scripts/wheel-build-requirements.txt is
# installed with --require-hashes into a throwaway environment and the build runs
# with --no-build-isolation, so no build requirement is resolved at build time.
# PPF_LINUX_WHEELHOUSE (config.sh) replaces the index for that install, as it does
# for warmup.sh.
#
# WHAT IT LEAVES: OUT_DIR/<wheel>.whl, and OUT_DIR/<wheel>.sources.txt naming the
# repositories, commits and flags it was built from, which bundle.sh copies into
# the distribution's licenses/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_LINUX="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCES_ROOT="$BUILD_LINUX/downloads/src"
WORK_ROOT="$BUILD_LINUX/.wheel-build"
CONTRACTION_FLAGS="-ffp-contract=off"

die() {
    printf 'ERROR: %s\n' "$1" >&2
    shift
    for line in "$@"; do
        printf '       %s\n' "$line" >&2
    done
    exit 1
}

[ "$#" -eq 3 ] || die "usage: scripts/source-wheel.sh PYTHON PACKAGE OUT_DIR"
PYTHON="$1"
PACKAGE="$2"
OUT_DIR="$3"

# shellcheck source=build-linux-native/config.sh
. "$BUILD_LINUX/config.sh"
# shellcheck source=build-linux-native/scripts/load-downloads.sh
. "$SCRIPT_DIR/load-downloads.sh"
load_downloads "$SCRIPT_DIR/downloads.txt" || die "could not read scripts/downloads.txt"

# EACH PACKAGE IS DESCRIBED BY WHAT DIFFERS BETWEEN THEM AND NOTHING ELSE: its
# repository and commit, its submodules as path|url|commit, and the extension
# module inside the wheel whose instructions are counted.
case "$PACKAGE" in
    triangle)
        REPO_URL="$URL_TRIANGLE_GIT"
        REPO_COMMIT="$TRIANGLE_COMMIT"
        SUBMODULES=("c|$URL_TRIANGLE_C_GIT|$TRIANGLE_C_COMMIT")
        EXTENSION_GLOB="triangle/core.*.so"
        ;;
    *)
        die "no source build is described for $PACKAGE" \
            "Each package this script builds is named in its case statement, with its" \
            "pins in scripts/downloads.txt."
        ;;
esac

for tool in git cc objdump readelf rsync sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool not found on PATH: $tool" \
        "Run warmup.sh, which names what the host must provide."
done

# Every interpreter this script starts sees only its own installation.
bare_py() {
    env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$@"
}

[ -x "$PYTHON" ] || die "no interpreter at $PYTHON"
PY_TAG="$(bare_py "$PYTHON" -c 'import sys; print("cp%d%d" % sys.version_info[:2])')" || die \
    "$PYTHON does not run"
MACHINE="$(uname -m)"
case "$MACHINE" in
    x86_64) FUSED_PATTERN='[[:space:]]vfn?m(add|sub)[0-9a-z]*[[:space:]]' ; READELF_MACHINE="X86-64" ;;
    aarch64) FUSED_PATTERN='[[:space:]](fn?m(add|sub)|fml[as])[[:space:]]' ; READELF_MACHINE="AArch64" ;;
    *) die "this builds on x86_64 and aarch64, and uname -m reports $MACHINE" ;;
esac
WHEEL_SUFFIX="$PY_TAG-$PY_TAG-linux_$MACHINE"

# The record written beside the wheel, and compared against on the next run.
source_record() {
    local sub path url commit
    printf '%s, built from upstream source for %s\n' "$PACKAGE" "$WHEEL_SUFFIX"
    printf '  %s @ %s\n' "$REPO_URL" "$REPO_COMMIT"
    for sub in "${SUBMODULES[@]}"; do
        IFS='|' read -r path url commit <<<"$sub"
        printf '  submodule %s: %s @ %s\n' "$path" "$url" "$commit"
    done
    printf '  CFLAGS and CXXFLAGS: %s\n' "$CONTRACTION_FLAGS"
    printf '  build tools: scripts/wheel-build-requirements.txt, sha256 %s\n' \
        "$(sha256sum "$SCRIPT_DIR/wheel-build-requirements.txt" | cut -d' ' -f1)"
}

# check_wheel WHEEL: the one extension module inside is built for this host and
# carries no fused multiply-add instruction.
check_wheel() {
    local wheel="$1" scratch extension_rel machine fused
    scratch="$(mktemp -d "${TMPDIR:-/tmp}/ppf-wheel-check.XXXXXX")"
    if ! extension_rel="$(bare_py "$PYTHON" - "$wheel" "$scratch" "$EXTENSION_GLOB" <<'PY'
import fnmatch
import sys
import zipfile

wheel, destination, pattern = sys.argv[1:4]
with zipfile.ZipFile(wheel) as archive:
    names = [name for name in archive.namelist() if fnmatch.fnmatch(name, pattern)]
    if len(names) != 1:
        sys.exit("%s holds %d files matching %s, and exactly one extension module is expected"
                 % (wheel, len(names), pattern))
    archive.extract(names[0], destination)
    print(names[0])
PY
    )"; then
        rm -rf "$scratch"
        return 1
    fi
    machine="$(readelf -h "$scratch/$extension_rel" | awk -F: '/Machine:/ { sub(/^[ \t]+/, "", $2); print $2 }')"
    case "$machine" in
        *"$READELF_MACHINE"*) ;;
        *)
            printf '%s is built for %s, and this host is %s\n' "$extension_rel" "${machine:-nothing}" "$MACHINE" >&2
            rm -rf "$scratch"
            return 1
            ;;
    esac
    # grep -c prints 0 and exits 1 when nothing matches, which is the passing case.
    fused="$(objdump -d --no-show-raw-insn "$scratch/$extension_rel" | grep -cE "$FUSED_PATTERN" || true)"
    rm -rf "$scratch"
    printf '  %s: %s, %s fused multiply-add instructions\n' "$(basename "$wheel")" "$machine" "$fused"
    if [ "$fused" -ne 0 ]; then
        printf '%s carries %s fused multiply-add instructions, and its predicates are exact only without them\n' \
            "$extension_rel" "$fused" >&2
        return 1
    fi
}

mkdir -p "$OUT_DIR"
EXPECTED_RECORD="$(source_record)"

shopt -s nullglob
existing=("$OUT_DIR/$PACKAGE"-*-"$WHEEL_SUFFIX".whl)
shopt -u nullglob
[ "${#existing[@]}" -le 1 ] || die \
    "$OUT_DIR holds ${#existing[@]} $PACKAGE wheels for $WHEEL_SUFFIX" \
    "Remove them and re-run:  rm -f $OUT_DIR/$PACKAGE-*-$WHEEL_SUFFIX.*"
if [ "${#existing[@]}" -eq 1 ] \
    && [ "$(cat "${existing[0]%.whl}.sources.txt" 2>/dev/null || true)" = "$EXPECTED_RECORD" ]; then
    printf 'Found %s, built from the pinned sources. Checking rather than rebuilding.\n' \
        "$(basename "${existing[0]}")"
    check_wheel "${existing[0]}" || die \
        "the $PACKAGE wheel in $OUT_DIR does not verify (see above)" \
        "Remove it and re-run:  rm -f ${existing[0]%.whl}.*"
    exit 0
fi

# ---------------------------------------------------------------------------
# The source, at its pins
# ---------------------------------------------------------------------------

SRC="$SOURCES_ROOT/$PACKAGE"
if [ ! -d "$SRC/.git" ]; then
    # Fetched into a .part directory first, so an interrupted fetch is never
    # taken for a complete one.
    printf 'Fetching %s at %s\n' "$REPO_URL" "$REPO_COMMIT"
    rm -rf "$SRC.part"
    mkdir -p "$SOURCES_ROOT"
    git init -q "$SRC.part"
    git -C "$SRC.part" fetch -q --depth 1 "$REPO_URL" "$REPO_COMMIT" || die \
        "could not fetch $REPO_COMMIT from $REPO_URL" \
        "On a host that cannot reach it, place a clone at that commit, with its" \
        "submodules, in $SRC and re-run."
    git -C "$SRC.part" -c advice.detachedHead=false checkout -q FETCH_HEAD
    for sub in "${SUBMODULES[@]}"; do
        IFS='|' read -r path url commit <<<"$sub"
        printf 'Fetching submodule %s from %s at %s\n' "$path" "$url" "$commit"
        rm -rf "${SRC:?}.part/$path"
        git init -q "$SRC.part/$path"
        git -C "$SRC.part/$path" fetch -q --depth 1 "$url" "$commit" || die \
            "could not fetch $commit from $url"
        git -C "$SRC.part/$path" -c advice.detachedHead=false checkout -q FETCH_HEAD
    done
    mv "$SRC.part" "$SRC"
fi

SOURCE_REPAIR="rm -rf $SRC && $0 $*"
[ "$(git -C "$SRC" rev-parse HEAD)" = "$REPO_COMMIT" ] || die \
    "$SRC is not at the pinned commit $REPO_COMMIT" "Repair with:  $SOURCE_REPAIR"
[ -z "$(git -C "$SRC" status --porcelain --ignore-submodules=all)" ] || die \
    "$SRC carries local changes" \
    "The build uses upstream's source unmodified. Repair with:  $SOURCE_REPAIR"
for sub in "${SUBMODULES[@]}"; do
    IFS='|' read -r path url commit <<<"$sub"
    recorded="$(git -C "$SRC" ls-tree HEAD "$path" | awk '$2 == "commit" { print $3 }')"
    [ "$recorded" = "$commit" ] || die \
        "$REPO_URL at $REPO_COMMIT records $path at ${recorded:-nothing}, and downloads.txt pins $commit" \
        "A submodule pin that disagrees with its parent tree builds a combination" \
        "upstream never had. Move the two pins together."
    [ "$(git -C "$SRC/$path" rev-parse HEAD 2>/dev/null)" = "$commit" ] || die \
        "$SRC/$path is not at the pinned commit $commit" "Repair with:  $SOURCE_REPAIR"
    [ -z "$(git -C "$SRC/$path" status --porcelain)" ] || die \
        "$SRC/$path carries local changes" "Repair with:  $SOURCE_REPAIR"
done
printf 'Source: %s at %s\n' "$SRC" "$REPO_COMMIT"

# ---------------------------------------------------------------------------
# The compiler and the interpreter's headers
# ---------------------------------------------------------------------------

INCLUDE_DIR="$(bare_py "$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
[ -f "$INCLUDE_DIR/Python.h" ] || die \
    "no Python.h for $PYTHON (looked in $INCLUDE_DIR)" \
    "An extension module is compiled against its interpreter's headers. Install them," \
    "for example:  sudo apt install python3-dev    or    sudo dnf install python3-devel"

BUILD_ENV=(CFLAGS="$CONTRACTION_FLAGS" CXXFLAGS="$CONTRACTION_FLAGS")
# THE COMPILER THE INTERPRETER RECORDS NEED NOT EXIST HERE. python-build-standalone
# records the one it was built with (`clang -pthread` in the 20260901 release,
# alongside compile flags GCC also accepts), and setuptools invokes that name.
# setuptools takes CC from the environment instead when it is set, and rewrites
# the leading compiler of its link command to match, so naming the host's is
# the whole change.
SYSCONFIG_CC="$(bare_py "$PYTHON" -c 'import sysconfig; print((sysconfig.get_config_var("CC") or "").split(" ")[0])')"
if [ -n "$SYSCONFIG_CC" ] && ! command -v "$SYSCONFIG_CC" >/dev/null 2>&1; then
    printf '%s records %s as its compiler, which this host does not have; building with cc.\n' \
        "$PYTHON" "$SYSCONFIG_CC"
    BUILD_ENV+=(CC="cc -pthread" CXX="c++ -pthread")
fi

# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------

WORK="$WORK_ROOT/$PACKAGE-$PY_TAG"
rm -rf "$WORK"
mkdir -p "$WORK/dist"
# A copy without the repositories' metadata, so the build's own output lands in
# the work directory and downloads/src stays exactly what was fetched.
rsync -a --exclude .git "$SRC/" "$WORK/src/"

bare_py "$PYTHON" -m venv "$WORK/venv" || die \
    "python -m venv failed for $PYTHON" \
    "The build environment needs the interpreter's venv module (python3-venv)."
PIP_SOURCE=()
if [ -n "${PPF_LINUX_WHEELHOUSE:-}" ]; then
    PIP_SOURCE=(--no-index --find-links "$PPF_LINUX_WHEELHOUSE")
fi
bare_py "$WORK/venv/bin/python" -m pip install -q "${PIP_SOURCE[@]}" \
    --require-hashes --only-binary=:all: -r "$SCRIPT_DIR/wheel-build-requirements.txt" || die \
    "installing the pinned build tools failed" \
    "Every tool and hash is in scripts/wheel-build-requirements.txt."

printf 'Building %s for %s with %s\n' "$PACKAGE" "$WHEEL_SUFFIX" "$CONTRACTION_FLAGS"
if ! env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "${BUILD_ENV[@]}" \
    "$WORK/venv/bin/python" -m pip wheel -v --no-build-isolation --no-deps \
    -w "$WORK/dist" "$WORK/src" > "$WORK/build.log" 2>&1; then
    tail -n 40 "$WORK/build.log" >&2
    die "pip wheel failed; the last lines of $WORK/build.log are above"
fi
# The flag has to have reached a compiler command, not just the environment.
grep -qF -- "$CONTRACTION_FLAGS" "$WORK/build.log" || die \
    "no compiler command in $WORK/build.log carries $CONTRACTION_FLAGS" \
    "The build system did not take CFLAGS, so the result is not the build this" \
    "script describes."

shopt -s nullglob
built=("$WORK/dist/$PACKAGE"-*.whl)
shopt -u nullglob
[ "${#built[@]}" -eq 1 ] || die "pip wheel left ${#built[@]} $PACKAGE wheels in $WORK/dist, and one is expected"
case "$(basename "${built[0]}")" in
    "$PACKAGE"-*-"$WHEEL_SUFFIX".whl) ;;
    *) die "the built wheel $(basename "${built[0]}") is not tagged $WHEEL_SUFFIX" ;;
esac
check_wheel "${built[0]}" || die "the $PACKAGE wheel just built does not verify (see above)"

rm -f "$OUT_DIR/$PACKAGE"-*-"$WHEEL_SUFFIX".whl "$OUT_DIR/$PACKAGE"-*-"$WHEEL_SUFFIX".sources.txt
mv "${built[0]}" "$OUT_DIR/"
printf '%s\n' "$EXPECTED_RECORD" > "$OUT_DIR/$(basename "${built[0]%.whl}").sources.txt"
rm -rf "$WORK"
printf '  [OK] %s in %s\n' "$(basename "${built[0]}")" "$OUT_DIR"
