#!/usr/bin/env bash
# File: build-linux-native/build.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Builds the solver on Linux, the counterpart of build-mac-native/build.sh and
# build-win-native/build.bat. Run warmup.sh once first.
#
# It is safe to run twice: cargo decides what to recompile, and everything else
# here is a check or an overwrite.
#
# ONE BUILD PER BACKEND, EACH IN ITS OWN TARGET DIRECTORY. The backends come from
# scripts/platform.sh: every one this architecture can build, which is
# `cuda rocm cpu` on x86_64 and `cuda cpu` on aarch64, narrowed by
# PPF_LINUX_BACKENDS. Each is built with `--features <backend>` into
# target/<backend>/release, the layout `backend_target_dir` names. Every backend
# links the same executable name and crates/ppf-cts-solver/build.rs refuses to
# put two in one directory, so the directories are what keep them apart, and
# `frontend.get_backend` is what chooses among them when a run starts.
#
# A BUILD WITHOUT THE CUDA BACKEND NEEDS NO TOOLKIT AND CHECKS NONE. Everything
# below about the toolkit applies when CUDA is among the backends.
#
# THE TOOLKIT IS THE ONE warmup.sh PROVISIONED, SELECTED BY PPF_CUDA_ROOT. The
# CUDA recipe compiles through $(PPF_CUDA_ROOT)/bin/nvcc and the solver's build
# script checks that same path's release, so this directory's toolkit is used
# whatever /usr/local/cuda holds on this host. Its bin directory also goes first
# on PATH, which is where the build's FP64 guard finds cuobjdump.
#
# THE TOOLKIT IS NEEDED TO BUILD AND NOT TO RUN. The CUDA library links the CUDA
# runtime statically and loads only the driver's libcuda.so.1 when it starts, so
# a built solver runs on a machine with the NVIDIA driver and no toolkit. Step 3
# checks that property on the artifact, because it is what lets the
# distribution ship no toolkit library at all.
#
# THE ROCm SDK IS THE ONE warmup.sh UNPACKED, SELECTED BY ROCM_PATH, AND THE BUILD
# IS FOR AMD'S PLATFORM. The same HIP sources build for NVIDIA hardware under
# HIP_PLATFORM=nvidia, and that library loads and reports rocm too, so this script
# sets amd and refuses an environment that asks for anything else. ROCm's runtime
# is loaded dynamically, so bundle.sh ships its libraries rather than relying on a
# static link, and step 3 checks the library's ABI and device image instead of a
# toolkit dependency.

set -euo pipefail

BUILD_LINUX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$BUILD_LINUX/.." && pwd)"
LOGFILE="$BUILD_LINUX/build.log"

if [ -z "${PPF_LINUX_BUILD_LOGGING:-}" ]; then
    printf 'Logging to %s\n' "$LOGFILE"
    set +e
    PPF_LINUX_BUILD_LOGGING=1 "${BASH_SOURCE[0]}" "$@" 2>&1 | tee "$LOGFILE"
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

if [ "$#" -gt 0 ]; then
    die "build.sh takes no arguments, and got: $*" \
        "The backends come from PPF_LINUX_BACKENDS in config.sh, resolved for this" \
        "architecture by scripts/platform.sh."
fi

printf '============================================================\n'
printf "  ZOZO's Contact Solver, native Linux build\n"
printf '============================================================\n'
printf 'Build directory:  %s\n' "$BUILD_LINUX"
printf 'Source directory: %s\n' "$SRC_DIR"

# ---------------------------------------------------------------------------
step "[1/4] Checking the host and the provisioned toolkit"
# ---------------------------------------------------------------------------

[ "$(uname -s)" = "Linux" ] || die "this builds on Linux, and uname reports $(uname -s)"
for tool in make g++ readelf nm python3; do
    # python3 renders the neutral kernels (crates/ppf-cts-compute/seam/kernelgen.py)
    # inside every build.
    command -v "$tool" >/dev/null 2>&1 || die "required tool not found on PATH: $tool" \
        "Run warmup.sh, which names what the host must provide."
done

# shellcheck source=build-linux-native/config.sh
. "$BUILD_LINUX/config.sh"
# shellcheck source=build-linux-native/scripts/platform.sh
. "$BUILD_LINUX/scripts/platform.sh"
resolve_platform || die "this host, or PPF_LINUX_BACKENDS, cannot be built (see above)"
printf 'arch:      %s\n' "$PPF_LINUX_ARCH"
printf 'backends:  %s\n' "$PPF_LINUX_BACKENDS"

CUDA_ROOT="$BUILD_LINUX/cuda"
if has_backend cuda; then
    [ -x "$CUDA_ROOT/bin/nvcc" ] || die \
        "no CUDA toolkit at $CUDA_ROOT" \
        "Run warmup.sh first; it assembles the toolkit this build selects."
    export PPF_CUDA_ROOT="$CUDA_ROOT"
    export CUDA_PATH="$CUDA_ROOT"
    export PATH="$CUDA_ROOT/bin:$PATH"

    # Asserted here rather than left to the build: the FP64 SASS guard in the
    # solver's build script only warns when cuobjdump is missing, which would turn
    # one of the release build's always-on gates into nothing.
    [ "$(command -v cuobjdump)" = "$CUDA_ROOT/bin/cuobjdump" ] || die \
        "cuobjdump does not resolve to $CUDA_ROOT/bin/cuobjdump" \
        "It resolves to: $(command -v cuobjdump || printf 'nothing')" \
        "The build's FP64 guard reads the device image with it and only warns when" \
        "it is absent. Re-run warmup.sh."
    printf 'nvcc:      %s\n' "$("$CUDA_ROOT/bin/nvcc" --version | grep -m1 release)"
    printf 'cuobjdump: %s\n' "$(command -v cuobjdump)"
else
    printf 'CUDA:      not among the backends, so no toolkit is needed or checked\n'
fi

ROCM_ROOT="$(rocm_sdk_dir)"
if has_backend rocm; then
    [ -x "$ROCM_ROOT/bin/hipcc" ] || die \
        "no ROCm SDK at $ROCM_ROOT" \
        "Run warmup.sh first; it unpacks the SDK this build selects."
    if [ -n "${HIP_PLATFORM:-}" ] && [ "$HIP_PLATFORM" != amd ]; then
        die "HIP_PLATFORM=$HIP_PLATFORM is set, and a distribution's ROCm backend is built for amd" \
            "The nvidia platform compiles the same sources into a library for NVIDIA" \
            "hardware that still reports rocm. Unset HIP_PLATFORM for this build."
    fi
    export ROCM_PATH="$ROCM_ROOT"
    export HIP_PLATFORM=amd
    printf 'hipcc:     %s\n' "$("$ROCM_ROOT/bin/hipcc" --version | grep -m1 'HIP version')"
fi

RUST_DIR="$BUILD_LINUX/rust"
if [ -x "$RUST_DIR/bin/cargo" ]; then
    printf 'Using the Rust in this directory: %s\n' "$RUST_DIR"
    export PATH="$RUST_DIR/bin:$PATH"
    export CARGO_HOME="$RUST_DIR"
    export RUSTUP_HOME="$RUST_DIR/rustup"
elif [ -x "$HOME/.cargo/bin/cargo" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi
command -v cargo >/dev/null 2>&1 || die \
    "cargo not found" \
    "Run warmup.sh first; it installs Rust into $RUST_DIR when the host has none."
printf 'cargo:     %s (%s)\n' "$(command -v cargo)" "$(cargo --version)"

# NO PATH OF THIS MACHINE IS COMPILED INTO THE RUST BINARIES. rustc records the
# absolute source path of every dependency crate for its panic locations, and
# those crates live under cargo's home, so without this the solver, the server
# and the Python extension carry /home/<user>/.cargo/registry/... or, where
# warmup.sh installed Rust, the build directory itself. The remap replaces that
# prefix with /cargo. It is appended through CARGO_ENCODED_RUSTFLAGS, which
# carries a path with spaces intact, and keeps any flags the caller already set;
# step 3 refuses a binary that still names cargo's home.
CARGO_HOME_REAL="$(cd "${CARGO_HOME:-$HOME/.cargo}" && pwd -P)" || die \
    "cargo's home directory does not exist: ${CARGO_HOME:-$HOME/.cargo}"
REMAP_FLAG="--remap-path-prefix=$CARGO_HOME_REAL=/cargo"
if [ -n "${CARGO_ENCODED_RUSTFLAGS:-}" ]; then
    export CARGO_ENCODED_RUSTFLAGS="$CARGO_ENCODED_RUSTFLAGS"$'\x1f'"$REMAP_FLAG"
else
    ENCODED=""
    # shellcheck disable=SC2086 # RUSTFLAGS is a space-separated flag list by definition
    for flag in ${RUSTFLAGS:-}; do
        ENCODED="$ENCODED$flag"$'\x1f'
    done
    export CARGO_ENCODED_RUSTFLAGS="$ENCODED$REMAP_FLAG"
    unset RUSTFLAGS
fi
printf 'remap:     %s -> /cargo\n' "$CARGO_HOME_REAL"

# ---------------------------------------------------------------------------
step "[2/4] Building with cargo"
# ---------------------------------------------------------------------------

cd "$SRC_DIR"
# EVERY BACKEND INTO ITS OWN TARGET DIRECTORY, the layout `scripts/platform.sh`
# names. They link the same executable name, so `crates/ppf-cts-solver/build.rs`
# refuses to put two in one directory, and a distribution carrying several needs
# them apart. CARGO_TARGET_DIR is set for one command at a time, so no later
# step can read one build while meaning another, and the feature is named rather
# than left to the build script's own search, so what lands in `target/cuda` is
# what this loop asked for.
for backend in $PPF_LINUX_BACKENDS; do
    target="$SRC_DIR/$(backend_target_dir "$backend")"
    printf '\nBuilding the %s backend into %s\n' "$backend" "${target#"$SRC_DIR"/}"
    CARGO_TARGET_DIR="$target" cargo build --release --features "$backend" || die \
        "cargo build --release --features $backend failed" \
        "Where the failure is inside crates/ppf-cts-compute/$backend, the make output is" \
        "above: that Makefile is driven by the solver's build script and compiles the" \
        "backend. The cpu backend is pure Rust and has no such directory."
done

# ---------------------------------------------------------------------------
step "[3/4] Verifying the artifacts"
# ---------------------------------------------------------------------------

# One release directory per backend built, in the order platform.sh lists them.
ARTIFACTS=()
BACKEND_DIRS=()
for backend in $PPF_LINUX_BACKENDS; do
    release_dir="$SRC_DIR/$(backend_target_dir "$backend")/release"
    BACKEND_DIRS+=("$release_dir:$backend")
    for artifact in ppf-contact-solver ppf-cts-server lib_ppf_cts_py.so; do
        ARTIFACTS+=("$release_dir/$artifact")
    done
done
for artifact in "${ARTIFACTS[@]}"; do
    [ -f "$artifact" ] || die \
        "cargo reported success but $artifact is missing" \
        "Every build writes the solver, the server and the Python extension; a" \
        "missing one is something the frontend or the add-on loads."
    printf '  [OK] %s\n' "${artifact#"$SRC_DIR"/}"
    # grep exits 1 when it finds nothing, which is the passing case, and under
    # pipefail that status would end this script with no message.
    cargo_refs="$({ grep -a -o -F -- "$CARGO_HOME_REAL" "$artifact" || true; } | wc -l)"
    [ "$cargo_refs" -eq 0 ] || die \
        "${artifact#"$SRC_DIR"/} names cargo's home $cargo_refs times" \
        "  $CARGO_HOME_REAL" \
        "The remap in step 1 did not reach this build. A distribution carrying it" \
        "names the machine that built it."
done
printf '  [OK] no artifact names cargo'"'"'s home (%s)\n' "$CARGO_HOME_REAL"

# THE CUDA SOLVER HAS TO LINK THE CUDA LIBRARY. Naming a library on the link
# line does not make the binary use it: a build whose dispatch path resolved to
# the host renderings links, runs, computes correct numbers and never touches
# the GPU while answering --backend cuda. The NEEDED entry is the evidence, and
# it is resolved through the binary's own search path rather than a glob over
# target/release/build, where stale output directories coexist.
if has_backend cuda; then
    CUDA_REL="$SRC_DIR/$(backend_target_dir cuda)/release"
    # shellcheck source=build-linux-native/scripts/backend-path.sh
    . "$BUILD_LINUX/scripts/backend-path.sh"
    BACKEND_PATH="$(resolve_needed "$CUDA_REL/ppf-contact-solver" libppfbe_cuda.so)" || die \
        "could not resolve the CUDA backend the solver loads (see above)" \
        "A solver that does not list libppfbe_cuda.so as NEEDED runs its kernels on" \
        "the host while reporting CUDA. Check what build.rs selected above."
    printf '  [OK] backend %s\n' "$BACKEND_PATH"

    # THE BACKEND NEEDS NO TOOLKIT LIBRARY TO RUN. It carries the CUDA runtime
    # statically, which the distribution depends on: nothing from cuda/lib64
    # ships. A dependency on a toolkit library, or an undefined CUDA runtime
    # symbol left for one to provide, would be a machine with only the driver
    # failing to load the solver.
    while IFS= read -r soname; do
        case "$soname" in
            libcudart* | libnvrtc* | libnvJitLink* | libcublas* | libcusparse* | libcurand* | libnvToolsExt*)
                die "$BACKEND_PATH needs the toolkit library $soname at run time" \
                    "The distribution ships no toolkit library, so a machine with only the" \
                    "NVIDIA driver could not load the solver. Link it statically, or change" \
                    "bundle.sh to ship it deliberately." ;;
        esac
    done < <(elf_needed "$BACKEND_PATH")
    UNDEFINED_CUDA="$(nm -D --undefined-only "$BACKEND_PATH" | awk '$2 ~ /^cuda[A-Z]/ { print $2 }' | head -5)"
    [ -z "$UNDEFINED_CUDA" ] || die \
        "$BACKEND_PATH leaves CUDA runtime symbols undefined: $(printf '%s' "$UNDEFINED_CUDA" | tr '\n' ' ')" \
        "Something else would have to provide them at run time, and nothing ships to."
    printf '  [OK] the backend carries the CUDA runtime and needs no toolkit library\n'
fi

# THE ROCm SOLVER HAS TO LINK THE ROCm LIBRARY, BUILT FOR AMD, WITH EVERY TARGET IN
# IT. The same NEEDED evidence as the CUDA branch, and three things only this
# backend needs, since no AMD GPU runs here to stand behind the build:
#   - the library exports the whole C ABI the neutral driver calls, counted from
#     backend_abi.h, because one missing function links and then fails at be_open;
#   - it is AMD's platform: a library built under HIP_PLATFORM=nvidia makes the
#     solver print `linked: hip-nvidia` under --backend while still reporting rocm;
#   - its device image carries every target in rocm_arch.txt with a non-zero
#     instruction count and no FP64 instruction, and no compile selects a
#     fast-math device library. The two scripts under .github/workflows/scripts
#     say what each refuses and why; this is where a distribution build meets them.
if has_backend rocm; then
    ROCM_REL="$SRC_DIR/$(backend_target_dir rocm)/release"
    # shellcheck source=build-linux-native/scripts/backend-path.sh
    . "$BUILD_LINUX/scripts/backend-path.sh"
    BACKEND_PATH="$(resolve_needed "$ROCM_REL/ppf-contact-solver" libppfbe_rocm.so)" || die \
        "could not resolve the ROCm backend the solver loads (see above)" \
        "A solver that does not list libppfbe_rocm.so as NEEDED runs its kernels on" \
        "the host while reporting ROCm. Check what build.rs selected above."
    printf '  [OK] backend %s\n' "$BACKEND_PATH"

    ABI_WANT="$(grep -oE '\bbe_[a-z_]+\(' "$SRC_DIR/crates/ppf-cts-solver/src/kernels/seam/backend_abi.h" \
        | sort -u | wc -l | tr -d ' ')"
    ABI_GOT="$({ nm -D --defined-only "$BACKEND_PATH" | grep -c ' T be_'; } || true)"
    [ "$ABI_WANT" -gt 0 ] || die "parsed no be_* function out of backend_abi.h"
    [ "$ABI_GOT" = "$ABI_WANT" ] || die \
        "$BACKEND_PATH exports $ABI_GOT of the $ABI_WANT backend ABI functions" \
        "A library short of one links and then fails at be_open on a user's machine."
    printf '  [OK] the backend exports all %s ABI functions\n' "$ABI_WANT"

    # THE RUN AND THE READING ARE SEPARATE STEPS. A binary that dies on a
    # library it cannot find writes to stderr and leaves stdout EMPTY, and an
    # empty answer parses as "no linked line", which is the one result this
    # check treats as success. Reading the two together would therefore report a
    # binary that never started as the case that passes.
    BACKEND_REPORT="$(build_tree_run rocm "$ROCM_REL/ppf-contact-solver" --backend)" || die \
        "${ROCM_REL#"$SRC_DIR"/}/ppf-contact-solver --backend failed" \
        "A binary that will not run here cannot be packaged."
    LINKED="$(printf '%s\n' "$BACKEND_REPORT" | sed -n 's/^linked: //p')"
    [ -z "$LINKED" ] || die \
        "${ROCM_REL#"$SRC_DIR"/}/ppf-contact-solver links the $LINKED backend library" \
        "A distribution's ROCm backend is built for AMD (HIP_PLATFORM=amd). A library" \
        "built for another platform loads and reports rocm on hardware it is not for."
    printf '  [OK] the backend is built for the AMD platform\n'

    python3 "$SRC_DIR/.github/workflows/scripts/check-rocm-fp-flags.py" || die \
        "a ROCm compile selects a fast-math device library (see above)"
    python3 "$SRC_DIR/.github/workflows/scripts/check-rocm-code-objects.py" \
        --library "$BACKEND_PATH" \
        --arch-file "$SRC_DIR/crates/ppf-cts-compute/rocm/rocm_arch.txt" \
        --rocm-path "$ROCM_ROOT" || die \
        "the ROCm library's device image does not pass the code-object check (see above)"
    printf '  [OK] the device image carries every target in rocm_arch.txt, with no FP64\n'
fi

# ASK EACH ARTIFACT WHAT IT IS RATHER THAN INFERRING IT FROM ITS PATH.
# build_tree_run supplies the one thing the build tree cannot: a ROCm binary
# names the HIP runtime directly and reaches it only through the SDK until
# bundle.sh stages it into the payload. Every other backend is run with no
# LD_LIBRARY_PATH, since nothing a user has would supply one.
for expected in "${BACKEND_DIRS[@]}"; do
    dir="${expected%:*}"
    want="${expected##*:}"
    got="$(build_tree_run "$want" "$dir/ppf-contact-solver" --backend)" || die \
        "$dir/ppf-contact-solver --backend failed" \
        "A binary that will not run here cannot be packaged."
    [ "$got" = "$want" ] || die \
        "$dir holds the $got backend and this build expects $want" \
        "Each backend owns a target directory and they must not be mixed." \
        "Remove ${dir#"$SRC_DIR"/} and re-run build.sh."
    printf '  [OK] %s is the %s backend\n' "${dir#"$SRC_DIR"/}" "$got"
done

# ---------------------------------------------------------------------------
step "[4/4] Writing the launcher"
# ---------------------------------------------------------------------------

if [ -z "${PPF_CTS_VENV:-}" ]; then
    PPF_CTS_VENV="$(python3 - "$SRC_DIR/warmup.py" <<'PY'
import runpy
import sys

ns = runpy.run_path(sys.argv[1])
print(ns["get_venv_path"]())
PY
)"
fi
[ -n "$PPF_CTS_VENV" ] || die "could not determine a venv path"
if [ ! -x "$PPF_CTS_VENV/bin/python" ]; then
    printf 'WARNING: no interpreter at %s/bin/python.\n' "$PPF_CTS_VENV" >&2
    printf '         The build is fine, but start.sh and every example need the\n' >&2
    printf '         frontend dependencies. Run warmup.sh.\n' >&2
fi

# NO TARGET DIRECTORY IS PINNED FOR THE DEVELOPER LAUNCHER. Every backend builds
# into its own directory, so there is no `target/release` to default to, and
# `frontend/__init__.py` searches the named ones and resolves which backend a run
# uses (`frontend.get_backend`): the one GPU build present, or the first of them
# whose solver reports a usable device. A caller who wants one names it, with
# CARGO_TARGET_DIR or `frontend.set_backend`, and that choice is honored.
START="$BUILD_LINUX/start.sh"
cat > "$START" <<EOF
#!/usr/bin/env bash
# Generated by build-linux-native/build.sh. Edits are lost on the next build;
# change build.sh instead. Port and environment come from config.sh.
#
# This is the DEVELOPER entry point: it serves the source tree's examples from
# the developer environment. The distribution bundle.sh produces has its own
# launcher.
set -euo pipefail

BUILD_LINUX="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
SRC="\$(cd "\$BUILD_LINUX/.." && pwd)"

# shellcheck source=/dev/null
. "\$BUILD_LINUX/config.sh"

PPF_CTS_VENV="\${PPF_CTS_VENV:-$PPF_CTS_VENV}"
VENV_PY="\$PPF_CTS_VENV/bin/python"
if [ ! -x "\$VENV_PY" ]; then
    printf 'ERROR: no interpreter at %s. Run warmup.sh.\n' "\$VENV_PY" >&2
    exit 1
fi

# frontend/__init__.py resolves the tree root from its own __file__ and loads
# this tree's lib_ppf_cts_py.so by absolute path, so PYTHONPATH is all it needs
# to find this tree's build.
export PYTHONPATH="\$SRC\${PYTHONPATH:+:\$PYTHONPATH}"

# Keep Jupyter's state inside this directory rather than in the user profile.
export JUPYTER_CONFIG_DIR="\$BUILD_LINUX/jupyter/config"
export JUPYTER_DATA_DIR="\$BUILD_LINUX/jupyter/data"
export IPYTHONDIR="\$BUILD_LINUX/jupyter/ipython"

exec "\$VENV_PY" -m jupyterlab --no-browser --port="\$PORT" \\
    --ServerApp.token="" --notebook-dir="\$SRC/examples"
EOF
chmod +x "$START"
bash -n "$START" || die "the generated start.sh does not parse"
printf '  [OK] %s\n' "$START"

# ---------------------------------------------------------------------------
step "BUILD COMPLETE"
# ---------------------------------------------------------------------------
printf 'JupyterLab:   %s   (port %s, from config.sh)\n' "$START" "$PORT"
for entry in "${BACKEND_DIRS[@]}"; do
    printf '%-6s solver: %s\n' "${entry##*:}" "${entry%:*}/ppf-contact-solver"
done
printf '\nTo package a distribution, run %s/bundle.sh\n' "$BUILD_LINUX"
