#!/usr/bin/env bash
# File: build-mac-native/build.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Builds the solver on macOS, the counterpart of build-win-native/build.bat.
# Run warmup.sh once first.
#
# It is safe to run twice: cargo decides what to recompile, and everything
# else here is a check or an overwrite.
#
# THE BUILD IS ONE COMMAND, AND THE FLAG LIST IS EMPTY ON PURPOSE.
# `cargo build --release` with no features selects the real backend for the
# host, which on macOS is Metal (crates/ppf-cts-solver/build.rs). Everything
# this script does around that one command is verifying the host up front,
# pre-compiling the shader libraries, and verifying the artifacts after.
#
# THE SHADER STEP IS OPTIONAL BY DESIGN AND IS NOT A PREREQUISITE.
# The backend can always assemble its shader and compile it at run time through
# newLibraryWithSource, which costs about 6.5 s on a machine whose OS shader
# cache has been evicted. Step 4 removes that by compiling the same text ahead
# of time into `.metallib` artifacts the run loads in about 4 ms instead. It
# needs the Metal Toolchain, which is a separate downloadable component
# (`xcodebuild -downloadComponent MetalToolchain`) that full Xcode does not
# supply, so the step TESTS for it by invoking `xcrun metal --version` and skips
# with one line when it is absent. A machine without it builds and runs exactly
# as before. PPF_MAC_METALLIB=0 skips the step outright.

set -euo pipefail

BUILD_MAC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$BUILD_MAC/.." && pwd)"
LOGFILE="$BUILD_MAC/build.log"

# Re-run under tee so the build is logged. The status has to come out of
# PIPESTATUS, because a pipeline's status is its last command's and tee
# succeeds whatever cargo did.
if [ -z "${PPF_MAC_BUILD_LOGGING:-}" ]; then
    printf 'Logging to %s\n' "$LOGFILE"
    set +e
    PPF_MAC_BUILD_LOGGING=1 "${BASH_SOURCE[0]}" "$@" 2>&1 | tee "$LOGFILE"
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
        "The backend is chosen by the host, not by a flag. To force one on a" \
        "machine that could serve either, run cargo yourself:" \
        "  cargo build --release --features metal"
fi

printf '============================================================\n'
printf "  ZOZO's Contact Solver, native macOS build\n"
printf '============================================================\n'
printf 'Build directory:  %s\n' "$BUILD_MAC"
printf 'Source directory: %s\n' "$SRC_DIR"

# ---------------------------------------------------------------------------
step "[1/5] Checking the host"
# ---------------------------------------------------------------------------

CLT_HINT="Install the Xcode command line tools:  xcode-select --install"

[ "$(uname -s)" = "Darwin" ] || die "this builds on macOS, and uname reports $(uname -s)"
[ "$(uname -m)" = "arm64" ] || die \
    "this build targets Apple silicon and uname -m reports $(uname -m)"

command -v xcrun >/dev/null 2>&1 || die "xcrun not found on PATH" "$CLT_HINT"
# The backend Makefile shells out to python3 to embed the shared device
# sources into the dylib, so a build without one fails inside make.
command -v python3 >/dev/null 2>&1 || die \
    "python3 not found on PATH" \
    "crates/ppf-cts-compute/metal/Makefile runs embed_source.py with it." \
    "$CLT_HINT"
SDK_PATH="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
[ -n "$SDK_PATH" ] || die "xcrun could not report a macOS SDK path" "$CLT_HINT"
[ -d "$SDK_PATH/System/Library/Frameworks/Metal.framework" ] || die \
    "the macOS SDK at $SDK_PATH carries no Metal.framework" "$CLT_HINT"

# shellcheck source=build-mac-native/config.sh
. "$BUILD_MAC/config.sh"

RUST_DIR="$BUILD_MAC/rust"
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
printf 'cargo: %s (%s)\n' "$(command -v cargo)" "$(cargo --version)"
printf 'SDK:   %s\n' "$SDK_PATH"

# ---------------------------------------------------------------------------
step "[2/5] Building with cargo"
# ---------------------------------------------------------------------------

cd "$SRC_DIR"
cargo build --release || die \
    "cargo build --release failed" \
    "If it failed inside crates/ppf-cts-compute/metal, the make output" \
    "is above: that Makefile is driven by build.rs and compiles the backend."

# The CPU backend, into its OWN target directory.
#
# TWO BACKENDS CANNOT SHARE ONE DIRECTORY, and `crates/ppf-cts-solver/build.rs`
# refuses rather than letting them try: every backend links the same executable
# name, so `--features cpu` on the directory above would replace the Metal
# solver in place with one that is roughly 30x slower, leaving nothing to say
# it had changed. `CARGO_TARGET_DIR` is the variable that file names in its own
# refusal, and it is what the frontend reads too, so the cdylib lands beside
# the binary it belongs to.
#
# WHY THE DISTRIBUTION CARRIES IT. The add-on offers a GPU/CPU choice, and a
# device is available exactly when a build for it exists under the selected
# root, so without this the choice can never be satisfied by a download: the
# artist is told to run a cargo command, which is the one thing someone running
# an unzipped release cannot do. It is also the only backend a parity or
# determinism run can be compared against on a machine that has no second GPU.
#
# It is EXPORTED for one command rather than set for the file, so nothing below
# this line can pick it up: every later step reads target/release, and a leaked
# value would silently point all of them at the CPU build.
step "[2b/5] Building the CPU backend"
CPU_TARGET="$SRC_DIR/target/cpu"
CARGO_TARGET_DIR="$CPU_TARGET" cargo build --release --features cpu || die \
    "cargo build --release --features cpu failed" \
    "This is the portable Rust backend the distribution ships beside Metal." \
    "It compiles the same neutral kernel bodies through the host C++ compiler," \
    "so a failure here is a compiler or toolchain question rather than a Metal" \
    "one; the Metal build above already succeeded."

# ---------------------------------------------------------------------------
step "[3/5] Verifying the artifacts"
# ---------------------------------------------------------------------------

TARGET_REL="$SRC_DIR/target/release"
SOLVER_BIN="$TARGET_REL/ppf-contact-solver"
SERVER_BIN="$TARGET_REL/ppf-cts-server"
PY_DYLIB="$TARGET_REL/lib_ppf_cts_py.dylib"

for artifact in "$SOLVER_BIN" "$SERVER_BIN" "$PY_DYLIB"; do
    [ -f "$artifact" ] || die \
        "cargo reported success but $artifact is missing" \
        "All three workspace default-members land in target/release; a missing" \
        "one means the build did not produce what the frontend and the addon load."
    printf '  [OK] %s\n' "${artifact#"$SRC_DIR"/}"
done

# The same three from the CPU build. All three are needed, not just the solver:
# the add-on spawns that directory's OWN ppf-cts-server, and the build worker
# it starts loads that directory's OWN cdylib, which is what makes the session
# script name the CPU solver rather than the Metal one.
CPU_REL="$CPU_TARGET/release"
for artifact in \
    "$CPU_REL/ppf-contact-solver" \
    "$CPU_REL/ppf-cts-server" \
    "$CPU_REL/lib_ppf_cts_py.dylib"; do
    [ -f "$artifact" ] || die \
        "cargo reported success but $artifact is missing" \
        "The CPU build writes the same three default-members into its own" \
        "target directory. A missing one means --features cpu did not produce" \
        "what the add-on's CPU device and the frontend load."
    printf '  [OK] %s\n' "${artifact#"$SRC_DIR"/}"
done

# ASK THE ARTIFACT WHAT IT IS RATHER THAN INFERRING IT FROM ITS PATH. Every
# backend links this same executable name, so `target/cpu/release` is a
# convention and this answer is evidence. A directory that held the wrong
# backend would ship a distribution whose CPU device runs Metal, or the
# reverse, with the paths saying otherwise.
for expected_dir_backend in "$TARGET_REL:metal" "$CPU_REL:cpu"; do
    dir="${expected_dir_backend%:*}"
    want="${expected_dir_backend##*:}"
    got="$("$dir/ppf-contact-solver" --backend)" || die \
        "$dir/ppf-contact-solver --backend failed" \
        "A binary that will not run here cannot be packaged."
    [ "$got" = "$want" ] || die \
        "$dir holds the $got backend and this build expects $want" \
        "The two builds each own a target directory and must not be mixed." \
        "Remove target and target/cpu and re-run build.sh."
    printf '  [OK] %s is the %s backend\n' "${dir#"$SRC_DIR"/}" "$got"
done

# The backend is a dylib the solver binary links dynamically, and its path is
# recorded in the binary itself. Read it off the binary rather than globbing
# target/release/build, where several stale build-script output directories
# can coexist and only one of them is the one that is loaded.
command -v otool >/dev/null 2>&1 || die "otool not found on PATH" "$CLT_HINT"
. "$BUILD_MAC/scripts/backend-path.sh"
BACKEND_PATH="$(resolve_backend "$SOLVER_BIN")" || die \
    "could not resolve the Metal backend $SOLVER_BIN loads (see above)" \
    "On macOS build.rs links the Metal backend dynamically. A binary without" \
    "that load command was built against a different backend, so check what" \
    "build.rs selected in the cargo output above. If the load command is there" \
    "but the file is not, re-run build.sh; a cargo clean removes that directory."
printf '  [OK] backend %s\n' "$BACKEND_PATH"

# ---------------------------------------------------------------------------
step "[4/5] Pre-compiling the shader libraries"
# ---------------------------------------------------------------------------

# WHAT THIS PRODUCES AND WHY IT IS SAFE TO SKIP.
#
# The backend assembles its MSL and hands it to newLibraryWithSource. That
# compile measured about 6.5 s on an Apple M1 for this backend's four libraries,
# and macOS caches the result itself, so it is paid only when that cache is cold
# or has been evicted. Compiling the same text offline into a .metallib and
# loading it with newLibraryWithURL took 0.5 ms for all four on the same
# machine, and does not depend on a cache the system may evict at any time.
#
# It is a LOOKUP in front of the runtime compile, never a replacement, so
# everything below may fail or be absent without costing this machine the
# ability to build or run. What must never happen is a STALE artifact loading:
# a library that no longer matches the kernels is a wrong answer rather than a
# slow start. The artifact is therefore NAMED by a hash over the assembled
# source, the math mode read back out of the compile options and the language
# version, so an edited kernel names a file that is not there and the run
# compiles from source. This script never computes that name: it asks the
# backend, through the dump tool, which is what makes the two sides agree.
#
# The artifacts land beside the dylib the solver resolves, under target/. They
# are build outputs and must not be written under crates/ppf-cts-solver/src,
# where two build scripts watch recursively and cargo reads a directory in
# rerun-if-changed as "rerun if any descendant changes".

METALLIB_SUMMARY="skipped"

metallib_step() {
    if [ "${PPF_MAC_METALLIB:-1}" = "0" ]; then
        printf 'PPF_MAC_METALLIB=0, so no shader libraries are pre-compiled.\n'
        printf 'The backend will assemble and compile its shader at run time.\n'
        METALLIB_SUMMARY="skipped (PPF_MAC_METALLIB=0)"
        return 0
    fi

    # TEST THE COMPILER BY INVOKING IT, NEVER BY LOCATING IT. `xcrun --find
    # metal` resolves to a path whether or not the Metal Toolchain component is
    # installed, so a check that only looks for the binary reports a toolchain
    # that cannot run. `xcrun metal --version` is what settles it.
    local metal_version
    if ! metal_version="$(xcrun metal --version 2>&1)"; then
        printf 'The Metal Toolchain is not installed on this machine, so the\n'
        printf 'shader libraries are not pre-compiled. This is not an error:\n'
        printf 'the backend compiles its shader at run time, which is what it\n'
        printf 'does on any machine that reaches this line.\n\n'
        printf 'To pre-compile them, install the component and re-run:\n'
        printf '    xcodebuild -downloadComponent MetalToolchain\n\n'
        printf 'xcrun metal --version reported:\n%s\n' "$metal_version"
        METALLIB_SUMMARY="skipped (no Metal Toolchain)"
        return 0
    fi
    printf 'Metal Toolchain: %s\n' "$(printf '%s\n' "$metal_version" | head -1)"

    local lib_dir dump_bin
    lib_dir="$(dirname "$BACKEND_PATH")"
    dump_bin="$(dirname "$lib_dir")/bin/ppf-metal-shader-dump"
    if [ ! -x "$dump_bin" ]; then
        # The shader-dump tool is not built by the default cargo/make path today:
        # it calls metal_backend::backend_bringup, which lives in bringup.mm and is
        # deliberately absent from the shipped libppfbe_metal.dylib, so it no longer
        # links against the runtime backend. Restoring the pre-compile means porting
        # the tool to the ABI bring-up objects (crates/ppf-cts-compute/metal/Makefile
        # carries the note). Until then this step is a graceful skip, exactly as it is
        # when the Metal Toolchain is absent: the backend assembles and compiles its
        # shader at run time (about 6.5 s, OS-cached), so nothing here is broken, the
        # first solver run is just slower.
        printf 'The shader-dump tool is not present at\n  %s\n' "$dump_bin"
        printf 'so the shader libraries are not pre-compiled. This is not an error:\n'
        printf 'the backend compiles its shader at run time. The tool needs porting to\n'
        printf 'the ABI backend before the pre-compile can run (see the Makefile note).\n'
        METALLIB_SUMMARY="skipped (shader-dump tool needs porting to the ABI backend)"
        return 0
    fi

    # Removed on the way out of this function, and deliberately NOT on a
    # failure: every message below names a file in here, and the assembled
    # translation unit is the only artifact a compile diagnostic can be read
    # against.
    local dump_dir
    dump_dir="$(mktemp -d "${TMPDIR:-/tmp}/ppf-metallib.XXXXXX")"

    # Run the backend far enough to assemble every library, with no scene. It
    # writes <key>.metal and <key>.flags per library, runs both parity
    # self-tests on the way (so a machine that cannot reproduce this backend's
    # arithmetic cannot ship an artifact), and reports what the caches did.
    printf '\nAssembling the shader libraries (this also runs the parity self-tests)\n'
    local dump_log="$dump_dir/dump.log"
    PPF_METAL_LIBRARY_DUMP_DIR="$dump_dir" "$dump_bin" > "$dump_log" 2>&1 || {
        cat "$dump_log"
        die "the shader dump tool failed" \
            "It brings the backend up exactly as initialize() does, so a" \
            "failure here is a failure this machine would also see on the" \
            "first solver run. The output above names the step."
    }
    sed 's/^/  | /' "$dump_log"

    local sources
    sources="$(find "$dump_dir" -maxdepth 1 -name '*.metal' | sort)"
    [ -n "$sources" ] || die \
        "the dump tool wrote no shader source" \
        "Without one there is nothing to compile, and a step that silently" \
        "produced nothing would report success."

    # Compile each library under the flags the RUNTIME reported. Reading them
    # rather than restating them is what stops this script from compiling under
    # settings the backend would not have used; mathMode is a correctness
    # setting here, not a speed knob.
    printf '\nCompiling\n'
    local built=0 reused=0 key src flags_file flags
    local keys=""
    while IFS= read -r src; do
        [ -n "$src" ] || continue
        key="$(basename "$src" .metal)"
        keys="$keys $key"
        flags_file="$dump_dir/$key.flags"
        [ -f "$flags_file" ] || die \
            "no $key.flags beside $key.metal" \
            "The backend writes the flags file only when the compile options" \
            "it read back map onto offline arguments. Without it the artifact" \
            "cannot be built under settings known to match the runtime."
        flags="$(cat "$flags_file")"
        # A second reading of an invariant the backend already refuses to run
        # without: context_check_math_mode ends the process when the compiler
        # is not in MTLMathModeSafe. Fast math deletes Kahan compensation
        # outright and damages the eigenvalue-floored inverse on 94.5% of
        # threads, so an artifact compiled under it would be plausible and
        # wrong.
        case "$flags" in
        *-fmetal-math-mode=safe*) ;;
        *) die \
            "the backend reported offline flags that are not safe math: $flags" \
            "mathMode Safe is a correctness setting for this solver. Refusing" \
            "to compile an artifact under anything else." ;;
        esac

        if [ -f "$lib_dir/$key.metallib" ]; then
            reused=$((reused + 1))
            printf '  [--] %s.metallib is already built\n' "$key"
            continue
        fi

        # Word splitting on the flags is intended: the file holds an argument
        # list, one line, written by the backend.
        local -a flag_args
        read -r -a flag_args <<< "$flags"
        # The compiler's own output goes to a file, because a successful
        # compile of this shader is not silent: it emits dozens of warnings,
        # most of them the same few repeated per translation unit, and pasting
        # all of them into the build log buries the four lines that say what
        # was produced. Nothing is discarded. A failure prints the log whole,
        # and a success prints each DISTINCT warning once with its count, so a
        # real one is still visible. These are diagnostics about the assembled
        # shader, which `newLibraryWithSource` compiles too and does not show.
        local compile_log="$dump_dir/$key.compile.log"
        if ! xcrun metal "${flag_args[@]}" -c "$src" -o "$dump_dir/$key.air" \
                > "$compile_log" 2>&1; then
            cat "$compile_log"
            die "xcrun metal failed on the assembled shader" \
                "The text it was given is $src, which is byte for byte what" \
                "this backend hands newLibraryWithSource. The diagnostics" \
                "above name real source files and lines, because the host" \
                "injects #line as it splices each segment." \
                "PPF_MAC_METALLIB=0 skips this step if the run itself is what" \
                "you need."
        fi
        if ! xcrun metallib "$dump_dir/$key.air" -o "$dump_dir/$key.metallib" \
                >> "$compile_log" 2>&1; then
            cat "$compile_log"
            die "xcrun metallib failed to link $key.air"
        fi
        mv "$dump_dir/$key.metallib" "$lib_dir/$key.metallib"
        built=$((built + 1))
        printf '  [OK] %s.metallib (%s bytes) from %s\n' \
            "$key" "$(wc -c < "$lib_dir/$key.metallib" | tr -d ' ')" "$flags"
        # `|| true` because a clean compile makes grep exit 1, and under
        # `set -o pipefail` that would end the build with no message at all,
        # which is the quietest possible failure and the one this file is
        # least entitled to.
        local warnings
        warnings="$(grep -o 'warning: .*' "$compile_log" | sort | uniq -c || true)"
        if [ -n "$warnings" ]; then
            printf '%s\n' "$warnings" | sed 's/^ */       /'
        fi
    done <<EOF
$sources
EOF

    # Drop artifacts for keys this build no longer asks for. Each is named by
    # its own content hash, so an edited kernel leaves the previous one behind
    # forever; the solver library alone is about 4 MB.
    local pruned=0 stale leaf
    while IFS= read -r stale; do
        [ -n "$stale" ] || continue
        leaf="$(basename "$stale" .metallib)"
        case " $keys " in
        *" $leaf "*) continue ;;
        esac
        rm -f "$stale"
        pruned=$((pruned + 1))
        printf '  [rm] %s.metallib is no longer asked for\n' "$leaf"
    done <<EOF
$(find "$lib_dir" -maxdepth 1 -name '*.metallib' | sort)
EOF

    # VERIFY BY LOADING, not by having written the files. The name is computed
    # by the backend and the artifact is loaded by the backend, so the only
    # thing that says the two agree is a run that reports every library
    # pre-compiled.
    printf '\nVerifying the artifacts load\n'
    local verify_log="$dump_dir/verify.log"
    "$dump_bin" > "$verify_log" 2>&1 || {
        cat "$verify_log"
        die "the shader dump tool failed on the verification run"
    }
    sed 's/^/  | /' "$verify_log"
    local counts total prebuilt
    counts="$(awk '/^\[shader-dump\] libraries /{print $3, $5}' "$verify_log")"
    total="${counts% *}"
    prebuilt="${counts#* }"
    if [ -z "$counts" ] || [ "$total" != "$prebuilt" ]; then
        # Leave nothing half-installed: an artifact the backend will not load
        # is dead weight, and removing it puts the tree back in the state a
        # run-time compile expects.
        while IFS= read -r stale; do
            [ -n "$stale" ] || continue
            rm -f "$stale"
        done <<EOF
$(find "$lib_dir" -maxdepth 1 -name '*.metallib' | sort)
EOF
        die \
            "the pre-compiled libraries did not load (${prebuilt:-0} of ${total:-?})" \
            "The artifacts were compiled from the text this backend assembles," \
            "under the flags it reported, and installed beside the dylib it is" \
            "loaded from, so a miss here means one of those three no longer" \
            "holds. They have been removed, so the build tree is back to" \
            "compiling the shader at run time." \
            "PPF_MAC_METALLIB=0 skips this step."
    fi
    printf '  [OK] %s of %s libraries loaded pre-compiled\n' "$prebuilt" "$total"
    METALLIB_SUMMARY="$prebuilt pre-compiled ($built built, $reused reused, $pruned pruned)"
    rm -rf "$dump_dir"
}

metallib_step

# ---------------------------------------------------------------------------
step "[5/5] Writing the launcher"
# ---------------------------------------------------------------------------

# start.sh needs an interpreter carrying the frontend dependencies. Resolve it
# the same way warmup.sh did, asking warmup.py for the canonical location
# rather than writing that path down a second time, and bake the answer into
# the generated launcher. The launcher still honors an override at run time.
if [ -z "${PPF_CTS_VENV:-}" ]; then
    command -v python3 >/dev/null 2>&1 || die \
        "python3 not found, so the venv location cannot be resolved" "$CLT_HINT"
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
    printf '         The build is fine, but start.sh and every example need\n' >&2
    printf '         the frontend dependencies. Run warmup.sh.\n' >&2
fi

START="$BUILD_MAC/start.sh"
cat > "$START" <<EOF
#!/usr/bin/env bash
# Generated by build-mac-native/build.sh. Edits are lost on the next build;
# change build.sh instead. Port and environment come from config.sh.
set -euo pipefail

BUILD_MAC="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
SRC="\$(cd "\$BUILD_MAC/.." && pwd)"

# shellcheck source=/dev/null
. "\$BUILD_MAC/config.sh"

PPF_CTS_VENV="\${PPF_CTS_VENV:-$PPF_CTS_VENV}"
VENV_PY="\$PPF_CTS_VENV/bin/python"
if [ ! -x "\$VENV_PY" ]; then
    printf 'ERROR: no interpreter at %s. Run warmup.sh.\n' "\$VENV_PY" >&2
    exit 1
fi

# frontend/__init__.py resolves the tree root from its own __file__ and loads
# \$SRC/target/release/lib_ppf_cts_py.dylib by absolute path, so PYTHONPATH is
# all it needs to find this tree's build.
export PYTHONPATH="\$SRC\${PYTHONPATH:+:\$PYTHONPATH}"

# Keep Jupyter's state inside this directory rather than in the user profile.
export JUPYTER_CONFIG_DIR="\$BUILD_MAC/jupyter/config"
export JUPYTER_DATA_DIR="\$BUILD_MAC/jupyter/data"
export IPYTHONDIR="\$BUILD_MAC/jupyter/ipython"

exec "\$VENV_PY" -m jupyterlab --no-browser --port="\$PORT" \\
    --ServerApp.token="" --notebook-dir="\$SRC/examples"
EOF
chmod +x "$START"
printf '  [OK] %s\n' "$START"

# ---------------------------------------------------------------------------
step "BUILD COMPLETE"
# ---------------------------------------------------------------------------
printf 'JupyterLab:   %s   (port %s, from config.sh)\n' "$START" "$PORT"
printf 'Solver:       %s\n' "$SOLVER_BIN"
printf 'Server:       %s\n' "$SERVER_BIN"
printf 'Backend:      %s\n' "$BACKEND_PATH"
printf 'Shader:       %s\n' "$METALLIB_SUMMARY"
printf '\nTo package a distributable, run %s/bundle.sh\n' "$BUILD_MAC"
