#!/usr/bin/env bash
# Build everything: all Rust binaries (the CUDA solver driver from
# crates/ppf-cts-solver and the Rust solver host from
# crates/ppf-cts-server) plus the PyO3 Python wheel.
#
# Run from anywhere; the script cd's to the repo root.
#
# macOS: nvcc isn't available, so the script automatically enables the
# `emulated` feature on both binary crates. This skips the C++/CUDA
# compile in build.rs and stubs the extern "C" kernel calls with
# Rust-side kinematics. The resulting binaries are suitable for the
# debug test rig but not for production simulations.
#
# Linux + Windows (when nvcc is on PATH or the bundled CUDA toolkit
# from build-win-native/ is set up): full CUDA build.
#
# Requirements:
#   - cargo on PATH (or the embedded toolchain on Windows; that path
#     goes through build-win-native/build.bat instead, see its header).
#   - A Python venv at $HOME/.local/share/ppf-cts/venv with `maturin`
#     installed. If absent, the script falls back to whatever maturin
#     is on PATH and warns.
#
# PyO3 install location: the maturin output (the `_ppf_cts_py.so` plus
# its dist-info) lands in `<tree-root>/.tree-pyo3/`, NOT in the shared
# venv's site-packages. The shared venv stays branch-independent, so
# multiple checkouts of this repo on a single host don't clobber each
# other's PyO3 module through the shared venv. `frontend/__init__.py`
# prepends `<tree-root>/.tree-pyo3/` to `sys.path` at import time, so
# Python finds the right checkout's PyO3 module automatically. Windows
# is already tree-local (each tree has its own embedded Python under
# `build-win-native/python/`); this script doesn't apply there.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

VENV="$HOME/.local/share/ppf-cts/venv"

CARGO_FLAGS=(--release --workspace)
if [ "$(uname -s)" = "Darwin" ]; then
    # No nvcc on macOS; emulated mode skips the C++/CUDA build path.
    # Also exclude ppf-cts-py from the workspace cargo step: PyO3 has a
    # long-running symbol-resolution mismatch with arm64 Python that
    # makes `cargo build -p ppf-cts-py` fail at link
    # ("Undefined symbol: __Py_TrueStruct"). The maturin block below
    # attempts the same compile via its own toolchain and falls back
    # gracefully when it fails too (see feedback_build_all_macos_pyo3).
    CARGO_FLAGS+=(
        --exclude ppf-cts-py
        --no-default-features
        --features ppf-cts-solver/emulated,ppf-cts-server/emulated
    )
    echo "==> macOS detected: building with --features emulated (ppf-cts-py excluded)"
fi

# --- Content-hash mtime healer ------------------------------------------------
#
# Cargo decides whether to recompile a unit by comparing source-file mtimes
# against the build artifact's mtime. `rsync -a` (and other sync tools that
# preserve timestamps) drop a file onto the remote with the source machine's
# mtime, which can be older than the artifact already on the remote. Cargo
# then reports "Finished release in 0.07s" without compiling, the wheel
# packages a stale `.so`, and a fixed bug in the source isn't actually in
# the binary that ships.
#
# This pass closes that hole. For every Rust source file in the workspace
# we hash its content (sha256, since coreutils ships it everywhere; the
# task asked for "md5-style", and any cryptographic content hash works
# the same here) and compare against the hash recorded on the last
# successful build. If the hash changed -- or there is no recorded hash
# yet -- we `touch` the file so cargo treats it as freshly modified.
# Files that haven't actually changed keep their original mtime, so cargo
# correctly skips them.
#
# Net effect: cargo recompiles based on *content* changes, not mtime
# changes. Rsync-with-mtime-preserved no longer hides a real edit; an
# rsync that drops in identical content doesn't trigger a wasteful
# rebuild either.
HASH_DB="$REPO_ROOT/target/.build-all-hashes"
NEW_DB="$HASH_DB.new"
mkdir -p "$(dirname "$HASH_DB")"
: > "$NEW_DB"

if command -v sha256sum >/dev/null 2>&1; then
    HASHER=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
    # macOS ships shasum but not sha256sum; the -a 256 flag matches output.
    HASHER=(shasum -a 256)
else
    echo "warn: no sha256sum/shasum on PATH; skipping content-hash mtime fix"
    HASHER=()
fi

if [ "${#HASHER[@]}" -gt 0 ]; then
    touched=0
    # NUL-delimited so paths with spaces don't break the loop. Restrict
    # to .rs because that's all rustc reads as source; build.rs scripts
    # and Cargo.toml have their own fingerprint paths handled by cargo.
    while IFS= read -r -d '' f; do
        # Skip target/ and any nested target dirs (workspace + per-crate).
        case "$f" in
            */target/*) continue ;;
        esac
        # Compute hash; bail on read errors.
        line=$("${HASHER[@]}" "$f" 2>/dev/null) || continue
        h=${line%% *}
        printf '%s  %s\n' "$h" "$f" >> "$NEW_DB"
        # Look up previous hash for this exact path.
        if [ -f "$HASH_DB" ]; then
            prev=$(awk -v p="$f" '{n=index($0, "  "); if (substr($0, n+2) == p) { print substr($0, 1, n-1); exit }}' "$HASH_DB")
        else
            prev=""
        fi
        if [ "$prev" != "$h" ]; then
            touch "$f"
            touched=$((touched + 1))
        fi
    done < <(find "$REPO_ROOT" -type f -name '*.rs' -not -path '*/target/*' -print0)
    if [ "$touched" -gt 0 ]; then
        echo "==> content-hash check: $touched file(s) changed; mtimes bumped"
    fi
    # Promote the new DB only after the touch pass succeeds; if the build
    # itself fails we still want the DB to reflect the actual on-disk
    # state for the next run.
    mv "$NEW_DB" "$HASH_DB"
else
    rm -f "$NEW_DB"
fi
# --- end content-hash mtime healer --------------------------------------------

echo "==> cargo build ${CARGO_FLAGS[*]}"
cargo build "${CARGO_FLAGS[@]}"

TREE_PYO3="$REPO_ROOT/.tree-pyo3"

echo "==> maturin build --release (ppf-cts-py) → $TREE_PYO3"
if [ -f "$VENV/bin/activate" ]; then
    # Some venv activate scripts touch unset variables; relax `-u` while sourcing.
    set +u
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    set -u
    # Self-heal: older venvs predate maturin / cbor2 / psutil being
    # in `python_packages()`. Running `warmup.py` again is heavyweight
    # (re-installs the whole toolchain). We just pip-install the three
    # so the build proceeds. Safe to no-op when already present
    # (`pip install` is idempotent).
    missing=()
    for pkg in maturin cbor2 psutil; do
        if ! python -c "import importlib; importlib.import_module('$pkg')" \
                >/dev/null 2>&1; then
            missing+=("$pkg")
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "==> bootstrapping missing venv packages: ${missing[*]}"
        pip install --quiet "${missing[@]}"
    fi
else
    echo "warn: venv not found at $VENV; using whatever maturin / pip is on PATH"
fi

# Defense in depth: a `_ppf_cts_py` left over in the shared venv from
# an older `maturin develop` would still satisfy `import _ppf_cts_py`
# from any tree where `.tree-pyo3/` was not built yet, and would
# silently shadow this tree's build until a fresh maturin runs. We
# strip it once on every build-all.sh run; `pip uninstall -y` is a
# no-op when the package isn't installed.
if [ -x "$VENV/bin/pip" ]; then
    if "$VENV/bin/pip" show ppf-cts-py >/dev/null 2>&1; then
        echo "==> scrubbing stale ppf-cts-py from shared venv"
        "$VENV/bin/pip" uninstall -y ppf-cts-py >/dev/null
    fi
fi

# Build the wheel via maturin into the workspace's target/wheels/, then
# install it into THIS tree's .tree-pyo3/. `--force-reinstall --no-deps`
# guarantees an in-place overwrite even when the dist-info already
# exists from a previous run.
#
# macOS link gotcha: pyo3 vs the system / venv Python on arm64 has a
# long-running symbol-resolution mismatch (see feedback_build_all_macos_pyo3).
# `maturin build` fails the same way `maturin develop` did, with
# "Undefined symbol: __Py_TrueStruct" type errors. We attempt the
# build, fall back to a warning on failure, and leave .tree-pyo3/
# empty in that case -- frontend/__init__.py's sys.path hook then
# falls through to whatever _ppf_cts_py is already in the active
# Python's site-packages (typically a previous successful build, or
# a wheel built remotely on Linux and rsync'd in).
set +e
(cd "$REPO_ROOT/crates/ppf-cts-py" && maturin build --release)
MATURIN_RC=$?
set -e
if [ "$MATURIN_RC" -ne 0 ]; then
    echo "warn: maturin build failed (rc=$MATURIN_RC). $TREE_PYO3 left untouched;"
    echo "      frontend/__init__.py falls back to the currently-installed _ppf_cts_py."
else
    WHEEL="$(ls -1t "$REPO_ROOT/target/wheels/"ppf_cts_py-*.whl 2>/dev/null | head -n 1)"
    if [ -z "$WHEEL" ]; then
        echo "warn: maturin reported success but produced no wheel under $REPO_ROOT/target/wheels/"
    else
        mkdir -p "$TREE_PYO3"
        pip install --quiet --target "$TREE_PYO3" --force-reinstall --no-deps "$WHEEL"
        echo "==> _ppf_cts_py installed in $TREE_PYO3"
    fi
fi

echo "==> done"
echo "    target/release/ppf-contact-solver"
echo "    target/release/ppf-cts-server"
