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

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

VENV="$HOME/.local/share/ppf-cts/venv"

CARGO_FLAGS=(--release --workspace)
if [ "$(uname -s)" = "Darwin" ]; then
    # No nvcc on macOS; emulated mode skips the C++/CUDA build path.
    CARGO_FLAGS+=(
        --no-default-features
        --features ppf-cts-solver/emulated,ppf-cts-server/emulated
    )
    echo "==> macOS detected: building with --features emulated"
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

echo "==> maturin develop --release (ppf-cts-py)"
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
    echo "warn: venv not found at $VENV; using whatever maturin is on PATH"
fi
(cd "$REPO_ROOT/crates/ppf-cts-py" && maturin develop --release)

echo "==> done"
echo "    target/release/ppf-contact-solver"
echo "    target/release/ppf-cts-server"
echo "    _ppf_cts_py installed in the venv"
