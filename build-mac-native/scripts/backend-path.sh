#!/usr/bin/env bash
# File: build-mac-native/scripts/backend-path.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Resolves the Metal backend a Mach-O binary will actually load. Source it, do
# not execute it:
#
#     . "$BUILD_MAC/scripts/backend-path.sh"
#     backend=$(resolve_backend "target/release/ppf-contact-solver") || exit 1
#
# It prints the resolved path on stdout and a diagnosis on stderr when it
# cannot resolve one, so a caller reports the failure with its own `die`.
#
# WHY THIS IS NOT `otool -L | awk` PLUS A FILE TEST. `otool -L` prints each
# dependency's INSTALL NAME, and the backend's install name is
# `@rpath/libppfbe_metal.dylib`. `@rpath` is a placeholder the dynamic
# loader expands against the binary's own LC_RPATH entries; it is not a
# directory. Testing that string with -f therefore asks whether a file literally
# named `@rpath/libppfbe_metal.dylib` exists, which is always false, and
# copying from it is equally wrong. The resolution has to consult LC_RPATH.
#
# Reading LC_RPATH is also what makes the answer the RIGHT one rather than
# merely a found one. Several build-script output directories can coexist under
# target/release/build, each holding a `libppfbe_metal.dylib` from a
# different build, so a glob there can match a stale library that the loader
# would never choose. Exactly one of those directories is on this binary's
# rpath, and that is the library it will load.

# Print the path to the Metal backend `$1` resolves to, or fail.
resolve_backend() {
    local macho="$1"

    if [ ! -f "$macho" ]; then
        printf 'no such binary: %s\n' "$macho" >&2
        return 1
    fi
    if ! command -v otool >/dev/null 2>&1; then
        printf 'otool not found on PATH\n' >&2
        return 1
    fi

    # awk runs to completion rather than exiting at the first match. An awk that
    # exits early can close the pipe under otool, and with `set -o pipefail` in
    # the caller that turns a successful read into a failed pipeline.
    local install_name
    install_name="$(otool -L "$macho" |
        awk '/libppfbe_metal\.dylib/ { print $1 }')"

    if [ -z "$install_name" ]; then
        printf '%s links no libppfbe_metal.dylib\n' "$macho" >&2
        return 1
    fi

    # An absolute or relative install name needs no expansion. Only `@rpath`
    # does. `@loader_path` and `@executable_path` are deliberately not handled:
    # nothing in this build emits them, and quietly guessing at a form that has
    # not been seen would hide the day one appears.
    case "$install_name" in
    @rpath/*) ;;
    @*)
        printf 'unsupported install name %s on %s\n' "$install_name" "$macho" >&2
        return 1
        ;;
    *)
        if [ ! -f "$install_name" ]; then
            printf 'the binary records a backend at %s, and no file is there\n' \
                "$install_name" >&2
            return 1
        fi
        printf '%s\n' "$install_name"
        return 0
        ;;
    esac

    local leaf="${install_name#@rpath/}"
    local rpath_dir
    while IFS= read -r rpath_dir; do
        [ -n "$rpath_dir" ] || continue
        if [ -f "$rpath_dir/$leaf" ]; then
            printf '%s\n' "$rpath_dir/$leaf"
            return 0
        fi
    done < <(otool -l "$macho" |
        awk '/LC_RPATH/ { in_rpath = 1; next }
             in_rpath && $1 == "path" { print $2; in_rpath = 0 }')

    printf 'the binary records a backend at %s, and no LC_RPATH entry on %s carries %s\n' \
        "$install_name" "$macho" "$leaf" >&2
    return 1
}
