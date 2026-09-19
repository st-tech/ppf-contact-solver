#!/usr/bin/env bash
# File: build-linux-native/scripts/backend-path.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Resolves a shared library an ELF binary will load through its own RPATH or
# RUNPATH. Source it, do not execute it:
#
#     . "$BUILD_LINUX/scripts/backend-path.sh"
#     backend=$(resolve_needed "target/release/ppf-contact-solver" libppfbe_cuda.so) || exit 1
#
# It prints the resolved path on stdout and a diagnosis on stderr when it cannot
# resolve one, so a caller reports the failure with its own `die`.
#
# WHY THE SEARCH PATH AND NOT A GLOB. Several build-script output directories
# coexist under target/release/build, each holding a `libppfbe_cuda.so` from a
# different configuration, so a glob there can match a stale library the loader
# would never choose. Exactly one of those directories is on the binary's search
# path, and that is the library it loads. Measured on a CUDA host: an orphan
# library two weeks old sat beside one rebuilt minutes before.
#
# WHY NOT ldd. ldd runs the loader against this host's cache and environment,
# so it answers where the library resolves HERE, including through
# LD_LIBRARY_PATH, which is a different question from what the binary itself
# records.

# Print the directories an ELF file's RPATH and RUNPATH entries name, one per
# line, with $ORIGIN expanded to the directory holding the file.
elf_search_dirs() {
    local elf="$1" origin entry
    origin="$(cd "$(dirname "$elf")" && pwd)"
    readelf -d "$elf" 2>/dev/null |
        awk '/\((RPATH|RUNPATH)\)/ { sub(/.*\[/, ""); sub(/\].*/, ""); print }' |
        tr ':' '\n' |
        while IFS= read -r entry; do
            [ -n "$entry" ] || continue
            entry="${entry//\$ORIGIN/$origin}"
            entry="${entry//\$\{ORIGIN\}/$origin}"
            printf '%s\n' "$entry"
        done
}

# Print the NEEDED sonames of an ELF file, one per line.
elf_needed() {
    readelf -d "$1" 2>/dev/null |
        awk '/\(NEEDED\)/ { sub(/.*\[/, ""); sub(/\].*/, ""); print }'
}

# Print the path to library `$2` that ELF file `$1` resolves through its own
# search path, or fail.
resolve_needed() {
    local elf="$1" lib="$2" dir

    if [ ! -f "$elf" ]; then
        printf 'no such file: %s\n' "$elf" >&2
        return 1
    fi
    if ! command -v readelf >/dev/null 2>&1; then
        printf 'readelf not found on PATH (it ships with binutils)\n' >&2
        return 1
    fi
    if ! elf_needed "$elf" | grep -qxF -- "$lib"; then
        printf '%s does not list %s as NEEDED\n' "$elf" "$lib" >&2
        return 1
    fi
    while IFS= read -r dir; do
        if [ -f "$dir/$lib" ]; then
            printf '%s\n' "$dir/$lib"
            return 0
        fi
    done < <(elf_search_dirs "$elf")
    printf '%s needs %s, and no RPATH or RUNPATH entry on it holds that file\n' \
        "$elf" "$lib" >&2
    return 1
}

# Print every library ELF file $1 loads TRANSITIVELY that is not supplied by the
# machine, one absolute path per line, resolving each through the search path of
# the object that names it. $2... are the sonames the machine supplies.
#
# WHY A WALK RATHER THAN A LIST. The CUDA backend links its runtime STATICALLY,
# so its closure is empty and a list would have been three names written once
# and never revisited. ROCm's runtime is dynamic, so the set is a property of the
# SDK the library was built against and moves with it: a version that splits a
# library, or adds one, changes the answer, and a hardcoded list would ship a
# distribution missing a file with nothing failing until a user ran it.
#
# It is deliberately STRICT about an unresolvable name: a library the walk cannot
# find is one that would not travel, which is the whole failure this exists to
# prevent, so it reports the name and the object that wanted it and fails. The
# caller decides whether that is a system library it forgot to declare.
#
# This walk DECIDES WHAT TO COPY. elf-audit.py's gate B then checks the result
# independently, over the payload, against its own allowlist. Two readings of the
# same property, which is the point: this one can be wrong about a name and the
# gate is what refuses to ship it.
runtime_closure() {
    local root_elf="$1"
    shift
    local -A seen=() emitted=()
    local -a queue=("$root_elf")
    local elf soname dir resolved is_system candidate

    while [ ${#queue[@]} -gt 0 ]; do
        elf="${queue[0]}"
        queue=("${queue[@]:1}")
        [ -n "${seen[$elf]:-}" ] && continue
        seen[$elf]=1
        while IFS= read -r soname; do
            [ -n "$soname" ] || continue
            is_system=0
            for candidate in "$@"; do
                if [ "$soname" = "$candidate" ]; then
                    is_system=1
                    break
                fi
            done
            [ "$is_system" -eq 1 ] && continue
            resolved=""
            while IFS= read -r dir; do
                if [ -f "$dir/$soname" ]; then
                    # THE DIRECTORY CANONICALIZED, THE SONAME KEPT. A search path
                    # such as $ORIGIN/../lib reaches a library already found
                    # through another spelling, and compared as text the two are
                    # different entries: measured on TheRock 10.0.0, libLLVM and
                    # two of its sysdeps libraries were listed and copied twice.
                    # The file name stays the soname, which is the name the
                    # loader looks for in bin/.
                    resolved="$(cd "$dir" && pwd -P)/$soname"
                    break
                fi
            done < <(elf_search_dirs "$elf")
            if [ -z "$resolved" ]; then
                printf '%s needs %s, and no RPATH or RUNPATH entry on it holds that file\n' \
                    "$elf" "$soname" >&2
                return 1
            fi
            if [ -z "${emitted[$resolved]:-}" ]; then
                emitted[$resolved]=1
                printf '%s\n' "$resolved"
            fi
            queue+=("$resolved")
        done < <(elf_needed "$elf")
    done
    return 0
}
