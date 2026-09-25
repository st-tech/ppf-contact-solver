#!/usr/bin/env bash
# File: build-linux-native/scripts/platform.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The host architecture and the backends built for it, decided in one place for
# warmup.sh, build.sh and bundle.sh. Source it, do not execute it, and source
# config.sh first, since PPF_LINUX_BACKENDS is read from there:
#
#     . "$BUILD_LINUX/config.sh"
#     . "$BUILD_LINUX/scripts/platform.sh"
#     resolve_platform || die "..."
#
# After the call these are set and exported:
#   PPF_LINUX_ARCH       x86_64 or aarch64, read off uname -m
#   PPF_LINUX_ARCH_KEY   X86_64 or AARCH64, the suffix this architecture's
#                        entries carry in scripts/downloads.txt
#   PPF_LINUX_LOADER     the soname of glibc's dynamic loader on this architecture,
#                        the one system library whose name differs between them
#   PPF_LINUX_BACKENDS   the backends built and shipped, space separated and in
#                        this order: `cuda rocm cpu`, `cuda cpu`, `rocm cpu` or
#                        `cpu`
#   PPF_LINUX_SOURCE_WHEELS
#                        the frontend packages this architecture has no binary
#                        wheel for, which warmup.sh builds from pinned upstream
#                        source with scripts/source-wheel.sh; empty on x86_64
#
# THE ARCHITECTURE IS THE HOST'S, NEVER A SETTING. Everything here builds for the
# machine it runs on: the bundled interpreter, the wheels pip installs into it and
# the slim ffmpeg are all the host's, so a build for another architecture would be
# a distribution whose pieces do not run together. Nothing cross-compiles.
#
# THE DEFAULT IS EVERY BACKEND THIS ARCHITECTURE CAN BUILD: `cuda rocm cpu` on
# x86_64 and `cuda cpu` on aarch64. downloads.txt pins NVIDIA's x86_64
# redistributable and its aarch64 server platform (linux-sbsa) at the same
# release; ROCm on aarch64 is refused by name, because AMD publishes no aarch64
# ROCm distribution. PPF_LINUX_BACKENDS narrows the set, for a build that wants
# one backend rather than the release's. A set without cpu is refused: the CPU
# backend is what every distribution runs on a machine with no supported GPU, so
# none ships without it.
#
# EVERY BACKEND BUILDS INTO ITS OWN DIRECTORY, `target/<backend>`, which is what
# lets one distribution carry several. They link the same executable name, so
# `crates/ppf-cts-solver/build.rs` refuses to put two in one directory and names
# that variable in its own refusal. `backend_target_dir` below is the one place
# that spelling lives, and `App.get_backend` is what chooses among the
# directories at run time.
#
# No `set -euo pipefail` here: this file is sourced, and those options would leak
# into the caller's shell. Each function returns non-zero on failure after saying
# why, and every caller checks.

resolve_platform() {
    local machine requested backend want_cuda=0 want_rocm=0 want_cpu=0

    machine="$(uname -m)"
    case "$machine" in
        x86_64)
            PPF_LINUX_ARCH=x86_64
            PPF_LINUX_ARCH_KEY=X86_64
            PPF_LINUX_LOADER=ld-linux-x86-64.so.2
            PPF_LINUX_SOURCE_WHEELS=""
            ;;
        aarch64)
            PPF_LINUX_ARCH=aarch64
            PPF_LINUX_ARCH_KEY=AARCH64
            PPF_LINUX_LOADER=ld-linux-aarch64.so.1
            # No release of triangle has published an aarch64 wheel, and its
            # last source distribution predates the version the frontend uses.
            PPF_LINUX_SOURCE_WHEELS="triangle"
            ;;
        *)
            printf 'ERROR: this build supports x86_64 and aarch64, and uname -m reports %s\n' \
                "$machine" >&2
            return 1
            ;;
    esac

    requested="${PPF_LINUX_BACKENDS:-}"
    if [ -z "$requested" ]; then
        case "$PPF_LINUX_ARCH" in
            x86_64) requested="cuda rocm cpu" ;;
            aarch64) requested="cuda cpu" ;;
        esac
    fi
    for backend in $requested; do
        case "$backend" in
            cuda) want_cuda=1 ;;
            rocm) want_rocm=1 ;;
            cpu) want_cpu=1 ;;
            *)
                printf 'ERROR: PPF_LINUX_BACKENDS names "%s", and the backends are cuda, rocm and cpu\n' \
                    "$backend" >&2
                return 1
                ;;
        esac
    done
    if [ "$want_cpu" -eq 0 ]; then
        printf 'ERROR: PPF_LINUX_BACKENDS="%s" leaves out cpu\n' "$requested" >&2
        printf '       Every distribution ships the CPU backend, which is what runs on a\n' >&2
        printf '       machine with no supported GPU. Name cpu, or leave the setting empty.\n' >&2
        return 1
    fi
    if [ "$want_rocm" -eq 1 ] && [ "$PPF_LINUX_ARCH" != x86_64 ]; then
        printf 'ERROR: PPF_LINUX_BACKENDS asks for rocm, and this host is %s\n' "$PPF_LINUX_ARCH" >&2
        printf '       AMD publishes ROCm (its apt repository and TheRock SDK tarballs) for\n' >&2
        printf '       x86_64 hosts only, so there is no aarch64 toolkit to build against.\n' >&2
        printf '       Name cuda cpu or cpu on this host.\n' >&2
        return 1
    fi

    # Built additively and in a fixed order, so every consumer sees the same
    # list whatever order the caller named them in, and the GPU backends come
    # before the CPU one as they do everywhere else.
    PPF_LINUX_BACKENDS=""
    [ "$want_cuda" -eq 0 ] || PPF_LINUX_BACKENDS="cuda"
    [ "$want_rocm" -eq 0 ] || PPF_LINUX_BACKENDS="${PPF_LINUX_BACKENDS:+$PPF_LINUX_BACKENDS }rocm"
    PPF_LINUX_BACKENDS="${PPF_LINUX_BACKENDS:+$PPF_LINUX_BACKENDS }cpu"
    export PPF_LINUX_ARCH PPF_LINUX_ARCH_KEY PPF_LINUX_LOADER PPF_LINUX_BACKENDS \
        PPF_LINUX_SOURCE_WHEELS
    return 0
}

# gpu_backends: the GPU backends among PPF_LINUX_BACKENDS, one per line, in the
# order they are built and searched. A distribution can carry more than one, and
# every step that walks them takes the order from here.
gpu_backends() {
    local backend
    for backend in ${PPF_LINUX_BACKENDS:-}; do
        [ "$backend" = cpu ] || printf '%s\n' "$backend"
    done
}

# backend_target_dir NAME: the target directory NAME builds into, relative to
# the source root.
#
# ONE PLACE DECIDES THE LAYOUT. build.sh builds there, bundle.sh copies from
# there, the launcher and verify-distribution.sh read it, and
# `frontend._backends_` searches the same names on the other side, so the
# spelling is written once rather than in each of them.
backend_target_dir() {
    printf 'target/%s\n' "$1"
}

# rocm_sdk_dir: the ROCm SDK this build compiles and links against, the
# directory warmup.sh unpacks TheRock's tarball into. Derived from this file's
# own location so it depends on nothing the caller has set, and written once
# here because build.sh compiles against it and build_tree_run reads the HIP
# runtime out of it.
rocm_sdk_dir() {
    local build_linux
    build_linux="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
    printf '%s/rocm\n' "$build_linux"
}

# build_tree_run BACKEND CMD...: run one of the BUILD TREE's own binaries.
#
# A PACKAGED BINARY RUNS WITH NO LD_LIBRARY_PATH, AND A BUILD-TREE ROCm ONE
# CANNOT. The solver, the server and the cdylib each name libamdhip64.so.7 as a
# DIRECT dependency, because check_gpu asks the HIP runtime which device is
# present rather than loading the backend library to find out. In the build tree
# that runtime exists only inside the SDK, while the binary's RUNPATH names the
# directory holding its backend library, so nothing on its own search path
# resolves it and it does not start. bundle.sh copies the runtime into the
# payload's bin/ and rewrites the RPATH to reach it, which is what makes a
# packaged binary run from the payload alone; bundle.sh's poison test and
# verify-distribution.sh prove that over the payload, with no LD_LIBRARY_PATH
# and against one naming a broken copy, and those runs are deliberately left
# alone.
#
# EVERY OTHER BACKEND IS RUN WITH LD_LIBRARY_PATH REMOVED, so a CUDA or CPU
# binary that acquired a dependency this machine happens to satisfy still fails
# here rather than in a user's hands.
build_tree_run() {
    local backend="$1"
    shift
    if [ "$backend" = rocm ]; then
        env LD_LIBRARY_PATH="$(rocm_sdk_dir)/lib" "$@"
    else
        env -u LD_LIBRARY_PATH "$@"
    fi
}

# has_backend NAME: true when NAME is one of PPF_LINUX_BACKENDS.
has_backend() {
    case " ${PPF_LINUX_BACKENDS:-} " in
        *" $1 "*) return 0 ;;
    esac
    return 1
}

# select_arch_downloads BASE...: for each BASE, sets and exports URL_BASE,
# FILE_BASE and SHA256_BASE from this architecture's URL_BASE_<ARCH_KEY>,
# FILE_BASE_<ARCH_KEY> and SHA256_BASE_<ARCH_KEY>, which load_downloads has
# already exported. A missing one is an error rather than a fallback to another
# architecture's file, which would install something this host cannot run.
#
# The manifest keys keep their suffix, so a caller naming an entry to
# check-downloads.sh names URL_BASE_<ARCH_KEY>, the key the manifest defines.
select_arch_downloads() {
    local base part source_var target_var
    for base in "$@"; do
        for part in URL FILE SHA256; do
            source_var="${part}_${base}_${PPF_LINUX_ARCH_KEY}"
            target_var="${part}_${base}"
            if [ -z "${!source_var:-}" ]; then
                printf 'ERROR: scripts/downloads.txt defines no %s, so %s has no %s entry\n' \
                    "$source_var" "$PPF_LINUX_ARCH" "$base" >&2
                return 1
            fi
            printf -v "$target_var" '%s' "${!source_var}"
            export "${target_var?}"
        done
    done
    return 0
}
