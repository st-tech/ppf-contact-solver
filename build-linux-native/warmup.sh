#!/usr/bin/env bash
# File: build-linux-native/warmup.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# One-time provisioning for a native Linux build of the solver, the counterpart
# of build-mac-native/warmup.sh and build-win-native/warmup.bat. Run it once per
# machine, then ./build.sh.
#
# It is safe to run twice: every step checks for what it would install and skips
# the install while still verifying the result, so a second run is a check
# rather than a reinstall.
#
# IT NEEDS NO ROOT. It writes into this directory and into the developer
# environment at PPF_CTS_VENV, and nowhere else.
#
# WHAT IT PROVISIONS
#   - Rust, into build-linux-native/rust, only when the host has no cargo.
#   - The CUDA 12.8 toolkit into build-linux-native/cuda, from NVIDIA's
#     per-component archives, which build.sh selects through PPF_CUDA_ROOT.
#     Only when CUDA is among the backends (scripts/platform.sh), which is the
#     default on both architectures.
#   - The ROCm SDK, AMD's TheRock tarball, into build-linux-native/rocm, which
#     build.sh selects through ROCM_PATH. Only when ROCm is among the backends,
#     which is x86_64 alone.
#   - Wheels built from pinned upstream source for the frontend packages this
#     architecture has no binary wheel for, into build-linux-native/wheels
#     (scripts/source-wheel.sh; none on x86_64, triangle on aarch64).
#   - A developer Python environment carrying the frontend dependencies, from
#     the canonical list in the repository's own warmup.py.
#   - A relocatable CPython in build-linux-native/python carrying the same
#     list, for bundle.sh to ship inside the distribution.
#   - patchelf, which bundle.sh rewrites library search paths with.
#   - nasm, only on x86_64 and only when the host has no nasm 2.13 or newer,
#     for x264's x86 assembly.
#   - The slim ffmpeg, into build-linux-native/ffmpeg.
#
# WHAT THE HOST PROVIDES: glibc, a C and C++ compiler (one CUDA 12.8 accepts
# when CUDA is built), make, binutils, curl, tar, xz, git, rsync, pkg-config,
# and a Python 3.10 or newer for the developer environment, with its headers
# where a wheel is built from source. Each one found missing is named.
#
# A HOST WITH NO ROUTE TO THE DOWNLOAD HOSTS CAN STILL PROVISION. A file from
# scripts/downloads.txt that is already in downloads/ and matches its checksum
# is neither probed nor fetched; the ffmpeg sources are used from downloads/src
# when they are there; and PPF_LINUX_WHEELHOUSE (config.sh) installs the Python
# packages from a directory with no index. A run that needs nothing from the
# network says so and opens no connection for its downloads.
#
# SWITCHES, each defaulting to on:
#   PPF_LINUX_PYTHON=0  skips the bundled interpreter; bundle.sh then refuses.
#   PPF_LINUX_FFMPEG=0  skips the slim ffmpeg; bundle.sh then refuses.

set -euo pipefail

BUILD_LINUX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$BUILD_LINUX/.." && pwd)"
LOGFILE="$BUILD_LINUX/warmup.log"
DOWNLOADS="$BUILD_LINUX/downloads"

# Re-run under tee so the whole provision is logged. The exit status comes out
# of PIPESTATUS: a pipeline's status is its LAST command's, which is tee's, and
# tee succeeds whatever the script did.
if [ -z "${PPF_LINUX_WARMUP_LOGGING:-}" ]; then
    printf 'Logging to %s\n' "$LOGFILE"
    set +e
    PPF_LINUX_WARMUP_LOGGING=1 "${BASH_SOURCE[0]}" "$@" 2>&1 | tee "$LOGFILE"
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

step() { printf '\n=== %s ===\n' "$1"; }

HOST_HINT="Install it with the system package manager, for example
         sudo apt install build-essential git curl xz-utils rsync pkg-config python3-venv
         sudo dnf install gcc-c++ make binutils git curl xz rsync pkgconf-pkg-config python3"

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die \
        "required tool not found on PATH: $1" \
        "$HOST_HINT"
}

printf "=== ZOZO's Contact Solver, native Linux environment setup ===\n"
printf 'Build directory:  %s\n' "$BUILD_LINUX"
printf 'Source directory: %s\n' "$SRC_DIR"
printf 'Log file:         %s\n' "$LOGFILE"

# ---------------------------------------------------------------------------
step "Checking the host"
# ---------------------------------------------------------------------------

[ "$(uname -s)" = "Linux" ] || die \
    "this script provisions a Linux host, and uname reports $(uname -s)" \
    "On macOS use build-mac-native/warmup.sh; on Windows build-win-native/warmup.bat."
# shellcheck source=build-linux-native/config.sh
. "$BUILD_LINUX/config.sh"
# shellcheck source=build-linux-native/scripts/platform.sh
. "$BUILD_LINUX/scripts/platform.sh"
resolve_platform || die "this host, or PPF_LINUX_BACKENDS, cannot be provisioned (see above)"

# glibc, asked by name. The distribution links against glibc and the bundled
# interpreter is a glibc build, so a musl host such as Alpine can neither build
# nor run it, and the failure there would otherwise be a loader message naming
# a file that exists.
HOST_LIBC="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
case "$HOST_LIBC" in
    "glibc "*) ;;
    *) die "this host's C library is not glibc (getconf GNU_LIBC_VERSION reports: ${HOST_LIBC:-nothing})" \
           "The distribution is built against glibc and cannot be built on another C library." ;;
esac

for tool in gcc g++ make readelf strip curl tar xz git rsync sha256sum pkg-config; do
    require_cmd "$tool"
done

HOST_OS="$(. /etc/os-release 2>/dev/null && printf '%s' "${PRETTY_NAME:-}")" || HOST_OS=""
printf 'Host:  %s, %s, %s\n' "${HOST_OS:-unknown distribution}" "$HOST_LIBC" "$(uname -m)"
printf 'C++:   %s\n' "$(g++ --version | head -1)"
printf 'Build: %s, backends: %s\n' "$PPF_LINUX_ARCH" "$PPF_LINUX_BACKENDS"

# shellcheck source=build-linux-native/scripts/load-downloads.sh
. "$BUILD_LINUX/scripts/load-downloads.sh"

# The ffmpeg entries, read one by one from the manifest the Windows build and
# the slim ffmpeg script share. Loading that file whole would also bring its
# Windows URL_PYTHON and URL_RUSTUP into this environment.
WIN_MANIFEST="$SRC_DIR/build-win-native/scripts/downloads.txt"
for key in URL_FFMPEG_GIT FFMPEG_TAG URL_X264_GIT X264_COMMIT URL_ZLIB FILE_ZLIB SHA256_ZLIB; do
    value="$(manifest_value "$WIN_MANIFEST" "$key")" || die "could not read $key from $WIN_MANIFEST"
    printf -v "$key" '%s' "$value"
done
load_downloads "$BUILD_LINUX/scripts/downloads.txt" || die \
    "could not read build-linux-native/scripts/downloads.txt"
# The interpreter and patchelf are per-architecture entries. This makes the
# host's the ones every step below reads as URL_PYTHON, FILE_PATCHELF and so on.
select_arch_downloads PYTHON PATCHELF || die \
    "scripts/downloads.txt has no complete $PPF_LINUX_ARCH entry (see above)"

# Every URL_CUDA_*_<ARCH_KEY> entry is one toolkit component for that
# architecture; see downloads.txt. select_arch_downloads makes the host's the
# ones read below as URL_CUDA_NVCC, FILE_CUDA_NVCC and so on, and refuses a
# component this architecture has no entry for rather than taking another's.
CUDA_SUFFIXES=""
if has_backend cuda; then
    for key in $PPF_DOWNLOAD_KEYS; do
        case "$key" in
            URL_CUDA_*_"$PPF_LINUX_ARCH_KEY")
                base="${key#URL_}"
                CUDA_SUFFIXES="$CUDA_SUFFIXES ${base%_"$PPF_LINUX_ARCH_KEY"}" ;;
        esac
    done
    [ -n "$CUDA_SUFFIXES" ] || die \
        "scripts/downloads.txt defines no URL_CUDA_*_$PPF_LINUX_ARCH_KEY entries"
    # shellcheck disable=SC2086 # a space-separated list of entry names
    select_arch_downloads $CUDA_SUFFIXES || die \
        "scripts/downloads.txt has no complete $PPF_LINUX_ARCH CUDA entry (see above)"

    # NVIDIA's name for the platform, which every archive name carries:
    # linux-x86_64, and linux-sbsa for its aarch64 server platform.
    case "$PPF_LINUX_ARCH" in
        x86_64) CUDA_PLATFORM=linux-x86_64 ;;
        aarch64) CUDA_PLATFORM=linux-sbsa ;;
    esac

    # The CUDA release, read off the compiler archive's name rather than written
    # down a second time: cuda_nvcc-<platform>-<major>.<minor>.<patch>-archive.
    CUDA_NVCC_VERSION="${FILE_CUDA_NVCC#cuda_nvcc-"$CUDA_PLATFORM"-}"
    CUDA_NVCC_VERSION="${CUDA_NVCC_VERSION%-archive.tar.xz}"
    CUDA_RELEASE="$(printf '%s' "$CUDA_NVCC_VERSION" | cut -d. -f1-2)"
    case "$CUDA_RELEASE" in
        [0-9]*.[0-9]*) ;;
        *) die "could not read a CUDA release out of FILE_CUDA_NVCC=$FILE_CUDA_NVCC" ;;
    esac
fi

# The ROCm SDK is one archive per architecture, and only x86_64 has one;
# platform.sh has already refused rocm anywhere else.
if has_backend rocm; then
    select_arch_downloads ROCM_SDK || die \
        "scripts/downloads.txt has no complete $PPF_LINUX_ARCH ROCm SDK entry (see above)"
    [ -n "${ROCM_HIP_VERSION:-}" ] || die "scripts/downloads.txt defines no ROCM_HIP_VERSION"
fi

sha256_of() {
    sha256sum "$1" | cut -d' ' -f1
}

# True when downloads/<FILE_x> is present and, for an entry carrying SHA256_x,
# matches it. $1 is the suffix after URL_/FILE_/SHA256_.
cached_ok() {
    local file_var="FILE_$1" sha_var="SHA256_$1" path
    path="$DOWNLOADS/${!file_var}"
    [ -f "$path" ] || return 1
    if [ -n "${!sha_var:-}" ]; then
        [ "$(sha256_of "$path")" = "${!sha_var}" ] || return 1
    fi
    return 0
}

# Make downloads/<FILE_x> present and verified, fetching it when it is absent.
# The transfer writes to a .part file first, so an interrupted download is never
# mistaken for a cached one on the next run.
fetch() {
    local url_var="URL_$1" file_var="FILE_$1" sha_var="SHA256_$1" path have
    path="$DOWNLOADS/${!file_var}"
    mkdir -p "$DOWNLOADS"
    if [ -f "$path" ]; then
        printf 'Using the cached %s\n' "${!file_var}"
    else
        printf 'Downloading %s\n' "${!file_var}"
        if ! curl -fL --retry 3 --retry-delay 2 --connect-timeout 15 \
            -o "$path.part" "${!url_var}"; then
            rm -f "$path.part"
            die "failed to download ${!url_var}" \
                "On a host that cannot reach it, copy ${!file_var} into" \
                "$DOWNLOADS by hand and re-run: a file that is there and matches" \
                "its checksum is used without any network."
        fi
        mv "$path.part" "$path"
    fi
    if [ -n "${!sha_var:-}" ]; then
        have="$(sha256_of "$path")"
        [ "$have" = "${!sha_var}" ] || die \
            "checksum mismatch on $path" \
            "expected  ${!sha_var}" \
            "measured  $have" \
            "Delete the file and re-run so it is fetched again:" \
            "  rm -f $path"
    fi
}

# ---------------------------------------------------------------------------
# What this run has to provision, decided before anything is fetched
# ---------------------------------------------------------------------------

RUST_DIR="$BUILD_LINUX/rust"
CUDA_ROOT="$BUILD_LINUX/cuda"
ROCM_ROOT="$BUILD_LINUX/rocm"
PY_ROOT="$BUILD_LINUX/python"
PATCHELF_DIR="$BUILD_LINUX/patchelf"
NASM_DIR="$BUILD_LINUX/nasm"
FFMPEG_DIR="$BUILD_LINUX/ffmpeg"
FFMPEG_SOURCES="$DOWNLOADS/src"
# scripts/source-wheel.sh keeps each package's repository under the same
# downloads/src and writes the wheels here.
SOURCE_WHEEL_SOURCES="$DOWNLOADS/src"
SOURCE_WHEELS_DIR="$BUILD_LINUX/wheels"

CARGO=""
if command -v cargo >/dev/null 2>&1; then
    CARGO="$(command -v cargo)"
elif [ -x "$HOME/.cargo/bin/cargo" ]; then
    CARGO="$HOME/.cargo/bin/cargo"
elif [ -x "$RUST_DIR/bin/cargo" ]; then
    CARGO="$RUST_DIR/bin/cargo"
fi

# A host nasm is used when it is new enough for x264, which needs 2.13.
host_nasm_ok() {
    local version major minor
    command -v nasm >/dev/null 2>&1 || return 1
    version="$(nasm -v 2>/dev/null | awk '{ print $3; exit }')"
    major="${version%%.*}"
    minor="$(printf '%s' "$version" | cut -d. -f2)"
    case "$major$minor" in '' | *[!0-9]*) return 1 ;; esac
    [ "$major" -gt 2 ] || { [ "$major" -eq 2 ] && [ "$minor" -ge 13 ]; }
}

NEED_RUST=0; [ -n "$CARGO" ] || NEED_RUST=1
NEED_CUDA=0
if has_backend cuda && [ ! -e "$CUDA_ROOT" ]; then NEED_CUDA=1; fi
NEED_ROCM=0
if has_backend rocm && [ ! -e "$ROCM_ROOT" ]; then NEED_ROCM=1; fi
NEED_PYTHON=0
[ "${PPF_LINUX_PYTHON:-1}" = "0" ] || [ -e "$PY_ROOT" ] || NEED_PYTHON=1
NEED_PATCHELF=0; [ -e "$PATCHELF_DIR" ] || NEED_PATCHELF=1
NEED_FFMPEG=0
[ "${PPF_LINUX_FFMPEG:-1}" = "0" ] || [ -e "$FFMPEG_DIR/ffmpeg" ] || NEED_FFMPEG=1
# nasm assembles x264's x86 code, so only an x86_64 build asks for it.
NEED_NASM=0
if [ "$PPF_LINUX_ARCH" = x86_64 ] && [ "$NEED_FFMPEG" -eq 1 ] && ! host_nasm_ok \
    && [ ! -e "$NASM_DIR" ]; then
    NEED_NASM=1
fi

zlib_cached_ok() {
    [ -f "$FFMPEG_SOURCES/$FILE_ZLIB" ] &&
        [ "$(sha256_of "$FFMPEG_SOURCES/$FILE_ZLIB")" = "$SHA256_ZLIB" ]
}

# ---------------------------------------------------------------------------
step "Checking download pointers"
# ---------------------------------------------------------------------------

PROBE_LINUX=""
PROBE_WIN=""
if [ "$NEED_RUST" -eq 1 ] && ! cached_ok RUSTUP; then PROBE_LINUX="$PROBE_LINUX URL_RUSTUP"; fi
# A per-architecture entry is probed under the key the manifest defines.
if [ "$NEED_PYTHON" -eq 1 ] && ! cached_ok PYTHON; then
    PROBE_LINUX="$PROBE_LINUX URL_PYTHON_$PPF_LINUX_ARCH_KEY"
fi
if [ "$NEED_PATCHELF" -eq 1 ] && ! cached_ok PATCHELF; then
    PROBE_LINUX="$PROBE_LINUX URL_PATCHELF_$PPF_LINUX_ARCH_KEY"
fi
if [ "$NEED_NASM" -eq 1 ] && ! cached_ok NASM; then PROBE_LINUX="$PROBE_LINUX URL_NASM"; fi
if [ "$NEED_CUDA" -eq 1 ]; then
    for suffix in $CUDA_SUFFIXES; do
        cached_ok "$suffix" || PROBE_LINUX="$PROBE_LINUX URL_${suffix}_$PPF_LINUX_ARCH_KEY"
    done
fi
if [ "$NEED_ROCM" -eq 1 ] && ! cached_ok ROCM_SDK; then
    PROBE_LINUX="$PROBE_LINUX URL_ROCM_SDK_$PPF_LINUX_ARCH_KEY"
fi
if [ "$NEED_FFMPEG" -eq 1 ]; then
    [ -d "$FFMPEG_SOURCES/x264/.git" ] || PROBE_WIN="$PROBE_WIN URL_X264_GIT"
    [ -d "$FFMPEG_SOURCES/ffmpeg-$FFMPEG_TAG/.git" ] || PROBE_WIN="$PROBE_WIN URL_FFMPEG_GIT"
    zlib_cached_ok || PROBE_WIN="$PROBE_WIN URL_ZLIB"
fi
# A source-built wheel's repositories, unless its sources are already under
# downloads/src, which scripts/source-wheel.sh then uses with no network.
for pkg in $PPF_LINUX_SOURCE_WHEELS; do
    [ -d "$SOURCE_WHEEL_SOURCES/$pkg/.git" ] && continue
    pkg_key="$(printf '%s' "$pkg" | tr 'a-z-' 'A-Z_')"
    for key in $PPF_DOWNLOAD_KEYS; do
        case "$key" in
            "URL_${pkg_key}_GIT" | "URL_${pkg_key}_"*"_GIT") PROBE_LINUX="$PROBE_LINUX $key" ;;
        esac
    done
done

if [ -z "$PROBE_LINUX$PROBE_WIN" ]; then
    printf 'Every download this run needs is already in downloads/ and verified,\n'
    printf 'so nothing is probed and nothing is fetched.\n'
else
    if [ -n "$PROBE_LINUX" ]; then
        # shellcheck disable=SC2086
        "$BUILD_LINUX/scripts/check-downloads.sh" $PROBE_LINUX || die \
            "one or more download URLs are unreachable" \
            "Fix the pointer in build-linux-native/scripts/downloads.txt, or place the" \
            "file in $DOWNLOADS by hand; a verified cached file is not probed."
    fi
    if [ -n "$PROBE_WIN" ]; then
        # shellcheck disable=SC2086
        "$BUILD_LINUX/scripts/check-downloads.sh" --manifest "$WIN_MANIFEST" $PROBE_WIN || die \
            "one or more ffmpeg source URLs are unreachable" \
            "Fix the pointer in build-win-native/scripts/downloads.txt, or place the" \
            "sources in $FFMPEG_SOURCES by hand (see make-slim-ffmpeg.sh)."
    fi
fi

# ---------------------------------------------------------------------------
step "Rust"
# ---------------------------------------------------------------------------

if [ "$NEED_RUST" -eq 0 ]; then
    printf 'cargo: %s\n' "$CARGO"
else
    printf 'No cargo on this host. Installing rustup into %s\n' "$RUST_DIR"
    fetch RUSTUP
    # --no-modify-path and the two HOME overrides keep the install inside this
    # directory: nothing is written to ~/.cargo, ~/.rustup or a shell profile.
    CARGO_HOME="$RUST_DIR" RUSTUP_HOME="$RUST_DIR/rustup" \
        sh "$DOWNLOADS/$FILE_RUSTUP" -y --no-modify-path --profile minimal \
        --default-toolchain stable || die "rustup install failed"
    CARGO="$RUST_DIR/bin/cargo"
fi
if [ "$CARGO" = "$RUST_DIR/bin/cargo" ]; then
    CARGO_HOME="$RUST_DIR" RUSTUP_HOME="$RUST_DIR/rustup" "$CARGO" --version \
        || die "$CARGO does not run"
else
    "$CARGO" --version || die "$CARGO does not run"
fi

# ---------------------------------------------------------------------------
step "CUDA toolkit"
# ---------------------------------------------------------------------------

# The archive names this toolkit was assembled from, one per line. A toolkit
# whose stamp differs was built from a different pin than downloads.txt names.
cuda_expected_stamp() {
    local suffix file_var
    for suffix in $CUDA_SUFFIXES; do
        file_var="FILE_$suffix"
        printf '%s\n' "${!file_var}"
    done
}

verify_cuda() {
    local banner tool
    banner="$("$CUDA_ROOT/bin/nvcc" --version 2>&1)" || {
        printf '%s\n' "$banner" >&2
        printf 'nvcc at %s does not run\n' "$CUDA_ROOT/bin/nvcc" >&2
        return 1
    }
    case "$banner" in
        *"release $CUDA_RELEASE,"*) ;;
        *) printf 'nvcc reports: %s, and the pin is %s\n' \
               "$(printf '%s\n' "$banner" | grep -m1 'release' || printf 'no release line')" \
               "$CUDA_RELEASE" >&2
           return 1 ;;
    esac
    for tool in cuobjdump nvdisasm; do
        [ -x "$CUDA_ROOT/bin/$tool" ] || { printf 'no %s in %s/bin\n' "$tool" "$CUDA_ROOT" >&2; return 1; }
    done
    [ -f "$CUDA_ROOT/lib64/libcudart.so.12" ] || {
        printf 'no lib64/libcudart.so.12 in %s\n' "$CUDA_ROOT" >&2
        return 1
    }
    [ "$(cat "$CUDA_ROOT/.ppf-cuda-components" 2>/dev/null)" = "$(cuda_expected_stamp)" ] || {
        printf '%s was assembled from different archives than downloads.txt names\n' "$CUDA_ROOT" >&2
        return 1
    }
    printf 'nvcc: %s\n' "$(printf '%s\n' "$banner" | grep -m1 'release')"
}

CUDA_REPAIR="rm -rf $CUDA_ROOT && $BUILD_LINUX/warmup.sh"
if ! has_backend cuda; then
    printf 'CUDA is not among the backends (%s), so no toolkit is provisioned.\n' "$PPF_LINUX_BACKENDS"
    CUDA_STATUS="not needed, backends: $PPF_LINUX_BACKENDS"
elif [ "$NEED_CUDA" -eq 0 ]; then
    printf 'Found %s. Verifying rather than reinstalling.\n' "$CUDA_ROOT"
    verify_cuda || die "the toolkit at $CUDA_ROOT does not match scripts/downloads.txt" \
        "Repair with:  $CUDA_REPAIR"
    CUDA_STATUS="verified at $CUDA_ROOT"
else
    # Assembled in a scratch directory and moved into place, so an interrupted
    # extraction never leaves something that looks provisioned. The archives
    # share one layout (bin, include, lib, nvvm), so extracting them over each
    # other merges them into one root, which is what nvcc expects around it.
    CUDA_STAGE="$BUILD_LINUX/.cuda-stage"
    rm -rf "$CUDA_STAGE"
    mkdir -p "$CUDA_STAGE"
    for suffix in $CUDA_SUFFIXES; do
        fetch "$suffix"
        file_var="FILE_$suffix"
        printf 'Extracting %s\n' "${!file_var}"
        tar -xJf "$DOWNLOADS/${!file_var}" -C "$CUDA_STAGE" --strip-components=1 || die \
            "tar could not extract $DOWNLOADS/${!file_var}"
    done
    # A system install calls its library directory lib64 and nvcc links against
    # that name; the redistributable archives call it lib. Both names are made
    # to resolve.
    if [ -d "$CUDA_STAGE/lib" ] && [ ! -e "$CUDA_STAGE/lib64" ]; then
        mv "$CUDA_STAGE/lib" "$CUDA_STAGE/lib64"
        ln -s lib64 "$CUDA_STAGE/lib"
    fi
    cuda_expected_stamp > "$CUDA_STAGE/.ppf-cuda-components"
    mv "$CUDA_STAGE" "$CUDA_ROOT"
    verify_cuda || die "the toolkit assembled at $CUDA_ROOT does not run" \
        "Repair with:  $CUDA_REPAIR"
    CUDA_STATUS="installed at $CUDA_ROOT"
fi

# ---------------------------------------------------------------------------
step "ROCm SDK"
# ---------------------------------------------------------------------------

# THE SDK IS AMD'S TheRock TARBALL, UNPACKED WITH NO INSTALLER AND NO ROOT. The
# HIP release hipcc reports is checked against ROCM_HIP_VERSION in downloads.txt,
# and the root records the archive it came from, so an SDK assembled from another
# release is refused rather than built against.
verify_rocm() {
    local banner lib
    [ -x "$ROCM_ROOT/bin/hipcc" ] || { printf 'no bin/hipcc in %s\n' "$ROCM_ROOT" >&2; return 1; }
    banner="$("$ROCM_ROOT/bin/hipcc" --version 2>&1)" || {
        printf '%s\n' "$banner" >&2
        printf 'hipcc at %s does not run\n' "$ROCM_ROOT/bin/hipcc" >&2
        return 1
    }
    case "$banner" in
        *"HIP version: $ROCM_HIP_VERSION."*) ;;
        *) printf 'hipcc reports: %s, and the pin is HIP %s\n' \
               "$(printf '%s\n' "$banner" | grep -m1 'HIP version' || printf 'no HIP version line')" \
               "$ROCM_HIP_VERSION" >&2
           return 1 ;;
    esac
    for lib in libamdhip64.so.7 libhsa-runtime64.so.1; do
        [ -e "$ROCM_ROOT/lib/$lib" ] || { printf 'no lib/%s in %s\n' "$lib" "$ROCM_ROOT" >&2; return 1; }
    done
    [ "$(cat "$ROCM_ROOT/.ppf-rocm-archive" 2>/dev/null)" = "$FILE_ROCM_SDK" ] || {
        printf '%s was unpacked from a different archive than downloads.txt names\n' "$ROCM_ROOT" >&2
        return 1
    }
    printf 'hipcc: %s\n' "$(printf '%s\n' "$banner" | grep -m1 'HIP version')"
}

ROCM_REPAIR="rm -rf $ROCM_ROOT && $BUILD_LINUX/warmup.sh"
if ! has_backend rocm; then
    printf 'ROCm is not among the backends (%s), so no SDK is provisioned.\n' "$PPF_LINUX_BACKENDS"
    ROCM_STATUS="not needed, backends: $PPF_LINUX_BACKENDS"
elif [ "$NEED_ROCM" -eq 0 ]; then
    printf 'Found %s. Verifying rather than reinstalling.\n' "$ROCM_ROOT"
    verify_rocm || die "the ROCm SDK at $ROCM_ROOT does not match scripts/downloads.txt" \
        "Repair with:  $ROCM_REPAIR"
    ROCM_STATUS="verified at $ROCM_ROOT"
else
    # Unpacked in a scratch directory and moved into place, so an interrupted
    # extraction never leaves something that looks provisioned. The archive has no
    # top-level directory: its entries are the SDK root itself.
    ROCM_STAGE="$BUILD_LINUX/.rocm-stage"
    rm -rf "$ROCM_STAGE"
    mkdir -p "$ROCM_STAGE"
    fetch ROCM_SDK
    # SIX PATHS ARE LEFT IN THE TARBALL, each a math library this project neither
    # compiles against nor ships: MIOpen's libraries and its database, rocBLAS and
    # hipBLASLt's kernel directories, the device operation archives, and the .kpack
    # payloads those libraries load. Measured on TheRock 10.0.0 they are 15 of its
    # 20 GiB, which is the difference between an SDK a hosted CI runner's disk can
    # hold beside the tarball and the build, and one it cannot.
    # build-win-native/warmup.bat leaves out the same families. Anything not named
    # is unpacked, so a file the toolchain turns out to need is present rather
    # than filtered out by a guess.
    ROCM_EXCLUDES=(
        --exclude=./.kpack
        --exclude='./lib/libMIOpen*'
        --exclude=./lib/rocblas
        --exclude=./lib/hipblaslt
        --exclude='./lib/libdevice_*_operations.a'
        --exclude=./share/miopen
    )
    printf 'Extracting %s (about 5.3 GiB unpacked, the math libraries left out)\n' "$FILE_ROCM_SDK"
    tar -xzf "$DOWNLOADS/$FILE_ROCM_SDK" -C "$ROCM_STAGE" "${ROCM_EXCLUDES[@]}" || die \
        "tar could not extract $DOWNLOADS/$FILE_ROCM_SDK"
    printf '%s' "$FILE_ROCM_SDK" > "$ROCM_STAGE/.ppf-rocm-archive"
    mv "$ROCM_STAGE" "$ROCM_ROOT"
    verify_rocm || die "the ROCm SDK unpacked at $ROCM_ROOT does not run" \
        "Repair with:  $ROCM_REPAIR"
    # THE ARCHIVE IS DROPPED ONLY WHERE IT CAN BE FETCHED AGAIN, so this is OFF
    # by default and CI turns it on. It is 8.23 GiB and pure cache once the SDK
    # above is unpacked and verified, which is the difference between a hosted
    # runner holding this build and not: an x86_64 distribution carries CUDA and
    # ROCm together, and the tree peaks at 23 GB with the archive against about
    # 15 GB without it, where the runner has 18 GB free beside the swap file its
    # CUDA device link needs. A dev box has no outbound network at all, so an
    # archive deleted there cannot be fetched again and the build stops being
    # repeatable; the boxes therefore never set this.
    if [ -n "${PPF_LINUX_DROP_ROCM_ARCHIVE:-}" ]; then
        rm -f "$DOWNLOADS/$FILE_ROCM_SDK"
        printf 'Dropped %s: the SDK is unpacked and verified, and PPF_LINUX_DROP_ROCM_ARCHIVE is set\n' \
            "$FILE_ROCM_SDK"
    fi
    ROCM_STATUS="installed at $ROCM_ROOT"
fi

# ---------------------------------------------------------------------------
step "patchelf"
# ---------------------------------------------------------------------------

PATCHELF_VERSION="${FILE_PATCHELF#patchelf-}"
PATCHELF_VERSION="${PATCHELF_VERSION%%-*}"

verify_patchelf() {
    local reported
    reported="$("$PATCHELF_DIR/bin/patchelf" --version 2>&1)" || return 1
    [ "$reported" = "patchelf $PATCHELF_VERSION" ] || {
        printf 'patchelf reports "%s", and the pin is %s\n' "$reported" "$PATCHELF_VERSION" >&2
        return 1
    }
    printf '%s\n' "$reported"
}

if [ "$NEED_PATCHELF" -eq 1 ]; then
    fetch PATCHELF
    PATCHELF_STAGE="$BUILD_LINUX/.patchelf-stage"
    rm -rf "$PATCHELF_STAGE"
    mkdir -p "$PATCHELF_STAGE"
    tar -xzf "$DOWNLOADS/$FILE_PATCHELF" -C "$PATCHELF_STAGE" || die \
        "tar could not extract $DOWNLOADS/$FILE_PATCHELF"
    mv "$PATCHELF_STAGE" "$PATCHELF_DIR"
fi
verify_patchelf || die "the patchelf at $PATCHELF_DIR does not match the pin" \
    "Repair with:  rm -rf $PATCHELF_DIR && $BUILD_LINUX/warmup.sh"

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
step "Python interpreter"

PY=""
if [ -n "${PPF_CTS_PYTHON:-}" ]; then
    [ -x "$PPF_CTS_PYTHON" ] || die \
        "PPF_CTS_PYTHON is set to $PPF_CTS_PYTHON, which is not executable"
    PY="$PPF_CTS_PYTHON"
else
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1 &&
            "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
                >/dev/null 2>&1; then
            PY="$(command -v "$candidate")"
            break
        fi
    done
fi
[ -n "$PY" ] || die \
    "no Python 3.10 or newer found on PATH" \
    "The frontend uses PEP 604 unions, which 3.9 does not parse. Install one" \
    "with the system package manager, or point PPF_CTS_PYTHON at one (config.sh)."
printf 'Using %s (%s)\n' "$PY" "$("$PY" -c 'import platform; print(platform.python_version())')"

# Where pip takes packages from. A wheelhouse is an explicit choice that
# replaces the index entirely, so an incomplete one fails by package name
# rather than falling back to a network the host may not have.
PIP_SOURCE_ARGS=()
if [ -n "${PPF_LINUX_WHEELHOUSE:-}" ]; then
    [ -d "$PPF_LINUX_WHEELHOUSE" ] || die \
        "PPF_LINUX_WHEELHOUSE names $PPF_LINUX_WHEELHOUSE, which is not a directory"
    PIP_SOURCE_ARGS=(--no-index --find-links "$PPF_LINUX_WHEELHOUSE")
    printf 'Packages come from the wheelhouse %s, with no index.\n' "$PPF_LINUX_WHEELHOUSE"
else
    printf 'Packages come from the configured package index.\n'
fi

# The packages this architecture needs built from pinned source. pip finds their
# wheels through --find-links beside the index or wheelhouse above, so every
# other package still comes from where it did.
if [ -n "$PPF_LINUX_SOURCE_WHEELS" ]; then
    mkdir -p "$SOURCE_WHEELS_DIR"
    PIP_SOURCE_ARGS+=(--find-links "$SOURCE_WHEELS_DIR")
    printf 'Built from pinned source on %s: %s, into %s\n' \
        "$PPF_LINUX_ARCH" "$PPF_LINUX_SOURCE_WHEELS" "$SOURCE_WHEELS_DIR"
fi

# build_source_wheels PYTHON: each of them, for one interpreter. A wheel is
# specific to the interpreter's version, so the developer environment and the
# bundled interpreter each get their own when they differ.
build_source_wheels() {
    local python="$1" pkg
    for pkg in $PPF_LINUX_SOURCE_WHEELS; do
        "$BUILD_LINUX/scripts/source-wheel.sh" "$python" "$pkg" "$SOURCE_WHEELS_DIR" || die \
            "building the $pkg wheel from source for $python failed; its output is above" \
            "No binary wheel of $pkg exists for $PPF_LINUX_ARCH, and the frontend does not" \
            "install without it."
    done
}

# The dependency set is read out of warmup.py rather than restated here.
PACKAGES=()
while IFS= read -r pkg; do
    [ -n "$pkg" ] && PACKAGES+=("$pkg")
done < <("$PY" - "$SRC_DIR/warmup.py" <<'PY'
import runpy
import sys

ns = runpy.run_path(sys.argv[1])
for name in list(ns["python_packages"]()) + list(ns["tetra_packages"]()):
    print(name)
PY
)
[ "${#PACKAGES[@]}" -gt 0 ] || die \
    "warmup.py yielded no package list" \
    "python_packages() and tetra_packages() are what this reads; if either" \
    "was renamed, update this script in the same change."

step "Frontend environment"

if [ -z "${PPF_CTS_VENV:-}" ]; then
    PPF_CTS_VENV="$("$PY" - "$SRC_DIR/warmup.py" <<'PY'
import runpy
import sys

ns = runpy.run_path(sys.argv[1])
print(ns["get_venv_path"]())
PY
)"
fi
[ -n "$PPF_CTS_VENV" ] || die "could not determine a venv path"
printf 'Environment: %s\n' "$PPF_CTS_VENV"

if [ ! -x "$PPF_CTS_VENV/bin/python" ]; then
    if [ -e "$PPF_CTS_VENV" ]; then
        die "$PPF_CTS_VENV exists but holds no bin/python" \
            "Remove it and re-run, or point PPF_CTS_VENV somewhere else."
    fi
    printf 'Creating the environment...\n'
    "$PY" -m venv "$PPF_CTS_VENV" || die "python -m venv failed" "$HOST_HINT"
fi
VENV_PY="$PPF_CTS_VENV/bin/python"
[ -x "$VENV_PY" ] || die "no interpreter at $VENV_PY after venv creation"

printf 'Installing %d packages from warmup.py...\n' "${#PACKAGES[@]}"
"$VENV_PY" -m pip install "${PIP_SOURCE_ARGS[@]}" --upgrade pip || die "pip upgrade failed"
build_source_wheels "$VENV_PY"
"$VENV_PY" -m pip install "${PIP_SOURCE_ARGS[@]}" --only-binary=:all: "${PACKAGES[@]}" || die \
    "pip install failed" \
    "Nothing downstream can work around a missing frontend dependency: a build" \
    "worker without scipy silently takes a different pin-diffusion path, and one" \
    "without a tetrahedralizer cannot build a SOLID scene at all."

"$VENV_PY" - "$SRC_DIR/warmup.py" <<'PY' || die "the frontend environment is incomplete"
import importlib.util as util
import runpy
import sys

ns = runpy.run_path(sys.argv[1])
required = list(ns["REQUIRED_PACKAGES"])
missing = [name for name in required if util.find_spec(name) is None]
if missing:
    sys.stderr.write("missing required packages: " + ", ".join(missing) + "\n")
    sys.exit(1)
print("verified %d required packages" % len(required))
PY

# ---------------------------------------------------------------------------
step "Bundled interpreter"
# ---------------------------------------------------------------------------

# Packages install INTO the extracted tree rather than into a venv over it. A
# venv records an absolute path in its pyvenv.cfg and its bin/python names the
# base interpreter by absolute path, so a venv does not survive being copied; a
# python-build-standalone install_only tree is relocatable as an installation.
PY_BUNDLED="$PY_ROOT/bin/python3"
PY_BUNDLED_VERSION="${FILE_PYTHON#cpython-}"
PY_BUNDLED_VERSION="${PY_BUNDLED_VERSION%%+*}"
case "$PY_BUNDLED_VERSION" in
    '' | *[!0-9.]*)
        die "could not read a version out of FILE_PYTHON=$FILE_PYTHON" ;;
esac

# EVERY USE OF THE BUNDLED INTERPRETER HERE SEES ONLY ITS OWN TREE. A user
# site-packages directory (~/.local/lib/python3.X/site-packages) is on sys.path
# by default, and pip treats a package found there as already satisfied, so an
# install over a host whose user site holds numpy leaves numpy out of the tree
# and reports success; a check run the same way then finds it and passes. The
# launcher sets PYTHONNOUSERSITE, so the shipped copy would fail at its first
# import. PYTHONPATH and PYTHONHOME are cleared for the same reason.
bundled_py() {
    env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$PY_BUNDLED" "$@"
}

verify_bundled_python() {
    local jlab_version
    bundled_py - "$SRC_DIR/warmup.py" "$PY_ROOT" <<'PY' || return 1
import importlib.util as util
import os
import runpy
import sys

ns = runpy.run_path(sys.argv[1])
root = os.path.realpath(sys.argv[2]) + os.sep
required = list(ns["REQUIRED_PACKAGES"])
# jupyterlab is asked for by name because REQUIRED_PACKAGES does not carry it,
# and launching it is the whole purpose of the distribution's launcher.
required.append("jupyterlab")
missing, outside = [], []
for name in required:
    spec = util.find_spec(name)
    if spec is None:
        missing.append(name)
        continue
    where = spec.origin or next(iter(spec.submodule_search_locations or []), "")
    if not os.path.realpath(where).startswith(root):
        outside.append("%s (%s)" % (name, where))
if missing:
    sys.stderr.write("missing packages: " + ", ".join(missing) + "\n")
if outside:
    sys.stderr.write("resolved outside the interpreter tree: " + ", ".join(outside) + "\n")
if missing or outside:
    sys.exit(1)
print("verified %d packages inside the tree, jupyterlab among them" % len(required))
PY
    # Every installed distribution's own requirements, inside the tree.
    bundled_py -m pip check >&2 || return 1
    # The entry point the launcher uses, which imports the whole server stack.
    jlab_version="$(bundled_py -m jupyterlab --version)" || return 1
    printf 'python -m jupyterlab reports %s\n' "$jlab_version"
}

if [ "${PPF_LINUX_PYTHON:-1}" = "0" ]; then
    PY_BUNDLED_STATUS="skipped this run, PPF_LINUX_PYTHON=0"
    printf 'PPF_LINUX_PYTHON=0, so no interpreter is provisioned for the\n'
    printf 'distribution. bundle.sh refuses to package without one, and names\n'
    printf 'this switch when it does.\n'
elif [ "$NEED_PYTHON" -eq 0 ]; then
    printf 'Found %s. Verifying rather than reinstalling.\n' "$PY_BUNDLED"
    PY_REPAIR="rm -rf $PY_ROOT && $BUILD_LINUX/warmup.sh"
    [ -x "$PY_BUNDLED" ] || die "$PY_ROOT exists and carries no runnable bin/python3" \
        "Repair with:  $PY_REPAIR"
    PY_HAVE="$("$PY_BUNDLED" -c 'import platform; print(platform.python_version())')" || die \
        "the interpreter at $PY_BUNDLED does not run" "Repair with:  $PY_REPAIR"
    [ "$PY_HAVE" = "$PY_BUNDLED_VERSION" ] || die \
        "$PY_BUNDLED reports Python $PY_HAVE and the pin names $PY_BUNDLED_VERSION" \
        "Repair with:  $PY_REPAIR"
    verify_bundled_python || die \
        "the bundled interpreter is missing packages the distribution needs" \
        "Repair with:  $PY_REPAIR"
    PY_BUNDLED_STATUS="verified at $PY_ROOT"
else
    printf 'Provisioning %s\n' "$PY_ROOT"
    fetch PYTHON
    PY_EXTRACT="$BUILD_LINUX/.python-extract"
    rm -rf "$PY_EXTRACT"
    mkdir -p "$PY_EXTRACT"
    tar -xzf "$DOWNLOADS/$FILE_PYTHON" -C "$PY_EXTRACT" || die \
        "tar could not extract $DOWNLOADS/$FILE_PYTHON"
    [ -x "$PY_EXTRACT/python/bin/python3" ] || die \
        "the archive carries no python/bin/python3" \
        "This step expects a python-build-standalone install_only build."
    mv "$PY_EXTRACT/python" "$PY_ROOT"
    rmdir "$PY_EXTRACT"

    PY_HAVE="$("$PY_BUNDLED" -c 'import platform; print(platform.python_version())')" || die \
        "the extracted interpreter at $PY_BUNDLED does not run"
    [ "$PY_HAVE" = "$PY_BUNDLED_VERSION" ] || die \
        "the extracted interpreter reports Python $PY_HAVE, and $FILE_PYTHON names $PY_BUNDLED_VERSION"
    printf 'Python %s at %s\n' "$PY_HAVE" "$PY_ROOT"

    if ! bundled_py -m pip --version >/dev/null 2>&1; then
        # ensurepip installs from a wheel inside the standard library, so it
        # needs no network of its own.
        bundled_py -m ensurepip --upgrade || die "ensurepip failed in $PY_ROOT"
    fi

    # pip writes this build tree's path into the shebang of every console
    # script it installs. Nothing here rewrites them: this tree is run from
    # where it sits. bundle.sh rewrites them in the COPY it ships, and its
    # build-tree scan fails the build if that step is ever dropped.
    printf 'Installing %d packages from warmup.py...\n' "${#PACKAGES[@]}"
    bundled_py -m pip install "${PIP_SOURCE_ARGS[@]}" --upgrade pip || die \
        "pip upgrade failed for the bundled interpreter"
    build_source_wheels "$PY_BUNDLED"
    bundled_py -m pip install "${PIP_SOURCE_ARGS[@]}" --only-binary=:all: "${PACKAGES[@]}" || die \
        "pip install failed for the bundled interpreter" \
        "Every package must install as a wheel. One with no wheel for this host's" \
        "glibc raises the floor of what the distribution runs on, which is a" \
        "decision to make, not something to compile around here."
    verify_bundled_python || die \
        "the bundled interpreter is incomplete after the install" \
        "pip can report success overall and still leave a package out."
    PY_BUNDLED_STATUS="installed at $PY_ROOT"
fi

# ---------------------------------------------------------------------------
step "ffmpeg"
# ---------------------------------------------------------------------------

# The slim ffmpeg's run-time dependencies, checked on the binary. The script
# that builds it makes the same check; this repeats it for a tree built earlier.
verify_ffmpeg() {
    local lib
    "$FFMPEG_DIR/ffmpeg" -hide_banner -version >/dev/null 2>&1 || {
        printf '%s does not run\n' "$FFMPEG_DIR/ffmpeg" >&2
        return 1
    }
    while IFS= read -r lib; do
        [ -n "$lib" ] || continue
        case "$lib" in
            libc.so.* | libm.so.* | libpthread.so.* | libdl.so.* | librt.so.* | "${PPF_LINUX_LOADER%%.so.*}".so.*) ;;
            *) printf 'ffmpeg needs %s at run time\n' "$lib" >&2; return 1 ;;
        esac
    done < <(readelf -d "$FFMPEG_DIR/ffmpeg" | awk '/\(NEEDED\)/ { sub(/.*\[/, ""); sub(/\].*/, ""); print }')
    [ -s "$FFMPEG_DIR/ffmpeg.sources.txt" ] || {
        printf 'no ffmpeg.sources.txt beside the binary\n' >&2
        return 1
    }
    printf 'ffmpeg: %s\n' "$("$FFMPEG_DIR/ffmpeg" -hide_banner -version | head -1)"
}

FFMPEG_REPAIR="rm -rf $FFMPEG_DIR && $BUILD_LINUX/warmup.sh"
if [ "${PPF_LINUX_FFMPEG:-1}" = "0" ]; then
    FFMPEG_STATUS="skipped this run, PPF_LINUX_FFMPEG=0"
    printf 'PPF_LINUX_FFMPEG=0, so no ffmpeg is built. bundle.sh refuses to\n'
    printf 'package without one, and names this switch when it does.\n'
elif [ "$NEED_FFMPEG" -eq 0 ]; then
    verify_ffmpeg || die "the ffmpeg at $FFMPEG_DIR does not verify" "Repair with:  $FFMPEG_REPAIR"
    FFMPEG_STATUS="verified at $FFMPEG_DIR"
else
    # nasm first, which x264 assembles its x86 code with. An aarch64 build needs
    # none: x264's aarch64 assembly goes through the C compiler.
    NASM_PATH=""
    if [ "$PPF_LINUX_ARCH" != x86_64 ]; then
        printf 'nasm: not needed on %s\n' "$PPF_LINUX_ARCH"
    elif host_nasm_ok; then
        printf 'nasm: %s (%s)\n' "$(command -v nasm)" "$(nasm -v)"
    else
        if [ ! -e "$NASM_DIR" ]; then
            printf 'No nasm 2.13 or newer on this host. Building one into %s\n' "$NASM_DIR"
            fetch NASM
            NASM_BUILD="$BUILD_LINUX/.nasm-build"
            rm -rf "$NASM_BUILD"
            mkdir -p "$NASM_BUILD"
            tar -xJf "$DOWNLOADS/$FILE_NASM" -C "$NASM_BUILD" --strip-components=1 || die \
                "tar could not extract $DOWNLOADS/$FILE_NASM"
            (
                cd "$NASM_BUILD"
                ./configure --prefix="$NASM_DIR.part"
                make -j"$(nproc)"
                make install
            ) || { rm -rf "$NASM_DIR.part"; die "building nasm failed; its output is above"; }
            mv "$NASM_DIR.part" "$NASM_DIR"
            rm -rf "$NASM_BUILD"
        fi
        "$NASM_DIR/bin/nasm" -v || die "the nasm at $NASM_DIR does not run" \
            "Repair with:  rm -rf $NASM_DIR && $BUILD_LINUX/warmup.sh"
        NASM_PATH="$NASM_DIR/bin"
    fi

    # The sources, kept in downloads/src so a later run on this host, or on one
    # the directory is relayed to, builds with no network. Each clone lands
    # under a .part name first, so an interrupted clone is not taken for one.
    mkdir -p "$FFMPEG_SOURCES"
    if [ ! -d "$FFMPEG_SOURCES/x264/.git" ]; then
        rm -rf "$FFMPEG_SOURCES/x264.part"
        git clone -q "$URL_X264_GIT" "$FFMPEG_SOURCES/x264.part" || die "could not clone $URL_X264_GIT"
        mv "$FFMPEG_SOURCES/x264.part" "$FFMPEG_SOURCES/x264"
    fi
    if [ ! -d "$FFMPEG_SOURCES/ffmpeg-$FFMPEG_TAG/.git" ]; then
        rm -rf "$FFMPEG_SOURCES/ffmpeg-$FFMPEG_TAG.part"
        git clone -q --depth 1 --branch "$FFMPEG_TAG" "$URL_FFMPEG_GIT" \
            "$FFMPEG_SOURCES/ffmpeg-$FFMPEG_TAG.part" || die "could not clone $URL_FFMPEG_GIT at $FFMPEG_TAG"
        mv "$FFMPEG_SOURCES/ffmpeg-$FFMPEG_TAG.part" "$FFMPEG_SOURCES/ffmpeg-$FFMPEG_TAG"
    fi
    if ! zlib_cached_ok; then
        rm -f "$FFMPEG_SOURCES/$FILE_ZLIB"
        curl -fL --retry 3 --retry-delay 2 --connect-timeout 15 \
            -o "$FFMPEG_SOURCES/$FILE_ZLIB.part" "$URL_ZLIB" || {
            rm -f "$FFMPEG_SOURCES/$FILE_ZLIB.part"
            die "could not download $URL_ZLIB"
        }
        mv "$FFMPEG_SOURCES/$FILE_ZLIB.part" "$FFMPEG_SOURCES/$FILE_ZLIB"
        zlib_cached_ok || die "checksum mismatch on $FFMPEG_SOURCES/$FILE_ZLIB" \
            "Delete it and re-run:  rm -f $FFMPEG_SOURCES/$FILE_ZLIB"
    fi

    rm -rf "$FFMPEG_DIR.part"
    PATH="${NASM_PATH:+$NASM_PATH:}$PATH" \
        FFMPEG_INSTALL_DIR="$FFMPEG_DIR.part" \
        FFMPEG_WORK_DIR="$BUILD_LINUX/.ffmpeg-work" \
        FFMPEG_SOURCE_DIR="$FFMPEG_SOURCES" \
        "$SRC_DIR/.github/workflows/scripts/make-slim-ffmpeg.sh" || {
        rm -rf "$FFMPEG_DIR.part"
        die "building the slim ffmpeg failed; its output is above"
    }
    mv "$FFMPEG_DIR.part" "$FFMPEG_DIR"
    verify_ffmpeg || die "the ffmpeg just built does not verify" "Repair with:  $FFMPEG_REPAIR"
    FFMPEG_STATUS="built at $FFMPEG_DIR"
fi

# ---------------------------------------------------------------------------
step "Setup complete"
# ---------------------------------------------------------------------------
printf 'Architecture:         %s, backends: %s\n' "$PPF_LINUX_ARCH" "$PPF_LINUX_BACKENDS"
printf 'CUDA toolkit:         %s\n' "$CUDA_STATUS"
printf 'ROCm SDK:             %s\n' "$ROCM_STATUS"
printf 'Source-built wheels:  %s\n' "${PPF_LINUX_SOURCE_WHEELS:-none needed on $PPF_LINUX_ARCH}"
printf 'Frontend environment: %s\n' "$PPF_CTS_VENV"
printf 'Bundled interpreter:  %s\n' "$PY_BUNDLED_STATUS"
printf 'ffmpeg:               %s\n' "$FFMPEG_STATUS"
printf 'Next: %s/build.sh\n' "$BUILD_LINUX"
