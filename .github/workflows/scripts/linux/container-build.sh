#!/usr/bin/env bash
# File: .github/workflows/scripts/linux/container-build.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Runs INSIDE the AlmaLinux 8 container release.yml starts, with the
# checkout mounted at $BUILD_MOUNT. It installs the build host's tools and then runs the
# three scripts a developer runs, unchanged:
#
#     build-linux-native/warmup.sh  build.sh  bundle.sh
#
# WHY THIS CONTAINER. What a Linux binary needs from the system is decided by
# the glibc it was linked against, so the release is linked against glibc 2.28,
# the C library of RHEL/AlmaLinux/Rocky 8, and runs there and on anything newer.
# bundle.sh reads the floor off the payload and refuses one above
# PPF_LINUX_MAX_GLIBC and PPF_LINUX_MAX_GLIBCXX, which the workflow sets, so a
# dependency that raises either fails here by name.
#
# WHY gcc-toolset-13. CUDA 12.8's nvcc accepts GCC up to 14, and the system GCC 8
# is too old for the solver's C++. A gcc-toolset compiler links the libstdc++
# symbols newer than the system's statically into each binary, which is what
# keeps the GLIBCXX requirement at the system's 3.4.25 while compiling C++20.
#
# Environment, set by the workflow: PPF_LINUX_DIST_NAME, PPF_LINUX_MAX_GLIBC,
# PPF_LINUX_MAX_GLIBCXX, and HOST_UID / HOST_GID, which the checkout is handed
# back to on exit so the runner can clean its workspace.

set -euo pipefail

# The workflow names the mount point; bundle.sh refuses one too short to search
# for, such as /src.
SRC="${BUILD_MOUNT:?BUILD_MOUNT names where the checkout is mounted}"
LOGS="$SRC/ci-logs"
mkdir -p "$LOGS"

give_back() {
    if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
        chown -R "$HOST_UID:$HOST_GID" "$SRC" || true
    fi
}
trap give_back EXIT

printf 'Container: %s, %s\n' "$(. /etc/os-release && printf '%s' "$PRETTY_NAME")" "$(getconf GNU_LIBC_VERSION)"
case "$(getconf GNU_LIBC_VERSION)" in
    "glibc 2.28") ;;
    *) echo "ERROR: the release container is expected to carry glibc 2.28" >&2; exit 1 ;;
esac

# python3.12-devel carries the headers the developer environment's interpreter
# compiles against. On aarch64 warmup.sh builds a wheel from source for each
# frontend package that publishes no aarch64 wheel
# (build-linux-native/scripts/source-wheel.sh), and it refuses without them.
dnf -y -q install \
    gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ gcc-toolset-13-binutils \
    make git rsync xz tar gzip bzip2 curl which diffutils findutils file procps-ng \
    pkgconf-pkg-config python3.12 python3.12-pip python3.12-devel

# nasm FROM POWERTOOLS, ON x86_64 ONLY. warmup.sh takes a host nasm of 2.13 or
# newer for x264's x86 assembly and only otherwise fetches one from nasm.us,
# which the CI network refuses (Build Release #21, 2026-09-17: the pointer
# check timed out on it and the x86_64 build ended before compiling anything).
# Measured in this image on dev-head with the firewall open: AlmaLinux 8's nasm
# (2.15.03) lives in the PowerTools repository, disabled by default, and the
# enabled repositories answer `Unable to find a match: nasm`; the banner
# `NASM version 2.15.03 compiled on May 20 2021` satisfies host_nasm_ok, so
# URL_NASM is never probed here. nasm is an x86 assembler with no aarch64
# package (Build Release #22 lost its aarch64 leg to naming it for both), and
# x264's aarch64 code goes through the C compiler, so aarch64 installs nothing.
if [ "$(uname -m)" = x86_64 ]; then
    dnf -y -q --enablerepo=powertools install nasm
    printf 'nasm: %s\n' "$(nasm -v)"
fi

# The enable script reads variables it does not define.
set +u
# shellcheck source=/dev/null
. /opt/rh/gcc-toolset-13/enable
set -u
printf 'Compiler: %s\n' "$(g++ --version | head -1)"

# bundle.sh runs its ELF audit with python3, and the solver's build script runs
# one helper with it; the system python3 of this image is 3.6.
ln -sf /usr/bin/python3.12 /usr/local/bin/python3
hash -r
printf 'python3: %s\n' "$(python3 --version)"

# The checkout belongs to the runner's user and this container runs as root.
git config --global --add safe.directory '*'

export PPF_CTS_PYTHON=/usr/bin/python3.12
cd "$SRC/build-linux-native"
./warmup.sh 2>&1 | tee "$LOGS/warmup.log"
./build.sh 2>&1 | tee "$LOGS/build.log"
./bundle.sh 2>&1 | tee "$LOGS/bundle.log"
