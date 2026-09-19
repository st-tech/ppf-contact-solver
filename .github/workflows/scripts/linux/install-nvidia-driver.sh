#!/usr/bin/env bash
# File: .github/workflows/scripts/linux/install-nvidia-driver.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Installs the NVIDIA DRIVER and nothing else on a plain Ubuntu instance, for
# release.yml's verification of the Linux distribution:
#
#     install-nvidia-driver.sh [MIN_BRANCH]      (default 570)
#
# WHAT IT INSTALLS IS WHAT A USER OF THE DISTRIBUTION IS ASSUMED TO HAVE: the
# kernel module, the user-space compute library that carries libcuda.so.1, and
# nvidia-smi, all from ONE server driver branch. No CUDA toolkit, no cuda-drivers
# meta package, no libcudart.
#
# THE BRANCH IS THE LOWEST ONE AT OR ABOVE MIN_BRANCH THAT UBUNTU PUBLISHES A
# PREBUILT MODULE FOR ON THE RUNNING KERNEL. 570 is the oldest branch the CUDA
# 12.8 runtime supports, so it is the default minimum, but the archive does not
# build every branch for every kernel. Measured on Ubuntu 24.04 with a 7.0 AWS
# kernel: only the 535, 580 and 595 server branches carried modules, and asking
# ubuntu-drivers for 570 there installed a 580 user space with no module at all,
# which nvidia-smi then could not reach. A prebuilt module needs no compiler and
# matches the running kernel. When no branch qualifies this fails by name rather
# than installing something else.

set -euo pipefail

MIN_BRANCH="${1:-570}"
case "$MIN_BRANCH" in
    '' | *[!0-9]*) echo "ERROR: MIN_BRANCH must be a driver branch number, got '$MIN_BRANCH'" >&2; exit 2 ;;
esac
export DEBIAN_FRONTEND=noninteractive
APT=(sudo -E apt-get -y -q -o DPkg::Lock::Timeout=900)

# cloud-init and unattended-upgrades hold the package lock for the first
# minutes of an instance's life; the lock timeout above covers the second.
cloud-init status --wait >/dev/null 2>&1 || true

"${APT[@]}" update
KERNEL="$(uname -r)"

# Every server branch with a prebuilt module for exactly this kernel. The search
# pattern is a regular expression, so the kernel is compared as a string after.
AVAILABLE="$(apt-cache search --names-only '^linux-modules-nvidia-[0-9]+-server-' \
    | awk '{ print $1 }' \
    | sed -nE 's/^linux-modules-nvidia-([0-9]+)-server-(.*)$/\1 \2/p' \
    | awk -v kernel="$KERNEL" '$2 == kernel { print $1 }' \
    | sort -n -u)"
printf 'Kernel %s. Server branches with a prebuilt module: %s\n' \
    "$KERNEL" "$(printf '%s' "$AVAILABLE" | tr '\n' ' ')"

BRANCH=""
for candidate in $AVAILABLE; do
    if [ "$candidate" -ge "$MIN_BRANCH" ]; then
        BRANCH="$candidate"
        break
    fi
done
if [ -z "$BRANCH" ]; then
    echo "ERROR: no NVIDIA server driver branch at or above $MIN_BRANCH has a prebuilt" >&2
    echo "       module for kernel $KERNEL in this image's archive." >&2
    exit 1
fi
printf 'Installing the %s-server driver, the lowest branch at or above %s for this kernel.\n' \
    "$BRANCH" "$MIN_BRANCH"
"${APT[@]}" install --no-install-recommends \
    "linux-modules-nvidia-${BRANCH}-server-${KERNEL}" \
    "nvidia-headless-no-dkms-${BRANCH}-server" \
    "nvidia-utils-${BRANCH}-server"

# The assertions this workflow exists to keep true: a driver, no toolkit, and
# one branch, since a user space and a module from different branches do not
# talk to each other.
INSTALLED="$(dpkg -l | awk '$1 == "ii" { print $2 }')"
if printf '%s\n' "$INSTALLED" | grep -E '^(cuda-|nvidia-cuda-toolkit|libcudart)'; then
    echo "ERROR: a CUDA toolkit package was installed alongside the driver" >&2
    exit 1
fi
OTHER="$(printf '%s\n' "$INSTALLED" \
    | grep -E '^(nvidia-utils|libnvidia-compute|nvidia-headless-no-dkms|linux-modules-nvidia)-[0-9]+' \
    | grep -v -F -- "-${BRANCH}-server" || true)"
if [ -n "$OTHER" ]; then
    echo "ERROR: packages of another driver branch are installed beside ${BRANCH}-server:" >&2
    printf '%s\n' "$OTHER" >&2
    exit 1
fi

# A module built for the running kernel loads now. The workflow still reboots
# and asks nvidia-smi again, and that later check is the one that fails the job.
if sudo modprobe nvidia && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; then
    printf 'The %s-server driver is loaded.\n' "$BRANCH"
else
    printf 'The %s-server module did not load before a reboot; the check after the reboot decides.\n' "$BRANCH"
fi
