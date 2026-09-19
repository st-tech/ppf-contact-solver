#!/usr/bin/env bash
# File: .github/workflows/scripts/linux/launch-build.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Runs ON the Linux build instance: unpacks the repository the runner sent,
# then starts `container-build.sh` inside the build image DETACHED and
# returns at once.
#
#     bash launch-build.sh MOUNT IMAGE DIST_NAME
#
# Needs `repo.tar.gz` in $HOME. Leaves the container's whole output in
# ~/build/build.log and its exit code in ~/build/exit.txt, which appears only
# when the build is done; `poll-build.sh` reads both. The checkout is placed
# at MOUNT, the same distinctive path the runner used, because bundle.sh
# searches every shipped file for that path as text.
#
# WHY DETACHED. The runner reaches this instance through an
# ec2-instance-connect websocket with a one hour session cap, and the
# credential that opens it lasts an hour. The build is well inside that on
# this instance class, but a connection that dies mid-build would take the
# build with it; detached, the build's lifetime is its own, and the runner
# polls.
set -euo pipefail
MOUNT="$1"
IMAGE="$2"
DIST_NAME="$3"
WORK="$HOME/build"

rm -rf "$WORK"
mkdir -p "$WORK"
sudo rm -rf "$MOUNT"
sudo mkdir -p "$MOUNT"
sudo chown "$(id -u):$(id -g)" "$MOUNT"
tar -xzf "$HOME/repo.tar.gz" -C "$MOUNT"
rm -f "$HOME/repo.tar.gz"
[ -f "$MOUNT/.github/workflows/scripts/linux/container-build.sh" ] || {
    echo "ERROR: the repository did not unpack to $MOUNT with container-build.sh" >&2
    exit 1
}

# The same invocation release.yml ran on the GitHub-hosted runner, with the
# same environment; HOST_UID/HOST_GID hand the checkout back to this user.
setsid nohup bash -c '
    docker run --rm \
        -v "$1:$1" \
        -e BUILD_MOUNT="$1" \
        -e PPF_LINUX_DIST_NAME="$3" \
        -e PPF_LINUX_DROP_ROCM_ARCHIVE=1 \
        -e PPF_LINUX_MAX_GLIBC=2.28 \
        -e PPF_LINUX_MAX_GLIBCXX=3.4.25 \
        -e HOST_UID="$(id -u)" \
        -e HOST_GID="$(id -g)" \
        "$2" bash "$1/.github/workflows/scripts/linux/container-build.sh" \
        > "$4/build.log" 2>&1
    echo $? > "$4/exit.txt"
' _ "$MOUNT" "$IMAGE" "$DIST_NAME" "$WORK" > /dev/null 2>&1 < /dev/null &
printf 'build started for %s in %s at %s\n' "$DIST_NAME" "$IMAGE" "$MOUNT"
