#!/usr/bin/env bash
# File: .github/workflows/scripts/linux/launch-verify.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Runs ON the verification instance, in $HOME, where release.yml copied the
# archive, verify-distribution.sh and the scene harness it runs:
#
#     launch-verify.sh ARCHIVE DIST_NAME CUDA_SCENES [GPU_BACKEND]
#
# It unpacks the distribution the way a user would, starts the verification
# DETACHED so it outlives the ssh that started it, and returns. The verification
# writes ~/verify/exit.txt when it ends, whatever the outcome, which is what
# poll-verify.sh reads.
#
# GPU_BACKEND NAMES WHICH BACKEND THIS MACHINE RUNS THE SCENES ON, for a
# distribution that carries more than one: this instance has an NVIDIA GPU, so
# it is `cuda` there. Left out, verify-distribution.sh takes the only GPU
# backend a distribution stamps and refuses to guess between two.

set -euo pipefail

ARCHIVE="$1"
DIST_NAME="$2"
CUDA_SCENES="$3"
GPU_BACKEND="${4:-}"
WORK="$HOME/verify"

rm -rf "$WORK"
mkdir -p "$WORK"
tar -xzf "$HOME/$ARCHIVE" -C "$WORK"
rm -f "$HOME/$ARCHIVE"
[ -x "$WORK/$DIST_NAME/ppf-contact-solver" ] || {
    echo "ERROR: the archive did not unpack to $WORK/$DIST_NAME with its launcher" >&2
    exit 1
}
mv "$HOME/verify-distribution.sh" "$WORK/verify-distribution.sh"
# run_suite.py IS NOT IN THE ARCHIVE, so it arrives beside the verifier and has
# to stay beside it: verify-distribution.sh resolves the harness from its own
# directory, which is $WORK once the move above has happened.
mv "$HOME/run_suite.py" "$WORK/run_suite.py"

# shellcheck disable=SC2016 # the inner script takes its values as arguments
setsid nohup bash -c '
    backend=()
    [ -z "$4" ] || backend=(--gpu-backend "$4")
    bash "$1/verify-distribution.sh" "$1/$2" "$1/report" --driver-only --cuda-scenes "$3" \
        "${backend[@]}" > "$1/verify.log" 2>&1
    echo $? > "$1/exit.txt"
' _ "$WORK" "$DIST_NAME" "$CUDA_SCENES" "$GPU_BACKEND" > /dev/null 2>&1 < /dev/null &
printf 'verification started for %s (GPU backend: %s, scenes: %s)\n' \
    "$DIST_NAME" "${GPU_BACKEND:-the only one stamped}" "$CUDA_SCENES"
