#!/usr/bin/env bash
# File: .github/workflows/scripts/linux/poll-build.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Runs ON the Linux build instance: reports the detached build's state in
# one line, `STATE|<exit code or RUNNING>|<newest output line>`, the shape
# `wait-verify.sh` reads. The progress field is the container's newest line,
# which is what proves the build advanced: cargo and make write steadily, so
# a line that does not change between polls is a stall, and a live process
# is not evidence of anything.
WORK="$HOME/build"
if [ -f "$WORK/exit.txt" ]; then
    code="$(tr -d '[:space:]' < "$WORK/exit.txt")"
else
    code=RUNNING
fi
progress="$(tail -n 1 "$WORK/build.log" 2>/dev/null | tr -d '\r' | cut -c1-160)"
printf 'STATE|%s|%s\n' "${code:-1}" "${progress:-no output yet}"
