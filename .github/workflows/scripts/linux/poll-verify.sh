#!/usr/bin/env bash
# File: .github/workflows/scripts/linux/poll-verify.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Runs ON the verification instance and prints one line:
#
#     STATE|<exit code, or RUNNING>|<progress>
#
# The progress names the check in flight and the last line of its log, which is
# what shows a wedged run: the same line printed poll after poll.

WORK="$HOME/verify"
if [ -f "$WORK/exit.txt" ]; then
    code="$(tr -d '[:space:]' < "$WORK/exit.txt")"
else
    code=RUNNING
fi
newest="$(find "$WORK/report" -maxdepth 1 -name '*.log' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [ -n "$newest" ]; then
    progress="$(basename "$newest" .log): $(tail -n 1 "$newest" | tr -d '\r' | cut -c1-160)"
else
    progress="no check has started"
fi
printf 'STATE|%s|%s\n' "${code:-1}" "$progress"
