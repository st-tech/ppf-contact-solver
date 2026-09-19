#!/usr/bin/env bash
# File: build-linux-native/scripts/load-downloads.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Reads scripts/downloads.txt and exports every KEY=VALUE pair into the
# caller's environment. Source it, do not execute it:
#
#     . "$BUILD_LINUX/scripts/load-downloads.sh"
#     load_downloads "$BUILD_LINUX/scripts/downloads.txt"
#
# After the call the URL_* / FILE_* / SHA256_* variables are exported, and
# PPF_DOWNLOAD_KEYS holds the key names in file order so a caller can iterate
# them without parsing the manifest a second time.
#
# This parses rather than sourcing the manifest. Sourcing would run whatever
# the file contains, and would accept a malformed line as a no-op; the parser
# rejects anything that is not blank, a # comment, or KEY=VALUE with an
# upper-case key, and names the file and line number when it does. It accepts
# the format build-win-native/scripts/downloads.txt documents, so it reads that
# manifest too.
#
# No `set -euo pipefail` here: this file is sourced, and those options would
# leak into the caller's shell. Every function returns non-zero on failure and
# every caller checks.

# PPF_DOWNLOAD_KEYS is reset by each call, so two manifests in one shell do not
# accumulate into each other.
PPF_DOWNLOAD_KEYS=""

load_downloads() {
    local manifest="$1"
    local line key value lineno

    if [ ! -f "$manifest" ]; then
        printf 'ERROR: download manifest not found: %s\n' "$manifest" >&2
        return 1
    fi

    PPF_DOWNLOAD_KEYS=""
    lineno=0
    # `|| [ -n "$line" ]` so a final line with no trailing newline is still
    # read rather than silently dropped.
    while IFS= read -r line || [ -n "$line" ]; do
        lineno=$((lineno + 1))
        case "$line" in
            '' | '#'*) continue ;;
        esac
        if [ "${line#*=}" = "$line" ]; then
            printf 'ERROR: %s:%d is neither blank, a # comment, nor KEY=VALUE: %s\n' \
                "$manifest" "$lineno" "$line" >&2
            return 1
        fi
        key="${line%%=*}"
        value="${line#*=}"
        if ! [[ "$key" =~ ^[A-Z][A-Z0-9_]*$ ]]; then
            printf 'ERROR: %s:%d has a key that is not an upper-case identifier: %s\n' \
                "$manifest" "$lineno" "$key" >&2
            return 1
        fi
        if [ -z "$value" ]; then
            printf 'ERROR: %s:%d gives %s an empty value\n' \
                "$manifest" "$lineno" "$key" >&2
            return 1
        fi
        export "$key=$value"
        PPF_DOWNLOAD_KEYS="$PPF_DOWNLOAD_KEYS $key"
    done < "$manifest"

    if [ -z "$PPF_DOWNLOAD_KEYS" ]; then
        printf 'ERROR: %s defines no entries\n' "$manifest" >&2
        return 1
    fi
    return 0
}

# Print the value of one key from a manifest, without exporting anything else.
# This is how the Linux build reads the ffmpeg entries out of the Windows
# manifest: that file also defines URL_PYTHON and URL_RUSTUP for Windows, and
# loading it whole would put those in the caller's environment beside the Linux
# ones of the same name.
manifest_value() {
    local manifest="$1" want="$2" line key found=""
    if [ ! -f "$manifest" ]; then
        printf 'ERROR: download manifest not found: %s\n' "$manifest" >&2
        return 1
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            '' | '#'*) continue ;;
        esac
        key="${line%%=*}"
        if [ "$key" = "$want" ]; then
            found="${line#*=}"
        fi
    done < "$manifest"
    if [ -z "$found" ]; then
        printf 'ERROR: %s does not set %s\n' "$manifest" "$want" >&2
        return 1
    fi
    printf '%s\n' "$found"
}
