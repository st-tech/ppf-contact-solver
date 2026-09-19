#!/usr/bin/env bash
# File: build-linux-native/scripts/check-downloads.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Verifies URL_* entries of a download manifest are reachable. warmup.sh runs
# this BEFORE its first download and treats a failure as fatal, so a pointer
# that has rotted upstream costs the seconds this script takes instead of
# failing part way through a provision.
#
#     scripts/check-downloads.sh                        every URL_* in downloads.txt
#     scripts/check-downloads.sh URL_PYTHON_X86_64 URL_NASM    only the named entries
#     scripts/check-downloads.sh --manifest FILE [KEY...]
#
# Naming entries is how warmup.sh probes only what it is about to fetch: an
# entry whose file is already in downloads/ and matches its checksum needs no
# network, and a host that cannot reach the hosts at all can still provision
# from a relayed downloads/ directory.
#
# Exit status is 0 when every probed URL answered and 1 when any did not. There
# is no flag to downgrade a failure to a warning, and the timeouts below are not
# there to be relaxed until a run passes: an unreachable pointer is a real result
# about the state of the world.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/downloads.txt"
if [ "${1:-}" = "--manifest" ]; then
    [ "$#" -ge 2 ] || { printf 'ERROR: --manifest needs a file\n' >&2; exit 1; }
    MANIFEST="$2"
    shift 2
fi

# shellcheck source=build-linux-native/scripts/load-downloads.sh
. "$SCRIPT_DIR/load-downloads.sh"

if ! command -v curl >/dev/null 2>&1; then
    printf 'ERROR: curl not found on PATH. Install it with the system package manager.\n' >&2
    exit 1
fi

load_downloads "$MANIFEST"

# Retry policy, matching the macOS and Windows probes. A single attempt is not a
# sound reachability verdict: a rotted pointer and a dropped SYN look identical
# from one sample. --connect-timeout bounds each attempt, which --max-time never
# reaches when no connection is established at all. Neither retry flag retries
# an HTTP 4xx, so a genuinely dead pointer still fails on its first attempt.
RETRY_ARGS=(--retry 2 --retry-delay 1 --retry-connrefused --connect-timeout 8 --max-time 30)

probe() {
    # $1 = manifest key, $2 = url
    local name="$1" url="$2" rc=0 head_out=""
    printf 'Checking %s\n' "$name"
    # HEAD first; some hosts reject HEAD, so fall back to a one-byte ranged GET.
    # The HEAD's -w line is held back until it succeeds, so an "HTTP 000" from a
    # failed first attempt is not printed as if it were the verdict.
    if head_out="$(curl -fsSLI "${RETRY_ARGS[@]}" -o /dev/null \
        -w "  HTTP %{http_code}  $url"$'\n' "$url" 2>/dev/null)"; then
        printf '%s' "$head_out"
        return 0
    fi
    set +e
    curl -fsSL "${RETRY_ARGS[@]}" -r 0-0 -o /dev/null \
        -w "  HTTP %{http_code}  $url"$'\n' "$url"
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        # curl's own exit code separates a dead pointer (22, an HTTP 4xx) from a
        # transport failure (28 timeout, 7 refused, 6 DNS).
        printf '  [FAIL] %s unreachable (curl exit %d): %s\n' "$name" "$rc" "$url" >&2
        return 1
    fi
    return 0
}

if [ "$#" -gt 0 ]; then
    KEYS="$*"
else
    KEYS=""
    for key in $PPF_DOWNLOAD_KEYS; do
        case "$key" in URL_*) KEYS="$KEYS $key" ;; esac
    done
fi

printf '=== Checking download URLs in %s ===\n\n' "$MANIFEST"

has_error=0
checked=0
url=""
for key in $KEYS; do
    case "$key" in
        URL_*) ;;
        *)
            printf '  [FAIL] %s is not a URL_* key\n' "$key" >&2
            has_error=1
            continue
            ;;
    esac
    case " $PPF_DOWNLOAD_KEYS " in
        *" $key "*) ;;
        *)
            printf '  [FAIL] %s is not defined in %s\n' "$key" "$MANIFEST" >&2
            has_error=1
            continue
            ;;
    esac
    checked=$((checked + 1))
    url="${!key}"
    if ! probe "$key" "$url"; then
        has_error=1
    fi
done

printf '\n'
if [ "$checked" -eq 0 ] && [ "$has_error" -eq 0 ]; then
    # Nothing to probe would pass vacuously, which reads as a clean check and
    # certifies nothing. warmup.sh does not call this when it needs no network.
    printf '=== [FAIL] no URL_* entries were named or defined, so nothing was checked ===\n' >&2
    exit 1
fi
if [ "$has_error" -ne 0 ]; then
    printf '=== [FAIL] One or more URLs are not reachable ===\n' >&2
    printf 'Update the offending entries in %s and re-run.\n' "$MANIFEST" >&2
    exit 1
fi
printf '=== [OK] All %d download URLs reachable ===\n' "$checked"
