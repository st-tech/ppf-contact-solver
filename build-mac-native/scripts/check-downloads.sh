#!/usr/bin/env bash
# File: build-mac-native/scripts/check-downloads.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Verifies every URL_* in scripts/downloads.txt is reachable. warmup.sh runs
# this BEFORE its first download and treats a failure as fatal, so a pointer
# that has rotted upstream costs the seconds this script takes instead of
# failing part way through a provision. Run it by hand after editing the
# manifest; if a URL fails, fix that entry and re-run.
#
# Exit status is 0 when every URL answered and 1 when any did not. There is no
# flag to downgrade a failure to a warning, and the timeouts below are not
# there to be relaxed until a run passes: an unreachable pointer is a real
# result about the state of the world, and proceeding past it wastes the whole
# provision instead of the probe.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${1:-$SCRIPT_DIR/downloads.txt}"

# shellcheck source=build-mac-native/scripts/load-downloads.sh
. "$SCRIPT_DIR/load-downloads.sh"

if ! command -v curl >/dev/null 2>&1; then
    printf 'ERROR: curl not found on PATH. It ships with the Xcode command line tools;\n' >&2
    printf '       install them with  xcode-select --install\n' >&2
    exit 1
fi

load_downloads "$MANIFEST"

# Retry policy, matching the Windows probe. A single attempt is not a sound
# reachability verdict: a rotted pointer and a dropped SYN look identical from
# one sample, and this verdict is fatal before the first download, so one
# transient drop would discard a whole provision. --connect-timeout bounds each
# attempt, which --max-time never reaches when no connection is established at
# all. --retry covers that timeout class and --retry-connrefused adds the RST a
# rate-limiting host sends. Neither retries an HTTP 4xx, so a genuinely dead
# pointer still fails on its first attempt rather than being retried into a
# slow abort.
RETRY_ARGS=(--retry 2 --retry-delay 1 --retry-connrefused --connect-timeout 8 --max-time 30)

probe() {
    # $1 = manifest key, $2 = url
    local name="$1" url="$2" rc=0 head_out=""
    printf 'Checking %s\n' "$name"
    # HEAD first. Some hosts (S3, signed URLs, certain CDNs) reject HEAD, so
    # fall back to a one-byte ranged GET, which pulls almost no data. The
    # HEAD's -w line is held back until it succeeds: curl writes it whatever
    # happened, and an "HTTP 000" printed from a failed attempt reads like a
    # verdict when it is only the first of two probes.
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
        # Report curl's own exit code, not just "unreachable". It is the only
        # thing that separates a dead pointer (22, an HTTP 4xx) from a
        # transport failure (28 timeout, 7 refused, 6 DNS), and without it the
        # verdict has to be diagnosed by inference from the surrounding log.
        printf '  [FAIL] %s unreachable (curl exit %d): %s\n' "$name" "$rc" "$url" >&2
        return 1
    fi
    return 0
}

printf '=== Checking download URLs in %s ===\n\n' "$MANIFEST"

has_error=0
checked=0
url=""
for key in $PPF_DOWNLOAD_KEYS; do
    case "$key" in
        URL_*) ;;
        *) continue ;;
    esac
    checked=$((checked + 1))
    # Indirect expansion, so the loop reads whatever the manifest defined.
    eval "url=\${$key}"
    if ! probe "$key" "$url"; then
        has_error=1
    fi
done

printf '\n'
if [ "$checked" -eq 0 ]; then
    # A manifest with no URL_* entries would pass vacuously, which reads as a
    # clean check and certifies nothing.
    printf '=== [FAIL] %s defines no URL_* entries, so nothing was checked ===\n' "$MANIFEST" >&2
    exit 1
fi
if [ "$has_error" -ne 0 ]; then
    printf '=== [FAIL] One or more URLs are not reachable ===\n' >&2
    printf 'Update the offending entries in %s and re-run.\n' "$MANIFEST" >&2
    exit 1
fi
printf '=== [OK] All %d download URLs reachable ===\n' "$checked"
