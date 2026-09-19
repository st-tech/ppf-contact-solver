#!/usr/bin/env bash
# File: build-linux-native/scripts/verify-distribution.sh
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Verifies an unpacked Linux distribution ON THE MACHINE THAT RUNS IT, and writes
# a report. release.yml runs it on a GPU instance carrying only the NVIDIA
# driver and in an AlmaLinux 8 container; a developer runs it on a dev box.
#
#     verify-distribution.sh DIST REPORT_DIR [options]
#
#   --driver-only        also require that the machine has the GPU DRIVER and no
#                        SDK, in the terms of whichever backend the distribution
#                        stamps: for cuda, no nvcc on PATH, no libcudart in the
#                        loader cache, no /usr/local/cuda*, and an NVIDIA driver
#                        present; for rocm, no hipcc, no libamdhip64 in the
#                        cache, no /opt/rocm*, and a readable /dev/kfd. The ROCm
#                        arm has never been run: no machine here has an AMD GPU.
#   --cuda-scenes LIST   run these example notebooks on the GPU backend through
#                        run_suite.py --fast-check; `all` for the whole suite,
#                        `none` to skip (the default)
#   --cpu-scenes LIST    the same on the CPU backend (default `none`)
#
# What it checks is what bundle.sh cannot, because bundle.sh runs on the machine
# that built the distribution:
#   - the launcher answers --help naming its real path, and both solvers, both
#     servers and ffmpeg start with no LD_LIBRARY_PATH;
#   - the loader takes the distribution's own bin/libppfbe_<backend>.so;
#   - the bundled interpreter imports the frontend and JupyterLab through the
#     launcher's own environment (`ppf-contact-solver python`);
#   - the example scenes run, judged by run_suite.py on the frames they produce;
#   - every shared library a running GPU solver maps is inside the distribution
#     or a system library directory, no toolkit library is among them, and the
#     vendor's own driver library is, which is what shows a solve reached the
#     GPU (libcuda for NVIDIA, libhsakmt or libdrm_amdgpu for AMD);
#   - the distribution writes nothing under HOME, which is an empty directory for
#     the run, the GPU driver's own `~/.nv` compute cache excepted because the
#     driver writes it from inside libcuda and no setting here reaches it;
#   - nothing appears in the distribution outside local/, cache/ and examples/.
#
# Every check runs whatever the previous one did, so one report names every
# failure; the exit status is 0 only when all passed. Each check's full output
# is REPORT_DIR/<check>.log, and REPORT_DIR/report.txt is the summary.

set -uo pipefail

if [ "$#" -lt 2 ]; then
    printf 'usage: %s DIST REPORT_DIR [--driver-only] [--cuda-scenes LIST] [--cpu-scenes LIST] [--gpu-backend NAME]\n' "$0" >&2
    exit 2
fi
DIST="$(cd "$1" 2>/dev/null && pwd -P)" || { printf 'ERROR: no distribution at %s\n' "$1" >&2; exit 2; }
if ! mkdir -p "$2" || ! REPORT_DIR="$(cd "$2" && pwd -P)"; then
    printf 'ERROR: cannot use %s as the report directory\n' "$2" >&2
    exit 2
fi
shift 2

DRIVER_ONLY=0
CUDA_SCENES=none
# Which GPU backend the scene run uses, where the distribution carries more than
# one. Empty means "the only one there is", and a distribution with two of them
# and no choice named is a failure rather than a guess.
GPU_BACKEND_ARG=""
CPU_SCENES=none
while [ "$#" -gt 0 ]; do
    case "$1" in
        --driver-only) DRIVER_ONLY=1; shift ;;
        --cuda-scenes) CUDA_SCENES="${2:?--cuda-scenes needs a list}"; shift 2 ;;
        --cpu-scenes) CPU_SCENES="${2:?--cpu-scenes needs a list}"; shift 2 ;;
        --gpu-backend) GPU_BACKEND_ARG="${2:?--gpu-backend needs a backend name}"; shift 2 ;;
        *) printf 'ERROR: unknown option %s\n' "$1" >&2; exit 2 ;;
    esac
done

LAUNCHER="$DIST/ppf-contact-solver"
REPORT="$REPORT_DIR/report.txt"

# THE BACKENDS THIS DISTRIBUTION SHIPS, as bundle.sh stamped them into the
# launcher. Every check below asks for exactly those, so a CPU-only distribution
# is checked for what it carries and a missing CUDA part of one that ships CUDA
# still fails.
BACKENDS="$(sed -n 's/^PPF_DIST_BACKENDS=//p' "$LAUNCHER" 2>/dev/null | head -1 | tr ',' ' ')"
if [ -z "$BACKENDS" ]; then
    printf 'ERROR: %s carries no PPF_DIST_BACKENDS stamp\n' "$LAUNCHER" >&2
    exit 2
fi
has_backend() {
    case " $BACKENDS " in
        *" $1 "*) return 0 ;;
    esac
    return 1
}
# WHICH GPU BACKENDS THE STAMP NAMES. A distribution carries every backend it was
# built with, each in its own target/<backend>, so this is a list. Asking only
# about `cuda` once left a rocm,cpu distribution with RELEASE_DIRS naming the CPU
# directory alone, so every check over the GPU half was skipped and the run still
# reported PASS. That is a pass bought by the coverage vanishing, which is worse
# than a red suite, so an unknown backend stops this script rather than narrowing
# it.
GPU_BACKENDS=""
for candidate in cuda rocm; do
    has_backend "$candidate" || continue
    GPU_BACKENDS="${GPU_BACKENDS:+$GPU_BACKENDS }$candidate"
done
for name in $BACKENDS; do
    case "$name" in
        cpu | cuda | rocm) ;;
        *)
            printf 'ERROR: %s stamps backend "%s", which this script has no checks for\n' \
                "$LAUNCHER" "$name" >&2
            printf '       Add them rather than letting this run report a pass over a\n' >&2
            printf '       backend it did not verify.\n' >&2
            exit 1 ;;
    esac
done
# One release directory per stamped backend, in the stamp's own order.
RELEASE_DIRS=""
for name in $BACKENDS; do
    RELEASE_DIRS="${RELEASE_DIRS:+$RELEASE_DIRS }target/$name/release"
done

# WHICH GPU BACKEND THIS RUN EXERCISES, decided once, here, because both the
# driver check and the scene run ask about it. --gpu-backend names it; with one
# GPU backend stamped that is the answer; with several and no choice named there
# is no answer, and the checks that need one fail by name rather than guessing.
SUITE_BACKEND="$GPU_BACKEND_ARG"
if [ -z "$SUITE_BACKEND" ] && [ "$(printf '%s\n' $GPU_BACKENDS | grep -c .)" -eq 1 ]; then
    SUITE_BACKEND="$GPU_BACKENDS"
fi
if [ -n "$SUITE_BACKEND" ] && ! printf '%s\n' $GPU_BACKENDS | grep -qx "$SUITE_BACKEND"; then
    printf 'ERROR: --gpu-backend %s, and %s stamps %s\n' \
        "$SUITE_BACKEND" "$LAUNCHER" "$BACKENDS" >&2
    exit 2
fi
: > "$REPORT"
PASSED=0
FAILED=0

note() {
    printf '%s\n' "$*" | tee -a "$REPORT"
}

# check NAME FUNCTION: run it with its output in NAME.log and record the verdict.
check() {
    local name="$1" log="$REPORT_DIR/$1.log" start rc
    shift
    start="$(date +%s)"
    "$@" > "$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        PASSED=$((PASSED + 1))
        note "PASS  $name  ($(( $(date +%s) - start )) s)"
    else
        FAILED=$((FAILED + 1))
        note "FAIL  $name  (exit $rc, $(( $(date +%s) - start )) s)"
        tail -n 25 "$log" | sed 's/^/        /' | tee -a "$REPORT"
    fi
}

system_info() {
    uname -srm
    (. /etc/os-release 2>/dev/null && printf '%s\n' "${PRETTY_NAME:-unknown distribution}")
    getconf GNU_LIBC_VERSION
    printf 'cpus: %s\n' "$(nproc)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        printf 'no nvidia-smi on this machine\n'
    fi
    printf 'distribution: %s\n' "$DIST"
    printf 'backends: %s\n' "$BACKENDS"
    return 0
}

# THE ROCm ANALOGUE OF driver_only, asserting the same property: this machine has
# a DRIVER and no SDK, so a pass means the distribution ran on its own runtime
# rather than on one the host happened to provide.
#
# NOT EXERCISED. No machine available to this project has an AMD GPU, so this
# arm has never returned 0. It is written by direct analogy with the CUDA one
# and its first real run will be its first test; read a pass from it as a claim
# about the checks, not yet as evidence about a machine.
rocm_driver_only() {
    local bad=0
    if command -v hipcc >/dev/null 2>&1; then
        printf 'hipcc is on PATH: %s\n' "$(command -v hipcc)"
        bad=1
    fi
    if ldconfig -p 2>/dev/null | grep -q libamdhip64; then
        printf 'the loader cache holds a HIP runtime:\n'
        ldconfig -p | grep libamdhip64
        bad=1
    fi
    if compgen -G '/opt/rocm*' >/dev/null; then
        printf 'ROCm installation directories exist:\n'
        ls -d /opt/rocm*
        bad=1
    fi
    # The kernel side, not rocm-smi: that is a ROCm package, and requiring it
    # would contradict the property being checked.
    if [ ! -e /dev/kfd ]; then
        printf 'no /dev/kfd, so no amdgpu compute device to verify against\n'
        bad=1
    elif [ ! -r /dev/kfd ]; then
        printf '/dev/kfd is not readable by this user (render or video group)\n'
        bad=1
    fi
    [ "$bad" -eq 0 ] && printf 'an amdgpu device and no ROCm installation on this machine\n'
    return "$bad"
}

cuda_driver_only() {
    local bad=0
    if command -v nvcc >/dev/null 2>&1; then
        printf 'nvcc is on PATH: %s\n' "$(command -v nvcc)"
        bad=1
    fi
    if ldconfig -p 2>/dev/null | grep -q libcudart; then
        printf 'the loader cache holds a CUDA runtime:\n'
        ldconfig -p | grep libcudart
        bad=1
    fi
    if compgen -G '/usr/local/cuda*' >/dev/null; then
        printf 'toolkit directories exist:\n'
        ls -d /usr/local/cuda*
        bad=1
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf 'no nvidia-smi, so no NVIDIA driver to verify against\n'
        bad=1
    fi
    [ "$bad" -eq 0 ] && printf 'an NVIDIA driver and no CUDA toolkit on this machine\n'
    return "$bad"
}

# THE BACKEND THIS MACHINE IS BEING ASKED ABOUT. A machine has one vendor's GPU,
# so a distribution carrying two is verified against the one this machine can
# run, and both the driver check and the scene run ask about that one.
driver_only() {
    case "$SUITE_BACKEND" in
        cuda) cuda_driver_only ;;
        rocm) rocm_driver_only ;;
        "") printf 'this distribution stamps %s, so name the backend with --gpu-backend\n' "$BACKENDS"
           return 1 ;;
        *) printf 'this distribution ships no GPU backend, so there is no driver to verify\n'
           return 1 ;;
    esac
}

layout() {
    local bad=0 entry dir
    local -a entries=(ppf-contact-solver config.sh README.txt THIRD_PARTY_LICENSES.txt licenses
        .ppf-selfcontained .git/branch_name.txt bin/ffmpeg python/bin/python3 frontend examples crates)
    for backend in $GPU_BACKENDS; do
        entries+=("bin/libppfbe_$backend.so")
    done
    for dir in $RELEASE_DIRS; do
        entries+=("$dir/ppf-contact-solver" "$dir/ppf-cts-server" "$dir/lib_ppf_cts_py.so" "$dir/.ppf-backend")
    done
    for entry in "${entries[@]}"; do
        if [ -e "$DIST/$entry" ]; then
            printf 'present  %s\n' "$entry"
        else
            printf 'MISSING  %s\n' "$entry"
            bad=1
        fi
    done
    if [ ! -x "$LAUNCHER" ] || [ "$(head -c 2 "$LAUNCHER")" != '#!' ] || ! bash -n "$LAUNCHER"; then
        printf 'the launcher is not an executable script that parses\n'
        bad=1
    fi
    return "$bad"
}

launcher_help() {
    local out
    out="$("$LAUNCHER" --help)" || return 1
    printf '%s\n' "$out"
    case "$out" in
        *"$DIST"*) return 0 ;;
        *) printf 'the help text does not name %s\n' "$DIST"; return 1 ;;
    esac
}

binaries() {
    local bad=0 dir got want
    for dir in $RELEASE_DIRS; do
        want="$(tr -d '[:space:]' < "$DIST/$dir/.ppf-backend")"
        if ! got="$(env -u LD_LIBRARY_PATH "$DIST/$dir/ppf-contact-solver" --backend)"; then
            printf '%s: the solver does not start\n' "$dir"
            bad=1
            continue
        fi
        printf '%s: binary says %s, marker says %s\n' "$dir" "$got" "$want"
        [ "$got" = "$want" ] || bad=1
        if env -u LD_LIBRARY_PATH "$DIST/$dir/ppf-cts-server" --help >/dev/null; then
            printf '%s: the server answers --help\n' "$dir"
        else
            printf '%s: the server does not start\n' "$dir"
            bad=1
        fi
    done
    "$DIST/bin/ffmpeg" -hide_banner -version | head -1 || bad=1
    return "$bad"
}

# loader BACKEND: the solver of that backend takes its library from bin/.
loader() {
    local backend="$1" lib="libppfbe_$1.so" loaded
    # Everything after "calling init: " is the path, which may carry spaces.
    loaded="$(env -u LD_LIBRARY_PATH LD_DEBUG=libs "$DIST/target/$backend/release/ppf-contact-solver" --backend 2>&1 >/dev/null |
        sed -n "s|^.*calling init: \(.*${lib%.so}[^/]*\)$|\1|p" | head -1)"
    printf 'the loader took %s from: %s\n' "$lib" "${loaded:-nowhere}"
    [ -n "$loaded" ] && [ "$(readlink -f "$loaded")" = "$DIST/bin/$lib" ]
}

python_env() {
    local out
    # PPF_CTS_VENV is the launcher's documented override, cleared so the check is
    # of the interpreter the distribution ships.
    out="$(env -u PPF_CTS_VENV "$LAUNCHER" python -c 'import sys, frontend, jupyterlab; print(sys.executable); print(frontend.__file__); print("jupyterlab", jupyterlab.__version__)')" || return 1
    printf '%s\n' "$out"
    case "$out" in
        "$DIST/python/"*"$DIST/frontend/"*) return 0 ;;
        *) printf 'the interpreter or the frontend is not the distribution'"'"'s own\n'; return 1 ;;
    esac
}

# What the frontend answers for each compute device, which is the question the
# add-on and every notebook ask. The binaries answer --backend honestly whatever
# the frontend makes of them, so a backend name the frontend does not count as
# accelerated passes `binaries` and still resolves the GPU build as the CPU
# device. Only asking the frontend itself sees that.
devices() {
    local out bad=0 backend directory
    out="$(env -u PPF_CTS_VENV -u CARGO_TARGET_DIR "$LAUNCHER" python -c \
        'import frontend
built = frontend.list_backends()
for name in sorted(built):
    print(name, built[name])')" || return 1
    printf '%s\n' "$out"
    # EVERY STAMPED BACKEND IS ONE THE FRONTEND FINDS, AND IN ITS OWN DIRECTORY.
    # The binaries answer --backend honestly whatever the frontend makes of them,
    # so a name the frontend does not know would pass `binaries` and then be
    # invisible to every run. Only asking the frontend itself sees that.
    for backend in $BACKENDS; do
        directory="$(printf '%s\n' "$out" | awk -v want="$backend" '$1 == want { print $2 }')"
        if [ -z "$directory" ]; then
            printf 'the frontend does not list the %s build this distribution stamps\n' "$backend"
            bad=1
        elif [ "$(readlink -f "$directory")" != "$(readlink -f "$DIST/target/$backend/release")" ]; then
            printf 'the frontend resolves %s to %s, not to target/%s/release\n' \
                "$backend" "$directory" "$backend"
            bad=1
        fi
    done
    if [ "$(printf '%s\n' "$out" | grep -c .)" -ne "$(printf '%s\n' $BACKENDS | grep -c .)" ]; then
        printf 'the frontend lists builds this distribution does not stamp\n'
        bad=1
    fi
    return "$bad"
}

# suite BACKEND SCENES: run_suite.py through the launcher's own environment. The
# CPU run names this distribution's CPU build, the one CARGO_TARGET_DIR value the
# launcher keeps.
suite() {
    local backend="$1" scenes="$2"
    local -a only=() selector=()
    if [ "$scenes" != "all" ]; then
        # shellcheck disable=SC2206 # a space-separated list of scene names
        only=(--only $scenes)
    fi
    # THE BACKEND IS NAMED RATHER THAN LEFT TO THE RULE. A distribution can carry
    # several GPU builds, and this run is about ONE of them, so its own target
    # directory is what the launcher is given; the automatic choice is what
    # `devices` covers.
    # PPF_DIAG_SELFTEST makes every backend open prove, on this machine, that a
    # failed device check reaches the host: the backend fires one deliberately
    # failing check from a real kernel and refuses to open unless the record
    # comes back. It is off by default because a run should not pay a kernel
    # launch for the proof, and on here because this is the pass that asks
    # whether the archive reports what it is supposed to report. A channel that
    # is unattached or drained privately leaves every gate green while the
    # guarantee-class checks report nothing, and the failure a firing check then
    # produces is an illegal-address fault naming no cause.
    selector=(env "CARGO_TARGET_DIR=$DIST/target/$backend" "PPF_DIAG_SELFTEST=1")
    (cd "$DIST" && env -u PPF_CTS_VENV "${selector[@]}" "$LAUNCHER" python "$DIST/examples/run_suite.py" \
        --backend "$backend" --fast-check --shape-gate off \
        --python "$DIST/python/bin/python3" \
        --data-root "$DIST/local/share/ppf-cts" \
        --out "$REPORT_DIR/suite-$backend.json" "${only[@]}")
}

# Samples the shared libraries every running CUDA solver of this distribution has
# mapped, once a second, until told to stop.
SAMPLER_PID=""
MAPPED="$REPORT_DIR/mapped-libraries.txt"
start_sampler() {
    : > "$MAPPED"
    rm -f "$REPORT_DIR/.sampler-stop"
    (
        while [ ! -e "$REPORT_DIR/.sampler-stop" ]; do
            for proc in /proc/[0-9]*; do
                [ "$(readlink "$proc/exe" 2>/dev/null)" = "$DIST/target/$SUITE_BACKEND/release/ppf-contact-solver" ] || continue
                awk '$6 ~ /\.so/ { print $6 }' "$proc/maps" 2>/dev/null
            done >> "$MAPPED"
            sleep 1
        done
    ) &
    SAMPLER_PID=$!
}
stop_sampler() {
    touch "$REPORT_DIR/.sampler-stop"
    [ -z "$SAMPLER_PID" ] || wait "$SAMPLER_PID" 2>/dev/null
    rm -f "$REPORT_DIR/.sampler-stop"
    sort -u -o "$MAPPED" "$MAPPED"
}

mapped_libraries() {
    local count outside
    count="$(grep -c . "$MAPPED" || true)"
    printf '%s distinct shared libraries mapped by a running CUDA solver\n' "$count"
    cat "$MAPPED"
    if [ "$count" -eq 0 ]; then
        printf 'the sampler saw no running CUDA solver, so nothing was checked\n'
        return 1
    fi
    outside="$(grep -vE "^($DIST/|/lib/|/lib64/|/usr/lib/|/usr/lib64/)" "$MAPPED" || true)"
    if [ -n "$outside" ]; then
        printf 'mapped from outside the distribution and the system library directories:\n%s\n' "$outside"
        return 1
    fi
    if grep -E 'libcudart|/usr/local/cuda' "$MAPPED"; then
        printf 'a CUDA toolkit library was mapped\n'
        return 1
    fi
    # THE WITNESS THAT A SOLVE REACHED THE GPU IS THE DRIVER'S OWN LIBRARY, and
    # each vendor has a different one: libcuda.so is NVIDIA's, and the AMD
    # equivalent a HIP process maps is libhsakmt or the amdgpu DRM library. The
    # ROCm pattern is UNEXERCISED here for want of a machine, like
    # rocm_driver_only above.
    case "$SUITE_BACKEND" in
        rocm) DRIVER_WITNESS='/libhsakmt\.so\|/libdrm_amdgpu\.so' ; DRIVER_NAME="the amdgpu user-mode driver" ;;
        *)    DRIVER_WITNESS='/libcuda\.so' ; DRIVER_NAME="the driver's libcuda" ;;
    esac
    if ! grep -q "$DRIVER_WITNESS" "$MAPPED"; then
        printf '%s was never mapped, so no solve reached the GPU\n' "$DRIVER_NAME"
        return 1
    fi
    printf 'every mapped library is the distribution'"'"'s or the system'"'"'s, and %s is among them\n' "$DRIVER_NAME"
}

snapshot_tree() {
    (cd "$DIST" && find . -path ./local -prune -o -path ./cache -prune -o -type f -print | sort)
}

# WHAT THE DISTRIBUTION WRITES UNDER HOME, WHICH IS NOT EVERYTHING THAT APPEARS
# THERE. The GPU driver owns `~/.nv`, its compute cache, and writes it from
# inside libcuda when a solve opens a device. No environment variable the
# launcher sets reaches it: the launcher roots this program's own state in the
# distribution (MPLCONFIGDIR, NUMBA_CACHE_DIR, JUPYTER_*, IPYTHONDIR), and the
# driver's cache is not among the things it can point anywhere.
#
# MEASURED, because the obvious readings are all wrong. It appeared on both
# architectures at once, and against the release before it every artifact-level
# comparison came back identical: the same device image (`sm_61, sm_75, sm_86,
# sm_89, sm_90, sm_100, sm_120` from the same nvcc 12.8.93), the same
# cuda_arch.txt, no change at all under crates/, the same T4G and L40S, the same
# 580.178.04 driver, and the same physics (drape disp_mean=2.283e-03). So it is
# not a missing cubin JIT-compiling, not device LTO left to be linked at load,
# not a newer driver package and not the machine. Why the earlier release left
# HOME empty on the same hardware is UNEXPLAINED, and is a property of the
# driver's cache rather than of anything shipped here.
#
# The allowance is exactly `.nv` and its contents. Anything else under HOME is
# still a failure, which is what this check exists to catch: this distribution
# putting its own state in a user's home instead of in its folder.
home_untouched() {
    local found
    found="$(find "$FAKE_HOME" -mindepth 1 -name .nv -prune -o -print | head -50)"
    if [ -n "$found" ]; then
        printf 'written under HOME:\n%s\n' "$found"
        return 1
    fi
    if [ -e "$FAKE_HOME/.nv" ]; then
        printf 'HOME (%s) holds only the GPU driver'"'"'s own .nv cache\n' "$FAKE_HOME"
    else
        printf 'HOME (%s) is still empty\n' "$FAKE_HOME"
    fi
}

tree_untouched() {
    local added outside
    added="$(snapshot_tree | comm -13 "$REPORT_DIR/tree-before.txt" -)"
    outside="$(printf '%s\n' "$added" | grep -v -e '^\./examples/' -e '^$' || true)"
    printf 'new files in the distribution outside local/ and cache/: %s\n' "$(printf '%s\n' "$added" | grep -c . || true)"
    printf '%s\n' "$added" | head -40
    if [ -n "$outside" ]; then
        printf 'new files outside local/, cache/ and examples/:\n%s\n' "$outside"
        return 1
    fi
}

note "verify-distribution: $DIST"
note "report: $REPORT_DIR  ($(date -u '+%F %T UTC'))"
check system system_info
[ "$DRIVER_ONLY" -eq 0 ] || check driver-only driver_only
check layout layout
check launcher-help launcher_help
check binaries binaries
for backend in $GPU_BACKENDS; do
    check "loader-$backend" loader "$backend"
done

# From here on HOME is an empty directory, so anything the distribution writes
# outside itself is found rather than mixed into a real home.
FAKE_HOME="$(mktemp -d "${TMPDIR:-/tmp}/ppf-verify-home.XXXXXX")"
export HOME="$FAKE_HOME"
unset XDG_CACHE_HOME XDG_CONFIG_HOME XDG_DATA_HOME XDG_STATE_HOME
snapshot_tree > "$REPORT_DIR/tree-before.txt"

check python-env python_env
check devices devices
# Asked for and impossible is a failure, never a skip: a run meant to prove the
# GPU path would otherwise report every check it did run as a pass.
if [ "$CUDA_SCENES" != "none" ]; then
    if [ -z "$SUITE_BACKEND" ]; then
        check gpu-suite sh -c \
            'printf "--cuda-scenes was given and this distribution stamps %s, so name one with --gpu-backend\n" "$1"; exit 1' \
            _ "$BACKENDS"
    else
        start_sampler
        check "$SUITE_BACKEND-suite" suite "$SUITE_BACKEND" "$CUDA_SCENES"
        stop_sampler
        check mapped-libraries mapped_libraries
    fi
fi
if [ "$CPU_SCENES" != "none" ]; then
    check cpu-suite suite cpu "$CPU_SCENES"
fi
check home-untouched home_untouched
check tree-untouched tree_untouched
# home-untouched.log already names anything that appeared there.
rm -rf "$FAKE_HOME"

note "SUMMARY  passed=$PASSED  failed=$FAILED"
[ "$FAILED" -eq 0 ]
