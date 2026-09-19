# File: scenarios/rig_device_diagnostic_channel.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A FAILED DEVICE CHECK REACHES THE HOST, on the build this rig is running.
#
# WHY THIS EXISTS. A neutral kernel body reports a violated invariant by
# handing ``DIAG_ASSERT4`` a handle it never dereferences itself, so what that
# handle points at is decided entirely by the backend. Two ways for that to be
# wrong, and both were real at once on CUDA and on ROCm:
#
#   * the channel is not attached, so a firing check writes through a null
#     device pointer. The report becomes an illegal-address fault naming no
#     cause, and only on the runs where a check had something to say.
#   * the channel is the wrong one. Every generated entry is handed
#     ``diagnostics::global()``, so a backend that creates and drains a private
#     channel of its own reads an object no kernel writes. A recorded violation
#     can then never be read, with every build and every gate green.
#
# WHAT MAKES THIS WORTH A SCENARIO RATHER THAN A UNIT TEST. Neither defect is
# visible to a build, to a unit test, or to a scene that passes: a healthy run
# touches those pages from the device zero times. The only thing that settles
# it is firing a real check on the machine in front of you, which is what
# ``PPF_DIAG_SELFTEST`` makes ``be_create`` do.
#
# ``--probe`` is the vehicle. It already opens the device through the
# constructor a run uses, so nothing had to be added to the solver for this,
# and the backend refuses the open when the record does not come back.
#
# WHAT A SKIP MEANS IS RECORDED, NEVER IMPLIED. A leg whose backend has no
# usable device here cannot exercise the channel, and a scenario that says
# nothing about that reports a pass over nothing, which is the failure this
# whole file exists to prevent.
#
# Subtests:
#   A. a_build_is_present_and_says_which_backend_it_is
#   B. the_self_test_confirms_a_failed_check_reached_the_host
#   C. the_switch_is_off_by_default
#   D. no_probe_output_names_an_illegal_access

from __future__ import annotations

import os
import subprocess
import sys

from . import _runner as r


# No Blender. The subject is the solver binary this tree built and the device
# it opens, and the answer is the same whichever backend is underneath, so the
# scenario is established against the real one.
BACKENDS = ("real",)

# The line the backend prints on stderr from the self-test's success path.
# A SUCCESSFUL OPEN IS NOT EVIDENCE ON ITS OWN: it is what a build that never
# reads the switch also produces, so a check resting on the exit status would
# keep passing the day the mechanism is dropped.
CONFIRMATION = "the device diagnostic channel reports (PPF_DIAG_SELFTEST)"

# The two phrases that must never appear. The first is what a null channel
# produces; the second is the backend's own refusal.
FORBIDDEN = (
    "illegal memory access",
    "the device diagnostic channel does not report",
)

# The backends whose ``be_create`` implements the switch. Metal binds the
# channel buffer to every command buffer that dispatches, so its header is
# never null and its equivalent gate is the poisoned-fixture suite; the CPU
# backend passes a real per-chunk record and a row that disagrees with its
# rendering is a name error at the ``extern``.
SELFTEST_BACKENDS = ("cuda", "rocm")

# `--probe` answers this when the build has no usable device here. Not 1,
# which a panic also produces, so a crash is never read as "unusable".
PROBE_UNUSABLE = 3


def _solver(build_dir: str) -> str:
    """The solver binary inside a build directory.

    ``server_build_dir`` returns the PROFILE directory, `<target>/release`,
    rather than the target directory above it, so the binary and the marker sit
    directly inside it.
    """
    name = "ppf-contact-solver.exe" if sys.platform == "win32" else "ppf-contact-solver"
    return os.path.join(build_dir, name)


def _probe(build_dir: str, selftest: bool) -> tuple[int, str]:
    env = dict(os.environ)
    if selftest:
        env["PPF_DIAG_SELFTEST"] = "1"
    else:
        env.pop("PPF_DIAG_SELFTEST", None)
    # A LAUNCH NAMES THE BUILD DIRECTORY, so the server and the solve cannot
    # come from different builds. `CARGO_TARGET_DIR` is the TARGET directory,
    # which is the parent of the profile directory this scenario was handed.
    env["CARGO_TARGET_DIR"] = os.path.dirname(build_dir.rstrip(os.sep))
    done = subprocess.run(
        [_solver(build_dir), "--probe"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    return done.returncode, done.stdout + done.stderr


def run(ctx: r.ScenarioContext) -> dict:
    checks: dict = {}

    # THE ORCHESTRATOR RESOLVES THE BUILD DIRECTORY, by the frontend's own
    # rule, and no scenario may spell one: `target/release` is one of several
    # layouts a build lands in and is not the one a multi-backend build uses.
    # Imported here rather than at module scope, as `bl_rust_binary_protocol`
    # does, because the module is on the path the orchestrator sets up.
    import orchestrator

    build_dir = orchestrator.server_build_dir()
    marker = os.path.join(build_dir, ".ppf-backend")
    backend = ""
    if os.path.isfile(marker):
        with open(marker, encoding="utf-8") as handle:
            backend = handle.read().strip()
    checks["a_build_is_present_and_says_which_backend_it_is"] = {
        "ok": os.path.isfile(_solver(build_dir)) and bool(backend),
        "details": {"build_dir": build_dir, "backend": backend or None},
    }
    if not checks["a_build_is_present_and_says_which_backend_it_is"]["ok"]:
        return r.report_named_checks(checks)

    code, output = _probe(build_dir, selftest=True)
    named = [phrase for phrase in FORBIDDEN if phrase in output]
    checks["no_probe_output_names_an_illegal_access"] = {
        "ok": not named,
        "details": {"named": named, "tail": output[-400:]},
    }

    if backend not in SELFTEST_BACKENDS:
        checks["the_self_test_confirms_a_failed_check_reached_the_host"] = {
            "ok": True,
            "details": {
                "backend": backend,
                "note": "this backend does not implement the switch; its "
                        "channel is checked by other means, named in the "
                        "header of this file",
            },
        }
    elif code == PROBE_UNUSABLE:
        # No device for this backend on this machine. Recorded rather than
        # passed silently: the channel was not exercised and the reader has to
        # be able to see that.
        checks["the_self_test_confirms_a_failed_check_reached_the_host"] = {
            "ok": True,
            "details": {
                "backend": backend,
                "note": "no usable device here, so the channel was not "
                        "exercised",
                "probe": output[-200:],
            },
        }
    else:
        checks["the_self_test_confirms_a_failed_check_reached_the_host"] = {
            "ok": code == 0 and CONFIRMATION in output,
            "details": {
                "backend": backend, "exit": code,
                "confirmation_present": CONFIRMATION in output,
                "tail": output[-400:],
            },
        }

    # THE PROPERTY HERE IS THAT NOTHING RAN, so the check is the ABSENCE of the
    # confirmation. A run must not pay a kernel launch for a proof it did not
    # ask for, and asserting only the exit status would pass for a build that
    # ran the self-test on every open.
    off_code, off_output = _probe(build_dir, selftest=False)
    checks["the_switch_is_off_by_default"] = {
        "ok": off_code in (0, PROBE_UNUSABLE) and CONFIRMATION not in off_output,
        "details": {
            "exit": off_code,
            "confirmation_present": CONFIRMATION in off_output,
            "tail": off_output[-300:],
        },
    }

    return r.report_named_checks(checks)
