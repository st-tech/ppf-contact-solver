# File: frontend/tests/_device_diagnostics_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# THE DEVICE DIAGNOSTIC CHANNEL REPORTS, on this machine, for every backend
# this tree has built.
#
# A neutral kernel body reports a violated invariant by handing `DIAG_ASSERT4`
# a handle it never dereferences itself, so what that handle points at is
# decided entirely by the backend. Two ways for that to be wrong, and both were
# real at once on CUDA and on ROCm:
#
#   * the channel is not attached, so a firing check writes through a null
#     device pointer. That is an illegal-address fault naming no cause, and it
#     only ever happens on the runs where a check had something to say.
#   * the channel is the wrong one, because every generated entry is handed
#     `diagnostics::global()` while the backend drained a private channel of its
#     own. The device writes one object and the host reads another, so a
#     recorded violation can never be read, with every build and every gate
#     green.
#
# NEITHER IS VISIBLE TO A BUILD, A UNIT TEST OR A SCENE THAT PASSES. A healthy
# run touches those pages from the device zero times. So this asks the question
# the only way it can be answered: it opens each built backend with
# `PPF_DIAG_SELFTEST` set, which makes the backend fire one deliberately failing
# check from a real kernel and refuse the open unless the record comes back.
#
# `--probe` is the vehicle because it already opens the device through the
# constructor a run uses, so no mode had to be added to the solver for this.
#
# WHAT A SKIP MEANS IS PRINTED, never implied. A machine with no device for a
# backend cannot exercise it, and a run that says nothing about that reads as a
# pass over nothing, which is the failure mode this whole file exists to
# prevent.

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The backends whose `be_create` implements `PPF_DIAG_SELFTEST`. For these, a
# successful open with the switch set IS the proof: the backend refuses to open
# when the record does not come back.
SELFTEST_BACKENDS = ("cuda", "rocm")

# The other two reach the same property by other means, and each is named so a
# reader is not left thinking they are unchecked. Metal binds the channel buffer
# to every command buffer that dispatches, so its header is never null, and its
# equivalent of this test is the fixture suite
# (`make -C crates/ppf-cts-compute/metal fixtures-poisoned`). The CPU backend
# passes a real per-chunk record whenever an entry declares the lane, and a row
# that disagrees with its rendering is a name error at the `extern` rather than
# a null handed to a body that will write through it.
OTHER_BACKENDS = {
    "metal": "the channel is bound per command buffer; "
    "its poisoned-fixture suite is the equivalent gate",
    "cpu": "the channel is a per-chunk record and a mismatch is a link error",
}

# What must never appear, whatever else does. The first is the fault a null
# channel produces; the second is the backend's own refusal.
FORBIDDEN = (
    "illegal memory access",
    "the device diagnostic channel does not report",
)

# WHAT PROVES THE SELF-TEST RAN. A successful open is not evidence: it is what a
# build that never reads the switch also produces, so a test resting on the exit
# status alone would keep passing the day the mechanism is dropped. The backend
# prints this line on stderr from the success path, and the two assertions below
# require it to be PRESENT with the switch set and ABSENT without it.
CONFIRMATION = "the device diagnostic channel reports (PPF_DIAG_SELFTEST)"


def _load_backends():
    path = REPO_ROOT / "frontend" / "_backends_.py"
    spec = importlib.util.spec_from_file_location("_backends_for_diag", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _solver_name() -> str:
    return "ppf-contact-solver.exe" if sys.platform == "win32" else "ppf-contact-solver"


def _built_backends() -> dict[str, Path]:
    """Every backend this tree has built, by name, as a target directory.

    Both layouts a build produces: `target/release` from a plain
    `cargo build --release`, and `target/<backend>/release` as the bundlers and
    the Windows batch files lay them out. The marker inside says what a
    directory IS, which is the only thing that does: every backend links the
    same executable name.
    """
    backends = _load_backends()
    target = REPO_ROOT / "target"
    if not target.is_dir():
        return {}
    candidates = [target] + [d for d in sorted(target.iterdir()) if d.is_dir()]
    found: dict[str, Path] = {}
    unmarked: list[str] = []
    for directory in candidates:
        release = directory / "release"
        if not (release / _solver_name()).is_file():
            continue
        marker = release / backends.MARKER
        if not marker.is_file():
            # AN UNMARKED BUILD IS A DEFECT, NOT A REASON TO LOOK AWAY. The
            # marker is what says which backend a directory holds, and both the
            # frontend's resolution and the add-on's device row key on it, so a
            # build without one is present and invisible. Skipping it here would
            # report that state as "nothing built".
            unmarked.append(str(release))
            continue
        name = marker.read_text().strip()
        # A plain `target/release` and a `target/<backend>/release` can hold the
        # same backend; either answers the question, so the first found wins.
        found.setdefault(name, directory)
    if unmarked:
        raise AssertionError(
            "these directories hold a solver binary and no "
            f"{backends.MARKER} marker, so nothing can say which backend they "
            "are: " + ", ".join(unmarked)
        )
    return found


def _probe(target: Path) -> tuple[int, str]:
    env = dict(os.environ)
    env["PPF_DIAG_SELFTEST"] = "1"
    env["CARGO_TARGET_DIR"] = str(target)
    result = subprocess.run(
        [str(target / "release" / _solver_name()), "--probe"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    return result.returncode, result.stdout + result.stderr


def test_every_built_backend_reports_a_failed_device_check() -> None:
    backends = _load_backends()
    built = _built_backends()
    if not built:
        print("    no built backend in this tree: nothing was exercised")
        return

    exercised = []
    for name, target in sorted(built.items()):
        code, output = _probe(target)
        for phrase in FORBIDDEN:
            assert phrase not in output, (
                f"{name}: `--probe` under PPF_DIAG_SELFTEST said {phrase!r}. "
                f"A failed device check must reach the host as a report, not as "
                f"a fault naming no cause.\n{output}"
            )
        if name in SELFTEST_BACKENDS:
            if code == backends.PROBE_UNUSABLE:
                # No device for this backend here. The switch never ran, and
                # saying so is the point: a silent pass over nothing is what
                # this file exists to prevent.
                print(f"    {name}: no usable device here, channel not exercised")
                continue
            assert code == 0, (
                f"{name}: `--probe` exited {code} under PPF_DIAG_SELFTEST. The "
                f"backend refuses to open when a deliberately failing device "
                f"check does not reach the host.\n{output}"
            )
            assert CONFIRMATION in output, (
                f"{name}: `--probe` succeeded under PPF_DIAG_SELFTEST and did "
                f"not print the self-test's confirmation. A successful open is "
                f"also what a build that never reads the switch produces, so "
                f"without the line this proves nothing.\n{output}"
            )
            exercised.append(name)
            print(f"    {name}: a failed device check reached the host: PASS")
        else:
            reason = OTHER_BACKENDS.get(name, "no self-test switch on this backend")
            print(f"    {name}: {reason}")

    if not exercised:
        print("    no backend implementing the switch had a device here")


def test_the_switch_is_off_by_default() -> None:
    """A run pays nothing for the proof unless it asks for it.

    The self-test costs a kernel launch and fires a real assert, so it must not
    be on by default. This is what keeps that true: `--probe` with the switch
    unset opens the same device and answers the same way.
    """
    built = _built_backends()
    if not built:
        print("    no built backend in this tree: nothing was exercised")
        return
    backends = _load_backends()
    for name, target in sorted(built.items()):
        env = dict(os.environ)
        env.pop("PPF_DIAG_SELFTEST", None)
        env["CARGO_TARGET_DIR"] = str(target)
        result = subprocess.run(
            [str(target / "release" / _solver_name()), "--probe"],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        combined = result.stdout + result.stderr
        assert result.returncode in (0, backends.PROBE_UNUSABLE), (
            f"{name}: `--probe` with the switch unset exited "
            f"{result.returncode}\n{combined}"
        )
        # THE PROPERTY THIS NAMES IS THAT NOTHING RAN, so the check is the
        # ABSENCE of the confirmation. Asserting only the exit status would pass
        # for a build that ran the self-test on every open, which is the cost
        # the default exists to avoid.
        assert CONFIRMATION not in combined, (
            f"{name}: the self-test ran with PPF_DIAG_SELFTEST unset, so every "
            f"backend open pays for a kernel launch nobody asked for\n{combined}"
        )
        print(f"    {name}: unset is quiet: PASS")


def run_tests() -> bool:
    """Run the device diagnostic channel tests. True if all pass."""
    print("=" * 50)
    print("Device Diagnostic Channel Tests")
    print("=" * 50)

    try:
        test_every_built_backend_reports_a_failed_device_check()
        test_the_switch_is_off_by_default()
        print("\nAll device diagnostic tests PASSED!")
        return True
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nTest ERROR: {e}")
        return False
