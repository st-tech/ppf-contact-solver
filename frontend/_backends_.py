# File: _backends_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Which backend a run uses, where a tree or a distribution carries several.

A distribution carries one directory per backend (``target/cuda/release``,
``target/rocm/release``, ``target/cpu/release``), each holding that backend's
solver under the one name every backend links, and a dev tree can hold several
the same way. The rule that chooses among them is written once, here, and it is
this:

1. An explicit choice wins and never falls back.
2. Otherwise each GPU backend built here is asked through its own solver's
   ``--probe``, and the first with a usable device in :data:`GPU_BACKENDS` order
   is the answer.
3. Where none of them has a usable device, the CPU backend is the answer where
   one is built, and the run SAYS SO, naming what each GPU backend reported.
   Where none is built, that is a refusal naming the same reasons.
4. Where no GPU backend is built at all, the CPU backend is the answer, with
   nothing to report.

RULE 3 IS A FALLBACK AND IT IS DELIBERATE, WHICH RULE 1 IS WHAT MAKES SAFE. A
machine with no GPU is the ordinary case for someone reading this project's
notebooks, and the answer that serves them is the one backend that will run.
Refusing instead leaves them one recourse, ``frontend.set_backend("cpu")`` at
the top of every notebook, which is worse than the fallback: that line is then
also there on the machine that HAS a GPU, where it pins a run to the slow
backend for no reason anyone will remember. What must never happen is a SILENT
substitution, so the caller is handed the reason and prints it. An explicit
choice never falls back: someone who named a backend gets that backend or an
error, never the other one.

NOTHING HERE IMPORTS THE EXTENSION MODULE OR STARTS A PROCESS. The package
supplies the directories to search and the probe to run
(``frontend/__init__.py``), and ``frontend/tests/_backend_resolution_.py``
drives the same functions with directories it made and a probe it controls,
which is how the branch where an AMD device answers runs on machines that have
none.
"""

import os

from collections.abc import Callable, Iterable
from typing import NamedTuple, Optional

# The GPU backends, in the order the automatic rule prefers them when more than
# one has a usable device. CUDA before ROCm is the project owner's decision for
# a machine with both an NVIDIA and an AMD GPU; Metal never shares a machine
# with either.
GPU_BACKENDS = ("cuda", "rocm", "metal")
BACKENDS = GPU_BACKENDS + ("cpu",)

# The marker `crates/ppf-cts-solver/build.rs` and every bundler write beside a
# build, naming the backend it holds.
MARKER = ".ppf-backend"

# `ppf-contact-solver --probe`'s exit status for "built, and no usable device on
# this machine" (`crates/ppf-cts-solver/src/main.rs`). Any other nonzero status
# is a failure to answer, and is never read as that answer, with the one
# exception below.
PROBE_UNUSABLE = 3

# THE STATUSES WINDOWS RETURNS WHEN ITS LOADER REFUSES TO START THE PROCESS, so
# the solver never reaches its own `--probe` and prints nothing at all. They
# mean the same thing `PROBE_UNUSABLE` means, and they have to, because a
# distribution carries every backend its architecture can build while a machine
# carries only the runtimes it has: an x64 archive holds both the CUDA and the
# ROCm solver, and the ROCm one imports an AMD runtime DLL that an NVIDIA
# machine does not have. Reading these as a failure to answer would let one
# absent runtime abort a run that a present backend could serve.
#
#   0xC0000135  the imported DLL was not found
#   0xC0000139  it was found and does not export what the image imports
#   0xC0000142  its initializer failed
WINDOWS_LOADER_FAILURE = (3221225781, 3221225785, 3221225794)


class Probe(NamedTuple):
    """One solver's answer to ``--probe``."""

    backend: str
    usable: bool
    # The device's name when usable, and the backend's own reason when not.
    detail: str
    # `ppf_cts_formats::SOURCE_STAMP` as the solver was built with it.
    stamp: str


def read_marker(build_dir: str) -> Optional[str]:
    """The backend a build directory holds, or ``None`` when it holds none."""
    try:
        with open(os.path.join(build_dir, MARKER)) as handle:
            found = handle.read().strip()
    except OSError:
        return None
    return found or None


def builds(target_dirs: Iterable[str], profile: str = "release") -> dict[str, str]:
    """Every build under *target_dirs*, as ``{backend: build directory}``.

    THE FIRST DIRECTORY HOLDING A BACKEND WINS, so a caller lists the directory
    it was told about before the ones it searches. The marker says what a
    directory holds, whatever the directory is called.
    """
    found: dict[str, str] = {}
    for target in target_dirs:
        directory = os.path.join(target, profile)
        name = read_marker(directory)
        if name is not None and name not in found:
            found[name] = directory
    return found


def parse_probe(
    stdout: str,
    stderr: str,
    returncode: int,
    solver: str,
    backend: Optional[str] = None,
) -> Probe:
    """Read a solver's ``--probe`` output, refusing anything that is not an answer.

    *backend* is what the build directory's marker says the solver is. It is
    needed only for the loader-failure case, where the solver prints nothing and
    so cannot name itself.
    """
    fields: dict[str, str] = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition(": ")
        if separator and key not in fields:
            fields[key] = value.strip()
    if "backend" in fields and "stamp" in fields:
        if returncode == 0 and "device" in fields:
            return Probe(fields["backend"], True, fields["device"], fields["stamp"])
        if returncode == PROBE_UNUSABLE and "unusable" in fields:
            return Probe(fields["backend"], False, fields["unusable"], fields["stamp"])
    if backend is not None and returncode in WINDOWS_LOADER_FAILURE:
        return Probe(
            backend,
            False,
            "its runtime library is not installed on this machine, so the "
            f"loader refused to start it (exit status {returncode})",
            "",
        )
    printed = (stdout + stderr).strip() or "nothing"
    raise RuntimeError(
        f"{solver} --probe did not answer (exit status {returncode}). A solver "
        f"built before --probe existed has to be rebuilt. It printed: {printed}"
    )


class Resolution(NamedTuple):
    """The backend a run uses, and what the caller has to tell the user.

    *notice* is empty for every answer that needs no explanation, which is
    almost all of them. It carries a sentence exactly when the automatic rule
    did something the user did not ask for and would otherwise not see: falling
    back to the CPU backend because no GPU here can run. The caller prints it
    once; this module starts no process and writes nothing.
    """

    backend: str
    notice: str = ""


def resolve(
    built: dict[str, str],
    choice: Optional[str],
    probe: Optional[Callable[[str], Probe]],
) -> Resolution:
    """The backend a run uses, by the rule in this module's docstring.

    *built* is :func:`builds`'s answer and *choice* the explicitly chosen
    backend, or ``None``. *probe* takes a backend name and returns that
    backend's :class:`Probe`.

    THE PROBE IS ASKED FOR A SINGLE GPU BUILD TOO, and that is what makes rule
    3 reach the common case. A distribution for one accelerator carries exactly
    one GPU backend beside the CPU one, so the machine with no GPU at all, the
    one this fallback exists for, is precisely the machine where there is
    nothing to choose between. Skipping the question there would answer "cuda"
    and leave the run to fail later, at a point that names the backend rather
    than the machine.

    It IS skipped where the answer cannot change: with no CPU build to fall
    back to and one GPU backend, that backend is the answer whether or not it
    can run, and the failure belongs to the run rather than to this choice.
    """
    listing = ", ".join(sorted(built)) or "nothing"
    if choice is not None:
        if choice not in BACKENDS:
            raise ValueError(
                f"unknown backend {choice!r}: expected one of {', '.join(BACKENDS)}"
            )
        if choice not in built:
            raise RuntimeError(
                f"the {choice} backend is selected, and there is no {choice} "
                f"build here (built: {listing})"
            )
        return Resolution(choice)
    gpus = [name for name in GPU_BACKENDS if name in built]
    if not gpus:
        if "cpu" in built:
            return Resolution("cpu")
        raise RuntimeError(f"no solver is built here (built: {listing})")
    if len(gpus) == 1 and "cpu" not in built:
        return Resolution(gpus[0])
    if probe is None:
        raise ValueError("a GPU backend is built, so a probe is required")
    answers = {name: probe(name) for name in gpus}
    for name in gpus:
        if answers[name].usable:
            return Resolution(name)
    reasons = "; ".join(f"{name}: {answers[name].detail}" for name in gpus)
    if "cpu" in built:
        return Resolution(
            "cpu",
            f"No usable GPU was found on this machine ({reasons}), so the CPU "
            f"backend is selected. It needs no GPU and is substantially slower. "
            f"Choose another with frontend.set_backend(name).",
        )
    raise RuntimeError(
        f"no GPU backend built here has a usable device on this machine "
        f"({reasons}), and there is no CPU build here to fall back to (built: "
        f"{listing}). Build one with "
        f"`CARGO_TARGET_DIR=target/cpu cargo build --release --features cpu`."
    )
