# File: orchestrator.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Orchestrator: spawns the debug solver server in an isolated per-worker
# temp directory, runs a scenario against it, collects the verdict +
# artifacts, and tears the worker down. Supports both sequential and
# multiprocessing-parallel modes; per-worker isolation is identical in
# either case.
#
# Layout (one slot, "<run-id>" allocated per orchestrator invocation):
#
#   $TMPDIR/ppf-debug/<run-id>/
#       worker-NN/
#           server/   ppf-cts-server CWD; progress.log + server.log land here
#           project/  PPF_CTS_DATA_ROOT shadow; data.pickle / vert_*.bin
#           probe/    Blender-side probe artifacts (Blender-driven scenarios)
#           scenario.log
#       report.json

from __future__ import annotations

import json
import multiprocessing as mp
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field

# Self-contained: the orchestrator is launched from a host shell, not
# from Blender, so we can't rely on package-relative imports. Same trick
# as debug/main.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scenarios import _runner as r  # noqa: E402
import scenarios  # noqa: E402


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SERVER_EXE = "ppf-cts-server.exe" if os.name == "nt" else "ppf-cts-server"
SOLVER_EXE = "ppf-contact-solver.exe" if os.name == "nt" else "ppf-contact-solver"
# Long enough for a GPU runtime's first initialization on a cold machine
# (`frontend._PROBE_TIMEOUT_S`).
_PROBE_TIMEOUT_S = 120

# EVERY FILE THIS MODULE READS AS TEXT WAS WRITTEN BY ANOTHER PROCESS, so the
# rig decodes them as UTF-8 and never as the locale's encoding. The server's
# `stdout.log` and `stderr.log` are opened "wb" and carry whatever the Rust
# server emitted, which includes the build worker's tqdm bars; Blender and the
# solver are UTF-8 writers too. `open()` with no encoding picks
# `locale.getpreferredencoding()`, which is cp1252 on the Windows rigs, and
# cp1252 leaves five bytes undefined. A tqdm bar whose filled fraction is not a
# whole eighth of its width draws a PARTIAL block, and U+258D encodes to
# `e2 96 8d`: byte 0x8d, undefined in cp1252. Decoding that raised
# UnicodeDecodeError, which is a ValueError and not an OSError, so it went
# straight past `_read_text`'s guard, out of `run_one` and out of `main`.
# Blender CI run 35448531061 died 30 scenarios into its Windows shard 3 that
# way, and the 26 of 55 it had not reached reported no verdict at all.
#
# `errors="replace"` is deliberate and is not hiding a defect: these files are
# EVIDENCE, read so a human can see what a run did. A byte that is not valid
# UTF-8 is a byte of a log, and a run that crashed mid-write leaves plenty of
# them; losing one to U+FFFD costs a character of a progress bar, while raising
# costs every scenario the rig had left. Decisions are never taken on this
# text, only reported.
_LOG_ENCODING = "utf-8"
_LOG_ERRORS = "replace"


def _backend_rule():
    """``frontend/_backends_.py``, loaded by path.

    It holds the one rule for which build a run uses, and it imports nothing
    of the package. Importing it as ``frontend._backends_`` would run
    ``frontend/__init__.py``, which loads the extension module into THIS
    process; the orchestrator only decides which server to spawn, and the
    build worker the server starts is where ``frontend`` belongs.
    """
    import importlib.util
    path = os.path.join(REPO_ROOT, "frontend", "_backends_.py")
    spec = importlib.util.spec_from_file_location("_ppf_backend_rule", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_target_dirs() -> list[str]:
    """The target directories a build may be in, most specific first.

    The same list ``frontend._candidate_target_dirs`` searches:
    ``CARGO_TARGET_DIR`` when set, then the tree's own ``target``, then the
    named per-backend siblings a tree or a distribution holding several
    backends uses (``target/cuda``, ...).
    """
    seen: list[str] = []
    root = os.path.join(REPO_ROOT, "target")
    override = os.environ.get("CARGO_TARGET_DIR")
    if override:
        seen.append(override if os.path.isabs(override)
                    else os.path.join(REPO_ROOT, override))
    if root not in seen:
        seen.append(root)
    for name in ("cpu", "cuda", "metal", "rocm", "gpu"):
        cand = os.path.join(root, name)
        if cand not in seen:
            seen.append(cand)
    return seen


def _run_solver(directory: str, flag: str) -> subprocess.CompletedProcess:
    """Run the solver in *directory* with one *flag*, bounded, output captured.

    The solver imports its backend library; on Windows that is found through
    PATH, and the launcher that runs this rig puts the build's library
    directories there. The build's own directory goes first so a backend
    whose library sits beside the solver resolves without any of that.
    """
    env = os.environ.copy()
    env["PATH"] = directory + os.pathsep + env.get("PATH", "")
    return subprocess.run(
        [os.path.join(directory, SOLVER_EXE), flag], capture_output=True,
        text=True, encoding=_LOG_ENCODING, errors=_LOG_ERRORS,
        timeout=_PROBE_TIMEOUT_S, env=env,
    )


def _one_line(text: str) -> str:
    """*text*'s non-empty lines joined with `` | ``, for a one-line log entry."""
    return " | ".join(line.strip() for line in text.splitlines() if line.strip())


def _solver_backend_answer(directory: str) -> str:
    """What the solver in *directory* prints for ``--backend``, in one line.

    The marker beside a build says what the bundler or build script believed
    it produced; the binary says what it is, and prints a ``linked:`` line
    when the library it loaded answers differently. This is the answer the
    server asks for itself at startup, put in the rig's own log so a report
    read after the machine is gone still carries it.
    """
    try:
        result = _run_solver(directory, "--backend")
    except (subprocess.TimeoutExpired, OSError) as error:
        return f"<not answered: {error}>"
    answer = _one_line(result.stdout) or "<no output>"
    if result.returncode != 0:
        answer += f" (exit status {result.returncode})"
    return answer


_LAUNCHER_NAMES = ("command.bat", "command.sh")
_LAUNCHER_SOLVER_PREFIXES = ("set SOLVER_PATH=", 'SOLVER_PATH="')


def _session_solvers(project_dir: str) -> list[str]:
    """Every solver path a session launcher under *project_dir* names.

    The build worker writes the launcher (``command.bat`` or ``command.sh``)
    with the solver it resolved, so this is the one record of which binary a
    scenario's solves executed, as opposed to which server they went through.
    """
    found: list[str] = []
    for root, _dirs, files in os.walk(project_dir):
        for name in files:
            if name not in _LAUNCHER_NAMES:
                continue
            for line in _read_text(os.path.join(root, name)).splitlines():
                stripped = line.strip()
                for prefix in _LAUNCHER_SOLVER_PREFIXES:
                    if stripped.startswith(prefix):
                        path = stripped[len(prefix):].rstrip('"')
                        if path not in found:
                            found.append(path)
    return found


def _server_build_line(server_dir: str) -> str:
    """The ``solver build: target_dir=... backend=...`` line the server logged.

    ``ppf-cts-server`` prints it at startup, from its own environment and the
    solver's ``--backend`` answer, so it names the build a server that has
    already run its scenario was actually spawned with.
    """
    marker = "solver build:"
    for name in ("stdout.log", "stderr.log"):
        for line in _read_text(os.path.join(server_dir, name)).splitlines():
            at = line.find(marker)
            if at >= 0:
                return line[at + len(marker):].strip()
    return ""


def _probe_build(rule, directory: str):
    """Ask the solver in *directory* ``--probe`` whether its device is usable."""
    solver = os.path.join(directory, SOLVER_EXE)
    started = time.monotonic()
    try:
        result = _run_solver(directory, "--probe")
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"{solver} --probe did not answer within {_PROBE_TIMEOUT_S} s"
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"{solver} --probe could not be started: {error}"
        ) from error
    # THE ANSWER AND ITS COST GO IN THE LOG, because both decide the run and
    # neither is visible from outside it: which build the servers come from
    # follows from these answers, and a probe is paid again by every build
    # worker a scenario's server starts. Run 35095357463's Windows rig held
    # every solving scenario for minutes and was never collected, and this
    # line is what would have said whether the probe was part of that.
    print(f"[orchestrator] probe {solver}: exit {result.returncode} in "
          f"{time.monotonic() - started:.1f}s: "
          f"{_one_line(result.stdout) or '<no output>'}",
          flush=True)
    # A solver whose backend DLLs are not on PATH does not answer "unusable":
    # Windows refuses to start it, with STATUS_DLL_NOT_FOUND, and parse_probe
    # would report that as a solver built before --probe existed. Name the
    # actual cause, because the remedy is the launcher's PATH, not a rebuild.
    if os.name == "nt" and result.returncode == 0xC0000135:
        raise RuntimeError(
            f"{solver} --probe could not start: a DLL it imports is not on PATH "
            f"(STATUS_DLL_NOT_FOUND). Run the rig from the launcher build.bat "
            f"writes, or put that backend's library directories on PATH the way "
            f"scripts/win/run-blender-rig.ps1 does for every backend."
        )
    return rule.parse_probe(result.stdout, result.stderr, result.returncode,
                            solver)


def resolve_server_build() -> str:
    """The build directory the rig's servers are spawned from.

    THE RIG TAKES ITS SERVER FROM THE BUILD A RUN IN THIS TREE WOULD USE,
    by the rule ``frontend/_backends_.py`` is the single source of: a tree
    or a distribution can hold one directory per backend, every one of them
    holding a ``ppf-cts-server`` beside its solver, and only the
    ``.ppf-backend`` marker says which is which. A plain ``cargo build
    --release`` marks ``target/release`` and that is the answer on a host
    with one build; ``build-win-native/build.bat`` builds CUDA, ROCm and the
    CPU backend each into ``target/<backend>/release`` and leaves no
    ``target/release`` at all, and there the first GPU backend whose solver
    reports a usable device is the answer, exactly as the add-on's launcher
    and the server's build worker decide it.

    Refused by name when nothing is built, when a build holds a marker but no
    server, or when no GPU build here has a usable device: a server from a
    directory the frontend would not choose is the split
    ``blender_addon/core/connection.py`` (``_apply_target_dir``) exists to
    prevent, and the rig must not be the one process that makes it.
    """
    rule = _backend_rule()
    built = rule.builds(_candidate_target_dirs())
    if not built:
        searched = "\n".join(f"  {d}" for d in _candidate_target_dirs())
        raise FileNotFoundError(
            "no solver build found under any of\n"
            f"{searched}\n"
            "(a build directory carries a .ppf-backend marker beside its "
            "solver). The rig drives a server binary; build one with a "
            "solver beside it:\n"
            "  cargo build --release                 (real backend: CUDA "
            "on a CUDA host, Metal on macOS)\n"
            "  cargo build --release -p ppf-cts-server\n"
            "  cargo build --release --features cpu  (the Rust CPU "
            "backend: real physics, about 30x slower; this rig does not "
            "target it yet)"
        )
    # `resolve` answers with (backend, notice): the notice is the automatic
    # rule explaining a choice the caller did not make, which today is the CPU
    # backend taken because no GPU here can run. The rig prints it rather than
    # dropping it, because a whole sweep landing on the CPU backend is the
    # difference between minutes and hours and must not be discovered from the
    # clock.
    answer = rule.resolve(
        built, None, lambda backend: _probe_build(rule, built[backend])
    )
    name = answer.backend
    if answer.notice:
        print(f"[orchestrator] {answer.notice}", flush=True)
    directory = built[name]
    server = os.path.join(directory, SERVER_EXE)
    if not os.path.isfile(server):
        raise FileNotFoundError(
            f"the {name} build in {directory} holds no {SERVER_EXE}. The rig "
            "drives a server binary out of the same directory as the solver; "
            "build it there:\n"
            "  cargo build --release -p ppf-cts-server   (with the same "
            "CARGO_TARGET_DIR as the solver build)"
        )
    return directory


_SERVER_BUILD_DIR: str | None = None


def server_build_dir() -> str:
    """`resolve_server_build`, decided once per orchestrator process."""
    global _SERVER_BUILD_DIR
    if _SERVER_BUILD_DIR is None:
        _SERVER_BUILD_DIR = resolve_server_build()
    return _SERVER_BUILD_DIR
# The debug server runs the **real** ``frontend`` Python module, so it
# needs numpy / scipy / numba / pythreejs / pytetwild / tetgen ... installed.
# These are pre-installed in the project ``.venv``. Falls back to
# ``sys.executable`` for callers who explicitly opt in via --python.
# Windows venvs put the interpreter in ``Scripts\python.exe`` (vs
# POSIX's ``bin/python``), so the dirname has to switch by os.
if os.name == "nt":
    DEFAULT_PYTHON = os.path.join(REPO_ROOT, ".venv", "Scripts", "python.exe")
else:
    DEFAULT_PYTHON = os.path.join(REPO_ROOT, ".venv", "bin", "python")
if not os.path.isfile(DEFAULT_PYTHON):
    DEFAULT_PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Addon-side cbor2 prep
# ---------------------------------------------------------------------------

def warmup_addon_install(*, timeout: float = 60.0) -> tuple[bool, str]:
    """Force Blender's manifest-driven extension wheel install to complete
    before any worker spawns.

    ``install-blender-addon.sh`` creates a symlink at
    ``extensions/user_default/ppf_contact_solver``. The first Blender
    boot that enables the extension triggers the manifest's wheels
    (declared in ``blender_addon/blender_manifest.toml``) to be
    installed into ``extensions/.local/lib/python<X.Y>/site-packages/``
    with the matching ABI for Blender's bundled Python. Doing that
    here, synchronously, means slot 0 never races against the install
    and ``import cbor2`` in a scenario driver always resolves to the
    correctly-ABI'd wheel.

    The probe doubles as a smoke test: if ``import cbor2`` fails inside
    Blender after enable, the orchestrator aborts the whole run with a
    clear error before burning 15 minutes on doomed worker spawns.
    Production users get the same install path via Blender's Remote
    Repository (``bpy.ops.extensions.package_install_files``); this
    function brings the rig into alignment with that flow.
    """
    import blender_harness as bh
    bbin = bh.find_blender()
    if not bbin:
        return False, "Blender binary not found (set PPF_BLENDER_BIN)"

    try:
        probe = subprocess.run(
            [bbin, "-b",
             "--addons", "bl_ext.user_default.ppf_contact_solver",
             "--python-expr",
             "import cbor2; print('CBOR2_OK ' + getattr(cbor2, '__version__', '?'))"],
            capture_output=True, text=True, timeout=timeout,
            encoding=_LOG_ENCODING, errors=_LOG_ERRORS,
        )
    except subprocess.TimeoutExpired as e:
        return False, f"Blender warmup timed out after {timeout}s\n{e}"
    out = (probe.stdout or "") + (probe.stderr or "")
    if probe.returncode != 0 or "CBOR2_OK" not in (probe.stdout or ""):
        return False, (
            f"cbor2 import inside Blender failed (returncode={probe.returncode}):\n"
            f"{out[-2000:]}"
        )
    return True, out


# ---------------------------------------------------------------------------
# Numba environment detection
# ---------------------------------------------------------------------------

def precompile_numba(*, python: str = DEFAULT_PYTHON,
                     timeout: float = 180.0) -> tuple[bool, str]:
    """Run frontend's numba self-tests once: compile + smoke-check.

    Two jobs in one:

      1. **Precompile.** Numba ``@njit(cache=True)`` writes ``.nbi`` /
         ``.nbc`` files to ``frontend/__pycache__/`` on first compile.
         Without this prep step the first test scenario pays the full
         JIT cost (tens of seconds) every run; subsequent ones reuse
         the cache.
      2. **Smoke.** Numba parallel kernels can crash at runtime in
         platform-specific ways (e.g. the workqueue layer aborts on
         "Concurrent access" if Python-level threads call parallel
         njits concurrently). Running ``frontend.tests._runner_``
         here exercises every parallel njit path with realistic
         shapes so a regression surfaces *before* a scenario fails
         opaquely 60 seconds into a build.

    Returns (ok, log_text). On failure, callers should abort the rig."""
    code = (
        "import sys; sys.path.insert(0, %r);\n"
        "from frontend.tests._runner_ import run_all_tests;\n"
        "sys.exit(0 if run_all_tests() else 1)\n"
    ) % REPO_ROOT
    try:
        result = subprocess.run(
            [python, "-c", code],
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding=_LOG_ENCODING,
            errors=_LOG_ERRORS,
        )
    except subprocess.TimeoutExpired as e:
        return False, f"numba precompile timed out after {timeout}s\n{e}"
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output


def _scenario_needs_blender(scenario_module) -> bool:
    """A scenario opts in by setting ``NEEDS_BLENDER = True``."""
    return bool(getattr(scenario_module, "NEEDS_BLENDER", False))


# ---------------------------------------------------------------------------
# Run / worker layout
# ---------------------------------------------------------------------------

def _new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(2)


def _run_root(run_id: str) -> str:
    # tempfile.gettempdir() honors %TEMP% on Windows and TMPDIR on POSIX,
    # so the resulting paths are always absolute and use the platform's
    # native separators. The previous "/tmp" hard-coded default broke
    # Blender on Windows because the bootstrap path was relative.
    import tempfile
    base = os.environ.get(
        "PPF_DEBUG_ROOT",
        os.path.join(tempfile.gettempdir(), "ppf-debug"),
    )
    path = os.path.join(base, run_id)
    os.makedirs(path, exist_ok=True)
    return path


def _alloc_port(start: int = 19090) -> int:
    """Return a free TCP port at or above ``start``. We bind+release a
    socket to let the OS pick rather than scanning, since some macOS
    setups deny SO_REUSEADDR-rebinding fast enough to cause a race."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@dataclass
class WorkerSpec:
    slot: int
    workspace: str
    server_dir: str
    project_dir: str
    probe_dir: str
    scenario_log: str
    server_port: int


@dataclass
class WorkerResult:
    slot: int
    scenario: str
    status: str
    duration_s: float
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    server_stdout: str = ""
    server_stderr: str = ""
    server_progress: str = ""
    # Filled only for a failure, by ``_attach_failure_logs``. A pass
    # carries none of it, which is what keeps a 200-scenario report small.
    worker_dir: str = ""
    scenario_log: str = ""
    blender_stdout: str = ""
    blender_stderr: str = ""
    driver_result: str = ""
    # True when the scenario needed Blender and Blender wrote nothing at
    # all, which means it never reached the addon's registration line and
    # so never ran. ``run_many`` stops a run on a second one in a row.
    blender_never_started: bool = False
    # Filled for every result, pass or fail: the build the worker's server
    # reported at startup, and the solver path each session launcher under
    # the worker named. A passing scenario's worker directory is deleted, so
    # without these the report cannot say which binary its solves ran on.
    server_build: str = ""
    session_solvers: list[str] = field(default_factory=list)


def _provision_worker(run_root: str, slot: int) -> WorkerSpec:
    workspace = os.path.join(run_root, f"worker-{slot:02d}")
    server_dir = os.path.join(workspace, "server")
    project_dir = os.path.join(workspace, "project")
    probe_dir = os.path.join(workspace, "probe")
    for d in (server_dir, project_dir, probe_dir):
        os.makedirs(d, exist_ok=True)
    scenario_log = os.path.join(workspace, "scenario.log")
    server_port = _alloc_port()
    return WorkerSpec(
        slot=slot,
        workspace=workspace,
        server_dir=server_dir,
        project_dir=project_dir,
        probe_dir=probe_dir,
        scenario_log=scenario_log,
        server_port=server_port,
    )


def _spawn_server(spec: WorkerSpec, *, python: str,
                  knobs: dict[str, str]) -> subprocess.Popen:
    """Launch the Rust ``ppf-cts-server`` binary with the worker's CWD and
    PPF_CTS_DATA_ROOT shadow. Returns the Popen handle so the
    orchestrator can wait/kill it.

    ``python`` is unused now that the server is a native binary; the
    parameter is kept so the existing CLI ``--python`` knob still
    parses, and so callers don't need to thread a different argument
    through the orchestrator entry points."""
    del python  # native binary; no interpreter
    build_dir = server_build_dir()
    server_bin = os.path.join(build_dir, SERVER_EXE)
    env = os.environ.copy()
    env["PPF_CTS_DATA_ROOT"] = spec.project_dir
    # NAME THE BUILD THE SERVER CAME OUT OF, as the add-on's launcher does
    # (`connection._apply_target_dir`). The server's build worker loads the
    # cdylib from `CARGO_TARGET_DIR` when it is set and from the tree's
    # `target` otherwise, and names the solver beside that cdylib in the
    # session launcher; the server reports that directory as
    # `solver_target_dir`, and a scenario that attaches the add-on to this
    # server compares it with the build the add-on resolved under the same
    # root (`connection.check_running_server`). A server spawned out of
    # `target/cuda/release` with the variable unset would run the solve out
    # of whichever directory `frontend` found first and report `target`, so
    # the attach would be refused as another build, correctly.
    env["CARGO_TARGET_DIR"] = os.path.dirname(build_dir)
    # THE SERVER IS THE PROCESS THAT SCANS, so the worker-scoped flag belongs
    # HERE and not only in the Blender environment. `solver_busy()` is
    # host-global by default, and this server's monitor adopts any live
    # `ppf-contact-solver` it finds: at `--parallel N` that is routinely
    # another worker's, and the scenario then drives its state machine off a
    # run it does not own. `PPF_SOLVER_SCAN_DESCENDANTS` restricts the scan to
    # this server's own descendants, which is what
    # `ppf-cts-core/src/utils.rs` provides it for.
    env.setdefault("PPF_SOLVER_SCAN_DESCENDANTS", "1")
    # The Rust server spawns a python build worker that imports
    # ``frontend``. The orchestrator runs under the project venv (which
    # has ``frontend`` on its sys.path), so point the server's worker
    # at the same interpreter; otherwise it falls through to a bare
    # ``python3`` and ``ModuleNotFoundError: No module named 'frontend'``.
    if os.path.isfile(DEFAULT_PYTHON):
        env.setdefault("PPF_CTS_BUILD_PYTHON", DEFAULT_PYTHON)
    env.update(knobs)
    stdout_path = os.path.join(spec.server_dir, "stdout.log")
    stderr_path = os.path.join(spec.server_dir, "stderr.log")
    stdout = open(stdout_path, "wb")
    stderr = open(stderr_path, "wb")
    proc = subprocess.Popen(
        [server_bin, "--port", str(spec.server_port), "--debug"],
        cwd=spec.server_dir,
        env=env,
        stdout=stdout,
        stderr=stderr,
        # New process group so SIGTERM doesn't propagate to the orchestrator.
        start_new_session=True,
    )
    # Hold the file handles on the Popen so they aren't GC'd before close.
    proc._ppf_stdout = stdout  # type: ignore[attr-defined]
    proc._ppf_stderr = stderr  # type: ignore[attr-defined]
    return proc


def _wait_for_server_ready(spec: WorkerSpec, *, timeout: float = 15.0) -> None:
    """Poll progress.log + a TCP connect until SERVER_READY is visible."""
    progress_path = os.path.join(spec.server_dir, "progress.log")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(progress_path):
            try:
                with open(progress_path, encoding=_LOG_ENCODING,
                          errors=_LOG_ERRORS) as f:
                    if "SERVER_READY" in f.read():
                        # Confirm the socket actually accepts.
                        try:
                            with socket.create_connection(
                                ("127.0.0.1", spec.server_port), timeout=1.0,
                            ):
                                return
                        except OSError:
                            pass
            except OSError:
                pass
        time.sleep(0.1)
    raise TimeoutError(
        f"server (slot {spec.slot}) did not reach SERVER_READY within "
        f"{timeout}s. progress.log may have details."
    )


def _shutdown_server(proc: subprocess.Popen, *, timeout: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    # Reuse the harness's cross-platform tree-kill so server-only and
    # Blender-driven runs share the same teardown semantics on POSIX
    # and Windows.
    import blender_harness as _bh
    try:
        _bh._kill_tree(proc, timeout=timeout)
    finally:
        for h in (
            getattr(proc, "_ppf_stdout", None),
            getattr(proc, "_ppf_stderr", None),
        ):
            if h:
                try:
                    h.close()
                except OSError:
                    pass


def _read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, encoding=_LOG_ENCODING, errors=_LOG_ERRORS) as f:
            return f.read()
    except OSError as e:
        return f"<read failed: {e}>"


# A failure is usually read later, out of a report file, by someone who cannot
# rerun it: the worker directory stays on the machine that ran the scenario,
# and in CI that machine is gone by the time anyone reads the result. So a
# failing result carries the worker's logs inline. A violation names the check
# that failed and says nothing about what the run was doing when it did, which
# is the half that costs a debugging session to recover.
_LOG_TAIL_CHARS = 16 * 1024


def _tail(text: str, limit: int = _LOG_TAIL_CHARS) -> str:
    """The last *limit* characters of *text*, marked when it was cut."""
    if len(text) <= limit:
        return text
    return f"<truncated, showing the last {limit} characters>\n" + text[-limit:]


def _attach_failure_logs(result: WorkerResult, spec: WorkerSpec, bspec=None) -> None:
    """Attach everything the worker wrote to a failing *result*."""
    result.worker_dir = spec.workspace
    result.scenario_log = _tail(_read_text(spec.scenario_log))
    result.server_stdout = _tail(
        _read_text(os.path.join(spec.server_dir, "stdout.log")))
    result.server_stderr = _tail(
        _read_text(os.path.join(spec.server_dir, "stderr.log")))
    result.server_progress = _tail(
        _read_text(os.path.join(spec.server_dir, "progress.log")))
    if bspec is not None:
        result.blender_stdout = _tail(_read_text(bspec.stdout_path))
        result.blender_stderr = _tail(_read_text(bspec.stderr_path))
        # Blender prints its addon registration line before a driver runs,
        # so an empty stdout is not a quiet scenario: the process never got
        # far enough to run one. A display that has stopped answering, an
        # extension directory being rewritten underneath it and a binary
        # that cannot start all look like this, and all of them are about
        # the machine rather than the scenario.
        result.blender_never_started = not result.blender_stdout.strip()
        # Every named check with its details, passing ones included: the
        # state a scenario was in before the failing check is what says
        # whether the check or the run went wrong.
        result.driver_result = _tail(_read_text(bspec.result_path))


# ---------------------------------------------------------------------------
# Public entry: run one scenario in one worker slot
# ---------------------------------------------------------------------------

def run_one(scenario_name: str, *, slot: int, run_root: str,
            backend: str,
            python: str = sys.executable,
            knobs: dict[str, str] | None = None,
            timeout: float = 60.0) -> WorkerResult:
    """Provision a worker, launch the debug server, run the scenario,
    return the verdict. The worker dir is left in place so the caller
    can decide whether to keep it (failure) or delete it (success).

    ``backend`` is a required keyword rather than a defaulted one: a
    default here would silently mislabel every run that omitted it."""
    scenario = scenarios.get(scenario_name)
    if scenario is None:
        return WorkerResult(
            slot=slot, scenario=scenario_name, status="fail",
            duration_s=0.0,
            violations=[f"unknown scenario: {scenario_name}"],
        )

    # The selection filter in scenarios.all_names() does NOT reach a
    # scenario named explicitly on a command line, so this is the last
    # gate before a scenario runs against a backend it was never written
    # for. It FAILS rather than skips: a scenario the caller asked for
    # and did not get must never be counted as fine.
    reason = scenarios.backend_unsupported_reason(scenario, backend)
    if reason is not None:
        return WorkerResult(
            slot=slot, scenario=scenario_name, status="fail",
            duration_s=0.0,
            violations=[f"{scenario_name} cannot run on backend "
                        f"{backend!r}: {reason}"],
        )

    # Scenarios may declare their own default knobs. CLI ``--knob`` flags
    # override per-scenario defaults so a developer can still poke at
    # edge cases manually.
    scenario_knobs = dict(getattr(scenario, "KNOBS", {}) or {})
    # Force the co-located backends (local / win_native / mac_native) onto the
    # streamed TCP transport by default so every scenario keeps
    # exercising the wire handlers that SSH/Docker rely on in
    # production. The one scenario that targets the direct-disk path
    # opts back out via its own KNOBS; a CLI --knob still overrides
    # either.
    effective_knobs = {
        "PPF_FORCE_TCP_TRANSFER": "1",
        **scenario_knobs,
        **(knobs or {}),
    }
    spec = _provision_worker(run_root, slot)
    proc = _spawn_server(spec, python=python, knobs=effective_knobs)
    started_at = time.monotonic()
    blender_proc = None
    bspec = None
    try:
        try:
            _wait_for_server_ready(spec, timeout=15.0)
        except TimeoutError as e:
            failed = WorkerResult(
                slot=slot, scenario=scenario_name, status="fail",
                duration_s=time.monotonic() - started_at,
                violations=[str(e)],
            )
            _attach_failure_logs(failed, spec)
            return failed

        # The Rust ppf-cts-server stores uploads at
        # ``<PPF_CTS_DATA_ROOT>/<name>`` (see crates/ppf-cts-server/src/
        # wire.rs::handle_tcmd's root synthesis). Scenarios need the same
        # absolute path so they can stat ``data.pickle`` etc. after the
        # upload lands. There is no extra ``git-debug`` segment: the Rust
        # path resolver does not go through frontend's branch lookup.
        project_name = f"slot{slot:02d}"
        project_root = os.path.join(spec.project_dir, project_name)
        os.makedirs(project_root, exist_ok=True)

        ctx = r.ScenarioContext(
            host="127.0.0.1",
            server_port=spec.server_port,
            project_name=project_name,
            workspace=spec.workspace,
            project_root=project_root,
            timeout=timeout,
            log_path=spec.scenario_log,
            knobs=dict(effective_knobs),
            backend=backend,
        )

        # Bring up Blender if the scenario asked for it. The harness is
        # imported lazily so server-only scenarios don't pay the import cost.
        if _scenario_needs_blender(scenario):
            import blender_harness as bh

            blender_bin = bh.find_blender()
            if not blender_bin:
                failed = WorkerResult(
                    slot=slot, scenario=scenario_name, status="fail",
                    duration_s=time.monotonic() - started_at,
                    violations=[
                        "scenario requires Blender but no binary found "
                        "(set PPF_BLENDER_BIN, put `blender` on PATH, or "
                        "run ./install-blender.sh)"
                    ],
                )
                _attach_failure_logs(failed, spec)
                return failed
            # Scenario must export ``build_driver(ctx) -> str`` returning
            # the Python source to exec inside Blender. The bootstrap
            # writes its result to bspec.result_path.
            build_driver = getattr(scenario, "build_driver", None)
            if not callable(build_driver):
                failed = WorkerResult(
                    slot=slot, scenario=scenario_name, status="fail",
                    duration_s=time.monotonic() - started_at,
                    violations=[
                        f"scenario {scenario_name} sets NEEDS_BLENDER but "
                        f"does not export build_driver(ctx)"
                    ],
                )
                _attach_failure_logs(failed, spec)
                return failed
            driver_src = build_driver(ctx)
            # The PC2 cache lands in ``<tempdir>/data`` for as long as
            # the .blend is unsaved, which it is in every scenario that
            # does not save one. That directory is shared by every
            # scenario the machine has ever run, so a capture one
            # scenario leaves behind is still on disk for the next one to
            # find: "Clear All Deformations" gates its poll on that
            # directory (ui/solver.py) and clears the orphans it sees
            # there, so a scenario that captured nothing is still told a
            # cache exists. Give Blender a temp root of our own instead.
            # It is emptied here rather than named per scenario so the
            # path stays short for the Windows leg, and because this runs
            # once per scenario it is the emptying that separates them;
            # a scenario that relaunches Blender reuses this spec, so the
            # directory holds for the whole of it.
            scenario_tmp = os.path.join(spec.workspace, "tmp")
            shutil.rmtree(scenario_tmp, ignore_errors=True)
            os.makedirs(scenario_tmp, exist_ok=True)
            blender_env = dict(effective_knobs)
            # tempfile.gettempdir() reads TMPDIR, then TEMP, then TMP;
            # POSIX honors the first and Windows the other two.
            blender_env.setdefault("TMPDIR", scenario_tmp)
            blender_env.setdefault("TEMP", scenario_tmp)
            blender_env.setdefault("TMP", scenario_tmp)
            bspec = bh.BlenderSpec(
                blender_bin=blender_bin,
                workspace=spec.workspace,
                probe_dir=spec.probe_dir,
                blend_file="",
                driver_source=driver_src,
                env_extra=blender_env,
            )
            blender_proc = bh.spawn(bspec)
            ctx.artifacts["blender_spec"] = bspec
            ctx.artifacts["blender_proc"] = blender_proc

        try:
            verdict = scenario.run(ctx)
        except Exception as exc:  # noqa: BLE001
            verdict = {
                "status": "fail",
                "violations": [f"{type(exc).__name__}: {exc}"],
                "notes": [traceback.format_exc()],
            }

        result = WorkerResult(
            slot=slot,
            scenario=scenario_name,
            status=verdict.get("status", "fail"),
            duration_s=time.monotonic() - started_at,
            violations=list(verdict.get("violations") or []),
            notes=list(verdict.get("notes") or []),
            server_build=_server_build_line(spec.server_dir),
            session_solvers=_session_solvers(spec.project_dir),
        )
        if result.status != "pass":
            _attach_failure_logs(result, spec, bspec)
        return result
    finally:
        if blender_proc is not None:
            import blender_harness as bh
            bh.shutdown(blender_proc)
        _shutdown_server(proc)


# ---------------------------------------------------------------------------
# Public entry: run a list of scenarios, optionally in parallel
# ---------------------------------------------------------------------------

# How much of a failure to put on the console. The whole of it is in the
# report file; what belongs here is enough to name the cause while reading a
# CI log, without burying the 200 lines around it.
_LOG_ATTACH_LIMIT = 25
# How many scenarios in a row may fail with Blender never starting before the
# run gives up. One is a scenario that lost a race; two in a row is the
# machine, and every scenario after it will fail the same way, each paying the
# full per-scenario timeout. On a 236-scenario suite at a 360 s timeout that
# is the difference between one legible failure and twenty hours of them.
_NEVER_STARTED_ABORT = 2
_PRINT_VIOLATIONS = 20
_PRINT_VIOLATION_CHARS = 2000
_PRINT_LOG_LINES = 20


def _print_log_tail(label: str, text: str, lines: int = _PRINT_LOG_LINES) -> None:
    body = (text or "").strip()
    if not body:
        return
    tail = body.splitlines()[-lines:]
    print(f"[orchestrator]   {label} (last {len(tail)} lines):", flush=True)
    for line in tail:
        print(f"[orchestrator]     {line}", flush=True)


def _never_started_streak(streak: int, r_dict: dict) -> int:
    """Consecutive results whose Blender never started, *r_dict* included."""
    return streak + 1 if r_dict.get("blender_never_started") else 0


def _abort_reason(streak: int) -> str:
    reason = (
        f"{streak} scenarios in a row failed with Blender writing nothing at "
        f"all, so it is not starting on this machine (DISPLAY="
        f"{os.environ.get('DISPLAY', 'unset')}). Every scenario left would "
        f"fail the same way, one per-scenario timeout at a time, so the run "
        f"stops here."
    )
    print(f"[orchestrator] ABORTING RUN: {reason}", flush=True)
    return reason


def _print_result(r_dict: dict) -> None:
    """One line per result, plus why it failed when it did."""
    print(f"[orchestrator] slot {r_dict['slot']:02d} "
          f"<- {r_dict['scenario']} {r_dict['status']} "
          f"({r_dict.get('duration_s', 0):.1f}s)",
          flush=True)
    # Which build this worker's solves went through, on every result: a pass
    # over the wrong backend is the case these two lines exist for.
    if r_dict.get("server_build"):
        print(f"[orchestrator]   server build: {r_dict['server_build']}",
              flush=True)
    for path in r_dict.get("session_solvers") or []:
        print(f"[orchestrator]   session solver: {path}", flush=True)
    if r_dict.get("status") == "pass":
        return
    if r_dict.get("worker_dir"):
        print(f"[orchestrator]   worker dir: {r_dict['worker_dir']}", flush=True)
    violations = r_dict.get("violations") or []
    for v in violations[:_PRINT_VIOLATIONS]:
        print(f"[orchestrator]   violation: {str(v)[:_PRINT_VIOLATION_CHARS]}",
              flush=True)
    if len(violations) > _PRINT_VIOLATIONS:
        print(f"[orchestrator]   ... {len(violations) - _PRINT_VIOLATIONS} "
              f"more violation(s) in the report", flush=True)
    for n in r_dict.get("notes") or []:
        print(f"[orchestrator]   note: {str(n)[:_PRINT_VIOLATION_CHARS]}",
              flush=True)
    # A scenario that crashed rather than asserted leaves its reason in a log
    # and nothing in its violations, so the tails go out on every failure.
    _print_log_tail("blender stderr", r_dict.get("blender_stderr", ""))
    _print_log_tail("server stderr", r_dict.get("server_stderr", ""))


def _pool_task(task: dict) -> dict:
    """Pool-friendly entry. Pickled across the process boundary, so we
    accept and return plain dicts instead of dataclasses."""
    result = run_one(
        task["scenario"],
        slot=task["slot"],
        run_root=task["run_root"],
        python=task["python"],
        knobs=task["knobs"],
        timeout=task["timeout"],
        backend=task["backend"],
    )
    return asdict(result)


def _abort_summary(run_id: str, run_root: str, backend: str, parallel: int,
                   repeat: int, unrunnable: dict[str, str], *,
                   scenario: str, violation: str,
                   note: str | None = None) -> dict:
    """A one-failure summary for a run that aborted before any scenario.

    It carries the SAME key set as a completed run, ``unrunnable``
    included. A caller that reads a key here reads it on every path, so a
    setup failure cannot turn into a KeyError that hides the setup
    failure."""
    return {
        "run_id": run_id,
        "run_root": run_root,
        "backend": backend,
        "parallel": parallel,
        "repeat": repeat,
        "total": 0,
        "passed": 0,
        "failed": 1,
        "unrunnable_count": len(unrunnable),
        "unrunnable": dict(unrunnable),
        "results": [{
            "slot": -1, "scenario": scenario,
            "status": "fail", "duration_s": 0.0,
            "violations": [violation],
            "notes": [note] if note is not None else [],
        }],
    }


def run_many(scenario_names: list[str], *,
             backend: str,
             python: str = DEFAULT_PYTHON,
             knobs: dict[str, str] | None = None,
             keep_on_fail: bool = True,
             keep_all: bool = False,
             timeout: float = 60.0,
             parallel: int = 1,
             repeat: int = 1,
             report_path: str | None = None,
             unrunnable: dict[str, str] | None = None) -> dict:
    """Run every named scenario in its own fresh worker, optionally
    repeated and / or parallelized. Returns the aggregated report dict
    (also written to ``report_path`` if given).

    With ``parallel=1`` the runner is sequential. For ``parallel>1`` the
    runs are dispatched via a ``multiprocessing.Pool`` of size ``parallel``;
    each pool worker provisions its own slot, so isolation is identical to
    the sequential path. Port allocation uses bind-to-zero in the
    orchestrator before forking, so collisions are impossible across slots.

    ``unrunnable`` is the caller's map of scenario name -> why *backend*
    cannot host it. It is recorded in the summary so a report artifact
    carries the lost coverage beside the passing count, which otherwise
    reads as a complete suite."""
    run_id = _new_run_id()
    run_root = _run_root(run_id)
    unrunnable = dict(unrunnable or {})
    print(f"[orchestrator] run_id={run_id} root={run_root} "
          f"backend={backend} parallel={parallel} repeat={repeat} "
          f"selected={len(scenario_names)} unrunnable={len(unrunnable)}")
    # Decide the server build here in the parent, before any worker: a tree
    # with nothing built fails at once and by name, not at slot 00, and the
    # log records which directory every server of this run came out of.
    print(f"[orchestrator] server build: {server_build_dir()}", flush=True)
    # And what the solver beside that server says it is. The directory's
    # marker and the binary's answer can disagree, and only the binary is
    # evidence about what the run's solves will call.
    print(f"[orchestrator] solver --backend: "
          f"{_solver_backend_answer(server_build_dir())}", flush=True)

    # Resolve the display once, here in the parent, so the spawned pool
    # workers inherit it through the environment and the whole run shares
    # one server. Only scenarios that drive Blender need one, so a
    # server-only selection still runs on a host with no X at all.
    if any(getattr(scenarios.get(n), "NEEDS_BLENDER", False)
           for n in scenario_names):
        import blender_harness as bh
        try:
            bh.ensure_display()
        except RuntimeError as exc:
            print(f"[orchestrator] {exc}")
            return _abort_summary(
                run_id, run_root, backend, parallel, repeat, unrunnable,
                scenario="<display>", violation=str(exc),
            )

    # Precompile + smoke-check numba kernels once. Failing here means
    # frontend's parallel njit code is broken on this host (e.g. the
    # workqueue layer crashes on "Concurrent access"); aborting now
    # gives a clear error instead of a 60s build timeout per worker.
    # Trigger Blender's manifest-driven extension wheel install (which
    # installs cbor2 with the correct ABI for Blender's bundled Python)
    # synchronously, so no worker races against the install on first
    # boot. This mirrors what Blender's Remote Repository install does
    # for production users.
    print("[orchestrator] warming addon install (manifest wheels)...")
    cbor_ok, cbor_log = warmup_addon_install()
    if not cbor_ok:
        print("[orchestrator] addon warmup FAILED:")
        print(cbor_log[-2000:])
        return _abort_summary(
            run_id, run_root, backend, parallel, repeat, unrunnable,
            scenario="<addon warmup>",
            violation="cbor2 import inside Blender failed; "
                      "manifest wheel install didn't satisfy import",
            note=cbor_log[-1500:],
        )

    print("[orchestrator] precompiling numba kernels...")
    ok, numba_log = precompile_numba(python=python)
    if not ok:
        log_path = os.path.join(run_root, "numba_precompile.log")
        with open(log_path, "w", encoding=_LOG_ENCODING,
                  errors=_LOG_ERRORS) as f:
            f.write(numba_log)
        print(f"[orchestrator] numba precompile FAILED. log: {log_path}")
        print(numba_log[-2000:])
        return _abort_summary(
            run_id, run_root, backend, parallel, repeat, unrunnable,
            scenario="<numba precompile>",
            violation="numba precompile/smoke failed -- see "
                      "numba_precompile.log",
            note=numba_log[-1500:],
        )
    print("[orchestrator] numba precompile OK")

    # Build the task list, partitioning serial-only scenarios from the
    # parallel-eligible ones. Scenarios opt out of parallel by setting
    # ``NOT_PARALLELIZABLE = True`` on the module (typically because
    # they hold cross-cycle solver/server state that the host load can
    # disturb, or they race the live-fetch apply queue under
    # multi-worker dispatch). Serial entries run after the parallel
    # batch finishes so the parallel speedup still applies to the
    # majority of scenarios.
    parallel_tasks: list[dict] = []
    serial_tasks: list[dict] = []
    slot = 0
    for _ in range(max(1, repeat)):
        for name in scenario_names:
            mod = scenarios.get(name)
            serial_only = bool(getattr(mod, "NOT_PARALLELIZABLE", False))
            task = {
                "slot": slot,
                "scenario": name,
                "run_root": run_root,
                "python": python,
                "knobs": knobs or {},
                "timeout": timeout,
                "backend": backend,
            }
            if serial_only:
                serial_tasks.append(task)
            else:
                parallel_tasks.append(task)
            slot += 1

    results: list[dict] = []
    never_started = 0
    aborted = ""
    use_parallel = parallel > 1 and len(parallel_tasks) > 1
    if not use_parallel:
        # Sequential path: parallel and serial sets collapse into one
        # list, original-order. Used when ``parallel<=1`` or the
        # parallel set is degenerate (0 or 1 task).
        for task in parallel_tasks + serial_tasks:
            print(f"[orchestrator] slot {task['slot']:02d} -> {task['scenario']}",
                  flush=True)
            r_dict = _pool_task(task)
            _print_result(r_dict)
            results.append(r_dict)
            never_started = _never_started_streak(never_started, r_dict)
            if never_started >= _NEVER_STARTED_ABORT:
                aborted = _abort_reason(never_started)
                break
    else:
        # ``spawn`` keeps macOS happy and avoids inheriting any
        # half-initialized state from the parent.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=parallel) as pool:
            for r_dict in pool.imap_unordered(_pool_task, parallel_tasks):
                _print_result(r_dict)
                results.append(r_dict)
                never_started = _never_started_streak(never_started, r_dict)
                if never_started >= _NEVER_STARTED_ABORT:
                    aborted = _abort_reason(never_started)
                    pool.terminate()
                    break
        # Serial postlude: NOT_PARALLELIZABLE scenarios get a clean
        # single-worker host without the parallel batch's load.
        if serial_tasks and not aborted:
            print(f"[orchestrator] serial postlude: "
                  f"{len(serial_tasks)} scenario(s)", flush=True)
            for task in serial_tasks:
                print(f"[orchestrator] slot {task['slot']:02d} -> "
                      f"{task['scenario']} (serial)", flush=True)
                r_dict = _pool_task(task)
                _print_result(r_dict)
                results.append(r_dict)
                never_started = _never_started_streak(never_started, r_dict)
                if never_started >= _NEVER_STARTED_ABORT:
                    aborted = _abort_reason(never_started)
                    break

    # Sort by slot for stable reporting regardless of finish order.
    results.sort(key=lambda x: x["slot"])

    # Cleanup: per-result, decide whether to keep the worker dir.
    for rec in results:
        worker_dir = os.path.join(run_root, f"worker-{rec['slot']:02d}")
        keep = keep_all or (keep_on_fail and rec["status"] != "pass")
        if not keep:
            shutil.rmtree(worker_dir, ignore_errors=True)

    failed_results = [x for x in results if x["status"] != "pass"]
    # When a whole run fails the same way (no Blender, no server binary, a
    # broken display), the first failures carry the reason and the rest carry
    # copies of it. Keep the report readable by attaching logs to the first
    # few and leaving the others to their worker directories.
    for rec in failed_results[_LOG_ATTACH_LIMIT:]:
        stripped = False
        for key in ("scenario_log", "blender_stdout", "blender_stderr",
                    "driver_result", "server_stdout", "server_stderr",
                    "server_progress"):
            if rec.get(key):
                rec[key] = ""
                stripped = True
        if stripped:
            rec.setdefault("notes", []).append(
                f"logs left out of this report: more than {_LOG_ATTACH_LIMIT} "
                f"scenarios failed, so they are only in the worker directory")
    if failed_results:
        # A long run scrolls the failures out of sight, so name them again at
        # the end with the worker directory each one left behind.
        print(f"[orchestrator] {len(failed_results)} of {len(results)} "
              f"scenario(s) failed:", flush=True)
        for rec in failed_results:
            first = (rec.get("violations") or ["(no violation recorded)"])[0]
            print(f"[orchestrator]   {rec['scenario']} "
                  f"(slot {rec['slot']:02d}): {str(first)[:_PRINT_VIOLATION_CHARS]}",
                  flush=True)
            # --no-keep has already removed it by this point, so name it only
            # when it is still there to look at.
            if rec.get("worker_dir") and os.path.isdir(rec["worker_dir"]):
                print(f"[orchestrator]     {rec['worker_dir']}", flush=True)

    if aborted:
        not_run = len(parallel_tasks) + len(serial_tasks) - len(results)
        print(f"[orchestrator] {not_run} scenario(s) were not run", flush=True)

    summary = {
        "run_id": run_id,
        "run_root": run_root,
        "backend": backend,
        "parallel": parallel,
        "repeat": repeat,
        # Empty unless the run gave up early; the reason names what stopped
        # it, and `total` then counts the scenarios that ran, not the set
        # that was asked for.
        "aborted": aborted,
        "requested": len(parallel_tasks) + len(serial_tasks),
        "total": len(results),
        "passed": sum(1 for x in results if x["status"] == "pass"),
        "failed": sum(1 for x in results if x["status"] != "pass"),
        # "total" counts what was SELECTED. These two say what was not,
        # so a shrinking suite cannot read as a healthy one.
        "unrunnable_count": len(unrunnable),
        "unrunnable": unrunnable,
        "results": results,
    }

    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".",
                    exist_ok=True)
        with open(report_path, "w", encoding=_LOG_ENCODING) as f:
            json.dump(summary, f, indent=2)
    # Always drop a copy inside the run dir so artifacts stay co-located.
    with open(os.path.join(run_root, "report.json"), "w",
              encoding=_LOG_ENCODING) as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Run debug scenarios in isolated workers.",
    )
    parser.add_argument(
        "scenarios", nargs="*",
        help="Scenario names (default: all).",
    )
    parser.add_argument("--list", action="store_true",
                        help="List registered scenarios and exit.")
    parser.add_argument("--shard", default="",
                        help="I/N: run only every N-th scenario of the "
                             "selection, starting at the I-th (0-based). "
                             "The split is over the registry's order, so "
                             "N hosts given 0/N .. N-1/N together run the "
                             "whole selection exactly once.")
    parser.add_argument("--python", default=DEFAULT_PYTHON,
                        help="Python interpreter for spawned servers "
                             "(default: project .venv).")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Per-scenario timeout (s).")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Worker pool size. 1 = sequential (default).")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Run the scenario list this many times. "
                             "Useful for shaking out flakes.")
    parser.add_argument("--keep-all", action="store_true",
                        help="Keep every worker dir, even passing ones.")
    parser.add_argument("--no-keep", action="store_true",
                        help="Delete worker dirs even on failure (debug builds only).")
    parser.add_argument("--report", default=None,
                        help="Write the aggregated report to this path.")
    parser.add_argument("--knob", action="append", default=[],
                        help='Extra env knob, "KEY=value". Repeatable.')
    # Required and choice-free, matching main.py runtests. See the comment
    # there: a defaulted backend name would label a run it never targeted,
    # and an unknown one must answer through `resolve_backend` rather than
    # with argparse's "invalid choice", which reads as a typo.
    parser.add_argument("--backend", required=True,
                        help="Solver backend the run targets. 'real' is a "
                             "backend that computes real physics: CUDA, "
                             "Metal or the Rust CPU backend, whichever the "
                             "tree was built for.")
    args = parser.parse_args(argv)

    try:
        backend = scenarios.resolve_backend(args.backend)
    except scenarios.BackendUnavailable as exc:
        print(f"orchestrator: {exc}", file=sys.stderr)
        return 2

    unrunnable = scenarios.unrunnable_names(backend)

    if args.list:
        for name in scenarios.all_names(backend):
            print(name)
        _print_unrunnable(backend, unrunnable, stream=sys.stderr)
        return 0

    knobs = {}
    for kv in args.knob:
        if "=" not in kv:
            print(f"--knob expects KEY=value, got {kv!r}", file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        knobs[k] = v

    if args.scenarios:
        # Explicit names bypass the selection filter; catch them here so a
        # scenario that cannot run says so instead of running blind.
        named_dead = {n: unrunnable[n] for n in args.scenarios
                      if n in unrunnable}
        if named_dead:
            print(f"orchestrator: {len(named_dead)} named scenario(s) cannot "
                  f"run on backend {backend!r}:", file=sys.stderr)
            for name, reason in named_dead.items():
                print(f"  {name}: {reason}", file=sys.stderr)
            return 2
        names = list(args.scenarios)
    else:
        names = scenarios.all_names(backend)
        _print_unrunnable(backend, unrunnable, stream=sys.stdout)

    if args.shard:
        try:
            names = select_shard(names, args.shard)
        except ValueError as exc:
            print(f"--shard: {exc}", file=sys.stderr)
            return 2

    summary = run_many(
        names,
        python=args.python,
        knobs=knobs,
        keep_on_fail=not args.no_keep,
        keep_all=args.keep_all,
        timeout=args.timeout,
        parallel=args.parallel,
        repeat=args.repeat,
        report_path=args.report,
        backend=backend,
        unrunnable=unrunnable,
    )
    print(json.dumps({
        "run_id": summary["run_id"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "total": summary["total"],
        "unrunnable": summary["unrunnable_count"],
    }, indent=2))
    return 0 if summary["failed"] == 0 else 1


def select_shard(names: list[str], spec: str) -> list[str]:
    """Return the ``I/N`` share of *names*: every N-th, starting at the I-th.

    THE SPLIT IS BY INDEX IN THE REGISTRY'S ORDER, which is what ``--list``
    prints and what every host sees identically, so N hosts given
    ``0/N .. N-1/N`` partition the selection with no overlap and no gap;
    CI runs one instance per shard. A malformed spec raises rather than
    reading as "everything".
    """
    try:
        index, count = (int(part) for part in spec.split("/", 1))
    except ValueError:
        raise ValueError(f"expected I/N, got {spec!r}") from None
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"{spec!r}: need 0 <= I < N and N >= 1")
    share = [n for k, n in enumerate(names) if k % count == index]
    # On stderr: `runtests --list` prints the share on stdout, and a line
    # about the share is not a scenario name.
    print(f"[orchestrator] shard {index}/{count}: {len(share)} of "
          f"{len(names)} scenarios", file=sys.stderr)
    return share


def _print_unrunnable(backend: str, unrunnable: dict, *, stream) -> None:
    """Name every scenario this backend cannot host, and why."""
    if not unrunnable:
        return
    print(f"\n[orchestrator] {len(unrunnable)} registered scenario(s) CANNOT "
          f"RUN on backend {backend!r} and were not selected. This is lost "
          f"coverage, not a pass:", file=stream)
    by_reason: dict[str, list[str]] = {}
    for name, reason in unrunnable.items():
        by_reason.setdefault(reason, []).append(name)
    for reason, names in by_reason.items():
        print(f"  reason: {reason}", file=stream)
        for name in sorted(names):
            print(f"    {name}", file=stream)
    print("", file=stream)


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
