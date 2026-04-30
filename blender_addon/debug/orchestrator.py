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
#           server/   server.py CWD; progress.log + server.log land here
#           project/  PPF_CTS_DATA_ROOT shadow; data.pickle / vert_*.bin
#           probe/    Blender-side probe artifacts (Phase 2)
#           scenario.log
#       report.json

from __future__ import annotations

import json
import multiprocessing as mp
import os
import secrets
import shutil
import signal
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
SERVER_PY = os.path.join(REPO_ROOT, "server.py")
# The debug server runs the **real** ``frontend`` Python module, so it
# needs numpy / scipy / numba / pythreejs / pytetwild ... installed.
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
    """Launch ``server.py --debug --port <port>`` with the worker's CWD
    and PPF_CTS_DATA_ROOT shadow. Returns the Popen handle so the
    orchestrator can wait/kill it."""
    env = os.environ.copy()
    env["PPF_CTS_DATA_ROOT"] = spec.project_dir
    env.update(knobs)
    stdout_path = os.path.join(spec.server_dir, "stdout.log")
    stderr_path = os.path.join(spec.server_dir, "stderr.log")
    stdout = open(stdout_path, "wb")
    stderr = open(stderr_path, "wb")
    proc = subprocess.Popen(
        [python, SERVER_PY, "--debug", "--port", str(spec.server_port)],
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
                with open(progress_path) as f:
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
        with open(path) as f:
            return f.read()
    except OSError as e:
        return f"<read failed: {e}>"


# ---------------------------------------------------------------------------
# Public entry: run one scenario in one worker slot
# ---------------------------------------------------------------------------

def run_one(scenario_name: str, *, slot: int, run_root: str,
            python: str = sys.executable,
            knobs: dict[str, str] | None = None,
            timeout: float = 60.0) -> WorkerResult:
    """Provision a worker, launch the debug server, run the scenario,
    return the verdict. The worker dir is left in place so the caller
    can decide whether to keep it (failure) or delete it (success)."""
    scenario = scenarios.get(scenario_name)
    if scenario is None:
        return WorkerResult(
            slot=slot, scenario=scenario_name, status="fail",
            duration_s=0.0,
            violations=[f"unknown scenario: {scenario_name}"],
        )

    spec = _provision_worker(run_root, slot)
    # Scenarios may declare their own default knobs (e.g. the
    # intersection-records test needs PPF_EMULATED_FAIL_AT_FRAME set
    # so the Rust binary actually trips its synthetic failure). CLI
    # ``--knob`` flags override per-scenario defaults so a developer
    # can still poke at edge cases manually.
    scenario_knobs = dict(getattr(scenario, "KNOBS", {}) or {})
    effective_knobs = {**scenario_knobs, **(knobs or {})}
    proc = _spawn_server(spec, python=python, knobs=effective_knobs)
    started_at = time.monotonic()
    blender_proc = None
    try:
        try:
            _wait_for_server_ready(spec, timeout=15.0)
        except TimeoutError as e:
            return WorkerResult(
                slot=slot, scenario=scenario_name, status="fail",
                duration_s=time.monotonic() - started_at,
                violations=[str(e)],
                server_stdout=_read_text(os.path.join(spec.server_dir, "stdout.log")),
                server_stderr=_read_text(os.path.join(spec.server_dir, "stderr.log")),
                server_progress=_read_text(os.path.join(spec.server_dir, "progress.log")),
            )

        # The shadow PPF_CTS_DATA_ROOT plus a "git-debug" subdir matches
        # what the server emulator's _make_fake_root produces. Scenarios
        # need the absolute project root so they can stat files later.
        project_name = f"slot{slot:02d}"
        project_root = os.path.join(spec.project_dir, "git-debug", project_name)
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
        )

        # Bring up Blender if the scenario asked for it. The harness is
        # imported lazily so server-only scenarios don't pay the import cost.
        if _scenario_needs_blender(scenario):
            import blender_harness as bh

            blender_bin = bh.find_blender()
            if not blender_bin:
                return WorkerResult(
                    slot=slot, scenario=scenario_name, status="fail",
                    duration_s=time.monotonic() - started_at,
                    violations=[
                        "scenario requires Blender but no binary found "
                        "(set PPF_BLENDER_BIN or install Blender to "
                        "/Applications/Blender.app)"
                    ],
                )
            # Scenario must export ``build_driver(ctx) -> str`` returning
            # the Python source to exec inside Blender. The bootstrap
            # writes its result to bspec.result_path.
            build_driver = getattr(scenario, "build_driver", None)
            if not callable(build_driver):
                return WorkerResult(
                    slot=slot, scenario=scenario_name, status="fail",
                    duration_s=time.monotonic() - started_at,
                    violations=[
                        f"scenario {scenario_name} sets NEEDS_BLENDER but "
                        f"does not export build_driver(ctx)"
                    ],
                )
            driver_src = build_driver(ctx)
            bspec = bh.BlenderSpec(
                blender_bin=blender_bin,
                workspace=spec.workspace,
                probe_dir=spec.probe_dir,
                blend_file="",
                driver_source=driver_src,
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

        return WorkerResult(
            slot=slot,
            scenario=scenario_name,
            status=verdict.get("status", "fail"),
            duration_s=time.monotonic() - started_at,
            violations=list(verdict.get("violations") or []),
            notes=list(verdict.get("notes") or []),
        )
    finally:
        if blender_proc is not None:
            import blender_harness as bh
            bh.shutdown(blender_proc)
        _shutdown_server(proc)


# ---------------------------------------------------------------------------
# Public entry: run a list of scenarios sequentially (Phase 1 = single proc)
# ---------------------------------------------------------------------------

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
    )
    return asdict(result)


def run_many(scenario_names: list[str], *,
             python: str = DEFAULT_PYTHON,
             knobs: dict[str, str] | None = None,
             keep_on_fail: bool = True,
             keep_all: bool = False,
             timeout: float = 60.0,
             parallel: int = 1,
             repeat: int = 1,
             report_path: str | None = None) -> dict:
    """Run every named scenario in its own fresh worker, optionally
    repeated and / or parallelized. Returns the aggregated report dict
    (also written to ``report_path`` if given).

    With ``parallel=1`` the runner is sequential (Phase 1 behavior). For
    ``parallel>1`` the runs are dispatched via a ``multiprocessing.Pool``
    of size ``parallel``; each pool worker provisions its own slot, so
    isolation is identical to the sequential path. Port allocation uses
    bind-to-zero in the orchestrator before forking, so collisions are
    impossible across slots."""
    run_id = _new_run_id()
    run_root = _run_root(run_id)
    print(f"[orchestrator] run_id={run_id} root={run_root} "
          f"parallel={parallel} repeat={repeat}")

    # Precompile + smoke-check numba kernels once. Failing here means
    # frontend's parallel njit code is broken on this host (e.g. the
    # workqueue layer crashes on "Concurrent access"); aborting now
    # gives a clear error instead of a 60s build timeout per worker.
    print("[orchestrator] precompiling numba kernels...")
    ok, numba_log = precompile_numba(python=python)
    if not ok:
        log_path = os.path.join(run_root, "numba_precompile.log")
        with open(log_path, "w") as f:
            f.write(numba_log)
        print(f"[orchestrator] numba precompile FAILED. log: {log_path}")
        print(numba_log[-2000:])
        return {
            "run_id": run_id, "run_root": run_root,
            "parallel": parallel, "repeat": repeat,
            "total": 0, "passed": 0, "failed": 1,
            "results": [{
                "slot": -1, "scenario": "<numba precompile>",
                "status": "fail", "duration_s": 0.0,
                "violations": ["numba precompile/smoke failed -- see numba_precompile.log"],
                "notes": [numba_log[-1500:]],
            }],
        }
    print("[orchestrator] numba precompile OK")

    # Build the full task list: one entry per (scenario, repeat) pair.
    tasks: list[dict] = []
    slot = 0
    for _ in range(max(1, repeat)):
        for name in scenario_names:
            tasks.append({
                "slot": slot,
                "scenario": name,
                "run_root": run_root,
                "python": python,
                "knobs": knobs or {},
                "timeout": timeout,
            })
            slot += 1

    results: list[dict] = []
    if parallel <= 1 or len(tasks) <= 1:
        for task in tasks:
            print(f"[orchestrator] slot {task['slot']:02d} -> {task['scenario']}",
                  flush=True)
            r_dict = _pool_task(task)
            print(f"[orchestrator] slot {r_dict['slot']:02d} "
                  f"<- {r_dict['scenario']} {r_dict['status']} "
                  f"({r_dict.get('duration_s', 0):.1f}s)",
                  flush=True)
            results.append(r_dict)
    else:
        # ``spawn`` keeps macOS happy and avoids inheriting any
        # half-initialized state from the parent.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=parallel) as pool:
            for r_dict in pool.imap_unordered(_pool_task, tasks):
                print(f"[orchestrator] slot {r_dict['slot']:02d} "
                      f"<- {r_dict['scenario']} {r_dict['status']} "
                      f"({r_dict.get('duration_s', 0):.1f}s)",
                      flush=True)
                results.append(r_dict)

    # Sort by slot for stable reporting regardless of finish order.
    results.sort(key=lambda x: x["slot"])

    # Cleanup: per-result, decide whether to keep the worker dir.
    for rec in results:
        worker_dir = os.path.join(run_root, f"worker-{rec['slot']:02d}")
        keep = keep_all or (keep_on_fail and rec["status"] != "pass")
        if not keep:
            shutil.rmtree(worker_dir, ignore_errors=True)

    summary = {
        "run_id": run_id,
        "run_root": run_root,
        "parallel": parallel,
        "repeat": repeat,
        "total": len(results),
        "passed": sum(1 for x in results if x["status"] == "pass"),
        "failed": sum(1 for x in results if x["status"] != "pass"),
        "results": results,
    }

    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".",
                    exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)
    # Always drop a copy inside the run dir so artifacts stay co-located.
    with open(os.path.join(run_root, "report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Run Phase-1 debug scenarios in isolated workers.",
    )
    parser.add_argument(
        "scenarios", nargs="*",
        help="Scenario names (default: all).",
    )
    parser.add_argument("--list", action="store_true",
                        help="List registered scenarios and exit.")
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
    args = parser.parse_args(argv)

    if args.list:
        for name in scenarios.all_names():
            print(name)
        return 0

    knobs = {}
    for kv in args.knob:
        if "=" not in kv:
            print(f"--knob expects KEY=value, got {kv!r}", file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        knobs[k] = v

    names = args.scenarios or scenarios.all_names()
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
    )
    print(json.dumps({
        "run_id": summary["run_id"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "total": summary["total"],
    }, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
