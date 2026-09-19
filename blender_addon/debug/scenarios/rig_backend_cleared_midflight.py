# File: scenarios/rig_backend_cleared_midflight.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A DISCONNECT LANDING MID-OPERATION IS A TRANSPORT FAILURE, NEVER AN
# AttributeError (``blender_addon/core/effect_runner.py``).
#
# WHY THIS EXISTS. Every operation ON a connection is queued onto the I/O
# worker (``_submit_cmd(self._do_stop_server)`` and the rest), but DISCONNECT IS
# NOT: ``DoDisconnect`` calls ``_do_disconnect`` directly, on Blender's main
# thread. So the two do not serialize against each other, and ``_do_disconnect``
# sets ``self._backend = None`` while the worker may be in the middle of a long
# operation.
#
# ``_do_stop_server`` guarded the attribute ONCE at entry and then re-read it
# five more times across a wait loop with ``time.sleep(0.25)`` between passes.
# A disconnect inside that window turned the next read into ``None`` and the
# loop raised ``'NoneType' object has no attribute 'query'``, which the worker's
# handler turned into a panel error naming the method. Measured in a live
# session, one second apart:
#
#     [06:40:59] Disconnected.
#     [06:40:59] [_do_stop_server] 'NoneType' object has no attribute 'query'
#
# The launch path had the same shape and a longer window: ``_do_launch_server``,
# its ``_launch_native_server`` helper and the ``_wait_for_native_server``
# readiness loop, which polls until a deadline.
#
# WHAT THE FIX IS, AND WHY IT IS NOT MERELY A NULL CHECK. Each operation binds
# the backend ONCE, after its guard, and works on that object. Acting on the
# backend the operation was DISPATCHED for is the right answer rather than a
# safe one: it is the server the user asked to stop, and a transport that has
# since closed then reports a transport failure, which is true, instead of an
# AttributeError, which describes the add-on rather than the machine.
#
# HOW IT IS MEASURED. No Blender, no server, no network. The scenario drives
# the real ``EffectRunner`` methods with a stand-in backend whose ``query``
# clears ``runner._backend`` the first time it is called, which is exactly what
# a main-thread disconnect does and is deterministic where a real thread race is
# not. A run that raises ``AttributeError`` has reproduced the defect.
#
# Subtests:
#   A. stop_server_survives_a_disconnect_mid_wait
#   B. stop_server_reports_the_transport_not_the_addon
#   C. stop_server_still_returns_early_when_already_disconnected
#   D. the_wait_loop_does_not_reread_the_attribute
#   E. every_method_taking_a_backend_keeps_self_first
#   F. every_rig_call_of_these_methods_matches_the_signature

from __future__ import annotations

import ast
import importlib.util
import os
import re

from . import _runner as r


# No Blender and no solver: the subject is the runner's own threading contract,
# and it behaves identically under every backend.
BACKENDS = ("real",)


def _load_effect_runner():
    """Load ``core/effect_runner.py`` without importing the add-on package.

    The package ``__init__`` imports ``bpy``; this scenario runs outside
    Blender. Loading the file on its own also pins the gate to THIS tree rather
    than to whichever add-on a shared extension symlink points at.
    """
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(here, "core", "effect_runner.py")
    spec = importlib.util.spec_from_file_location("ppf_effect_runner_src", path)
    return path, spec


class _StandInBackend:
    """A backend that disconnects itself the first time it is queried.

    It reproduces the main-thread disconnect deterministically: the real one
    lands at an arbitrary point inside the wait loop, and a test that waited for
    a real thread to interleave would be a flake.
    """

    backend_type = "ssh"
    server_port = 9090
    container = "ppf-contact-solver"

    def __init__(self, on_first_query) -> None:
        self._on_first_query = on_first_query
        self.queries = 0
        self.exec_calls = 0

    def exec_command(self, command, shell=False, timeout=None):
        self.exec_calls += 1
        return {"exit_code": 0, "stdout": [], "stderr": []}

    def query(self, request, project, chunk):
        self.queries += 1
        if self.queries == 1 and self._on_first_query is not None:
            # What `_do_disconnect` does, from the other thread.
            self._on_first_query()
        # Still answering, so the loop makes another pass and would re-read
        # the attribute if it kept one.
        return ({}, True)

    def is_alive(self):
        return True

    def stop_server(self):
        return None

    def disconnect(self):
        return None


def _source() -> str:
    path, _ = _load_effect_runner()
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _method_body(source: str, name: str) -> str:
    start = source.index(f"    def {name}(")
    end = source.index("\n    def ", start + 1)
    return source[start:end]


def run(ctx: r.ScenarioContext) -> dict:
    checks: dict = {}
    source = _source()

    # ---- D: the shape, read off the source ---------------------------------
    # The three windows the defect lived in must each bind once and then use
    # the local. This is the cheap check and it is the one that fails first if
    # someone reintroduces the attribute read.
    shape: dict = {}
    for name in ("_do_stop_server", "_do_launch_server"):
        body = _method_body(source, name)
        after_guard = body.split("backend = self._backend", 1)
        shape[name] = {
            "binds_once": len(after_guard) == 2,
            # Comments may still NAME the attribute; code must not read it.
            "rereads": sum(
                1 for line in (after_guard[1].splitlines() if len(after_guard) == 2 else [])
                if "self._backend" in line and not line.strip().startswith("#")
            ),
        }
    checks["the_wait_loop_does_not_reread_the_attribute"] = {
        "ok": all(v["binds_once"] and v["rereads"] == 0 for v in shape.values()),
        "details": shape,
    }

    # ---- E: the parameter ORDER, which a compile cannot check ---------------
    # Threading the backend through as a parameter is what the fix does, and
    # inserting it ahead of `self` is valid Python that binds the runner to
    # `backend` and the backend to `self`. It shipped exactly that way and
    # surfaced only at run time, on the launch path, as
    # "'EffectRunner' object has no attribute 'backend_type'" - a path no
    # subtest here drives. So the order is asserted directly, for every method
    # that takes one, rather than relying on a caller to exercise it.
    order: dict = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef):
            continue
        for fn in node.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = [a.arg for a in fn.args.args]
            if "backend" in names:
                order[fn.name] = names
    checks["every_method_taking_a_backend_keeps_self_first"] = {
        "ok": bool(order) and all(v and v[0] == "self" for v in order.values()),
        "details": order or {"note": "no method takes a backend, which is "
                                     "itself a change worth failing on"},
    }

    # ---- F: the rig's OWN calls, checked against the signature --------------
    # Scenarios call these private methods directly, so changing one's
    # parameters breaks them and nothing in the add-on reports it: the failure
    # arrives as a TypeError inside whichever scenario exercises the path. That
    # shipped. `bl_solver_gpu_select` calls `_wait_for_native_server` twice and
    # monkeypatches a stand-in for it, and a signature change left all three
    # stale while the two scenarios written FOR the change passed.
    #
    # SCANNED AS TEXT, NOT AS AN AST, and that is the whole reason this check
    # works. A `bl_*` scenario builds its driver as a STRING and sends it to
    # Blender to execute, so the calls are string contents rather than code:
    # an `ast.walk` over the scenario finds zero of them, and a gate written
    # that way passes over the very defect it was added for. The CI traceback
    # said `File "<string>", line 443`, which is the tell.
    sig: dict = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == "EffectRunner":
            for fn in node.body:
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names = [a.arg for a in fn.args.args]
                    n_default = len(fn.args.defaults)
                    # `self` is bound by the attribute access, so a call site
                    # supplies everything after it.
                    sig[fn.name] = (len(names) - 1 - n_default, len(names) - 1)

    watched = ("_wait_for_native_server", "_launch_native_server",
               "_do_stop_server", "_do_launch_server", "_do_connect")

    def positional(arglist: str) -> int:
        """How many POSITIONAL arguments a call's text passes.

        Split at top-level commas only, so a nested call or a tuple does not
        read as several arguments, and a `name=` argument is a keyword.
        """
        depth, current, parts = 0, "", []
        for ch in arglist:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(current); current = ""
            else:
                current += ch
        parts.append(current)
        count = 0
        for part in parts:
            text = part.strip()
            if not text:
                continue
            if re.match(r"^[A-Za-z_][A-Za-z_0-9]*\s*=[^=]", text):
                continue
            count += 1
        return count

    mismatches = []
    here = os.path.dirname(os.path.abspath(__file__))
    for name in sorted(os.listdir(here)):
        if not name.endswith(".py") or name == os.path.basename(__file__):
            continue
        text = open(os.path.join(here, name), encoding="utf-8").read()
        for method in watched:
            if method not in sig:
                continue
            low, high = sig[method]
            for m in re.finditer(re.escape(method) + r"\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", text):
                line = text[: m.start()].count("\n") + 1
                # A definition or an assignment of the attribute is not a call.
                lead = text[max(0, m.start() - 40): m.start()]
                if "def " in lead.split("\n")[-1]:
                    continue
                given = positional(m.group(1))
                if not (low <= given <= high):
                    mismatches.append({
                        "scenario": name, "line": line, "method": method,
                        "given": given, "accepts": f"{low}..{high}",
                    })
    checks["every_rig_call_of_these_methods_matches_the_signature"] = {
        "ok": not mismatches,
        "details": mismatches or {
            "signatures": {k: sig[k] for k in watched if k in sig},
            "note": "scanned as text because bl_* drivers are string templates",
        },
    }

    # ---- A, B, C: drive the real method -------------------------------------
    # The module imports the add-on package at import time, so the method is
    # exercised by binding it to a minimal stand-in rather than constructing a
    # whole EffectRunner. What is under test is the method's own body.
    body = _method_body(source, "_do_stop_server")

    class _Stub:
        """The attributes `_do_stop_server` touches, and nothing else."""

        def __init__(self) -> None:
            self._backend = None
            self._project_name = "rig"
            self._chunk_size = 1 << 16
            self.last_kill_report = None
            self._response_cache = type(
                "_C", (), {"clear": lambda self: None}
            )()
            self._engine = type(
                "_E", (), {"dispatch": lambda self, event: None}
            )()

    # The body is compiled on its own with the names it closes over supplied,
    # so the gate reads the shipping source rather than a transcription of it.
    namespace = {
        "NATIVE_BACKENDS": {"linux_native": "Linux", "mac_native": "macOS",
                            "win_native": "Windows"},
        "kill_remote_server": lambda *a, **k: None,
        "time": __import__("time"),
        "ErrorOccurred": type("ErrorOccurred", (), {"__init__": lambda s, **k: None}),
        "ServerStopped": type("ServerStopped", (), {"__init__": lambda s, **k: None}),
        "getattr": getattr,
    }
    compiled = compile(
        "class _Host:\n" + body + "\n", "effect_runner.py", "exec"
    )
    exec(compiled, namespace)  # noqa: S102
    method = namespace["_Host"]._do_stop_server

    # A: a disconnect on the first query must not raise.
    stub = _Stub()
    stub._backend = _StandInBackend(on_first_query=lambda: setattr(stub, "_backend", None))
    raised = None
    try:
        method(stub)
    except Exception as exc:  # noqa: BLE001 - the verdict
        raised = f"{type(exc).__name__}: {exc}"
    checks["stop_server_survives_a_disconnect_mid_wait"] = {
        "ok": raised is None,
        "details": {"raised": raised, "queries": stub._backend.queries
                    if stub._backend else "backend cleared, as intended"},
    }

    # B: and the failure it CAN report is about the TRANSPORT, never a defect
    # in the add-on. Written as "no programming error" rather than "no
    # AttributeError": the original was an AttributeError, but a half-applied
    # fix raises NameError from the same window, and a check naming only the
    # one symptom passes the other. Measured: a partial revert of the binding
    # produced `NameError: name 'backend' is not defined` and an
    # AttributeError-only check reported pass.
    programming_errors = ("AttributeError", "NameError", "TypeError",
                          "UnboundLocalError")
    checks["stop_server_reports_the_transport_not_the_addon"] = {
        "ok": raised is None or not any(e in raised for e in programming_errors),
        "details": {"raised": raised, "rejected": list(programming_errors)},
    }

    # C: the entry guard still works, so nothing runs with no backend at all.
    idle = _Stub()
    idle._backend = None
    guard_raised = None
    try:
        method(idle)
    except Exception as exc:  # noqa: BLE001
        guard_raised = f"{type(exc).__name__}: {exc}"
    checks["stop_server_still_returns_early_when_already_disconnected"] = {
        "ok": guard_raised is None and idle.last_kill_report is None,
        "details": {"raised": guard_raised,
                    "kill_report": idle.last_kill_report},
    }

    return r.report_named_checks(checks)
