# File: scenarios/_driver_lib.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Source-string library for Blender-side scenario drivers.
#
# Drivers run as Python source ``exec()``'d inside Blender by the
# bootstrap (see ``debug/blender_harness.py``). They share a lot of
# pipeline boilerplate (connect, build, run, drain, fetch, snapshot
# PC2) that we don't want to copy-paste into every scenario.
#
# This module exposes :data:`DRIVER_LIB`, a Python source fragment
# that every driver template prepends with
# ``str.replace("<<DRIVER_LIB>>", DRIVER_LIB)``. The fragment
# defines a ``DriverHelpers`` class; scenarios instantiate it and
# call ``dh.connect_local(...)`` etc.
#
# :data:`BUILD_FAILURE_LIB` is the leading slice of that fragment and
# is also prependable on its own. It defines ``build_failure_message``
# as a plain function of ``(facade, com)``, so a driver that raises on
# solver=FAILED names the same cause whether or not it carries the rest
# of the library (``_pin_fidelity_common`` drives the pipeline directly
# and has no ``DriverHelpers`` instance to reach a method through).
#
# We deliberately do NOT bundle this as a real importable module
# inside Blender. Drivers run with ``exec_globals`` that already
# carry ``bpy``, ``pkg``, and ``result``; injecting helpers via
# source string keeps the import surface clean (no sys.path tricks)
# and keeps the helpers visible to syntax-aware tooling because they
# stay in real .py files on disk.
#
# Internal note: helper docstrings here use single-line ``#`` comments
# rather than triple-quoted docstrings because the whole library is
# embedded in a raw string -- triple-quotes inside ``r\"\"\"...\"\"\"``
# can't be escaped without changing semantics.

from __future__ import annotations


BUILD_FAILURE_LIB = r"""
import os


def clip_log_line(line, limit):
    # One log line, bounded, saying so when it is cut. A build worker can
    # emit a single enormous line (a repr of a whole scene, a traceback
    # whose newlines arrived as \r), and a message built from these lines
    # lands in result["errors"][0] of the scenario report, so the bound
    # belongs on every line rather than on the byte window alone.
    if len(line) <= limit:
        return line
    return line[:limit] + f" ...<{len(line) - limit} chars elided>"


def server_log_tail(*, lines=15, tail_bytes=65536, max_line_chars=300,
                    max_total_chars=4000):
    # Tail of the rig server's own log, as (lines, directory, saw_build).
    # Each worker owns <workspace>/{server,project,probe} and the
    # orchestrator runs ppf-cts-server with CWD=<workspace>/server,
    # capturing its log4rs console output as stdout.log there. That is
    # where a build worker's failure is legible: executor/build.rs
    # drains the worker's stderr into the server log tagged
    # [BUILD stderr], and its spawn line names the interpreter it ran,
    # which is the whole answer when a dependency is missing. Prefer
    # those lines, fall back to the raw tail when none are present, and
    # report which of the two through ``saw_build`` plus a per-file
    # heading, so no line of the report claims to be build worker output
    # when it is not.
    probe_dir = (os.environ.get("PPF_DEBUG_PROBE_DIR", "")
                 or globals().get("PROBE_DIR", ""))
    if not probe_dir:
        return [], "", False
    server_dir = os.path.join(os.path.dirname(probe_dir), "server")
    sections = []
    saw_build = False
    for name in ("stdout.log", "stderr.log"):
        path = os.path.join(server_dir, name)
        try:
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                if size > tail_bytes:
                    f.seek(size - tail_bytes)
                blob = f.read()
        except FileNotFoundError:
            continue
        except OSError as exc:
            # Say the log could not be read; do not let that stand in
            # for the build failure being reported.
            sections.append(
                (f"{name}: <unreadable: {clip_log_line(str(exc), 200)}>",
                 [], 0))
            continue
        text = blob.decode("utf-8", "replace")
        window = ""
        if size > tail_bytes:
            # The read window itself drops everything older, and the
            # heading says so: a count of elided lines can only speak for
            # lines this function has seen.
            window = f", last {tail_bytes}B of the file"
            # The seek lands mid-line, so the first line is a fragment.
            # Drop it only when something survives: a window holding one
            # long line plus the file's trailing newline partitions into
            # that line and "", and taking the remainder there would
            # report an empty tail for a log that is not empty.
            _, sep, rest = text.partition("\n")
            if sep and rest.strip():
                text = rest
        got = [clip_log_line(ln.rstrip(), max_line_chars)
               for ln in text.splitlines() if ln.strip()]
        build = [ln for ln in got if "[BUILD" in ln]
        # Each section carries the count it had BEFORE the ``lines``
        # slice, so the elision marker below can name every line the
        # report is missing rather than only the ones the char budget cut.
        if build:
            saw_build = True
            sections.append((f"{name} ([BUILD] lines{window}):",
                             build[-lines:], len(build)))
        elif got:
            sections.append((f"{name} (tail{window}, no [BUILD] lines):",
                             got[-lines:], len(got)))
    if not sections:
        return [], server_dir, False

    # Spend the total budget on the NEWEST lines of each section: they
    # sit nearest the failure. Every section keeps at least its last
    # line, so a budget smaller than one line still reports something.
    budget = max_total_chars // len(sections)
    out = []
    for heading, picked, available in sections:
        kept = []
        used = 0
        for line in reversed(picked):
            if kept and used + len(line) > budget:
                break
            kept.append(line)
            used += len(line)
        kept.reverse()
        if len(kept) < available:
            kept.insert(
                0, f"<{available - len(kept)} earlier line(s) elided>")
        out.append(heading)
        out.extend(kept)
    return out, server_dir, saw_build


def build_failure_message(facade, com, *, prefix="build failed",
                          max_cause_chars=600, max_extra_chars=300,
                          max_message_chars=8000):
    # A build failure must name a cause. The scenario name alone does not
    # tell a build worker whose Python lacks scipy / pyvista / tetgen
    # apart from a malformed scene, and both arrive here as solver=FAILED.
    #
    # Four places can hold that cause, and they are filled by four
    # different writers, so one of them being set says nothing about
    # whether the others are redundant:
    #
    #   - ``state.server_error`` is where ``ServerPolled``
    #     (core/transitions.py) files the server's own text, on every
    #     branch it takes.
    #   - ``state.error`` is client-side. The same transition's ordinary
    #     branch rebuilds the state with ``error=""``, so it is empty
    #     exactly when the server reported the failure; its build-desync
    #     branch sets it to a fixed addon sentence about the desync WHILE
    #     also filing the server's text, and ``ErrorOccurred`` /
    #     ``FetchFailed`` set it from a client-side transport failure that
    #     says nothing about the build. On none of those branches does it
    #     duplicate what the server reported.
    #   - the raw response is the fallback for a message the reducer
    #     dropped.
    #   - the rig server's log is where the build worker's OWN stderr
    #     arrives (tagged [BUILD stderr]); no addon field ever holds it,
    #     so it adds information whenever it has any.
    #
    # So the head names the first carrier holding text, every other
    # carrier holding DIFFERENT text follows it, and the log is attached
    # unconditionally. The head order puts the client-side field first to
    # match the addon's own top-of-panel ERROR label; nothing is lost by
    # that choice because the rest is reported directly below.
    s = facade.engine.state
    notes = []
    try:
        response = dict(com.info.response)
    except Exception as exc:
        # A broken response cache is reported beside the build
        # failure, never in place of it.
        response = {}
        notes.append("response unavailable ({}: {})".format(
            type(exc).__name__, clip_log_line(str(exc), 200)))
    carriers = [
        ("state.error", (s.error or "").strip()),
        ("state.server_error", (s.server_error or "").strip()),
        ("server response", str(response.get("error", "") or "").strip()),
    ]
    reported = [(label, text) for label, text in carriers if text]

    # Gather the detail before the context line below, which reports the
    # notes that gathering can add.
    detail = []
    if reported:
        head_label, head_text = reported[0]
        head = (f"{prefix}: {clip_log_line(head_text, max_cause_chars)} "
                f"[{head_label}]")
        # Two carriers holding the same string relay one fact, so the
        # copy is dropped; two holding different strings are two facts,
        # and which of them names the real cause is not decidable here.
        shown = {head_text}
        for label, text in reported[1:]:
            if text in shown:
                continue
            shown.add(text)
            detail.append(
                f"  also {label}: {clip_log_line(text, max_extra_chars)}")
    else:
        head = (f"{prefix}: no cause reported (state.error, "
                "state.server_error and the server's last response "
                "are all empty)")

    try:
        tail, server_dir, saw_build = server_log_tail()
    except Exception as exc:
        # Same rule as the response cache above: a failure while
        # gathering detail is reported beside the build failure.
        tail, server_dir, saw_build = [], "", False
        notes.append("server log unreadable ({}: {})".format(
            type(exc).__name__, clip_log_line(str(exc), 200)))
    if tail:
        detail.append(f"  server log in {clip_log_line(server_dir, 200)}:")
        detail.extend("    " + line for line in tail)
        if not saw_build:
            detail.append(
                "  (that log carries no [BUILD] line; the build "
                "worker's stderr reaches it tagged [BUILD stderr])")
    elif server_dir:
        detail.append(
            "  no build worker output under "
            f"{clip_log_line(server_dir, 200)}; the worker's stderr "
            "reaches the server log tagged [BUILD stderr]")
    else:
        detail.append(
            "  build worker output not reachable "
            "(PPF_DEBUG_PROBE_DIR unset, so the rig worker "
            "directory is unknown); the worker's stderr reaches "
            "the server log tagged [BUILD stderr]")

    context = [f"solver={s.solver.name}", f"activity={s.activity.name}"]
    if s.violations:
        context.append(f"violations={len(s.violations)}")
    message = "\n".join(
        [head, "  (" + ", ".join(context + notes) + ")"] + detail)
    # Backstop. Every part above carries its own bound: the cause and the
    # extra carriers by their clip limits, the log by ``max_total_chars``
    # over at most two sections of at most ``lines`` lines each. Feeding
    # megabyte-sized carriers and log lines swept across every length that
    # packs a section's budget tops out under 6000 characters, so a message
    # that reaches this cap means one of those bounds is wrong. Cut the
    # end, which holds the OLDEST log line, and keep the head, which names
    # the cause.
    if len(message) > max_message_chars:
        message = (message[:max_message_chars]
                   + f"\n<message capped, "
                     f"{len(message) - max_message_chars} chars elided>")
    return message
"""


DRIVER_LIB = BUILD_FAILURE_LIB + r"""
import bpy, os, struct, time
import numpy as np


class DriverHelpers:
    # Pipeline helpers reused across Blender-driven scenarios.

    def __init__(self, pkg, result):
        self.pkg = pkg
        self.result = result
        self.facade = __import__(pkg + ".core.facade",
                                 fromlist=["engine", "tick"])
        self.client = __import__(pkg + ".core.client",
                                 fromlist=["communicator", "apply_animation"])
        self.events = __import__(pkg + ".core.events",
                                 fromlist=["PollTick", "QueryRequested",
                                           "FetchRequested",
                                           "SaveAndQuitRequested",
                                           "ResumeRequested",
                                           "BuildPipelineRequested",
                                           "RunRequested",
                                           "AbortRequested"])
        self.encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                                       fromlist=["encode_obj"])
        self.encoder_param = __import__(pkg + ".core.encoder.params",
                                        fromlist=["encode_param"])
        self.api = __import__(pkg + ".ops.api", fromlist=["solver"])
        self.groups = __import__(pkg + ".models.groups",
                                 fromlist=["get_addon_data"])
        self.com = self.client.communicator

    # -- logging --

    def log(self, msg):
        self.result.setdefault("phases", []).append(
            (round(time.time(), 3), msg)
        )

    def record(self, name, ok, details):
        self.result.setdefault("checks", {})[name] = {
            "ok": bool(ok), "details": details,
        }

    def record_subtest(self, name, ok, details):
        self.result.setdefault("subtests", {})[name] = {
            "ok": bool(ok), "details": details,
        }

    # -- scene factory --

    def reset_scene_to_pinned_plane(self, *, name="Mesh", pin_group="AllPin"):
        # Wipe the scene and create a unit plane with every vertex in a
        # single pin vertex group. Saves the .blend so subsequent ID
        # writes that require a saved file (encoder, fcurve sync) work.
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
        plane = bpy.context.active_object
        plane.name = name
        n = len(plane.data.vertices)
        vg = plane.vertex_groups.new(name=pin_group)
        vg.add(list(range(n)), 1.0, "REPLACE")
        return plane

    def save_blend(self, probe_dir, basename):
        path = os.path.join(os.path.dirname(probe_dir), basename)
        bpy.ops.wm.save_as_mainfile(filepath=path)
        return path

    def configure_state(self, *, project_name, frame_count, frame_rate=100,
                        step_size=0.01, gravity=(0.0, 0.0, 0.0)):
        # Apply the standard test-rig state defaults so emulated runs
        # are short, deterministic, and free of contact / wind / gravity
        # side effects unless the caller asks for them.
        root = self.groups.get_addon_data(bpy.context.scene)
        s = root.state
        s.project_name = project_name
        s.frame_count = frame_count
        s.frame_rate = frame_rate
        s.step_size = step_size
        s.disable_contact = True
        s.gravity_3d = gravity
        s.air_density = 0.0
        s.wind_strength = 0.0
        return root

    # -- connection / pipeline --

    def connect_local(self, *, local_path, server_port, project_name,
                      timeout=30.0):
        root = self.groups.get_addon_data(bpy.context.scene)
        root.ssh_state.server_type = "LOCAL"
        root.ssh_state.local_path = local_path
        root.ssh_state.docker_port = server_port
        self.com.set_project_name(project_name)
        self.com.connect_local(local_path, server_port=server_port)

        deadline = time.time() + timeout
        while time.time() < deadline:
            self.facade.engine.dispatch(self.events.PollTick())
            self.facade.tick()
            s = self.facade.engine.state
            if s.phase.name == "ONLINE" and s.server.name == "RUNNING":
                return
            time.sleep(0.2)
        raise RuntimeError(
            f"server never reached RUNNING within {timeout}s "
            f"(phase={self.facade.engine.state.phase.name}, "
            f"server={self.facade.engine.state.server.name})"
        )

    def connect_win_native(self, *, local_path, server_port, project_name,
                           timeout=30.0):
        # Windows-native counterpart to connect_local. The rig owns the
        # ppf-cts-server (PPF_WIN_NATIVE_NO_SPAWN=1 in the worker env), so
        # the addon attaches to the server already listening on
        # server_port instead of spawning its own. win_native_path points
        # at the repo root; resolve_win_native_root walks up to it if a
        # subdirectory is given.
        root = self.groups.get_addon_data(bpy.context.scene)
        root.ssh_state.server_type = "WIN_NATIVE"
        root.ssh_state.win_native_path = local_path
        root.ssh_state.docker_port = server_port
        self.com.set_project_name(project_name)
        self.com.connect_win_native(local_path, server_port)

        deadline = time.time() + timeout
        while time.time() < deadline:
            self.facade.engine.dispatch(self.events.PollTick())
            self.facade.tick()
            s = self.facade.engine.state
            if s.phase.name == "ONLINE" and s.server.name == "RUNNING":
                return
            time.sleep(0.2)
        raise RuntimeError(
            f"win_native server never reached RUNNING within {timeout}s "
            f"(phase={self.facade.engine.state.phase.name}, "
            f"server={self.facade.engine.state.server.name})"
        )

    def connect(self, *, local_path, server_port, project_name, timeout=30.0):
        # Platform-appropriate connect: WIN_NATIVE on Windows, LOCAL
        # elsewhere. Both attach to the rig-owned server on server_port,
        # so a single cross-platform scenario runs on the emulated
        # macOS/Linux jobs (LOCAL) and the real-GPU Windows job
        # (WIN_NATIVE) without branching in the scenario body.
        import sys as _sys
        if _sys.platform.startswith("win"):
            return self.connect_win_native(
                local_path=local_path, server_port=server_port,
                project_name=project_name, timeout=timeout)
        return self.connect_local(
            local_path=local_path, server_port=server_port,
            project_name=project_name, timeout=timeout)

    def connect_ssh(self, *, host, port, username, key_path, remote_path,
                    server_port, project_name, timeout=60.0):
        # Connect to a REMOTE ppf-cts-server over SSH (server_type
        # CUSTOM). Unlike LOCAL/WIN_NATIVE this does not attach to the
        # rig-owned local server; it drives the addon's paramiko backend
        # to a server on another machine (a GPU box). The remote server
        # is expected to be already listening on server_port at the
        # remote loopback -- pre-start it so this call reaches
        # phase=ONLINE, server=RUNNING without the start_server dance.
        # ``remote_path`` is the remote repo root holding
        # target/release/ppf-cts-server. paramiko must be importable in
        # Blender's Python.
        root = self.groups.get_addon_data(bpy.context.scene)
        root.ssh_state.server_type = "CUSTOM"
        root.ssh_state.host = host
        root.ssh_state.port = port
        root.ssh_state.username = username
        root.ssh_state.key_path = key_path
        root.ssh_state.ssh_remote_path = remote_path
        # docker_port is the shared "server port on the remote loopback"
        # field for every backend (see models/defaults).
        root.ssh_state.docker_port = server_port
        self.com.set_project_name(project_name)
        self.com.connect_ssh(host=host, port=port, username=username,
                             key_path=key_path, path=remote_path,
                             container=None, server_port=server_port)

        deadline = time.time() + timeout
        while time.time() < deadline:
            self.facade.engine.dispatch(self.events.PollTick())
            self.facade.tick()
            s = self.facade.engine.state
            if s.phase.name == "ONLINE" and s.server.name == "RUNNING":
                return
            time.sleep(0.3)
        s = self.facade.engine.state
        raise RuntimeError(
            f"ssh server never reached ONLINE/RUNNING within {timeout}s "
            f"(phase={s.phase.name}, server={s.server.name}, "
            f"error={getattr(s, 'error', None)!r})"
        )

    def encode_payload(self):
        return (self.encoder_mesh.encode_obj(bpy.context),
                self.encoder_param.encode_param(bpy.context))

    @staticmethod
    def decode_addon_blob(blob):
        # Decode the bytes returned by ``encode_obj`` / ``encode_param``.
        # The producers flipped from pickle to a CBOR envelope
        # ({version, kind, payload}) during the Rust migration; old
        # on-disk saves are still pickle. Pickle frames always start
        # with 0x80 (PROTO opcode), so the first byte is enough to
        # dispatch. (Comment-style on purpose: this method lives
        # inside the raw-string DRIVER_LIB, so triple-quotes here
        # would close the outer literal.)
        import pickle as _pickle
        if blob and blob[0] == 0x80:
            return _pickle.loads(blob)
        import cbor2  # type: ignore
        env = cbor2.loads(blob)
        if isinstance(env, dict) and "payload" in env:
            return env["payload"]
        return env

    def build_and_wait(self, data_bytes, param_bytes, message,
                       *, timeout=90.0):
        # Distinct from naive solver-state polling: a previous run can
        # leave solver=READY before the new build's transitions even
        # fire, so we require activity to return to IDLE first.
        self.com.build_pipeline(data=data_bytes, param=param_bytes,
                                message=message)
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.facade.engine.dispatch(self.events.PollTick())
            self.facade.tick()
            s = self.facade.engine.state
            if (s.activity.name == "IDLE"
                    and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
                break
            time.sleep(0.3)
        s = self.facade.engine.state
        if s.solver.name == "FAILED":
            raise RuntimeError(build_failure_message(self.facade, self.com))

    def run_and_wait(self, *, timeout=90.0):
        self.com.run()
        return self._await_running_then_ready(timeout=timeout)

    def resume_and_wait(self, *, timeout=90.0):
        self.com.resume()
        return self._await_running_then_ready(timeout=timeout)

    def _await_running_then_ready(self, *, timeout):
        # 0.05s poll cadence: with PPF_EMULATED_STEP_MS=100 a single
        # solver step's RUNNING phase can be ~150 ms wall-clock, which
        # the previous 0.3s sleep often missed entirely. We treat
        # ``frame growth since entry`` as conclusive evidence the solver
        # did transition through RUNNING, even if the poll cadence
        # skipped the phase label. ``> start_frame`` (not ``> 0``) so
        # a stale tail from a prior run cannot trigger saw_running
        # before the new run has actually advanced.
        start_frame = self.facade.engine.state.frame
        saw_running = False
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.facade.engine.dispatch(self.events.PollTick())
            self.facade.tick()
            s = self.facade.engine.state
            if s.solver.name == "RUNNING" or s.frame > start_frame:
                saw_running = True
            if saw_running and s.solver.name in ("READY", "RESUMABLE"):
                return saw_running
            if s.solver.name == "FAILED":
                return saw_running
            time.sleep(0.05)
        return saw_running

    def force_frame_query(self, *, expected_frames, timeout=30.0):
        # PollTick stops emitting DoQuery once solver leaves the
        # sim-running set, so state.frame can lag behind the actual
        # final count after a run completes. QueryRequested has only
        # the ``not state.busy`` guard, so we drive it until state.frame
        # catches up.
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.facade.engine.state.busy:
                self.facade.engine.dispatch(
                    self.events.QueryRequested(request={}, message=""))
            self.facade.tick()
            if self.facade.engine.state.frame >= expected_frames:
                return
            time.sleep(0.2)

    def settle_idle(self, *, timeout=15.0, stable_for=1.0):
        # Wait for activity=IDLE for ``stable_for`` consecutive seconds.
        # Drains in-flight live-fetch / query side effects so the next
        # dispatch starts from a known clean state.
        deadline = time.time() + timeout
        stable_since = None
        while time.time() < deadline:
            self.facade.engine.dispatch(self.events.PollTick())
            self.facade.tick()
            if self.facade.engine.state.activity.name == "IDLE":
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= stable_for:
                    return True
            else:
                stable_since = None
            time.sleep(0.1)
        return False

    def fetch_and_drain(self, *, fetch_timeout=60.0, drain_timeout=30.0):
        # Reset _fetched (forces re-download of all frames so the
        # live-fetch race we documented in memory can't drop any),
        # dispatch FetchRequested, drain the modal in-driver via
        # direct apply_animation calls (the modal can't fire while we
        # hold the main thread).
        self.facade.runner.clear_fetched_frames()
        self.com.fetch()
        deadline = time.time() + fetch_timeout
        while time.time() < deadline:
            self.facade.engine.dispatch(self.events.PollTick())
            self.facade.tick()
            s = self.facade.engine.state
            if s.activity.name == "APPLYING":
                break
            if (s.activity.name == "IDLE"
                    and s.solver.name in ("READY", "RESUMABLE")):
                break
            time.sleep(0.2)

        runner = self.facade.runner
        deadline = time.time() + drain_timeout
        applied = total = 0
        while time.time() < deadline:
            self.facade.tick()
            self.client.apply_animation()
            with runner._anim_lock:
                queued = len(runner._anim_frames)
                applied = runner._anim_applied
                total = runner._anim_total
            if queued == 0 and total > 0 and applied >= total:
                return applied, total
            time.sleep(0.1)
        return applied, total

    # -- artifact lookup --

    def find_pc2_for(self, obj):
        for mod in obj.modifiers:
            if mod.type == "MESH_CACHE" and mod.cache_format == "PC2":
                return bpy.path.abspath(mod.filepath) if mod.filepath else None
        return None

    def has_mesh_cache(self, obj):
        return any(m.type == "MESH_CACHE" for m in obj.modifiers)

    def read_pc2(self, path):
        with open(path, "rb") as f:
            f.read(12)
            version, n_verts = struct.unpack("<II", f.read(8))
            f.read(8)
            (n_samples,) = struct.unpack("<I", f.read(4))
            body = f.read(n_samples * n_verts * 3 * 4)
        return np.frombuffer(body, dtype="<f4").reshape(n_samples, n_verts, 3)

    # -- staged-modal operator probe --

    def staged_stub(self, op_cls):
        # Fresh StagedStub bound to ``op_cls`` (SOLVER_OT_Run /
        # SOLVER_OT_Transfer) and the addon's AsyncOperator machinery.
        async_op = __import__(self.pkg + ".core.async_op",
                              fromlist=["AsyncOperator", "StageAbort"])
        return StagedStub(async_op, op_cls)


class StagedStub:
    # Stand-in operator ``self`` for the staged-modal action operators
    # (Transfer / Run). Their execute() defers the real work -- scene
    # encode, click-time drift checks, the engine dispatch -- to
    # AsyncOperator.start_stages, which runs one (label, fn) stage per
    # modal TIMER tick. The rig driver holds the main thread so no TIMER
    # fires, so this stub borrows the real stage machinery and exposes
    # drain_stages() to pump every stage synchronously, reproducing what
    # the modal does across ticks. report() is captured for assertions
    # and setup_modal() is faked so no real window-manager timer /
    # modal_handler is registered.
    #
    # execute() builds its stage list from ``self._stage_*`` bound methods
    # that live on the real operator class, so __getattr__ delegates any
    # attribute this stub does not define to ``op_cls``, bound to ``self``
    # -- the operator's own execute()/_stage_*()/is_complete() then run
    # with self=stub while resolving their module globals normally.
    auto_redraw = False
    _stages = None
    _stage_index = 0
    _timer = None

    def __init__(self, async_op_mod, op_cls):
        self._aop = async_op_mod
        self._op_cls = op_cls
        self.captured = []
        self.modal_set_up = False
        self._mode = None
        self._start_time = 0.0

    def __getattr__(self, name):
        # Only reached for attributes not found on the instance/StagedStub.
        # Never delegate the dunders we set in __init__ (guards recursion).
        if name in ("_aop", "_op_cls"):
            raise AttributeError(name)
        attr = getattr(self._op_cls, name)
        if callable(attr):
            return attr.__get__(self, type(self))
        return attr

    def report(self, kind, msg):
        self.captured.append((tuple(kind), msg))

    def setup_modal(self, context):
        self.modal_set_up = True

    def error(self, needle=""):
        # First captured ERROR message containing ``needle`` (any if empty).
        for kind, msg in self.captured:
            if "ERROR" in kind and (not needle or needle in msg):
                return msg
        return ""

    # start_stages / _run_stage_tick / _end_stages / cleanup_modal are
    # called as ``self.<name>(...)`` from the operator's execute() and from
    # drain_stages; delegate each to the real unbound AsyncOperator method.
    def start_stages(self, context, stages):
        return self._aop.AsyncOperator.start_stages(self, context, stages)

    def _end_stages(self):
        return self._aop.AsyncOperator._end_stages(self)

    def cleanup_modal(self, context):
        return self._aop.AsyncOperator.cleanup_modal(self, context)

    def drain_stages(self, context):
        # Pump every staged (label, fn) the way AsyncOperator.modal would,
        # one per tick. Returns {"CANCELLED"} if a stage raised StageAbort
        # (the message is in ``captured``), or None when all stages ran
        # (the real modal would then fall through to the is_complete wait).
        tick = self._aop.AsyncOperator._run_stage_tick
        while self._stages is not None:
            res = tick(self, context)
            if res is None:
                return None
            if res == {"CANCELLED"}:
                return res
            # {"PASS_THROUGH"}: more stages remain; keep pumping.
        return None
"""
