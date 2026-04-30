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


DRIVER_LIB = r"""
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

    def encode_payload(self):
        return (self.encoder_mesh.encode_obj(bpy.context),
                self.encoder_param.encode_param(bpy.context))

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
            raise RuntimeError(f"build failed: {s.error!r}")

    def run_and_wait(self, *, timeout=90.0):
        self.com.run()
        return self._await_running_then_ready(timeout=timeout)

    def resume_and_wait(self, *, timeout=90.0):
        self.com.resume()
        return self._await_running_then_ready(timeout=timeout)

    def _await_running_then_ready(self, *, timeout):
        # 0.05s poll cadence: with PPF_EMULATED_STEP_MS=100 a single
        # solver step's RUNNING phase can be ~150 ms wall-clock, which
        # the previous 0.3s sleep often missed entirely. We also treat
        # ``state.frame > 0`` as conclusive evidence the solver did
        # transition through RUNNING, even if the poll cadence skipped
        # over the phase label.
        saw_running = False
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.facade.engine.dispatch(self.events.PollTick())
            self.facade.tick()
            s = self.facade.engine.state
            if s.solver.name == "RUNNING" or s.frame > 0:
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
"""
