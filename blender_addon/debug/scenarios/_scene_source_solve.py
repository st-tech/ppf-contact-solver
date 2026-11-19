# File: scenarios/_scene_source_solve.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# Shared by the "Take ... from Scene" scenarios that follow a setting from
# Blender into the solver. What the solver is given is read from the
# session's param.toml, the file the solver itself parses, which the build
# writes, so a build is enough for that. One short solve then shows what the
# solver did with it. Kept short on purpose: these run on every CI leg.

SOLVE_LIB = r"""
import os
import time as _time
import tomllib


def _find_key(node, name):
    # The first value stored under *name* anywhere in a parsed TOML tree.
    if isinstance(node, dict):
        if name in node:
            return node[name]
        for value in node.values():
            found = _find_key(value, name)
            if found is not None:
                return found
    return None


def build_and_read_params(dh, *, project, connected, label):
    # Encode the scene as it stands and build it, then return the gravity and
    # world_scaling the solver's param.toml holds.
    data_bytes, param_bytes = dh.encode_payload()
    if not connected:
        dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                   project_name=project)
    dh.build_and_wait(data_bytes, param_bytes, label + ":build")
    toml_path = os.path.join(str(dh.facade.engine.state.remote_root),
                             "session", "param.toml")
    with open(toml_path, "rb") as handle:
        parsed = tomllib.load(handle)
    gravity = _find_key(parsed, "gravity")
    scaling = _find_key(parsed, "world_scaling")
    return {
        "param_gravity": None if gravity is None else [float(v) for v in gravity],
        "param_world_scaling": None if scaling is None else float(scaling),
    }


def run_built_solve(dh, sheet, *, frames, label):
    # Run what was just built and return the cloth's mean displacement over
    # the run, in scene units and Blender axes. ONE SOLVE PER OBJECT: the PC2
    # is read as this run's, which holds only while no earlier solve of the
    # same object left one behind. The solver has to have finished, and the
    # PC2 has to hold every frame (it may also hold the start pose); frames already applied while the run went
    # on are not fetched again, so the file, not the fetch's counters, says
    # when the run is all there.
    dh.run_and_wait(timeout=120.0)
    deadline = _time.time() + 120.0
    while _time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        if dh.facade.engine.state.solver.name in ("RESUMABLE", "READY", "FAILED"):
            break
        _time.sleep(0.1)
    if dh.facade.engine.state.solver.name == "FAILED":
        raise RuntimeError(f"{label}: the solver failed")
    arr = None
    deadline = _time.time() + 60.0
    while _time.time() < deadline:
        dh.force_frame_query(expected_frames=frames - 1, timeout=2.0)
        dh.fetch_and_drain(fetch_timeout=3.0, drain_timeout=3.0)
        pc2 = dh.find_pc2_for(sheet)
        if pc2 and os.path.isfile(pc2):
            arr = dh.read_pc2(pc2)
            if arr.shape[0] >= frames:
                break
    if arr is None or arr.shape[0] < frames:
        raise RuntimeError(
            f"{label}: the PC2 holds {0 if arr is None else arr.shape[0]} of "
            f"{frames} frames")
    move = (arr[-1] - arr[0]).mean(axis=0)
    return {"move": [float(v) for v in move], "samples": int(arr.shape[0])}
"""
