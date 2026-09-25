# File: scenarios/bl_blender_examples.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Every script in `examples/blender/` built, transferred, solved and fetched
# through the add-on, the way an artist runs one.
#
# The scripts only author a scene: nothing else in the rig runs them, and
# `run_suite.py` sweeps the notebooks alone. Yet each one reaches the solver
# through the frontend's decoder (`frontend/_decoder_.py`), which turns the
# add-on's payload into `Scene` / `Object` / `PinHolder` calls, so a change
# to either side can break a published example with every other gate green.
# This drives each script at its own resolution and parameters and shortens
# only the run, to FRAMES (10) frames: what it checks is that the example still
# decodes, builds, runs and comes back, not how it looks at the end.
#
# `character-anim.py` plays back the Codim-IPC Rumba_Dancing OBJ sequence,
# which lives in the asset cache the `fitting` notebook warms
# (`~/.cache/ppf-cts/Codim-IPC`). Without it that example FAILS by name
# rather than being skipped, since a skip would report the suite green over
# one example fewer.
#
# Per example, one check:
#   `<stem>`: the build reaches READY, the run writes FRAMES frames, and every
#   object of a non-STATIC group comes back with a PC2 cache of at least two
#   samples, at least one of which moved.
#
# ON DEMAND: seven full-resolution examples, two of which tetrahedralize, is
# more than a default sweep should pay for coverage no other scenario needs.
# Run it by name, with a per-scenario timeout of an hour, before changing the
# decoder, the scene builder or an example script.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it on CUDA and on Metal.
BACKENDS = ("real",)

ON_DEMAND = True


_DRIVER_BODY = r"""
import glob
import importlib.util
import os
import time
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
EXAMPLES_DIR = os.path.join(LOCAL_PATH, "examples", "blender")
# The smallest Frame Count the add-on accepts (the property's min).
FRAMES = 10
RUMBA = os.path.expanduser(
    "~/.cache/ppf-cts/Codim-IPC/Projects/FEMShell/input/Rumba_Dancing"
)
# Arguments an example needs to run at all, and nothing that changes its scene.
ARGS = {
    "character-anim": {"obj_dir": RUMBA, "max_frames": FRAMES},
}


def _load(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(
        "rig_example_" + stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dynamic_objects(scene):
    for group in dh.groups.iterate_active_object_groups(scene):
        if group.object_type == "STATIC":
            continue
        for assigned in group.assigned_objects:
            if assigned.included and assigned.uuid:
                obj = dh.uuid_registry.get_object_by_uuid(assigned.uuid)
                if obj is not None:
                    yield group.object_type, obj


try:
    dh = DriverHelpers(pkg, result)
    dh.uuid_registry = __import__(pkg + ".core.uuid_registry",
                                  fromlist=["get_object_by_uuid"])
    pc2_mod = __import__(pkg + ".core.pc2",
                         fromlist=["get_pc2_path", "object_pc2_key_readonly"])
    solver = dh.api.solver
    connected = False
    paths = sorted(glob.glob(os.path.join(EXAMPLES_DIR, "*.py")))
    dh.log(f"examples={[os.path.basename(p) for p in paths]}")
    if not paths:
        raise RuntimeError(f"no example scripts under {EXAMPLES_DIR}")
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        started = time.time()
        try:
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.delete(use_global=False)
            solver.clear()
            if stem == "character-anim" and not os.path.isdir(RUMBA):
                raise FileNotFoundError(
                    f"{RUMBA} is absent: run the fitting notebook once on this "
                    "host to warm the Codim-IPC asset cache")
            _load(path).build(**ARGS.get(stem, {}))
            solver.param.frame_count = FRAMES
            dh.save_blend(PROBE_DIR, f"example_{stem}.blend")
            root = dh.groups.get_addon_data(bpy.context.scene)
            root.state.project_name = "blender_examples"
            if not connected:
                dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                           project_name="blender_examples")
                connected = True
            data_bytes, param_bytes = dh.encode_payload()
            dh.build_and_wait(data_bytes, param_bytes, f"example:{stem}",
                              timeout=1200.0)
            dh.run_and_wait(timeout=1200.0)
            dh.force_frame_query(expected_frames=FRAMES - 1, timeout=60.0)
            dh.settle_idle(timeout=30.0)
            dh.fetch_and_drain(fetch_timeout=300.0, drain_timeout=300.0)
            state = dh.facade.engine.state
            objects = {}
            moved = 0.0
            for kind, obj in _dynamic_objects(bpy.context.scene):
                pc2 = dh.find_pc2_for(obj)
                if not pc2 and obj.type == "CURVE":
                    # A curve carries no mesh-cache modifier: its cache is
                    # keyed by the object's UUID and a frame handler applies
                    # it to the control points.
                    pc2 = pc2_mod.get_pc2_path(
                        pc2_mod.object_pc2_key_readonly(obj))
                samples = 0
                if pc2 and os.path.isfile(pc2):
                    arr = dh.read_pc2(pc2)
                    samples = int(arr.shape[0])
                    if samples >= 2:
                        moved = max(moved, float(np.abs(arr[-1] - arr[0]).max()))
                objects[obj.name] = {"type": kind, "samples": samples}
            ok = (
                state.solver.name in ("READY", "RESUMABLE")
                and state.frame >= FRAMES - 1
                and bool(objects)
                and all(o["samples"] >= 2 for o in objects.values())
                and moved > 0.0
            )
            dh.record(stem, ok, {
                "solver": state.solver.name,
                "frame": state.frame,
                "max_displacement": moved,
                "objects": dict(list(objects.items())[:8]),
                "n_objects": len(objects),
                "seconds": round(time.time() - started, 1),
            })
        except Exception as exc:
            dh.record(stem, False, {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
                "seconds": round(time.time() - started, 1),
            })
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 3600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
