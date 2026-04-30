# File: scenarios/bl_static_op_anim.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# STATIC group with a UI-assigned MOVE_BY op drives PC2 playback. This
# scenario is intentionally focused on a single op kind so it can make
# an analytical equality assertion that would fail if MOVE_BY were
# silently broken. SPIN and SCALE composition is exercised separately
# by ``bl_pin_compose_*`` and ``bl_pin_spin_*``, which compare to
# ``frontend.FixedScene.time()`` rather than a closed-form expression.
#
# Static-op API: ops are PropertyGroup-based on
# ``AssignedObject.static_ops`` (a CollectionProperty of
# ``StaticOpItem``). The encoder serializes each op into
# ``info["static_ops"]`` (see ``core/encoder/mesh.py`` around line 376),
# converting Blender frame indices to solver time via
# ``t = (frame - 1) / fps``. The frontend decoder then turns those
# entries into ``pin().move_by/spin/scale`` calls on a zero-stiffness
# pin shell wrapping the static object.
#
# Composition rule observed (see ``frontend/_scene_.py``,
# ``MoveByOperation.apply`` etc.):
#   * MOVE_BY is sticky: at ``time < t_start`` returns vertex unchanged,
#     at ``time >= t_end`` returns ``vertex + delta``, in between
#     ``vertex + progress * delta`` where progress is linear in
#     ``[t_start, t_end]``. The op does NOT undo itself after t_end.
#   * SPIN clamps ``t = min(time, t_end) - t_start`` so the rotation
#     freezes at its final angle past t_end (also sticky).
#   * SCALE behaves like MOVE_BY (sticky at the final factor).
# Operations chain in the order stored in ``static_ops``: the output
# of op N becomes the input vertex of op N+1. They do NOT gate by
# disjoint frame windows; an op authored on a later window still sees
# all earlier ops' accumulated transforms as its input. That is why
# the previous draft's "MOVE_BY end-frame equals rest + delta"
# assertion failed when SPIN / SCALE were also present in the same
# scene: by the time the assertion ran on the MOVE_BY end frame,
# their windows had not yet started, but later samples had all three
# ops latched on top of MOVE_BY.
#
# Time mapping: PC2 sample 0 is the rest pose recorded when the
# shell is created (solver time 0). Sample ``i >= 1`` corresponds
# to remote frame ``i``. The Rust solver records the actual
# (frame, time) pairs in ``output/data/frame_to_time.out``; for
# this scenario's frame_rate=100 / step_size=0.01 the recorded
# mapping is ``time(i) = (i + 1) * step_size`` (the solver does
# one initial advance bringing frame 1 to t=0.02, then each
# subsequent frame adds step_size). The off-by-one matters: a
# naive ``i * dt`` mapping would put sample 5 at 0.05 / progress
# 0.5, but the solver actually evaluates the op at 0.06 /
# progress 0.6. The expected-position formula below uses
# ``(i + 1) * dt`` to match what the solver did.
#
# Subtests:
#   A. static_pc2_exists_and_has_expected_shape
#         After fetch + drain, the cube has a PC2 with
#         ``(>=frame_count - 1, n_verts, 3)`` shape.
#   B. static_move_by_mid_window_position
#         Vertex 0 at the mid-window PC2 sample matches
#         ``rest + 0.5 * delta`` within 1e-3.
#   C. static_move_by_end_window_position
#         Vertex 0 at the t_end PC2 sample matches ``rest + delta``.
#   D. static_move_by_sticks_after_window
#         A sample past t_end still matches ``rest + delta`` (sticky
#         behavior).
#   E. static_move_by_axis_isolation
#         Y and Z components of vertex 0 stay within 1e-3 of rest at
#         every sample. A regression that swapped the delta axis or
#         leaked motion into Y/Z would fail this.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


# Timeline. frame_rate=100, step_size=0.01 means PC2 sample i is at
# time i * 0.01s. MOVE_BY frame_start=1 -> t_start=0.0 (i.e. it ramps
# from sample 0 up to its end frame). Encoder formula:
# t = (blender_frame - 1) / fps.
_FRAME_COUNT = 13
_MOVE_FRAME_START = 1
_MOVE_FRAME_END = 11
_MOVE_DELTA_X = 0.4

# PC2 sample indices to assert at. _MID_SAMPLE is halfway through the
# MOVE_BY window; _END_SAMPLE is the first sample at or past t_end;
# _STICKY_SAMPLE is past t_end to verify the latched-final behavior.
_MID_SAMPLE = 5
_END_SAMPLE = 10
_STICKY_SAMPLE = 12

_TOLERANCE = 1e-3


_DRIVER_BODY = r"""
import os
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>
MOVE_FRAME_START = <<MOVE_FRAME_START>>
MOVE_FRAME_END = <<MOVE_FRAME_END>>
MOVE_DELTA_X = <<MOVE_DELTA_X>>
MID_SAMPLE = <<MID_SAMPLE>>
END_SAMPLE = <<END_SAMPLE>>
STICKY_SAMPLE = <<STICKY_SAMPLE>>
TOL = <<TOLERANCE>>


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # size=2 places the eight cube corners at +/-1 on each axis. We
    # don't assume which corner is index 0 -- we read it back and
    # use its rest position as the reference for the assertion.
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "StaticCube"
    rest_v0 = tuple(cube.data.vertices[0].co)
    dh.log(f"cube rest_v0={rest_v0}")

    dh.save_blend(PROBE_DIR, "static_op_anim.blend")
    # frame_rate=100 + step_size=0.01 makes solver_time = sample_idx
    # * 0.01s -- a clean rational mapping for the analytical check.
    root = dh.configure_state(
        project_name="static_op_anim",
        frame_count=FRAME_COUNT,
        frame_rate=100,
        step_size=0.01,
    )

    static_group = dh.api.solver.create_group("Stat", "STATIC")
    static_group.add(cube.name)

    # Locate the AssignedObject so we can mutate ``static_ops`` raw.
    # The public solver API does not expose a fluent static-op
    # builder; the addon's UI operator and the MCP handler both
    # mutate the property collection directly.
    group_pg = dh.groups.get_group_by_uuid(bpy.context.scene, static_group.uuid)
    if group_pg is None:
        raise RuntimeError("could not locate STATIC group property group")
    uuid_registry = __import__(
        pkg + ".core.uuid_registry", fromlist=["get_or_create_object_uuid"]
    )
    cube_uuid = uuid_registry.get_or_create_object_uuid(cube)
    assigned = None
    for a in group_pg.assigned_objects:
        if a.uuid == cube_uuid:
            assigned = a
            break
    if assigned is None:
        raise RuntimeError("cube was not registered as an assigned object")

    # Author a single MOVE_BY op. Only one op so we don't need the
    # add-then-move-to-front dance the multi-op driver did.
    move_op = assigned.static_ops.add()
    move_op.op_type = "MOVE_BY"
    move_op.delta = (MOVE_DELTA_X, 0.0, 0.0)
    move_op.frame_start = MOVE_FRAME_START
    move_op.frame_end = MOVE_FRAME_END
    move_op.transition = "LINEAR"

    op_types = [op.op_type for op in assigned.static_ops]
    dh.log(f"static_ops storage order={op_types}")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="static_op_anim:build")
    dh.log("built")

    dh.run_and_wait(timeout=90.0)
    dh.log(f"ran solver={dh.facade.engine.state.solver.name}")
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=15.0)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total}")

    # ----- A: PC2 exists with expected shape ----------------------
    pc2_path = dh.find_pc2_for(cube)
    pc2_arr = None
    if pc2_path and os.path.isfile(pc2_path):
        pc2_arr = dh.read_pc2(pc2_path)
    expected_n_verts = len(cube.data.vertices)
    shape_ok = (
        pc2_arr is not None
        and pc2_arr.ndim == 3
        and pc2_arr.shape[1] == expected_n_verts
        and pc2_arr.shape[2] == 3
        and pc2_arr.shape[0] >= FRAME_COUNT - 1
    )
    dh.record(
        "A_static_pc2_exists_and_has_expected_shape",
        shape_ok,
        {
            "pc2_path": pc2_path,
            "shape": list(pc2_arr.shape) if pc2_arr is not None else None,
            "expected_n_verts": expected_n_verts,
            "expected_min_samples": FRAME_COUNT - 1,
        },
    )

    # No point evaluating the analytical checks if PC2 is missing or
    # the wrong shape -- record them as failures with explanatory
    # details so the scenario reports cleanly.
    if not shape_ok:
        for name in (
            "B_static_move_by_mid_window_position",
            "C_static_move_by_end_window_position",
            "D_static_move_by_sticks_after_window",
            "E_static_move_by_axis_isolation",
        ):
            dh.record(name, False, {"reason": "PC2 unavailable or wrong shape"})
    else:
        # Solver time for sample i, given fps=100, step_size=0.01.
        # See header doc: solver records time(i) = (i + 1) * dt
        # (verified against output/data/frame_to_time.out). Sample 0
        # is special -- it is the rest pose written at shell creation,
        # not a fetched simulation frame -- so the closed-form below
        # is only meaningful for i >= 1.
        dt = 0.01
        t_start = (MOVE_FRAME_START - 1) / 100.0  # 0.0
        t_end = (MOVE_FRAME_END - 1) / 100.0  # 0.10 for FRAME_END=11
        window = t_end - t_start  # 0.10

        rest = pc2_arr[0, 0, :].astype(float)

        def sample_time(idx):
            return (idx + 1) * dt

        def expected_v0_at_sample(idx):
            # Replicates MoveByOperation.apply() with the encoder's
            # frame-to-time mapping. This is the closed-form check
            # that the previous draft dropped.
            t = sample_time(idx)
            if t < t_start:
                progress = 0.0
            elif t >= t_end:
                progress = 1.0
            else:
                progress = (t - t_start) / window
            return (
                rest[0] + progress * MOVE_DELTA_X,
                rest[1],
                rest[2],
            )

        # ----- B: mid-window analytical equality ------------------
        actual_mid = pc2_arr[MID_SAMPLE, 0, :].astype(float).tolist()
        exp_mid = expected_v0_at_sample(MID_SAMPLE)
        err_mid = max(abs(actual_mid[k] - exp_mid[k]) for k in range(3))
        dh.record(
            "B_static_move_by_mid_window_position",
            err_mid < TOL,
            {
                "sample_index": MID_SAMPLE,
                "t": sample_time(MID_SAMPLE),
                "actual": actual_mid,
                "expected": list(exp_mid),
                "max_abs_error": err_mid,
                "tolerance": TOL,
            },
        )

        # ----- C: end-of-window analytical equality ---------------
        actual_end = pc2_arr[END_SAMPLE, 0, :].astype(float).tolist()
        exp_end = expected_v0_at_sample(END_SAMPLE)
        err_end = max(abs(actual_end[k] - exp_end[k]) for k in range(3))
        dh.record(
            "C_static_move_by_end_window_position",
            err_end < TOL,
            {
                "sample_index": END_SAMPLE,
                "t": sample_time(END_SAMPLE),
                "actual": actual_end,
                "expected": list(exp_end),
                "max_abs_error": err_end,
                "tolerance": TOL,
            },
        )

        # ----- D: sticky after t_end ------------------------------
        actual_stick = pc2_arr[STICKY_SAMPLE, 0, :].astype(float).tolist()
        exp_stick = expected_v0_at_sample(STICKY_SAMPLE)
        err_stick = max(abs(actual_stick[k] - exp_stick[k]) for k in range(3))
        dh.record(
            "D_static_move_by_sticks_after_window",
            err_stick < TOL,
            {
                "sample_index": STICKY_SAMPLE,
                "t": sample_time(STICKY_SAMPLE),
                "actual": actual_stick,
                "expected": list(exp_stick),
                "max_abs_error": err_stick,
                "tolerance": TOL,
            },
        )

        # ----- E: axis isolation ---------------------------------
        # No motion was authored on Y or Z. Every sample's vertex 0
        # must keep the rest Y / Z. A regression that swapped axes
        # in the encoder or decoder would fail here even if the
        # X-axis magnitude assertions still passed by coincidence.
        n_samples = pc2_arr.shape[0]
        max_yz_drift = 0.0
        worst_sample = -1
        for i in range(n_samples):
            dy = abs(float(pc2_arr[i, 0, 1]) - rest[1])
            dz = abs(float(pc2_arr[i, 0, 2]) - rest[2])
            drift = max(dy, dz)
            if drift > max_yz_drift:
                max_yz_drift = drift
                worst_sample = i
        dh.record(
            "E_static_move_by_axis_isolation",
            max_yz_drift < TOL,
            {
                "n_samples": n_samples,
                "max_yz_drift": max_yz_drift,
                "worst_sample": worst_sample,
                "tolerance": TOL,
            },
        )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))
        .replace("<<MOVE_FRAME_START>>", str(_MOVE_FRAME_START))
        .replace("<<MOVE_FRAME_END>>", str(_MOVE_FRAME_END))
        .replace("<<MOVE_DELTA_X>>", repr(_MOVE_DELTA_X))
        .replace("<<MID_SAMPLE>>", str(_MID_SAMPLE))
        .replace("<<END_SAMPLE>>", str(_END_SAMPLE))
        .replace("<<STICKY_SAMPLE>>", str(_STICKY_SAMPLE))
        .replace("<<TOLERANCE>>", repr(_TOLERANCE))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
