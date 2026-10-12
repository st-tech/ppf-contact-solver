# File: scenarios/bl_force_field_fluid_volume_solve.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A Fluid Flow reading a Volume object's VDB SEQUENCE whose voxels move from
# frame to frame, held to a closed form end to end: payload and solve.
#
# The sequence holds an AFFINE velocity field, u(x) = G (x - c) + b_k in
# Blender world axes and meters per second (Velocity Unit Second), with G a
# full matrix (1/s on the diagonal, off-diagonal terms besides) and b_k a
# different offset per frame (consecutive ones about 0.8 m/s apart).
# Trilinear interpolation between voxel centers reproduces an affine field
# EXACTLY, so every expected value is a closed form, and anything that places
# a voxel wrongly shows up as G times the misplacement. Fluid frame k is
# Blender frame 1 + k and file `aff_<1 + k>.vdb`, k = 0 to 9 (the add-on's
# Frame Count is at least 10). What moves per frame:
#
#   * the ACTIVE BOX grows and its low corner moves: voxels lo_k = (0, -k, 0)
#     to lo_k + n_k - 1 with n_k = (7 + 2k, 6 + k, 6 + k), so the resolution
#     the payload carries differs every frame, the brick table grows from 1
#     to 4 bricks along x, and the box starts at a negative index;
#   * the VDB TRANSFORM: voxel 0.08, rotated 10 + k degrees about the file's
#     z and translated along the file's x by 0.06 per frame up to frame 2 and
#     0.015 per frame after;
#   * the VOLUME OBJECT: located, scaled 1.5 uniformly, and its rotation
#     keyframed at every frame ((0.2, 0, 0.4 + 0.015 (f - 1)) at Blender frame
#     f), so the add-on must compose each frame's file transform with that
#     frame's object matrix.
#
# The cloth is a SHELL with Young's modulus 0.01 (the property's minimum) and
# no bending, under a Fluid Flow of Strength 5 and Flow 1, at 100 fps with
# one 0.01 s step per frame, gravity and air off:
#
#   * SHEET, 5 x 5 vertices inside the INTERIOR of fluid frames 1 to 9
#     (between their outer voxel centers, where trilinear equals the affine
#     field); measured at least 1.2 voxels inside at the solver's own
#     positions, asserted above 0.25;
#   * EDGE, one triangle whose vertices sit 0.25 voxel inside fluid frame 1's
#     low x face, in its outer half cell, where the field is the affine one
#     held at the outer voxel center (the trilinear clamp), and outside fluid
#     frames 2 to 9 (frame 2's face has moved 0.75 voxel past frame 1's).
#
# Each tolerance below sits between what the tree measures and what a defect
# measures. The defects were made by replacing the add-on's `fluid_frame`
# in-process for one run each: the corner slipped half a cell along the
# domain's first axis (CORNER), fluid frame 0's object matrix used for every
# frame (MATRIX), and each frame carrying the previous frame's fluid (LATE).
#
# Subtests:
#   O. this_blender_ships_openvdb: the scenario writes the sequence with
#      Blender's openvdb module; without it every other check is absent, and
#      this one names why.
#   A. the_payload_reads_the_affine_field_per_frame: every frame travels with
#      its own growing resolution, and the decoded volume
#      (`field_eval.fluid_frame`) reads G (x - c) + b_k at 20 random interior
#      points of each frame within 1e-5 m/s. Measured: at most 2.8e-7 against
#      field values up to 3.3 m/s (float32 storage). CORNER reads 0.066 off,
#      MATRIX 0.0086 at frame 1 and up to 2.7 later, LATE carries the wrong
#      resolutions and reads up to 2.9 off.
#   B. each_frames_presence_ends_at_its_own_box: 0.15 voxel inside each of
#      the six faces of each frame's box (the outer voxel centers plus half a
#      voxel) the presence is 1, and 0.15 voxel outside it is 0, 120 points.
#      Measured: none wrong. CORNER turns 20 of them, MATRIX 31, LATE 43.
#   C. the_sheet_steps_by_backward_euler: for each step F -> F + 1, the
#      sheet's velocity change over dt equals the backward-Euler solution of
#      a = s (U_{F+1}(y) - v), y the step's end position and U_{F+1} the
#      closed form at fluid frame F + 1, within 1e-2 of the largest push (7.0
#      m/s^2). Measured: at most 1.2e-3, growing over the steps; rounding
#      the PC2's positions to float32 and differencing them twice can alone
#      account for up to 3.4e-4.
#      CORNER reads 4.6e-2 at every step, MATRIX up to 9.8e-2 (6e-3 at the
#      first step, where the matrices differ least), LATE 0.65 to 0.71. The
#      scene keeps its power to catch a frame read one off: the reference
#      with U_F in place of U_{F+1} differs by 73% of the push, and the check
#      asserts at least 30%.
#   D. a_vertex_only_in_frame_1s_box_is_pushed_only_by_the_first_step: the
#      EDGE triangle's first step equals the backward-Euler solution with the
#      clamped field of fluid frame 1 within the same 1e-2 of the push, and
#      every later step changes its velocity by less than 1e-2 of the push
#      (no push and no damping outside the box). Measured: 1.3e-4 and
#      1.8e-4, the first push 28% of the largest (asserted above 10%).
#      CORNER leaves the triangle unpushed (0.28 off at the first step); LATE
#      pushes it at the second step (0.28). The geometric margins are asserted
#      at the solver's own positions so the case cannot drift out of what it
#      tests: measured 0.25 voxel inside frame 1 and 0.41 outside frames 2 to
#      9, asserted above 0.1.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
BACKENDS = ("real",)

_FRAME_COUNT = 10

_DRIVER_BODY = r"""
import math
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
NF = <<FRAME_COUNT>>

VOX = 0.08
STRENGTH = 5.0
DT = 0.01
FPS = 100.0
G = np.array([[1.0, 0.5, -0.3], [-0.4, 1.0, 0.6], [0.2, -0.5, 1.0]])
B0 = np.array([0.6, -0.4, 0.3])
# Where the sheet is centered, in fluid frame 5's index space: the point
# deepest inside the interiors of fluid frames 1 to 9 (searched once on a
# grid; the margins are asserted below, not assumed).
SHEET_CENTER_G5 = np.array([3.5, 1.75, 2.9])


def solver(v):
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


def vdb_matrix(k):
    # OpenVDB's row-vector 4 x 4: world = [i j k 1] @ A, translation last row.
    th = math.radians(10.0 + 1.0 * k)
    c, s = math.cos(th), math.sin(th)
    shift = 0.06 * min(k, 2) + 0.015 * max(k - 2, 0)
    return np.array([[c * VOX, s * VOX, 0.0, 0.0], [-s * VOX, c * VOX, 0.0, 0.0],
                     [0.0, 0.0, VOX, 0.0], [shift, 0.0, 0.0, 1.0]])


def box(k):
    lo = np.array([0, -k, 0], dtype=np.int64)
    n = np.array([7 + 2 * k, 6 + k, 6 + k], dtype=np.int64)
    return lo, lo + n - 1


def b_of(k):
    # A different offset every frame, consecutive ones about 0.8 m/s apart.
    return B0 + 0.5 * np.array([math.cos(2.0 * k), math.sin(2.0 * k + 0.4),
                                math.cos(2.0 * k + 1.1)])


class NoOpenVDB(Exception):
    pass


try:
    try:
        import openvdb
    except ImportError as exc:
        dh_missing = DriverHelpers(pkg, result)
        dh_missing.record("O_this_blender_ships_openvdb", False,
                          {"error": "this Blender has no openvdb module (%s); the "
                                    "scenario writes its VDB sequence with it" % exc})
        raise NoOpenVDB()
    dh = DriverHelpers(pkg, result)
    dh.record("O_this_blender_ships_openvdb", True, {})
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.fps = int(FPS)
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = NF

    # ----- the Volume object, rotation keyframed at every frame -------------
    vol = bpy.data.volumes.new("FFAffine")
    vol_obj = bpy.data.objects.new("FFAffine", vol)
    scene.collection.objects.link(vol_obj)
    vol_obj.location = (0.4, -0.3, 0.6)
    vol_obj.scale = (1.5, 1.5, 1.5)
    for f in range(1, NF + 1):
        vol_obj.rotation_euler = (0.2, 0.0, 0.4 + 0.015 * (f - 1))
        vol_obj.keyframe_insert("rotation_euler", frame=f)
    M = []
    for k in range(NF):
        scene.frame_set(1 + k)
        M.append(np.array(vol_obj.matrix_world, dtype=np.float64))

    def index_to_world(k, g):
        A = vdb_matrix(k)
        p = np.asarray(g, dtype=np.float64) @ A[:3, :3] + A[3, :3]
        return p @ M[k][:3, :3].T + M[k][:3, 3]

    def world_to_index(k, x):
        A = vdb_matrix(k)
        p = (np.asarray(x, dtype=np.float64) - M[k][:3, 3]) @ np.linalg.inv(M[k][:3, :3]).T
        return (p - A[3, :3]) @ np.linalg.inv(A[:3, :3])

    center = index_to_world(5, SHEET_CENTER_G5)

    def affine(k, x):
        return (np.asarray(x) - center) @ G.T + b_of(k)

    def depth(k, x):
        # Voxels inside fluid frame k's box (its outer voxel centers plus half
        # a voxel), the smallest over the six faces: negative outside.
        g = world_to_index(k, x)
        lo, hi = box(k)
        return np.min(np.minimum(g - (lo - 0.5), (hi + 0.5) - g), axis=-1)

    def interior(k, x):
        # Voxels inside fluid frame k's outer voxel centers, where trilinear
        # interpolation is the affine field itself.
        g = world_to_index(k, x)
        lo, hi = box(k)
        return np.min(np.minimum(g - lo, hi - g), axis=-1)

    def field_at(k, x):
        # The closed form as the solver reads it: inside the box the affine
        # field at the point held to the outer voxel centers (trilinear
        # clamps there), outside nothing.
        x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
        g = world_to_index(k, x)
        lo, hi = box(k)
        inside = np.all((g >= lo - 0.5) & (g <= hi + 0.5), axis=1)
        held = index_to_world(k, np.clip(g, lo, hi))
        return affine(k, held) * inside[:, None], inside.astype(np.float64)

    # ----- the sequence -----------------------------------------------------
    vdb_dir = os.path.join(PROBE_DIR, "fluid_volume_solve_vdb")
    os.makedirs(vdb_dir, exist_ok=True)
    boxes_written = []
    for k in range(NF):
        lo, hi = box(k)
        n = hi - lo + 1
        ii, jj, kk = np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi)], indexing="ij")
        g = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3).astype(np.float64)
        u_world = affine(k, index_to_world(k, g))
        # The object's transform turns a file velocity into a world one
        # (rotation and scale), so the file holds its inverse image.
        u_file = u_world @ np.linalg.inv(M[k][:3, :3]).T
        grid = openvdb.Vec3SGrid()
        grid.name = "vel"
        grid.transform = openvdb.createLinearTransform(vdb_matrix(k).tolist())
        # C order, explicitly: copyFromArray reads raw memory.
        grid.copyFromArray(np.ascontiguousarray(u_file.reshape(tuple(n) + (3,)),
                                                dtype=np.float32), ijk=tuple(int(v) for v in lo))
        bb = grid.evalActiveVoxelBoundingBox()
        boxes_written.append([list(bb[0]), list(bb[1])])
        openvdb.write(os.path.join(vdb_dir, "aff_%04d.vdb" % (1 + k)), grids=[grid])
    want_boxes = [[box(k)[0].tolist(), box(k)[1].tolist()] for k in range(NF)]
    if boxes_written != want_boxes:
        raise RuntimeError("the written active boxes %r are not the authored %r"
                           % (boxes_written, want_boxes))
    vol.filepath = os.path.join(vdb_dir, "aff_0001.vdb")
    vol.is_sequence = True
    vol.frame_start = 1
    vol.frame_duration = NF
    # The files hold m/s. Set the unit after the load, which names the grids.
    vol.grids.load()
    vol.velocity_unit = "SECOND"

    # ----- the cloth --------------------------------------------------------
    def mesh_object(name, verts, faces):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(float(c) for c in v) for v in verts], [], faces)
        me.update()
        o = bpy.data.objects.new(name, me)
        scene.collection.objects.link(o)
        return o

    e1 = np.array([1.0, 0.3, 0.2])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, np.array([0.2, -0.4, 1.0]))
    e2 /= np.linalg.norm(e2)
    ticks = np.linspace(-1.4, 1.4, 5)
    sheet_g = np.array([SHEET_CENTER_G5 + a * e1 + b * e2 for b in ticks for a in ticks])
    sheet_verts = index_to_world(5, sheet_g)
    sheet_faces = [(j * 5 + i, j * 5 + i + 1, (j + 1) * 5 + i + 1, (j + 1) * 5 + i)
                   for j in range(4) for i in range(4)]
    sheet = mesh_object("FFSheet", sheet_verts, sheet_faces)
    lo1, hi1 = box(1)
    mid = 0.5 * (lo1 + hi1)
    edge_g = np.array([[-0.25, mid[1] - 0.6, mid[2] - 0.6], [-0.25, mid[1] + 0.6, mid[2] - 0.6],
                       [-0.25, mid[1], mid[2] + 0.6]])
    edge = mesh_object("FFEdge", index_to_world(1, edge_g), [(0, 1, 2)])

    scene.frame_set(1)
    dh.save_blend(PROBE_DIR, "force_field_fluid_volume_solve.blend")
    root = dh.configure_state(project_name="force_field_fluid_volume_solve",
                              frame_count=NF, frame_rate=int(FPS), step_size=DT)
    state = root.state
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    cloth.add(edge.name)
    cloth.param.shell_young_modulus = 0.01
    cloth.param.bend = 0.0

    bpy.ops.object.effector_add(type="FLUID", location=(0.0, 0.0, 0.0))
    flow = bpy.context.object
    flow.name = "FFFlow"
    flow.field.source_object = vol_obj
    flow.field.strength = STRENGTH
    flow.field.flow = 1.0

    # ----- A and B: the payload, before any solve ---------------------------
    params_mod = __import__(pkg + ".core.encoder.params", fromlist=["_build_param_dict"])
    fe = __import__(pkg + ".core.field_eval", fromlist=["fluid_frame"])
    (sent,) = [f for f in params_mod._build_param_dict(bpy.context)["force_field"]["fields"]
               if f["type"] == "fluid"]
    v = sent["fluid"]
    volume = {"resolution": [tuple(x) for x in v["resolution"]], "cell": np.asarray(v["cell"]),
              "corner": np.asarray(v["corner"]), "axes": np.asarray(v["axes"]),
              "start": v["start"], "step": v["step"],
              "tables": [np.frombuffer(b"".join(t), dtype="<u4") for t in v["tables"]],
              "bricks": [np.frombuffer(b"".join(b), dtype="<f4").reshape(-1, 512, 4)
                         for b in v["bricks"]]}
    rng = np.random.default_rng(7)
    per_frame = []
    for k in range(min(NF, len(volume["tables"]))):
        lo, hi = box(k)
        g = lo + 0.1 + rng.random((20, 3)) * (hi - lo - 0.2)
        x = index_to_world(k, g)
        got = fe.fluid_frame(volume, k, solver(x))[0]
        want = solver(affine(k, x))
        per_frame.append({"frame": k, "resolution": list(volume["resolution"][k]),
                          "error": float(np.abs(got - want).max()),
                          "scale": round(float(np.abs(want).max()), 4)})
    resolutions = [list(volume["resolution"][k]) for k in range(len(volume["tables"]))]
    want_res = [(box(k)[1] - box(k)[0] + 1).tolist() for k in range(NF)]
    worst = max((p["error"] for p in per_frame), default=float("inf"))
    dh.record("A_the_payload_reads_the_affine_field_per_frame",
              resolutions == want_res and worst < 1e-5
              and min((p["scale"] for p in per_frame), default=0.0) > 0.5,
              {"resolutions": resolutions, "want_resolutions": want_res,
               "worst_error": worst, "frames": per_frame})

    wrong = []
    for k in range(min(NF, len(volume["tables"]))):
        lo, hi = box(k)
        c3 = 0.5 * (lo + hi)
        for axis in range(3):
            for side, face in ((-1.0, lo[axis] - 0.5), (1.0, hi[axis] + 0.5)):
                for offset, expect in ((-0.15, 1.0), (0.15, 0.0)):
                    g = c3.copy()
                    g[axis] = face + side * offset
                    point = solver(index_to_world(k, g[None]))
                    presence = float(fe.fluid_frame(volume, k, point)[2][0])
                    if presence != expect:
                        wrong.append({"frame": k, "axis": axis, "side": side,
                                      "offset": offset, "presence": presence})
    dh.record("B_each_frames_presence_ends_at_its_own_box", not wrong and len(per_frame) == NF,
              {"wrong": wrong[:12], "wrong_count": len(wrong), "points": 12 * len(per_frame)})

    # ----- the solve --------------------------------------------------------
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, message="force_field_fluid_volume_solve:build",
                      timeout=300.0)
    dh.run_and_wait(timeout=600.0)
    solver_state = dh.facade.engine.state.solver.name
    dh.force_frame_query(expected_frames=NF - 1, timeout=120.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()

    def rows(obj):
        path = dh.find_pc2_for(obj)
        arr = dh.read_pc2(path) if path else None
        if arr is None or arr.shape[0] != NF or not np.all(np.isfinite(arr)):
            return None
        # PC2 row i is Blender frame 1 + i, row 0 the rest pose; the meshes
        # are authored in world coordinates under an identity matrix.
        m = np.array(obj.matrix_world, dtype=np.float64)
        return arr.astype(np.float64) @ m[:3, :3].T + m[:3, 3]

    def be_velocity(k, x, vel):
        # Backward Euler of a = s p (U(y) - v), y = x + dt v, by successive
        # substitution;
        # the contraction factor is dt^2 s |G| / (1 + dt s), below 1e-3.
        w = vel.copy()
        for _ in range(30):
            U, p = field_at(k, x + DT * w)
            w = (vel + DT * STRENGTH * p[:, None] * U) / (1.0 + DT * STRENGTH * p)[:, None]
        return w

    def step_table(P, frame_shift=0):
        # Per step F -> F + 1: the solver's acceleration and the reference's,
        # the reference read at fluid frame F + 1 + frame_shift.
        out = []
        for F in range(NF - 1):
            x = P[F]
            vel = np.zeros_like(x) if F == 0 else (P[F] - P[F - 1]) / DT
            a_s = ((P[F + 1] - P[F]) / DT - vel) / DT
            k = min(max(F + 1 + frame_shift, 0), NF - 1)
            a_ref = (be_velocity(k, x, vel) - vel) / DT
            out.append((a_s, a_ref))
        return out

    detail = {"solver_state": solver_state, "error": dh.facade.engine.state.error}
    S = rows(sheet)
    E = rows(edge)
    detail["rows"] = None if S is None else int(S.shape[0])
    ok_c = False
    if S is not None:
        table = step_table(S)
        push = max(float(np.linalg.norm(a_ref, axis=1).max()) for _, a_ref in table)
        errs = [float(np.linalg.norm(a_s - a_ref, axis=1).max()) / push for a_s, a_ref in table]
        late = step_table(S, frame_shift=-1)
        slip = max(float(np.linalg.norm(a0[1] - a1[1], axis=1).max()) / push
                   for a0, a1 in zip(table, late))
        margin = min(float(interior(F + 1, S[F + 1]).min()) for F in range(NF - 1))
        detail.update({"push": round(push, 4), "relative_errors": [round(e, 5) for e in errs],
                       "frame_slip_relative": round(slip, 4),
                       "interior_margin_voxels": round(margin, 3)})
        ok_c = max(errs) < 1e-2 and slip > 0.3 and margin > 0.25
    dh.record("C_the_sheet_steps_by_backward_euler", ok_c, detail)

    edge_detail = {}
    ok_d = False
    if E is not None and S is not None:
        table = step_table(E)
        first = float(np.linalg.norm(table[0][0] - table[0][1], axis=1).max()) / push
        first_push = float(np.linalg.norm(table[0][1], axis=1).max()) / push
        later = max(float(np.linalg.norm(a_s, axis=1).max()) / push for a_s, _ in table[1:])
        inside_1 = float(depth(1, E[1]).min())
        outside_later = min(float(-depth(F + 1, E[F + 1]).max()) for F in range(1, NF - 1))
        edge_detail = {"first_step_relative_error": round(first, 5),
                       "first_step_push_relative": round(first_push, 4),
                       "later_steps_relative": round(later, 6),
                       "inside_frame_1_voxels": round(inside_1, 3),
                       "outside_frames_2_to_9_voxels": round(outside_later, 3)}
        ok_d = (first < 1e-2 and first_push > 0.1 and later < 1e-2
                and inside_1 > 0.1 and outside_later > 0.1)
    dh.record("D_a_vertex_only_in_frame_1s_box_is_pushed_only_by_the_first_step", ok_d,
              edge_detail)

except NoOpenVDB:
    pass
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
        .replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
