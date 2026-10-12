# File: scenarios/bl_force_field_fluid_parity.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A Fluid Flow field against BLENDER ITSELF, not against this project's own
# reference: Blender's particles in baked Mantaflow domains are the oracle,
# and the add-on's payload and the solver's cloth are held to them.
#
# Blender's Fluid Flow law is linear in the velocity, a = A(x) - C(x) v,
# with C exactly 0 outside a domain and s * flow * W * rho inside (measured).
# Two NEWTON particle systems at the same points, one at rest and one at
# V_REF, give A and C per point; a particle born at frame F reads frame
# F + 1's grid over its first step. The solver's backward-Euler step from
# velocity v is then v' = (v + dt A) / (1 + dt C), with A and C taken where
# the solver evaluates the fluid: at the step's end position (the Newton
# iterate). Read at the step's start instead, the first step is 13x
# further off (C's second number), past C's bound, and C requires that
# separation, so the check tells the two reads apart.
#
# Scene (one solve at scene fps 100, step 0.01, start frame 8, 10 frames,
# so one solver step per Blender frame; gravity, air and contact off):
#   * FPMain: a size-2 cube scaled 0.9, rotated (0.3, -0.2, 0.6) and offset,
#     baked (Cache Type All), its inflow keyed CONSTANT to a new direction or
#     speed at every solve frame, so consecutive grids differ strongly.
#   * FPNonCubic1 (4 x 1.3 x 2, 16 x 5 x 8 cells) and FPNonCubic2
#     (2 x 2 x 2.2, 15 x 15 x 16 cells), each with a steady inflow.
#   * Fields, in payload order (sorted by name), so a fluid's field index
#     differs from its fluid index: a Force (Strength 0.5, Max Distance 1.6,
#     reaching the Jet and not the non-cubic domains), a Fluid Flow on FPMain
#     (Strength 1, Flow 1), a second one on FPMain (Strength 3, Flow 2, Apply
#     Density, falloff power 1, Max Distance 1.2), one per non-cubic domain.
#   * One SHELL group (Young's modulus 0.01, its minimum; bend 0; strain limit
#     off), every object in two copies: at rest and with initial velocity
#     V_REF. The copies overlap exactly, so the group allows inter-object
#     intersections. Jet: a 7 x 7 sheet across the main jet. Faces: small
#     triangles 0.2 cell inside and outside FPMain's -X and +Y faces. NC1 and
#     NC2: small triangles inside each non-cubic domain (most in its jet) and
#     0.03 m inside and outside each face, NC2 also 0.03 m past its +X and
#     +Y faces where a uniform cell would overhang them.
# A second solve (U, V) reuses FPMain's bake at scene fps 25 with step 0.01,
# four solver steps per Blender frame, one Fluid Flow at Strength 1, Flow 0.
#
# Subtests:
#   A. the_payload_reads_blenders_acceleration: the payload, decoded the way
#      the frontend's decoder does and summed over every field by the
#      frontend's reference laws at fluid time 0.01, gives both particle
#      systems' acceleration at Jet and Faces.
#   B. a_read_one_frame_early_would_be_seen: the same read at fluid time 0
#      is far from Blender, so this scene can see a frame slip.
#   C. a_resting_sheet_takes_blenders_step, D. a_moving_sheet_...: the
#      solver's first step of the Jet copies is the backward-Euler step from
#      Blender's A and C. C also requires the same step from A and C read at
#      the step's START to lie more than 1.5 times its bound away, so a
#      solver reading the fluid there cannot pass it.
#   E. presence_at_the_domain_faces_is_blenders: C from the solver's two
#      Faces copies is over 0.5 exactly where Blender's is, which is where the
#      triangles were built inside, and is Blender's value.
#   F. every_streamed_step_takes_blenders_step: all nine steps of the Jet at
#      rest, each from the solver's own velocity, against Blender's A and C at
#      the solver's end position of that step.
#   G. every_fluid_frame_differs_from_every_other: the payload's fluid frames
#      at the Jet differ pairwise, so a held, stale or swapped slot fails F.
#   H, I. noncubic_*_presence_and_speed_are_blenders: presence by C at every
#      face (and overhang) triangle; at interior vertices the fluid moves, the
#      solver's undamped push has Blender's speed; and its direction is
#      Blender's vector skewed back by diag(longest extent / extent) in the
#      domain's axes. Blender skews a non-cubic domain's direction and keeps
#      its speed (measured); the solver reads the grid's own vector, a
#      deliberate difference that this pins.
#   U. substeps_blend_the_two_bracketing_frames: over one Blender frame of
#      four solver steps the sheet moves by dt^2 sum_j (5 - j) a_j, with a_j
#      the push blended (1 - j/4, j/4) between frames 8 and 9, read through
#      the payload at the solver's own positions.
#   V. blender_holds_the_next_frame_a_recorded_difference: Blender's own
#      particles over the same frame (timestep 0.04, 3 subframes) gain
#      0.04 A9, holding frame 9 for every substep; the solver blends.
#      Both are asserted, so either drifting fails.
#   Z. bookkeeping_particles_match_their_vertices.
#
# MEASURED on dev-head (CPU backend, Blender 5.2), each check's value against
# its bound; the CPU leg is bit-reproducible run to run:
#   A  8.3e-7 relative          < 1e-5   (12x; Blender computes in float32)
#   B  0.43 of max|A|           >= 0.3
#   C  1.6e-4                   < 1e-3   (6x); the step from A, C at the
#      step's start lies 2.0e-3 from it   > 1.5e-3, and a solver reading the
#      fluid there measured 2.0e-3 (2x past the bound)
#   D  1.7e-4                   < 5e-3   (30x)
#   E  0 mismatches of 24; |C_S - C_B| 4.8e-3 < 0.05 (10x); Blender's C
#      outside is 3e-6, inside exactly 1
#   F  1.5e-3 of the run's peak |A| (5.8)   < 1e-2 (6.8x); it grows from
#      1.3e-4 at the first step to that at the ninth
#   G  closest pair 0.33        >= 0.2
#   H  speed 5.9e-4, unskewed vector 8.1e-4 < 5e-3 (6x); Blender's own vector
#      is 13.3% off the solver's, the deliberate difference
#   I  speed 9.5e-4, unskewed vector 1.3e-3 < 5e-3 (3.8x); 2.6% off
#   U  2.2e-4                   < 5e-3   (23x); blend and hold 30% apart
#   V  hold 1.2e-3 < 1e-2 (8x); Blender against the blend 17.6% > 0.1
#   Z  particle-to-vertex match 2.1e-7 < 1e-5; birth velocity 1.5e-9
# The floor under C, D, H and I is the float32 PC2: half a float32 spacing
# of a position divided by dt^2. Every coordinate stays under 4 in magnitude
# so that floor stays near 1e-3 m/s^2; with the non-cubic domains 8 from the
# origin it was 5e-3, and subtest I measured 3.7e-3.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
BACKENDS = ("real",)
# Two build cycles on one server, the second reading the first's verdict
# off persistent state (the orchestrator's stated reason for serial runs).
NOT_PARALLELIZABLE = True

_DRIVER_BODY = r"""
import importlib
import math
import os
import sys
import traceback
import types

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>

START = 8
FRAME_COUNT = 10
LAST = START + FRAME_COUNT - 1
DT = 0.01
V_REF = np.array([0.1, -0.06, 0.03])


def to_s(v):
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


def from_s(v):
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)


def rnd(x, k=5):
    return float(round(float(x), k))


def span(x, k=5):
    x = np.asarray(x, dtype=np.float64)
    return [rnd(x.min(), k), rnd(x.max(), k)] if x.size else None


try:
    dh = DriverHelpers(pkg, result)
    params_mod = __import__(pkg + ".core.encoder.params", fromlist=["_build_param_dict"])
    scene = bpy.context.scene

    # ----- the reference: the frontend's own decode and laws ---------------
    # The payload is decoded into field records the way the frontend's
    # decoder does (FluidVolume.from_bricks, then make_field), and evaluated
    # by the frontend's reference laws. The package is reached under a name
    # of its own, so its __init__ (which loads the cdylib) never runs.
    def reference_modules():
        name = "ppf_fluid_parity_reference"
        if name not in sys.modules:
            ref_pkg = types.ModuleType(name)
            ref_pkg.__path__ = [os.path.join(LOCAL_PATH, "frontend")]
            sys.modules[name] = ref_pkg
        keep = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            return (importlib.import_module(name + "._force_field_"),
                    importlib.import_module(name + "._field_eval_"))
        finally:
            sys.dont_write_bytecode = keep

    ffm, fe = reference_modules()

    def payload_records():
        out = []
        for f in params_mod._build_param_dict(bpy.context)["force_field"]["fields"]:
            args = dict(f)
            kind = args.pop("type")
            args.pop("groups", None)
            vol = args.pop("fluid", None)
            if vol is not None:
                frames = len(vol["tables"])
                tables = [np.frombuffer(b"".join(t), dtype="<u4") for t in vol["tables"]]
                bricks = [np.frombuffer(b"".join(b), dtype="<f4") for b in vol["bricks"]]
                args["fluid"] = ffm.FluidVolume.from_bricks(
                    vol["resolution"], tables, bricks,
                    cell=np.asarray(vol["cell"], dtype=np.float64),
                    has_density=bool(args.get("use_density", False)),
                    corner=vol["corner"], axes=vol["axes"], start=vol["start"],
                    step=vol["step"] if frames > 1 else None)
            out.append(ffm.make_field(kind, **args))
        return out

    def reference_accel(recs, points, velocities, fluid_time):
        # Blender-world points and velocities in, Blender-world acceleration
        # out; every field summed, each fluid read at fluid_time.
        p = to_s(points)
        v = to_s(np.broadcast_to(velocities, np.shape(points)))
        a = np.zeros_like(p)
        for rec in recs:
            a += fe.evaluate(rec, p, velocities=v, time=0.0, normals=np.zeros_like(p),
                             fluid_time=fluid_time)[0]
        return from_s(a)

    # ----- the scene --------------------------------------------------------
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene.render.fps = 100
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = LAST
    scene.use_gravity = False

    def keyed_constant(obj):
        ad = obj.animation_data
        if ad is None or ad.action is None:
            return
        try:
            curves = list(ad.action.fcurves)
        except AttributeError:
            curves = [fc for layer in ad.action.layers for strip in layer.strips
                      for bag in strip.channelbags for fc in bag.fcurves]
        for fc in curves:
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"

    def bake(dom):
        bake_fluid_domain(scene, dom, result.setdefault("bake_notes", []))

    def gas_domain(name, location, scale, rotation, inflow_offset, inflow_size, keys):
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)
        dom = bpy.context.object
        dom.name = name
        dom.scale = scale
        dom.rotation_euler = rotation
        mod = dom.modifiers.new("fluid", "FLUID")
        mod.fluid_type = "DOMAIN"
        ds = mod.domain_settings
        ds.domain_type = "GAS"
        ds.resolution_max = 16
        ds.alpha = ds.beta = 0.0
        ds.gravity = (0.0, 0.0, 0.0)
        ds.use_adaptive_timesteps = False
        ds.cache_type = "ALL"
        ds.cache_directory = os.path.join(PROBE_DIR, "parity_cache_" + name)
        ds.cache_frame_start, ds.cache_frame_end = 1, LAST
        bpy.ops.mesh.primitive_cube_add(
            size=inflow_size,
            location=tuple(np.asarray(location) + np.asarray(inflow_offset)))
        src = bpy.context.object
        src.name = name + "Inflow"
        fm = src.modifiers.new("fluid", "FLUID")
        fm.fluid_type = "FLOW"
        fl = fm.flow_settings
        fl.flow_type = "SMOKE"
        fl.flow_behavior = "INFLOW"
        fl.use_initial_velocity = True
        fl.velocity_coord = keys[0][1]
        for frame, vel in keys:
            fl.velocity_coord = vel
            fl.keyframe_insert("velocity_coord", frame=frame)
        keyed_constant(src)
        src.hide_render = True
        bpy.context.view_layer.update()
        bake(dom)
        return dom

    # THE MAIN DOMAIN: rotated, scaled and offset, its inflow turning to a new
    # direction or speed at every solve frame (CONSTANT keys), so every grid
    # of the solve differs from every other near the emitter (G) and a frame
    # slip cannot hide.
    s2 = 1.5 / math.sqrt(2.0)
    s3 = 2.5 / math.sqrt(2.0)
    main_keys = [(1, (1.5, 0.0, 0.0)), (9, (0.0, 1.5, 0.0)), (10, (0.0, 0.0, -1.5)),
                 (11, (-1.5, 0.0, 0.0)), (12, (0.0, -1.5, 0.0)), (13, (0.0, 0.0, 1.5)),
                 (14, (s2, s2, 0.0)), (15, (0.0, -s2, s2)), (16, (-s3, 0.0, -s3)),
                 (17, (s2, -s2, 0.0))]
    main = gas_domain("FPMain", (0.05, 0.05, -0.05), (0.9, 0.9, 0.9), (0.3, -0.2, 0.6),
                      (-0.65, -0.05, 0.05), 0.4, main_keys)
    # THE NON-CUBIC DOMAINS, beside the main one and out of the Force field's
    # reach, each with a steady inflow. Every coordinate stays under 4 in
    # magnitude, where a float32 PC2 position is good to 2.4e-7.
    NC1_AT = np.array([0.0, 3.0, 0.0])
    NC1_EXT = np.array([4.0, 1.3, 2.0])
    nc1 = gas_domain("FPNonCubic1", tuple(NC1_AT), tuple(NC1_EXT / 2.0), (0.0, 0.0, 0.0),
                     (-1.2, 0.1, 0.0), 0.6, [(1, (1.5, 0.0, 0.0))])
    NC2_AT = np.array([0.0, -3.0, 0.0])
    NC2_EXT = np.array([2.0, 2.0, 2.2])
    nc2 = gas_domain("FPNonCubic2", tuple(NC2_AT), tuple(NC2_EXT / 2.0), (0.0, 0.0, 0.0),
                     (-0.6, 0.1, 0.0), 0.4, [(1, (1.5, 0.0, 0.0))])
    dh.log("baked")

    # ----- the fields, named so the payload's order is this one -------------
    def field(kind, name, loc):
        bpy.ops.object.effector_add(type=kind, location=loc)
        o = bpy.context.object
        o.name = name
        return o

    f0 = field("FORCE", "FPa_Force", (-0.45, 0.0, -1.2))
    f0.field.strength = 0.5
    # It reaches the whole Jet (0.95 to 1.47 away) and stops short of the
    # non-cubic domains (2.0 away at the nearest), so there the fluid alone
    # acts and a speed can be compared.
    f0.field.use_max_distance = True
    f0.field.distance_max = 1.6
    f1 = field("FLUID", "FPb_Fluid", (0.0, 0.0, 0.0))
    f1.field.source_object = main
    f1.field.strength = 1.0
    f1.field.flow = 1.0
    f2 = field("FLUID", "FPc_FluidDensity", (0.2, 0.3, 0.1))
    f2.field.source_object = main
    f2.field.strength = 3.0
    f2.field.flow = 2.0
    f2.field.use_smoke_density = True
    f2.field.falloff_power = 1.0
    f2.field.use_max_distance = True
    f2.field.distance_max = 1.2
    f3 = field("FLUID", "FPd_FluidNonCubic1", tuple(NC1_AT))
    f3.field.source_object = nc1
    f3.field.strength = 1.0
    f3.field.flow = 1.0
    f4 = field("FLUID", "FPe_FluidNonCubic2", tuple(NC2_AT))
    f4.field.source_object = nc2
    f4.field.strength = 1.0
    f4.field.flow = 1.0

    # ----- the cloth --------------------------------------------------------
    def mesh_object(name, verts, faces):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(float(c) for c in v) for v in verts], [],
                       [tuple(int(i) for i in f) for f in faces])
        me.update()
        ob = bpy.data.objects.new(name, me)
        scene.collection.objects.link(ob)
        return ob

    def triangles(centers, t1s, t2s, radius):
        # One small triangle per center, its three vertices in the plane of
        # t1 and t2, so they share the center's distance from a face.
        verts, faces = [], []
        for c, t1, t2 in zip(centers, t1s, t2s):
            c, t1, t2 = (np.asarray(x, dtype=np.float64) for x in (c, t1, t2))
            base = len(verts)
            for ang in (0.0, 2.0943951, 4.1887902):
                verts.append(c + radius * (math.cos(ang) * t1 + math.sin(ang) * t2))
            faces.append((base, base + 1, base + 2))
        return np.array(verts), faces

    # Jet: a 7 x 7 sheet across the jet, its plane normal to Blender X.
    ys = np.linspace(-0.25, 0.25, 7)
    jet_verts = np.array([(-0.45, y, z) for z in ys for y in ys])
    jet_faces = []
    for j in range(6):
        for i in range(6):
            a = j * 7 + i
            jet_faces += [(a, a + 1, a + 8), (a, a + 8, a + 7)]

    # Faces: triangles 0.2 cell inside and outside the main domain's -X and
    # +Y faces, laid out in the domain's own coordinates (a size-2 cube, so
    # a cell there is 2 / 16) and taken to the world by its matrix.
    M = np.array(main.matrix_world, dtype=np.float64)
    local_cell = 2.0 / 16.0
    centers, t1s, t2s = [], [], []
    for side in (-1.0, 1.0):
        off = 1.0 + side * 0.2 * local_cell
        for (u, w) in ((0.2, -0.1), (-0.3, 0.35)):
            centers.append((-off, u, w))
            t1s.append((0.0, 1.0, 0.0))
            t2s.append((0.0, 0.0, 1.0))
        for (u, w) in ((-0.5, 0.0), (0.1, -0.4)):
            centers.append((u, off, w))
            t1s.append((1.0, 0.0, 0.0))
            t2s.append((0.0, 0.0, 1.0))
    local_verts, faces_faces = triangles(centers, t1s, t2s, 0.04)
    faces_verts = local_verts @ M[:3, :3].T + M[:3, 3]
    # +1 outside, -1 inside, per vertex, by construction.
    faces_side = np.repeat([-1.0 if c[0] > -1.0 and c[1] < 1.0 else 1.0
                            for c in centers], 3)

    rng = np.random.default_rng(7)

    def noncubic_points(at, ext, jet_lo, jet_hi, overhang):
        centers, t1s, t2s, kinds = [], [], [], []
        ex, ey, ez = np.eye(3)
        placed = []
        while len(placed) < 14:
            if len(placed) < 10:
                c = jet_lo + rng.random(3) * (jet_hi - jet_lo)
            else:
                c = -ext / 2.0 + 0.15 + rng.random(3) * (ext - 0.3)
            # Triangles of radius 0.02 whose centers are 0.06 apart cannot
            # touch, so the build's intersection check stays clean.
            if any(np.linalg.norm(c - q) < 0.06 for q in placed):
                continue
            placed.append(c)
            centers.append(at + c)
            t1s.append(ex)
            t2s.append(ey)
            kinds.append("interior")
        for ax in range(3):
            for side in (-1.0, 1.0):
                for d in (-0.03, 0.03):
                    p = np.zeros(3)
                    p[ax] = side * (ext[ax] / 2.0 + d)
                    p[(ax + 1) % 3] = 0.1
                    others = [e for i, e in enumerate((ex, ey, ez)) if i != ax]
                    centers.append(at + p)
                    t1s.append(others[0])
                    t2s.append(others[1])
                    kinds.append("inside" if d < 0.0 else "outside")
        for p in overhang:
            ax = int(np.argmax(np.abs(p)))
            others = [e for i, e in enumerate((ex, ey, ez)) if i != ax]
            centers.append(at + np.asarray(p))
            t1s.append(others[0])
            t2s.append(others[1])
            kinds.append("overhang")
        verts, faces = triangles(centers, t1s, t2s, 0.02)
        return verts, faces, np.repeat(kinds, 3)

    nc1_verts, nc1_faces, nc1_kind = noncubic_points(
        NC1_AT, NC1_EXT, np.array([-1.7, -0.45, -0.6]), np.array([-0.2, 0.5, 0.6]), [])
    # Case 2's grid is 15 x 15 x 16 cells; a uniform cell would overhang its
    # +X and +Y faces by 6.25 cm, so two triangles sit 3 cm past each.
    nc2_verts, nc2_faces, nc2_kind = noncubic_points(
        NC2_AT, NC2_EXT, np.array([-0.9, -0.5, -0.6]), np.array([0.3, 0.6, 0.6]),
        [(1.03, -0.3, 0.2), (-0.3, 1.03, -0.2)])

    CLOTH = {}
    for name, verts, faces in (("Jet", jet_verts, jet_faces),
                               ("Faces", faces_verts, faces_faces),
                               ("NC1", nc1_verts, nc1_faces),
                               ("NC2", nc2_verts, nc2_faces)):
        CLOTH[name] = (mesh_object("FP" + name, verts, faces),
                       mesh_object("FP" + name + "Moving", verts, faces),
                       np.asarray(verts, dtype=np.float64))

    scene.frame_set(1)
    dh.save_blend(PROBE_DIR, "force_field_fluid_parity.blend")
    root = dh.configure_state(project_name="force_field_fluid_parity",
                              frame_count=FRAME_COUNT, frame_rate=100, step_size=DT)
    state = root.state
    state.use_scene_frame_start = False
    state.frame_start = START
    state.use_scene_fps = False
    state.time_scale = 1.0
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    speed = float(np.linalg.norm(V_REF))
    for name, (rest_obj, moving_obj, _) in CLOTH.items():
        cloth.add(rest_obj.name)
        cloth.add(moving_obj.name)
        cloth.set_velocity(moving_obj.name, direction=tuple(V_REF / speed), speed=speed,
                           frame=START)
    cloth.param.shell_young_modulus = 0.01
    cloth.param.bend = 0.0
    cloth.param.enable_strain_limit = False
    # Each moving copy lies exactly on its resting twin; the build's
    # intersection check refuses that unless the pairs are declared
    # non-contact, which is what this allowance says.
    dh.groups.get_group_by_uuid(scene, cloth.uuid).allow_inter_object_intersection = True

    recs = payload_records()
    dh.log("payload decoded: %d fields" % len(recs))

    # ----- the solve --------------------------------------------------------
    def build_cycle(message, timeout=300.0):
        data_hash = dh.encoder_mesh.compute_data_hash(bpy.context)
        param_hash = dh.encoder_param.compute_param_hash(bpy.context)
        data_bytes = dh.encoder_mesh.encode_obj(bpy.context)
        param_bytes = dh.encoder_param.encode_param(bpy.context)
        dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
            data=data_bytes, param=param_bytes, data_hash=data_hash,
            param_hash=param_hash, message=message))
        deadline = time.time() + timeout
        while time.time() < deadline:
            dh.facade.engine.dispatch(dh.events.PollTick())
            dh.facade.tick()
            s = dh.facade.engine.state
            if (s.activity.name == "IDLE"
                    and s.solver.name in ("READY", "RESUMABLE", "FAILED")
                    and s.server_data_hash == data_hash
                    and s.server_param_hash == param_hash):
                break
            time.sleep(0.2)
        s = dh.facade.engine.state
        if s.solver.name == "FAILED":
            raise RuntimeError(build_failure_message(dh.facade, dh.com))
        if s.server_param_hash != param_hash or s.solver.name not in ("READY", "RESUMABLE"):
            raise RuntimeError("%s: the build did not land (solver=%s activity=%s)"
                               % (message, s.solver.name, s.activity.name))

    def solve(message, frames):
        build_cycle(message)
        dh.run_and_wait(timeout=600.0)
        if dh.facade.engine.state.solver.name == "FAILED":
            raise RuntimeError("%s: the run failed: %r" % (message, dh.facade.engine.state.error))
        dh.force_frame_query(expected_frames=frames - 1, timeout=120.0)
        dh.settle_idle(timeout=15.0)
        dh.fetch_and_drain()

    def rows(obj):
        path = dh.find_pc2_for(obj)
        if not path:
            raise RuntimeError("no PC2 for %s" % obj.name)
        arr = np.asarray(dh.read_pc2(path), dtype=np.float64)
        m = np.array(obj.matrix_world, dtype=np.float64)
        return arr @ m[:3, :3].T + m[:3, 3]

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=state.project_name)
    solve("force_field_fluid_parity:build", FRAME_COUNT)
    SOLVED = {name: (rows(ro), rows(mo)) for name, (ro, mo, _) in CLOTH.items()}
    result["rows"] = {name: list(v[0].shape) for name, v in SOLVED.items()}
    short = {name: [len(v[0]), len(v[1])] for name, v in SOLVED.items()
             if len(v[0]) != FRAME_COUNT or len(v[1]) != FRAME_COUNT}
    if short:
        raise RuntimeError("PC2 rows short of the %d frames: %s" % (FRAME_COUNT, short))
    dh.log("solved")

    # ----- Blender's own particles: the oracle -------------------------------
    EMITTERS = {}

    def emitter(tag, points, frame, v0, timestep=DT, subframes=0):
        me = bpy.data.meshes.new("FPEmit_" + tag)
        me.from_pydata([tuple(float(c) for c in p) for p in points], [], [])
        ob = bpy.data.objects.new("FPEmit_" + tag, me)
        scene.collection.objects.link(ob)
        ob.modifiers.new("ps", "PARTICLE_SYSTEM")
        ps = ob.particle_systems[0].settings
        ps.count = len(points)
        ps.frame_start = frame
        ps.frame_end = frame
        ps.lifetime = 1000
        ps.emit_from = "VERT"
        ps.use_emit_random = False
        ps.normal_factor = 0.0
        ps.object_align_factor = tuple(float(c) for c in v0)
        ps.factor_random = 0.0
        ps.physics_type = "NEWTON"
        ps.mass = 1.0
        ps.effector_weights.gravity = 0.0
        ps.use_rotations = False
        ps.integrator = "EULER"
        ps.drag_factor = ps.brownian_factor = ps.damping = 0.0
        ps.timestep = timestep
        ps.subframes = subframes
        EMITTERS[tag] = {"obj": ob, "frame": frame, "points": np.asarray(points, dtype=np.float64),
                         "v0": np.asarray(v0, dtype=np.float64), "timestep": timestep}

    def walk(last):
        got = {}
        for fr in range(1, last + 1):
            scene.frame_set(fr)
            dg = bpy.context.evaluated_depsgraph_get()
            for tag, e in EMITTERS.items():
                if fr not in (e["frame"], e["frame"] + 1):
                    continue
                parts = e["obj"].evaluated_get(dg).particle_systems[0].particles
                n = len(parts)
                loc = np.empty(3 * n, dtype=np.float32)
                vel = np.empty(3 * n, dtype=np.float32)
                parts.foreach_get("location", loc)
                parts.foreach_get("velocity", vel)
                got[(tag, fr)] = (loc.reshape(n, 3).astype(np.float64),
                                  vel.reshape(n, 3).astype(np.float64))
        return got

    def response(got, tag):
        # Each particle matched to its vertex by position at its birth frame;
        # returns the acceleration over the frame and the bookkeeping errors.
        e = EMITTERS[tag]
        loc0, vel0 = got[(tag, e["frame"])]
        loc1, vel1 = got[(tag, e["frame"] + 1)]
        if len(loc0) != len(e["points"]):
            raise RuntimeError("emitter %s holds %d particles for %d points"
                               % (tag, len(loc0), len(e["points"])))
        d = np.linalg.norm(e["points"][:, None, :] - loc0[None, :, :], axis=2)
        perm = np.argmin(d, axis=1)
        match = float(d[np.arange(len(perm)), perm].max())
        unique = len(set(perm.tolist())) == len(perm)
        v0, v1 = vel0[perm], vel1[perm]
        born = float(np.abs(v0 - e["v0"]).max())
        return ((v1 - v0) / e["timestep"], v0,
                {"match": match, "unique": unique, "born": born, "end": loc1[perm]})

    def coefficients(got, tag):
        # A, the acceleration at rest, and C, the damping per unit velocity,
        # from two systems at the same points and different velocities.
        a_rest, v_rest, b_rest = response(got, tag + "_rest")
        a_mov, v_mov, b_mov = response(got, tag + "_moving")
        b_rest.pop("end")
        b_mov.pop("end")
        dv = v_mov - v_rest
        C = -np.sum((a_mov - a_rest) * dv, axis=1) / np.sum(dv * dv, axis=1)
        book = {"match": max(b_rest["match"], b_mov["match"]),
                "unique": b_rest["unique"] and b_mov["unique"],
                "born": max(b_rest["born"], b_mov["born"])}
        return a_rest, a_mov, C, book

    def pair(tag, points, frame):
        emitter(tag + "_rest", points, frame, (0.0, 0.0, 0.0))
        emitter(tag + "_moving", points, frame, V_REF)

    for name in CLOTH:
        pair("start_" + name, CLOTH[name][2], START)
    J, Jv = SOLVED["Jet"]
    # At the solver's own end-of-step positions, where it evaluates the
    # fluid (the implicit term is read at the Newton iterate).
    for k in range(FRAME_COUNT - 1):
        pair("end_rest_%d" % k, J[k + 1], START + k)
    pair("end_moving_0", Jv[1], START)
    for name in ("NC1", "NC2"):
        pair("end_" + name, SOLVED[name][0][1], START)
    got = walk(LAST)
    dh.log("particles walked")

    BOOK = {}

    def coeff(tag):
        A, a_mov, C, book = coefficients(got, tag)
        BOOK[tag] = book
        return A, a_mov, C

    # ===== A, B: the payload against Blender at the start points ============
    pay = []
    slip = []
    for name in ("Jet", "Faces"):
        x8 = CLOTH[name][2]
        A, a_mov, C = coeff("start_" + name)
        ref_rest = reference_accel(recs, x8, np.zeros(3), DT)
        ref_mov = reference_accel(recs, x8, V_REF, DT)
        scale = float(np.abs(A).max())
        pay.append({"set": name, "scale": rnd(scale, 4),
                    "rest": float(np.abs(ref_rest - A).max()),
                    "moving": float(np.abs(ref_mov - a_mov).max())})
        if name == "Jet":
            early = reference_accel(recs, x8, np.zeros(3), 0.0)
            slip.append({"set": name, "scale": rnd(scale, 4),
                         "relative": rnd(float(np.abs(early - A).max()) / scale, 4)})
    worst_pay = max(max(p["rest"], p["moving"]) / p["scale"] for p in pay)
    dh.record("A_the_payload_reads_blenders_acceleration",
              worst_pay < 1e-5 and pay[0]["scale"] > 1.0,
              {"sets": pay, "worst_relative": worst_pay, "book": BOOK})
    dh.record("B_a_read_one_frame_early_would_be_seen",
              min(s["relative"] for s in slip) >= 0.3, {"sets": slip})

    # ===== C, D: one solver step against Blender's coefficients =============
    A_end, _, C_end = coeff("end_rest_0")
    a_S = (J[1] - J[0]) / DT ** 2
    want = A_end / (1.0 + DT * C_end[:, None])
    scale = float(np.abs(A_end).max())
    A0, _, C0 = coeff("start_Jet")
    want_start = A0 / (1.0 + DT * C0[:, None])
    at_start = float(np.abs(a_S - want_start).max()) / scale
    rest_err = float(np.abs(a_S - want).max()) / scale
    # How far the step read at the start lies from the one read at the end:
    # a solver reading the fluid at the start lands near the first, so C
    # rejects it only while this is well past C's bound.
    separation = float(np.abs(want_start - want).max()) / scale
    dh.record("C_a_resting_sheet_takes_blenders_step",
              rest_err < 1e-3 and separation > 1.5e-3 and scale > 1.0,
              {"relative": rest_err, "scale": rnd(scale, 4),
               "start_read_separation": separation,
               "relative_with_coefficients_at_the_step_start": at_start,
               "damping_C": [rnd(C_end.min(), 4), rnd(C_end.max(), 4)]})

    Am, _, Cm = coeff("end_moving_0")
    v1 = (Jv[1] - Jv[0]) / DT
    a_moving = (v1 - V_REF) / DT
    want_m = ((V_REF + DT * Am) / (1.0 + DT * Cm[:, None]) - V_REF) / DT
    scale_m = float(np.abs(Am).max())
    mov_err = float(np.abs(a_moving - want_m).max()) / scale_m
    dh.record("D_a_moving_sheet_takes_blenders_step",
              mov_err < 5e-3 and scale_m > 1.0,
              {"relative": mov_err, "scale": rnd(scale_m, 4)})

    # ===== E: presence at the main domain's faces ===========================
    def solver_C(rest_rows, moving_rows):
        a_r = (rest_rows[1] - rest_rows[0]) / DT ** 2
        a_m = ((moving_rows[1] - moving_rows[0]) / DT - V_REF) / DT
        c = -np.sum((a_m - a_r) * V_REF, axis=1) / float(np.dot(V_REF, V_REF))
        return c / (1.0 - DT * c), a_r

    Fr, Fv = SOLVED["Faces"]
    C_S, _ = solver_C(Fr, Fv)
    _, _, C_B = coeff("start_Faces")
    inside_B = C_B > 0.5
    inside_S = C_S > 0.5
    built_inside = faces_side < 0.0
    dh.record("E_presence_at_the_domain_faces_is_blenders",
              bool(np.all(inside_B == inside_S)) and bool(np.all(inside_B == built_inside))
              and int(inside_B.sum()) >= 6 and int((~inside_B).sum()) >= 6
              and float(np.abs(C_S - C_B).max()) < 0.05,
              {"mismatches": int((inside_B != inside_S).sum()),
               "C_difference": rnd(np.abs(C_S - C_B).max(), 5),
               "blender_inside": int(inside_B.sum()), "vertices": len(C_B),
               "C_blender_inside": span(C_B[inside_B], 4),
               "C_blender_outside": span(C_B[~inside_B], 6),
               "C_solver_inside": span(C_S[inside_S], 4),
               "C_solver_outside": span(C_S[~inside_S], 6)})

    # ===== F, G: the trajectory, every step held to Blender =================
    steps = []
    for k in range(FRAME_COUNT - 1):
        A, _, C = coeff("end_rest_%d" % k)
        v_k = np.zeros(3) if k == 0 else (J[k] - J[k - 1]) / DT
        v_next = (J[k + 1] - J[k]) / DT
        want = (v_k + DT * A) / (1.0 + DT * C[:, None])
        steps.append({"step": "%d->%d" % (START + k, START + k + 1),
                      "scale": rnd(float(np.abs(A).max()), 4),
                      "error": float(np.abs(v_next - want).max()) / DT,
                      "speed": rnd(float(np.linalg.norm(v_next, axis=1).max()), 4)})
    # Judged against the largest field of the run, the scale G's frame
    # differences are measured on: a weak step is not held to a tighter
    # absolute bound than a strong one.
    peak = max(s["scale"] for s in steps)
    for s_ in steps:
        s_["relative"] = rnd(s_.pop("error") / peak, 6)
    worst_step = max(s_["relative"] for s_ in steps)
    dh.record("F_every_streamed_step_takes_blenders_step",
              len(steps) == FRAME_COUNT - 1 and worst_step < 1e-2 and peak > 1.0,
              {"steps": steps, "worst_relative": worst_step, "peak": peak})

    frames_A = [reference_accel(recs, jet_verts, np.zeros(3), k * DT)
                for k in range(FRAME_COUNT)]
    scale_f = max(float(np.abs(a).max()) for a in frames_A)
    pairs = []
    for i in range(FRAME_COUNT):
        for j in range(i + 1, FRAME_COUNT):
            pairs.append((rnd(float(np.abs(frames_A[i] - frames_A[j]).max()) / scale_f, 4),
                          i, j))
    matrix = [[0.0] * FRAME_COUNT for _ in range(FRAME_COUNT)]
    for d_, i, j in pairs:
        matrix[i][j] = matrix[j][i] = round(d_, 2)
    pairs.sort()
    dh.record("G_every_fluid_frame_differs_from_every_other",
              pairs[0][0] >= 0.2,
              {"closest_pairs": pairs[:4], "scale": rnd(scale_f, 4), "matrix": matrix})

    # ===== H, I: the non-cubic domains ======================================
    def noncubic(case, ext, kinds):
        Nr, Nv = SOLVED[case]
        C_S, a_r = solver_C(Nr, Nv)
        _, _, C_B = coeff("start_" + case)
        in_B = C_B > 0.5
        in_S = C_S > 0.5
        face = kinds != "interior"
        expect_in = (kinds == "interior") | (kinds == "inside")
        presence = {"mismatches": int((in_B != in_S).sum()),
                    "blender_unlike_construction": int((in_B != expect_in).sum()),
                    "vertices": len(in_B), "outside_vertices": int((~expect_in).sum()),
                    "C_blender_inside": span(C_B[in_B], 4),
                    "C_blender_outside": span(C_B[~in_B], 6),
                    "C_solver_inside": span(C_S[in_S], 4),
                    "C_solver_outside": span(C_S[~in_S], 6)}
        presence_ok = (presence["mismatches"] == 0
                       and presence["blender_unlike_construction"] == 0)
        # SPEED at interior vertices the fluid moves: the solver's undamped
        # push has Blender's speed, both read where the solver read it.
        A, _, C_end = coeff("end_" + case)
        scale = float(np.linalg.norm(A, axis=1).max())
        moving = (~face) & (np.linalg.norm(A, axis=1) > 0.05 * scale)
        undamped = a_r * (1.0 + DT * C_end[:, None])
        sp_S = np.linalg.norm(undamped, axis=1)
        sp_B = np.linalg.norm(A, axis=1)
        speed_err = float(np.abs(sp_S - sp_B)[moving].max()) / scale
        # DIRECTION, the deliberate difference: Blender skews the direction
        # by diag(extent / longest extent) in the domain's axes and keeps the
        # speed; the solver reads the grid's own vector. So the solver's push
        # is Blender's speed along diag(longest extent / extent) times
        # Blender's vector.
        unskew = A * (float(ext.max()) / ext)
        unskew = unskew / np.maximum(np.linalg.norm(unskew, axis=1), 1e-30)[:, None] * sp_B[:, None]
        dir_err = float(np.abs(undamped - unskew)[moving].max()) / scale
        raw_err = float(np.abs(undamped - A)[moving].max()) / scale
        detail = {"presence": presence, "moving_vertices": int(moving.sum()),
                  "scale": rnd(scale, 4), "speed_relative": speed_err,
                  "unskewed_vector_relative": dir_err,
                  "blender_vector_relative": rnd(raw_err, 4)}
        return presence_ok, speed_err, dir_err, raw_err, int(moving.sum()), detail

    for letter, case, ext, kinds in (("H", "NC1", NC1_EXT, nc1_kind),
                                     ("I", "NC2", NC2_EXT, nc2_kind)):
        presence_ok, speed_err, dir_err, raw_err, moving, detail = noncubic(case, ext, kinds)
        dh.record("%s_noncubic_%s_presence_and_speed_are_blenders" % (letter, case.lower()),
                  presence_ok and speed_err < 5e-3 and dir_err < 5e-3 and moving >= 10,
                  detail)

    dh.record("Z_bookkeeping_particles_match_their_vertices",
              all(b["match"] < 1e-5 and b["unique"] and b["born"] < 1e-6
                  for b in BOOK.values()),
              {"worst_match": max(b["match"] for b in BOOK.values()),
               "worst_born_velocity": max(b["born"] for b in BOOK.values())})

    # ===== U: several solver steps per Blender frame ========================
    # The same baked grids, at scene fps 25 and step 0.01: four solver steps
    # a frame, Strength 1, Flow 0, one field.
    for tag, e in list(EMITTERS.items()):
        bpy.data.objects.remove(e["obj"])
    EMITTERS.clear()
    dh.api.solver.delete_all_groups()
    for obj in [f0, f2, f3, f4] + [o for v in CLOTH.values() for o in v[:2]]:
        bpy.data.objects.remove(obj)
    f1.field.strength = 1.0
    f1.field.flow = 0.0
    scene.render.fps = 25
    state.frame_rate = 25
    STEPS = 4
    FRAME_DT = STEPS * DT
    jet_u = mesh_object("FPJetSubsteps", jet_verts, jet_faces)
    cloth_u = dh.api.solver.create_group("ClothSubsteps", "SHELL")
    cloth_u.add(jet_u.name)
    cloth_u.param.shell_young_modulus = 0.01
    cloth_u.param.bend = 0.0
    cloth_u.param.enable_strain_limit = False
    recs_u = payload_records()
    solve("force_field_fluid_parity:substeps", FRAME_COUNT)
    U = rows(jet_u)
    if len(U) != FRAME_COUNT:
        raise RuntimeError("the substep solve's PC2 holds %d rows, not %d"
                           % (len(U), FRAME_COUNT))
    x0, x1 = U[0], U[1]
    # Where the sheet is after substep j of the frame, from rest under a
    # nearly constant push: x0 + (x1 - x0) j (j + 1) / 20.
    xs = [x0 + (x1 - x0) * (j * (j + 1) / 20.0) for j in range(1, STEPS + 1)]
    A8 = [reference_accel(recs_u, x, np.zeros(3), 0.0) for x in xs]
    A9 = [reference_accel(recs_u, x, np.zeros(3), FRAME_DT) for x in xs]
    # x1 - x0 = dt^2 sum_j (STEPS + 1 - j) a_j, with a_j read at the blend
    # b_j = j / STEPS (the solver) or at frame 9 throughout (Blender).
    blend = sum((STEPS + 1 - j) * ((1.0 - j / STEPS) * A8[j - 1] + (j / STEPS) * A9[j - 1])
                for j in range(1, STEPS + 1)) * DT ** 2
    hold = sum((STEPS + 1 - j) * A9[j - 1] for j in range(1, STEPS + 1)) * DT ** 2
    moved = x1 - x0
    sc = float(np.abs(moved).max())
    blend_err = float(np.abs(moved - blend).max()) / sc
    hold_err = float(np.abs(moved - hold).max()) / sc
    apart = float(np.abs(blend - hold).max()) / sc
    # Blender's own particles over the same frame, four substeps of 0.01.
    emitter("substeps_rest", x0, START, (0.0, 0.0, 0.0), timestep=FRAME_DT, subframes=STEPS - 1)
    got_u = walk(START + 1)
    a_B, _, book_u = response(got_u, "substeps_rest")
    gain_B = a_B * FRAME_DT
    xm = 0.5 * (x0 + book_u.pop("end"))
    gain_hold = FRAME_DT * reference_accel(recs_u, xm, np.zeros(3), FRAME_DT)
    gain_blend = FRAME_DT * (0.375 * reference_accel(recs_u, xm, np.zeros(3), 0.0)
                             + 0.625 * reference_accel(recs_u, xm, np.zeros(3), FRAME_DT))
    gsc = float(np.abs(gain_B).max())
    blender_hold = float(np.abs(gain_B - gain_hold).max()) / gsc
    blender_blend = float(np.abs(gain_B - gain_blend).max()) / gsc
    dh.record("U_substeps_blend_the_two_bracketing_frames",
              blend_err < 5e-3 and apart > 0.1 and sc > 1e-4,
              {"blend_relative": blend_err, "hold_relative": rnd(hold_err, 4),
               "blend_vs_hold": rnd(apart, 4), "moved": rnd(sc, 6)})
    dh.record("V_blender_holds_the_next_frame_a_recorded_difference",
              blender_hold < 1e-2 and blender_blend > 0.1 and book_u["match"] < 1e-5,
              {"hold_relative": blender_hold, "blend_relative": rnd(blender_blend, 4),
               "gain_scale": rnd(gsc, 4), "book": book_u})

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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 900.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
