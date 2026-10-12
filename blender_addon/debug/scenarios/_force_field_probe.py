# File: scenarios/_force_field_probe.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The shared probe behind the `rig_force_field_*` and `rig_frame_step`
# scenarios: the external force field (issues #151 and #114) and the
# frame-step API, driven through the public frontend and judged by the frames
# the solver wrote.
#
# EVERY CASE IS A REAL SOLVE, and the verdicts are about MOTION rather than
# about exit status, because a field that never reached the target would
# still finish cleanly. The controls matter as much as the positives: a
# weight-0 object, a sheet a field's falloff does not reach and a Wind in air
# of no density each must stay still, which is what separates "the field
# acts" from "something moved".
#
# A FIELD OBJECT IS HELD TO ITS REFERENCE NUMERICALLY. One step from rest on a
# sheet with almost no stiffness moves each vertex by dt^2 times its
# acceleration, so the displacement read back is the kernel's field at each
# vertex, compared against `frontend._field_eval_`, the double-precision
# statement of every field law.
#
# The probe runs in a SUBPROCESS for the reason `rig_lock_axes` gives: it
# imports `frontend`, which loads the per-tree cdylib and installs the debug
# patches, and the orchestrator must not inherit either.

from __future__ import annotations

import json
import os
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


PROBE = r'''
import json
import math
import os
import sys
import time

import numpy as np

REPO_ROOT = sys.argv[1]
SECTION = sys.argv[2]
sys.path.insert(0, REPO_ROOT)

from frontend._debug_runtime_ import install_debug_patches
install_debug_patches()

from frontend import App, FluidVolume, ForceFieldScriptError
from frontend import _rust

cases = {}


def check(name, ok, details):
    cases[name] = {"ok": bool(ok), "details": details}
    print("PPFCASE %s %s" % (name, "pass" if ok else "fail"), flush=True)


def begin(label):
    print("PPFSTART " + label, flush=True)


def new_app(label):
    app = App.create("rig_force_field_" + label)
    V, F = app.mesh.square(res=6, size=0.5, ex=[1, 0, 0], ey=[0, 0, 1])
    app.asset.add.tri("sheet", V, F)
    Vv, Fv = app.mesh.square(res=6, size=0.5, ex=[1, 0, 0], ey=[0, 1, 0])
    app.asset.add.tri("wall", Vv, Fv)
    return app


def solve(app, scene, frames, params=None, blocking=True):
    fixed = scene.build(quiet=True)
    session = app.session.create(fixed)
    session.param.set("frames", frames).set("gravity", [0.0, 0.0, 0.0])
    for k, v in (params or {}).items():
        session.param.set(k, v)
    session = session.build()
    if blocking:
        try:
            session.start(blocking=True)
        except Exception as e:  # a refused run raises; the caller reads the log
            print("PPFNOTE start raised: %r" % (e,), flush=True)
    return fixed, session


def frames_of(session, n):
    got, _ = session.get.vertex(n)
    return np.asarray(got, dtype=np.float64)[:, :3]


# A field frame, columns local X, Y, Z in solver axes, with local Z along +x.
ALONG_X = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def fluid_in_solver_units(f, world_scaling):
    # A Fluid record as the solver holds it under World Scaling ws: its
    # velocity U times ws. (Its corner and cell are times ws in solver units
    # too; `step_reference` reads every field at the scene point, where the
    # unscaled corner and cell are that placement.) Any other record as it is.
    if f["type"] != "fluid" or world_scaling == 1.0:
        return f
    volume = dict(f["fluid"])
    bricks = []
    for k in range(len(volume["tables"])):
        b = np.array(volume["bricks"][k], dtype=np.float64).reshape(-1, 512, 4)
        b[..., :3] *= world_scaling
        bricks.append(b)
    volume["bricks"] = bricks
    record = dict(f)
    record["fluid"] = volume
    return record


def step_reference(fields, x, v0, dt, normals=None, time=0.0, fluid_time=None,
                   world_scaling=1.0, playback=1.0, fluid_playback=None):
    # The reference for one backward Euler step of a free vertex, solving
    # what the solver solves, in SOLVER units (scene units times World
    # Scaling ws): (Y - X - V0 dt) / dt^2 equals the explicit laws at the
    # step's start (X, V0) plus the implicit ones at the iterate
    # (Y, (Y - X) / dt), by Newton with a finite-difference Jacobian; the
    # result is returned in scene units, Y / ws. As in the solver, a field is
    # read at the SCENE point (it is authored in scene units), the velocity it
    # reads is the solver's, and its result is an acceleration World Scaling
    # never scales. A Fluid's volume sits at corner * ws with cells of
    # cell * ws in solver units, which is where its corner and cell place it
    # at the scene point, and its velocity enters as U * ws
    # (`fluid_in_solver_units`). It is read at `fluid_time`, by default the
    # step's END (`time + dt`), as the solver reads it.
    #
    # PLAYBACK. `dt` is the step on the scene CLOCK, where `time`, the
    # fluid's frames and the frames the solver writes sit; at Playback Speed
    # `playback` the step moves the vertices by `dt * playback` seconds of
    # motion (step.rs), so the velocities here, `v0` included, are per
    # physical second, and the fluid's U, per clock second, is divided by
    # `fluid_playback` (by default `playback`, as the solver divides it).
    from frontend._field_eval_ import evaluate as reference
    ws = float(world_scaling)
    if fluid_time is None:
        fluid_time = time + dt
    if fluid_playback is None:
        fluid_playback = playback
    dtp = dt * float(playback)
    solver_fields = [fluid_in_solver_units(f, ws) for f in fields]
    x = np.asarray(x, dtype=np.float64)
    X = ws * x
    V0 = ws * np.asarray(v0, dtype=np.float64)
    explicit = sum(reference(f, x, V0, time, normals=normals, part="explicit")[0]
                   for f in fields)

    def implicit_at(Y):
        return sum(reference(f, Y / ws, (Y - X) / dtp, time, part="implicit",
                             fluid_time=fluid_time, playback=fluid_playback)[0]
                   for f in solver_fields)

    Y = X + V0 * dtp + dtp * dtp * explicit
    h = 1e-6
    for _ in range(60):
        residual = (Y - X - V0 * dtp) / (dtp * dtp) - explicit - implicit_at(Y)
        jac = np.zeros((len(x), 3, 3))
        for k in range(3):
            e = np.zeros(3)
            e[k] = h
            jac[:, :, k] = np.eye(3)[k] / (dtp * dtp) - (
                implicit_at(Y + e) - implicit_at(Y - e)) / (2.0 * h)
        dY = np.linalg.solve(jac, -residual[..., None])[..., 0]
        Y = Y + dY
        if float(np.abs(dY).max()) < 1e-13:
            break
    return Y / ws


def one_step_acceleration(label, place, add_fields, params=None):
    # The field at every vertex of a soft sheet placed at `place`, read as
    # the displacement of one step from rest over dt^2, in scene units; and
    # the reference at the same vertices. World Scaling divides the
    # displacement read back in scene units, since the field's result is an
    # acceleration it never scales.
    from frontend._field_eval_ import evaluate
    app = new_app(label)
    scene = app.scene.create()
    obj = scene.add("sheet").at(*place)
    obj.param.set("young-mod", 1e-3).set("bend", 0.0)
    add_fields(scene.force_field)
    dt = 1e-2
    run_params = {"dt": dt, "fps": 100.0, "air-density": 0.0}
    run_params.update(params or {})
    fixed, session = solve(app, scene, 1, run_params)
    p0 = frames_of(session, 0)
    scaling = float(run_params.get("world-scaling", 1.0))
    got = (frames_of(session, 1) - p0) / (dt * dt) * scaling
    if scaling == 1.0:
        want = (step_reference(scene.force_field.fields, p0, np.zeros_like(p0), dt) - p0) / (dt * dt)
    else:
        # Under World Scaling the implicit terms would scale with it; the
        # checks run there use explicit fields only.
        want = sum(evaluate(f, p0, np.zeros_like(p0), 0.0, part="explicit")[0]
                   for f in scene.force_field.fields)
    scale = float(np.abs(want).max())
    err = float(np.abs(got - want).max() / scale) if scale > 0.0 else float("inf")
    return session, err, scale


def log_text(session):
    out, err = _rust.stdout_error_log_paths(session.info.path)
    text = ""
    for p in (out, err):
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                text += f.read()
    return text


# ---------------------------------------------------------------------------
if SECTION == "sources":
    # A uniform field object and a constant script each reproduce plain
    # gravity: all three enter the same target, so a single sheet with no
    # contact moves identically. A Plane Force pushes AWAY from its plane, so
    # the uniform downward push is a plane facing up placed ABOVE the sheet.
    frames = 10
    results = {}
    for label in ("gravity", "field", "script"):
        begin(label)
        app = new_app("sources_" + label)
        scene = app.scene.create()
        scene.add("sheet").at(0.0, 1.0, 0.0)
        params = {}
        if label == "gravity":
            params["gravity"] = [0.0, -9.8, 0.0]
        elif label == "field":
            scene.force_field.add("force", shape="plane", strength=9.8,
                                  position=(0.0, 10.0, 0.0))
        else:
            scene.force_field.script(
                "def eval(x, y, z, t):\n    return (0.0, -9.8, 0.0)\n")
        _, session = solve(app, scene, frames, params)
        d = frames_of(session, frames) - frames_of(session, 0)
        results[label] = {"finished": bool(session.finished()),
                          "mean": d.mean(axis=0).tolist()}
    g = np.array(results["gravity"]["mean"])
    for label in ("field", "script"):
        m = np.array(results[label]["mean"])
        ok = (results[label]["finished"] and g[1] < -0.05
              and float(np.abs(m - g).max()) <= 1e-6 + 1e-4 * abs(g[1]))
        check("uniform_%s_matches_gravity" % label, ok,
              {"gravity": g.tolist(), label: m.tolist()})

    # The per-object weight: a weight-0 sheet beside a weight-1 sheet.
    begin("weight")
    app = new_app("weight")
    scene = app.scene.create()
    scene.add("sheet").at(-1.0, 1.0, 0.0)
    held = scene.add("sheet").at(1.0, 1.0, 0.0)
    held.param.set("force-field-weight", 0.0)
    scene.force_field.script("def eval(x, y, z, t):\n    return (0.0, -9.8, 0.0)\n")
    fixed, session = solve(app, scene, 10)
    d = frames_of(session, 10) - frames_of(session, 0)
    x0 = frames_of(session, 0)[:, 0]
    moved = d[x0 < 0][:, 1].mean()
    still = np.abs(d[x0 > 0]).max()
    check("weight_zero_opts_an_object_out", session.finished() and moved < -0.05 and still < 1e-7,
          {"weighted_dy": float(moved), "unweighted_max_disp": float(still)})

# ---------------------------------------------------------------------------
elif SECTION == "spatial":
    # The kernel against the reference, one configuration per solve so a
    # failure names the law: a Point Force with a falloff and a maximum, a
    # rotated Plane Vortex, a Turbulence with Noise Amount, and a Cone with Z
    # Direction and Gravitation.
    cases_ = {
        "point_force": lambda ff: ff.add(
            "force", position=(0.3, 0.6, -0.2), strength=3.0, power=1.0,
            max_distance=3.0),
        "plane_vortex": lambda ff: ff.add(
            "vortex", position=(0.1, 0.0, 0.05), shape="plane", strength=2.0,
            power=1.0, axes=ALONG_X),
        "turbulence_with_noise": lambda ff: ff.add(
            "turbulence", position=(0.0, 1.0, 0.0), strength=4.0, size=0.4,
            noise=0.5, seed=3),
        "cone_gravitation": lambda ff: ff.add(
            "force", position=(0.0, 0.2, 0.0), falloff="cone", radial_power=0.5,
            radial_max=80.0, z_direction="positive", gravitation=True, strength=2.0),
        # Noise Amount sized by a NON-UNIFORM object scale, the path a
        # Turbulence (sized by its Size) does not take.
        "force_noise_scaled": lambda ff: ff.add(
            "force", position=(0.3, 0.6, -0.2), strength=3.0, noise=0.8,
            noise_scale=(0.3, 0.6, 0.45), seed=4, axes=ALONG_X),
    }
    for name, add_fields in cases_.items():
        begin(name)
        session, err, scale = one_step_acceleration(name, (0.2, 0.9, -0.3), add_fields)
        check("kernel_%s_matches_the_reference" % name,
              session.finished() and err < 2e-2 and scale > 0.05,
              {"relative_error": err, "max_accel": scale})

    # WIND FACTOR against the reference. The sheet lies in XZ, so every vertex
    # normal is +-Y exactly (the law reads (a . n) n, blind to the sign), and
    # a Plane Force tilted 40 degrees from +Y toward +X pushes it partly
    # along its surface; at Wind Factor 0.5 half of the along-surface push is
    # removed.
    from frontend._field_eval_ import evaluate as reference
    begin("wind_factor")
    c, s_ = math.cos(math.radians(40.0)), math.sin(math.radians(40.0))
    local_z = np.array([s_, c, 0.0])
    local_x = np.array([c, -s_, 0.0])
    tilted = np.stack([local_x, np.cross(local_z, local_x), local_z], axis=1)
    app = new_app("wind_factor")
    scene = app.scene.create()
    obj = scene.add("sheet").at(0.2, 0.9, -0.3)
    obj.param.set("young-mod", 1e-3).set("bend", 0.0)
    scene.force_field.add("force", shape="plane", strength=4.0, axes=tilted,
                          position=(0.0, -1.0, 0.0), wind_factor=0.5)
    dt = 1e-2
    fixed, session = solve(app, scene, 1, {"dt": dt, "fps": 100.0, "air-density": 0.0})
    p0 = frames_of(session, 0)
    got = (frames_of(session, 1) - p0) / (dt * dt)
    normals = np.tile([0.0, 1.0, 0.0], (len(p0), 1))
    field = scene.force_field.fields[0]
    want = reference(field, p0, np.zeros_like(p0), 0.0, normals=normals)[0]
    whole = reference(field, p0, np.zeros_like(p0), 0.0, normals=np.zeros_like(p0))[0]
    err = float(np.abs(got - want).max() / np.abs(want).max())
    check("kernel_wind_factor_matches_the_reference",
          session.finished() and err < 2e-2
          and float(np.abs(want - whole).max()) > 0.5,
          {"relative_error": err, "removed": float(np.abs(want - whole).max())})

    # A WIND'S GUSTS THROUGH THE DRAG. With Air Friction 0 the face drag is
    # linear in the wind's component along the vertex normal and assembles
    # per vertex, so on a flat sheet with almost no stiffness each vertex's
    # normal displacement over one step from rest is a per-vertex constant
    # times that component. The ratio of a run with Noise Amount to one
    # without cancels the constant, leaving the reference's air velocities'
    # ratio. The sheet lies in XZ, so its normal is Y, and the Plane Wind
    # blows along +Y. THE STEP IS SHORT ON PURPOSE: the drag reads the normal
    # of the ITERATE, so uneven gusts bend the sheet within the step and tilt
    # its normals, an error proportional to the displacement over the vertex
    # spacing, measured 0.23, 0.022 and 0.002 at dt = 1e-2, 1e-3 and 1e-4.
    # (The gusts' drift with the flow needs t > 0, which a one-step test
    # cannot reach; the frontend tests hold the reference to it.)
    begin("wind_gusts")
    displacement = {}
    for label, noise in (("plain", 0.0), ("gusts", 0.6)):
        app = new_app("wind_gusts_" + label)
        scene = app.scene.create()
        obj = scene.add("sheet").at(0.2, 0.9, -0.3)
        obj.param.set("young-mod", 1e-3).set("bend", 0.0)
        scene.force_field.add("wind", shape="plane", strength=3.0, noise=noise,
                              noise_scale=(0.3, 0.3, 0.3), seed=6)
        dt = 1e-4
        fixed, session = solve(app, scene, 1, {"dt": dt, "fps": 1.0 / dt,
                                               "air-density": 1.0, "air-friction": 0.0})
        p0 = frames_of(session, 0)
        displacement[label] = (session, p0, frames_of(session, 1) - p0,
                               scene.force_field.fields[0])
    (s_plain, p0, d_plain, f_plain), (s_gusts, _, d_gusts, f_gusts) = (
        displacement["plain"], displacement["gusts"])
    want = (reference(f_gusts, p0, np.zeros_like(p0), 0.0)[1][:, 1]
            / reference(f_plain, p0, np.zeros_like(p0), 0.0)[1][:, 1])
    got = d_gusts[:, 1] / d_plain[:, 1]
    err = float(np.abs(got - want).max())
    spread = float(np.ptp(want))
    check("kernel_wind_gusts_match_the_reference_through_the_drag",
          s_plain.finished() and s_gusts.finished() and err < 1e-2 and spread > 0.1,
          {"max_ratio_error": err, "ratio_spread": spread})

    # A FIELD FAR FROM THE SCENE. A Point Force 200 m away, whose origin the
    # session carries in double precision; the push is still the
    # reference's, direction and all.
    begin("far_origin")
    session, err, scale = one_step_acceleration(
        "far_origin", (0.2, 0.9, -0.3),
        lambda ff: ff.add("force", position=(200.0, 30.0, -50.0), strength=-5.0))
    check("a_far_field_matches_the_reference",
          session.finished() and err < 1e-3 and scale > 1.0,
          {"relative_error": err, "max_accel": scale})

    # World Scaling: a field is authored in scene units, so the same scene
    # simulated at twice its size meets the same falloff at the same scene
    # positions.
    begin("world_scaling")
    session, err, scale = one_step_acceleration(
        "world_scaling", (0.2, 0.9, -0.3),
        lambda ff: ff.add("force", position=(0.3, 0.6, -0.2), strength=3.0, power=2.0,
                          max_distance=1.2),
        {"world-scaling": 2.0})
    check("a_field_is_authored_in_scene_units_under_world_scaling",
          session.finished() and err < 2e-2 and scale > 0.05,
          {"relative_error": err, "max_accel": scale})

    # Time: a Plane Force facing up whose origin climbs from below the sheet
    # (y = 1) to above it between 0.2 s and 0.3 s, passing it near 0.23 s,
    # pushes the sheet up and then down: by 1 s it is well below where it
    # stood at 0.2 s.
    begin("time")
    app = new_app("time")
    scene = app.scene.create()
    scene.add("sheet").at(0.0, 1.0, 0.0)
    scene.force_field.add("force", shape="plane", strength=10.0,
                          position=[(0, 0, 0), (0, 0, 0), (0, 3, 0)], start=0.1, step=0.1)
    fixed, session = solve(app, scene, 60, {"fps": 60})
    y = [float(frames_of(session, n)[:, 1].mean()) for n in (0, 12, 60)]
    check("a_moving_origin_reverses_the_motion",
          session.finished() and y[1] > y[0] + 0.05 and y[2] < y[1] - 0.5,
          {"y0": y[0], "y12": y[1], "y60": y[2]})

    # Air velocity: a vertical sheet facing +z in a Wind blowing +z moves
    # downstream when the air has density, and not at all when it has none.
    for density in (1.0, 0.0):
        begin("air_%g" % density)
        app = new_app("air_%g" % density)
        scene = app.scene.create()
        scene.add("wall").at(0.0, 1.0, 0.0)
        scene.force_field.add("wind", shape="plane", strength=5.0, axes=np.eye(3))
        fixed, session = solve(app, scene, 10, {"air-density": density})
        dz = float((frames_of(session, 10) - frames_of(session, 0))[:, 2].mean())
        if density > 0:
            check("wind_drives_the_drag", session.finished() and dz > 1e-3, {"dz": dz})
        else:
            check("wind_needs_density", session.finished() and abs(dz) < 1e-7, {"dz": dz})

    # A falloff that does not reach is a still sheet: a Max Distance of 1 m
    # around a field 3 m away.
    begin("unreached")
    app = new_app("unreached")
    scene = app.scene.create()
    scene.add("sheet").at(0.0, 1.0, 0.0)
    scene.force_field.add("force", position=(3.0, 1.0, 0.0), strength=50.0,
                          max_distance=1.0)
    fixed, session = solve(app, scene, 10)
    still = float(np.abs(frames_of(session, 10) - frames_of(session, 0)).max())
    check("a_field_out_of_reach_leaves_the_sheet_still",
          session.finished() and still < 1e-7, {"max_disp": still})

# ---------------------------------------------------------------------------
elif SECTION == "laws":
    # The laws `spatial` does not reach, each against the reference one step
    # from rest (or from a known velocity) on a soft sheet.
    from frontend._field_eval_ import evaluate as reference
    laws = {
        "point_vortex": lambda ff: ff.add(
            "vortex", position=(0.1, 0.4, -0.2), strength=2.0, power=1.0),
        "tube": lambda ff: ff.add(
            "force", position=(0.0, 0.5, 0.0), falloff="tube", power=1.0,
            radial_power=2.0, radial_max=1.5, strength=3.0),
        "z_direction_negative": lambda ff: ff.add(
            "force", position=(0.2, 1.3, -0.3), power=1.0, strength=3.0,
            z_direction="negative"),
        "turbulence_global": lambda ff: ff.add(
            "turbulence", position=(0.4, 0.2, 0.1), strength=4.0, size=0.5, seed=8,
            global_coordinates=True, axes=ALONG_X),
    }
    for name, add_fields in laws.items():
        begin(name)
        session, err, scale = one_step_acceleration(name, (0.2, 0.9, -0.3), add_fields)
        check("kernel_%s_matches_the_reference" % name,
              session.finished() and err < 2e-2 and scale > 0.05,
              {"relative_error": err, "max_accel": scale})

    # THE IMAGE TEXTURE, its sampling and its three modes. An 8 x 8 image of
    # seeded colors; the sheet lies in XZ at y = 0.93 and is placed off the
    # pixel grid (a texel is 0.25 wide), so no vertex sits on a boundary
    # where single and double precision could pick different texels, and the
    # nabla is off the grid too.
    texture_image = np.random.default_rng(13).uniform(0.0, 1.0, (8, 8, 3))
    textures = {
        "texture_rgb_nearest": dict(texture_mode="rgb", interpolation=False),
        "texture_rgb_bilinear": dict(texture_mode="rgb", interpolation=True),
        "texture_gradient": dict(texture_mode="gradient", interpolation=True, nabla=0.11),
        "texture_curl": dict(texture_mode="curl", interpolation=True, nabla=0.11),
        "texture_object_coordinates": dict(texture_mode="rgb", interpolation=True,
                                           object_coordinates=True, axes=ALONG_X,
                                           position=(0.3, 0.2, 0.1)),
        # In Blender's axes, as the add-on sends every texture: the world
        # lookup at (x, -z, y) of the solver point, and the image's vector
        # turned from Blender's axes to the solver's; in object coordinates
        # only the turn.
        "texture_z_up_curl": dict(texture_mode="curl", interpolation=True, nabla=0.11,
                                  z_up=True),
        "texture_z_up_object_coordinates": dict(texture_mode="rgb", interpolation=True,
                                                object_coordinates=True, axes=ALONG_X,
                                                position=(0.3, 0.2, 0.1), z_up=True),
    }
    for name, settings in textures.items():
        begin(name)
        session, err, scale = one_step_acceleration(
            name, (0.21, 0.93, -0.3),
            lambda ff, settings=settings: ff.add("texture", image=texture_image,
                                                 strength=3.0, **settings))
        check("kernel_%s_matches_the_reference" % name,
              session.finished() and err < 2e-2 and scale > 0.05,
              {"relative_error": err, "max_accel": scale})

    # MAGNETIC acts on the VELOCITY, so it is held from a known one: every
    # vertex starts at the object's initial velocity, and one step later the
    # displacement less that velocity's share is the field's acceleration.
    # Were the field reading a velocity other than the solver's, it would
    # miss by the whole field.
    for shape in ("point", "plane"):
        name = "magnetic_" + shape
        begin(name)
        app = new_app(name)
        scene = app.scene.create()
        v0 = np.array([1.5, 0.5, -0.8])
        obj = scene.add("sheet").at(0.2, 0.9, -0.3).velocity(*v0)
        obj.param.set("young-mod", 1e-3).set("bend", 0.0)
        scene.force_field.add("magnetic", shape=shape, position=(0.0, 0.5, 0.1),
                              strength=4.0, axes=ALONG_X)
        dt = 1e-2
        fixed, session = solve(app, scene, 1, {"dt": dt, "fps": 100.0, "air-density": 0.0})
        p0 = frames_of(session, 0)
        got = (frames_of(session, 1) - p0 - dt * v0) / (dt * dt)
        field = scene.force_field.fields[0]
        want = reference(field, p0, np.tile(v0, (len(p0), 1)), 0.0)[0]
        scale = float(np.abs(want).max())
        err = float(np.abs(got - want).max() / scale)
        check("kernel_%s_matches_the_reference" % name,
              session.finished() and err < 2e-2 and scale > 0.5,
              {"relative_error": err, "max_accel": scale})

    # THE TRACK'S BLEND, origin AND rotation, at t = 0: a track starting
    # 0.05 s before the run, its samples 0.1 s apart, puts t = 0 halfway
    # between two samples whose origins and axes both differ.
    begin("track_blend")
    turned = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    session, err, scale = one_step_acceleration(
        "track_blend", (0.2, 0.9, -0.3),
        lambda ff: ff.add("vortex", shape="plane", strength=2.0, power=1.0,
                          position=[(0.0, 0.5, 0.0), (0.4, 0.7, -0.2)],
                          axes=[ALONG_X, turned], start=-0.05, step=0.1))
    check("kernel_track_blend_matches_the_reference",
          session.finished() and err < 2e-2 and scale > 0.05,
          {"relative_error": err, "max_accel": scale})

    # A WIND'S GUSTS DRIFT WITH THE FLOW, in the kernel: the air has no
    # density until 0.05 s, so the sheet rests until then, and its first
    # moving step starts from rest at t > 0, where the pattern has been
    # carried downstream by strength * t. The ratio test of `spatial`
    # applies; a pattern that did not drift would match the reference at
    # t = 0 instead.
    begin("wind_drift")
    dt = 1e-4
    runs = {}
    for label, noise in (("plain", 0.0), ("gusts", 0.6)):
        app = new_app("wind_drift_" + label)
        scene = app.scene.create()
        obj = scene.add("sheet").at(0.2, 0.9, -0.3)
        obj.param.set("young-mod", 1e-3).set("bend", 0.0)
        scene.force_field.add("wind", shape="plane", strength=3.0, noise=noise,
                              noise_scale=(0.3, 0.3, 0.3), seed=6)
        fixed = scene.build(quiet=True)
        session = app.session.create(fixed)
        frames = 520
        session.param.set("frames", frames).set("gravity", [0.0, 0.0, 0.0])
        for k, v in {"dt": dt, "fps": 1.0 / dt, "air-density": 0.0,
                     "air-friction": 0.0}.items():
            session.param.set(k, v)
        session.param.dyn("air-density").time(0.05).hold().time(0.05 + dt).change(1.0)
        session = session.build()
        try:
            session.start(blocking=True)
        except Exception as e:
            print("PPFNOTE start raised: %r" % (e,), flush=True)
        runs[label] = (session, scene.force_field.fields[0])
    (s_plain, f_plain), (s_gusts, f_gusts) = runs["plain"], runs["gusts"]
    ok_runs = s_plain.finished() and s_gusts.finished()
    details = {}
    if ok_runs:
        y = [frames_of(s_plain, n)[:, 1] for n in range(frames + 1)]
        first = next((n for n in range(1, frames + 1)
                      if float(np.abs(y[n] - y[0]).max()) > 1e-9), None)
        details["first_moving_frame"] = first
        if first is not None:
            p_start = frames_of(s_plain, first - 1)
            d_plain = frames_of(s_plain, first) - p_start
            d_gusts = frames_of(s_gusts, first) - frames_of(s_gusts, first - 1)
            t_start = (first - 1) * dt
            zero = np.zeros_like(p_start)
            want = (reference(f_gusts, p_start, zero, t_start)[1][:, 1]
                    / reference(f_plain, p_start, zero, t_start)[1][:, 1])
            frozen = (reference(f_gusts, p_start, zero, 0.0)[1][:, 1]
                      / reference(f_plain, p_start, zero, 0.0)[1][:, 1])
            got = d_gusts[:, 1] / d_plain[:, 1]
            details.update({"t_start": t_start,
                            "error": float(np.abs(got - want).max()),
                            "error_if_frozen": float(np.abs(got - frozen).max())})
    check("kernel_wind_gusts_drift_with_the_flow",
          ok_runs and details.get("error", 1.0) < 1e-2
          and details.get("error_if_frozen", 0.0) > 0.1 and details.get("t_start", 0.0) > 0.04,
          details)

    # THE IMPLICIT TERMS, solved at the Newton iterate, against
    # `step_reference` solving the same backward Euler step per vertex from a
    # known velocity. The sheet is soft enough that its vertices do not
    # couple. The stiff Harmonic (omega dt = 1) is one an explicit evaluation
    # could not hold.
    implicit = {
        "drag": dict(kind="drag", position=(0.2, 0.5, -0.3), power=1.0,
                     linear_drag=0.8, quadratic_drag=0.5),
        "harmonic": dict(kind="harmonic", position=(0.1, 0.5, -0.2), strength=30.0,
                         rest_length=0.2, harmonic_damping=0.7),
        "harmonic_stiff": dict(kind="harmonic", position=(0.1, 0.5, -0.2),
                               strength=1.0e4, harmonic_damping=0.3),
        "harmonic_plane": dict(kind="harmonic", shape="plane", position=(0.0, 0.7, 0.0),
                               strength=20.0, power=1.0, harmonic_damping=0.5),
        "plane_vortex_damping": dict(kind="vortex", shape="plane", strength=0.0,
                                     inflow=1.0, power=1.0, position=(0.1, 0.0, 0.0)),
        "point_vortex_inflow": dict(kind="vortex", strength=0.0, inflow=2.0, power=1.0,
                                    position=(0.0, 0.4, 0.0)),
    }
    v0 = np.array([0.6, 1.2, -0.4])
    for name, settings in implicit.items():
        begin("implicit_" + name)
        settings = dict(settings)
        kind = settings.pop("kind")
        app = new_app("implicit_" + name)
        scene = app.scene.create()
        obj = scene.add("sheet").at(0.2, 0.9, -0.3).velocity(*v0)
        obj.param.set("young-mod", 1e-3).set("bend", 0.0)
        scene.force_field.add(kind, **settings)
        dt = 1e-2
        fixed, session = solve(app, scene, 1, {"dt": dt, "fps": 100.0, "air-density": 0.0})
        details = {"finished": bool(session.finished())}
        ok = session.finished()
        if ok:
            x = frames_of(session, 0)
            got = frames_of(session, 1)
            want = step_reference(scene.force_field.fields, x, np.tile(v0, (len(x), 1)), dt)
            pushed = float(np.abs(want - x - v0 * dt).max())
            err = float(np.abs(got - want).max()) / pushed
            details.update({"relative_error": err, "field_displacement": pushed})
            ok = err < 2e-2 and pushed > 1e-5
        check("kernel_implicit_%s_matches_the_reference" % name, ok, details)

    # GRAVITATION AT ITS ORIGIN is unbounded, as measured, and evaluated
    # exactly: a vertex exactly on the origin fails the run naming the field
    # kernel, rather than being pushed by a clamp. The triangle's vertex at
    # (0, 1, 0) is held exactly, as a vertex and as the origin alike.
    begin("gravitation_origin")
    app = App.create("rig_force_field_gravitation_origin")
    V = np.array([[0.0, 1.0, 0.0], [0.5, 1.0, 0.0], [0.0, 1.0, 0.5]])
    F = np.array([[0, 1, 2]], dtype=np.uint32)
    app.asset.add.tri("tri", V, F)
    scene = app.scene.create()
    scene.add("tri")
    scene.force_field.add("force", position=(0.0, 1.0, 0.0), gravitation=True)
    fixed, session = solve(app, scene, 2)
    log = log_text(session)
    # The CPU backend names the kernel's source, a GPU one its launch label.
    named = "external_field" in log or "field.external" in log
    check("gravitation_at_its_origin_fails_the_run",
          not session.finished() and named,
          {"finished": bool(session.finished()),
           "log_names_the_kernel": named,
           "log_tail": log[-600:]})

# ---------------------------------------------------------------------------
elif SECTION == "script":
    # An exact vortex about +y turns a sheet counterclockwise seen from +y.
    begin("vortex")
    app = new_app("vortex")
    scene = app.scene.create()
    scene.add("sheet").at(0.0, 1.0, 0.0)
    scene.force_field.script(
        "import math\n"
        "def eval(x, y, z, t):\n"
        "    r = math.hypot(x, z)\n"
        "    if r < 1e-6:\n"
        "        return (0.0, 0.0, 0.0)\n"
        "    return (-z / r * 5.0, 0.0, x / r * 5.0)\n")
    fixed, session = solve(app, scene, 10)
    p0 = frames_of(session, 0)
    p1 = frames_of(session, 10)
    a0 = np.arctan2(p0[:, 2], p0[:, 0])
    a1 = np.arctan2(p1[:, 2], p1[:, 0])
    turn = np.angle(np.exp(1j * (a1 - a0)))
    r0 = np.hypot(p0[:, 0], p0[:, 2])
    mean_turn = float(turn[r0 > 0.05].mean())
    check("script_vortex_turns_the_sheet", session.finished() and mean_turn > 0.01,
          {"mean_turn_rad": mean_turn})

    # The compiler refuses by line, before any solve.
    refused = {}
    for label, src, line in (
        ("import", "def eval(x, y, z, t):\n    import os\n    return (0, 0, 0)\n", 2),
        ("while", "def eval(x, y, z, t):\n    while x:\n        x = 0\n    return (0, 0, 0)\n", 2),
        ("unbound", "def eval(x, y, z, t):\n    if x > 0:\n        q = 1.0\n    return (q, 0, 0)\n", 4),
        ("no_return", "def eval(x, y, z, t):\n    if x > 0:\n        return (1, 0, 0)\n", 1),
        ("attribute", "def eval(x, y, z, t):\n    return (os.sep, 0, 0)\n", 2),
    ):
        app = new_app("refuse_" + label)
        scene = app.scene.create()
        try:
            scene.force_field.script(src)
            refused[label] = None
        except ForceFieldScriptError as e:
            refused[label] = e.lineno
    check("compiler_refuses_by_line",
          refused == {"import": 2, "while": 2, "unbound": 4, "no_return": 1, "attribute": 2},
          refused)

    # The solver's loader re-proves the bytecode: a program whose jump points
    # backward is refused at load and the run never steps.
    begin("loader")
    app = new_app("loader")
    scene = app.scene.create()
    scene.add("sheet").at(0.0, 1.0, 0.0)
    scene.force_field.script("def eval(x, y, z, t):\n    return (0.0, -9.8, 0.0)\n")
    fixed = scene.build(quiet=True)
    session = app.session.create(fixed)
    session.param.set("frames", 5)
    session = session.build()
    code_path = os.path.join(session.info.path, "bin", "force_field", "script-0-code.bin")
    np.array([20 | (0 << 8)], dtype="<u4").tofile(code_path)
    try:
        session.start(blocking=True)
    except Exception as e:
        print("PPFNOTE start raised: %r" % (e,), flush=True)
    log = log_text(session)
    check("loader_refuses_a_backward_jump",
          not session.finished() and "only forward jumps" in log,
          {"finished": bool(session.finished()), "logged": "only forward jumps" in log})

    # A script that yields NaN stops the run loudly rather than moving the
    # cloth by garbage.
    begin("nan")
    app = new_app("nan")
    scene = app.scene.create()
    scene.add("sheet").at(0.0, 1.0, 0.0)
    scene.force_field.script(
        "import math\ndef eval(x, y, z, t):\n    return (0.0, math.sqrt(-1.0 - x * x), 0.0)\n")
    fixed, session = solve(app, scene, 5)
    check("non_finite_field_fails_the_run", not session.finished(),
          {"finished": bool(session.finished())})

# ---------------------------------------------------------------------------
elif SECTION == "targets":
    # Sources reach the groups they name: a field object for group "a" pushes
    # down, a script for group "b" pushes up, a script for everyone pushes +x,
    # and an ungrouped sheet gets only the last. Two scripts on one sheet add
    # up.
    begin("targets")

    def run_targets(label, with_b_script):
        app = new_app(label)
        scene = app.scene.create()
        scene.add("sheet").at(-1.4, 1.0, 0.0).group("a")
        scene.add("sheet").at(0.0, 1.0, 0.0).group("b")
        scene.add("sheet").at(1.4, 1.0, 0.0)
        # A plane above the sheets pushes them down, away from it.
        scene.force_field.add("force", shape="plane", strength=8.0,
                              position=(0.0, 10.0, 0.0), groups=["a"])
        if with_b_script:
            scene.force_field.script("def eval(x, y, z, t):\n    return (0.0, 8.0, 0.0)\n",
                                     groups=["b"])
        scene.force_field.script("def eval(x, y, z, t):\n    return (4.0, 0.0, 0.0)\n")
        fixed, session = solve(app, scene, 10)
        x0 = frames_of(session, 0)[:, 0]
        d = frames_of(session, 10) - frames_of(session, 0)
        return session, [d[x0 < -0.7].mean(axis=0), d[np.abs(x0) < 0.7].mean(axis=0),
                         d[x0 > 0.7].mean(axis=0)]

    session, (a, b, c) = run_targets("targets", True)
    # "Add up" is judged against the SAME sheets in a run without group b's
    # script, not against the other sheets: each sheet sits at its own place,
    # and a solver that is not exactly translation invariant moves sheets at
    # different places by slightly different amounts under one push.
    control, (a0, b0, c0) = run_targets("targets_control", False)
    ok = (session.finished() and control.finished()
          and a[1] < -0.05 and b[1] > 0.05 and abs(c[1]) < 1e-6
          and min(a[0], b[0], c[0]) > 0.02
          and np.allclose([a[0], b[0], c[0]], [a0[0], b0[0], c0[0]], rtol=1e-3))
    check("sources_reach_only_their_groups", ok,
          {"a": a.tolist(), "b": b.tolist(), "ungrouped": c.tolist(),
           "x_without_b_script": [a0[0], b0[0], c0[0]]})
    try:
        bad = new_app("targets_unknown").scene.create()
        bad.add("sheet")
        bad.force_field.script("def eval(x, y, z, t):\n    return (0.0, 0.0, 0.0)\n",
                               groups=["nobody"])
        bad.build(quiet=True)
        refused = None
    except ValueError as e:
        refused = str(e)
    check("an_unknown_group_is_refused_at_build",
          refused is not None and "nobody" in refused, {"error": refused})

# ---------------------------------------------------------------------------
elif SECTION == "noise":
    # The kernel's noise is the Python reference's algorithm in single
    # precision. One step of dt from rest on a sheet with almost no stiffness
    # moves each vertex by dt^2 times its acceleration, so the displacement
    # read back is the kernel's curl_noise at each vertex.
    from frontend import curl_noise
    begin("noise")
    app = new_app("noise")
    scene = app.scene.create()
    obj = scene.add("sheet").at(0.2, 0.9, -0.3)
    obj.param.set("young-mod", 1e-3).set("bend", 0.0)
    scene.force_field.script(
        "def eval(x, y, z, t):\n"
        "    return curl_noise(2.0 * x, 2.0 * y, 2.0 * z, octaves=3, seed=11)\n")
    dt = 1e-2
    fixed, session = solve(app, scene, 1, {"dt": dt, "fps": 100.0, "air-density": 0.0})
    p0 = frames_of(session, 0)
    got = (frames_of(session, 1) - p0) / (dt * dt)
    want = np.stack(curl_noise(2.0 * p0[:, 0], 2.0 * p0[:, 1], 2.0 * p0[:, 2], 3, 11), axis=1)
    err = float(np.abs(got - want).max() / np.abs(want).max())
    check("kernel_curl_noise_matches_the_reference",
          session.finished() and err < 2e-2 and float(np.abs(want).max()) > 0.1,
          {"relative_error": err, "max_accel": float(np.abs(want).max())})

    # The same, evolved and faded: a fixed time far from 0 exercises the
    # kernel's fourth coordinate and the compiler's decay instructions in one
    # step (the step itself starts at t = 0), against the reference at that
    # time. A pattern that did not evolve would miss by the full magnitude.
    app = new_app("noise-evolved")
    scene = app.scene.create()
    obj = scene.add("sheet").at(0.2, 0.9, -0.3)
    obj.param.set("young-mod", 1e-3).set("bend", 0.0)
    scene.force_field.script(
        "def eval(x, y, z, t):\n"
        "    return curl_noise(2.0 * x, 2.0 * y, 2.0 * z, octaves=2, seed=5,\n"
        "                      time=1.7 + t, frequency=0.8, decay=0.4)\n")
    fixed, session = solve(app, scene, 1, {"dt": dt, "fps": 100.0, "air-density": 0.0})
    p0 = frames_of(session, 0)
    got = (frames_of(session, 1) - p0) / (dt * dt)
    args = (2.0 * p0[:, 0], 2.0 * p0[:, 1], 2.0 * p0[:, 2], 2, 5)
    want = np.stack(curl_noise(*args, time=1.7, frequency=0.8, decay=0.4), axis=1)
    frozen = np.stack(curl_noise(*args), axis=1) * np.exp(-0.4 * 1.7)
    scale = float(np.abs(want).max())
    err = float(np.abs(got - want).max() / scale)
    moved = float(np.abs(want - frozen).max() / scale)
    check("kernel_evolved_decayed_noise_matches_the_reference",
          session.finished() and err < 2e-2 and moved > 0.2 and scale > 0.05,
          {"relative_error": err, "evolved_vs_frozen": moved, "max_accel": scale})

# ---------------------------------------------------------------------------
elif SECTION == "frame_step":
    begin("frame_step")
    app = new_app("frame_step")
    scene = app.scene.create()
    scene.add("sheet").at(0.0, 1.0, 0.0)
    scene.force_field.script("def eval(x, y, z, t):\n    return (0.0, -9.8, 0.0)\n")
    fixed = scene.build(quiet=True)
    session = app.session.create(fixed)
    session.param.set("frames", 16).set("gravity", [0.0, 0.0, 0.0])
    session = session.build()
    session.run_until_frame(3)
    held = session.held_frame()
    alive = session.is_running()
    latest = session.get.latest_frame()
    time.sleep(0.5)
    check("run_until_frame_holds_a_live_solver",
          held == 3 and alive and latest == 3 and session.get.latest_frame() == 3
          and session.is_running(),
          {"held": held, "alive": alive, "latest": latest})

    session.step_frame()
    check("step_frame_advances_one", session.held_frame() == 4,
          {"held": session.held_frame()})

    y4 = float(frames_of(session, 4)[:, 1].mean())
    scene.force_field.clear().script("def eval(x, y, z, t):\n    return (0.0, 30.0, 0.0)\n")
    session.update_force_field(scene.force_field)
    session.step_frame(4)
    y8 = float(frames_of(session, 8)[:, 1].mean())
    check("updated_field_takes_effect_while_held",
          session.held_frame() == 8 and y8 > y4 + 0.01, {"y4": y4, "y8": y8})

    # Save while held, then resume: the checkpoint carries the run and the
    # session carries the updated field, so the sheet keeps climbing.
    session.save_and_quit()
    deadline = time.time() + 120
    while session.is_running() and time.time() < deadline:
        time.sleep(0.1)
    saved = session.get.saved()
    check("save_and_quit_while_held_checkpoints",
          not session.is_running() and bool(saved) and max(saved) == 8,
          {"saved": list(saved)})
    session.release(blocking=False)
    session.resume(blocking=True)
    y16 = float(frames_of(session, 16)[:, 1].mean())
    check("resume_keeps_the_updated_field",
          session.finished() and y16 > y8 + 0.05, {"y8": y8, "y16": y16})

elif SECTION in ("fluid", "fluid_shortfall"):
    # A FLUID FIELD against the reference, whose law the
    # frontend tests hold to Blender's measurements. Every check runs a real
    # solve on a soft sheet (vertices that do not couple), from a known
    # velocity, against `step_reference` solving the same backward Euler
    # step; a case of several steps holds each one to the reference from the
    # solver's own state at its start. The measured margin of every
    # tolerance is in `rig_force_field_fluid.py`'s header.
    #
    # FRAME F IS THE SOLVER'S STATE AFTER F STEPS, to a part in 1e4 per
    # frame: the solver keeps a step at 0.9999 of the frame period, so frame
    # F is interpolated to its time from inside the (F + 1)-th step, F * 1e-4
    # of the way in. A case of several steps therefore reads each step from
    # two frames, and a held run is held one step past the frame it names.
    #
    # NO VERTEX CROSSES A DOMAIN FACE IN A CASE OF SEVERAL STEPS. Presence
    # jumps there, so the backward Euler step of a vertex crossing a face has
    # no unique answer to compare (measured: a vertex 0.2 cells inside a
    # face, pushed out by the fluid, lands just inside it in the solver and
    # just outside in the reference, 70% of the push apart). Those cases keep
    # every vertex at least a quarter cell from every face of the frames each
    # step reads, and check it.
    from frontend._field_eval_ import (FLUID_BRICK, FLUID_EMPTY, fluid_at, fluid_bracket,
                                       fluid_bricks)
    dt = 1e-2
    v0 = np.array([0.6, 1.2, -0.4])
    center = np.array([0.2, 0.9, -0.3])
    box_corner = (-0.8, -0.1, -1.6)

    def rotation(axis, angle):
        c, s_ = math.cos(angle), math.sin(angle)
        if axis == "x":
            return np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])
        if axis == "y":
            return np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
        return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])

    def rotation_y(angle):
        return rotation("y", angle)

    def smooth_volume(n, phase=0.0):
        # A smooth velocity and a density present over part of the domain.
        i, j, k = np.meshgrid(*(np.arange(m, dtype=np.float64) for m in n), indexing="ij")
        u = np.stack([1.5 * np.sin(0.5 * i + phase) + 0.3 * k,
                      0.8 * np.cos(0.4 * j - 0.3 * k),
                      -1.0 + 0.2 * i], axis=-1)
        rho = np.clip(0.2 * (k - 2.0), 0.0, None)
        return u, rho

    def uniform(value, n=(4, 4, 4)):
        return np.broadcast_to(np.asarray(value, float), tuple(n) + (3,)).copy()

    def sheet_run(name, add_fields, frames=1, velocity=v0, place=(0.2, 0.9, -0.3),
                  params=None):
        app = new_app(name)
        scene = app.scene.create()
        obj = scene.add("sheet").at(*place).velocity(*velocity)
        obj.param.set("young-mod", 1e-3).set("bend", 0.0)
        add_fields(scene.force_field)
        run_params = {"dt": dt, "fps": 100.0, "air-density": 0.0}
        run_params.update(params or {})
        fixed, session = solve(app, scene, frames, run_params)
        return scene, session

    def step_error(fields, session, F, velocity=v0, ws=1.0, fluid_time=None, playback=1.0):
        # Step F -> F + 1 against the reference from the solver's own state:
        # its positions at frame F and its velocity there, the initial one at
        # F = 0 and the difference of the two frames before after that (per
        # clock second, so over `playback` per physical second).
        x = frames_of(session, F)
        if F == 0:
            v = np.tile(np.asarray(velocity, dtype=np.float64), (len(x), 1))
        else:
            v = (x - frames_of(session, F - 1)) / (dt * playback)
        want = step_reference(fields, x, v, dt, time=F * dt, fluid_time=fluid_time,
                              world_scaling=ws, playback=playback)
        got = frames_of(session, F + 1)
        pushed = float(np.abs(want - x - v * dt * playback).max())
        miss = float(np.abs(got - want).max())
        return x, v, got, want, pushed, miss

    def steps_match(fields, session, steps, velocity=v0, coast=()):
        # Each step of `steps` within 2e-2 of the push the reference gives it,
        # and that push above 1e-5; a step in `coast` has nothing acting on
        # it, so it must move by its velocity alone, to 1e-5.
        ok = True
        rows = []
        for F in steps:
            _, _, _, _, pushed, miss = step_error(fields, session, F, velocity)
            if F in coast:
                good = pushed < 1e-9 and miss < 1e-5
            else:
                good = pushed > 1e-5 and miss < 2e-2 * pushed
            ok = ok and good
            rows.append({"step": F, "field_displacement": pushed, "miss": miss,
                         "relative_error": miss / pushed if pushed > 0.0 else None})
        return ok, rows

    def one_step(name, add_fields, place=(0.2, 0.9, -0.3), params=None, velocity=v0):
        begin(name)
        scene, session = sheet_run(name, add_fields, place=place, params=params,
                                   velocity=velocity)
        ws = float((params or {}).get("world-scaling", 1.0))
        details = {"finished": bool(session.finished())}
        ok = session.finished()
        if ok:
            _, _, _, _, pushed, miss = step_error(scene.force_field.fields, session, 0,
                                                  velocity, ws)
            err = miss / pushed
            details.update({"relative_error": err, "field_displacement": pushed})
            ok = err < 2e-2 and pushed > 1e-5
        check("kernel_%s_matches_the_reference" % name, ok, details)
        return scene, session

    def frame_layout(volume, k):
        vol = volume.as_dict()
        return (np.asarray(vol["resolution"][k], dtype=np.int64),
                np.asarray(vol["corner"][k], dtype=np.float64),
                np.asarray(vol["axes"][k], dtype=np.float64),
                np.broadcast_to(np.asarray(vol["cell"][k], dtype=np.float64), (3,)),
                np.asarray(vol["tables"][k]))

    def bricks_under(volume, k, points):
        # Whether each point is inside frame k, the brick its cell lies in
        # and that brick's table entry.
        n, corner, axes, cell, table = frame_layout(volume, k)
        g = ((np.asarray(points) - corner) @ axes) / cell
        inside = np.all((g >= 0.0) & (g <= n), axis=1)
        brick = np.clip(np.floor(g).astype(np.int64), 0, n - 1) // FLUID_BRICK
        nb = (n + FLUID_BRICK - 1) // FLUID_BRICK
        slot = table[brick[:, 0] + nb[0] * (brick[:, 1] + nb[1] * brick[:, 2])]
        return inside, brick, slot

    def capacity(volume):
        # The largest table and the most bricks of any frame: what the driver
        # sizes a slot for.
        vol = volume.as_dict()
        frames = range(len(vol["tables"]))
        return (max(len(np.asarray(vol["tables"][k])) for k in frames),
                max(np.asarray(vol["bricks"][k]).size // (512 * 4) for k in frames))

    def clearance(volume, frames, points):
        # The least distance, in cells, from any point to any face of any of
        # `frames`, negative for a point outside one.
        least = float("inf")
        for k in frames:
            n, corner, axes, cell, _ = frame_layout(volume, k)
            g = ((np.asarray(points) - corner) @ axes) / cell
            least = min(least, float(np.min(np.minimum(g, n - g))))
        return least

    def steps_clear(volume, session, steps):
        # The least clearance over the steps, of each step's start and end
        # against the two frames it blends (`fluid_bracket` at its end).
        least = float("inf")
        for F in steps:
            k0, k1, _ = fluid_bracket(volume.as_dict(), (F + 1) * dt)
            for frame in (F, F + 1):
                least = min(least, clearance(volume, (k0, k1), frames_of(session, frame)))
        return least

    def stop_held(session):
        # A held run must not outlive a case that failed while holding it:
        # the next case's start would find a solver running (`Utils.busy`
        # scans every process) and wait on it.
        try:
            if session.is_running():
                session.save_and_quit()
                deadline = time.time() + 60
                while session.is_running() and time.time() < deadline:
                    time.sleep(0.1)
        except Exception as e:
            print("PPFNOTE stopping a held run raised: %r" % (e,), flush=True)

    # A rotated, offset domain of 12 x 6 x 13 cells (bricks cut at 8), over
    # part of the sheet only, so some vertices read the volume, some its
    # clamped outer cells and some nothing; density weighing, Flow 1 and a
    # falloff from the field object all on.
    n = (12, 6, 13)
    u, rho = smooth_volume(n)
    axes = rotation_y(0.5)
    corner = center - axes @ np.array([0.45, 0.2, 0.3])
    volume = FluidVolume([u], cell=0.06, corner=corner, axes=axes, density=[rho])

if SECTION == "fluid_shortfall":
    # THE OPEN DEFECT OF A NEGATIVE COUPLING, run on demand by
    # `rig_force_field_fluid_shortfall`, whose header
    # records the numbers. A negative Strength or Flow makes the damping
    # coefficient `s W rho flow` negative, so the solver applies its gradient
    # alone, and the Newton loop stops on its TOI
    # target after two linear solves, which without that Hessian leave about
    # 0.54 |s flow dt| of the push unsolved. Each case holds one step to the
    # backward Euler reference at the 2e-2 every other fluid case is held to,
    # and FAILS while the defect is open. A twin of each, with
    # `min-newton-steps` 6, is the control that the gradient is right and
    # the miss is the loop's early stop: it must land within 2e-2.
    for label, strength, flow in (("strength_minus_8", -8.0, 1.0),
                                  ("flow_minus_1_at_strength_8", 8.0, -1.0)):
        for steps, suffix in ((0, "lands_on_backward_euler"),
                              (6, "converged_control_lands_on_backward_euler")):
            name = "fluid_shortfall_%s_%s" % (label, "control" if steps else "open")
            begin(name)
            scene, session = sheet_run(name, lambda ff, s_=strength, f_=flow: ff.add(
                "fluid", fluid=volume, strength=s_, flow=f_),
                params={"min-newton-steps": steps})
            details = {"finished": bool(session.finished()), "strength": strength,
                       "flow": flow, "strength_flow_dt": strength * flow * dt,
                       "min_newton_steps": steps}
            ok = session.finished()
            if ok:
                _, _, _, _, pushed, miss = step_error(scene.force_field.fields, session, 0)
                details.update({"relative_error": miss / pushed, "field_displacement": pushed})
                ok = miss < 2e-2 * pushed and pushed > 1e-5
            check("fluid_%s_%s" % (label, suffix), ok, details)

if SECTION == "fluid":
    one_step("fluid_relative_coupling", lambda ff: ff.add(
        "fluid", fluid=volume, strength=40.0, flow=1.0, use_density=True,
        power=1.0, position=(0.0, 0.5, 0.0)))
    one_step("fluid_flow_zero_pushes", lambda ff: ff.add(
        "fluid", fluid=volume, strength=3.0, flow=0.0, position=(0.0, 0.5, 0.0)))

    # THE CELL IS PER AXIS. The same domain with a different cell along each
    # of its axes: the session's transform carries the three in the domain's
    # x, y, z order, and any other order moves every vertex to another cell.
    aniso = FluidVolume([u], cell=(0.05, 0.075, 0.06), corner=corner, axes=axes,
                        density=[rho])
    one_step("fluid_per_axis_cell", lambda ff: ff.add(
        "fluid", fluid=aniso, strength=40.0, flow=1.0, use_density=True,
        power=1.0, position=(0.0, 0.5, 0.0)))

    # NEGATIVE STRENGTH OR FLOW. The damping coefficient `s W rho flow` is
    # then negative, so the solver applies its gradient alone (a Hessian
    # there would be indefinite) and the step must still land on the
    # backward Euler solution; a gradient of the wrong sign or size misses
    # it by the damping's whole share of the push. |s flow dt| is kept at
    # 0.01: the solver ends a step after two linear solves from the step's
    # start, which without that Hessian leave about 0.54 |s flow dt| of the
    # push unsolved (measured 4.3% at Strength -8, 3.0% at Flow -1 with
    # Strength 8), reported as a defect rather than allowed for here.
    # The cases at the magnitudes users reach, which miss by 3% to 4%, are
    # the on-demand known failure `rig_force_field_fluid_shortfall`.
    one_step("fluid_negative_strength", lambda ff: ff.add(
        "fluid", fluid=volume, strength=-1.0, flow=1.0))
    one_step("fluid_negative_flow", lambda ff: ff.add(
        "fluid", fluid=volume, strength=1.0, flow=-1.0))

    # PLAYBACK SPEED. At Playback Speed p a step moves the vertices by
    # dt p seconds of motion and the scene clock, where the fluid's frames
    # sit, by dt, so a fluid's U, per clock second, is divided by p before it
    # meets the vertex velocity: the sheet then rides
    # the fluid at U per clock second, as the frames advect. Slowed (0.5)
    # and sped up (2), each against the reference solving that step; one
    # that left U undivided must miss it by over 0.3 of the push, so the
    # case can tell the two apart.
    for label, playback in (("half", 0.5), ("double", 2.0)):
        name = "fluid_playback_" + label
        begin(name)
        scene, session = sheet_run(name, lambda ff: ff.add(
            "fluid", fluid=volume, strength=40.0, flow=1.0, use_density=True,
            power=1.0, position=(0.0, 0.5, 0.0)), params={"playback": playback})
        details = {"finished": bool(session.finished()), "playback": playback}
        ok = session.finished()
        if ok:
            fields = scene.force_field.fields
            x, v, got, want, pushed, miss = step_error(fields, session, 0, playback=playback)
            undivided = step_reference(fields, x, v, dt, playback=playback, fluid_playback=1.0)
            details.update({
                "relative_error": miss / pushed, "field_displacement": pushed,
                "undivided_separation": float(np.abs(undivided - want).max()) / pushed,
                "from_the_undivided_reference": float(np.abs(got - undivided).max()) / pushed})
            ok = (miss < 2e-2 * pushed and pushed > 1e-5
                  and details["undivided_separation"] > 0.3)
        check("kernel_fluid_playback_%s_divides_the_velocity" % label, ok, details)

    # WORLD SCALING. A fluid's velocity is in scene units per second and the
    # solver scales it with the domain, so the scene simulated at twice its
    # size moves, in scene units, as the unscaled reference does; a velocity
    # left unscaled would push it at half the rate. From rest, so no initial
    # velocity's units enter.
    begin("fluid_world_scaling")
    app = new_app("fluid_world_scaling")
    scene = app.scene.create()
    scene.add("sheet").at(0.2, 0.9, -0.3).param.set("young-mod", 1e-3).set("bend", 0.0)
    scene.force_field.add("fluid", fluid=volume, strength=40.0, flow=1.0, use_density=True,
                          power=1.0, position=(0.0, 0.5, 0.0))
    fixed, session = solve(app, scene, 1, {"dt": dt, "fps": 100.0, "air-density": 0.0,
                                           "world-scaling": 2.0})
    details = {"finished": bool(session.finished())}
    ok = session.finished()
    if ok:
        x = frames_of(session, 0)
        got = frames_of(session, 1)
        want = step_reference(scene.force_field.fields, x, np.zeros_like(x), dt,
                              world_scaling=2.0)
        pushed = float(np.abs(want - x).max())
        err = float(np.abs(got - want).max()) / pushed
        details.update({"relative_error": err, "field_displacement": pushed})
        ok = err < 2e-2 and pushed > 1e-5
    check("fluid_velocity_is_scaled_with_the_scene", ok, details)

    # World Scaling 0.5 with the implicit fluid and a moving sheet: the
    # solver holds the domain's corner at corner * ws, the velocity at U * ws
    # and the sheet's velocity at v * ws, and the reference solves the same
    # step in those units. A control below the domain's floor, 0.02 out and
    # moving away, must coast: had the corner not been scaled, the domain
    # would sit at corner / ws in scene units, over most of that sheet.
    one_step("fluid_world_scaling_half", lambda ff: ff.add(
        "fluid", fluid=volume, strength=40.0, flow=1.0, use_density=True,
        power=1.0, position=(0.0, 0.5, 0.0)), params={"world-scaling": 0.5})
    box = FluidVolume([uniform((3.0, 0.0, 0.0)), uniform((-3.0, 2.0, 0.0))],
                      cell=0.5, corner=box_corner, step=0.02)
    begin("fluid_outside_the_scaled_domain")
    below = (0.6, -1.2, -0.4)
    scene, session = sheet_run("fluid_outside_scaled", lambda ff: ff.add(
        "fluid", fluid=box, strength=20.0, flow=1.0), velocity=below,
        place=(-0.2, -0.12, -1.35), params={"world-scaling": 0.5})
    ok = session.finished()
    drift = float("inf")
    if ok:
        drift = float(np.abs(frames_of(session, 1) - frames_of(session, 0)
                             - dt * np.asarray(below)).max())
        ok = drift < 1e-6
    check("fluid_outside_the_scaled_domain_is_still", ok, {"drift": drift})

    # A SPARSE VOLUME. 20 x 13 x 30 cells of 0.045 x 0.06 x 0.035 make
    # 3 x 2 x 4 bricks, a different count along each axis, so a stride taken
    # along the wrong axis reads another brick. The velocity is nonzero in
    # four bricks only, in both rows of y bricks, so the table renumbers them
    # (brick 7 is slot 0) and the other 20 entries are FLUID_EMPTY. The sheet
    # lies across kept bricks of both rows and the empty bricks beside them:
    # inside the domain an empty brick reads a zero velocity at full
    # presence, so Flow damps a vertex there and nothing pushes it.
    sparse_n = (20, 13, 30)
    sparse_cell = np.array([0.045, 0.06, 0.035])
    sparse_axes = rotation("y", 0.35) @ rotation("x", 0.15)
    sparse_corner = center - sparse_axes @ (np.array([8.6, 8.1, 15.3]) * sparse_cell)
    kept = ((1, 0, 1), (0, 1, 2), (1, 1, 2), (2, 1, 3))
    i, j, k = np.meshgrid(*(np.arange(m, dtype=np.float64) for m in sparse_n), indexing="ij")
    u_sparse = np.stack([1.2 + 0.5 * np.sin(0.3 * i + 0.2 * k),
                         0.6 * np.cos(0.25 * j) - 0.3,
                         -0.8 + 0.04 * k + 0.1 * np.sin(0.4 * i)], axis=-1)
    keep = np.zeros(sparse_n, dtype=bool)
    for bx, by, bz in kept:
        keep[8 * bx:8 * bx + 8, 8 * by:8 * by + 8, 8 * bz:8 * bz + 8] = True
    u_sparse[~keep] = 0.0
    sparse = FluidVolume([u_sparse], cell=sparse_cell, corner=sparse_corner, axes=sparse_axes)
    scene, session = one_step("fluid_sparse_bricks", lambda ff: ff.add(
        "fluid", fluid=sparse, strength=20.0, flow=1.0))
    table = np.asarray(sparse.as_dict()["tables"][0])
    full = np.nonzero(table != FLUID_EMPTY)[0]
    details = {"bricks": [(m + FLUID_BRICK - 1) // FLUID_BRICK for m in sparse_n],
               "kept_entries": full.tolist(), "slots": table[full].tolist()}
    ok = session.finished()
    if ok:
        inside, brick, slot = bricks_under(sparse, 0, frames_of(session, 0))
        in_empty = inside & (slot == FLUID_EMPTY)
        in_kept = inside & (slot != FLUID_EMPTY)
        rows = sorted(set(int(b) for b in brick[in_kept, 1]))
        details.update({"in_empty_bricks": int(in_empty.sum()),
                        "in_kept_bricks": int(in_kept.sum()), "kept_y_rows_read": rows})
        ok = (len(set(details["bricks"])) == 3 and len(full) == 4
              and bool(np.any(table[full] != full)) and int(table[7]) == 0
              and in_empty.sum() >= 6 and in_kept.sum() >= 6 and rows == [0, 1])
    check("fluid_sparse_case_reads_empty_and_kept_bricks", ok, details)

    # A PER-FRAME LAYOUT. Four frames that differ in their brick counts
    # (2x3x2, 3x2x2, 2x4x1, 2x2x3), corner, axes, cell along each axis and
    # density, 0.023 s apart, so each of six steps ends strictly inside a
    # frame interval and the run crosses two frame boundaries: (0, 1) are
    # read from slots 0 and 1, (1, 2) from slots 1 and 0, since frame 2
    # takes the slot frame 0 leaves, and (2, 3) from slots 0 and 1 again.
    # Each frame holds the whole sheet, every vertex a quarter cell or more
    # inside each face the step reads.
    layout_frames = [
        ((14, 17, 12), rotation("y", 0.5), (0.08, 0.066, 0.095), (0.03, 0.02, -0.04)),
        ((19, 9, 13), rotation("y", -0.3) @ rotation("x", 0.2), (0.06, 0.125, 0.087),
         (-0.05, -0.03, 0.02)),
        ((13, 27, 7), rotation("z", 0.25) @ rotation("y", 0.9), (0.095, 0.045, 0.18),
         (0.04, 0.03, 0.03)),
        ((16, 12, 18), rotation("x", -0.4) @ rotation("z", -0.2), (0.071, 0.094, 0.063),
         (-0.02, -0.05, -0.06)),
    ]
    velocities, densities, corners, frame_axes, cells = [], [], [], [], []
    for f, (nf, af, cf, offset) in enumerate(layout_frames):
        uf, rf = smooth_volume(nf, phase=0.8 * f)
        velocities.append(uf)
        densities.append((0.6 + 0.3 * f) * rf)
        corners.append(center + np.asarray(offset) - af @ (np.asarray(nf) * np.asarray(cf) / 2.0))
        frame_axes.append(af)
        cells.append(cf)
    layout = FluidVolume(velocities, cell=np.asarray(cells), corner=np.asarray(corners),
                         axes=np.asarray(frame_axes), density=densities, step=0.023)
    begin("fluid_per_frame_layout")
    scene, session = sheet_run("fluid_per_frame_layout", lambda ff: ff.add(
        "fluid", fluid=layout, strength=40.0, flow=1.0, use_density=True), frames=6)
    brackets = [fluid_bracket(layout.as_dict(), (F + 1) * dt) for F in range(6)]
    details = {"finished": bool(session.finished()),
               "brackets": [[k0, k1, round(b, 4)] for k0, k1, b in brackets]}
    ok = (session.finished() and all(0.0 < b < 1.0 for _, _, b in brackets)
          and sorted(set((k0, k1) for k0, k1, _ in brackets)) == [(0, 1), (1, 2), (2, 3)])
    if ok:
        details["least_clearance_cells"] = steps_clear(layout, session, range(6))
        ok, details["steps"] = steps_match(scene.force_field.fields, session, range(6))
        ok = ok and details["least_clearance_cells"] >= 0.25
    check("kernel_fluid_per_frame_layout_matches_the_reference", ok, details)

    # TWO FLUIDS BESIDE A FORCE FIELD. The Force comes first, so the sparse
    # fluid is field 1 and fluid 0, the layout field 2 and fluid 1, whose
    # slots start past the first fluid's two (table entry 48 and float 16384,
    # nonzero and unequal, so a fluid read at another's base, or a table base
    # taken for a data base, reads the wrong values). A twin without the
    # second fluid must match its own reference and miss the full one.
    def two_fluids(ff, second=True):
        ff.add("force", strength=4.0, position=(1.4, 0.9, -0.3))
        ff.add("fluid", fluid=sparse, strength=20.0, flow=1.0)
        if second:
            ff.add("fluid", fluid=layout, strength=40.0, flow=1.0, use_density=True)

    scene, session = one_step("fluid_two_beside_a_force", two_fluids)
    begin("fluid_two_twin")
    twin_scene, twin = sheet_run("fluid_two_twin", lambda ff: two_fluids(ff, False))
    table_a, bricks_a = capacity(sparse)
    details = {"finished": bool(session.finished()), "twin_finished": bool(twin.finished()),
               "second_table_base": 2 * table_a, "second_data_base": 2 * bricks_a * 2048}
    ok = session.finished() and twin.finished()
    if ok:
        _, _, got, want, pushed, _ = step_error(scene.force_field.fields, session, 0)
        _, _, twin_got, twin_want, twin_pushed, twin_miss = step_error(
            twin_scene.force_field.fields, twin, 0)
        x0 = frames_of(session, 0)
        in_a, _, _ = bricks_under(sparse, 0, x0)
        in_b = bricks_under(layout, 0, x0)[0] | bricks_under(layout, 1, x0)[0]
        details.update({
            "twin_relative_error": twin_miss / twin_pushed,
            "second_share": float(np.abs(want - twin_want).max()) / pushed,
            "solver_second_share": float(np.abs(got - twin_got).max()) / pushed,
            "inside_both": int(np.sum(in_a & in_b)),
            "least_clearance_cells": min(
                clearance(sparse, (0,), np.concatenate([x0, got])),
                clearance(layout, (0, 1), np.concatenate([x0, got]))),
        })
        ok = (twin_miss < 2e-2 * twin_pushed and details["second_share"] > 0.2
              and details["inside_both"] >= 6 and 2 * table_a > 0
              and 2 * table_a != 2 * bricks_a * 2048
              and details["least_clearance_cells"] >= 0.25)
    check("fluid_second_fluid_acts", ok, details)

    # THE STEP READS THE FLUID AT ITS END. Two frames 0.02 s apart with
    # opposite velocities: a step from 0 to dt = 0.01 reads their mean. The
    # same implicit step reading the volume at the step's START, frame 0
    # alone, lands far from it, and the solver must land on the first and
    # not the second; both references are full backward Euler solves, so
    # the check measures the read time and nothing else.
    scene, session = one_step("fluid_frames_blend_at_the_step_end", lambda ff: ff.add(
        "fluid", fluid=box, strength=20.0, flow=1.0))
    details = {"finished": bool(session.finished())}
    ok = session.finished()
    if ok:
        x = frames_of(session, 0)
        v = np.tile(v0, (len(x), 1))
        got = frames_of(session, 1)
        end = step_reference(scene.force_field.fields, x, v, dt)
        start = step_reference(scene.force_field.fields, x, v, dt, fluid_time=0.0)
        pushed = float(np.abs(end - x - v * dt).max())
        details.update({
            "separation": float(np.abs(end - start).max()) / pushed,
            "from_the_start_read": float(np.abs(got - start).max()) / pushed,
            "from_the_end_read": float(np.abs(got - end).max()) / pushed,
        })
        ok = (details["separation"] > 0.5 and details["from_the_start_read"] > 0.5
              and details["from_the_start_read"] > 10.0 * details["from_the_end_read"])
    check("fluid_read_at_the_step_start_would_miss", ok, details)

    # A FLUID THAT STARTS LATE. Frames from 0.03 s, 0.02 s apart: the first
    # three steps end before or on the first frame, which holds, and the
    # next two blend into the second and reach it. A volume that ignored its
    # start would blend frames 0 and 1 already at the first step, which the
    # case shows the reference can tell apart.
    late = FluidVolume([uniform((3.0, 0.0, 0.0)), uniform((-3.0, 2.0, 0.0)),
                        uniform((0.0, -1.5, 2.5))], cell=0.5, corner=box_corner,
                       start=0.03, step=0.02)
    begin("fluid_late_start")
    scene, session = sheet_run("fluid_late_start", lambda ff: ff.add(
        "fluid", fluid=late, strength=20.0, flow=1.0), frames=5)
    details = {"finished": bool(session.finished())}
    ok = session.finished()
    if ok:
        ok, details["steps"] = steps_match(scene.force_field.fields, session, range(5))
        x, v, _, want, pushed, _ = step_error(scene.force_field.fields, session, 0)
        early = dict(scene.force_field.fields[0])
        early["fluid"] = dict(early["fluid"], start=0.0)
        details["start_ignored_separation"] = float(np.abs(
            step_reference([early], x, v, dt) - want).max()) / pushed
        ok = ok and details["start_ignored_separation"] > 0.3
    check("kernel_fluid_late_start_matches_the_reference", ok, details)

    # A FLUID FINER THAN THE STEP. Ten frames 0.0045 s apart against
    # dt = 0.01: the brackets of consecutive steps, (2, 3), (4, 5), (6, 7),
    # (8, 9), share no frame, so every step replaces both resident frames.
    quick = FluidVolume([uniform((2.0 * math.cos(0.9 * f), math.sin(0.7 * f), 0.6 * f - 2.4))
                         for f in range(10)], cell=0.5, corner=box_corner, step=0.0045)
    begin("fluid_finer_than_the_step")
    scene, session = sheet_run("fluid_finer_than_the_step", lambda ff: ff.add(
        "fluid", fluid=quick, strength=20.0, flow=1.0), frames=4)
    brackets = [fluid_bracket(quick.as_dict(), (F + 1) * dt)[:2] for F in range(4)]
    details = {"finished": bool(session.finished()), "brackets": brackets}
    ok = session.finished() and all(not set(a) & set(b) for a, b in zip(brackets, brackets[1:]))
    if ok:
        ok, details["steps"] = steps_match(scene.force_field.fields, session, range(4))
    check("kernel_fluid_finer_than_the_step_matches_the_reference", ok, details)

    # AN EMPTY FRAME HOLDS NO FLUID. The box's first frame followed by an
    # empty one: the step's end lies halfway between them, where velocity
    # and presence each blend to half, so the push is a quarter of the full
    # frame's and the damping a half.
    fading = FluidVolume([uniform((3.0, 0.0, 0.0)), np.zeros((0, 0, 0, 3))],
                         cell=0.5, corner=box_corner, step=0.02)
    one_step("fluid_empty_frame_blends", lambda ff: ff.add(
        "fluid", fluid=fading, strength=20.0, flow=1.0))

    # An empty frame in the MIDDLE of a sequence: the second step ends on it,
    # where presence is 0 everywhere, so the sheet coasts on its velocity;
    # the steps either side blend into and out of it.
    gap = FluidVolume([uniform((3.0, 0.0, 0.0)), np.zeros((0, 0, 0, 3)),
                       uniform((-2.0, 1.5, 1.0))], cell=0.5, corner=box_corner, step=0.02)
    begin("fluid_empty_middle_frame")
    scene, session = sheet_run("fluid_empty_middle_frame", lambda ff: ff.add(
        "fluid", fluid=gap, strength=20.0, flow=1.0), frames=4)
    details = {"finished": bool(session.finished())}
    ok = session.finished()
    if ok:
        ok, details["steps"] = steps_match(scene.force_field.fields, session, range(4),
                                           coast=(1,))
    check("kernel_fluid_empty_middle_frame_matches_the_reference", ok, details)

    # A STRONG COUPLING IS STABLE AND THE FRAMES STREAM. Strength 1e4 at
    # Flow 1 makes the sheet's velocity the fluid's at each step's end, which
    # an explicit evaluation could not survive (dt * strength = 100). Six
    # fluid frames over the ten solver steps, each different, so the driver
    # replaces a resident frame at every other step; the sheet's velocity at
    # each frame is then the fluid's there.
    ramp = [uniform((math.cos(0.9 * k), 0.5 * math.sin(0.7 * k), 0.3 * k - 0.8))
            for k in range(6)]
    stream = FluidVolume(ramp, cell=0.5, corner=box_corner, step=0.02)
    begin("fluid_stream")
    scene, session = sheet_run("fluid_stream", lambda ff: ff.add(
        "fluid", fluid=stream, strength=1.0e4, flow=1.0), frames=10)
    details = {"finished": bool(session.finished())}
    ok = session.finished()
    if ok:
        worst = 0.0
        for f in range(1, 11):
            v = (frames_of(session, f) - frames_of(session, f - 1)) / dt
            U = fluid_at(stream.as_dict(), [[0.0, 0.0, 0.0]], f * dt)[0][0]
            # Backward Euler of dv/dt = s (U - v): v = (v_prev + s dt U) / (1 + s dt),
            # within 1 / (1 + s dt) of U.
            worst = max(worst, float(np.abs(v - U).max()))
        details["worst_velocity_gap"] = worst
        ok = worst < 0.05
    check("fluid_strong_coupling_tracks_a_streamed_volume", ok, details)

    # CONTROLS: a sheet outside the domain does not move off its course, and
    # a malformed frame stops the run at load, naming the frame.
    begin("fluid_outside")
    scene, session = sheet_run("fluid_outside", lambda ff: ff.add(
        "fluid", fluid=box, strength=20.0, flow=1.0), place=(3.0, 0.9, -0.3))
    ok = session.finished()
    drift = float("inf")
    if ok:
        drift = float(np.abs(frames_of(session, 1) - frames_of(session, 0) - dt * v0).max())
        ok = drift < 1e-6
    check("fluid_outside_its_domain_is_still", ok, {"drift": drift})

    begin("fluid_malformed")
    app = new_app("fluid_malformed")
    scene = app.scene.create()
    scene.add("sheet").at(0.2, 0.9, -0.3)
    scene.force_field.add("fluid", fluid=box, strength=1.0)
    fixed = scene.build(quiet=True)
    session = app.session.create(fixed)
    session.param.set("frames", 2).set("gravity", [0.0, 0.0, 0.0])
    session = session.build()
    bricks = os.path.join(session.info.path, "bin", "force_field", "field-0-fluid",
                          "bricks-00001.bin")
    values = np.fromfile(bricks, dtype="<f4")
    values[5] = np.nan
    values.tofile(bricks)
    try:
        session.start(blocking=True)
    except Exception as e:
        print("PPFNOTE start raised: %r" % (e,), flush=True)
    log = log_text(session)
    named = "frame 1" in log and "bricks-00001.bin" in log
    check("fluid_malformed_frame_fails_the_run_by_name",
          not session.finished() and named,
          {"finished": bool(session.finished()), "named": named, "log_tail": log[-600:]})

    # A HELD RUN. Twelve uniform frames 0.02 s apart at Strength 1e3, Flow 1,
    # over 16 steps. Held at 3, then stepped 2 and 3 frames, the run must
    # match the same run left unheld frame for frame. Then, still held at 8,
    # the field is replaced by a fluid of 16^3 cells (8 bricks a frame,
    # against the old fluid's 1, so its slots must be resized) pushing the
    # other way, and the run is stepped 2 frames. A run held at frame 8 is a
    # step past it (see the top of this section), so the step frames 8 to 9
    # record was taken before the update and is only recorded here; the
    # first step the update reaches is the one frames 9 to 10 record, which
    # must match the reference of the NEW field and miss the old one's. Saved
    # and resumed, each step to the end must still match the new field, read
    # at the restored time. Last, the session saved at build must still
    # rebuild: the update rewrote the files its fluid was read from, so the
    # update saves the built scene again, which the frontend's tests check
    # with the hold patched in and this run checks with the solver holding,
    # its process live while the update saves. Recovered from disk by name,
    # the session rebuilds with the fluid it was built with, its files equal
    # to the built volume's frames byte for byte.
    held_u = [uniform((1.5 * math.cos(0.8 * f), 0.8 * math.sin(0.6 * f) + 0.3, 0.2 * f - 1.0))
              for f in range(12)]
    held_volume = FluidVolume(held_u, cell=0.5, corner=box_corner, step=0.02)

    def held_field(ff, volume=held_volume):
        ff.add("fluid", fluid=volume, strength=1.0e3, flow=1.0)

    held_checks = ("fluid_held_run_matches_an_unheld_one",
                   "fluid_update_while_held_reinstalls_a_larger_volume",
                   "fluid_resume_reads_the_updated_volume_at_its_time",
                   "fluid_session_saved_at_build_rebuilds_after_the_update")
    session = None
    try:
        begin("fluid_unheld")
        # The unheld run's fluid is a volume of its own with the same frames:
        # a volume keeps every place it was written to and reads a frame from
        # any of them, so sharing one would let the rebuild below read the
        # unheld run's files rather than test the held session's own save.
        _, unheld = sheet_run("fluid_unheld", lambda ff: held_field(ff, FluidVolume(
            held_u, cell=0.5, corner=box_corner, step=0.02)), frames=16)
        begin("fluid_held")
        app = new_app("fluid_held")
        scene = app.scene.create()
        scene.add("sheet").at(0.2, 0.9, -0.3).velocity(*v0).param.set(
            "young-mod", 1e-3).set("bend", 0.0)
        held_field(scene.force_field)
        old_fields = scene.force_field.fields
        fixed = scene.build(quiet=True)
        session = app.session.create(fixed, "rig_fluid_held")
        session.param.set("frames", 16).set("gravity", [0.0, 0.0, 0.0]).set("dt", dt).set(
            "fps", 100.0).set("air-density", 0.0)
        session = session.build()
        session.run_until_frame(3)
        session.step_frame(2)
        session.step_frame(3)
        held = session.held_frame()
        details = {"held": held, "unheld_finished": bool(unheld.finished())}
        ok = held == 8 and unheld.finished()
        if ok:
            a0 = frames_of(unheld, 0)
            scale = max(float(np.abs(frames_of(unheld, f) - a0).max()) for f in range(1, 9))
            gap_ab = max(float(np.abs(frames_of(session, f) - frames_of(unheld, f)).max())
                         for f in range(1, 9))
            details.update({"largest_difference": gap_ab, "displacement": scale})
            ok = gap_ab <= 1e-5 * scale
        check("fluid_held_run_matches_an_unheld_one", ok, details)

        fine = [uniform((-1.2 * math.cos(0.5 * f), -0.6 - 0.2 * f,
                         0.9 * math.sin(0.9 * f) + 0.5), n=(16, 16, 16)) for f in range(6)]
        larger = FluidVolume(fine, cell=0.125, corner=box_corner, start=8 * dt, step=0.02)
        scene.force_field.clear().add("fluid", fluid=larger, strength=1.0e3, flow=1.0)
        session.update_force_field(scene.force_field)
        session.step_frame(2)
        held = session.held_frame()
        new_bricks = capacity(larger)[1]
        old_bricks = capacity(held_volume)[1]
        details = {"held": held, "bricks_per_frame": [old_bricks, new_bricks]}
        ok = held == 10 and new_bricks >= 4 * old_bricks
        if ok:
            new_fields = scene.force_field.fields
            ok, details["steps"] = steps_match(new_fields, session, (9,))
            x, v, _, want, pushed, _ = step_error(new_fields, session, 9)
            details["old_field_separation"] = float(np.abs(
                step_reference(old_fields, x, v, dt, time=9 * dt) - want).max()) / pushed
            ok = ok and details["old_field_separation"] > 0.5
            # Recorded, not judged: the step taken before the update, against
            # each field.
            x, v, got, want, pushed, miss = step_error(new_fields, session, 8)
            old = step_reference(old_fields, x, v, dt, time=8 * dt)
            details["frames_8_to_9_from_the_new_field"] = miss / pushed
            details["frames_8_to_9_from_the_old_field"] = float(
                np.abs(got - old).max() / np.abs(old - x - v * dt).max())
        check("fluid_update_while_held_reinstalls_a_larger_volume", ok, details)

        session.save_and_quit()
        deadline = time.time() + 120
        while session.is_running() and time.time() < deadline:
            time.sleep(0.1)
        saved = list(session.get.saved())
        session.release(blocking=False)
        session.resume(blocking=True)
        details = {"saved": saved, "finished": bool(session.finished())}
        ok = session.finished() and bool(saved) and max(saved) == 10
        if ok:
            ok, details["steps"] = steps_match(scene.force_field.fields, session,
                                               range(10, 16))
        check("fluid_resume_reads_the_updated_volume_at_its_time", ok, details)

        rebuilt = App.recover("rig_fluid_held").session.build()
        directory = os.path.join(rebuilt.info.path, "bin", "force_field", "field-0-fluid")
        same = []
        for k in range(len(held_u)):
            table = np.fromfile(os.path.join(directory, "table-%05d.bin" % k), dtype="<u4")
            bricks = np.fromfile(os.path.join(directory, "bricks-%05d.bin" % k), dtype="<f4")
            same.append(bool(np.array_equal(table, np.asarray(held_volume.tables[k]).ravel())
                             and np.array_equal(bricks,
                                                np.asarray(held_volume.bricks[k]).ravel())))
        check("fluid_session_saved_at_build_rebuilds_after_the_update", all(same),
              {"frames_equal": same})
    except Exception as e:
        for name in held_checks:
            if name not in cases:
                check(name, False, {"raised": repr(e)})
    finally:
        if session is not None:
            stop_held(session)

    # A FRAME FILE GROWN UNDER A HELD RUN. Twelve frames of 24 x 16 x 8
    # cells (3 x 2 x 1 bricks), two of them occupied, 0.025 s apart. Held at
    # frame 1, frame 4's table and bricks are rewritten as a consistent frame
    # of five bricks, past the two-brick slots the run sized at load. Frame 4
    # first enters at the eighth step, ending at 0.08 s (bracket (3, 4)),
    # which would write frame 7 (see the top of this section); that step must
    # stop the run naming the file, the frame and both counts, before a byte
    # of the file lands in a slot, so frame 6 is the last written.
    grown_n = (24, 16, 8)

    def occupied(pairs, value):
        u_g = np.zeros(grown_n + (3,))
        for bx, by in pairs:
            u_g[8 * bx:8 * bx + 8, 8 * by:8 * by + 8] = value
        return u_g

    grow = FluidVolume([occupied(((0, 0), (2, 1)), (1.0 + 0.2 * f, 0.5, -0.3))
                        for f in range(12)], cell=0.1,
                       corner=center - np.array([1.2, 0.8, 0.4]), step=0.025)
    begin("fluid_grown_under_a_held_run")
    app = new_app("fluid_grown")
    scene = app.scene.create()
    scene.add("sheet").at(0.2, 0.9, -0.3).velocity(*v0).param.set(
        "young-mod", 1e-3).set("bend", 0.0)
    scene.force_field.add("fluid", fluid=grow, strength=20.0, flow=1.0)
    fixed = scene.build(quiet=True)
    session = app.session.create(fixed)
    session.param.set("frames", 12).set("gravity", [0.0, 0.0, 0.0]).set("dt", dt).set(
        "fps", 100.0).set("air-density", 0.0)
    session = session.build()
    held = None
    raised = None
    failure = None
    try:
        session.run_until_frame(1)
        held = session.held_frame()
        table, bricks = fluid_bricks(occupied(((0, 0), (1, 0), (2, 0), (0, 1), (2, 1)),
                                              (2.0, 0.5, -0.3)))
        directory = os.path.join(session.info.path, "bin", "force_field", "field-0-fluid")
        np.ascontiguousarray(table, dtype="<u4").tofile(
            os.path.join(directory, "table-00004.bin"))
        np.ascontiguousarray(bricks, dtype="<f4").tofile(
            os.path.join(directory, "bricks-00004.bin"))
        try:
            session.step_frame(10)
        except RuntimeError as e:
            raised = str(e)
        deadline = time.time() + 60
        while session.is_running() and time.time() < deadline:
            time.sleep(0.1)
    except Exception as e:
        failure = repr(e)
    finally:
        stop_held(session)
    log = log_text(session)
    latest = session.get.latest_frame()
    named = ("bricks-00004.bin" in log and "frame 4" in log and "5 bricks" in log
             and "2 at load" in log)
    lines = [l for l in log.splitlines() if "bricks-00004.bin" in l]
    details = {"held": held, "raised": raised is not None, "failure": failure,
               "latest_frame": latest,
               "named": named, "panicked": "panicked" in log,
               "running": bool(session.is_running()), "finished": bool(session.finished()),
               "message": lines[-1][-400:] if lines else log[-600:]}
    check("fluid_frame_grown_under_a_held_run_fails_by_name",
          failure is None and held == 1 and raised is not None and not session.finished()
          and not session.is_running() and named and "panicked" not in log and latest == 6,
          details)

print("PPFRESULT" + json.dumps(cases))
'''


def run_section(ctx: r.ScenarioContext, section: str, label: str) -> dict:
    """Run one section of the probe and report its cases."""
    env = dict(os.environ)
    env["PPF_CTS_DATA_ROOT"] = ctx.workspace
    env["PYTHONPATH"] = REPO_ROOT_POSIX
    # THE SCRIPT GOES THROUGH A FILE, not `-c`: it is longer than the 32,767
    # characters a Windows command line holds ([WinError 206]), which Linux
    # and macOS never show.
    os.makedirs(ctx.workspace, exist_ok=True)
    script = os.path.join(ctx.workspace, "force_field_probe.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(PROBE)
    try:
        proc = subprocess.run(
            [sys.executable, script, REPO_ROOT_POSIX, section],
            capture_output=True,
            text=True,
            env=env,
            timeout=max(ctx.timeout, 900.0),
        )
    except subprocess.TimeoutExpired as expired:
        out = expired.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        begun = [l[len("PPFSTART "):] for l in out.splitlines()
                 if l.startswith("PPFSTART ")]
        return r.failed([
            f"probe section {section!r} timed out after {expired.timeout:.0f}s "
            f"inside {begun[-1] if begun else 'no solve'!r}"
        ])
    marker = [
        line for line in proc.stdout.splitlines() if line.startswith("PPFRESULT")
    ]
    if not marker:
        return r.failed([
            "probe produced no result marker; "
            f"rc={proc.returncode} stdout={proc.stdout[-800:]!r} "
            f"stderr={proc.stderr[-800:]!r}"
        ])
    cases = json.loads(marker[-1][len("PPFRESULT"):])
    return r.report_named_checks(cases, label=label)
