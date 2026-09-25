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
# weight-0 object, a sheet outside the grid and a zero-density air grid each
# must stay still, which is what separates "the field acts" from "something
# moved".
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

from frontend import App, ForceFieldScriptError
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
    # A uniform acceleration grid and a constant script each reproduce plain
    # gravity: all three enter the same target, so a single sheet with no
    # contact moves identically.
    frames = 10
    results = {}
    for label in ("gravity", "grid", "script"):
        begin(label)
        app = new_app("sources_" + label)
        scene = app.scene.create()
        scene.add("sheet").at(0.0, 1.0, 0.0)
        params = {}
        if label == "gravity":
            params["gravity"] = [0.0, -9.8, 0.0]
        elif label == "grid":
            vals = np.zeros((2, 2, 2, 3))
            vals[..., 1] = -9.8
            scene.force_field.grid(vals, (-2, -2, -2), (2, 4, 2))
        else:
            scene.force_field.script(
                "def eval(x, y, z, t):\n    return (0.0, -9.8, 0.0)\n")
        _, session = solve(app, scene, frames, params)
        d = frames_of(session, frames) - frames_of(session, 0)
        results[label] = {"finished": bool(session.finished()),
                          "mean": d.mean(axis=0).tolist()}
    g = np.array(results["gravity"]["mean"])
    for label in ("grid", "script"):
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
    # Space: a grid pushing -x on its left half and +x on its right half moves
    # two sheets apart; a third sheet outside the box does not move, and the
    # solver says how many free vertices were outside every grid.
    begin("spatial")
    app = new_app("spatial")
    scene = app.scene.create()
    scene.add("sheet").at(-0.6, 1.0, 0.0)
    scene.add("sheet").at(0.6, 1.0, 0.0)
    scene.add("sheet").at(3.0, 1.0, 0.0)
    xs = np.linspace(-1.2, 1.2, 9)
    vals = np.zeros((2, 2, 9, 3))
    vals[..., 0] = np.sign(xs)[None, None, :] * 4.0
    scene.force_field.grid(vals, (-1.2, 0.0, -1.0), (1.2, 2.0, 1.0))
    fixed, session = solve(app, scene, 10)
    x0 = frames_of(session, 0)[:, 0]
    d = frames_of(session, 10) - frames_of(session, 0)
    left = d[x0 < 0][:, 0].mean()
    right = d[(x0 > 0) & (x0 < 2)][:, 0].mean()
    outside = np.abs(d[x0 > 2]).max()
    log = log_text(session)
    check("grid_pushes_by_position", session.finished() and left < -0.02 and right > 0.02,
          {"left_dx": float(left), "right_dx": float(right)})
    check("outside_the_box_is_unforced_and_logged",
          outside < 1e-7 and "outside every grid" in log,
          {"outside_max_disp": float(outside),
           "logged": "outside every grid" in log})

    # World Scaling: the grid is authored in scene units, so the same scene
    # simulated at twice its size is pushed by the same halves of the box;
    # were the box read in solver units, the sheet at x = 0.6 (1.2 in the
    # solver) would sit on the box's edge and the one at 3.0 inside nothing.
    begin("world_scaling")
    app = new_app("world_scaling")
    scene = app.scene.create()
    scene.add("sheet").at(-0.6, 1.0, 0.0)
    scene.add("sheet").at(0.6, 1.0, 0.0)
    scene.add("sheet").at(3.0, 1.0, 0.0)
    scene.force_field.grid(vals, (-1.2, 0.0, -1.0), (1.2, 2.0, 1.0))
    fixed, session = solve(app, scene, 10, {"world-scaling": 2.0})
    x0 = frames_of(session, 0)[:, 0]
    d = frames_of(session, 10) - frames_of(session, 0)
    left = d[x0 < 0][:, 0].mean()
    right = d[(x0 > 0) & (x0 < 2)][:, 0].mean()
    outside = np.abs(d[x0 > 2]).max()
    check("grid_is_authored_in_scene_units_under_world_scaling",
          session.finished() and left < -0.005 and right > 0.005 and outside < 1e-7,
          {"left_dx": float(left), "right_dx": float(right),
           "outside_max_disp": float(outside)})

    # Time: two samples, held at each end, push down until t = 0.2 s and up
    # hard after 0.21 s, so the sheet falls and then climbs.
    begin("time")
    app = new_app("time")
    scene = app.scene.create()
    scene.add("sheet").at(0.0, 1.0, 0.0)
    vals = np.zeros((2, 2, 2, 2, 3))
    vals[0, ..., 1] = -10.0
    vals[1, ..., 1] = 30.0
    scene.force_field.grid(vals, (-2, -2, -2), (2, 4, 2), times=[0.2, 0.21])
    fixed, session = solve(app, scene, 30, {"fps": 60})
    y = [float(frames_of(session, n)[:, 1].mean()) for n in (0, 12, 30)]
    check("time_samples_reverse_the_motion",
          session.finished() and y[1] < y[0] - 0.05 and y[2] > y[1] + 0.05,
          {"y0": y[0], "y12": y[1], "y30": y[2]})

    # Air velocity: a vertical sheet facing +z in a +z flow moves downstream
    # when the air has density, and not at all when it has none.
    for density in (1.0, 0.0):
        begin("air_%g" % density)
        app = new_app("air_%g" % density)
        scene = app.scene.create()
        scene.add("wall").at(0.0, 1.0, 0.0)
        vals = np.zeros((2, 2, 2, 3))
        vals[..., 2] = 5.0
        scene.force_field.grid(vals, (-2, -2, -2), (2, 4, 2), kind="air-velocity")
        fixed, session = solve(app, scene, 10, {"air-density": density})
        dz = float((frames_of(session, 10) - frames_of(session, 0))[:, 2].mean())
        if density > 0:
            check("air_velocity_drives_the_drag", session.finished() and dz > 1e-3, {"dz": dz})
        else:
            check("air_velocity_needs_density", session.finished() and abs(dz) < 1e-7, {"dz": dz})

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
    # Sources reach the groups they name: a grid for group "a" pushes down, a
    # script for group "b" pushes up, a script for everyone pushes +x, and an
    # ungrouped sheet gets only the last. Two scripts on one sheet add up.
    begin("targets")

    def run_targets(label, with_b_script):
        app = new_app(label)
        scene = app.scene.create()
        scene.add("sheet").at(-1.4, 1.0, 0.0).group("a")
        scene.add("sheet").at(0.0, 1.0, 0.0).group("b")
        scene.add("sheet").at(1.4, 1.0, 0.0)
        vals = np.zeros((2, 2, 2, 3))
        vals[..., 1] = -8.0
        scene.force_field.grid(vals, (-3, -2, -2), (3, 4, 2), groups=["a"])
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

print("PPFRESULT" + json.dumps(cases))
'''


def run_section(ctx: r.ScenarioContext, section: str, label: str) -> dict:
    """Run one section of the probe and report its cases."""
    env = dict(os.environ)
    env["PPF_CTS_DATA_ROOT"] = ctx.workspace
    env["PYTHONPATH"] = REPO_ROOT_POSIX
    try:
        proc = subprocess.run(
            [sys.executable, "-c", PROBE, REPO_ROOT_POSIX, section],
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
