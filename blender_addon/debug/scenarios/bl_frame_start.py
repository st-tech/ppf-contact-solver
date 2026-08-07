# File: scenarios/bl_frame_start.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The simulation's starting frame has two possible sources, and exactly one of
# them may reach the encoders:
#
#   use_scene_frame_start = False -> the `frame_start` field
#   use_scene_frame_start = True  -> the Blender scene's start frame
#
# `resolve_start_frame()` (core/encoder/__init__.py) is the single place that
# decides. It is the time ORIGIN: every encoder converts a Blender frame to
# seconds as `(frame - start) / fps`, and playback maps solver frame n back to
# Blender frame `n + start`. This mirrors `bl_fps_source` (which covers the
# rate) and carries the same trap: `frame_start` KEEPS ITS LAST VALUE while the
# scene override is on, so the scene is deliberately left in the disagreeing
# state and every hop is asserted against the effective value.
#
# The substantive assertion is D. Before this feature the encoders hardcoded
# `(frame - 1) / fps`, so a solve could only ever occupy frames 1..N. A
# keyframe authored ON the starting frame must encode to t=0, not to
# `(start - 1) / fps`, or the whole schedule is shifted by the lead-in and the
# solver would sit idle through it.
#
# Subtests:
#   A. field_mode_resolves:    override off -> resolve_start_frame is the
#         `frame_start` field.
#   B. scene_mode_resolves:    override on  -> the scene's start frame, even
#         though `frame_start` still reads its old value.
#   C. mcp_reports_effective:  get_scene_parameters() surfaces
#         `effective_start_frame` = the frame actually in use, not the stale
#         field.
#   D. keyframes_relative_to_start: a STATIC's transform keys authored at
#         START and START+k encode to t=0 and t=k/fps. This is the hop that
#         would silently shift every schedule if an encoder kept the old
#         `- 1`.
#   E. default_is_frame_one:   a fresh state resolves to 1, so every existing
#         scene keeps its exact current behavior.
#   F. param_toml_frames_is_a_count (host-side): `frames` in the solver's
#         literal on-disk input does NOT move with the starting frame. It is a
#         count; the offset lives entirely in the addon.
#   G. active_until_is_rebased: the invisible-collider "Active Until (frame)"
#         cutoff is an absolute Blender frame, not a duration, so it rebases
#         on the start like a keyframe. Un-rebased it lands ~S frames late and
#         a collider the artist retired early never turns off.
#   H. pre_start_velocity_key_clamps: a translational velocity keyframe
#         authored BEFORE the start frame is the velocity the object enters
#         the solve with, so it clamps onto t=0 rather than being dropped from
#         both the initial value and the schedule.
#   I. panel_updates_expand_timeline: changing Starting Frame or Frame Count
#         through their scene properties expands a short Blender timeline to
#         include every configured simulation frame.
#   J. mcp_updates_expand_timeline: set_scene_parameters applies the same
#         timeline synchronization as the Scene panel properties.

from __future__ import annotations

import glob
import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# Backend-agnostic: every assertion is a value-propagation invariant, and
# param.toml is written by the frontend before any backend runs.
BACKENDS = ("emulated", "real")

# Above the `frame_count` property's min=10, so the host-side `frames` check
# asserts the value it was actually given rather than a silently clamped one.
_FRAME_COUNT = 12
_FPS = 100
# Deliberately different from each other, from 1, and from every default, so a
# hop that reads the wrong source (or keeps the old hardcoded 1) is
# unmistakable.
_FIELD_START = 30
_SCENE_START = 17
# Offset of the second transform key from the starting frame.
_KEY_OFFSET = 4


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>
FPS = <<FPS>>
FIELD_START = <<FIELD_START>>
SCENE_START = <<SCENE_START>>
KEY_OFFSET = <<KEY_OFFSET>>

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(size=1.0, x_subdivisions=3,
                                    y_subdivisions=3, location=(0.0, 0.0, 0.0))
    cloth = bpy.context.active_object
    cloth.name = "StartCloth"

    dh.save_blend(PROBE_DIR, "frame_start.blend")
    root = dh.configure_state(project_name="frame_start",
                              frame_count=FRAME_COUNT,
                              frame_rate=FPS,
                              step_size=1.0 / 60.0)
    state = root.state
    bpy.context.scene.render.fps = FPS
    bpy.context.scene.frame_start = SCENE_START
    bpy.context.scene.frame_end = SCENE_START + FRAME_COUNT - 1

    group = dh.api.solver.create_group("Cloth", "SHELL")
    group.add(cloth.name)

    encoder = __import__(pkg + ".core.encoder",
                         fromlist=["resolve_start_frame"])

    # ---- E: a fresh state resolves to 1 ------------------------------
    # Asserted BEFORE the field is touched: the whole compatibility story
    # for existing scenes rests on this default.
    default_start = int(encoder.resolve_start_frame(state))
    dh.record(
        "E_default_is_frame_one",
        default_start == 1 and state.use_scene_frame_start is False,
        {"resolve_start_frame": default_start, "expected": 1,
         "use_scene_frame_start": bool(state.use_scene_frame_start)},
    )

    # ---- I/J: configured output range expands a short timeline -------
    # Playback maps solver indices 0..N-1 to Blender frames
    # start..start+N-1. A shorter scene range truncates Fetch/PC2 even
    # though the UI still reports all N configured frames.
    bpy.context.scene.frame_end = 22
    state.frame_start = 7
    state.frame_count = 250
    panel_expected_end = 7 + 250 - 1
    panel_actual_end = int(bpy.context.scene.frame_end)
    dh.record(
        "I_panel_updates_expand_timeline",
        panel_actual_end >= panel_expected_end,
        {"frame_end": panel_actual_end, "expected_at_least": panel_expected_end,
         "frame_start": int(state.frame_start),
         "frame_count": int(state.frame_count)},
    )

    bpy.context.scene.frame_end = 22
    remote = __import__(pkg + ".mcp.handlers.remote",
                        fromlist=["set_scene_parameters"])
    remote.set_scene_parameters({"frame_start": 13, "frame_count": 40})
    mcp_expected_end = 13 + 40 - 1
    mcp_actual_end = int(bpy.context.scene.frame_end)
    dh.record(
        "J_mcp_updates_expand_timeline",
        mcp_actual_end >= mcp_expected_end,
        {"frame_end": mcp_actual_end, "expected_at_least": mcp_expected_end,
         "frame_start": int(state.frame_start),
         "frame_count": int(state.frame_count)},
    )

    state.frame_count = FRAME_COUNT
    bpy.context.scene.frame_end = SCENE_START + FRAME_COUNT - 1
    state.frame_start = FIELD_START

    # ---- A: override off -> the field is the source ------------------
    state.use_scene_frame_start = False
    start_field_mode = int(encoder.resolve_start_frame(state))
    dh.record(
        "A_field_mode_resolves",
        start_field_mode == FIELD_START,
        {"resolve_start_frame": start_field_mode, "expected": FIELD_START,
         "scene_frame_start": int(bpy.context.scene.frame_start)},
    )

    # ---- B: override on -> the scene is the source -------------------
    # `frame_start` is deliberately left at FIELD_START. This is the state
    # the confusion comes from: the field says one thing, the solve starts
    # somewhere else.
    state.use_scene_frame_start = True
    start_scene_mode = int(encoder.resolve_start_frame(state))
    dh.record(
        "B_scene_mode_resolves",
        start_scene_mode == SCENE_START
        and int(state.frame_start) == FIELD_START,
        {"resolve_start_frame": start_scene_mode, "expected": SCENE_START,
         "frame_start_field_still": int(state.frame_start)},
    )

    # ---- C: MCP surfaces the EFFECTIVE frame, not the stale field ----
    remote = __import__(pkg + ".mcp.handlers.remote",
                        fromlist=["get_scene_parameters"])
    params = remote.get_scene_parameters({})
    reported = params.get("parameters", {})
    eff = reported.get("effective_start_frame")
    dh.record(
        "C_mcp_reports_effective",
        eff is not None and int(eff) == SCENE_START,
        {"effective_start_frame": eff, "expected": SCENE_START,
         "frame_start_reported": reported.get("frame_start"),
         "use_scene_frame_start": reported.get("use_scene_frame_start")},
    )

    # ---- D: keyframe times are relative to the starting frame --------
    # A STATIC collider with object-level transform keys ON the starting
    # frame and KEY_OFFSET frames later. get_transform_keyframes is the
    # encoder's sparse-sample path; its times must read [0, KEY_OFFSET/fps].
    # Under the old hardcoded origin the first sample would land at
    # (SCENE_START - 1) / FPS instead, i.e. the solver would idle through
    # the entire lead-in before the collider started moving.
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0.0, 0.0, 2.0))
    collider = bpy.context.active_object
    collider.name = "StartCollider"
    static_group = dh.api.solver.create_group("Collider", "STATIC")
    static_group.add(collider.name)

    collider.location = (0.0, 0.0, 2.0)
    collider.keyframe_insert("location", frame=SCENE_START)
    collider.location = (0.0, 0.0, 1.0)
    collider.keyframe_insert("location", frame=SCENE_START + KEY_OFFSET)

    utils = __import__(pkg + ".core.utils",
                       fromlist=["get_transform_keyframes"])
    kf = utils.get_transform_keyframes(
        collider, bpy.context,
        encoder.resolve_start_frame(state),
    )
    # v2 wire: FRAME offsets relative to the start frame; the decoder
    # derives seconds from the Param fps.
    offsets = [float(t) for t in (kf or {}).get("frame_offset", [])]
    expected_offsets = [0.0, float(KEY_OFFSET)]
    times_ok = (
        len(offsets) == len(expected_offsets)
        and all(abs(a - b) < 1e-9 for a, b in zip(offsets, expected_offsets))
    )
    dh.record(
        "D_keyframes_relative_to_start",
        times_ok,
        {"frame_offsets": offsets, "expected": expected_offsets,
         "start_frame_in_use": int(encoder.resolve_start_frame(state)),
         "old_hardcoded_would_be": [
             float(SCENE_START - 1),
             float(SCENE_START + KEY_OFFSET - 1),
         ]},
    )

    # ---- G: "Active Until (frame)" is an absolute frame ---------------
    # Despite the property being named `active_duration`, it is the first
    # Blender frame at which the collider stops acting, so it must be
    # rebased on the starting frame like any keyframe. Left un-rebased, the
    # cutoff lands ~S frames late and a collider the artist retired early
    # stays active for the whole solve.
    dyn = __import__(pkg + ".core.encoder.dyn",
                     fromlist=["_active_duration_cutoff"])

    class _ColliderStub:
        enable_active_duration = True
        active_duration = SCENE_START + 20

    cutoff = float(dyn._active_duration_cutoff(
        _ColliderStub(), float(FPS), SCENE_START,
    ))
    # 20 frames into the solve, less the half-frame boundary margin.
    expected_cutoff = (20 - 0.5) / float(FPS)
    dh.record(
        "G_active_until_is_rebased",
        abs(cutoff - expected_cutoff) < 1e-9,
        {"cutoff": cutoff, "expected": expected_cutoff,
         "active_until_frame": SCENE_START + 20,
         "start_frame": SCENE_START,
         "un_rebased_would_be": (SCENE_START + 20 - 1.5) / float(FPS)},
    )

    # ---- H: a velocity key before the start is not dropped -----------
    # It is the velocity the object ENTERS the solve with, so it clamps onto
    # t=0. The old `== start_frame` match silently dropped it from both the
    # initial value and the schedule, launching the object at rest.
    params_mod = __import__(pkg + ".core.encoder.params",
                            fromlist=["_initial_translational_velocity"])

    class _VelKeyStub:
        def __init__(self, frame, speed):
            self.frame = frame
            self.speed = speed
            self.direction = (0.0, 0.0, 1.0)
            self.enable_translational = True

    class _AssignedStub:
        # Deliberately unsorted, and both keys sit BEFORE the start frame:
        # the later one is the value in effect at t=0.
        velocity_keyframes = [_VelKeyStub(1, 3.0), _VelKeyStub(5, 7.0)]

    vel = params_mod._initial_translational_velocity(
        _AssignedStub(), SCENE_START,
    )
    speed_out = float(max(abs(c) for c in vel))
    dh.record(
        "H_pre_start_velocity_key_clamps",
        abs(speed_out - 7.0) < 1e-6,
        {"initial_velocity": [float(c) for c in vel],
         "expected_speed": 7.0, "start_frame": SCENE_START,
         "note": "keys at frames 1 and 5 both precede the start; last wins"},
    )

    # Leave the scene in the disagreeing state for the build: override on,
    # field still FIELD_START. param.toml's `frames` must be a plain count.
    state.use_scene_frame_start = True

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="frame_start:build")

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
        .replace("<<FPS>>", str(_FPS))
        .replace("<<FIELD_START>>", str(_FIELD_START))
        .replace("<<SCENE_START>>", str(_SCENE_START))
        .replace("<<KEY_OFFSET>>", str(_KEY_OFFSET))
    )


def _check_param_toml_frames(workspace: str) -> tuple[bool, dict]:
    """Host-side: `frames` is a COUNT and must not move with the starting
    frame.

    The solver has no notion of a starting frame at all: it runs from t=0 and
    derives its own frame index as floor(time * fps). Offsetting `frames` here
    would be a plausible-looking "fix" that silently changes how long every
    scene simulates, so the count is pinned explicitly.
    """
    matches = glob.glob(os.path.join(workspace, "**", "param.toml"),
                        recursive=True)
    if not matches:
        return False, {"error": "no param.toml under workspace",
                       "workspace": workspace}
    found = []
    for path in matches:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("frames"):
                        try:
                            found.append(int(float(s.split("=", 1)[1].strip())))
                        except (ValueError, IndexError):
                            pass
        except OSError:
            continue
    expected = _FRAME_COUNT - 1
    ok = len(found) > 0 and all(v == expected for v in found)
    return ok, {"frames_in_param_toml": found, "expected": expected,
                "starting_frame_that_must_not_leak": _SCENE_START,
                "param_toml_paths": matches}


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    checks = dict(result.get("checks", {}))
    ok, details = _check_param_toml_frames(ctx.workspace)
    checks["F_param_toml_frames_is_a_count"] = {"ok": ok, "details": details}
    return r.report_named_checks(checks)
