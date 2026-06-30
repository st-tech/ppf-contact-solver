# File: scenarios/bl_world_scaling_encoder_scales.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Encode-level check of the addon-side world_scaling decisions (no
# build / run). The encoder is the half of world_scaling that decides
# which quantities scale with the geometry and which are fixed physical
# distances. The Rust solver shrinks the mesh by world_scaling, so:
#
#   * a RELATIVE contact gap (a fraction of the mesh bbox diagonal) is
#     multiplied by world_scaling so it stays the same fraction of the
#     scaled mesh;
#   * an ABSOLUTE contact gap is a fixed world-space distance, so it too
#     is multiplied by world_scaling (the solver shrinks the geometry by
#     the same factor), keeping it the same physical size as the mesh;
#   * a seeded frame-1 VELOCITY (written to vel.bin) must NOT be scaled
#     by the encoder: the solver multiplies it by world_scaling on
#     ingest, exactly like the geometry, so pre-scaling it here would
#     double-scale the launch to world_scaling^2 (commit c8a61856);
#   * a frame>1 VELOCITY SCHEDULE entry is a dyn_param the solver does
#     NOT scale on ingest, so the encoder must multiply it by
#     world_scaling to keep the scheduled launch physical.
#
# This rig encodes the SAME scene at world_scaling = 1 and = 10 and
# inspects the decoded param to assert exactly those behaviors. It is the
# contact-gap and velocity-scaling coverage the emulated solver cannot
# provide: the CUDA-free emulator has no contact pipeline (gaps never
# affect its output) and applies the vel.bin scaling on the solver side,
# but the encoder's scaling math is observable directly here.
#
# Subtests:
#   A. relative_gap_scales        - relative contact-gap goes 10x.
#   B. absolute_gap_scales        - absolute contact-gap goes 10x too.
#   C. seeded_velocity_unscaled   - frame-1 vel.bin stays fixed (the
#                                   solver, not the encoder, scales it).
#   D. velocity_schedule_scales   - frame>1 schedule velocity goes 10x.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import math
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def _raw_group(dh):
    for g in dh.groups.iterate_active_object_groups(bpy.context.scene):
        return g
    return None


def _encode_group_at(dh, root, ws):
    # Set world_scaling, encode the param, decode it, and return the
    # single group's decoded param dict. Each "group" entry is a
    # (params, objects, object_uuids) tuple (a list after the CBOR
    # round-trip), so the per-group material dict is element [0].
    root.state.world_scaling = float(ws)
    param_bytes = dh.encoder_param.encode_param(bpy.context)
    data = dh.decode_addon_blob(param_bytes)
    groups = data.get("group") or []
    if not groups:
        raise RuntimeError("decoded param has no group entries")
    return groups[0][0]


def _vel_mag(gd):
    vel = gd.get("velocity") or {}
    vecs = [v for v in vel.values() if v is not None]
    if not vecs:
        return 0.0
    v = vecs[0]
    return math.sqrt(sum(float(c) * float(c) for c in v))


def _sched_mag(gd):
    # Magnitude of the first frame>1 velocity-schedule entry. Each entry
    # is a (t, vec) pair (a list after the CBOR round-trip), so the
    # velocity vector is element [1].
    sched = gd.get("velocity-schedule") or {}
    for entries in sched.values():
        if entries:
            vec = entries[0][1]
            return math.sqrt(sum(float(c) * float(c) for c in vec))
    return 0.0


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="WsEncoderMesh")
    root = dh.configure_state(project_name="ws_encoder_scales",
                              frame_count=10, frame_rate=100)
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    cloth.create_pin(plane.name, "AllPin")
    # Frame-1 launch (vel.bin, scaled by the solver on ingest) plus a
    # frame>1 schedule entry (dyn_param, scaled by the encoder): the two
    # velocity paths scale on opposite sides of the wire, so author both
    # and check each independently.
    cloth.set_velocity(plane.name, direction=(1.0, 0.0, 0.0),
                       speed=2.0, frame=1)
    cloth.set_velocity(plane.name, direction=(1.0, 0.0, 0.0),
                       speed=3.0, frame=5)
    grp = _raw_group(dh)
    if grp is None:
        raise RuntimeError("no active group after create")

    # ---- relative gap + velocity: both scale with world_scaling ----
    grp.use_group_bounding_box_diagonal = True
    grp.contact_gap_rat = 0.01
    grp.contact_offset_rat = 0.02
    rel1 = _encode_group_at(dh, root, 1.0)
    rel10 = _encode_group_at(dh, root, 10.0)
    g1 = float(rel1["contact-gap"])
    g10 = float(rel10["contact-gap"])
    v1 = _vel_mag(rel1)
    v10 = _vel_mag(rel10)
    vs1 = _sched_mag(rel1)
    vs10 = _sched_mag(rel10)

    dh.record(
        "A_relative_gap_scales",
        g1 > 0.0 and abs(g10 - 10.0 * g1) <= 1e-4 * max(g10, 1.0),
        {"gap_ws1": g1, "gap_ws10": g10, "ratio": (g10 / g1) if g1 else None},
    )
    # Frame-1 seeded velocity is vel.bin: the solver scales it by
    # world_scaling on ingest, so the encoder must leave it fixed
    # (pre-scaling would double-scale the launch to world_scaling^2).
    dh.record(
        "C_seeded_velocity_unscaled",
        v1 > 0.0 and abs(v10 - v1) <= 1e-4 * max(v1, 1.0),
        {"vel_ws1": round(v1, 6), "vel_ws10": round(v10, 6),
         "ratio": (v10 / v1) if v1 else None},
    )
    # Frame>1 schedule velocity is a dyn_param the solver does not scale,
    # so the encoder multiplies it by world_scaling (10x here).
    dh.record(
        "D_velocity_schedule_scales",
        vs1 > 0.0 and abs(vs10 - 10.0 * vs1) <= 1e-4 * max(vs10, 1.0),
        {"sched_ws1": round(vs1, 6), "sched_ws10": round(vs10, 6),
         "ratio": (vs10 / vs1) if vs1 else None},
    )

    # ---- absolute gap: a world-space distance, so it scales 10x with
    # world_scaling and equals the authored value at ws=1.0.
    # contact_gap is clamped to [0.001, 0.01] by the property, so read
    # back the stored value as the ws=1.0 expectation rather than hard-coding.
    grp.use_group_bounding_box_diagonal = False
    grp.contact_gap = 0.008
    abs_authored = float(grp.contact_gap)
    abs1 = _encode_group_at(dh, root, 1.0)
    abs10 = _encode_group_at(dh, root, 10.0)
    a1 = float(abs1["contact-gap"])
    a10 = float(abs10["contact-gap"])
    dh.record(
        "B_absolute_gap_scales",
        a1 > 0.0 and abs(a1 - abs_authored) <= 1e-6
        and abs(a10 - 10.0 * a1) <= 1e-4 * max(a10, 1.0),
        {"gap_ws1": a1, "gap_ws10": a10, "authored": abs_authored,
         "ratio": (a10 / a1) if a1 else None},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 90.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
