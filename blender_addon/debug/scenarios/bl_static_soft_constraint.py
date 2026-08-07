# File: scenarios/bl_static_soft_constraint.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# "Apply Soft Constraints" on a STATIC group: the collider's pins change from
# exact Dirichlet boundary conditions to Hookean springs of the group's
# Stiffness, so contact can push the collider off its animated path.
#
# One build carries three colliders that differ only in this setting:
#   Soft   - animated, soft constraints on  -> pin block with pull == STIFFNESS
#   Hard   - animated, soft constraints off -> pin block with pull == 0 (exact)
#   Rest   - NOT animated, soft constraints on -> still gets a pin block
#
# The third is the subtle one. A rest-pose STATIC is normally a disjoint
# contact-only collision mesh with no pins at all, so there would be nothing
# for a stiffness to act on and the checkbox would be a silent no-op. The
# decoder promotes it into the solved namespace instead, the same way it
# already promotes a cross-stitch endpoint.
#
# Subtests:
#   A. soft_group_becomes_pull_pin:
#         the soft collider's pin block carries pull == STIFFNESS.
#   B. unset_group_stays_exact_pin:
#         the hard collider's pin block carries pull == 0.0, i.e. the default
#         is unchanged and this feature is opt-in.
#   C. rest_pose_soft_collider_is_promoted:
#         the non-animated soft collider produces a pin block at all.
#   D. rna_registered_with_defaults:
#         both properties exist on the group with the documented defaults.
#   E. non_positive_stiffness_rejected:
#         the encoder refuses a non-positive stiffness rather than shipping it.
#         The solver reads a pull weight of zero as "this pin is an exact fix",
#         so forwarding one would hand back the exact constraint the group
#         asked to drop, with nothing in the UI or the log to say so. The RNA
#         minimum does not cover this on its own: Blender clamps a plain 0.0,
#         but a NaN defeats every clamp comparison and would reach the wire.
#
# Why this checks the BUILT scene rather than runtime motion: the emulated
# (CPU) solver the rig runs does not integrate a soft-pull follow, so a
# spring-held collider stays bit-for-bit at rest there and no amount of
# stiffness would show up in the PC2. What this proves end to end is that the
# UI setting reaches the solver scene as the right KIND of pin, which is
# exactly what the feature changes. The physics is exercised on real CUDA.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True


_FRAME_COUNT = 6
_STIFFNESS = 12.5


_DRIVER_BODY = r"""
import glob
import os
import re
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = int("<<SERVER_PORT>>")
FRAME_COUNT = int("<<FRAME_COUNT>>")
STIFFNESS = float("<<STIFFNESS>>")
WORKSPACE = "<<WORKSPACE>>"


def _parse_one(path):
    # {object name: pull weight} for a single info.toml. The holder header
    # carries `pin_group_id = "<object>:<pin name>"`, which is what ties a pin
    # block back to the collider it came from; the `[pin-N-op-M]` sub-blocks
    # carry no `pull` and are skipped.
    out = {}
    txt = open(path).read()
    for body in re.findall(r"\[pin-\d+\]\n(.*?)(?=\n\[|\Z)", txt, re.S):
        pm = re.search(r"pull\s*=\s*([0-9.eE+-]+)", body)
        gm = re.search(r'pin_group_id\s*=\s*"([^":]+)', body)
        if pm is None or gm is None:
            continue
        out[gm.group(1)] = float(pm.group(1))
    return out


def _parse_pin_blocks():
    # The worker lays the built session down under its own workspace, and a
    # run can leave more than one candidate (the project dir plus a symlink
    # view of it), so take the one that actually carries pin holders rather
    # than whichever the glob happens to order first.
    hits = sorted(glob.glob(os.path.join(WORKSPACE, "**", "session",
                                         "info.toml"), recursive=True))
    best, best_path = {}, ""
    for h in hits:
        try:
            got = _parse_one(h)
        except OSError:
            continue
        if len(got) > len(best):
            best, best_path = got, h
    return best, best_path


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAME_COUNT

    # A cloth sheet so the scene has a dynamic object; the colliders sit below
    # it. Nothing here depends on them actually touching.
    bpy.ops.mesh.primitive_grid_add(size=1.0, x_subdivisions=6,
                                    y_subdivisions=6, location=(0, 0, 1.0))
    cloth = bpy.context.active_object
    cloth.name = "Cloth"

    def _cube(name, location, animate):
        bpy.ops.mesh.primitive_cube_add(size=0.4, location=location)
        o = bpy.context.active_object
        o.name = name
        if animate:
            # Two location keys make the encoder emit transform_animation, so
            # this collider decodes to a moving pin shell.
            o.location = location
            o.keyframe_insert(data_path="location", frame=1)
            o.location = (location[0], location[1], location[2] + 0.15)
            o.keyframe_insert(data_path="location", frame=FRAME_COUNT)
        return o

    soft_obj = _cube("SoftCollider", (0.0, 0.0, 0.0), True)
    hard_obj = _cube("HardCollider", (1.0, 0.0, 0.0), True)
    rest_obj = _cube("RestCollider", (2.0, 0.0, 0.0), False)

    shell = dh.api.solver.create_group("Cloth", "SHELL")
    shell.add(cloth.name)
    g_soft = dh.api.solver.create_group("Soft", "STATIC")
    g_soft.add(soft_obj.name)
    g_hard = dh.api.solver.create_group("Hard", "STATIC")
    g_hard.add(hard_obj.name)
    g_rest = dh.api.solver.create_group("Rest", "STATIC")
    g_rest.add(rest_obj.name)

    addon_root = dh.groups.get_addon_data(scene)
    groups_mod = __import__(pkg + ".models.groups", fromlist=["get_group_slot_index"])
    # Address a group by its SLOT, never by ObjectGroup.index: the two agree
    # only while every slot below the group is active.
    def _rna(facade_group):
        slot = groups_mod.get_group_slot_index(scene, facade_group.uuid)
        return getattr(addon_root, "object_group_%d" % slot)

    rna_soft = _rna(g_soft)
    rna_hard = _rna(g_hard)
    rna_rest = _rna(g_rest)

    # D: the RNA must have registered with the documented defaults. A missing
    # property here means the addon was soft-reloaded instead of restarted.
    defaults_ok = (
        hasattr(rna_hard, "enable_soft_constraint")
        and hasattr(rna_hard, "soft_constraint_stiffness")
        and rna_hard.enable_soft_constraint is False
        and abs(rna_hard.soft_constraint_stiffness - 10.0) < 1e-6
    )
    dh.record(
        "D_rna_registered_with_defaults", defaults_ok,
        {"has_enable": hasattr(rna_hard, "enable_soft_constraint"),
         "has_stiffness": hasattr(rna_hard, "soft_constraint_stiffness"),
         "enable_default": getattr(rna_hard, "enable_soft_constraint", None),
         "stiffness_default": getattr(rna_hard, "soft_constraint_stiffness", None)},
    )

    rna_soft.enable_soft_constraint = True
    rna_soft.soft_constraint_stiffness = STIFFNESS
    rna_rest.enable_soft_constraint = True
    rna_rest.soft_constraint_stiffness = STIFFNESS
    # rna_hard is left at its defaults on purpose: it is the control.

    # E: the encoder guard. Blender clamps a plain 0.0 to the RNA minimum, so
    # drive the helper directly with the value that defeats every clamp.
    params_mod = __import__(pkg + ".core.encoder.params",
                            fromlist=["_encode_soft_constraint"])

    class _Bypass:
        name = "Bypass"
        enable_soft_constraint = True
        soft_constraint_stiffness = float("nan")
        assigned_objects = []

    raised = False
    try:
        params_mod._encode_soft_constraint(_Bypass())
    except ValueError:
        raised = True
    # A positive stiffness must still pass, or the guard is just broken.
    class _Ok(_Bypass):
        soft_constraint_stiffness = 3.0
    ok_passes = params_mod._encode_soft_constraint(_Ok()) == {}
    dh.record(
        "E_non_positive_stiffness_rejected", raised and ok_passes,
        {"nan_raised": raised, "positive_still_encodes": ok_passes,
         "note": "zero is how the solver spells an exact pin, so a "
                 "non-positive stiffness must fail loudly, not silently "
                 "re-rigidify the collider"},
    )

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=addon_root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes,
                      message="soft_constraint:build", timeout=240.0)
    dh.log("built")

    pulls, info_path = _parse_pin_blocks()
    dh.log("pin blocks: %r" % (pulls,))

    soft_pull = pulls.get(soft_obj.name)
    dh.record(
        "A_soft_group_becomes_pull_pin",
        soft_pull is not None and abs(soft_pull - STIFFNESS) < 1e-4,
        {"pull": soft_pull, "expected": STIFFNESS, "all_pins": pulls,
         "info_toml": info_path},
    )

    hard_pull = pulls.get(hard_obj.name)
    dh.record(
        "B_unset_group_stays_exact_pin",
        hard_pull is not None and hard_pull == 0.0,
        {"pull": hard_pull, "expected": 0.0, "all_pins": pulls},
    )

    dh.record(
        "C_rest_pose_soft_collider_is_promoted",
        rest_obj.name in pulls
        and abs(pulls[rest_obj.name] - STIFFNESS) < 1e-4,
        {"pull": pulls.get(rest_obj.name), "expected": STIFFNESS,
         "note": "a rest-pose STATIC has no pins unless the decoder promotes "
                 "it, so without promotion the checkbox is a silent no-op"},
    )

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
        .replace("<<STIFFNESS>>", repr(_STIFFNESS))
        .replace("<<WORKSPACE>>", ctx.workspace.replace("\\", "/"))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
