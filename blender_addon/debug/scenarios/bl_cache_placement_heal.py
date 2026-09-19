# File: scenarios/bl_cache_placement_heal.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression guard: a ContactSolverCache bound while an object carried no
# enabled deformer must be RE-PLACED once a deformer appears, instead of
# staying above it forever.
#
# Placement is decided at bind time, and a bind only happens on the first
# frame written to a PC2 that is not already on disk. So an artist who
# solves first and rigs afterwards (or un-mutes an Armature that was off
# during the solve) keeps a cache at index 0 with the deformer below it,
# and re-running does not correct it: the PC2 is still there, so the frame
# write appends instead of rebinding. MESH_CACHE replays in OVERWRITE
# mode, so the deformer re-applies its deformation on top of the solver
# output that already carries it, and every driven vertex moves twice.
# ``heal_mesh_caches_if_stale`` is the only thing that runs afterwards, so
# it owns the correction; its staleness predicate is what this locks.
#
#   A. deformer_enabled_after_bind_is_rehealed:
#         bind with the Armature muted (cache lands at 0), un-mute it,
#         heal, and the cache must sit after the Armature and before the
#         Subdivision.
#   B. no_deformer_cache_stays_first:
#         negative. A dynamic object with no deformer keeps the cache
#         FIRST, decorators after, so a fix cannot be "push the cache
#         down the stack".
#   C. already_placed_cache_untouched:
#         negative. A stack that is already correct is not churned.
#   D. deform_only_geonodes_rehealed:
#         a Set Position group counts as a deformer, so the cache must
#         end up after it.
#   E. generative_geonodes_keeps_cache_first:
#         positive control for the conservative branch. A group that both
#         writes position AND changes the vertex count is a topology
#         boundary, so the cache stays first. This is the only case that
#         evaluates ``_stack_preserves_vertex_count``, whose depsgraph
#         round-trip the staleness predicate defers.
#   F. heal_is_idempotent:
#         a second heal changes nothing. The predicate and the move it
#         triggers have to agree on the index, or the frame pump rewrites
#         the modifier stack on every tick.
#
# The frame pump is not running here, so the driver calls
# ``heal_mesh_caches_if_stale`` directly: a timer body is invisible to the
# rig. Assertion-only, no server connection and no solve, so it is fast
# and deterministic on every host.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)


_DRIVER_TEMPLATE = r"""
import bpy, os, time, traceback
import numpy as np
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


def _new_geo_group(name):
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT",
                            socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT",
                            socket_type="NodeSocketGeometry")
    gin = ng.nodes.new("NodeGroupInput")
    gout = ng.nodes.new("NodeGroupOutput")
    return ng, gin, gout


def _set_position_group():
    # Moves every vertex, leaves the vertex count alone: a deformer.
    ng, gin, gout = _new_geo_group("_pcs_heal_set_position")
    sp = ng.nodes.new("GeometryNodeSetPosition")
    sp.inputs["Offset"].default_value = (0.0, 0.0, 0.25)
    ng.links.new(gin.outputs[0], sp.inputs["Geometry"])
    ng.links.new(sp.outputs["Geometry"], gout.inputs[0])
    return ng


def _set_position_and_subdivide_group():
    # Moves vertices AND adds them: deforming and generative at once, so
    # the cache has to stay above it.
    ng, gin, gout = _new_geo_group("_pcs_heal_set_position_subdiv")
    sp = ng.nodes.new("GeometryNodeSetPosition")
    sp.inputs["Offset"].default_value = (0.0, 0.0, 0.25)
    sub = ng.nodes.new("GeometryNodeSubdivideMesh")
    sub.inputs["Level"].default_value = 1
    ng.links.new(gin.outputs[0], sp.inputs["Geometry"])
    ng.links.new(sp.outputs["Geometry"], sub.inputs["Mesh"])
    ng.links.new(sub.outputs["Mesh"], gout.inputs[0])
    return ng


try:
    pc2m = __import__(pkg + ".core.pc2",
                      fromlist=["MODIFIER_NAME", "get_pc2_path",
                                "object_pc2_key", "create_pc2_file",
                                "append_pc2_frame", "setup_mesh_cache_modifier",
                                "cache_placement_is_stale"])
    client = __import__(pkg + ".core.client",
                        fromlist=["heal_mesh_caches_if_stale",
                                  "_needs_after_deformers"])
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_group_by_uuid"])
    enc = __import__(pkg + ".core.encoder",
                     fromlist=["resolve_start_frame_or_default"])
    api_mod = __import__(pkg + ".ops.api", fromlist=["solver"])
    solver_api = api_mod.solver
    ctx = bpy.context
    NAME = pc2m.MODIFIER_NAME

    log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    ctx.scene.frame_start = 1
    ctx.scene.frame_end = 12
    ctx.scene.frame_set(1)

    bpy.ops.object.armature_add(location=(0.0, 0.0, 2.0))
    arm = ctx.active_object
    arm.name = "HealArmature"

    group = solver_api.create_group("HealCloth", "SHELL")
    raw_group = groups_mod.get_group_by_uuid(ctx.scene, group.uuid)
    # heal_mesh_caches_if_stale walks ACTIVE groups only; a scenario that
    # silently landed an inactive group would assert nothing.
    group_active = bool(raw_group.active)
    if not group_active:
        raw_group.active = True

    def make_plane(name, loc):
        bpy.ops.mesh.primitive_grid_add(size=2.0, x_subdivisions=3,
                                        y_subdivisions=3, location=loc)
        obj = ctx.active_object
        obj.name = name
        group.add(obj.name)
        return obj

    def bind_cache(obj, place_after):
        # Stand in for the bind the first fetched frame performs: write a
        # one-frame PC2 at the object's rest pose, then attach the cache
        # exactly as _write_mesh_frame_to_pc2 would.
        key = pc2m.object_pc2_key(obj)
        path = pc2m.get_pc2_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        n = len(obj.data.vertices)
        co = np.empty(n * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", co)
        pc2m.create_pc2_file(path, n, start=0.0, sampling=1.0)
        pc2m.append_pc2_frame(path, co.reshape(n, 3), n)
        pc2m.setup_mesh_cache_modifier(obj, path, frame_start=1.0,
                                       place_after_deformers=place_after)
        return path

    def order(obj):
        return [m.name for m in obj.modifiers]

    def index_of(obj, name):
        return order(obj).index(name) if name in order(obj) else -1

    def stale(obj):
        return pc2m.cache_placement_is_stale(
            obj, client._needs_after_deformers(raw_group.object_type, obj)
        )

    # ---- A: deformer enabled after the bind ---------------------------
    a = make_plane("HealArmatureLate", (0.0, 0.0, 0.0))
    a_arm = a.modifiers.new("Armature", "ARMATURE")
    a_arm.object = arm
    a_arm.show_viewport = False
    a.modifiers.new("Subdivision", "SUBSURF")
    ctx.view_layer.update()
    a_path = bind_cache(a, client._needs_after_deformers("SHELL", a))
    a_bound = order(a)
    a_arm.show_viewport = True
    ctx.view_layer.update()
    a_stale_before = stale(a)

    # ---- B: no deformer at all ---------------------------------------
    b = make_plane("HealNoDeformer", (3.0, 0.0, 0.0))
    b.modifiers.new("Subdivision", "SUBSURF")
    ctx.view_layer.update()
    bind_cache(b, client._needs_after_deformers("SHELL", b))
    b_bound = order(b)
    b_stale_before = stale(b)

    # ---- C: already correctly placed ---------------------------------
    c = make_plane("HealAlreadyPlaced", (6.0, 0.0, 0.0))
    c_arm = c.modifiers.new("Armature", "ARMATURE")
    c_arm.object = arm
    c.modifiers.new("Subdivision", "SUBSURF")
    ctx.view_layer.update()
    bind_cache(c, client._needs_after_deformers("SHELL", c))
    c_bound = order(c)
    c_stale_before = stale(c)

    # ---- D: deform-only Geometry Nodes -------------------------------
    d = make_plane("HealGeoNodeDeform", (9.0, 0.0, 0.0))
    d_gn = d.modifiers.new("GeoDeform", "NODES")
    d_gn.node_group = _set_position_group()
    d_gn.show_viewport = False
    ctx.view_layer.update()
    bind_cache(d, client._needs_after_deformers("SHELL", d))
    d_bound = order(d)
    d_gn.show_viewport = True
    ctx.view_layer.update()
    d_stale_before = stale(d)

    # ---- E: generative Geometry Nodes --------------------------------
    e = make_plane("HealGeoNodeGenerative", (12.0, 0.0, 0.0))
    e_gn = e.modifiers.new("GeoGenerative", "NODES")
    e_gn.node_group = _set_position_and_subdivide_group()
    e_gn.show_viewport = False
    ctx.view_layer.update()
    bind_cache(e, client._needs_after_deformers("SHELL", e))
    e_bound = order(e)
    e_gn.show_viewport = True
    ctx.view_layer.update()
    e_stale_before = stale(e)

    log("bound")

    # The frame pump owns this in production; call it directly because a
    # timer body never runs inside the driver's single main-thread exec.
    heal_frame_start = float(enc.resolve_start_frame_or_default(ctx.scene))
    client.heal_mesh_caches_if_stale()
    ctx.view_layer.update()
    planes = (a, b, c, d, e)
    first_pass = {o.name: order(o) for o in planes}
    # The frame pump calls this about ten times a second, so a placement
    # rule that disagrees with the index the move actually produces would
    # rewrite the stack on every tick. A second call must change nothing.
    client.heal_mesh_caches_if_stale()
    ctx.view_layer.update()
    second_pass = {o.name: order(o) for o in planes}
    stale_after = {o.name: stale(o) for o in planes}
    log("healed")

    # Every bind used frame_start 1.0 against a scene starting at frame 1,
    # so no other term of the heal predicate can fire. Anything that moved
    # moved because of the placement check.
    bound_frame_start = a.modifiers[NAME].frame_start

    record(
        "A_deformer_enabled_after_bind_is_rehealed",
        a_bound[0] == NAME
        and a_stale_before is True
        and order(a) == ["Armature", NAME, "Subdivision"],
        {"order_at_bind": a_bound, "stale_before_heal": a_stale_before,
         "order_after_heal": order(a),
         "heal_frame_start": heal_frame_start,
         "bound_frame_start": bound_frame_start,
         "group_active_on_create": group_active},
    )
    record(
        "B_no_deformer_cache_stays_first",
        b_bound == [NAME, "Subdivision"]
        and b_stale_before is False
        and order(b) == [NAME, "Subdivision"],
        {"order_at_bind": b_bound, "stale_before_heal": b_stale_before,
         "order_after_heal": order(b)},
    )
    record(
        "C_already_placed_cache_untouched",
        c_bound == ["Armature", NAME, "Subdivision"]
        and c_stale_before is False
        and order(c) == ["Armature", NAME, "Subdivision"],
        {"order_at_bind": c_bound, "stale_before_heal": c_stale_before,
         "order_after_heal": order(c)},
    )
    record(
        "D_deform_only_geonodes_rehealed",
        d_bound[0] == NAME
        and d_stale_before is True
        and order(d) == ["GeoDeform", NAME],
        {"order_at_bind": d_bound, "stale_before_heal": d_stale_before,
         "order_after_heal": order(d)},
    )
    record(
        "E_generative_geonodes_keeps_cache_first",
        e_bound == [NAME, "GeoGenerative"]
        and e_stale_before is False
        and order(e) == [NAME, "GeoGenerative"],
        {"order_at_bind": e_bound, "stale_before_heal": e_stale_before,
         "order_after_heal": order(e)},
    )

    record(
        "F_heal_is_idempotent",
        first_pass == second_pass and not any(stale_after.values()),
        {"after_first_heal": first_pass, "after_second_heal": second_pass,
         "stale_after": stale_after},
    )

    log("checks=" + str(len(result["checks"])) + " done")
except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
