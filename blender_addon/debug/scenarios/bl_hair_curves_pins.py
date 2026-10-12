# File: scenarios/bl_hair_curves_pins.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Hair Curves in the add-on, everything short of a solve (issue #146):
# which groups take a hair object, how its pins are made, named, renamed
# and selected, what the encoder refuses, where the playback modifier goes
# in a stack that generates children, and what the operators that cannot
# act on hair say instead. bl_hair_curves_run and bl_hair_curves_scalp
# cover the solve.
#
# A Hair Curves object has no vertex groups, so each of these paths has a
# branch of its own beside the mesh and the legacy-curve ones, and a branch
# nobody runs is a branch that breaks. None of them needs the solver.
#
# Coverage targets:
#   A. only_a_rod_group_takes_hair: a SHELL group leaves a hair object
#      unassigned and a ROD group takes it, through the UI operator and
#      through the scripting API alike.
#   B. roots_pin_is_derived: the pin menu offers Roots with no attribute
#      behind it, it holds the first segment of every curve, and it follows
#      the groom when a curve is added.
#   C. create_from_selection: Create in Edit Mode stores the selected
#      points as a boolean point attribute and adds the pin.
#   D. rename_and_select: Rename renames the attribute and the pin, Roots
#      refuses a rename by name, and Select and Deselect write exactly the
#      pin's points.
#   E. pin_by_indices: the scripting API pins points by index, and refuses
#      an index the object does not have.
#   F. encoder_refusals: a Bezier curve, a curve of one point and a pin
#      whose attribute is gone are each refused by name.
#   G. fur_stack_cut: on Blender's own Quick Fur stack the shipped object
#      is the guides, the cache is bound just in front of the modifier that
#      generates children, the children follow a played-back frame, and a
#      cache moved to the wrong place is put back.
#   H. operators_name_hair: Bake Animation refuses a hair object by name,
#      the export lists it as excluded, and removing an orphaned copy's
#      cache takes its playback attribute with it.
#   I. pinned_follow_is_refused_until_unpinned: Blender pins its Hair
#      Dynamics modifier to the end of the stack, so the cache sits in
#      front of it even on a stack that generates nothing. While the
#      surface only moves as a whole that is accepted; once the surface
#      changes shape the transfer is refused by name, and unpinning the
#      modifier and moving it in front of the child generator makes the
#      shipped pose follow the surface.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# Declared for the real backend like every scenario; it never solves.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import os
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def evaluated(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    deps.update()
    data = obj.evaluated_get(deps).data
    flat = np.empty(len(data.points) * 3, dtype=np.float32)
    data.attributes["position"].data.foreach_get("vector", flat)
    offsets = np.empty(len(data.curve_offset_data), dtype=np.int32)
    data.curve_offset_data.foreach_get("value", offsets)
    return flat.reshape(-1, 3), offsets


def new_hair(name, sizes, x0=0.0):
    curves = bpy.data.hair_curves.new(name)
    curves.add_curves(list(sizes))
    points = []
    for c, n in enumerate(sizes):
        for i in range(n):
            points.append((x0 + 0.1 * c, 0.0, 0.03 * i))
    curves.attributes["position"].data.foreach_set(
        "vector", np.array(points, dtype=np.float32).ravel())
    obj = bpy.data.objects.new(name, curves)
    bpy.context.collection.objects.link(obj)
    return obj


def op_error(call):
    # (ran, message): an operator that reports an ERROR raises from bpy.ops.
    try:
        return True, str(sorted(call()))
    except RuntimeError as exc:
        return False, str(exc)


def encode_error():
    try:
        encoder_mesh.encode_obj(bpy.context)
        return ""
    except ValueError as exc:
        return str(exc)


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    hair_mod = __import__(pkg + ".core.hair_curves", fromlist=["x"])
    pc2_mod = __import__(pkg + ".core.pc2", fromlist=["x"])
    client_mod = __import__(pkg + ".core.client", fromlist=["x"])
    groups_mod = __import__(pkg + ".models.groups", fromlist=["x"])
    encoder_mesh = __import__(pkg + ".core.encoder.mesh", fromlist=["x"])
    bake_mod = __import__(pkg + ".ui.dynamics.bake_ops", fromlist=["x"])
    export_mod = __import__(pkg + ".ui.dynamics.export_ops", fromlist=["x"])

    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    root = dh.configure_state(project_name="hair_curves_pins", frame_count=4)
    hair = new_hair("Hair", [5, 4, 6])
    n_points = 15
    first_segments = [0, 1, 5, 6, 9, 10]

    # ----- A: which groups take hair --------------------------------
    shell = dh.api.solver.create_group("Cloth", "SHELL")
    rod = dh.api.solver.create_group("Strands", "ROD")
    shell_group = root.object_group_0
    rod_group = root.object_group_1
    bpy.ops.object.select_all(action="DESELECT")
    hair.select_set(True)
    bpy.context.view_layer.objects.active = hair
    bpy.ops.object.add_objects_to_group(group_index=0)
    in_shell = len(shell_group.assigned_objects)
    bpy.ops.object.add_objects_to_group(group_index=1)
    in_rod = len(rod_group.assigned_objects)
    scripted = new_hair("Scripted", [3, 3], x0=1.0)
    rod.add(scripted.name)
    dh.record(
        "A_only_a_rod_group_takes_hair",
        in_shell == 0 and in_rod == 1
        and [a.name for a in rod_group.assigned_objects] == ["Hair", "Scripted"],
        {"in_shell": in_shell, "in_rod": in_rod,
         "rod_members": [a.name for a in rod_group.assigned_objects]},
    )

    # ----- B: the derived Roots pin --------------------------------
    items = [i[0] for i in groups_mod.get_vertex_group_items(rod_group, None)]
    roots_item = f"[{hair.name}][{hair_mod.ROOTS_PIN_NAME}]"
    derived = hair_mod.pin_indices(hair, hair_mod.ROOTS_PIN_NAME)
    rod_group.pin_vertex_group_items = roots_item
    bpy.ops.object.add_pin_vertex_group("EXEC_DEFAULT", group_index=1)
    pins_after_add = [p.name for p in rod_group.pin_vertex_groups]
    # A curve added to the groom is rooted too, with no pin edited.
    hair.data.add_curves([3])
    grown = hair_mod.pin_indices(hair, hair_mod.ROOTS_PIN_NAME)
    hair.data.remove_curves(indices=[3])
    dh.record(
        "B_roots_pin_is_derived",
        roots_item in items
        and hair_mod.ROOTS_PIN_NAME not in hair.data.attributes
        and derived == first_segments
        and pins_after_add == [roots_item]
        and grown == first_segments + [15, 16]
        and hair_mod.pin_indices(hair, hair_mod.ROOTS_PIN_NAME) == first_segments,
        {"menu": items, "derived": derived, "pins": pins_after_add,
         "after_adding_a_curve": grown},
    )

    # ----- C: Create from a selection ------------------------------
    tips = [4, 8, 14]
    bpy.ops.object.select_all(action="DESELECT")
    hair.select_set(True)
    bpy.context.view_layer.objects.active = hair
    bpy.ops.object.mode_set(mode="EDIT")
    edit_mode = bpy.context.mode
    hair_mod.set_point_selection(hair, range(n_points), False)
    hair_mod.set_point_selection(hair, tips, True)
    created = bpy.ops.object.create_pin_vertex_group(
        "EXEC_DEFAULT", group_index=1, vg_name="tips")
    attribute = hair.data.attributes.get("tips")
    dh.record(
        "C_create_from_selection",
        edit_mode == "EDIT_CURVES" and created == {"FINISHED"}
        and attribute is not None and attribute.domain == "POINT"
        and attribute.data_type == "BOOLEAN"
        and hair_mod.pin_indices(hair, "tips") == tips
        and f"[{hair.name}][tips]" in [p.name for p in rod_group.pin_vertex_groups],
        {"mode": edit_mode, "result": sorted(created),
         "stored": hair_mod.pin_indices(hair, "tips"),
         "pins": [p.name for p in rod_group.pin_vertex_groups]},
    )

    # ----- D: rename and select ------------------------------------
    names = [p.name for p in rod_group.pin_vertex_groups]
    rod_group.pin_vertex_groups_index = names.index(f"[{hair.name}][tips]")
    hair_mod.set_point_selection(hair, range(n_points), False)
    bpy.ops.object.select_pin_vertices("EXEC_DEFAULT", group_index=1)
    selected = np.flatnonzero(hair_mod.selected_point_mask(hair)).tolist()
    hair_mod.set_point_selection(hair, range(n_points), True)
    bpy.ops.object.deselect_pin_vertices("EXEC_DEFAULT", group_index=1)
    remaining = np.flatnonzero(hair_mod.selected_point_mask(hair)).tolist()
    bpy.ops.object.mode_set(mode="OBJECT")
    renamed = bpy.ops.object.rename_pin_vertex_group(
        "EXEC_DEFAULT", group_index=1, new_name="ends")
    names_after = [p.name for p in rod_group.pin_vertex_groups]
    rod_group.pin_vertex_groups_index = names_after.index(roots_item)
    roots_ran, roots_message = op_error(
        lambda: bpy.ops.object.rename_pin_vertex_group(
            "EXEC_DEFAULT", group_index=1, new_name="follicles"))
    dh.record(
        "D_rename_and_select",
        selected == tips
        and remaining == [i for i in range(n_points) if i not in tips]
        and renamed == {"FINISHED"}
        and "ends" in hair.data.attributes
        and "tips" not in hair.data.attributes
        and f"[{hair.name}][ends]" in names_after
        and hair_mod.pin_indices(hair, "ends") == tips
        and not roots_ran and "derived" in roots_message
        and roots_item in [p.name for p in rod_group.pin_vertex_groups],
        {"selected": selected, "remaining": remaining,
         "pins": names_after, "roots_rename": roots_message[:200]},
    )

    # ----- E: pins by index, through the scripting API -------------
    rod.create_pin(hair.name, "mid", indices=[2, 7, 11])
    try:
        rod.create_pin(hair.name, "beyond", indices=[2, 99])
        beyond_message = ""
    except ValueError as exc:
        beyond_message = str(exc)
    dh.record(
        "E_pin_by_indices",
        hair_mod.pin_indices(hair, "mid") == [2, 7, 11]
        and f"[{hair.name}][mid]" in [p.name for p in rod_group.pin_vertex_groups]
        and "99" in beyond_message and "does not exist" in beyond_message
        and "beyond" not in hair.data.attributes,
        {"mid": hair_mod.pin_indices(hair, "mid"),
         "beyond": beyond_message[:200]},
    )

    # ----- F: what the encoder refuses -----------------------------
    clean = encode_error()
    hair.data.set_types(type="BEZIER")
    bezier = encode_error()
    hair.data.set_types(type="CATMULL_ROM")
    stub = new_hair("Stub", [4, 1], x0=2.0)
    rod.add(stub.name)
    short = encode_error()
    rod.remove(stub.name)
    hair.data.attributes.remove(hair.data.attributes["mid"])
    gone = encode_error()
    names = [p.name for p in rod_group.pin_vertex_groups]
    rod_group.pin_vertex_groups_index = names.index(f"[{hair.name}][mid]")
    bpy.ops.object.remove_pin_vertex_group("EXEC_DEFAULT", group_index=1)
    restored = encode_error()
    dh.record(
        "F_encoder_refusals",
        clean == "" and restored == ""
        and "Bezier" in bezier and "'Hair'" in bezier
        and "'Stub'" in short and "curve 1 has 1 point" in short
        and "mid" in gone and "no longer exists" in gone,
        {"clean": clean[:160], "bezier": bezier[:200], "short": short[:200],
         "gone": gone[:200], "restored": restored[:160]},
    )

    # ----- G: a stack that generates children ----------------------
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(5.0, 0.0, 0.0))
    cube = bpy.context.active_object
    bpy.ops.object.quick_fur()
    fur = next(o for o in bpy.data.objects
               if o.type == "CURVES" and o.data.surface == cube)
    stack = [m.name for m in fur.modifiers]
    cut = hair_mod.cache_insertion_index(fur)
    n_guides = len(fur.data.points)
    children_before, offsets_before = evaluated(fur)
    fur_group_api = dh.api.solver.create_group("Fur", "ROD")
    fur_group = root.object_group_2
    bpy.ops.object.select_all(action="DESELECT")
    fur.select_set(True)
    bpy.context.view_layer.objects.active = fur
    bpy.ops.object.add_objects_to_group(group_index=2)
    fur_pins = [p.name for p in fur_group.pin_vertex_groups]
    decoded = dh.decode_addon_blob(encoder_mesh.encode_obj(bpy.context))
    shipped = None
    for entry in decoded:
        for candidate in entry.get("object", []):
            if candidate.get("name") == fur.name:
                shipped = candidate
    # One played-back frame, written the way a fetch writes it: every
    # guide sheared so its tip is 0.5 higher and its root where it was.
    guides = hair_mod.local_positions(fur).copy()
    offsets = hair_mod.curve_offsets(fur.data)
    sheared = guides.copy()
    for c in range(len(offsets) - 1):
        count = int(offsets[c + 1] - offsets[c])
        sheared[offsets[c]:offsets[c + 1], 2] += (
            0.5 * np.arange(count) / (count - 1))
    start_frame = int(bpy.context.scene.frame_start)
    client_mod._write_mesh_frame_to_pc2(fur, sheared, start_frame, start_frame)
    bpy.context.scene.frame_set(start_frame + 1)
    bpy.context.scene.frame_set(start_frame)
    bound = [m.name for m in fur.modifiers]
    children_after, offsets_after = evaluated(fur)
    tips_before = children_before[offsets_before[1:] - 1]
    tips_after = children_after[offsets_after[1:] - 1]
    mean_tip_rise = float((tips_after - tips_before)[:, 2].mean())
    # A cache dragged behind the modifier that generates children no
    # longer feeds it; the heal puts it back. It is dragged to just in
    # front of Hair Dynamics, which Blender pins to the end of the stack
    # and lets nothing pass.
    fur.modifiers.move(bound.index(pc2_mod.MODIFIER_NAME), len(bound) - 2)
    dragged = [m.name for m in fur.modifiers]
    misplaced = hair_mod.cache_modifier_is_bound(fur)
    client_mod.heal_mesh_caches_if_stale()
    healed = [m.name for m in fur.modifiers]
    fur_facts = {
        "cut_is_the_child_generator":
            cut < len(stack) and "Interpolate" in stack[cut],
        "ships_the_guides":
            shipped is not None and len(shipped.get("vert", [])) == n_guides,
        "roots_pinned_on_joining":
            fur_pins == [f"[{fur.name}][{hair_mod.ROOTS_PIN_NAME}]"],
        "cache_bound_at_the_cut":
            bound.index(pc2_mod.MODIFIER_NAME) == cut
            and bound[cut + 1] == stack[cut],
        "children_still_generated":
            len(offsets_after) == len(offsets_before)
            and len(offsets_after) - 1 > 10 * (len(offsets) - 1),
        "children_follow_the_frame": mean_tip_rise > 0.2,
        "groom_untouched":
            bool(np.array_equal(hair_mod.local_positions(fur), guides)),
        "misplaced_cache_is_seen": dragged != bound and not misplaced,
        "heal_puts_it_back": healed == bound,
    }
    dh.record(
        "G_fur_stack_cut",
        all(fur_facts.values()),
        {
            "facts": fur_facts,
            "stack": stack, "cut": cut, "guides": n_guides,
            "shipped": len(shipped.get("vert", [])) if shipped else None,
            "pins": fur_pins, "bound": bound, "healed": healed,
            "children": int(len(offsets_after) - 1),
            "mean_tip_rise": mean_tip_rise,
        },
    )

    # ----- H: operators that cannot act on hair say so -------------
    fur_group.assigned_objects_index = 0
    baked, bake_message = op_error(
        lambda: bpy.ops.object.bake_animation("EXEC_DEFAULT", group_index=2))
    left_on_cache = bake_mod._hair_curves_left_on_cache(bpy.context)
    excluded = export_mod._excluded_sim_curves(bpy.context)
    # A copy inherits the modifier and the attribute but belongs to no
    # group; removing its cache must leave it showing its own groom.
    copy = fur.copy()
    copy.data = fur.data.copy()
    bpy.context.collection.objects.link(copy)
    uuid_mod = __import__(pkg + ".core.uuid_registry", fromlist=["x"])
    if "_solver_uuid" in copy:
        del copy["_solver_uuid"]
    orphans = [o.name for o in pc2_mod.find_orphan_solver_caches()]
    removed = pc2_mod.remove_orphan_solver_cache(copy)
    dh.record(
        "H_operators_name_hair",
        not baked and "Hair Curves" in bake_message and fur.name in bake_message
        and fur.name in left_on_cache and fur.name in excluded
        and copy.name in orphans and removed
        and copy.modifiers.get(pc2_mod.MODIFIER_NAME) is None
        and hair_mod.SIM_POSITION_ATTRIBUTE not in copy.data.attributes
        and fur.modifiers.get(pc2_mod.MODIFIER_NAME) is not None
        and hair_mod.SIM_POSITION_ATTRIBUTE in fur.data.attributes
        and pc2_mod.has_mesh_cache(fur),
        {
            "bake": bake_message[:200], "left_on_cache": left_on_cache,
            "excluded_from_export": excluded, "orphans": orphans,
            "copy_attribute_left":
                hair_mod.SIM_POSITION_ATTRIBUTE in copy.data.attributes,
            "original_still_bound":
                fur.modifiers.get(pc2_mod.MODIFIER_NAME) is not None,
        },
    )

    # ----- I: a modifier pinned behind the cache -------------------
    # An Empty Hair object carries Hair Dynamics alone, pinned to last.
    bpy.ops.object.select_all(action="DESELECT")
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.curves_empty_hair_add()
    empty = bpy.context.view_layer.objects.active
    empty.data.add_curves([4])
    empty_stack = [(m.name, bool(m.use_pin_to_last)) for m in empty.modifiers]
    empty_cut = hair_mod.cache_insertion_index(empty)
    hair_mod.setup_cache_modifier(empty)
    empty_bound = [m.name for m in empty.modifiers]
    empty_ok = hair_mod.cache_modifier_is_bound(empty)
    hair_mod.clear_playback(empty)
    bpy.data.objects.remove(empty)

    rigid_surface = encode_error()
    # The surface now changes shape: a keyed shape key lifts its top.
    cube.shape_key_add(name="Basis")
    lifted = cube.shape_key_add(name="Lifted")
    for v in cube.data.vertices:
        if v.co.z > 0.0:
            lifted.data[v.index].co.z += 0.5
    lifted.value = 1.0
    lifted.keyframe_insert("value", frame=start_frame)
    lifted.value = 0.0
    lifted.keyframe_insert("value", frame=start_frame + 10)
    bpy.context.scene.frame_set(start_frame)
    pinned_refusal = encode_error()
    # Unpinned and moved in front of the child generator, the modifier is
    # part of the pose the solver starts from.
    dynamics = next(m for m in fur.modifiers if m.use_pin_to_last)
    dynamics.use_pin_to_last = False
    names = [m.name for m in fur.modifiers]
    fur.modifiers.move(names.index(dynamics.name),
                       names.index(pc2_mod.MODIFIER_NAME))
    client_mod.heal_mesh_caches_if_stale()
    reordered = [m.name for m in fur.modifiers]
    unpinned = encode_error()
    followed = None
    if not unpinned:
        for entry in dh.decode_addon_blob(encoder_mesh.encode_obj(bpy.context)):
            for candidate in entry.get("object", []):
                if candidate.get("name") == fur.name:
                    followed = np.array(candidate["vert"], dtype=np.float32)
    rise = (
        float((followed[:, 2] - guides[:, 2]).max())
        if followed is not None else 0.0
    )
    pinned_facts = {
        "lone_pinned_modifier_is_the_boundary":
            empty_stack == [(empty_stack[0][0], True)] and empty_cut == 0
            and empty_bound[0] == pc2_mod.MODIFIER_NAME and empty_ok,
        "accepted_while_the_surface_is_rigid": rigid_surface == "",
        "refused_once_the_surface_deforms":
            dynamics.name in pinned_refusal
            and "Pin to Last" in pinned_refusal
            and f"'{cube.name}'" in pinned_refusal,
        "cache_follows_the_unpinned_modifier":
            reordered.index(dynamics.name)
            < reordered.index(pc2_mod.MODIFIER_NAME)
            < reordered.index(stack[cut])
            and hair_mod.cache_modifier_is_bound(fur),
        "accepted_once_unpinned": unpinned == "",
        "shipped_pose_follows_the_surface": abs(rise - 0.5) < 1.0e-3,
    }
    dh.record(
        "I_pinned_follow_is_refused_until_unpinned",
        all(pinned_facts.values()),
        {
            "facts": pinned_facts, "empty_stack": empty_stack,
            "empty_bound": empty_bound,
            "refusal": pinned_refusal[:400], "reordered": reordered,
            "after_unpinning": unpinned[:200], "largest_rise": rise,
        },
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
