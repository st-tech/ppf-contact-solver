# File: scenarios/bl_hair_curves_weights.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Per-point weights on Hair Curves (issue #146): a pin whose pull fades
# along a strand, and a material that varies along it.
#
# A Hair Curves object has no vertex groups, so a weight is a point
# attribute. A pin stored as a FLOAT attribute holds every point with a
# weight above zero, and once it pulls, each point is pulled with the pin's
# strength times its own weight. A Spatial Material Map on the ROD group
# reads a point attribute and blends each rod segment from the group's
# slider toward the map's target by the mean weight of its two ends.
#
# A pull pin replaces its point's own inertia and gravity with its spring,
# so a pulled point stays at its groom unless the rest of the strand, or a
# contact, drags it off. The pulled strands here are therefore pulled over
# their base and left free over their tail: the tail falls, drags the base
# through the strand's own stiffness, and the base yields by its weight.
#
# The solve holds four identical strands side by side, so each feature is
# judged against a strand that does not have it:
#   strand 0  no weight anywhere: the reference
#   strand 1  bend map weight 1: stays straight where strand 0 curls
#   strand 2  base pulled at weight 1: yields least to its falling tail
#   strand 3  base pulled at weight 0.1: yields more than strand 2, and
#             less than the unheld base of strand 0
# The pull pin covers the roots of strands 2 and 3 as well, and the Roots
# pin is listed AFTER it, so the later pin takes those points back and they
# stay exactly where they started.
#
# The mapped strand is judged by its SHAPE and not by how far its tip falls.
# A rod has no bending term at a vertex that is held exactly, so every
# strand turns about the second point of its roots whatever its stiffness,
# and a stiff one turning as a bar drops its tip further than a limp one
# falling freely. What the stiffness decides is whether the strand bows.
#
# Coverage targets:
#   A. payload_carries_weights: each pulled point ships the pin's strength
#      and its own weight, the roots the later pin took back ship nothing,
#      and the map ships one weight per point.
#   B. session_varies_per_segment_and_per_point: the built session holds
#      the target bend on exactly the mapped strand's segments and one pull
#      per pulled point, strength times weight.
#   C. solve_reads_both: the mapped strand stays straight where the
#      reference bows, the pulled bases move, by less at the larger weight
#      and by less than the reference's unheld base, and every root stays
#      where it started.
#   D. later_pin_wins_a_shared_point: with Roots listed first instead, the
#      pull pin takes the roots, and they ship its strength and weight.
#   E. float_attribute_is_a_pin: the pin menu offers a float point
#      attribute and none of Blender's own, it holds the points above zero,
#      a weight above one reads as one, and a weight that is not a number
#      is refused by name at encode.
#   F. create_keeps_a_soft_selection: Create in Edit Mode stores a
#      selection with partly selected points as a weighted pin, and a float
#      selection of zeros and ones as a plain one.
#   G. scripted_weights: the scripting API stores weights beside indices,
#      and refuses a zero weight, a missing weight and weights without
#      indices, each by name.
#   H. exact_pin_holds_every_weighted_point: with Pull off a weighted pin
#      ships every point above zero and no weight.
#   I. map_sources_on_hair: a map reads an attribute a Geometry Nodes
#      modifier stores along each strand and a boolean attribute as 0 and
#      1; a vertex group source, a missing attribute, a map keyed over
#      time, a parameter a rod does not read and a legacy Curve in the
#      group are each refused by name; and a map added to a group that
#      holds only hair starts on the attribute source.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND: it solves, and the checks read the solve.
BACKENDS = ("real",)


_N_STRANDS = 4
_N_POINTS = 8
_SEGMENT = 0.05
# The pull of a point at weight one, and the lighter weight on strand 3.
_PULL_STRENGTH = 400.0
_LIGHT_WEIGHT = 0.1
# The mapped strand's bending stiffness, as a multiple of the group's own:
# enough that the strand stays a straight bar over the solve, where the
# group's own value leaves it to curl as it falls.
_BEND_FACTOR = 1.0e5
# Points of a pulled strand that carry a weight: its first segment, which the
# Roots pin takes back, and the three points after it. The rest are free.
_PULLED_POINTS = 5


_DRIVER_BODY = r"""
import glob
import os
import struct
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_ROOT = "<<PROJECT_ROOT>>"
N_STRANDS = <<N_STRANDS>>
N_POINTS = <<N_POINTS>>
SEGMENT = <<SEGMENT>>
PULLED_POINTS = <<PULLED_POINTS>>
PULL_STRENGTH = <<PULL_STRENGTH>>
LIGHT_WEIGHT = <<LIGHT_WEIGHT>>
BEND_FACTOR = <<BEND_FACTOR>>


def new_hair(name, sizes, x0=0.0, y0=0.0):
    curves = bpy.data.hair_curves.new(name)
    curves.add_curves(list(sizes))
    points = []
    for c, n in enumerate(sizes):
        for i in range(n):
            points.append((x0 + SEGMENT * i, y0, 0.1 * c))
    curves.attributes["position"].data.foreach_set(
        "vector", np.array(points, dtype=np.float32).ravel())
    obj = bpy.data.objects.new(name, curves)
    bpy.context.collection.objects.link(obj)
    return obj


def set_float(obj, name, values):
    attribute = obj.data.attributes.get(name)
    if attribute is None:
        attribute = obj.data.attributes.new(name, "FLOAT", "POINT")
    attribute.data.foreach_set("value", np.asarray(values, dtype=np.float32))
    obj.data.update_tag()


def get_float(obj, name):
    attribute = obj.data.attributes[name]
    values = np.empty(len(attribute.data), dtype=np.float32)
    attribute.data.foreach_get("value", values)
    return values


def refusal(call):
    # The message of the ValueError *call* raises, or "" when it raises none.
    try:
        call()
        return ""
    except ValueError as exc:
        return str(exc)


def read_floats(path):
    raw = open(path, "rb").read()
    return list(struct.unpack("<%df" % (len(raw) // 4), raw))


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    hair_mod = __import__(pkg + ".core.hair_curves", fromlist=["x"])
    pc2_mod = __import__(pkg + ".core.pc2", fromlist=["x"])
    client_mod = __import__(pkg + ".core.client", fromlist=["x"])
    groups_mod = __import__(pkg + ".models.groups", fromlist=["x"])
    uuid_mod = __import__(pkg + ".core.uuid_registry", fromlist=["x"])
    encoder_mesh = __import__(pkg + ".core.encoder.mesh", fromlist=["x"])
    encoder_pin = __import__(pkg + ".core.encoder.pin", fromlist=["x"])
    encoder_maps = __import__(pkg + ".core.encoder.material_maps",
                              fromlist=["x"])

    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    hair = new_hair("HairObj", [N_POINTS] * N_STRANDS)
    n_points = N_STRANDS * N_POINTS
    groom = hair_mod.local_positions(hair).copy()

    def strand(s):
        return list(range(s * N_POINTS, (s + 1) * N_POINTS))

    roots = [s * N_POINTS + i for s in range(N_STRANDS) for i in (0, 1)]
    tips = [s * N_POINTS + N_POINTS - 1 for s in range(N_STRANDS)]

    stiff = np.zeros(n_points, dtype=np.float32)
    stiff[strand(1)] = 1.0
    set_float(hair, "stiff", stiff)
    goal = np.zeros(n_points, dtype=np.float32)
    goal[strand(2)[:PULLED_POINTS]] = 1.0
    goal[strand(3)[:PULLED_POINTS]] = LIGHT_WEIGHT
    set_float(hair, "Goal", goal)
    # The last pulled point of each strand, where a falling tail drags most.
    bases = [s * N_POINTS + PULLED_POINTS - 1 for s in range(N_STRANDS)]
    light = float(np.float32(LIGHT_WEIGHT))

    blend_path = os.path.join(os.path.dirname(PROBE_DIR), "hair_weights.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    root = dh.configure_state(
        project_name="hair_curves_weights", frame_count=12,
        gravity=(0.0, 0.0, -9.8),
    )
    rod = dh.api.solver.create_group("Hair", "ROD")
    group = root.object_group_0
    rod.add(hair.name)
    hair_uuid = uuid_mod.get_or_create_object_uuid(hair)

    # The pull pin first and the roots after it: the later pin wins a point
    # two pins share, so the roots stay exact under a pull that covers them.
    rod.create_pin(hair.name, "Goal").pull(strength=PULL_STRENGTH)
    rod.create_pin(hair.name, hair_mod.ROOTS_PIN_NAME)

    base_bend = float(group.bend)
    target_bend = base_bend * BEND_FACTOR
    entry = group.material_maps.add()
    entry.parameter = "bend"
    entry.source_type = "ATTRIBUTE"
    entry.source_name = "stiff"
    entry.target_value = target_bend

    def pin_config():
        return encoder_pin._encode_pin_config(
            bpy.context, [group], root.state).get(hair_uuid, {})

    # ----- A: what the payload carries -----------------------------
    data_bytes, param_bytes = dh.encode_payload()
    cfg = pin_config()
    weighted = strand(2)[:PULLED_POINTS] + strand(3)[:PULLED_POINTS]
    pulled = [i for i in weighted if i not in roots]
    weights_sent = {i: cfg[i].get("pull_weight") for i in pulled if i in cfg}
    blob = dh.decode_addon_blob(param_bytes)
    sent_maps = {}
    for e in blob.get("group", []):
        if e[0].get("material-maps"):
            sent_maps = e[0]["material-maps"]
    bend_map = sent_maps.get("bend") or {}
    map_weights = list((bend_map.get("weights") or {}).get(hair_uuid, []))
    dh.record(
        "A_payload_carries_weights",
        sorted(cfg) == pulled
        and all(cfg[i].get("pull_strength") == PULL_STRENGTH for i in pulled)
        and all(weights_sent[i] == 1.0 for i in strand(2) if i in pulled)
        and all(weights_sent[i] == light for i in strand(3) if i in pulled)
        and map_weights == stiff.tolist()
        and abs(bend_map.get("target", 0.0) - target_bend) <= 1e-4 * target_bend,
        {"configured": sorted(cfg), "expected": pulled,
         "weights": {str(k): v for k, v in sorted(weights_sent.items())},
         "map_weights": map_weights, "target": bend_map.get("target"),
         "expected_target": target_bend},
    )

    # ----- build, and B: what the session holds --------------------
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, message="hair_curves_weights")
    dh.log("built")
    search_root = os.path.dirname(PROJECT_ROOT.rstrip("/")) or PROJECT_ROOT
    hits = glob.glob(os.path.join(search_root, "**", "session"), recursive=True)
    if not hits:
        raise RuntimeError("no session directory under %s" % search_root)
    session = sorted(hits, key=os.path.getmtime)[-1]
    bend_values = read_floats(
        os.path.join(session, "bin", "param", "rod-bend.bin"))
    stiff_segments = sum(
        1 for v in bend_values if abs(v - target_bend) <= 1e-4 * target_bend)
    base_segments = sum(
        1 for v in bend_values if abs(v - base_bend) <= 1e-4 * base_bend)
    pull_files = sorted(glob.glob(os.path.join(session, "bin", "pin-pullw-*.bin")))
    pulls = sorted(read_floats(pull_files[0])) if len(pull_files) == 1 else []
    per_strand = PULLED_POINTS - 2
    expected_pulls = sorted(
        [float(np.float32(PULL_STRENGTH * light))] * per_strand
        + [float(np.float32(PULL_STRENGTH))] * per_strand
    )
    dh.record(
        "B_session_varies_per_segment_and_per_point",
        len(bend_values) == N_STRANDS * (N_POINTS - 1)
        and stiff_segments == N_POINTS - 1
        and base_segments == (N_STRANDS - 1) * (N_POINTS - 1)
        and pulls == expected_pulls,
        {"segments": len(bend_values), "at_target": stiff_segments,
         "at_base": base_segments, "pull_files": len(pull_files),
         "pulls": pulls, "expected_pulls": expected_pulls},
    )

    # ----- run, and C: what the solve did with them ----------------
    dh.run_and_wait(timeout=300.0)
    run_state = dh.facade.engine.state
    if run_state.solver.name == "FAILED":
        raise RuntimeError(
            "the solve failed: "
            f"{run_state.server_error or run_state.error}"
        )
    dh.force_frame_query(expected_frames=1, timeout=15.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()
    dh.log("fetched")
    pc2_path = pc2_mod.get_pc2_path(client_mod.object_pc2_key(hair))
    pc2_arr = dh.read_pc2(pc2_path) if os.path.isfile(pc2_path) else None
    def bow_of(points):
        # How far the strand bows off the line from its second point, the
        # one it turns about, to its tip.
        start, end = points[1], points[-1]
        along = (end - start) / np.linalg.norm(end - start)
        off = points[1:] - start
        return float(np.max(np.linalg.norm(
            off - np.outer(off @ along, along), axis=1)))

    root_err = float("inf")
    drop = []
    bow = []
    moved = []
    if pc2_arr is not None and pc2_arr.shape[1] == n_points:
        root_err = float(np.max(np.abs(pc2_arr[:, roots, :] - groom[roots])))
        drop = [float(groom[t, 2] - pc2_arr[-1, t, 2]) for t in tips]
        bow = [
            bow_of(pc2_arr[-1, strand(s)].astype(np.float64))
            for s in range(N_STRANDS)
        ]
        moved = [
            float(np.linalg.norm(pc2_arr[-1, b] - groom[b])) for b in bases
        ]
    # Each comparison is against a strand that differs in one thing, by a
    # margin a run-to-run difference in the solve cannot close. A pulled
    # base that did not move at all would be an exact hold, not a pull.
    dh.record(
        "C_solve_reads_both",
        len(drop) == N_STRANDS and root_err <= 1.0e-5
        and drop[0] > 1.0e-3 and drop[1] > 1.0e-3
        and bow[0] > 1.0e-2
        and bow[1] < 0.1 * bow[0]
        and moved[2] > 1.0e-6
        and moved[2] < 0.5 * moved[3]
        and moved[3] < 0.5 * moved[0],
        {"max_root_error": root_err,
         "tip_drop": {"reference": drop[0], "bend_mapped": drop[1]}
         if len(drop) == N_STRANDS else drop,
         "bow": {"reference": bow[0], "bend_mapped": bow[1]}
         if len(bow) == N_STRANDS else bow,
         "base_moved": {"reference": moved[0], "pull_weight_1": moved[2],
                        "pull_weight_light": moved[3]}
         if len(moved) == N_STRANDS else moved,
         "pc2_shape": list(pc2_arr.shape) if pc2_arr is not None else None},
    )

    # ----- D: the later pin wins a shared point --------------------
    names = [p.name for p in group.pin_vertex_groups]
    group.pin_vertex_groups.move(1, 0)
    flipped = [p.name for p in group.pin_vertex_groups]
    cfg_flipped = pin_config()
    shared = [i for i in weighted if i in roots]
    dh.record(
        "D_later_pin_wins_a_shared_point",
        flipped == names[::-1]
        and sorted(cfg_flipped) == sorted(weighted)
        and all(cfg_flipped[i].get("pull_strength") == PULL_STRENGTH
                and cfg_flipped[i].get("pull_weight") in (1.0, light)
                for i in shared),
        {"order": flipped, "configured": sorted(cfg_flipped),
         "shared": shared},
    )
    group.pin_vertex_groups.move(1, 0)

    # ----- the rest reads no solve: a second object and group ------
    probe = new_hair("Probe", [5, 4, 6], y0=2.0)
    n_probe = 15
    probe_rod = dh.api.solver.create_group("Probe", "ROD")
    probe_group = root.object_group_1
    probe_rod.add(probe.name)
    probe_uuid = uuid_mod.get_or_create_object_uuid(probe)

    def probe_config():
        return encoder_pin._encode_pin_config(
            bpy.context, [probe_group], root.state).get(probe_uuid, {})

    def encode_error():
        return refusal(lambda: encoder_mesh.encode_obj(bpy.context))

    # ----- E: a float point attribute is a pin ---------------------
    soft = np.zeros(n_probe, dtype=np.float32)
    soft[[1, 2, 3, 4]] = [1.0, 0.5, 2.5, 0.25]
    soft[7] = -1.0
    set_float(probe, "soft", soft)
    set_float(probe, "radius", np.full(n_probe, 0.01, dtype=np.float32))
    menu = [i[0] for i in groups_mod.get_vertex_group_items(probe_group, None)]
    held = hair_mod.pin_indices(probe, "soft")
    held_weights = hair_mod.pin_weights(probe, "soft").tolist()
    probe_rod.create_pin(probe.name, "soft").pull(strength=4.0)
    sent = probe_config()
    soft[2] = np.nan
    set_float(probe, "soft", soft)
    not_a_number = encode_error()
    soft[2] = 0.5
    set_float(probe, "soft", soft)
    dh.record(
        "E_float_attribute_is_a_pin",
        f"[{probe.name}][soft]" in menu
        and f"[{probe.name}][radius]" not in menu
        and hair_mod.pin_is_weighted(probe, "soft")
        and not hair_mod.pin_is_weighted(probe, hair_mod.ROOTS_PIN_NAME)
        and hair_mod.pin_weights(probe, hair_mod.ROOTS_PIN_NAME) is None
        and held == [1, 2, 3, 4]
        and held_weights == [1.0, 0.5, 1.0, 0.25]
        and sorted(sent) == [1, 2, 3, 4]
        and [sent[i]["pull_weight"] for i in (1, 2, 3, 4)] == [1.0, 0.5, 1.0, 0.25]
        and all(sent[i]["pull_strength"] == 4.0 for i in sent)
        and "'soft'" in not_a_number and "point 2" in not_a_number
        and "not a finite number" in not_a_number
        and encode_error() == "",
        {"menu": menu, "held": held, "weights": held_weights,
         "sent": {str(k): v.get("pull_weight") for k, v in sorted(sent.items())},
         "not_a_number": not_a_number[:200], "clean": encode_error()[:200]},
    )

    # ----- F: Create keeps a soft selection ------------------------
    bpy.ops.object.select_all(action="DESELECT")
    probe.select_set(True)
    bpy.context.view_layer.objects.active = probe
    painted = np.zeros(n_probe, dtype=np.float32)
    painted[[10, 11, 12, 13, 14]] = [0.2, 0.4, 0.6, 0.8, 1.0]
    data = probe.data
    if data.attributes.get(".selection") is not None:
        data.attributes.remove(data.attributes[".selection"])
    data.attributes.new(".selection", "FLOAT", "POINT").data.foreach_set(
        "value", painted)
    bpy.ops.object.mode_set(mode="EDIT")
    created_soft = bpy.ops.object.create_pin_vertex_group(
        "EXEC_DEFAULT", group_index=1, vg_name="painted")
    soft_attribute = probe.data.attributes.get("painted")
    soft_type = soft_attribute.data_type if soft_attribute else None
    soft_values = get_float(probe, "painted").tolist() if soft_type == "FLOAT" else []
    # The same selection with every point fully in or out carries no weight.
    data = probe.data
    data.attributes.remove(data.attributes[".selection"])
    data.attributes.new(".selection", "FLOAT", "POINT").data.foreach_set(
        "value", (painted > 0.5).astype(np.float32))
    bpy.ops.object.mode_set(mode="EDIT")
    created_plain = bpy.ops.object.create_pin_vertex_group(
        "EXEC_DEFAULT", group_index=1, vg_name="picked")
    plain_attribute = probe.data.attributes.get("picked")
    plain_type = plain_attribute.data_type if plain_attribute else None
    pins_now = [p.name for p in probe_group.pin_vertex_groups]
    dh.record(
        "F_create_keeps_a_soft_selection",
        created_soft == {"FINISHED"} and created_plain == {"FINISHED"}
        and soft_type == "FLOAT"
        and np.allclose(soft_values, painted, atol=1e-7)
        and hair_mod.pin_indices(probe, "painted") == [10, 11, 12, 13, 14]
        and plain_type == "BOOLEAN"
        and hair_mod.pin_indices(probe, "picked") == [12, 13, 14]
        and f"[{probe.name}][painted]" in pins_now
        and f"[{probe.name}][picked]" in pins_now,
        {"soft_type": soft_type, "soft_values": soft_values,
         "plain_type": plain_type,
         "picked": hair_mod.pin_indices(probe, "picked"), "pins": pins_now},
    )

    # ----- G: weights through the scripting API --------------------
    probe_rod.create_pin(
        probe.name, "scripted", indices=[5, 6, 8], weights=[1.0, 0.5, 0.125])
    scripted_type = probe.data.attributes["scripted"].data_type
    scripted_weights = hair_mod.pin_weights(probe, "scripted").tolist()
    zero_weight = refusal(lambda: probe_rod.create_pin(
        probe.name, "zeroed", indices=[5, 6], weights=[1.0, 0.0]))
    short_weights = refusal(lambda: probe_rod.create_pin(
        probe.name, "short", indices=[5, 6], weights=[1.0]))
    no_indices = refusal(lambda: probe_rod.create_pin(
        probe.name, "soft", weights=[1.0]))
    dh.record(
        "G_scripted_weights",
        scripted_type == "FLOAT"
        and hair_mod.pin_indices(probe, "scripted") == [5, 6, 8]
        and scripted_weights == [1.0, 0.5, 0.125]
        and "point 6" in zero_weight and "weight of zero" in zero_weight
        and "zeroed" not in probe.data.attributes
        and "1 weights were given for 2" in short_weights
        and "short" not in probe.data.attributes
        and "no indices were given" in no_indices,
        {"type": scripted_type, "weights": scripted_weights,
         "zero": zero_weight[:160], "short": short_weights[:160],
         "no_indices": no_indices[:160]},
    )

    # ----- H: an exact pin holds every weighted point --------------
    by_name = {p.name: p for p in probe_group.pin_vertex_groups}
    scripted_pin = by_name[f"[{probe.name}][scripted]"]
    exact_cfg = probe_config()
    raw = dh.decode_addon_blob(encoder_mesh.encode_obj(bpy.context))
    shipped_pin = []
    for e in raw:
        for candidate in e.get("object", []):
            if candidate.get("name") == probe.name:
                shipped_pin = [int(i) for i in candidate.get("pin", [])]
    scripted_pin.use_pull = True
    scripted_pin.pull_strength = 3.0
    pulling_cfg = probe_config()
    scripted_pin.use_pull = False
    dh.record(
        "H_exact_pin_holds_every_weighted_point",
        not any(i in exact_cfg for i in (5, 6, 8))
        and all(i in shipped_pin for i in (5, 6, 8))
        and [pulling_cfg[i]["pull_weight"] for i in (5, 6, 8)] == [1.0, 0.5, 0.125],
        {"exact_cfg_keys": sorted(exact_cfg), "shipped_pin": sorted(set(shipped_pin)),
         "pulling": {str(i): pulling_cfg.get(i, {}).get("pull_weight")
                     for i in (5, 6, 8)}},
    )

    # ----- I: where a map on hair reads its weights ----------------
    def map_payload():
        payload, _schedules = encoder_maps.encode_material_maps(
            bpy.context, [probe_group], 100.0, 1)
        return payload.get(probe_group.uuid, {})

    def map_error():
        return refusal(map_payload)

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.ppf_add_material_map(slot=1)
    row = probe_group.material_maps[0]
    default_source = row.source_type
    row.parameter = "young-mod"
    row.target_value = float(probe_group.rod_young_modulus) * 4.0

    # An attribute a modifier stores along each strand, root to tip.
    tree = bpy.data.node_groups.new("StoreRamp", "GeometryNodeTree")
    tree.is_modifier = True
    tree.interface.new_socket(
        "Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(
        "Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    node_in = tree.nodes.new("NodeGroupInput")
    node_out = tree.nodes.new("NodeGroupOutput")
    store = tree.nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type = "FLOAT"
    store.domain = "POINT"
    store.inputs["Name"].default_value = "ramp"
    parameter = tree.nodes.new("GeometryNodeSplineParameter")
    tree.links.new(node_in.outputs[0], store.inputs["Geometry"])
    tree.links.new(parameter.outputs["Factor"], store.inputs["Value"])
    tree.links.new(store.outputs["Geometry"], node_out.inputs[0])
    modifier = probe.modifiers.new("Ramp", "NODES")
    modifier.node_group = tree
    row.source_name = "ramp"
    ramp = list((map_payload().get("young-mod") or {}).get(
        "weights", {}).get(probe_uuid, []))
    on_the_groom = "ramp" in probe.data.attributes

    row.source_name = "picked"
    as_boolean = list((map_payload().get("young-mod") or {}).get(
        "weights", {}).get(probe_uuid, []))

    row.source_name = "nowhere"
    missing = map_error()
    row.source_name = "ramp"
    row.source_type = "VERTEX_GROUP"
    vertex_group = map_error()
    row.source_type = "ATTRIBUTE"
    sample = row.samples.add()
    sample.frame = 5
    sample.source_type = "ATTRIBUTE"
    sample.source_name = "picked"
    keyed = map_error()
    row.samples.remove(0)
    row.parameter = "bend-warp"
    unread = map_error()
    row.parameter = "young-mod"

    legacy = bpy.data.curves.new("Legacy", "CURVE")
    legacy.dimensions = "3D"
    spline = legacy.splines.new("POLY")
    spline.points.add(3)
    for i, point in enumerate(spline.points):
        point.co = (0.1 * i, 4.0, 0.0, 1.0)
    legacy_obj = bpy.data.objects.new("Legacy", legacy)
    bpy.context.collection.objects.link(legacy_obj)
    probe_rod.add(legacy_obj.name)
    with_curve = map_error()
    probe_rod.remove(legacy_obj.name)
    clean = map_error()

    starts = [0, 5, 9]
    ends = [4, 8, 14]
    dh.record(
        "I_map_sources_on_hair",
        default_source == "ATTRIBUTE"
        and not on_the_groom and len(ramp) == n_probe
        and all(abs(ramp[i]) <= 1e-6 for i in starts)
        and all(abs(ramp[i] - 1.0) <= 1e-6 for i in ends)
        and all(ramp[i] < ramp[i + 1] for i in (0, 1, 2, 3, 5, 6, 7))
        and as_boolean == [1.0 if i in (12, 13, 14) else 0.0 for i in range(n_probe)]
        and "no attribute 'nowhere'" in missing and "'Probe'" in missing
        and "no vertex groups" in vertex_group and "Attribute" in vertex_group
        and "keyed over time" in keyed and "ROD" in keyed
        and "bend-warp" in unread and "not available as a map on a ROD" in unread
        and "'Legacy'" in with_curve and "is a Curve" in with_curve
        and clean == "",
        {"default_source": default_source, "ramp": ramp,
         "ramp_on_the_groom": on_the_groom, "as_boolean": as_boolean,
         "missing": missing[:200], "vertex_group": vertex_group[:200],
         "keyed": keyed[:200], "unread": unread[:200],
         "with_curve": with_curve[:200], "clean": clean[:200]},
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
        .replace("<<PROJECT_ROOT>>", ctx.project_root.replace("\\", "/"))
        .replace("<<N_STRANDS>>", str(_N_STRANDS))
        .replace("<<N_POINTS>>", str(_N_POINTS))
        .replace("<<SEGMENT>>", str(_SEGMENT))
        .replace("<<PULLED_POINTS>>", str(_PULLED_POINTS))
        .replace("<<PULL_STRENGTH>>", str(_PULL_STRENGTH))
        .replace("<<LIGHT_WEIGHT>>", str(_LIGHT_WEIGHT))
        .replace("<<BEND_FACTOR>>", str(_BEND_FACTOR))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
