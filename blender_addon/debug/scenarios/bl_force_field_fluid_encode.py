# File: scenarios/bl_force_field_fluid_encode.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The add-on's Fluid Flow ENCODE (issue #155): what a Fluid
# Flow field's Domain Object, a baked Mantaflow gas domain or a Volume object
# read through openvdb, becomes in the PARAM payload, held to closed forms
# written from the source itself (the domain's own evaluated grid, the values
# written into the VDB file, the object's keyframes), never from the add-on's
# own arithmetic. Everything here is the add-on's half; no solve runs.
#
# Clocks: the scene plays at 24 fps with Use Scene FPS on and Time Scale 1,
# so the rate a fluid's velocity is converted by (`fluid_rate`, solver fps /
# scene fps) is 1 unless a check changes it. The solve reads frames 3..12.
#
# Subtests, one recorded check per name. Every tolerance is followed by the
# margin measured on dev-head (Blender 5.2); a check with no tolerance
# compares exactly.
#
#   A. Replay refusal and baked determinism, through the add-on's own hash
#      path (`compute_param_hash`, `encode_param_with_hash`):
#      A_baked_domain_hash_is_deterministic: a domain baked over the solve
#        frames hashes alike read fresh, read fresh again after the timeline is
#        played and the playhead parked on another frame, and reused from the
#        encode cache, and the hash is the SHA-256 of the encoded bytes. Each
#        fresh read walks all 10 solve frames, the reused one none (fluid reads
#        are counted by patching the reader the encode calls), the playhead is
#        where it was left, and the payload moves (frames 2 and 6 differ by 40%
#        of the peak), so the hashes compare something.
#      A_replay_domain_refused_by_name: a Replay domain simulated up to the
#        solve is refused naming the field, the domain and the remedy (were it
#        admitted, the details record whether a read-back after a walk hashes
#        like the live walk did).
#      A_unbaked_domain_refused_by_name, A_bake_range_refused_by_name: a freed
#        bake (FFEFreed, a domain of its own), and a bake over 2..13 against a
#        solve of 3..14 and of 1..10.
#   B. Every source refusal by name. The message names the field ("Force field
#      'FFEFlow'"), the object it read (a Volume's file named with its Volume,
#      "its Volume 'FFEV0''s file"), the grid at fault where there is one, and
#      the frame for an error met while reading. One check
#      per case: no Domain Object; an object that is neither a domain nor a
#      Volume; Noise with Apply Density; a modifier after the Fluid modifier
#      (disabled, it is no longer the reason); Noise Amount, Wind Factor and
#      Absorption on a Fluid Flow, which Blender applies there, each with the
#      way to reach a setting the Fluid Flow panel does not show (set the
#      type to Force and back); Gravitation on a Fluid Flow, which Blender
#      ignores there (I below), sent off rather than refused
#      (B_gravitation_on_fluid_flow_is_sent_off; B_a_valid_volume_encodes is
#      the baseline, and it sends no gravitation either); Apply Density on a
#      velocity-only file; a
#      Volume Displace and a Geometry Nodes modifier on the Volume; a Volume
#      sheared by a non-uniformly scaled parent, and one scaled non-uniformly
#      across voxels the file rotates (the same file scaled uniformly is
#      admitted); density on other voxels than the velocity; an empty file
#      path; a float grid named as the velocity; a Clip sequence one frame
#      short; a Volume reading a domain's cache folder; a Mantaflow cache file
#      copied elsewhere; a domain sheared by its parent.
#   C. Volume sources honored, against the values written into the file:
#      C_staggered_grid_is_destaggered, C_staggered_split_grids: a field
#        linear in the index, u(x) = A x + b with off-diagonal A, written on the
#        faces (component c of voxel i at i - 1/2 along c, class staggered), as
#        one vector grid and as three float grids, reads A i + b at every voxel
#        whose upper faces are in the file. Tol 1e-5 absolute on values up to
#        4.6; measured 1.9e-7. Reading the faces as centers is off by
#        |A_cc| / 2 >= 0.25.
#      C_split_component_grids: vel.x, vel.y, vel.z with different active
#        voxels read per component over their union. Tol 1e-6; measured 0.
#      C_named_velocity_grid: Velocity Grid set to a second vector grid reads
#        that grid, not 'velocity'. Tol 1e-6; measured 0.
#      C_velocity_unit_and_scale: a new Volume's default Velocity Unit FRAME
#        reads the file times the scene fps (24), SECOND as it is, and Velocity
#        Scale multiplies either. Rel tol 1e-6; measured 0 (these factors are
#        exact on the file's values).
#      C_rate_factor_on_volume: Time Scale 0.5 halves U, an explicit Frame Rate
#        48 doubles it (48 / 24), FRAME under Frame Rate 48 is the file times 48
#        (per frame times the solver fps), and the two together cancel; at
#        29.97 (FPS 30, Base 1.001) under Use Scene FPS, SECOND is the file
#        times 1.001 (the solver runs at the whole-number 30) and FRAME the
#        file times 30. Rel tol 1e-6; measured 0, and 8.6e-8 for the factor
#        1.001, which float32 does not hold exactly (Base ignored misses by
#        1.0e-3).
#      C_empty_first_frame: a sequence whose first file holds an empty
#        velocity grid sends that frame as resolution 0 with no table and no
#        bricks, and the fluid there is absent: presence 0 at the voxel the
#        index origin names and at the voxels the next file fills, which read
#        presence 1 and the file's value at the next frame (tol 1e-6; measured
#        4.4e-16).
#   D. Mantaflow domains honored, against the domain's own evaluated grid:
#      D_noncubic_<case>_cells_and_box: a 4 x 1.3 x 2 domain (16 x 5 x 8
#        cells) and a 2 x 2 x 2.2 one (15 x 15 x 16), both rotated: the cell is
#        extent_i / n_i along each axis and Blender's own cell_size times the
#        object's scale (rel tol 1e-6; measured at most 2.5e-8 and 4.0e-8,
#        Blender's float32 matrix and cell_size), and corner + axes (n * cell)
#        spans the box (tol 1e-9; measured 0, the same float32 matrix in
#        double).
#      D_noncubic_<case>_presence_at_faces: a point 3 cm inside each of the
#        six faces reads presence 1, 3 cm outside 0. The details count the
#        points one cubic cell (the longest axis's) would misjudge, at least one
#        or the case does not discriminate: the first case's cubic cells end
#        5 cm short of the 1.3 m face, the second's run 6.25 cm past two faces.
#      D_noncubic_<case>_values: the payload is the grid times 2.5 cells of the
#        longest axis per unit, in the domain's axes. Rel tol 1e-6; measured 0
#        (the units, 0.625 and 0.34375, are short binary fractions).
#      D_rate_factor_on_domain: Time Scale 0.5 halves U, an explicit Frame Rate
#        48 doubles it, and 29.97 (FPS 30, Base 1.001) multiplies it by 1.001,
#        against the grid. Rel tol 1e-6; measured 0, and 9.5e-8 at 29.97
#        (Base ignored misses by 1.0e-3).
#      D_domain_time_scale_on_U: the domain's own Time Scale 2, re-baked,
#        doubles U against that bake's grid. Rel tol 1e-6; measured 0.
#      D_moving_domain_per_frame_corner_and_axes: a domain keyed LINEAR in
#        location and rotation is placed at each solve frame by that frame's
#        transform, against the keys' interpolation in double. Tol 1e-6;
#        measured 8.3e-8 (corner) and 3.6e-8 (axes), against a one-frame slip
#        of 8.8 cm. The source moves, so both encodes read every frame.
#   E. A sparse grid with a large active box (801 x 401 x 41 voxels, 13 active
#      in 12 bricks), read without densifying it:
#      E_sparse_box_read_in_bricks_memory_bound,
#      E_sparse_box_read_in_rows_memory_bound (the cost model forced to rows):
#        the tracemalloc peak of the whole 10-frame encode stays under a tenth
#        of the box's float32 four-channel image, 21.1 MB; measured 2.5 MB by
#        bricks and 5.3 MB by rows, where reading the whole box at once peaks
#        at 476 MB. Each frame keeps exactly the 12 bricks the active voxels
#        fall in, over the full table, and reads their values (tol 1e-6;
#        measured 0); the two paths send the same bytes.
#   F. Chunks, the size report and the encode cache, on a 64 x 48 x 48 volume
#      whose 181 occupied bricks (1.48 MB a frame) exceed one chunk:
#      F_payload_bytes_are_chunked: every table and brick string is a list of
#        `bytes`, each at most 1 MiB, ceil(size / 1 MiB) of them, and joined
#        they read the file's values (rel tol 1e-6; measured 0).
#      F_size_report: volume_report() names the field and the source and
#        counts frames, the largest resolution, kept bricks, table slots,
#        occupancy and bytes exactly as the payload holds them; its one-line
#        form names the field, the source, the frames and the resolution.
#      F_encode_cache_reuse: a second encode reads no frame, sends the same
#        volume and reports it reused, and so does the Run check's hash. A
#        rewritten file, a changed Velocity Scale and a changed Time Scale each
#        read every frame again and send the new values (rel tol 1e-6;
#        measured 0); a source that moves is never reused.
#      F_a_renamed_velocity_grid_is_reused: a Volume whose grids were never
#        loaded names the default Velocity Grid "velocity", and the first read
#        renames it to the file's "vel" (measured); the next encode still
#        reads no frame.
#   G. G_volume_per_axis_cells_and_presence: a rotated Volume scaled (1, 2,
#      0.5) over axis-aligned voxels is honored: cells (0.2, 0.4, 0.1) (rel tol
#      1e-6; measured 7.3e-9), the corner half a voxel below the first active
#      voxel (tol 1e-9; measured 0), the velocity turned and scaled by the
#      object (rel tol 1e-5; measured 2.0e-7, the float32 matrix's departure
#      from orthogonality), and presence 1 a sixth of a voxel inside each
#      face, 0 a sixth outside.
#   H. What was baked, the Cache Offset, and a domain the view layer skips,
#      on a gas domain baked over 1..12 whose jet is not its child:
#      H_a_domain_the_view_layer_skips_is_refused_by_cause: its Fluid
#        modifier off in the viewport, the domain disabled in viewports, its
#        collection disabled in viewports and its collection excluded each
#        read a (0, 0, 0) grid (measured), and each is refused by its cause,
#        in Transfer and in the listing (`refusal`), with the encode cache
#        warm from a read while it was visible, and the status listing's
#        fluid entry adds no bake advice of its own (problem and note None,
#        since the empty grid says nothing about the bake); hidden by the eye
#        icon and in renders it reads exactly what it reads visible. FFEDet
#        disabled in viewports is not refused and reads the same as visible:
#        its parented jet, visible, keeps it evaluated (measured).
#      H_cache_offset_is_honored: Cache Offset 3 plays cache frames 1..12 at
#        scene frames 4..15, so the solve's 3..12 is refused naming the frames
#        it plays and the frames to bake, and a solve over 4..13 sends, at
#        each of its frames, the grid Blender plays there (rel tol 1e-6;
#        measured 2.7e-8, float32); the Visualize key names the files of the
#        cache frame a scene frame plays.
#      H_a_frame_missing_from_the_bake_is_refused_by_name: End raised from 8
#        to 12 after a bake over 1..8 (FFEBake8, a domain baked once), and then
#        frame 5's file removed, each
#        refused naming the first missing frame and its file. Without the
#        check on the files, the first is refused only as an empty grid met
#        at frame 9, and the second is not seen at all.
#   I. I_blender_ignores_gravitation_on_a_fluid_flow: nine particles in
#      FFEDet's jet, pushed by the Fluid Flow alone from frame 3 to 8, end
#      exactly where they end with Gravitation on (measured 0 of a 0.025
#      move), while Noise Amount 2 at seed 7 moves them 0.014: the
#      measurement that sending Gravitation off rests on, so the add-on
#      matches Blender.
#   J. J_the_cap_counts_the_whole_upload, with the 4 GiB cap lowered in
#      process: a reused volume beside a Texture image is refused by the
#      encode once the two pass the cap, naming each, and so is a fresh read;
#      under it they encode; the upload then refused by `prepare_upload` and
#      by Transfer's encode stage once the scene geometry and parameters
#      together pass it, naming the total exactly and both payloads, and
#      sent at exactly the cap.
#   K. K_the_volume_report_is_cleared_when_a_file_opens: the size report of
#      the last encode is empty once another file is opened. Last, since it
#      opens one.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
BACKENDS = ("real",)

_DRIVER_BODY = r"""
import math
import os
import shutil
import time
import tracemalloc
import traceback

import numpy as np
import openvdb
from mathutils import Vector

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

FC = 10          # solve frames
START = 3        # the first solve frame; the solve reads frames 3..12
SCENE_FPS = 24
MIB = 1 << 20
EMPTY = 0xFFFFFFFF
FIELD = "Force field 'FFEFlow'"

dh = DriverHelpers(pkg, result)
ff = __import__(pkg + ".core.force_field", fromlist=["encode"])
ff_status = __import__(pkg + ".ui.dynamics.force_field_status", fromlist=["fluid_info"])
params_mod = __import__(pkg + ".core.encoder.params", fromlist=["_build_param_dict"])
digest = __import__(pkg + ".core.encoder.cbor_encode", fromlist=["payload_digest"]).payload_digest
fe = __import__(pkg + ".core.field_eval", fromlist=["fluid_at"])
scene = bpy.context.scene


# ----- helpers ---------------------------------------------------------------

def solver(v):
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


def build():
    return params_mod._build_param_dict(bpy.context)


def fluid_entry(tree=None):
    tree = build() if tree is None else tree
    found = [f for f in (tree.get("force_field") or {}).get("fields", [])
             if f["type"] == "fluid"]
    if len(found) != 1:
        raise RuntimeError("expected one Fluid Flow in the payload, found %d" % len(found))
    return found[0]["fluid"]


def refused(*fragments):
    # (ok, message, missing fragments): ok when the encode raises ValueError
    # holding every fragment. Any other exception is not a refusal by name
    # (Transfer would report it as an encoding failure), so it fails the
    # check without ending the section.
    try:
        build()
    except ValueError as e:
        msg = str(e)
        missing = [f for f in fragments if f not in msg]
        return not missing, msg, missing
    except Exception as e:
        return False, "not a ValueError: {}: {}".format(type(e).__name__, e), list(fragments)
    return False, "no error", list(fragments)


def record_refusal(name, *fragments):
    ok, msg, missing = refused(*fragments)
    dh.record(name, ok, {"error": msg, "missing": missing})
    return ok


def payload_volume(v):
    # The payload's volume as field_eval reads it: chunks joined.
    return {"resolution": [tuple(int(x) for x in n) for n in v["resolution"]],
            "cell": np.asarray(v["cell"], dtype=np.float64),
            "corner": np.asarray(v["corner"], dtype=np.float64),
            "axes": np.asarray(v["axes"], dtype=np.float64),
            "start": v["start"], "step": v["step"],
            "tables": [np.frombuffer(b"".join(t), dtype="<u4") for t in v["tables"]],
            "bricks": [np.frombuffer(b"".join(b), dtype="<f4").reshape(-1, 512, 4)
                       for b in v["bricks"]]}


def dense(v, k):
    # Frame k back to a dense (nx, ny, nz, 4) grid, decoded here from the
    # layout (table x fastest; cells x fastest in a brick), not by the
    # add-on's reader.
    n = [int(x) for x in v["resolution"][k]]
    if min(n) == 0:
        return np.zeros((0, 0, 0, 4))
    table = np.frombuffer(b"".join(v["tables"][k]), dtype="<u4")
    bricks = np.frombuffer(b"".join(v["bricks"][k]), dtype="<f4").reshape(-1, 512, 4)
    nb = [(x + 7) // 8 for x in n]
    if len(table) != nb[0] * nb[1] * nb[2]:
        raise RuntimeError("frame %d: table of %d for %s bricks" % (k, len(table), nb))
    out = np.zeros((nb[0] * 8, nb[1] * 8, nb[2] * 8, 4))
    for idx, slot in enumerate(table):
        if slot == EMPTY:
            continue
        bx, by, bz = idx % nb[0], idx // nb[0] % nb[1], idx // (nb[0] * nb[1])
        out[bx * 8:bx * 8 + 8, by * 8:by * 8 + 8, bz * 8:bz * 8 + 8] = (
            bricks[slot].reshape(8, 8, 8, 4).transpose(2, 1, 0, 3))
    return out[:n[0], :n[1], :n[2]]


def cell_value(v, k, ijk):
    # One cell's four channels through the table, for a frame too large to
    # decode densely.
    n = [int(x) for x in v["resolution"][k]]
    nb = [(x + 7) // 8 for x in n]
    table = np.frombuffer(b"".join(v["tables"][k]), dtype="<u4")
    i, j, kk = (int(c) for c in ijk)
    slot = table[i // 8 + nb[0] * (j // 8 + nb[1] * (kk // 8))]
    if slot == EMPTY:
        return np.zeros(4)
    bricks = np.frombuffer(b"".join(v["bricks"][k]), dtype="<f4").reshape(-1, 512, 4)
    return bricks[slot, i % 8 + 8 * (j % 8 + 8 * (kk % 8))].astype(np.float64)


def euler_xyz(angles):
    # Blender's XYZ Euler as a rotation matrix, in double: X first, then Y,
    # then Z.
    x, y, z = (float(a) for a in angles)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(x), -math.sin(x)],
                   [0.0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0.0, math.sin(y)], [0.0, 1.0, 0.0],
                   [-math.sin(y), 0.0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0.0], [math.sin(z), math.cos(z), 0.0],
                   [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def rel_err(got, want):
    want = np.asarray(want, dtype=np.float64)
    scale = max(float(np.abs(want).max()), 1e-30)
    return float(np.abs(np.asarray(got, dtype=np.float64) - want).max()) / scale


class ReadCount:
    # Counts the add-on's per-frame fluid reads by patching the reader the
    # encode calls.
    def __enter__(self):
        self.n = 0
        self.orig = ff.fluid_frame

        def counted(*args, **kwargs):
            self.n += 1
            return self.orig(*args, **kwargs)

        ff.fluid_frame = counted
        return self

    def __exit__(self, *exc):
        ff.fluid_frame = self.orig
        return False


def reset():
    state.time_scale = 1.0
    state.use_scene_fps = True
    state.frame_rate = 100
    state.use_scene_frame_start = False
    if int(state.frame_count) != FC:
        state.frame_count = FC
    if int(state.frame_start) != START:
        state.frame_start = START
    f = flow.field
    f.strength = 1.0
    f.flow = 0.0
    f.use_smoke_density = False
    f.noise = 0.0
    f.wind_factor = 0.0
    f.use_gravity_falloff = False
    f.use_absorption = False
    scene.frame_set(1)
    ff.release_encode_cache()


def section(fn):
    try:
        reset()
        fn()
    except Exception as exc:
        dh.record(fn.__name__ + "_raised", False,
                  {"error": "{}: {}".format(type(exc).__name__, exc),
                   "traceback": traceback.format_exc()})


def drop(*objs):
    for o in objs:
        if o is None:
            continue
        data = o.data if o.type == "VOLUME" else None
        bpy.data.objects.remove(o)
        if data is not None and data.users == 0:
            bpy.data.volumes.remove(data)


# --- Mantaflow ---------------------------------------------------------------

def fcurves_of(obj):
    # The F-curves of an object's action: through its slot's channel bag,
    # or the action's own list where the layered API is absent.
    ad = obj.animation_data
    try:
        from bpy_extras import anim_utils
        return list(anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot).fcurves)
    except (ImportError, AttributeError):
        return list(ad.action.fcurves)


def bake(dom):
    bake_fluid_domain(scene, dom, result.setdefault("bake_notes", []))


def free(dom):
    with bpy.context.temp_override(scene=scene, object=dom, active_object=dom,
                                   selected_objects=[dom]):
        bpy.ops.fluid.free_all()


def gas_domain(name, location, scale=(1.0, 1.0, 1.0), rotation=(0.0, 0.0, 0.0), res=16,
               frames=(1, 13), cache_type="ALL", bake_it=True, keys=None):
    # A size-2 cube (its bounds -1..1 locally) as a gas domain, an inflow
    # jet inside it parented to it, keyed by `keys` (frame -> (location,
    # rotation)) before the bake when given.
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)
    dom = bpy.context.object
    dom.name = name
    dom.scale = scale
    dom.rotation_euler = rotation
    mod = dom.modifiers.new("Fluid", "FLUID")
    mod.fluid_type = "DOMAIN"
    ds = mod.domain_settings
    ds.domain_type = "GAS"
    ds.resolution_max = res
    ds.alpha = ds.beta = 0.0
    ds.gravity = (0.0, 0.0, 0.0)
    ds.use_adaptive_timesteps = False
    ds.cache_type = cache_type
    ds.cache_data_format = "OPENVDB"
    ds.cache_directory = os.path.join(PROBE_DIR, "ffe_cache_" + name)
    ds.cache_frame_start, ds.cache_frame_end = frames
    bpy.context.view_layer.update()
    at = dom.matrix_world @ Vector((-0.6, 0.0, 0.0))
    bpy.ops.mesh.primitive_cube_add(size=0.4 * min(scale), location=at)
    jet = bpy.context.object
    jet.name = name + "Jet"
    fm = jet.modifiers.new("Fluid", "FLUID")
    fm.fluid_type = "FLOW"
    fl = fm.flow_settings
    fl.flow_type = "SMOKE"
    fl.flow_behavior = "INFLOW"
    fl.use_initial_velocity = True
    fl.velocity_coord = (1.5, 0.3, 0.0)
    jet.parent = dom
    jet.matrix_parent_inverse = dom.matrix_world.inverted()
    if keys:
        for frame, (loc, rot) in sorted(keys.items()):
            dom.location = loc
            dom.rotation_euler = rot
            dom.keyframe_insert("location", frame=frame)
            dom.keyframe_insert("rotation_euler", frame=frame)
        # LINEAR on the curves themselves (measured: the preference for new
        # keys does not reach keyframe_insert, which keys Bezier).
        for fc in fcurves_of(dom):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
            fc.update()
        bpy.context.view_layer.update()
    if bake_it:
        bake(dom)
    return dom, jet


def domain_grid(dom, frame):
    # The domain's own evaluated grid at `frame`, (nx, ny, nz, 3) in its
    # grid units and local axes, and its resolution; the playhead restored.
    keep = scene.frame_current
    scene.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    ds = dom.evaluated_get(dg).modifiers["Fluid"].domain_settings
    n = tuple(int(x) for x in ds.domain_resolution)
    grid = np.array(ds.velocity_grid, dtype=np.float64).reshape(n[2], n[1], n[0], 3)
    cell_size = np.array(ds.cell_size, dtype=np.float64)
    scene.frame_set(keep)
    return n, grid.transpose(2, 1, 0, 3), cell_size


# --- openvdb -----------------------------------------------------------------

VDB_ROOT = os.path.join(PROBE_DIR, "ffe_vdb")


def folder(tag):
    path = os.path.join(VDB_ROOT, tag)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path


def vdb_file(tag, grids, name="v.vdb"):
    # Each source gets a folder of its own: the encode cache keys a Volume
    # on its folder's files.
    path = os.path.join(folder(tag), name)
    openvdb.write(path, grids=grids)
    return path


def vec_grid(name, voxels=None, array=None, voxel=0.2, cls=None, transform=None):
    g = openvdb.Vec3SGrid()
    g.name = name
    g.transform = transform if transform is not None else openvdb.createLinearTransform(
        voxelSize=voxel)
    if cls is not None:
        g.gridClass = cls
    if array is not None:
        # C order: copyFromArray reads raw memory, without strides.
        g.copyFromArray(np.ascontiguousarray(array, dtype=np.float32))
    if voxels:
        acc = g.getAccessor()
        for ijk, value in voxels.items():
            acc.setValueOn(tuple(int(c) for c in ijk), tuple(float(c) for c in value))
    return g


def float_grid(name, voxels, voxel=0.2, cls=None):
    g = openvdb.FloatGrid()
    g.name = name
    g.transform = openvdb.createLinearTransform(voxelSize=voxel)
    if cls is not None:
        g.gridClass = cls
    acc = g.getAccessor()
    for ijk, value in voxels.items():
        acc.setValueOn(tuple(int(c) for c in ijk), float(value))
    return g


def volume(name, path, unit="SECOND"):
    # The unit is set after the load, which is when Blender names the
    # velocity grid it finds (measured: a name set before is cleared).
    data = bpy.data.volumes.new(name)
    data.filepath = path
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    data.grids.load()
    if unit is not None:
        data.velocity_unit = unit
    bpy.context.view_layer.update()
    return obj


def use(src):
    flow.field.source_object = src
    bpy.context.view_layer.update()


# ----- the scene -------------------------------------------------------------

try:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene.render.fps = SCENE_FPS
    scene.render.fps_base = 1.0
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3, size=0.5,
                                    location=(0.0, 0.0, 30.0))
    sheet = bpy.context.object
    sheet.name = "FFESheet"
    dh.save_blend(PROBE_DIR, "force_field_fluid_encode.blend")
    root = dh.configure_state(project_name="force_field_fluid_encode", frame_count=FC)
    state = root.state
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    bpy.ops.object.effector_add(type="FLUID", location=(0.0, 0.0, 28.0))
    flow = bpy.context.object
    flow.name = "FFEFlow"
    reset()
    # The domain the determinism and refusal checks read, baked over frames
    # 2..13 around the solve's 3..12.
    det, det_jet = gas_domain("FFEDet", (0.0, 8.0, 0.0), frames=(2, 13))
except Exception as exc:
    result["errors"].append("setup: {}: {}".format(type(exc).__name__, exc))
    result["errors"].append(traceback.format_exc())
    raise


# ----- A: Replay refusal and baked determinism --------------------------------

def A_determinism():
    use(det)
    flow.field.strength = 2.0
    flow.field.flow = 1.0
    flow.field.use_smoke_density = True
    scene.frame_set(1)
    with ReadCount() as c1:
        h1 = params_mod.compute_param_hash(bpy.context)
    # Play the timeline as an artist would, park the playhead elsewhere, and
    # read every frame again from scratch.
    for f in range(1, 15):
        scene.frame_set(f)
    scene.frame_set(9)
    ff.release_encode_cache()
    with ReadCount() as c2:
        data2, h2 = params_mod.encode_param_with_hash(bpy.context)
    with ReadCount() as c3:
        h3 = params_mod.compute_param_hash(bpy.context)
    parked = scene.frame_current
    v = fluid_entry()
    d2, d6 = dense(v, 2)[..., :3], dense(v, 6)[..., :3]
    moving = rel_err(d6, d2)
    ok = (h1 == h2 == h3 and digest(data2) == h2
          and c1.n == FC and c2.n == FC and c3.n == 0 and parked == 9
          and float(np.abs(d6).max()) > 0.1 and moving > 0.05)
    dh.record("A_baked_domain_hash_is_deterministic", ok,
              {"hashes": [h1, h2, h3], "reads": [c1.n, c2.n, c3.n], "parked": parked,
               "peak_velocity": float(np.abs(d6).max()), "frame_2_to_6_change": moving})

    # A Replay domain, simulated live up to the solve as playing to it does.
    rep, rep_jet = gas_domain("FFEReplay", (-8.0, 0.0, 0.0), cache_type="REPLAY",
                              bake_it=False)
    use(rep)
    scene.frame_set(1)
    for f in range(1, START):
        scene.frame_set(f)
    # The first encode walks the solve frames while the domain simulates
    # them, so its hash is the live one.
    live = None
    try:
        live = params_mod.compute_param_hash(bpy.context)
        msg = "no error"
    except ValueError as e:
        msg = str(e)
    except Exception as e:
        msg = "not a ValueError: {}: {}".format(type(e).__name__, e)
    want = (FIELD, "'FFEReplay'", "Replay cache", "set Cache Type to All or Modular and bake it")
    missing = [f for f in want if f not in msg]
    detail = {"error": msg, "missing": missing}
    if live is not None:
        # Admitted, against the contract: record whether a read-back after
        # a walk sends what the live walk sent.
        for f in range(1, 15):
            scene.frame_set(f)
        ff.release_encode_cache()
        try:
            detail["read_back_hash_equals_live"] = (
                params_mod.compute_param_hash(bpy.context) == live)
        except ValueError as e:
            detail["read_back"] = str(e)
    dh.record("A_replay_domain_refused_by_name", live is None and not missing, detail)
    drop(rep_jet, rep)

    # A baked domain whose bake was freed, a domain of its own so that no
    # domain is baked again after a free: that crashed Blender 5.1.2 on
    # Windows.
    freed, freed_jet = gas_domain("FFEFreed", (8.0, 8.0, 0.0), frames=(2, 13))
    use(freed)
    free(freed)
    record_refusal("A_unbaked_domain_refused_by_name", FIELD, "'FFEFreed'", "is not baked",
                   "bake its data over the solve frames")
    drop(freed_jet, freed)
    use(det)
    state.frame_count = 12
    over = refused(FIELD, "'FFEDet'", "is baked over frames 2..13",
                   "the solve reads frames 3..14")
    state.frame_count = FC
    state.frame_start = 1
    under = refused(FIELD, "'FFEDet'", "is baked over frames 2..13",
                    "the solve reads frames 1..10")
    state.frame_start = START
    inside = fluid_entry()
    dh.record("A_bake_range_refused_by_name",
              over[0] and under[0] and len(inside["tables"]) == FC,
              {"past_the_end": over[1], "before_the_start": under[1]})


# ----- B: every source refusal by name ---------------------------------------

def B_refusals():
    use(None)
    record_refusal("B_no_domain_object", FIELD, "no Domain Object")

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, -20.0, 0.0))
    plain = bpy.context.object
    plain.name = "FFEPlain"
    use(plain)
    record_refusal("B_neither_domain_nor_volume", FIELD, "'FFEPlain'",
                   "neither a fluid domain nor a Volume")
    drop(plain)

    # Checked before the bake state, so this domain is never baked.
    rdom, rjet = gas_domain("FFERefuse", (0.0, 0.0, 8.0), res=12, bake_it=False)
    use(rdom)
    rdom.modifiers["Fluid"].domain_settings.use_noise = True
    flow.field.use_smoke_density = True
    record_refusal("B_noise_with_apply_density", FIELD, "'FFERefuse'", "has Noise on",
                   "density grid")
    flow.field.use_smoke_density = False
    rdom.modifiers["Fluid"].domain_settings.use_noise = False
    arr = rdom.modifiers.new("FFEArray", "ARRAY")
    ok, msg, missing = refused(FIELD, "'FFERefuse'",
                               "has modifier 'FFEArray' after its Fluid modifier")
    arr.show_viewport = False
    arr.show_render = False
    later = ff.refusal(flow) or ""
    dh.record("B_modifier_after_fluid", ok and "FFEArray" not in later,
              {"error": msg, "missing": missing, "disabled_reason": later})
    drop(rjet, rdom)

    # A valid Volume, the base for the field-setting refusals.
    v0_path = vdb_file("refuse_v0", [vec_grid("vel", {(0, 0, 0): (1.0, 0.0, 0.0),
                                                      (1, 0, 0): (1.0, 0.0, 0.0)})],
                       name="v0.vdb")
    v0 = volume("FFEV0", v0_path)
    use(v0)
    base = refused()
    sent = {}
    if base[1] == "no error":
        sent = [f for f in build()["force_field"]["fields"] if f["type"] == "fluid"][0]
    # Blender applies no Gravitation to a Fluid Flow (measured), so none is sent.
    dh.record("B_a_valid_volume_encodes",
              base[1] == "no error" and sent.get("gravitation") is False,
              {"error": base[1], "gravitation": sent.get("gravitation")})
    # Gravitation, which Blender ignores on a Fluid Flow (measured: on and
    # off move its particles and cloth alike, and B_blender_ignores_...
    # holds that), is sent off rather than refused, since the Fluid Flow
    # panel does not show it.
    flow.field.use_gravity_falloff = True
    grav = refused()
    sent_g = {}
    if grav[1] == "no error":
        sent_g = [f for f in build()["force_field"]["fields"] if f["type"] == "fluid"][0]
    dh.record("B_gravitation_on_fluid_flow_is_sent_off",
              grav[1] == "no error" and sent_g.get("gravitation") is False
              and ff.refusal(flow) is None,
              {"error": grav[1], "gravitation": sent_g.get("gravitation"),
               "listed": ff.refusal(flow)})
    flow.field.use_gravity_falloff = False
    # Noise Amount, Wind Factor and Absorption, which Blender applies on a
    # Fluid Flow, stay refused, each with the way to reach a setting the
    # Fluid Flow panel does not show.
    hint = "set the field's type to Force, change it there, then set the type back"
    flow.field.noise = 0.5
    record_refusal("B_noise_amount_on_fluid_flow", FIELD,
                   "Noise Amount on a Fluid Flow field is not supported", hint)
    flow.field.noise = 0.0
    flow.field.wind_factor = 0.5
    record_refusal("B_wind_factor_on_fluid_flow", FIELD,
                   "Wind Factor on a Fluid Flow field is not supported", hint)
    flow.field.wind_factor = 0.0
    flow.field.use_absorption = True
    record_refusal("B_absorption_on_fluid_flow", FIELD,
                   "Absorption is not supported yet", hint)
    flow.field.use_absorption = False
    flow.field.use_smoke_density = True
    record_refusal("B_apply_density_on_a_velocity_only_file", FIELD + " at frame 3",
                   "its Volume 'FFEV0''s file", "v0.vdb", "Apply Density is on",
                   "holds no density grid", "(it holds vel)")
    flow.field.use_smoke_density = False

    disp = v0.modifiers.new("FFEDisplace", "VOLUME_DISPLACE")
    ok_d, msg_d, miss_d = refused(FIELD, "its Volume 'FFEV0'", "has modifier 'FFEDisplace'")
    v0.modifiers.remove(disp)
    nodes = v0.modifiers.new("FFENodes", "NODES")
    ok_n, msg_n, miss_n = refused(FIELD, "its Volume 'FFEV0'", "has modifier 'FFENodes'")
    v0.modifiers.remove(nodes)
    dh.record("B_volume_with_a_modifier", ok_d and ok_n,
              {"displace": msg_d, "nodes": msg_n, "missing": miss_d + miss_n})

    # A non-uniformly scaled parent turns a rotated child's voxels into a
    # shear.
    stretch = bpy.data.objects.new("FFEStretch", None)
    scene.collection.objects.link(stretch)
    stretch.scale = (1.0, 3.0, 1.0)
    v0.parent = stretch
    v0.rotation_euler = (0.0, 0.0, 0.5)
    bpy.context.view_layer.update()
    record_refusal("B_volume_sheared_by_its_parent", FIELD + " at frame 3", "'FFEV0'",
                   "sheared, mirrored or of zero size")
    v0.parent = None
    v0.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()

    # Voxels rotated in the file, stretched by the object along its own axes.
    c, s = math.cos(0.4), math.sin(0.4)
    rot_vox = openvdb.createLinearTransform(
        [[0.2 * c, 0.2 * s, 0.0, 0.0], [-0.2 * s, 0.2 * c, 0.0, 0.0],
         [0.0, 0.0, 0.2, 0.0], [0.0, 0.0, 0.0, 1.0]])
    rv = volume("FFERotVox", vdb_file("refuse_rotvox", [
        vec_grid("vel", {(0, 0, 0): (1.0, 0.0, 0.0)}, transform=rot_vox)]))
    use(rv)
    rv.scale = (1.5, 1.5, 1.5)
    bpy.context.view_layer.update()
    uniform = refused()
    rv.scale = (1.0, 2.0, 1.0)
    bpy.context.view_layer.update()
    ok, msg, missing = refused(FIELD + " at frame 3", "'FFERotVox'",
                               "sheared, mirrored or of zero size")
    dh.record("B_volume_scaled_across_rotated_voxels", ok and uniform[1] == "no error",
              {"error": msg, "missing": missing, "uniform_scale": uniform[1]})
    drop(rv)

    mis = vdb_file("refuse_mismatch", [vec_grid("vel", {(0, 0, 0): (1.0, 0.0, 0.0)}, voxel=0.2),
                                       float_grid("density", {(0, 0, 0): 0.7}, voxel=0.1)],
                   name="dens_mismatch.vdb")
    mv = volume("FFEMismatch", mis)
    use(mv)
    flow.field.use_smoke_density = True
    record_refusal("B_density_on_other_voxels_than_velocity", FIELD + " at frame 3",
                   "its Volume 'FFEMismatch''s file", "dens_mismatch.vdb",
                   "the grid 'density' is not on the voxels of 'vel'")
    flow.field.use_smoke_density = False
    drop(mv)

    nofile = bpy.data.volumes.new("FFENoFile")
    nf = bpy.data.objects.new("FFENoFile", nofile)
    scene.collection.objects.link(nf)
    use(nf)
    record_refusal("B_empty_file_path", FIELD + " at frame 3", "its Volume 'FFENoFile'",
                   "resolves no file", "it names no file")
    drop(nf)

    fv = volume("FFEFloatVel", vdb_file("refuse_float", [float_grid("vel", {(0, 0, 0): 1.0})],
                                        name="float_vel.vdb"))
    fv.data.velocity_grid = "vel"
    use(fv)
    record_refusal("B_float_grid_as_the_velocity", FIELD + " at frame 3",
                   "its Volume 'FFEFloatVel''s file", "float_vel.vdb",
                   "the velocity grid 'vel'", "is float, not vec3s or vec3d")
    drop(fv)

    clip_dir = folder("refuse_clip")
    for i in range(1, FC):
        openvdb.write(os.path.join(clip_dir, "clip_%04d.vdb" % i),
                      grids=[vec_grid("vel", {(0, 0, 0): (float(i), 0.0, 0.0)})])
    cv = volume("FFEClip", os.path.join(clip_dir, "clip_0001.vdb"))
    cv.data.is_sequence = True
    cv.data.frame_start = START
    cv.data.frame_duration = FC - 1
    cv.data.sequence_mode = "CLIP"
    use(cv)
    record_refusal("B_clip_sequence_one_frame_short", FIELD + " at frame 12",
                   "its Volume 'FFEClip' resolves no file", "Sequence Mode Clip")
    drop(cv)

    # A Mantaflow cache: read in the domain's folder, and copied elsewhere.
    cached = os.path.join(det.modifiers["Fluid"].domain_settings.cache_directory,
                          "data", "fluid_data_0005.vdb")
    have_cache = os.path.isfile(cached)
    in_dir = volume("FFECacheVol", cached)
    use(in_dir)
    ok, msg, missing = refused(FIELD, "its Volume 'FFECacheVol'",
                               "reads the cache of fluid domain 'FFEDet'")
    dh.record("B_volume_reading_a_domain_cache", ok and have_cache,
              {"error": msg, "missing": missing, "file": cached, "exists": have_cache})
    drop(in_dir)
    copied = os.path.join(folder("refuse_copied"), "copied_cache.vdb")
    if have_cache:
        shutil.copy(cached, copied)
    cpv = volume("FFECopiedVol", copied)
    use(cpv)
    record_refusal("B_mantaflow_cache_file_copied_elsewhere", FIELD + " at frame 3",
                   "its Volume 'FFECopiedVol''s file", "copied_cache.vdb",
                   "is a Mantaflow cache file",
                   "point the field at its domain")
    drop(cpv)
    drop(v0, stretch)

    # A sheared domain: a non-uniformly scaled parent over a rotated domain.
    use(det)
    det_stretch = bpy.data.objects.new("FFEDetStretch", None)
    scene.collection.objects.link(det_stretch)
    det_stretch.scale = (1.0, 3.0, 1.0)
    det.parent = det_stretch
    det.rotation_euler = (0.0, 0.0, 0.5)
    bpy.context.view_layer.update()
    record_refusal("B_sheared_domain", FIELD + " at frame 3", "'FFEDet'",
                   "is sheared, mirrored or of zero scale")
    det.parent = None
    det.rotation_euler = (0.0, 0.0, 0.0)
    drop(det_stretch)
    bpy.context.view_layer.update()


# ----- C: Volume sources honored ---------------------------------------------

def C_volume_sources():
    # De-staggering: u(x) = A x + b in index space, written on the faces.
    A = np.array([[0.5, 0.2, -0.1], [0.3, -1.0, 0.25], [-0.2, 0.15, 2.0]])
    b = np.array([0.1, -0.2, 0.3])
    box = (5, 4, 4)
    faces = {}
    comps = ({}, {}, {})
    for i in range(box[0]):
        for j in range(box[1]):
            for k in range(box[2]):
                value = []
                for c in range(3):
                    x = np.array([i, j, k], dtype=np.float64)
                    x[c] -= 0.5
                    value.append(float(A[c] @ x + b[c]))
                    comps[c][(i, j, k)] = value[c]
                faces[(i, j, k)] = tuple(value)
    full = [(i, j, k) for i in range(box[0] - 1) for j in range(box[1] - 1)
            for k in range(box[2] - 1)]
    want = np.array([A @ np.array(x, dtype=np.float64) + b for x in full])

    def staggered_case(name, grids):
        vol = volume(name, vdb_file("stag_" + name, grids))
        use(vol)
        v = fluid_entry()
        g = dense(v, 0)[..., :3]
        got = np.array([g[x] for x in full])
        err = float(np.abs(got - want).max())
        corner = np.asarray(v["corner"][0])
        out = {"error": err, "scale": float(np.abs(want).max()),
               "resolution": v["resolution"][0], "corner": corner.tolist(),
               "names": [vol.data.velocity_grid, vol.data.velocity_x_grid,
                         vol.data.velocity_y_grid, vol.data.velocity_z_grid]}
        ok = (err < 1e-5 and list(v["resolution"][0]) == list(box)
              and np.allclose(corner, solver([-0.125, -0.125, -0.125]), atol=1e-9))
        drop(vol)
        return ok, out

    ok, out = staggered_case("FFEStag", [vec_grid(
        "vel", faces, voxel=0.25, cls=openvdb.GridClass.STAGGERED)])
    dh.record("C_staggered_grid_is_destaggered", ok, out)
    ok, out = staggered_case("FFEStagSplit", [
        float_grid("vel." + "xyz"[c], comps[c], voxel=0.25, cls=openvdb.GridClass.STAGGERED)
        for c in range(3)])
    dh.record("C_staggered_split_grids", ok, out)

    # Split component grids, each with its own active voxels.
    xs = {(0, 0, 0): 1.0, (1, 0, 0): 2.0}
    ys = {(0, 0, 0): 3.0, (0, 1, 0): -1.0}
    zs = {(1, 1, 0): 4.0}
    split = volume("FFESplit", vdb_file("split", [float_grid("vel.x", xs),
                                                  float_grid("vel.y", ys),
                                                  float_grid("vel.z", zs)]))
    use(split)
    v = fluid_entry()
    g = dense(v, 0)[..., :3]
    want_split = np.zeros((2, 2, 1, 3))
    for c, comp in enumerate((xs, ys, zs)):
        for ijk, value in comp.items():
            want_split[ijk + (c,)] = value
    err = float(np.abs(g - want_split).max()) if g.shape == want_split.shape else None
    dh.record("C_split_component_grids",
              err is not None and err < 1e-6 and split.data.velocity_x_grid == "vel.x",
              {"error": err, "resolution": v["resolution"][0],
               "names": [split.data.velocity_x_grid, split.data.velocity_y_grid,
                         split.data.velocity_z_grid]})
    drop(split)

    # The Volume's Velocity Grid names which vector grid is read.
    two = volume("FFETwo", vdb_file("two", [vec_grid("velocity", {(0, 0, 0): (1.0, 0.0, 0.0)}),
                                            vec_grid("flowvel", {(0, 0, 0): (5.0, -2.0, 0.5)})]))
    two.data.velocity_grid = "flowvel"
    use(two)
    got = dense(fluid_entry(), 0)[0, 0, 0, :3]
    dh.record("C_named_velocity_grid", float(np.abs(got - [5.0, -2.0, 0.5]).max()) < 1e-6,
              {"got": got.tolist()})
    drop(two)

    # Velocity Unit and Scale, and the rate factor, on one file.
    U = {(0, 0, 0): (1.0, -2.0, 0.5), (2, 1, 0): (0.25, 0.0, 3.0), (1, 2, 3): (-1.5, 1.0, 2.0)}
    unit_path = vdb_file("units", [vec_grid("vel", U)])
    lo = (0, 0, 0)

    def read_u():
        ff.release_encode_cache()
        v = fluid_entry()
        g = dense(v, 0)
        return np.array([g[tuple(np.subtract(ijk, lo))][:3] for ijk in U])

    raw = np.array(list(U.values()))
    fresh = volume("FFEUnits", unit_path, unit=None)
    default_unit = fresh.data.velocity_unit
    use(fresh)
    cases = {}
    cases["default_FRAME_x24"] = rel_err(read_u(), 24.0 * raw)
    fresh.data.velocity_unit = "SECOND"
    cases["SECOND_x1"] = rel_err(read_u(), raw)
    fresh.data.velocity_scale = 0.5
    cases["SECOND_scale_0.5"] = rel_err(read_u(), 0.5 * raw)
    fresh.data.velocity_unit = "FRAME"
    cases["FRAME_scale_0.5_x12"] = rel_err(read_u(), 12.0 * raw)
    dh.record("C_velocity_unit_and_scale",
              default_unit == "FRAME" and max(cases.values()) < 1e-6,
              {"default_unit": default_unit, "relative_errors": cases})

    fresh.data.velocity_scale = 1.0
    fresh.data.velocity_unit = "SECOND"
    rates = {}
    state.time_scale = 0.5
    rates["time_scale_0.5"] = rel_err(read_u(), 0.5 * raw)
    state.time_scale = 1.0
    state.use_scene_fps = False
    state.frame_rate = 48
    rates["frame_rate_48"] = rel_err(read_u(), 2.0 * raw)
    fresh.data.velocity_unit = "FRAME"
    rates["FRAME_frame_rate_48_x48"] = rel_err(read_u(), 48.0 * raw)
    fresh.data.velocity_unit = "SECOND"
    state.time_scale = 0.5
    rates["frame_rate_48_time_scale_0.5"] = rel_err(read_u(), raw)
    state.time_scale = 1.0
    state.use_scene_fps = True
    # A scene at 29.97 (FPS 30, Base 1.001): Use Scene FPS runs the solver at
    # the whole-number 30, so a velocity per scene second is 1.001 times it
    # per solver second, while a FRAME velocity, per frame times the scene's
    # 29.97, comes to the file times 30.
    scene.render.fps, scene.render.fps_base = 30, 1.001
    rates["fps_29.97_SECOND_x1.001"] = rel_err(read_u(), 1.001 * raw)
    fresh.data.velocity_unit = "FRAME"
    rates["fps_29.97_FRAME_x30"] = rel_err(read_u(), 30.0 * raw)
    fresh.data.velocity_unit = "SECOND"
    scene.render.fps, scene.render.fps_base = SCENE_FPS, 1.0
    dh.record("C_rate_factor_on_volume", max(rates.values()) < 1e-6,
              {"relative_errors": rates})
    drop(fresh)

    # An empty first frame: the sequence's first file holds an empty grid.
    seq = folder("empty_first")
    e = openvdb.Vec3SGrid()
    e.name = "vel"
    e.transform = openvdb.createLinearTransform(voxelSize=0.2)
    openvdb.write(os.path.join(seq, "seq_0001.vdb"), grids=[e])
    filled = {(3, 1, 2): (1.0, 0.5, -0.25), (4, 1, 2): (2.0, 0.0, 0.0)}
    openvdb.write(os.path.join(seq, "seq_0002.vdb"), grids=[vec_grid("vel", filled)])
    sv = volume("FFEEmptyFirst", os.path.join(seq, "seq_0001.vdb"))
    sv.data.is_sequence = True
    sv.data.frame_start = START
    sv.data.frame_duration = 2
    sv.data.sequence_mode = "EXTEND"
    use(sv)
    v = fluid_entry()
    vol = payload_volume(v)
    # The index origin's voxel center and the voxels the next file fills.
    m = np.array(sv.matrix_world, dtype=np.float64)
    pts = [(0, 0, 0)] + list(filled)
    world = np.array([(m @ np.append(0.2 * np.array(p, dtype=np.float64), 1.0))[:3] for p in pts])
    _, _, p0 = fe.fluid_at(vol, solver(world), 0.0)
    u1, _, p1 = fe.fluid_at(vol, solver(world), v["step"])
    want1 = solver(np.array(list(filled.values())))
    ok = (list(v["resolution"][0]) == [0, 0, 0] and v["tables"][0] == []
          and v["bricks"][0] == [] and all(math.isfinite(x) for x in v["corner"][0])
          and np.allclose(v["cell"][0], 0.2, rtol=0.0, atol=1e-9)
          and float(np.abs(p0).max()) == 0.0
          and list(v["resolution"][1]) == [2, 1, 1] and np.all(p1[1:] == 1.0)
          and float(p1[0]) == 0.0 and float(np.abs(u1[1:] - want1).max()) < 1e-6)
    dh.record("C_empty_first_frame", ok,
              {"resolutions": v["resolution"][:2], "tables_0": len(v["tables"][0]),
               "presence_frame_0": p0.tolist(), "presence_frame_1": p1.tolist(),
               "velocity_frame_1": u1[1:].tolist(), "names": sv.data.velocity_grid})
    drop(sv)


# ----- D: Mantaflow domains honored ------------------------------------------

def D_domains():
    K = 4      # the payload frame compared, Blender frame START + K

    def unit_of(scale, n, domain_time_scale=1.0, rate=1.0):
        extent = 2.0 * np.asarray(scale, dtype=np.float64)
        return 2.5 * float(extent.max()) / max(n) * domain_time_scale * rate

    def values_error(dom, v, scale, factor=1.0, domain_time_scale=1.0):
        n, grid, _ = domain_grid(dom, START + K)
        want = grid * unit_of(scale, n, domain_time_scale) * factor
        got = dense(v, K)[..., :3]
        return rel_err(got, want), float(np.abs(want).max())

    def face_points(dom, scale, d=0.03):
        # Domain-local points d inside and d outside each face, at a little
        # off the face's center, in world space; and inside or not.
        m = np.array(dom.matrix_world, dtype=np.float64)
        half = np.asarray(scale, dtype=np.float64)
        pts, inside = [], []
        for a in range(3):
            for side in (-1.0, 1.0):
                for sign, inner in ((-1.0, True), (1.0, False)):
                    p = np.array([0.1, 0.1, 0.1])
                    p[a] = side * (1.0 + sign * d / half[a])
                    pts.append((m @ np.append(p, 1.0))[:3])
                    inside.append(inner)
        return np.array(pts), np.array(inside)

    cases = (("4x1.3x2", "FFENc1", (0.0, -8.0, 0.0), (2.0, 0.65, 1.0), (0.2, -0.1, 0.5)),
             ("2x2x2.2", "FFENc2", (8.0, 0.0, 0.0), (1.0, 1.0, 1.1), (0.0, 0.0, 0.3)))
    made = {}
    for label, name, loc, scale, rot in cases:
        dom, jet = gas_domain(name, loc, scale=scale, rotation=rot)
        made[label] = (dom, jet, scale)
        use(dom)
        v = fluid_entry()
        n, _, cell_size = domain_grid(dom, START + K)
        extent = 2.0 * np.asarray(scale, dtype=np.float64)
        cell = np.asarray(v["cell"][K], dtype=np.float64)
        m = np.array(dom.matrix_world, dtype=np.float64)
        lo = solver((m @ np.array([-1.0, -1.0, -1.0, 1.0]))[:3])
        hi = solver((m @ np.array([1.0, 1.0, 1.0, 1.0]))[:3])
        corner = np.asarray(v["corner"][K], dtype=np.float64)
        axes = np.asarray(v["axes"][K], dtype=np.float64)
        far = corner + axes @ (np.array(n) * cell)
        cell_err = rel_err(cell, extent / np.array(n))
        blender_err = rel_err(cell, cell_size * np.asarray(scale))
        ok = (list(v["resolution"][K]) == list(n) and cell_err < 1e-6 and blender_err < 1e-6
              and float(np.abs(corner - lo).max()) < 1e-9
              and float(np.abs(far - hi).max()) < 1e-9)
        dh.record("D_noncubic_%s_cells_and_box" % label, ok,
                  {"resolution": v["resolution"][K], "cell": cell.tolist(),
                   "blender_cell": (cell_size * np.asarray(scale)).tolist(),
                   "cell_error": cell_err, "blender_cell_error": blender_err,
                   "corner_error": float(np.abs(corner - lo).max()),
                   "far_corner_error": float(np.abs(far - hi).max())})

        pts, inside = face_points(dom, scale)
        vol = payload_volume(v)
        _, _, presence = fe.fluid_frame(vol, K, solver(pts))
        got_in = presence > 0.5
        # What one cubic cell, the longest axis's, would judge.
        g = ((solver(pts) - corner) @ axes) / (extent.max() / max(n))
        cubic_in = np.all((g >= 0.0) & (g <= np.array(n)), axis=1)
        dh.record("D_noncubic_%s_presence_at_faces" % label,
                  bool(np.all(got_in == inside)) and int(np.sum(cubic_in != inside)) >= 1,
                  {"presence": presence.tolist(), "want_inside": inside.tolist(),
                   "cubic_cell_misjudges": int(np.sum(cubic_in != inside))})

        err, peak = values_error(dom, v, scale)
        dh.record("D_noncubic_%s_values" % label, err < 1e-6 and peak > 0.1,
                  {"relative_error": err, "peak": peak})

    # The rate factor on a domain's U.
    dom, _, scale = made["4x1.3x2"]
    use(dom)
    rates = {}
    state.time_scale = 0.5
    ff.release_encode_cache()
    rates["time_scale_0.5"] = values_error(dom, fluid_entry(), scale, 0.5)[0]
    state.time_scale = 1.0
    state.use_scene_fps = False
    state.frame_rate = 48
    ff.release_encode_cache()
    rates["frame_rate_48"] = values_error(dom, fluid_entry(), scale, 2.0)[0]
    state.use_scene_fps = True
    # At 29.97 (FPS 30, Base 1.001) the solver runs at the whole-number 30.
    scene.render.fps, scene.render.fps_base = 30, 1.001
    ff.release_encode_cache()
    rates["fps_29.97_x1.001"] = values_error(dom, fluid_entry(), scale, 30.0 / (30.0 / 1.001))[0]
    scene.render.fps, scene.render.fps_base = SCENE_FPS, 1.0
    dh.record("D_rate_factor_on_domain", max(rates.values()) < 1e-6,
              {"relative_errors": rates})

    # The domain's own Time Scale.
    dom, _, scale = made["2x2x2.2"]
    use(dom)
    dom.modifiers["Fluid"].domain_settings.time_scale = 2.0
    bake(dom)
    ff.release_encode_cache()
    err, peak = values_error(dom, fluid_entry(), scale, 1.0, domain_time_scale=2.0)
    dh.record("D_domain_time_scale_on_U", err < 1e-6 and peak > 0.1,
              {"relative_error": err, "peak": peak})
    for dom, jet, _ in made.values():
        drop(jet, dom)

    # A moving domain, keyed LINEAR from frame 1 to 14.
    loc0, loc1 = np.array([0.2, 0.1, -0.1]), np.array([0.6, -0.1, 0.0])
    rot0, rot1 = np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.0, 0.6])
    mdom, mjet = gas_domain("FFEMove", tuple(loc0), frames=(1, 14),
                            keys={1: (tuple(loc0), tuple(rot0)),
                                  14: (tuple(loc1), tuple(rot1))})
    use(mdom)
    with ReadCount() as c1:
        fluid_entry()
    with ReadCount() as c2:
        v = fluid_entry()
    corner_err, axes_err, slip, matrix_err = 0.0, 0.0, 0.0, 0.0
    for k in range(FC):
        scene.frame_set(START + k)
        m = np.array(mdom.matrix_world, dtype=np.float64)
        matrix_err = max(matrix_err, float(np.abs(
            np.asarray(v["corner"][k]) - solver((m @ np.array([-1.0, -1.0, -1.0, 1.0]))[:3])).max()))
        t = (START + k - 1) / 13.0
        loc = loc0 + (loc1 - loc0) * t
        rot = euler_xyz(rot0 + (rot1 - rot0) * t)
        want_corner = solver(loc + rot @ np.array([-1.0, -1.0, -1.0]))
        want_axes = solver(rot.T).T
        corner_err = max(corner_err, float(np.abs(np.asarray(v["corner"][k]) - want_corner).max()))
        axes_err = max(axes_err, float(np.abs(np.asarray(v["axes"][k]) - want_axes).max()))
        if k:
            slip = max(slip, float(np.linalg.norm(np.asarray(v["corner"][k])
                                                  - np.asarray(v["corner"][k - 1]))))
    dh.record("D_moving_domain_per_frame_corner_and_axes",
              corner_err < 1e-6 and axes_err < 1e-6 and c1.n == FC and c2.n == FC,
              {"corner_error": corner_err, "axes_error": axes_err,
               "against_matrix_world": matrix_err, "one_frame_slip": slip,
               "reads": [c1.n, c2.n]})
    scene.frame_set(1)
    drop(mjet, mdom)


# ----- E: a sparse grid with a large active box -------------------------------

def E_sparse():
    span = (800, 400, 40)
    voxels = {}
    for c in range(8):
        ijk = tuple(span[a] if (c >> a) & 1 else 0 for a in range(3))
        voxels[ijk] = (1.0 + c, -0.5 * c, 0.25)
    for ijk in ((400, 200, 20), (401, 200, 20), (9, 300, 33), (700, 17, 5), (55, 55, 12)):
        voxels[ijk] = (0.5, 1.5, -2.0)
    sv = volume("FFESparse", vdb_file("sparse", [vec_grid("vel", voxels, voxel=0.01)]))
    use(sv)
    n = [s + 1 for s in span]
    image = float(np.prod(n)) * 16.0   # the box as float32, four channels
    bound = 0.1 * image
    want_bricks = len({(i // 8, j // 8, k // 8) for i, j, k in voxels})
    slots = int(np.prod([(x + 7) // 8 for x in n]))

    def measure():
        ff.release_encode_cache()
        # Leave tracing as it was found, should anything else trace.
        tracing = tracemalloc.is_tracing()
        if not tracing:
            tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            base = tracemalloc.get_traced_memory()[0]
            t0 = time.time()
            v = fluid_entry()
            took = time.time() - t0
            peak = tracemalloc.get_traced_memory()[1] - base
        finally:
            if not tracing:
                tracemalloc.stop()
        bricks = [len(b"".join(b)) // (512 * 16) for b in v["bricks"]]
        tables = [len(b"".join(t)) // 4 for t in v["tables"]]
        err = max(float(np.abs(cell_value(v, k, ijk)[:3] - value).max())
                  for k in (0, FC - 1) for ijk, value in voxels.items())
        ok = (peak < bound and all(list(r) == n for r in v["resolution"])
              and all(b == want_bricks for b in bricks) and all(t == slots for t in tables)
              and err < 1e-6)
        return ok, v, {"peak_bytes": int(peak), "bound_bytes": int(bound),
                       "bricks_per_frame": bricks[0], "want_bricks": want_bricks,
                       "table_slots": tables[0], "value_error": err, "seconds": round(took, 3)}

    ok, by_bricks, detail = measure()
    dh.record("E_sparse_box_read_in_bricks_memory_bound", ok, detail)
    saved = ff.SPARSE_VISIT_SECONDS
    ff.SPARSE_VISIT_SECONDS = 1e9    # never worth visiting: rows
    try:
        ok, by_rows, detail = measure()
    finally:
        ff.SPARSE_VISIT_SECONDS = saved
    same = (by_rows["tables"] == by_bricks["tables"] and by_rows["bricks"] == by_bricks["bricks"])
    detail["same_payload_as_bricks"] = same
    dh.record("E_sparse_box_read_in_rows_memory_bound", ok and same, detail)
    drop(sv)


# ----- F: chunks, the size report, the encode cache ---------------------------

def F_chunks_report_cache():
    shape = (64, 48, 48)
    arr = np.zeros(shape + (3,), dtype=np.float32)
    i, j, k = np.meshgrid(*(np.arange(s, dtype=np.float32) for s in shape), indexing="ij")
    arr[..., 0] = 1.0 + 0.01 * i
    arr[..., 1] = 0.5 + 0.02 * j
    arr[..., 2] = -0.25 + 0.01 * k
    arr[40:] = 0.0
    arr[63, 47, 47] = (1.0, 1.0, 1.0)
    occupied = np.any(arr.reshape(8, 8, 6, 8, 6, 8, 3) != 0.0, axis=(1, 3, 5, 6))
    kept = int(occupied.sum())
    slots = int(occupied.size)
    tag = "dense"
    path = vdb_file(tag, [vec_grid("vel", array=arr, voxel=0.05)])
    dv = volume("FFEDense", path)
    use(dv)
    ff.release_encode_cache()
    with ReadCount() as c1:
        tree1 = build()
    report1 = ff.volume_report()
    v = fluid_entry(tree1)

    # Chunks.
    sizes, kinds, counts_ok = [], set(), True
    for chunks in v["tables"] + v["bricks"]:
        total = sum(len(c) for c in chunks)
        kinds.update(type(c).__name__ for c in chunks)
        sizes.extend(len(c) for c in chunks)
        counts_ok &= len(chunks) == -(-total // MIB)
    brick_bytes = len(b"".join(v["bricks"][0]))
    err = rel_err(dense(v, 0)[..., :3], arr)
    dh.record("F_payload_bytes_are_chunked",
              kinds == {"bytes"} and max(sizes) <= MIB and counts_ok
              and len(v["bricks"][0]) >= 2 and brick_bytes == kept * 512 * 16 and err < 1e-6,
              {"chunk_types": sorted(kinds), "largest_chunk": max(sizes),
               "brick_chunks_per_frame": len(v["bricks"][0]), "brick_bytes": brick_bytes,
               "kept_bricks": kept, "value_error": err})

    # The size report.
    nbytes = sum(len(b"".join(t)) + len(b"".join(b)) for t, b in zip(v["tables"], v["bricks"]))
    entry = report1[0] if len(report1) == 1 else {}
    line = ff.volume_report_line(entry) if entry else ""
    want = {"field": "FFEFlow", "source": "FFEDense", "frames": FC,
            "resolution": list(shape), "bricks": FC * kept, "brick_slots": FC * slots,
            "bytes": nbytes, "reused": False}
    ok = (len(report1) == 1 and all(entry.get(key) == value for key, value in want.items())
          and abs(entry.get("occupancy", -1.0) - kept / slots) < 1e-12
          and all(s in line for s in ("FFEFlow", "FFEDense", "%d frames" % FC, "64x48x48")))
    dh.record("F_size_report", ok, {"report": report1, "want": want, "line": line})

    # The encode cache.
    steps = {"first_reads": c1.n}
    with ReadCount() as c2:
        tree2 = build()
    steps["second_reads"] = c2.n
    steps["second_reused"] = bool(ff.volume_report()[0]["reused"])
    steps["second_same"] = fluid_entry(tree2) == v
    with ReadCount() as c3:
        params_mod.compute_param_hash(bpy.context)
    steps["hash_reads"] = c3.n
    # The file rewritten (twice the velocity), with a new modification time.
    openvdb.write(path, grids=[vec_grid("vel", array=2.0 * arr, voxel=0.05)])
    later = os.stat(path).st_mtime_ns + 10 ** 9
    os.utime(path, ns=(later, later))
    with ReadCount() as c4:
        v4 = fluid_entry()
    steps["file_changed_reads"] = c4.n
    steps["file_changed_error"] = rel_err(dense(v4, 0)[..., :3], 2.0 * arr)
    dv.data.velocity_scale = 0.5
    with ReadCount() as c5:
        v5 = fluid_entry()
    steps["scale_changed_reads"] = c5.n
    steps["scale_changed_error"] = rel_err(dense(v5, 0)[..., :3], arr)
    state.time_scale = 0.5
    with ReadCount() as c6:
        v6 = fluid_entry()
    steps["time_scale_changed_reads"] = c6.n
    steps["time_scale_changed_error"] = rel_err(dense(v6, 0)[..., :3], 0.5 * arr)
    state.time_scale = 1.0
    dv.data.velocity_scale = 1.0
    dv.location = (0.0, 0.0, 0.0)
    dv.keyframe_insert("location", frame=1)
    dv.location = (0.5, 0.0, 0.0)
    dv.keyframe_insert("location", frame=13)
    with ReadCount() as c7:
        fluid_entry()
    with ReadCount() as c8:
        fluid_entry()
    steps["moving_reads"] = [c7.n, c8.n]
    steps["moving_reused"] = bool(ff.volume_report()[0]["reused"])
    dv.animation_data_clear()
    ok = (steps["first_reads"] == FC and steps["second_reads"] == 0 and steps["second_reused"]
          and steps["second_same"] and steps["hash_reads"] == 0
          and steps["file_changed_reads"] == FC and steps["file_changed_error"] < 1e-6
          and steps["scale_changed_reads"] == FC and steps["scale_changed_error"] < 1e-6
          and steps["time_scale_changed_reads"] == FC
          and steps["time_scale_changed_error"] < 1e-6
          and steps["moving_reads"] == [FC, FC] and not steps["moving_reused"])
    dh.record("F_encode_cache_reuse", ok, steps)
    drop(dv)

    # A Volume whose grids were never loaded: the encode's own read renames
    # its Velocity Grid, and the encode after must still be reused.
    fresh = bpy.data.volumes.new("FFEFresh")
    fresh.filepath = vdb_file("fresh", [vec_grid("vel", {(0, 0, 0): (1.0, 0.0, 0.0)})])
    fo = bpy.data.objects.new("FFEFresh", fresh)
    scene.collection.objects.link(fo)
    use(fo)
    named_before = fresh.velocity_grid
    with ReadCount() as f1:
        fluid_entry()
    named_after = fresh.velocity_grid
    with ReadCount() as f2:
        fluid_entry()
    dh.record("F_a_renamed_velocity_grid_is_reused",
              named_before == "velocity" and named_after == "vel" and f1.n == FC
              and f2.n == 0,
              {"named_before": named_before, "named_after": named_after,
               "first_reads": f1.n, "second_reads": f2.n})
    drop(fo)


# ----- G: a non-uniformly scaled Volume, honored per axis ----------------------

def G_volume_per_axis():
    lo, hi = np.array([1, 2, 3]), np.array([4, 5, 7])
    voxels = {}
    for c in range(8):
        ijk = tuple(int(hi[a] if (c >> a) & 1 else lo[a]) for a in range(3))
        voxels[ijk] = (1.0 + 0.5 * c, -1.0, 0.25 * c)
    gv = volume("FFEPerAxis", vdb_file("per_axis", [vec_grid("vel", voxels, voxel=0.2)]))
    gv.location = (0.5, -0.5, 0.2)
    gv.rotation_euler = (0.3, 0.0, 0.5)
    gv.scale = (1.0, 2.0, 0.5)
    bpy.context.view_layer.update()
    use(gv)
    v = fluid_entry()
    vol = payload_volume(v)
    m = np.array(gv.matrix_world, dtype=np.float64)

    def world(index_point):
        return (m @ np.append(0.2 * np.asarray(index_point, dtype=np.float64), 1.0))[:3]

    cell = np.asarray(v["cell"][0], dtype=np.float64)
    corner = np.asarray(v["corner"][0], dtype=np.float64)
    cell_err = rel_err(cell, [0.2, 0.4, 0.1])
    ok_place = (cell_err < 1e-6
                and float(np.abs(corner - solver(world(lo - 0.5))).max()) < 1e-9)
    # Values at voxel centers: the file's velocity turned and scaled by the
    # object, in solver axes.
    pts = np.array([world(ijk) for ijk in voxels])
    u, _, _ = fe.fluid_frame(vol, 0, solver(pts))
    want = solver(np.array([m[:3, :3] @ np.array(value) for value in voxels.values()]))
    err = rel_err(u, want)
    # Presence a sixth of a voxel inside and outside every face of the box.
    mid = 0.5 * (lo + hi)
    face_pts, inside = [], []
    for a in range(3):
        for face, outward in ((lo[a] - 0.5, -1.0), (hi[a] + 0.5, 1.0)):
            for sign, inner in ((-1.0, True), (1.0, False)):
                p = mid.astype(np.float64).copy()
                p[a] = face + outward * sign / 6.0
                face_pts.append(world(p))
                inside.append(inner)
    _, _, presence = fe.fluid_frame(vol, 0, solver(np.array(face_pts)))
    ok_presence = bool(np.all((presence > 0.5) == np.array(inside)))
    dh.record("G_volume_per_axis_cells_and_presence",
              ok_place and err < 1e-5 and ok_presence,
              {"cell": cell.tolist(), "cell_error": cell_err, "corner_error": float(np.abs(
                  corner - solver(world(lo - 0.5))).max()), "value_error": err,
               "presence": presence.tolist(), "want_inside": inside})
    drop(gv)


# ----- H: what was baked, the Cache Offset, and a domain the view layer skips

def H_bake_offset_and_visibility():
    # The jet is unparented before the bake: a visible child pulls its
    # parent into the view layer's evaluation even while the parent is
    # hidden (measured, and held below with FFEDet, whose jet is parented).
    dom, jet = gas_domain("FFEBake", (-8.0, -8.0, 0.0), res=12, frames=(1, 12),
                          bake_it=False)
    at = jet.matrix_world.copy()
    jet.parent = None
    jet.matrix_world = at
    bake(dom)
    ds = dom.modifiers["Fluid"].domain_settings
    use(dom)
    unit = 2.5 * 2.0 / 12.0   # 2.5 cells of the longest axis per grid unit

    # A DOMAIN THE VIEW LAYER DOES NOT EVALUATE reads a (0, 0, 0) grid
    # (measured), so each such state is refused by its cause, also with the
    # encode cache warm from a read while it was visible; the eye icon and
    # Disable in Renders leave it evaluated, and it reads. So does FFEDet
    # disabled in viewports, which its visible parented jet keeps evaluated:
    # it reads the grid it reads when visible, and is not refused.
    ff.release_encode_cache()
    visible = fluid_entry()
    mod = dom.modifiers["Fluid"]
    coll = bpy.data.collections.new("FFEHidden")
    scene.collection.children.link(coll)
    homes = list(dom.users_collection)
    for c in homes:
        c.objects.unlink(dom)
    coll.objects.link(dom)
    layer = bpy.context.view_layer.layer_collection.children["FFEHidden"]
    cases = (
        ("fluid_modifier_off_in_viewport", lambda on: setattr(mod, "show_viewport", not on),
         "has its Fluid modifier off in the viewport", "turn on the modifier's Show in Viewport"),
        ("disabled_in_viewports", lambda on: setattr(dom, "hide_viewport", on),
         "is not evaluated in this view layer", "enable it in the viewport"),
        ("collection_disabled_in_viewports", lambda on: setattr(coll, "hide_viewport", on),
         "is not evaluated in this view layer", "enable it in the viewport"),
        ("collection_excluded", lambda on: setattr(layer, "exclude", on),
         "is not evaluated in this view layer", "enable it in the viewport"),
    )
    rows, ok = {}, True
    try:
        for label, toggle, cause, remedy in cases:
            toggle(True)
            try:
                bpy.context.view_layer.update()
                listed = ff.refusal(flow) or ""
                # The status listing adds no "holds no fluid; bake it" of its
                # own: the empty grid comes from the view layer, not the bake.
                info = ff_status.fluid_info(flow, scene, state,
                                            bpy.context.evaluated_depsgraph_get())
                good, msg, missing = refused(FIELD, "'FFEBake'", cause, remedy)
                rows[label] = {"transfer": msg, "listed": listed, "missing": missing,
                               "fluid_problem": info["problem"], "fluid_note": info["note"]}
                ok = (ok and good and cause in listed and "not baked" not in msg
                      and info["problem"] is None and info["note"] is None)
            finally:
                toggle(False)
                bpy.context.view_layer.update()
        dom.hide_set(True)
        dom.hide_render = True
        bpy.context.view_layer.update()
        ff.release_encode_cache()
        hidden = refused()
        same = hidden[1] == "no error" and fluid_entry() == visible
        rows["eye_icon_and_render_hidden"] = {"transfer": hidden[1], "reads_the_same": same}
        ok = ok and same
    finally:
        dom.hide_set(False)
        dom.hide_render = False
        coll.objects.unlink(dom)
        for c in homes:
            c.objects.link(dom)
        bpy.data.collections.remove(coll)
        bpy.context.view_layer.update()
    rows["visible_again"] = refused()[1]
    use(det)
    ff.release_encode_cache()
    det_visible = fluid_entry()
    det.hide_viewport = True
    try:
        bpy.context.view_layer.update()
        ff.release_encode_cache()
        kept = refused()
        rows["parent_of_a_visible_child"] = {
            "transfer": kept[1], "listed": ff.refusal(flow),
            "reads_the_same": kept[1] == "no error" and fluid_entry() == det_visible}
    finally:
        det.hide_viewport = False
        bpy.context.view_layer.update()
    ok = (ok and rows["visible_again"] == "no error"
          and rows["parent_of_a_visible_child"]["reads_the_same"])
    dh.record("H_a_domain_the_view_layer_skips_is_refused_by_cause", ok, rows)
    use(dom)

    # CACHE OFFSET 3 plays cache frames 1..12 at scene frames 4..15
    # (measured), so the solve's 3..12 starts before it and is refused with
    # the frames it plays, while a solve over 4..13 reads the grid Blender
    # plays at each of those scene frames.
    ds.cache_frame_offset = 3
    dom.update_tag()
    bpy.context.view_layer.update()
    before = refused(FIELD, "'FFEBake'", "is baked over frames 1..12",
                     "its Cache Offset 3 plays at frames 4..15",
                     "the solve reads frames 3..12", "bake it over frames 0..9")
    state.frame_start = 4
    errors, peak, payload = [], 0.0, None
    try:
        ff.release_encode_cache()
        payload = fluid_entry()
        for k in range(FC):
            n, grid, _ = domain_grid(dom, 4 + k)
            got = dense(payload, k)[..., :3]
            errors.append(rel_err(got, grid * unit) if got.shape[:3] == tuple(n) else 1.0)
            peak = max(peak, float(np.abs(grid).max()) * unit)
        # The Visualize cache keys a frame on the files of the cache frame it
        # plays: scene frame 5 on cache frame 2's.
        files = ff._source_state(dom, 5)[-1]
        keyed = (any("_0002." in e[0] for e in files)
                 and not any("_0005." in e[0] for e in files))
        msg = "no error"
    except ValueError as e:
        msg, keyed = str(e), False
    state.frame_start = START
    dh.record("H_cache_offset_is_honored",
              before[0] and msg == "no error" and len(errors) == FC
              and max(errors) < 1e-6 and peak > 0.1 and keyed,
              {"before_it_plays": before[1], "missing": before[2], "encode": msg,
               "relative_errors": errors, "peak": peak, "visualize_key_on_cache_frame": keyed})
    ds.cache_frame_offset = 0
    dom.update_tag()
    bpy.context.view_layer.update()

    # A FRAME THE BAKE DID NOT REACH, inside the cache's range: End raised
    # from 8 to 12 after a bake over 1..8, which once the file is reopened
    # reads frame 8's grid at frames 9 to 12 (measured). A domain of its own,
    # baked once (A_unbaked_domain_refused_by_name says why).
    drop(jet, dom)
    dom, jet = gas_domain("FFEBake8", (-8.0, -8.0, 0.0), res=12, frames=(1, 8))
    ds = dom.modifiers["Fluid"].domain_settings
    use(dom)
    ds.cache_frame_end = 12
    # Blender 5.2 leaves frame 9's file off the disk, so the file check names
    # it; Blender 5.1 (measured on Windows) does not, and the frame then
    # reads no grid, which the read refuses. Either way frame 9 is refused by
    # name with the bake remedy, and nothing ships frame 8's grid for it.
    raised = refused(FIELD, "'FFEBake8'", "frame 9", "bake it over the solve frames")
    if raised[0]:
        files_gone = ("has no baked data for frame 9" in raised[1]
                      and "'fluid_data_0009.vdb'" in raised[1])
        read_empty = "at frame 9" in raised[1] and "holds no fluid" in raised[1]
        if not (files_gone or read_empty):
            raised = (False, raised[1], ["either the missing file or the empty read"])
    # A FILE REMOVED from inside the bake, frame 5's: the first frame the
    # solve would miss is now that one.
    data_dir = os.path.join(bpy.path.abspath(ds.cache_directory), "data")
    os.remove(os.path.join(data_dir, "fluid_data_0005.vdb"))
    removed = refused(FIELD, "'FFEBake8'", "has no baked data for frame 5",
                      "'fluid_data_0005.vdb'")
    dh.record("H_a_frame_missing_from_the_bake_is_refused_by_name",
              raised[0] and removed[0],
              {"end_raised_past_the_bake": raised[1], "file_removed": removed[1],
               "missing": raised[2] + removed[2]})
    drop(jet, dom)


# ----- I: Blender itself ignores Gravitation on a Fluid Flow -----------------

def I_blender_ignores_gravitation():
    # Particles at rest in the jet of FFEDet (baked over 2..13), pushed by
    # the Fluid Flow alone from frame 3 to 8, walked with Gravitation off and
    # on: Blender moves them identically, which is why the add-on sends it
    # off (B_gravitation_on_fluid_flow_is_sent_off). Noise Amount, which the
    # add-on refuses, is the control that the walk sees a setting at all.
    use(det)
    flow.field.flow = 1.0
    # Noise Amount's pattern follows the seed, so the control is the same on
    # every run.
    flow.field.seed = 7
    pts = [(x, 8.0 + y, 0.0) for x in (-0.3, 0.0, 0.3) for y in (-0.15, 0.0, 0.15)]
    me = bpy.data.meshes.new("FFEPts")
    me.from_pydata(pts, [], [])
    em = bpy.data.objects.new("FFEPts", me)
    scene.collection.objects.link(em)
    em.modifiers.new("ps", "PARTICLE_SYSTEM")
    ps = em.particle_systems[0].settings
    ps.count = len(pts)
    ps.frame_start = ps.frame_end = 3
    ps.lifetime = 200
    ps.emit_from = "VERT"
    ps.use_emit_random = False
    ps.normal_factor = 0.0
    ps.physics_type = "NEWTON"
    ps.mass = 1.0
    ps.effector_weights.gravity = 0.0

    def walk():
        with bpy.context.temp_override(scene=scene):
            bpy.ops.ptcache.free_bake_all()
        for frame in range(1, 9):
            scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        out = np.array([tuple(q.location) for q in em.evaluated_get(dg).particle_systems[0]
                        .particles], dtype=np.float64)
        scene.frame_set(1)
        return out

    try:
        off = walk()
        flow.field.use_gravity_falloff = True
        on = walk()
        flow.field.use_gravity_falloff = False
        flow.field.noise = 2.0
        noisy = walk()
        flow.field.noise = 0.0
        moved = float(np.abs(off - np.array(pts)).max())
        gravitation = float(np.abs(on - off).max())
        noise = float(np.abs(noisy - off).max())
        dh.record("I_blender_ignores_gravitation_on_a_fluid_flow",
                  len(off) == len(pts) and moved > 0.01 and gravitation == 0.0
                  and noise > 0.1 * moved,
                  {"moved": moved, "gravitation_changes_them_by": gravitation,
                   "noise_changes_them_by": noise})
    finally:
        bpy.data.objects.remove(em)
        bpy.data.meshes.remove(me)


# ----- J: the transfer cap over the whole upload ------------------------------

def J_upload_cap():
    # The cap is the server's 4 GiB, lowered here in-process so a scene of a
    # few megabytes stands in for one past it. The encode's own check counts
    # the fluid volumes and images it sends, reused ones included; the
    # upload's counts the scene geometry and every parameter with them, and
    # names its total and its parts.
    enc = __import__(pkg + ".core.encoder", fromlist=["prepare_upload"])
    solver_ui = __import__(pkg + ".ui.solver", fromlist=["SOLVER_OT_Transfer"])
    async_op = __import__(pkg + ".core.async_op", fromlist=["StageAbort"])
    use(det)
    ff.release_encode_cache()
    build()
    fluid_bytes = int(ff.volume_report()[0]["bytes"])
    img = bpy.data.images.new("FFEMap", 64, 64, alpha=True, float_buffer=True)
    tex = bpy.data.textures.new("FFEMapTex", "IMAGE")
    tex.image = img
    bpy.ops.object.effector_add(type="TEXTURE", location=(0.0, 0.0, 26.0))
    tf = bpy.context.object
    tf.name = "FFETex"
    tf.field.texture = tex
    image_bytes = 64 * 64 * 3 * 4
    cap = ff.TRANSPORT_CAP_BYTES
    rows = {}
    try:
        data, param = enc.prepare_upload(bpy.context)[:2]
        total = len(data) + len(param)
        ff.TRANSPORT_CAP_BYTES = fluid_bytes + image_bytes - 1
        rows["reused_volume_and_image"] = refused(
            "a transfer carries", "Texture 'FFETex', image 'FFEMap'",
            "Fluid Flow 'FFEFlow', fluid of 'FFEDet'")
        ff.release_encode_cache()
        rows["read_volume_and_image"] = refused("a transfer carries", FIELD)
        ff.TRANSPORT_CAP_BYTES = fluid_bytes + image_bytes + 1
        rows["volumes_and_images_under_it"] = refused()
        ff.TRANSPORT_CAP_BYTES = total - 1
        try:
            enc.prepare_upload(bpy.context)
            msg = "no error"
        except ValueError as e:
            msg = str(e)
        want = ["The upload comes to " + ff._past_cap(total),
                "scene geometry " + ff._size(len(data)), "parameters " + ff._size(len(param)),
                "Fluid Flow 'FFEFlow'", "Texture 'FFETex'"]
        rows["prepare_upload"] = (all(w in msg for w in want), msg,
                                  [w for w in want if w not in msg])

        class Stage:
            def __init__(self):
                self._payload = {"data": data, "data_hash": ""}

            def report(self, kind, text):
                pass

        try:
            solver_ui.SOLVER_OT_Transfer._stage_encode_params(Stage(), bpy.context)
            msg = "no error"
        except async_op.StageAbort as e:
            msg = str(e)
        rows["transfer_stage"] = ("The upload comes to " + ff._past_cap(total) in msg, msg, [])
        ff.TRANSPORT_CAP_BYTES = total
        try:
            enc.prepare_upload(bpy.context)
            rows["at_the_cap"] = (True, "no error", [])
        except ValueError as e:
            rows["at_the_cap"] = (False, str(e), [])
    finally:
        ff.TRANSPORT_CAP_BYTES = cap
        bpy.data.objects.remove(tf)
        bpy.data.textures.remove(tex)
        bpy.data.images.remove(img)
    ok = (rows["reused_volume_and_image"][0] and rows["read_volume_and_image"][0]
          and rows["volumes_and_images_under_it"][1] == "no error"
          and rows["prepare_upload"][0] and rows["transfer_stage"][0] and rows["at_the_cap"][0])
    dh.record("J_the_cap_counts_the_whole_upload", ok,
              {"fluid_bytes": fluid_bytes, "image_bytes": image_bytes,
               "upload_bytes": [len(data), len(param)],
               "rows": {k: {"error": v[1], "missing": v[2]} for k, v in rows.items()}})


# ----- K: the size report belongs to the file that is open ---------------------

def K_report_cleared_on_load():
    # Last, since it opens another file: what the last encode read describes
    # sources of the file that was open, so opening a file clears it.
    use(det)
    ff.release_encode_cache()
    build()
    had = ff.volume_report()
    path = os.path.join(PROBE_DIR, "ffe_reopen.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
    bpy.ops.wm.open_mainfile(filepath=path)
    after = ff.volume_report()
    dh.record("K_the_volume_report_is_cleared_when_a_file_opens",
              len(had) == 1 and after == [],
              {"before": [r.get("source") for r in had], "after": after})


# ----- L: a domain linked from a library, and a sheared Use Coordinates -------

def L_linked_and_sheared():
    # A baked domain LINKED from a library .blend in another folder, its cache
    # directory relative ("//"), which Blender resolves against the LIBRARY.
    # The library is made and baked by a child Blender, so this session's
    # state is untouched.
    import subprocess

    lib_dir = folder("library")
    lib_path = os.path.join(lib_dir, "fluid_library.blend")
    script = "\n".join([
        "import bpy",
        "for o in list(bpy.data.objects): bpy.data.objects.remove(o)",
        "sc = bpy.context.scene",
        "sc.render.fps = %d" % SCENE_FPS,
        "bpy.ops.mesh.primitive_cube_add(size=2.0)",
        "dom = bpy.context.object; dom.name = 'FFELinkedDomain'",
        "m = dom.modifiers.new('Fluid', 'FLUID'); m.fluid_type = 'DOMAIN'",
        "ds = m.domain_settings; ds.domain_type = 'GAS'; ds.resolution_max = 16",
        "ds.alpha = ds.beta = 0.0; ds.gravity = (0, 0, 0); ds.use_adaptive_timesteps = False",
        "ds.cache_type = 'ALL'; ds.cache_data_format = 'OPENVDB'",
        "ds.cache_frame_start, ds.cache_frame_end = %d, %d" % (START - 1, START + FC),
        "bpy.ops.mesh.primitive_cube_add(size=0.4, location=(-0.6, 0, 0))",
        "jet = bpy.context.object; f = jet.modifiers.new('Fluid', 'FLUID'); f.fluid_type = 'FLOW'",
        "fl = f.flow_settings; fl.flow_type = 'SMOKE'; fl.flow_behavior = 'INFLOW'",
        "fl.use_initial_velocity = True; fl.velocity_coord = (1.5, 0.3, 0.0)",
        "bpy.ops.wm.save_as_mainfile(filepath=%r)" % lib_path,
        "ds.cache_directory = '//library_cache'",
        "with bpy.context.temp_override(scene=sc, object=dom, active_object=dom, selected_objects=[dom]):",
        "    assert bpy.ops.fluid.bake_all() == {'FINISHED'}",
        "bpy.ops.wm.save_mainfile()",
    ])
    child = subprocess.run([bpy.app.binary_path, "-b", "--factory-startup", "--python-expr", script],
                           capture_output=True, text=True, timeout=600)
    baked = os.path.isdir(os.path.join(lib_dir, "library_cache", "data"))
    with bpy.data.libraries.load(lib_path, link=True) as (data_from, data_to):
        data_to.objects = ["FFELinkedDomain"]
    linked = data_to.objects[0] if data_to.objects else None
    detail = {"child_rc": child.returncode, "baked": baked,
              "child_tail": (child.stdout + child.stderr)[-600:]}
    ok = linked is not None and baked and child.returncode == 0
    if ok:
        scene.collection.objects.link(linked)
        use(linked)
        why = ff.refusal(flow)
        detail["refusal"] = why
        ok = why is None
        if ok:
            v = fluid_entry()
            peaks = [float(np.abs(dense(v, k)[..., :3]).max()) for k in range(len(v["tables"]))]
            detail["frames"] = len(peaks)
            detail["peak"] = max(peaks)
            ok = len(peaks) == FC and max(peaks) > 0.1
        use(None)
        scene.collection.objects.unlink(linked)
    dh.record("L_a_linked_domain_reads_its_librarys_cache", ok, detail)

    # A Texture field with Use Coordinates under a rotated, non-uniformly
    # scaled parent: its world matrix is sheared, which the record (a rotation
    # and a scale per axis) cannot carry, so it is refused by name; unsheared
    # or in world coordinates it is sent.
    bpy.ops.object.empty_add(location=(0.0, -6.0, 0.0))
    parent = bpy.context.object
    parent.name = "FFEShearParent"
    parent.scale = (2.0, 0.6, 1.0)
    bpy.ops.object.effector_add(type="TEXTURE", location=(0.0, -6.0, 0.0))
    tex_field = bpy.context.object
    tex_field.name = "FFEShearTexture"
    img = bpy.data.images.new("ffe_shear", 4, 4, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    img.pixels = [0.5, 0.25, 0.75, 1.0] * 16
    tex = bpy.data.textures.new("ffe_shear", "IMAGE")
    tex.image = img
    tex.use_interpolation = False
    tex_field.field.texture = tex
    tex_field.field.use_object_coords = True
    tex_field.parent = parent
    tex_field.rotation_euler = (0.2, -0.1, 0.7)
    bpy.context.view_layer.update()
    sheared = ff.refusal(tex_field)
    tex_field.field.use_object_coords = False
    world = ff.refusal(tex_field)
    tex_field.field.use_object_coords = True
    tex_field.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    aligned = ff.refusal(tex_field)
    dh.record("L_a_sheared_use_coordinates_texture_is_refused_by_name",
              sheared is not None and "sheared" in sheared and "Use Coordinates" in sheared
              and world is None and aligned is None,
              {"sheared": sheared, "world_coordinates": world, "unsheared": aligned})
    bpy.data.objects.remove(tex_field)
    bpy.data.objects.remove(parent)


for _section in (A_determinism, B_refusals, C_volume_sources, D_domains, E_sparse,
                 F_chunks_report_cache, G_volume_per_axis, H_bake_offset_and_visibility,
                 I_blender_ignores_gravitation, J_upload_cap, L_linked_and_sheared,
                 K_report_cleared_on_load):
    dh.log(_section.__name__)
    section(_section)
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 900.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
