# File: sand_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Convert a closed solid MESH into a "particle mesh": a faceless mesh of
# N loose vertices (grain centers) plus a Geometry-Nodes "Particle Mesh"
# modifier that renders each vertex as a sphere (Mesh to Points). This is
# the SAND group's input representation. The seeding logic (BVH
# closest-surface erosion) mirrors the validated prototype.

import math

import bmesh  # pyright: ignore
import bpy  # pyright: ignore
import numpy as np
from bpy.app.translations import pgettext_iface as iface_, pgettext_tip as tip_
from bpy.types import Operator  # pyright: ignore
from mathutils import Vector  # pyright: ignore
from mathutils.bvhtree import BVHTree  # pyright: ignore


# Poisson-disk seeding constants. Grains are placed by blue-noise elimination
# with a minimum center-to-center separation of
# 2*radius*(1 + POISSON_MARGIN) + extra_spacing, so two grains never start
# inside each other's contact radius (the solver's barrier is undefined there
# and an overlapping cloud is rejected at init) and the user controls density
# directly via extra_spacing. Candidates come from a jittered grid CAND_PER_CELL
# times finer than the separation along each axis, which gives the elimination
# enough well-distributed candidates to pack the volume fully without the huge
# counts random dart-throwing would need. MAX_CANDIDATES caps the cost.
POISSON_MARGIN = 0.05
CAND_PER_CELL = 2.5
MAX_CANDIDATES = 2_500_000


def seed_inside_eroded(obj, radius, extra_spacing, rng_seed=0):
    """Fill ``obj`` with non-overlapping grains of ``radius`` at a chosen spacing.

    Grains are placed by Poisson-disk (blue-noise) elimination with a minimum
    center-to-center separation of ``2*radius*(1 + POISSON_MARGIN) +
    extra_spacing``: ``extra_spacing == 0`` packs them as tightly as the
    non-overlap rule allows, larger values give a looser, sparser cloud. There
    is no grid-alignment artifact, and the grain count is simply whatever fills
    the volume at that separation. A BVH closest-surface sign test with an
    erosion margin of ``radius`` keeps every grain sphere inside the surface.
    Returns ``(cloud, stats)`` where ``cloud`` is an (N, 3) float32 array of
    local-space positions.
    """
    dg = bpy.context.evaluated_depsgraph_get()
    # Seed in WORLD space. The solver consumes world-space vertices and a world
    # contact offset, so the non-overlap separation must hold AFTER the object's
    # transform is applied. Seeding in local space lets a non-unit (or
    # non-uniform) object scale shrink the world spacing below 2*radius, so the
    # cloud is rejected at init. Building the BVH and candidate grid from
    # world-space geometry makes r_min and the erosion correct for any scale;
    # the kept points are mapped back to local at the end so the object's
    # matrix_world re-applies to land them exactly where they were seeded.
    world = obj.matrix_world
    bm = bmesh.new()
    bm.from_object(obj, dg)
    bm.transform(world)
    bvh = BVHTree.FromBMesh(bm)
    co = (
        np.array([v.co[:] for v in bm.verts], np.float32)
        if bm.verts
        else np.zeros((0, 3), np.float32)
    )
    bm.free()
    co = co.reshape(-1, 3)
    mn = co.min(0)
    mx = co.max(0)
    Vbox = float(np.prod(mx - mn))

    # Probe a point far outside the bounding box to learn the surface
    # orientation. If the nearest normal points away from the probe the
    # mesh has inverted normals; the sign flip keeps the inside test valid.
    far = Vector(
        (
            float(mx[0]) + (float(mx[0]) - float(mn[0])) + 1.0,
            (float(mn[1]) + float(mx[1])) * 0.5,
            (float(mn[2]) + float(mx[2])) * 0.5,
        )
    )
    loc, nrm, idx, d = bvh.find_nearest(far)
    sign = -1.0 if (far - loc).dot(nrm) < 0 else 1.0
    fn = bvh.find_nearest

    def inside_mask(pts):
        """BVH sign test + erosion: keep grains whose whole sphere is inside."""
        keep = np.zeros(len(pts), bool)
        for i in range(len(pts)):
            p = pts[i]
            lc, nr, ix, dd = fn((float(p[0]), float(p[1]), float(p[2])))
            if lc is None:
                continue
            if (
                sign * ((p[0] - lc[0]) * nr[0] + (p[1] - lc[1]) * nr[1] + (p[2] - lc[2]) * nr[2]) < 0
                and dd >= radius
            ):
                keep[i] = True
        return keep

    rng = np.random.default_rng(rng_seed)

    # Minimum grain separation: the non-overlap floor plus the user's extra gap.
    r_min = 2.0 * radius * (1.0 + POISSON_MARGIN) + max(float(extra_spacing), 0.0)

    # Candidate cloud: a jittered grid finer than r_min, so the elimination
    # below has enough well-distributed candidates to pack the volume fully
    # (far fewer than random dart-throwing needs for the same fill).
    g = r_min / CAND_PER_CELL
    axes = [np.arange(mn[i] + 0.5 * g, mx[i], g) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    cand = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1).astype(np.float32)
    if len(cand) > MAX_CANDIDATES:
        cand = cand[rng.choice(len(cand), MAX_CANDIDATES, replace=False)]
    cand += (rng.random(cand.shape, np.float32) - 0.5) * g

    # Poisson-disk elimination: visit candidates in random order; keep one only
    # if no already-kept candidate lies within r_min, and reject every
    # undecided candidate inside its r_min ball. The kept set is a blue-noise
    # sample with guaranteed minimum separation r_min. mathutils' KD-tree does
    # the neighbor search in C so the Python loop is one pass over the candidates.
    from mathutils.kdtree import KDTree

    n_cand = len(cand)
    kd = KDTree(n_cand)
    for i in range(n_cand):
        kd.insert((float(cand[i, 0]), float(cand[i, 1]), float(cand[i, 2])), i)
    kd.balance()

    state = np.zeros(n_cand, np.int8)  # 0 undecided, 1 kept, -1 rejected
    kept = []
    for i in rng.permutation(n_cand):
        if state[i]:
            continue
        state[i] = 1
        kept.append(int(i))
        for _co, j, _d in kd.find_range(
            (float(cand[i, 0]), float(cand[i, 1]), float(cand[i, 2])), r_min
        ):
            if state[j] == 0:
                state[j] = -1
    pts = cand[kept]

    pts = pts[inside_mask(pts)]
    # Map the world-space seed points back to local so obj.data stores local
    # vertices; the encoder re-applies obj.matrix_world to recover the exact
    # world positions (with their guaranteed >= r_min world separation).
    winv = np.array(world.inverted())
    local = pts.astype(np.float64) @ winv[:3, :3].T + winv[:3, 3]
    return local.astype(np.float32), dict(
        inverted=sign < 0,
        candidates=n_cand,
        kept=len(kept),
        final=len(local),
        r_min=r_min,
    )


def _build_sand_material():
    """Create (or reuse) the PPF_Sand material with per-grain color variation."""
    mat = bpy.data.materials.get("PPF_Sand")
    if mat is not None:
        return mat
    mat = bpy.data.materials.new("PPF_Sand")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        return mat
    bsdf.inputs["Roughness"].default_value = 0.85

    # Point Info > Random -> Color Ramp -> Base Color drives per-grain
    # color variation so the cloud does not look like a flat blob.
    pinfo = nt.nodes.new("GeometryNodePointInfo") if hasattr(
        bpy.types, "GeometryNodePointInfo"
    ) else None
    if pinfo is None:
        # Point Info is a geometry-node-only node; in the shader graph the
        # equivalent per-instance random is an Object Info > Random output.
        # Fall back to that so the material still varies per grain.
        pinfo = nt.nodes.new("ShaderNodeObjectInfo")
        random_socket = pinfo.outputs.get("Random")
    else:
        random_socket = pinfo.outputs.get("Random")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    el = ramp.color_ramp.elements
    el[0].color = (0.72, 0.55, 0.32, 1.0)
    el[1].color = (0.88, 0.74, 0.46, 1.0)
    if random_socket is not None:
        nt.links.new(random_socket, ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _build_particle_mesh_node_group(grain_radius, material):
    """Create (or rebuild) the PPF_ParticleMesh geometry node group."""
    ng = bpy.data.node_groups.get("PPF_ParticleMesh")
    if ng is not None:
        bpy.data.node_groups.remove(ng)
    ng = bpy.data.node_groups.new("PPF_ParticleMesh", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nd = ng.nodes
    lk = ng.links
    gi = nd.new("NodeGroupInput")
    go = nd.new("NodeGroupOutput")
    m2p = nd.new("GeometryNodeMeshToPoints")
    sm = nd.new("GeometryNodeSetMaterial")
    for sset in m2p.inputs:
        if sset.name == "Radius" and hasattr(sset, "default_value"):
            sset.default_value = grain_radius
    for sset in sm.inputs:
        if sset.name == "Material":
            sset.default_value = material
    lk.new(gi.outputs[0], m2p.inputs[0])
    lk.new(m2p.outputs[0], sm.inputs[0])
    lk.new(sm.outputs[0], go.inputs[0])
    return ng


def build_and_commit_particle_mesh(obj, radius, extra_spacing=0.0, rng_seed=0):
    """Seed grains inside ``obj`` and commit the particle-mesh representation.

    Headless-callable (no operator context, no dialog): the test rig calls
    this directly. ``radius`` is the user-chosen grain radius and the single
    locked source of truth for both the rendered sphere and the contact skin;
    ``extra_spacing`` is the gap added beyond touching (0 = densest). The grain
    count is whatever fills the volume at that separation. Replaces ``obj.data``
    with a faceless mesh of N loose vertices, adds the "Particle Mesh"
    geometry-nodes modifier, and stamps the ``ppf_particle_mesh`` /
    ``ppf_seed_count`` / ``ppf_grain_radius`` custom properties. Returns N.
    """
    cloud, _stats = seed_inside_eroded(obj, radius, extra_spacing, rng_seed)
    grain_radius = float(radius)

    nm = bpy.data.meshes.new(obj.name + "_particles")
    n = len(cloud)
    nm.vertices.add(n)
    nm.vertices.foreach_set("co", cloud.ravel())
    nm.update()
    old = obj.data
    obj.data = nm
    if old.users == 0:
        bpy.data.meshes.remove(old)

    mat = _build_sand_material()
    ng = _build_particle_mesh_node_group(grain_radius, mat)
    mod = obj.modifiers.get("Particle Mesh")
    if mod is None or mod.type != "NODES":
        mod = obj.modifiers.new("Particle Mesh", "NODES")
    mod.node_group = ng

    obj["ppf_particle_mesh"] = 1
    obj["ppf_seed_count"] = n
    obj["ppf_grain_radius"] = grain_radius
    return n


def _is_convertible_solid_mesh(obj):
    """True if ``obj`` is a SELECTED closed-ish solid MESH eligible for conversion.

    Requires the object to be selected (not merely the lingering active object),
    a MESH with faces, and not already a particle mesh. ``select_get`` is guarded
    so an object outside the active view layer never raises.
    """
    if obj is None or obj.type != "MESH":
        return False
    try:
        selected = obj.select_get()
    except RuntimeError:
        selected = False
    return (
        selected
        and len(obj.data.polygons) > 0
        and not obj.get("ppf_particle_mesh")
    )


class OBJECT_OT_ConvertToParticleMesh(Operator):
    """Replace this solid mesh with a faceless cloud of loose vertices
    (grain centers) plus a render-only Particle Mesh modifier. Destructive:
    the original faces are discarded."""

    bl_idname = "ppf.convert_to_particle_mesh"
    bl_label = "Convert To Solid Particle Mesh"
    bl_options = {"REGISTER", "UNDO"}

    grain_radius: bpy.props.FloatProperty(
        name="Grain Radius",
        description=(
            "Physical grain radius (also the contact skin). Locked after "
            "conversion, since the non-overlapping spacing is derived from it."
        ),
        default=0.05,
        min=1e-5,
        soft_max=1.0,
        precision=5,
        unit="LENGTH",
    )  # pyright: ignore
    extra_spacing: bpy.props.FloatProperty(
        name="Extra Spacing",
        description=(
            "Gap added between grains beyond touching. 0 packs them as densely "
            "as non-overlap allows; larger values give a looser, sparser cloud. "
            "The grain count is whatever fills the volume at this spacing."
        ),
        default=0.0,
        min=0.0,
        soft_max=1.0,
        precision=5,
        unit="LENGTH",
    )  # pyright: ignore
    rng: bpy.props.IntProperty(
        name="Random Seed",
        default=0,
    )  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return _is_convertible_solid_mesh(context.active_object)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "grain_radius")
        layout.prop(self, "extra_spacing")
        layout.label(
            text="Grains fill the volume at radius + spacing (count is the result).",
            icon="INFO",
        )
        layout.prop(self, "rng")
        warn = layout.column()
        warn.alert = True
        warn.label(text="Destructive: original faces are discarded", icon="ERROR")
        warn.label(text="and replaced by a loose-vertex particle cloud.")

    def execute(self, context):
        obj = context.active_object
        if not _is_convertible_solid_mesh(obj):
            self.report({"ERROR"}, iface_("Active object must be a solid mesh with faces"))
            return {"CANCELLED"}

        n = build_and_commit_particle_mesh(
            obj, self.grain_radius, self.extra_spacing, rng_seed=self.rng
        )
        if n < 1:
            self.report(
                {"ERROR"},
                iface_("No grains fit; reduce the grain radius or the extra spacing"),
            )
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            iface_("Converted '{name}' to {count} grain(s)").format(name=obj.name, count=n),
        )
        return {"FINISHED"}


classes = (OBJECT_OT_ConvertToParticleMesh,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
