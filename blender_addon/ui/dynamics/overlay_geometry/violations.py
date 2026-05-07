# File: violations.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import gpu  # pyright: ignore

from mathutils import Vector  # pyright: ignore

from .primitives import _line_to_tris


def _build_violation_batches(scene, depsgraph, violations):
    """Build GPU batches to highlight validation violations.

    Returns list of (batch, primitive_type, color) tuples.
    """
    from gpu_extras.batch import batch_for_shader

    batches = []
    if not violations:
        return batches

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")

    COLORS = {
        "self_intersection": (1.0, 0.1, 0.1, 0.85),
        "contact_offset": (1.0, 0.5, 0.0, 0.85),
        "wall": (1.0, 0.1, 0.1, 0.9),
        "sphere": (0.8, 0.1, 1.0, 0.9),
        "runtime_intersection": (1.0, 0.05, 0.05, 0.9),
    }
    LABELS = {
        "self_intersection": "Self-Intersections",
        "contact_offset": "Contact-Offset Violations",
        "wall": "Wall Violations",
        "sphere": "Sphere Violations",
        "runtime_intersection": "Runtime Intersection(s)",
    }
    EDGE_THICKNESS = 0.006

    def _solver_to_blender(pos):
        """Convert solver Y-up position to Blender Z-up."""
        return Vector((pos[0], -pos[2], pos[1]))

    labels = []

    for violation in violations:
        vtype = violation.get("type", "")
        color = COLORS.get(vtype, (1.0, 0.0, 0.0, 0.9))
        label_text = LABELS.get(vtype, "Violation")
        count = violation.get("count", 0)

        if vtype in ("wall", "sphere"):
            points = []
            for entry in violation.get("vertices", []):
                pos = entry.get("pos")
                if pos:
                    points.append(_solver_to_blender(pos))
            if points:
                batch = batch_for_shader(shader, "POINTS", {"pos": points})
                batches.append((batch, "POINTS", color))
                c = points[0]
                labels.append({
                    "pos_3d": c + Vector((0, 0, 0.05)),
                    "text": f"{count} {label_text}",
                    "color": color[:3] + (1.0,),
                })

        elif vtype == "contact_offset":
            tri_verts = []
            centers = []
            for pair in violation.get("pairs", []):
                for key in ("ei", "ej"):
                    etype = pair.get(f"{key}_type", "")
                    pos_list = pair.get(f"{key}_pos", [])
                    if not pos_list:
                        continue
                    bverts = [_solver_to_blender(p) for p in pos_list]
                    if etype == "triangle" and len(bverts) >= 3:
                        tri_verts.extend([bverts[0], bverts[1], bverts[2]])
                        if not centers:
                            centers.append((bverts[0] + bverts[1] + bverts[2]) / 3.0)
                    elif etype == "edge" and len(bverts) >= 2:
                        tri_verts.extend(_line_to_tris(bverts[0], bverts[1], EDGE_THICKNESS))
                        if not centers:
                            centers.append((bverts[0] + bverts[1]) / 2.0)
            if tri_verts:
                batch = batch_for_shader(shader, "TRIS", {"pos": tri_verts})
                batches.append((batch, "TRIS", color))
                if centers:
                    labels.append({
                        "pos_3d": centers[0] + Vector((0, 0, 0.05)),
                        "text": f"{count} {label_text}",
                        "color": color[:3] + (1.0,),
                    })

        elif vtype == "self_intersection":
            tri_verts = []
            centers = []
            for tri_pair in violation.get("tris", []):
                for tri_pos in tri_pair:
                    if len(tri_pos) >= 3:
                        bv = [_solver_to_blender(p) for p in tri_pos]
                        tri_verts.extend([bv[0], bv[1], bv[2]])
                        if not centers:
                            centers.append((bv[0] + bv[1] + bv[2]) / 3.0)
            if tri_verts:
                batch = batch_for_shader(shader, "TRIS", {"pos": tri_verts})
                batches.append((batch, "TRIS", color))
                if centers:
                    labels.append({
                        "pos_3d": centers[0] + Vector((0, 0, 0.05)),
                        "text": f"{count} {label_text}",
                        "color": color[:3] + (1.0,),
                    })

        elif vtype == "runtime_intersection":
            tri_verts = []
            centers = []
            for entry in violation.get("entries", []):
                itype = entry.get("itype", "")
                pos0 = entry.get("positions0", [])
                pos1 = entry.get("positions1", [])
                if itype in ("face_edge", "collision_mesh"):
                    # pos0 = face (3 verts), pos1 = edge (2 verts)
                    if len(pos0) >= 3:
                        bv = [_solver_to_blender(p) for p in pos0]
                        tri_verts.extend([bv[0], bv[1], bv[2]])
                        if not centers:
                            centers.append((bv[0] + bv[1] + bv[2]) / 3.0)
                    if len(pos1) >= 2:
                        bv = [_solver_to_blender(p) for p in pos1]
                        tri_verts.extend(_line_to_tris(bv[0], bv[1], EDGE_THICKNESS))
                elif itype == "edge_edge":
                    # pos0 = edge0 (2 verts), pos1 = edge1 (2 verts)
                    for edge_pos in (pos0, pos1):
                        if len(edge_pos) >= 2:
                            bv = [_solver_to_blender(p) for p in edge_pos]
                            tri_verts.extend(_line_to_tris(bv[0], bv[1], EDGE_THICKNESS))
                            if not centers:
                                centers.append((bv[0] + bv[1]) / 2.0)
            if tri_verts:
                batch = batch_for_shader(shader, "TRIS", {"pos": tri_verts})
                batches.append((batch, "TRIS", color))
                if centers:
                    labels.append({
                        "pos_3d": centers[0] + Vector((0, 0, 0.05)),
                        "text": f"{count} {label_text}",
                        "color": color[:3] + (1.0,),
                    })

    return batches, labels
