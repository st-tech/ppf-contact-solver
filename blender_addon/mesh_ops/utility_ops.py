# File: utility_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# General mesh utilities exposed in the "Utility Tools" panel.
#
# Symmetric triangulate: poke every face (insert a center vertex and fan
# it into triangles). For a quad that is a 4-triangle fan, which is
# mirror-symmetric under both in-plane axes. A single-diagonal
# triangulation is NOT mirror-symmetric, so on a symmetric mesh it makes
# bending / folding develop asymmetrically in the solver (the bending
# hinges inherit the diagonal bias). Poking removes that bias, so a
# symmetric rest shape folds symmetrically.

import bmesh  # pyright: ignore
import bpy  # pyright: ignore
from bpy.types import Operator  # pyright: ignore


class OBJECT_OT_SymmetricTriangulate(Operator):
    """Symmetrically triangulate the selected mesh objects by poking each
    face (insert a center vertex and fan it into triangles). Unlike a
    single-diagonal triangulation this is mirror-symmetric, so a symmetric
    mesh keeps its symmetry under bending in the simulation. Adds one
    vertex per face"""

    bl_idname = "object.ppf_symmetric_triangulate"
    bl_label = "Symmetric Triangulate"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and any(
            o.type == "MESH" for o in context.selected_objects
        )

    def execute(self, context):
        seen_data = set()
        n_obj = 0
        for obj in context.selected_objects:
            if obj.type != "MESH":
                continue
            n_obj += 1
            me = obj.data
            # Objects sharing one mesh data-block (linked duplicates) would
            # otherwise be poked once per user; poke the data only once.
            if me.name in seen_data:
                continue
            seen_data.add(me.name)
            bm = bmesh.new()
            try:
                bm.from_mesh(me)
                if bm.faces:
                    bmesh.ops.poke(bm, faces=bm.faces[:])
                    bm.to_mesh(me)
                    me.update()
            finally:
                bm.free()

        if n_obj == 0:
            self.report({"WARNING"}, "Select one or more mesh objects")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Symmetric-triangulated {n_obj} object(s) "
            f"({len(seen_data)} mesh data-block(s))",
        )
        return {"FINISHED"}


classes = (OBJECT_OT_SymmetricTriangulate,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
