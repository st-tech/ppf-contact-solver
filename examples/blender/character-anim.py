# File: character-anim.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Launch inside Blender (Scripting workspace, or
# `blender --python character-anim.py`) to play back the Codim-IPC
# "Rumba_Dancing" FEM-shell sequence through the ppf_contact_solver
# addon.
#
# What it does.
#   1. Loads Rumba_Dancing/shell0.obj as a Blender mesh object.
#   2. Wraps that object in a SHELL dynamics group.
#   3. Pins every vertex of the mesh (one vertex group holding all
#      vertices, bound as a single pin).
#   4. Hands the whole shellN.obj sequence to ``pin.set_animation``,
#      which stores it in the addon's pin-input PC2 cache. A persistent
#      frame-change handler plays it back in the viewport, and the
#      solver encoder reads the same cache to drive the pinned shell.
#
#   Because all vertices are pinned and animated, the shell carries no
#   free degrees of freedom: it is driven entirely by the OBJ sequence.
#   That makes it useful as a moving, self-colliding obstacle that
#   other (unpinned) groups in the same scene can interact with.
#
# Coordinate frame.
#   The Codim-IPC OBJ files are Y-up (the dancer's height runs along
#   OBJ +Y). The addon converts Blender Z-up to solver Y-up with
#   ``zup_to_yup``: solver = (Bx, Bz, -By). To keep the dancer upright
#   in Blender's native Z-up *and* land the solver-frame position back
#   on the original OBJ coordinate, each OBJ vertex (ox, oy, oz) is
#   placed at Blender (ox, -oz, oy).
#
# Storage.
#   The per-frame vertex positions live in a PC2 binary in the addon's
#   data/ sidecar folder (the same mechanism rod animation uses), not
#   in Blender F-curves. Re-running the script or calling
#   ``solver.clear()`` discards the cache.

import glob
import importlib
import os
import re

import bpy
import numpy as np


def _resolve_addon() -> str:
    """Return the importable module name of the installed addon."""
    import addon_utils

    name = next(
        (m.__name__ for m in addon_utils.modules()
         if m.__name__.endswith(".ppf_contact_solver")),
        None,
    )
    if name is None:
        raise ImportError(
            "ZOZO's Contact Solver addon not found; enable it in "
            "Preferences > Add-ons first."
        )
    return name


def _find_rumba_dir() -> str | None:
    """Locate the Rumba_Dancing OBJ folder under ~/Desktop/blender-examples."""
    base = os.path.expanduser("~/Desktop/blender-examples")
    if not os.path.isdir(base):
        return None
    for root, _dirs, files in os.walk(base):
        if os.path.basename(root) == "Rumba_Dancing" and "shell0.obj" in files:
            return root
    return None


def _shell_files(obj_dir: str) -> list[str]:
    """Return shellN.obj paths sorted by their numeric index."""
    files = glob.glob(os.path.join(obj_dir, "shell*.obj"))

    def _index(path: str) -> int:
        m = re.search(r"shell(\d+)\.obj$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    files = [f for f in files if _index(f) >= 0]
    files.sort(key=_index)
    return files


def _load_obj(path: str) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Parse an OBJ file into (vertices, faces).

    Faces are returned 0-indexed; vertex/normal/uv tokens like ``a/b/c``
    are reduced to their vertex component.
    """
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                faces.append(tuple(
                    int(tok.split("/")[0]) - 1 for tok in line.split()[1:]
                ))
    return np.asarray(verts, dtype=np.float64), faces


def _load_obj_verts(path: str) -> np.ndarray:
    """Parse only the vertex positions from an OBJ file."""
    verts: list[tuple[float, float, float]] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
    return np.asarray(verts, dtype=np.float64)


def _to_blender(verts: np.ndarray) -> np.ndarray:
    """Remap OBJ Y-up coords (ox, oy, oz) to Blender Z-up (ox, -oz, oy)."""
    return np.column_stack([verts[:, 0], -verts[:, 2], verts[:, 1]])


def build(obj_dir: str | None = None, max_frames: int | None = None,
          fps: int = 24, object_name: str = "RumbaDancer",
          group_name: str = "RumbaDancer", vertex_group: str = "pinned",
          shell_density: float = 0.3, shell_young_modulus: float = 5.0e5,
          bend: float = 1.0e-3, friction: float = 0.2):
    addon = _resolve_addon()
    solver = importlib.import_module(f"{addon}.ops.api").solver

    if obj_dir is None:
        obj_dir = _find_rumba_dir()
    if not obj_dir or not os.path.isdir(obj_dir):
        raise FileNotFoundError(
            "Could not locate the Rumba_Dancing OBJ folder. Pass "
            "obj_dir=... explicitly (the directory holding shell0.obj)."
        )

    files = _shell_files(obj_dir)
    if not files:
        raise FileNotFoundError(f"No shellN.obj files found in {obj_dir}")
    if max_frames is not None:
        files = files[:max_frames]
    n_frames = len(files)

    # Frame 1 mesh: shell0.obj supplies geometry and topology.
    rest_verts, faces = _load_obj(files[0])
    n_verts = len(rest_verts)
    print(f"character-anim: {n_verts} verts, {len(faces)} faces, "
          f"{n_frames} frames from {obj_dir}")

    # Stack every frame's vertex positions into (n_frames, n_verts, 3),
    # remapped to Blender Z-up.
    positions = np.empty((n_frames, n_verts, 3), dtype=np.float32)
    positions[0] = _to_blender(rest_verts)
    for f_i, path in enumerate(files[1:], start=1):
        v = _load_obj_verts(path)
        if len(v) != n_verts:
            raise ValueError(
                f"{os.path.basename(path)} has {len(v)} verts, expected "
                f"{n_verts}; the sequence must share one topology."
            )
        positions[f_i] = _to_blender(v)
        if f_i % 20 == 0:
            print(f"character-anim: loaded {f_i}/{n_frames - 1} frames")

    # Reset solver state, then drop any object left by a previous run.
    solver.clear()
    if object_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[object_name], do_unlink=True)
    if object_name in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes[object_name])

    # Build the mesh object from shell0.obj.
    mesh = bpy.data.meshes.new(object_name)
    mesh.from_pydata([tuple(v) for v in positions[0]], [], faces)
    mesh.update()
    obj = bpy.data.objects.new(object_name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    for poly in mesh.polygons:
        poly.use_smooth = True

    # One vertex group holding every vertex; this is what gets pinned.
    vg = obj.vertex_groups.new(name=vertex_group)
    vg.add(list(range(n_verts)), 1.0, "REPLACE")

    # SHELL group wrapping the dancer.
    grp = solver.create_group(group_name, type="SHELL")
    grp.add(object_name)
    grp.param.shell_density = shell_density
    grp.param.shell_young_modulus = shell_young_modulus
    # Shell bend is density-normalized in the solver, so divide by density to
    # reproduce the pre-normalization drape (this example uses density 0.3).
    grp.param.bend = bend / shell_density
    grp.param.friction = friction

    # Pin every vertex, then hand it the whole OBJ sequence. set_animation
    # writes the pin-input PC2 cache and registers the playback handler.
    pin = grp.create_pin(object_name, vertex_group)
    pin.set_animation(positions)

    # Session timeline: one solver frame per OBJ file.
    solver.param.frame_count = n_frames
    solver.param.frame_rate = fps
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = n_frames
    bpy.context.scene.frame_set(1)

    print(f"character-anim: SHELL '{group_name}' with {n_verts} pinned "
          f"vertices, {n_frames} frames @ {fps} fps")
    return obj


if __name__ == "__main__":
    build()
