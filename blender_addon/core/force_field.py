# File: force_field.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Blender force fields, and the exact force-field script, for the solver.

The solver takes an external force field as sampled W x H x D x T grids and
one exact script (``frontend/_force_field_.py``). This module turns the
scene's force-field objects into those grids, and is also what the viewport's
Visualize overlay draws from, so what is drawn is exactly what is sent.

WHAT EACH BLENDER FIELD BECOMES. Every quantity is evaluated in Blender's
world space at a sample point ``p``, with ``o`` the field object's origin and
``z`` its local Z axis:

* **Force**: an acceleration ``strength * falloff`` along ``(p - o)/|p - o|``
  (shape Point) or along ``+-z`` by side (shape Plane). Positive pushes away.
* **Wind**: an AIR VELOCITY ``strength * falloff`` along ``z``, added to the
  scene wind inside the aerodynamic drag, so it acts only where Air Density is
  positive, and on a surface by how it faces the flow, as the scene wind does.
* **Vortex**: an acceleration ``strength * falloff`` along ``z x (p - o)``,
  normalized, circling the field's Z axis.
* **Turbulence**: an acceleration ``strength * falloff * n(q / size)``, ``n``
  this project's own seeded gradient noise, ``q`` the point in the field's
  local frame (or in world space with Global Coordinates).

Strength is read as m/s^2 for the accelerations and m/s for Wind, the units
gravity and the scene wind use. Falloff follows the field's settings: the
distance is ``|p - o|`` (Point) or the distance to the field's XY plane
(Plane); Sphere applies ``(1 + d - min)^-power`` with the minimum and maximum
distances; Tube applies that along the axis times the radial version; Z
Direction keeps one side only.

EVERYTHING ELSE IS REFUSED BY NAME at Transfer rather than ignored: another
field type, another shape, the Cone falloff, a nonzero Flow on anything but Wind, a nonzero Noise Amount,
and Wind in a scene with no Air Density. A refused field reported loudly is
better than a turbulence that silently does nothing, which is issue #114.

THE TURBULENCE PATTERN IS NOT BLENDER'S. Blender's noise implementation is
GPL and cannot be copied into this project, so the noise is this project's own
(`core/noise.py`, the same algorithm the solver runs for a script's `noise`):
Size, Strength and Seed mean what they mean in Blender, and the swirls are a
different random pattern.

EACH SOURCE REACHES EVERY GROUP OR THE GROUPS IT NAMES
(`models/force_field_targets.py`). Field objects with the same targets are
summed into one grid per kind; a different set of targets is a grid of its
own.

WHERE AND HOW FINELY, WITH NOTHING TO SET BUT TWO LENGTHS. A grid covers the
objects its fields push, at the starting frame, grown by Padding on every
side; its points are at most Spacing apart along every axis, so the counts
along X, Y and Z follow from the box. One plan (:func:`grid_plan`) answers
for Transfer, the panel's estimate and the grid count, so what the panel
promises is what Transfer sends.
"""

from __future__ import annotations

import json
import math
import threading
import zlib

import numpy as np

from . import noise as _noise
from . import script_api as _script_api

SUPPORTED_TYPES = ("FORCE", "WIND", "VORTEX", "TURBULENCE")
SUPPORTED_SHAPES = ("POINT", "PLANE")
SUPPORTED_FALLOFF = ("SPHERE", "TUBE")

# Arrow colors for the overlay: acceleration and air velocity.
COLOR_ACCELERATION = (1.0, 0.55, 0.15, 0.9)
COLOR_AIR = (0.3, 0.8, 1.0, 0.9)
COLOR_SCRIPT = (0.85, 0.4, 1.0, 0.9)


# --- estimates -----------------------------------------------------------


def estimate_bytes(shapes, samples: int) -> int:
    """Bytes the solver holds for grids of the given ``(W, H, D)`` shapes at
    ``samples`` instants, 3-vectors of float32."""
    return sum(int(w) * int(h) * int(d) for w, h, d in shapes) * int(samples) * 3 * 4


def estimate_line(shapes, samples: int) -> str:
    """The ``[Info]`` line every place a grid is authored shows."""
    mb = estimate_bytes(shapes, samples) / 1.0e6
    dims = ", ".join(f"{w}x{h}x{d}x{samples}" for w, h, d in shapes)
    return f"[Info] Force field {dims}: {mb:.1f} MB estimated"


# --- which objects ---------------------------------------------------------


def field_objects(scene, state) -> list:
    """The objects whose force field the solver receives.

    Every object with a field in the chosen collection (its children
    included), or in the whole scene when none is chosen.
    """
    coll = getattr(state, "force_field_collection", None)
    source = coll.all_objects if coll is not None else scene.objects
    found = []
    for obj in source:
        field = getattr(obj, "field", None)
        if field is not None and field.type != "NONE":
            found.append(obj)
    found.sort(key=lambda o: o.name)
    return found


def refusal(obj) -> str | None:
    """Why ``obj``'s field cannot be sent, or ``None`` when it can."""
    f = obj.field
    if f.type not in SUPPORTED_TYPES:
        return (
            f"'{obj.name}' is a {f.type.title()} field; the supported types are "
            "Force, Wind, Vortex and Turbulence"
        )
    if f.shape not in SUPPORTED_SHAPES:
        return f"'{obj.name}' has shape {f.shape.title()}; use Point or Plane"
    if f.falloff_type not in SUPPORTED_FALLOFF:
        return f"'{obj.name}' has a {f.falloff_type.title()} falloff; use Sphere or Tube"
    # A Wind field IS an air flow here, which is what Blender's Flow setting
    # asks for (and Wind defaults to Flow 1), so it is read for no other type.
    if f.type != "WIND" and abs(float(f.flow)) > 0.0:
        return (
            f"'{obj.name}' has Flow {f.flow:g}; only a Wind field acts as an air "
            "flow here, set it to 0"
        )
    if abs(float(f.noise)) > 0.0:
        return (
            f"'{obj.name}' has Noise Amount {f.noise:g}; it is not supported, set it "
            "to 0 (use a Turbulence field for noise)"
        )
    return None


# --- snapshot and evaluation -----------------------------------------------


def snapshot(objs) -> list[dict]:
    """Each field's settings and world transform at the current frame."""
    out = []
    for obj in objs:
        f = obj.field
        m = np.array(obj.matrix_world, dtype=np.float64)
        rot = m[:3, :3].copy()
        # The field's own axes, unscaled: a scaled empty must not scale the
        # distances its falloff is measured in.
        norms = np.linalg.norm(rot, axis=0)
        norms[norms == 0.0] = 1.0
        axes = rot / norms
        out.append({
            "name": obj.name,
            "type": f.type,
            "shape": f.shape,
            "falloff_type": f.falloff_type,
            "strength": float(f.strength),
            "power": float(f.falloff_power),
            "use_min": bool(f.use_min_distance),
            "min": float(f.distance_min),
            "use_max": bool(f.use_max_distance),
            "max": float(f.distance_max),
            "use_radial_min": bool(f.use_radial_min),
            "radial_min": float(f.radial_min),
            "use_radial_max": bool(f.use_radial_max),
            "radial_max": float(f.radial_max),
            "radial_power": float(f.radial_falloff),
            "z_direction": f.z_direction,
            "size": float(f.size),
            "seed": int(f.seed),
            "global_coords": bool(f.use_global_coords),
            "origin": m[:3, 3].copy(),
            "axes": axes,
        })
    return out


def _falloff(d, use_min, dmin, use_max, dmax, power):
    d = np.asarray(d, dtype=np.float64)
    base = dmin if use_min else 0.0
    f = np.power(1.0 + np.maximum(d - base, 0.0), -power) if power != 0.0 else np.ones_like(d)
    if use_min:
        f = np.where(d < dmin, 1.0, f)
    if use_max:
        f = np.where(d > dmax, 0.0, f)
    return f


def _field_weight(s, local):
    """The falloff factor at points given in the field's local frame."""
    lz = local[:, 2]
    if s["shape"] == "PLANE" or s["falloff_type"] == "TUBE":
        axial = np.abs(lz)
    else:
        axial = np.linalg.norm(local, axis=1)
    w = _falloff(axial, s["use_min"], s["min"], s["use_max"], s["max"], s["power"])
    if s["falloff_type"] == "TUBE":
        radial = np.hypot(local[:, 0], local[:, 1])
        w = w * _falloff(radial, s["use_radial_min"], s["radial_min"],
                         s["use_radial_max"], s["radial_max"], s["radial_power"])
    if s["z_direction"] == "POSITIVE":
        w = np.where(lz >= 0.0, w, 0.0)
    elif s["z_direction"] == "NEGATIVE":
        w = np.where(lz <= 0.0, w, 0.0)
    return w


def evaluate(samples: list[dict], points) -> tuple[np.ndarray, np.ndarray]:
    """Sum every field at ``points`` (N x 3, Blender world).

    Returns ``(acceleration, air_velocity)``, each N x 3 in Blender axes.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    accel = np.zeros_like(p)
    air = np.zeros_like(p)
    for s in samples:
        rel = p - s["origin"]
        local = rel @ s["axes"]
        z = s["axes"][:, 2]
        w = _field_weight(s, local) * s["strength"]
        t = s["type"]
        if t == "FORCE":
            if s["shape"] == "PLANE":
                side = np.where(local[:, 2] >= 0.0, 1.0, -1.0)
                accel += (w * side)[:, None] * z[None, :]
            else:
                r = np.linalg.norm(rel, axis=1)
                safe = np.where(r > 1e-9, r, 1.0)
                accel += (w / safe * (r > 1e-9))[:, None] * rel
        elif t == "WIND":
            air += w[:, None] * z[None, :]
        elif t == "VORTEX":
            tangent = np.cross(z[None, :], rel)
            n = np.linalg.norm(tangent, axis=1)
            safe = np.where(n > 1e-9, n, 1.0)
            accel += (w / safe * (n > 1e-9))[:, None] * tangent
        elif t == "TURBULENCE":
            q = (p if s["global_coords"] else local) / max(s["size"], 1e-6)
            accel += w[:, None] * np.stack(
                _noise.noise_vector(q[:, 0], q[:, 1], q[:, 2], 1, s["seed"]), axis=1)
    return accel, air


# --- where and how finely ---------------------------------------------------


def objects_box(objs, padding: float) -> tuple[np.ndarray, np.ndarray]:
    """The world bounds of ``objs`` at the current frame, grown by
    ``padding`` on every side."""
    pts = []
    for o in objs:
        pts.extend(o.matrix_world @ _vec(c) for c in o.bound_box)
    pts = np.array(pts)
    lo, hi = pts.min(axis=0) - padding, pts.max(axis=0) + padding
    hi = np.maximum(hi, lo + 1e-3)
    return lo, hi


def grid_shape(lo, hi, spacing: float) -> tuple[int, int, int]:
    """``(W, H, D)``: the fewest points along each axis of the box that are
    at most ``spacing`` apart (at least two)."""
    extent = np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)
    n = [max(2, int(math.ceil(e / spacing - 1e-9)) + 1) for e in extent]
    return n[0], n[1], n[2]


def _vec(c):
    from mathutils import Vector  # pyright: ignore

    return Vector(c)


def grid_points(lo, hi, resolution) -> np.ndarray:
    """(D, H, W, 3) Blender-world corner points of a W x H x D grid."""
    w, h, d = (int(v) for v in resolution)
    xs = np.linspace(lo[0], hi[0], w)
    ys = np.linspace(lo[1], hi[1], h)
    zs = np.linspace(lo[2], hi[2], d)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.stack([X, Y, Z], axis=-1)


# --- transfer ---------------------------------------------------------------


def _to_solver(v):
    """Blender (x, y, z) vectors to the solver's (x, z, -y)."""
    v = np.asarray(v)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


def _kinds(objs) -> list:
    kinds = []
    if any(o.field.type != "WIND" for o in objs):
        kinds.append("acceleration")
    if any(o.field.type == "WIND" for o in objs):
        kinds.append("air-velocity")
    return kinds


def dynamic_objects(scene) -> list:
    """Every object of an active non-Static group that the solve includes."""
    from ..models.groups import iterate_active_object_groups
    from .uuid_registry import resolve_assigned

    out = []
    for group in iterate_active_object_groups(scene):
        if str(group.object_type) == "STATIC":
            continue
        for assigned in group.assigned_objects:
            if assigned.included:
                obj = resolve_assigned(assigned)
                if obj is not None:
                    out.append(obj)
    return out


def _group_objects(scene, uuids) -> list:
    from ..models.groups import get_group_by_uuid
    from .uuid_registry import resolve_assigned

    out = []
    for uid in uuids:
        group = get_group_by_uuid(scene, uid)
        if group is None:
            continue
        for assigned in group.assigned_objects:
            if assigned.included:
                obj = resolve_assigned(assigned)
                if obj is not None:
                    out.append(obj)
    return out


def grid_plan(scene, state, pushed_by_all=None) -> list[dict]:
    """The grids Transfer sends, one entry per distinct target set:

    ``{"uuids": None or sorted group uuids, "fields": [field objects],
    "kinds": ["acceleration", "air-velocity"], "lo", "hi", "shape"}``.

    The box covers the objects the fields push (``pushed_by_all``, or every
    simulated object, for fields reaching every group) at the current frame,
    grown by Padding. Raises ``ValueError`` naming the field when the objects
    it pushes are none, and naming the objects when their box is thinner than
    one Spacing along some axis.
    """
    from ..models.force_field_targets import target_group_uuids

    by_target: dict = {}
    for obj in field_objects(scene, state):
        if refusal(obj) is None:
            key = target_group_uuids(state, obj)
            by_target.setdefault(None if key is None else tuple(sorted(key)), []).append(obj)
    spacing = float(state.force_field_spacing)
    padding = float(state.force_field_padding)
    plan = []
    for key, fields in by_target.items():
        if key is None:
            pushed = dynamic_objects(scene) if pushed_by_all is None else pushed_by_all
        else:
            pushed = _group_objects(scene, key)
        if not pushed:
            names = ", ".join(f"'{o.name}'" for o in fields)
            raise ValueError(
                f"Force field {names} pushes no object: assign objects to the "
                "groups it reaches"
            )
        lo, hi = objects_box(pushed, padding)
        thin = [axis for axis, e in zip("XYZ", hi - lo) if e < spacing * (1.0 - 1e-6)]
        if thin:
            # A box thinner than one Spacing is a sliver the objects leave as
            # soon as they move (a flat sheet with Padding 0 lies on its
            # face), after which the field no longer reaches them. Refused
            # here rather than letting the push quietly stop.
            names = ", ".join(f"'{o.name}'" for o in pushed[:3])
            if len(pushed) > 3:
                names += f" and {len(pushed) - 3} more"
            depth = min(float(e) for e in hi - lo)
            raise ValueError(
                f"Force field: the sampled box around {names} is only "
                f"{depth * 1000.0:.1f} mm deep along {', '.join(thin)}, less than one "
                f"Spacing ({spacing:g} m), so the objects would leave it as soon as "
                "they move and the field would stop acting on them. Raise Padding"
            )
        plan.append({"uuids": key, "fields": fields, "kinds": _kinds(fields),
                     "lo": lo, "hi": hi, "shape": grid_shape(lo, hi, spacing)})
    return plan


def plan_shapes(plan) -> list:
    """One ``(W, H, D)`` per grid of ``plan``: a shape per kind."""
    return [entry["shape"] for entry in plan for _ in entry["kinds"]]


def encode(context, state, dynamic_objects, start_frame: int, frame_count: int,
           solver_fps: float, position_by_uuid: dict) -> dict | None:
    """The PARAM payload's ``force_field`` entry, or ``None`` when the scene
    carries neither a supported field nor a script.

    Samples every field at the grid's corners at T instants spread evenly over
    the solve, restoring the current frame afterwards; fields with the same
    target groups share a grid. ``position_by_uuid`` maps an active group's
    uuid to its position in the payload's group list. Raises ``ValueError``
    naming the object for anything it cannot send.
    """
    from ..models.force_field_targets import SCRIPT, encode_positions

    scene = context.scene
    objs = field_objects(scene, state)
    for obj in objs:
        why = refusal(obj)
        if why:
            raise ValueError(f"Force field: {why}")
    text = getattr(state, "force_field_script", None)
    source = text.as_string() if text is not None else ""
    if not objs and not source.strip():
        return None
    if any(o.field.type == "WIND" for o in objs) and float(state.air_density) <= 0.0:
        raise ValueError(
            "Force field: a Wind field blows the air, and this scene's Air Density "
            "is 0, so it would move nothing. Raise Air Density, or use a Force field"
        )
    out: dict = {"grids": [], "scripts": []}
    if source.strip():
        out["scripts"].append({
            "source": source, "name": text.name,
            "groups": encode_positions(scene, state, SCRIPT, position_by_uuid),
        })
    if not objs:
        return out

    for obj in objs:
        # Refuses a stale or Static target reference by the field's name.
        encode_positions(scene, state, obj, position_by_uuid)
    # The boxes are taken at the starting frame, as the objects start there.
    current = scene.frame_current, scene.frame_subframe
    scene.frame_set(int(start_frame))
    try:
        plan = grid_plan(scene, state, dynamic_objects)
    finally:
        scene.frame_set(current[0], subframe=current[1])
    samples = max(1, int(state.force_field_time_samples))
    shapes = plan_shapes(plan)
    nbytes = estimate_bytes(shapes, samples)
    print(estimate_line(shapes, samples))
    cap = float(state.force_field_max_mb) * 1.0e6
    if nbytes > cap:
        raise ValueError(
            f"Force field: the sampled grids are {nbytes / 1e6:.1f} MB, past the "
            f"{state.force_field_max_mb:g} MB limit. Raise the Spacing or lower the "
            "Time Samples, or raise the limit"
        )

    last = start_frame + max(frame_count - 1, 0)
    frames = [float(start_frame)] if samples == 1 else list(
        np.linspace(start_frame, last, samples))
    # The solver's grid runs over ITS axes: its (ix, iy, iz) walk x, z, -y.
    # Reorder the Blender-space sample array accordingly: Blender (D=z, H=y,
    # W=x) becomes solver (D=-y, H=z, W=x).
    points = []
    data = []
    for entry in plan:
        w, h, d = entry["shape"]
        points.append(grid_points(entry["lo"], entry["hi"], (w, h, d)).reshape(-1, 3))
        data.append({kind: np.empty((samples, h, d, w, 3), dtype=np.float32)
                     for kind in entry["kinds"]})
    try:
        for k, f in enumerate(frames):
            scene.frame_set(int(math.floor(f)), subframe=f - math.floor(f))
            for entry, pts, arrays in zip(plan, points, data):
                w, h, d = entry["shape"]
                acc, air = evaluate(snapshot(entry["fields"]), pts)
                for kind, vec in (("acceleration", acc), ("air-velocity", air)):
                    if kind not in arrays:
                        continue
                    v = _to_solver(vec).reshape(d, h, w, 3)
                    # (z, y, x) -> (y reversed, z, x): solver D axis is -y.
                    arrays[kind][k] = np.transpose(v, (1, 0, 2, 3))[::-1]
    finally:
        scene.frame_set(current[0], subframe=current[1])
    times = [(f - start_frame) / solver_fps for f in frames]
    for entry, arrays in zip(plan, data):
        lo, hi = entry["lo"], entry["hi"]
        positions = encode_positions(scene, state, entry["fields"][0], position_by_uuid)
        for kind, arr in arrays.items():
            arr = np.ascontiguousarray(arr)
            out["grids"].append({
                "kind": kind,
                "shape": list(arr.shape),
                # The solver's box: Blender (x, y, z) -> (x, z, -y), so y's
                # extremes swap.
                "min": [float(lo[0]), float(lo[2]), float(-hi[1])],
                "max": [float(hi[0]), float(hi[2]), float(-lo[1])],
                "times": times,
                "groups": positions,
                "data": zlib.compress(arr.tobytes(), 6),
            })
    return out


# --- the exact script, locally, for drawing only -----------------------------

def _import_math_only(name, *args, **kwargs):
    # `import math` is how a script names the math functions, and the
    # compiler admits that one import; anything else is refused here too.
    if name != "math":
        raise ImportError(f"a force field script may import only math, not {name!r}")
    return math


_SCRIPT_BUILTINS = {"abs": abs, "min": min, "max": max, "float": float, "range": range,
                    "__import__": _import_math_only}
# The builtins the compiler admits beyond `math`, the same algorithm the
# solver runs.
_SCRIPT_GLOBALS = {"noise": _noise.noise, "curl_noise": _noise.curl_noise}
# The list the panel shows (`core/script_api.py`) names exactly these.
assert set(_SCRIPT_BUILTINS) - {"__import__"} == {n for n, _, _ in _script_api.PYTHON}
assert set(_SCRIPT_GLOBALS) == {n for n, _, _ in _script_api.NOISE}


def builtins_reference() -> list[dict]:
    """Every function and constant a script may use, as
    ``{"section", "name", "signature", "description"}`` rows, for MCP and the
    Python API."""
    return [{"section": title, "name": name, "signature": sig, "description": what}
            for title, entries in _script_api.SECTIONS for name, sig, what in entries]


def local_script(source: str):
    """The script's function, for the Visualize overlay only.

    Run with ``math`` and five builtins in reach, the same subset the compiler
    admits. What reaches the solver is the SERVER's compile of the text, never
    this.
    """
    namespace = {"math": math, "__builtins__": _SCRIPT_BUILTINS, **_SCRIPT_GLOBALS}
    exec(compile(source, "<force field script>", "exec"), namespace)
    fn = namespace.get("eval")
    if fn is None:
        fns = [v for k, v in namespace.items()
               if callable(v) and k not in _SCRIPT_BUILTINS and k != "__builtins__"
               and k not in _SCRIPT_GLOBALS and not isinstance(v, type(math))]
        fn = fns[0] if len(fns) == 1 else None
    if fn is None:
        raise ValueError("define def eval(x, y, z, t)")
    return fn


# --- Compile and Check --------------------------------------------------------

_check_lock = threading.Lock()
_check_state: dict = {"running": False, "result": None, "text": "", "digest": None}


def check_state() -> dict:
    with _check_lock:
        return dict(_check_state)


def _digest(source: str) -> str:
    import hashlib

    return hashlib.sha1(source.encode("utf-8")).hexdigest()


def check_result_for(source: str) -> dict | None:
    """The last answer, when it was for exactly this text."""
    with _check_lock:
        if _check_state["digest"] == _digest(source):
            return _check_state["result"]
    return None


def request_check(open_channel, source: str, text_name: str) -> None:
    """Ask the server to compile ``source``, on a thread of its own.

    ``open_channel`` is the live connection's channel opener. The answer lands
    in :func:`check_state`; the caller polls it. A connect attempt or a Run
    never waits on this, since it holds no I/O worker.
    """
    with _check_lock:
        if _check_state["running"]:
            return
        _check_state.update(running=True, result=None, text=text_name, digest=_digest(source))

    def work():
        try:
            answer = json_request(open_channel, {
                "request": "force_field_check", "source": source, "z_up": True,
            })
            if "error" in answer:
                result = {"ok": False, "error": str(answer["error"]), "line": None,
                          "transport": True}
            else:
                result = dict(answer.get("result") or {})
        except Exception as e:  # the connection itself failed
            result = {"ok": False, "error": f"the server could not be asked: {e}",
                      "line": None, "transport": True}
        with _check_lock:
            _check_state.update(running=False, result=result)

    threading.Thread(target=work, name="ppf-force-field-check", daemon=True).start()


def json_request(open_channel, request: dict, timeout: float = 60.0) -> dict:
    """Send one JSON request and read the one-line JSON answer."""
    from .protocol import _send_json_header

    channel = open_channel()
    try:
        channel.settimeout(timeout)
        _send_json_header(channel, request)
        response = b""
        while b"\n" not in response:
            chunk = channel.recv(65536)
            if not chunk:
                break
            response += chunk
    finally:
        channel.close()
    line = response.split(b"\n", 1)[0]
    if not line:
        raise RuntimeError("the server closed the connection without answering")
    return json.loads(line.decode("utf-8"))
