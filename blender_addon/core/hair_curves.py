# File: hair_curves.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Hair Curves as rods. Single source of truth for everything the add-on
# reads from or writes to a Hair Curves object (Blender object type
# ``CURVES``): its rod topology, the pose it ships at, its pins, and the
# playback of a solve.
#
# A Hair Curves point lies ON its strand, so a point is a rod vertex and
# the index spaces are the same: point ``i`` is rod vertex ``i`` in the
# encoder, the pin lists, the solver output and the playback cache. That
# is what separates this file from ``curve_rod.py``, whose legacy Curve
# objects carry off-curve control points and need a fit in both directions.
#
# Three things a Hair Curves object cannot do the way a mesh does, and what
# stands in for each:
#   - It has no vertex groups, so a pin is a point attribute, plus one pin
#     that is derived and stored nowhere: the roots, which hold the first
#     segment of every strand. A boolean attribute holds the points it marks;
#     a float one holds every point it gives a weight above zero, and that
#     weight scales the pull of a pin that pulls.
#   - Blender refuses a Mesh Cache modifier on it, so a solve plays back
#     through a point attribute the frame-change handler writes and a
#     Geometry Nodes modifier that sets the positions from it.
#   - Every modifier on it is a Geometry Nodes modifier, which can do
#     anything to the point count, so where the stack stops preserving the
#     points is measured by evaluating it, never read off a modifier type.

import numpy as np

import bpy  # pyright: ignore

HAIR_OBJECT_TYPE = "CURVES"

# The pin that holds the first SEGMENT of every curve: its first point, where
# the strand leaves the scalp, and its second. The segment that lies on the
# scalp is then prescribed end to end and follows it exactly. It does NOT
# clamp the direction the strand leaves in: the solver has no bending term at
# a vertex that is held exactly, so the strand turns about its second point
# whatever its stiffness, and a pull pin over the points after the roots is
# what keeps that direction. Derived from the curve offsets each time it is
# read, so it follows a groom through any edit.
ROOTS_PIN_NAME = "Roots"

# Point attribute holding the object-local positions of the frame being
# played back. Written on the ORIGINAL data, so the groom in ``position``
# is never overwritten by a solve.
SIM_POSITION_ATTRIBUTE = "_solver_position"

# Marks the node group the playback modifier runs as the add-on's own, so a
# node group that merely shares its name is never adopted or removed.
_CACHE_NODE_GROUP_TAG = "_solver_hair_cache"
_CACHE_NODE_GROUP_NAME = "ContactSolverCache"

# Float point attributes Blender itself gives a meaning: a size, an angle and
# a spline weight. None of them is a pin weight, so none is offered as a pin.
_RESERVED_FLOAT_ATTRIBUTES = frozenset({"radius", "tilt", "nurbs_weight"})

# ``curve_type`` values Blender stores per curve. Catmull Rom is the default
# and is stored as the ABSENCE of the attribute.
_CURVE_TYPE_NAMES = {0: "Catmull Rom", 1: "Poly", 2: "Bezier", 3: "NURBS"}
_ON_CURVE_TYPES = (0, 1)


def is_hair_curves(obj) -> bool:
    """True when *obj* is a Hair Curves object."""
    return obj is not None and getattr(obj, "type", None) == HAIR_OBJECT_TYPE


def point_count(obj) -> int:
    """Number of points of *obj*'s original data, which is its rod vertex count."""
    return len(obj.data.points)


def curve_offsets(data) -> np.ndarray:
    """``(n_curves + 1,)`` first-point index of every curve, then the point count."""
    offsets = np.empty(len(data.curve_offset_data), dtype=np.int32)
    data.curve_offset_data.foreach_get("value", offsets)
    return offsets


def _curve_attribute(data, name, dtype, default):
    """Per-curve values of a CURVE-domain attribute, or *default* for every
    curve when the attribute is absent, which is how Blender stores a default."""
    n_curves = len(data.curves)
    attribute = data.attributes.get(name)
    if attribute is None or attribute.domain != "CURVE":
        return np.full(n_curves, default, dtype=dtype)
    values = np.empty(n_curves, dtype=dtype)
    attribute.data.foreach_get("value", values)
    return values


def refuse_unsimulatable(obj, group_name) -> None:
    """Raise ``ValueError`` naming the first thing that keeps *obj* from being
    simulated as rods.

    A curve needs two points to be a rod segment, and its points must lie on
    the strand. Bezier and NURBS curves carry handles or off-curve control
    points, which a rod vertex cannot stand in for.
    """
    data = obj.data
    where = f"Hair Curves '{obj.name}' in group '{group_name}'"
    if len(data.curves) == 0:
        raise ValueError(
            f"{where} holds no curves, so there is nothing to simulate. Add "
            "curves to it, or remove it from the group."
        )
    offsets = curve_offsets(data)
    lengths = np.diff(offsets)
    short = np.flatnonzero(lengths < 2)
    if short.size:
        raise ValueError(
            f"{where}: curve {int(short[0])} has {int(lengths[short[0]])} "
            f"point(s) ({short.size} such curve(s) in all), and a rod segment "
            "needs two. Delete those curves or give them a second point."
        )
    types = _curve_attribute(data, "curve_type", np.int8, 0)
    off_curve = np.flatnonzero(~np.isin(types, _ON_CURVE_TYPES))
    if off_curve.size:
        kind = _CURVE_TYPE_NAMES.get(int(types[off_curve[0]]), "unknown")
        raise ValueError(
            f"{where}: curve {int(off_curve[0])} is a {kind} curve "
            f"({off_curve.size} such curve(s) in all), whose control points do "
            "not lie on the strand. In Edit Mode, select all and use Curves > "
            "Set Curve Type to make them Catmull Rom or Poly."
        )


def rod_edges(obj) -> np.ndarray:
    """``(E, 2)`` uint32 rod segments: consecutive points of every curve, plus
    the closing segment of a cyclic curve of three or more points."""
    data = obj.data
    offsets = curve_offsets(data)
    n_points = int(offsets[-1]) if offsets.size else 0
    if n_points < 2:
        return np.zeros((0, 2), dtype=np.uint32)
    first = np.arange(n_points - 1, dtype=np.uint32)
    # A pair (i, i + 1) is a segment unless point i + 1 starts another curve.
    is_segment = np.ones(n_points - 1, dtype=bool)
    interior_starts = offsets[1:-1]
    is_segment[interior_starts[interior_starts > 0] - 1] = False
    first = first[is_segment]
    edges = np.stack([first, first + 1], axis=1)
    cyclic = _curve_attribute(data, "cyclic", bool, False)
    lengths = np.diff(offsets)
    closing = np.flatnonzero(cyclic & (lengths >= 3))
    if closing.size:
        closing_edges = np.stack(
            [offsets[closing + 1] - 1, offsets[closing]], axis=1,
        ).astype(np.uint32)
        edges = np.concatenate([edges, closing_edges], axis=0)
    return np.ascontiguousarray(edges, dtype=np.uint32)


def _positions_of(data) -> np.ndarray:
    n = len(data.points)
    flat = np.empty(n * 3, dtype=np.float32)
    data.attributes["position"].data.foreach_get("vector", flat)
    return flat.reshape(n, 3)


def local_positions(obj) -> np.ndarray:
    """``(N, 3)`` float32 object-local positions of *obj*'s original points."""
    return _positions_of(obj.data)


def root_indices(obj) -> np.ndarray:
    """Point index of the first point of every curve."""
    return curve_offsets(obj.data)[:-1].astype(np.int64)


def root_segment_indices(obj) -> np.ndarray:
    """Point indices of the first two points of every curve, in point order:
    the segment the roots pin holds. A curve of a single point contributes
    that point alone."""
    offsets = curve_offsets(obj.data).astype(np.int64)
    first = offsets[:-1]
    second = first + 1
    second = second[second < offsets[1:]]
    return np.sort(np.concatenate([first, second]))


# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------


def _is_pin_attribute(attribute) -> bool:
    """True when *attribute* can store a pin: a point attribute that is a
    boolean, or a float Blender gives no meaning of its own."""
    if attribute.domain != "POINT" or attribute.name.startswith("."):
        return False
    if attribute.data_type == "BOOLEAN":
        return True
    return (
        attribute.data_type == "FLOAT"
        and attribute.name not in _RESERVED_FLOAT_ATTRIBUTES
    )


def _pin_attribute(obj, name):
    """The point attribute that stores the pin *name*, or ``None``."""
    if not name:
        return None
    attribute = obj.data.attributes.get(name)
    if attribute is None or not _is_pin_attribute(attribute):
        return None
    return attribute


def _float_values(attribute) -> np.ndarray:
    values = np.empty(len(attribute.data), dtype=np.float32)
    attribute.data.foreach_get("value", values)
    return values


def pin_names(obj) -> list:
    """Names a pin on *obj* can take: every point attribute that can store
    one, then the derived roots pin unless an attribute already carries its
    name."""
    names = [a.name for a in obj.data.attributes if _is_pin_attribute(a)]
    if ROOTS_PIN_NAME not in names:
        names.append(ROOTS_PIN_NAME)
    return names


def pin_exists(obj, name) -> bool:
    """True when *obj* can resolve the pin *name*. The roots pin always can."""
    return bool(name) and (
        name == ROOTS_PIN_NAME or _pin_attribute(obj, name) is not None
    )


def pin_is_weighted(obj, name) -> bool:
    """True when the pin *name* is stored as a float attribute, whose values
    are the weights of the points it holds. Reads no point data, so a panel
    may ask on every draw."""
    attribute = _pin_attribute(obj, name)
    return attribute is not None and attribute.data_type == "FLOAT"


def pin_indices(obj, name) -> list:
    """Point indices the pin *name* holds on *obj*.

    A point attribute of that name is the pin: a boolean one holds the points
    it marks, a float one every point whose weight is above zero. With no
    such attribute, the name ``Roots`` is the first segment of every curve
    (:func:`root_segment_indices`). An attribute named ``Roots`` therefore
    replaces the derived pin, which lets an artist hold a different set
    under the same name.
    """
    attribute = _pin_attribute(obj, name)
    if attribute is not None:
        if attribute.data_type == "FLOAT":
            # A weight that is not a number compares false and holds nothing
            # here; the encoder refuses it by name (`pin_weights`).
            return np.flatnonzero(_float_values(attribute) > 0.0).tolist()
        mask = np.empty(len(obj.data.points), dtype=bool)
        attribute.data.foreach_get("value", mask)
        return np.flatnonzero(mask).tolist()
    if name == ROOTS_PIN_NAME:
        return root_segment_indices(obj).tolist()
    return []


def pin_weights(obj, name):
    """``(K,)`` float32 weight of every point the pin *name* holds, in the
    order of :func:`pin_indices`, or ``None`` for a pin that carries none: a
    boolean attribute and the derived roots.

    The weights lie in ``(0, 1]``. A value above one is read as one, since
    the pull strength is the pull at a weight of one, and a point at zero or
    below is not held at all. A weight that is not a finite number is refused
    by name: it compares false against zero, so it would drop its point from
    the pin and leave nothing to say so.
    """
    attribute = _pin_attribute(obj, name)
    if attribute is None or attribute.data_type != "FLOAT":
        return None
    values = _float_values(attribute)
    bad = np.flatnonzero(~np.isfinite(values))
    if bad.size:
        raise ValueError(
            f"Pin '{name}' on Hair Curves '{obj.name}': the weight at point "
            f"{int(bad[0])} is {float(values[bad[0]])} ({bad.size} such "
            "point(s) in all), which is not a finite number. Write a weight "
            "between 0 and 1 there."
        )
    held = values[values > 0.0]
    return np.minimum(held, np.float32(1.0))


def selected_point_mask(obj) -> np.ndarray:
    """``(N,)`` bool mask of the points selected in Edit Mode.

    Blender stores the selection as ``.selection`` on the point or the curve
    domain, as a boolean or a float, and stores "everything selected" as the
    absence of the attribute.
    """
    data = obj.data
    n_points = len(data.points)
    attribute = data.attributes.get(".selection")
    if attribute is None:
        return np.ones(n_points, dtype=bool)
    count = len(attribute.data)
    if attribute.data_type == "BOOLEAN":
        values = np.empty(count, dtype=bool)
        attribute.data.foreach_get("value", values)
    else:
        raw = np.empty(count, dtype=np.float32)
        attribute.data.foreach_get("value", raw)
        values = raw > 0.0
    if attribute.domain == "CURVE":
        return np.repeat(values, np.diff(curve_offsets(data)))
    return values


def selected_point_weights(obj):
    """``(N,)`` float32 weight of every point from a selection Blender stores
    as a float, or ``None`` when the selection carries no weight.

    A selection painted in Sculpt Mode is a float per point, and Edit Mode
    keeps it. Such a selection carries a weight only when some point is
    partly selected: one whose every value is 0 or 1 is a plain selection
    stored as floats, and makes the same pin a boolean selection does.
    """
    data = obj.data
    attribute = data.attributes.get(".selection")
    if attribute is None or attribute.data_type != "FLOAT":
        return None
    values = _float_values(attribute)
    if attribute.domain == "CURVE":
        values = np.repeat(values, np.diff(curve_offsets(data)))
    if not np.all(np.isfinite(values)):
        return None
    weights = np.clip(values, 0.0, 1.0)
    if not np.any((weights > 0.0) & (weights < 1.0)):
        return None
    return weights


def set_point_selection(obj, indices, select) -> int:
    """Select or deselect the points *indices* in Edit Mode, leaving every
    other point as it was. Returns how many points were written.

    The selection is rewritten as a boolean point attribute, which is the
    form Blender itself stores once a point is picked, so a selection kept
    per curve or as a float keeps its meaning.
    """
    data = obj.data
    mask = selected_point_mask(obj).copy()
    indices = np.asarray(sorted(indices), dtype=np.int64)
    indices = indices[(indices >= 0) & (indices < len(mask))]
    mask[indices] = bool(select)
    attribute = data.attributes.get(".selection")
    if attribute is not None and (
        attribute.domain != "POINT" or attribute.data_type != "BOOLEAN"
    ):
        data.attributes.remove(attribute)
        attribute = None
    if attribute is None:
        attribute = data.attributes.new(".selection", "BOOLEAN", "POINT")
    attribute.data.foreach_set("value", mask)
    data.update_tag()
    return int(indices.size)


def _pin_attribute_for_writing(obj, name, data_type):
    """The point attribute of *data_type* that stores the pin *name*, made
    new unless one of that type is already there. A pin of the other type is
    replaced; an attribute that is not a pin is refused by name."""
    data = obj.data
    attribute = data.attributes.get(name)
    if attribute is not None and not _is_pin_attribute(attribute):
        raise ValueError(
            f"Hair Curves '{obj.name}' already has an attribute named "
            f"'{name}' that cannot store a pin (a pin is a boolean or a float "
            "point attribute, and not one of Blender's own). Choose another "
            "name."
        )
    if attribute is not None and attribute.data_type != data_type:
        data.attributes.remove(attribute)
        attribute = None
    if attribute is None:
        attribute = data.attributes.new(name, data_type, "POINT")
    return attribute


def store_pin(obj, name, mask) -> None:
    """Store *mask* as the pin *name*, replacing a pin of that name."""
    attribute = _pin_attribute_for_writing(obj, name, "BOOLEAN")
    attribute.data.foreach_set("value", np.asarray(mask, dtype=bool))
    obj.data.update_tag()


def store_pin_weights(obj, name, weights) -> None:
    """Store *weights*, one per point of *obj*, as the weighted pin *name*,
    replacing a pin of that name. A point with a weight of zero is not held.

    Every weight must be a finite number in ``[0, 1]``, and some point must
    be held, so a pin that reads back empty is never what was written.
    """
    n_points = len(obj.data.points)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    where = f"Pin '{name}' on Hair Curves '{obj.name}'"
    if weights.size != n_points:
        raise ValueError(
            f"{where}: {weights.size} weights were given for {n_points} points."
        )
    bad = np.flatnonzero(~np.isfinite(weights) | (weights < 0.0) | (weights > 1.0))
    if bad.size:
        raise ValueError(
            f"{where}: the weight at point {int(bad[0])} is "
            f"{float(weights[bad[0]])} ({bad.size} such point(s) in all). A "
            "weight lies between 0 and 1."
        )
    if not np.any(weights > 0.0):
        raise ValueError(
            f"{where}: every weight is zero, so the pin would hold nothing."
        )
    attribute = _pin_attribute_for_writing(obj, name, "FLOAT")
    attribute.data.foreach_set("value", weights)
    obj.data.update_tag()


def store_pin_indices(obj, name, indices, weights=None) -> None:
    """Store the pin *name* as the points *indices*, refusing by name an
    index the object does not have.

    With *weights*, one per entry of *indices*, the pin is weighted: each
    listed point takes its weight and every other point none.
    """
    n_points = len(obj.data.points)
    indices = [int(i) for i in indices]
    beyond = [i for i in indices if not 0 <= i < n_points]
    if beyond:
        raise ValueError(
            f"Hair Curves '{obj.name}' has {n_points} points, so point "
            f"{beyond[0]} does not exist and cannot be pinned."
        )
    if weights is None:
        mask = np.zeros(n_points, dtype=bool)
        mask[indices] = True
        store_pin(obj, name, mask)
        return
    weights = [float(w) for w in weights]
    where = f"Pin '{name}' on Hair Curves '{obj.name}'"
    if len(weights) != len(indices):
        raise ValueError(
            f"{where}: {len(weights)} weights were given for {len(indices)} "
            "points."
        )
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"{where}: a point is listed more than once, so its weight is "
            "ambiguous."
        )
    unheld = [i for i, w in zip(indices, weights, strict=True) if w == 0.0]
    if unheld:
        raise ValueError(
            f"{where}: point {unheld[0]} is given a weight of zero, which "
            "holds nothing. Leave it out of the pin."
        )
    per_point = np.zeros(n_points, dtype=np.float32)
    per_point[indices] = weights
    store_pin_weights(obj, name, per_point)


def rename_pin(obj, old_name, new_name) -> None:
    """Rename the stored pin *old_name*. The derived roots pin has no record
    to rename, so it is refused by name."""
    attribute = _pin_attribute(obj, old_name)
    if attribute is None:
        if old_name == ROOTS_PIN_NAME:
            raise ValueError(
                f"The '{ROOTS_PIN_NAME}' pin on '{obj.name}' is derived from "
                "its curves and has no stored record to rename."
            )
        raise ValueError(
            f"Hair Curves '{obj.name}' has no pin named '{old_name}'."
        )
    if obj.data.attributes.get(new_name) is not None:
        raise ValueError(
            f"Hair Curves '{obj.name}' already has an attribute named "
            f"'{new_name}'."
        )
    attribute.name = new_name


# ---------------------------------------------------------------------------
# Where the modifier stack stops preserving the points
# ---------------------------------------------------------------------------


def _own_modifiers(obj) -> list:
    from .pc2 import MODIFIER_NAME
    return [m for m in obj.modifiers if m.name != MODIFIER_NAME]


def _topology(data) -> tuple:
    return len(data.points), len(data.curves)


def _evaluated_topology(obj, context) -> tuple:
    data = obj.evaluated_get(context.evaluated_depsgraph_get()).data
    if not isinstance(data, bpy.types.Curves):
        return (-1, -1)
    return _topology(data)


class _HiddenModifiers:
    """Hide modifiers from the viewport evaluation and restore them on exit.

    Only ``show_viewport`` of modifiers that are currently shown is touched,
    so a modifier the artist muted stays muted.
    """

    def __init__(self):
        self._toggled = []

    def hide(self, modifiers) -> None:
        for m in modifiers:
            if m.show_viewport:
                m.show_viewport = False
                self._toggled.append(m)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        for m in self._toggled:
            m.show_viewport = True
        return False


# {object pointer: (stack signature, insertion index)}. The boundary costs
# one evaluation per modifier to find and is asked once per captured frame,
# so it is remembered for as long as the stack reads the same.
_boundary_memo: dict = {}


def _pinned_to_last(modifier) -> bool:
    """True for a modifier Blender keeps at the end of the stack. Nothing can
    be placed after one, the ContactSolverCache included: a move past it is
    not performed and raises nothing."""
    return bool(getattr(modifier, "use_pin_to_last", False))


def _stack_signature(obj) -> tuple:
    return tuple(
        (m.name, m.type, bool(m.show_viewport), _pinned_to_last(m),
         getattr(getattr(m, "node_group", None), "name", None))
        for m in _own_modifiers(obj)
    ) + _topology(obj.data)


def forget_cache_boundary(obj=None) -> None:
    """Drop the remembered boundary of *obj*, or of every object."""
    if obj is None:
        _boundary_memo.clear()
    else:
        _boundary_memo.pop(obj.as_pointer(), None)


def cache_insertion_index(obj, context=None) -> int:
    """Index, over the stack without the ContactSolverCache, of the first
    modifier that changes the point or curve count. The cache sits there:
    after every modifier that keeps the points, before the first that does
    not, and last when none does.

    Measured by evaluating growing prefixes of the stack. A Hair Curves
    modifier is a Geometry Nodes group that may deform, generate children,
    or both, and nothing short of running it says which.

    A modifier pinned to the end of the stack is a boundary whatever it
    does, because the cache cannot be placed behind it. Blender's own Hair
    Dynamics is one. What it does to the hair therefore runs on top of the
    solver's output and is never part of the pose the solver starts from
    (see :func:`refuse_unseen_surface_follow`).
    """
    if context is None:
        context = bpy.context
    key = obj.as_pointer()
    signature = _stack_signature(obj)
    remembered = _boundary_memo.get(key)
    if remembered is not None and remembered[0] == signature:
        return remembered[1]

    from .pc2 import MODIFIER_NAME
    base = _topology(obj.data)
    modifiers = _own_modifiers(obj)
    index = len(modifiers)
    with _HiddenModifiers() as hidden:
        hidden.hide(m for m in obj.modifiers if m.name == MODIFIER_NAME)
        if _evaluated_topology(obj, context) != base:
            # Something changes the count. Hide everything, then show one
            # more modifier per step until the count moves.
            shown = [m for m in modifiers if m.show_viewport]
            hidden.hide(shown)
            for m in shown:
                m.show_viewport = True
                if _evaluated_topology(obj, context) != base:
                    index = modifiers.index(m)
                    break
                # Kept shown for the next prefix; __exit__ restores it too.
    for position, m in enumerate(modifiers):
        if _pinned_to_last(m):
            index = min(index, position)
            break
    _boundary_memo[key] = (signature, index)
    return index


def modifiers_after_cache_boundary(obj, context=None) -> list:
    """The modifiers the ContactSolverCache sits in front of: the first one
    that changes the point count or is pinned to the end of the stack, and
    every one after it."""
    return _own_modifiers(obj)[cache_insertion_index(obj, context):]


def _read_at_cut(obj, context, exclude_modifier_names, read):
    """What *read* returns for *obj*'s geometry in the pose the solver starts
    from, evaluated at the current frame, or ``None`` when the points cannot
    be read.

    The stack is evaluated cut where the ContactSolverCache sits
    (:func:`cache_insertion_index`), with *exclude_modifier_names* hidden as
    well. On display the cache replaces the output of everything in front of
    it and the modifiers after it run on top of the solver output, so the cut
    stack is the pose the solver's output stands in for. A modifier that
    follows the scalp in front of a modifier that generates children reaches
    the result; one after it does not.

    ``None`` means the cut stack does not return the original points, which
    takes a stack edited since the boundary was measured in a way its
    signature does not show. The boundary is measured again before giving up.
    *read* is handed the evaluated Curves data, whose points are the
    original ones in the original order, and must copy what it keeps.
    """
    if context is None:
        context = bpy.context
    from .pc2 import MODIFIER_NAME
    base = _topology(obj.data)
    # The cache replays a previous solve, which is never part of the pose
    # the next one starts from, whatever the caller listed.
    excluded = set(exclude_modifier_names) | {MODIFIER_NAME}

    def evaluate():
        # Measured before anything is hidden, since the measurement reads
        # which modifiers are shown.
        after = modifiers_after_cache_boundary(obj, context)
        with _HiddenModifiers() as hidden:
            hidden.hide(m for m in obj.modifiers if m.name in excluded)
            hidden.hide(after)
            data = obj.evaluated_get(context.evaluated_depsgraph_get()).data
            if not isinstance(data, bpy.types.Curves) or _topology(data) != base:
                return None
            return read(data)

    result = evaluate()
    if result is None:
        forget_cache_boundary(obj)
        result = evaluate()
    return result


def eval_local_positions(obj, context=None, exclude_modifier_names=()):
    """``(N, 3)`` float32 object-local positions of *obj*'s points in the pose
    the solver starts from (:func:`_read_at_cut`), or ``None`` when the
    points cannot be read."""
    return _read_at_cut(
        obj, context, exclude_modifier_names,
        lambda data: _positions_of(data).copy(),
    )


# Attribute types a per-point weight can be read from. A boolean reads as 0
# or 1, so the points a pin marks can also be the region a map covers.
_WEIGHT_DATA_TYPES = ("FLOAT", "INT", "BOOLEAN")


def eval_point_weights(obj, name, context=None) -> np.ndarray:
    """``(N,)`` float64 values of the point attribute *name*, one per point
    of *obj*, read where the pose the solver starts from is read
    (:func:`_read_at_cut`).

    That is the groom's own attribute, or one a Geometry Nodes modifier in
    front of the ContactSolverCache writes, such as a Store Named Attribute
    fed by the Spline Parameter for a value that runs from root to tip. A
    modifier behind the cache runs on the solver's output and its attributes
    are not read.

    Raises ``ValueError`` naming what keeps the values from being read: no
    such attribute, one that is not per point, one that is not a number, or
    a stack that no longer returns the original points.
    """
    where = f"Hair Curves '{obj.name}'"
    found = {}

    def read(data):
        attribute = data.attributes.get(name)
        if attribute is None:
            found["missing"] = True
            return False
        found["domain"] = attribute.domain
        found["data_type"] = attribute.data_type
        if (
            attribute.domain != "POINT"
            or attribute.data_type not in _WEIGHT_DATA_TYPES
        ):
            return False
        dtype = {"FLOAT": np.float32, "INT": np.int32, "BOOLEAN": bool}[
            attribute.data_type
        ]
        values = np.empty(len(attribute.data), dtype=dtype)
        attribute.data.foreach_get("value", values)
        return values.astype(np.float64)

    values = _read_at_cut(obj, context, (), read)
    if values is None:
        raise ValueError(
            f"{where}: its modifier stack no longer returns the points of "
            f"the groom, so attribute '{name}' cannot be matched to the points "
            "the solver receives."
        )
    if values is False:
        if found.get("missing"):
            raise ValueError(
                f"{where} has no attribute '{name}'. The attribute is read "
                "from the groom and from the modifiers in front of the "
                "first one that generates curves, so a Geometry Nodes "
                "output has to be stored there."
            )
        if found["domain"] != "POINT":
            raise ValueError(
                f"attribute '{name}' on {where} is on the {found['domain']} "
                "domain; a weight is read per point (POINT)"
            )
        raise ValueError(
            f"attribute '{name}' on {where} has type {found['data_type']}; a "
            "weight is a float, an integer or a boolean"
        )
    return values


def _surface_changes_shape(surface, context) -> bool:
    """True when the mesh *surface* changes SHAPE over the timeline, as
    opposed to moving as a whole.

    A deformer that is not Geometry Nodes (an Armature, a Lattice, ...) and
    animated shape keys are taken at their word. A Geometry Nodes modifier is
    measured instead, by comparing the evaluated mesh at the two ends of the
    timeline: Blender's own Fur setup puts one on the surface that records
    its rest shape and moves nothing, and reading its node tree cannot tell
    that from one that deforms.
    """
    from .utils import (
        _DEFORMING_MODIFIER_TYPES,
        _depsgraph_mesh_differs_across_range,
        _has_shape_key_animation,
        _mesh_shape_could_animate,
    )
    if surface is None or surface.type != "MESH":
        return False
    for m in surface.modifiers:
        if (
            m.show_viewport and m.type != "NODES"
            and m.type in _DEFORMING_MODIFIER_TYPES
        ):
            return True
    if _has_shape_key_animation(surface):
        return True
    return bool(
        _mesh_shape_could_animate(surface)
        and _depsgraph_mesh_differs_across_range(surface, context)
    )


def refuse_unseen_surface_follow(obj, group_name, context=None) -> None:
    """Raise ``ValueError`` when the hair follows a surface that changes shape
    through a modifier the solver cannot see.

    A modifier pinned to the end of the stack runs behind the
    ContactSolverCache, on top of the solver's output. While the surface the
    hair is attached to only moves as a whole, that changes nothing: the
    hair is carried by its object. When the surface changes SHAPE (an
    armature, shape keys), such a modifier is what carries each root along
    with the skin, and it does so after the solve, so the solver would hold
    the roots where the undeformed skin was while the body it collides with
    has moved on. Blender's Hair Dynamics, which its own Empty Hair and Fur
    add, is pinned this way.
    """
    surface = obj.data.surface
    if surface is None:
        return
    from .utils import _nodes_modifier_can_deform
    pinned = [
        m for m in _own_modifiers(obj)
        if m.show_viewport and _pinned_to_last(m)
        and (m.type != "NODES" or _nodes_modifier_can_deform(m))
    ]
    if not pinned:
        return
    if context is None:
        context = bpy.context
    if not _surface_changes_shape(surface, context):
        return
    name = pinned[0].name
    raise ValueError(
        f"Hair Curves '{obj.name}' in group '{group_name}' is attached to "
        f"'{surface.name}', whose shape changes over the timeline, and its "
        f"modifier '{name}' is pinned to the end of the stack, behind the "
        f"solver's cache. Whatever '{name}' does to carry the hair along "
        f"with '{surface.name}' would happen after the solve, so the solver "
        "would never see the roots move with the skin. Turn off Pin to Last "
        f"on '{name}' and move it above the first modifier that generates "
        "children, or disable it."
    )


def has_deforming_modifier_stack(obj) -> bool:
    """True when a shown modifier of *obj* can move the points the solver
    holds. Declarative and cheap enough for a panel draw.

    Only a modifier in front of the ContactSolverCache reaches the pose the
    solver starts from; one behind it runs on top of the solver's output.
    Where the cache sits is measured (:func:`cache_insertion_index`), which a
    draw may not do, so the remembered answer is used when the stack still
    reads the way it did when it was measured. Until then every modifier is
    counted, which errs toward offering a capture that records no motion.
    """
    from .utils import _nodes_modifier_can_deform
    modifiers = _own_modifiers(obj)
    remembered = _boundary_memo.get(obj.as_pointer())
    if remembered is not None and remembered[0] == _stack_signature(obj):
        modifiers = modifiers[:remembered[1]]
    for m in modifiers:
        if not getattr(m, "show_viewport", True):
            continue
        if m.type == "NODES":
            if _nodes_modifier_can_deform(m):
                return True
        else:
            return True
    return False


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------


def _cache_node_group():
    """The node group the playback modifier runs, created on first use.

    It sets each point's position from :data:`SIM_POSITION_ATTRIBUTE` and
    leaves the points alone while that attribute is absent, so a geometry
    that lost the attribute upstream keeps its shape.
    """
    for group in bpy.data.node_groups:
        if group.get(_CACHE_NODE_GROUP_TAG):
            return group
    group = bpy.data.node_groups.new(_CACHE_NODE_GROUP_NAME, "GeometryNodeTree")
    group[_CACHE_NODE_GROUP_TAG] = True
    group.is_modifier = True
    group.interface.new_socket(
        "Geometry", in_out="INPUT", socket_type="NodeSocketGeometry",
    )
    group.interface.new_socket(
        "Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry",
    )
    group_input = group.nodes.new("NodeGroupInput")
    group_output = group.nodes.new("NodeGroupOutput")
    attribute = group.nodes.new("GeometryNodeInputNamedAttribute")
    attribute.data_type = "FLOAT_VECTOR"
    attribute.inputs["Name"].default_value = SIM_POSITION_ATTRIBUTE
    set_position = group.nodes.new("GeometryNodeSetPosition")
    group_input.location = (-400.0, 0.0)
    attribute.location = (-400.0, -160.0)
    set_position.location = (-160.0, 0.0)
    group_output.location = (80.0, 0.0)
    links = group.links
    links.new(group_input.outputs[0], set_position.inputs["Geometry"])
    links.new(attribute.outputs["Exists"], set_position.inputs["Selection"])
    links.new(attribute.outputs["Attribute"], set_position.inputs["Position"])
    links.new(set_position.outputs["Geometry"], group_output.inputs[0])
    return group


def cache_modifier_is_bound(obj) -> bool:
    """True when *obj* carries a ContactSolverCache that runs the playback
    node group and sits where :func:`cache_insertion_index` puts it."""
    from .pc2 import MODIFIER_NAME
    index = obj.modifiers.find(MODIFIER_NAME)
    if index < 0:
        return False
    modifier = obj.modifiers[index]
    group = getattr(modifier, "node_group", None)
    if modifier.type != "NODES" or group is None:
        return False
    if not group.get(_CACHE_NODE_GROUP_TAG):
        return False
    return index == cache_insertion_index(obj)


def setup_cache_modifier(obj):
    """Add or repair the ContactSolverCache on *obj* and place it at the cut."""
    from .pc2 import MODIFIER_NAME
    modifier = obj.modifiers.get(MODIFIER_NAME)
    if modifier is not None and modifier.type != "NODES":
        obj.modifiers.remove(modifier)
        modifier = None
    if modifier is None:
        modifier = obj.modifiers.new(name=MODIFIER_NAME, type="NODES")
    group = _cache_node_group()
    if modifier.node_group != group:
        modifier.node_group = group
    target = cache_insertion_index(obj)
    index = obj.modifiers.find(MODIFIER_NAME)
    if index != target:
        obj.modifiers.move(index, target)
        landed = obj.modifiers.find(MODIFIER_NAME)
        if landed != target:
            # Blender declines some moves without raising. Left where it
            # is, the cache would replay the solve in front of a modifier
            # the solve already includes, or behind one it must feed.
            raise RuntimeError(
                f"Could not place the ContactSolverCache of '{obj.name}' at "
                f"index {target} of its modifier stack: Blender left it at "
                f"index {landed}."
            )
    return modifier


def apply_frame(obj, positions) -> bool:
    """Play one frame: write *positions* (``(N, 3)`` object-local) to the
    playback attribute. Returns False, writing nothing, when the object no
    longer has the point count the frame was recorded with."""
    data = obj.data
    n_points = len(data.points)
    if len(positions) != n_points:
        return False
    attribute = data.attributes.get(SIM_POSITION_ATTRIBUTE)
    if attribute is not None and (
        attribute.domain != "POINT" or attribute.data_type != "FLOAT_VECTOR"
    ):
        data.attributes.remove(attribute)
        attribute = None
    if attribute is None:
        attribute = data.attributes.new(
            SIM_POSITION_ATTRIBUTE, "FLOAT_VECTOR", "POINT",
        )
    attribute.data.foreach_set(
        "vector", np.ascontiguousarray(positions, dtype=np.float32).ravel(),
    )
    data.update_tag()
    return True


def displayed_local_positions(obj) -> np.ndarray:
    """``(N, 3)`` object-local positions the viewport shows *obj*'s own points
    at: the frame being played back when a solve is bound and shown, the
    groom otherwise. Read-only, for the overlays.

    A modifier that moves the groom in front of the cache is not followed
    before a solve exists. Reading its result means hiding the modifiers
    after the cut, which a draw is not allowed to do.
    """
    from .pc2 import MODIFIER_NAME
    data = obj.data
    modifier = obj.modifiers.get(MODIFIER_NAME)
    attribute = data.attributes.get(SIM_POSITION_ATTRIBUTE)
    if (
        modifier is not None and modifier.show_viewport
        and attribute is not None
        and attribute.domain == "POINT"
        and attribute.data_type == "FLOAT_VECTOR"
    ):
        n = len(data.points)
        flat = np.empty(n * 3, dtype=np.float32)
        attribute.data.foreach_get("vector", flat)
        return flat.reshape(n, 3)
    return _positions_of(data)


def clear_playback(obj) -> None:
    """Take back everything a solve left on *obj*: the ContactSolverCache,
    the playback attribute, and the node group once nothing uses it."""
    from .pc2 import MODIFIER_NAME
    modifier = obj.modifiers.get(MODIFIER_NAME)
    if modifier is not None:
        obj.modifiers.remove(modifier)
    attribute = obj.data.attributes.get(SIM_POSITION_ATTRIBUTE)
    if attribute is not None:
        obj.data.attributes.remove(attribute)
        obj.data.update_tag()
    forget_cache_boundary(obj)
    for group in list(bpy.data.node_groups):
        if group.get(_CACHE_NODE_GROUP_TAG) and group.users == 0:
            bpy.data.node_groups.remove(group)
