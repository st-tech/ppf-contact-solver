# File: force_field_status.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# What the add-on says about each force field object, in one place: the
# Force Fields panel, the MCP listing (`get_force_field_settings`) and the
# Python API (`solver.get_force_fields()`) all read it, so the three cannot
# disagree about a field.
#
# A FIELD'S STATUS IS ITS REFUSAL PLUS A CHEAP CHECK OF THE CURRENT FRAME.
# `core.force_field.refusal` judges the settings. Transfer also refuses what
# only a read of the scene or the source can show: a Wind with no Air
# Density, a group reference that cannot be sent, a Volume whose file names
# no velocity grid. The checks here find those from metadata alone (Blender's
# own properties and a VDB file's header, never its voxels), so they are
# cheap enough for a panel redraw. Transfer stays the gate: it walks every
# solve frame and reads every voxel, and refuses what no current-frame check
# can see (a later frame, a setting keyed to change mid-solve).
#
# THE CURRENT FRAME'S CHECK IS A REFUSAL ONLY WHERE TRANSFER READS THAT
# FRAME. Transfer reads a Fluid Flow's source at the solve frames and nowhere
# else, so a baked domain that holds no fluid at a frame past its bake, or a
# Volume sequence that resolves no file outside its frames, is refused only
# when that frame is one of the solve frames. Elsewhere the same finding is
# reported as a note, since Transfer would not meet it. A Volume that is not
# a sequence reads the same file at every frame, so what its header shows is
# a refusal wherever the timeline stands.
#
# A FLUID FLOW'S SIZE IS AN UPPER BOUND, never a read of the grid: a frame
# travels as the solver's bricks (`core/field_eval.py` `fluid_bricks`, 8^3
# cells of velocity and density in float32, plus one uint32 table entry per
# brick), and only the bricks holding a non-zero value are kept. Which bricks
# hold one is a property of the voxels, which metadata does not show: a VDB
# file's header records how many voxels are active, but not the grid's
# background (the value every inactive voxel reads), and a staggered grid or
# Apply Density's density grid fills bricks no active velocity voxel lies
# in. So the bound is every brick of the frame's box.

from __future__ import annotations

import functools
import math
import os

from ...core.field_eval import FLUID_BRICK

# One brick of the wire format: FLUID_BRICK^3 cells of four float32 channels
# (velocity x, y, z, then density), and its uint32 table entry.
FLUID_CHANNELS = 4
BRICK_BYTES = FLUID_BRICK ** 3 * FLUID_CHANNELS * 4
TABLE_ENTRY_BYTES = 4


# --- names ------------------------------------------------------------------


def type_name(field_type: str) -> str:
    """Blender's own name for a ``FieldSettings.type`` identifier, the one its
    Physics tab shows ("Fluid Flow" for ``FLUID_FLOW``, "Magnetic" for
    ``MAGNET``), in English. Raises ``KeyError`` for an identifier Blender
    does not have."""
    import bpy  # pyright: ignore

    item = bpy.types.FieldSettings.bl_rna.properties["type"].enum_items.get(field_type)
    if item is None:
        raise KeyError(f"Blender has no force field type {field_type!r}")
    return item.name


def enum_name(owner_type, prop: str, identifier: str) -> str:
    """Blender's display name of one item of an RNA enum, in English."""
    item = owner_type.bl_rna.properties[prop].enum_items.get(identifier)
    if item is None:
        raise KeyError(f"{owner_type.__name__}.{prop} has no item {identifier!r}")
    return item.name


# --- sizes -------------------------------------------------------------------


def brick_count(resolution) -> int:
    """Bricks the solver cuts a frame of ``resolution`` cells into."""
    return math.prod((int(n) + FLUID_BRICK - 1) // FLUID_BRICK for n in resolution)


def frame_bytes_bound(resolution) -> int:
    """The most bytes one fluid frame of ``resolution`` cells can take on the
    wire: its table, one entry per brick of the box, and every brick kept.
    A frame keeps only the bricks holding a non-zero value, so it is often
    smaller; which ones those are is not known without its voxels."""
    total = brick_count(resolution)
    return total * (TABLE_ENTRY_BYTES + BRICK_BYTES)


def format_size(nbytes: int) -> str:
    """A byte count for a label, in the binary units the add-on's transfer
    readout uses (KB, MB and GB of 1024)."""
    value = float(nbytes)
    for unit in ("B", "KB", "MB"):
        if value < 1024.0:
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


# --- wrapping ----------------------------------------------------------------


def wrap_lines(text: str, width: float, measure) -> list[str]:
    """``text`` cut into lines no wider than ``width`` as ``measure`` (a str
    to its drawn width) reads them: at spaces where it can, inside a word
    only when the word alone is too wide, which is how a language written
    without spaces wraps too. Never an empty list for a non-empty text."""
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else f"{current} {word}"
        if measure(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        # The word alone does not fit: cut it where the width runs out.
        while measure(word) > width and len(word) > 1:
            cut = len(word) - 1
            while cut > 1 and measure(word[:cut]) > width:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        current = word
    if current or not lines:
        lines.append(current)
    return lines


# Space a nested label loses to the panel margins and to each box around
# it, and the icon in front of it, in UI units at scale 1.
_PANEL_MARGIN_PX = 28.0
_BOX_MARGIN_PX = 12.0
_ICON_PX = 22.0
# The width a label is wrapped to when there is no region to measure (a
# draw outside the viewport's sidebar), in characters.
_FALLBACK_CHARS = 64


def draw_wrapped(layout, context, text: str, icon: str = "NONE", depth: int = 1) -> None:
    """Draw ``text`` across as many label rows as the sidebar's width needs.

    A label renders one line and Blender clips what does not fit without
    marking the cut, and a refusal's actionable end ("turn it off") is what
    gets cut. ``icon`` goes on the first row and a blank of the same width on
    the rest, so the block reads as one message. ``depth`` is how many boxes
    the label sits in. ``text`` is drawn as given, already translated.
    """
    region = getattr(context, "region", None)
    width = float(getattr(region, "width", 0) or 0)
    if width > 0.0:
        import blf  # pyright: ignore

        prefs = context.preferences
        # 0 in a background session, which draws at scale 1.
        scale = float(prefs.system.ui_scale) or 1.0
        points = float(prefs.ui_styles[0].widget.points)
        avail = width - scale * (_PANEL_MARGIN_PX + depth * _BOX_MARGIN_PX + _ICON_PX)
        blf.size(0, points * scale)
        lines = wrap_lines(text, max(avail, 40.0 * scale), lambda s: blf.dimensions(0, s)[0])
    else:
        lines = wrap_lines(text, _FALLBACK_CHARS, len)
    blank = "NONE" if icon == "NONE" else "BLANK1"
    for i, line in enumerate(lines):
        layout.label(text=line, icon=icon if i == 0 else blank, translate=False)


# --- the field's source, and what Transfer would refuse -----------------------


def static_refusal(obj) -> str | None:
    """Why ``obj``'s field settings cannot be sent, or None: the core's
    judgment, the one Transfer applies first."""
    from ...core import force_field as ff

    return ff.refusal(obj)


def _vdb_header(path: str):
    """``{grid name: {"type", "class", "bbox", "mantaflow"}}`` from the file's
    header, its voxels never read. Raises ValueError naming the file."""
    try:
        st = os.stat(path)
    except OSError as e:
        raise ValueError(f"cannot read {path}: {e.strerror or e}") from e
    return _vdb_header_cached(path, st.st_mtime_ns, st.st_size)


@functools.lru_cache(maxsize=64)
def _vdb_header_cached(path: str, mtime_ns: int, size: int):
    # Keyed on the file's stat, so a rewritten file is read again; a
    # sequence playing back reads each frame's header once.
    del mtime_ns, size
    from ...core.force_field import MANTAFLOW_CACHE_METADATA

    try:
        import openvdb  # pyright: ignore  (Blender ships it)
    except ImportError as e:
        raise ValueError("this Blender has no openvdb module to read the file with") from e
    try:
        grids = openvdb.readAllGridMetadata(path)
    except Exception as e:  # openvdb raises its own types
        raise ValueError(f"cannot read {path}: {e}") from e
    out = {}
    for grid in grids:
        meta = dict(grid.metadata)
        lo, hi = meta.get("file_bbox_min"), meta.get("file_bbox_max")
        bbox = None
        if lo is not None and hi is not None:
            bbox = (tuple(int(v) for v in lo), tuple(int(v) for v in hi))
        out[grid.name] = {"type": grid.valueTypeName, "class": str(grid.gridClass),
                          "bbox": bbox, "mantaflow": MANTAFLOW_CACHE_METADATA in meta}
    return out


def volume_velocity_grids(vol) -> list[str]:
    """The grid names a Volume datablock's Velocity settings name: its three
    component grids when all are set, else its one vector grid, else none."""
    parts = [vol.velocity_x_grid, vol.velocity_y_grid, vol.velocity_z_grid]
    if all(parts):
        return list(parts)
    return [vol.velocity_grid] if vol.velocity_grid else []


def header_problem(header: dict, names: list[str], use_density: bool, volume: str,
                   path: str) -> str | None:
    """Why Transfer would refuse the file of ``header`` for the grids the
    Volume named ``volume`` reads (``names``, as
    `core.force_field.volume_velocity_names` gives them), or None, in
    Transfer's own words: both ask `core.force_field.vdb_grid_problem`."""
    from ...core import force_field as ff

    grids = {name: (g["type"], bool(g.get("mantaflow"))) for name, g in header.items()}
    return ff.vdb_grid_problem(grids, names, use_density, volume, path)


def header_extent(header: dict, names: list[str]):
    """The resolution of the box the named grids' active voxels span, from
    their header bounds, the box Transfer cuts into bricks; None when a
    header does not record them, and ``(0, 0, 0)`` when no voxel is
    active."""
    lo = hi = None
    for name in names:
        grid = header.get(name)
        if grid is None or grid["bbox"] is None:
            return None
        g_lo, g_hi = grid["bbox"]
        lo = g_lo if lo is None else tuple(min(a, b) for a, b in zip(lo, g_lo))
        hi = g_hi if hi is None else tuple(max(a, b) for a, b in zip(hi, g_hi))
    if lo is None or any(b < a for a, b in zip(lo, hi)):
        return (0, 0, 0)
    return tuple(b - a + 1 for a, b in zip(lo, hi))


def _fluid_domain_modifier(obj):
    for mod in getattr(obj, "modifiers", ()):
        if mod.type == "FLUID" and mod.fluid_type == "DOMAIN":
            return mod
    return None


def fluid_info(obj, scene, state, depsgraph) -> dict | None:
    """What the panel, MCP and the Python API say about a Fluid Flow field's
    source at the current frame, from metadata only; None for another type.

    ``source`` and ``kind`` (``"domain"``, ``"volume"`` or None when the
    Domain Object is neither), ``frame``, ``solve_start`` and
    ``solve_frames`` (the first solve frame and how many there are: the
    frames Transfer reads), ``resolution`` (cells along the source's x, y, z
    at this frame, or None when unknown), ``bytes_per_frame_max`` and
    ``bytes_max`` (the upper bound per frame and over every solve frame at
    this frame's resolution), ``last_read`` (what the last parameter encode
    read of it: ``frames``, ``resolution``, ``bricks``, ``brick_slots``,
    ``occupancy`` and ``bytes``, or None when no encode has read this
    source), ``problem`` (why Transfer would refuse the source, as far as
    the metadata at this frame shows, or None) and ``note`` (what the
    metadata at this frame shows wrong when Transfer does not read this
    frame, which is therefore no refusal, or None). A domain adds
    ``domain_type`` (``"GAS"`` or ``"LIQUID"``), ``cache_type``, ``baked``
    and ``cache_frames``; a Volume adds ``file``, ``sequence``,
    ``velocity_grids`` and ``density_grid``.
    """
    import bpy  # pyright: ignore
    from ...core import force_field as ff
    from ...core.encoder import resolve_start_frame

    f = obj.field
    if f.type != "FLUID_FLOW":
        return None
    frame = int(scene.frame_current)
    solve_start = int(resolve_start_frame(state))
    solve_frames = max(1, int(state.frame_count))
    # Transfer reads the source at the solve frames only (`core.force_field`
    # `encode`), so what this frame shows is its finding only at one of them.
    read_here = solve_start <= frame < solve_start + solve_frames
    # What the last encode read of THIS source: a report from the field's
    # previous Domain Object describes that object, not the current one.
    src_name = f.source_object.name if f.source_object is not None else None
    last = next((r for r in ff.volume_report()
                 if r["field"] == obj.name and r["source"] == src_name), None)
    if last is not None:
        last = {k: last[k] for k in ("frames", "resolution", "bricks", "brick_slots",
                                      "occupancy", "bytes")}
    info = {"source": None, "kind": None, "frame": frame, "solve_start": solve_start,
            "solve_frames": solve_frames, "resolution": None, "bytes_per_frame_max": None,
            "bytes_max": None, "last_read": last, "problem": None, "note": None}
    src = f.source_object
    if src is None:
        return info
    info["source"] = src.name
    evaluated = src.evaluated_get(depsgraph)
    if src.type == "VOLUME":
        info["kind"] = "volume"
        # The Velocity settings are the ORIGINAL datablock's, with its grids
        # loaded, as Transfer reads them (`core.force_field` `_volume_frame`):
        # loading is when Blender names the velocity grid the file holds, and
        # the evaluated copy keeps a stale name after a scripted edit
        # (measured). Only the file is the evaluated copy's, the one this
        # frame resolves to.
        data = src.data
        live = evaluated.data
        raw = live.grids.frame_filepath
        path = bpy.path.abspath(raw) if raw else ""
        if path and not data.grids.is_loaded:
            data.grids.load()
        velocity = volume_velocity_grids(data)
        density = ff.VDB_DENSITY_NAME if f.use_smoke_density else None
        # A sequence resolves a file per frame; another Volume reads its one
        # file at every frame, the solve frames included.
        sequence = bool(data.is_sequence)
        info.update(file=path or None, sequence=sequence, velocity_grids=velocity,
                    density_grid=density)
        refused_here = read_here or not sequence
        if not path:
            if refused_here:
                # Transfer's own words (`_volume_frame`), so this reason
                # appears in its error as given here.
                info["problem"] = (f"its Volume '{src.name}' resolves no file: "
                                   f"{ff._no_file_reason(data, live)}")
            else:
                why = live.grids.error_message
                info["note"] = (f"its Volume '{src.name}' resolves no file at frame {frame}"
                                + (f" ({why})" if why else ""))
            return info
        # The file named with its Volume, in the words Transfer refuses it in.
        at = f"its Volume '{src.name}''s file {path}"
        try:
            header = _vdb_header(path)
            why = header_problem(header, ff.volume_velocity_names(data), density is not None,
                                 src.name, path)
        except ValueError as e:
            header, why = None, str(e).replace(path, at, 1)
        if why is None:
            info["resolution"] = header_extent(header, velocity)
        elif refused_here:
            info["problem"] = why
        else:
            info["note"] = why
    else:
        original_mod = _fluid_domain_modifier(src)
        if original_mod is None:
            return info
        original = original_mod.domain_settings
        info["kind"] = "domain"
        info.update(domain_type=original.domain_type, cache_type=original.cache_type,
                    baked=bool(original.has_cache_baked_data),
                    cache_frames=[int(original.cache_frame_start),
                                  int(original.cache_frame_end)])
        # The evaluated copy holds the grid and its resolution (measured: the
        # original's read empty), as Transfer reads them.
        domain_mod = _fluid_domain_modifier(evaluated)
        domain = None if domain_mod is None else domain_mod.domain_settings
        n = (0, 0, 0) if domain is None else tuple(int(v) for v in domain.domain_resolution)
        if math.prod(n) > 0:
            info["resolution"] = n
        # A domain the view layer does not evaluate reads an empty grid,
        # baked or not, so its empty grid says nothing about the bake: the
        # field's own refusal names the cause (`core.force_field`
        # `fluid_source_refusal`), and "bake it" would send the artist the
        # wrong way.
        evaluated_here = original_mod.show_viewport and ff._evaluated_in_view_layer(src)
        if evaluated_here and (math.prod(n) == 0
                               or len(domain.velocity_grid) != 3 * math.prod(n)):
            empty = f"its Domain Object '{src.name}' holds no fluid at frame {frame}"
            if read_here:
                info["problem"] = f"{empty}; bake it over the solve frames"
            else:
                info["note"] = empty
    if info["resolution"] is not None:
        per_frame = frame_bytes_bound(info["resolution"])
        info["bytes_per_frame_max"] = per_frame
        info["bytes_max"] = per_frame * solve_frames
    return info


def scene_problem(scene, state, obj) -> str | None:
    """Why Transfer would refuse ``obj`` for the scene around it rather than
    for its own settings, or None: a Wind with no air to blow, or a target
    group that cannot be used."""
    from ...models import force_field_targets as targets

    if obj.field.type == "WIND" and float(state.air_density) <= 0.0:
        return ("a Wind field blows the air, and this scene's Air Density is 0, so it "
                "would move nothing. Raise Air Density, or use a Force field")
    apply_all, groups, _, _ = targets.source_view(state, obj)
    if apply_all or groups is None:
        return None
    for ref in groups:
        why = targets.ref_problem(scene, ref)
        if why:
            return f"it targets a group that cannot be used: {why}"
    if len(groups) == 0:
        return ("it reaches no group: Apply to All Groups is off and no group is chosen; "
                "add a group or turn Apply to All Groups back on")
    return None


def field_problem(scene, state, obj, fluid: dict | None = None) -> str | None:
    """Why Transfer would refuse ``obj`` at the current frame, or None: its
    settings first, then the scene around it, then a Fluid Flow's source
    (``fluid``, from :func:`fluid_info`)."""
    return (static_refusal(obj) or scene_problem(scene, state, obj)
            or (fluid or {}).get("problem"))


def field_row(scene, state, obj, depsgraph) -> dict:
    """One field object as the MCP listing and the Python API report it."""
    from ...core import force_field as ff
    from ...models import force_field_targets as targets

    fluid = fluid_info(obj, scene, state, depsgraph)
    row = {
        "object": obj.name,
        "type": obj.field.type,
        "type_name": type_name(obj.field.type),
        "refused": field_problem(scene, state, obj, fluid),
        "groups": targets.target_group_uuids(state, obj),
        "pushes": [p.name for p in ff.pushed_objects(scene, state, obj)],
        # Transfer samples a field at every solve frame when anything can
        # move or animate it, and always reads a fluid at every one.
        "tracked": bool(ff._could_move(obj)) or obj.field.type == "FLUID_FLOW",
    }
    if fluid is not None:
        row["fluid"] = {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in fluid.items()}
    return row


def field_rows(scene, state) -> list[dict]:
    """Every field object Transfer would consider, as :func:`field_row`."""
    import bpy  # pyright: ignore
    from ...core import force_field as ff

    depsgraph = bpy.context.evaluated_depsgraph_get()
    return [field_row(scene, state, obj, depsgraph)
            for obj in ff.field_objects(scene, state)]


# --- the panel's lines for a Fluid Flow ---------------------------------------


def fluid_lines(info: dict) -> list[tuple[str, str]]:
    """``(text, icon)`` rows describing a Fluid Flow's source, translated."""
    import bpy  # pyright: ignore
    from bpy.app.translations import pgettext_iface as iface_  # pyright: ignore

    rows: list[tuple[str, str]] = []
    if info["kind"] == "domain":
        text = (iface_("Domain Object: {name}, a gas domain") if info["domain_type"] == "GAS"
                else iface_("Domain Object: {name}, a liquid domain"))
        rows.append((text.format(name=info["source"]), "MOD_FLUIDSIM"))
        cache = iface_(enum_name(bpy.types.FluidDomainSettings, "cache_type",
                                 info["cache_type"]))
        if info["baked"]:
            start, end = info["cache_frames"]
            rows.append((iface_("Cache: {cache}, baked, frames {start} to {end}").format(
                cache=cache, start=start, end=end), "BLANK1"))
        else:
            rows.append((iface_("Cache: {cache}, not baked").format(cache=cache), "BLANK1"))
    elif info["kind"] == "volume":
        rows.append((iface_("Domain Object: {name}, a Volume").format(name=info["source"]),
                     "VOLUME_DATA"))
        if info.get("file"):
            name = os.path.basename(info["file"])
            text = (iface_("File: {file}, one of a sequence") if info["sequence"]
                    else iface_("File: {file}"))
            rows.append((text.format(file=name), "BLANK1"))
        if info.get("velocity_grids"):
            rows.append((iface_("Velocity grid: {names}").format(
                names=", ".join(info["velocity_grids"])), "BLANK1"))
    else:
        return rows
    if info["note"]:
        # A finding at a frame Transfer does not read: information, not a
        # refusal, so the row keeps its check mark.
        rows.append((iface_("Not a solve frame (the solve reads frames {first} to {last}): "
                            "{note}").format(
            first=info["solve_start"], last=info["solve_start"] + info["solve_frames"] - 1,
            note=info["note"]), "INFO"))
    n = info["resolution"]
    if n is not None:
        rows.append((iface_("Resolution at frame {frame}: {nx} x {ny} x {nz}").format(
            frame=info["frame"], nx=n[0], ny=n[1], nz=n[2]), "BLANK1"))
    if info["bytes_per_frame_max"] is not None:
        rows.append((iface_("Up to {frame_size} a frame at this resolution, "
                            "{total_size} over {count} solve frames").format(
            frame_size=format_size(info["bytes_per_frame_max"]),
            total_size=format_size(info["bytes_max"]), count=info["solve_frames"]),
            "BLANK1"))
    last = info["last_read"]
    if last is not None:
        rows.append((iface_("Last read: {frames} frames, {size}, {occupancy} of bricks "
                            "occupied").format(
            frames=last["frames"], size=format_size(last["bytes"]),
            occupancy=f"{last['occupancy']:.0%}"), "BLANK1"))
    return rows
