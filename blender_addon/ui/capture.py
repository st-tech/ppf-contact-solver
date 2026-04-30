# File: capture.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Screenshot + widget-locator primitives for building walkthrough docs.

Three layers:

    Layer 1 — Blender primitives (require ``bpy``):
        force_redraw, screenshot, stub_panel_draw, introspect_panel,
        row_height_px, get_ui_pixel_size.

    Layer 2 — Image primitives (require NumPy + Pillow):
        diff_bbox, annotate_image.

    Layer 3 — High-level locator:
        WidgetLocator.widget_rect(panel_idname, draw_string) -> Rect.

The locator mixes two techniques that we verified work on this addon:

  * ``UILayout.introspect()`` on the wrapped ``Panel.draw`` returns a full
    widget tree keyed by ``draw_string`` — the exact label the user sees.
    That gives us the target's row/column structure.

  * Pixel-diffing a stubbed ``draw`` against the real baseline nails down
    the panel's pixel-space extent (Blender doesn't reflow siblings when
    one panel shrinks).

Typical usage (inside Blender, e.g. via ``blender_addon/debug/main.py exec``)::

    import importlib, sys
    pkg = [n for n in sys.modules if n.endswith('.ui.capture')][0].rsplit('.', 1)[0]
    cap = importlib.import_module(pkg + '.capture')

    baseline = '/tmp/shots/base.png'
    cap.screenshot(baseline)

    loc = cap.WidgetLocator(baseline, '/tmp/shots')
    rect = loc.widget_rect('SSH_PT_SolverPanel', 'Transfer')

    cap.annotate_image(
        baseline,
        [{'rect': rect, 'style': 'box', 'padding': 2}],
        '/tmp/shots/out.png',
        crop=rect.expand(400),
    )

Notes on collapsed panels
-------------------------

Panels with ``bl_options = {'DEFAULT_CLOSED'}`` (e.g. Snap-and-Merge,
Visualization) don't render a body in a fresh Blender session — their
content only draws when the user clicks the panel header. As a result:

* ``introspect_panel(panel)`` returns an empty tree.
* ``stub_panel_draw(panel, stub_noop)`` vs. baseline diff is empty, so
  ``diff_bbox`` returns ``None``.

To capture these panels either open them manually in a .blend you load
before capture, or drive the add-on interactively and expand them first.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Callable, Iterable

# bpy is only available inside Blender. The image primitives and the
# introspect-tree walker in Layer 3 work standalone, so we let the module
# import cleanly on host-side Python and raise only when a Blender-only
# function is actually called.
try:
    import bpy  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - host-side import
    bpy = None  # type: ignore[assignment]


def _require_bpy() -> None:
    if bpy is None:
        raise RuntimeError(
            "this function requires Blender's bpy module; run it inside a "
            "Blender session (e.g. via blender_addon/debug/main.py exec)."
        )


# ===========================================================================
# Layer 1 — Blender primitives
# ===========================================================================

#: Blender's canonical UI row height in "logical" pixels (see BLI_UNIT_Y in
#: source). Multiply by ``get_ui_pixel_size()`` to get screenshot pixels.
UI_UNIT_Y = 20


def get_ui_pixel_size() -> float:
    """Effective HiDPI scale — 1.0 on standard displays, 2.0 on Retina."""
    _require_bpy()
    return bpy.context.preferences.system.pixel_size


def row_height_px() -> int:
    """Pixel height of one ``row.operator(...)`` button at current DPI."""
    return int(UI_UNIT_Y * get_ui_pixel_size())


def force_redraw(iterations: int = 3) -> None:
    """Tag every area and flush Blender's draw pipeline.

    Screenshots taken from Python after changing panel state can otherwise
    return a stale front buffer. ``wm.redraw_timer`` is Blender's built-in
    "force N draws synchronously" hook — overkill for its original purpose
    (benchmarking) but the only reliable way to commit a redraw from
    Python.
    """
    _require_bpy()
    wm = bpy.context.window_manager
    if not wm:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            area.tag_redraw()
    bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=iterations)


def screenshot(path: str) -> str:
    """Write a full-window PNG to ``path`` and return the path.

    Removes any pre-existing file so callers can detect capture failure by
    the file simply not appearing.
    """
    _require_bpy()
    if os.path.exists(path):
        os.remove(path)
    bpy.ops.screen.screenshot(filepath=path)
    return path


@contextmanager
def stub_panel_draw(panel_idname: str, draw_fn: Callable):
    """Monkey-patch ``Panel.draw`` for the duration of the ``with`` block.

    On entry forces a redraw so screenshots taken inside reflect ``draw_fn``;
    on exit restores the original and redraws again.

    Raises ValueError if ``panel_idname`` isn't registered or has no
    ``draw`` method of its own (sub-panels that inherit draw don't qualify).
    """
    _require_bpy()
    panel_cls = getattr(bpy.types, panel_idname, None)
    if panel_cls is None:
        raise ValueError(f"no panel registered with bl_idname={panel_idname!r}")
    orig = panel_cls.__dict__.get("draw")
    if orig is None:
        raise ValueError(f"{panel_idname} has no own draw method to stub")
    panel_cls.draw = draw_fn
    force_redraw(iterations=3)
    try:
        yield
    finally:
        panel_cls.draw = orig
        force_redraw(iterations=1)


def introspect_panel(panel_idname: str) -> list:
    """Return the ``UILayout.introspect()`` tree produced by one redraw.

    Tree shape::

        [
          {"type": "LAYOUT_ROOT", "items": [
            {"type": "LAYOUT_BOX", "items": [
              {"type": 24, "draw_string": "", ...},            # separator
              {"type": "LAYOUT_ROW", "items": [
                {"type": 1, "draw_string": "Transfer",
                 "operator": "bpy.ops.solver.transfer()", ...},
                ...
              ]},
              ...
            ]},
            ...
          ]}
        ]
    """
    _require_bpy()
    panel_cls = getattr(bpy.types, panel_idname, None)
    if panel_cls is None:
        raise ValueError(f"no panel registered with bl_idname={panel_idname!r}")
    orig = panel_cls.__dict__.get("draw")
    captured: dict = {}

    def wrapper(self, context):
        result = orig(self, context)
        try:
            captured["tree"] = self.layout.introspect()
        except Exception as exc:
            captured["error"] = repr(exc)
        return result

    panel_cls.draw = wrapper
    try:
        force_redraw(iterations=2)
    finally:
        panel_cls.draw = orig
        force_redraw(iterations=1)

    if "error" in captured:
        raise RuntimeError(f"introspect failed: {captured['error']}")
    return captured.get("tree", [])


def stub_noop() -> Callable:
    """Renders a single empty label. Diffing this against baseline yields
    the panel's pixel extent (Blender keeps sibling panels pinned)."""
    def _draw(self, context):
        self.layout.label(text="")
    return _draw


# ===========================================================================
# Layer 2 — Image primitives (NumPy required; Pillow optional)
# ===========================================================================
#
# ``diff_bbox`` only needs NumPy: PNG decoding falls through to Blender's
# ``bpy.data.images.load`` when Pillow isn't installed, which is the default
# in Blender's bundled Python. ``annotate_image`` still needs Pillow (drawing
# text + lines onto a PNG) — it's only called host-side by the docs pipeline.


def _require_numpy() -> None:
    try:
        import numpy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "NumPy is required for image diffing. It's bundled with Blender's "
            "Python; on host Python install via `pip install numpy`."
        ) from exc


def _require_pillow() -> None:
    try:
        from PIL import Image, ImageDraw  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for annotate_image(). Install with "
            "`pip install pillow`, or call it from host-side Python after "
            "Blender has written the PNGs."
        ) from exc


def _load_rgb_int16(path: str):
    """Load a PNG at ``path`` as an HxWx3 ``np.int16`` array (top-down rows,
    RGB channels, values 0..255). Prefers Pillow (fast, small dependency),
    falls back to Blender's ``bpy.data.images.load`` when Pillow isn't
    available — so ``diff_bbox`` works inside Blender's default Python.
    """
    import numpy as np
    try:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    except ImportError:
        pass
    if bpy is None:
        raise ImportError(
            "neither Pillow nor bpy is available — cannot decode PNG"
        )
    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size
        # bpy stores images bottom-up in OpenGL orientation; pixels is a flat
        # float32 RGBA array in 0..1. Flip vertically so row 0 is the top
        # (matching PIL's orientation and our pixel-diff math).
        flat = np.asarray(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
        return (flat[::-1, :, :3] * 255.0).astype(np.int16)
    finally:
        bpy.data.images.remove(img)


class Rect(tuple):
    """Immutable ``(left, top, right, bottom)`` with convenience accessors.

    Subclass of ``tuple`` so it works transparently with PIL.Image.crop()
    and similar.
    """

    __slots__ = ()

    def __new__(cls, left, top, right, bottom):
        return super().__new__(cls, (int(left), int(top), int(right), int(bottom)))

    left = property(lambda self: self[0])
    top = property(lambda self: self[1])
    right = property(lambda self: self[2])
    bottom = property(lambda self: self[3])
    width = property(lambda self: self[2] - self[0])
    height = property(lambda self: self[3] - self[1])

    def expand(self, px: int) -> "Rect":
        return Rect(self[0] - px, self[1] - px, self[2] + px, self[3] + px)

    def shift(self, dx: int, dy: int) -> "Rect":
        return Rect(self[0] + dx, self[1] + dy, self[2] + dx, self[3] + dy)


def diff_bbox(
    path_a: str,
    path_b: str,
    threshold: int = 20,
    roi: Rect | tuple[int, int, int, int] | None = None,
) -> Rect | None:
    """Tight bbox of pixels differing by ``≥ threshold`` between two PNGs.

    Per-pixel metric is max-channel absolute difference (0..255).

    Threshold guide:
        * **5**  — catches everything including faint AA halos; useful for
          finding panel/content extents.
        * **20** — filters AA around borders while keeping the full button
          body; best default for box-inner padding.
        * **40+** — high-contrast transitions only (text glyphs, icons).

    ``roi`` optionally restricts the compare region; returned coordinates
    are still in the full-image PNG space.

    Returns ``None`` if no pixel meets the threshold.
    """
    _require_numpy()
    import numpy as np

    arr_a = _load_rgb_int16(path_a)
    arr_b = _load_rgb_int16(path_b)

    x0 = y0 = 0
    if roi is not None:
        x0, y0, x1, y1 = roi
        arr_a = arr_a[y0:y1, x0:x1]
        arr_b = arr_b[y0:y1, x0:x1]

    diff = np.abs(arr_a - arr_b).max(axis=2)
    mask = diff >= threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return Rect(int(xs.min()) + x0, int(ys.min()) + y0,
                int(xs.max()) + x0, int(ys.max()) + y0)


_DEFAULT_BOX_COLOR = (230, 30, 30)


def annotate_image(
    src_png: str,
    annotations: Iterable[dict],
    dst_png: str,
    crop: Rect | tuple[int, int, int, int] | None = None,
) -> str:
    """Render ``annotations`` on top of ``src_png`` and save to ``dst_png``.

    Each annotation is a dict. Supported styles:

        {'rect': Rect, 'style': 'box', 'color': (r,g,b), 'width': 5,
         'padding': 0}
            Stroke a rectangle. ``padding`` expands the rect outward.

        {'rect': Rect, 'style': 'caption', 'text': str, 'color': (r,g,b),
         'font_size': 28, 'offset': (dx, dy)}
            Draw a text label relative to the rect's top-left.

    ``crop`` optionally trims the output to the given rect (same pixel
    coordinates as ``src_png``).
    """
    _require_pillow()
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(src_png).convert("RGB")
    drw = ImageDraw.Draw(img)

    for ann in annotations:
        rect = ann.get("rect")
        if rect is None:
            raise ValueError(f"annotation missing 'rect': {ann}")
        style = ann.get("style", "box")
        color = tuple(ann.get("color", _DEFAULT_BOX_COLOR))
        stroke = int(ann.get("width", 5))
        pad = int(ann.get("padding", 0))
        r = rect.expand(pad) if isinstance(rect, Rect) else Rect(*rect).expand(pad)

        if style == "box":
            drw.rectangle(tuple(r), outline=color, width=stroke)

        elif style == "caption":
            font_size = int(ann.get("font_size", 28))
            dx, dy = ann.get("offset", (0, -int(font_size * 1.4)))
            try:
                font = ImageFont.truetype(
                    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                    font_size,
                )
            except (OSError, IOError):
                font = ImageFont.load_default()
            drw.text((r.left + dx, r.top + dy), ann.get("text", ""),
                     fill=color, font=font)

        else:
            raise ValueError(f"unknown annotation style: {style!r}")

    if crop is not None:
        img = img.crop(tuple(crop))
    img.save(dst_png)
    return dst_png


# ===========================================================================
# Layer 3 — High-level widget locator
# ===========================================================================


class WidgetNotRendered(RuntimeError):
    """Raised when a widget exists in the introspect tree but is not
    visibly rendered — hidden by a collapsed section, scrolled off-screen,
    icon-only with no swappable text, or a standalone label."""


def find_widget_leaf(
    tree: list, draw_string: str,
) -> tuple[dict, list[int]] | None:
    """Locate a widget leaf by its ``draw_string``. Returns
    ``(widget_dict, path)`` where ``path`` is the list of item indices from
    ``tree[0]["items"]`` down to the leaf, or ``None`` if not found.

    Walks the entire tree, including inside ``LAYOUT_BOX``, ``LAYOUT_ROW``,
    and ``LAYOUT_COLUMN`` containers.
    """
    def search(items: list, path: list[int]) -> tuple[dict, list[int]] | None:
        for i, item in enumerate(items):
            here = path + [i]
            if item.get("draw_string") == draw_string:
                return (item, here)
            sub = item.get("items")
            if isinstance(sub, list) and sub:
                got = search(sub, here)
                if got is not None:
                    return got
        return None

    for root in tree:
        if root.get("type") == "LAYOUT_ROOT":
            return search(root.get("items", []), [])
    return None


def find_widget_by_operator(
    tree: list, op_idname: str,
) -> tuple[dict, list[int]] | None:
    """Locate a widget leaf by its operator idname (e.g. ``"ssh.save_profile"``).

    Useful for icon-only buttons whose ``draw_string`` is empty.
    """
    target = f"bpy.ops.{op_idname}("

    def search(items: list, path: list[int]) -> tuple[dict, list[int]] | None:
        for i, item in enumerate(items):
            here = path + [i]
            op = item.get("operator", "")
            if isinstance(op, str) and op.startswith(target):
                return (item, here)
            sub = item.get("items")
            if isinstance(sub, list) and sub:
                got = search(sub, here)
                if got is not None:
                    return got
        return None

    for root in tree:
        if root.get("type") == "LAYOUT_ROOT":
            return search(root.get("items", []), [])
    return None


def _operator_idname_of(leaf: dict) -> str | None:
    """Return the Blender operator bl_idname referenced by this widget leaf
    (e.g. ``"solver.transfer"``), or None if the leaf isn't an operator."""
    op_str = leaf.get("operator") or ""
    if not op_str.startswith("bpy.ops."):
        return None
    return op_str[len("bpy.ops."):].split("(", 1)[0]


def _resolve_operator_class(op_idname: str):
    """Look up the ``bpy.types`` class registered for ``op_idname``. Uses
    the bound ``bpy.ops.<ns>.<name>.idname()`` result (e.g.
    ``"SOLVER_OT_transfer"``) rather than a manual case-munging of the
    idname string."""
    _require_bpy()
    try:
        ns, name = op_idname.split(".", 1)
        bound = getattr(getattr(bpy.ops, ns), name)
        type_name = bound.idname()
    except (AttributeError, ValueError):
        return None
    return getattr(bpy.types, type_name, None)


@contextmanager
def swap_property_value(obj, attr: str, new_value):
    """Temporarily set ``obj.attr`` to ``new_value`` and force a redraw,
    restoring the original value on exit. If ``new_value`` is a tuple of
    the form ``("array", idx, scalar)``, assigns just that element of an
    array-typed property (e.g., FloatVectorProperty elements).

    Example::

        state = bpy.context.scene.zozo_contact_solver.state
        with swap_property_value(state, "use_frame_rate_in_output",
                                  not state.use_frame_rate_in_output):
            screenshot(swap_png)
        # diff baseline vs swap_png → pixel rect of the checkbox
    """
    _require_bpy()
    is_array = isinstance(new_value, tuple) and len(new_value) == 3 and new_value[0] == "array"
    if is_array:
        _, idx, scalar = new_value
        arr = getattr(obj, attr)
        old = float(arr[idx])
        arr[idx] = scalar
    else:
        old = getattr(obj, attr)
        setattr(obj, attr, new_value)
    force_redraw(iterations=2)
    try:
        yield
    finally:
        if is_array:
            arr = getattr(obj, attr)
            arr[idx] = old
        else:
            setattr(obj, attr, old)
        force_redraw(iterations=1)


@contextmanager
def _install_patched_draw(panel_idname: str, transform):
    """Source-patch a panel's ``draw`` method for the duration of the
    ``with`` block. ``transform`` takes the dedented source string and
    returns the patched source; on entry we compile + install the new
    draw and force a redraw, on exit we restore the original.

    Shared machinery behind ``swap_draw_text`` and ``swap_operator_icon``.
    """
    _require_bpy()
    import inspect, textwrap

    panel_cls = getattr(bpy.types, panel_idname, None)
    if panel_cls is None:
        raise ValueError(f"no panel {panel_idname!r}")
    orig = panel_cls.__dict__.get("draw")
    if orig is None:
        raise ValueError(f"{panel_idname} has no own draw method")

    src = textwrap.dedent(inspect.getsource(orig))
    patched = transform(src)

    ns: dict = {}
    mod = inspect.getmodule(orig)
    if mod is not None:
        ns.update(vars(mod))
    exec(compile(patched, f"<patched {panel_idname}.draw>", "exec"), ns)
    patched_fn = ns.get(orig.__name__)
    if patched_fn is None:
        raise RuntimeError("failed to compile patched draw")

    panel_cls.draw = patched_fn
    force_redraw(iterations=3)
    try:
        yield
    finally:
        panel_cls.draw = orig
        force_redraw(iterations=1)


@contextmanager
def swap_operator_icon(panel_idname: str, op_idname: str, new_icon: str = "ERROR"):
    """Source-patch a panel's draw to change the icon for one operator.
    Used for icon-only buttons where bl_label swap has no visible effect.

    Finds the ``operator("op.idname", ... icon="ORIG")`` call in the
    panel source and replaces the icon value.
    """
    import re
    pattern = re.compile(
        r'(\.operator\(\s*["\']' + re.escape(op_idname) + r'["\']'
        r'[^)]*icon\s*=\s*["\'])([A-Z_]+)(["\'])',
    )

    def transform(src: str) -> str:
        patched, count = pattern.subn(r'\g<1>' + new_icon + r'\3', src)
        if count == 0:
            raise ValueError(
                f"icon pattern for {op_idname!r} not found in "
                f"{panel_idname}.draw source"
            )
        return patched

    with _install_patched_draw(panel_idname, transform):
        yield


@contextmanager
def swap_draw_text(panel_idname: str, draw_string: str, marker: str = ""):
    """Source-patch a panel's draw to replace one ``text="..."`` literal.
    Works for ``row.label(text=...)``, ``row.operator(..., text=...)``,
    ``row.prop(..., text=...)``, or any call with a ``text="..."`` kwarg.

    The marker defaults to an all-``X`` string of the SAME length as
    ``draw_string`` — preserves layout width so the diff isolates the
    widget without row reflow.
    """
    if not marker:
        marker = "X" * max(len(draw_string), 1)
    escaped = draw_string.replace('"', '\\"')
    old_text = f'text="{escaped}"'
    new_text = f'text="{marker}"'

    def transform(src: str) -> str:
        if old_text not in src:
            raise ValueError(
                f"{old_text!r} not found in {panel_idname}.draw source"
            )
        return src.replace(old_text, new_text, 1)

    with _install_patched_draw(panel_idname, transform):
        yield


@contextmanager
def swap_operator_label(op_idname: str, marker_label: str):
    """Temporarily replace a registered operator with a fresh class that has
    the same ``bl_idname`` but a distinctive ``bl_label``. Blender reads
    ``bl_label`` off the registered class at draw time, so pixel-diffing a
    screenshot taken inside this ``with`` block against a baseline reveals
    the exact pixels where the operator's button renders.

    The original class is restored on exit. Unlike in-place
    ``cls.bl_label = ...`` mutation (which Blender caches after first
    registration), unregistering + registering a fresh class forces Blender
    to re-read the label.
    """
    _require_bpy()
    orig = _resolve_operator_class(op_idname)
    if orig is None:
        raise RuntimeError(f"no registered operator with idname {op_idname!r}")

    # Fresh replacement class — must have the same bl_idname; poll() defers
    # to original so the button's enabled/disabled state (and color) matches.
    orig_poll = getattr(orig, "poll", None)

    attrs = {
        "bl_idname": op_idname,
        "bl_label": marker_label,
        "execute": lambda self, context: {"FINISHED"},
    }
    if orig_poll is not None and callable(orig_poll):
        attrs["poll"] = classmethod(lambda cls, ctx: orig_poll(ctx))
    Replacement = type(
        "CaptureReplacement_" + op_idname.replace(".", "_"),
        (bpy.types.Operator,), attrs,
    )

    bpy.utils.unregister_class(orig)
    bpy.utils.register_class(Replacement)
    force_redraw(iterations=2)
    try:
        yield
    finally:
        bpy.utils.unregister_class(Replacement)
        bpy.utils.register_class(orig)
        force_redraw(iterations=1)


#: Label used for the replacement operator when swapping. Long, all-same-
#: character, so the rendered text differs from any real addon button label
#: — guarantees a clean pixel diff.
_MARKER_LABEL = "X" * 48


def _auto_swap_value(rna_property, current_value, index: int = 0):
    """Given a Blender RNA property descriptor and its current value, return
    a distinctive alternative value that forces the widget's rendering to
    differ. For array/Vector properties, ``index`` selects the element and
    a tuple is returned (caller uses ``obj.prop[index] = new_scalar``).
    Returns None for property types we don't know how to swap."""
    t = rna_property.type
    array_len = getattr(rna_property, "array_length", 0) or 0
    is_array = array_len > 1

    if is_array:
        try:
            scalar = float(current_value[index])
        except (TypeError, IndexError, ValueError):
            return None
        if t == "INT":
            return ("array", index, int(scalar) + 12345)
        if t == "FLOAT":
            return ("array", index, scalar + 12345.0)
        if t == "BOOLEAN":
            return ("array", index, not bool(scalar))
        return None

    if t == "BOOLEAN":
        return not current_value
    if t == "INT":
        hard_max = getattr(rna_property, "hard_max", None)
        hard_min = getattr(rna_property, "hard_min", None)
        candidate = current_value + 12345
        if hard_max is not None and candidate > hard_max:
            candidate = current_value - 12345
        if hard_min is not None and candidate < hard_min:
            candidate = current_value + 1
        return int(candidate)
    if t == "FLOAT":
        hard_max = getattr(rna_property, "hard_max", None)
        hard_min = getattr(rna_property, "hard_min", None)
        candidate = float(current_value) + 12345.0
        if hard_max is not None and candidate > hard_max:
            candidate = float(current_value) - 12345.0
        if hard_min is not None and candidate < hard_min:
            # If both bounds clamp us back to the current value, the
            # swap would be a no-op — return None so the caller skips.
            mid = (hard_min + (hard_max if hard_max is not None else hard_min)) / 2
            if abs(mid - current_value) < 1e-6:
                return None
            candidate = mid
        return float(candidate)
    if t == "STRING":
        return "XXXX_CAPTURE_MARKER_XXXX"
    if t == "ENUM":
        items = [it.identifier for it in getattr(rna_property, "enum_items", [])
                 if it.identifier != current_value]
        return items[0] if items else None
    return None


def _parse_rna_ref(rna_string: str) -> tuple[str, str, int] | None:
    """Parse an introspect ``"rna"`` field like ``"SSHState.server_type[0]"``
    into ``("SSHState", "server_type", 0)``. For scalar properties the
    index is 0; for FloatVectorProperty sub-widgets the index selects the
    element (X/Y/Z). Returns None if unparseable."""
    if not rna_string or "." not in rna_string:
        return None
    class_part, rest = rna_string.split(".", 1)
    attr = rest.split("[", 1)[0]
    if not class_part or not attr:
        return None
    import re
    m = re.search(r"\[(\d+)\]", rna_string)
    idx = int(m.group(1)) if m else 0
    return (class_part, attr, idx)


def _find_prop_owner_by_rna_class(candidates: list, rna_class_name: str):
    """Given a list of candidate PropertyGroup instances, return the one
    whose ``bl_rna.name`` (or ``type.__name__``) matches ``rna_class_name``.
    Returns None if no match."""
    for obj in candidates:
        bl_rna = getattr(type(obj), "bl_rna", None) or getattr(obj, "bl_rna", None)
        if bl_rna is not None and bl_rna.name == rna_class_name:
            return obj
        if type(obj).__name__ == rna_class_name:
            return obj
    return None


def _collect_prop_owners(panel_idname: str) -> list:
    """Collect candidate PropertyGroup instances that a panel might reference.
    Walks the addon's root data and yields each PointerProperty child."""
    _require_bpy()
    results = []
    scene = bpy.context.scene
    for attr in dir(scene):
        try:
            val = getattr(scene, attr)
        except Exception:
            continue
        bl_rna = getattr(type(val), "bl_rna", None)
        if bl_rna is None:
            continue
        results.append(val)
        for prop in bl_rna.properties:
            if prop.type == "POINTER":
                try:
                    child = getattr(val, prop.identifier)
                except Exception:
                    continue
                if child is not None and getattr(type(child), "bl_rna", None):
                    results.append(child)
    return results


def _leaf_at_path(tree: list, path: list[int]) -> dict | None:
    """Navigate to the leaf at ``path`` (sequence of item indices from
    ``tree[0]["items"]``). Returns None if the path goes out of range."""
    for root in tree:
        if root.get("type") != "LAYOUT_ROOT":
            continue
        items = root.get("items", [])
        cur = None
        for i in path:
            if not (0 <= i < len(items)):
                return None
            cur = items[i]
            items = cur.get("items") if isinstance(cur.get("items"), list) else []
        return cur
    return None


def _diff_row_bands(
    path_a: str, path_b: str, roi: Rect, threshold: int = 20, gap: int = 4,
) -> list[Rect]:
    """Segment a diff image into horizontal row bands.

    Scans the diff vertically within ``roi``, groups contiguous rows of
    changed pixels (allowing gaps of up to ``gap`` unchanged rows to merge
    minor splits from anti-aliasing), and returns a list of ``Rect`` objects
    for each band in top-to-bottom order. Each Rect's left/right span only
    the changed columns in that band.
    """
    import numpy as np
    a = _load_rgb_int16(path_a)
    b = _load_rgb_int16(path_b)
    x0, y0, x1, y1 = roi
    diff = np.abs(a[y0:y1, x0:x1].astype(np.int16) - b[y0:y1, x0:x1].astype(np.int16))
    mask = diff.max(axis=2) >= threshold

    row_any = mask.any(axis=1)

    bands: list[Rect] = []
    in_band = False
    band_start = 0
    gap_count = 0
    for i in range(len(row_any)):
        if row_any[i]:
            if not in_band:
                band_start = i
                in_band = True
            gap_count = 0
        else:
            if in_band:
                gap_count += 1
                if gap_count > gap:
                    band_end = i - gap_count
                    cols = np.where(mask[band_start:band_end + 1].any(axis=0))[0]
                    if len(cols) > 0:
                        bands.append(Rect(
                            int(cols[0]) + x0, band_start + y0,
                            int(cols[-1]) + x0, band_end + y0,
                        ))
                    in_band = False
                    gap_count = 0
    if in_band:
        band_end = len(row_any) - 1 - gap_count
        cols = np.where(mask[band_start:band_end + 1].any(axis=0))[0]
        if len(cols) > 0:
            bands.append(Rect(
                int(cols[0]) + x0, band_start + y0,
                int(cols[-1]) + x0, band_end + y0,
            ))
    return bands


def _isolate_widget_band(
    baseline_png: str,
    swap_png: str,
    raw_bbox: Rect,
    panel_roi: Rect,
    leaf: dict,
    path: list[int],
    tree: list,
) -> Rect:
    """Narrow a raw diff bbox down to just the target widget's row band.

    Handles two kinds of diff spillover:

    1. **Vertical cascade** — an enum/bool change alters the layout below
       the widget. Detected by multiple row bands. Fixed by picking the
       topmost band and re-diffing at low threshold in that vertical slice.

    2. **Horizontal bleed** — a widget shares a row with unrelated widgets
       that subtly re-render. Detected by a single band that's much wider
       than the high-threshold diff. Fixed by using the high-threshold
       horizontal extent with the low-threshold vertical extent.
    """
    lo_bands = _diff_row_bands(baseline_png, swap_png, panel_roi, threshold=5, gap=4)

    if len(lo_bands) <= 1:
        # Single row diff — check for horizontal bleed by comparing
        # high-threshold (text-only) width vs low-threshold (full) width.
        hi = diff_bbox(baseline_png, swap_png, threshold=30, roi=panel_roi)
        if hi is not None and raw_bbox.width > hi.width * 3:
            # Horizontal bleed: use high-threshold X range, low-threshold Y range
            return Rect(hi.left, raw_bbox.top, hi.right, raw_bbox.bottom)
        return raw_bbox

    # Multiple bands — vertical cascade. Use high-threshold to find the
    # target widget's band (topmost changed pixels = the swapped widget).
    hi_bands = _diff_row_bands(baseline_png, swap_png, panel_roi, threshold=30, gap=4)
    if not hi_bands:
        return raw_bbox

    target_band = hi_bands[0]

    margin = 6
    band_roi = Rect(
        panel_roi.left,
        max(target_band.top - margin, panel_roi.top),
        panel_roi.right,
        min(target_band.bottom + margin, panel_roi.bottom),
    )
    refined = diff_bbox(baseline_png, swap_png, threshold=5, roi=band_roi)
    return refined if refined is not None else target_band



class WidgetLocator:
    """Resolve a widget's pixel rect from its ``draw_string``.

    Unified three-tier strategy that handles every widget type in the
    introspect tree:

    Tier 1 — **Operator buttons** (``"operator"`` key present):
        Temporarily replace the operator class with one carrying a
        distinctive ``bl_label``, screenshot, diff. The real panel layout
        is preserved exactly.

    Tier 2 — **Property-backed widgets** (``"rna"`` key present):
        Parse the ``"rna"`` field (e.g. ``"SSHState.server_type[0]"``) to
        identify the owning PropertyGroup and attribute, swap the value to
        something visually distinct, screenshot, diff.

    Tier 3 — **Static labels** (neither ``"operator"`` nor ``"rna"``):
        Source-patch the panel's ``text="..."`` literal and diff. If the
        label has no ``text=`` kwarg (e.g. a positional label) fall back
        to swapping an adjacent sibling in the same row and inferring
        the label's rect from what's left of the row band.
    """

    def __init__(self, baseline_png: str, scratch_dir: str):
        self.baseline_png = baseline_png
        self.scratch_dir = scratch_dir
        os.makedirs(scratch_dir, exist_ok=True)
        self._panel_bbox: dict[str, Rect] = {}
        self._introspect: dict[str, list] = {}
        self._prop_owners: list | None = None

    def introspect(self, panel_idname: str) -> list:
        tree = self._introspect.get(panel_idname)
        if tree is None:
            tree = introspect_panel(panel_idname)
            self._introspect[panel_idname] = tree
            # introspect wraps+restores the panel's draw. Even though the
            # resulting screenshot matches the pre-introspect state, the
            # wrap-and-restore primes Blender to cascade a full-panel
            # re-render on the next draw change. Refresh the baseline so
            # subsequent swap diffs don't mix in that cascade.
            force_redraw(iterations=3)
            screenshot(self.baseline_png)
        return tree

    @contextmanager
    def auto_expand_sections(self, panel_idname: str):
        """Temporarily expand collapsible sections in ``panel_idname``.

        Detects every bool property whose identifier starts with
        ``show_`` in the panel's introspect tree, sets them all to True,
        and refreshes the baseline so subsequent ``widget_rect`` calls
        can locate widgets that are normally hidden inside collapsed
        sections (e.g. Wind → wind_direction, wind_strength).

        Restores every toggled property on exit.

        Example::

            with loc.auto_expand_sections("SSH_PT_ObjectGroupsManager"):
                rect = loc.widget_rect("SSH_PT_ObjectGroupsManager",
                                       "Strength: 1.000")
        """
        tree = self.introspect(panel_idname)
        owners = self._get_prop_owners()
        restores: list[tuple] = []

        def walk(items):
            for item in items:
                rna = item.get("rna", "")
                parsed = _parse_rna_ref(rna)
                if parsed is not None:
                    rna_class, rna_attr, _ = parsed
                    if rna_attr.startswith("show_"):
                        owner = _find_prop_owner_by_rna_class(owners, rna_class)
                        if owner is not None:
                            bl_rna = getattr(type(owner), "bl_rna", None)
                            prop = bl_rna.properties.get(rna_attr) if bl_rna else None
                            if prop is not None and prop.type == "BOOLEAN":
                                old = getattr(owner, rna_attr)
                                if not old:
                                    setattr(owner, rna_attr, True)
                                    restores.append((owner, rna_attr, old))
                sub = item.get("items")
                if isinstance(sub, list):
                    walk(sub)

        for root in tree:
            if root.get("type") == "LAYOUT_ROOT":
                walk(root.get("items", []))

        # Re-introspect with sections expanded, refresh baseline
        self._introspect.pop(panel_idname, None)
        force_redraw(iterations=3)
        introspect_panel(panel_idname)  # prime
        force_redraw(iterations=3)
        screenshot(self.baseline_png)
        self._introspect[panel_idname] = introspect_panel(panel_idname)
        force_redraw(iterations=2)
        screenshot(self.baseline_png)

        try:
            yield
        finally:
            for owner, attr, old in restores:
                setattr(owner, attr, old)
            self._introspect.pop(panel_idname, None)
            force_redraw(iterations=2)

    def panel_bbox(self, panel_idname: str) -> Rect:
        """Pixel extent of the sidebar region containing the panel.

        Uses Blender's Region dimensions directly. The returned rect
        covers the entire UI region; it's a superset of the panel's own
        content but that's fine for diff ROIs — swaps only touch the
        target widget's pixels.
        """
        bbox = self._panel_bbox.get(panel_idname)
        if bbox is not None:
            return bbox
        _require_bpy()
        panel_cls = getattr(bpy.types, panel_idname, None)
        if panel_cls is None:
            raise ValueError(f"no panel registered with bl_idname={panel_idname!r}")
        space_type = getattr(panel_cls, "bl_space_type", "VIEW_3D")
        region_type = getattr(panel_cls, "bl_region_type", "UI")

        region = None
        img_height = 0
        scale = int(bpy.context.preferences.system.pixel_size)
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type != space_type:
                    continue
                for r in area.regions:
                    if r.type == region_type and r.width > 0 and r.height > 0:
                        region = r
                        img_height = win.height * scale
                        break
                if region is not None:
                    break
            if region is not None:
                break

        if region is None:
            raise RuntimeError(
                f"panel {panel_idname!r}: no {region_type} region found "
                f"in any {space_type} area"
            )

        # Region coords are already in device pixels (the same space as
        # the screenshot); y is from window bottom, screenshot y is from top.
        x0 = region.x
        x1 = region.x + region.width
        y1 = img_height - region.y
        y0 = img_height - (region.y + region.height)

        bbox = Rect(x0, y0, x1, y1)
        self._panel_bbox[panel_idname] = bbox
        return bbox

    def _get_prop_owners(self) -> list:
        if self._prop_owners is None:
            self._prop_owners = _collect_prop_owners("")
        return self._prop_owners

    def _ensure_panel_visible(self, panel_idname: str) -> Rect:
        panel_roi = self.panel_bbox(panel_idname)
        min_useful_height = row_height_px()
        if panel_roi.height < min_useful_height:
            raise RuntimeError(
                f"panel {panel_idname!r} is only {panel_roi.height} px tall "
                f"({tuple(panel_roi)}). Scroll or resize so it's fully visible."
            )
        return panel_roi

    def _fresh_baseline(self) -> str:
        """Take a fresh "pre-swap" screenshot and return its path.
        Each swap operation compares against a just-taken baseline so
        drift from prior operations doesn't leak into the diff."""
        pre_png = os.path.join(self.scratch_dir, "_pre_swap.png")
        force_redraw(iterations=3)
        screenshot(pre_png)
        return pre_png

    def _locate_by_operator_swap(
        self, panel_idname: str, draw_string: str, leaf: dict, panel_roi: Rect,
    ) -> Rect:
        op_idname = _operator_idname_of(leaf)
        op_cls = _resolve_operator_class(op_idname)
        if op_cls is None:
            raise WidgetNotRendered(
                f"skip: operator {op_idname!r} — class not found "
                f"(may be a built-in Blender operator)"
            )
        # Icon-only buttons (draw_string=="") or text= overrides don't
        # respond to bl_label swap — skip directly to icon / text swap.
        label_matches = draw_string != "" and op_cls.bl_label == draw_string
        if label_matches:
            pre = self._fresh_baseline()
            swap_png = os.path.join(
                self.scratch_dir,
                f"{panel_idname}__swap_{op_idname.replace('.', '_')}.png",
            )
            with swap_operator_label(op_idname, _MARKER_LABEL):
                screenshot(swap_png)
            bbox = diff_bbox(pre, swap_png, threshold=5, roi=panel_roi)
            if bbox is not None:
                return bbox

        # text= override or bl_label swap had no effect → try source-
        # patching the draw method to change the text= literal.
        if draw_string:
            bbox = self._locate_by_draw_text_swap(
                panel_idname, draw_string, panel_roi,
            )
            if bbox is not None:
                return bbox

        if draw_string != "":
            raise WidgetNotRendered(
                f"skip: {draw_string!r} ({op_idname}) — not rendered "
                f"in the current panel state"
            )

        # Icon-only button: bl_label swap has no effect. Try icon swap.
        # Use a double-swap (two different icons) so cascading layout
        # drift cancels out; the diff isolates just the icon change.
        try:
            a_png = os.path.join(
                self.scratch_dir,
                f"{panel_idname}__iconA_{op_idname.replace('.', '_')}.png",
            )
            b_png = os.path.join(
                self.scratch_dir,
                f"{panel_idname}__iconB_{op_idname.replace('.', '_')}.png",
            )
            with swap_operator_icon(panel_idname, op_idname, "ERROR"):
                screenshot(a_png)
            with swap_operator_icon(panel_idname, op_idname, "INFO"):
                screenshot(b_png)
            bbox = diff_bbox(a_png, b_png, threshold=5, roi=panel_roi)
            if bbox is not None:
                # Sanity check: an icon button is a single row. If the diff
                # covers more than 3 row-heights, it's cascade pollution —
                # isolate to the first high-threshold row band.
                max_icon_height = row_height_px() * 3
                if bbox.height > max_icon_height:
                    hi_bands = _diff_row_bands(
                        a_png, b_png, panel_roi, threshold=30, gap=4,
                    )
                    # Keep only bands whose height is plausible for an icon.
                    icon_bands = [b for b in hi_bands if b.height <= max_icon_height]
                    if icon_bands:
                        top = icon_bands[0]
                        margin = 4
                        band_roi = Rect(
                            panel_roi.left,
                            max(top.top - margin, panel_roi.top),
                            panel_roi.right,
                            min(top.bottom + margin, panel_roi.bottom),
                        )
                        refined = diff_bbox(a_png, b_png, threshold=5, roi=band_roi)
                        if refined is not None:
                            return refined
                return bbox
        except (ValueError, RuntimeError):
            pass

        raise WidgetNotRendered(
            f"skip: icon-only button ({op_idname}) — not rendered "
            f"in the current panel state"
        )

    def _locate_by_property_swap(
        self,
        panel_idname: str,
        draw_string: str,
        leaf: dict,
        path: list[int],
        tree: list,
        panel_roi: Rect,
    ) -> Rect:
        rna_ref = leaf.get("rna", "")
        parsed = _parse_rna_ref(rna_ref)
        if parsed is None:
            raise ValueError(
                f"widget {draw_string!r} has unparseable rna={rna_ref!r}"
            )
        rna_class, rna_attr, rna_idx = parsed
        owner = _find_prop_owner_by_rna_class(self._get_prop_owners(), rna_class)
        if owner is None:
            raise ValueError(
                f"no PropertyGroup matching rna class {rna_class!r} found "
                f"for widget {draw_string!r}"
            )
        bl_rna = getattr(type(owner), "bl_rna", None) or getattr(owner, "bl_rna", None)
        rna_prop = bl_rna.properties.get(rna_attr) if bl_rna else None
        if rna_prop is None:
            raise ValueError(
                f"property {rna_attr!r} not found on {rna_class!r}"
            )
        current = getattr(owner, rna_attr)
        swap_to = _auto_swap_value(rna_prop, current, index=rna_idx)
        if swap_to is None:
            raise ValueError(
                f"don't know how to swap {rna_class}.{rna_attr} "
                f"(type={rna_prop.type})"
            )

        pre = self._fresh_baseline()
        swap_png = os.path.join(
            self.scratch_dir,
            f"{panel_idname}__swap_{rna_class}_{rna_attr}_{rna_idx}.png",
        )
        with swap_property_value(owner, rna_attr, swap_to):
            screenshot(swap_png)

        bbox = diff_bbox(pre, swap_png, threshold=5, roi=panel_roi)
        if bbox is None:
            # Property swap didn't produce a visible diff — try text= swap
            if draw_string:
                bbox = self._locate_by_draw_text_swap(
                    panel_idname, draw_string, panel_roi,
                )
            if bbox is None:
                raise WidgetNotRendered(
                    f"skip: {draw_string!r} ({rna_class}.{rna_attr}) — "
                    f"not rendered in the current panel state"
                )

        # For property swaps that can cascade (enum changing the layout
        # below, bool toggling visibility of a section, or a widget sharing
        # a row with unrelated buttons), isolate the target widget's row
        # band from the raw diff by scanning for contiguous vertical bands
        # of changed pixels and selecting the one containing the widget.
        bbox = _isolate_widget_band(
            pre, swap_png, bbox, panel_roi, leaf, path, tree,
        )
        return bbox

    def _locate_by_draw_text_swap(
        self, panel_idname: str, draw_string: str, panel_roi: Rect,
    ) -> Rect | None:
        """Locate any widget by source-patching the panel's draw method
        to replace its ``text="..."`` literal.

        After several prior draw-function swaps, the swap diff can cascade
        over the whole panel (Blender re-renders more than just the text
        region). We isolate the target by taking the first high-threshold
        row band as the widget's row, then low-threshold diff within that
        row gives the full widget extent.
        """
        pre = self._fresh_baseline()
        swap_png = os.path.join(
            self.scratch_dir,
            f"{panel_idname}__text_{draw_string[:20].replace(' ', '_')}.png",
        )
        try:
            with swap_draw_text(panel_idname, draw_string):
                screenshot(swap_png)
        except ValueError:
            return None

        raw = diff_bbox(pre, swap_png, threshold=5, roi=panel_roi)
        if raw is None:
            return None

        # Check for cascade: if the low-threshold diff is much taller than
        # one row, use the first high-threshold row-band (the target text
        # change — cascading layout shifts appear below it) as the row.
        hi_bands = _diff_row_bands(pre, swap_png, panel_roi, threshold=30, gap=4)
        if not hi_bands:
            return raw
        top_band = hi_bands[0]
        if raw.height > top_band.height * 3:
            margin = 4
            band_roi = Rect(
                panel_roi.left,
                max(top_band.top - margin, panel_roi.top),
                panel_roi.right,
                min(top_band.bottom + margin, panel_roi.bottom),
            )
            refined = diff_bbox(pre, swap_png, threshold=5, roi=band_roi)
            return refined if refined is not None else top_band
        return raw

    def _locate_by_sibling_swap(
        self, panel_idname: str, draw_string: str,
        leaf: dict, path: list[int], tree: list, panel_roi: Rect,
    ) -> Rect:
        """Locate a static label by swapping an adjacent property widget
        in the same row. Uses the sibling's high-threshold diff (just the
        changed text) to determine horizontal boundary and vertical extent.
        """
        parent_path = path[:-1]
        parent = _leaf_at_path(tree, parent_path) if parent_path else None
        if parent is None and not parent_path:
            for root in tree:
                if root.get("type") == "LAYOUT_ROOT":
                    parent = root
                    break
        if parent is None or not isinstance(parent.get("items"), list):
            raise ValueError(
                f"label {draw_string!r}: can't find parent row in tree"
            )

        sibling = None
        sibling_path = None
        label_idx = path[-1]
        for i, item in enumerate(parent["items"]):
            if item is leaf:
                continue
            if item.get("rna") or _operator_idname_of(item) is not None:
                sibling = item
                sibling_path = (parent_path or []) + [i]
                break

        if sibling is None:
            raise WidgetNotRendered(
                f"skip: label {draw_string!r} — standalone label with "
                f"no swappable sibling in the same row"
            )

        # Swap the sibling's value and take a screenshot
        rna_ref = sibling.get("rna", "")
        parsed = _parse_rna_ref(rna_ref)
        if parsed is None:
            raise ValueError(
                f"label {draw_string!r}: sibling has no parseable rna"
            )
        rna_class, rna_attr, rna_idx = parsed
        owner = _find_prop_owner_by_rna_class(self._get_prop_owners(), rna_class)
        if owner is None:
            raise ValueError(
                f"label {draw_string!r}: can't find owner for {rna_class}"
            )
        bl_rna = getattr(type(owner), "bl_rna", None) or getattr(owner, "bl_rna", None)
        rna_prop = bl_rna.properties.get(rna_attr) if bl_rna else None
        if rna_prop is None:
            raise ValueError(
                f"label {draw_string!r}: property {rna_attr} not found"
            )
        current = getattr(owner, rna_attr)
        swap_to = _auto_swap_value(rna_prop, current, index=rna_idx)
        if swap_to is None:
            raise ValueError(
                f"label {draw_string!r}: can't compute swap for sibling"
            )

        pre = self._fresh_baseline()
        swap_png = os.path.join(
            self.scratch_dir,
            f"{panel_idname}__label_{rna_class}_{rna_attr}.png",
        )
        with swap_property_value(owner, rna_attr, swap_to):
            screenshot(swap_png)

        # High-threshold row bands: the first band is the sibling's own
        # row (cascading layout changes appear in later bands below).
        hi_bands = _diff_row_bands(
            pre, swap_png, panel_roi, threshold=30, gap=4,
        )
        if not hi_bands:
            raise WidgetNotRendered(
                f"skip: label {draw_string!r} — sibling "
                f"{rna_class}.{rna_attr} swap produced no visible diff"
            )
        row = hi_bands[0]

        # Label shares the row with sibling. Place label on the opposite
        # side of the sibling's changed-text band.
        sibling_idx = (sibling_path or [0])[-1]
        if label_idx < sibling_idx:
            return Rect(panel_roi.left, row.top, row.left, row.bottom)
        else:
            return Rect(row.right, row.top, panel_roi.right, row.bottom)

    def widget_rect(
        self,
        panel_idname: str,
        draw_string: str,
        *,
        op_idname: str | None = None,
    ) -> Rect:
        """Pixel rect of the widget labeled ``draw_string`` in
        ``panel_idname``.

        Pass ``op_idname`` (e.g. ``"ssh.save_profile"``) instead of
        ``draw_string`` to locate icon-only operator buttons whose
        ``draw_string`` is empty.

        Automatically dispatches to the right strategy based on the
        introspect tree:

        - Operator buttons → ``swap_operator_label`` (bl_label swap)
        - Property-backed widgets → ``swap_property_value`` (value swap)
        - Static labels → ``swap_draw_text`` with sibling-swap fallback

        Raises ``RuntimeError`` if the panel or widget isn't visible.
        """
        tree = self.introspect(panel_idname)

        if op_idname is not None:
            found = find_widget_by_operator(tree, op_idname)
            if found is None:
                raise ValueError(
                    f"operator {op_idname!r} not found in {panel_idname}"
                )
        elif draw_string == "":
            raise ValueError(
                "empty draw_string — pass op_idname= to locate "
                "icon-only buttons, or use a non-empty label"
            )
        else:
            found = find_widget_leaf(tree, draw_string)
            if found is None:
                raise ValueError(
                    f"widget {draw_string!r} not found in {panel_idname}"
                )

        leaf, path = found
        panel_roi = self._ensure_panel_visible(panel_idname)

        # Tier 1: operator button
        if _operator_idname_of(leaf) is not None:
            return self._locate_by_operator_swap(
                panel_idname, draw_string, leaf, panel_roi,
            )

        # Tier 2: property-backed widget
        if leaf.get("rna"):
            return self._locate_by_property_swap(
                panel_idname, draw_string, leaf, path, tree, panel_roi,
            )

        # Tier 3: static label — source-patch text=, fall back to sibling
        bbox = self._locate_by_draw_text_swap(
            panel_idname, draw_string, panel_roi,
        )
        if bbox is not None:
            return bbox
        return self._locate_by_sibling_swap(
            panel_idname, draw_string, leaf, path, tree, panel_roi,
        )


__all__ = [
    # Constants / env
    "UI_UNIT_Y",
    "get_ui_pixel_size",
    "row_height_px",
    # Layer 1: Blender primitives
    "force_redraw",
    "screenshot",
    "stub_panel_draw",
    "introspect_panel",
    "stub_noop",
    # Layer 2: image primitives
    "Rect",
    "diff_bbox",
    "annotate_image",
    # Layer 3: widget locator
    "WidgetNotRendered",
    "find_widget_leaf",
    "find_widget_by_operator",
    "swap_operator_label",
    "swap_property_value",
    "WidgetLocator",
]
