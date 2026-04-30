# File: perf.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Draw-time profiler for the addon's Panel, UIList, and viewport-overlay
handlers.

Usage from `blender_addon/debug/main.py exec -`:

    import importlib, sys
    pkg = [n for n in sys.modules if n.endswith('.ui.perf')][0].rsplit('.', 1)[0]
    m = importlib.import_module(pkg + '.perf')
    m.enable()
    # ... trigger redraws ...
    print(m.report())
    m.disable()
"""

import time

import bpy  # pyright: ignore

# __package__ is e.g. "bl_ext.user_default.ppf_contact_solver.ui" under
# the extension layout, or "ppf_contact_solver.ui" under a legacy layout.
# Strip the trailing ".ui" so _TOP is the addon's root package and the
# class-ownership filter below stays scoped to our modules.
_TOP = __package__.rsplit(".", 1)[0]

_stats: dict[str, dict] = {}
_panel_originals: dict[type, callable] = {}
_uilist_originals: dict[type, dict[str, callable]] = {}
_overlay_originals: dict[str, callable] = {}
_enabled = False


def _record(name: str, dt: float) -> None:
    entry = _stats.get(name)
    if entry is None:
        entry = {"count": 0, "total": 0.0, "max": 0.0, "min": float("inf")}
        _stats[name] = entry
    entry["count"] += 1
    entry["total"] += dt
    if dt > entry["max"]:
        entry["max"] = dt
    if dt < entry["min"]:
        entry["min"] = dt


def _mark(wrapper, name: str, fn):
    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    wrapper.__qualname__ = getattr(fn, "__qualname__", name)
    wrapper._perf_orig = fn
    return wrapper


def _wrap_no_args(name: str, fn):
    """For POST_VIEW / POST_PIXEL draw_handler callbacks (no args)."""

    def wrapper():
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            _record(name, time.perf_counter() - t0)

    return _mark(wrapper, name, fn)


def _wrap_draw(name: str, fn):
    """For Panel.draw(self, context). Blender validates the signature."""

    def wrapper(self, context):
        t0 = time.perf_counter()
        try:
            return fn(self, context)
        finally:
            _record(name, time.perf_counter() - t0)

    return _mark(wrapper, name, fn)


def _wrap_draw_item(name: str, fn):
    """For UIList.draw_item — full Blender signature."""

    def wrapper(
        self, context, layout, data, item, icon,
        active_data, active_propname, index=0, flt_flag=0,
    ):
        t0 = time.perf_counter()
        try:
            return fn(
                self, context, layout, data, item, icon,
                active_data, active_propname, index, flt_flag,
            )
        finally:
            _record(name, time.perf_counter() - t0)

    return _mark(wrapper, name, fn)


def _wrap_filter_items(name: str, fn):
    """For UIList.filter_items(self, context, data, propname)."""

    def wrapper(self, context, data, propname):
        t0 = time.perf_counter()
        try:
            return fn(self, context, data, propname)
        finally:
            _record(name, time.perf_counter() - t0)

    return _mark(wrapper, name, fn)


def _wrap_draw_filter(name: str, fn):
    """For UIList.draw_filter(self, context, layout)."""

    def wrapper(self, context, layout):
        t0 = time.perf_counter()
        try:
            return fn(self, context, layout)
        finally:
            _record(name, time.perf_counter() - t0)

    return _mark(wrapper, name, fn)


_UILIST_WRAPPERS = {
    "draw_item": _wrap_draw_item,
    "filter_items": _wrap_filter_items,
    "draw_filter": _wrap_draw_filter,
}


def _iter_subclasses(base):
    seen = set()
    stack = list(base.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        yield cls


def _is_ours(cls) -> bool:
    mod = getattr(cls, "__module__", "") or ""
    return mod == _TOP or mod.startswith(_TOP + ".")


def _patch_panels() -> int:
    n = 0
    for cls in _iter_subclasses(bpy.types.Panel):
        if not _is_ours(cls):
            continue
        draw = cls.__dict__.get("draw")
        if draw is None or getattr(draw, "_perf_orig", None) is not None:
            continue
        key = f"Panel:{cls.__name__}.draw"
        cls.draw = _wrap_draw(key, draw)
        _panel_originals[cls] = draw
        n += 1
    return n


def _patch_uilists() -> int:
    n = 0
    for cls in _iter_subclasses(bpy.types.UIList):
        if not _is_ours(cls):
            continue
        saved = {}
        for meth, wrap in _UILIST_WRAPPERS.items():
            fn = cls.__dict__.get(meth)
            if fn is None or getattr(fn, "_perf_orig", None) is not None:
                continue
            key = f"UIList:{cls.__name__}.{meth}"
            setattr(cls, meth, wrap(key, fn))
            saved[meth] = fn
        if saved:
            _uilist_originals[cls] = saved
            n += len(saved)
    return n


def _patch_overlay() -> int:
    from .dynamics import overlay as mod

    n = 0
    # Re-register the POST_VIEW / POST_PIXEL handlers so Blender picks up the
    # wrapped callables (it stores function pointers at draw_handler_add time).
    had = mod.overlay_handler is not None or mod._text_handler is not None
    if had:
        mod.unregister_overlay()

    for attr in ("draw_overlay_callback", "_draw_overlay_labels"):
        fn = getattr(mod, attr, None)
        if fn is None or getattr(fn, "_perf_orig", None) is not None:
            continue
        key = f"Overlay:{attr}"
        setattr(mod, attr, _wrap_no_args(key, fn))
        _overlay_originals[attr] = fn
        n += 1

    if had:
        mod.register_overlay()
    return n


def _unpatch_overlay() -> None:
    from .dynamics import overlay as mod

    had = mod.overlay_handler is not None or mod._text_handler is not None
    if had:
        mod.unregister_overlay()
    for attr, fn in _overlay_originals.items():
        setattr(mod, attr, fn)
    _overlay_originals.clear()
    if had:
        mod.register_overlay()


def _unpatch_panels() -> None:
    for cls, fn in _panel_originals.items():
        try:
            cls.draw = fn
        except Exception:
            pass
    _panel_originals.clear()


def _unpatch_uilists() -> None:
    for cls, methods in _uilist_originals.items():
        for meth, fn in methods.items():
            try:
                setattr(cls, meth, fn)
            except Exception:
                pass
    _uilist_originals.clear()


def enable(panels: bool = True, uilists: bool = True, overlay: bool = True) -> dict:
    """Install timing wrappers. Idempotent — safe to call repeatedly."""
    global _enabled
    counts = {"panels": 0, "uilists": 0, "overlay": 0}
    if panels:
        counts["panels"] = _patch_panels()
    if uilists:
        counts["uilists"] = _patch_uilists()
    if overlay:
        counts["overlay"] = _patch_overlay()
    _enabled = True
    _tag_redraw_all()
    return counts


def disable() -> None:
    """Restore original draw methods."""
    global _enabled
    _unpatch_overlay()
    _unpatch_panels()
    _unpatch_uilists()
    _enabled = False
    _tag_redraw_all()


def reset() -> None:
    _stats.clear()


def _tag_redraw_all() -> None:
    wm = bpy.context.window_manager
    if not wm:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def tag_redraw_all() -> None:
    """Force every area to redraw (used to gather fresh timing samples)."""
    _tag_redraw_all()


def report_json(top_n: int = 0) -> dict:
    rows = []
    for name, s in _stats.items():
        count = s["count"]
        total = s["total"]
        rows.append(
            {
                "name": name,
                "count": count,
                "total_ms": total * 1000.0,
                "mean_ms": (total / count) * 1000.0 if count else 0.0,
                "max_ms": s["max"] * 1000.0,
                "min_ms": (s["min"] if s["min"] != float("inf") else 0.0) * 1000.0,
            }
        )
    rows.sort(key=lambda r: r["total_ms"], reverse=True)
    if top_n > 0:
        rows = rows[:top_n]
    return {"enabled": _enabled, "rows": rows}


def report(top_n: int = 20) -> str:
    data = report_json(top_n)
    lines = []
    lines.append(
        f"{'name':<60s} {'count':>6s} {'total(ms)':>10s} {'mean(ms)':>9s} {'max(ms)':>9s}"
    )
    lines.append("-" * 100)
    for r in data["rows"]:
        lines.append(
            f"{r['name']:<60s} {r['count']:>6d} {r['total_ms']:>10.2f} "
            f"{r['mean_ms']:>9.3f} {r['max_ms']:>9.3f}"
        )
    if not data["rows"]:
        lines.append("(no samples — trigger a redraw after enable())")
    return "\n".join(lines)


def profile_once(iterations: int = 3) -> str:
    """Convenience: reset, redraw `iterations` times via the PyAPI region
    draw forcing trick, and return the report. The actual GPU redraw is
    asynchronous, so the caller typically needs to wait a tick or mouse
    over the panels. Prefer calling `enable()`, interacting, then
    `report()` manually when the timing matters.
    """
    reset()
    for _ in range(iterations):
        _tag_redraw_all()
    return report()
