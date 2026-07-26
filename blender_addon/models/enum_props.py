# File: enum_props.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Structural guard against the freed-memory dropdown bug.
#
# Blender does NOT copy the strings a dynamic ``EnumProperty(items=<callable>)``
# returns; its C side holds pointers into the Python ``str`` objects. If the
# callback returns a freshly-built list and Python drops the last reference on
# return, those strings are garbage-collected and the dropdown draws freed
# memory: garbled characters and many bogus entries. The corruption is latent
# with pure-ASCII names (the freed bytes often survive to draw time) but
# DETERMINISTIC with non-ASCII, because a Japanese name allocates a separate
# per-str UTF-8 cache buffer that is freed with the object.
#
# The remedy used to be "remember to stash the returned list in a module-level
# variable" per callback, which is exactly the kind of convention that gets
# forgotten (the pin vertex-group dropdown missed it and garbled "固定"). This
# module makes it impossible to forget:
#
#   * ``dynamic_enum_items`` is the ONLY sanctioned way to write a dynamic items
#     callback. It keeps the last returned list referenced and marks the
#     callback as retention-safe.
#   * ``EnumProperty`` is a drop-in for ``bpy.props.EnumProperty`` that RAISES
#     ``TypeError`` at class-definition (import) time if ``items`` is a callable
#     that was not wrapped with ``dynamic_enum_items``. A forgotten decorator
#     therefore fails the addon load loudly instead of shipping a garbled UI.
#
# Never call ``bpy.props.EnumProperty`` directly; always import ``EnumProperty``
# from here. The ``bl_enum_props_guard`` scenario scans the source tree and
# fails if a raw ``bpy.props.EnumProperty`` reintroduces a bypass.

from __future__ import annotations

import functools

import bpy  # pyright: ignore

# Attribute stamped on a callback by ``dynamic_enum_items``; ``EnumProperty``
# gates on it. Kept private to this module so nothing can forge the mark
# without going through the decorator (which is what does the retention).
_RETAINED_MARKER = "_ppf_retained_enum_items"

# Attribute exposing the retention cell for tests / introspection.
_HOLDER_ATTR = "_ppf_enum_items_holder"


def dynamic_enum_items(fn):
    """Required decorator for every dynamic ``EnumProperty`` items callback.

    Wraps *fn* so the list it returns is kept referenced past the callback
    return (Blender reads freed string memory otherwise) and stamps the wrapper
    so :func:`EnumProperty` accepts it. Exactly the most recent list is held,
    matching the historical single-cache-per-callback behavior; the cell is
    replaced (not appended) each call so it cannot grow across redraws.
    """
    holder: list = []

    # Blender validates an items callback's signature (co_argcount) and rejects
    # anything that is not exactly ``(self, context)``, so the wrapper must
    # present those two parameters, not ``*args``. Defaults keep direct callers
    # (tests, get_snap_objects) that pass fewer arguments working.
    @functools.wraps(fn)
    def wrapper(self=None, context=None):
        items = fn(self, context)
        holder[:] = [items]
        return items

    setattr(wrapper, _RETAINED_MARKER, True)
    setattr(wrapper, _HOLDER_ATTR, holder)
    return wrapper


def is_retained(items) -> bool:
    """True when *items* is a callback marked retention-safe by
    :func:`dynamic_enum_items` (or a non-callable static items list)."""
    return (not callable(items)) or bool(getattr(items, _RETAINED_MARKER, False))


def EnumProperty(*args, **kwargs):
    """Drop-in for ``bpy.props.EnumProperty`` that forbids an unretained
    dynamic items callback.

    A callable ``items`` MUST be wrapped with :func:`dynamic_enum_items`;
    otherwise this raises ``TypeError`` at import time so the freed-memory
    dropdown bug cannot be reintroduced. A static ``items`` list (or the
    ``bpy.props``-supplied string-preset forms) passes straight through. Only
    ``items`` is gated: the ``update`` / ``get`` / ``set`` callbacks do not
    return retained strings, so they are untouched.
    """
    # ``items`` is EnumProperty's first positional parameter; accept either
    # spelling so a positional caller cannot slip past the gate.
    items = kwargs.get("items")
    if items is None and args:
        items = args[0]
    if callable(items) and not getattr(items, _RETAINED_MARKER, False):
        name = getattr(items, "__name__", repr(items))
        raise TypeError(
            "EnumProperty items callback %r must be decorated with "
            "@dynamic_enum_items (models/enum_props.py): an unretained dynamic "
            "items callback makes Blender read freed string memory, so the "
            "dropdown renders garbled entries for non-ASCII names." % name
        )
    return bpy.props.EnumProperty(*args, **kwargs)
