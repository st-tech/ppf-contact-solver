# File: example_catalog.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The Blender example scripts (`examples/blender/*.py`), as the Examples panel
# lists them: one entry per script with a `build()`, its title and description
# from that function's docstring, and one field per keyword parameter that a
# panel can edit.
#
# READ AS STRUCTURE, NEVER EXECUTED. Listing parses each script with `ast` and
# reads the signature and docstring off the tree; nothing runs until the artist
# presses Populate. A script that fails to parse is listed with the reason, so
# a broken example shows up in the panel rather than disappearing from it.
#
# A script opts out with a top-level `PANEL = False`, for one that needs files
# the add-on does not ship (`character-anim.py` plays back an OBJ sequence).
# It stays a script to run by hand.
#
# No `bpy` here, so the host tests can read the real catalog without Blender.

from __future__ import annotations

import ast
import os
from dataclasses import dataclass


# The parameter kinds a panel field can hold.
INT, FLOAT, BOOL, STR, VEC3 = "INT", "FLOAT", "BOOL", "STR", "VEC3"

_SCALARS = {"int": INT, "float": FLOAT, "bool": BOOL, "str": STR}


@dataclass(frozen=True)
class Param:
    """One keyword parameter of an example's `build()`.

    ``default`` is the script's own default, already in the field's type (an
    int default of a float parameter is a float). ``optional`` marks a
    parameter whose default is ``None``: the panel shows it with a switch, and
    a switched-off field passes ``None`` so the script picks its own value.
    """

    name: str
    kind: str
    default: object
    optional: bool

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").capitalize()


@dataclass(frozen=True)
class Example:
    """One script. ``error`` is set, and ``params`` empty, when it cannot be
    populated from the panel (a parse failure, no `build()`, or a parameter
    without a default)."""

    key: str
    path: str
    title: str
    description: str
    params: tuple[Param, ...]
    skipped: tuple[str, ...] = ()
    error: str = ""


def addon_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def examples_dir() -> str | None:
    """Where the example scripts are.

    A released extension carries them in its own `examples/` folder, which the
    release workflow fills from the repository's `examples/blender/`. An add-on
    installed from a checkout (the extension directory symlinked to
    `blender_addon/`) reads them from that checkout instead; `realpath` above is
    what resolves the symlink back to it.
    """
    root = addon_root()
    for candidate in (
        os.path.join(root, "examples"),
        os.path.join(os.path.dirname(root), "examples", "blender"),
    ):
        if os.path.isdir(candidate):
            return candidate
    return None


def _annotation_kind(node: ast.expr | None) -> tuple[str | None, bool]:
    """The scalar kind an annotation names, and whether it admits ``None``."""
    if node is None:
        return None, False
    if isinstance(node, ast.Name):
        if node.id == "tuple":
            return VEC3, False
        return _SCALARS.get(node.id), False
    if isinstance(node, ast.Constant) and node.value is None:
        return None, True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, left_none = _annotation_kind(node.left)
        right, right_none = _annotation_kind(node.right)
        return left or right, left_none or right_none
    if isinstance(node, ast.Subscript):
        head = node.value
        name = head.id if isinstance(head, ast.Name) else getattr(head, "attr", "")
        if name == "Optional":
            kind, _ = _annotation_kind(node.slice)
            return kind, True
        if name == "tuple":
            return VEC3, False
    return None, False


def _default_kind(value: object) -> str | None:
    # `bool` first: it is a subclass of `int`.
    if isinstance(value, bool):
        return BOOL
    if isinstance(value, int):
        return INT
    if isinstance(value, float):
        return FLOAT
    if isinstance(value, str):
        return STR
    if (
        isinstance(value, tuple)
        and len(value) == 3
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    ):
        return VEC3
    return None


def _param(arg: ast.arg, default: ast.expr) -> Param | None:
    try:
        value = ast.literal_eval(default)
    except (ValueError, SyntaxError):
        return None
    kind, optional = _annotation_kind(arg.annotation)
    if value is None:
        if kind is None or kind == VEC3:
            return None
        optional = True
    else:
        if kind is None:
            kind = _default_kind(value)
        elif _default_kind(value) == INT and kind == FLOAT:
            value = float(value)
        if kind is None or _default_kind(value) != kind:
            return None
        if kind == VEC3:
            value = tuple(float(v) for v in value)
    return Param(arg.arg, kind, value, optional)


def _offered(tree: ast.Module) -> bool:
    """False when the script sets a top-level `PANEL = False`."""
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "PANEL" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and node.value.value is False
        ):
            return False
    return True


def read_example(path: str) -> Example | None:
    """The script's entry, or None when it opts out of the panel."""
    key = os.path.splitext(os.path.basename(path))[0]
    fallback = key.replace("-", " ").replace("_", " ").title()
    try:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        return Example(key, path, fallback, "", (), error=f"{type(exc).__name__}: {exc}")
    if not _offered(tree):
        return None
    build = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build"),
        None,
    )
    if build is None:
        return Example(key, path, fallback, "", (), error="the script defines no build()")
    doc = ast.get_docstring(build) or ""
    paragraphs = [" ".join(p.split()) for p in doc.split("\n\n") if p.strip()]
    title = paragraphs[0].rstrip(".") if paragraphs else fallback
    description = paragraphs[1] if len(paragraphs) > 1 else ""

    args = build.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    pairs = list(zip(positional, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults))
    params: list[Param] = []
    skipped: list[str] = []
    for arg, default in pairs:
        if default is None:
            return Example(
                key, path, title, description, (),
                error=f"build() parameter '{arg.arg}' has no default",
            )
        param = _param(arg, default)
        if param is None:
            # Left at the script's own default: the panel has no field for it.
            skipped.append(arg.arg)
        else:
            params.append(param)
    return Example(key, path, title, description, tuple(params), tuple(skipped))


def read_catalog(directory: str | None = None) -> list[Example]:
    """Every `*.py` in the examples directory that the panel offers, sorted by
    file name."""
    directory = directory or examples_dir()
    if not directory or not os.path.isdir(directory):
        return []
    names = sorted(n for n in os.listdir(directory) if n.endswith(".py") and not n.startswith("_"))
    found = (read_example(os.path.join(directory, n)) for n in names)
    return [e for e in found if e is not None]
