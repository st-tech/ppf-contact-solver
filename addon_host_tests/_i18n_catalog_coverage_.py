# File: addon_host_tests/_i18n_catalog_coverage_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Every English string the add-on shows is a key of ``i18n/en.json``, and
# every locale catalog carries every key.
#
# Blender translates a string only when the registered catalog holds it
# verbatim, so a string missing from ``en.json`` stays English in every
# language with nothing to say so. The Blender-side test
# (``blender_addon/tests/test_i18n.py``) checks the catalogs against each
# other and a few enum lists; it cannot see a panel label or a tooltip that
# never reached the master. This test reads the add-on source instead, the
# same places Blender reads text from:
#
#   * the first argument of ``iface_`` / ``tip_`` / ``pgettext_*`` calls;
#   * ``text=`` on layout calls (``label``, ``operator``, ``prop``, ...),
#     which Blender translates on its own;
#   * ``name=`` and ``description=`` of ``bpy.props`` properties, except
#     hidden ones, and the name and description of static enum items;
#   * ``bl_label`` / ``bl_description`` and, for a registered operator with
#     no ``bl_description``, its docstring, which Blender uses as the
#     tooltip.
#
# Only literal strings are read. A string built at run time cannot be a
# catalog key anyway, and the call that shows it is where it gets wrapped.
#
# An empty locale value is allowed and means "not translated yet" (see
# ``blender_addon/i18n/README.md``); a MISSING key is not, because a
# translator works from the locale file and cannot see it.

from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ADDON_ROOT = REPO_ROOT / "blender_addon"
I18N_DIR = ADDON_ROOT / "i18n"

# Directories that hold no user-facing text: the rig and its driver, the
# Blender-side tests, the catalogs themselves, and the MCP server, whose
# strings go to a client program rather than to the artist.
_SKIP_DIRS = {"debug", "tests", "i18n", "mcp"}

_TRANSLATE_CALLS = {
    "iface_", "tip_", "pgettext", "pgettext_iface", "pgettext_tip",
    "pgettext_rpt", "rpt_",
}
_LAYOUT_CALLS = {
    "label", "operator", "prop", "prop_enum", "menu", "popover",
    "operator_menu_enum", "template_list",
}

# Literal text that reaches a layout but is not language: a shell command
# the artist copies, and a pure format template with no words in it.
_NOT_LANGUAGE = {
    "CARGO_TARGET_DIR=target/cpu cargo build --release --features cpu",
    "{label} ({unit})",
}

# Registered operators that no button, menu or search ever shows: their
# docstrings describe mechanism for a reader of the source.
_NEVER_SHOWN_OPERATORS = {
    "PPF_OT_FramePump",
}


def _literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal(node.left), _literal(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _call_name(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _is_hidden(call):
    for kw in call.keywords:
        if kw.arg == "options":
            text = ast.unparse(kw.value)
            if "HIDDEN" in text:
                return True
    return False


def _class_assign(cls, name):
    for stmt in cls.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == name):
            return stmt.value
    return None


def _shown_strings():
    """{string: "path:line"} for every literal the add-on shows."""
    found = {}

    def add(text, path, node):
        if text is None or not text.strip() or text in _NOT_LANGUAGE:
            return
        found.setdefault(
            text, f"{path.relative_to(REPO_ROOT)}:{getattr(node, 'lineno', 0)}")

    for path in sorted(ADDON_ROOT.rglob("*.py")):
        rel = path.relative_to(ADDON_ROOT)
        if rel.parts[0] in _SKIP_DIRS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node)
                if name in _TRANSLATE_CALLS and node.args:
                    add(_literal(node.args[0]), path, node)
                if name in _LAYOUT_CALLS:
                    for kw in node.keywords:
                        if kw.arg == "text":
                            add(_literal(kw.value), path, node)
                if name.endswith("Property") and not _is_hidden(node):
                    for kw in node.keywords:
                        if kw.arg in ("name", "description"):
                            add(_literal(kw.value), path, node)
                        if (kw.arg == "items" and name == "EnumProperty"
                                and isinstance(kw.value, (ast.List, ast.Tuple))):
                            for item in kw.value.elts:
                                if isinstance(item, ast.Tuple) and len(item.elts) >= 3:
                                    add(_literal(item.elts[1]), path, item)
                                    add(_literal(item.elts[2]), path, item)
            elif isinstance(node, ast.ClassDef):
                for attr in ("bl_label", "bl_description"):
                    value = _class_assign(node, attr)
                    if value is not None:
                        add(_literal(value), path, value)
                registered = _class_assign(node, "bl_idname") is not None
                is_operator = any(
                    ast.unparse(b).split(".")[-1] == "Operator"
                    or ast.unparse(b).startswith("_")
                    or ast.unparse(b).endswith("Base")
                    for b in node.bases
                )
                if (registered and is_operator
                        and node.name not in _NEVER_SHOWN_OPERATORS
                        and _class_assign(node, "bl_description") is None):
                    doc = ast.get_docstring(node, clean=True)
                    if doc:
                        # Blender shows the docstring with its line breaks
                        # joined, which is the key it looks up.
                        add(" ".join(doc.split()), path, node)
    return found


def _messages(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k != "_meta" and isinstance(v, str)}


def test_every_shown_string_is_a_catalog_key():
    master = _messages(I18N_DIR / "en.json")
    missing = {s: where for s, where in _shown_strings().items() if s not in master}
    assert not missing, (
        f"{len(missing)} string(s) the add-on shows are not keys of "
        "i18n/en.json, so no language can translate them. Add each to en.json "
        "and to every locale catalog:\n"
        + "\n".join(f"  {where}: {s!r}" for s, where in sorted(
            missing.items(), key=lambda kv: kv[1]))
    )


def test_every_locale_carries_every_key():
    master = set(_messages(I18N_DIR / "en.json"))
    for path in sorted(I18N_DIR.glob("*.json")):
        if path.name == "en.json":
            continue
        keys = set(_messages(path))
        absent = sorted(master - keys)
        assert not absent, (
            f"{path.name} lacks {len(absent)} key(s) of en.json, so a "
            f"translator working from it cannot see them: {absent[:10]!r}"
        )
