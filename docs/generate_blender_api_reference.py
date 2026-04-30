#!/usr/bin/env python3
"""AST-based generator for the Blender Python API reference.

Parses ``blender_addon/ops/api.py`` *without importing it* (avoids needing a
running Blender environment), collects every class decorated with
``@blender_api`` plus each public method that is also decorated with
``@blender_api``, and emits ``docs/blender_addon/integrations/python_api_reference.rst``.

Fails loudly (non-zero exit) if:
  - ``blender_api`` or ``blender_api_hide`` is imported under an alias.
    The AST matches decorators by bare name, so aliasing would silently
    drop symbols from the docs.
  - A decorated class or method has an empty docstring.
  - A decorated method has parameters other than ``self``/``cls`` but no
    ``Args:`` block (or ``:param`` field list) in its docstring.

Run from anywhere::

    python docs/generate_blender_api_reference.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

from sphinx.ext.napoleon import GoogleDocstring


REPO_ROOT = Path(__file__).resolve().parent.parent
API_FILE = REPO_ROOT / "blender_addon" / "ops" / "api.py"
OUTPUT_FILE = (
    REPO_ROOT
    / "docs"
    / "blender_addon"
    / "integrations"
    / "python_api_reference.rst"
)
MARKER_NAMES = {"blender_api", "blender_api_hide"}

# Rendering order for the reference page.  Classes not in this list are
# appended at the end so a newly added @blender_api class still shows up,
# just without an assigned priority.
CLASS_RENDER_ORDER = (
    "Solver",
    "SceneParam",
    "DynParam",
    "Group",
    "GroupParam",
    "Pin",
    "Wall",
    "Sphere",
    "ColliderParam",
)


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"generate_blender_api_reference: error: {msg}", file=sys.stderr)
    sys.exit(1)


def _decorator_names(node: ast.AST) -> set[str]:
    """Bare names of every decorator on *node*."""
    names: set[str] = set()
    for dec in getattr(node, "decorator_list", []):
        if isinstance(dec, ast.Name):
            names.add(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.add(dec.attr)
        elif isinstance(dec, ast.Call):
            target = dec.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def _is_marked(node: ast.AST) -> bool:
    return "blender_api" in _decorator_names(node)


def _is_hidden(node: ast.AST) -> bool:
    return "blender_api_hide" in _decorator_names(node)


def _is_property(node: ast.AST) -> bool:
    return "property" in _decorator_names(node)


def _validate_import_aliases(tree: ast.Module) -> None:
    """Reject any import that brings the marker decorators in under an alias."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in MARKER_NAMES and alias.asname is not None:
                    _fail(
                        f"{API_FILE.name}:{node.lineno}: decorator "
                        f"'{alias.name}' imported under alias "
                        f"'{alias.asname}'. Import it by its bare name so "
                        f"the AST-based generator can see it."
                    )


def _collect_aliases(tree: ast.Module) -> dict[str, str]:
    """Return ``{runtime_name: alias_name}`` for module-level ``Alias = _Target``.

    Only single-target ``Name = Name`` assignments count, and only when
    the target resolves to a class defined in the same module that is
    marked ``@blender_api`` (verified by the caller).
    """
    aliases: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Name)
        ):
            aliases[node.value.id] = node.targets[0].id
    return aliases


def _strip_self(args: ast.arguments) -> ast.arguments:
    """Return a copy of *args* without the implicit ``self`` / ``cls``.

    ``self`` always appears first in positional args and has no default,
    so removing it never disturbs ``defaults`` alignment.
    """
    filtered = [a for a in args.args if a.arg not in ("self", "cls")]
    return ast.arguments(
        posonlyargs=args.posonlyargs,
        args=filtered,
        vararg=args.vararg,
        kwonlyargs=args.kwonlyargs,
        kw_defaults=args.kw_defaults,
        kwarg=args.kwarg,
        defaults=args.defaults,
    )


def _apply_display_names(text: str, aliases: dict[str, str]) -> str:
    """Rewrite runtime class names (``_Pin``) to display names (``Pin``).

    Handles bare references as well as string forward references like
    ``-> "_Pin"`` which ``ast.unparse`` preserves with quotes.
    """
    for runtime, display in aliases.items():
        # Quoted forward references: "_Pin" or '_Pin' → Pin (unquoted).
        text = re.sub(
            rf"(['\"]){re.escape(runtime)}\1",
            display,
            text,
        )
        # Bare references, whole-word only so "_PinX" wouldn't match.
        text = re.sub(rf"(?<!\w){re.escape(runtime)}(?!\w)", display, text)
    return text


def _signature(
    func: ast.FunctionDef | ast.AsyncFunctionDef, aliases: dict[str, str]
) -> str:
    """Render a call-site signature: ``(arg: T = default, ...) -> Return``.

    ``self`` / ``cls`` are stripped.  Runtime class names are rewritten
    to their public alias names via *aliases*.
    """
    args_src = ast.unparse(_strip_self(func.args))
    sig = f"({args_src})"
    if func.returns is not None:
        sig += f" -> {ast.unparse(func.returns)}"
    return _apply_display_names(sig, aliases)


def _validate_method_docstring(
    owner: str, func: ast.FunctionDef | ast.AsyncFunctionDef
) -> str:
    doc = ast.get_docstring(func)
    if not doc or not doc.strip():
        _fail(
            f"{API_FILE.name}:{func.lineno}: {owner}.{func.name} is "
            f"marked @blender_api but has no docstring."
        )
    params: list[ast.arg] = [
        a for a in func.args.args if a.arg not in ("self", "cls")
    ]
    params.extend(func.args.kwonlyargs)
    if func.args.vararg is not None:
        params.append(func.args.vararg)
    if func.args.kwarg is not None:
        params.append(func.args.kwarg)
    if params and "Args:" not in doc and ":param" not in doc:
        _fail(
            f"{API_FILE.name}:{func.lineno}: {owner}.{func.name} has "
            f"parameters but no 'Args:' block in its docstring."
        )
    return doc


def _validate_class_docstring(cls: ast.ClassDef) -> str:
    doc = ast.get_docstring(cls)
    if not doc or not doc.strip():
        _fail(
            f"{API_FILE.name}:{cls.lineno}: class {cls.name} is marked "
            f"@blender_api but has no docstring."
        )
    return doc


def _render_docstring(doc: str) -> str:
    """Google-style docstring → reST via sphinx.ext.napoleon.

    Strip trailing whitespace so the generated output diffs cleanly.
    """
    return str(GoogleDocstring(doc)).rstrip()


def _indent(text: str, prefix: str = "   ") -> str:
    lines = text.splitlines()
    return "\n".join(prefix + line if line else "" for line in lines)


def _render_method(
    owner: str,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: dict[str, str],
) -> str:
    doc = _validate_method_docstring(owner, func)
    if _is_property(func):
        header = f".. py:property:: {owner}.{func.name}"
        type_line = ""
        if func.returns is not None:
            type_line = "   :type: " + _apply_display_names(
                ast.unparse(func.returns), aliases
            )
    else:
        header = f".. py:method:: {owner}.{func.name}{_signature(func, aliases)}"
        type_line = ""
    body = _render_docstring(doc)
    parts = [header, ""]
    if type_line:
        parts.append(type_line)
        parts.append("")
    parts.append(_indent(body))
    return "\n".join(parts)


def _render_class(
    cls: ast.ClassDef, display_name: str, aliases: dict[str, str]
) -> str:
    class_doc = _validate_class_docstring(cls)
    parts = [
        f".. py:class:: {display_name}",
        "",
        _indent(_render_docstring(class_doc)),
        "",
    ]
    for child in cls.body:
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _is_hidden(child):
            continue
        if not _is_marked(child):
            continue
        parts.append(_indent(_render_method(display_name, child, aliases)))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _build_toc(classes: list[tuple[str, ast.ClassDef]]) -> str:
    lines = ["**Classes:**", ""]
    for display_name, _ in classes:
        lines.append(f"- :class:`{display_name}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if not API_FILE.exists():
        _fail(f"expected api.py at {API_FILE}")

    source = API_FILE.read_text()
    tree = ast.parse(source, filename=str(API_FILE))

    _validate_import_aliases(tree)
    aliases = _collect_aliases(tree)

    classes: list[tuple[str, ast.ClassDef]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_marked(node):
            display = aliases.get(node.name, node.name)
            classes.append((display, node))

    if not classes:
        _fail(
            f"no classes in {API_FILE.name} are decorated with @blender_api; "
            "refusing to emit an empty reference file."
        )

    # Sort by the explicit render order; anything unknown sinks to the end
    # in its source-declaration order.
    order_index = {name: i for i, name in enumerate(CLASS_RENDER_ORDER)}
    classes.sort(key=lambda item: order_index.get(item[0], len(CLASS_RENDER_ORDER)))

    header = [
        "🐍 Blender Python API Reference",
        "==================================",
        "",
        ".. This file is auto-generated from blender_addon/ops/api.py.",
        ".. Do not edit by hand. Regenerate via:",
        "..     python docs/generate_blender_api_reference.py",
        "",
        "Every public class and method listed here is reachable after "
        "``from zozo_contact_solver import solver``.",
        "See :doc:`python_api` for a narrative walkthrough of the same surface.",
        "",
        _build_toc(classes),
    ]
    rendered = [_render_class(cls, display, aliases) for display, cls in classes]

    output = "\n".join(header) + "\n" + "\n".join(rendered)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output)

    method_count = sum(
        1
        for _, cls in classes
        for child in cls.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _is_marked(child)
        and not _is_hidden(child)
    )
    print(
        f"wrote {OUTPUT_FILE.relative_to(REPO_ROOT)} "
        f"({len(classes)} classes, {method_count} methods)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
