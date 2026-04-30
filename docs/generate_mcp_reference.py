#!/usr/bin/env python3
"""AST-based generator for the MCP tool reference.

Parses every ``blender_addon/mcp/handlers/*.py`` module plus
``blender_addon/mcp/blender_handlers.py`` *without importing them* (avoids
needing a running Blender environment), collects every function decorated
with one of the MCP handler decorators, and emits
``docs/blender_addon/integrations/mcp_reference.rst``.

Fails loudly (non-zero exit) if:
  - Any marker decorator is imported under an alias.  The AST matches
    decorators by bare name, so aliasing would silently drop tools.
  - A decorated function has an empty docstring.
  - A decorated function has parameters but no ``Args:`` block (or
    ``:param`` field list) in its docstring.

Run from anywhere::

    python docs/generate_mcp_reference.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from sphinx.ext.napoleon import GoogleDocstring


REPO_ROOT = Path(__file__).resolve().parent.parent
HANDLERS_DIR = REPO_ROOT / "blender_addon" / "mcp" / "handlers"
BLENDER_HANDLERS_FILE = REPO_ROOT / "blender_addon" / "mcp" / "blender_handlers.py"
OUTPUT_FILE = (
    REPO_ROOT
    / "docs"
    / "blender_addon"
    / "integrations"
    / "mcp_reference.rst"
)

MARKER_NAMES = {
    "mcp_handler",
    "connection_handler",
    "group_handler",
    "simulation_handler",
    "debug_handler",
    "remote_handler",
}

# Category key → (display label, rendering order).  The key comes from the
# source module basename (``handlers/<name>.py`` → ``<name>``;
# ``blender_handlers.py`` → ``blender``).  Unknown keys render last in
# source order, so a newly added handler module still appears.
CATEGORIES: tuple[tuple[str, str], ...] = (
    ("connection", "Connection"),
    ("group", "Group"),
    ("object_ops", "Object operations"),
    ("simulation", "Simulation"),
    ("scene", "Scene"),
    ("dyn_params", "Dynamic parameters"),
    ("remote", "Remote"),
    ("console", "Console"),
    ("debug", "Debug"),
    ("blender", "Blender"),
)


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"generate_mcp_reference: error: {msg}", file=sys.stderr)
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


def _is_tool(node: ast.AST) -> bool:
    return bool(_decorator_names(node) & MARKER_NAMES)


def _validate_import_aliases(tree: ast.Module, filename: str) -> None:
    """Reject any import that brings a marker decorator in under an alias."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in MARKER_NAMES and alias.asname is not None:
                    _fail(
                        f"{filename}:{node.lineno}: decorator "
                        f"'{alias.name}' imported under alias "
                        f"'{alias.asname}'. Import it by its bare name so "
                        f"the AST-based generator can see it."
                    )


def _signature(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a call-site signature for a free function."""
    args_src = ast.unparse(func.args)
    sig = f"({args_src})"
    if func.returns is not None:
        sig += f" -> {ast.unparse(func.returns)}"
    return sig


def _validate_docstring(
    filename: str, func: ast.FunctionDef | ast.AsyncFunctionDef
) -> str:
    doc = ast.get_docstring(func)
    if not doc or not doc.strip():
        _fail(
            f"{filename}:{func.lineno}: {func.name} is marked as an MCP "
            f"tool but has no docstring."
        )
    params: list[ast.arg] = list(func.args.args)
    params.extend(func.args.kwonlyargs)
    if func.args.vararg is not None:
        params.append(func.args.vararg)
    if func.args.kwarg is not None:
        params.append(func.args.kwarg)
    if params and "Args:" not in doc and ":param" not in doc:
        _fail(
            f"{filename}:{func.lineno}: {func.name} has parameters but "
            f"no 'Args:' block in its docstring."
        )
    return doc


def _render_docstring(doc: str) -> str:
    """Google-style docstring → reST via sphinx.ext.napoleon."""
    return str(GoogleDocstring(doc)).rstrip()


def _indent(text: str, prefix: str = "   ") -> str:
    lines = text.splitlines()
    return "\n".join(prefix + line if line else "" for line in lines)


def _render_tool(
    filename: str, func: ast.FunctionDef | ast.AsyncFunctionDef
) -> str:
    doc = _validate_docstring(filename, func)
    header = f".. py:function:: {func.name}{_signature(func)}"
    body = _render_docstring(doc)
    return "\n".join([header, "", _indent(body)])


def _category_for(path: Path) -> str:
    """Derive a category key from a handler module path."""
    if path.name == "blender_handlers.py":
        return "blender"
    return path.stem


def _collect_tools() -> dict[str, list[tuple[str, ast.FunctionDef]]]:
    """Walk every handler source file and group tools by category.

    Returns ``{category: [(filename_for_errors, func_node), ...]}``.
    """
    sources: list[Path] = sorted(HANDLERS_DIR.glob("*.py"))
    sources = [p for p in sources if not p.name.startswith("_")]
    sources.append(BLENDER_HANDLERS_FILE)

    by_category: dict[str, list[tuple[str, ast.FunctionDef]]] = {}
    for path in sources:
        if not path.exists():
            _fail(f"expected handler source at {path}")
        tree = ast.parse(path.read_text(), filename=str(path))
        _validate_import_aliases(tree, path.name)
        category = _category_for(path)
        bucket = by_category.setdefault(category, [])
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_tool(node):
                bucket.append((path.name, node))

    if not any(by_category.values()):
        _fail("no MCP-decorated functions found; refusing to emit an empty reference file.")
    return by_category


def _render_category(label: str, tools: list[tuple[str, ast.FunctionDef]]) -> str:
    underline = "-" * len(label)
    parts = [label, underline, ""]
    for filename, func in tools:
        parts.append(_render_tool(filename, func))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _build_toc(ordered: list[tuple[str, str, list[tuple[str, ast.FunctionDef]]]]) -> str:
    lines = ["**Categories:**", ""]
    for _, label, tools in ordered:
        lines.append(f"- `{label}`_ ({len(tools)})")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    by_category = _collect_tools()

    order = {key: i for i, (key, _) in enumerate(CATEGORIES)}
    labels = dict(CATEGORIES)

    ordered: list[tuple[str, str, list[tuple[str, ast.FunctionDef]]]] = []
    # Known categories first, in declared order.
    for key, label in CATEGORIES:
        if key in by_category:
            ordered.append((key, label, by_category[key]))
    # Unknown categories fall through so a newly added module still appears.
    for key in sorted(by_category):
        if key not in order:
            ordered.append((key, key.replace("_", " ").title(), by_category[key]))

    total_tools = sum(len(tools) for _, _, tools in ordered)
    header = [
        "🤖 MCP Tool Reference",
        "========================",
        "",
        ".. This file is auto-generated from blender_addon/mcp/handlers/*.py",
        ".. and blender_addon/mcp/blender_handlers.py. Do not edit by hand.",
        ".. Regenerate via:",
        "..     python docs/generate_mcp_reference.py",
        "",
        "Every tool listed here is callable over the MCP Streamable HTTP "
        "server (``POST /mcp`` with ``tools/call``), and equivalently via "
        "``bpy.ops.zozo_contact_solver.<tool_name>()`` inside Blender.",
        "See :doc:`mcp` for protocol, transport, and security notes.",
        "",
        _build_toc(ordered),
    ]

    rendered = [
        _render_category(label, tools) for _, label, tools in ordered
    ]

    output = "\n".join(header) + "\n" + "\n".join(rendered)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output)

    print(
        f"wrote {OUTPUT_FILE.relative_to(REPO_ROOT)} "
        f"({len(ordered)} categories, {total_tools} tools)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
