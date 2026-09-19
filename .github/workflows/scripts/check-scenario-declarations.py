#!/usr/bin/env python3
"""Refuse a rig scenario whose ``BACKENDS`` declaration is not module-level code.

``scenarios/__init__.py`` reads the declaration with
``getattr(mod, "BACKENDS", _DEFAULT_BACKENDS)``, an attribute lookup on the
IMPORTED module. So a declaration that is not a module-level assignment is not a
declaration at all: the file still imports, the attribute is simply absent, and
the scenario names no backend at all and is REFUSED, while reading, to a grep
and to a human, exactly like one that opted in.

That is not hypothetical. A scripted pass over the scenario tree anchored its
insertion on the last ``import`` line and matched one inside a driver-script
TEMPLATE STRING, writing ``BACKENDS = ("real",)`` into text that Blender later
executes in a subprocess. Every gate stayed green, ``grep '^BACKENDS = '`` found
the line, and the only symptom was a runnable count that moved less than it
should have. This gate reads the AST instead, which cannot make that mistake.

It checks two things per file, because they fail in opposite directions:

  1. If the module-level assignment is ABSENT while the source text mentions one,
     the declaration went somewhere that does not count (a string, a function
     body, a conditional branch).
  2. If a STRING LITERAL contains one, that is the template-injection case above,
     flagged even when a correct module-level assignment also exists, since the
     string copy is dead text that will mislead the next reader.
"""

from __future__ import annotations

import ast
import io
import pathlib
import sys
import tokenize

SCENARIOS = pathlib.Path(__file__).resolve().parents[3] / "blender_addon" / "debug" / "scenarios"
NAME = "BACKENDS"


def module_level_declaration(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == NAME for t in node.targets
        ):
            return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == NAME:
            return True
    return False


def string_literals_mentioning(src: str) -> bool:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return False
    return any(t.type == tokenize.STRING and f"{NAME} = " in t.string for t in tokens)


def main() -> int:
    if not SCENARIOS.is_dir():
        print(f"check-scenario-declarations: {SCENARIOS} not found", file=sys.stderr)
        return 2

    files = sorted(p for p in SCENARIOS.glob("*.py") if p.name != "__init__.py")
    if not files:
        # A gate that parsed nothing must never report clean over it.
        print("check-scenario-declarations: FAILED -- no scenario files found",
              file=sys.stderr)
        return 1

    stray, in_string, declared = [], [], 0
    for path in files:
        src = path.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            print(f"check-scenario-declarations: {path.name} does not parse: {exc}",
                  file=sys.stderr)
            return 1
        at_module = module_level_declaration(tree)
        declared += at_module
        if not at_module and f"{NAME} = " in src:
            stray.append(path.name)
        if string_literals_mentioning(src):
            in_string.append(path.name)

    print(f"check-scenario-declarations: {len(files)} scenarios parsed, "
          f"{declared} declare {NAME} at module level")

    if stray:
        print(f"\n{len(stray)} mention {NAME} but declare none at module level, so the "
              f"declaration is invisible to getattr() and the scenario silently takes "
              f"the default (deleted) backend:", file=sys.stderr)
        for name in stray:
            print(f"  {name}", file=sys.stderr)
    if in_string:
        print(f"\n{len(in_string)} carry a {NAME} assignment inside a STRING LITERAL. A "
              f"driver-script template is not the scenario's own declaration; move it to "
              f"module level:", file=sys.stderr)
        for name in in_string:
            print(f"  {name}", file=sys.stderr)

    if stray or in_string:
        print("\ncheck-scenario-declarations: FAILED", file=sys.stderr)
        return 1
    print("check-scenario-declarations: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
