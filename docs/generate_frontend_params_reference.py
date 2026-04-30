#!/usr/bin/env python3
"""Generator for the JupyterLab Python API parameter / log reference pages.

Emits three auto-generated pages under ``docs/jupyterlab_api/``:

* ``simulation_parameters.rst`` from :func:`frontend._param_.app_param`
* ``material_parameters.rst`` from :func:`frontend._param_.object_param`
* ``log_channels.rst`` from the ``//``-comment blocks parsed out of
  ``src/`` ``.cu`` / ``.rs`` sources by
  :meth:`frontend._parse_.CppRustDocStringParser.get_logging_docstrings`

The two ``_param_.py`` / ``_parse_.py`` modules are loaded directly via
``importlib`` so the frontend package ``__init__`` (which pulls in numba,
psutil, pythreejs, etc.) is never imported. Only numpy-free stdlib code
runs here, which keeps the docs venv lean.

Run from anywhere::

    python docs/generate_frontend_params_reference.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
OUT_DIR = DOCS_DIR / "jupyterlab_api"
FRONTEND_DIR = REPO_ROOT / "frontend"
SRC_DIR = REPO_ROOT / "src"


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        print(
            f"generate_frontend_params_reference: cannot load {path}",
            file=sys.stderr,
        )
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _escape_rst(text: str) -> str:
    """Escape bare ``|`` so docutils does not read it as a substitution ref.

    Parameter descriptions contain expressions like ``|S-1|`` and
    ``|θ-θ_rest|`` which RST would otherwise parse as substitution
    references and fail the ``-W`` build.
    """
    return text.replace("|", r"\|")


def _render_param_table(name: str, rows: list[tuple[str, str]]) -> list[str]:
    out: list[str] = [f"{name}", "-" * len(name), "", ".. list-table::", ""]
    for label, value in rows:
        out.append(f"   * - {label}")
        out.append(f"     - {_escape_rst(value)}")
    out.append("")
    return out


def _render_global(param_mod: types.ModuleType) -> str:
    params = param_mod.app_param()
    title = "⚙️ Simulation Parameters"
    lines: list[str] = [
        title,
        "=" * (len(title) + 4),
        "",
        ".. This file is auto-generated from ``frontend/_param_.py``.",
        ".. Regenerate via: python docs/generate_frontend_params_reference.py",
        "",
        "Application-wide simulation parameters. Accessed via "
        ":meth:`frontend.App.get_default_param` and overridden per session "
        "with ``session.param.set(key, value)`` / ``session.param.get(key)`` "
        "(see :class:`frontend.ParamManager`). Keys use hyphens, not "
        "underscores.",
        "",
    ]
    for name, (value, short_desc, long_desc) in params.items():
        rows = [
            ("Key", f"``{name}``"),
            ("Default", f"``{value!r}``"),
            ("Label", short_desc),
        ]
        if long_desc:
            rows.append(("Description", long_desc))
        lines.extend(_render_param_table(name, rows))
    return "\n".join(lines).rstrip() + "\n"


def _render_objects(param_mod: types.ModuleType) -> str:
    obj_types = ("tri", "tet", "rod")
    all_params = {t: param_mod.object_param(t) for t in obj_types}

    all_names: list[str] = []
    seen: set[str] = set()
    for t in obj_types:
        for name in all_params[t]:
            if name not in seen:
                seen.add(name)
                all_names.append(name)

    title = "🧪 Material Parameters"
    lines: list[str] = [
        title,
        "=" * (len(title) + 4),
        "",
        ".. This file is auto-generated from ``frontend/_param_.py``.",
        ".. Regenerate via: python docs/generate_frontend_params_reference.py",
        "",
        "Per-object material parameters. Set via ``object.param.set(key, "
        "value)`` on a :class:`frontend.Object`. The solver exposes three "
        "element types, each with its own defaults:",
        "",
        "- ``tri``: triangle shells (cloth).",
        "- ``tet``: tetrahedral solids.",
        "- ``rod``: rod / edge elements.",
        "",
        "``(not applicable)`` in a default column means the given element "
        "type does not expose that parameter.",
        "",
    ]
    for name in all_names:
        rows: list[tuple[str, str]] = [("Key", f"``{name}``")]
        short_desc: str | None = None
        long_desc: str | None = None
        for t in obj_types:
            entry = all_params[t].get(name)
            if entry is None:
                rows.append((f"Default ({t})", "(not applicable)"))
                continue
            value, sd, ld = entry
            rows.append((f"Default ({t})", f"``{value!r}``"))
            if short_desc is None:
                short_desc = sd
                long_desc = ld
        if short_desc:
            rows.append(("Label", short_desc))
        if long_desc:
            rows.append(("Description", long_desc))
        lines.extend(_render_param_table(name, rows))
    return "\n".join(lines).rstrip() + "\n"


def _render_logs(parse_mod: types.ModuleType) -> str:
    if not SRC_DIR.is_dir():
        print(
            f"generate_frontend_params_reference: missing src/ at {SRC_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)
    entries = parse_mod.CppRustDocStringParser.get_logging_docstrings(
        str(SRC_DIR)
    )
    title = "📡 Log Channels"
    lines: list[str] = [
        title,
        "=" * (len(title) + 4),
        "",
        ".. This file is auto-generated from ``src/`` ``.cu`` / ``.rs`` "
        "sources.",
        ".. Regenerate via: python docs/generate_frontend_params_reference.py",
        "",
        "Named log streams emitted by the solver "
        "(``SimpleLog`` / ``logging.push`` / ``logging.mark`` in the "
        "C++/Rust sources). Pull a channel out of a finished session with "
        "``session.get.log.numbers(name)`` for ``(x, y)`` pairs or "
        "``session.get.log.number(name)`` for the latest scalar; "
        "``session.get.log.stdout()`` / ``stderr()`` return the raw text "
        "streams. See :class:`frontend.SessionGet`.",
        "",
    ]
    if not entries:
        lines.append("*(No logging docstrings found.)*")
        lines.append("")
    for name, doc in entries.items():
        rows: list[tuple[str, str]] = []
        for key, value in doc.items():
            if key == "filename" or not value:
                continue
            rows.append((key, value))
        lines.extend(_render_param_table(name, rows))
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    param_mod = _load_module("_param_doc", FRONTEND_DIR / "_param_.py")
    parse_mod = _load_module("_parse_doc", FRONTEND_DIR / "_parse_.py")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[Path, str]] = [
        (OUT_DIR / "simulation_parameters.rst", _render_global(param_mod)),
        (OUT_DIR / "material_parameters.rst", _render_objects(param_mod)),
        (OUT_DIR / "log_channels.rst", _render_logs(parse_mod)),
    ]
    for path, body in outputs:
        path.write_text(body)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
