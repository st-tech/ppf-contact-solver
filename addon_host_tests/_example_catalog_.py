# File: addon_host_tests/_example_catalog_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The Examples panel's catalog (`blender_addon/core/example_catalog.py`)
# reads every script in `examples/blender/` the panel can offer.
#
# The panel builds one field per `build()` keyword from the script's signature
# and docstring, read with `ast`. A new example whose `build()` the catalog
# cannot express (a parameter with no default, a default it has no field for)
# would show in the panel as an error rather than as a scene; this catches it
# here, without Blender. The Blender-side half, populating each example, is
# the `bl_examples_populate` rig scenario.

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples" / "blender"

_spec = importlib.util.spec_from_file_location(
    "ppf_example_catalog", REPO_ROOT / "blender_addon" / "core" / "example_catalog.py")
catalog = importlib.util.module_from_spec(_spec)
# Registered first: `dataclass` resolves the module's annotations through it.
sys.modules[_spec.name] = catalog
_spec.loader.exec_module(catalog)


def _opted_out(path: Path) -> bool:
    return not catalog._offered(ast.parse(path.read_text(encoding="utf-8")))


def test_every_example_script_is_listed_and_populatable():
    examples = catalog.read_catalog(str(EXAMPLES))
    scripts = sorted(p.stem for p in EXAMPLES.glob("*.py") if not _opted_out(p))
    assert sorted(e.key for e in examples) == scripts
    for e in examples:
        assert not e.error, f"{e.key}: {e.error}"
        assert e.params, f"{e.key}: build() has no parameter the panel can show"
        assert not e.skipped, f"{e.key}: no field for {e.skipped}"
        # The title is the first docstring line, not the file-name fallback.
        assert e.title != e.key.replace("-", " ").replace("_", " ").title(), (
            f"{e.key}: build() has no docstring title")
        assert e.description, f"{e.key}: build() docstring has no description"


def test_character_anim_is_not_offered():
    # Its OBJ sequence does not ship with the add-on.
    assert _opted_out(EXAMPLES / "character-anim.py")
    assert "character-anim" not in {e.key for e in catalog.read_catalog(str(EXAMPLES))}


def test_defaults_are_typed_for_their_fields():
    for e in catalog.read_catalog(str(EXAMPLES)):
        for p in e.params:
            if p.default is None:
                assert p.optional, f"{e.key}.{p.name}"
                continue
            expected = {
                catalog.INT: int, catalog.FLOAT: float, catalog.BOOL: bool,
                catalog.STR: str, catalog.VEC3: tuple,
            }[p.kind]
            assert type(p.default) is expected, f"{e.key}.{p.name}: {p.default!r}"
            if p.kind == catalog.VEC3:
                assert len(p.default) == 3 and all(type(v) is float for v in p.default)


def test_signature_kinds(tmp_path):
    # The annotation forms the scripts use, and the ones the catalog refuses.
    (tmp_path / "probe.py").write_text(
        "def build(a: int = 1, b: float = 2, c: bool = True, d: str = 'x',\n"
        "          e: tuple = (0, 1.5, 2), f: str | None = None,\n"
        "          g: 'Optional[int]' = None, h=3.0, i: list = [1], *, j: int = 4):\n"
        "    '''Probe.\n\n    A probe script.\n    '''\n"
    )
    (tmp_path / "nodefault.py").write_text("def build(a, b=1):\n    pass\n")
    (tmp_path / "nobuild.py").write_text("x = 1\n")
    (tmp_path / "broken.py").write_text("def build(:\n")
    (tmp_path / "optout.py").write_text("PANEL = False\n\ndef build(a=1):\n    pass\n")
    found = {e.key: e for e in catalog.read_catalog(str(tmp_path))}
    probe = found["probe"]
    assert probe.title == "Probe" and probe.description == "A probe script."
    kinds = {p.name: (p.kind, p.default, p.optional) for p in probe.params}
    assert kinds["a"] == (catalog.INT, 1, False)
    assert kinds["b"] == (catalog.FLOAT, 2.0, False)
    assert kinds["c"] == (catalog.BOOL, True, False)
    assert kinds["d"] == (catalog.STR, "x", False)
    assert kinds["e"] == (catalog.VEC3, (0.0, 1.5, 2.0), False)
    assert kinds["f"] == (catalog.STR, None, True)
    assert kinds["h"] == (catalog.FLOAT, 3.0, False)
    assert kinds["j"] == (catalog.INT, 4, False)
    # A string annotation names nothing the catalog reads, and a list has no field.
    assert set(probe.skipped) == {"g", "i"}
    assert "no default" in found["nodefault"].error
    assert "no build()" in found["nobuild"].error
    assert found["broken"].error.startswith("SyntaxError")
    assert "optout" not in found
    assert os.path.basename(found["probe"].path) == "probe.py"
