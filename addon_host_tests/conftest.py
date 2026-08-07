# File: addon_host_tests/conftest.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Loader for add-on modules that are pure logic but sit behind a top-level
# ``import bpy``.
#
# Blender's Python is the only place ``bpy`` exists, so a plain
# ``import blender_addon.core.<mod>`` fails on any other interpreter. The
# modules covered here compute file paths, lengths and byte layouts and
# touch no Blender data, so a stub that satisfies the import machinery is
# enough to exercise them; anything that reads real scene state belongs in
# the Blender rig (``blender_addon/debug/scenarios``) instead.
#
# WHY THIS DIRECTORY IS NOT UNDER blender_addon/: pytest imports the
# ``__init__.py`` of every package on the path from the rootdir to a test
# file, and ``blender_addon/__init__.py`` imports bpy on its first line.
# A conftest nested inside that tree is loaded too late to stub anything,
# so the collection fails before any test runs. Kept at the top level,
# nothing on the path is a package and the stub below is installed first.
# ``blender_addon/tests`` stays where it is and is run by Blender.
#
# Precedent for the synthetic-parent-package technique:
# ``crates/ppf-cts-formats/tests/scripts/gen_fixtures.py``.

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ADDON_ROOT = REPO_ROOT / "blender_addon"


def _install_stub_bpy() -> None:
    """Register a ``bpy`` package deep enough for the loaded modules.

    Every level has to be a real entry in ``sys.modules``, not an attribute
    on a namespace object: ``core/pc2.py`` reaches its translation helpers
    with ``from bpy.app.translations import ...``, which the import system
    resolves through ``sys.modules`` and cannot satisfy from an attribute.
    A partial stub therefore raises ``ImportError``, which that module
    catches and answers by setting ``bpy = None``, and it then applies
    ``@bpy.app.handlers.persistent`` at module scope regardless and fails
    with an ``AttributeError`` on ``None``.

    The three targets are the persistent-handler decorator (identity) and
    the two translation helpers (identity outside Blender).
    """
    if "bpy" in sys.modules:
        return
    bpy = types.ModuleType("bpy")
    app = types.ModuleType("bpy.app")
    handlers = types.ModuleType("bpy.app.handlers")
    translations = types.ModuleType("bpy.app.translations")

    handlers.persistent = lambda fn: fn
    translations.pgettext_iface = lambda text, *a, **k: text
    translations.pgettext_tip = lambda text, *a, **k: text
    app.handlers = handlers
    app.translations = translations
    bpy.app = app
    # A path-less data block: ``get_pc2_dir`` reads ``bpy.data.filepath``
    # and falls back to a temp dir when it is empty. Tests redirect that
    # directory explicitly rather than relying on the fallback.
    bpy.data = types.SimpleNamespace(filepath="")
    bpy.types = types.SimpleNamespace()

    sys.modules.update(
        {
            "bpy": bpy,
            "bpy.app": app,
            "bpy.app.handlers": handlers,
            "bpy.app.translations": translations,
        }
    )


def _ensure_package(name: str, path: Path) -> None:
    """Register a package entry for *name* rooted at *path*, if absent.

    The entry carries a real ``__path__`` so a loaded module's relative
    imports resolve against the files on disk, while the package's own
    ``__init__.py`` is never executed. That is what keeps the add-on's
    bpy-dependent init chain out of the picture.
    """
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod


def load_addon_module(dotted: str):
    """Import ``blender_addon.<dotted>`` from source with ``bpy`` stubbed.

    Returns the loaded module. Repeated calls return the same object, so a
    module's constants stay identical across tests in one session.
    """
    fqname = f"blender_addon.{dotted}"
    if fqname in sys.modules:
        return sys.modules[fqname]

    _install_stub_bpy()

    _ensure_package("blender_addon", ADDON_ROOT)
    parts = dotted.split(".")
    for depth in range(1, len(parts)):
        sub = ".".join(parts[:depth])
        _ensure_package(f"blender_addon.{sub}", ADDON_ROOT.joinpath(*parts[:depth]))

    source = ADDON_ROOT.joinpath(*parts).with_suffix(".py")
    spec = importlib.util.spec_from_file_location(fqname, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[fqname] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[fqname]
        raise
    return module


def _stub_submodule(name: str, **attrs) -> None:
    """Register ``blender_addon.<name>`` as a placeholder carrying *attrs*.

    Used where a loaded module needs a symbol from a sibling whose own
    import chain reaches Blender. The placeholder is registered before the
    module under test runs, so the real file is never executed.
    """
    fqname = f"blender_addon.{name}"
    if fqname in sys.modules:
        return
    mod = types.ModuleType(fqname)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[fqname] = mod


@pytest.fixture(scope="session")
def statistics_cache():
    """``blender_addon.core.statistics_cache``, loaded from source.

    Its ``_cbor2()`` helper reaches ``core.module.get_cbor2``, whose own
    module pulls in the add-on's console and from there Blender's RNA. The
    helper only has to hand back the cbor2 package, so it is supplied
    directly.
    """
    _ensure_package("blender_addon.core", ADDON_ROOT / "core")
    _stub_submodule("core.module", get_cbor2=lambda: __import__("cbor2"))
    return load_addon_module("core.statistics_cache")


@pytest.fixture(scope="session")
def addon_utils():
    """``blender_addon.core.utils``, loaded from source.

    Its module-level ``from ..models.groups import ...`` and
    ``from .transform import world_matrix`` reach Blender's RNA and
    ``mathutils``. The functions under test here touch neither, so those
    two imports are satisfied with placeholders rather than executed.
    """
    _ensure_package("blender_addon.models", ADDON_ROOT / "models")
    _stub_submodule(
        "models.groups",
        decode_vertex_group_identifier=lambda *a, **k: None,
        iterate_active_object_groups=lambda *a, **k: iter(()),
    )
    _stub_submodule("core.transform", world_matrix=lambda *a, **k: None)
    return load_addon_module("core.utils")


@pytest.fixture(scope="session")
def cdylib():
    """The tree-local ``_ppf_cts_py``, reached the way the frontend reaches it."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from frontend import _rust
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"_ppf_cts_py cdylib unavailable: {e}")
    return _rust
