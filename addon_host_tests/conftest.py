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


def _install_stub_mathutils() -> None:
    """Register a ``mathutils`` deep enough for the modules loaded here.

    `mathutils` is Blender's own, so it does not exist on this interpreter any
    more than `bpy` does, and the chain that reaches it is not obvious from a
    test's own imports: `core/backends.py` imports `models/console.py`, which
    imports `core/utils.py`, which imports `core/transform.py`, whose first
    line is `from mathutils import Matrix`. A suite that only wanted the ssh
    transport therefore fails at collection with a name none of its own code
    mentions.

    ONLY THE IMPORT HAS TO SUCCEED. Nothing loaded here computes a transform:
    the matrix helpers in `core/transform.py` are reached from the encoder,
    which needs real Blender objects and belongs in the rig. So `Matrix` is a
    placeholder that raises if anything actually tries to USE it, which keeps
    a future test from silently asserting against a fake linear algebra.
    """
    if "mathutils" in sys.modules:
        return
    mathutils = types.ModuleType("mathutils")

    class _Matrix:
        """Importable, and loud if exercised."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "mathutils.Matrix is a stub here: this tier runs on a plain "
                "interpreter and computes no transforms. A test that needs "
                "real matrix maths belongs in the Blender rig "
                "(blender_addon/debug/scenarios)."
            )

        @staticmethod
        def Identity(*args, **kwargs):  # noqa: N802 - Blender's own spelling
            raise RuntimeError("mathutils.Matrix.Identity is a stub here")

    mathutils.Matrix = _Matrix
    mathutils.Vector = _Matrix
    sys.modules["mathutils"] = mathutils


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

    # ``mathutils`` ships with Blender the same way ``bpy`` does, and
    # ``core/transform.py`` imports Matrix from it at module scope. Any module
    # that reaches ``core.utils`` therefore fails to IMPORT here unless a test
    # happened to stub ``core.transform`` first, which made a module's
    # importability depend on fixture order.
    #
    # The stub satisfies the import and nothing else: every attribute raises
    # when used, so a test that actually reaches this math fails loudly and
    # names the reason, rather than computing a plausible wrong answer from a
    # do-nothing placeholder.
    mathutils = types.ModuleType("mathutils")

    def _unavailable(name):
        class _Unavailable:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    f"mathutils.{name} is a stub outside Blender. A test that "
                    f"needs real {name} math belongs in the Blender rig "
                    "(blender_addon/debug/scenarios), not here."
                )

        _Unavailable.__name__ = name
        return _Unavailable

    for _name in ("Matrix", "Vector", "Euler", "Quaternion", "Color"):
        setattr(mathutils, _name, _unavailable(_name))

    sys.modules.update(
        {
            "bpy": bpy,
            "bpy.app": app,
            "bpy.app.handlers": handlers,
            "bpy.app.translations": translations,
            "mathutils": mathutils,
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
    _install_stub_mathutils()

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

    A name the caller did not list still IMPORTS, and raises when used. The
    placeholder lives in ``sys.modules`` for the whole session, so a later
    test importing a symbol this caller had no reason to name would otherwise
    fail at import with an error naming the placeholder rather than the
    missing behavior, and which symbols are needed then depends on the order
    the tests happened to run in.
    """
    fqname = f"blender_addon.{name}"
    if fqname in sys.modules:
        return
    mod = types.ModuleType(fqname)
    for key, value in attrs.items():
        setattr(mod, key, value)

    def __getattr__(attr: str):  # noqa: N807 - PEP 562 module-level hook
        if attr.startswith("__"):
            raise AttributeError(attr)

        def _unavailable(*args, **kwargs):
            raise RuntimeError(
                f"{fqname}.{attr} is a placeholder outside Blender. A test "
                f"that needs the real {name} belongs in the Blender rig "
                "(blender_addon/debug/scenarios), not here."
            )

        return _unavailable

    mod.__getattr__ = __getattr__
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
        # THE STUB IS SESSION-SCOPED AND LEAKS, so it has to satisfy every
        # importer this tier reaches, not only `core/utils.py`. Once it is
        # installed no later module gets the real `models/groups.py`, and
        # `models/console.py` imports `get_addon_data` from it: a suite that
        # loads `core/backends.py` (the ssh transport one does) then fails at
        # collection with "cannot import name get_addon_data ... (unknown
        # location)", which names neither this fixture nor the ordering that
        # produced it. It only appears when that suite RUNS, which needs
        # paramiko, so it is invisible locally and red on the POSIX CI legs.
        get_addon_data=lambda *a, **k: None,
    )
    _stub_submodule("core.transform", world_matrix=lambda *a, **k: None)
    return load_addon_module("core.utils")


@pytest.fixture(scope="session")
def ssh_config():
    """``blender_addon.core.ssh_config``, loaded from source.

    It reads text files and resolves host aliases, so it imports nothing
    from Blender and the stub in ``load_addon_module`` is never consulted.
    """
    _ensure_package("blender_addon.core", ADDON_ROOT / "core")
    return load_addon_module("core.ssh_config")


@pytest.fixture(scope="session")
def ssh_command(ssh_config):
    """``blender_addon.core.ssh_command``, loaded from source.

    Its ``from .ssh_config import split_host_spec`` resolves against the
    real sibling, which the fixture above has already registered.
    """
    return load_addon_module("core.ssh_command")


@pytest.fixture(scope="session")
def backends(ssh_config):
    """``blender_addon.core.backends``, loaded from source.

    Only the jump-chain helpers are reachable this way, and that is all
    this fixture is for. Everything else in the module needs paramiko or
    the docker package, which the add-on imports lazily inside
    ``create_backend`` through ``core.module``, so importing the module
    itself pulls in neither.
    """
    _ensure_package("blender_addon.core", ADDON_ROOT / "core")
    return load_addon_module("core.backends")


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
