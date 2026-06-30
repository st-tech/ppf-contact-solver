# File: __init__.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Entry point for the frontend package.

Import :class:`App` from this package to start the application; the other
names listed in ``__all__`` expose the scene, session, mesh, plotting,
asset, and utility APIs.
"""

# Resolve the Rust extension module exactly once, at the top of the
# package. Submodules import it as ``from . import _rust`` so a missing
# build surfaces this single, actionable error instead of a cascade of
# ``ModuleNotFoundError`` from whichever submodule loaded first.
#
# Tree-local cdylib load. ``_ppf_cts_py`` is built by ``cargo build
# --release`` (it is a workspace default-member) into
# ``<tree-root>/target/<profile>/`` as ``lib_ppf_cts_py.so`` (Linux),
# ``lib_ppf_cts_py.dylib`` (macOS) or ``_ppf_cts_py.dll`` (Windows). We
# load that file DIRECTLY by absolute path via importlib and register it
# as ``sys.modules["_ppf_cts_py"]``, so two worktrees (e.g. one on `dev`,
# one on `main`) on the same host always pick up their own build with no
# shared venv / site-packages in the picture. There is no maturin wheel
# and no fallback: a missing cdylib is a hard, actionable error. The tree
# root is taken relative to *this file*, so the right tree's build wins
# regardless of caller cwd.
import importlib.machinery as _machinery
import importlib.util as _ilu
import os as _os
import sys as _sys

_FRONTEND_TREE_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))


def _cdylib_filename():
    """The cargo cdylib filename for ``_ppf_cts_py`` on this platform."""
    if _sys.platform == "darwin":
        return "lib_ppf_cts_py.dylib"
    if _sys.platform == "win32":
        return "_ppf_cts_py.dll"
    return "lib_ppf_cts_py.so"


def _find_cdylib():
    """Locate the tree-local cdylib, preferring a release build over debug."""
    name = _cdylib_filename()
    for profile in ("release", "debug"):
        cand = _os.path.join(_FRONTEND_TREE_ROOT, "target", profile, name)
        if _os.path.isfile(cand):
            return cand
    return None


class _RustCdylibFinder:
    """A ``sys.meta_path`` finder that maps ``_ppf_cts_py`` to the tree-local
    cdylib. ``sys.modules`` registration below already satisfies a plain
    ``import _ppf_cts_py``; this keeps the name resolvable if the module is
    ever evicted from ``sys.modules`` (e.g. an addon reload)."""

    def __init__(self, path):
        self._path = path

    def find_spec(self, name, path=None, target=None):
        if name != "_ppf_cts_py":
            return None
        loader = _machinery.ExtensionFileLoader(name, self._path)
        return _ilu.spec_from_file_location(name, self._path, loader=loader)


def _load_rust():
    """Load and register the ``_ppf_cts_py`` cdylib built into this tree."""
    if "_ppf_cts_py" in _sys.modules:
        return _sys.modules["_ppf_cts_py"]
    path = _find_cdylib()
    if path is None:
        name = _cdylib_filename()
        raise ImportError(
            f"_ppf_cts_py extension not found (looked for {name} in "
            f"{_os.path.join(_FRONTEND_TREE_ROOT, 'target', 'release')} and "
            f"{_os.path.join(_FRONTEND_TREE_ROOT, 'target', 'debug')}). "
            "Build it with `cargo build --release` from the repo root (or "
            "`cargo build-emul` on macOS / any host without nvcc). There is "
            "no fallback to an installed wheel."
        )
    if not any(isinstance(f, _RustCdylibFinder) for f in _sys.meta_path):
        _sys.meta_path.insert(0, _RustCdylibFinder(path))
    loader = _machinery.ExtensionFileLoader("_ppf_cts_py", path)
    spec = _ilu.spec_from_file_location("_ppf_cts_py", path, loader=loader)
    module = _ilu.module_from_spec(spec)
    loader.exec_module(module)
    _sys.modules["_ppf_cts_py"] = module
    return module


# Load the tree-local cdylib (built by `cargo build --release`) and register
# it as the top-level `_ppf_cts_py` module. A missing cdylib raises a clear
# ImportError from _load_rust; there is no fallback to an installed wheel.
#
# No cross-tree guard is needed: _load_rust always loads the cdylib from THIS
# frontend's own tree (``<_FRONTEND_TREE_ROOT>/target/<profile>/``), so it can
# never pick up another tree's module the way a shared-venv wheel once could.
# A guard comparing the cdylib's compile-time ``__build_manifest_dir__`` to the
# tree root would also wrongly reject a distribution bundle, where bundle.bat
# copies a dll built in the source tree into ``dist/target/release/``.
_rust = _load_rust()  # noqa: N816

__all__ = [
    "App",
    "get_cache_dir",
    "AssetManager",
    "AssetFetcher",
    "AssetUploader",
    "SceneManager",
    "Scene",
    "SceneInfo",
    "ObjectAdder",
    "FixedScene",
    "Object",
    "InvisibleAdder",
    "Wall",
    "Sphere",
    "Extra",
    "MeshManager",
    "CreateManager",
    "Rod",
    "TetMesh",
    "TriMesh",
    "PlotManager",
    "Plot",
    "SessionManager",
    "Session",
    "FixedSession",
    "SessionInfo",
    "SessionExport",
    "SessionOutput",
    "SessionGet",
    "CppRustDocStringParser",
    "ParamManager",
    "Utils",
    "BlenderApp",
    "sdf",
]

from . import _sdf_ as sdf
from ._app_ import App
from ._asset_ import AssetFetcher, AssetManager, AssetUploader
from ._decoder_ import BlenderApp
from ._extra_ import Extra
from ._mesh_ import CreateManager, MeshManager, Rod, TetMesh, TriMesh
from ._parse_ import CppRustDocStringParser
from ._plot_ import Plot, PlotManager
from ._scene_ import (
    FixedScene,
    InvisibleAdder,
    Object,
    ObjectAdder,
    Scene,
    SceneInfo,
    SceneManager,
    Sphere,
    Wall,
)

# Backward-compat: FixedScene / EnumColor / ValidationError moved from the
# former ``_scene_fixed_`` module into ``_scene_``. Pickles written before
# that merge (``fixed_session.pickle``, ``app_state.pickle``) store the old
# ``frontend._scene_fixed_`` class path, so alias the old module name to
# ``_scene_`` and let those pickles keep unpickling.
_sys.modules.setdefault(f"{__name__}._scene_fixed_", _sys.modules[f"{__name__}._scene_"])
from ._session_ import (
    FixedSession,
    ParamManager,
    Session,
    SessionExport,
    SessionGet,
    SessionInfo,
    SessionManager,
    SessionOutput,
)
from ._utils_ import Utils, get_cache_dir
