# File: frontend/tests/_contract_frontend_api_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Public-API regression baseline for the `frontend` package.
#
# Locks the public Python surface of `frontend` so the PyO3 layer
# (ppf-cts-py) stays a drop-in replacement. The Rust migration replaces
# the implementation file by file, but notebook callers must keep
# importing `App`, `MeshManager`, etc. with the same signatures. This
# test fails loudly the moment a public symbol moves, gets renamed, or
# changes signature.
#
# How drift surfaces: each assertion compares the live module/class
# against a hard-coded snapshot. PRs that intentionally evolve the API
# update the snapshot in this file in the same diff, which makes the
# review explicit.

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Snapshot: every name in frontend.__all__, plus the two helpers that
# lack obvious __all__ membership but are documented entry points.
EXPECTED_PUBLIC_NAMES = {
    "App",
    "AssetFetcher",
    "AssetManager",
    "AssetUploader",
    "BlenderApp",
    "CppRustDocStringParser",
    "CreateManager",
    "Extra",
    "FixedScene",
    "FixedSession",
    "InvisibleAdder",
    "MeshManager",
    "Object",
    "ObjectAdder",
    "ParamDecoder",
    "ParamManager",
    "Plot",
    "PlotManager",
    "Rod",
    "Scene",
    "SceneDecoder",
    "SceneInfo",
    "SceneManager",
    "Session",
    "SessionExport",
    "SessionGet",
    "SessionInfo",
    "SessionManager",
    "SessionOutput",
    "Sphere",
    "TetMesh",
    "TriMesh",
    "Utils",
    "Wall",
    "get_cache_dir",
    "sdf",
}


# Snapshot: App's public class members. Statics + properties + methods.
# Format: {name: kind} where kind is one of "static", "classmethod",
# "property", "method". Drift in either the name set or the kind fails.
EXPECTED_APP_MEMBERS = {
    # Static factories + class-level helpers
    "create": "static",
    "load": "static",
    "get_proj_root": "static",
    "get_default_param": "static",
    "busy": "static",
    "is_fast_check": "static",
    "set_fast_check": "static",
    "terminate": "static",
    "recover": "static",
    "get_data_dirpath": "static",
    "run_tests": "static",
    # Instance properties
    "name": "property",
    "mesh": "property",
    "plot": "property",
    "scene": "property",
    "asset": "property",
    "extra": "property",
    "session": "property",
    "ci": "property",
    "cache_dir": "property",
    "ci_dir": "property",
    # Instance methods
    "clear": "method",
    "save": "method",
    "clear_cache": "method",
}


# Snapshot: signature of every static/method on App that takes
# arguments. Format: {name: "(arg1: T, arg2: T = ...) -> R"}. Computed
# via inspect.signature so default reprs etc. match the live code.
EXPECTED_APP_SIGNATURES = {
    "create": "(name: str, cache_dir: str = '') -> 'App'",
    "load": "(name: str, cache_dir: str = '') -> 'App'",
    "get_proj_root": "() -> str",
    "get_default_param": "() -> frontend._session_.ParamManager",
    "busy": "() -> bool",
    "is_fast_check": "() -> bool",
    "set_fast_check": "(enabled: bool = True)",
    "terminate": "()",
    "recover": "(name: str) -> frontend._session_.FixedSession",
    "get_data_dirpath": "()",
    "run_tests": "() -> bool",
    "clear": "(self) -> 'App'",
    "save": "(self) -> 'App'",
    "clear_cache": "(self) -> 'App'",
}


def _import_frontend():
    try:
        import frontend  # noqa: PLC0415
    except ImportError as e:
        pytest.skip(f"frontend not importable in this environment: {e}")
    return frontend


def test_public_names_match_snapshot():
    """Every name in frontend.__all__ is the snapshot, no more no less."""
    frontend = _import_frontend()
    declared = set(frontend.__all__)
    assert declared == EXPECTED_PUBLIC_NAMES, (
        f"frontend.__all__ drifted from snapshot.\n"
        f"  added: {sorted(declared - EXPECTED_PUBLIC_NAMES)}\n"
        f"  removed: {sorted(EXPECTED_PUBLIC_NAMES - declared)}"
    )


def test_public_names_actually_resolvable():
    """Every name in __all__ resolves to a real attribute of the package."""
    frontend = _import_frontend()
    missing = [name for name in frontend.__all__ if not hasattr(frontend, name)]
    assert not missing, f"names in __all__ that don't resolve: {missing}"


def _classify(cls, name):
    raw = inspect.getattr_static(cls, name)
    if isinstance(raw, staticmethod):
        return "static"
    if isinstance(raw, classmethod):
        return "classmethod"
    if isinstance(raw, property):
        return "property"
    if callable(raw):
        return "method"
    return "attribute"


def test_app_public_members_match_snapshot():
    frontend = _import_frontend()
    actual = {
        name: _classify(frontend.App, name)
        for name in vars(frontend.App)
        if not name.startswith("_")
    }
    assert actual == EXPECTED_APP_MEMBERS, (
        f"App public surface drifted.\n"
        f"  added: {sorted(set(actual) - set(EXPECTED_APP_MEMBERS))}\n"
        f"  removed: {sorted(set(EXPECTED_APP_MEMBERS) - set(actual))}\n"
        f"  kind changed: "
        f"{sorted((k, EXPECTED_APP_MEMBERS[k], actual[k]) for k in actual.keys() & EXPECTED_APP_MEMBERS.keys() if actual[k] != EXPECTED_APP_MEMBERS[k])}"
    )


def test_app_method_signatures_match_snapshot():
    """Catches arg renames/reorderings/default changes."""
    frontend = _import_frontend()
    drift = {}
    for name, expected_sig in EXPECTED_APP_SIGNATURES.items():
        member = inspect.getattr_static(frontend.App, name)
        # unwrap staticmethod for signature inspection on Python <3.10
        target = member.__func__ if isinstance(member, staticmethod) else member
        if isinstance(target, property):
            continue
        actual_sig = str(inspect.signature(target))
        if actual_sig != expected_sig:
            drift[name] = (expected_sig, actual_sig)
    assert not drift, (
        "App method signatures drifted from snapshot:\n"
        + "\n".join(f"  {n}:\n    expected {e}\n    actual   {a}" for n, (e, a) in drift.items())
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
