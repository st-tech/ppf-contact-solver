# File: scenarios/bl_win_native_bundle_layout.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The Windows Native probe accepts the layout the published bundle really has,
# on Windows, with Windows paths.
#
# ``release.yml`` zips the CONTENTS of ``build-win-native\dist`` (it passes
# ``dist\*``), so the extracted root is the dist tree itself, with no wrapper
# directory. ``bundle.bat`` copies the two Rust executables into
# ``dist\target\release`` and puts only DLLs and ffmpeg in ``dist\bin``. So the
# binary a Windows artist has is at ``<extract root>\target\release\
# ppf-cts-server.exe``, and ``bin\`` alone never makes a root valid.
#
# This is a Windows-only scenario on purpose. The questions it settles are
# questions about Windows path handling: a drive letter, backslash separators,
# a space in the user profile directory (``C:\Users\John Smith``, which is
# ordinary and which the shell-safety test rejects), and Blender's ``//``
# notation resolving across those. ``os.path`` answers all of them differently
# on POSIX, so measuring them anywhere but Windows would prove nothing about
# the platform where the report came from.
#
# Checks:
#
#   * the dist layout the release really ships validates from its root.
#   * a ``bin\`` holding only DLLs does not validate, so the probe cannot be
#     satisfied by the half of the bundle that carries no server.
#   * a root whose path holds a space validates, since the Windows Native
#     launch passes the directory to ``subprocess.Popen`` as ``cwd`` with an
#     argv list and never builds a shell command from it.
#   * Blender's ``//``-relative form of a Windows path resolves to the same
#     root as its absolute spelling.
#   * the not-found message names the directory examined.
#
# The scenario builds fake trees under a temp dir (empty marker files, never
# executed) and never connects to a server.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True
PLATFORMS = ("win32",)
# Pure path logic; nothing here reaches a solver.
BACKENDS = ("real",)


_DRIVER_BODY = r'''
import os
import shutil
import tempfile
import traceback

result.setdefault("errors", [])
result.setdefault("checks", {})


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w"):
        pass


def _same(a, b):
    return bool(a) and os.path.normcase(os.path.normpath(a)) == \
        os.path.normcase(os.path.normpath(b))


tmp = None
try:
    conn = __import__(pkg + ".core.connection",
                      fromlist=["win_native_server_binary"])
    utils = __import__(pkg + ".core.utils", fromlist=["resolve_local_path"])
    probe = conn.win_native_server_binary
    resolve_root = conn.resolve_win_native_root
    resolve_local_path = utils.resolve_local_path

    tmp = tempfile.mkdtemp(prefix="ppf_win_bundle_")
    record("temp_dir_is_on_a_drive", len(os.path.splitdrive(tmp)[0]) == 2,
           {"tmp": tmp})

    # ---- the layout release.yml actually publishes ----
    # bundle.bat: TARGET_DIR = dist\target\release holds the two .exe files,
    # BIN_DIR = dist\bin holds the CUDA DLLs and ffmpeg.
    dist = os.path.join(tmp, "ppf-contact-solver-2026-01-01-00-00-win64")
    _touch(os.path.join(dist, "target", "release", "ppf-cts-server.exe"))
    _touch(os.path.join(dist, "target", "release", "ppf-contact-solver.exe"))
    _touch(os.path.join(dist, "target", "release", "_ppf_cts_py.dll"))
    _touch(os.path.join(dist, "bin", "libsimbackend_cuda.dll"))
    _touch(os.path.join(dist, "bin", "ffmpeg.exe"))
    _touch(os.path.join(dist, "python", "python.exe"))
    _touch(os.path.join(dist, "start.bat"))

    found = probe(dist)
    record("published_dist_layout_validates_from_its_root",
           _same(found, os.path.join(dist, "target", "release",
                                     "ppf-cts-server.exe")),
           {"got": found})
    record("published_dist_root_resolves_to_itself",
           _same(resolve_root(dist), dist), {"got": resolve_root(dist)})
    record("published_dist_target_release_walks_up",
           _same(resolve_root(os.path.join(dist, "target", "release")), dist),
           {"got": resolve_root(os.path.join(dist, "target", "release"))})

    # ---- bin\ alone carries no server ----
    dll_only = os.path.join(tmp, "dll-only")
    _touch(os.path.join(dll_only, "bin", "libsimbackend_cuda.dll"))
    _touch(os.path.join(dll_only, "bin", "ffmpeg.exe"))
    record("bin_without_server_does_not_validate", probe(dll_only) is None,
           {"got": probe(dll_only)})

    # ---- an ordinary Windows profile path holds a space ----
    spaced = os.path.join(tmp, "John Smith", "Downloads",
                          "ppf-contact-solver-win64")
    _touch(os.path.join(spaced, "target", "release", "ppf-cts-server.exe"))
    record("root_with_a_space_validates", probe(spaced) is not None,
           {"path": spaced, "got": probe(spaced)})
    record("root_with_a_space_resolves", _same(resolve_root(spaced), spaced),
           {"got": resolve_root(spaced)})

    # ---- Blender's relative notation over a Windows path ----
    project = os.path.join(tmp, "project")
    os.makedirs(project, exist_ok=True)
    blend_path = os.path.join(project, "scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    rel_dist = bpy.path.relpath(dist, start=project)
    record("relative_form_is_blender_notation", rel_dist.startswith("//"),
           {"rel": rel_dist})
    record("relative_windows_path_resolves_to_the_same_root",
           _same(resolve_local_path(rel_dist), dist),
           {"got": resolve_local_path(rel_dist), "want": dist})
    record("relative_windows_path_validates",
           probe(resolve_local_path(rel_dist)) is not None,
           {"got": probe(resolve_local_path(rel_dist))})

    # ---- the refusal names the directory that was examined ----
    empty = os.path.join(tmp, "not-a-bundle")
    os.makedirs(empty, exist_ok=True)
    message = conn.win_native_not_found_message(empty)
    record("not_found_message_names_the_directory", empty in message,
           {"message": message})

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
finally:
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
'''


def build_driver(ctx: r.ScenarioContext) -> str:
    """Return the Python source the bootstrap will exec inside Blender.

    No substitutions are needed: the scenario builds its own fake bundle trees
    and its own saved .blend under a temp dir, and only exercises the path
    probe, the root resolver and the message helper. It never executes the
    fake ``.exe`` marker files and never connects to a server.
    """
    return _DRIVER_BODY


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
