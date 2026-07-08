# File: scenarios/bl_win_native_root_resolve.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Windows Native solver-root resolution. In Windows Native mode the user
# points the addon at the directory that holds ``ppf-cts-server.exe`` (a
# distributable bundle ships it under ``bin/``, a dev/main repo checkout under
# ``target/release/``). Community users routinely select a *subdirectory* of
# the real root instead (``target/release``, ``bin``, or the embedded
# ``python`` folder) and used to hit a confusing "ppf-cts-server.exe not
# found" error. ``core.connection.resolve_win_native_root`` now walks up parent
# directories to the real root so those selections just work. This scenario
# exercises that inside real Blender:
#
#   * core.connection.resolve_win_native_root returns a valid root unchanged,
#     ascends from any subdirectory (including the binary file itself) to its
#     parent root, and returns None for an unrelated / blank path.
#   * ui.main_panel._draw_win_native_status draws a CHECKMARK "Solver path
#     valid" line for a subdir selection plus a second line naming the resolved
#     root, a CHECKMARK with no extra line for an exact root, and an ERROR line
#     for a directory with no solver under it (a fake layout records the
#     label() calls, since a real UILayout can't be built outside draw).
#   * selecting Windows Native with a subdirectory keeps the Connect button's
#     poll() reachable (ssh.run_command.poll()).
#
# The scenario builds its own fake solver trees under a temp dir (empty marker
# files, never executed) and never connects to a server, so it runs on any
# host, including macOS.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r'''
import os
import shutil
import tempfile
import traceback

result.setdefault("errors", [])
result.setdefault("checks", {})


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


class _FakeLayout:
    """Records label() calls so _draw_win_native_status can be exercised
    without a real Blender UILayout (which can't be built outside draw)."""

    def __init__(self):
        self.labels = []

    def label(self, text="", icon="", **kw):
        self.labels.append((text, icon))


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w"):
        pass


tmp = None
try:
    conn = __import__(pkg + ".core.connection", fromlist=["resolve_win_native_root"])
    main_panel = __import__(pkg + ".ui.main_panel", fromlist=["_draw_win_native_status"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    resolve = conn.resolve_win_native_root

    # ---- fake solver trees: a distributable bundle and a repo checkout ----
    tmp = tempfile.mkdtemp(prefix="ppf_win_native_")
    bundle = os.path.join(tmp, "bundle")
    _touch(os.path.join(bundle, "bin", "ppf-cts-server.exe"))
    _touch(os.path.join(bundle, "python", "python.exe"))
    repo = os.path.join(tmp, "repo")
    _touch(os.path.join(repo, "target", "release", "ppf-cts-server.exe"))
    _touch(os.path.join(repo, "build-win-native", "python", "python.exe"))
    empty = os.path.join(tmp, "unrelated")
    os.makedirs(empty, exist_ok=True)

    def _same(a, b):
        return a is not None and os.path.normpath(a) == os.path.normpath(b)

    # ---- resolver: a valid root selects itself, a subdir walks up to it ----
    record("resolve_bundle_root_self", _same(resolve(bundle), bundle),
           {"got": resolve(bundle)})
    record("resolve_bundle_bin_subdir",
           _same(resolve(os.path.join(bundle, "bin")), bundle),
           {"got": resolve(os.path.join(bundle, "bin"))})
    record("resolve_bundle_python_subdir",
           _same(resolve(os.path.join(bundle, "python")), bundle),
           {"got": resolve(os.path.join(bundle, "python"))})
    record("resolve_repo_root_self", _same(resolve(repo), repo),
           {"got": resolve(repo)})
    record("resolve_repo_target_release_subdir",
           _same(resolve(os.path.join(repo, "target", "release")), repo),
           {"got": resolve(os.path.join(repo, "target", "release"))})
    record("resolve_repo_target_subdir",
           _same(resolve(os.path.join(repo, "target")), repo),
           {"got": resolve(os.path.join(repo, "target"))})
    record("resolve_repo_buildwinnative_python_subdir",
           _same(resolve(os.path.join(repo, "build-win-native", "python")), repo),
           {"got": resolve(os.path.join(repo, "build-win-native", "python"))})
    record("resolve_binary_file_itself",
           _same(resolve(os.path.join(repo, "target", "release", "ppf-cts-server.exe")), repo),
           {"got": resolve(os.path.join(repo, "target", "release", "ppf-cts-server.exe"))})
    record("resolve_unrelated_dir_none", resolve(empty) is None, {"got": resolve(empty)})
    record("resolve_blank_none", resolve("   ") is None, {"got": resolve("   ")})

    # ---- panel status line: subdir validates and names the resolved root ----
    fl_sub = _FakeLayout()
    main_panel._draw_win_native_status(fl_sub, os.path.join(bundle, "bin"))
    icons_sub = [ic for _, ic in fl_sub.labels]
    texts_sub = [tx for tx, _ in fl_sub.labels]
    record(
        "panel_subdir_shows_valid_and_root",
        "CHECKMARK" in icons_sub
        and "ERROR" not in icons_sub
        and any("Solver path valid" in t for t in texts_sub)
        and any("Using solver root" in t for t in texts_sub),
        {"labels": fl_sub.labels},
    )

    fl_root = _FakeLayout()
    main_panel._draw_win_native_status(fl_root, bundle)
    record(
        "panel_exact_root_valid_no_extra_line",
        [ic for _, ic in fl_root.labels] == ["CHECKMARK"]
        and any("Solver path valid" in t for t, _ in fl_root.labels)
        and not any("Using solver root" in t for t, _ in fl_root.labels),
        {"labels": fl_root.labels},
    )

    fl_bad = _FakeLayout()
    main_panel._draw_win_native_status(fl_bad, empty)
    record(
        "panel_invalid_shows_error",
        [ic for _, ic in fl_bad.labels] == ["ERROR"]
        and any("not found" in t for t, _ in fl_bad.labels),
        {"labels": fl_bad.labels},
    )

    fl_blank = _FakeLayout()
    main_panel._draw_win_native_status(fl_blank, "")
    record("panel_blank_silent", fl_blank.labels == [], {"labels": fl_blank.labels})

    # ---- select Windows Native with a subdir: Connect stays reachable ----
    root = groups.get_addon_data(bpy.context.scene)
    root.state.project_name = "win_native_resolve"
    props = root.ssh_state
    props.server_type = "WIN_NATIVE"
    props.win_native_path = os.path.join(bundle, "bin")
    record(
        "poll_win_native_subdir_enabled",
        bool(bpy.ops.ssh.run_command.poll()) is True,
        {"path": props.win_native_path},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
finally:
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
'''


def build_driver(ctx: r.ScenarioContext) -> str:
    """Return the Python source the bootstrap will exec inside Blender.

    No substitutions are needed: the scenario creates its own fake solver
    trees under a temp dir and only exercises the resolver, the panel status
    helper, and the Connect operator's poll. It never connects to a server or
    executes the fake ``.exe`` marker files, so it runs on any host, macOS
    included.
    """
    return _DRIVER_BODY


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
