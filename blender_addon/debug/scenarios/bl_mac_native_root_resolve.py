# File: scenarios/bl_mac_native_root_resolve.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# macOS Native solver-root resolution. In macOS Native mode the user points
# the addon at the directory that holds ``target/release/ppf-cts-server``,
# which is where both a repo checkout and the distributable bundle keep it.
# Selecting a subdirectory of that root (``target``, ``target/release``, or
# the bundle's ``bin``) is an easy mistake to make, so
# ``core.connection.resolve_mac_native_root`` walks up parent directories to
# the real root. This scenario exercises that inside real Blender:
#
#   * core.connection.resolve_mac_native_root returns a valid root unchanged,
#     ascends from any subdirectory (including the binary file itself) to its
#     parent root, and returns None for an unrelated / blank path.
#   * a directory holding only ``bin/ppf-cts-server`` does NOT resolve: the
#     macOS layout ships the server under target/release alone, and bin/ holds
#     the Metal backend dylib and the shader libraries.
#   * ui.main_panel._draw_native_status draws a CHECKMARK "Solver path
#     valid" line for a subdir selection plus a second line naming the
#     resolved root, a CHECKMARK with no extra line for an exact root, and an
#     ERROR line for a directory with no solver under it (a fake layout
#     records the label() calls, since a real UILayout cannot be built outside
#     draw).
#   * selecting macOS Native with a subdirectory keeps the Connect button's
#     poll() reachable (ssh.run_command.poll()).
#
# The scenario builds its own fixture trees under a temp dir (empty marker
# files, never executed) and never connects to a server, so it runs on any
# host.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True
# Pure path-resolution logic (resolve_mac_native_root); backend-agnostic, and
# runs wherever the rig runs.
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


class _FakeLayout:
    """Records label() calls so _draw_native_status can be exercised
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
    conn = __import__(pkg + ".core.connection", fromlist=["resolve_mac_native_root"])
    main_panel = __import__(pkg + ".ui.main_panel", fromlist=["_draw_native_status"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    resolve = conn.resolve_mac_native_root

    # ---- fixture trees: a distributable bundle and a repo checkout ----
    #
    # THE SANDBOX MUST HAVE NO VALID SOLVER ROOT AMONG ITS ANCESTORS, and
    # `tempfile.mkdtemp()` alone no longer guarantees that. `resolve` walks UP
    # from what it is given, so if any ancestor of the sandbox is itself a root,
    # the three cases below that are supposed to resolve to NOTHING resolve to
    # that ancestor instead, and the scenario fails while the resolver is doing
    # exactly what it documents.
    #
    # It bites on a CI runner rather than on a workstation, which is why it was
    # invisible until it ran there. The rig gives each scenario its own TMPDIR
    # under the debug root, and on the Linux GPU leg that root sits inside the
    # transferred repository at `/home/ubuntu`, which holds
    # `target/release/ppf-cts-server` and so IS a valid root. `mkdtemp` honors
    # TMPDIR, so the sandbox landed underneath it.
    #
    # Pick the first base whose own ancestors are clean, and ASSERT it below
    # rather than trusting the choice: a scenario that silently tests nothing is
    # worse than one that fails.
    _base = None
    for _candidate in (tempfile.gettempdir(), "/tmp", os.path.expanduser("~")):
        if _candidate and os.path.isdir(_candidate) and resolve(_candidate) is None:
            _base = _candidate
            break
    tmp = tempfile.mkdtemp(prefix="mac_native_", dir=_base)
    record(
        "sandbox_has_no_root_ancestor",
        resolve(tmp) is None,
        {"tmp": tmp, "base": _base, "resolved_ancestor": resolve(tmp)},
    )
    bundle = os.path.join(tmp, "bundle")
    _touch(os.path.join(bundle, "target", "release", "ppf-cts-server"))
    _touch(os.path.join(bundle, "target", "release", "ppf-contact-solver"))
    _touch(os.path.join(bundle, "bin", "libppfbe_metal.dylib"))
    repo = os.path.join(tmp, "repo")
    _touch(os.path.join(repo, "target", "release", "ppf-cts-server"))
    empty = os.path.join(tmp, "unrelated")
    os.makedirs(empty, exist_ok=True)
    # A Windows-shaped layout: the server under bin/ alone. The macOS probe
    # must refuse it, because bin/ holds the backend dylib and the shader
    # libraries and a root shaped this way cannot launch.
    bin_only = os.path.join(tmp, "bin_only")
    _touch(os.path.join(bin_only, "bin", "ppf-cts-server"))

    def _same(a, b):
        return a is not None and os.path.normpath(a) == os.path.normpath(b)

    # ---- resolver: a valid root selects itself, a subdir walks up to it ----
    record("resolve_bundle_root_self", _same(resolve(bundle), bundle),
           {"got": resolve(bundle)})
    record("resolve_bundle_target_release_subdir",
           _same(resolve(os.path.join(bundle, "target", "release")), bundle),
           {"got": resolve(os.path.join(bundle, "target", "release"))})
    record("resolve_bundle_target_subdir",
           _same(resolve(os.path.join(bundle, "target")), bundle),
           {"got": resolve(os.path.join(bundle, "target"))})
    record("resolve_bundle_bin_subdir",
           _same(resolve(os.path.join(bundle, "bin")), bundle),
           {"got": resolve(os.path.join(bundle, "bin"))})
    record("resolve_repo_root_self", _same(resolve(repo), repo),
           {"got": resolve(repo)})
    record("resolve_binary_file_itself",
           _same(resolve(os.path.join(repo, "target", "release", "ppf-cts-server")), repo),
           {"got": resolve(os.path.join(repo, "target", "release", "ppf-cts-server"))})
    record("resolve_unrelated_dir_none", resolve(empty) is None, {"got": resolve(empty)})
    record("resolve_blank_none", resolve("   ") is None, {"got": resolve("   ")})
    record("resolve_bin_only_layout_none", resolve(bin_only) is None,
           {"got": resolve(bin_only)})

    # ---- panel status line: subdir validates and names the resolved root ----
    fl_sub = _FakeLayout()
    main_panel._draw_native_status(fl_sub, "MAC_NATIVE", os.path.join(bundle, "target", "release"))
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
    main_panel._draw_native_status(fl_root, "MAC_NATIVE", bundle)
    record(
        "panel_exact_root_valid_no_extra_line",
        [ic for _, ic in fl_root.labels] == ["CHECKMARK"]
        and any("Solver path valid" in t for t, _ in fl_root.labels)
        and not any("Using solver root" in t for t, _ in fl_root.labels),
        {"labels": fl_root.labels},
    )

    fl_bad = _FakeLayout()
    main_panel._draw_native_status(fl_bad, "MAC_NATIVE", empty)
    record(
        "panel_invalid_shows_error",
        [ic for _, ic in fl_bad.labels] == ["ERROR"]
        and any("not found" in t for t, _ in fl_bad.labels),
        {"labels": fl_bad.labels},
    )

    fl_blank = _FakeLayout()
    main_panel._draw_native_status(fl_blank, "MAC_NATIVE", "")
    record("panel_blank_silent", fl_blank.labels == [], {"labels": fl_blank.labels})

    # ---- select macOS Native with a subdir: Connect stays reachable ----
    root = groups.get_addon_data(bpy.context.scene)
    root.state.project_name = "mac_native_resolve"
    props = root.ssh_state
    props.server_type = "MAC_NATIVE"
    props.mac_native_path = os.path.join(bundle, "target", "release")
    record(
        "poll_mac_native_subdir_enabled",
        bool(bpy.ops.ssh.run_command.poll()) is True,
        {"path": props.mac_native_path},
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

    No substitutions are needed: the scenario creates its own fixture trees
    under a temp dir and only exercises the resolver, the panel status helper,
    and the Connect operator's poll. It never connects to a server or executes
    the marker files, so it runs on any host.
    """
    return _DRIVER_BODY


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
