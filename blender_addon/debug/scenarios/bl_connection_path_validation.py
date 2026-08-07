# File: scenarios/bl_connection_path_validation.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Backend Connection path validation. A connection path that holds a space
# or a shell-unsafe character (& | ; ... ) breaks once it is interpolated
# into a launch/ssh command, so the panel warns and the Connect button is
# disabled. This scenario exercises that behavior inside real Blender:
#
#   * core.utils.find_invalid_path_char flags spaces / metacharacters and
#     passes clean paths (including Windows drive letters and ``~/``).
#   * ui.main_panel._draw_path_warning emits a single ERROR line for a bad
#     path and stays silent for a good one (a fake layout records the
#     label() calls, since a real UILayout can't be built outside draw).
#   * ssh.run_command.poll() (the Connect button's enable gate) returns
#     False for a bad path and True for a clean one, for the two backend
#     types whose poll has no external-module dependency: LOCAL and
#     WIN_NATIVE.
#   * the project name is held to a stricter rule (find_invalid_name_char):
#     no spaces, special characters, or path separators. The same warning
#     line and poll gate apply.
#   * core.utils.windows_path_too_long projects the deepest cache-file path
#     the build writes under a Windows solver root and flags it when it
#     reaches MAX_PATH (260), and ui.main_panel._draw_long_path_warning emits
#     the two-line warning for it. This turns the bare mid-Transfer
#     FileNotFoundError on an over-long Windows path into a warning shown when
#     the path is set. The warning is suppressed when long-path support is
#     enabled system-wide (core.utils.windows_long_paths_enabled), since the
#     limit no longer applies there.
#   * the projection covers the LONGEST cache filename the server composes,
#     not only the shortest. The addon cannot import the frontend, so
#     core/utils.py mirrors that filename; the mirror is checked here against
#     the Rust composer itself (datamodel/mesh.rs tetra_cache_name), so a
#     change to the cache key surfaces as a failing check rather than as a
#     Transfer that dies on a path the panel called acceptable.

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import _runner as r


NEEDS_BLENDER = True
# Connection-path validation UI behavior; backend-agnostic.
BACKENDS = ("emulated", "real")

# Directory segments the server puts between the solver root and the
# tetrahedralize cache file (``datamodel/app.rs`` ``compose_data_dir``). The
# branch is ``unknown`` for a packaged Windows build, which is the case the
# guard is for.
_SERVER_CACHE_DIRS = ("local", "share", "ppf-cts", "git-unknown")

# The widest per-object fTetWild override set the Blender encoder can emit,
# with float values rendered the way a float32 property reaches str(). Used
# to ask the composer for the longest filename a build can write.
_WIDEST_TETRA_KWARGS = [
    ("edge_length_fac", "0.05000000074505806"),
    ("epsilon", "0.0010000000474974513"),
    ("stop_energy", "10.0"),
    ("num_opt_iter", "80"),
    ("optimize", "True"),
    ("simplify", "True"),
    ("coarsen", "True"),
]


def _longest_tetra_cache_component() -> int:
    """Length of the longest tetrahedralize cache filename the server writes.

    Read from the Rust composer through the frontend, so this is the value
    the addon's mirror has to match. A tree whose cdylib is not built cannot
    answer the question, and this raises there rather than certifying the
    mirror against nothing.
    """
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from frontend import _rust  # type: ignore[attr-defined]

    hash64 = "f" * 64
    name, _key = _rust.mesh_tetra_cache_key(hash64, [], _WIDEST_TETRA_KWARGS)
    return len(os.path.basename(_rust.mesh_cache_path("/c", hash64, name)))


_DRIVER_BODY = r'''
import traceback

result.setdefault("errors", [])
result.setdefault("checks", {})


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


class _FakeLayout:
    """Records label() calls so _draw_path_warning can be exercised without a
    real Blender UILayout (which can't be instantiated outside a draw)."""

    def __init__(self):
        self.labels = []

    def label(self, text="", icon="", **kw):
        self.labels.append((text, icon))


try:
    utils = __import__(pkg + ".core.utils", fromlist=["find_invalid_path_char"])
    main_panel = __import__(pkg + ".ui.main_panel", fromlist=["_draw_path_warning"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    fic = utils.find_invalid_path_char

    bs = chr(92)  # backslash, kept out of the source to dodge escaping
    win_backslash = "C:" + bs + "ppf-contact-solver" + bs + "build"

    # ---- validator: clean paths pass, dangerous chars are caught ----
    record("validator_clean_unix", fic("/home/user/work") is None,
           {"v": fic("/home/user/work")})
    record("validator_clean_win_fwd", fic("C:/ppf-contact-solver/build") is None,
           {"v": fic("C:/ppf-contact-solver/build")})
    record("validator_clean_win_backslash", fic(win_backslash) is None,
           {"v": fic(win_backslash)})
    record("validator_tilde_ok", fic("~/work/project") is None,
           {"v": fic("~/work/project")})
    record("validator_space", fic("/home/user/my work") == " ",
           {"v": fic("/home/user/my work")})
    record("validator_ampersand", fic("/data&run") == "&",
           {"v": fic("/data&run")})

    # ---- draw helper: warning + hint on bad, silent on good ----
    fl_bad = _FakeLayout()
    bad_ret = main_panel._draw_path_warning(fl_bad, "/data&run")
    icons_bad = [ic for _, ic in fl_bad.labels]
    record(
        "warning_drawn_on_bad_path",
        bad_ret is True and len(fl_bad.labels) == 1 and icons_bad[0] == "ERROR",
        {"ret": bad_ret, "labels": fl_bad.labels},
    )

    fl_good = _FakeLayout()
    good_ret = main_panel._draw_path_warning(fl_good, "/home/user/work")
    record(
        "warning_silent_on_good_path",
        good_ret is False and len(fl_good.labels) == 0,
        {"ret": good_ret, "labels": fl_good.labels},
    )

    # ---- Connect button poll() gate (real operator) ----
    # LOCAL and WIN_NATIVE poll branches don't require paramiko/docker, so
    # the gate is exercised purely through path validity + project name.
    root = groups.get_addon_data(bpy.context.scene)
    root.state.project_name = "path_validation"
    props = root.ssh_state

    def connect_enabled():
        return bool(bpy.ops.ssh.run_command.poll())

    props.server_type = "LOCAL"
    props.local_path = "/home/user/work"
    record("poll_local_clean_enabled", connect_enabled() is True,
           {"path": props.local_path})
    props.local_path = "/home/user/my work"
    record("poll_local_space_disabled", connect_enabled() is False,
           {"path": props.local_path})
    props.local_path = "/data&run"
    record("poll_local_ampersand_disabled", connect_enabled() is False,
           {"path": props.local_path})
    props.local_path = ""
    record("poll_local_empty_enabled", connect_enabled() is True,
           {"path": "<empty>"})

    props.server_type = "WIN_NATIVE"
    props.win_native_path = "C:/ppf-contact-solver/build"
    record("poll_win_clean_enabled", connect_enabled() is True,
           {"path": props.win_native_path})
    props.win_native_path = "C:/Program Files/ppf"
    record("poll_win_space_disabled", connect_enabled() is False,
           {"path": props.win_native_path})

    # ---- project name validator (stricter: no path separators either) ----
    fin = utils.find_invalid_name_char
    record("name_clean", fin("drape_test-01.v2") is None, {"v": fin("drape_test-01.v2")})
    record("name_space", fin("my project") == " ", {"v": fin("my project")})
    record("name_ampersand", fin("proj&run") == "&", {"v": fin("proj&run")})
    record("name_slash_rejected", fin("a/b") == "/", {"v": fin("a/b")})

    # ---- project-name warning helper ----
    fl_name_bad = _FakeLayout()
    name_bad_ret = main_panel._draw_name_warning(fl_name_bad, "my project")
    record(
        "name_warning_drawn_on_bad",
        name_bad_ret is True and len(fl_name_bad.labels) == 1
        and fl_name_bad.labels[0][1] == "ERROR",
        {"ret": name_bad_ret, "labels": fl_name_bad.labels},
    )
    fl_name_good = _FakeLayout()
    name_good_ret = main_panel._draw_name_warning(fl_name_good, "clean_name")
    record(
        "name_warning_silent_on_good",
        name_good_ret is False and len(fl_name_good.labels) == 0,
        {"ret": name_good_ret, "labels": fl_name_good.labels},
    )

    # ---- Windows long-path projection + warning ----
    wptl = utils.windows_path_too_long
    pwcpl = utils.projected_windows_cache_path_len

    def expected_projection(root, project):
        """Longest path the server can write under *root* for *project*.

        Assembled from the composer's own filename length (substituted in
        by the scenario module) plus the server's directory layout, so the
        addon's mirror of that filename is checked, not restated.
        """
        dirs = bs.join(list(SERVER_CACHE_DIRS) + [project, ".cash", ""])
        return len(root) + 1 + len(dirs) + LONGEST_TETRA_CACHE_COMPONENT

    probe_root = "C:" + bs + "probe"
    record("projection_matches_the_server_composer",
           pwcpl(probe_root, "violet") == expected_projection(probe_root, "violet"),
           {"v": pwcpl(probe_root, "violet"),
            "expected": expected_projection(probe_root, "violet"),
            "component": LONGEST_TETRA_CACHE_COMPONENT})

    # The root from the original bug report: 64 characters, which overflows
    # the 260-character limit.
    long_root = ("C:" + bs + "Users" + bs + "alexa" + bs + "Desktop" + bs
                 + "ppf-contact-solver-2026-06-01-13-25-win64")
    record("longpath_flags_reported_case",
           wptl(long_root, "violet") == expected_projection(long_root, "violet"),
           {"v": wptl(long_root, "violet"),
            "expected": expected_projection(long_root, "violet")})
    record("longpath_silent_on_short_root", wptl("C:" + bs + "dev", "violet") is None,
           {"v": wptl("C:" + bs + "dev", "violet")})
    record("longpath_silent_on_empty", wptl("", "violet") is None,
           {"v": wptl("", "violet")})
    # A short root can still overflow via a long project name.
    record("longpath_flags_long_project_name",
           wptl("C:" + bs + "dev", "x" * 120) is not None,
           {"v": wptl("C:" + bs + "dev", "x" * 120)})
    # A 50-character root leaves too little for the longest cache path, so
    # it must warn even though the shortest cache path would still fit.
    root50 = "C:" + bs + "x" * 47
    record("longpath_flags_fifty_char_root",
           len(root50) == 50 and wptl(root50, "violet") is not None,
           {"v": wptl(root50, "violet"), "len": len(root50)})

    # windows_long_paths_enabled() gates the warning; force both states so the
    # draw checks are independent of the host's actual registry setting.
    record("longpaths_enabled_is_bool",
           isinstance(utils.windows_long_paths_enabled(), bool),
           {"v": utils.windows_long_paths_enabled()})

    _orig_lpe = main_panel.windows_long_paths_enabled
    try:
        main_panel.windows_long_paths_enabled = lambda: False
        fl_long = _FakeLayout()
        long_ret = main_panel._draw_long_path_warning(fl_long, long_root, "violet")
        record(
            "longpath_warning_drawn_when_disabled",
            long_ret is True and len(fl_long.labels) == 2
            and fl_long.labels[0][1] == "ERROR",
            {"ret": long_ret, "labels": fl_long.labels},
        )
        fl_short = _FakeLayout()
        short_ret = main_panel._draw_long_path_warning(fl_short, "C:" + bs + "dev", "violet")
        record(
            "longpath_warning_silent_on_short",
            short_ret is False and len(fl_short.labels) == 0,
            {"ret": short_ret, "labels": fl_short.labels},
        )
        # With long paths enabled the limit no longer applies: stay silent
        # even for the over-long root.
        main_panel.windows_long_paths_enabled = lambda: True
        fl_enabled = _FakeLayout()
        enabled_ret = main_panel._draw_long_path_warning(fl_enabled, long_root, "violet")
        record(
            "longpath_warning_silent_when_longpaths_enabled",
            enabled_ret is False and len(fl_enabled.labels) == 0,
            {"ret": enabled_ret, "labels": fl_enabled.labels},
        )
    finally:
        main_panel.windows_long_paths_enabled = _orig_lpe

    # ---- Connect button poll() gate on project name ----
    # Hold a known-clean LOCAL path so only the project name varies.
    props.server_type = "LOCAL"
    props.local_path = "/home/user/work"
    root.state.project_name = "clean_name"
    record("poll_name_clean_enabled", connect_enabled() is True,
           {"name": root.state.project_name})
    root.state.project_name = "bad name"
    record("poll_name_space_disabled", connect_enabled() is False,
           {"name": root.state.project_name})
    root.state.project_name = "proj&x"
    record("poll_name_special_disabled", connect_enabled() is False,
           {"name": root.state.project_name})

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
'''


def build_driver(ctx: r.ScenarioContext) -> str:
    """Return the Python source the bootstrap will exec inside Blender.

    The only substitutions are the server-side cache-name facts. They are
    resolved out here, in a process that can import the frontend, because
    Blender's Python holds the addon alone and the addon deliberately does
    not depend on the frontend.
    """
    preamble = (
        f"SERVER_CACHE_DIRS = {_SERVER_CACHE_DIRS!r}\n"
        f"LONGEST_TETRA_CACHE_COMPONENT = {_longest_tetra_cache_component()}\n"
    )
    return preamble + _DRIVER_BODY


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
