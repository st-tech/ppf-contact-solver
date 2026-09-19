# File: scenarios/bl_connection_path_relative.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Client-side connection paths survive Blender's relative-path notation.
#
# ``local_path`` and ``win_native_path`` are ``DIR_PATH`` properties, so the
# directory picker writes them in whatever form
# ``Preferences > File Paths > Relative Paths`` asks for. That preference ships
# ENABLED, so once the .blend is saved the picker stores ``//``-prefixed paths
# relative to the .blend. ``//`` is Blender's own notation: ``os.path`` reads it
# as a plain relative name, so probing it finds no solver and the panel reports
# a folder the user just picked as one holding no ``ppf-cts-server``. That is
# the community report "Connection failed: ppf-cts-server.exe not found" from a
# user whose bundle was exactly where they said it was.
#
# This scenario exercises the resolution inside real Blender, with a real saved
# .blend so ``//`` means what it means in the field:
#
#   * core.utils.resolve_local_path expands the ``//`` form, passes an absolute
#     path through unchanged, and leaves a blank path blank.
#   * ui.main_panel._draw_native_status draws CHECKMARK for a bundle named in
#     the ``//`` form (a fake layout records the label() calls, since a real
#     UILayout cannot be built outside draw).
#   * ui.main_panel._draw_long_path_warning measures the resolved path, not the
#     short ``//`` spelling that would hide an over-long root.
#   * REMOTE_OT_Connect.get_remote_path returns the SSH / Docker paths verbatim,
#     since those name a directory on the solver host where the client's .blend
#     location means nothing, while a native path is resolved to an absolute
#     directory at its own call site.
#   * the not-found message names the directory examined and tells a user with
#     no toolchain what to do, rather than only prescribing a cargo build.
#
# The scenario builds fake solver trees under a temp dir (empty marker files,
# never executed) and never connects to a server, so it runs on any host.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True
# Pure path-resolution logic; nothing here reaches a solver.
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
    """Records label() calls so the panel helpers can be exercised without a
    real Blender UILayout (which cannot be built outside draw)."""

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
    utils = __import__(pkg + ".core.utils", fromlist=["resolve_local_path"])
    conn = __import__(pkg + ".core.connection",
                      fromlist=["win_native_not_found_message"])
    main_panel = __import__(pkg + ".ui.main_panel",
                            fromlist=["_draw_native_status"])
    conn_ops = __import__(pkg + ".ui.connection_ops",
                          fromlist=["REMOTE_OT_Connect"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    resolve_local_path = utils.resolve_local_path

    tmp = tempfile.mkdtemp(prefix="ppf_relpath_")
    # The .blend and the bundle are SIBLINGS, so the relative form has to
    # climb out of the project directory. A bundle inside the project dir
    # would pass even with a resolver that only strips the "//" prefix.
    project = os.path.join(tmp, "project")
    os.makedirs(project, exist_ok=True)
    bundle = os.path.join(tmp, "ppf-contact-solver-win64")
    _touch(os.path.join(bundle, "target", "release", "ppf-cts-server.exe"))
    _touch(os.path.join(bundle, "python", "python.exe"))

    blend_path = os.path.join(project, "scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    record("blend_saved", bool(bpy.data.filepath), {"filepath": bpy.data.filepath})

    # ---- the relative form Blender's picker actually writes ----
    rel_bundle = bpy.path.relpath(bundle, start=project)
    record("relative_form_is_blender_notation", rel_bundle.startswith("//"),
           {"rel": rel_bundle})

    def _same(a, b):
        return bool(a) and os.path.normpath(a) == os.path.normpath(b)

    record("resolve_expands_relative", _same(resolve_local_path(rel_bundle), bundle),
           {"got": resolve_local_path(rel_bundle), "want": bundle})
    record("resolve_passes_absolute_through",
           _same(resolve_local_path(bundle), bundle),
           {"got": resolve_local_path(bundle)})
    record("resolve_keeps_blank_blank", resolve_local_path("") == "",
           {"got": repr(resolve_local_path(""))})
    record("resolve_keeps_whitespace_only_unresolved",
           resolve_local_path("   ").strip() == "",
           {"got": repr(resolve_local_path("   "))})

    # ---- the panel validates the folder the user picked ----
    fl_rel = _FakeLayout()
    main_panel._draw_native_status(fl_rel, "WIN_NATIVE", rel_bundle)
    record(
        "panel_relative_path_validates",
        any(ic == "CHECKMARK" for _, ic in fl_rel.labels)
        and not any(ic == "ERROR" for _, ic in fl_rel.labels),
        {"labels": fl_rel.labels},
    )

    fl_abs = _FakeLayout()
    main_panel._draw_native_status(fl_abs, "WIN_NATIVE", bundle)
    record(
        "panel_absolute_path_still_validates",
        any(ic == "CHECKMARK" for _, ic in fl_abs.labels),
        {"labels": fl_abs.labels},
    )

    # A relative spelling of a directory with no solver still reports the
    # error: resolving a path must not turn a wrong folder into a right one.
    empty = os.path.join(tmp, "unrelated")
    os.makedirs(empty, exist_ok=True)
    rel_empty = bpy.path.relpath(empty, start=project)
    fl_empty = _FakeLayout()
    main_panel._draw_native_status(fl_empty, "WIN_NATIVE", rel_empty)
    record(
        "panel_relative_path_without_solver_still_errors",
        any(ic == "ERROR" for _, ic in fl_empty.labels),
        {"labels": fl_empty.labels},
    )

    # ---- the long-path projection measures the resolved path ----
    # The "//" spelling is short by construction, so a projection that reads it
    # verbatim would call an over-long Windows root acceptable.
    #
    # windows_path_too_long is a pure measurement of a string, so the deep
    # root is NAMED rather than created: Windows refuses to create a
    # directory this deep without system-wide long-path support, which is
    # precisely the condition the warning exists for, so building the tree
    # would fail on the platform the check is about.
    deep_name = "d" * 90
    deep = os.path.join(tmp, deep_name, deep_name, deep_name)
    rel_deep = "//" + os.path.relpath(deep, project).replace(os.sep, "/")
    windows_path_too_long = utils.windows_path_too_long
    resolved_deep = resolve_local_path(rel_deep)
    record(
        "relative_deep_form_is_shorter_than_what_it_names",
        len(rel_deep) < len(deep),
        {"rel_len": len(rel_deep), "abs_len": len(deep)},
    )
    record(
        "long_path_projection_uses_resolved_path",
        windows_path_too_long(resolved_deep, "proj")
        == windows_path_too_long(deep, "proj"),
        {
            "resolved": windows_path_too_long(resolved_deep, "proj"),
            "absolute": windows_path_too_long(deep, "proj"),
            "resolved_path_len": len(resolved_deep),
        },
    )

    # ---- the operator hands the backend an absolute directory ----
    root = groups.get_addon_data(bpy.context.scene)
    root.state.project_name = "relpath_scenario"
    props = root.ssh_state
    op = conn_ops.REMOTE_OT_Connect

    # A NATIVE path is resolved where it is used, not by get_remote_path: the
    # picker writes it in the ``//`` form, and the connect arm expands it with
    # resolve_local_path before handing it to the facade.
    props.server_type = "LINUX_NATIVE"
    props.linux_native_path = rel_bundle
    got_native = utils.resolve_local_path(props.linux_native_path)
    record("native_path_reaches_backend_absolute", _same(got_native, bundle),
           {"got": got_native, "want": bundle})

    # A REMOTE path names a directory on the solver host, so it is passed
    # through untouched: expanding it against the client's .blend would point
    # the server at a directory that exists only on the artist's machine.
    props.server_type = "CUSTOM"
    props.ssh_remote_path = "/home/ubuntu/ppf-contact-solver"
    record("ssh_remote_path_passes_through",
           op.get_remote_path(op, props) == "/home/ubuntu/ppf-contact-solver",
           {"got": op.get_remote_path(op, props)})

    props.server_type = "DOCKER"
    props.docker_path = "/root/ppf-contact-solver"
    record("docker_path_passes_through",
           op.get_remote_path(op, props) == "/root/ppf-contact-solver",
           {"got": op.get_remote_path(op, props)})

    # ---- a Windows path with a space is not a shell problem ----
    # The Windows Native root is an os.path.join base and a Popen cwd; it is
    # never interpolated into a shell command, so the space rule that a
    # REMOTE path needs would only cost the user an unpressable button.
    spaced = os.path.join(tmp, "First Last", "ppf-contact-solver-win64")
    _touch(os.path.join(spaced, "target", "release", "ppf-cts-server.exe"))
    record("space_is_not_shell_unsafe",
           utils.find_shell_unsafe_path_char(spaced) is None
           and utils.find_invalid_path_char(spaced) == " ",
           {"shell_unsafe": utils.find_shell_unsafe_path_char(spaced),
            "invalid": utils.find_invalid_path_char(spaced)})
    record("metacharacter_still_refused_on_both",
           utils.find_shell_unsafe_path_char("/root/a;rm -rf /") == ";",
           {"got": utils.find_shell_unsafe_path_char("/root/a;rm -rf /")})

    props.server_type = "WIN_NATIVE"
    props.win_native_path = spaced
    record("win_native_connect_enabled_with_a_space",
           bool(bpy.ops.ssh.run_command.poll()) is True,
           {"path": spaced})

    fl_spaced = _FakeLayout()
    main_panel._draw_path_warning(fl_spaced, spaced, shell_bound=False)
    record("panel_does_not_warn_about_the_space", fl_spaced.labels == [],
           {"labels": fl_spaced.labels})

    # ---- Connect refuses a root with no solver ----
    # The panel draws "not found" for this directory, so connecting anyway
    # would put Connected and that line on screen in the same frame.
    #
    # The rig runs with PPF_WIN_NATIVE_NO_SPAWN set, which is the CI contract
    # that an external orchestrator owns the server and the binary need not
    # be under this root. Both halves are checked: cleared, connect refuses;
    # set, connect permits.
    saved_no_spawn = os.environ.pop("PPF_WIN_NATIVE_NO_SPAWN", None)
    try:
        refused = None
        try:
            conn.connect_win_native(empty, 9090)
        except FileNotFoundError as exc:
            refused = exc
        record("connect_refuses_a_root_without_a_solver",
               refused is not None and empty in str(refused),
               {"error": str(refused) if refused else None})
    finally:
        if saved_no_spawn is not None:
            os.environ["PPF_WIN_NATIVE_NO_SPAWN"] = saved_no_spawn

    if saved_no_spawn is not None:
        permitted = None
        try:
            info, _p = conn.connect_win_native(empty, 9090)
            permitted = info.current_directory
        except Exception as exc:
            permitted = f"raised {type(exc).__name__}: {exc}"
        record("no_spawn_mode_still_permits_a_bare_root",
               _same(permitted, empty), {"got": permitted})

    accepted = None
    try:
        info, _proc = conn.connect_win_native(bundle, 9090)
        accepted = info.current_directory
    except Exception as exc:
        accepted = f"raised {type(exc).__name__}: {exc}"
    record("connect_accepts_a_real_bundle_root", _same(accepted, bundle),
           {"got": accepted})

    # ---- the refusal a Windows artist reads ----
    message = conn.win_native_not_found_message(empty)
    record("not_found_message_names_the_directory", empty in message,
           {"message": message})
    record("not_found_message_names_the_layout",
           "target" in message and "ppf-cts-server.exe" in message,
           {"message": message})
    record(
        "not_found_message_is_not_only_a_build_instruction",
        "extracted" in message.lower() or "bundle" in message.lower(),
        {"message": message},
    )

    # A refusal IS the whole report, so the panel wraps it instead of clipping
    # it: the part a single label drops is the part that says what to do.
    fl_err = _FakeLayout()
    main_panel._draw_error_lines(fl_err, message)
    record("panel_wraps_a_long_error_instead_of_clipping",
           len(fl_err.labels) > 1
           and "".join(t for t, _ in fl_err.labels).replace(" ", "")
               == message.replace(" ", ""),
           {"rows": len(fl_err.labels), "msg_len": len(message)})
    record("panel_error_block_carries_one_icon",
           [ic for _, ic in fl_err.labels].count("ERROR") == 1,
           {"icons": [ic for _, ic in fl_err.labels]})

    fl_short = _FakeLayout()
    main_panel._draw_error_lines(fl_short, "Connection lost.")
    record("panel_short_error_stays_one_row",
           fl_short.labels == [("Connection lost.", "ERROR")],
           {"labels": fl_short.labels})

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
    trees and its own saved .blend under a temp dir, and only exercises the
    path resolver, the panel helpers and the Connect operator's path
    accessor. It never connects to a server or executes the fake ``.exe``
    marker files, so it runs on any host.
    """
    return _DRIVER_BODY


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
