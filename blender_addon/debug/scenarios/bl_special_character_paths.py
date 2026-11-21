# File: scenarios/bl_special_character_paths.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# A solver path may hold anything its filesystem allows. The Connect button
# used to refuse any path with a shell metacharacter, on the stated grounds
# that such characters are illegal in a Windows filename anyway, and most of
# them are not: `C:\Program Files (x86)\...`, a second download unpacked as
# `ppf-contact-solver (1)`, or a user folder `O'Brien` greyed the button out.
# The refusal was hiding a real defect behind it, the session launcher the
# server writes for every solve: on Windows its unquoted `set` lines split at
# `&` and its `if (...)` block closed early on the `)` of `(x86)`, and the
# server ran it through `cmd /C`, whose quote stripping split the path again.
#
#   A_connect_accepts_legal_paths: Connect's gate (`connect_refusal`) has no
#      objection to any of those paths, on each native type and on the remote
#      directory fields, which every remote command now quotes. The one
#      exception is ';' in a Windows Native root, below.
#   B_connect_refuses_unusable_paths: what no filename can hold is still
#      refused, by name: a control character anywhere, and on the Windows
#      Native root the characters NTFS forbids. A Windows Native root is also
#      refused for ';': the released distribution's Python cannot start from
#      a folder whose path holds one.
#   C_solve_under_special_folder: a real solve whose session lives under a
#      folder named with `(`, `)`, `&`, `'`, `%`, `^`, `#`, `!`, `$` and `;`
#      (`PROJECT_DIR_NAME`, the server's data root here). The server writes
#      the session launcher into that folder and starts the solver through it,
#      so this is where the old launcher broke; the frames have to arrive.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)

# Legal on Windows, macOS and Linux alike, and short, because a Windows rig
# root plus a session's cache file names must stay under MAX_PATH.
PROJECT_DIR_NAME = "p (1)&O'B 100%^#!$x;"


_DRIVER_BODY = r"""
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_DIR_NAME = <<PROJECT_DIR_NAME>>
FRAME_COUNT = 8

try:
    dh = DriverHelpers(pkg, result)
    conn = __import__(pkg + ".ui.connection_ops", fromlist=["connect_refusal"])
    root = dh.groups.get_addon_data(bpy.context.scene)
    props = root.ssh_state
    root.state.project_name = "special_paths"

    def refusal(kind, field, value):
        props.server_type = kind
        setattr(props, field, value)
        return conn.connect_refusal(bpy.context)

    native = [("WIN_NATIVE", "win_native_path"),
              ("LINUX_NATIVE", "linux_native_path"),
              ("MAC_NATIVE", "mac_native_path")]
    legal = [
        r"C:\Users\me\Downloads\ppf-contact-solver (1)",
        r"C:\Program Files (x86)\ppf-contact-solver",
        r"C:\Users\O'Brien\ppf-contact-solver",
        r"D:\Sim & FX\100% done\[v2] {final} ^ #! $x ;",
        "/home/me/My Work/ppf-contact-solver (1)",
    ]
    # A, natives: no refusal at all is expected, except that a Windows Native
    # root may not hold ';' (B checks that refusal).
    native_refusals = {}
    for kind, field in native:
        for path in legal:
            if kind == "WIN_NATIVE" and ";" in path:
                continue
            why = refusal(kind, field, path)
            if why is not None:
                native_refusals[f"{kind} {path}"] = why
    # A, remote directories: other requirements (a host, paramiko) may still
    # refuse, so only a refusal that names the path field counts.
    remote = [("CUSTOM", "ssh_remote_path"), ("DOCKER", "docker_path")]
    remote_legal = ["/home/me/My Work/ppf (1) & O'Brien's $x ;", "~/my ppf"]
    remote_refusals = {}
    for kind, field in remote:
        for path in remote_legal:
            why = refusal(kind, field, path) or ""
            label = props.bl_rna.properties[field].name
            if label in why:
                remote_refusals[f"{kind} {path}"] = why
    dh.record("A_connect_accepts_legal_paths",
              not native_refusals and not remote_refusals,
              {"native_refusals": native_refusals,
               "remote_refusals": remote_refusals})

    # B: still refused, naming the character.
    unusable = [
        ("WIN_NATIVE", "win_native_path", "C:\\a|b", "|"),
        ("WIN_NATIVE", "win_native_path", "C:\\a?b", "?"),
        ("WIN_NATIVE", "win_native_path", 'C:\\a"b', '"'),
        ("WIN_NATIVE", "win_native_path", r"D:\Sim & FX\100% done\[v2] {final} ^ #! $x ;", ";"),
        ("LINUX_NATIVE", "linux_native_path", "/a\nb", "\n"),
        ("CUSTOM", "ssh_remote_path", "/a\tb", "\t"),
    ]
    # The SSH requirements are checked in order, so give the type a host:
    # otherwise "Host is empty" answers before the path is looked at.
    props.host = "solver.invalid"
    missed = {}
    for kind, field, path, ch in unusable:
        why = refusal(kind, field, path) or ""
        if repr(ch) not in why:
            missed[f"{kind} {path!r}"] = why
    dh.record("B_connect_refuses_unusable_paths", not missed, {"missed": missed})

    # C: a real solve whose session sits under the awkward folder.
    plane = dh.reset_scene_to_pinned_plane(name="SpecialPathPlane")
    root = dh.configure_state(project_name="special_paths",
                              frame_count=FRAME_COUNT,
                              gravity=(0.0, 0.0, -9.8))
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.2, 0.0, 0.0), frame_start=1, frame_end=FRAME_COUNT,
                transition="LINEAR")
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, "special-paths:build")
    dh.run_and_wait(timeout=180.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=60.0)
    applied, total = dh.fetch_and_drain()
    session_root = str(dh.facade.engine.state.remote_root)
    session = os.path.join(session_root, "session")
    launchers = [n for n in ("command.sh", "command.bat")
                 if os.path.isfile(os.path.join(session, n))]
    out = os.path.join(session, "output")
    frames = sorted(n for n in (os.listdir(out) if os.path.isdir(out) else [])
                    if n.startswith("vert_") and n.endswith(".bin"))
    pc2 = dh.find_pc2_for(plane)
    arr = dh.read_pc2(pc2) if pc2 and os.path.isfile(pc2) else None
    dh.record("C_solve_under_special_folder",
              PROJECT_DIR_NAME in session_root and bool(launchers)
              and len(frames) >= FRAME_COUNT - 1 and arr is not None
              and bool(np.all(np.isfinite(arr))),
              {"session_root": session_root, "launchers": launchers,
               "frames_on_disk": len(frames), "applied": applied,
               "total": total,
               "pc2_samples": None if arr is None else int(arr.shape[0]),
               "solver": dh.facade.engine.state.solver.name})
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<PROJECT_DIR_NAME>>", repr(PROJECT_DIR_NAME))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
