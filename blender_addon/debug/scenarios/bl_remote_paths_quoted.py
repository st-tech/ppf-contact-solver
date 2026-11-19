# File: scenarios/bl_remote_paths_quoted.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The commands the add-on sends to a REMOTE solver host carry a path the
# artist typed, and they used to paste it in bare: a space split it, and a
# `$`, a quote or a `;` was read by the remote shell. Connect refused any such
# path instead of carrying it. Every one of those commands is now built by
# `core.remote_shell`, and this scenario runs the exact strings, wrapped the
# way the SSH backend wraps them (`/bin/sh -c` inside `cd <dir> &&`), through
# a real POSIX shell on THIS machine against a directory named with the
# characters that used to break them. No remote host is needed: the commands
# are POSIX shell, and a shell is what runs them on the solver host too.
#
#   A_in_directory: the SSH backend's `cd <dir> && ...` lands in the directory.
#   B_file_commands: clear, read and tail a file under it.
#   C_list_frames: the frame listing finds the solver's `vert_*.bin` files,
#      with the glob still expanded outside the quotes.
#   D_tilde_still_expands: `~/<dir>` still means the home directory, as it did
#      when paths were pasted in bare.
#   E_server_launch_script: the detached server launch starts a stand-in
#      server in that directory with its exact arguments, CARGO_TARGET_DIR and
#      GPU selection, and writes its log there.
#   F_build_probe_lists_the_build: the build listing finds the server under it.
#
# POSIX only: Windows has no `/bin/sh` to run the commands with, and a remote
# solver host is never Windows.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)
PLATFORMS = ("linux", "darwin")


_DRIVER_BODY = r"""
import os
import shlex
import subprocess
import tempfile
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    dh = DriverHelpers(pkg, result)
    rs = __import__(pkg + ".core.remote_shell", fromlist=["path_word"])
    rb = __import__(pkg + ".core.remote_builds",
                    fromlist=["probe_command", "parse_listing"])
    gd = __import__(pkg + ".core.gpu_devices", fromlist=["shell_prefix"])

    home = tempfile.mkdtemp(prefix="ppf-remote-")
    name = "ppf (1) & O'Brien's $HOME `id` \"q\" ; x"
    root = os.path.join(home, name)
    os.makedirs(os.path.join(root, "session", "output"))

    def remote(command, *, shell=True, cwd=root):
        # What `SSHBackend.exec_command` sends for a command with no
        # container, run here by a POSIX shell with HOME pointed at `home`.
        if shell:
            command = f"/bin/sh -c {shlex.quote(command)}"
        full = rs.in_directory(cwd, command)
        done = subprocess.run(["/bin/sh", "-c", full], capture_output=True,
                              text=True, env={**os.environ, "HOME": home},
                              timeout=30)
        return done.returncode, done.stdout, done.stderr

    rc, out, err = remote("pwd", shell=False)
    dh.record("A_in_directory",
              rc == 0 and os.path.realpath(out.strip()) == os.path.realpath(root),
              {"rc": rc, "pwd": out.strip(), "stderr": err[-300:]})

    progress = os.path.join(root, "progress.log")
    with open(progress, "w") as f:
        f.write("".join(f"line {i}\n" for i in range(30)) + "SERVER_READY\n")
    rc_r, out_r, _ = remote(rs.read_file(progress))
    rc_t, out_t, _ = remote(rs.tail_file(progress, 3))
    rc_x, _, err_x = remote(rs.remove_file(progress))
    dh.record("B_file_commands",
              rc_r == 0 and out_r.endswith("SERVER_READY\n")
              and out_t.splitlines() == ["line 28", "line 29", "SERVER_READY"]
              and rc_x == 0 and not os.path.exists(progress),
              {"read_tail": out_r[-40:], "tail": out_t.splitlines(),
               "removed": not os.path.exists(progress), "stderr": err_x[-300:]})

    output = os.path.join(root, "session", "output")
    for i in (0, 1, 2):
        open(os.path.join(output, f"vert_{i}.bin"), "wb").close()
    rc, out, err = remote(rs.list_frames(output))
    names = sorted(line.strip().rsplit("/", 1)[-1] for line in out.splitlines())
    dh.record("C_list_frames",
              rc == 0 and names == ["vert_0.bin", "vert_1.bin", "vert_2.bin"],
              {"rc": rc, "names": names, "stderr": err[-300:]})

    rc, out, err = remote("pwd", shell=False, cwd="~/" + name)
    dh.record("D_tilde_still_expands",
              rc == 0 and os.path.realpath(out.strip()) == os.path.realpath(root),
              {"rc": rc, "pwd": out.strip(), "stderr": err[-300:]})

    # E: a stand-in server that records what it was started with.
    release = os.path.join(root, "target", "release")
    os.makedirs(release)
    record = os.path.join(home, "server-record.txt")
    server = os.path.join(release, "ppf-cts-server")
    with open(server, "w") as f:
        f.write("#!/bin/sh\n"
                f"{{ pwd; printf '%s\\n' \"$CARGO_TARGET_DIR\" "
                f"\"$CUDA_VISIBLE_DEVICES\"; for a in \"$@\"; do "
                f"printf '%s\\n' \"$a\"; done; }} > {shlex.quote(record)}\n"
                "echo started\n")
    os.chmod(server, 0o755)
    log = os.path.join(root, "server.log")
    script = rs.server_launch_script(
        directory=root, server_binary=server,
        server_args=["--port", "9090", "--idle-timeout", "0"],
        target_dir=os.path.join(root, "target"),
        env_prefix=gd.shell_prefix(1),
        activate_clause="", server_log=log)
    script_path = os.path.join(home, "start_server.sh")
    # Written and run the way the add-on does it on the solver host.
    rc1, _, err1 = remote(f"cat <<'EOF' > {shlex.quote(script_path)}\n{script}EOF\n")
    rc2, _, err2 = remote(f"chmod +x {shlex.quote(script_path)}")
    rc3, _, err3 = remote(shlex.quote(script_path))
    deadline = time.time() + 10.0
    while time.time() < deadline and not os.path.exists(record):
        time.sleep(0.1)
    got = open(record).read().splitlines() if os.path.exists(record) else []
    expected = [os.path.realpath(root), os.path.join(root, "target"), "1",
                "--port", "9090", "--idle-timeout", "0"]
    got_norm = [os.path.realpath(got[0])] + got[1:] if got else []
    dh.record("E_server_launch_script",
              rc1 == rc2 == rc3 == 0 and got_norm == expected
              and os.path.isfile(log),
              {"rc": [rc1, rc2, rc3], "got": got, "expected": expected,
               "log_exists": os.path.isfile(log),
               "stderr": (err1 + err2 + err3)[-400:], "script": script})

    rc, out, err = remote(rb.probe_command(root))
    listing = rb.parse_listing(out.splitlines())
    dh.record("F_build_probe_lists_the_build",
              rc == 0 and release in listing,
              {"rc": rc, "listing": listing, "stderr": err[-300:]})
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 120.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
