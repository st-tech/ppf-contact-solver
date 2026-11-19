# File: remote_shell.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# EVERY SHELL COMMAND THE ADD-ON SENDS TO A REMOTE SOLVER HOST, built here so
# that each path in it is one quoted shell word.
#
# A remote solver path is whatever the artist typed, and it is interpolated
# into commands that run under a POSIX shell on the solver host (over SSH, or
# inside a container). These used to paste the path in bare, so a space split
# it and a `$`, a `;` or a quote was interpreted by that shell, and the
# Connect button refused any such path rather than carry it. Built here, a path
# holding any character a POSIX filename allows reaches the command intact, and
# a scenario runs these exact commands against such a path
# (`debug/scenarios/bl_remote_paths_quoted.py`).

import shlex


def shell_word(value: str) -> str:
    """*value* as one single-quoted shell word, taken literally."""
    return "'" + value.replace("'", "'\\''") + "'"


def path_word(path: str) -> str:
    """*path* as one shell word, with a leading `~` still expanded.

    `~/ppf-contact-solver` is a reasonable remote path to type, and it worked
    while paths were pasted in bare, because the remote shell expanded the
    tilde. Quoted whole it would not expand, so the home directory is spelled
    `"$HOME"` in front of the quoted rest.
    """
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"' + shell_word(path[1:])
    return shell_word(path)


def in_directory(cwd: str, command: str) -> str:
    """*command* run from *cwd* on the solver host."""
    return f"cd {path_word(cwd)} && {command}"


def docker_exec(cwd: str, container: str, command: str) -> str:
    """*command* run from *cwd* inside *container*.

    `docker exec -w` takes the directory verbatim, with no shell in between
    to expand a tilde, so the directory is passed as a literal word.
    """
    return f"docker exec -w {shell_word(cwd)} {shlex.quote(container)} {command}"


def remove_file(path: str) -> str:
    return f"rm -f {path_word(path)}"


def read_file(path: str) -> str:
    return f"cat {path_word(path)} 2>/dev/null"


def tail_file(path: str, lines: int = 20, *, quiet: bool = True) -> str:
    redirect = " 2>/dev/null" if quiet else ""
    return f"tail -{int(lines)} {path_word(path)}{redirect}"


def list_frames(output_dir: str) -> str:
    """List the solver's `vert_*.bin` frames in *output_dir*.

    The glob stays outside the quotes, where the shell expands it.
    """
    return f"ls -1 {path_word(output_dir)}/vert_*.bin 2>/dev/null"


def server_launch_script(*, directory: str, server_binary: str,
                         server_args: list[str], target_dir: str | None,
                         env_prefix: str, activate_clause: str,
                         server_log: str) -> str:
    """The script that starts `ppf-cts-server` detached on the solver host.

    The server command runs under a second `bash -c`, so it is built as a
    string of shell words and then passed to that shell as ONE quoted word;
    every path inside it is quoted once for the inner shell and the whole is
    quoted once for the outer, so neither shell splits or expands a path.

    Detached without `nohup`, which refuses to run on macOS under a session
    with no terminal and starts nothing. What `nohup` is for is done by hand:
    the hangup signal is ignored (an ignored signal stays ignored across
    `exec`), standard input is /dev/null and the output goes to the log, in a
    background subshell the launching shell does not wait for.
    """
    inner = (
        activate_clause
        + env_prefix
        + " ".join([path_word(server_binary)] + [shlex.quote(a) for a in server_args])
    )
    target = (
        f"export CARGO_TARGET_DIR={path_word(target_dir)}\n" if target_dir else ""
    )
    return (
        "#!/bin/bash\n"
        f"cd {path_word(directory)} || exit 1\n"
        f"{target}"
        f'( trap "" HUP; exec bash -c {shell_word(inner)} ) '
        f"> {path_word(server_log)} 2>&1 < /dev/null &\n"
    )
