# File: remote_builds.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Which solver builds a REMOTE solver host holds, asked once per connection.
#
# WHY THIS EXISTS. Choosing GPU or CPU is choosing which build DIRECTORY the
# server comes from: the solver links one backend, and
# `crates/ppf-cts-solver/build.rs` refuses to put a second one in a directory
# that already holds another. A native connection answers "which directories are
# there" with `os.path`, because the builds are on this machine. An SSH or
# Docker connection cannot: the builds are on the solver host, and the only way
# to reach it is a command over the backend.
#
# ONE PROBE, CACHED, ON THE WORKER THREAD. The panel redraws many times a
# second and a backend command is a blocking round trip (SSH wraps it in
# `timeout --signal=KILL`), so a probe from `draw` is a hang rather than a
# slowdown. This module is shaped exactly like `gpu_devices`, which solves the
# same problem for the GPU list: the probe runs once when a connection is
# established and again on an explicit Refresh, the answer is cached here, and
# the panel only ever reads the cache.
#
# WHAT IS ASKED. One `sh` command per connection, listing the directories under
# the root that hold a server and what each one's `.ppf-backend` marker says.
# The marker is the only evidence of what a directory HOLDS, since every backend
# links the same executable name; `connection._Reported` resolves that listing
# with the same rule the native resolvers apply to a filesystem.

from __future__ import annotations

import posixpath

# How long the listing command may take on the solver host. The GPU probe is
# bounded at 5 s for the same reason and says why: a wedged host must not stall
# Connect. This one is a handful of `test -f` and `cat` calls, so it is bounded
# tighter than the work it does could ever need.
PROBE_TIMEOUT_SECONDS = 10.0

# The directories a remote root can hold a server in, relative to it. The same
# rows `connection._LINUX_DEVICE_SUBDIRS` and `connection._GPU_BACKEND_SUBDIRS`
# carry, spelled here as the POSIX paths the probe command interpolates.
#
# LISTED RATHER THAN DISCOVERED, so the command is fixed text with the root as
# its only variable. A `find` over the root would be an unbounded walk of
# someone's home directory, and a glob would have to be parsed back into
# directories anyway.
_CANDIDATE_SUBDIRS = (
    "target/release",
    "target/cuda/release",
    "target/rocm/release",
    "target/metal/release",
    "target/cpu/release",
)

_SERVER_NAME = "ppf-cts-server"
_MARKER_NAME = ".ppf-backend"

# What separates the two fields of the probe's output. A tab cannot occur in
# either: the directories are this module's own literals, and a marker is a
# backend name `build.rs` wrote.
#
# THE MARKER COMES FIRST AND THE DIRECTORY LAST, which is not cosmetic. Every
# backend's `exec_command` returns `stdout.decode().strip().splitlines()`, so
# the whole output is stripped before this module ever sees it. A directory
# holding a server and NO marker prints an empty second field, and with the
# directory first that line ENDS in the separator: the strip takes it, the line
# no longer parses, and the directory vanishes from the listing. It is the last
# line of the output that loses it, so which directory disappears depends on
# what else the host has, and an unmarked build is exactly what a distribution
# from before markers looks like. With the marker first, the line always ends
# in a directory, which is never empty.
_FIELD_SEP = "\t"


def normalize_root(root: str) -> str:
    """*root* in the one spelling every reader of the listing uses.

    THE LISTING IS KEYED BY ABSOLUTE DIRECTORY, so the root the probe was built
    from and the root a later lookup joins onto have to agree character for
    character. A remote path is passed to the backend exactly as the artist
    typed it, trailing slash and all, which a native path is not: the natives
    normalize theirs at connect. So `/srv/ppf/` and `/srv/ppf` name one
    directory to the solver host and two different keys here, and the panel
    would report a host with no builds while the launch, joining onto the same
    unnormalized string, found them.

    A root that is nothing but separators keeps one, since that is the
    filesystem root rather than an empty path.
    """
    trimmed = (root or "").rstrip("/")
    if not trimmed:
        return "/" if root else ""
    return trimmed


def probe_command(root: str) -> str:
    """The shell command that lists *root*'s builds on the solver host.

    Prints one `<marker><TAB><absolute directory>` line per directory that holds
    a server, with the marker empty where there is none. A directory with no
    server prints nothing, which is the answer "there is no build here". The
    field order is load-bearing; `_FIELD_SEP` says why.

    THE ROOT IS INTERPOLATED INTO A SHELL COMMAND, so it is quoted here. Every
    path that reaches this point has already been through the add-on's
    metacharacter gate (`core.utils.find_invalid_path_char`, which the Connect
    button applies to a remote path), and this quoting is the second of the two
    rather than the only one.

    `2>/dev/null` on the marker read, not on the whole command: a directory
    holding a server and no marker is the normal shape of a downloaded
    distribution, and its `cat` failing is the expected case rather than an
    error worth seeing.
    """
    root = normalize_root(root)
    parts = []
    for subdir in _CANDIDATE_SUBDIRS:
        # JOINED WITH `posixpath` AND QUOTED WHOLE, never concatenated around
        # the quotes. The listing is keyed by the directory, and every lookup
        # joins the root to a subdirectory with `posixpath.join`, so the probe
        # has to print exactly what that produces. Concatenating a quoted root
        # onto "/<subdir>" gives `//target/release` for a root of `/`, which no
        # lookup computes, and the host is then reported as holding no solver.
        directory = posixpath.join(root, subdir)
        tested = _tested(posixpath.join(directory, _SERVER_NAME))
        marker = _tested(posixpath.join(directory, _MARKER_NAME))
        parts.append(
            f'if [ -f {tested} ]; then '
            f'printf "%s\\t%s\\n" '
            f'"$(cat {marker} 2>/dev/null)" '
            f'{_shell_quote(directory)}; fi'
        )
    return "; ".join(parts)


def _tested(path: str) -> str:
    """*path* as a shell word the test can read, with a leading `~` expanded.

    A remote path is whatever the artist typed, and `~/ppf-contact-solver` is a
    reasonable thing to type: the SSH backend runs its commands through a shell
    that would expand it. Quoted, a tilde does not expand, and
    `[ -f '~/x/...' ]` is false on a host that has the build.

    ONLY THE TEST USES THIS SPELLING. The printed directory keeps the tilde,
    because the listing is keyed by it and every later lookup joins onto the
    root the panel and the launch hold, which is still the tilde spelling.
    Printing the expanded path would key the listing by something no lookup
    computes.
    """
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"' + _shell_quote(path[1:])
    return _shell_quote(path)


def _shell_quote(value: str) -> str:
    """*value* as one single-quoted shell word."""
    return "'" + value.replace("'", "'\\''") + "'"


def parse_listing(lines) -> dict[str, str]:
    """The probe's output as ``{absolute directory: marker}``.

    Each line is `<marker><TAB><directory>`; `_FIELD_SEP` says why that way
    round. A marker is stripped of surrounding whitespace, which is what a
    marker written by `echo` on Windows carries (`b"cpu\\r\\n"` was measured)
    and what `$(...)` leaves of a file with a trailing newline.

    A line without the separator is IGNORED rather than raising. The command
    runs under the user's own login shell on a machine this add-on does not
    control, so a banner or an `mesg` warning printed ahead of the output is a
    thing that happens; refusing the whole listing over one would report a host
    with no builds, which reads as a wrong path.
    """
    found: dict[str, str] = {}
    for line in lines or ():
        # RSTRIPPED OF LINE ENDINGS ONLY, never of whitespace in general. A
        # directory holding a server and NO marker prints its separator with
        # nothing after it, which is the normal shape of a distribution built
        # before markers; a general rstrip takes that separator with the
        # newline, the line no longer parses, and the directory vanishes from
        # the listing entirely. The marker's own value is stripped below.
        text = str(line).rstrip("\r\n")
        if _FIELD_SEP not in text:
            continue
        marker, _, directory = text.partition(_FIELD_SEP)
        directory = directory.strip()
        if directory:
            found[directory] = marker.strip()
    return found


# THE CACHE. One connection's answer, kept whether it found builds or failed, so
# a host that cannot be asked is not re-asked on every redraw. Refresh is what
# re-runs it, exactly as it is for the GPU list.

_builds: dict[str, str] = {}
_probe_error: str = ""
_probed: bool = False
# THE ROOT THE LISTING'S KEYS ARE UNDER, recorded by the probe rather than
# derived again by each reader.
#
# WHAT GOES WRONG WITHOUT IT, and it shipped: the listing is keyed by ABSOLUTE
# directories under the solver root, so resolving a device means joining that
# root with a subdirectory and looking the result up. Every reader that spells
# the root itself is a chance to spell a DIFFERENT one, and nothing makes the
# spellings meet, because a lookup that misses is indistinguishable from a host
# that holds no build. The panel derived its own from
# `communicator.normalized_remote_root()`, which is the DATA root
# (`<share>/ppf-cts/git-<branch>/<project>`) and not the solver root, so on a
# real Docker-over-SSH connection it looked up
# `<data root>/target/release`, found nothing, and reported "No solver build
# found under ..." while the probe's own listing sat in this module holding both
# builds. Recording the root here removes the second spelling entirely.
_root: str = ""


def load_builds(lines, root: str = "") -> dict[str, str]:
    """Install the listing parsed from the probe's output, and the root it used.

    *root* is the directory the probe was pointed at, already normalized. It is
    stored rather than re-derived so every reader resolves against the same one.
    """
    global _builds, _probe_error, _probed, _root
    _builds = parse_listing(lines)
    _root = root
    _probe_error = ""
    _probed = True
    return _builds


def record_probe_failure(message: str) -> None:
    """Record that the solver host could not be asked, with the reason."""
    global _builds, _probe_error, _probed, _root
    _builds = {}
    _root = ""
    _probe_error = message
    _probed = True


def forget_builds() -> None:
    """Drop what is known, so nothing is offered until the next probe.

    Called when a connection ends: the next one may reach a different machine,
    where a listing left over from this one would name directories that are not
    there.
    """
    global _builds, _probe_error, _probed, _root
    _builds = {}
    _root = ""
    _probe_error = ""
    _probed = False


def cached_builds() -> dict[str, str]:
    """The solver host's build listing, empty before it has been probed.

    Never probes: reaching the solver host means a command over the backend,
    which belongs on the worker thread that owns the connection, not in the
    panel draw that calls this.
    """
    return _builds


def probed_root() -> str:
    """The root the listing's keys are under, empty before a successful probe.

    READ THIS RATHER THAN DERIVING A ROOT. It is the one the probe actually
    used, so a reader cannot resolve the listing against a directory the keys
    were never built from.
    """
    return _root


def probe_error() -> str:
    """The message from the last probe, or "" if it succeeded or none has run."""
    return _probe_error


def has_probed() -> bool:
    """True once the solver host has been asked for this connection."""
    return _probed
