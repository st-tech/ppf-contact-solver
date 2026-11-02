# File: crates/ppf-cts-compute/seam/kernelgen_deps.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
"""Run kernelgen.py once and record everything its output was read from.

The build recipes cache each rendering under the caller's OUT_DIR and reuse it
only while every input it was rendered from is unchanged. WHICH inputs that is
cannot be listed by the caller: a body rendering reads its own source, every
neutral source in its include component (both directions, because an includer's
entry declarations seed the address-space inference), and every header and
kernel in the tree for the record layouts. A list written by hand would be a
second copy of the generator's dependency logic, and a stale rendering is a
silently wrong build. So the generator is run under an audit hook and the read
set is OBSERVED rather than predicted:

  read <sha256> <path>   a file the generator opened for reading
  probe <kind> <path>    a stat-type existence check, kind f, d, o or n
  call <sha256> <json>   a whole-tree scan, recorded by the value it returned

The third kind is what makes the cache useful at all. `pointee_sizes` and
`_transitive_includers` read every file in the tree, so recording their reads
would make every rendering depend on every kernel and any edit would miss the
whole cache. They are pure functions of `KERNEL_ROOT`, their arguments and the
files on disk, so they are recorded by the repr of what they returned, and
`validate` calls the generator's OWN function again to check it. Nothing here
re-implements what they compute. `_kernel_root_of` is the same kind of function,
a walk up the directory tree.

ANYTHING THIS CANNOT ACCOUNT FOR MAKES THE RENDERING UNCACHEABLE, never assumed
harmless: a directory listing outside a summarized call, an open for writing of
anything but `--out`, a path that is not a string, or a summarized call that
raised. The rendering is then produced by running the generator every build,
which is slower and never stale.

Usage, by the recipe and never by hand:
  kernelgen_deps.py run <manifest> <kernelgen arguments...>
  kernelgen_deps.py validate   (call specs on stdin, one digest per line out)
"""

import hashlib
import json
import os
import sys

# The generator sits beside this file. An interpreter with a `._pth` file (the
# embedded Python the Windows build runs as `python3`) does not put a script's
# own directory on `sys.path`, so it is put there by name.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kernelgen  # noqa: E402 - needs the path above

# Whole-tree scans, recorded by value. Each must be a pure function of
# `kernelgen.KERNEL_ROOT`, its arguments and the files on disk.
SUMMARIZED = ("pointee_sizes", "_transitive_includers", "_kernel_root_of")

LISTING_EVENTS = ("os.listdir", "os.scandir", "glob.glob", "glob.glob/2")


def _result_digest(value):
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _classify(real_stat, path):
    import stat as stat_mod
    try:
        mode = real_stat(path).st_mode
    except (OSError, ValueError):
        return "n"
    if stat_mod.S_ISREG(mode):
        return "f"
    if stat_mod.S_ISDIR(mode):
        return "d"
    return "o"


def run(manifest, argv):
    if "--out" not in argv or argv.index("--out") + 1 >= len(argv):
        out = None
    else:
        out = os.path.abspath(argv[argv.index("--out") + 1])
    lines, seen = [], set()
    # `quiet` is raised while this module does its own I/O and while a
    # summarized call runs, so neither is recorded as the generator's.
    state = {"quiet": 0, "cacheable": out is not None}
    real_stat = os.stat
    real_open = open

    def note(line):
        if "\n" in line or "\r" in line:
            state["cacheable"] = False
        elif line not in seen:
            seen.add(line)
            lines.append(line)

    def as_path(path):
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if not isinstance(path, str):
            return None
        return os.path.abspath(path)

    def probe(path):
        resolved = as_path(path)
        if resolved is None:
            state["cacheable"] = False
            return
        if resolved == out:
            return
        state["quiet"] += 1
        try:
            note(f"probe {_classify(real_stat, resolved)} {resolved}")
        finally:
            state["quiet"] -= 1

    def hook(event, args):
        if state["quiet"]:
            return
        if event == "open":
            path, mode, flags = args[0], args[1], args[2]
            resolved = as_path(path)
            if resolved is None:
                state["cacheable"] = False
                return
            if resolved == out:
                return
            if mode is not None:
                writing = any(c in mode for c in "wax+")
            else:
                writing = bool(flags & (os.O_WRONLY | os.O_RDWR))
            if writing:
                state["cacheable"] = False
                return
            state["quiet"] += 1
            try:
                note(f"read {_file_digest(resolved)} {resolved}")
            except OSError:
                note(f"probe {_classify(real_stat, resolved)} {resolved}")
            finally:
                state["quiet"] -= 1
        elif event in LISTING_EVENTS:
            state["cacheable"] = False

    def stat_probe(real):
        def wrapper(path, *args, **kwargs):
            if not state["quiet"]:
                probe(path)
            state["quiet"] += 1
            try:
                return real(path, *args, **kwargs)
            finally:
                state["quiet"] -= 1
        return wrapper

    def summarize(name):
        real = getattr(kernelgen, name)

        def wrapper(*args, **kwargs):
            if state["quiet"]:
                return real(*args, **kwargs)
            state["quiet"] += 1
            try:
                root = kernelgen.KERNEL_ROOT
                try:
                    result = real(*args, **kwargs)
                except BaseException:
                    state["cacheable"] = False
                    raise
                try:
                    spec = json.dumps({"fn": name, "root": root,
                                       "args": list(args), "kwargs": kwargs},
                                      sort_keys=True)
                except (TypeError, ValueError):
                    state["cacheable"] = False
                    return result
                note(f"call {_result_digest(result)} {spec}")
                return result
            finally:
                state["quiet"] -= 1
        setattr(kernelgen, name, wrapper)

    for name in SUMMARIZED:
        summarize(name)
    os.stat = stat_probe(os.stat)
    for name in ("isfile", "isdir", "exists"):
        setattr(os.path, name, stat_probe(getattr(os.path, name)))
    sys.addaudithook(hook)

    code = kernelgen.main(["kernelgen.py"] + argv)

    state["quiet"] += 1
    if code == 0 and state["cacheable"]:
        with real_open(manifest, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(line + "\n" for line in lines))
    return code


def validate():
    """One digest per call spec on stdin, in order, or `error` if it raised."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            spec = json.loads(line)
            kernelgen.KERNEL_ROOT = spec["root"]
            function = getattr(kernelgen, spec["fn"])
            result = function(*spec["args"], **spec["kwargs"])
            sys.stdout.write(_result_digest(result) + "\n")
        except Exception:
            sys.stdout.write("error\n")
    sys.stdout.flush()
    return 0


def main(argv):
    if len(argv) >= 3 and argv[1] == "run":
        return run(argv[2], argv[3:])
    if len(argv) == 2 and argv[1] == "validate":
        return validate()
    sys.stderr.write("kernelgen_deps.py: usage: run <manifest> <kernelgen args>"
                     " | validate\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
