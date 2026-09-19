#!/usr/bin/env python3
# File: build-linux-native/scripts/elf-audit.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
"""Audit every ELF file in a distribution directory, for bundle.sh.

    elf-audit.py PAYLOAD --build-tree SRC_DIR --arch {x86_64,aarch64}
                 [--exempt RELGLOB:SONAME ...] [--floor-exempt RELGLOB ...]
                 [--machine-exempt RELGLOB ...]

Run with the BUILD HOST's python3, never the payload's interpreter, so the audit
does not depend on the thing it audits.

It walks the whole payload once, identifies ELF files by their magic bytes (not
by name, so a library whose name says nothing is still found), and reads each
one's dynamic section and version requirements with `readelf`. It prints the
count before any verdict, because a walk that read nothing reports no errors.

Checks, each fatal:

  walk      every regular file named *.so or *.so.<n> is on the ELF list. A
            short walk passes every other check over the files it missed.
  gate A    no NEEDED, RPATH or RUNPATH entry names the build tree.
  gate B    every NEEDED entry resolves, the way the glibc loader would, to a
            file inside the payload, or is on the SYSTEM allowlist below, or is
            exempted by name for one file with a reason bundle.sh states.
  gate D    no library owned by the NVIDIA driver is in the payload. It has to
            match the kernel module of the machine that runs it.
  gate E    every ELF file is a 64-bit little-endian object for --arch, read
            off its header's e_machine field, or matches --machine-exempt. A
            wheel that claims no platform can still carry a native library
            built for another one, and the loader reports that only when
            something tries to load it, which may be never on the build host
            and always on a user's machine.

And one report, which bundle.sh reads: the highest GLIBC_, GLIBCXX_ and CXXABI_
version any ELF file requires, printed as `FLOOR <name> <version> <file>`. A
file matching --floor-exempt is left out of that report and printed instead as
`FLOOR-EXEMPT GLIBC <its highest version> <file>`, so what was left out stays
visible. A floor exemption that matches no ELF file is an error, as an unused
--exempt or --machine-exempt is.

HOW GATE B RESOLVES A NAME. glibc searches, for an object with no DT_RUNPATH,
the DT_RPATH of the object and then the DT_RPATH of each object that loaded it
up to the executable; LD_LIBRARY_PATH; the object's own DT_RUNPATH; the loader
cache; the default directories. The last three belong to the host, so only the
first and third count here. An extension module under python/ is loaded by the
interpreter, whose DT_RPATH names python/lib, so that directory is inherited by
every module there that has no DT_RUNPATH of its own; that is how the bundled
Tcl/Tk libraries reach _tkinter. A NEEDED entry containing a slash is a path,
with $ORIGIN expanded, and PBS uses that form in libpython3.so. ldd is never
used: it answers where a name resolves on THIS host, including through its
cache and environment, which is the question the gate exists not to ask.
"""

import argparse
import fnmatch
import os
import re
import subprocess
import sys

# Libraries a distribution may take from the machine it runs on.
#
# The glibc family is part of the C library itself. libstdc++ and libgcc_s are
# the two the manylinux policy treats as system, which every compiled wheel in
# the payload already relies on. libz.so.1 is left unbundled by that same policy
# in the wheels this payload carries (Pillow, the scipy runtime, llvmlite), so
# they load the host's copy whether or not anything else does.
#
# Nothing is added here to make a build pass. A name that fails gate B is either
# a file that did not travel, or a decision that belongs in --exempt with a
# reason.
SYSTEM = {
    "libc.so.6",
    "libm.so.6",
    "libdl.so.2",
    "librt.so.1",
    "libpthread.so.0",
    "libutil.so.1",
    "libresolv.so.2",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    "libz.so.1",
}

# glibc's dynamic loader, the one member of the family whose soname differs by
# architecture, so it joins SYSTEM for --arch alone.
LOADER = {
    "x86_64": "ld-linux-x86-64.so.2",
    "aarch64": "ld-linux-aarch64.so.1",
}

# The e_machine value of each architecture this audits, from the System V ABI
# (EM_X86_64 and EM_AARCH64), and names for the values a wrong file is likely to
# carry, so gate E says what it found rather than a number.
MACHINE = {"x86_64": 62, "aarch64": 183}
MACHINE_NAMES = {3: "i386", 40: "arm", 62: "x86_64", 183: "aarch64", 243: "riscv"}

# The four a ROCm payload adds, and ONLY a ROCm payload: enabled by --rocm so a
# CUDA distribution that somehow acquired a dependency on libdrm_amdgpu still
# fails gate B rather than inheriting an allowance written for another backend.
#
# MEASURED rather than assumed, from the 7.2.4 packages with no install:
# `readelf -d` over the three load-time libraries names exactly these
# four outside the glibc family and libstdc++/libgcc_s. `libhsa-runtime64.so.1`
# needs all four, `libamdhip64.so.7` needs none of them, and
# `librocprofiler-register.so.0` needs none.
#
# They are the same shape as the CUDA story's relationship with the driver: the
# user supplies a machine with working graphics, not a ROCm installation.
ROCM_SYSTEM = {
    "libelf.so.1",
    "libdrm.so.2",
    "libdrm_amdgpu.so.1",
    "libnuma.so.1",
}

# Libraries that belong to the NVIDIA driver and must never ship.
DRIVER_OWNED = ("libcuda.so*", "libnvidia-*", "libnvcuvid.so*", "libcudadebugger.so*")

SO_NAME = re.compile(r"\.so(\.[0-9]+)*$")
VERSION_NAME = re.compile(r"Name: (GLIBC|GLIBCXX|CXXABI)_([0-9][0-9.]*)\b")


def is_elf(path):
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def elf_header(path):
    """(EI_CLASS, EI_DATA, e_machine) off the file's own header.

    Read directly rather than through readelf, whose wording for the machine
    field varies between binutils releases. e_machine is the two bytes at offset
    18, in the byte order EI_DATA names: 1 for little-endian, 2 for big-endian.
    """
    with open(path, "rb") as handle:
        head = handle.read(20)
    if len(head) < 20:
        return None
    order = "little" if head[5] == 1 else "big"
    return head[4], head[5], int.from_bytes(head[18:20], order)


def version_key(text):
    return tuple(int(part) for part in text.split("."))


class Elf:
    def __init__(self, path):
        self.path = path
        out = subprocess.run(
            ["readelf", "-d", "-V", "-W", path],
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode != 0:
            raise RuntimeError(f"readelf failed on {path}: {out.stderr.strip()}")
        text = out.stdout
        self.needed = re.findall(r"\(NEEDED\)\s+Shared library: \[(.+?)\]", text)
        self.rpath = self._paths(re.findall(r"\(RPATH\)\s+Library rpath: \[(.*?)\]", text))
        self.runpath = self._paths(re.findall(r"\(RUNPATH\)\s+Library runpath: \[(.*?)\]", text))
        self.versions = VERSION_NAME.findall(text)
        self.header = elf_header(path)

    @staticmethod
    def _paths(entries):
        return [part for entry in entries for part in entry.split(":") if part]

    def expand(self, entries):
        origin = os.path.dirname(self.path)
        return [e.replace("${ORIGIN}", origin).replace("$ORIGIN", origin) for e in entries]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("payload")
    parser.add_argument("--build-tree", required=True)
    parser.add_argument("--exempt", action="append", default=[], metavar="RELGLOB:SONAME")
    parser.add_argument("--floor-exempt", action="append", default=[], metavar="RELGLOB")
    parser.add_argument("--arch", required=True, choices=sorted(MACHINE),
                        help="the architecture every ELF file in the payload is "
                             "built for, which is the build host's")
    parser.add_argument("--machine-exempt", action="append", default=[], metavar="RELGLOB",
                        help="a file gate E leaves alone, named with its reason "
                             "by the caller")
    parser.add_argument("--rocm", action="store_true",
                        help="the payload carries the ROCm runtime, so the four "
                             "names in ROCM_SYSTEM are supplied by the machine "
                             "for the files --rocm-scope names")
    parser.add_argument("--rocm-scope", action="append", default=[], metavar="RELGLOB",
                        help="a file the ROCM_SYSTEM allowance applies to, which "
                             "is the ROCm side of the payload; required with --rocm")
    args = parser.parse_args()
    floor_exemptions = [(glob, [False]) for glob in args.floor_exempt]
    machine_exemptions = [(glob, [False]) for glob in args.machine_exempt]
    system = SYSTEM | {LOADER[args.arch]}
    # THE ROCm ALLOWANCE IS SCOPED TO THE ROCm FILES, NOT TO THE PAYLOAD. A
    # distribution carries every backend it was built with, so a CUDA library
    # that acquired a dependency on libdrm_amdgpu sits in the same payload as
    # the ROCm runtime that legitimately needs it, and it must still fail gate B
    # rather than inherit an allowance written for the other backend. The caller
    # names the files it applies to, which is the ROCm target directory, its
    # backend library and the runtime closure bundle.sh walked.
    rocm_scope = list(args.rocm_scope)
    if args.rocm and not rocm_scope:
        sys.exit("elf-audit: --rocm needs at least one --rocm-scope RELGLOB")
    if rocm_scope and not args.rocm:
        sys.exit("elf-audit: --rocm-scope was given without --rocm")
    if args.rocm:
        print("elf-audit: ROCm payload, so the machine supplies "
              + " ".join(sorted(ROCM_SYSTEM)) + " for " + " ".join(rocm_scope))

    payload = os.path.realpath(args.payload)
    build_tree = os.path.realpath(args.build_tree)
    exemptions = []
    for item in args.exempt:
        glob, sep, soname = item.rpartition(":")
        if not sep or not glob or not soname:
            sys.exit(f"elf-audit: --exempt wants RELGLOB:SONAME, got {item!r}")
        exemptions.append((glob, soname, [False]))

    elves, named_libraries = [], []
    for directory, _, files in os.walk(payload):
        for name in files:
            path = os.path.join(directory, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            if is_elf(path):
                elves.append(path)
            elif SO_NAME.search(name):
                named_libraries.append(path)
    elves.sort()
    print(f"elf-audit: {len(elves)} ELF files under {payload}")
    if not elves:
        print("ERROR: [walk] the walk found no ELF file, which is a failed walk, not a clean payload")
        return 1

    errors = 0
    rel = lambda p: os.path.relpath(p, payload)

    # A file named like a shared library that is not ELF is either a linker
    # script or something the walk misread; neither is expected in a payload.
    for path in named_libraries:
        print(f"ERROR: [walk] {rel(path)} is named like a shared library and is not ELF")
        errors += 1

    parsed = {}
    for path in elves:
        try:
            parsed[path] = Elf(path)
        except RuntimeError as exc:
            print(f"ERROR: [walk] {exc}")
            errors += 1

    # The interpreter's DT_RPATH, inherited by modules under python/ that carry
    # no DT_RUNPATH of their own.
    interpreter = os.path.realpath(os.path.join(payload, "python", "bin", "python3"))
    inherited = []
    if interpreter in parsed and not parsed[interpreter].runpath:
        inherited = parsed[interpreter].expand(parsed[interpreter].rpath)
    python_root = os.path.join(payload, "python") + os.sep

    floors = {}
    want_machine = MACHINE[args.arch]
    matched_machine = 0
    for path, elf in parsed.items():
        name = rel(path)

        machine_exempt = [e for e in machine_exemptions if fnmatch.fnmatch(name, e[0])]
        for e in machine_exempt:
            e[1][0] = True
        if elf.header is None:
            print(f"ERROR: [gate E] {name} has an ELF magic and no complete header")
            errors += 1
        elif elf.header == (2, 1, want_machine):
            matched_machine += 1
        elif machine_exempt:
            found = MACHINE_NAMES.get(elf.header[2], f"e_machine {elf.header[2]}")
            print(f"MACHINE-EXEMPT {found} {name}")
        else:
            ei_class, ei_data, machine = elf.header
            found = MACHINE_NAMES.get(machine, f"e_machine {machine}")
            width = {1: "32-bit", 2: "64-bit"}.get(ei_class, f"EI_CLASS {ei_class}")
            order = {1: "little-endian", 2: "big-endian"}.get(ei_data, f"EI_DATA {ei_data}")
            print(f"ERROR: [gate E] {name} is a {width} {order} {found} object, "
                  f"and this payload is {args.arch}")
            errors += 1

        for entry in elf.needed + elf.rpath + elf.runpath:
            if build_tree in entry:
                print(f"ERROR: [gate A] {name} names the build tree: {entry}")
                errors += 1

        if elf.runpath:
            search = elf.expand(elf.runpath)
        else:
            search = elf.expand(elf.rpath)
            if path.startswith(python_root) and path != interpreter:
                search += inherited
        for soname in elf.needed:
            if "/" in soname:
                target = elf.expand([soname])[0]
                if os.path.exists(target) and os.path.realpath(target).startswith(payload + os.sep):
                    continue
            elif any(os.path.exists(os.path.join(d, soname)) for d in search
                     if os.path.realpath(d).startswith(payload)):
                continue
            elif soname in system:
                continue
            elif soname in ROCM_SYSTEM and any(
                    fnmatch.fnmatch(name, glob) for glob in rocm_scope):
                continue
            exempt = [e for e in exemptions if e[1] == soname and fnmatch.fnmatch(name, e[0])]
            if exempt:
                for e in exempt:
                    e[2][0] = True
                continue
            print(f"ERROR: [gate B] {name} needs {soname}, which is not in the payload on its search path and is not a system library")
            errors += 1

        base = os.path.basename(path)
        if any(fnmatch.fnmatch(base, pattern) for pattern in DRIVER_OWNED):
            print(f"ERROR: [gate D] {name} belongs to the NVIDIA driver and must not ship")
            errors += 1

        floor_exempt = [e for e in floor_exemptions if fnmatch.fnmatch(name, e[0])]
        for e in floor_exempt:
            e[1][0] = True
        if floor_exempt:
            glibc = [version for family, version in elf.versions if family == "GLIBC"]
            highest = max(glibc, key=version_key) if glibc else "none"
            print(f"FLOOR-EXEMPT GLIBC {highest} {name}")
            continue
        for family, version in elf.versions:
            best = floors.get(family)
            if best is None or version_key(version) > version_key(best[0]):
                floors[family] = (version, name)

    # An exemption that matched nothing is a stale statement about the payload,
    # and a stale exemption is the kind that later covers something it was never
    # written for.
    for glob, soname, used in exemptions:
        if not used[0]:
            print(f"ERROR: [gate B] the exemption {glob}:{soname} matched nothing")
            errors += 1
    # A scope naming no file is a stale statement about the payload, the same
    # way an exemption that matched nothing is. It is checked against the files
    # rather than against the names they needed: a ROCm library that needs none
    # of the four is still part of the ROCm side.
    payload_names = [rel(path) for path in parsed]
    for glob in rocm_scope:
        if not any(fnmatch.fnmatch(name, glob) for name in payload_names):
            print(f"ERROR: [gate B] the ROCm scope {glob} matches no file in the payload")
            errors += 1
    for glob, used in floor_exemptions:
        if not used[0]:
            print(f"ERROR: [floor] the floor exemption {glob} matched no ELF file")
            errors += 1
    for glob, used in machine_exemptions:
        if not used[0]:
            print(f"ERROR: [gate E] the machine exemption {glob} matched no ELF file")
            errors += 1
    print(f"elf-audit: [gate E] {matched_machine} of {len(parsed)} ELF files are 64-bit little-endian {args.arch}")

    for family in ("GLIBC", "GLIBCXX", "CXXABI"):
        if family in floors:
            version, name = floors[family]
            print(f"FLOOR {family} {version} {name}")

    if errors:
        print(f"elf-audit: {errors} error(s) over {len(elves)} ELF files")
        return 1
    print(f"elf-audit: OK over {len(elves)} ELF files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
