#!/usr/bin/env python3
# File: build-win-native/scripts/pe-audit.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Reads Windows PE images for the build and bundle scripts, which have no reliable
# way to parse one from cmd.exe. Three questions, one subcommand each:
#
#   machine   every PE image under a distribution is built for one architecture,
#             read off its own header's Machine field, or matches a named
#             exemption. This is the Windows counterpart of build-mac-native's
#             GATE D (assert_arm64_only) and build-linux-native/scripts/
#             elf-audit.py's gate E.
#   backend   a GPU backend DLL exports exactly the ABI its module-definition file
#             lists, and imports the device runtime its platform names and not the
#             other one.
#   closure   the DLLs a library loads, walked through the directories they may be
#             shipped from, so a distribution carries every file its backend needs
#             and nothing it does not declare as the system's.
#
# THE HEADER IS READ WITH PYTHON, NOT WITH dumpbin, for two reasons. The bundle
# step runs where no MSVC environment is loaded, and dumpbin's text output is a
# format for people: a parser over it reports nothing at all the day a column
# moves, which is a gate passing over nothing.
#
# WHY THE MACHINE GATE EXISTS. Windows on ARM runs x64 code under emulation, so an
# x64 file in an ARM64 distribution runs, answers every smoke test, and is slower
# and unverified on the machine it ships to. Nothing else in the build notices:
# pip installs whatever wheel it resolves, and a none-any wheel may carry native
# helpers for several architectures. The gate reads every image, and a file of
# another architecture fails the bundle unless it is exempted by name with its
# reason.
#
# FOUR WAYS A GATE LIKE THIS REPORTS CLEAN OVER NOTHING, each refused here:
#   * a walk that found no images, which is the walk failing rather than a clean
#     payload, since a distribution certainly holds a solver and an interpreter;
#   * a file named like a binary that is not one, which a gate keyed on the
#     extension would skip without saying so;
#   * an exemption that matches nothing, which stays in the list after the file it
#     excused is gone and would excuse the next file to take that name;
#   * a count that is printed without the files behind it, which is why every
#     mismatch and every exempted file is listed.
import argparse
import fnmatch
import os
import struct
import sys

MACHINE_NAMES = {
    0x014C: "x86",
    0x01C4: "ARM (Thumb-2)",
    0x8664: "x64",
    0xAA64: "ARM64",
    0xA641: "ARM64EC",
    0xA64E: "ARM64X",
}
ARCH_MACHINE = {"x64": 0x8664, "arm64": 0xAA64}
# The extensions Windows loads code from. A file carrying one of them that is not
# a PE image is reported, and any other file is read only when it starts with an
# MZ header. `.node` is deliberately absent: Triangle's data files use it for mesh
# vertices, and the gate found 26 of them in the x64 payload.
BINARY_EXTENSIONS = (".exe", ".dll", ".pyd", ".sys", ".ocx", ".cpl")

# The DLLs Windows itself supplies, which a distribution must NOT carry. The
# api-ms-win-* and ext-ms-win-* names are API sets the loader resolves to system
# modules; the rest are modules of every supported Windows installation.
#
# THE VISUAL C++ RUNTIME IS DECLARED HERE AS THE SYSTEM'S, and that is a
# statement about what the distribution already relies on rather than a new
# allowance: the solver, the server and the CUDA backend all link it dynamically
# today, and the clean-instance verification in release.yml runs them with
# nothing installed but the NVIDIA driver.
SYSTEM_DLLS = {
    "kernel32.dll", "kernelbase.dll", "ntdll.dll", "user32.dll", "gdi32.dll",
    "advapi32.dll", "shell32.dll", "ole32.dll", "oleaut32.dll", "ws2_32.dll",
    "bcrypt.dll", "bcryptprimitives.dll", "crypt32.dll", "secur32.dll",
    "userenv.dll", "dbghelp.dll", "setupapi.dll", "cfgmgr32.dll", "version.dll",
    "shlwapi.dll", "psapi.dll", "iphlpapi.dll", "winmm.dll", "powrprof.dll",
    "rpcrt4.dll", "comctl32.dll", "comdlg32.dll", "imm32.dll", "ncrypt.dll",
    "dxgi.dll", "d3d11.dll", "d3d12.dll", "dxcore.dll", "wintrust.dll",
    "normaliz.dll", "netapi32.dll", "mswsock.dll", "ucrtbase.dll", "msvcrt.dll",
    "pdh.dll", "wtsapi32.dll", "dwmapi.dll", "uxtheme.dll", "wldap32.dll",
    "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll",
    "msvcp140_2.dll", "concrt140.dll", "vcomp140.dll",
}
SYSTEM_PREFIXES = ("api-ms-win-", "ext-ms-win-")


class PE:
    """The parts of a PE image this script reads: machine, exports, imports."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as handle:
            self.data = handle.read()
        data = self.data
        if len(data) < 0x40 or data[:2] != b"MZ":
            raise ValueError("no MZ header")
        self.pe = struct.unpack_from("<I", data, 0x3C)[0]
        if self.pe + 24 > len(data) or data[self.pe:self.pe + 4] != b"PE\0\0":
            raise ValueError("an MZ header and no PE signature")
        (self.machine, self.nsections, _time, _symptr, _nsym, self.optsize,
         _chars) = struct.unpack_from("<HHIIIHH", data, self.pe + 4)
        self.opt = self.pe + 24
        magic = struct.unpack_from("<H", data, self.opt)[0]
        if magic == 0x10B:
            self.dirs = self.opt + 96
        elif magic == 0x20B:
            self.dirs = self.opt + 112
        else:
            raise ValueError(f"optional header magic 0x{magic:X}")
        self.ndirs = struct.unpack_from("<I", data, self.dirs - 4)[0]
        self.sections = []
        table = self.opt + self.optsize
        for index in range(self.nsections):
            base = table + 40 * index
            name = data[base:base + 8].rstrip(b"\0").decode("ascii", "replace")
            vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, base + 8)
            self.sections.append((name, vaddr, max(vsize, rawsize), rawptr))

    def il_only(self):
        """True for a .NET assembly holding IL and no code for a fixed machine.

        Such an assembly records Machine 0x14C whatever it runs as: the runtime
        compiles its IL for the process that loads it, so the header's machine is
        a placeholder and not the architecture of any instruction in the file.
        COMIMAGE_FLAGS_ILONLY (0x1) set and COMIMAGE_FLAGS_32BITREQUIRED (0x2)
        clear is that case; an assembly with the second flag, or with native code
        beside its IL, is an image of the machine it names.
        """
        rva, size = self.directory(14)
        if not rva or size < 20:
            return False
        flags = struct.unpack_from("<I", self.data, self.offset(rva) + 16)[0]
        return bool(flags & 0x1) and not flags & 0x2

    def directory(self, index):
        if index >= self.ndirs:
            return 0, 0
        return struct.unpack_from("<II", self.data, self.dirs + 8 * index)

    def offset(self, rva):
        for _name, vaddr, size, rawptr in self.sections:
            if vaddr <= rva < vaddr + size:
                return rawptr + (rva - vaddr)
        raise ValueError(f"RVA 0x{rva:X} lies in no section")

    def string(self, rva):
        start = self.offset(rva)
        end = self.data.index(b"\0", start)
        return self.data[start:end].decode("ascii", "replace")

    def exports(self):
        rva, size = self.directory(0)
        if not rva:
            return []
        base = self.offset(rva)
        count, names = struct.unpack_from("<II", self.data, base + 24)[0], \
            struct.unpack_from("<I", self.data, base + 32)[0]
        table = self.offset(names) if count else 0
        return [self.string(struct.unpack_from("<I", self.data, table + 4 * i)[0])
                for i in range(count)]

    def imports(self):
        """Every DLL name the image imports, static and delay-loaded, as written."""
        found = []
        rva, _size = self.directory(1)
        if rva:
            cursor = self.offset(rva)
            while True:
                entry = struct.unpack_from("<IIIII", self.data, cursor)
                if not any(entry):
                    break
                found.append(self.string(entry[3]))
                cursor += 20
        rva, _size = self.directory(13)
        if rva:
            cursor = self.offset(rva)
            while True:
                entry = struct.unpack_from("<IIIIIIII", self.data, cursor)
                if not any(entry):
                    break
                name_rva = entry[1]
                # A delay descriptor with attribute bit 0 clear holds virtual
                # addresses rather than RVAs. The linkers this project uses emit
                # RVAs, and an image that does otherwise is named rather than
                # misread.
                if not entry[0] & 1:
                    raise ValueError("a delay-import descriptor that does not use RVAs")
                found.append(self.string(name_rva))
                cursor += 32
        return found


def fail(message, *lines):
    print(f"ERROR: {message}", file=sys.stderr)
    for line in lines:
        print(f"       {line}", file=sys.stderr)
    sys.exit(1)


def machine_name(value):
    return MACHINE_NAMES.get(value, f"machine 0x{value:04X}")


def walk(root):
    for directory, _dirs, names in os.walk(root):
        for name in names:
            yield os.path.join(directory, name)


def cmd_machine(args):
    root = os.path.abspath(args.root)
    want = ARCH_MACHINE[args.arch]
    exemptions = []
    for spec in args.exempt:
        pattern, sep, reason = spec.partition("=")
        if not sep or not pattern or not reason.strip():
            fail(f"--exempt {spec!r} is not PATTERN=REASON",
                 "An exemption names what it excuses and why, or it is not accepted.")
        exemptions.append([pattern.replace("\\", "/"), reason.strip(), 0])

    images = mismatched = neutral = 0
    counts = {}
    errors = []
    exempted = []
    for path in walk(root):
        rel = os.path.relpath(path, root).replace("\\", "/")
        named_binary = rel.lower().endswith(BINARY_EXTENSIONS)
        with open(path, "rb") as handle:
            head = handle.read(2)
        if head != b"MZ" and not named_binary:
            continue
        try:
            image = PE(path)
        except (ValueError, struct.error) as problem:
            if named_binary:
                errors.append(f"{rel} is named like a binary and is not a PE image ({problem})")
            continue
        images += 1
        if image.il_only():
            neutral += 1
            continue
        counts[image.machine] = counts.get(image.machine, 0) + 1
        if image.machine == want:
            continue
        mismatched += 1
        matched = [e for e in exemptions if fnmatch.fnmatchcase(rel, e[0])]
        if matched:
            for exemption in matched:
                exemption[2] += 1
            exempted.append(f"{rel} ({machine_name(image.machine)}): {matched[0][1]}")
        else:
            errors.append(f"{rel} is {machine_name(image.machine)}")

    print(f"PE images under {root}: {images}")
    print(f"  IL-only .NET assemblies, no machine code: {neutral}")
    for value, count in sorted(counts.items()):
        print(f"  {machine_name(value)}: {count}")
    if images == 0:
        fail(f"no PE image was found under {root}",
             "That is a failure of the walk, not a clean payload: a distribution",
             "certainly contains a solver and an interpreter.")
    for line in exempted:
        print(f"  exempt: {line}")
    for pattern, reason, hits in exemptions:
        if hits == 0:
            errors.append(f"the exemption {pattern!r} matches no {args.arch}-foreign image. "
                          f"Remove it: an exemption that outlives its file excuses the next "
                          f"file to take its name. It said: {reason}")
    if errors:
        for line in errors:
            print(f"ERROR: [machine] {line}", file=sys.stderr)
        fail(f"{len(errors)} problem(s) in the {args.arch} machine gate",
             f"This distribution is {args.arch} by construction, and every PE image in it",
             f"must be {machine_name(want)} or be exempted by name with its reason.")
    print(f"[OK] every PE image is {machine_name(want)} or exempted by name "
          f"({images - mismatched - neutral} native, {neutral} IL-only, {mismatched} exempted)")


def cmd_backend(args):
    try:
        image = PE(args.dll)
    except (ValueError, struct.error, OSError) as problem:
        fail(f"{args.dll} cannot be read as a PE image: {problem}")
    want = ARCH_MACHINE[args.arch]
    if image.machine != want:
        fail(f"{args.dll} is {machine_name(image.machine)}, and this is a {args.arch} build")
    listed = []
    with open(args.def_file, encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if token and not token.upper().startswith(("EXPORTS", "LIBRARY", ";")):
                listed.append(token.split()[0])
    if not listed:
        fail(f"{args.def_file} lists no export, so there is nothing to compare against")
    exported = image.exports()
    missing = sorted(set(listed) - set(exported))
    extra = sorted(set(exported) - set(listed))
    print(f"{os.path.basename(args.dll)}: {len(exported)} exports, {len(listed)} listed in "
          f"{os.path.basename(args.def_file)}")
    if missing:
        fail(f"{args.dll} does not export {len(missing)} ABI function(s): {', '.join(missing)}",
             "The solver imports every be_* by name, so it would not start.")
    if extra:
        fail(f"{args.dll} exports {len(extra)} name(s) the module-definition file does not list: "
             f"{', '.join(extra)}")
    imported = [name.lower() for name in image.imports()]
    print(f"  imports: {', '.join(imported)}")
    for name in args.must_import:
        if name.lower() not in imported:
            fail(f"{args.dll} does not import {name}",
                 "The platform this library was built for loads that runtime, so a library",
                 "without it was built for another platform than the one named.")
    for name in args.must_not_import:
        if name.lower() in imported:
            fail(f"{args.dll} imports {name}, which belongs to the other platform")
    print("[OK] the backend DLL exports the whole ABI and imports its platform's runtime")


def resolve(name, search):
    for directory in search:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def cmd_closure(args):
    search = [os.path.abspath(d) for d in args.search]
    present = [os.path.abspath(d) for d in args.present]
    for flag, directories in (("--search", search), ("--present", present)):
        for directory in directories:
            if not os.path.isdir(directory):
                fail(f"{flag} {directory} is not a directory")
    shipped = []
    unresolved = []
    seen = set()
    queue = [os.path.abspath(f) for f in args.file]
    while queue:
        path = queue.pop(0)
        key = os.path.basename(path).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            names = PE(path).imports()
        except (ValueError, struct.error, OSError) as problem:
            fail(f"{path} cannot be read as a PE image: {problem}")
        for name in names:
            lower = name.lower()
            if lower in SYSTEM_DLLS or lower.startswith(SYSTEM_PREFIXES) or lower in seen:
                continue
            if lower in {os.path.basename(f).lower() for f in args.file}:
                continue
            # Carried by the distribution already, from a directory it copies
            # whole, so it resolves without being listed for a copy.
            if resolve(name, present) is not None:
                continue
            found = resolve(name, search)
            if found is None:
                unresolved.append(f"{os.path.basename(path)} imports {name}")
                continue
            if found not in shipped:
                shipped.append(found)
            queue.append(found)
    if unresolved:
        for line in unresolved:
            print(f"ERROR: [closure] {line}, which is neither in "
                  f"{' or '.join(search + present)} nor a declared system DLL", file=sys.stderr)
        fail(f"{len(unresolved)} import(s) resolve nowhere",
             "Each is either a library that has to travel with the distribution or a",
             "system library this script does not declare. Neither is fixed by dropping",
             "the check.")
    for path in shipped:
        print(path)


def main():
    ap = argparse.ArgumentParser(description="Read Windows PE images for the build scripts.")
    sub = ap.add_subparsers(dest="command", required=True)

    machine = sub.add_parser("machine", help="every PE image under ROOT is one architecture")
    machine.add_argument("--arch", required=True, choices=sorted(ARCH_MACHINE))
    machine.add_argument("--root", required=True)
    machine.add_argument("--exempt", action="append", default=[], metavar="PATTERN=REASON",
                         help="a path under ROOT, forward slashes, fnmatch wildcards allowed")

    backend = sub.add_parser("backend", help="a backend DLL's exports and device runtime")
    backend.add_argument("--arch", required=True, choices=sorted(ARCH_MACHINE))
    backend.add_argument("--dll", required=True)
    backend.add_argument("--def", dest="def_file", required=True)
    backend.add_argument("--must-import", action="append", default=[])
    backend.add_argument("--must-not-import", action="append", default=[])

    closure = sub.add_parser("closure", help="the non-system DLLs FILE loads, one path per line")
    closure.add_argument("--file", action="append", required=True)
    closure.add_argument("--search", action="append", required=True)
    closure.add_argument("--present", action="append", default=[],
                         help="a directory the distribution carries whole, such as its "
                              "interpreter: an import found there resolves and is not listed")

    args = ap.parse_args()
    {"machine": cmd_machine, "backend": cmd_backend, "closure": cmd_closure}[args.command](args)


main()
