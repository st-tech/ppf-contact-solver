#!/usr/bin/env python3
# File: .github/workflows/scripts/check-rocm-code-objects.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The AMD half of the ROCm backend has no run to stand behind it, so the static
# evidence carries the weight those runs would have carried. This asserts that
# evidence rather than
# printing it, and it asks the SHIPPED LIBRARY: every target in rocm_arch.txt
# must be present in its device image, each image must disassemble to a NON-ZERO
# instruction count, and none of them may contain an FP64 instruction.
#
# IT READS THE LINKED LIBRARY AND NOT THE ENTRY OBJECTS, and that is forced
# rather than preferred. The build passes `-fgpu-rdc`, which defers device code
# generation to the device LINK, so a `-c` object's bundle holds LLVM bitcode:
# `clang-offload-bundler` extracts it happily and `llvm-objdump` then refuses the
# result as "not a valid object file". There is no amdgcn instruction anywhere in
# the tree until the library is linked.
#
# FOUR WAYS THIS COULD REPORT CLEAN OVER NOTHING, each measured somewhere in
# this tree and each refused here by name:
#
#   * `llvm-objdump --mcpu=<target>` FAILS OPEN. An image whose target does not
#     match exits 0 with an EMPTY disassembly, the same shape `cuobjdump
#     --dump-sass -arch <sm>` has on the CUDA side. The unbundled ELF names
#     its own target, so the flag can only silence the dump. It is not passed, and the
#     instruction count is asserted before any verdict is read off it.
#   * A SUBSTRING scan for `f64` over-reports: a clean image contains exactly
#     one hit, `elf64-amdgpu` in objdump's own header line. The pattern here is
#     a whole mnemonic ending in `_f64`, the rule the CUDA FP64 guard states.
#   * `clang-offload-bundler --unbundle` EXITS 0 FOR A TARGET THE BUNDLE DOES
#     NOT CARRY, writing a file that is not an object. The target triples are
#     therefore READ from `--list` rather than spelled: a `-c` bundle names its
#     entries `hip-amdgcn-...` and a linked fatbin names them `hipv4-amdgcn-...`,
#     so a constructed triple silently extracts nothing on one of the two.
#   * A LIBRARY BUILT FOR THE OTHER PLATFORM looks like a clean AMD result with
#     no device code to find. `HIP_PLATFORM=nvidia` produces the same file name
#     carrying `.nv_fatbin`, so the section is checked by name and an NVIDIA
#     library is refused here rather than reported as an AMD one with nothing
#     in it.
#
import argparse
import os
import re
import subprocess
import sys
import tempfile

# A whole mnemonic ending in _f64, optionally with one suffix group. Never a
# substring scan: see the header.
FP64 = re.compile(r"\b[a-z][a-z0-9_]*_f64(?:_[a-z0-9]+)?\b")
# llvm-objdump prints the encoding after a `//` on each instruction line.
INSTR = re.compile(r"//\s+[0-9A-Fa-f]{8,}:")


def fail(message):
    print(f"\nFAIL: {message}")
    sys.exit(1)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def targets_from_manifest(path):
    """The shipped AMD target list, read from its single source.

    Spelled the way crates/ppf-cts-compute/rocm/Makefile reads it, so this gate
    and the build cannot disagree about what was asked for.
    """
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split(";", 1)[0].split()
            if len(parts) >= 2 and parts[0] == "target":
                out.append(parts[1])
    if not out:
        fail(f"no 'target' line in {path}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", required=True,
                    help="the linked libppfbe_rocm.so to inspect")
    ap.add_argument("--arch-file", required=True, help="rocm_arch.txt")
    ap.add_argument("--rocm-path", default=os.environ.get("ROCM_PATH", ""),
                    help="ROCm SDK root; llvm/bin supplies the three tools")
    args = ap.parse_args()

    # The Windows SDK names every tool with `.exe`, and a lookup of the bare name
    # there reports a complete SDK as missing its tools.
    suffix = ".exe" if os.name == "nt" else ""
    tools = {}
    for name in ("clang-offload-bundler", "llvm-objdump", "llvm-objcopy"):
        path = os.path.join(args.rocm_path, "llvm", "bin", name + suffix)
        if not os.path.exists(path):
            fail(f"{path} is missing: --rocm-path {args.rocm_path} does not look "
                 f"like a ROCm SDK root")
        tools[name] = path
    if not os.path.exists(args.library):
        fail(f"no library at {args.library}. Build it first: "
             f"make -C crates/ppf-cts-compute/rocm abi")

    wanted = targets_from_manifest(args.arch_file)
    print(f"library : {args.library} ({os.path.getsize(args.library)} bytes)")
    print(f"targets : {' '.join(wanted)}")

    print("\n=== 0. this is an AMD-platform library ===")
    headers = run([tools["llvm-objdump"], "-h", args.library])
    if headers.returncode:
        fail(f"llvm-objdump could not read {args.library}:\n"
             f"{headers.stderr.strip()[:400]}")
    # THE SECTION IS FOUND BY THE NAME THE LIBRARY RECORDS, not by the ELF
    # spelling. An ELF library names it `.hip_fatbin`; a PE image stores at most
    # eight bytes of a section name in its header, so a Windows DLL can carry the
    # same section under a shortened name. The first token of each objdump
    # section row is compared as a whole name, and the name found is the one
    # dumped below.
    names = []
    for line in headers.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].isdigit():
            names.append(fields[1])
    hip_names = [n for n in names if n == ".hip_fatbin" or (os.name == "nt" and n == ".hip_fat")]
    nv_names = [n for n in names if n == ".nv_fatbin" or (os.name == "nt" and n == ".nv_fatb")]
    has_hip = bool(hip_names)
    has_nv = bool(nv_names)
    print(f"  sections: {' '.join(names)}")
    print(f"  AMD device section: {hip_names or 'none'}    NVIDIA device section: {nv_names or 'none'}")
    if has_nv and not has_hip:
        fail("this library was built with HIP_PLATFORM=nvidia: it carries "
             "`.nv_fatbin` and no `.hip_fatbin`, so it holds CUDA device code "
             "and there is no AMD image to inspect. Build the AMD platform "
             "(plain `make abi`) before running this gate.")
    if not has_hip:
        fail("this library carries no `.hip_fatbin` section, so it holds no AMD "
             "device code at all. A gate reporting clean over that is the "
             "mistake this refuses to make.")

    with tempfile.TemporaryDirectory() as tmp:
        fatbin = os.path.join(tmp, "device.fatbin")
        # llvm-objcopy wants an output file even when only dumping a section. It
        # writes a scratch file rather than the null device, which llvm-objcopy
        # cannot rename its output onto on Windows.
        dump = run([tools["llvm-objcopy"], f"--dump-section={hip_names[0]}={fatbin}",
                    args.library, os.path.join(tmp, "discarded")])
        if dump.returncode or not os.path.exists(fatbin):
            fail(f"could not dump .hip_fatbin out of {args.library}:\n"
                 f"{dump.stderr.strip()[:400]}")
        print(f"  .hip_fatbin: {os.path.getsize(fatbin)} bytes")

        print("\n=== 1. every target in the manifest is in the image ===")
        listing = run([tools["clang-offload-bundler"], "--type=o", "--list",
                       "--input=" + fatbin])
        if listing.returncode:
            fail(f"clang-offload-bundler could not list the fatbin:\n"
                 f"{listing.stderr.strip()[:400]}")
        entries = listing.stdout.split()
        print(f"  bundle entries: {len(entries)}")
        for entry in sorted(entries):
            print(f"    {entry}")
        # Suffix rather than equality: the entry carries a bundle-format prefix
        # and an environment field around the target name. The matched string is
        # kept and used verbatim below, for the reason the header gives.
        entry_of = {}
        absent = []
        for target in wanted:
            hit = [e for e in entries if e.endswith(target)]
            if hit:
                entry_of[target] = hit[0]
            else:
                absent.append(target)
        if absent:
            fail(f"the device image carries no code for {' '.join(absent)}. "
                 f"A target named in {args.arch_file} is a compile-coverage "
                 f"claim, and a shipped library short of one makes it false.")
        print(f"  all {len(wanted)} manifest targets present")

        print("\n=== 2. instruction counts and FP64, per target ===")
        bad = []
        for target in wanted:
            elf = os.path.join(tmp, target + ".elf")
            extract = run([tools["clang-offload-bundler"], "--type=o",
                           "--unbundle", "--input=" + fatbin,
                           "--targets=" + entry_of[target], "--output=" + elf])
            if extract.returncode or not os.path.exists(elf):
                fail(f"could not unbundle {target}:\n"
                     f"{extract.stderr.strip()[:400]}")
            # NO --mcpu: it fails open, see the header.
            asm = run([tools["llvm-objdump"], "-d", elf])
            if asm.returncode:
                fail(f"llvm-objdump refused the {target} image. The unbundle "
                     f"reported success, so this is the fail-open the header "
                     f"names: the extraction produced something that is not an "
                     f"object.\n{asm.stderr.strip()[:400]}")
            count = len(INSTR.findall(asm.stdout))
            offenders = {}
            for hit in FP64.findall(asm.stdout):
                offenders[hit] = offenders.get(hit, 0) + 1
            print(f"  {target:<18} instructions={count:<10} "
                  f"distinct FP64 mnemonics={len(offenders)}")
            # THE COUNT IS ASSERTED BEFORE THE VERDICT. An empty disassembly
            # reported as zero FP64 is the fail-open this exists to refuse.
            if count == 0:
                bad.append(f"{target}: disassembled to NOTHING, so no statement "
                           f"about FP64 can be made for it")
            if offenders:
                for name, n in sorted(offenders.items(), key=lambda kv: -kv[1])[:8]:
                    print(f"      {n:>6}  {name}")
                bad.append(f"{target}: {len(offenders)} distinct FP64 mnemonic(s)")

    if bad:
        print()
        for line in bad:
            print(f"  {line}")
        fail("GPU compute in this project is float32 only, and an AMD code "
             "object carrying an FP64 instruction breaks that whether or not "
             "any AMD device ever runs it.")

    print(f"\nOK: the shipped library carries all {len(wanted)} targets, every "
          f"image disassembles to real instructions, and none contains FP64")
    return 0


sys.exit(main())
