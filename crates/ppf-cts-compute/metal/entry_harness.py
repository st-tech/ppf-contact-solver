# File: entry_harness.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Writes the translation unit that hands ONE generated `*.entry.metal` to the
# Metal shader compiler.
#
# WHY IT EXISTS. A generated entry point is rendered for four targets and three
# of them are compiled by somebody: nvcc compiles `*.entry.cu` into the C ABI
# library, the host compiler compiles `*.entry.cpp` into the CPU backend's shim,
# and rustc compiles `*.entry.rs` into the driver. The MSL rendering was
# compiled by NOTHING, and the cost of that was measured rather than argued: the
# rendering of every gathered struct element did not compile at all, on 71 sites
# across 26 declarations, because `buffer[index]` is a `device` lvalue on Metal
# and a body parameter written `[[seam::thread]] const T &` renders as `thread
# const T &`. Nothing said so for as long as no shader compiler read one.
#
# WHAT THIS UNIT IS, AND WHAT IT IS NOT. It is the backend's own MSL prologue,
# the neutral type vocabulary, the neutral body, and the entry rendering under
# test, in that order. It is NOT the assembled shader: the shipped libraries are
# spliced by the backend in an order this script does not know and must not
# reproduce, and their `#line` and `DIAG_FILE_ID` directives belong to that
# assembly. So this proves that the ENTRY POINT is valid MSL against the real
# prologue and the real body; it proves nothing about segment order, and the
# fixture suite is what proves that.
#
# WHY IT USES `#include` WHEN THE RUN-TIME SHADER CANNOT. `newLibraryWithSource`
# is handed one string with no filesystem behind it, which is why the assembler
# neutralizes every quoted include as it splices. `xcrun metal` is clang and has
# a filesystem, so here the includes are live and the compiler resolves what the
# splice order resolves at run time. That is the whole reason an offline check
# is possible at all without a second copy of the segment list.
#
# THE INCLUDE WALK. A neutral body names its dependencies with quoted includes
# and the rule for a body is that they are unconditional, so this follows them:
# another `*.kernel.cpp` becomes its own `*.kernel.metal` rendering and is
# walked first, and any other header is emitted as itself. A header is NOT
# walked, and it does not have to be: this compiler has a filesystem, so a
# quoted include inside a header is LIVE here and resolves itself. What belongs
# in the caller's context list is only what a unit reaches WITHOUT naming, and a
# miss there is a compile error naming the undeclared symbol.

import argparse
import os
import re
import sys

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')
KERNEL_SUFFIX = ".kernel.cpp"


def fail(message):
    sys.stderr.write(f"entry_harness: {message}\n")
    raise SystemExit(2)


def quoted_includes(path):
    """The quoted include targets of one file, in order.

    Line-based and deliberately so: a neutral kernel body carries no
    preprocessor directive other than `#pragma once` and a quoted `#include`
    (kernelgen.py enforces that), so there is no conditional here to interpret
    and no reason to build a preprocessor to find out.
    """
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = INCLUDE_RE.match(line)
                if m:
                    out.append(m.group(1))
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    return out


def walk(kernel_root, rel_source, seen, emitted):
    """Depth-first, dependencies before the file that names them.

    `emitted` collects ("body"|"header", path-relative-to-its-root) pairs. A
    body's path is relative to the GENERATED tree and a header's to the neutral
    tree; the caller puts both on the include path, and the two cannot collide
    because a rendering never carries a `.hpp` name.
    """
    if rel_source in seen:
        return
    seen.add(rel_source)
    source = os.path.join(kernel_root, rel_source)
    if not os.path.isfile(source):
        fail(f"{source} does not exist")
    base = os.path.dirname(rel_source)
    for target in quoted_includes(source):
        rel = os.path.normpath(os.path.join(base, target))
        if rel.startswith(".."):
            fail(f"{rel_source} includes \"{target}\", which resolves outside "
                 f"the neutral kernel tree")
        if rel.endswith(KERNEL_SUFFIX):
            walk(kernel_root, rel, seen, emitted)
            emitted.append(("body", rel[:-len(KERNEL_SUFFIX)] +
                            ".kernel.metal"))
        else:
            if not os.path.isfile(os.path.join(kernel_root, rel)):
                fail(f"{rel_source} includes \"{target}\", which resolves to "
                     f"{rel}, and that file does not exist")
            if ("header", rel) not in emitted:
                emitted.append(("header", rel))


def main(argv):
    parser = argparse.ArgumentParser(
        description="Write the translation unit that compiles one generated "
                    "MSL entry rendering.")
    parser.add_argument("--kernel-root", required=True)
    parser.add_argument("--stem", required=True,
                        help="the neutral body's path under the kernel root, "
                             "without the .kernel.cpp suffix")
    parser.add_argument("--prologue", required=True,
                        help="the file msl_prologue.py wrote")
    parser.add_argument("--context", default="",
                        help="neutral headers every unit needs, space "
                             "separated. A SET rather than an order: each one "
                             "includes what it uses, so permuting them cannot "
                             "change what compiles. The caller's recipe says "
                             "why the list is not empty.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv[1:])

    kernel_root = os.path.abspath(args.kernel_root)
    rel_source = args.stem + KERNEL_SUFFIX
    if not os.path.isfile(os.path.join(kernel_root, rel_source)):
        fail(f"{args.stem} names no neutral kernel source under {kernel_root}")

    emitted = []
    walk(kernel_root, rel_source, set(), emitted)
    # The body this entry point wraps, after everything it names. `walk` emits
    # what a file INCLUDES and not the file itself, so that a body reached
    # twice is emitted once; the one it is called with is the exception,
    # because nothing includes it.
    emitted.append(("body", args.stem + ".kernel.metal"))

    lines = [
        f"// Generated by ppf-cts-compute/metal/entry_harness.py for\n"
        f"// {args.stem}{KERNEL_SUFFIX}. Do not edit.\n"
        "//\n"
        "// The translation unit that hands one generated entry rendering to\n"
        "// the Metal shader compiler. Read entry_harness.py for what this\n"
        "// proves and what it does not.\n"
        "\n",
        f'#include "{os.path.basename(args.prologue)}"\n',
        "\n"
        "// The neutral type vocabulary the caller named. Each of these\n"
        "// includes what it uses, so this is a set and not an order:\n"
        "// permuting the lines cannot change what compiles. They are here at\n"
        "// all because they are reached without being named, which the\n"
        "// caller's recipe states one entry at a time.\n",
    ]
    for header in args.context.split():
        lines.append(f'#include "{header}"\n')
    lines.append(
        "\n"
        "// THE ARENA HANDLE AS AN ENTRY POINT SEES IT.\n"
        "//\n"
        "// Four `uint`, which is `BeHandle` from seam/backend_abi.h. That\n"
        "// header cannot be read here: it opens with <stddef.h> and <stdint.h>\n"
        "// and it declares a `BeDiagRecord` that would collide with the\n"
        "// diagnostic channel's own. The layout is not taken on trust for\n"
        "// that: the entry rendering below asserts its record's size and every\n"
        "// field offset, which pins this to 16 bytes, and the field NAMES are\n"
        "// checked by the cu and cpp renderings of the same declaration, which\n"
        "// reach the real type through arena_handle.hpp.\n"
        "struct ArenaHandle {\n"
        "    uint arena;\n"
        "    uint off;\n"
        "    uint size;\n"
        "    uint allocated;\n"
        "};\n"
        "\n"
        "// The neutral body this entry point wraps, and whatever it names,\n"
        "// dependencies first. These are the renderings, never the neutral\n"
        "// sources: the address spaces have to be present as keywords.\n")
    for kind, rel in emitted:
        lines.append(f'#include "{rel}"\n')
    lines.append(
        "\n"
        "// The artifact under test, byte for byte as the generator wrote it.\n"
        f'#include "{args.stem}.entry.metal"\n')

    rendered = "".join(lines)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Written whole and compared first, so a rebuild that changes nothing does
    # not move the mtime and does not force a recompile.
    if os.path.isfile(out):
        with open(out, "r", encoding="utf-8") as f:
            if f.read() == rendered:
                return 0
    with open(out, "w", encoding="utf-8") as f:
        f.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
