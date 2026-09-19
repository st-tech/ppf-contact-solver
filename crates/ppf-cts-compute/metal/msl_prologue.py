# File: msl_prologue.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Writes this backend's MSL prologue to a file, so that a translation unit
# compiled OFFLINE by `xcrun metal` sees exactly the names a shader compiled at
# run time sees.
#
# WHY THIS EXISTS RATHER THAN A SECOND COPY OF THE TEXT. The prologue is the
# seam: the SM_ spellings, the arena binding ABI and the diagnostic channel. A
# check that compiled a generated entry point against its own copy of those
# names would pass while the real prologue drifted, which is the failure mode
# the whole single-source arrangement is built against. So the text is not
# written here; it is READ from the two files that define it, and this script
# fails rather than guessing when it cannot find one.
#
# WHY IT IS EXTRACTION RATHER THAN A CALL INTO THE BACKEND. `shader_create`
# assembles the same three pieces and would need no parsing, but it takes a
# `Context` and a `Diagnostics`, so reaching it means creating a Metal device
# and a device buffer. The check this feeds needs the shader COMPILER and not a
# device, and the two are separately available: a machine can carry the Metal
# toolchain with no GPU this backend would accept. Extraction keeps the check
# runnable there.
#
# WHAT IT DOES NOT REPRODUCE, stated so nobody reads more into the output than
# it carries: the per-segment `#line` and `DIAG_FILE_ID` directives the run-time
# assembler emits as it splices, and the segment order itself. Both are
# properties of an assembled LIBRARY, and the thing this feeds is one entry
# point compiled on its own.

import argparse
import os
import re
import sys


def fail(message):
    sys.stderr.write(f"msl_prologue: {message}\n")
    return 2


def raw_string(path, name):
    """The body of `const char *const <name> = R"MSL( ... )MSL";`.

    Returned without the delimiters and without interpretation, so what lands
    in the output is byte for byte what the backend concatenates.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        raise SystemExit(fail(f"cannot read {path}: {exc}"))
    pattern = (r'const char \*const ' + re.escape(name) +
               r'\s*=\s*R"MSL\((.*?)\)MSL";')
    match = re.search(pattern, text, re.S)
    if not match:
        raise SystemExit(fail(
            f"{path} carries no `const char *const {name} = R\"MSL(...)MSL\";`. "
            f"That constant IS the prologue this script exists to copy, so a "
            f"rename here must be answered here rather than left to a check "
            f"that would then compile against nothing."))
    if re.search(pattern, text[match.end():], re.S):
        raise SystemExit(fail(
            f"{path} carries {name} more than once, so which one the backend "
            f"concatenates is not decidable from the text."))
    return match.group(1)


def diagnostic_binding(path):
    """The buffer index the diagnostic channel is bound at.

    Read rather than assumed: the run-time prologue defines DIAG_BINDING
    from the index the allocator reserved, and every kernel's DIAG_ARG
    expands to a `[[buffer(N)]]` on it. A wrong value here would compile and
    would put the offline check on a different ABI from the shipped one.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        raise SystemExit(fail(f"cannot read {path}: {exc}"))
    arenas = re.search(r"constexpr unsigned kMaxArenas\s*=\s*(\d+)\s*;", text)
    if not arenas:
        raise SystemExit(fail(
            f"{path} declares no `constexpr unsigned kMaxArenas = N;`"))
    # Pinned to the one spelling, because this script evaluates the expression
    # rather than the compiler: a different one would be read wrong instead of
    # being refused.
    dedicated = re.search(
        r"constexpr unsigned kDedicatedBindingIndex\s*=\s*kMaxArenas \+ 1\s*;",
        text)
    if not dedicated:
        raise SystemExit(fail(
            f"{path} no longer spells kDedicatedBindingIndex as "
            f"`kMaxArenas + 1`. This script evaluates that expression itself, "
            f"so a new one has to be taught here rather than silently "
            f"mis-read."))
    return int(arenas.group(1)) + 1


def main(argv):
    parser = argparse.ArgumentParser(
        description="Write this backend's MSL prologue to a file.")
    parser.add_argument("--shader-compiler", required=True,
                        help="metal/shader_compiler.mm, which defines "
                             "kMslMacroSeam")
    parser.add_argument("--diagnostics", required=True,
                        help="metal/diagnostics.mm, which defines DIAG_MSL")
    parser.add_argument("--arena", required=True,
                        help="metal/arena.hpp, which fixes the diagnostic "
                             "channel's binding index")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv[1:])

    seam = raw_string(args.shader_compiler, "kMslMacroSeam")
    diag = raw_string(args.diagnostics, "DIAG_MSL")
    binding = diagnostic_binding(args.arena)

    rendered = (
        "// Generated by ppf-cts-compute/metal/msl_prologue.py. Do not edit.\n"
        "//\n"
        "// This backend's MSL prologue, copied out of the two files that\n"
        "// define it so an offline translation unit sees the same names a\n"
        "// shader compiled at run time sees. The macro seam comes from\n"
        f"// {os.path.basename(args.shader_compiler)} and the diagnostic\n"
        f"// channel from {os.path.basename(args.diagnostics)}.\n"
        "//\n"
        "// <metal_stdlib> is opened here rather than in either source. At run\n"
        "// time the first spliced segment does it (linalg/la_traits.hpp,\n"
        "// under __METAL_VERSION__), and an offline unit has no segments.\n"
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "\n"
        "// ---- kMslMacroSeam ----\n"
        f"{seam}\n"
        "// ---- the diagnostic channel ----\n"
        f"#define DIAG_BINDING {binding}\n"
        f"{diag}\n")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Written whole and compared first, so a rebuild that changes nothing does
    # not move the mtime and does not force every dependent object to
    # recompile. kernelgen.py beside this one does the same.
    if os.path.isfile(out):
        with open(out, "r", encoding="utf-8") as f:
            if f.read() == rendered:
                return 0
    with open(out, "w", encoding="utf-8") as f:
        f.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
