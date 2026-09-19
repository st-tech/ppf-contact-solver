#!/usr/bin/env python3
# File: test_kernelgen.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Tests for ppf-cts-compute/seam/kernelgen.py, and above all for its ENTRY path.

Run it anywhere: `python3 crates/ppf-cts-compute/seam/test_kernelgen.py`.
No CUDA toolkit, no Metal device, no Rust toolchain. It does need a host C++
compiler (`$CXX`, default `g++`), because one section COMPILES and RUNS a
generated shim rather than reading it; that is not a new dependency, since the
CPU backend compiles the generated shims with the same compiler, so a machine
without one cannot build the solver either. Fixtures are written to a
temporary directory the test makes and removes, never under
`crates/ppf-cts-solver/src`, because two build scripts watch that tree
recursively and cargo reads a directory in `rerun-if-changed` as "rerun if any
descendant changes".

WHY THIS FILE EXISTS. `kernelgen.py` was a LEXICAL converter, and a lexical
converter's failure mode is loud: it does not understand a construct, so it
refuses it. The entry path PARSES, and a parser that mis-reads a parameter type
emits a wrong record layout, which on a backend that never faults is a silent
wrong answer rather than a crash. Four lexical traps were already measured in
the converter and each let a forbidden construct through WITH A ZERO EXIT, so
each gets a test HERE, in the entry path, rather than being assumed to still
hold because it holds for bodies:

  * a digit separator (`1'000`) read as a character literal, masking a span of
    real code,
  * an attribute broken across two lines, passing through untranslated,
  * a line comment ending in a backslash, splicing the next line in,
  * a multi-line block comment, whose mask did not carry newlines. That one was
    LOUD and wrong rather than silent, and it is now supported, so its test
    asserts the parse SUCCEEDS and lands on the right line.
"""

import atexit
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELGEN = os.path.join(HERE, "kernelgen.py")

# A minimal stand-in for the caller's kernel tree.
#
# A generated ENTRY point includes one header from that tree, so an entry
# rendering has to be told where it is: the transcompiler lives in
# `ppf-cts-compute` and the kernel tree belongs to whoever calls it, so the two
# are not siblings and no relative path from here can find it. The stub below
# is empty on purpose. What is under test is the RENDERING, and the generator
# only checks that the header exists, so reaching into a real kernel tree would
# couple this file to another crate's layout and test nothing extra.
STUB_ROOT = tempfile.mkdtemp(prefix="kernelgen-stub-root-")
atexit.register(shutil.rmtree, STUB_ROOT, True)
for _stub in ("arena_handle.hpp",):
    with open(os.path.join(STUB_ROOT, _stub), "w", encoding="utf-8") as _f:
        _f.write("#pragma once\n")
KERNEL_ROOT_ARGS = ["--kernel-root", STUB_ROOT]

FAILURES = []
COUNT = 0

# A body every fixture can wrap, so a fixture's declaration is the only thing
# under test.
BODY = """// File: fixture.kernel.cpp
#pragma once

template <class T>
[[seam::host_device_fn]] inline void
fix_scale([[seam::device]] const T *source, [[seam::device]] T *destination,
              T scale, unsigned index) {
    destination[index] = scale * source[index];
}
"""

GOOD_ENTRY = """
[[seam::args]] [[seam::entry]] void fix_scale(
    [[seam::device]] const float *source, [[seam::device]] float *destination,
    float scale, [[seam::count]] unsigned count,
    [[seam::index]] unsigned index);
"""


def run(source_text, target="cu", emit="entry", extra=()):
    """Renders `source_text`, returning (returncode, stdout, stderr, output)."""
    tmp = tempfile.mkdtemp(prefix="kernelgen-test-")
    try:
        src = os.path.join(tmp, "fixture.kernel.cpp")
        out = os.path.join(tmp, "out.txt")
        with open(src, "w", encoding="utf-8") as f:
            f.write(source_text)
        # A TABLE FRAGMENT SPELLS ITS ARGUMENT HEADER FROM THE KERNEL ROOT,
        # so a fixture written outside that root is refused by the generator
        # before anything under test runs. The fixture's own directory is
        # therefore the root for this emit kind, which is exactly what a build
        # passes: the tree the source sits in.
        root_args = (["--kernel-root", tmp] if emit == "table"
                     else KERNEL_ROOT_ARGS)
        proc = subprocess.run(
            [sys.executable, "-B", KERNELGEN, "--target", target,
             "--emit", emit, "--out", out, src]
            + root_args + list(extra),
            capture_output=True, text=True)
        rendered = ""
        if os.path.isfile(out):
            with open(out, encoding="utf-8") as f:
                rendered = f.read()
        return proc.returncode, proc.stdout, proc.stderr, rendered
    finally:
        shutil.rmtree(tmp)


def check(name, condition, detail=""):
    global COUNT
    COUNT += 1
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{(': ' + detail) if detail else ''}")
        FAILURES.append(name)


def rejects(name, source_text, needle, target="cu", emit="entry"):
    """The generator must REFUSE this, with a message naming `needle`."""
    code, _, err, _ = run(source_text, target=target, emit=emit)
    check(name, code == 1 and needle in err,
          f"exit {code}, stderr {err.strip()!r}")


def accepts(name, source_text, target="cu", emit="entry"):
    code, _, err, rendered = run(source_text, target=target, emit=emit)
    check(name, code == 0, f"exit {code}, stderr {err.strip()!r}")
    return rendered


def accepts_with(name, source_text, extra, target="rust", emit="entry"):
    """As `accepts`, with extra command-line arguments."""
    code, _, err, rendered = run(source_text, target=target, emit=emit,
                                 extra=extra)
    check(name, code == 0, f"exit {code}, stderr {err.strip()!r}")
    return rendered


# ---------------------------------------------------------------------------
print("address-space specialization, which MSL forces and CUDA does not")

# A HELPER REACHED WITH TWO ADDRESS SPACES. It needs no template: a plain
# function taking `const float *v`, called once with a buffer and once with a
# stack array, is the whole case. CUDA and the host have one address space and
# compile it as written; MSL has no qualifier meaning "either", so it cannot be
# one function there and no attribute an author could write would make it one.
TWO_SPACES = """
[[seam::device_fn]] inline float sum2(const float *v) {
    return v[0] + v[1];
}

[[seam::device_fn]] inline void use_both(const float *buffer, float *out,
                                         unsigned index) {
    float local[2];
    local[0] = buffer[index];
    local[1] = buffer[index];
    out[index] = sum2(buffer) + sum2(local);
}

[[seam::args]] [[seam::entry]] void use_both(
    const float *buffer, float *out,
    [[seam::count]] unsigned count, [[seam::index]] unsigned index);
"""

_two = accepts("a body reached with two address spaces renders",
               TWO_SPACES, target="metal", emit="body")
check("the definition is specialized per address space",
      "inline float sum2__asd(device const float *v)" in _two, _two)
check("the other signature is appended as its own function",
      "inline float sum2__ast(thread const float *v)" in _two, _two)
check("each call site names the specialization it needs",
      "sum2__asd(buffer) + sum2__ast(local)" in _two, _two)
# CUDA AND THE HOST GET ONE FUNCTION. Every copy would be identical there, so a
# second would be a redefinition, and the address spaces render as nothing.
_two_cu = accepts("the same source renders once for cu", TWO_SPACES,
                  target="cu", emit="body")
check("cu carries no specialization",
      "sum2__as" not in _two_cu and "float sum2(" in _two_cu, _two_cu)

# ---------------------------------------------------------------------------
print("the four measured lexical traps, in the entry path")
# ---------------------------------------------------------------------------

# 1. Digit separator. Read as a character literal it opens a literal running to
#    the next apostrophe, which masks a span of real code, and a masked span is
#    exempt from every check. Refused at the lexer.
rejects(
    "digit separator inside an entry declaration is refused",
    BODY + """
[[seam::args]] [[seam::entry]] void fix_scale(
    [[seam::device]] const float *source, [[seam::device]] float *destination,
    float scale, [[seam::count]] unsigned count = 1'000,
    [[seam::index]] unsigned index);
""",
    "digit separator")

# The same trap in its dangerous form: an EVEN number of separators masks the
# span between them, which is where a forbidden spelling would hide.
rejects(
    "two digit separators masking a span are refused, not silently accepted",
    BODY + """
[[seam::args]] [[seam::entry]] void fix_scale(
    [[seam::device]] const float *source, [[seam::device]] float *destination,
    float scale = 1'0, [[seam::count]] unsigned count = 2'0,
    [[seam::index]] unsigned index);
""",
    "digit separator")

# 2. An attribute broken across two lines. nvcc answers an unrecognized
#    [[seam::device_fn]] with a warning and makes the function __host__, so only
#    a device call site fails, elsewhere. Here it would silently drop the
#    address space and turn a buffer into a scalar.
rejects(
    "an attribute split across two lines is refused",
    BODY + """
[[seam::args]] [[seam::entry]] void fix_scale(
    [[
     seam::device]] const float *source, [[seam::device]] float *destination,
    float scale, [[seam::count]] unsigned count,
    [[seam::index]] unsigned index);
""",
    "not a complete [[seam::...]] attribute")

# 3. A line comment ending in a backslash splices the NEXT line into the
#    comment before comments are recognized, so the script would read as code a
#    line all three compilers read as comment.
rejects(
    "a line comment ending in a backslash is refused",
    BODY + """
[[seam::args]] [[seam::entry]] void fix_scale(
    [[seam::device]] const float *source, // the input \\
    [[seam::device]] float *destination,
    float scale, [[seam::count]] unsigned count,
    [[seam::index]] unsigned index);
""",
    "splices the next line")

# 4. A multi-line block comment. This one was LOUD and WRONG rather than
#    silent, so the test is that it now WORKS and that a diagnostic below it
#    lands on the right line.
rendered = accepts(
    "a multi-line block comment inside a parameter list parses",
    BODY + """
[[seam::args]] [[seam::entry]] void fix_scale(
    [[seam::device]] const float *source, /* the input,
       which the block comment below must not shift off its line */
    [[seam::device]] float *destination,
    float scale, [[seam::count]] unsigned count,
    [[seam::index]] unsigned index);
""")
check("block comment does not leak into the record",
      "must not shift" not in rendered and "destination" in rendered)

code, _, err, _ = run(BODY + """
[[seam::args]] [[seam::entry]] void fix_scale(
    [[seam::device]] const float *source, /* two
       line
       comment */
    [[seam::device]] float *destination,
    double scale, [[seam::count]] unsigned count,
    [[seam::index]] unsigned index);
""")
# The neutral body is 9 lines and the blank is 10, so the declaration opens on
# line 11 and `double scale` sits on line 16.
check("a diagnostic below a multi-line block comment names the right line",
      ":16:" in err, err.strip())

# ---------------------------------------------------------------------------
print("types a record may not hold")
# ---------------------------------------------------------------------------

def with_scalar(spelling):
    return BODY + GOOD_ENTRY.replace("float scale", f"{spelling} scale")

rejects("double is refused, by name", with_scalar("double"),
        "MSL has no double")
rejects("float3 is refused, by name", with_scalar("float3"),
        "vector or matrix type")
rejects("float3x3 is refused, by name", with_scalar("float3x3"),
        "vector or matrix type")
rejects("bool is refused as narrower than 4 bytes", with_scalar("bool"),
        "narrower than 4 bytes")
rejects("char is refused as narrower than 4 bytes", with_scalar("char"),
        "narrower than 4 bytes")
rejects("an unknown type is refused", with_scalar("Mat3x3f"),
        "A record field is one of")
rejects(
    "a reference parameter is refused",
    BODY + GOOD_ENTRY.replace("float scale", "[[seam::thread]] const float &scale"),
    "is a reference")
rejects(
    "a pointer to a pointer is refused",
    BODY + GOOD_ENTRY.replace("const float *source", "const float **source"),
    "pointer to a pointer")
# A RECORD FIELD NAMES AN ALLOCATION, so `device` is the only address space it
# can have and the parser refuses every other spelling (the next case). An
# attribute that can say one thing carries no information, so it is OPTIONAL:
# omitting it is the same declaration, and writing it still reads identically,
# which is what keeps sources predating the default working.
accepts(
    "a pointer with no address space defaults to device",
    BODY + GOOD_ENTRY.replace("[[seam::device]] const float *source",
                              "const float *source"))
rejects(
    "a pointer with TWO address spaces is refused",
    BODY + GOOD_ENTRY.replace("[[seam::device]] const float *source",
                              "[[seam::device]] [[seam::thread]] const float *source"),
    "address space attributes")
rejects(
    "a threadgroup pointer is refused as a record field",
    BODY + GOOD_ENTRY.replace("[[seam::device]] const float *source",
                              "[[seam::threadgroup]] const float *source"),
    "threadgroup scratch is a launch property")
rejects(
    "an address space on a scalar is refused",
    BODY + GOOD_ENTRY.replace("float scale", "[[seam::device]] float scale"),
    "An address space qualifies a pointer")

# ---------------------------------------------------------------------------
print("names MSL takes")
# ---------------------------------------------------------------------------

rejects(
    "a field named `half` is refused by name",
    BODY + GOOD_ENTRY.replace("float scale", "float half"),
    "MSL 16-bit float type")
# `vertex`, `device` and friends are refused for every identifier in a neutral
# body, so the entry path inherits that; a VECTOR type name is the one a field
# plausibly carries and only this check sees it.
rejects(
    "a field named `float4` is refused by name",
    BODY + GOOD_ENTRY.replace("float scale", "float float4"),
    "is a name MSL takes")

# ---------------------------------------------------------------------------
print("the shape of a declaration")
# ---------------------------------------------------------------------------

rejects(
    "an entry with a BODY is refused, which is what keeps logic out of one",
    BODY + """
[[seam::args]] [[seam::entry]] void fix_scale(
    [[seam::device]] const float *source, [[seam::device]] float *destination,
    float scale, [[seam::count]] unsigned count,
    [[seam::index]] unsigned index) {
    if (scale > 0.0f) {
        fix_scale(source, destination, scale, index);
    }
}
""",
    "an entry declaration has no body")
# AN ENTRY IMPLIES ITS RECORD. There is no entry point without one, so writing
# both said one thing twice; `[[seam::args]]` alone still means the other half,
# a record with no launch.
_implied = accepts("[[seam::entry]] alone renders the record too",
                   BODY + GOOD_ENTRY.replace("[[seam::args]] ", ""))
_written = accepts("both attributes render the same thing", BODY + GOOD_ENTRY)


def _without_paths(text):
    """The rendering minus the lines naming the fixture's own temp path."""
    return "\n".join(line for line in text.split("\n")
                     if "kernel.cpp" not in line and ".args.cuh" not in line)


check("the implied record is the one [[seam::args]] would have rendered",
      _without_paths(_implied) == _without_paths(_written),
      _implied)
rejects(
    "a second [[seam::index]] is refused",
    BODY + GOOD_ENTRY.replace("[[seam::count]] unsigned count",
                              "[[seam::index]] unsigned other"),
    "a second [[seam::index]] parameter")
rejects(
    "a second [[seam::count]] is refused",
    BODY + GOOD_ENTRY.replace("float scale", "[[seam::count]] unsigned other"),
    "a second [[seam::count]] parameter")
rejects(
    "an entry whose thread index reaches nothing is refused",
    BODY + GOOD_ENTRY.replace("[[seam::index]] unsigned index",
                              "unsigned index"),
    "thread index reaches nothing")
rejects(
    "an entry naming its thread count neither way is refused",
    BODY + GOOD_ENTRY.replace("[[seam::count]] unsigned count",
                              "unsigned count"),
    "needs a thread count")

# THE COUNT RIDES THE ENTRY, the way DISPATCH_START(count) carries it in the
# reference: naming the parameter on the attribute is the same fact in the same
# place, so it must render exactly what the parameter marker renders.
_ON_ENTRY = GOOD_ENTRY.replace("[[seam::count]] unsigned count",
                               "unsigned count").replace(
                                   "[[seam::entry]]", "[[seam::entry(count)]]")
_named = accepts("[[seam::entry(count)]] names the thread count",
                 BODY + _ON_ENTRY)
check("naming the count on the entry renders what marking it renders",
      _without_paths(_named) == _without_paths(_written), _named)
rejects(
    "[[seam::entry(count)]] beside [[seam::count]] is refused",
    BODY + GOOD_ENTRY.replace("[[seam::entry]]", "[[seam::entry(count)]]"),
    "One of the two says it")
# THE INDEX RIDES THE ENTRY TOO, so one attribute carries both coordinates the
# launch supplies, which is what `DISPATCH_START(count)` plus the dispatch
# lambda's `(unsigned i)` carry between them in the reference.
_BOTH = GOOD_ENTRY.replace("[[seam::count]] unsigned count",
                           "unsigned count").replace(
                               "[[seam::index]] unsigned index",
                               "unsigned index").replace(
                                   "[[seam::entry]]",
                                   "[[seam::entry(count, index)]]")
_pair = accepts("[[seam::entry(count, index)]] names both coordinates",
                BODY + _BOTH)
check("naming both on the entry renders what marking both renders",
      _without_paths(_pair) == _without_paths(_written), _pair)
rejects(
    "an index named on the entry and marked as well is refused",
    BODY + _BOTH.replace("unsigned index)", "[[seam::index]] unsigned index)"),
    "One of the two says it")
rejects(
    "[[seam::entry]] naming an index no parameter carries is refused",
    BODY + _BOTH.replace("[[seam::entry(count, index)]]",
                         "[[seam::entry(count, nosuch)]]"),
    "no parameter of this declaration is called that")
rejects(
    "[[seam::entry]] naming three coordinates is refused",
    BODY + _BOTH.replace("[[seam::entry(count, index)]]",
                         "[[seam::entry(count, index, extra)]]"),
    "then optionally the thread index")
rejects(
    "[[seam::entry]] naming no parameter of the declaration is refused",
    BODY + _ON_ENTRY.replace("[[seam::entry(count)]]",
                             "[[seam::entry(nosuch)]]"),
    "names no parameter of this declaration")
rejects(
    "a non-void entry is refused",
    BODY + GOOD_ENTRY.replace("void fix_scale", "float fix_scale"),
    "returns void")
rejects(
    "an entry taking the generator's reserved prefix is refused",
    BODY + GOOD_ENTRY.replace("void fix_scale", "void seam_fix_scale"),
    "reserved")
rejects(
    "an unterminated declaration is refused",
    BODY + GOOD_ENTRY.replace("unsigned index);", "unsigned index)"),
    "never terminated by ';'")
rejects(
    "two declarations of one entry are refused",
    BODY + GOOD_ENTRY + GOOD_ENTRY,
    "a second entry declaration")
rejects(
    "[[seam::count]] outside a declaration is refused",
    BODY + """
[[seam::device_fn]] inline void stray([[seam::count]] unsigned n) {
    (void)n;
}
""",
    "outside an entry declaration")
rejects(
    "a record over the portable size cap is refused",
    BODY + "\n[[seam::args]] [[seam::entry]] void fix_scale(\n" +
    "".join(f"    [[seam::device]] const float *b{i},\n" for i in range(256)) +
    "    [[seam::count]] unsigned count, [[seam::index]] unsigned index);\n",
    "over the 4096 byte portable cap")

# ---------------------------------------------------------------------------
print("the gather decision is read off the body, not written twice")
# ---------------------------------------------------------------------------

# A body taking one element BY REFERENCE, beside one taking the array by
# pointer. The declaration names neither access, so both are inferred.
ELEMENT_BODY = """// File: fixture.kernel.cpp
#pragma once

[[seam::device_fn]] inline void
scale_one(const float &source, const float *table,
          float *destination, unsigned index) {
    destination[index] = source * table[0];
}
"""
ELEMENT_ENTRY = """
[[seam::args]] [[seam::entry]] void scale_one(
    const float *source, const float *table,
    float *destination,
    [[seam::count]] unsigned count, [[seam::index]] unsigned index);
"""

inferred = accepts("a body's reference parameter renders a gather",
                   ELEMENT_BODY + ELEMENT_ENTRY, target="metal")
# The gathered element is copied into thread space before the call and the
# array is passed as a device pointer, which is the whole of the distinction.
check("the reference parameter is gathered into thread space",
      "const float seam_local_source = source[index];" in inferred, inferred)
check("the pointer parameter stays a device pointer",
      "device const float *table" in inferred
      and "seam_local_table" not in inferred, inferred)

rejects(
    "[[seam::gather]] on a parameter the body takes by pointer is refused",
    ELEMENT_BODY + ELEMENT_ENTRY.replace(
        "const float *table", "[[seam::gather]] const float *table"),
    "takes it by pointer, which is the other shape")
rejects(
    "a body whose parameters do not line up is refused",
    ELEMENT_BODY + ELEMENT_ENTRY.replace(
        "    float *destination,\n", "    float *destination, const float *extra,\n"),
    "do not line up with its body")

# ---------------------------------------------------------------------------
print("the sink and the diagnostic channel are read off the body too")
# ---------------------------------------------------------------------------

# A body that RETURNS a value has somewhere to put it, and the destination is
# the one buffer the declaration names that the body does not take.
SINK_BODY = """// File: fixture.kernel.cpp
#pragma once

[[seam::device_fn]] inline float
pick_one(const float &source, unsigned index) {
    return source * (float)index;
}
"""
SINK_ENTRY = """
[[seam::entry(count, index)]] void pick_one(
    const float *source, float *destination,
    unsigned count, unsigned index);
"""
_sunk = accepts("the destination is inferred from the body's return type",
                SINK_BODY + SINK_ENTRY, target="metal")
check("the inferred sink is written, not passed",
      "destination[index] =" in _sunk, _sunk)
rejects(
    "a returning body with nowhere to put its value is refused",
    SINK_BODY + SINK_ENTRY.replace("    const float *source, float *destination,\n",
                                   "    const float *source,\n"),
    "nowhere to put the value")
# TWO CANDIDATES OF THE SAME ARITY is the shape this refuses rather than
# guesses at: the body takes three parameters and the declaration names four
# forwarded ones, so dropping EITHER non-const buffer lines them up.
TWO_BODY = """// File: fixture.kernel.cpp
#pragma once

[[seam::device_fn]] inline float
pick_two(const float &source, const float &second, unsigned index) {
    return source + second * (float)index;
}
"""
rejects(
    "two buffers that would both fit as the destination are refused",
    TWO_BODY + """
[[seam::entry(count, index)]] void pick_two(
    const float *source, float *destination, float *other,
    unsigned count, unsigned index);
""",
    "which one takes the value cannot be read off the body")

# `DiagHandle` is the one type name the three prologues agree on and it names
# nothing else, so the type IS the declaration.
DIAG_BODY = """// File: fixture.kernel.cpp
#pragma once

[[seam::device_fn]] inline void
guarded(const float &source, float *destination, DiagHandle diag,
        unsigned index) {
    DIAG_ASSERT4(diag, source == source, 0u, index, 0u, 0u);
    destination[index] = source;
}
"""
DIAG_ENTRY = """
[[seam::entry(count, index)]] void guarded(
    const float *source, float *destination, DiagHandle diag,
    unsigned count, unsigned index);
"""
_diag = accepts("a DiagHandle parameter is the diagnostic channel",
                DIAG_BODY + DIAG_ENTRY)
check("the inferred channel reaches the entry point",
      "diag" in _diag, _diag)
rejects(
    "[[seam::diag]] on a parameter that is not a DiagHandle is refused",
    DIAG_BODY + DIAG_ENTRY.replace("DiagHandle diag,",
                                   "[[seam::diag]] float diag,"),
    "The diagnostic channel is 'DiagHandle'")

# ---------------------------------------------------------------------------
print("a body may declare itself an entry, the way a dispatch lambda does")
# ---------------------------------------------------------------------------

# THE REFERENCE WRITES ONE CONSTRUCT PER DISPATCH, `DISPATCH_START(n)` and a
# lambda. A body carrying `[[seam::entry(...)]]` on the line above its signature
# is that shape: the record is synthesized from the body's own parameters, and
# the thread count is implicit because it is not a parameter of the body at all.
SELF_DECLARING = """// File: fixture.kernel.cpp
#pragma once

[[seam::entry(index)]]
[[seam::device_fn]] inline void
scale_one(const float &source, float *destination, unsigned index) {
    destination[index] = source * 2.0f;
}
"""
# The same kernel written the long way, as a body and a separate declaration.
WRITTEN_OUT = """// File: fixture.kernel.cpp
#pragma once

[[seam::device_fn]] inline void
scale_one(const float &source, float *destination, unsigned index) {
    destination[index] = source * 2.0f;
}

[[seam::entry(count, index)]] void scale_one(
    const float *source,
    float *destination,
    unsigned index,
    unsigned count);
"""
for _target in ("cu", "metal", "rust"):
    _self = accepts(f"{_target}: a self-declaring body renders an entry",
                    SELF_DECLARING, target=_target)
    _long = accepts(f"{_target}: the written-out form renders one too",
                    WRITTEN_OUT, target=_target)
    check(f"{_target}: both forms render the same entry",
          _without_paths(_self) == _without_paths(_long), _self)

rejects(
    "the entry attribute on the signature itself is refused",
    SELF_DECLARING.replace("[[seam::entry(index)]]\n[[seam::device_fn]]",
                           "[[seam::entry(index)]] [[seam::device_fn]]"),
    "carries the entry attribute on its own signature")
rejects(
    "a void body naming a destination is refused",
    SELF_DECLARING.replace("[[seam::entry(index)]]", "[[seam::entry(nosuch)]]"),
    "returns void, so there is no value to put there")

# A NAME THE BODY TAKES IS THE INDEX; ONE IT DOES NOT TAKE IS THE DESTINATION.
# The destination is a record field the body has no parameter for, which is what
# the sink inference looks for once it is declared, so naming it is all that is
# needed and no new syntax is.
RETURNING = """// File: fixture.kernel.cpp
#pragma once

[[seam::entry(index, out)]]
[[seam::device_fn]] inline float
pick_one(const float &source, unsigned index) {
    return source * (float)index;
}
"""
_ret = accepts("a self-declaring body names where its value goes",
               RETURNING, target="metal")
check("the named destination is written at the thread index",
      "out[index] = pick_one(" in _ret, _ret)
rejects(
    "a returning body naming no destination is refused",
    RETURNING.replace("[[seam::entry(index, out)]]", "[[seam::entry(index)]]"),
    "names no destination for it")
rejects(
    "two destinations are refused",
    RETURNING.replace("[[seam::entry(index, out)]]",
                      "[[seam::entry(index, out, other)]]"),
    "One call returns one value")

# ---------------------------------------------------------------------------
print("what a good declaration renders to")
# ---------------------------------------------------------------------------

good = BODY + GOOD_ENTRY
sizes = {}
for target in ("cu", "metal", "cpp", "rust"):
    text = accepts(f"{target} entry renders", good, target=target)
    sizes[target] = text
    check(f"{target} carries the entry symbol",
          "fix_scale_entry" in text)

# THE LAYOUT ASSERTIONS ARE READ FROM WHICHEVER HALF HOLDS THEM. On `cu` the
# record and its assertions live in the `--emit args` rendering the entry
# rendering includes, because a caller in a second translation unit needs the
# type without the two definitions beside it. The other three targets have one
# consumer each and are not split. What is under test is unchanged either way:
# all four languages assert the SAME computed layout.
sizes["cu"] += accepts("cu args renders", good, emit="args")

# Two handles at 16 plus scale, count and the generator-owned arena count at 4
# is 44 bytes, and every rendering has to agree on it.
check("cu asserts sizeof 44", "sizeof(FixScaleArgs) == 44" in sizes["cu"])
check("metal asserts sizeof 44",
      "sizeof(FixScaleArgs) == 44" in sizes["metal"])
check("cpp asserts sizeof 44", "sizeof(FixScaleArgs) == 44" in sizes["cpp"])
check("rust asserts size_of 44",
      "size_of::<FixScaleArgs>() == 44" in sizes["rust"])

for name, offset in (("source", 0), ("destination", 16), ("scale", 32),
                     ("count", 36), ("seam_arena_count", 40)):
    per_target = [
        f"offsetof(FixScaleArgs, {name}) == {offset}" in sizes["cu"],
        f"__builtin_offsetof(FixScaleArgs, {name}) == {offset}"
        in sizes["metal"],
        f"offsetof(FixScaleArgs, {name}) == {offset}" in sizes["cpp"],
        f"offset_of!(FixScaleArgs, {name}) == {offset}" in sizes["rust"],
    ]
    check(f"all four assert {name} at {offset}", all(per_target),
          str(per_target))

for target, spelling in (("cu", "alignof(FixScaleArgs) == 4"),
                         ("metal", "alignof(FixScaleArgs) == 4"),
                         ("cpp", "alignof(FixScaleArgs) == 4"),
                         ("rust", "align_of::<FixScaleArgs>() == 4")):
    check(f"{target} asserts the record's alignment",
          spelling in sizes[target])

check("the guard tests the [[seam::count]] field",
      "if (index >= args.count)" in sizes["cu"] and
      "if (index >= args.count)" in sizes["metal"])
check("[[seam::count]] is not forwarded to the body",
      "args.count,\n" not in sizes["cu"])
# The declaration puts the index last, so the generated call has to. Read
# against the end of the __global__ rather than the end of the file: the file
# closes with the launcher that dispatches this entry point, which carries no
# call into the body at all.
check("the thread index is forwarded in its declared position",
      sizes["cu"].split('extern "C"')[0].rstrip().endswith("index);\n}"))
check("cu resolves a const pointer through resolve_const",
      "compute::arena::resolve_const<float>(args.source)" in sizes["cu"])
check("cu resolves a mutable pointer through resolve",
      "compute::arena::resolve<float>(args.destination)" in sizes["cu"])
check("metal checks every arena id against the live count",
      sizes["metal"].count("< args.seam_arena_count") == 2)
check("the cpu shim is a range, not a single element",
      "unsigned begin, unsigned end" in sizes["cpp"])
check("the rust twin declares handle offsets",
      "FIX_SCALE_HANDLE_OFFSETS: &[u16] = &[0, 16];" in sizes["rust"])

check("the rust twin names the driver's id rather than a literal",
      "const KERNEL: KernelId = id::FIX_SCALE;" in sizes["rust"] and
      "KernelId(" not in sizes["rust"].split("unsafe impl")[1])
check("--kernel-id is gone, so a number cannot be spelled twice",
      run(good, target="rust", emit="entry",
          extra=["--kernel-id", "7"])[0] == 2)

# ---------------------------------------------------------------------------
print("a record migrates FIELD BY FIELD from an address to a handle")
# ---------------------------------------------------------------------------
# The driver's buffers become device allocations one at a time, and a record
# names several. So the Rust twin has to be able to spell one field `Handle`
# while its neighbor is still `HostRef`, and the declaration has to say WHICH,
# because that is the list a target walks to bind the remaining addresses.

check("the rust twin lists both buffers as still addressed",
      "FIX_SCALE_HOST_REF_OFFSETS: &[u16] = &[0, 16];" in sizes["rust"] and
      "pub source: HostRef," in sizes["rust"] and
      "pub destination: HostRef," in sizes["rust"])

half = accepts_with("one field migrates and its neighbor does not", good,
                    ["--handle-field", "FixScaleArgs.destination"])
check("the migrated field is a handle and the other is not",
      "pub destination: Handle," in half and
      "pub source: HostRef," in half)
check("the full buffer list is unchanged by a migration",
      "FIX_SCALE_HANDLE_OFFSETS: &[u16] = &[0, 16];" in half)
check("the addressed list drops exactly the migrated field",
      "FIX_SCALE_HOST_REF_OFFSETS: &[u16] = &[0];" in half)
check("a fully migrated record names no address at all",
      "FIX_SCALE_HOST_REF_OFFSETS: &[u16] = &[];" in
      accepts_with("both fields migrate", good,
                   ["--handle-field", "FixScaleArgs.source",
                    "--handle-field", "FixScaleArgs.destination"]))

# THE MISS IS SILENT WITHOUT THIS. A misspelled field simply does not migrate,
# the record keeps an address in it, and the only symptom is that the buffer
# count the caller thinks it moved never falls.
code, _, err, _ = run(good, target="rust", emit="entry",
                      extra=["--handle-field", "FixScaleArgs.destinaton"])
check("a field name no declaration carries is refused",
      code == 1 and "names no buffer field" in err and
      "FixScaleArgs.destination" in err,
      f"exit {code}, stderr {err.strip()!r}")
code, _, err, _ = run(good, target="rust", emit="entry",
                      extra=["--handle-field", "FixScaleArgs.scale"])
check("a SCALAR field is refused, not silently ignored",
      code == 1 and "names no buffer field" in err,
      f"exit {code}, stderr {err.strip()!r}")

# Every C++ rendering spells a buffer field as an ArenaHandle whatever the
# caller puts in it, so there is nothing for the flag to change there and
# accepting it would suggest otherwise.
for target in ("cu", "metal", "cpp"):
    code, _, err, _ = run(good, target=target, emit="entry",
                          extra=["--handle-field", "FixScaleArgs.destination"])
    check(f"--handle-field is refused for --target {target}",
          code == 2 and "applies only to --target rust" in err,
          f"exit {code}, stderr {err.strip()!r}")
code, _, err, _ = run(good, target="rust", emit="table",
                      extra=["--handle-field", "FixScaleArgs.destination"])
check("--handle-field is refused for --emit table",
      code == 2 and "applies only to --target rust" in err,
      f"exit {code}, stderr {err.strip()!r}")

# ---------------------------------------------------------------------------
print("a template argument list in a parameter type is refused by its own name")
# ---------------------------------------------------------------------------
# The parameter split is a plain scan for top-level commas, justified on the
# ground that no parameter may contain a parenthesis. Angle brackets get no such
# treatment, so `SMatf<3, 6> *g` arrives as two pieces: one with no name and one
# named `g` whose type is `6>`. Refusing it as an unnamed parameter is true of
# the piece the split invented and false of what was written, and it sends a
# reader to look for a missing identifier. The remedy is the alias `data.hpp`
# declares for that size, so the message names it.
rejects(
    "a templated pointee names the template rather than a missing name",
    BODY + """
[[seam::args]] [[seam::entry]] void fix_scale(
    [[seam::device]] const float *source, [[seam::device]] float *destination,
    [[seam::device]] [[seam::pod(72)]] const SMatf<3, 6> *gradient,
    float scale, [[seam::count]] unsigned count,
    [[seam::index]] unsigned index);
""",
    "a template argument list in a parameter type")

# ---------------------------------------------------------------------------
print("a body that returns a value cannot be silently discarded")
# ---------------------------------------------------------------------------

# THE HAZARD THIS COVERS WAS MEASURED, not imagined: before this, an entry
# declared `void` over a body that RETURNS its result rendered on all four
# targets with exit 0, and the generated call discarded the value. The launcher
# that used to store it is exactly what a generated entry replaces, so the
# kernel would have run and written nothing. The generator cannot see the
# body's return type without parsing another file, so every C++ rendering makes
# the call through `return`, which is well-formed in a void function only when
# the expression is itself void.
for target, needle in (("cu", "return fix_scale("),
                       ("metal", "return fix_scale("),
                       ("cpp", "return fix_scale(")):
    check(f"{target} calls the body through return",
          needle in accepts(f"{target} entry renders", good, target=target))
check("the cpp range calls through a void-returning forwarder",
      "-> void {" in sizes["cpp"])

# ---------------------------------------------------------------------------
print("the host-callable launcher, which only the cu target renders")
# ---------------------------------------------------------------------------

# A __global__ is a C++ symbol with a launch syntax rather than a function
# whose address can be taken, so an entry point rendered as the __global__
# alone is reachable from a translation unit nvcc compiles and from nowhere
# else: a name-to-launcher table cannot hold one and a caller above the seam
# cannot call one. The launcher is the other half, and it is generated for the
# reason the __global__ is. Written by hand it fills a generated record field
# by field, which is a second statement of the record's contents with nothing
# linking it to the first: a field added to the declaration is absorbed by
# position or left unset by name, and neither is a compile error.
# TWICE over the two halves, once each: the args rendering DECLARES it, for a
# caller in another translation unit, and the entry rendering DEFINES it. Both
# come from one parse, which is what stops a hand-written declaration from
# drifting from the definition it names.
check("cu renders one launcher per entry point",
      sizes["cu"].count('extern "C" void fix_scale_entry_launch(') == 2)
check("the launcher takes the record as opaque bytes and a queue",
      "    const void *record, cudaStream_t queue) {" in sizes["cu"])
check("the launcher dispatches the __global__ beside it",
      "fix_scale_entry<<<blocks, block_size, 0, queue>>>(args);"
      in sizes["cu"])
check("the launch is checked",
      "CUDA_HANDLE_ERROR(cudaGetLastError());" in sizes["cu"])

# THE RECORD IS COPIED IN, and each half of the reason is checked because each
# stands alone: the caller's bytes carry no alignment the rendered file may
# assume, and the arena count is the library's fact rather than the caller's.
check("the launcher copies the record in rather than reading it in place",
      "std::memcpy(&args, record, sizeof(args));" in sizes["cu"] and
      "#include <cstring>" in sizes["cu"])
check("the launcher fills the arena count from the live allocator",
      "args.seam_arena_count = "
      "compute::arena::arena_count(compute::arena::active());" in sizes["cu"])

# THE GRID READS THE FIELD THE GUARD READS. A second number passed beside the
# record could disagree with it, and one direction of that disagreement is
# silent: a grid narrower than the count leaves elements unprocessed with
# nothing to see.
check("the grid is sized from the record's own count field",
      "choose_block_size(args.count)" in sizes["cu"] and
      "(args.count + block_size - 1) / block_size" in sizes["cu"])
check("an empty extent launches nothing",
      "if (args.count == 0) {" in sizes["cu"])

# The two headers the launcher needs are named without a path, so the CUDA
# build's include roots supply them rather than this generator, which is the
# same treatment the allocator already gets.
check("the launcher names its two headers without a path",
      '#include "common.hpp"' in sizes["cu"] and
      '#include "cuda_utils.hpp"' in sizes["cu"])

# A RECORD WITHOUT A HANDLE HAS NO ARENA COUNT TO FILL, and the launcher must
# not name a field the record does not carry. No kernel in the tree is shaped
# this way today, so the case exists here or nowhere.
scalar_only = """// File: fixture.kernel.cpp
#pragma once

[[seam::device_fn]] inline void fix_tick(float scale, unsigned index) {
    (void)scale;
    (void)index;
}

[[seam::args]] [[seam::entry]] void fix_tick(
    float scale, [[seam::count]] unsigned count,
    [[seam::index]] unsigned index);
"""
text = accepts("an entry over a handle-free record renders", scalar_only)
check("a handle-free record carries no arena count",
      "unsigned seam_arena_count;" not in text)
check("its launcher is rendered and fills no arena count",
      'extern "C" void fix_tick_entry_launch(' in text and
      "args.seam_arena_count" not in text)

# The launcher belongs to the one target whose entry point cannot be called
# directly. The host target's shim IS a callable function and takes a thread
# range instead, and a Metal entry point is reached through a pipeline state
# object looked up by name, so neither has a launcher to render.
for target in ("metal", "cpp", "rust"):
    check(f"{target} renders no launcher",
          "_entry_launch" not in sizes[target])

# ---------------------------------------------------------------------------
print("a struct pointee, and the size it has to declare")
# ---------------------------------------------------------------------------

# A record field is a 16-byte handle whatever it addresses, so a struct pointee
# moves no layout. What it DOES bring is a type whose size this script cannot
# compute from one file, and whose MSL mirror is hand-written today: the
# declaration therefore states the size and all three C++ renderings assert the
# same literal, so a type the Metal shader compiler lays out differently fails
# to compile in the shader rather than reading the wrong bytes on a backend
# that never faults.
POD_BODY = """// File: fixture.kernel.cpp
#pragma once

[[seam::device_fn]] inline void
fix_scale([[seam::device]] const Vec3f *source,
              [[seam::device]] float *destination, unsigned index) {
    destination[index] = 0.0f;
}
"""

POD_ENTRY = """
[[seam::args]] [[seam::entry]] void fix_scale(
    [[seam::device]] [[seam::pod(12)]] const Vec3f *source,
    [[seam::device]] float *destination,
    [[seam::count]] unsigned count, [[seam::index]] unsigned index);
"""

pod = {}
for target in ("cu", "metal", "cpp", "rust"):
    pod[target] = accepts(f"a struct pointee renders for {target}",
                          POD_BODY + POD_ENTRY, target=target)
# On `cu` the size assertion travels with the RECORD, into the args half: it is
# what a caller filling that record needs to have checked, since the handle it
# writes addresses those same bytes.
pod["cu"] += accepts("a struct pointee renders for the cu args half",
                     POD_BODY + POD_ENTRY, emit="args")
for target in ("cu", "metal", "cpp"):
    check(f"{target} asserts the declared pointee size",
          "static_assert(sizeof(Vec3f) == 12," in pod[target])
check("cu resolves a struct pointee as that struct",
      "compute::arena::resolve_const<Vec3f>(args.source)" in pod["cu"])
check("metal resolves a struct pointee as that struct",
      "device const Vec3f *source = (device const Vec3f *)" in pod["metal"])
check("cpp resolves a struct pointee as that struct",
      "const Vec3f *source = reinterpret_cast<const Vec3f *>" in pod["cpp"])
check("a struct pointee is still a 16-byte handle field",
      "size_of::<FixScaleArgs>() == 40" in pod["rust"] and
      "FIX_SCALE_HANDLE_OFFSETS: &[u16] = &[0, 16];" in pod["rust"] and
      "pub source: HostRef," in pod["rust"])

rejects("a struct pointee with no declared size is refused",
        POD_BODY + POD_ENTRY.replace("[[seam::pod(12)]] ", ""),
        "[[seam::pod(N)]]")
rejects("[[seam::pod]] with no argument is refused",
        POD_BODY + POD_ENTRY.replace("[[seam::pod(12)]]", "[[seam::pod]]"),
        "needs its argument")
rejects("[[seam::pod]] on a scalar pointee is refused",
        POD_BODY + POD_ENTRY.replace(
            "[[seam::device]] float *destination",
            "[[seam::device]] [[seam::pod(4)]] float *destination"),
        "already knows")
rejects("[[seam::pod]] on a scalar is refused",
        POD_BODY + POD_ENTRY.replace(
            "[[seam::count]] unsigned count",
            "[[seam::pod(4)]] [[seam::count]] unsigned count"),
        "states the size of a POINTEE")
rejects("a pointee size that is not a multiple of 4 is refused",
        POD_BODY + POD_ENTRY.replace("pod(12)", "pod(13)"),
        "multiple of 4 bytes")
rejects("a zero pointee size is refused",
        POD_BODY + POD_ENTRY.replace("pod(12)", "pod(0)"),
        "multiple of 4 bytes")
rejects("a non-numeric pointee size is refused",
        POD_BODY + POD_ENTRY.replace("pod(12)", "pod(twelve)"),
        "is not a byte count")
# A size written as an expression is refused too, and by the lexical half
# rather than the parser: the attribute grammar holds no nested parenthesis, so
# `pod(sizeof(Vec3f))` is not a complete attribute at all. That refusal is the
# right one either way, since the point of the number is to be a literal all
# three renderings assert.
rejects("a pointee size written as an expression is refused",
        POD_BODY + POD_ENTRY.replace("pod(12)", "pod(sizeof(Vec3f))"),
        "not a complete [[seam::...]] attribute")
rejects("a qualified pointee name is refused",
        POD_BODY + POD_ENTRY.replace("const Vec3f *source",
                                     "const linalg::Vec3f *source"),
        "plain identifier")
rejects("an MSL type name as a pointee is refused",
        POD_BODY + POD_ENTRY.replace("const Vec3f *source",
                                     "const float3 *source"),
        "MSL")
rejects("one type with two declared sizes in one file is refused",
        POD_BODY + POD_ENTRY + POD_ENTRY.replace(
            "fix_scale", "fix_scale2").replace("pod(12)", "pod(16)"),
        "One type, one size")
# The digit separator, inside the one attribute that takes an argument. The
# lexical mask runs before the parse, so the refusal comes from the same place
# it comes from anywhere else in the file.
rejects("a repeated attribute on one parameter is refused",
        POD_BODY + POD_ENTRY.replace("[[seam::pod(12)]]",
                                     "[[seam::pod(12)]] [[seam::pod(16)]]"),
        "appears twice")
rejects("a digit separator inside the pod argument is refused",
        POD_BODY + POD_ENTRY.replace("pod(12)", "pod(1'2)"),
        "digit separator")

# ---------------------------------------------------------------------------
print("[[seam::args]] on its own, and the body rendering")
# ---------------------------------------------------------------------------

record_only = BODY + GOOD_ENTRY.replace("[[seam::args]] [[seam::entry]]",
                                        "[[seam::args]]")
text = accepts("[[seam::args]] alone renders the record", record_only,
               emit="args")
check("[[seam::args]] alone emits no entry point",
      "FixScaleArgs" in text and "fix_scale_entry" not in text)
# And the entry half of the same declaration is empty of both, which is what
# "a record shared by an entry and a host-side filler" means once the record
# has a file of its own: there is no entry point to render.
entry_half = accepts("[[seam::args]] alone renders an empty entry half",
                     record_only)
check("that entry half carries neither record nor entry point",
      "struct FixScaleArgs" not in entry_half
      and "fix_scale_entry" not in entry_half)

for target in ("cu", "metal", "cpp"):
    body = accepts(f"{target} body rendering still works", good,
                   target=target, emit="body")
    neutral_lines = good.count("\n")
    rendered_lines = body.count("\n") - (0 if target == "metal" else 2)
    check(f"{target} body rendering preserves the line count",
          rendered_lines == neutral_lines,
          f"{rendered_lines} against {neutral_lines}")
    check(f"{target} body rendering neutralizes the declaration",
          "[kernelgen] entry declaration: fix_scale" in body and
          "seam::args" not in body)

code, _, err, _ = run(good, target="rust", emit="body")
check("--target rust --emit body is refused",
      code == 2 and "no Rust rendering of a kernel BODY" in err)

# Rendered from ONE source path twice, because the banner carries the absolute
# path of the neutral file and two temporary directories would differ there for
# a reason that is not the generator's.
_tmp = tempfile.mkdtemp(prefix="kernelgen-test-")
try:
    _src = os.path.join(_tmp, "fixture.kernel.cpp")
    with open(_src, "w", encoding="utf-8") as _f:
        _f.write(good)
    _renders = []
    for _i in range(2):
        _out = os.path.join(_tmp, f"out{_i}.cu")
        subprocess.run([sys.executable, "-B", KERNELGEN, "--target", "cu",
                        "--emit", "entry", "--out", _out, _src]
                       + KERNEL_ROOT_ARGS, check=True)
        with open(_out, encoding="utf-8") as _f:
            _renders.append(_f.read())
    check("rendering is deterministic",
          _renders[0] == _renders[1] and _renders[0] != "")
finally:
    shutil.rmtree(_tmp)

# A kernel with no entry declaration still produces a file, so a build rule can
# run over every neutral kernel without knowing which carry one.
text = accepts("a kernel with no entry declaration still renders", BODY)
check("a kernel with no entry declaration says so",
      "declares no [[seam::args]] entry" in text)

# ---------------------------------------------------------------------------
print("the three element shapes")
# ---------------------------------------------------------------------------

# A generated entry used to pass every buffer as a BASE POINTER, which serves a
# body that does its own addressing and no other. Most neutral bodies in this
# tree take an ELEMENT or return one, so the per-element addressing lived in the
# hand-written launcher, which is the text a generated entry exists to remove.
# These three attributes name that addressing on the parameter it applies to.

ELEMENT_BODY = """// File: fixture.kernel.cpp
#pragma once

[[seam::device_fn]] inline Vec3f
fix_step([[seam::thread]] const Vec3f &position,
             [[seam::device]] const float *direction, unsigned vert,
             float scale) {
    return position;
}
"""

ELEMENT_ENTRY = """
[[seam::args]] [[seam::entry]] void fix_step(
    [[seam::device]] [[seam::gather]] [[seam::scatter]] [[seam::pod(12)]]
    Vec3f *eval_x,
    [[seam::device]] const float *direction, [[seam::index]] unsigned vert,
    float scale, [[seam::count]] unsigned count);
"""

# THE GENERATED STATEMENT IS THE LAUNCHER'S, CHARACTER FOR CHARACTER, which is
# the whole acceptance test for this form: fp32 is not associative and this
# codebase calibrates its guards against measured round-off, so a generated
# entry that reassociates or reorders anything is a wrong answer rather than a
# style difference.
element = ELEMENT_BODY + ELEMENT_ENTRY
for target, expected in (
        ("cu", "eval_x[vert] = fix_step(\n        eval_x[vert],"),
        ("cpp", "eval_x[vert] = fix_step(\n            eval_x[vert],")):
    text = accepts(f"{target} renders the in-place element update",
                   element, target=target)
    check(f"{target} gathers and scatters one element of one buffer",
          expected in text, text)

# METAL IS THE ONE TARGET THAT COPIES FIRST, and the reason is an address space
# rather than a preference. A resolved buffer is a `device` pointer there, so
# `eval_x[vert]` is a `device` lvalue and a body parameter written
# `[[seam::thread]] const Vec3f &` renders as `thread const Vec3f &`; MSL will
# not bind the one to the other and answers "cannot bind reference in address
# space 'device' to object in default address space", naming the BODY's line.
# Measured on `xcrun metal` 32023.864: without the copy the shader does not
# compile, with it the same declaration builds a .air. The other two targets
# have one address space and are unchanged, which the pair above pins.
element_metal = accepts("metal renders the in-place element update",
                        element, target="metal")
check("metal copies the gathered element into thread space first",
      "const Vec3f seam_local_eval_x = eval_x[vert];" in element_metal,
      element_metal)
check("metal passes the copy and still writes the buffer at the index",
      "eval_x[vert] = fix_step(\n        seam_local_eval_x," in element_metal,
      element_metal)
# THE COPY IS TAKEN ONCE, before the statement that overwrites the element it
# came from. An in-place update reads and writes one slot, so a copy taken
# after the write would hand the body the value it had just produced.
check("metal takes the copy before the write it feeds",
      element_metal.index("const Vec3f seam_local_eval_x")
      < element_metal.index("eval_x[vert] = fix_step"),
      element_metal)
# ONE RESOLUTION, NAMED, rather than two resolution expressions in one
# statement. The gather and the scatter name the same buffer, so an entry built
# out of resolution expressions would resolve it twice per element, and on CUDA
# each resolution carries the allocator's live asserts.
element_cu = accepts("cu renders the in-place update again", element)
check("the gathered and scattered buffer is resolved once",
      element_cu.count("compute::arena::resolve<Vec3f>(args.eval_x)") == 1,
      element_cu)

STRIDE_BODY = """// File: fixture.kernel.cpp
#pragma once

[[seam::device_fn]] inline Vec3f
fix_mul([[seam::device]] const float *matrix,
            [[seam::device]] const float *vector) {
    return Vec3f();
}
"""

STRIDE_ENTRY = """
[[seam::args]] [[seam::entry]] void fix_mul(
    [[seam::device]] [[seam::stride(9)]] const float *matrix,
    [[seam::device]] [[seam::stride(3)]] const float *vector,
    [[seam::device]] [[seam::scatter]] [[seam::pod(12)]] Vec3f *result,
    [[seam::count]] unsigned count);
"""

strided = STRIDE_BODY + STRIDE_ENTRY
for target in ("cu", "metal", "cpp"):
    text = accepts(f"{target} renders a strided base plus offset", strided,
                   target=target)
    check(f"{target} advances each base by its own stride",
          "matrix + 9 * seam_index" in text and "vector + 3 * seam_index" in text,
          text)
    check(f"{target} writes the return value at the index",
          "result[seam_index] = fix_mul(" in text, text)
# A body taking no thread index declares no [[seam::index]], so the generator
# names the index it guards on. That name is why no parameter may carry the
# `seam_` prefix.
check("an undeclared thread index takes the generator's own name",
      "unsigned seam_index = blockIdx.x" in accepts(
          "cu renders the generated index", strided))

# The Rust twin gains no field and no bytes from any of this: the driver fills a
# handle whatever shape the entry reaches it in.
rust = accepts("rust renders the strided record", strided, target="rust")
check("the strided record is three handles, a count and the arena count",
      rust.count("pub matrix: HostRef") == 1 and
      "core::mem::size_of::<FixMulArgs>() == 56" in rust, rust)

# ---------------------------------------------------------------------------
print("shapes the form does not express, refused by name")
# ---------------------------------------------------------------------------

# THE TEST OF A SMALL FORM IS WHAT IT REFUSES. A launcher fitting none of the
# three shapes needs the BODY changed, which changes its CUDA and Metal call
# sites in the same edit, so it is a deliberate change and not a generator
# feature. Each refusal below names the shapes that do exist.
rejects(
    "[[seam::gather]] with [[seam::stride]] is refused",
    STRIDE_BODY + STRIDE_ENTRY.replace(
        "[[seam::stride(9)]] const float *matrix",
        "[[seam::stride(9)]] [[seam::gather]] const float *matrix"),
    "reaches its data in one of five shapes")
rejects(
    "a const [[seam::scatter]] is refused",
    STRIDE_BODY + STRIDE_ENTRY.replace(
        "[[seam::scatter]] [[seam::pod(12)]] Vec3f *result",
        "[[seam::scatter]] [[seam::pod(12)]] const Vec3f *result"),
    "is [[seam::scatter]] and const")
rejects(
    "a second [[seam::scatter]] is refused",
    STRIDE_BODY + STRIDE_ENTRY.replace(
        "[[seam::stride(3)]] const float *vector",
        "[[seam::scatter]] float *vector"),
    "One call returns one value")
rejects(
    "an access attribute on a scalar is refused",
    ELEMENT_BODY + ELEMENT_ENTRY.replace("float scale",
                                         "[[seam::gather]] float scale"),
    "a scalar arrives in the record itself")
rejects(
    "[[seam::stride(0)]] is refused",
    STRIDE_BODY + STRIDE_ENTRY.replace("[[seam::stride(9)]]",
                                       "[[seam::stride(0)]]"),
    "hands every thread the same pointer")
rejects(
    "[[seam::stride]] with no argument is refused",
    STRIDE_BODY + STRIDE_ENTRY.replace("[[seam::stride(9)]]", "[[seam::stride]]"),
    "needs its argument")
rejects(
    "[[seam::pod]] on the thread index is refused",
    BODY + GOOD_ENTRY.replace("[[seam::index]] unsigned index",
                              "[[seam::index]] [[seam::pod(4)]] unsigned index"),
    "no address space and no pointee")
rejects(
    "an access attribute on the declaration is named, not misreported",
    STRIDE_BODY + STRIDE_ENTRY.replace(
        "[[seam::args]] [[seam::entry]] void",
        "[[seam::args]] [[seam::entry]] [[seam::stride(3)]] void"),
    "belongs on a parameter, not on the declaration")
rejects(
    "an access attribute without [[seam::entry]] is refused",
    STRIDE_BODY + STRIDE_ENTRY.replace("[[seam::args]] [[seam::entry]]",
                                       "[[seam::args]]"),
    "emits the record and no call")
rejects(
    "[[seam::gather]] beside [[seam::index]] is refused",
    ELEMENT_BODY + ELEMENT_ENTRY.replace("[[seam::index]] unsigned vert",
                                         "[[seam::index]] [[seam::gather]] "
                                         "unsigned vert"),
    "neither of those is a buffer")

# A REFUSAL IS AN ERROR WITH A FILE, LINE AND COLUMN, which is the generator's
# contract for every construct it does not understand, and the reason a
# contributor can act on one. Checked as a position rather than as prose: the
# offending parameter is on the declaration's own line, not on line 1.
source = STRIDE_BODY + STRIDE_ENTRY.replace(
    "[[seam::stride(3)]] const float *vector",
    "[[seam::stride(3)]] [[seam::gather]] const float *vector")
code, _, err, _ = run(source)
position = re.search(r"fixture\.kernel\.cpp:(\d+):(\d+): ", err)
# The line the offending parameter is written on, and the column its first
# character sits at, computed from the fixture rather than written down twice.
wanted = next(i for i, line in enumerate(source.split("\n"), start=1)
              if "[[seam::gather]]" in line)
check("an unsupported shape is refused with a file, line and column",
      code == 1 and position is not None and
      int(position.group(1)) == wanted and int(position.group(2)) == 5,
      f"exit {code}, stderr {err.strip()!r}, wanted line {wanted}")

# ---------------------------------------------------------------------------
print("two parameters with the same name")
# ---------------------------------------------------------------------------

# A KNOWN GAP, CLOSED HERE. Two parameters of one name were accepted and
# emitted a record with two identical fields. g++ refuses that struct, so the
# failure was loud but LATE, in a generated file rather than in the declaration,
# and it is a construct this script does not understand: refused at the parse,
# with a position, like every other one.
rejects(
    "two parameters with the same name are refused",
    BODY + GOOD_ENTRY.replace("float scale", "float source"),
    "a second parameter named 'source'")
rejects(
    "a parameter taking a generator-owned name is refused",
    BODY + GOOD_ENTRY.replace("float scale", "float args"),
    "already use")
rejects(
    "a parameter with the reserved seam_ prefix is refused",
    BODY + GOOD_ENTRY.replace("float scale", "float seam_scale"),
    "already use")

# ---------------------------------------------------------------------------
print("the four lexical traps, on the element-shape attributes")
# ---------------------------------------------------------------------------

# Each of the four once let a forbidden construct through WITH A ZERO EXIT, and
# each is handled in the lexer that both paths read, not in the parser. That is
# the claim under test: the attributes added since do not get their own
# handling and do not need one.
rejects(
    "a digit separator inside a stride argument is refused",
    STRIDE_BODY + STRIDE_ENTRY.replace("[[seam::stride(9)]]",
                                       "[[seam::stride(1'0)]]"),
    "digit separator")
rejects(
    "[[seam::gather]] broken across two lines is refused",
    ELEMENT_BODY + ELEMENT_ENTRY.replace("[[seam::gather]]",
                                         "[[seam::\n    gather]]"),
    "not a complete [[seam::...]] attribute")
rejects(
    "a line comment ending in a backslash is refused beside a stride",
    STRIDE_BODY + STRIDE_ENTRY.replace(
        "[[seam::stride(9)]] const float *matrix,",
        "[[seam::stride(9)]] const float *matrix, // the block \\"),
    "splices the next line")
# The block comment must WORK, and the parameter below it must still be
# reported at its own line: the mask carries the newlines a block comment spans.
source = STRIDE_BODY + STRIDE_ENTRY.replace(
    "    [[seam::device]] [[seam::stride(3)]] const float *vector,",
    "    /* the vector this\n       block multiplies */\n"
    "    double vector,")
code, _, err, _ = run(source)
position = re.search(r"fixture\.kernel\.cpp:(\d+):(\d+): ", err)
wanted = next(i for i, line in enumerate(source.split("\n"), start=1)
              if "double vector" in line)
check("a multi-line block comment inside a parameter list carries its lines",
      code == 1 and "MSL has no double" in err and
      position is not None and int(position.group(1)) == wanted,
      f"exit {code}, stderr {err.strip()!r}, wanted line {wanted}")

# ---------------------------------------------------------------------------
print()
print("--emit args: the record half, for a caller in another translation unit")
# ---------------------------------------------------------------------------

# A record is a TYPE and a caller that fills one needs it; the `__global__` and
# the launcher beside it are DEFINITIONS and only one translation unit may hold
# them. Until the two were split, a launch written in a header two translation
# units include could not become a generated entry point at all, which is what
# every remaining hand-written CUDA launch in the neutral kernel tree is.

ARGS_BODY = """#pragma once

[[seam::device_fn]] inline float
split_probe([[seam::device]] const float *source, unsigned index) {
    return source[index];
}
"""

ARGS_ENTRY = """
[[seam::args]] [[seam::entry]] void split_probe(
    [[seam::device]] const float *source,
    [[seam::device]] [[seam::scatter]] float *destination,
    [[seam::count]] unsigned count, [[seam::index]] unsigned index);
"""

args_text = accepts("--emit args renders", ARGS_BODY + ARGS_ENTRY, emit="args")
entry_text = accepts("--emit entry still renders", ARGS_BODY + ARGS_ENTRY)

# THE SPLIT IS BY DEFINITION VERSUS DECLARATION, and each half is checked for
# what it must NOT carry as well as what it must.
check("the args half carries the record",
      "struct SplitProbeArgs {" in args_text)
check("the args half carries the layout assertions",
      "offsetof(SplitProbeArgs, destination) == 16" in args_text)
check("the args half declares the launcher",
      'extern "C" void split_probe_entry_launch(\n'
      "    const void *record, cudaStream_t queue);" in args_text)
# "no definition" is checked on the two spellings a DEFINITION takes, not on
# the bare words: this file's own prose explains what it does not carry, and a
# substring test over that prose would pass or fail on the comment rather than
# on the code.
check("the args half defines no __global__",
      "__global__ void" not in args_text)
check("the args half defines no launcher body",
      "cudaGetLastError" not in args_text)

check("the entry half defines the __global__",
      "__global__ void split_probe_entry(" in entry_text)
check("the entry half defines the launcher",
      'extern "C" void split_probe_entry_launch(\n'
      "    const void *record, cudaStream_t queue) {" in entry_text)
check("the entry half no longer defines the record",
      "struct SplitProbeArgs {" not in entry_text)
check("the entry half includes the args half",
      '#include "fixture.args.cuh"' in entry_text)

# ONE DECLARATION, so the two halves cannot disagree about the record: the
# entry rendering names the record it never defines.
check("the entry half names the record it does not define",
      "SplitProbeArgs args" in entry_text)

# The pointee size assertion belongs with the record, because it is what a
# FILLER needs to have checked: it addresses those bytes too.
POD_ENTRY = """
[[seam::args]] [[seam::entry]] void split_probe(
    [[seam::device]] [[seam::pod(36)]] const Mat3x3f *source,
    [[seam::device]] [[seam::scatter]] float *destination,
    [[seam::count]] unsigned count, [[seam::index]] unsigned index);
"""
pod_args = accepts("--emit args renders a struct pointee",
                   ARGS_BODY + POD_ENTRY, emit="args")
check("the args half asserts the declared pointee size",
      "static_assert(sizeof(Mat3x3f) == 36," in pod_args)

# A FILE WITH NO ENTRY STILL RENDERS, so a build rule runs over every neutral
# kernel without a hand-kept list, exactly as the entry half already does.
plain = accepts("--emit args renders a file with no entry declaration",
                ARGS_BODY, emit="args")
check("that rendering says the file declares no entry",
      "declares no [[seam::args]] entry" in plain)

# METAL SPLITS TOO, AND FOR A REASON THAT IS EASY TO MISS. Its entry rendering
# is spliced into one shader, so the record looks like it needs one home. The
# host is the second: the shader READS the record and the ObjC++ that fills it
# WRITES it, and those are different translation units. Without this rendering
# the host writes its half by hand, which is a mirror pair.
metal_args = accepts("--emit args renders for --target metal",
                     ARGS_BODY + ARGS_ENTRY, target="metal", emit="args")
check("the metal args half asserts the record layout",
      "static_assert(sizeof(" in metal_args and
      "static_assert(offsetof(" in metal_args)
check("the metal args half declares no launcher and names no CUDA type",
      "cudaStream_t" not in metal_args and 'extern "C"' not in metal_args)

# THE OTHER TWO TARGETS HAVE ONE CONSUMER EACH, so splitting them would be a
# second file nobody includes. Refused by name rather than rendered empty.
for target in ("cpp", "rust"):
    code, _, err, _ = run(ARGS_BODY + ARGS_ENTRY, target=target, emit="args")
    check(f"--emit args is refused for --target {target}",
          code == 2 and "has no " + target + " rendering" in err,
          f"exit {code}, stderr {err.strip()!r}")


# ---------------------------------------------------------------------------
print()
print("[[seam::group]]: the second launch shape")
# ---------------------------------------------------------------------------

# ONE GROUP PER ELEMENT, not one thread per element. The group count, the group
# width and the group-local scratch are all semantic here, which is why the
# element form cannot express a kernel that folds one aggregate through a
# barrier: its guard is per thread, its index is flat, and it has nowhere to
# put scratch.

GROUP_BODY = """#pragma once

[[seam::device_fn]] inline void
group_probe(unsigned aggregate, unsigned lane, unsigned width,
                [[seam::device]] const float *values,
                [[seam::device]] float *totals,
                [[seam::threadgroup]] float *scratch, unsigned length) {
    compute::threadgroup_barrier();
}
"""

GROUP_ENTRY = """
[[seam::args]] [[seam::entry]] [[seam::group]] void group_probe(
    [[seam::index]] unsigned aggregate, [[seam::lane]] unsigned lane,
    [[seam::width]] unsigned width,
    [[seam::device]] const float *values,
    [[seam::device]] float *totals,
    [[seam::threadgroup]] [[seam::scratch(12)]] float *scratch,
    unsigned length, [[seam::count]] unsigned count);
"""

group = {}
for target in ("cu", "metal", "cpp", "rust"):
    group[target] = accepts(f"a group entry renders for {target}",
                            GROUP_BODY + GROUP_ENTRY, target=target)

# THE THREE COORDINATES, each read from the launch rather than the record.
check("cu indexes by block, not by a flat thread id",
      "const unsigned aggregate = blockIdx.x;" in group["cu"]
      and "blockIdx.x * blockDim.x" not in group["cu"])
check("cu takes the lane from threadIdx and the width from blockDim",
      "const unsigned lane = threadIdx.x;" in group["cu"]
      and "const unsigned width = blockDim.x;" in group["cu"])
check("metal indexes by threadgroup position",
      "uint aggregate [[threadgroup_position_in_grid]]," in group["metal"]
      and "uint lane [[thread_position_in_threadgroup]]," in group["metal"]
      and "uint width [[threads_per_threadgroup]]," in group["metal"])

# NONE OF THE THREE IS A RECORD FIELD: the launch supplies them, so a driver
# that filled one would be writing a number the backend overwrites.
for coordinate in ("aggregate", "lane", "width"):
    check(f"'{coordinate}' is not a record field",
          f"    unsigned {coordinate};" not in group["cu"]
          and f"    pub {coordinate}:" not in group["rust"])

# SCRATCH IS DECLARED BY THE ENTRY, statically, and is not a record field
# either: a driver cannot fill storage that exists only while a group runs.
check("cu declares the scratch array",
      "__shared__ float scratch[12];" in group["cu"])
check("metal declares the scratch array",
      "threadgroup float scratch[12];" in group["metal"])
check("scratch is not a record field",
      "    ArenaHandle scratch;" not in group["cu"])

# THE GUARD IS ON THE GROUP INDEX, which is what makes it safe before a
# barrier: a group returns whole or not at all.
check("the guard tests the group index against the count",
      "if (aggregate >= args.count) {" in group["cu"])

# THE WIDTH IS THE CALLER'S. The element launcher picks a block size because
# that choice computes no value; here it is how many threads cooperate on one
# problem, which is a statement about the kernel that the generator cannot make.
check("the group launcher takes the width from its caller",
      "    const void *record, unsigned seam_group_width,\n"
      "    cudaStream_t queue) {" in group["cu"]
      and "choose_block_size" not in group["cu"])
check("the group launch is one block per element",
      "group_probe_entry<<<args.count, seam_group_width, 0, queue>>>(args);"
      in group["cu"])
group_args = accepts("a group entry renders its args half",
                     GROUP_BODY + GROUP_ENTRY, emit="args")
check("the args half declares the group launcher shape",
      "    const void *record, unsigned seam_group_width,\n"
      "    cudaStream_t queue);" in group_args)

# THE HOST SHIM RUNS THE LANES IN ORDER, one whole group per step, and its
# range counts GROUPS rather than threads. Lanes of a group exchange data only
# through the group-local scratch and through device memory, and a read of
# another lane's write is ordered by a barrier on both device targets and is a
# race without one there too, so for a body that does not synchronize one order
# computes what every other order computes. The negative half of that claim, a
# body that DOES synchronize, is measured further down by compiling one.
check("the cpp shim takes the width and a range over groups",
      "    unsigned seam_group_width, unsigned begin, unsigned end) {"
      in group["cpp"])
check("the cpp shim runs one whole group per step of the outer loop",
      "    for (unsigned aggregate = begin;\n"
      "         aggregate < seam_last; ++aggregate) {" in group["cpp"])
check("the cpp shim runs every lane of that group, in order",
      "        for (unsigned lane = 0;\n"
      "             lane < seam_group_width; ++lane) {" in group["cpp"])
check("the cpp shim takes the body's width from the caller's",
      "    const unsigned width = seam_group_width;" in group["cpp"])
# ONE ARRAY PER GROUP, and uninitialized: threadgroup memory arrives with
# unspecified contents on both device targets, so a body reading its scratch
# before writing it is a defect this rendering reproduces rather than hides.
# Inside the group loop is what gives it the group's lifetime; outside the lane
# loop is what makes it shared by the lanes.
# `find` rather than `index`, so a missing landmark REPORTS rather than
# raising and taking the rest of this file's checks with it.
_scratch_at = group["cpp"].find("float scratch[12];")
_groups_at = group["cpp"].find("for (unsigned aggregate = begin;")
_lanes_at = group["cpp"].find("for (unsigned lane = 0;")
check("the cpp shim declares the scratch once per group",
      -1 < _groups_at < _scratch_at < _lanes_at,
      f"group loop {_groups_at}, scratch {_scratch_at}, lane loop {_lanes_at}")
check("the cpp rendering still emits the record",
      "struct GroupProbeArgs {" in group["cpp"])

# THE DRIVER IS TOLD THE SHAPE, because the two extents are not interchangeable.
check("the rust twin names the shape and its scratch",
      "pub const GROUP_PROBE_IS_GROUP: bool = true;" in group["rust"]
      and "pub const GROUP_PROBE_SCRATCH_BYTES: u32 = 48;" in group["rust"])
check("an element entry claims neither",
      "_IS_GROUP" not in sizes["rust"])

# --- what a group declaration may not do ----------------------------------

rejects("a lane on an element entry is refused",
        GROUP_BODY + GROUP_ENTRY.replace(" [[seam::group]]", ""),
        "ELEMENT entry, which has no groups")
rejects("scratch on an element entry is refused",
        GROUP_BODY + GROUP_ENTRY.replace(" [[seam::group]]", "")
        .replace("[[seam::lane]] unsigned lane,", "unsigned lane,")
        .replace("[[seam::width]] unsigned width,", "unsigned width,"),
        "ELEMENT entry, which has no groups")
rejects("[[seam::group]] without [[seam::entry]] is refused",
        GROUP_BODY + GROUP_ENTRY.replace("[[seam::args]] [[seam::entry]]",
                                         "[[seam::args]]"),
        "names a LAUNCH SHAPE")
rejects("a group entry with no lane is refused",
        GROUP_BODY + GROUP_ENTRY.replace("[[seam::lane]] unsigned lane,",
                                         "unsigned lane,"),
        "every thread of a group computes the whole of it")
rejects("a group entry with no group index is refused",
        GROUP_BODY + GROUP_ENTRY.replace("[[seam::index]] unsigned aggregate,",
                                         "unsigned aggregate,"),
        "every group computes the same value")
rejects("two lanes are refused",
        GROUP_BODY + GROUP_ENTRY.replace(
            "[[seam::width]] unsigned width,",
            "[[seam::lane]] unsigned width,"),
        "a second [[seam::lane]] parameter")
# A gather is an element at the THREAD index, and a group entry's index is its
# GROUP's, so every thread of the group would read one element.
rejects("a gather on a group entry is refused",
        GROUP_BODY + GROUP_ENTRY.replace(
            "[[seam::device]] const float *values,",
            "[[seam::device]] [[seam::gather]] const float *values,"),
        "every thread of the group would read the same element")
rejects("scratch without a threadgroup address space is refused",
        GROUP_BODY + GROUP_ENTRY.replace(
            "[[seam::threadgroup]] [[seam::scratch(12)]] float *scratch,",
            "[[seam::device]] [[seam::scratch(12)]] float *scratch,"),
        "Scratch IS the threadgroup address space")
rejects("const scratch is refused",
        GROUP_BODY + GROUP_ENTRY.replace(
            "[[seam::scratch(12)]] float *scratch,",
            "[[seam::scratch(12)]] const float *scratch,"),
        "exists to be written")
rejects("a struct scratch pointee is refused",
        GROUP_BODY + GROUP_ENTRY.replace(
            "[[seam::scratch(12)]] float *scratch,",
            "[[seam::scratch(12)]] Vec3f *scratch,"),
        "its size has to be a literal")
rejects("scratch with no element count is refused",
        GROUP_BODY + GROUP_ENTRY.replace("[[seam::scratch(12)]]",
                                         "[[seam::scratch]]"),
        "the group-local array's length, in ELEMENTS")
rejects("zero-element scratch is refused",
        GROUP_BODY + GROUP_ENTRY.replace("scratch(12)", "scratch(0)"),
        "asks for 0 elements")
# Metal enforces the 32768 byte static threadgroup cap at pipeline creation,
# so a declaration over it would build on two backends and not the third.
rejects("scratch over the static threadgroup cap is refused",
        GROUP_BODY + GROUP_ENTRY.replace("scratch(12)", "scratch(8193)"),
        "over the 32768 byte cap")
# One element under the cap is the boundary that must still pass, so the
# refusal above is measured at the edge rather than somewhere past it.
accepts("scratch exactly at the cap is accepted",
        GROUP_BODY + GROUP_ENTRY.replace("scratch(12)", "scratch(8192)"))


# ---------------------------------------------------------------------------
print()
print("the group host rendering, compiled and RUN")
# ---------------------------------------------------------------------------

# A RENDERING THAT IS ONLY GREPPED IS NOT KNOWN TO WORK. Everything above reads
# the generated text; what a host rendering owes is an ANSWER, because it is
# the oracle a parity gate compares a device backend against. So this section
# compiles the shim with a real C++ compiler and runs it, against a body that
# records which lane of which group called it.
#
# A HOST C++ COMPILER IS NOT A NEW DEPENDENCY: the CPU backend compiles the
# generated shims with one, so a machine that cannot run this section cannot
# build the solver either. It is named rather than skipped for the reason this
# tree names every absent backend: a section quietly dropped leaves a smaller
# count and a clean summary that nobody diffs.

CXX = os.environ.get("CXX", "g++")

# The body writes one float per (group, lane) pair it is called with, so the
# result is a full census of the calls rather than a sum that several different
# call sets could produce. `width` is written too, so a shim passing a width
# the caller did not ask for is visible.
RUN_BODY = """#pragma once

[[seam::device_fn]] inline void
group_census(unsigned aggregate, unsigned lane, unsigned width,
                 [[seam::device]] float *seen,
                 [[seam::threadgroup]] float *scratch, unsigned stride) {
    scratch[0] = float(aggregate);
    seen[aggregate * stride + lane] = 1000.0f * scratch[0] + float(width);
}
"""

RUN_ENTRY = """
[[seam::args]] [[seam::entry]] [[seam::group]] void group_census(
    [[seam::index]] unsigned aggregate, [[seam::lane]] unsigned lane,
    [[seam::width]] unsigned width,
    [[seam::device]] float *seen,
    [[seam::threadgroup]] [[seam::scratch(4)]] float *scratch,
    unsigned stride, [[seam::count]] unsigned count);
"""

# The driver below is what a caller of the shim does: it lays one arena out by
# hand, fills the record with an (arena, offset) handle into it, and calls the
# shim over a range of GROUPS. `ArenaHandle` is the seam's own four-field
# handle; the stub root this test renders against has an empty header of that
# name, so the driver declares it here, matching the layout the generated
# static_asserts check.
RUN_DRIVER = """
#include <cstdio>

int main() {
    const unsigned groups = 5, width = 3, stride = 8;
    static float seen[groups * stride];
    for (unsigned i = 0; i < groups * stride; ++i) {
        seen[i] = -1.0f;
    }
    unsigned char *bases[1] = {reinterpret_cast<unsigned char *>(seen)};
    GroupCensusArgs args{};
    args.seen.arena = 0;
    args.seen.off = 0;
    args.seen.size = groups * stride;
    args.seen.allocated = groups * stride;
    args.stride = stride;
    args.count = groups;
    args.seam_arena_count = 1;
    // A range that is a strict SUBSET of the groups on both ends, so a shim
    // that ignored `begin` or ran past `end` is visible in the census.
    group_census_entry(&args, bases, width, 1, 4);
    for (unsigned g = 0; g < groups; ++g) {
        for (unsigned l = 0; l < stride; ++l) {
            printf("%u %u %.0f\\n", g, l, seen[g * stride + l]);
        }
    }
    return 0;
}
"""

_tmp = tempfile.mkdtemp(prefix="kernelgen-run-")
try:
    # A KERNEL ROOT WITH A REAL HANDLE IN IT. The stub root the rest of this
    # file renders against carries an EMPTY arena_handle.hpp, which is right
    # for a test that only reads the generated text; a rendering that has to
    # compile needs the type the record's fields are declared with.
    _root = os.path.join(_tmp, "gen")
    os.makedirs(_root)
    with open(os.path.join(_root, "arena_handle.hpp"), "w",
              encoding="utf-8") as _f:
        _f.write("#pragma once\n"
                 "struct ArenaHandle {\n"
                 "    unsigned arena;\n"
                 "    unsigned off;\n"
                 "    unsigned size;\n"
                 "    unsigned allocated;\n"
                 "};\n")
    _src = os.path.join(_tmp, "census.kernel.cpp")
    with open(_src, "w", encoding="utf-8") as _f:
        _f.write(RUN_BODY + RUN_ENTRY)
    # Both renderings land in the same directory under the names the shim
    # spells, which is what a build does: the shim includes the rendered BODY
    # by the neutral file's basename.
    _shim = os.path.join(_root, "census.entry.cpp")
    _body = os.path.join(_root, "census.kernel.cpp")
    for _emit, _out in (("body", _body), ("entry", _shim)):
        subprocess.run([sys.executable, "-B", KERNELGEN, "--target", "cpp",
                        "--emit", _emit, "--out", _out, _src,
                        "--kernel-root", _root], check=True)
    _main = os.path.join(_tmp, "main.cpp")
    with open(_main, "w", encoding="utf-8") as _f:
        _f.write('#include "census.entry.cpp"\n' + RUN_DRIVER)
    _exe = os.path.join(_tmp, "census")
    _build = subprocess.run([CXX, "-std=c++17", "-O0", "-I", _root, _main,
                             "-o", _exe], capture_output=True, text=True)
    check("the group host shim compiles", _build.returncode == 0,
          _build.stderr.strip())
    if _build.returncode == 0:
        _out = subprocess.run([_exe], capture_output=True, text=True).stdout
        _seen = {}
        for _line in _out.splitlines():
            _g, _l, _v = _line.split()
            _seen[(int(_g), int(_l))] = float(_v)
        # EVERY LANE OF EVERY GROUP IN THE RANGE, and nothing outside it. The
        # value encodes the group the body was told it was in and the width it
        # was handed, so a shim that ran the right number of calls with the
        # wrong coordinates fails here rather than passing on a count.
        _wanted = {(_g, _l): 1000.0 * _g + 3.0
                   for _g in (1, 2, 3) for _l in (0, 1, 2)}
        _got = {_k: _v for _k, _v in _seen.items() if _v >= 0.0}
        check("every lane of every group in the range ran, once, with its own "
              "coordinates", _got == _wanted,
              f"got {sorted(_got.items())}")
        check("no group outside [begin, end) ran",
              all(_seen[(_g, _l)] == -1.0
                  for _g in (0, 4) for _l in range(8)))
        check("no lane at or past the width ran",
              all(_seen[(_g, _l)] == -1.0
                  for _g in (1, 2, 3) for _l in (3, 4, 5, 6, 7)))

    # THE NEGATIVE CONTROL, and it is the half that makes the rendering honest.
    # A body that synchronizes must NOT compile here: lanes running one after
    # another cannot honor a barrier, and the host seam therefore leaves
    # `compute::threadgroup_barrier` undefined. The stand-in below is a
    # `compute` namespace holding some other name, which is what the real host
    # seam is, so the error names the missing FUNCTION and not the namespace.
    _sync = os.path.join(_tmp, "sync.cpp")
    with open(_sync, "w", encoding="utf-8") as _f:
        _f.write("namespace compute {\n"
                 "inline float sq(float x) { return x * x; }\n"
                 "}\n"
                 '#include "census.entry.cpp"\n'
                 "static void probe() { compute::threadgroup_barrier(); }\n")
    _refused = subprocess.run([CXX, "-std=c++17", "-fsyntax-only", "-I", _root,
                               _sync], capture_output=True, text=True)
    check("a body that synchronizes has no host rendering",
          _refused.returncode != 0
          and "threadgroup_barrier" in _refused.stderr,
          _refused.stderr.strip()[:200])
finally:
    shutil.rmtree(_tmp)


# ---------------------------------------------------------------------------
print("the indirect gather's host rendering, compiled and RUN")
# ---------------------------------------------------------------------------

# THE FOURTH ADDRESSING SHAPE OWES AN ANSWER FOR THE SAME REASON THE GROUP ONE
# DOES, and one more besides: its whole justification is the BOUNDS CHECK, and
# a check is only known to work when an input it must refuse has been run
# through it. So this section compiles the shim, runs it over an index list
# holding both good and bad slots, and re-runs it with the check neutered.
#
# The body multiplies its three arguments by descending powers of ten, so the
# result names WHICH element reached WHICH argument. A generator that gathered
# at the thread index instead of at the slots, or that passed the slots out of
# order, produces a different number here rather than a plausible one.

IND_BODY = """#pragma once

[[seam::device_fn]] inline float
indirect_census(float a, float b, float c) {
    return 100.0f * a + 10.0f * b + c;
}
"""

IND_ENTRY = """
[[seam::args]] [[seam::entry]] void indirect_census(
    [[seam::device]] [[seam::through]] const float *values,
    [[seam::device]] [[seam::indices(3)]] const unsigned *element,
    [[seam::bound]] unsigned value_count,
    [[seam::device]] [[seam::scatter]] float *out,
    [[seam::count]] unsigned count);
"""

# ONE ARENA HOLDING ALL THREE BUFFERS, which is what an allocator hands out:
# the three handles differ only in their offset, so a shim that resolved one
# through another's offset is visible. Element 1 names a slot EQUAL to the
# bound and element 2 names one past it, which are the two an off-by-one in
# either direction gets wrong; both are inside the pool, so the neutered build
# below reads a defined value rather than wandering off.
IND_DRIVER = """
#include <cstdio>

int main() {
    const unsigned values = 6, elements = 4, count_off = 32, out_off = 80;
    static unsigned char pool[256];
    float *v = reinterpret_cast<float *>(pool);
    unsigned *e = reinterpret_cast<unsigned *>(pool + count_off);
    float *o = reinterpret_cast<float *>(pool + out_off);
    for (unsigned i = 0; i < values; ++i) {
        v[i] = float(i + 1);
    }
    const unsigned slots[elements * 3] = {
        0, 1, 2,
        3, 4, 6,
        5, 9, 0,
        2, 2, 5,
    };
    for (unsigned i = 0; i < elements * 3; ++i) {
        e[i] = slots[i];
    }
    for (unsigned i = 0; i < elements; ++i) {
        o[i] = -1.0f;
    }
    unsigned char *bases[1] = {pool};
    IndirectCensusArgs args{};
    args.values.arena = 0;
    args.values.off = 0;
    args.values.size = values;
    args.values.allocated = values;
    args.element.arena = 0;
    args.element.off = count_off;
    args.element.size = elements * 3;
    args.element.allocated = elements * 3;
    args.value_count = values;
    args.out.arena = 0;
    args.out.off = out_off;
    args.out.size = elements;
    args.out.allocated = elements;
    args.count = elements;
    args.seam_arena_count = 1;
    indirect_census_entry(&args, bases, 0, elements);
    for (unsigned i = 0; i < elements; ++i) {
        printf("%u %.0f\\n", i, o[i]);
    }
    return 0;
}
"""


def _indirect_run(shim_text, defines):
    """Compile one rendering of the shim with the driver and run it.

    Returns (compiled, exit code, {element: value}). A build that does not
    compile reports an empty census rather than raising, so the check that
    names it is the one that fails.
    """
    root = os.path.join(_ind_tmp, "gen-" + str(len(os.listdir(_ind_tmp))))
    os.makedirs(root)
    with open(os.path.join(root, "arena_handle.hpp"), "w",
              encoding="utf-8") as handle:
        handle.write("#pragma once\n"
                     "struct ArenaHandle {\n"
                     "    unsigned arena;\n"
                     "    unsigned off;\n"
                     "    unsigned size;\n"
                     "    unsigned allocated;\n"
                     "};\n")
    subprocess.run([sys.executable, "-B", KERNELGEN, "--target", "cpp",
                    "--emit", "body", "--out",
                    os.path.join(root, "census.kernel.cpp"), _ind_src,
                    "--kernel-root", root], check=True)
    with open(os.path.join(root, "census.entry.cpp"), "w",
              encoding="utf-8") as shim:
        shim.write(shim_text)
    main = os.path.join(root, "main.cpp")
    with open(main, "w", encoding="utf-8") as driver:
        driver.write('#include "census.entry.cpp"\n' + IND_DRIVER)
    exe = os.path.join(root, "census")
    build = subprocess.run([CXX, "-std=c++17", "-O0", "-I", root] +
                           list(defines) + [main, "-o", exe],
                           capture_output=True, text=True)
    if build.returncode != 0:
        return False, build.returncode, {}, build.stderr.strip()
    ran = subprocess.run([exe], capture_output=True, text=True)
    census = {}
    for line in ran.stdout.splitlines():
        index, value = line.split()
        census[int(index)] = float(value)
    return True, ran.returncode, census, ran.stderr.strip()


_ind_tmp = tempfile.mkdtemp(prefix="kernelgen-indirect-")
try:
    _ind_src = os.path.join(_ind_tmp, "census.kernel.cpp")
    with open(_ind_src, "w", encoding="utf-8") as _f:
        _f.write(IND_BODY + IND_ENTRY)
    _ind_root = os.path.join(_ind_tmp, "render")
    os.makedirs(_ind_root)
    with open(os.path.join(_ind_root, "arena_handle.hpp"), "w",
              encoding="utf-8") as _f:
        _f.write("#pragma once\n"
                 "struct ArenaHandle {\n"
                 "    unsigned arena;\n"
                 "    unsigned off;\n"
                 "    unsigned size;\n"
                 "    unsigned allocated;\n"
                 "};\n")
    _ind_shim = subprocess.run(
        [sys.executable, "-B", KERNELGEN, "--target", "cpp", "--emit", "entry",
         "--out", "/dev/stdout", _ind_src, "--kernel-root", _ind_root],
        capture_output=True, text=True, check=True).stdout

    # WITHOUT THE ASSERT, so a refused element is observable rather than fatal.
    # The guard is two statements, an assert and a return, and only the second
    # survives NDEBUG; that this build still refuses element 1 and element 2 is
    # what says the check is a GUARD and not only a diagnostic.
    _ok, _code, _census, _err = _indirect_run(_ind_shim, ("-DNDEBUG",))
    check("the indirect gather's host shim compiles", _ok, _err)
    if _ok:
        # 0: values[0..2] = 1, 2, 3   -> 100 + 20 + 3
        # 3: values[2],[2],[5] = 3, 3, 6 -> 300 + 30 + 6
        check("an element gathers its three values AT ITS OWN SLOTS, in slot "
              "order", _census.get(0) == 123.0 and _census.get(3) == 336.0,
              f"got {sorted(_census.items())}")
        check("a slot EQUAL to the bound is out of range, so its element is "
              "skipped", _census.get(1) == -1.0, f"got {_census.get(1)}")
        check("a slot past the bound is out of range, so its element is "
              "skipped", _census.get(2) == -1.0, f"got {_census.get(2)}")
        check("the run finished rather than trapping, with NDEBUG",
              _code == 0, f"exit {_code}")

    # WITH THE ASSERT LIVE, which is how the CUDA release build compiles: the
    # same input must TRAP rather than quietly skipping. A guard that only
    # returned would pass every check above and lose the loud failure this
    # project requires of a violated invariant.
    _ok_a, _code_a, _, _err_a = _indirect_run(_ind_shim, ())
    check("the same out-of-range slot TRAPS when the assert is live",
          _ok_a and _code_a != 0, f"compiled {_ok_a}, exit {_code_a}")

    # INJECTION ONE: the bound test admits its own boundary. Element 1's slot
    # equals the bound, so `<=` is exactly the off-by-one this catches.
    _wide = _ind_shim.replace("seam_slot[seam_k] < args->value_count",
                              "seam_slot[seam_k] <= args->value_count")
    check("the boundary injection changed the rendering",
          _wide != _ind_shim)
    _ok_w, _, _census_w, _ = _indirect_run(_wide, ("-DNDEBUG",))
    check("with `<=` for `<`, the element whose slot equals the bound is no "
          "longer refused", _ok_w and _census_w.get(1) != -1.0,
          f"got {_census_w.get(1)}")

    # INJECTION TWO: no check at all. Both bad elements then run, so the guard
    # is what refuses them rather than something else about the shape.
    _open = _ind_shim.replace("seam_slot[seam_k] < args->value_count", "true")
    check("the no-check injection changed the rendering", _open != _ind_shim)
    _ok_o, _, _census_o, _ = _indirect_run(_open, ("-DNDEBUG",))
    check("with the bound test removed, both out-of-range elements run",
          _ok_o and _census_o.get(1) != -1.0 and _census_o.get(2) != -1.0,
          f"got {sorted(_census_o.items())}")
    check("removing the bound test does not disturb the elements that were "
          "in range", _ok_o and _census_o.get(0) == 123.0
          and _census_o.get(3) == 336.0, f"got {sorted(_census_o.items())}")

    # INJECTION THREE: the slots reach the body in the wrong order. The census
    # is what separates that from a gather that merely ran.
    _swapped = _ind_shim.replace("values[seam_slot[0]],\n            "
                                 "values[seam_slot[1]]",
                                 "values[seam_slot[1]],\n            "
                                 "values[seam_slot[0]]")
    check("the slot-order injection changed the rendering",
          _swapped != _ind_shim)
    _ok_s, _, _census_s, _ = _indirect_run(_swapped, ("-DNDEBUG",))
    check("swapping two slots changes what the body is handed",
          _ok_s and _census_s.get(0) == 213.0, f"got {_census_s.get(0)}")
finally:
    shutil.rmtree(_ind_tmp)


# ---------------------------------------------------------------------------
print("the table fragment, which is the kernel table's per-source half")
# ---------------------------------------------------------------------------

# A fragment is read TWICE by its caller, so its two roles have to be
# separable by the preprocessor rather than by where the text sits.
_cu_table = accepts("a cu table fragment renders", BODY + GOOD_ENTRY,
                    emit="table")
check("the cu fragment carries both roles behind one guard",
      "#ifdef PPF_BE_TABLE_ARGS_INCLUDES" in _cu_table
      and "#else" in _cu_table and "#endif" in _cu_table,
      _cu_table)
check("the cu fragment includes its argument header from the kernel root",
      '#include "fixture.args.cuh"' in _cu_table, _cu_table)
check("the cu fragment carries one row naming the record and the launcher",
      "PPF_BE_KERNEL(fix_scale_entry, FixScaleArgs, "
      "fix_scale_entry_launch, fix_scale_entry_seam_handle_offsets, 2)"
      in _cu_table, _cu_table)
# THE ROW CARRIES WHERE ITS HANDLES SIT, because the backend checks each one's
# arena against the live binding table before it launches, and that check can
# only be made at dispatch. The array is emitted in the ARGS inclusion, beside
# the record it describes, so the row that names it is declared after it.
check("the cu fragment emits the handle offsets it names",
      "static const unsigned fix_scale_entry_seam_handle_offsets[] = {0, 16};"
      in _cu_table, _cu_table)
# THE ROW CARRIES NO ID. A dense id is a property of the concatenation, so a
# fragment that spelled one would be asserting something it cannot know.
check("a cu row carries no kernel id",
      "PPF_BE_KERNEL(0" not in _cu_table and "= 0)" not in _cu_table,
      _cu_table)

_rust_table = accepts("a rust table fragment renders", BODY + GOOD_ENTRY,
                      target="rust", emit="table")
check("the rust fragment is a tuple row per entry",
      '("fix_scale_entry", ' in _rust_table, _rust_table)
check("the rust row carries the handle offsets and the launch shape",
      _rust_table.rstrip().endswith("false, 0),"), _rust_table)

_metal_table = accepts("a metal table fragment renders", BODY + GOOD_ENTRY,
                       target="metal", emit="table")
# PURE DATA, WHICH IS WHAT SEPARATES THIS FROM THE cu FRAGMENT. A CUDA row names
# a record TYPE and a launcher SYMBOL and so needs the record's header; Metal
# binds by NAME out of a compiled library and copies the record as opaque bytes,
# so the row is a name, a size, a launch shape and a scratch figure, and the
# fragment includes nothing.
check("the metal fragment is a data row per entry",
      'PPF_BE_KERNEL_METAL("fix_scale_entry", ' in _metal_table, _metal_table)
check("the metal fragment includes no header",
      "#include" not in _metal_table, _metal_table)
check("the metal row carries the launch shape, scratch and the arena slot",
      _metal_table.rstrip().endswith("false, 0, 40)"), _metal_table)
# A RECORD WITH NO HANDLE CARRIES NO `seam_arena_count`, so its row says so
# rather than naming an offset the library would then patch and thereby corrupt
# a scalar. `0xffffffff` is that statement.
_NO_HANDLE_BODY = """// File: fixture.kernel.cpp
#pragma once

[[seam::host_device_fn]] inline void
bump_counter([[seam::device]] unsigned *out, unsigned index) {
    out[index] = index;
}
"""
_no_handle = accepts(
    "a scalar-only entry renders a metal row", _NO_HANDLE_BODY + """
[[seam::args]] [[seam::entry]] void bump_counter(
    float scale, [[seam::count]] unsigned count,
    [[seam::index]] unsigned index);
""", target="metal", emit="table")
check("a handle-free row names no arena slot",
      "0xffffffffu)" in _no_handle, _no_handle)
# THE ROW CARRIES NO ID, on the same terms as the cu one: a dense id is a
# property of the concatenation.
check("a metal row carries no kernel id",
      "PPF_BE_KERNEL_METAL(0" not in _metal_table, _metal_table)

# The consumers of a table are the two libraries and the driver. `cpp` looks no
# kernel up by id, so asking for one is a mistake worth naming rather than an
# omission worth filling.
# Exit 2 rather than 1: this is a USAGE error, the same class as asking for a
# Rust body or a non-cu args rendering, and not a construct the generator
# failed to understand.
_code, _, _err, _ = run(BODY + GOOD_ENTRY, target="cpp", emit="table")
check("--emit table --target cpp is refused by name",
      _code == 2 and "--emit table has no" in _err,
      f"exit {_code}, stderr {_err.strip()!r}")

# A build rule renders a fragment for EVERY neutral kernel, so a file with no
# entry has to produce a file rather than fail, and that file must contribute
# no row.
_empty = accepts("a kernel with no entry still renders a fragment",
                 BODY, emit="table")
check("a kernel with no entry contributes no row",
      "PPF_BE_KERNEL" not in _empty and "contributes no row" in _empty,
      _empty)

# `[[seam::args]]` alone declares a RECORD and no entry point, so there is
# nothing to launch and nothing to put in a table.
_args_only = accepts("an args-only declaration renders a fragment",
                     BODY + GOOD_ENTRY.replace("[[seam::args]] [[seam::entry]]",
                                               "[[seam::args]]"),
                     emit="table")
check("an args-only declaration contributes no row",
      "PPF_BE_KERNEL" not in _args_only, _args_only)

# A GROUP ENTRY IS A DIFFERENT ROW SHAPE, because its launcher takes the group
# width and its scratch is static. Sharing one row would make the width a
# value the caller had to know not to pass.
_group_table = accepts("a group entry renders a group row",
                       GROUP_BODY + GROUP_ENTRY, emit="table")
check("a group row is PPF_BE_KERNEL_GROUP and carries its scratch bytes",
      "PPF_BE_KERNEL_GROUP(group_probe_entry, GroupProbeArgs, "
      "group_probe_entry_launch, 48, group_probe_entry_seam_handle_offsets"
      in _group_table, _group_table)
_group_rust = accepts("a group entry renders a rust group row",
                      GROUP_BODY + GROUP_ENTRY, target="rust", emit="table")
check("the rust group row says group and carries its scratch bytes",
      "true, 48)," in _group_rust, _group_rust)

# The relative spelling is what lets the caller concatenate fragments, so a
# source outside the root it was given cannot produce one.
_code, _, _err, _ = run(BODY + GOOD_ENTRY, emit="table",
                        extra=("--kernel-root", STUB_ROOT))
check("a source outside the kernel root is refused",
      _code == 1 and "is not inside the kernel root" in _err,
      f"exit {_code}, stderr {_err.strip()!r}")


# ---------------------------------------------------------------------------
# THE SPELLING IN A MESSAGE AND THE SPELLING THE PARSER READS ARE THE SAME ONE.
#
# kernelgen.py routes its parsing through ATTRIBUTE_NAMESPACE and writes the
# attribute spelling out in its diagnostics, which reads better than
# interpolating the constant into two dozen messages. That is safe only while
# the two agree, so this reads the generator's own source and asserts that every
# `[[<namespace>::` literal in it names the namespace the parser reads. A rename
# that missed the messages would leave them naming a spelling no body can carry.
with open(os.path.join(HERE, "kernelgen.py")) as _f:
    _GEN_SRC = _f.read()
_PARSED_NS = re.search(r'^ATTRIBUTE_NAMESPACE = "([a-z_]+)"$', _GEN_SRC, re.M)
if _PARSED_NS is None:
    sys.exit("test_kernelgen: kernelgen.py declares no ATTRIBUTE_NAMESPACE; "
             "either the constant moved or this check is broken, and either "
             "way it cannot pass.")
_LITERAL_NS = set(re.findall(r"\[\[([A-Za-z_][A-Za-z0-9_]*)::", _GEN_SRC))
check("every attribute spelling written into a message names the parsed "
      "namespace",
      _LITERAL_NS == {_PARSED_NS.group(1)},
      f"kernelgen.py writes {sorted(_LITERAL_NS)} where the parser reads "
      f"{_PARSED_NS.group(1)!r}")


# ---------------------------------------------------------------------------
print("the diagnostic lane")

# A body that reports through the channel cannot NAME the channel: the three
# targets bind three different things behind one alias, and the record cannot
# carry it, because which channel a dispatch reports through is a property of
# the process rather than of the kernel's data. So the entry declaration marks
# one parameter as the lane and the generator threads the target's own handle
# into that position.
DIAG_BODY = """
[[seam::device_fn]] inline unsigned diag_probe(
    [[seam::device]] const unsigned *a, unsigned element, DiagHandle diag) {
    DIAG_ASSERT4(diag, a[element] != 0u, (float)element, 0.0f, 0.0f, 0.0f);
    return a[element];
}
"""

DIAG_ENTRY = DIAG_BODY + """
[[seam::args]] [[seam::entry]] void diag_probe(
    [[seam::device]] const unsigned *a,
    [[seam::index]] unsigned element,
    [[seam::diag]] DiagHandle diag,
    [[seam::device]] [[seam::scatter]] unsigned *out,
    [[seam::count]] unsigned count);
"""

_diag_cu = accepts("an entry may declare a diagnostic lane", DIAG_ENTRY)
check("the CUDA entry takes the channel as a parameter",
      "DiagProbeArgs args, DIAG_ARG)" in _diag_cu, _diag_cu)
check("the CUDA entry binds the channel under the declared name",
      "DiagHandle diag = DIAG_BIND(element);" in _diag_cu, _diag_cu)
check("the CUDA launcher hands the kernel the global channel",
      "(args, diagnostics::global());" in _diag_cu, _diag_cu)
check("a CUDA entry file with a lane includes the channel's transport",
      '#include "diagnostics/diagnostics.hpp"' in _diag_cu, _diag_cu)

_diag_cpp = accepts("the cpp rendering takes a diagnostic lane",
                    DIAG_ENTRY, target="cpp")
check("the cpp entry takes the channel as a trailing parameter",
      "DiagHandle diag)" in _diag_cpp, _diag_cpp)
check("the cpp entry forwards the channel in its declared position",
      "        element,\n            diag);" in _diag_cpp, _diag_cpp)

_diag_metal = accepts("the metal rendering takes a diagnostic lane",
                      DIAG_ENTRY, target="metal")
check("the metal entry binds the channel it already carried",
      _diag_metal.count("DIAG_BIND(element)") == 1, _diag_metal)

# THE LANE IS NOT A RECORD FIELD, which is the whole point: two dispatches of
# one entry cannot then disagree about where a failed assert lands.
_diag_rust = accepts("the rust rendering takes a diagnostic lane",
                     DIAG_ENTRY, target="rust")
check("the diagnostic lane is not a record field",
      "pub diag" not in _diag_rust and "DiagHandle" not in _diag_rust,
      _diag_rust)

rejects("a diagnostic lane outside an entry declaration is refused",
        DIAG_BODY.replace("DiagHandle diag",
                          "[[seam::diag]] DiagHandle diag"),
        "outside an entry declaration")
rejects("a diagnostic lane on the wrong type is refused",
        DIAG_ENTRY.replace("[[seam::diag]] DiagHandle diag",
                           "[[seam::diag]] unsigned diag"),
        "DiagHandle")
rejects("a diagnostic lane under another name is refused",
        DIAG_ENTRY.replace("[[seam::diag]] DiagHandle diag",
                           "[[seam::diag]] DiagHandle channel")
                  .replace("            diag);", "            channel);"),
        "'diag'")
rejects("a second diagnostic lane is refused",
        DIAG_ENTRY.replace(
            "    [[seam::count]] unsigned count);",
            "    [[seam::diag]] DiagHandle diag,\n"
            "    [[seam::count]] unsigned count);"),
        "diag")
rejects("a diagnostic lane taken by pointer is refused",
        DIAG_ENTRY.replace("[[seam::diag]] DiagHandle diag",
                           "[[seam::diag]] DiagHandle *diag"),
        "value")
rejects("a diagnostic lane in an address space is refused",
        DIAG_ENTRY.replace("[[seam::diag]] DiagHandle diag",
                           "[[seam::diag]] [[seam::device]] DiagHandle diag"),
        "diag")


# ---------------------------------------------------------------------------
print("rule (1-LANE): a cooperative body and its serial twin")

TWIN = """#pragma once

[[seam::device_fn]] [[seam::cooperative]] inline unsigned
probe_rank(unsigned value, unsigned lane) {
    unsigned mask = compute::simd_ballot(value != 0u);
    return bits::popcount(mask & ((1u << lane) - 1u));
}

[[seam::device_fn]] [[seam::serial]] inline unsigned
probe_rank(unsigned value, unsigned lane) {
    (void)value;
    return lane;
}
"""

for _target, _wanted, _unwanted in (("cu", "simd_ballot", "return lane;"),
                                    ("metal", "simd_ballot", "return lane;"),
                                    ("cpp", "return lane;", "simd_ballot")):
    _rendered = accepts(f"the {_target} rendering takes one twin",
                        TWIN, target=_target, emit="body")
    check(f"the {_target} rendering keeps the body it wants",
          _wanted in _rendered, f"{_wanted!r} missing")
    # THE OTHER TWIN IS NEUTRALIZED TO COMMENTS, not deleted, so the line count
    # is unchanged and the Metal `#line` mapping stays exact.
    check(f"the {_target} rendering drops the other twin",
          _unwanted not in _rendered.replace("//", "\n//"),
          f"{_unwanted!r} still rendered")
    check(f"the {_target} rendering keeps the neutral line count",
          _rendered.count("\n") >= TWIN.count("\n"),
          "the twin was deleted rather than neutralized")

rejects("a cooperative body with no serial twin is refused",
        TWIN[:TWIN.index("[[seam::device_fn]] [[seam::serial]]")],
        "no [[seam::serial]] twin", target="cpp", emit="body")
rejects("a serial body with no cooperative twin is refused",
        TWIN.replace("[[seam::cooperative]]", "[[seam::serial]]", 1)
            .replace("""[[seam::device_fn]] [[seam::serial]] inline unsigned
probe_rank(unsigned value, unsigned lane) {
    (void)value;
    return lane;
}
""", "", 1),
        "no [[seam::cooperative]] twin", target="cpp", emit="body")


print()
if FAILURES:
    print(f"{len(FAILURES)} of {COUNT} checks FAILED:")
    for name in FAILURES:
        print(f"  {name}")
    sys.exit(1)
print(f"{COUNT} checks passed")
