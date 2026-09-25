#!/usr/bin/env python3
"""Refuse a `Scatter::Disjoint` row whose kernel can reach a FLOAT seam atomic.

A `Scatter::Disjoint` row runs as concurrent rayon chunks on the host backend
(`ppf-cts-compute/cpu/host.rs`); `Atomic` and `Claim` run as ONE serial
ascending pass. So the row is what decides whether a kernel's threads can
overlap, and two different properties ride on it.

The UNSIGNED operations in `seam/seam_host.h` are genuinely atomic, so a
counter or a claim is safe from a Disjoint row: that is what an atomic is for,
and CUDA and MSL have said so all along.

The FLOAT add is deliberately a plain `+=`, and no atomic would fix what it
needs. A float fold is reproducible only if its ORDER is fixed, and an atomic
fixes indivisibility rather than order, so a concurrent float accumulation
would be correct and non-deterministic: two runs of one scene would differ.
The serial pass is what fixes both, which is why every float accumulator sits
on a non-Disjoint row and why THIS is the pairing worth a gate.

Failing here means one of two things. Either the row is wrong and the kernel
wants `Scatter::Atomic`, or the kernel should not be accumulating a float
through an atomic at all. It is never a reason to relax the check: the defect
it prevents is silent on CUDA, silent on Metal, and shows up as a scene that
does not reproduce its own results.
"""
import re, pathlib, sys, collections, subprocess, tempfile

ROOT = pathlib.Path("crates/ppf-cts-solver/src/kernels")
# THE NAME CONSTANTS ARE RENDERED HERE, not read out of a build tree, because a
# build tree is the wrong reading twice over. This gate runs in a CI job that
# compiles nothing, so on a clean checkout there was no tree to read at all and
# the step failed before a single row was checked, on every run in its history.
# And a tree an earlier build left behind is a state of the MACHINE rather than
# of the source: it carries constants for kernels since renamed and none for
# kernels since added, so the count assertion below reported on whoever last
# built here (measured: `parsed 285 scatter rows out of 286`).
# Rendering every neutral source is the complete reading, it runs the same
# generator build.rs runs rather than a second copy of its naming rule, and it
# costs about ten seconds.
GENERATOR = pathlib.Path("crates/ppf-cts-compute/seam/kernelgen.py")
if not GENERATOR.is_file():
    sys.exit(f"check-atomic-scatter: no such file: {GENERATOR}")

IDENT = re.compile(r"[A-Za-z_]\w*$")

# A neutral body spells every atomic `compute::atomic_*` with no type in the
# call, so the float ones are recognized by the POINTEE a reachable body
# declares. `atomic_float_t` is the seam name and the only spelling for it.
MARKER = "atomic_float_t"

def functions(src):
    """(name, body) for every brace block whose head parses as a signature."""
    out, depth, open_at = [], 0, None
    for i, ch in enumerate(src):
        if ch == "{":
            if depth == 0:
                open_at = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and open_at is not None:
                # walk back over the parameter list to the name
                j = open_at - 1
                while j >= 0 and src[j] in " \t\n\r":
                    j -= 1
                # an initializer list or a trailing `const`/`noexcept` is skipped
                if j >= 0 and src[j] != ")":
                    k = j
                    while k >= 0 and (src[k].isalnum() or src[k] in "_ \t\n:,"):
                        k -= 1
                    if k >= 0 and src[k] == ")":
                        j = k
                if j >= 0 and src[j] == ")":
                    d = 0
                    while j >= 0:
                        if src[j] == ")":
                            d += 1
                        elif src[j] == "(":
                            d -= 1
                            if d == 0:
                                break
                        j -= 1
                    head = src[:j].rstrip()
                    m = IDENT.search(head)
                    if m:
                        # FROM THE NAME, not from the opening brace. The atomic
                        # POINTEE is a parameter type, so it lives in the
                        # signature; storing the braces alone cannot see it,
                        # and a marker search over that text reports every
                        # accumulator clean.
                        out.append((m.group(0), src[m.start():i + 1]))
                open_at = None
    return out

def blank_comments(src):
    """`src` with every comment replaced by spaces of the same length.

    NEEDED FOR THE ENTRY SCAN AND NOT FOR THE BODIES. A comment that merely
    NAMES an attribute opens an attribute run for the scanner, and the head that
    follows a run is bounded only by the next `(`, so a mention several lines
    above a declaration swallows it: `re.finditer` does not overlap, and the
    real declaration underneath is never offered. Measured in
    primitives/vec_ops.kernel.cpp, whose prose names `[[seam::index]]` sixty
    lines above `vec_fill_u32`.

    Bodies are deliberately NOT masked: a marker inside a comment there makes
    this gate report a row it need not, which is the safe direction.
    """
    out = list(src)
    for m in re.finditer(r"/\*.*?\*/|//[^\n]*", src, re.S):
        for i in range(m.start(), m.end()):
            if out[i] != "\n":
                out[i] = " "
    return "".join(out)


bodies = collections.defaultdict(str)
entry_of = {}
for f in sorted(list(ROOT.rglob("*.kernel.cpp")) + list(ROOT.rglob("*.hpp"))):
    src = f.read_text()
    # [[seam::entry]] IMPLIES ITS RECORD, so [[seam::args]] is not written
    # beside it and a pattern demanding both matches nothing. This gate reports
    # a COUNT, and zero entries reports zero findings, so the shape of the
    # pattern decides whether it checks anything at all.
    #
    # THREE THINGS THIS PATTERN MUST NOT PIN. Demanding `void` immediately
    # after the attribute run skips every self-declaring body, since those
    # write `[[seam::device_fn]] inline void name(`: measured,
    # 117 entries seen against 286 declared, 59 percent of the tree unchecked
    # while the printed numbers looked healthy. It cannot pin `inline` either,
    # because an entry may RETURN a value (`aabb_leaf_face` answers `AABB`, and
    # a value-returning body gets a scatter sink like any other), so the name is
    # read as the LAST identifier before the parameter list and everything
    # before it is the return type. And it must read comment-masked source, for
    # the reason `blank_comments` gives.
    #
    # The count assertion below is what makes the next such miss loud. Widening
    # a pattern does not stop the next construct from falling outside it.
    for m in re.finditer(
            r"((?:\[\[seam::\w+(?:\([^)]*\))?\]\]\s*)+)([^;{}()]*?)\(",
            blank_comments(src)):
        if not re.search(r"\[\[seam::entry(?:\([^)]*\))?\]\]", m.group(1)):
            continue
        name = re.search(r"([A-Za-z_]\w*)\s*$", m.group(2))
        if name:
            entry_of[name.group(1)] = f.name
    # A visitor's method name is fixed by the traversal contract and is shared
    # across visitors on purpose, so names are UNIONED: for a reachability
    # question the union is the conservative reading.
    for name, body in functions(src):
        bodies[name] += body

CALL = re.compile(r"\b(\w+)\s*[\(<]")

ANY_ATOMIC = re.compile(r"compute::atomic_\w+")


def reaches_any_atomic(name, seen):
    """Does `name` reach ANY seam atomic, unsigned ones included.

    A DIFFERENT QUESTION FROM `reaches`, which asks only about the float. This
    one asks whether a serial row has an atomic to justify it at all.
    """
    if name in seen:
        return False
    seen.add(name)
    body = bodies.get(name, "")
    if ANY_ATOMIC.search(body):
        return True
    return any(reaches_any_atomic(c, seen) for c in sorted(set(CALL.findall(body)))
               if c != name and c in bodies)


def camel_args(entry):
    """The argument record the generator emits for `entry`, `<Camel>Args`."""
    return "".join(w.capitalize() for w in entry.split("_")) + "Args"


def reaches(name, stack, seen):
    if name in seen:
        return None
    seen.add(name)
    body = bodies.get(name)
    if not body:
        return None
    if MARKER in body:
        return stack + [name]
    for callee in sorted(set(CALL.findall(body))):
        if callee == name or callee not in bodies:
            continue
        got = reaches(callee, stack + [name], seen)
        if got:
            return got
    return None

names = {}
with tempfile.TemporaryDirectory() as tmp:
    for source in sorted(ROOT.rglob("*.kernel.cpp")):
        out = pathlib.Path(tmp) / source.relative_to(ROOT).with_suffix(".rs")
        out.parent.mkdir(parents=True, exist_ok=True)
        rendered = subprocess.run(
            [sys.executable, "-B", str(GENERATOR), "--target", "rust", "--emit",
             "entry", "--kernel-root", str(ROOT), "--out", str(out),
             str(source)], capture_output=True, text=True)
        if rendered.returncode != 0:
            sys.exit(f"check-atomic-scatter: kernelgen could not render "
                     f"{source}: {' '.join(rendered.stderr.split())}")
        for c, v in re.findall(r'pub const (\w+_NAME): &str = "(\w+)";',
                               out.read_text()):
            names[c] = v
kr = pathlib.Path("crates/ppf-cts-solver/src/driver/kernels.rs").read_text()
# A row commonly carries a comment between its name and its scatter, saying why
# that scatter was chosen, so the two are separated by "whitespace and line
# comments" rather than by whitespace. Reading only the adjacent form SKIPS
# every commented row, and a skipped row is a row this gate never checks, which
# is how a gate fails open: it reports clean over what it never read. The count
# assertion below is what makes a future regression here loud instead.
GAP = r"(?:\s|//[^\n]*\n)*"
scat = {}
for c, sc in re.findall(
        r"decl_generated\w*\(" + GAP + r"id::\w+," + GAP +
        r"(\w+_NAME)," + GAP + r"Scatter::(\w+)", kr):
    if c in names:
        scat[names[c].removesuffix("_entry")] = sc

declared = len(re.findall(r"decl_generated\w*\(" + GAP + r"id::\w+," + GAP +
                          r"\w+_NAME," + GAP + r"Scatter::", kr))
if len(scat) != declared:
    sys.exit(f"check-atomic-scatter: parsed {len(scat)} scatter rows out of "
             f"{declared} declarations. Every unparsed row is one this gate "
             f"silently does not check, so the parse is the check.")

disjoint = [e for e in entry_of if scat.get(e) == "Disjoint"]
bad = []
for entry in sorted(disjoint):
    path = reaches(entry, [], set())
    if path:
        bad.append((entry, entry_of[entry], path))

# AN EMPTY READING IS A FAILURE, and this gate learned it the expensive way.
# It reports a COUNT of offending entries, so a parser that matches nothing
# reports zero findings and exits clean: when [[seam::entry]] stopped being
# written beside [[seam::args]], this saw 0 of 284 entries and still passed. It
# was a sibling gate failing loudly that caught it, which is luck rather than
# coverage. Neither number below can legitimately be zero in this tree.
if not entry_of or not bodies:
    sys.exit(f"check-atomic-scatter: parsed {len(bodies)} bodies and "
             f"{len(entry_of)} entries, and a comparison needs both. Either "
             f"the declarations changed shape or this parser is broken; it is "
             f"not a pass")
# AND NON-EMPTY IS NOT ENOUGH, which is the second half of the same lesson. The
# guard above refuses only ZERO, so a pattern matching MOST of the tree passes
# it while silently skipping the rest, and that is what happened here: 117 of
# 286, printed as a clean result for as long as it stood. Every declared row has
# an entry by construction and `declared` is read from a table this gate already
# parses completely, so the two numbers are equal or this parser is short.
# Do NOT soften this to an inequality or to a tolerance.
missing = sorted(set(scat) - set(entry_of))
if len(entry_of) != declared:
    sys.exit(f"check-atomic-scatter: parsed {len(entry_of)} entry "
             f"declarations against {declared} rows in kernels.rs"
             + (f", missing {', '.join(missing[:12])}" if missing else "")
             + ". Every unmatched declaration is a kernel this gate does not "
             f"check, and it would still print a clean count. Widen the entry "
             f"pattern above rather than this comparison.")
print(f"bodies parsed: {len(bodies)}   entries: {len(entry_of)}   "
      f"rows resolved: {len(scat)}   Disjoint: {len(disjoint)}")
print(f"\nDisjoint entries that reach a FLOAT seam atomic: {len(bad)}")
for e, f, path in bad:
    print(f"  {e}\n      {f}: {' -> '.join(path)}")
if bad:
    print("\nEither the row wants Scatter::Atomic, or the kernel should not be\n"
          "accumulating a float through an atomic. See this file's docstring.")
# ---------------------------------------------------------------------------
# THE OTHER DIRECTION: A PLACEHOLDER ROW THAT HAS ACQUIRED A CALLER.
#
# THE RULE THIS AUTOMATES: an undispatched kernel's `Scatter` and per-item cost
# are placeholders, and they become load-bearing the day something launches it,
# so the row is read against the BODY when a kernel is given its first caller.
# Nothing checked that, and the day arrived:
# `vec_fill_u32`'s body is `array[index] = value;` at its own thread
# index, its row said `Scatter::Atomic`, and the driver had grown SIX dispatch
# sites (contact, lbvh, dyncsr, schwarz, devsort, intersection), each running a
# full vertex-count fill single-threaded on the host backend for no reason.
#
# WHAT THIS CAN AND CANNOT DECIDE. It cannot derive the row: `bitonic_step`
# writes `key[partner]` as well as its own slot and reaches no atomic at all, so
# "no atomic" does not imply disjoint, and `override_velocity_seed_listed` is
# serial by a contract its declaration cannot carry (a keyframe index list that
# nothing proves holds each vertex once). What it CAN do is notice that a serial
# row with no atomic behind it has acquired a dispatch, and make someone look.
#
# A kernel is DISPATCHED when the driver names its generated argument record
# outside the two tables that declare every kernel. That is how a dispatch is
# spelled: `device.launch("label", &args, count)` identifies the kernel by the
# record's type, never by its id.
DRIVER = pathlib.Path("crates/ppf-cts-solver/src/driver")
if not DRIVER.is_dir():
    sys.exit(f"check-atomic-scatter: no such directory: {DRIVER}. The dispatch "
             f"reading below would find nothing and report every placeholder "
             f"clean, so this is not a pass.")
driver_text = "\n".join(
    f.read_text() for f in sorted(DRIVER.rglob("*.rs"))
    if f.name not in ("kernels.rs", "launch.rs"))

# Serial rows that reach no atomic AND are dispatched. Each needs a reason here,
# because the row is load-bearing the moment it has a caller. Adding a name to
# this table is a claim that someone read the body; it is not a way past the
# check.
SERIAL_WITHOUT_ATOMIC = {
    "override_velocity_seed_listed":
        "SERIAL BY CONTRACT, and the declaration cannot say so. Both seeds "
        "write `prev[vi]` at a slot read from a keyframe index list, and "
        "nothing proves that list holds each vertex at most once. The linear "
        "seed is idempotent under a duplicate; the ANGULAR one reads `prev` and "
        "writes it back, so a duplicate makes the result depend on which write "
        "lands last, and under a parallel partition it is a data race. "
        "main/override_seed.kernel.cpp carries the argument and says BOTH ROWS "
        "MUST KEEP IT. The work is a keyframe-sized subset of the vertices, so "
        "serial costs nothing measurable.",
    "override_angular_seed_listed":
        "The other half of the pair above, and the half the argument is really "
        "about: it accumulates onto what the linear seed left, so a duplicated "
        "index is order-dependent rather than idempotent. Listed separately "
        "because the two rows are separate and a future edit could move one.",
    "pcg_beta_resident":
        "ONE ELEMENT. The PCG beta update is dispatched over a single element "
        "(driver/pcg.rs), so no two threads exist to overlap and the row costs "
        "nothing either way. It writes several scalars through pointers rather "
        "than one slot at a thread index, so Disjoint would also be a claim "
        "this gate cannot check.",
}

unjustified = []
for entry in sorted(entry_of):
    if scat.get(entry) in (None, "Disjoint"):
        continue
    if reaches_any_atomic(entry, set()):
        continue
    if not re.search(r"\b" + camel_args(entry) + r"\b", driver_text):
        continue          # a placeholder with no caller, which is the fine case
    if entry not in SERIAL_WITHOUT_ATOMIC:
        unjustified.append(entry)

print(f"serial rows with no atomic behind them, and dispatched: "
      f"{len(SERIAL_WITHOUT_ATOMIC) + len(unjustified)} "
      f"({len(unjustified)} unjustified)")
for e in unjustified:
    print(f"  {e}\n      {entry_of[e]}: row is Scatter::{scat[e]}, reaches no "
          f"seam atomic, and the driver dispatches it")
if unjustified:
    print("\nA Scatter row is a placeholder only while nothing launches the\n"
          "kernel. These have callers, so read each body: if it writes one slot\n"
          "at its own thread index the row is Disjoint, and if it is serial for\n"
          "a reason the declaration cannot carry, say so in\n"
          "SERIAL_WITHOUT_ATOMIC above.")

sys.exit(1 if bad or unjustified else 0)
