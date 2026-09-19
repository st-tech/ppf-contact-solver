#!/usr/bin/env python3
"""The Rust seam must agree with the entry declarations.

THREE STEPS OF A KERNEL CONVERSION SIT AT THE SEAM RATHER THAN IN THE BODY, and
only one of them fails at the build. `driver/launch.rs` declares an `extern` for
every generated entry point and picks a thunk macro for it, and BOTH are written
RENDERED from the declaration, both being decided entirely by facts the
generator already computes: whether the entry is group-shaped and whether it
takes a diagnostic channel.

SO THIS CHECKS THE RENDERING, AND `launch.rs` FOR THE ABSENCE OF A HAND-WRITTEN
ONE. Generating a seam does not retire the class of defect below, it moves where
the defect would have to come from: the renderer choosing a macro or an arity
against an entry's shape. That choice is recomputed here from the entry rather
than read from the renderer, so the two derivations have to agree. And a thunk
written back into `launch.rs` by hand is refused outright, because a partly
hand-written seam is exactly the state the generation removed.

WHAT GOES WRONG WITHOUT THIS CHECK IS NOT A COMPILE ERROR. A group entry's
symbol takes an extra `group_width` between the arena base and the range, so a
four-parameter declaration against a five-parameter definition reads the group
width as `begin` and whatever follows as `end`. Rust cannot see it: the
declaration is what it believes. The symptom is a SIGSEGV in an unrelated test,
which is the good case, and silently wrong bounds otherwise. Picking the plain
thunk for a group entry fails the same way from the other side, passing no width
at all.

THE RULES, and they are the whole of it:

    args, arena_base, [group_width if group,] begin, end, [diag if diag]

    generated_thunk_group!  when the entry is group-shaped
    generated_thunk_diag!   when it takes a [[seam::diag]] channel
    generated_thunk!        otherwise

This reads the declarations through `kernelgen.py` itself rather than
re-parsing them, so the check cannot drift from the generator's own idea of what
an entry is.
"""

import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
KERNELS = ROOT / "crates/ppf-cts-solver/src/kernels"
LAUNCH = ROOT / "crates/ppf-cts-solver/src/driver/launch.rs"
GEN = ROOT / "crates/ppf-cts-compute/seam/kernelgen.py"

BASE_HEAD = ["args: *const u8", "arena_base: *const *mut u8"]
BASE_TAIL = ["begin: u32", "end: u32"]
GROUP_PARAM = "group_width: u32"
DIAG_PARAM = "diag: *mut DiagRecord"


def load_generator():
    spec = importlib.util.spec_from_file_location("kernelgen", GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected(entry):
    """The extern parameter list and thunk macro this entry requires."""
    params = list(BASE_HEAD)
    if entry.is_group:
        params.append(GROUP_PARAM)
    params += BASE_TAIL
    if entry.diag_name:
        params.append(DIAG_PARAM)
    if entry.is_group and entry.diag_name:
        macro = None          # no such macro exists; reported as a finding
    elif entry.is_group:
        macro = "generated_thunk_group"
    elif entry.diag_name:
        macro = "generated_thunk_diag"
    else:
        macro = "generated_thunk"
    return params, macro


def rendered_seam(kernelgen):
    """The externs and thunks the generator renders, by symbol name.

    Rendering per source rather than reading `OUT_DIR` keeps the check runnable
    without a build, and reads the same function `build.rs` calls.
    """
    externs, thunks = {}, {}
    for path in sorted(KERNELS.rglob("*.kernel.cpp")):
        _, _, _, entries = kernelgen.read_source(str(path))
        emitted = [entry for entry in entries if entry.emit_entry]
        if not emitted:
            continue
        for name, params in parse_launch(
                kernelgen.render_externs_rust(str(path), emitted))[0].items():
            externs[name] = params
        for name, macro in parse_launch(
                kernelgen.render_thunks_rust(str(path), emitted))[1].items():
            thunks[name] = macro
    return externs, thunks


def declared_entries(kernelgen):
    """Every generated entry point in the neutral tree, by symbol name."""
    out = {}
    for path in sorted(KERNELS.rglob("*.kernel.cpp")):
        _, _, _, entries = kernelgen.read_source(str(path))
        for entry in entries:
            if not entry.emit_entry:
                continue
            out[entry.name + "_entry"] = (entry, path.relative_to(ROOT))
    return out


# A declaration is written on one line or spread over several, and both forms
# occur. Reading only the multi-line form silently reports the compact ones as
# absent, which is a false finding in the direction that hides a real one.
FN_OPEN = re.compile(r"^\s+fn ([A-Za-z0-9_]+_entry)\(\s*$")
FN_ONE = re.compile(r"^\s+fn ([A-Za-z0-9_]+_entry)\((.*)\);\s*$")
PARAM = re.compile(r"^\s+([A-Za-z0-9_]+: [^,]+),\s*$")
THUNK = re.compile(
    r"generated_thunk(_group|_diag)?!\(\s*"
    r"([A-Za-z0-9_]+)\s*,\s*kernels::([A-Za-z0-9_]+)\s*,\s*"
    r"([A-Za-z0-9_]+_entry)\s*,?\s*\)", re.S)


def parse_launch(text):
    externs, current, params = {}, None, []
    for line in text.splitlines():
        if current is None:
            flat = FN_ONE.match(line)
            if flat:
                externs[flat.group(1)] = [
                    part.strip() for part in flat.group(2).split(",")
                    if part.strip()]
                continue
            match = FN_OPEN.match(line)
            if match:
                current, params = match.group(1), []
            continue
        if line.strip() == ");":
            externs[current] = params
            current = None
            continue
        match = PARAM.match(line)
        if match:
            params.append(match.group(1))
    thunks = {}
    for match in THUNK.finditer(text):
        suffix = match.group(1) or ""
        thunks[match.group(4)] = "generated_thunk" + suffix
    return externs, thunks


def main():
    kernelgen = load_generator()
    entries = declared_entries(kernelgen)
    externs, thunks = rendered_seam(kernelgen)

    problems = []
    # A WALL AT ZERO. The seam is generated whole, so any occurrence here is a
    # hand-written one creeping back, which is the state that let a four-param
    # declaration stand against a five-param definition.
    written = parse_launch(LAUNCH.read_text())
    for symbol in sorted(written[0]):
        problems.append(
            f"driver/launch.rs declares {symbol} by hand. The extern block is "
            f"rendered by `kernelgen.py --emit externs`; delete the declaration.")
    for symbol in sorted(written[1]):
        problems.append(
            f"driver/launch.rs writes a thunk for {symbol} by hand. The thunks "
            f"are rendered by `kernelgen.py --emit thunks`; delete it.")
    for symbol, (entry, source) in sorted(entries.items()):
        want_params, want_macro = expected(entry)
        if want_macro is None:
            problems.append(
                f"{source}: {symbol} is BOTH group-shaped and diagnostic, and "
                f"no thunk macro spells that pairing. Add one or split the entry.")
            continue
        got_params = externs.get(symbol)
        if got_params is None:
            problems.append(
                f"{source}: {symbol} gets no extern from `--emit externs`.")
        elif got_params != want_params:
            problems.append(
                f"{source}: {symbol} renders ({', '.join(got_params)}) and the "
                f"definition takes ({', '.join(want_params)}). Rust believes the "
                f"declaration, so this miscounts the arguments at run time "
                f"rather than failing to build.")
        got_macro = thunks.get(symbol)
        if got_macro is None:
            problems.append(
                f"{source}: {symbol} gets no thunk from `--emit thunks`.")
        elif got_macro != want_macro:
            problems.append(
                f"{source}: {symbol} uses {got_macro}! and its shape needs "
                f"{want_macro}!.")

    print(f"checked {len(entries)} generated entry points against "
          f"{len(externs)} rendered extern declarations and {len(thunks)} "
          f"rendered thunks, and driver/launch.rs for a hand-written seam")
    if problems:
        print()
        for problem in problems:
            print("  " + problem)
        print(f"\n{len(problems)} problem(s)")
        return 1
    print("the rendered seam agrees with every entry declaration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
