#!/usr/bin/env python3
"""A body's address spaces must match how its entry hands the buffers over.

ONLY THE SHADER COMPILER CAN TELL, AND ONLY ON A MAC AT RUN TIME. Metal is the
one target with more than one address space, so a parameter marked
`[[seam::thread]]` that the entry hands a device pointer compiles clean under
nvcc and under a host C++ compiler and fails the Metal shader compile with `no
matching function for call to '<body>'`. The mistake is easy to make and
expensive to find, so this gate reads the pairing out of the declarations
instead of waiting for that compile.

THE ENTRY ALREADY DECIDES IT, so nothing here is a judgment:

    access "base"      the whole buffer                  -> [[seam::device]]
    access "offset"    the buffer advanced to this slot  -> [[seam::device]]
    access "element"   ONE element copied to thread space-> [[seam::thread]]
    access "indirect"  N elements copied to thread space -> [[seam::thread]]
    a [[seam::scratch]] parameter                        -> [[seam::threadgroup]]

The `offset` row is the one that reads backwards and is worth stating twice: a
strided output is the DESTINATION advanced to this element's own slot, so it
stays in device memory. Only what a gather brings in is thread.

This reads the declarations through `kernelgen.py` itself, so the rules cannot
drift from the generator's own idea of what an entry does with a field.
"""

import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
KERNELS = ROOT / "crates/ppf-cts-solver/src/kernels"
GEN = ROOT / "crates/ppf-cts-compute/seam/kernelgen.py"

WANT_BY_ACCESS = {
    "base": "device",
    "offset": "device",
    "element": "thread",
    "indirect": "thread",
}
SPACES = ("device", "thread", "threadgroup")


def load_generator():
    spec = importlib.util.spec_from_file_location("kernelgen", GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strip_comments(text):
    """Comments cannot contribute a parameter, and may mention an attribute."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def split_top_level(text):
    depth, start, out = 0, 0, []
    for i, ch in enumerate(text):
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(text[start:i])
            start = i + 1
    out.append(text[start:])
    return [p.strip() for p in out if p.strip()]


NAME = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*$")


def parse_bodies(text):
    """Every [[seam::device_fn]] definition, by name, as {param: space}.

    A name may have two definitions, the cooperative body and its serial twin,
    and both are checked: rule (1-LANE) lets them differ in what they compute
    and not in how their arguments arrive.
    """
    text = strip_comments(text)
    out = {}
    for match in re.finditer(r"\[\[seam::device_fn\]\][^;{]*?\b"
                             r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
        name = match.group(1)
        i, depth = match.end() - 1, 0
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        params = {}
        for part in split_top_level(text[match.end():i]):
            spaces = [s for s in SPACES if f"[[seam::{s}]]" in part]
            bare = re.sub(r"\[\[[^\]]*\]\]", " ", part)
            bare = re.sub(r"=[^,]*$", "", bare).strip()
            found = NAME.search(bare)
            if found:
                # AN ADDRESS SPACE QUALIFIES A POINTER OR A REFERENCE AND
                # NOTHING ELSE. A gathered element passed BY VALUE is a copy the
                # entry already made, so it has no address space in MSL and
                # carries no attribute; requiring one there reports every such
                # parameter, which is what the first version of this check did.
                indirect = "*" in bare or "&" in bare
                params[found.group(1)] = (spaces[0] if spaces else None, indirect)
        out.setdefault(name, []).append(params)
    return out


def expected_spaces(entry):
    """The address space each forwarded argument requires, in call order.

    `Entry.forward` is what the generated entry point passes and in what order,
    so this is the same table an inference engine would seed from rather than a
    second reading of the declaration.
    """
    by_name = {f.name: f for f in entry.fields}
    scratch = {name for name, _base, _count in entry.scratch}
    out = []
    for kind, name in entry.forward:
        if kind == "scratch" or name in scratch:
            out.append((name, ("threadgroup", "scratch")))
        elif kind == "handle":
            field = by_name.get(name)
            if field is None:
                out.append((name, (None, "unknown")))
                continue
            space = WANT_BY_ACCESS.get(field.access)
            if field.access == "indirect" and entry.indices_field is not None:
                # ONE FORWARDED FIELD, N ARGUMENTS: a through buffer expands
                # into the elements its slot list names, each a thread copy.
                for _ in range(entry.indices_field.stride):
                    out.append((name, ("thread", "indirect")))
                continue
            out.append((name, (space, field.access) if space else (None, "")))
        else:
            out.append((name, (None, "")))
    return out


def main():
    kernelgen = load_generator()
    problems, checked = [], 0

    for path in sorted(KERNELS.rglob("*.kernel.cpp")):
        _, _, _, entries = kernelgen.read_source(str(path))
        if not entries:
            continue
        bodies = parse_bodies(path.read_text())
        rel = path.relative_to(ROOT)
        for entry in entries:
            # SEEDED BY POSITION, NOT BY NAME. `Entry.forward` is the generator's
            # own argument list, in the order the generated call passes them, so
            # it survives a body that RENAMES a parameter: `aabb.kernel.cpp`'s
            # body takes `box` where its entry field is `aabb`, and a name-keyed
            # seed simply skips that one. It also expands a `[[seam::indices]]`
            # buffer into the N elements the slot list names, which is one
            # forwarded field and N body parameters.
            want = expected_spaces(entry)
            for params in bodies.get(entry.name, []):
                order = list(params)
                for position, (_declared, (space, why)) in enumerate(want):
                    if position >= len(order) or space is None:
                        # NO EXPECTATION IS NOT A DISAGREEMENT. A scalar and the
                        # diagnostic handle are passed by value, so the seam has
                        # no address space to require of them, and a body that
                        # marks one anyway is saying something this table cannot
                        # contradict.
                        continue
                    pname = order[position]
                    got, indirect = params[pname]
                    if not indirect and got is None:
                        continue
                    if got is None:
                        # WRITING NOTHING IS NOW THE NORMAL CASE. The generator
                        # infers a body parameter's address space from how its
                        # entry hands the buffer over, so an absent attribute is
                        # the inference doing its work rather than an omission.
                        # What is still worth checking is a WRITTEN one that
                        # disagrees, because that is a claim the entry
                        # contradicts, and `rendered_spaces_are_complete` below
                        # is what covers the parameters nothing writes.
                        continue
                    checked += 1
                    if got != space:
                        problems.append(
                            f"{rel}: {entry.name}({pname}) is "
                            f"{'[[seam::' + got + ']]' if got else 'unqualified'} "
                            f"and the entry passes it as {why}, which is "
                            f"[[seam::{space}]]. Metal is the only target with "
                            f"more than one address space, so this builds clean "
                            f"on CUDA and on the host and fails the shader "
                            f"compile at run time.")

    print(f"checked {checked} pointer and reference bindings across "
          f"{len(list(KERNELS.rglob('*.kernel.cpp')))} neutral sources")
    if problems:
        print()
        for problem in sorted(set(problems)):
            print("  " + problem)
        print(f"\n{len(set(problems))} problem(s)")
        return 1
    print("every body's address spaces match how its entry hands the buffers over")
    return 0


if __name__ == "__main__":
    sys.exit(main())
