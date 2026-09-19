#!/usr/bin/env python3
"""Put the kernel ids, the driver TABLE and the LAUNCH table back in canonical
order, which is one sorted walk of the neutral tree then declaration order.

# When to run this

After adding, removing or renaming a `[[seam::args]] [[seam::entry]]`
declaration. The canonical order is a SORTED WALK of `crates/ppf-cts-solver/src/kernels`,
not the build's `KERNELS` list, so a file added mid-walk or an entry deleted from
the middle of one moves the id of everything after it. A kernel id is a TABLE
INDEX: `Device::decl` is `table.get(id)`, so a row out of position hands one
kernel's bytes to another's launcher.

# Why it is a script and not a hand edit

Measured: adding one file mid-walk put 93 of 218 rows out of position. Doing that
by hand is a silent wrong-answer edit on any row whose record happens to be the
same SIZE as its neighbour's, because `check_shape` catches only a size
mismatch.

# What checks it

`check-shared-wiring.py` rule 13 fails when the driver TABLE has drifted from
this order, and `the_table_is_indexed_by_its_own_ids` fails when a row is not at
the position its own id names. Run both after this; neither is a substitute for
the other.
"""
import os, re, sys, pathlib

ROOT = pathlib.Path(".")
KDIR = ROOT / "crates/ppf-cts-solver/src/kernels"
# [[seam::entry]] implies its record, so the attributes are matched as a RUN and
# the run is required to carry `entry`. Demanding [[seam::args]] first matches
# nothing, and this script REWRITES three tables from what it matches.
ENTRY_DECL_RE = re.compile(
    r"((?:\[\[seam::\w+(?:\([^)]*\))?\]\]\s*)+)void\s+(\w+)\s*\(")


def entry_names(text):
    return [m.group(2) for m in ENTRY_DECL_RE.finditer(text)
            if re.search(r"\[\[seam::entry(?:\([^)]*\))?\]\]", m.group(1))]

def _entries_of(path):
    """Entry names in declaration order, read through the transcompiler.

    An entry is declared in two spellings, a separate declaration and a body
    that declares itself, and a regex keyed on either is blind to the other.
    This script REWRITES three tables from what it finds, so blindness here
    erases them rather than reordering them.
    """
    import importlib.util
    gen = ROOT / "crates/ppf-cts-compute/seam/kernelgen.py"
    spec = importlib.util.spec_from_file_location("kernelgen_renumber", gen)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.KERNEL_ROOT = os.path.abspath(str(KDIR))
    return [e.name for e in module.read_source(str(path))[3] if e.emit_entry]


sources = sorted(str(p.relative_to(KDIR)) for p in KDIR.rglob("*.kernel.cpp"))
canon = []
for rel in sources:
    canon.extend(_entries_of(KDIR / rel))
if not canon:
    sys.exit("renumber-kernel-ids: no entry declarations matched. This script "
             "REWRITES the id, table and launch tables from what it matches, so "
             "an empty reading would erase them rather than reorder them")
print(f"canonical entries: {len(canon)}")

kp = ROOT / "crates/ppf-cts-solver/src/driver/kernels.rs"
ks = kp.read_text()

# NAME constant -> entry symbol, from the generated renderings.
names = {}
for g in pathlib.Path("target/release/build").glob("ppf-cts-solver-*/out/kernelgen/**/*.entry.rs"):
    for c, v in re.findall(r'pub const (\w+_NAME): &str = "(\w+)";', g.read_text()):
        names[c] = v

# TABLE rows, each `decl_generated*(  id::X,  X_NAME, ...)` through its `    ),`
GAP = r"(?:\s|//[^\n]*\n)*"
row_re = re.compile(r"[ \t]*decl_generated\w*\(" + GAP + r"id::(\w+)," + GAP + r"(\w+_NAME),.*?\n[ \t]*\),\n", re.S)
rows = list(row_re.finditer(ks))
print(f"TABLE rows: {len(rows)}")
by_entry = {}
for m in rows:
    sym = names.get(m.group(2))
    if sym is None:
        sys.exit(f"no NAME constant for {m.group(2)}")
    by_entry[sym.removesuffix("_entry")] = (m.group(1), m.group(0))
missing = [c for c in canon if c not in by_entry]
extra = [e for e in by_entry if e not in canon]
if missing or extra:
    sys.exit(f"TABLE and the tree disagree: missing={missing} extra={extra}")

# 1. TABLE, rewritten in canonical order.
start, end = rows[0].start(), rows[-1].end()
ks = ks[:start] + "".join(by_entry[c][1] for c in canon) + ks[end:]

# 2. the id constants, renumbered to canonical position.
want = {by_entry[c][0]: i for i, c in enumerate(canon)}
def setid(m):
    return f"{m.group(1)}{want[m.group(2)]})" if m.group(2) in want else m.group(0)
ks = re.sub(r"(pub const (\w+): KernelId = KernelId\()\d+\)",
            lambda m: f"pub const {m.group(2)}: KernelId = KernelId({want[m.group(2)]})"
                      if m.group(2) in want else m.group(0), ks)
ks = re.sub(r"pub const COUNT: usize = \d+;", f"pub const COUNT: usize = {len(canon)};", ks)
kp.write_text(ks)
print("kernels.rs: TABLE reordered and ids renumbered")

# 3. LAUNCH, in the same order.
lp = ROOT / "crates/ppf-cts-solver/src/driver/launch.rs"
ls_ = lp.read_text()
# THE LAUNCHER'S NAME FOLLOWS THE ENTRY'S, which `check-launch-seam.py` holds
# to and which the thunks are generated from: they live in OUT_DIR now, so
# parsing `launch.rs` for them finds none. Deriving the name is what the
# generator itself does.
want_launch = [f"launch_{c}" for c in canon]
m = re.search(r"(static LAUNCH: \[Launch; id::COUNT\] = \[\n)(.*?)(\n\];)", ls_, re.S)
if not m:
    sys.exit("renumber-kernel-ids: cannot find `static LAUNCH` in driver/launch.rs")
have = [line.strip().rstrip(",") for line in m.group(2).split("\n")
        if line.strip() and not line.strip().startswith("//")]
if sorted(have) != sorted(want_launch):
    sys.exit(f"LAUNCH set differs: only-in-table={sorted(set(have)-set(want_launch))} "
             f"only-in-canon={sorted(set(want_launch)-set(have))}")
ls_ = (ls_[:m.start(2)]
       + ",\n".join(f"    {t}" for t in want_launch) + ","
       + ls_[m.end(2):])
lp.write_text(ls_)
print("launch.rs: LAUNCH reordered")
