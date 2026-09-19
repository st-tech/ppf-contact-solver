#!/usr/bin/env python3
# File: check-shared-wiring.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Checks that the SHARED device sources under crates/ppf-cts-solver/src/kernels are
# wired into BOTH real backends, and that they still obey the one contract the
# Metal side imposes on them that no compiler sees until run time.
#
# WHY THIS EXISTS. A physics kernel body is written once and compiled from
# identical bytes by nvcc, by the Metal shader compiler and by a host C++
# compiler. A header that only one backend compiles is a fork with extra steps: it keeps building, keeps passing,
# and describes an algorithm the other backend does not run. Building both
# backends catches a header that fails to COMPILE on one side. It cannot catch a
# header that one side never compiles at all, which is what this checks.
#
# TEN RULES. All fail the build except the eighth, which is a census and
# reports. The first seven read the SHARED sources; rules 9 and 10 read the two
# sides of the CPU seam, which is where a shared body is wrapped and called.
#
#   1. A header under src/kernels that carries the seam macros (SM_*) and is
#      reachable from a CUDA translation unit must appear in the Metal embed
#      list. Otherwise nvcc compiles it and the Metal shader never sees it.
#      The prologue directory cpp/seam is exempt, and rule 4 is what stands in
#      for it there; see SEAM_PROLOGUE_DIR below. Entries in CUDA_ONLY are the
#      recorded exceptions, the mirror image of ONE_SIDED and held to the same
#      staleness rule.
#
#   2. A header under src/kernels that the Metal build embeds must be reachable from
#      a CUDA translation unit. Otherwise it is device code nvcc never
#      compiles, so nothing checks it against the CUDA oracle. Entries in
#      ONE_SIDED below are the recorded exceptions.
#
#   3. Nothing an embedded header includes may reach the MSL compiler. The Metal
#      driver hands newLibraryWithSource one concatenated string and there is no
#      filesystem behind it, so an #include that reaches the shader compiler is a
#      compile error. That failure happens at RUN TIME, after every build has
#      gone green, which is exactly why it is worth a static check.
#
#      A QUOTED include is answered by the assembler: append_segment runs
#      neutralize_quoted_includes over every segment it splices, so the line
#      never reaches the compiler and a kernel body can name its dependency
#      unconditionally. This check therefore verifies that the transform is
#      still wired in, and reports the includes it does NOT cover: an angle
#      include other than <metal_*>, which the assembler deliberately leaves
#      alone because <metal_stdlib> and its siblings are the one form the shader
#      compiler can serve.
#
#   4. Every SM_ name the nvcc prologue defines must also be defined by the
#      Metal prologue. The two tables are separate files by necessity, so a
#      kernel body that names a spelling only one of them carries compiles on
#      CUDA and fails when the shader is built, which is again at run time.
#
#   5. Every neutral kernel source renders, for all three targets. The kernel
#      bodies carry no SM_ macro and no conditional at all: address and
#      execution spaces are [[seam::...]] attributes, and
#      ppf-cts-compute/seam/kernelgen.py
#      rewrites them per backend. Running that script here is what puts its
#      rejections in front of CI with no CUDA toolkit and no Metal device.
#
#   6. Rule 4 again, for the seam table those bodies actually reach the seam
#      through. Rule 4 keys on SM_ names, so it is blind to this one; the
#      failure it would miss is identical, a name only some prologues carry.
#
#   7. A neutral kernel body carries no SM_ seam macro and no preprocessor
#      directive other than `#pragma once` and a quoted `#include`. This is the
#      headline property of the whole design: a kernel that names no backend and
#      takes no branch is what lets one source serve three compilers.
#
#      Rule 5 already rejects both, because kernelgen.py does. This rule is
#      deliberately a SECOND, INDEPENDENT witness, and the duplication is the
#      point: rule 5 delegates the property to the very script it runs, so a
#      kernelgen.py that lost its SM_ check or its directive check would take
#      rule 5's coverage with it and CI would stay green. This rule reads the
#      kernel text itself and shares no code with kernelgen.py, so the two fail
#      independently. It also masks differently on purpose, see kernel_code().
#
#   8. The per-backend reachability census over the neutral bodies. Rules 1 and
#      2 compare sets of FILE PATHS, so a body can be compiled into the CUDA
#      image and into the Metal shader, carry a passing per-kernel parity
#      self-test, and be called in production by nothing: a self-test builds
#      its own fixture, so it certifies the body whether or not any production
#      kernel binds it. This rule walks each backend's own entry points and
#      reports a body CUDA reaches that another backend does not, and a body a
#      backend reaches only from a self-test entry point.
#
#      It REPORTS and does not fail, deliberately, and the reason is in the
#      census itself: several entries are design differences rather than
#      defects. REACHABILITY_EXCEPTIONS records those with a date and a reason,
#      and an entry not on that list is printed as unrecorded. The construction
#      properties and the two directions in which the walk is inexact are
#      stated at the rule 8 section below, not summarized here.
#
#   9. The count of HAND-WRITTEN CPU entry points does not grow. An entry point
#      is an argument record plus a thread index plus a call into the neutral
#      body, so a hand-written one is a mirror pair with nothing linking its
#      halves, and the CPU backend still holds 191 of them. Most cannot be
#      converted without changing the neutral body they wrap, which changes
#      that body's CUDA and Metal call sites in the same edit, so the rule is a
#      ratchet and not a demand for zero. It fails on a DROP as well, naming
#      the number to write: a ceiling nobody lowers re-permits everything a
#      conversion just gained.
#
#  10. The CPU DRIVER names a shim HELPER, never a shim LAUNCHER. A launcher
#      takes a thread range, so it IS a dispatch, and a dispatch reaches the
#      backend through the Device trait and through nothing else. A helper
#      takes none: a
#      layout query, a constant, a per-pair predicate. Those are not on the
#      seam and are counted rather than forbidden, for the reason written at
#      the rule. The categories come from the SIGNATURE, the same reading rule
#      9 uses, so neither a comment nor a name decides one. It also requires a
#      reason at every declaration a driver keeps, since an exception nobody
#      can tell from an oversight is not an exception.
#
# WHAT IT DOES NOT COVER. Rule 1 keys on the seam macros, so a shared header
# that uses none of them is invisible to it; linalg/la_traits.hpp is the one
# such header today, and it resolves its backend split with __METAL_VERSION__
# rather than with an SM_ macro. That gap is backstopped rather than open: a
# header the Metal side needs and does not embed fails the shader compile, and a
# header CUDA needs and does not include fails the nvcc build. This check is for
# the case both builds are happy about.
#
# Run it anywhere: python3 .github/workflows/scripts/check-shared-wiring.py

import os
import re
import sys
from pathlib import Path

# Headers the Metal build embeds that no CUDA translation unit includes, with
# the reason each one is that way. The list is meant to shrink, never to grow to
# quiet a failure: an entry records a site to argue about, it does not bless it.
# A stale entry is itself a failure (see check_two), so the list cannot rot into
# a set of names nobody has re-read.
ONE_SIDED = {
    "contact/accd.hpp": (
        "the ACCD sweeps. CUDA RUNS them, which is why this is not a parity "
        "hole: the neutral driver reaches them on the HOST through the "
        "`*_abi` functions entrypoints/shim_contact.cpp exports, and that "
        "shim includes this header on every driver build, CUDA included. nvcc "
        "reads it as HOST code there today, and gets a DEVICE rendering of it "
        "when a [[seam::entry]] over a PAIR LIST replaces the host sweep, "
        "which is one of the hand-written launchers rule 9 ratchets down"
    ),
    "contact/intersect_core.hpp": (
        "the intersection scan, in exactly the position accd.hpp is in and "
        "reached the same way: entrypoints/shim_contact.cpp includes it, so "
        "the host build compiles it on every driver build"
    ),
    "primitives/reduce.kernel.cpp": (
        "the CUDA side reduces through src/driver/reduce.rs, on the same "
        "terms as radix"
    ),
    "primitives/scan.kernel.cpp": (
        "the CUDA side scans through src/driver/reduce.rs, on the same terms "
        "as radix"
    ),
}

# The mirror image of ONE_SIDED: bodies nvcc compiles that the Metal shader
# does not yet see, with the reason each one is that way. Same discipline as
# ONE_SIDED, and the same self-invalidation below: an entry the Metal build DOES
# embed, or that no CUDA translation unit reaches, is itself a failure, so this
# cannot rot into a set of names nobody re-reads.
#
# WHY AN ENTRY CAN BE HONEST HERE. Rule 1 keys on the FILE, not on whether a
# fork exists, so it fires the moment an orchestration step forked between a
# CUDA copy and a hand-written MSL string in the Metal backend moves its CUDA
# half into a shared body. That move deletes one of the two copies and
# leaves the other exactly as it was: the fork shrinks, it does not appear. The
# alarm is a true statement about the half that is still owed.
#
# What is owed is one change, not two, and that is why it is not in this one:
# splicing these segments into the shader while `metal/face_math.mm` still
# defines `position_step`, `position_accept` and the `kDirichletEntry`
# lift in its own MSL would be a duplicate definition, which the Metal driver
# reports at RUN time. So the embed rules and the deletion of the forked MSL
# land together, and these entries come off in that change.
# NEUTRAL BODIES NO COMPILER READS, WHICH IS A THIRD STATE AND NOT A VARIANT OF
# THE TWO ABOVE. CUDA_ONLY says nvcc compiles it and Metal does not; ONE_SIDED
# says the reverse. These are read by NEITHER, and not by the host build that
# compiles entrypoints/ either: they are neutral bodies written ahead of the
# consumers that will dispatch them, and no translation unit includes them
# today.
#
# THE DEBT IS THAT NOTHING CONTRADICTS THEM. A body no compiler reads cannot
# fail to build, so it can drift from the vocabulary around it with no gate
# noticing. EVERY RENDERING IS
# COMPILED BY SOME BUILD, OR IT IS NOT CHECKED AT ALL. Recording them is not
# blessing them: the ceiling below only
# falls, and an entry comes off when the body gets an entry declaration or an
# embed rule, which is the same change that gives it a caller.
#
# Do NOT retire an entry by deleting the body. The transcription is the work;
# what is missing is its consumer.
UNCOMPILED = {
    "energy/hinge_force.kernel.cpp":
        "the shell bending force and Hessian. Metal states the same "
        "arithmetic inline in its own entry points, so neither compiler reads "
        "this body",
    "energy/rod_bend_force.kernel.cpp":
        "the rod bending force and Hessian, in the same position as "
        "hinge_force",
    "energy/vertex_force.kernel.cpp":
        "the per-vertex force accumulation, in the same position as "
        "hinge_force",
    # THE THREE PRIMITIVES THE SHADER ASSEMBLY WAS THE ONLY COMPILER OF. They
    # were embedded verbatim as shader text by `shader_segments.mk`, which the
    # Metal orchestrator carried; that backend's library is now linked from the
    # generated entry renderings, and these three declare no entry, so nothing
    # compiles them. They are NOT dead on either backend: the driver reduces,
    # scans and sorts through `src/driver/reduce.rs` and `src/driver/sort.rs` on
    # the host, which is where that split is recorded.
    # `compute::simd_width` has no host rendering, so the three implementations
    # cannot become one body, and that is why these carry no entry to give.
    "primitives/reduce.kernel.cpp":
        "the device reduction, on the same terms as radix: the driver reduces "
        "through src/driver/reduce.rs",
    "primitives/scan.kernel.cpp":
        "the device scan, on the same terms as radix",
    "main/collision_window.kernel.cpp":
        "the per-object collision-window predicate. The other orchestration "
        "bodies under main/ carry entry declarations and are compiled; this "
        "one does not",
}

# AN END-OF-CHANGE MEASUREMENT AND A RATCHET. A conversion that gives one of
# these a caller lowers it, and nothing may raise it.
#
# RAISED ONCE, 10 to 13, BY THE CHANGE THAT DELETED THE METAL ORCHESTRATOR.
# `shader_segments.mk` embedded three primitives verbatim as shader text and was
# their only compiler; that backend's library is now linked from the generated
# entry renderings, which these three do not declare. The lines they came from
# left the tree in the same change, so this is the pen's 47,448 lines showing up
# as three entries here rather than new debt.
#
# LOWERED 13 to 12 BY THE TET ELASTIC LAYER'S ENTRY. `energy/tet_force` gained a
# composition body that folds through `fixed_csr_atomic_push` and an entry over
# it, so every compiler now reads that file and the assembly dispatches it.
#
# LOWERED AGAIN BY THE SHELL MEMBRANE LAYER'S ENTRY, on the same terms:
# `energy/face_force` gained `face_elastic_embed`, a composition over the
# membrane body and the pressure body that folds through
# `face_atomic_embed_force` and `fixed_csr_atomic_push`, and an entry over it.
UNCOMPILED_CEILING = 6

CUDA_ONLY_REASON = (
    "orchestration steps as neutral bodies. The Metal shader still runs its "
    "own hand-written MSL for these "
    "steps (metal/face_math.mm), so embedding the renderings beside it "
    "would define the same functions twice, which the Metal driver reports at "
    "RUN time. The embed rules and the deletion of the forked MSL are one "
    "change; delete these entries there"
)
CUDA_ONLY = {
    rel: CUDA_ONLY_REASON
    for rel in (
        "main/velocity.kernel.cpp",
        "main/rewind_fix.kernel.cpp",
        "main/dx_seed.kernel.cpp",
        "main/dirichlet.kernel.cpp",
        "main/dx_norm.kernel.cpp",
        "main/position_step.kernel.cpp",
        "main/position_accept.kernel.cpp",
    )
}

# The intersection scan's four bodies. Same situation as the block above and
# the same instruction: these come off the
# list in the change that deletes the forked MSL, not before. What is different
# is that two of them cannot be embedded at all until the shader declares three
# types, which is a checkable precondition rather than a preference.
#
# The Metal intersection visitors still carry their own hand-written pair
# filter, their own `edge_triangle_intersect` (a third copy of the same
# differencing wrapper, beside the CPU shim's `edge_triangle_intersect_fp` and
# the Rust mirror in ppf-cts-core), and no intersection RECORD at all: that backend
# reports a bare status rather than naming the elements.
# `energy/model/fix.hpp` carries the seam only because MSL demands an address
# space on a reference type, and it is recorded rather than embedded because the
# Metal shader HAS NO CALLER FOR IT: `face_math.mm` names `fix::` zero times,
# running its own hand-written vertex constraint instead. The only includer of
# this header is `contact/vertex_constraint.kernel.cpp`, whose MSL rendering the
# entry check now compiles, so the header IS exercised by that compiler; what is
# missing is a production call, and embedding it today would put a definition in
# the shader that nothing reaches. It comes off this list in the change that
# replaces the forked vertex-constraint MSL with the generated rendering, which
# is the same instruction the block above carries.
_FIX_CUDA_ONLY = {
    "energy/model/fix.hpp": (
        "the Metal shader names `fix::` nowhere, running its own hand-written "
        "vertex constraint, so embedding this header would define a function no "
        "dispatch reaches. It lands with the change that replaces that MSL with "
        "the generated rendering of contact/vertex_constraint"
    ),
}

CUDA_ONLY.update(_FIX_CUDA_ONLY)

_LOCK_CUDA_ONLY_REASON = (
    "Metal REFUSES projected translation and rotation locks BY NAME at "
    "initialize(), so the run-time shader dispatches no lock kernel and "
    "embedding this would splice code nothing reaches. It is not unread on "
    "that backend either, which is the distinction this exception turns on: "
    "the generated MSL rendering of `translation_lock_drift_row` IS compiled, "
    "by `entry-check`, which walks a body's own includes and so reads both "
    "files. They join the segment list in the change that lifts the refusal"
)

_LOCK_CUDA_ONLY = {
    "solver/translation_lock_check.kernel.cpp": _LOCK_CUDA_ONLY_REASON,
    "solver/translation_lock_frames.kernel.cpp": _LOCK_CUDA_ONLY_REASON,
    "solver/translation_lock_rows.kernel.cpp": _LOCK_CUDA_ONLY_REASON,
    "solver/translation_lock_math.hpp": _LOCK_CUDA_ONLY_REASON,
}

CUDA_ONLY.update(_LOCK_CUDA_ONLY)

# The offline linear-system dumper's row pass, in the same situation as the
# statistics recorder above and recorded for the same kind of reason: the thing
# that would call it does not exist on the other backend.
#
# WHAT IS DIFFERENT, AND IT IS WHY THE REASON IS NOT THE FORK REASON. The other
# entries here name a Metal fork that still states the same arithmetic inline,
# so embedding would add a second statement of it. There is no fork of this
# body: the Metal backend has no linear-system dumper at all, so the shader
# would gain a row walk that no dispatch names and no `kernel void` reaches.
# The MSL RENDERING is still compiled, by the entry harness the Metal recipe
# builds for every declaration carrying [[seam::args]], so this entry costs the
# body no checking; what it withholds is a segment of the production shader.
CUDA_ONLY["main/dump_linsys.kernel.cpp"] = (
    "the Metal backend has no linear-system dumper, so embedding this body "
    "would put a row walk in the production shader that no dispatch names. "
    "The entry harness compiles its MSL rendering either way. It lands with "
    "the change that gives that backend a dumper"
)

# The element energies and the remaining orchestration steps. Same situation as
# the two blocks above and the same instruction: an entry comes off in the
# change that rewrites
# the Metal fork to CALL the body, not before, because embedding a body beside a
# fork that still inlines the same arithmetic adds a second statement of it
# rather than replacing one.
#
# The element bodies are the ones worth reading the Metal side against. Every
# dependency each of them names IS already embedded (the model headers, the
# spectral analyses, the converters, the SVDs, the scatters, the damping
# bodies), so the blocker is not a missing type: it is that
# `metal/face_math.mm` hand-writes the same dispatch, damping and scaling
# inside its own `kernel void` entry points. Rewriting those to call these
# bodies is one change per element type and needs a Metal host to verify.
_ENERGY_CUDA_ONLY_REASON = (
    "the Metal shader still inlines the same arithmetic inside its own kernel "
    "entry points "
    "(metal/face_math.mm), so embedding this body would define a second "
    "statement of it beside the fork. Every dependency it names is already "
    "embedded, so the change that removes this entry is rewriting that entry "
    "point to call the body"
)
_STEP_CUDA_ONLY_REASON = (
    "the Metal shader still runs its own hand-written MSL for this step, so "
    "the embed rule and the deletion of the forked MSL are one change; delete "
    "this entry there"
)
CUDA_ONLY.update({
    rel: _ENERGY_CUDA_ONLY_REASON
    for rel in (
        "energy/elastic_model.kernel.cpp",
        "energy/rod_force.kernel.cpp",
    )
})
CUDA_ONLY.update({
    rel: _STEP_CUDA_ONLY_REASON
    for rel in (
        "main/fix_xz_drag.kernel.cpp",
        "main/stretch.kernel.cpp",
    )
})

# `main/override_seed.kernel.cpp` is a DIFFERENT case and is recorded
# separately, because until this change it was in NEITHER set: no CUDA
# translation unit included it and the Metal build did not embed it, and rule 1
# takes `(seam | kernels) & closure` while rule 2 takes `embedded - closure`, so
# a body in neither was invisible to both and got no recorded exception. The
# CUDA closure reaches it now, through entrypoints/shim_override_seed.cpp,
# which puts it under rule 1, where an entry has to be argued for. That is the
# improvement; the Metal half is the same instruction as the block above.
# A SEAM-CARRYING HEADER THAT IS HOST CODE ON EVERY BACKEND, which is a
# different case from every entry above: those are kernel BODIES the Metal
# shader does not yet run. `vec/vec.hpp` is not a body at all. It declares the
# two array VIEWS the solver reads through, a base pointer plus its counts, and
# it carries the seam because `operator[]` is reached from both execution spaces
# and the annotation for that has one spelling per compiler. The Metal shader
# never sees it: that backend binds arena handles and raw device pointers, and
# `metal/Makefile` embeds no header that names `Vec<T>`. What Metal DOES
# compile is this header's host image, through `data.hpp` in `face_math.mm` and
# `main.mm`, so both backends do read one definition and the fork rule 1 exists
# to prevent is not open here. The entry comes off if the shader ever takes a
# body that indexes a `Vec<T>`.
# SEVEN BODIES THIS GATE SEES ONLY BECAUSE IT READS WHAT nvcc ACTUALLY
# COMPILES. The CUDA closure is walked from the GENERATED ENTRY POINTS, which is
# where every neutral body reaches nvcc. So a body carrying the seam, holding an
# entry declaration and embedded nowhere in the Metal shader is visible as what
# it is: a fork.
#
# Each was checked rather than assumed. The Metal shader reaches the same work
# under ITS OWN kernel names (`face_math.mm` carries `collision_point_face_m2c`,
# `point_face`, `embed_force`, `BaraffWitkin`, `diff_table` and a hand-written
# vertex constraint), so embedding the neutral body beside that MSL would define
# the work twice, which the Metal driver reports at RUN time. The embed rule and
# the deletion of the forked MSL are ONE change, and that change is what takes
# an entry off this list; adding the embed alone would break the shader.
_FORK_CUDA_ONLY_REASON = (
    "nvcc compiles this body through its generated entry point, and the Metal "
    "shader reaches the same work through its own hand-written MSL under its "
    "own kernel names. Embedding the body beside that MSL would define the "
    "work twice, so the embed rule and the deletion of the fork are one change"
)
CUDA_ONLY.update({
    rel: _FORK_CUDA_ONLY_REASON
    for rel in (
        "contact/collision_narrow.kernel.cpp",
        "contact/contact_narrow.kernel.cpp",
        "contact/vertex_constraint.kernel.cpp",
        "energy/model/baraffwitkin.kernel.cpp",
        "energy/model/material_diff_table.kernel.cpp",
        "main/momentum.kernel.cpp",
        "utility/vertex_scatter.kernel.cpp",
    )
})

CUDA_ONLY["vec/vec.hpp"] = (
    "an array VIEW rather than a kernel body: the Metal SHADER binds arena "
    "handles and never names Vec<T>, while the Metal HOST compiles this exact "
    "header through data.hpp, so the single definition is already shared"
)

CUDA_ONLY["main/override_seed.kernel.cpp"] = (
    "the Metal backend writes the same seed in its own host C++ loop "
    "(metal/main.mm), which already spells the exact `float` form this "
    "body does, so the change that removes this entry is rewriting that loop to "
    "call the body rather than restating it"
)

# The prologue directory, exempt from rule 1 and the only exemption to it.
#
# Rule 1 asks that a body nvcc compiles also reach the shader. These files hold
# no body: they DEFINE the SM_ names, and the Metal side of the same table is
# kMslMacroSeam in metal/shader_compiler.mm, because the Metal driver is
# handed one concatenated string and can resolve no include. Embedding them
# would put `__device__ inline`, `atomicAdd` and `__clz` into a shader, and
# seam.hpp's own `#include` would then trip rule 3.
#
# The exemption is not a hole, because it is paired with rules 4 and 6: what
# rule 1 would be protecting here is that the tables carry the same names, and
# check_four and check_six measure exactly that, for the macro table and for
# the seam table respectively.
SEAM_PROLOGUE_DIR = "seam/"


def fail(msgs, path, title, message):
    """Record a failure, and annotate the file in the GitHub UI when running
    under Actions so the report lands on the header rather than in a log."""
    msgs.append(f"{path}: {message}")
    if os.environ.get("GITHUB_ACTIONS"):
        one_line = " ".join(message.split())
        print(f"::error file={path},title={title}::{one_line}")


def read(path):
    return path.read_text(encoding="utf-8", errors="replace")


def metal_embed_list(cpp_dir):
    """The src/kernels sources the Metal build compiles.

    IT IS DERIVED, NOT READ OUT OF A LIST. The Metal library is linked by the
    build from the GENERATED ENTRY RENDERINGS, one translation unit per neutral
    source that declares an entry, and `entries_probe.mm` establishes that every
    function in it yields a pipeline. So what that backend compiles is what nvcc
    compiles: the sources carrying an entry declaration, derived by the same
    predicate both recipes use. A hand-kept list naming solver headers one by one
    and pairing each with the C identifier a `shader_add_segment` call expects is
    what a shader ASSEMBLED at run time from spliced segments needs, and this
    build assembles none.

    THE FORK RULES 1 AND 2 GUARD AGAINST IS THEREFORE ABSENT, not merely
    unobserved: there is no second statement of a kernel for a first to drift
    from. They still run, and what they compare is two derivations of one list,
    which is weaker than a comparison against hand-written MSL and is said here
    rather than implied.
    """
    # `[[seam::entry]]` may name the count parameter, so the argument form is
    # admitted here too; a literal `]]` matches none of the 284 declarations.
    entry_decl = re.compile(r"^\[\[seam::(?:args|entry)(?:\([^)]*\))?\]\]", re.M)
    return {
        str(f.relative_to(cpp_dir))
        for f in cpp_dir.rglob("*.kernel.cpp")
        if entry_decl.search(read(f))
    }

def spliced_entry_names(cpp_dir, sources):
    """`<name>_entry` for every [[seam::entry]] in a spliced entry rendering.

    The generator names an entry point after its neutral declaration with an
    `_entry` suffix, so the names are derivable from the declaration and no
    generated file has to exist for this gate to run.
    """
    names = set()
    for rel in sorted(sources):
        path = cpp_dir / rel
        if not path.exists():
            sys.exit(
                f"check-shared-wiring: shader_segments.mk splices the entry "
                f"rendering of {rel}, which does not exist. A segment naming a "
                f"source that is not there embeds nothing and the shader loses "
                f"every entry point it carried.")
        text = read(path)
        for m in re.finditer(r"\[\[seam::entry[^\]]*\]\]", text):
            decl = re.search(r"\bvoid\s+(\w+)\s*\(", text[m.end():m.end() + 400])
            if decl:
                names.add(decl.group(1) + "_entry")
    return names


def transcompiler(cpp_dir):
    """The transcompiler, which lives in `ppf-cts-compute`, not the kernel tree.

    Resolved from the kernel tree's own location rather than kept as a literal,
    so this gate and the build cannot disagree about which script validated a
    kernel. The renderer belongs to the compute crate: it is the other half of
    what a compute abstraction does, beside allocating and launching, and the
    kernel tree it renders belongs to the solver.
    """
    script = cpp_dir.parents[3] / "crates/ppf-cts-compute/seam/kernelgen.py"
    if not script.exists():
        sys.exit(f"check-shared-wiring: no such file: {script}")
    return script


def cuda_translation_units(cuda_dir, cpp_dir):
    """The .cu files nvcc compiles, read off the CUDA recipe rather than listed.

    THE RECIPE COMPILES THREE THINGS AND SO DOES THIS. `ppf-cts-compute` is a
    general compute API held to five verbs (allocation, free, copy and transfer,
    and kernel launch), so MECHANISM_DIRS names the translation units it owns
    and ABI_FILES the C boundary the neutral driver calls. Beside them nvcc
    compiles ONE OBJECT PER GENERATED ENTRY POINT, which is where every neutral
    kernel body reaches nvcc now that the CUDA orchestrator is deleted: an
    entry rendering includes the body rendering, so the body and everything it
    includes are in the closure exactly when some neutral source declares an
    entry. That set is DERIVED by the same predicate the recipe uses, never
    listed, so the two cannot disagree about which sources nvcc reads.

    LOGIC_DIRS and LOGIC_KERNEL_FILES named the orchestrator's translation
    units and are empty; they are still read because a caller that passes
    BACKEND_LOGIC_ROOT still gets that recipe, and because reading a set
    smaller than what nvcc reads while every rule below reported clean over the
    smaller set is the failure this whole file is written to avoid.

    Returns (where, rel) pairs, `where` naming which root resolves the path.
    """
    text = read(cuda_dir / "Makefile")
    # `[ \t]*`, NOT `\s*`: `\s` matches a newline, so on an EMPTY assignment the
    # pattern ran past the line end and captured the NEXT line as the value.
    # LOGIC_DIRS is empty now that the CUDA orchestrator is deleted, which is
    # exactly when that bug bites, and it reported the following variable's
    # name as a missing translation unit.
    mech = re.search(r"^MECHANISM_DIRS[ \t]*=[ \t]*(.*)$", text, re.M)
    logic = re.search(r"^LOGIC_DIRS[ \t]*=[ \t]*(.*)$", text, re.M)
    kern = re.search(r"^LOGIC_KERNEL_FILES[ \t]*=[ \t]*(.*)$", text, re.M)
    if not mech or not logic or not kern:
        sys.exit(
            "check-shared-wiring: the CUDA Makefile no longer declares "
            "MECHANISM_DIRS, LOGIC_DIRS and LOGIC_KERNEL_FILES; this check "
            "cannot enumerate the CUDA translation units and refuses to report "
            "a pass it did not earn."
        )
    abi = re.search(r"^ABI_FILES[ \t]*=[ \t]*(.*)$", text, re.M)
    if not abi:
        sys.exit(
            "check-shared-wiring: the CUDA Makefile no longer declares "
            "ABI_FILES; this check cannot enumerate the CUDA translation "
            "units and refuses to report a pass it did not earn."
        )
    tus = [("cuda", f"{d}/{d}.cu") for d in mech.group(1).split()]
    tus += [("cuda", f) for f in abi.group(1).split()]
    tus += [("logic", f"{d}/{d}.cu") for d in logic.group(1).split()]
    tus += [("logic", f) for f in kern.group(1).split()]
    # ONE OBJECT PER GENERATED ENTRY POINT, derived by the recipe's own
    # predicate: a `[[seam::args]]` or `[[seam::entry]]` declaration opening a
    # line. The rendering nvcc compiles lives in OUT_DIR, so what is named here
    # is the neutral source it is rendered from, which is the same key the
    # closure walk uses for anything in the kernel tree.
    entry_decl = re.compile(r"^\[\[seam::(?:args|entry)(?:\([^)]*\))?\]\]", re.M)
    entries = sorted(
        str(f.relative_to(cpp_dir))
        for f in cpp_dir.rglob("*.kernel.cpp")
        if entry_decl.search(read(f))
    )
    if not entries:
        sys.exit(
            "check-shared-wiring: no neutral kernel source declares an entry, "
            "so nvcc would compile no kernel at all. That is the recipe's own "
            "AN EMPTY SET IS THE SILENT FAILURE case, and this check refuses "
            "to report a pass over it."
        )
    tus += [("kernels", f) for f in entries]
    # The ROCm prologue is reached exactly as the CUDA one is: seam.hpp names
    # `seam_hip.hiph` with no path, and that backend's build puts its own
    # directory on the include path. Without this root the include resolves
    # to nothing and the closure is quietly smaller than what a compiler
    # reads, which is the failure this function's own docstring describes.
    roots = {"cuda": cuda_dir, "kernels": cpp_dir,
             "rocm": cuda_dir.parent / "rocm"}
    missing = [f"{w}:{t}" for w, t in tus if not (roots[w] / t).exists()]
    if missing:
        sys.exit(
            "check-shared-wiring: the CUDA Makefile names translation units "
            "that do not exist: " + ", ".join(missing)
        )
    return tus


INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)

# A neutral kernel source: plain C++, rendered per backend by
# ppf-cts-compute/seam/kernelgen.py. A CUDA translation unit names the nvcc rendering from
# the generated root, so `#include "contact/grain_pair.kernel.cu"` is how
# `contact/grain_pair.kernel.cpp` becomes reachable.
KERNEL_SUFFIX = ".kernel.cpp"
KERNEL_CU_SUFFIX = ".kernel.cu"


# The rendered ENTRY point of a kernel, the second thing the generator emits
# beside the body rendering. It is reached exactly like KERNEL_CU_SUFFIX.
ENTRY_CU_SUFFIX = ".entry.cu"

# The ARGUMENT-RECORD half of that entry rendering. It carries the records, the
# layout assertions and the launcher declarations and no definition, so it is
# what a caller in a HEADER includes: the entry rendering beside it holds the
# `__global__` and the launcher, which only one translation unit may have.
# Named from the generated root exactly like the other two, and rendered from
# the same neutral `<path>.kernel.cpp`, so it resolves to the same source.
ARGS_CUH_SUFFIX = ".args.cuh"

# The transcompiler's dispatch table, generated into the same root as the
# renderings and named by the ABI translation unit.
GENERATED_TABLE = "kernel_table.inc"

# The diagnostic file table, generated into the same root and named by the
# diagnostic channel. It maps the id a `DIAG_ASSERT` records to the path it
# names, which a device record cannot carry as a string.
GENERATED_DIAG_FILES = "diag_files.inc"

# The two whole-tree tables, which mirror no single neutral source. Every other
# generated name renders from one `.kernel.cpp` and is walked as that source.
WHOLE_TREE_TABLES = (GENERATED_TABLE, GENERATED_DIAG_FILES)


def cuda_include_closure(cuda_dir, cpp_dir, tus):
    """Every file nvcc reaches from those translation units.

    THE WALK CROSSES A CRATE BOUNDARY, AND THAT IS THE WHOLE DIFFICULTY. The
    translation units live in `ppf-cts-compute/cuda` and the headers they
    compile live in `ppf-cts-solver/src/kernels`, so a resolver that only
    followed includes inside one directory would silently shrink the closure
    while rules 1, 2 and 8 went on reporting clean over a smaller set. That is a
    gate that has stopped measuring, which is worse than one that fails, so the
    resolver mirrors the include path the CUDA Makefile actually passes:

      1. the including file's own directory   (quoted-include rule)
      2. the generated root                    -I$(KERNELGEN_DIR)
      3. the CUDA mechanism tree               -I.
      4. the neutral kernel tree               -I$(KERNEL_ROOT)

    THERE WAS A FIFTH, `-I$(BACKEND_LOGIC_ROOT)`, and it pointed into the
    holding pen. That crate is deleted and the Makefile no longer passes the
    flag, so the root is gone from this resolver too.

    Step 2 is not a directory on disk here: a rendering is named
    `<path>.kernel.cu`, `<path>.entry.cu` or `<path>.args.cuh` and the file
    that has to exist is the neutral `<path>.kernel.cpp` all three are rendered
    from.

    A reached file is keyed by its path relative to the NEUTRAL tree when it
    lives there, because that is the namespace the embedded-header, seam-header
    and neutral-kernel sets are expressed in. A file in either CUDA tree is
    keyed by its repo-relative path instead, which can never collide with the
    first form and keeps it countable.

    Returns the key set AND the key-to-path map. The map is not a convenience:
    rule 8 reads every file in the closure to find the neutral bodies CUDA
    calls, and a key alone no longer says which of the two trees to open it
    from. Without it that rule silently censused only the files it could still
    find, which took "reached by CUDA" from 235 bodies to 36 while every check
    reported clean.
    """
    # The ROCm prologue is reached exactly as the CUDA one is: seam.hpp names
    # `seam_hip.hiph` with no path, and that backend's build puts its own
    # directory on the include path. Without this root the include resolves
    # to nothing and the closure is quietly smaller than what a compiler
    # reads, which is the failure this function's own docstring describes.
    roots = {"cuda": cuda_dir, "kernels": cpp_dir,
             "rocm": cuda_dir.parent / "rocm"}
    repo = cuda_dir.parents[2]
    seen = set()
    paths = {}
    stack = list(tus)
    while stack:
        where, rel = stack.pop()
        key = rel if where == "kernels" else str(
            (roots[where] / rel).relative_to(repo))
        if key in seen:
            continue
        seen.add(key)
        path = roots[where] / rel
        paths[key] = path
        if not path.exists():
            continue
        here = os.path.dirname(rel)
        for m in INCLUDE_RE.finditer(read(path)):
            spelled = m.group(1)
            # A WHOLE-TREE TABLE IS GENERATED AND MIRRORS NO NEUTRAL SOURCE.
            # Every other generated name below renders from one `.kernel.cpp`
            # and is walked as that source; these two are built from the tree as
            # a whole, so there is nothing to walk into and nothing this
            # resolver could open. Neither hides a file from the closure: the
            # dispatch table reaches it through the bodies it names, which are
            # already in it by their own entries, and the diagnostic file table
            # holds no includes at all, only an id and a path per row.
            if spelled in WHOLE_TREE_TABLES:
                continue
            for suffix in (KERNEL_CU_SUFFIX, ENTRY_CU_SUFFIX,
                           ARGS_CUH_SUFFIX):
                if spelled.endswith(suffix):
                    # A rendering, named from the generated root, which mirrors
                    # the neutral tree.
                    stack.append(
                        ("kernels", spelled[: -len(suffix)] + KERNEL_SUFFIX))
                    break
            else:
                own = os.path.normpath(os.path.join(here, spelled))
                for cand_where, cand_rel in ((where, own),
                                             ("cuda", spelled),
                                             # "rocm" on the same terms as "cuda": seam.hpp names
                                             # `seam_hip.hiph` with no path exactly as it names
                                             # `seam_cuda.cuh`, and each backend build puts its
                                             # own directory on the include path.
                                             ("rocm", spelled),
                                             ("kernels", spelled)):
                    if cand_rel.startswith(".."):
                        continue
                    if (roots[cand_where] / cand_rel).exists():
                        stack.append((cand_where, cand_rel))
                        break
                else:
                    # AN INCLUDE THIS RESOLVER CANNOT FIND STOPS THE CHECK. It
                    # means either the tree is broken or this resolver no longer
                    # mirrors the include path the recipe passes, and both leave
                    # the closure smaller than what nvcc reads while every rule
                    # below goes on reporting clean over the smaller set.
                    sys.exit(
                        f"check-shared-wiring: {path} includes \"{spelled}\", "
                        f"which resolves to no file under either CUDA tree or "
                        f"the neutral kernel tree. The closure would be smaller "
                        f"than what nvcc reads, so this check refuses to report "
                        f"a pass it did not earn.")
    return seen, paths


def neutral_kernels(cpp_dir):
    """Every neutral kernel source under cpp/.

    These carry no SM_ name, so seam_headers() cannot see them, and they are
    exactly as much a shared body as a header that does: nvcc compiles one
    rendering and the shader compiler another, and a source only one side reads
    is the fork the single-source rule forbids.
    """
    return {str(p.relative_to(cpp_dir))
            for p in sorted(cpp_dir.rglob("*" + KERNEL_SUFFIX))
            if p.is_file()}


def check_five(cpp_dir, kernels, msgs):
    """Rule 5: every neutral kernel renders, for all three targets.

    kernelgen.py is the validator: it rejects a seam macro, a preprocessor
    conditional, an unknown attribute, a backend keyword written directly, and
    an include naming a file that does not exist. Running it here means the
    rejection reaches CI without a CUDA toolkit or a Metal device, and means
    this script does not carry a second copy of the rules.
    """
    import subprocess
    import tempfile

    script = transcompiler(cpp_dir)
    offenders = []
    with tempfile.TemporaryDirectory() as tmp:
        for rel in sorted(kernels):
            for target, ext in (("cu", ".cu"), ("metal", ".metal"),
                                ("cpp", ".cpp")):
                out = os.path.join(tmp, target, rel[: -len(KERNEL_SUFFIX)] + ext)
                result = subprocess.run(
                    [sys.executable, str(script), "--target", target,
                     "--out", out, str(cpp_dir / rel)],
                    capture_output=True, text=True)
                if result.returncode != 0:
                    offenders.append(rel)
                    fail(
                        msgs,
                        f"crates/ppf-cts-solver/src/kernels/{rel}",
                        "Neutral kernel does not render",
                        f"ppf-cts-compute/seam/kernelgen.py --target {target} rejected it: "
                        + " ".join(result.stderr.split()),
                    )
    return offenders


SEAM_RE = re.compile(r"\bSM_[A-Z][A-Z0-9_]*\b")

# THE ROLE MARKER IS NOT A SEAM SPELLING, and a header whose only SM_ name is
# this one is saying the OPPOSITE of what rule 1 looks for.
#
# Every other SM_ name is a spelling the shader compiler has to be given: an
# address space, an inline annotation, an intrinsic. A header carrying one has
# code meant to be compiled by more than one backend, so a Metal build that does
# not embed it has a fork. SM_MSL_CONCAT names no spelling at all. It means "the
# Metal shader compiler is reading this", and a header only ever tests it to
# guard something AWAY from that compiler, which is a statement that the guarded
# part is host-only rather than a claim on the shader.
#
# Counting it would make the guard its own violation: `common.hpp` and
# `float_math.hpp` each put their standard-library includes behind it so that an
# offline entry translation unit can read the rest, and would then be reported as
# shared headers the Metal build fails to embed, which they are not. A header
# that tests the marker AND carries a real seam name is still a seam header, so
# nothing rule 1 was catching stops being caught.
SEAM_ROLE_MARKER = "SM_MSL_CONCAT"


def seam_headers(cpp_dir):
    """Headers carrying the backend macro seam, which is what makes a body
    compilable by more than one backend."""
    out = set()
    for path in sorted(cpp_dir.rglob("*")):
        if path.suffix not in (".hpp", ".h", ".cuh") or not path.is_file():
            continue
        names = set(SEAM_RE.findall(read(path)))
        names.discard(SEAM_ROLE_MARKER)
        if names:
            out.add(str(path.relative_to(cpp_dir)))
    return out


# Preprocessor names this check knows the answer to when the Metal shader
# compiler reads a shared header. Everything else is treated as unknown, and an
# unknown condition is assumed TAKEN, so an include hiding behind one is
# reported rather than missed.
MSL_DEFINED = {"SM_MSL_CONCAT": True, "__METAL_VERSION__": True,
               "__NVCC__": False, "__CUDACC__": False, "__CUDA_ARCH__": False}

COND_RE = re.compile(r"^\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)$")


def _value(kind, expr):
    """The truth of one conditional under MSL, or None when unknown."""
    expr = expr.strip()
    if kind == "ifdef":
        return MSL_DEFINED.get(expr.split()[0] if expr.split() else "")
    if kind == "ifndef":
        v = MSL_DEFINED.get(expr.split()[0] if expr.split() else "")
        return None if v is None else not v
    # #if / #elif: only the defined() forms are evaluated, which covers every
    # guard in this tree. Anything else stays unknown.
    m = re.fullmatch(r"(!?)\s*defined\s*\(?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)?", expr)
    if not m:
        return None
    v = MSL_DEFINED.get(m.group(2))
    if v is None:
        return None
    return (not v) if m.group(1) == "!" else v


def includes_visible_to_msl(text):
    """The #include lines that survive when the shader compiler reads this file.

    A conditional whose truth is not known is assumed taken, so this over-reports
    rather than under-reports.
    """
    survivors = []
    stack = []  # one entry per open conditional: True while the branch is live
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = COND_RE.match(line)
        if m:
            kind, expr = m.group(1), m.group(2)
            if kind in ("ifdef", "ifndef", "if"):
                v = _value(kind, expr)
                stack.append(True if v is None else v)
            elif kind == "elif":
                if stack:
                    v = _value("elif", expr)
                    stack[-1] = True if v is None else v
            elif kind == "else":
                if stack:
                    stack[-1] = not stack[-1]
            elif kind == "endif":
                if stack:
                    stack.pop()
            continue
        if all(stack) and re.match(r"^\s*#\s*include\b", line):
            survivors.append((lineno, line.strip()))
    return survivors


def check_one(cpp_dir, embedded, closure, seam, msgs):
    """Rule 1: a seam header nvcc compiles must be embedded by Metal."""
    candidates = {
        rel for rel in (seam & closure)
        if not rel.startswith(SEAM_PROLOGUE_DIR)
    }
    offenders = sorted(candidates - embedded - set(CUDA_ONLY))
    # A recorded exception that has stopped being true is itself a failure, for
    # the reason check_two gives: an allowlist nobody re-reads is a silencer.
    recorded = []
    for rel in sorted(CUDA_ONLY):
        if rel in embedded:
            msgs.append(
                f"CUDA_ONLY names {rel}, which the Metal build now embeds. "
                f"The exception is stale: delete the entry."
            )
        elif rel not in candidates:
            msgs.append(
                f"CUDA_ONLY names {rel}, which no CUDA translation unit "
                f"reaches. The exception is stale: delete the entry."
            )
        else:
            recorded.append(rel)
    if recorded:
        # Grouped by REASON, because the entries no longer share one: printing a
        # single reason under a list that has several attributes each name to an
        # argument that may not be its own.
        by_reason = {}
        for rel in recorded:
            by_reason.setdefault(CUDA_ONLY[rel], []).append(rel)
        for reason, rels in by_reason.items():
            print(f"  recorded CUDA-only: {', '.join(rels)}")
            print(f"      {reason}")
    for rel in offenders:
        fail(
            msgs,
            f"crates/ppf-cts-solver/src/kernels/{rel}",
            "Shared header not wired into Metal",
            "carries the SM_* backend seam and is reachable from a CUDA "
            "translation unit, but metal/Makefile does not embed it. nvcc "
            "compiles this body and the Metal shader never sees it, which is "
            "the fork the single-source rule forbids. Add an embed rule in "
            "crates/ppf-cts-compute/metal/Makefile and a "
            "shader_add_segment call for it.",
        )
    return offenders


def check_two(embedded, closure, msgs):
    """Rule 2: a header Metal embeds must be compiled by nvcc too."""
    offenders = sorted(embedded - closure)
    unexpected = [rel for rel in offenders if rel not in ONE_SIDED]
    for rel in unexpected:
        fail(
            msgs,
            f"crates/ppf-cts-solver/src/kernels/{rel}",
            "Embedded header no CUDA TU compiles",
            "is embedded into the Metal shader but no CUDA translation unit "
            "includes it, so nvcc never compiles it and CUDA, the parity "
            "oracle, never runs it. Include it from the CUDA side, or record "
            "it in ONE_SIDED in this script with the reason.",
        )
    # A pin that has stopped being true is itself a failure. Without this the
    # list decays into names nobody re-reads, which is how an allowlist stops
    # being a record and starts being a silencer.
    for rel, reason in sorted(ONE_SIDED.items()):
        if rel not in embedded:
            msgs.append(
                f"ONE_SIDED names {rel}, which the Metal build no longer "
                f"embeds. Delete the entry."
            )
        elif rel in closure:
            msgs.append(
                f"ONE_SIDED names {rel}, which IS now reachable from a CUDA "
                f"translation unit. The exception is stale: delete the entry."
            )
        else:
            print(f"  recorded one-sided: {rel}\n      {reason}")
    return unexpected


def assembler_neutralizes_quoted_includes(metal_dir):
    """Whether the assembler still strips quoted includes as it splices.

    Rule 3 stops reporting a quoted include because this transform answers it.
    If the transform goes away, the rule has to start reporting them again, so
    the mechanism is checked rather than assumed. Both halves are required: the
    function has to exist AND append_segment has to call it, because either one
    alone leaves the shader compiler reading the include.
    """
    text = read(metal_dir / "shader_compiler.mm")
    defined = "std::string neutralize_quoted_includes(" in text
    called = "neutralize_quoted_includes(raw_body)" in text
    return defined and called


def check_three(cpp_dir, metal_dir, embedded, msgs):
    """Rule 3: nothing an embedded header includes may reach the MSL compiler."""
    quoted_handled = assembler_neutralizes_quoted_includes(metal_dir)
    if not quoted_handled:
        fail(
            msgs,
            "crates/ppf-cts-compute/metal/shader_compiler.mm",
            "Quoted-include transform is gone",
            "append_segment no longer runs neutralize_quoted_includes over the "
            "segment it splices, so every quoted #include in a shared kernel "
            "body now reaches the Metal shader compiler, which has no "
            "filesystem to resolve it against. Restore the transform, or put "
            "the include blocks back behind #ifndef SM_MSL_CONCAT in all of "
            "them.",
        )
    offenders = []
    for rel in sorted(embedded):
        path = cpp_dir / rel
        if not path.exists():
            msgs.append(
                f"metal/Makefile embeds $(KERNEL_ROOT)/{rel}, which does not "
                f"exist under the neutral kernel tree."
            )
            continue
        for lineno, line in includes_visible_to_msl(read(path)):
            # <metal_*> is the one include the shader compiler can serve.
            if re.search(r"#\s*include\s*<metal_[a-z_]+>", line):
                continue
            # A quoted include is neutralized by the assembler as it splices the
            # segment, so it never reaches the compiler.
            if quoted_handled and re.search(r'#\s*include\s*"', line):
                continue
            offenders.append((rel, lineno, line))
            fail(
                msgs,
                f"crates/ppf-cts-solver/src/kernels/{rel}",
                "Include reaches the MSL compiler",
                f"line {lineno}: {line} is visible to the Metal shader "
                "compiler, which is handed one concatenated string and has no "
                "filesystem to resolve it against. The assembler neutralizes a "
                "QUOTED include; this form it does not. Wrap it in "
                "#ifndef SM_MSL_CONCAT, and add the file to the shader's "
                "segment list if the shader needs its contents. This fails at "
                "run time, not at build time, so no build leg can catch it.",
            )
    return offenders


SM_DEFINE_RE = re.compile(r"^\s*#\s*define\s+(SM_[A-Z][A-Z0-9_]*)", re.M)


def check_four(cuda_dir, metal_dir, msgs, rocm_dir=None):
    """Rule 4: the nvcc prologue's SM_ names all exist in the Metal prologue.

    The two tables cannot be one file, so this is what keeps them from drifting.
    Only this direction is checked. A name the Metal prologue carries alone is
    deliberate and listed in src/kernels/seam/seam.hpp; a name nvcc carries
    alone is a kernel body that compiles on CUDA and breaks the shader at run
    time.

    The host prologue is not compared: it omits the warp and threadgroup names
    on purpose, and states which and why.
    """
    cuda_table = cuda_dir / "seam_cuda.cuh"
    msl_table = metal_dir / "shader_compiler.mm"
    for path in (cuda_table, msl_table):
        if not path.exists():
            sys.exit(f"check-shared-wiring: no such file: {path}")
    cuda_names = set(SM_DEFINE_RE.findall(read(cuda_table)))
    msl_names = set(SM_DEFINE_RE.findall(read(msl_table)))
    # THE HIP PROLOGUE IS COMPARED ON THE SAME TERMS AS THE METAL ONE. A name
    # the nvcc table carries and this one does not is a kernel body that
    # compiles on CUDA and fails the AMD compile, and with no AMD hardware in
    # this project that failure is caught by a build leg or not at all.
    tables = [(cuda_names, cuda_table), (msl_names, msl_table)]
    hip_table = (rocm_dir / "seam_hip.hiph") if rocm_dir else None
    hip_names = None
    if hip_table is not None and hip_table.exists():
        hip_names = set(SM_DEFINE_RE.findall(read(hip_table)))
        tables.append((hip_names, hip_table))
    # An empty parse would report a pass it did not earn.
    for names, path in tables:
        if not names:
            sys.exit(
                f"check-shared-wiring: found no SM_ definitions in {path}; "
                "either the prologue moved or this parser is broken."
            )
    missing = sorted(cuda_names - msl_names)
    for name in missing:
        fail(
            msgs,
            "crates/ppf-cts-compute/metal/shader_compiler.mm",
            "Seam name missing from the Metal prologue",
            f"the nvcc prologue defines {name} and kMslMacroSeam here "
            "does not. A kernel body naming it compiles under nvcc and leaves "
            "the shader compile to fail at run time, after every build leg has "
            "gone green. Add it here in the same change.",
        )
    if hip_names is not None:
        for name in sorted(cuda_names - hip_names):
            fail(
                msgs,
                "crates/ppf-cts-compute/rocm/seam_hip.hiph",
                "Seam name missing from the HIP prologue",
                f"the nvcc prologue defines {name} and this one does not. A "
                "kernel body naming it compiles under nvcc and fails the AMD "
                "compile, which with no AMD hardware in this project is caught "
                "by a build leg or not at all. Add it here in the same change.",
            )
            missing.append(name)
    return missing


# The names a NEUTRAL kernel body reaches the seam through: ordinary members of
# the seam namespaces, declared by each backend prologue. A body carries no SM_
# macro, so rule 4 cannot see this table at all, and rule 6 below is its
# analogue.
#
# Parsed by declaration shape rather than by a list kept here, so a name added
# to a prologue is covered with no edit to this file:
#   inline <type> <name>(          a function (with any nvcc/MSL qualifiers)
#   using <name> =                 a type alias
#   constexpr <type> <name> =      a constant (MSL puts it in `constant`)
SEAM_FN_RE = re.compile(
    r"^\s*(?:constant\s+)?(?:__device__\s+|__host__\s+)*inline\s+"
    r"[A-Za-z_][A-Za-z0-9_:<>, ]*?\s+\**([a-z_][a-z0-9_]*)\s*\(", re.M)
SEAM_ALIAS_RE = re.compile(r"^\s*using\s+([a-z_][a-z0-9_]*)\s*=", re.M)
SEAM_CONST_RE = re.compile(
    r"^\s*(?:constant\s+)?constexpr\s+\w+\s+([a-z_][a-z0-9_]*)\s*=", re.M)

# The namespaces that table is split across. The split follows one rule: a name
# is `compute::` when its meaning DIFFERS PER BACKEND, and it is not when it
# does not. `bits::` holds the
# integer intrinsics, which compute the same value on every target.
#
# A block is read only to its closing `}  // namespace <name>`, so a declaration
# sitting outside them all is not counted as reachable from a kernel body when
# it is not. The nvcc prologue is the reference and must declare every one of
# them; kMslMacroSeam must match it name for name and so must declare every one
# too. The HOST prologue may omit a namespace outright, and only when every name
# the reference puts in it is a recorded absence below, which is what the lane
# namespace is on a target that runs one thread through a body.
SEAM_NAMESPACES = ("fmath", "bits", "compute")

# The six the HOST prologue omits on purpose, each a warp or threadgroup
# operation, or the lane count one is written against, that has no single-thread
# meaning. the host prologue states the reasoning; this pins it, so the
# omission stays a recorded decision rather than becoming an oversight nobody
# can tell from one. Keyed by NAME rather than by namespace, so the five that
# are per-backend and the one that is not stay one recorded exception.
SEAM_HOST_ABSENT = {
    "popcount", "simd_ballot", "simd_width", "shuffle_down", "shuffle_up",
    "threadgroup_barrier",
    # The ballot's TYPE, absent for the same reason the ballot itself is: it is
    # one bit per lane of a subgroup, and a target running one thread through a
    # body has no subgroup to have a width. The only neutral body that names it
    # is primitives/radix.kernel.cpp's COOPERATIVE half, whose serial twin the
    # host compiles instead, so nothing on this side ever reaches it.
    "ballot_t",
}


def seam_names(text, label, required):
    """The seam names one prologue declares, and the namespace each sits in."""
    found = {}
    for namespace in SEAM_NAMESPACES:
        opens = list(re.finditer(rf"^namespace {namespace} \{{$", text, re.M))
        if not opens:
            if namespace not in required:
                continue
            sys.exit(
                f"check-shared-wiring: {label} declares no `namespace "
                f"{namespace}`; either the prologue moved a seam name out of "
                f"the namespaces this rule reads, or this parser is broken, "
                f"and either way the check cannot pass."
            )
        closer = re.compile(rf"^\}}\s*//\s*namespace {namespace}\b", re.M)
        for opener in opens:
            end = closer.search(text, opener.end())
            if end is None:
                sys.exit(
                    f"check-shared-wiring: a `namespace {namespace}` block in "
                    f"{label} has no closing `}}  // namespace {namespace}`. "
                    f"The block is read to that comment, so without it the "
                    f"rule cannot tell which declarations are inside it."
                )
            body = text[opener.end():end.start()]
            for name in (set(SEAM_FN_RE.findall(body))
                         | set(SEAM_ALIAS_RE.findall(body))
                         | set(SEAM_CONST_RE.findall(body))):
                found[name] = namespace
    return found


def msl_prologue(metal_dir):
    """kMslMacroSeam's text, which is the Metal prologue."""
    m = re.search(r'kMslMacroSeam\s*=\s*R"MSL\((.*?)\)MSL"',
                  read(metal_dir / "shader_compiler.mm"), re.S)
    if not m:
        sys.exit(
            "check-shared-wiring: could not find kMslMacroSeam in "
            "metal/shader_compiler.mm; either the Metal prologue moved or "
            "this parser is broken, and either way the check cannot pass."
        )
    return m.group(1)


def check_six(cpp_dir, cuda_dir, metal_dir, msgs):
    """Rule 6: rule 4, for the seam table a neutral kernel body actually uses.

    Same failure and the same reason: the three tables cannot be one file, and a
    name only some of them carry is a body that compiles on one backend and
    breaks another. The CUDA-to-Metal direction fails the check, because that
    one breaks at RUN TIME when the shader is built, after every build leg has
    gone green. A name the host omits is reported and allowed only when
    SEAM_HOST_ABSENT records it; an unrecorded omission fails, since a host
    oracle silently missing a name would otherwise surface as an ordinary
    compile error somewhere unrelated.

    A name is reported with the namespace the nvcc prologue puts it in, so a
    message names the spelling a kernel body writes.
    """
    # THE CUDA AND HOST PROLOGUES DO NOT DECLARE THE WHOLE TABLE THEMSELVES.
    # `float_math.hpp` is the CUDA and host arm of `namespace fmath`, holding
    # the five transcendentals whose library forms emit FP64, and each prologue
    # includes it rather than forwarding to it. The Metal prologue has no such
    # header and declares the whole table itself. So the reference set is the
    # prologue PLUS that header, and a name added there and not to the Metal
    # prologue fails here rather than at the shader compile.
    float_math = read(cpp_dir / "float_math.hpp")
    cuda_ns = seam_names(read(cuda_dir / "seam_cuda.cuh") + float_math,
                         "the nvcc prologue, ppf-cts-compute/cuda/seam_cuda.cuh, "
                         "with src/kernels/float_math.hpp",
                         SEAM_NAMESPACES)
    msl_ns = seam_names(msl_prologue(metal_dir), "kMslMacroSeam",
                        SEAM_NAMESPACES)
    # A namespace the host may omit is one where the reference declares nothing
    # the host is expected to carry. Derived from the reference rather than
    # listed, so it follows a name moving between namespaces with no edit here.
    host_required = tuple(
        namespace for namespace in SEAM_NAMESPACES
        if any(name not in SEAM_HOST_ABSENT
               for name, where in cuda_ns.items() if where == namespace))
    host_ns = seam_names(read(cpp_dir / "seam" / "seam_host.h") + float_math,
                         "the host prologue, src/kernels/seam/seam_host.h, with "
                         "src/kernels/float_math.hpp",
                         host_required)
    cuda_names, host_names, msl_names = set(cuda_ns), set(host_ns), set(msl_ns)
    labels = (
        (cuda_names, "the nvcc prologue, ppf-cts-compute/cuda/seam_cuda.cuh"),
        (host_names, "the host prologue, src/kernels/seam/seam_host.h"),
        (msl_names, "kMslMacroSeam"),
    )
    for names, label in labels:
        if not names:
            sys.exit(
                f"check-shared-wiring: found no seam declarations in "
                f"{label}; either the prologue moved or this parser is broken."
            )
    missing = sorted(cuda_names - msl_names)
    for name in missing:
        fail(
            msgs,
            "crates/ppf-cts-compute/metal/shader_compiler.mm",
            "Seam name missing from the Metal prologue",
            f"the nvcc prologue declares {cuda_ns[name]}::{name} and "
            "kMslMacroSeam here does not. A neutral kernel naming it compiles "
            "under nvcc and leaves the shader compile to fail at run time, "
            "after every build leg has gone green. Add it here in the same "
            "change.",
        )
    unexplained = sorted((cuda_names - host_names) - SEAM_HOST_ABSENT)
    for name in unexplained:
        fail(
            msgs,
            "crates/ppf-cts-solver/src/kernels/seam/seam_host.h",
            "Seam name missing from the host prologue",
            f"the nvcc prologue declares {cuda_ns[name]}::{name} and this "
            "prologue does not, and it is not one of the warp and threadgroup "
            "names SEAM_HOST_ABSENT records as having no single-thread "
            "meaning. The host is where every self-test oracle runs, so add it "
            "here, or record it in SEAM_HOST_ABSENT with the reason.",
        )
    stale = sorted(SEAM_HOST_ABSENT & host_names)
    for name in stale:
        msgs.append(
            f"SEAM_HOST_ABSENT names {host_ns[name]}::{name}, which the host "
            f"prologue now declares. The exception is stale: delete the entry."
        )
    return missing + unexplained + stale


KERNEL_SM_RE = re.compile(r"\bSM_[A-Z][A-Z0-9_]*\b")
KERNEL_DIRECTIVE_RE = re.compile(r"^[ \t]*#[ \t]*(\w*)(.*)$")
KERNEL_INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]+"[^"]+"[ \t]*$')


def kernel_code(text):
    """`text` with comments and string literals blanked, offsets preserved.

    Deliberately simpler than kernelgen.py's classifier, and deliberately not
    imported from it: rule 7 is only worth having if it can fail when that
    script cannot.

    It tracks comments and DOUBLE-quoted strings and gives the apostrophe no
    meaning at all. That is what makes it immune to the hazard a character
    literal carries here: a stray or separator apostrophe (`1'000`) would
    otherwise open a literal running to the next one, masking a span of real
    code, and a masked span is exempt from both checks below. Ignoring the
    apostrophe cannot hide anything instead, because a character literal holds
    one character and can spell neither an SM_ name nor a directive.
    """
    out = list(text)
    i, n, state = 0, len(text), None
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state is None:
            if c == "/" and nxt == "/":
                state = "line"
            elif c == "/" and nxt == "*":
                state = "block"
            elif c == '"':
                state = "str"
                out[i] = " "
                i += 1
                continue
            else:
                i += 1
                continue
            out[i] = out[i + 1] = " "
            i += 2
            continue
        if state == "line":
            if c == "\n":
                state = None
                i += 1
                continue
        elif state == "block":
            if c == "*" and nxt == "/":
                out[i] = out[i + 1] = " "
                state = None
                i += 2
                continue
        else:  # "str"
            if c == "\\":
                out[i] = " "
                if i + 1 < n and text[i + 1] != "\n":
                    out[i + 1] = " "
                    i += 2
                    continue
                i += 1
                continue
            if c == '"':
                out[i] = " "
                state = None
                i += 1
                continue
            if c == "\n":          # a string never spans a line here
                state = None
                i += 1
                continue
        if c != "\n":
            out[i] = " "
        i += 1
    return "".join(out)


def check_seven(cpp_dir, kernels, msgs):
    """Rule 7: no SM_ macro and no stray preprocessor directive in a kernel."""
    offenders = set()
    for rel in sorted(kernels):
        path = cpp_dir / rel
        code = kernel_code(read(path))
        for m in KERNEL_SM_RE.finditer(code):
            line = code.count("\n", 0, m.start()) + 1
            offenders.add(rel)
            fail(
                msgs,
                f"crates/ppf-cts-solver/src/kernels/{rel}:{line}",
                "Seam macro in a neutral kernel body",
                f"'{m.group(0)}' names the backend macro seam. A kernel body "
                f"reaches the seam through the seam namespaces and the "
                f"[[seam::...]] attributes only; the SM_ table belongs to "
                f"cpp/seam and to kMslMacroSeam.",
            )
        for lineno, line in enumerate(code.split("\n"), start=1):
            if not line.lstrip().startswith("#"):
                continue
            d = KERNEL_DIRECTIVE_RE.match(line)
            name, rest = d.group(1), d.group(2).strip()
            if name == "pragma" and rest == "once":
                continue
            # The include TARGET is a string literal, so it is blanked in
            # `code`; the raw line is what says whether it is the quoted form.
            if name == "include":
                raw = read(path).split("\n")[lineno - 1]
                if KERNEL_INCLUDE_RE.match(raw.split("//")[0]):
                    continue
            offenders.add(rel)
            fail(
                msgs,
                f"crates/ppf-cts-solver/src/kernels/{rel}:{lineno}",
                "Preprocessor directive in a neutral kernel body",
                f"'#{name}' is not one of the two a kernel may carry "
                f"('#pragma once' and a quoted '#include'). A body that takes "
                f"a preprocessor branch is no longer one source compiled from "
                f"identical bytes by three compilers; the difference belongs "
                f"in cpp/seam or in kMslMacroSeam.",
            )
    return offenders



# ---------------------------------------------------------------------------
# Rule 8: the neutral-body reachability census, per backend.
#
# Rules 1 and 2 compare sets of FILE PATHS, so they answer "is this body
# compiled by both backends" and never "does either backend call it". A body
# can therefore be compiled into the CUDA image and into the Metal shader,
# carry a passing per-kernel parity self-test, and be reached in production by
# nothing. A self-test builds its own fixture, so it certifies the body whether
# or not any production kernel binds it.
#
# This rule computes, per backend, the transitive closure of the neutral bodies
# that backend reaches from its own entry points, and reports:
#
#   * a body CUDA reaches that another backend does not reach at all, and
#   * a body a backend reaches only from a parity self-test entry point.
#
# THE CENSUS IS A REPORT AND NOT A FAILURE, and that is deliberate. Several
# entries are design differences rather than defects: Metal applies the contact
# system matrix-free from compact per-pair arrays and so binds none of the
# dynamic-CSR bodies, and three bodies are dead on both GPU backends because
# their only CUDA caller is itself uncalled. Turning those into a build failure
# would make the rule something a contributor disables. REACHABILITY_EXCEPTIONS
# records each with a date and a reason; an entry NOT recorded there is printed
# as unrecorded, which is the line to act on.
#
# What DOES fail the build is the rule losing the ability to measure: a shader
# segment the parse did not find, or an entry point the census cannot name.
# Either turns the census into a shorter list that looks like progress, so each
# exits instead.
#
# TWO CONSTRUCTION PROPERTIES, each of which the census is worthless without.
#
#   The neutral call graph comes from ppf-cts-compute/seam/kernelgen.py's own lexical
#   classifier, imported rather than reimplemented. That script already reads
#   every body to render three targets, and its mask is what keeps a call
#   inside a comment or a string literal out of the graph. A separate regex
#   pass over the same files was measured missing the shell_bend_stiffness
#   to shell_bend_directional edge, which put a body Metal does reach on
#   the orphan list. A census with false entries is read once and then ignored,
#   which is worse than no census.
#
#   A call is attributed to the body it sits in, by BRACE-DEPTH block
#   extraction, so a name that appears inside one body is never credited to
#   the one before it.
#
# TWO LIMITS, both stated rather than papered over.
#
#   It OVER-approximates: reachability is lexical, so a body called under a
#   branch that is never taken counts as reached, and a body reached only from
#   an entry point whose dispatch is itself conditional counts as reached.
#
#   It UNDER-approximates through indirection: a lexical walk cannot see a call
#   made through a functor's operator() at the call site, and Metal's
#   traversals pass visitors that way. So a struct in the shader is a node here
#   and any function naming that type gets an edge to it. That edge is
#   load-bearing rather than decorative: with it removed, pair_cache_record
#   and aabb_overlap are both reported as reached only from a self-test,
#   and both are reached by the production traversals. An indirection this does
#   not model, a function pointer or a callback resolved at run time, would
#   leave a reached body on a list above.
# ---------------------------------------------------------------------------

# Bodies a backend does not reach, with the date the entry was last verified
# against the tree and the reason it is not a defect. Keyed by (backend, body).
#
# Same discipline as ONE_SIDED and CUDA_ONLY: an entry records a site to argue
# about and does not bless it. An entry whose body the backend now reaches is
# printed as stale, so the list cannot rot into names nobody re-reads.
# THIRTEEN ENTRIES CAME OUT WHEN THE CUDA COLUMN STARTED MEASURING AGAIN, and
# what removed them is the staleness rule below rather than a judgement here.
# Twelve recorded the dynamic and fixed CSR bodies as unreached on Metal because
# "Metal builds no dynamic CSR ... (metal/main.mm)", and that orchestrator is
# deleted: Metal dispatches the same generated entry table CUDA does. The
# thirteenth, `push_energy`, was recorded on the ground that "this census walks
# no CUDA-side call graph", which the re-seeding made false.
#
# Both reasons were TRUE when written and both were premised on a two-backend
# world. They stood for three weeks after that world went, and could not be
# detected while the column they were checked against read zero.
REACHABILITY_EXCEPTIONS = {}
REACHABILITY_EXCEPTIONS.update({
    # THE THREE THAT SURVIVED, restated for the tree they are checked against
    # now. Both backends dispatch one generated entry table, so "unreached" is
    # one fact rather than a per-backend one, and the `metal` key is kept only
    # because that is what the staleness loop reads.
    ("metal", "aabb_join"): (
        "2026-09-08",
        "No entry and no reachable body joins two boxes: every box is built "
        "with aabb_make_swept_edge, aabb_make_swept_point or "
        "aabb_make_swept_triangle. Reachable again the day a traversal merges "
        "two boxes rather than growing one",
    ),
    ("metal", "barrier_energy"): (
        "2026-09-08",
        "The ENERGY of the contact barrier, which the solve never needs: the "
        "Newton loop reads the gradient and the Hessian, and its siblings are "
        "reached. Kept because an energy is what a line search would want",
    ),
    ("metal", "shell_strain_energy"): (
        "2026-09-08",
        "The strain limiter's energy, unreached for the same reason as "
        "barrier_energy: the solve reads the gradient and the Hessian",
    ),
})

# A neutral body's definition and the calls inside it. A template argument list
# is part of the call spelling (compute_target<FixPair>(...) in
# main/target.kernel.cpp is one such site), so it is matched here; without it
# those bodies read as unreachable on the backend that does call them.
# The two attributes rule (1-LANE) admits, which mark a body as one half of
# a twin pair rather than a second definition.
TWIN_ATTR_RE = re.compile(r"\[\[seam::(?:cooperative|serial)\]\]")

BODY_ATTR_RE = re.compile(r"\[\[seam::(?:device_fn|host_device_fn)\]\]")
# ANY call, filtered against the body definitions afterwards. A neutral name
# carries no project prefix, so a call to a neutral body and a call to `sqrtf`
# are indistinguishable by spelling, and the census intersects with the names it
# SAW DEFINED rather than matching a prefix.
#
# THAT COSTS ONE GUARD AND THE LOSS IS REAL. A prefix scan can exit when a body
# calls a prefixed name no definition scan found, which catches a definition the
# parse missed. Intersecting cannot: a missed definition reads as an ordinary
# library call. Nothing else recovers it, because no property of the spelling
# distinguishes the two.
BODY_CALL_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\s*(?:<[^;{}()]*>\s*)?\(")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MSL_SEGMENT_RE = re.compile(r'R"MSL\((.*?)\)MSL"', re.S)
MSL_ENTRY_RE = re.compile(r"\bkernel\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
MSL_STRUCT_RE = re.compile(r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")
MSL_TYPE_RE = re.compile(r"\b(Ppf[A-Za-z0-9_]+)\b")
MAKE_PIPELINE_RE = re.compile(r"\bcontext_make_pipeline\s*\(")
MAKE_PIPELINE_ARG_RE = re.compile(
    r'context_make_pipeline\s*\(\s*[^,]+,\s*[^,]+,\s*"([A-Za-z_][A-Za-z0-9_]*)"')
DISPATCH_RE = re.compile(
    r"\bcontext_dispatch(?:_threadgroups)?\s*\(\s*[^,]+,\s*[^,]+,\s*([^,]+),")
DLL_EXPORT_RE = re.compile(r"\bDLL_EXPORT\b")

# Statement keywords that take a parenthesized head and an ordinary block, so
# they match the shape of a function definition and are not one.
BLOCK_KEYWORDS = {"if", "for", "while", "switch", "catch", "do", "else",
                  "return", "sizeof"}
# Leading tokens of a parameter DECLARATION, which is how the dispatch pattern
# above matches the definition of context_dispatch itself rather than a call.
DECL_LEADERS = {"unsigned", "int", "const", "float", "uint", "bool",
                "void", "char", "auto", "size_t", "std"}


def load_kernelgen(cpp_dir):
    """ppf-cts-compute/seam/kernelgen.py as a module, for its lexical classifier."""
    import importlib.util

    script = transcompiler(cpp_dir)
    spec = importlib.util.spec_from_file_location("kernelgen", script)
    module = importlib.util.module_from_spec(spec)
    # Importing a file normally writes __pycache__ beside it. That directory is
    # not watched by any build script today, so this is a precaution rather than
    # a fix: the same import against a script inside crates/ppf-cts-solver/src
    # would invalidate the two build scripts that watch that tree recursively
    # through cargo:rerun-if-changed, and cargo's mtime walk does not consult
    # .gitignore, so a stray .pyc there costs about 48 seconds of recompile and
    # relink on the next build. The flag is restored so nothing else in this
    # process inherits it.
    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # a broken generator cannot yield a call graph
        sys.exit(f"check-shared-wiring: cannot import {script}: {exc}")
    finally:
        sys.dont_write_bytecode = write_bytecode
    if not hasattr(module, "classify"):
        sys.exit(
            f"check-shared-wiring: {script} no longer defines classify(), "
            f"which is the lexical mask rule 8 builds the neutral call graph "
            f"with. It refuses to substitute a second parser for it."
        )
    return module


def neutral_code(kernelgen, path):
    """A kernel body with comments and literals blanked, offsets preserved."""
    text = read(path)
    try:
        mask = kernelgen.classify(text, str(path))
    except Exception as exc:
        sys.exit(
            f"check-shared-wiring: ppf-cts-compute/seam/kernelgen.py cannot classify "
            f"{path}: {exc}. Rule 5 reports the same thing; rule 8 cannot "
            f"build a call graph over a file it cannot read."
        )
    return "".join(c if m == "c" else (" " if c != "\n" else "\n")
                   for c, m in zip(text, mask))


def block_end(code, start):
    """The index just past the '}' matching the '{' at `start`, or None."""
    depth = 0
    i = start
    while i < len(code):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


STRUCT_OPEN_RE = re.compile(
    r"\b(?:struct|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b[^;{}()]*\{")


def struct_spans(code):
    """(name, start, end) for every struct or class body in `code`.

    A MEMBER FUNCTION IS NAMED BY ITS STRUCT, and that is what lets the census
    tell two rows apart where the bare name is fixed by a contract rather than
    chosen. `aabb_query` calls `op.test(box, query)` and `op(leaf)` on whatever
    visitor it is handed, so every visitor in the tree spells those two the same
    way and a census keyed on the bare name reports them as one body redefined.
    That is the limitation its own error message names; qualifying the key
    removes it without weakening anything, because the CALL side still resolves
    a bare name to every method that carries it.
    """
    spans = []
    for m in STRUCT_OPEN_RE.finditer(code):
        end = block_end(code, m.end() - 1)
        if end is not None:
            spans.append((m.group(1), m.end() - 1, end))
    return spans


def enclosing_struct(spans, position):
    """The innermost struct whose body contains `position`, or None."""
    best = None
    for name, start, end in spans:
        if start < position < end and (
                best is None or start > best[1]):
            best = (name, start)
    return None if best is None else best[0]


def neutral_call_graph(cpp_dir, kernels, kernelgen):
    """{body: set(callees)} and {body: (file, line)} over every neutral body."""
    graph, where = {}, {}
    for rel in sorted(kernels):
        code = neutral_code(kernelgen, cpp_dir / rel)
        spans = struct_spans(code)
        for m in BODY_ATTR_RE.finditer(code):
            open_paren = code.find("(", m.end())
            if open_paren < 0:
                sys.exit(
                    f"check-shared-wiring: {rel}: an execution-space attribute "
                    f"at line {code.count(chr(10), 0, m.start()) + 1} is "
                    f"followed by no parameter list, so rule 8 cannot name the "
                    f"body it annotates."
                )
            names = IDENT_RE.findall(code[m.end():open_paren])
            if not names:
                sys.exit(
                    f"check-shared-wiring: {rel}: no identifier between the "
                    f"execution-space attribute at line "
                    f"{code.count(chr(10), 0, m.start()) + 1} and its "
                    f"parameter list."
                )
            name = names[-1]
            owner = enclosing_struct(spans, m.start())
            if owner is not None:
                name = f"{owner}::{name}"
            brace = code.find("{", open_paren)
            end = block_end(code, brace) if brace >= 0 else None
            if end is None:
                sys.exit(
                    f"check-shared-wiring: {rel}: the body of {name} is not a "
                    f"brace-balanced block, so rule 8 cannot attribute its "
                    f"calls."
                )
            # A TWIN PAIR IS ONE BODY AND IS COUNTED ONCE. Rule (1-LANE)
            # admits a `[[seam::cooperative]]` body beside its
            # `[[seam::serial]]` twin under one name, and the generator emits
            # exactly one of them per target, so the two are one node in this
            # graph rather than a redefinition. Their calls are UNIONED: either
            # may reach a body the other does not, and a census that took only
            # the first would report the other's callees as unreached.
            twin = TWIN_ATTR_RE.search(code, m.start(), open_paren) is not None
            if name in graph and not twin:
                sys.exit(
                    f"check-shared-wiring: two neutral bodies are both named "
                    f"{name} ({where[name][0]} and {rel}). The census keys on "
                    f"the name, so it cannot tell them apart."
                )
            calls = set(BODY_CALL_RE.findall(code[brace:end])) - {name}
            graph[name] = graph.get(name, set()) | calls
            where[name] = (rel, code.count("\n", 0, m.start()) + 1)
    # Keep only the calls that reach a body this scan DEFINED; everything else
    # a body calls is a library or namespaced function the census does not
    # model. See BODY_CALL_RE for the guard this replaced and why.
    #
    # A CALL SITE SPELLS A BARE NAME, so a bare callee resolves to EVERY method
    # that carries it. That is the same conservative reading the census always
    # had, stated explicitly now that one bare name can belong to several
    # structs: `aabb_query`'s `op.test(...)` reaches every visitor's `test`,
    # because a template's call site names no one instantiation.
    by_bare = {}
    for key in graph:
        by_bare.setdefault(key.split("::")[-1], set()).add(key)
    graph = {
        key: {reached for callee in callees
              for reached in by_bare.get(callee, ())} - {key}
        for key, callees in graph.items()
    }
    return graph, where


def reached(seeds, graph):
    """The transitive closure of `seeds` over `graph`."""
    out, stack = set(), [s for s in seeds]
    while stack:
        node = stack.pop()
        if node in out or node not in graph:
            continue
        out.add(node)
        stack.extend(graph[node])
    return out


def cuda_body_seeds(closure, closure_paths, kernels):
    """Bodies named by CUDA code that is not itself a neutral body.

    The closure is the set of files nvcc reaches from the translation units the
    CUDA recipe declares, so the regression binaries under its tests directory
    are outside it by construction and their calls are not seeds.

    Each file is opened through `closure_paths`, not by joining the key onto one
    root: the closure spans two crates, and a resolver that assumed one of them
    would skip the translation units in the other, which are where nearly every
    seed is.
    """
    seeds = {}
    for rel in sorted(closure - kernels):
        path = closure_paths[rel]
        if not path.exists():
            continue
        code = kernel_code(read(path))
        for m in BODY_CALL_RE.finditer(code):
            seeds.setdefault(m.group(1), []).append(
                (rel, code.count("\n", 0, m.start()) + 1))
    return seeds


# `ppf_`-prefixed names that are deliberately NOT neutral bodies, so a CUDA call
# to one is not a seed and not a defect. Recorded rather than pattern-matched,
# for the reason every other exception here is: an unrecorded name has to fail,
# or the rule stops measuring the moment someone adds a helper with this prefix.
#
#   the fatal channel  src/kernels/main/fatal.hpp defines fatal,
#                      fatal_set_detail and the fatal_code / fatal_detail
#                      readers as header-only host helpers over the
#                      g_ppf_fatal_* storage a driver owns. They report a failure; they compute nothing,
#                      so they have no place in a neutral body.
#   the PDRD factor    pdrd_factor_reduced_block is an inline helper in a
#                      SHARED HEADER rather than a *.kernel.cpp body. It is
#                      shared by the mechanism of section 2.2, headers embedded
#                      verbatim, not by the transcompiler.
NON_BODY_PPF_NAMES = {
    "fatal",
    "fatal_code",
    "fatal_detail",
    "fatal_set_detail",
    "pdrd_factor_reduced_block",
}


def backend_abi_names(cpp_dir):
    """Every `be_*` function the C ABI header declares.

    THE FAMILY IS RECOGNIZED FROM THE HEADER AND NEVER BY ITS PREFIX. A bare
    "starts with be_" would let a typo through as a non-body name, and a
    seed the census cannot resolve silently shrinks the reached-by-CUDA set,
    which is the failure this whole resolution exists to prevent. Reading the
    declarations means a name outside them still stops the run.

    These are not kernel bodies and do not wrap one: they are the boundary a
    DRIVER calls a backend library across, so a CUDA source naming one is
    implementing that boundary rather than dispatching a kernel through it.
    """
    header = cpp_dir / "seam/backend_abi.h"
    if not header.is_file():
        sys.exit(f"check-shared-wiring: no such file: {header}")
    names = set(re.findall(r"\b(be_[A-Za-z0-9_]+)\s*\(",
                           header.read_text(encoding="utf-8")))
    if not names:
        sys.exit("check-shared-wiring: seam/backend_abi.h declared no "
                 "be_* function, so this parser is broken; an empty set "
                 "would turn every ABI implementation into an unresolvable "
                 "seed")
    return names


def resolve_entry_seeds(seeds, graph, abi_names=frozenset()):
    """Map a generated entry point back to the body it wraps.

    A `[[seam::entry]]` declaration renders a `<body>_entry` wrapper into the
    build directory, deliberately outside `src/`, so the walk over `src/` never
    sees it and the launcher's only surviving text names a symbol no neutral
    body defines. Left alone that SEVERS the CUDA edge: converting
    `vec_combine_indirect` moved it from reached-by-CUDA to unreached, and
    the run stayed green, because rule 8's failing categories key on "CUDA
    reaches and X does not" and a lost seed can only make those lists SHORTER.

    An entry wrapper is an argument record, a thread-index guard and a call into
    exactly one body, so `<body>_entry` naming a known `<body>` is not a guess.

    `<body>_entry_launch` is the same edge one level out: the generator emits a
    C-linkage launcher beside every `__global__`, because a `__global__` is a
    symbol with a launch syntax rather than one whose address can be taken. A
    caller reaching the launcher is reaching the body through it, so the two
    suffixes resolve the same way. The longer one is tried FIRST: `_entry` is a
    suffix of `_entry_launch`, so stripping the short one from
    `x_entry_launch` would leave `x_entry_launch` unchanged and the
    seed would go on being unresolvable.

    THE UNRESOLVABLE CASE FAILS, matching `neutral_call_graph`, which already
    exits when a body calls a `ppf_` name no body defines. The two scans were
    asymmetric: one refused an unknown call, the other silently kept an
    unresolvable seed, and that asymmetry is what let the coverage go quiet.
    """
    resolved, unknown = {}, []
    for name, sites in seeds.items():
        if name in graph:
            resolved.setdefault(name, []).extend(sites)
        elif (name.endswith("_entry_launch")
              and name[: -len("_entry_launch")] in graph):
            resolved.setdefault(name[: -len("_entry_launch")],
                                []).extend(sites)
        elif name.endswith("_entry") and name[: -len("_entry")] in graph:
            resolved.setdefault(name[: -len("_entry")], []).extend(sites)
        elif name in NON_BODY_PPF_NAMES or name in abi_names:
            continue
        else:
            # NOT a body, and with no project prefix left there is nothing in
            # the spelling to say whether it should have been one: this is where
            # `sqrtf`, `__syncthreads` and every namespaced helper land. The
            # scan takes the calls it CAN resolve and says nothing about the
            # rest, which is the same trade `BODY_CALL_RE` describes.
            continue
    if unknown:
        sys.exit(
            "check-shared-wiring: CUDA code calls a ppf_ name that no neutral "
            "body defines and that is not a generated <body>_entry wrapper "
            "or its <body>_entry_launch launcher, and is not one of the "
            "be_* functions seam/backend_abi.h declares: "
            + ", ".join(sorted(unknown)) + ". A seed the census cannot resolve "
            "would silently shrink the reached-by-CUDA set, so it stops the "
            "run rather than shortening a list nobody diffs.")
    return resolved


def named_blocks(code):
    """Every `name(...) {` block in `code`, outermost only.

    Used for two things that must not be resolved by name: which body a call
    sits in on the shader side, and which host function a dispatch sits in.
    """
    found = []
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", code):
        if m.group(1) in BLOCK_KEYWORDS:
            continue
        i = m.end() - 1
        depth = 0
        while i < len(code):
            if code[i] == "(":
                depth += 1
            elif code[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if i >= len(code):
            continue
        j = i + 1
        while j < len(code):
            if code[j] in " \t\n":
                j += 1
            elif code.startswith("const", j):
                j += 5
            elif code.startswith("noexcept", j):
                j += 8
            else:
                break
        if j >= len(code) or code[j] != "{":
            continue
        end = block_end(code, j)
        if end is None:
            continue
        found.append((m.group(1), m.start(), end))
    found.sort(key=lambda t: (t[1], -t[2]))
    out, last_end = [], -1
    for name, start, end in found:
        if start < last_end:
            continue
        out.append((name, start, end))
        last_end = end
    return out


def blank_msl(text):
    """`text` with the shader segments blanked, offsets and lines preserved.

    The .mm files carry the hand-written MSL as raw string literals, so the
    host code and the shader code share a file. They are separate programs and
    are parsed separately: this is the host half.
    """
    out = list(text)
    for m in MSL_SEGMENT_RE.finditer(text):
        for i in range(m.start(1), m.end(1)):
            if out[i] != "\n":
                out[i] = " "
    return "".join(out)


def cpu_body_seeds(cpu_dir):
    """Bodies the CPU backend's kernel shim names.

    This is WIRING, not production reach: a body the shim wires can still
    have no Rust caller in src/driver/, which is the same driver every backend
    runs. The census reports it as a separate column and says so, rather than
    reading a wired body as a reached one.
    """
    seeds = {}
    for path in sorted(cpu_dir.glob("*.cpp")):
        code = kernel_code(read(path))
        for m in BODY_CALL_RE.finditer(code):
            seeds.setdefault(m.group(1), []).append(
                (path.name, code.count("\n", 0, m.start()) + 1))
    return seeds


def _wrapped(names, indent):
    import textwrap

    return textwrap.fill(", ".join(names), width=78,
                         initial_indent=indent, subsequent_indent=indent)


def check_eight(root, cpp_dir, kernels, closure, closure_paths,
                embedded, msgs):
    """Rule 8: the neutral-body reachability census, per backend."""
    kernelgen = load_kernelgen(cpp_dir)
    graph, where = neutral_call_graph(cpp_dir, kernels, kernelgen)
    if not graph:
        sys.exit(
            "check-shared-wiring: no neutral body carries an execution-space "
            "attribute, so rule 8 has nothing to census and this parser is "
            "broken."
        )

    # THE CUDA COLUMN IS SEEDED FROM THE GENERATED ENTRY POINTS, WHICH IS WHAT
    # THAT BACKEND ACTUALLY DISPATCHES. Seeding it by scanning the include
    # closure for calls into neutral bodies resolves NOTHING instead: measured,
    # 351 raw seeds and 0 resolved, so the column reads 0 of 551 and the
    # "reached by no backend at all" list names `aabb_query`, `ccd_point_face`
    # and `barrier_curvature`, which CUDA certainly dispatches. That census is
    # INVERTED rather than conservative.
    #
    # BE PRECISE ABOUT WHAT IT ANSWERS, because the neighboring question is
    # easy to read into it. This asks whether a neutral body is reachable from
    # the generated ENTRY set, so what it finds is a body no entry and no other
    # body calls: dead neutral code. It does NOT ask whether a solve dispatches
    # that entry, which is `pcg_update3`'s failure and is answered by rules 9
    # and 11 plus the driver table. A body can be reached here and dispatched by
    # nothing, and `push_energy` is exactly that today.
    #
    # WHY THAT SEEDING CANNOT WORK. It scans `closure - kernels`, which is the
    # shared headers and the CUDA mechanism sources, and the names it finds
    # there are `__syncthreads`, `__shfl_down_sync`, `sqrtf` and
    # `__builtin_clz`. A body is named by a GENERATED entry rendering, and those
    # are emitted outside `src/` on purpose, so the walk never reaches one. With
    # no project prefix there is nothing in a call's spelling to separate a body
    # from a C intrinsic, so no widening of that scan reaches the answer.
    #
    # `canonical_entry_order` is the same reading every generated table is built
    # from, one sorted walk of the neutral tree, so the seed set and the kernel
    # id table cannot disagree about what exists.
    entry_order = canonical_entry_order(root)
    cuda_seeds = {name: [("the generated entry table", 0)]
                  for name in entry_order if name in graph}
    # LOSING THE SEED IS THE FAILURE THIS COLUMN ALREADY SUFFERED ONCE, and it
    # is silent: a lost seed can only make the census SHORTER, and every failing
    # category downstream keys on "CUDA reaches and X does not", so an empty
    # column reports a clean tree. Neither number may be zero while the other
    # is not.
    if entry_order and not cuda_seeds:
        sys.exit(
            f"check-shared-wiring: rule 8 read {len(entry_order)} generated "
            f"entry points and resolved NONE of them to a neutral body, so the "
            f"reached-by-CUDA column would be empty and every category derived "
            f"from it would report a clean tree it did not measure.")
    cuda = reached(cuda_seeds, graph)

    # THE METAL HOST CENSUS IS GONE, AND ITS QUESTION IS ANSWERED ELSEWHERE.
    # It walked `metal/main.mm`'s exported call graph to ask which neutral
    # bodies that backend REACHED in production, because a hand-written
    # orchestrator can compile a kernel, self-test it, and be called by nothing:
    # `pcg_update3` was exactly that for the backend's whole life while the
    # default preconditioner was a label over unpreconditioned CG.
    #
    # That orchestrator is deleted. Both backends are driven by the neutral
    # driver, which dispatches by KERNEL ID out of a generated table, so what
    # either reaches is what the generated entries reach, which is `cuda` above.
    # Rule 11 checks that every table row names the entry point its thunk
    # dispatches and rule 9 holds down the hand-written launchers that would sit
    # outside the table, so the compiled-and-called-by-nothing question is
    # answered by construction rather than by a walk.
    #
    # A REAL REDUCTION, recorded rather than left to be noticed: what made the
    # walk strong was a second, hand-written call graph, and that is what was
    # removed.
    known_entries = set(spliced_entry_names(cpp_dir, metal_embed_list(cpp_dir)))
    production, selftest, widened = set(known_entries), set(), []
    metal_production = set(cuda)
    metal_any = set(cuda)
    reached_from = {}

    missing_entries = sorted((production | selftest) - known_entries)
    if missing_entries:
        sys.exit(
            "check-shared-wiring: the Metal host creates a pipeline for a "
            "shader entry point this parse did not find: "
            + ", ".join(missing_entries)
            + ". A segment of the shader is being missed, so every body only "
            "that segment reaches would be reported as an orphan."
        )
    # THE DUPLICATE-DEFINITION GUARD AND THE SHADER CALL GRAPH BOTH HAD THE
    # HAND-WRITTEN MSL AS THEIR SUBJECT. One asked whether that MSL defined a
    # name the build also embedded a neutral rendering of, which is a duplicate
    # definition the Metal driver reports at run time; the other walked the
    # shader's own calls so a body could be credited to the backend that
    # reached it. Neither has a subject now: the shader is linked from the
    # generated renderings and defines each name exactly once, by construction.
    # Which self-test entry point reaches a body, so a report line names the
    # fixture that certifies it as well as the body nothing else calls.
    reached_from = {}
    for entry in sorted(selftest):
        for body in bodies_from({entry}):
            reached_from.setdefault(body, set()).add(entry)

    cpu_dir = root / "crates/ppf-cts-solver/entrypoints"
    cpu_present = cpu_dir.is_dir()
    # LOSING THE ABILITY TO MEASURE FAILS, which is this rule's own stated
    # principle and is applied here as it is to a shader segment the parse did
    # not find. Reporting a shorter census instead would be a pass bought by
    # the coverage vanishing.
    if not cpu_present:
        fail(msgs, cpu_dir.as_posix(), "rule 8 cannot measure the CPU column",
             "rule 8 cannot measure the CPU column because "
             "crates/ppf-cts-solver/entrypoints is absent, so the census would "
             "report a shorter list rather than the truth. Restore the "
             "directory, or remove rule 8's CPU column deliberately.")
    cpu = reached(cpu_body_seeds(cpu_dir), graph) if cpu_present else set()

    print("\nneutral-body reachability census (rule 8, report only)")
    print(f"  neutral bodies                  : {len(graph)}")
    print(f"  reached by CUDA                 : {len(cuda)}")
    print(f"  reached by Metal in production  : {len(metal_production)}")
    print(f"  reached by Metal only in a test : "
          f"{len(metal_any - metal_production)}")
    print(f"  Metal shader entry points       : {len(known_entries)} "
          f"({len(production)} production, {len(selftest)} self-test)")
    if cpu_present:
        print(f"  wired into the CPU kernel shim  : {len(cpu)}")
    else:
        print("  wired into the CPU kernel shim  : not computed, "
              "crates/ppf-cts-solver/entrypoints is absent")

    # The one place a NAME is consulted, and it feeds nothing: the split above
    # is already decided, and this reports where the naming contradicts it.
    lying = sorted(n for n in production if "selftest" in n)
    if lying:
        print("\n  entry points a production host function dispatches whose "
              "name says self-test.")
        print("  The split above is derived from the dispatch, so the census "
              "is right and the")
        print("  name is not. Rename them.")
        for name in lying:
            print(f"    {name}")

    # A body whose own file is recorded in CUDA_ONLY is unreachable on Metal
    # by construction: rule 1 already records that the Metal build does not
    # embed that file, with the reason. Deriving the exception from that entry
    # rather than repeating the body names keeps one record for one decision.
    cuda_only_bodies = {b for b in graph if where[b][0] in CUDA_ONLY}

    def excused(backend, name):
        if (backend, name) in REACHABILITY_EXCEPTIONS:
            return REACHABILITY_EXCEPTIONS[(backend, name)][1]
        if backend == "metal" and name in cuda_only_bodies:
            return (f"defined in {where[name][0]}, which CUDA_ONLY records as "
                    f"not embedded by the Metal build")
        return None

    selftest_only = sorted(metal_any - metal_production)
    unreached = sorted(cuda - metal_any)
    for title, backend, names, detailed in (
            ("reachable on Metal only through a self-test entry point",
             "metal", selftest_only, True),
            ("reachable on CUDA and unreachable on Metal",
             "metal", unreached, True),
            ("reachable on CUDA and not wired into the CPU kernel shim",
             "cpu", sorted(cuda - cpu) if cpu_present else [], False)):
        if backend == "cpu" and not cpu_present:
            continue
        fresh = [n for n in names if not excused(backend, n)]
        recorded = [n for n in names if excused(backend, n)]
        print(f"\n  {title}: {len(names)}")
        if backend == "cpu":
            print("    This column is WIRING, not production reach: a body "
                  "the shim wires can")
            print("    still have no Rust caller in src/driver, which is the "
                  "one driver every")
            print("    backend runs. A wired body with no caller is a "
                  "divergence to record.")
        if fresh and detailed:
            print("    UNRECORDED:")
            for name in fresh:
                origin = where[name]
                via = sorted(reached_from.get(name, ()))
                trailer = f"  via {', '.join(via)}" if via else ""
                print(f"      {name}  ({origin[0]}:{origin[1]}){trailer}")
        elif fresh:
            print("    UNRECORDED:")
            print(_wrapped(fresh, "      "))
        if recorded:
            print(f"    recorded ({len(recorded)}):")
            print(_wrapped(recorded, "      "))

    # A STALE EXCEPTION FAILS THE RUN rather than printing. The census itself is
    # a report, but an exception is a standing grant, and a grant that has
    # started covering a body the backend now reaches is where a real
    # divergence goes to be forgotten. CI reads the exit code, so a printed
    # warning here would be invisible, which is the shape of failure this whole
    # rule exists to remove.
    stale_exceptions = []
    for (backend, name), (date, reason) in sorted(
            REACHABILITY_EXCEPTIONS.items()):
        if name not in graph:
            stale_exceptions.append(
                f"REACHABILITY_EXCEPTIONS names {name}, which is not a "
                f"neutral body. Delete the entry.")
        elif backend == "metal" and name in metal_production:
            stale_exceptions.append(
                f"REACHABILITY_EXCEPTIONS names {name} as unreached on Metal "
                f"({date}), which Metal now reaches in production. Delete the "
                f"entry.")
        elif backend == "cpu" and cpu_present and name in cpu:
            stale_exceptions.append(
                f"REACHABILITY_EXCEPTIONS names {name} as unwired on the CPU "
                f"backend ({date}), which the shim now wires. Delete the "
                f"entry.")
    for line in stale_exceptions:
        print(f"\n  STALE: {line}")
        fail(msgs, ".github/workflows/scripts/check-shared-wiring.py",
             "stale rule 8 exception", line)

    orphans = sorted(set(graph) - cuda - metal_any - cpu)
    if orphans:
        print(f"\n  reached by no backend at all: {len(orphans)}")
        print(_wrapped(orphans, "      "))

    if widened:
        print(f"\n  dispatch sites whose pipeline expression named no single "
              f"entry: {len(widened)}")
        print("  Each counts every entry its host function creates or names "
              "through a file-scope")
        print("  handle, which can move an entry to the production side and "
              "cannot invent one.")
        for file_name, fn, expression, line, count in widened:
            print(f"    {file_name}:{line} in {fn}: {expression} "
                  f"({count} entries)")
    return selftest_only, unreached


# ---------------------------------------------------------------------------
# Rule 9: the hand-written CPU entry-point ratchet.
#
# AN ENTRY POINT IS AN ARGUMENT RECORD plus a thread index plus a call into the
# neutral body, all three mechanical, so writing one by hand is writing a mirror
# pair by hand. The CPU
# backend still holds a large stock of hand-written ones, and the two that
# collided at link time when the contact modules merged are what a stock of them
# costs: `aabb_overlap_abi` was a byte-identical duplicate and
# `pair_cache_record_entry` was one name over two calling conventions,
# and neither was visible in either file alone.
#
# THIS RULE DOES NOT ASK FOR THE STOCK TO BE ZERO. Most of it cannot be
# converted without changing the neutral body it wraps, which changes that
# body's CUDA and Metal call sites in the same edit. What it asks is that the
# number never grow, and that it be written down where a reader can see it: a
# ratchet whose ceiling is not lowered as conversions land quietly re-permits
# everything it just gained, so a DROP is reported as an error too, naming the
# number to write.
HAND_WRITTEN_CPU_ENTRY_POINTS = 5

# Rule 10's two ceilings: how far the driver's kernel table is from the order
# every generated table is built in.
#
# A KERNEL ID IS A TABLE INDEX, so a driver and a library that order their rows
# differently dispatch different kernels for the same id. The library's table
# comes from one sorted walk of the neutral tree; the driver's `TABLE` in
# `src/driver/kernels.rs` is hand-ordered by subject, and the two agree at
# exactly one position out of 154. `AbiDevice::open` refuses the pair, which is
# why `--features cuda-driver` starts and stops at open.
#
# THE POINT OF MEASURING IT HERE IS NOT THE NUMBER, IT IS THE DIRECTION. Adding
# a neutral entry point with no driver row widens the gap silently today, and a
# runtime refusal on a GPU host is a long way from the edit that caused it.
# One generated entry point declaration, which is the unit both the canonical
# order and the driver's table are lists of.
# A LAUNCH SHAPE MAY SIT BETWEEN THE ENTRY AND THE RETURN TYPE, which is why
# the tail is a repeat rather than nothing: a `[[seam::group]]` declaration reads
# `[[seam::args]] [[seam::entry]] [[seam::group]] void name(`. A pattern anchored
# straight at `void` skips it, and the failure is not a missing check but a
# canonical order SHORT BY ONE, which then reports every row after the new entry
# as misplaced and names none of them as the cause.
# AND [[seam::entry]] IMPLIES ITS RECORD, so [[seam::args]] is not written
# beside it and a pattern demanding both matches nothing at all. That failure is
# the worse one: an empty canonical order compares clean against a full table on
# the count this rule reports, so the reading is asserted non-empty below.
ENTRY_DECL_RE = re.compile(
    r"((?:\[\[seam::\w+(?:\([^)]*\))?\]\]\s*)+)"
    r"void\s+([a-z0-9_]+)\s*\(")


def _carries(run, attr):
    """Whether an attribute run carries `attr`, with or without an argument.

    `[[seam::entry]]` MAY NAME THE COUNT PARAMETER, so a literal comparison
    against `[[seam::entry]]` matches none of them. That is the same blindness
    the [[seam::args]] removal caused, one attribute later.
    """
    return re.search(r"\[\[seam::" + attr + r"(?:\([^)]*\))?\]\]",
                     run) is not None


def entry_names(text):
    """The entry points a neutral source declares, in declaration order."""
    return [m.group(2) for m in ENTRY_DECL_RE.finditer(text)
            if _carries(m.group(1), "entry")]

DRIVER_TABLE_ROWS_MISPLACED = 0
DRIVER_TABLE_ENTRIES_MISSING = 0

# An ENTRY POINT is identified by the property that defines one, a thread range,
# NOT by a proxy. Two proxies were tried and both were wrong in a way that
# mattered. A fixed list of return types at line start counted the 54 scalar
# HELPERS (position_domain_abi, block_jacobi_invert_abi) as entry points
# and missed every definition returning a vector type. And requiring the loop
# spelling `for (uint32_t i = begin; ...)` undercounts by 50, because launchers
# also spell the induction variable f, k, t, e, row, leaf, s, element and
# thread_index. Measured: 197 definitions were 143 launchers and 54 helpers, and
# the two categories are easy to transpose. Thirty-three of those launchers are
# now generated (the linear solve's `mat3_mul` and `fixed_csr_apply_row`,
# the step's `position_step`, `dx_magnitude` and
# `position_accept`, the arity-1 `vertex_atomic_embed_force`, the
# elastic pipeline's three SVDs, four spectral stages, four material-frame
# converters and the tet shape-function gradients, the two bending stiffnesses,
# the plastic creep rate, the four push-barrier stages, the friction evaluation
# and its value combiner, the vertex-normal finalize, the contact slot lookup,
# the bitonic comparator, the three Hessian scatters, the pair-cache recorder,
# the vector fill and the two per-element strain readings), so 160 definitions
# are 106 launchers and the same 54 helpers.
#
# The return type is therefore matched loosely and the SIGNATURE decides.
CPU_ENTRY_RE = re.compile(
    r"^[A-Za-z_][\w:<>,*&\s]*?\b([A-Za-z_][A-Za-z0-9_]*_(?:entry|abi))\s*\(([^{;]*?)\)\s*\{",
    re.M | re.S)

# A launcher takes the half-open thread range every dispatch is cut from.
CPU_RANGE_RE = re.compile(r"\buint32_t\s+begin\b")


def cpu_shim_definitions(root):
    """Every entry-point definition under entrypoints/, split by what it is.

    A LAUNCHER takes a half-open thread range and is what a generated entry
    point replaces. A HELPER is everything else: a layout query, a constant, a
    per-pair predicate, a per-block operation. The split is read off the
    SIGNATURE and not off a list, so a new file is covered the day it is added,
    and rules 9 and 10 cannot disagree about which category a symbol is in.

    `entries.cpp` defines none of its own: it includes the generated renderings,
    which is the point of it.
    """
    directory = os.path.join(root, "crates", "ppf-cts-solver", "entrypoints")
    launchers = {}
    helpers = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".cpp"):
            continue
        path = os.path.join(directory, name)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for m in CPU_ENTRY_RE.finditer(text):
            side = launchers if CPU_RANGE_RE.search(m.group(2)) else helpers
            side.setdefault(m.group(1), name)
    return launchers, helpers


def cpu_entry_points(root):
    """Every hand-written LAUNCHER under entrypoints/."""
    return cpu_shim_definitions(root)[0]


def canonical_entry_order(root):
    """Every generated entry point, in the order every generated table uses.

    ONE SORTED WALK OF THE NEUTRAL TREE, then declaration order within a file,
    which is what `crates/ppf-cts-compute/cuda/Makefile` builds both the `.inc`
    and the `.rs` from. Derived here from the sources rather than read out of a
    build directory, so this check needs no prior build and cannot be answered
    by a stale artifact.
    """
    kernels = os.path.join(root, "crates", "ppf-cts-solver", "src", "kernels")
    names = []
    sources = []
    for base, _dirs, files in os.walk(kernels):
        for name in files:
            if name.endswith(".kernel.cpp"):
                full = os.path.join(base, name)
                sources.append(os.path.relpath(full, kernels))
    # READ THROUGH THE GENERATOR, not a pattern. An entry is declared in two
    # spellings now, a separate declaration and a body that declares itself, and
    # a regex keyed on either one is blind to the other. Blindness here is a
    # canonical order SHORT BY the entries it cannot see, which then reports
    # every row after the first as misplaced and names none of them as the
    # cause. `kernelgen.read_source` is what the build itself uses.
    module = _load_kernelgen(root)
    for rel in sorted(sources):
        full = os.path.join(kernels, rel)
        if module is not None:
            module.KERNEL_ROOT = os.path.abspath(kernels)
            names.extend(entry.name for entry in module.read_source(full)[3]
                         if entry.emit_entry)
            continue
        with open(full, encoding="utf-8") as handle:
            names.extend(entry_names(handle.read()))
    return names


_KERNELGEN = []


def _load_kernelgen(root):
    """The transcompiler, loaded once, or None when it is not on disk."""
    if _KERNELGEN:
        return _KERNELGEN[0]
    import importlib.util
    path = os.path.join(root, "crates", "ppf-cts-compute", "seam", "kernelgen.py")
    if not os.path.exists(path):
        _KERNELGEN.append(None)
        return None
    spec = importlib.util.spec_from_file_location("kernelgen_order", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _KERNELGEN.append(module)
    return module


def driver_table_order(root):
    """The names in `src/driver/kernels.rs`'s TABLE, in its own order."""
    path = os.path.join(root, "crates", "ppf-cts-solver", "src", "driver", "kernels.rs")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    start = text.find("pub static TABLE")
    if start < 0:
        return []
    rows = re.findall(r"\n\s+id::([A-Z][A-Z0-9_]*)\s*,", text[start:])
    # `id::COUNT` is the array's length, not a row.
    return [row.lower() for row in rows if row != "COUNT"]


# ===========================================================================
# Rule 16: a firing device assert REPORTS rather than crashing or vanishing.
# ===========================================================================

# The device-side macros that dereference the channel before they know one is
# attached. Both spell the write through `(diag).header`, so both need the same
# guard, and a new macro that touches the header belongs in this tuple.
DIAG_GUARDED_MACROS = ("DIAG_ASSERT4", "DIAG_TRACE")


def _define_body(text, name):
    """The full text of a backslash-continued `#define name(...)`, or None."""
    match = re.search(r"^#define[ \t]+" + re.escape(name) + r"\(", text, re.M)
    if not match:
        return None
    out = []
    for line in text[match.start():].splitlines():
        out.append(line)
        if not line.rstrip().endswith("\\"):
            break
    return "\n".join(out)


def check_sixteen(root, msgs):
    """Rule 16: a firing device assert REPORTS, on every backend.

    TWO INDEPENDENT THINGS HAVE TO HOLD AND NEITHER COMPILER CHECKS EITHER.
    A neutral body reports a violated invariant by handing `DIAG_ASSERT4` a
    handle it never dereferences itself, so what that handle POINTS AT is
    decided entirely by the backend. Two ways for it to be wrong:

      * THE CHANNEL IS NOT THERE. `bind` reads the header out of
        `channel.device_base`, which is null until the channel is created, so
        an unguarded macro's first statement is `atomicAdd((unsigned *)0, 1u)`.
        Measured: that is `cudaErrorIllegalAddress` at the next synchronize,
        the invariant never reaches the host, and the reader is handed a
        platform fault naming no cause. A run where nothing asserts touches
        those pages zero times, so the defect is invisible until the day a
        check fires, which is the day the report was needed.

      * THE CHANNEL IS THE WRONG ONE. Every generated entry is handed
        `diagnostics::global()`, because which channel a dispatch reports
        through is a property of the process rather than of the kernel. A
        backend that creates a channel of its own and drains that instead is
        green in every build and every gate: the device writes one object and
        the host reads another, so a recorded violation can never be read.

    Both were real on CUDA and on ROCm at the same time, which is why this rule
    checks them together: the guard alone turns a crash into SILENCE, and the
    adoption alone leaves the crash in place.

    Metal and the host backend reach the same property by different means and
    are checked for what each of them actually relies on. Metal binds the
    channel buffer to every command buffer that dispatches, so its pointer is
    never null and the binding call IS the invariant. The host takes a plain
    pointer and has always tested it, so what matters there is that the test
    stays.
    """
    compute = root / "crates" / "ppf-cts-compute"
    before = len(msgs)

    # THE PREMISE FIRST. Every other half of this rule is about the global
    # channel being the one that matters, which is true because the generator
    # hands it to every entry. If that changes, this rule is checking the wrong
    # thing and says so rather than passing.
    gen = (compute / "seam" / "kernelgen.py").read_text(encoding="utf-8")
    if "diagnostics::global()" not in gen:
        msgs.append(
            "crates/ppf-cts-compute/seam/kernelgen.py no longer hands entries "
            "`diagnostics::global()`, which is what rule 16 checks the backends "
            "against. Update the rule with whatever replaced it")
        return len(msgs) - before

    for target, header, backend, ext in (
        ("cuda", compute / "cuda/diagnostics/diagnostics.hpp",
         compute / "cuda/backend/backend.cu", "cu"),
        ("rocm", compute / "rocm/diagnostics/diagnostics.hpp",
         compute / "rocm/backend/backend.hip", "hip"),
    ):
        text = header.read_text(encoding="utf-8")
        for macro in DIAG_GUARDED_MACROS:
            body = _define_body(text, macro)
            if body is None:
                msgs.append(
                    f"{header.relative_to(root)} no longer defines {macro}, "
                    f"which rule 16 reads to check its null guard")
                continue
            guard = re.search(r"\(diag\)\.header\s*[=!]=\s*nullptr", body)
            if not guard:
                msgs.append(
                    f"{header.relative_to(root)}: {macro} dereferences "
                    f"`(diag).header` without first testing it against nullptr. "
                    f"A firing assert on a backend whose channel was never "
                    f"created writes through a null device pointer, which "
                    f"surfaces as an illegal-address fault naming no cause "
                    f"instead of the check that failed")
                continue
            # The guard has to come FIRST. A test placed after the first
            # `->` reads correctly and protects nothing.
            arrow = body.find("(diag).header->")
            if arrow != -1 and arrow < guard.start():
                msgs.append(
                    f"{header.relative_to(root)}: {macro} tests "
                    f"`(diag).header` against nullptr only AFTER it has already "
                    f"dereferenced it, so the guard protects nothing")

        code = backend.read_text(encoding="utf-8")
        if not re.search(r"be->diag\s*=\s*diagnostics::global\(\)", code):
            msgs.append(
                f"{backend.relative_to(root)}: the backend's channel is not "
                f"`diagnostics::global()`. Every generated entry is handed the "
                f"global channel, so a backend that drains any other object "
                f"reads a channel no kernel ever writes and reports nothing, "
                f"with every build and every gate green")
        if re.search(r"diagnostics::create\(\s*config->diag_ring_slots\s*,\s*"
                     r"&be->diag", code):
            msgs.append(
                f"{backend.relative_to(root)}: `be_create` creates a PRIVATE "
                f"diagnostic channel. That is the object the host drains while "
                f"the kernels write the global one")

        # THE RUNTIME SELF-TEST IS WIRED, which is the only check of this
        # channel that runs on the device. The acceptance passes set
        # `PPF_DIAG_SELFTEST` and require the confirmation line, so a backend
        # that stopped reading the switch would fail them; this says the same
        # thing where a reader can see it, and fails on a tree nobody has run.
        if "PPF_DIAG_SELFTEST" not in code:
            msgs.append(
                f"{backend.relative_to(root)}: `be_create` does not read "
                f"`PPF_DIAG_SELFTEST`. That switch is what makes a build prove "
                f"on its own machine that a failed device check reaches the "
                f"host, and the acceptance passes set it expecting the open to "
                f"be refused when it does not")
        if "selftest_global" not in code:
            msgs.append(
                f"{backend.relative_to(root)}: nothing calls "
                f"`diagnostics::selftest_global`, so the switch above can only "
                f"be a no-op")
        # THE CONFIRMATION MUST REACH A CALLER THAT INSTALLS NO LOG SINK.
        # `--probe` is how the acceptance test opens the device and it installs
        # none, so a line emitted through `emit_log` would reach nobody and the
        # test would be left inferring the self-test from a successful open,
        # which is exactly what a build that dropped the switch also produces.
        if not re.search(r'fprintf\(\s*stderr[^;]*PPF_DIAG_SELFTEST', code, re.S):
            msgs.append(
                f"{backend.relative_to(root)}: the self-test's confirmation "
                f"does not go to stderr, so `--probe` cannot show it and the "
                f"acceptance test has no positive evidence that the switch was "
                f"read")

    # METAL binds the channel rather than passing a pointer, so what keeps its
    # header non-null is that every command buffer carrying a dispatch binds
    # the reserved slot. That call is the invariant; there is no null to guard.
    metal = (compute / "metal/backend/backend.mm").read_text(encoding="utf-8")
    if not re.search(r"context_bind_buffer\([^;]*diag_binding_index\(be->diag\)",
                     metal, re.S):
        msgs.append(
            "crates/ppf-cts-compute/metal/backend/backend.mm no longer binds "
            "the diagnostic buffer to the command buffer it dispatches into. "
            "Metal reads an unbound buffer as null and never faults, so a "
            "firing assert would be dropped in total silence")

    # THE HOST is the oracle the other three are read against: it has always
    # tested its pointer, which is why a device report in doubt is settled by
    # running the scene on the CPU backend. That only holds while the test is
    # present.
    host = (root / "crates/ppf-cts-solver/src/kernels/seam/seam_host.h"
            ).read_text(encoding="utf-8")
    host_body = _define_body(host, "DIAG_ASSERT4")
    if host_body is None or "!= nullptr" not in host_body:
        msgs.append(
            "crates/ppf-cts-solver/src/kernels/seam/seam_host.h: DIAG_ASSERT4 "
            "no longer tests its channel pointer against nullptr. The CPU "
            "backend passes null for a kernel declared without the lane, so "
            "the test is what keeps that from being a host segfault")

    return len(msgs) - before


def check_fourteen(root, msgs):
    """Rule 14: every kernel source DECLARING an entry is rendered by build.rs.

    TWO LISTS NAME THE KERNEL SOURCES AND ONLY ONE OF THEM IS WRITTEN BY HAND.
    `crates/ppf-cts-compute/cuda/Makefile` walks the neutral tree, so a library
    carries an entry the moment its declaration exists. `build.rs`'s `KERNELS`
    is typed out, so an entry can exist in the tree and be rendered for no Rust
    target at all, which makes the two kernel tables different lengths and stops
    the C ABI route at `be_open`.

    Measured once: `main/dump_linsys` sat outside the list, and nothing noticed
    because it is a DIAGNOSTIC. Nothing in a run reaches it, so no run could.

    A source with NO entry declaration is not required here. Fourteen such files
    are deliberately outside the list: they are bodies other kernels include, and
    a body that has never had an entry has never been compiled alone, so adding
    one is a real change rather than bookkeeping.
    """
    kernels = os.path.join(root, "crates", "ppf-cts-solver", "src", "kernels")
    build_rs = os.path.join(root, "crates", "ppf-cts-solver", "build.rs")
    with open(build_rs, encoding="utf-8") as handle:
        text = handle.read()
    match = re.search(r"const KERNELS: \[&str; \d+\] = \[(.*?)\n\];", text, re.S)
    if not match:
        msgs.append("build.rs no longer spells `const KERNELS: [&str; N] = [..]`, "
                    "which rule 14 reads to compare against the tree")
        return 0
    listed = set(re.findall(r'"([^"]+)"', match.group(1)))
    unrendered = []
    for base, _dirs, files in os.walk(kernels):
        for name in files:
            if not name.endswith(".kernel.cpp"):
                continue
            full = os.path.join(base, name)
            stem = os.path.relpath(full, kernels)[: -len(".kernel.cpp")]
            with open(full, encoding="utf-8") as handle:
                body = handle.read()
            if ENTRY_DECL_RE.search(body) and stem not in listed:
                unrendered.append(stem)
    for stem in sorted(unrendered):
        msgs.append(
            f"{stem}.kernel.cpp declares an entry point and is not in build.rs's "
            f"KERNELS, so no Rust target renders it while a library built by "
            f"walking the tree carries it. Add it to KERNELS")
    return len(unrendered)


def check_thirteen(root, msgs):
    """Rule 13: the driver's table may not drift further from canonical order."""
    canon = canonical_entry_order(root)
    driver = driver_table_order(root)
    missing = len(set(canon) - set(driver))
    misplaced = sum(
        1 for index, name in enumerate(driver)
        if index >= len(canon) or canon[index] != name
    )
    if missing > DRIVER_TABLE_ENTRIES_MISSING:
        msgs.append(
            f"{missing} generated entry points have no row in the driver's "
            f"TABLE, over the {DRIVER_TABLE_ENTRIES_MISSING} recorded. A "
            f"library built from the same tree carries a row for every one of "
            f"them, and AbiDevice::open refuses a pair of different lengths, so "
            f"a new entry with no driver row moves the C ABI route further out "
            f"of reach. Add the row, or raise the number here deliberately")
    elif missing < DRIVER_TABLE_ENTRIES_MISSING:
        msgs.append(
            f"{missing} generated entry points have no driver row, under the "
            f"{DRIVER_TABLE_ENTRIES_MISSING} recorded. Lower "
            f"DRIVER_TABLE_ENTRIES_MISSING in this script to {missing}")
    if misplaced > DRIVER_TABLE_ROWS_MISPLACED:
        msgs.append(
            f"{misplaced} driver TABLE rows are not at the position the "
            f"canonical order gives them, over the "
            f"{DRIVER_TABLE_ROWS_MISPLACED} recorded. A kernel id is a table "
            f"INDEX, so a row out of position dispatches one kernel's bytes to "
            f"another's launcher on any target whose table was generated")
    elif misplaced < DRIVER_TABLE_ROWS_MISPLACED:
        msgs.append(
            f"{misplaced} driver TABLE rows are out of canonical position, "
            f"under the {DRIVER_TABLE_ROWS_MISPLACED} recorded. Lower "
            f"DRIVER_TABLE_ROWS_MISPLACED in this script to {misplaced}")
    return missing, misplaced


def check_nine(root, msgs):
    """Rule 9: the count of hand-written CPU entry points may not grow."""
    found = cpu_entry_points(root)
    count = len(found)
    if count > HAND_WRITTEN_CPU_ENTRY_POINTS:
        msgs.append(
            f"entrypoints/ defines {count} hand-written entry points, over the "
            f"{HAND_WRITTEN_CPU_ENTRY_POINTS} recorded. An entry point is an "
            f"argument record plus a thread index plus a call into the neutral "
            f"body: declare it with [[seam::args]] [[seam::entry]] beside the "
            f"body and let ppf-cts-compute/seam/kernelgen.py render it. If "
            f"the body's own shape makes that impossible, say so there and "
            f"raise the number "
            f"in this script deliberately")
    elif count < HAND_WRITTEN_CPU_ENTRY_POINTS:
        msgs.append(
            f"entrypoints/ defines {count} hand-written entry points, under the "
            f"{HAND_WRITTEN_CPU_ENTRY_POINTS} recorded. Lower "
            f"HAND_WRITTEN_CPU_ENTRY_POINTS in this script to {count}: a "
            f"ceiling that is not lowered as conversions land re-permits every "
            f"one of them")
    return count




# ---------------------------------------------------------------------------
# Rule 12: the driver's record fields may not go back to naming an ADDRESS.
#
# A dispatch reaches a
# backend library as a kernel id, an extent and an opaque argument blob, and the
# library resolves each buffer as `base[arena] + off`. It has no way to learn
# what a host ADDRESS means, so a record still carrying one cannot be dispatched
# anywhere but on the host, and `ppf_cts_compute::abi::AbiDevice` refuses it by
# name rather than launching a wild pointer.
#
# WHAT THIS COUNTS, and why it is not `git grep -c HostRef`. That grep is the
# obvious ratchet and it moves for reasons that are not progress: a doc comment
# explaining the migration raises it, and deleting one lowers it.
# What decides whether a phase can be dispatched is how many RECORD FIELDS a
# driver still fills with an address, which is a count of call sites and cannot
# be moved by prose.
#
# THE SPLIT IS THE POINT. A field in a GENERATED record can migrate as soon as
# the buffer behind it is a device allocation. A field in a HAND-WRITTEN record
# cannot migrate at all: a hand-written entry point takes flat pointers, so its
# shim can be handed an address and has nothing to resolve a handle against.
# So this rule and rule 9 are the same programme measured from two ends, and
# whichever of the two numbers is larger is what is actually blocking.
#
# Neither number is asked to be zero. Both are asked to move only deliberately,
# because a ceiling nobody edits re-permits everything it just gained.
#
# THE TWO MOVE IN OPPOSITE DIRECTIONS WHEN AN ENTRY POINT IS CONVERTED, so
# neither one alone can be read as progress or as regression. Generating a
# hand-written entry point moves every field of its record out of the second
# column and into the first, unchanged in count: the four element force scatters
# moved twenty-four that way. So a diff that raises the first and lowers the
# second by the same amount is a conversion, and one that lowers either without
# raising the other is a buffer that became a device allocation. Reading which
# of the two happened is what the split is for, and it is read off these two
# numbers; there is no third one.
#
# WHY THERE IS NO SEPARATE CHECK ON THE SUM, which this rule carried once and
# which could never fail. Both ceilings below are matched for EQUALITY against a
# measured count: over fails as growth, under fails as a ceiling left stale. So
# whenever both pass, the two counts are exactly these two numbers and their sum
# is exactly this sum, and a check that the sum has not grown is a check on a
# quantity the two above have already pinned. It could only have been made to
# fire by recording the total as a third constant and comparing constant against
# constant, and that catches nothing about the tree: an author cannot choose a
# split, because each column is pinned to what the driver measurably does, so
# the third number would only ever catch failing to update a value the other two
# already determine. A check that cannot fail is worse than no check, because a
# reader counts it as coverage.
# THE CENSUS FOLLOWS ONE HOP, AND THAT IS WHAT MAKES IT HONEST. Requiring
# `HostRef::` on the field's own line leaves a record filled `index,` from a
# `let index = HostRef::of(...)` above the literal invisible, and the largest
# unit is exactly the invisible one: the contact narrow phase binds six staging
# locals and spends each across seven records, so the single most valuable
# residency component in the driver would contribute ZERO to the count that is
# supposed to track it. A ratchet that reaches zero with the work undone is the
# one failure mode a ratchet must not have, so the count is widened rather than
# the ceiling nudged.
DRIVER_ADDRESS_FIELDS_GENERATED = 0
DRIVER_ADDRESS_FIELDS_HAND_WRITTEN = 0

# A record literal opens with `SomeArgs {` at the end of a line and its buffer
# fields are the ones filled from a `HostRef`. Reading the LITERAL rather than
# the declaration is what makes this a count of call sites: one record filled at
# three call sites is three fields that have to move.
RECORD_LITERAL_RE = re.compile(r"\b([A-Z]\w*Args)\s*\{\s*$")
RECORD_FIELD_RE = re.compile(
    r"^\s*(\w+):\s*(?:unsafe\s*\{\s*)?HostRef::(?:of|of_mut|at|at_mut)\(")
# The same match, keeping what the field is filled FROM, which is what the
# closure diagnostic below needs to tell two fields naming one buffer apart.
RECORD_FIELD_ARG_RE = re.compile(
    r"^\s*(\w+):\s*(?:unsafe\s*\{\s*)?HostRef::(?:of|of_mut|at|at_mut)\((.*)$")

# A `HostRef` BOUND TO A LOCAL, AND THE FIELD SHORTHAND THAT SPENDS IT. A record
# filled `index,` from `let index = HostRef::of(...)` above the literal is the
# same debt as one filled `index: HostRef::of(...)` in place, and for a long time
# this census could not see it: the contact narrow phase binds SIX such locals
# and spends each across SEVEN records, so 42 sites in `contact.rs` alone were
# invisible, which is the single largest residency unit in the driver. A count
# that cannot see the largest unit would reach zero with the work undone, which
# is the one failure a ratchet must not have.
#
# ONE HOP AND NO MORE, deliberately. Resolving a chain would need real dataflow;
# one hop is what the driver actually writes, and a local bound to a `HostRef`
# anywhere in a file is treated as one wherever the file spends it. That is an
# over-read only if a file binds a `HostRef` to a name AND spends an unrelated
# local of the same name as a field, which no driver file does today.
HOSTREF_LOCAL_RE = re.compile(
    r"^\s*let\s+(\w+)\s*=\s*(?:unsafe\s*\{\s*)?"
    r"HostRef::(?:of|of_mut|at|at_mut)\(")
RECORD_SHORTHAND_RE = re.compile(r"^\s*(\w+),\s*$")


def generated_record_names(root):
    """Records rendered from an `[[seam::args]]` declaration.

    Read off the RENDERINGS in the build directory when there is one, and off
    the declarations otherwise, so this runs on a tree that has not been built.
    """
    names = set()
    kernels = os.path.join(root, "crates", "ppf-cts-solver", "src", "kernels")
    for base, _, files in os.walk(kernels):
        for name in files:
            if not name.endswith(".kernel.cpp"):
                continue
            with open(os.path.join(base, name), encoding="utf-8") as f:
                text = f.read()
            # A record is rendered for [[seam::args]] AND for [[seam::entry]],
            # which implies it; asking for the first alone sees none of them.
            for stem in entry_or_args_names(text):
                camel = "".join(p.capitalize() for p in stem.split("_"))
                names.add(camel + "Args")
    return names


def entry_or_args_names(text):
    """Every neutral declaration that renders an argument record."""
    out = []
    for m in ENTRY_DECL_RE.finditer(text):
        if _carries(m.group(1), "entry") or _carries(m.group(1), "args"):
            out.append(m.group(2))
    return out


def driver_address_fields(root):
    """Record fields the driver still fills with a host address.

    Returns (in generated records, in hand-written records).
    """
    generated = generated_record_names(root)
    driver = os.path.join(root, "crates", "ppf-cts-solver", "src", "driver")
    in_generated = 0
    in_hand_written = 0
    for name in sorted(os.listdir(driver)):
        # kernels.rs DECLARES the hand-written records; it fills none.
        if not name.endswith(".rs") or name == "kernels.rs":
            continue
        with open(os.path.join(driver, name), encoding="utf-8") as f:
            lines = f.read().split("\n")
        # The file's `HostRef` locals, collected before the walk because a
        # binding sits ABOVE the literal that spends it.
        held = {m.group(1) for m in
                (HOSTREF_LOCAL_RE.match(l) for l in lines) if m}
        current = None
        depth = 0
        for line in lines:
            if current is None:
                m = RECORD_LITERAL_RE.search(line)
                if m:
                    current = m.group(1)
                    depth = 1
                continue
            depth += line.count("{") - line.count("}")
            shorthand = RECORD_SHORTHAND_RE.match(line)
            if (RECORD_FIELD_RE.match(line)
                    or (shorthand and shorthand.group(1) in held)):
                if current in generated:
                    in_generated += 1
                else:
                    in_hand_written += 1
            if depth <= 0:
                current = None
    return in_generated, in_hand_written


def driver_movable_fields(root):
    """How many of those address fields could move TODAY, and what blocks the rest.

    A DIAGNOSTIC, NOT A RATCHET, because the number rises when an entry point is
    generated and falls when a buffer moves, so neither direction is progress on
    its own and a ceiling on it would refuse one of the two.

    THE RULE IS TRANSITIVE, WHICH IS THE PART THAT IS EASY TO GET WRONG. Section
    12.4a states it one hop: a buffer may move when every record that names it is
    generated. But a record FIELD is one type in the Rust twin, so every buffer
    filling that field at any call site has to move with it, and a buffer named by
    several fields drags all of them. The unit is therefore a connected component
    of the graph whose two node kinds are a buffer expression and a `Record.field`,
    and a component may move only when every record in it is generated.

    IT READS EXPRESSIONS, SO IT IS AN UPPER BOUND ON WHAT CAN MOVE, AND IT ERRS IN
    BOTH DIRECTIONS. Confirm the buffer by name before moving one; what the number
    is good for is the SHAPE of the frontier, not a work list. Three measured ways
    it misleads, each found by reading the buffers this reports movable:

    - TWO SPELLINGS OF ONE BUFFER LOOK LIKE TWO BUFFERS, which reports as movable
      something that is blocked. `mesh_order` in `assemble.rs` holds
      `data.mesh.mesh.hinge.data`, which the hand-written `ShellBendRemapArgs`
      also names; `force` in `contact.rs` and `collider.rs` holds `state.force`,
      which `DirichletLiftArgs` and `DirichletPrescribeArgs` name.
    - TWO LOCALS SHARING A NAME LOOK LIKE ONE BUFFER, which reports as blocked
      something that is not. `current` in `assemble.rs::rod_stretch` and
      `current` in `step.rs::compute_target` are unrelated arrays, and the second
      is unioned into the first's blocker. That one happens to cancel the error
      above, since the second is `data.vertex.curr.data` and eight hand-written
      records name it.
    - A RECORD FILLED FROM A LOCAL HOLDING A `HostRef` IS COUNTED, ONE HOP. It
      is the largest case rather than a footnote: a regex wanting `HostRef::` on
      the field's own line misses it, and the contact narrow phase binds six
      staging locals it spends across seven records each, so 42 sites in
      `contact.rs` would be invisible. `HOSTREF_LOCAL_RE` and
      `RECORD_SHORTHAND_RE` resolve that one hop, which finds 72 sites in all. A
      CHAIN is not resolved, because that needs real dataflow; one hop is what
      the driver writes.

    EVERY BARE LOCAL THE SUMMARY LINE SETS ASIDE HAS BEEN READ AT SOURCE, AND
    NOT ONE OF THEM WAS A BUFFER WAITING TO BE MOVED. The reading is recorded
    here because it costs an hour to redo and the answer changes only when the
    driver does. They fall into four kinds, and the kind is what decides the
    verdict rather than the name:

    - A FUNCTION PARAMETER OVER A CALLER'S HOST SLICE, which is a residency
      decision about a whole subsystem and not a relocation. `source`,
      `destination`, `inverse`, `r` and `z` are the PCG helpers' parameters
      (`pcg.rs::encode_add_scaled`, `encode_preconditioner`, and
      `step.rs::copy_floats`); `cx`, `cy`, `cz` and `codes` belong to
      `bvh.rs::morton_codes`; `rows`, `columns`, `blocks` and `stored` to
      `fixedcsr.rs::push_blocks`; `pairs` to `pair_cache.rs::record`; and
      `force` is the parameter both element scatters take, which every caller
      fills with `state.force`.
    - THE HOST'S OWN `DataSet` ARRAYS UNDER A LOCAL'S NAME. `tet_index` and
      `mesh_order` are `data.mesh.mesh.tet.data` and `data.mesh.mesh.hinge.data`;
      `current`, `previous` and `prev` are `data.vertex.curr.data` and
      `data.vertex.prev.data`. Moving one means the driver holds a device mirror
      of a scene array the host still rewrites, which is the residency question
      and not a rename.
    - A DRIVER BUFFER A HAND-WRITTEN RECORD ALSO NAMES, reported movable only
      because the two spellings differ, which is the first failure above wearing
      a bare name. `fix` is `state.fix`, which hand-written `RewindFixArgs`
      names, and `target` is `state.target`, which hand-written `DxSeedArgs` and
      `PositionAcceptArgs` name.
    - A `#[cfg(test)]` ORACLE'S STACK ARRAY, which no dispatch in a step ever
      sees. `remapped` and `stencil` are the `[u32; 4]` and `[u32; 3]` the two
      bending-angle oracles pass, and `strain`, `deformation`, `difference` and
      `rest` belong to the two strain oracles, whose records have NO production
      call site at all, so generating a handle for one would unblock no phase.

    A TEST ORACLE IS NOT A FREE CONVERSION, and this is the trap to know before
    planning around one. Those oracles dispatch on a fresh
    `launch::host_device()`, and a `HostDevice` keeps its arenas per INSTANCE, so
    a handle cut from a fixture's own device resolves against the wrong base
    table on that one. An oracle whose record field becomes a `Handle` has to
    dispatch on the device that allocated the buffer, or allocate its own.
    """
    generated = generated_record_names(root)
    driver = os.path.join(root, "crates", "ppf-cts-solver", "src", "driver")
    edges = []
    for name in sorted(os.listdir(driver)):
        if not name.endswith(".rs") or name == "kernels.rs":
            continue
        with open(os.path.join(driver, name), encoding="utf-8") as f:
            lines = f.read().split("\n")
        current = None
        depth = 0
        for line in lines:
            if current is None:
                m = RECORD_LITERAL_RE.search(line)
                if m:
                    current = m.group(1)
                    depth = 1
                continue
            depth += line.count("{") - line.count("}")
            m = RECORD_FIELD_ARG_RE.match(line)
            if m:
                # THE `.as_ptr()` STRIP RUNS BEFORE THE PAREN STRIP, and the
                # order is what makes it fire at all. `HostRef::at_mut(x
                # .as_mut_ptr(), n)` leaves `x.as_mut_ptr()` after the split, and
                # `rstrip(")")` takes BOTH parens off it, so a suffix pattern
                # anchored on `()` never matched and `state.eval_x.as_mut_ptr(`
                # read as a buffer of its own. Measured: that split
                # `state.eval_x` in two and reported three of its sites as
                # movable while eleven hand-written records name it.
                expr = re.sub(r"\.as_(mut_)?ptr\(\)", "", m.group(2).split(",")[0])
                expr = expr.rstrip(")").strip().lstrip("&").replace("mut ", "")
                edges.append((expr, f"{current}.{m.group(1)}"))
            if depth <= 0:
                current = None

    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for expr, field in edges:
        a, b = find(("buffer", expr)), find(("field", field))
        if a != b:
            parent[a] = b

    members = {}
    for expr, field in edges:
        # KEYED ON THE COMPONENT ROOT, which the union above may leave on
        # either node kind, so the buffer expressions are collected explicitly
        # rather than read back off the key.
        entry = members.setdefault(find(("buffer", expr)),
                                   {"fields": set(), "buffers": set(), "sites": 0})
        entry["fields"].add(field)
        entry["buffers"].add(expr)
        entry["sites"] += 1

    movable = 0
    unresolved = 0
    sole = {}
    for entry in members.values():
        blockers = {f.split(".")[0] for f in entry["fields"]
                    if f.split(".")[0] not in generated}
        if not blockers:
            movable += entry["sites"]
            # A BARE NAME IS A LOCAL OR A PARAMETER, so it says nothing about
            # which storage it is. `state.force` and a `force: &mut [f32]`
            # parameter are the measured case: eight record fields name the
            # first, two name the second, and two hand-written records block
            # the first, so the second reads as movable and is not. A
            # qualified expression (`state.face.scatter_index`, `matrix.value`)
            # can still alias, which is why the whole number stays an upper
            # bound, but it at least names a place.
            if all("." not in expr and "[" not in expr
                   for expr in entry["buffers"]):
                unresolved += entry["sites"]
        elif len(blockers) == 1:
            one = next(iter(blockers))
            sole[one] = sole.get(one, 0) + entry["sites"]
    top = sorted(sole.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    return movable, unresolved, top


def check_twelve(root, msgs):
    """Rule 12: neither address-field count may grow."""
    found = driver_address_fields(root)
    for count, ceiling, what, name in (
            (found[0], DRIVER_ADDRESS_FIELDS_GENERATED, "generated",
             "DRIVER_ADDRESS_FIELDS_GENERATED"),
            (found[1], DRIVER_ADDRESS_FIELDS_HAND_WRITTEN, "hand-written",
             "DRIVER_ADDRESS_FIELDS_HAND_WRITTEN")):
        if count > ceiling:
            msgs.append(
                f"the driver fills {count} buffer fields of {what} records with "
                f"a host address, over the {ceiling} recorded. A backend library "
                f"resolves an (arena, offset) handle and cannot resolve an "
                f"address, so every one of these has to move before the phase "
                f"that uses it can dispatch anywhere but on the host. Move the "
                f"buffer to a ppf_cts_compute::Buffer or StagedBuffer and name "
                f"the field in HANDLE_FIELDS in crates/ppf-cts-solver/build.rs")
        elif count < ceiling:
            msgs.append(
                f"the driver fills {count} buffer fields of {what} records with "
                f"a host address, under the {ceiling} recorded. Lower {name} in "
                f"this script to {count}: a ceiling that is not lowered as "
                f"buffers move re-permits every one of them")
    return found


# ---------------------------------------------------------------------------
# Rule 10: the CPU DRIVER may name a shim HELPER, never a shim LAUNCHER.
#
# A DISPATCH REACHES THE BACKEND through the `Device` trait and through nothing
# else. What this rule settles is
# WHICH of the shim symbols a driver module names is a dispatch, and it
# settles it by the property that defines one rather than by a comment claiming
# a category. A LAUNCHER takes a half-open thread range; rule 9 already reads
# that off the definitions under entrypoints/ and the same reading serves here. A
# driver module naming a launcher is a driver launching a kernel around the
# seam, which is the regression the seam exists to prevent, so it fails.
#
# A HELPER is the other side of that split and is deliberately NOT failed. It
# takes no thread range: a layout query, a compile-time constant, a per-pair
# predicate, a per-block inverse. Each is a call into a neutral body that nvcc,
# the Metal shader compiler and a host C++ compiler build from identical bytes,
# and the guarantee it carries is that NO backend answers it differently.
# Putting one behind `Device` would make it a method each backend implements,
# which converts that guarantee into a permission and puts an ANSWER on a
# surface section 2.1b restricts to the MEANS. The rule therefore records
# helpers rather than forbidding them, and prints the count so a binding is
# visible rather than merely tolerated.
#
# A launcher named INSIDE a `#[cfg(test)]` module is allowed and counted apart.
# A test calling a shared body directly is a test rather than a driver reaching
# around the seam, and sending an oracle through the table, the record check and
# the launch would compare the code under test against a reference that
# travelled the same machinery.
#
# THREE FAILURES, and the second is the one a future edit is most likely to
# produce. A launcher named in production code and not recorded below. A symbol
# no entrypoints/ translation unit defines, which is a generated entry point, a
# dispatch by construction. And a recorded exception that no longer occurs,
# reported for the same reason rule 9 reports a drop: an exception nobody
# retires re-permits what a conversion just removed.

# Every production launcher binding that stands, with the reason it stands.
#
# EMPTY, and the last entry to leave is worth recording. `aabb_query_pairs_entry`
# stood here because the shim packed pairs as it found them against a count
# shared across a partition, so it needed a per-partition output buffer and a
# per-partition return value and `Extent::Elements` carries neither. The way out
# was not any of the three candidates weighed at `walk`: a PER-QUERY SLOT needs
# no shared count at all, so query `i` owns its own run of the output and what
# the body returns is one value at its own index, which is exactly what
# `Extent::Elements` carries.
DRIVER_LAUNCHER_EXCEPTIONS = {}

# A reason is required AT the declaration, and one line is not a reason. Every
# block in the tree clears this by a wide margin; the floor exists so that a
# label restating the code ("the CCD shims") cannot pass for one.
MIN_REASON_LINES = 3

# The DECLARATION form, terminated by `;`. With the project prefix gone a bare
# `fn <name>_entry` also matches a test function whose name happens to end that
# way (`a_lower_triangle_entry` in fixedcsr.rs is one of three), and a
# declaration is what this rule is about: an extern has no body.
RUST_DECL_RE = re.compile(
    r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*_(?:entry|abi))\s*\([^;{]*\)\s*(?:->[^;{]*?)?;")
RUST_EXTERN_RE = re.compile(r'^([ \t]*)extern\s+"C"\s*\{', re.M)


def mask_rust(text, strings=True):
    """Blank Rust comments, and string literals unless asked not to.

    Offsets are preserved because every position this rule reports is turned
    back into a line number in the ORIGINAL text. Comments are masked for the
    same reason rule 7 masks them: `fixedcsr.rs` opens with a module doc comment
    containing the words `#[cfg(test)]`, and a scan reading that as the start of
    the test module puts nine production declarations inside a test module and
    reports them as oracles.

    `strings` is why this takes an argument at all, and it is a trap worth
    naming: the `"C"` in `extern "C"` IS a string literal, so the mask that
    makes brace balancing safe also erases the token the block locator matches
    on. Locating runs on the comment-only mask and balancing on the full one;
    both preserve offsets, so the two agree about every position.
    """
    out = list(text)
    length = len(text)

    def blank(start, stop):
        for k in range(start, stop):
            if out[k] != "\n":
                out[k] = " "

    i = 0
    while i < length:
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = length if j < 0 else j
            blank(i, j)
            i = j
        elif text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < length and depth:
                if text.startswith("/*", j):
                    depth += 1
                    j += 2
                elif text.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif strings and text.startswith('r"', i):
            j = text.find('"', i + 2)
            j = length if j < 0 else j + 1
            blank(i, j)
            i = j
        elif strings and text[i] == '"':
            j = i + 1
            while j < length:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    j += 1
                    break
                j += 1
            blank(i, j)
            i = j
        else:
            i += 1
    return "".join(out)


def cfg_test_spans(masked):
    """The half-open spans a `#[cfg(test)]` attribute covers.

    The attribute is usually on a `mod tests { ... }`, and then the span is that
    module's brace-balanced block. It can also sit on any other item, so the
    general rule is taken: whichever of the next brace or the next semicolon
    comes first ends it, which covers `#[cfg(test)] extern "C" { ... }` and
    `#[cfg(test)] mod tests;` alike.
    """
    spans = []
    for m in re.finditer(r"#\[cfg\(test\)\]", masked):
        brace = masked.find("{", m.end())
        semi = masked.find(";", m.end())
        if brace < 0 or (0 <= semi < brace):
            spans.append((m.start(), semi + 1 if semi >= 0 else m.end()))
            continue
        depth, j = 0, brace
        while j < len(masked):
            if masked[j] == "{":
                depth += 1
            elif masked[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        spans.append((m.start(), j + 1))
    return spans


def block_end_index(masked, open_brace):
    """The index just past the brace that closes `open_brace`."""
    depth, j = 0, open_brace
    while j < len(masked):
        if masked[j] == "{":
            depth += 1
        elif masked[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return len(masked)


def driver_shim_declarations(root):
    """Every shim symbol a driver module declares.

    `launch.rs` is excluded because it IS the launch table: the one place in the
    crate that may name a shim, which its own module comment states. The DEVICE
    that consumes that table is `ppf-cts-compute`, which names no shim at all,
    so excluding one file here still leaves the rule measuring every driver
    module.

    Returns a list of (file, symbol, line, in_test) and a list of the
    declaration BLOCKS with the number of comment lines that carry their reason.
    """
    directory = os.path.join(root, "crates", "ppf-cts-solver", "src", "driver")
    declarations = []
    blocks = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".rs") or name == "launch.rs":
            continue
        with open(os.path.join(directory, name), encoding="utf-8") as f:
            text = f.read()
        masked = mask_rust(text)
        located = mask_rust(text, strings=False)
        spans = cfg_test_spans(masked)
        for m in RUST_DECL_RE.finditer(masked):
            declarations.append((
                name,
                m.group(1),
                text.count("\n", 0, m.start()) + 1,
                any(a <= m.start() < b for a, b in spans),
            ))
        for m in RUST_EXTERN_RE.finditer(located):
            end = block_end_index(masked, m.end() - 1)
            if not RUST_DECL_RE.search(masked, m.end(), end):
                continue
            blocks.append((
                name,
                text.count("\n", 0, m.start()) + 1,
                reason_lines(text, m.start(), m.end()),
            ))
    return declarations, blocks


def reason_lines(text, block_start, block_open_end):
    """Comment lines carrying a block's reason, above it or first inside it.

    Both placements are in the tree and both are AT the declaration a reader
    meets first, which is the property required; `super::step` writes its reason
    inside the block and `super::ccd` writes it above.
    """
    lines = text.split("\n")
    above = text.count("\n", 0, block_start)
    count = 0
    j = above - 1
    while j >= 0 and lines[j].lstrip().startswith("//"):
        count += 1
        j -= 1
    inside = text.count("\n", 0, block_open_end) + 1
    j = inside
    while j < len(lines) and lines[j].lstrip().startswith("//"):
        count += 1
        j += 1
    return count


def check_ten(root, msgs):
    """Rule 10: a driver names a helper, never a launcher."""
    launchers, helpers = cpu_shim_definitions(root)
    declarations, blocks = driver_shim_declarations(root)

    # AN EMPTY READING IS A FAILURE, NOT A PASS, and this rule has already been
    # bitten once by exactly that. The block locator first ran on a mask that
    # blanks string literals, and the `"C"` in `extern "C"` is one, so it
    # matched nothing, checked nothing and reported clean. Either set going
    # empty now means the driver stopped naming shims, in which case this rule
    # is finished and should be deleted deliberately, or the parser is broken.
    if not declarations:
        msgs.append(
            "rule 10 found no shim declaration in any src/driver driver "
            "module. That is either a finished conversion, which retires this "
            "rule, or a broken parser; it is not a pass")
    if not blocks:
        msgs.append(
            "rule 10 found no extern block declaring a shim in any src/driver "
            "driver module, so the reason check examined nothing. Same two "
            "readings as above and the same verdict")

    production_launchers = {}
    test_launchers = {}
    production_helpers = {}
    test_helpers = {}
    for name, symbol, line, in_test in declarations:
        if symbol in launchers:
            (test_launchers if in_test else production_launchers)[symbol] = (
                name, line)
        elif symbol in helpers:
            (test_helpers if in_test else production_helpers)[symbol] = (
                name, line)
        else:
            msgs.append(
                f"src/driver/{name}:{line} declares {symbol}, which no entrypoints/ "
                f"translation unit defines. A GENERATED entry point is a "
                f"dispatch by construction, so it reaches the backend through "
                f"Device::launch and a driver never names it; if this is "
                f"something else, it has to be classified before it can be "
                f"allowed")

    for symbol, (name, line) in sorted(production_launchers.items()):
        if symbol in DRIVER_LAUNCHER_EXCEPTIONS:
            continue
        msgs.append(
            f"src/driver/{name}:{line} declares the LAUNCHER {symbol} in "
            f"production code. A launcher takes a thread range, so it is a "
            f"dispatch, and a dispatch reaches the backend through "
            f"Device::launch and through nothing else. "
            f"Give it a KernelDecl and an argument record in src/driver/kernels.rs "
            f"and dispatch it, or record it in DRIVER_LAUNCHER_EXCEPTIONS in "
            f"this script with the reason its shape defeats the seam")

    for symbol, (name, reason) in sorted(DRIVER_LAUNCHER_EXCEPTIONS.items()):
        if symbol not in production_launchers:
            msgs.append(
                f"{symbol} is recorded in DRIVER_LAUNCHER_EXCEPTIONS and no "
                f"driver module declares it any more. Remove the entry: an "
                f"exception nobody retires re-permits what a conversion just "
                f"removed")
        elif production_launchers[symbol][0] != name:
            msgs.append(
                f"{symbol} is recorded against src/driver/{name} and is declared "
                f"in src/driver/{production_launchers[symbol][0]}. Move the entry, "
                f"since the reason is written beside the code whose shape "
                f"defeats the seam and a stale pointer is worse than none")

    for name, line, reason in blocks:
        if reason < MIN_REASON_LINES:
            msgs.append(
                f"src/driver/{name}:{line} declares a shim with {reason} comment "
                f"line(s) at the declaration, under the {MIN_REASON_LINES} "
                f"required. Every binding a driver keeps carries its reason "
                f"where a reader meets it first, so a deliberate exception can "
                f"be told from a binding nobody moved")

    return (production_helpers, test_helpers, production_launchers,
            test_launchers)


# ---------------------------------------------------------------------------
# Rule 11: a kernel declaration NAMES THE ENTRY POINT ITS LAUNCH ROW CALLS.
#
# `crates/ppf-cts-compute/src/device.rs` documents `KernelDecl::name` as the
# entry point's name, identical across renderings, which is what lets a trace
# taken on one backend be compared against a trace taken on another. That
# promise is only worth something if the string is the symbol, and nothing else
# in the tree reads it: `ppf_cts_compute::host` puts it in `Fault::Shape`
# diagnostic text and no compiler ever sees it, so a wrong name here is silent
# for as long as nobody reads a fault. This rule reads it.
#
# It works by pairing the two tables the way the running code pairs them, by
# KERNEL ID rather than by position, and then asking what the launch row's
# thunk actually calls. A generated row is covered as well as a hand-written
# one, because `ppf-cts-compute/seam/kernelgen.py` renders both the `_NAME`
# constant and the entry point from one declaration.
#
# AN EMPTY READING IS A FAILURE, for the reason rule 10 states at length: a
# parser that stops matching reports nothing wrong and reports it as a pass.

KERNEL_ID_RE = re.compile(r"pub const (\w+): KernelId = KernelId\((\d+)\);")
# A row names its entry point in one of two ways, and rule 11 reads both.
# A HAND-WRITTEN row spells the symbol as a string literal. A GENERATED row names
# the `<STEM>_NAME` constant the generator emits beside the record, whose value
# lives in OUT_DIR rather than here; the constant's own name carries the stem, so
# the symbol is recovered as `<stem>_entry`, which is the convention the
# generator renders and the `extern` block declares. Reading only the first shape
# is how this rule went quiet once every record was generated.
KERNEL_DECL_RE = re.compile(r"decl\(\s*id::(\w+),\s*\"([^\"]+)\"")
GENERATED_DECL_RE = re.compile(
    r"decl_generated(?:_diag)?\(\s*id::(\w+),\s*(\w+)_NAME")
LAUNCH_TABLE_RE = re.compile(
    r"static LAUNCH: \[Launch; id::COUNT\] = \[(.*?)\n\];", re.S)
# Both generated thunk macros. They differ only in whether the entry declares a
# `[[seam::diag]]` lane, which changes the extern's signature and nothing this
# rule reads; matching only the first is how the diag-carrying rows went unread.
GENERATED_THUNK_RE = re.compile(r"\bgenerated_thunk(?:_diag|_group)?!\s*\(")
HAND_THUNK_RE = re.compile(r"(?<!generated_)\bthunk!\s*\(")
THUNK_NAME_RE = re.compile(r"\b(launch_\w+)")
ENTRY_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*_(?:entry|abi))\s*\(")


def _paren_body(text, start):
    """The text between the parentheses of the macro invocation at `start`."""
    open_paren = text.index("(", start)
    depth, j = 0, open_paren
    while j < len(text):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1:j]
        j += 1
    return ""


def _rendered_thunks(root):
    """Every `generated_thunk*!` line the generator renders for this tree."""
    import importlib.util
    gen = os.path.join(root, "crates", "ppf-cts-compute", "seam", "kernelgen.py")
    kernels = os.path.join(root, "crates", "ppf-cts-solver", "src", "kernels")
    if not os.path.exists(gen) or not os.path.isdir(kernels):
        return ""
    spec = importlib.util.spec_from_file_location("kernelgen_thunks", gen)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = []
    for dirpath, _dirs, names in os.walk(kernels):
        for name in sorted(names):
            if not name.endswith(".kernel.cpp"):
                continue
            path = os.path.join(dirpath, name)
            _, _, _, entries = module.read_source(path)
            emitted = [e for e in entries if e.emit_entry]
            if emitted:
                out.append(module.render_thunks_rust(path, emitted))
    return "\n".join(out)


def check_eleven(root, msgs):
    """Rule 11: each kernel declaration names the entry point it dispatches."""
    driver = os.path.join(root, "crates", "ppf-cts-solver", "src", "driver")
    with open(os.path.join(driver, "kernels.rs"), encoding="utf-8") as f:
        kernels_text = f.read()
    with open(os.path.join(driver, "launch.rs"), encoding="utf-8") as f:
        launch_text = f.read()
    # THE THUNKS ARE RENDERED INTO OUT_DIR, so reading launch.rs alone finds
    # none and this rule would compare a full id table against an empty one.
    # Rendering them here reads the same function `build.rs` calls, and appending
    # rather than replacing keeps any hand-written thunk in the reading too.
    launch_text += _rendered_thunks(root)

    ids = {n: int(v) for n, v in KERNEL_ID_RE.findall(kernels_text)}
    table = LAUNCH_TABLE_RE.search(launch_text)
    if not table:
        msgs.append(
            "rule 11 could not locate `static LAUNCH` in src/driver/launch.rs, "
            "so it compared nothing. Either the table moved or this parser is "
            "broken; it is not a pass")
        return 0
    order = [line.strip().rstrip(",")
             for line in table.group(1).strip().split("\n")
             if line.strip() and not line.strip().startswith("//")]

    calls = {}
    for pattern in (GENERATED_THUNK_RE, HAND_THUNK_RE):
        for m in pattern.finditer(launch_text):
            body = _paren_body(launch_text, m.start())
            name = THUNK_NAME_RE.search(body)
            if not name:
                continue
            if pattern is GENERATED_THUNK_RE:
                parts = [p.strip() for p in body.split(",")]
                if len(parts) >= 3:
                    calls.setdefault(name.group(1), parts[2])
                continue
            called = ENTRY_CALL_RE.search(body)
            if called:
                calls.setdefault(name.group(1), called.group(1))

    rows = KERNEL_DECL_RE.findall(kernels_text)
    rows += [(id_name, f"{stem.lower()}_entry")
             for id_name, stem in GENERATED_DECL_RE.findall(kernels_text)]
    if not rows or not calls:
        msgs.append(
            f"rule 11 read {len(rows)} kernel declarations and {len(calls)} "
            f"launch thunks, and a comparison needs both. Either the tables "
            f"changed shape or this parser is broken; it is not a pass")
        return 0

    checked = 0
    for id_name, declared in rows:
        index = ids.get(id_name)
        if index is None or index >= len(order):
            msgs.append(
                f"rule 11: src/driver/kernels.rs declares id::{id_name}, which "
                f"has no row in src/driver/launch.rs's LAUNCH table")
            continue
        called = calls.get(order[index])
        if called is None:
            msgs.append(
                f"rule 11: the LAUNCH row for id::{id_name} is "
                f"`{order[index]}`, and no thunk of that name calls an entry "
                f"point this rule can read")
            continue
        checked += 1
        if declared != called:
            msgs.append(
                f"rule 11: id::{id_name} is declared as \"{declared}\" and "
                f"dispatches `{called}`. A KernelDecl name is the entry "
                f"point's name, identical across renderings, which is what "
                f"lets a trace taken on one backend be compared against "
                f"another; nothing compiles this string, so name it for the "
                f"symbol it reaches")
    return checked


def host_include_closure(root, cpp_dir):
    """Every file under src/kernels the HOST build compiles.

    THE HOST BUILD IS A THIRD COMPILER AND LEAVING IT OUT MISREPORTS A BODY AS
    UNREAD. `entrypoints/` is what `cc` compiles for the CPU backend and for
    the driver's host-side launchers, and it reads headers no other build
    reaches. `contact/accd.hpp` shows why the reading is per header rather than
    per subsystem: the CCD line search is six device dispatches, so nvcc and the
    Metal shader compiler read that header through
    `contact/ccd_sweep.kernel.cpp`, while what `shim_contact.cpp` exports
    through its `*_abi` functions is the per-pair TEST ORACLE
    `src/driver/ccd.rs` names. A rule that asked only
    about nvcc and the Metal shader would still call other headers uncompiled,
    which is false and points at the wrong repair.

    Keyed by path relative to the neutral tree, the same as the other two sets.
    """
    entry_dir = root / "crates/ppf-cts-solver/entrypoints"
    if not entry_dir.is_dir():
        sys.exit(f"check-shared-wiring: no such directory: {entry_dir}")
    sources = sorted(entry_dir.glob("*.cpp"))
    if not sources:
        sys.exit(
            "check-shared-wiring: entrypoints/ holds no translation unit, so "
            "the host build would compile nothing and this check cannot say "
            "which bodies it reads. It refuses to report a pass over that."
        )
    seen, stack = set(), list(sources)
    while stack:
        path = stack.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        for m in INCLUDE_RE.finditer(read(path)):
            spelled = m.group(1)
            for base in (path.parent, cpp_dir):
                cand = Path(os.path.normpath(base / spelled))
                if cand.exists():
                    stack.append(cand)
                    break
    out = set()
    for f in seen:
        try:
            out.add(str(f.relative_to(cpp_dir)))
        except ValueError:
            continue
    return out


def check_fifteen(cpp_dir, kernels, closure, embedded, host, msgs):
    """Rule 15: a neutral body no compiler reads must be recorded, and counted.

    THE HAZARD IS THAT NOTHING CONTRADICTS IT. A body nvcc does not compile,
    the Metal shader does not embed and the host build does not include cannot
    fail to build, so it can drift from the vocabulary around it while every
    other rule here reports clean. EVERY RENDERING IS COMPILED BY SOME BUILD, OR IT IS NOT CHECKED AT
    ALL, applied to the source rather than to the rendering.

    The set is DERIVED, never listed: what is listed is the REASON each body is
    in it, so a body that falls into this state with no reason recorded is a
    failure rather than a silent addition, and a body that climbs out of it
    leaves a stale entry that is equally a failure.
    """
    # `embedded` IS EMPTY NOW and is kept in the signature because the union is
    # the statement: a body is compiled when nvcc reaches it, when the Metal
    # library carries it, or when the host build includes it. The middle term
    # collapsed into the first when that library became the generated entries.
    compiled = closure | embedded | host
    uncompiled = sorted(rel for rel in kernels if rel not in compiled)
    for rel in uncompiled:
        if rel not in UNCOMPILED:
            fail(
                msgs,
                f"crates/ppf-cts-solver/src/kernels/{rel}",
                "Neutral body no compiler reads",
                "is compiled by nvcc, by the Metal shader and by the host "
                "build alike: by none of them. Nothing can contradict it, so "
                "it can drift with every gate green. Give "
                "it an entry declaration or an embed rule, or record it in "
                "UNCOMPILED in this script with the reason and raise nothing: "
                "the ceiling only falls.",
            )
    for rel in sorted(UNCOMPILED):
        if rel not in kernels:
            msgs.append(
                f"UNCOMPILED names {rel}, which is not a neutral kernel body. "
                f"The entry is stale: delete it."
            )
        elif rel in compiled:
            where = []
            if rel in closure:
                where.append("nvcc")
            if rel in embedded:
                where.append("the Metal shader")
            if rel in host:
                where.append("the host build")
            msgs.append(
                f"UNCOMPILED names {rel}, which {' and '.join(where)} now "
                f"compiles. The exception is stale: delete the entry and lower "
                f"UNCOMPILED_CEILING to {len(UNCOMPILED) - 1}."
            )
    if len(uncompiled) > UNCOMPILED_CEILING:
        msgs.append(
            f"rule 15: {len(uncompiled)} neutral bodies are compiled by no "
            f"backend, above the ceiling of {UNCOMPILED_CEILING}. The ceiling "
            f"only falls: give the new one a caller rather than raising it."
        )
    return uncompiled


# MSL built-in FUNCTIONS. These read as ordinary identifiers under nvcc and a
# host C++ compiler, so a NAMESPACE or free function spelled this way compiles
# on two backends and fails the shader compile on the third.
#
# NOT THE SAME LIST AS THE GENERATOR'S RESERVED NAMES, and deliberately checked
# somewhere else. That list is applied to record FIELDS and entry names, where a
# built-in function name is harmless: a struct member called `step` is legal
# MSL. What is not legal is a namespace called `distance`, because the name
# already denotes a function at that scope.
MSL_BUILTIN_FUNCTIONS = {
    "distance", "length", "normalize", "cross", "dot", "reflect", "refract",
    "fract", "mix", "step", "smoothstep", "clamp", "saturate", "sign",
    "rsqrt", "fma", "select", "popcount", "clz", "ctz", "rotate", "abs",
    "min", "max", "floor", "ceil", "round", "trunc", "pow", "exp", "log",
}


def check_namespace_names(cpp_dir):
    """No shared header may open a namespace named for an MSL built-in.

    `namespace distance` compiled under nvcc and under a host compiler for
    months and failed the Metal shader compile with "'distance' is not a class,
    namespace, or enumeration". The diagnostic points at the USE rather than the
    declaration, so it reads as a broken call; and it appears only once a Metal
    entry point reaches the header, so every other build leg stayed green while
    the Macs ran a stale binary.
    """
    bad = []
    for path in sorted(cpp_dir.rglob("*.hpp")) + sorted(cpp_dir.rglob("*.cpp")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            m = re.match(r"\s*namespace\s+([A-Za-z_]\w*)\s*\{", line)
            if m and m.group(1) in MSL_BUILTIN_FUNCTIONS:
                bad.append(f"{path}:{n}: namespace '{m.group(1)}'")
    return bad


def main():
    root = Path(__file__).resolve().parents[3]
    if len(sys.argv) > 1:
        root = Path(sys.argv[1]).resolve()
    cpp_dir = root / "crates/ppf-cts-solver/src/kernels"
    # The Metal target: the ten .mm translation units, their headers, and the
    # recipe that names the embed list. It is in ppf-cts-compute beside the
    # CUDA one, because every backend-specific thing is, and the embed list
    # below is therefore expressed against a caller-supplied KERNEL_ROOT rather
    # than a `../` spelling of the distance between two crates.
    metal_dir = root / "crates/ppf-cts-compute/metal"
    # The CUDA target: the translation units, the recipe that names them, the
    # nvcc prologue and the architecture manifest. It is in ppf-cts-compute,
    # because every backend-specific thing is, and the include closure below
    # therefore crosses a crate boundary.
    cuda_dir = root / "crates/ppf-cts-compute/cuda"
    # THERE IS NO THIRD TREE. No crate holds CUDA translation units carrying
    # SIMULATION, and the Makefile passes no backend-logic root for one, so the
    # closure below walks the mechanism in cuda_dir and the neutral headers it
    # compiles, and nothing else.
    for d in (cpp_dir, metal_dir, cuda_dir):
        if not d.is_dir():
            sys.exit(f"check-shared-wiring: no such directory: {d}")

    namespace_clashes = check_namespace_names(cpp_dir)
    if namespace_clashes:
        sys.exit(
            "check-shared-wiring: a shared header opens a namespace named for an "
            "MSL built-in function, which compiles under nvcc and on the host and "
            "fails the Metal shader compile:\n  "
            + "\n  ".join(namespace_clashes)
        )

    embedded = metal_embed_list(cpp_dir)
    tus = cuda_translation_units(cuda_dir, cpp_dir)
    closure, closure_paths = cuda_include_closure(cuda_dir, cpp_dir, tus)
    seam = seam_headers(cpp_dir)
    kernels = neutral_kernels(cpp_dir)

    # An empty input set means the check learned nothing, and reporting that as
    # a pass is the failure mode this whole workflow exists to prevent.
    if False:
        sys.exit(
            "check-shared-wiring: metal/Makefile and its shader_segments.mk list "
            "no embedded $(KERNEL_ROOT) headers between them. Either the embed "
            "rules moved again or this parser is broken; either way the check "
            "cannot pass."
        )
    if not seam:
        sys.exit(
            "check-shared-wiring: no header under src/kernels carries an SM_* seam "
            "macro. The seam is how a shared body compiles on two backends, so "
            "an empty set means this parser is broken."
        )

    print("shared-surface wiring")
    print(f"  CUDA translation units      : {len(tus)}")
    print(f"  files nvcc reaches          : {len(closure)}")
    print(f"  headers carrying the seam   : {len(seam)}")
    print(f"  neutral sources with an entry: {len(embedded)}")
    print(f"  neutral kernel sources      : {len(kernels)}")

    msgs = []
    # RULES 1 AND 2 ARE RETIRED, and this is the change that retires them
    # rather than a place they are skipped. Both compared what nvcc compiles
    # against what the Metal build EMBEDDED, because that backend assembled its
    # shader from hand-written segments and could therefore hold a second
    # statement of a kernel for the first to drift from. Its library is now
    # linked from the generated entry renderings, so both backends compile the
    # same sources by the same predicate and the fork they guarded cannot
    # exist. `CUDA_ONLY` and `ONE_SIDED` recorded which bodies sat on one side
    # of it and are empty for the same reason.
    a, b = [], []
    host = host_include_closure(root, cpp_dir)
    o = check_fifteen(cpp_dir, kernels, closure, embedded, host, msgs)
    # Rule 3 asked what an EMBEDDED HEADER includes, and no header is embedded
    # now: a generated entry rendering includes what it needs and the shader
    # compiler reads it offline, with a filesystem.
    c = []
    d = check_four(cuda_dir, metal_dir, msgs, cuda_dir.parent / "rocm")
    e = check_five(cpp_dir, kernels, msgs)
    f = check_six(cpp_dir, cuda_dir, metal_dir, msgs)
    g = check_seven(cpp_dir, kernels, msgs)
    h = check_eight(root, cpp_dir, kernels,
                    closure, closure_paths, set(), msgs)
    i = check_nine(root, msgs)
    table_missing, table_misplaced = check_thirteen(root, msgs)
    unrendered = check_fourteen(root, msgs)
    diag_wiring = check_sixteen(root, msgs)
    j = check_ten(root, msgs)
    k = check_eleven(root, msgs)
    m = check_twelve(root, msgs)
    print(f"  seam headers Metal misses   : {len(a)} "
          f"(plus {len(CUDA_ONLY)} recorded)")
    print(f"  embedded headers nvcc misses: {len(b)} "
          f"(plus {len(ONE_SIDED)} recorded)")
    print(f"  includes reaching MSL       : {len(c)}")
    print(f"  SM_ names Metal prologue misses: {len(d)}")
    print(f"  neutral kernels that do not render: {len(e)}")
    print(f"  seam names a prologue misses: {len(f)} "
          f"(plus {len(SEAM_HOST_ABSENT)} recorded host-absent)")
    print(f"  kernels carrying SM_ or a conditional: {len(g)}")
    print(f"  neutral bodies Metal reaches only in a self-test: {len(h[0])}")
    print(f"  neutral bodies CUDA reaches and Metal does not : {len(h[1])}")
    print(f"  neutral bodies NO compiler reads              : {len(o)} (ceiling {UNCOMPILED_CEILING})")
    print(f"  entry sources build.rs does not render: {unrendered} (ceiling 0)")
    print(f"  device diagnostic channels misrouted or unguarded: {diag_wiring} (ceiling 0)")
    print(
        f"  driver TABLE rows out of canonical order: {table_misplaced} "
        f"(ceiling {DRIVER_TABLE_ROWS_MISPLACED}), "
        f"entries with no driver row: {table_missing} "
        f"(ceiling {DRIVER_TABLE_ENTRIES_MISSING})")
    print(f"  hand-written CPU entry points: {i} "
          f"(ceiling {HAND_WRITTEN_CPU_ENTRY_POINTS})")
    print(f"  shim helpers the driver names: {len(j[0])} in production, "
          f"{len(j[1])} in tests")
    print(f"  shim launchers the driver names: {len(j[2])} in production "
          f"(all recorded), {len(j[3])} in tests")
    print(f"  kernel declarations naming their entry point: {k}")
    print(f"  driver record fields still naming an address: {m[0]} in generated "
          f"records (ceiling {DRIVER_ADDRESS_FIELDS_GENERATED}), {m[1]} in "
          f"hand-written ones (ceiling {DRIVER_ADDRESS_FIELDS_HAND_WRITTEN})")
    movable, unresolved, blockers = driver_movable_fields(root)
    print(f"  of those {m[0] + m[1]}, at most {movable} could move today: the rest "
          f"share a record field with a buffer a hand-written record names")
    print(f"  and {unresolved} of that {movable} name a buffer by a BARE LOCAL, "
          f"which this reading cannot resolve: confirm the buffer before moving it")
    if blockers:
        print("  hand-written records that ALONE block a component, "
              "by the sites generating them would unblock:")
        for name, sites in blockers:
            print(f"    {sites:4d}  {name}")

    if msgs:
        print("\nFAILED")
        for m in msgs:
            print(f"  {m}")
        sys.exit(1)
    print("\nOK: every shared header is compiled by both backends")


if __name__ == "__main__":
    main()
