#!/usr/bin/env python3
# File: .github/workflows/scripts/check-compute-neutral.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# `ppf-cts-compute` CONTAINS NO SIMULATION.
#
# The rule, in the repository owner's words: "ppf-cts-compute NEVER contains any
# info about the solver logics. It should only contain general wrappers for GPU
# memory allocs, free, copy and transfer, and kernel launches." That is a CLOSED
# LIST OF FIVE VERBS, plus the platform machinery those five require and nothing
# more: queue and stream management, command encoding, pipeline and library
# loading and caching, and the transport half of the diagnostic channel.
#
# THE TEST IS ONE SENTENCE: could this crate be published on its own and used by
# a program that is not a physics solver?
#
# WHY THIS SCRIPT EXISTS. The crate was created by moving every backend file out
# of `ppf-cts-solver`, which satisfied the rule A SCRIPT CHECKS
# (check-neutral-solver.py: no backend name in the solver) and broke the rule
# ONLY A READER CHECKS (no simulation in the compute crate). An unenforced rule
# in this repository was broken the first time it was tested. So the repair is
# not finished when the files move. It is finished when a check fails on the
# next file that tries to enter.
#
# IT IS A RATCHET, NOT A WALL, for the reason check-neutral-solver.py is one:
# the rule is being converged on rather than held. A wall would fail every build
# until the last file moved, so it would be disabled, and the rule would again
# be enforced by memory. This fails on GROWTH, and it fails equally when a count
# FALLS and its ceiling does not, because a ceiling left stale re-permits every
# move already made.
#
# WHERE THE CONVERGENCE STANDS, measured by this script rather than recalled.
# BOTH HALVES ARE DONE, so the debt ceilings are 0 and 0 and this check is now a
# WALL for what it measures: a file classified DEBT fails outright. CUDA moved
# first (33 files, 18,376 lines at the time) and Metal second (8 files and
# 48,075 lines, `face_math.mm` alone 41,489 of them), separately and in that
# order because the Metal half needs a Mac to verify and mixing the two would
# have made a failure unattributable.
#
# WHAT THE METAL MOVE HAD TO CUT rather than carry, because two files were part
# mechanism and part simulation. `bringup.mm` hard-coded the two parity
# self-tests as steps; it now runs whatever the linked backend lists through
# `backend_module_table`, so it knows a module's name and nothing else, and
# `shader_dump.mm` names no module either. The recipe split the same way: which
# solver headers become shader segments is `shader_segments.mk`, which the
# caller supplies through `BACKEND_LOGIC_ROOT`, and what stays here is flags,
# kernelgen, objects and the link.
#
# HOW THAT MOVE WAS VERIFIED, since neither this check nor any other in CI can
# see it: a pre-compiled Metal library is named by a hash over the ASSEMBLED
# shader text, so the four `.metallib` keys are a fingerprint of every segment,
# every provenance string and the order they are spliced in. On an Apple M1 the
# four keys and the four assembled texts are byte-identical before and after,
# and every parity self-test passes on both sides of the move.
#
# ======================================================================
# WHAT THIS CHECK CAN CATCH, AND WHAT IT CANNOT. Read this before
# trusting a green run.
# ======================================================================
#
# RULE 0, CARGO ISOLATION, IS EXACT FOR THE RUST HALF AND WORTHLESS FOR THE
# REST. The strongest check available for a crate is that it builds alone, and
# it cannot be satisfied by renaming a file. For this crate it is ALREADY TRUE
# and it proves almost nothing: `cargo tree -p ppf-cts-compute` shows exactly
# one dependency, rayon, because cargo compiles only the `.rs` files here. The
# 66,000 lines of `.cu` and `.mm` are compiled by the SOLVER's build script
# through the Makefiles this crate owns, so cargo never sees them. Rule 0
# therefore checks what it can honestly claim: that the manifest names no
# `ppf-cts-*` crate, which is a real guarantee that no Rust file in here can
# reference a solver type, and says nothing at all about the C++ half.
#
# RULE 3, THE INCLUDE CLOSURE, IS THE REAL GATE, AND IT IS THE C++ ANALOGUE OF
# RULE 0. The goal state is that the mechanism sources compile against a kernel
# root holding only neutral headers. That real compile cannot run in the wiring
# job (no nvcc, no Xcode, and the whole point of that job is that it needs no
# toolchain), so this computes its static form: the transitive quoted-include
# closure of every MECHANISM C/C++/Objective-C++ file, restricted to headers
# under the solver's kernel root, must lie inside NEUTRAL_KERNEL_HEADERS. A C++
# file cannot USE a solver type without including the header that declares it,
# so this is a type-level check with no false positives from prose, and it is
# transitive, so the allowlist defends itself: if a listed header ever grew an
# include of `data.hpp`, `data.hpp` would appear in the closure and fail.
# THAT REAL COMPILE IS NOW POSSIBLE ON BOTH TARGETS, which was this gate's
# stated goal. Every mechanism file on both sides reaches only
# NEUTRAL_KERNEL_HEADERS, which is why CLOSURE_EXCEPTIONS is empty. The Metal
# recipe goes one step further and makes it enforced rather than measured: it
# passes NO include path toward the solver, so a mechanism source that reached
# for a solver type by name does not compile at all.
#
# THERE IS NO RULE 6, AND THE GAP IN THE NUMBERING IS THE RESULT IT WAS AFTER.
# It was the ratchet on a third crate, which existed because backend-specific
# code that is none of the five verbs fits neither this crate nor
# `ppf-cts-solver`. Every such file has been separated: the physics went
# neutral and the mechanism came here, the crate went with its workspace member
# entry, and the ratchet that counted it down came out in the same change,
# which is what the rule's own text said would retire it. What refuses a fourth
# crate is rule 0 below, not a count over a directory that is not there.
#
# RULE 4 IS A CENSUS, NOT A JUDGMENT. Every tracked file must appear in exactly
# one of the two tables below with a recorded reason. That is only as honest as
# its entries, which is the known weakness of a per-file allowlist. What it buys
# is that the next file to enter the crate CANNOT do so silently: it fails until
# somebody writes down which of the five verbs it is, in a diff a reviewer sees.
#
# THERE IS DELIBERATELY NO KEYWORD SCAN, AND THAT IS A MEASUREMENT, NOT AN
# OVERSIGHT. A 23-word vocabulary of solver terms run over the mechanism source
# files fails 24 of 28, and 15 of 28 with the word "solver" itself removed from
# it: "Schwarz" in `cpu/mem.rs` and `metal/metal_context.hpp`, "PCG" in
# `cuda/utility/dispatcher.cu`, `cuda/seam_cuda.cuh` and `cuda/cuda_vec.hpp`,
# "DataSet" in `src/device.rs`, `cpu/host.rs` and `metal/arena.hpp`, "Newton" in
# `cpu/sched.rs`. Every one is prose: a comment explaining why a pool grows the
# way it does, or why a helper is `__device__` alone, and the `seam_cuda.cuh`
# "barrier" is a THREADGROUP barrier, an unrelated meaning of the word. Keyword density already produced a wrong answer on this crate once, in
# the opposite direction, classifying `contact.cu` as mechanism because its 323
# solver references were a small fraction of its 3,251 lines. A file is
# classified by what it DOES.
#
# WHAT NOTHING HERE CATCHES, stated so the next reader does not assume it is
# covered: simulation added INSIDE an already-classified mechanism file, in a
# way that needs no new include and defines no new entry point. A convergence
# test written in plain arithmetic inside `arena.mm` would pass every rule
# below. The defense against that is review, and the reason it is a smaller hole
# than it sounds is that the five verbs are narrow enough that such a function
# has no caller in this crate: it would have to be called across the C ABI,
# which puts its declaration in a header this gate reads.

import os
import re
import subprocess
import sys

CRATE = "crates/ppf-cts-compute"

# The neutral kernel tree, which belongs to the SOLVER. This crate's recipes
# reach it only through a caller-supplied KERNEL_ROOT, which is what lets the
# crate be published on its own at all, so the path is here only to resolve
# includes for rule 3.
KERNEL_ROOT = "crates/ppf-cts-solver/src/kernels"

# ======================================================================
# THE CENSUS. Every tracked file under the crate appears exactly once,
# in MECHANISM or in DEBT, with a reason a reviewer can check against the
# file. A file in neither fails the build.
# ======================================================================

# MECHANISM: one of the five verbs, or the platform machinery they require.
# The reason names the verb.
MECHANISM = {
    # ---- the Rust host, the crate as cargo sees it -------------------
    "Cargo.toml":
        "PLATFORM: declares links=ppfctscompute and one dependency, rayon",
    "build.rs":
        "PLATFORM: creates the libdir and prints cargo:LIBDIR; compiles nothing",
    "src/lib.rs":
        "PLATFORM: module declarations and a re-export",
    "src/device.rs":
        "ALLOC/FREE/COPY/LAUNCH: the trait, whose whole method set is alloc, "
        "grow, free, write, read, run, record, replay, counters",
    "src/abi.rs":
        "ALLOC/FREE/COPY/LAUNCH: the C ABI target; binds be_* and carries "
        "the one Rust declaration of the seam's own header. Its test double is "
        "a fake LIBRARY, not a fake solver: it allocates, copies and counts, "
        "and computes nothing",
    "cpu/host.rs":
        "ALLOC/FREE/COPY/LAUNCH: the host target; binds a caller-supplied "
        "kernel table and calls through it",
    "cpu/mem.rs":
        "ALLOC/FREE: the block allocator; picks the smallest fitting block",
    "cpu/sched.rs":
        "LAUNCH: the host range cut. FLAGGED: chunk_for divides a fixed work "
        "budget by a caller-supplied cost, which is a tuning constant rather "
        "than a physical quantity, and it is the one number in the mechanism "
        "set worth re-reading if the rule is ever tightened",
    "src/build/mod.rs":
        "PLATFORM: module declaration",
    "cpu/recipe.rs":
        "PLATFORM: runs the generator and drives cc::Build",

    # ---- the transcompiler -------------------------------------------
    "seam/kernelgen.py":
        "LAUNCH/PLATFORM: renders one neutral body per target; its only "
        "solver-adjacent strings are MSL reserved words",
    "seam/test_kernelgen.py":
        "PLATFORM: the generator's own tests",

    # ---- recipes ------------------------------------------------------
    "cpu/tests/Makefile":
        "PLATFORM: a recipe taking KERNEL_ROOT; its test sources stayed in the "
        "solver, which is this rule applied correctly",
    "cuda/Makefile":
        "PLATFORM: recipe; derives its source list mechanically with find",
    "cuda/tests/Makefile":
        "PLATFORM: recipe, for the two binaries that exercise one of the five "
        "verbs; the thirteen whose subject is simulation moved with their "
        "sources",
    "cuda/cuda_arch.txt":
        "PLATFORM: which cubins the device link emits",

    # ---- CUDA mechanism ------------------------------------------------
    "metal/backend/backend.mm":
        "ALL FIVE VERBS: this target as the C ABI a driver calls, on the same "
        "terms as cuda/backend/backend.cu. It answers seam/backend_abi.h and "
        "nothing else, NAMES NO KERNEL, and reads no field of an argument "
        "record; its table is generated from the caller's own [[seam::entry]] "
        "declarations and a record crosses as opaque bytes whose only checked "
        "property is its length. It assembles no shader either: the kernels "
        "reach it as a pre-built library the build linked from the generated "
        "entry renderings, which it loads by path",
    "cuda/backend/backend.cu":
        "ALL FIVE VERBS: this target as the C ABI a driver calls. It answers "
        "seam/backend_abi.h and nothing else: open, allocate, grow, free, "
        "write, read, encode, submit, record, replay, counters. It NAMES NO "
        "KERNEL and reads no field of an argument record; its table is "
        "generated from the caller's own [[seam::entry]] declarations and a "
        "record crosses it as opaque bytes whose only checked property is its "
        "length",
    "cuda/arena/arena.cu":
        "ALLOC/FREE/COPY: bump-pointer arenas with a coalescing free list",
    "cuda/arena/arena.hpp":
        "ALLOC/FREE/COPY: the allocator's interface, and nvcc's alone: "
        "g_bases and resolve carry __device__, so a compiler that does not "
        "know that keyword cannot read it",
    "cuda/utility/dispatcher.cu":
        "LAUNCH: the dispatch primitive and its macro",
    "cuda/utility/dispatcher.hpp":
        "LAUNCH: the name a caller reaches the dispatch primitive by; the "
        "recipe compiles dispatcher.cu into no object of its own, so this is "
        "how the eleven callers pull it in",
    "cuda/cuda_utils.hpp":
        "PLATFORM: maps a cudaError_t to a fatal code; stages a transfer "
        "through pinned scratch",
    # ---- the ROCm target ---------------------------------------------
    "rocm/seam_hip.hiph":
        "PLATFORM: the hipcc prologue, the HIP arm of what seam_cuda.cuh is "
        "for nvcc. It spells lane geometry, atomics and math for one "
        "compiler and computes nothing",
    "rocm/hip_utils.hpp":
        "PLATFORM: maps a hipError_t to a fatal code for the launcher a "
        "generated entry point carries. No allocation and no simulation",
    "rocm/diagnostics/diagnostics.hpp":
        "PLATFORM: the transport half of the diagnostic channel for this "
        "backend. The RECORD layout is declared above the seam, not here",
    "rocm/arena/arena.hpp":
        "ALLOC/FREE/COPY: the (arena, offset) data model for this backend, "
        "on the same terms as cuda/arena/arena.hpp and metal/arena.hpp",
    "rocm/rocm_arch.txt":
        "PLATFORM: the shipped AMD target list, the single source every ROCm "
        "consumer derives from, as cuda/cuda_arch.txt is for CUDA",
    "rocm/generic_targets.txt":
        "PLATFORM: which physical parts each GENERIC target in rocm_arch.txt "
        "covers, read from the pinned toolchain's own documentation. It is a "
        "property of the compiler rather than of the simulation: a device "
        "reports gfx1100 and never gfx11-generic, so anything asking whether "
        "this build can run here expands the one list through this one",
    "rocm/Makefile":
        "PLATFORM: recipe; derives its source list mechanically with find, and "
        "its AMD target list from rocm_arch.txt beside it",
    "rocm/arena/arena.hip":
        "ALLOC/FREE/COPY: bump-pointer arenas with a coalescing free list, the "
        "HIP rendering of what cuda/arena/arena.cu is for nvcc",
    "rocm/backend/backend.hip":
        "ALL FIVE VERBS: this target as the C ABI a driver calls. It answers "
        "seam/backend_abi.h and nothing else, on the same terms as "
        "cuda/backend/backend.cu: it NAMES NO KERNEL and reads no field of an "
        "argument record, its table is generated from the caller's own "
        "[[seam::entry]] declarations, and a record crosses it as opaque bytes "
        "whose only checked property is its length",
    "rocm/diagnostics/diagnostics.hip":
        "PLATFORM: the ring and assert channel, create/reset/read/destroy",
    "rocm/hip_vec.hpp":
        "ALLOC/FREE: the hipcc definition of ArenaPtr/ArenaVec/ArenaVecVec, the "
        "(arena, offset) handles the allocator hands out. It names vec/vec.hpp "
        "rather than data.hpp, so the scene model is not in its closure",
    "rocm/mem.hpp":
        "COPY: hipMemcpy in both directions, over the neutral view types and "
        "the arena handles beside it. It states no size of its own: every "
        "count is the caller's",
    "rocm/utility/dispatcher.hip":
        "LAUNCH: the dispatch primitive and its macro",
    "rocm/utility/dispatcher.hpp":
        "LAUNCH: the name a caller reaches the dispatch primitive by, as "
        "cuda/utility/dispatcher.hpp is on that target",
    "cuda/seam_cuda.cuh":
        "PLATFORM: the nvcc rendering table, flat defines with no conditional",
    "cuda/cuda_vec.hpp":
        "ALLOC/FREE: the nvcc definition of ArenaPtr/ArenaVec/ArenaVecVec, the "
        "(arena, offset) handles the allocator hands out. It names vec/vec.hpp "
        "rather than data.hpp, so the scene model is not in its closure",
    "cuda/mem.hpp":
        "COPY: cudaMemcpy in both directions, over the neutral view types and "
        "the arena handles beside it. It states no size of its own: every "
        "count is the caller's",
    "cuda/diagnostics/diagnostics.hpp":
        "PLATFORM: the channel's interface and the device-side macros a kernel "
        "body writes through. The RECORD layout is not here: it is what the "
        "two sides agree on, so it stays in the neutral tree",
    "cuda/diagnostics/diagnostics.cu":
        "PLATFORM: the ring and assert channel, create/reset/read/destroy, "
        "and nothing beyond those four",
    "cuda/tests/backend_globals.cu":
        "PLATFORM: storage for the driver's extern globals so a test links",
    "cuda/tests/test_cuda_arena.cu":
        "ALLOC/FREE: exercises the allocator",
    "cuda/tests/test_cuda_diagnostics.cu":
        "PLATFORM: exercises the diagnostic channel",

    # ---- Metal mechanism -----------------------------------------------
    "metal/metal_context.mm":
        "PLATFORM: owns the device, the command queue and the buffer "
        "lifecycle, and the one mathMode Safe options helper",
    "metal/metal_context.hpp":
        "PLATFORM: its interface",
    "metal/arena.mm":
        "ALLOC/FREE/COPY: allocator_alloc/grow/free/write/read/fill over one "
        "MTLBuffer per arena",
    "metal/arena.hpp":
        "ALLOC/FREE/COPY: its interface",
    "metal/pipeline_cache.mm":
        "PLATFORM: MTLBinaryArchive open, make, flush, prune",
    "metal/pipeline_cache.hpp":
        "PLATFORM: its interface",
    "metal/shader_compiler.mm":
        "PLATFORM: the MSL rendering table and segment concatenation",
    "metal/shader_compiler.hpp":
        "PLATFORM: its interface",
    "metal/diagnostics.mm":
        "PLATFORM: the diagnostic channel's transport half",
    "metal/diagnostics.hpp":
        "PLATFORM: its interface",
    "metal/shader_dump.mm":
        "PLATFORM: runs bring-up with no scene and writes the assembled source; "
        "names no module, taking the list from backend_module_table",
    "metal/embed_source.py":
        "PLATFORM: wraps a source file as a C string literal",
    "metal/msl_prologue.py":
        "PLATFORM: copies this backend's own shader prologue out of "
        "shader_compiler.mm and diagnostics.mm into a file, so an offline "
        "compile sees the names a run-time compile sees. It reads two of this "
        "crate's own sources and knows nothing of what they are a prologue for",
    "metal/entry_harness.py":
        "LAUNCH: assembles the translation unit that hands one GENERATED entry "
        "point to the shader compiler. It follows the quoted includes of a "
        "neutral source under a caller-supplied kernel root and names no file "
        "in it; an entry point is an argument record plus a thread index plus "
        "a call, which is the launch verb and nothing else",
    "metal/entries_probe.mm":
        "LAUNCH, and the check that the launch verb can be performed at all: it "
        "loads a compiled Metal library and creates a compute pipeline for "
        "every function name the library exports. It names no kernel, reads no "
        "argument record and dispatches nothing; the library path arrives on "
        "argv. Any program with a .metallib could run it",
    "metal/bringup.mm":
        "PLATFORM: the order the device, allocator, diagnostic channel and "
        "shader prologue are started in, then whatever modules the linked "
        "backend lists. It knows a module's name and that it either succeeds "
        "or explains itself, and nothing else",
    "metal/bringup.hpp":
        "PLATFORM: its interface, plus BringupModule and the "
        "backend_module_table seam the linked backend fills",
    "metal/Makefile":
        "PLATFORM: recipe. Which shader segments there are is not in it: that "
        "list names solver headers one by one and travels with the sources "
        "that splice them, included from a caller-supplied BACKEND_LOGIC_ROOT",
}

# DEBT: carries simulation. Each entry leaves when its logic becomes neutral
# Rust in the driver, or a generated entry point, and the ceilings below are
# lowered in the SAME change. A file that is part mechanism and part simulation
# is counted here in full, because it is not yet clean; its reason records the
# split so the reader knows what is inside.
DEBT = {
    # EMPTY. Both halves of the violation are repaired: no `.cu` and no `.mm`
    # here carries simulation, because the simulation is neutral now, one
    # `*.kernel.cpp` per kernel in `ppf-cts-solver`, and neither recipe here
    # compiles physics from anywhere.
    #
    # WITH THE CEILINGS AT ZERO THIS TABLE IS A WALL RATHER THAN A RATCHET. A
    # file classified DEBT fails the build outright, so the only way to add one
    # is to raise a ceiling in the same diff, which is the reviewer's cue. That
    # is the state this check was written to reach and it must not be relaxed
    # to accommodate a file: a backend file is written neutral, or it is one of
    # the five verbs and belongs here. There is no third destination, and
    # wanting one is the diagnostic that the separation has not been done.
}

# ======================================================================
# RULE 3's ALLOWLIST. A header under the solver's kernel root that a
# mechanism file may include. Each is one of the five verbs' own
# interface, or the seam that renders it. The list is transitive-safe:
# if any of these ever included a simulation header, that header would
# appear in the closure and fail.
# ======================================================================
NEUTRAL_KERNEL_HEADERS = {
    "seam/backend_abi.h":
        "ALL FIVE VERBS: the C ABI itself. It DECLARES allocation, free, "
        "transfer and launch plus the platform machinery those need, and "
        "nothing else; an argument record crosses it as opaque bytes, so it "
        "carries no simulation and could not, being the one declaration a "
        "target implements. It is the most admissible header on this list "
        "rather than the least, and it is under the solver's kernel root "
        "because it belongs to the SEAM rather than to either side of it",
    "arena_handle.hpp":
        "ALLOC: the (arena, offset, size) handle, now an alias of the ABI's "
        "own BeHandle rather than a second declaration of it",
    "main/fatal.hpp":
        "PLATFORM: the fatal code enum the diagnostic transport reports",
    "diagnostic_record.hpp":
        "PLATFORM: the diagnostic record's names, over the ABI's own "
        "BeDiagRecord rather than a second definition of it",
    "common.hpp":
        "PLATFORM: the execution-space and limit macros every compiler needs",
    "float_math.hpp":
        "PLATFORM: the float-only transcendental spellings the rendering "
        "tables map onto",
    "vec/vec.hpp":
        "ALLOC/COPY: Vec<T> and VecVec<T>, the plain-old-data VIEW an "
        "allocation is handed back as. Bounds-checked subscript, resize, and "
        "clear through kernels::fill_view, which it declares and no backend "
        "header does; no arithmetic on what it holds. NOT primitives/vec_ops.hpp, "
        "which declares the same fill beside the PCG's scalar_div and its "
        "breakdown latch",
    "seam/seam.hpp":
        "PLATFORM: picks the rendering table for the compiler in hand, which "
        "is the definition of how a device is made to compute",
    "seam/seam_host.h":
        "PLATFORM: the host C++ rendering table, reached through seam.hpp",
}

# A mechanism file whose closure reaches a header NOT in the list above,
# recorded with the reason and the change that removes it. THE COUNT IS
# RATCHETED: an exception is a debt, not a permission.
CLOSURE_EXCEPTIONS = {
    # EMPTY, AND THAT IS THE MEASUREMENT WORTH KEEPING. Every mechanism file in
    # this crate compiles against the allowlist above and nothing else. An entry
    # here would be a mechanism file reaching a solver header: an allocator that
    # reaches `data.hpp`, for instance, pulls the whole scene model into its own
    # include closure.
}

# ======================================================================
# RULE 5's ALLOWLIST. Rule (1b): an entry point is GENERATED from the
# neutral body, never hand-written. A mechanism file that defines a
# device entry point must say which mechanism needs it, and the COUNT
# per file is recorded so a new one fails even in an allowed file.
# ======================================================================
KERNEL_ENTRIES_ALLOWED = {
    "cuda/utility/dispatcher.cu": (
        2, "LAUNCH: indexed_apply and indexed_apply_diag ARE the dispatch "
           "primitive"),
    "cuda/tests/test_cuda_arena.cu": (
        3, "ALLOC: a device kernel is the only way to read back what the "
           "allocator handed out"),
    "cuda/tests/test_cuda_diagnostics.cu": (
        3, "PLATFORM: the channel is written from the device, so exercising it "
           "needs a device writer. The third is `always_fails`, shared by the "
           "two cases that check what an assert does when the channel is "
           "UNATTACHED and when it is the global one a generated launcher "
           "hands over"),
    "cuda/diagnostics/diagnostics.cu": (
        1, "PLATFORM: `selftest_kernel` is the one check of this channel that "
           "runs on the device, and it is the only one that can fail on a "
           "machine whose source is correct. Everything else about the channel "
           "is established by reading code, and neither defect this catches was "
           "visible to any build: a private channel drained instead of the "
           "global one is green everywhere, and an unattached one is green "
           "until the first check fires and then is an illegal address naming "
           "no cause"),
    "metal/metal_context.mm": (
        1, "PLATFORM: sentinel_math_mode is the mathMode Safe probe, which "
           "is a compile-options question rather than a computed value"),
}

# ======================================================================
# THE CEILINGS. Lower each in the change that moves a file, never
# separately: a ceiling left stale re-permits every move already made.
# ======================================================================
# ======================================================================
CEILING_DEBT_LINES = {"cuda": 0, "metal": 0}
CEILING_DEBT_FILES = {"cuda": 0, "metal": 0}
CEILING_CLOSURE_EXCEPTIONS = 0
CEILING_KERNEL_ENTRY_FILES = 5

INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*(?:include|import)[ \t]+"([^"]+)"', re.M)
GLOBAL_RE = re.compile(r"\b__global__\b")
MSL_KERNEL_RE = re.compile(r"\bkernel[ \t]+void\b")
CXX_EXT = (".cu", ".cuh", ".hpp", ".h", ".cpp", ".mm", ".metal")


def tracked(prefix):
    """Every tracked path under a prefix.

    Read from git rather than the filesystem so a build directory, a stale
    artifact or an editor's scratch file cannot change the verdict.
    """
    out = subprocess.run(["git", "ls-files", prefix],
                         capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if p]


def untracked(prefix):
    """Source files under a prefix that git does not list yet.

    THE VERDICT STAYS OVER TRACKED FILES, for the reason `tracked` states. This
    answers the confusing half: every ceiling here is matched for EQUALITY, so a
    NEW file that is not staged leaves the counts exactly where they were and
    the run PASSES, hiding growth until the moment it is added. The ratchet does
    catch it then, as growth, but a reader who saw the pass first has to work
    out why. Naming the files turns that into one line. Only source extensions,
    because a scratch note beside a backend is not a backend.
    """
    out = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", prefix],
        capture_output=True, text=True, check=True).stdout
    keep = (".cu", ".cuh", ".mm", ".metal", ".msl", ".hpp", ".h", ".cpp", ".rs")
    return sorted(p for p in out.splitlines() if p and p.endswith(keep))


def line_count(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def backend_of(rel):
    head = rel.split("/")[0]
    return head if head in ("cuda", "metal") else "(rust/seam/host)"


def resolve_include(inc, curdir, search):
    for d in [curdir] + search:
        cand = os.path.normpath(os.path.join(d, inc))
        if os.path.isfile(cand):
            return cand
    return None


def kernel_root_closure(rel):
    """Headers under KERNEL_ROOT that a mechanism file transitively includes.

    Quoted includes only, resolved the way both recipes resolve them: relative
    to the including file, then against KERNEL_ROOT and the backend directory.
    An include that resolves to neither is a system or framework header and is
    not this rule's business.
    """
    start = os.path.join(CRATE, rel)
    search = [KERNEL_ROOT, os.path.join(CRATE, rel.split("/")[0])]
    seen, stack, reached = set(), [start], set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        try:
            with open(cur, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        for inc in INCLUDE_RE.findall(text):
            got = resolve_include(inc, os.path.dirname(cur), search)
            if got is None:
                continue
            if os.path.commonpath([os.path.abspath(got),
                                   os.path.abspath(KERNEL_ROOT)]) == \
                    os.path.abspath(KERNEL_ROOT):
                reached.add(os.path.relpath(got, KERNEL_ROOT))
            stack.append(got)
    return reached


def main():
    paths = tracked(CRATE)
    if not paths:
        sys.exit(f"check-compute-neutral: git lists no tracked file under "
                 f"{CRATE}. Either the crate moved or this script is looking in "
                 f"the wrong place, and reporting a clean count over nothing is "
                 f"the one outcome worse than reporting a violation.")
    if not os.path.isdir(KERNEL_ROOT):
        sys.exit(f"check-compute-neutral: {KERNEL_ROOT} does not exist, so rule "
                 f"3 would resolve no include and pass over nothing.")

    rels = sorted(p[len(CRATE) + 1:] for p in paths)
    failures = []
    stale = []

    # ---- RULE 0: cargo isolation -------------------------------------
    manifest = os.path.join(CRATE, "Cargo.toml")
    with open(manifest, encoding="utf-8") as f:
        manifest_text = f.read()
    solver_deps = sorted(set(re.findall(
        r"^\s*(ppf-cts-[a-z-]+)\s*=", manifest_text, re.M)))
    rule0 = not solver_deps

    # ---- RULE 4: the census ------------------------------------------
    unclassified = [r for r in rels if r not in MECHANISM and r not in DEBT]
    both = [r for r in rels if r in MECHANISM and r in DEBT]
    # A DIRECTORY THIS TREE DOES NOT CARRY AT ALL IS DORMANT, NOT STALE, AND THE
    # TWO MUST NOT BE CONFLATED.
    #
    # The tables below are shared byte for byte by two trees. The public-bound
    # one carries no test directory at all, by the rule that holds every
    # `*/tests/` back from a drop, so every row naming one would read as "a move
    # left the census behind" there. That message asks the reader to delete a row
    # that is correct in the tree it was written for, and following it would
    # un-record entry points that still exist.
    #
    # The test is whether ANY tracked file sits under the row's directory. When
    # none does, the whole directory is absent from this tree and the row is
    # dormant; when some do, a row naming a missing file is a genuine move and
    # still fails. So renaming one file under a directory its siblings still
    # occupy is caught exactly as before.
    #
    # DORMANT ROWS ARE COUNTED AND PRINTED, never dropped in silence: a
    # disappearing directory is the one way this could hide a real move, and a
    # reader sees the number on every run.
    tracked_dirs = {os.path.dirname(r) for r in rels}
    def dormant(rel):
        d = os.path.dirname(rel)
        return bool(d) and not any(
            t == d or t.startswith(d + "/") for t in tracked_dirs)

    ghosts = sorted(r for r in (set(MECHANISM) | set(DEBT)) - set(rels)
                    if not dormant(r))
    dormant_rows = sorted(r for r in (set(MECHANISM) | set(DEBT)) - set(rels)
                          if dormant(r))

    # ---- RULE 2: the debt ratchet ------------------------------------
    debt_lines, debt_files = {}, {}
    for r in rels:
        if r in DEBT:
            b = backend_of(r)
            debt_lines[b] = debt_lines.get(b, 0) + line_count(
                os.path.join(CRATE, r))
            debt_files[b] = debt_files.get(b, 0) + 1

    # ---- RULE 3: the include closure ---------------------------------
    closure_violations = {}
    for r in rels:
        if r not in MECHANISM or not r.endswith(CXX_EXT):
            continue
        bad = sorted(kernel_root_closure(r) - set(NEUTRAL_KERNEL_HEADERS))
        if bad:
            closure_violations[r] = bad
    unexpected_closure = {k: v for k, v in closure_violations.items()
                          if k not in CLOSURE_EXCEPTIONS}
    unused_closure_exc = sorted(set(CLOSURE_EXCEPTIONS) - set(closure_violations))

    # ---- RULE 5: hand-written entry points ---------------------------
    entry_counts = {}
    for r in rels:
        if r not in MECHANISM or not r.endswith(CXX_EXT):
            continue
        with open(os.path.join(CRATE, r), encoding="utf-8",
                  errors="replace") as f:
            text = f.read()
        n = len(GLOBAL_RE.findall(text)) + len(MSL_KERNEL_RE.findall(text))
        if n:
            entry_counts[r] = n
    entry_bad = []
    for r, n in sorted(entry_counts.items()):
        allowed = KERNEL_ENTRIES_ALLOWED.get(r)
        if allowed is None:
            entry_bad.append(f"{r} defines {n} device entry point(s) and is not "
                             f"in KERNEL_ENTRIES_ALLOWED")
        elif n > allowed[0]:
            entry_bad.append(f"{r} defines {n} device entry point(s), over the "
                             f"{allowed[0]} recorded")
    entry_stale = [f"{r} defines {entry_counts.get(r, 0)}, under the {n} recorded"
                   for r, (n, _) in sorted(KERNEL_ENTRIES_ALLOWED.items())
                   if entry_counts.get(r, 0) < n and not dormant(r)]
    unused_entry_exc = sorted(r for r in set(KERNEL_ENTRIES_ALLOWED)
                              - set(entry_counts) if not dormant(r))
    # The ceiling counts the files a row was recorded for, so the files this tree
    # does not carry come off it rather than off the ceiling. One number then
    # serves both trees, which is what keeps the two copies of this file
    # identical.
    dormant_entry_files = sorted(r for r in KERNEL_ENTRIES_ALLOWED if dormant(r))
    entry_ceiling = CEILING_KERNEL_ENTRY_FILES - len(dormant_entry_files)

    # ================= report =========================================
    print("compute-neutral census (ratchet)")
    print(f"  tracked files              : {len(rels)}")
    print(f"  classified mechanism       : {sum(1 for r in rels if r in MECHANISM)}")
    print(f"  classified debt            : {sum(1 for r in rels if r in DEBT)}")
    print(f"  rule 0 cargo isolation     : "
          f"{'clean' if rule0 else 'FAILED: ' + ', '.join(solver_deps)}")
    for b in sorted(CEILING_DEBT_LINES):
        print(f"  debt {b:<6}                : {debt_files.get(b, 0)} files "
              f"(ceiling {CEILING_DEBT_FILES[b]}), "
              f"{debt_lines.get(b, 0)} lines (ceiling {CEILING_DEBT_LINES[b]})")
    print(f"  closure exceptions         : {len(closure_violations)} "
          f"(ceiling {CEILING_CLOSURE_EXCEPTIONS})")
    print(f"  files with an entry point  : {len(entry_counts)} "
          f"(ceiling {entry_ceiling})")
    if dormant_rows:
        print(f"  rows for directories this tree does not carry: "
              f"{len(dormant_rows)} dormant, "
              f"{len(dormant_entry_files)} of them entry rows")
        for r in dormant_rows:
            print(f"      dormant: {r}")
    for r, bad in sorted(closure_violations.items()):
        mark = "allowed" if r in CLOSURE_EXCEPTIONS else "UNEXPECTED"
        print(f"      closure {mark}: {r} -> {', '.join(bad)}")

    # ================= verdict ========================================
    if not rule0:
        failures.append(
            f"Cargo.toml names {', '.join(solver_deps)}. This crate depends on "
            f"no ppf-cts crate: the dependency edge points one way, and a Rust "
            f"file here must not be able to reference a solver type.")
    if unclassified:
        failures.append(
            "these files are in neither MECHANISM nor DEBT, so nobody has said "
            "which of the five verbs they are:\n      "
            + "\n      ".join(unclassified))
    if both:
        failures.append("these files are in BOTH tables: " + ", ".join(both))
    if ghosts:
        stale.append("these table entries name no tracked file, so a move left "
                     "the census behind: " + ", ".join(ghosts))
    for b in sorted(CEILING_DEBT_LINES):
        if debt_files.get(b, 0) > CEILING_DEBT_FILES[b]:
            failures.append(f"{b}: {debt_files[b]} debt files, over the "
                            f"{CEILING_DEBT_FILES[b]} recorded")
        if debt_lines.get(b, 0) > CEILING_DEBT_LINES[b]:
            failures.append(f"{b}: {debt_lines[b]} debt lines, over the "
                            f"{CEILING_DEBT_LINES[b]} recorded")
        if debt_files.get(b, 0) < CEILING_DEBT_FILES[b]:
            stale.append(f'CEILING_DEBT_FILES["{b}"] to {debt_files.get(b, 0)}')
        if debt_lines.get(b, 0) < CEILING_DEBT_LINES[b]:
            stale.append(f'CEILING_DEBT_LINES["{b}"] to {debt_lines.get(b, 0)}')
    for r, bad in sorted(unexpected_closure.items()):
        failures.append(
            f"{r} is classified MECHANISM and its include closure reaches "
            f"{', '.join(bad)} under {KERNEL_ROOT}. A mechanism file may reach "
            f"only the interfaces of the five verbs; reaching a simulation "
            f"header means it is not mechanism.")
    if len(closure_violations) > CEILING_CLOSURE_EXCEPTIONS:
        failures.append(f"{len(closure_violations)} closure exceptions, over "
                        f"the {CEILING_CLOSURE_EXCEPTIONS} recorded")
    if len(closure_violations) < CEILING_CLOSURE_EXCEPTIONS:
        stale.append(f"CEILING_CLOSURE_EXCEPTIONS to {len(closure_violations)}")
    if unused_closure_exc:
        stale.append("these CLOSURE_EXCEPTIONS no longer apply and re-permit a "
                     "move already made: " + ", ".join(unused_closure_exc))
    failures.extend(entry_bad)
    if len(entry_counts) > entry_ceiling:
        failures.append(f"{len(entry_counts)} mechanism files define a device "
                        f"entry point, over the {entry_ceiling} recorded")
    if len(entry_counts) < entry_ceiling:
        stale.append(f"CEILING_KERNEL_ENTRY_FILES to "
                     f"{len(entry_counts) + len(dormant_entry_files)}")
    stale.extend(entry_stale)
    if unused_entry_exc:
        stale.append("these KERNEL_ENTRIES_ALLOWED entries no longer apply: "
                     + ", ".join(unused_entry_exc))


    if failures:
        print("\nFAILED")
        for m in failures:
            print(f"  {m}")
        print(f"\n  {CRATE} holds device memory allocation, free, copy and "
              f"transfer, and kernel launch, plus the platform machinery those "
              f"five require. Nothing else. The test is one sentence: could "
              f"this crate be published on its own and used by a program that "
              f"is not a physics solver?")
        sys.exit(1)
    if stale:
        print("\nFAILED")
        print("  a count fell but its ceiling did not. Lower it in this change: "
              "a ceiling left stale re-permits every move already made.")
        for m in stale:
            print(f"      {m}")
        sys.exit(1)

    loose = untracked(CRATE)
    if loose:
        print(f"\n  NOT COUNTED, because `git ls-files` does not list them yet: "
              f"{len(loose)} untracked source file(s) under these crates. Every "
              f"ceiling above is matched for EQUALITY, so staging them can only "
              f"RAISE a count, and a pass taken before they are staged says "
              f"nothing about them.")
        for path in loose:
            print(f"      unstaged  {path}")

    total_debt = sum(debt_lines.values())
    if total_debt == 0:
        print("\nOK: the compute crate holds mechanism only")
    else:
        print(f"\nOK: no growth ({total_debt} lines of simulation remain; the "
              f"ceilings are the debt)")


if __name__ == "__main__":
    main()
