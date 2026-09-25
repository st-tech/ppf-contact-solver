#!/usr/bin/env python3
# File: kernelgen.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Renders one neutral kernel source into the form a single backend compiles.

A kernel body lives once, as a `*.kernel.cpp` file under `cpp/`. That file is
plain C++: a compiler-independent parse, no preprocessor conditional, no macro
of its own, and no spelling that belongs to one backend. It carries the two
things a backend cannot infer, and it carries them as C++ attributes:

    [[seam::thread]] [[seam::device]] [[seam::threadgroup]] [[seam::constant]]
        the address space of a pointer or reference parameter
    [[seam::device_fn]] [[seam::host_device_fn]]
        the execution space of a function

This script reads such a file and writes one of three outputs, selected by
`--target`:

    cu      what nvcc compiles.       Address spaces are dropped, the execution
                                      space becomes __device__ or
                                      __host__ __device__.
    metal   what the Metal shader     Address spaces become the MSL keywords,
            compiler compiles.        the execution space is dropped.
    cpp     what a host C++ compiler  Both are dropped: one address space, one
            compiles.                 execution space.

WHY A GENERATOR AND NOT A MACRO. The two annotations cannot both be
preprocessor work. MSL requires an address space on every pointer and reference
type, and an unknown attribute cannot stand in for one: the Metal shader
compiler answers `[[seam::thread]] const float &x` with "reference type must
have explicit address space qualifier" and ignores the attribute
(-Wunknown-attributes). So `thread`, `device` and `threadgroup` must be present
as keywords in the text the shader compiler reads. Meanwhile nvcc expands
`__device__` to `__attribute__((device))`, so a `#define device` erasing the
MSL address space in a CUDA build also erases nvcc's own function annotation,
and every device function in scope becomes a host function. Rewriting the text
per target dissolves that collision: no name has to mean two things at once.

LINE NUMBERS ARE EXACT, and the mechanism is that no transform ever adds or
removes a line. Every rewrite is a within-line substitution. The `metal` output
is therefore line-for-line identical in count to the neutral file, which is what
`../metal/shader_compiler.mm` needs: it injects `#line 1 "<path>"` itself and
maps a diagnostic back by counting lines in the segment it spliced. The `cu` and
`cpp` outputs carry a two-line preamble ending in `#line 1 "<neutral file>"`,
so a diagnostic from nvcc or clang names the neutral file at the neutral line.
Columns shift by the length difference of the substituted text, so a caret sits
on the generated column; the generated file is on disk and is the text the
compiler read, so it can be opened.

FAILING LOUDLY. A construct this script does not understand is an error, never
a pass-through. It rejects any preprocessor directive other than `#pragma once`
and a quoted `#include`, any `SM_` seam macro, any `[[...]]` attribute outside
the table above, any CUDA or MSL keyword written directly in a neutral body,
and any quoted include naming a file that does not exist. Substitutions are
applied only outside comments and string literals, so prose naming an attribute
is not rewritten and not validated.

Usage:
    kernelgen.py --target {cu,metal,cpp} --out <file> <source>
"""

import argparse
import functools
import os
import re
import sys

# ---------------------------------------------------------------------------
# The table. Three columns, one per target, and every neutral spelling appears
# in all three so a reader can see what each backend is handed.
#
# `inline` is NOT part of this table. It is written literally in the neutral
# body, because all three compilers spell it the same way, and the attribute
# then carries only the execution space.
# ---------------------------------------------------------------------------

# `hip` is LAST on purpose: ADDRESS_SPACE and EXECUTION_SPACE are read by
# `TARGETS.index(target)`, so appending keeps every existing column where it is
# and a reordered table cannot silently hand one target another's spelling.
TARGETS = ("cu", "metal", "cpp", "hip")

ADDRESS_SPACE = {
    # neutral attribute      cu    metal          cpp   hip
    "thread": ("", "thread", "", ""),
    "device": ("", "device", "", ""),
    "threadgroup": ("", "threadgroup", "", ""),
    "constant": ("", "constant", "", ""),
}

EXECUTION_SPACE = {
    "device_fn": ("__device__", "", "", "__device__"),
    "host_device_fn": ("__host__ __device__", "", "", "__host__ __device__"),
}

# THE ONE AUTHORIZED FORK, AND IT IS FOR LANE COOPERATION ONLY. A body may be
# written twice under one name, a COOPERATIVE form that may use the lane and
# barrier names `seam_host.h` deliberately leaves undefined, and a SERIAL TWIN
# that may not. This script emits the first for the
# device targets and the second for the host, so the neutral file holds both and
# every rendering holds exactly one.
#
# THEY RENDER AS NOTHING, like the execution space above: what they select is
# WHICH LINES survive, and the selection happens in `convert` where the twin the
# target does not want is neutralized to comments, line for line, exactly as an
# entry declaration is. That is what keeps every rendering's line count equal to
# the neutral file's and the Metal `#line` mapping exact.
TWIN_ATTRIBUTES = {
    "cooperative": "the device form; may use the lane and barrier names",
    "serial": "the host twin of a cooperative body; may not",
}
TWIN_TARGETS = {"cooperative": ("cu", "metal", "hip"), "serial": ("cpp",)}

ATTRIBUTES = dict(ADDRESS_SPACE)
ATTRIBUTES.update(EXECUTION_SPACE)
for _twin in TWIN_ATTRIBUTES:
    ATTRIBUTES[_twin] = ("", "", "", "")
# A SELF-DECLARING BODY carries `[[seam::entry(...)]]` on its own signature, and
# there it IS a within-line substitution: the declaration it stands for is
# generated separately, so on a body target the attribute renders as nothing on
# all three. On a declaration the whole line is replaced and this never applies.
ATTRIBUTES["entry"] = ("", "", "", "")

# ---------------------------------------------------------------------------
# The entry-point family.
#
# These four attributes appear only inside an ENTRY DECLARATION, which is a
# function declaration (no body, terminated by `;`) that this script turns into
# a kernel entry point and an argument record. They are NOT in the table above,
# because none of them is a within-line substitution: the whole declaration is
# replaced, by the generated artifact on an entry target and by comment text on
# a body target.
#
#   [[seam::args]]   this declaration's parameter list IS the argument record.
#                   Required. On its own it emits the record and no entry
#                   point, which is what a record shared by an entry and a
#                   host-side filler needs.
#   [[seam::entry]]  emit the kernel entry point as well. Requires
#                   [[seam::args]]: an entry point with no argument record
#                   is not expressible.
#   [[seam::count]]  exactly one parameter carries it. It is a record field, and
#                   it is the bound the in-kernel guard tests the thread index
#                   against. It is NOT forwarded to the neutral body; a body
#                   that also wants the count declares a second, plain
#                   parameter, so what the body receives is never inferred.
#   [[seam::pod(N)]] the pointee of a [[seam::device]] pointer is a struct of N
#                   bytes rather than one of the three scalars. N is asserted
#                   in every C++ rendering, from this one parse.
#   [[seam::index]]  at most one parameter carries it. It is the thread index,
#                   so it is NOT a record field: the launch supplies it. It IS
#                   forwarded, in the position it is declared in, which is how
#                   a body whose index is not its last parameter is served. A
#                   body that takes no index declares none, and the generator
#                   names the index itself; see THE THREAD INDEX below.
#
# THE THREE ELEMENT SHAPES. The four attributes above pass a buffer as a BASE
# POINTER, which serves a body that does its own addressing. Most neutral
# bodies in this tree take an ELEMENT or return one, and the per-element
# addressing then lives in the launcher, which is exactly the hand-written text
# a generated entry exists to remove. These three name that addressing, on the
# parameter it applies to, so it is still written down rather than inferred:
#
#   [[seam::gather]]    the body receives `buffer[index]`, one element, rather
#                      than the base pointer. The record field is unchanged: a
#                      handle names the allocation either way.
#   [[seam::scatter]]   the body's RETURN VALUE is written to `buffer[index]`.
#                      At most one parameter carries it, because one call
#                      returns one value. It is not forwarded unless it also
#                      carries [[seam::gather]], which is the in-place element
#                      update `x[i] = f(x[i], ...)`.
#   [[seam::stride(N)]] the body receives `buffer + N * index`. For a body that
#                      reads a fixed run per thread: a 3x3 block at stride 9,
#                      its vector at stride 3.
#
# THE FOURTH SHAPE, AND WHY IT IS THE ENTRY'S WORK RATHER THAN THE BODY'S.
#
#   [[seam::indices(N)]] a `const unsigned *` whose N slots at the thread index
#                      are the element's own index list. At most one per
#                      entry, and NOT forwarded.
#   [[seam::through]]   the body receives this buffer's elements at those N
#                      slots, as N arguments in this parameter's position.
#   [[seam::bound]]     an `unsigned`: the size of the index space those slots
#                      may name. A record field, not forwarded.
#
# An element reading its data through its own index list, `x[tet[4 * e + k]]`,
# is the shape every per-element physics kernel here takes. It is expressible
# WITHOUT this trio, by passing `x` as a base pointer and `tet` at
# [[seam::stride(N)]] and letting the body subscript both, and that spelling is
# still accepted; what it cannot do is CHECK the index, because the body holds
# no bound and there is nothing in a base pointer to compare against. Metal
# never faults on an out-of-bounds read and returns 0.0 for one, so an
# unchecked indirect gather there is a wrong answer with plausible floats and a
# Completed status, which is the failure this generator exists to convert into
# a loud one. Hence the bound is declared and the entry does the subscripting.
#
# WHY THE BOUND IS DECLARED AND NOT READ OFF THE HANDLE. A handle carries
# `size`, and reading it would need no declaration at all, but its unit is the
# element size the ALLOCATOR was given rather than the pointee this declaration
# states: a coordinate buffer allocated as `3 * vertices` floats and declared
# here as `Vec3f *` would yield a bound three times the index space. That error
# fails OPEN, admitting exactly the indices a check exists to refuse, so the
# number comes from the declaration instead, where it means one thing and the
# caller states it once. It is the same kind of number as [[seam::count]] and is
# trusted exactly as far.
#
# WHAT A FAILING SLOT DOES: it asserts on the target's own channel and the
# element returns without calling the body. That is what the [[seam::count]]
# guard already does for the thread index, and what the hand-written Metal
# kernels this replaces already do for these very slots.
#
# THE THREAD INDEX. An entry always has one, and it reaches the body in one of
# two ways: as a forwarded [[seam::index]] parameter, or through the addressing
# above. A declaration doing neither is refused, because every thread would
# then compute the same thing. When no parameter carries [[seam::index]] the
# generator names the index itself, `seam_index`, which is why no parameter may
# take a `seam_` name.
#
# Everything else about an entry is generated, and that is the point of the
# form: a declaration cannot express a branch, a phase order or a convergence
# test, so an entry point cannot acquire logic without changing this script.
ENTRY_ATTRIBUTES = {
    "args": "marks this declaration's parameter list as an argument record",
    "entry": "emit a kernel entry point for this declaration",
    "group": "launch one GROUP per element instead of one thread per element",
    "count": "the guard bound; a record field, not forwarded to the body",
    "index": "the thread index; forwarded to the body, not a record field",
    "diag": "the diagnostic channel the entry already binds; forwarded to the "
            "body, not a record field",
    "lane": "the thread's position inside its group; a group entry only",
    "width": "the group's thread count; a group entry only",
    "pod": "the pointee is a struct of the given size, not one of the scalars",
    "gather": "the body receives this buffer's element at the thread index",
    "scatter": "the body's return value is written here at the thread index",
    "stride": "the body receives this buffer's base advanced by N * index",
    "scratch": "group-local scratch of N elements, declared by the entry",
    "indices": "this element's N-slot index list; a record field, not "
               "forwarded to the body",
    "through": "the body receives this buffer's elements at the N slots the "
               "index list names",
    "bound": "the size of the index space [[seam::indices]] names; a record "
             "field, not forwarded to the body",
}

# What a parameter IS. At most one, and never beside an access attribute: a
# guard bound, a thread index, a lane and a group width are not buffers.
ROLE_ATTRIBUTES = {"count", "index", "lane", "width", "bound", "diag"}

# THE SECOND LAUNCH SHAPE, and the reason it is a shape rather than a parameter.
#
# An ELEMENT entry is one thread per element: the backend picks the group width
# because that choice computes no value, and the guard tests one flat index. A
# GROUP entry is one group per element, and all three of the group count, the
# group width and the group-local scratch are SEMANTIC. A kernel that cooperates
# over one problem per group indexes by GROUP position, reads its own lane to
# split the work, and synchronizes with `compute::threadgroup_barrier()`; a
# ceil-div over a thread count cannot express any of that. `BeExtent` in the
# backend ABI already names the two kinds for the same reason.
#
# WHAT A GROUP MEANS ON EACH OF THE THREE TARGETS, because the shape is not
# finished until all three say something true. On CUDA it is one block per
# element, the group index `blockIdx.x` and the lane `threadIdx.x`. On Metal it
# is one threadgroup per element, indexed by `threadgroup_position_in_grid` and
# `thread_position_in_threadgroup`. On the HOST the lanes run ONE AFTER ANOTHER,
# lane 0 first, one whole group per step, which is exact for a body that does
# not synchronize: the only channels between lanes of a group are the
# group-local scratch and device memory, and a read of another lane's write is
# ordered by `compute::threadgroup_barrier()` on both device targets and is a
# race without one there too. A body that DOES synchronize has no host
# rendering, and the host COMPILER says so rather than this script: the host
# seam leaves `compute::threadgroup_barrier` undefined, so such a body is
# refused at the neutral file and line, through any depth of included helper.
# What the host does not promise is the fold ORDER through an atomic, which a
# device target reaches concurrently; that is the reassociation every backend
# here already carries against every other.
#
# WHAT IS DELIBERATELY NOT HERE. Dynamic group-local scratch, sized at dispatch.
# `[[seam::scratch(N)]]` is a STATIC array declared inside the entry point,
# which is the safer of the two on the platform that decides it: Metal validates
# a static threadgroup array against the 32768 byte cap at pipeline creation and
# validates the dynamic path not at all, returning wrong data past the cap on a
# backend that never faults. A kernel whose scratch is genuinely sized at
# dispatch is a change to this script, not a use of it.
GROUP_ONLY_ATTRIBUTES = {"lane", "width", "scratch"}

# THE GUARD IS BLOCK-UNIFORM IN A GROUP ENTRY, and that is what makes it safe to
# put before a barrier. `blockIdx.x` / `threadgroup_position_in_grid` is the
# same value for every thread of a group, so the whole group returns or none of
# it does, and a `compute::threadgroup_barrier()` in the body is still reached
# by every thread that is still running. An ELEMENT entry's guard is per thread
# and would not have that property, which is the other half of why a barrier-
# carrying body cannot be launched in the element shape.
GROUP_GUARD_LINES = (
    "One GROUP per element. The guard is on the group index, which is the same",
    "value for every thread of a group, so a group returns whole and a",
    "compute::threadgroup_barrier() in the body is still reached by every",
    "thread of every group that did not. An element entry's guard is per",
    "THREAD and has no such property, which is the other half of why a",
    "barrier-carrying body cannot be launched in the element shape.")


def _guard_note(indent):
    """The group guard's comment, wrapped, at one indent."""
    pad = " " * indent
    return "".join(f"{pad}// {line}\n" for line in GROUP_GUARD_LINES)

# How a [[seam::device]] buffer is REACHED. `gather` and `scatter` combine, and
# that pair is the in-place element update; every other pairing is refused.
ACCESS_ATTRIBUTES = {"gather", "scatter", "stride", "indices", "through"}

# The two attributes that carry an argument, each because the argument is the
# whole point. A struct pointee's size cannot be computed from this file, so
# the declaration states it and every C++ rendering asserts it; a stride is the
# run length one thread reads and there is nothing else to derive it from.
ATTRIBUTES_WITH_ARGUMENT = {"pod", "stride", "scratch", "indices"}
# `[[seam::entry]]` is the one attribute whose argument is OPTIONAL: with it,
# the argument names the parameter carrying the thread count, the way
# `DISPATCH_START(count)` names it in the reference; without it, a parameter
# carries `[[seam::count]]` instead. Both spellings say the same thing, and a
# declaration using both is refused.
ATTRIBUTES_WITH_OPTIONAL_ARGUMENT = frozenset(("entry",))


# What each of those arguments IS, quoted back at an author who left it out.
# Three different quantities in three different units, so one word for all
# three ("the size") names the wrong thing for two of them.
ATTRIBUTE_ARGUMENT_IS = {
    "pod": "the pointee struct's size in BYTES",
    "stride": "the run one thread reads, in ELEMENTS",
    "scratch": "the group-local array's length, in ELEMENTS",
    "indices": "the element's SLOT COUNT: how many indices it names",
}

ALL_ATTRIBUTES = dict(ATTRIBUTES)
ALL_ATTRIBUTES.update(ENTRY_ATTRIBUTES)

# ---------------------------------------------------------------------------
# What a generated argument record may hold.
#
# Deliberately narrow, and every exclusion below is a measured hazard rather
# than a style preference:
#
#   * `double` is refused because MSL has no `double` at all, so a record
#     carrying one has no Metal rendering.
#   * a vector or matrix type is refused because MSL `float3` is 16 bytes
#     against `Vec3f` at 12 and `float3x3` is 48 against `Mat3x3f` at 36, so a
#     record carrying one has a different layout per target while still
#     compiling everywhere.
#   * anything narrower than 4 bytes is refused because it introduces padding,
#     and padding is where a later field hides: a field that fits in EXISTING
#     padding moves no size, which is the residual mirror-drift hole a `sizeof`
#     assert cannot see. With every field 4-byte aligned and 4-byte or 16-byte
#     wide, a record has no padding at all and its size is the sum of its
#     fields, so the size assert is total.
#
# A pointer parameter is not a field type at all: it becomes an ArenaHandle,
# because Metal has 31 compiler-enforced buffer binding slots and a record
# carrying device addresses could not be bound.
SCALAR_TYPES = {
    # neutral      cu          metal   cpp         rust   size
    "float":    ("float",    "float", "float",    "f32", 4),
    "int":      ("int",      "int",   "int",      "i32", 4),
    "unsigned": ("unsigned", "uint",  "unsigned", "u32", 4),
}

# Pointee types a [[seam::device]] parameter may address. The same table as
# above minus the sizes: what matters is that the generated resolution casts to
# a type all four renderings spell.
POINTEE_TYPES = SCALAR_TYPES

# What GROUP-LOCAL SCRATCH may be an array of: the three scalars, and the two
# ACCUMULATOR SLOTS beside them.
#
# THE SLOTS ARE HERE BECAUSE THEIR SIZE IS KNOWN, which is the whole of the rule
# the message above states. `compute::atomic_uint_t` and `compute::atomic_float_t`
# hold the same four bytes their plain counterparts do, which every record in
# this tree already asserts by spelling them `[[seam::pod(4)]]`; what makes them
# their own type is HOW a kernel reaches the storage, not how much of it there
# is. A histogram shared by a group's lanes is the case: the lanes increment the
# same slot at once, so the array has to be reachable atomically, and refusing it
# here would push the kernel to a private count per lane and a second reduction
# to combine them.
SCRATCH_TYPES = dict(SCALAR_TYPES)
SCRATCH_TYPES["compute::atomic_uint_t"] = (
    "compute::atomic_uint_t", "compute::atomic_uint_t",
    "compute::atomic_uint_t", "u32", 4)
SCRATCH_TYPES["compute::atomic_float_t"] = (
    "compute::atomic_float_t", "compute::atomic_float_t",
    "compute::atomic_float_t", "f32", 4)

# The namespaces the seam's own vocabulary lives in.
# `crates/ppf-cts-solver/src/kernels/backend/contract.md` states every name in
# them, each backend prologue defines all of them, and a neutral body already
# calls them by exactly these spellings, so the three compilers do read one
# text. `compute::` is what
# differs per backend and `fmath::` and `bits::` are what does not, which is the
# rule that decides which namespace a new name belongs in.
SEAM_NAMESPACES = ("compute", "fmath", "bits")
SEAM_QUALIFIED_RE = re.compile(
    r"(?:(?:" + "|".join(SEAM_NAMESPACES) + r")::)?[A-Za-z_][A-Za-z0-9_]*")

HANDLE_SIZE = 16
HANDLE_ALIGN = 4

# The STRONGEST alignment an arena will serve a pointee, and it is the HOST
# arena's that is binding: `ARENA_ALIGN` in `ppf-cts-compute/cpu/host.rs` is 64
# and `allocate_span` refuses a larger request outright, while the CUDA arena
# takes the alignment as an argument with no upper bound of its own. So 64 is
# the tightest of the three and is what a pointee is held to here.
ARENA_MAX_ALIGN = 64

# THE ALIGNMENT EACH POINTEE ASKS FOR, which a host-side handle validator needs
# and which Python cannot compute, since `alignof` is the C++ compiler's to
# answer.
#
# IT DEFAULTS TO 4 AND THE GENERATED C++ CHECKS THE GUESS. Every handle field
# emits `static_assert(alignof(T) == <the value below>)`, so a type whose real
# alignment differs fails the BUILD naming the instantiation, exactly as
# `resolve<T> [with T=AABB]` did when this was first attempted as a blanket
# bound. That is what makes a defaulted table safe: it is never silently wrong,
# nothing has to enumerate the record types up front, and a future `alignas`
# stops the build rather than quietly weakening a check.
POINTEE_ALIGN_DEFAULT = 4
POINTEE_ALIGN = {
    # 32 so a random tree load is one 32-byte sector rather than straddling two
    # (`data_records.hpp:405`).
    "AABB": 32,
}


def pointee_align(base):
    """The alignment `base` asks for, which the generated C++ then asserts."""
    return POINTEE_ALIGN.get(base, POINTEE_ALIGN_DEFAULT)

# The thread index's name when the declaration does not give it one, which is
# the case for a body that takes no index and reads its element through
# [[seam::gather]] or [[seam::stride]].
GENERATED_INDEX = "seam_index"

# THE PREFIX OF THE MSL RENDERING'S THREAD-SPACE COPIES, and why it is not the
# bare `seam_` it uses. Metal is the one target that copies a gathered
# element into `thread` before handing it to the body, and the copy is named
# for its field, so a field named `index` produced `seam_index`, which is
# GENERATED_INDEX above: the entry point then declared its thread index and a
# `Vec6u` under one name and did not compile. `utility/stitch_scatter` has such
# a field and nothing had ever compiled its MSL rendering. Prefixing takes the
# copies out of the namespace the generator's own locals live in.
THREAD_LOCAL = "seam_local_"

# Identifiers the generated entry points already use, in one rendering or
# another: the record parameter, the CPU shim's half-open range, the Metal
# diagnostic handle. A parameter taking one of these names would be shadowed by
# it or would shadow it, and every generated buffer local is named for its
# field, so the collision is silent in exactly the place this script exists to
# make loud. The `seam_` PREFIX is reserved for the same reason and covers the
# generator's own locals (`seam_index`, `seam_arena_count`, `seam_call`,
# `seam_last`, `seam_arena_base`, `seam_arenas_valid`, `seam_group_width`,
# `seam_slot`, `seam_slots_valid`, `seam_k`, and the MSL rendering's
# `seam_<field>` thread-space copies).
#
# IT IS A PREFIX RATHER THAN A LIST BECAUSE ONE OF THEM IS DERIVED. The MSL
# thread-space copies are named for the FIELD they copy, so the set is not known
# until a declaration is read, and only a reserved prefix can cover a name that
# does not exist yet. `seam_` says which layer owns them, which a project-wide
# prefix did not.
RESERVED_PARAMETER_NAMES = {"args", "begin", "end", "diag"}

# The one name a [[seam::diag]] parameter may take, and it is the name the entry
# already binds on every target. Requiring it rather than allowing any spelling
# is what keeps the MSL rendering from carrying two handles to one channel: that
# target binds `diag` for its own arena and slot checks, and a body asking for
# the channel gets the same local rather than a second one. It is in
# RESERVED_PARAMETER_NAMES above for every other role, which is why the check
# there admits exactly this pairing.
DIAG_TYPE_NAME = "DiagHandle"
DIAG_PARAMETER_NAME = "diag"

# The most slots one element may name. A bound rather than a judgement: the
# generated prologue reads every slot into a local array before the body is
# called, so the arity is thread storage on every target, and an arity large
# enough to matter is a kernel whose element does not fit the shape. The widest
# in this tree is the cross-stitch's six.
MAX_INDEX_SLOTS = 16

# What a [[seam::device]] pointer parameter may ask for, quoted back at an
# author whose declaration asks for something else. A launcher fitting none of
# these shapes is work for the BODY: it takes the base pointer and the index,
# which changes its CUDA and Metal call sites in the same edit, and that is a
# deliberate change rather than a generator feature.
ELEMENT_SHAPES = (
    "A [[seam::device]] pointer reaches its data in one of five shapes: the "
    "BASE pointer (no access attribute), one ELEMENT at the thread index "
    "([[seam::gather]]), the base advanced by a fixed run "
    "([[seam::stride(N)]]), the element's own index list "
    "([[seam::indices(N)]]), or that list's elements of another buffer "
    "([[seam::through]]). A body's RETURN VALUE reaches memory in one shape: "
    "written at the thread index to the [[seam::scatter]] buffer. A branch, a "
    "second call or a reduction is none of these and belongs in the neutral "
    "body")

# The portable per-argument size cap. Metal's device cap is 32752 B and 32756 B
# kills the process with SIGABRT and nothing on stdout or stderr, so the cap
# cannot be discovered at run time; 4096 B is the portable contract the Metal
# backend already names (../metal/metal_context.mm) and it also keeps a record
# inside CUDA's kernel parameter space. Refused at generation time, which is
# the only place the failure can be made loud.
MAX_ARGS_BYTES = 4096

# The static group-local scratch cap. Metal's threadgroup memory limit is 32768
# bytes and the shader compiler enforces it against a STATIC array at pipeline
# creation, which is the whole reason [[seam::scratch(N)]] renders a static
# array rather than a dynamically sized one: the dynamic path validates nothing
# there and returns wrong data past the cap on a backend that never faults.
MAX_SCRATCH_BYTES = 32768

# What the Rust twin calls a buffer field.
#
# THE SEAM'S BUFFER REFERENCE IS AN (arena, offset) HANDLE, which is what every
# C++ rendering above spells (`ArenaHandle`), and the Rust twin will spell it
# too. It does not yet, because the driver's buffers are Rust-owned `Vec`s and
# the host's own `DataSet` arrays rather than device allocations, so what the
# driver has to put in a record today is an address: `HostRef`. The two types
# are both 16 bytes at 4-byte alignment, so the layout assertions below are the
# same literals either way and the migration moves no field.
#
# THE MIGRATION IS FIELD BY FIELD, and it has to be. A record names several
# buffers and they do not all move together: the elastic scratch a stage passes
# between its own kernels can be a device allocation while the scene arrays and
# the Newton iterate beside it in the same record are still the host's own.
# `--handle-field` names the ones that have moved, so a record can be half
# migrated and still dispatch; the target's binding lays its own arena bases in
# the low slots and binds each remaining address above them.
#
# The list lives in the BUILD and never in a neutral kernel source: a body says
# nothing about the host's buffer provenance, and a migration state written into
# one would be a kernel taking a branch on its caller. When the list covers
# every field of every record, `HostRef` has no user left and this constant goes
# with it.
RUST_BUFFER_TYPE = "HostRef"
RUST_HANDLE_TYPE = "Handle"

# Names MSL takes that read as ordinary identifiers in CUDA and in host C++, so
# a record field or an entry name spelled this way compiles on two backends and
# fails on the third with a diagnostic pointing somewhere else. The address
# spaces, the shader-stage qualifiers and `half` are already refused for every
# identifier in a neutral body; the vector and matrix type names are added here
# because a FIELD name is the place they plausibly appear.
MSL_RESERVED_NAMES = set()
for _base in ("char", "uchar", "short", "ushort", "int", "uint", "long",
              "ulong", "float", "half", "bool"):
    MSL_RESERVED_NAMES.add(_base)
    for _n in (2, 3, 4):
        MSL_RESERVED_NAMES.add(f"{_base}{_n}")
        for _m in (2, 3, 4):
            MSL_RESERVED_NAMES.add(f"{_base}{_n}x{_m}")
MSL_RESERVED_NAMES.update({"kernel", "vertex", "fragment", "thread", "device",
                           "threadgroup", "constant", "ray_data",
                           "threadgroup_imageblock", "object_data"})


# Spellings that belong to one backend and must never appear in a neutral body.
# The first four are what this script exists to supply; `kernel`, `vertex`,
# `fragment` and `half` are MSL keywords that read as ordinary identifiers in
# CUDA and in host C++, so a body using one of them as a name compiles on two
# backends and fails on the third with a diagnostic that points somewhere else.
FORBIDDEN_IDENTIFIERS = {
    "thread": "address space keyword; write [[seam::thread]] instead",
    "device": "address space keyword; write [[seam::device]] instead",
    "threadgroup": "address space keyword; write [[seam::threadgroup]] instead",
    "constant": "address space keyword; write [[seam::constant]] instead",
    "__device__": "CUDA execution space; write [[seam::device_fn]] instead",
    "__host__": "CUDA execution space; write [[seam::host_device_fn]] instead",
    "__global__": "a neutral body holds no entry point, only callable bodies",
    "__shared__": "CUDA storage class; threadgroup memory is passed in as a "
                  "[[seam::threadgroup]] pointer parameter",
    "__constant__": "CUDA storage class, and a silent wrong-answer trap: it "
                    "reads as zero on the host",
    "kernel": "MSL shader-stage qualifier",
    "vertex": "MSL shader-stage qualifier; spell the index `vert`",
    "fragment": "MSL shader-stage qualifier",
    "half": "MSL 16-bit float type; name the quantity instead",
    "BOUNDED_LIMIT": "float_math.hpp's own constant, declared by the CUDA and "
                     "host arm of namespace fmath and by no other; the Metal "
                     "arm takes the library's accurate paths and needs no "
                     "bound. Assert the caller's own bound instead",
}

SEAM_MACRO_RE = re.compile(r"\bSM_[A-Z][A-Z0-9_]*\b")
ATTRIBUTE_RE = re.compile(r"\[\[\s*([A-Za-z_][A-Za-z0-9_:]*)\s*(\([^)]*\))?\s*\]\]")
IDENT_ONLY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# The attribute namespace, spelled once. A neutral body declares what it needs
# of a target through `[[seam::...]]`, which is a declaration language this
# generator reads rather than an API anything calls, and this tree already calls
# that boundary the seam: src/kernels/seam/, seam.hpp, seam_host.h,
# seam_cuda.cuh, ppf-cts-compute/seam/.
#
# A C++ ATTRIBUTE NAMESPACE IS IGNORED WITHOUT COMPLAINT by nvcc, by the Metal
# shader compiler and by host clang alike, each of which was measured answering
# an unknown one with a warning and a zero exit. So a spelling one of them stops
# ignoring, or starts ignoring differently, changes what this generator emits
# without failing any build, and the spelling is kept in one place for that
# reason rather than for tidiness.
ATTRIBUTE_NAMESPACE = "seam"
ATTRIBUTE_PREFIX = ATTRIBUTE_NAMESPACE + "::"
# `[[seam::entry]]` MAY CARRY THE COUNT PARAMETER'S NAME, so the opener admits
# an argument. A pattern demanding a bare `]]` opens no span for such a
# declaration, and every attribute inside it is then reported as sitting outside
# an entry declaration, which names the symptom and not the cause.
ENTRY_DECL_RE = re.compile(
    r"\[\[\s*" + ATTRIBUTE_NAMESPACE + r"::(args|entry)\s*(?:\([^)]*\))?\s*\]\]")
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
INCLUDE_RE = re.compile(r'^(\s*#\s*include\s+)"([^"]*)"(.*)$')
DIRECTIVE_RE = re.compile(r"^\s*#\s*(\w*)(.*)$")

# The marker ../metal/shader_compiler.mm writes over a quoted include as it
# splices a segment. Emitting the identical text here means the two paths agree
# and the assembler's own pass is a no-op over a generated segment.
METAL_INCLUDE_MARKER = "// [metal] host-resolved include: "

KERNEL_SUFFIX = ".kernel.cpp"
GENERATED_SUFFIX = {"cu": ".kernel.cu", "metal": ".kernel.metal",
                    "cpp": ".kernel.cpp", "hip": ".kernel.hip"}

# `--emit entry` renders the entry points and argument records declared in the
# file, and it has a fourth target the body form does not: the Rust twin the
# driver holds. There is no Rust rendering of a kernel BODY, so `rust` is
# rejected for `--emit body` rather than silently producing nothing.
EMIT_KINDS = ("body", "entry", "args", "table", "diagfile", "externs", "thunks")
ENTRY_TARGETS = ("cu", "metal", "cpp", "rust", "hip")
ENTRY_SUFFIX = {"cu": ".entry.cu", "metal": ".entry.metal",
               "hip": ".entry.hip",
                "cpp": ".entry.cpp", "rust": ".entry.rs"}

# `--emit args` renders the half of an entry rendering that a SECOND
# translation unit may read: the argument records, their layout assertions, the
# pointee size assertions and the launcher declarations. It exists for exactly
# one target.
#
# WHY ONLY `cu`. An entry rendering carries definitions (the `__global__` and
# the C-linkage launcher), so exactly one translation unit may include it. That
# is the whole file on every other target, because every other target has one
# consumer: the Metal rendering is spliced into one assembled shader, the `cpp`
# rendering is compiled into one entry translation unit, and the Rust rendering
# is one module. Only the CUDA rendering is included by a HEADER that several
# translation units read, and a caller that fills a record needs the record's
# TYPE, which until now it could not have without also acquiring the two
# definitions and a duplicate-symbol link error.
#
# THE HALVES ARE NOT A CHOICE THE CALLER MAKES. `--emit args` is a strict
# prefix of what `--emit entry` emits, and the entry rendering INCLUDES
# it rather than repeating it, so the record a filler sees and the record the
# entry point is compiled against are one definition and cannot drift. A
# generated launcher DECLARATION sits in the args header for the same reason a
# hand-written one would be wrong: it is the same text the definition is
# rendered from.
# The two backends that hash `__FILE__` into a device record and therefore need
# a table to reverse it. The host backend stores the string itself and Metal
# registers a path per segment as it assembles the shader.
DIAGFILE_TARGETS = ("cu", "hip")

ARGS_TARGETS = ("cu", "metal", "hip")
ARGS_SUFFIX = {"cu": ".args.cuh", "metal": ".args.h", "hip": ".args.hiph"}

# WHAT SEPARATES THE TWO DEVICE-C++ TARGETS, AND IT IS THIS TABLE AND NOTHING
# ELSE.
#
# CUDA and HIP spell `__global__`, `__device__`, `__shared__`, `blockIdx`,
# `threadIdx`, `blockDim`, `extern "C"` and the triple-chevron launch
# identically, so ONE renderer serves both and the whole difference is the four
# strings below. They are a TABLE rather than a branch inside the renderers for
# the same reason a kernel is written once: two renderers that start out alike
# drift, and nothing reads the drift. A further device-C++ target is
# a row here and no new renderer.
DEVICE_CXX = {
    "cu": {
        "stream_type": "cudaStream_t",
        "last_error": "cudaGetLastError",
        "error_macro": "CUDA_HANDLE_ERROR",
        "utils_include": "cuda_utils.hpp",
        "language": "CUDA",
    },
    "hip": {
        "stream_type": "hipStream_t",
        "last_error": "hipGetLastError",
        "error_macro": "HIP_HANDLE_ERROR",
        "utils_include": "hip_utils.hpp",
        "language": "HIP",
    },
}

# `--emit table` renders one FRAGMENT of the kernel table: the rows for the
# entry points this file declares, and nothing else.
#
# WHY A FRAGMENT AND NOT A TABLE. A kernel id is a dense index over EVERY entry
# point a backend library carries, so no single neutral source can know one:
# this script sees one file per invocation, by design, because that is what
# makes a rendering a pure function of its source and lets a build rule run
# over the tree without a hand-kept list. The id is therefore assigned by the
# POSITION of a row in the concatenation, and the concatenation is the caller's
# to make.
#
# THAT THE TWO CONCATENATIONS AGREE IS NOT ASSUMED. The library's table and the
# driver's are concatenated by two different programs (a Makefile and a build
# script), both over the same sorted source order, and `be_open` compares
# them id by id, by NAME and by record size, before any dispatch. A
# disagreement is a named refusal at open rather than a dispatch of the wrong
# kernel with the right bytes.
#
# TWO TARGETS, and they are the two SIDES of one boundary rather than two
# languages that happen to want a table. `cu` is the library's half, which
# holds the launcher's address. `rust` is the driver's half, which holds what
# the seam's `KernelDecl` needs. There is no `metal` or `cpp` row here because
# neither has a caller yet: the Metal recipe renders no entry file at all
# today, and the host target reaches its entry points by symbol from a
# hand-written launch table rather than by id.
TABLE_TARGETS = ("cu", "metal", "rust", "hip")
TABLE_SUFFIX = {"cu": ".table.inc", "metal": ".table.h",
                "rust": ".table.rs"}


class KernelGenError(Exception):
    """A construct this script does not understand, reported with a position."""


def fail(path, lineno, col, message):
    raise KernelGenError(f"{path}:{lineno}:{col}: {message}")


# ---------------------------------------------------------------------------
# Lexical classification.
#
# Every substitution below is applied to code only. The mask says, character by
# character, whether a position is code ('c') or the inside of a comment or a
# literal ('x'), and a substitution is applied to the original text at positions
# the mask calls code. Nothing is rewritten inside a comment, so a body may name
# an attribute in prose, and nothing inside a string literal is rewritten
# either.
# ---------------------------------------------------------------------------

def is_digit_separator(text, i):
    """True when the apostrophe at `i` separates digits inside a number.

    The token it sits in is scanned backwards over the characters a numeric
    constant can hold. It is a separator only when that run STARTS with a
    digit, which distinguishes `1'000` and `0x1F'FF` from the encoding prefix
    of a character literal (`L'a'`, `u8'x'`), where the run starts with a
    letter.
    """
    j = i - 1
    while j >= 0 and (text[j].isalnum() or text[j] in "._"):
        j -= 1
    run = text[j + 1:i]
    return bool(run) and run[0].isdigit()


def classify(text, path):
    """Returns a mask of the same length as `text`.

    A newline is masked as itself, so splitting the mask on '\\n' yields exactly
    one mask per line of the source. Every other position is 'c' for code or 'x'
    for the inside of a comment or a literal.
    """
    mask = ["c"] * len(text)
    i = 0
    n = len(text)
    lineno = 1
    line_start = 0

    def mark(begin, end):
        # A newline is masked as ITSELF even inside a comment, because the
        # caller splits the mask on '\n' to pair one mask with one line. Only
        # the top-level loop below reaches a newline that ends a line of code;
        # the newlines inside a multi-line block comment are consumed here, so
        # writing 'x' over them (or leaving them 'c') would make the mask hold
        # fewer lines than the source and every line below the comment would
        # pair with the wrong mask.
        for k in range(begin, end):
            mask[k] = "\n" if text[k] == "\n" else "x"

    while i < n:
        c = text[i]
        if c == "\n":
            mask[i] = "\n"
            i += 1
            lineno += 1
            line_start = i
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            end = text.find("\n", i)
            end = n if end == -1 else end
            # A backslash at the end of a line comment splices the NEXT line
            # into the comment, before comments are recognized. This classifier
            # ends the comment at the newline, so it would read that next line
            # as code while all three compilers read it as comment: the line
            # would be validated and rewritten, and its body would exist in no
            # backend. Refused rather than approximated.
            if text[i:end].rstrip("\r").endswith("\\"):
                fail(path, lineno, i - line_start + 1,
                     "a line comment ending in a backslash splices the next "
                     "line into the comment. Close the comment on its own line, "
                     "or use a block comment")
            mark(i, end)
            i = end
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            start_line = lineno
            end = text.find("*/", i + 2)
            if end == -1:
                fail(path, start_line, i - line_start + 1,
                     "block comment is never closed")
            end += 2
            mark(i, end)
            lineno += text.count("\n", i, end)
            last_nl = text.rfind("\n", i, end)
            if last_nl != -1:
                line_start = last_nl + 1
            i = end
            continue
        if c == "R" and i + 1 < n and text[i + 1] == '"':
            fail(path, lineno, i - line_start + 1,
                 "raw string literal: this script classifies comments and "
                 "literals lexically and does not implement the raw form")
        if c == "'" and is_digit_separator(text, i):
            # A C++ digit separator (1'000, 0x1F'FF) is not a character literal.
            # Read as one it opens a literal that runs to the NEXT apostrophe,
            # masking an arbitrary span of code, and a masked span is exempt
            # from every check below: an SM_ macro or a bare `device` keyword
            # between two separators reaches the generated backend source with
            # a zero exit. An even number of separators on a line therefore
            # defeats the validation silently, which is why this is refused at
            # the lexer rather than diagnosed later.
            fail(path, lineno, i - line_start + 1,
                 "digit separator in a numeric constant. This script "
                 "classifies literals lexically and reads the apostrophe as a "
                 "character literal, so it cannot be carried through; spell "
                 "the constant without it")
        if c == '"' or c == "'":
            quote = c
            start_col = i - line_start + 1
            j = i + 1
            closed = False
            while j < n and text[j] != "\n":
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == quote:
                    closed = True
                    j += 1
                    break
                j += 1
            if not closed:
                kind = "string" if quote == '"' else "character"
                fail(path, lineno, start_col,
                     f"unterminated {kind} literal. A digit separator (1'000) "
                     f"reads as one here; spell the constant without it")
            mark(i, j)
            i = j
            continue
        i += 1
    return "".join(mask)


def code_only(line, mask):
    """`line` with every comment and literal character blanked to a space, so a
    regex sees code positions at their real columns and nothing else."""
    return "".join(ch if m == "c" else " " for ch, m in zip(line, mask))


# ---------------------------------------------------------------------------
# Entry declarations
#
# An entry point is an argument record, plus a thread index, plus a call into
# the neutral body. All three are mechanical, so writing one by hand is writing
# a fork by hand, and every backend that hand-writes them accumulates a mirror
# pair per kernel that nothing links to its twin. This section reads ONE
# declaration and renders four of them.
#
# THE PARSE IS THE VALIDATOR, AND IT IS ALSO WHAT MAKES A MISPARSE LOUD. Two
# separate mechanisms, and the second is the one that matters:
#
#   1. Every artifact comes from ONE Field list: the C++, MSL and Rust struct
#      text, the resolution expressions, the forwarded call, and every
#      `sizeof` / `offsetof` / `size_of` / `offset_of` assertion. So the
#      assertions cannot disagree with the structs they are asserted against by
#      construction. What they DO catch is the thing no single parse can:
#      disagreement between the four LANGUAGES, since all four assert the same
#      literal offsets and the same literal size.
#
#   2. A misparse of a TYPE cannot survive the C++ compiler, because the
#      generated entry FORWARDS the parsed parameters to the neutral body,
#      whose real signature the same compiler reads. Read `float a` as a
#      pointer and the entry emits a resolution expression where the body wants
#      a float; read a pointer as a scalar and it emits a float where the body
#      wants `const float *`. Either way the call does not compile. That is why
#      the generated entry calls the body rather than reimplementing it, and it
#      is the answer to the one hazard this feature introduces: `kernelgen.py`
#      was a lexical converter whose failure mode was loud, and a parser that
#      mis-reads a parameter type would otherwise emit a wrong layout, which on
#      a backend that never faults is a silent wrong answer.
#
# The four lexical traps the converter already measured are handled once, for
# both paths, because this parser reads the MASKED code `classify` produced
# rather than the raw text: a digit separator and an unterminated literal abort
# in the lexer, a line comment ending in a backslash aborts there too, a block
# comment inside a parameter list is blanked with its newlines carried, and an
# attribute broken across two lines is refused by the '[[' check in
# `check_line`. No trap has a second implementation here to get wrong.
# ---------------------------------------------------------------------------

# One token of a parameter declaration. The `\S` alternative is what makes `*`,
# `&` and every stray punctuation character its own token, so a construct the
# grammar below does not model shows up as an unexpected token rather than
# being absorbed into an identifier.
PARAM_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\S")


class Field:
    """One argument-record field, with the layout this script computed for it.

    `offset` and `size` are what every assertion in every rendering is written
    against, so a compiler that lays the record out differently fails to
    compile rather than running.
    """

    def __init__(self, name, kind, base, is_const, offset, size, lineno, col,
                 pod_bytes=None, access="base", stride=None, is_sink=False):
        self.name = name
        self.kind = kind          # "handle" or "scalar"
        self.base = base          # neutral scalar name, or the pointee's
        self.is_const = is_const
        self.offset = offset
        self.size = size
        self.lineno = lineno
        self.col = col
        # For a handle whose pointee is a struct rather than one of the three
        # scalars: the size the declaration states, asserted in every C++
        # rendering. None for a scalar field and for a scalar pointee, whose
        # size the table above already knows.
        self.pod_bytes = pod_bytes
        # How the body reaches this buffer, and it changes the CALL only: the
        # record field is a handle in all three cases, because a handle names
        # an allocation and the element arithmetic is the entry point's.
        # "base", "element", "offset", "indices" or "indirect"
        self.access = access
        self.stride = stride      # the N of "offset", else None
        # Whether the body's return value is written here at the thread index.
        self.is_sink = is_sink


class Entry:
    """One parsed entry declaration."""

    def __init__(self, name, fields, forward, count_field, index_name,
                 emit_entry, first_line, last_line, is_group=False,
                 lane_name=None, width_name=None, scratch=(),
                 indices_field=None, bound_field=None, diag_name=None):
        self.name = name                  # the neutral body this wraps
        self.fields = fields              # record fields, in declaration order
        self.forward = forward            # what the generated call passes
        # The name the body takes the diagnostic channel under, or None. Every
        # target binds one; this says whether the BODY asked for it.
        self.diag_name = diag_name
        self.count_field = count_field    # the guard bound
        # The element's own index list and the size of the space its slots may
        # name. Both None unless the declaration carries [[seam::indices(N)]],
        # and both present when it does: the arity is on the list and the
        # bound is what every slot is checked against before it subscripts a
        # [[seam::through]] buffer.
        self.indices_field = indices_field
        self.bound_field = bound_field
        # The thread index's name. The declared [[seam::index]] parameter when
        # there is one, and the generator's own otherwise: an entry always has
        # a thread index, and whether the BODY takes one is a separate
        # question the declaration answers by carrying that attribute or not.
        self.index_name = index_name or GENERATED_INDEX
        self.emit_entry = emit_entry      # False for [[seam::args]] alone
        self.self_declared = False        # True when the BODY declares it
        self.attribute_line = None        # where its attribute sits
        self.first_line = first_line
        self.last_line = last_line
        # THE LAUNCH SHAPE. False is one thread per element and the index above
        # is that thread's; True is one GROUP per element and the index is the
        # group's, because that is what the guard tests and what one call
        # computes. See GROUP_ONLY_ATTRIBUTES.
        self.is_group = is_group
        self.lane_name = lane_name        # thread position inside its group
        self.width_name = width_name      # threads per group, if the body asks
        # (name, pointee, element count) per [[seam::scratch(N)]] parameter.
        # Group-local storage is a launch property, not something a driver
        # fills, so it is NOT a record field: the entry declares the array and
        # passes it.
        self.scratch = list(scratch)

    @property
    def has_handles(self):
        return any(f.kind == "handle" for f in self.fields)

    @property
    def handles(self):
        return [f for f in self.fields if f.kind == "handle"]

    @property
    def sink(self):
        """The field the body's return value is written to, or None."""
        for f in self.fields:
            if f.is_sink:
                return f
        return None

    @property
    def size(self):
        return sum(f.size for f in self.fields)

    @property
    def stem(self):
        return self.name

    @property
    def camel(self):
        return "".join(part.capitalize() for part in self.stem.split("_"))

    @property
    def record_cpp(self):
        # THE SAME NAME AS THE RUST TWIN, which is what a mirror should look
        # like: the two are one declaration rendered twice, and they live in
        # different languages and different files, so nothing can confuse them.
        return self.camel + "Args"

    @property
    def record_rust(self):
        return self.camel + "Args"

    @property
    def entry_symbol(self):
        return self.name + "_entry"

    @property
    def launch_symbol(self):
        """The host-callable launcher's symbol, on the targets that need one.

        A target whose entry point is only reachable through a launch syntax
        needs a second symbol beside it, and the name is derived here so the
        table that holds it and the file that defines it cannot spell it two
        ways.
        """
        return self.entry_symbol + "_launch"


def find_entry_spans(path, lines, masks):
    """Line ranges (1-based, inclusive) of every entry declaration.

    A span opens on the line carrying `[[seam::args]]` or `[[seam::entry]]` and
    closes on the `;` at parenthesis depth zero. A `{` before that `;` is
    refused: an entry declaration has no body, which is what makes a branch,
    a phase order or a convergence test unrepresentable inside one.
    """
    spans = []
    start = None
    depth = 0
    for lineno, (line, line_mask) in enumerate(zip(lines, masks), start=1):
        code = code_only(line, line_mask)
        if start is None:
            m = ENTRY_DECL_RE.search(code)
            if not m:
                continue
            # A SELF-DECLARING BODY carries `[[seam::entry(...)]]` on its own
            # signature, which is a DEFINITION and not a declaration. Reading it
            # as one refuses it for having a body, which names the brace rather
            # than the shape; `find_body_entry_spans` handles those.
            if BODY_DECL_RE.search(code):
                continue
            # A SELF-DECLARING BODY writes the attribute on its OWN line, above
            # a `[[seam::device_fn]]` definition. Reading that as a declaration
            # refuses it for having a body, which names the brace rather than
            # the shape; `find_body_entry_spans` handles those.
            ahead = lineno
            while (ahead < len(lines)
                   and not code_only(lines[ahead], masks[ahead]).strip()):
                ahead += 1
            if (ahead < len(lines)
                    and not code.replace(m.group(0), "").strip()
                    and BODY_DECL_RE.search(
                        code_only(lines[ahead], masks[ahead]))):
                continue
            start = lineno
            depth = 0
        for ch in code:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "{" and depth == 0:
                fail(path, lineno, 1,
                     "an entry declaration has no body. Write it as a "
                     "declaration terminated by ';': the record, the guard, "
                     "the arena resolution and the call into the neutral body "
                     "are all generated, and a body is where logic would enter")
            elif ch == ";" and depth == 0:
                spans.append((start, lineno))
                start = None
                break
        if start is not None and depth < 0:
            fail(path, lineno, 1,
                 "unbalanced ')' in an entry declaration")
    if start is not None:
        fail(path, start, 1,
             "entry declaration is never terminated by ';'")
    return spans


def _flatten(lines, masks, first, last):
    """The span's CODE as one string, with a (line, column) per character."""
    text = []
    locs = []
    for lineno in range(first, last + 1):
        code = code_only(lines[lineno - 1], masks[lineno - 1])
        for col, ch in enumerate(code, start=1):
            text.append(ch)
            locs.append((lineno, col))
        text.append("\n")
        locs.append((lineno, len(code) + 1))
    return "".join(text), locs


def _at(path, locs, index, message):
    lineno, col = locs[min(index, len(locs) - 1)]
    fail(path, lineno, col, message)


def _check_msl_name(path, locs, index, name, what):
    if name in MSL_RESERVED_NAMES:
        _at(path, locs, index,
            f"{what} '{name}' is a name MSL takes. It reads as an ordinary "
            f"identifier in CUDA and in host C++, so a record spelled this way "
            f"compiles on two backends and fails on the third with a "
            f"diagnostic pointing somewhere else")


BODY_PARAM_NAME_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*$")
ATTRIBUTE_GROUP_RE = re.compile(r"\[\[[^\]]*\]\]")


def body_return_type(lines, masks, name):
    """The neutral body's return type, or None when the body is not in file.

    A body that RETURNS a value has somewhere for that value to go, and a body
    returning void does not. That is what decides whether an entry has a sink at
    all, so it is read from the body rather than written beside the buffer.
    """
    for found, first, last in body_signature_spans(lines, masks):
        if found != name:
            continue
        blob = " ".join(code_only(lines[i], masks[i])
                        for i in range(first - 1, last))
        head = ATTRIBUTE_GROUP_RE.sub(" ", blob[:blob.index(name)])
        tokens = [t for t in head.replace("inline", " ").split() if t]
        return tokens[-1] if tokens else None
    return None


def body_parameter_shapes(lines, masks, name):
    """"element" or "pointer" for each parameter of the neutral body `name`.

    A body says how it wants each buffer reached, and says it in the only place
    that cannot drift from the code that reads it: its own parameter list. A
    parameter taken by REFERENCE or by VALUE is one element, so the entry point
    gathers `array[index]` into thread space and passes that; a parameter taken
    by POINTER is the array itself, so the entry passes the base. That is the
    whole of the gather-versus-base decision, which is why it is derived here
    rather than written twice.

    Returns None when the body is not in this file, which leaves the decision to
    an explicit attribute rather than guessing.
    """
    for found, first, last in body_signature_spans(lines, masks):
        if found != name:
            continue
        blob = " ".join(code_only(lines[i], masks[i])
                        for i in range(first - 1, last))
        try:
            opened = blob.index("(", blob.index(name))
            inner = blob[opened + 1:blob.rindex(")")]
        except ValueError:
            return None
        shapes = []
        for piece, _offset in _split_top_level(inner):
            # A PARAMETER MAY LEAD WITH AN ATTRIBUTE, and stripping the array
            # suffix by cutting at the first `[` would then cut the whole
            # parameter away and leave the list short by one. Attributes come
            # off first, and only what remains is cut at `[`.
            piece = ATTRIBUTE_GROUP_RE.sub(" ", piece).strip()
            if not piece:
                continue
            # An array parameter decays to a pointer; the name sits before `[`.
            head = piece.split("[")[0].strip()
            if not BODY_PARAM_NAME_RE.search(head):
                continue
            shapes.append("element" if ("&" in piece or "*" not in piece)
                          else "pointer")
        return shapes
    return None


def parse_entry(path, lines, masks, first, last):
    """One entry declaration into an `Entry`."""
    text, locs = _flatten(lines, masks, first, last)
    # The parameters whose access an author spelled out. Everything else is
    # read off the body's own signature below.
    written_access = set()

    # The parameter list's own '(', found with the attributes masked out. Two
    # of them carry an argument, so a misplaced [[seam::pod(N)]] or
    # [[seam::stride(N)]] before the function name would otherwise be read as
    # the parameter list and reported as something unrelated.
    masked = ATTRIBUTE_RE.sub(lambda m: " " * (m.end() - m.start()), text)
    open_paren = masked.find("(")
    if open_paren < 0:
        _at(path, locs, 0, "entry declaration has no parameter list")
    # Counted on the masked text too, so an attribute's own parentheses can
    # never move the depth. They are balanced, so this changes no well-formed
    # declaration; it means a malformed one is reported where it is wrong.
    depth = 0
    close_paren = -1
    for i in range(open_paren, len(masked)):
        if masked[i] == "(":
            depth += 1
        elif masked[i] == ")":
            depth -= 1
            if depth == 0:
                close_paren = i
                break
    if close_paren < 0:
        _at(path, locs, open_paren, "parameter list is never closed")

    head = text[:open_paren]
    head_args = {}
    head_attrs = []
    for m in ATTRIBUTE_RE.finditer(head):
        attr = m.group(1)[len(ATTRIBUTE_PREFIX):]
        head_attrs.append(attr)
        if m.group(2):
            head_args[attr] = m.group(2)[1:-1].strip()
    for name in head_attrs:
        if name not in ("args", "entry", "group"):
            _at(path, locs, 0,
                f"'[[seam::{name}]]' belongs on a parameter, not on the "
                f"declaration")
    # THE THREAD COUNT RIDES THE ENTRY, the way `DISPATCH_START(count)` carries
    # it in the reference: `[[seam::entry(count)]]` names the parameter that
    # says how many threads there are, so the parameter itself needs no second
    # marker. It is the same fact in the same place rather than a marker on the
    # declaration and another on the parameter.
    # `[[seam::entry(count)]]` names the thread count, and
    # `[[seam::entry(count, index)]]` names the thread index beside it, which is
    # the pair the reference carries as `DISPATCH_START(count)` plus the
    # dispatch lambda's own `(unsigned i)`. Both are coordinates the LAUNCH
    # supplies, so both belong on the construct that declares the launch rather
    # than scattered onto the parameters they land on.
    declared_count = declared_index = None
    if "entry" in head_args:
        pieces = [piece.strip() for piece in head_args["entry"].split(",")]
        if len(pieces) > 2:
            _at(path, locs, 0,
                f"'[[seam::entry({head_args['entry']})]]' takes the thread "
                f"count, then optionally the thread index. A launch supplies "
                f"one of each")
        declared_count = pieces[0] or None
        declared_index = pieces[1] if len(pieces) > 1 and pieces[1] else None
    for attr, argument in head_args.items():
        if attr != "entry":
            _at(path, locs, 0,
                f"'[[seam::{attr}({argument})]]' takes no argument. Only "
                f"[[seam::entry]] does, naming the parameter that gives the "
                f"thread count")
    for role, named in (("thread count", declared_count),
                        ("thread index", declared_index)):
        if named is not None and not IDENT_ONLY_RE.match(named):
            _at(path, locs, 0,
                f"'[[seam::entry]]' names the parameter carrying the {role}, "
                f"so '{named}' is one identifier")
    emit_entry = "entry" in head_attrs
    # AN ENTRY POINT IMPLIES ITS RECORD. A backend may define no data type that
    # crosses the seam, so an entry's arguments are generated from this
    # declaration or they do not exist: there is no entry without a record, and
    # writing both said one thing twice. `[[seam::args]]` alone stays legal and
    # means the other half, a record with no launch, which is what a phase that
    # is not dispatched yet declares.
    # Unreachable while `find_entry_spans` opens a span on one of the two, and
    # kept so that widening what opens one cannot quietly emit nothing.
    if not emit_entry and "args" not in head_attrs:
        _at(path, locs, 0,
            "this declaration emits nothing: add [[seam::entry]] for an entry "
            "point and its record, or [[seam::args]] for the record alone")
    is_group = "group" in head_attrs
    if is_group and not emit_entry:
        _at(path, locs, 0,
            "[[seam::group]] names a LAUNCH SHAPE, and [[seam::args]] alone "
            "emits no launch. Add [[seam::entry]] or drop [[seam::group]]")

    head_clean = ATTRIBUTE_RE.sub(lambda m: " " * (m.end() - m.start()), head)
    head_tokens = PARAM_TOKEN_RE.findall(head_clean)
    if len(head_tokens) != 2 or head_tokens[0] != "void":
        _at(path, locs, 0,
            "an entry declaration is 'void <name>(...)': it returns void, "
            "because a kernel entry point has nowhere to return a value to")
    name = head_tokens[1]
    # THE NAME IS THE BODY'S, and nothing here has to check that: every
    # rendering CALLS the body by this name, so an entry naming a body that does
    # not exist fails to compile on all four targets. What this does refuse is
    # the one name the generator owns, since a body called `seam_*` would be
    # shadowed by the locals the entry point emits around its call.
    if name.startswith("seam_"):
        _at(path, locs, 0,
            f"entry '{name}' takes the 'seam_' prefix, which is reserved for "
            f"the locals a generated entry point emits around its call")
    _check_msl_name(path, locs, 0, name, "entry name")

    tail = text[close_paren + 1:].strip()
    if tail != ";":
        _at(path, locs, close_paren,
            "an entry declaration ends at the ';' after its parameter list")

    # Parameters. No parameter may contain a parenthesis (a function-pointer
    # parameter or a default argument with a call in it is refused by the type
    # table below anyway), so a top-level comma split is exact.
    params = []
    piece_start = open_paren + 1
    for i in range(open_paren + 1, close_paren + 1):
        if text[i] == "," or i == close_paren:
            # The reported position is the parameter's first non-blank
            # character, not the comma that precedes it: a parameter list is
            # written one per line, so the comma sits at the end of the
            # PREVIOUS line and a diagnostic anchored there names the wrong
            # line. A blank run here is also what a masked-out block comment
            # leaves behind.
            head = piece_start
            while head < i and text[head] in " \t\n":
                head += 1
            params.append((head if head < i else piece_start,
                           text[piece_start:i]))
            piece_start = i + 1
    if len(params) == 1 and not params[0][1].strip():
        params = []

    fields = []
    forward = []
    scratch = []
    count_field = None
    index_name = None
    diag_name = None
    lane_name = None
    width_name = None
    sink_name = None
    indices_field = None
    bound_field = None
    through_names = []
    declared_names = {}
    offset = 0
    for start, piece in params:
        attrs = []
        attr_argument = {}
        for m in ATTRIBUTE_RE.finditer(piece):
            short = m.group(1)[len(ATTRIBUTE_PREFIX):]
            # A repeat is refused rather than resolved. Two [[seam::pod]] on one
            # parameter would otherwise take the LAST size and assert it, which
            # is a wrong layout that agrees with its own assertion.
            if short in attrs:
                _at(path, locs, start,
                    f"'[[seam::{short}]]' appears twice on one parameter")
            attrs.append(short)
            if m.group(2):
                attr_argument[short] = m.group(2)[1:-1].strip()
        clean = ATTRIBUTE_RE.sub(lambda m: " " * (m.end() - m.start()), piece)
        tokens = PARAM_TOKEN_RE.findall(clean)
        if not tokens:
            _at(path, locs, start, "empty parameter")
        # A TEMPLATE-ID IS REFUSED HERE SO THAT IT IS NOT REPORTED AS SOMETHING
        # ELSE. The split above is on every top-level comma, and a template
        # argument list carries commas of its own, so `SMatf<3, 6> *gradient`
        # arrives as two pieces: one with no name and one whose name is right
        # and whose type is `6>`. The name check below would then report the
        # first as unnamed, which is true of the piece and false of what was
        # written. Every matrix and vector width this tree passes has an alias
        # in `data.hpp` (`Mat3x6f` for `SMatf<3, 6>`), which is also what a
        # sibling declaration spells, so the remedy is a name rather than a
        # wider grammar.
        if "<" in tokens or ">" in tokens:
            _at(path, locs, start,
                "a template argument list in a parameter type. A parameter "
                "list is split on its top-level commas, and the commas inside "
                "'<...>' are not distinguishable from those, so the type must "
                "arrive under a single name: use the alias data.hpp declares "
                "for it")
        pname = tokens[-1]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", pname):
            _at(path, locs, start,
                "every parameter must be named: the name is the record field "
                "name and the driver fills the record by field")
        # TWO NAMES THE SAME IS A RECORD WITH TWO IDENTICAL FIELDS, and it used
        # to be accepted: g++ then refuses the struct, which is loud but LATE
        # and in the wrong file, and the Rust rendering would carry the same
        # pair with only the compiler that reads it last complaining. It is a
        # construct this script does not understand, so it is refused here,
        # with a file, line and column, like every other one.
        if pname in declared_names:
            _at(path, locs, start,
                f"a second parameter named '{pname}'. A parameter name is a "
                f"record field name, and the driver fills the record by field, "
                f"so two of them name one field twice")
        declared_names[pname] = start
        # `diag` IS a generator-owned name, and the diagnostic channel is the
        # one parameter allowed to take it, identified by its TYPE now that
        # `[[seam::diag]]` is inferred from that type rather than written.
        if pname == DIAG_PARAMETER_NAME and (
                "diag" in attrs or DIAG_TYPE_NAME in tokens):
            pass
        elif pname in RESERVED_PARAMETER_NAMES or pname.startswith("seam_"):
            _at(path, locs, start,
                f"parameter '{pname}' takes a name the generated entry points "
                f"already use. The record parameter is 'args', the CPU shim's "
                f"range is 'begin' and 'end', the Metal handle is 'diag', and "
                f"every generator-owned local carries the 'seam_' prefix "
                f"('{GENERATED_INDEX}', 'seam_arena_count'). Each generated "
                f"buffer local is named for its field, so such a name is a "
                f"silent shadow rather than a compile error")
        rest = tokens[:-1]
        if "&" in rest:
            _at(path, locs, start,
                f"parameter '{pname}' is a reference. A record field cannot be "
                f"a reference: the record is bytes the driver fills and the "
                f"backend hands to a kernel")
        stars = rest.count("*")
        rest = [t for t in rest if t != "*"]
        is_const = "const" in rest
        rest = [t for t in rest if t != "const"]
        base = " ".join(rest)
        # `PARAM_TOKEN_RE` makes every punctuation character its own token, so
        # a qualified pointee arrives as `ppf : : atomic_float_t`. Only a
        # DOUBLE colon is rejoined: a single one is not valid here and stays
        # separated, so the type is refused by the tables below rather than
        # silently repaired.
        base = re.sub(r"\s*:\s*:\s*", "::", base)
        if base == "unsigned int":
            base = "unsigned"

        space_attrs = [a for a in attrs if a in ADDRESS_SPACE]
        # Three groups, and only the first says what the parameter IS. `pod`
        # qualifies a pointee and the access attributes say how a buffer is
        # reached, so both legitimately sit beside an address space.
        role_attrs = [a for a in attrs if a in ROLE_ATTRIBUTES]
        access_attrs = [a for a in attrs if a in ACCESS_ATTRIBUTES]
        for a in attrs:
            if a not in ADDRESS_SPACE and a not in ENTRY_ATTRIBUTES:
                _at(path, locs, start,
                    f"'[[seam::{a}]]' is not a parameter attribute")
            if a in GROUP_ONLY_ATTRIBUTES and not is_group:
                _at(path, locs, start,
                    f"parameter '{pname}' carries '[[seam::{a}]]' on an "
                    f"ELEMENT entry, which has no groups: it launches one "
                    f"thread per element and the backend picks the group "
                    f"width. Add [[seam::group]] to the declaration if this "
                    f"kernel cooperates over one problem per group")
        if len(role_attrs) > 1:
            _at(path, locs, start,
                f"parameter '{pname}' carries {len(role_attrs)} of "
                f"[[seam::count]] / [[seam::index]]; at most one is meaningful")
        if role_attrs and access_attrs:
            _at(path, locs, start,
                f"parameter '{pname}' carries [[seam::{role_attrs[0]}]] and "
                f"[[seam::{access_attrs[0]}]]. The first names what the "
                f"parameter IS, a guard bound or the thread index; the second "
                f"names how a BUFFER is reached, and neither of those is a "
                f"buffer")
        if access_attrs and is_group:
            _at(path, locs, start,
                f"parameter '{pname}' carries [[seam::{access_attrs[0]}]] on a "
                f"[[seam::group]] entry. Every access attribute addresses "
                f"one element at the thread index, and a group entry's index "
                f"is its GROUP's: every thread of the group would read the "
                f"same element. A group kernel splits the work by lane "
                f"itself, so it takes the BASE pointer and does its own "
                f"addressing")
        if access_attrs and not emit_entry:
            _at(path, locs, start,
                f"parameter '{pname}' carries [[seam::{access_attrs[0]}]] on a "
                f"declaration with no [[seam::entry]]. An access attribute "
                f"describes the CALL, and [[seam::args]] alone emits the "
                f"record and no call")

        # THE DIAGNOSTIC CHANNEL. Not a record field and not a coordinate: the
        # entry already binds one on every target, and this forwards it so a
        # body with an invariant to report can say so through the same channel
        # the entry's own arena and slot checks use. `seam/seam_host.h` states
        # the contract; `DiagHandle` is the one type name the three
        # prologues agree on.
        # THE TYPE IS THE DECLARATION. `DiagHandle` is the one type name the
        # three prologues agree on and it names nothing else, so a parameter of
        # that type inside an entry declaration IS the diagnostic channel and
        # `[[seam::diag]]` beside it restated the type. The attribute is still
        # accepted, and is checked against the type rather than trusted.
        if "diag" in role_attrs or base == DIAG_TYPE_NAME:
            if "diag" in role_attrs and base != DIAG_TYPE_NAME:
                _at(path, locs, start,
                    f"'{pname}' carries [[seam::diag]] and is '{base}'. The "
                    f"diagnostic channel is '{DIAG_TYPE_NAME}', which is the "
                    f"one type name the three prologues agree on")
            if pname != DIAG_PARAMETER_NAME:
                _at(path, locs, start,
                    f"the [[seam::diag]] parameter is named '{pname}'. It must "
                    f"be '{DIAG_PARAMETER_NAME}', which is the name the entry "
                    f"binds on every target, so the MSL rendering hands the "
                    f"body the handle it already has rather than a second one")
            if stars or space_attrs or is_const:
                _at(path, locs, start,
                    "the [[seam::diag]] parameter is 'DiagHandle', by "
                    "value, with no address space: it is a handle the entry "
                    "binds, not a buffer the driver fills")
            if diag_name is not None:
                _at(path, locs, start,
                    "a second [[seam::diag]] parameter. An entry binds one "
                    "channel")
            diag_name = pname
            forward.append(("diag", pname))
            continue

        # A THREAD INDEX NAMED ON THE ENTRY is the same coordinate the
        # [[seam::index]] marker names, so it is handled here with the other
        # two and is likewise NOT a record field: the launch supplies it.
        if pname == declared_index:
            if "index" in role_attrs:
                _at(path, locs, start,
                    f"'{pname}' is named by [[seam::entry]] as the thread "
                    f"index and also carries [[seam::index]]. One of the two "
                    f"says it")
            # THE POD SIZE IS INFERRED NOW, so it is non-None for any sized
            # type and says nothing about how a parameter was written. What
            # disqualifies a coordinate is a WRITTEN [[seam::pod]], which is
            # what the marker branch below tests.
            if (stars or space_attrs or base != "unsigned" or is_const
                    or "pod" in attr_argument):
                _at(path, locs, start,
                    f"the thread index parameter '{pname}' is 'unsigned', with "
                    f"no address space and no pointee: it is a coordinate the "
                    f"launch supplies, not a buffer")
            if index_name is not None:
                _at(path, locs, start,
                    "a second thread index parameter. A launch supplies one")
            index_name = pname
            forward.append(("index", pname))
            continue

        # The three launch coordinates. None of them is a record field: the
        # launch supplies all three, and each is forwarded in the position it
        # is declared in, which is how a body whose index is not its last
        # parameter is served.
        coordinate = next((a for a in role_attrs
                           if a in ("index", "lane", "width")), None)
        if coordinate is not None:
            if (stars or space_attrs or base != "unsigned" or is_const
                    or "pod" in attr_argument):
                _at(path, locs, start,
                    f"the [[seam::{coordinate}]] parameter is 'unsigned', with "
                    f"no address space and no pointee: it is a coordinate the "
                    f"launch supplies, not a buffer")
            taken = {"index": index_name, "lane": lane_name,
                     "width": width_name}[coordinate]
            if taken is not None:
                _at(path, locs, start,
                    f"a second [[seam::{coordinate}]] parameter. A launch "
                    f"supplies one of each")
            if coordinate == "index":
                index_name = pname
            elif coordinate == "lane":
                lane_name = pname
            else:
                width_name = pname
            forward.append((coordinate, pname))
            continue

        if stars > 1:
            _at(path, locs, start,
                f"parameter '{pname}' is a pointer to a pointer. A record "
                f"field addresses one allocation, as an (arena, offset) handle")
        if stars == 1 and "scratch" in attrs:
            # GROUP-LOCAL SCRATCH. Not a record field and not an allocation:
            # the entry point declares the array itself and passes it, because
            # a driver cannot fill storage that only exists for the duration of
            # one group.
            # THE ADDRESS SPACE IS OPTIONAL HERE BECAUSE IT IS FORCED.
            # `[[seam::scratch]]` IS the threadgroup address space, so the
            # attribute beside it can say one thing and is checked rather than
            # read. Writing it is allowed and adds nothing; omitting it is the
            # same declaration with one fewer thing an author can get wrong.
            if space_attrs and space_attrs != ["threadgroup"]:
                _at(path, locs, start,
                    f"parameter '{pname}' carries [[seam::scratch]] with "
                    f"[[seam::{space_attrs[0]}]]. Scratch IS the threadgroup "
                    f"address space; device memory is a record field and "
                    f"thread storage is local to a thread")
            if is_const:
                _at(path, locs, start,
                    f"scratch parameter '{pname}' is const. Group-local "
                    f"scratch exists to be written; a const one would be read "
                    f"before it was ever set, which on a backend that never "
                    f"faults is whatever the last group left there")
            if base not in SCRATCH_TYPES:
                _at(path, locs, start,
                    f"scratch parameter '{pname}' is '{base} *'. Group-local "
                    f"scratch is an array of one of {sorted(SCRATCH_TYPES)}: "
                    f"its size has to be a literal this script can put in "
                    f"every rendering, and a struct's size is the one thing it "
                    f"cannot compute from this file")
            count = _scratch_count(path, locs, start, pname, attr_argument)
            _check_msl_name(path, locs, start, pname, "scratch array")
            scratch.append((pname, base, count))
            forward.append(("scratch", pname))
            continue

        if stars == 1:
            # THE ADDRESS SPACE IS OPTIONAL HERE BECAUSE IT IS FORCED. A record
            # field names an ALLOCATION, and the two lines below refuse every
            # spelling but `device`, so the attribute carries no information a
            # reader or this parser could act on: it is checked, never read.
            # Omitting it is the same declaration. Writing it is still allowed,
            # so a source predating this reads identically.
            if len(space_attrs) > 1:
                _at(path, locs, start,
                    f"pointer parameter '{pname}' carries "
                    f"{len(space_attrs)} address space attributes, and a "
                    f"record field has one: it names an allocation")
            if space_attrs and space_attrs[0] != "device":
                _at(path, locs, start,
                    f"pointer parameter '{pname}' is "
                    f"[[seam::{space_attrs[0]}]]. A record field is device "
                    f"memory: threadgroup scratch is a "
                    f"launch property and thread storage is local to a thread, "
                    f"so neither can be a field the driver fills")
            pod_bytes = _pod_bytes(path, locs, start, pname, base,
                                   attr_argument)
            if "count" in role_attrs:
                _at(path, locs, start,
                    "[[seam::count]] marks the guard bound, which is an "
                    "unsigned, not a buffer")
            _check_msl_name(path, locs, start, pname, "record field")
            gather = "gather" in access_attrs
            scatter = "scatter" in access_attrs
            if access_attrs:
                written_access.add(pname)
            stride = None
            if "indices" in access_attrs:
                others = [a for a in access_attrs if a != "indices"]
                if others:
                    _at(path, locs, start,
                        f"parameter '{pname}' carries [[seam::indices]] and "
                        f"[[seam::{others[0]}]]. An index list is reached one "
                        f"way, by its own slots at the thread index, so it "
                        f"takes no second access attribute. {ELEMENT_SHAPES}")
                if indices_field is not None:
                    _at(path, locs, start,
                        f"a second [[seam::indices]] parameter, after "
                        f"'{indices_field.name}'. One entry names one index "
                        f"space, because one [[seam::bound]] is what its slots "
                        f"are checked against; a kernel gathering through two "
                        f"different lists is a change to this script")
                if base != "unsigned" or pod_bytes is not None:
                    _at(path, locs, start,
                        f"[[seam::indices]] parameter '{pname}' addresses "
                        f"'{base}'. An index list is 'const unsigned *': its "
                        f"slots subscript another buffer, and this script "
                        f"generates that subscript")
                if not is_const:
                    _at(path, locs, start,
                        f"[[seam::indices]] parameter '{pname}' is not const. "
                        f"An entry READS an element's index list to reach its "
                        f"data; a kernel that rewrites one is scattering, "
                        f"which is a different shape")
                arity = _slot_count(path, locs, start, pname, attr_argument)
                _check_msl_name(path, locs, start, pname, "record field")
                indices_field = Field(pname, "handle", base, is_const, offset,
                                      HANDLE_SIZE, *locs[start],
                                      pod_bytes=pod_bytes, access="indices",
                                      stride=arity)
                fields.append(indices_field)
                offset += HANDLE_SIZE
                # NOT forwarded. The body is handed the ELEMENTS the slots
                # name, never the slots, so an entry point holds the only
                # subscript and the body holds no index arithmetic to get
                # wrong.
                continue
            if "through" in access_attrs:
                others = [a for a in access_attrs if a != "through"]
                if others:
                    _at(path, locs, start,
                        f"parameter '{pname}' carries [[seam::through]] and "
                        f"[[seam::{others[0]}]]. A through buffer is reached "
                        f"at the slots an index list names, and the other "
                        f"shapes are reached at the thread index, so the two "
                        f"describe different elements. {ELEMENT_SHAPES}")
                if not is_const:
                    _at(path, locs, start,
                        f"[[seam::through]] parameter '{pname}' is not const. "
                        f"The N elements it names are passed to the body as N "
                        f"arguments, so a write through one would be a write "
                        f"to a copy on no target and to the buffer on all of "
                        f"them, depending on how the body spells its "
                        f"parameter")
                _check_msl_name(path, locs, start, pname, "record field")
                fields.append(Field(pname, "handle", base, is_const, offset,
                                    HANDLE_SIZE, *locs[start],
                                    pod_bytes=pod_bytes, access="indirect"))
                offset += HANDLE_SIZE
                through_names.append(pname)
                forward.append(("handle", pname))
                continue
            if "stride" in access_attrs:
                if gather or scatter:
                    other = "gather" if gather else "scatter"
                    _at(path, locs, start,
                        f"parameter '{pname}' carries [[seam::stride]] and "
                        f"[[seam::{other}]]. A stride advances a BASE pointer "
                        f"and the other two name one ELEMENT, so the two "
                        f"describe different things to hand the body. "
                        f"{ELEMENT_SHAPES}")
                stride = _stride_count(path, locs, start, pname, attr_argument)
            if scatter:
                if is_const:
                    _at(path, locs, start,
                        f"parameter '{pname}' is [[seam::scatter]] and const. "
                        f"The scatter is where the body's return value is "
                        f"WRITTEN")
                if sink_name is not None:
                    _at(path, locs, start,
                        f"a second [[seam::scatter]] parameter, after "
                        f"'{sink_name}'. One call returns one value, so there "
                        f"is one place to put it. {ELEMENT_SHAPES}")
                sink_name = pname
            access = "element" if gather else ("offset" if stride else "base")
            fields.append(Field(pname, "handle", base, is_const, offset,
                                HANDLE_SIZE, *locs[start],
                                pod_bytes=pod_bytes, access=access,
                                stride=stride, is_sink=scatter))
            offset += HANDLE_SIZE
            # A scatter-only parameter is the SINK, not an argument: the
            # generated entry assigns to it and the body never sees it. With
            # [[seam::gather]] beside it, it is both, which is the in-place
            # element update `x[i] = f(x[i], ...)`.
            if not scatter or gather:
                forward.append(("handle", pname))
            continue

        if space_attrs:
            _at(path, locs, start,
                f"parameter '{pname}' is a scalar carrying "
                f"[[seam::{space_attrs[0]}]]. An address space qualifies a "
                f"pointer")
        if access_attrs:
            _at(path, locs, start,
                f"parameter '{pname}' is a scalar carrying "
                f"[[seam::{access_attrs[0]}]]. An access attribute says how a "
                f"BUFFER is reached, and a scalar arrives in the record "
                f"itself. {ELEMENT_SHAPES}")
        if "pod" in attr_argument:
            _at(path, locs, start,
                f"parameter '{pname}' is a scalar carrying [[seam::pod]]. That "
                f"attribute states the size of a POINTEE, so it belongs on a "
                f"[[seam::device]] pointer")
        if base not in SCALAR_TYPES:
            _at(path, locs, start, _scalar_refusal(pname, base))
        _check_msl_name(path, locs, start, pname, "record field")
        field = Field(pname, "scalar", base, False, offset,
                      SCALAR_TYPES[base][4], *locs[start])
        fields.append(field)
        offset += field.size
        if "bound" in role_attrs:
            if base != "unsigned":
                _at(path, locs, start,
                    "the [[seam::bound]] parameter is 'unsigned': it is the "
                    "size of an index space")
            if bound_field is not None:
                _at(path, locs, start,
                    "a second [[seam::bound]] parameter. One entry names one "
                    "index space, so one number says how large it is")
            bound_field = field
            # NOT forwarded, for the reason [[seam::count]] is not: a body that
            # also wants the number declares a second, plain parameter, so
            # what the body receives is written down rather than inferred.
            continue
        if "count" in role_attrs or pname == declared_count:
            if "count" in role_attrs and pname == declared_count:
                _at(path, locs, start,
                    f"'{pname}' is named by [[seam::entry({declared_count})]] "
                    f"and also carries [[seam::count]]. One of the two says it")
            if base != "unsigned":
                _at(path, locs, start,
                    "the [[seam::count]] parameter is 'unsigned': it is a "
                    "thread count")
            if count_field is not None:
                _at(path, locs, start,
                    "a second [[seam::count]] parameter. One guard, one bound")
            count_field = field
            # NOT forwarded. A body that also wants the count declares a
            # second, plain parameter, so what the body receives is written
            # down rather than inferred.
            continue
        forward.append(("scalar", pname))

    # THE THREAD INDEX MUST REACH SOMETHING. It does so in one of two ways: a
    # forwarded [[seam::index]] parameter, or the per-element addressing of a
    # gather, a scatter or a stride. A declaration doing neither renders an
    # entry point whose every thread computes the same thing from the same
    # bytes, which no launch geometry can rescue, so it is refused here rather
    # than shipped as a kernel that runs and does nothing per thread.
    # THE INDEX LIST, ITS BOUND AND ITS READERS ARRIVE TOGETHER OR NOT AT ALL,
    # and each of the three missing halves fails differently, so each is named.
    # Without the bound there is nothing to check a slot against, which is the
    # whole reason the shape is the entry's work; without the list a
    # [[seam::through]] buffer has no slots to be read at; and a list nothing
    # reads through generates a record field the body never sees, which is a
    # buffer bound for no reason and reads as a lost [[seam::through]].
    if indices_field is not None and bound_field is None:
        _at(path, locs, 0,
            f"'{indices_field.name}' is [[seam::indices]] and this declaration "
            f"has no [[seam::bound]] parameter. A slot read out of that list "
            f"subscripts another buffer, and Metal returns 0.0 for an "
            f"out-of-bounds read rather than faulting, so the size of the "
            f"index space is declared and every slot is checked against it")
    if indices_field is None and bound_field is not None:
        _at(path, locs, 0,
            f"'{bound_field.name}' is [[seam::bound]] and this declaration has "
            f"no [[seam::indices]] parameter. A bound is the size of the index "
            f"space an index list names; with no list there is no slot to "
            f"check against it")
    if indices_field is None and through_names:
        _at(path, locs, 0,
            f"'{through_names[0]}' is [[seam::through]] and this declaration "
            f"has no [[seam::indices]] parameter. A through buffer is read at "
            f"the slots an index list names, and there is no list here to "
            f"name them")
    if indices_field is not None and not through_names:
        _at(path, locs, 0,
            f"'{indices_field.name}' is [[seam::indices]] and nothing is "
            f"[[seam::through]] it. The slots are not forwarded, so this "
            f"declaration reads an index list, checks it, and hands the body "
            f"nothing from it")
    # A GROUP ENTRY NEEDS BOTH COORDINATES, and the two failures it prevents
    # are different. Without the group index every group computes the same
    # value from the same bytes, which is the element form's rule. Without the
    # lane every THREAD of a group computes the whole of it, so the group is
    # doing its work as many times over as it has threads, writing the same
    # answer from every one of them; nothing in the output says so and on a
    # backend that never faults nothing traps.
    if is_group:
        missing = [name for name, value in (("index", index_name),
                                            ("lane", lane_name))
                   if value is None]
        if missing:
            _at(path, locs, 0,
                "a [[seam::group]] entry forwards both coordinates: "
                "[[seam::index]] is the GROUP's position and [[seam::lane]] is "
                "the thread's position inside it. This declaration is missing "
                + " and ".join(f"[[seam::{m}]]" for m in missing) +
                ". Without the group index every group computes the same "
                "value; without the lane every thread of a group computes the "
                "whole of it")
    if declared_index is not None and index_name != declared_index:
        _at(path, locs, 0,
            f"[[seam::entry]] names '{declared_index}' as the thread index and "
            f"no parameter of this declaration is called that")
    if declared_count is not None and count_field is None:
        _at(path, locs, 0,
            f"[[seam::entry({declared_count})]] names no parameter of this "
            f"declaration. It names the one carrying the thread count")
    if emit_entry and count_field is None:
        _at(path, locs, 0,
            "an entry point needs a thread count: name the parameter with "
            "[[seam::entry(<name>)]], or mark it [[seam::count]]. Metal never "
            "faults on an out-of-bounds access, so the in-kernel guard is the "
            "bound, and a launch rounds its grid up to whole threadgroups on "
            "every backend. On a [[seam::group]] entry it is the GROUP count")
    # Metal's static threadgroup limit, enforced there at pipeline creation.
    # Refused here so the failure names the declaration rather than arriving as
    # a pipeline that would not build on one of the three backends.
    scratch_bytes = sum(SCRATCH_TYPES[base][4] * count
                        for _, base, count in scratch)
    if scratch_bytes > MAX_SCRATCH_BYTES:
        fail(path, first, 1,
             f"the group-local scratch is {scratch_bytes} bytes, over the "
             f"{MAX_SCRATCH_BYTES} byte cap a static threadgroup array has on "
             f"Metal. That cap is enforced at pipeline creation there, so a "
             f"declaration over it builds on two backends and not the third")

    entry = Entry(name, fields, forward, count_field, index_name, emit_entry,
                  first, last, is_group=is_group, lane_name=lane_name,
                  width_name=width_name, scratch=scratch,
                  indices_field=indices_field, bound_field=bound_field,
                  diag_name=diag_name)
    if entry.has_handles:
        # Generator-owned, and every rendering carries it. ARENA_PTR
        # resolves an out-of-range arena id to the LAST arena rather than
        # faulting, so without a live bound the assert below cannot be written
        # and a wrong arena id is a silent wrong answer with plausible floats.
        arena_count = Field("seam_arena_count", "scalar", "unsigned", False,
                            offset, 4, first, 1)
        entry.fields.append(arena_count)
        offset += 4
    if offset > MAX_ARGS_BYTES:
        fail(path, first, 1,
             f"argument record is {offset} bytes, over the {MAX_ARGS_BYTES} "
             f"byte portable cap. Metal's own cap is 32752 bytes and 32756 "
             f"kills the process with SIGABRT and nothing on stdout or "
             f"stderr, so this has to be refused here")
    # ------------------------------------------------------------------
    # THE SINK IS THE BUFFER THE BODY HAS NO PARAMETER FOR.
    #
    # A body that RETURNS a value has somewhere for that value to go; a body
    # returning void has none. And the destination is the one buffer this
    # declaration names that the body's own parameter list does not accept, so
    # both halves are read off the body rather than written beside the buffer.
    #
    # The search tries each non-const buffer as the one the body does not take,
    # and the sink is the candidate whose removal makes the forwarded list line
    # up with the body's parameters. TWO candidates fitting is an ambiguity this
    # refuses rather than guesses at, and none fitting is a body with nowhere to
    # put its result; both say so by name.
    #
    # THE IN-PLACE UPDATE IS THE ONE SHAPE THIS CANNOT SEE, `x[i] = f(x[i], ...)`,
    # where the destination is ALSO an argument and so is not missing from the
    # body at all. Those three declarations still write [[seam::scatter]].
    if emit_entry and sink_name is None:
        returns = body_return_type(lines, masks, name)
        shapes = body_parameter_shapes(lines, masks, name)
        if returns not in (None, "void") and shapes is not None:
            by_name = {f.name: f for f in fields}

            def _aligns(without):
                expanded = []
                for kind, fname in forward:
                    if fname == without:
                        continue
                    field = by_name.get(fname)
                    repeat = 1
                    if (kind == "handle" and field is not None
                            and field.access == "indirect"
                            and indices_field is not None):
                        repeat = indices_field.stride
                    expanded += [(kind, fname)] * repeat
                # ARITY ONLY, and deliberately. Whether a buffer is gathered
                # or passed whole is inferred BELOW, from these same shapes, so
                # it is not known yet here and comparing against it would
                # reject every declaration whose buffers are gathered. What
                # makes this safe is that a second candidate fitting the same
                # arity is reported as ambiguous rather than guessed.
                return len(expanded) == len(shapes)

            fits = [f.name for f in fields
                    if f.kind == "handle" and not f.is_const
                    and f.name in {n for _k, n in forward} and _aligns(f.name)]
            if len(fits) == 1:
                sink_name = fits[0]
                by_name[sink_name].is_sink = True
                forward[:] = [(k, n) for k, n in forward if n != sink_name]
            elif not fits:
                _at(path, locs, 0,
                    f"'{name}' returns {returns} and no buffer this declaration "
                    f"names is missing from its parameter list, so there is "
                    f"nowhere to put the value. Name the destination with "
                    f"[[seam::scatter]], or return void")
            else:
                _at(path, locs, 0,
                    f"'{name}' returns {returns} and {len(fits)} of this "
                    f"declaration's buffers would fit as the destination "
                    f"({', '.join(fits)}), so which one takes the value cannot "
                    f"be read off the body. Name it with [[seam::scatter]]")

    # ------------------------------------------------------------------
    # THE GATHER DECISION IS READ OFF THE BODY, NOT WRITTEN TWICE.
    #
    # A body's parameter list already says how it wants each buffer reached: by
    # REFERENCE or by VALUE is one element, so the entry gathers `array[index]`
    # into thread space; by POINTER is the array, so the entry passes the base.
    # Writing `[[seam::gather]]` beside that restates a fact the body owns, and
    # a restatement is a place for two declarations to disagree.
    #
    # Pairing walks the forwarded list against the body's parameters, expanding
    # a [[seam::through]] buffer into the elements its index list names, which
    # is exactly what the generated entry does. A `[[seam::gather]]` that IS
    # written is kept and checked, so an author may still say it and cannot say
    # it wrongly.
    shapes = body_parameter_shapes(lines, masks, name)
    if shapes is not None:
        by_name = {f.name: f for f in fields}
        expanded = []
        for kind, fname in forward:
            field = by_name.get(fname)
            repeat = 1
            if (kind == "handle" and field is not None
                    and field.access == "indirect"
                    and indices_field is not None):
                repeat = indices_field.stride
            expanded += [(kind, fname)] * repeat
        if len(expanded) == len(shapes):
            for (kind, fname), shape in zip(expanded, shapes):
                if kind != "handle":
                    continue
                field = by_name.get(fname)
                if field is None or field.access not in ("base", "element"):
                    continue
                wanted = "element" if shape == "element" else "base"
                if fname not in written_access:
                    field.access = wanted
                elif field.access != wanted:
                    _at(path, locs, 0,
                        f"parameter '{fname}' is declared "
                        f"[[seam::{'gather' if field.access == 'element' else 'scatter'}]] "
                        f"and '{name}' takes it "
                        f"{'by pointer' if shape == 'pointer' else 'by reference'}, "
                        f"which is the other shape. The body decides this: a "
                        f"reference or a value is one element and a pointer is "
                        f"the array, so delete the attribute or change the "
                        f"body's parameter")
        elif any(f.kind == "handle" and f.name not in written_access
                 and f.access in ("base", "element") for f in fields):
            _at(path, locs, 0,
                f"the forwarded arguments of '{name}' do not line up with its "
                f"body's {len(shapes)} parameters, so how each buffer is "
                f"reached cannot be read off the body. Give every buffer an "
                f"explicit access attribute, or fix the declaration to match "
                f"the body it calls")

    # THE THREAD INDEX MUST REACH SOMETHING, asked of the access each field
    # ENDED UP with rather than of the attributes written down, because a
    # gathered buffer now says so through the body's parameter list. Asking
    # before the inference reads every such entry as unindexed.
    indexed = any(f.access != "base" or f.is_sink for f in fields)
    if emit_entry and not is_group and index_name is None and not indexed:
        _at(path, locs, 0,
            "an entry point's thread index reaches nothing. Forward it with a "
            "[[seam::index]] parameter, or address a buffer with it (a body "
            "parameter taken by reference is gathered at the index; "
            "[[seam::scatter]], [[seam::stride(N)]] and [[seam::indices(N)]] "
            "name the other three); otherwise every thread computes the same "
            "value from the same bytes")
    _check_msl_name(path, locs, 0, entry.record_cpp, "record name")
    return entry


# ---------------------------------------------------------------------------
# THE SIZE OF A STRUCT POINTEE, WHICH THIS SCRIPT COMPUTES RATHER THAN ASKING
# THE DECLARATION FOR.
#
# `[[seam::pod(N)]]` states how many bytes a handle's pointee occupies, and it
# looks like the one thing about a generated entry that this script cannot
# compute. It can: every such type is either a scalar, an alias in
# `linalg/type_aliases.hpp` that resolves to `SMat<T, R, C>`, or a record in
# `data_records.hpp` whose fields are themselves one of those.
#
# THE ASSERTION IS WHAT MAKES THIS SAFE. All three C++ renderings assert
# `sizeof(T)` against the number, so a size computed wrongly here, or a type the
# Metal shader compiler lays out differently from host C++, fails to COMPILE
# rather than reading the wrong bytes. Deriving the number moves who writes it,
# not what checks it.

# SIZE AND ALIGNMENT, because a record is laid out and not merely summed:
# `VertexProp` mixes `float`, `unsigned`, `bool` and `unsigned char`, so a sum
# of field sizes gives 42 where the type is 44. Each field starts at its own
# alignment and the whole rounds up to the struct's.
SCALAR_POINTEE_BYTES = {
    "float": (4, 4), "unsigned": (4, 4), "int": (4, 4),
    "bool": (1, 1), "char": (1, 1), "unsigned char": (1, 1),
    "signed char": (1, 1), "short": (2, 2), "unsigned short": (2, 2),
    "compute::atomic_float_t": (4, 4), "compute::atomic_uint_t": (4, 4),
    "atomic_float_t": (4, 4), "atomic_uint_t": (4, 4),
}
ALIAS_RE = re.compile(
    r"\busing\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);")
TEMPLATE_ALIAS_RE = re.compile(
    r"\btemplate\s*<[^>]*>\s*using\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);")
SMAT_RE = re.compile(r"\bSMat\s*<[^,]+,\s*(\d+)\s*,\s*(\d+)\s*>")
SVEC_RE = re.compile(r"\bSVec\s*<[^,]+,\s*(\d+)\s*>")
STRUCT_DEF_RE = re.compile(
    r"\bstruct\s+(?:alignas\s*\(\s*(\d+)\s*\)\s*)?"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\{")
FIELD_RE = re.compile(
    r"^\s*(?:const\s+)?([A-Za-z_][A-Za-z0-9_:]*(?:\s*<[^;{}]*>)?)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[\s*(\d+)\s*\])?\s*;", re.M)

_POINTEE_SIZES = None


def pointee_sizes(source_path=None):
    """{type: (bytes, alignment)} for every pointee a declaration can name.

    MEMOIZED PER ROOT, not once. A caller that has not set `KERNEL_ROOT`, which
    every checker under `.github/workflows/scripts` is, would otherwise cache a
    scalar-only table on the first call and then report every struct pointee as
    unresolvable. The root is deduced from the source being parsed when it was
    not given.
    """
    global _POINTEE_SIZES
    root = KERNEL_ROOT
    if root is None and source_path is not None:
        try:
            root = _kernel_root_of(source_path)
        except Exception:
            root = None
    if _POINTEE_SIZES is not None and _POINTEE_SIZES[0] == root:
        return _POINTEE_SIZES[1]
    sizes = dict(SCALAR_POINTEE_BYTES)
    if root is None or not os.path.isdir(root):
        _POINTEE_SIZES = (root, sizes)
        return sizes
    KERNEL_ROOT_LOCAL = root

    def read(relative):
        path = os.path.join(KERNEL_ROOT_LOCAL, relative)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "",
                                                      handle.read(), flags=re.S))
        except OSError:
            return ""

    aliases = read("linalg/type_aliases.hpp")
    # `template <class T> using Vec3 = SVec<T, 3>;` gives the SHAPE; the plain
    # alias below names the element type.
    shapes = {}
    for name, target in TEMPLATE_ALIAS_RE.findall(aliases):
        mat = SMAT_RE.search(target)
        vec = SVEC_RE.search(target)
        if mat:
            shapes[name] = int(mat.group(1)) * int(mat.group(2))
        elif vec:
            shapes[name] = int(vec.group(1))
    for name, target in ALIAS_RE.findall(aliases):
        if name in shapes:
            continue
        mat = SMAT_RE.search(target)
        vec = SVEC_RE.search(target)
        head = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*<", target)
        if mat:
            sizes[name] = (int(mat.group(1)) * int(mat.group(2)) * 4, 4)
        elif vec:
            sizes[name] = (int(vec.group(1)) * 4, 4)
        elif head and head.group(1) in shapes:
            sizes[name] = (shapes[head.group(1)] * 4, 4)

    # A RECORD IS ITS FIELDS, all of which are scalars or the aliases above, and
    # every one of those is 4-byte aligned, so the sum is the size until an
    # explicit `alignas` rounds it up. `AABB` is the one that does.
    # EVERY NEUTRAL SOURCE, not only `data_records.hpp`: a record may be
    # declared beside the kernel that uses it (`PdrdRigidState`) or in a shared
    # header (`LockFrame`), and a type this cannot see keeps writing its size.
    chunks = []
    for folder, _dirs, names in os.walk(KERNEL_ROOT_LOCAL):
        for filename in sorted(names):
            if filename.endswith((".hpp", ".h", ".kernel.cpp")):
                chunks.append(read(os.path.relpath(
                    os.path.join(folder, filename), KERNEL_ROOT_LOCAL)))
    records = "\n".join(chunks)
    # AN ENUM IS AN INT UNLESS IT SAYS OTHERWISE, and a record may hold one:
    # `FaceParam` opens with `Model model;`. Without this the record is
    # unresolvable and its `pod` would have to stay written.
    for match in re.finditer(
            r"\benum(?:\s+(?:class|struct))?\s+([A-Za-z_][A-Za-z0-9_]*)"
            r"\s*(?::\s*([A-Za-z_][A-Za-z0-9_ ]*?))?\s*\{", records):
        name, underlying = match.group(1), (match.group(2) or "int").strip()
        if name not in sizes and underlying in sizes:
            sizes[name] = sizes[underlying]

    for _round in range(4):
        for match in STRUCT_DEF_RE.finditer(records):
            align, name = match.group(1), match.group(2)
            if name in sizes:
                continue
            index, depth = match.end() - 1, 0
            while index < len(records):
                if records[index] == "{":
                    depth += 1
                elif records[index] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                index += 1
            body = records[match.end():index]
            offset, widest, known = 0, 1, True
            for kind, _field, count in FIELD_RE.findall(body):
                base = re.sub(r"\s*<.*", "", kind).strip()
                if base not in sizes:
                    known = False
                    break
                size, alignment = sizes[base]
                offset = (offset + alignment - 1) // alignment * alignment
                offset += size * (int(count) if count else 1)
                widest = max(widest, alignment)
            if known and offset:
                step = int(align) if align else widest
                sizes[name] = ((offset + step - 1) // step * step, step)
    _POINTEE_SIZES = (root, sizes)
    return sizes


def _pod_bytes(path, locs, start, pname, base, attr_argument):
    """The pointee size for a [[seam::device]] pointer, or None for a scalar.

    A pointee is one of the three scalars, whose size this script knows, or a
    struct, whose size it cannot compute from one file. The struct case states
    the size in the declaration and every C++ rendering asserts it, so a type
    laid out differently by one of the three compilers fails to compile there
    rather than reading the wrong bytes on a backend that never faults.
    """
    declared = attr_argument.get("pod")
    if base in POINTEE_TYPES:
        if declared is not None:
            _at(path, locs, start,
                f"parameter '{pname}' addresses '{base}', whose size this "
                f"script already knows, so [[seam::pod]] would be a second "
                f"place for it to be written and a second place to be wrong")
        return None
    if declared is None:
        # THE SIZE IS COMPUTED, NOT DECLARED. `pointee_sizes` lays the type out
        # from the neutral tree's own definitions: a scalar, an alias resolving
        # to `SMat<T, R, C>`, an enum, or a record whose fields are those. Every
        # C++ rendering still ASSERTS the number, so a size computed wrongly
        # here, or a type the Metal shader compiler lays out differently, fails
        # to compile rather than reading the wrong bytes on a backend that never
        # faults. Deriving it moves who writes the number, not what checks it.
        computed = pointee_sizes(path).get(base)
        if computed is None:
            _at(path, locs, start,
                f"parameter '{pname}' addresses '{base}', which is not one of " +
                ", ".join(sorted(POINTEE_TYPES)) +
                " and whose layout this script could not compute from the "
                "neutral tree. Declare it as [[seam::pod(N)]], or give the type "
                "a definition the size pass can read")
        return computed[0]
    # A SIZE WRITTEN BY HAND IS STILL CHECKED against the computed one, so the
    # two cannot drift while both spellings are accepted.
    computed = pointee_sizes(path).get(base)
    if computed is not None and computed[0] != int(declared):
        _at(path, locs, start,
            f"parameter '{pname}' declares [[seam::pod({declared})]] for "
            f"'{base}', which this script lays out at {computed[0]} bytes. One "
            f"of the two is wrong, and the declaration is the one that can be "
            f"deleted: the size is computed when it is absent")
    # A plain identifier, or ONE name out of a seam namespace (SEAM_NAMESPACES
    # above). `compute::atomic_float_t` is the case that needs it: it is
    # `float` on CUDA and on the host and `atomic_float` on MSL, so a scatter
    # target declared as `float *` would compile on two backends and fail on
    # the third. Any OTHER qualified or templated name stays refused, because
    # nothing guarantees the three compilers spell it alike.
    if not SEAM_QUALIFIED_RE.fullmatch(base):
        _at(path, locs, start,
            f"parameter '{pname}' addresses '{base}'. A struct pointee is a "
            f"plain identifier or one seam name: any other qualified "
            f"or templated name is spelled differently by the three compilers")
    tail = base.split("::")[-1]
    if tail in FORBIDDEN_IDENTIFIERS:
        _at(path, locs, start,
            f"parameter '{pname}' addresses '{base}', which is a backend "
            f"spelling: {FORBIDDEN_IDENTIFIERS[tail]}")
    _check_msl_name(path, locs, start, tail, "pointee type")
    if not re.fullmatch(r"[0-9]+", declared):
        _at(path, locs, start,
            f"[[seam::pod({declared})]] on '{pname}' is not a byte count")
    size = int(declared)
    if size == 0 or size % 4 != 0:
        _at(path, locs, start,
            f"[[seam::pod({size})]] on '{pname}': a pointee shared with a "
            f"device is a multiple of 4 bytes, because every field this "
            f"project shares with a kernel is 4-byte aligned and anything "
            f"narrower reintroduces the padding a size assert cannot see")
    return size


def _slot_count(path, locs, start, pname, attr_argument):
    """The N of `[[seam::indices(N)]]`: how many indices one element names.

    A literal, positive and capped, for three separate reasons. The generated
    prologue reads all N slots into thread storage before the body is called,
    so N is a stack cost on every target; it also fixes how many arguments a
    [[seam::through]] parameter expands into, which is what makes the generated
    call match a body signature the generator does not parse; and a run-time N
    would put the expansion out of reach of a generator at all.
    """
    declared = attr_argument["indices"]
    if not re.fullmatch(r"[0-9]+", declared):
        _at(path, locs, start,
            f"[[seam::indices({declared})]] on '{pname}' is not a slot count")
    slots = int(declared)
    if slots == 0:
        _at(path, locs, start,
            f"[[seam::indices(0)]] on '{pname}': an element naming no index "
            f"gathers nothing, so the declaration would generate a call with "
            f"no argument where the body expects one")
    if slots > MAX_INDEX_SLOTS:
        _at(path, locs, start,
            f"[[seam::indices({slots})]] on '{pname}' is over the "
            f"{MAX_INDEX_SLOTS} slot cap. Every slot is read into thread "
            f"storage before the body is called and every [[seam::through]] "
            f"buffer expands into that many arguments, so an element naming "
            f"this many indices is not the shape this form shortens")
    return slots


def _stride_count(path, locs, start, pname, attr_argument):
    """The N of `[[seam::stride(N)]]`, the run one thread reads.

    A literal, positive, and nothing else: the whole value of the form is that
    the generated expression `base + N * index` is readable beside the launcher
    it replaces, and an expression here would be the first place a branch or a
    scene-dependent quantity could enter an entry point.
    """
    declared = attr_argument.get("stride")
    if declared is None:
        _at(path, locs, start,
            f"[[seam::stride]] on '{pname}' needs its argument: the run one "
            f"thread reads cannot be computed from this file, which is why it "
            f"is written down")
    if not re.fullmatch(r"[0-9]+", declared):
        _at(path, locs, start,
            f"[[seam::stride({declared})]] on '{pname}' is not an element "
            f"count. It is a literal, because an expression here is where a "
            f"branch or a scene-dependent quantity would enter an entry point")
    count = int(declared)
    if count == 0:
        _at(path, locs, start,
            f"[[seam::stride(0)]] on '{pname}' hands every thread the same "
            f"pointer, which is the base pointer with extra text; write no "
            f"access attribute if that is what the body wants")
    return count


def _scratch_count(path, locs, start, pname, attr_argument):
    """The element count of a [[seam::scratch(N)]] array.

    A literal, for the same reason a stride is: it is written into every
    rendering as an array bound, and there is nothing in this file to derive it
    from. The cap is the one platform figure that decides it, Metal's 32768
    byte static threadgroup limit, which that platform enforces at pipeline
    creation; refusing here makes the failure name the declaration instead.
    """
    declared = attr_argument.get("scratch")
    if not declared or not declared.isdigit():
        _at(path, locs, start,
            f"scratch parameter '{pname}' needs an element count: "
            f"[[seam::scratch(N)]]. It is an array bound in every rendering, "
            f"so it is a literal and not an expression")
    count = int(declared)
    if count == 0:
        _at(path, locs, start,
            f"scratch parameter '{pname}' asks for 0 elements. A group with no "
            f"scratch declares no scratch parameter")
    return count


def _scalar_refusal(pname, base):
    """Why a scalar type is refused, in the terms that make it actionable."""
    if base in ("double", "long double"):
        return (f"parameter '{pname}' is '{base}'. MSL has no double at all, "
                f"so a record carrying one has no Metal rendering, and GPU "
                f"compute in this project is float32 only")
    if base in ("bool", "char", "signed char", "unsigned char", "short",
                "unsigned short"):
        return (f"parameter '{pname}' is '{base}', narrower than 4 bytes. A "
                f"record holds only 4-byte scalars and 16-byte handles so it "
                f"has NO padding and its size is the sum of its fields; a "
                f"field that fits in existing padding moves no size, which is "
                f"the one drift a sizeof assert cannot see")
    if re.fullmatch(r"(float|half|int|uint|char|uchar|short|ushort|long|ulong|"
                    r"bool)\d(x\d)?", base):
        return (f"parameter '{pname}' is the vector or matrix type '{base}'. "
                f"MSL float3 is 16 bytes against Vec3f at 12 and float3x3 is "
                f"48 against Mat3x3f at 36, so a record carrying one has a "
                f"different layout per target while still compiling "
                f"everywhere. Pass the components")
    return (f"parameter '{pname}' has type '{base}'. A record field is one of "
            + ", ".join(sorted(SCALAR_TYPES)) +
            ", or a [[seam::device]] pointer to one of those")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_line(path, lineno, line, mask, in_entry=False):
    """Rejects everything this script does not understand. Runs before any
    substitution, so a message quotes the neutral source as written.

    `in_entry` says the line sits inside an entry declaration, which is the
    only place the [[seam::args]] / [[seam::entry]] / [[seam::count]] /
    [[seam::index]] family is legal.
    """
    code = code_only(line, mask)

    m = SEAM_MACRO_RE.search(code)
    if m:
        fail(path, lineno, m.start() + 1,
             f"'{m.group(0)}': the backend macro seam is not available to a "
             f"neutral kernel body. Address and execution spaces are the "
             f"[[seam::...]] attributes; everything else is a "
             f"namespaced "
             f"function or type the backend prologue defines (fmath::sqrt, "
             f"compute::atomic_uint_t, ...)")

    for m in ATTRIBUTE_RE.finditer(code):
        name = m.group(1)
        if not name.startswith(ATTRIBUTE_PREFIX):
            fail(path, lineno, m.start() + 1,
                 f"attribute '[[{name}]]': only the seam:: attributes are "
                 f"translated, and an attribute this script passes through "
                 f"would have to be accepted by all three compilers")
        short = name[len(ATTRIBUTE_PREFIX):]
        if short not in ALL_ATTRIBUTES:
            fail(path, lineno, m.start() + 1,
                 f"unknown attribute '[[seam::{short}]]'. Known: " +
                 ", ".join(sorted(f"[[seam::{k}]]" for k in ALL_ATTRIBUTES)))
        if short in ENTRY_ATTRIBUTES and not in_entry:
            fail(path, lineno, m.start() + 1,
                 f"'[[seam::{short}]]' outside an entry declaration. The entry "
                 f"family is legal only inside a declaration carrying "
                 f"[[seam::entry]] or [[seam::args]], which must be a "
                 f"declaration (no body) terminated by ';'")
        if (m.group(2) and short not in ATTRIBUTES_WITH_ARGUMENT
                and short not in ATTRIBUTES_WITH_OPTIONAL_ARGUMENT):
            fail(path, lineno, m.start() + 1,
                 f"'[[seam::{short}]]' takes no argument")
        if short in ATTRIBUTES_WITH_ARGUMENT and not m.group(2):
            fail(path, lineno, m.start() + 1,
                 f"'[[seam::{short}]]' needs its argument, "
                 f"{ATTRIBUTE_ARGUMENT_IS[short]}. It cannot be computed from "
                 f"this file, which is why it is written down")

    # Identifier scan. The attribute text itself holds `thread` and `device`
    # after the namespace, so attribute ranges are excluded before the scan.
    blanked = list(code)
    for m in ATTRIBUTE_RE.finditer(code):
        for i in range(m.start(), m.end()):
            blanked[i] = " "
    blanked = "".join(blanked)
    for m in IDENTIFIER_RE.finditer(blanked):
        word = m.group(0)
        if word in FORBIDDEN_IDENTIFIERS:
            fail(path, lineno, m.start() + 1,
                 f"'{word}' is a backend spelling and cannot appear in a "
                 f"neutral kernel body: {FORBIDDEN_IDENTIFIERS[word]}")

    # Every attribute match above has been blanked, so a surviving `[[` opens
    # an attribute this script did not recognize: broken across two lines, most
    # likely, since the pattern is matched one line at a time. Passing it
    # through is the quiet failure worth preventing. nvcc answers an
    # unrecognized `[[seam::device_fn]]` with warning #2803-D and makes the
    # function __host__, so the definition compiles and only a device CALL SITE
    # fails, elsewhere; a body with no device caller in that translation unit
    # compiles clean and is simply absent from the device image. MSL and host
    # C++ ignore the attribute, so CUDA is the backend that diverges.
    # Only the opening bracket is checked: `]]` closes a nested subscript
    # (`transpose_offset[index[slot]]`) and is ordinary code.
    m = re.search(r"\[\[", blanked)
    if m:
        fail(path, lineno, m.start() + 1,
             f"'[[' that is not a complete [[seam::...]] "
             f"attribute. An attribute is recognized one line at a time, so it "
             f"must be written on a single line")

    stripped = code.strip()
    if not stripped.startswith("#"):
        return
    d = DIRECTIVE_RE.match(code)
    directive = d.group(1)
    rest = d.group(2).strip()
    if directive == "pragma":
        if rest != "once":
            fail(path, lineno, 1,
                 f"'#pragma {rest}': only '#pragma once' is carried through")
        return
    if directive == "include":
        # Parsed off the RAW line: the include target is a string literal, so
        # the masked form has it blanked out.
        if not INCLUDE_RE.match(line):
            fail(path, lineno, 1,
                 "an #include in a neutral kernel body must be the quoted "
                 "form. An angle include names a standard header, and the "
                 "Metal shader compiler is handed one concatenated string with "
                 "no filesystem behind it, so it can serve none of them")
        return
    fail(path, lineno, 1,
         f"'#{directive}': a neutral kernel body carries no preprocessor "
         f"directive other than '#pragma once' and a quoted '#include'. The "
         f"backend difference this one would express belongs in cpp/seam or in "
         f"kMslMacroSeam in ../metal/shader_compiler.mm")


# ---------------------------------------------------------------------------
# Rewriting
# ---------------------------------------------------------------------------

def substitute_attributes(line, mask, target, insertions=(), replacements=()):
    """Replaces every [[seam::...]] in code with this target's spelling.

    Applied right to left so an earlier match's position stays valid. The
    replacement never contains a newline, so the line count cannot move.

    `insertions` are (column, address space) pairs for parameters that did not
    WRITE an address space and whose space the entry decides. They are applied
    in the SAME right-to-left pass, because both they and the substitutions are
    expressed in the ORIGINAL line's columns: doing them in two passes would
    make the second one's positions stale.
    """
    code = code_only(line, mask)
    index = TARGETS.index(target)
    # AN EXPLICIT REPLACEMENT OWNS ITS RANGE. A specialization rewrites the
    # address space a parameter was written with, and the ordinary substitution
    # would otherwise edit the same characters: two overlapping edits applied in
    # one pass corrupt the line, which is what `[[seam::thread]]3fp &y` was.
    covered = [(start - 1, end - 1) for start, end, _text in replacements]
    edits = [(m.start(), m.end(),
              ATTRIBUTES[m.group(1)[len(ATTRIBUTE_PREFIX):]][index])
             for m in ATTRIBUTE_RE.finditer(code)
             if not any(m.start() < stop and start < m.end()
                        for start, stop in covered)]
    for start_col, end_col, text in replacements:
        edits.append((start_col - 1, end_col - 1, text))
    for column, space in insertions:
        spelling = ADDRESS_SPACE[space][index]
        # A target that spells the space as nothing needs no insertion, and
        # emitting one would put a stray space into a rendering that is
        # otherwise character for character what the author wrote.
        if spelling:
            edits.append((column - 1, column - 1, spelling + " "))
    out = line
    for start, end, text in sorted(edits, reverse=True):
        out = out[:start] + text + out[end:]
    return out


def rewrite_include(path, lineno, line, mask, target, source_dir):
    """Points a quoted include at the file this target actually reads.

    A neutral body names its dependency the way the source tree spells it. The
    generated file sits elsewhere, so the target of the include is resolved here
    against the neutral file's own directory:

      * another `*.kernel.cpp` becomes the generated sibling for this target,
        by the same relative path. The generated tree mirrors the source tree,
        so the spelling is unchanged apart from the extension.
      * any other header becomes its absolute path, which resolves from
        wherever the generated file is compiled.
      * a target that does not exist is an error.

    On the `metal` target there is no include mechanism at all: the line is
    commented out with the same marker the shader assembler writes, so the
    result is identical whether or not the assembler's own pass also runs.
    """
    code = code_only(line, mask)
    if not code.lstrip().startswith("#"):
        return None
    # Matched against the RAW line, because the include target is a string
    # literal and the masked form has it blanked out. The mask above is what
    # establishes that the '#' is code rather than the inside of a comment.
    m = INCLUDE_RE.match(line)
    if not m:
        return None
    target_name = m.group(2)
    resolved = os.path.normpath(os.path.join(source_dir, target_name))
    if not os.path.isfile(resolved):
        fail(path, lineno, 1,
             f"#include \"{target_name}\" resolves to {resolved}, which does "
             f"not exist")

    if target == "metal":
        return METAL_INCLUDE_MARKER + line

    if resolved.endswith(KERNEL_SUFFIX):
        # A kernel dependency keeps its relative spelling, which resolves
        # inside the mirrored generated tree.
        stem = target_name[: -len(KERNEL_SUFFIX)]
        replacement = stem + GENERATED_SUFFIX[target]
    else:
        replacement = os.path.abspath(resolved)
    return f'{m.group(1)}"{replacement}"{m.group(3).rstrip()}'


# ---------------------------------------------------------------------------
# Entry renderings
#
# Four artifacts from one declaration: the __global__ nvcc compiles, the
# `kernel void` the Metal shader compiler compiles, the extern "C" shim a host
# C++ compiler compiles, and the Rust #[repr(C)] twin the driver holds. The
# first three carry ONE symbol name, `<stem>_entry`, which is what lets a
# single driver call site be answered by whichever backend was built.
# All four carry `sizeof` / `offsetof` assertions written against the SAME
# computed layout, so the four languages cannot disagree silently.
# ---------------------------------------------------------------------------

# The neutral kernel tree this run renders from.
#
# It is SET FROM AN ARGUMENT rather than deduced from this script's location,
# and that is the rule rather than a convenience. The transcompiler belongs to
# `ppf-cts-compute` while the kernel tree belongs to the caller, so the two are
# not siblings and no relative path from here can find it. A deduced root is
# also the coupling that hides the two entries below: with the script inside the
# kernel tree, an absolute include into that tree reads as a local file rather
# than as a dependency the generator has grown.
KERNEL_ROOT = None


def _cpp_header(relative):
    """An absolute path to a header in the caller's kernel tree.

    Refused if it is not there, and refused if no kernel root was given. The
    existence check is load-bearing: it is what makes this generator's remaining
    dependency on the kernel tree visible from here, so do not weaken it to a
    silent pass-through. That dependency is `arena_handle.hpp` and nothing else:
    the handle layout is neutral, so every target's entry rendering names it
    from the caller's tree, while the allocator that resolves a handle belongs
    to a target and is named without a path.
    """
    if KERNEL_ROOT is None:
        raise KernelGenError(
            "a generated entry point needs a header from the kernel tree, but "
            "no --kernel-root was given. This script does not live inside the "
            "kernel tree and cannot deduce where it is")
    resolved = os.path.join(KERNEL_ROOT, relative)
    if not os.path.isfile(resolved):
        raise KernelGenError(
            f"{resolved}: a generated entry point needs this header and it is "
            f"not there. The kernel tree is the --kernel-root argument")
    return resolved


def _entry_banner(target, source_path, emit="entry"):
    absolute = os.path.abspath(source_path)
    return (f"// Generated by ppf-cts-compute/seam/kernelgen.py --target {target} --emit "
            f"{emit} from\n// {absolute}. Do not edit.\n"
            f"//\n"
            f"// An entry point is an argument record, plus a thread index, "
            f"plus a call into\n"
            f"// the neutral body. Writing one by hand is writing a mirror "
            f"pair by hand: two\n"
            f"// declarations that can disagree, with nothing linking them. "
            f"The record's field\n"
            f"// offsets and its size are asserted below against the same "
            f"layout every other\n"
            f"// rendering of this declaration asserts, so the four languages "
            f"cannot drift.\n")


def _sizeof_asserts(entry, size_expr, offset_expr, terminator):
    """The layout assertions, spelled for one language.

    Every field is asserted, not just the size. A `sizeof` assert alone cannot
    see a field that fits in existing padding, which is why this record admits
    no padding, and it cannot see two fields exchanged, which is what a
    per-field offset assert catches.
    """
    out = [f"{size_expr(entry.size)}{terminator}"]
    for field in entry.fields:
        out.append(f"{offset_expr(field.name, field.offset)}{terminator}")
    return out


def render_args_cu(source_path, entries, target="cu"):
    """The CUDA argument records, includable by any number of callers.

    THE HALF OF AN ENTRY RENDERING THAT IS NOT A DEFINITION. A record is a
    struct, its layout assertions and its pointee size assertions are
    assertions, and a launcher declaration is a declaration, so all of it is
    legal in as many translation units as read it. What is NOT here is the
    `__global__` and the launcher body, which are definitions and live next
    door in the `.entry.cu` that includes this file.

    WHY THE SPLIT EXISTS. A caller that fills a record needs the record's TYPE,
    and until this file existed the only place that type was written was beside
    the two definitions, so a second translation unit could not have it without
    also acquiring them and failing to link. Every remaining hand-written CUDA
    launch in the neutral kernel tree sits in a header that two or three
    translation units include, so that is not an incidental restriction: it is
    what stops those launches from becoming generated entry points.

    THE RECORD IS STILL DEFINED EXACTLY ONCE. The entry rendering includes this
    file rather than repeating it, so the bytes a filler writes and the bytes
    the entry point reads are one declaration by construction, which is the
    same argument that makes the entry point generated in the first place.
    """
    body_include = os.path.basename(source_path)[:-len(KERNEL_SUFFIX)] + \
        GENERATED_SUFFIX[target]
    out = [_entry_banner(target, source_path, emit="args"),
           "//\n"
           "// THE RECORDS ONLY. This file carries no definition, so ANY "
           "number of\n"
           "// translation units may include it: a caller that fills one of "
           "these records\n"
           "// needs its type, and the `__global__` and the launcher beside it "
           "would not\n"
           "// survive a second inclusion. Those two are in the .entry.cu that "
           "includes\n"
           "// this file, which exactly one translation unit may read.\n"
           "#pragma once\n\n"
           "#include <cstddef>\n"
           f'#include "{_cpp_header("arena_handle.hpp")}"\n'
           "// The neutral body, for the types a pointee names. It is named "
           "WITHOUT a\n"
           "// path because the CUDA build puts the rendering directory on the "
           "include\n"
           "// path, and it carries #pragma once, so a translation unit that "
           "also\n"
           "// includes it directly is unaffected.\n"
           f'#include "{body_include}"\n']
    out.extend("\n" + a for a in _pod_asserts(entries))
    for entry in entries:
        out.append("\n" + _record_cpp(entry, "ArenaHandle",
                                      lambda f: SCALAR_TYPES[f.base][0]))
        out.extend(_sizeof_asserts(
            entry,
            lambda n: (f"static_assert(sizeof({entry.record_cpp}) == {n},\n"
                       f'              "{entry.record_cpp} layout changed")'),
            lambda f, o: (f"static_assert(offsetof({entry.record_cpp}, {f}) "
                          f"== {o},\n"
                          f'              "{entry.record_cpp}.{f} moved")'),
            ";\n"))
        out.append(f"static_assert(alignof({entry.record_cpp}) == "
                   f"{HANDLE_ALIGN},\n"
                   f'              "{entry.record_cpp} alignment changed");\n')
        if not entry.emit_entry:
            continue
        # The launcher's DECLARATION, rendered from the same parse as its
        # definition. A caller writing this line by hand would be writing one
        # half of a mirror pair, which is the construct the entry form exists
        # to remove.
        if entry.is_group:
            out.append(f'extern "C" void {entry.launch_symbol}(\n'
                       f"    const void *record, unsigned seam_group_width,\n"
                       f"    {DEVICE_CXX[target]['stream_type']} queue);\n")
        else:
            out.append(f'extern "C" void {entry.launch_symbol}(\n'
                       f"    const void *record, {DEVICE_CXX[target]['stream_type']} queue);\n")
    return "".join(out)


def render_args_metal(source_path, entries):
    """The Metal argument records, for the HOST that fills them.

    THE SECOND TRANSLATION UNIT IS THE HOST, NOT A SECOND SHADER. A Metal entry
    rendering is spliced into one assembled shader, so for a long time the
    record looked like it needed only one home. It does not. The shader READS
    the record; the ObjC++ host WRITES it and hands it to `context_set_bytes`,
    and those are different translation units compiled by different compilers.
    Until this rendering existed the host wrote its half by hand, which is a
    mirror pair: two declarations of one layout with nothing linking them, the
    exact construct the entry form exists to remove.

    WHY THIS IS NOT THE CUDA RENDERING. That one declares an `extern "C"`
    launcher taking a `cudaStream_t`, because on CUDA the launch is a function.
    On Metal it is a pipeline object the host dispatches, so there is no
    launcher to declare and no CUDA type to name, and the CUDA header cannot
    stand in for this one.

    THE ASSERTIONS ARE THE LINK. The struct here and the struct in the
    `.entry.metal` rendering come from one parse of one declaration, and each
    carries the same size, offset and alignment assertions computed from it. The
    host's are checked when the backend is compiled and the shader's when the
    framework compiles the assembled source at `initialize()`, so a neutral
    declaration that moves under either half is a compile error on that half
    rather than a wrong number at run time.
    """
    body_include = os.path.basename(source_path)[:-len(KERNEL_SUFFIX)] + \
        GENERATED_SUFFIX["cpp"]
    out = [_entry_banner("metal", source_path, emit="args"),
           "//\n"
           "// THE RECORDS ONLY, for the host side of the Metal seam. This "
           "file carries\n"
           "// no definition and no launcher, so any number of translation "
           "units may\n"
           "// include it. The shader's copy of these records is in the "
           ".entry.metal\n"
           "// rendering, spliced rather than included, and both are rendered "
           "from the\n"
           "// same parse of the same declaration.\n"
           "#pragma once\n\n"
           "#include <cstddef>\n"
           f'#include "{_cpp_header("arena_handle.hpp")}"\n'
           "// The host rendering of the neutral body, for the types a pointee "
           "names. A\n"
           "// quoted include resolves against this file's own directory, "
           "which is where\n"
           "// that rendering is written, and it carries #pragma once.\n"
           f'#include "{body_include}"\n']
    out.extend("\n" + a for a in _pod_asserts(entries))
    for entry in entries:
        out.append("\n" + _record_cpp(entry, "ArenaHandle",
                                       lambda f: SCALAR_TYPES[f.base][0]))
        out.extend(_sizeof_asserts(
            entry,
            lambda n: (f"static_assert(sizeof({entry.record_cpp}) == {n},\n"
                       f'              "{entry.record_cpp} layout changed")'),
            lambda f, o: (f"static_assert(offsetof({entry.record_cpp}, {f}) "
                          f"== {o},\n"
                          f'              "{entry.record_cpp}.{f} moved")'),
            ";\n"))
        out.append(f"static_assert(alignof({entry.record_cpp}) == "
                   f"{HANDLE_ALIGN},\n"
                   f'              "{entry.record_cpp} alignment changed");\n')
    return "".join(out)


def render_entry_cu(source_path, entries, target="cu"):
    args_include = os.path.basename(source_path)[:-len(KERNEL_SUFFIX)] + \
        ARGS_SUFFIX[target]
    out = [_entry_banner(target, source_path),
           "//\n"
           "// EXACTLY ONE translation unit may include this file: it DEFINES "
           "a __global__\n"
           "// and a launcher, and #pragma once does not reach across "
           "translation units.\n"
           "// The RECORDS those two are written against are not here: they "
           "are in the\n"
           "// .args.cuh included below, which any number of callers may read, "
           "so a caller\n"
           "// that fills a record does not have to acquire these definitions "
           "with it.\n"
           "//\n"
           "// The handles resolve through compute::arena::resolve, which "
           "asserts "
           "the arena id,\n"
           "// the size against the allocation and the offset's alignment. "
           "Those asserts are\n"
           "// LIVE in the release build, so CUDA's bound check is the "
           "allocator's own.\n"
           "//\n"
           "// EACH ENTRY POINT BELOW GETS TWO SYMBOLS. The __global__ is a "
           "C++ symbol with\n"
           "// a launch syntax rather than a function whose address can be "
           "taken, so only a\n"
           "// translation unit nvcc compiles can reach one. The C-linkage "
           "launcher beside\n"
           "// it is the half anything holding the record's bytes can call, "
           "which is what a\n"
           "// name-to-launcher table holds and what a caller above the seam "
           "reaches.\n"
           "//\n"
           "// A LAUNCHER COPIES THE RECORD IN rather than reading it in "
           "place, and each of\n"
           "// the two reasons stands on its own. The caller's bytes carry no "
           "alignment this\n"
           "// file may assume, and seam_arena_count is the LIBRARY's fact "
           "rather than the\n"
           "// caller's, being the live arena count of the allocator these "
           "handles resolve\n"
           "// against, which a caller above the seam does not have.\n"
           "//\n"
           "// THE GRID IS SIZED FROM THE RECORD'S OWN COUNT FIELD, never "
           "from a second\n"
           "// number passed beside it. A grid wider than the count costs the "
           "extra threads\n"
           "// their guard and nothing else, while a grid narrower than it "
           "leaves elements\n"
           "// unprocessed with nothing to see, so the launch and the guard "
           "read one field.\n"
           "#pragma once\n\n"
           "#include <cstring>\n"
           '// Three headers named WITHOUT a path, none of them this '
           "generator's to\n"
           '// place. The allocator and the error check belong to the CUDA '
           "target and\n"
           "// the block-size choice to the caller's kernel tree; the CUDA "
           "build puts\n"
           "// both roots on the include path, so all three resolve for "
           "exactly the\n"
           "// compiler that reads a .entry.cu at all.\n"
           '#include "arena/arena.hpp"\n'
           '#include "common.hpp"\n'
           f'#include "{DEVICE_CXX[target]["utils_include"]}"\n']
    # The channel's transport, included whenever some entry in this file takes
    # one. Unconditionally would be simpler and is wrong: a file whose entries
    # report nothing would then carry a dependency on the backend's
    # diagnostics, which is exactly the coupling the seam's one-name contract
    # exists to avoid.
    if any(e.diag_name for e in entries):
        out.append('#include "diagnostics/diagnostics.hpp"\n')
    out.append(f'#include "{args_include}"\n')
    for entry in entries:
        if not entry.emit_entry:
            continue
        out.append(f"\n__global__ void {entry.entry_symbol}("
                   f"{entry.record_cpp} args"
                   f"{_diag_cu_param(entry)}) {{\n")
        if entry.is_group:
            out.append(_guard_note(4))
            out.append(f"    const unsigned {entry.index_name} = blockIdx.x;\n"
                       f"    if ({entry.index_name} >= args."
                       f"{entry.count_field.name}) {{\n"
                       f"        return;\n"
                       f"    }}\n"
                       f"    const unsigned {entry.lane_name} = threadIdx.x;\n")
            if entry.width_name:
                out.append(f"    const unsigned {entry.width_name} = "
                           f"blockDim.x;\n")
            for name, base, count in entry.scratch:
                out.append(f"    __shared__ {SCRATCH_TYPES[base][0]} "
                           f"{name}[{count}];\n")
        else:
            out.append(f"    const unsigned {entry.index_name} = "
                       f"blockIdx.x * blockDim.x + threadIdx.x;\n"
                       f"    if ({entry.index_name} >= args."
                       f"{entry.count_field.name}) {{\n"
                       f"        return;\n"
                       f"    }}\n")
        out.append(_diag_cu_bind(entry))
        out.extend(_hoist(entry, _resolve_cu, "args.", 0))
        out.append(_slots(entry, "args.", 4, "assert(seam_slots_valid);"))
        out.append(_call(entry, "args.") + "}\n")
        out.append(_launcher_cu(entry, target))
    return "".join(out)


def _launcher_cu(entry, target="cu"):
    """The host-callable half of one CUDA entry point.

    The rendered file's own header states why there are two symbols and where
    each number in this one comes from; what is here is the shape.

    It is generated for the same reason the __global__ is. A hand-written
    launcher fills a generated record field by field, so it is a second
    declaration of the record's contents with nothing linking it to the first:
    a field added to the declaration is absorbed silently by a positional
    initializer and left unset by a designated one, and neither shows up as a
    compile error.
    """
    count = entry.count_field.name
    arena_count = ""
    if entry.has_handles:
        # Only a record holding a handle carries the field, and only the
        # library knows what to put in it.
        arena_count = (
            "    args.seam_arena_count = "
            "compute::arena::arena_count(compute::arena::active());\n")
    if entry.is_group:
        # THE GROUP WIDTH IS THE CALLER'S, and that is the difference between
        # the two shapes rather than an inconsistency. In the element shape the
        # width computes no value, so the launcher picks it; in the group shape
        # it is how many threads cooperate on one problem, which is a statement
        # about the kernel and its data that nothing here can make. The group
        # COUNT still comes from the record's own count field, so the launch
        # and the guard read one number.
        return (f'\nextern "C" void {entry.launch_symbol}(\n'
                f"    const void *record, unsigned seam_group_width,\n"
                f"    {DEVICE_CXX[target]['stream_type']} queue) {{\n"
                f"    {entry.record_cpp} args;\n"
                f"    std::memcpy(&args, record, sizeof(args));\n"
                f"{arena_count}"
                f"    if (args.{count} == 0 || seam_group_width == 0) {{\n"
                f"        return;\n"
                f"    }}\n"
                f"    {entry.entry_symbol}<<<args.{count}, seam_group_width, "
                f"0, queue>>>(args{_diag_cu_arg(entry)});\n"
                f"    {DEVICE_CXX[target]['error_macro']}"
            f"({DEVICE_CXX[target]['last_error']}());\n"
                f"}}\n")
    return (f'\nextern "C" void {entry.launch_symbol}(\n'
            f"    const void *record, {DEVICE_CXX[target]['stream_type']} queue) {{\n"
            f"    {entry.record_cpp} args;\n"
            f"    std::memcpy(&args, record, sizeof(args));\n"
            f"{arena_count}"
            f"    if (args.{count} == 0) {{\n"
            f"        return;\n"
            f"    }}\n"
            f"    const unsigned block_size = "
            f"choose_block_size(args.{count});\n"
            f"    const unsigned blocks = "
            f"(args.{count} + block_size - 1) / block_size;\n"
            f"    {entry.entry_symbol}<<<blocks, block_size, 0, queue>>>"
            f"(args{_diag_cu_arg(entry)});\n"
            f"    {DEVICE_CXX[target]['error_macro']}"
            f"({DEVICE_CXX[target]['last_error']}());\n"
            f"}}\n")


def render_entry_metal(source_path, entries):
    out = [_entry_banner("metal", source_path),
           "//\n"
           "// Spliced into the assembled shader as a segment. It names "
           "ArenaHandle,\n"
           "// ARENA_ARGS, ARENA_PTR and the DIAG / DIAG_ASSERT "
           "channel, so the\n"
           "// segment carrying those must be spliced first, exactly as every "
           "hand-written\n"
           "// argument record in the Metal backend already requires.\n"
           "//\n"
           "// The arena-id check is not defensive. ARENA_PTR resolves an "
           "id past the\n"
           "// last arena to the LAST arena rather than faulting, and Metal "
           "never faults on\n"
           "// an out-of-bounds access at all, so without this check a wrong "
           "arena id is a\n"
           "// silent wrong answer with plausible floats and a Completed "
           "status.\n"]
    out.extend("\n" + a for a in _pod_asserts(entries))
    for entry in entries:
        out.append("\n" + _record_cpp(entry, "ArenaHandle",
                                      lambda f: SCALAR_TYPES[f.base][1]))
        out.extend(_sizeof_asserts(
            entry,
            lambda n: (f"static_assert(sizeof({entry.record_cpp}) == {n},\n"
                       f'              "{entry.record_cpp} layout changed")'),
            lambda f, o: (f"static_assert(__builtin_offsetof("
                          f"{entry.record_cpp}, {f}) == {o},\n"
                          f'              "{entry.record_cpp}.{f} moved")'),
            ";\n"))
        # `alignof` is a C++11 keyword the Metal shader compiler accepts;
        # `offsetof` is not, because MSL has no <cstddef>, hence the clang
        # builtin above.
        out.append(f"static_assert(alignof({entry.record_cpp}) == "
                   f"{HANDLE_ALIGN},\n"
                   f'              "{entry.record_cpp} alignment changed");\n')
        if not entry.emit_entry:
            continue
        handles = entry.handles
        valid = " &&\n        ".join(
            f"args.{f.name}.arena < args.seam_arena_count" for f in handles)
        out.append(f"\nkernel void {entry.entry_symbol}(\n")
        if handles:
            out.append("    ARENA_ARGS,\n")
        out.append(f"    constant {entry.record_cpp} &args [[buffer(29)]],\n")
        if entry.is_group:
            out.append(f"    uint {entry.index_name} "
                       f"[[threadgroup_position_in_grid]],\n"
                       f"    uint {entry.lane_name} "
                       f"[[thread_position_in_threadgroup]],\n")
            if entry.width_name:
                out.append(f"    uint {entry.width_name} "
                           f"[[threads_per_threadgroup]],\n")
        else:
            out.append(f"    uint {entry.index_name} "
                       f"[[thread_position_in_grid]],\n")
        out.append(f"    DIAG_ARG) {{\n")
        if entry.is_group:
            out.append(_guard_note(4))
        out.append(f"    if ({entry.index_name} >= args."
                   f"{entry.count_field.name}) {{\n"
                   f"        return;\n"
                   f"    }}\n")
        if entry.is_group:
            for name, base, count in entry.scratch:
                out.append(f"    threadgroup {SCRATCH_TYPES[base][1]} "
                           f"{name}[{count}];\n")
        if handles or entry.diag_name:
            out.append(f"    Diag diag = "
                       f"DIAG_BIND({entry.index_name});\n")
        if handles:
            out.append(f"    const bool seam_arenas_valid =\n"
                       f"        {valid};\n"
                       f"    DIAG_ASSERT(diag, seam_arenas_valid);\n"
                       f"    if (!seam_arenas_valid) {{\n"
                       f"        return;\n"
                       f"    }}\n")
        # The resolution comes AFTER the arena check, and the order is
        # load-bearing: ARENA_PTR resolves an id past the last arena to the
        # LAST arena rather than faulting, so resolving first would compute a
        # plausible address out of a wrong id before anything had rejected it.
        out.extend(_hoist(entry, _resolve_metal, "args.", 1,
                          address_space="device "))
        out.append(_slots(entry, "args.", 4,
                          "DIAG_ASSERT(diag, seam_slots_valid);",
                          unsigned=SCALAR_TYPES["unsigned"][1]))
        # THREAD-SPACE COPIES, and only on this target. The other two have one
        # address space, so only MSL needs a `device` element moved into
        # `thread` before a body that takes it by reference can be handed it.
        out.append(_thread_locals(entry))
        # COPY-OUT, after the call and before the closing brace. Without it a
        # body writing through a gathered element would write into a thread
        # local and the buffer would keep whatever it held, which on Metal is
        # a plausible answer rather than a fault.
        writebacks = _thread_writebacks(entry)
        # A COPY-OUT MAKES THE CALL A STATEMENT RATHER THAN A `return`, which
        # gives up one refusal on this target and keeps it on the other two. A
        # sink-less call is normally made through `return`, so a body that
        # returns a value does not compile; nothing may follow a `return`,
        # though, and the copy-out must. The cu and cpp renderings of the SAME
        # declaration still take the call through `return`, and both are
        # compiled (nvcc into the C ABI library, the host compiler into the CPU
        # shim), so the body is still held to returning void.
        out.append(_call(entry, "args.", thread_locals=True,
                         statement=bool(writebacks)))
        out.append(writebacks)
        out.append("}\n")
    return "".join(out)


def _diag_cu_param(entry):
    """The `__global__`'s channel parameter, or nothing.

    `DIAG_ARG` names the parameter itself rather than expanding to a type,
    because the three targets carry three different things across that boundary
    and only the CUDA one is a by-value struct. So the entry cannot spell the
    parameter and hand the macro a name; it emits the macro whole and relies on
    `DIAG_BIND` to produce the handle under the declared name.
    """
    if entry.diag_name is None:
        return ""
    return ", DIAG_ARG"


def _diag_cu_bind(entry):
    """Bind the channel to this thread, under the name the body declared."""
    if entry.diag_name is None:
        return ""
    return (f"    DiagHandle {entry.diag_name} = "
            f"DIAG_BIND({entry.index_name});\n")


def _diag_cu_arg(entry):
    """The channel the launcher hands the kernel.

    The GLOBAL channel, not one the record carries: which channel a dispatch
    reports through is a property of the process rather than of the kernel or
    its data, and a record field would let two dispatches of one entry disagree
    about where a failed assert lands.
    """
    if entry.diag_name is None:
        return ""
    return ", diagnostics::global()"


def _diag_cpp_param(entry):
    """The host entry's diagnostic parameter, or nothing.

    THE HOST LAUNCH ALREADY CARRIES ONE. `ppf_cts_compute`'s `Launch` is
    `(args, begin, end, diag)`, and the driver's generated thunk has been
    discarding that fourth argument since the first conversion. An entry whose
    body asked for the channel takes it as a fifth parameter and the thunk
    passes it through, which is why this is additive: an entry that did not ask
    keeps the four-argument signature every existing thunk calls.
    """
    if entry.diag_name is None:
        return ""
    return f",\n    DiagHandle {entry.diag_name}"


def render_entry_cpp(source_path, entries):
    body_include = os.path.basename(source_path)
    out = [_entry_banner("cpp", source_path),
           "//\n"
           "// EXACTLY ONE translation unit may include this file: it DEFINES "
           "the range\n"
           "// shim, and #pragma once does not reach across translation "
           "units.\n"
           "//\n"
           "// The range shim is the CPU backend's launch: one call covers "
           "[begin, end), so\n"
           "// a scheduler chooses the chunking and this file states none. "
           "The arena base\n"
           "// table is a parameter rather than a global, because resolving a "
           "handle is the\n"
           "// caller's allocator's business and this file may not know whose "
           "it is.\n"
           "#pragma once\n\n"
           "#include <cassert>\n"
           "#include <cstddef>\n"
           f'#include "{_cpp_header("arena_handle.hpp")}"\n'
           f'#include "{body_include}"\n']
    out.extend("\n" + a for a in _pod_asserts(entries))
    for entry in entries:
        out.append("\n" + _record_cpp(entry, "ArenaHandle",
                                      lambda f: SCALAR_TYPES[f.base][2]))
        out.extend(_sizeof_asserts(
            entry,
            lambda n: (f"static_assert(sizeof({entry.record_cpp}) == {n},\n"
                       f'              "{entry.record_cpp} layout changed")'),
            lambda f, o: (f"static_assert(offsetof({entry.record_cpp}, {f}) "
                          f"== {o},\n"
                          f'              "{entry.record_cpp}.{f} moved")'),
            ";\n"))
        out.append(f"static_assert(alignof({entry.record_cpp}) == "
                   f"{HANDLE_ALIGN},\n"
                   f'              "{entry.record_cpp} alignment changed");\n')
        if not entry.emit_entry:
            continue
        if entry.is_group:
            out.append(_shim_cpp_group(entry))
            continue
        out.append(f"\nextern \"C\" void {entry.entry_symbol}(\n"
                   f"    const {entry.record_cpp} *args,\n"
                   f"    unsigned char *const *seam_arena_base,\n"
                   f"    unsigned begin, unsigned end"
                   f"{_diag_cpp_param(entry)}) {{\n")
        for f in entry.handles:
            out.append(f"    assert(args->{f.name}.arena < "
                       f"args->seam_arena_count);\n")
        out.extend(_hoist(entry, _resolve_cpp, "args->", 2))
        # THE CALL IS MADE THROUGH A VOID-RETURNING FORWARDER, and that is not
        # a style choice. A plain call statement DISCARDS whatever the body
        # returns, so an entry declared over a body that returns its result
        # would compile, run, and write nothing: storing that value is the job
        # of the hand-written launcher a generated entry replaces.
        # `return expr;` in a void function is well-formed only when `expr` is
        # itself void, so the compiler refuses that entry here instead. With a
        # [[seam::scatter]] the check runs the other way and is the compiler's
        # just the same: the call is the right-hand side of an assignment, so a
        # body returning void is refused there.
        out.append(f"    const auto seam_call = "
                   f"[&](unsigned {entry.index_name}) -> void {{\n"
                   f"{_slots(entry, 'args->', 8, 'assert(seam_slots_valid);')}"
                   f"{_call(entry, 'args->', indent=8)}"
                   f"    }};\n"
                   f"    const unsigned seam_last =\n"
                   f"        end < args->{entry.count_field.name} ? end : "
                   f"args->{entry.count_field.name};\n"
                   f"    for (unsigned {entry.index_name} = begin;\n"
                   f"         {entry.index_name} < seam_last; "
                   f"++{entry.index_name}) {{\n"
                   f"        seam_call({entry.index_name});\n"
                   f"    }}\n}}\n")

    return "".join(out)


def _shim_cpp_group(entry):
    """The host rendering of a [[seam::group]] entry: its lanes, in order.

    A GROUP'S LANES RUN ONE AFTER ANOTHER, lane 0 through lane width - 1, one
    whole group per step of the outer loop. That is a rendering of this shape
    rather than a stand-in for one, and what makes it exact is the narrow set
    of channels a group has. Lanes of one group exchange data two ways and no
    others: through the group-local scratch, and through device memory. On
    both device targets a read of another lane's write is ordered by
    `compute::threadgroup_barrier()` and is a data race without one. So for a
    body that does not synchronize, one order visits the elements every other
    order visits and computes what every other order computes.

    A BODY THAT DOES SYNCHRONIZE HAS NO HOST RENDERING, and the compiler says
    so rather than this script. `compute::threadgroup_barrier` is one of the six
    names the host seam leaves undefined (`seam/seam_host.h`), each because it
    has no meaning for a lane that runs to completion before the next one
    starts. A body naming it fails to compile in this rendering, at the neutral
    file and line the `#line` directive carries, instead of compiling into an
    oracle that computes something else. This script does not read the body and
    could not make that judgment; the seam's missing name makes it exactly, for
    a barrier reached through any depth of included helper.

    WHAT THE RANGE COUNTS. `[begin, end)` is over GROUPS, not over threads. The
    count field bounds groups in this shape, which is the same field the CUDA
    launcher sizes its grid in BLOCKS from, so the two targets read one number.
    A caller that chunks the work reads the Rust twin's `_IS_GROUP` constant to
    know which of the two ranges it is holding.

    WHAT IS NOT PROMISED. The lane order is an order, not the order: a device
    target runs the lanes at once, so a fold through an atomic lands in a
    different sequence and an fp32 sum can differ in its last bits. That is the
    reassociation every backend of this tree already carries against every
    other, and it is why a parity gate over an atomic fold compares within a
    tolerance rather than bit for bit.
    """
    count = entry.count_field.name
    out = ["\n"
           "// A GROUP'S LANES RUN ONE AFTER ANOTHER, lane 0 first, one whole "
           "group per step\n"
           "// of the outer loop. Lanes exchange data through the group-local "
           "scratch and\n"
           "// through device memory, and on both device targets a read of "
           "another lane's\n"
           "// write is ordered by compute::threadgroup_barrier() and is a "
           "race without one,\n"
           "// so for a body that does not synchronize this order computes "
           "what every other\n"
           "// order computes. A body that DOES synchronize does not compile "
           "here: the host\n"
           "// seam leaves compute::threadgroup_barrier undefined, so it is "
           "refused at the\n"
           "// neutral file and line rather than rendered into an oracle that "
           "computes\n"
           "// something else.\n"
           "//\n"
           "// THE RANGE IS OVER GROUPS, not over threads: the count field "
           "bounds groups in\n"
           "// this shape, which is the field the CUDA launcher sizes its grid "
           "in BLOCKS\n"
           "// from, so the two targets read one number.\n"
           f'extern "C" void {entry.entry_symbol}(\n'
           f"    const {entry.record_cpp} *args,\n"
           f"    unsigned char *const *seam_arena_base,\n"
           f"    unsigned seam_group_width, unsigned begin, unsigned end"
           f"{_diag_cpp_param(entry)}) {{\n"]
    for f in entry.handles:
        out.append(f"    assert(args->{f.name}.arena < "
                   f"args->seam_arena_count);\n")
    out.extend(_hoist(entry, _resolve_cpp, "args->", 2))
    if entry.width_name:
        out.append(f"    const unsigned {entry.width_name} = "
                   f"seam_group_width;\n")
    out.append(f"    const unsigned seam_last =\n"
               f"        end < args->{count} ? end : args->{count};\n"
               f"    for (unsigned {entry.index_name} = begin;\n"
               f"         {entry.index_name} < seam_last; "
               f"++{entry.index_name}) {{\n")
    if entry.scratch:
        # LEFT UNINITIALIZED, matching both device targets: threadgroup memory
        # arrives with unspecified contents, so a body that reads its scratch
        # before writing it is a defect this rendering reproduces rather than
        # hides. One array per group, so its lifetime is the group's.
        out.append("        // Uninitialized, as threadgroup memory is on "
                   "both device targets:\n"
                   "        // a body reading its scratch before writing it "
                   "is a defect this\n"
                   "        // rendering reproduces rather than hides.\n")
    for name, base, length in entry.scratch:
        out.append(f"        {SCRATCH_TYPES[base][2]} {name}[{length}];\n")
    # The lambda is what refuses a body whose return value nothing stores, for
    # the reason the element shim states; it is declared inside the group loop
    # so that it can capture that group's scratch.
    out.append(f"        const auto seam_call = "
               f"[&](unsigned {entry.lane_name}) -> void {{\n"
               f"{_call(entry, 'args->', indent=12)}"
               f"        }};\n"
               f"        for (unsigned {entry.lane_name} = 0;\n"
               f"             {entry.lane_name} < seam_group_width; "
               f"++{entry.lane_name}) {{\n"
               f"            seam_call({entry.lane_name});\n"
               f"        }}\n"
               f"    }}\n}}\n")
    return "".join(out)


def render_entry_rust(source_path, entries, migrated=()):
    # Plain `//`, never `//!`. This file is `include!`d into a module, and an
    # inner doc comment is legal only at the top of one, so a `//!` banner is
    # twelve E0753 errors in whichever module adopts the artifact.
    out = [_entry_banner("rust", source_path),
           "//\n"
           "// Written to be `include!`d by the driver module that owns "
           f"`{RUST_BUFFER_TYPE}`, `{RUST_HANDLE_TYPE}`,\n"
           "// `KernelId`, `KernelArgs` and the `id` module, so those names "
           "are declared\n"
           "// once and this file re-declares none of them.\n"
           "//\n"
           "// THE KERNEL ID IS REFERENCED, NOT SPELLED. It is a dense index "
           "over every\n"
           "// entry the driver dispatches, so only the driver can assign it; "
           "a literal\n"
           "// here would put the same number in the build script as well, "
           "with nothing\n"
           "// linking the two. Naming `id::<STEM>` makes a missing or "
           "renamed id a\n"
           "// compile error in the driver instead.\n"]
    for entry in entries:
        upper = entry.stem.upper()
        out.append("\n#[repr(C)]\n"
                   "#[derive(Clone, Copy, Debug, PartialEq)]\n"
                   f"pub struct {entry.record_rust} {{\n")
        for f in entry.fields:
            if f.kind != "handle":
                ty = SCALAR_TYPES[f.base][3]
            elif _is_migrated(entry, f, migrated):
                ty = RUST_HANDLE_TYPE
            else:
                ty = RUST_BUFFER_TYPE
            if f.name == "seam_arena_count":
                out.append("    /// Generator-owned: the backend fills it "
                           "with its own live arena\n"
                           "    /// count as it binds the buffers, so a "
                           "driver leaves it at zero.\n")
            elif f.pod_bytes is not None:
                out.append(f"    /// Addresses `{f.base}`, "
                           f"{f.pod_bytes} bytes per element.\n")
            # How the entry point reaches it. The driver fills a handle either
            # way, so this changes nothing it has to do; it is here because the
            # element width the reference must cover follows from it, and that
            # is the driver's to get right.
            for note in _access_note(f):
                out.append(f"    /// {note}\n")
            out.append(f"    pub {f.name}: {ty},\n")
        out.append("}\n")
        out.append(f"const _: () = assert!(\n"
                   f"    core::mem::size_of::<{entry.record_rust}>() == "
                   f"{entry.size}\n);\n")
        out.append(f"const _: () = assert!(\n"
                   f"    core::mem::align_of::<{entry.record_rust}>() == "
                   f"{HANDLE_ALIGN}\n);\n")
        for f in entry.fields:
            out.append(f"const _: () = assert!(\n"
                       f"    core::mem::offset_of!({entry.record_rust}, "
                       f"{f.name}) == {f.offset}\n);\n")
        if not entry.emit_entry:
            continue
        offsets = ", ".join(str(f.offset) for f in entry.fields
                            if f.kind == "handle")
        # The buffers this record still names by ADDRESS, which is what the
        # declaration's `host_refs` must be: a target walks that list to check
        # each reference is wholly wired and to bind it to an arena of its own,
        # and a field already carrying a handle needs neither. Emitted beside
        # the full list rather than derived from it by the driver, so the two
        # cannot disagree about which fields have moved.
        host_ref_offsets = ", ".join(
            str(f.offset) for f in entry.fields
            if f.kind == "handle" and not _is_migrated(entry, f, migrated))
        out.append(f"\n/// The entry point's name, identical in all four "
                   f"renderings.\n"
                   f'pub const {upper}_NAME: &str = "{entry.entry_symbol}";\n')
        if entry.is_group:
            # THE DRIVER HAS TO KNOW THE SHAPE, because the two extents are not
            # interchangeable: dispatching this one as ELEMENTS would launch
            # `count` threads over `count` groups' worth of work and read the
            # group index as a thread index. The declaration is the only place
            # that knows, so it says so here rather than leaving the driver to
            # infer it from the kernel's name.
            out.append(f"\n/// This entry is GROUP-shaped: one group per "
                       f"element, the group\n"
                       f"/// width chosen by the driver, and the guard on the "
                       f"GROUP index. It must\n"
                       f"/// be dispatched with `EXTENT_GROUPS`; the "
                       f"element extent would read\n"
                       f"/// its group index as a thread index.\n"
                       f"pub const {upper}_IS_GROUP: bool = true;\n")
            scratch_bytes = sum(SCRATCH_TYPES[base][4] * count
                                for _, base, count in entry.scratch)
            out.append(f"\n/// Static group-local scratch the entry point "
                       f"declares for itself, in\n"
                       f"/// bytes. It is NOT dynamic scratch: a dispatch does "
                       f"not size it and must\n"
                       f"/// not ask for any.\n"
                       f"pub const {upper}_SCRATCH_BYTES: u32 = "
                       f"{scratch_bytes};\n")
        count_name = entry.count_field.name
        out.append(f"\n/// Byte offsets of the buffer fields, which is what "
                   f"lets an untyped\n"
                   f"/// dispatch trace substitute an `AllocLabel` for a "
                   f"handle's bytes and so\n"
                   f"/// compare two backends whose arena packing differs.\n"
                   f"pub const {upper}_HANDLE_OFFSETS: &[u16] = "
                   f"&[{offsets}];\n")
        aligns = ", ".join(str(pointee_align(f.base)) for f in entry.fields
                           if f.kind == "handle")
        out.append(f"\n/// The alignment each buffer field's POINTEE asks for, "
                   f"positionally\n"
                   f"/// matching `_HANDLE_OFFSETS`. A host validator needs "
                   f"both to check\n"
                   f"/// `off % align == 0` without knowing the pointee's type, "
                   f"and the\n"
                   f"/// generated C++ asserts each value against `alignof(T)` "
                   f"so a wrong one\n"
                   f"/// fails the build rather than weakening the check.\n"
                   f"pub const {upper}_HANDLE_ALIGNS: &[u16] = "
                   f"&[{aligns}];\n")
        out.append(f"\n/// Byte offsets of the buffer fields this record still "
                   f"names by ADDRESS,\n"
                   f"/// which is what a `KernelDecl`'s `host_refs` takes. It "
                   f"shrinks as the\n"
                   f"/// driver's buffers become device allocations and is "
                   f"empty when they all\n"
                   f"/// have; a record with an empty one is dispatchable on a "
                   f"backend that\n"
                   f"/// resolves a handle and cannot resolve an address.\n"
                   f"pub const {upper}_HOST_REF_OFFSETS: &[u16] = "
                   f"&[{host_ref_offsets}];\n")
        out.append(f"\n// Safety: #[repr(C)], no padding (every field is 4- or "
                   f"16-byte wide at\n"
                   f"// 4-byte alignment), no f64, no float3 or float3x3, and "
                   f"the layout is\n"
                   f"// asserted above against the same offsets the three C++ "
                   f"renderings assert.\n"
                   f"unsafe impl KernelArgs for {entry.record_rust} {{\n"
                   f"    const KERNEL: KernelId = id::{upper};\n"
                   f"    const NAME: &'static str = {upper}_NAME;\n"
                   f"    fn guard_count(&self) -> Option<u32> {{\n"
                   f"        Some(self.{count_name})\n"
                   f"    }}\n"
                   f"    fn handle_offsets() -> &'static [u16] {{\n"
                   f"        {upper}_HANDLE_OFFSETS\n"
                   f"    }}\n"
                   f"    fn handle_aligns() -> &'static [u16] {{\n"
                   f"        {upper}_HANDLE_ALIGNS\n"
                   f"    }}\n"
                   f"}}\n")
    return "".join(out)


def _is_migrated(entry, field, migrated):
    """Whether this buffer field is already a device allocation.

    Named as `<record>.<field>` with the RUST record name, because that is what
    the driver's own call site spells and so the one name a reader can grep for
    on both sides of the seam.
    """
    return f"{entry.record_rust}.{field.name}" in migrated


def _access_note(field):
    """How the entry point reaches this field, as Rust doc lines."""
    notes = []
    if field.access == "element":
        notes.append("The entry reads element `index` of this buffer.")
    elif field.access == "offset":
        notes.append(f"The entry hands the body this buffer advanced by "
                     f"{field.stride} * `index`.")
    if field.access == "indices":
        notes.append(f"This element's own index list: {field.stride} slots "
                     f"at `index`.")
        notes.append("The entry reads them and checks each against the "
                     "bound;")
        notes.append("the body never sees a slot.")
    if field.access == "indirect":
        notes.append("The entry hands the body this buffer's elements at "
                     "the slots the")
        notes.append("index list names, as one argument per slot.")
    if field.is_sink:
        notes.append("The body's return value is written here at `index`.")
    return notes


def _pointee(field, index):
    """How this target spells the type a handle field addresses.

    A scalar pointee is spelled per target (`unsigned` is `uint` on MSL); a
    struct pointee is one identifier shared by all three, which is why
    `_pod_bytes` refuses a qualified or templated name.
    """
    if field.pod_bytes is None:
        return SCALAR_TYPES[field.base][index]
    return field.base


def _pod_asserts(entries):
    """One size assertion per struct pointee named in the file.

    A struct pointee's layout is the one thing about a generated entry that
    this script cannot compute, so the declaration states it and all three C++
    renderings assert the SAME literal. A type the Metal shader compiler lays
    out differently from host C++ then fails to compile in the shader instead
    of reading the wrong bytes, which on a backend that never faults is a
    silent wrong answer with plausible floats.
    """
    seen = {}
    out = []
    for entry in entries:
        for f in entry.fields:
            if f.pod_bytes is None:
                continue
            if f.base in seen:
                if seen[f.base] != f.pod_bytes:
                    raise KernelGenError(
                        f"{entry.name}: '{f.base}' is declared "
                        f"[[seam::pod({f.pod_bytes})]] here and "
                        f"[[seam::pod({seen[f.base]})]] earlier in the same "
                        f"file. One type, one size")
                continue
            seen[f.base] = f.pod_bytes
            out.append(f"static_assert(sizeof({f.base}) == {f.pod_bytes},\n"
                       f'              "{f.base} is not the '
                       f'[[seam::pod({f.pod_bytes})]] this entry declares");\n')
            # AND ITS ALIGNMENT, which is a RANGE rather than the single value
            # this asserted until 2026-08-23, because the arena serves more than
            # four and the old equality said otherwise.
            #
            # WHAT EACH PATH ACTUALLY PROMISES, read at the source rather than
            # assumed: the CUDA arena takes the alignment as an ARGUMENT,
            # validates it is a power of two of at least four, aligns the offset
            # up to it, and `compute::arena::resolve<T>` asserts
            # `handle.off % alignof(T) == 0` at RUN TIME as a live backstop; the
            # host arena's base is `ARENA_ALIGN` (64) and `allocate_span` honors
            # any request up to that and refuses a larger one; and the host
            # backend binds a driver buffer as `(k, 0)` with base `k` the
            # reference's own address, so the pointee's alignment is the Rust
            # array's own.
            #
            # THE EQUALITY WAS NOT PROTECTING ANYTHING THE RANGE GIVES UP. It
            # excluded `AABB`, which is `alignas(32)` so its random tree loads
            # are one 32-byte sector each, and the Metal shader ALREADY resolves
            # `device AABB *` out of an arena handle at 56 hand-written sites.
            # What both forms leave to the caller is the same thing, that the
            # allocation really is aligned to what the pointee asks; the
            # equality merely narrowed which pointees could ask.
            out.append(f"static_assert(alignof({f.base}) >= {HANDLE_ALIGN} &&\n"
                       f"                  alignof({f.base}) <= "
                       f"{ARENA_MAX_ALIGN},\n"
                       f'              "{f.base} asks an alignment outside the '
                       f'range an arena serves");\n')
            # AND THAT IT IS THE VALUE THE RUST TABLE CARRIES, which is what
            # lets a host validator check `off % align == 0` without knowing
            # the pointee's type. Python cannot compute `alignof`, so the table
            # guesses and this equality is the guess's audit.
            out.append(f"static_assert(alignof({f.base}) == "
                       f"{pointee_align(f.base)},\n"
                       f'              "{f.base}: POINTEE_ALIGN in '
                       f'seam/kernelgen.py disagrees with alignof; fix the '
                       f'table rather than this assertion");\n')
    return out


def _record_cpp(entry, handle_type, scalar_type):
    lines = [f"struct {entry.record_cpp} {{\n"]
    for f in entry.fields:
        ty = handle_type if f.kind == "handle" else scalar_type(f)
        lines.append(f"    {ty} {f.name};\n")
    lines.append("};\n")
    return "".join(lines)


def _resolve_cu(field, prefix):
    fn = "resolve_const" if field.is_const else "resolve"
    ty = _pointee(field, 0)
    return f"compute::arena::{fn}<{ty}>({prefix}{field.name})"


def _resolve_metal(field, prefix):
    const = "const " if field.is_const else ""
    ty = _pointee(field, 1)
    return (f"(device {const}{ty} *)\n"
            f"        ARENA_PTR({prefix}{field.name}.arena, "
            f"{prefix}{field.name}.off)")


def _resolve_cpp(field, prefix):
    const = "const " if field.is_const else ""
    ty = _pointee(field, 2)
    return (f"reinterpret_cast<{const}{ty} *>(\n"
            f"        seam_arena_base[{prefix}{field.name}.arena] + "
            f"{prefix}{field.name}.off)")


def _hoist(entry, resolve, prefix, target_index, address_space=""):
    """One local per handle field, resolved once above the call.

    EVERY RENDERING RESOLVES ONCE AND NAMES THE RESULT, which is what makes an
    element gather and a scatter to the same buffer one resolution rather than
    two: `x[i] = f(x[i], ...)` written against a resolution expression would
    resolve twice in one statement, and on CUDA each resolution carries the
    allocator's live asserts. The local is named for its field, which is why no
    parameter may take a generator-owned name.
    """
    lines = []
    for f in entry.handles:
        const = "const " if f.is_const else ""
        ty = _pointee(f, target_index)
        lines.append(f"    {address_space}{const}{ty} *{f.name} = "
                     f"{resolve(f, prefix)};\n")
    return lines


def _slots(entry, prefix, indent, assert_line, unsigned="unsigned"):
    """The slot read and its bounds check, or nothing when there is no list.

    Every slot of the element's index list is read into thread storage and
    compared against the [[seam::bound]] BEFORE any [[seam::through]] buffer is
    subscripted, and a failing element asserts on the target's own channel and
    returns without calling the body. The order is load-bearing on Metal, where
    an out-of-bounds read does not fault and yields 0.0: checking after the
    read would report a slot the kernel had already acted on.

    The check is a fold into one flag rather than a return inside the loop, so
    every slot is examined and the assert fires once per element however many
    of them are wrong.
    """
    if entry.indices_field is None:
        return ""
    pad = " " * indent
    slots = entry.indices_field.stride
    listing = entry.indices_field.name
    return (f"{pad}{unsigned} seam_slot[{slots}];\n"
            f"{pad}bool seam_slots_valid = true;\n"
            f"{pad}for ({unsigned} seam_k = 0; seam_k < {slots}u; ++seam_k) {{\n"
            f"{pad}    seam_slot[seam_k] = "
            f"{listing}[{slots} * {entry.index_name} + seam_k];\n"
            f"{pad}    seam_slots_valid = seam_slots_valid &&\n"
            f"{pad}        seam_slot[seam_k] < "
            f"{prefix}{entry.bound_field.name};\n"
            f"{pad}}}\n"
            f"{pad}{assert_line}\n"
            f"{pad}if (!seam_slots_valid) {{\n"
            f"{pad}    return;\n"
            f"{pad}}}\n")


def _call(entry, prefix, indent=4, thread_locals=False, statement=False):
    """The statement that calls the neutral body, with its arguments.

    Two forms, and the difference is the whole of what [[seam::scatter]] adds:
    without one the call is made through `return`, so a body returning a value
    is refused by the compiler; with one the call is the right-hand side of an
    assignment to the sink's element, so a body returning VOID is refused
    there. Neither refusal is this script's to make, because the body's
    signature lives in a file it does not parse.

    `statement` drops the `return` from the first form, for the one caller that
    has to emit code AFTER the call: the MSL rendering's copy-out. The refusal
    it gives up there is kept by the other renderings of the same declaration.
    """
    pad = " " * indent
    args = _forward(entry, prefix, indent=indent + 4,
                    thread_locals=thread_locals)
    sink = entry.sink
    if sink is None:
        lead = "" if statement else "return "
        return f"{pad}{lead}{entry.name}(\n{args});\n"
    return (f"{pad}{sink.name}[{entry.index_name}] = {entry.name}(\n"
            f"{args});\n")


def _thread_locals(entry, indent=4):
    """The MSL rendering's thread-space copy of every element it passes.

    METAL IS THE ONLY ONE OF THE THREE TARGETS WITH MORE THAN ONE ADDRESS
    SPACE, and that is the whole reason this exists. A resolved buffer is a
    `device` pointer there, so `buffer[index]` is a `device` lvalue, and a body
    parameter declared `[[seam::thread]] const T &` renders as `thread const T
    &`: MSL refuses to bind the one to the other, with "cannot bind reference
    in address space 'device' to object in default address space", naming the
    BODY's line rather than the entry's. CUDA and host C++ have one address
    space and no such question, so their renderings pass the subscript
    directly and are unchanged by this.

    The copy is unconditional rather than taken only for a struct, because
    this script does not parse the body and so cannot see whether a parameter
    is a reference or a value. A copy is correct either way and a scalar's is
    free; guessing would be wrong exactly when the body is edited later.

    It is also what the hand-written Metal kernels this form replaces already
    do: `metal/face_math.mm` reads its element into a local array before
    calling the same body.

    A MUTABLE GATHER IS COPY-IN, COPY-OUT, and the copy cannot be const there.
    A buffer declared without `const` and without [[seam::scatter]] is one the
    body WRITES THROUGH, an out-parameter taken as `[[seam::thread]] T &`, and
    two things follow. A `const` local does not bind to it at all, which is a
    compile error naming the body's line; and a local the body wrote is the
    answer, so it has to reach the buffer, which `_thread_writebacks` does
    after the call. Aliasing cannot make that wrong: an element entry covers
    one element per thread, so no two threads name the same slot.
    """
    pad = " " * indent
    lines = []
    for kind, name in entry.forward:
        if kind != "handle":
            continue
        field = next(f for f in entry.fields if f.name == name)
        pointee = _pointee(field, 1)
        const = "const " if _reads_only(field) else ""
        if field.access == "element":
            lines.append(f"{pad}{const}{pointee} {THREAD_LOCAL}{name} = "
                         f"{name}[{entry.index_name}];\n")
        elif field.access == "indirect":
            for k in range(entry.indices_field.stride):
                lines.append(f"{pad}{const}{pointee} {THREAD_LOCAL}{name}_{k} = "
                             f"{name}[seam_slot[{k}]];\n")
    return "".join(lines)


def _reads_only(field):
    """Whether the body only READS this gathered element.

    A `const` pointee says so outright. A [[seam::scatter]] sink is the other
    case: the body returns its new value and the entry assigns it, so the
    element the body was HANDED is an input however the pointer is spelled.
    Everything else the body writes through.
    """
    return field.is_const or field.is_sink


def _thread_writebacks(entry, indent=4):
    """The MSL rendering's copy-out, one per element the body wrote through.

    Only this target needs it, and for the same reason it needs the copy-in:
    the local the body was handed lives in `thread` and the buffer lives in
    `device`, so nothing has reached the buffer until this runs. CUDA and host
    C++ hand the body the subscript itself and the write has already landed.
    """
    pad = " " * indent
    lines = []
    for kind, name in entry.forward:
        if kind != "handle":
            continue
        field = next(f for f in entry.fields if f.name == name)
        if _reads_only(field):
            continue
        if field.access == "element":
            lines.append(f"{pad}{name}[{entry.index_name}] = "
                         f"{THREAD_LOCAL}{name};\n")
        elif field.access == "indirect":
            for k in range(entry.indices_field.stride):
                lines.append(f"{pad}{name}[seam_slot[{k}]] = "
                             f"{THREAD_LOCAL}{name}_{k};\n")
    return "".join(lines)


def _forward(entry, prefix, indent=8, thread_locals=False):
    """The generated call's argument list, one per line, indented.

    A handle reaches the body in the shape its access attribute named: the
    hoisted base pointer, one element at the thread index, or that base
    advanced by a fixed run. The three spellings are the ones the launcher this
    replaces already wrote, character for character, which is what keeps the
    generated entry from changing an answer.
    """
    by_name = {f.name: f for f in entry.fields}
    parts = []
    for kind, name in entry.forward:
        if kind == "handle":
            field = by_name[name]
            if field.access == "element":
                parts.append(f"{THREAD_LOCAL}{name}" if thread_locals
                             else f"{name}[{entry.index_name}]")
            elif field.access == "offset":
                parts.append(f"{name} + {field.stride} * {entry.index_name}")
            elif field.access == "indirect":
                # THE ONE PARAMETER THAT IS NOT ONE ARGUMENT. A through buffer
                # expands into the N elements the index list names, in slot
                # order, in this parameter's position, which is what lets a
                # body whose signature spells those elements one by one be
                # reached without changing it. The slots were read and checked
                # into thread storage above.
                parts.extend(
                    (f"{THREAD_LOCAL}{name}_{k}" if thread_locals
                     else f"{name}[seam_slot[{k}]]")
                    for k in range(entry.indices_field.stride))
            else:
                parts.append(name)
        elif kind == "scalar":
            parts.append(f"{prefix}{name}")
        else:
            # index, lane, width and scratch are all locals of the entry
            # point, named exactly as the declaration named them.
            parts.append(name)
    pad = " " * indent
    return ",\n".join(f"{pad}{p}" for p in parts)


def _table_banner(target, source_path):
    absolute = os.path.abspath(source_path)
    return (f"// Generated by ppf-cts-compute/seam/kernelgen.py --target "
            f"{target} --emit table from\n// {absolute}. Do not edit.\n"
            f"//\n"
            f"// ONE FRAGMENT OF THE KERNEL TABLE: the rows for the entry "
            f"points this\n"
            f"// neutral source declares, in declaration order. A kernel id "
            f"is a dense\n"
            f"// index over every entry point a backend library carries, so "
            f"no single\n"
            f"// source can know one; the id is the row's POSITION in the "
            f"caller's\n"
            f"// concatenation of these fragments, and the two sides of the "
            f"boundary\n"
            f"// concatenate the same sorted source order. That the two "
            f"agree is checked\n"
            f"// at open, name by name and size by size, rather than assumed.\n")


def _args_include(source_path, target="cu"):
    """The `.args.cuh` this fragment's rows are written against.

    Spelled from the GENERATED ROOT rather than as a bare filename, because a
    fragment is read in two places: beside its own siblings, where a bare name
    would resolve, and inlined into the caller's concatenation at the root,
    where it would not. The generated root is on the include path in both
    cases, so one spelling serves both and the concatenation may be a plain
    `cat`.
    """
    if KERNEL_ROOT is None:
        raise KernelGenError(
            f"{source_path}: --emit table needs --kernel-root, to spell the "
            f"argument header this fragment's rows are written against")
    relative = os.path.relpath(os.path.abspath(source_path), KERNEL_ROOT)
    if relative.startswith(".."):
        raise KernelGenError(
            f"{source_path}: is not inside the kernel root {KERNEL_ROOT}")
    return relative[:-len(KERNEL_SUFFIX)] + ARGS_SUFFIX[target]


def render_table_cu(source_path, entries, target="cu"):
    """The library's half of the kernel table, for one neutral source.

    THE FILE HAS TWO ROLES AND THE CALLER PICKS ONE PER INCLUSION, because a
    row names a record TYPE and a launcher SYMBOL and both have to be declared
    before the array that holds them. A caller therefore includes the
    concatenation twice: once with PPF_BE_TABLE_ARGS_INCLUDES defined, which
    yields the declarations, and once without it inside the array initializer,
    which yields the rows. Splitting instead into two generated files per
    source would double the fragment count and let a build concatenate the two
    halves in different orders, which is the one mistake this whole scheme is
    arranged to make impossible.

    The row carries no id. See the module's TABLE_TARGETS note.
    """
    out = [_table_banner(target, source_path)]
    emitted = [e for e in entries if e.emit_entry]
    if not emitted:
        out.append("//\n// This neutral kernel declares no entry point, so it "
                   "contributes no row.\n")
        return "".join(out)
    out.append("#ifdef PPF_BE_TABLE_ARGS_INCLUDES\n")
    out.append(f'#include "{_args_include(source_path, target)}"\n')
    # WHERE THE HANDLES SIT IN THE RECORD, so the backend can check each one's
    # arena against the LIVE binding table at dispatch. It is emitted in the
    # first inclusion, beside the record it describes, because the rows in the
    # second inclusion name these arrays and a C++ array must be declared
    # before it is used.
    #
    # A record with no handle still gets an array, because a zero-length array
    # is not C++; its row carries a count of zero and the loop does not run.
    for entry in emitted:
        offsets = [f.offset for f in entry.fields if f.kind == "handle"]
        body = ", ".join(str(o) for o in offsets) if offsets else "0"
        out.append(f"static const unsigned {entry.entry_symbol}"
                   f"_seam_handle_offsets[] = {{{body}}};\n")
    out.append("#else\n")
    for entry in emitted:
        handles = len([f for f in entry.fields if f.kind == "handle"])
        offsets_name = f"{entry.entry_symbol}_seam_handle_offsets"
        if entry.is_group:
            # A GROUP LAUNCHER TAKES THE WIDTH, so it is a different function
            # type and cannot share a row shape with the element one. The
            # scratch figure travels with it because a group entry declares
            # its scratch STATICALLY and a dispatch must not ask for any: the
            # caller checks a request against this rather than discovering the
            # refusal at pipeline creation.
            scratch_bytes = sum(SCRATCH_TYPES[base][4] * count
                                for _, base, count in entry.scratch)
            out.append(f"PPF_BE_KERNEL_GROUP({entry.entry_symbol}, "
                       f"{entry.record_cpp}, {entry.launch_symbol}, "
                       f"{scratch_bytes}, {offsets_name}, {handles})\n")
        else:
            out.append(f"PPF_BE_KERNEL({entry.entry_symbol}, "
                       f"{entry.record_cpp}, {entry.launch_symbol}, "
                       f"{offsets_name}, {handles})\n")
    out.append("#endif\n")
    return "".join(out)


def render_table_metal(source_path, entries):
    """The Metal library's half of the kernel table, for one neutral source.

    PURE DATA, WHICH IS THE DIFFERENCE FROM THE `cu` FRAGMENT AND THE REASON
    THIS TARGET EXISTS RATHER THAN REUSING THAT ONE. A CUDA row names a record
    TYPE and a launcher SYMBOL, so its fragment has to include the record's
    header and the caller has to read the concatenation twice, once for the
    declarations and once for the rows. Metal binds an entry point by NAME out
    of a compiled library and copies the argument record as opaque bytes, so a
    row needs no type and no symbol: the name, the record's size, the launch
    shape, and the scratch a group entry declares statically.

    That the size is a LITERAL here rather than a `sizeof` is not a weakening.
    The same parse emits the `static_assert(sizeof(...) == n)` into the argument
    header, so a record that disagrees with this number fails to compile
    wherever that header is read; and `be_open` compares this table against the
    driver's row by row, by name and by size, so two trees that disagree are a
    named refusal at open rather than a dispatch of the wrong bytes.

    The row carries no id. See the module's TABLE_TARGETS note.
    """
    out = [_table_banner("metal", source_path)]
    emitted = [e for e in entries if e.emit_entry]
    if not emitted:
        out.append("//\n// This neutral kernel declares no entry point, so it "
                   "contributes no row.\n")
        return "".join(out)
    for entry in emitted:
        group = "true" if entry.is_group else "false"
        scratch_bytes = sum(SCRATCH_TYPES[base][4] * count
                            for _, base, count in entry.scratch)
        # WHERE `seam_arena_count` SITS, WHICH ONLY THE LIBRARY CAN FILL AND
        # ONLY THIS ROW CAN LOCATE. The field is generator-owned: `ARENA_PTR`
        # resolves an out-of-range arena id to the LAST arena rather than
        # faulting, so the bound has to be live or a wrong id is a silent wrong
        # answer with plausible floats. On the CUDA side the GENERATED LAUNCHER
        # writes it, because there is one; Metal binds an entry point by name
        # and hands it the record as bytes, so its library patches the field
        # itself and needs the offset to do it.
        #
        # `0xffffffff` for a record with no handle, which carries no such field.
        arena_off = next((f.offset for f in entry.fields
                          if f.name == "seam_arena_count"), None)
        arena_off = "0xffffffffu" if arena_off is None else f"{arena_off}"
        out.append(f'PPF_BE_KERNEL_METAL("{entry.entry_symbol}", '
                   f"{entry.size}, {group}, {scratch_bytes}, {arena_off})\n")
    return "".join(out)


def render_thunks_rust(source_path, entries):
    """The `generated_thunk*!` lines for this file's entry points.

    THE MACRO IS THE SAME TWO FACTS THE DECLARATION IS: group-shaped takes the
    width, a `[[seam::diag]]` entry takes the channel, and everything else takes
    neither. The wrapper's name and its argument record follow the entry's name,
    which is what let this be generated at all: 25 launchers were renamed so the
    rule holds without exception.
    """
    out = [_entry_banner("rust", source_path, emit="thunks")]
    emitted = [e for e in entries if e.emit_entry]
    if not emitted:
        out.append("//\n// This neutral kernel declares no entry point.\n")
        return "".join(out)
    for entry in emitted:
        record = "".join(part.capitalize() for part in entry.name.split("_"))
        if entry.is_group:
            macro = "generated_thunk_group"
        elif entry.diag_name:
            macro = "generated_thunk_diag"
        else:
            macro = "generated_thunk"
        out.append(f"{macro}!(\n    launch_{entry.name},\n"
                   f"    kernels::{record}Args,\n"
                   f"    {entry.entry_symbol}\n);\n")
    return "".join(out)


def render_externs_rust(source_path, entries):
    """The Rust `extern` declarations for this file's entry points.

    HAND-WRITING THESE IS WHAT `check-launch-seam.py` EXISTS TO POLICE, and it
    found four live mismatches when it was added: a declaration's arity is a
    function of two facts the generator already computes, whether the entry is
    group-shaped and whether it takes a diagnostic channel, and Rust believes
    the declaration, so a wrong one miscounts arguments at run time rather than
    failing to build. Generating them removes the class rather than checking it.
    """
    out = [_entry_banner("rust", source_path, emit="externs")]
    emitted = [e for e in entries if e.emit_entry]
    if not emitted:
        out.append("//\n// This neutral kernel declares no entry point.\n")
        return "".join(out)
    for entry in emitted:
        out.append(f"    fn {entry.entry_symbol}(\n")
        out.append("        args: *const u8,\n")
        out.append("        arena_base: *const *mut u8,\n")
        if entry.is_group:
            out.append("        group_width: u32,\n")
        out.append("        begin: u32,\n")
        out.append("        end: u32,\n")
        if entry.diag_name:
            out.append("        diag: *mut DiagRecord,\n")
        out.append("    );\n")
    return "".join(out)


def render_table_rust(source_path, entries):
    """The driver's half of the kernel table, for one neutral source.

    A ROW IS A TUPLE RATHER THAN A STRUCT LITERAL, and that is deliberate. The
    seam's `KernelDecl` carries fields no generated row can honestly fill (a
    per-item cost measured for one target's scheduler, whether the caller's own
    references are still host addresses), so a generated `KernelDecl { .. }`
    would have to invent them. What is generated is exactly what the
    declaration knows: the entry point's name, the record's size, where its
    handles sit, and which of the two launch shapes it takes. The caller turns
    a row into a `KernelDecl` in one place, where the id is assigned from the
    row's position.
    """
    out = [_table_banner("rust", source_path)]
    emitted = [e for e in entries if e.emit_entry]
    if not emitted:
        out.append("//\n// This neutral kernel declares no entry point, so it "
                   "contributes no row.\n")
        return "".join(out)
    for entry in emitted:
        offsets = ", ".join(str(f.offset) for f in entry.fields
                            if f.kind == "handle")
        group = "true" if entry.is_group else "false"
        scratch_bytes = sum(SCRATCH_TYPES[base][4] * count
                            for _, base, count in entry.scratch)
        out.append(f'("{entry.entry_symbol}", {entry.size}, '
                   f"&[{offsets}], {group}, {scratch_bytes}),\n")
    return "".join(out)


ENTRY_RENDERERS = {
    "cu": render_entry_cu,
    "metal": render_entry_metal,
    "cpp": render_entry_cpp,
    # SHARES the CUDA renderer rather than forking it. See DEVICE_CXX.
    "hip": functools.partial(render_entry_cu, target="hip"),
}

# One target, and the table exists rather than a branch so that adding a second
# is adding a row. See ARGS_TARGETS for why there is only one.
ARGS_RENDERERS = {
    "cu": render_args_cu,
    "metal": render_args_metal,
    "hip": functools.partial(render_args_cu, target="hip"),
}

# The two sides of one boundary. See TABLE_TARGETS for why there are two and
# why there are not four.
TABLE_RENDERERS = {
    "cu": render_table_cu,
    "metal": render_table_metal,
    "rust": render_table_rust,
    "hip": functools.partial(render_table_cu, target="hip"),
}


def neutral_file_path(source_path):
    """The spelling a `#line` directive gives a neutral source, and therefore
    the string `__FILE__` expands to inside it.

    ONE FUNCTION BECAUSE TWO THINGS HAVE TO AGREE EXACTLY. `preamble` writes
    this into the rendering as `#line 1 "<path>"`, so it is what `__FILE__`
    expands to at every `DIAG_ASSERT` site in the body; `render_diagfile_cu`
    hashes it into the table a device report is resolved through. A difference
    of one character between the two makes every report from that file
    unresolvable, and nothing would say so: the reader would just get the
    number.
    """
    return os.path.abspath(source_path).replace("\\", "/")


def file_id(text):
    """FNV-1a over `text`, mirroring `diagnostics::file_id` in the two device
    backends' `diagnostics.hpp`.

    The device side is a `constexpr` recursion over `__FILE__` and this is the
    host side of the same number. They are two spellings of one algorithm, so a
    change to either is a change to both; the offsets are the standard 32-bit
    FNV-1a basis and prime, which is what makes that safe to say.
    """
    h = 2166136261
    for byte in text.encode("utf-8"):
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return h


def render_diagfile_cu(source_path, target="cu"):
    """One row of the table that turns a device report's file id into a path.

    WHY A TABLE IS NEEDED AT ALL. A device record cannot carry a string, so a
    `DIAG_ASSERT` site records `file_id(__FILE__)`, a hash. A hash is not
    reversible, so without this the host prints the number and the reader is
    left to find the check by hand, which is most of what the report was for.

    Every neutral source gets a row, INCLUDING one that declares no entry
    point: an assert does not need an entry to exist, and a body reached only by
    inclusion still reports under its own `__FILE__`.

    A ROW CARRIES TWO IDS BECAUSE THE TWO BACKENDS DISAGREE ABOUT `__FILE__`,
    and that disagreement is deliberate on both sides. `preamble` writes
    `#line 1 "<absolute path>"`, so under nvcc `__FILE__` is absolute. The ROCm
    recipe then compiles with `-fmacro-prefix-map=<kernel root>/=`, which
    rewrites it to the path relative to that root. One spelling cannot serve
    both, and picking either would leave the other backend's reports
    unresolvable with nothing to say so, so the row holds the hash of each and a
    lookup matches whichever arrives.

    THE STORED STRING IS THE RELATIVE ONE, AND THAT IS NOT COSMETIC.
    `bundle.sh`'s gate C greps every file in a distribution for the build tree's
    path and refuses a hit; a row holding an absolute path would put 93 of them
    in the backend library as data, where stripping does not reach. It also
    reads better in a report, which names a source the way this repository does.
    """
    absolute = neutral_file_path(source_path)
    if not os.path.isfile(source_path):
        raise KernelGenError(
            f"{source_path} is not a file, so it can carry no assert and owes "
            f"no diagnostic-file row")
    root = KERNEL_ROOT or os.path.dirname(os.path.abspath(source_path))
    shown = os.path.relpath(os.path.abspath(source_path), root).replace("\\", "/")
    if shown.startswith("../"):
        raise KernelGenError(
            f"{source_path} is outside the kernel root {root}, so the path a "
            f"report would show for it is not a name in the neutral tree")
    for bad, name in (('"', "a quote"), ("\\", "a backslash"), ("\n", "a newline")):
        if bad in shown:
            raise KernelGenError(
                f"the path {shown!r} contains {name}, so it cannot be written "
                f"as the C string literal a diagnostic-file row is")
    return (f"// Generated by ppf-cts-compute/seam/kernelgen.py --target "
            f"{target} --emit diagfile from\n// {absolute}. Do not edit.\n"
            f"//\n"
            f"// ONE ROW of the diagnostic file table: the two ids a "
            f"`DIAG_ASSERT` in this\n"
            f"// source can record, and the path a report shows for it. nvcc "
            f"sees the\n"
            f"// absolute path this file's rendering names in its `#line` "
            f"directive; the ROCm\n"
            f"// recipe rewrites that to a root-relative one with "
            f"`-fmacro-prefix-map`. The\n"
            f"// row carries both hashes so it serves either, and STORES the "
            f"relative path,\n"
            f"// because gate C refuses the build tree's path in any file a "
            f"release carries.\n"
            f'{{ {file_id(absolute)}u, {file_id(shown)}u, "{shown}" }},\n')


def preamble(target, source_path):
    """The two lines that precede a `cu` or `cpp` body.

    The second is what makes a diagnostic name the neutral file. `#line 1`
    numbers the NEXT line, so physical line 3 of the generated file is line 1 of
    the neutral source and every line below it matches.

    The `metal` target gets no preamble: ../metal/shader_compiler.mm injects
    its own `#line` as it splices the segment and maps diagnostics by counting
    lines from there, so a second directive inside the body would shift that
    mapping.
    """
    if target == "metal":
        return ""
    # A `#line` PATH IS A STRING LITERAL, so a Windows separator inside it is an
    # ESCAPE. Measured on amdclang 23 building on Windows: a path through
    # `...\\utility\\...` fails with `\\u used with no following hex digits`,
    # which is a hard error and not a warning, while `\\m` and `\\d` are merely
    # warnings. So the failure depends on the first letter of a directory name,
    # which is why 41 of 80 entries compiled and the rest did not.
    #
    # Forward slashes are accepted by every compiler this tree uses, MSVC-hosted
    # clang included, and they need no escaping, so the separator is normalized
    # rather than the backslash escaped, which `neutral_file_path` does. The
    # banner is normalized with it so the two name the same file.
    absolute = neutral_file_path(source_path)
    return (f"// Generated by ppf-cts-compute/seam/kernelgen.py --target {target} from "
            f"{absolute}. Do not edit.\n"
            f"#line 1 \"{absolute}\"\n")


GENERATED_BANNER = "// Generated by ppf-cts-compute/seam/kernelgen.py"


SYNTHESIS_SCALARS = frozenset(("float", "unsigned", "int", "bool"))


def find_body_entry_spans(path_of_lines, lines, masks):
    """(name, first, last, argument) for each BODY that declares itself an entry.

    A KERNEL MAY CARRY ITS OWN ENTRY, which is what the reference does: one
    `DISPATCH_START(n)` and one lambda, not a body and a second declaration
    restating the body's parameters as pointers. A `[[seam::device_fn]]`
    definition carrying `[[seam::entry(...)]]` IS its own declaration, and the
    record is synthesized from its signature below.

    The argument names the THREAD INDEX parameter, or is empty when the launch
    supplies an index the body does not take. The thread count is implicit: it
    is not a parameter of the body at all, so there is nothing to name.
    """
    found = []
    for name, first, last in body_signature_spans(lines, masks):
        # ON THE SIGNATURE ITSELF IS A SILENT NO-ENTRY. The declaration scan
        # skips a definition and this one reads the line ABOVE, so an attribute
        # written between them belongs to neither and the kernel would simply
        # not exist, with nothing said. Refused by name instead.
        head = " ".join(code_only(lines[i], masks[i])
                        for i in range(first - 1, last))
        if re.search(r"\[\[\s*" + ATTRIBUTE_NAMESPACE + r"::(entry|args)\b",
                     head):
            fail(path_of_lines, first, 1,
                 f"'{name}' carries the entry attribute on its own signature. "
                 f"Write it on the line ABOVE the definition: the attribute "
                 f"takes parentheses, and every scan that walks a parameter "
                 f"list would read them as that list")
        # THE ATTRIBUTE SITS ON ITS OWN LINE ABOVE THE SIGNATURE, not inside it.
        # `[[seam::entry(index)]]` carries parentheses, and every scan that
        # walks a body's parameter list would read them as that list: the plan
        # builder placed an address space inside the attribute's argument on the
        # first try. Above the signature it is outside all of them.
        above = first - 2
        while above >= 0 and not code_only(lines[above], masks[above]).strip():
            above -= 1
        if above < 0:
            continue
        match = re.match(r"\s*\[\[\s*" + ATTRIBUTE_NAMESPACE +
                         r"::entry\s*(?:\(([^)]*)\))?\s*\]\]\s*$",
                         code_only(lines[above], masks[above]))
        if match:
            found.append((name, first, last, (match.group(1) or "").strip(),
                          above + 1))
    return found


def _synthesized_parameter(piece):
    """One body parameter as an entry declaration would spell it.

    A body takes ONE ELEMENT by reference and the ARRAY by pointer, and a
    declaration names the buffer either way, so both become a pointer here and
    the gather inference decides which is which from the body afterwards. A
    value of scalar type is a scalar field, and the diagnostic handle keeps its
    own type.
    """
    head = piece.split("[")[0].strip()
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", head)
    if match is None:
        return None
    name = match.group(1)
    before = head[:match.start(1)]
    base = " ".join(t for t in before.replace("*", " ").replace("&", " ").split()
                    if t != "const")
    qualifier = "const " if "const" in before else ""
    # POINTER OR REFERENCE FIRST. A `float *` is a BUFFER of floats and a plain
    # `float` is a scalar field, so testing the base type before the indirection
    # turns every float buffer into a scalar, which renders a record that
    # compiles and reads the wrong bytes.
    if "*" in before or "&" in before:
        return f"{qualifier}{base} *{name}"
    if base in SYNTHESIS_SCALARS or base == DIAG_TYPE_NAME:
        return f"{base} {name}"
    return f"{qualifier}{base} *{name}"


def synthesize_declaration(path, lines, masks, name, first, last, argument):
    """The entry declaration a self-declaring body stands for."""
    blob = " ".join(code_only(lines[i], masks[i]) for i in range(first - 1, last))
    opened = blob.index("(", blob.index(name))
    inner = blob[opened + 1:blob.rindex(")")]
    pieces = []
    for piece, _offset in _split_top_level(inner):
        piece = ATTRIBUTE_GROUP_RE.sub(" ", piece).strip()
        if not piece:
            continue
        spelled = _synthesized_parameter(piece)
        if spelled is None:
            fail(path, first, 1,
                 f"'{name}' declares itself an entry and this script cannot "
                 f"spell its parameter '{piece}' as a record field. Write the "
                 f"declaration out instead")
        pieces.append(spelled)
    # THE ARGUMENT LIST NAMES BY WHAT THE NAME IS, not by position. A name the
    # body takes as a parameter is the THREAD INDEX; a name it does not take is
    # the DESTINATION for its return value, which is a record field the body has
    # no parameter for and so cannot be read off the signature. That is the same
    # property the sink inference uses to find it once it is declared.
    taken = set()
    for piece in pieces:
        taken.add(piece.rsplit(None, 1)[-1].lstrip("*"))
    index_name = sink_name = None
    for named in [n.strip() for n in argument.split(",") if n.strip()]:
        if named in taken:
            if index_name is not None:
                fail(path, first, 1,
                     f"'{name}' names '{index_name}' and '{named}' as the "
                     f"thread index. A launch supplies one")
            index_name = named
        else:
            if sink_name is not None:
                fail(path, first, 1,
                     f"'{name}' names '{sink_name}' and '{named}' as the "
                     f"destination. One call returns one value")
            sink_name = named
    returns = body_return_type(lines, masks, name)
    if sink_name is not None and returns in (None, "void"):
        fail(path, first, 1,
             f"'{name}' names '{sink_name}' as the destination and returns "
             f"void, so there is no value to put there")
    if sink_name is None and returns not in (None, "void"):
        fail(path, first, 1,
             f"'{name}' returns {returns} and names no destination for it. A "
             f"self-declaring body names the buffer its value is written to, "
             f"because that buffer is not one of its parameters")
    tail = ([f"    {returns} *{sink_name},"] if sink_name else [])
    head = f"[[seam::entry(count{', ' + index_name if index_name else ''})]]"
    return ([f"{head} void {name}("]
            + [f"    {piece}," for piece in pieces]
            + tail
            + ["    unsigned count);"])


def read_source(source_path):
    """The neutral file, its lines, its per-line masks and its entry spans.

    Shared by both emissions, so the entry path and the body path agree on
    where a declaration starts and ends by construction rather than by two
    scans that can disagree.
    """
    text = open(source_path, "r", encoding="utf-8").read()
    # The host rendering of a kernel carries the same `.kernel.cpp` name as the
    # neutral source, so a build whose generated tree sits inside the source
    # tree could offer one back as an input. The banner is what makes that
    # loud instead of a second rendering of an already rendered file.
    if text.startswith(GENERATED_BANNER):
        raise KernelGenError(
            f"{source_path}:1: this is a RENDERED kernel, not a neutral "
            f"source. Point the rule at the *.kernel.cpp under cpp/, and keep "
            f"the generated tree out of the search that finds sources")
    mask = classify(text, source_path)
    if len(mask) != len(text):
        raise KernelGenError(
            f"{source_path}: internal error: the lexical mask is "
            f"{len(mask)} characters against {len(text)} of source")

    lines = text.split("\n")
    masks = mask.split("\n")
    if len(lines) != len(masks):
        raise KernelGenError(
            f"{source_path}: internal error: {len(lines)} lines of source "
            f"against {len(masks)} of mask")

    spans = find_entry_spans(source_path, lines, masks)
    body_entries = find_body_entry_spans(source_path, lines, masks)
    in_entry = set()
    for first, last in spans:
        in_entry.update(range(first, last + 1))
    # A SELF-DECLARING BODY carries the entry family on its own signature, so
    # those lines are inside a declaration for the purpose of this check.
    for _name, first, last, _argument, attr_line in body_entries:
        in_entry.update(range(first, last + 1))
        in_entry.add(attr_line)
    for lineno, (line, line_mask) in enumerate(zip(lines, masks), start=1):
        check_line(source_path, lineno, line, line_mask,
                   in_entry=lineno in in_entry)
    entries = [parse_entry(source_path, lines, masks, first, last)
               for first, last in spans]
    # THE SYNTHESIZED DECLARATION IS PARSED IN THIS FILE'S CONTEXT, appended
    # after the source rather than parsed alone, because the gather and sink
    # inferences read the BODY out of these same lines. Parsed in isolation
    # they see no body and every buffer falls back to a base pointer.
    for name, first, last, argument, _attr_line in body_entries:
        synthetic = synthesize_declaration(source_path, lines, masks,
                                           name, first, last, argument)
        extended = lines + synthetic
        extended_masks = masks + [classify(line + "\n", source_path)[:-1]
                                  for line in synthetic]
        try:
            made = parse_entry(source_path, extended, extended_masks,
                               len(lines) + 1, len(extended))
            # THE SYNTHESIZED SPAN POINTS PAST THE FILE. Give the entry the
            # BODY's position instead: it is where the declaration now lives,
            # so it is what orders the entries and what a diagnostic should
            # name. A kernel id is a table INDEX derived from that order.
            made.first_line, made.last_line = first, last
            # A SELF-DECLARING ENTRY OWNS NO DECLARATION TEXT. Its span is the
            # BODY, which orders it and which a diagnostic should name, and the
            # body rendering must NOT neutralize it: doing so deletes the kernel
            # from the rendering while every record still looks right.
            made.self_declared = True
            # The attribute line carries no code and is where the marker goes,
            # so the host rendering still says one entry lives here and the two
            # recognizers in `build.rs` keep counting the same thing.
            made.attribute_line = _attr_line
            entries.append(made)
        except KernelGenError as exc:
            # The position in the message points into the synthesized text,
            # which is not in the file. Name the body instead, which is.
            detail = str(exc).split(": ", 1)[-1]
            fail(source_path, first, 1,
                 f"'{name}' declares itself an entry and the declaration that "
                 f"stands for is refused: {detail}")
    # ONE ORDER, THE SOURCE'S. A declaration and a self-declaring body are two
    # spellings of the same thing, so entries are ordered by where they sit in
    # the file rather than by which spelling produced them; the canonical kernel
    # order every generated table is built from is that order.
    entries.sort(key=lambda e: e.first_line)
    seen = {}
    for entry in entries:
        if entry.name in seen:
            fail(source_path, entry.first_line, 1,
                 f"a second entry declaration for '{entry.name}'. One "
                 f"declaration renders four artifacts; two declarations are a "
                 f"mirror pair again")
        seen[entry.name] = entry
    # ONE TYPE, ONE SIZE, checked at the PARSE rather than inside a rendering.
    # It is a property of the declarations in this file, not of any one target,
    # so a file that contradicts itself must be refused for every target and
    # for both emit kinds. Checking it only where the C++ size assertions are
    # written would narrow it to one of the two halves of the args/entry split.
    _pod_asserts(entries)
    return text, lines, masks, entries


TWIN_DECL_RE = re.compile(r"\[\[seam::(cooperative|serial)\]\]")


def twin_spans(source_path, lines, masks):
    """Every `[[seam::cooperative]]` / `[[seam::serial]]` body, by line span.

    THE SPAN ENDS AT THE FUNCTION'S CLOSING BRACE, found by counting braces
    outside comments and string literals, which is what `masks` is for. A body
    is neutralized whole or not at all: leaving its signature and dropping its
    statements would render a declaration with no definition, and the link error
    would name the symbol rather than the fork.

    EVERY COOPERATIVE BODY OWES A SERIAL TWIN OF THE SAME NAME, which rule
    (1-LANE) states and this enforces: a cooperative body alone renders to
    nothing on the host target, so the first host caller fails on a missing
    symbol far from the declaration that caused it.
    """
    found = []
    for lineno, (line, line_mask) in enumerate(zip(lines, masks), start=1):
        m = TWIN_DECL_RE.search(line)
        if not m:
            continue
        if line_mask[m.start()] != "c":
            continue
        kind = m.group(1)
        # The name is the identifier before the parameter list, which for these
        # bodies sits on the declaration's own line or the next one.
        name = None
        depth = 0
        opened = False
        last = None
        for scan in range(lineno, len(lines) + 1):
            text = lines[scan - 1]
            mask = masks[scan - 1]
            if name is None:
                nm = re.search(r"([a-z_][a-z0-9_]*)\s*\(", text)
                if nm and mask[nm.start(1)] == "c":
                    name = nm.group(1)
            for column, ch in enumerate(text):
                if mask[column] != "c":
                    continue
                if ch == "{":
                    depth += 1
                    opened = True
                elif ch == "}":
                    depth -= 1
                    if opened and depth == 0:
                        last = scan
                        break
            if last is not None:
                break
        if last is None:
            raise KernelGenError(
                f"{source_path}:{lineno}: a [[seam::{kind}]] body has no "
                f"closing brace this script can find. A twin is neutralized "
                f"whole, so its extent has to be exact")
        found.append((kind, name, lineno, last))
    names = {}
    for kind, name, first, _ in found:
        names.setdefault(name, {})[kind] = first
    for name, kinds in names.items():
        if len(kinds) != 2:
            have = next(iter(kinds))
            want = "serial" if have == "cooperative" else "cooperative"
            raise KernelGenError(
                f"{source_path}: `{name}` is declared [[seam::{have}]] with no "
                f"[[seam::{want}]] twin. Rule (1-LANE) admits a second body for "
                f"LANE COOPERATION only and requires both: a cooperative body "
                f"alone renders to nothing on the host target, and the first "
                f"host caller then fails on a missing symbol a long way from "
                f"this line")
    return found


# ---------------------------------------------------------------------------
# INFERRING A BODY'S ADDRESS SPACES.
#
# A body parameter's address space is not the author's choice: it is decided by
# how the ENTRY hands the buffer over. A `[[seam::gather]]` element arrives as a
# copy in thread space, a strided output is the destination advanced to this
# element's own slot and stays in device memory, and a `[[seam::scratch]]` array
# is threadgroup. Writing it again in the body is restating the entry, and
# getting it wrong compiles clean under nvcc and under a host C++ compiler and
# fails the Metal shader compile at RUN TIME, which is why it was worth removing
# from the authoring surface rather than only checking.
#
# WHAT IS INFERRED IS ONLY WHAT AN ENTRY DECIDES. A body reached from another
# body takes its spaces from its CALLER, which is a call-graph question this
# does not answer; such a parameter still writes its own attribute and
# `check-address-spaces.py` still verifies every one that is written.

# BOTH EXECUTION SPACES DECLARE A BODY. A `[[seam::host_device_fn]]` is a
# neutral body like any other and its parameters need address spaces on Metal
# exactly the same way; scanning only for `device_fn` left 83 of them invisible
# to inference, which is why they still wrote their own.
BODY_DECL_RE = re.compile(r"\[\[seam::(?:host_)?device_fn\]\]")


def body_signature_spans(lines, masks):
    """(name, first_line, last_line) for every `[[seam::device_fn]]` signature.

    The span ends at the parameter list's closing parenthesis, counted outside
    comments and string literals, which is what `masks` is for.
    """
    found = []
    for lineno, (line, line_mask) in enumerate(zip(lines, masks), start=1):
        code = code_only(line, line_mask)
        if not BODY_DECL_RE.search(code):
            continue
        # The name is the identifier before the parameter list, which may sit on
        # this line or a later one when the attributes are long.
        depth = 0
        opened = False
        name = None
        for scan in range(lineno, len(lines) + 1):
            # ATTRIBUTES ARE MASKED OUT FIRST. `[[seam::entry(index)]]` carries
            # its own parentheses, which the name search would read as the
            # parameter list and the depth count would read as a nesting level,
            # so a self-declaring body would be named 'entry' and close its
            # span one line early.
            scan_code = ATTRIBUTE_GROUP_RE.sub(
                " ", code_only(lines[scan - 1], masks[scan - 1]))
            if name is None:
                match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", scan_code)
                if match:
                    name = match.group(1)
            for ch in scan_code:
                if ch == "(":
                    depth += 1
                    opened = True
                elif ch == ")":
                    depth -= 1
            if opened and depth == 0:
                if name:
                    found.append((name, lineno, scan))
                break
    return found


def _split_top_level(text):
    """(piece, offset) for each top-level comma-separated piece."""
    out = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append((text[start:i], start))
            start = i + 1
    out.append((text[start:], start))
    return out


def entry_forwarded_spaces(entry):
    """The address space each forwarded argument needs, in call order.

    `Entry.forward` is what the generated entry point passes and in what order,
    so this survives a body that RENAMES a parameter and expands a through
    buffer into the elements its slot list names.
    """
    by_name = {f.name: f for f in entry.fields}
    scratch = {name for name, _base, _count in entry.scratch}
    out = []
    for kind, name in entry.forward:
        if kind == "scratch" or name in scratch:
            out.append("threadgroup")
        elif kind == "handle":
            field = by_name.get(name)
            if field is None:
                out.append(None)
            elif field.access == "element":
                out.append("thread")
            elif field.access == "indirect" and entry.indices_field is not None:
                out.extend(["thread"] * entry.indices_field.stride)
            elif field.access in ("base", "offset"):
                out.append("device")
            else:
                out.append(None)
        else:
            out.append(None)
    return out


IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# A local whose type carries no `*` or `&` is an OBJECT on the stack, so a
# callee taking it by reference takes a THREAD reference. It is what lets the
# propagation start inside a body that builds its own arguments, which is what
# the traversal visitors do: `aabb_query` declares `box` and hands it to
# `op.test(box, query)`.
# THE TYPE MAY CARRY TEMPLATE ARGUMENTS, and a great many locals do:
# `SMatf<3, N> local_force;` is the shape every elastic body builds its work in.
# A pattern that stopped at the first `<` matched none of them, so their space
# was never known and every callee taking one kept writing its own.
LOCAL_OBJECT_RE = re.compile(
    r"^\s*(?:const\s+)?([A-Za-z_][A-Za-z0-9_:]*(?:\s*<[^;{}=]*>)?)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\]\s*)?(?:=|;|\{)", re.M)
# A local POINTER or REFERENCE aliases something, and inherits its space:
# `const float *m = value + 9 * k;` is device because `value` is.
LOCAL_ALIAS_RE = re.compile(
    r"\b(?:const\s+)?[A-Za-z_][A-Za-z0-9_:<>, ]*?[*&]\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);")
NOT_A_TYPE = frozenset((
    "return", "else", "const", "for", "if", "while", "do", "struct", "case",
    "break", "continue", "switch", "using", "typedef", "template"))


def _body_close(lines, masks, first):
    """The line holding the body's closing brace, or None."""
    depth = 0
    opened = False
    for scan in range(first, len(lines) + 1):
        for ch in code_only(lines[scan - 1], masks[scan - 1]):
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
        if opened and depth == 0:
            return scan
    return None


def _leading_ident(expr):
    match = IDENT_RE.search(expr)
    return match.group(0) if match else None


KERNEL_INCLUDE_RE = re.compile(r'#include\s+"([^"]+\.kernel\.cpp)"')


def _transitive_includers(source_path):
    """Every neutral source that reaches this one through quoted includes.

    A CALL TO THIS FILE'S BODY CAN ONLY APPEAR IN A FILE THAT INCLUDES IT, which
    is what bounds the work: the address spaces a body takes from its CALLERS
    can only be decided by the files that can see it, so the analysis needs
    those and nothing else. Scanning the tree's include lines costs about 8 ms
    and the closures are small, a median of 0 and a maximum of 15, so this is
    cheaper than parsing the tree and gives the same answer for this file's
    bodies as a whole-tree fixpoint would.
    """
    if KERNEL_ROOT is None or not os.path.isdir(KERNEL_ROOT):
        return []
    includes = {}
    for folder, _dirs, names in os.walk(KERNEL_ROOT):
        for name in names:
            if not name.endswith(".kernel.cpp"):
                continue
            path = os.path.join(folder, name)
            try:
                text = _read_text(path)
            except OSError:
                continue
            targets = set()
            for match in KERNEL_INCLUDE_RE.finditer(text):
                candidate = os.path.normpath(
                    os.path.join(os.path.dirname(path), match.group(1)))
                if not os.path.exists(candidate):
                    candidate = os.path.normpath(
                        os.path.join(KERNEL_ROOT, match.group(1)))
                if os.path.exists(candidate):
                    targets.add(os.path.abspath(candidate))
            includes[os.path.abspath(path)] = targets
    reverse = {}
    for includer, targets in includes.items():
        for target in targets:
            reverse.setdefault(target, set()).add(includer)
    # BOTH DIRECTIONS, and the second is not symmetry for its own sake.
    #
    # A file that INCLUDES this one can call this file's bodies, which is the
    # obvious direction. A file this one includes can ALSO call into it: a
    # traversal takes a visitor and calls `op.test(box, query)`, so
    # `aabb_query`, which lives in a file every sweep INCLUDES, is the caller
    # that decides those visitors' parameters. Reading only the includers left
    # every `test` method uninferred.
    start = os.path.abspath(source_path)
    seen, stack = set(), [start]
    while stack:
        current = stack.pop()
        for neighbour in tuple(reverse.get(current, ())) + tuple(
                includes.get(current, ())):
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    seen.discard(start)
    return sorted(seen)


def _read_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def inferred_body_spaces(lines, masks, entries, source_path=None):
    """The insertions alone, for callers that do not render."""
    return inferred_body_plan(lines, masks, entries, source_path)[0]


def inferred_body_plan(lines, masks, entries, source_path=None):
    """{lineno: [(column, space)]} for body parameters that wrote none.

    Keyed on the ORIGINAL line's columns, which is what `substitute_attributes`
    applies them against.

    TWO SOURCES, and the second is why this is a fixpoint rather than a lookup.
    A body an ENTRY wraps takes its spaces from how that entry hands the buffers
    over. A body another BODY calls takes them from the CALL, so the spaces flow
    along the call graph from the entries inward until nothing more moves.

    IT PROPAGATES WITHIN ONE FILE, because that is the unit being rendered: a
    body in an included file gets its space from the INCLUDER, which is not
    present when the included file renders and inserts. Measured over the tree,
    443 of the 589 propagable parameters have all their callers in their own
    file; the rest still write their attribute, and the MSL completeness check
    is what guarantees none is simply missing.
    """
    seeded = {}
    for entry in entries:
        if not entry.emit_entry:
            continue
        spaces = entry_forwarded_spaces(entry)
        if any(spaces):
            seeded.setdefault(entry.name, spaces)

    # Every body in this file, with each parameter's position and any space it
    # wrote for itself.
    bodies = {}
    for name, first, last in body_signature_spans(lines, masks):
        text, locs = _flatten(lines, masks, first, last)
        open_at = text.find("(")
        close_at = text.rfind(")")
        if open_at < 0 or close_at < open_at:
            continue
        params = []
        for piece, offset in _split_top_level(text[open_at + 1:close_at]):
            written = None
            span = None
            for match in ATTRIBUTE_RE.finditer(piece):
                short = match.group(1)[len(ATTRIBUTE_PREFIX):]
                if short in ADDRESS_SPACE:
                    written = short
                    start = open_at + 1 + offset + match.start()
                    stop = open_at + 1 + offset + match.end()
                    if (start < len(locs) and stop - 1 < len(locs)
                            and locs[start][0] == locs[stop - 1][0]):
                        span = (locs[start][0], locs[start][1],
                                locs[stop - 1][1] + 1)
            bare = ATTRIBUTE_RE.sub(" ", piece)
            names = IDENT_RE.findall(bare)
            lead = len(piece) - len(piece.lstrip())
            index = open_at + 1 + offset + lead
            params.append({
                "name": names[-1] if names else None,
                "written": written,
                "span": span,
                "pointer": "*" in bare or "&" in bare,
                "reference": "&" in bare and "*" not in bare,
                "space": written,
                "at": locs[index] if index < len(locs) else None,
            })
        close = _body_close(lines, masks, last)
        flat = _flatten(lines, masks, last, close) if close else ("", [])
        bodies.setdefault(name, []).append({"params": params,
                                            "text": flat[0], "locs": flat[1],
                                            "name": name, "own": True,
                                            "signature": (first, last),
                                            "sig_text": text, "sig_locs": locs,
                                            "close": close})

    # THE FILES THAT CAN CALL INTO THIS ONE, so a body whose space is decided by
    # a caller in another file is inferred too. Their bodies join the graph and
    # their entries join the seeds; only THIS file's parameters are emitted,
    # because insertion happens where the text lives.
    own = len(bodies)
    for other in _transitive_includers(source_path) if source_path else []:
        try:
            other_lines, other_masks, other_entries = read_source(other)[1:]
        except OSError:
            # Not on disk: nothing to read, and nothing this file can be wrong
            # about. A source that IS there and does not parse is the other case.
            continue
        except KernelGenError as exc:
            # A NEUTRAL SOURCE THAT DOES NOT PARSE MUST NOT BE SKIPPED. Its
            # entries are SEEDS for this inference, so dropping it runs the
            # propagation on a partial graph and emits DIFFERENT address spaces
            # with no diagnostic at all. Measured: a broken declaration in
            # `contact_narrow.kernel.cpp` silently turned a `thread const
            # SMatf<...> &` parameter of `contact_stiffness.kernel.cpp` into a
            # `device` one, in a file with no change of its own. The error names
            # both files because the one to fix is not the one being rendered.
            raise KernelGenError(
                f"{source_path}: its includer {other} does not parse, and that "
                f"file's entry declarations SEED the address-space inference "
                f"for this one. Skipping it would render this file with "
                f"different address spaces and no diagnostic. Fix that file "
                f"first: {exc}") from exc
        for entry in other_entries:
            if entry.emit_entry:
                spaces = entry_forwarded_spaces(entry)
                if any(spaces):
                    seeded.setdefault(entry.name, spaces)
        for name, first, last in body_signature_spans(other_lines, other_masks):
            text, locs = _flatten(other_lines, other_masks, first, last)
            open_at = text.find("(")
            close_at = text.rfind(")")
            if open_at < 0 or close_at < open_at:
                continue
            params = []
            for piece, _offset in _split_top_level(text[open_at + 1:close_at]):
                written = None
                for match in ATTRIBUTE_RE.finditer(piece):
                    short = match.group(1)[len(ATTRIBUTE_PREFIX):]
                    if short in ADDRESS_SPACE:
                        written = short
                bare = ATTRIBUTE_RE.sub(" ", piece)
                names = IDENT_RE.findall(bare)
                params.append({
                    "name": names[-1] if names else None,
                    "written": written,
                    "pointer": "*" in bare or "&" in bare,
                    "space": written,
                    "at": None,          # never emitted: another file's text
                })
            close = _body_close(other_lines, other_masks, last)
            body_text = (_flatten(other_lines, other_masks, last, close)[0]
                         if close else "")
            bodies.setdefault(name, []).append({"params": params,
                                                "text": body_text,
                                                "locs": [],
                                                "name": name, "own": False,
                                                "signature": None,
                                                "close": None})
    del own

    for name, spaces in seeded.items():
        for body in bodies.get(name, []):
            for position, space in enumerate(spaces):
                if position < len(body["params"]) and space:
                    param = body["params"][position]
                    if param["pointer"] and param["space"] is None:
                        param["space"] = space
                        # AN ENTRY IS AUTHORITATIVE: it states how the buffer is
                        # handed over, so no call-site proposal may contradict
                        # it or make it ambiguous.
                        param["proposals"] = {space: "its entry"}

    # THE FIXPOINT. Bounded by the number of parameters, and in practice it
    # settles in a handful of rounds; the cap is a guard against a cycle this
    # does not expect rather than a tuning figure.
    structs = struct_fields(lines, masks)
    for _round in range(16):
        moved = 0
        for bodies_named in bodies.values():
            for body in bodies_named:
                moved += _propagate_from(body, bodies)
                if body.get("own"):
                    moved += _propagate_fields(body, structs)
        if not moved:
            break

    # A BODY REACHED TWO WAYS BECOMES TWO FUNCTIONS. See `_specialize`: MSL has
    # no qualifier meaning "either", so the duplication is the only faithful
    # rendering, and it is Metal's alone because CUDA and the host have one
    # address space.
    renames, extra = _specialize(bodies, source_path)

    out = {}
    for name, bodies_named in bodies.items():
        for body in bodies_named:
            if not body.get("own", True):
                continue
            for param in body["params"]:
                if param["written"] or not param["pointer"] or not param["at"]:
                    continue
                space = param["space"]
                if space is None and len(param.get("proposals") or ()) > 1:
                    # Ambiguous, not unreached: `_specialize` owns this one.
                    continue
                if space is None:
                    # NO CALL REACHES IT, so its space follows from what it IS.
                    #
                    # A REFERENCE binds an OBJECT, and an object a body holds
                    # by reference is on the stack: `aabb_join(const AABB &a)`
                    # calls `a.min[dimension]`, and `SMat::operator[]` is
                    # declared in the default address space, so a device object
                    # has no viable overload. That is why the shared headers
                    # spell such a parameter `SM_THREAD` and why defaulting
                    # these to device failed the Metal compile with 167 errors.
                    #
                    # A POINTER names a BUFFER, which is device memory.
                    space = "thread" if param.get("reference") else "device"
                lineno, column = param["at"]
                out.setdefault(lineno, []).append((column, space))

    # THE DEFAULTS SETTLE BEFORE ANYTHING READS THEM. A local aliasing a
    # parameter takes that parameter's space, so a parameter still unresolved
    # when the local is computed leaves the local unresolved too, and the MSL
    # compile then fails on the LOCAL rather than on the parameter.
    for bodies_named in bodies.values():
        for body in bodies_named:
            for param in body["params"]:
                if (param["written"] or not param["pointer"]
                        or param["space"] is not None):
                    continue
                if len(param.get("proposals") or ()) > 1:
                    continue
                param["space"] = ("thread" if param.get("reference")
                                  else "device")

    # A LOCAL POINTER NEEDS AN ADDRESS SPACE IN MSL as much as a parameter
    # does, and the alias rule already knows which: `float *block = dense + k;`
    # is device because `dense` is. Emitting it here is the same answer the
    # author was writing by hand.
    for bodies_named in bodies.values():
        for body in bodies_named:
            if not body.get("own", True):
                continue
            known = _known_locals(body)
            locs = body.get("locs") or []
            for match in LOCAL_ALIAS_RE.finditer(body["text"]):
                local = match.group(1)
                entry_space = known.get(local)
                if not (entry_space and entry_space.get("space")):
                    continue
                head = match.start()
                if head >= len(locs):
                    continue
                # AN ATTRIBUTE ALREADY THERE SITS BEFORE THE TYPE, so it is
                # OUTSIDE this match, which begins at the type. Looking only
                # inside produced `device device float *block`.
                before = body["text"][max(0, match.start() - 48):match.start()]
                if any(attribute.group(1)[len(ATTRIBUTE_PREFIX):]
                       in ADDRESS_SPACE
                       for attribute in ATTRIBUTE_RE.finditer(before)):
                    continue
                if ATTRIBUTE_RE.search(
                        body["text"][match.start():match.end()]):
                    continue
                lineno, column = locs[head]
                out.setdefault(lineno, []).append(
                    (column, entry_space["space"]))

    for fields in structs.values():
        for field in fields:
            if (field["written"] or not field["pointer"]
                    or field["space"] is None or field["at"] is None):
                continue
            out.setdefault(field["at"][0], []).append(
                (field["at"][1], field["space"]))

    # THE COPY THAT STAYS IN PLACE TAKES THE FIRST SIGNATURE. A parameter
    # reached two ways has no single settled space, so the loop above emits
    # nothing for it; the specialization decides it instead, one signature per
    # copy, and this is the one the file keeps.
    for plan in extra:
        pointers = [param for param in plan["body"]["params"]
                    if param["pointer"]]
        for param, space in zip(pointers, plan["order"][0]):
            if not param["at"]:
                continue
            if param["written"]:
                # A SPECIALIZED COPY OVERRIDES what the author wrote, because
                # the signature is what decides it now. Replacing the attribute
                # rather than inserting beside it is what stops the rendering
                # coming out `thread thread const Vec3f &y`.
                if param["span"] and param["written"] != space:
                    plan.setdefault("primary_replacements", []).append(
                        (param["span"], space))
                continue
            out.setdefault(param["at"][0], []).append((param["at"][1], space))
    return out, renames, extra


SPACE_TAG = {"device": "d", "thread": "t", "threadgroup": "g"}


def _call_sites(body, bodies):
    """(callee, name span in source, argument spaces) for each call it makes.

    Run AFTER the fixpoint, when every parameter's space is settled, so a call's
    signature is the tuple its arguments actually carry.
    """
    known = _known_locals(body)
    out = []
    for match in re.finditer(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^<>();]*>\s*)?\(",
            body["text"]):
        callee = match.group(1)
        if callee not in bodies or callee == body["name"]:
            continue
        index, depth = match.end() - 1, 0
        while index < len(body["text"]):
            if body["text"][index] == "(":
                depth += 1
            elif body["text"][index] == ")":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        spaces = []
        for argument, _offset in _split_top_level(body["text"][match.end():index]):
            source = known.get(_leading_ident(argument))
            spaces.append(source["space"]
                          if source and source["pointer"] else None)
        locs = body.get("locs") or []
        start = match.start()
        at = locs[start] if start < len(locs) else None
        out.append((callee, at, len(callee), tuple(spaces)))
    return out


def _known_locals(body):
    """This body's parameters plus the locals whose space follows from them."""
    known = {p["name"]: p for p in body["params"] if p["name"]}
    # STACK OBJECTS FIRST, because an alias reads them: `const float *flat =
    # p.data();` takes its space from `p`, and registering the aliases first
    # left `p` unknown at the moment `flat` needed it. The alias pass then runs
    # to a fixpoint so a chain of them settles whatever order they appear in.
    for match in LOCAL_OBJECT_RE.finditer(body["text"]):
        kind, local = match.group(1), match.group(2)
        if local in known or kind in NOT_A_TYPE:
            continue
        known[local] = {"name": local, "pointer": True, "space": "thread",
                        "written": None}
    for _round in range(8):
        added = 0
        for match in LOCAL_ALIAS_RE.finditer(body["text"]):
            local, expr = match.group(1), match.group(2)
            if local in known:
                continue
            source = known.get(_leading_ident(expr))
            if source and source["pointer"] and source["space"]:
                known[local] = {"name": local, "pointer": True,
                                "space": source["space"], "written": None}
                added += 1
        if not added:
            break
    return known


def _name_position(body):
    """Where the body's own name sits in the source, for renaming in place."""
    text = body.get("sig_text")
    locs = body.get("sig_locs")
    if not text or not locs:
        return None
    for match in re.finditer(
            r"\b" + re.escape(body["name"]) + r"\s*\(", text):
        start = match.start()
        if start < len(locs):
            return locs[start]
    return None


STRUCT_RE = re.compile(r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")


def struct_fields(lines, masks):
    """{struct: [field]} for every struct declared here, in declaration order.

    A POINTER MEMBER NEEDS AN ADDRESS SPACE TOO, and its space is decided by
    what the constructing code puts in it rather than by a call argument: the
    traversal visitors are built by aggregate initialization,
    `AabbPairCollect collect{out, ...}`, so field 0 is whatever `out` is. That
    is the same positional mapping a call has, one step removed.
    """
    text, locs = _flatten(lines, masks, 1, len(lines))
    out = {}
    for match in STRUCT_RE.finditer(text):
        name = match.group(1)
        index, depth = match.end() - 1, 0
        while index < len(text):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        fields = []
        depth = 0
        start = match.end()
        for cursor in range(match.end(), index):
            char = text[cursor]
            if char in "({[<":
                depth += 1
            elif char in ")}]>":
                depth -= 1
            elif char == ";" and depth == 0:
                piece = text[start:cursor]
                start = cursor + 1
                bare = ATTRIBUTE_RE.sub(" ", piece)
                # A member FUNCTION is not a field, and neither is a nested
                # declaration; both carry a parameter list.
                if "(" in bare or not bare.strip():
                    continue
                written = span = None
                for attribute in ATTRIBUTE_RE.finditer(piece):
                    short = attribute.group(1)[len(ATTRIBUTE_PREFIX):]
                    if short in ADDRESS_SPACE:
                        written = short
                        first = match.end() + 0
                        a = start - len(piece) - 1 + attribute.start()
                        b = start - len(piece) - 1 + attribute.end()
                        while b < len(text) and text[b] == " ":
                            b += 1
                        if (a < len(locs) and b - 1 < len(locs)
                                and locs[a][0] == locs[b - 1][0]):
                            span = (locs[a][0], locs[a][1], locs[b - 1][1] + 1)
                names = IDENT_RE.findall(bare)
                lead = len(piece) - len(piece.lstrip())
                at_index = start - len(piece) - 1 + lead
                fields.append({
                    "name": names[-1] if names else None,
                    "written": written,
                    "span": span,
                    "pointer": "*" in bare or "&" in bare,
                    "space": written,
                    "at": locs[at_index] if at_index < len(locs) else None,
                })
        if fields:
            out[name] = fields
    return out


AGGREGATE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?:=\s*)?\{([^{}]*)\}")


FIELD_ASSIGN_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);")
LOCAL_TYPED_RE = re.compile(
    r"^\s*(?:const\s+)?([A-Z][A-Za-z0-9_:]*)(?:\s*<[^;{}=]*>)?\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\]\s*)?(?:=|;|\{)", re.M)


def _propagate_fields(body, structs):
    """Fill a struct's pointer members from how the constructing code fills them.

    TWO SHAPES, and both occur. An aggregate initializer maps positionally,
    `AabbPairCollect collect{out, ...}`; a field assignment names the member,
    `op.x0 = x0;`. The second needs the local's TYPE to know which struct's
    member is being set, which the declaration gives.
    """
    known = _known_locals(body)
    moved = 0
    types = {match.group(2): match.group(1)
             for match in LOCAL_TYPED_RE.finditer(body["text"])}
    for match in FIELD_ASSIGN_RE.finditer(body["text"]):
        local, member, expression = match.groups()
        fields = structs.get(types.get(local, ""))
        if not fields:
            continue
        for field in fields:
            if field["name"] != member or not field["pointer"]:
                continue
            if field["space"] is not None:
                continue
            source = known.get(_leading_ident(expression))
            if source and source["pointer"] and source["space"]:
                field["space"] = source["space"]
                moved += 1
    for match in AGGREGATE_RE.finditer(body["text"]):
        kind = match.group(1)
        fields = structs.get(kind)
        if not fields:
            continue
        arguments = _split_top_level(match.group(3))
        for position, (argument, _offset) in enumerate(arguments):
            if position >= len(fields):
                break
            field = fields[position]
            if not field["pointer"] or field["space"] is not None:
                continue
            source = known.get(_leading_ident(argument))
            if source and source["pointer"] and source["space"]:
                field["space"] = source["space"]
                moved += 1
    return moved


def _specialize(bodies, source_path):
    """Plan one function per address-space signature a body is called with.

    MSL HAS NO GENERIC ADDRESS SPACE. A helper reached with a device pointer at
    one call site and a thread pointer at another cannot be one function there,
    and no attribute an author could write would make it one. CUDA and the host
    have a single address space and compile the original as written, so this
    duplication is Metal's alone.

    IT DOES NOT NEED A TEMPLATE TO HAPPEN. A plain function taking `const float
    *v`, called once with a buffer and once with a stack array, is exactly this
    case.

    Returns (renames, extra), where `renames` maps a source position to the
    specialized name to write there, and `extra` lists the copies to append.
    """
    signatures = {}
    for name, group in bodies.items():
        for body in group:
            for callee, _at, _length, spaces in _call_sites(body, bodies):
                for target in bodies[callee]:
                    pointers = tuple(
                        space for space, param in zip(spaces, target["params"])
                        if param["pointer"])
                    if pointers and all(pointers):
                        signatures.setdefault(callee, set()).add(pointers)
    renames, extra = {}, []
    for callee, seen in signatures.items():
        if len(seen) < 2:
            continue
        ordered = sorted(seen)
        naming = {sig: f"{callee}__as{''.join(SPACE_TAG[s] for s in sig)}"
                  for sig in ordered}
        for body in bodies[callee]:
            if not (body.get("own") and body.get("signature")):
                continue
            # The definition takes the FIRST signature's name in place; the
            # others are appended as whole copies.
            spot = _name_position(body)
            if spot is not None:
                renames[(spot[0], spot[1], len(callee))] = naming[ordered[0]]
            extra.append({"name": callee, "body": body, "order": ordered,
                          "naming": naming})
        for name, group in bodies.items():
            for body in group:
                if not body.get("own"):
                    continue
                for called, at, length, spaces in _call_sites(body, bodies):
                    if called != callee or at is None:
                        continue
                    for target in bodies[callee]:
                        pointers = tuple(
                            space for space, param
                            in zip(spaces, target["params"]) if param["pointer"])
                        if pointers in naming:
                            renames[(at[0], at[1], length)] = naming[pointers]
    return renames, extra


def _propagate_from(body, bodies):
    """Push this body's known spaces into the bodies it calls. Returns moves."""
    known = {p["name"]: p for p in body["params"] if p["name"]}
    for match in LOCAL_ALIAS_RE.finditer(body["text"]):
        local, expr = match.group(1), match.group(2)
        source = known.get(_leading_ident(expr))
        if local not in known and source and source["pointer"] and source["space"]:
            known[local] = {"name": local, "pointer": True,
                            "space": source["space"], "written": None}
    for match in LOCAL_OBJECT_RE.finditer(body["text"]):
        kind, local = match.group(1), match.group(2)
        if local in known or kind in NOT_A_TYPE:
            continue
        known[local] = {"name": local, "pointer": True, "space": "thread",
                        "written": None}

    moved = 0
    # A TEMPLATE CALL NAMES ITS ARGUMENTS BETWEEN THE NAME AND THE LIST, and
    # `contact_stiffness<N>(local_hessian, ...)` is how every templated neutral
    # body is reached. Matching only `name(` skipped all of them.
    for match in re.finditer(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^<>();]*>\s*)?\(",
            body["text"]):
        callee = match.group(1)
        if callee not in bodies or callee == body["name"]:
            continue
        index, depth = match.end() - 1, 0
        while index < len(body["text"]):
            if body["text"][index] == "(":
                depth += 1
            elif body["text"][index] == ")":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        arguments = _split_top_level(body["text"][match.end():index])
        for target in bodies[callee]:
            if len(arguments) != len(target["params"]):
                continue
            for position, (argument, _offset) in enumerate(arguments):
                param = target["params"][position]
                if not param["pointer"] or param["written"]:
                    continue
                source = known.get(_leading_ident(argument))
                if not (source and source["pointer"] and source["space"]):
                    continue
                calls = body.setdefault("calls", [])
                proposals = param.setdefault("proposals", {})
                if source["space"] in proposals:
                    continue
                proposals[source["space"]] = body["name"]
                # ONE PROPOSAL IS AN INFERENCE. TWO IS A DEFECT IN THE KERNEL,
                # not a case to fall back from.
                #
                # MSL HAS NO GENERIC ADDRESS SPACE. A function reached with a
                # device pointer at one call site and a thread pointer at
                # another cannot be written once: there is no qualifier that
                # means either, so the author cannot express it any more than
                # this can infer it. Falling back to whatever the author wrote
                # would leave a body that compiles here and fails the shader
                # compile on a Mac, which is the whole failure class this
                # inference exists to remove.
                #
                # The caller is recorded with the space so the refusal can name
                # both sites rather than only the disagreement.
                param["space"] = (next(iter(proposals))
                                  if len(proposals) == 1 else None)
                param["ambiguous"] = len(proposals) > 1
                param["owner"] = target.get("name")
                moved += 1
    return moved


def check_msl_spaces_complete(source_path, rendered, lines, masks):
    """Every pointer and reference in an MSL body must carry an address space.

    THIS IS THE CHECK THAT SURVIVES THE ATTRIBUTES BEING INFERRED. While a body
    wrote its own spaces, an omission was visible in the SOURCE; now that the
    generator supplies the ones an entry decides, what matters is whether the
    RENDERING came out qualified, and that is a property of the emitted text
    rather than of the declaration.

    MSL requires the qualifier on every pointer and reference type. nvcc and a
    host C++ compiler do not, so an unqualified one reaches the Metal shader
    compiler and no earlier: `pointer type must have explicit address space
    qualifier`, at run time, on a Mac, after every build leg has gone green.

    A parameter this cannot reach is one whose space its CALLER decides, which
    the author still writes; that is why this reads the rendering rather than
    the inference's own table, and so covers both.
    """
    spaces = tuple(ADDRESS_SPACE)
    # COMMENTS ARE NOT CODE, and this scan reads a RENDERING rather than the
    # masked source, so it has to say so itself: a prose sentence containing an
    # ampersand otherwise reads as an unqualified reference parameter, which is
    # how the first version of this check reported `embed_contact`.
    scan = re.sub(r"/\*.*?\*/", lambda m: " " * (m.end() - m.start()),
                  rendered, flags=re.S)
    scan = re.sub(r"//[^\n]*", lambda m: " " * (m.end() - m.start()), scan)
    # A DEFINITION, NOT A CALL, and the difference is the brace that follows.
    # Requiring `inline` missed every struct METHOD, which is what the six
    # traversal visitors' `test(const AABB &box, ...)` is, and those are exactly
    # the parameters an entry does not decide. A call is `name(args);` and has
    # no body, so looking for the opening brace separates the two without
    # naming the shapes a definition can take.
    # A CONTROL-FLOW KEYWORD LOOKS EXACTLY LIKE A DEFINITION under the rule
    # below: `if (a.active && b.active) {` is a name, a parenthesized list and a
    # brace. They are excluded by name because there is no shape that separates
    # them.
    control = frozenset(("if", "for", "while", "switch", "catch", "do",
                         "else", "return"))
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", scan):
        if match.group(1) in control:
            continue
        i, depth = match.end() - 1, 0
        while i < len(scan):
            if scan[i] == "(":
                depth += 1
            elif scan[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        after = scan[i + 1:i + 40].lstrip()
        while after.startswith("const"):
            after = after[len("const"):].lstrip()
        if not after.startswith("{"):
            continue
        for piece, _offset in _split_top_level(scan[match.end():i]):
            bare = piece.strip()
            if not bare or ("*" not in bare and "&" not in bare):
                continue
            # A pointer to a pointer is refused elsewhere; a function pointer
            # and a template parameter pack do not occur in a neutral body.
            if any(re.search(rf"\b{space}\b", bare) for space in spaces):
                continue
            fail(source_path, 1, 1,
                 f"the MSL rendering of '{match.group(1)}' has a pointer or "
                 f"reference parameter with no address space: '{bare.strip()}'. "
                 f"MSL requires one on every pointer and reference type, and "
                 f"nothing before the shader compiler will say so. If an entry "
                 f"decides this parameter, the generator supplies the space and "
                 f"this means the entry and the body disagree about the "
                 f"argument order; if a CALLER decides it, the body must write "
                 f"the space itself")


SPECIALIZATION_BANNER = "\n// ---- address-space specializations "


def _render_specializations(plans, lines, masks, target, source_path):
    """One copy of a body per address-space signature it is called with."""
    out = [SPECIALIZATION_BANNER + "-" * 30 + "\n"]
    for plan in plans:
        body = plan["body"]
        first, last = body["signature"]
        close = body["close"]
        if close is None:
            continue
        spot = _name_position(body)
        for signature in plan["order"][1:]:
            pointers = [param for param in body["params"] if param["pointer"]]
            edits = {}
            renames = {}
            for param, space in zip(pointers, signature):
                if param["written"]:
                    if param["span"]:
                        lineno, start, stop = param["span"]
                        renames.setdefault(lineno, []).append(
                            (start, stop,
                             ADDRESS_SPACE[space][TARGETS.index(target)]))
                    continue
                if param["at"]:
                    edits.setdefault(param["at"][0], []).append(
                        (param["at"][1], space))
            if spot:
                renames.setdefault(spot[0], []).append(
                    (spot[1], spot[1] + len(plan["name"]),
                     plan["naming"][signature]))
            out.append(f'#line {first} "{os.path.basename(source_path)}"\n')
            for lineno in range(first, close + 1):
                out.append(substitute_attributes(
                    lines[lineno - 1], masks[lineno - 1], target,
                    edits.get(lineno, ()), renames.get(lineno, ())) + "\n")
    return "".join(out)


def convert(source_path, target):
    text, lines, masks, entries = read_source(source_path)
    source_dir = os.path.dirname(os.path.abspath(source_path))

    # THE ADDRESS SPACES THE BODIES DID NOT WRITE, decided by how each entry
    # hands its buffers over. Computed once per file against the ORIGINAL
    # lines, because that is the coordinate system the substitution applies
    # them in.
    inferred, renames, specializations = inferred_body_plan(
        lines, masks, entries, source_path)
    # SPECIALIZATION IS METAL'S ALONE. CUDA and the host have one address space,
    # so every copy would be identical there and the second would be a
    # redefinition; the address spaces render as nothing on those targets and
    # the original body is already correct.
    if target != "metal":
        renames, specializations = {}, []
    by_line = {}
    for (lineno, column, length), name in renames.items():
        by_line.setdefault(lineno, []).append((column, column + length, name))

    # An entry declaration is not code any compiler reads: it is the input to
    # `--emit entry`. It is replaced line for line with comment text, which is
    # what keeps the body rendering's line count equal to the neutral file's
    # and the Metal `#line` mapping exact.
    neutralized = {}
    # THE TWIN THIS TARGET DOES NOT WANT, neutralized before the entries so a
    # body and an entry cannot both claim a line. They never overlap in practice
    # and the dict would silently keep the last writer if they did.
    for kind, _name, first, last in twin_spans(source_path, lines, masks):
        if target in TWIN_TARGETS[kind]:
            continue
        neutralized[first] = (f"// [kernelgen] {kind} body, not rendered for "
                              f"{target} (rule 1-LANE)")
        for lineno in range(first + 1, last + 1):
            neutralized[lineno] = "//"
    for entry in entries:
        # A self-declaring entry's span IS its body; there is no declaration
        # text to replace, and replacing it would delete the kernel. Its
        # ATTRIBUTE line takes the marker instead, which is a line carrying no
        # code and which keeps the rendering's entry count readable.
        if entry.self_declared:
            if entry.attribute_line is not None:
                neutralized[entry.attribute_line] = (
                    f"// [kernelgen] entry declaration: {entry.name} "
                    f"(rendered by --emit entry)")
            continue
        neutralized[entry.first_line] = (
            f"// [kernelgen] entry declaration: {entry.name} "
            f"(rendered by --emit entry)")
        for lineno in range(entry.first_line + 1, entry.last_line + 1):
            neutralized[lineno] = "//"

    out_lines = []
    for lineno, (line, line_mask) in enumerate(zip(lines, masks), start=1):
        if lineno in neutralized:
            out_lines.append(neutralized[lineno])
            continue
        rewritten = rewrite_include(source_path, lineno, line, line_mask,
                                    target, source_dir)
        if rewritten is None:
            rewritten = substitute_attributes(
                line, line_mask, target, inferred.get(lineno, ()),
                by_line.get(lineno, ()))
        if "\n" in rewritten:
            raise KernelGenError(
                f"{source_path}:{lineno}: internal error: a substitution "
                f"introduced a newline, which would break line attribution")
        out_lines.append(rewritten)

    body = "\n".join(out_lines)

    if specializations:
        body += _render_specializations(specializations, lines, masks, target,
                                        source_path)

    result = preamble(target, source_path) + body

    if target == "metal":
        check_msl_spaces_complete(source_path, body, lines, masks)

    # The one invariant worth asserting rather than trusting: the body carries
    # exactly the neutral file's lines. Address-space specializations are
    # APPENDED after them, so the mapping from a rendered line to a source line
    # is unchanged for everything the `#line 1` preamble covers, and the copies
    # carry their own.
    measured = body.split(SPECIALIZATION_BANNER)[0]
    if measured.count("\n") != text.count("\n"):
        raise KernelGenError(
            f"{source_path}: internal error: the {target} body has "
            f"{measured.count(chr(10))} newlines against {text.count(chr(10))} of "
            f"source")
    return result


def convert_entry(source_path, target, emit="entry", migrated=()):
    """The entry artifacts for one target, or a banner if the file has none.

    A file with no entry declaration still produces a file, so a build rule can
    run over every neutral kernel without knowing which ones carry an entry.
    That holds for the `args` half too: a build rule renders one per neutral
    kernel and a caller includes the ones it fills.

    `migrated` names the buffer fields of THIS file's records that the caller
    already fills with a handle, as `<record>.<field>`. Every C++ rendering
    spells a buffer field the same way whatever is in it, so the list reaches
    only the Rust twin.
    """
    _, _, _, entries = read_source(source_path)
    if not entries:
        if emit in ("externs", "thunks"):
            return (_entry_banner(target, source_path, emit=emit) +
                    "//\n// This neutral kernel declares no entry point.\n")
        if emit == "table":
            return (_table_banner(target, source_path) +
                    "//\n// This neutral kernel declares no [[seam::args]] "
                    "entry, so it contributes no row.\n")
        return (_entry_banner(target, source_path, emit=emit) +
                "//\n"
                "// This neutral kernel declares no [[seam::args]] entry.\n")
    if emit == "externs":
        return render_externs_rust(source_path, entries)
    if emit == "thunks":
        return render_thunks_rust(source_path, entries)
    if emit == "table":
        return TABLE_RENDERERS[target](source_path, entries)
    if emit == "args":
        return ARGS_RENDERERS[target](source_path, entries)
    if target == "rust":
        _check_migrated(source_path, entries, migrated)
        return render_entry_rust(source_path, entries, migrated)
    return ENTRY_RENDERERS[target](source_path, entries)


def _check_migrated(source_path, entries, migrated):
    """Refuse a `--handle-field` this file's declarations do not have.

    THE MISS IS SILENT WITHOUT THIS, and it is silent in the direction that
    matters: a misspelled field simply does not migrate, the record keeps an
    address in it, and the only symptom is that the buffer count the caller
    thinks it moved never falls. Worse, the driver's own call site would then
    fail to compile against a field it expected to be a handle and the message
    would name the record rather than the typo.
    """
    known = {f"{entry.record_rust}.{f.name}"
             for entry in entries for f in entry.fields if f.kind == "handle"}
    for name in migrated:
        if name in known:
            continue
        offered = ", ".join(sorted(known)) or "none"
        raise KernelGenError(
            f"{source_path}: --handle-field {name} names no buffer field of "
            f"any record this file declares. Its buffer fields are: {offered}")


def _kernel_root_of(source):
    """The kernel tree a source file sits in, for a hand invocation.

    A build always passes --kernel-root explicitly. This is the fallback for
    someone rendering one body from a shell, and it looks for the tree-wide
    header every kernel tree has at its top rather than counting directory
    levels, which would depend on how deeply the body is filed.
    """
    directory = os.path.dirname(os.path.abspath(source))
    while True:
        if os.path.isfile(os.path.join(directory, "arena_handle.hpp")):
            return directory
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def main(argv):
    parser = argparse.ArgumentParser(
        description="Render a neutral *.kernel.cpp for one backend.")
    parser.add_argument("--target", required=True, choices=ENTRY_TARGETS)
    parser.add_argument("--emit", default="body", choices=EMIT_KINDS,
                        help="body: the kernel bodies, today's rendering. "
                             "entry: the entry points declared with "
                             "[[seam::args]]. args: those declarations' "
                             "argument records alone, includable by any "
                             "number of callers (--target cu only). table: "
                             "one fragment of the kernel table, the rows for "
                             "this file's entry points (--target cu or rust).")
    parser.add_argument("--out", required=True)
    parser.add_argument("--kernel-root", default=None,
                        help="Root of the neutral kernel tree. Only an entry "
                             "rendering needs it, to resolve the headers a "
                             "generated entry point includes. Defaults to the "
                             "source file's own tree.")
    parser.add_argument("--handle-field", action="append", default=[],
                        metavar="RECORD.FIELD",
                        help="A buffer field the caller already fills with an "
                             "(arena, offset) handle rather than an address, "
                             "named with the RUST record name. Repeatable. "
                             "Only the Rust entry rendering reads it; every "
                             "C++ rendering spells a buffer field the same way "
                             "whatever the caller puts in it. A name no "
                             "declaration in this file carries is an error.")
    parser.add_argument("source")
    args = parser.parse_args(argv[1:])

    global KERNEL_ROOT
    KERNEL_ROOT = (os.path.abspath(args.kernel_root)
                   if args.kernel_root else _kernel_root_of(args.source))

    if not args.source.endswith(KERNEL_SUFFIX):
        sys.stderr.write(
            f"kernelgen: {args.source} is not a neutral kernel source; the "
            f"name must end in {KERNEL_SUFFIX}\n")
        return 2
    if args.emit == "body" and args.target == "rust":
        sys.stderr.write(
            "kernelgen: there is no Rust rendering of a kernel BODY. Rust is "
            "the driver's language, and what it holds is the argument record: "
            "--target rust --emit entry\n")
        return 2
    if args.emit == "table" and args.target not in TABLE_TARGETS:
        sys.stderr.write(
            f"kernelgen: --emit table has no {args.target} rendering. A table "
            f"row exists to be looked up by a dense kernel id, and only two "
            f"consumers do that: the backend library, which holds the "
            f"launcher's address, and the driver, which holds the seam's "
            f"KernelDecl. The Metal recipe renders no entry file at all, and "
            f"the host target reaches its entry points by symbol rather than "
            f"by id\n")
        return 2
    if args.emit == "diagfile" and args.target not in DIAGFILE_TARGETS:
        sys.stderr.write(
            f"kernelgen: --emit diagfile has no {args.target} rendering. Only "
            f"the two backends that hash __FILE__ into a device record need a "
            f"table to reverse it: the host backend stores the string itself, "
            f"and the Metal one registers a path per segment as it assembles "
            f"the shader and resolves it lazily\n")
        return 2
    if args.handle_field and (args.target != "rust" or args.emit != "entry"):
        sys.stderr.write(
            "kernelgen: --handle-field applies only to --target rust --emit "
            "entry. Every C++ rendering spells a buffer field as an "
            "ArenaHandle whatever the caller puts in it, so there is nothing "
            "for the flag to change there and accepting it would suggest "
            "otherwise\n")
        return 2
    if args.emit == "args" and args.target not in ARGS_TARGETS:
        sys.stderr.write(
            f"kernelgen: --emit args has no {args.target} rendering. An entry "
            f"rendering has to be split only where the record is needed in "
            f"more than one translation unit. CUDA and Metal both are: the "
            f"CUDA record is included by every caller that fills one, and the "
            f"Metal record is READ by the spliced shader and WRITTEN by the "
            f"ObjC++ host. The cpp rendering is compiled into one entry "
            f"translation unit and the Rust rendering is one module, so "
            f"neither splits. Use --emit entry\n")
        return 2
    try:
        if args.emit == "diagfile":
            rendered = render_diagfile_cu(args.source, args.target)
        elif args.emit in ("entry", "args", "table", "externs", "thunks"):
            rendered = convert_entry(args.source, args.target, emit=args.emit,
                                     migrated=tuple(args.handle_field))
        else:
            rendered = convert(args.source, args.target)
    except KernelGenError as exc:
        sys.stderr.write(f"kernelgen: {exc}\n")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # Written whole and compared first, so a rebuild that changes nothing does
    # not move the mtime and does not force every dependent object to recompile.
    if os.path.isfile(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            if f.read() == rendered:
                return 0
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
