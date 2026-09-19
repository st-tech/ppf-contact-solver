// File: bringup.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The fixed sequence that brings the backend's shader libraries up, defined
// once and called by both processes that need it: `initialize()` in main.mm,
// and the offline shader-dump tool the build runs to produce `.metallib`
// artifacts.
//
// WHY IT IS SHARED RATHER THAN REPEATED. A pre-compiled library is found by a
// hash over the assembled shader text, and two things in that text are decided
// by this sequence rather than by the sources:
//
//   DIAG_BINDING  the diagnostic channel's binding index, which
//                     diag_create settles.
//   DIAG_FILE_ID       one per spliced segment, handed out by
//                     diag_register_file in CALL ORDER.
//
// So a second caller that brought the same modules up in a different order, or
// skipped one, would assemble text that differs from the solver's and hash to a
// different name. The artifact would then be built under a name the solver
// never asks for: no wrong answer, and no error either, just a build that
// silently produces nothing usable and a run that quietly compiles its shader
// as before. One definition removes that failure mode instead of leaving it to
// be remembered.
//
// WHAT IT KNOWS ABOUT THE MODULES IT BRINGS UP: their names, and that each
// either succeeds or explains itself. Nothing more. Which modules there are,
// what they compile and what they compare is decided by the backend library
// that is linked, through `backend_module_table` below, so this sequence holds
// the ORDER (which the two identifiers above are handed out in) without holding
// the list.
//
// That separation is the rule this crate is held to rather than a preference: a
// bring-up naming a solver's parity gates could not be published on its own and
// used by a program that is not a physics solver.

#pragma once

#include <string>

namespace metal_backend {

struct Context;
struct Allocator;
struct Diagnostics;
struct ShaderSource;

// What backend_bringup produces. Every member is non-null on success and
// owned by the caller, which for both callers today means owned until the
// process exits.
struct Bringup {
    Context *ctx = nullptr;
    Allocator *alloc = nullptr;
    Diagnostics *diag = nullptr;
    ShaderSource *shader = nullptr;
};

// Which step failed, so a caller can answer differently where the answers
// differ. Only MathMode is special today: a shader compiler that is not in
// MTLMathModeSafe is a correctness failure rather than an unusable host, so
// initialize() ends the process there instead of returning false.
//
// Every module failure reports as the one `Module` step, and the module's name
// is put at the front of '*err' rather than into a value here. A step is
// something this sequence does; a module is something it was handed.
enum class BringupStep {
    None,
    Device,
    MathMode,
    Allocator,
    Diagnostics,
    ShaderPrologue,
    Module,
};

// One module the backend library asks to have brought up after the platform is
// ready. 'name' appears in the log and in a failure message; 'run' either
// succeeds or fills '*err' with the reason.
//
// The signature is what the platform has to offer and no more: a device, an
// allocator and the diagnostic channel. A module needing anything else is
// asking this sequence to know what it computes.
struct BringupModule {
    const char *name;
    bool (*run)(Context *ctx, Allocator *alloc, Diagnostics *diag,
                std::string *err);
};

// THE SEAM. Supplied by whoever links this backend, exactly as `print_rust` is,
// and for the same reason: the alternative is for this crate to name the
// modules, which would put the solver's parity gates in a general compute API.
//
// Returns how many modules '*out' points at, and may return 0 with '*out'
// untouched: a caller that brings the platform up to allocate and dispatch,
// with no shader modules of its own, is a complete use of this crate.
//
// The ORDER is the caller's and is load-bearing, because DIAG_FILE_ID is handed
// out in the order the modules register their segments and the assembled text
// is what a pre-compiled library is keyed on. Two callers listing the same
// modules in different orders assemble text that hashes to different names.
unsigned backend_module_table(const BringupModule **out);

// Runs the sequence. Returns true with '*out' filled, or false with '*err'
// carrying the reason and '*failed' naming the step.
//
// 'log' is the NON-FATAL message channel and is required. In the solver process
// stderr is the crash channel, so a note about a recompiled shader must not go
// there; see context_create.
//
// Progress is reported through logging::info, so the solver's log and the dump
// tool's stdout carry the same account of what was brought up and what the
// caches did.
bool backend_bringup(void (*log)(const char *message), Bringup *out,
                     BringupStep *failed, std::string *err);

}  // namespace metal_backend
