// File: shader_compiler.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Assembles the MSL translation unit the Metal backend compiles at run time,
// and hands it to the context for compilation.
//
// The backend compiles shaders through newLibraryWithSource: rather than
// shipping an offline .metallib. Metal
// has no filesystem for the shader compiler, so an #include in shared device
// code is a compile error: the HOST concatenates the sources itself. That makes
// the host the include mechanism, and it is therefore the one component that
// knows the provenance of every line of the assembled text.
//
// This module spends that knowledge in two places:
//
//   1. It injects `#line 1 "<path>"` per segment, so __LINE__ inside the shader
//      and the file named in a compile diagnostic both refer to the real source
//      file rather than to an offset into one anonymous concatenated blob.
//   2. It injects a DIAG_FILE_ID define per segment, registered with the
//      diagnostic channel, so a device-side assert record carries a file id the
//      host can resolve back to a path and quote. A per-translation-unit -D
//      could not do this: most penetration-critical asserts live in headers
//      (accd.hpp, distance.hpp, pdrd_rigid.hpp) that are pulled into many
//      translation units, and a -D would attribute every one of them to
//      whichever unit happened to include the header.
//
// Both properties are measured, not assumed: the line-attribution probe
// passes 6 of 6 on this device, including a compile error reported as
// "contact/accd.hpp:215:10: error: use of undeclared identifier ...".
//
// Compilation itself is NOT performed here. It goes through
// context_compile_library (metal_context.hpp), which owns the single
// mathMode = MTLMathModeSafe options helper. A second options path in this file
// would be a second chance to ship a fast-math library, and fast math is a
// wrong-answer setting for this solver rather than a speed trade.

#pragma once

#include <string>
#include <vector>

namespace metal_backend {

struct Context;
struct Diagnostics;

struct ShaderSource;

// Creates an assembler and emits the prologue: the MSL standard include, the
// backend macro seam, and the diagnostic channel's shader prologue, in that
// order. 'ctx' compiles the result and 'diag' assigns the file ids; neither may
// be null. Returns null, after reporting on stderr, if either is.
ShaderSource *shader_create(Context *ctx, Diagnostics *diag);
void shader_destroy(ShaderSource *s);

// Appends one source segment, injecting
//     #undef  DIAG_FILE_ID
//     #define DIAG_FILE_ID <id>
//     #line 1 "<path>"
// ahead of it, so __LINE__ and the diagnostic channel's file id both track the
// real file even across header boundaries.
//
// The directive order is load-bearing: `#line N` numbers the NEXT physical
// line, so it has to be the last directive before the body. With the #line
// first, the two define lines would consume lines 1 and 2 of the named file and
// every reported line number would be off by exactly two.
//
// A failure here (an unusable path, a body that is not text) is latched and
// reported by the next shader_compile, because this entry point has no error
// channel of its own and silently dropping a segment would surface much later
// as a kernel missing from the library with nothing to say why.
void shader_add_segment(ShaderSource *s, const char *path, const char *text);

// Reads a file from disk and appends it as a segment. The path is registered
// verbatim, so pass the path the diagnostic channel should later open when it
// resolves a record and quotes the offending line.
bool shader_add_file(ShaderSource *s, const char *path, std::string *err);

// The assembled source, prologue first.
const std::string &shader_assembled(ShaderSource *s);

// Compiles the assembled source. On failure 'err' carries the compiler
// diagnostic with the INJECTED filename already in it, plus the offending
// source line quoted from the assembled text.
//
// Returns a library id (> 0) for context_make_pipeline, or 0 with 'err' set.
// The id space is the context's, not this module's.
//
// Setting PPF_METAL_SHADER_DUMP=<path> in the environment dumps the assembled
// translation unit on every compile, which is the only way to inspect what the
// compiler actually saw.
unsigned shader_compile(ShaderSource *s, std::string *err);

// Writes the assembled source to 'path'. A concatenated translation unit is
// otherwise impossible to inspect: no file on disk corresponds to it.
bool shader_dump(ShaderSource *s, const char *path, std::string *err);

// The segment paths, in the order they were concatenated, as they appear in the
// injected directives. This is the provenance list to check first when a
// diagnostic names a file that is not the file you expected to be compiling.
std::vector<std::string> shader_segment_paths(ShaderSource *s);

}  // namespace metal_backend
