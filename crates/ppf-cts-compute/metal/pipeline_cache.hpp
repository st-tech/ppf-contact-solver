// File: pipeline_cache.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The on-disk compute-pipeline cache: one MTLBinaryArchive per compiled
// library, keyed on a hash of the exact bytes that decide what the pipelines
// contain.
//
// WHY THIS EXISTS. Creating an MTLComputePipelineState is the AIR to ISA step,
// and on a cold machine it is the dominant startup cost of this backend, larger
// than the MSL compile. Measured on an Apple M1 over this backend's four
// libraries, with the OS caches removed first: 6.5 s to compile the source and
// 13.7 s to create 129 pipeline states, about 106 ms each. An offline .metallib
// would remove the first number and NOT the second, because AIR to ISA is paid
// whether the library came from source or from a .metallib. So the two halves
// need two mechanisms:
//
//   library compile   newLibraryWithSource. Cached only by the OS, in
//                     DARWIN_USER_CACHE_DIR/com.apple.metal, keyed on the
//                     source text.
//   pipeline creation THIS module, plus an OS cache of the same step in the
//                     same directory keyed on the function.
//
// macOS DOES cache both across process exits, so on a machine that has run this
// shader before, both numbers fall to single-digit milliseconds with no archive
// at all. What this module adds is a cache we OWN: the OS caches live in a
// directory the system evicts, and when they go, the measured startup returns to
// 24.7 s. With an archive on disk the same eviction costs 10.5 s, all of it the
// library compile, because the 129 pipelines come back in 3.1 ms.
//
// THREE PROPERTIES ARE LOAD-BEARING AND NONE OF THEM IS OPTIONAL.
//
//   1. A miss must fall back to compiling, per pipeline. A binary archive is a
//      cache keyed to a device and a driver, and Apple may invalidate it across
//      an OS update. A backend that refuses to start because its cache went
//      stale is worse than one that is slow, so every entry point here returns
//      "no cache" rather than an error, and pipeline_cache_make always produces
//      a pipeline when the device can produce one at all. A corrupt or
//      truncated file is exactly this case: newBinaryArchiveWithDescriptor
//      reports "invalid format", and the run continues against an empty
//      archive.
//
//   2. The key covers the SOURCE, so an edited kernel cannot load a stale
//      archive. That is the difference between a slow start and a wrong answer,
//      and this cache is one we own: the OS shader cache keys itself, ours is
//      ours to get right. A stale key MISSES (there is no file under the new
//      name) rather than loading anything.
//
//   3. The key covers the COMPILE OPTIONS, through the math mode read back out
//      of the options object rather than a literal. mathMode = Safe is a
//      correctness setting for this solver, not a speed trade: fast math
//      deletes Kahan compensation outright. A pipeline archive stores compiled
//      ISA, so an archive written by a fast-math build must never be loadable
//      by a safe-math one. Keying on the READ-BACK value makes that structural
//      instead of remembered.
//
// WHERE THE CACHE LIVES. $PPF_METAL_ARCHIVE_DIR if set, otherwise
// ~/.cache/ppf-cts/metal-pipeline-archive, matching the cache root the frontend
// uses for a developer checkout (ppf-cts-core/src/datamodel/app.rs
// default_cache_dir). A self-contained tree, which keeps all of its state inside
// itself, gets the variable from the session launcher
// (ppf-cts-core/src/datamodel/session/scripts.rs) pointing into that tree's own
// cache directory, so a distribution never reaches this fallback. It is
// deliberately NOT under crates/ppf-cts-solver/src: two build scripts watch
// that tree recursively and cargo reads a directory in rerun-if-changed as
// "rerun if any descendant changes", so an artifact anywhere inside costs a
// measured 48.3 s rebuild of two crates on every subsequent build. Being
// gitignored does not help, because cargo's mtime walk does not consult
// .gitignore.
//
// The directory is bounded: each write prunes the oldest archives until the
// total is back under a byte budget (see kMaxArchiveBytes). One archive is
// written per distinct shader source, so an unbounded directory would grow by
// one solver-library archive, 20 MB on an M1, per kernel edit.
//
// PPF_METAL_ARCHIVE=0 disables the cache completely: no load, no write, and
// pipelines are created exactly as they were before this module existed. That
// is the A/B lever for measuring what the cache buys.
//
// This header names Metal types, so it is includable only from Objective-C++.

#pragma once

#import <Metal/Metal.h>

#include <string>

namespace metal_backend {

// One library's archive. Opaque: the caller holds it for the life of the
// library and never inspects it.
struct PipelineCache;

// What one pipeline creation did. Reported per call so the caller can account
// for the cache without this module owning a global counter.
enum class PipelineCacheOutcome {
    // The pipeline came out of the archive with no compilation.
    Hit,
    // The archive did not have it. It was compiled, and added to the archive
    // so the next run hits.
    Miss,
    // No archive at all: caching disabled, unsupported, or the device refused
    // to open one. The pipeline was compiled directly.
    Unavailable,
};

// Opens the archive for a library whose source is 'source'.
//
// Returns null when caching is disabled or unavailable, which every caller must
// tolerate: pipeline_cache_make accepts a null cache and compiles directly.
//
// 'math_mode' is the value read back out of the compile options that produced
// the library, and it enters the key (property 3 above).
//
// 'library_origin' names the route that produced the MTLLibrary: "source" for
// newLibraryWithSource, or the offline library's own key for one loaded from a
// .metallib. It enters the key too, because the two routes do not produce
// interchangeable functions: measured on an M1, an archive written from one
// returns a miss for every function of the other. A shared file would still be
// correct, and would cost all 129 pipelines on every switch between them.
//
// 'open_ms' receives the wall time spent looking for and opening the file, and
// 'loaded' receives true when an existing archive was opened and false when the
// archive starts empty. Both may be null.
//
// 'log' receives any non-fatal message this call has to make (a stale file, a
// device that will not open an archive). It is a parameter and not a write to
// stderr because in this process stderr is the CRASH channel: it lands in the
// session's error.log and the frontend reports a run with a non-empty
// error.log as a failure. A cache that reported a stale file that way would
// turn a slow start into a reported crash.
PipelineCache *pipeline_cache_open(id<MTLDevice> device, const char *source,
                                   long long math_mode,
                                   const char *library_origin,
                                   void (*log)(const char *message),
                                   double *open_ms, bool *loaded);

void pipeline_cache_close(PipelineCache *pc);

// Creates the pipeline for 'fn', preferring the archive.
//
// The order is: try the archive with FailOnBinaryArchiveMiss; on a miss add the
// function to the archive and try the archive again; if that still misses,
// create with no archive at all. The middle step is what keeps a COLD run from
// paying twice, and it is measured rather than assumed: on an M1 the retry
// after the add hits, so a cold run costs one compilation per pipeline, the
// same as before the cache existed.
//
// Returns nil only when the device cannot create the pipeline at all, with
// 'err' filled. A cache that cannot help is never a reason to fail.
id<MTLComputePipelineState> pipeline_cache_make(PipelineCache *pc,
                                                id<MTLDevice> device,
                                                id<MTLFunction> fn,
                                                const char *fn_name,
                                                void (*log)(const char *message),
                                                PipelineCacheOutcome *outcome,
                                                std::string *err);

// Writes the archive to disk if anything was added since it was opened.
//
// Writes to a sibling temporary name and renames, so a reader never sees a
// partial file and a crash mid-write cannot leave a corrupt archive under the
// real name. Returns true when a file was written; a clean cache writes nothing
// and returns false with no error. 'ms' receives the wall time either way and
// may be null.
//
// A failure here is reported through 'err' and is not fatal: the next run
// simply misses.
bool pipeline_cache_flush(PipelineCache *pc, double *ms, std::string *err);

// The resolved cache directory, or an empty string when caching is disabled.
// Stable for the process; safe to call before any cache is opened.
const std::string &pipeline_cache_directory();

}  // namespace metal_backend
