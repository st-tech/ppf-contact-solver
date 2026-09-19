// File: metal_context.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Device, command queue, compile options and command-buffer lifecycle for
// the Metal backend. Every other module in the backend reaches Metal through
// this one, which is what keeps the single mathMode=Safe compile path and the
// single status-plus-error check from being duplicated and drifting.
//
// A Context is NOT thread safe. It is driven from the one thread that runs
// the solver's step loop, matching the CUDA backend, and its id tables are
// plain vectors with no lock. Every call that has no error out-parameter
// aborts the process on a violated precondition rather than returning,
// because the alternative on Metal is not an exception but a plausible wrong
// answer with status Completed.

#pragma once

#include <cstddef>
#include <string>

namespace metal_backend {

// Opaque handles so the header stays includable from plain C++ (.cpp) as
// well as Objective-C++ (.mm). Only the .mm files see MTL types.
struct Context;

// Creates the system default device and one command queue. Returns null and
// fills 'err' if no Metal device exists. There is no fallback: a host with
// no Metal device must fail loudly.
//
// 'log' is where this module's NON-FATAL messages go, and it is required
// rather than optional because the obvious default is a defect. Anything this
// process writes to stderr lands in the session's error.log, and the frontend
// reports a run with a non-empty error.log as `*** Solver FAILED ***`
// (frontend/_session_.py). So a message like "the pipeline cache was stale, I
// recompiled" turns a slow start into a reported crash if it goes to stderr.
// Passing the backend's own logging channel here is the only correct wiring,
// and making it a parameter is what stops a later caller from omitting it.
// Fatal paths still abort through stderr, which is exactly what that channel
// is for.
Context *context_create(void (*log)(const char *message), std::string *err);
void context_destroy(Context *ctx);

const char *context_device_name(Context *ctx);
unsigned long long context_max_buffer_length(Context *ctx);
unsigned context_max_threads_per_threadgroup(Context *ctx);
unsigned context_max_threadgroup_memory(Context *ctx);

// Compiles MSL source into a library. ALWAYS uses mathMode Safe. Returns a
// library id (>0) or 0 with 'err' filled.
//
// Also opens this library's on-disk pipeline archive (pipeline_cache.hpp),
// keyed on the source, the device, the OS build and the math mode. A missing,
// stale, corrupt or unusable archive is not an error: the library still
// compiles and every pipeline below is created the way it was before the cache
// existed.
unsigned context_compile_library(Context *ctx, const char *source,
                                 std::string *err);

// Loads a PRE-BUILT `.metallib` from a path and registers it, returning a
// library id (>0) or 0 with 'err' filled.
//
// THE DIFFERENCE FROM `context_compile_library` IS THAT NO SOURCE EXISTS HERE,
// and that is what it is for. A backend that dispatches by kernel id binds
// entry points by NAME out of a library the build produced, so it needs neither
// a source string nor the segment order assembling one demands: `xcrun metal`
// has a filesystem, each generated entry compiles offline as its own
// translation unit with live includes, and `xcrun metallib` links the AIR.
//
// It registers NO PIPELINE ARCHIVE. That cache is keyed on source bytes, and
// there are none; the archive it would want is keyed to a device and a driver
// and cannot be shipped anyway, so a caller that wants one builds it the way
// `context_compile_library` does. Pipeline creation is still the AIR-to-ISA
// step and is still this backend's dominant startup cost.
unsigned context_load_library(Context *ctx, const char *path,
                              std::string *err);

// Builds a compute pipeline for 'fn_name' in the given library. Returns a
// pipeline id (>0) or 0 with 'err' filled.
//
// Served from the library's pipeline archive when it is there. Creating a
// compute pipeline state is the AIR to ISA step and is the backend's dominant
// startup cost, so this is where the cache pays; a miss falls back PER
// PIPELINE, so a partially invalidated archive costs only the pipelines it
// lost.
unsigned context_make_pipeline(Context *ctx, unsigned library,
                               const char *fn_name, std::string *err);

// Writes every dirty pipeline archive to disk. Call it once after the last
// pipeline of the run has been created; the backend creates all of them during
// initialize(), so that is the point.
//
// Nothing is written when every pipeline came out of an archive, which is the
// steady state. A failure is reported through 'err' and returns false, and the
// only consequence is that the next run misses, so a caller may log it and
// continue. context_destroy flushes as well, in case a run ends before
// reaching the explicit call.
bool context_archive_flush(Context *ctx, std::string *err);

// What the pipeline cache did this run. Reported as data rather than logged
// here, because this module deliberately depends on nothing but Metal.
//
// 'pipeline_hits + pipeline_misses' is the number of pipelines created; a run
// with no cache leaves both at zero and 'pipelines_uncached' non-zero.
struct ShaderCacheReport {
    unsigned libraries;           // libraries brought up, either way
    unsigned libraries_prebuilt;  // of those, loaded from an offline .metallib
    unsigned archives_loaded;     // archives found on disk and opened
    unsigned archives_started;    // archives that began empty (key miss)
    unsigned archives_written;    // archives serialized by the flush
    unsigned pipeline_hits;       // pipelines taken from an archive
    unsigned pipeline_misses;     // pipelines compiled, then archived
    unsigned pipelines_uncached;  // pipelines compiled with no archive at all
    double compile_ms;            // newLibraryWithSource, summed
    double prebuilt_load_ms;      // newLibraryWithURL on a .metallib, summed
    double archive_open_ms;       // finding and opening archives, summed
    double pipeline_ms;           // pipeline creation, summed
    double serialize_ms;          // writing archives, summed
};
void context_shader_cache_report(Context *ctx, ShaderCacheReport *out);

// The directory the archives live in, or an empty string when the cache is
// disabled (PPF_METAL_ARCHIVE=0) or has no home directory to use.
const char *context_shader_cache_dir(Context *ctx);

// The directory offline .metallib artifacts are read from, or "(disabled)".
// $PPF_METAL_LIBRARY_DIR when set, otherwise the directory this backend's dylib
// was loaded from, which is where the build installs the artifact.
const char *context_prebuilt_library_dir(Context *ctx);

// Device memory. The arena allocator (arena.hpp) is the intended caller: it
// takes one buffer per arena here and hands out (buffer id, offset) handles,
// and the id it gets back is exactly what context_bind_buffer accepts. The
// allocation lives in this module because the device does, and because
// binding has to resolve an id to an MTLBuffer that no header outside a .mm
// is allowed to name.
//
// Storage mode is Shared, so context_buffer_contents is the host-visible
// mapping of the same memory and no blit is needed; the contents are valid
// to read on the host once context_commit_and_wait has returned.
// Freshly allocated bytes are NOT zeroed: initialize what you read.
//
// Returns a buffer id (>0) or 0 with 'err' filled. Note that a length beyond
// what an arena handle's 32-bit offset can address is the ALLOCATOR's bound
// to enforce, not this one; this call only rejects what the device rejects.
unsigned context_new_buffer(Context *ctx, unsigned long long length,
                            std::string *err);
void *context_buffer_contents(Context *ctx, unsigned buffer_id);
unsigned long long context_buffer_length(Context *ctx, unsigned buffer_id);

// Fills [offset, offset + length) of a buffer with a repeated byte, through a
// blit encoder, then commits and waits and checks both halves exactly as
// context_commit_and_wait does. A zero length fills nothing and succeeds.
//
// A blit rather than a memset through context_buffer_contents, for two reasons
// that outlive the current execution model. The fill is ORDERED against the
// dispatches that read and write the buffer by the queue, so it stays correct
// if this backend ever stops waiting after every command buffer, whereas a host
// memset is ordered only by that wait. And it costs no host bandwidth and no
// scratch allocation on a buffer that can be hundreds of megabytes; the arrays
// cleared per Newton step are the largest the backend owns.
//
// It takes its own command buffer because a slot from
// context_begin_command_buffer already carries an open COMPUTE encoder, and
// Metal permits one encoder at a time on a command buffer.
//
// 'offset' must be a multiple of 4, which every arena block satisfies by
// construction, and the range must lie inside the buffer. Both are checked
// here: Metal drops a fill that runs past the end and still reports Completed.
bool context_fill_buffer(Context *ctx, unsigned buffer_id,
                         unsigned long long offset, unsigned long long length,
                         unsigned char value, std::string *err);

// One command buffer's worth of work. Encoders are created by the caller
// through the dispatch helpers below.
unsigned context_begin_command_buffer(Context *ctx);

// Binds an arena buffer (see arena.hpp) at 'index'. Fails loudly if index
// is outside 0..30, because Metal accepts index 31 in release with no
// exception and the shader compiler rejects it, so the two halves disagree.
void context_bind_buffer(Context *ctx, unsigned cmd, unsigned index,
                         unsigned buffer_id, unsigned long long offset);

// setBytes with the size checked by US. Exceeding the device cap is a
// silent SIGABRT with no diagnostic of any kind, so this must refuse first.
bool context_set_bytes(Context *ctx, unsigned cmd, unsigned index,
                       const void *data, size_t length, std::string *err);

void context_dispatch(Context *ctx, unsigned cmd, unsigned pipeline,
                      unsigned grid_x, unsigned threadgroup_x);

// Dispatches an exact number of threadgroups, each with a run-time-sized block
// of threadgroup memory bound at index 0. Two things separate it from
// context_dispatch above, and a caller needs both or neither.
//
// The grid is threadgroupS, not threads, because a kernel that cooperates over
// one problem per group indexes with [[threadgroup_position_in_grid]] and a
// ceil-div over a thread count cannot express "one group per aggregate" when the
// group size is chosen for occupancy rather than for the problem.
//
// The threadgroup allocation is dynamic because the block is sized by a
// parameter the shader cannot see (the Schwarz dense-block cap), the same reason
// the CUDA side passes it as the launch's third argument. 'bytes' is rounded UP
// to the 16-byte granularity Metal requires and is REFUSED when it exceeds this
// device's per-threadgroup limit, rather than being clamped: a clamp would run
// the kernel against memory it would then index past, which this platform does
// not fault on.
bool context_dispatch_threadgroups(Context *ctx, unsigned cmd,
                                   unsigned pipeline, unsigned groups,
                                   unsigned threads_per_group,
                                   unsigned long long bytes, std::string *err);

// Commits, waits, then checks BOTH status and error. Returns false and
// fills 'err' on either. A killed command buffer reports Completed with
// plausible stale data, so status alone is not enough. The slot is retired on
// both outcomes, so a caller owes it back only on the paths that never reach
// here.
//
// THE WAIT IS PART OF THE CONTRACT A DIRECTLY ADDRESSED ALLOCATION RESTS ON.
// It is what leaves no device work in flight at any point a caller of the
// backend runs, which is the first of the two conditions backend_abi.h binds a
// backend to before it may return a host address. Removing the
// waitUntilCompleted, here or in context_fill_buffer, therefore requires
// setting kSynchronousExecution in backend/backend.mm to false, which makes
// be_host_ptr refuse and puts every caller back on the copy path.
bool context_commit_and_wait(Context *ctx, unsigned cmd, std::string *err);

// Ends the encoder and retires the slot WITHOUT committing, so nothing that
// was encoded runs. It is what a caller leaving a command buffer by an error
// path owes: a slot that is begun and never returned stays open for the life
// of the context, and context_live_commands counts it from then on.
void context_abandon_command(Context *ctx, unsigned cmd);

// How many command buffers have been created and not yet retired.
//
// It counts the slots context_begin_command_buffer hands out against those
// context_commit_and_wait and context_abandon_command return.
// context_fill_buffer takes a command buffer of its own outside that table and
// waits on it before returning, so it is never in flight at a point this could
// be asked.
//
// EVERY EXECUTOR IN THIS BACKEND COMMITS AND WAITS INSIDE THE CALL THAT
// CREATED THE COMMAND BUFFER, so this reads zero at every point a caller runs,
// and be_host_ptr uses it as a cheap sanity check when a host address is
// asked for. It is NOT the check that keeps such an address safe. An
// asynchronous executor would report the same zero here, because an address is
// asked for between seam calls, which is exactly when nothing is in flight;
// that condition is keyed to kSynchronousExecution in backend/backend.mm
// instead.
unsigned context_live_commands(Context *ctx);

// The two always-on numeric sentinels from plan Section 5. They cost
// nothing and they are the only thing standing between a lost mathMode flag
// and a silently wrong solver. Run both at startup; either failing is fatal.
//   kahan:         a shader Kahan sum of [1e8, 1.0 x 4095] must return
//                  100004096. Returning 100000000 means compensation was
//                  optimized away, which is what fast math does.
//   reassociation: (a+b)+c must DIFFER from a+(b+c) for a=1e8, b=-1e8, c=1.
// The sentinel's own library, pipeline and two small buffers stay resident
// for the life of the context (about 16 KB), since there is no teardown for
// an individual object and the check runs once.
bool context_check_math_mode(Context *ctx, std::string *err);

}  // namespace metal_backend
