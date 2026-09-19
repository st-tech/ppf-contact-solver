// File: bringup.mm
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Implementation of the shared bring-up sequence declared in bringup.hpp.

#include "bringup.hpp"

#include "common.hpp"

#include "arena.hpp"
#include "diagnostics.hpp"
#include "metal_context.hpp"
#include "shader_compiler.hpp"

namespace metal_backend {

namespace {

// How many trace records one dispatch can leave before the channel starts
// DROPPING (it never wraps: a wrapping ring tears, measured). 256 is the size
// the ring was characterized at on this hardware
// (1,048,576 writers into 256 slots gave exactly 256 written, 1,048,320
// dropped, 0 torn).
//
// It is a run-time field of the diagnostic buffer's header and reaches no
// shader text, so it does not enter the pre-compiled library's key.
constexpr unsigned kDiagRingSlots = 256;

bool step_failed(BringupStep step, const std::string &message,
                 BringupStep *failed, std::string *err) {
    if (failed) {
        *failed = step;
    }
    if (err) {
        *err = message;
    }
    return false;
}

}  // namespace

bool backend_bringup(void (*log)(const char *message), Bringup *out,
                     BringupStep *failed, std::string *err) {
    if (failed) {
        *failed = BringupStep::None;
    }
    if (!out) {
        return step_failed(BringupStep::Device,
                           "backend_bringup: null output", failed, err);
    }
    *out = Bringup();

    std::string reason;
    out->ctx = context_create(log, &reason);
    if (!out->ctx) {
        return step_failed(BringupStep::Device,
                           "no usable Metal device: " + reason, failed, err);
    }
    logging::info("metal: device '%s'", context_device_name(out->ctx));
    logging::info("metal: maxBufferLength %llu B, maxTotalThreadsPerThreadgroup "
                  "%u, threadgroup memory %u B",
                  context_max_buffer_length(out->ctx),
                  context_max_threads_per_threadgroup(out->ctx),
                  context_max_threadgroup_memory(out->ctx));

    // The two numeric sentinels, run before anything else touches the device.
    // Metal's compile DEFAULT is fast math, which deletes Kahan compensation
    // outright and damages the eigenvalue-floored inverse on 94.5% of threads
    // on the threads measured; MTLMathModeRelaxed is not
    // a middle ground and fails the same sentinels. A backend that reaches this
    // point with fast math enabled would produce plausible, wrong physics with
    // nothing to show for it, so the caller ends the process rather than
    // degrading.
    if (!context_check_math_mode(out->ctx, &reason)) {
        return step_failed(BringupStep::MathMode, reason, failed, err);
    }
    logging::info("metal: math-mode sentinels pass (mathMode Safe)");

    out->alloc = allocator_create(out->ctx);
    if (!out->alloc) {
        return step_failed(BringupStep::Allocator,
                           "could not create the arena allocator", failed, err);
    }

    out->diag = diag_create(out->ctx, out->alloc, kDiagRingSlots, &reason);
    if (!out->diag) {
        return step_failed(BringupStep::Diagnostics,
                           "could not create the diagnostic channel: " + reason,
                           failed, err);
    }
    logging::info("metal: diagnostic channel at binding index %u, %u ring slots",
                  diag_binding_index(out->diag), kDiagRingSlots);

    // Compile the shader prologue alone. It carries the backend macro seam and
    // the diagnostic channel's MSL, which every kernel is concatenated after,
    // so a defect in either is a defect in all of them. It costs one small
    // compile at startup and the alternative is discovering it from the first
    // kernel's error message.
    out->shader = shader_create(out->ctx, out->diag);
    if (!out->shader) {
        return step_failed(BringupStep::ShaderPrologue,
                           "could not create the shader assembler", failed, err);
    }
    if (shader_compile(out->shader, &reason) == 0) {
        return step_failed(BringupStep::ShaderPrologue,
                           "the shader prologue does not compile: " + reason,
                           failed, err);
    }
    logging::info("metal: shader prologue compiles at mathMode Safe");

    // The backend library's own modules, in the order it lists them. This is
    // where its shader libraries are compiled, so the order decides the
    // DIAG_FILE_ID each segment is given and therefore the name a pre-compiled
    // library is looked up under; it is the library's to choose and this
    // sequence's to preserve.
    //
    // A module reports its own progress through logging::info, so the log keeps
    // an ordered account without this loop restating any of it. On failure the
    // module's NAME goes at the front of the reason, which is the one place a
    // name is used and why each is written as a noun phrase.
    const BringupModule *modules = nullptr;
    const unsigned module_count = backend_module_table(&modules);
    // A count with no table is the one shape of that seam this loop cannot
    // survive, so it is refused rather than dereferenced. Zero with no table is
    // legitimate and falls through: a caller may bring the platform up to
    // allocate and dispatch and list no modules at all.
    if (module_count != 0 && !modules) {
        return step_failed(BringupStep::Module,
                           "backend_module_table reported " +
                               std::to_string(module_count) +
                               " module(s) and left the table null",
                           failed, err);
    }
    for (unsigned i = 0; i < module_count; ++i) {
        const BringupModule &mod = modules[i];
        if (!mod.name || !mod.run) {
            return step_failed(BringupStep::Module,
                               "backend_module_table returned an incomplete "
                               "entry at index " + std::to_string(i),
                               failed, err);
        }
        if (!mod.run(out->ctx, out->alloc, out->diag, &reason)) {
            return step_failed(BringupStep::Module,
                               std::string(mod.name) + ": " + reason, failed,
                               err);
        }
    }

    // Every pipeline this backend uses has now been created: a module's shader
    // library is compiled as the module comes up, and none creates one later.
    // So this is the point at which the run's pipeline archives are complete
    // and worth writing.
    //
    // A failure to write is reported and not fatal. The archive is a cache;
    // losing it costs the next run its startup time and nothing else.
    std::string flush_err;
    context_archive_flush(out->ctx, &flush_err);
    if (!flush_err.empty()) {
        logging::info("metal: pipeline archive not written: %s",
                      flush_err.c_str());
    }
    ShaderCacheReport cache = {};
    context_shader_cache_report(out->ctx, &cache);
    const char *cache_dir = context_shader_cache_dir(out->ctx);
    logging::info(
        "metal: %u shader librar(ies), %u pre-compiled in %.1f ms and %u "
        "compiled from source in %.1f ms; %u pipeline(s) "
        "from archive, %u compiled, %u uncached, in %.1f ms; archives %u "
        "loaded, %u new, %u written (open %.1f ms, write %.1f ms); cache '%s'; "
        "library dir '%s'",
        cache.libraries, cache.libraries_prebuilt, cache.prebuilt_load_ms,
        cache.libraries - cache.libraries_prebuilt, cache.compile_ms,
        cache.pipeline_hits, cache.pipeline_misses, cache.pipelines_uncached,
        cache.pipeline_ms, cache.archives_loaded, cache.archives_started,
        cache.archives_written, cache.archive_open_ms, cache.serialize_ms,
        (cache_dir && cache_dir[0]) ? cache_dir : "(disabled)",
        context_prebuilt_library_dir(out->ctx));
    return true;
}

}  // namespace metal_backend
