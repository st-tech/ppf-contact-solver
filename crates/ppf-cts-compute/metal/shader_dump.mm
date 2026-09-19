// File: shader_dump.mm
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// `ppf-metal-shader-dump`, the build-side half of the pre-compiled Metal
// library. It brings the backend's shader libraries up exactly as a solver run
// does, with no scene, and reports what the caches did.
//
// WHAT IT IS FOR. `MTLLibrary` has no serialization of its own, so a
// pre-compiled library is a `.metallib` produced by `xcrun metal` at build
// time. To compile one, the build needs the assembled translation unit and the
// exact compile options the runtime would have used; `$PPF_METAL_LIBRARY_DUMP_DIR`
// makes the backend write both, as `<key>.metal` and `<key>.flags`, and this
// program is what runs the backend far enough to produce them without a
// dataset, a session directory or a GPU-resident scene.
//
// WHY IT LINKS THE DYLIB RATHER THAN THE OBJECTS. Two properties follow from it
// and neither is incidental. The text dumped is what the SHIPPED backend
// assembles, not what a second build of the same sources would. And the
// directory the pre-compiled lookup resolves is the one holding
// libppfbe_metal.dylib, because `library_directory()` asks `dladdr` where
// its own code came from, so this program looks for artifacts exactly where the
// solver will and where the build installs them.
//
// IT IS ALSO A GATE ON WHATEVER THE LINKED BACKEND CHECKS. Bringing a module up
// is what compiles its shader library in the first place, and a module is free
// to refuse; the backend this links declares its modules through
// backend_module_table. So a machine whose backend refuses at bring-up cannot
// produce an artifact here, and a build that ships one has run whatever that
// backend checks on the hardware it shipped from.
//
// IT WRITES NOTHING BY ITSELF. With no `$PPF_METAL_LIBRARY_DUMP_DIR` set it is
// a startup timing probe and a parity check. `PPF_METAL_LIBRARY=0` disables the
// pre-compiled lookup, and `PPF_METAL_LIBRARY_DIR` moves it, which is how the
// build measures with and without.

#include "common.hpp"

#include "bringup.hpp"
#include "metal_context.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// The other half of the backend's logging seam. The dylib deliberately leaves
// `print_rust` undefined and binds it at load time to whatever loaded it, which
// for the solver is the Rust host and here is this program. Without a
// definition the dylib's first log line would resolve to address 0.
extern "C" void print_rust(const char *message) {
    std::printf("[backend] %s\n", message ? message : "(null message)");
    std::fflush(stdout);
}

namespace {

using namespace metal_backend;

// Non-fatal backend messages. Routed through the same logging channel as
// everything else so the build log carries one ordered account.
void tool_log(const char *message) {
    logging::info("%s", message ? message : "(null message)");
}

const char *step_name(BringupStep step) {
    switch (step) {
    case BringupStep::None:
        return "none";
    case BringupStep::Device:
        return "device";
    case BringupStep::MathMode:
        return "math-mode sentinels";
    case BringupStep::Allocator:
        return "arena allocator";
    case BringupStep::Diagnostics:
        return "diagnostic channel";
    case BringupStep::ShaderPrologue:
        return "shader prologue";
    case BringupStep::Module:
        // The module's own name is at the front of the error text, so this
        // stays generic on purpose: which modules a backend library brings up
        // is its business, and repeating the list here would put it in two
        // places, one of which nothing checks.
        return "backend module";
    }
    return "unknown";
}

}  // namespace

int main() {
    // A dump directory that does not exist is a caller mistake, and the
    // backend's writes would fail one file at a time with a message each. Say
    // it once, up front, and refuse.
    const char *dump_dir = std::getenv("PPF_METAL_LIBRARY_DUMP_DIR");
    if (dump_dir && dump_dir[0] != '\0') {
        if (FILE *probe = std::fopen((std::string(dump_dir) + "/.writable").c_str(),
                                     "wb")) {
            std::fclose(probe);
            std::remove((std::string(dump_dir) + "/.writable").c_str());
        } else {
            std::fprintf(stderr,
                         "ppf-metal-shader-dump: PPF_METAL_LIBRARY_DUMP_DIR "
                         "'%s' is not a writable directory\n",
                         dump_dir);
            return 2;
        }
    }

    Bringup brought_up;
    BringupStep failed_step = BringupStep::None;
    std::string err;
    if (!backend_bringup(&tool_log, &brought_up, &failed_step, &err)) {
        std::fprintf(stderr, "ppf-metal-shader-dump: %s failed: %s\n",
                     step_name(failed_step), err.c_str());
        return 1;
    }

    ShaderCacheReport cache = {};
    context_shader_cache_report(brought_up.ctx, &cache);

    // One machine-readable block, so the build script reads counts rather than
    // parsing prose. The line the build turns on is the second: every library
    // pre-compiled means the artifacts on disk are the ones this backend asks
    // for, under the names it computes.
    std::printf("[shader-dump] device %s\n",
                context_device_name(brought_up.ctx));
    std::printf("[shader-dump] libraries %u prebuilt %u source %u\n",
                cache.libraries, cache.libraries_prebuilt,
                cache.libraries - cache.libraries_prebuilt);
    std::printf("[shader-dump] library_ms load %.1f compile %.1f\n",
                cache.prebuilt_load_ms, cache.compile_ms);
    std::printf("[shader-dump] pipelines archived %u compiled %u uncached %u "
                "in %.1f ms\n",
                cache.pipeline_hits, cache.pipeline_misses,
                cache.pipelines_uncached, cache.pipeline_ms);
    std::printf("[shader-dump] library_dir %s\n",
                context_prebuilt_library_dir(brought_up.ctx));
    std::printf("[shader-dump] dump_dir %s\n",
                (dump_dir && dump_dir[0]) ? dump_dir : "(none)");
    std::printf("[shader-dump] OK\n");
    std::fflush(stdout);
    return 0;
}
