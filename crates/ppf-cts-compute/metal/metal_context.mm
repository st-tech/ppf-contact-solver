// File: metal_context.mm
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Objective-C++ half of metal_context.hpp. Owns the MTLDevice, the single
// MTLCommandQueue, the one compile-options helper, and the command-buffer
// lifecycle.
//
// Three properties of this API are silent-failure paths on Metal, so each is
// converted here into a loud one at the earliest point:
//   * A compile with fast math (the DEFAULT) deletes Kahan compensation and
//     damages the eigenvalue-floored inverse. One helper sets mathMode Safe,
//     reads it back, and refuses to hand out options that do not read Safe.
//   * setBytes past 32752 B kills the process with SIGABRT and prints
//     nothing at all, so the length is checked before the call.
//   * A command buffer that was killed still reports status Completed and
//     leaves plausible stale data, so status and error are both checked.

#include "metal_context.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "pipeline_cache.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <libgen.h>
#include <vector>

#if !__has_feature(objc_arc)
#error "metal_context.mm must be compiled with -fobjc-arc"
#endif

namespace metal_backend {

namespace {

// Measured on this hardware: 32752 B of setBytes reads back bit-exact and
// 32756 B kills the process. The cap is per argument, not a pool shared
// across the encoder. The PORTABLE contract is 4 KB, so anything designed
// against the number below is relying on a device property rather than on
// the specification; keep inline payloads at or under 4096 B by design and
// treat this constant purely as the refusal threshold.
constexpr size_t kSetBytesCapBytes = 32752;
constexpr size_t kSetBytesPortableContractBytes = 4096;

// [[buffer(31)]] is a shader compile error while the runtime accepts index
// 31 in release with no exception, so the two halves of a 32-slot API
// disagree. The usable range is the one both halves accept.
constexpr unsigned kMaxBufferIndex = 30;

// MTLMathMode, from MTLLibrary.h. Spelled numerically because the enum is
// macOS 15 and later, while the KVC path below works on any SDK.
constexpr NSInteger kMathModeSafe = 0;

// What make_compile_options reports when it had to reach the setting through
// the deprecated fastMathEnabled property instead of mathMode. It is a
// DIFFERENT compiler configuration reached through a different API, so it must
// key the pipeline archive differently; any value outside the MTLMathMode enum
// would do, and a negative one cannot collide with a future enumerator.
constexpr long long kMathModeLegacyFastMathOff = -1;

// A command-buffer id packs a 16-bit slot index (1-based, 0 reserved for
// "invalid") and a 16-bit generation. Slots are recycled, and the generation
// is what makes a retired id fail loudly instead of silently naming whatever
// command buffer moved into its slot afterwards.
constexpr unsigned kCmdIndexMask = 0xFFFFu;
constexpr unsigned kCmdIndexShift = 16u;
constexpr unsigned kMaxCommandSlots = kCmdIndexMask - 1u;

// Used where the failing call has no error channel and continuing would
// produce a silently wrong answer instead of a crash. This is not a printed
// and swallowed error: nothing continues past it.
[[noreturn]] void fatal(const std::string &msg) {
    std::fprintf(stderr, "ppf-cts metal backend: fatal: %s\n", msg.c_str());
    std::fflush(stderr);
    std::abort();
}

void set_err(std::string *err, const std::string &msg) {
    if (err) {
        *err = msg;
    }
}

std::string ns_error_text(NSError *error) {
    if (!error) {
        return std::string("(no NSError)");
    }
    NSString *text = [error localizedDescription];
    return std::string(text ? [text UTF8String] : "(no description)");
}

const char *status_name(MTLCommandBufferStatus status) {
    switch (status) {
    case MTLCommandBufferStatusNotEnqueued:
        return "NotEnqueued";
    case MTLCommandBufferStatusEnqueued:
        return "Enqueued";
    case MTLCommandBufferStatusCommitted:
        return "Committed";
    case MTLCommandBufferStatusScheduled:
        return "Scheduled";
    case MTLCommandBufferStatusCompleted:
        return "Completed";
    case MTLCommandBufferStatusError:
        return "Error";
    }
    return "Unknown";
}

// The one place a finished command buffer is judged, so the compute path and
// the blit path cannot come to judge it differently. Both halves, every time: a
// command buffer killed by the driver reports status Completed and leaves
// plausible stale data in its output buffers, so the status alone would pass;
// conversely a status that is not Completed can arrive with no NSError
// attached. 'what' names the caller in the message.
bool finished_ok(id<MTLCommandBuffer> cb, const char *what, std::string *err) {
    const MTLCommandBufferStatus status = [cb status];
    NSError *cb_error = [cb error];
    if (status == MTLCommandBufferStatusCompleted && cb_error == nil) {
        return true;
    }
    set_err(err, std::string(what) + " failed: status " + status_name(status) +
                     " (" + std::to_string(static_cast<long long>(status)) +
                     "), error " + ns_error_text(cb_error));
    return false;
}

struct CommandSlot {
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    unsigned generation = 0;
    bool open = false;
};

}  // namespace

struct Context {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    std::string device_name;

    // Where non-fatal messages go. Required at construction, because the
    // alternative default, stderr, is read by the frontend as a crashed run.
    void (*log)(const char *message) = nullptr;

    // Element 0 of each table is the reserved invalid id, so a zeroed field
    // can never name a real object.
    std::vector<id<MTLLibrary>> libraries;
    std::vector<id<MTLComputePipelineState>> pipelines;
    std::vector<id<MTLBuffer>> buffers;

    // One pipeline archive per library, indexed by library id. An entry is
    // null when this library has no usable cache, which every reader tolerates.
    std::vector<PipelineCache *> library_caches;

    ShaderCacheReport cache_report = {};

    std::vector<CommandSlot> commands;
    std::vector<unsigned> free_commands;
};

namespace {

Context *require_context(Context *ctx, const char *what) {
    if (!ctx) {
        fatal(std::string(what) + ": null context");
    }
    return ctx;
}

id<MTLLibrary> resolve_library(Context *ctx, unsigned library) {
    if (library == 0 || library >= ctx->libraries.size()) {
        fatal("library id " + std::to_string(library) + " is not valid");
    }
    return ctx->libraries[library];
}

id<MTLComputePipelineState> resolve_pipeline(Context *ctx, unsigned pipeline) {
    if (pipeline == 0 || pipeline >= ctx->pipelines.size()) {
        fatal("pipeline id " + std::to_string(pipeline) + " is not valid");
    }
    return ctx->pipelines[pipeline];
}

id<MTLBuffer> resolve_buffer(Context *ctx, unsigned buffer_id) {
    if (buffer_id == 0 || buffer_id >= ctx->buffers.size()) {
        fatal("buffer id " + std::to_string(buffer_id) + " is not valid");
    }
    return ctx->buffers[buffer_id];
}

unsigned pack_cmd_id(unsigned index, unsigned generation) {
    return ((generation & kCmdIndexMask) << kCmdIndexShift) | (index + 1u);
}

CommandSlot &resolve_command(Context *ctx, unsigned cmd) {
    const unsigned index = cmd & kCmdIndexMask;
    if (index == 0 || index > ctx->commands.size()) {
        fatal("command buffer id " + std::to_string(cmd) + " is not valid");
    }
    CommandSlot &slot = ctx->commands[index - 1u];
    const unsigned generation = (cmd >> kCmdIndexShift) & kCmdIndexMask;
    if (!slot.open || (slot.generation & kCmdIndexMask) != generation) {
        fatal("command buffer id " + std::to_string(cmd) +
              " has already been committed or was never begun");
    }
    return slot;
}

void retire_command(Context *ctx, unsigned index) {
    CommandSlot &slot = ctx->commands[index];
    slot.cb = nil;
    slot.enc = nil;
    slot.open = false;
    slot.generation += 1u;
    ctx->free_commands.push_back(index);
}

// The single place compile options are built. Every library in the backend
// comes through here, because fast math is Metal's compile DEFAULT and it
// deletes exactly the compensated arithmetic this solver's accuracy rests
// on. The value is read BACK rather than assumed: a setter that silently
// did nothing would leave every kernel compiled fast with no other symptom.
//
// 'math_mode' receives the value that was read back, and it is what keys the
// pipeline archive. A binary archive stores compiled ISA, so an archive written
// under one math mode must never be loadable under another; taking the key
// component from the READ-BACK value rather than from a literal makes that hold
// by construction instead of by review.
MTLCompileOptions *make_compile_options(std::string *err,
                                        long long *math_mode) {
    if (math_mode) {
        *math_mode = 0;
    }
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (!options) {
        set_err(err, "MTLCompileOptions allocation failed");
        return nil;
    }

    // mathMode is macOS 15 and later. Reaching it by name rather than by
    // selector literal keeps this file compilable against an older SDK,
    // where the fastMathEnabled fallback below is the only switch there is.
    SEL setter = NSSelectorFromString(@"setMathMode:");
    if ([options respondsToSelector:setter]) {
        @try {
            [options setValue:@(kMathModeSafe) forKey:@"mathMode"];
        } @catch (NSException *ex) {
            set_err(err, std::string("setting mathMode = Safe threw: ") +
                             [[ex reason] UTF8String]);
            return nil;
        }
        NSNumber *readback = nil;
        @try {
            readback = [options valueForKey:@"mathMode"];
        } @catch (NSException *ex) {
            set_err(err, std::string("reading mathMode back threw: ") +
                             [[ex reason] UTF8String]);
            return nil;
        }
        if (![readback isKindOfClass:[NSNumber class]] ||
            [readback integerValue] != kMathModeSafe) {
            set_err(err,
                    std::string("mathMode did not read back as Safe (0); got ") +
                        (readback ? [[readback description] UTF8String] : "nil") +
                        ". Compiling in any other mode reassociates float "
                        "arithmetic and drops Kahan compensation.");
            return nil;
        }
        if (math_mode) {
            *math_mode = static_cast<long long>([readback integerValue]);
        }
        return options;
    }

// Older SDK or older OS. fastMathEnabled is the same setting under its
// previous name, so NO here means Safe.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
    const BOOL fast_readback = options.fastMathEnabled;
#pragma clang diagnostic pop
    if (fast_readback != NO) {
        set_err(err, "fastMathEnabled did not read back as NO, and the "
                     "mathMode property is unavailable on this SDK, so there "
                     "is no way to compile with safe math on this host");
        return nil;
    }
    if (math_mode) {
        *math_mode = kMathModeLegacyFastMathOff;
    }
    return options;
}

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch())
        .count();
}

// ---------------------------------------------------------------------------
// THE PRE-COMPILED LIBRARY, AND WHY IT IS A LOOKUP RATHER THAN A REPLACEMENT.
//
// newLibraryWithSource is the only way to get an MTLLibrary out of MSL text,
// and on a cold machine it is 6.5 s for this backend's solver library. An
// offline `.metallib`, compiled by `xcrun metal` at BUILD time, is the same
// program in AIR form, and newLibraryWithURL loads one in about 4 ms. That is
// what makes the Metal backend pre-compiled rather than compiled at run time.
//
// It is an ADDITION and never a replacement. The Metal Toolchain that produces
// a `.metallib` is a separate downloadable component, absent on some machines
// with full Xcode installed, so a build that required it would put a download
// between a contributor and their first build. Every failure below therefore
// falls back to compiling the source, which is why this function returns nil
// rather than an error on every path.
//
// STALENESS IS THE HAZARD, NOT SPEED. A `.metallib` that does not match the
// kernels now in the tree is a wrong answer, not a slow start, and unlike the
// OS shader cache this one is ours to key. The file is NAMED by a hash of
// everything that decides what the AIR contains, so an edited kernel names a
// file that is not there and the run compiles from source. Three components,
// each load-bearing:
//
//   source          the assembled translation unit, byte for byte. This is the
//                   component an edited kernel moves.
//   math mode       read BACK out of the compile options, never a literal, so
//                   the value that keys the file cannot drift from the value
//                   the source would have been compiled under. mathMode Safe is
//                   a correctness setting for this solver: fast math deletes
//                   Kahan compensation outright, so a `.metallib` built fast
//                   must never load into a safe-math run.
//   language ver    read back out of the same options. The offline compiler
//                   and the runtime front end each have a DEFAULT standard, and
//                   they agree today (both Metal 4.0, measured); if an OS
//                   update moves either, the key moves and the artifact misses
//                   rather than loading a program compiled to other rules.
//
// The DEVICE is deliberately NOT in the key. A `.metallib` holds AIR, which is
// device-independent; the ISA half is the pipeline archive's problem, and that
// one is keyed on the device, the registry id and the OS build. One artifact
// therefore serves every Apple GPU. A driver that will not accept the AIR at
// all returns nil from newLibraryWithURL, which falls back like any other miss.
//
// WHERE THE FILE LIVES. $PPF_METAL_LIBRARY_DIR if set, otherwise the directory
// holding this dylib, which is where the build installs it and where a bundle
// keeps it next to the backend it belongs to. It is a BUILD ARTIFACT and must
// never be written under crates/ppf-cts-solver/src: two build scripts watch
// that tree recursively and cargo reads a directory in rerun-if-changed as
// "rerun if any descendant changes", so an artifact anywhere inside costs a
// measured 48.3 s rebuild of two crates on every later build.
//
// PPF_METAL_LIBRARY=0 disables the lookup, which is the A/B lever for measuring
// what it buys.
constexpr char kLibraryKeyRecipeVersion[] =
    "ppf-cts metal offline library key v1";
constexpr char kLibrarySuffix[] = ".metallib";

// Every component is length-prefixed so that no two different tuples can
// concatenate to the same bytes.
void hash_library_component(CC_SHA256_CTX *ctx, const char *label,
                            const void *data, size_t length) {
    const unsigned long long label_len = std::strlen(label);
    const unsigned long long data_len = length;
    CC_SHA256_Update(ctx, &label_len, sizeof(label_len));
    CC_SHA256_Update(ctx, label, static_cast<CC_LONG>(label_len));
    CC_SHA256_Update(ctx, &data_len, sizeof(data_len));
    const unsigned char *bytes = static_cast<const unsigned char *>(data);
    size_t remaining = length;
    while (remaining > 0) {
        const size_t chunk = remaining < (1u << 30) ? remaining : (1u << 30);
        CC_SHA256_Update(ctx, bytes, static_cast<CC_LONG>(chunk));
        bytes += chunk;
        remaining -= chunk;
    }
}

std::string compute_library_key(const char *source, long long math_mode,
                                long long language_version) {
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    hash_library_component(&ctx, "recipe", kLibraryKeyRecipeVersion,
                           std::strlen(kLibraryKeyRecipeVersion));
    hash_library_component(&ctx, "mathmode", &math_mode, sizeof(math_mode));
    hash_library_component(&ctx, "langver", &language_version,
                           sizeof(language_version));
    hash_library_component(&ctx, "source", source, std::strlen(source));
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &ctx);
    static const char *const hex = "0123456789abcdef";
    std::string out;
    out.reserve(sizeof(digest) * 2);
    for (unsigned char byte : digest) {
        out.push_back(hex[byte >> 4]);
        out.push_back(hex[byte & 0x0F]);
    }
    return out;
}

// The directory pre-compiled libraries are read from, resolved once. Empty
// means the lookup is off, which is never an error.
const std::string &library_directory() {
    static const std::string dir = []() -> std::string {
        const char *disable = std::getenv("PPF_METAL_LIBRARY");
        if (disable && std::strcmp(disable, "0") == 0) {
            return std::string();
        }
        const char *override_dir = std::getenv("PPF_METAL_LIBRARY_DIR");
        if (override_dir && override_dir[0] != '\0') {
            return std::string(override_dir);
        }
        // The directory this dylib was loaded from. Taking it from the loader
        // rather than from a build-time string is what makes a relocated
        // bundle find its own artifact: the backend is loaded by @rpath, so
        // several build directories can each hold a copy and only the one the
        // loader actually resolved is the right neighbor to read.
        Dl_info info;
        std::memset(&info, 0, sizeof(info));
        if (dladdr(reinterpret_cast<const void *>(&compute_library_key),
                   &info) == 0 ||
            !info.dli_fname || info.dli_fname[0] == '\0') {
            return std::string();
        }
        std::string path(info.dli_fname);
        std::vector<char> mutable_path(path.begin(), path.end());
        mutable_path.push_back('\0');
        const char *parent = dirname(mutable_path.data());
        return parent ? std::string(parent) : std::string();
    }();
    return dir;
}

// Loads the pre-compiled library for 'source', or nil when there is not one to
// load. Never an error: a miss is the ordinary case on a machine whose build
// had no Metal Toolchain, and on an edited kernel.
id<MTLLibrary> load_prebuilt_library(id<MTLDevice> device,
                                     const std::string &key,
                                     void (*log)(const char *message),
                                     double *load_ms) {
    if (load_ms) {
        *load_ms = 0.0;
    }
    const std::string &dir = library_directory();
    if (dir.empty()) {
        return nil;
    }
    const std::string path = dir + "/" + key + kLibrarySuffix;
    const double started = now_ms();
    NSString *ns = [NSString stringWithUTF8String:path.c_str()];
    if (!ns || ![[NSFileManager defaultManager] fileExistsAtPath:ns]) {
        return nil;
    }
    NSError *load_error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithURL:[NSURL fileURLWithPath:ns] error:&load_error];
    if (load_ms) {
        *load_ms = now_ms() - started;
    }
    if (!library && log) {
        // Through the log sink and never stderr: in this process stderr is the
        // crash channel, and a run that recovers by compiling its source did
        // not crash.
        const std::string message =
            "[metal] the pre-compiled library " + path +
            " is present but did not load (" + ns_error_text(load_error) +
            "); compiling the shader source instead";
        log(message.c_str());
    }
    return library;
}

// The build-side half: writes what an offline compilation needs, and nothing
// that would let it be performed under the wrong settings.
//
// $PPF_METAL_LIBRARY_DUMP_DIR receives, per library, `<key>.metal` (the
// assembled unit, byte for byte what this run compiled) and `<key>.flags` (the
// arguments `xcrun metal` must be given to produce AIR matching this run's
// compile options). The producing script reads the flags rather than restating
// them: mathMode is a correctness setting here, so a build that hardcoded
// `-fmetal-math-mode=safe` would be one edit away from silently shipping fast
// math, while a build that copies what the runtime read back cannot be.
//
// The key names the file, so the artifact the build produces lands under
// exactly the name the run-time lookup will ask for.
void dump_library_source(const char *source, const std::string &key,
                         long long math_mode, long long language_version,
                         void (*log)(const char *message)) {
    const char *dir = std::getenv("PPF_METAL_LIBRARY_DUMP_DIR");
    if (!dir || dir[0] == '\0' || key.empty()) {
        return;
    }
    auto complain = [&](const std::string &what) {
        if (log) {
            const std::string message =
                "[metal] PPF_METAL_LIBRARY_DUMP_DIR is set but " + what;
            log(message.c_str());
        }
    };

    const std::string base = std::string(dir) + "/" + key;
    FILE *f = std::fopen((base + ".metal").c_str(), "wb");
    if (!f) {
        complain("the shader could not be written to " + base + ".metal");
        return;
    }
    const size_t n = std::strlen(source);
    const bool wrote = (n == 0) || (std::fwrite(source, 1, n, f) == n);
    const bool closed = (std::fclose(f) == 0);
    if (!wrote || !closed) {
        complain("writing " + base + ".metal failed part way");
        return;
    }

    // MTLMathMode: 0 Safe, 1 Relaxed, 2 Fast. kMathModeLegacyFastMathOff (-1)
    // means the setting was reached through the deprecated fastMathEnabled
    // property, which is Safe under its previous name.
    const char *math_flag = nullptr;
    switch (math_mode) {
    case 0:
    case kMathModeLegacyFastMathOff:
        math_flag = "-fmetal-math-mode=safe";
        break;
    case 1:
        math_flag = "-fmetal-math-mode=relaxed";
        break;
    case 2:
        math_flag = "-fmetal-math-mode=fast";
        break;
    default:
        break;
    }
    if (!math_flag) {
        complain("the math mode read back as " + std::to_string(math_mode) +
                 ", which has no offline flag; no flags file was written, so "
                 "no artifact can be built for this library");
        return;
    }
    // MTLLanguageVersion packs (major << 16) | minor.
    const long long major = (language_version >> 16) & 0xFFFF;
    const long long minor = language_version & 0xFFFF;

    f = std::fopen((base + ".flags").c_str(), "wb");
    if (!f) {
        complain("the flags could not be written to " + base + ".flags");
        return;
    }
    std::fprintf(f, "%s -std=metal%lld.%lld\n", math_flag, major, minor);
    if (std::fclose(f) != 0) {
        complain("writing " + base + ".flags failed part way");
    }
}

}  // namespace

Context *context_create(void (*log)(const char *message), std::string *err) {
    if (!log) {
        fatal("context_create: a log sink is required. Non-fatal messages must "
              "not reach stderr, which the frontend reads as a crashed run.");
    }
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            set_err(err, "no Metal device on this host");
            return nullptr;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            set_err(err, "MTLDevice::newCommandQueue returned nil");
            return nullptr;
        }

        Context *ctx = new Context();
        ctx->device = device;
        ctx->queue = queue;
        ctx->log = log;
        NSString *name = [device name];
        ctx->device_name = name ? [name UTF8String] : "(unnamed Metal device)";
        ctx->libraries.push_back(nil);
        ctx->library_caches.push_back(nullptr);
        ctx->pipelines.push_back(nil);
        ctx->buffers.push_back(nil);
        return ctx;
    }
}

void context_destroy(Context *ctx) {
    if (!ctx) {
        return;
    }
    @autoreleasepool {
        // Teardown persists what is still pending, so this function releases
        // nothing a caller would have to remember to save first. It is not the
        // path that writes an archive today: nothing calls context_destroy
        // (the solver process exits and the OS reclaims), and main.mm flushes
        // explicitly once the last pipeline exists. Clean caches write
        // nothing, so a flush that has already happened costs one predicate
        // per library.
        std::string flush_err;
        if (!context_archive_flush(ctx, &flush_err) && !flush_err.empty() &&
            ctx->log) {
            ctx->log((flush_err + " (the next run recompiles its pipelines)")
                         .c_str());
        }
        for (PipelineCache *cache : ctx->library_caches) {
            pipeline_cache_close(cache);
        }
        ctx->library_caches.clear();

        // An open command buffer at teardown holds encoded work that was
        // never committed, which is what an error return between begin and
        // commit leaves behind. Dropping it is the correct outcome (the work
        // never ran and never will), but it is reported rather than dropped
        // quietly, because an unbalanced begin outside that error path is a
        // defect in the caller.
        for (size_t i = 0; i < ctx->commands.size(); ++i) {
            CommandSlot &slot = ctx->commands[i];
            if (!slot.open) {
                continue;
            }
            std::fprintf(stderr,
                         "ppf-cts metal backend: discarding uncommitted "
                         "command buffer (slot %zu) at context teardown\n",
                         i);
            [slot.enc endEncoding];
            slot.enc = nil;
            slot.cb = nil;
            slot.open = false;
        }
        std::fflush(stderr);
        delete ctx;
    }
}

const char *context_device_name(Context *ctx) {
    require_context(ctx, "context_device_name");
    return ctx->device_name.c_str();
}

unsigned long long context_max_buffer_length(Context *ctx) {
    require_context(ctx, "context_max_buffer_length");
    return static_cast<unsigned long long>([ctx->device maxBufferLength]);
}

unsigned context_max_threads_per_threadgroup(Context *ctx) {
    require_context(ctx, "context_max_threads_per_threadgroup");
    return static_cast<unsigned>([ctx->device maxThreadsPerThreadgroup].width);
}

unsigned context_max_threadgroup_memory(Context *ctx) {
    require_context(ctx, "context_max_threadgroup_memory");
    return static_cast<unsigned>([ctx->device maxThreadgroupMemoryLength]);
}

unsigned context_compile_library(Context *ctx, const char *source,
                                 std::string *err) {
    require_context(ctx, "context_compile_library");
    if (!source) {
        set_err(err, "context_compile_library: null source");
        return 0;
    }
    @autoreleasepool {
        long long math_mode = 0;
        MTLCompileOptions *options = make_compile_options(err, &math_mode);
        if (!options) {
            return 0;
        }
        NSString *text = [NSString stringWithUTF8String:source];
        if (!text) {
            set_err(err, "shader source is not valid UTF-8");
            return 0;
        }

        // The pre-compiled library first. The language version is read off the
        // same options object the source compile would use, so the artifact is
        // keyed on the standard this run would actually have compiled to.
        const long long language_version =
            static_cast<long long>([options languageVersion]);
        const std::string library_key =
            compute_library_key(source, math_mode, language_version);
        double load_ms = 0.0;
        id<MTLLibrary> library =
            load_prebuilt_library(ctx->device, library_key, ctx->log, &load_ms);
        const bool prebuilt = (library != nil);
        if (library) {
            ctx->cache_report.libraries_prebuilt += 1u;
            ctx->cache_report.prebuilt_load_ms += load_ms;
        } else {
            const double compile_started = now_ms();
            NSError *compile_error = nil;
            library = [ctx->device newLibraryWithSource:text
                                                options:options
                                                  error:&compile_error];
            ctx->cache_report.compile_ms += now_ms() - compile_started;
            if (!library) {
                set_err(err,
                        "MSL compilation failed: " + ns_error_text(compile_error));
                return 0;
            }
        }

        // The shader dump, when asked for. It writes the assembled unit under
        // its own key together with the exact compiler arguments this run's
        // options amount to, which is what a build step compiles offline. The
        // flags are DERIVED from the options object rather than restated, so
        // the artifact cannot be built under settings the runtime would not
        // have used.
        dump_library_source(source, library_key, math_mode, language_version,
                            ctx->log);
        if (ctx->libraries.size() > 0xFFFFFFFFull - 1ull) {
            set_err(err, "library id space exhausted");
            return 0;
        }

        // The library's pipeline archive. Keyed on the same source bytes, on
        // the math mode just read back, and on which route produced this
        // library, so an edited kernel names a file that does not exist rather
        // than loading one that no longer describes it, and the two routes do
        // not each miss on the other's file.
        double open_ms = 0.0;
        bool loaded = false;
        const std::string archive_origin =
            prebuilt ? ("metallib:" + library_key) : std::string("source");
        PipelineCache *cache =
            pipeline_cache_open(ctx->device, source, math_mode,
                                archive_origin.c_str(), ctx->log, &open_ms,
                                &loaded);
        ctx->cache_report.archive_open_ms += open_ms;
        ctx->cache_report.libraries += 1u;
        if (cache) {
            if (loaded) {
                ctx->cache_report.archives_loaded += 1u;
            } else {
                ctx->cache_report.archives_started += 1u;
            }
        }

        ctx->libraries.push_back(library);
        ctx->library_caches.push_back(cache);
        return static_cast<unsigned>(ctx->libraries.size() - 1);
    }
}

unsigned context_load_library(Context *ctx, const char *path,
                              std::string *err) {
    require_context(ctx, "context_load_library");
    if (!path || !*path) {
        set_err(err, "context_load_library: null or empty path");
        return 0;
    }
    @autoreleasepool {
        NSString *ns = [NSString stringWithUTF8String:path];
        if (!ns) {
            set_err(err, "context_load_library: path is not valid UTF-8");
            return 0;
        }
        NSError *load_error = nil;
        id<MTLLibrary> library =
            [ctx->device newLibraryWithURL:[NSURL fileURLWithPath:ns]
                                     error:&load_error];
        if (!library) {
            set_err(err, std::string("newLibraryWithURL failed for ") + path +
                             ": " +
                             (load_error
                                  ? load_error.localizedDescription.UTF8String
                                  : "(no error reported)"));
            return 0;
        }
        if (ctx->libraries.size() > 0xFFFFFFFFull - 1ull) {
            set_err(err, "library id space exhausted");
            return 0;
        }
        // No pipeline archive: that cache is keyed on source bytes and there
        // are none. The slot is still pushed so the two vectors stay index
        // aligned, which every resolve below depends on.
        ctx->cache_report.libraries += 1u;
        ctx->libraries.push_back(library);
        ctx->library_caches.push_back(nullptr);
        return static_cast<unsigned>(ctx->libraries.size() - 1);
    }
}

unsigned context_make_pipeline(Context *ctx, unsigned library,
                               const char *fn_name, std::string *err) {
    require_context(ctx, "context_make_pipeline");
    if (!fn_name) {
        set_err(err, "context_make_pipeline: null function name");
        return 0;
    }
    @autoreleasepool {
        id<MTLLibrary> lib = resolve_library(ctx, library);
        NSString *name = [NSString stringWithUTF8String:fn_name];
        if (!name) {
            set_err(err, "function name is not valid UTF-8");
            return 0;
        }
        id<MTLFunction> function = [lib newFunctionWithName:name];
        if (!function) {
            set_err(err, std::string("kernel function '") + fn_name +
                             "' is not present in library " +
                             std::to_string(library));
            return 0;
        }
        const double pipeline_started = now_ms();
        PipelineCacheOutcome outcome = PipelineCacheOutcome::Unavailable;
        id<MTLComputePipelineState> pipeline =
            pipeline_cache_make(ctx->library_caches[library], ctx->device,
                                function, fn_name, ctx->log, &outcome, err);
        ctx->cache_report.pipeline_ms += now_ms() - pipeline_started;
        switch (outcome) {
        case PipelineCacheOutcome::Hit:
            ctx->cache_report.pipeline_hits += 1u;
            break;
        case PipelineCacheOutcome::Miss:
            ctx->cache_report.pipeline_misses += 1u;
            break;
        case PipelineCacheOutcome::Unavailable:
            ctx->cache_report.pipelines_uncached += 1u;
            break;
        }
        if (!pipeline) {
            return 0;
        }
        ctx->pipelines.push_back(pipeline);
        return static_cast<unsigned>(ctx->pipelines.size() - 1);
    }
}

bool context_archive_flush(Context *ctx, std::string *err) {
    require_context(ctx, "context_archive_flush");
    if (err) {
        err->clear();
    }
    bool wrote_any = false;
    for (PipelineCache *cache : ctx->library_caches) {
        if (!cache) {
            continue;
        }
        double ms = 0.0;
        std::string flush_err;
        const bool wrote = pipeline_cache_flush(cache, &ms, &flush_err);
        ctx->cache_report.serialize_ms += ms;
        if (wrote) {
            ctx->cache_report.archives_written += 1u;
            wrote_any = true;
        } else if (!flush_err.empty()) {
            // First failure wins the message; the loop still tries the rest,
            // because one unwritable archive says nothing about another.
            if (err && err->empty()) {
                *err = flush_err;
            }
        }
    }
    return wrote_any;
}

void context_shader_cache_report(Context *ctx, ShaderCacheReport *out) {
    require_context(ctx, "context_shader_cache_report");
    if (out) {
        *out = ctx->cache_report;
    }
}

const char *context_shader_cache_dir(Context *ctx) {
    require_context(ctx, "context_shader_cache_dir");
    return pipeline_cache_directory().c_str();
}

const char *context_prebuilt_library_dir(Context *ctx) {
    require_context(ctx, "context_prebuilt_library_dir");
    const std::string &dir = library_directory();
    return dir.empty() ? "(disabled)" : dir.c_str();
}

unsigned context_new_buffer(Context *ctx, unsigned long long length,
                            std::string *err) {
    require_context(ctx, "context_new_buffer");
    if (length == 0) {
        set_err(err, "context_new_buffer: zero-length allocation");
        return 0;
    }
    const unsigned long long cap =
        static_cast<unsigned long long>([ctx->device maxBufferLength]);
    if (length > cap) {
        set_err(err, "context_new_buffer: requested " + std::to_string(length) +
                         " B exceeds this device's maxBufferLength of " +
                         std::to_string(cap) + " B");
        return 0;
    }
    @autoreleasepool {
        id<MTLBuffer> buffer =
            [ctx->device newBufferWithLength:static_cast<NSUInteger>(length)
                                     options:MTLResourceStorageModeShared];
        if (!buffer) {
            set_err(err, "newBufferWithLength returned nil for " +
                             std::to_string(length) + " B");
            return 0;
        }
        ctx->buffers.push_back(buffer);
        return static_cast<unsigned>(ctx->buffers.size() - 1);
    }
}

void *context_buffer_contents(Context *ctx, unsigned buffer_id) {
    require_context(ctx, "context_buffer_contents");
    id<MTLBuffer> buffer = resolve_buffer(ctx, buffer_id);
    return [buffer contents];
}

unsigned long long context_buffer_length(Context *ctx, unsigned buffer_id) {
    require_context(ctx, "context_buffer_length");
    id<MTLBuffer> buffer = resolve_buffer(ctx, buffer_id);
    return static_cast<unsigned long long>([buffer length]);
}

bool context_fill_buffer(Context *ctx, unsigned buffer_id,
                         unsigned long long offset, unsigned long long length,
                         unsigned char value, std::string *err) {
    require_context(ctx, "context_fill_buffer");
    if (length == 0) {
        return true;
    }
    @autoreleasepool {
        id<MTLBuffer> buffer = resolve_buffer(ctx, buffer_id);
        const unsigned long long buffer_length =
            static_cast<unsigned long long>([buffer length]);
        // Metal checks neither bound. A fill that runs past the end is dropped
        // and the command buffer still reports Completed, so a range error here
        // would leave the caller believing memory it never wrote is zero, which
        // is the exact failure this call exists to prevent.
        if (offset > buffer_length || length > buffer_length - offset) {
            set_err(err, "context_fill_buffer: range [" +
                             std::to_string(offset) + ", " +
                             std::to_string(offset + length) +
                             ") lies outside buffer " +
                             std::to_string(buffer_id) + " of " +
                             std::to_string(buffer_length) + " B");
            return false;
        }
        // The same alignment contract context_bind_buffer enforces, for the
        // same reason: every arena block starts on at least a 4-byte boundary,
        // and an offset that does not is a caller error worth naming rather
        // than a range Metal is documented to accept.
        if ((offset & 3ull) != 0ull) {
            set_err(err, "context_fill_buffer: offset " +
                             std::to_string(offset) +
                             " is not a multiple of 4");
            return false;
        }
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (!cb) {
            fatal("MTLCommandQueue::commandBuffer returned nil for a fill");
        }
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (!blit) {
            fatal("MTLCommandBuffer::blitCommandEncoder returned nil");
        }
        [blit fillBuffer:buffer
                   range:NSMakeRange(static_cast<NSUInteger>(offset),
                                     static_cast<NSUInteger>(length))
                   value:value];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return finished_ok(cb, "buffer fill", err);
    }
}

unsigned context_begin_command_buffer(Context *ctx) {
    require_context(ctx, "context_begin_command_buffer");
    @autoreleasepool {
        unsigned index;
        if (!ctx->free_commands.empty()) {
            index = ctx->free_commands.back();
            ctx->free_commands.pop_back();
        } else {
            if (ctx->commands.size() >= kMaxCommandSlots) {
                fatal("more than " + std::to_string(kMaxCommandSlots) +
                      " command buffers are open at once; a committed "
                      "command buffer returns its slot, so this means they "
                      "are being begun and never committed");
            }
            ctx->commands.push_back(CommandSlot());
            index = static_cast<unsigned>(ctx->commands.size() - 1);
        }
        CommandSlot &slot = ctx->commands[index];
        slot.cb = [ctx->queue commandBuffer];
        if (!slot.cb) {
            fatal("MTLCommandQueue::commandBuffer returned nil");
        }
        // One encoder per command buffer. Plan Section 7a: 128 dispatches
        // through one command buffer and one encoder cost 5.27 us each,
        // against 100.62 us with a command buffer and a wait per dispatch.
        slot.enc = [slot.cb computeCommandEncoder];
        if (!slot.enc) {
            fatal("MTLCommandBuffer::computeCommandEncoder returned nil");
        }
        slot.open = true;
        return pack_cmd_id(index, slot.generation);
    }
}

void context_bind_buffer(Context *ctx, unsigned cmd, unsigned index,
                         unsigned buffer_id, unsigned long long offset) {
    require_context(ctx, "context_bind_buffer");
    if (index > kMaxBufferIndex) {
        fatal("buffer binding index " + std::to_string(index) +
              " is out of range 0.." + std::to_string(kMaxBufferIndex) +
              "; the runtime accepts index 31 in release with no exception "
              "while the shader compiler rejects it, so a binding there is "
              "unreachable from the kernel");
    }
    @autoreleasepool {
        CommandSlot &slot = resolve_command(ctx, cmd);
        id<MTLBuffer> buffer = resolve_buffer(ctx, buffer_id);
        const unsigned long long length =
            static_cast<unsigned long long>([buffer length]);
        if (offset >= length) {
            fatal("buffer " + std::to_string(buffer_id) + " offset " +
                  std::to_string(offset) + " is past its length of " +
                  std::to_string(length) + " B");
        }
        // A misaligned setBuffer offset is silently FLOORED to a multiple of
        // 4: offsets 1, 2 and 3 all read the value at offset 0, with status
        // Completed and no diagnostic. The arena allocator asserts natural
        // alignment per element type at allocation time; this is the second
        // half of that check, at the point the offset actually reaches Metal.
        if ((offset & 3ull) != 0ull) {
            fatal("buffer " + std::to_string(buffer_id) + " offset " +
                  std::to_string(offset) +
                  " is not a multiple of 4; Metal floors a misaligned offset "
                  "silently and the kernel then reads the wrong element");
        }
        [slot.enc setBuffer:buffer
                     offset:static_cast<NSUInteger>(offset)
                    atIndex:index];
    }
}

bool context_set_bytes(Context *ctx, unsigned cmd, unsigned index,
                       const void *data, size_t length, std::string *err) {
    require_context(ctx, "context_set_bytes");
    if (index > kMaxBufferIndex) {
        set_err(err, "context_set_bytes: binding index " +
                         std::to_string(index) + " is out of range 0.." +
                         std::to_string(kMaxBufferIndex));
        return false;
    }
    if (!data || length == 0) {
        set_err(err, "context_set_bytes: null or zero-length payload");
        return false;
    }
    // This check is the whole reason the wrapper exists. Exceeding the cap
    // kills the process with SIGABRT (exit 134) and prints nothing on stdout
    // or stderr, no Metal assertion text of any kind, so there is no way to
    // diagnose it after the fact. The cap is per argument, not a pool shared
    // across the encoder.
    if (length > kSetBytesCapBytes) {
        set_err(err,
                "context_set_bytes: payload of " + std::to_string(length) +
                    " B exceeds this device's per-argument setBytes cap of " +
                    std::to_string(kSetBytesCapBytes) +
                    " B (the PORTABLE contract is only " +
                    std::to_string(kSetBytesPortableContractBytes) +
                    " B; anything this large belongs in a buffer)");
        return false;
    }
    @autoreleasepool {
        CommandSlot &slot = resolve_command(ctx, cmd);
        [slot.enc setBytes:data length:length atIndex:index];
    }
    return true;
}

void context_dispatch(Context *ctx, unsigned cmd, unsigned pipeline,
                      unsigned grid_x, unsigned threadgroup_x) {
    require_context(ctx, "context_dispatch");
    @autoreleasepool {
        CommandSlot &slot = resolve_command(ctx, cmd);
        id<MTLComputePipelineState> pso = resolve_pipeline(ctx, pipeline);
        if (threadgroup_x == 0) {
            fatal("context_dispatch: threadgroup size of 0");
        }
        const unsigned tg_max =
            static_cast<unsigned>([pso maxTotalThreadsPerThreadgroup]);
        if (threadgroup_x > tg_max) {
            fatal("context_dispatch: threadgroup size " +
                  std::to_string(threadgroup_x) +
                  " exceeds this pipeline's maxTotalThreadsPerThreadgroup of " +
                  std::to_string(tg_max));
        }
        if (grid_x == 0) {
            // No work. Encoding zero threadgroups is not a useful command,
            // and an empty element list is an ordinary scene, not an error.
            return;
        }
        // dispatchThreadgroups with a ceil-div grid, matching the CUDA launch
        // shape, with the in-kernel `if (gid >= n) return;` guard doing the
        // trimming. Verified bit-identical to native dispatchThreads (0 of
        // 1000192 elements differing at n = 1000003), and it keeps every
        // threadgroup full size. That is the point: under dispatchThreads the
        // tail threadgroup is genuinely short AND [[threads_per_simdgroup]]
        // reports 32 inside a live-26-lane group, so a reduction
        // transliterated from CUDA would read garbage lanes there.
        const unsigned groups = (grid_x + threadgroup_x - 1u) / threadgroup_x;
        [slot.enc setComputePipelineState:pso];
        [slot.enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threadgroup_x, 1, 1)];
    }
}

bool context_dispatch_threadgroups(Context *ctx, unsigned cmd,
                                   unsigned pipeline, unsigned groups,
                                   unsigned threads_per_group,
                                   unsigned long long bytes,
                                   std::string *err) {
    require_context(ctx, "context_dispatch_threadgroups");
    if (threads_per_group == 0) {
        fatal("context_dispatch_threadgroups: threadgroup size of 0");
    }
    if (groups == 0) {
        // No work. An empty aggregate list is an ordinary scene, not an error.
        return true;
    }
    // Metal requires the threadgroup allocation to be a multiple of 16 B. Round
    // UP: a kernel asking for 18688 B gets 18688 here, and one asking for a
    // ragged size gets the next multiple, which is strictly more than it reads.
    const unsigned long long granularity = 16ull;
    const unsigned long long rounded =
        ((bytes + granularity - 1ull) / granularity) * granularity;
    const unsigned long long limit =
        static_cast<unsigned long long>([ctx->device maxThreadgroupMemoryLength]);
    if (rounded > limit) {
        if (err) {
            *err = "context_dispatch_threadgroups: " + std::to_string(rounded) +
                   " B of threadgroup memory exceeds this device's limit of " +
                   std::to_string(limit) + " B";
        }
        return false;
    }
    @autoreleasepool {
        CommandSlot &slot = resolve_command(ctx, cmd);
        id<MTLComputePipelineState> pso = resolve_pipeline(ctx, pipeline);
        const unsigned tg_max =
            static_cast<unsigned>([pso maxTotalThreadsPerThreadgroup]);
        if (threads_per_group > tg_max) {
            if (err) {
                *err = "context_dispatch_threadgroups: threadgroup size " +
                       std::to_string(threads_per_group) +
                       " exceeds this pipeline's maxTotalThreadsPerThreadgroup "
                       "of " +
                       std::to_string(tg_max);
            }
            return false;
        }
        [slot.enc setComputePipelineState:pso];
        [slot.enc setThreadgroupMemoryLength:static_cast<NSUInteger>(rounded)
                                     atIndex:0];
        [slot.enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
    }
    return true;
}

bool context_commit_and_wait(Context *ctx, unsigned cmd, std::string *err) {
    require_context(ctx, "context_commit_and_wait");
    @autoreleasepool {
        CommandSlot &slot = resolve_command(ctx, cmd);
        const unsigned index = (cmd & kCmdIndexMask) - 1u;

        [slot.enc endEncoding];
        [slot.cb commit];
        // THE WAIT IS LOAD-BEARING BEYOND THIS CALL: it is what leaves no
        // device work in flight while caller code runs, which is what lets
        // be_host_ptr hand out an address into an allocation a dispatch reads.
        // Removing it requires setting kSynchronousExecution in
        // backend/backend.mm to false in the same change.
        [slot.cb waitUntilCompleted];

        const bool ok = finished_ok(slot.cb, "command buffer", err);
        retire_command(ctx, index);
        return ok;
    }
}

void context_abandon_command(Context *ctx, unsigned cmd) {
    require_context(ctx, "context_abandon_command");
    @autoreleasepool {
        CommandSlot &slot = resolve_command(ctx, cmd);
        const unsigned index = (cmd & kCmdIndexMask) - 1u;
        // The encoder is ended even though nothing is committed, because
        // dropping a command buffer whose encoder is still open is a Metal
        // API violation rather than a no-op. Nothing is submitted, so whatever
        // was encoded into this buffer never runs.
        [slot.enc endEncoding];
        retire_command(ctx, index);
    }
}

unsigned context_live_commands(Context *ctx) {
    require_context(ctx, "context_live_commands");
    // A retired slot is pushed onto the free list exactly once and popped
    // exactly once when it is reused, so the free list is never longer than
    // the table and the difference is the number of slots still open.
    return static_cast<unsigned>(ctx->commands.size() -
                                 ctx->free_commands.size());
}

namespace {

// Both sentinels in one kernel. Every operand is read from device memory and
// the trip count is a runtime value, so a correct compiler cannot fold
// either result: the only thing that can remove the compensation term is the
// reassociation that fast math is allowed to perform.
const char *const kMathModeSentinelSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void sentinel_math_mode(device const float *src [[buffer(0)]],
                                   device float *out [[buffer(1)]],
                                   constant uint &n [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) {
        return;
    }

    // Kahan sum of src[0..n-1] = [1e8, 1.0 x (n-1)]. Safe math keeps the
    // compensation and returns 100004096; fast math deletes it and returns
    // the naive 100000000, because 1e8 + 1.0 rounds back to 1e8.
    float sum = 0.0f;
    float comp = 0.0f;
    for (uint i = 0u; i < n; ++i) {
        float y = src[i] - comp;
        float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    out[0] = sum;

    // Reassociation, with a = 1e8, b = -1e8, c = 1. In IEEE order
    // (a + b) + c is 1 and a + (b + c) is 0. Equal outputs mean the compiler
    // reassociated the two into one expression.
    float a = src[0];
    float b = -a;
    float c = src[1];
    out[1] = (a + b) + c;
    out[2] = a + (b + c);
}
)MSL";

constexpr unsigned kSentinelCount = 4096;
constexpr float kSentinelKahanExpected = 100004096.0f;
constexpr float kSentinelKahanNaive = 100000000.0f;

std::string format_float(float value) {
    char text[64];
    std::snprintf(text, sizeof(text), "%.9g", static_cast<double>(value));
    return std::string(text);
}

}  // namespace

bool context_check_math_mode(Context *ctx, std::string *err) {
    require_context(ctx, "context_check_math_mode");

    const unsigned library = context_compile_library(ctx, kMathModeSentinelSource, err);
    if (library == 0) {
        return false;
    }
    const unsigned pipeline =
        context_make_pipeline(ctx, library, "sentinel_math_mode", err);
    if (pipeline == 0) {
        return false;
    }

    const unsigned src_buffer =
        context_new_buffer(ctx, sizeof(float) * kSentinelCount, err);
    if (src_buffer == 0) {
        return false;
    }
    const unsigned out_buffer = context_new_buffer(ctx, sizeof(float) * 4, err);
    if (out_buffer == 0) {
        return false;
    }

    float *src = static_cast<float *>(context_buffer_contents(ctx, src_buffer));
    src[0] = 1.0e8f;
    for (unsigned i = 1; i < kSentinelCount; ++i) {
        src[i] = 1.0f;
    }
    float *out = static_cast<float *>(context_buffer_contents(ctx, out_buffer));
    // A pattern the kernel cannot produce, so a kernel that writes nothing
    // fails the comparison below instead of inheriting a plausible zero.
    std::memset(out, 0xCD, sizeof(float) * 4);

    const unsigned cmd = context_begin_command_buffer(ctx);
    context_bind_buffer(ctx, cmd, 0, src_buffer, 0);
    context_bind_buffer(ctx, cmd, 1, out_buffer, 0);
    const unsigned count = kSentinelCount;
    if (!context_set_bytes(ctx, cmd, 2, &count, sizeof(count), err)) {
        // RETIRE THE SLOT ON THE WAY OUT. A return between beginning a command
        // buffer and committing it leaves the slot live for the process, and
        // `context_live_commands` is read by `be_host_ptr`, so a leak here
        // would answer every later host view with a refusal naming a cause
        // that is not the one that happened. This path fails `be_open`, so no
        // host view can follow it today, and it is retired anyway because the
        // reason it would mislead does not depend on that.
        context_abandon_command(ctx, cmd);
        return false;
    }
    context_dispatch(ctx, cmd, pipeline, 1, 1);
    if (!context_commit_and_wait(ctx, cmd, err)) {
        return false;
    }

    if (out[0] != kSentinelKahanExpected) {
        std::string msg = "math-mode sentinel (Kahan) returned " +
                          format_float(out[0]) + ", expected " +
                          format_float(kSentinelKahanExpected);
        if (out[0] == kSentinelKahanNaive) {
            msg += "; that is exactly the naive sum, so the compensation term "
                   "was optimized away and the library was compiled with fast "
                   "math rather than mathMode Safe";
        }
        set_err(err, msg);
        return false;
    }
    if (out[1] == out[2]) {
        set_err(err, "math-mode sentinel (reassociation) returned the same "
                     "value " + format_float(out[1]) +
                     " for (a+b)+c and a+(b+c) with a = 1e8, b = -1e8, c = 1; "
                     "float addition is not associative, so the compiler "
                     "reassociated, which mathMode Safe forbids");
        return false;
    }
    return true;
}

}  // namespace metal_backend
