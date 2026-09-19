// File: pipeline_cache.mm
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Objective-C++ half of pipeline_cache.hpp. The header carries the rationale
// and the three load-bearing properties; this file carries the mechanics.

#include "pipeline_cache.hpp"

#import <Foundation/Foundation.h>

#include <CommonCrypto/CommonDigest.h>

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !__has_feature(objc_arc)
#error "pipeline_cache.mm must be compiled with -fobjc-arc"
#endif

namespace metal_backend {

namespace {

// Bump this when anything about what an archive CONTAINS changes that the
// source text alone does not describe: the pipeline descriptor's settings, the
// key recipe itself, the file naming. Every existing file then misses, which is
// the correct outcome and costs one cold start.
constexpr char kKeyRecipeVersion[] = "ppf-cts metal pipeline archive key v1";

constexpr char kArchiveSuffix[] = ".metalar";

// The directory is capped by TOTAL BYTES, not by file count and not by age.
//
// By bytes, because the archives differ by three orders of magnitude: measured
// on an M1, the solver library's 123 pipelines serialize to 20,088,736 B while
// the math-mode sentinel's single pipeline takes 14,704 B. A file-count cap
// would bound the directory anywhere between a few hundred kilobytes and half
// a gigabyte depending on which files happened to be newest.
//
// Not by age, because an archive is written once per distinct shader source: a
// session that edits kernels accumulates one file per edit, and any age-based
// rule lets a busy day fill the user's home before the first file expires.
//
// 256 MiB holds about a dozen generations of the solver library, so
// alternating between two branches still hits on both.
constexpr unsigned long long kMaxArchiveBytes = 256ull * 1024ull * 1024ull;

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch())
        .count();
}

std::string ns_error_text(NSError *error) {
    if (!error) {
        return std::string("(no NSError)");
    }
    NSString *text = [error localizedDescription];
    return std::string(text ? [text UTF8String] : "(no description)");
}

// Every non-fatal message goes through the caller's sink. Nothing in this file
// writes to stderr: stderr is the crash channel (see the header), and a run
// reported as crashed because its cache went stale would be a worse defect than
// the slow start it is describing. A null sink drops the message, which is why
// the sink is a required parameter rather than an option.
void emit(void (*log)(const char *), const std::string &message) {
    if (log) {
        log(message.c_str());
    }
}

bool env_flag_disables(const char *name) {
    const char *value = std::getenv(name);
    return value && std::strcmp(value, "0") == 0;
}

// Resolved once. An empty string means "no cache", and every entry point in
// this file treats it that way, so the disable check lives in exactly one
// place.
const std::string &resolve_directory() {
    static const std::string dir = [] {
        if (env_flag_disables("PPF_METAL_ARCHIVE")) {
            return std::string();
        }
        const char *override_dir = std::getenv("PPF_METAL_ARCHIVE_DIR");
        if (override_dir && override_dir[0] != '\0') {
            return std::string(override_dir);
        }
        const char *home = std::getenv("HOME");
        if (!home || home[0] == '\0') {
            // No home to put a cache in. Running without one is correct and
            // costs only start-up time, so this is not an error.
            return std::string();
        }
        return std::string(home) + "/.cache/ppf-cts/metal-pipeline-archive";
    }();
    return dir;
}

// Every component is length-prefixed, so no two different tuples can
// concatenate to the same bytes. Without that, a device named "Apple M1" with
// one source and a device named "Apple M" with a source starting in "1" would
// hash alike.
void hash_component(CC_SHA256_CTX *ctx, const char *label, const void *data,
                    size_t length) {
    const unsigned long long label_len = std::strlen(label);
    const unsigned long long data_len = length;
    CC_SHA256_Update(ctx, &label_len, sizeof(label_len));
    CC_SHA256_Update(ctx, label, static_cast<CC_LONG>(label_len));
    CC_SHA256_Update(ctx, &data_len, sizeof(data_len));
    // CC_LONG is 32 bits, so a source larger than 4 GiB has to be fed in
    // chunks. The backend's shader is about a megabyte; the loop is here so the
    // bound is a property of the code rather than of today's shader.
    const unsigned char *bytes = static_cast<const unsigned char *>(data);
    size_t remaining = length;
    while (remaining > 0) {
        const size_t chunk = std::min<size_t>(remaining, 1u << 30);
        CC_SHA256_Update(ctx, bytes, static_cast<CC_LONG>(chunk));
        bytes += chunk;
        remaining -= chunk;
    }
}

std::string compute_key(id<MTLDevice> device, const char *source,
                        long long math_mode, const char *library_origin) {
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);

    hash_component(&ctx, "recipe", kKeyRecipeVersion,
                   std::strlen(kKeyRecipeVersion));

    NSString *device_name = [device name];
    const char *device_utf8 = device_name ? [device_name UTF8String] : "";
    hash_component(&ctx, "device", device_utf8, std::strlen(device_utf8));

    // The registry id separates two devices that report the same name, which
    // the device name alone cannot.
    const unsigned long long registry =
        static_cast<unsigned long long>([device registryID]);
    hash_component(&ctx, "registry", &registry, sizeof(registry));

    // Carries the BUILD number ("Version 26.5.2 (Build 25F84)"), which is what
    // moves when a driver update invalidates an archive.
    NSString *os = [[NSProcessInfo processInfo] operatingSystemVersionString];
    const char *os_utf8 = os ? [os UTF8String] : "";
    hash_component(&ctx, "os", os_utf8, std::strlen(os_utf8));

    // Property 3 in the header: the compile options the ISA was produced under.
    hash_component(&ctx, "mathmode", &math_mode, sizeof(math_mode));

    // Property 4: which route produced the MTLLibrary. Measured on an M1 with
    // one source, one device and one math mode: an archive written from
    // functions of a source-compiled library returns 129 of 129 MISSES for the
    // same functions of a library loaded from an offline .metallib, and the
    // reverse also misses, so the two routes do not produce interchangeable
    // functions. Sharing one file between them is not a wrong answer (a miss
    // compiles), but it costs every pipeline on every switch and leaves both
    // runs reporting a hitless archive, so the two get separate files.
    const char *origin = library_origin ? library_origin : "";
    hash_component(&ctx, "origin", origin, std::strlen(origin));

    hash_component(&ctx, "source", source, std::strlen(source));

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

NSString *ns_path(const std::string &path) {
    return [NSString stringWithUTF8String:path.c_str()];
}

// Deletes the oldest archives until the directory fits kMaxArchiveBytes.
//
// The newest file is ALWAYS kept, even when it alone exceeds the cap. Deleting
// what was just written would make every run a cold run while still paying to
// write, which is worse than being over budget.
//
// Best effort throughout: a failure to prune is not a reason to fail a run, and
// the next write tries again.
void prune_directory(const std::string &dir) {
    @autoreleasepool {
        NSFileManager *fm = [NSFileManager defaultManager];
        NSString *suffix = [NSString stringWithUTF8String:kArchiveSuffix];
        NSArray<NSString *> *names =
            [fm contentsOfDirectoryAtPath:ns_path(dir) error:nil];
        if (!names) {
            return;
        }
        struct Entry {
            std::string path;
            double mtime;
            unsigned long long bytes;
        };
        std::vector<Entry> entries;
        for (NSString *name in names) {
            if (![name hasSuffix:suffix]) {
                continue;
            }
            const std::string full = dir + "/" + [name UTF8String];
            NSDictionary *attrs = [fm attributesOfItemAtPath:ns_path(full)
                                                       error:nil];
            if (!attrs) {
                continue;
            }
            NSDate *modified = [attrs fileModificationDate];
            entries.push_back(Entry{
                full, modified ? [modified timeIntervalSince1970] : 0.0,
                static_cast<unsigned long long>([attrs fileSize])});
        }
        std::sort(entries.begin(), entries.end(),
                  [](const Entry &a, const Entry &b) {
                      return a.mtime > b.mtime;
                  });
        unsigned long long kept = 0;
        for (size_t i = 0; i < entries.size(); ++i) {
            kept += entries[i].bytes;
            if (i == 0 || kept <= kMaxArchiveBytes) {
                continue;
            }
            [fm removeItemAtPath:ns_path(entries[i].path) error:nil];
            kept -= entries[i].bytes;
        }
    }
}

}  // namespace

struct PipelineCache {
    id<MTLBinaryArchive> archive = nil;
    std::string path;
    // Set the first time a function is added. Nothing is serialized without
    // it, so a run that hits on every pipeline rewrites no file.
    bool dirty = false;
    // Cleared when an add is refused. A loaded archive that will not take new
    // functions still SERVES the ones it has, so the run keeps its hits and
    // only loses the chance to persist the misses.
    bool writable = true;
};

const std::string &pipeline_cache_directory() { return resolve_directory(); }

PipelineCache *pipeline_cache_open(id<MTLDevice> device, const char *source,
                                   long long math_mode,
                                   const char *library_origin,
                                   void (*log)(const char *message),
                                   double *open_ms, bool *loaded) {
    if (open_ms) {
        *open_ms = 0.0;
    }
    if (loaded) {
        *loaded = false;
    }
    const std::string &dir = resolve_directory();
    if (dir.empty() || !device || !source) {
        return nullptr;
    }
    if (![device respondsToSelector:@selector(newBinaryArchiveWithDescriptor:
                                                                       error:)]) {
        // Older OS. Not an error: pipelines are created the way they always
        // were.
        return nullptr;
    }

    const double started = now_ms();
    @autoreleasepool {
        NSFileManager *fm = [NSFileManager defaultManager];
        NSError *dir_error = nil;
        if (![fm createDirectoryAtPath:ns_path(dir)
                withIntermediateDirectories:YES
                                 attributes:nil
                                      error:&dir_error]) {
            emit(log, "metal: cannot create the pipeline cache directory '" +
                          dir + "' (" + ns_error_text(dir_error) +
                          "); running without it");
            if (open_ms) {
                *open_ms = now_ms() - started;
            }
            return nullptr;
        }

        const std::string path =
            dir + "/" + compute_key(device, source, math_mode, library_origin) +
            kArchiveSuffix;
        const bool exists = [fm fileExistsAtPath:ns_path(path)];

        MTLBinaryArchiveDescriptor *descriptor =
            [MTLBinaryArchiveDescriptor new];
        // A nil url starts an empty archive, which is what a key miss wants.
        descriptor.url = exists ? [NSURL fileURLWithPath:ns_path(path)] : nil;

        id<MTLBinaryArchive> archive = nil;
        NSError *open_error = nil;
        @try {
            archive = [device newBinaryArchiveWithDescriptor:descriptor
                                                       error:&open_error];
        } @catch (NSException *ex) {
            archive = nil;
            emit(log, "metal: opening the pipeline archive '" + path +
                          "' threw (" +
                          ([ex reason] ? [[ex reason] UTF8String] : "no reason") +
                          ")");
        }

        if (!archive && exists) {
            // The file is there and unusable: corrupt, truncated, or written
            // by a Metal that no longer accepts it. Start empty and let the
            // run rewrite it. Reported rather than silent, because a cache
            // that never loads would otherwise look exactly like a cache that
            // works.
            emit(log, "metal: the pipeline archive '" + path +
                          "' could not be opened (" +
                          ns_error_text(open_error) +
                          "); recompiling and rewriting it");
            descriptor.url = nil;
            open_error = nil;
            @try {
                archive = [device newBinaryArchiveWithDescriptor:descriptor
                                                           error:&open_error];
            } @catch (NSException *) {
                archive = nil;
            }
        }

        if (!archive) {
            emit(log, "metal: this device would not create a pipeline archive "
                      "(" + ns_error_text(open_error) +
                          "); running without the cache");
            if (open_ms) {
                *open_ms = now_ms() - started;
            }
            return nullptr;
        }

        PipelineCache *pc = new PipelineCache();
        pc->archive = archive;
        pc->path = path;
        if (open_ms) {
            *open_ms = now_ms() - started;
        }
        if (loaded) {
            *loaded = exists && descriptor.url != nil;
        }
        return pc;
    }
}

void pipeline_cache_close(PipelineCache *pc) {
    if (!pc) {
        return;
    }
    @autoreleasepool {
        pc->archive = nil;
        delete pc;
    }
}

id<MTLComputePipelineState> pipeline_cache_make(PipelineCache *pc,
                                                id<MTLDevice> device,
                                                id<MTLFunction> fn,
                                                const char *fn_name,
                                                void (*log)(const char *message),
                                                PipelineCacheOutcome *outcome,
                                                std::string *err) {
    if (outcome) {
        *outcome = PipelineCacheOutcome::Unavailable;
    }
    @autoreleasepool {
        MTLComputePipelineDescriptor *descriptor =
            [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = fn;

        // One lookup against the archive, asked twice below.
        // FailOnBinaryArchiveMiss is what makes a miss REPORT itself: without
        // it the call silently compiles, and a cache that never hits would be
        // indistinguishable from one that always does.
        auto lookup = [&]() -> id<MTLComputePipelineState> {
            NSError *lookup_error = nil;
            return [device
                newComputePipelineStateWithDescriptor:descriptor
                                              options:
                                                  MTLPipelineOptionFailOnBinaryArchiveMiss
                                           reflection:NULL
                                                error:&lookup_error];
        };

        if (pc) {
            descriptor.binaryArchives = @[ pc->archive ];
            if (id<MTLComputePipelineState> hit = lookup()) {
                if (outcome) {
                    *outcome = PipelineCacheOutcome::Hit;
                }
                return hit;
            }

            // A miss. Compile it INTO the archive and take the pipeline from
            // there, so the cold run compiles once rather than twice.
            if (pc->writable) {
                NSError *add_error = nil;
                const BOOL added = [pc->archive
                    addComputePipelineFunctionsWithDescriptor:descriptor
                                                        error:&add_error];
                if (added) {
                    pc->dirty = true;
                    if (id<MTLComputePipelineState> made = lookup()) {
                        if (outcome) {
                            *outcome = PipelineCacheOutcome::Miss;
                        }
                        return made;
                    }
                } else {
                    // One report, then stop trying: an archive that refuses one
                    // add refuses the rest, and 123 identical lines would bury
                    // the reason.
                    pc->writable = false;
                    emit(log, "metal: the pipeline archive '" + pc->path +
                                  "' would not accept '" +
                                  (fn_name ? fn_name : "?") + "' (" +
                                  ns_error_text(add_error) +
                                  "); this run compiles its pipelines and "
                                  "writes nothing");
                }
            }
            if (outcome) {
                *outcome = PipelineCacheOutcome::Miss;
            }
        }

        // No archive, or the archive could not serve or accept this function.
        // This is the path the backend took before the cache existed.
        descriptor.binaryArchives = nil;
        NSError *plain_error = nil;
        id<MTLComputePipelineState> plain =
            [device newComputePipelineStateWithDescriptor:descriptor
                                                  options:MTLPipelineOptionNone
                                               reflection:NULL
                                                    error:&plain_error];
        if (!plain && err) {
            *err = std::string("pipeline creation for '") +
                   (fn_name ? fn_name : "?") +
                   "' failed: " + ns_error_text(plain_error);
        }
        return plain;
    }
}

bool pipeline_cache_flush(PipelineCache *pc, double *ms, std::string *err) {
    if (ms) {
        *ms = 0.0;
    }
    if (!pc || !pc->dirty || !pc->writable) {
        return false;
    }
    const double started = now_ms();
    bool wrote = false;
    @autoreleasepool {
        // A sibling temporary, then a rename. serializeToURL refuses to
        // overwrite, and a reader must never see a half-written archive: the
        // rename is atomic within the directory, so the file under the real
        // name is either the old complete one or the new complete one.
        const std::string temp =
            pc->path + ".tmp." + std::to_string(static_cast<long long>(getpid()));
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm removeItemAtPath:ns_path(temp) error:nil];

        NSError *serialize_error = nil;
        BOOL ok = NO;
        @try {
            ok = [pc->archive serializeToURL:[NSURL fileURLWithPath:ns_path(temp)]
                                       error:&serialize_error];
        } @catch (NSException *ex) {
            ok = NO;
            if (err) {
                *err = std::string("serializing the pipeline archive threw: ") +
                       ([ex reason] ? [[ex reason] UTF8String] : "no reason");
            }
        }
        if (!ok) {
            if (err && err->empty()) {
                *err = "serializing the pipeline archive to '" + temp +
                       "' failed: " + ns_error_text(serialize_error);
            }
            [fm removeItemAtPath:ns_path(temp) error:nil];
        } else if (std::rename(temp.c_str(), pc->path.c_str()) != 0) {
            if (err) {
                *err = "renaming the pipeline archive '" + temp + "' onto '" +
                       pc->path + "' failed: " + std::strerror(errno);
            }
            [fm removeItemAtPath:ns_path(temp) error:nil];
        } else {
            wrote = true;
            pc->dirty = false;
        }
    }
    if (wrote) {
        prune_directory(resolve_directory());
    }
    if (ms) {
        *ms = now_ms() - started;
    }
    return wrote;
}

}  // namespace metal_backend
