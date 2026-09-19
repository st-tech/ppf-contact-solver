// File: entries_probe.mm
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Proves that a PREBUILT library of every generated entry point yields a
// compute pipeline for every one of them.
//
// WHY THIS IS WORTH A PROGRAM. The shipped Metal backend assembles its shader
// at RUN time: `newLibraryWithSource` is handed one string with no filesystem
// behind it, so the segments have to be spliced in a hand-kept order, quoted
// includes have to be neutralized as they go, and a segment placed before the
// body it calls fails with `use of undeclared identifier`. That order is real
// work and it is checked by nothing until a scene starts.
//
// A backend that dispatches by kernel id does not need any of it. `xcrun metal`
// is clang and HAS a filesystem, so each generated entry compiles offline as its
// own translation unit with live includes, and `xcrun metallib` links the AIR
// objects into one library. This program is what establishes that the library is
// then USABLE: that Metal loads it and hands back a pipeline for every entry.
//
// LOADING IS NOT THE QUESTION, AND THAT IS THE POINT OF CREATING EVERY
// PIPELINE. A library that parses proves only that the container is well formed.
// Pipeline creation is the AIR-to-ISA step, the larger of Metal's two startup
// costs and the one that can still reject a function. So this creates one for
// every name the library exports and fails on the first that does not.
//
// It reports the COUNT first and treats zero as a failure, for the reason the
// FP64 guard states on the CUDA side: an extraction that found nothing exits 0
// and reads as a clean result.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, const char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: entries_probe <path to .metallib>\n");
        return 2;
    }
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "entries_probe: no Metal device\n");
            return 1;
        }
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        if (!library) {
            fprintf(stderr, "entries_probe: newLibraryWithURL failed: %s\n",
                    error ? error.localizedDescription.UTF8String : "(no error)");
            return 1;
        }
        NSArray<NSString *> *names = [library functionNames];
        printf("  library exports %lu function(s)\n",
               (unsigned long)names.count);
        if (names.count == 0) {
            fprintf(stderr,
                    "entries_probe: the library exports nothing, so this "
                    "would report a pass over an empty set\n");
            return 1;
        }
        unsigned made = 0;
        std::vector<std::string> failures;
        for (NSString *name in names) {
            id<MTLFunction> function = [library newFunctionWithName:name];
            if (!function) {
                failures.push_back(std::string(name.UTF8String) +
                                   ": newFunctionWithName returned nil");
                continue;
            }
            NSError *pipeline_error = nil;
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function
                                                      error:&pipeline_error];
            if (pipeline) {
                ++made;
            } else {
                failures.push_back(
                    std::string(name.UTF8String) + ": " +
                    (pipeline_error
                         ? pipeline_error.localizedDescription.UTF8String
                         : "(no error)"));
            }
        }
        printf("  compute pipelines created: %u of %lu\n", made,
               (unsigned long)names.count);
        for (const std::string &f : failures) {
            fprintf(stderr, "  FAILED %s\n", f.c_str());
        }
        return failures.empty() ? 0 : 1;
    }
}
