// File: metal_probe.mm
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Host-callable C ABI runtime probe backing the Metal arm of
// `check_gpu` (crates/ppf-cts-core/src/utils.rs). check_gpu is a
// pre-flight capability query the server runs before ever loading the
// real backend, so it
// deliberately does NOT depend on, or duplicate, the real Metal
// backend's own context setup in
// crates/ppf-cts-compute/metal/metal_context.mm; this file asks
// only the two device-level questions the Metal arm needs and stops.
//
// Compiled only on macOS by ppf-cts-core/build.rs (via the `cc` crate,
// as Objective-C++, linked against Metal.framework + Foundation.framework).
// Mirrors the existing intersect_ffi.cpp pattern in this same directory:
// a tiny extern "C" shim, no C++ standard library dependency beyond
// <cstring>.

#import <Metal/Metal.h>
#import <IOKit/IOKitLib.h>
#import <dispatch/dispatch.h>
#include <cstring>

extern "C" {

// Queries the default Metal device (the one the real Metal backend
// would itself select via MTLCreateSystemDefaultDevice) and reports:
//
//   * `*out_device_found`: whether any Metal device is visible at all.
//     False means Metal is unusable on this host right now (no GPU
//     with Metal access), regardless of the reason.
//   * `*out_family_ok`: whether the device supports MTLGPUFamilyApple7
//     (the M1 / A14 generation) or newer. Only meaningful when
//     `*out_device_found` is true; left false otherwise. Apple7 is the
//     minimum gated on because the backend targets Apple Silicon Macs.
//     A device with no Apple GPU family at all (an Intel Mac's discrete
//     or integrated AMD/Intel GPU) fails this gate outright.
//   * `out_name` / `out_name_cap`: the device's localized name, copied
//     truncated and NUL-terminated into the caller's buffer, for
//     actionable error messages. Left as an empty string when no
//     device was found.
//
// Never throws: Objective-C exceptions are not expected from this
// narrow a set of API calls, and there is nothing here for Rust to
// catch across the FFI boundary regardless.
void ppf_cts_core_metal_probe(bool *out_device_found, bool *out_family_ok,
                               char *out_name,
                               unsigned long out_name_cap) {
    *out_device_found = false;
    *out_family_ok = false;
    if (out_name != nullptr && out_name_cap > 0) {
        out_name[0] = '\0';
    }
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return;
        }
        *out_device_found = true;
        *out_family_ok = [device supportsFamily:MTLGPUFamilyApple7] ? true : false;
        if (out_name != nullptr && out_name_cap > 0) {
            const char *cname = [[device name] UTF8String];
            if (cname != nullptr) {
                std::strncpy(out_name, cname, out_name_cap - 1);
                out_name[out_name_cap - 1] = '\0';
            }
        }
    }
}

// Reports the default Metal device's display name and the amount of memory it
// can work with, for the add-on's "Remote Hardware" block. Returns whether a
// device was found at all; on false, neither output is meaningful.
//
// SEPARATE FROM THE PROBE ABOVE ON PURPOSE. That one is a guarantee-class
// preflight `check_gpu` runs before any backend loads, and it answers exactly
// two questions. This one is descriptive: it feeds a panel, nothing gates on
// it, and a failure here must never be able to change what that preflight
// decides.
//
// `recommendedMaxWorkingSetSize` IS THE HONEST NUMBER ON APPLE SILICON, where
// there is no separate VRAM: memory is unified with the CPU, so a "total VRAM"
// figure does not exist to report. This is the working set Metal recommends
// staying under, which is what a reader wanting to know how large a scene can
// be actually needs.
bool ppf_cts_core_metal_device_info(char *out_name, unsigned long out_name_cap,
                                    unsigned long long *out_working_set_bytes,
                                    unsigned int *out_apple_family) {
    if (out_name != nullptr && out_name_cap > 0) {
        out_name[0] = '\0';
    }
    if (out_working_set_bytes != nullptr) {
        *out_working_set_bytes = 0;
    }
    if (out_apple_family != nullptr) {
        *out_apple_family = 0;
    }
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return false;
        }
        if (out_name != nullptr && out_name_cap > 0) {
            const char *cname = [[device name] UTF8String];
            if (cname != nullptr) {
                std::strncpy(out_name, cname, out_name_cap - 1);
                out_name[out_name_cap - 1] = '\0';
            }
        }
        if (out_working_set_bytes != nullptr) {
            *out_working_set_bytes = [device recommendedMaxWorkingSetSize];
        }
        if (out_apple_family != nullptr) {
            // THE HIGHEST APPLE FAMILY THIS DEVICE SUPPORTS, which is the
            // Metal analogue of a CUDA SM level: families are cumulative, so a
            // device supporting AppleN supports every family below it and the
            // highest one is the whole answer.
            //
            // WALKED UPWARD FROM A NAMED CONSTANT RATHER THAN WRITTEN AS
            // NUMBERS. `MTLGPUFamilyApple7` is used by the preflight above, so
            // it exists in any SDK that compiles this file, and offsetting from
            // its raw value reaches families whose own constants a older SDK
            // would not declare. Naming `MTLGPUFamilyApple9` directly would put
            // an SDK floor on the build for a row that only feeds a panel, and
            // hard-coding 1007 would put the same assumption somewhere a reader
            // cannot check.
            //
            // The scan stops at the first unsupported family instead of running
            // to the end, so a value Metal does not recognize is never asked
            // about beyond the one step past what the device answers yes to.
            const NSInteger base = (NSInteger)MTLGPUFamilyApple7;
            unsigned int highest = 0;
            if ([device supportsFamily:MTLGPUFamilyApple7]) {
                highest = 7;
                // Bounded rather than open: eight steps past Apple7 is far more
                // headroom than Apple has used, and a loop with no bound would
                // depend on `supportsFamily:` answering NO for every value it
                // does not know.
                for (NSInteger step = 1; step <= 8; ++step) {
                    if (![device supportsFamily:(MTLGPUFamily)(base + step)]) {
                        break;
                    }
                    highest = (unsigned int)(7 + step);
                }
            }
            *out_apple_family = highest;
        }
        return true;
    }
}

// Live GPU utilization and in-use memory for the add-on's "Realtime
// Statistics" rows, the Metal counterpart of what nvidia-smi answers on CUDA.
// Returns whether a counter was read at all.
//
// THIS IS IOKIT, NOT METAL. Metal exposes no device-wide utilization counter:
// its counter sample buffers time individual encoders, which is a different
// question. The number here is the one Activity Monitor shows, read from the
// same place, and it is SYSTEM-WIDE rather than per process, which is also what
// nvidia-smi reports on the CUDA side.
//
// NO SUBPROCESS AND NO ROOT. `powermetrics` answers this too and refuses to run
// as anyone but the superuser, which rules it out for a server an artist
// starts. The registry read needs no privilege, and being in-process it costs
// no spawn on a path that runs on every status poll.
//
// THE FIRST READ IN A PROCESS IS GARBAGE AND MUST BE DISCARDED. Measured on an
// M2: the first `IORegistryEntryCreateCFProperties` read of this key returns 99
// whatever the GPU is doing, on an idle machine and a loaded one alike. So the
// handle is opened once and primed with a throwaway read. Without that priming
// a server reports 99% GPU on its first status poll, which is a plausible
// number at exactly the moment someone is watching for one.
//
// IT IS AN INTERVAL MEASURE, so priming removes the 99 but does not make the
// next read meaningful: a read taken microseconds after the priming one covers
// microseconds of GPU time and reads 0. The caller's FIRST poll therefore
// reports 0% and every later one is accurate, which is exactly what the
// sysinfo CPU counter on the Rust side already does and says about itself.
// Measured at the 250 ms poll interval under a sustained load: poll 0 reads
// 0%, polls 1 onward read 99%; and from a long-lived handle, 0 while idle, 95
// one sample after a load starts, 99 sustained, 0 again one sample after it
// stops.
//
// THE KEYS ARE REGISTRY PROPERTIES, NOT PUBLISHED API, so a macOS that renames
// or drops one must present as "no row" rather than as a failure: every miss
// below leaves the output at zero and the caller omits the row.
static io_object_t g_accelerator = IO_OBJECT_NULL;
static dispatch_once_t g_accelerator_once;

// Read one unsigned value out of the accelerator's PerformanceStatistics.
// Returns false when the entry, the dictionary or the key is not there.
static bool read_perf_stat(io_object_t service, CFStringRef key, long long *out) {
    if (service == IO_OBJECT_NULL) {
        return false;
    }
    CFMutableDictionaryRef props = nullptr;
    if (IORegistryEntryCreateCFProperties(service, &props, kCFAllocatorDefault, 0) !=
            KERN_SUCCESS ||
        props == nullptr) {
        return false;
    }
    bool ok = false;
    CFTypeRef raw = CFDictionaryGetValue(props, CFSTR("PerformanceStatistics"));
    if (raw != nullptr && CFGetTypeID(raw) == CFDictionaryGetTypeID()) {
        CFTypeRef value = CFDictionaryGetValue((CFDictionaryRef)raw, key);
        if (value != nullptr && CFGetTypeID(value) == CFNumberGetTypeID()) {
            ok = CFNumberGetValue((CFNumberRef)value, kCFNumberLongLongType, out);
        }
    }
    // ARC does not manage CoreFoundation objects, so this is released by hand.
    CFRelease(props);
    return ok;
}

// Find the accelerator once and prime it. Held for the life of the process:
// re-finding it per poll would repeat the enumeration AND, if the garbage first
// read is a property of the handle rather than of the process, would return the
// garbage every time.
static void open_accelerator(void) {
    io_iterator_t iter = IO_OBJECT_NULL;
    if (IOServiceGetMatchingServices(0, IOServiceMatching("IOAccelerator"),
                                     &iter) != KERN_SUCCESS) {
        return;
    }
    io_object_t service = IO_OBJECT_NULL;
    while ((service = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
        long long ignored = 0;
        if (read_perf_stat(service, CFSTR("Device Utilization %"), &ignored)) {
            // Keep this one: the reference is deliberately not released.
            g_accelerator = service;
            break;
        }
        IOObjectRelease(service);
    }
    IOObjectRelease(iter);
    if (g_accelerator != IO_OBJECT_NULL) {
        // THE PRIMING READ, DISCARDED. See the note above: this is the 99.
        long long discarded = 0;
        (void)read_perf_stat(g_accelerator, CFSTR("Device Utilization %"), &discarded);
    }
}

bool ppf_cts_core_metal_usage(unsigned int *out_device_util_pct,
                              unsigned long long *out_in_use_bytes) {
    if (out_device_util_pct != nullptr) {
        *out_device_util_pct = 0;
    }
    if (out_in_use_bytes != nullptr) {
        *out_in_use_bytes = 0;
    }
    dispatch_once(&g_accelerator_once, ^{
        open_accelerator();
    });
    if (g_accelerator == IO_OBJECT_NULL) {
        return false;
    }
    long long pct = 0;
    if (!read_perf_stat(g_accelerator, CFSTR("Device Utilization %"), &pct)) {
        return false;
    }
    if (pct < 0) {
        pct = 0;
    }
    if (pct > 100) {
        pct = 100;
    }
    if (out_device_util_pct != nullptr) {
        *out_device_util_pct = (unsigned int)pct;
    }
    long long used = 0;
    if (read_perf_stat(g_accelerator, CFSTR("In use system memory"), &used) && used > 0) {
        if (out_in_use_bytes != nullptr) {
            *out_in_use_bytes = (unsigned long long)used;
        }
    }
    return true;
}

} // extern "C"
