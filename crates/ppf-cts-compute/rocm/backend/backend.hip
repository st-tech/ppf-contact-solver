// File: backend.cu
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// THE CUDA TARGET, AS THE C ABI THE NEUTRAL DRIVER CALLS.
//
// `kernels/seam/backend_abi.h` declares one boundary and this file answers
// it with nvcc. What crosses is allocation, free, transfer and launch, plus
// the platform machinery those four need, and nothing else: no convergence
// test, no phase ordering, no fallback, no parameter interpretation, no branch
// on a physical quantity. An argument record crosses as OPAQUE BYTES whose
// length is the only thing checked here, and the launcher that binds them is
// GENERATED from the same declaration the driver's record is generated from.
// So this file names no kernel, declares no argument record and reads no field
// of one.
//
// WHAT MAKES THAT POSSIBLE IS THE GENERATED TABLE, and it is worth stating
// where the table's authority comes from rather than leaving it to be inferred.
// `kernelgen.py --emit table --target cu` renders one fragment per neutral
// kernel source; the build concatenates them in sorted source order into
// `kernel_table.inc`; this file includes that concatenation twice, once for the
// declarations and once for the rows. A kernel id is a row's POSITION. The
// driver builds its half the same way from the `rust` fragments, and
// `be_open` compares the two id by id, by name and by record size, so two
// trees that disagree are a named refusal at open rather than a dispatch of the
// wrong kernel with the right bytes.
//
// THE ONE PLACE THIS LIBRARY IS NOT YET THE ABI IT IMPLEMENTS, stated here
// rather than discovered: the arena and the diagnostic channel reach CUDA
// through `HIP_HANDLE_ERROR`, which calls `exit(1)`. A device out-of-memory
// inside `compute::arena::alloc` therefore ends the process instead of
// returning STATUS_PLATFORM. That is loud rather than silent, so it is a
// divergence and not a hazard, and it is the production allocator's own
// behavior, shared with the path that runs every scene today. Everything this
// file adds returns a status.

#include "seam/backend_abi.h"

#include "arena/arena.hpp"
#include "hip_utils.hpp"
#include "diagnostics/diagnostics.hpp"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <map>
#include <string>
#include <vector>

// THE ONE INTERNAL USE OF THIS ABI. Closing a backend with an encoder still
// open discards that encoder, and discarding one is exactly what
// `be_encode_abandon` is, so it is called rather than reimplemented: a
// second copy of "end the capture, drop the list, clear the slot" is a place
// for the two to disagree about whether a capture was ended.
extern "C" void be_encode_abandon(BeEncoder *enc);

// ===========================================================================
// THE GLOBALS A DRIVER OWNS
//
// `fatal.hpp` and `cuda_utils.hpp` declare these `extern` and leave the storage
// to whoever DRIVES, which is what lets `fatal` and the device-allocation
// tallies be header-only. This library drives: it links no orchestrator, so it
// owes them exactly as a regression binary does, and without them it loads
// nowhere ("undefined symbol: g_ppf_fatal_code" at the first dlopen).
//
// Nothing here stands in for behavior. `fatal` still exits, and the
// watchdog flag starts at -1 rather than 0 because this library runs no
// preflight that reads it: negative is "not measured", and claiming "not
// armed" would be reporting a state nothing here looked at.
// ===========================================================================

extern "C" unsigned char g_ppf_fatal_code = 0;
extern "C" char g_ppf_fatal_detail[512] = {0};
unsigned long long g_device_alloc_count = 0;
unsigned long long g_device_free_count = 0;
int g_ppf_kernel_timeout_enabled = -1;

// The arena's binding budget and the ABI's are the same number by construction
// rather than by coincidence: a handle's arena id means one thing across every
// target, so a library whose allocator could hand out a thirtieth arena would
// be handing out an id no other target can resolve.
static_assert(compute::arena::kMaxArenas == PPF_BE_MAX_ARENAS,
              "the HIP arena's binding budget and the ABI's disagree");

// ===========================================================================
// THE KERNEL TABLE
//
// First inclusion: the argument records and the launcher declarations the rows
// below name. Second inclusion, inside the array, the rows themselves. One
// generated concatenation serves both, which is what stops a build from
// ordering the two halves differently.
// ===========================================================================

#define PPF_BE_TABLE_ARGS_INCLUDES
#include "kernel_table.inc"
#undef PPF_BE_TABLE_ARGS_INCLUDES

namespace {

// A launcher's two shapes. They are different function TYPES because the group
// shape takes the width: in the element shape the group width computes no
// value, so the launcher picks it, and in the group shape it is how many
// threads cooperate on one problem, which is a statement about the kernel that
// only the driver can make.
using ElementLaunch = void (*)(const void *, hipStream_t);
using GroupLaunch = void (*)(const void *, unsigned, hipStream_t);

struct KernelRow {
    const char *name;
    unsigned args_bytes;
    // Exactly one of these is non-null, and which one is the launch shape.
    ElementLaunch element;
    GroupLaunch group;
    // Static group-local scratch the generated entry declares for itself. A
    // dispatch must not ask for any, so this is what a request is checked
    // against rather than something the caller supplies.
    unsigned scratch_bytes;
    // WHERE THIS RECORD'S HANDLES SIT, so each one's arena can be checked
    // against the live binding table before the launch. Generated beside the
    // record, so a field added to one moves the other.
    const unsigned *handle_offsets;
    unsigned handle_count;
};

#define PPF_BE_KERNEL(entry, Args, launch, offsets, handles)                   \
    KernelRow{#entry, static_cast<unsigned>(sizeof(Args)), &launch, nullptr,   \
              0, offsets, handles},
#define PPF_BE_KERNEL_GROUP(entry, Args, launch, scratch, offsets, handles)    \
    KernelRow{#entry, static_cast<unsigned>(sizeof(Args)), nullptr, &launch,   \
              static_cast<unsigned>(scratch), offsets, handles},

const KernelRow kKernelTable[] = {
#include "kernel_table.inc"
};

#undef PPF_BE_KERNEL
#undef PPF_BE_KERNEL_GROUP

constexpr unsigned kKernelCount =
    static_cast<unsigned>(sizeof(kKernelTable) / sizeof(kKernelTable[0]));

// A library carrying no kernel would pass every cross-check at open by
// vacuity: the driver's table would have to be empty too, and an empty table
// dispatches nothing while reporting a clean open. The concatenation is
// produced by a build rule over the whole kernel tree, so an empty one means
// the rule ran over nothing, which is a build defect rather than a
// configuration.
static_assert(kKernelCount > 0,
              "the generated kernel table is empty: kernel_table.inc was "
              "concatenated from no fragments");

// ===========================================================================
// ERROR REPORTING
// ===========================================================================

BeStatus set_error(BeError *err, BeStatus status, long long platform_code,
                    const std::string &detail) {
    if (err) {
        err->status = static_cast<int32_t>(status);
        err->platform_code = platform_code;
        const size_t room = sizeof(err->detail) - 1;
        err->truncated = detail.size() > room ? 1 : 0;
        const size_t copied = detail.size() < room ? detail.size() : room;
        std::memcpy(err->detail, detail.data(), copied);
        err->detail[copied] = '\0';
    }
    return status;
}

BeStatus misuse(BeError *err, const std::string &detail) {
    return set_error(err, STATUS_MISUSE, 0, detail);
}

BeStatus platform(BeError *err, hipError_t code, const std::string &what) {
    return set_error(err, STATUS_PLATFORM, static_cast<long long>(code),
                     what + ": " + hipGetErrorString(code));
}

// One encoded item. The bytes are copied at the dispatch call, so a caller may
// reuse its buffer immediately and a recorded region owns its own copy.
struct Item {
    // A fill is spelled by a null kernel row, so one ordered list carries both
    // and there is nowhere for a fill to be reordered against a dispatch.
    bool is_fill;
    unsigned kernel_id;
    BeExtent extent;
    std::vector<unsigned char> args;
    // Fill only.
    BeHandle dst;
    unsigned long long byte_offset;
    unsigned long long bytes;
    unsigned char value;
};

} // namespace

struct BeRegion {
    bool deferred;
    // THE NAME THE REGION WAS RECORDED UNDER, kept so a REPLAY can be charged
    // to the same row an immediate submit would be. Without it `PPF_REGION_STATS`
    // reported only what still submits immediately, which stopped being a
    // reasonable summary the moment a production pass began recording.
    std::string label;
    hipGraphExec_t exec;
    // The re-issue form, kept when capture was refused or failed.
    std::vector<Item> items;
    unsigned dispatch_count;
    unsigned fill_count;
    unsigned long long allocator_generation;
};

struct BeEncoder {
    BeBackend *backend;
    const char *region;
    BeEncodeMode mode;
    std::vector<Item> items;
    unsigned dispatch_count;
    unsigned fill_count;
    // Set when capture was begun at `be_encode_begin`, so the close knows
    // whether it has a capture to end.
    bool capturing;
};

struct BeBackend {
    compute::arena::Allocator *allocator;
    // A NON-DEFAULT STREAM, and that is required rather than tidy: CUDA
    // refuses stream capture on the legacy default stream, so a library that
    // launched on stream 0 could never record a region and would report the
    // fallback for a reason that is its own doing.
    hipStream_t queue;
    diagnostics::Channel diag;
    BeCounters counters;
    BeLogFn log;
    void *log_context;
    int poison_byte;
    bool require_deferred;
    bool device_lost;
    unsigned long long generation;
    BeEncoder *open_encoder;
    // (arena, offset) to the caller's label. The caller guarantees a label
    // outlives the backend, so nothing is copied and nothing is owned.
    std::map<unsigned long long, const char *> labels;
    // PER-LABEL TRANSFER TALLY, which is what a parity run against another
    // backend reads. `nsys` gives totals and
    // per-kernel names; it cannot say WHICH BUFFER moved 63 GB, because a
    // transfer carries no name of its own. The label the allocation was made
    // under is that name, and this keys on it.
    //
    // It is off unless PPF_TRANSFER_STATS is set, and it counts rather than
    // times: a count is a property of the algorithm and travels between
    // machines, where a time is a property of the box.
    struct TransferTally {
        unsigned long long h2d_calls = 0, h2d_bytes = 0;
        unsigned long long d2h_calls = 0, d2h_bytes = 0;
    };
    std::map<std::string, TransferTally> transfers;
    bool transfer_stats = false;
    // PER-REGION SUBMIT TALLY, the same idea one level up. A submit ENDS IN A
    // SYNCHRONIZE, measured at 75.3 percent of this tree's CUDA API time, and
    // nsys can say how many
    // there were but not WHICH PHASE issued them: a synchronize carries no name
    // either. The region label a boundary was opened under is that name.
    //
    // Off unless PPF_REGION_STATS is set, and a COUNT rather than a time for
    // the reason the tally above is: a count is a property of the driver and
    // travels between machines.
    struct RegionTally {
        unsigned long long submits = 0;
        double seconds = 0.0;
        // SPLIT AT THE SUBMIT. `encode` is host time spent building the
        // boundary, argument marshalling included; `wait` is what
        // `hipStreamSynchronize` blocks for, which is the device actually
        // working. A region that is mostly encode is a host cost the GPU never
        // sees.
        double encode = 0.0;
        double wait = 0.0;
    };
    std::map<std::string, RegionTally> regions;
    bool region_stats = false;
    // The region currently open and when it opened, so the submit can charge
    // its wall time to the right row. One encoder is open at a time, which
    // `be_encode_begin` already enforces.
    const char *open_region = nullptr;
    std::chrono::steady_clock::time_point open_at{};
};

namespace {

unsigned long long block_key(const BeHandle &handle) {
    return (static_cast<unsigned long long>(handle.arena) << 32) | handle.off;
}

void emit_log(BeBackend *be, int32_t level, const std::string &message) {
    if (be && be->log) {
        be->log(be->log_context, level, message.c_str());
    }
}

// The one instance this library supports. A second `be_open` returns
// MISUSE rather than a second handle onto shared state, because the arena and
// the device symbol table it writes are process-wide.
BeBackend *g_backend = nullptr;

// Every call except close, counters and the diagnostic drain answers
// DEVICE_LOST once a fault has been reported, so a driver that ignores the
// first fault cannot proceed on stale device data.
bool lost(BeBackend *be, BeError *err) {
    if (be && be->device_lost) {
        set_error(err, STATUS_DEVICE_LOST, 0,
                  "the HIP context was lost by an earlier device fault and "
                  "has not been reopened");
        return true;
    }
    return false;
}

// A sticky CUDA error poisons the context, so it is latched here rather than
// reported and forgotten.
BeStatus fault(BeBackend *be, BeError *err, hipError_t code,
                const std::string &what) {
    be->device_lost = true;
    return set_error(err, STATUS_DEVICE_FAULT,
                     static_cast<long long>(code),
                     what + ": " + hipGetErrorString(code));
}

// Defined with the transfers it also serves, below. Declared here because a
// fill is ordered against the dispatches around it and so is executed from
// this list, and the bound it is checked against must be the same one a
// transfer is checked against rather than a second spelling of it.
BeStatus window(BeBackend *be, BeHandle handle, size_t byte_offset,
                 size_t bytes, unsigned char **base, BeError *err);

BeStatus run_items(BeBackend *be, const std::vector<Item> &items,
                    BeError *err) {
    for (const Item &item : items) {
        if (item.is_fill) {
            unsigned char *base = nullptr;
            const BeStatus status =
                window(be, item.dst, static_cast<size_t>(item.byte_offset),
                       static_cast<size_t>(item.bytes), &base, err);
            if (status != STATUS_OK) {
                return status;
            }
            if (item.bytes == 0) {
                continue;
            }
            const hipError_t code =
                hipMemsetAsync(base, item.value,
                                static_cast<size_t>(item.bytes), be->queue);
            if (code != hipSuccess) {
                return fault(be, err, code, "hipMemsetAsync");
            }
            continue;
        }
        const KernelRow &row = kKernelTable[item.kernel_id];
        // EVERY HANDLE'S ARENA, CHECKED HERE, WHICH IS THE ONLY PLACE THAT CAN.
        //
        // Asserting it inside `compute::arena::resolve`, per handle per
        // THREAD, costs 1,460 ms of 22,018 on `drape`, 6.6 percent of GPU
        // time. It is the same verdict for every thread of a dispatch, so it
        // is computed once here instead.
        //
        // AND IT HAS TO BE HERE RATHER THAN AT ENCODE TIME. An arena count is a
        // property of the DEVICE at the instant of dispatch, not of the handle:
        // a region recorded while there were N arenas can be replayed after
        // another was added, so a check at encode time would assert something
        // that can change before the dispatch it is about. The check therefore
        // belongs at replay time, and this loop is where replay reaches it.
        const unsigned live =
            compute::arena::arena_count(compute::arena::active());
        for (unsigned k = 0; k < row.handle_count; ++k) {
            const unsigned at = row.handle_offsets[k];
            // The record's own size is checked at encode; this guards the read
            // rather than trusting that check from here.
            if (at + sizeof(unsigned) > item.args.size()) {
                return misuse(err, std::string("kernel ") + row.name +
                                       " names a handle past the end of its "
                                       "argument record");
            }
            unsigned arena = 0;
            std::memcpy(&arena, item.args.data() + at, sizeof(arena));
            if (arena >= live) {
                return misuse(err, std::string("kernel ") + row.name +
                                       " names arena " + std::to_string(arena) +
                                       " and only " + std::to_string(live) +
                                       " are bound, so the record is malformed");
            }
        }
        if (row.element) {
            row.element(item.args.data(), be->queue);
        } else {
            row.group(item.args.data(), item.extent.threads, be->queue);
        }
        const hipError_t code = hipGetLastError();
        if (code != hipSuccess) {
            return fault(be, err, code,
                         std::string("launching ") + row.name);
        }
    }
    return STATUS_OK;
}

// Reads the diagnostic channel into the ABI's summary and resets it for the
// next boundary. The channel is the LIBRARY's transport; what the counts mean
// is the driver's, which is why this fills a report and returns no verdict.
void drain_diag(BeBackend *be, BeDiagSummary *out) {
    if (!out) {
        return;
    }
    std::memset(out, 0, sizeof(*out));
    diagnostics::Readback back;
    std::string detail;
    if (!diagnostics::read(be->diag, &back, &detail)) {
        return;
    }
    out->failures = back.fail_count;
    out->assert_hit = back.assert_hit ? 1 : 0;
    out->assert_record = back.assert_record;
    out->trace_written = back.ring_written;
    out->trace_dropped = back.ring_dropped;
    out->trace_count = static_cast<uint32_t>(back.ring.size());
    // THE CLEAR IS 32 BYTES OF HOST MEMORY. The channel is mapped, so this is
    // a plain store rather than a memset plus an upload: those would run on
    // every boundary, and measured that way were 8,552 memsets and about 8,280
    // of the 8,979 host-to-device copies on `drape` at 3 frames, against a
    // reference run that issues 105 memsets in total.
    //
    // THE RING BODY NEEDS NO CLEARING. A reader takes `ring_cursor` records and
    // nothing past it, so a stale record below the cursor is unreachable, which
    // is why this only ever touched the header.
    diagnostics::reset(be->diag);
}

} // namespace

// ===========================================================================
// IDENTITY, CALLABLE BEFORE OPEN
// ===========================================================================

extern "C" uint32_t be_abi_version(void) { return PPF_BE_ABI_VERSION; }

// THE IDENTITY DISTINGUISHES THE TWO PLATFORMS, AND THAT IS LOAD-BEARING HERE.
//
// One source builds for two platforms: `HIP_PLATFORM=amd` targets a Radeon GPU
// and `HIP_PLATFORM=nvidia` compiles the same HIP through nvcc and runs on an
// NVIDIA GPU. The second is how this port is execution-qualified at all, with
// no AMD hardware in the project.
//
// So a single name would let an NVIDIA-platform measurement be filed as an AMD
// result, and there would never be an AMD run to contradict it: it would simply
// stand. Naming the two platforms apart is the rule, and this function is
// where it is kept.
extern "C" const char *be_backend_name(void) {
#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__)
    return "hip-nvidia";
#else
    return "rocm";
#endif
}

extern "C" void be_layout_probe(BeLayoutProbe *out) {
    if (!out) {
        return;
    }
    std::memset(out, 0, sizeof(*out));
    out->abi_version = PPF_BE_ABI_VERSION;
    out->handle_size = static_cast<uint32_t>(sizeof(BeHandle));
    out->handle_align = static_cast<uint32_t>(alignof(BeHandle));
    out->diag_record_size = static_cast<uint32_t>(sizeof(BeDiagRecord));
    out->diag_record_align = static_cast<uint32_t>(alignof(BeDiagRecord));
    out->extent_size = static_cast<uint32_t>(sizeof(BeExtent));
    out->error_size = static_cast<uint32_t>(sizeof(BeError));
    out->diag_summary_size = static_cast<uint32_t>(sizeof(BeDiagSummary));
    out->counters_size = static_cast<uint32_t>(sizeof(BeCounters));
    out->device_info_size = static_cast<uint32_t>(sizeof(BeDeviceInfo));
    out->config_size = static_cast<uint32_t>(sizeof(BeBackendConfig));
    out->region_info_size = static_cast<uint32_t>(sizeof(BeRegionInfo));
    out->shader_cache_report_size =
        static_cast<uint32_t>(sizeof(BeShaderCacheReport));
    out->max_args_bytes = PPF_BE_MAX_ARGS_BYTES;
    out->max_arenas = PPF_BE_MAX_ARENAS;
}

// ===========================================================================
// LIFECYCLE
// ===========================================================================

// Defined below; declared here because be_open registers it with atexit.
static void dump_transfer_stats(BeBackend *be);

static void dump_region_stats(BeBackend *be);

extern "C" BeStatus be_open(const BeBackendConfig *config,
                                 BeBackend **out, BeError *err) {
    // PPF_TRANSFER_STATS names every buffer that crosses the seam and how much
    // it moved. Off by default: it costs a map lookup per transfer.

    if (!config || !out) {
        return misuse(err, "be_open needs a config and an output slot");
    }
    if (config->abi_version != PPF_BE_ABI_VERSION) {
        return misuse(err, "the caller was built against ABI version " +
                               std::to_string(config->abi_version) +
                               " and this library implements " +
                               std::to_string(PPF_BE_ABI_VERSION));
    }
    if (g_backend) {
        return misuse(err, "this library supports one live backend and one is "
                           "already open");
    }
    if (config->diag_ring_slots == 0) {
        return misuse(err, "the diagnostic ring needs at least one slot");
    }
    // The device is selected and the context created here rather than lazily,
    // so a host with no usable device fails at open, by name, with nothing to
    // fall back to.
    int devices = 0;
    hipError_t code = hipGetDeviceCount(&devices);
    if (code != hipSuccess) {
        return platform(err, code, "hipGetDeviceCount");
    }
    if (devices == 0) {
        return set_error(err, STATUS_PLATFORM, 0,
                         "no AMD device is visible to this process");
    }
    code = hipFree(nullptr);
    if (code != hipSuccess) {
        return platform(err, code, "creating the HIP context");
    }

    auto *be = new BeBackend();
    be->transfer_stats = getenv("PPF_TRANSFER_STATS") != nullptr;
    be->region_stats = getenv("PPF_REGION_STATS") != nullptr;
    if (be->transfer_stats) {
        atexit([]() { dump_transfer_stats(g_backend); });
    }
    // AT EXIT, NOT AT `be_close`, for the reason the tally above is: the solver
    // process does not always reach `be_close`, so a dump hung there prints
    // nothing and reads as "no boundaries were counted". Measured the hard way.
    if (be->region_stats) {
        atexit([]() { dump_region_stats(g_backend); });
    }
    be->log = config->log;
    be->log_context = config->log_context;
    be->poison_byte = config->poison_byte;
    be->require_deferred = config->require_deferred_regions != 0;
    be->device_lost = false;
    be->generation = 0;
    be->open_encoder = nullptr;
    std::memset(&be->counters, 0, sizeof(be->counters));

    code = hipStreamCreate(&be->queue);
    if (code != hipSuccess) {
        delete be;
        return platform(err, code, "hipStreamCreate");
    }
    be->allocator = compute::arena::create();
    if (!be->allocator) {
        hipStreamDestroy(be->queue);
        delete be;
        return set_error(err, STATUS_PLATFORM, 0,
                         "the HIP arena allocator refused to open; one is "
                         "already active in this process");
    }
    std::string detail;
    // THE BACKEND'S CHANNEL IS THE GLOBAL ONE, as on CUDA and for the same
    // reason: every generated entry is handed `diagnostics::global()` by the
    // shared generator, and the launcher signature carries no parameter for a
    // channel of its own. A backend holding any other object drains something no
    // kernel writes, so a recorded violation could never be read, and a channel
    // nothing created leaves every kernel a null header.
    if (diagnostics::global().device_base == nullptr &&
        !diagnostics::create_global(config->diag_ring_slots, &detail)) {
        compute::arena::destroy(be->allocator);
        hipStreamDestroy(be->queue);
        delete be;
        return set_error(err, STATUS_PLATFORM, 0,
                         "the diagnostic channel could not be created: " +
                             detail);
    }
    be->diag = diagnostics::global();
    diagnostics::reset(be->diag);
    // PPF_DIAG_SELFTEST, as on CUDA and for the same reason: prove on THIS
    // machine that a failed device check reaches the host, before anything
    // depends on it doing so. Off by default because a healthy channel is the
    // common case; on in the acceptance passes, where what is being asked is
    // whether this build reports what it is supposed to report. A failure
    // refuses the open rather than warning, because every guarantee-class
    // check in the solver reports through this channel.
    if (getenv("PPF_DIAG_SELFTEST") != nullptr) {
        std::string why;
        if (!diagnostics::selftest_global(&why)) {
            compute::arena::destroy(be->allocator);
            hipStreamDestroy(be->queue);
            delete be;
            return set_error(err, STATUS_PLATFORM, 0,
                             "the device diagnostic channel does not report: " +
                                 why);
        }
        // ON stderr, NOT through the log sink. The confirmation exists to be
        // READ by the acceptance passes, and `--probe` installs no sink, so a
        // line emitted at level 0 would reach nobody there and a caller could
        // only infer the self-test from the open succeeding. Inferring it is
        // what makes the check vacuous the day the switch stops being read.
        fprintf(stderr, "ppf: the device diagnostic channel reports (PPF_DIAG_SELFTEST)\n");
    }
    // The channel's own storage is an allocation, so the generation starts
    // above zero and a region recorded before any driver allocation is still
    // compared against a value that moves.
    be->generation = 1;
    g_backend = be;
    *out = be;
    emit_log(be, 0, "the ROCm (HIP) backend library is open");
    return STATUS_OK;
}

// The per-label transfer table. Printed from `be_close` and from an `atexit`
// handler, because a solver that exits without closing its backend would
// otherwise produce nothing, which is what happened the first time this was
// run. Printing twice is prevented by clearing the map.
static void dump_transfer_stats(BeBackend *be) {
    if (!be || !be->transfer_stats || be->transfers.empty()) {
        return;
    }
    fprintf(stderr, "\n=== PPF_TRANSFER_STATS: bytes across the seam, by buffer ===\n");
    fprintf(stderr, "%-44s %10s %14s %10s %14s\n", "label", "h2d calls",
            "h2d MB", "d2h calls", "d2h MB");
    for (const auto &row : be->transfers) {
        fprintf(stderr, "%-44s %10llu %14.1f %10llu %14.1f\n",
                row.first.c_str(),
                (unsigned long long)row.second.h2d_calls,
                row.second.h2d_bytes / 1048576.0,
                (unsigned long long)row.second.d2h_calls,
                row.second.d2h_bytes / 1048576.0);
    }
    be->transfers.clear();
}

// WHICH PHASE PAID FOR THE STALLS. Sorted by count rather than by name, because
// the question this answers is always "what dominates" and a reader should not
// have to sort 200 rows by eye.
static void dump_region_stats(BeBackend *be) {
    if (!be || !be->region_stats) {
        return;
    }
    if (be->regions.empty()) {
        fprintf(stderr, "\n=== PPF_REGION_STATS: no boundaries counted ===\n");
        return;
    }
    std::vector<std::pair<std::string, BeBackend::RegionTally>> rows(be->regions.begin(),
                                                                     be->regions.end());
    std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
        return a.second.seconds > b.second.seconds;
    });
    unsigned long long total = 0;
    for (const auto &row : rows) {
        total += row.second.submits;
    }
    double seconds = 0.0;
    for (const auto &row : rows) {
        seconds += row.second.seconds;
    }
    fprintf(stderr, "\n=== PPF_REGION_STATS: submits (each a synchronize) and the wall time inside them ===\n");
    double encode = 0.0, wait = 0.0;
    for (const auto &row : rows) {
        encode += row.second.encode;
        wait += row.second.wait;
    }
    fprintf(stderr, "%-40s %9s %9s %9s %9s\n", "region", "submits", "seconds",
            "encode", "wait");
    for (const auto &row : rows) {
        fprintf(stderr, "%-40s %9llu %9.2f %9.2f %9.2f\n", row.first.c_str(),
                (unsigned long long)row.second.submits, row.second.seconds,
                row.second.encode, row.second.wait);
    }
    fprintf(stderr, "%-40s %9s %9.2f %9.2f %9.2f\n", "  (encode is HOST, wait is DEVICE)",
            "", 0.0, encode, wait);
    fprintf(stderr, "%-46s %10llu %8s %10.2f\n", "TOTAL IN REGIONS",
            (unsigned long long)total, "", seconds);
    fprintf(stderr, "  Anything the process spends OUTSIDE these is host work\n"
                    "  between boundaries: subtract this from the run's wall clock.\n");
    be->regions.clear();
}

extern "C" void be_close(BeBackend *be) {
    dump_transfer_stats(be);
    if (!be) {
        return;
    }
    if (be->open_encoder) {
        be_encode_abandon(be->open_encoder);
    }
    // NOT DESTROYED HERE, for the reason the CUDA backend states: `be->diag`
    // is a copy of the process-wide global channel that every generated kernel
    // addresses by name, so freeing its pages on backend teardown would leave
    // the next kernel launched in this process writing through a dangling
    // pointer. It is one small mapped allocation, reused for the life of the
    // process.
    compute::arena::destroy(be->allocator);
    hipStreamDestroy(be->queue);
    if (g_backend == be) {
        g_backend = nullptr;
    }
    delete be;
}

extern "C" void be_info(BeBackend *be, BeDeviceInfo *out) {
    if (!out) {
        return;
    }
    std::memset(out, 0, sizeof(*out));
    out->max_arenas = PPF_BE_MAX_ARENAS;
    out->max_arena_bytes = compute::arena::kMaxArenaBytes;
    // CUDA FAULTS ON AN OUT-OF-BOUNDS DEVICE ACCESS, which is the opposite of
    // Metal and is reported rather than assumed anywhere: a driver may not
    // branch on it, and a report that names its verdict's fault model needs it.
    out->faults_on_oob = 1;
    out->supports_deferred_regions = 1;
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
        std::snprintf(out->device_name, sizeof(out->device_name), "%s",
                      "unknown AMD device");
        return;
    }
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
        std::snprintf(out->device_name, sizeof(out->device_name), "%s",
                      "unknown AMD device");
        return;
    }
    std::snprintf(out->device_name, sizeof(out->device_name), "%s", prop.name);
    out->max_threads_per_group =
        static_cast<uint32_t>(prop.maxThreadsPerBlock);
    out->max_group_scratch_bytes =
        static_cast<uint32_t>(prop.sharedMemPerBlock);
}

// ===========================================================================
// THE KERNEL TABLE, AS THE DRIVER SEES IT
// ===========================================================================

extern "C" uint32_t be_kernel_count(BeBackend *) { return kKernelCount; }

extern "C" const char *be_kernel_name(BeBackend *, uint32_t kernel_id) {
    return kernel_id < kKernelCount ? kKernelTable[kernel_id].name : nullptr;
}

extern "C" BeStatus be_kernel_id_by_name(BeBackend *, const char *name,
                                              uint32_t *out, BeError *err) {
    if (!name || !out) {
        return misuse(err, "be_kernel_id_by_name needs a name and an "
                           "output slot");
    }
    for (unsigned i = 0; i < kKernelCount; ++i) {
        if (std::strcmp(kKernelTable[i].name, name) == 0) {
            *out = i;
            return STATUS_OK;
        }
    }
    return set_error(err, STATUS_NO_KERNEL, 0,
                     std::string("this library carries no kernel named ") +
                         name);
}

extern "C" uint32_t be_kernel_args_bytes(BeBackend *, uint32_t kernel_id) {
    return kernel_id < kKernelCount ? kKernelTable[kernel_id].args_bytes : 0;
}

extern "C" int32_t be_kernel_present(BeBackend *, uint32_t kernel_id) {
    // EVERY ROW IN THIS TABLE IS AN IMPLEMENTATION. A row exists because a
    // generated entry point was compiled into this library, so absence is
    // absence from the table rather than a row with nothing behind it. A
    // target that compiles a subset answers this differently, which is why the
    // question is on the ABI at all.
    return kernel_id < kKernelCount ? 1 : 0;
}

extern "C" BeStatus be_prepare_kernels(BeBackend *be,
                                            const uint32_t *ids, uint32_t count,
                                            BeError *err) {
    if (lost(be, err)) {
        return STATUS_DEVICE_LOST;
    }
    if (count > 0 && !ids) {
        return misuse(err, "be_prepare_kernels was given a count and no "
                           "ids");
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (ids[i] >= kKernelCount) {
            return set_error(err, STATUS_NO_KERNEL, 0,
                             "kernel id " + std::to_string(ids[i]) +
                                 " is outside this library's table of " +
                                 std::to_string(kKernelCount));
        }
    }
    // CUDA HAS NO PIPELINE CREATION STEP. Its kernels are cubins linked into
    // this library at build time, so there is nothing to create and nothing to
    // cache; the ids are still validated, because a driver naming an id this
    // library lacks must learn it here rather than mid-step.
    return STATUS_OK;
}

extern "C" void be_shader_cache_report(BeBackend *,
                                           BeShaderCacheReport *out) {
    if (out) {
        // All zero, and honestly so: this target compiles no shader at run
        // time and keeps no pipeline cache, so every figure a report would
        // carry is genuinely nothing rather than unmeasured.
        std::memset(out, 0, sizeof(*out));
    }
}

// ===========================================================================
// MEMORY
// ===========================================================================

extern "C" BeStatus be_alloc(BeBackend *be, size_t count,
                                  size_t elem_size, size_t align,
                                  const char *label, BeHandle *out,
                                  BeError *err) {
    if (lost(be, err)) {
        return STATUS_DEVICE_LOST;
    }
    if (!be || !out) {
        return misuse(err, "be_alloc needs a backend and an output slot");
    }
    std::string detail;
    if (!compute::arena::alloc(be->allocator, count, elem_size, align, out,
                               &detail)) {
        return set_error(err, STATUS_BAD_ALLOC, 0, detail);
    }
    be->generation += 1;
    // A ZERO-COUNT REQUEST REGISTERS NO BLOCK, so it gets no label. The key is
    // (arena, offset) and such a handle names (0, 0), which is also where the
    // first real allocation lands: labelling it would overwrite that block's
    // label, and freeing it would erase it.
    unsigned long long capacity = 0;
    const bool has_block =
        compute::arena::block_bytes(be->allocator, *out, &capacity, &detail);
    if (has_block) {
        be->labels[block_key(*out)] = label;
    }
    // FRESH BYTES ARE UNSPECIFIED UNLESS THE DRIVER SAID OTHERWISE, and the
    // default must not be zero: a buffer accumulated into but never cleared is
    // correct only while the memory it happens to get is still zero, and fresh
    // device memory frequently is, so zeroing here would hide exactly the
    // defect it looks like it prevents.
    if (be->poison_byte >= 0 && has_block && capacity > 0) {
        unsigned char *base = static_cast<unsigned char *>(
            compute::arena::host_resolve(be->allocator, *out, &detail));
        if (base) {
            const hipError_t code = hipMemset(
                base, be->poison_byte, static_cast<size_t>(capacity));
            if (code != hipSuccess) {
                return fault(be, err, code, "poisoning a fresh allocation");
            }
        }
    }
    return STATUS_OK;
}

extern "C" BeStatus be_grow(BeBackend *be, BeHandle *handle,
                                 size_t new_count, size_t elem_size,
                                 size_t align, BeError *err) {
    if (lost(be, err)) {
        return STATUS_DEVICE_LOST;
    }
    if (!be || !handle) {
        return misuse(err, "be_grow needs a backend and a handle");
    }
    const unsigned long long was = block_key(*handle);
    const char *label = nullptr;
    auto it = be->labels.find(was);
    if (it != be->labels.end()) {
        label = it->second;
    }
    std::string detail;
    if (!compute::arena::grow(be->allocator, handle, new_count, elem_size,
                              align, &detail)) {
        return set_error(err, STATUS_BAD_ALLOC, 0, detail);
    }
    // THE BLOCK MAY HAVE MOVED, so every copy of the handle held elsewhere is
    // stale, a recorded region included. The generation is what a replay
    // compares against.
    be->generation += 1;
    be->labels.erase(was);
    be->labels[block_key(*handle)] = label;
    return STATUS_OK;
}

extern "C" BeStatus be_free(BeBackend *be, BeHandle *handle,
                                 BeError *err) {
    if (lost(be, err)) {
        return STATUS_DEVICE_LOST;
    }
    if (!be || !handle) {
        return misuse(err, "be_free needs a backend and a handle");
    }
    const unsigned long long was = block_key(*handle);
    std::string detail;
    if (!compute::arena::free(be->allocator, handle, &detail)) {
        return misuse(err, detail);
    }
    be->generation += 1;
    be->labels.erase(was);
    return STATUS_OK;
}

extern "C" const char *be_handle_label(BeBackend *be, BeHandle handle) {
    if (!be) {
        return nullptr;
    }
    auto it = be->labels.find(block_key(handle));
    return it == be->labels.end() ? nullptr : it->second;
}

namespace {

// The window rule every transfer and every fill shares: a byte offset from the
// START of the block, with both ends checked against the block's capacity, so a
// caller may move a window of a block holding several arrays end to end.
//
// THE BOUND IS THE ALLOCATOR'S AND NOT THE HANDLE'S, and reading it off the
// handle is the mistake this comment exists to prevent. `BeHandle::size` and
// `BeHandle::allocated` are ELEMENT counts; the byte capacity is the element
// count times an element size only the allocator recorded. Using `allocated` as
// a byte bound refuses every window past the element count and admits none past
// the real end, which is a refusal that looks like a bounds check and is not
// one.
//
// A zero-byte window at offset zero is accepted without resolving anything: a
// count of 0 yields a real zero-length handle naming a bound arena, and every
// caller below returns before touching `base` in that case.
BeStatus window(BeBackend *be, BeHandle handle, size_t byte_offset,
                 size_t bytes, unsigned char **base, BeError *err) {
    *base = nullptr;
    if (bytes == 0 && byte_offset == 0) {
        return STATUS_OK;
    }
    std::string detail;
    unsigned long long capacity = 0;
    if (!compute::arena::block_bytes(be->allocator, handle, &capacity,
                                     &detail)) {
        return misuse(err, detail);
    }
    if (byte_offset > capacity ||
        bytes > static_cast<size_t>(capacity) - byte_offset) {
        return misuse(err, "the window [" + std::to_string(byte_offset) + ", " +
                               std::to_string(byte_offset + bytes) +
                               ") is outside the " + std::to_string(capacity) +
                               " bytes this handle names");
    }
    unsigned char *resolved = static_cast<unsigned char *>(
        compute::arena::host_resolve(be->allocator, handle, &detail));
    if (!resolved) {
        return misuse(err, detail);
    }
    *base = resolved + byte_offset;
    return STATUS_OK;
}

} // namespace

extern "C" BeStatus be_write(BeBackend *be, BeHandle handle,
                                  size_t byte_offset, const void *src,
                                  size_t bytes, BeError *err) {
    if (lost(be, err)) {
        return STATUS_DEVICE_LOST;
    }
    if (!be || (!src && bytes > 0)) {
        return misuse(err, "be_write needs a backend and a source");
    }
    unsigned char *base = nullptr;
    const BeStatus status = window(be, handle, byte_offset, bytes, &base, err);
    if (status != STATUS_OK) {
        return status;
    }
    if (be->transfer_stats) {
        const char *name = be_handle_label(be, handle);
        auto &row = be->transfers[name ? name : "<unlabelled>"];
        row.h2d_calls += 1;
        row.h2d_bytes += bytes;
    }
    if (bytes == 0) {
        return STATUS_OK;
    }
    const hipError_t code =
        hipMemcpy(base, src, bytes, hipMemcpyHostToDevice);
    if (code != hipSuccess) {
        return fault(be, err, code, "hipMemcpy host to device");
    }
    be->counters.bytes_uploaded += bytes;
    return STATUS_OK;
}

extern "C" BeStatus be_read(BeBackend *be, BeHandle handle,
                                 size_t byte_offset, void *dst, size_t bytes,
                                 BeError *err) {
    if (lost(be, err)) {
        return STATUS_DEVICE_LOST;
    }
    if (!be || (!dst && bytes > 0)) {
        return misuse(err, "be_read needs a backend and a destination");
    }
    unsigned char *base = nullptr;
    const BeStatus status = window(be, handle, byte_offset, bytes, &base, err);
    if (status != STATUS_OK) {
        return status;
    }
    if (be->transfer_stats) {
        const char *name = be_handle_label(be, handle);
        auto &row = be->transfers[name ? name : "<unlabelled>"];
        row.d2h_calls += 1;
        row.d2h_bytes += bytes;
    }
    if (bytes == 0) {
        return STATUS_OK;
    }
    const hipError_t code =
        hipMemcpy(dst, base, bytes, hipMemcpyDeviceToHost);
    if (code != hipSuccess) {
        return fault(be, err, code, "hipMemcpy device to host");
    }
    be->counters.bytes_downloaded += bytes;
    be->counters.syncs += 1;
    return STATUS_OK;
}

extern "C" BeStatus be_copy(BeBackend *be, BeHandle dst,
                                 size_t dst_byte_offset, BeHandle src,
                                 size_t src_byte_offset, size_t bytes,
                                 BeError *err) {
    if (lost(be, err)) {
        return STATUS_DEVICE_LOST;
    }
    if (!be) {
        return misuse(err, "be_copy needs a backend");
    }
    unsigned char *dst_base = nullptr;
    BeStatus status = window(be, dst, dst_byte_offset, bytes, &dst_base, err);
    if (status != STATUS_OK) {
        return status;
    }
    unsigned char *src_base = nullptr;
    status = window(be, src, src_byte_offset, bytes, &src_base, err);
    if (status != STATUS_OK) {
        return status;
    }
    if (bytes == 0) {
        return STATUS_OK;
    }
    // NOT COUNTED AS A TRANSFER, because nothing crosses the bus. The rows
    // `PPF_TRANSFER_STATS` prints are host-to-device and device-to-host, and a
    // device-to-device copy belongs in neither; counting it in both is what the
    // host bounce this replaces effectively did.
    const hipError_t code =
        hipMemcpy(dst_base, src_base, bytes, hipMemcpyDeviceToDevice);
    if (code != hipSuccess) {
        return fault(be, err, code, "hipMemcpy device to device");
    }
    return STATUS_OK;
}

extern "C" BeStatus be_read_scalars(BeBackend *be, BeHandle handle,
                                         uint32_t first, float *dst,
                                         uint32_t count, BeError *err) {
    if (lost(be, err)) {
        return STATUS_DEVICE_LOST;
    }
    if (!be || (!dst && count > 0)) {
        return misuse(err, "be_read_scalars needs a backend and a "
                           "destination");
    }
    if (count == 0) {
        return STATUS_OK;
    }
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    unsigned char *base = nullptr;
    const BeStatus status =
        window(be, handle, static_cast<size_t>(first) * sizeof(float), bytes,
               &base, err);
    if (status != STATUS_OK) {
        return status;
    }
    // THE PINNED STAGING BUFFER IS THE WHOLE REASON THIS CALL IS SEPARATE FROM
    // be_read. A device-to-host copy into PAGEABLE host memory degrades to
    // a blocking staged transfer costing about 100 microseconds of host time
    // per call whatever the size; with a pinned destination the same copy is a
    // direct DMA at about 5 microseconds.
    float *stage = static_cast<float *>(pinned_scratch(bytes));
    const hipError_t code =
        hipMemcpy(stage, base, bytes, hipMemcpyDeviceToHost);
    if (code != hipSuccess) {
        return fault(be, err, code, "hipMemcpy device to host (scalars)");
    }
    std::memcpy(dst, stage, bytes);
    be->counters.bytes_downloaded += bytes;
    be->counters.syncs += 1;
    return STATUS_OK;
}

// NO HOST VIEW, AND THAT IS A FACT ABOUT THE PLATFORM RATHER THAN A GAP. An
// arena's base is a hipMalloc, so the host cannot address the bytes at all;
// `host_resolve` computes a DEVICE address on the host and is not a mapping.
// The header makes NULL the answer for such a target, and the caller's copy
// path through be_write and be_read is what serves it, unchanged.
//
// THE HANDLE AND THE WINDOW ARE STILL CHECKED, THROUGH THE SAME `window` EVERY
// TRANSFER USES, because the header asks for the refusal on a backend that
// answers NULL as well as on one that answers an address. This library is the
// reference the other targets are checked against, so validating only where a
// pointer comes back would answer a driver that invented a handle with
// STATUS_OK here and STATUS_MISUSE on Metal: the defect would be reported only
// where the memory happens to be mapped. The resolved base is discarded, which
// is the point of reusing the helper rather than restating the bound. What is
// wanted from it is the capacity, and the capacity is the allocator's rather
// than the handle's.
extern "C" BeStatus be_host_ptr(BeBackend *be, BeHandle handle,
                                size_t byte_offset, size_t bytes, void **out,
                                BeError *err) {
    if (!be || !out) {
        return misuse(err, "be_host_ptr needs a backend and an output slot");
    }
    *out = nullptr;
    if (bytes == 0) {
        return STATUS_OK;
    }
    unsigned char *base = nullptr;
    return window(be, handle, byte_offset, bytes, &base, err);
}

extern "C" uint64_t be_allocator_generation(BeBackend *be) {
    return be ? be->generation : 0;
}

extern "C" uint32_t be_arena_count(BeBackend *be) {
    return be ? compute::arena::arena_count(be->allocator) : 0;
}

extern "C" uint64_t be_bytes_used(BeBackend *be) {
    return be ? compute::arena::bytes_used(be->allocator) : 0;
}

extern "C" uint64_t be_bytes_reserved(BeBackend *be) {
    return be ? compute::arena::bytes_reserved(be->allocator) : 0;
}

// ===========================================================================
// EXECUTION
// ===========================================================================

extern "C" BeStatus be_encode_begin(BeBackend *be, const char *region,
                                         BeEncodeMode mode, BeEncoder **out,
                                         BeError *err) {
    if (lost(be, err)) {
        return STATUS_DEVICE_LOST;
    }
    if (!be || !out) {
        return misuse(err, "be_encode_begin needs a backend and an output "
                           "slot");
    }
    if (be->open_encoder) {
        return misuse(err, "an encoder is already open on this backend");
    }
    if (mode != ENCODE_IMMEDIATE && mode != ENCODE_RECORD) {
        return misuse(err, "encode mode " +
                               std::to_string(static_cast<int>(mode)) +
                               " is not one this ABI declares");
    }
    // COUNTED HERE BECAUSE THIS IS WHERE THE NAME IS. An IMMEDIATE boundary ends
    // in a submit and therefore a synchronize, so it is charged at that
    // synchronize below. A RECORD boundary does not submit, so it is charged
    // where it DOES stall, at `be_replay`, under the name kept on the region.
    // Counting the record here as well would double a recorded pass.
    if (be->region_stats && mode == ENCODE_IMMEDIATE) {
        be->open_region = region ? region : "(unnamed)";
        be->open_at = std::chrono::steady_clock::now();
        be->regions[be->open_region].submits += 1;
    }
    auto *enc = new BeEncoder();
    enc->backend = be;
    enc->region = region;
    enc->mode = mode;
    enc->dispatch_count = 0;
    enc->fill_count = 0;
    enc->capturing = false;
    // CAPTURE BEGINS HERE AND NOT AT THE CLOSE, which is why the mode is
    // stated at open at all: CUDA has to be capturing before the first launch
    // is issued, so a library cannot decide after the fact which of the two it
    // was building.
    if (mode == ENCODE_RECORD) {
        const hipError_t code = hipStreamBeginCapture(
            be->queue, hipStreamCaptureModeThreadLocal);
        if (code == hipSuccess) {
            enc->capturing = true;
        } else if (be->require_deferred) {
            delete enc;
            return set_error(err, STATUS_NO_DEFERRAL,
                             static_cast<long long>(code),
                             std::string("stream capture could not begin: ") +
                                 hipGetErrorString(code));
        } else {
            // A FALLBACK IS NEVER SILENT. The replay will re-issue ordinary
            // dispatches, which produces bit-identical numbers and shows up
            // only as a slowdown, and a slowdown is inside this project's
            // measured run-to-run envelope. The counter is the only thing that
            // can see it, so it is bumped at the close rather than logged here.
            emit_log(be, 1,
                     std::string("stream capture could not begin (") +
                         hipGetErrorString(code) +
                         "); this region will replay by re-issuing dispatches");
        }
    }
    be->open_encoder = enc;
    *out = enc;
    return STATUS_OK;
}

extern "C" BeStatus be_encode_dispatch(BeEncoder *enc, uint32_t kernel_id,
                                            const BeExtent *extent,
                                            const void *args,
                                            uint32_t args_bytes,
                                            BeError *err) {
    if (!enc || !extent) {
        return misuse(err, "be_encode_dispatch needs an encoder and an "
                           "extent");
    }
    if (lost(enc->backend, err)) {
        return STATUS_DEVICE_LOST;
    }
    if (kernel_id >= kKernelCount) {
        return set_error(err, STATUS_NO_KERNEL, 0,
                         "kernel id " + std::to_string(kernel_id) +
                             " is outside this library's table of " +
                             std::to_string(kKernelCount));
    }
    const KernelRow &row = kKernelTable[kernel_id];
    if (args_bytes != row.args_bytes) {
        return misuse(err, std::string("the record for ") + row.name + " is " +
                               std::to_string(row.args_bytes) +
                               " bytes in this library and the caller passed " +
                               std::to_string(args_bytes));
    }
    if (args_bytes > PPF_BE_MAX_ARGS_BYTES) {
        return misuse(err, std::string("the record for ") + row.name + " is " +
                               std::to_string(args_bytes) +
                               " bytes, over the " +
                               std::to_string(PPF_BE_MAX_ARGS_BYTES) +
                               " this ABI carries");
    }
    if (args_bytes > 0 && !args) {
        return misuse(err, std::string("the record for ") + row.name +
                               " is declared non-empty and no bytes were "
                               "passed");
    }
    // THE EXTENT KIND MUST MATCH THE SHAPE THE KERNEL WAS GENERATED FOR, and
    // the two are not interchangeable: an element launch of a group kernel
    // would start `count` threads over `count` groups' worth of work and read
    // the group index as a thread index, which is a wrong answer rather than a
    // failure.
    const bool wants_groups = row.group != nullptr;
    const bool got_groups = extent->kind == EXTENT_GROUPS;
    if (extent->kind != EXTENT_ELEMENTS &&
        extent->kind != EXTENT_GROUPS) {
        return misuse(err, "extent kind " + std::to_string(extent->kind) +
                               " is not one this ABI declares");
    }
    if (wants_groups != got_groups) {
        return misuse(err, std::string(row.name) + " is a " +
                               (wants_groups ? "GROUP" : "ELEMENT") +
                               "-shaped entry point and was dispatched with " +
                               (got_groups ? "GROUPS" : "ELEMENTS"));
    }
    if (got_groups && extent->threads == 0) {
        return misuse(err, std::string(row.name) +
                               " is group-shaped and the group width is 0");
    }
    // A GENERATED GROUP ENTRY DECLARES ITS SCRATCH STATICALLY, so a dispatch
    // must not ask for any. Refused rather than clamped: Metal validates the
    // static path at pipeline creation and the dynamic path not at all, so a
    // clamp would run a kernel against memory it then indexes past on a
    // platform that does not fault.
    if (extent->scratch_bytes != 0) {
        return misuse(err, std::string(row.name) +
                               " declares its group-local scratch statically (" +
                               std::to_string(row.scratch_bytes) +
                               " bytes); a dispatch may not request any");
    }

    Item item;
    item.is_fill = false;
    item.kernel_id = kernel_id;
    item.extent = *extent;
    item.args.assign(static_cast<const unsigned char *>(args),
                     static_cast<const unsigned char *>(args) + args_bytes);
    item.dst = BeHandle{};
    item.byte_offset = 0;
    item.bytes = 0;
    item.value = 0;
    enc->items.push_back(std::move(item));
    enc->dispatch_count += 1;
    // WHILE CAPTURING, THE WORK IS ISSUED NOW. Stream capture records the
    // launches as they are made, so a region's list has to be issued between
    // begin and end rather than replayed later; the copy above is still kept,
    // because a capture that fails falls back to re-issuing it.
    if (enc->capturing) {
        const Item &back = enc->items.back();
        if (row.element) {
            row.element(back.args.data(), enc->backend->queue);
        } else {
            row.group(back.args.data(), back.extent.threads,
                      enc->backend->queue);
        }
        const hipError_t code = hipGetLastError();
        if (code != hipSuccess) {
            return fault(enc->backend, err, code,
                         std::string("capturing a launch of ") + row.name);
        }
    }
    return STATUS_OK;
}

extern "C" BeStatus be_encode_fill(BeEncoder *enc, BeHandle dst,
                                        size_t byte_offset, uint64_t bytes,
                                        uint8_t value, BeError *err) {
    if (!enc) {
        return misuse(err, "be_encode_fill needs an encoder");
    }
    if (lost(enc->backend, err)) {
        return STATUS_DEVICE_LOST;
    }
    unsigned char *base = nullptr;
    const BeStatus status = window(enc->backend, dst, byte_offset,
                                    static_cast<size_t>(bytes), &base, err);
    if (status != STATUS_OK) {
        return status;
    }
    Item item;
    item.is_fill = true;
    item.kernel_id = 0;
    item.extent = BeExtent{};
    item.dst = dst;
    item.byte_offset = byte_offset;
    item.bytes = bytes;
    item.value = value;
    enc->items.push_back(std::move(item));
    enc->fill_count += 1;
    if (enc->capturing && bytes > 0) {
        const hipError_t code = hipMemsetAsync(
            base, value, static_cast<size_t>(bytes), enc->backend->queue);
        if (code != hipSuccess) {
            return fault(enc->backend, err, code, "capturing a fill");
        }
    }
    return STATUS_OK;
}

extern "C" uint32_t be_encoder_length(BeEncoder *enc) {
    return enc ? static_cast<uint32_t>(enc->items.size()) : 0;
}

namespace {

void close_encoder(BeEncoder *enc) {
    if (enc->backend->open_encoder == enc) {
        enc->backend->open_encoder = nullptr;
    }
    delete enc;
}

} // namespace

extern "C" BeStatus be_encode_submit(BeEncoder *enc,
                                          BeDiagSummary *out_diag,
                                          BeError *err) {
    if (!enc) {
        return misuse(err, "be_encode_submit needs an encoder");
    }
    BeBackend *be = enc->backend;
    const auto submit_at = std::chrono::steady_clock::now();
    if (enc->mode != ENCODE_IMMEDIATE) {
        const BeStatus status =
            misuse(err, "be_encode_submit is valid only on an immediate "
                        "encoder; a recording one closes with "
                        "be_encode_record");
        close_encoder(enc);
        return status;
    }
    if (lost(be, err)) {
        close_encoder(enc);
        return STATUS_DEVICE_LOST;
    }
    const std::vector<Item> items = std::move(enc->items);
    const unsigned dispatches = enc->dispatch_count;
    const unsigned fills = enc->fill_count;
    // THE ENCODER IS INVALID AFTER THIS CALL WHATEVER THE RETURN VALUE, so it
    // is closed before the work rather than on the way out of each branch.
    close_encoder(enc);

    // A FAILING BOUNDARY STILL DELIVERS WHAT THE DEVICE RECORDED, on both exits
    // below, as on CUDA and for the same reason: the record is what EXPLAINS
    // the failure, naming the file, the line and four captured values, while
    // the status names only the API call that reported it. The read is safe
    // after a fault because the channel is mapped host memory that `read` walks
    // in place, so the drain issues no device call.
    const BeStatus status = run_items(be, items, err);
    if (status != STATUS_OK) {
        // SYNCHRONIZED FIRST, BECAUSE A LAUNCH FAILING DOES NOT STOP THE ONES
        // ALREADY ON THE STREAM. Draining here without waiting reads the
        // channel while earlier kernels are still writing it, which is a race
        // whose most likely result is a report that looks complete and is
        // short. The wait is allowed to fail: a poisoned context still leaves
        // the mapped pages readable, and what the drain finds is then whatever
        // the device managed to record.
        hipStreamSynchronize(be->queue);
        drain_diag(be, out_diag);
        return status;
    }
    const hipError_t code = hipStreamSynchronize(be->queue);
    if (code != hipSuccess) {
        const BeStatus failed = fault(be, err, code, "hipStreamSynchronize");
        drain_diag(be, out_diag);
        return failed;
    }
    be->counters.dispatches += dispatches;
    be->counters.fills += fills;
    be->counters.syncs += 1;
    // CHARGED AT THE SYNCHRONIZE, so the row carries encode plus launch plus
    // wait, which is exactly what the host pays for this boundary.
    if (be->region_stats && be->open_region) {
        const auto now = std::chrono::steady_clock::now();
        auto &row = be->regions[be->open_region];
        row.seconds += std::chrono::duration<double>(now - be->open_at).count();
        row.encode += std::chrono::duration<double>(submit_at - be->open_at).count();
        row.wait += std::chrono::duration<double>(now - submit_at).count();
        be->open_region = nullptr;
    }
    drain_diag(be, out_diag);
    return STATUS_OK;
}

extern "C" BeStatus be_encode_record(BeEncoder *enc, BeRegion **out,
                                          BeError *err) {
    if (!enc || !out) {
        return misuse(err, "be_encode_record needs an encoder and an "
                           "output slot");
    }
    BeBackend *be = enc->backend;
    if (enc->mode != ENCODE_RECORD) {
        const BeStatus status =
            misuse(err, "be_encode_record is valid only on a recording "
                        "encoder; an immediate one closes with "
                        "be_encode_submit");
        close_encoder(enc);
        return status;
    }
    if (lost(be, err)) {
        close_encoder(enc);
        return STATUS_DEVICE_LOST;
    }

    auto *region = new BeRegion();
    region->deferred = false;
    region->label = enc->region ? enc->region : "(unnamed)";
    region->exec = nullptr;
    region->dispatch_count = enc->dispatch_count;
    region->fill_count = enc->fill_count;
    region->allocator_generation = be->generation;
    region->items = std::move(enc->items);

    if (enc->capturing) {
        hipGraph_t graph = nullptr;
        hipError_t code = hipStreamEndCapture(be->queue, &graph);
        if (code == hipSuccess && graph) {
            hipGraphExec_t exec = nullptr;
            // HIP KEEPS THE FIVE-ARGUMENT FORM. CUDA 12 added a three-argument
                // `cudaGraphInstantiate(exec, graph, flags)`; HIP declares only
                // `(exec, graph, errorNode, logBuffer, bufferSize)`, so the two
                // diagnostic outputs are passed as absent rather than dropped.
                code = hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
            hipGraphDestroy(graph);
            if (code == hipSuccess) {
                region->deferred = true;
                region->exec = exec;
            }
        }
        if (!region->deferred) {
            // The capture is already invalid at this point, so the stream is
            // usable again and re-issuing is the only thing left. Whether that
            // is acceptable was decided at BEGIN, by require_deferred_regions.
            (void)hipGetLastError();
            if (be->require_deferred) {
                const BeStatus status = set_error(
                    err, STATUS_NO_DEFERRAL, static_cast<long long>(code),
                    std::string("stream capture could not be realized as a "
                                "graph: ") +
                        hipGetErrorString(code));
                delete region;
                close_encoder(enc);
                return status;
            }
            emit_log(be, 1,
                     std::string("stream capture did not yield a graph (") +
                         hipGetErrorString(code) +
                         "); this region will replay by re-issuing dispatches");
        }
    }
    close_encoder(enc);

    be->counters.regions_recorded += 1;
    if (region->deferred) {
        be->counters.regions_deferred += 1;
    } else {
        be->counters.regions_fallback += 1;
    }
    *out = region;
    return STATUS_OK;
}

extern "C" void be_encode_abandon(BeEncoder *enc) {
    if (!enc) {
        return;
    }
    if (enc->capturing) {
        hipGraph_t graph = nullptr;
        if (hipStreamEndCapture(enc->backend->queue, &graph) == hipSuccess &&
            graph) {
            hipGraphDestroy(graph);
        }
        (void)hipGetLastError();
    }
    close_encoder(enc);
}

extern "C" void be_region_info(BeRegion *region, BeRegionInfo *out) {
    if (!out) {
        return;
    }
    std::memset(out, 0, sizeof(*out));
    if (!region) {
        return;
    }
    out->deferred = region->deferred ? 1 : 0;
    out->dispatch_count = region->dispatch_count;
    out->fill_count = region->fill_count;
    out->allocator_generation = region->allocator_generation;
}

extern "C" BeStatus be_replay(BeBackend *be, BeRegion *region,
                                   uint32_t repeats, BeDiagSummary *out_diag,
                                   BeError *err) {
    if (lost(be, err)) {
        return STATUS_DEVICE_LOST;
    }
    if (!be || !region) {
        return misuse(err, "be_replay needs a backend and a region");
    }
    // A GROW MAY HAVE MOVED A BLOCK, so a recorded argument record can hold a
    // stale handle and replaying it would address memory the allocator has
    // since handed to something else.
    if (region->allocator_generation != be->generation) {
        return set_error(err, STATUS_STALE_REGION, 0,
                         "the region was recorded at allocator generation " +
                             std::to_string(region->allocator_generation) +
                             " and the allocator is now at " +
                             std::to_string(be->generation));
    }
    if (repeats == 0) {
        return STATUS_OK;
    }
    const auto replay_at = std::chrono::steady_clock::now();
    // A FAILING REPEAT STILL DELIVERS WHAT THE DEVICE RECORDED, for the reason
    // `be_encode_submit` does: the record names the check, the file, the line
    // and four captured values, while the status names only the call that
    // reported the failure. The wait before the drain is what stops the read
    // racing repeats already on the stream, and it is allowed to fail, since a
    // poisoned context still leaves the mapped pages readable.
    for (uint32_t i = 0; i < repeats; ++i) {
        if (region->deferred) {
            const hipError_t code = hipGraphLaunch(region->exec, be->queue);
            if (code != hipSuccess) {
                const BeStatus failed = fault(be, err, code, "hipGraphLaunch");
                hipStreamSynchronize(be->queue);
                drain_diag(be, out_diag);
                return failed;
            }
        } else {
            const BeStatus status = run_items(be, region->items, err);
            if (status != STATUS_OK) {
                hipStreamSynchronize(be->queue);
                drain_diag(be, out_diag);
                return status;
            }
        }
    }
    // ONE WAIT AND ONE DRAIN FOR THE WHOLE BATCH, which is the contract a
    // recorded body is accepted under: a device failure raised on repeat k is
    // reported after repeat repeats - 1 has run, so a body may be recorded
    // only if its failure mode is latch and continue.
    const auto replay_submitted = std::chrono::steady_clock::now();
    const hipError_t code = hipStreamSynchronize(be->queue);
    if (code != hipSuccess) {
        const BeStatus failed = fault(be, err, code, "hipStreamSynchronize");
        drain_diag(be, out_diag);
        return failed;
    }
    be->counters.replays += 1;
    be->counters.replay_repeats += repeats;
    be->counters.dispatches +=
        static_cast<uint64_t>(region->dispatch_count) * repeats;
    be->counters.fills += static_cast<uint64_t>(region->fill_count) * repeats;
    // CHARGED LIKE AN IMMEDIATE SUBMIT, one row per region name, so a recorded
    // pass appears beside the ones that never left the immediate path. The
    // repeats are one submit: the whole batch runs back to back and waits once,
    // which is the reason the region exists.
    if (be->region_stats) {
        auto &row = be->regions[region->label.empty() ? "(unnamed)"
                                                      : region->label];
        row.submits += 1;
        const auto now = std::chrono::steady_clock::now();
        row.seconds += std::chrono::duration<double>(now - replay_at).count();
        row.encode +=
            std::chrono::duration<double>(replay_submitted - replay_at).count();
        row.wait +=
            std::chrono::duration<double>(now - replay_submitted).count();
    }
    be->counters.syncs += 1;
    drain_diag(be, out_diag);
    return STATUS_OK;
}

extern "C" void be_region_release(BeBackend *, BeRegion *region) {
    if (!region) {
        return;
    }
    if (region->exec) {
        hipGraphExecDestroy(region->exec);
    }
    delete region;
}

// ===========================================================================
// THE DIAGNOSTIC CHANNEL
// ===========================================================================

extern "C" BeStatus be_diag_drain(BeBackend *be, BeDiagRecord *out,
                                       uint32_t capacity, uint32_t *out_count,
                                       BeError *err) {
    if (!be || !out_count) {
        return misuse(err, "be_diag_drain needs a backend and a count "
                           "slot");
    }
    diagnostics::Readback back;
    std::string detail;
    if (!diagnostics::read(be->diag, &back, &detail)) {
        return set_error(err, STATUS_PLATFORM, 0,
                         "the diagnostic channel could not be read: " + detail);
    }
    *out_count = static_cast<uint32_t>(back.ring.size());
    const uint32_t copy = *out_count < capacity ? *out_count : capacity;
    if (copy > 0 && out) {
        std::memcpy(out, back.ring.data(), copy * sizeof(BeDiagRecord));
    }
    return STATUS_OK;
}

extern "C" BeStatus be_diag_file_path(BeBackend *, uint32_t file_id,
                                           char *out, size_t capacity,
                                           BeError *err) {
    if (!out || capacity == 0) {
        return misuse(err, "be_diag_file_path needs a buffer");
    }
    // THE ID IS A COMPILE-TIME FNV-1a HASH OF __FILE__, so it is reversed by a
    // table rather than by arithmetic. `kernelgen` emits one row per neutral
    // source from the same function that writes that source's `#line`
    // directive, which is what makes the hash here and the `__FILE__` there the
    // same string.
    const char *path = diagnostics::file_path(file_id);
    if (path == nullptr) {
        // The honest answer for an id this library did not compile in. A
        // plausible path invented here would put the wrong file in a device
        // assert's report, which is worse than the number.
        return set_error(err, STATUS_MISUSE, 0,
                         "id " + std::to_string(file_id) +
                             " names no source this library carries");
    }
    const size_t length = std::strlen(path);
    if (length + 1 > capacity) {
        return set_error(err, STATUS_MISUSE, 0,
                         "the path for id " + std::to_string(file_id) + " is " +
                             std::to_string(length) +
                             " bytes and the caller's buffer holds " +
                             std::to_string(capacity));
    }
    std::memcpy(out, path, length + 1);
    return STATUS_OK;
}

// ===========================================================================
// COUNTERS
// ===========================================================================

extern "C" void be_counters(BeBackend *be, BeCounters *out) {
    if (!out) {
        return;
    }
    if (!be) {
        std::memset(out, 0, sizeof(*out));
        return;
    }
    *out = be->counters;
}

extern "C" void be_counters_reset(BeBackend *be) {
    if (be) {
        std::memset(&be->counters, 0, sizeof(be->counters));
    }
}
