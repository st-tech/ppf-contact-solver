// File: backend.mm
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// THE METAL TARGET, AS THE C ABI THE NEUTRAL DRIVER CALLS.
//
// `kernels/seam/backend_abi.h` declares one boundary and this file answers it
// with Metal, on the same terms `backend.cu` answers it with nvcc. What crosses
// is allocation, free, transfer and launch, plus the platform machinery those
// four need, and nothing else: no convergence test, no phase ordering, no
// fallback, no parameter interpretation, no branch on a physical quantity. An
// argument record crosses as OPAQUE BYTES whose length is the only thing
// checked here. So this file names no kernel, declares no argument record and
// reads no field of one.
//
// IT ASSEMBLES NO SHADER, AND THAT IS THE DIFFERENCE FROM THE BACKEND IN THE
// HOLDING PEN. That one splices segments into one string at run time, in a
// hand-kept order, because `newLibraryWithSource` is handed source with no
// filesystem behind it. This one loads a library the BUILD produced: `xcrun
// metal` is clang and has a filesystem, so each generated entry compiles
// offline as its own translation unit with live includes and `xcrun metallib`
// links the AIR. `entries_probe.mm` is what established that every entry in
// such a library yields a pipeline.
//
// WHAT MAKES THAT POSSIBLE IS THE GENERATED TABLE. `kernelgen.py --emit table
// --target metal` renders one fragment per neutral kernel source; the build
// concatenates them in sorted source order; a kernel id is a row's POSITION.
// The driver builds its half the same way from the `rust` fragments and
// compares the two id by id, by name and by record size, so two trees that
// disagree are a named refusal at open rather than a dispatch of the wrong
// kernel with the right bytes.
//
// WHERE THIS IS NOT YET THE ABI IT IMPLEMENTS, stated here rather than
// discovered. A region is recorded as the OPS it holds and replayed by
// re-encoding them, which is correct and is not the deferred execution the
// header describes: Metal's indirect command buffers are what would make a
// replay skip encoding, and `be_info` reports `supports_deferred_regions = 0`
// so a driver that requires them is refused at open rather than served slowly
// without being told.

#include "seam/backend_abi.h"

#include "arena.hpp"
#include "diagnostics.hpp"
#include "metal_context.hpp"

// The compute crate's Metal mechanism is one namespace. This file is the C
// boundary over it, so every name it uses comes from there and none of what it
// DEFINES does: the `be_*` functions are `extern "C"` at global scope, which is
// what the ABI is.
using namespace metal_backend;

#include <cstring>
#include <string>
#include <vector>

// THE ONE INTERNAL USE OF THIS ABI. Closing a backend with an encoder still
// open discards that encoder, and discarding one is exactly what
// `be_encode_abandon` is, so it is called rather than reimplemented: a second
// copy of "drop the ops and clear the slot" is a place for the two to disagree
// about whether an encoder was ended.
extern "C" void be_encode_abandon(BeEncoder *enc);

namespace {

// ===========================================================================
// THE GENERATED KERNEL TABLE
//
// A row is a name, the record's size, the launch shape and the scratch a group
// entry declares statically. Nothing here is hand-written: the concatenation is
// `kernelgen.py --emit table --target metal` over every neutral kernel source,
// in sorted source order, and a kernel id is a row's position in it.
// ===========================================================================

struct KernelRow {
    const char *name;
    uint32_t args_bytes;
    bool group;
    uint32_t scratch_bytes;
    // WHERE `seam_arena_count` SITS IN THE RECORD, or kNoArenaSlot when the
    // record holds no handle and therefore carries no such field. Only the
    // LIBRARY knows the live arena count, and `ARENA_PTR` resolves an
    // out-of-range arena id to the LAST arena rather than faulting, so without
    // a live bound a wrong id is a silent wrong answer with plausible floats.
    // The CUDA side fills it in the generated launcher, because it has one;
    // this backend binds an entry point by name and hands it the record as
    // bytes, so it patches the field itself.
    uint32_t arena_count_offset;
};

const uint32_t kNoArenaSlot = 0xffffffffu;

#define PPF_BE_KERNEL_METAL(name, size, group, scratch, arena_off)            \
    KernelRow{name, size, group, scratch, arena_off},
const KernelRow kKernels[] = {
#include "kernel_table_metal.h"
};
#undef PPF_BE_KERNEL_METAL

const uint32_t kKernelCount =
    static_cast<uint32_t>(sizeof(kKernels) / sizeof(kKernels[0]));

// AN EMPTY TABLE IS THE SILENT FAILURE. A library carrying no row would open,
// report zero kernels, and let the driver's cross-check pass over nothing.
static_assert(sizeof(kKernels) / sizeof(kKernels[0]) > 0,
              "the generated kernel table is empty: the build concatenated no "
              "table fragment, so this library would dispatch nothing and "
              "report that as agreement");

// The seam's handle and the allocator's are layout-identical by construction,
// which the header states and this checks rather than assumes.
static_assert(sizeof(BeHandle) == sizeof(ArenaHandle),
              "BeHandle and ArenaHandle have diverged");

// THE WINDOW BOUND, CHECKED HERE RATHER THAN LEFT TO THE ALLOCATOR, because
// the two answers differ in both wording and STATUS: an allocator refusing its
// own call is a platform refusal, and a caller addressing outside a block it
// holds is a driver defect, which is STATUS_MISUSE. The CUDA library states it
// this way and a driver reads one contract at this boundary, so the message is
// the same one on purpose.
//
// The bound is the BLOCK's byte capacity. A handle's `size` and `allocated` are
// ELEMENT counts, so a bound taken from one of them would refuse every window
// past the element count and admit none past the real end.
BeStatus check_window(Allocator *alloc, const BeHandle &handle,
                      size_t byte_offset, size_t bytes, BeError *err);

ArenaHandle to_arena(const BeHandle &h) {
    ArenaHandle out{};
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

BeHandle from_arena(const ArenaHandle &h) {
    BeHandle out{};
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

// ===========================================================================
// ONE OP, WHICH IS WHAT A REGION HOLDS
//
// Encoding is DEFERRED TO SUBMIT even in immediate mode, because a fill takes
// its own command buffer (a slot from context_begin_command_buffer already
// carries an open compute encoder, and Metal permits one at a time) and a
// sequence that mixes the two has to be replayed in order either way. Holding
// the ops is what lets one code path serve submit and replay.
// ===========================================================================

struct Op {
    bool is_fill;
    // Dispatch.
    uint32_t kernel;
    BeExtent extent;
    std::vector<unsigned char> args;
    // Fill.
    BeHandle dst;
    uint32_t value;
    uint64_t byte_offset;
    uint64_t byte_length;
};

} // namespace

struct BeRegion {
    std::vector<Op> ops;
    uint64_t generation;
    uint32_t dispatches;
    uint32_t fills;
};

struct BeEncoder {
    BeBackend *be;
    BeEncodeMode mode;
    std::vector<Op> ops;
    std::string region;
};

struct BeBackend {
    Context *ctx;
    Allocator *alloc;
    Diagnostics *diag;
    unsigned library;
    // One pipeline per kernel id, created lazily or by be_prepare_kernels. Zero
    // means "not created yet", which is the same sentinel context_make_pipeline
    // returns on failure, so a creation failure is reported at the call rather
    // than cached as an absence.
    std::vector<unsigned> pipelines;
    BeLogFn log;
    void *log_context;
    int32_t poison_byte;
    // A REGION HERE IS RECORDED OPS RE-ENCODED, not deferred execution.
    // The header puts that refusal at RECORD time rather than at open, so
    // a gate sees it before any measurement is taken, which is why this is
    // remembered rather than acted on above.
    bool require_deferred;
    bool device_lost;
    uint64_t generation;
    BeEncoder *open_encoder;
    BeCounters counters;
};

namespace {

BeBackend *g_backend = nullptr;

// THE CALLER'S SINK, REACHED FROM A CALLBACK THAT CARRIES NO CONTEXT POINTER.
// `context_create` takes a plain `void (*)(const char *)` and REQUIRES one:
// this backend's non-fatal messages must not reach stderr, which the frontend
// reads as a crashed run. These are file statics rather than fields because the
// bridge is called during `be_open`, before there is a backend to hang them on,
// and because this library supports one live backend at a time, which is the
// same fact `g_backend` already rests on.
BeLogFn g_open_log = nullptr;
void *g_open_log_context = nullptr;

void context_log_bridge(const char *message) {
    if (g_open_log && message) {
        g_open_log(g_open_log_context, 0, message);
    }
}

void emit_log(BeBackend *be, int32_t level, const std::string &message) {
    if (be && be->log) {
        be->log(be->log_context, level, message.c_str());
    }
}

BeStatus set_error(BeError *err, BeStatus status, int64_t platform_code,
                   const std::string &detail) {
    if (err) {
        err->status = static_cast<int32_t>(status);
        err->platform_code = platform_code;
        const size_t n = detail.size() < (PPF_BE_ERROR_DETAIL - 1)
                             ? detail.size()
                             : (PPF_BE_ERROR_DETAIL - 1);
        err->truncated = detail.size() > n ? 1 : 0;
        std::memcpy(err->detail, detail.data(), n);
        err->detail[n] = '\0';
    }
    return status;
}

BeStatus misuse(BeError *err, const std::string &detail) {
    return set_error(err, STATUS_MISUSE, 0, detail);
}

BeStatus check_window(Allocator *alloc, const BeHandle &handle,
                      size_t byte_offset, size_t bytes, BeError *err) {
    unsigned long long capacity = 0;
    std::string detail;
    if (!allocator_block_bytes(alloc, to_arena(handle), &capacity, &detail)) {
        return misuse(err, detail);
    }
    if (byte_offset > capacity ||
        bytes > static_cast<size_t>(capacity) - byte_offset) {
        return misuse(err, "the window [" + std::to_string(byte_offset) + ", " +
                               std::to_string(byte_offset + bytes) +
                               ") is outside the " + std::to_string(capacity) +
                               " bytes this handle names");
    }
    return STATUS_OK;
}

BeStatus platform(BeError *err, const std::string &detail) {
    return set_error(err, STATUS_PLATFORM, 0, detail);
}

// The pipeline for one kernel id, created on demand.
unsigned pipeline_for(BeBackend *be, uint32_t kernel_id, std::string *err) {
    if (be->pipelines[kernel_id] != 0) {
        return be->pipelines[kernel_id];
    }
    const unsigned p = context_make_pipeline(
        be->ctx, be->library, kKernels[kernel_id].name, err);
    if (p != 0) {
        be->pipelines[kernel_id] = p;
    }
    return p;
}

// The width one element-wise dispatch uses. The header says the backend picks
// it, because that choice computes no value; 128 is what every element-wise
// dispatch in the shipped Metal backend uses.
constexpr unsigned kElementThreads = 128;

// WHETHER THIS BACKEND STILL EXECUTES SYNCHRONOUSLY, which is the first of the
// two conditions `backend_abi.h` binds a backend to before it may hand the
// host an address inside a live block. It holds here: `run_ops` below is the
// sole executor, every command buffer it creates is committed and waited on
// inside the call that created it, and `context_fill_buffer` waits on its own
// before returning.
//
// IT IS A CONSTANT RATHER THAN A MEASUREMENT, BECAUSE THE PROPERTY BELONGS TO
// THE EXECUTOR AND NOT TO A MOMENT. A caller takes an address between seam
// calls, which is exactly the point at which nothing is in flight, so no
// runtime state read there separates a synchronous backend from an
// asynchronous one: an asynchronous `run_ops` would answer the same question
// with the same zero, and the address a caller holds is dereferenced long
// after any such reading anyway. What a change to the execution model does
// touch is this backend's own source, so the refusal the header asks for is
// keyed to this line. Dropping the `waitUntilCompleted` in
// `context_commit_and_wait`, or letting `run_ops` return with work
// outstanding, requires setting this to false; `be_host_ptr` then refuses
// every request by name and every caller falls back to the copy path through
// `be_write` and `be_read`.
constexpr bool kSynchronousExecution = true;

// Returns a command slot when a run of dispatches is left by any path other
// than a commit.
//
// `context_begin_command_buffer` hands out a slot and only
// `context_commit_and_wait` gives one back, so a return between the two would
// leave that slot open for the life of the context. Nothing about the run that
// follows would look wrong, since the next one takes a fresh slot, but
// `context_live_commands` would never fall back to zero again and every later
// reader of that count would read the leak rather than the state it asked
// about. `be_host_ptr` is such a reader, so one failed dispatch would
// otherwise refuse every host view for the rest of the process while naming
// the submit path, which had nothing to do with it.
class CommandGuard {
  public:
    CommandGuard(Context *ctx, unsigned cmd) : ctx_(ctx), cmd_(cmd) {}
    CommandGuard(const CommandGuard &) = delete;
    CommandGuard &operator=(const CommandGuard &) = delete;
    ~CommandGuard() {
        if (ctx_ != nullptr) {
            context_abandon_command(ctx_, cmd_);
        }
    }

    // Call once the slot has been returned. `context_commit_and_wait` retires
    // it whether the command buffer succeeded or failed, so this is called on
    // both of its outcomes and the destructor then has nothing to do.
    void retired() { ctx_ = nullptr; }

  private:
    Context *ctx_;
    unsigned cmd_;
};

// Encodes one recorded sequence and waits for it. Shared by submit and replay,
// so the two cannot disagree about what an op means.
BeStatus run_ops(BeBackend *be, const std::vector<Op> &ops,
                 BeDiagSummary *out_diag, BeError *err) {
    std::string detail;
    diag_reset(be->diag);
    size_t index = 0;
    while (index < ops.size()) {
        if (ops[index].is_fill) {
            const Op &op = ops[index];
            const unsigned buffer = allocator_buffer_id(be->alloc, op.dst.arena);
            if (buffer == 0) {
                return misuse(err, "be_encode_fill names arena " +
                                       std::to_string(op.dst.arena) +
                                       ", which the allocator does not hold");
            }
            if (!context_fill_buffer(be->ctx, buffer,
                                     op.dst.off + op.byte_offset,
                                     op.byte_length,
                                     static_cast<unsigned char>(op.value),
                                     &detail)) {
                be->device_lost = true;
                return platform(err, "fill: " + detail);
            }
            be->counters.fills += 1;
            be->counters.syncs += 1;
            ++index;
            continue;
        }
        // A RUN OF DISPATCHES SHARES ONE COMMAND BUFFER, which is what makes
        // this worth encoding rather than submitting one at a time: a fill
        // breaks the run because it needs a command buffer of its own.
        const unsigned cmd = context_begin_command_buffer(be->ctx);
        // Every exit between here and the commit below leaves without
        // committing: three error returns out of the encode loop, and the
        // `continue` that skips a run with nothing in it. Each one owes the
        // slot back, and the guard is what pays it.
        CommandGuard guard(be->ctx, cmd);
        allocator_bind_all(be->alloc, be->ctx, cmd);
        context_bind_buffer(be->ctx, cmd, diag_binding_index(be->diag),
                            diag_buffer_id(be->diag), 0);
        bool any = false;
        for (; index < ops.size() && !ops[index].is_fill; ++index) {
            const Op &op = ops[index];
            const unsigned pipeline = pipeline_for(be, op.kernel, &detail);
            if (pipeline == 0) {
                return platform(err, "creating the pipeline for '" +
                                         std::string(kKernels[op.kernel].name) +
                                         "': " + detail);
            }
            // THE ARENA COUNT IS PATCHED IN AT DISPATCH, not at encode: a
            // region recorded before an allocation and replayed after it must
            // see the count that is live NOW, and the generated bound check
            // inside the entry point is the only thing standing between a
            // wrong arena id and a plausible wrong answer.
            std::vector<unsigned char> record = op.args;
            if (kKernels[op.kernel].arena_count_offset != kNoArenaSlot) {
                const uint32_t live = allocator_arena_count(be->alloc);
                std::memcpy(record.data() +
                                kKernels[op.kernel].arena_count_offset,
                            &live, sizeof(live));
            }
            if (!context_set_bytes(be->ctx, cmd, 29, record.data(),
                                   record.size(), &detail)) {
                return misuse(err, "binding the argument record for '" +
                                       std::string(kKernels[op.kernel].name) +
                                       "': " + detail);
            }
            if (op.extent.kind == EXTENT_GROUPS) {
                if (!context_dispatch_threadgroups(
                        be->ctx, cmd, pipeline, op.extent.count,
                        op.extent.threads, op.extent.scratch_bytes, &detail)) {
                    return misuse(err, "dispatching '" +
                                           std::string(kKernels[op.kernel].name) +
                                           "': " + detail);
                }
            } else {
                context_dispatch(be->ctx, cmd, pipeline, op.extent.count,
                                 kElementThreads);
            }
            be->counters.dispatches += 1;
            any = true;
        }
        if (!any) {
            continue;
        }
        const bool committed = context_commit_and_wait(be->ctx, cmd, &detail);
        // The commit retires the slot on both of its outcomes, so the guard is
        // released before the failure is reported rather than after it.
        guard.retired();
        if (!committed) {
            be->device_lost = true;
            return set_error(err, STATUS_DEVICE_LOST, 0,
                             "the command buffer failed: " + detail);
        }
        be->counters.syncs += 1;
    }

    DiagReadback readback{};
    if (!diag_read(be->diag, &readback, &detail)) {
        return platform(err, "reading the diagnostic channel: " + detail);
    }
    if (out_diag) {
        std::memset(out_diag, 0, sizeof(*out_diag));
        out_diag->failures = readback.fail_count;
        out_diag->assert_hit = readback.assert_hit ? 1 : 0;
        // THE RECORD ITSELF, which this had memset to zero and never filled.
        // `assert_hit` alone says a check failed and names neither the check
        // nor its values, so every Metal device diagnostic reported
        // `file 0, line 0, [0 0 0 0]` however much the shader had written.
        // `backend.cu:349` carries the same line, and the two backends have to
        // agree here or a Metal failure cannot be read at all.
        out_diag->assert_record = readback.assert_record;
        out_diag->trace_written = readback.ring_written;
        out_diag->trace_dropped = readback.ring_dropped;
        out_diag->trace_count = static_cast<uint32_t>(readback.ring.size());
    }
    return STATUS_OK;
}

} // namespace

// ===========================================================================
// IDENTITY
// ===========================================================================

extern "C" uint32_t be_abi_version(void) { return PPF_BE_ABI_VERSION; }

extern "C" const char *be_backend_name(void) { return "metal"; }

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

extern "C" BeStatus be_open(const BeBackendConfig *config, BeBackend **out,
                            BeError *err) {
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

    std::string detail;
    auto *be = new BeBackend();
    be->log = config->log;
    be->log_context = config->log_context;
    be->poison_byte = config->poison_byte;
    be->require_deferred = config->require_deferred_regions != 0;
    be->device_lost = false;
    be->generation = 0;
    be->open_encoder = nullptr;
    be->library = 0;
    std::memset(&be->counters, 0, sizeof(be->counters));

    // The device is created here rather than lazily, so a host with no usable
    // device fails at open, by name, with nothing to fall back to.
    g_open_log = config->log;
    g_open_log_context = config->log_context;
    be->ctx = context_create(context_log_bridge, &detail);
    if (!be->ctx) {
        g_open_log = nullptr;
        g_open_log_context = nullptr;
        delete be;
        return platform(err, "creating the Metal context: " + detail);
    }
    be->alloc = allocator_create(be->ctx);
    if (!be->alloc) {
        context_destroy(be->ctx);
        delete be;
        return platform(err, "the Metal arena allocator refused to open");
    }
    be->diag = diag_create(be->ctx, be->alloc, config->diag_ring_slots, &detail);
    if (!be->diag) {
        allocator_destroy(be->alloc);
        context_destroy(be->ctx);
        delete be;
        return platform(err,
                        "the diagnostic channel could not be created: " + detail);
    }
    diag_reset(be->diag);

    // WHERE THE ENTRY LIBRARY IS, ASKED OF THE CALLER FIRST AND OF OURSELVES
    // OTHERWISE. This library loads a pre-built .metallib rather than compiling
    // source, so it has to find one. A caller that knows where the build put it
    // says so; a caller that does not gets the directory this dylib was itself
    // loaded from, which `context_prebuilt_library_dir` resolves through
    // `dladdr` on our own code.
    //
    // THE SECOND ARM IS WHAT MAKES A RELOCATED COPY WORK, and it is not a
    // convenience. The alternative is for the shipped binary to carry an
    // absolute path to the machine that built it, baked in at compile time,
    // which resolves to nothing on anyone else's machine and fails at run time
    // with every load command looking correct. `ppf_entries.metallib` sits
    // beside this dylib in both layouts the project ships, the build tree's
    // library directory and the bundle's own, so asking our own location is
    // the question with the same answer in both.
    //
    // The check moved here from be_open's argument validation because it can
    // only be answered once `be->ctx` exists.
    const char *library_dir =
        (config->library_dir && *config->library_dir)
            ? config->library_dir
            : context_prebuilt_library_dir(be->ctx);
    if (!library_dir || !*library_dir) {
        diag_destroy(be->diag);
        allocator_destroy(be->alloc);
        context_destroy(be->ctx);
        delete be;
        return misuse(err, "no directory to load the entry library from: the "
                           "caller named none and this dylib could not resolve "
                           "its own location");
    }
    const std::string path = std::string(library_dir) + "/ppf_entries.metallib";
    be->library = context_load_library(be->ctx, path.c_str(), &detail);
    if (be->library == 0) {
        diag_destroy(be->diag);
        allocator_destroy(be->alloc);
        context_destroy(be->ctx);
        delete be;
        return platform(err, "loading the generated entry library: " + detail);
    }
    be->pipelines.assign(kKernelCount, 0u);

    // The channel's own storage is an allocation, so the generation starts
    // above zero and a region recorded before any driver allocation is still
    // compared against a value that moves.
    be->generation = 1;
    g_backend = be;
    *out = be;
    emit_log(be, 0, "the Metal backend library is open, " +
                        std::to_string(kKernelCount) + " kernels from " + path);
    return STATUS_OK;
}

extern "C" void be_close(BeBackend *be) {
    if (!be) {
        return;
    }
    if (be->open_encoder) {
        be_encode_abandon(be->open_encoder);
    }
    diag_destroy(be->diag);
    allocator_destroy(be->alloc);
    context_destroy(be->ctx);
    if (g_backend == be) {
        g_backend = nullptr;
        g_open_log = nullptr;
        g_open_log_context = nullptr;
    }
    delete be;
}

extern "C" void be_info(BeBackend *be, BeDeviceInfo *out) {
    if (!out) {
        return;
    }
    std::memset(out, 0, sizeof(*out));
    if (!be) {
        return;
    }
    const char *name = context_device_name(be->ctx);
    if (name) {
        std::strncpy(out->device_name, name, sizeof(out->device_name) - 1);
    }
    out->max_arenas = PPF_BE_MAX_ARENAS;
    out->max_arena_bytes = context_max_buffer_length(be->ctx);
    out->max_threads_per_group = context_max_threads_per_threadgroup(be->ctx);
    out->max_group_scratch_bytes = context_max_threadgroup_memory(be->ctx);
    // METAL NEVER FAULTS ON AN OUT-OF-BOUNDS ACCESS: a read returns 0, a write
    // is dropped, and a write a gigabyte past the end completes with no error.
    // The driver reads this to decide whether it may rely on a trap.
    out->faults_on_oob = 0;
    out->supports_deferred_regions = 0;
}

// ===========================================================================
// THE KERNEL TABLE
// ===========================================================================

extern "C" uint32_t be_kernel_count(BeBackend *) { return kKernelCount; }

extern "C" const char *be_kernel_name(BeBackend *, uint32_t kernel_id) {
    return kernel_id < kKernelCount ? kKernels[kernel_id].name : nullptr;
}

extern "C" BeStatus be_kernel_id_by_name(BeBackend *, const char *name,
                                         uint32_t *out, BeError *err) {
    if (!name || !out) {
        return misuse(err, "be_kernel_id_by_name needs a name and a slot");
    }
    for (uint32_t id = 0; id < kKernelCount; ++id) {
        if (std::strcmp(name, kKernels[id].name) == 0) {
            *out = id;
            return STATUS_OK;
        }
    }
    return set_error(err, STATUS_NO_KERNEL, 0,
                     std::string("no kernel named '") + name +
                         "' is in this library's table");
}

extern "C" uint32_t be_kernel_args_bytes(BeBackend *, uint32_t kernel_id) {
    return kernel_id < kKernelCount ? kKernels[kernel_id].args_bytes : 0u;
}

extern "C" int32_t be_kernel_present(BeBackend *be, uint32_t kernel_id) {
    if (!be || kernel_id >= kKernelCount) {
        return 0;
    }
    // PRESENT MEANS A PIPELINE CAN BE CREATED, which on this backend is the
    // real question: the table is generated from the neutral declarations and
    // the library from the same ones, but a build that rendered an entry and
    // failed to link it would leave a row naming a function the library does
    // not carry.
    std::string detail;
    return pipeline_for(be, kernel_id, &detail) != 0 ? 1 : 0;
}

extern "C" BeStatus be_prepare_kernels(BeBackend *be, const uint32_t *ids,
                                       uint32_t count, BeError *err) {
    if (!be || (count > 0 && !ids)) {
        return misuse(err, "be_prepare_kernels needs a backend and ids");
    }
    std::string detail;
    for (uint32_t i = 0; i < count; ++i) {
        if (ids[i] >= kKernelCount) {
            return set_error(err, STATUS_NO_KERNEL, 0,
                             "kernel id " + std::to_string(ids[i]) +
                                 " is outside this library's table of " +
                                 std::to_string(kKernelCount));
        }
        if (pipeline_for(be, ids[i], &detail) == 0) {
            return platform(err, "creating the pipeline for '" +
                                     std::string(kKernels[ids[i]].name) +
                                     "': " + detail);
        }
    }
    return STATUS_OK;
}

extern "C" void be_shader_cache_report(BeBackend *be,
                                       BeShaderCacheReport *out) {
    if (!out) {
        return;
    }
    std::memset(out, 0, sizeof(*out));
    if (!be) {
        return;
    }
    ShaderCacheReport report{};
    context_shader_cache_report(be->ctx, &report);
    out->libraries = report.libraries;
    out->libraries_prebuilt = report.libraries_prebuilt;
    out->caches_loaded = report.archives_loaded;
    out->caches_started = report.archives_started;
    out->caches_written = report.archives_written;
    out->pipeline_hits = report.pipeline_hits;
    out->pipeline_misses = report.pipeline_misses;
    out->pipelines_uncached = report.pipelines_uncached;
    out->compile_ms = report.compile_ms;
    out->prebuilt_load_ms = report.prebuilt_load_ms;
    out->cache_open_ms = report.archive_open_ms;
    out->pipeline_ms = report.pipeline_ms;
    out->serialize_ms = report.serialize_ms;
}

// ===========================================================================
// MEMORY
// ===========================================================================

extern "C" BeStatus be_alloc(BeBackend *be, size_t count, size_t elem_size,
                             size_t align, const char *label, BeHandle *out,
                             BeError *err) {
    if (!be || !out) {
        return misuse(err, "be_alloc needs a backend and an output slot");
    }
    ArenaHandle handle{};
    std::string detail;
    // THE LABEL IS DROPPED, and `be_handle_label` reports that absence rather
    // than inventing a string: this allocator takes no label, and a name this
    // library made up would read as the caller's own.
    (void)label;
    if (!allocator_alloc(be->alloc, count, elem_size, align, &handle,
                         &detail)) {
        return set_error(err, STATUS_BAD_ALLOC, 0, detail);
    }
    // POISON EVERY FRESH ALLOCATION WHEN THE CALLER ASKED FOR IT. A buffer
    // accumulated into but never cleared is correct only while the memory it
    // happens to get is still zero, and fresh device memory frequently is, so
    // a run that does not poison cannot tell the two apart. `poison_byte` is
    // negative when the caller wants production behavior.
    //
    // A BLIT rather than a host memset, which is what `allocator_fill` is: the
    // fill is ordered against the dispatches that read the block by the queue,
    // and it costs no host bandwidth on a block that can be hundreds of
    // megabytes. One per allocation, paid only when the caller asked.
    if (be->poison_byte >= 0 && count > 0) {
        if (!allocator_fill(be->alloc, handle,
                            static_cast<unsigned char>(be->poison_byte),
                            count * elem_size, &detail)) {
            return platform(err, "poisoning a fresh allocation: " + detail);
        }
    }
    be->generation += 1;
    *out = from_arena(handle);
    return STATUS_OK;
}

extern "C" BeStatus be_grow(BeBackend *be, BeHandle *handle, size_t new_count,
                            size_t elem_size, size_t align, BeError *err) {
    if (!be || !handle) {
        return misuse(err, "be_grow needs a backend and a handle");
    }
    ArenaHandle h = to_arena(*handle);
    std::string detail;
    if (!allocator_grow(be->alloc, &h, new_count, elem_size, align,
                        &detail)) {
        return set_error(err, STATUS_BAD_ALLOC, 0, detail);
    }
    // A GROW MAY RELOCATE, so the generation moves whether or not it did: a
    // region recorded before this call holds handles that may now name memory
    // the allocator has handed to something else, and STATUS_STALE_REGION is
    // how a replay says so.
    be->generation += 1;
    *handle = from_arena(h);
    return STATUS_OK;
}

extern "C" BeStatus be_free(BeBackend *be, BeHandle *handle, BeError *err) {
    if (!be || !handle) {
        return misuse(err, "be_free needs a backend and a handle");
    }
    ArenaHandle h = to_arena(*handle);
    std::string detail;
    if (!allocator_free(be->alloc, &h, &detail)) {
        return set_error(err, STATUS_BAD_ALLOC, 0, detail);
    }
    be->generation += 1;
    *handle = from_arena(h);
    return STATUS_OK;
}

extern "C" const char *be_handle_label(BeBackend *, BeHandle) {
    // The Metal allocator carries a label for its own diagnostics and exposes
    // no reader for one block, so this reports absence rather than inventing a
    // string. The header allows a null return.
    return nullptr;
}

extern "C" BeStatus be_write(BeBackend *be, BeHandle handle,
                             size_t byte_offset, const void *src, size_t bytes,
                             BeError *err) {
    if (!be) {
        return misuse(err, "be_write needs a backend");
    }
    if (bytes == 0) {
        return STATUS_OK;
    }
    if (!src) {
        return misuse(err, "be_write needs a source for a non-empty write");
    }
    const BeStatus window = check_window(be->alloc, handle, byte_offset, bytes,
                                         err);
    if (window != STATUS_OK) {
        return window;
    }
    std::string detail;
    if (!allocator_write_at(be->alloc, to_arena(handle), byte_offset, src,
                            bytes, &detail)) {
        return platform(err, "write: " + detail);
    }
    be->counters.bytes_uploaded += bytes;
    return STATUS_OK;
}

extern "C" BeStatus be_read(BeBackend *be, BeHandle handle, size_t byte_offset,
                            void *dst, size_t bytes, BeError *err) {
    if (!be) {
        return misuse(err, "be_read needs a backend");
    }
    if (bytes == 0) {
        return STATUS_OK;
    }
    if (!dst) {
        return misuse(err, "be_read needs a destination for a non-empty read");
    }
    const BeStatus window = check_window(be->alloc, handle, byte_offset, bytes,
                                         err);
    if (window != STATUS_OK) {
        return window;
    }
    std::string detail;
    if (!allocator_read_at(be->alloc, to_arena(handle), byte_offset, dst, bytes,
                           &detail)) {
        return platform(err, "read: " + detail);
    }
    be->counters.bytes_downloaded += bytes;
    return STATUS_OK;
}

extern "C" BeStatus be_copy(BeBackend *be, BeHandle dst,
                            size_t dst_byte_offset, BeHandle src,
                            size_t src_byte_offset, size_t bytes,
                            BeError *err) {
    if (!be) {
        return misuse(err, "be_copy needs a backend");
    }
    if (bytes == 0) {
        return STATUS_OK;
    }
    BeStatus w = check_window(be->alloc, dst, dst_byte_offset, bytes, err);
    if (w != STATUS_OK) {
        return w;
    }
    w = check_window(be->alloc, src, src_byte_offset, bytes, err);
    if (w != STATUS_OK) {
        return w;
    }
    std::string detail;
    // NOT COUNTED as a transfer: nothing crosses a bus, and the counters this
    // backend keeps are the upload and download ones.
    if (!allocator_copy_at(be->alloc, to_arena(dst), dst_byte_offset,
                           to_arena(src), src_byte_offset, bytes, &detail)) {
        return platform(err, "copy: " + detail);
    }
    return STATUS_OK;
}

extern "C" BeStatus be_read_scalars(BeBackend *be, BeHandle handle,
                                    uint32_t first, float *dst, uint32_t count,
                                    BeError *err) {
    if (!be || (count > 0 && !dst)) {
        return misuse(err, "be_read_scalars needs a backend and a destination");
    }
    if (count == 0) {
        return STATUS_OK;
    }
    return be_read(be, handle, static_cast<size_t>(first) * sizeof(float), dst,
                   static_cast<size_t>(count) * sizeof(float), err);
}

// THE ADDRESS, WHICH THIS TARGET HAS. Arenas are allocated
// MTLResourceStorageModeShared, so the bytes a dispatch reads are already
// mapped on the host and this hands back a pointer into that mapping rather
// than a copy of it. Nothing is transferred, so neither transfer counter moves.
//
// THE WINDOW IS CHECKED THE WAY `be_read` CHECKS ITS OWN, through
// `check_window` and then again inside the allocator, so a handle naming no
// live block and a window past the end are both refused here exactly as they
// are there. That is what keeps this from being a way to manufacture an
// address the transfer calls would have rejected.
//
// THE HEADER'S FIRST CONDITION IS ANSWERED BY `kSynchronousExecution`, THE
// CONSTANT BESIDE `run_ops`, AND NOT BY ANYTHING SAMPLED HERE. The header
// requires that a backend returning a non-NULL pointer runs no device work
// past the call that started it. That is a property of the executor, and the
// change that would break it is a change to this backend's own source, so the
// refusal is keyed to a line in it. A caller asks for an address between seam
// calls, which is exactly when nothing is in flight, and it dereferences the
// address it gets long after this call has returned, so no state read at this
// point could stand in for the condition.
//
// THE TWO RUNTIME TESTS BELOW ARE CHEAP SANITY CHECKS ON POINTER ACQUISITION,
// AND THAT IS ALL THEY ARE. They cover the two states this backend can name as
// wrong at the moment an address is asked for: a region still being encoded,
// and a command buffer that has not been retired. Neither is the guarantee.
// The address outlives both readings, so a check made here says nothing about
// any later access through it, and a reader should not take a passing call as
// evidence that the accesses that follow are ordered.
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
    if (!kSynchronousExecution) {
        return misuse(err, "be_host_ptr is refused because "
                           "kSynchronousExecution in the Metal backend is "
                           "false: this backend no longer completes its device "
                           "work inside the call that started it, so an "
                           "address handed out here could be read while the "
                           "GPU is writing through it. The copy path through "
                           "be_write and be_read is what serves this target "
                           "now");
    }
    if (be->open_encoder) {
        return misuse(err, "be_host_ptr was called while a region was being "
                           "encoded; a host view is addressable only between "
                           "seam calls");
    }
    if (context_live_commands(be->ctx) != 0) {
        return misuse(err, "be_host_ptr was called with a command buffer "
                           "outstanding. Every executor here commits and waits "
                           "inside the call that creates one and returns the "
                           "slot on every path out, so the likely cause is a "
                           "submit path that no longer waits, which would also "
                           "make kSynchronousExecution wrong");
    }
    const BeStatus window = check_window(be->alloc, handle, byte_offset, bytes,
                                         err);
    if (window != STATUS_OK) {
        return window;
    }
    std::string detail;
    if (!allocator_host_ptr(be->alloc, to_arena(handle), byte_offset, bytes,
                            out, &detail)) {
        return platform(err, "host view: " + detail);
    }
    return STATUS_OK;
}

extern "C" uint64_t be_allocator_generation(BeBackend *be) {
    return be ? be->generation : 0;
}

extern "C" uint32_t be_arena_count(BeBackend *be) {
    return be ? allocator_arena_count(be->alloc) : 0;
}

extern "C" uint64_t be_bytes_used(BeBackend *be) {
    return be ? allocator_bytes_used(be->alloc) : 0;
}

extern "C" uint64_t be_bytes_reserved(BeBackend *be) {
    return be ? allocator_bytes_reserved(be->alloc) : 0;
}

// ===========================================================================
// THE ENCODER
// ===========================================================================

extern "C" BeStatus be_encode_begin(BeBackend *be, const char *region,
                                    BeEncodeMode mode, BeEncoder **out,
                                    BeError *err) {
    if (!be || !out) {
        return misuse(err, "be_encode_begin needs a backend and a slot");
    }
    if (be->open_encoder) {
        return misuse(err, "this library supports one open encoder and one is "
                           "already open");
    }
    if (be->device_lost) {
        return set_error(err, STATUS_DEVICE_LOST, 0,
                         "the device context is no longer usable");
    }
    auto *enc = new BeEncoder();
    enc->be = be;
    enc->mode = mode;
    enc->region = region ? region : "";
    be->open_encoder = enc;
    *out = enc;
    return STATUS_OK;
}

extern "C" BeStatus be_encode_dispatch(BeEncoder *enc, uint32_t kernel_id,
                                       const BeExtent *extent_in,
                                       const void *args, uint32_t args_bytes,
                                       BeError *err) {
    if (!enc || !extent_in) {
        return misuse(err, "be_encode_dispatch needs an encoder and an extent");
    }
    const BeExtent extent = *extent_in;
    if (kernel_id >= kKernelCount) {
        return set_error(err, STATUS_NO_KERNEL, 0,
                         "kernel id " + std::to_string(kernel_id) +
                             " is outside this library's table of " +
                             std::to_string(kKernelCount));
    }
    const KernelRow &row = kKernels[kernel_id];
    if (args_bytes != row.args_bytes) {
        return misuse(err, std::string("'") + row.name + "' declares an " +
                               std::to_string(row.args_bytes) +
                               " byte argument record and the caller passed " +
                               std::to_string(args_bytes));
    }
    if (args_bytes > PPF_BE_MAX_ARGS_BYTES) {
        return misuse(err, "the argument record is longer than "
                           "PPF_BE_MAX_ARGS_BYTES");
    }
    if (args_bytes > 0 && !args) {
        return misuse(err, "a non-empty argument record needs a pointer");
    }
    // THE LAUNCH SHAPE IS THE DECLARATION'S, NOT THE CALLER'S. A group entry
    // indexes by group position and an element entry by thread index, so a
    // mismatch would run the kernel over the wrong space with the right bytes.
    const bool wants_groups = row.group;
    const bool got_groups = extent.kind == EXTENT_GROUPS;
    if (wants_groups != got_groups) {
        return misuse(err, std::string("'") + row.name + "' takes the " +
                               (wants_groups ? "GROUPS" : "ELEMENTS") +
                               " extent and the caller passed the other");
    }
    if (got_groups) {
        const uint32_t limit =
            context_max_threadgroup_memory(enc->be->ctx);
        // REFUSED ABOVE THE LIMIT, NEVER CLAMPED: Metal's dynamic threadgroup
        // path validates nothing and returns wrong data past the cap on a
        // platform that does not fault.
        if (extent.scratch_bytes > limit) {
            return misuse(err, "the dispatch asks for " +
                                   std::to_string(extent.scratch_bytes) +
                                   " bytes of group scratch and this device "
                                   "allows " + std::to_string(limit));
        }
        if (row.scratch_bytes != 0 && extent.scratch_bytes != 0) {
            return misuse(err, std::string("'") + row.name +
                                   "' declares its scratch statically, so a "
                                   "dispatch must not ask for any");
        }
        if (extent.threads == 0) {
            return misuse(err, "a GROUPS extent needs a group width");
        }
    }

    Op op{};
    op.is_fill = false;
    op.kernel = kernel_id;
    op.extent = extent;
    op.args.assign(static_cast<const unsigned char *>(args),
                   static_cast<const unsigned char *>(args) + args_bytes);
    enc->ops.push_back(std::move(op));
    return STATUS_OK;
}

extern "C" BeStatus be_encode_fill(BeEncoder *enc, BeHandle dst,
                                   size_t byte_offset, uint64_t bytes,
                                   uint8_t value, BeError *err) {
    if (!enc) {
        return misuse(err, "be_encode_fill needs an encoder");
    }
    if (bytes == 0) {
        return STATUS_OK;
    }
    Op op{};
    op.is_fill = true;
    op.dst = dst;
    op.value = value;
    op.byte_offset = byte_offset;
    op.byte_length = bytes;
    enc->ops.push_back(std::move(op));
    return STATUS_OK;
}

extern "C" uint32_t be_encoder_length(BeEncoder *enc) {
    return enc ? static_cast<uint32_t>(enc->ops.size()) : 0u;
}

extern "C" BeStatus be_encode_submit(BeEncoder *enc, BeDiagSummary *out_diag,
                                     BeError *err) {
    if (!enc) {
        return misuse(err, "be_encode_submit needs an encoder");
    }
    BeBackend *be = enc->be;
    const BeStatus status = run_ops(be, enc->ops, out_diag, err);
    be->open_encoder = nullptr;
    delete enc;
    return status;
}

extern "C" BeStatus be_encode_record(BeEncoder *enc, BeRegion **out,
                                     BeError *err) {
    if (!enc || !out) {
        return misuse(err, "be_encode_record needs an encoder and a slot");
    }
    BeBackend *be = enc->be;
    if (be->require_deferred) {
        return set_error(err, STATUS_NO_DEFERRAL, 0,
                         "this library replays a region by re-encoding its "
                         "dispatches rather than deferring them, and the "
                         "caller asked for deferred regions");
    }
    auto *region = new BeRegion();
    region->generation = be->generation;
    region->dispatches = 0;
    region->fills = 0;
    for (const Op &op : enc->ops) {
        if (op.is_fill) {
            region->fills += 1;
        } else {
            region->dispatches += 1;
        }
    }
    region->ops = std::move(enc->ops);
    be->counters.regions_recorded += 1;
    // FALLBACK, NOT DEFERRED, and counted as such: the ops are held and
    // re-encoded on replay. `be_info` reports supports_deferred_regions = 0 and
    // `be_open` refuses a caller that requires them, so nothing here is a
    // surprise to the driver.
    be->counters.regions_fallback += 1;
    be->open_encoder = nullptr;
    delete enc;
    *out = region;
    return STATUS_OK;
}

extern "C" void be_encode_abandon(BeEncoder *enc) {
    if (!enc) {
        return;
    }
    BeBackend *be = enc->be;
    if (be && be->open_encoder == enc) {
        be->open_encoder = nullptr;
    }
    delete enc;
}

// ===========================================================================
// REGIONS
// ===========================================================================

extern "C" void be_region_info(BeRegion *region, BeRegionInfo *out) {
    if (!out) {
        return;
    }
    std::memset(out, 0, sizeof(*out));
    if (!region) {
        return;
    }
    out->deferred = 0;
    out->dispatch_count = region->dispatches;
    out->fill_count = region->fills;
    out->allocator_generation = region->generation;
}

extern "C" BeStatus be_replay(BeBackend *be, BeRegion *region,
                              uint32_t repeats, BeDiagSummary *out_diag,
                              BeError *err) {
    if (!be || !region) {
        return misuse(err, "be_replay needs a backend and a region");
    }
    // A RECORDED REGION HOLDS HANDLES, and a grow may relocate a block, so a
    // region recorded before an allocation change can address memory the
    // allocator has since handed to something else. The generation is what
    // catches that, and it is a refusal rather than a silent replay.
    if (region->generation != be->generation) {
        return set_error(err, STATUS_STALE_REGION, 0,
                         "the region was recorded at allocator generation " +
                             std::to_string(region->generation) +
                             " and the allocator is now at " +
                             std::to_string(be->generation));
    }
    for (uint32_t i = 0; i < repeats; ++i) {
        const BeStatus status = run_ops(be, region->ops, out_diag, err);
        if (status != STATUS_OK) {
            return status;
        }
    }
    be->counters.replays += 1;
    be->counters.replay_repeats += repeats;
    return STATUS_OK;
}

extern "C" void be_region_release(BeBackend *, BeRegion *region) {
    delete region;
}

// ===========================================================================
// DIAGNOSTICS AND COUNTERS
// ===========================================================================

extern "C" BeStatus be_diag_drain(BeBackend *be, BeDiagRecord *out,
                                 uint32_t capacity, uint32_t *written,
                                 BeError *err) {
    if (!be || !written) {
        return misuse(err, "be_diag_drain needs a backend and a count slot");
    }
    DiagReadback readback{};
    std::string detail;
    if (!diag_read(be->diag, &readback, &detail)) {
        return platform(err, "reading the diagnostic channel: " + detail);
    }
    const uint32_t n = readback.ring.size() < capacity
                           ? static_cast<uint32_t>(readback.ring.size())
                           : capacity;
    if (n > 0 && out) {
        std::memcpy(out, readback.ring.data(), n * sizeof(BeDiagRecord));
    }
    *written = n;
    return STATUS_OK;
}

extern "C" BeStatus be_diag_file_path(BeBackend *, uint32_t, char *out,
                                      size_t capacity, BeError *err) {
    if (!out || capacity == 0) {
        return misuse(err, "be_diag_file_path needs a buffer");
    }
    // The Metal diagnostic channel registers files by id for the `#line`
    // injection the assembler performs on SOURCE. This library assembles no
    // source, so it has no path to report and says so with an empty string
    // rather than a stale one.
    out[0] = '\0';
    return STATUS_OK;
}

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
