// File: backend_abi.h
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// THE BACKEND ABI, AND THE WHOLE OF IT.
//
// A backend library holds MECHANISM ONLY: device memory allocation,
// host-to-device and device-to-host transfer, free, and kernel launch, plus
// the platform machinery those four require (command encoding, queue and
// stream management, pipeline and library loading and caching, and the
// transport half of the diagnostic channel). Everything else lives above this
// header, in the Rust driver, written once.
//
// The test to apply to any symbol added below: could a library implementing it
// be compiled with no knowledge of what a Newton step is, what a barrier is, or
// what the scene contains? If not, it does not belong here. Every function in
// this file passes that test, and several of the near misses are called out at
// the site where the line was drawn.
//
// Each backend compiles this header, implements every symbol it declares, and
// links as its own external library (`libppf_backend_cuda.so`,
// `libppf_backend_metal.dylib`, `libppf_backend_cpu.{so,dylib,dll}`). The
// separation is what makes the independence checkable rather than merely
// intended: a library that links on its own cannot have reached into solver
// logic, because that logic is not in its compilation unit to reach.
//
// WHERE THIS HEADER SITS IN THE OWNERSHIP RULE. A BACKEND file may define no
// data type that crosses the seam. This file is not a backend file. It is the
// seam's own declaration, owned above the seam, compiled by every backend and
// declared once on the Rust side, so the types below are the single declaration
// that rule asks for rather than an exception to it. Two mechanisms verify that
// claim:
// `be_layout_probe` reports the sizes the LOADED library was compiled
// against, so a stale library fails by name instead of by SIGBUS, and the
// argument records that ride `be_encode_dispatch` are opaque bytes here and
// are generated per target by `ppf-cts-compute/seam/kernelgen.py` from one declaration.
//
// C-linkage declarations, compiled as C99 or later and as C++11 or later. No
// callback into the caller except the log function in `BeBackendConfig`, no
// ownership transfer in either direction, no allocation on the caller's
// behalf, and no exception or panic may cross any function below.

#ifndef BACKEND_ABI_H
#define BACKEND_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===========================================================================
// VERSION AND LAYOUT
// ===========================================================================

// Bumped on any change to a declaration below. The driver checks it against
// `be_abi_version()` before it calls anything else, because the tree has a
// measured history of a stale backend library surfacing as `Bus error: 10`
// rather than as a named failure.
#define PPF_BE_ABI_VERSION 2u

// The portable per-argument size cap, and the size above which
// `be_encode_dispatch` refuses a record. Not Metal's own 32752 B device
// cap: 32756 B kills the process with SIGABRT and nothing on stdout or stderr,
// so the margin is taken well short of the cliff, and 4096 B also keeps a
// record inside CUDA's kernel parameter space. `ppf-cts-compute/seam/kernelgen.py` refuses
// a larger record at generation time, so this is a second reading of the same
// bound rather than the only one.
#define PPF_BE_MAX_ARGS_BYTES 4096u

// Arena slots. The bound is the Metal buffer binding budget, which is
// compiler-enforced at 31 slots: arenas take 0 through 28, the argument record
// takes 29 and the diagnostic buffer takes 30. CUDA and the CPU backend have no
// such limit and honor the same number so a handle means one thing everywhere.
#define PPF_BE_MAX_ARENAS 29u

// The longest message `BeError` carries, including the terminating NUL.
#define PPF_BE_ERROR_DETAIL 1024u

// THREADING. A handle this library returns is owned by the PROCESS, not by the
// thread that obtained it: a driver may call any function below from a
// different thread than the one that called `be_open`, provided it serializes
// so that no two calls are ever inside the library at once. It is a
// one-at-a-time contract and NOT a concurrency one, so a library may hold state
// with no lock of its own, and a driver may not call two functions in parallel.
//
// The requirement is not decorative. The Rust seam keeps its device in a
// process-wide mutex, which the language requires to be `Send`, so a library
// that pinned its context to the opening thread would be a use-after-move that
// no signature expresses. CUDA's runtime API and Metal's device objects both
// already meet it.

// The ABI version the loaded library was compiled against. Callable before
// `be_open`, and the only function that is.
uint32_t be_abi_version(void);

// A stable identifier for this backend: "cuda", "metal" or "cpu". For reporting
// and for a gate that asserts which library it loaded. Callable before
// `be_open`. The returned pointer is a string literal in the library and
// lives for the life of the process.
const char *be_backend_name(void);

// Sizes and alignments as the LOADED library was compiled, so a driver built
// against a different revision of this header fails by name and names the field
// that moved. Callable before `be_open`.
//
// Every field is filled by the library from its own `sizeof` and `alignof`. The
// driver compares against its own and reports the first disagreement. A
// mismatch is never recoverable and must not be tolerated: a wrong `BeHandle`
// size is a wrong arena id, and a wrong arena id is a silent wrong answer with
// plausible floats on a backend that never faults.
typedef struct BeLayoutProbe {
    uint32_t abi_version;
    uint32_t handle_size;
    uint32_t handle_align;
    uint32_t diag_record_size;
    uint32_t diag_record_align;
    uint32_t extent_size;
    uint32_t error_size;
    uint32_t diag_summary_size;
    uint32_t counters_size;
    uint32_t device_info_size;
    uint32_t config_size;
    uint32_t region_info_size;
    uint32_t shader_cache_report_size;
    uint32_t max_args_bytes;
    uint32_t max_arenas;
    uint32_t reserved[3];
} BeLayoutProbe;

void be_layout_probe(BeLayoutProbe *out);

// ===========================================================================
// STATUS AND ERRORS
// ===========================================================================

// Every fallible call returns one of these. A backend returns no other value
// and raises nothing: an exception or a panic crossing this boundary is
// undefined, so a backend written in C++ catches at its own outermost frame and
// a backend written in Rust catches its unwind before returning.
typedef enum BeStatus {
    STATUS_OK = 0,

    // The call was well formed and the platform refused it: out of device
    // memory, a buffer longer than the device accepts, a pipeline that could
    // not be created. `BeError::detail` names the platform call.
    STATUS_PLATFORM = 1,

    // The caller violated a precondition stated in this header: a NULL out
    // pointer, an encoder submitted twice, a record longer than
    // PPF_BE_MAX_ARGS_BYTES, an extent whose kind the kernel does not accept, a
    // second `be_open` on a library that supports one instance. A driver
    // defect, never a scene's fault, and never recoverable.
    STATUS_MISUSE = 2,

    // The kernel id is outside the table this library carries, or the library
    // carries the id and has no implementation for it. The DRIVER decides what
    // to refuse and by what name; this only reports the absence.
    STATUS_NO_KERNEL = 3,

    // An allocation request the allocator refuses on its own terms: an
    // alignment below 4 bytes (Metal silently floors a misaligned setBuffer
    // offset to a multiple of 4 and returns the wrong data with status
    // Completed), an alignment that does not divide the element size, a request
    // exceeding what a single arena's 32-bit offset can address, or a request
    // needing a thirtieth arena.
    STATUS_BAD_ALLOC = 4,

    // A recorded region names allocations that have moved or been freed since
    // it was recorded. `grow` may relocate a block, so a recorded argument
    // record can hold a stale handle, and replaying it would address memory the
    // allocator has since handed to something else.
    STATUS_STALE_REGION = 5,

    // The platform aborted the work and the device context is no longer usable:
    // a CUDA sticky error, a Metal command buffer whose status or error reports
    // a fault, a trapped live release assert. NOT the same thing as the
    // diagnostic channel reporting failing threads, which is
    // `STATUS_OK` with a non-zero `BeDiagSummary::failures`. See the
    // diagnostic section for why that distinction is the whole answer to
    // three fault models under one driver.
    STATUS_DEVICE_FAULT = 6,

    // A previous call returned STATUS_DEVICE_FAULT and the context has not
    // been reopened. Every call except `be_close`, `be_counters` and
    // the diagnostic drain answers this from then on, so a driver that ignores
    // the first fault cannot accidentally proceed on stale device data.
    STATUS_DEVICE_LOST = 7,

    // The driver asked for a deferred region (`require_deferred_regions`) and
    // this backend could not record one. Returned from `be_encode_record`,
    // at record time rather than after a run, so a gate sees it before any
    // measurement is taken. With the flag clear the same condition succeeds,
    // reports `BeRegionInfo::deferred == 0`, and counts in
    // `BeCounters::regions_fallback`.
    STATUS_NO_DEFERRAL = 8
} BeStatus;

// Filled on any non-OK return. Caller-owned storage, fixed size, never
// allocated by the library, so there is no ownership question and no failure
// path inside a failure path.
//
// `detail` is always NUL-terminated. `truncated` is 1 when the message did not
// fit, which matters because a device diagnostic message carries a source quote
// and can be long.
typedef struct BeError {
    int32_t status;
    int32_t truncated;
    // A platform-specific numeric code when one exists (a `cudaError_t`, an
    // `MTLCommandBufferError` code), else 0. For reporting only. A driver that
    // branched on it would be reading platform policy back through the seam.
    int64_t platform_code;
    char detail[PPF_BE_ERROR_DETAIL];
} BeError;

// ===========================================================================
// THE SEAM'S DATA VOCABULARY
//
// Four types cross this boundary alongside kernel work, and they are declared
// here, once: a memory handle, a diagnostic record, the summary a boundary
// reports, and a launch extent. Nothing else does. An ARGUMENT RECORD is
// opaque bytes here, and its layout is generated for each target from one
// declaration by `ppf-cts-compute/seam/kernelgen.py`, which is what keeps 73 hand-written
// mirror pairs from reappearing at this boundary instead of inside a backend.
// ===========================================================================

// A device allocation, addressed as (arena id, byte offset) rather than as a
// pointer.
//
// A pointer cannot serve, and the reason is measured rather than stylistic:
// `DataSet` carries 76 transitive containers and 90 transitive device pointers,
// the contact assembly needs about 60 buffers even maximally flattened, and
// Metal offers 31 compiler-enforced buffer binding slots, so binding one buffer
// per container is impossible rather than awkward. Argument-buffer Tier 2 and
// `MTLBuffer.gpuAddress` both exist on the measured devices and neither relaxes
// the signature limit, and a raw device address is a bare pointer with no bound
// and no provenance on a backend that never faults.
//
// `allocated` is load-bearing and not redundant with `size`: a caller that
// wants spare capacity allocates the capacity and lowers `size` itself.
typedef struct BeHandle {
    uint32_t arena;
    uint32_t off;
    uint32_t size;
    uint32_t allocated;
} BeHandle;

// A handle naming nothing. Distinct from a zero-LENGTH handle, which names a
// real bound arena so a bounds check on it fails loudly rather than indexing an
// unbound slot. Zero is not usable as the sentinel because arena 0 is a real
// arena and a value-initialized handle would resolve silently to its first
// byte.
#define HANDLE_NONE_ARENA 0xFFFFFFFFu

// One device diagnostic record. Fixed 32-byte cross-backend layout, shared by
// the CUDA, Metal and CPU channels.
//
// Indices travel as exact uint bit patterns in the float payload slots, so a
// record does not round an index above 2^24. `file_id` is resolved through
// `be_diag_file_path`, never inside the library: `__FILE__` is a string
// literal and cannot live in a device buffer, so the file becomes an integer
// whose meaning is a property of whichever mechanism produced the source (a
// `#line` injection on Metal, a constexpr FNV-1a hash on CUDA).
typedef struct BeDiagRecord {
    uint32_t assert_id;
    uint32_t file_id;
    uint32_t line;
    uint32_t thread_id;
    float payload[4];
} BeDiagRecord;

// Reserved diagnostic ids. A trace site picks its own id at or above
// PPF_BE_DIAG_ID_FIRST_USER so it cannot be confused with a violation.
#define PPF_BE_DIAG_ID_ASSERT 1u
#define PPF_BE_DIAG_ID_BOUNDS 2u
#define PPF_BE_DIAG_ID_FIRST_USER 16u

// What a boundary reports back through the diagnostic channel.
//
// This is a REPORT, not a verdict. See the diagnostic section below for
// where the line between the two falls and why it falls there.
typedef struct BeDiagSummary {
    // Threads that failed an assert or a bounds check across everything this
    // boundary executed. A count alone does not locate the check, which is what
    // the records are for.
    uint64_t failures;
    // 1 when the single-record assert slot fired. The first claimant wins by
    // compare-and-swap; every other failing thread only increments the counter.
    int32_t assert_hit;
    int32_t reserved;
    BeDiagRecord assert_record;
    // Trace records written and dropped. The ring hands out a distinct slot per
    // writer and drops the overflow rather than wrapping.
    uint64_t trace_written;
    uint64_t trace_dropped;
    // How many records `be_diag_drain` will return.
    uint32_t trace_count;
    uint32_t reserved2;
} BeDiagSummary;

// How much work one dispatch covers, and the only launch geometry a call site
// supplies.
typedef enum BeExtentKind {
    // One thread per element. `count` is the thread count; the backend picks
    // the group width, because that choice computes no value. The generated
    // entry point guards the index against a count carried in the argument
    // record, which the backend cannot see and does not check.
    EXTENT_ELEMENTS = 0,

    // The group count, the group width and the group-local scratch are all
    // semantic, so the driver states them. A kernel that cooperates over one
    // problem per group indexes by group position, and a ceil-div over a thread
    // count cannot express "one group per aggregate" when the group size is
    // chosen for occupancy rather than for the problem.
    EXTENT_GROUPS = 1
} BeExtentKind;

typedef struct BeExtent {
    // A BeExtentKind.
    uint32_t kind;
    // ELEMENTS: threads. GROUPS: groups.
    uint32_t count;
    // GROUPS only. Ignored for ELEMENTS.
    uint32_t threads;
    // GROUPS only, dynamic group-local scratch.
    //
    // REFUSED above the device limit, never clamped. Metal's static threadgroup
    // arrays are capped at exactly 32768 B and enforced at pipeline creation
    // while the dynamic path validates nothing and silently returns wrong data
    // past the cap, so a clamp would run a kernel against memory it then
    // indexes past on a platform that does not fault. The limit to check
    // against is `BeDeviceInfo::max_group_scratch_bytes`, which is the
    // DOCUMENTED figure and not an empirical ceiling.
    uint32_t scratch_bytes;
} BeExtent;

// TWO OTHER HEADERS NAME THE TWO TYPES ABOVE RATHER THAN REDECLARING THEM, so
// every consumer keeps its own spelling. `kernels/arena_handle.hpp` includes
// this file and adds `typedef BeHandle ArenaHandle;`, and
// `kernels/diagnostic_record.hpp` includes it and defines its three
// `constexpr uint32_t DIAG_ID_*` names from the macros above. Either one
// defining its type again would be a second declaration of a seam type, which
// is what this header exists to prevent, and a translation unit including both
// would not even compile.

// ===========================================================================
// LIFECYCLE
// ===========================================================================

typedef struct BeBackend BeBackend;

// Where the library's non-fatal messages go.
//
// Required rather than optional, because the obvious default is a defect:
// anything this process writes to stderr lands in the session's error.log, and
// the frontend reports a run with a non-empty error.log as a failure, so a
// message like "the pipeline archive was stale, I recompiled" would turn a slow
// start into a reported crash.
//
// The callback must not unwind and must not call back into the library. `level`
// follows the driver's own scale and is passed through untouched; the library
// chooses only whether a message is worth emitting, which is reporting rather
// than policy.
typedef void (*BeLogFn)(void *context, int32_t level, const char *message);

typedef struct BeBackendConfig {
    // Must equal PPF_BE_ABI_VERSION. Checked first, before anything else in
    // this struct is read.
    uint32_t abi_version;

    // 0 to 255 fills every fresh allocation with that byte; a negative value
    // leaves fresh bytes UNSPECIFIED, which is the production setting.
    //
    // THE DEFAULT IS NOT ZERO AND MUST NOT BE. A buffer that is accumulated
    // into but never cleared is correct only while the memory it happens to get
    // is still zero, and fresh device memory frequently is, so zeroing here
    // hides exactly the defect it looks like it prevents. The clear belongs to
    // the pass that OPENS an assembly, spelled by the driver as
    // `be_encode_fill`, unconditionally: a clear placed behind a condition
    // is skipped on exactly the scene that has no other writer.
    //
    // 0x5a is the value the gates use. It is loud under every reading of the
    // bytes: as a float 1.9e16, as an index 1515870810, recognizable in a dump.
    // 0xFF is wrong because a NaN propagates and hides where it entered, and 0
    // is wrong because 0 is what the defect already reads.
    //
    // This is a driver option rather than an environment variable read inside
    // the library, so a run states its own poisoning rather than inheriting it,
    // and so one gate can drive all three backends the same way.
    int32_t poison_byte;

    // Non-zero makes `be_encode_record` return STATUS_NO_DEFERRAL
    // rather than fall back to an ordinary dispatch list.
    //
    // THE POINT IS THAT A FALLBACK IS NEVER SILENT. CUDA already catches any
    // graph capture failure, latches capture off, warns once behind a static
    // flag and replays by direct launch, which produces BIT-IDENTICAL numbers.
    // The only observable effect is a slowdown, and a slowdown is inside this
    // project's own measured run-to-run envelope, so every value gate stays
    // green while a mechanism the design depends on has stopped working. A
    // performance gate sets this flag and gets a named failure at record time
    // instead of a number that drifts.
    int32_t require_deferred_regions;

    // Records the diagnostic ring can hold before it starts DROPPING, never
    // wrapping. A wrapping ring tears, measured at 1 record in 1024 on Metal
    // and 512 in 1024 on CUDA, and the low Metal rate is the more dangerous
    // outcome rather than the safer one.
    uint32_t diag_ring_slots;

    // Where a compiled-pipeline cache may live, or NULL to disable caching.
    // Driver-supplied so the library reads no environment variable to find it.
    // Ignored by a backend with no such cache.
    const char *pipeline_cache_dir;

    // Where pre-compiled kernel artifacts are read from, or NULL to compile at
    // open. Driver-supplied for the same reason. Pre-compiled and cached rather
    // than JIT is condition 4 of the plan, so a NULL here is a development
    // setting and a shipped build always names a directory.
    const char *library_dir;

    BeLogFn log;
    void *log_context;
} BeBackendConfig;

// Opens the device, brings up the allocator and the diagnostic channel, and
// loads or compiles the kernel library. Does not create pipelines: that is
// `be_prepare_kernels`, because pipeline creation is the larger of the two
// startup costs (16.2 s for 129 pipelines against 6.4 s to compile the source
// on an Apple M1) and the driver states the set it needs.
//
// A host with no usable device fails here, by name, with no fallback. A device
// that cannot run this library's kernel image fails here too: on CUDA that is
// the empty probe kernel and its three error codes, which is a question about
// the IMAGE and not about the scene, so it is mechanism.
//
// A library that supports only one live instance refuses a second call with
// STATUS_MISUSE rather than returning a second handle onto shared state.
BeStatus be_open(const BeBackendConfig *config, BeBackend **out,
                      BeError *err);

// Releases every arena, region and pipeline. Safe on a device-lost context, and
// safe on NULL.
void be_close(BeBackend *be);

// ===========================================================================
// IDENTITY AND LIMITS
// ===========================================================================

// Facts about the device. Reporting and limits only.
//
// A DRIVER THAT BRANCHED ON ONE OF THESE would be a backend owning policy
// through the back door, which is the one failure a boundary cannot prevent by
// itself. `faults_on_oob` in particular is 0 on Metal and is not a licence to
// skip a bounds check anywhere: it exists so a report can name the fault model
// that produced its verdict. The mechanical guard is a wiring rule rather than
// this comment: no backend name and no backend-conditional compilation anywhere
// in the driver.
typedef struct BeDeviceInfo {
    char device_name[128];
    uint32_t max_arenas;
    uint64_t max_arena_bytes;
    uint32_t max_threads_per_group;
    // The DOCUMENTED per-group limit, never an empirical ceiling. 65536 B of
    // dynamic group storage is correct on an M4 Pro and returns wrong values on
    // an M1 at 64 concurrent groups, so the permissive figure is a property of
    // one device.
    uint32_t max_group_scratch_bytes;
    // 1 where an out-of-bounds device access faults, 0 where it does not. Metal
    // is 0: an OOB read returns 0.0, an OOB write is dropped, a write 1 GiB
    // past the end completes with no error, and bounds are checked against the
    // ALLOCATION rather than the logical length.
    int32_t faults_on_oob;
    // 1 where this backend can record a replayable region in a deferred form.
    int32_t supports_deferred_regions;
} BeDeviceInfo;

void be_info(BeBackend *be, BeDeviceInfo *out);

// ===========================================================================
// THE KERNEL TABLE
//
// Kernel entry points and their argument records are GENERATED, per target,
// from one declaration. The table below is generated with them and compiled
// into this library, so the mapping from a dense id to a pipeline or a function
// pointer is machinery rather than a hand-kept list.
//
// A kernel must have a NAME the driver can pass, and CUDA's kernels do not have
// one today: its dispatch macro captures an extended device lambda at the call
// site, so 132 of its dispatch sites are anonymous closures with no address.
// Generating the entry-point layer is what makes an id possible at all, and it
// is the half of this design that has to land first.
// ===========================================================================

// How many ids exist. Ids are dense in [0, count).
uint32_t be_kernel_count(BeBackend *be);

// The entry point's name, identical across every rendering of the same
// declaration, or NULL for an id outside the table. The returned pointer lives
// for the life of the backend.
const char *be_kernel_name(BeBackend *be, uint32_t kernel_id);

// The id this library gives a name, so a driver can assert at startup that its
// own generated ids agree with the loaded library's. Both tables are generated,
// and both can be generated from different trees.
BeStatus be_kernel_id_by_name(BeBackend *be, const char *name,
                                   uint32_t *out, BeError *err);

// `sizeof` the argument record this library was compiled against, or 0 for an
// id outside the table. The run-time half of the layout cross-check: the
// compile-time half is the generated `size_of` assertion on each side.
uint32_t be_kernel_args_bytes(BeBackend *be, uint32_t kernel_id);

// 1 when this library carries an implementation, 0 when it does not.
//
// THE DRIVER MAPS ABSENCE TO A NAMED REFUSAL AND REFUSES THE SCENE at
// initialize, by name and count. The library never decides what to refuse and
// never substitutes anything for a kernel it lacks.
int32_t be_kernel_present(BeBackend *be, uint32_t kernel_id);

// Creates every pipeline these ids need, now.
//
// The driver states the set, before a frame is written; the library does the
// work. An id that cannot be supplied fails HERE by name, rather than mid-step
// or by quietly substituting something else.
BeStatus be_prepare_kernels(BeBackend *be, const uint32_t *ids,
                                 uint32_t count, BeError *err);

// What the pipeline and library caches did this run. Data rather than a log
// line, because the library depends on nothing but its platform and the driver
// owns every message.
typedef struct BeShaderCacheReport {
    uint32_t libraries;
    uint32_t libraries_prebuilt;
    uint32_t caches_loaded;
    uint32_t caches_started;
    uint32_t caches_written;
    uint32_t pipeline_hits;
    uint32_t pipeline_misses;
    uint32_t pipelines_uncached;
    double compile_ms;
    double prebuilt_load_ms;
    double cache_open_ms;
    double pipeline_ms;
    double serialize_ms;
} BeShaderCacheReport;

void be_shader_cache_report(BeBackend *be, BeShaderCacheReport *out);

// ===========================================================================
// MEMORY
// ===========================================================================

// Allocates `count` elements of `elem_size` bytes at `align`.
//
// `label` is a stable name for the allocation, used in this library's own error
// text and readable back through `be_handle_label`. The caller guarantees
// it outlives the backend, so nothing is copied and nothing is owned. Handles
// differ between backends because arena packing does; labels do not, which is
// what lets an acceptance trace taken on two backends be compared byte for byte
// once each handle field is replaced by its label.
//
// ASSERTS NATURAL ALIGNMENT AT ALLOCATION TIME, which is the earliest point the
// question can be asked and the only point at which it can be answered loudly:
// Metal silently floors a misaligned buffer offset to a multiple of 4 and
// returns the wrong data with status Completed and no diagnostic. `align` must
// be a power of two, at least 4, and must divide `elem_size`, with one
// sanctioned exception: an element smaller than the alignment needs at most its
// own size, so any offset inside a block that starts aligned satisfies it.
//
// A count of 0 yields a real zero-length handle naming a bound arena, never a
// sentinel, so a bounds check on it fails loudly instead of indexing an unbound
// slot.
//
// FRESH BYTES ARE UNSPECIFIED unless `BeBackendConfig::poison_byte` says
// otherwise. See that field for why zeroing is forbidden as a default.
BeStatus be_alloc(BeBackend *be, size_t count, size_t elem_size,
                       size_t align, const char *label, BeHandle *out,
                       BeError *err);

// Grows an allocation, preserving its contents.
//
// THE BLOCK MAY MOVE, so every copy of the handle held elsewhere is stale after
// this call, a recorded region included. That is what
// `be_allocator_generation` exists to catch. `elem_size` and `align` must
// match the original allocation and are checked; a `new_count` below the
// current capacity is refused rather than silently shrinking.
BeStatus be_grow(BeBackend *be, BeHandle *handle, size_t new_count,
                      size_t elem_size, size_t align, BeError *err);

// Frees the block and zeroes the handle, so a use after free addresses arena 0
// with `allocated` 0 and every bounds check rejects it.
BeStatus be_free(BeBackend *be, BeHandle *handle, BeError *err);

// The label an allocation was given, or NULL for a handle naming no live block.
const char *be_handle_label(BeBackend *be, BeHandle handle);

// Host to device. `byte_offset` is measured from the start of the block, so a
// caller may write a window of a block that carries several arrays end to end.
// Both ends of the window are checked against the block's capacity.
//
// A handle whose `off` has been advanced by hand names no live block and is
// refused, which is the correct answer to a handle that has been invented.
BeStatus be_write(BeBackend *be, BeHandle handle, size_t byte_offset,
                       const void *src, size_t bytes, BeError *err);

// Device to host, same window rule.
BeStatus be_read(BeBackend *be, BeHandle handle, size_t byte_offset,
                      void *dst, size_t bytes, BeError *err);

// Device to device, both windows checked as `be_write` and `be_read` check
// theirs. The two blocks may be the same one, in which case the windows must
// not overlap; a caller that needs an overlapping move wants two calls.
//
// WITHOUT THIS, A CALLER PAYS A DOWNLOAD AND AN UPLOAD TO DO NO WORK. The Rust
// side implemented `copy` as `read` into a host bounce followed by `write`,
// which is correct and the wrong shape, and it was the largest device-to-host
// row this tree measured: the per-Newton-step matrix snapshot on `drape`
// moved 17,510,760 bytes each way, 84 times in 12 frames.
BeStatus be_copy(BeBackend *be, BeHandle dst, size_t dst_byte_offset,
                      BeHandle src, size_t src_byte_offset, size_t bytes,
                      BeError *err);

// The hot small readback: a few floats read between replays of a region.
//
// Separate from `be_read` because the library keeps a pinned staging buffer
// for it, and on CUDA that is worth about 20x: a pageable destination degrades
// a small device-to-host copy to a roughly 100 microsecond blocking staged copy
// against about 5 microseconds for a direct DMA. `first` is an element index,
// not a byte offset.
BeStatus be_read_scalars(BeBackend *be, BeHandle handle, uint32_t first,
                              float *dst, uint32_t count, BeError *err);

// The host address of a window inside a live block, or NULL where this
// backend's device memory is not host-addressable.
//
// A backend whose allocations the host can address returns a pointer to byte
// `byte_offset` of the block `handle` names, valid for `bytes` bytes. A
// backend whose memory the host cannot address returns STATUS_OK with `*out`
// NULL, which is a fact about the platform and not a failure: the caller's
// copy path through `be_write` and `be_read` is what serves such a target, and
// it is what every target served before this call existed.
//
// THE WINDOW IS CHECKED WHETHER OR NOT A POINTER COMES BACK, exactly as
// `be_read` checks its own: a handle whose `off` has been advanced by hand
// names no live block and is refused, and a window running past the capacity
// the allocator recorded is refused, on a backend that answers NULL as well as
// on one that answers an address. That is what keeps this from being a way to
// manufacture an address the transfer calls would have rejected.
//
// THE CHECK IS UNCONDITIONAL BECAUSE THE ALTERNATIVE MAKES THE REFUSAL A
// PROPERTY OF THE TARGET. A library that validated only when it had a
// pointer to hand back would answer a driver that invented a handle with
// STATUS_OK on CUDA and STATUS_MISUSE on Metal, so a driver defect would be
// reported only where the memory happens to be mapped, and the implementation
// every other target is checked against would be the permissive one. What it
// costs a library that serves no view is one capacity lookup and one bounds
// test per call, and this call is made to acquire an address rather than to
// move an element.
//
// A `bytes` of 0 returns STATUS_OK with `*out` NULL on every backend, ahead of
// all of that: a zero-length window addresses nothing, so there is nothing to
// bound.
//
// TWO CONDITIONS BIND ANY BACKEND THAT RETURNS A NON-NULL POINTER, and both
// are requirements on the LIBRARY rather than advice to the caller.
//
// FIRST, EXECUTION MUST BE SYNCHRONOUS. Every entry point that runs device
// work must have completed that work before it returns, so that no device work
// is ever in flight at a point where caller code runs. That is what makes a
// host access to the returned address safe without a copy, and it is what the
// copy never provided: a memcpy through a shared allocation carries no
// ordering of its own. A library that made submission asynchronous must stop
// returning a pointer here, and must refuse loudly rather than quietly, since
// a torn read of a plausible float is what a caller would otherwise get.
//
// THAT REQUIREMENT IS KEYED TO A COMPILE-TIME CONSTANT BESIDE THE CODE THAT
// WOULD HAVE TO CHANGE, AND NOT TO ANYTHING SAMPLED AT RUN TIME. The Metal
// library declares `kSynchronousExecution` next to its executor, refuses every
// view by name while that constant is false, and carries the note at the wait
// itself saying that removing the wait means flipping it. So the edit that
// would make a view unsafe cannot be made without passing the declaration that
// says it is safe, which is the only place the question is decidable.
//
// A RUN-TIME COUNT OF DEVICE WORK IN FLIGHT DOES NOT DECIDE IT, and reading a
// library's own refusals as though it did is the mistake this paragraph exists
// to prevent. A caller takes the address once, when it sizes its buffer, and
// dereferences it at every later access without calling the library again, so
// there is no call at which such a count could be consulted for the access it
// would be about. Sizing happens between submits, where the count is zero in a
// library that waits and would be zero in one that did not. What the Metal
// library's refusals for an open encoder and for an outstanding command buffer
// cover is therefore POINTER ACQUISITION and nothing after it: a view taken
// while a region is being encoded, and a view taken while that library holds
// work it believes it has not retired. They are worth having and they are not
// the guarantee.
//
// SECOND, THE ADDRESS IS INVALIDATED BY `be_grow` OR `be_free` ON THIS HANDLE
// AND BY NOTHING ELSE. An allocation, a growth or a free on any OTHER handle
// must not relocate this block. A caller that re-takes the address whenever it
// grows or frees its own block therefore holds a valid one, and needs no
// generation check on the hot path.
//
// ORDERING WITHIN THE CALLER'S OWN ACCESSES IS THE CALLER'S, on exactly the
// terms `be_write` and `be_read` state it. This returns an address. It
// performs no synchronization and it does not make one unnecessary.
BeStatus be_host_ptr(BeBackend *be, BeHandle handle, size_t byte_offset,
                          size_t bytes, void **out, BeError *err);

// Bumped by every alloc, grow and free. A recorded region carries the value it
// was recorded at and `be_replay` refuses an older one.
uint64_t be_allocator_generation(BeBackend *be);

uint32_t be_arena_count(BeBackend *be);
uint64_t be_bytes_used(BeBackend *be);
uint64_t be_bytes_reserved(BeBackend *be);

// ===========================================================================
// EXECUTION
//
// Execution is IMMEDIATE, with exactly one deferred construct, and that split
// is a measured result rather than a preference.
//
// CUDA outside its linear solve is already synchronous: its dispatch macro ends
// every non-queue site with a stream synchronize, and 122 of the 128 macro
// dispatch sites reachable from one step take that form. So an immediate
// primitive costs CUDA nothing across essentially the whole solver, and the
// hard problem collapses to ONE PLACE. There, CUDA issues five launches per
// iteration captured into one graph launch and reads the residual only every
// fourth iteration, which is 0.25 host round trips per iteration far from
// tolerance and 1.0 near it, while Metal pays six command-buffer round trips
// per iteration for the same loop, 24x as many.
//
// Hence: `be_encode_record` plus `be_replay`, and nothing else defers.
// A recorded region is an inert list. It hands back no control flow: the trip
// count, the observation cadence and the convergence predicate are all driver
// code between calls. A `solve()` entry point here would hand a library the
// recurrence, the stopping rule and the breakdown classification, which are
// precisely the three things whose divergence produced the defect this design
// exists to make unrepresentable.
// ===========================================================================

typedef struct BeEncoder BeEncoder;
typedef struct BeRegion BeRegion;

typedef enum BeEncodeMode {
    // Encode and execute once. Closed by `be_encode_submit`.
    ENCODE_IMMEDIATE = 0,
    // Encode without executing, into a replayable region. Closed by
    // `be_encode_record`.
    //
    // The mode is stated at OPEN rather than at close, because CUDA must begin
    // stream capture before the first launch is issued, so a library cannot
    // decide after the fact which of the two it was building.
    ENCODE_RECORD = 1
} BeEncodeMode;

// Opens an encoder. `region` is a stable label carried into any error this
// encoder produces, so a fault names the phase the driver had opened. The
// caller guarantees it outlives the encoder.
//
// One encoder is open at a time per backend. A second call while one is open is
// STATUS_MISUSE.
//
// AN ENCODER CANNOT READ AND CANNOT ALLOCATE, and both omissions are
// load-bearing rather than tidy. A host read inside a region would be a host
// dependency no platform can express inside a captured graph or a committed
// command buffer, and an allocation inside one is what CUDA's capture forbids.
// Neither is expressible through this ABI, so neither needs to be checked.
BeStatus be_encode_begin(BeBackend *be, const char *region,
                              BeEncodeMode mode, BeEncoder **out,
                              BeError *err);

// Appends one dispatch.
//
// `args_bytes` must equal `be_kernel_args_bytes(be, kernel_id)`, which is
// the ONLY check the library can make on the record: its fields are opaque
// here, and the guard bound the entry point tests the thread index against
// lives inside the record where the library cannot see it. The extent's kind
// must match the shape the kernel was generated for.
//
// The bytes are COPIED at this call, so the caller's buffer may be reused
// immediately, and a recorded region owns its own copy.
//
// Consecutive entries are ordered with a full memory barrier between them, on
// every backend: same-stream launch ordering on CUDA, a compute encoder's
// default serial dispatch type on Metal, and sequence on the CPU. There is
// therefore no barrier primitive and nothing for a library to decide.
// Concurrency is deliberately not expressible: reordering is the one freedom
// that could move fp32 association order with no statement from the driver.
BeStatus be_encode_dispatch(BeEncoder *enc, uint32_t kernel_id,
                                 const BeExtent *extent, const void *args,
                                 uint32_t args_bytes, BeError *err);

// Appends a device fill, ordered against the dispatches around it.
//
// On the encoder rather than as a standalone memset for one reason: a buffer
// whose consumers read a range its producers are not guaranteed to write is
// cleared by the pass that OPENS the assembly, unconditionally, and putting the
// fill in the same ordered stream as that pass is what makes "before the first
// assembly of the round" a property of the encoding rather than of a comment. A
// standalone fill is an immediate encoder carrying one fill.
//
// Implementation note that is easy to get wrong on Metal: a fill helper that
// takes its own command buffer and waits would be a hidden host round trip
// inside a region. The fill must end the compute encoder, run a blit on the
// same command buffer, and reopen a compute encoder.
BeStatus be_encode_fill(BeEncoder *enc, BeHandle dst,
                             size_t byte_offset, uint64_t bytes, uint8_t value,
                             BeError *err);

// How many entries an encoder holds so far. For a gate that asserts the shape
// of a phase without running it.
uint32_t be_encoder_length(BeEncoder *enc);

// Executes the encoded list once, waits, and drains the diagnostic channel.
// Closes the encoder, which is invalid afterward whatever the return value.
// Valid only on ENCODE_IMMEDIATE.
//
// RETURNS STATUS_OK FOR A DISPATCH THAT RAN, EVEN WITH FAILING THREADS. The
// counts land in `out_diag` and the driver renders the verdict. See the
// diagnostic section.
BeStatus be_encode_submit(BeEncoder *enc, BeDiagSummary *out_diag,
                               BeError *err);

// Closes the encoder into a replayable region. Valid only on
// ENCODE_RECORD. Nothing has executed when this returns.
//
// With `require_deferred_regions` set, a backend that cannot record a deferred
// form returns STATUS_NO_DEFERRAL here rather than producing a region that
// replays by ordinary dispatch.
BeStatus be_encode_record(BeEncoder *enc, BeRegion **out,
                               BeError *err);

// Discards an open encoder without executing or recording it. For an error path
// between `be_encode_begin` and its close.
void be_encode_abandon(BeEncoder *enc);

typedef struct BeRegionInfo {
    // 1 when the region is held in the platform's deferred form (a captured
    // CUDA graph, one Metal command buffer), 0 when replay will re-issue
    // ordinary dispatches.
    //
    // ASSERT ON THIS, do not merely log it. A region that quietly stopped being
    // deferred still produces bit-identical numbers and shows up only as a
    // slowdown, and a slowdown is inside this project's measured run-to-run
    // envelope.
    int32_t deferred;
    uint32_t dispatch_count;
    uint32_t fill_count;
    uint64_t allocator_generation;
} BeRegionInfo;

void be_region_info(BeRegion *region, BeRegionInfo *out);

// Executes a recorded region exactly `repeats` times, back to back, with no
// host round trip anywhere inside, then waits and drains the diagnostic channel
// ONCE for the whole batch.
//
// THE CONTRACT THAT FOLLOWS, and it must be read before recording anything: a
// device failure raised on repeat k is reported after repeat `repeats - 1` has
// run. A body may therefore be recorded only if its failure mode is LATCH AND
// CONTINUE. Anything whose failure must stop the step before the next dispatch
// is encoded stays a one-shot submit, which is where the contact, CCD and
// intersection phases belong.
//
// The recorded argument bytes are replayed unchanged, so a loop whose kernel
// arguments differ per iteration cannot be recorded. The linear solve does not
// need it: both GPU backends already compute the coefficients in-kernel from
// device-resident scalars, so the iteration body is straight-line and takes no
// device branch that changes which kernels run.
//
// Refuses with STATUS_STALE_REGION when the allocator has moved since the
// region was recorded.
BeStatus be_replay(BeBackend *be, BeRegion *region, uint32_t repeats,
                        BeDiagSummary *out_diag, BeError *err);

void be_region_release(BeBackend *be, BeRegion *region);

// ===========================================================================
// THE DIAGNOSTIC CHANNEL
//
// WHERE THE LINE FALLS: THE TRANSPORT IS HERE, THE VERDICT IS NOT.
//
// The predicate that decides a failure is the first argument of an assert
// written inside a neutral kernel body, so the physics half is already
// single-sourced. What a library owns is the claim, the counter, the ring, the
// reset and the readback. What it does NOT own is the answer to "is this run
// over", because the same channel carries three different kinds of failure
// under one buffer: an arena-validity check that is a driver defect, a bounds
// check that exists only because one platform does not fault, and about a dozen
// assertions that enforce non-penetration. Only the second is a backend
// concern, and collapsing all three into one return value at this level is how
// a solver breakdown gets reported as an ordinary step failure and loses its
// diagnosis.
//
// So `be_encode_submit` and `be_replay` return STATUS_OK for work
// that RAN, with the counts in `BeDiagSummary`, and the driver decides. Three
// fault models reach one driver through that one rule:
//
//   Metal never faults. An OOB read returns 0.0, an OOB write is dropped, and
//   bounds are checked against the allocation rather than the logical length,
//   so the generated entry points carry explicit checks and this channel is the
//   only report there is.
//
//   CUDA traps live release asserts, which poisons the context. That is not a
//   report and cannot be turned into one: the library returns
//   STATUS_DEVICE_FAULT and every later call answers STATUS_DEVICE_LOST.
//
//   The CPU backend must not unwind across this boundary. A range shim records
//   into the channel and returns; a genuine host trap ends the process and is
//   outside this ABI.
// ===========================================================================


// Copies this boundary's trace records into caller memory. Returns as many as
// fit and reports how many existed, so a short buffer is visible rather than
// silent.
//
// RECORDS ARE RETURNED IN SLOT ORDER, WHICH IS NOT TIME ORDER. The cursor hands
// out distinct slots, it does not order them, and recorded thread ids were
// ascending in 0 of 32 repeats on both GPU backends. Key on record content,
// never on record order.
BeStatus be_diag_drain(BeBackend *be, BeDiagRecord *out,
                            uint32_t capacity, uint32_t *out_count,
                            BeError *err);

// The source path a record's `file_id` names, or STATUS_MISUSE for an id
// this library did not assign.
//
// The library owns the id because it produced the source: Metal assembles its
// shader text and injects the mapping, CUDA hashes `__FILE__` at compile time.
// It does NOT open the file and quote the line, which is the driver's job, and
// that split is why a device assert reads identically whichever backend
// produced it.
BeStatus be_diag_file_path(BeBackend *be, uint32_t file_id, char *out,
                                size_t capacity, BeError *err);

// ===========================================================================
// COUNTERS
//
// Counters a GATE asserts on, not diagnostics a human reads.
// ===========================================================================

typedef struct BeCounters {
    // Host round trips. A per-phase assertion on this is what stops a shared
    // driver from silently reacquiring the synchronizations it exists to
    // remove.
    uint64_t syncs;
    uint64_t dispatches;
    uint64_t fills;
    uint64_t regions_recorded;
    // Regions realized in the platform's deferred form.
    uint64_t regions_deferred;
    // Regions that had to fall back to ordinary dispatches. MUST BE ASSERTED,
    // not merely logged, for the reason given at
    // `BeBackendConfig::require_deferred_regions`. Paired with
    // `regions_deferred` so a region that was never recorded deferred is
    // distinguishable from one that was and then fell back.
    uint64_t regions_fallback;
    uint64_t replays;
    uint64_t replay_repeats;
    uint64_t bytes_uploaded;
    uint64_t bytes_downloaded;
} BeCounters;

void be_counters(BeBackend *be, BeCounters *out);
void be_counters_reset(BeBackend *be);

#ifdef __cplusplus
}  // extern "C"
#endif

// Layout pins. Every one of these is asserted again on the Rust side against
// the same literals, and again at run time through `be_layout_probe`, so a
// disagreement between the two compilations of this header is named rather than
// discovered as a wrong arena id.
#if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
#if defined(__cplusplus)
#define PPF_BE_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define PPF_BE_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
PPF_BE_STATIC_ASSERT(sizeof(BeHandle) == 16, "BeHandle ABI changed");
PPF_BE_STATIC_ASSERT(sizeof(BeDiagRecord) == 32, "BeDiagRecord ABI changed");
PPF_BE_STATIC_ASSERT(sizeof(BeExtent) == 16, "BeExtent ABI changed");
PPF_BE_STATIC_ASSERT(sizeof(BeDiagSummary) == 72, "BeDiagSummary ABI changed");
PPF_BE_STATIC_ASSERT(sizeof(BeCounters) == 80, "BeCounters ABI changed");
#endif

#endif  // BACKEND_ABI_H
