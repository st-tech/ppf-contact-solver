// File: diagnostics.mm
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Implementation of the device-side assert and trace channel. See
// diagnostics.hpp for what the channel is and for the measurements the design
// is pinned to.
//
// This translation unit contains no Objective-C symbol. The diagnostic buffer
// is device memory like any other, so the Allocator creates and owns it, and
// what is left here is a memory layout, the shader text that agrees with that
// layout, and the host-side accounting and formatting. The file stays .mm to
// match the rest of the backend's translation units and because the contract it
// depends on is a Metal one: shared storage, host-coherent after
// waitUntilCompleted.
//
// The seam to the rest of the backend is deliberately one function wide,
// declared by the peer header arena.hpp:
//
//   alloc_reserve_dedicated(Allocator *, const char *name, size_t bytes,
//                           unsigned *out_buffer_id,
//                           unsigned *out_binding_index,
//                           void **out_host_ptr, std::string *err) -> bool
//       Allocates a host-visible buffer that lives for the whole backend, and
//       reserves one buffer id and one binding index for it. The allocator
//       owns both id spaces, so the reservation has to be made there or a
//       later arena could take the slot, and a wrong arena id is a SILENT
//       wrong answer on this backend.
//
//   Nothing from Context. The context is held only so the channel cannot be
//   created against one device and read against another; the buffer that backs
//   it belongs to the allocator, which belongs to the context.

#include "arena.hpp"
#include "diagnostics.hpp"
#include "metal_context.hpp"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace metal_backend {

namespace {

// ---------------------------------------------------------------- layout

// Host mirror of the shader's DiagHeader. Field order and size must match
// the MSL struct in DIAG_MSL below exactly.
//
// The counters are 32-bit because this device has no 64-bit atomics (measured
// absent, and provisionally so: the capture came from a paravirtual device).
// They are read here as plain uint32_t rather than as std::atomic because the
// GPU is the only writer and the host reads only after waitUntilCompleted.
struct DiagBufferHeader {
    uint32_t assert_claim;   // atomic: 0 = free, 1 = claimed by one thread
    uint32_t assert_ready;   // atomic: published LAST by the claiming thread
    uint32_t fail_count;     // atomic: threads that failed an assert
    uint32_t ring_cursor;    // atomic: total ring claims, INCLUDING drops
    uint32_t ring_dropped;   // atomic: claims that found no slot
    uint32_t ring_capacity;  // plain: written by the host, read by the shader
    uint32_t reserved0;
    uint32_t reserved1;
};

static_assert(sizeof(DiagBufferHeader) == 32, "diagnostic header layout");
static_assert(sizeof(DiagRecord) == 32, "diagnostic record layout");
static_assert(alignof(DiagRecord) == 4, "diagnostic record alignment");

// A dispatch would have to record more than 2^32 violations for a counter to
// wrap, which is beyond the largest grid this solver dispatches. There is no
// wider atomic available to make the bound structural, so it is stated here
// and the readback below checks the accounting identity that a wrap would
// break.
constexpr uint64_t DIAG_COUNTER_MODULUS = 1ull << 32;

// The proven configurations were 256 and 65536 slots. The cap is well above
// both and exists so an absurd request fails at creation instead of reserving
// hundreds of megabytes that no report would ever read.
constexpr unsigned DIAG_MAX_RING_SLOTS = 1u << 20;

// ------------------------------------------------------------ file table

// The file table is process-wide so an id means the same thing in every record
// the process ever formats, including across a backend teardown and recreate.
// Registration happens while shaders are being concatenated, formatting happens
// after a dispatch fails, and neither is on a hot path, so one mutex covers it.
struct FileTable {
    std::mutex mutex;
    std::vector<std::string> paths;
    std::unordered_map<std::string, unsigned> ids;
};

FileTable &file_table() {
    static FileTable table;
    return table;
}

void set_err(std::string *err, const std::string &msg) {
    if (err) {
        *err = msg;
    }
}

std::string fmt(const char *f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char *f, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

// ------------------------------------------------------------ shader text

// The MSL half of the channel. Concatenated ahead of every solver shader.
//
// DIAG_BINDING is defined by the host immediately before this text, from
// the binding index the allocator reserved.
const char *const DIAG_MSL = R"MSL(
// ---------------------------------------------------------------------------
// ppf diagnostics channel. Generated by metal/diagnostics.mm. Editing a
// copy of this text does nothing; edit the generator.
//
// MSL has no assert and no printf. A violation is recorded into one device
// buffer that the host reads after waitUntilCompleted. Two independent halves
// share that buffer: a single ASSERT slot claimed by compare-and-swap, and a
// RING of ring_capacity slots handed out by an atomic cursor.
//
// The ring does NOT wrap. A writer whose claim lands past the capacity records
// nothing and increments ring_dropped. The wrapping form lets several writers
// own one slot and was measured to tear, at a rate low enough on this device
// that its output would read as correct.
//
// Records are handed out in slot order, which is NOT time order. Nothing may
// depend on record ORDER, only on record CONTENT.
//
// Every atomic is memory_order_relaxed: it is the only order this device
// accepts, and waitUntilCompleted is the synchronization point.
//
// This fragment uses atomic_uint, atomic_fetch_add_explicit and
// memory_order_relaxed unqualified, so it must be concatenated AFTER the
// backend's MSL prologue, which is what includes <metal_stdlib> and opens
// namespace metal.
// ---------------------------------------------------------------------------

struct DiagHeader {
    atomic_uint assert_claim;
    atomic_uint assert_ready;
    atomic_uint fail_count;
    atomic_uint ring_cursor;
    atomic_uint ring_dropped;
    uint        ring_capacity;
    uint        reserved0;
    uint        reserved1;
};

struct BeDiagRecord {
    uint  assert_id;
    uint  file_id;
    uint  line;
    uint  thread_id;
    float payload[4];
};

// Reserved assert ids, matching diagnostics.hpp. A DIAG_TRACE site picks its
// own id from DIAG_ID_FIRST_USER upward.
#define DIAG_ID_ASSERT 1u
#define DIAG_ID_BOUNDS 2u
#define DIAG_ID_FIRST_USER 16u

// File attribution. The host concatenates the shader source itself, so it IS
// the include mechanism and knows the provenance of every line. It re-defines
// DIAG_FILE_ID and emits a #line directive ahead of each segment; this default
// applies only to text no segment claimed.
#define DIAG_FILE_ID 0u

// The channel as a kernel sees it. The two device pointers and the thread id
// live in thread space, which is where a struct of device pointers belongs in
// MSL and which is measured to resolve correctly.
struct Diag {
    device DiagHeader *hdr;
    device BeDiagRecord *rec;  // rec[0] is the assert slot, rec[1 + i] ring i
    uint tid;
};

// The diagnostic buffer occupies a binding index reserved for the lifetime of
// the backend. Every kernel declares it with DIAG_ARG and binds it once at
// entry with DIAG_BIND.
#define DIAG_ARG                                                           \
    device DiagHeader *diag_buffer [[buffer(DIAG_BINDING)]]

inline Diag diag_bind(device DiagHeader *hdr, uint tid) {
    Diag d;
    d.hdr = hdr;
    // The records follow the header contiguously in the same allocation. The
    // header is 32 bytes and a record needs 4-byte alignment, so this offset
    // is exact; the host asserts the same arithmetic against its own mirror.
    d.rec = (device BeDiagRecord *)(hdr + 1);
    d.tid = tid;
    return d;
}

#define DIAG_BIND(tid) diag_bind(diag_buffer, (tid))

// THE ONE NAME A NEUTRAL BODY MAY SPELL for the handle DIAG_BIND yields.
// The three targets bind three different things, a `Diag` here, a
// `diagnostics::Device` under nvcc and a pointer on the host, and a body
// takes it BY VALUE and passes it to the macros without ever dereferencing it,
// so they need agree on nothing but the name. It is a THREAD-space value here,
// which is where a struct of device pointers belongs in MSL, so a body's
// parameter carries no address space of its own.
// `seam/seam_host.h` states the contract in full.
typedef Diag DiagHandle;

// The record write. Kept in a macro rather than a function because __LINE__
// and DIAG_FILE_ID must be captured at the CALL SITE; a helper function would
// attribute every violation in the program to this file.
#define DIAG_FILL(rec_ptr, id, tid, p0, p1, p2, p3)                        \
    do {                                                                       \
        device BeDiagRecord *_ppf_r = (rec_ptr);                              \
        _ppf_r->assert_id = (id);                                              \
        _ppf_r->file_id = DIAG_FILE_ID;                                         \
        _ppf_r->line = __LINE__;                                               \
        _ppf_r->thread_id = (tid);                                             \
        _ppf_r->payload[0] = (p0);                                             \
        _ppf_r->payload[1] = (p1);                                             \
        _ppf_r->payload[2] = (p2);                                             \
        _ppf_r->payload[3] = (p3);                                             \
    } while (0)

// The assert, with a 4-float payload.
//
// THE FAILING THREAD RECORDS AND CONTINUES. There is no early-return variant
// and there must never be one: the CUDA source has 52 barrier call sites, and
// exiting a subset of threads before a barrier is undefined behavior, which
// would turn a loud assert into exactly the silent failure this project
// forbids. The early exit buys nothing real, because the results of a dispatch
// that tripped an assert are discarded anyway. The macro expands to a
// STATEMENT, so it constrains nothing about the enclosing function's return
// type.
//
// PLACEMENT IS A RULE, NOT A PREFERENCE. Assert at kernel entry and exit and
// at loop boundaries, NEVER in an innermost loop. Measured cost at
// once-per-kernel placement is -0.0% to +4.4%, which is why the CUDA build can
// keep its asserts live in production; in the innermost loop it is +84% to
// +121%, and most of that is the predicate and its control flow rather than
// the capture, so the placement rule is not fixed by making the payload
// cheaper.
//
// The claim is a compare-and-swap so that only the FIRST failing thread
// writes. MSL's compare-exchange is WEAK and may fail spuriously, so the retry
// condition is `expected == 0`: a spurious failure leaves the observed value
// at 0 and retries, a genuine loss observes 1 and stops. Measured at 1,048,576
// simultaneously failing threads: exactly one record.
#define DIAG_ASSERT4(diag, cond, p0, p1, p2, p3)                                \
    do {                                                                       \
        if (!(cond)) {                                                         \
            atomic_fetch_add_explicit(&(diag).hdr->fail_count, 1u,             \
                                      memory_order_relaxed);                   \
            uint _ppf_seen = 0u;                                               \
            bool _ppf_won = false;                                             \
            while (!_ppf_won && _ppf_seen == 0u) {                             \
                _ppf_won = atomic_compare_exchange_weak_explicit(              \
                    &(diag).hdr->assert_claim, &_ppf_seen, 1u,                 \
                    memory_order_relaxed, memory_order_relaxed);               \
            }                                                                  \
            if (_ppf_won) {                                                    \
                DIAG_FILL((diag).rec, DIAG_ID_ASSERT, (diag).tid,      \
                              (p0), (p1), (p2), (p3));                         \
                /* published last, so the host can tell a complete record   */ \
                /* from a claim whose writer never finished                 */ \
                atomic_store_explicit(&(diag).hdr->assert_ready, 1u,           \
                                      memory_order_relaxed);                   \
            }                                                                  \
        }                                                                      \
    } while (0)

#define DIAG_ASSERT(diag, cond)                                                 \
    DIAG_ASSERT4((diag), (cond), 0.0f, 0.0f, 0.0f, 0.0f)

// An explicit index bounds check, and it is mandatory rather than defensive.
// Metal NEVER faults on an out-of-bounds access: a read returns 0.0, a write
// is dropped, a write 1 GiB past the end of a buffer completes with no error,
// and bounds are checked against the ALLOCATION rather than the logical
// length, so every pooled arena and padded array has a fully silent logical-OOB
// path. An index bug that CUDA catches with a device assert otherwise passes
// here in total silence.
//
// The index and the count are recorded as floats, so a value above 2^24 is
// rounded in the report. That is a property of the record layout, and it costs
// nothing that matters: the report names the file and line, and the index is
// there to say which end of the range was overrun.
#define DIAG_BOUNDS_CHECK(diag, index, count)                                   \
    do {                                                                       \
        uint _ppf_i = (uint)(index);                                           \
        uint _ppf_n = (uint)(count);                                           \
        if (!(_ppf_i < _ppf_n)) {                                              \
            atomic_fetch_add_explicit(&(diag).hdr->fail_count, 1u,             \
                                      memory_order_relaxed);                   \
            uint _ppf_seen = 0u;                                               \
            bool _ppf_won = false;                                             \
            while (!_ppf_won && _ppf_seen == 0u) {                             \
                _ppf_won = atomic_compare_exchange_weak_explicit(              \
                    &(diag).hdr->assert_claim, &_ppf_seen, 1u,                 \
                    memory_order_relaxed, memory_order_relaxed);               \
            }                                                                  \
            if (_ppf_won) {                                                    \
                DIAG_FILL((diag).rec, DIAG_ID_BOUNDS, (diag).tid,      \
                              (float)_ppf_i, (float)_ppf_n, 0.0f, 0.0f);       \
                atomic_store_explicit(&(diag).hdr->assert_ready, 1u,           \
                                      memory_order_relaxed);                   \
            }                                                                  \
        }                                                                      \
    } while (0)

// The trace, which is what replaces a device printf. Unconditional: every
// thread that reaches it claims a slot, the first ring_capacity claims write a
// record, and the rest are counted as dropped and nothing is written. The same
// placement rule as DIAG_ASSERT4 applies, for the same measured reason.
//
// 'id' identifies the site to the reader and must be at least
// DIAG_ID_FIRST_USER so it cannot be read as a violation.
#define DIAG_TRACE(diag, id, p0, p1, p2, p3)                                    \
    do {                                                                       \
        uint _ppf_slot = atomic_fetch_add_explicit(&(diag).hdr->ring_cursor,   \
                                                   1u, memory_order_relaxed);  \
        if (_ppf_slot < (diag).hdr->ring_capacity) {                           \
            DIAG_FILL((diag).rec + 1u + _ppf_slot, (id), (diag).tid,       \
                          (p0), (p1), (p2), (p3));                             \
        } else {                                                               \
            atomic_fetch_add_explicit(&(diag).hdr->ring_dropped, 1u,           \
                                      memory_order_relaxed);                   \
        }                                                                      \
    } while (0)
)MSL";

}  // namespace

// ------------------------------------------------------------------ state

struct Diagnostics {
    Context *ctx = nullptr;
    Allocator *alloc = nullptr;
    unsigned ring_slots = 0;
    unsigned buffer_id = 0;
    unsigned binding_index = 0;
    DiagBufferHeader *header = nullptr;  // host mapping of the shared buffer
    DiagRecord *records = nullptr;       // records[0] assert, [1 + i] ring i
    std::string prologue;
};

// ----------------------------------------------------------- construction

Diagnostics *diag_create(Context *ctx, Allocator *alloc, unsigned ring_slots,
                         std::string *err) {
    if (!ctx) {
        set_err(err, "diag_create: null Context");
        return nullptr;
    }
    if (!alloc) {
        set_err(err, "diag_create: null Allocator");
        return nullptr;
    }
    if (ring_slots == 0) {
        set_err(err, "diag_create: ring_slots must be at least 1; a zero-slot "
                     "ring would drop every trace record and report nothing "
                     "beyond the count");
        return nullptr;
    }
    if (ring_slots > DIAG_MAX_RING_SLOTS) {
        set_err(err, fmt("diag_create: ring_slots %u exceeds the maximum %u",
                         ring_slots, DIAG_MAX_RING_SLOTS));
        return nullptr;
    }

    const size_t bytes =
        sizeof(DiagBufferHeader) +
        static_cast<size_t>(ring_slots + 1u) * sizeof(DiagRecord);

    unsigned buffer_id = 0;
    unsigned binding_index = 0;
    void *host_ptr = nullptr;
    std::string alloc_err;
    if (!alloc_reserve_dedicated(alloc, "diagnostics", bytes, &buffer_id,
                                 &binding_index, &host_ptr, &alloc_err)) {
        set_err(err, "diag_create: reserving the diagnostic buffer failed: " +
                         alloc_err);
        return nullptr;
    }
    if (!host_ptr) {
        set_err(err, "diag_create: the allocator reserved the diagnostic "
                     "buffer but returned no host mapping; the channel is read "
                     "on the host and cannot work without one");
        return nullptr;
    }

    Diagnostics *d = new Diagnostics();
    d->ctx = ctx;
    d->alloc = alloc;
    d->ring_slots = ring_slots;
    d->buffer_id = buffer_id;
    d->binding_index = binding_index;
    d->header = static_cast<DiagBufferHeader *>(host_ptr);
    // The shader derives its record pointer the same way, from the end of the
    // header. Both sides do the arithmetic on their own struct, so a layout
    // disagreement would be a wrong answer rather than a crash; the two
    // static_asserts above pin the sizes that make them agree.
    d->records = reinterpret_cast<DiagRecord *>(d->header + 1);

    d->prologue = "#define DIAG_BINDING " + std::to_string(binding_index) +
                  "\n" + DIAG_MSL;

    diag_reset(d);
    return d;
}

void diag_destroy(Diagnostics *d) {
    // The buffer belongs to the allocator, which outlives this object and
    // releases it with the rest of the backend's memory.
    delete d;
}

// -------------------------------------------------------------- accessors

const char *diag_shader_prologue(Diagnostics *d) { return d->prologue.c_str(); }

unsigned diag_binding_index(Diagnostics *d) { return d->binding_index; }

unsigned diag_buffer_id(Diagnostics *d) { return d->buffer_id; }

// ------------------------------------------------------------------ reset

void diag_reset(Diagnostics *d) {
    std::memset(d->header, 0, sizeof(DiagBufferHeader));
    d->header->ring_capacity = d->ring_slots;
    // The assert slot is cleared so a capture of the raw buffer reads cleanly.
    std::memset(&d->records[0], 0, sizeof(DiagRecord));
    // The ring slots are deliberately NOT cleared. Slot i is written exactly
    // when claim i was made, so the written prefix is [0, min(cursor,
    // capacity)) and nothing outside it is ever read. Clearing them would cost
    // a memset of the whole ring before every dispatch to establish a fact the
    // cursor already establishes.
}

// --------------------------------------------------------------- readback

bool diag_read(Diagnostics *d, DiagReadback *out, std::string *err) {
    if (!out) {
        set_err(err, "diag_read: null readback");
        return false;
    }

    const DiagBufferHeader h = *d->header;

    if (h.ring_capacity != d->ring_slots) {
        set_err(err, fmt("diag_read: the diagnostic buffer reports capacity %u "
                         "where this channel was created with %u; the buffer "
                         "was overwritten by something else",
                         h.ring_capacity, d->ring_slots));
        return false;
    }
    if (h.assert_claim == 0 && h.assert_ready != 0) {
        set_err(err, "diag_read: the assert slot is published but unclaimed; "
                     "the buffer was overwritten by something else");
        return false;
    }
    if (h.assert_claim == 0 && h.fail_count != 0) {
        set_err(err, fmt("diag_read: %u threads failed an assert but no thread "
                         "claimed the slot; the buffer was overwritten by "
                         "something else",
                         h.fail_count));
        return false;
    }
    if (h.assert_claim != 0 && h.assert_ready == 0) {
        set_err(err, "diag_read: a thread claimed the assert slot and never "
                     "published it, so the dispatch stopped between the two; "
                     "check the command buffer status and error");
        return false;
    }

    const uint32_t written =
        h.ring_cursor < h.ring_capacity ? h.ring_cursor : h.ring_capacity;
    const uint32_t expected_dropped = h.ring_cursor - written;
    if (h.ring_dropped != expected_dropped) {
        // The cursor and the drop counter are incremented by the same macro on
        // complementary branches, so this identity holds by construction. It
        // breaking means the shader prologue and this reader have gone out of
        // agreement, or a counter wrapped.
        set_err(err,
                fmt("diag_read: ring accounting is inconsistent: cursor %u, "
                    "capacity %u, written %u, dropped %u where %u was required "
                    "(counters wrap at %llu)",
                    h.ring_cursor, h.ring_capacity, written, h.ring_dropped,
                    expected_dropped,
                    static_cast<unsigned long long>(DIAG_COUNTER_MODULUS)));
        return false;
    }

    out->assert_hit = h.assert_ready != 0;
    out->assert_record = d->records[0];
    out->fail_count = h.fail_count;
    out->ring_written = written;
    out->ring_dropped = h.ring_dropped;
    out->ring.assign(d->records + 1, d->records + 1 + written);
    return true;
}

// ------------------------------------------------------------- file table

unsigned diag_register_file(Diagnostics *d, const char *path) {
    (void)d;  // the table is process-wide, see FileTable
    FileTable &table = file_table();
    const std::string key = path ? std::string(path) : std::string();
    std::lock_guard<std::mutex> lock(table.mutex);
    auto it = table.ids.find(key);
    if (it != table.ids.end()) {
        return it->second;
    }
    // Id 0 is the prologue's default for text no segment claimed, so real
    // files start at 1.
    const unsigned id = static_cast<unsigned>(table.paths.size()) + 1u;
    table.paths.push_back(key);
    table.ids.emplace(key, id);
    return id;
}

namespace {

bool lookup_file(unsigned id, std::string *path) {
    FileTable &table = file_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    if (id == 0 || id > table.paths.size()) {
        return false;
    }
    *path = table.paths[id - 1u];
    return true;
}

std::string trim(const std::string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return std::string();
    }
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Opens the file and returns the requested line. Every failure returns a
// message that says what went wrong instead of reporting nothing: a
// diagnostic path must not itself become a failure path, and the file, the
// line and the payload are already in hand even when the source is not.
std::string quote_source(const std::string &path, uint32_t line) {
    if (path.empty()) {
        return "(source unavailable: no path registered)";
    }
    std::ifstream in(path);
    if (!in) {
        return fmt("(source unavailable: cannot open \"%s\")", path.c_str());
    }
    if (line == 0) {
        return "(source unavailable: line 0)";
    }
    std::string text;
    uint32_t n = 0;
    while (std::getline(in, text)) {
        if (++n == line) {
            const std::string quoted = trim(text);
            return quoted.empty() ? std::string("(source line is blank)")
                                  : quoted;
        }
    }
    return fmt("(source unavailable: \"%s\" has %u lines)", path.c_str(), n);
}

const char *record_label(uint32_t assert_id) {
    switch (assert_id) {
    case DIAG_ID_ASSERT:
        return "assert failed";
    case DIAG_ID_BOUNDS:
        return "bounds check failed";
    default:
        return "trace";
    }
}

}  // namespace

std::string diag_format(Diagnostics *d, const DiagRecord &r) {
    (void)d;  // the table is process-wide, see FileTable

    std::string path;
    const bool known = lookup_file(r.file_id, &path);
    const std::string shown =
        known ? path : fmt("<unregistered file id %u>", r.file_id);

    std::string head;
    if (r.assert_id == DIAG_ID_BOUNDS) {
        head = fmt("%s:%u: bounds check failed [thread %u] index %.9g not "
                   "below %.9g",
                   shown.c_str(), r.line, r.thread_id,
                   static_cast<double>(r.payload[0]),
                   static_cast<double>(r.payload[1]));
    } else {
        head = fmt("%s:%u: %s", shown.c_str(), r.line,
                   record_label(r.assert_id));
        if (r.assert_id >= DIAG_ID_FIRST_USER) {
            head += fmt(" id %u", r.assert_id);
        }
        head += fmt(" [thread %u] payload=(%.9g, %.9g, %.9g, %.9g)",
                    r.thread_id, static_cast<double>(r.payload[0]),
                    static_cast<double>(r.payload[1]),
                    static_cast<double>(r.payload[2]),
                    static_cast<double>(r.payload[3]));
    }

    const std::string source =
        known ? quote_source(path, r.line)
              : fmt("(source unavailable: file id %u was never registered)",
                    r.file_id);
    return head + " | " + source;
}

std::string diag_records(Diagnostics *d, const DiagReadback &readback) {
    std::string out;
    if (readback.assert_hit) {
        out += "\n  assert: " + diag_format(d, readback.assert_record);
    }
    // Four is what fits in a log line a human reads; the count below says how
    // many were held back, and ring_dropped how many the channel never took.
    const size_t shown = std::min<size_t>(readback.ring.size(), 4u);
    for (size_t i = 0; i < shown; ++i) {
        out += "\n  [" + std::to_string(i) + "] " +
               diag_format(d, readback.ring[i]);
    }
    if (readback.ring.size() > shown) {
        out += "\n  ... " + std::to_string(readback.ring.size() - shown) +
               " more record(s)";
    }
    if (readback.ring_dropped != 0) {
        out += "\n  " + std::to_string(readback.ring_dropped) +
               " record(s) dropped on ring overflow";
    }
    return out;
}

}  // namespace metal_backend
