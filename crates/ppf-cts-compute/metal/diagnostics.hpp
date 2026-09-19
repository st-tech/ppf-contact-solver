// File: diagnostics.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The device-side assert and trace channel for the Metal backend: the
// replacement for CUDA's device assert() and printf(), neither of which exists
// in MSL (`printf` is an undeclared identifier there, and there is no assert at
// all).
//
// The channel is one device buffer with two independent halves, both of which
// were measured on the target device before this file was written
// (the ring characterization below, and the single-record capture that
// preceded it):
//
//   ASSERT slot   One record, claimed with a compare-and-swap, so the FIRST
//                 failing thread writes it and every other failing thread only
//                 increments a counter. Measured at 1,048,576 simultaneously
//                 failing threads: exactly 1 record, write_count 1, fail_count
//                 1048576, 0 survivors lost, contents exactly correct.
//
//   RING          `ring_slots` records, each writer claiming a DISTINCT slot
//                 from an atomic cursor and the overflow DROPPED and counted.
//                 It deliberately does not wrap. Measured at 1,048,576 writers
//                 into 256 slots: cursor 1048576, exactly 256 written, exactly
//                 1048320 dropped, no empty slot, 0 torn records over 8192
//                 checked field by field, 0 duplicate thread ids. The wrapping
//                 variant (slot = cursor % N) tears, at a rate low enough on
//                 Metal (1 record in 1024) that its output would read as
//                 correct.
//
// This channel is MORE capable than what it replaces on three axes, and the
// difference is worth knowing when reading a report: records survive the
// dispatch that wrote them (CUDA's assert traps and poisons the context, taking
// the printf FIFO with it), nothing is lost silently (overflow is counted
// exactly), and a violation COUNT is available (CUDA reports one message and
// nothing about how many threads hit it).
//
// Everything on the device side is memory_order_relaxed, which is the only
// order this device accepts. That is sound here because the only reader is the
// host and it reads only after waitUntilCompleted.

#pragma once

#include "diagnostic_record.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace metal_backend {

struct Context;
struct Allocator;

// One violation record. Layout is shared with the shader, so it is a plain
// aggregate of 32-bit fields with no float3 anywhere (MSL float3 is 16 bytes
// against Vec3f at 12, so a vector type in a shared struct would silently
// disagree on offsets).
using DiagRecord = ::BeDiagRecord;

// Reserved assert ids. A DIAG_TRACE site picks its own id and must take it from
// DIAG_ID_FIRST_USER upward so it cannot be confused with a violation.
// THE RIGHT-HAND SIDES KEEP THEIR `PPF_BE_` PREFIX, and that is not an
// oversight of the prefix removal: the C ABI's macros are one of the three
// name sets that keep it, because they are a public boundary shared with
// callers this tree does not own. Stripping it here made each line initialize
// the constant from ITSELF, which compiles nowhere and broke the Metal build.
constexpr uint32_t DIAG_ID_ASSERT = PPF_BE_DIAG_ID_ASSERT;
constexpr uint32_t DIAG_ID_BOUNDS = PPF_BE_DIAG_ID_BOUNDS;
constexpr uint32_t DIAG_ID_FIRST_USER = PPF_BE_DIAG_ID_FIRST_USER;

struct Diagnostics;

// 'ring_slots' is how many records the ring can hold before it starts
// DROPPING (never wrapping: a wrapping ring tears, measured).
Diagnostics *diag_create(Context *ctx, Allocator *alloc, unsigned ring_slots,
                         std::string *err);
void diag_destroy(Diagnostics *d);

// The MSL source fragment declaring the record struct, the buffer layout and
// the DIAG_ASSERT / DIAG_TRACE macros. Prepended to every solver shader.
//
// The returned pointer is owned by 'd' and is valid until diag_destroy.
//
// The fragment names atomic_uint and memory_order_relaxed unqualified, so it
// goes after the backend's MSL prologue (the one that includes <metal_stdlib>
// and opens namespace metal) and before any source that asserts.
//
// The macro spellings (DIAG_ASSERT, DIAG_ASSERT4, DIAG_BOUNDS_CHECK, DIAG_TRACE,
// DIAG_ARG, DIAG_BIND) are shared with the CUDA backend, which expands
// the same names over the same record layout. Kernels are written once, so a
// name that exists on only one backend would fork the source. What differs per
// backend is the expansion and the file id mechanism: the host injects the id
// here because it concatenates the source, while CUDA has a real #include and
// hashes __FILE__ instead.
//
// The fragment leaves DIAG_FILE_ID defined as 0. The shader compiler owns file
// attribution and must emit, ahead of each source segment it concatenates:
//
//     #undef  DIAG_FILE_ID
//     #define DIAG_FILE_ID <id from diag_register_file>
//     #line 1 "<path as passed to diag_register_file>"
//
// with the #line LAST, so the directives themselves do not consume line
// numbers belonging to the segment. Verified on this device: the Metal
// compiler accepts `#line N "path"`, __LINE__ follows it across segment
// boundaries, an injected define is visible to the shader, and compile
// diagnostics carry the injected filename.
const char *diag_shader_prologue(Diagnostics *d);

// Which binding index the diagnostic buffer occupies. Reserved for the
// lifetime of the backend so no kernel may take it. Only 31 slots exist and
// the limit is compiler-enforced, so the reservation is not a convention that
// can be quietly ignored: an arena bound here would displace the channel.
unsigned diag_binding_index(Diagnostics *d);
unsigned diag_buffer_id(Diagnostics *d);

// Clears the counters before a dispatch. Records survive the dispatch that
// wrote them, unlike CUDA's assert, which poisons the context and loses the
// printf FIFO with it.
void diag_reset(Diagnostics *d);

// Reads the channel back after waitUntilCompleted. 'assert_hit' is whether
// the single-record assert slot fired; 'fail_count' is how many threads hit
// it, which CUDA cannot report at all. 'dropped' is ring overflow.
//
// Ring records are returned in SLOT order, which is not time order: the cursor
// hands out distinct slots, it does not order them, and recorded thread ids
// were ascending in 0 of 32 repeats on both backends. Key on record CONTENT,
// never on record order.
//
// Call only after waitUntilCompleted AND after both cb.status and cb.error
// have been checked. A killed command buffer reports Completed and leaves
// plausible stale data behind, so a readback taken without those checks
// describes a dispatch that may never have run.
struct DiagReadback {
    bool     assert_hit;
    DiagRecord assert_record;
    uint64_t fail_count;
    uint64_t ring_written;
    uint64_t ring_dropped;
    std::vector<DiagRecord> ring;
};
bool diag_read(Diagnostics *d, DiagReadback *out, std::string *err);

// Registers a source file and returns its id, for the #line injection the
// shader compiler performs. Ids are stable for the process lifetime.
unsigned diag_register_file(Diagnostics *d, const char *path);

// Turns a record into a human-readable line by opening the source file and
// quoting the offending line. Resolving LAZILY means no generated id table
// to go stale, and the expression text comes for free.
std::string diag_format(Diagnostics *d, const DiagRecord &r);

// Every record a readback holds, rendered as a suffix for the message its
// caller is already building, and empty when nothing fired.
//
// A COUNT ALONE DOES NOT LOCATE THE CHECK. A dispatch runs many kernels worth
// of asserts and bounds tests, so "the dispatch reported 3 device diagnostic
// failure(s)" leaves the reader to find which of them by hand. The record
// carries the file id and line the shader assembler injected with its #line
// directives together with four captured floats, and diag_format resolves
// them against the file on disk, which is what makes a device assert name the
// kernel source the way a host assert names its own. Injecting that mapping is
// one of the reasons the host assembles the shader itself, so a report built
// from the count alone leaves the mechanism unused.
std::string diag_records(Diagnostics *d, const DiagReadback &readback);

}  // namespace metal_backend
