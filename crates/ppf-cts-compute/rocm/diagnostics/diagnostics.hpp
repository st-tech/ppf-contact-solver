// File: diagnostics.hpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// THE DIAGNOSTIC CHANNEL'S TRANSPORT FOR THE CUDA TARGET.
//
// A device has no way to report an assertion or a trace to the host except
// through memory the host can read back, so the channel is platform machinery
// the five verbs require: an arena block, a header, a ring of records, and the
// device-side macros a kernel body writes through. The Metal backend answers
// the same macro names from `metal/diagnostics.hpp`, which is what lets one
// kernel body compile on both.
//
// The RECORD is not here. `diagnostic_record.hpp` stays in the neutral kernel
// tree, because its layout is what the two sides agree on and both targets
// must read one declaration of it.

// WHY THIS IS A SEPARATE FILE FROM cuda/diagnostics/diagnostics.hpp. The
// diagnostic channel's TRANSPORT is platform machinery, which rule (1b)
// puts inside a backend library; the RECORD layout it moves is declared
// once above the seam in kernels/seam/backend_abi.h and is not restated
// here. So the two files are near-identical by construction and neither
// includes the other, which is what keeps one backend's header out of
// another backend's compilation.
#ifndef CUDA_DIAGNOSTICS_HPP
#define CUDA_DIAGNOSTICS_HPP

#include "arena/arena.hpp"
#include "diagnostic_record.hpp"

#include <string>
#include <vector>

namespace diagnostics {

struct Header {
    unsigned assert_claim;
    unsigned assert_ready;
    unsigned fail_count;
    unsigned ring_cursor;
    unsigned ring_dropped;
    unsigned ring_capacity;
    unsigned reserved0;
    unsigned reserved1;
};

static_assert(sizeof(Header) == 32, "diagnostic header ABI changed");

// THE CHANNEL LIVES IN MAPPED HOST MEMORY, NOT IN THE ARENA, and that is a
// transfer decision rather than a storage one. It is drained once per BOUNDARY
// to learn whether a kernel recorded anything, and out of device memory that
// question cost a device-to-host copy every time, on runs where the answer is
// almost always no. Mapped host memory makes the drain a host READ: the device
// gets a pointer into the same pages through `hipHostGetDevicePointer`, so no
// copy exists to count.
//
// IT COSTS NOTHING ON THE HOT PATH BECAUSE THE DEVICE BARELY WRITES IT.
// `DIAG_ASSERT4` reaches the header only when its condition FAILS, and no
// neutral kernel in the tree uses `DIAG_TRACE`, so a healthy run touches these
// pages from the device zero times. A run that does touch them is reporting a
// violated invariant and is about to abort, which is not a path to tune.
//
// `host_base` and `device_base` ADDRESS THE SAME BYTES. The host side reads and
// clears through the first; `bind` hands the second to a kernel. Both are null
// until `create`.
struct Channel {
    unsigned char *host_base{nullptr};
    unsigned char *device_base{nullptr};
    size_t bytes{0};
    unsigned ring_capacity{0};
};

struct Readback {
    bool assert_hit{false};
    BeDiagRecord assert_record{};
    unsigned fail_count{0};
    unsigned ring_written{0};
    unsigned ring_dropped{0};
    std::vector<BeDiagRecord> ring;
};

bool create(unsigned ring_capacity, Channel *out, std::string *error);
// The channel's own pages are mapped host memory and owe the arena nothing.
void destroy(Channel *channel);
void reset(const Channel &channel);
bool read(const Channel &channel, Readback *out, std::string *error);

bool create_global(unsigned ring_capacity, std::string *error);
const Channel &global();

// The path a device report's file id names, or null when this library compiled
// no such file. See `file_path` for why a miss is not guessed at.
const char *file_path(unsigned id);

// Fires one deliberately failing check from a kernel and reports whether the
// record reached the host, leaving the channel clean either way. This is the
// only check of the diagnostic channel that runs on the device, so it is the
// only one that can fail on a machine whose source is correct; `be_create`
// runs it when `PPF_DIAG_SELFTEST` is set.
bool selftest_global(std::string *error);
void reset_global();
bool read_global(Readback *out, std::string *error);
void reset_assert_global();
void check_assert_global();


struct Device {
    Header *header;
    BeDiagRecord *records;
    unsigned thread_id;
};


__device__ inline Device bind(const Channel &channel, unsigned thread_id) {
    unsigned char *base = channel.device_base;
    Device out{reinterpret_cast<Header *>(base),
               reinterpret_cast<BeDiagRecord *>(base + sizeof(Header)),
               thread_id};
    return out;
}

// FNV-1a over the OCTETS of the path. The `unsigned char` cast is what makes
// that true: `char` is signed on every platform this tree builds for, so
// hashing `*text` directly sign-extends any byte above 0x7f and produces a
// number no other implementation of FNV-1a agrees with. The host mirror in
// `seam/kernelgen.py` hashes octets, and this is the side that has to match it.
__host__ __device__ constexpr uint32_t file_id(const char *text,
                                                uint32_t hash = 2166136261u) {
    return *text ? file_id(text + 1,
                           (hash ^ static_cast<uint32_t>(
                                static_cast<unsigned char>(*text))) * 16777619u)
                 : hash;
}

} // namespace diagnostics

#define DIAG_ARG diagnostics::Channel diag_channel
#define DIAG_BIND(tid) diagnostics::bind(diag_channel, (tid))

// THE ONE NAME A NEUTRAL BODY MAY SPELL for the handle `DIAG_BIND` yields.
// The three targets bind three different things, a `Device` here, a `Diag`
// under MSL and a pointer on the host, and a body takes it BY VALUE and passes
// it to the macros without ever dereferencing it, so they need agree on nothing
// but the name. `seam/seam_host.h` states the contract in full.
using DiagHandle = diagnostics::Device;

#define CUDA_DIAG_FILL(record, id, diag, p0, p1, p2, p3)                  \
    do {                                                                       \
        (record)->assert_id = (id);                                             \
        (record)->file_id = diagnostics::file_id(__FILE__);               \
        (record)->line = __LINE__;                                              \
        (record)->thread_id = (diag).thread_id;                                 \
        (record)->payload[0] = (p0);                                            \
        (record)->payload[1] = (p1);                                            \
        (record)->payload[2] = (p2);                                            \
        (record)->payload[3] = (p3);                                            \
    } while (0)

// THE NULL CHECK, for the same reason as the CUDA twin.
//
// `bind` yields `header` from `channel.device_base`, which is null until the
// channel is created, so a kernel handed an uncreated channel gets a null
// header and the first statement on a failing condition becomes
// `atomicAdd((unsigned *)0, 1u)`: a null device write reported later as an
// illegal access, with the violated invariant never reaching the host. The
// report becomes the crash, and only on runs where something was already
// wrong. The host spelling in `kernels/seam/seam_host.h` tests its pointer for
// the same reason; this is that check on the device side.
//
// It does NOT replace attaching a channel: a guarded assert with no channel is
// silent. `be_create` wires the backend's channel in as the global one so these
// records reach the drain.
#define DIAG_ASSERT4(diag, cond, p0, p1, p2, p3)                               \
    do {                                                                       \
        if (!(cond)) {                                                         \
            if ((diag).header == nullptr) {                                    \
                break;                                                         \
            }                                                                  \
            atomicAdd(&(diag).header->fail_count, 1u);                         \
            if (atomicCAS(&(diag).header->assert_claim, 0u, 1u) == 0u) {       \
                CUDA_DIAG_FILL(&(diag).records[0], DIAG_ID_ASSERT,     \
                                   diag, p0, p1, p2, p3);                      \
                __threadfence_system();                                        \
                atomicExch(&(diag).header->assert_ready, 1u);                  \
            }                                                                  \
        }                                                                      \
    } while (0)

#define DIAG_ASSERT(diag, cond)                                                  \
    DIAG_ASSERT4(diag, cond, 0.0f, 0.0f, 0.0f, 0.0f)

#define DIAG_BOUNDS_CHECK(diag, index, count)                                   \
    do {                                                                       \
        unsigned _ppf_i = (unsigned)(index);                                   \
        unsigned _ppf_n = (unsigned)(count);                                   \
        DIAG_ASSERT4(diag, _ppf_i < _ppf_n, (float)_ppf_i, (float)_ppf_n,      \
                    0.0f, 0.0f);                                              \
    } while (0)

// Guarded for the same reason as DIAG_ASSERT4: with no channel attached the
// cursor increment is a null device write, so a trace site would crash a build
// that merely forgot to create the channel.
#define DIAG_TRACE(diag, id, p0, p1, p2, p3)                                   \
    do {                                                                       \
        if ((diag).header == nullptr) {                                        \
            break;                                                             \
        }                                                                      \
        unsigned _ppf_slot = atomicAdd(&(diag).header->ring_cursor, 1u);       \
        if (_ppf_slot < (diag).header->ring_capacity) {                        \
            CUDA_DIAG_FILL(&(diag).records[1u + _ppf_slot], id, diag, p0, \
                               p1, p2, p3);                                    \
        } else {                                                               \
            atomicAdd(&(diag).header->ring_dropped, 1u);                       \
        }                                                                      \
    } while (0)

#endif
