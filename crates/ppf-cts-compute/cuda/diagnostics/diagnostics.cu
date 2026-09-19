// File: diagnostics.cu
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "diagnostics/diagnostics.hpp"

#include "../cuda_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace diagnostics {

constexpr unsigned kMaxRingCapacity = 1u << 20;
Channel global_channel;

bool create(unsigned ring_capacity, Channel *out, std::string *error) {
    if (!out || ring_capacity == 0 || ring_capacity > kMaxRingCapacity) {
        if (error) {
            *error = "diagnostic channel needs an output and a ring capacity "
                     "between 1 and 1048576";
        }
        return false;
    }
    const size_t bytes =
        sizeof(Header) +
        (static_cast<size_t>(ring_capacity) + 1) * sizeof(BeDiagRecord);
    // MAPPED HOST MEMORY, so the per-boundary drain is a host read rather than
    // a device-to-host copy. See the Channel comment for why this costs nothing
    // on the hot path.
    void *host = nullptr;
    const cudaError_t host_code =
        cudaHostAlloc(&host, bytes, cudaHostAllocMapped);
    if (host_code != cudaSuccess) {
        if (error) {
            *error = std::string("the diagnostic channel could not be mapped: ") +
                     cudaGetErrorString(host_code);
        }
        return false;
    }
    void *device = nullptr;
    const cudaError_t map_code = cudaHostGetDevicePointer(&device, host, 0);
    if (map_code != cudaSuccess) {
        cudaFreeHost(host);
        if (error) {
            *error =
                std::string("the diagnostic channel has no device pointer: ") +
                cudaGetErrorString(map_code);
        }
        return false;
    }
    out->host_base = static_cast<unsigned char *>(host);
    out->device_base = static_cast<unsigned char *>(device);
    out->bytes = bytes;
    out->ring_capacity = ring_capacity;
    // THE WHOLE ALLOCATION STARTS ZEROED, ring included: `reset` clears only the
    // header, so a record slot is never cleared again and a reader must not be
    // handed whatever the allocator happened to return.
    std::memset(out->host_base, 0, bytes);
    reset(*out);
    return true;
}

void destroy(Channel *channel) {
    if (!channel || channel->host_base == nullptr) {
        return;
    }
    CUDA_HANDLE_ERROR(cudaFreeHost(channel->host_base));
    channel->host_base = nullptr;
    channel->device_base = nullptr;
    channel->bytes = 0;
    channel->ring_capacity = 0;
}

void reset(const Channel &channel) {
    if (channel.host_base == nullptr) {
        return;
    }
    // PLAIN HOST STORES. These pages are mapped, so clearing the header is not
    // a memset the device has to be told about and writing the capacity back is
    // not an upload.
    auto *header = reinterpret_cast<Header *>(channel.host_base);
    std::memset(header, 0, sizeof(Header));
    header->ring_capacity = channel.ring_capacity;
}

bool read(const Channel &channel, Readback *out, std::string *error) {
    if (!out) {
        if (error) {
            *error = "diagnostic readback output is null";
        }
        return false;
    }
    // A HOST READ OF MAPPED PAGES, NOT A TRANSFER. This runs once per boundary
    // whether or not a kernel recorded anything; out of device memory it cost a
    // copy the size of the whole ring every time, measured on `drape` at 3
    // frames as 8,277 copies of about 33 KB, 276 MB and roughly 85 percent of
    // every device-to-host memcpy the process issued. The header carries the
    // counts, so the records are worth LOOKING at only when it says there are
    // some, and neither step copies anything now.
    if (channel.host_base == nullptr) {
        if (error) {
            *error = "the diagnostic channel was not created";
        }
        return false;
    }
    Header header{};
    std::memcpy(&header, channel.host_base, sizeof(header));
    if ((header.assert_claim == 0) != (header.assert_ready == 0)) {
        if (error) {
            *error = "diagnostic assert claim/ready accounting is inconsistent";
        }
        return false;
    }
    const unsigned written =
        header.ring_cursor < header.ring_capacity ? header.ring_cursor
                                                  : header.ring_capacity;
    if (header.ring_dropped != header.ring_cursor - written) {
        if (error) {
            *error = "diagnostic ring overflow accounting is inconsistent";
        }
        return false;
    }
    out->assert_hit = header.assert_ready != 0;
    out->fail_count = header.fail_count;
    out->ring_written = written;
    out->ring_dropped = header.ring_dropped;
    // THE RECORDS ARE READ IN PLACE. They are in mapped host memory, so there
    // is nothing to copy in and nothing to size: a clean boundary touches the
    // header and stops, and a dirty one walks the records it says exist.
    auto *records = reinterpret_cast<const BeDiagRecord *>(
        channel.host_base + sizeof(Header));
    out->assert_record = records[0];
    out->ring.assign(records + 1, records + 1 + written);
    return true;
}

bool create_global(unsigned ring_capacity, std::string *error) {
    if (global_channel.host_base != nullptr) {
        if (error) {
            *error = "global diagnostic channel was created twice";
        }
        return false;
    }
    if (!create(ring_capacity, &global_channel, error)) {
        return false;
    }
    return true;
}

const Channel &global() { return global_channel; }

// THE FILE TABLE, which is what turns a device report's id back into a path.
//
// A device record cannot carry a string, so a `DIAG_ASSERT` site records
// `file_id(__FILE__)`. A hash is not reversible, so without this the host
// prints the number and the reader has to find the failing check by hand,
// which is most of what the report exists to give them.
//
// The rows are GENERATED, one per neutral source, by the same function that
// writes each rendering's `#line` directive, so the string hashed here and the
// string `__FILE__` expands to there cannot drift apart.
namespace {
struct DiagFile {
    // TWO IDS, because the two backends disagree about what `__FILE__` is. A
    // rendering's `#line` directive names the neutral source absolutely, which
    // is what nvcc sees; the ROCm recipe compiles with
    // `-fmacro-prefix-map=<kernel root>/=` and sees the path relative to that
    // root. The row carries the hash of each so it serves either.
    unsigned absolute_id;
    unsigned relative_id;
    const char *path;
};
const DiagFile kDiagFiles[] = {
#include "diag_files.inc"   // generated; $(KERNELGEN_DIR) is on the include path
};
}  // namespace

const char *file_path(unsigned id) {
    for (const DiagFile &row : kDiagFiles) {
        if (row.absolute_id == id || row.relative_id == id) {
            return row.path;
        }
    }
    // NOT A GUESS. An id this library did not compile in belongs to a file it
    // does not carry, and inventing a plausible path would put the wrong source
    // in a report that exists to name the right one.
    return nullptr;
}


// THE CHANNEL PROVES ITSELF, on the device, when asked to.
//
// Everything else about the diagnostic channel is checked by reading code: the
// generator hands every entry `diagnostics::global()`, the backend attaches
// that channel, and the macro guards its header. None of that is evidence that
// a record written by a kernel on THIS machine reaches the host, and the two
// defects this exists to catch were both invisible to every build and every
// gate. A private channel drained instead of the global one is green
// everywhere; an unattached channel is green until the first check fires, and
// then it is an illegal address naming no cause.
//
// So this fires a real assert from a real kernel against the channel the
// production launchers use, and reports whether the record came back. It is
// the only check of the four that could fail on a machine where the source is
// correct, which is why it is worth a kernel launch.
// Spelled the way every generated entry point is spelled: the channel crosses
// as a `Channel` through `DIAG_ARG` and is bound on the device with
// `DIAG_BIND`. Taking the bound handle as a parameter instead would test a
// path no kernel uses.
namespace {
__global__ void selftest_kernel(DIAG_ARG) {
    DiagHandle diag = DIAG_BIND(0);
    DIAG_ASSERT4(diag, false, 1.0f, 2.0f, 3.0f, 4.0f);
}
}  // namespace

bool selftest_global(std::string *error) {
    if (global_channel.device_base == nullptr) {
        if (error) {
            *error = "no diagnostic channel is attached, so a failed device "
                     "check would be recorded nowhere";
        }
        return false;
    }
    reset(global_channel);
    selftest_kernel<<<1, 1>>>(global_channel);
    const cudaError_t launched = cudaDeviceSynchronize();
    if (launched != cudaSuccess) {
        if (error) {
            *error = std::string("a deliberately failing device check was not "
                                 "survivable: ") +
                     cudaGetErrorString(launched);
        }
        return false;
    }
    Readback seen;
    if (!read(global_channel, &seen, error)) {
        return false;
    }
    // Left clean for the run that follows, whichever way this went.
    reset(global_channel);
    if (!seen.assert_hit || seen.fail_count != 1) {
        if (error) {
            *error = "a deliberately failing device check did not reach the "
                     "host: the channel the kernels write is not the channel "
                     "the host reads";
        }
        return false;
    }
    return true;
}


void reset_global() { reset(global_channel); }

bool read_global(Readback *out, std::string *error) {
    return read(global_channel, out, error);
}

void reset_assert_global() {
    if (global_channel.host_base == nullptr) {
        return;
    }
    std::memset(global_channel.host_base + offsetof(Header, assert_claim), 0,
                3 * sizeof(unsigned));
}

void check_assert_global() {
    if (global_channel.host_base == nullptr) {
        return;
    }
    Header header{};
    BeDiagRecord record{};
    unsigned char *base = global_channel.host_base;
    std::memcpy(&header, base, sizeof(header));
    if (header.assert_ready == 0) {
        return;
    }
    std::memcpy(&record, base + sizeof(Header), sizeof(record));
    std::fprintf(stderr,
                 "### cuda device assertion: file-id=%08x line=%u thread=%u "
                 "payload=[%g,%g,%g,%g], failures=%u\n",
                 record.file_id, record.line, record.thread_id,
                 record.payload[0], record.payload[1], record.payload[2],
                 record.payload[3], header.fail_count);
    g_ppf_fatal_code = 3;
    std::exit(1);
}

} // namespace diagnostics
