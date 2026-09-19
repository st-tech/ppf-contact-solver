// File: rocm_probe.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Host-callable C ABI runtime probe backing the ROCm arm of `check_gpu`
// (crates/ppf-cts-core/src/utils.rs), the shape `cpp/metal_probe.mm` beside it
// already has. `check_gpu` is a pre-flight capability query the server runs
// BEFORE the real backend is ever loaded, so this file deliberately does not
// depend on, or duplicate, crates/ppf-cts-compute/rocm's own context setup. It
// asks the two device-level questions that arm needs and stops.
//
// IT INCLUDES THE REAL HIP HEADERS RATHER THAN DECLARING WHAT IT NEEDS, and
// that is rule (1b) rather than convenience: `gcnArchName` sits inside
// `hipDeviceProp_t`, a large struct whose layout has already changed once
// (`hip_runtime_api.h` carries `#define hipGetDeviceProperties
// hipGetDevicePropertiesR0600`, and the runtime still exports the R0000 form
// beside it). A hand-written mirror of that struct is two declarations that can
// disagree, and the disagreement would be a wrong string read out of the wrong
// offset rather than a link error.
//
// WHY IT ASKS A DIFFERENT QUESTION ON EACH PLATFORM. `HIP_PLATFORM=nvidia`
// compiles the same HIP through nvcc, where the device is an NVIDIA GPU and
// `gcnArchName` names nothing this project ships code objects for. Gating that
// build on SUPPORTED_GFX would refuse every machine it is meant to run on, so
// the arch is reported and the gate is left to the caller, which applies it
// only where the library is the AMD one. That staging platform establishes that
// this path compiles and runs, never that any part named in SUPPORTED_GFX has
// executed a kernel.
//
// NOTHING HERE DECIDES ANYTHING. It reports what it saw and lets utils.rs
// phrase the refusal, because the wording is a host contract: a rejection must
// not be readable as a tested-hardware claim.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstring>

extern "C" {

// Reports, without deciding:
//
//   * return value: 0 when the query completed, non-zero when the HIP runtime
//     could not be asked at all. A non-zero return with `*out_count == 0` is a
//     broken or absent runtime, which on a shipped distribution means the
//     payload is incomplete rather than that the machine has no GPU.
//   * `*out_count`: how many devices this build can see. Zero is the ordinary
//     "no AMD GPU here" answer on a machine with no amdgpu driver or no
//     /dev/kfd, and it is not an error.
//   * `out_arch` / `out_arch_cap`: device 0's `gcnArchName`, truncated and
//     NUL-terminated into the caller's buffer. It carries the target-id feature
//     suffix the device reports (`gfx90a:sramecc+:xnack-`); trimming that is the
//     caller's job, since what to do with the features is a policy question.
//     Left empty when there is no device.
//   * `out_status`: the hipError_t as an int, so a caller can tell "no device"
//     from "the driver is there but refused", which are different user actions.
int ppf_cts_core_rocm_probe(int *out_count, char *out_arch, size_t out_arch_cap,
                            int *out_status) {
    if (out_count != nullptr) {
        *out_count = 0;
    }
    if (out_arch != nullptr && out_arch_cap > 0) {
        out_arch[0] = '\0';
    }
    if (out_status != nullptr) {
        *out_status = 0;
    }

    int count = 0;
    hipError_t status = hipGetDeviceCount(&count);
    if (out_status != nullptr) {
        *out_status = static_cast<int>(status);
    }
    // hipErrorNoDevice is the ordinary answer on a machine with no AMD GPU, and
    // it arrives with count already 0. Anything else that is not success is a
    // runtime that could not answer, which the caller reports differently.
    if (status != hipSuccess) {
        return (status == hipErrorNoDevice) ? 0 : 1;
    }
    if (out_count != nullptr) {
        *out_count = count;
    }
    if (count <= 0) {
        return 0;
    }

    // Device 0 only. `check_gpu` gates the device the solver will actually
    // launch on, and the backend takes device 0 unless the environment narrows
    // the visible set, in which case device 0 is already the narrowed one.
    hipDeviceProp_t properties;
    std::memset(&properties, 0, sizeof(properties));
    status = hipGetDeviceProperties(&properties, 0);
    if (out_status != nullptr) {
        *out_status = static_cast<int>(status);
    }
    if (status != hipSuccess) {
        return 1;
    }

#if defined(__HIP_PLATFORM_AMD__)
    const char *arch = properties.gcnArchName;
#else
    // On the NVIDIA staging platform there is no gcnArchName worth reading, so
    // the device NAME is reported instead. The caller does not gate on it; see
    // the header.
    const char *arch = properties.name;
#endif
    if (out_arch != nullptr && out_arch_cap > 0 && arch != nullptr) {
        std::strncpy(out_arch, arch, out_arch_cap - 1);
        out_arch[out_arch_cap - 1] = '\0';
    }
    return 0;
}

}  // extern "C"
