// File: seam.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// THE BACKEND SEAM, IN ONE PLACE.
//
// A kernel body under src/kernels is written once and compiled from identical bytes
// by three compilers: nvcc for the device, a host C++ compiler for the oracles
// and the regression tests, and the Metal shader compiler. Those three disagree
// on how to spell an inline annotation, an address space, a square root, an
// atomic and a few intrinsics. Each of those spellings is named by an SM_ macro,
// and each target defines every one of them. A kernel body names the macro and
// carries no preprocessor conditional itself.
//
// THIS FILE HOLDS THE ONLY BACKEND SELECTION. The `#ifdef __CUDACC__` below is
// the single switch that picks a prologue, and NO KERNEL BODY branches on the
// compiler: every *.kernel.cpp under src/kernels carries zero preprocessor
// directives beyond `#pragma once` and a quoted `#include`, which
// ppf-cts-compute/seam/kernelgen.py enforces by rejecting anything else.
//
// Four other headers under src/kernels do carry a compiler conditional, and
// none of them is a kernel body. `float_math.hpp` and `float.hpp` test
// `__CUDA_ARCH__`, which selects between the device and the host arm WITHIN one
// nvcc compilation and is therefore a different question from which backend is
// being built. `linalg/la_traits.hpp` tests
// `__METAL_VERSION__` because the shader compiler needs a different scalar trait
// set, and `contact/intersect_core.hpp`, `contact/intersect_policy.hpp` and
// `energy/model/pdrd_polar.hpp` each define an annotation for themselves so they
// need no include to define it away for a plain host compiler. Adding to that
// list is a decision to argue for; adding a conditional to a kernel body is not.
//
// MSL REACHES THIS FILE ONLY TO BE SENT AWAY AGAIN. The Metal driver is handed
// one concatenated string with no filesystem behind it, so a shader compiled at
// RUN TIME resolves no include and never opens this file. An OFFLINE
// translation unit handed to `xcrun metal` does have a filesystem, and there a
// kernel body naming this header reaches it like any other; the first arm of
// the selection below is what stops it choosing a prologue for a compiler that
// already has one. The Metal prologue is emitted by
// metal/shader_compiler.mm (`kMslMacroSeam`), which is that backend's copy
// of the same table. A name added here must be added there in the same change,
// or a converted kernel body compiles on two backends out of three and the
// third fails at run time, after every build leg has gone green.
//
// WHERE THIS IS INCLUDED. Once, from data.hpp, which is the one header every
// CUDA translation unit and every host translation unit that reads a shared
// kernel body already includes first, and which the shared bodies already
// depend on for `Vec3f` and its siblings.
//
// WHAT IS DELIBERATELY ABSENT, and must stay absent:
//
//   SM_INLINE. It is the one seam name whose meaning is not agreed across the
//   headers: most bodies want device only, and a few are also called from the
//   host and want both execution spaces. Defining it centrally would pick one
//   answer for all of them. The prologues offer the spellings under distinct
//   names instead (SM_INLINE_DEVICE and SM_INLINE_DEVICE_HOST), so a body
//   selects by name. A neutral kernel body spells the same distinction as
//   [[seam::device_fn]] or [[seam::host_device_fn]] and reaches no macro at
//   all; the remaining users of the bare name are headers that define it for
//   themselves (contact/intersect_policy.hpp, energy/model/baraffwitkin.hpp).
//
//   SM_MSL_CONCAT. It is a role marker meaning "the Metal shader compiler is
//   reading this", and it must remain undefined on both targets here. No
//   *.kernel.cpp kernel body tests it: a quoted `#include` there is
//   unconditional, and metal/shader_compiler.mm neutralizes the line as it
//   splices the segment, so the backend difference is spent in the backend. The
//   headers that still test it gate more than a quoted include, an angle
//   `<cmath>` among them, which the shader compiler cannot serve either. This
//   file is one of them, for the reason the selection block below states.
//
//   The MSL-only names (SM_CONSTANT, SM_MULHI, SM_AS_UINT, SM_AS_FLOAT,
//   SM_ATOMIC_CAS_UINT, SM_ATOMIC_MAX_UINT, SM_ATOMIC_ADD_FLOAT_VIA_CAS,
//   SM_SEG8_REDUCE, SM_SEG8_REDUCE_DOWN, SM_SIMD_SUM, SM_LANE_ID). The Metal
//   prologue defines them; no shared kernel body under src/kernels names any of
//   them today. Giving them CUDA and host bodies here would add definitions no
//   compiler checks against a call site, so the first use would be the first
//   test. Add one at the point a kernel needs it, not before.

// THE ONE BACKEND SELECTION, AND WHY ONE HALF IS A BARE FILENAME. `seam_host.h`
// sits beside this file, in the neutral tree, because a host C++ compiler is
// what every neutral body is read by first. `seam_cuda.cuh` is nvcc's prologue,
// so by rule (1-0) it lives in ppf-cts-compute with the rest of the CUDA
// target, and it is named WITHOUT a path: the CUDA build puts its own directory
// on the include path, so the neutral tree carries no route into that crate and
// this line resolves for exactly the compiler the guard already selected.
//
// THE MSL ARM COMES FIRST AND SELECTS NOTHING, which is the point of it.
// SM_MSL_CONCAT is the role marker the Metal prologue defines, so its presence
// means that prologue is already in scope and every SM_ name below is already
// spelled. Choosing either of the other two there would redefine the whole
// table for a compiler neither was written for, and `seam_host.h` alone would
// pull <cmath>, which the shader compiler cannot serve. The arm costs the
// run-time path nothing, because the run-time path never opens this file.
//
// THE HIP ARM COMES AFTER THE CUDA ONE, AND THE ORDER IS LOAD-BEARING. Under
// `HIP_PLATFORM=nvidia` hipcc invokes nvcc, so `__CUDACC__` and `__HIPCC__` are
// BOTH defined, and the CUDA prologue is the correct one there because the
// intrinsics available are CUDA's. Testing HIP first would hand an NVIDIA
// compile the AMD spellings. `seam_hip.hiph` is named without a path for the
// same reason `seam_cuda.cuh` is: it lives in ppf-cts-compute with the rest of
// that target, and the HIP build puts its own directory on the include path, so
// the neutral tree carries no route into that crate.
#if defined(SM_MSL_CONCAT)
#elif defined(__CUDACC__)
#include "seam_cuda.cuh"
#elif defined(__HIPCC__)
#include "seam_hip.hiph"
#else
#include "seam_host.h"
#endif
