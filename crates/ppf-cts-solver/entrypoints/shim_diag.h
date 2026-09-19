// File: entrypoints/shim_diag.h
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// THE CHANNEL IS NOT HERE ANY MORE. `ChunkDiag`, `DiagHandle` and the
// `DIAG_ASSERT4` family moved into `src/kernels/seam/seam_host.h`, which every
// shim already reaches through `data.hpp`.
//
// They moved because of WHO CAN SEE THEM. A neutral kernel body reaches the
// seam and reaches nothing under `entrypoints/`, so a body with an invariant to
// report needs the handle's name to come from the seam: it takes a
// `DiagHandle` by value, hands it to `DIAG_ASSERT4`, and the entry's
// `[[seam::diag]]` lane threads each target's own channel into that position.
// The four contact and three collision visitors are what needed it and are
// generated entry points because of it.
//
// This file is kept as the note above rather than deleted so a reader who
// follows an old include, or greps for the record beside its callers, is told
// where it went instead of finding nothing.

#ifndef SHIM_DIAG_H
#define SHIM_DIAG_H

#include "../src/kernels/data.hpp"

#endif // SHIM_DIAG_H
