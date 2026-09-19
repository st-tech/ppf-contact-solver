// File: diagnostic_record.hpp
// Code: Claude Code and GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// THE DIAGNOSTIC RECORD'S NAMES, AND NOT A SECOND DEFINITION OF THE RECORD.
//
// `BeDiagRecord` is declared once, in `seam/backend_abi.h`, because it is a
// type that crosses the boundary between a driver and a backend library and a
// second declaration of one is what that header exists to prevent. This file
// used to define it again with the same four fields, which compiled only while
// no translation unit read both; the first one that did got a redefinition
// error rather than a silent disagreement, which is the right direction to fail
// but is still a failure the alias below removes.
//
// What remains here is the SOLVER's own vocabulary over that record: the two
// reserved ids the ABI fixes, spelled as `constexpr` so device code can use
// them in a constant expression, and the trace-id bases this solver assigns
// above `PPF_BE_DIAG_ID_FIRST_USER`. The bases are not in the ABI header and
// must not be: which trace sites exist is a statement about what this solver
// computes.

#ifndef DIAGNOSTIC_RECORD_HPP
#define DIAGNOSTIC_RECORD_HPP

#include <cstdint>

#include "seam/backend_abi.h"

constexpr uint32_t DIAG_ID_ASSERT = PPF_BE_DIAG_ID_ASSERT;
constexpr uint32_t DIAG_ID_BOUNDS = PPF_BE_DIAG_ID_BOUNDS;
constexpr uint32_t DIAG_ID_FIRST_USER = PPF_BE_DIAG_ID_FIRST_USER;
// This solver's own first trace family, at and above the ABI's first user id so
// it cannot be confused with a violation.
constexpr uint32_t DIAG_ID_CCD_OVERLAP_BASE = 32u;
static_assert(DIAG_ID_CCD_OVERLAP_BASE >= DIAG_ID_FIRST_USER,
              "a trace id below the ABI's first user id would be read as a "
              "violation");

static_assert(sizeof(BeDiagRecord) == 32, "diagnostic record ABI changed");
static_assert(alignof(BeDiagRecord) == 4,
              "diagnostic record alignment changed");

#endif
