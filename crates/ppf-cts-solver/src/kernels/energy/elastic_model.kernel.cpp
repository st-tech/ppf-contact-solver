// File: elastic_model.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts.
//
// THE ELASTIC MODEL IDS, as unsigned enumerators rather than as an enumeration
// type.
//
// `enum class Model` in data_records.hpp is the source of this numbering: each
// value below is the position of the matching enumerator there, and the two
// must move together. The list is restated here because the Metal shader is
// handed one concatenated string in which a record header need not appear, so a
// shader knows an element's model only as the `unsigned` its parameter record
// carries, and because MSL rejects a `constexpr` variable at program scope,
// which is why these are enumerators with a fixed underlying type rather than
// constants.
//
// NOTHING CHECKS THE TWO LISTS AGAINST EACH OTHER AT COMPILE TIME, because the
// enumerators here are untyped `unsigned` precisely so a shader can compare
// them against a record field. A value renumbered on one side and not the other
// therefore surfaces as a material chosen wrongly at run time, so edit both in
// the same change.
//
// ELASTIC_MODEL_PDRD NAMES A MODEL WITH NO ELASTIC ENERGY AT ALL. A rigid
// body's faces and tets carry it with `mu == 0`, because their shape is held by
// the reduced rigid solve, so they leave every element body at its `mu > 0`
// gate before any dispatch. A backend that validates model ids must ADMIT it,
// or it refuses every scene containing a rigid body.
enum : unsigned {
    ELASTIC_MODEL_ARAP = 0u,
    ELASTIC_MODEL_STVK = 1u,
    ELASTIC_MODEL_BARAFF_WITKIN = 2u,
    ELASTIC_MODEL_SNHK = 3u,
    ELASTIC_MODEL_PDRD = 4u
};
