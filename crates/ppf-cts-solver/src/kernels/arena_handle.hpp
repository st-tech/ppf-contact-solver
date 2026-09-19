// File: arena_handle.hpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// THE NAME `ArenaHandle`, AND NOT A SECOND DECLARATION OF WHAT IT NAMES.
//
// A device allocation crosses every boundary in this project as (arena id, byte
// offset) rather than as a pointer, and `seam/backend_abi.h` is where that
// type is declared: one declaration, compiled by nvcc, by the Metal shader
// compiler's host half, by a host C++ compiler and mirrored once in Rust. This
// file used to carry a second definition of the same four fields, which is the
// mirror pair the seam exists to remove, and it stopped being merely untidy the
// moment one translation unit read both: the two are distinct types to a C++
// compiler however identical their layout, so a call passing one where the
// other is expected does not compile, and a translation unit including both
// headers does not compile at all.
//
// So the alias is the whole file. Every existing consumer keeps its spelling,
// and `ArenaHandle` and `BeHandle` are now one type rather than two that agree.

#ifndef ARENA_HANDLE_HPP
#define ARENA_HANDLE_HPP

#include "seam/backend_abi.h"

typedef BeHandle ArenaHandle;

// Kept, and not redundant with the ABI header's own pins. Those state what the
// seam requires; these state what the code spelling `ArenaHandle` requires, and
// the alignment one in particular is what a generated entry point's offset
// arithmetic depends on.
static_assert(sizeof(ArenaHandle) == 16, "arena handle ABI changed");
static_assert(alignof(ArenaHandle) == alignof(unsigned),
              "arena handle alignment changed");

// AN OWNING ALLOCATION, SPELLED THE WAY A GENERATED ENTRY POINT READS IT.
//
// A record field addresses an allocation by (arena, offset) handle rather than
// by pointer, and an owning vector already holds exactly those four values, so
// this is the whole conversion rather than a second bookkeeping path that could
// disagree with the pointer beside it.
//
// IT IS A TEMPLATE OVER THE CONTAINER, NOT OVER ITS ELEMENT, so that this
// header names no vector type: `ArenaVec` is the compute crate's, and a solver
// header may not reach for it. Any owning allocation carrying `data.arena`,
// `data.off`, `size` and `allocated` satisfies it, which is what every caller
// passes today.
template <class V> inline ArenaHandle handle_of(const V &v) {
    ArenaHandle out{};
    out.arena = v.data.arena;
    out.off = v.data.off;
    out.size = v.size;
    out.allocated = v.allocated;
    return out;
}

#endif
