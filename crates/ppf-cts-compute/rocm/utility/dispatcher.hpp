// File: dispatcher.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// THE DISPATCH PRIMITIVE'S INTERFACE FOR THE CUDA TARGET.
//
// Launching a body over a range is one of the five verbs this crate exists for,
// so the primitive and the name a caller reaches it by both live here. This
// file is the name: `dispatcher.cu` beside it holds the two kernels and the
// four macros, and is an INCLUDE rather than a translation unit, which is why
// the recipe compiles it into no object of its own and eleven callers pull it
// in through here instead.

#ifndef CUDA_DISPATCHER_HPP
#define CUDA_DISPATCHER_HPP

#include "utility/dispatcher.cu"

#endif
