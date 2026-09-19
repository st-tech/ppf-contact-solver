// File: type_aliases.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// THE PROJECT'S NAMES FOR `linalg::SMat`, AND NOTHING ELSE.
//
// Every name below is a spelling for a `linalg::SMat` or a small aggregate of
// them: `Vec3f` for a 3x1 of float, `Mat9x9f` for a 9x9 of float, `DiffTable2`
// for a pair of derivative tables. There is no constant here and no function
// and nothing that reads the scene, and every include is QUOTED and resolves
// inside this tree, which is what lets the file be read by a compiler that can
// serve no standard library.
//
// WHY IT IS A FILE RATHER THAN THE TOP OF data.hpp.
// data.hpp is what a CUDA or host translation unit includes to get the whole
// shared vocabulary, and it reaches `vec/vec.hpp` for the view types the
// dataset is built out of. Those views spell a raw `T *`, which the Metal
// shader compiler refuses outright: MSL has no default address space, so every
// pointer and every reference must carry one. So data.hpp cannot be read by
// that compiler at all, and the aliases, which can, were unreachable behind it.
//
// WHO NEEDS THEM SEPARATELY. A GENERATED ENTRY POINT compiled on its own
// (ppf-cts-compute/metal/Makefile's `entry-check`) is one neutral body plus the
// argument record around it, and the body names these types. It needs the
// vocabulary and none of the dataset, and this file is exactly that boundary.
// The shipped Metal shader splices it as a segment for the same reason, so the
// aliases the shader sees are these and not a transcription of them.
//
// `data_records.hpp` is the second half of that boundary and is split from
// data.hpp on the same line: the spellings here, the fixed-size records built
// out of them there. data.hpp includes both, so a CUDA or host translation unit
// still reaches the whole vocabulary through that one header.

#ifndef LINALG_TYPE_ALIASES_HPP
#define LINALG_TYPE_ALIASES_HPP

// WHAT THIS FILE IS A SPELLING OF, NAMED RATHER THAN ASSUMED. Every alias below
// is a `linalg::SMat`, so the header that declares it is included here instead
// of being expected from whoever read this one. It is quoted, so the run-time
// assembler neutralizes the line as it splices and the shipped shader still
// takes the declaration from its segment order; what it buys is that an OFFLINE
// unit naming this header does not have to be handed that one first.
#include "smat.hpp"

using linalg::map;
template <class T, unsigned N> using SVec = linalg::SVec<T, N>;
template <unsigned N> using SVecf = SVec<float, N>;
template <unsigned N> using SVecu = SVec<unsigned, N>;

template <class T> using Vec1 = SVec<T, 1>;
template <class T> using Vec2 = SVec<T, 2>;
template <class T> using Vec3 = SVec<T, 3>;
template <class T> using Vec4 = SVec<T, 4>;
template <class T> using Vec6 = SVec<T, 6>;
template <class T> using Vec9 = SVec<T, 9>;
template <class T> using Vec12 = SVec<T, 12>;

using Vec1f = Vec1<float>;
using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec4f = Vec4<float>;
using Vec6f = Vec6<float>;
using Vec9f = Vec9<float>;
using Vec12f = Vec12<float>;

using Vec1u = Vec1<unsigned>;
using Vec2u = Vec2<unsigned>;
using Vec3u = Vec3<unsigned>;
using Vec4u = Vec4<unsigned>;
using Vec6u = Vec6<unsigned>;

template <class T, unsigned R, unsigned C>
using SMat = linalg::SMat<T, R, C>;
template <unsigned R, unsigned C> using SMatf = linalg::SMat<float, R, C>;
template <class T> using Mat3x2 = SMat<T, 3, 2>;
template <class T> using Mat2x2 = SMat<T, 2, 2>;
template <class T> using Mat2x3 = SMat<T, 2, 3>;
template <class T> using Mat3x3 = SMat<T, 3, 3>;
template <class T> using Mat3x4 = SMat<T, 3, 4>;
template <class T> using Mat3x5 = SMat<T, 3, 5>;
template <class T> using Mat3x6 = SMat<T, 3, 6>;
template <class T> using Mat4x3 = SMat<T, 4, 3>;
template <class T> using Mat4x4 = SMat<T, 4, 4>;
template <class T> using Mat3x9 = SMat<T, 3, 9>;
template <class T> using Mat6x6 = SMat<T, 6, 6>;
template <class T> using Mat6x9 = SMat<T, 6, 9>;
template <class T> using Mat9x9 = SMat<T, 9, 9>;
template <class T> using Mat9x12 = SMat<T, 9, 12>;
template <class T> using Mat12x12 = SMat<T, 12, 12>;
// The cross-stitch's 18x18, one 3x3 block per pair of its six barycentric
// slots. It carries a NAME rather than being spelled `SMatf<18, 18>` at its use
// sites because a generated entry point's parameter type must arrive under a
// single name: the transcompiler splits a parameter list on its top-level
// commas and cannot tell those from the commas inside `<...>`, and it refuses
// the template spelling saying so.
template <class T> using Mat18x18 = SMat<T, 18, 18>;

using Mat2x3f = Mat2x3<float>;
using Mat3x2f = Mat3x2<float>;
using Mat2x2f = Mat2x2<float>;
using Mat3x3f = Mat3x3<float>;
using Mat3x4f = Mat3x4<float>;
using Mat3x5f = Mat3x5<float>;
using Mat3x6f = Mat3x6<float>;
using Mat4x3f = Mat4x3<float>;
using Mat4x4f = Mat4x4<float>;
using Mat3x9f = Mat3x9<float>;
using Mat6x6f = Mat6x6<float>;
using Mat6x9f = Mat6x9<float>;
using Mat9x9f = Mat9x9<float>;
using Mat9x12f = Mat9x12<float>;
using Mat12x12f = Mat12x12<float>;
using Mat18x18f = Mat18x18<float>;

// THE DIFFERENTIATION AGGREGATE, which is a spelling too: a scalar energy's
// first and second derivatives with respect to N invariants, carried together
// because every material model returns both. It is here rather than beside the
// dataset for the same reason the aliases are: the material headers name it
// (`energy/model/{arap,stvk,snhk}.hpp`), those headers are read by the Metal
// shader compiler, and nothing about it needs the dataset.
template <unsigned N> struct DiffTable {
    SVecf<N> deda;
    SMatf<N, N> d2ed2a;
};

using DiffTable2 = DiffTable<2>;
using DiffTable3 = DiffTable<3>;

#endif  // LINALG_TYPE_ALIASES_HPP
