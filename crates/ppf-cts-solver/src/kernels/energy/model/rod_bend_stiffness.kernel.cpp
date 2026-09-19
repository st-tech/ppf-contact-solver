// File: rod_bend_stiffness.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The scalar that multiplies a rod's turning-angle force and Hessian: the
// resolution-independent, density-normalized discrete-rod bending coefficient,
// and the two-segment averages the interior vertex forms its material from.
//
// It is separated from rod_bend.kernel.cpp (the per-vertex turning-angle math)
// because it is what the two backends have to agree on BEFORE either evaluates
// the stencil: the same scalar scales the elastic block AND the lagged Rayleigh
// damping block built on top of it, so a backend that formed it its own way
// would differ in both at once. It is also the only quantity in the whole rod
// bending term that reaches no special-function unit, which makes it the one
// output a cross-backend gate can hold to the bit.

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend:
// no preprocessor conditional, no macro of its own, and no spelling that only
// nvcc or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders
// it into the three forms the three compilers read, and the build hands each
// compiler its own form. The one fact a backend cannot infer here is the
// execution space, written as the C++ attribute `[[seam::device_fn]]`; both
// bodies take their arguments by value, so no address space appears.
//
// Everything here is float arithmetic, and the single division goes through
// `fmath::div`, which the backend prologue defines. That is a correctness
// spelling rather than a style: MSL's plain `/` is not correctly rounded even
// at mathMode Safe, while `precise::divide` is in every math mode and CUDA's is
// already. common.hpp is this tree's scalar header and is the only file this
// body includes.
#include "../../common.hpp"

// An interior rod vertex is shared by two segments and each carries its own
// material, so every per-segment quantity the stencil needs is the mean of the
// two. Both the bending stiffness and the bending damping come through here, so
// the two backends cannot come to average them differently, and a `0.5f *`
// product of an exact sum is exact in fp32 for any pair of finite inputs.
[[seam::device_fn]] inline float rod_bend_segment_average(float first,
                                                             float second) {
    return 0.5f * (first + second);
}

// The per-vertex bending stiffness.
//
// Resolution independence: the convergent per-vertex stiffness is k = B / l,
// where l is the Voronoi length (half the two incident rest lengths): the
// continuum energy int 0.5*B*kappa^2 ds discretizes at an interior vertex as
// 0.5*B*(phi/l)^2*l = 0.5*(B/l)*phi^2, with phi the turning angle the rod
// bending energy measures.
//
// Density normalization: B = bend * linear_density is the density-normalized
// flexural rigidity, so `bend` alone sets the bent shape and density stays a
// free knob, which is the same normalization the shell hinge applies through
// areal density.
//
// The 1/l^2 is what makes the shape mesh independent, and both powers are
// load-bearing. Linear density is mass/l, so B/l expands to bend*mass/l^2; and
// the lumped vertex mass is itself proportional to l, since builder.rs sums
// half of each incident rod segment's mass. A coefficient of bend*mass alone
// therefore scales as l where the discretization needs 1/l, leaving a rod four
// times floppier for every halving of its segment length, so two resolutions of
// one physical rod would not describe the same rod.
//
// REF_LENGTH sets only the numeric range of the user `bend` parameter, as
// BEND_SCALE does for the shell; it does not affect mesh independence. It is
// anchored at 1 cm, which is the segment length the rod scenes in examples/ sit
// nearest, so established `bend` values keep their meaning there.
//
// `bend` and `mass` are non-negative (scene.rs, builder.rs) and the guard keeps
// the divisor away from zero, so the result can never turn negative and flip
// the sign of the stencil block: SPD-by-assembly holds.
[[seam::device_fn]] inline float rod_bend_stiffness(float bend, float mass,
                                                       float length0,
                                                       float length1) {
    const float REF_LENGTH = 1e-2f;
    const float voronoi = 0.5f * (length0 + length1);
    return voronoi > 0.0f
               ? fmath::div(bend * mass * (REF_LENGTH * REF_LENGTH),
                          voronoi * voronoi)
               : 0.0f;
}

// The entry point, declared once and rendered for four targets. Four element
// gathers into one element scatter, the rod analogue of the shell form.
// `rod_bend_segment_average` beside it declares none: it is a scalar
// helper the driver calls per element rather than a kernel it dispatches.
[[seam::entry(count)]] void rod_bend_stiffness(
    const float *bend,
    const float *mass,
    const float *length0,
    const float *length1,
    float *stiffness,
    unsigned count);
