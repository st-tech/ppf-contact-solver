// File: tet_convert.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: plain C++ that belongs to no backend, with no
// preprocessor conditional, no macro of its own, and no spelling that only one
// of the three compilers accepts. ppf-cts-compute/seam/kernelgen.py renders it into the
// form each compiler reads. `[[seam::device_fn]]` is the execution space and
// `[[seam::thread]]` is the address space MSL requires on every reference and
// pointer type. It is supplied by the backend prologue: cpp/seam under nvcc and
// on the host, and `kMslMacroSeam` in metal/shader_compiler.mm under MSL.

[[seam::device_fn]] inline Mat3x3f
tet_deformation_gradient(const Vec3f &x0,
                             const Vec3f &x1,
                             const Vec3f &x2,
                             const Vec3f &x3,
                             const Mat3x3f &inverse_rest) {
    Mat3x3f edges;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        edges(dimension, 0) =
            static_cast<float>(x1[dimension] - x0[dimension]);
        edges(dimension, 1) =
            static_cast<float>(x2[dimension] - x0[dimension]);
        edges(dimension, 2) =
            static_cast<float>(x3[dimension] - x0[dimension]);
    }
    return edges * inverse_rest;
}

// THE FOUR GRADIENTS COME BACK AS A MATRIX, NOT THROUGH A POINTER, and the
// reason is an address space rather than taste. A caller inside this file has
// them in thread storage and the generated entry point has them in a device
// buffer; MSL has no way to write one function that takes either, so a body
// taking `[[seam::thread]] Vec3f *` cannot be reached from an entry point at
// all on that backend, while the same declaration compiles on the other two.
// Returning the run makes the shape representable everywhere: `Mat3x4f` is
// column major, so its 48 bytes are exactly the four `Vec3f` the buffer holds,
// in the same order.
[[seam::device_fn]] inline Mat3x4f
tet_shape_gradients(const Mat3x3f &inverse_rest) {
    Vec3f column[4];
    column[0] =
        -inverse_rest.row(0) - inverse_rest.row(1) - inverse_rest.row(2);
    column[1] = inverse_rest.row(0);
    column[2] = inverse_rest.row(1);
    column[3] = inverse_rest.row(2);
    Mat3x4f gradient;
    for (unsigned vertex_index = 0; vertex_index < 4; ++vertex_index) {
        for (unsigned dimension = 0; dimension < 3; ++dimension) {
            gradient(dimension, vertex_index) = column[vertex_index][dimension];
        }
    }
    return gradient;
}

// The entry points, declared once each and rendered for four targets. The two
// converters are two element gathers into one element scatter, as the face pair
// is. The deformation gradient below reads its four positions THROUGH the
// element's own index list, which is the indirect gather; the face twin in
// `utility/face_deformation.kernel.cpp` carries the same shape and states why
// the bound is the entry's work rather than the body's.
// **THE MASS IS APPLIED HERE, NOT IN A SECOND PASS.** `energy.cu` writes
// `mass * convert_force(...)` inside the one lambda that assembles a tet, with
// the result in registers. Scaling in a separate `element_add_scaled` dispatch
// meant materializing the UNSCALED pack to global memory and reading all of it
// back to multiply it, which for the Hessian is 576 bytes a tet each way. The
// multiply is the same multiply by the same mass on the same value: the
// destination it accumulated into opened at zero, so `+= mass * v` and
// `= mass * v` are one expression.
[[seam::entry(force)]]
[[seam::device_fn]] inline Mat3x4f
tet_convert_force(const Mat3x3f &gradient_f,
                      const Mat3x3f &inverse_rest, const float &mass) {
    const Mat3x4f gradient = tet_shape_gradients(inverse_rest);
    Mat3x4f result;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        for (unsigned vertex_index = 0; vertex_index < 4; ++vertex_index) {
            result(dimension, vertex_index) =
                mass * gradient.col(vertex_index).dot(gradient_f.row(dimension));
        }
    }
    return result;
}

[[seam::entry(hessian)]]
[[seam::device_fn]] inline Mat12x12f
tet_convert_hessian(const Mat9x9f &hessian_f,
                        const Mat3x3f &inverse_rest, const float &mass) {
    const Mat3x4f gradient = tet_shape_gradients(inverse_rest);
    Mat12x12f result;
    for (unsigned a = 0; a < 4; ++a) {
        for (unsigned b = 0; b < 4; ++b) {
            Mat3x3f block = Mat3x3f::Zero();
            for (unsigned d = 0; d < 3; ++d) {
                for (unsigned e = 0; e < 3; ++e) {
                    block +=
                        (gradient.col(a)[d] * gradient.col(b)[e]) *
                        hessian_f.template block<3, 3>(3 * d, 3 * e);
                }
            }
            result.template block<3, 3>(3 * a, 3 * b) = mass * block;
        }
    }
    return result;
}

// The shape-function gradients, whose four `Vec3f` per tet are a RUN and not an
// element, which is why `gradient` is a stride rather than a scatter: the body
// returns the four gradients as one column-major `Mat3x4f`, so the entry
// scatters 48 bytes at the tet's own index, which is the same memory the
// launcher this replaces addressed as `out + 4 * t`.
//
// NOTHING IN THIS DRIVER DISPATCHES IT. Its id is declared past `id::COUNT` in
// `src/driver/kernels.rs` with no row in the dispatch table, the arrangement
// `vec_fill` and `vec_combine_indirect` already use, so `decl_of`
// answers `Fault::MissingKernel` by name rather than reading past the table.
// The two converters above reach the same body directly, each computing the
// four gradients into its own thread storage.
[[seam::entry(count)]] void tet_shape_gradients(
    const Mat3x3f *inverse_rest,
    Mat3x4f *gradient,
    unsigned count);

[[seam::entry(count)]] void tet_deformation_gradient(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(4)]] const unsigned *tet,
    [[seam::bound]] unsigned vertex_count,
    
    const Mat3x3f *inverse_rest,
    Mat3x3f *deformation,
    unsigned count);
