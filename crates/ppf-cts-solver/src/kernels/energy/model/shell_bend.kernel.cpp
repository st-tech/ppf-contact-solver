// File: shell_bend.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py
// renders it into the three forms the three compilers read, and the build hands
// each compiler its own form. The two facts a backend cannot infer are written
// as C++ attributes: `[[seam::device_fn]]` is the execution space, and
// `[[seam::thread]]` is the address space of a reference parameter. MSL
// requires the second on every reference and pointer type; CUDA and the host
// have one address space and are handed the same declarations with it removed.
//
// EVERY DIVISION AND EVERY ROOT HERE GOES THROUGH `fmath::div` AND
// `fmath::sqrt`, and the dihedral angle through `fmath::atan2`, which the
// backend prologue defines. That is a correctness spelling rather than a style:
// MSL's plain `/` and `sqrt` are not correctly rounded even at mathMode Safe,
// while `precise::divide` and `precise::sqrt` are in every math mode and CUDA's
// are already.

#include "../../contact/distance.hpp"
#include "../../linalg/eigsolve.hpp"
// THE THREE BODIES `shell_bend_embed` COMPOSES. Each is a thread-space
// function of its own file: the lagged damping add, the four-vertex force
// scatter and the CSR block push. They are included here rather than the
// composition living in one of theirs because the composition is a bending
// kernel that embeds, not a scatter that bends.
#include "../../csrmat/fixed_csr.kernel.cpp"
#include "../../utility/hinge_damping.kernel.cpp"
#include "../../utility/hinge_scatter.kernel.cpp"
#include "shell_bend_stiffness.kernel.cpp"

[[seam::device_fn]] inline float
shell_bend_angle(const Vec3f &v0,
                     const Vec3f &v1,
                     const Vec3f &v2,
                     const Vec3f &v3) {
    const Vec3f edge10 = proximity::difference<float, float>(v1, v0);
    const Vec3f edge20 = proximity::difference<float, float>(v2, v0);
    const Vec3f edge23 = proximity::difference<float, float>(v2, v3);
    const Vec3f edge13 = proximity::difference<float, float>(v1, v3);
    const Vec3f normal1 = edge10.cross(edge20);
    const Vec3f normal2 = edge23.cross(edge13);
    const Vec3f shared_edge =
        proximity::difference<float, float>(v1, v2);
    const float edge_norm = shared_edge.norm();
    if (edge_norm <= 0.0f) {
        return 0.0f;
    }
    return fmath::atan2(
        fmath::div(normal2.cross(normal1).dot(shared_edge), edge_norm),
        normal1.dot(normal2));
}

[[seam::device_fn]] inline bool shell_bend_angle_gradient(
    const Vec3f &v2, const Vec3f &v0,
    const Vec3f &v1, const Vec3f &v3,
    Mat3x4f &gradient,
    float &normal1_squared,
    float &normal2_squared,
    float &edge_norm) {
    const Vec3f edge0 = proximity::difference<float, float>(v1, v0);
    const Vec3f edge1 = proximity::difference<float, float>(v2, v0);
    const Vec3f edge2 = proximity::difference<float, float>(v3, v0);
    const Vec3f edge3 = proximity::difference<float, float>(v2, v1);
    const Vec3f edge4 = proximity::difference<float, float>(v3, v1);
    const Vec3f normal1 = edge0.cross(edge1);
    const Vec3f normal2 = edge2.cross(edge0);
    normal1_squared = normal1.squaredNorm();
    normal2_squared = normal2.squaredNorm();
    edge_norm = edge0.norm();
    if (normal1_squared <= 0.0f || normal2_squared <= 0.0f ||
        edge_norm <= 0.0f) {
        gradient = Mat3x4f::Zero();
        return false;
    }
    gradient.col(0) =
        -fmath::div(edge_norm, normal1_squared) * normal1;
    gradient.col(1) =
        -fmath::div(edge0.dot(edge3), edge_norm * normal1_squared) * normal1 -
        fmath::div(edge0.dot(edge4), edge_norm * normal2_squared) * normal2;
    gradient.col(2) =
        fmath::div(edge0.dot(edge1), edge_norm * normal1_squared) * normal1 +
        fmath::div(edge0.dot(edge2), edge_norm * normal2_squared) * normal2;
    gradient.col(3) =
        -fmath::div(edge_norm, normal2_squared) * normal2;
    return true;
}

[[seam::device_fn]] inline bool shell_bend_force(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, const Vec3f &x3,
    float rest_angle, Mat3x4f &force,
    float &normal1_squared,
    float &normal2_squared,
    float &edge_norm) {
    Mat3x4f angle_gradient;
    const bool valid = shell_bend_angle_gradient(
        x0, x1, x2, x3, angle_gradient, normal1_squared, normal2_squared,
        edge_norm);
    force =
        (shell_bend_angle(x0, x1, x2, x3) - rest_angle) * angle_gradient;
    return valid;
}

[[seam::device_fn]] inline void shell_bend_accumulate_f_mode(
    float eigenvalue, const Vec3f &qa,
    const Vec3f &qb, const Mat3x3f &m0,
    const Mat3x3f &m1,
    const Mat3x3f &eta2_transpose,
    const Mat3x3f &eta3_transpose,
    const Mat3x3f &projection,
    Mat12x12f &hessian) {
    if (eigenvalue <= 0.0f) {
        return;
    }
    const float norm_squared = qa.dot(qa) + qb.dot(qb);
    if (norm_squared <= 1.0e-20f) {
        return;
    }
    const Vec3f block0 = m0 * qa + m1 * qb;
    const Vec3f block1 = eta2_transpose * qa + eta3_transpose * qb;
    const Vec3f block2 = projection * qa;
    const Vec3f block3 = projection * qb;
    float vector[12];
    for (int component = 0; component < 3; ++component) {
        vector[component] = block0[component];
        vector[3 + component] = block1[component];
        vector[6 + component] = block2[component];
        vector[9 + component] = block3[component];
    }
    const float scale = fmath::div(eigenvalue, norm_squared);
    for (int row = 0; row < 12; ++row) {
        for (int column = 0; column < 12; ++column) {
            hessian(row, column) +=
                scale * vector[row] * vector[column];
        }
    }
}

[[seam::device_fn]] inline Vec3f shell_bend_divide_vector(
    const Vec3f &value, float divisor) {
    return Vec3f(fmath::div(value[0], divisor), fmath::div(value[1], divisor),
                 fmath::div(value[2], divisor));
}

// Exact force and unconditionally PSD-projected Hessian for
// E = 1/2 (theta - theta0)^2. The exact Hessian is generally indefinite, so
// the projection uses the closed-form shell hinge eigensystem of Wu and Kim
// 2023. It projects the six-mode F term and rank-four W term separately, then
// adds them. The force remains exact.
//
// The implementation includes four corrections verified against finite
// differences: the rod u-vector b sign, the shell section-three sign
// S3 = -sign(tb.t1), the p3 x0-block c signs in equation 57, and the missing
// kappa1 factor in equation 49's d coefficients.
[[seam::device_fn]] inline bool shell_bend_force_hessian(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, const Vec3f &x3,
    float rest_angle, Mat3x4f &force,
    Mat12x12f &hessian,
    float &normal1_squared,
    float &normal2_squared,
    float &shared_edge_norm) {
    const float angle = shell_bend_angle(x0, x1, x2, x3);
    const bool geometry_valid = shell_bend_force(
        x0, x1, x2, x3, rest_angle, force, normal1_squared,
        normal2_squared, shared_edge_norm);
    hessian = Mat12x12f::Zero();
    if (!geometry_valid) {
        return false;
    }

    // Wu and Kim's stencil is (P0,P1=edge,P2,P3=flaps). The solver's hinge
    // order maps it to (P0,P1,P2,P3) = (x2,x1,x0,x3). Geometry is kept in
    // paper order until the final block permutation.
    const Vec3f y1 = proximity::difference<float, float>(x1, x2);
    const Vec3f y2 = proximity::difference<float, float>(x0, x2);
    const Vec3f y3 = proximity::difference<float, float>(x3, x2);
    const float y1_norm = y1.norm();
    if (y1_norm <= 1.0e-9f) {
        return true;
    }
    const Vec3f tangent =
        shell_bend_divide_vector(y1, y1_norm);
    const Mat3x3f projection =
        Mat3x3f::Identity() - tangent * tangent.transpose();
    const Vec3f z0 = y2 - tangent * tangent.dot(y2);
    const Vec3f z1 = y3 - tangent * tangent.dot(y3);
    const float z0_norm = z0.norm();
    const float z1_norm = z1.norm();
    const Vec3f binormal_cross = z0.cross(z1);
    const float binormal_cross_norm = binormal_cross.norm();
    if (z0_norm <= 1.0e-9f || z1_norm <= 1.0e-9f ||
        binormal_cross_norm <= 1.0e-9f * z0_norm * z1_norm) {
        return true;
    }
    const Vec3f binormal =
        shell_bend_divide_vector(binormal_cross,
                                     binormal_cross_norm);
    const float geometry_sign =
        binormal.dot(tangent) >= 0.0f ? 1.0f : -1.0f;
    const float angle_difference = angle - rest_angle;

    Mat12x12f paper_hessian = Mat12x12f::Zero();

    // F term: the six angle-energy eigenpairs mapped through the projected
    // edge Jacobian.
    const Vec3f edge0_perpendicular = z0.cross(binormal);
    const Vec3f edge1_perpendicular = z1.cross(binormal);
    const float section_sign = -geometry_sign;
    const float signed_angle_difference =
        section_sign * angle_difference;
    const float gamma =
        fmath::div(z1_norm * z1_norm, z0_norm * z0_norm);
    const float gamma_minus_one = gamma - 1.0f;
    const float gamma_plus_one = gamma + 1.0f;
    const float radical = fmath::sqrt(
        4.0f * signed_angle_difference * signed_angle_difference *
            fmath::div(gamma_minus_one, gamma_plus_one) *
            fmath::div(gamma_minus_one, gamma_plus_one) +
        1.0f);
    const float root_minus = fmath::sqrt(fmath::max(
        0.0f, 2.0f *
                  (2.0f * signed_angle_difference *
                       signed_angle_difference +
                   1.0f - radical)));
    const float root_plus = fmath::sqrt(fmath::max(
        0.0f, 2.0f *
                  (2.0f * signed_angle_difference *
                       signed_angle_difference +
                   1.0f + radical)));
    const float length_hessian =
        fmath::div(1.0f, z0_norm * z0_norm) +
        fmath::div(1.0f, z1_norm * z1_norm);
    const float coupling =
        fmath::div(4.0f * signed_angle_difference, gamma_plus_one);
    const Mat3x3f eta2 =
        (-fmath::div(1.0f, y1_norm)) *
        (tangent.dot(y2) * projection + tangent * z0.transpose());
    const Mat3x3f eta3 =
        (-fmath::div(1.0f, y1_norm)) *
        (tangent.dot(y3) * projection + tangent * z1.transpose());
    const Mat3x3f eta2_transpose = eta2.transpose();
    const Mat3x3f eta3_transpose = eta3.transpose();
    const Mat3x3f map0 = -eta2_transpose - projection;
    const Mat3x3f map1 = -eta3_transpose - projection;
    const int radical_sign[4] = {-1, -1, 1, 1};
    const float root[4] = {root_minus, root_minus, root_plus, root_plus};
    const int root_sign[4] = {-1, 1, -1, 1};
    for (int mode = 0; mode < 4; ++mode) {
        const float root_value = root[mode];
        const float a =
            signed_angle_difference *
            (fmath::div(gamma_minus_one * gamma_minus_one, gamma_plus_one) +
             radical_sign[mode] * gamma_plus_one * radical +
             root_sign[mode] * gamma_minus_one * root_value);
        const float b =
            -0.5f * gamma * root_value * root_value +
            2.0f * signed_angle_difference * signed_angle_difference -
            root_sign[mode] * 0.5f *
                (gamma_minus_one +
                 radical_sign[mode] * gamma_plus_one * radical) *
                root_value;
        const float d =
            1.0f + radical_sign[mode] * radical +
            root_sign[mode] * root_value;
        shell_bend_accumulate_f_mode(
            0.25f * length_hessian * d,
            a * z0 + b * edge0_perpendicular,
            coupling * z1 + d * edge1_perpendicular, map0, map1,
            eta2_transpose, eta3_transpose, projection, paper_hessian);
    }
    const float cosine =
        fmath::div(z0.dot(z1), z0_norm * z1_norm);
    const float sine =
        fmath::div(binormal_cross_norm, z0_norm * z1_norm);
    const float beta =
        (fmath::div(z1_norm, z0_norm) - fmath::div(z0_norm, z1_norm)) * cosine;
    const float alpha =
        0.5f * (-beta + fmath::sqrt(beta * beta + 4.0f));
    const float inverse_sine_angle =
        fmath::div(signed_angle_difference, sine);
    shell_bend_accumulate_f_mode(
        inverse_sine_angle *
            (fmath::div(cosine, z1_norm * z1_norm) -
             fmath::div(alpha, z0_norm * z1_norm)),
        alpha * binormal, binormal, map0, map1, eta2_transpose,
        eta3_transpose, projection, paper_hessian);
    shell_bend_accumulate_f_mode(
        inverse_sine_angle *
            (fmath::div(cosine, z1_norm * z1_norm) +
             fmath::div(1.0f, alpha * z0_norm * z1_norm)),
        binormal, (-alpha) * binormal, map0, map1, eta2_transpose,
        eta3_transpose, projection, paper_hessian);

    // W term: rank-four generalized eigensystem. The Gram matrix is
    // block-diagonal because p0 is orthogonal to p1 and both are orthogonal
    // to p2 and p3.
    const Vec3f tau0 =
        shell_bend_divide_vector(z0, z0_norm);
    const Vec3f tau1 =
        shell_bend_divide_vector(z1, z1_norm);
    const Vec3f tau1_perpendicular = tau1.cross(binormal);
    const float a0 =
        fmath::div(tangent.dot(y2), y1_norm * z0_norm);
    const float a1 =
        fmath::div(tangent.dot(y3), y1_norm * z1_norm);
    const float c0 = fmath::div(1.0f, z0_norm);
    const float c1 = fmath::div(1.0f, z1_norm);
    const float kappa0 =
        -fmath::div(4.0f * tau0.dot(tau1_perpendicular),
                  y1_norm * y1_norm);
    const float kappa1 =
        fmath::div(2.0f * tau0.dot(tau1_perpendicular), y1_norm);
    const Vec3f tau_sum = tau0 + tau1;
    const Vec3f tau_difference = tau0 - tau1;
    float basis[4][12];
    for (int component = 0; component < 3; ++component) {
        basis[0][component] = tau_sum[component];
        basis[0][3 + component] = -tau_sum[component];
        basis[0][6 + component] = 0.0f;
        basis[0][9 + component] = 0.0f;
        basis[1][component] = tau_difference[component];
        basis[1][3 + component] = -tau_difference[component];
        basis[1][6 + component] = 0.0f;
        basis[1][9 + component] = 0.0f;
        basis[2][component] =
            (-a1 - a0 + c1 + c0) * tangent[component];
        basis[2][3 + component] =
            (a1 + a0) * tangent[component];
        basis[2][6 + component] = -c0 * tangent[component];
        basis[2][9 + component] = -c1 * tangent[component];
        basis[3][component] =
            (-a1 + a0 + c1 - c0) * tangent[component];
        basis[3][3 + component] =
            (a1 - a0) * tangent[component];
        basis[3][6 + component] = c0 * tangent[component];
        basis[3][9 + component] = -c1 * tangent[component];
    }
    float gram_diagonal[4];
    float gram23 = 0.0f;
    for (int row = 0; row < 4; ++row) {
        float sum = 0.0f;
        for (int element = 0; element < 12; ++element) {
            sum += basis[row][element] * basis[row][element];
        }
        gram_diagonal[row] = sum;
    }
    for (int element = 0; element < 12; ++element) {
        gram23 += basis[2][element] * basis[3][element];
    }

    // Near-flat and near-fold hinges can round one basis norm to zero. The F
    // term is already PSD, and the omitted W term tends to zero there.
    const float determinant23 =
        gram_diagonal[2] * gram_diagonal[3] - gram23 * gram23;
    if (gram_diagonal[0] > 1.0e-5f &&
        gram_diagonal[1] > 1.0e-5f &&
        determinant23 >
            1.0e-6f * gram_diagonal[2] * gram_diagonal[3]) {
        const float weight = -angle_difference * geometry_sign;
        float energy[4][4];
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                energy[row][column] = 0.0f;
            }
        }
        energy[0][0] = weight * (-kappa0 * gram_diagonal[0]);
        energy[1][1] = weight * (kappa0 * gram_diagonal[1]);
        energy[0][2] = weight * (kappa1 * gram_diagonal[2]);
        energy[2][0] = energy[0][2];
        energy[0][3] = weight * (kappa1 * gram23);
        energy[3][0] = energy[0][3];
        energy[1][2] = weight * (kappa1 * gram23);
        energy[2][1] = energy[1][2];
        energy[1][3] = weight * (kappa1 * gram_diagonal[3]);
        energy[3][1] = energy[1][3];

        float lower_inverse[4][4];
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                lower_inverse[row][column] = 0.0f;
            }
        }
        lower_inverse[0][0] =
            fmath::div(1.0f, fmath::sqrt(gram_diagonal[0]));
        lower_inverse[1][1] =
            fmath::div(1.0f, fmath::sqrt(gram_diagonal[1]));
        const float lower22 = fmath::sqrt(gram_diagonal[2]);
        const float lower32 = fmath::div(gram23, lower22);
        const float lower33 = fmath::sqrt(
            fmath::max(1.0e-30f,
                     gram_diagonal[3] - lower32 * lower32));
        lower_inverse[2][2] = fmath::div(1.0f, lower22);
        lower_inverse[3][3] = fmath::div(1.0f, lower33);
        lower_inverse[3][2] =
            -fmath::div(lower32, lower22 * lower33);

        float lower_inverse_energy[4][4];
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                float sum = 0.0f;
                for (int inner = 0; inner < 4; ++inner) {
                    sum += lower_inverse[row][inner] *
                           energy[inner][column];
                }
                lower_inverse_energy[row][column] = sum;
            }
        }
        SMat<float, 4, 4> symmetric;
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                float sum = 0.0f;
                for (int inner = 0; inner < 4; ++inner) {
                    sum += lower_inverse_energy[row][inner] *
                           lower_inverse[column][inner];
                }
                symmetric(row, column) = sum;
            }
        }
        linalg::psd_project_symmetric<4>(symmetric, 0.0f);

        float lower_transpose_symmetric[4][4];
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                float sum = 0.0f;
                for (int inner = 0; inner < 4; ++inner) {
                    sum += lower_inverse[inner][row] *
                           symmetric(inner, column);
                }
                lower_transpose_symmetric[row][column] = sum;
            }
        }
        float core[4][4];
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                float sum = 0.0f;
                for (int inner = 0; inner < 4; ++inner) {
                    sum += lower_transpose_symmetric[row][inner] *
                           lower_inverse[inner][column];
                }
                core[row][column] = sum;
            }
        }
        float basis_core[12][4];
        for (int element = 0; element < 12; ++element) {
            for (int column = 0; column < 4; ++column) {
                float sum = 0.0f;
                for (int row = 0; row < 4; ++row) {
                    sum += basis[row][element] * core[row][column];
                }
                basis_core[element][column] = sum;
            }
        }
        for (int row = 0; row < 12; ++row) {
            for (int column = 0; column < 12; ++column) {
                float sum = 0.0f;
                for (int inner = 0; inner < 4; ++inner) {
                    sum += basis_core[row][inner] *
                           basis[inner][column];
                }
                paper_hessian(row, column) += sum;
            }
        }
    }

    const int permutation[4] = {2, 1, 0, 3};
    for (int block_row = 0; block_row < 4; ++block_row) {
        for (int block_column = 0; block_column < 4; ++block_column) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    hessian(3 * permutation[block_row] + row,
                            3 * permutation[block_column] + column) =
                        paper_hessian(3 * block_row + row,
                                      3 * block_column + column);
                }
            }
        }
    }
    return true;
}


// The hinge's node permutation, (2, 1, 0, 3).
//
// The solver stores a hinge as (flap, edge, edge, flap) in mesh order; the
// dihedral math above wants (P0, P1 = edge, P2, P3 = flap) and reaches it by
// this exchange of the first and third nodes. It is named rather than written
// out at each site because THREE things depend on the same permutation and two
// of them are silent if it drifts: the force and Hessian above are evaluated in
// this order, `builder.rs` emits the hinge Hessian slot table in this order, so
// slot[ii * 4 + jj] targets the pair a push would, and the plasticity pass
// reads the rest angle through it.
[[seam::device_fn]] inline Vec4u
shell_bend_remap(const Vec4u &hinge) {
    return Vec4u(hinge[2], hinge[1], hinge[0], hinge[3]);
}

// The same evaluation with the verdict RAISED ON THE DEVICE rather than
// mirrored to the host.
//
// `energy.cu` answers all three degeneracies with a live release assert, so a
// hinge whose dihedral angle is undefined stops the run where it is found. The
// verdict beside this one is a word per hinge that the driver downloaded and
// walked every pass to reach the same conclusion, which is a whole-hinge-set
// copy across the seam to carry one bit that is almost always clear.
//
// THE THREE SCALARS THE DIAGNOSTIC CARRIES ARE THE THREE DEGENERACIES, which is
// what makes this the better channel rather than merely the cheaper one: the
// verdict alone said a hinge was degenerate, and these say WHICH of the two
// squared normals or the shared edge length went to zero, without a record
// field for any of them.
//
// THE REST ANGLE COMES OFF THE HINGE'S OWN RECORD, which is where plasticity
// creeps it: a staged array beside the record was a copy of one field, refilled
// and re-uploaded every pass out of the same `HingeProp` the device already
// holds, and a copy of a crept field is a second place for it to go stale.
//
// `live` IS WHAT THE CALLER PRICES THIS HINGE BY, and it is a parameter because
// the two poses price it differently: the iterate's evaluation is consumed
// wherever the bending STIFFNESS is positive, and the start-of-step evaluation
// only where the bending DAMPING is. A degenerate hinge whose contribution is
// multiplied by zero is not a scene defect at either pose, so asserting on it
// would stop a run over geometry that reaches no result. Both callers already
// hold the array they gate on, so this costs no buffer of its own.
[[seam::device_fn]] inline void shell_bend_force_hessian_checked(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, const Vec3f &x3,
    const HingeProp &prop, float live, float scale,
    Mat3x4f &force, Mat12x12f &hessian,
    DiagHandle diag, unsigned element) {
    const float rest_angle = prop.rest_angle;
    float normal1_squared = 0.0f;
    float normal2_squared = 0.0f;
    float shared_edge_norm = 0.0f;
    const bool ok = shell_bend_force_hessian(x0, x1, x2, x3, rest_angle, force,
                                             hessian, normal1_squared,
                                             normal2_squared, shared_edge_norm);
    DIAG_ASSERT4(diag, ok || live <= 0.0f, static_cast<float>(element),
                 normal1_squared, normal2_squared, shared_edge_norm);
    // THE STIFFNESS IS APPLIED WHERE THE VALUE IS FORMED, in the same pass that
    // forms it. Writing the force and the Hessian out raw and accumulating
    // `scale * raw` into a seeded-zero destination afterwards would take two
    // further passes over hinge-sized arrays for identical arithmetic:
    // `fma(scale, raw, 0)` into an array the seed has just zeroed is
    // `scale * raw` exactly, so applying it here removes the passes and moves
    // no bit.
    //
    // `scale` IS NOT `live`. The two callers gate on different arrays, the
    // iterate's on the bending stiffness and the start-of-step's on the
    // damping, and both scale by the STIFFNESS. Folding them into one
    // parameter would silently scale the damping pose by the wrong array.
    force *= scale;
    hessian *= scale;
}

// FOUR POSITIONS READ THROUGH THE HINGE'S OWN INDEX LIST, which is what
// `[[seam::indices(4)]]` over a `[[seam::through]]` buffer expresses, and the
// `[[seam::bound]]` beside it is not bookkeeping: a slot is DATA rather than
// the thread index, so the thread-count guard says nothing about it, and Metal
// answers an out-of-bounds read with 0.0 rather than a fault, which would put a
// plausible bending force on a corrupt hinge table.
//
// THE FORCE AND THE HESSIAN ARE NON-CONST GATHERS AND NOT SCATTERS. A gather
// hands the body `buffer[index]`, which is an lvalue, so a non-const one IS the
// write. Both are fully assigned by the body before it reads anything of its
// own, so neither carries what the previous element left.
//
// THERE IS NO SCATTER SINK. The verdict is raised through the diagnostic lane
// rather than deposited as a word per hinge, so nothing here is written for the
// host to read back.
[[seam::entry(count, element)]] void shell_bend_force_hessian_checked(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(4)]] const unsigned *hinge,
    [[seam::bound]] unsigned vertex_count,
    const HingeProp *prop,
    const float *live,
    const float *scale,
    Mat3x4f *force,
    Mat12x12f *hessian,
    DiagHandle diag, unsigned element,
    unsigned count);

// THE HINGE IS COMPUTED AND EMBEDDED IN ONE KERNEL: form the force and the
// Hessian at the iterate, scale by the stiffness, damp against the lagged pose
// where the material asks for it, then scatter the force and push the blocks,
// all from registers. The 12x12 never reaches memory.
//
// WHAT A SPLIT PIPELINE WOULD COST, measured on `bench_drape`: a checked pass
// writing 144 floats per hinge to `hessian_x`, a damping pass reading and
// rewriting them, a force scatter reading 12 per hinge back, and a push reading
// all 144 back again, about 69 MB each way per assembly per Newton step for the
// Hessian alone.
//
// THE GATE IS THE STIFFNESS, as `push_hinge_hessians` has it and for the reason
// it gives: a hinge outside the run has no registered stencil, so pushing its
// zero blocks would count refusals over blocks nobody wanted.
//
// THE QUAD IS READ FROM THE DEVICE LIST BY THIS BODY, beside the four positions
// the entry gathered through that same list. A `[[seam::indices(4)]]` gather
// brings positions and not the indices they were reached by, and the scatter
// and the push both need the indices, so the record names the list twice: once
// for the gather with its bound, once as a plain pointer for this read.
//
// TWO POSES, ONE INDEX ORDER. `hinge` is the REMAPPED (2,1,0,3) list the
// dihedral math wants, and both the iterate and the start-of-step read through
// it, so the lagged Hessian is in the same vertex order as the one it damps.
[[seam::device_fn]] inline void shell_bend_embed(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, const Vec3f &x3,
    const Vec3f &current0,
    const Vec3f &current1,
    const Vec3f &current2,
    const Vec3f &current3,
    const unsigned *quad,
    const HingeProp &prop,
    const HingeParam *hinge_param,
    const VertexProp *vertex_prop, unsigned kind,
    float dt, compute::atomic_float_t *force,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value, unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness,
    const unsigned *hess_slots, unsigned has_hess_slots,
    DiagHandle diag, unsigned element) {
    // THE MESH ORDER, RECOVERED IN THE THREAD. `quad` carries the remapped
    // `(2,1,0,3)` order every later stage uses, and the areal density below is
    // an fp32 running sum whose VALUE depends on the order its four terms are
    // added in, so it must be summed in MESH order or it is a different float.
    // The permutation is an INVOLUTION, so applying it to the remapped list
    // recovers the mesh list exactly and no second index array is needed,
    // which is also what keeps the entry inside the generator's one-index-list
    // rule.
    unsigned mesh_order[4];
    mesh_order[0] = quad[4u * element + 2u];
    mesh_order[1] = quad[4u * element + 1u];
    mesh_order[2] = quad[4u * element + 0u];
    mesh_order[3] = quad[4u * element + 3u];
    float vertex_mass[4];
    float vertex_area[4];
    for (unsigned k = 0; k < 4u; ++k) {
        const VertexProp vert = vertex_prop[mesh_order[k]];
        vertex_mass[k] = vert.mass;
        vertex_area[k] = vert.area;
    }
    const float areal_density =
        shell_bend_areal_density(vertex_mass, vertex_area);
    const float stiffness =
        shell_bend_stiffness_from_records(prop, hinge_param, kind, areal_density);
    if (stiffness <= 0.0f) {
        return;
    }
    // THE DAMPING IS GATED BY THE STIFFNESS, which is why the two are formed
    // together: the term it scales is built from the stiffness-scaled
    // start-of-step Hessian, so a hinge carrying no bending stiffness has
    // nothing to damp.
    const float damping = hinge_param[prop.param_index].bend_damping;
    Mat3x4f gradient;
    Mat12x12f hessian;
    // THE ITERATE, checked and scaled by the shared body, so the degeneracy
    // verdict and the scaling have one statement.
    shell_bend_force_hessian_checked(x0, x1, x2, x3, prop, stiffness,
                                     stiffness, gradient, hessian, diag,
                                     element);
    if (damping > 0.0f) {
        // THE LAGGED POSE, scaled by the same stiffness, exactly as
        // `energy.cu` forms `K_lag`. Its own degeneracy is gated on the
        // damping here as the separate pass gated it, because a start-of-step
        // hinge that damps by zero reaches no result.
        Mat3x4f lagged_gradient;
        Mat12x12f lagged;
        shell_bend_force_hessian_checked(current0, current1, current2,
                                         current3, prop, damping, stiffness,
                                         lagged_gradient, lagged, diag,
                                         element);
        const Vec3f x[4] = {x0, x1, x2, x3};
        const Vec3f current[4] = {current0, current1, current2, current3};
        hinge_add_stiffness_damping_lagged(x, current, damping, dt, gradient,
                                           hessian, lagged);
    }
    Vec4u indices;
    unsigned slots[4];
    for (unsigned k = 0; k < 4u; ++k) {
        slots[k] = quad[4u * element + k];
        indices[k] = slots[k];
    }
    hinge_atomic_embed_force(indices, gradient, force);
    // THE REFERENCE'S OWN BRANCH, `energy.cu:548`. `hinge_hess_slots` was built
    // in the SAME remapped (2,1,0,3) order `quad` carries, so `slot[ii*4+jj]`
    // targets the block the row search would have found for
    // `(slots[ii], slots[jj])`; the two orders agreeing is what makes the
    // table usable here without a permutation of its own.
    if (has_hess_slots != 0u) {
        fixed_push_blocks_thread_at(hess_slots, hessian.m, 4u, fixed_value,
                                    element);
    } else {
        fixed_push_blocks_thread(slots, hessian.m, 4u, fixed_index,
                                 fixed_offset, fixed_value, row_count, refused,
                                 witness);
    }
}

[[seam::entry(count, element)]] void shell_bend_embed(
    [[seam::through]] const Vec3f *x,
    [[seam::through]] const Vec3f *current,
    [[seam::indices(4)]] const unsigned *hinge,
    [[seam::bound]] unsigned vertex_count,
    const unsigned *quad,
    const HingeProp *prop,
    const HingeParam *hinge_param,
    const VertexProp *vertex_prop,
    const unsigned *kind,
    float dt,
    compute::atomic_float_t *force,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value,
    unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness,
    const unsigned *hess_slots,
    unsigned has_hess_slots,
    DiagHandle diag, unsigned element,
    unsigned count);

// THE DIHEDRAL ANGLE, reading the hinge's four vertices THROUGH its own index
// list, with the bound each slot is checked against carried in the record. The
// angle is the body's RETURN value, so it reaches memory as the one shape a
// return value has: written at the thread index to the scatter buffer.
[[seam::entry(count)]] void shell_bend_angle(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(4)]] const unsigned *hinge,
    [[seam::bound]] unsigned vertex_count,
    float *angle,
    unsigned count);

// THE HINGE REORDERING, which reads and writes index quadruples and touches no
// position at all. Both are `Vec4u`, so this is an ordinary element gather into
// an element scatter and needs neither an index list nor a bound: the quadruple
// IS the element here rather than a list of slots into something else.
[[seam::entry(count)]] void shell_bend_remap(
    const Vec4u *hinge,
    Vec4u *remapped,
    unsigned count);
