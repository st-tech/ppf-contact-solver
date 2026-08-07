// License: Apache v2.0

#ifndef TRANSLATION_LOCK_HPP
#define TRANSLATION_LOCK_HPP

#include "../buffer/buffer.hpp"
#include "../data.hpp"
#include "../main/cuda_utils.hpp"
#include "../utility/dispatcher.hpp"
#include "translation_lock_math.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

// Exact aggregate translation and rotation constraints.
//
// A deformable lock has up to four rows C: two mass-weighted center-of-mass
// translation rows and one or two best-fit angular rows. They share one
// Euclidean projector,
//
//   Q = I - C^T (C C^T)^+ C,
//
// rather than composing translation and rotation projectors. The pseudoinverse
// handles a compatible partially prescribed group whose free rows are
// rank-deficient, while an incompatible affine right-hand side fails loudly.
//
// The rotation rows use the current geometry. Allow-only uses a tangent basis
// b perpendicular to the requested axis a. Prohibit-axis uses b = a.
//
//   b^T I^-1 sum_i m_i (r_i x dx_i)
//       = sum_i [m_i ((I^-1 b) x r_i)]^T dx_i.
//
// All r_i are formed by subtracting a shared anchor before accumulation.
namespace translation_lock {

constexpr unsigned UNSET = 0xffffffffu;
constexpr unsigned TRANSLATION_ROW0 = 1u << 0;
constexpr unsigned TRANSLATION_ROW1 = 1u << 1;
constexpr unsigned ROTATION_ROW0 = 1u << 2;
constexpr unsigned ROTATION_ROW1 = 1u << 3;

struct LockFrame {
    Vec3f translation_basis0;
    Vec3f translation_basis1;
    Vec3f rotation_basis0;
    Vec3f rotation_basis1;
    Vec3f com_relative;
    Mat3x3f inv_inertia;
    Mat4x4f gram_pinv;
    Vec4f rhs;
    Vec4f fixed;
    unsigned row_mask;
};

__host__ __device__ inline bool axis_enabled(const Vec3f &axis) {
    return axis[0] != 0.0f || axis[1] != 0.0f || axis[2] != 0.0f;
}

__host__ __device__ inline bool rotation_mode_valid(unsigned mode) {
    return mode == ROTATION_LOCK_ALLOW_ONLY ||
           mode == ROTATION_LOCK_PROHIBIT_AXIS;
}

// A deterministic orthonormal tangent basis for a normalized axis.
__host__ __device__ inline void tangent_basis(const Vec3f &axis, Vec3f &b0,
                                              Vec3f &b1) {
    const Vec3f reference =
        fabsf(axis[2]) < 0.9f ? Vec3f(0.0f, 0.0f, 1.0f)
                              : Vec3f(1.0f, 0.0f, 0.0f);
    b0 = axis.cross(reference);
    b0.normalize();
    b1 = axis.cross(b0);
    b1.normalize();
}

__device__ inline void atomic_add_vec3(Vec3f *dst, unsigned index,
                                       const Vec3f &v) {
    atomicAdd(&dst[index][0], v[0]);
    atomicAdd(&dst[index][1], v[1]);
    atomicAdd(&dst[index][2], v[2]);
}

__device__ inline void atomic_add_vec4(Vec4f *dst, unsigned index,
                                       const Vec4f &v) {
    atomicAdd(&dst[index][0], v[0]);
    atomicAdd(&dst[index][1], v[1]);
    atomicAdd(&dst[index][2], v[2]);
    atomicAdd(&dst[index][3], v[3]);
}

__device__ inline Vec3f matvec3(const Mat3x3f &m, const Vec3f &v) {
    return Vec3f(m(0, 0) * v[0] + m(0, 1) * v[1] + m(0, 2) * v[2],
                 m(1, 0) * v[0] + m(1, 1) * v[1] + m(1, 2) * v[2],
                 m(2, 0) * v[0] + m(2, 1) * v[1] + m(2, 2) * v[2]);
}

__host__ inline Vec3f host_matvec3(const Mat3x3f &m, const Vec3f &v) {
    return Vec3f(m(0, 0) * v[0] + m(0, 1) * v[1] + m(0, 2) * v[2],
                 m(1, 0) * v[0] + m(1, 1) * v[1] + m(1, 2) * v[2],
                 m(2, 0) * v[0] + m(2, 1) * v[1] + m(2, 2) * v[2]);
}

__device__ inline Vec4f matvec4(const Mat4x4f &m, const Vec4f &v) {
    Vec4f out = Vec4f::Zero();
#pragma unroll
    for (unsigned row = 0; row < 4; ++row) {
#pragma unroll
        for (unsigned col = 0; col < 4; ++col) {
            out[row] += m(row, col) * v[col];
        }
    }
    return out;
}

__host__ inline Vec4f host_matvec4(const Mat4x4f &m, const Vec4f &v) {
    Vec4f out = Vec4f::Zero();
    for (unsigned row = 0; row < 4; ++row)
        for (unsigned col = 0; col < 4; ++col)
            out[row] += m(row, col) * v[col];
    return out;
}

// Scalar coefficients of the up-to-four aggregate rows for one vertex. A
// separate Vec3f is returned for each row because each row acts on a vertex's
// xyz block.
struct RowCoefficients {
    Vec3f row[4];
};

__device__ inline RowCoefficients
row_coefficients(const TranslationLock &lock, const LockFrame &frame,
                 const Vec3f &position, float mass) {
    RowCoefficients out{};
    out.row[0] = Vec3f::Zero();
    out.row[1] = Vec3f::Zero();
    out.row[2] = Vec3f::Zero();
    out.row[3] = Vec3f::Zero();
    if (frame.row_mask & TRANSLATION_ROW0) {
        out.row[0] = mass * frame.translation_basis0;
    }
    if (frame.row_mask & TRANSLATION_ROW1) {
        out.row[1] = mass * frame.translation_basis1;
    }
    if (frame.row_mask & (ROTATION_ROW0 | ROTATION_ROW1)) {
        const Vec3f r =
            (position - lock.anchor).cast<float>() - frame.com_relative;
        if (frame.row_mask & ROTATION_ROW0) {
            const Vec3f u = matvec3(frame.inv_inertia, frame.rotation_basis0);
            out.row[2] = mass * u.cross(r);
        }
        if (frame.row_mask & ROTATION_ROW1) {
            const Vec3f u = matvec3(frame.inv_inertia, frame.rotation_basis1);
            out.row[3] = mass * u.cross(r);
        }
    }
    return out;
}

__device__ inline Vec3f rows_transpose_times(const RowCoefficients &c,
                                              const Vec4f &lambda) {
    Vec3f out = Vec3f::Zero();
#pragma unroll
    for (unsigned row = 0; row < 4; ++row) {
        out += lambda[row] * c.row[row];
    }
    return out;
}

__device__ inline Vec4f rows_times_vector(const RowCoefficients &c,
                                           const Vec3f &v) {
    Vec4f out = Vec4f::Zero();
#pragma unroll
    for (unsigned row = 0; row < 4; ++row) {
        out[row] = c.row[row].dot(v);
    }
    return out;
}

inline bool has_any(const DataSet &data) {
    return data.translation_lock.size != 0;
}

[[noreturn]] inline void fatal_geometry(unsigned dmap_index,
                                        const char *detail) {
    ppf_fatal(PPF_FATAL_SOLVER_INVARIANT,
              "PPF FATAL: rotation lock displacement group %u has no finite "
              "best-fit angular frame: %s. Lock Rotation requires a "
              "non-collinear positive-mass shape resolved in float32.\n",
              dmap_index, detail);
}

// Host-only SPD inversion for the 3x3 physical inertia. Device code receives
// only its float32 image. The rank floor is a float32 resolution test, not a
// compliance tolerance: below it I^-1 is not a finite quantity the GPU can
// represent reliably.
inline Mat3x3f invert_inertia_host(const Mat3x3f &input,
                                   unsigned dmap_index) {
    double a[3][3];
    double scale = 0.0;
    for (unsigned i = 0; i < 3; ++i) {
        for (unsigned j = 0; j < 3; ++j) {
            const double value =
                0.5 * (static_cast<double>(input(i, j)) +
                       static_cast<double>(input(j, i)));
            if (!std::isfinite(value)) {
                fatal_geometry(dmap_index, "the inertia reduction is non-finite");
            }
            a[i][j] = value;
            scale = std::fmax(scale, std::fabs(value));
        }
    }
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        fatal_geometry(dmap_index, "the inertia is singular");
    }

    constexpr double float_eps = 1.1920928955078125e-7;
    const double rank_floor = 128.0 * float_eps * scale;
    double l[3][3] = {};
    for (unsigned i = 0; i < 3; ++i) {
        for (unsigned j = 0; j <= i; ++j) {
            double sum = a[i][j];
            for (unsigned k = 0; k < j; ++k) {
                sum -= l[i][k] * l[j][k];
            }
            if (i == j) {
                if (!(sum > rank_floor) || !std::isfinite(sum)) {
                    fatal_geometry(
                        dmap_index,
                        "the inertia is singular or below float32 rank resolution");
                }
                l[i][j] = std::sqrt(sum);
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    Mat3x3f inverse = Mat3x3f::Zero();
    for (unsigned col = 0; col < 3; ++col) {
        double y[3] = {};
        for (unsigned i = 0; i < 3; ++i) {
            double sum = i == col ? 1.0 : 0.0;
            for (unsigned k = 0; k < i; ++k) {
                sum -= l[i][k] * y[k];
            }
            y[i] = sum / l[i][i];
        }
        double x[3] = {};
        for (int i = 2; i >= 0; --i) {
            double sum = y[i];
            for (unsigned k = unsigned(i) + 1; k < 3; ++k) {
                sum -= l[k][i] * x[k];
            }
            x[i] = sum / l[i][i];
        }
        for (unsigned row = 0; row < 3; ++row) {
            if (!std::isfinite(x[row]) ||
                std::fabs(x[row]) > std::numeric_limits<float>::max()) {
                fatal_geometry(dmap_index, "the inverse inertia is non-finite");
            }
            inverse(row, col) = static_cast<float>(x[row]);
        }
    }
    return inverse;
}

// Host-only Jacobi pseudoinverse of a 4x4 symmetric positive-semidefinite
// Gram matrix. It returns the exact orthogonal projector for the resolved row
// rank, including the compatible all-pinned case (rank zero).
inline Mat4x4f pseudoinverse_gram_host(const Mat4x4f &input) {
    double raw[4][4];
    double a[4][4];
    double v[4][4] = {};
    double row_scale[4] = {};
    double scale = 0.0;
    for (unsigned i = 0; i < 4; ++i) {
        v[i][i] = 1.0;
        for (unsigned j = 0; j < 4; ++j) {
            const double value =
                0.5 * (static_cast<double>(input(i, j)) +
                       static_cast<double>(input(j, i)));
            if (!std::isfinite(value)) {
                ppf_fatal(PPF_FATAL_SOLVER_INVARIANT,
                          "PPF FATAL: aggregate lock Gram reduction is "
                          "non-finite.\n");
            }
            raw[i][j] = value;
        }
        row_scale[i] =
            raw[i][i] > 0.0 ? 1.0 / std::sqrt(raw[i][i]) : 0.0;
    }
    // Symmetric row equilibration prevents a physically valid rotation row
    // from being classified as rank zero merely because a translation row
    // uses a different dimensional scale. If D = diag(row_scale), solve the
    // eigensystem of D G D, then map its pseudoinverse back as
    // G^+ = D (D G D)^+ D.
    for (unsigned i = 0; i < 4; ++i) {
        for (unsigned j = 0; j < 4; ++j) {
            a[i][j] = row_scale[i] * raw[i][j] * row_scale[j];
            scale = std::fmax(scale, std::fabs(a[i][j]));
        }
    }
    if (scale == 0.0) {
        return Mat4x4f::Zero();
    }

    const double off_floor = 1.0e-14 * scale;
    for (unsigned sweep = 0; sweep < 32; ++sweep) {
        unsigned p = 0, q = 1;
        double largest = 0.0;
        for (unsigned i = 0; i < 4; ++i) {
            for (unsigned j = i + 1; j < 4; ++j) {
                const double value = std::fabs(a[i][j]);
                if (value > largest) {
                    largest = value;
                    p = i;
                    q = j;
                }
            }
        }
        if (largest <= off_floor) {
            break;
        }
        const double app = a[p][p], aqq = a[q][q], apq = a[p][q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0 ? 1.0 : -1.0) /
                         (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;
        for (unsigned k = 0; k < 4; ++k) {
            if (k == p || k == q) {
                continue;
            }
            const double akp = a[k][p], akq = a[k][q];
            a[k][p] = a[p][k] = c * akp - s * akq;
            a[k][q] = a[q][k] = s * akp + c * akq;
        }
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = a[q][p] = 0.0;
        for (unsigned k = 0; k < 4; ++k) {
            const double vkp = v[k][p], vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }

    double largest = 0.0;
    for (unsigned i = 0; i < 4; ++i) {
        if (!std::isfinite(a[i][i])) {
            ppf_fatal(PPF_FATAL_SOLVER_INVARIANT,
                      "PPF FATAL: aggregate lock Gram eigensolve failed.\n");
        }
        largest = std::fmax(largest, std::fabs(a[i][i]));
    }
    const double rank_floor = 256.0 * 1.1920928955078125e-7 * largest;
    Mat4x4f inverse = Mat4x4f::Zero();
    for (unsigned k = 0; k < 4; ++k) {
        if (!(a[k][k] > rank_floor)) {
            continue;
        }
        const double inv = 1.0 / a[k][k];
        for (unsigned i = 0; i < 4; ++i) {
            for (unsigned j = 0; j < 4; ++j) {
                inverse(i, j) +=
                    static_cast<float>(
                        row_scale[i] * v[i][k] * inv * v[j][k] *
                        row_scale[j]);
            }
        }
    }
    return inverse;
}

class FullProjector {
  public:
    FullProjector(const DataSet &data, const Vec<unsigned> &dof_mask,
                  Vec<LockFrame> frames, Vec<Mat4x4f> gram,
                  Vec<Vec4f> sums, Vec<Vec3f> drift, Vec<Vec3f> torque)
        : data_(data), dof_mask_(dof_mask), frames_(frames), gram_(gram),
          sums_(sums), drift_(drift), torque_(torque) {}

    unsigned lock_count() const { return data_.translation_lock.size; }
    Vec<Vec3f> drift() const { return drift_; }

    // Construct the affine feasible Newton correction q. Exact pin increments
    // are retained on removed rows; the free portion is the minimum-norm
    // solution C_free q_free = h - C_fixed p.
    void prepare(const Vec<Vec3f> &positions, const Vec<float> &seed,
                 Vec<float> &q) {
        if (lock_count() == 0) {
            return;
        }
        if (data_.translation_lock_index.size != positions.size ||
            data_.translation_lock_initial.size != positions.size ||
            dof_mask_.size != positions.size ||
            seed.size != 3u * positions.size ||
            q.size != 3u * positions.size) {
            ppf_fatal(PPF_FATAL_SOLVER_INVARIANT,
                      "PPF FATAL: aggregate-lock dataset arrays do not match "
                      "the vertex count.\n");
        }
        positions_ = positions;
        initialize_frames(positions);
        assemble_constraints(positions, seed);

        const DataSet data = data_;
        const Vec<unsigned> mask = dof_mask_;
        const Vec<LockFrame> frames = frames_;
        DISPATCH_START(q.size / 3u)
        [data, mask, positions, frames, seed, q] __device__(unsigned i) mutable {
            if (mask.data[i] != 0u) {
                q.data[3 * i + 0] = seed.data[3 * i + 0];
                q.data[3 * i + 1] = seed.data[3 * i + 1];
                q.data[3 * i + 2] = seed.data[3 * i + 2];
                return;
            }
            const unsigned li = data.translation_lock_index.data[i];
            if (li == UNSET ||
                data.translation_lock.data[li].pdrd_body_index != 0) {
                q.data[3 * i + 0] = 0.0f;
                q.data[3 * i + 1] = 0.0f;
                q.data[3 * i + 2] = 0.0f;
                return;
            }
            const TranslationLock &lock = data.translation_lock.data[li];
            const LockFrame &frame = frames.data[li];
            const RowCoefficients c =
                row_coefficients(lock, frame, positions.data[i],
                                 data.prop.vertex.data[i].mass);
            const Vec4f lambda = matvec4(frame.gram_pinv, frame.rhs);
            const Vec3f value = rows_transpose_times(c, lambda);
            q.data[3 * i + 0] = value[0];
            q.data[3 * i + 1] = value[1];
            q.data[3 * i + 2] = value[2];
        }
        DISPATCH_END;

        // Refine the affine free solution in constraint space. The first
        // float32 pseudoinverse application can leave a visible residual when
        // translation and rotation rows have different scales. Two residual
        // corrections retain exact fixed rows and drive
        // C_free q_free = rhs to the same projected round-off as Q.
        for (unsigned refinement = 0; refinement < 2; ++refinement) {
            sums_.clear(Vec4f::Zero());
            const Vec<Vec4f> sums = sums_;
            DISPATCH_START(q.size / 3u)
            [data, mask, positions, frames, sums,
             q] __device__(unsigned i) mutable {
                if (mask.data[i] != 0u) {
                    return;
                }
                const unsigned li = data.translation_lock_index.data[i];
                if (li == UNSET ||
                    data.translation_lock.data[li].pdrd_body_index != 0) {
                    return;
                }
                const RowCoefficients c = row_coefficients(
                    data.translation_lock.data[li], frames.data[li],
                    positions.data[i], data.prop.vertex.data[i].mass);
                const Vec3f value(q.data[3 * i + 0], q.data[3 * i + 1],
                                  q.data[3 * i + 2]);
                atomic_add_vec4(sums.data, li,
                                rows_times_vector(c, value));
            }
            DISPATCH_END;
            DISPATCH_START(q.size / 3u)
            [data, mask, positions, frames, sums,
             q] __device__(unsigned i) mutable {
                if (mask.data[i] != 0u) {
                    return;
                }
                const unsigned li = data.translation_lock_index.data[i];
                if (li == UNSET ||
                    data.translation_lock.data[li].pdrd_body_index != 0) {
                    return;
                }
                const LockFrame &frame = frames.data[li];
                const RowCoefficients c = row_coefficients(
                    data.translation_lock.data[li], frame,
                    positions.data[i], data.prop.vertex.data[i].mass);
                const Vec4f residual = frame.rhs - sums.data[li];
                const Vec4f lambda =
                    matvec4(frame.gram_pinv, residual);
                const Vec3f correction =
                    rows_transpose_times(c, lambda);
                q.data[3 * i + 0] += correction[0];
                q.data[3 * i + 1] += correction[1];
                q.data[3 * i + 2] += correction[2];
            }
            DISPATCH_END;
        }
    }

    // Q projects a full vector onto the tangent space of every deformable
    // aggregate constraint. PDRD rows live in the reduced body vector and are
    // deliberately left to PDRD::launch_project_bodies.
    void project(Vec<float> &v, unsigned refinements = 1u) const {
        if (lock_count() == 0) {
            return;
        }
        if (positions_.size != data_.translation_lock_index.size ||
            v.size != 3u * positions_.size) {
            ppf_fatal(PPF_FATAL_SOLVER_INVARIANT,
                      "PPF FATAL: aggregate-lock projector was used without "
                      "a matching prepared position frame.\n");
        }
        const DataSet data = data_;
        const Vec<unsigned> mask = dof_mask_;
        const Vec<LockFrame> frames = frames_;
        const Vec<Vec3f> positions = positions_;
        const Vec<Vec4f> sums = sums_;
        // Repeat the constraint-space correction so the float32
        // pseudoinverse acts as an idempotent projector even when combined
        // translation and rotation rows have different scales.
        for (unsigned refinement = 0; refinement < refinements; ++refinement) {
            sums_.clear(Vec4f::Zero());
            DISPATCH_START(v.size / 3u)
            [data, mask, positions, frames, sums,
             v] __device__(unsigned i) mutable {
                if (mask.data[i] != 0u) {
                    return;
                }
                const unsigned li = data.translation_lock_index.data[i];
                if (li == UNSET ||
                    data.translation_lock.data[li].pdrd_body_index != 0) {
                    return;
                }
                const TranslationLock &lock =
                    data.translation_lock.data[li];
                const RowCoefficients c = row_coefficients(
                    lock, frames.data[li], positions.data[i],
                    data.prop.vertex.data[i].mass);
                const Vec3f value(v.data[3 * i + 0], v.data[3 * i + 1],
                                  v.data[3 * i + 2]);
                atomic_add_vec4(sums.data, li,
                                rows_times_vector(c, value));
            }
            DISPATCH_END;

            DISPATCH_START(v.size / 3u)
            [data, mask, positions, frames, sums,
             v] __device__(unsigned i) mutable {
                if (mask.data[i] != 0u) {
                    v.data[3 * i + 0] = 0.0f;
                    v.data[3 * i + 1] = 0.0f;
                    v.data[3 * i + 2] = 0.0f;
                    return;
                }
                const unsigned li = data.translation_lock_index.data[i];
                if (li == UNSET ||
                    data.translation_lock.data[li].pdrd_body_index != 0) {
                    return;
                }
                const TranslationLock &lock =
                    data.translation_lock.data[li];
                const LockFrame &frame = frames.data[li];
                const RowCoefficients c = row_coefficients(
                    lock, frame, positions.data[i],
                    data.prop.vertex.data[i].mass);
                const Vec4f lambda =
                    matvec4(frame.gram_pinv, sums.data[li]);
                const Vec3f correction =
                    rows_transpose_times(c, lambda);
                v.data[3 * i + 0] -= correction[0];
                v.data[3 * i + 1] -= correction[1];
                v.data[3 * i + 2] -= correction[2];
            }
            DISPATCH_END;
        }
    }

    // Verify, without changing it, that a solved deformable/SAND correction
    // carries no forbidden best-fit angular increment. PDRD uses its exact
    // reduced six-DOF projector instead. This is an incremental invariant:
    // rotation locks constrain an incremental best-fit angular component, not
    // an absolute pose, so no post-step snapping is meaningful or performed.
    void check_tangent(const Vec<Vec3f> &positions, const Vec<float> &dx,
                       const char *where) const {
        const unsigned nl = lock_count();
        if (nl == 0) {
            return;
        }
        if (positions.size != data_.translation_lock_index.size ||
            dx.size != 3u * positions.size) {
            ppf_fatal(PPF_FATAL_SOLVER_INVARIANT,
                      "PPF FATAL: aggregate-lock tangent check received "
                      "mismatched vector sizes.\n");
        }
        torque_.clear(Vec3f::Zero());
        const DataSet data = data_;
        const Vec<LockFrame> frames = frames_;
        const Vec<Vec3f> torque = torque_;
        DISPATCH_START(positions.size)
        [data, positions, frames, torque, dx] __device__(unsigned i) mutable {
            const unsigned li = data.translation_lock_index.data[i];
            if (li == UNSET) {
                return;
            }
            const TranslationLock &lock = data.translation_lock.data[li];
            if (lock.pdrd_body_index != 0 ||
                !axis_enabled(lock.rotation_axis)) {
                return;
            }
            const LockFrame &frame = frames.data[li];
            const Vec3f r =
                (positions.data[i] - lock.anchor).cast<float>() -
                frame.com_relative;
            const Vec3f step(dx.data[3 * i + 0], dx.data[3 * i + 1],
                              dx.data[3 * i + 2]);
            atomic_add_vec3(torque.data, li,
                            data.prop.vertex.data[i].mass * r.cross(step));
        }
        DISPATCH_END;

        std::vector<TranslationLock> locks(nl);
        std::vector<LockFrame> frames_host(nl);
        std::vector<Vec3f> torque_host(nl);
        CUDA_HANDLE_ERROR(cudaMemcpy(locks.data(), data_.translation_lock.data,
                                     nl * sizeof(TranslationLock),
                                     cudaMemcpyDeviceToHost));
        CUDA_HANDLE_ERROR(cudaMemcpy(frames_host.data(), frames_.data,
                                     nl * sizeof(LockFrame),
                                     cudaMemcpyDeviceToHost));
        CUDA_HANDLE_ERROR(cudaMemcpy(torque_host.data(), torque_.data,
                                     nl * sizeof(Vec3f),
                                     cudaMemcpyDeviceToHost));
        constexpr float eps = 1.19209290e-7f;
        for (unsigned li = 0; li < nl; ++li) {
            const TranslationLock &lock = locks[li];
            if (lock.pdrd_body_index != 0 ||
                !axis_enabled(lock.rotation_axis)) {
                continue;
            }
            const Vec3f omega =
                host_matvec3(frames_host[li].inv_inertia, torque_host[li]);
            const float magnitude =
                lock.rotation_mode == ROTATION_LOCK_PROHIBIT_AXIS
                    ? fabsf(lock.rotation_axis.dot(omega))
                    : perpendicular(omega, lock.rotation_axis).norm();
            const float omega_scale = omega.norm();
            const float torque_scale = torque_host[li].norm();
            const float inverse_scale =
                fmaxf(fabsf(frames_host[li].inv_inertia(0, 0)),
                      fmaxf(fabsf(frames_host[li].inv_inertia(1, 1)),
                            fabsf(frames_host[li].inv_inertia(2, 2))));
            const float bound =
                4096.0f * eps *
                fmaxf(1.0f, omega_scale + torque_scale * inverse_scale);
            if (!std::isfinite(magnitude) || magnitude > bound) {
                std::vector<Vec3f> positions_host(positions.size);
                std::vector<float> dx_host(dx.size);
                std::vector<unsigned> lock_index_host(positions.size);
                std::vector<VertexProp> vertex_prop_host(positions.size);
                CUDA_HANDLE_ERROR(cudaMemcpy(
                    positions_host.data(), positions.data,
                    positions.size * sizeof(Vec3f), cudaMemcpyDeviceToHost));
                CUDA_HANDLE_ERROR(cudaMemcpy(
                    dx_host.data(), dx.data, dx.size * sizeof(float),
                    cudaMemcpyDeviceToHost));
                CUDA_HANDLE_ERROR(cudaMemcpy(
                    lock_index_host.data(), data_.translation_lock_index.data,
                    positions.size * sizeof(unsigned), cudaMemcpyDeviceToHost));
                CUDA_HANDLE_ERROR(cudaMemcpy(
                    vertex_prop_host.data(), data_.prop.vertex.data,
                    positions.size * sizeof(VertexProp),
                    cudaMemcpyDeviceToHost));

                double row_sum[2] = {0.0, 0.0};
                double row_abs[2] = {0.0, 0.0};
                const unsigned active_mask = frames_host[li].row_mask >> 2u;
                unsigned contribution_count = 0;
                for (unsigned i = 0; i < positions.size; ++i) {
                    if (lock_index_host[i] != li) {
                        continue;
                    }
                    const Vec3f r =
                        (positions_host[i] - lock.anchor).cast<float>() -
                        frames_host[li].com_relative;
                    const Vec3f u0 = host_matvec3(
                        frames_host[li].inv_inertia,
                        frames_host[li].rotation_basis0);
                    const Vec3f u1 = host_matvec3(
                        frames_host[li].inv_inertia,
                        frames_host[li].rotation_basis1);
                    const float mass = vertex_prop_host[i].mass;
                    const Vec3f coefficient[2] = {
                        mass * u0.cross(r), mass * u1.cross(r)};
                    for (unsigned row = 0; row < 2; ++row) {
                        if (!(active_mask & (1u << row))) {
                            continue;
                        }
                        double contribution = 0.0;
                        for (unsigned component = 0; component < 3;
                             ++component) {
                            contribution +=
                                (double)coefficient[row][component] *
                                (double)dx_host[3 * i + component];
                        }
                        row_sum[row] += contribution;
                        row_abs[row] += std::fabs(contribution);
                    }
                    ++contribution_count;
                }

                // Each constraint-row reduction performs three products and
                // their additions per vertex. The projector and this check
                // each incur one such reduction, so four gamma_n bounds cover
                // both passes plus the fp32 row-coefficient arithmetic.
                const double operations =
                    4.0 * (double)contribution_count + 32.0;
                const double unit_roundoff = (double)eps;
                const double gamma =
                    operations * unit_roundoff /
                    (1.0 - operations * unit_roundoff);
                const double row_bound0 = (active_mask & 1u)
                    ? 4.0 * gamma * std::fmax(1.0, row_abs[0])
                    : 0.0;
                const double row_bound1 = (active_mask & 2u)
                    ? 4.0 * gamma * std::fmax(1.0, row_abs[1])
                    : 0.0;
                const double verified_magnitude = std::sqrt(
                    row_sum[0] * row_sum[0] + row_sum[1] * row_sum[1]);
                const double verified_bound = std::sqrt(
                    row_bound0 * row_bound0 + row_bound1 * row_bound1);
                if (std::isfinite(verified_magnitude) &&
                    verified_magnitude <= verified_bound) {
                    continue;
                }
                ppf_fatal(PPF_FATAL_SOLVER_INVARIANT,
                          "PPF FATAL: rotation lock group %u violated at %s: "
                          "host-verified forbidden constraint-row magnitude "
                          "%.6e exceeds the fp32 reduction bound %.6e "
                          "(device probe %.6e). This is an invariant check "
                          "only; the solver does not snap the state.\n",
                          lock.dmap_index, where, verified_magnitude,
                          verified_bound, (double)magnitude);
            }
        }
    }

  public:
    void initialize_frames(const Vec<Vec3f> &positions) {
        const unsigned nl = lock_count();
        std::vector<TranslationLock> locks(nl);
        CUDA_HANDLE_ERROR(cudaMemcpy(locks.data(), data_.translation_lock.data,
                                     nl * sizeof(TranslationLock),
                                     cudaMemcpyDeviceToHost));
        std::vector<LockFrame> host_frames(nl);
        for (unsigned li = 0; li < nl; ++li) {
            LockFrame frame{};
            const TranslationLock &lock = locks[li];
            if (!rotation_mode_valid(lock.rotation_mode)) {
                ppf_fatal(PPF_FATAL_SOLVER_INVARIANT,
                          "PPF FATAL: rotation lock displacement group %u has "
                          "invalid mode %u.\n",
                          lock.dmap_index, lock.rotation_mode);
            }
            if (axis_enabled(lock.axis)) {
                tangent_basis(lock.axis, frame.translation_basis0,
                              frame.translation_basis1);
                frame.row_mask |= TRANSLATION_ROW0 | TRANSLATION_ROW1;
            }
            if (lock.pdrd_body_index == 0 &&
                axis_enabled(lock.rotation_axis)) {
                if (lock.rotation_mode == ROTATION_LOCK_PROHIBIT_AXIS) {
                    frame.rotation_basis0 = lock.rotation_axis;
                    frame.row_mask |= ROTATION_ROW0;
                } else {
                    tangent_basis(lock.rotation_axis, frame.rotation_basis0,
                                  frame.rotation_basis1);
                    frame.row_mask |= ROTATION_ROW0 | ROTATION_ROW1;
                }
            }
            host_frames[li] = frame;
        }
        CUDA_HANDLE_ERROR(cudaMemcpy(frames_.data, host_frames.data(),
                                     nl * sizeof(LockFrame),
                                     cudaMemcpyHostToDevice));

        auto &pool = buffer::get();
        auto com_buffer = pool.get<Vec3f>(nl);
        auto inertia_buffer = pool.get<Mat3x3f>(nl);
        Vec<Vec3f> com = com_buffer.as_vec();
        Vec<Mat3x3f> inertia = inertia_buffer.as_vec();
        com.clear(Vec3f::Zero());
        inertia.clear(Mat3x3f::Zero());
        const DataSet data = data_;
        DISPATCH_START(positions.size)
        [data, positions, com] __device__(unsigned i) mutable {
            const unsigned li = data.translation_lock_index.data[i];
            if (li == UNSET) {
                return;
            }
            const TranslationLock &lock = data.translation_lock.data[li];
            if (lock.pdrd_body_index != 0 ||
                !axis_enabled(lock.rotation_axis)) {
                return;
            }
            const float mass = data.prop.vertex.data[i].mass;
            const Vec3f relative = (positions.data[i] - lock.anchor).cast<float>();
            atomic_add_vec3(com.data, li, mass * relative);
        }
        DISPATCH_END;
        const Vec<LockFrame> frames = frames_;
        DISPATCH_START(nl)
        [data, com, frames] __device__(unsigned li) mutable {
            if (!axis_enabled(data.translation_lock.data[li].rotation_axis)) {
                return;
            }
            frames.data[li].com_relative =
                com.data[li] / data.translation_lock.data[li].total_mass;
        }
        DISPATCH_END;
        DISPATCH_START(positions.size)
        [data, positions, frames, inertia] __device__(unsigned i) mutable {
            const unsigned li = data.translation_lock_index.data[i];
            if (li == UNSET) {
                return;
            }
            const TranslationLock &lock = data.translation_lock.data[li];
            if (lock.pdrd_body_index != 0 ||
                !axis_enabled(lock.rotation_axis)) {
                return;
            }
            const Vec3f r =
                (positions.data[i] - lock.anchor).cast<float>() -
                frames.data[li].com_relative;
            const float mass = data.prop.vertex.data[i].mass;
            const float r2 = r.dot(r);
            for (unsigned row = 0; row < 3; ++row) {
                for (unsigned col = 0; col < 3; ++col) {
                    const float value =
                        mass * ((row == col ? r2 : 0.0f) - r[row] * r[col]);
                    atomicAdd(&inertia.data[li](row, col), value);
                }
            }
        }
        DISPATCH_END;

        CUDA_HANDLE_ERROR(cudaMemcpy(host_frames.data(), frames_.data,
                                     nl * sizeof(LockFrame),
                                     cudaMemcpyDeviceToHost));
        std::vector<Mat3x3f> host_inertia(nl);
        CUDA_HANDLE_ERROR(cudaMemcpy(host_inertia.data(), inertia.data,
                                     nl * sizeof(Mat3x3f),
                                     cudaMemcpyDeviceToHost));
        for (unsigned li = 0; li < nl; ++li) {
            if (locks[li].pdrd_body_index == 0 &&
                axis_enabled(locks[li].rotation_axis)) {
                host_frames[li].inv_inertia =
                    invert_inertia_host(host_inertia[li], locks[li].dmap_index);
            }
        }
        CUDA_HANDLE_ERROR(cudaMemcpy(frames_.data, host_frames.data(),
                                     nl * sizeof(LockFrame),
                                     cudaMemcpyHostToDevice));
    }

    void assemble_constraints(const Vec<Vec3f> &positions,
                              const Vec<float> &seed) {
        const unsigned nl = lock_count();
        gram_.clear(Mat4x4f::Zero());
        drift_.clear(Vec3f::Zero());
        const DataSet data = data_;
        const Vec<unsigned> mask = dof_mask_;
        const Vec<LockFrame> frames = frames_;
        const Vec<Mat4x4f> gram = gram_;
        const Vec<Vec3f> drift = drift_;
        DISPATCH_START(nl)
        [frames] __device__(unsigned li) mutable {
            frames.data[li].fixed = Vec4f::Zero();
            frames.data[li].rhs = Vec4f::Zero();
        }
        DISPATCH_END;
        DISPATCH_START(positions.size)
        [data, mask, positions, seed, frames, gram,
         drift] __device__(unsigned i) mutable {
            const unsigned li = data.translation_lock_index.data[i];
            if (li == UNSET) {
                return;
            }
            const TranslationLock &lock = data.translation_lock.data[li];
            const LockFrame &frame = frames.data[li];
            const float mass = data.prop.vertex.data[i].mass;
            if (axis_enabled(lock.axis)) {
                const Vec3f delta =
                    (positions.data[i] - data.translation_lock_initial.data[i])
                        .cast<float>();
                atomic_add_vec3(drift.data, li,
                                mass * perpendicular(delta, lock.axis));
            }
            // PDRD aggregate DOFs are projected in the reduced six-vector.
            if (lock.pdrd_body_index != 0) {
                return;
            }
            const RowCoefficients c =
                row_coefficients(lock, frame, positions.data[i], mass);
            if (mask.data[i] != 0u) {
                const Vec3f p(seed.data[3 * i + 0], seed.data[3 * i + 1],
                              seed.data[3 * i + 2]);
                atomic_add_vec4(&frames.data[li].fixed, 0u,
                                rows_times_vector(c, p));
                return;
            }
            Mat4x4f outer = Mat4x4f::Zero();
            for (unsigned row = 0; row < 4; ++row) {
                for (unsigned col = 0; col < 4; ++col) {
                    outer(row, col) = c.row[row].dot(c.row[col]);
                    atomicAdd(&gram.data[li](row, col), outer(row, col));
                }
            }
        }
        DISPATCH_END;

        std::vector<TranslationLock> locks(nl);
        std::vector<LockFrame> frames_host(nl);
        std::vector<Mat4x4f> gram_host(nl);
        std::vector<Vec3f> drift_host(nl);
        CUDA_HANDLE_ERROR(cudaMemcpy(locks.data(), data_.translation_lock.data,
                                     nl * sizeof(TranslationLock),
                                     cudaMemcpyDeviceToHost));
        CUDA_HANDLE_ERROR(cudaMemcpy(frames_host.data(), frames_.data,
                                     nl * sizeof(LockFrame),
                                     cudaMemcpyDeviceToHost));
        CUDA_HANDLE_ERROR(cudaMemcpy(gram_host.data(), gram_.data,
                                     nl * sizeof(Mat4x4f),
                                     cudaMemcpyDeviceToHost));
        CUDA_HANDLE_ERROR(cudaMemcpy(drift_host.data(), drift_.data,
                                     nl * sizeof(Vec3f),
                                     cudaMemcpyDeviceToHost));
        constexpr float eps = 1.19209290e-7f;
        for (unsigned li = 0; li < nl; ++li) {
            LockFrame &frame = frames_host[li];
            const TranslationLock &lock = locks[li];
            Vec4f rhs = -frame.fixed;
            if (axis_enabled(lock.axis)) {
                rhs[0] += frame.translation_basis0.dot(drift_host[li]);
                rhs[1] += frame.translation_basis1.dot(drift_host[li]);
            }
            frame.rhs = rhs;
            if (lock.pdrd_body_index != 0) {
                continue;
            }
            frame.gram_pinv = pseudoinverse_gram_host(gram_host[li]);
            const Vec4f lambda = host_matvec4(frame.gram_pinv, rhs);
            const Vec4f resolved = host_matvec4(gram_host[li], lambda);
            const float residual = (rhs - resolved).norm();
            const float scale = fmaxf(1.0f, rhs.norm());
            const float bound = 4096.0f * eps * scale;
            if (!std::isfinite(residual) || residual > bound) {
                ppf_fatal(PPF_FATAL_SOLVER_INVARIANT,
                          "PPF FATAL: aggregate lock group %u is infeasible: "
                          "exact fix pins leave a constraint residual %.6e "
                          "outside the free tangent space (bound %.6e).\n",
                          lock.dmap_index, (double)residual, (double)bound);
            }
        }
        CUDA_HANDLE_ERROR(cudaMemcpy(frames_.data, frames_host.data(),
                                     nl * sizeof(LockFrame),
                                     cudaMemcpyHostToDevice));
    }

  private:
    const DataSet &data_;
    Vec<unsigned> dof_mask_;
    Vec<LockFrame> frames_;
    Vec<Mat4x4f> gram_;
    mutable Vec<Vec4f> sums_;
    Vec<Vec3f> drift_;
    mutable Vec<Vec3f> torque_;
    Vec<Vec3f> positions_;
};

// Verify the absolute translation invariant without modifying positions.
// Rotation has no absolute-pose invariant; FullProjector::check_tangent verifies
// its solved tangent direction before the line search applies it.
inline void check_invariant(const DataSet &data, const Vec<Vec3f> &positions,
                            const char *where) {
    const unsigned nl = data.translation_lock.size;
    if (nl == 0) {
        return;
    }
    auto sums_buf = buffer::get().get<Vec3f>(nl);
    auto max_buf = buffer::get().get<float>(nl);
    Vec<Vec3f> sums = sums_buf.as_vec();
    Vec<float> max_disp = max_buf.as_vec();
    sums.clear(Vec3f::Zero());
    max_disp.clear(0.0f);
    DISPATCH_START(positions.size)
    [data, positions, sums, max_disp] __device__(unsigned i) mutable {
        const unsigned li = data.translation_lock_index.data[i];
        if (li == UNSET) {
            return;
        }
        const TranslationLock &lock = data.translation_lock.data[li];
        if (!axis_enabled(lock.axis)) {
            return;
        }
        const Vec3f delta =
            (positions.data[i] - data.translation_lock_initial.data[i])
                .cast<float>();
        atomic_add_vec3(sums.data, li,
                        data.prop.vertex.data[i].mass *
                            perpendicular(delta, lock.axis));
        atomicMax(reinterpret_cast<int *>(&max_disp.data[li]),
                  __float_as_int(fmaxf(fabsf(delta[0]),
                                       fmaxf(fabsf(delta[1]), fabsf(delta[2])))));
    }
    DISPATCH_END;
    std::vector<Vec3f> sums_host(nl);
    std::vector<float> max_host(nl);
    std::vector<TranslationLock> locks(nl);
    CUDA_HANDLE_ERROR(cudaMemcpy(sums_host.data(), sums.data,
                                 nl * sizeof(Vec3f), cudaMemcpyDeviceToHost));
    CUDA_HANDLE_ERROR(cudaMemcpy(max_host.data(), max_disp.data,
                                 nl * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_HANDLE_ERROR(cudaMemcpy(locks.data(), data.translation_lock.data,
                                 nl * sizeof(TranslationLock),
                                 cudaMemcpyDeviceToHost));
    // Absolute floor on the bound, so a group sitting near the origin is not
    // held to a purely relative tolerance (the fp32 term below vanishes with
    // the displacement magnitude, and a residual is never resolvable below
    // this scale).
    constexpr float abs_floor = 1.0f / 134217728.0f;
    constexpr float fp32_eps = 1.19209290e-7f;
    for (unsigned li = 0; li < nl; ++li) {
        const TranslationLock &lock = locks[li];
        if (!axis_enabled(lock.axis)) {
            continue;
        }
        const float residual = sums_host[li].norm() / lock.total_mass;
        const float bound =
            8.0f * abs_floor + 256.0f * fp32_eps * fmaxf(1.0f, max_host[li]);
        if (!std::isfinite(residual) || residual > bound) {
            ppf_fatal(PPF_FATAL_SOLVER_INVARIANT,
                      "PPF FATAL: translation lock group %u violated at %s: "
                      "perpendicular COM drift %.6e exceeds the fp32 "
                      "round-off bound %.6e. The solver never snaps this "
                      "state; inspect the constrained Newton direction.\n",
                      lock.dmap_index, where, (double)residual, (double)bound);
        }
    }
}

} // namespace translation_lock

#endif
