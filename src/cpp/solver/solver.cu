// File: solver.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../buffer/buffer.hpp"
#include "../csrmat/csrmat.hpp"
#include "../kernels/reduce.hpp"
#include "../kernels/vec_ops.hpp"
#include "../utility/dispatcher.hpp"
#include "../utility/utility.hpp"
#include "solver.hpp"

namespace solver {

struct UnrolledMat3x3f {
    const float *data;
    __device__ UnrolledMat3x3f(const float *data) : data(data) {}
    __device__ Vec3f operator*(const float *b) const {
        Vec3f result;
        result[0] = data[0] * b[0] + data[3] * b[1] + data[6] * b[2];
        result[1] = data[1] * b[0] + data[4] * b[1] + data[7] * b[2];
        result[2] = data[2] * b[0] + data[5] * b[1] + data[8] * b[2];
        return result;
    }
    __device__ Vec3f operator^(const float *b) const {
        Vec3f result;
        result[0] = data[0] * b[0] + data[1] * b[1] + data[2] * b[2];
        result[1] = data[3] * b[0] + data[4] * b[1] + data[5] * b[2];
        result[2] = data[6] * b[0] + data[7] * b[1] + data[8] * b[2];
        return result;
    }
};

void apply(const DynCSRMat &A, const FixedCSRMat &B, const Vec<Mat3x3f> &C,
           float D, const Vec<float> &x, Vec<float> &result) {
    DISPATCH_START(A.nrow)
    [A, B, C, D, result, x] __device__(unsigned i) mutable {
        Vec3f sum = Vec3f::Zero();
        for (unsigned k = 0; k < A.rows[i].head; ++k) {
            const float *m =
                reinterpret_cast<const float *>(A.rows[i].value + k);
            unsigned j = A.rows[i].index[k];
            sum += UnrolledMat3x3f(m) * (x.data + 3 * j);
        }
        for (unsigned k = 0; k < A.rows[i].ref_head; ++k) {
            const float *m = reinterpret_cast<const float *>(
                A.dyn_value_buff.data + A.rows[i].ref_value[k]);
            unsigned j = A.rows[i].ref_index[k];
            sum += UnrolledMat3x3f(m) ^ (x.data + 3 * j);
        }
        for (unsigned k = B.index.offset[i]; k < B.index.offset[i + 1]; ++k) {
            const float *m = reinterpret_cast<const float *>(B.value.data + k);
            unsigned j = B.index.data[k];
            sum += UnrolledMat3x3f(m) * (x.data + 3 * j);
        }
        for (unsigned k = B.transpose.offset[i]; k < B.transpose.offset[i + 1];
             ++k) {
            Vec2u ref = B.transpose.data[k];
            const float *m =
                reinterpret_cast<const float *>(B.value.data + ref[1]);
            sum += UnrolledMat3x3f(m) ^ (x.data + 3 * ref[0]);
        }
        sum += UnrolledMat3x3f(C[i].data()) * (x.data + 3 * i);
        if (D) {
            for (unsigned k = 0; k < 3; ++k) {
                sum[k] += D * x[3 * i + k];
            }
        }
        Map<Vec3f>(result.data + 3 * i) = sum;
    } DISPATCH_END;
}

class DeviceOperators {
  public:
    DeviceOperators(const DynCSRMat &A, const FixedCSRMat &B,
                    const Vec<Mat3x3f> &C, const Vec<Mat3x3f> &P)
        : A(A), B(B), C(C), P(P) {}
    void apply(const Vec<float> &x, Vec<float> &result) const {
        const DynCSRMat &A = this->A;
        const FixedCSRMat &B = this->B;
        const Vec<Mat3x3f> &C = this->C;
        solver::apply(A, B, C, 0.0f, x, result);
    }
    void precond(const Vec<float> &x, Vec<float> &result) const {
        const Vec<Mat3x3f> &inv_diag = this->P;
        DISPATCH_START(A.nrow)
        [x, result, inv_diag] __device__(unsigned i) mutable {
            Map<Vec3f>(result.data + 3 * i) =
                inv_diag[i] * Map<Vec3f>(x.data + 3 * i);
        } DISPATCH_END;
    }
    float norm(const Vec<float> &r, Vec<float> &tmp) const {
        DISPATCH_START(r.size)
        [r, tmp] __device__(unsigned i) mutable {
            tmp[i] = fabs(r[i]);
        } DISPATCH_END;
        return kernels::sum_array(tmp.data, r.size);
    }
    const DynCSRMat &A;
    const FixedCSRMat &B;
    const Vec<Mat3x3f> &C;
    const Vec<Mat3x3f> &P;
};

__device__ static Mat3x3f invert(const Mat3x3f &m) {
    float det = m(0, 0) * m(1, 1) * m(2, 2) - m(0, 0) * m(1, 2) * m(1, 2) -
                m(0, 1) * m(0, 1) * m(2, 2) +
                2.0f * m(0, 1) * m(0, 2) * m(1, 2) -
                m(0, 2) * m(0, 2) * m(1, 1);
    Mat3x3f minv;
    if (det) {
        float invdet = 1.0f / det;
        minv(0, 0) = (m(1, 1) * m(2, 2) - m(1, 2) * m(1, 2)) * invdet;
        minv(0, 1) = (m(0, 2) * m(1, 2) - m(0, 1) * m(2, 2)) * invdet;
        minv(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * invdet;
        minv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(0, 2)) * invdet;
        minv(1, 2) = (m(0, 1) * m(0, 2) - m(0, 0) * m(1, 2)) * invdet;
        minv(2, 2) = (m(0, 0) * m(1, 1) - m(0, 1) * m(0, 1)) * invdet;
        minv(1, 0) = minv(0, 1);
        minv(2, 0) = minv(0, 2);
        minv(2, 1) = minv(1, 2);
    } else {
        minv = Mat3x3f::Zero();
        for (unsigned j = 0; j < 3; j++) {
            minv(j, j) = 1.0f / m(j, j);
        }
    }
    return minv;
}

std::tuple<bool, unsigned, float> cg(const DeviceOperators &op, Vec<float> &r,
                                     Vec<float> &x, unsigned max_iter,
                                     float tol) {
    unsigned vertex_count = op.A.nrow;
    buffer::MemoryPool &pool = buffer::get();
    auto tmp = pool.get<float>(3 * vertex_count);
    auto z = pool.get<float>(3 * vertex_count);
    auto p = pool.get<float>(3 * vertex_count);
    auto r0 = pool.get<float>(3 * vertex_count);

    op.apply(x, tmp);
    kernels::add_scaled(tmp.data, r.data, -1.0f, r.size);
    kernels::copy(r.data, r0.data, r0.size);
    op.precond(r, z);
    kernels::copy(z.data, p.data, p.size);

    unsigned iter = 1;
    double rz0 = kernels::inner_product(r.data, z.data, r.size);

    double err0 = op.norm(r, tmp);
    if (!err0) {
        return {true, iter, 0.0f};
    } else {
        while (true) {
            op.apply(p, tmp);
            double alpha = rz0 / (double)kernels::inner_product(p.data, tmp.data, p.size);
            kernels::add_scaled(p.data, x.data, (float)alpha, x.size);
            kernels::add_scaled(tmp.data, r.data, (float)-alpha, r.size);
            double err = op.norm(r, tmp);
            double reresid = err / err0;
            if (reresid < tol) {
                return {true, iter, reresid};
            } else if (iter >= max_iter || std::isnan(reresid)) {
                return {false, iter, reresid};
            }
            op.precond(r, z);
            double rz1 = kernels::inner_product(r.data, z.data, r.size);
            double beta = rz1 / rz0;
            kernels::combine(z.data, p.data, p.data, 1.0f, (float)beta, p.size);
            rz0 = rz1;
            iter++;
        }
    }
    // PooledVec buffers auto-release when exiting function
}

bool solve(const DynCSRMat &A, const FixedCSRMat &B, const Vec<Mat3x3f> &C,
           Vec<float> b, float tol, unsigned max_iter, Vec<float> x,
           unsigned &iter, float &resid) {

    unsigned vertex_count = A.nrow;
    buffer::MemoryPool &pool = buffer::get();
    auto inv_diag = pool.get<Mat3x3f>(vertex_count);

    Vec<Mat3x3f> inv_diag_vec = inv_diag.as_vec();
    DISPATCH_START(A.nrow)
    [A, B, C, inv_diag_vec] __device__(unsigned i) mutable {
        inv_diag_vec[i] = invert(A(i, i) + B(i, i) + C[i]);
    } DISPATCH_END;

    DeviceOperators ops(A, B, C, inv_diag_vec);
    bool success;
    std::tie(success, iter, resid) = cg(ops, b, x, max_iter, tol);

    // PooledVec auto-releases when exiting function
    return success;
}

} // namespace solver
