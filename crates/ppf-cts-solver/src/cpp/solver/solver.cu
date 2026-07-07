// File: solver.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../buffer/buffer.hpp"
#include "../csrmat/csrmat.hpp"
#include "../kernels/reduce.hpp"
#include "../kernels/vec_ops.hpp"
#include "../schwarz/schwarz.hpp"
#include "../utility/dispatcher.hpp"
#include "../utility/utility.hpp"
#include "../energy/model/pdrd_rigid.hpp"
#include "solver.hpp"
#include <cmath>
#include <cstdlib>

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
           float D, const Vec<float> &x, Vec<float> &result,
           cudaStream_t queue = 0) {
    // Issued on `queue`. Default-stream callers (queue 0: host cg(), cg_rigid,
    // the self-test) are synced at the end below, preserving the original
    // synchronizing-dispatch behavior so none of them can race on `result`. The
    // PCG fast path passes its own stream and skips the sync, so a whole CG
    // iteration chains with no host round-trip.
    DISPATCH_QUEUE_START(A.nrow, queue)
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
    } DISPATCH_QUEUE_END;
    if (queue == 0) {
        CUDA_HANDLE_ERROR(cudaStreamSynchronize(0));
    }
}

class DeviceOperators {
  public:
    DeviceOperators(const DynCSRMat &A, const FixedCSRMat &B,
                    const Vec<Mat3x3f> &C, const Vec<Mat3x3f> &P)
        : A(A), B(B), C(C), P(P) {}
    void apply(const Vec<float> &x, Vec<float> &result,
               cudaStream_t queue = 0) const {
        const DynCSRMat &A = this->A;
        const FixedCSRMat &B = this->B;
        const Vec<Mat3x3f> &C = this->C;
        solver::apply(A, B, C, 0.0f, x, result, queue);
    }
    void precond(const Vec<float> &x, Vec<float> &result,
                 cudaStream_t queue = 0) const {
        // Base preconditioner writes result = M_base^-1 x. PDRD scenes never
        // reach this path (they solve in reduced coordinates and return earlier
        // in solve()); this is the cloth / general PCG preconditioner.
        if (H && !force_bj) {
            // Aggregate-Schwarz base over the assembled operator
            // M = A_dyn + B_fixed + C_diag. SPD by construction; the rz<=0
            // force_bj fallback remains as a defensive net. Runs on `queue`: the
            // device-resident PCG loop passes its own stream so the apply chains
            // sync-free, while default-stream callers are synced below.
            schwarz::apply(*H, x, result, queue);
            if (queue == 0) {
                CUDA_HANDLE_ERROR(cudaStreamSynchronize(0));
            }
        } else {
            // 3x3 block-Jacobi base preconditioner: result = M_base^-1 x.
            const Vec<Mat3x3f> &inv_diag = this->P;
            DISPATCH_QUEUE_START(A.nrow, queue)
            [x, result, inv_diag] __device__(unsigned i) mutable {
                // Use the column-major UnrolledMat3x3f matvec like apply()
                // above; the device Eigen Mat3x3f * Map<Vec3f> form is silently
                // wrong on this backend. inv_diag is symmetric (built by
                // invert()), so layout is moot, but this keeps the one matvec
                // here consistent with the proven path.
                Map<Vec3f>(result.data + 3 * i) =
                    UnrolledMat3x3f(inv_diag[i].data()) * (x.data + 3 * i);
            } DISPATCH_QUEUE_END;
            // Default-stream callers (the host cg() block-Jacobi fallback) are
            // synced here so they cannot race on `result`; the fast path passes
            // its own stream and stays sync-free.
            if (queue == 0) {
                CUDA_HANDLE_ERROR(cudaStreamSynchronize(0));
            }
        }
    }
    void set_schwarz(const schwarz::SchwarzHierarchy *h) { H = h; }
    float norm(const Vec<float> &r, Vec<float> &tmp) const {
        DISPATCH_START(r.size)
        [r, tmp] __device__(unsigned i) mutable {
            tmp[i] = fabsf(r[i]);
        } DISPATCH_END;
        return kernels::sum_array(tmp.data, r.size);
    }
    // Device-landing L1 norm for the sync-free fast path: |r| into tmp, then a
    // reduction that leaves the scalar at device address `out`. Same value as
    // norm() above, but no device-to-host copy.
    void norm_into(const Vec<float> &r, Vec<float> &tmp, float *out,
                   cudaStream_t queue) const {
        DISPATCH_QUEUE_START(r.size, queue)
        [r, tmp] __device__(unsigned i) mutable {
            tmp[i] = fabsf(r[i]);
        } DISPATCH_QUEUE_END;
        kernels::sum_into(tmp.data, out, r.size, queue);
    }
    const DynCSRMat &A;
    const FixedCSRMat &B;
    const Vec<Mat3x3f> &C;
    const Vec<Mat3x3f> &P;
    // Optional aggregate-Schwarz preconditioner. nullptr -> block-Jacobi base.
    // Set via set_schwarz() when ParamSet::precond == Schwarz.
    const schwarz::SchwarzHierarchy *H{nullptr};
    // Latched by cg() if the Schwarz base ever yields rz <= 0; forces the
    // SPD-safe block-Jacobi branch for the remainder of that solve.
    mutable bool force_bj{false};
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

// One persistent blocking stream carries the device-resident PCG inner loop
// (block-Jacobi or Schwarz; the Schwarz preconditioner runs on it via
// op.precond). A blocking stream implicitly orders with prior legacy-stream work
// (the matrix assembly that produced A/B/C and the residual), so the loop sees a
// consistent operator without an explicit barrier. Created lazily; lives for the
// process.
static cudaStream_t cg_device_stream() {
    static cudaStream_t s = nullptr;
    if (!s) {
        CUDA_HANDLE_ERROR(cudaStreamCreate(&s));
    }
    return s;
}

// ---- Fused CG iteration kernels (block-Jacobi path) --------------------------
// The per-iteration vector work was ~10 small launches (two 2-launch
// reductions, two axpys, the precond, the |r| pass, and three scalar kernels),
// each a few microseconds of graph-node/launch latency that dominates the
// non-SpMV time at small system sizes. cg_fused_update_kernel does the x/r
// updates, the block-Jacobi z = P r, and the block partials of BOTH r.z and
// ||r||_1 in ONE pass (r is read/written once instead of three times);
// cg_reduce2_kernel folds both partial arrays in one launch (block 0 -> rz,
// block 1 -> err); cg_beta_kernel fuses beta = rz1/rz0 (scalar_div semantics,
// including the breakdown flag) with the rz0 <- rz1 roll. Summation order
// differs from inner_product_kernel_optimized by reduction shape only; the
// trajectory stays within the run-to-run band the atomicAdd assembly already
// has.
static constexpr unsigned CGF_BLOCK = 256;

__global__ void cg_fused_update_kernel(const float *p, const float *Ap,
                                       float *x, float *r, float *z,
                                       const Mat3x3f *inv_diag,
                                       const float *d_rz0, const float *d_pAp,
                                       int *breakdown, unsigned nrow,
                                       float *rz_partials,
                                       float *err_partials) {
    __shared__ float s_rz[CGF_BLOCK];
    __shared__ float s_err[CGF_BLOCK];
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float rz = 0.0f, err = 0.0f;
    if (i < nrow) {
        // alpha = rz0 / pAp computed per thread from the device scalars
        // (read-only, all threads agree), replicating scalar_div's breakdown
        // semantics; folds the former scalar_div launch into this kernel.
        const float pAp = *d_pAp;
        float alpha;
        if (pAp > 0.0f) {
            alpha = (*d_rz0) / pAp;
        } else {
            alpha = 0.0f;
            if (breakdown && i == 0) {
                *breakdown = 1;
            }
        }
        float ri[3];
        for (unsigned k = 0; k < 3; ++k) {
            const unsigned j = 3 * i + k;
            x[j] += alpha * p[j];
            ri[k] = r[j] - alpha * Ap[j];
            r[j] = ri[k];
        }
        const Vec3f zi = UnrolledMat3x3f(inv_diag[i].data()) * ri;
        for (unsigned k = 0; k < 3; ++k) {
            z[3 * i + k] = zi[k];
            rz += ri[k] * zi[k];
            err += fabsf(ri[k]);
        }
    }
    s_rz[threadIdx.x] = rz;
    s_err[threadIdx.x] = err;
    __syncthreads();
    for (unsigned w = CGF_BLOCK / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            s_rz[threadIdx.x] += s_rz[threadIdx.x + w];
            s_err[threadIdx.x] += s_err[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        rz_partials[blockIdx.x] = s_rz[0];
        err_partials[blockIdx.x] = s_err[0];
    }
}

__global__ void cg_reduce2_kernel(const float *rz_partials,
                                  const float *err_partials, unsigned count,
                                  float *d_rz1, float *d_err, float *d_beta,
                                  float *d_rz0, int *breakdown, int *d_rzbad,
                                  int check_spd) {
    __shared__ float s[CGF_BLOCK];
    const float *src = (blockIdx.x == 0) ? rz_partials : err_partials;
    float acc = 0.0f;
    for (unsigned k = threadIdx.x; k < count; k += CGF_BLOCK) {
        acc += src[k];
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned w = CGF_BLOCK / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            s[threadIdx.x] += s[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) {
            // rz1 landing plus the scalar tail of the iteration, folded here:
            // beta = rz1 / rz0 (scalar_div semantics incl. the breakdown
            // flag), the SPD sentinel, and the rz0 <- rz1 roll. Single
            // reader/writer, so the read-then-write order is safe.
            const float rz1 = s[0];
            *d_rz1 = rz1;
            if (check_spd && rz1 <= 0.0f) {
                *d_rzbad = 1;
            }
            const float rz0 = *d_rz0;
            if (rz0 > 0.0f) {
                *d_beta = rz1 / rz0;
            } else {
                *d_beta = 0.0f;
                if (breakdown) {
                    *breakdown = 1;
                }
            }
            *d_rz0 = rz1;
        } else {
            *d_err = s[0];
        }
    }
}

// x/r update for the external-preconditioner (Schwarz) path: x += alpha p,
// r -= alpha Ap with alpha computed in-thread (scalar_div semantics), plus the
// ||r||_1 block partials. z = M^-1 r is produced afterwards by schwarz::apply,
// and r.z by an inner product over its output.
__global__ void cg_update_xr_kernel(const float *p, const float *Ap, float *x,
                                    float *r, const float *d_rz0,
                                    const float *d_pAp, int *breakdown,
                                    unsigned n, float *err_partials) {
    __shared__ float s_err[CGF_BLOCK];
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    float err = 0.0f;
    if (j < n) {
        const float pAp = *d_pAp;
        float alpha;
        if (pAp > 0.0f) {
            alpha = (*d_rz0) / pAp;
        } else {
            alpha = 0.0f;
            if (breakdown && j == 0) {
                *breakdown = 1;
            }
        }
        x[j] += alpha * p[j];
        const float rj = r[j] - alpha * Ap[j];
        r[j] = rj;
        err = fabsf(rj);
    }
    s_err[threadIdx.x] = err;
    __syncthreads();
    for (unsigned w = CGF_BLOCK / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            s_err[threadIdx.x] += s_err[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        err_partials[blockIdx.x] = s_err[0];
    }
}

// Single-block strided fold of `count` partials into one device scalar (one
// launch, vs the generic multi-launch reduce chain).
__global__ void cg_reduce1_kernel(const float *partials, unsigned count,
                                  float *out) {
    __shared__ float s[CGF_BLOCK];
    float acc = 0.0f;
    for (unsigned k = threadIdx.x; k < count; k += CGF_BLOCK) {
        acc += partials[k];
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned w = CGF_BLOCK / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            s[threadIdx.x] += s[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *out = s[0];
    }
}

// Fused symmetric SpMV + partial dot, 8 lanes per row. The dyn and fixed
// matrices store each off-diagonal block ONCE (upper triangle: push() keeps
// row <= col; the ref/transpose lists are mirrors into the same value buffer).
// Instead of walking both the direct and mirror lists (reading every block
// value twice plus 8 B of ref indirection per pair), each stored block (i,j)
// is read once and scattered BOTH ways: H p_j accumulates into the row sum and
// H^T p_i is atomically added to out_j (SRBK-style symmetric SpMV, Huang 2025).
// The p.(Mp) partial comes for free: each off-diagonal block contributes
// 2 p_i . (H p_j) and each diagonal block p_i . (H p_i), independent of the
// output vector's completion, so the dot fusion survives the scatter. The
// caller zeroes `result` before the launch; atomic accumulation makes the
// result nondeterministic in summation order (same tolerance band as the
// atomicAdd assembly).
static constexpr unsigned CG_ROW_LANES = 8;

__global__ void cg_apply_dot_sym_kernel(DynCSRMat A, FixedCSRMat B, Vec<Mat3x3f> C,
                                    Vec<float> p, Vec<float> result,
                                    float *dot_partials) {
    __shared__ float s_dot[CGF_BLOCK];
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned i = t / CG_ROW_LANES;
    const unsigned lane = threadIdx.x % CG_ROW_LANES;
    float dot = 0.0f;
    Vec3f sum = Vec3f::Zero();
    if (i < A.nrow) {
        const float *xi = p.data + 3 * i;
        const unsigned head = A.rows[i].head;
        for (unsigned k = lane; k < head; k += CG_ROW_LANES) {
            const float *m =
                reinterpret_cast<const float *>(A.rows[i].value + k);
            const unsigned j = A.rows[i].index[k];
            const UnrolledMat3x3f H(m);
            const Vec3f v = H * (p.data + 3 * j);
            sum += v;
            if (j != i) {
                const Vec3f w = H ^ xi;
                atomicAdd(result.data + 3 * j + 0, w[0]);
                atomicAdd(result.data + 3 * j + 1, w[1]);
                atomicAdd(result.data + 3 * j + 2, w[2]);
                dot += 2.0f * (xi[0] * v[0] + xi[1] * v[1] + xi[2] * v[2]);
            } else {
                dot += xi[0] * v[0] + xi[1] * v[1] + xi[2] * v[2];
            }
        }
        const unsigned b0 = B.index.offset[i], b1 = B.index.offset[i + 1];
        for (unsigned k = b0 + lane; k < b1; k += CG_ROW_LANES) {
            const float *m = reinterpret_cast<const float *>(B.value.data + k);
            const unsigned j = B.index.data[k];
            const UnrolledMat3x3f H(m);
            const Vec3f v = H * (p.data + 3 * j);
            sum += v;
            if (j != i) {
                const Vec3f w = H ^ xi;
                atomicAdd(result.data + 3 * j + 0, w[0]);
                atomicAdd(result.data + 3 * j + 1, w[1]);
                atomicAdd(result.data + 3 * j + 2, w[2]);
                dot += 2.0f * (xi[0] * v[0] + xi[1] * v[1] + xi[2] * v[2]);
            } else {
                dot += xi[0] * v[0] + xi[1] * v[1] + xi[2] * v[2];
            }
        }
    }
    // Width-8 shuffle reduction of the row sum; unguarded so tail warps never
    // diverge at the full-mask sync (out-of-range lane groups contribute 0).
    for (unsigned off = CG_ROW_LANES / 2; off > 0; off >>= 1) {
        for (unsigned c = 0; c < 3; ++c) {
            sum[c] += __shfl_down_sync(0xffffffffu, sum[c], off, CG_ROW_LANES);
        }
    }
    if (i < A.nrow && lane == 0) {
        const float *xi = p.data + 3 * i;
        const Vec3f v = UnrolledMat3x3f(C[i].data()) * xi;
        sum += v;
        dot += xi[0] * v[0] + xi[1] * v[1] + xi[2] * v[2];
        // The row's own accumulation must also be atomic: transpose scatters
        // from other rows target the same slots.
        atomicAdd(result.data + 3 * i + 0, sum[0]);
        atomicAdd(result.data + 3 * i + 1, sum[1]);
        atomicAdd(result.data + 3 * i + 2, sum[2]);
    }
    s_dot[threadIdx.x] = dot;
    __syncthreads();
    for (unsigned w = CGF_BLOCK / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            s_dot[threadIdx.x] += s_dot[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dot_partials[blockIdx.x] = s_dot[0];
    }
}

// Both-triangles row walk (the pre-symmetric form): each row reads its direct
// blocks plus the mirror (ref / transpose) lists, no atomics, deterministic.
// Faster than the symmetric scatter on SMALL systems, where the mirror value
// reads are L2-served and the atomic traffic dominates instead (measured:
// trapped 45k rows: walk 8.34 vs sym 9.65 ms/solve; twist ~200k rows: walk
// 94 vs sym 86 ms/step). Selected by CG_SYM_SPMV_MIN_ROWS below.
__global__ void cg_apply_dot_walk_kernel(DynCSRMat A, FixedCSRMat B,
                                         Vec<Mat3x3f> C, Vec<float> p,
                                         Vec<float> result,
                                         float *dot_partials) {
    __shared__ float s_dot[CGF_BLOCK];
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned i = t / CG_ROW_LANES;
    const unsigned lane = threadIdx.x % CG_ROW_LANES;
    float dot = 0.0f;
    Vec3f sum = Vec3f::Zero();
    if (i < A.nrow) {
        const unsigned head = A.rows[i].head;
        for (unsigned k = lane; k < head; k += CG_ROW_LANES) {
            const float *m =
                reinterpret_cast<const float *>(A.rows[i].value + k);
            unsigned j = A.rows[i].index[k];
            sum += UnrolledMat3x3f(m) * (p.data + 3 * j);
        }
        const unsigned ref_head = A.rows[i].ref_head;
        for (unsigned k = lane; k < ref_head; k += CG_ROW_LANES) {
            const float *m = reinterpret_cast<const float *>(
                A.dyn_value_buff.data + A.rows[i].ref_value[k]);
            unsigned j = A.rows[i].ref_index[k];
            sum += UnrolledMat3x3f(m) ^ (p.data + 3 * j);
        }
        const unsigned b0 = B.index.offset[i], b1 = B.index.offset[i + 1];
        for (unsigned k = b0 + lane; k < b1; k += CG_ROW_LANES) {
            const float *m = reinterpret_cast<const float *>(B.value.data + k);
            unsigned j = B.index.data[k];
            sum += UnrolledMat3x3f(m) * (p.data + 3 * j);
        }
        const unsigned t0 = B.transpose.offset[i], t1 = B.transpose.offset[i + 1];
        for (unsigned k = t0 + lane; k < t1; k += CG_ROW_LANES) {
            Vec2u ref = B.transpose.data[k];
            const float *m =
                reinterpret_cast<const float *>(B.value.data + ref[1]);
            sum += UnrolledMat3x3f(m) ^ (p.data + 3 * ref[0]);
        }
    }
    for (unsigned off = CG_ROW_LANES / 2; off > 0; off >>= 1) {
        for (unsigned c = 0; c < 3; ++c) {
            sum[c] += __shfl_down_sync(0xffffffffu, sum[c], off, CG_ROW_LANES);
        }
    }
    if (i < A.nrow && lane == 0) {
        sum += UnrolledMat3x3f(C[i].data()) * (p.data + 3 * i);
        for (unsigned k = 0; k < 3; ++k) {
            result[3 * i + k] = sum[k];
            dot += p[3 * i + k] * sum[k];
        }
    }
    s_dot[threadIdx.x] = dot;
    __syncthreads();
    for (unsigned w = CGF_BLOCK / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            s_dot[threadIdx.x] += s_dot[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dot_partials[blockIdx.x] = s_dot[0];
    }
}

// Row-count crossover for the symmetric scatter SpMV (see the two kernels).
static constexpr unsigned CG_SYM_SPMV_MIN_ROWS = 100000;

__global__ void cg_beta_kernel(float *d_beta, float *d_rz0, const float *d_rz1,
                               int *breakdown) {
    const float num = *d_rz1;
    const float den = *d_rz0;
    if (den > 0.0f) {
        *d_beta = num / den;
    } else {
        *d_beta = 0.0f;
        if (breakdown) {
            *breakdown = 1;
        }
    }
    *d_rz0 = num;
}

// Device-resident PCG, used for both the block-Jacobi and (with check_spd) the
// Schwarz preconditioner. The same recurrence as the host cg() loop, numerically
// equivalent up to float rounding (FMA AXPYs and float32 coefficient divisions,
// each <= 1 ULP, so iteration counts and the final residual stay within the
// existing float-reduction noise). Every per-iteration scalar (rz, pAp, alpha,
// beta, residual) stays on the device, so an iteration issues as one unbroken
// stream of kernels with no host round-trip; the host reads the residual only
// periodically (every RESID_CHECK_STRIDE iterations far from the tolerance, then
// every iteration once near it) so convergence detection matches cg()'s. The
// iteration is recorded into a CUDA graph and replayed, collapsing its launches
// into one cudaGraphLaunch (with a direct-launch fallback). When check_spd is set
// (the Schwarz path) a one-thread kernel flags rz <= 0 on the device; the batched
// host read returns that as the 4th tuple element so the caller can fall back to
// the block-Jacobi loop. Block-Jacobi is SPD (rz > 0), so it passes
// check_spd = false and the 4th element is always false.
static std::tuple<bool, unsigned, float, bool>
cg_device(const DeviceOperators &op, Vec<float> &r, Vec<float> &x,
          unsigned max_iter, float tol, bool check_spd) {
    const unsigned vertex_count = op.A.nrow;
    const cudaStream_t q = cg_device_stream();
    buffer::MemoryPool &pool = buffer::get();
    auto tmp = pool.get<float>(3 * vertex_count);
    auto z = pool.get<float>(3 * vertex_count);
    auto p = pool.get<float>(3 * vertex_count);

    // Persistent device scalars: rz0, rz1, pAp, alpha, beta, err, a breakdown
    // flag (pAp/rz0 <= 0 in a divide) and an rz<=0 flag (non-SPD preconditioner).
    // err and the two flags are adjacent so one batched copy reads all three.
    auto sc = pool.get<float>(8);
    float *d_rz0 = sc.data + 0;
    float *d_rz1 = sc.data + 1;
    float *d_pAp = sc.data + 2;
    float *d_alpha = sc.data + 3;
    float *d_beta = sc.data + 4;
    float *d_err = sc.data + 5;
    int *d_break = reinterpret_cast<int *>(sc.data + 6);
    int *d_rzbad = reinterpret_cast<int *>(sc.data + 7);
    CUDA_HANDLE_ERROR(cudaMemsetAsync(d_break, 0, 2 * sizeof(int), q));

    // Fused iteration path (block-Jacobi base only; the Schwarz apply keeps
    // the legacy launch sequence). The choice is fixed for the whole solve
    // (force_bj only changes across cg() invocations), so the captured graph
    // is consistent. Partial buffers are allocated before capture.
    const bool fused = (op.H == nullptr || op.force_bj);
    const bool use_sym_spmv = (vertex_count >= CG_SYM_SPMV_MIN_ROWS);
    const unsigned grid_f = (vertex_count + CGF_BLOCK - 1) / CGF_BLOCK;
    // The 8-lane SpMV launches CG_ROW_LANES threads per row.
    const unsigned grid_mv =
        (vertex_count * CG_ROW_LANES + CGF_BLOCK - 1) / CGF_BLOCK;
    // err partials: grid_f blocks on the fused BJ path, 3n/CGF_BLOCK blocks on
    // the Schwarz path's component-wise update; allocate the larger.
    const unsigned grid_err =
        fused ? grid_f : (3 * vertex_count + CGF_BLOCK - 1) / CGF_BLOCK;
    auto fuse_partials = pool.get<float>(grid_f + grid_err + grid_mv);
    float *rz_partials = fuse_partials.data;
    float *err_partials = fuse_partials.data + grid_f;
    float *pAp_partials = fuse_partials.data + grid_f + grid_err;

    // Residual sampling: batch the host read every RESID_CHECK_STRIDE iterations
    // while the residual is far from the tolerance (sync-free bulk), then drop to
    // every iteration once within NEAR_TOL_FACTOR of tol so a sub-tol crossing is
    // never sampled past, matching cg()'s per-iteration convergence test.
    const unsigned RESID_CHECK_STRIDE = 4;
    const double NEAR_TOL_FACTOR = 8.0;

    // Experimental (PPF_CG_SKIPNORM=1): the residual L1-norm (op.norm_into) is
    // consumed by the host only at scheduled check iters, so computing it every
    // iteration is discarded work. When set, omit it from the captured iteration
    // body and compute it on demand at the check site instead. Result-identical:
    // r is not modified between the in-body norm point and end-of-iteration
    // (precond only reads r), so the on-demand norm reads the same residual and
    // the convergence decision at each check iter is unchanged.
    static bool skip_norm = [] {
        const char *e = std::getenv("PPF_CG_SKIPNORM");
        return e && e[0] == '1';
    }();

    // r holds b on entry; form the true residual r = b - A x.
    op.apply(x, tmp, q);
    kernels::add_scaled(tmp.data, r.data, -1.0f, r.size, q);

    // err0 = ||r||_1. The one host read at setup (also handles the trivial
    // already-converged case).
    op.norm_into(r, tmp, d_err, q);
    // Pinned staging: a pageable destination degrades every small D2H to a
    // ~100 us blocking staged copy (measured: the strided residual probes alone
    // were ~14.9 s of host API time on the trapped bench); a pinned destination
    // is a direct DMA (~5 us API). One persistent buffer serves the setup read
    // and every in-loop probe; each value is consumed before the next read.
    float *probe_pin = static_cast<float *>(pinned_scratch(4 * sizeof(float)));
    CUDA_HANDLE_ERROR(cudaMemcpyAsync(probe_pin, d_err, sizeof(float),
                                      cudaMemcpyDeviceToHost, q));
    CUDA_HANDLE_ERROR(cudaStreamSynchronize(q));
    const double err0 = probe_pin[0];
    if (err0 == 0.0) {
        return {true, 1u, 0.0f, false};
    }

    op.precond(r, z, q);                                          // z = M^-1 r
    kernels::inner_product_into(r.data, z.data, d_rz0, r.size, q); // rz0
    if (check_spd) {
        kernels::flag_if_nonpositive(d_rz0, d_rzbad, q);
    }
    kernels::copy(z.data, p.data, p.size, q);                    // p = z

    // The body of one CG iteration, queued on `q` with no host read. Used
    // directly for the warm-up iteration and as the recorded body of the CUDA
    // graph for the rest. Pointers and grid sizes are loop-invariant, so the
    // recorded launches replay correctly each iteration.
    auto issue_iteration = [&]() {
        if (fused) {
            // Five launches per iteration: fused SpMV+dot, the pAp fold, the
            // fused x/r/z update (alpha computed in-thread), the 2-way reduce
            // with the beta/roll/SPD scalar tail folded into its block 0, and
            // the p recurrence below.
            if (use_sym_spmv) {
                CUDA_HANDLE_ERROR(
                    cudaMemsetAsync(tmp.data, 0,
                                    3 * vertex_count * sizeof(float), q));
                cg_apply_dot_sym_kernel<<<grid_mv, CGF_BLOCK, 0, q>>>(
                    op.A, op.B, op.C, p, tmp, pAp_partials);
            } else {
                cg_apply_dot_walk_kernel<<<grid_mv, CGF_BLOCK, 0, q>>>(
                    op.A, op.B, op.C, p, tmp, pAp_partials);
            }
            CUDA_HANDLE_ERROR(cudaGetLastError());
            cg_reduce1_kernel<<<1, CGF_BLOCK, 0, q>>>(pAp_partials, grid_mv,
                                                      d_pAp);
            CUDA_HANDLE_ERROR(cudaGetLastError());
            cg_fused_update_kernel<<<grid_f, CGF_BLOCK, 0, q>>>(
                p.data, tmp.data, x.data, r.data, z.data, op.P.data, d_rz0,
                d_pAp, d_break, vertex_count, rz_partials, err_partials);
            CUDA_HANDLE_ERROR(cudaGetLastError());
            cg_reduce2_kernel<<<2, CGF_BLOCK, 0, q>>>(
                rz_partials, err_partials, grid_f, d_rz1, d_err, d_beta,
                d_rz0, d_break, d_rzbad, check_spd ? 1 : 0);
            CUDA_HANDLE_ERROR(cudaGetLastError());
        } else {
            // Schwarz path: same fused SpMV+dot and x/r update as the
            // block-Jacobi path (the SpMV is preconditioner-agnostic; alpha is
            // computed in-thread; the residual norm is a free by-product), with
            // schwarz::apply supplying z and an inner product supplying r.z.
            if (use_sym_spmv) {
                CUDA_HANDLE_ERROR(
                    cudaMemsetAsync(tmp.data, 0,
                                    3 * vertex_count * sizeof(float), q));
                cg_apply_dot_sym_kernel<<<grid_mv, CGF_BLOCK, 0, q>>>(
                    op.A, op.B, op.C, p, tmp, pAp_partials);
            } else {
                cg_apply_dot_walk_kernel<<<grid_mv, CGF_BLOCK, 0, q>>>(
                    op.A, op.B, op.C, p, tmp, pAp_partials);
            }
            CUDA_HANDLE_ERROR(cudaGetLastError());
            cg_reduce1_kernel<<<1, CGF_BLOCK, 0, q>>>(pAp_partials, grid_mv,
                                                      d_pAp);
            CUDA_HANDLE_ERROR(cudaGetLastError());
            const unsigned grid_3n = (3 * vertex_count + CGF_BLOCK - 1) /
                                     CGF_BLOCK;
            cg_update_xr_kernel<<<grid_3n, CGF_BLOCK, 0, q>>>(
                p.data, tmp.data, x.data, r.data, d_rz0, d_pAp, d_break,
                3 * vertex_count, err_partials);
            CUDA_HANDLE_ERROR(cudaGetLastError());
            cg_reduce1_kernel<<<1, CGF_BLOCK, 0, q>>>(err_partials, grid_3n,
                                                      d_err);
            CUDA_HANDLE_ERROR(cudaGetLastError());
            op.precond(r, z, q);                                 // z = M^-1 r
            kernels::inner_product_into(r.data, z.data, d_rz1, r.size, q);
            if (check_spd) {
                kernels::flag_if_nonpositive(d_rz1, d_rzbad, q);
            }
            kernels::scalar_div(d_beta, d_rz1, d_rz0, d_break, q); // beta
            kernels::scalar_assign(d_rz0, d_rz1, q);             // rz0 <- rz1
        }
        kernels::combine_indirect(z.data, p.data, p.data, 1.0f, d_beta, p.size,
                                  q);                            // p = z + beta p
    };

    // CUDA graph of one iteration, captured lazily after the warm-up iteration so
    // capture records no cudaMalloc (the reductions' pool scratch is already
    // allocated) and the stream is idle (the iter-1 residual read synced it).
    // Re-captured per solve; the guard frees it on every return path.
    struct GraphGuard {
        cudaGraph_t graph{nullptr};
        cudaGraphExec_t exec{nullptr};
        ~GraphGuard() {
            if (exec) {
                cudaGraphExecDestroy(exec);
            }
            if (graph) {
                cudaGraphDestroy(graph);
            }
        }
    } gg;
    // Attempt CUDA-graph capture of the iteration; latched off on any failure so
    // the rest of the solve replays with direct launches.
    bool try_graph = true;
    bool graph_ready = false;

    unsigned iter = 1;
    unsigned stride = RESID_CHECK_STRIDE;
    unsigned next_check = 1;
    while (true) {
        // Record the iteration once (after one warm-up iteration) and replay it.
        // cudaStreamBeginCapture records without executing, so this step does not
        // advance the iterate.
        if (try_graph && !graph_ready && iter >= 2) {
            cudaError_t cap =
                cudaStreamBeginCapture(q, cudaStreamCaptureModeThreadLocal);
            if (cap == cudaSuccess) {
                issue_iteration();
                cap = cudaStreamEndCapture(q, &gg.graph);
            }
            if (cap == cudaSuccess) {
                cap = cudaGraphInstantiate(&gg.exec, gg.graph, 0ull);
            }
            if (cap == cudaSuccess) {
                graph_ready = true;
            } else {
                // Capture/instantiate failed: clear the sticky error, abandon any
                // in-progress capture cleanly, and fall back to direct launches.
                (void)cudaGetLastError();
                cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
                if (cudaStreamIsCapturing(q, &st) == cudaSuccess &&
                    st != cudaStreamCaptureStatusNone) {
                    cudaGraph_t aborted = nullptr;
                    cudaStreamEndCapture(q, &aborted);
                    if (aborted) {
                        cudaGraphDestroy(aborted);
                    }
                }
                if (gg.graph) {
                    cudaGraphDestroy(gg.graph);
                    gg.graph = nullptr;
                }
                gg.exec = nullptr; // instantiate sets *exec=NULL on failure; be explicit
                (void)cudaGetLastError();
                try_graph = false;
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    fprintf(stderr, "[cg] CUDA graph capture unavailable; "
                                    "replaying the inner loop with direct launches\n");
                }
            }
        }

        if (graph_ready) {
            CUDA_HANDLE_ERROR(cudaGraphLaunch(gg.exec, q));
        } else {
            issue_iteration();
        }

        // The iteration cap is tested every iteration (a free host integer
        // compare) so the loop never overshoots max_iter; the residual is read
        // only at a scheduled check or at the cap.
        const bool at_cap = (iter >= max_iter);
        if (iter == next_check || at_cap) {
            if (false) { // err is a free by-product on both fused paths
                // Compute the residual L1-norm on demand (it was omitted from the
                // iteration body); r is the post-iteration residual, so this is
                // the same value the in-body norm would have produced.
                op.norm_into(r, tmp, d_err, q);
            }
            // Read err + the breakdown and rz<=0 flags once; d_err/d_break/
            // d_rzbad (sc.data+5,+6,+7) are adjacent, so one 3-float copy covers
            // all three.
            // Re-fetch: the scratch pointer is stable today, but re-fetching
            // makes this robust to any future scratch growth between checks.
            probe_pin = static_cast<float *>(pinned_scratch(4 * sizeof(float)));
            CUDA_HANDLE_ERROR(cudaMemcpyAsync(probe_pin, d_err,
                                              3 * sizeof(float),
                                              cudaMemcpyDeviceToHost, q));
            CUDA_HANDLE_ERROR(cudaStreamSynchronize(q));
            const double reresid = (double)probe_pin[0] / err0;
            const int broke = *reinterpret_cast<const int *>(&probe_pin[1]);
            const int rzbad = *reinterpret_cast<const int *>(&probe_pin[2]);
            // Convergence is tested first, matching the host cg() ordering: a
            // finite residual below tol wins even if a flag latched earlier (a
            // NaN/Inf residual fails the < tol test and falls through).
            if (reresid < (double)tol) {
                return {true, iter, (float)reresid, false};
            }
            // Non-SPD preconditioner residual (Schwarz only): signal the caller
            // to fall back to the block-Jacobi loop (4th element true). The flag
            // latches but is read only at a scheduled check, so up to a few
            // corrupted iterations can run first. This stays harmless only
            // because the Gram-SPD apply rounds rz <= 0 just when r is already
            // tiny, by which point NEAR_TOL_FACTOR has forced stride == 1, so the
            // corrupted window is ~1 iteration. Convergence is tested above this:
            // a residual already below tol is accepted (matching cg_hostsync,
            // which also tests convergence before the rz<=0 restart). Do not
            // widen RESID_CHECK_STRIDE / NEAR_TOL_FACTOR without revisiting this.
            if (rzbad) {
                return {false, iter, (float)reresid, true};
            }
            if (broke || !std::isfinite(reresid)) {
                return {false, iter, (float)reresid, false};
            }
            if (at_cap) {
                return {false, iter, (float)reresid, false};
            }
            // Within reach of the tolerance: check every iteration from here so
            // the crossing is detected at the same iter as the host loop.
            if (reresid < NEAR_TOL_FACTOR * (double)tol) {
                stride = 1;
            }
            next_check = iter + stride;
        }
        iter++;
    }
}

// Host-synchronizing PCG. Returns {success, iters, relative residual, fell_back}.
// fell_back is true if the Schwarz preconditioner produced a non-SPD residual
// (rz <= 0) and the loop latched the SPD-safe block-Jacobi fallback for the rest
// of the solve. This is the reference loop, and the rare-fallback path the
// device-resident Schwarz solve restores into when it detects rz <= 0.
static std::tuple<bool, unsigned, float, bool>
cg_hostsync(const DeviceOperators &op, Vec<float> &r, Vec<float> &x,
            unsigned max_iter, float tol) {
    unsigned vertex_count = op.A.nrow;
    buffer::MemoryPool &pool = buffer::get();
    auto tmp = pool.get<float>(3 * vertex_count);
    auto z = pool.get<float>(3 * vertex_count);
    auto p = pool.get<float>(3 * vertex_count);
    bool fell_back = false;

    op.apply(x, tmp);
    kernels::add_scaled(tmp.data, r.data, -1.0f, r.size);
    double err0 = op.norm(r, tmp);
    op.precond(r, z);

    unsigned iter = 1;
    double rz0 = kernels::inner_product(r.data, z.data, r.size);
    // SPD safety net: a fixed SPD preconditioner gives rz > 0. If the Schwarz
    // path yields rz <= 0 (or non-finite), latch block-Jacobi and recompute z.
    if ((rz0 <= 0.0 || !std::isfinite(rz0)) && op.H && !op.force_bj) {
        op.force_bj = true;
        fell_back = true;
        op.precond(r, z);
        rz0 = kernels::inner_product(r.data, z.data, r.size);
    }
    kernels::copy(z.data, p.data, p.size);

    if (!err0) {
        return {true, iter, 0.0f, fell_back};
    } else {
        while (true) {
            op.apply(p, tmp);
            double alpha = rz0 / (double)kernels::inner_product(p.data, tmp.data, p.size);
            kernels::add_scaled(p.data, x.data, (float)alpha, x.size);
            kernels::add_scaled(tmp.data, r.data, (float)-alpha, r.size);
            double err = op.norm(r, tmp);
            double reresid = err / err0;
            if (reresid < tol) {
                return {true, iter, reresid, fell_back};
            } else if (iter >= max_iter || std::isnan(reresid)) {
                return {false, iter, reresid, fell_back};
            }
            op.precond(r, z);
            double rz1 = kernels::inner_product(r.data, z.data, r.size);
            bool restart = false;
            if ((rz1 <= 0.0 || !std::isfinite(rz1)) && op.H && !op.force_bj) {
                op.force_bj = true;
                fell_back = true;
                restart = true;
                op.precond(r, z);
                rz1 = kernels::inner_product(r.data, z.data, r.size);
            }
            if (restart) {
                // Restart the search direction with the new (block-Jacobi)
                // preconditioner: p = z, drop the stale beta-combine.
                kernels::copy(z.data, p.data, p.size);
            } else {
                double beta = rz1 / rz0;
                kernels::combine(z.data, p.data, p.data, 1.0f, (float)beta, p.size);
            }
            rz0 = rz1;
            iter++;
        }
    }
    // PooledVec buffers auto-release when exiting function
}

// PCG entry. Both preconditioners run the device-resident loop; Schwarz also
// passes check_spd so a non-SPD residual is detected on the device, in which case
// the inputs are restored and the proven host-sync block-Jacobi fallback runs
// (rare; the Gram apply keeps Schwarz SPD). PDRD scenes do not reach here.
std::tuple<bool, unsigned, float, bool> cg(const DeviceOperators &op,
                                           Vec<float> &r, Vec<float> &x,
                                           unsigned max_iter, float tol) {
    if (op.H == nullptr) {
        // Block-Jacobi is SPD, so no rz<=0 fallback is possible or needed.
        return cg_device(op, r, x, max_iter, tol, /*check_spd=*/false);
    }

    // Device-resident Schwarz. Keep b and the initial x for the rare rz<=0
    // re-solve (r holds b on entry; cg_device overwrites both in place). The
    // copies run on the PCG stream so they are ordered against cg_device's work
    // on the same stream without relying on the default-stream model.
    const cudaStream_t q = cg_device_stream();
    buffer::MemoryPool &pool = buffer::get();
    auto bcopy = pool.get<float>(r.size);
    auto x0copy = pool.get<float>(x.size);
    kernels::copy(r.data, bcopy.data, r.size, q);
    kernels::copy(x.data, x0copy.data, x.size, q);

    bool ok;
    unsigned it;
    float rs;
    bool rzbad;
    std::tie(ok, it, rs, rzbad) =
        cg_device(op, r, x, max_iter, tol, /*check_spd=*/true);
    if (!rzbad) {
        return {ok, it, rs, false};
    }

    // Non-SPD Schwarz residual (a rare near-convergence float edge): restore b
    // and the original guess and re-solve from scratch with the host loop's
    // block-Jacobi latch. Unlike cg_hostsync's in-place restart this discards the
    // (nearly converged) Schwarz progress, but it also wipes any NaN the
    // corrupted window left in x, and the device solve already paid for itself.
    kernels::copy(bcopy.data, r.data, r.size, q);
    kernels::copy(x0copy.data, x.data, x.size, q);
    op.force_bj = true;
    bool ok2;
    unsigned it2;
    float rs2;
    bool fb2;
    std::tie(ok2, it2, rs2, fb2) = cg_hostsync(op, r, x, max_iter, tol);
    (void)fb2;
    return {ok2, it2, rs2, true};
}

// Rigid reduced-coordinate PCG: solve R u = P^T b with R = P^T M P, then x = P u,
// where P is the rigid Jacobian (translation + rotation per body, identity on
// cloth). Reducing the per-vertex inertia diagonal of M through P yields the
// exact rigid 6x6 inertia; reducing contact yields the rigid contact coupling.
// Preconditioner: per-body 6x6 block (analytic rigid inertia + contact diagonal)
// + 3x3 block-Jacobi for cloth. Returns {success, iters, resid}.
std::tuple<bool, unsigned, float>
cg_rigid(const DynCSRMat &A, const FixedCSRMat &B, const Vec<Mat3x3f> &C,
         PDRD::RigidMap &rm, PDRD::RigidPrecond &P, const Vec<Mat3x3f> &inv_diag,
         Vec<float> bvec, Vec<float> x, unsigned max_iter, float tol,
         Vec<float> dtheta_out) {
    buffer::MemoryPool &pool = buffer::get();
    const unsigned dim = rm.dim;
    const unsigned nrow = rm.nrow;
    auto fb = pool.get<float>(dim);
    auto xrb = pool.get<float>(dim);
    auto rb = pool.get<float>(dim);
    auto zb = pool.get<float>(dim);
    auto pb = pool.get<float>(dim);
    auto rpb = pool.get<float>(dim);
    auto tb = pool.get<float>(dim);
    auto xvb = pool.get<float>(3 * nrow);
    auto mxvb = pool.get<float>(3 * nrow);
    Vec<float> f = fb.as_vec(), xr = xrb.as_vec(), r = rb.as_vec(),
               z = zb.as_vec(), p = pb.as_vec(), Rp = rpb.as_vec(),
               tmp = tb.as_vec(), xv = xvb.as_vec(), mxv = mxvb.as_vec();

    auto Rapply = [&](Vec<float> in, Vec<float> out) {
        PDRD::launch_prolong_rigid(rm, in, xv);
        solver::apply(A, B, C, 0.0f, xv, mxv);
        PDRD::launch_restrict_rigid(rm, mxv, out);
        PDRD::launch_project_bodies(rm, out);
    };
    auto l1norm = [&](Vec<float> w) {
        DISPATCH_START(w.size)
        [w, tmp] __device__(unsigned i) mutable { tmp.data[i] = fabsf(w.data[i]); }
        DISPATCH_END;
        return (double)kernels::sum_array(tmp.data, dim);
    };

    PDRD::launch_restrict_rigid(rm, bvec, f); // f = P^T b
    PDRD::launch_project_bodies(rm, f);       // restrict the rhs to the joint subspace
    CUDA_HANDLE_ERROR(cudaMemset(xr.data, 0, dim * sizeof(float)));
    kernels::copy(f.data, r.data, dim); // r = f - R*0
    double err0 = l1norm(r);
    if (err0 == 0.0) {
        CUDA_HANDLE_ERROR(cudaMemset(x.data, 0, 3 * nrow * sizeof(float)));
        return {true, 1u, 0.0f};
    }
    PDRD::apply_rigid_precond(P, rm, inv_diag, r, z);
    PDRD::launch_project_bodies(rm, z);
    kernels::copy(z.data, p.data, dim);
    double rz0 = kernels::inner_product(r.data, z.data, dim);
    unsigned iter = 1;
    float reresid = 1.0f;
    bool success = false;
    while (true) {
        Rapply(p, Rp);
        double pRp = kernels::inner_product(p.data, Rp.data, dim);
        double alpha = rz0 / (pRp != 0.0 ? pRp : 1.0);
        kernels::add_scaled(p.data, xr.data, (float)alpha, dim);
        kernels::add_scaled(Rp.data, r.data, (float)-alpha, dim);
        double err = l1norm(r);
        reresid = (float)(err / err0);
        if (reresid < tol) {
            success = true;
            break;
        }
        if (iter >= max_iter || std::isnan((double)reresid)) {
            success = false;
            break;
        }
        PDRD::apply_rigid_precond(P, rm, inv_diag, r, z);
        PDRD::launch_project_bodies(rm, z);
        double rz1 = kernels::inner_product(r.data, z.data, dim);
        double beta = rz1 / rz0;
        kernels::combine(z.data, p.data, p.data, 1.0f, (float)beta, dim);
        rz0 = rz1;
        iter++;
    }
    PDRD::launch_project_bodies(rm, xr); // defensive: keep the solution in-subspace
    // Export the per-body rotation DOFs so the caller can integrate the applied
    // rotation onto the persistent R_run for the anchored rigidify.
    if (dtheta_out.data) PDRD::launch_extract_body_dtheta(rm, xr, dtheta_out);
    PDRD::launch_prolong_rigid(rm, xr, x); // x = P u
    return {success, iter, reresid};
}

// Rigid-operator self-test (S2): on the live scene, build the rigid reduced map
// and per-body 6x6 preconditioner, then (a) check R = P^T M P is symmetric and
// positive definite via random probes and (b) run the preconditioned rigid PCG
// on a synthetic RHS and report the residual reduction. Env-gated; no live-
// solver effect.
void rigid_operator_selftest(const DynCSRMat &A, const FixedCSRMat &B,
                                    const Vec<Mat3x3f> &C,
                                    const Vec<Mat3x3f> &inv_diag,
                                    const DataSet &data,
                                    const Vec<Vec3f> &positions, float dt,
                                    unsigned nrow) {
    unsigned nb = data.prop.pdrd_body.size;
    if (nb == 0 || nrow == 0) return;
    buffer::MemoryPool &pool = buffer::get();

    auto stb = pool.get<PDRD::RigidState>(nb);
    Vec<PDRD::RigidState> state = stb.as_vec();
    PDRD::launch_fit_rigid(data, positions, state);
    PDRD::RigidMap rm;
    PDRD::build_rigid_map(rm, data, state, nrow);
    PDRD::RigidPrecond P;
    PDRD::build_rigid_precond(P, data, state, A, B, dt);

    const unsigned dim = rm.dim;
    auto ub = pool.get<float>(dim);
    auto vb = pool.get<float>(dim);
    auto rub = pool.get<float>(dim);
    auto rvb = pool.get<float>(dim);
    auto xb = pool.get<float>(3 * nrow);
    auto mxb = pool.get<float>(3 * nrow);
    Vec<float> u = ub.as_vec(), v = vb.as_vec(), Ru = rub.as_vec(),
               Rv = rvb.as_vec(), xv = xb.as_vec(), mxv = mxb.as_vec();
    auto fill = [](Vec<float> w, unsigned seed) {
        DISPATCH_START(w.size)
        [w, seed] __device__(unsigned i) mutable {
            unsigned h = (i + seed) * 2654435761u;
            w.data[i] = ((float)(h % 2000u) / 1000.0f) - 1.0f;
        } DISPATCH_END;
    };
    auto Rapply = [&](Vec<float> in, Vec<float> out) {
        PDRD::launch_prolong_rigid(rm, in, xv);
        solver::apply(A, B, C, 0.0f, xv, mxv);
        PDRD::launch_restrict_rigid(rm, mxv, out);
        PDRD::launch_project_bodies(rm, out);
    };
    fill(u, 1u);
    fill(v, 2u);
    // Restrict the probes to the joint subspace so the symmetry / PD checks
    // measure the actual projected operator Pi R Pi the live solver applies.
    PDRD::launch_project_bodies(rm, u);
    PDRD::launch_project_bodies(rm, v);
    Rapply(u, Ru);
    Rapply(v, Rv);
    double uRu = kernels::inner_product(u.data, Ru.data, dim);
    double uRv = kernels::inner_product(u.data, Rv.data, dim);
    double vRu = kernels::inner_product(v.data, Ru.data, dim);
    double sym =
        std::fabs(uRv - vRu) / (std::fabs(uRv) + std::fabs(vRu) + 1e-30);

    auto rhsb = pool.get<float>(3 * nrow);
    auto solb = pool.get<float>(3 * nrow);
    Vec<float> rhs = rhsb.as_vec(), sol = solb.as_vec();
    fill(rhs, 7u);
    bool ok;
    unsigned it;
    float rr;
    std::tie(ok, it, rr) =
        cg_rigid(A, B, C, rm, P, inv_diag, rhs, sol, 2000u, 1e-5f, Vec<float>{});
    fprintf(stderr,
            "[pdrd rigid operator selftest] dim=%u (cloth=%u bodies=%u) "
            "PD<u,Ru>=%.3e sym_rel=%.3e PCG: ok=%d iters=%u resid=%.3e\n",
            dim, rm.n_cloth, nb, uRu, sym, (int)ok, it, rr);
    P.free_all();
    rm.free_all();
}


bool solve(const DynCSRMat &A, const FixedCSRMat &B, const Vec<Mat3x3f> &C,
           Vec<float> b, float tol, unsigned max_iter, Vec<float> x,
           const Vec<Vec3f> &positions, const ParamSet &prm, unsigned &iter,
           float &resid, unsigned &schwarz_fallback, const DataSet &data,
           float dt, Vec<float> pdrd_dtheta_out) {

    unsigned vertex_count = A.nrow;
    buffer::MemoryPool &pool = buffer::get();
    auto inv_diag = pool.get<Mat3x3f>(vertex_count);

    Vec<Mat3x3f> inv_diag_vec = inv_diag.as_vec();
    DISPATCH_START(A.nrow)
    [A, B, C, inv_diag_vec] __device__(unsigned i) mutable {
        inv_diag_vec[i] = invert(A(i, i) + B(i, i) + C[i]);
    } DISPATCH_END;

    unsigned n_pdrd_bodies = data.prop.pdrd_body.size;

    // Exact-rigid kinematics self-test (S1) + reduced-operator self-test (S2):
    // polar fit, reconstruct, rigid Jacobian, reduced rigid mass block, then the
    // rigid reduced operator R = P^T M P (SPD/symmetric) and a preconditioned
    // rigid PCG solve on the live scene's bodies. Env-gated, no live effect.
    if (n_pdrd_bodies && std::getenv("PPF_PDRD_RIGID_SELFTEST")) {
        static bool done = false;
        if (!done) {
            done = true;
            PDRD::selftest_rigid(data, positions);
            rigid_operator_selftest(A, B, C, inv_diag_vec, data, positions, dt,
                                    vertex_count);
        }
    }

    DeviceOperators ops(A, B, C, inv_diag_vec);

    // Reduced-coordinate rigid PDRD mode. PDRD bodies are solved in reduced 6-DOF
    // RIGID coordinates u_b = (dx_b, dtheta_b). The body carries NO penalty
    // energy; the assembled matrix M holds only per-vertex inertia (diagonal C)
    // and contact (A), so R = P^T M P is the exact rigid inertia + contact-
    // projected system. Rigidity itself is enforced by the per-iteration
    // reconstruct in main.cu; the search direction here is restricted to rigid
    // motions by P. Per-body 6x6 block preconditioner (analytic rigid inertia +
    // contact diagonal). Gated on the scene actually having PDRD bodies.
    const bool reduced_rigid = n_pdrd_bodies > 0;
    if (reduced_rigid) {
        auto stb = pool.get<PDRD::RigidState>(n_pdrd_bodies);
        Vec<PDRD::RigidState> state = stb.as_vec();
        PDRD::launch_fit_rigid(data, positions, state);
        PDRD::RigidMap rm;
        PDRD::build_rigid_map(rm, data, state, vertex_count);
        static PDRD::RigidPrecond s_rprec;
        PDRD::build_rigid_precond(s_rprec, data, state, A, B, dt);
        bool ok;
        std::tie(ok, iter, resid) =
            cg_rigid(A, B, C, rm, s_rprec, inv_diag_vec, b, x, max_iter, tol,
                     pdrd_dtheta_out);
        rm.free_all();
        schwarz_fallback = 0u;
        return ok;
    }

    // Build the aggregate-Schwarz hierarchy just before PCG (the linear system
    // M = A_dyn + B_fixed + C_diag is fully known here). Selected by
    // ParamSet::precond (default Schwarz; "block-jacobi" picks the simpler base).
    // schwarz::build owns a persistent cache; the hierarchy holds views into it
    // and stays valid for the whole solve. (PDRD scenes return above; this is the
    // cloth / general PCG path.)
    schwarz::SchwarzHierarchy hierarchy;
    const bool use_schwarz = prm.precond == PrecondMode::Schwarz;
    if (use_schwarz) {
        schwarz::build(hierarchy, A, B, C, positions, prm.schwarz_levels);
        ops.set_schwarz(&hierarchy);
    }

    bool success;
    bool fell_back;
    std::tie(success, iter, resid, fell_back) = cg(ops, b, x, max_iter, tol);
    schwarz_fallback = fell_back ? 1u : 0u;

    // PooledVec auto-releases when exiting function
    return success;
}

} // namespace solver
