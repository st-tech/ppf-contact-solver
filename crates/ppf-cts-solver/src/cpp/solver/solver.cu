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
#include <vector>

namespace solver {

// A correct assembly makes the Newton system matrix and the preconditioner both
// SPD, so p^T A p > 0 and r^T M^-1 r > 0 for every PCG search direction. If
// either turns non-positive the assembly is wrong: a per-element Hessian missing
// its SPD projection, a sign error, a stale/garbage block, or an off-diagonal
// block silently dropped because its (i, j) pair was never registered in the
// fixed CSR sparsity pattern (which turns even a PSD element indefinite). Rather
// than limp on with a broken solve -- or silently mask a non-SPD preconditioner
// behind the block-Jacobi fallback -- abort at the detection site so the wrong
// matrix is spotted immediately.
//
// That test only means something when the sign it reads is real. p^T A p is a
// sum of ~2*nnz SIGNED contributions in fp32, and on a contact-saturated scene
// those contributions total ~1e11 while the curvature itself is ~1e3, so the
// result is decided entirely by round-off: the same sum reordered moves by 20x,
// and its sign with it. Aborting a simulation on that sign reports a defect that
// is not there. Each SpMV therefore also reduces sum|contribution|, and eps times
// that sum bounds the round-off of the dot; the sign is only acted on once the
// curvature clears the bound. Below it the direction carries no information, so
// the solve truncates and returns its iterate (standard practice on a
// non-positive-curvature direction) and the Newton loop line-searches it under
// CCD. Beyond it -- an actual assembly error puts the Rayleigh quotient at
// order 1, six orders above this bound -- the abort stands unchanged.
// Cause codes latched at the breakdown site, so the report names the event that
// actually happened rather than assuming one of them.
enum {
    BREAK_CAUSE_NONE = 0,
    BREAK_CAUSE_PAP = 1,   // p^T A p negative BEYOND the round-off bound
    BREAK_CAUSE_RZ = 2,    // r^T M^-1 r non-positive or NaN (preconditioner)
    BREAK_CAUSE_NOISE = 3, // |p^T A p| within the round-off bound: the
                           // curvature is not resolvable in fp32, so the
                           // direction carries no usable information. Not an
                           // assembly error, so this truncates the solve
                           // instead of aborting the run.
};

// Round-off slack for the sign test on p^T A p. The quadratic form is a sum of
// ~2*nnz signed contributions reduced through a per-thread serial accumulation,
// a 256-wide shared tree and a strided fold, so the summation error is bounded
// by roughly (tree depth) * eps * absdot, with each contribution carrying its
// own eps on top. 256 covers both with margin and is still ~3e-5 in relative
// terms, orders below the order-1 Rayleigh quotient that an actual assembly
// error (a missing SPD projection, a sign flip, a dropped block) produces. It
// is a round-off bound, NOT a tolerance on the guard: anything more negative
// than this still aborts.
static constexpr float CG_CURVATURE_NOISE_SLACK = 256.0f;
static constexpr float CG_FLOAT_EPS = 1.19209290e-7f;

// The verdict on a curvature reading, shared by every PCG path.
enum CurvatureVerdict {
    CURVATURE_OK = 0,       // resolvable positive curvature: alpha = rz / pAp
    CURVATURE_TRUNCATE = 1, // within its own round-off: no resolvable sign, so
                            // the direction carries no information and the
                            // solve stops here (Steihaug-style) without abort
    CURVATURE_FATAL = 2,    // negative BEYOND the bound, or NaN: a real
                            // assembly regression, not fp32 cancellation
};

// Round-off floor of the reduction that produced a p^T A p, given the scale of
// that reduction. The two device paths pass the SpMV's sum|contribution|; the
// host loop, whose SpMV produces no such sum, passes the Cauchy-Schwarz
// surrogate |p|_2 |Ap|_2 >= sum|p_i (Ap)_i|.
//
// FLOAT, NOT TEMPLATED, and deliberately so: these run on the GPU, where float64
// is banned outright, and a `__host__ __device__` template instantiated with
// double by the host caller would emit exactly the double device instantiation
// that rule forbids. Float is also the right precision on the merits -- every
// input is reduced from an fp32 operator, and the verdict is a sign test against
// a 256*eps slack, not a quantity read to the last ulp. The host loop keeps
// double for its own arithmetic and narrows only at this call.
__host__ __device__ inline float cg_curvature_bound(float reduction_scale) {
    return CG_CURVATURE_NOISE_SLACK * CG_FLOAT_EPS * reduction_scale;
}

// Only the SCALE of the bound differs between the paths; the decision made from
// it must not, so it lives here rather than being spelled out at each call site.
// The device paths run this on every iteration of every solve, which is what
// gives the rarely-reached host loop (entered only on a Schwarz rz<=0 fallback)
// its coverage.
__host__ __device__ inline CurvatureVerdict cg_curvature_verdict(float pAp,
                                                                 float bound) {
    if (pAp > bound) {
        return CURVATURE_OK;
    }
    // A NaN fails every comparison, so it falls through the test above and must
    // be named here rather than being reported as a negative curvature or
    // silently truncating. Spelled (pAp != pAp) so one definition serves both
    // the float device kernels and the double host loop, with no dependence on
    // a host-only <cmath> isnan overload.
    if (pAp < -bound || pAp != pAp) {
        return CURVATURE_FATAL;
    }
    return CURVATURE_TRUNCATE;
}

[[noreturn]] static void
fatal_indefinite_system_matrix(unsigned break_iter, unsigned check_iter,
                               double reresid, double pAp_f32, double pAp_exact,
                               double pp, double sym_zAp, double sym_pAz,
                               double noise_bound) {
    const char *kind = std::isnan(pAp_f32) ? "not-a-number" : "negative";
    const double rayleigh = pp > 0.0 ? pAp_exact / pp : 0.0;
    fprintf(stderr,
            "PPF FATAL: PCG breakdown -- p^T A p is %s at iter %u (detected at "
            "check iter %u, reresid %.3e).\n"
            "  p^T A p  (fp32 device reduction, the value that tripped it) = "
            "%.6e\n"
            "  p^T A p  (recomputed in double from the same iterate)        = "
            "%.6e\n"
            "  |p|^2                                                       = "
            "%.6e\n"
            "  Rayleigh quotient p^T A p / |p|^2 (double)                  = "
            "%.6e\n"
            "  round-off bound on the fp32 sum                             = "
            "%.6e  (exceeded, hence a real defect and not cancellation)\n"
            "  symmetry probe: z^T(A p) = %.6e vs p^T(A z) = %.6e (relative "
            "gap %.3e)\n"
            "A double-precision value that is POSITIVE means the assembled "
            "matrix is SPD and the fp32 reduction cancelled; a NEGATIVE one of "
            "order-1-and-up magnitude means a genuine indefinite term (a "
            "per-element Hessian missing its SPD projection, a sign/assembly "
            "error, or a dropped off-diagonal block). A symmetry gap well above "
            "float epsilon means the applied operator is not symmetric, in "
            "which case p^T A p is not a curvature at all.\n",
            kind, break_iter, check_iter, reresid, pAp_f32, pAp_exact, pp,
            rayleigh, noise_bound, sym_zAp, sym_pAz,
            (std::fabs(sym_zAp) + std::fabs(sym_pAz)) > 0.0
                ? std::fabs(sym_zAp - sym_pAz) * 2.0 /
                      (std::fabs(sym_zAp) + std::fabs(sym_pAz))
                : 0.0);
    fflush(stderr);
    std::abort();
}

[[noreturn]] static void fatal_nonpositive_rz(unsigned break_iter,
                                              unsigned check_iter,
                                              double reresid, double rz) {
    fprintf(stderr,
            "PPF FATAL: PCG breakdown -- r^T M^-1 r is %s at iter %u (detected "
            "at check iter %u, reresid %.3e), value = %.6e. The preconditioner "
            "is not SPD. Every block-Jacobi diagonal block is inverted through "
            "a floored symmetric eigendecomposition, so each per-vertex term "
            "r_i^T M_i^-1 r_i is positive by construction and their sum cannot "
            "be negative in exact arithmetic: a non-positive value here means "
            "a block is NaN/Inf or the residual itself is not finite.\n",
            std::isnan(rz) ? "not-a-number" : "non-positive", break_iter,
            check_iter, reresid, rz);
    fflush(stderr);
    std::abort();
}

[[noreturn]] static void fatal_indefinite_hostsync(unsigned iter, double pAp,
                                                   double bound) {
    fprintf(stderr,
            "PPF FATAL: PCG breakdown -- p^T A p is %s at iter %u on the host "
            "fallback loop: value %.6e against a round-off bound of %.6e. "
            "Beyond that bound the sign is real, so the assembled Newton "
            "Hessian is not SPD (a per-element Hessian missing its SPD "
            "projection, a sign/assembly error, or a dropped off-diagonal "
            "block). This loop keeps no captured iterate; re-run so the "
            "device path takes the solve to get the double-precision "
            "recomputation and the symmetry probe.\n",
            std::isnan(pAp) ? "not-a-number" : "negative", iter, pAp, bound);
    fflush(stderr);
    std::abort();
}

[[noreturn]] static void fatal_nonspd_preconditioner(unsigned iter,
                                                     double reresid) {
    fprintf(stderr,
            "PPF FATAL: non-SPD preconditioner -- r^T M^-1 r <= 0 in the PCG "
            "solve at iter %u (reresid %.3e). The preconditioner (Schwarz / "
            "block-Jacobi) is not SPD.\n",
            iter, reresid);
    fflush(stderr);
    std::abort();
}

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

// Invert a symmetric 3x3 block-Jacobi diagonal block. The block is SPD by
// assembly (a strict mass/dt^2 inertia floor plus PSD-projected elastic /
// bending / contact / friction blocks), but it is not well conditioned: in a
// tight drape the elasticity-inclusive dynamic barrier stiffness ~mass/gap^2
// pushes the normal-direction stiffness to ~1e11 against a tangential floor
// ~1e2, i.e. a condition number ~1e9. A raw cofactor inverse then forms the
// tangential cofactors as differences of huge*floor products whose surviving
// digits fall below fp32 epsilon, so it can return a garbage or even
// sign-flipped, non-SPD block, which corrupts the PCG search direction.
// Invert through the symmetric eigendecomposition instead: symm3x3 max-abs-
// scales the block (so no cross-magnitude cancellation) and each 1/lambda_k is
// formed independently, making the result SPD and bounded at any conditioning.
// Eigenvalues are floored because symm3x3 cannot resolve one below ~eps*lambda_max;
// the floor keeps the preconditioner SPD with a bounded condition number and
// only affects preconditioning quality (never correctness) in that unresolvable
// subspace. Well-conditioned blocks are unaffected: no eigenvalue is clamped and
// the reconstruction agrees with the cofactor inverse.
__device__ static Mat3x3f invert(const Mat3x3f &m) {
    // Enforce exact fp32 symmetry (the assembled block is symmetric only up to
    // atomic-accumulation order) before the eigensolve.
    Mat3x3f sym;
    for (int a = 0; a < 3; ++a) {
        sym(a, a) = m(a, a);
        for (int b = a + 1; b < 3; ++b) {
            float s = 0.5f * (m(a, b) + m(b, a));
            sym(a, b) = s;
            sym(b, a) = s;
        }
    }
    Vec3f lambda;
    Mat3x3f Q; // columns are the (ascending) eigenvectors
    utility::solve_symm_eigen3x3(sym, lambda, Q);
    float lmax = fmaxf(lambda[0], fmaxf(lambda[1], lambda[2]));
    // The diagonal block carries a strict mass/dt^2 inertia floor (and, for a
    // massless pinned vertex, the fix-pin barrier), so it cannot be
    // non-positive or non-finite unless the assembly is broken upstream. Trap
    // here rather than return something plausible: a zero preconditioner block
    // would NOT trip the PCG SPD guards (it contributes exactly 0 to r.z and
    // pAp, never a negative), it would silently freeze this vertex's DOF at its
    // seed for every CG iteration. Asserts are live in the production build.
    assert(isfinite(lmax));
    assert(lmax > 0.0f);
    Mat3x3f minv = Mat3x3f::Zero();
    const float lambda_floor = lmax * 1.0e-6f;
    for (int k = 0; k < 3; ++k) {
        float inv_lk = 1.0f / fmaxf(lambda[k], lambda_floor);
        // Rank-1 q q^T accumulation with raw scalar arithmetic (device Eigen
        // matrix*vector / outer products are silently wrong; scalar fills safe).
        float q0 = Q(0, k), q1 = Q(1, k), q2 = Q(2, k);
        minv(0, 0) += inv_lk * q0 * q0;
        minv(0, 1) += inv_lk * q0 * q1;
        minv(0, 2) += inv_lk * q0 * q2;
        minv(1, 1) += inv_lk * q1 * q1;
        minv(1, 2) += inv_lk * q1 * q2;
        minv(2, 2) += inv_lk * q2 * q2;
    }
    minv(1, 0) = minv(0, 1);
    minv(2, 0) = minv(0, 2);
    minv(2, 1) = minv(1, 2);
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
                                       const float *d_absdot, int *breakdown,
                                       unsigned nrow, float *rz_partials,
                                       float *err_partials, int *break_cause,
                                       float *break_val, const int *iter_ctr,
                                       int *break_iter, float *p_break,
                                       float *Ap_break) {
    __shared__ float s_rz[CGF_BLOCK];
    __shared__ float s_err[CGF_BLOCK];
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float rz = 0.0f, err = 0.0f;
    if (i < nrow) {
        // alpha = rz0 / pAp computed per thread from the device scalars
        // (read-only, all threads agree), replicating scalar_div's breakdown
        // semantics; folds the former scalar_div launch into this kernel.
        const float pAp = *d_pAp;
        // Curvature below its own round-off bound has no resolvable sign in
        // fp32, so alpha = rz/pAp would be meaningless (and unboundedly large)
        // whichever side of zero the sum happened to land on. Require the
        // curvature to clear that bound before dividing by it.
        const float noise = cg_curvature_bound(*d_absdot);
        const CurvatureVerdict verdict = cg_curvature_verdict(pAp, noise);
        float alpha;
        if (verdict == CURVATURE_OK) {
            alpha = (*d_rz0) / pAp;
        } else {
            // Breakdown. cg_curvature_verdict has already separated a NaN from a
            // genuinely negative curvature -- the report must not call a NaN
            // "negative". Latch the
            // offending iterate HERE, in the kernel that observes it: the host
            // check runs up to RESID_CHECK_STRIDE iterations later, by which
            // point d_pAp and p have both been overwritten, and reporting those
            // stale values describes a different iterate entirely. Capturing p
            // and Ap lets the abort site recompute the quadratic form in double
            // and tell a genuine indefinite term from an fp32 cancellation.
            alpha = 0.0f;
            if (i == 0) {
                *breakdown = 1;
                // Only a curvature that is negative BEYOND the round-off bound
                // is evidence of a non-SPD assembly; within the bound it is an
                // unresolvable zero and the solve simply stops here.
                *break_cause = (verdict == CURVATURE_FATAL) ? BREAK_CAUSE_PAP
                                                            : BREAK_CAUSE_NOISE;
                *break_val = pAp;
                *break_iter = *iter_ctr;
            }
            for (unsigned k = 0; k < 3; ++k) {
                const unsigned j = 3 * i + k;
                p_break[j] = p[j];
                Ap_break[j] = Ap[j];
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
                                  int check_spd, int *break_cause,
                                  float *break_val, int *iter_ctr,
                                  int *break_iter) {
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
                // A SECOND, distinct cause for the same flag: r^T M^-1 r is
                // non-positive (or NaN), which is a preconditioner event, not
                // an indefinite system matrix. Latch the cause so the report
                // does not blame p^T A p for it.
                *d_beta = 0.0f;
                if (breakdown && *break_cause == BREAK_CAUSE_NONE) {
                    *breakdown = 1;
                    *break_cause = BREAK_CAUSE_RZ;
                    *break_val = rz0;
                    *break_iter = *iter_ctr;
                }
            }
            *d_rz0 = rz1;
            // One increment per iteration: block 0 owns the scalar tail.
            *iter_ctr += 1;
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
                                    const float *d_pAp, const float *d_absdot,
                                    int *breakdown, unsigned n,
                                    float *err_partials, int *break_cause,
                                    float *break_val, float *p_break,
                                    float *Ap_break) {
    __shared__ float s_err[CGF_BLOCK];
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    float err = 0.0f;
    if (j < n) {
        const float pAp = *d_pAp;
        const float noise = cg_curvature_bound(*d_absdot);
        const CurvatureVerdict verdict = cg_curvature_verdict(pAp, noise);
        float alpha;
        if (verdict == CURVATURE_OK) {
            alpha = (*d_rz0) / pAp;
        } else {
            // Same latch as the fused path: record the offending iterate where
            // it is observed, not at the later host check. This kernel is
            // component-wise (n = 3 * vertex_count), so the capture is a plain
            // per-index copy.
            alpha = 0.0f;
            if (j == 0) {
                *breakdown = 1;
                *break_cause = (verdict == CURVATURE_FATAL) ? BREAK_CAUSE_PAP
                                                            : BREAK_CAUSE_NOISE;
                *break_val = pAp;
            }
            p_break[j] = p[j];
            Ap_break[j] = Ap[j];
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

// Two-array fold (block 0 = the dot, block 1 = its magnitude bound), reduced in
// one launch so the pair is guaranteed to describe the same iterate.
__global__ void cg_reduce_dot_kernel(const float *dot_partials,
                                     const float *absdot_partials,
                                     unsigned count, float *d_dot,
                                     float *d_absdot) {
    __shared__ float s[CGF_BLOCK];
    const float *src = (blockIdx.x == 0) ? dot_partials : absdot_partials;
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
            *d_dot = s[0];
        } else {
            *d_absdot = s[0];
        }
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
                                    float *dot_partials,
                                    float *absdot_partials) {
    __shared__ float s_dot[CGF_BLOCK];
    __shared__ float s_absdot[CGF_BLOCK];
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned i = t / CG_ROW_LANES;
    const unsigned lane = threadIdx.x % CG_ROW_LANES;
    float dot = 0.0f;
    // Sum of the magnitudes of the very same contributions that form `dot`.
    // p^T A p is a sum of ~2*nnz SIGNED terms, so when the true curvature is
    // small next to the individual terms the fp32 result is dominated by
    // round-off; eps * absdot is the bound on that error and is what lets the
    // caller tell a cancelled zero from a genuinely negative curvature.
    float absdot = 0.0f;
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
                const float term =
                    2.0f * (xi[0] * v[0] + xi[1] * v[1] + xi[2] * v[2]);
                dot += term;
                absdot += fabsf(term);
            } else {
                const float term = xi[0] * v[0] + xi[1] * v[1] + xi[2] * v[2];
                dot += term;
                absdot += fabsf(term);
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
                const float term =
                    2.0f * (xi[0] * v[0] + xi[1] * v[1] + xi[2] * v[2]);
                dot += term;
                absdot += fabsf(term);
            } else {
                const float term = xi[0] * v[0] + xi[1] * v[1] + xi[2] * v[2];
                dot += term;
                absdot += fabsf(term);
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
        const float term = xi[0] * v[0] + xi[1] * v[1] + xi[2] * v[2];
        dot += term;
        absdot += fabsf(term);
        // The row's own accumulation must also be atomic: transpose scatters
        // from other rows target the same slots.
        atomicAdd(result.data + 3 * i + 0, sum[0]);
        atomicAdd(result.data + 3 * i + 1, sum[1]);
        atomicAdd(result.data + 3 * i + 2, sum[2]);
    }
    s_dot[threadIdx.x] = dot;
    s_absdot[threadIdx.x] = absdot;
    __syncthreads();
    for (unsigned w = CGF_BLOCK / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            s_dot[threadIdx.x] += s_dot[threadIdx.x + w];
            s_absdot[threadIdx.x] += s_absdot[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dot_partials[blockIdx.x] = s_dot[0];
        absdot_partials[blockIdx.x] = s_absdot[0];
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
                                         float *dot_partials,
                                         float *absdot_partials) {
    __shared__ float s_dot[CGF_BLOCK];
    __shared__ float s_absdot[CGF_BLOCK];
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned i = t / CG_ROW_LANES;
    const unsigned lane = threadIdx.x % CG_ROW_LANES;
    float dot = 0.0f;
    float absdot = 0.0f;
    Vec3f sum = Vec3f::Zero();
    // Component-wise magnitudes of the same contributions, so the caller gets
    // the same eps * absdot round-off bound as the symmetric path. This form
    // walks the row, so the bound covers the cancellation inside the row sum as
    // well as the final dot.
    Vec3f abssum = Vec3f::Zero();
    if (i < A.nrow) {
        const unsigned head = A.rows[i].head;
        for (unsigned k = lane; k < head; k += CG_ROW_LANES) {
            const float *m =
                reinterpret_cast<const float *>(A.rows[i].value + k);
            unsigned j = A.rows[i].index[k];
            const Vec3f v = UnrolledMat3x3f(m) * (p.data + 3 * j);
            sum += v;
            for (unsigned c = 0; c < 3; ++c) abssum[c] += fabsf(v[c]);
        }
        const unsigned ref_head = A.rows[i].ref_head;
        for (unsigned k = lane; k < ref_head; k += CG_ROW_LANES) {
            const float *m = reinterpret_cast<const float *>(
                A.dyn_value_buff.data + A.rows[i].ref_value[k]);
            unsigned j = A.rows[i].ref_index[k];
            const Vec3f v = UnrolledMat3x3f(m) ^ (p.data + 3 * j);
            sum += v;
            for (unsigned c = 0; c < 3; ++c) abssum[c] += fabsf(v[c]);
        }
        const unsigned b0 = B.index.offset[i], b1 = B.index.offset[i + 1];
        for (unsigned k = b0 + lane; k < b1; k += CG_ROW_LANES) {
            const float *m = reinterpret_cast<const float *>(B.value.data + k);
            unsigned j = B.index.data[k];
            const Vec3f v = UnrolledMat3x3f(m) * (p.data + 3 * j);
            sum += v;
            for (unsigned c = 0; c < 3; ++c) abssum[c] += fabsf(v[c]);
        }
        const unsigned t0 = B.transpose.offset[i], t1 = B.transpose.offset[i + 1];
        for (unsigned k = t0 + lane; k < t1; k += CG_ROW_LANES) {
            Vec2u ref = B.transpose.data[k];
            const float *m =
                reinterpret_cast<const float *>(B.value.data + ref[1]);
            const Vec3f v = UnrolledMat3x3f(m) ^ (p.data + 3 * ref[0]);
            sum += v;
            for (unsigned c = 0; c < 3; ++c) abssum[c] += fabsf(v[c]);
        }
    }
    for (unsigned off = CG_ROW_LANES / 2; off > 0; off >>= 1) {
        for (unsigned c = 0; c < 3; ++c) {
            sum[c] += __shfl_down_sync(0xffffffffu, sum[c], off, CG_ROW_LANES);
            abssum[c] +=
                __shfl_down_sync(0xffffffffu, abssum[c], off, CG_ROW_LANES);
        }
    }
    if (i < A.nrow && lane == 0) {
        const Vec3f v = UnrolledMat3x3f(C[i].data()) * (p.data + 3 * i);
        sum += v;
        for (unsigned k = 0; k < 3; ++k) {
            abssum[k] += fabsf(v[k]);
            result[3 * i + k] = sum[k];
            dot += p[3 * i + k] * sum[k];
            absdot += fabsf(p[3 * i + k]) * abssum[k];
        }
    }
    s_dot[threadIdx.x] = dot;
    s_absdot[threadIdx.x] = absdot;
    __syncthreads();
    for (unsigned w = CGF_BLOCK / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) {
            s_dot[threadIdx.x] += s_dot[threadIdx.x + w];
            s_absdot[threadIdx.x] += s_absdot[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dot_partials[blockIdx.x] = s_dot[0];
        absdot_partials[blockIdx.x] = s_absdot[0];
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
    // flag (pAp/rz0 <= 0 in a divide) and an rz<=0 flag (non-SPD preconditioner),
    // followed by the breakdown diagnostics latched at the offending iterate:
    // which of the two causes fired, the offending scalar, the iteration index,
    // and a running iteration counter. err through break_iter are adjacent so
    // one batched copy reads all of them at a scheduled check.
    auto sc = pool.get<float>(13);
    float *d_rz0 = sc.data + 0;
    float *d_rz1 = sc.data + 1;
    float *d_pAp = sc.data + 2;
    float *d_alpha = sc.data + 3;
    float *d_beta = sc.data + 4;
    float *d_err = sc.data + 5;
    int *d_break = reinterpret_cast<int *>(sc.data + 6);
    int *d_rzbad = reinterpret_cast<int *>(sc.data + 7);
    int *d_break_cause = reinterpret_cast<int *>(sc.data + 8);
    float *d_break_val = sc.data + 9;
    int *d_break_iter = reinterpret_cast<int *>(sc.data + 10);
    int *d_iter_ctr = reinterpret_cast<int *>(sc.data + 11);
    // Sum of |contribution| for the current iterate's p^T A p, reduced from the
    // same kernel that forms it, so eps * absdot bounds that sum's round-off.
    float *d_absdot = sc.data + 12;
    CUDA_HANDLE_ERROR(cudaMemsetAsync(d_break, 0, 2 * sizeof(int), q));
    CUDA_HANDLE_ERROR(cudaMemsetAsync(d_break_cause, 0, 4 * sizeof(float), q));

    // Capture buffers for the breakdown iterate's p and A p. Written only on the
    // breaking iteration (a branch that is not taken in a healthy solve), and
    // read only at the abort site, where the quadratic form is recomputed in
    // double to separate a genuine indefinite term from an fp32 cancellation.
    auto p_break = pool.get<float>(3 * vertex_count);
    auto Ap_break = pool.get<float>(3 * vertex_count);

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
    auto fuse_partials = pool.get<float>(grid_f + grid_err + 2 * grid_mv);
    float *rz_partials = fuse_partials.data;
    float *err_partials = fuse_partials.data + grid_f;
    float *pAp_partials = fuse_partials.data + grid_f + grid_err;
    float *absdot_partials = pAp_partials + grid_mv;

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

    // The relative-residual denominator MUST be the seeded initial residual
    // ||b - A x0||_1, never ||b||_1. cg_device is never entered with a zero x:
    // main.cu pre-loads the pinned rows of dx with their exact prescribed
    // correction before every solve, and those rows carry the pin-barrier
    // stiffness, so in any scene with a driven collider or pin they dominate
    // ||b||_1 by orders of magnitude while the seed annihilates exactly that
    // component. Dividing by ||b||_1 therefore makes tol * ||b||_1 exceed the
    // entire dynamic residual: PCG "converges" in a handful of iterations, the
    // stiff membrane stretch modes never resolve, and the sheet stretches (the
    // strain limiter then truncates every step to a fraction of its span). This
    // was shipped once and reverted; do not re-land it.

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

    // PPF_CG_SYMPROBE=1: measure how symmetric the APPLIED operator is, once
    // per solve. A is assembled as an upper triangle plus a transpose scatter,
    // so u^T(A v) and v^T(A u) must agree to fp32 round-off; a gap far above
    // that means either a structurally missing/duplicated transpose (a real
    // assembly bug) or that the fp32 SpMV output is itself swamped by
    // cancellation. Watching the gap as conditioning degrades separates the
    // two: a structural fault is constant, cancellation grows.
    // PPF_CG_SYMPROBE=N runs the probe on every N-th solve (it costs two extra
    // SpMVs and three device-to-host vector copies, so leaving it on every
    // solve would distort the run it is measuring).
    static unsigned sym_probe_stride = [] {
        const char *e = std::getenv("PPF_CG_SYMPROBE");
        return e ? (unsigned)std::atoi(e) : 0u;
    }();
    static unsigned sym_probe_calls = 0;
    const bool sym_probe =
        sym_probe_stride && (sym_probe_calls++ % sym_probe_stride == 0);
    if (sym_probe) {
        const size_t n3 = 3 * (size_t)vertex_count;
        auto u = pool.get<float>(n3);
        auto Au = pool.get<float>(n3);
        auto Av = pool.get<float>(n3);
        // v = r (the seeded residual); u = a deterministic, index-varying
        // vector so the probe is not accidentally special-cased by structure.
        std::vector<float> hu(n3);
        for (size_t k = 0; k < n3; ++k) {
            hu[k] = (float)(((k * 2654435761u) % 2048u) / 1024.0 - 1.0);
        }
        CUDA_HANDLE_ERROR(cudaMemcpy(u.data, hu.data(), n3 * sizeof(float),
                                     cudaMemcpyHostToDevice));
        op.apply(u, Au, q);
        op.apply(r, Av, q);
        CUDA_HANDLE_ERROR(cudaStreamSynchronize(q));
        std::vector<float> hAu(n3), hAv(n3), hv(n3);
        CUDA_HANDLE_ERROR(cudaMemcpy(hAu.data(), Au.data, n3 * sizeof(float),
                                     cudaMemcpyDeviceToHost));
        CUDA_HANDLE_ERROR(cudaMemcpy(hAv.data(), Av.data, n3 * sizeof(float),
                                     cudaMemcpyDeviceToHost));
        CUDA_HANDLE_ERROR(cudaMemcpy(hv.data(), r.data, n3 * sizeof(float),
                                     cudaMemcpyDeviceToHost));
        double uAv = 0.0, vAu = 0.0, absu = 0.0;
        for (size_t k = 0; k < n3; ++k) {
            uAv += (double)hu[k] * (double)hAv[k];
            vAu += (double)hv[k] * (double)hAu[k];
            absu += std::fabs((double)hu[k] * (double)hAv[k]);
        }
        const double denom = std::fabs(uAv) + std::fabs(vAu);
        printf("* sym probe: u^T(Av) %.6e  v^T(Au) %.6e  relgap %.3e  "
               "sum|terms| %.6e\n",
               uAv, vAu, denom > 0.0 ? std::fabs(uAv - vAu) * 2.0 / denom : 0.0,
               absu);
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
                    op.A, op.B, op.C, p, tmp, pAp_partials, absdot_partials);
            } else {
                cg_apply_dot_walk_kernel<<<grid_mv, CGF_BLOCK, 0, q>>>(
                    op.A, op.B, op.C, p, tmp, pAp_partials, absdot_partials);
            }
            CUDA_HANDLE_ERROR(cudaGetLastError());
            cg_reduce_dot_kernel<<<2, CGF_BLOCK, 0, q>>>(
                pAp_partials, absdot_partials, grid_mv, d_pAp, d_absdot);
            CUDA_HANDLE_ERROR(cudaGetLastError());
            cg_fused_update_kernel<<<grid_f, CGF_BLOCK, 0, q>>>(
                p.data, tmp.data, x.data, r.data, z.data, op.P.data, d_rz0,
                d_pAp, d_absdot, d_break, vertex_count, rz_partials,
                err_partials, d_break_cause, d_break_val, d_iter_ctr,
                d_break_iter, p_break.data, Ap_break.data);
            CUDA_HANDLE_ERROR(cudaGetLastError());
            cg_reduce2_kernel<<<2, CGF_BLOCK, 0, q>>>(
                rz_partials, err_partials, grid_f, d_rz1, d_err, d_beta,
                d_rz0, d_break, d_rzbad, check_spd ? 1 : 0, d_break_cause,
                d_break_val, d_iter_ctr, d_break_iter);
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
                    op.A, op.B, op.C, p, tmp, pAp_partials, absdot_partials);
            } else {
                cg_apply_dot_walk_kernel<<<grid_mv, CGF_BLOCK, 0, q>>>(
                    op.A, op.B, op.C, p, tmp, pAp_partials, absdot_partials);
            }
            CUDA_HANDLE_ERROR(cudaGetLastError());
            cg_reduce_dot_kernel<<<2, CGF_BLOCK, 0, q>>>(
                pAp_partials, absdot_partials, grid_mv, d_pAp, d_absdot);
            CUDA_HANDLE_ERROR(cudaGetLastError());
            const unsigned grid_3n = (3 * vertex_count + CGF_BLOCK - 1) /
                                     CGF_BLOCK;
            cg_update_xr_kernel<<<grid_3n, CGF_BLOCK, 0, q>>>(
                p.data, tmp.data, x.data, r.data, d_rz0, d_pAp, d_absdot,
                d_break, 3 * vertex_count, err_partials, d_break_cause,
                d_break_val, p_break.data, Ap_break.data);
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
            probe_pin = static_cast<float *>(pinned_scratch(8 * sizeof(float)));
            CUDA_HANDLE_ERROR(cudaMemcpyAsync(probe_pin, d_err,
                                              6 * sizeof(float),
                                              cudaMemcpyDeviceToHost, q));
            CUDA_HANDLE_ERROR(cudaStreamSynchronize(q));
            const double reresid = (double)probe_pin[0] / err0;
            const int broke = *reinterpret_cast<const int *>(&probe_pin[1]);
            const int rzbad = *reinterpret_cast<const int *>(&probe_pin[2]);
            const int break_cause = *reinterpret_cast<const int *>(&probe_pin[3]);
            const float break_val = probe_pin[4];
            const unsigned break_iter =
                (unsigned)*reinterpret_cast<const int *>(&probe_pin[5]);
            // Convergence is tested first, matching the host cg() ordering: a
            // finite residual below tol wins even if a flag latched earlier (a
            // NaN/Inf residual fails the < tol test and falls through).
            if (reresid < (double)tol) {
                return {true, iter, (float)reresid, false};
            }
            // Hard invariant checks (convergence is tested above, so these fire
            // only on a NON-converged breakdown, never on a benign near-tol
            // float edge). p^T A p <= 0 means the assembled Newton system matrix
            // is not SPD; r^T M^-1 r <= 0 means the preconditioner is not SPD.
            // Both are assembly bugs, so abort loudly with the exact site instead
            // of silently failing (or masking a non-SPD preconditioner with the
            // block-Jacobi fallback) -- an indefinite matrix must surface here.
            if (broke) {
                if (break_cause == BREAK_CAUSE_NOISE) {
                    // The curvature along p was smaller than the round-off of
                    // its own reduction, so its sign carries no information and
                    // alpha = rz/pAp cannot be formed. This is the conditioning
                    // limit of fp32 on a contact-dominated system, NOT a broken
                    // assembly, so the solve stops here and returns the iterate
                    // it has (truncated CG, as on any non-positive-curvature
                    // direction). The Newton loop line-searches that direction
                    // under CCD, so an inexact one is safe; aborting the whole
                    // simulation for it is not.
                    printf("* cg truncated: curvature %.3e within round-off of "
                           "its own reduction at iter %u (reresid %.3e)\n",
                           (double)break_val, break_iter, reresid);
                    return {true, iter, (float)reresid, false};
                }
                if (break_cause == BREAK_CAUSE_RZ) {
                    fatal_nonpositive_rz(break_iter, iter, reresid,
                                         (double)break_val);
                }
                if (break_cause == BREAK_CAUSE_NONE) {
                    // The flag was raised by a site that does not latch a cause
                    // (kernels::scalar_div on the Schwarz path). Say so instead
                    // of blaming p^T A p, which is what the old single-message
                    // report did for every cause.
                    fprintf(stderr,
                            "PPF FATAL: PCG breakdown with no latched cause at "
                            "check iter %u (reresid %.3e). Raised by a divide "
                            "guard outside the fused path.\n",
                            iter, reresid);
                    fflush(stderr);
                    std::abort();
                }
                // p^T A p breakdown. The iterate that tripped it was captured in
                // the kernel, so the quadratic form can be recomputed exactly:
                // sum in double on the host (device code stays fp32-only) and
                // compare against the fp32 device reduction. Equal-and-negative
                // is a genuine indefinite term; positive-in-double is an fp32
                // cancellation. This runs once, on the way to abort, so its cost
                // is irrelevant.
                const size_t n3 = 3 * (size_t)vertex_count;
                std::vector<float> hp(n3), hAp(n3), hz(n3), hAz(n3);
                CUDA_HANDLE_ERROR(cudaMemcpy(hp.data(), p_break.data,
                                             n3 * sizeof(float),
                                             cudaMemcpyDeviceToHost));
                CUDA_HANDLE_ERROR(cudaMemcpy(hAp.data(), Ap_break.data,
                                             n3 * sizeof(float),
                                             cudaMemcpyDeviceToHost));
                double pAp_exact = 0.0, pp = 0.0;
                for (size_t k = 0; k < n3; ++k) {
                    pAp_exact += (double)hp[k] * (double)hAp[k];
                    pp += (double)hp[k] * (double)hp[k];
                }
                // Symmetry probe. A is fixed for the whole solve, so for a
                // symmetric operator z^T(A p) and p^T(A z) must agree to fp32
                // noise. A wide gap means the applied operator is not symmetric,
                // which would make p^T A p meaningless as a curvature.
                op.apply(z, tmp, q);
                CUDA_HANDLE_ERROR(cudaStreamSynchronize(q));
                CUDA_HANDLE_ERROR(cudaMemcpy(hz.data(), z.data,
                                             n3 * sizeof(float),
                                             cudaMemcpyDeviceToHost));
                CUDA_HANDLE_ERROR(cudaMemcpy(hAz.data(), tmp.data,
                                             n3 * sizeof(float),
                                             cudaMemcpyDeviceToHost));
                double sym_zAp = 0.0, sym_pAz = 0.0;
                for (size_t k = 0; k < n3; ++k) {
                    sym_zAp += (double)hz[k] * (double)hAp[k];
                    sym_pAz += (double)hp[k] * (double)hAz[k];
                }
                float absdot_host = 0.0f;
                CUDA_HANDLE_ERROR(cudaMemcpy(&absdot_host, d_absdot,
                                             sizeof(float),
                                             cudaMemcpyDeviceToHost));
                fatal_indefinite_system_matrix(
                    break_iter, iter, reresid, (double)break_val, pAp_exact, pp,
                    sym_zAp, sym_pAz,
                    (double)cg_curvature_bound(absdot_host));
            }
            if (rzbad) {
                fatal_nonspd_preconditioner(iter, reresid);
            }
            if (!std::isfinite(reresid)) {
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
            const double pAp =
                (double)kernels::inner_product(p.data, tmp.data, p.size);
            // Same rule and the SAME decision function as the device path: the
            // sign of the curvature is only acted on once it clears the
            // round-off of its own sum. That path reduces sum|contribution|
            // alongside the dot; this SpMV does not produce it, so bound the
            // same quantity through Cauchy-Schwarz, sum|p_i (Ap)_i| <= |p|_2
            // |Ap|_2, which needs no extra kernel and errs toward truncating
            // rather than aborting. Only that scale is path-specific; the
            // verdict drawn from it is shared, so the device paths' constant
            // exercise covers this rarely-entered loop. Without any test at all
            // (as this loop stood) a curvature of zero divided straight through
            // to an infinite alpha.
            const double p_norm =
                std::sqrt((double)kernels::inner_product(p.data, p.data, p.size));
            const double ap_norm = std::sqrt(
                (double)kernels::inner_product(tmp.data, tmp.data, tmp.size));
            const float bound = cg_curvature_bound((float)(p_norm * ap_norm));
            const CurvatureVerdict verdict =
                cg_curvature_verdict((float)pAp, bound);
            if (verdict != CURVATURE_OK) {
                if (verdict == CURVATURE_FATAL) {
                    fatal_indefinite_hostsync(iter, pAp, (double)bound);
                }
                printf("* cg truncated (host loop): curvature %.3e within "
                       "round-off bound %.3e at iter %u\n",
                       pAp, bound, iter);
                // op.norm reuses tmp as scratch, which is free to clobber now
                // that Ap has been consumed.
                return {true, iter, (float)(op.norm(r, tmp) / err0), fell_back};
            }
            double alpha = rz0 / pAp;
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
//
// CONVERGENCE IS TESTED PER DOF GROUP, NEVER ON ONE NORM OVER THE WHOLE REDUCED
// VECTOR. The reduced vector holds two incommensurable kinds of row: the cloth's
// per-vertex force rows, and each body's SIX rows, which are a force/torque
// wrench summed over ALL of that body's vertices. One relative residual over the
// union is therefore scaled by whichever group happens to be largest, and the
// other group's accuracy rides on that unrelated number.
//
// It is not a theoretical worry, it is the failure this criterion exists to
// prevent. A body far from the cloth, touching nothing, still enters that norm.
// Give it some speed (its inertia row is mass * (u/dt + g), so a body moving at
// u contributes 1 + u/(g dt) times its own weight, ~60x within a second of free
// fall at dt = 0.01) or some mass, and it owns ||r||_1. Its 6x6 block is then
// preconditioned by the EXACT reduced inertia and, absent contact, is coupled to
// nothing, so a single CG step annihilates it and drops ||r||_1 below tol * err0
// at iteration ONE, with the cloth's rows still carrying their full residual.
// The cloth's Newton direction is then one Jacobi sweep, which is exactly the
// under-converged-PCG signature (a stretching sheet, a strain limiter clamping
// every step) that the seeded-residual denominator rule also guards against.
// examples/pdrd_cloth_isolation.py is the gate: it stands that scene up and
// fails if a solve stops short of the cloth's own tolerance.
//
// So each group is measured against ITS OWN seeded initial residual, and the
// solve stops when the WORST group is below tol. Absent contact between them the
// cloth's scale is then bit-for-bit the scale it would have if no body existed (P
// is the identity on cloth rows and the seed zeroes the body block, so the cloth
// rows of f - R xr are exactly b - M x0 there, reduced over the same span in the
// same order), which is the invariant a user expects: adding a distant rigid body
// must not change the cloth. Per body, not bodies as one group, for the same
// reason: a heavy body must not set a light body's scale.
std::tuple<bool, unsigned, float>
cg_rigid(const DynCSRMat &A, const FixedCSRMat &B, const Vec<Mat3x3f> &C,
         PDRD::RigidMap &rm, PDRD::RigidPrecond &P, const Vec<Mat3x3f> &inv_diag,
         Vec<float> bvec, Vec<float> x, unsigned max_iter, float tol,
         Vec<float> dtheta_out) {
    buffer::MemoryPool &pool = buffer::get();
    const unsigned dim = rm.dim;
    const unsigned nrow = rm.nrow;
    // One group per body plus the cloth block. n_bodies is nonzero on every path
    // that reaches here (solve() gates on it), so ngrp >= 2 in practice.
    const unsigned nb = rm.n_bodies;
    const unsigned body_base = rm.body_base;
    const unsigned ngrp = 1u + nb;
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
    // Per-group residual scratch, three ngrp-long spans in ONE pooled buffer:
    // the seeded initial L1 per group, the current L1, and the ratio the max
    // reduction consumes. The pool hands out the first free buffer that fits, so
    // three separate ngrp requests would each occupy a whole vertex-sized slot.
    auto gb = pool.get<float>(3 * ngrp);
    float *g0 = gb.data, *gc = g0 + ngrp, *gr = gc + ngrp;

    auto Rapply = [&](Vec<float> in, Vec<float> out) {
        PDRD::launch_prolong_rigid(rm, in, xv);
        solver::apply(A, B, C, 0.0f, xv, mxv);
        PDRD::launch_restrict_rigid(rm, mxv, out);
        PDRD::launch_project_bodies(rm, out);
    };
    // Per-group L1 norms of `w` into `out[ngrp]`: out[0] is the cloth block
    // (rows [0, body_base)), out[1 + b] is body b's six reduced wrench rows.
    auto group_l1 = [&](Vec<float> w, float *out) {
        if (body_base) {
            DISPATCH_START(body_base)
            [w, tmp] __device__(unsigned i) mutable {
                tmp.data[i] = fabsf(w.data[i]);
            } DISPATCH_END;
            kernels::sum_into(tmp.data, out, body_base);
        } else {
            // A scene with no cloth row at all: the group exists but is empty.
            CUDA_HANDLE_ERROR(cudaMemset(out, 0, sizeof(float)));
        }
        DISPATCH_START(nb)
        [w, out, body_base] __device__(unsigned b) mutable {
            const float *q = w.data + body_base + 6u * b;
            float s = 0.0f;
            for (unsigned k = 0; k < 6; ++k) {
                s += fabsf(q[k]);
            }
            out[1u + b] = s;
        } DISPATCH_END;
    };

    PDRD::launch_restrict_rigid(rm, bvec, f); // f = P^T b
    PDRD::launch_project_bodies(rm, f);       // restrict the rhs to the joint subspace
    // Seed the reduced solve with the caller's initial guess instead of zero.
    // main.cu pre-loads every prescribed (Dirichlet) row of `x` with its exact
    // correction before each solve, and `reduced_rigid` routes the WHOLE system
    // through here -- cloth rows included -- as soon as any PDRD body exists. A
    // memset therefore threw that seed away, so (i) a prescribed row converged
    // only to PCG tolerance instead of exactly, and (ii) its RHS (a
    // displacement) entered err0 alongside genuine force rows, which is the
    // pin-dominated `||b||` denominator that must never be the tolerance scale.
    // Seeding restores both: the prescribed rows start at residual zero and drop
    // out of err0, leaving the dynamic residual as the scale.
    PDRD::launch_seed_restrict(rm, x, xr);
    PDRD::launch_project_bodies(rm, xr);
    Rapply(xr, Rp);                     // Rp = R * xr
    kernels::copy(f.data, r.data, dim); // r = f - R*xr
    kernels::add_scaled(Rp.data, r.data, -1.0f, dim);
    // Seeded initial residual, per group. The host copy is ngrp floats read once
    // per solve; every in-loop test stays on the device but for one scalar.
    group_l1(r, g0);
    std::vector<float> h_g0(ngrp);
    CUDA_HANDLE_ERROR(cudaMemcpy(h_g0.data(), g0, ngrp * sizeof(float),
                                 cudaMemcpyDeviceToHost));
    double err0 = 0.0;
    for (unsigned g = 0; g < ngrp; ++g) {
        err0 += (double)h_g0[g];
    }
    if (err0 == 0.0) {
        // Already exact: prolong the seed back out rather than zeroing x (a
        // memset here would discard the prescribed rows the caller seeded).
        PDRD::launch_prolong_rigid(rm, xr, x);
        return {true, 1u, 0.0f};
    }
    const float err0_all = (float)err0;
    // Relative residual of the WORST group. A group whose seeded initial
    // residual is exactly zero has no scale of its own (it starts solved, and can
    // only pick up residual through coupling to another group), so it is measured
    // against the whole system's initial scale; err0 > 0 here, so every scale is
    // positive and a ratio can only go non-finite if the residual itself does.
    auto worst_reresid = [&]() {
        group_l1(r, gc);
        DISPATCH_START(ngrp)
        [gc, g0, gr, err0_all] __device__(unsigned g) mutable {
            float s0 = g0[g];
            float ratio = gc[g] / (s0 > 0.0f ? s0 : err0_all);
            // A max reduction DROPS a NaN (fmaxf(NaN, x) == x), which would read
            // as convergence. Map it to infinity so it survives the reduce and
            // trips the non-finite check below.
            gr[g] = isfinite(ratio) ? ratio : INFINITY;
        } DISPATCH_END;
        return (double)kernels::max_array(gr, ngrp, 0.0f);
    };
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
        reresid = (float)worst_reresid();
        if (reresid < tol) {
            success = true;
            break;
        }
        // Non-finite covers the old NaN test plus the infinity the group ratio
        // maps a NaN residual to; neither can be mistaken for convergence.
        if (iter >= max_iter || !std::isfinite((double)reresid)) {
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
        // Persist the rigid DOF map across Newton iterations: build_rigid_map
        // rebuilds only the per-iteration prot, reusing the cached topology
        // partition (see pdrd_rigid.hpp). Mirrors the static s_rprec below; the
        // process is one-shot per session so the cache never outlives its scene.
        static PDRD::RigidMap rm;
        PDRD::build_rigid_map(rm, data, state, vertex_count);
        static PDRD::RigidPrecond s_rprec;
        PDRD::build_rigid_precond(s_rprec, data, state, A, B, dt);
        bool ok;
        std::tie(ok, iter, resid) =
            cg_rigid(A, B, C, rm, s_rprec, inv_diag_vec, b, x, max_iter, tol,
                     pdrd_dtheta_out);
        schwarz_fallback = 0u;
        return ok;
    }

    // Build the aggregate-Schwarz hierarchy just before PCG (the linear system
    // M = A_dyn + B_fixed + C_diag is fully known here). Selected by
    // ParamSet::precond (default block-jacobi; "schwarz" picks the aggregate-Schwarz base).
    // schwarz::build owns a persistent cache; the hierarchy holds views into it
    // and stays valid for the whole solve. (PDRD scenes return above; this is the
    // cloth / general PCG path.)
    schwarz::SchwarzHierarchy hierarchy;
    const bool use_schwarz = prm.precond == PrecondMode::Schwarz;
    bool schwarz_degraded = false;
    if (use_schwarz) {
        schwarz::build(hierarchy, A, B, C, positions, prm.schwarz_levels);
        if (hierarchy.n_agg > 0) {
            ops.set_schwarz(&hierarchy);
        } else if (vertex_count > 0) {
            // The Schwarz memory guard could not fit even the single-level base
            // in PPF_SCHWARZ_MEM_FRAC of free VRAM (H.n_agg == 0). Leave the
            // hierarchy uninstalled so cg() takes the SPD block-Jacobi base
            // (op.H == nullptr); latch the fallback flag for logging.
            schwarz_degraded = true;
        }
    }

    bool success;
    bool fell_back;
    std::tie(success, iter, resid, fell_back) = cg(ops, b, x, max_iter, tol);
    schwarz_fallback = (fell_back || schwarz_degraded) ? 1u : 0u;

    // PooledVec auto-releases when exiting function
    return success;
}

} // namespace solver
