// File: schwarz.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../buffer/buffer.hpp"
#include "../kernels/exclusive_scan.hpp"
#include "../kernels/radix_sort.hpp"
#include "../main/cuda_utils.hpp"
#include "../simplelog/SimpleLog.h"
#include "../utility/dispatcher.hpp"
#include "schwarz.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>


namespace schwarz {

static constexpr unsigned UNSET = 0xffffffffu;
static constexpr unsigned BLOCK_THREADS = 64;
static constexpr unsigned NACC = 256; // dyn-hash accumulators (reduce contention)

__device__ inline unsigned long long mix64(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// Local index of global vertex j within the aggregate member list mem[0..m), or
// -1 if j is not a member (an external coupling, excluded from A_loc).
__device__ inline int find_local(const unsigned *mem, unsigned m, unsigned j) {
    for (unsigned a = 0; a < m; ++a) {
        if (mem[a] == j) {
            return (int)a;
        }
    }
    return -1;
}

// Packed lower-triangle layout for a d x d lower-triangular matrix: row i holds
// entries [i,0..i], so its base is i(i+1)/2 and (i,j) (i>=j) lives at
// i(i+1)/2 + j. Used to store each aggregate's inverse Cholesky factor G = L^-1
// (lower triangular) in half the memory; the apply stages it into shared memory
// once and forms z = G^T (G r).
__host__ __device__ inline unsigned tri_size(unsigned d) {
    return d * (d + 1u) / 2u;
}
__device__ inline unsigned tri_idx(unsigned i, unsigned j) { // requires i >= j
    return i * (i + 1u) / 2u + j;
}

// Transpose a 3x3 block (for the L2/L4 lower-triangle reconstruction when
// materializing M as a both-triangle block-CSR).
__device__ inline Mat3x3f transpose3(const Mat3x3f &b) {
    Mat3x3f t;
    for (int r = 0; r < 3; ++r)
        for (int s = 0; s < 3; ++s)
            t(r, s) = b(s, r);
    return t;
}

// Frobenius norm of a 3x3 block.
__device__ inline float frob9(const Mat3x3f &m) {
    float s = 0.0f;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            float v = m(r, c);
            s += v * v;
        }
    return sqrtf(s);
}

// Capacity to request when growing a contact-driven per-build device buffer:
// 1.5x headroom (amortized growth, like std::vector) so a buffer whose size
// tracks the fluctuating contact-matrix nnz / aggregate counts reuses the slack
// on a new high instead of reallocating every step. The high-water guards below
// only fire on a genuine jump past the headroom, so the per-build alloc/free
// churn settles to zero once the scene's contact count stabilizes.
__host__ static inline unsigned grow_cap(size_t n) {
    return (unsigned)(n + n / 2);
}

// Dynamic shared-memory the per-aggregate dense factor (factor_kernel /
// factor_kernel_coarse) needs for a chunk size of `k` vertices: two dense
// max_dim x max_dim work matrices plus a pivot column (floats) and the member
// list (unsigned), with max_dim = 3*k DOF. Grows quadratically in the chunk
// size, so it is the gate on which GPUs a given kmax can run.
__host__ static inline size_t factor_shmem(unsigned k) {
    size_t md = 3u * (size_t)k;
    return (2u * md * md + md) * sizeof(float) + (size_t)k * sizeof(unsigned);
}

// Grant CUDA function `fn` up to `need` bytes of dynamic shared memory, opting
// past the ~48KB default only when required and at most once per high-water
// mark (`cache`, seeded with the default). The cudaFuncSetAttribute result is
// CHECKED: a device that cannot provide `need` (its per-block opt-in cap is
// smaller) returns cudaErrorInvalidValue, which we consume here instead of
// letting it leak to a later cudaGetLastError() and abort an unrelated launch
// on small-shared-memory GPUs. Callers clamp kmax via factor_shmem() so `need`
// never exceeds the cap on the live path; the false return is a defensive net.
static bool ensure_dyn_shmem(const void *fn, size_t need, int &cache) {
    if ((int)need <= cache) {
        return true;
    }
    cudaError_t err = cudaFuncSetAttribute(
        fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)need);
    if (err != cudaSuccess) {
        cudaGetLastError(); // consume; do not let it surface at the next check
        return false;
    }
    cache = (int)need;
    return true;
}

// Reallocate-on-grow a device Vec, with 1.5x headroom on the backing capacity
// so a size that creeps up with the contact count reuses the slack instead of
// reallocating every build. `allocated` tracks the real capacity; `size` is the
// requested logical length (callers index with explicit counts, never with the
// buffer's own .size, so reporting the logical n here is both correct and what
// the high-water guard above each cap-tracked buffer expects).
template <class T> static void ensure(Vec<T> &v, size_t n) {
    if (v.data == nullptr || v.allocated < n) {
        v.free();
        v = Vec<T>::alloc(grow_cap(n > 0 ? n : 1));
    }
    v.size = (unsigned)n;
}

// Try to attach UNSET vertex i to the aggregate of an assigned neighbor j,
// respecting the kmax member cap (strict, via atomic claim/unclaim).
__device__ inline bool try_attach(Vec<unsigned> agg, Vec<unsigned> cnt,
                                  unsigned i, unsigned j, unsigned kmax) {
    if (j == i || agg.data[j] == UNSET) {
        return false;
    }
    const unsigned a = agg.data[j];
    const unsigned old = atomicAdd(&cnt.data[a], 1u);
    if (old < kmax) {
        agg.data[i] = a;
        return true;
    }
    atomicSub(&cnt.data[a], 1u);
    return false;
}

// One CUDA block per aggregate. Gathers the dense SPD submatrix A_loc of
// M = A_dyn + B_fixed + C_diag restricted to the aggregate's DOF, applies an
// absolute diagonal floor, Cholesky-factors it (A_loc = L L^T) and stores the
// inverse factor G = L^-1 (packed lower triangle) into ainv. Factor + triangular
// inverse are COOPERATIVE over the whole thread block; all working matrices live
// in shared memory. Applying A_loc^-1 = G^T G as a Gram form keeps the
// preconditioner SPD in float32.
__global__ void factor_kernel(DynCSRMat A, FixedCSRMat B, Vec<Mat3x3f> C,
                              Vec<unsigned> agg_off, Vec<unsigned> members,
                              Vec<unsigned> ainv_off, Vec<float> ainv,
                              unsigned max_dim) {
    extern __shared__ float smem[];
    float *Aloc = smem;                              // max_dim*max_dim
    float *Inv = smem + (size_t)max_dim * max_dim;   // max_dim*max_dim
    float *fcol = Inv + (size_t)max_dim * max_dim;   // max_dim (pivot column)
    unsigned *mem = (unsigned *)(fcol + max_dim);    // members

    const unsigned g = blockIdx.x;
    const unsigned base = agg_off.data[g];
    const unsigned m = agg_off.data[g + 1] - base;
    const unsigned d = 3u * m;
    const unsigned tid = threadIdx.x;
    const unsigned nthreads = blockDim.x;

    for (unsigned a = tid; a < m; a += nthreads) {
        mem[a] = members.data[base + a];
    }
    for (unsigned e = tid; e < d * d; e += nthreads) {
        Aloc[e] = 0.0f;
    }
    __syncthreads();

    // Gather: thread a owns member row a (A_loc rows 3a..3a+2), race-free. Walk
    // the four neighbor loops apply() uses, keeping intra-aggregate columns.
    for (unsigned a = tid; a < m; a += nthreads) {
        const unsigned i = mem[a];
        const Row &row = A.rows.data[i];
        for (unsigned k = 0; k < row.head; ++k) { // (L1) dyn upper
            int b = find_local(mem, m, row.index[k]);
            if (b >= 0) {
                const Mat3x3f &blk = row.value[k];
                for (int r = 0; r < 3; ++r)
                    for (int s = 0; s < 3; ++s)
                        Aloc[(3 * a + r) * d + (3 * b + s)] += blk(r, s);
            }
        }
        for (unsigned k = 0; k < row.ref_head; ++k) { // (L2) dyn lower / transpose
            int b = find_local(mem, m, row.ref_index[k]);
            if (b >= 0) {
                const Mat3x3f &blk = A.dyn_value_buff.data[row.ref_value[k]];
                for (int r = 0; r < 3; ++r)
                    for (int s = 0; s < 3; ++s)
                        Aloc[(3 * a + r) * d + (3 * b + s)] += blk(s, r);
            }
        }
        for (unsigned k = B.index.offset[i]; k < B.index.offset[i + 1]; ++k) { // (L3) fixed upper
            int b = find_local(mem, m, B.index.data[k]);
            if (b >= 0) {
                const Mat3x3f &blk = B.value.data[k];
                for (int r = 0; r < 3; ++r)
                    for (int s = 0; s < 3; ++s)
                        Aloc[(3 * a + r) * d + (3 * b + s)] += blk(r, s);
            }
        }
        for (unsigned k = B.transpose.offset[i]; k < B.transpose.offset[i + 1];
             ++k) { // (L4) fixed lower / transpose
            Vec2u ref = B.transpose.data[k];
            int b = find_local(mem, m, ref[0]);
            if (b >= 0) {
                const Mat3x3f &blk = B.value.data[ref[1]];
                for (int r = 0; r < 3; ++r)
                    for (int s = 0; s < 3; ++s)
                        Aloc[(3 * a + r) * d + (3 * b + s)] += blk(s, r);
            }
        }
        const Mat3x3f &cb = C.data[i]; // diagonal C[i]
        for (int r = 0; r < 3; ++r)
            for (int s = 0; s < 3; ++s)
                Aloc[(3 * a + r) * d + (3 * a + s)] += cb(r, s);
    }
    __syncthreads();

    // Absolute diagonal floor (handles pinned/dead blocks; never 1/0). `s_floor`
    // also serves as the Cholesky pivot clamp below.
    (void)fcol;
    __shared__ float s_floor;
    if (tid == 0) {
        float tr = 0.0f;
        for (unsigned k = 0; k < d; ++k) {
            tr += Aloc[k * d + k];
        }
        const float fl = 1e-6f * (tr / (float)d) + 1e-8f;
        for (unsigned k = 0; k < d; ++k) {
            Aloc[k * d + k] += fl;
        }
        s_floor = fl;
    }
    __syncthreads();

    // Cooperative Cholesky A_loc = L L^T, overwriting the lower triangle of Aloc
    // with L. A_loc is a principal submatrix of the SPD Newton operator, so it is
    // SPD; the pivot is clamped to the diagonal floor so float32 cancellation (or
    // a near-indefinite block) still yields a valid SPD factor (we then condition
    // with a nearby SPD matrix). Column j depends on columns < j (sequential in
    // j); within a column the sub-diagonal rows fill in parallel.
    for (unsigned j = 0; j < d; ++j) {
        if (tid == 0) {
            float s = Aloc[j * d + j];
            for (unsigned k = 0; k < j; ++k) {
                const float ljk = Aloc[j * d + k];
                s -= ljk * ljk;
            }
            Aloc[j * d + j] = sqrtf(s > s_floor ? s : s_floor);
        }
        __syncthreads();
        const float inv_ljj = 1.0f / Aloc[j * d + j];
        for (unsigned i = j + 1 + tid; i < d; i += nthreads) {
            float s = Aloc[i * d + j];
            for (unsigned k = 0; k < j; ++k) {
                s -= Aloc[i * d + k] * Aloc[j * d + k];
            }
            Aloc[i * d + j] = s * inv_ljj;
        }
        __syncthreads();
    }

    // Triangular inverse G = L^{-1} (lower-triangular) into Inv. Column `col` of G
    // depends only on L (read-only now) and earlier rows of the same column, so
    // columns are independent: each thread owns a stripe of columns with no inner
    // sync. We store G (not A_loc^{-1}) so the apply z = G^T (G r) is SPD by
    // construction, r . z = ||G r||^2 >= 0 in float32 (no rz<=0 breakdown).
    for (unsigned col = tid; col < d; col += nthreads) {
        Inv[col * d + col] = 1.0f / Aloc[col * d + col];
        for (unsigned i = col + 1; i < d; ++i) {
            float s = 0.0f;
            for (unsigned k = col; k < i; ++k) {
                s += Aloc[i * d + k] * Inv[k * d + col];
            }
            Inv[i * d + col] = -s / Aloc[i * d + i];
        }
    }
    __syncthreads();

    // Store the lower triangle of G = L^{-1}, packed.
    const unsigned abase = ainv_off.data[g];
    for (unsigned e = tid; e < d * d; e += nthreads) {
        const unsigned i = e / d, j = e % d;
        if (i >= j) {
            ainv.data[abase + tri_idx(i, j)] = Inv[e];
        }
    }
}

// One CUDA block per aggregate. result_segment = A_loc^{-1} * x_segment, applied
// as z = G^T (G r) with G = L^{-1} the stored inverse Cholesky factor (packed
// lower triangle). This is SPD by construction: r . z = ||G r||^2 >= 0 in float32,
// so the aggregate term never produces the rz<=0 residual that latches the
// block-Jacobi fallback. G is staged into shared memory once; the two passes are
// a lower-triangular matvec (y = G r) then an upper-triangular matvec (z = G^T y),
// together the same flops as the previous symmetric full matvec. Each thread owns
// one output row, so writes are conflict-free.
__global__ void apply_kernel(Vec<unsigned> agg_off, Vec<unsigned> members,
                             Vec<unsigned> ainv_off, Vec<float> ainv,
                             Vec<float> x, Vec<float> result, unsigned max_dim) {
    extern __shared__ float smem[];
    float *T = smem;                              // tri_size(max_dim): packed L^-1
    float *rloc = T + tri_size(max_dim);          // max_dim
    float *yloc = rloc + max_dim;                 // max_dim: y = G r
    unsigned *mem = (unsigned *)(yloc + max_dim);

    const unsigned g = blockIdx.x;
    const unsigned base = agg_off.data[g];
    const unsigned m = agg_off.data[g + 1] - base;
    const unsigned d = 3u * m;
    const unsigned tid = threadIdx.x;
    const unsigned nthreads = blockDim.x;

    for (unsigned a = tid; a < m; a += nthreads) {
        mem[a] = members.data[base + a];
    }
    __syncthreads();
    for (unsigned p = tid; p < d; p += nthreads) {
        rloc[p] = x.data[3u * mem[p / 3] + (p % 3)];
    }
    const unsigned abase = ainv_off.data[g];
    const unsigned tsz = tri_size(d);
    for (unsigned e = tid; e < tsz; e += nthreads) {
        T[e] = ainv.data[abase + e];
    }
    __syncthreads();

    // y = G r  (G lower-triangular: G[p][q] = T[tri_idx(p,q)] for q <= p)
    for (unsigned p = tid; p < d; p += nthreads) {
        float acc = 0.0f;
        for (unsigned q = 0; q <= p; ++q) {
            acc += T[tri_idx(p, q)] * rloc[q];
        }
        yloc[p] = acc;
    }
    __syncthreads();
    // z = G^T y  ((G^T)[p][q] = G[q][p] = T[tri_idx(q,p)] for q >= p)
    for (unsigned p = tid; p < d; p += nthreads) {
        float acc = 0.0f;
        for (unsigned q = p; q < d; ++q) {
            acc += T[tri_idx(q, p)] * yloc[q];
        }
        result.data[3u * mem[p / 3] + (p % 3)] = acc;
    }
}

// ---------------------------------------------------------------------------
// Persistent setup cache. The aggregation (which DOF group together) is a
// topological property of the matrix graph (elastic + contact). It is recomputed
// only when the graph changes: a structural change (vertex count, kmax, or a
// reallocated fixed pattern) OR a contact-connectivity change detected by an
// order-independent structural hash of the dyn column sets (which catches
// rewiring at constant per-row nnz). The dense factors (ainv) are rebuilt every
// solve from live values. Aggregation is fully parallel, so contact-aware
// re-aggregation is cheap even when it fires every solve.
// ---------------------------------------------------------------------------
// One coarse level of the multilevel additive Schwarz hierarchy (level >= 1).
// `A` is the Galerkin operator A_l = C_l A_{l-1} C_l^T (both-triangle block-CSR);
// the level's smoother is the SAME per-domain dense Cholesky/Gram block-Schwarz
// as level 0, run over A_l's matrix-connected domains. `map_fine` is the COMPOSED
// direct map from a level-0 vertex to this level's coarse node (the restriction C_l), used by
// restrict/prolong with no per-apply level chain.
struct SchwarzLevel {
    unsigned n{0};     // nodes at this level (== n_agg of the finer level)
    unsigned n_agg{0}; // domains here (== nodes at the next level)
    unsigned max_dim{0};
    CoarseMat A;
    Vec<unsigned> agg, agg_off, members, ainv_off, cnt, scratch, map_fine;
    Vec<float> ainv;   // packed per-domain inverse Cholesky factors G_l = L_l^-1
    Vec<float> rc, ec; // [3n] restricted residual C_l r, smoothed coarse S_l(C_l r)
    void free_all() {
        A.free();
        agg.free();
        agg_off.free();
        members.free();
        ainv_off.free();
        cnt.free();
        scratch.free();
        map_fine.free();
        ainv.free();
        rc.free();
        ec.free();
        n = n_agg = 0;
    }
};
static constexpr unsigned MAX_LEVELS = 8;

struct SchwarzCache {
    bool valid{false};
    unsigned nrow{0};
    unsigned kmax{0};
    unsigned n_agg{0};
    unsigned max_dim{0};
    uintptr_t bptr{0};
    unsigned long long dyn_hash{0};
    Vec<unsigned> agg;
    Vec<unsigned> agg_off;
    Vec<unsigned> members;
    Vec<unsigned> ainv_off;
    Vec<unsigned> scratch;
    Vec<unsigned> cnt;
    Vec<unsigned> fine_off; // [nrow+1] level-0 matrix-graph CSR offsets
    Vec<unsigned> fine_col; // [fine_cap] concatenated four-loop neighbor columns
    Vec<float> fine_wgt;    // [fine_cap] per-edge coupling-block frob norm (HEM weight)
    unsigned fine_cap{0};
    Vec<unsigned> must_partner; // [nrow] stiffest-contact neighbor per vertex (UNSET=none)
    Vec<float> ainv;        // packed per-aggregate inverse Cholesky factors G = L^-1
    // Multilevel additive Schwarz state (built only when levels >= 2).
    CoarseMat M0;             // materialized level-0 operator (Galerkin source)
    SchwarzLevel lvl[MAX_LEVELS]; // coarse levels 1..nlev-1 (index 0 unused)
    unsigned nlev{1};         // total levels including the fine level 0
    unsigned levels_req{1};   // requested level cap (PPF_SCHWARZ_LEVELS)
    float coarse_weight{1.0f}; // additive coarse-correction damping (PPF_SCHWARZ_WEIGHT)
};
static SchwarzCache K;
static Vec<unsigned long long> d_hash;
static Vec<unsigned> d_forced; // PPF_SCHWARZ_DIAG: count of final-sweep singletons
static Vec<unsigned> d_weight; // per-round unset-degree weights (degree seeding)

// Non-static for Windows nvcc: a function enclosing an extended __device__
// lambda (the DISPATCH macro) must have external linkage. Harmless elsewhere.
unsigned long long compute_dyn_hash(const DynCSRMat &A) {
    if (d_hash.data == nullptr) {
        d_hash = Vec<unsigned long long>::alloc(NACC);
    }
    CUDA_HANDLE_ERROR(
        cudaMemset(d_hash.data, 0, NACC * sizeof(unsigned long long)));
    DynCSRMat Acap = A;
    unsigned long long *hp = d_hash.data;
    DISPATCH_START(A.nrow)
    [Acap, hp] __device__(unsigned i) mutable {
        const Row &row = Acap.rows.data[i];
        unsigned long long s = 0;
        for (unsigned k = 0; k < row.head; ++k) {
            unsigned long long key =
                (((unsigned long long)i) << 32) | (unsigned long long)row.index[k];
            s += mix64(key);
        }
        if (s) {
            atomicAdd(&hp[i & (NACC - 1)], s);
        }
    } DISPATCH_END;
    unsigned long long host[NACC];
    CUDA_HANDLE_ERROR(cudaMemcpy(host, d_hash.data,
                                 NACC * sizeof(unsigned long long),
                                 cudaMemcpyDeviceToHost));
    unsigned long long h = 0;
    for (unsigned t = 0; t < NACC; ++t) {
        h += host[t];
    }
    return h;
}

// Materialize the level-0 matrix-graph adjacency as a both-triangle block-CSR
// (fine_off, fine_col): per row, the concatenation of the four neighbor loops
// apply() walks (L1 dyn upper, L2 dyn lower, L3 fixed upper, L4 fixed lower). The
// greedy aggregation needs neither dedup nor sort (duplicates and order are
// harmless to seed/attach), so a plain concatenation suffices, and it is exactly
// the neighbor set the seed/attach loops visit.
// Non-static for Windows nvcc (extended __device__ lambda needs external linkage).
void build_fine_graph(const DynCSRMat &A, const FixedCSRMat &B) {
    const unsigned nrow = K.nrow;
    Vec<unsigned> fine_off = K.fine_off;
    {
        DynCSRMat Ac = A;
        FixedCSRMat Bc = B;
        DISPATCH_START(nrow)
        [Ac, Bc, fine_off] __device__(unsigned i) mutable {
            const Row &row = Ac.rows.data[i];
            fine_off.data[i] =
                row.head + row.ref_head +
                (Bc.index.offset[i + 1] - Bc.index.offset[i]) +
                (Bc.transpose.offset[i + 1] - Bc.transpose.offset[i]);
        } DISPATCH_END;
    }
    unsigned fine_nnz = kernels::exclusive_scan(fine_off.data, nrow);
    CUDA_HANDLE_ERROR(cudaMemcpy(fine_off.data + nrow, &fine_nnz,
                                 sizeof(unsigned), cudaMemcpyHostToDevice));
    if (fine_nnz > K.fine_cap) {
        unsigned cap = grow_cap(fine_nnz > 0 ? fine_nnz : 1);
        K.fine_col.free();
        K.fine_wgt.free();
        K.fine_col = Vec<unsigned>::alloc(cap);
        K.fine_wgt = Vec<float>::alloc(cap);
        K.fine_cap = cap;
    }
    Vec<unsigned> off = K.fine_off;
    Vec<unsigned> col = K.fine_col;
    Vec<float> wgt = K.fine_wgt;
    {
        DynCSRMat Ac = A;
        FixedCSRMat Bc = B;
        DISPATCH_START(nrow)
        [Ac, Bc, off, col, wgt] __device__(unsigned i) mutable {
            unsigned w = off.data[i];
            const Row &row = Ac.rows.data[i];
            for (unsigned k = 0; k < row.head; ++k) {
                col.data[w] = row.index[k];
                wgt.data[w] = frob9(row.value[k]);
                ++w;
            }
            for (unsigned k = 0; k < row.ref_head; ++k) {
                col.data[w] = row.ref_index[k];
                wgt.data[w] = frob9(Ac.dyn_value_buff.data[row.ref_value[k]]);
                ++w;
            }
            for (unsigned k = Bc.index.offset[i]; k < Bc.index.offset[i + 1];
                 ++k) {
                col.data[w] = Bc.index.data[k];
                wgt.data[w] = frob9(Bc.value.data[k]);
                ++w;
            }
            for (unsigned k = Bc.transpose.offset[i];
                 k < Bc.transpose.offset[i + 1]; ++k) {
                Vec2u ref = Bc.transpose.data[k];
                col.data[w] = ref[0];
                wgt.data[w] = frob9(Bc.value.data[ref[1]]);
                ++w;
            }
        } DISPATCH_END;
    }
}

// Heavy-edge connectivity partition: a balanced graph-partition greedy graph-growing
// alternative to agg_core's parallel independent-set seeding. Grows each domain
// to kmax vertices by repeatedly absorbing the unset neighbor with the largest
// summed coupling weight to the current domain, so the stiffest couplings stay
// INSIDE one dense block instead of being cut across an aggregate boundary (a cut
// stiff block is an uncaptured extreme eigenvalue that stalls CG). Seeds are
// visited in descending incident-weight order (start domains in the stiffest
// regions). Host-side because the fine graph is small and this runs only on
// re-aggregation. Produces the same K.agg + n_agg contract as agg_core (dense
// ids in [0,n_agg), every aggregate <= kmax). PPF_SCHWARZ_HEM.
void hem_partition(unsigned &n_agg_out) {
    const unsigned nrow = K.nrow;
    const unsigned kmax = K.kmax;
    unsigned nnz = 0;
    CUDA_HANDLE_ERROR(cudaMemcpy(&nnz, K.fine_off.data + nrow, sizeof(unsigned),
                                 cudaMemcpyDeviceToHost));
    std::vector<unsigned> off(nrow + 1), col(nnz ? nnz : 1);
    std::vector<float> wgt(nnz ? nnz : 1);
    CUDA_HANDLE_ERROR(cudaMemcpy(off.data(), K.fine_off.data,
                                 (size_t)(nrow + 1) * sizeof(unsigned),
                                 cudaMemcpyDeviceToHost));
    if (nnz) {
        CUDA_HANDLE_ERROR(cudaMemcpy(col.data(), K.fine_col.data,
                                     (size_t)nnz * sizeof(unsigned),
                                     cudaMemcpyDeviceToHost));
        CUDA_HANDLE_ERROR(cudaMemcpy(wgt.data(), K.fine_wgt.data,
                                     (size_t)nnz * sizeof(float),
                                     cudaMemcpyDeviceToHost));
    }
    std::vector<unsigned> agg(nrow, UNSET);
    std::vector<double> incw(nrow, 0.0);
    for (unsigned i = 0; i < nrow; ++i)
        for (unsigned k = off[i]; k < off[i + 1]; ++k)
            if (col[k] != i) incw[i] += wgt[k];
    std::vector<unsigned> order(nrow);
    for (unsigned i = 0; i < nrow; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](unsigned a, unsigned b) { return incw[a] > incw[b]; });
    std::vector<float> gain(nrow, 0.0f);
    std::vector<unsigned> touched;
    touched.reserve(8 * kmax);
    unsigned nid = 0;
    auto absorb = [&](unsigned v) {
        for (unsigned k = off[v]; k < off[v + 1]; ++k) {
            unsigned u = col[k];
            if (u == v || u >= nrow || agg[u] != UNSET) continue;
            if (gain[u] == 0.0f) touched.push_back(u);
            gain[u] += wgt[k];
        }
    };
    for (unsigned s = 0; s < nrow; ++s) {
        unsigned seed = order[s];
        if (agg[seed] != UNSET) continue;
        agg[seed] = nid;
        unsigned size = 1;
        touched.clear();
        absorb(seed);
        while (size < kmax) {
            float best = 0.0f;
            unsigned bv = UNSET;
            for (unsigned t = 0; t < touched.size(); ++t) {
                unsigned v = touched[t];
                if (agg[v] == UNSET && gain[v] > best) {
                    best = gain[v];
                    bv = v;
                }
            }
            if (bv == UNSET) break; // no connected unset neighbor left
            agg[bv] = nid;
            ++size;
            absorb(bv);
        }
        for (unsigned t = 0; t < touched.size(); ++t) gain[touched[t]] = 0.0f;
        ++nid;
    }
    // Fully isolated leftovers (no edges) get their own singleton aggregate.
    for (unsigned i = 0; i < nrow; ++i)
        if (agg[i] == UNSET) agg[i] = nid++;
    CUDA_HANDLE_ERROR(cudaMemcpy(K.agg.data, agg.data(),
                                 (size_t)nrow * sizeof(unsigned),
                                 cudaMemcpyHostToDevice));
    n_agg_out = nid;
}

// Generic parallel connectivity aggregation on a both-triangle block-CSR graph
// (offset, col). PPF_SCHWARZ_ROUNDS rounds of independent-set root seeding + atomic
// bounded try_attach, then a leftover-absorption phase and a singleton sweep.
// Writes agg[] and the aggregate count (returned via n_agg_out, left in
// scratch[0]). cnt is per-aggregate member scratch.
//
// Two refinements over plain id-local-min first-fit seeding (id-ordered roots
// let dense regions pack early aggregates to the full kmax, orphaning DOF as
// forced singletons that then get only block-Jacobi treatment):
//   - Degree-weighted seeding (PPF_SCHWARZ_DEGSEED, default on): a node roots when it
//     has the largest unset-degree among its unset neighbors (mix64/id
//     tiebreak), spreading roots into a maximal independent set keyed by local
//     connectivity rather than raw vertex id. High-degree hubs root first and
//     gather balanced patches, so coverage is uniform and forced singletons drop
//     to zero, reducing cg-iters on dense contact scenes.
//   - An ABSORPTION phase of extra attach sweeps after the rounds mops up any
//     last stragglers. A soft attach cap (ksoft below) is available as a knob
//     but defaults off, since it fragments the balanced patches.
// Any aggregation yields a valid SPD preconditioner (each block is the inverse of
// an SPD principal submatrix, applied as a Gram form), so these only affect
// convergence rate, never correctness.
// Non-static for Windows nvcc (extended __device__ lambda needs external linkage).
void agg_core(const unsigned *offset, const unsigned *col, unsigned nrow,
                     unsigned kmax, Vec<unsigned> agg, Vec<unsigned> cnt,
                     Vec<unsigned> scratch, const unsigned *must_partner,
                     unsigned &n_agg_out) {
    DISPATCH_START(nrow)
    [agg, cnt] __device__(unsigned i) mutable {
        agg.data[i] = UNSET;
        cnt.data[i] = 0;
    } DISPATCH_END;
    CUDA_HANDLE_ERROR(cudaMemset(scratch.data, 0, sizeof(unsigned)));
    if (d_forced.data == nullptr) {
        d_forced = Vec<unsigned>::alloc(1);
    }
    CUDA_HANDLE_ERROR(cudaMemset(d_forced.data, 0, sizeof(unsigned)));

    // Contact-aware pre-seed (PPF_SCHWARZ_CONTACT_AGG via must_partner != null):
    // co-aggregate MUTUAL stiffest-contact pairs into a 2-member aggregate FIRST,
    // so the stiff (1000x) contact coupling enters one A_loc rather than being cut
    // (factor_kernel's find_local drops every inter-aggregate block, leaving a cut
    // stiff coupling as an uncaptured extreme eigenvalue that stalls CG). Race-free:
    // only the lower id of a mutual pair claims the aggregate.
    if (must_partner) {
        Vec<unsigned> aggv = agg;
        Vec<unsigned> cntv = cnt;
        Vec<unsigned> ctr = scratch;
        const unsigned *mp = must_partner;
        DISPATCH_START(nrow)
        [aggv, cntv, ctr, mp] __device__(unsigned i) mutable {
            unsigned j = mp[i];
            if (j == UNSET || j <= i) {
                return;
            }
            if (mp[j] != i) {
                return; // mutual stiffest pairs only (a matching, so race-free)
            }
            unsigned id = atomicAdd(&ctr.data[0], 1u);
            aggv.data[i] = id;
            aggv.data[j] = id;
            cntv.data[id] = 2;
        } DISPATCH_END;
    }

    unsigned rounds = 8;
    if (const char *e = getenv("PPF_SCHWARZ_ROUNDS")) {
        rounds = (unsigned)atoi(e);
    }
    // Soft attach cap during the rounds. Default OFF (== kmax): a soft cap below
    // kmax fragments the balanced patches degree seeding builds, raising
    // singletons and iters, so it is disabled by default and kept only as a knob.
    // The absorption phase below still runs (harmless extra attach sweeps when
    // ksoft == kmax).
    unsigned ksoft = kmax;
    if (const char *e = getenv("PPF_SCHWARZ_KSOFT")) {
        ksoft = (unsigned)atoi(e);
    }
    if (ksoft < 1) {
        ksoft = 1;
    }
    if (ksoft > kmax) {
        ksoft = kmax;
    }
    bool degseed = true;
    if (const char *e = getenv("PPF_SCHWARZ_DEGSEED")) {
        degseed = atoi(e) != 0;
    }
    unsigned absorb = 4;
    if (const char *e = getenv("PPF_SCHWARZ_ABSORB")) {
        absorb = (unsigned)atoi(e);
    }
    if (degseed) {
        if (d_weight.data == nullptr || d_weight.size < nrow) {
            d_weight.free();
            d_weight = Vec<unsigned>::alloc(nrow > 0 ? nrow : 1);
        }
    }
    Vec<unsigned> wv = d_weight;

    for (unsigned round = 0; round < rounds; ++round) {
        if (degseed) { // unset-degree per UNSET node (closed under the round's set)
            Vec<unsigned> aggv = agg;
            DISPATCH_START(nrow)
            [offset, col, aggv, wv] __device__(unsigned i) mutable {
                if (aggv.data[i] != UNSET) {
                    wv.data[i] = 0;
                    return;
                }
                unsigned w = 0;
                for (unsigned k = offset[i]; k < offset[i + 1]; ++k) {
                    unsigned j = col[k];
                    if (j != i && aggv.data[j] == UNSET) {
                        ++w;
                    }
                }
                wv.data[i] = w;
            } DISPATCH_END;
        }
        { // Seed: a node roots when it is the local maximum among UNSET neighbors,
          // by (unset-degree, mix64(id)) when degseed else (-id) [id local min].
            Vec<unsigned> aggv = agg;
            Vec<unsigned> cntv = cnt;
            Vec<unsigned> ctr = scratch;
            const bool ds = degseed;
            DISPATCH_START(nrow)
            [offset, col, aggv, cntv, ctr, wv, ds] __device__(unsigned i) mutable {
                if (aggv.data[i] != UNSET) {
                    return;
                }
                bool is_root = true;
                if (ds) {
                    unsigned long long ki =
                        ((unsigned long long)wv.data[i] << 32) |
                        (mix64(i) & 0xffffffffu);
                    for (unsigned k = offset[i]; k < offset[i + 1] && is_root;
                         ++k) {
                        unsigned j = col[k];
                        if (j == i || aggv.data[j] != UNSET) {
                            continue;
                        }
                        unsigned long long kj =
                            ((unsigned long long)wv.data[j] << 32) |
                            (mix64(j) & 0xffffffffu);
                        if (kj > ki || (kj == ki && j > i)) {
                            is_root = false;
                        }
                    }
                } else {
                    for (unsigned k = offset[i]; k < offset[i + 1] && is_root;
                         ++k) {
                        unsigned j = col[k];
                        if (j != i && j < i && aggv.data[j] == UNSET) {
                            is_root = false;
                        }
                    }
                }
                if (is_root) {
                    unsigned id = atomicAdd(&ctr.data[0], 1u);
                    aggv.data[i] = id;
                    cntv.data[id] = 1;
                }
            } DISPATCH_END;
        }
        { // Attach: each UNSET node joins the first neighbor aggregate with room
          // under the SOFT cap, keeping aggregate sizes balanced. With
          // contact-aware aggregation, try the stiffest-contact partner's aggregate
          // FIRST so the stiff coupling is captured intra-aggregate.
            Vec<unsigned> aggv = agg;
            Vec<unsigned> cntv = cnt;
            const unsigned ks = ksoft;
            const unsigned *mp = must_partner;
            DISPATCH_START(nrow)
            [offset, col, aggv, cntv, ks, mp] __device__(unsigned i) mutable {
                if (aggv.data[i] != UNSET) {
                    return;
                }
                if (mp && mp[i] != UNSET && try_attach(aggv, cntv, i, mp[i], ks)) {
                    return;
                }
                for (unsigned k = offset[i]; k < offset[i + 1]; ++k) {
                    if (try_attach(aggv, cntv, i, col[k], ks)) {
                        return;
                    }
                }
            } DISPATCH_END;
        }
    }
    // Absorption: attach remaining leftovers to any neighbor aggregate with room
    // up to the HARD kmax (the soft cap left headroom). Several sweeps because a
    // leftover may only become attachable after an adjacent leftover joins first.
    for (unsigned a = 0; a < absorb; ++a) {
        Vec<unsigned> aggv = agg;
        Vec<unsigned> cntv = cnt;
        const unsigned km = kmax;
        DISPATCH_START(nrow)
        [offset, col, aggv, cntv, km] __device__(unsigned i) mutable {
            if (aggv.data[i] != UNSET) {
                return;
            }
            for (unsigned k = offset[i]; k < offset[i + 1]; ++k) {
                if (try_attach(aggv, cntv, i, col[k], km)) {
                    return;
                }
            }
        } DISPATCH_END;
    }
    { // Any node still unassigned (whole neighborhood full) becomes a singleton.
        Vec<unsigned> aggv = agg;
        Vec<unsigned> ctr = scratch;
        Vec<unsigned> fc = d_forced;
        DISPATCH_START(nrow)
        [aggv, ctr, fc] __device__(unsigned i) mutable {
            if (aggv.data[i] == UNSET) {
                aggv.data[i] = atomicAdd(&ctr.data[0], 1u);
                atomicAdd(&fc.data[0], 1u);
            }
        } DISPATCH_END;
    }
    unsigned n_agg = 0;
    CUDA_HANDLE_ERROR(cudaMemcpy(&n_agg, scratch.data, sizeof(unsigned),
                                 cudaMemcpyDeviceToHost));
    n_agg_out = n_agg;
}

// For each vertex i, the stiffest neighbor j whose coupling block Frobenius norm
// exceeds 100x the C[i] diagonal block (a stiff contact), else UNSET. Used by
// contact-aware aggregation to must-link stiff contact pairs into one aggregate.
// Non-static for Windows nvcc (extended __device__ lambda needs external linkage).
void compute_must_partner(const DynCSRMat &A, const FixedCSRMat &B,
                          const Vec<Mat3x3f> &C) {
    const unsigned nrow = K.nrow;
    ensure(K.must_partner, nrow);
    DynCSRMat Ac = A;
    FixedCSRMat Bc = B;
    Vec<Mat3x3f> Cc = C;
    Vec<unsigned> mp = K.must_partner;
    DISPATCH_START(nrow)
    [Ac, Bc, Cc, mp] __device__(unsigned i) mutable {
        float best = 100.0f * frob9(Cc.data[i]);
        unsigned bestj = UNSET;
        const Row &row = Ac.rows.data[i];
        for (unsigned k = 0; k < row.head; ++k) {
            unsigned j = row.index[k];
            if (j != i) {
                float bn = frob9(row.value[k]);
                if (bn > best) { best = bn; bestj = j; }
            }
        }
        for (unsigned k = 0; k < row.ref_head; ++k) {
            unsigned j = row.ref_index[k];
            if (j != i) {
                float bn = frob9(Ac.dyn_value_buff.data[row.ref_value[k]]);
                if (bn > best) { best = bn; bestj = j; }
            }
        }
        for (unsigned k = Bc.index.offset[i]; k < Bc.index.offset[i + 1]; ++k) {
            unsigned j = Bc.index.data[k];
            if (j != i) {
                float bn = frob9(Bc.value.data[k]);
                if (bn > best) { best = bn; bestj = j; }
            }
        }
        for (unsigned k = Bc.transpose.offset[i]; k < Bc.transpose.offset[i + 1];
             ++k) {
            Vec2u ref = Bc.transpose.data[k];
            unsigned j = ref[0];
            if (j != i) {
                float bn = frob9(Bc.value.data[ref[1]]);
                if (bn > best) { best = bn; bestj = j; }
            }
        }
        mp.data[i] = bestj;
    } DISPATCH_END;
}

// Aggregation: materialize the matrix graph from the four live loops, run the
// generic agg_core on it, then build the member CSR + per-aggregate factor offsets.
// Non-static for Windows nvcc (extended __device__ lambda needs external linkage).
void recompute_aggregation(const DynCSRMat &A, const FixedCSRMat &B,
                           const Vec<Mat3x3f> &C) {
    const unsigned nrow = K.nrow;
    const unsigned kmax = K.kmax;
    if (getenv("PPF_SCHWARZ_DIAG")) {
        static int s_reagg = 0;
        fprintf(stderr, "[schwarz] reaggregation #%d\n", ++s_reagg);
    }
    build_fine_graph(A, B);
    unsigned n_agg = 0;
    if (getenv("PPF_SCHWARZ_GREEDY")) {
        // Opt out of the connectivity partition: use the parallel independent-set
        // agg_core (connectivity-blind) instead. Default off.
        const unsigned *mp = nullptr;
        if (getenv("PPF_SCHWARZ_CONTACT_AGG")) {
            compute_must_partner(A, B, C);
            mp = K.must_partner.data;
        }
        agg_core(K.fine_off.data, K.fine_col.data, nrow, kmax, K.agg, K.cnt,
                 K.scratch, mp, n_agg);
    } else {
        // Heavy-edge connectivity partition (default): keeps the stiffest
        // couplings inside each dense block so the two-level coarsening is
        // effective (without it, a second level over connectivity-blind
        // aggregates over-corrects and loses to single-level).
        hem_partition(n_agg);
    }
    K.n_agg = n_agg;

    // Member counts -> CSR offsets.
    Vec<unsigned> agg_off = K.agg_off;
    DISPATCH_START(n_agg)
    [agg_off] __device__(unsigned g) mutable { agg_off.data[g] = 0; } DISPATCH_END;
    {
        Vec<unsigned> aggv = K.agg;
        Vec<unsigned> off = K.agg_off;
        DISPATCH_START(nrow)
        [aggv, off] __device__(unsigned i) mutable {
            atomicAdd(&off.data[aggv.data[i]], 1u);
        } DISPATCH_END;
    }
    unsigned total = kernels::exclusive_scan(agg_off.data, n_agg);
    CUDA_HANDLE_ERROR(cudaMemcpy(agg_off.data + n_agg, &total, sizeof(unsigned),
                                 cudaMemcpyHostToDevice));

    // Scatter members grouped by aggregate (cursor reuses scratch).
    Vec<unsigned> cursor = K.scratch;
    DISPATCH_START(n_agg)
    [cursor] __device__(unsigned g) mutable { cursor.data[g] = 0; } DISPATCH_END;
    {
        Vec<unsigned> aggv = K.agg;
        Vec<unsigned> off = K.agg_off;
        Vec<unsigned> members = K.members;
        Vec<unsigned> cur = K.scratch;
        DISPATCH_START(nrow)
        [aggv, off, members, cur] __device__(unsigned i) mutable {
            unsigned g = aggv.data[i];
            unsigned pos = atomicAdd(&cur.data[g], 1u);
            members.data[off.data[g] + pos] = i;
        } DISPATCH_END;
    }

    // Per-aggregate dense-inverse float offsets (d*d, d = 3*members).
    Vec<unsigned> ainv_off = K.ainv_off;
    {
        Vec<unsigned> off = K.agg_off;
        DISPATCH_START(n_agg)
        [ainv_off, off] __device__(unsigned g) mutable {
            unsigned d = 3u * (off.data[g + 1] - off.data[g]);
            ainv_off.data[g] = tri_size(d); // packed lower triangle of A^-1
        } DISPATCH_END;
    }
    unsigned ainv_total = kernels::exclusive_scan(ainv_off.data, n_agg);
    CUDA_HANDLE_ERROR(cudaMemcpy(ainv_off.data + n_agg, &ainv_total,
                                 sizeof(unsigned), cudaMemcpyHostToDevice));

    // Aggregation-quality diagnostics (PPF_SCHWARZ_DIAG): aggregate-size histogram +
    // forced-singleton count (nodes the rounds failed to place, assigned by the
    // final sweep). A heavy singleton/tiny-aggregate tail weakens the Schwarz
    // coupling (those vertices fall back to block-Jacobi), so this is the signal
    // for tuning root selection. Diagnostic-only; no solver-path effect.
    if (getenv("PPF_SCHWARZ_DIAG")) {
        static Vec<unsigned> d_hist;
        if (d_hist.data == nullptr || d_hist.size < kmax + 1) {
            d_hist.free();
            d_hist = Vec<unsigned>::alloc(kmax + 1);
        }
        CUDA_HANDLE_ERROR(
            cudaMemset(d_hist.data, 0, (kmax + 1) * sizeof(unsigned)));
        Vec<unsigned> off = K.agg_off;
        Vec<unsigned> hist = d_hist;
        const unsigned km = kmax;
        DISPATCH_START(n_agg)
        [off, hist, km] __device__(unsigned g) mutable {
            unsigned sz = off.data[g + 1] - off.data[g];
            if (sz > km) {
                sz = km;
            }
            if (sz > 0) {
                atomicAdd(&hist.data[sz], 1u);
            }
        } DISPATCH_END;
        std::vector<unsigned> hh(kmax + 1, 0u);
        CUDA_HANDLE_ERROR(cudaMemcpy(hh.data(), d_hist.data,
                                     (kmax + 1) * sizeof(unsigned),
                                     cudaMemcpyDeviceToHost));
        unsigned forced = 0;
        if (d_forced.data) {
            CUDA_HANDLE_ERROR(cudaMemcpy(&forced, d_forced.data, sizeof(unsigned),
                                         cudaMemcpyDeviceToHost));
        }
        fprintf(stderr,
                "[schwarz agg] nrow=%u n_agg=%u coarsen=%.2fx forced_singletons=%u "
                "| sizes",
                nrow, n_agg, n_agg ? (double)nrow / (double)n_agg : 0.0, forced);
        for (unsigned s = 1; s <= kmax; ++s) {
            if (hh[s]) {
                fprintf(stderr, " %u:%u", s, hh[s]);
            }
        }
        fprintf(stderr, "\n");
    }
}

// ---------------------------------------------------------------------------
// Multilevel additive Schwarz coarse machinery. Built only
// when PPF_SCHWARZ_LEVELS >= 2; the level-0 path above is untouched, so
// PPF_SCHWARZ_LEVELS == 1 is bit-for-bit the single-level smoother. The coarse
// preconditioner is z = S_0 r + sum_{l>=1} C_l^T S_l (C_l r): every level reuses
// the SAME dense Cholesky / Gram block-Schwarz (so SPD in float32 by the same
// r.z = ||G r||^2 >= 0 argument), run on a Galerkin-coarsened operator. No fine
// matvec and no exact coarse solve appear in the apply, so it stays purely
// additive and SPD.
// ---------------------------------------------------------------------------

// One CUDA block per coarse domain. Identical to factor_kernel (floor, cooperative
// Cholesky A_loc = L L^T, triangular inverse G = L^-1 stored packed lower-tri), but
// the four-loop gather is replaced by one walk of the CoarseMat block-CSR (which
// already holds both triangles, so A_loc is symmetric and there is no separate C
// diagonal to add). The Gram apply z = G^T (G r) then keeps each coarse level SPD.
__global__ void factor_kernel_coarse(CoarseMat A, Vec<unsigned> agg_off,
                                     Vec<unsigned> members,
                                     Vec<unsigned> ainv_off, Vec<float> ainv,
                                     unsigned max_dim) {
    extern __shared__ float smem[];
    float *Aloc = smem;                            // max_dim*max_dim
    float *Inv = smem + (size_t)max_dim * max_dim; // max_dim*max_dim
    float *fcol = Inv + (size_t)max_dim * max_dim; // max_dim (unused; layout parity)
    unsigned *mem = (unsigned *)(fcol + max_dim);

    const unsigned g = blockIdx.x;
    const unsigned base = agg_off.data[g];
    const unsigned m = agg_off.data[g + 1] - base;
    const unsigned d = 3u * m;
    const unsigned tid = threadIdx.x;
    const unsigned nthreads = blockDim.x;

    for (unsigned a = tid; a < m; a += nthreads) {
        mem[a] = members.data[base + a];
    }
    for (unsigned e = tid; e < d * d; e += nthreads) {
        Aloc[e] = 0.0f;
    }
    __syncthreads();

    // Gather: thread a owns member row a, race-free. One CSR walk; keep
    // intra-domain columns. Mat3x3f is col-major, so blk(r,s) is the (r,s) entry.
    for (unsigned a = tid; a < m; a += nthreads) {
        const unsigned i = mem[a];
        for (unsigned k = A.offset.data[i]; k < A.offset.data[i + 1]; ++k) {
            int b = find_local(mem, m, A.col.data[k]);
            if (b >= 0) {
                const Mat3x3f &blk = A.value.data[k];
                for (int r = 0; r < 3; ++r)
                    for (int s = 0; s < 3; ++s)
                        Aloc[(3 * a + r) * d + (3 * b + s)] += blk(r, s);
            }
        }
    }
    __syncthreads();

    (void)fcol;
    __shared__ float s_floor;
    if (tid == 0) {
        float tr = 0.0f;
        for (unsigned k = 0; k < d; ++k) {
            tr += Aloc[k * d + k];
        }
        const float fl = 1e-6f * (tr / (float)d) + 1e-8f;
        for (unsigned k = 0; k < d; ++k) {
            Aloc[k * d + k] += fl;
        }
        s_floor = fl;
    }
    __syncthreads();

    for (unsigned j = 0; j < d; ++j) {
        if (tid == 0) {
            float s = Aloc[j * d + j];
            for (unsigned k = 0; k < j; ++k) {
                const float ljk = Aloc[j * d + k];
                s -= ljk * ljk;
            }
            Aloc[j * d + j] = sqrtf(s > s_floor ? s : s_floor);
        }
        __syncthreads();
        const float inv_ljj = 1.0f / Aloc[j * d + j];
        for (unsigned i = j + 1 + tid; i < d; i += nthreads) {
            float s = Aloc[i * d + j];
            for (unsigned k = 0; k < j; ++k) {
                s -= Aloc[i * d + k] * Aloc[j * d + k];
            }
            Aloc[i * d + j] = s * inv_ljj;
        }
        __syncthreads();
    }

    for (unsigned col = tid; col < d; col += nthreads) {
        Inv[col * d + col] = 1.0f / Aloc[col * d + col];
        for (unsigned i = col + 1; i < d; ++i) {
            float s = 0.0f;
            for (unsigned k = col; k < i; ++k) {
                s += Aloc[i * d + k] * Inv[k * d + col];
            }
            Inv[i * d + col] = -s / Aloc[i * d + i];
        }
    }
    __syncthreads();

    const unsigned abase = ainv_off.data[g];
    for (unsigned e = tid; e < d * d; e += nthreads) {
        const unsigned i = e / d, j = e % d;
        if (i >= j) {
            ainv.data[abase + tri_idx(i, j)] = Inv[e];
        }
    }
}

// Materialize the level-0 operator M = A_dyn + B_fixed + C_diag as a both-triangle
// block-CSR (the Galerkin source for level 1). No dedup: each four-loop
// contribution is a separate oriented entry, plus one (i,i)=C[i] per row; the
// Galerkin sum collapses duplicates.
// Non-static for Windows nvcc (extended __device__ lambda needs external linkage).
void materialize_level0(const DynCSRMat &A, const FixedCSRMat &B,
                        const Vec<Mat3x3f> &C, CoarseMat &M0) {
    const unsigned nrow = A.nrow;
    M0.n = nrow;
    if (M0.offset.size < nrow + 1) {
        M0.offset.free();
        M0.offset = Vec<unsigned>::alloc(nrow + 1);
    }
    Vec<unsigned> off = M0.offset;
    {
        DynCSRMat Ac = A;
        FixedCSRMat Bc = B;
        DISPATCH_START(nrow)
        [Ac, Bc, off] __device__(unsigned i) mutable {
            const Row &row = Ac.rows.data[i];
            off.data[i] = row.head + row.ref_head +
                          (Bc.index.offset[i + 1] - Bc.index.offset[i]) +
                          (Bc.transpose.offset[i + 1] - Bc.transpose.offset[i]) +
                          1u; // + the C[i] diagonal entry
        } DISPATCH_END;
    }
    unsigned nnz = kernels::exclusive_scan(off.data, nrow);
    CUDA_HANDLE_ERROR(cudaMemcpy(off.data + nrow, &nnz, sizeof(unsigned),
                                 cudaMemcpyHostToDevice));
    M0.nnz = nnz;
    if (nnz > M0.cap) {
        unsigned cap = grow_cap(nnz > 0 ? nnz : 1);
        M0.col.free();
        M0.value.free();
        M0.col = Vec<unsigned>::alloc(cap);
        M0.value = Vec<Mat3x3f>::alloc(cap);
        M0.cap = cap;
    }
    Vec<unsigned> col = M0.col;
    Vec<Mat3x3f> val = M0.value;
    Vec<Mat3x3f> Cc = C;
    {
        DynCSRMat Ac = A;
        FixedCSRMat Bc = B;
        DISPATCH_START(nrow)
        [Ac, Bc, Cc, off, col, val] __device__(unsigned i) mutable {
            unsigned w = off.data[i];
            const Row &row = Ac.rows.data[i];
            for (unsigned k = 0; k < row.head; ++k) {
                col.data[w] = row.index[k];
                val.data[w] = row.value[k];
                ++w;
            }
            for (unsigned k = 0; k < row.ref_head; ++k) {
                col.data[w] = row.ref_index[k];
                val.data[w] = transpose3(Ac.dyn_value_buff.data[row.ref_value[k]]);
                ++w;
            }
            for (unsigned k = Bc.index.offset[i]; k < Bc.index.offset[i + 1]; ++k) {
                col.data[w] = Bc.index.data[k];
                val.data[w] = Bc.value.data[k];
                ++w;
            }
            for (unsigned k = Bc.transpose.offset[i];
                 k < Bc.transpose.offset[i + 1]; ++k) {
                Vec2u ref = Bc.transpose.data[k];
                col.data[w] = ref[0];
                val.data[w] = transpose3(Bc.value.data[ref[1]]);
                ++w;
            }
            col.data[w] = i;
            val.data[w] = Cc.data[i];
            ++w;
        } DISPATCH_END;
    }
}

// Galerkin coarse operator with translation injection: dst = P^T src P, i.e.
// dst(g,h) = sum_{agg[i]==g, agg[j]==h} src(i,j). Assembled COO->CSR by radix
// sorting packed (g,h) keys (g*n_agg+h fits 32 bits for our sizes), then a
// per-edge segmented float32 sum (the sort makes each coarse edge's source entries
// contiguous, so no atomics). Generic over any source CoarseMat, so it serves
// every level. agg maps src nodes (0..src.n) to coarse nodes (0..n_agg).
// Non-static for Windows nvcc (extended __device__ lambda needs external linkage).
void galerkin(const CoarseMat &src, const unsigned *agg, unsigned n_agg,
              CoarseMat &dst) {
    const unsigned nnz = src.nnz;
    buffer::MemoryPool &pool = buffer::get();
    auto key = pool.get<unsigned>(nnz);
    auto perm = pool.get<unsigned>(nnz);
    auto tkey = pool.get<unsigned>(nnz);
    auto tperm = pool.get<unsigned>(nnz);
    Vec<unsigned> keyv = key.as_vec();
    Vec<unsigned> permv = perm.as_vec();

    {
        Vec<unsigned> off = src.offset;
        Vec<unsigned> col = src.col;
        const unsigned na = n_agg;
        DISPATCH_START(src.n)
        [off, col, keyv, permv, agg, na] __device__(unsigned i) mutable {
            unsigned g = agg[i];
            for (unsigned k = off.data[i]; k < off.data[i + 1]; ++k) {
                keyv.data[k] = g * na + agg[col.data[k]];
                permv.data[k] = k;
            }
        } DISPATCH_END;
    }

    kernels::radix_sort_pairs(keyv.data, permv.data, nnz, tkey.as_vec().data,
                              tperm.as_vec().data);

    auto edge = pool.get<unsigned>(nnz);
    Vec<unsigned> eid = edge.as_vec();
    DISPATCH_START(nnz)
    [keyv, eid] __device__(unsigned s) mutable {
        eid.data[s] = (s == 0 || keyv.data[s] != keyv.data[s - 1]) ? 1u : 0u;
    } DISPATCH_END;
    unsigned nnz1 = kernels::exclusive_scan(eid.data, nnz);

    dst.n = n_agg;
    dst.nnz = nnz1;
    if (dst.offset.size < n_agg + 1) {
        dst.offset.free();
        dst.offset = Vec<unsigned>::alloc(grow_cap(n_agg + 1));
    }
    if (nnz1 > dst.cap) {
        unsigned cap = grow_cap(nnz1 > 0 ? nnz1 : 1);
        dst.col.free();
        dst.value.free();
        dst.col = Vec<unsigned>::alloc(cap);
        dst.value = Vec<Mat3x3f>::alloc(cap);
        dst.cap = cap;
    }

    Vec<unsigned> doff = dst.offset;
    Vec<unsigned> dcol = dst.col;
    auto estart = pool.get<unsigned>(nnz1 + 1);
    Vec<unsigned> es = estart.as_vec();
    DISPATCH_START(n_agg)
    [doff] __device__(unsigned g) mutable { doff.data[g] = 0; } DISPATCH_END;
    {
        const unsigned na = n_agg;
        DISPATCH_START(nnz)
        [keyv, eid, dcol, doff, es, na] __device__(unsigned s) mutable {
            bool start = (s == 0) || (keyv.data[s] != keyv.data[s - 1]);
            if (start) {
                unsigned e = eid.data[s];
                dcol.data[e] = keyv.data[s] % na;
                es.data[e] = s;
                atomicAdd(&doff.data[keyv.data[s] / na], 1u);
            }
        } DISPATCH_END;
    }
    CUDA_HANDLE_ERROR(cudaMemcpy(es.data + nnz1, &nnz, sizeof(unsigned),
                                 cudaMemcpyHostToDevice));
    unsigned tot = kernels::exclusive_scan(doff.data, n_agg);
    CUDA_HANDLE_ERROR(cudaMemcpy(doff.data + n_agg, &tot, sizeof(unsigned),
                                 cudaMemcpyHostToDevice));

    Vec<Mat3x3f> dvalue = dst.value;
    {
        Vec<unsigned> permd = permv;
        Vec<Mat3x3f> sval = src.value;
        DISPATCH_START(nnz1)
        [es, permd, sval, dvalue] __device__(unsigned e) mutable {
            float acc[9];
            for (int t = 0; t < 9; ++t) {
                acc[t] = 0.0f;
            }
            for (unsigned s = es.data[e]; s < es.data[e + 1]; ++s) {
                const float *m =
                    reinterpret_cast<const float *>(&sval.data[permd.data[s]]);
                for (int t = 0; t < 9; ++t) {
                    acc[t] += m[t];
                }
            }
            float *o = reinterpret_cast<float *>(&dvalue.data[e]);
            for (int t = 0; t < 9; ++t) {
                o[t] = acc[t];
            }
        } DISPATCH_END;
    }
}

// From an aggregation agg[0..n) (ids in [0,n_agg)), build the domain member CSR
// (agg_off,members) and the per-domain packed-inverse float offsets (ainv_off).
// scratch is an [>=n_agg] cursor. Returns the total ainv float count.
// Non-static for Windows nvcc (extended __device__ lambda needs external linkage).
unsigned build_domains(Vec<unsigned> agg, unsigned n, unsigned n_agg,
                       Vec<unsigned> agg_off, Vec<unsigned> members,
                       Vec<unsigned> ainv_off, Vec<unsigned> scratch) {
    DISPATCH_START(n_agg)
    [agg_off] __device__(unsigned g) mutable { agg_off.data[g] = 0; } DISPATCH_END;
    {
        Vec<unsigned> aggv = agg, off = agg_off;
        DISPATCH_START(n)
        [aggv, off] __device__(unsigned i) mutable {
            atomicAdd(&off.data[aggv.data[i]], 1u);
        } DISPATCH_END;
    }
    unsigned total = kernels::exclusive_scan(agg_off.data, n_agg);
    CUDA_HANDLE_ERROR(cudaMemcpy(agg_off.data + n_agg, &total, sizeof(unsigned),
                                 cudaMemcpyHostToDevice));
    DISPATCH_START(n_agg)
    [scratch] __device__(unsigned g) mutable { scratch.data[g] = 0; } DISPATCH_END;
    {
        Vec<unsigned> aggv = agg, off = agg_off, mem = members, cur = scratch;
        DISPATCH_START(n)
        [aggv, off, mem, cur] __device__(unsigned i) mutable {
            unsigned g = aggv.data[i];
            unsigned pos = atomicAdd(&cur.data[g], 1u);
            mem.data[off.data[g] + pos] = i;
        } DISPATCH_END;
    }
    {
        Vec<unsigned> off = agg_off, ai = ainv_off;
        DISPATCH_START(n_agg)
        [off, ai] __device__(unsigned g) mutable {
            unsigned d = 3u * (off.data[g + 1] - off.data[g]);
            ai.data[g] = tri_size(d);
        } DISPATCH_END;
    }
    unsigned ainv_total = kernels::exclusive_scan(ainv_off.data, n_agg);
    CUDA_HANDLE_ERROR(cudaMemcpy(ainv_off.data + n_agg, &ainv_total,
                                 sizeof(unsigned), cudaMemcpyHostToDevice));
    return ainv_total;
}

// Restrict the fine residual into level l: rc[g] = sum_{i: map[i]==g} x[i]
// (piecewise-constant restriction C_l). zeroes rc[0..3*n_l) first.
// Non-static for Windows nvcc (extended __device__ lambda needs external linkage).
void restrict_direct(const Vec<unsigned> &map_fine, const Vec<float> &x,
                     Vec<float> &rc, unsigned nrow, unsigned n_l,
                     cudaStream_t stream = 0) {
    Vec<float> rcv = rc;
    DISPATCH_QUEUE_START(3u * n_l, stream)
    [rcv] __device__(unsigned i) mutable { rcv.data[i] = 0.0f; } DISPATCH_QUEUE_END;
    Vec<unsigned> mp = map_fine;
    Vec<float> xv = x;
    DISPATCH_QUEUE_START(nrow, stream)
    [mp, xv, rcv] __device__(unsigned i) mutable {
        unsigned g = mp.data[i];
        atomicAdd(&rcv.data[3u * g + 0], xv.data[3u * i + 0]);
        atomicAdd(&rcv.data[3u * g + 1], xv.data[3u * i + 1]);
        atomicAdd(&rcv.data[3u * g + 2], xv.data[3u * i + 2]);
    } DISPATCH_QUEUE_END;
}

// Prolong-correct z += w * C_l^T ec: z[i] += w * ec[map[i]] (translation
// injection, with a per-level additive damping weight w to curb the low-frequency
// double-counting between the smoother and the coarse correction; w == 1 is the
// undamped form, PPF_SCHWARZ_WEIGHT overrides).
// Non-static for Windows nvcc (extended __device__ lambda needs external linkage).
void prolong_add_direct(const Vec<unsigned> &map_fine, const Vec<float> &ec,
                        Vec<float> &z, unsigned nrow, float w,
                        cudaStream_t stream = 0) {
    Vec<unsigned> mp = map_fine;
    Vec<float> ecv = ec, zv = z;
    DISPATCH_QUEUE_START(nrow, stream)
    [mp, ecv, zv, w] __device__(unsigned i) mutable {
        unsigned g = mp.data[i];
        zv.data[3u * i + 0] += w * ecv.data[3u * g + 0];
        zv.data[3u * i + 1] += w * ecv.data[3u * g + 1];
        zv.data[3u * i + 2] += w * ecv.data[3u * g + 2];
    } DISPATCH_QUEUE_END;
}

// Build the coarse levels 1..L on top of the cached level-0 fine aggregation: per
// level, Galerkin-coarsen the operator, aggregate its nodes into the next level's
// supernodes (= this level's smoother domains), factor the per-domain dense
// inverse factors, and compose the direct fine->level map. Stops when a level fits
// a single domain, coarsening stalls, or the level cap is hit. Rebuilt every solve
// (values change); cheap because node counts shrink ~kmax x per level.
// Non-static for Windows nvcc (extended __device__ lambda needs external linkage).
void build_mas(const DynCSRMat &A, const FixedCSRMat &B, const Vec<Mat3x3f> &C,
               unsigned max_total_levels, unsigned kmax) {
    const unsigned nrow = K.nrow;
    const unsigned md = 3u * kmax;
    const size_t fshmem =
        (2u * (size_t)md * md + md) * sizeof(float) + (size_t)kmax * sizeof(unsigned);
    // Opt into only the shared memory the coarse factor needs, checked, so a
    // device whose opt-in cap is below the request cannot leak the error to a
    // later launch's cudaGetLastError(). kmax is clamped to the device in
    // build() so fshmem fits. (md == 3*kmax, same chunk size as the fine factor.)
    static int s_coarse_shmem = 48 * 1024;
    ensure_dyn_shmem((const void *)factor_kernel_coarse, fshmem, s_coarse_shmem);

    materialize_level0(A, B, C, K.M0);

    CoarseMat *srcA = &K.M0;
    Vec<unsigned> srcAgg = K.agg; // agg_(0): fine -> level 1
    unsigned cur_n = K.n_agg;     // nodes at level 1
    unsigned L = 0;               // highest coarse level built

    for (unsigned lvl = 1; lvl < MAX_LEVELS && lvl < max_total_levels; ++lvl) {
        SchwarzLevel &lev = K.lvl[lvl];
        lev.n = cur_n;
        lev.max_dim = md;
        galerkin(*srcA, srcAgg.data, cur_n, lev.A); // A_(lvl) = C A_(lvl-1) C^T
        ensure(lev.agg, cur_n);
        ensure(lev.cnt, cur_n);
        ensure(lev.scratch, cur_n);
        ensure(lev.agg_off, cur_n + 1);
        ensure(lev.members, cur_n);
        ensure(lev.ainv_off, cur_n + 1);
        ensure(lev.map_fine, nrow);
        ensure(lev.rc, 3u * cur_n);
        ensure(lev.ec, 3u * cur_n);

        unsigned nnext = 0;
        agg_core(lev.A.offset.data, lev.A.col.data, cur_n, kmax, lev.agg, lev.cnt,
                 lev.scratch, nullptr, nnext);
        lev.n_agg = nnext;
        unsigned ainv_total = build_domains(lev.agg, cur_n, nnext, lev.agg_off,
                                            lev.members, lev.ainv_off, lev.scratch);
        ensure(lev.ainv, ainv_total > 0 ? ainv_total : 1);
        factor_kernel_coarse<<<nnext, BLOCK_THREADS, fshmem>>>(
            lev.A, lev.agg_off, lev.members, lev.ainv_off, lev.ainv, md);
        CUDA_HANDLE_ERROR(cudaGetLastError());

        // map_fine[i] = level-lvl node of fine node i.
        if (lvl == 1) {
            Vec<unsigned> dst = lev.map_fine, src = K.agg;
            DISPATCH_START(nrow)
            [dst, src] __device__(unsigned i) mutable {
                dst.data[i] = src.data[i];
            } DISPATCH_END;
        } else {
            Vec<unsigned> dst = lev.map_fine, pm = K.lvl[lvl - 1].map_fine,
                          pa = K.lvl[lvl - 1].agg;
            DISPATCH_START(nrow)
            [dst, pm, pa] __device__(unsigned i) mutable {
                dst.data[i] = pa.data[pm.data[i]];
            } DISPATCH_END;
        }

        L = lvl;
        if (cur_n <= kmax) {
            break; // top level is a single domain (exact-ish local solve)
        }
        if (nnext >= cur_n) {
            break; // coarsening stalled
        }
        srcA = &lev.A;
        srcAgg = lev.agg;
        cur_n = nnext;
    }
    K.nlev = L + 1;

    if (getenv("PPF_SCHWARZ_DIAG")) {
        fprintf(stderr, "[schwarz mas] nlev=%u | L0 nrow=%u n_agg=%u", K.nlev,
                nrow, K.n_agg);
        for (unsigned l = 1; l < K.nlev; ++l) {
            fprintf(stderr, " | L%u n=%u n_agg=%u nnz=%u ratio=%.2f", l,
                    K.lvl[l].n, K.lvl[l].n_agg, K.lvl[l].A.nnz,
                    K.lvl[l].n_agg ? (double)K.lvl[l].n / K.lvl[l].n_agg : 0.0);
        }
        fprintf(stderr, "\n");
    }
}

// Apply the preconditioner: result = M_schwarz^-1 x, one additive aggregate-
// Schwarz sweep. Each block computes z_seg = A_loc^-1 r_seg = G^T (G r_seg), with
// G the stored inverse Cholesky factor, so result is SPD by construction. For
// nlev >= 2 the coarse levels add their corrections additively (multilevel additive Schwarz):
// z += C_l^T S_l (C_l x), still SPD, no fine matvec, no exact coarse solve.
void apply(const SchwarzHierarchy &H, const Vec<float> &x, Vec<float> &result,
           cudaStream_t stream) {
    if (H.n_agg == 0) {
        return;
    }
    // T (packed G = L^-1) + rloc + yloc + members.
    const size_t shmem =
        (size_t)(tri_size(H.max_dim) + 2u * H.max_dim) * sizeof(float) +
        (size_t)H.kmax * sizeof(unsigned);
    apply_kernel<<<H.n_agg, BLOCK_THREADS, shmem, stream>>>(
        H.agg_off, H.members, H.ainv_off, H.ainv, x, result, H.max_dim);
    CUDA_HANDLE_ERROR(cudaGetLastError());

    // Coarse additive corrections (multilevel additive Schwarz). Per level: restrict the fine
    // residual, one Gram block-Schwarz sweep over the coarse domains, prolong-add.
    for (unsigned l = 1; l < H.nlev; ++l) {
        SchwarzLevel &lev = K.lvl[l];
        if (lev.n_agg == 0) {
            continue;
        }
        restrict_direct(lev.map_fine, x, lev.rc, H.nrow, lev.n, stream);
        const size_t cshmem =
            (size_t)(tri_size(lev.max_dim) + 2u * lev.max_dim) * sizeof(float) +
            (size_t)H.kmax * sizeof(unsigned);
        apply_kernel<<<lev.n_agg, BLOCK_THREADS, cshmem, stream>>>(
            lev.agg_off, lev.members, lev.ainv_off, lev.ainv, lev.rc, lev.ec,
            lev.max_dim);
        CUDA_HANDLE_ERROR(cudaGetLastError());
        prolong_add_direct(lev.map_fine, lev.ec, result, H.nrow, K.coarse_weight,
                           stream);
    }
}

// Build the single-level additive aggregate-Schwarz hierarchy from the assembled
// split system, just before each PCG solve. Aggregation is cached and reused
// while the matrix graph is unchanged; the per-aggregate factors are rebuilt
// every solve from the live values.
void build(SchwarzHierarchy &H, const DynCSRMat &A, const FixedCSRMat &B,
           const Vec<Mat3x3f> &C, const Vec<Vec3f> &positions,
           unsigned levels_param) {
    (void)positions; // aggregation is matrix-graph based, not geometric.
    const unsigned nrow = A.nrow;
    // Schwarz dense-block cap in vertices. Default 16: the connectivity (heavy-edge)
    // partition densifies each subdomain, so the smaller, cheaper 48x48 dense block
    // converges as well as a connectivity-blind 32 while costing less to factor and
    // apply. PPF_SCHWARZ_KMAX overrides it.
    unsigned kmax_env = 16;
    if (const char *e = getenv("PPF_SCHWARZ_KMAX")) {
        int v = atoi(e);
        if (v > 0) {
            kmax_env = (unsigned)v;
        }
    }
    unsigned kmax = kmax_env > 0 ? kmax_env : 1;
    // Reduce the dense-block chunk size until the per-aggregate factor's dynamic
    // shared memory fits this GPU's per-block opt-in limit, so schwarz runs on
    // smaller-shared-memory architectures (e.g. Turing/T4 at 64KB) by using
    // smaller, cheaper dense blocks instead of failing the kernel launch. On
    // roomy GPUs the configured kmax is kept as-is. PPF_SCHWARZ_KMAX still sets
    // the upper bound. Logged once per distinct clamp so the reduction is
    // visible. Default kmax=16 (~18KB) already fits every supported arch, so
    // this only engages for large kmax on constrained hardware.
    {
        int dev = 0;
        CUDA_HANDLE_ERROR(cudaGetDevice(&dev));
        int max_optin = 0;
        CUDA_HANDLE_ERROR(cudaDeviceGetAttribute(
            &max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        unsigned kmax_req = kmax;
        while (kmax > 1 && factor_shmem(kmax) > (size_t)max_optin) {
            --kmax;
        }
        if (kmax != kmax_req) {
            static unsigned s_last_clamp = 0;
            if (s_last_clamp != kmax) {
                s_last_clamp = kmax;
                SimpleLog::message(
                    "[schwarz] kmax reduced %u -> %u to fit %d B per-block "
                    "shared memory (%zu B needed)",
                    kmax_req, kmax, max_optin, factor_shmem(kmax_req));
            }
        }
    }
    const unsigned max_dim = 3u * kmax;
    // Number of additive levels (multilevel additive Schwarz). Comes from
    // ParamSet::schwarz_levels (the schwarz-levels parameter; 2 = the two-level
    // coarse correction over the connectivity partition below, which reliably
    // reduces the worst PCG step on stiff multibody contact; 1 == single-level
    // smoother). A 0 (an unset parameter) falls back to 2. PPF_SCHWARZ_LEVELS
    // still overrides for debugging; clamped to MAX_LEVELS. Coarsening also stops
    // early when a level fits one domain.
    unsigned levels = levels_param > 0 ? levels_param : 2;
    if (const char *e = getenv("PPF_SCHWARZ_LEVELS")) {
        int v = atoi(e);
        if (v > 0) {
            levels = (unsigned)v;
        }
    }
    if (levels > MAX_LEVELS) {
        levels = MAX_LEVELS;
    }
    K.coarse_weight = 1.0f;
    if (const char *e = getenv("PPF_SCHWARZ_WEIGHT")) {
        float v = (float)atof(e);
        if (v > 0.0f) {
            K.coarse_weight = v;
        }
    }
    H.nrow = nrow;
    H.kmax = kmax;
    H.max_dim = max_dim;
    H.nlev = 1;
    if (nrow == 0) {
        H.n_agg = 0;
        return;
    }

    const bool struct_same = K.valid && K.nrow == nrow && K.kmax == kmax &&
                             K.bptr == (uintptr_t)B.value.data;
    const unsigned long long h = compute_dyn_hash(A);
    const bool reuse = struct_same && h == K.dyn_hash;

    if (!struct_same) {
        K.agg.free();
        K.agg_off.free();
        K.members.free();
        K.ainv_off.free();
        K.scratch.free();
        K.cnt.free();
        K.fine_off.free();
        K.fine_col.free();

        K.must_partner.free();
        K.ainv.free();
        K.M0.free();
        for (unsigned l = 0; l < MAX_LEVELS; ++l) {
            K.lvl[l].free_all();
        }
        K.nlev = 1;
        K.agg = Vec<unsigned>::alloc(nrow);
        K.agg_off = Vec<unsigned>::alloc(nrow + 1);
        K.members = Vec<unsigned>::alloc(nrow);
        K.ainv_off = Vec<unsigned>::alloc(nrow + 1);
        K.scratch = Vec<unsigned>::alloc(nrow);
        K.cnt = Vec<unsigned>::alloc(nrow);
        K.fine_off = Vec<unsigned>::alloc(nrow + 1);
        K.fine_cap = 0; // fine_col is (re)allocated on demand in build_fine_graph
        // ainv: worst-case 9*kmax floats per vertex (aggregate of m<=kmax stores
        // the packed lower triangle of G=L^-1), so it never needs realloc on
        // re-aggregation.
        K.ainv = Vec<float>::alloc((size_t)9 * kmax * nrow);
        K.nrow = nrow;
        K.kmax = kmax;
        K.max_dim = max_dim;
        K.bptr = (uintptr_t)B.value.data;
        K.valid = false;
    }

    if (!reuse || !K.valid) {
        recompute_aggregation(A, B, C);
        K.dyn_hash = h;
        K.valid = true;
    }

    const size_t shmem =
        (2u * (size_t)max_dim * max_dim + max_dim) * sizeof(float) +
        (size_t)kmax * sizeof(unsigned);
    // Opt into only the dynamic shared memory this launch actually needs (past
    // the ~48KB default), and only if it exceeds the default. kmax is already
    // clamped below build()'s device query so `shmem` fits the per-block opt-in
    // cap, so this never fails on the live path; the result is checked anyway so
    // a failure can never leak to the cudaGetLastError() after the launch.
    static int s_factor_shmem = 48 * 1024;
    ensure_dyn_shmem((const void *)factor_kernel, shmem, s_factor_shmem);
    factor_kernel<<<K.n_agg, BLOCK_THREADS, shmem>>>(
        A, B, C, K.agg_off, K.members, K.ainv_off, K.ainv, max_dim);
    CUDA_HANDLE_ERROR(cudaGetLastError());
    // No explicit sync: factor and the subsequent cg ops share the default
    // stream (ordered), and cg's first thrust op synchronizes anyway.


    K.levels_req = levels;
    if (levels >= 2) {
        // Multilevel additive Schwarz coarse correction (Galerkin + per-level
        // domain factor) over the connectivity partition. Rebuilt every solve;
        // K.nlev is the achieved depth.
        build_mas(A, B, C, levels, kmax);
    } else {
        K.nlev = 1;
    }

    H.n_agg = K.n_agg;
    H.agg = K.agg;
    H.agg_off = K.agg_off;
    H.members = K.members;
    H.ainv_off = K.ainv_off;
    H.ainv = K.ainv;
    H.nlev = K.nlev;
}


void free(SchwarzHierarchy &H) {
    // The hierarchy holds views into the persistent cache; nothing to release.
    (void)H;
}

} // namespace schwarz
