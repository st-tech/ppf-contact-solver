// File: schwarz.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef SCHWARZ_HPP
#define SCHWARZ_HPP

#include "../csrmat/csrmat.hpp"
#include "../data.hpp"
#include "../main/cuda_utils.hpp" // cudaStream_t
#include "../vec/vec.hpp"

namespace schwarz {

// Plain BOTH-triangle block-CSR operator for the coarse levels (and the
// materialized level-0 Galerkin source) of the multilevel additive-Schwarz
// extension. Symmetry is owned by storing both triangles, so
// the gather is a single walk with no transpose bookkeeping. The materialized
// level-0 source stores the four loops' oriented partial blocks (A, A^T, B, B^T,
// C) without dedup; the Galerkin product dedups via a sort. col/value are
// reallocate-on-grow.
struct CoarseMat {
    unsigned n{0};        // block-row count
    unsigned nnz{0};      // stored blocks (both triangles)
    unsigned cap{0};      // current col/value capacity
    Vec<unsigned> offset; // [n+1]   block-row CSR offsets
    Vec<unsigned> col;    // [nnz]   column block ids
    Vec<Mat3x3f> value;   // [nnz]   one 3x3 block per stored entry
    void free() {
        offset.free();
        col.free();
        value.free();
        n = nnz = cap = 0;
    }
};

// Single-level additive aggregate-Schwarz preconditioner, rebuilt just before
// each PCG solve (the linear system M = A_dyn + B_fixed + C_diag is fully known
// at that point). Vertices are grouped into matrix-connected aggregates of
// <= kmax members; for each aggregate the dense SPD submatrix A_loc of M is
// gathered, floored, Cholesky-factored (A_loc = L L^T), and its inverse factor
// G = L^-1 stored. The preconditioner applies the per-aggregate inverse
// additively as a Gram form, z_seg = A_loc^-1 r_seg = G^T (G r_seg), which is SPD
// by construction (r . z = ||G r||^2 >= 0 even in float32). kmax == 1 reduces to
// block-Jacobi. Connectivity is the matrix graph (the four neighbor loops the
// matvec walks), never geometry, so disconnected components are handled
// gracefully by construction.
//
// All Vec fields are NON-OWNING views into a persistent cache that build() owns;
// the hierarchy stays valid until the next build(). free() is a no-op.
struct SchwarzHierarchy {
    unsigned nrow{0};       // vertex DOF count (== A.nrow)
    unsigned n_agg{0};      // number of aggregates (set by build)
    unsigned kmax{1};       // max members per aggregate
    unsigned max_dim{0};    // max 3*members over aggregates (for launch/shmem)
    Vec<unsigned> agg;      // [nrow]   aggregate id per vertex
    Vec<unsigned> agg_off;  // [nrow+1] CSR offsets into members (n_agg+1 used)
    Vec<unsigned> members;  // [nrow]   vertex ids grouped by aggregate
    Vec<unsigned> ainv_off; // [nrow+1] float offsets into ainv (n_agg+1 used)
    Vec<float> ainv;        // packed per-aggregate inverse Cholesky factors G=L^-1
    // Multilevel additive Schwarz. nlev == 1 is exactly the
    // single-level smoother above (bit-for-bit). For nlev >= 2 the coarse levels
    // live in the file-static cache and apply() adds their corrections additively
    // (z = S_0 r + sum_{l>=1} C_l^T S_l C_l r); no fine matvec, no exact coarse
    // solve. Selected by PPF_SCHWARZ_LEVELS (defaults to 2 when unset).
    unsigned nlev{1};
};

// Build the hierarchy from the assembled split system. `positions` is reserved
// (aggregation is matrix-graph based, id-ordered seeding). kmax defaults to 16
// and is overridable via PPF_SCHWARZ_KMAX; the aggregation knobs are the other
// PPF_SCHWARZ_* environment variables.
void build(SchwarzHierarchy &H, const DynCSRMat &A, const FixedCSRMat &B,
           const Vec<Mat3x3f> &C, const Vec<Vec3f> &positions, unsigned levels);

// Apply the preconditioner: result = M_schwarz^{-1} x (one additive Schwarz
// sweep). SPD by construction, so PCG stays valid. All kernels run on `stream`
// (default legacy stream); the device-resident PCG loop passes its own stream so
// the preconditioner chains without a host sync and the iteration is graph-
// capturable.
void apply(const SchwarzHierarchy &H, const Vec<float> &x, Vec<float> &result,
           cudaStream_t stream = 0);

// No-op: all buffers are owned by the persistent cache.
void free(SchwarzHierarchy &H);

} // namespace schwarz

#endif // SCHWARZ_HPP
