// File: buffer.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../csrmat/csrmat.hpp"
#include "../data.hpp"
#include "buffer.hpp"

namespace buffer {

// Global memory pool instance. Pre-seeded by reserve_for_mesh() at solver init
// and grown to its high-water mark on demand by get<T>().
static MemoryPool global_pool;

MemoryPool &get() { return global_pool; }

void reserve_for_mesh(unsigned n_verts, unsigned n_edges, unsigned n_faces,
                      unsigned n_bodies) {
    // Worst-case concurrent demand by buffer class (in floats), drawn from the
    // hot-loop allocation inventory. The high-water get<T>() growth covers any
    // class this under-provisions, so these counts only need to be in the right
    // ballpark; they front-load the common buffers off the first step.
    const size_t V = n_verts ? n_verts : 1;
    const size_t E = n_edges ? n_edges : 1;
    const size_t F = n_faces ? n_faces : 1;
    const size_t B = n_bodies; // 0 for cloth-only scenes

    std::vector<std::pair<size_t, size_t>> slots;
    // Vertex-DOF class (3*V + 6*B floats): the PCG / rigid-PCG work vectors
    // (tmp/z/p, cg_rigid's f/xr/r/z/p/Rp/tmp/xv/mxv) plus advance()'s per-vertex
    // eval_x/target/velocity/force/dx/rigid_tgt and contact_force. ~16 live at
    // the deepest nesting of the rigid path.
    slots.push_back({3 * V + 6 * B, 16});
    // Per-vertex 3x3 Hessian diagonal class (9*V): diag_hess + inv_diag.
    slots.push_back({9 * V, 3});
    // Per-vertex scalar class (V): contact counters, TOIs, scalar reductions.
    slots.push_back({V, 4});
    // Per-edge class (E): edge contact counters, edge TOIs, intersection flags.
    slots.push_back({E, 3});
    // Per-face class (12*F): the per-face SVD scratch.
    slots.push_back({12 * F, 2});
    // Per-body class (36*B): rigid 6x6 block preconditioner + rigid state.
    if (B) {
        slots.push_back({36 * B, 2});
    }
    // Small fixed-size scratch: block-reduction partials (double-backed, capped
    // at 1024 elements) and the intersection-record pool.
    slots.push_back({2048, 3});
    slots.push_back({5120, 1});

    global_pool.reserve(slots);
}

} // namespace buffer
