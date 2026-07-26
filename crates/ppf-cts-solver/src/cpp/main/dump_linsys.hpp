// File: dump_linsys.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Offline linear-system dumper for analyzing the assembled Newton system out of
// band. Env-gated by PPF_DUMP_LINSYS=<k>: on the k-th call to solver::solve
// (0-based), write the assembled Newton system M x = b (M = A_dyn + B_fixed +
// C_diag, upper triangle since both CSRs store the upper triangle and imply the
// transpose) to $PPF_DUMP_DIR/linsys_<k>.bin, then exit(0). Off by default and
// compiled but never run unless the env var is set, so it cannot affect any
// production build. File format (little-endian):
//   u32 magic 0x4C535953 ("LSYS"), u32 nrow, u32 n_offdiag
//   n_offdiag records: u32 row, u32 col, 9x f32   (upper-triangle i<j blocks of A+B)
//   nrow blocks:        9x f32                      (diagonal blocks A(i,i)+B(i,i)+C[i])
//   3*nrow f32                                      (rhs b)

#ifndef DUMP_LINSYS_HPP
#define DUMP_LINSYS_HPP

#include "../csrmat/csrmat.hpp"
#include "../data.hpp"
#include "../vec/vec.hpp"
#include "cuda_utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace dump_linsys {

// One thread per row: append this row's stored (upper-triangle) blocks to a
// shared COO buffer via an atomic cursor. Diagonal blocks (col == row) are split
// out into `diag` (summed with C separately on the host); strictly-upper blocks
// (col > row) go to the off-diagonal COO. Lower-triangle blocks are not stored
// by either CSR (they are implied transposes), so they are not emitted.
__global__ void dyn_coo_kernel(Vec<Row> rows, unsigned nrow, unsigned *cursor,
                               unsigned *out_row, unsigned *out_col,
                               Mat3x3f *out_val, Mat3x3f *diag, unsigned cap) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrow) {
        return;
    }
    const Row &r = rows.data[i];
    for (unsigned k = 0; k < r.head; ++k) {
        unsigned j = r.index[k];
        if (j == i) {
            diag[i] += r.value[k];
        } else if (j > i) {
            unsigned pos = atomicAdd(cursor, 1u);
            if (pos < cap) {
                out_row[pos] = i;
                out_col[pos] = j;
                out_val[pos] = r.value[k];
            }
        }
    }
}

__global__ void fixed_coo_kernel(FixedCSRMat B, unsigned nrow, unsigned *cursor,
                                 unsigned *out_row, unsigned *out_col,
                                 Mat3x3f *out_val, Mat3x3f *diag, unsigned cap) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrow) {
        return;
    }
    unsigned start = B.index.offset[i];
    unsigned end = B.index.offset[i + 1];
    for (unsigned k = start; k < end; ++k) {
        unsigned j = B.index.data[k];
        if (j == i) {
            diag[i] += B.value.data[k];
        } else if (j > i) {
            unsigned pos = atomicAdd(cursor, 1u);
            if (pos < cap) {
                out_row[pos] = i;
                out_col[pos] = j;
                out_val[pos] = B.value.data[k];
            }
        }
    }
}

// Add C (the per-vertex diagonal inertia+contact block) into diag.
__global__ void add_diag_kernel(const Mat3x3f *C, Mat3x3f *diag, unsigned nrow) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nrow) {
        diag[i] += C[i];
    }
}

// Returns true if a dump happened (caller may then exit). Runs on the default
// stream; correctness, not speed, matters here.
inline void maybe_dump(const DynCSRMat &A, const FixedCSRMat &B,
                       const Vec<Mat3x3f> &C, const Vec<float> &b) {
    static int target = [] {
        const char *e = std::getenv("PPF_DUMP_LINSYS");
        return e ? std::atoi(e) : -1;
    }();
    static int counter = 0;
    if (target < 0) {
        return;
    }
    if (counter++ != target) {
        return;
    }

    const unsigned nrow = A.nrow;
    // Upper-bound the off-diagonal COO by the total stored nnz of both CSRs.
    unsigned dyn_nnz = 0;
    CUDA_HANDLE_ERROR(cudaMemcpy(&dyn_nnz, A.fixed_row_offsets.data + nrow,
                                 sizeof(unsigned), cudaMemcpyDeviceToHost));
    const unsigned cap = dyn_nnz + B.value.size + nrow + 16u;

    auto d_cursor = Vec<unsigned>::alloc(1).clear(0);
    auto d_row = Vec<unsigned>::alloc(cap);
    auto d_col = Vec<unsigned>::alloc(cap);
    auto d_val = Vec<Mat3x3f>::alloc(cap);
    auto d_diag = Vec<Mat3x3f>::alloc(nrow).clear(Mat3x3f::Zero());

    const unsigned blk = 256;
    const unsigned grid = (nrow + blk - 1) / blk;
    Vec<Row> rows = A.rows;
    dyn_coo_kernel<<<grid, blk>>>(rows, nrow, d_cursor.data, d_row.data,
                                  d_col.data, d_val.data, d_diag.data, cap);
    CUDA_HANDLE_ERROR(cudaGetLastError());
    fixed_coo_kernel<<<grid, blk>>>(B, nrow, d_cursor.data, d_row.data,
                                    d_col.data, d_val.data, d_diag.data, cap);
    CUDA_HANDLE_ERROR(cudaGetLastError());
    add_diag_kernel<<<grid, blk>>>(C.data, d_diag.data, nrow);
    CUDA_HANDLE_ERROR(cudaGetLastError());
    CUDA_HANDLE_ERROR(cudaDeviceSynchronize());

    unsigned n_off = 0;
    CUDA_HANDLE_ERROR(cudaMemcpy(&n_off, d_cursor.data, sizeof(unsigned),
                                 cudaMemcpyDeviceToHost));
    if (n_off > cap) {
        fprintf(stderr, "[dump_linsys] COO overflow %u > cap %u\n", n_off, cap);
        n_off = cap;
    }

    std::vector<unsigned> h_row(n_off), h_col(n_off);
    std::vector<Mat3x3f> h_val(n_off), h_diag(nrow);
    std::vector<float> h_b(3u * nrow);
    if (n_off) {
        CUDA_HANDLE_ERROR(cudaMemcpy(h_row.data(), d_row.data,
                                     n_off * sizeof(unsigned),
                                     cudaMemcpyDeviceToHost));
        CUDA_HANDLE_ERROR(cudaMemcpy(h_col.data(), d_col.data,
                                     n_off * sizeof(unsigned),
                                     cudaMemcpyDeviceToHost));
        CUDA_HANDLE_ERROR(cudaMemcpy(h_val.data(), d_val.data,
                                     n_off * sizeof(Mat3x3f),
                                     cudaMemcpyDeviceToHost));
    }
    CUDA_HANDLE_ERROR(cudaMemcpy(h_diag.data(), d_diag.data,
                                 nrow * sizeof(Mat3x3f),
                                 cudaMemcpyDeviceToHost));
    CUDA_HANDLE_ERROR(cudaMemcpy(h_b.data(), b.data, 3u * nrow * sizeof(float),
                                 cudaMemcpyDeviceToHost));

    const char *dir = std::getenv("PPF_DUMP_DIR");
    char path[1024];
    snprintf(path, sizeof(path), "%s/linsys_%d.bin", dir ? dir : ".", target);
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[dump_linsys] cannot open %s\n", path);
        return;
    }
    unsigned magic = 0x4C535953u;
    fwrite(&magic, sizeof(unsigned), 1, f);
    fwrite(&nrow, sizeof(unsigned), 1, f);
    fwrite(&n_off, sizeof(unsigned), 1, f);
    for (unsigned k = 0; k < n_off; ++k) {
        fwrite(&h_row[k], sizeof(unsigned), 1, f);
        fwrite(&h_col[k], sizeof(unsigned), 1, f);
        fwrite(h_val[k].data(), sizeof(float), 9, f);
    }
    for (unsigned i = 0; i < nrow; ++i) {
        fwrite(h_diag[i].data(), sizeof(float), 9, f);
    }
    fwrite(h_b.data(), sizeof(float), 3u * nrow, f);
    fclose(f);
    fprintf(stderr,
            "[dump_linsys] wrote %s: nrow=%u n_offdiag=%u (solve #%d)\n", path,
            nrow, n_off, target);

    d_cursor.free();
    d_row.free();
    d_col.free();
    d_val.free();
    d_diag.free();
}

} // namespace dump_linsys

#endif // DUMP_LINSYS_HPP
