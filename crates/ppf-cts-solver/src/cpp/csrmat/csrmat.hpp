// File: csrmat.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef CSR_MAT_DEF_HPP
#define CSR_MAT_DEF_HPP

#include "../common.hpp"
#include "../data.hpp"
#include "../vec/vec.hpp"

// Windows defines OVERFLOW as a macro - undefine it
#ifdef OVERFLOW
#undef OVERFLOW
#endif

struct Row {
    enum State { SUCCESS, OVERFLOW, COUNTING };
    __device__ void alloc();
    __device__ void clear();
    __device__ void finalize();
    __device__ void dry_push(unsigned i);
    __device__ void push(unsigned i, const Mat3x3f &val);
    unsigned max_dyn_rows;
    unsigned head{0};
    unsigned ref_head{0};
    State state;
    unsigned *index;
    unsigned *ref_index;
    Mat3x3f *value;
    unsigned *ref_value;
    unsigned *fixed_index;
    unsigned fixed_nnz{0};
};

struct DynCSRMat {
    static DynCSRMat alloc(unsigned nrow, unsigned max_nnz);
    void fetch(unsigned *index, Mat3x3f *value, unsigned *offset);
    void update(unsigned *index, unsigned *offset);
    void start_rebuild_buffer();
    void finish_rebuild_buffer(unsigned &max_nnz_row, float &consumed);
    void free();
    __device__ Mat3x3f operator()(unsigned i, unsigned j) const;
    __device__ void dry_push(unsigned row, unsigned col);
    __device__ void push(unsigned row, unsigned col, const Mat3x3f &val);
    __device__ unsigned nnz(unsigned row) const;
    DynCSRMat clear();
    void finalize();
    bool check();
    Vec<Row> rows;
    Vec<unsigned> dyn_row_offsets;
    Vec<unsigned> dyn_index_buff;
    Vec<Mat3x3f> dyn_value_buff;
    Vec<unsigned> fixed_row_offsets;
    Vec<unsigned> fixed_index_buff;
    Vec<unsigned> ref_row_offsets;
    Vec<unsigned> ref_index_buff;
    Vec<unsigned> ref_value_buff;
    Vec<unsigned> tmp_array;
    unsigned nrow{0};
    unsigned max_nnz{0};
    unsigned peak_nnz{0};
};

struct FixedCSRMat {
    static FixedCSRMat alloc(VecVec<unsigned> index_table,
                             VecVec<Vec2u> transpose_table);
    void free();
    void clear();
    __device__ Mat3x3f operator()(unsigned i, unsigned j) const;
    __device__ bool push(unsigned i, unsigned j, const Mat3x3f &val);
    __device__ bool exists(unsigned i, unsigned j) const;
    bool check();
    void copy(const FixedCSRMat &other);
    bool finalize();
    VecVec<unsigned> index;
    VecVec<Vec2u> transpose;
    Vec<Mat3x3f> value;
    unsigned nrow{0};
    unsigned *status;
};

#endif
