// File: csrmat.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../kernels/exclusive_scan.hpp"
#include "../kernels/reduce.hpp"
#include "../kernels/vec_ops.hpp"
#include "../main/cuda_utils.hpp"
#include "../simplelog/SimpleLog.h"
#include "../utility/dispatcher.hpp"
#include "../utility/utility.hpp"
#include "csrmat.hpp"

__device__ void Row::alloc() {
    head = 0;
    ref_head = 0;
    max_dyn_rows = 0;
}

__device__ void Row::clear() {
    state = SUCCESS;
    index = nullptr;
    value = nullptr;
    fixed_index = nullptr;
    ref_value = nullptr;
    ref_index = nullptr;
    fixed_nnz = 0;
    max_dyn_rows = 0;
    head = 0;
    ref_head = 0;
}

__device__ void Row::finalize() {
    assert(state == SUCCESS);
    unsigned nnz = head;
    head = 0;
    for (unsigned i = 0; i < nnz; ++i) {
        unsigned j = index[i];
        Mat3x3f &val = value[i];
        if (!val.isZero()) {
            bool found = false;
            for (unsigned k = 0; k < head; ++k) {
                if (index[k] == j) {
                    value[k] += val;
                    found = true;
                    break;
                }
            }
            if (!found) {
                unsigned _head = head++;
                index[_head] = j;
                value[_head] = val;
            }
        }
    }
}

__device__ void Row::dry_push(unsigned i) {
    assert(state == COUNTING);
    for (unsigned j = 0; j < fixed_nnz; ++j) {
        if (fixed_index[j] == i) {
            return;
        }
    }
    atomicAdd(&max_dyn_rows, 1);
}

__device__ void Row::push(unsigned i, const Mat3x3f &val) {
    assert(state != COUNTING);
    for (unsigned j = 0; j < fixed_nnz; ++j) {
        if (index[j] == i) {
            float *ptr = (float *)(value + j);
            for (unsigned ii = 0; ii < 9; ++ii) {
                float y = Map<const Vec9f>(val.data())[ii];
                if (y) {
                    atomicAdd(ptr + ii, y);
                }
            }
            return;
        }
    }
    unsigned offset = atomicAdd(&head, 1);
    index[offset] = i;
    value[offset] = val;
}

DynCSRMat DynCSRMat::alloc(unsigned nrow, unsigned max_nnz) {
    DynCSRMat result;
    result.rows = Vec<Row>::alloc(nrow);
    result.max_nnz = max_nnz;
    result.dyn_row_offsets = Vec<unsigned>::alloc(nrow + 1).clear(0);
    result.dyn_index_buff = Vec<unsigned>::alloc(max_nnz).clear(0);
    result.dyn_value_buff = Vec<Mat3x3f>::alloc(max_nnz).clear(Mat3x3f::Zero());
    result.ref_row_offsets = Vec<unsigned>::alloc(nrow).clear(0);
    result.ref_index_buff = Vec<unsigned>::alloc(max_nnz).clear(0);
    result.ref_value_buff = Vec<unsigned>::alloc(max_nnz).clear(0);
    result.fixed_row_offsets = Vec<unsigned>::alloc(nrow + 1).clear(0);
    result.fixed_index_buff = Vec<unsigned>::alloc(max_nnz).clear(0);
    result.tmp_array = Vec<unsigned>::alloc(nrow).clear(0);
    result.nrow = nrow;
    float tmp_1;
    unsigned tmp_2;
    result.finish_rebuild_buffer(tmp_2, tmp_1);
    return result;
}

void DynCSRMat::fetch(unsigned *index, Mat3x3f *value, unsigned *offset) {
    CUDA_HANDLE_ERROR(cudaMemcpy(offset, fixed_row_offsets.data,
                                 (nrow + 1) * sizeof(unsigned),
                                 cudaMemcpyDeviceToHost));
    unsigned count = offset[nrow];
    CUDA_HANDLE_ERROR(cudaMemcpy(index, fixed_index_buff.data,
                                 count * sizeof(unsigned),
                                 cudaMemcpyDeviceToHost));
}

void DynCSRMat::update(unsigned *value, unsigned *offset) {
    CUDA_HANDLE_ERROR(cudaMemcpy(fixed_row_offsets.data, offset,
                                 (nrow + 1) * sizeof(unsigned),
                                 cudaMemcpyHostToDevice));
    CUDA_HANDLE_ERROR(cudaMemcpy(fixed_index_buff.data, value,
                                 offset[nrow] * sizeof(unsigned),
                                 cudaMemcpyHostToDevice));
}

void DynCSRMat::start_rebuild_buffer() {
    Vec<Row> rows = this->rows;
    Vec<unsigned> fixed_row_offsets = this->fixed_row_offsets;
    Vec<unsigned> fixed_index_buff = this->fixed_index_buff;
    DISPATCH_START(nrow)
    [rows, fixed_row_offsets, fixed_index_buff] __device__(unsigned i) mutable {
        unsigned nnz = fixed_row_offsets[i + 1] - fixed_row_offsets[i];
        rows[i].clear();
        rows[i].state = Row::COUNTING;
        rows[i].fixed_index = fixed_index_buff.data + fixed_row_offsets[i];
        rows[i].fixed_nnz = nnz;
        rows[i].max_dyn_rows = nnz;
    } DISPATCH_END;
}

void DynCSRMat::finish_rebuild_buffer(unsigned &max_nnz_row,
                                      float &consumed_rat) {
    Vec<unsigned> fixed_row_offsets = this->fixed_row_offsets;
    Vec<unsigned> fixed_index_buff = this->fixed_index_buff;
    Vec<Row> rows = this->rows;
    Vec<unsigned> dyn_row_offsets = this->dyn_row_offsets;
    Vec<unsigned> tmp_array = this->tmp_array;

    DISPATCH_START(nrow)
    [dyn_row_offsets, tmp_array, rows] __device__(unsigned i) mutable {
        dyn_row_offsets[i] = rows[i].max_dyn_rows;
        tmp_array[i] = rows[i].max_dyn_rows + rows[i].fixed_nnz;
    } DISPATCH_END;

    max_nnz_row = kernels::max_array(tmp_array.data, nrow, 0u);

    unsigned num_nnz = 0;
    if (max_nnz_row) {
        num_nnz = kernels::exclusive_scan(dyn_row_offsets.data, nrow);
        if (num_nnz >= max_nnz) {
            printf("finish_rebuild_buffer: num_nnz %u, max_nnz: %u\n", num_nnz,
                   max_nnz);
            assert(false);
        }
    } else {
        dyn_row_offsets.clear(0);
    }

    Vec<unsigned> dyn_index_buff = this->dyn_index_buff;
    Vec<Mat3x3f> dyn_value_buff = this->dyn_value_buff;

    DISPATCH_START(nrow)
    [dyn_row_offsets, dyn_index_buff, dyn_value_buff,
     rows] __device__(unsigned i) mutable {
        rows[i].index = dyn_index_buff.data + dyn_row_offsets[i];
        rows[i].value = dyn_value_buff.data + dyn_row_offsets[i];
    } DISPATCH_END;

    DISPATCH_START(nrow)
    [fixed_row_offsets, fixed_index_buff, rows] __device__(unsigned i) mutable {
        unsigned fixed_row_nnz =
            fixed_row_offsets[i + 1] - fixed_row_offsets[i];
        for (unsigned j = 0; j < fixed_row_nnz; ++j) {
            rows[i].index[j] = fixed_index_buff[fixed_row_offsets[i] + j];
            rows[i].value[j] = Mat3x3f::Zero();
        }
        rows[i].head = fixed_row_nnz;
        rows[i].state = Row::SUCCESS;
    } DISPATCH_END;

    consumed_rat = num_nnz / (float)max_nnz;
    this->peak_nnz = num_nnz;
}

void DynCSRMat::free() {
    rows.free();
    dyn_row_offsets.free();
    dyn_index_buff.free();
    dyn_value_buff.free();
    fixed_index_buff.free();
    fixed_row_offsets.free();
}

__device__ void DynCSRMat::dry_push(unsigned row, unsigned col) {
    if (row <= col) {
        rows[row].dry_push(col);
    }
}

__device__ void DynCSRMat::push(unsigned row, unsigned col,
                                const Mat3x3f &val) {
    if (row <= col) {
        rows[row].push(col, val);
    }
}

DynCSRMat DynCSRMat::clear() {
    Vec<Row> rows = this->rows;
    DISPATCH_START(rows.size)
    [rows] __device__(unsigned i) mutable { rows[i].clear(); } DISPATCH_END;
    return *this;
}

void DynCSRMat::finalize() {
    assert(check());
    Vec<Row> rows = this->rows;
    DISPATCH_START(rows.size)
    [rows] __device__(unsigned i) mutable { rows[i].finalize(); } DISPATCH_END;
    assert(check());

    Vec<unsigned> fixed_row_offsets = this->fixed_row_offsets;
    Vec<unsigned> fixed_index_buff = this->fixed_index_buff;

    DISPATCH_START(nrow)
    [fixed_row_offsets, rows] __device__(unsigned i) mutable {
        fixed_row_offsets[i] = rows[i].head;
    } DISPATCH_END;

    unsigned num_fixed_nnz =
        kernels::exclusive_scan(fixed_row_offsets.data, nrow);
    if (num_fixed_nnz > max_nnz) {
        printf("num_fixed_nnz: %u, max_nnz: %u\n", num_fixed_nnz, max_nnz);
        assert(false);
    }

    CUDA_HANDLE_ERROR(cudaMemcpy(fixed_row_offsets.data + nrow, &num_fixed_nnz,
                                 sizeof(unsigned), cudaMemcpyHostToDevice));

    DISPATCH_START(nrow)
    [fixed_row_offsets, fixed_index_buff, rows] __device__(unsigned i) mutable {
        for (int j = 0; j < rows[i].head; j++) {
            unsigned k = fixed_row_offsets[i] + j;
            fixed_index_buff[k] = rows[i].index[j];
        }
    } DISPATCH_END;

    Vec<unsigned> ref_index_buff = this->ref_index_buff;
    Vec<unsigned> ref_index_offsets = this->ref_row_offsets;
    Vec<unsigned> ref_value_buff = this->ref_value_buff;

    ref_index_offsets.clear(0);
    DISPATCH_START(nrow)
    [rows, ref_index_offsets] __device__(unsigned i) mutable {
        for (unsigned k = 0; k < rows[i].head; ++k) {
            unsigned j = rows[i].index[k];
            if (i != j) {
                atomicAdd(ref_index_offsets.data + j, 1);
            }
        }
    } DISPATCH_END;

    unsigned num_nnz = kernels::exclusive_scan(ref_index_offsets.data, nrow);
    if (num_nnz >= max_nnz) {
        printf("transpose num_nnz %u, max_nnz: %u\n", num_nnz, max_nnz);
        assert(false);
    }

    DISPATCH_START(nrow)
    [rows, ref_index_offsets, ref_index_buff,
     ref_value_buff] __device__(unsigned i) mutable {
        unsigned offset = ref_index_offsets[i];
        rows[i].ref_head = 0;
        rows[i].ref_index = ref_index_buff.data + offset;
        rows[i].ref_value = ref_value_buff.data + offset;
    } DISPATCH_END;

    Vec<unsigned> dyn_row_offsets = this->dyn_row_offsets;
    DISPATCH_START(nrow)
    [rows, ref_index_offsets, ref_index_buff, dyn_row_offsets,
     ref_value_buff] __device__(unsigned i) mutable {
        for (unsigned k = 0; k < rows[i].head; ++k) {
            unsigned j = rows[i].index[k];
            if (i != j) {
                unsigned offset = atomicAdd(&rows[j].ref_head, 1);
                rows[j].ref_index[offset] = i;
                rows[j].ref_value[offset] = dyn_row_offsets[i] + k;
            }
        }
    } DISPATCH_END;
}

bool DynCSRMat::check() {
    Vec<Row> rows = this->rows;
    DISPATCH_START(rows.size)[rows] __device__(unsigned i) {
        assert(rows[i].state == Row::SUCCESS);
    }
    DISPATCH_END;
    return true;
}

__device__ unsigned DynCSRMat::nnz(unsigned row) const {
    return rows[row].head;
}

__device__ Mat3x3f DynCSRMat::operator()(unsigned i, unsigned j) const {
    Mat3x3f val = Mat3x3f::Zero();
    if (i >= rows.size) {
        return val;
    }
    if (i <= j) {
        for (unsigned k = 0; k < rows[i].head; ++k) {
            if (rows[i].index[k] == j) {
                val += rows[i].value[k];
            }
        }
        return val;
    } else {
        for (unsigned k = 0; k < rows[j].head; ++k) {
            if (rows[j].index[k] == i) {
                val += rows[j].value[k];
            }
        }
        return val.transpose();
    }
}

FixedCSRMat FixedCSRMat::alloc(VecVec<unsigned> index_table,
                               VecVec<Vec2u> transpose_table) {
    FixedCSRMat result;
    result.index = index_table;
    result.transpose = transpose_table;
    result.value = Vec<Mat3x3f>::alloc(index_table.nnz).clear(Mat3x3f::Zero());
    result.nrow = index_table.size;
    CUDA_HANDLE_ERROR(cudaMalloc(&result.status, sizeof(unsigned)));
    return result;
}

void FixedCSRMat::free() {
    value.free();
    CUDA_HANDLE_ERROR(cudaFree(status));
}

void FixedCSRMat::clear() {
    value.clear(Mat3x3f::Zero());
    CUDA_HANDLE_ERROR(cudaMemset(status, 0, sizeof(unsigned)));
}

__device__ Mat3x3f FixedCSRMat::operator()(unsigned i, unsigned j) const {
    Mat3x3f val = Mat3x3f::Zero();
    if (!value.data) {
        return val;
    }
    bool tr = false;
    if (i > j) {
        unsigned tmp = i;
        i = j;
        j = tmp;
        tr = true;
    }
    unsigned nrow = index.offset[index.size];
    if (i < nrow) {
        unsigned start = index.offset[i];
        unsigned end = index.offset[i + 1];
        for (unsigned k = start; k < end; ++k) {
            if (index.data[k] == j) {
                val += value.data[k];
            }
        }
    }
    if (tr) {
        return val.transpose();
    } else {
        return val;
    }
}

__device__ bool FixedCSRMat::push(unsigned i, unsigned j, const Mat3x3f &val) {
    bool found = false;
    if (i <= j) {
        unsigned nrow = index.offset[index.size];
        unsigned start = index.offset[i];
        unsigned end = index.offset[i + 1];
        if (i < nrow) {
            for (unsigned k = start; k < end; ++k) {
                if (index.data[k] == j) {
                    float *ptr = (float *)(value.data + k);
                    for (unsigned ii = 0; ii < 9; ++ii) {
                        float y = Map<const Vec9f>(val.data())[ii];
                        if (y) {
                            atomicAdd(ptr + ii, y);
                        }
                    }
                    found = true;
                    break;
                } else if (index.data[k] > j) {
                    break;
                }
            }
        }
    }
    return found;
}

__device__ bool FixedCSRMat::exists(unsigned i, unsigned j) const {
    bool found = false;
    if (i <= j) {
        unsigned nrow = index.offset[index.size];
        unsigned start = index.offset[i];
        unsigned end = index.offset[i + 1];
        if (i < nrow) {
            for (unsigned k = start; k < end; ++k) {
                if (index.data[k] == j) {
                    found = true;
                    break;
                } else if (index.data[k] > j) {
                    break;
                }
            }
        }
    }
    return found;
}

bool FixedCSRMat::check() {
    unsigned host_status;
    CUDA_HANDLE_ERROR(cudaMemcpy(&host_status, status, sizeof(unsigned),
                                 cudaMemcpyDeviceToHost));
    return host_status == 0;
}

void FixedCSRMat::copy(const FixedCSRMat &other) {
    kernels::copy(other.value.data, this->value.data, this->value.size);
}

bool FixedCSRMat::finalize() { return check(); }
