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
#include "asm_profile.hpp"
#include "csrmat.hpp"
#include "row_dedupe.hpp"
#include "row_pattern.hpp"
#include <cstdlib>

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
    split = 0;
    ref_head = 0;
}

__device__ void Row::finalize() {
    assert(state == SUCCESS);
    // fixed_nnz is exactly the width push() searched before appending, which is
    // the boundary row_dedupe needs. See row_dedupe.hpp for why that boundary
    // is what keeps this out of the row's width squared.
    head = row_dedupe(index, value, head, fixed_nnz, &split);

    // row_dedupe is allowed to skip comparing an appended column against the
    // carried ones only because push() appends a column solely after failing to
    // find it there. Check that rather than take it on faith: the pattern is
    // sorted, so this bisects instead of scanning and costs a handful of probes
    // per appended column. Without it a broken appender is close to
    // undetectable, because the row it produces is still NUMERICALLY right
    // (operator() and the transpose build both accumulate over every matching
    // entry) and the only symptom is a step that takes tens of seconds.
    //
    // Search the pattern push() ACTUALLY searched, not the compacted survivors.
    // A column that push() kept missing is pushed nowhere, so its carried slot
    // keeps the zero finish_rebuild_buffer wrote and row_dedupe drops it before
    // `split` exists; searching below `split` would therefore look for the
    // offending column in the one place it is guaranteed not to be, and pass on
    // exactly the failure it is written to catch. fixed_index still holds that
    // pattern here, since fixed_row_offsets is not rewritten until after this
    // dispatch completes.
    for (unsigned k = split; k < head; ++k) {
        assert(find_sorted(fixed_index, fixed_nnz, index[k]) == fixed_nnz);
    }
}

__device__ void Row::dry_push(unsigned i) {
    assert(state == COUNTING);
    // The carried pattern is sorted (see row_pattern.hpp), so a column that is
    // already in it is found by bisection rather than by walking the row.
    if (find_sorted(fixed_index, fixed_nnz, i) == fixed_nnz) {
        atomicAdd(&max_dyn_rows, 1);
    }
}

// Accumulate one block into this row. A column already present in the carried
// pattern is folded into its existing slot; anything else is appended.
//
// NOTE: the search below is what makes an appended column provably distinct
// from every carried one, and finalize() relies on exactly that to skip
// comparisons it knows cannot match. Appending without first failing this
// search would leave duplicates that finalize() does not look for.
__device__ void Row::push(unsigned i, const Mat3x3f &val) {
    assert(state != COUNTING);
    // The prefix holds the carried pattern in the sorted order finalize() left
    // it in, so this is a bisection, not a walk over the row.
    const unsigned slot = find_sorted(index, fixed_nnz, i);
    if (slot != fixed_nnz) {
        float *ptr = (float *)(value + slot);
        for (unsigned ii = 0; ii < 9; ++ii) {
            float y = Map<const Vec9f>(val.data())[ii];
            if (y) {
                atomicAdd(ptr + ii, y);
            }
        }
        return;
    }
    unsigned offset = atomicAdd(&head, 1);
    // The row's slab is sized by the counting pass (max_dyn_rows starts at
    // fixed_nnz and dry_push adds one per dynamic entry), so in a consistent
    // run `offset` never reaches it. If the two passes ever disagree, writing
    // here would run past this row's slab and into the next row's blocks, which
    // corrupts a Hessian the solver then treats as valid. Refuse the write and
    // latch the row so the host turns it into a loud failure instead. Every
    // thread that trips it writes the same value, so the race is benign.
    if (offset >= max_dyn_rows) {
        state = OVERFLOW;
        return;
    }
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
    double tmp_3;
    result.finish_rebuild_buffer(tmp_2, tmp_1, tmp_3);
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
        // Both assembly passes locate a column in this pattern by bisection, so
        // establish the ordering they need HERE, where a step starts using the
        // pattern, rather than trusting whoever last wrote it. finalize() is
        // not the only writer: update_dyn (main.cu) restores the pattern from a
        // saved state, so a run resumed from a checkpoint would otherwise
        // bisect an array that was never ordered. sort_pattern returns after a
        // single pass when the pattern is already in order, which is every step
        // but the first after a restore.
        sort_pattern(rows[i].fixed_index, nnz);
    } DISPATCH_END;
}

void DynCSRMat::finish_rebuild_buffer(unsigned &max_nnz_row,
                                      float &consumed_rat,
                                      double &report_overhead_ms) {
    report_overhead_ms = 0.0;
    Vec<unsigned> fixed_row_offsets = this->fixed_row_offsets;
    Vec<unsigned> fixed_index_buff = this->fixed_index_buff;
    Vec<Row> rows = this->rows;
    Vec<unsigned> dyn_row_offsets = this->dyn_row_offsets;
    Vec<unsigned> tmp_array = this->tmp_array;

    DISPATCH_START(nrow)
    [dyn_row_offsets, tmp_array, rows] __device__(unsigned i) mutable {
        // max_dyn_rows already IS this row's full slab width: it was seeded
        // with the carried pattern and then took one slot per column the dry
        // pass did not find there. Both the offsets and the reported width are
        // that one quantity, so neither adds the carried pattern on top of it.
        dyn_row_offsets[i] = rows[i].max_dyn_rows;
        tmp_array[i] = rows[i].max_dyn_rows;
    } DISPATCH_END;

    max_nnz_row = kernels::max_array(tmp_array.data, nrow, 0u);

    if (asm_profile::enabled()) {
        // Everything below is diagnostic and sits inside a window the caller is
        // timing, so hand back what it cost and let the caller subtract it. A
        // profiler that reports its own overhead as part of the phase it is
        // profiling makes the phase look expensive for the one reason that
        // vanishes when the profiler is turned off.
        const auto t_report = asm_profile::tick();
        // tmp_array holds this row's reserved slab: the carried-forward pattern
        // plus one slot per column the dry pass did not find in it.
        std::vector<unsigned> reserved(nrow);
        CUDA_HANDLE_ERROR(cudaMemcpy(reserved.data(), tmp_array.data,
                                     nrow * sizeof(unsigned),
                                     cudaMemcpyDeviceToHost));
        std::vector<unsigned> carried(nrow + 1);
        CUDA_HANDLE_ERROR(cudaMemcpy(carried.data(), fixed_row_offsets.data,
                                     (nrow + 1) * sizeof(unsigned),
                                     cudaMemcpyDeviceToHost));
        unsigned argmax = 0, peak = 0;
        for (unsigned i = 0; i < nrow; ++i) {
            if (reserved[i] > peak) {
                peak = reserved[i];
                argmax = i;
            }
        }
        // fixed_row_offsets is a prefix-sum, so the carried pattern width of a
        // row is the gap to the next offset.
        unsigned carried_nnz = carried[argmax + 1] - carried[argmax];
        SimpleLog::message(
            "[asm-prof] argmax row %u: carried_pattern=%u new_columns=%u",
            argmax, carried_nnz, peak - carried_nnz);
        asm_profile::report_widths("reserved_width", reserved);
        report_overhead_ms = asm_profile::ms_since(t_report);
    }

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
    const bool prof = asm_profile::enabled();
    auto t_phase = asm_profile::tick();

    // Not `assert(check())`: an assert would take the check out with it if this
    // ever built with NDEBUG, and an incomplete Hessian must never reach the
    // solve. check() has already printed what went wrong.
    if (!check()) {
        ppf_fatal(PPF_FATAL_SOLVER_INVARIANT,
                  "PPF FATAL: dynamic CSR pattern check failed; see the "
                  "report above.\n");
    }
    const double ms_check = asm_profile::ms_since(t_phase);

    t_phase = asm_profile::tick();
    Vec<Row> rows = this->rows;
    DISPATCH_START(rows.size)
    [rows] __device__(unsigned i) mutable { rows[i].finalize(); } DISPATCH_END;
    const double ms_rows = asm_profile::ms_since(t_phase);
    assert(check());

    t_phase = asm_profile::tick();
    Vec<unsigned> fixed_row_offsets = this->fixed_row_offsets;
    Vec<unsigned> fixed_index_buff = this->fixed_index_buff;

    DISPATCH_START(nrow)
    [fixed_row_offsets, rows] __device__(unsigned i) mutable {
        fixed_row_offsets[i] = rows[i].head;
    } DISPATCH_END;

    // fixed_row_offsets currently holds the per-row width AFTER the dedupe,
    // i.e. the true number of distinct nonzero blocks in that row. Snapshot it
    // before the scan below turns it back into a prefix-sum. The snapshot is a
    // blocking copy sitting inside a phase this function is timing, so charge
    // it back out rather than let a diagnostic inflate the number it reports.
    std::vector<unsigned> deduped;
    double ms_snapshot = 0.0;
    if (prof) {
        const auto t_snapshot = asm_profile::tick();
        deduped.resize(nrow);
        CUDA_HANDLE_ERROR(cudaMemcpy(deduped.data(), fixed_row_offsets.data,
                                     nrow * sizeof(unsigned),
                                     cudaMemcpyDeviceToHost));
        ms_snapshot = asm_profile::ms_since(t_snapshot);
    }

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
        // Hand the next step this row's columns, in ascending order, so that
        // the step picking them up finds them already sorted and does not have
        // to sort a row to be able to bisect it.
        //
        // The row is two ascending runs (row_dedupe leaves the carried entries
        // in the order they arrived, and that order was ascending), so this is
        // a merge, not a sort: linear, and it replaces a copy that cost the
        // same. Only the indices travel; the values are zeroed when the pattern
        // is read back, so there is nothing to permute alongside them, and the
        // row's own arrays are left as they are because every reader of those
        // walks the whole row.
        //
        unsigned *pattern = fixed_index_buff.data + fixed_row_offsets[i];
        const unsigned nnz = rows[i].head;
        const unsigned split = rows[i].split;
        merge_runs(rows[i].index, split, rows[i].index + split, nnz - split,
                   pattern);
    } DISPATCH_END;

    const double ms_pattern = asm_profile::ms_since(t_phase) - ms_snapshot;
    t_phase = asm_profile::tick();

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

    if (prof) {
        SimpleLog::message("[asm-prof] finalize: check %.1f ms  dedupe_rows "
                           "%.1f ms  pattern %.1f ms  transpose %.1f ms",
                           ms_check, ms_rows, ms_pattern,
                           asm_profile::ms_since(t_phase));
        asm_profile::report_widths("deduped_width", deduped);
    }
}

bool DynCSRMat::check() {
    Vec<Row> rows = this->rows;
    Vec<unsigned> flags = this->tmp_array;
    // Count the offending rows and report from the HOST. A bare device assert
    // here would abort with no message (device printf is dropped on an
    // assert-abort), which for a matrix-assembly fault leaves nothing to act on.
    DISPATCH_START(rows.size)
    [rows, flags] __device__(unsigned i) mutable {
        flags[i] = (rows[i].state == Row::OVERFLOW) ? 1u : 0u;
    } DISPATCH_END;
    const unsigned overflowed = kernels::sum_array(flags.data, rows.size);
    if (overflowed) {
        fprintf(stderr,
                "PPF FATAL: dynamic CSR overflow in %u of %u rows. The counting "
                "pass (dry_push) reserved fewer entries for those rows than the "
                "fill pass (push) then wrote, so the fill was refused to keep it "
                "from running into the neighboring row's blocks. The two passes "
                "must visit the same pairs: a predicate that differs between "
                "them, or a contact set that changed between counting and "
                "filling, will do this. The assembled matrix is incomplete, so "
                "the step is abandoned rather than solved against a silently "
                "wrong Hessian.\n",
                overflowed, rows.size);
        fflush(stderr);
        return false;
    }
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
                break;
            } else if (index.data[k] > j) {
                // Rows are sorted (exists() relies on this); no match past j.
                break;
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
        // NOTE: a false return here is NOT an error and must not be flagged as
        // one. Callers push a block speculatively and fall back to the dynamic
        // matrix when the pair has no fixed slot (contact assembly takes
        // exactly that route), so a miss is ordinary control flow: wiring it to
        // `status` fires on healthy scenes, headless.py included. That is why
        // `status` and check() are unwired. The hazard worth catching, an
        // element stencil never registered in builder.rs fixed_index_table, is
        // indistinguishable from this fallback at push time and has to be
        // caught where the pattern is built, not here.
    }
    return found;
}

// Accumulate `val` directly into a precomputed value slot, skipping the
// per-block row search push() performs. `slot` is the index into `value`
// returned by the host-side slot precomputation (builder.rs, matching the
// CVecVec row-major layout: offset[i] + position-of-j-in-row-i). atomicAdd
// keeps it safe to fold onto a slot another source might also touch. Used by
// the folded PDRD assembly to keep the per-body fill O(N^2) instead of O(N^3).
__device__ void FixedCSRMat::push_at(unsigned slot, const Mat3x3f &val) {
    float *ptr = (float *)(value.data + slot);
    for (unsigned ii = 0; ii < 9; ++ii) {
        float y = Map<const Vec9f>(val.data())[ii];
        if (y) {
            atomicAdd(ptr + ii, y);
        }
    }
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
