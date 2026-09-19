// File: dynamic_csr.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: plain C++ belonging to no backend, rendered into the
// three forms the three compilers read by ppf-cts-compute/seam/kernelgen.py.
// The two facts a compiler cannot infer are C++ attributes,
// `[[seam::device_fn]]` and `[[seam::host_device_fn]]` for the execution space
// and `[[seam::device]]` for the address space of a pointer parameter. Every
// row array here is a buffer the whole grid shares, so `device` is the only
// address space that appears. `Mat3x3f`, `fmath::abs` and the atomics arrive
// from whichever backend prologue is in scope:
// ppf-cts-compute/cuda/seam_cuda.cuh under nvcc, kernels/seam/seam_host.h on
// the host, and `kMslMacroSeam` in ppf-cts-compute/metal/shader_compiler.mm
// under MSL.
//
// TWO EXECUTION SPACES, AND THE SPLIT IS LOAD-BEARING.
//
// Most bodies here are pure arithmetic over a row's index and value arrays, and
// they are run from a kernel AND from the host: the entries below call them on
// the device, while tests/kernels/csr_row_dedupe.cpp and
// tests/kernels/csr_row_pattern.cpp call the same bodies on the host to compare
// them against a brute-force oracle. Those take `[[seam::host_device_fn]]`.
//
// The two transpose bodies reach a device-only atomic, so they cannot carry
// `__host__` under nvcc: marking them for both execution spaces makes nvcc
// reject the atomic call site rather than the caller. They take
// `[[seam::device_fn]]`, which is device-only under nvcc and nothing at all on
// the host and under MSL, neither of which has execution-space qualifiers.

// Position of `column` in the sorted `index[0, count)`, or `count` if it is
// not there. `count` is never a valid position, so it doubles as the absent
// marker.
//
// The carried column pattern is kept sorted (dynamic_csr_sort_pattern) so
// that a step locating a column walks the row's logarithm rather than the row.
[[seam::host_device_fn]] inline unsigned
dynamic_csr_find_sorted(const unsigned *index,
                            unsigned count, unsigned column) {
    unsigned low = 0;
    unsigned high = count;
    while (low < high) {
        const unsigned middle = low + ((high - low) >> 1);
        const unsigned probe = index[middle];
        if (probe == column) {
            return middle;
        }
        if (probe < column) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return count;
}

// Restore the heap property at `root` over `index[0, count)`.
[[seam::host_device_fn]] inline void
dynamic_csr_pattern_sift(unsigned *index, unsigned count,
                             unsigned root) {
    for (;;) {
        unsigned largest = root;
        const unsigned left = 2 * root + 1;
        const unsigned right = left + 1;
        if (left < count && index[left] > index[largest]) {
            largest = left;
        }
        if (right < count && index[right] > index[largest]) {
            largest = right;
        }
        if (largest == root) {
            return;
        }
        const unsigned temporary = index[root];
        index[root] = index[largest];
        index[largest] = temporary;
        root = largest;
    }
}

// Sort `index[0, count)` ascending, in place, with no scratch and no
// recursion.
//
// Most calls are handed an array that is already in order, because the step
// before left it that way. Heapsort has no adaptive path and would rediscover
// that at full cost, so spend one linear pass to find out instead. That is
// what makes it affordable to re-establish the ordering on every step rather
// than trust whoever produced it: it sorts for real only when the pattern came
// from somewhere that made no such promise, which today means a matrix
// restored from a saved state. The same n log n on every input, so the repair
// path has no input that degrades it.
[[seam::host_device_fn]] inline void
dynamic_csr_sort_pattern(unsigned *index, unsigned count) {
    if (count < 2) {
        return;
    }
    bool ordered = true;
    for (unsigned i = 1; i < count; ++i) {
        if (index[i - 1] > index[i]) {
            ordered = false;
            break;
        }
    }
    if (ordered) {
        return;
    }
    for (unsigned i = count / 2; i-- > 0;) {
        dynamic_csr_pattern_sift(index, count, i);
    }
    for (unsigned end = count; end-- > 1;) {
        const unsigned temporary = index[0];
        index[0] = index[end];
        index[end] = temporary;
        dynamic_csr_pattern_sift(index, end, 0);
    }
}

// Merge two ascending runs into `out`, which must not overlap either of them.
// The caller has them side by side inside one row and writes the result into
// the pattern buffer, which is separate storage, so there is nothing to alias.
[[seam::host_device_fn]] inline void
dynamic_csr_merge_runs(const unsigned *a, unsigned na,
                           const unsigned *b, unsigned nb,
                           unsigned *out) {
    unsigned i = 0;
    unsigned j = 0;
    unsigned w = 0;
    while (i < na && j < nb) {
        out[w++] = (a[i] <= b[j]) ? a[i++] : b[j++];
    }
    while (i < na) {
        out[w++] = a[i++];
    }
    while (j < nb) {
        out[w++] = b[j++];
    }
}

// A block counts as zero when every coefficient is within Eigen's dummy
// precision of it, matching Mat3x3f::isZero (linalg/smat.hpp), which is the
// test the CUDA path has always applied. Exact equality is NOT the same rule
// and drops fewer blocks, which changes the pattern the next step carries.
// Written as a negated comparison so a NaN coefficient reports as non-zero,
// exactly as isZero does.
[[seam::host_device_fn]] inline bool
dynamic_csr_block_is_zero(const Mat3x3f *value,
                              unsigned slot) {
    for (unsigned element = 0; element < 9; ++element) {
        if (!(fmath::abs(value[slot].m[element]) <= 1.0e-5f)) {
            return false;
        }
    }
    return true;
}

// Restore the heap property at `root` over the parallel arrays
// `index[0, count)` and `value[0, count)`, ordering on the index and carrying
// the block with it.
[[seam::host_device_fn]] inline void
dynamic_csr_pair_sift(unsigned *index,
                          Mat3x3f *value, unsigned count,
                          unsigned root) {
    for (;;) {
        unsigned largest = root;
        const unsigned left = 2 * root + 1;
        const unsigned right = left + 1;
        if (left < count && index[left] > index[largest]) {
            largest = left;
        }
        if (right < count && index[right] > index[largest]) {
            largest = right;
        }
        if (largest == root) {
            return;
        }
        const unsigned temporary_index = index[root];
        index[root] = index[largest];
        index[largest] = temporary_index;
        const Mat3x3f temporary_value = value[root];
        value[root] = value[largest];
        value[largest] = temporary_value;
        root = largest;
    }
}

// Sort `index[0, count)` ascending, carrying `value[0, count)` alongside. In
// place, no scratch, no recursion, and the same n log n on every input, which
// is the point: the counts this runs over are usually tiny but are not
// bounded.
[[seam::host_device_fn]] inline void
dynamic_csr_sort_pairs(unsigned *index,
                           Mat3x3f *value, unsigned count) {
    if (count < 2) {
        return;
    }
    for (unsigned i = count / 2; i-- > 0;) {
        dynamic_csr_pair_sift(index, value, count, i);
    }
    for (unsigned end = count; end-- > 1;) {
        const unsigned temporary_index = index[0];
        index[0] = index[end];
        index[end] = temporary_index;
        const Mat3x3f temporary_value = value[0];
        value[0] = value[end];
        value[end] = temporary_value;
        dynamic_csr_pair_sift(index, value, end, 0);
    }
}

// Compaction of one dynamic CSR row, in place. Drops blocks that stayed zero
// and folds repeated columns together, returning the surviving entry count.
//
// The slab handed in has two parts, and the boundary between them is what
// makes this cheap:
//
//   index[0, carried)   the pattern carried over from the previous step. A
//                       previous call to this function produced it, so its
//                       columns are already distinct.
//   index[carried, nnz) the columns appended this step. Row::push appends only
//                       after failing to find the column among the carried
//                       ones, so no appended column equals a carried one.
//
// The two parts are therefore disjoint, and duplicates can arise only among
// the appended entries: two threads that both miss the carried pattern each
// take their own slot, neither able to see the other. Comparing a carried
// entry against anything, or an appended entry against a carried one, can
// never match, so those comparisons are never made. Checking every entry
// against every earlier survivor costs the row's full width squared where this
// costs the count of columns that are NEW this step, and on a crowded row
// those differ by orders of magnitude. One thread walks a whole row, so that
// is the difference between a row costing microseconds and a row costing tens
// of seconds.
//
// Both produce the same set of columns, each holding the same sum, but NOT in
// the same order: the surviving carried entries keep the order they arrived
// in, and the appended ones come out sorted. Every reader of a row walks all
// of it, so the order is free to differ; what it buys is that the caller
// receives two ascending runs and can merge them rather than sort a row.
//
// The caller owns the invariant: `carried` must be the width of the pattern
// Row::push searched before appending. Passing a smaller value is safe but
// wasteful; passing a larger one would skip real duplicate checks.
//
// `appended_begin` reports where the surviving carried entries end and the
// surviving appended ones start.
[[seam::host_device_fn]] inline unsigned
dynamic_csr_finalize(unsigned *index,
                         Mat3x3f *value, unsigned nnz,
                         unsigned carried,
                         unsigned &appended_begin) {
    // THREAD SPACE, BECAUSE THE CALLER PASSES A LOCAL. MSL has separate address
    // spaces and `&local` is a `thread unsigned *`, which does not convert to a
    // `device unsigned *`; CUDA and the host have one address space and compile
    // either spelling, so only the shader compiler can tell them apart.
    if (carried > nnz) {
        carried = nnz;
    }
    unsigned out = 0;
    for (unsigned i = 0; i < carried; ++i) {
        if (!dynamic_csr_block_is_zero(value, i)) {
            if (out != i) {
                index[out] = index[i];
                value[out] = value[i];
            }
            ++out;
        }
    }
    // Everything from here on is an appended entry, so it only ever has to be
    // reconciled against the other appended entries. Gather the ones that
    // survive, order them, and fold equal neighbors: n log n in the count of
    // NEW columns, with no case where it degrades. Searching each one against
    // the ones already kept would be simpler to read, but it is quadratic in
    // that count, and the count is only small while the pattern is warm. The
    // step that starts from nothing (a fresh matrix meeting a scene that is
    // already in contact) has every column new at once, and that is exactly
    // where a row of tens of thousands would cost hundreds of millions of
    // comparisons on a single thread.
    const unsigned first_appended = out;
    appended_begin = first_appended;
    for (unsigned i = carried; i < nnz; ++i) {
        if (!dynamic_csr_block_is_zero(value, i)) {
            if (out != i) {
                index[out] = index[i];
                value[out] = value[i];
            }
            ++out;
        }
    }
    dynamic_csr_sort_pairs(index + first_appended, value + first_appended,
                               out - first_appended);
    unsigned kept = first_appended;
    for (unsigned k = first_appended; k < out; ++k) {
        if (kept > first_appended && index[kept - 1] == index[k]) {
            Mat3x3f sum = value[kept - 1];
            const Mat3x3f contribution = value[k];
            sum += contribution;
            value[kept - 1] = sum;
        } else {
            if (kept != k) {
                index[kept] = index[k];
                value[kept] = value[k];
            }
            ++kept;
        }
    }
    return kept;
}


[[seam::device_fn]] inline void dynamic_csr_count_transpose_row(
    unsigned row, const unsigned *index, unsigned begin,
    unsigned end, compute::atomic_uint_t *transpose_count) {
    for (unsigned slot = begin; slot < end; ++slot) {
        const unsigned column = index[slot];
        if (column != row) {
            compute::atomic_add(transpose_count + column, 1u);
        }
    }
}

[[seam::device_fn]] inline void dynamic_csr_scatter_transpose_row(
    unsigned row, const unsigned *index, unsigned begin,
    unsigned end, const unsigned *transpose_offset,
    compute::atomic_uint_t *cursor,
    unsigned *transpose_index,
    unsigned *transpose_value) {
    for (unsigned slot = begin; slot < end; ++slot) {
        const unsigned column = index[slot];
        if (column != row) {
            const unsigned output = transpose_offset[column] +
                                    compute::atomic_add(cursor + column, 1u);
            transpose_index[output] = row;
            transpose_value[output] = slot;
        }
    }
}

// THE TWO ROWS ABOVE, EACH AS ONE ELEMENT OF AN ELEMENT-WISE PASS, which is the
// shape the transpose is built in: a COUNT dispatch over `nrow` that atomically
// increments the per-column counter, a prefix sum over those counters, and a
// SCATTER dispatch over `nrow` that claims each column's next slot with another
// atomic increment. Neither of the two dispatches knows anything about a row but
// its own, so a thread index is the whole of their addressing and there is
// nothing between them to serialize.
//
// A COMPOSITION BODY RATHER THAN A WIDER OVERLOAD OF THE ROW BODY, which is
// the rule an entry conversion follows here: C++ would resolve the two on
// arity for a green build, and `check-shared-wiring.py` then refuses it,
// because its census keys a neutral body on its NAME and cannot tell two rows
// apart. The row bodies keep the range they take; the new function takes the
// new name.
//
// AND THESE ARE THE ONLY CALLERS THE TWO ROW BODIES HAVE. That matters because
// a body no call site reaches is a body nothing can contradict: it compiles,
// it renders, and no gate reads its answer.
//
// THE RUN THIS WALKS IS DATA, NOT THE THREAD INDEX. `row_offset[row]` to
// `row_offset[row + 1]` is read out of a buffer, so no `[[seam::bound]]` on the
// entry can cover the slots it reaches; the bound that does is the host's, that
// the last entry of `row_offset` is the width of `index`, asserted in
// `DynCsrMat::to_flat_into` where both are on the host at once.
// THE TWO ELEMENT-WISE HALVES OF THE TRANSPOSE, ONE ROW PER THREAD.
//
// The counter and the claim cursor are `[[seam::pod(4)]]` because
// `compute::atomic_uint_t` is a different type on each target, `unsigned` under
// nvcc and on the host and `atomic_uint` under MSL, and a record addresses a
// pointee by its WIDTH. Neither is a gather: a column is reached through the
// row's stored index and not through the thread index, so both stay base
// pointers and the body indexes them itself.
//
// NEITHER CARRIES A `[[seam::bound]]`, and that is a property of the data
// rather than an omission. The run a row walks is `row_offset[row]` to
// `row_offset[row + 1]`, read out of a buffer, so the slot a thread touches is
// not a function of its own index and no in-kernel bound can be stated against
// it. What bounds it is the host's assertion that the last entry of
// `row_offset` is the width of `index`, made where both are on the host at
// once, in `DynCsrMat::to_flat_into`.
[[seam::entry(row)]]
[[seam::device_fn]] inline void dyn_count_transpose_pass(
    const unsigned *row_offset,
    const unsigned *index,
    compute::atomic_uint_t *transpose_count, unsigned row) {
    dynamic_csr_count_transpose_row(row, index, row_offset[row],
                                        row_offset[row + 1], transpose_count);
}

[[seam::entry(row)]]
[[seam::device_fn]] inline void dyn_scatter_transpose_pass(
    const unsigned *row_offset,
    const unsigned *index,
    const unsigned *transpose_offset,
    compute::atomic_uint_t *cursor,
    unsigned *transpose_index,
    unsigned *transpose_value, unsigned row) {
    dynamic_csr_scatter_transpose_row(row, index, row_offset[row],
                                          row_offset[row + 1],
                                          transpose_offset, cursor,
                                          transpose_index, transpose_value);
}


// ---------------------------------------------------------------------------
// THE MATRIX ITSELF, AS SIX ELEMENT-WISE PASSES OVER FLAT DEVICE STORAGE.
//
// The five bodies above are the per-row ALGORITHMS: bisect, sort, compact,
// merge. THE STORAGE THEY RUN OVER IS FLAT: one row-offset array, one flat
// column array and one flat block array, so a row is an OFFSET and a dispatch
// over rows is an ordinary element-wise pass. A matrix held as one `Vec` per row
// could not be dispatched that way, because the pass would have to name an array
// of pointers, and every row would then be reachable only one at a time from the
// host.
//
// THE ORDER OF THE PASSES IS NOT A CHOICE. A step is
//
//   1. begin    per row: order the carried pattern, seed the reserve with it
//   2. dry push per contribution: reserve one slot for a column not carried
//   3. scan     the reserves into per-row slab offsets
//   4. seed     per row: lay the carried pattern into the slab, zero the slab
//   5. push     per contribution: fold onto a carried slot or claim a new one
//   6. compact  per row: drop zero blocks, fold repeats, leave two runs
//   7. scan     the surviving widths into the next step's pattern offsets
//   8. emit     per row: merge the two runs into the pattern, copy out the row
//
// Steps 1 and 2 are `start_rebuild_buffer` and `Row::dry_push`, step 3 and 4 are
// `finish_rebuild_buffer`, step 5 is `Row::push`, and steps 6 to 8 are
// `DynCSRMat::finalize`. WHAT MAKES THE COUNTING PASS NECESSARY rather than
// merely tidy is that a row's slab has to be sized before anything writes into
// it, and a row whose fill runs past its slab writes into the NEXT row's blocks,
// which is a Hessian the solver would then treat as valid. The fill refuses that
// write and counts the refusal instead, and the host turns a non-zero count into
// a loud failure.
//
// WHY THE VALUE ARRAY IS NAMED WITH TWO DIFFERENT POINTEES. The fill accumulates
// onto a carried slot with a float atomic, so it names the blocks as
// `compute::atomic_float_t`, nine per entry; the compaction moves whole blocks
// and names them as `Mat3x3f`. Those are the same bytes and the same buffer. One
// `Mat3x3f *` cast to `float *` would cover both roles, and MSL does not admit
// that: `atomic_float` is a distinct type there, not a qualifier on a float. Two
// records naming one buffer is what the seam offers instead of a cast.

// One row's carried pattern, put in order, and its reserve opened at the width
// of that pattern.
//
// THE ORDERING IS ESTABLISHED WHERE THE STEP STARTS USING THE PATTERN, not
// where the previous step produced one. `dynamic_csr_merge_runs` leaves it
// ascending, so this is one linear pass in the ordinary case; what it covers is
// the pattern a checkpoint restore writes directly, which has been through no
// merge and which both the counting and the filling pass would otherwise
// bisect unordered. That is `start_rebuild_buffer`'s reason for calling
// `sort_pattern` and it is unchanged here.
//
// THE RESERVE OPENS AT THE CARRIED WIDTH because a carried column keeps its
// slot whether or not anything lands on it this step: the slab holds the whole
// pattern first and the counting pass adds one slot per column that is new.
[[seam::entry(row)]]
[[seam::device_fn]] inline void
dyn_row_begin_pass(const unsigned *fixed_offset,
                   unsigned *fixed_index,
                   unsigned *reserve, unsigned row) {
    const unsigned begin = fixed_offset[row];
    const unsigned carried = fixed_offset[row + 1] - begin;
    dynamic_csr_sort_pattern(fixed_index + begin, carried);
    reserve[row] = carried;
}

// One contribution, counted rather than stored.
//
// `Row::dry_push`, and the guard is `DynCSRMat::dry_push`'s: only the upper
// triangle is stored, so a contribution whose row is past its column belongs to
// the entry on the other side of the diagonal and is not counted here.
//
// THE COUNTING AND THE FILLING PASS MUST VISIT THE SAME CONTRIBUTIONS IN THE
// SAME WAY, which is what makes the reserve exact rather than approximate: a
// column found in the carried pattern reserves nothing and folds onto its
// existing slot, and every other contribution reserves one slot and takes one,
// duplicates among the new columns included, since two contributions that both
// miss the pattern cannot see each other and each takes its own.
[[seam::entry(element)]]
[[seam::device_fn]] inline void
dyn_dry_push_pass(const unsigned *fixed_offset,
                  const unsigned *fixed_index,
                  const unsigned *push_row,
                  const unsigned *push_column,
                  compute::atomic_uint_t *reserve,
                  unsigned element) {
    const unsigned row = push_row[element];
    const unsigned column = push_column[element];
    if (row > column) {
        return;
    }
    const unsigned begin = fixed_offset[row];
    const unsigned carried = fixed_offset[row + 1] - begin;
    if (dynamic_csr_find_sorted(fixed_index + begin, carried, column) ==
        carried) {
        compute::atomic_add(reserve + row, 1u);
    }
}

// One row's slab, opened for the fill.
//
// `finish_rebuild_buffer`'s second and third dispatches: the carried pattern is
// laid into the head of the row's slab with zero blocks, so a contribution to a
// column that was in contact last step lands in place, and the row's width opens
// at that pattern's width.
//
// THE WHOLE SLAB IS ZEROED AND NOT ONLY THE CARRIED PREFIX. The fill accumulates
// an appended block with a float atomic rather than storing it outright, because
// a plain store into `compute::atomic_float_t` has no spelling the three
// prologues share, so an appended slot has to start at zero for the accumulate
// to be that store. The slots past the reserve are not
// read by anything, and the ones inside it are exactly the ones a fill may
// claim.
[[seam::entry(row)]]
[[seam::device_fn]] inline void
dyn_row_seed_pass(const unsigned *fixed_offset,
                  const unsigned *fixed_index,
                  const unsigned *dyn_offset,
                  unsigned *dyn_index,
                  float *dyn_value,
                  unsigned *head, unsigned row) {
    const unsigned begin = fixed_offset[row];
    const unsigned carried = fixed_offset[row + 1] - begin;
    const unsigned base = dyn_offset[row];
    const unsigned reserved = dyn_offset[row + 1] - base;
    for (unsigned slot = 0; slot < carried; ++slot) {
        dyn_index[base + slot] = fixed_index[begin + slot];
    }
    for (unsigned element = 0; element < 9 * reserved; ++element) {
        dyn_value[9 * base + element] = 0.0f;
    }
    head[row] = carried;
}

// One contribution, stored.
//
// `Row::push`. The carried run is the row's first `carried` slots and it is
// sorted, so a column already in it is found by bisection and its block folded
// onto the slot it already has; anything else is appended, and the search having
// failed is what makes an appended column provably distinct from every carried
// one. `dynamic_csr_finalize` skips exactly the comparisons that fact rules out,
// so appending without first failing this search would leave duplicates it does
// not look for.
//
// THE REFUSAL IS NOT AN OPTIMIZATION AND MUST NOT BE SOFTENED. The row's slab
// was sized by the counting pass, so in a consistent step `offset` never reaches
// `reserved`; if the two passes ever disagree, the write would run past this
// row's slab and into the next row's blocks, and the Hessian that came out would
// still look assembled. The write is refused and counted, and the host reports
// the count and abandons the step.
[[seam::entry(element)]]
[[seam::device_fn]] inline void
dyn_push_pass(const unsigned *fixed_offset,
              const unsigned *dyn_offset,
              unsigned *dyn_index,
              compute::atomic_float_t *dyn_value,
              const unsigned *push_row,
              const unsigned *push_column,
              const float *push_block,
              compute::atomic_uint_t *head,
              compute::atomic_uint_t *refused,
              unsigned element) {
    const unsigned row = push_row[element];
    const unsigned column = push_column[element];
    if (row > column) {
        return;
    }
    const unsigned carried = fixed_offset[row + 1] - fixed_offset[row];
    const unsigned base = dyn_offset[row];
    const unsigned reserved = dyn_offset[row + 1] - base;
    const unsigned slot =
        dynamic_csr_find_sorted(dyn_index + base, carried, column);
    if (slot != carried) {
        for (unsigned component = 0; component < 9; ++component) {
            const float term = push_block[9 * element + component];
            if (term != 0.0f) {
                compute::atomic_add(dyn_value + 9 * (base + slot) + component,
                                    term);
            }
        }
        return;
    }
    const unsigned offset = compute::atomic_add(head + row, 1u);
    if (offset >= reserved) {
        compute::atomic_add(refused, 1u);
        return;
    }
    dyn_index[base + offset] = column;
    for (unsigned component = 0; component < 9; ++component) {
        const float term = push_block[9 * element + component];
        if (term != 0.0f) {
            compute::atomic_add(dyn_value + 9 * (base + offset) + component,
                                term);
        }
    }
}

// One row, compacted.
//
// `Row::finalize`. `dynamic_csr_finalize` drops the blocks that stayed zero,
// folds repeated columns, and reports where the surviving carried run ends and
// the surviving appended run begins. The width it returns replaces the row's
// head, and the scan that follows turns those widths into the next step's
// pattern offsets.
//
// THE DISJOINTNESS CHECK IS THE ONE THAT MUST STAY. Its violation leaves the row
// NUMERICALLY correct, because the duplicate's carried entry goes unpushed and
// drops as zero, so nothing else would report it and the only symptom is a step
// that costs tens of seconds instead of a fraction of one. It is affordable
// because the carried run is sorted: one bisection per appended column.
//
// SEARCH THE PATTERN THE FILL ACTUALLY SEARCHED, NOT THE COMPACTED SURVIVORS.
// A column the fill kept missing is pushed nowhere, so its carried slot keeps
// the zero the seed wrote and the compaction drops it before `split` exists;
// searching below `split` would look for the offending column in the one place
// it is guaranteed not to be and would pass on exactly the failure this is
// written to catch. `fixed_index` still holds that pattern here, because the
// merge that rewrites it runs in a later dispatch.
[[seam::entry(row)]]
[[seam::device_fn]] inline void
dyn_row_compact_pass(const unsigned *fixed_offset,
                     const unsigned *fixed_index,
                     const unsigned *dyn_offset,
                     unsigned *dyn_index,
                     Mat3x3f *dyn_value,
                     unsigned *head,
                     unsigned *split, unsigned row,
                     DiagHandle diag) {
    const unsigned begin = fixed_offset[row];
    const unsigned carried = fixed_offset[row + 1] - begin;
    const unsigned base = dyn_offset[row];
    const unsigned nnz = head[row];
    unsigned appended_begin = 0;
    const unsigned kept = dynamic_csr_finalize(
        dyn_index + base, dyn_value + base, nnz, carried, appended_begin);
    for (unsigned k = appended_begin; k < kept; ++k) {
        const unsigned column = dyn_index[base + k];
        DIAG_ASSERT4(diag,
                     dynamic_csr_find_sorted(fixed_index + begin, carried,
                                             column) == carried,
                     static_cast<float>(row), static_cast<float>(column),
                     static_cast<float>(carried), static_cast<float>(kept));
    }
    split[row] = appended_begin;
    head[row] = kept;
}

// One row, handed on.
//
// The last dispatch of `DynCSRMat::finalize`, plus the copy the linear operator
// needs. Two things come out of a compacted row:
//
//   the PATTERN the next step carries, which is the row's two ascending runs
//   merged, so the step picking it up bisects it without sorting a row;
//
//   the row's entries in the CONTIGUOUS form the sparse matvec reads. A matvec
//   that walked a row through its own head would tolerate the gap between the
//   row's width and its reserve; `fixed_csr_apply_row` walks `offset[row]` to
//   `offset[row + 1]` instead, which admits no gap, so the row is copied into a
//   flat array whose offsets are the widths scanned. That is a difference in the
//   STORAGE and not in the arithmetic, and it is what a fused assembly removes
//   along with the staging.
//
// ONLY THE INDICES ARE MERGED. The values that accompany the pattern are zeroed
// when the next step lays it back into a slab, so there is nothing to permute
// alongside them, and the row's own arrays are left as they are because every
// reader of those walks the whole row.
[[seam::entry(row)]]
[[seam::device_fn]] inline void
dyn_row_emit_pass(const unsigned *dyn_offset,
                  const unsigned *dyn_index,
                  const float *dyn_value,
                  const unsigned *head,
                  const unsigned *split,
                  unsigned *pattern,
                  unsigned *flat_index,
                  float *flat_value, unsigned row) {
    const unsigned base = dyn_offset[row];
    const unsigned out = head[row];
    const unsigned width = head[row + 1] - out;
    const unsigned boundary = split[row];
    dynamic_csr_merge_runs(dyn_index + base, boundary,
                           dyn_index + base + boundary, width - boundary,
                           pattern + out);
    for (unsigned slot = 0; slot < width; ++slot) {
        flat_index[out + slot] = dyn_index[base + slot];
        for (unsigned component = 0; component < 9; ++component) {
            flat_value[9 * (out + slot) + component] =
                dyn_value[9 * (base + slot) + component];
        }
    }
}
