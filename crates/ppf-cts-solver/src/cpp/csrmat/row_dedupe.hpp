// File: row_dedupe.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef ROW_DEDUPE_HPP
#define ROW_DEDUPE_HPP

#include "../data.hpp"

// Restore the heap property at `root` over the parallel arrays `index[0, n)`
// and `value[0, n)`, ordering on the index and carrying the block with it.
__host__ __device__ inline void pair_sift(unsigned *index, Mat3x3f *value,
                                          unsigned n, unsigned root) {
    for (;;) {
        unsigned largest = root;
        const unsigned l = 2 * root + 1;
        const unsigned r = l + 1;
        if (l < n && index[l] > index[largest]) {
            largest = l;
        }
        if (r < n && index[r] > index[largest]) {
            largest = r;
        }
        if (largest == root) {
            return;
        }
        const unsigned ti = index[root];
        index[root] = index[largest];
        index[largest] = ti;
        const Mat3x3f tv = value[root];
        value[root] = value[largest];
        value[largest] = tv;
        root = largest;
    }
}

// Sort `index[0, n)` ascending, carrying `value[0, n)` alongside. In place, no
// scratch, no recursion, and the same n log n on every input, which is the
// point: the counts this runs over are usually tiny but are not bounded.
__host__ __device__ inline void sort_pairs(unsigned *index, Mat3x3f *value,
                                           unsigned n) {
    if (n < 2) {
        return;
    }
    for (unsigned i = n / 2; i-- > 0;) {
        pair_sift(index, value, n, i);
    }
    for (unsigned end = n; end-- > 1;) {
        const unsigned ti = index[0];
        index[0] = index[end];
        index[end] = ti;
        const Mat3x3f tv = value[0];
        value[0] = value[end];
        value[end] = tv;
        pair_sift(index, value, end, 0);
    }
}

// Compaction of one dynamic CSR row, in place. Drops blocks that stayed zero
// and folds repeated columns together, returning the surviving entry count.
//
// The slab handed in has two parts, and the boundary between them is what makes
// this cheap:
//
//   index[0, carried)   the pattern carried over from the previous step. A
//                       previous call to this function produced it, so its
//                       columns are already distinct.
//   index[carried, nnz) the columns appended this step. Row::push appends only
//                       after failing to find the column among the carried
//                       ones, so no appended column equals a carried one.
//
// The two parts are therefore disjoint, and duplicates can arise only among the
// appended entries: two threads that both miss the carried pattern each take
// their own slot, neither able to see the other. Comparing a carried entry
// against anything, or an appended entry against a carried one, can never
// match, so those comparisons are never made. The obvious alternative, checking
// every entry against every earlier survivor, costs the row's full width
// squared where this costs the count of columns that are NEW this step, and on
// a crowded row those differ by orders of magnitude. One thread walks a whole
// row, so that is the difference between a row costing microseconds and a row
// costing tens of seconds.
//
// Both produce the same set of columns, each holding the same sum, but NOT in
// the same order: the surviving carried entries keep the order they arrived in,
// and the appended ones come out sorted. Every reader of a row walks all of it,
// so the order is free to differ; what it buys is that the caller receives two
// ascending runs and can merge them rather than sort a row.
//
// The caller owns the invariant: `carried` must be the width of the pattern
// Row::push searched before appending. Passing a smaller value is safe but
// wasteful; passing a larger one would skip real duplicate checks.
//
// `appended_begin` reports where the surviving carried entries end and the
// surviving appended ones start.
__host__ __device__ inline unsigned row_dedupe(unsigned *index, Mat3x3f *value,
                                               unsigned nnz, unsigned carried,
                                               unsigned *appended_begin) {
    if (carried > nnz) {
        carried = nnz;
    }
    unsigned out = 0;
    for (unsigned i = 0; i < carried; ++i) {
        if (!value[i].isZero()) {
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
    *appended_begin = first_appended;
    for (unsigned i = carried; i < nnz; ++i) {
        if (!value[i].isZero()) {
            if (out != i) {
                index[out] = index[i];
                value[out] = value[i];
            }
            ++out;
        }
    }
    sort_pairs(index + first_appended, value + first_appended,
               out - first_appended);
    unsigned kept = first_appended;
    for (unsigned k = first_appended; k < out; ++k) {
        if (kept > first_appended && index[kept - 1] == index[k]) {
            value[kept - 1] += value[k];
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

#endif
