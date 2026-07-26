// File: row_pattern.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef ROW_PATTERN_HPP
#define ROW_PATTERN_HPP

// This header needs nothing but `unsigned`, so it keeps itself compilable by a
// plain C++ front end the same way linalg/smat.hpp does.
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

// The column pattern a dynamic CSR row carries from one step to the next is
// kept sorted, so that assembling a step finds a column by bisection rather
// than by scanning the row.
//
// Why it is worth keeping sorted: both assembly passes ask the same question
// once per block they contribute, "is this column already in the pattern?".
// Scanning answers it in time proportional to the row's width. That is
// invisible on an ordinary row (a handful of columns) but not when a coarse
// collider meets a finely sampled deformable: there a single row reaches tens
// of thousands of columns, every push into it walks half of them, and the
// traffic, not the arithmetic, becomes the assembly cost. Bisection turns a
// walk over the whole row into a walk over its logarithm.
//
// Sorting is cheap here for a reason worth stating: this pattern is indices
// ONLY. The values that accompany it are zeroed when the pattern is read back
// in (finish_rebuild_buffer), so there is no value ordering to keep in step,
// nothing to permute alongside, and the elements being moved are 4 bytes each.
//
// In the ordinary case the pattern arrives already ordered, because finalize
// merged it that way, and sort_pattern returns after one pass. It sorts for real
// only when the pattern came from somewhere that made no such promise, which
// today means a matrix restored from a saved state. Heapsort suits that: no
// recursion, no scratch, and the same n log n whether it is reordering a whole
// row or confirming one, so the repair path has no input that degrades it.

// Restore the heap property at `root` over `a[0, n)`.
__host__ __device__ inline void pattern_sift(unsigned *a, unsigned n,
                                             unsigned root) {
    for (;;) {
        unsigned largest = root;
        const unsigned l = 2 * root + 1;
        const unsigned r = l + 1;
        if (l < n && a[l] > a[largest]) {
            largest = l;
        }
        if (r < n && a[r] > a[largest]) {
            largest = r;
        }
        if (largest == root) {
            return;
        }
        const unsigned t = a[root];
        a[root] = a[largest];
        a[largest] = t;
        root = largest;
    }
}

// Sort `a[0, n)` ascending, in place, with no scratch and no recursion.
__host__ __device__ inline void sort_pattern(unsigned *a, unsigned n) {
    if (n < 2) {
        return;
    }
    // Most calls are handed an array that is already in order, because the
    // step before left it that way. Heapsort has no adaptive path and would
    // rediscover that at full cost, so spend one linear pass to find out
    // instead. This is also what makes it affordable to re-establish the
    // ordering on every step rather than trusting whoever produced it.
    bool ordered = true;
    for (unsigned i = 1; i < n; ++i) {
        if (a[i - 1] > a[i]) {
            ordered = false;
            break;
        }
    }
    if (ordered) {
        return;
    }
    for (unsigned i = n / 2; i-- > 0;) {
        pattern_sift(a, n, i);
    }
    for (unsigned end = n; end-- > 1;) {
        const unsigned t = a[0];
        a[0] = a[end];
        a[end] = t;
        pattern_sift(a, end, 0);
    }
}

// Merge two ascending runs into `out`, which must not overlap either of them.
// The caller has them side by side inside one row and writes the result into
// the pattern buffer, which is separate storage, so there is nothing to alias.
__host__ __device__ inline void merge_runs(const unsigned *a, unsigned na,
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

// Position of `key` in the sorted `a[0, n)`, or `n` if it is not there. `n` is
// never a valid position, so it doubles as the absent marker.
__host__ __device__ inline unsigned find_sorted(const unsigned *a, unsigned n,
                                                unsigned key) {
    unsigned lo = 0;
    unsigned hi = n;
    while (lo < hi) {
        const unsigned mid = lo + ((hi - lo) >> 1);
        const unsigned v = a[mid];
        if (v == key) {
            return mid;
        }
        if (v < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return n;
}

#endif
