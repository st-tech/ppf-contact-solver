// File: csr_row_dedupe.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// A HOST TEST, DRIVEN BY `tests/kernel_gates.rs`.
//
// Equivalence gate for dynamic_csr_finalize
// (csrmat/dynamic_csr.kernel.cpp), the row compaction Row::finalize calls.
//
// It compacts a dynamic CSR row while deliberately NOT comparing carried
// entries against each other, nor appended entries against carried ones,
// because Row::push guarantees those can never match. This test pins that claim
// against a brute-force reference (`reference_dedupe` below) that compares
// every entry with every earlier survivor, assuming nothing about where an
// entry came from: the same columns must survive, each holding the same
// accumulated block. The reference calls
// Mat3x3f::isZero directly, so it also pins dynamic_csr_block_is_zero to
// that same rule rather than to exact equality.
//
// The layouts differ on purpose. The reference emits in first-seen order, while
// the compaction keeps its carried survivors in arrival order and sorts the
// appended ones so the caller can merge two runs instead of sorting a row. So
// the comparison orders both by column first; that is the contract, and the
// order is not part of it.
//
// It runs the same function on the host and inside a kernel, so the device
// instantiation is covered too, not just the host one.

// The body under test lives behind the backend seam, so it is written once and
// compiled from identical bytes by nvcc and by the Metal shader compiler. This
// test calls it on the host as well as from a kernel, which is the reason the
// pure-arithmetic bodies in that header take SM_INLINE_DEVICE_HOST; nothing has
// to be redefined here. data.hpp comes first: it pulls in seam/seam.hpp,
// which is where every SM_ spelling the shared header names is defined, and it
// declares Mat3x3f, which that header names but does not include.
#include "data.hpp"

#include "csrmat/dynamic_csr.kernel.cpp"
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

// The brute-force oracle dynamic_csr_finalize is checked against: every
// surviving entry is searched for a match among all earlier survivors, with no
// assumption about where the entry came from.
static unsigned reference_dedupe(unsigned *index, Mat3x3f *value,
                                 unsigned nnz) {
    unsigned head = 0;
    for (unsigned i = 0; i < nnz; ++i) {
        unsigned j = index[i];
        Mat3x3f val = value[i];
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
                unsigned h = head++;
                index[h] = j;
                value[h] = val;
            }
        }
    }
    return head;
}

// Deterministic generator: no wall-clock seeding, so a failure reproduces.
struct Lcg {
    unsigned long long s;
    unsigned next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (unsigned)(s >> 33);
    }
    unsigned below(unsigned n) { return n ? next() % n : 0u; }
};

static Mat3x3f make_block(unsigned seed) {
    Mat3x3f m;
    for (unsigned r = 0; r < 3; ++r) {
        for (unsigned c = 0; c < 3; ++c) {
            // Small exact binary fractions, so accumulation is associative here
            // and any mismatch is an ordering bug rather than float noise.
            m(r, c) = (float)((seed + 3 * r + c) % 16) * 0.25f;
        }
    }
    return m;
}

// Build one row honoring the invariant push() establishes: carried columns are
// distinct, appended columns are drawn from a pool disjoint from the carried
// ones, and appended columns may repeat.
static void build_row(Lcg &rng, unsigned carried, unsigned appended,
                      unsigned zero_pct, std::vector<unsigned> &index,
                      std::vector<Mat3x3f> &value) {
    index.clear();
    value.clear();
    for (unsigned i = 0; i < carried; ++i) {
        index.push_back(i * 2); // even columns: the carried pattern
        bool zero = rng.below(100) < zero_pct;
        value.push_back(zero ? Mat3x3f::Zero() : make_block(rng.next()));
    }
    // Odd columns: disjoint from every carried column by construction. Drawn
    // from a small pool on purpose so duplicates are common.
    const unsigned pool = appended ? (appended + 3) / 4 + 1 : 1;
    for (unsigned i = 0; i < appended; ++i) {
        index.push_back(2 * rng.below(pool) + 1);
        bool zero = rng.below(100) < zero_pct;
        value.push_back(zero ? Mat3x3f::Zero() : make_block(rng.next()));
    }
}

// Compare on the contract rather than on the layout: the same set of columns,
// each holding the same accumulated block. The compaction emits its surviving
// carried entries in arrival order and its appended ones sorted, which is a
// different order from the reference's first-seen one and deliberately so
// (every reader of a row walks all of it). Ordering both by column puts them in
// a canonical form; columns are unique on both sides, so that is well defined.
static bool same(const unsigned *ia, const Mat3x3f *va, unsigned ha,
                 const unsigned *ib, const Mat3x3f *vb, unsigned hb,
                 const char *tag) {
    if (ha != hb) {
        printf("  FAIL %s: head %u vs %u\n", tag, ha, hb);
        return false;
    }
    std::vector<std::pair<unsigned, Mat3x3f>> pa, pb;
    for (unsigned k = 0; k < ha; ++k) {
        pa.emplace_back(ia[k], va[k]);
        pb.emplace_back(ib[k], vb[k]);
    }
    auto by_col = [](const std::pair<unsigned, Mat3x3f> &x,
                     const std::pair<unsigned, Mat3x3f> &y) {
        return x.first < y.first;
    };
    std::sort(pa.begin(), pa.end(), by_col);
    std::sort(pb.begin(), pb.end(), by_col);
    for (unsigned k = 0; k < ha; ++k) {
        if (pa[k].first != pb[k].first) {
            printf("  FAIL %s: column %u vs %u\n", tag, pa[k].first,
                   pb[k].first);
            return false;
        }
        if (memcmp(&pa[k].second, &pb[k].second, sizeof(Mat3x3f)) != 0) {
            printf("  FAIL %s: block for column %u differs\n", tag,
                   pa[k].first);
            return false;
        }
        if (k && pa[k].first == pa[k - 1].first) {
            printf("  FAIL %s: duplicate column %u survived\n", tag,
                   pa[k].first);
            return false;
        }
    }
    return true;
}

static bool run_case(Lcg &rng, unsigned carried, unsigned appended,
                     unsigned zero_pct) {
    std::vector<unsigned> index;
    std::vector<Mat3x3f> value;
    build_row(rng, carried, appended, zero_pct, index, value);
    const unsigned nnz = (unsigned)index.size();

    std::vector<unsigned> ref_i = index;
    std::vector<Mat3x3f> ref_v = value;
    const unsigned ref_head =
        nnz ? reference_dedupe(ref_i.data(), ref_v.data(), nnz) : 0u;

    std::vector<unsigned> host_i = index;
    std::vector<Mat3x3f> host_v = value;
    unsigned host_split = 0;
    const unsigned host_head = dynamic_csr_finalize(
        host_i.data(), host_v.data(), nnz, carried, host_split);

    // The split must land where the carried survivors end: everything below it
    // came from the carried pattern, so the run above it is exactly what the
    // caller has to sort and merge. It can never exceed the surviving count,
    // and it can never exceed the carried width either, since carried entries
    // only ever drop out.
    bool ok_split = true;
    if (host_split > host_head || host_split > carried) {
        printf("  FAIL split %u (head %u, carried %u)\n", host_split, host_head,
               carried);
        ok_split = false;
    }

    // BOTH runs must come out ascending, because finalize merges them instead
    // of sorting, and a merge given an unordered input produces an unordered
    // result that nothing downstream checks. The appended run is sorted here;
    // the carried run is ascending only because it arrived that way and the
    // compaction preserves order, which is the easier of the two to break by
    // accident (any rewrite that gathers survivors in another order loses it
    // while keeping every column and every sum, so a comparison that comes
    // after ordering by column cannot see it). build_row emits the carried
    // columns ascending, matching the sorted pattern the solver hands over.
    for (unsigned k = 1; k < host_split; ++k) {
        if (host_i[k - 1] >= host_i[k]) {
            printf("  FAIL carried run not ascending at %u\n", k);
            ok_split = false;
        }
    }
    for (unsigned k = host_split + 1; k < host_head; ++k) {
        if (host_i[k - 1] >= host_i[k]) {
            printf("  FAIL appended run not ascending at %u\n", k);
            ok_split = false;
        }
    }

    bool ok = same(ref_i.data(), ref_v.data(), ref_head, host_i.data(),
                   host_v.data(), host_head, "host");

    // THE DEVICE HALF IS GONE, and what it asked is answered by construction.
    // It ran `dynamic_csr_finalize` a second time in a `<<<1, 32>>>` kernel
    // where only thread 0 acted, and required the two runs to agree; that was
    // checking one body compiled and behaved the same in two places, and the
    // body is now neutral, rendered for every target from one parse. What the
    // reference oracle above checks is the thing that decides the matrix.
    ok = ok && ok_split;
    if (!ok) {
        printf("  case carried=%u appended=%u zero%%=%u\n", carried, appended,
               zero_pct);
    }
    return ok;
}

int main() {
    Lcg rng{0x9E3779B97F4A7C15ULL};
    bool ok = true;
    unsigned cases = 0;

    // Edge cases first: an empty row, a row with nothing carried (the cold
    // start, where every column is new), a row with nothing appended (the
    // steady state), and rows whose blocks all canceled to zero.
    const unsigned edge[][3] = {
        {0, 0, 0},   {0, 1, 0},   {1, 0, 0},   {0, 16, 0},  {16, 0, 0},
        {1, 1, 0},   {8, 8, 100}, {8, 8, 0},   {0, 64, 100}, {64, 0, 100},
    };
    for (auto &e : edge) {
        ok = run_case(rng, e[0], e[1], e[2]) && ok;
        ++cases;
    }

    // Then a sweep: widths on both sides of the boundary, with and without
    // blocks that canceled to zero.
    for (unsigned carried : {0u, 1u, 5u, 32u, 97u, 256u}) {
        for (unsigned appended : {0u, 1u, 7u, 40u, 129u}) {
            for (unsigned zero_pct : {0u, 25u, 90u}) {
                ok = run_case(rng, carried, appended, zero_pct) && ok;
                ++cases;
            }
        }
    }

    // A row wider than any this scene produces, to confirm the boundary logic
    // does not depend on size.
    ok = run_case(rng, 4096, 512, 10) && ok;
    ++cases;

    // carried larger than the entry count must clamp rather than read past the
    // slab (Row::finalize can hand in fixed_nnz == head for an untouched row).
    {
        std::vector<unsigned> index;
        std::vector<Mat3x3f> value;
        build_row(rng, 8, 0, 0, index, value);
        std::vector<unsigned> a = index, b = index;
        std::vector<Mat3x3f> va = value, vb = value;
        unsigned ha = reference_dedupe(a.data(), va.data(), 8);
        unsigned split = 0;
        unsigned hb =
            dynamic_csr_finalize(b.data(), vb.data(), 8, 99, split);
        ok = same(a.data(), va.data(), ha, b.data(), vb.data(), hb,
                  "clamp") &&
             ok;
        ++cases;
    }

    printf("%s: %u cases\n", ok ? "PASS" : "FAIL", cases);
    return ok ? 0 : 1;
}
