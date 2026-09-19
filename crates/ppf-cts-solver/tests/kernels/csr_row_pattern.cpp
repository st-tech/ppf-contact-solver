// File: csr_row_pattern.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// A HOST TEST, DRIVEN BY `tests/kernel_gates.rs`.
//
// Gate for the sorted carried pattern (csrmat/dynamic_csr.kernel.cpp).
//
// Row::push and Row::dry_push locate a column by bisection, which is only
// correct if the pattern finalize() hands them is sorted. This pins both ends
// of that contract: dynamic_csr_sort_pattern really sorts (and loses
// nothing), and dynamic_csr_find_sorted agrees with a linear scan on every
// key, present or absent. dynamic_csr_merge_runs is covered too, in the
// overlapping arrangement finalize uses.
//
// Each check runs against an independent oracle: an std::sort for the pattern
// and a linear scan for the search.

// The bodies under test are neutral kernel sources, written once and rendered
// for every target from one parse, so this host compile reads the same bytes
// nvcc and the Metal shader compiler read. Calling them from the host is what
// the pure-arithmetic bodies' `[[seam::host_device_fn]]` permits; nothing has
// to be redefined here. data.hpp comes first: it pulls in seam/seam.hpp, which
// is where every SM_ spelling the shared header names is defined, and it
// declares Mat3x3f, which that header names but does not include.
#include "data.hpp"

#include "csrmat/dynamic_csr.kernel.cpp"
#include <algorithm>
#include <cstdio>
#include <vector>

// THE ORACLES DECIDE THIS TEST, not a second run of the same body. Running a
// neutral body once more inside a `<<<1, 32>>>` launch with only thread 0
// acting would compare one body against itself, since there is a single parsed
// body rather than a host copy beside a device copy. What decides whether the
// bisecting row search is right is the pair below: an `std::sort` for the
// pattern and a linear scan for the search.

struct Lcg {
    unsigned long long s;
    unsigned next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (unsigned)(s >> 33);
    }
    unsigned below(unsigned n) { return n ? next() % n : 0u; }
};

static bool check(bool cond, const char *what, unsigned n) {
    if (!cond) {
        printf("  FAIL %s (n=%u)\n", what, n);
    }
    return cond;
}

// `in` is the pattern as finalize would hand it over. Verifies that the sort
// orders the pattern, that nothing was gained or lost, and that bisection
// agrees with a linear scan for every key that is present and a spread that is
// not.
static bool run_case(std::vector<unsigned> in, const char *tag) {
    const unsigned n = (unsigned)in.size();

    std::vector<unsigned> expect = in;
    std::sort(expect.begin(), expect.end());

    std::vector<unsigned> host = in;
    dynamic_csr_sort_pattern(host.data(), n);
    bool ok = check(host == expect, "host sort", n);

    // Every key that is present must be found at a slot holding it; absent keys
    // must come back as n. Compared against a linear scan, which is exactly the
    // search the solver used before it started bisecting.
    std::vector<unsigned> keys;
    for (unsigned i = 0; i < n; ++i) {
        keys.push_back(host[i]);
    }
    Lcg rng{0xDEADBEEFCAFEF00DULL};
    for (unsigned i = 0; i < 64; ++i) {
        keys.push_back(rng.below(4 * (n + 4)));
    }
    const unsigned nkeys = (unsigned)keys.size();

    std::vector<unsigned> want(nkeys);
    for (unsigned k = 0; k < nkeys; ++k) {
        unsigned pos = n;
        for (unsigned i = 0; i < n; ++i) {
            if (host[i] == keys[k]) {
                pos = i;
                break;
            }
        }
        want[k] = pos;
    }

    for (unsigned k = 0; k < nkeys; ++k) {
        const unsigned got =
            dynamic_csr_find_sorted(host.data(), n, keys[k]);
        const bool present = want[k] != n;
        if (present) {
            // Duplicates would make the exact slot ambiguous; the pattern never
            // holds them, but assert on the value rather than the slot so this
            // stays a statement about membership.
            ok = check(got < n && host[got] == keys[k], "host find present",
                       n) &&
                 ok;
        } else {
            ok = check(got == n, "host find absent", n) && ok;
        }
    }

    if (!ok) {
        printf("  case: %s\n", tag);
    }
    return ok;
}

// Exercise dynamic_csr_merge_runs the way finalize does: the two runs side
// by side in one array, the result written into separate storage.
static bool run_merge_case(std::vector<unsigned> a, std::vector<unsigned> b,
                           const char *tag) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    const unsigned na = (unsigned)a.size();
    const unsigned nb = (unsigned)b.size();

    std::vector<unsigned> expect;
    expect.reserve(na + nb);
    expect.insert(expect.end(), a.begin(), a.end());
    expect.insert(expect.end(), b.begin(), b.end());
    std::sort(expect.begin(), expect.end());

    std::vector<unsigned> src;
    src.insert(src.end(), a.begin(), a.end());
    src.insert(src.end(), b.begin(), b.end());
    std::vector<unsigned> out(na + nb + 1, 0xFFFFFFFFu);
    dynamic_csr_merge_runs(src.data(), na, src.data() + na, nb, out.data());
    out.resize(na + nb);
    bool ok = check(out == expect, "host merge", na + nb);

    if (!ok) {
        printf("  merge case: %s (na=%u nb=%u)\n", tag, na, nb);
    }
    return ok;
}

int main() {
    Lcg rng{0x9E3779B97F4A7C15ULL};
    bool ok = true;
    unsigned cases = 0;

    // Degenerate sizes.
    for (unsigned n : {0u, 1u, 2u, 3u}) {
        std::vector<unsigned> v;
        for (unsigned i = 0; i < n; ++i) {
            v.push_back(n - i);
        }
        ok = run_case(v, "tiny reversed") && ok;
        ++cases;
    }

    // Already sorted, reversed, and all-equal: heapsort has no adaptive path,
    // but these are where an off-by-one in the sift bounds would show.
    for (unsigned n : {5u, 16u, 17u, 64u, 255u, 256u, 1000u}) {
        std::vector<unsigned> asc, desc, same;
        for (unsigned i = 0; i < n; ++i) {
            asc.push_back(2 * i);
            desc.push_back(2 * (n - i));
            same.push_back(7);
        }
        ok = run_case(asc, "ascending") && ok;
        ok = run_case(desc, "descending") && ok;
        ok = run_case(same, "all equal") && ok;
        cases += 3;
    }

    // The shape the solver actually produces: a sorted carried run followed by
    // a short unsorted one, the columns appended this step.
    for (unsigned carried : {0u, 1u, 40u, 500u, 5000u}) {
        for (unsigned appended : {0u, 1u, 7u, 60u}) {
            std::vector<unsigned> v;
            for (unsigned i = 0; i < carried; ++i) {
                v.push_back(2 * i);
            }
            for (unsigned i = 0; i < appended; ++i) {
                v.push_back(2 * rng.below(4 * (carried + appended) + 8) + 1);
            }
            ok = run_case(v, "carried + appended") && ok;
            ++cases;
        }
    }

    // Random, and one row wider than this scene reaches.
    for (unsigned n : {31u, 333u, 4096u}) {
        std::vector<unsigned> v;
        for (unsigned i = 0; i < n; ++i) {
            v.push_back(rng.below(10 * n));
        }
        ok = run_case(v, "random") && ok;
        ++cases;
    }
    {
        std::vector<unsigned> v;
        for (unsigned i = 0; i < 40000; ++i) {
            v.push_back(rng.next());
        }
        ok = run_case(v, "40k wide") && ok;
        ++cases;
    }

    // merge_runs, in the overlapping arrangement finalize uses. The lopsided
    // shapes matter most: a long carried run with a handful of new columns is
    // the steady state, and an empty run on either side is the boundary where
    // the write head meets the read head immediately.
    {
        const unsigned shapes[][2] = {
            {0, 0},   {1, 0},    {0, 1},     {1, 1},    {8, 0},
            {0, 8},   {64, 1},   {1, 64},    {1000, 3}, {3, 1000},
            {97, 97}, {512, 40}, {5000, 17}, {40000, 220},
        };
        for (auto &s : shapes) {
            std::vector<unsigned> a, b;
            // Disjoint pools, matching how carried and appended columns relate.
            for (unsigned i = 0; i < s[0]; ++i) {
                a.push_back(2 * rng.below(4 * s[0] + 8));
            }
            for (unsigned i = 0; i < s[1]; ++i) {
                b.push_back(2 * rng.below(4 * s[1] + 8) + 1);
            }
            ok = run_merge_case(a, b, "disjoint") && ok;
            ++cases;
        }
        // Interleaved and fully overlapping ranges, so the merge actually has
        // to alternate rather than concatenate.
        for (unsigned n : {5u, 200u, 3000u}) {
            std::vector<unsigned> a, b;
            for (unsigned i = 0; i < n; ++i) {
                a.push_back(rng.below(2 * n));
                b.push_back(rng.below(2 * n));
            }
            ok = run_merge_case(a, b, "interleaved") && ok;
            ++cases;
        }
    }

    printf("%s: %u cases\n", ok ? "PASS" : "FAIL", cases);
    return ok ? 0 : 1;
}
