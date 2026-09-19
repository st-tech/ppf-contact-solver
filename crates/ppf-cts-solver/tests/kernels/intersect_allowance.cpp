// File: intersect_allowance.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// A HOST TEST, DRIVEN BY `tests/kernel_gates.rs`.
//
// IT RUNS EACH CASE ON THE HOST ONLY, AND THAT COVERS THE RULE. Running each
// case a second time inside a kernel would be checking that one header compiled
// and behaved the same in two places. The header is a NEUTRAL BODY rendered for
// every target from one parse, with `check-shared-wiring.py` asserting the
// renderings exist and are reached, so that question is answered by
// construction.
//
// WHAT IT COVERS is the whole of what the policy decides: the truth table, case
// by case, and that exchanging the two sides never changes a verdict.

// Truth-table gate for the intersection allowance rule of issue #138,
// `isect::intersection_tolerated` in
// `src/kernels/contact/intersect_policy.hpp`.
//
// The rule decides which intersecting pairs are REPORTED, so both of its
// verdicts are expensive to get wrong in opposite directions: granting an
// allowance that was not asked for hides a real tangle and lets a run finish
// with geometry the solver never resolved, and withholding one that was asked
// for aborts a run the user configured to continue.
//
// Every case is evaluated twice on the host, once as written and once with the
// two sides exchanged.
//
//   * HOST AND DEVICE. One definition serves every backend: the rule lives in
//     `src/kernels/contact/intersect_policy.hpp` and is compiled by nvcc, by
//     the Metal shader compiler and by a host C++ compiler. A divergence
//     between those compilations is possible in principle and would otherwise
//     surface only as two backends disagreeing about a scene, with no local
//     symptom in either. Three toolchains make that risk wider rather than
//     narrower.
//   * SIDES EXCHANGED. The pin and inter-object allowances are worded "either
//     side opts in", and the four testers do not agree on an order: the
//     face-edge tester passes the face first, the edge-edge tester the
//     higher-indexed edge, the point-point tester the higher-indexed vertex,
//     and the collision-mesh tester the dynamic edge. A verdict that depended
//     on which side arrives first would therefore hold in some testers and
//     not in others. The table deliberately carries no same-object pair whose
//     two sides hold DIFFERENT policy bytes, because that configuration cannot
//     occur: `intersect_policy` is resolved per object, so two vertices of one
//     object always carry the same byte. Asserting a verdict on it would pin
//     an arbitrary choice between reading side A's bits and reading both.
//
// The pin allowance the rule takes is the ELEMENT's precomputed "all N of my
// vertices are pinned by an allowing pin" bit, which arrives as its own
// argument rather than being read off a vertex: the two sides of a pair are
// described here by exactly the four values the rule reads, so there is no
// vertex field for a mistaken implementation to reach for instead.
//
// Usage (from this directory):
//   make test-intersect-allow

#include "contact/intersect_policy.hpp"
#include <cstdio>

// Two distinct object identities, plus zero, which is a REAL object id and not
// a sentinel: the frontend numbers objects from 0, and the unknown-object
// marker is deliberately 0xFFFFFFFF so that it cannot collide with one.
constexpr unsigned OBJ_A = 3u;
constexpr unsigned OBJ_B = 7u;
constexpr unsigned OBJ_ZERO = 0u;

static_assert(NO_OBJECT_INDEX != OBJ_ZERO,
              "object id 0 is a real object, so the unknown-object sentinel "
              "must not be 0");
static_assert((INTERSECT_ALLOW_SELF & INTERSECT_ALLOW_INTER_OBJECT) == 0,
              "the self and inter-object allowances must be independent bits");

// Shorthands that keep the table readable.
constexpr unsigned char NONE = 0u;
constexpr unsigned char SELF = INTERSECT_ALLOW_SELF;
constexpr unsigned char INTER = INTERSECT_ALLOW_INTER_OBJECT;
constexpr unsigned char BOTH =
    INTERSECT_ALLOW_SELF | INTERSECT_ALLOW_INTER_OBJECT;

// Everything the rule reads, as plain data with no pointers, so one array
// serves the host loop and the kernel.
struct Inputs {
    unsigned a_object;
    unsigned b_object;
    unsigned char a_policy;
    unsigned char b_policy;
    bool a_pin;
    bool b_pin;
};

struct Case {
    const char *name;
    Inputs in;
    bool expect_tolerated;
};

inline Inputs exchanged(const Inputs &in) {
    Inputs out;
    out.a_object = in.b_object;
    out.b_object = in.a_object;
    out.a_policy = in.b_policy;
    out.b_policy = in.a_policy;
    out.a_pin = in.b_pin;
    out.b_pin = in.a_pin;
    return out;
}

inline bool evaluate(const Inputs &in) {
    return isect::intersection_tolerated(in.a_object, in.a_policy,
                                             in.b_object, in.b_policy,
                                             in.a_pin, in.b_pin);
}

// The trailing bool is the expected verdict: true = tolerated, meaning the
// pair is NOT reported.
static const Case CASES[] = {
    // ---------------------------------------------- nothing opted in
    {"same object, no allowance",
     {OBJ_A, OBJ_A, NONE, NONE, false, false}, false},
    {"different objects, no allowance",
     {OBJ_A, OBJ_B, NONE, NONE, false, false}, false},

    // ---------------------------------------------- the self allowance
    {"same object, allow-self",
     {OBJ_A, OBJ_A, SELF, SELF, false, false}, true},
    {"same object, allow-inter only",
     {OBJ_A, OBJ_A, INTER, INTER, false, false}, false},
    {"same object, both allowances",
     {OBJ_A, OBJ_A, BOTH, BOTH, false, false}, true},
    {"object id 0 against itself, allow-self",
     {OBJ_ZERO, OBJ_ZERO, SELF, SELF, false, false}, true},

    // ---------------------------------------------- the inter-object one
    {"different objects, allow-inter on A only",
     {OBJ_A, OBJ_B, INTER, NONE, false, false}, true},
    {"different objects, allow-inter on B only",
     {OBJ_A, OBJ_B, NONE, INTER, false, false}, true},
    {"different objects, allow-inter on both",
     {OBJ_A, OBJ_B, INTER, INTER, false, false}, true},
    {"different objects, allow-self on A only",
     {OBJ_A, OBJ_B, SELF, NONE, false, false}, false},
    {"different objects, allow-self on both",
     {OBJ_A, OBJ_B, SELF, SELF, false, false}, false},

    // ---------------------------------------------- the pin allowance
    {"same object, pin bit on A",
     {OBJ_A, OBJ_A, NONE, NONE, true, false}, true},
    {"same object, pin bit on B",
     {OBJ_A, OBJ_A, NONE, NONE, false, true}, true},
    {"different objects, pin bit on A",
     {OBJ_A, OBJ_B, NONE, NONE, true, false}, true},
    {"different objects, pin bit on B",
     {OBJ_A, OBJ_B, NONE, NONE, false, true}, true},
    {"different objects, both pin bits",
     {OBJ_A, OBJ_B, NONE, NONE, true, true}, true},
    {"pin bit grants what the wrong allowance does not",
     {OBJ_A, OBJ_B, SELF, SELF, false, true}, true},

    // ---------------------------------------------- both sides unknown
    //
    // The sentinel exists so that two unknown identities do NOT read as one
    // shared object. A regression here tolerates every pair in a session
    // directory written without bin/object_vert.bin, which is silent: the run
    // completes and reports nothing.
    {"both objects unknown, no allowance",
     {NO_OBJECT_INDEX, NO_OBJECT_INDEX, NONE, NONE, false, false}, false},
    {"both objects unknown, allow-self on both",
     {NO_OBJECT_INDEX, NO_OBJECT_INDEX, SELF, SELF, false, false}, false},
    {"both objects unknown, allow-inter on A",
     {NO_OBJECT_INDEX, NO_OBJECT_INDEX, INTER, NONE, false, false}, true},
    {"both objects unknown, pin bit on B",
     {NO_OBJECT_INDEX, NO_OBJECT_INDEX, NONE, NONE, false, true}, true},

    // ---------------------------------------------- one side unknown
    //
    // The shape the collision-mesh tester passes: a dynamic edge against the
    // rest-pose static mesh, whose side carries NO_OBJECT_INDEX and an empty
    // policy. The pair is inter-object by construction, so the inter-object
    // rule is the one that must apply.
    {"unknown A against real B, no allowance",
     {NO_OBJECT_INDEX, OBJ_B, NONE, NONE, false, false}, false},
    {"unknown A against real B, allow-inter on B",
     {NO_OBJECT_INDEX, OBJ_B, NONE, INTER, false, false}, true},
    {"real A against unknown B, allow-inter on A",
     {OBJ_A, NO_OBJECT_INDEX, INTER, NONE, false, false}, true},
    {"unknown A against real B, allow-self on both",
     {NO_OBJECT_INDEX, OBJ_B, SELF, SELF, false, false}, false},
    {"object id 0 against unknown, no allowance",
     {OBJ_ZERO, NO_OBJECT_INDEX, NONE, NONE, false, false}, false},
    {"object id 0 against unknown, allow-inter on the real side",
     {OBJ_ZERO, NO_OBJECT_INDEX, INTER, NONE, false, false}, true},
};

static const char *verdict(bool tolerated) {
    return tolerated ? "tolerated" : "reported";
}

int main() {
    const unsigned n = sizeof(CASES) / sizeof(CASES[0]);

    int failures = 0;
    printf("=== intersection allowance truth table (%u cases) ===\n", n);
    for (unsigned i = 0; i < n; ++i) {
        const bool expect = CASES[i].expect_tolerated;
        const bool got = evaluate(CASES[i].in);
        // EXCHANGING THE TWO SIDES MUST NOT CHANGE THE VERDICT. "Either side
        // opts in" is the rule for the pin and inter-object allowances, so a
        // policy that read only the first side would answer this table
        // correctly and be wrong on half of every real scene.
        const bool got_swapped = evaluate(exchanged(CASES[i].in));
        const bool ok = got == expect && got_swapped == expect;
        printf("  %-6s %-50s %s\n", ok ? "[ok]" : "[FAIL]", CASES[i].name,
               verdict(expect));
        if (!ok) {
            printf("         expected %s; got %s, sides exchanged: %s\n",
                   verdict(expect), verdict(got), verdict(got_swapped));
            ++failures;
        }
    }

    printf("%s\n", failures == 0
                       ? "intersection allowance truth table passed"
                       : "intersection allowance truth table FAILED");
    return failures == 0 ? 0 : 1;
}
