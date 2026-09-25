// File: intersect_policy.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Single source of truth for the intersection ALLOWANCE rule of issue #138:
// given the two sides of a pair, is this a pair the user allowed to intersect?
//
// One definition serves every backend. `contact/pair_filter.kernel.cpp`
// composes it into `contact_pair_admitted` and `intersect_pair_reported`, and
// the contact, CCD and intersection-scan visitors call those from the neutral
// bodies every target renders; `entrypoints/shim_contact.cpp` exposes
// the same body to the Rust host as `intersection_tolerated_abi`. A second copy
// would be a correctness hazard rather than a duplication nuisance: the gates
// must grant exactly the same set, and a scene that one backend reports and
// another does not is a defect with no local symptom. The rule is mirrored once
// more in Rust, as `VertexIntersectPolicy::tolerated` in
// ppf-cts-core/src/kernels/intersection.rs, for the build-time check that runs
// before any backend is reached.
//
// The full truth table is gated by
// crates/ppf-cts-solver/tests/kernels/intersect_allowance.cpp, which
// tests/kernel_gates.rs compiles with the host compiler and runs under
// `cargo test`. Every case is evaluated twice, once as written and once with
// the two sides exchanged, and both evaluations must match the expected
// verdict, so a policy that read only the first side fails it.
//
// AN ALLOWED PAIR IS OUT OF EVERY PASS, not only out of the report: contact
// assembles no barrier for it, the CCD line search does not filter the step
// against it, and the intersection scan does not report it, so the two elements
// pass through each other freely. `contact/pair_filter.kernel.cpp` states that
// composition once, as `contact_pair_admitted` and `intersect_pair_reported`,
// and `collider_intersection_allowed` for the static collision mesh.
//
// This routine is FLOAT-FREE and deliberately NOT a template. It reads only
// unsigned and bool values, so no floating-point type appears in it at all, and
// there is no scalar to template over. Both properties are worth stating rather
// than leaving to inspection: a `__host__ __device__` TEMPLATE is device code
// for every type it is instantiated with, so a host caller passing `double`
// emits a float64 device instantiation even though no kernel ever calls it,
// which `fp64_guard` in crates/ppf-cts-solver/build.rs fails the release build
// over. A plain non-templated function cannot do that.
//
// It takes PLAIN VALUES rather than a `VertexProp` for the same reason every
// other shared body does: the three compilers that read this file do not share
// a struct. `VertexProp` is declared in data.hpp, which the Metal shader
// compiler never sees (it is handed one concatenated string with no filesystem
// behind it), and the Metal side mirrors the struct by hand as `VertexProp`.
// Values cross that seam; a type does not. Each caller therefore reads the two
// fields off whichever struct its backend has, which is the same thing the
// address-space seam already requires of every other call site.

#ifndef CTS_INTERSECT_POLICY_HPP
#define CTS_INTERSECT_POLICY_HPP

// No include of its own, matching grain_pair.kernel.cpp and friction.kernel.cpp:
// the rule reads nothing but the values handed to it, and an include that
// survived to the Metal shader compiler would be a run-time compile error
// (check-shared-wiring.py rule 3). data.hpp includes THIS header rather than
// the reverse, so the two constants below have exactly one definition on every
// backend.

// The seam. `SM_INLINE` is `inline` under the Metal shader compiler, which
// defines it in the prologue (metal/shader_compiler.mm), so the fallback
// below is what nvcc and a host C++ compiler get.
//
// Two departures from the `#define SM_INLINE __device__ inline` its siblings
// use, and both are needed here rather than stylistic. The annotation carries
// `__host__` as well, because the rule is called from plain C++ and not only
// from a kernel: `entrypoints/shim_contact.cpp` compiles it into the host
// backend, and `tests/kernels/intersect_allowance.cpp` evaluates its truth
// table as a host program under `tests/kernel_gates.rs`. And the CUDA spelling
// is selected by `__CUDACC__` rather than emitted unconditionally, the same
// idiom the sibling intersect_core.hpp uses, so this header needs no include to
// define the annotations away for a host compiler and can therefore be included
// before common.hpp is.
#ifndef SM_INLINE
#if defined(__CUDACC__) || defined(__HIPCC__)
#define SM_INLINE __host__ __device__ inline
#else
#define SM_INLINE inline
#endif
#define ISECT_POLICY_UNDEF_INLINE
#endif

// VertexProp::object_index for a vertex whose source object is unknown.
// Deliberately not 0: two unknown indices must not compare equal, or every
// such pair would read as a self-intersection and take that allowance.
enum : unsigned { NO_OBJECT_INDEX = 0xFFFFFFFFu };

// VertexProp::group_index for a vertex that belongs to no group: the static
// collision mesh, which the add-on only ever builds from a STATIC group of its
// own. It matches no group, itself included, so a collision mesh counts as
// ANOTHER group from every object, just as NO_OBJECT_INDEX makes it another
// object.
enum : unsigned { NO_GROUP_INDEX = 0xFFFFFFFFu };

// Allow Existing Intersections' link table names two pools in one index: a
// dynamic vertex as itself, and a collision-mesh vertex with this bit set.
// `builder.rs` refuses a scene whose dynamic vertex count reaches it, so no
// dynamic index can carry the bit. Mirrored in data.rs.
enum : unsigned { START_LINK_COLLISION_VERTEX = 0x80000000u };

// VertexProp::intersect_policy bits. Mirrored in data.rs and in
// frontend/_scene_.py.
enum : unsigned char {
    INTERSECT_ALLOW_SELF = 1u << 0,
    INTERSECT_ALLOW_INTER_OBJECT = 1u << 1,
    INTERSECT_ALLOW_INTER_GROUP = 1u << 2,
};

namespace isect {

// Four allowances, and each side is described by its element's FIRST vertex
// (object and group identity and the material policy are per object, the
// convention `pdrd_body_index` and `collider` already use) plus the element's
// own precomputed "all N of my vertices are pinned by an allowing pin" bit.
//
// EITHER side is enough for the pin, inter-object and inter-group allowances,
// so flagging a garment covers it against the character it is fitted to
// without the character having to be flagged too. Self-intersection is asked
// of one object only, so there is one flag to read. Inter-group is the
// narrower of the two cross-object allowances: it covers a pair of objects
// only when they sit in different groups, so objects of one group still
// collide with each other.
SM_INLINE bool intersection_tolerated(unsigned a_object_index,
                                      unsigned a_group_index,
                                      unsigned char a_intersect_policy,
                                      unsigned b_object_index,
                                      unsigned b_group_index,
                                      unsigned char b_intersect_policy,
                                      bool a_pin_allows, bool b_pin_allows) {
    // A pin that allows intersections covers the geometry it holds, such as a
    // cuff pulled onto a wrist it starts inside. `either_dyn` already leaves a
    // pair whose BOTH sides are fully fix-pinned alone; the allowance is what
    // lets ONE pinned side be enough, and what extends it to pull pins, whose
    // hold is only as strong as their own force.
    if (a_pin_allows || b_pin_allows) {
        return true;
    }
    // NO_OBJECT_INDEX must not match itself: a pair of vertices whose objects
    // are both unknown is not evidence they share one.
    bool same_object =
        a_object_index != NO_OBJECT_INDEX && a_object_index == b_object_index;
    if (same_object) {
        return (a_intersect_policy & INTERSECT_ALLOW_SELF) != 0;
    }
    const unsigned char either = a_intersect_policy | b_intersect_policy;
    if ((either & INTERSECT_ALLOW_INTER_OBJECT) != 0) {
        return true;
    }
    // NO_GROUP_INDEX must not match itself either, for the same reason.
    bool same_group =
        a_group_index != NO_GROUP_INDEX && a_group_index == b_group_index;
    return !same_group && (either & INTERSECT_ALLOW_INTER_GROUP) != 0;
}

} // namespace isect

#ifdef ISECT_POLICY_UNDEF_INLINE
#undef SM_INLINE
#undef ISECT_POLICY_UNDEF_INLINE
#endif

#endif // CTS_INTERSECT_POLICY_HPP
