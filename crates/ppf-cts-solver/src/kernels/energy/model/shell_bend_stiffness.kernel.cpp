// File: shell_bend_stiffness.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The scalar that multiplies a shell hinge's dihedral force and Hessian: the
// resolution-independent, density-normalized Discrete Shells bending
// coefficient, plus the orientation-dependent weight that mixes the isotropic
// stiffness with the warp and weft ones.
//
// It is separated from shell_bend.kernel.cpp (the per-hinge dihedral math)
// because it is what the two backends have to agree on BEFORE either evaluates
// the hinge: the same three quantities scale the elastic block AND the lagged
// Rayleigh damping block built on top of it, so a backend that formed this
// scalar its own way would differ in both at once.

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only one of the three
// compilers accepts. ppf-cts-compute/seam/kernelgen.py renders it into the .cu, the .metal
// and the .cpp that nvcc, the Metal shader compiler and a host C++ compiler
// read. The two facts a compiler cannot infer are C++ attributes:
// `[[seam::device_fn]]` is the execution space, and `[[seam::thread]]` is the
// address space of a pointer parameter. MSL requires the second on every
// pointer and reference; CUDA and the host have one address space and are
// handed the same declarations with it removed.
//
// ONE DEPENDENCY, and it is named here so this body stands on its own:
// tests/test_bend_aniso.cpp compiles it with a plain C++ compiler and no
// data.hpp. The include below, seam/seam.hpp, is what supplies the `fmath::`
// arithmetic, which is the whole of what these three bodies need beyond float.
// The Metal backend resolves the include itself, splicing the segments it needs
// in dependency order and supplying `fmath::` from its own prologue.
#include "../../seam/seam.hpp"

// A hinge's dimensionless bending stiffness, as the isotropic value plus
// whatever the two directional ones contribute at this edge's orientation:
//
//   k(psi) = bend + warp * sin^2(psi) + weft * cos^2(psi)
//
// `psi` is the angle between the hinge's shared edge and the UV X (warp) axis,
// and the caller supplies sin^2 directly (HingeProp::uv_edge_sin2) so no
// trigonometry is needed here. sin^2 pairs with WARP and cos^2 with weft,
// which is the one counter-intuitive part and the only place the convention
// is pinned: a hinge folds ABOUT its shared edge, so the surface curves
// ACROSS it. An edge lying along warp therefore bends the sheet in the weft
// sense and picks up `weft`, while an edge along weft bends the warp fibers
// and picks up `warp`. Read at the fiber level it is the natural reading:
// `warp` is how stiff the warp fibers are to being bent.
//
// The form is additive and linear in sin^2, and two properties follow from
// that. Every term is non-negative for non-negative inputs, so the result
// cannot go negative at any orientation and there is no material-stability
// condition to enforce or to explain. And the directional pair ADDS to `bend`
// rather than redistributing it, so `bend` alone sets the isotropic stiffness
// at every orientation: the calibrated value (`BEND_SCALE` below, from
// `calibration/cusick_drape`) is what the presets carry, and they need no
// directional term to hold it.
//
// The cost is that the 45 degree bias stiffness is not independently settable;
// it is bend + (warp + weft)/2. Real woven fabric is floppiest on the bias,
// below both axes, which this cannot express.
//
// With warp and weft at their 0.0f defaults the result is `bend` exactly, for
// any orientation and including the no-UV sentinel, so an isotropic scene is
// bit-identical to one authored with no anisotropy available at all.
[[seam::device_fn]] inline float shell_bend_directional(float bend,
                                                           float warp,
                                                           float weft,
                                                           float sin2) {
    if (sin2 < 0.0f) {
        return bend; // no UV, so no direction is defined
    }
    return bend + warp * sin2 + weft * (1.0f - sin2);
}

// Areal density (kg/m^2) averaged over the hinge's four vertices, from the
// per-vertex mass and area arrays indexed in the hinge's OWN order.
//
// The order is load-bearing: this is an fp32 running sum of up to four terms,
// so a caller that passed the four vertices permuted would get a different
// float. The hinge dispatch therefore reads it in the mesh order
// (data.mesh.mesh.hinge[i]) and not in the (2,1,0,3) order the dihedral math
// wants, which is why the permutation happens after this and not before.
//
// A vertex with no area contributes nothing and is not counted, so a hinge
// whose four vertices all have zero area yields 0.0f and the stiffness below
// vanishes with it.
[[seam::device_fn]] inline float shell_bend_areal_density(
    const float *mass, const float *area) {
    float areal_density = 0.0f;
    int count = 0;
    for (int k = 0; k < 4; ++k) {
        const float a = area[k];
        if (a > 0.0f) {
            areal_density += fmath::div(mass[k], a);
            ++count;
        }
    }
    if (count > 0) {
        areal_density = fmath::div(areal_density, static_cast<float>(count));
    }
    return areal_density;
}

// The per-hinge bending stiffness.
//
// Resolution independence: the convergent per-hinge stiffness is
// k = B * |e| / h_e, proportional to |e|^2 / (A1 + A2), which is scale
// invariant under mesh refinement because |e|^2 / area stays O(1), so the bent
// shape does not depend on resolution. (Convergence: the integrated mean
// curvature on an edge is |e|*theta and the edge dual area is (A1+A2)/3, so
// the sum converges to int B*kappa^2 dA; Grinspun et al. 2003,
// Tamstorf-Grinspun 2013, Wang 2023.) A bare |e| factor in place of |e|^2/area
// would shrink as |e| -> |e|/s under refinement, so finer cloth would droop
// more.
//
// Density normalization: B = bend * areal_density is the density-normalized
// flexural rigidity, which is what makes the bent shape invariant to density
// and matches the rest of the solver, where the membrane and the rod bend
// already scale by mass. `bend` alone then sets the bent shape and density
// stays a free knob, which is what lets a very light fabric (silk) be mixed
// with dense bodies without conditioning trouble. Areal density (mass/area),
// not raw vertex mass, is what keeps the bend mesh independent.
//
// `area` is the COMBINED rest area of the two incident triangles (A1 + A2).
// The guard covers near-degenerate triangles, where area -> 0 would blow the
// stiffness up.
//
// Folding the directional weight in here rather than into the force is what
// covers the lagged damping Hessian by construction, since that scales by the
// same stiffness. All three of bend, warp and weft are non-negative (scene.rs)
// and so are sin^2 and cos^2, so the result can never turn negative and flip
// the sign of the hinge block: SPD-by-assembly holds.
[[seam::device_fn]] inline float shell_bend_stiffness(
    float bend, float warp, float weft, float sin2, float length, float area,
    float areal_density) {
    // Sets only the numeric range of the user-facing `bend` parameter, and NOT
    // the resolution independence, which is the |e|^2 / area factor beside it.
    // It is calibrated (calibration/cusick_drape) so the fabric presets and
    // existing scenes land where they are meant to at the usual mesh density.
    const float BEND_SCALE = 1.28e-5f;
    const float bend_directional =
        shell_bend_directional(bend, warp, weft, sin2);
    return (area > 1e-12f) ? BEND_SCALE * bend_directional *
                                 fmath::div(length * length, area) *
                                 areal_density
                           : 0.0f;
}

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `shell_bend_stiffness_entry` shim a host C++ compiler
// compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// Seven element gathers into one element scatter. Every input is a per-hinge
// float, so each is read at the thread index and the returned stiffness is
// written at the same one: the generated statement is the launcher's
// `stiffness[i] = shell_bend_stiffness(bend[i], ..., areal_density[i])`
// character for character.
// THE SAME STIFFNESS, READING ITS MATERIAL AND GEOMETRY OFF THE RECORDS.
//
// The six per-hinge floats the body above takes are read off records that are
// already resident: `bend`, `bend_warp` and `bend_weft` are `HingeParam` fields
// and `uv_edge_sin2`, `length` and `area` are `HingeProp` fields. Flattening
// them into six host-filled arrays instead would upload all six every assembly
// pass: measured on `drape` at 3 frames, that is 72 host-to-device calls
// carrying 66 MB.
//
// THE EXCLUSION GATE IS EXPLICIT BECAUSE NO SEEDED ARRAY CARRIES IT. The
// dispatch covers EVERY hinge, and the records hand each one its real
// stiffness, so a hinge that is `fixed`, a `collider` or carries a `kind` whose
// bit 0 marks it as lying on a SOLID's surface has to be turned away by this
// test. Returning 0.0 is what the caller's active-list scan reads.
//
// `hinge_param` IS NOT GATHERED, being indexed per MATERIAL rather than per
// hinge, so the body reaches it through the record's own `param_index` and owes
// the thread-space copy an entry would otherwise have made.
[[seam::device_fn]] inline float shell_bend_stiffness_from_records(
    const HingeProp &prop,
    const HingeParam *hinge_param, unsigned kind,
    float areal_density) {
    if (prop.fixed || prop.collider || (kind & 1u) != 0u) {
        return 0.0f;
    }
    const HingeParam param = hinge_param[prop.param_index];
    return shell_bend_stiffness(param.bend, param.bend_warp, param.bend_weft,
                                prop.uv_edge_sin2, prop.length, prop.area,
                                areal_density);
}

// The stiffness and the Rayleigh damping coefficient beside it, from the same
// two records.
//
// THE DAMPING IS GATED BY THE STIFFNESS, which is the whole reason it is
// computed here rather than gathered separately: the damping term this
// coefficient scales is built from the stiffness-scaled start-of-step Hessian,
// so a hinge that carries no bending stiffness contributes nothing to damp and
// its coefficient must read zero. The two questions have one answer and one
// pass rather than a host loop over an active list.
[[seam::device_fn]] inline void shell_bend_stiffness_and_damping(
    const HingeProp &prop,
    const HingeParam *hinge_param, unsigned kind,
    float areal_density, float *stiffness,
    float *damping, unsigned element) {
    const float k =
        shell_bend_stiffness_from_records(prop, hinge_param, kind, areal_density);
    stiffness[element] = k;
    damping[element] = k > 0.0f ? hinge_param[prop.param_index].bend_damping : 0.0f;
}

[[seam::entry(count, element)]] void shell_bend_stiffness_and_damping(
    const HingeProp *prop,
    const HingeParam *hinge_param,
    const unsigned *kind,
    const float *areal_density,
    float *stiffness, float *damping,
    unsigned element, unsigned count);

[[seam::entry(count)]] void shell_bend_stiffness(
    const float *bend,
    const float *warp,
    const float *weft,
    const float *sin2,
    const float *length,
    const float *area,
    const float *areal_density,
    float *stiffness,
    unsigned count);

// The areal density with the hinge's four vertices already gathered.
//
// A COMPOSITION AND NOTHING ELSE. It rebuilds the two four-element arrays the
// body above takes and calls it, in the order it has always been called in, so
// what it adds is a declaration an entry point can be generated from rather
// than any arithmetic. THE ORDER IS LOAD-BEARING and the body says why: the sum
// is fp32 over up to four terms, so a permuted hinge gives a different float.
// These eight parameters are the hinge's MESH order, slot 0 through slot 3 of
// each array, which is the order the entry's index list reads them in.
//
// IT TAKES EIGHT SCALARS RATHER THAN TWO ARRAYS because that is the shape an
// indirect gather hands a body: `[[seam::through]]` passes the N elements at
// the element's own slots as N arguments, one per slot. The arrays are rebuilt
// here, in thread space, so the body above is unchanged and keeps its other
// callers.
[[seam::device_fn]] inline float shell_bend_areal_density_gathered(
    float mass0, float mass1, float mass2, float mass3, float area0,
    float area1, float area2, float area3) {
    const float mass[4] = {mass0, mass1, mass2, mass3};
    const float area[4] = {area0, area1, area2, area3};
    return shell_bend_areal_density(mass, area);
}

// The same average with the four vertices read as RECORDS.
//
// TWO FLAT ARRAYS COLLAPSE INTO ONE THE DEVICE ALREADY HOLDS. The mass and the
// area are `VertexProp` fields, so a host loop that copied them into two staged
// arrays every pass and uploaded both was carrying two columns of a record the
// device holds whole. The bound below still applies: a slot is DATA rather than
// the thread index, and Metal returns 0.0 for an out-of-bounds read rather than
// faulting.
[[seam::device_fn]] inline float shell_bend_areal_density_from_records(
    const VertexProp &prop0,
    const VertexProp &prop1,
    const VertexProp &prop2,
    const VertexProp &prop3) {
    return shell_bend_areal_density_gathered(prop0.mass, prop1.mass, prop2.mass,
                                             prop3.mass, prop0.area, prop1.area,
                                             prop2.area, prop3.area);
}

[[seam::entry(count)]] void shell_bend_areal_density_from_records(
    [[seam::through]] const VertexProp *vertex_prop,
    [[seam::indices(4)]] const unsigned *hinge,
    [[seam::bound]] unsigned vertex_count,
    float *areal_density,
    unsigned count);

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `shell_bend_areal_density_gathered_entry` shim a host C++
// compiler compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// AN INDIRECT GATHER, AND THE BOUND IS WHAT THE ENTRY FORM ADDS. The two
// per-vertex arrays are read at the hinge's own four slots, so a slot is DATA
// rather than the thread index and the `[[seam::count]]` guard says nothing
// about it. Metal returns 0.0 for an out-of-bounds read rather than faulting,
// which would turn a corrupt hinge index into a plausible stiffness instead of
// a stopped run, so `vertex_count` is declared and the entry checks every slot
// against it. A caller that subscripted the two arrays itself would carry no
// such check.
//
// BOTH ARRAYS ARE READ AT THE SAME SLOTS, which is what lets one index list
// serve the mass and the area. The hinge is read in MESH order here, not in the
// (2, 1, 0, 3) order the dihedral math wants, for the fp32 summation reason the
// body above states; the permutation happens after this pass and not before.
[[seam::entry(count)]] void shell_bend_areal_density_gathered(
    [[seam::through]] const float *vertex_mass,
    [[seam::through]] const float *vertex_area,
    [[seam::indices(4)]] const unsigned *hinge,
    [[seam::bound]] unsigned vertex_count,
    float *areal_density,
    unsigned count);
