// File: vertex_force.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. `[[seam::device_fn]]` is the
// execution space; `[[seam::thread]]` and `[[seam::device]]` are the address
// spaces of a reference into the caller's own frame and of a pointer into a
// buffer, which MSL requires and which is also what the code means.
//
// `Pull`, `TorqueVertex` and `TorqueGroup` are template parameters for the
// reason `main/target.kernel.cpp` gives for `Fix`: the record layout differs by
// backend binding, and a template instantiated over an address-space-qualified
// pointer is what MSL accepts. The body reads named fields of each.
//
// EVERY TERM A FREE VERTEX CONTRIBUTES TO THE NEWTON SYSTEM'S RIGHT-HAND SIDE
// AND BLOCK DIAGONAL, in one place and in one order.
//
// THE ORDER IS LOAD-BEARING. `gradient` and `hessian` are fp32 running sums, so
// reordering the accumulation changes the last bits of the assembled system:
// aerodynamic drag, then the pull pin, then the torque groups, then the inertia
// the pull pin replaces, then isotropic drag, then the `fix-xz` drag.

#include "../utility/vertex_normal.kernel.cpp"
#include "model/air_damper.hpp"
#include "model/momentum.hpp"
#include "model/torque.kernel.cpp"

// The right-hand side and diagonal block one free vertex contributes.
//
// `gradient` and `hessian` are WRITTEN, not accumulated: they are this vertex's
// whole contribution, and the caller adds them into the assembled arrays. The
// caller also owns the gate that skips a fix-pinned vertex entirely, because a
// prescribed vertex carries no momentum row: its DOF is eliminated and its
// right-hand side is written by the Dirichlet pass.
//
// `normal` is the area-weighted vertex normal, folded over this vertex's
// incident faces by the caller, because that is a walk over the neighbor table
// rather than arithmetic. A zero normal means the vertex presents no surface,
// and the aerodynamic term is then ABSENT rather than zero.
//
// THE WIND IS RAMPED AND IS GATED ON `inactive_momentum`. Both halves are here
// rather than at a call site: `air_damper::wind_weight` is a quarter-amplitude
// gust riding on a constant three quarters, so the field never reverses and
// never stalls, and a backend that applies `param.wind` raw drives a scene with
// a different force from the first frame. `inactive_momentum` is the mode in
// which nothing is integrated, so no wind may blow in it either.
//
// FIRST MATCH WINS FOR A PULL PIN, and the scan is over the whole pin array.
// The vertex property also names a pin, but reading it instead would be a
// second rule for which pin applies and the two could disagree after a rebuild.
//
// THERE IS NO `break` IN THE TORQUE LOOP, and that is the one per-vertex
// constraint scan here that accumulates rather than taking the first match: a
// vertex may belong to several torque groups and every one of them applies.
template <class Pull, class TorqueVertex, class TorqueGroup>
[[seam::device_fn]] inline void vertex_force_hessian(
    unsigned vert, const Vec3f &current,
    const Vec3f &iterate,
    const Vec3f &target,
    const Vec3f &normal, float mass, float area, float dt,
    const ParamSet &param, const Pull *pull,
    unsigned pull_count, const TorqueVertex *torque_vertex,
    unsigned torque_vertex_count,
    const TorqueGroup *torque_result,
    Vec3f &gradient, Mat3x3f &hessian) {
    Vec3f wind = Vec3f::Zero();
    if (!param.inactive_momentum) {
        wind = air_damper::wind_weight(param.time_f32) * param.wind;
    }

    gradient = Vec3f::Zero();
    hessian = Mat3x3f::Zero();
    if (normal.isZero() == false && param.air_density) {
        gradient += area * param.air_density *
                    air_damper::face_gradient(dt, iterate, current, normal,
                                              wind, param.air_friction);
        hessian += area * param.air_density *
                   air_damper::face_hessian(dt, normal, param.air_friction);
    }

    bool pulled(false);
    for (unsigned j = 0; j < pull_count; ++j) {
        if (vert == pull[j].index) {
            Vec3f position = pull[j].position;
            float weight = pull[j].weight;
            gradient += weight * (iterate - position).cast<float>();
            hessian += weight * Mat3x3f::Identity();
            pulled = true;
            break;
        }
    }

    for (unsigned j = 0; j < torque_vertex_count; ++j) {
        if (vert == torque_vertex[j].index) {
            const unsigned group = torque_vertex[j].group_id;
            Vec3f torque_force;
            Mat3x3f torque_hessian;
            // The term, and the PSD projection of its symmetric part, live in
            // energy/model/torque.kernel.cpp.
            torque_vertex_force_hessian(
                iterate, torque_result[group].center,
                torque_result[group].axis, torque_vertex[j].magnitude,
                torque_result[group].inv_r_perp_sq_sum, torque_force,
                torque_hessian);
            gradient -= torque_force;
            hessian += torque_hessian;
        }
    }

    if (!pulled) {
        gradient += mass * momentum::gradient(dt, iterate, target);
        hessian += mass * momentum::hessian(dt);
    }

    if (param.isotropic_air_friction) {
        gradient += momentum::isotropic_drag_gradient(
            dt, iterate, current, param.isotropic_air_friction);
        hessian += momentum::isotropic_drag_hessian(
            dt, param.isotropic_air_friction);
    }

    if (momentum::fix_xz_active(iterate, param.fix_xz)) {
        gradient +=
            momentum::fix_xz_gradient(dt, iterate, current, mass, param.fix_xz);
        hessian += momentum::fix_xz_hessian(dt, iterate, mass, param.fix_xz);
    }
}
