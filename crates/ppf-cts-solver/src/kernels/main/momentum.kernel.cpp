// File: momentum.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: one vertex's momentum row, plus the air damper, the
// soft PULL pin and the two global drags that share it.
//
// A PRESCRIBED VERTEX CARRIES NO MOMENTUM ROW: its DOF is eliminated and its
// right-hand side is written by the Dirichlet pass, so the body returns before
// touching either output.
//
// THE `ParamSet` IS NOT IN THE RECORD, the eight fields this body reads are. A
// generated record holds 4-byte scalars and 16-byte buffer handles, and
// `ParamSet::time` is an `f64` that MSL has no type for, so the struct could not
// cross as a pod pointee even where the size allowed it. `wind` is a `Vec3f` and
// arrives as its three components for the same reason.
//
// `distance.hpp` FIRST, for the reason `contact/contact_narrow.kernel.cpp`
// states: a neutral body includes no header of its own, so the type vocabulary
// reaches it from whichever SHARED HEADER an includer pulls in first.
#include "../contact/distance.hpp"

#include "../energy/model/air_damper.hpp"
#include "../energy/model/momentum.hpp"
#include "../energy/model/torque.kernel.cpp"
#include "../utility/vertex_normal.kernel.cpp"

[[seam::entry(i)]]
[[seam::device_fn]] inline void momentum_embed(
    const Vec3f *eval_x,
    const Vec3f *current,
    const Vec3f *target,
    const VertexProp *prop,
    const PullPair *pull, unsigned pull_count,
    const TorqueVertex *torque_vertex,
    unsigned torque_vertex_count,
    const TorqueGroupResult *torque_result,
    unsigned torque_group_count,
    const unsigned *neighbor_index,
    const unsigned *neighbor_offset, unsigned has_neighbor,
    const unsigned *face, float dt, unsigned inactive_momentum,
    float time_f32, float wind_x, float wind_y, float wind_z, float air_density,
    float air_friction, float isotropic_air_friction, float fix_xz,
    const float *field_air_velocity, unsigned has_field_air,
    float *force, Mat3x3f *diagonal,
    DiagHandle diag, unsigned i) {
    const Vec3f wind_vector(wind_x, wind_y, wind_z);

    // A prescribed vertex carries no momentum row: its DOF is eliminated
    // and its right-hand side is written by the Dirichlet pass.
    if (prop[i].fix_index > 0) {
        return;
    }
    const float mass = prop[i].mass;
    const float area = prop[i].area;
    // THREAD-SPACE COPIES, NOT REFERENCES, and that is a Metal requirement
    // rather than a style: an element reached through a `[[seam::device]]`
    // base pointer is a DEVICE-space lvalue, and MSL will not bind one to the
    // thread-space reference every body called below takes. The copy is what
    // the generator emits for a `[[seam::gather]]` parameter anyway, so it
    // costs nothing; these stay base pointers because the normal fold reaches
    // `eval_x` through the face's own index triple, which is an indirect
    // gather a record cannot express.
    const Vec3f x = current[i];
    const Vec3f y = eval_x[i];

    // The area-weighted vertex normal, accumulated over this vertex's
    // incident faces in the table's own order. The fold is inside this loop
    // rather than in the caller because it is per VERTEX and every term is
    // read from this one thread; nothing scatters.
    Vec3f normal = Vec3f::Zero();
    if (has_neighbor != 0u) {
        for (uint32_t j = neighbor_offset[i]; j < neighbor_offset[i + 1];
             ++j) {
            const uint32_t f = neighbor_index[j];
            const Vec3f z0 = eval_x[face[3 * f + 0]];
            const Vec3f z1 = eval_x[face[3 * f + 1]];
            const Vec3f z2 = eval_x[face[3 * f + 2]];
            normal += vertex_normal_face_term(z0, z1, z2);
        }
        normal = vertex_normal_finalize(normal);
    }

    Vec3f wind = Vec3f::Zero();
    // `== 0u` rather than `!inactive_momentum != 0u`, which is what this said
    // and which clang warns about on every build: `!` binds to the left
    // operand alone, so it reads as `(!x) != 0`. That happens to be the same
    // predicate for an unsigned x, but it is the exact shape of a real defect
    // and the compiler cannot tell the two apart. The reference spells it
    // `if (!param.inactive_momentum)` on a `bool` (`energy.cu:196`); the flag
    // crosses this seam as an unsigned, so the comparison is written out.
    if (inactive_momentum == 0u) {
        wind = air_damper::wind_weight(time_f32) * wind_vector;
        // THE FORCE FIELD'S AIR VELOCITY ADDS TO THE SCENE WIND WITHOUT THE
        // GUST RAMP. The ramp is a property of the one scene-wide wind; a
        // field's flow is authored per point and per instant already, so
        // ramping it again would change what the author sampled.
        if (has_field_air != 0u) {
            wind[0] += field_air_velocity[3u * i + 0u];
            wind[1] += field_air_velocity[3u * i + 1u];
            wind[2] += field_air_velocity[3u * i + 2u];
        }
    }

    Vec3f f = Vec3f::Zero();
    Mat3x3f H = Mat3x3f::Zero();
    if (normal.isZero() == false && air_density) {
        f += area * air_density *
             air_damper::face_gradient(dt, y, x, normal, wind, air_friction);
        H += area * air_density *
             air_damper::face_hessian(dt, normal, air_friction);
    }

    // FIRST MATCH WINS, and the scan is over the whole pin array, which is
    // what the device does. `pull_index` on the vertex prop names the same
    // pin, but reading it instead would be a second rule for which pin
    // applies, and the two could disagree after a rebuild.
    bool pulled = false;
    for (uint32_t j = 0; j < pull_count; ++j) {
        if (i == pull[j].index) {
            const Vec3f position = pull[j].position;
            const float weight = pull[j].weight;
            f += weight * (y - position).cast<float>();
            H += weight * Mat3x3f::Identity();
            pulled = true;
            break;
        }
    }

    // THE TORQUE TERM SITS HERE, between the pin scan and the momentum row,
    // because that is where `energy/vertex_force.kernel.cpp` sums it and the
    // fp32 accumulation order of this row is part of its answer. The scan is
    // over the whole member array for the same reason the pin scan is: a
    // vertex can be named by more than one group and every match contributes.
    //
    // THE GROUP'S FRAME IS READ, NEVER COMPUTED HERE. `torque_group_frame`
    // produced it in a pre-pass over the groups, which is what makes this a
    // per-vertex row at all: the centroid, axis and radius normalization are
    // properties of the whole group and cannot be reached from one member.
    for (unsigned j = 0; j < torque_vertex_count; ++j) {
        if (i == torque_vertex[j].index) {
            const unsigned group = torque_vertex[j].group_id;
            // THE GROUP ID IS DATA, so the thread guard says nothing about it:
            // a member naming a group the scene does not have reads past the
            // frame array, which faults on nothing under Metal and returns
            // zero. A zero frame is a plausible answer, so this is checked
            // rather than trusted.
            DIAG_ASSERT4(diag, group < torque_group_count,
                        static_cast<float>(group),
                        static_cast<float>(torque_group_count),
                        static_cast<float>(j), static_cast<float>(i));
            if (group >= torque_group_count) {
                continue;
            }
            // THE FRAME IS COPIED INTO A LOCAL FIRST. On Metal
            // `torque_result[group]` is a `const device` lvalue and the body
            // below takes its centroid and axis as `[[seam::thread]]`
            // references, so the value must cross address spaces here; a
            // generated entry does this for its own gathered arguments, and a
            // body reaching an array by an index of its own owes the same copy.
            // CUDA and the host have one address space and compile without it.
            const TorqueGroupResult frame = torque_result[group];
            Vec3f torque_force;
            Mat3x3f torque_hessian;
            // The term, and the PSD projection of its symmetric part, live in
            // energy/model/torque.kernel.cpp.
            torque_vertex_force_hessian(
                y, frame.center, frame.axis, torque_vertex[j].magnitude,
                frame.inv_r_perp_sq_sum, torque_force, torque_hessian);
            // SUBTRACTED, because the assembly accumulates energy gradients and
            // this is an applied force. The body returns the unsigned quantity
            // so the sign is visible at the call site.
            f -= torque_force;
            H += torque_hessian;
        }
    }

    if (!pulled) {
        const Vec3f target_i = target[i];
        f += mass * momentum::gradient(dt, y, target_i);
        H += mass * momentum::hessian(dt);
    }

    if (isotropic_air_friction) {
        f += momentum::isotropic_drag_gradient(dt, y, x,
                                               isotropic_air_friction);
        H += momentum::isotropic_drag_hessian(dt,
                                              isotropic_air_friction);
    }

    if (momentum::fix_xz_active(y, fix_xz)) {
        f += momentum::fix_xz_gradient(dt, y, x, mass, fix_xz);
        H += momentum::fix_xz_hessian(dt, y, mass, fix_xz);
    }

    for (unsigned k = 0; k < 3; ++k) {
        force[3 * i + k] += f[k];
    }
    // READ, ADD, WRITE BACK rather than `+=` in place, for the same reason:
    // `diagonal[i]` is a DEVICE-space lvalue and the matrix type's compound
    // assignment is declared for thread space, so `+=` finds no viable
    // overload there. The three loads, nine adds and three stores are the
    // ones the operator itself performs, in that order.
    Mat3x3f diagonal_i = diagonal[i];
    diagonal_i += H;
    diagonal[i] = diagonal_i;
}
