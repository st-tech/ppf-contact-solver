// File: friction_memory.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: the friction's memory of where each contact stuck.
//
// A CONTACT'S SLIP IS MEASURED FROM WHERE IT STUCK, NOT FROM WHERE THE STEP
// BEGAN. The friction law (`friction_evaluate`) holds a slip below
// `friction-eps` as a spring and caps the force at mu N beyond it. Measured from
// the step's own start, every stuck contact begins each step at zero slip and
// zero force and has to stretch its spring again, so it gives way by about
// friction_eps times its load over mu N every step: a held cloth creeps at a
// rate set by the step count, and how far a sliding contact gets within a step
// depends on how many Newton iterations the step took. A bias strip of yarns at
// friction 0.3 hung 16 degrees from its axis after 600 frames at the default
// Newton count and 29.4 at sixteen or more; at dt = 0.1 it held at 9.
//
// So every frictional pair carries the tangential slip it had at the end of the
// last committed step, keyed by the pair's identity: a pair's slip this step is
// that carried slip, projected onto this step's tangent plane, plus the step's
// own relative displacement. When the step is committed the slip is RETURN
// MAPPED and stored: kept whole while it is inside the friction law's stick
// range, and cut back to that range along its own direction once the contact
// has slid, which is the slider of an elastic-plastic friction spring moving
// with the slide. A stuck contact therefore holds its force from step to step,
// and the stick range is a length of the friction law alone, not of the step.
// A pair absent from the next step's contacts has separated and is forgotten.
//
// THE RECORD IS ALSO THE STEP'S FRICTION. Only a step's FIRST Newton iteration
// records, at the step's start pose x0, and every iteration of the step applies
// friction from that record (`friction_frozen_embed`, contact_narrow.kernel.cpp):
// the same pairs, each with the normal, the closest-point weights, the normal
// force and the coefficient it had at x0. Within a step the friction is then a
// fixed potential of the positions, sum mu N0 |u(x)| smoothed below the stick
// range, with a Hessian consistent with its force, which is IPC's lagged
// friction (Li et al. 2020). Read at the iterate instead, the friction term
// changes with x in ways its Hessian does not carry: a normal force that spikes
// when an iterate lands deep in a barrier, a closest point that jumps along two
// parallel edges, a pair that leaves and rejoins the contact set. Each turns the
// Newton iteration into a cycle on a house of cards, whose apex then slides
// before anything touches it. With the record frozen, the house's steps converge
// in the minimum count and it stands. A contact that forms during a step carries
// no friction until the next step, which is the price of the lag.
//
// A SAND GRAIN'S PAIRS ARE THE EXCEPTION: their slip includes the grains' spin,
// which a vertex motion cannot express, so their passes keep applying their own
// friction and record them for the memory only (`held` zero).
//
// Beside its slip each pair keeps its SLIDE, how far it moved along its tangent
// plane in the step just committed. The friction law anchors a contact loaded
// past its cone at no less than that (`friction_evaluate_anchored`), which is
// what releases a body its pair-local slip prediction holds back.
//
// A PAIR IS NAMED BY TWO WORDS, `(kind << KIND_SHIFT | a, b)`: its kind, the
// dynamic element `a` that carries the slip, and `b`, what it slides against.
// `b` is the OTHER SIDE'S OBJECT wherever a contact can slide from one element
// of it to the next (a vertex across faces and edges, an edge along a yarn,
// anything against the collision mesh), so the slip a contact carries survives
// the slide; it is the other element itself only where several contacts with
// one object are simultaneous by nature (a SAND grain among its neighbors) and
// for the analytic colliders, which are one element each. A vertex against a
// surface also names the SIDE it touches (`friction_memory_surface_kind`). The memory of the
// last committed step holds ONE entry per key, sorted (`friction_memory_settle`,
// `friction_memory_head`, `friction_memory_merge`), and a pass finds its pair
// in it by bisection.

// No include of its own, as `friction.kernel.cpp`: `Vec2u`, `Vec3f` and the
// seam vocabulary arrive from the includer, the contact passes, and from the
// entry context for this file's own entries.

// The kinds a pair can be, each pass passing its own, and the four bits of the
// first word the kind takes, so an element index must stay below 2^28; the
// driver refuses a scene with more elements of one kind at `initialize()`.
enum : unsigned {
    FRICTION_KIND_POINT_SURFACE = 1u,
    FRICTION_KIND_POINT_SURFACE_BACK = 2u,
    FRICTION_KIND_POINT_POINT = 3u,
    FRICTION_KIND_EDGE_EDGE = 4u,
    FRICTION_KIND_MESH_TO_COLLIDER = 5u,
    FRICTION_KIND_COLLIDER_TO_MESH = 6u,
    FRICTION_KIND_EDGE_COLLIDER_EDGE = 7u,
    FRICTION_KIND_SPHERE = 8u,
    FRICTION_KIND_FLOOR = 9u,
    FRICTION_KIND_SPHERE_FEATURE = 10u,
    FRICTION_KIND_SHIFT = 28u,
    FRICTION_ELEMENT_LIMIT = 1u << 28,
};

// A VERTEX AGAINST A SURFACE IS ON ONE OF ITS TWO SIDES, and the two sides are
// two contacts: a fold of cloth on itself touches one object from the front and
// the back at once, and one key for both would leave one of them nothing to
// carry. The side is the sign of `offset` (the vertex relative to any point of
// the plane) along `normal`, a winding normal of the face it touches or, for an
// edge, of a face the edge bounds, so a vertex sliding from a face over an edge
// to the next face of a consistently wound surface keeps its key.
[[seam::device_fn]] inline unsigned friction_memory_surface_kind(
    const Vec3f &normal, const Vec3f &offset) {
    return normal.dot(offset) < 0.0f ? FRICTION_KIND_POINT_SURFACE_BACK
                                     : FRICTION_KIND_POINT_SURFACE;
}

[[seam::device_fn]] inline Vec2u friction_memory_key(unsigned kind, unsigned a,
                                                     unsigned b) {
    return Vec2u((kind << FRICTION_KIND_SHIFT) | (a & (FRICTION_ELEMENT_LIMIT - 1u)), b);
}

[[seam::device_fn]] inline bool friction_memory_less(const Vec2u &p,
                                                     const Vec2u &q) {
    return p[0] < q[0] || (p[0] == q[0] && p[1] < q[1]);
}

// The slip the pair `key` carried out of the last committed step, and its
// slide then into `slide`, or zero for both for a pair that was not in contact.
// `count` keys of `key_word0` / `key_word1`, ascending, `slip`
// FRICTION_MEMORY_SLIP floats each: the slip, then the slide.
enum : unsigned { FRICTION_MEMORY_SLIP = 4u };

[[seam::device_fn]] inline Vec3f friction_memory_lookup(
    const Vec2u &key, const unsigned *key_word0, const unsigned *key_word1,
    const float *slip, unsigned count, float &slide) {
    slide = 0.0f;
    unsigned low = 0u;
    unsigned high = count;
    while (low < high) {
        const unsigned mid = low + (high - low) / 2u;
        const Vec2u here(key_word0[mid], key_word1[mid]);
        if (friction_memory_less(here, key)) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }
    if (low < count && key_word0[low] == key[0] && key_word1[low] == key[1]) {
        const float *entry = slip + FRICTION_MEMORY_SLIP * low;
        slide = entry[3];
        return Vec3f(entry[0], entry[1], entry[2]);
    }
    return Vec3f::Zero();
}

// The carried slip in this step's tangent plane. The normal turns between
// steps, and a slip along it is not a slip.
[[seam::device_fn]] inline Vec3f friction_memory_tangential(const Vec3f &slip,
                                                            const Vec3f &normal) {
    return slip - normal * normal.dot(slip);
}

// The return map: the slip a committed step leaves the pair, kept while it is
// inside the stick range `friction_eps` and cut back to it along its own
// direction past it.
[[seam::device_fn]] inline Vec3f friction_memory_return(const Vec3f &tangent,
                                                        float friction_eps) {
    const float length = fmath::sqrt(tangent.squaredNorm());
    if (length > friction_eps) {
        return tangent * fmath::div(friction_eps, length);
    }
    return tangent;
}

// THE SLIP IS RECORDED AS A RULE, NOT AS A NUMBER, because the pose it is
// needed at is not the pose it is assembled at. A Newton iteration assembles at
// its iterate and then moves it, so a slip recorded during the last iteration
// of a step misses that iteration's own correction, which at one iteration a
// step is the whole step's motion: a contact recorded that way could never
// slide. A record holds what the slip at ANY pose is made of: the pair's
// `offset` (the carried slip, and for two SAND grains their spin, which is not
// a vertex motion), its normal, and the at most four vertices and weights its
// relative motion `sum_k w_k (x_k - x0_k)` is taken over, less the pair's turn,
// which the key's kind and the vertex count name (`friction_frame_of`). The
// commit evaluates it at the committed pose (`friction_memory_settle`).
//
// Entry `slot` takes two key words, FRICTION_MEMORY_VERTICES vertices (unused
// ones FRICTION_MEMORY_UNUSED) and FRICTION_MEMORY_FLOATS floats: offset,
// normal, weights, the normal force, which decides between two contacts that
// share a key (`friction_memory_merge`), the friction coefficient, the
// pair's multiplicity, the number of times its force is deposited, which with
// the normal force give the cone and the barrier force the pair puts on its
// vertices (`friction_cone_gather`), and `held`, one where the frozen pass
// applies the pair's friction. `capacity` entries fit; a pair past it is
// counted and not written, and the driver refuses a commit whose record counted
// more. An iteration after a step's first opens the record with capacity zero,
// so the record stays as that first iteration wrote it.
//
// AN EDGE-EDGE PAIR ALSO NAMES ITS SIDES' COSSERAT FRAMES. Each entry holds
// FRICTION_RECORD_WORDS words, the FRICTION_MEMORY_VERTICES vertices and then
// one frame slot per side (FRICTION_MEMORY_UNUSED for a side whose edge carries
// no material frame), and floats 14 and 15 are the two sides' contact offsets,
// the radii their surfaces sit at from their centerlines ("A SIDE WITH A FRAME
// SLIPS BY ITS SURFACE" below).
enum : unsigned {
    FRICTION_MEMORY_VERTICES = 4u,
    FRICTION_RECORD_WORDS = 6u,
    FRICTION_MEMORY_FLOATS = 16u,
    FRICTION_MEMORY_UNUSED = 0xffffffffu,
};

[[seam::device_fn]] inline void friction_memory_record_framed(
    const Vec2u &key, const Vec3f &offset, const Vec3f &normal,
    float normal_force, float friction, float multiplicity,
    const unsigned *index, const float *weight, unsigned n, bool held,
    unsigned frame_a, unsigned frame_b, float radius_a, float radius_b,
    unsigned *record_word0, unsigned *record_word1, unsigned *record_vertex,
    float *record_float, compute::atomic_uint_t *record_count,
    unsigned capacity) {
    if (capacity == 0u) {
        return;
    }
    const unsigned slot = compute::atomic_add(record_count, 1u);
    if (slot < capacity) {
        record_word0[slot] = key[0];
        record_word1[slot] = key[1];
        for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
            record_vertex[FRICTION_RECORD_WORDS * slot + k] =
                k < n ? index[k] : FRICTION_MEMORY_UNUSED;
        }
        record_vertex[FRICTION_RECORD_WORDS * slot + 4u] = frame_a;
        record_vertex[FRICTION_RECORD_WORDS * slot + 5u] = frame_b;
        float *out = record_float + FRICTION_MEMORY_FLOATS * slot;
        for (unsigned i = 0; i < 3u; ++i) {
            out[i] = offset[i];
            out[3u + i] = normal[i];
        }
        for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
            out[6u + k] = k < n ? weight[k] : 0.0f;
        }
        out[10] = normal_force;
        out[11] = friction;
        out[12] = multiplicity;
        out[13] = held ? 1.0f : 0.0f;
        out[14] = radius_a;
        out[15] = radius_b;
    }
}

// A pair whose sides carry no material frame: `friction_memory_record_framed`
// with both frames unused.
[[seam::device_fn]] inline void friction_memory_record(
    const Vec2u &key, const Vec3f &offset, const Vec3f &normal,
    float normal_force, float friction, float multiplicity,
    const unsigned *index, const float *weight, unsigned n, bool held,
    unsigned *record_word0, unsigned *record_word1, unsigned *record_vertex,
    float *record_float, compute::atomic_uint_t *record_count,
    unsigned capacity) {
    friction_memory_record_framed(key, offset, normal, normal_force, friction,
                                  multiplicity, index, weight, n, held,
                                  FRICTION_MEMORY_UNUSED, FRICTION_MEMORY_UNUSED,
                                  0.0f, 0.0f, record_word0, record_word1,
                                  record_vertex, record_float, record_count,
                                  capacity);
}

// A PAIR THAT TURNS AS ONE BODY DOES NOT SLIP. The relative displacement
// `sum_k w_k (x_k - x0_k)` is the motion of the two CENTERLINE points the
// weights name, and those sit the contact's separation `d` apart, about the
// contact offset. A rigid turn by a small angle vector `w` moves them apart by
// `w x d`, which is tangential, so the bare measure reads a cloth swinging as a
// whole as every crossing in it sliding by `|w| |d|` a step. The surfaces that
// touch move together under that turn, and the friction law has to read their
// motion, not their centerlines'. Measured on a plain weave drape at mu 0.3
// (dt 0.01, 16 iterations, frame 300): the turn of each crossing's four
// vertices explained a median 97 percent of the step's tangential slip, and
// every crossing sat at its cone, so friction braked every bend and swing of
// the cloth. With the turn removed, a sheet sliding on a sheet is unchanged,
// and the drape's crossings still slide: at frame 300 friction dissipates
// about a fifth of gravity's work a step, with 79 percent of the pairs at the
// cone where all of them were before.
//
// THE SLIP CARRIED INTO A STEP TURNS WITH THE PAIR. It was stored in world
// coordinates, so a pair that pivots would carry it pointing where the pair
// no longer slides. Turned by the step's turn `w`, it adds `w x carried`,
// which folds into the same Jacobian as the lever `d0 - carried`.
//
// So the slip of a pair whose two sides have a frame is the relative
// displacement LESS the pair's turn: `sum_k w_k dx_k + d0 x w`, with `d0` the
// separation at the step's start and `w = sum_k W_k dx_k` the turn its
// primitives make, linear in the step's displacement with the coefficients
// `W_k` taken at the step's start. Linear, so the slip stays an affine function
// of the iterate with a Jacobian `J_k = w_k I + [d0]x W_k` that is constant over
// the step; the Newton iteration differentiates it exactly and the record's
// rule (`friction_memory_settle`, `friction_cone_gather`) evaluates the same
// expression at any pose. For a rigid turn it cancels the relative
// displacement to first order in the angle; what is left is of order
// `|w|^2 |d|`, far below `friction-eps` at a step's turn.
//
// THE FRAME COMES FROM THE PRIMITIVES, NEVER FROM THE SEPARATION. The direction
// of `d` turns with any slide that moves the closest points (at an edge's end,
// or a vertex past a face's border), so a turn read off it would cancel real
// sliding, and with weights held fixed it cancels all of it. Per pair kind:
//
// - A vertex against a face: the face's frame, its normal's turn plus its
//   spin about the normal, exact.
// - Two edges: the turn that best fits both edges' own turns, which is exact
//   for crossing edges. Two edges cannot see a turn about the line bisecting
//   them except through the angle between them, at strength `1 - cos(phi)`;
//   solving for it exactly would divide every difference in the two edges'
//   bending by that strength, without bound as the edges become parallel. The
//   fit counts it at strength 1 instead, which is exact for perpendicular
//   edges (every crossing of a woven cloth) and leaves the part a pair of
//   parallel edges cannot see measured as slip, as before.
// - A vertex against an edge: the edge's turn. A vertex has no frame and an
//   edge none about its own axis, so a turn about that axis stays slip.
// - Two vertices, and every pair against the collision mesh or an analytic
//   collider: no frame; the bare relative displacement.
enum : unsigned {
    FRICTION_FRAME_NONE = 0u,
    FRICTION_FRAME_EDGE_EDGE = 1u,
    FRICTION_FRAME_POINT_FACE = 2u,
    FRICTION_FRAME_POINT_EDGE = 3u,
};

// Which frame a pair of kind `kind` naming `n` vertices has. Every pass that
// records a pair and every reader of the record ask this one function, so the
// slip a pass assembled and the slip the record's rule evaluates are one
// expression. The vertex order is the contact passes': (a0, a1, b0, b1) for two
// edges, (vertex, f0, f1, f2) for a face and (vertex, e0, e1) for an edge.
[[seam::device_fn]] inline unsigned friction_frame_of(unsigned kind, unsigned n) {
    if (kind == FRICTION_KIND_EDGE_EDGE && n == 4u) {
        return FRICTION_FRAME_EDGE_EDGE;
    }
    if (kind == FRICTION_KIND_POINT_SURFACE ||
        kind == FRICTION_KIND_POINT_SURFACE_BACK) {
        if (n == 4u) {
            return FRICTION_FRAME_POINT_FACE;
        }
        if (n == 3u) {
            return FRICTION_FRAME_POINT_EDGE;
        }
    }
    return FRICTION_FRAME_NONE;
}

// The cross product as a matrix: `friction_cross_matrix(a) * b == a x b`.
[[seam::device_fn]] inline Mat3x3f friction_cross_matrix(const Vec3f &a) {
    Mat3x3f m = Mat3x3f::Zero();
    m(0, 1) = -a[2];
    m(0, 2) = a[1];
    m(1, 0) = a[2];
    m(1, 2) = -a[0];
    m(2, 0) = -a[1];
    m(2, 1) = a[0];
    return m;
}

// The vector from vertex `i` to vertex `j` at the pose `x`. The two positions
// are copied out of the pose first, because Metal has no difference of two
// device-space position vectors.
[[seam::device_fn]] inline Vec3f friction_frame_edge(const Vec3f *x,
                                                     unsigned i, unsigned j) {
    const Vec3f xi = x[i];
    const Vec3f xj = x[j];
    return (xj - xi);
}

// The turn's coefficients `W_k` at the step's start `x0`, for the vertices
// `v` in the frame's order: the pair's turn over the step is
// `sum_k turn[k] (x_k - x0_k)`. A vertex the frame does not read gets zero.
// Every edge and face a contact names has a positive length and area, which
// the build gate guarantees, so no division here is by zero.
[[seam::device_fn]] inline void friction_frame_turn(unsigned frame,
                                                    const Vec3f *x0,
                                                    const unsigned *v,
                                                    Mat3x3f *turn) {
    for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
        turn[k] = Mat3x3f::Zero();
    }
    if (frame == FRICTION_FRAME_EDGE_EDGE) {
        const Vec3f ea = friction_frame_edge(x0, v[0], v[1]);
        const Vec3f eb = friction_frame_edge(x0, v[2], v[3]);
        const float la = fmath::sqrt(ea.squaredNorm());
        const float lb = fmath::sqrt(eb.squaredNorm());
        const Vec3f ta = ea * fmath::div(1.0f, la);
        const Vec3f tb = eb * fmath::div(1.0f, lb);
        // Each edge's own turn is `w x t` less its part along `t`; the
        // residual of the fit is minimal at `M w = sum t x dt` with
        // `M = 2 I - ta ta^T - tb tb^T`. Its eigenvalue along the bisector
        // `u` of the acute angle is `1 - cos(phi)`, raised to 1 here (the
        // comment above); the other two are 2 and `1 + cos(phi)`.
        const float cosine = ta.dot(tb);
        const Vec3f bisector = cosine >= 0.0f ? Vec3f(ta + tb) : Vec3f(ta - tb);
        const Vec3f u = bisector * fmath::div(1.0f, fmath::sqrt(bisector.squaredNorm()));
        const Mat3x3f fit = 2.0f * Mat3x3f::Identity() - ta * ta.transpose() -
                            tb * tb.transpose() +
                            fmath::abs(cosine) * (u * u.transpose());
        const Mat3x3f inverse = fit.inverse();
        turn[1] = fmath::div(1.0f, la) * (inverse * friction_cross_matrix(ta));
        turn[0] = -1.0f * turn[1];
        turn[3] = fmath::div(1.0f, lb) * (inverse * friction_cross_matrix(tb));
        turn[2] = -1.0f * turn[3];
    } else if (frame == FRICTION_FRAME_POINT_FACE) {
        const Vec3f e1 = friction_frame_edge(x0, v[1], v[2]);
        const Vec3f e2 = friction_frame_edge(x0, v[1], v[3]);
        const Vec3f c = e1.cross(e2);
        const float area2 = fmath::sqrt(c.squaredNorm());
        const float l1 = fmath::sqrt(e1.squaredNorm());
        const Vec3f n = c * fmath::div(1.0f, area2);
        const Vec3f t = e1 * fmath::div(1.0f, l1);
        // The normal's turn `n x dn`, with `dn` the normal part of
        // `(de1 x e2 + e1 x de2) / |c|`, plus the spin `n . (t x dt)` along
        // `n`, `dt` the turn of the first edge.
        const Mat3x3f nx = friction_cross_matrix(n);
        const Mat3x3f spin = fmath::div(1.0f, l1) *
                             ((n * n.transpose()) * friction_cross_matrix(t));
        turn[2] = fmath::div(-1.0f, area2) * (nx * friction_cross_matrix(e2)) + spin;
        turn[3] = fmath::div(1.0f, area2) * (nx * friction_cross_matrix(e1));
        turn[1] = -1.0f * (turn[2] + turn[3]);
    } else if (frame == FRICTION_FRAME_POINT_EDGE) {
        const Vec3f e = friction_frame_edge(x0, v[1], v[2]);
        const float l = fmath::sqrt(e.squaredNorm());
        const Vec3f t = e * fmath::div(1.0f, l);
        turn[2] = fmath::div(1.0f, l) * friction_cross_matrix(t);
        turn[1] = -1.0f * turn[2];
    }
}

// The slip Jacobian's turn part, `[d0]x W_k`, for the separation `d0` at the
// step's start: the pair's slip is `sum_k (w_k I + rotation[k]) dx_k`.
[[seam::device_fn]] inline void friction_frame_jacobian(unsigned frame,
                                                        const Vec3f *x0,
                                                        const unsigned *v,
                                                        const Vec3f &d0,
                                                        Mat3x3f *rotation) {
    friction_frame_turn(frame, x0, v, rotation);
    const Mat3x3f dx = friction_cross_matrix(d0);
    for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
        rotation[k] = dx * rotation[k];
    }
}

// A recorded pair's separation at the step's start, `sum_k w_k x0_k`. The
// weights of a framed pair sum to zero, so it is a sum of differences from its
// first vertex and never an absolute position.
[[seam::device_fn]] inline Vec3f friction_frame_separation(const Vec3f *x0,
                                                           const unsigned *v,
                                                           const float *w) {
    Vec3f d0 = Vec3f::Zero();
    for (unsigned k = 1; k < FRICTION_MEMORY_VERTICES; ++k) {
        if (v[k] != FRICTION_MEMORY_UNUSED) {
            d0 += w[k] * friction_frame_edge(x0, v[0], v[k]);
        }
    }
    return d0;
}

// The step's slip of a recorded pair at the pose `x`: its relative displacement
// less its turn, both from the record's vertices `v` and weights `w` (unused
// vertices FRICTION_MEMORY_UNUSED), with the slip `carried` into the step
// turned along with the pair.
[[seam::device_fn]] inline Vec3f friction_frame_step(unsigned frame,
                                                     const Vec3f *x0,
                                                     const Vec3f *x,
                                                     const unsigned *v,
                                                     const float *w,
                                                     const Vec3f &carried) {
    Vec3f moved = Vec3f::Zero();
    for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
        if (v[k] != FRICTION_MEMORY_UNUSED) {
            const Vec3f xv = x[v[k]];
            const Vec3f x0v = x0[v[k]];
            moved += w[k] * (xv - x0v);
        }
    }
    if (frame == FRICTION_FRAME_NONE) {
        return moved;
    }
    Mat3x3f rotation[FRICTION_MEMORY_VERTICES];
    friction_frame_jacobian(frame, x0, v,
                            friction_frame_separation(x0, v, w) - carried,
                            rotation);
    for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
        if (v[k] != FRICTION_MEMORY_UNUSED) {
            const Vec3f xv = x[v[k]];
            const Vec3f x0v = x0[v[k]];
            moved += rotation[k] * (xv - x0v);
        }
    }
    return moved;
}

// The number of vertices a record entry names.
[[seam::device_fn]] inline unsigned friction_record_arity(const unsigned *v) {
    unsigned n = 0u;
    for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
        n += v[k] != FRICTION_MEMORY_UNUSED ? 1u : 0u;
    }
    return n;
}

// A SIDE WITH A FRAME SLIPS BY ITS SURFACE. A yarn's centerline sits its
// contact offset r behind the surface that touches, so where the yarn turns by
// theta in a step its touching surface moves by theta x (-r n) beside its
// centerline, n the pair's normal from side b toward side a. Side a's surface
// sits at -r_a n from its centerline and side b's at +r_b n, so the surfaces'
// relative step is the centerlines' less r_a theta_a x n and r_b theta_b x n:
//     slip = sum_k w_k dx_k + r_a n x theta_a + r_b n x theta_b.
// A centerline cannot say how a yarn ROLLS about its own axis, and the slip of
// two crossing yarns depends on it: a yarn sliding round a fixed post slips by
// its whole travel less r_a / R of it (R the bend's radius), while two yarns of
// a cloth that bends turn and roll together and do not slip. A Cosserat edge's
// material frame carries the roll, so for a side whose edge has one, theta is
// the frame's turn over the step; a side without one can only turn as its
// centerline does, so for an edge side theta is the edge's own turn,
// t x (dx_1 - dx_0) / l at the step's start, which holds it from rolling. The
// pair rule above ("A PAIR THAT TURNS AS ONE BODY DOES NOT SLIP") stays for a
// pair neither of whose sides has a frame. Measured on yarn wrapped half a turn
// over a drum of fixed yarns (the capstan): the pair rule removes the yarn's
// whole turn times the separation, about twice its surface's, and lets it slip
// at a tension ratio of exp(mu pi (1 - 2 r / R)) where a thick yarn's surface
// holds to exp(mu pi (1 - r / R)).
//
// The slip carried into the step is at most the stick range, so the turn it
// takes with a framed pair is of second order and is left out there.
//
// A rigid motion of the pair with both frames turning with it does not slip,
// nor does side a rolling on a still side b; a turn about the normal through
// the contact point reads no slip either, so a point contact carries no pivot
// friction.

// The turn of the frame at `slot` from the step's start, in the left
// perturbation the frames are solved in (R = exp(skew(theta)) R_start): the
// rotation part of q (x) conj(q_start), on the hemisphere of positive scalar
// part. Quaternions are (x, y, z, w), the Cosserat frames' storage.
[[seam::device_fn]] inline Vec3f friction_slot_turn(const float *quat,
                                                    const float *quat_start,
                                                    unsigned slot) {
    const Vec3f qv(quat[4u * slot], quat[4u * slot + 1u], quat[4u * slot + 2u]);
    const float qw = quat[4u * slot + 3u];
    const Vec3f pv(quat_start[4u * slot], quat_start[4u * slot + 1u],
                   quat_start[4u * slot + 2u]);
    const float pw = quat_start[4u * slot + 3u];
    float w = qw * pw + qv.dot(pv);
    Vec3f v = pw * qv - qw * pv - qv.cross(pv);
    if (w < 0.0f) {
        w = -w;
        v = -1.0f * v;
    }
    const float sine = fmath::sqrt(v.squaredNorm());
    if (!(sine > 0.0f)) {
        return Vec3f::Zero();
    }
    return fmath::div(2.0f * fmath::atan2(sine, w), sine) * v;
}

// The slip Jacobian of a framed pair: two edges, side a vertices 0 and 1 and
// side b vertices 2 and 3 (`contact_edge_edge_at`), or a vertex against an
// edge, side a vertex 0 and side b vertices 1 and 2 (`contact_point_edge_at`),
// told apart by the record's vertex count. `jacobian[k]` is vertex k's weight,
// plus the edge's own turn for an edge side without a frame; a vertex side has
// no turn. `frame_jacobian[s]` is `r_s [n]x` for a side with a frame and zero
// otherwise ("A SIDE WITH A FRAME SLIPS BY ITS SURFACE"). Constant over the
// step: it reads the start pose x0 and the record only.
[[seam::device_fn]] inline void friction_framed_jacobian(
    const Vec3f *x0, const unsigned *v, const float *w, const unsigned *slot,
    const float *radius, const Vec3f &normal, Mat3x3f *jacobian,
    Mat3x3f *frame_jacobian) {
    const Mat3x3f nx = friction_cross_matrix(normal);
    for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
        jacobian[k] = w[k] * Mat3x3f::Identity();
    }
    const bool point_edge = friction_record_arity(v) == 3u;
    for (unsigned s = 0; s < 2u; ++s) {
        frame_jacobian[s] = Mat3x3f::Zero();
        // The side's edge as its first vertex; a point-edge pair's side a is
        // a vertex and has none.
        const unsigned first = point_edge ? (s == 0u ? FRICTION_MEMORY_VERTICES : 1u)
                                          : 2u * s;
        if (slot[s] != FRICTION_MEMORY_UNUSED) {
            frame_jacobian[s] = radius[s] * nx;
        } else if (first < FRICTION_MEMORY_VERTICES) {
            const Vec3f e = friction_frame_edge(x0, v[first], v[first + 1u]);
            const Mat3x3f turn = fmath::div(radius[s], e.squaredNorm()) *
                                 (nx * friction_cross_matrix(e));
            jacobian[first + 1u] += turn;
            jacobian[first] += -1.0f * turn;
        }
    }
}

// The step's slip of a recorded pair, `moved`, at the pose x and frames quat,
// with its Jacobian: `jacobian[k]` on vertex k's step and, per side s with a
// frame (`slot[s]` not FRICTION_MEMORY_UNUSED), `frame_jacobian[s]` on that
// frame's turn. A pair without a frame on either side is the pair rule's
// (`friction_frame_step`, `friction_frame_jacobian`), with zero frame columns.
// `quat` and `quat_start` are read only for a framed pair.
[[seam::device_fn]] inline Vec3f friction_entry_step(
    unsigned frame, const Vec3f *x0, const Vec3f *x, const unsigned *v,
    const float *w, const unsigned *slot, const float *radius,
    const Vec3f &normal, const Vec3f &carried, const float *quat,
    const float *quat_start, Mat3x3f *jacobian, Mat3x3f *frame_jacobian) {
    frame_jacobian[0] = Mat3x3f::Zero();
    frame_jacobian[1] = Mat3x3f::Zero();
    const bool framed =
        slot[0] != FRICTION_MEMORY_UNUSED || slot[1] != FRICTION_MEMORY_UNUSED;
    if (!framed) {
        Mat3x3f rotation[FRICTION_MEMORY_VERTICES];
        for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
            rotation[k] = Mat3x3f::Zero();
        }
        if (frame != FRICTION_FRAME_NONE) {
            friction_frame_jacobian(frame, x0, v,
                                    friction_frame_separation(x0, v, w) - carried,
                                    rotation);
        }
        for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
            jacobian[k] = w[k] * Mat3x3f::Identity() + rotation[k];
        }
        return friction_frame_step(frame, x0, x, v, w, carried);
    }
    friction_framed_jacobian(x0, v, w, slot, radius, normal, jacobian,
                             frame_jacobian);
    Vec3f moved = Vec3f::Zero();
    for (unsigned s = 0; s < 2u; ++s) {
        if (slot[s] != FRICTION_MEMORY_UNUSED) {
            moved += frame_jacobian[s] * friction_slot_turn(quat, quat_start, slot[s]);
        }
    }
    for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
        if (v[k] != FRICTION_MEMORY_UNUSED) {
            const Vec3f xv = x[v[k]];
            const Vec3f x0v = x0[v[k]];
            moved += jacobian[k] * (xv - x0v);
        }
    }
    return moved;
}

// THE SORT'S SECOND PASS, one thread per entry: the first word of each entry,
// in the order the sort by the second word left them, for the stable sort by
// the first word that makes the order lexicographic.
[[seam::entry(entry)]]
[[seam::device_fn]] inline void friction_memory_gather_word(
    const unsigned *word, const unsigned *order, unsigned *out, unsigned entry) {
    out[entry] = word[order[entry]];
}

// THE COMMITTED STEP'S PAIRS IN KEY ORDER, one thread per entry: sorted entry
// `entry` is record `first[second[entry]]`, `second` the order of the stable
// sort by the first word over the entries `first`, the sort by the second
// word, had arranged. Its slip is the record's rule evaluated at the committed
// pose `x` against the step's start `x0`, in the pair's tangent plane, and
// return mapped to the stick range; its slide is the step's own tangential
// motion. Equal keys are adjacent here and merged by the two entries below.
[[seam::entry(entry)]]
[[seam::device_fn]] inline void friction_memory_settle(
    const Vec3f *x0, const Vec3f *x,
    const unsigned *record_word0, const unsigned *record_word1,
    const unsigned *record_vertex, const float *record_float,
    const unsigned *first, const unsigned *second, float friction_eps,
    const float *quat, const float *quat_start,
    unsigned *sorted_word0, unsigned *sorted_word1, float *sorted_slip,
    float *sorted_force, unsigned entry) {
    const unsigned source = first[second[entry]];
    sorted_word0[entry] = record_word0[source];
    sorted_word1[entry] = record_word1[source];
    const float *in = record_float + FRICTION_MEMORY_FLOATS * source;
    const Vec3f normal(in[3], in[4], in[5]);
    unsigned v[FRICTION_MEMORY_VERTICES];
    float w[FRICTION_MEMORY_VERTICES];
    for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
        v[k] = record_vertex[FRICTION_RECORD_WORDS * source + k];
        w[k] = in[6u + k];
    }
    const unsigned slot[2] = {record_vertex[FRICTION_RECORD_WORDS * source + 4u],
                              record_vertex[FRICTION_RECORD_WORDS * source + 5u]};
    const float radius[2] = {in[14], in[15]};
    const unsigned frame = friction_frame_of(
        record_word0[source] >> FRICTION_KIND_SHIFT, friction_record_arity(v));
    const Vec3f offset(in[0], in[1], in[2]);
    Mat3x3f jacobian[FRICTION_MEMORY_VERTICES];
    Mat3x3f frame_jacobian[2];
    const Vec3f moved =
        friction_entry_step(frame, x0, x, v, w, slot, radius, normal, offset,
                            quat, quat_start, jacobian, frame_jacobian);
    const Vec3f slip = offset + moved;
    const Vec3f held = friction_memory_return(
        friction_memory_tangential(slip, normal), friction_eps);
    float *out = sorted_slip + FRICTION_MEMORY_SLIP * entry;
    for (unsigned i = 0; i < 3u; ++i) {
        out[i] = held[i];
    }
    out[3] = fmath::sqrt(friction_memory_tangential(moved, normal).squaredNorm());
    sorted_force[entry] = in[10];
}

// THE FIRST ENTRY OF EACH RUN OF EQUAL KEYS, one thread per sorted entry:
// `head[entry]` is one where the key differs from the one before it. An
// exclusive scan over `head` then gives each run its slot in the memory and,
// one past the end, the number of distinct pairs.
[[seam::entry(entry)]]
[[seam::device_fn]] inline void friction_memory_head(
    const unsigned *sorted_word0, const unsigned *sorted_word1, unsigned *head,
    unsigned entry) {
    const bool first = entry == 0u ||
                       sorted_word0[entry] != sorted_word0[entry - 1u] ||
                       sorted_word1[entry] != sorted_word1[entry - 1u];
    head[entry] = first ? 1u : 0u;
}

// ONE ENTRY PER KEY, one thread per sorted entry, acting only at the head of a
// run. Two contacts share a key where one dynamic element touches one object
// in two places at once (a fold of cloth on itself), and the memory keeps the
// one bearing the larger normal force, with ties broken by the slip's
// components so the choice does not depend on the order the passes appended
// them in. A run is one entry almost always, so the walk is short. `slot` is
// the exclusive scan of `friction_memory_head`'s flags.
[[seam::entry(entry)]]
[[seam::device_fn]] inline void friction_memory_merge(
    const unsigned *sorted_word0, const unsigned *sorted_word1,
    const float *sorted_slip, const float *sorted_force, const unsigned *head,
    const unsigned *slot, unsigned entries, unsigned *memory_word0,
    unsigned *memory_word1, float *memory_slip, unsigned entry) {
    if (head[entry] == 0u) {
        return;
    }
    unsigned best = entry;
    for (unsigned j = entry + 1u; j < entries && head[j] == 0u; ++j) {
        bool better = sorted_force[j] > sorted_force[best];
        if (sorted_force[j] == sorted_force[best]) {
            for (unsigned i = 0; i < FRICTION_MEMORY_SLIP; ++i) {
                const float a = sorted_slip[FRICTION_MEMORY_SLIP * j + i];
                const float b = sorted_slip[FRICTION_MEMORY_SLIP * best + i];
                if (a != b) {
                    better = a < b;
                    break;
                }
            }
        }
        if (better) {
            best = j;
        }
    }
    const unsigned out = slot[entry];
    memory_word0[out] = sorted_word0[best];
    memory_word1[out] = sorted_word1[best];
    for (unsigned i = 0; i < FRICTION_MEMORY_SLIP; ++i) {
        memory_slip[FRICTION_MEMORY_SLIP * out + i] =
            sorted_slip[FRICTION_MEMORY_SLIP * best + i];
    }
}

// THE ANCHOR'S RELEASE TEST IS PER VERTEX, AGAINST WHAT FRICTION AT THE VERTEX
// HAS TO CARRY.
//
// The release test in `friction_evaluate_anchored` compares the tangential
// drive a pair reads off its vertices with the pair's own cone mu N. Two things
// make that drive wrong when a vertex carries several contacts, and each alone
// leaves the Newton iteration in an exact two-cycle across the stick range, a
// cycle the solver has no energy line search to break:
//
// THE LOAD IS SHARED. Every pair at a vertex reads the whole vertex's load, so
// three to twelve pairs that together hold it easily each see it past their own
// cone. So a pair reads, from each vertex i, the fraction of it its cone is of
// the vertex's summed cone, `friction_cone_fraction`, and release is decided
// against the cone the vertex actually has.
//
// THE NORMAL FORCES ARE NOT FRICTION'S LOAD. Weight another contact's surface
// holds by its normal force reads, through a residual without it, as a
// tangential drive on this one. So the load every pair reads is the residual
// before any contact force plus the barrier (normal) force of every contact at
// the vertex, and no friction force.
//
// THE SHARE IS ONE NEWTON ITERATION OLD; THE LOAD IS THE LAST COMMIT'S. A
// pair's normal force is known only once its assembly has run, so the sums are
// gathered from the record each iteration closes (this entry), and the next
// iteration's passes read the cone sum and pair count from them. The
// non-friction load does NOT come from the last iteration but from the sums as
// the step's commit left them (`FrictionMemory::commit` keeps a copy, and a
// checkpoint carries it). Read from the last iteration it closes a loop inside
// the Newton solve: an iterate that lands a pair deep in its barrier reports a
// normal force hundreds of times its weight for that iterate, the next
// iteration reads it as a tangential drive at every vertex near it, the drive
// passes the cone, the anchors release and move, and the moved iterate lands
// deeper still. A house of cards toppled from its apex in about one run of
// three that way, and a plain drape's friction work was carried by a few such
// iterates; with the committed load neither occurs, and the load a stuck
// contact is judged against is a state of the simulation rather than of one
// trial iterate. A vertex with no sum, as in the first step, reads its plain
// residual, and one with no cone its whole share, which is the law without
// either correction.
//
// `cone` holds FRICTION_CONE_FLOATS floats per vertex: the cone sum, the number
// of pairs in it, and the non-friction contact load (`friction_load_settle`).
enum : unsigned {
    FRICTION_CONE_FLOATS = 5u,
};

// The lagged cone sum of vertex `vert` and its pair count, for
// `friction_cone_fraction` (`friction.kernel.cpp`).
[[seam::device_fn]] inline void friction_cone_lagged(const float *cone,
                                                     unsigned vert, float &sum,
                                                     float &pairs) {
    const unsigned at = FRICTION_CONE_FLOATS * vert;
    sum = cone[at];
    pairs = cone[at + 1u];
}

// THE LOAD A FRICTION TERM READS, one thread per vertex, written once an
// iteration before any contact pass: the residual snapshot plus the vertex's
// lagged non-friction contact load. Every frictional pair at a vertex reads the
// same sum, so it is formed here once rather than by each pair, and the contact
// passes read one array (`friction_load`) beside the cone sums.
//
// IT IS ALSO WHAT KEEPS CUDA 12.8 AND 12.9 FROM MISCOMPILING THE CONTACT
// PASSES. With each pair reading the barrier slots of `cone` and adding them to
// the residual itself, beside the cone sum and pair count, the float-coordinate
// CUDA build of the self-contact traversals crashed with "an illegal memory
// access" (the kernel log's Xid 13, "Out Of Range Address", on a pointer whose
// upper 32 bits held a small integer): every replay of a captured `woven` scene
// under 12.8 and 12.9, none under 13.2, from the same source and flags, and
// about half of random `cards` and `woven` scenes. The values read were the
// zeros of a first iteration, and a build reading constant zeros instead never
// crashed, so the code the compiler emitted was at fault, not what it computed.
// ptxas -O1 and -O2 crashed too, and compute-sanitizer, cuda-gdb, a GPU core
// dump and (on most scenes) -lineinfo all hid it. With the load formed here, a
// contact pass reads the cone sum and pair count and nothing else from `cone`,
// and the same float build replays the captured scene 0 times in 10 and runs
// 20 of 20 random `cards` and `woven` scenes, and 0 in 10 and 16 of 16 built
// with -lineinfo, where that scene had crashed 9 times in 10. Do not move the
// barrier read back into the contact passes while the toolchain is CUDA 12.x.
[[seam::entry(vert)]]
[[seam::device_fn]] inline void friction_load_prepare(const float *residual,
                                                      const float *cone,
                                                      float *load,
                                                      unsigned vert) {
    const unsigned at = FRICTION_CONE_FLOATS * vert;
    for (unsigned d = 0; d < 3u; ++d) {
        load[3u * vert + d] = residual[3u * vert + d] + cone[at + 2u + d];
    }
}

// THE SUMS OF AN ITERATION, one thread per record entry, into `cone`, which the
// driver has zeroed: each vertex of the pair gains |w| m mu N in its cone sum,
// one pair, and the friction force the pair deposited on it, m w f, with m the
// multiplicity. The force is not stored by the passes; it is recomputed here
// from the record's rule at the iterate `x` the passes assembled at, against
// the step's start `x0`, exactly as the friction law formed it: the slip
// u = offset + sum_k J_k (x_k - x0_k) in the record's tangent plane
// (`friction_frame_step`), and
// f = mu N u_t / max(eps, |u_t|). That keeps every contact pass a reader of
// these sums and nothing else, where depositing the force from each pass would
// put a float accumulation on rows that run in parallel. A record that
// overflowed its capacity sums the pairs it wrote; a vertex missing some reads
// a share above its true one, and their friction as part of its load, for that
// iteration, and the driver grows the capacity for the next.
[[seam::entry(entry)]]
[[seam::device_fn]] inline void friction_cone_gather(
    const Vec3f *x0, const Vec3f *x, const unsigned *record_word0,
    const unsigned *record_vertex, const float *record_float, float friction_eps,
    const float *quat, const float *quat_start,
    compute::atomic_float_t *cone, unsigned entry) {
    const float *in = record_float + FRICTION_MEMORY_FLOATS * entry;
    const float multiplicity = in[12];
    const float pair_cone = in[11] * in[10];
    if (!(pair_cone > 0.0f)) {
        return;
    }
    const Vec3f normal(in[3], in[4], in[5]);
    unsigned v[FRICTION_MEMORY_VERTICES];
    float w[FRICTION_MEMORY_VERTICES];
    for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
        v[k] = record_vertex[FRICTION_RECORD_WORDS * entry + k];
        w[k] = in[6u + k];
    }
    const unsigned slot[2] = {record_vertex[FRICTION_RECORD_WORDS * entry + 4u],
                              record_vertex[FRICTION_RECORD_WORDS * entry + 5u]};
    const float radius[2] = {in[14], in[15]};
    const unsigned frame = friction_frame_of(
        record_word0[entry] >> FRICTION_KIND_SHIFT, friction_record_arity(v));
    const Vec3f offset(in[0], in[1], in[2]);
    Mat3x3f jacobian[FRICTION_MEMORY_VERTICES];
    Mat3x3f frame_jacobian[2];
    const Vec3f moved =
        friction_entry_step(frame, x0, x, v, w, slot, radius, normal, offset,
                            quat, quat_start, jacobian, frame_jacobian);
    const Vec3f tangent =
        friction_memory_tangential(offset + moved, normal);
    const Vec3f force =
        fmath::div(pair_cone,
                   fmath::max(friction_eps, fmath::sqrt(tangent.squaredNorm()))) *
        tangent;
    // The force reaches each vertex through the slip's Jacobian, `J_k^T f`,
    // as the frozen pass deposits it. A vertex the weights skip can still carry
    // a turn's part (an edge's far end). A frame's share is a torque, not a
    // vertex load, and is not summed here.
    for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
        if (v[k] == FRICTION_MEMORY_UNUSED) {
            continue;
        }
        const unsigned at = FRICTION_CONE_FLOATS * v[k];
        if (w[k] != 0.0f) {
            compute::atomic_add(cone + at, fmath::abs(w[k]) * multiplicity * pair_cone);
            compute::atomic_add(cone + at + 1u, 1.0f);
        }
        const Vec3f deposit = jacobian[k].transpose() * force;
        for (unsigned d = 0; d < 3u; ++d) {
            if (deposit[d] != 0.0f) {
                compute::atomic_add(cone + at + 2u + d, multiplicity * deposit[d]);
            }
        }
    }
}

// THE NON-FRICTION CONTACT LOAD OF AN ITERATION, one thread per vertex, after
// the gather: what the contact passes deposited into `force` beyond the
// `residual` they started from, less the friction the gather summed into the
// load's slots. That is every contact's barrier force, frictional or not, plus
// whatever else the passes hold the vertex with, the part of the load friction
// does not carry, and it replaces the friction sum in the same slots for the
// next iteration to read.
[[seam::entry(vert)]]
[[seam::device_fn]] inline void friction_load_settle(
    const float *force, const float *residual, float *cone, unsigned vert) {
    const unsigned at = FRICTION_CONE_FLOATS * vert;
    for (unsigned d = 0; d < 3u; ++d) {
        const unsigned row = 3u * vert + d;
        cone[at + 2u + d] = (force[row] - residual[row]) - cone[at + 2u + d];
    }
}
