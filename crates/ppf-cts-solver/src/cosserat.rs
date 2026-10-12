// File: crates/ppf-cts-solver/src/cosserat.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Cosserat rod per-edge orientation frames.
//
// A Cosserat rod (Painless Differentiable Rotation Dynamics, Romanya-
// Serrasolsas et al. 2025, section 6.2, following the Spillmann and Teschner
// CoRdE discretization) carries an SO(3) orientation R at every edge center in
// addition to node positions. This module initializes those orientations from
// the rest (initial) geometry and computes the rest Darboux vector at each
// interior node. Everything here is host-side, pure f32, and unit tested; the
// solver uploads the results to the device and reads them during assembly.
//
// Frame convention. Each edge orientation R in SO(3) has its LOCAL Z axis
// aligned with the edge director (the unit tangent from the edge's first node
// to its second). The remaining roll about the tangent is fixed by parallel
// transport along the rod (a discrete Bishop frame): consecutive frames differ
// only by the minimal rotation that carries one tangent onto the next, so the
// initial twist rate is zero and "twist" is later measured relative to this
// untwisted reference. Any consistent per-edge frame would leave the rest
// energy zero at frame 0 (rest = initial); parallel transport is the choice
// that avoids a spurious intrinsic twist.
//
// Darboux vector. At an interior node between the previous edge (orientation
// R_prev, the paper's R_{i-1}) and the next edge (R_next, the paper's R_i) the
// discrete Darboux vector is omega = (2 / l) vec(q(G0)), q(G0) the quaternion of
// the relative rotation G0 = R_prev^T R_next on the hemisphere of positive scalar
// part: a relative turn phi about the unit axis a gives (2 / l) sin(phi / 2) a,
// one-to-one below a half turn. l is the node integration length, the mean of
// the two adjacent edge rest lengths. The rest Darboux stored here is this omega
// evaluated on the initial frames, by the formula the device kernel evaluates
// (`darboux_of`), so the rest pose carries no strain.
//
// Quaternions are stored as [x, y, z, w] to match the device rotation helpers
// in kernels/energy/model/pdrd_polar.hpp (NOT the [w, x, y, z] datamodel quat).

use crate::data::Vec3f;
use nalgebra as na;

/// One Cosserat edge as handed to the frame builder. `v0`/`v1` are global
/// vertex indices, `tangent` is the unit director from `v0` to `v1` (the
/// caller differences the two endpoint positions before casting the result),
/// and `length` is the edge rest length.
#[derive(Debug, Clone, Copy)]
pub struct CosseratEdgeInput {
    pub edge: u32,
    pub v0: u32,
    pub v1: u32,
    pub tangent: Vec3f,
    pub length: f32,
    /// Intrinsic (authored) rest twist rate, in radians per unit length, added
    /// to the twist component of the rest Darboux vector. Zero for an
    /// untwisted rod. A straight rod with a non-zero rest twist stores twist
    /// energy in its straight configuration (the "intrinsically twisted rod"
    /// of helical perversion): the frames want to twist to reach the rest
    /// Darboux, and if the rod is held straight that stored energy buckles it.
    pub rest_twist: f32,
    /// When true, this edge's orientation is a candidate for an end-of-rod
    /// Dirichlet pin: if it is the first or last edge of its rod chain, its
    /// orientation is held fixed. Holding the end frames is what lets an
    /// intrinsic twist buckle the centerline (helical perversion) instead of
    /// unwinding freely.
    pub fix_ends: bool,
    /// Boundary twist rate in radians per second, applied only when this edge is
    /// a held chain end (requires `fix_ends`). The two ends of a chain are driven
    /// in OPPOSITE senses (the first end at `+rate`, the last at `-rate`) about
    /// their own tangent, so a straight cable held at both ends is twisted like a
    /// real cable clamped and turned oppositely, and above the critical twist its
    /// middle writhes into a supercoil / tangle. Zero for a statically held end.
    pub twist_rate: f32,
    /// When true, the rest state is STRAIGHT and untwisted (a zero rest Darboux)
    /// rather than the authored shape, so a curved or coiled rod springs toward
    /// straight instead of holding its modeled form. `rest_twist` still applies
    /// on top, which is how a straight-rest rod can still be intrinsically
    /// twisted.
    pub straight_rest: bool,
    /// Authored roll of this edge's material frame about its own tangent
    /// (radians), the mean of its two endpoints' curve tilt. Parallel transport
    /// fixes the frame only up to this roll, so applying it is what carries an
    /// artist's twist profile into the rod: neighbouring edges differing in roll
    /// IS an authored twist rate, and the rest Darboux picks it up for free
    /// because it is read from these frames.
    pub roll: f32,
}

/// Bending/twisting couples the two edges incident to an interior node. This is
/// that coupling site: `slot_prev`/`slot_next` index the frame arrays (the
/// reduced edge-DOF ordering), `node` is the shared global vertex, and
/// `rest_darboux` / `inv_len` are the precomputed rest curvature and the
/// reciprocal integration length used by the bending energy.
#[derive(Debug, Clone, Copy)]
pub struct CosseratBendSite {
    pub node: u32,
    pub slot_prev: u32,
    pub slot_next: u32,
    pub rest_darboux: [f32; 3],
    pub inv_len: f32,
}

/// Result of initializing a scene's Cosserat rods. All per-slot arrays are
/// parallel (index = frame slot = reduced edge-DOF index).
#[derive(Debug, Clone, Default)]
pub struct CosseratFrames {
    /// Global edge index for each frame slot (the reduced edge-DOF ordering).
    pub edge_order: Vec<u32>,
    /// Oriented endpoints per slot: the frame's local Z axis points from
    /// `node_from` to `node_to`. The shear energy must build its director from
    /// this orientation (d = normalize(x[node_to] - x[node_from])) so it agrees
    /// with the frame at rest.
    pub node_from: Vec<u32>,
    pub node_to: Vec<u32>,
    /// Initial orientation quaternion `[x, y, z, w]` per frame slot.
    pub quat: Vec<[f32; 4]>,
    /// Orientation-fixed flag per slot: 1 if this edge's orientation is a held
    /// Dirichlet BC (the first/last edge of a rod that opted into fixed ends),
    /// 0 otherwise.
    pub fixed: Vec<u8>,
    /// Signed boundary twist rate (radians per second) per slot: `+rate` on a
    /// chain's first held end, `-rate` on its last held end, 0 elsewhere. The
    /// exp-update twists these held frames about their own tangent by this rate,
    /// so a held cable is twisted at both ends in opposite senses.
    pub end_omega: Vec<f32>,
    /// One entry per interior node of every rod.
    pub bend_sites: Vec<CosseratBendSite>,
}

/// A bend site's REST METRIC: the two factors that make the bend energy's second
/// variation at rest the exact one.
///
/// The energy measures `w = (2 / l) vec(q)`, whose response to a change of the
/// relative turn is `cos(phi / 2) / l` along the turn's axis and
/// `(2 / l) sin(phi / 2) / phi` across it, where the exact rotation vector over
/// `l`, `phi a / l`, responds by `1 / l` both ways. Uncorrected, a joint resting
/// at a turn `phi0` would be `cos^2(phi0 / 2)` softer to bend further or back
/// along its own plane: 5 % at a crimp of 0.47 rad, 21 % at 0.96 rad, and a
/// crossing's joint against a float's straighter one is exactly the comparison a
/// woven cloth's bending rests on. With the metric `M`, `1 / cos(phi0 / 2)` along
/// the rest axis and `(phi0 / 2) / sin(phi0 / 2)` across it, the energy
/// `1/2 l (w - w0)^T M K M (w - w0)` has the exact second variation at rest and
/// keeps `w`'s one-to-one range. Returns `[along, across]`.
///
/// # Panics
/// If the rest Darboux vector is not a turn below a half turn over `l`.
pub fn rest_metric(rest_darboux: [f32; 3], inv_len: f32) -> [f32; 2] {
    let norm = rest_darboux
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();
    let x = 0.5 * norm / f64::from(inv_len);
    assert!(
        x < 1.0 - 1.0e-6,
        "a Cosserat bend site's rest curvature is a turn of half a turn or more over its length \
         (sin(phi / 2) = {x}); its intrinsic twist or authored shape is too sharp for one segment"
    );
    if x < 1.0e-7 {
        return [1.0, 1.0];
    }
    let half = x.asin();
    [(1.0 / (1.0 - x * x).sqrt()) as f32, (half / x) as f32]
}

/// The discrete Darboux vector of the relative rotation `g0` over the length
/// `1 / inv_len`: `axial_diff(g0) / (l sqrt(1 + tr g0))`, which is
/// `(2 / l) sin(phi / 2) a` for a turn `phi` about `a`, the device kernel's
/// formula (`cosserat_bend_grad_hess`).
///
/// # Panics
/// If the relative turn is within about a degree of a half turn, where the
/// vector has no defined direction: two consecutive frames of an authored rod
/// turned that far apart are a fold the rod cannot rest in.
fn darboux_of(g0: &na::Matrix3<f32>, inv_len: f32) -> Vec3f {
    let one_plus = 1.0 + g0.trace();
    assert!(
        one_plus > 1.0e-4,
        "two consecutive Cosserat frames are turned a half turn apart (1 + tr = {one_plus}); \
         the rod's authored shape folds back on itself there"
    );
    let scale = inv_len / one_plus.sqrt();
    Vec3f::new(
        scale * (g0[(2, 1)] - g0[(1, 2)]),
        scale * (g0[(0, 2)] - g0[(2, 0)]),
        scale * (g0[(1, 0)] - g0[(0, 1)]),
    )
}

/// A stable unit vector perpendicular to `t` (assumed unit). Picks the world
/// axis least aligned with `t` and orthogonalizes, avoiding a near-degenerate
/// cross product.
fn any_perpendicular(t: &Vec3f) -> Vec3f {
    let ax = t.x.abs();
    let ay = t.y.abs();
    let az = t.z.abs();
    let world = if ax <= ay && ax <= az {
        Vec3f::new(1.0, 0.0, 0.0)
    } else if ay <= az {
        Vec3f::new(0.0, 1.0, 0.0)
    } else {
        Vec3f::new(0.0, 0.0, 1.0)
    };
    let p = world - t * t.dot(&world);
    p.normalize()
}

/// Build the SO(3) frame of the first edge in a chain: Z = tangent, X a stable
/// perpendicular, Y = Z x X.
fn seed_frame(t: &Vec3f) -> na::Rotation3<f32> {
    let z = *t;
    let x = any_perpendicular(&z);
    let y = z.cross(&x);
    // Columns are the local axes expressed in world coordinates.
    let m = na::Matrix3::from_columns(&[x, y, z]);
    na::Rotation3::from_matrix_unchecked(m)
}

/// Minimal rotation carrying unit vector `a` onto unit vector `b`, built from
/// the half-angle quaternion q proportional to (1 + a.b, a x b): for unit
/// inputs with angle theta, 1 + a.b = 2 cos^2(theta/2) and |a x b| =
/// 2 sin(theta/2) cos(theta/2), so normalizing q yields exactly the minimal
/// rotation, with NO acos and NO square root of a possibly-negative number.
///
/// This construction is deliberate, not a style choice: nalgebra's
/// `Rotation3::rotation_between` computes `acos(a.dot(b))` on the RAW dot
/// product, and for two nearly-identical f32 tangents the dot rounds ABOVE 1,
/// making acos return NaN. A near-straight rod (straight plus a sub-millimeter
/// wobble) hits that on real geometry, and because the chain is framed by
/// SEQUENTIAL parallel transport, one NaN poisons every downstream frame and
/// rest Darboux, which then NaNs the assembled system. The half-angle form is
/// finite for every input pair; it degenerates only at exactly-opposed vectors
/// (w -> 0), where we rotate pi about any perpendicular axis (a fold, which a
/// well-formed rod does not produce, but we stay defined).
fn minimal_rotation(a: &Vec3f, b: &Vec3f) -> na::Rotation3<f32> {
    let c = a.cross(b);
    let w = 1.0 + a.dot(b);
    if w > 1e-6 {
        let q = na::Quaternion::new(w, c.x, c.y, c.z);
        na::UnitQuaternion::new_normalize(q).to_rotation_matrix()
    } else {
        let axis = any_perpendicular(a);
        na::Rotation3::from_axis_angle(
            &na::Unit::new_normalize(axis),
            std::f32::consts::PI,
        )
    }
}

fn quat_xyzw(r: &na::Rotation3<f32>) -> [f32; 4] {
    let q = na::UnitQuaternion::from_rotation_matrix(r);
    [q.i, q.j, q.k, q.w]
}

/// Signed angle, about the unit axis `axis`, that carries unit vector `from`
/// onto unit vector `to` (both assumed perpendicular to `axis`). Uses atan2, so
/// it is exact over the full (-pi, pi] range rather than only near zero.
fn signed_angle_about(from: &Vec3f, to: &Vec3f, axis: &Vec3f) -> f32 {
    let s = from.cross(to).dot(axis);
    let c = from.dot(to);
    s.atan2(c)
}

/// Initialize per-edge frames and per-node rest Darboux vectors for every
/// Cosserat rod described by `edges`. Rods are reconstructed from the
/// shared-endpoint adjacency of the given edges; each connected component is
/// framed independently by parallel transport.
///
/// A "rod" here is a maximal CHAIN: a path whose interior nodes each join
/// exactly two Cosserat edges. Every component shape is framed by the same
/// walk, which simply breaks wherever a node is not a simple pass-through:
///
/// - OPEN CHAIN (two endpoints): seeded at an endpoint and transported to the
///   other, so the initial twist rate is zero.
/// - BRANCHED ROD (a node joining three or more edges): split at its junctions
///   into one chain per branch, each framed independently. This is a real
///   limitation stated plainly rather than a silent approximation: bending
///   couples the TWO edges meeting at a node, so a junction of three or more
///   has no defined Darboux vector and gets NO bend site. The branches'
///   orientations are therefore independent AT the junction: twisting one
///   branch does not twist another through it. Their POSITIONS remain fully
///   coupled, because the branches genuinely share that vertex and the stretch
///   and shear energies act on it as on any other, so a branched rod holds
///   together and only its junction torque transfer is missing. A junction is
///   not a rod end either, so `fix_ends` and the end twist drive apply only at
///   real (single-edge) ends and never at a junction.
/// - CLOSED LOOP (no endpoint): seeded at its lowest-numbered edge and
///   transported all the way around. Parallel transport around a closed curve
///   does NOT return the frame to its seed: it comes back rotated about the
///   tangent by a holonomy angle `phi` (for a closed curve this is fixed by the
///   geometry, -2 pi times the writhe, and no framing can remove it; a planar
///   ring has phi = 0). Left raw, the whole of `phi` lands on the single node
///   where the walk closes, giving that one node a large spurious rest twist
///   and an artificially stiff spot, while every other node reads zero. Instead
///   the loop's frames are rotated by `k phi / N` about their own tangents,
///   which spreads the holonomy evenly: EVERY one of the N transitions then
///   carries exactly `phi / N` (including the closing one), so the rest Darboux
///   is uniform around the ring, which is the physical rest state of a ring.
///
/// Every Cosserat edge comes out framed, whatever the topology.
pub fn build_cosserat_frames(edges: &[CosseratEdgeInput]) -> CosseratFrames {
    let mut out = CosseratFrames::default();
    if edges.is_empty() {
        return out;
    }

    // Node -> list of local edge indices (into `edges`) incident on it.
    let mut incidence: std::collections::HashMap<u32, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, e) in edges.iter().enumerate() {
        incidence.entry(e.v0).or_default().push(i);
        incidence.entry(e.v1).or_default().push(i);
    }
    // frame slot per local edge index (usize::MAX until assigned).
    let mut slot_of = vec![u32::MAX; edges.len()];
    let mut visited = vec![false; edges.len()];

    // A chain BREAKS at any node that is not a simple pass-through, i.e. whose
    // degree is not exactly two: degree 1 is a rod end, degree >= 3 a junction.
    // Both terminate a chain, which is what splits a branched rod into one
    // chain per branch without any branch-specific code.
    let breaks: Vec<u32> = incidence
        .iter()
        .filter(|(_, inc)| inc.len() != 2)
        .map(|(&node, _)| node)
        .collect();

    // Deterministic order: sort so the frame-slot numbering is stable run to run
    // (the reduced DOF ordering must not depend on HashMap order).
    let mut breaks = breaks;
    breaks.sort_unstable();

    // Walk order. Every open chain is framed first, seeded at an endpoint.
    // Whatever edge is still unvisited afterwards belongs, by definition, to a
    // component with no endpoint: a closed loop. Seeding loops only after the
    // endpoint pass is what keeps a chain from ever being entered mid-span, and
    // makes the loop pass a plain scan in edge order (so the frame-slot
    // numbering stays deterministic run to run).
    // EVERY edge at a break node seeds a chain, not just the first: a junction
    // of degree three starts three chains. The visited guard drops a seed whose
    // chain another break node already walked from its far side.
    let mut starts: Vec<(u32, usize, bool)> = Vec::new();
    for &n in &breaks {
        for &e in &incidence[&n] {
            starts.push((n, e, false));
        }
    }

    let mut next_start = 0usize;
    let mut loops_seeded = false;
    loop {
        if next_start >= starts.len() {
            if loops_seeded {
                break;
            }
            loops_seeded = true;
            // Every edge still unvisited once the chains are framed seeds a
            // loop. Pushing all of them is deliberate and costs nothing: the
            // first seed of a ring frames the whole ring, and the visited guard
            // below drops that ring's remaining seeds.
            for i in 0..edges.len() {
                if !visited[i] {
                    starts.push((edges[i].v0, i, true));
                }
            }
            if next_start >= starts.len() {
                break;
            }
        }
        let (start_node, first_local, is_loop) = starts[next_start];
        next_start += 1;
        if visited[first_local] {
            continue; // already framed, by another endpoint or as part of a ring.
        }

        // Walk the chain, orienting each edge so its first node is the one we
        // arrive from (director points "forward" along the walk).
        let mut chain: Vec<(usize, na::Rotation3<f32>, Vec3f)> = Vec::new();
        // Node the walk stops on; with `start_node` it says whether each end of
        // this chain is a real rod end or a junction.
        let mut terminal_node = start_node;
        let mut cur_node = start_node;
        let mut cur_local = first_local;
        let mut prev_rot: Option<na::Rotation3<f32>> = None;
        let mut prev_forward: Option<Vec3f> = None;

        loop {
            visited[cur_local] = true;
            let e = &edges[cur_local];
            // Forward tangent along the walk direction (cur_node -> other).
            let (forward, _next_node) = if e.v0 == cur_node {
                (e.tangent, e.v1)
            } else {
                (-e.tangent, e.v0)
            };
            let next_node = if e.v0 == cur_node { e.v1 } else { e.v0 };

            // The transported (unrolled) frame. This, NOT the rolled result, is
            // what the next edge transports from: carrying the rolled frame
            // forward would make each edge's roll accumulate into the next, so a
            // CONSTANT tilt would read as a growing twist along the rod instead
            // of no twist at all.
            let transported = match (prev_rot, prev_forward) {
                (Some(pr), Some(pf)) => minimal_rotation(&pf, &forward) * pr,
                _ => seed_frame(&forward),
            };
            let mut rot = transported;
            // Authored roll about the frame's OWN tangent (a right-multiply, so
            // Z stays the tangent and only the material X/Y move). Applied AFTER
            // transport and NOT fed back into `prev_rot`, so it is an absolute
            // offset from the transported reference rather than something that
            // accumulates along the walk: two edges with equal tilt stay
            // untwisted relative to each other, which is what an artist means by
            // a constant tilt.
            let edge_roll = edges[cur_local].roll;
            if edge_roll != 0.0 {
                rot *= na::Rotation3::from_axis_angle(
                    &na::Vector3::z_axis(),
                    edge_roll,
                );
            }
            let slot = out.quat.len() as u32;
            slot_of[cur_local] = slot;
            out.edge_order.push(e.edge);
            out.node_from.push(cur_node);
            out.node_to.push(next_node);
            out.quat.push(quat_xyzw(&rot));
            out.fixed.push(0); // interior default; chain ends set below.
            out.end_omega.push(0.0); // interior default; chain ends set below.
            chain.push((cur_local, rot, forward));

            prev_rot = Some(transported);
            prev_forward = Some(forward);

            // Advance only THROUGH a simple pass-through node. At a junction
            // (degree >= 3) there is no next edge to prefer and no defined
            // Darboux pair, so the chain ends here and the other branches are
            // walked as their own chains from this same node.
            let inc = &incidence[&next_node];
            let mut advanced = false;
            if inc.len() == 2 {
                for &cand in inc {
                    if !visited[cand] {
                        cur_node = next_node;
                        cur_local = cand;
                        advanced = true;
                        break;
                    }
                }
            }
            if !advanced {
                terminal_node = next_node;
            }
            if !advanced {
                // A chain has reached its far endpoint; a ring has come back
                // round to its already-framed seed.
                break;
            }
        }

        // A ring's holonomy: transport the LAST frame back onto the first
        // edge's tangent and measure how far it missed the seed, as a rotation
        // about that tangent. Parallel transport around a closed curve cannot
        // close in general (the miss is -2 pi times the writhe, a property of
        // the curve, not of our seeding), so the angle has nowhere to go: it can
        // only be placed. Spread it evenly by twisting frame k about its own
        // tangent by k*phi/N. Then all N transitions, the closing one included,
        // carry exactly phi/N, and the ring's rest Darboux comes out uniform
        // instead of flat everywhere with one huge spike at the seam. A planar
        // ring has phi = 0 and is untouched.
        if is_loop && chain.len() >= 3 {
            let n = chain.len();
            let (_, r_first, t_first) = chain[0];
            let (_, r_last, t_last) = chain[n - 1];
            let r_close = minimal_rotation(&t_last, &t_first) * r_last;
            // Both frames' Z is t_first, so their X axes are perpendicular to it
            // and the miss is the signed angle between them about it.
            let phi = signed_angle_about(
                &r_close.matrix().column(0).into(),
                &r_first.matrix().column(0).into(),
                &t_first,
            );
            let step = phi / n as f32;
            for (k, entry) in chain.iter_mut().enumerate() {
                if k == 0 {
                    continue; // the seed anchors the framing.
                }
                // Right-multiply: a rotation about the frame's OWN Z, which is
                // its tangent, so the director is untouched and only the roll
                // moves.
                let twist = na::Rotation3::from_axis_angle(
                    &na::Vector3::z_axis(),
                    step * k as f32,
                );
                entry.1 *= twist;
                out.quat[slot_of[entry.0] as usize] = quat_xyzw(&entry.1);
            }
        }

        // Rest Darboux at each interior node: between consecutive chain edges.
        // A ring additionally has a site at its seam (last edge -> first edge),
        // which `windows(2)` does not cover, so it is appended below.
        let mut pairs: Vec<(usize, na::Rotation3<f32>, usize, na::Rotation3<f32>)> =
            chain
                .windows(2)
                .map(|w| (w[0].0, w[0].1, w[1].0, w[1].1))
                .collect();
        if is_loop && chain.len() >= 3 {
            let (last_local, r_last, _) = chain[chain.len() - 1];
            let (first_local, r_first, _) = chain[0];
            pairs.push((last_local, r_last, first_local, r_first));
        }
        for (local_prev, r_prev, local_next, r_next) in pairs {
            // Shared node is the one common to both edges.
            let ep = &edges[local_prev];
            let en = &edges[local_next];
            let node = if ep.v0 == en.v0 || ep.v0 == en.v1 {
                ep.v0
            } else {
                ep.v1
            };
            let l = 0.5 * (ep.length + en.length);
            let inv_len = if l > 0.0 { 1.0 / l } else { 0.0 };
            // Darboux vector via the RELATIVE rotation G0 = R_prev^T R_next
            // (R_prev is the paper's R_{i-1}, R_next is R_i). This is the
            // fp32-safe form the device kernel uses (it avoids differencing two
            // near-equal absolute rotations for a smooth rod); computing the rest
            // omega with the identical formula makes the deformation Delta omega
            // vanish exactly at the rest pose.
            let g0 = r_prev.matrix().transpose() * r_next.matrix();
            let mut omega = darboux_of(&g0, inv_len);
            // A straight rest discards the shape's own curvature and torsion, so
            // the rod carries the authored form as STRAIN and springs toward
            // straight. Only the geometric part is dropped: the authored
            // intrinsic twist below is an explicit knob and still applies, which
            // is exactly a straight-but-pretwisted rod.
            if ep.straight_rest || en.straight_rest {
                omega = Vec3f::zeros();
            }
            // Add the authored intrinsic twist rate (mean of the two edges) to
            // the twist component (local Z / tangent). A non-zero value makes a
            // straight rod intrinsically twisted (helical perversion).
            omega.z += 0.5 * (ep.rest_twist + en.rest_twist);
            out.bend_sites.push(CosseratBendSite {
                node,
                slot_prev: slot_of[local_prev],
                slot_next: slot_of[local_next],
                rest_darboux: [omega.x, omega.y, omega.z],
                inv_len,
            });
        }

        // End-of-rod orientation pins: hold the first and last edge's frame
        // fixed when the rod opted into fixed ends. Holding the twist at the
        // boundary is what lets an intrinsic twist buckle the centerline.
        if let (Some(&(first_local, _, _)), Some(&(last_local, _, _))) =
            (chain.first(), chain.last())
        {
            if is_loop {
                // A ring has no ends, so these have nothing to hold or turn.
                // Silently ignoring them would leave a user staring at a knob
                // that does nothing, so refuse the scene instead.
                assert!(
                    !edges[first_local].fix_ends,
                    "Cosserat edge {} lies on a closed loop with \
                     cosserat-fix-ends set; a loop has no end frames to hold",
                    edges[first_local].edge
                );
                assert!(
                    edges[first_local].twist_rate == 0.0,
                    "Cosserat edge {} lies on a closed loop with a non-zero \
                     cosserat-end-twist-rate; a loop has no end frames to turn",
                    edges[first_local].edge
                );
            } else {
                // Only a REAL rod end (a node with one edge) can be held. A
                // chain of a branched rod also stops at its junctions, and a
                // junction is not an end: holding a frame there would clamp the
                // middle of the rod at a spot the user never pointed at, so
                // each side is gated on its own node's degree.
                let start_is_end = incidence[&start_node].len() == 1;
                let terminal_is_end = incidence[&terminal_node].len() == 1;
                if edges[first_local].fix_ends && start_is_end {
                    out.fixed[slot_of[first_local] as usize] = 1;
                    // First held end: twist in the +rate sense about its tangent.
                    out.end_omega[slot_of[first_local] as usize] =
                        edges[first_local].twist_rate;
                }
                if edges[last_local].fix_ends && terminal_is_end {
                    out.fixed[slot_of[last_local] as usize] = 1;
                    // Last held end: twist in the -rate (opposite) sense.
                    out.end_omega[slot_of[last_local] as usize] =
                        -edges[last_local].twist_rate;
                }
            }
        }
    }

    // Every Cosserat edge must come out framed: open chains by the endpoint
    // pass, rings by the loop pass. That makes this unreachable by
    // construction, and it stays anyway because it costs one scan and the
    // failure it guards is silent: an unframed edge is still claimed by the
    // `cosserat_edge` marker but covered by no frame, so it would quietly lose
    // its orientation physics rather than fail.
    let unvisited: Vec<u32> = visited
        .iter()
        .enumerate()
        .filter(|(_, &v)| !v)
        .map(|(i, _)| edges[i].edge)
        .collect();
    assert!(
        unvisited.is_empty(),
        "Cosserat edges {unvisited:?} were not framed by any walk"
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(edge: u32, v0: u32, v1: u32, t: Vec3f, len: f32) -> CosseratEdgeInput {
        CosseratEdgeInput {
            edge,
            v0,
            v1,
            tangent: t.normalize(),
            length: len,
            rest_twist: 0.0,
            fix_ends: false,
            twist_rate: 0.0,
            straight_rest: false,
            roll: 0.0,
        }
    }

    #[test]
    fn straight_rod_has_zero_rest_darboux() {
        // Five collinear edges along +X.
        let t = Vec3f::new(1.0, 0.0, 0.0);
        let edges: Vec<_> = (0..5)
            .map(|i| edge(i, i, i + 1, t, 1.0))
            .collect();
        let f = build_cosserat_frames(&edges);
        assert_eq!(f.quat.len(), 5);
        assert_eq!(f.bend_sites.len(), 4);
        for s in &f.bend_sites {
            let mag = (s.rest_darboux[0].powi(2)
                + s.rest_darboux[1].powi(2)
                + s.rest_darboux[2].powi(2))
            .sqrt();
            assert!(mag < 1e-6, "straight rod rest Darboux should vanish, got {mag}");
        }
    }

    #[test]
    fn intrinsic_rest_twist_appears_in_rest_darboux() {
        // A straight rod (zero geometric curvature) with an authored rest twist
        // stores exactly that twist in the rest Darboux's twist component, and
        // zero bending; that stored twist is what buckles a straightened rod.
        let t = Vec3f::new(1.0, 0.0, 0.0);
        let twist = 3.0_f32;
        let edges: Vec<_> = (0..5)
            .map(|i| {
                let mut e = edge(i, i, i + 1, t, 1.0);
                e.rest_twist = twist;
                e
            })
            .collect();
        let f = build_cosserat_frames(&edges);
        for s in &f.bend_sites {
            // Bending components (X, Y) stay ~0; twist (Z) equals the authored rate.
            assert!(s.rest_darboux[0].abs() < 1e-5);
            assert!(s.rest_darboux[1].abs() < 1e-5);
            assert!((s.rest_darboux[2] - twist).abs() < 1e-5);
        }
    }

    #[test]
    fn all_frames_have_z_along_tangent() {
        // A gentle zig-zag so tangents differ.
        let ts = [
            Vec3f::new(1.0, 0.0, 0.0),
            Vec3f::new(1.0, 0.3, 0.0),
            Vec3f::new(1.0, -0.2, 0.1),
        ];
        let edges: Vec<_> = ts
            .iter()
            .enumerate()
            .map(|(i, t)| edge(i as u32, i as u32, i as u32 + 1, *t, 1.0))
            .collect();
        let f = build_cosserat_frames(&edges);
        for (slot, e) in edges.iter().enumerate() {
            let q = f.quat[slot];
            let quat = na::UnitQuaternion::from_quaternion(na::Quaternion::new(
                q[3], q[0], q[1], q[2],
            ));
            let r = quat.to_rotation_matrix();
            let z = r.matrix().column(2).into_owned();
            let t = e.tangent.normalize();
            let dot = z.dot(&t);
            assert!((dot - 1.0).abs() < 1e-5, "frame Z must equal tangent, dot={dot}");
        }
    }

    #[test]
    fn circular_arc_rest_darboux_matches_curvature() {
        // Sample a planar arc of radius R in the XY plane; the discrete Darboux
        // magnitude should approach the curvature 1/R as the sampling refines.
        let radius = 2.0_f32;
        let n = 40usize;
        let dphi = 0.02_f32; // small step -> segment length ~ radius*dphi
        // Tangent of a circle param (cos, sin): derivative (-sin, cos).
        let mut edges = Vec::new();
        for i in 0..n {
            let phi = i as f32 * dphi;
            let t = Vec3f::new(-phi.sin(), phi.cos(), 0.0);
            let len = radius * dphi;
            edges.push(edge(i as u32, i as u32, i as u32 + 1, t, len));
        }
        let f = build_cosserat_frames(&edges);
        // Interior sites away from the ends (endpoints of the chain).
        let mid = &f.bend_sites[n / 2 - 1];
        let mag = (mid.rest_darboux[0].powi(2)
            + mid.rest_darboux[1].powi(2)
            + mid.rest_darboux[2].powi(2))
        .sqrt();
        let curvature = 1.0 / radius;
        let rel = (mag - curvature).abs() / curvature;
        assert!(
            rel < 0.05,
            "arc Darboux {mag} should approach curvature {curvature} (rel {rel})"
        );
    }

    #[test]
    fn chain_reconstruction_is_order_independent() {
        // Shuffle the edge input order; frames/sites must be identical because
        // the walk is seeded from the sorted endpoints.
        let t = Vec3f::new(1.0, 0.1, 0.0);
        let mk = |order: &[usize]| {
            let all: Vec<_> = (0..4u32)
                .map(|i| edge(i, i, i + 1, t, 1.0))
                .collect();
            let shuffled: Vec<_> = order.iter().map(|&i| all[i]).collect();
            build_cosserat_frames(&shuffled)
        };
        let a = mk(&[0, 1, 2, 3]);
        let b = mk(&[3, 1, 0, 2]);
        assert_eq!(a.edge_order, b.edge_order);
        for (qa, qb) in a.quat.iter().zip(b.quat.iter()) {
            for k in 0..4 {
                assert!((qa[k] - qb[k]).abs() < 1e-6);
            }
        }
    }

    /// A closed planar ring of `n` edges in the XY plane, radius `r`, given in
    /// walk order so node i joins edge i-1 to edge i.
    fn ring_edges(n: u32, r: f32) -> Vec<CosseratEdgeInput> {
        let pt = |i: u32| {
            let a = 2.0 * std::f32::consts::PI * (i % n) as f32 / n as f32;
            Vec3f::new(r * a.cos(), r * a.sin(), 0.0)
        };
        (0..n)
            .map(|i| {
                let d = pt(i + 1) - pt(i);
                let len = d.norm();
                CosseratEdgeInput {
                    edge: i,
                    v0: i,
                    v1: (i + 1) % n,
                    tangent: d / len,
                    length: len,
                    rest_twist: 0.0,
                    fix_ends: false,
                    twist_rate: 0.0,
                    straight_rest: false,
                    roll: 0.0,
                }
            })
            .collect()
    }

    #[test]
    fn authored_roll_twists_only_where_it_varies() {
        // The tilt semantics, pinned down. A CONSTANT tilt is a rigid roll of
        // the whole material frame and must read as NO twist: it is the tilt's
        // variation along the rod, not its value, that is a twist rate. A
        // linearly varying tilt must read as a uniform twist of d(tilt)/ds.
        let t = Vec3f::new(1.0, 0.0, 0.0);
        let n = 6u32;
        let len = 0.5f32;
        let build = |rolls: &[f32]| {
            let edges: Vec<_> = (0..n)
                .map(|i| CosseratEdgeInput {
                    edge: i,
                    v0: i,
                    v1: i + 1,
                    tangent: t,
                    length: len,
                    rest_twist: 0.0,
                    fix_ends: false,
                    twist_rate: 0.0,
                    straight_rest: false,
                    roll: rolls[i as usize],
                })
                .collect();
            build_cosserat_frames(&edges)
        };
        // Constant tilt -> no twist anywhere.
        let flat = build(&[0.7; 6]);
        for b in &flat.bend_sites {
            assert!(
                b.rest_darboux[2].abs() < 1e-5,
                "a constant tilt must not twist the rod: {}",
                b.rest_darboux[2]
            );
        }
        // Linearly varying tilt -> uniform twist of step/len per site.
        let step = 0.1f32;
        let rolls: Vec<f32> = (0..n).map(|i| step * i as f32).collect();
        let ramp = build(&rolls);
        for b in &ramp.bend_sites {
            let want = step / len; // d(tilt)/ds, with inv_len = 1/len here
            assert!(
                (b.rest_darboux[2] - want).abs() < 1e-3,
                "a tilt ramp must read as its rate: got {} want {want}",
                b.rest_darboux[2]
            );
            assert!(
                b.rest_darboux[0].hypot(b.rest_darboux[1]) < 1e-5,
                "a straight rod must carry no rest bend"
            );
        }
    }

    #[test]
    fn straight_rest_zeroes_the_shape_curvature_but_keeps_intrinsic_twist() {
        // A curved rod normally takes its own shape as rest (zero energy at
        // frame 0). With straight_rest it must instead read a zero rest
        // curvature, so the modeled bend becomes strain that springs it
        // straight, while an authored intrinsic twist still stands.
        let r = 2.0f32;
        let n = 8u32;
        let arc = |straight: bool, twist: f32| {
            let pt = |i: u32| {
                let a = 0.4 * i as f32 / n as f32;
                Vec3f::new(r * a.cos(), r * a.sin(), 0.0)
            };
            let edges: Vec<_> = (0..n)
                .map(|i| {
                    let d = pt(i + 1) - pt(i);
                    let len = d.norm();
                    CosseratEdgeInput {
                        edge: i,
                        v0: i,
                        v1: i + 1,
                        tangent: d / len,
                        length: len,
                        rest_twist: twist,
                        fix_ends: false,
                        twist_rate: 0.0,
                        straight_rest: straight,
                        roll: 0.0,
                    }
                })
                .collect();
            build_cosserat_frames(&edges)
        };
        // Default: the arc's curvature IS its rest, so rest bend is non-zero.
        let curved = arc(false, 0.0);
        let bend_default = curved.bend_sites[0].rest_darboux[0].hypot(curved.bend_sites[0].rest_darboux[1]);
        assert!(bend_default > 1e-3, "an arc's rest should carry its curvature");
        // straight_rest: the geometric part is gone everywhere.
        let straight = arc(true, 0.0);
        for b in &straight.bend_sites {
            let bend = b.rest_darboux[0].hypot(b.rest_darboux[1]);
            assert!(bend < 1e-6, "straight rest must carry no rest curvature: {bend:e}");
            assert!(b.rest_darboux[2].abs() < 1e-6, "and no rest twist");
        }
        // straight_rest + intrinsic twist: only the twist survives.
        let twisted = arc(true, 3.0);
        for b in &twisted.bend_sites {
            assert!(b.rest_darboux[0].hypot(b.rest_darboux[1]) < 1e-6);
            assert!(
                (b.rest_darboux[2] - 3.0).abs() < 1e-5,
                "authored intrinsic twist must survive a straight rest"
            );
        }
    }

    /// A Y: a trunk 0-1-2 with two branches leaving node 2, so node 2 joins
    /// three edges. Edge ids are global; node 2 is the junction.
    fn y_edges() -> Vec<CosseratEdgeInput> {
        let x = Vec3f::new(1.0, 0.0, 0.0);
        let up = Vec3f::new(0.0, 1.0, 0.0);
        vec![
            edge(0, 0, 1, x, 1.0),   // trunk
            edge(1, 1, 2, x, 1.0),   // trunk, into the junction
            edge(2, 2, 3, x, 1.0),   // branch A
            edge(3, 2, 4, up, 1.0),  // branch B
        ]
    }

    #[test]
    fn branched_rod_splits_at_its_junction() {
        // Every edge must still be framed, and the junction must carry NO bend
        // site: bending couples the TWO edges at a node, and node 2 has three,
        // so there is no pair to couple. The sites that remain are the ones
        // inside each branch.
        let f = build_cosserat_frames(&y_edges());
        assert_eq!(f.quat.len(), 4, "every branch edge is framed");
        for q in &f.quat {
            assert!(q.iter().all(|x| x.is_finite()));
        }
        let junction_sites = f.bend_sites.iter().filter(|b| b.node == 2).count();
        assert_eq!(junction_sites, 0, "a junction has no defined Darboux pair");
        // Node 1 (degree 2, inside the trunk) DOES couple.
        assert_eq!(f.bend_sites.iter().filter(|b| b.node == 1).count(), 1);
        // Nothing else can couple: nodes 0/3/4 are ends, node 2 is the junction.
        assert_eq!(f.bend_sites.len(), 1);
    }

    #[test]
    fn fix_ends_holds_real_ends_not_the_junction() {
        // fix_ends on a branched rod must clamp only its actual tips (nodes 0,
        // 3, 4), never the junction: a chain stopping at a junction stops in the
        // MIDDLE of the rod, and holding a frame there would clamp a spot the
        // user never pointed at.
        let mut edges = y_edges();
        for e in edges.iter_mut() {
            e.fix_ends = true;
        }
        let f = build_cosserat_frames(&edges);
        // Three chains (trunk, branch A, branch B) each have one real end, so
        // exactly three frames are held: one per tip.
        let held: usize = f.fixed.iter().filter(|&&x| x == 1).count();
        assert_eq!(held, 3, "one held frame per real rod tip, none at the junction");
    }

    #[test]
    fn closed_loop_is_framed_with_a_site_per_node() {
        // A ring of n edges has n nodes, every one of them interior (no ends),
        // so it must produce n frames AND n bend sites: the seam site (last edge
        // -> first edge) is the one windows(2) cannot see.
        let n = 12u32;
        let f = build_cosserat_frames(&ring_edges(n, 1.0));
        assert_eq!(f.quat.len(), n as usize, "one frame per ring edge");
        assert_eq!(f.bend_sites.len(), n as usize, "one site per ring node");
        // Every node index appears exactly once.
        let mut nodes: Vec<u32> = f.bend_sites.iter().map(|b| b.node).collect();
        nodes.sort_unstable();
        nodes.dedup();
        assert_eq!(nodes.len(), n as usize, "each ring node has exactly one site");
        for q in &f.quat {
            assert!(q.iter().all(|x| x.is_finite()));
        }
        // A ring has no ends to hold or drive.
        assert!(f.fixed.iter().all(|&x| x == 0));
        assert!(f.end_omega.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn planar_ring_rest_darboux_is_uniform_curvature_and_zero_twist() {
        // The physical check. A planar ring is a curve of constant curvature
        // 1/r and zero torsion, so its rest Darboux must read the SAME bend on
        // every node (the seam included) and no twist anywhere. This is what
        // the holonomy spreading buys: without it the seam node would carry the
        // entire closure angle as a spurious twist spike.
        let n = 16u32;
        let r = 2.0f32;
        let f = build_cosserat_frames(&ring_edges(n, r));
        let twists: Vec<f32> = f.bend_sites.iter().map(|b| b.rest_darboux[2]).collect();
        let bends: Vec<f32> = f
            .bend_sites
            .iter()
            .map(|b| (b.rest_darboux[0].powi(2) + b.rest_darboux[1].powi(2)).sqrt())
            .collect();
        let max_twist = twists.iter().fold(0.0f32, |m, t| m.max(t.abs()));
        let bmin = bends.iter().cloned().fold(f32::INFINITY, f32::min);
        let bmax = bends.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_twist < 1e-4, "planar ring must carry no rest twist, got {max_twist:e}");
        assert!(
            bmax - bmin < 1e-3 * bmax.max(1e-6),
            "planar ring curvature must be uniform: {bmin} .. {bmax}"
        );
        // And it must be the actual curvature of the circle, 1/r. The discrete
        // measure is the chord-wise turn per unit length, so it approaches 1/r
        // from below as n grows; 16 segments is within a few percent.
        let want = 1.0 / r;
        assert!(
            (bmax - want).abs() < 0.05 * want,
            "ring curvature {bmax} should be ~1/r = {want}"
        );
    }

    #[test]
    fn ring_and_open_rod_coexist() {
        // The two walks must not interfere: the endpoint pass frames the chain,
        // the loop pass then claims what is left.
        let t = Vec3f::new(1.0, 0.0, 0.0);
        let mut edges: Vec<_> = (0..3u32).map(|i| edge(i, 100 + i, 101 + i, t, 1.0)).collect();
        let mut ring = ring_edges(6, 1.0);
        for (k, e) in ring.iter_mut().enumerate() {
            e.edge = 3 + k as u32; // distinct global ids; ring nodes 0..5 are disjoint
        }
        edges.extend(ring);
        let f = build_cosserat_frames(&edges);
        assert_eq!(f.quat.len(), 9, "3 chain edges + 6 ring edges all framed");
        // 2 interior nodes on the chain + 6 on the ring.
        assert_eq!(f.bend_sites.len(), 8);
    }

    #[test]
    #[should_panic(expected = "no end frames to hold")]
    fn fix_ends_on_a_closed_loop_is_rejected() {
        // A ring has no end frames, so cosserat-fix-ends has nothing to hold;
        // it must say so rather than quietly do nothing.
        let mut edges = ring_edges(8, 1.0);
        for e in edges.iter_mut() {
            e.fix_ends = true;
        }
        let _ = build_cosserat_frames(&edges);
    }

    #[test]
    fn near_straight_wobble_frames_are_finite() {
        // Regression geometry for the near-straight-rod NaN (a straight rod
        // plus a tiny 3D sine wobble, snapped to a fine fixed grid and
        // differenced in f32 exactly like the builder does). Every frame
        // quaternion and rest Darboux must stay finite at every amplitude.
        let n = 40usize;
        let quant = |v: f64| {
            let q = (1u64 << 27) as f64;
            (v * q).round() / q
        };
        for &seed in &[0.0f64, 5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 8e-3] {
            let mut pts = Vec::new();
            for i in 0..n {
                let s = i as f64 / (n - 1) as f64;
                pts.push([
                    quant(s * 1.5),
                    quant(seed * (std::f64::consts::PI * s).sin() + 1.0),
                    quant(0.4 * seed * (2.0 * std::f64::consts::PI * s).sin()),
                ]);
            }
            let mut inputs = Vec::new();
            for i in 0..n - 1 {
                let d = Vec3f::new(
                    (pts[i + 1][0] - pts[i][0]) as f32,
                    (pts[i + 1][1] - pts[i][1]) as f32,
                    (pts[i + 1][2] - pts[i][2]) as f32,
                );
                let len = d.norm();
                inputs.push(CosseratEdgeInput {
                    edge: i as u32,
                    v0: i as u32,
                    v1: (i + 1) as u32,
                    tangent: d / len,
                    length: len,
                    rest_twist: 0.0,
                    fix_ends: true,
                    twist_rate: 0.0,
                    straight_rest: false,
                    roll: 0.0,
                });
            }
            let f = build_cosserat_frames(&inputs);
            let bad_q: Vec<usize> = f
                .quat
                .iter()
                .enumerate()
                .filter(|(_, q)| q.iter().any(|x| !x.is_finite()))
                .map(|(i, _)| i)
                .collect();
            let bad_d: Vec<usize> = f
                .bend_sites
                .iter()
                .enumerate()
                .filter(|(_, b)| b.rest_darboux.iter().any(|x| !x.is_finite()))
                .map(|(i, _)| i)
                .collect();
            assert!(
                bad_q.is_empty() && bad_d.is_empty(),
                "seed={seed:e}: non-finite quats at slots {bad_q:?}, \
                 non-finite rest Darboux at sites {bad_d:?}"
            );
        }
    }

    #[test]
    fn end_twist_rate_drives_the_two_ends_oppositely() {
        // A held chain with a boundary twist rate: the two ends get opposite
        // signs, the interior stays zero, and the whole thing sums to zero.
        let t = Vec3f::new(1.0, 0.0, 0.0);
        let rate = 2.5f32;
        let edges: Vec<_> = (0..4u32)
            .map(|i| CosseratEdgeInput {
                edge: i,
                v0: i,
                v1: i + 1,
                tangent: t.normalize(),
                length: 1.0,
                rest_twist: 0.0,
                fix_ends: true,
                twist_rate: rate,
                straight_rest: false,
                roll: 0.0,
            })
            .collect();
        let f = build_cosserat_frames(&edges);
        // Exactly two non-zero entries (the two ends), opposite in sign.
        let nonzero: Vec<f32> = f.end_omega.iter().copied().filter(|&w| w != 0.0).collect();
        assert_eq!(nonzero.len(), 2, "only the two ends should be driven");
        assert!((nonzero[0] + nonzero[1]).abs() < 1e-6, "ends must be opposite");
        assert!((nonzero.iter().map(|w| w.abs()).sum::<f32>() - 2.0 * rate).abs() < 1e-6);
        // The interior slots are undriven.
        let driven_interior = f
            .end_omega
            .iter()
            .zip(f.fixed.iter())
            .filter(|(&w, &fx)| w != 0.0 && fx == 0)
            .count();
        assert_eq!(driven_interior, 0, "no interior edge is driven");
    }
}
