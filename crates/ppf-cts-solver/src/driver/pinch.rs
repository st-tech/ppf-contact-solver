// File: crates/ppf-cts-solver/src/driver/pinch.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! Which pins and colliders hold a failed step.
//!
//! A Newton stall or a CCD failure in a scene with prescribed geometry is, in
//! the common case, cloth caught where a prescribed surface closes on other
//! geometry that cannot yield (public issue #160: a shape-keyed body folding
//! onto itself with a dress inside the fold), or a pinned part of the cloth
//! itself driven into such geometry (the same issue with Soft Constraint on the
//! body: a dress pinned to follow the arm, carried into a torso that keeps its
//! exact pin). The report used to name only the cloth vertex the line search
//! clamps on, which tells an artist nothing they can act on. What they can act
//! on is the OBJECT that cannot yield: a collider's Soft Constraint, a pin's
//! Pull mode, or the animation.
//!
//! WHY THIS RUNS AT THE STALL AND NOT BEFORE THE STEP.
//!
//! - Once a prescribed surface is driven past the room the cloth between it
//!   and another non-yielding surface needs, no accepted iterate reaches the
//!   step end, so the stall is certain and the Newton cap is only how long the
//!   creep lasts.
//! - That verdict does not follow from the vertex and its two surfaces alone,
//!   because the cloth may carry the vertex out of the pinch sideways. A check
//!   made before the step from local data would refuse steps that succeed. The
//!   stall is the first moment the failure is established, so the diagnosis is
//!   made there.
//! - Once the collider is held by a spring it is a degree of freedom again and
//!   an admissible state exists for every target, which is why the report
//!   recommends Soft Constraint. A pinned cloth vertex is the same case one
//!   level down: as a Pull pin it is a spring too, so the report recommends
//!   that for a pin, and Soft Constraint for whatever exact collider it is
//!   driven into. Soft Constraint on the OTHER body does not help: it lets that
//!   body yield, and neither the pin nor the exact collider moves.
//!
//! The search is host-side and brute force over the non-yielding triangles,
//! paid once, by a run that has already failed.

use crate::data::{DataSet, VertexProp, NO_OBJECT_INDEX};

/// What makes a surface non-yielding, which decides what the artist changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hold {
    /// A collider in the solved namespace whose every vertex is fix-pinned: a
    /// moving STATIC without Soft Constraint. Remedy: its Soft Constraint.
    Collider,
    /// Fix-pinned vertices of a cloth (or any non-collider) object: a pin that
    /// follows its path exactly. Remedy: make it a Pull pin.
    Pin,
    /// The static collision mesh: a STATIC that never moves and has no Soft
    /// Constraint. Remedy: its Soft Constraint, which promotes it to a body
    /// held by springs.
    Static,
}

/// One non-yielding surface near the stuck vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Presser {
    /// The object the surface belongs to, or `None` when the session carries
    /// no object map.
    pub object: Option<u32>,
    pub hold: Hold,
    /// Distance from the stuck vertex to the surface, in solver units.
    pub distance: f32,
}

/// The failed step's pinch: the stuck vertex and the two nearest
/// non-yielding surfaces of distinct objects, nearest first.
#[derive(Clone, Debug, PartialEq)]
pub struct Pinch {
    pub vertex: u32,
    pub stuck_object: Option<u32>,
    /// The stuck vertex is itself fix-pinned: a pin driven into the pressers,
    /// rather than free cloth caught between them.
    pub stuck_pinned: bool,
    pub pressers: Vec<Presser>,
}

fn object_of(prop: &VertexProp) -> Option<u32> {
    (prop.object_index != NO_OBJECT_INDEX).then_some(prop.object_index)
}

/// Closest distance from `p` to triangle `abc` (Ericson, Real-Time Collision
/// Detection, 5.1.5), in f64 because this reads a result and is not solver
/// arithmetic.
pub fn point_triangle_distance(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let sub = |u: [f64; 3], v: [f64; 3]| [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
    let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let at = |s: f64, t: f64| {
        [
            a[0] + s * (b[0] - a[0]) + t * (c[0] - a[0]),
            a[1] + s * (b[1] - a[1]) + t * (c[1] - a[1]),
            a[2] + s * (b[2] - a[2]) + t * (c[2] - a[2]),
        ]
    };
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    let q = if d1 <= 0.0 && d2 <= 0.0 {
        a
    } else {
        let bp = sub(p, b);
        let d3 = dot(ab, bp);
        let d4 = dot(ac, bp);
        if d3 >= 0.0 && d4 <= d3 {
            b
        } else {
            let vc = d1 * d4 - d3 * d2;
            if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
                at(d1 / (d1 - d3), 0.0)
            } else {
                let cp = sub(p, c);
                let d5 = dot(ab, cp);
                let d6 = dot(ac, cp);
                if d6 >= 0.0 && d5 <= d6 {
                    c
                } else {
                    let vb = d5 * d2 - d1 * d6;
                    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
                        at(0.0, d2 / (d2 - d6))
                    } else {
                        let va = d3 * d6 - d5 * d4;
                        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
                            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                            at(1.0 - w, w)
                        } else {
                            let denom = 1.0 / (va + vb + vc);
                            at(vb * denom, vc * denom)
                        }
                    }
                }
            }
        }
    };
    let d = sub(p, q);
    dot(d, d).sqrt()
}

/// The nearest non-yielding surfaces around dynamic vertex `vertex`.
///
/// `positions` holds `3 * vertices` dynamic components at the step start and
/// `props` the dynamic vertex properties. Non-yielding are: a dynamic triangle
/// whose three vertices are all fix-pinned (`fix_index != 0`; a Soft Constraint
/// or a Pull pin holds with pull pins instead, which yield), a fix-pinned
/// vertex of a non-collider object (a pin row need not span a whole triangle),
/// and every collision-mesh triangle. Geometry of the stuck vertex's own object
/// is skipped, because a body is not its own collider here. One presser is
/// kept per object and kind (its nearest element), and the two nearest are
/// returned.
///
/// `None` when the vertex is a collider, when nothing non-yielding of another
/// object exists, or when the vertex is free and nothing PRESCRIBED is among
/// what is near (free cloth on a static floor alone is not a pinch): the
/// failure then has some other cause and the existing report stands alone.
pub fn find(data: &DataSet, positions: &[f32], props: &[VertexProp], vertex: u32) -> Option<Pinch> {
    let v = vertex as usize;
    let prop = props.get(v)?;
    if prop.collider {
        return None;
    }
    let stuck_pinned = prop.fix_index != 0;
    let stuck_object = object_of(prop);
    let point = |i: usize| -> [f64; 3] {
        [
            f64::from(positions[3 * i]),
            f64::from(positions[3 * i + 1]),
            f64::from(positions[3 * i + 2]),
        ]
    };
    if positions.len() < 3 * (v + 1) {
        return None;
    }
    let p = point(v);

    // (object, hold) -> nearest distance.
    let mut best: Vec<Presser> = Vec::new();
    let mut offer = |object: Option<u32>, hold: Hold, distance: f64| {
        if object.is_some() && object == stuck_object {
            return;
        }
        let distance = distance as f32;
        match best.iter_mut().find(|b| b.object == object && b.hold == hold) {
            Some(b) => b.distance = b.distance.min(distance),
            None => best.push(Presser { object, hold, distance }),
        }
    };

    // Fix-pinned dynamic triangles.
    let faces = data.mesh.mesh.face.as_slice();
    let n = positions.len() / 3;
    for f in faces {
        let [a, b, c] = [f[0] as usize, f[1] as usize, f[2] as usize];
        if a >= n || b >= n || c >= n || a == v || b == v || c == v {
            continue;
        }
        let pinned = [a, b, c].iter().all(|&i| props.get(i).is_some_and(|q| q.fix_index != 0));
        if !pinned {
            continue;
        }
        let d = point_triangle_distance(p, point(a), point(b), point(c));
        let hold = if props[a].collider { Hold::Collider } else { Hold::Pin };
        offer(object_of(&props[a]), hold, d);
    }

    // Fix-pinned vertices of cloth (non-collider) objects.
    for (i, q) in props.iter().enumerate().take(n) {
        if i == v || q.fix_index == 0 || q.collider {
            continue;
        }
        let r = point(i);
        let d = ((p[0] - r[0]).powi(2) + (p[1] - r[1]).powi(2) + (p[2] - r[2]).powi(2)).sqrt();
        offer(object_of(q), Hold::Pin, d);
    }

    // The static collision mesh.
    let mesh = &data.constraint.mesh;
    let cv = mesh.vertex.as_slice();
    let cprops = mesh.prop.vertex.as_slice();
    let cpoint = |i: usize| -> [f64; 3] {
        [f64::from(cv[i][0]), f64::from(cv[i][1]), f64::from(cv[i][2])]
    };
    for f in mesh.face.as_slice() {
        let [a, b, c] = [f[0] as usize, f[1] as usize, f[2] as usize];
        if a >= cv.len() || b >= cv.len() || c >= cv.len() {
            continue;
        }
        let d = point_triangle_distance(p, cpoint(a), cpoint(b), cpoint(c));
        offer(cprops.get(a).and_then(object_of), Hold::Static, d);
    }

    best.sort_by(|x, y| x.distance.total_cmp(&y.distance));
    best.truncate(2);
    if best.is_empty() || (!stuck_pinned && best.iter().all(|b| b.hold == Hold::Static)) {
        // Nothing prescribed takes part: the failure is not a pinch.
        return None;
    }
    Some(Pinch { vertex, stuck_object, stuck_pinned, pressers: best })
}

/// The last stall's pinch, for the host to put in the crash detail.
static LAST: std::sync::Mutex<Option<Pinch>> = std::sync::Mutex::new(None);

pub fn record(pinch: Option<Pinch>) {
    if let Ok(mut slot) = LAST.lock() {
        *slot = pinch;
    }
}

/// Take the recorded pinch, if the last failure found one.
pub fn take() -> Option<Pinch> {
    LAST.lock().ok().and_then(|mut slot| slot.take())
}

/// The sentence the log and the crash detail carry. `name` turns an object
/// index into the user's name for it.
pub fn describe(pinch: &Pinch, name: &dyn Fn(Option<u32>) -> String) -> String {
    // The static collision mesh carries no object map in every session, so an
    // unnamed static surface is spelled as the kind alone.
    let surface = |p: &Presser| {
        let what = match (p.hold, p.object) {
            (Hold::Collider, object) => format!("pinned collider {}", name(object)),
            (Hold::Pin, object) => format!("the pinned vertices of {}", name(object)),
            (Hold::Static, Some(object)) => format!("static collider {}", name(Some(object))),
            (Hold::Static, None) => "a static collider".to_string(),
        };
        format!("{what} ({:.3e} away)", p.distance)
    };
    let pressed = match pinch.pressers.as_slice() {
        [one] => surface(one),
        [one, two] => format!("{} and {}", surface(one), surface(two)),
        _ => String::new(),
    };
    // One remedy per object that cannot yield, in the order named.
    let mut remedies: Vec<String> = Vec::new();
    let mut push = |remedy: String| {
        if !remedies.contains(&remedy) {
            remedies.push(remedy);
        }
    };
    if pinch.stuck_pinned {
        push(format!("make the pin on {} a Pull pin", name(pinch.stuck_object)));
    }
    for p in &pinch.pressers {
        match (p.hold, p.object) {
            (Hold::Pin, object) => push(format!("make the pin on {} a Pull pin", name(object))),
            (Hold::Collider, object) | (Hold::Static, object @ Some(_)) => {
                push(format!("enable Soft Constraint on the group of {}", name(object)))
            }
            (Hold::Static, None) => {
                push("enable Soft Constraint on the static collider's group".to_string())
            }
        }
    }
    let lead = if pinch.stuck_pinned {
        format!("A pinned vertex of {} is driven into {pressed}", name(pinch.stuck_object))
    } else {
        format!("{} is caught by {pressed}", name(pinch.stuck_object))
    };
    format!(
        "{lead}. A pin, and a collider without Soft Constraint, follow their path exactly \
         and cannot yield; {}, or re-author the animation so the cloth has room",
        remedies.join(", or ")
    )
}

#[cfg(test)]
mod tests {
    use super::{describe, point_triangle_distance, Hold, Pinch, Presser};

    #[test]
    fn the_report_names_the_pinned_collider_and_the_remedy() {
        let pinch = Pinch {
            vertex: 7,
            stuck_object: Some(0),
            stuck_pinned: false,
            pressers: vec![
                Presser { object: None, hold: Hold::Static, distance: 2.0e-5 },
                Presser { object: Some(1), hold: Hold::Collider, distance: 4.0e-5 },
            ],
        };
        let name = |object: Option<u32>| match object {
            Some(0) => "'Dress'".to_string(),
            Some(1) => "'Body'".to_string(),
            _ => "an unnamed object".to_string(),
        };
        let text = describe(&pinch, &name);
        assert!(text.starts_with("'Dress' is caught by a static collider (2.000e-5 away) and \
                                  pinned collider 'Body' (4.000e-5 away)."), "{text}");
        assert!(text.contains("enable Soft Constraint on the group of 'Body'"), "{text}");
    }

    #[test]
    fn a_pin_driven_into_an_exact_collider_names_the_pin_and_that_collider() {
        // The issue's case with Soft Constraint on the arm: the sleeve's pinned
        // edge follows the arm into a torso that keeps its exact pin. The arm
        // yields, so it is not named; the pin and the torso are.
        let pinch = Pinch {
            vertex: 3,
            stuck_object: Some(2),
            stuck_pinned: true,
            pressers: vec![Presser { object: Some(3), hold: Hold::Static, distance: 1.0e-4 }],
        };
        let name = |object: Option<u32>| match object {
            Some(2) => "'Sleeve'".to_string(),
            Some(3) => "'Torso'".to_string(),
            _ => "an unnamed object".to_string(),
        };
        let text = describe(&pinch, &name);
        assert!(text.starts_with("A pinned vertex of 'Sleeve' is driven into static collider \
                                  'Torso' (1.000e-4 away)."), "{text}");
        assert!(text.contains("make the pin on 'Sleeve' a Pull pin, or enable Soft Constraint \
                               on the group of 'Torso'"), "{text}");
    }

    #[test]
    fn the_distance_reaches_every_region_of_the_triangle() {
        let (a, b, c) = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        // Above the interior, at a vertex region, and at an edge region.
        assert!((point_triangle_distance([0.2, 0.2, 0.5], a, b, c) - 0.5).abs() < 1e-12);
        assert!((point_triangle_distance([-1.0, -1.0, 0.0], a, b, c) - 2f64.sqrt()).abs() < 1e-12);
        assert!((point_triangle_distance([0.5, -2.0, 0.0], a, b, c) - 2.0).abs() < 1e-12);
        assert!((point_triangle_distance([1.0, 1.0, 0.0], a, b, c) - 0.5f64.sqrt()).abs() < 1e-12);
    }
}
