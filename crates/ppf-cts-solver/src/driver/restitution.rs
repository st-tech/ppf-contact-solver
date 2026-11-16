// File: restitution.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//! A PDRD body's coefficient of restitution, applied once per impact.
//!
//! A body with `restitution > 0` integrates with weight `theta = 1/2`, whose
//! converged step with the contacts' secant pushes neither gains nor loses
//! energy, so an
//! impact through the barrier is elastic however many steps it spans and
//! wherever it falls within them. The coefficient is then applied ONCE, when
//! the impact is over: the body's velocity change while it was in contact
//! (gravity's share removed), linear and angular, is the elastic impulse, twice
//! the compression impulse, so scaling it by `(1 + e) / 2` leaves the body
//! moving apart at `e` times its approach speed (Poisson's restitution). Two
//! bodies of one coefficient change by equal and opposite impulses, so the
//! pair's momentum is kept. A coefficient applied this way does not depend on
//! the step size or on where in a step the impact lands, which a weight inside
//! the integrator cannot avoid (its loss is `(theta - 1/2) h^2 |a|^2` per
//! contact STEP, so it follows how the impact is split over steps).
//!
//! WHETHER A BODY IS IN CONTACT is read off the force: the contact passes add
//! into `force` after `residual` is snapshotted from it, and a vertex no
//! contact touched is left exactly at its snapshot. A body is in contact for a
//! step when any of its vertices was touched in any of the step's iterations.

use ppf_cts_compute::{Buffer, Device, ReadbackBuffer};

use super::FatalResult;
use crate::data::DataSet;

/// One body with a coefficient of restitution.
struct Body {
    /// The body's row in the scene's PDRD body table.
    index: usize,
    vertices: Vec<u32>,
    restitution: f64,
    /// Whether the body touched anything in this step, and in the last.
    touched: bool,
    in_impact: bool,
    /// The impact's velocity change so far, gravity's share removed.
    linear: [f64; 3],
    angular: [f64; 3],
}

/// The bodies with a coefficient, and per-vertex scratch.
#[derive(Default)]
pub struct Restitution {
    bodies: Vec<Body>,
    force: Vec<f32>,
    residual: Vec<f32>,
}

impl Restitution {
    /// The scene's bodies with `restitution > 0`, or `None` when it has none.
    ///
    /// # Safety
    /// `data` must address a live `DataSet`.
    pub unsafe fn build(data: &DataSet) -> Option<Self> {
        let values: Vec<f32> = super::scene::slice(&data.prop.pdrd_body)
            .iter()
            .map(|body| body.restitution)
            .collect();
        Self::with_coefficients(None, data, &values)
    }

    /// The bodies with `values[k] > 0` (one value per PDRD body, in the
    /// scene's table order), for a keyframed restitution. A body whose
    /// coefficient stays above zero keeps the impact it is in the middle of;
    /// one that reaches zero leaves the table and takes no release, which is
    /// the plain implicit-Euler step's behaviour; one that rises above zero
    /// starts clear of any impact.
    ///
    /// # Safety
    /// `data` must address a live `DataSet`.
    pub unsafe fn with_coefficients(
        previous: Option<Self>,
        data: &DataSet,
        values: &[f32],
    ) -> Option<Self> {
        let bodies = super::scene::slice(&data.prop.pdrd_body);
        let list = super::scene::slice(&data.pdrd_vert_list);
        let mut kept = previous.map(|p| p.bodies).unwrap_or_default();
        let mut out = Vec::new();
        for (index, (body, &value)) in bodies.iter().zip(values).enumerate() {
            if value <= 0.0 {
                continue;
            }
            if let Some(at) = kept.iter().position(|b| b.index == index) {
                let mut old = kept.swap_remove(at);
                old.restitution = f64::from(value);
                out.push(old);
                continue;
            }
            let start = body.vertex_start as usize;
            let count = body.vertex_count as usize;
            out.push(Body {
                index,
                vertices: list[start..start + count].to_vec(),
                restitution: f64::from(value),
                touched: false,
                in_impact: false,
                linear: [0.0; 3],
                angular: [0.0; 3],
            });
        }
        (!out.is_empty()).then(|| Self { bodies: out, ..Default::default() })
    }

    /// Mark the bodies a contact touched in this iteration, from the force the
    /// contact passes added on top of the residual snapshot.
    pub fn mark_touched<D: Device>(
        &mut self,
        device: &mut D,
        force: &mut ReadbackBuffer<f32>,
        residual: &Buffer<f32>,
    ) -> FatalResult<()> {
        force.download(device)?;
        self.force.clear();
        self.force.extend_from_slice(force.host());
        self.residual.resize(self.force.len(), 0.0);
        residual.read(device, 0, &mut self.residual)?;
        for body in &mut self.bodies {
            if body.touched {
                continue;
            }
            body.touched = body.vertices.iter().any(|&v| {
                let i = 3 * v as usize;
                (0..3).any(|k| self.force[i + k] != self.residual[i + k])
            });
        }
        Ok(())
    }

    /// At the commit: accumulate each body's impact and, on the first step
    /// after one ends, rescale it. `carrier` holds the history point the commit
    /// is about to store (the position whose difference from `end` is the
    /// carried velocity times `dt`), and is rewritten for a body whose impact
    /// ended. `start` and `previous` are the committed pose and history point
    /// the step began from, over `previous_dt`.
    #[allow(clippy::too_many_arguments)]
    pub fn commit<D: Device>(
        &mut self,
        device: &mut D,
        start: &mut ReadbackBuffer<f32>,
        previous: &mut ReadbackBuffer<f32>,
        end: &mut ReadbackBuffer<f32>,
        carrier: &mut ReadbackBuffer<f32>,
        previous_dt: f32,
        dt: f32,
        gravity: [f32; 3],
    ) -> FatalResult<()> {
        start.download(device)?;
        previous.download(device)?;
        end.download(device)?;
        carrier.download(device)?;
        let (x0, p0, x1) = (start.host(), previous.host(), end.host());
        let mut history = carrier.host().to_vec();
        let mut changed = false;
        let (h0, h1) = (f64::from(previous_dt), f64::from(dt));
        for body in &mut self.bodies {
            let before = rigid_velocity(&body.vertices, x0, p0, h0);
            let after = rigid_velocity(&body.vertices, x1, &history, h1);
            if body.touched {
                if !body.in_impact {
                    body.in_impact = true;
                    body.linear = [0.0; 3];
                    body.angular = [0.0; 3];
                }
                for k in 0..3 {
                    body.linear[k] += after.0[k] - before.0[k] - f64::from(gravity[k]) * h1;
                    body.angular[k] += after.1[k] - before.1[k];
                }
            } else if body.in_impact {
                // THE IMPACT IS OVER: keep `(1 + e) / 2` of its change, so
                // remove `(1 - e) / 2` of it from the carried velocity.
                //
                // BOUNDED BY THE SPEED THE BODY LEAVES WITH. For a clean impact
                // the change `J` is twice the speed it separates at along `J`
                // (the elastic step reverses the approach), so the two readings
                // agree and the removal leaves it at `e` times that speed. A
                // body that rested in contact before lifting off has also
                // collected its support against gravity in `J`, which is no
                // impact at all, while it leaves slowly: the removal then
                // follows the speed it leaves with, never more than all of it,
                // so a contact that flickers open under a resting body cannot
                // throw it.
                body.in_impact = false;
                let norm = body.linear.iter().map(|v| v * v).sum::<f64>().sqrt();
                let share = if norm > 0.0 {
                    let leaving = (0..3).map(|k| after.0[k] * body.linear[k]).sum::<f64>() / norm;
                    (2.0 * leaving / norm).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let scale = 0.5 * (1.0 - body.restitution) * share;
                let linear = body.linear.map(|v| -scale * v);
                let angular = body.angular.map(|w| -scale * w);
                let center = centroid(&body.vertices, x1);
                for &v in &body.vertices {
                    let i = 3 * v as usize;
                    let r = [
                        f64::from(x1[i]) - center[0],
                        f64::from(x1[i + 1]) - center[1],
                        f64::from(x1[i + 2]) - center[2],
                    ];
                    let spin = cross(angular, r);
                    for k in 0..3 {
                        // `history = end - dt * velocity`, so a velocity change
                        // `delta` moves it by `-dt * delta`.
                        history[i + k] -= (h1 * (linear[k] + spin[k])) as f32;
                    }
                }
                changed = true;
            }
            body.touched = false;
        }
        if changed {
            carrier.seed(device, &history)?;
        }
        Ok(())
    }
}

fn centroid(vertices: &[u32], x: &[f32]) -> [f64; 3] {
    let mut c = [0.0f64; 3];
    for &v in vertices {
        for k in 0..3 {
            c[k] += f64::from(x[3 * v as usize + k]);
        }
    }
    c.map(|s| s / vertices.len() as f64)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// A body's linear and angular velocity, from its pose `x` and history point
/// `p` (`(x - p) / h` per vertex): the mean velocity, and the angular velocity
/// that best explains the rest, `I^-1 sum r x v` with the body's own (uniform)
/// vertex masses, which cancel.
fn rigid_velocity(vertices: &[u32], x: &[f32], p: &[f32], h: f64) -> ([f64; 3], [f64; 3]) {
    let center = centroid(vertices, x);
    let n = vertices.len() as f64;
    let mut linear = [0.0f64; 3];
    for &v in vertices {
        let i = 3 * v as usize;
        for k in 0..3 {
            linear[k] += (f64::from(x[i + k]) - f64::from(p[i + k])) / h;
        }
    }
    linear = linear.map(|s| s / n);
    let mut momentum = [0.0f64; 3];
    let mut inertia = [[0.0f64; 3]; 3];
    for &v in vertices {
        let i = 3 * v as usize;
        let r = [
            f64::from(x[i]) - center[0],
            f64::from(x[i + 1]) - center[1],
            f64::from(x[i + 2]) - center[2],
        ];
        let u = [
            (f64::from(x[i]) - f64::from(p[i])) / h - linear[0],
            (f64::from(x[i + 1]) - f64::from(p[i + 1])) / h - linear[1],
            (f64::from(x[i + 2]) - f64::from(p[i + 2])) / h - linear[2],
        ];
        let l = cross(r, u);
        let rr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        for a in 0..3 {
            momentum[a] += l[a];
            for b in 0..3 {
                inertia[a][b] += if a == b { rr } else { 0.0 } - r[a] * r[b];
            }
        }
    }
    (linear, solve3(inertia, momentum))
}

/// `A^-1 b` for a symmetric positive definite 3x3, by Cramer's rule.
fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det == 0.0 {
        return [0.0; 3];
    }
    let mut out = [0.0f64; 3];
    for c in 0..3 {
        let mut m = a;
        for r in 0..3 {
            m[r][c] = b[r];
        }
        out[c] = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
            / det;
    }
    out
}
