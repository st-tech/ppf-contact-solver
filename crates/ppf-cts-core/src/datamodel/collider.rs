// File: crates/ppf-cts-core/src/datamodel/collider.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Invisible-collider data: `Wall` (planar) and `Sphere`.
// Direct port of frontend/_scene_.py:186-771 (WallParam, Wall,
// SphereParam, Sphere). Each collider holds a list of keyframe
// `entries` describing how the collider position (and, for Sphere,
// radius) changes over time. `_invisible_collider_.py` reads these
// to detect violation candidates per simulation frame.
//
// `interp` selects the time-blending mode between consecutive
// entries. Default is `Linear` to match the Python source.

use super::easing::TransitionKind;
use super::quat::Vec3;

#[derive(Debug, Clone, PartialEq)]
pub struct ColliderParam {
    pub contact_gap: f64,
    pub friction: f64,
    /// `-1.0` means "active forever". Mirrors the Python sentinel.
    pub active_duration: f64,
    pub thickness: f64,
}

impl Default for ColliderParam {
    fn default() -> Self {
        Self {
            contact_gap: 1e-3,
            friction: 0.0,
            active_duration: -1.0,
            thickness: 1.0,
        }
    }
}

/// Time-stamped wall position keyframe. The Python tuple is
/// `(pos, time)` (frontend/_scene_.py:241).
#[derive(Debug, Clone)]
pub struct WallEntry {
    pub position: Vec3,
    pub time: f64,
}

#[derive(Debug, Clone)]
pub struct Wall {
    pub normal: Vec3,
    pub entries: Vec<WallEntry>,
    pub interp: TransitionKind,
    pub param: ColliderParam,
}

#[derive(Debug, thiserror::Error)]
pub enum ColliderError {
    #[error("wall already exists")]
    AlreadyAdded,
    #[error("time must be greater than the last time")]
    TimeNotMonotonic,
    #[error("no entries; call add() first")]
    Empty,
}

impl Wall {
    /// Empty wall. `add()` seeds it with the initial keyframe.
    pub fn new() -> Self {
        Self {
            normal: [0.0, 1.0, 0.0],
            entries: vec![],
            interp: TransitionKind::Linear,
            param: ColliderParam::default(),
        }
    }

    /// Initial keyframe. Mirrors `Wall.add` (frontend/_scene_.py:257-277).
    pub fn add(&mut self, position: Vec3, normal: Vec3) -> Result<(), ColliderError> {
        if !self.entries.is_empty() {
            return Err(ColliderError::AlreadyAdded);
        }
        self.normal = normal;
        self.entries.push(WallEntry {
            position,
            time: 0.0,
        });
        Ok(())
    }

    /// Move-by keyframe. Position deltas are summed onto the previous
    /// keyframe's position (matching the Python relative semantics).
    pub fn move_by(&mut self, delta: Vec3, time: f64) -> Result<(), ColliderError> {
        self.check_time(time)?;
        let prev = self.entries.last().expect("non-empty after check_time").position;
        self.entries.push(WallEntry {
            position: [prev[0] + delta[0], prev[1] + delta[1], prev[2] + delta[2]],
            time,
        });
        Ok(())
    }

    /// Move-to keyframe. Absolute position.
    pub fn move_to(&mut self, position: Vec3, time: f64) -> Result<(), ColliderError> {
        self.check_time(time)?;
        self.entries.push(WallEntry { position, time });
        Ok(())
    }

    fn check_time(&self, time: f64) -> Result<(), ColliderError> {
        match self.entries.last() {
            None => Err(ColliderError::Empty),
            Some(e) if time <= e.time => Err(ColliderError::TimeNotMonotonic),
            _ => Ok(()),
        }
    }
}

impl Default for Wall {
    fn default() -> Self {
        Self::new()
    }
}

/// Sphere keyframe; also carries the radius so animation can
/// shrink/grow the collider over time. Mirrors the Python
/// `(pos, radius, time)` tuple.
#[derive(Debug, Clone)]
pub struct SphereEntry {
    pub position: Vec3,
    pub radius: f64,
    pub time: f64,
}

#[derive(Debug, Clone)]
pub struct Sphere {
    pub entries: Vec<SphereEntry>,
    pub interp: TransitionKind,
    pub param: ColliderParam,
    /// True == "vertices stay outside" (default). False == "vertices
    /// stay inside the sphere" (e.g. for a containment hemisphere).
    pub inverted: bool,
    /// Hemisphere mode: only the bottom half is active. Above the
    /// center, the constraint becomes a cylinder.
    pub hemisphere: bool,
}

impl Sphere {
    pub fn new() -> Self {
        Self {
            entries: vec![],
            interp: TransitionKind::Linear,
            param: ColliderParam::default(),
            inverted: false,
            hemisphere: false,
        }
    }

    pub fn add(&mut self, position: Vec3, radius: f64) -> Result<(), ColliderError> {
        if !self.entries.is_empty() {
            return Err(ColliderError::AlreadyAdded);
        }
        self.entries.push(SphereEntry {
            position,
            radius,
            time: 0.0,
        });
        Ok(())
    }

    pub fn move_by(
        &mut self,
        delta: Vec3,
        radius: f64,
        time: f64,
    ) -> Result<(), ColliderError> {
        self.check_time(time)?;
        let prev = self.entries.last().expect("non-empty after check_time").position;
        self.entries.push(SphereEntry {
            position: [prev[0] + delta[0], prev[1] + delta[1], prev[2] + delta[2]],
            radius,
            time,
        });
        Ok(())
    }

    pub fn move_to(
        &mut self,
        position: Vec3,
        radius: f64,
        time: f64,
    ) -> Result<(), ColliderError> {
        self.check_time(time)?;
        self.entries.push(SphereEntry {
            position,
            radius,
            time,
        });
        Ok(())
    }

    fn check_time(&self, time: f64) -> Result<(), ColliderError> {
        match self.entries.last() {
            None => Err(ColliderError::Empty),
            Some(e) if time <= e.time => Err(ColliderError::TimeNotMonotonic),
            _ => Ok(()),
        }
    }
}

impl Default for Sphere {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wall_add_seeds_initial_entry() {
        let mut w = Wall::new();
        w.add([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]).unwrap();
        assert_eq!(w.entries.len(), 1);
        assert_eq!(w.normal, [0.0, 1.0, 0.0]);
        assert_eq!(w.entries[0].time, 0.0);
    }

    #[test]
    fn wall_double_add_rejected() {
        let mut w = Wall::new();
        w.add([0.0; 3], [0.0, 1.0, 0.0]).unwrap();
        let err = w.add([1.0; 3], [0.0, 1.0, 0.0]).unwrap_err();
        assert!(matches!(err, ColliderError::AlreadyAdded));
    }

    #[test]
    fn wall_move_by_accumulates() {
        let mut w = Wall::new();
        w.add([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]).unwrap();
        w.move_by([1.0, 0.0, 0.0], 1.0).unwrap();
        w.move_by([2.0, 0.0, 0.0], 2.0).unwrap();
        assert_eq!(w.entries[2].position, [3.0, 0.0, 0.0]);
    }

    #[test]
    fn wall_move_to_overrides() {
        let mut w = Wall::new();
        w.add([0.0; 3], [0.0, 1.0, 0.0]).unwrap();
        w.move_to([10.0, 5.0, 0.0], 1.0).unwrap();
        assert_eq!(w.entries[1].position, [10.0, 5.0, 0.0]);
    }

    #[test]
    fn wall_non_monotonic_time_rejected() {
        let mut w = Wall::new();
        w.add([0.0; 3], [0.0, 1.0, 0.0]).unwrap();
        w.move_to([1.0, 0.0, 0.0], 2.0).unwrap();
        let err = w.move_to([2.0, 0.0, 0.0], 1.0).unwrap_err();
        assert!(matches!(err, ColliderError::TimeNotMonotonic));
        // Equal time also rejected.
        let err = w.move_to([3.0, 0.0, 0.0], 2.0).unwrap_err();
        assert!(matches!(err, ColliderError::TimeNotMonotonic));
    }

    #[test]
    fn sphere_add_with_radius() {
        let mut s = Sphere::new();
        s.add([0.0, 1.0, 0.0], 0.5).unwrap();
        assert_eq!(s.entries[0].radius, 0.5);
        assert!(!s.inverted);
        assert!(!s.hemisphere);
    }

    #[test]
    fn sphere_move_by_accumulates_position_radius_separate() {
        let mut s = Sphere::new();
        s.add([0.0, 0.0, 0.0], 0.5).unwrap();
        s.move_by([1.0, 0.0, 0.0], 0.7, 1.0).unwrap();
        s.move_by([0.0, 1.0, 0.0], 0.7, 2.0).unwrap();
        assert_eq!(s.entries[2].position, [1.0, 1.0, 0.0]);
        assert_eq!(s.entries[2].radius, 0.7);
    }

    #[test]
    fn collider_param_defaults_match_python() {
        let p = ColliderParam::default();
        assert_eq!(p.contact_gap, 1e-3);
        assert_eq!(p.friction, 0.0);
        assert_eq!(p.active_duration, -1.0);
        assert_eq!(p.thickness, 1.0);
    }
}
