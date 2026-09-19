// File: crates/ppf-cts-solver/src/driver/test_scene.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! Synthetic `repr(C)` scenes for the backend's unit tests.
//!
//! The pattern is the one `refusal.rs` established: `DataSet` is a `repr(C)`
//! aggregate of `CVec` handles with no `Drop` beyond the handles' own, so an
//! all-zero bit pattern is a valid inhabitant and the fields a test cares about
//! are filled in afterwards. What that buys is a test over the REAL struct, at
//! the real offsets, rather than over a stand-in that could agree with the
//! backend and disagree with the solver.
//!
//! It matters most for the fields a kernel reaches by offset. A pin index, an
//! element's `fixed` flag and a position all live at offsets the C++ side reads
//! by the same layout, so a test that constructs them here exercises exactly
//! the memory the shared kernel bodies are handed.

#![cfg(test)]

use crate::cvec::CVec;
use crate::data::{
    Constraint, DataSet, EdgeProp, FaceProp, FixPair, HingeProp, PullPair, TetProp, Vec2u, Vec3f,
    Vec3u, Vec4u, VertexProp,
};
use super::scene::SceneView;

/// A scene held at a stable address, with a view onto it.
///
/// The `Box` is what makes the view legitimate: the `DataSet` must not move
/// while a raw pointer to it is live, and a local would move on return.
pub struct TestScene {
    pub data: Box<DataSet>,
}

impl TestScene {
    /// A scene with `vertex_count` vertices, all at the origin.
    pub fn new(vertex_count: usize) -> Self {
        // Safety: `DataSet` is a repr(C) aggregate of `CVec` handles, plain
        // integers and nested aggregates of the same, with no references and no
        // niche-optimized fields, so an all-zero bit pattern is valid. A zeroed
        // `CVec` is a null pointer with size and capacity zero, which is what
        // `CVec::new()` produces and what `Drop` no-ops on.
        let mut data: Box<DataSet> = Box::new(unsafe { std::mem::zeroed() });
        let positions = vec![position(0.0, 0.0, 0.0); vertex_count];
        data.vertex.curr = CVec::from(&positions[..]);
        data.vertex.prev = CVec::from(&positions[..]);
        data.prop.vertex = CVec::from(&vec![VertexProp::default(); vertex_count][..]);
        Self { data }
    }

    pub fn view(&self) -> SceneView {
        // Safety: the `DataSet` lives in the box for as long as `self` does,
        // and every borrow taken through the view is dropped inside a test.
        unsafe { SceneView::new(&*self.data as *const DataSet) }
    }

    /// Place vertex `index` at a world position.
    pub fn place(&mut self, index: usize, x: f32, y: f32, z: f32) {
        let position = position(x, y, z);
        self.data.vertex.curr.as_mut_slice()[index] = position;
        self.data.vertex.prev.as_mut_slice()[index] = position;
    }

    pub fn curr(&self, index: usize) -> Vec3f {
        self.data.vertex.curr.as_slice()[index]
    }

    pub fn prev(&self, index: usize) -> Vec3f {
        self.data.vertex.prev.as_slice()[index]
    }

    pub fn vertex_props(&self) -> &[VertexProp] {
        self.data.prop.vertex.as_slice()
    }

    pub fn vertex_props_mut(&mut self) -> &mut [VertexProp] {
        self.data.prop.vertex.as_mut_slice()
    }

    /// Give the scene `faces`, with one `FaceProp` each.
    pub fn with_faces(mut self, faces: &[Vec3u]) -> Self {
        self.data.mesh.mesh.face = CVec::from(faces);
        self.data.prop.face = CVec::from(&vec![FaceProp::default(); faces.len()][..]);
        self.data.shell_face_count = faces.len() as u32;
        self
    }

    pub fn with_edges(mut self, edges: &[Vec2u]) -> Self {
        self.data.mesh.mesh.edge = CVec::from(edges);
        self.data.prop.edge = CVec::from(&vec![EdgeProp::default(); edges.len()][..]);
        self
    }

    pub fn with_tets(mut self, tets: &[Vec4u]) -> Self {
        self.data.mesh.mesh.tet = CVec::from(tets);
        self.data.prop.tet = CVec::from(&vec![TetProp::default(); tets.len()][..]);
        self
    }

    pub fn with_hinges(mut self, hinges: &[Vec4u]) -> Self {
        self.data.mesh.mesh.hinge = CVec::from(hinges);
        self.data.prop.hinge = CVec::from(&vec![HingeProp::default(); hinges.len()][..]);
        self
    }

    pub fn face_props(&self) -> &[FaceProp] {
        self.data.prop.face.as_slice()
    }

    pub fn edge_props(&self) -> &[EdgeProp] {
        self.data.prop.edge.as_slice()
    }

    pub fn tet_props(&self) -> &[TetProp] {
        self.data.prop.tet.as_slice()
    }

    pub fn hinge_props(&self) -> &[HingeProp] {
        self.data.prop.hinge.as_slice()
    }
}

/// A position.
pub fn position(x: f32, y: f32, z: f32) -> Vec3f {
    Vec3f::new(x, y, z)
}

/// The world-space displacement `a - b`, differenced in `f64`.
///
/// `f64` because this is a TEST reading a result, not solver arithmetic: the
/// quantity being measured is the round-off of the step under test, and
/// differencing at that same width would fold the measurement into the answer.
/// Widening an `f32` is exact, so the difference taken here is exact too.
pub fn position_delta(a: Vec3f, b: Vec3f) -> [f64; 3] {
    [
        f64::from(a[0]) - f64::from(b[0]),
        f64::from(a[1]) - f64::from(b[1]),
        f64::from(a[2]) - f64::from(b[2]),
    ]
}

/// A `Constraint` carrying the given fix and pull pins and nothing else.
pub fn constraint(fix: &[FixPair], pull: &[PullPair]) -> Constraint {
    // Safety: as `TestScene::new`. `Constraint`'s one non-`CVec` member,
    // `CollisionMesh`, is itself an aggregate of `CVec` handles.
    let mut out: Constraint = unsafe { std::mem::zeroed() };
    out.fix = CVec::from(fix);
    out.pull = CVec::from(pull);
    out
}

/// A static fix pin on one vertex.
pub fn fix_pin(index: u32) -> FixPair {
    FixPair {
        position: position(0.0, 0.0, 0.0),
        step_delta: Vec3f::zeros(),
        ghat: 0.0,
        index,
        kinematic: false,
        allow_intersection: false,
    }
}

/// A pull pin on one vertex.
pub fn pull_pin(index: u32) -> PullPair {
    PullPair {
        position: position(0.0, 0.0, 0.0),
        weight: 1.0,
        index,
        allow_intersection: false,
    }
}
