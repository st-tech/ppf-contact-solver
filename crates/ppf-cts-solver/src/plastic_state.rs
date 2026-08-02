// File: plastic_state.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! A per-frame copy of the rest shape, so `dataset.bin.gz` can be written once.
//!
//! Nothing here is a new quantity. A `DataSet` always carries the rest shape,
//! because the elastic kernels read it every Newton iteration: `inv_rest2x2`
//! is the inverse rest matrix of each shell face, `inv_rest3x3` that of each
//! tet, and `HingeProp::rest_angle` / `VertexProp::rest_bend_angle` are the
//! rest bend angles. Plasticity adds no field. It CREEPS those four in place
//! on the device, so what would otherwise be a build-time constant becomes the
//! one thing in a `DataSet` whose value depends on how far the run has got.
//! One kernel per array, in `cpp/plasticity/plasticity.cu`:
//!
//! | kernel | runs when | creeps |
//! | ------ | --------- | ------ |
//! | `update_face_plasticity` | `shell_face_count > 0`, `FaceParam::plasticity` | `inv_rest2x2` |
//! | `update_tet_plasticity` | any tet, `TetParam::plasticity` | `inv_rest3x3` |
//! | `update_hinge_plasticity` | any hinge, `HingeParam::plasticity` | `HingeProp::rest_angle` |
//! | `update_rod_bend_plasticity` | `rod_count > 0`, `EdgeParam::plasticity` | `VertexProp::rest_bend_angle` |
//!
//! Both halves of each condition matter, and mirror `main.cu`'s dispatch: a
//! tet asset gives its surface faces the object's `plasticity` parameter, so
//! the parameter test alone reports face plasticity on a scene that has no
//! shell face for the kernel to run over.
//!
//! Those four live inside `DataSet`, so keeping them current at every
//! checkpoint by re-serializing the dataset would carry the whole struct along
//! with them: mesh topology, and every property, parameter and constraint
//! array. On a 450k-tet scene that is roughly half a gigabyte per auto-save,
//! for four arrays that a scene with no plastic material never changes at all.
//!
//! So the rest shape is ALSO written on its own, per frame, next to the
//! `state_<N>.bin.gz` it belongs to, and `dataset.bin.gz` is written once,
//! like `meshset.bin.gz`. The dataset keeps its `inv_rest*` and rest-angle
//! fields, holding the build-time shape; the per-frame file overwrites them at
//! load with the shape as of that frame. It stores only the arrays whose
//! kernel is enabled, so a hinge-only scene does not drag the tet rest
//! matrices along, and a scene with no plasticity writes no file at all.
//!
//! Pairing the file with a frame, rather than overwriting a single one, is
//! what lets a resume from an older retained checkpoint restore the rest shape
//! of THAT frame. A single file would hold whichever shape was current at the
//! last save, so resuming from an earlier `state_<N>.bin.gz` would pair those
//! positions with creep from a later time.
//!
//! It is also why a missing per-frame file is ambiguous rather than simply
//! fatal. The dataset always deserializes to SOME rest shape, so there is no
//! null to detect; the question is only which frame's shape it is, and that
//! depends on which layout wrote the directory. Under this one the dataset
//! holds the build-time shape, which belongs to no checkpoint, so a missing
//! file has to fail. Under a layout that rewrites the dataset at every save it
//! holds the last-saved shape, which for the newest checkpoint is the right
//! one and must be left alone. Absence cannot tell those apart, and neither
//! can looking for other per-frame files: they are pruned with their
//! checkpoints, so "none present" is equally consistent with "never written"
//! and "all deleted".
//!
//! The witness is therefore a directory-level marker, `checkpoint_layout.txt`,
//! written once beside the dataset and never pruned. It makes the resume rule
//! exact, with no scan and no dependence on when it runs:
//!
//! | marker | scene creeps | `plastic_<N>` | resume |
//! | ------ | ------------ | ------------- | ------ |
//! | any | any | present | overwrite the dataset's rest shape with it |
//! | present | yes | absent | error: no frame's rest shape is on disk |
//! | present | no | absent | proceed, the build-time shape never changed |
//! | absent | any | absent | proceed, the dataset holds the last-saved shape |
//!
//! "Scene creeps" is `PlasticKinds::of(dataset).any()`, read from the loaded
//! dataset's parameter arrays. Those are build-time constants, so the value at
//! resume is the same one that decided whether to write the file at save.

use super::data::{
    DataSet, EdgeParam, FaceParam, HingeParam, HingeProp, Mat2x2f, Mat3x3f, TetParam, VertexProp,
};
use serde::{Deserialize, Serialize};

/// Which of the four plasticity kernels this scene actually runs: the element
/// count `main.cu` dispatches on, and the same `plasticity > 0` test the
/// kernel itself applies to its per-element parameter. The parameter arrays
/// are deduplicated per unique material (`builder::dedup_param`), so this scan
/// is over a handful of entries, not per element.
///
/// A kernel counts as active when ANY material enables it. The kernels
/// additionally skip individual elements (a fixed hinge, a rod vertex that is
/// not an interior 2-edge vertex), which only narrows what changes; treating
/// the whole array as mutable is the conservative direction, and is what makes
/// this safe to decide once, host-side, without consulting the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlasticKinds {
    pub face: bool,
    pub tet: bool,
    pub hinge: bool,
    pub rod_bend: bool,
}

/// How many elements each kernel dispatches over, in `main.cu`'s terms.
#[derive(Debug, Clone, Copy, Default)]
pub struct ElementCounts {
    pub shell_face: u32,
    pub tet: u32,
    pub hinge: u32,
    pub rod: u32,
}

impl PlasticKinds {
    pub fn of(dataset: &DataSet) -> Self {
        let p = &dataset.param_arrays;
        Self::from_params(
            ElementCounts {
                shell_face: dataset.shell_face_count,
                tet: dataset.mesh.mesh.tet.size,
                hinge: dataset.mesh.mesh.hinge.size,
                rod: dataset.rod_count,
            },
            p.face.as_slice(),
            p.tet.as_slice(),
            p.hinge.as_slice(),
            p.edge.as_slice(),
        )
    }

    pub fn from_params(
        counts: ElementCounts,
        face: &[FaceParam],
        tet: &[TetParam],
        hinge: &[HingeParam],
        edge: &[EdgeParam],
    ) -> Self {
        let enabled = |count: u32, any: bool| count > 0 && any;
        Self {
            face: enabled(counts.shell_face, face.iter().any(|x| x.plasticity > 0.0)),
            tet: enabled(counts.tet, tet.iter().any(|x| x.plasticity > 0.0)),
            hinge: enabled(counts.hinge, hinge.iter().any(|x| x.plasticity > 0.0)),
            rod_bend: enabled(counts.rod, edge.iter().any(|x| x.plasticity > 0.0)),
        }
    }

    /// True when some kernel can creep the rest shape, i.e. when a per-frame
    /// rest-shape file has to be written at all.
    pub fn any(&self) -> bool {
        self.face || self.tet || self.hinge || self.rod_bend
    }

    /// True when `inv_rest2x2` / `inv_rest3x3` can change, so the device copy
    /// has to be pulled back before serializing.
    pub fn needs_inv_rest(&self) -> bool {
        self.face || self.tet
    }

    /// True when `HingeProp::rest_angle` / `VertexProp::rest_bend_angle` can
    /// change, so the device property arrays have to be pulled back.
    pub fn needs_rest_angles(&self) -> bool {
        self.hinge || self.rod_bend
    }
}

/// The evolving rest shape at one checkpoint. Each array is populated only
/// when its kernel is enabled and is otherwise empty, which is how "write only
/// what changed" is expressed on the wire: an empty array means "this scene
/// never touches it, keep the build-time value from `dataset.bin.gz`".
#[derive(Serialize, Deserialize, Default)]
pub struct PlasticState {
    /// Per shell face, from `update_face_plasticity`.
    pub inv_rest2x2: Vec<Mat2x2f>,
    /// Per tet, from `update_tet_plasticity`.
    pub inv_rest3x3: Vec<Mat3x3f>,
    /// Per hinge, from `update_hinge_plasticity`.
    pub hinge_rest_angle: Vec<f32>,
    /// Per vertex, from `update_rod_bend_plasticity`.
    pub vertex_rest_bend_angle: Vec<f32>,
}

impl PlasticState {
    /// Copy the enabled kernels' rest state out of a dataset whose host arrays
    /// have just been refreshed from the device.
    pub fn extract(dataset: &DataSet, kinds: PlasticKinds) -> Self {
        Self::from_parts(
            kinds,
            dataset.inv_rest2x2.as_slice(),
            dataset.inv_rest3x3.as_slice(),
            dataset.prop.hinge.as_slice(),
            dataset.prop.vertex.as_slice(),
        )
    }

    pub fn from_parts(
        kinds: PlasticKinds,
        inv_rest2x2: &[Mat2x2f],
        inv_rest3x3: &[Mat3x3f],
        hinge: &[HingeProp],
        vertex: &[VertexProp],
    ) -> Self {
        Self {
            inv_rest2x2: if kinds.face {
                inv_rest2x2.to_vec()
            } else {
                Vec::new()
            },
            inv_rest3x3: if kinds.tet {
                inv_rest3x3.to_vec()
            } else {
                Vec::new()
            },
            hinge_rest_angle: if kinds.hinge {
                hinge.iter().map(|h| h.rest_angle).collect()
            } else {
                Vec::new()
            },
            vertex_rest_bend_angle: if kinds.rod_bend {
                vertex.iter().map(|v| v.rest_bend_angle).collect()
            } else {
                Vec::new()
            },
        }
    }

    /// Overlay this rest state onto a freshly loaded dataset, before it is
    /// uploaded to the device.
    pub fn apply(&self, dataset: &mut DataSet) {
        self.apply_to_parts(
            dataset.inv_rest2x2.as_mut_slice(),
            dataset.inv_rest3x3.as_mut_slice(),
            dataset.prop.hinge.as_mut_slice(),
            dataset.prop.vertex.as_mut_slice(),
        );
    }

    /// A length mismatch means the checkpoint and the scene disagree about how
    /// many elements exist, so the overlay would write a rest shape onto the
    /// wrong elements and produce a plausible but wrong simulation. It panics
    /// instead, naming the array and both counts.
    pub fn apply_to_parts(
        &self,
        inv_rest2x2: &mut [Mat2x2f],
        inv_rest3x3: &mut [Mat3x3f],
        hinge: &mut [HingeProp],
        vertex: &mut [VertexProp],
    ) {
        overlay("inv_rest2x2", &self.inv_rest2x2, inv_rest2x2, |d, s| {
            *d = *s
        });
        overlay("inv_rest3x3", &self.inv_rest3x3, inv_rest3x3, |d, s| {
            *d = *s
        });
        overlay("hinge_rest_angle", &self.hinge_rest_angle, hinge, |d, s| {
            d.rest_angle = *s
        });
        overlay(
            "vertex_rest_bend_angle",
            &self.vertex_rest_bend_angle,
            vertex,
            |d, s| d.rest_bend_angle = *s,
        );
    }
}

/// Write a saved array over the matching dataset array. An empty saved array
/// is the "kernel disabled, nothing was ever mutated" case, and leaves the
/// destination at its build-time value.
fn overlay<S, D>(name: &str, saved: &[S], dst: &mut [D], set: impl Fn(&mut D, &S)) {
    if saved.is_empty() {
        return;
    }
    assert_eq!(
        saved.len(),
        dst.len(),
        "saved rest shape {name} has {} entries but the dataset has {}; \
         the checkpoint does not belong to this scene",
        saved.len(),
        dst.len()
    );
    for (d, s) in dst.iter_mut().zip(saved) {
        set(d, s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kernel has elements to run over, so only the parameters decide.
    fn kinds(face: f32, tet: f32, hinge: f32, edge: f32) -> PlasticKinds {
        kinds_with(
            ElementCounts {
                shell_face: 1,
                tet: 1,
                hinge: 1,
                rod: 1,
            },
            face,
            tet,
            hinge,
            edge,
        )
    }

    fn kinds_with(
        counts: ElementCounts,
        face: f32,
        tet: f32,
        hinge: f32,
        edge: f32,
    ) -> PlasticKinds {
        PlasticKinds::from_params(
            counts,
            &[FaceParam {
                plasticity: face,
                ..Default::default()
            }],
            &[TetParam {
                plasticity: tet,
                ..Default::default()
            }],
            &[HingeParam {
                plasticity: hinge,
                ..Default::default()
            }],
            &[EdgeParam {
                plasticity: edge,
                ..Default::default()
            }],
        )
    }

    fn hinges(angles: &[f32]) -> Vec<HingeProp> {
        angles
            .iter()
            .map(|&rest_angle| HingeProp {
                rest_angle,
                ..Default::default()
            })
            .collect()
    }

    #[test]
    fn kinds_track_each_kernel_independently() {
        assert_eq!(kinds(0.0, 0.0, 0.0, 0.0), PlasticKinds::default());
        assert!(!kinds(0.0, 0.0, 0.0, 0.0).any());

        // Hinge creep touches no inverse rest matrix, so a hinge-only scene
        // must not pay for pulling or storing them.
        let only_hinge = kinds(0.0, 0.0, 1.0, 0.0);
        assert_eq!(
            only_hinge,
            PlasticKinds {
                hinge: true,
                ..Default::default()
            }
        );
        assert!(only_hinge.any());
        assert!(only_hinge.needs_rest_angles());
        assert!(!only_hinge.needs_inv_rest());

        let only_tet = kinds(0.0, 2.0, 0.0, 0.0);
        assert!(only_tet.needs_inv_rest());
        assert!(!only_tet.needs_rest_angles());

        // Rod bend is gated on the EDGE parameter, not the hinge one.
        let only_rod = kinds(0.0, 0.0, 0.0, 3.0);
        assert_eq!(
            only_rod,
            PlasticKinds {
                rod_bend: true,
                ..Default::default()
            }
        );
        assert!(only_rod.needs_rest_angles());
        assert!(!only_rod.needs_inv_rest());
    }

    #[test]
    fn a_kernel_with_no_elements_is_inactive_however_the_material_is_set() {
        // The shape a tet asset produces: its surface faces carry the object's
        // `plasticity`, but there is no shell face for the face kernel to run
        // over, so `inv_rest2x2` cannot change and must not be stored.
        let solid_only = kinds_with(
            ElementCounts {
                shell_face: 0,
                tet: 4,
                hinge: 0,
                rod: 0,
            },
            0.5,
            0.5,
            0.5,
            0.5,
        );
        assert_eq!(
            solid_only,
            PlasticKinds {
                tet: true,
                ..Default::default()
            }
        );
        assert!(solid_only.needs_inv_rest());
        assert!(!solid_only.needs_rest_angles());

        // A scene with the materials but no elements at all writes nothing.
        assert!(!kinds_with(ElementCounts::default(), 1.0, 1.0, 1.0, 1.0).any());
    }

    #[test]
    fn extract_carries_only_enabled_kernels() {
        let hinge = hinges(&[0.25, -0.5]);
        let vertex = vec![
            VertexProp {
                rest_bend_angle: 9.0,
                ..Default::default()
            };
            3
        ];
        let saved = PlasticState::from_parts(
            kinds(0.0, 0.0, 1.0, 0.0),
            &[Mat2x2f::identity()],
            &[Mat3x3f::identity()],
            &hinge,
            &vertex,
        );

        assert_eq!(saved.hinge_rest_angle, vec![0.25, -0.5]);
        // Rod bend is off, so its array stays empty even though the source
        // holds a non-default value.
        assert!(saved.vertex_rest_bend_angle.is_empty());
        assert!(saved.inv_rest2x2.is_empty());
        assert!(saved.inv_rest3x3.is_empty());
    }

    #[test]
    fn apply_restores_enabled_and_leaves_disabled_alone() {
        let saved = PlasticState::from_parts(
            kinds(0.0, 1.0, 1.0, 0.0),
            &[Mat2x2f::identity() * 7.0],
            &[Mat3x3f::identity() * 3.0],
            &hinges(&[0.25, -0.5]),
            &[VertexProp::default(); 2],
        );

        // A fresh build starts from its own build-time rest shape.
        let mut inv2 = [Mat2x2f::identity() * 11.0];
        let mut inv3 = [Mat3x3f::identity()];
        let mut hinge = hinges(&[0.0, 0.0]);
        let mut vertex = [VertexProp::default(); 2];
        saved.apply_to_parts(&mut inv2, &mut inv3, &mut hinge, &mut vertex);

        assert_eq!(hinge[0].rest_angle, 0.25);
        assert_eq!(hinge[1].rest_angle, -0.5);
        assert_eq!(inv3[0], Mat3x3f::identity() * 3.0);
        // Face plasticity is off, so inv_rest2x2 was never saved and keeps
        // whatever the build produced.
        assert_eq!(inv2[0], Mat2x2f::identity() * 11.0);
    }

    #[test]
    fn round_trips_through_the_wire_format() {
        let saved = PlasticState::from_parts(
            kinds(0.0, 0.0, 1.0, 0.0),
            &[],
            &[],
            &hinges(&[0.25, -0.5, 1.75]),
            &[],
        );
        let bytes = bincode::serialize(&saved).unwrap();
        let back: PlasticState = bincode::deserialize(&bytes).unwrap();

        let mut hinge = hinges(&[0.0, 0.0, 0.0]);
        back.apply_to_parts(&mut [], &mut [], &mut hinge, &mut []);
        assert_eq!(
            hinge.iter().map(|h| h.rest_angle).collect::<Vec<_>>(),
            vec![0.25, -0.5, 1.75]
        );
    }

    #[test]
    #[should_panic(expected = "does not belong to this scene")]
    fn apply_rejects_a_checkpoint_from_a_different_scene() {
        let saved = PlasticState::from_parts(
            kinds(0.0, 0.0, 1.0, 0.0),
            &[],
            &[],
            &hinges(&[0.25, -0.5, 1.75]),
            &[],
        );
        let mut hinge = hinges(&[0.0, 0.0]);
        saved.apply_to_parts(&mut [], &mut [], &mut hinge, &mut []);
    }
}
