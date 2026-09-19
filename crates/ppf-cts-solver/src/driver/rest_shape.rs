// File: crates/ppf-cts-solver/src/driver/rest_shape.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The streamed time-varying rest shape: `update_rest_shape`.
//!
//! A scene may drive each element's rest pose per frame (the
//! `rest_vert_schedule` path). The host interpolates the inverse rest matrices
//! for this step's target time and hands them over; the elastic kernels re-read
//! `inv_rest2x2` / `inv_rest3x3` every Newton iteration, so replacing them here
//! is what makes the rest pose follow the schedule.
//!
//! This module owns `rest_excluded` outright: it assigns the whole per-element
//! mask on every call, so there is no stale state and no ordering dependence on
//! `update_constraint`, which only touches `fixed`. The three flags `fixed`,
//! `rest_excluded` and `collider` are kept apart on purpose and none of them
//! aliases another.
//!
//! Plasticity does not coexist with a streamed rest shape: the frontend refuses
//! to ship both, leaving `plasticity == 0` for these elements, so there is no
//! in-place creep to preserve across this write.
//!
//! NOTHING HERE IS A DISPATCH, and that is a property of the operation rather
//! than an omission. Updating a streamed rest shape is a host walk over the
//! exclusion mask plus four bulk uploads of the inverse rest matrices, so every
//! element of it is either driver bookkeeping or a TRANSFER, which the seam
//! spells as `Device::write` and not as a kernel. This backend addresses the
//! scene's arrays directly, so each of those transfers is a `copy_from_slice`
//! here; they become `Device::write` calls in the change that turns the
//! destinations into arena handles, which is the debt `ppf_cts_compute::HostRef`
//! records. There is no shared body this module reaches around.
//!
//! WHY THE LENGTHS MUST MATCH EXACTLY here, where the CUDA backend's own
//! upload helper accepts a shorter source. `mem::copy_to_device`
//! (`ppf-cts-compute/cuda/mem.hpp`) copies `src.size` elements and then ASSIGNS
//! `dst.size = src.size`, so a short upload silently shortens the destination
//! container. Doing that here would mean writing the `DataSet`
//! record itself, which this backend never does: it only ever writes through
//! the buffer pointers the record holds (see `super::scene`). A source that
//! does not cover the destination is a rest shape that does not describe this
//! mesh, so it stops the run by name instead.

use crate::data::RestShapeUpdate;

use super::scene::{slice, Fatal, FatalResult, SceneView};

/// Replace the inverse rest matrices and the per-element exclusion mask.
///
/// # Safety
/// `view` must address a live `DataSet` and `update` a live `RestShapeUpdate`,
/// with no other reference to either alive for the duration.
pub unsafe fn apply(view: &SceneView, update: &RestShapeUpdate) -> FatalResult<()> {
    // An empty array is skipped rather than rejected: a tet-only solid has no
    // faces and a shell-only cloth has no tets, so one of the two is routinely
    // absent and that is not a disagreement about anything.
    let source2x2 = slice(&update.inv_rest2x2);
    if !source2x2.is_empty() {
        let destination = view.inv_rest2x2_mut();
        check_length("inv_rest2x2", source2x2.len(), destination.len())?;
        destination.copy_from_slice(source2x2);
    }

    let source3x3 = slice(&update.inv_rest3x3);
    if !source3x3.is_empty() {
        let destination = view.inv_rest3x3_mut();
        check_length("inv_rest3x3", source3x3.len(), destination.len())?;
        destination.copy_from_slice(source3x3);
    }

    let exclude_face = slice(&update.exclude_face);
    if !exclude_face.is_empty() {
        let props = view.face_props_mut();
        for (prop, flag) in props.iter_mut().zip(exclude_face.iter()) {
            prop.rest_excluded = *flag != 0;
        }
    }

    let exclude_tet = slice(&update.exclude_tet);
    if !exclude_tet.is_empty() {
        let props = view.tet_props_mut();
        for (prop, flag) in props.iter_mut().zip(exclude_tet.iter()) {
            prop.rest_excluded = *flag != 0;
        }
    }

    Ok(())
}

fn check_length(name: &str, source: usize, destination: usize) -> FatalResult<()> {
    if source != destination {
        return Err(Fatal::invariant(format!(
            "the streamed rest shape carries {source} {name} entries against {destination} in the \
             scene, so it does not describe this mesh"
        )));
    }
    Ok(())
}
