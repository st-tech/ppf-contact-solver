// File: crates/ppf-cts-py/src/errors.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Centralized PyErr mapping for the typed error enums in
// `ppf-cts-core` and `ppf-cts-formats`. Each typed enum gets one
// implementation of the local `IntoPyErr` trait that picks the right
// `PyErr` class per variant. PyO3 wrappers can then collapse the old
//
//     foo().map_err(|e| PyValueError::new_err(e.to_string()))?
//
// pattern to:
//
//     foo().map_err(IntoPyErr::into_py_err)?
//
// (or `.map_err(into_py_err)?` via the free function alias) so the
// mapping logic lives in one place and can be revised without
// touching every call site.
//
// The orphan rule prevents us from writing `From<Err> for PyErr`
// directly because both types are foreign to this crate, so a local
// trait is the idiomatic workaround.
//
// Conventions:
//   * `PyValueError`  -> input shape / value problems.
//   * `PyKeyError`    -> missing-name / missing-key lookups.
//   * `PyTypeError`   -> dtype / structural mismatch.
//   * `PyRuntimeError` -> system errors (I/O, GPU detect, etc.).
//   * `PyOSError`     -> filesystem-side errors that the Python source
//                        already raised as `OSError`.
//
// The mapping is conservative: when in doubt we pick `PyValueError`
// to preserve the existing Python exception class users see.

use pyo3::exceptions::{PyKeyError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::PyErr;

use ppf_cts_core::datamodel::app::AppPathError;
use ppf_cts_core::datamodel::asset::AssetError;
use ppf_cts_core::datamodel::collider::ColliderError;
use ppf_cts_core::datamodel::decoder::validate::DecoderValidationError;
use ppf_cts_core::datamodel::object::ObjectError;
use ppf_cts_core::datamodel::param_manager::ParamManagerError;
use ppf_cts_core::datamodel::params::ParamError;
use ppf_cts_core::datamodel::scene::SceneError;
use ppf_cts_core::datamodel::validators::ValidatorError;
use ppf_cts_core::extra::ExtraError;
use ppf_cts_core::kernels::fixed_scene_assemble::AssembleError;
use ppf_cts_core::kernels::scene_build::{
    DirectionError, RodTriOffsetViolation, SceneAssemblyError, SceneBuildError, TransformError,
};
use ppf_cts_core::kernels::scene_loops::SceneLoopsError;
use ppf_cts_core::utils::GpuError;
use ppf_cts_formats::FormatError;

/// Local conversion trait. Each typed error implements this to pick
/// the right PyErr class per variant. Use as
/// `.map_err(IntoPyErr::into_py_err)?` or via the free function
/// [`into_py_err`].
pub(crate) trait IntoPyErr {
    fn into_py_err(self) -> PyErr;
}

/// Free-function alias so call sites can write
/// `.map_err(crate::errors::into_py_err)?` without importing the trait.
#[inline]
pub(crate) fn into_py_err<E: IntoPyErr>(e: E) -> PyErr {
    e.into_py_err()
}

// ---------------------------------------------------------------------------
// validators.rs (cold-tail validators).

impl IntoPyErr for ValidatorError {
    fn into_py_err(self) -> PyErr {
        match self {
            // Lookup variants -> PyKeyError so callers can distinguish
            // a missing name from a malformed input.
            ValidatorError::SceneSelectMissing { .. }
            | ValidatorError::ParamKeyMissing { .. }
            | ValidatorError::SessionMissing { .. } => PyKeyError::new_err(self.to_string()),
            // Everything else is a value/shape problem.
            _ => PyValueError::new_err(self.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder validators (file format / scene assembly preconditions).

impl IntoPyErr for DecoderValidationError {
    fn into_py_err(self) -> PyErr {
        // Every decoder validation failure surfaces as ValueError in
        // the existing Python wrappers. Keep that mapping uniform.
        PyValueError::new_err(self.to_string())
    }
}

// ---------------------------------------------------------------------------
// Param holder + param manager.

impl IntoPyErr for ParamError {
    fn into_py_err(self) -> PyErr {
        match self {
            ParamError::NotFound(_) => PyKeyError::new_err(self.to_string()),
            ParamError::UnknownObjectKind(_) => PyValueError::new_err(self.to_string()),
        }
    }
}

impl IntoPyErr for ParamManagerError {
    fn into_py_err(self) -> PyErr {
        match self {
            ParamManagerError::UnknownKey(_) | ParamManagerError::NoKeySelected => {
                PyKeyError::new_err(self.to_string())
            }
            _ => PyValueError::new_err(self.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Scene + object data model.

impl IntoPyErr for SceneError {
    fn into_py_err(self) -> PyErr {
        match self {
            SceneError::ObjectNotFound(_) => PyKeyError::new_err(self.to_string()),
            _ => PyValueError::new_err(self.to_string()),
        }
    }
}

impl IntoPyErr for ObjectError {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

// ---------------------------------------------------------------------------
// Asset manager.

impl IntoPyErr for AssetError {
    fn into_py_err(self) -> PyErr {
        match self {
            AssetError::NotFound(_) => PyKeyError::new_err(self.to_string()),
            _ => PyValueError::new_err(self.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Collider (wall + sphere keyframe stream).

impl IntoPyErr for ColliderError {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

// ---------------------------------------------------------------------------
// App path resolution + cache wipe.

impl IntoPyErr for AppPathError {
    fn into_py_err(self) -> PyErr {
        // Both variants describe a missing on-disk session. Match the
        // Python source: `App.recover_session_path` raised a generic
        // `Exception`; we bias toward `PyKeyError` since "not found"
        // semantics matter to the test rig.
        PyKeyError::new_err(self.to_string())
    }
}

// ---------------------------------------------------------------------------
// Scene-build kernels.

impl IntoPyErr for SceneBuildError {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

impl IntoPyErr for SceneAssemblyError {
    fn into_py_err(self) -> PyErr {
        match self {
            SceneAssemblyError::MissingMapEntry { .. }
            | SceneAssemblyError::MissingDisplacementIndex { .. }
            | SceneAssemblyError::CrossStitchMissingSource { .. }
            | SceneAssemblyError::CrossStitchMissingTarget { .. }
            | SceneAssemblyError::MissingStaticDisplacement { .. } => {
                PyKeyError::new_err(self.to_string())
            }
            _ => PyValueError::new_err(self.to_string()),
        }
    }
}

impl IntoPyErr for DirectionError {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

impl IntoPyErr for TransformError {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

// ---------------------------------------------------------------------------
// Scene-loop kernels (TOML formatters, axis bounds, etc.).

impl IntoPyErr for SceneLoopsError {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

// ---------------------------------------------------------------------------
// FixedScene assembly violation.

impl IntoPyErr for RodTriOffsetViolation {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

impl IntoPyErr for AssembleError {
    fn into_py_err(self) -> PyErr {
        match self {
            AssembleError::RodTriOffset(v) => v.into_py_err(),
        }
    }
}

// ---------------------------------------------------------------------------
// CBOR envelope errors (ppf-cts-formats).

impl IntoPyErr for FormatError {
    fn into_py_err(self) -> PyErr {
        match self {
            FormatError::CborSer(_) | FormatError::CborDe(_) => {
                PyRuntimeError::new_err(self.to_string())
            }
            _ => PyValueError::new_err(self.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// GPU detection.

impl IntoPyErr for GpuError {
    fn into_py_err(self) -> PyErr {
        // The Python source raised `RuntimeError` so the test rig can
        // catch it. Preserve that.
        PyRuntimeError::new_err(self.to_string())
    }
}

// ---------------------------------------------------------------------------
// Extra (sparse-checkout helper, CIPC stitch loader). Filesystem / git
// failures default to OSError (matches the existing Python expectation
// in `extra_py.rs`).

impl IntoPyErr for ExtraError {
    fn into_py_err(self) -> PyErr {
        PyOSError::new_err(self.to_string())
    }
}
