# File: _param_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from typing import Any


def app_param() -> dict[str, tuple[Any, str, str]]:
    """Application parameters for the simulation."""
    return {
        "disable-contact": (
            False,
            "Disable Contact",
            "When enabled, the simulation will not perform any contact detection.",
        ),
        "keep-states": (
            10,
            "Keep States",
            "Number of simulation states to keep in the output directory.",
        ),
        "keep-verts": (
            0,
            "Keep Vertices",
            "Number of vertex data files to keep in the output directory. 0 means no limit (unlimited). Minimum is 1 to ensure visualization.",
        ),
        "dt": (
            1e-3,
            "Step Size",
            "Step size for the simulation. Small step size increases accuracy, large step size increases speed but may cause solver divergence.",
        ),
        "fitting": (
            False,
            "Fitting Mode",
            "Enable fitting mode for the simulation. Adjusts step size and disables inertia.",
        ),
        "playback": (
            1.0,
            "Playback Speed",
            "Playback speed. 1.0 is normal, >1.0 is faster, <1.0 is slower.",
        ),
        "min-newton-steps": (
            0,
            "Lower Bound of Newton Steps",
            "Minimal Newton's steps to advance a step. Recommended 32 if static friction is present.",
        ),
        "target-toi": (
            0.25,
            "Target Accumulated Time of Impact (TOI)",
            "Accumulated TOI threshold for Newton's loop termination.",
        ),
        "air-friction": (
            0.2,
            "Air Tangental Friction",
            "Ratio of tangential friction to normal friction for air drag/lift.",
        ),
        "line-search-max-t": (
            1.25,
            "Extended Line Search Maximum Time",
            "Factor to extend TOI for CCD to avoid possible solver divergence.",
        ),
        "constraint-ghat": (
            1e-3,
            "Gap Distance for Boundary Conditions",
            "Gap distance to activate boundary condition barriers.",
        ),
        "constraint-tol": (
            0.01,
            "Moving Constraint Minimum Gap Tolerance",
            "This factor is multiplied to the constraint gap to determine the CCD tolerance for moving constraints.",
        ),
        "fps": (
            60.0,
            "Frame Per Second for Video Frames",
            "Frame rate for output video.",
        ),
        "cg-max-iter": (
            10000,
            "Maximum Number of PCG Iterations",
            "PCG solver is regarded as diverged if this is exceeded.",
        ),
        "cg-tol": (
            1e-3,
            "Relative Tolerance for PCG",
            "Relative tolerance for PCG solver termination.",
        ),
        "ccd-eps": (
            1e-7,
            "ACCD Epsilon",
            "Small thickness tolerance for ACCD gap distance checks.",
        ),
        "ccd-reduction": (
            0.01,
            "CCD Reduction Factor",
            "This factor is multiplied to the initial gap to set the CCD threshold.",
        ),
        "ccd-max-iter": (
            4096,
            "Maximum CCD Iterations",
            "The maximum number of iterations for ACCD.",
        ),
        "max-dx": (
            1.0,
            "Maximum Search Direction",
            "Maximum allowable search direction magnitude during optimization.",
        ),
        "eiganalysis-eps": (
            1e-2,
            "Epsilon for Eigenvalue Analysis",
            "Epsilon for stable eigenvalue analysis when singular values are close.",
        ),
        "friction-eps": (
            1e-5,
            "Epsilon for Friction",
            "Small value to avoid division by zero in quadratic friction model.",
        ),
        "csrmat-max-nnz": (
            10000000,
            "Maximal Matrix Entries for Contact Matrix Entries on the GPU",
            "Pre-allocated contact matrix entries for GPU. Too large may cause OOM, too small may cause failure.",
        ),
        "bvh-alloc-factor": (
            2,
            "Extra Memory Allocation Factor for BVH on the GPU",
            "Factor to pre-allocate BVH memory on GPU.",
        ),
        "frames": (
            300,
            "Maximal Frame Count to Simulate",
            "Maximal number of frames to simulate.",
        ),
        "auto-save": (
            0,
            "Auto Save Interval",
            "Interval (in frames) for auto-saving simulation state. 0 disables auto-save.",
        ),
        "barrier": (
            "cubic",
            "Barrier Model for Contact",
            "Contact barrier potential model. Choices: cubic, quad, log.",
        ),
        "stitch-stiffness": (
            1.0,
            "Stiffness Factor for Stitches",
            "Stiffness factor for the stitches.",
        ),
        "air-density": (
            1e-3,
            "Air Density",
            "Air density for drag and lift force computation.",
        ),
        "isotropic-air-friction": (
            0.0,
            "Air Dragging Coefficient",
            "Per-vertex air dragging coefficient.",
        ),
        "gravity": (-9.8, "Gravity Coefficient", "Gravity coefficient."),
        "wind": (0.0, "Wind Coefficient", "Wind strength."),
        "wind-dim": (0, "Wind Direction", "Wind direction."),
        "include-face-mass": (
            False,
            "Flag to Include Shell Mass for Volume Solids",
            "Include shell mass for surface elements of volume solids.",
        ),
        "fix-xz": (
            0.0,
            "Whether to fix xz positions",
            "Fix xz positions for falling objects if y > this value. 0.0 disables. Use an extremely small value if nearly a zero is needed.",
        ),
        "fake-crash-frame": (
            -1,
            "Fake Crash Frame",
            "Frame number to intentionally crash simulation for testing. -1 disables.",
        ),
    }


def object_param(obj_type: str) -> dict[str, tuple[Any, str, str]]:
    """Material parameters for the object."""
    if obj_type == "tri":
        model = "baraff-witkin"
        young_mod = 100.0
        density = 1.0
        offset = 0.0
        bend = 2.0
    elif obj_type == "tet":
        model = "snhk"
        young_mod = 500.0
        density = 1000.0
        offset = 0.0
        bend = 1.0
    elif obj_type == "rod":
        model = "arap"
        young_mod = 1e4
        density = 1.0
        offset = 1e-3
        bend = 0.0
    else:
        raise ValueError(f"Unknown object type: {obj_type}")
    return {
        "model": (
            model,
            "Deformation Model",
            "Deformation model for the object. Choices are: arap, stvk, baraff-witkin, snhk.",
        ),
        "density": (
            density,
            "Density",
            "Material density per volume, area or length, depending on the material type.",
        ),
        "young-mod": (
            young_mod,
            "Young's Modulus",
            "Young's modulus for the material divided by the volumetric density.",
        ),
        "poiss-rat": (0.35, "Poisson's Ratio", "Poisson's ratio for the material."),
        "bend": (bend, "Bending Stiffness", "Bending stiffness factor."),
        "shrink": (1.0, "Shrink Factor", "Shrink factor for thin shells."),
        "contact-gap": (1e-3, "Contact Gap", "Gap distance for contact detection."),
        "contact-offset": (offset, "Contact Offset", "Offset of contact surface."),
        "strain-limit": (
            0.0,
            "Strain Limit",
            "Maximum strain limit. 0.0 disables strain limiting. Valid only for triangle elements.",
        ),
        "friction": (
            0.0,
            "Friction Coefficient",
            "Friction coefficient for the object.",
        ),
        "length-factor": (
            1.0,
            "Length Factor",
            "Length factor for rod objects.",
        ),
    }


class ParamHolder:
    def __init__(self, param: dict[str, tuple[Any, str, str]]):
        self._params = param
        self._default_param = param.copy()

    def clear_all(self) -> "ParamHolder":
        self._params = self._default_param.copy()
        return self

    def set(self, key: str, value: Any) -> "ParamHolder":
        if key in self._params:
            self._params[key] = (value, *self._params[key][1:])
        else:
            raise KeyError(f"Parameter '{key}' not found.")
        return self

    def get(self, key: str) -> Any:
        if key in self._params:
            return self._params[key][0]
        else:
            raise KeyError(f"Parameter '{key}' not found.")

    def get_desc(self, key: str) -> tuple[str, str]:
        if key in self._params:
            return (self._params[key][1], self._params[key][2])
        else:
            raise KeyError(f"Parameter '{key}' not found.")

    def key_list(self) -> list[str]:
        return list(self._params.keys())

    def items(self) -> list[tuple[str, Any]]:
        return [(key, value[0]) for key, value in self._params.items()]

    def copy(self) -> "ParamHolder":
        return ParamHolder(self._params.copy())
