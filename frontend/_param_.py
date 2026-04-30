# File: _param_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from typing import Any


def app_param() -> dict[str, tuple[Any, str, str]]:
    """Return the default application parameters for the simulation.

    Each entry maps a parameter key to a ``(default_value, display_name,
    description)`` tuple.
    """
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
        "inactive-momentum": (
            False,
            "Inactive Momentum",
            "Enable inactive momentum mode for the simulation. Adjusts step size and disables inertia.",
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
            "Factor multiplied to the initial gap to set the CCD threshold.",
        ),
        "ccd-max-iter": (
            4096,
            "Maximum CCD Iterations",
            "Maximum number of iterations for ACCD.",
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
        "friction-mode": (
            "min",
            "Friction Combination Mode",
            "How to combine the friction coefficients of two contacting elements. Choices: min, max, mean. Default min preserves the prior behavior (the more slippery side wins).",
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
        "gravity": ([0.0, -9.8, 0.0], "Gravity Vector", "Gravity acceleration vector (solver Y-up)."),
        "wind": ([0.0, 0.0, 0.0], "Wind Force", "Wind force vector (XYZ)."),
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
    """Return the default material parameters for an object of the given type.

    Args:
        obj_type: One of ``"tri"``, ``"tet"``, or ``"rod"``.

    Returns:
        Mapping from parameter key to ``(default_value, display_name,
        description)`` tuple. The returned keys depend on ``obj_type``.

    Raises:
        ValueError: If ``obj_type`` is not one of the supported values.
    """
    if obj_type == "tri":
        model = "baraff-witkin"
        young_mod = 1000.0
        density = 1.0
        offset = 0.0
        bend = 10.0
    elif obj_type == "tet":
        model = "arap"
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
    params = {
        "model": (
            model,
            "Deformation Model",
            "Elastic constitutive model used to evaluate stretch energy. Valid choices are 'arap', 'stvk', 'baraff-witkin' (shells only), and 'snhk' (stable neo-Hookean). Rods are always solved with ARAP internally.",
        ),
        "density": (
            density,
            "Density",
            "Rest-state mass density. Units depend on element type: kg/m^3 for 'tet' (volumetric), kg/m^2 for 'tri' (areal), and kg/m for 'rod' (linear). Must be positive.",
        ),
        "young-mod": (
            young_mod,
            "Young's Modulus",
            "Stiffness coefficient fed into the constitutive model, pre-normalized by density (units of Pa/rho). This decouples mass from stiffness, so doubling density alone leaves the motion unchanged. Must be positive.",
        ),
        "poiss-rat": (
            0.35,
            "Poisson's Ratio",
            "Poisson's ratio used together with Young's modulus to derive the Lame parameters. Valid range is (0, 0.5); values near 0.5 produce near-incompressible behavior. Used by 'tri' and 'tet' elements only.",
        ),
        "bend": (
            bend,
            "Bending Stiffness",
            "Dimensionless bending stiffness for shell hinges (between adjacent 'tri' faces) and for rod-joint bending at interior rod vertices. Must be non-negative. Not used by 'tet' elements.",
        ),
        "shrink-x": (
            1.0,
            "Shrink X",
            "Anisotropic rest-shape scale along the UV X (warp) direction for 'tri' shells. Values below 1.0 pre-shrink the cloth, above 1.0 pre-stretch it. Must be positive. Cannot be combined with a non-zero strain limit on the same face.",
        ),
        "shrink-y": (
            1.0,
            "Shrink Y",
            "Anisotropic rest-shape scale along the UV Y (weft) direction for 'tri' shells. Values below 1.0 pre-shrink the cloth, above 1.0 pre-stretch it. Must be positive. Cannot be combined with a non-zero strain limit on the same face.",
        ),
        "shrink": (
            1.0,
            "Shrink",
            "Isotropic rest-shape scale for 'tet' solids. Values below 1.0 pre-contract the solid so it pulls toward a smaller target, above 1.0 pre-inflate it. Must be positive. Ignored by 'tri' and 'rod' elements.",
        ),
        "contact-gap": (
            1e-3,
            "Contact Gap",
            "Barrier activation distance for contact detection, in scene units. At each pair the solver uses the mean of the two participants' gaps as the threshold. Must be positive.",
        ),
        "contact-offset": (
            offset,
            "Contact Offset",
            "Extra per-element padding added on top of the contact gap (scene units). At each pair the two participants' offsets are summed, acting like a skin thickness that guarantees minimum clearance. Must be non-negative.",
        ),
        "strain-limit": (
            0.0,
            "Strain Limit",
            "Upper bound on per-element tensile strain (dimensionless, e.g. 0.05 = 5% stretch). 0.0 disables strain limiting. Supported on 'tri' and 'rod' elements only, and incompatible with non-unit shrink-x/shrink-y on the same face.",
        ),
        "friction": (
            0.0,
            "Friction Coefficient",
            "Coulomb friction coefficient at contacts (dimensionless). When two elements touch, the solver uses the minimum of the two participants' coefficients, so the more slippery side wins. Must be non-negative.",
        ),
        "length-factor": (
            1.0,
            "Rod Rest-Length Factor",
            "Multiplier applied to each rod edge's rest length (dimensionless). Values below 1.0 pre-tension the rod, above 1.0 pre-compress it. Must be positive. Used by 'rod' elements only.",
        ),
        "pressure": (
            0.0,
            "Inflation Pressure",
            "Per-face inflation pressure pushing 'tri' shells outward along the face normal. Must be non-negative; 0.0 disables inflation. Ignored by 'tet' and 'rod' elements.",
        ),
    }
    if obj_type in ("tri", "tet"):
        params["plasticity"] = (
            0.0,
            "Plasticity Rate",
            "Rate constant (per second) for stretch plasticity, driving the rest shape toward the current deformation via alpha = 1 - exp(-rate * dt). 0.0 disables plasticity; higher values creep faster. Must be non-negative.",
        )
        params["plasticity-threshold"] = (
            0.0,
            "Plasticity Threshold",
            "Dead zone (dimensionless) around unit principal stretch S=1. Plasticity only activates on a singular value k when |S_k - 1| exceeds this threshold. 0.0 means any deviation triggers creep.",
        )
    # Note: "tet" keeps the bend-plasticity fields at 0.0 defaults even
    # though tetrahedra don't use them. This is so a solid's surface
    # triangles (tet-typed obj, F-data) can join `concat_tri_param`
    # alongside shell tris without a key-set mismatch. The scene
    # builder strips these keys from `concat_tet_param` (where the
    # solid's tetrahedra land) so the unused zeros never reach the
    # solver's tet data.
    if obj_type in ("tri", "rod", "tet"):
        params["bend-plasticity"] = (
            0.0,
            "Bend Plasticity Rate",
            "Rate constant (per second) for bending plasticity that drifts the rest angle of shell hinges and interior rod joints toward the current angle. 0.0 disables. Applies to 'tri' and 'rod'; kept on 'tet' only as a placeholder and ignored by the solver.",
        )
        params["bend-plasticity-threshold"] = (
            0.0,
            "Bend Plasticity Threshold",
            "Angular dead zone (radians) around the rest angle. Bend plasticity only activates when |theta - theta_rest| exceeds this threshold. 0.0 means any angular deviation triggers creep.",
        )
        params["bend-rest-from-geometry"] = (
            0.0,
            "Bend Rest From Initial Geometry",
            "If non-zero, initialize each hinge or rod-joint rest angle from the initial pose instead of the default (flat for shells, straight for rods). Treated as a boolean flag.",
        )
    return params


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
