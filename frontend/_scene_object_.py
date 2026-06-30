# File: _scene_object_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""The :class:`Object` class.

An ``Object`` is a placed instance of a registered asset. It carries a
4x4 transform, per-vertex colors, pins, stitches, and material
parameters. Instances are created via ``scene.add(...)`` in
``_scene_.py`` and are typically configured through chainable methods
defined on this class.

The class is re-exported from :mod:`frontend._scene_` for backward
compatibility.
"""

from typing import Optional, Union

import numpy as np

from . import _rust  # type: ignore[attr-defined]

from ._asset_ import AssetManager
from ._param_ import ParamHolder, object_param
from ._scene_ import EnumColor
from ._scene_pin_ import PinHolder
from ._scene_transform_ import (
    TransformAnimation,
    _apply_transform_to_verts,
    _axis_angle_to_quat,
    _quat_multiply,
    _quat_to_mat3,
)

EPS = 1e-3


class Object:
    """The object class.

    An ``Object`` is a placed instance of a registered asset. It carries a
    4x4 transform, per-vertex colors, pins, stitches, and material
    parameters. Instances are created via ``scene.add("<mesh_name>")`` and
    are typically configured through chainable methods.

    Example:
        Chain placement, material, and pinning onto a cloth sheet::

            sheet = scene.add("sheet").at(0, 0.6, 0).jitter()
            sheet.param.set("strain-limit", 0.05)
            sheet.pin(sheet.grab([-1, 0, -1]) + sheet.grab([1, 0, -1]))
    """

    def __init__(self, asset: AssetManager, name: str):
        self._asset = asset
        self._name = name
        self._static = False
        self._is_pdrd = False
        self._param = ParamHolder(object_param(self.obj_type))
        self.clear()

    @property
    def name(self) -> str:
        """Get name of the object.

        Example:
            Read the asset reference name back from an object::

                obj = scene.add("sheet")
                assert obj.name == "sheet"
        """
        return self._name

    @property
    def static(self) -> bool:
        """Get whether the object is static.

        Example:
            Pinning every vertex turns an object into a static collider::

                obj = scene.add("sphere")
                obj.pin()
                obj.update_static()
                assert obj.static
        """
        return self._static

    @property
    def pdrd(self) -> bool:
        """Get whether the object is a PDRD body.

        A PDRD body is a Tri-asset object whose vertices move
        rigidly through a single best-fit rigid transform shared by
        the whole body. Set via :meth:`as_pdrd`.

        Example:
            Flag a cube as PDRD and inspect the result::

                obj = scene.add("cube").as_pdrd()
                assert obj.pdrd
        """
        return self._is_pdrd

    @property
    def param(self) -> ParamHolder:
        """Get the material parameters of the object.

        Returns:
            ParamHolder: The material parameters of the object.

        Example:
            Configure the Young's modulus through the returned holder::

                scene.add("sheet").param.set("young-mod", 1e5)
        """
        return self._param

    @property
    def obj_type(self) -> str:
        """Get the type of the object.

        Returns:
            str: The type of the object, either "rod", "tri", "tet", or
            "points".

        Example:
            Branch on object topology when iterating the scene::

                for obj in scene.object_dict.values():
                    if obj.obj_type == "tet":
                        obj.param.set("young-mod", 1e6)
        """
        return self._asset.fetch.get_type(self._name)

    @property
    def object_color(self) -> Optional[list[float]]:
        """Get the object color.

        Example:
            Read the RGB color previously assigned via :meth:`color`::

                obj = scene.add("sheet").color(0.9, 0.3, 0.3)
                print(obj.object_color)
        """
        color = self._color
        if color is None:
            return None
        elif isinstance(color, list):
            return color
        else:
            return color.tolist()

    @property
    def position(self) -> list[float]:
        """Get the object translation from the transform matrix.

        Example:
            Read back the translation set by :meth:`at`::

                obj = scene.add("sheet").at(0, 0.6, 0)
                assert obj.position == [0.0, 0.6, 0.0]
        """
        return self._transform[:3, 3].tolist()

    @property
    def object_velocity(self) -> list[float] | np.ndarray:
        """Get the object velocity.

        Example:
            Inspect the initial velocity set via :meth:`velocity`::

                obj = scene.add("sheet").velocity(0, 0, -1)
                print(obj.object_velocity)
        """
        return self._velocity

    @property
    def uv_coords(self) -> Optional[list[np.ndarray]]:
        """Get the UV coordinates.

        Example:
            Check whether the asset carries UV data for texturing::

                obj = scene.add("sheet")
                if obj.uv_coords is not None:
                    print(obj.uv_coords[0].shape)
        """
        return self._uv

    @property
    def dynamic_color(self) -> EnumColor:
        """Get the dynamic color type.

        Example:
            Inspect the dynamic color mode previously selected::

                obj = scene.add("sheet")
                print(obj.dynamic_color)
        """
        return self._dyn_color

    @property
    def dynamic_intensity(self) -> float:
        """Get the dynamic color intensity.

        Example:
            Read the scalar intensity used by dynamic coloring::

                obj = scene.add("sheet")
                print(obj.dynamic_intensity)
        """
        return self._dyn_intensity

    @property
    def pin_list(self) -> list[PinHolder]:
        """Get the list of pin holders.

        Example:
            Iterate pin holders attached to an object::

                obj = scene.add("sheet")
                obj.pin([0, 1, 2])
                for holder in obj.pin_list:
                    print(len(holder.index))
        """
        return self._pin

    def clear(self):
        """Clear the object data.

        Example:
            Reset an object's transform and pin state::

                obj = scene.select("sheet")
                obj.clear()
                obj.at(0, 0.6, 0)
        """
        self._transform = np.eye(4)  # Single 4x4 matrix for all transforms
        self._color: Union[np.ndarray, list[float], None] = None
        self._dyn_color = EnumColor.NONE
        self._dyn_intensity = 1.0
        self._static_color = [0.75, 0.75, 0.75]
        self._default_color = [1.0, 0.85, 0.0]
        self._velocity = [0.0, 0.0, 0.0]
        self._angular_velocity = [0.0, 0.0, 0.0]
        # PDRD hinge spec: None = free body, or ("hinge", pca_axis_index)
        # where pca_axis_index in {0, 1, 2} selects a principal axis of the
        # rest shape as the world rotation axle. See :meth:`hinge`.
        self._joint: Optional[tuple[str, int]] = None
        self._velocity_schedule = []
        # Principal-axis angular velocity overwrite keyframes:
        # list of (time, pca_index, speed_rad_per_s). The spin axis is the
        # `pca_index`-th principal axis (0 = largest extent ... 2 = thinnest)
        # of the body's geometry, resolved dynamically in the solver from the
        # live (rotated / deformed) pose each time a keyframe fires. See
        # :meth:`angular_velocity_pca`.
        self._angular_velocity_schedule_pca = []
        # Fixed world-axis angular overwrite keyframes:
        # list of (time, [wx, wy, wz]) where the vector is the full angular
        # velocity (rad/s) in solver space. Used by World X/Y/Z and Custom
        # axis modes; the axis does NOT track the body (it is world-fixed).
        # See :meth:`angular_velocity_world`.
        self._angular_velocity_schedule_world = []
        self._collision_windows = []
        self._pin: list[PinHolder] = []
        # World-space (N, 3) reference vertices for the bending rest angle, or
        # None. When set, the solver computes this object's hinge rest angles
        # from these positions instead of the object's own initial pose. Set
        # by the decoder from the per-object ``bend_rest_vert`` scene field
        # (already transformed to world space, matching ``vertex(True)``).
        self._bend_rest_vert: Optional[np.ndarray] = None
        self._normalize = False
        self._stitch = None
        self._uv = None
        self._transform_animation: Optional[TransformAnimation] = None
        # Flagged by the decoder when the object is a static-moving mesh
        # driven by a pin-shell (either fcurve or UI-op). Propagates into
        # PinData.hide_in_preview so the JupyterLab viewer doesn't draw
        # the all-vertex pin as user-facing pin markers.
        self._is_static_moving = False

    def as_pdrd(self) -> "Object":
        """Switch the object to Painless Differentiable Rotation Dynamics.

        The body must reference a Tri (surface) asset; PDRD is realized
        as a single exact per-body rigid transform over the surface mesh,
        so no tetrahedralization is needed or used. Swaps the parameter set
        to the PDRD defaults (volumetric ``density``, contact/friction;
        elastic terms are kept as zero placeholders so the existing
        per-face param expansion continues to work without
        per-pipeline branching).

        PDRD and static colliders are mutually exclusive.

        Returns:
            Object: ``self``, to allow chaining with placement and
            velocity builders.

        Example:
            Drop a cube and let contact resolve it as a rigid body::

                scene.add("cube").at(0, 1.0, 0).as_pdrd()
        """
        if self.obj_type != "tri":
            raise ValueError(
                f"as_pdrd() requires a Tri (surface) asset; got {self.obj_type!r}"
            )
        if self._static:
            raise ValueError("as_pdrd() cannot be applied to a static object")
        self._is_pdrd = True
        self._param = ParamHolder(object_param("pdrd"))
        return self

    def report(self):
        """Report the object data.

        Example:
            Print a summary of an object after configuring it::

                obj = scene.add("sheet").at(0, 0.6, 0)
                obj.pin(obj.grab([-1, 0, -1]))
                obj.report()
        """
        print("transform:")
        print(self._transform)
        print("color:", self._color)
        print("velocity:", self._velocity)
        print("normalize:", self._normalize)
        self.update_static()
        if self.static:
            print("pin: static")
        else:
            print("pin:", sum([len(p.index) for p in self._pin]))

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the bounding box of the object.

        Returns:
            tuple[np.ndarray, np.ndarray]: The dimensions and center of the bounding box.

        Example:
            Read a sheet's size and center in world space::

                sheet = scene.add("sheet")
                size, center = sheet.bbox()
                print(size, center)
        """
        vert = self.get("V")
        if vert is None:
            raise ValueError("vertex does not exist")
        # apply_transform handles both the normalize and non-normalize
        # branches; passing identity to the bbox kernel just iterates
        # the already-transformed buffer.
        transformed = self.apply_transform(vert, False)
        v_in = np.ascontiguousarray(transformed, dtype=np.float64)
        M = np.eye(4, dtype=np.float64)
        size, center = _rust.scene_object_bbox(v_in, M)
        return (size, center)

    def normalize(self) -> "Object":
        """Normalize the object so that it fits within a unit cube.

        Returns:
            Object: The normalized object.

        Example:
            Normalize a tetrahedral asset before scaling it into place::

                arm = scene.add("armadillo").normalize().scale(0.75)
                arm.at(0, 1, 0)
        """
        _rust.scene_validate_object_normalize(bool(self._normalize))
        self._bbox, self._center = self.bbox()
        self._normalize = True
        return self

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get an associated value of the object with respect to the key.

        Args:
            key (str): The key of the value.
        Returns:
            Optional[np.ndarray]: The value associated with the key.

        Example:
            Fetch the rest-pose vertices and face list of an asset::

                sheet = scene.add("sheet")
                V = sheet.get("V")
                F = sheet.get("F")
        """
        if key == "color":
            if self._color is not None:
                return np.array(self._color)
            else:
                if self.static:
                    return np.array(self._static_color)
                else:
                    return np.array(self._default_color)
        elif key == "Ind":
            if self._stitch is not None:
                return self._stitch[0]
            else:
                return None
        elif key == "W":
            if self._stitch is not None:
                return self._stitch[1]
            else:
                return None
        else:
            result = self._asset.fetch.get(self._name)
            if key in result:
                return result[key]
            else:
                return None

    def vertex(self, translate: bool) -> np.ndarray:
        """Get the transformed vertices of the object.

        Args:
            translate (bool): Whether to translate the vertices.

        Returns:
            np.ndarray: The transformed vertices.

        Example:
            Read world-space vertex positions after ``.at`` has been set::

                sheet = scene.add("sheet").at(0, 0.6, 0)
                world_vert = sheet.vertex(True)
        """
        vert = self.get("V")
        if vert is None:
            raise ValueError("vertex does not exist")
        else:
            return self.apply_transform(vert, translate)

    def grab(self, direction: list[float], eps: float = 1e-3) -> list[int]:
        """Select vertices that are furthest along a specified direction.

        Args:
            direction (list[float]): The direction vector.
            eps (float, optional): Tolerance (in dot-product units) from the maximum. Defaults to 1e-3.

        Returns:
            list[int]: The indices of the selected vertices.

        Example:
            Pin the two top corners of a sheet to hang it::

                sheet = scene.add("sheet")
                sheet.pin(sheet.grab([-1, 1, 0]) + sheet.grab([1, 1, 0]))
        """
        vert = self.vertex(False)
        v = np.ascontiguousarray(vert, dtype=np.float64)
        d = [float(direction[0]), float(direction[1]), float(direction[2])]
        return [int(i) for i in _rust.scene_grab_indices(v, d, float(eps))]

    def mat4x4(self, matrix: np.ndarray) -> "Object":
        """Set the full 4x4 transformation matrix directly.

        Replaces the current transform entirely.  The matrix is applied as::

            world_pos = matrix[:3,:3] @ local_pos + matrix[:3,3]

        Args:
            matrix (np.ndarray): A 4x4 transformation matrix.

        Returns:
            Object: The object with the updated transform.

        Example:
            Apply a pre-computed transform imported from Blender::

                import numpy as np
                M = np.eye(4)
                M[:3, 3] = [0, 0.6, 0]
                scene.add("sheet").mat4x4(M)
        """
        self._transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        return self

    @property
    def transform_matrix(self) -> np.ndarray:
        """Get the current 4x4 transformation matrix.

        Example:
            Inspect the composed translation, rotation, and scale::

                obj = scene.add("sheet").at(0, 0.6, 0)
                print(obj.transform_matrix)
        """
        return self._transform

    def at(self, x: float, y: float, z: float) -> "Object":
        """Set the translation component of the transform.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
            z (float): The z-coordinate.

        Returns:
            Object: The object with the updated position.

        Example:
            Drop a sheet 0.6 units above the origin::

                scene.add("sheet").at(0, 0.6, 0)
        """
        self._transform[:3, 3] = [x, y, z]
        return self

    def jitter(self, r: float = 1e-2) -> "Object":
        """Add random jitter to the translation.

        Args:
            r (float, optional): The jitter magnitude.

        Returns:
            Object: The object with the jittered position.

        Example:
            Break symmetry for a falling armadillo on a trampoline::

                scene.add("armadillo").at(0, 1, 0).jitter().velocity(0, -5, 0)
        """
        self._transform[0, 3] += r * np.random.random()
        self._transform[1, 3] += r * np.random.random()
        self._transform[2, 3] += r * np.random.random()
        return self

    def scale(self, _scale: float) -> "Object":
        """Apply uniform scale to the transform.

        Args:
            _scale (float): The scale factor.

        Returns:
            Object: The object with the updated scale.

        Example:
            Shrink an armadillo to 0.75 of its original size::

                scene.add("armadillo").scale(0.75).at(0, 1, 0)
        """
        M = np.ascontiguousarray(self._transform, dtype=np.float64)
        self._transform = _rust.scene_mat4_apply_scale(M, float(_scale))
        return self

    def rotate(self, angle: float, axis: str) -> "Object":
        """Apply rotation around a specified axis to the transform.

        Args:
            angle (float): The rotation angle in degrees.
            axis (str): The rotation axis ('x', 'y', or 'z').

        Returns:
            Object: The object with the updated rotation.

        Example:
            Stand a sheet upright by rotating 90 degrees around x::

                scene.add("sheet").rotate(90, "x").at(0, 0.5, 0)
        """
        a = _rust.scene_validate_object_rotate_axis(axis)
        M = np.ascontiguousarray(self._transform, dtype=np.float64)
        self._transform = _rust.scene_mat4_apply_rotate(M, float(angle), a)
        return self

    def move(
        self, delta, t_start: float = 0.0, t_end: float = 1.0
    ) -> "Object":
        """Animate the static object by a translational delta over time.

        Args:
            delta: [dx, dy, dz] translation delta in world space.
            t_start (float): Start time in seconds.
            t_end (float): End time in seconds.

        Returns:
            Object: The object with the animation added.

        Example:
            Slide a static (fully pinned) collider along +x between t=0 and t=2::

                scene.add("sphere").at(-1, 0, 0).pin()
                scene.select("sphere").move([2, 0, 0], t_start=0.0, t_end=2.0)
        """
        delta = np.asarray(delta, dtype=np.float64)
        _rust.scene_validate_time_window(float(t_start), float(t_end))
        self._ensure_transform_animation()
        anim = self._transform_animation
        if len(anim.times) > 1:
            _rust.scene_validate_collider_time(float(anim.times[-1]), float(t_start))
        start_trans = anim.translations[-1].copy()
        start_quat = anim.quaternions[-1].copy()
        start_scale = anim.scales[-1].copy()
        self._append_transform_keyframe(
            anim, t_start, t_end,
            start_trans + delta, start_quat.copy(), start_scale.copy(),
        )
        return self

    def animate_rotate(
        self,
        axis,
        angle: float,
        center=None,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> "Object":
        """Animate the static object by rotating around an axis over time.

        Args:
            axis: [ax, ay, az] rotation axis (will be normalized).
            angle (float): Rotation angle in degrees.
            center: Center of rotation [cx, cy, cz]. Defaults to object centroid.
            t_start (float): Start time in seconds.
            t_end (float): End time in seconds.

        Returns:
            Object: The object with the animation added.

        Example:
            Rotate a pinned static collider 180 degrees around y over 2 seconds::

                roller = scene.add("cylinder").at(0, 0.5, 0).pin()
                scene.select("cylinder").animate_rotate(
                    [0, 1, 0], 180.0, t_start=0.0, t_end=2.0,
                )
        """
        axis = np.asarray(axis, dtype=np.float64)
        _rust.scene_validate_time_window(float(t_start), float(t_end))
        self._ensure_transform_animation()
        anim = self._transform_animation
        if len(anim.times) > 1:
            _rust.scene_validate_collider_time(float(anim.times[-1]), float(t_start))
        start_trans = anim.translations[-1].copy()
        start_quat = anim.quaternions[-1].copy()
        start_scale = anim.scales[-1].copy()
        if center is None:
            center = np.mean(
                _apply_transform_to_verts(anim.local_vert, start_trans, start_quat, start_scale),
                axis=0,
            )
        else:
            center = np.asarray(center, dtype=np.float64)
        rot_quat = _axis_angle_to_quat(axis, angle)
        R = _quat_to_mat3(rot_quat)
        new_trans = center + R @ (start_trans - center)
        new_quat = _quat_multiply(rot_quat, start_quat)
        self._append_transform_keyframe(
            anim, t_start, t_end,
            new_trans, new_quat, start_scale.copy(),
        )
        return self

    def _append_transform_keyframe(
        self, anim, t_start, t_end, new_trans, new_quat, new_scale
    ):
        """Append a transform keyframe, gap-filling the timeline if needed.

        When the last existing keyframe sits before ``t_start``, a holding
        keyframe carrying the current transform is inserted at ``t_start``
        so the new motion starts cleanly. The final keyframe at ``t_end``
        carries the supplied translation, rotation, and scale. Each appended
        array is a fresh copy so later callers cannot mutate shared state.
        """
        start_trans = anim.translations[-1].copy()
        start_quat = anim.quaternions[-1].copy()
        start_scale = anim.scales[-1].copy()
        if anim.times[-1] < t_start:
            anim.times.append(t_start)
            anim.translations.append(start_trans.copy())
            anim.quaternions.append(start_quat.copy())
            anim.scales.append(start_scale.copy())
        anim.times.append(t_end)
        anim.translations.append(new_trans)
        anim.quaternions.append(new_quat)
        anim.scales.append(new_scale)

    def _ensure_transform_animation(self):
        """Initialize TransformAnimation from current transform if not already set."""
        if self._transform_animation is not None:
            return
        vert = self.get("V")
        if vert is None:
            raise ValueError("Object has no vertices; cannot create transform animation")
        # When the object is normalized, the built static geometry is
        # M @ ((v - center) / max(bbox)) (see apply_transform). The
        # animation only folds the decomposed T*R*S of self._transform, so
        # bake the same normalize pre-step into local_vert here; otherwise
        # evaluate(0) would omit the centering/scaling and the animated
        # slice would teleport away from the static base at the first frame.
        # This is a no-op for the common non-normalized path.
        if self._normalize:
            center = np.asarray(self._center, dtype=np.float64)
            bbox_max = np.max(np.asarray(self._bbox, dtype=np.float64))
            local = (np.asarray(vert, dtype=np.float64) - center) / bbox_max
        else:
            local = vert
        translation, quat, scale = _rust.scene_decompose_trs(
            np.ascontiguousarray(self._transform, dtype=np.float64)
        )
        self._transform_animation = TransformAnimation(
            local_vert=local.copy(),
            times=[0.0],
            translations=[translation],
            quaternions=[quat],
            scales=[scale],
        )

    def max(self, dim: str) -> float:
        """Get the maximum coordinate value along a specified dimension.

        Args:
            dim (str): The dimension to get the maximum value along, either "x", "y", or "z".

        Returns:
            float: The maximum coordinate value.

        Example:
            Check that a sheet sits above y=0::

                sheet = scene.add("sheet").at(0, 0.6, 0)
                assert sheet.min("y") >= 0
                print(sheet.max("y"))
        """
        ax = {"x": 0, "y": 1, "z": 2}[dim]
        v = np.ascontiguousarray(self.vertex(True), dtype=np.float64)
        _, hi = _rust.scene_axis_min_max(v, ax)
        return float(hi)

    def min(self, dim: str) -> float:
        """Get the minimum coordinate value along a specified dimension.

        Args:
            dim (str): The dimension to get the minimum value along, either "x", "y", or "z".

        Returns:
            float: The minimum coordinate value.

        Example:
            Lift an object so its lowest point sits at y=0::

                obj = scene.add("armadillo")
                obj.at(0, -obj.min("y"), 0)
        """
        ax = {"x": 0, "y": 1, "z": 2}[dim]
        v = np.ascontiguousarray(self.vertex(True), dtype=np.float64)
        lo, _ = _rust.scene_axis_min_max(v, ax)
        return float(lo)

    def apply_transform(self, x: np.ndarray, translate: bool) -> np.ndarray:
        """Apply the object's transformation to a set of vertices.

        Args:
            x (np.ndarray): The vertices to transform (N, 3).
            translate (bool): Whether to include the translation component.

        Returns:
            np.ndarray: The transformed vertices (N, 3).

        Example:
            Manually map a batch of points through the object's current transform::

                obj = scene.add("sheet").at(0, 0.6, 0)
                world_pts = obj.apply_transform(obj.get("V"), True)
        """
        if len(x.shape) == 1:
            raise Exception("vertex should be 2D array")
        v_in = np.ascontiguousarray(x, dtype=np.float64)
        M = np.ascontiguousarray(self._transform, dtype=np.float64)
        if self._normalize:
            bbox = np.ascontiguousarray(self._bbox, dtype=np.float64)
            center = np.ascontiguousarray(self._center, dtype=np.float64)
            return _rust.scene_apply_transform_batch(
                v_in, M, bool(translate), bbox, center,
            )
        return _rust.scene_apply_transform_batch(v_in, M, bool(translate))

    def static_color(self, red: float, green: float, blue: float) -> "Object":
        """Set the static color of the object.

        Args:
            red (float): The red component.
            green (float): The green component.
            blue (float): The blue component.

        Returns:
            Object: The object with the updated static color.

        Example:
            Tint a fully pinned collider a light gray::

                scene.add("sphere").pin()
                scene.select("sphere").static_color(0.75, 0.75, 0.75)
        """
        self._static_color = [red, green, blue]
        return self

    def default_color(self, red: float, green: float, blue: float) -> "Object":
        """Set the default color of the object.

        Args:
            red (float): The red component.
            green (float): The green component.
            blue (float): The blue component.

        Returns:
            Object: The object with the updated default color.

        Example:
            Override the auto-assigned hue for a dynamic object::

                scene.add("sheet").default_color(1.0, 0.85, 0.0)
        """
        self._default_color = [red, green, blue]
        return self

    def color(self, red: float, green: float, blue: float) -> "Object":
        """Set the color of the object.

        Args:
            red (float): The red component.
            green (float): The green component.
            blue (float): The blue component.

        Returns:
            Object: The object with the updated color.

        Example:
            Color a falling armadillo light gray::

                scene.add("armadillo").color(0.75, 0.75, 0.75)
        """
        self._color = [red, green, blue]
        return self

    def vert_color(self, color: np.ndarray) -> "Object":
        """Set the vertex colors of the object.

        Args:
            color (np.ndarray): The vertex colors.

        Returns:
            Object: The object with the updated vertex colors.

        Example:
            Paint each vertex of a sheet with its own RGB::

                import numpy as np
                sheet = scene.add("sheet")
                n = len(sheet.get("V"))
                sheet.vert_color(np.random.rand(n, 3))
        """
        self._color = color
        return self

    def direction_color(self, x: float, y: float, z: float) -> "Object":
        """Set the color along the direction of the object.

        Args:
            x (float): The x-component of the direction.
            y (float): The y-component of the direction.
            z (float): The z-component of the direction.

        Returns:
            Object: The object with the updated color.

        Example:
            Shade a cylinder along its long axis::

                scene.add("cylinder").direction_color(1, 0, 0)
        """
        vertex = self.vertex(False)
        color = _rust.scene_direction_color(
            np.ascontiguousarray(vertex, dtype=np.float64),
            [float(x), float(y), float(z)],
        )
        return self.vert_color(color)

    def cylinder_color(
        self, center: list[float], direction: list[float], up: list[float]
    ) -> "Object":
        """Set the color along the cylinder direction.

        Args:
            center (list[float]): The center of the cylinder.
            direction (list[float]): The direction of the cylinder.
            up (list[float]): The up vector of the cylinder.

        Returns:
            Object: The object with the updated color.

        Example:
            Apply a cylinder gradient used in the twist demo::

                obj = scene.add("cylinder").cylinder_color(
                    [0, 0, 0], [1, 0, 0], [0, 1, 0],
                )
        """
        vertex = self.vertex(False)
        color = _rust.scene_cylinder_color(
            np.ascontiguousarray(vertex, dtype=np.float64),
            [float(center[0]), float(center[1]), float(center[2])],
            [float(direction[0]), float(direction[1]), float(direction[2])],
            [float(up[0]), float(up[1]), float(up[2])],
        )
        return self.vert_color(color)

    def dyn_color(self, color: str, intensity: float = 0.75) -> "Object":
        """Set the dynamic color of the object.

        Args:
            color (str): The dynamic color type. Currently only ``"area"`` is supported.
            intensity (float, optional): Blend intensity of the dynamic color. Defaults to 0.75.

        Returns:
            Object: The object with the updated dynamic color.

        Example:
            Highlight stretched triangles on a trampoline sheet::

                scene.add("sheet").dyn_color("area", 1.0)
        """
        if not _rust.scene_is_supported_dyn_color(color):
            raise Exception("invalid color type")
        self._dyn_color = EnumColor.AREA
        self._dyn_intensity = intensity
        return self

    def velocity(self, u: float, v: float, w: float, t: float = 0.0) -> "Object":
        """Set the velocity of the object.

        Args:
            u (float): The velocity in the x-direction.
            v (float): The velocity in the y-direction.
            w (float): The velocity in the z-direction.
            t (float): Time in seconds. 0.0 sets initial velocity; >0 adds a timed override.

        Returns:
            Object: The object with the updated velocity.

        Example:
            Give an armadillo a downward initial velocity of 5 m/s::

                scene.add("armadillo").at(0, 1, 0).velocity(0, -5, 0)
        """
        _rust.scene_validate_object_not_static(bool(self.static))
        replace, vel = _rust.scene_classify_velocity_entry(
            float(u), float(v), float(w), float(t)
        )
        if replace:
            self._velocity = np.array(list(vel))
        else:
            self._velocity_schedule.append((t, list(vel)))
        return self

    def angular_velocity(self, wx: float, wy: float, wz: float) -> "Object":
        """Set an initial angular velocity (radians per second) about
        the object's centroid.

        At scene-build time the angular velocity is baked into the
        per-vertex initial velocities as ``ω × (x̄ − c̄)``, on top of
        any linear velocity set via :meth:`velocity`. Primary use case
        is PDRD bodies that should start spinning; works for any
        dynamic object's surface vertices.

        Args:
            wx, wy, wz: Components of the angular velocity vector
                (rad/s), giving rotation about the object's centroid
                following the right-hand rule.

        Returns:
            Object: The object with the updated angular velocity.

        Example:
            A cube spinning at 1 rad/s about the y-axis::

                scene.add("cube").as_pdrd().angular_velocity(0, 1, 0)
        """
        _rust.scene_validate_object_not_static(bool(self.static))
        self._angular_velocity = [float(wx), float(wy), float(wz)]
        return self

    def hinge(self, pca_axis: int = 2) -> "Object":
        """Pin a PDRD body as a hinge about one of its principal axes.

        Filters the body's reduced rigid DOF so its position is fixed
        and its rotation is restricted to a single axle, a hinge / pin
        joint. The axle is one of the three principal axes (PCA axes) of
        the body's rest shape, selected by ``pca_axis`` (like Blender's
        torque-axis dropdown), and passes through the body centroid.

        This is the building block for gears: hinge each gear to its
        axle and let tooth contact transmit the torque (no explicit
        gear-ratio constraint is needed).

        Args:
            pca_axis: Which principal axis is the free rotation axle,
                ``0``, ``1`` or ``2``, ordered by descending rest-shape
                variance (``0`` = largest extent, ``2`` = smallest). A
                flat gear or disk spins about its smallest-extent axis,
                so ``pca_axis=2`` is the usual choice (the default).

        Returns:
            Object: ``self``, for chaining.

        Raises:
            ValueError: if the object is not a PDRD body, or ``pca_axis``
                is not in ``{0, 1, 2}``.

        Example:
            A gear hinged about its thin axis, given an initial spin::

                gear = scene.add("gear").as_pdrd().hinge(2)
                gear.angular_velocity(0, 0, 6.28)
        """
        if not self._is_pdrd:
            raise ValueError("hinge() requires a PDRD body; call as_pdrd() first")
        if pca_axis not in (0, 1, 2):
            raise ValueError(f"hinge() pca_axis must be 0, 1 or 2; got {pca_axis!r}")
        self._joint = ("hinge", int(pca_axis))
        return self

    def velocity_schedule(self, schedule: list) -> "Object":
        """Set a list of (time, [vx, vy, vz]) velocity overrides.

        Each entry is routed exactly like :meth:`velocity`: an entry at
        ``t <= 0`` sets the initial velocity (``self._velocity``), while
        ``t > 0`` entries are timed overrides. The object must not be static.

        Example:
            Give an initial downward kick, then stop the object a second later::

                obj = scene.add("armadillo").at(0, 1, 0)
                obj.velocity_schedule([
                    (0.0, [0, -5, 0]),
                    (1.0, [0, 0, 0]),
                ])
        """
        _rust.scene_validate_object_not_static(bool(self.static))
        self._velocity_schedule = []
        for t, vel in schedule:
            replace, vel = _rust.scene_classify_velocity_entry(
                float(vel[0]), float(vel[1]), float(vel[2]), float(t)
            )
            if replace:
                self._velocity = np.array(list(vel))
            else:
                self._velocity_schedule.append((t, list(vel)))
        return self

    def angular_velocity_pca(
        self, pca_index: int, speed: float, t: float = 0.0
    ) -> "Object":
        """Add a principal-axis angular velocity overwrite keyframe.

        Unlike :meth:`angular_velocity` (which bakes a single free-vector spin
        at t=0), this spins the body about one of its **principal axes**, and
        the axis is resolved dynamically in the solver from the body's live
        geometry at the keyframe time, so it tracks the simulated (rotated or
        deformed) pose. Applies to solid, shell, and PDRD objects.

        Args:
            pca_index: Principal axis to spin about, 0 = largest extent,
                1 = medium, 2 = thinnest (same convention as :meth:`hinge`).
            speed: Signed angular speed in radians per second (right-hand rule
                about the canonically-oriented axis).
            t: Time in seconds at which the spin is applied (0.0 = at start).

        Returns:
            Object: The object with the angular keyframe appended.

        Example:
            Spin a gear up about its thinnest axis at t = 0.5 s::

                scene.add("gear").as_pdrd().angular_velocity_pca(2, 6.28, t=0.5)
        """
        _rust.scene_validate_object_not_static(bool(self.static))
        if int(pca_index) not in (0, 1, 2):
            raise ValueError(f"pca_index must be 0, 1, or 2 (got {pca_index})")
        self._angular_velocity_schedule_pca.append(
            (float(t), int(pca_index), float(speed))
        )
        return self

    def angular_velocity_schedule_pca(self, schedule: list) -> "Object":
        """Set the list of principal-axis angular velocity overwrite keyframes.

        Replaces the current schedule with ``schedule``, a list of
        ``(time, pca_index, speed_rad_per_s)`` entries (see
        :meth:`angular_velocity_pca` for the axis convention). The object must
        not be static.

        Example:
            Spin up at 1 s, reverse at 2 s::

                obj.angular_velocity_schedule_pca([
                    (1.0, 2, 6.28),
                    (2.0, 2, -6.28),
                ])
        """
        _rust.scene_validate_object_not_static(bool(self.static))
        self._angular_velocity_schedule_pca = []
        for t, pca_index, speed in schedule:
            if int(pca_index) not in (0, 1, 2):
                raise ValueError(f"pca_index must be 0, 1, or 2 (got {pca_index})")
            self._angular_velocity_schedule_pca.append(
                (float(t), int(pca_index), float(speed))
            )
        return self

    def angular_velocity_world(
        self, wx: float, wy: float, wz: float, t: float = 0.0
    ) -> "Object":
        """Add a fixed world-axis angular velocity overwrite keyframe.

        Unlike :meth:`angular_velocity_pca` (whose axis tracks the body), the
        spin axis here is a fixed world-space direction: ``(wx, wy, wz)`` is
        the full angular velocity vector (rad/s), its direction the axle and
        its magnitude the speed. Applies to solid, shell, and PDRD objects.

        Args:
            wx, wy, wz: World-space angular velocity components (rad/s).
            t: Time in seconds at which the spin is applied.

        Returns:
            Object: The object with the angular keyframe appended.

        Example:
            Spin about the world Y axis at 2pi rad/s starting at t = 0.5 s::

                obj.angular_velocity_world(0.0, 6.28, 0.0, t=0.5)
        """
        _rust.scene_validate_object_not_static(bool(self.static))
        self._angular_velocity_schedule_world.append(
            (float(t), [float(wx), float(wy), float(wz)])
        )
        return self

    def angular_velocity_schedule_world(self, schedule: list) -> "Object":
        """Set the list of fixed world-axis angular overwrite keyframes.

        Replaces the current schedule with ``schedule``, a list of
        ``(time, [wx, wy, wz])`` entries (see :meth:`angular_velocity_world`).
        """
        _rust.scene_validate_object_not_static(bool(self.static))
        self._angular_velocity_schedule_world = []
        for t, vec in schedule:
            self._angular_velocity_schedule_world.append(
                (float(t), [float(vec[0]), float(vec[1]), float(vec[2])])
            )
        return self

    def collision_windows(self, windows: list) -> "Object":
        """Set collision active time windows: list of (t_start, t_end) pairs.

        Example:
            Enable contact only during t in [0.2, 1.0] and [2.0, 3.0]::

                scene.add("sheet").collision_windows([(0.2, 1.0), (2.0, 3.0)])
        """
        self._collision_windows = windows
        return self

    def update_static(self):
        """Recompute whether the object is static.

        When every vertex is pinned and no pin carries operations, a pull
        strength, or an unpin time, the object is treated as static. The
        result is cached on ``self._static``.

        Example:
            Typically invoked internally by :meth:`Scene.build` after pins
            are finalized, but can be called directly to refresh the cached
            flag after editing pin data in place::

                obj = scene.select("sheet")
                obj.pin()
                obj.update_static()
                assert obj.static
        """
        # A STATIC promoted into the dynamic namespace to be a cross-stitch
        # endpoint is fully pinned with no operations, which would otherwise
        # classify it static (and route it to the unreachable collision-mesh
        # pool). The _force_dynamic marker keeps it in dyn_objects so the
        # stitch index can address it; its immovable fixed pins still freeze
        # it at rest. Set by the frontend decoder's _populate_static.
        if getattr(self, "_force_dynamic", False):
            self._static = False
            return

        if not self._pin:
            self._static = False
            return

        for p in self._pin:
            if len(p.operations) > 0 or p.pull_strength or p.unpin_time is not None:
                self._static = False
                return

        vert = self.get("V")
        if vert is None:
            self._static = False
            return

        self._static = _rust.scene_all_vertices_pinned(
            int(len(vert)),
            [list(p.index) for p in self._pin],
        )

    def pin(self, ind: Optional[list[int]] = None) -> PinHolder:
        """Set specified vertices as pinned.

        An object with every vertex pinned is a *static collider*:
        its motion is prescribed, not simulated. Use the returned
        :class:`PinHolder` to animate it via
        :meth:`PinHolder.move_by`, :meth:`PinHolder.move_to`, or
        :meth:`PinHolder.transform_keyframes`.

        Args:
            ind (Optional[list[int]], optional): The indices of the vertices to pin.
            If None, all vertices are pinned. Defaults to None.

        Returns:
            PinHolder: The pin holder.

        Example:
            Static sphere that slides across the scene::

                (scene.add("sphere")
                      .at(-1, 0, 0)
                      .pin()
                      .move_by([8, 0, 0], t_start=0.0, t_end=5.0))
        """
        if ind is None:
            vert: np.ndarray = self.vertex(False)
            ind = list(range(len(vert)))

        holder = PinHolder(self, ind)
        self._pin.append(holder)
        return holder

    def stitch(self, name: str) -> "Object":
        """Apply stitch to the object.

        Args:
            name (str): The name of stitch registered in the asset manager.

        Returns:
            Object: The stitched object.

        Example:
            Attach a glue stitch registered earlier in the asset manager::

                app.asset.add.stitch("glue", stitch_data)
                scene.add("dress").stitch("glue").rotate(-90, "x")
        """
        # Static check first (matches Python source: avoids asset fetch
        # when the object is already pinned every-vertex).
        _rust.scene_validate_object_not_static(bool(self.static))
        stitch = self._asset.fetch.get(name)
        _rust.scene_validate_stitch_attach(
            False, "Ind" in stitch, "W" in stitch
        )
        self._stitch = (stitch["Ind"], stitch["W"])
        return self

    def set_uv(self, uv: list[np.ndarray]) -> "Object":
        """Set the UV coordinates of the object.

        Args:
            uv (list[np.ndarray]): The UV coordinates for each face.

        Returns:
            Object: The object with the updated UV coordinates.

        Example:
            Supply per-face UVs for a triangulated sheet::

                import numpy as np
                sheet = scene.add("sheet")
                n_faces = len(sheet.get("F"))
                uv = [np.zeros((3, 2), dtype=np.float32) for _ in range(n_faces)]
                sheet.set_uv(uv)
        """
        _rust.scene_validate_set_uv_obj_type(self.obj_type)
        self._uv = uv
        return self

    def set_bend_rest_vert(self, vert: np.ndarray) -> "Object":
        """Set the world-space reference vertices for the bending rest angle.

        ``vert`` is an ``(N, 3)`` array in the SAME vertex order and world
        space as :meth:`vertex` ``(True)``; the solver uses it to compute
        this object's hinge rest angles from the reference shape instead of
        the object's own initial pose.
        """
        self._bend_rest_vert = np.ascontiguousarray(vert, dtype=np.float64)
        return self

    def direction(self, _ex: list[float], _ey: list[float]) -> "Object":
        """Set two orthogonal directions of a shell required for Baraff-Witkin model.

        Args:
            _ex (list[float]): The 3D x-direction vector.
            _ey (list[float]): The 3D y-direction vector.

        Returns:
            Object: The object with the updated direction.

        Example:
            Pin the warp and weft directions of a flat sheet in the xz plane::

                sheet = scene.add("sheet")
                sheet.param.set("model", "baraff-witkin")
                sheet.direction([1, 0, 0], [0, 0, 1])
        """
        vert, tri = self.vertex(False), self.get("F")
        if vert is None:
            raise ValueError("vertex does not exist")
        if tri is None:
            raise ValueError("face does not exist")
        uv_arr = _rust.scene_uv_from_directions(
            np.ascontiguousarray(vert, dtype=np.float64),
            np.ascontiguousarray(tri, dtype=np.int64),
            [float(_ex[0]), float(_ex[1]), float(_ex[2])],
            [float(_ey[0]), float(_ey[1]), float(_ey[2])],
            EPS,
        )
        self._uv = [uv_arr[i] for i in range(uv_arr.shape[0])]
        return self
