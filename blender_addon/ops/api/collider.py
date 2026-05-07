"""Invisible collider proxies (walls, spheres) and shared collider param proxy.

See :mod:`blender_addon.ops.api` for the package overview.
"""

import bpy  # pyright: ignore

from ...models.collection_utils import (
    generate_unique_name,
    safe_update_index,
    sort_keyframes_by_frame,
    validate_no_duplicate_frame,
)
from ...models.groups import get_addon_data
from .._api_markers import blender_api


# ---------------------------------------------------------------------------
# Invisible collider builders
# ---------------------------------------------------------------------------


@blender_api
class _ColliderParamProxy:
    """Attribute proxy for invisible-collider parameters.

    Accessed via :attr:`Wall.param` or :attr:`Sphere.param`.  Attribute
    access is whitelisted: reading or writing a name outside the
    whitelist raises :class:`AttributeError`.

    Whitelisted attributes:

    - ``friction``: contact friction coefficient
    - ``contact_gap``: contact gap thickness
    - ``thickness``: wall/sphere shell thickness
    - ``enable_active_duration``: ``True`` to limit collider lifetime
    - ``active_duration``: number of frames the collider is active when
      ``enable_active_duration`` is set

    Example::

        wall = solver.add_wall((0, 0, 0), (0, 0, 1))
        wall.param.friction = 0.5
        wall.param.contact_gap = 0.002
        wall.param.thickness = 0.01
        wall.param.enable_active_duration = True
        wall.param.active_duration = 60  # active for frames 1-60

        sphere = solver.add_sphere((0, 0, 1), 0.5)
        sphere.param.friction = 0.3
    """

    def __init__(self, item):
        object.__setattr__(self, "_item", item)

    def __setattr__(self, key, value):
        item = object.__getattribute__(self, "_item")
        if key in ("contact_gap", "friction", "thickness", "enable_active_duration", "active_duration"):
            setattr(item, key, value)
        else:
            raise AttributeError(f"Unknown collider param '{key}'")

    def __getattr__(self, key):
        item = object.__getattribute__(self, "_item")
        if key in ("contact_gap", "friction", "thickness", "enable_active_duration", "active_duration"):
            return getattr(item, key)
        raise AttributeError(f"Unknown collider param '{key}'")

@blender_api
class _InvisibleWallBuilder:
    """Chainable builder for invisible wall colliders.

    Returned by :meth:`Solver.add_wall`.  Keyframe frames must be
    strictly increasing.  Every mutating method returns ``self``.

    Example::

        solver.add_wall((0, 0, 0), (0, 0, 1)).param.friction = 0.5
        (solver.add_wall((0, 0, 0), (0, 1, 0))
               .time(60).hold().time(61).move_to((0, 1, 0)))
    """

    def __init__(self, position, normal):
        state = get_addon_data(bpy.context.scene).state
        self._item = state.invisible_colliders.add()
        self._item.collider_type = "WALL"
        # Auto-name
        existing = [c.name for c in state.invisible_colliders if c.collider_type == "WALL"]
        self._item.name = generate_unique_name("Wall", existing)
        self._item.position = tuple(position)
        self._item.normal = tuple(normal)
        kf = self._item.keyframes.add()
        kf.frame = 1
        state.invisible_colliders_index = len(state.invisible_colliders) - 1
        self._frame = 1

    @classmethod
    def attach_to_last(cls) -> "_InvisibleWallBuilder":
        """Return a builder bound to the most-recently-added collider
        without re-adding.  Used by ``_Solver.add_wall`` after the service
        has already performed the add."""
        inst = cls.__new__(cls)
        state = get_addon_data(bpy.context.scene).state
        inst._item = state.invisible_colliders[-1]
        inst._frame = 1
        return inst

    @property
    @blender_api
    def param(self) -> _ColliderParamProxy:
        """Collider parameter proxy.  See :class:`ColliderParam`.

        Example::

            wall = solver.add_wall((0, 0, 0), (0, 0, 1))
            wall.param.friction = 0.5
        """
        return _ColliderParamProxy(self._item)

    @blender_api
    def time(self, frame: int) -> "_InvisibleWallBuilder":
        """Advance the keyframe cursor.

        Args:
            frame: Target frame (must be strictly greater than the current
                cursor position).

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If *frame* is not strictly increasing.

        Example::

            (solver.add_wall((0, 0, 0), (0, 0, 1))
                   .time(60).move_to((0, 0, 0.5)))
        """
        frame = int(frame)
        if frame <= self._frame:
            raise ValueError(f"Frame must be increasing: {frame} <= {self._frame}")
        self._frame = frame
        return self

    @blender_api
    def hold(self) -> "_InvisibleWallBuilder":
        """Hold the previous position at the current cursor frame.

        Returns:
            ``self`` for chaining.

        Example::

            (solver.add_wall((0, 0, 0), (0, 0, 1))
                   .time(60).hold().time(90).move_to((0, 0, 0.5)))
        """
        self._add_keyframe(use_hold=True)
        return self

    @blender_api
    def move_to(self, position) -> "_InvisibleWallBuilder":
        """Keyframe a new absolute position at the current cursor frame.

        Args:
            position: ``(x, y, z)`` world-space position.

        Returns:
            ``self`` for chaining.

        Example::

            (solver.add_wall((0, 0, 0), (0, 0, 1))
                   .time(60).move_to((0, 0, 1.0)))
        """
        self._add_keyframe(use_hold=False, position=position)
        return self

    @blender_api
    def move_by(self, delta) -> "_InvisibleWallBuilder":
        """Keyframe a position offset from the previous keyframe.

        Args:
            delta: ``(dx, dy, dz)`` offset added to the previous
                keyframed position.

        Returns:
            ``self`` for chaining.

        Example::

            (solver.add_wall((0, 0, 0), (0, 0, 1))
                   .time(60).move_by((0, 0, 0.25)))
        """
        # Get previous position
        if len(self._item.keyframes) > 0:
            last_kf = self._item.keyframes[len(self._item.keyframes) - 1]
            prev = list(last_kf.position) if last_kf.frame > 1 else list(self._item.position)
        else:
            prev = list(self._item.position)
        new_pos = [prev[i] + delta[i] for i in range(3)]
        self._add_keyframe(use_hold=False, position=new_pos)
        return self

    @blender_api
    def delete(self) -> None:
        """Remove this wall collider from the scene.

        Example::

            wall = solver.add_wall((0, 0, 0), (0, 0, 1))
            wall.delete()
        """
        state = get_addon_data(bpy.context.scene).state
        for i, item in enumerate(state.invisible_colliders):
            if item == self._item:
                state.invisible_colliders.remove(i)
                state.invisible_colliders_index = safe_update_index(
                    state.invisible_colliders_index,
                    len(state.invisible_colliders),
                )
                from ...models.groups import invalidate_overlays
                invalidate_overlays()
                break

    def _add_keyframe(self, use_hold=False, position=None):
        validate_no_duplicate_frame(self._item.keyframes, self._frame)
        kf = self._item.keyframes.add()
        kf.frame = self._frame
        kf.use_hold = use_hold
        if not use_hold and position is not None:
            kf.position = tuple(position)
        sort_keyframes_by_frame(self._item.keyframes)

@blender_api
class _InvisibleSphereBuilder:
    """Chainable builder for invisible sphere colliders.

    Returned by :meth:`Solver.add_sphere`.  Keyframe frames must be
    strictly increasing.  Every mutating method returns ``self``.

    Example::

        solver.add_sphere((0, 0, 0), 0.98).invert().hemisphere()
        (solver.add_sphere((0, 0, 0), 1.0)
               .time(60).hold().time(61).radius(0.5))
    """

    def __init__(self, position, radius):
        state = get_addon_data(bpy.context.scene).state
        self._item = state.invisible_colliders.add()
        self._item.collider_type = "SPHERE"
        existing = [c.name for c in state.invisible_colliders if c.collider_type == "SPHERE"]
        self._item.name = generate_unique_name("Sphere", existing)
        self._item.position = tuple(position)
        self._item.radius = float(radius)
        kf = self._item.keyframes.add()
        kf.frame = 1
        state.invisible_colliders_index = len(state.invisible_colliders) - 1
        self._frame = 1

    @classmethod
    def attach_to_last(cls) -> "_InvisibleSphereBuilder":
        """Bind to the most-recently-added sphere collider without
        adding a new one (the service already did)."""
        inst = cls.__new__(cls)
        state = get_addon_data(bpy.context.scene).state
        inst._item = state.invisible_colliders[-1]
        inst._frame = 1
        return inst

    @property
    @blender_api
    def param(self) -> _ColliderParamProxy:
        """Collider parameter proxy.  See :class:`ColliderParam`.

        Example::

            sphere = solver.add_sphere((0, 0, 0), 1.0)
            sphere.param.friction = 0.3
        """
        return _ColliderParamProxy(self._item)

    @blender_api
    def invert(self) -> "_InvisibleSphereBuilder":
        """Flip the sphere inside-out so contact is on the inside surface.

        Returns:
            ``self`` for chaining.

        Example::

            solver.add_sphere((0, 0, 0), 1.0).invert()
        """
        self._item.invert = True
        return self

    @blender_api
    def hemisphere(self) -> "_InvisibleSphereBuilder":
        """Treat this collider as a hemisphere rather than a full sphere.

        Returns:
            ``self`` for chaining.

        Example::

            solver.add_sphere((0, 0, 0), 1.0).hemisphere()
        """
        self._item.hemisphere = True
        return self

    @blender_api
    def time(self, frame: int) -> "_InvisibleSphereBuilder":
        """Advance the keyframe cursor.

        Args:
            frame: Target frame (must be strictly greater than the current
                cursor position).

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If *frame* is not strictly increasing.

        Example::

            (solver.add_sphere((0, 0, 0), 1.0)
                   .time(60).move_to((0, 0, 1.0)))
        """
        frame = int(frame)
        if frame <= self._frame:
            raise ValueError(f"Frame must be increasing: {frame} <= {self._frame}")
        self._frame = frame
        return self

    @blender_api
    def hold(self) -> "_InvisibleSphereBuilder":
        """Hold the previous position and radius at the current cursor frame.

        Returns:
            ``self`` for chaining.

        Example::

            (solver.add_sphere((0, 0, 0), 1.0)
                   .time(60).hold().time(90).radius(0.5))
        """
        self._add_keyframe(use_hold=True)
        return self

    @blender_api
    def move_to(self, position) -> "_InvisibleSphereBuilder":
        """Keyframe a new absolute position at the current cursor frame.

        Args:
            position: ``(x, y, z)`` world-space position.

        Returns:
            ``self`` for chaining.

        Example::

            (solver.add_sphere((0, 0, 0), 1.0)
                   .time(60).move_to((0, 0, 2.0)))
        """
        self._add_keyframe(use_hold=False, position=position)
        return self

    @blender_api
    def radius(self, r) -> "_InvisibleSphereBuilder":
        """Keyframe a new radius at the current cursor frame.

        Args:
            r: New radius.

        Returns:
            ``self`` for chaining.

        Example::

            (solver.add_sphere((0, 0, 0), 1.0)
                   .time(60).radius(0.25))  # shrink over 60 frames
        """
        self._add_keyframe(use_hold=False, radius=r)
        return self

    @blender_api
    def transform_to(self, position, radius) -> "_InvisibleSphereBuilder":
        """Keyframe both position and radius together.

        Args:
            position: ``(x, y, z)`` world-space position.
            radius: New radius.

        Returns:
            ``self`` for chaining.

        Example::

            (solver.add_sphere((0, 0, 0), 1.0)
                   .time(60).transform_to((0, 0, 1.0), 0.5))
        """
        self._add_keyframe(use_hold=False, position=position, radius=radius)
        return self

    @blender_api
    def delete(self) -> None:
        """Remove this sphere collider from the scene.

        Example::

            sphere = solver.add_sphere((0, 0, 0), 1.0)
            sphere.delete()
        """
        state = get_addon_data(bpy.context.scene).state
        for i, item in enumerate(state.invisible_colliders):
            if item == self._item:
                state.invisible_colliders.remove(i)
                state.invisible_colliders_index = safe_update_index(
                    state.invisible_colliders_index,
                    len(state.invisible_colliders),
                )
                from ...models.groups import invalidate_overlays
                invalidate_overlays()
                break

    def _add_keyframe(self, use_hold=False, position=None, radius=None):
        validate_no_duplicate_frame(self._item.keyframes, self._frame)
        kf = self._item.keyframes.add()
        kf.frame = self._frame
        kf.use_hold = use_hold
        if not use_hold:
            if position is not None:
                kf.position = tuple(position)
            else:
                # Keep previous position
                kf.position = tuple(self._item.position)
            if radius is not None:
                kf.radius = float(radius)
            else:
                kf.radius = self._item.radius
        sort_keyframes_by_frame(self._item.keyframes)
