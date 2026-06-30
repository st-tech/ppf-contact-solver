# File: _scene_collider_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Invisible collider primitives (Wall + Sphere) used by Scene.

Split out of ``_scene_.py``: see :func:`Scene.add.invisible.wall` /
``Scene.add.invisible.sphere`` for the typical entry points.
"""

from . import _rust  # type: ignore[attr-defined]


class ColliderParam:
    """A class to hold invisible collider parameters (shared by Wall and Sphere)."""

    def __init__(self):
        self._param = {
            "contact-gap": 1e-3,
            "friction": 0.0,
            "active-duration": -1.0,
            "thickness": 1.0,
        }

    def list(self) -> dict[str, float]:
        """List all the parameters for the collider.

        Returns:
            dict[str, float]: A dictionary of collider parameters.
        """
        return self._param

    def set(self, name: str, value: float) -> "ColliderParam":
        """Set a parameter for the collider.

        Args:
            name (str): The parameter name (must already exist in the defaults).
            value (float): The new value.

        Returns:
            ColliderParam: The updated collider parameters.
        """
        _rust.scene_validate_known_param_name(name, list(self._param.keys()))
        self._param[name] = value
        return self


class WallParam(ColliderParam):
    """A class to hold wall parameters."""


class Wall:
    """An invisible wall class.

    Example:
        Box a trampoline in with four invisible walls::

            gap = 0.025
            scene.add.invisible.wall([1 + gap, 0, 0], [-1, 0, 0])
            scene.add.invisible.wall([-1 - gap, 0, 0], [1, 0, 0])
            scene.add.invisible.wall([0, 0, 1 + gap], [0, 0, -1])
            scene.add.invisible.wall([0, 0, -1 - gap], [0, 0, 1])
    """

    def __init__(self):
        """Initialize the wall."""
        self._entry = []
        self._transition = "linear"
        self._param = WallParam()

    def get_entry(self) -> list[tuple[list[float], float]]:
        """Get a list of time-dependent wall entries.

        Returns:
            list[tuple[list[float], float]]: A list of time-dependent entries, each containing a position and time.

        Example:
            Inspect every keyframed position on an animated wall::

                wall = scene.add.invisible.wall([0, 0, 0], [0, 1, 0])
                wall.move_to([0, 0.2, 0], 2.0)
                for pos, t in wall.get_entry():
                    print(t, pos)
        """
        return self._entry

    def is_static_collider(self) -> bool:
        """Return whether this wall is a static collider.

        A wall is static when it has exactly one keyframe entry. An
        empty entry list and a multi-keyframe (kinematic) wall both
        return ``False``: kinematic walls are handled elsewhere.

        Returns:
            bool: ``True`` when the wall has exactly one keyframe.
        """
        return len(self._entry) == 1

    def add(self, pos: list[float], normal: list[float]) -> "Wall":
        """Add the initial wall entry.

        Args:
            pos (list[float]): The position of the wall.
            normal (list[float]): The outer normal of the wall.

        Returns:
            Wall: The invisible wall.

        Example:
            Place a ground wall at y=0 with the outer normal facing up::

                wall = Wall().add([0, 0, 0], [0, 1, 0])
        """
        _rust.scene_validate_collider_not_already_added(bool(self._entry), "wall")
        self._normal = normal
        self._entry.append((pos, 0.0))
        return self

    def _check_time(self, time: float):
        """Check if the time is valid.

        Args:
            time (float): The time to check.
        """
        _rust.scene_validate_collider_time(float(self._entry[-1][1]), float(time))

    def move_by(self, delta: list[float], time: float) -> "Wall":
        """Move the wall by a positional delta at a specific time.

        Args:
            delta (list[float]): The positional delta to move the wall.
            time (float): The absolute time to move the wall.

        Returns:
            Wall: The invisible wall.

        Example:
            Raise the ground wall by 0.2 units at t=2 seconds::

                wall = scene.add.invisible.wall([0, 0, 0], [0, 1, 0])
                wall.move_by([0, 0.2, 0], 2.0)
        """
        self._check_time(time)
        prev = self._entry[-1][0]
        pos = list(_rust.scene_wall_move_by_position(
            [float(prev[0]), float(prev[1]), float(prev[2])],
            [float(delta[0]), float(delta[1]), float(delta[2])],
        ))
        self._entry.append((pos, time))
        return self

    def move_to(self, pos: list[float], time: float) -> "Wall":
        """Move the wall to an absolute position at a specific time.

        Args:
            pos (list[float]): The target position of the wall.
            time (float): The absolute time to move the wall.

        Returns:
            Wall: The invisible wall.

        Example:
            Animate a piston wall toward an absolute target::

                wall = scene.add.invisible.wall([0, 0, 0], [1, 0, 0])
                wall.move_to([0.5, 0, 0], 3.0)
        """
        self._check_time(time)
        self._entry.append((pos, time))
        return self

    def interp(self, transition: str) -> "Wall":
        """Set the transition type for the wall.

        Args:
            transition (str): The transition type. Defaults to ``"linear"`` if never set.

        Returns:
            Wall: The invisible wall.

        Example:
            Ease the wall motion with a smoothstep instead of a linear ramp::

                wall.move_to([0.5, 0, 0], 3.0).interp("smooth")
        """
        self._transition = transition
        return self

    @property
    def normal(self) -> list[float]:
        """Get the wall normal.

        Example:
            Read the outer normal back from an added wall::

                wall = scene.add.invisible.wall([0, 0, 0], [0, 1, 0])
                print(wall.normal)
        """
        return self._normal

    @property
    def entry(self) -> list[tuple[list[float], float]]:
        """Get the wall entries.

        Example:
            Iterate the wall animation keyframes::

                wall = scene.add.invisible.wall([0, 0, 0], [0, 1, 0])
                for pos, t in wall.entry:
                    print(t, pos)
        """
        return self._entry

    @property
    def transition(self) -> str:
        """Get the wall transition.

        Example:
            Check the interpolation mode of an animated wall::

                wall = scene.add.invisible.wall([0, 0, 0], [0, 1, 0]).interp("smooth")
                assert wall.transition == "smooth"
        """
        return self._transition

    @property
    def param(self) -> WallParam:
        """Get the wall parameters.

        Example:
            Tweak a wall parameter through the returned holder::

                wall = scene.add.invisible.wall([0, 0, 0], [0, 1, 0])
                wall.param.set("friction", 0.2)
        """
        return self._param


class SphereParam(ColliderParam):
    """A class to hold sphere parameters."""


class Sphere:
    """An invisible sphere class.

    Example:
        Add an inverted hemispherical bowl as a collider::

            scene.add.invisible.sphere([0, 1, 0], 1.0).invert().hemisphere()
    """

    def __init__(self):
        """Initialize the sphere."""
        self._entry = []
        self._hemisphere = False
        self._invert = False
        self._transition = "linear"
        self._param = SphereParam()

    def hemisphere(self) -> "Sphere":
        """Turn the sphere into a hemisphere, so the top half becomes empty, like a bowl.

        Example:
            Cup-shaped collider by combining ``invert`` and ``hemisphere``::

                scene.add.invisible.sphere([0, 1, 0], 1.0).invert().hemisphere()
        """
        self._hemisphere = True
        return self

    def invert(self) -> "Sphere":
        """Invert the sphere, so the inside becomes empty and the outside becomes solid.

        Example:
            Trap objects inside an inverted sphere boundary::

                scene.add.invisible.sphere([0, 0, 0], 2.0).invert()
        """
        self._invert = True
        return self

    def interp(self, transition: str) -> "Sphere":
        """Set the transition type for the sphere.

        Args:
            transition (str): The transition type. Defaults to ``"linear"`` if never set.

        Returns:
            Sphere: The sphere.

        Example:
            Ease the sphere motion with a smoothstep::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.5)
                sphere.move_to([0, 0.5, 0], 2.0).interp("smooth")
        """
        self._transition = transition
        return self

    def get_entry(self) -> list[tuple[list[float], float, float]]:
        """Get the time-dependent sphere entries.

        Example:
            Iterate over an animated sphere's keyframes (position, radius, time)::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.25)
                sphere.transform_to([0, 0.5, 0], 0.5, 2.0)
                for pos, r, t in sphere.get_entry():
                    print(t, r, pos)
        """
        return self._entry

    def is_static_collider(self) -> bool:
        """Return whether this sphere is a static collider.

        A sphere is static when it has exactly one keyframe entry. An
        empty entry list and a multi-keyframe (kinematic) sphere both
        return ``False``: kinematic spheres are handled elsewhere.

        Returns:
            bool: ``True`` when the sphere has exactly one keyframe.
        """
        return len(self._entry) == 1

    def add(self, pos: list[float], radius: float) -> "Sphere":
        """Add an invisible sphere information.

        Args:
            pos (list[float]): The position of the sphere.
            radius (float): The radius of the sphere.

        Returns:
            Sphere: The sphere.

        Example:
            Instantiate a unit sphere at the origin::

                sphere = Sphere().add([0, 0, 0], 1.0)
        """
        _rust.scene_validate_collider_not_already_added(bool(self._entry), "sphere")
        self._entry.append((pos, radius, 0.0))
        return self

    def _check_time(self, time: float):
        """Check if the time is valid.

        Args:
            time (float): The time to check.
        """
        _rust.scene_validate_collider_time(float(self._entry[-1][2]), float(time))

    def transform_to(self, pos: list[float], radius: float, time: float) -> "Sphere":
        """Change the sphere to a new position and radius at a specific time.

        Args:
            pos (list[float]): The target position of the sphere.
            radius (float): The target radius of the sphere.
            time (float): The absolute time to transform the sphere.

        Returns:
            Sphere: The sphere.

        Example:
            Grow the collider sphere as it slides into place::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.25)
                sphere.transform_to([0, 0.5, 0], 0.5, 2.0)
        """
        self._check_time(time)
        self._entry.append((pos, radius, time))
        return self

    def move_by(self, delta: list[float], time: float) -> "Sphere":
        """Move the sphere by a positional delta at a specific time.

        Args:
            delta (list[float]): The positional delta to move the sphere.
            time (float): The absolute time to move the sphere.

        Returns:
            Sphere: The sphere.

        Example:
            Slide a collider sphere 1 unit along +x by t=2::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.5)
                sphere.move_by([1, 0, 0], 2.0)
        """
        self._check_time(time)
        prev = self._entry[-1][0]
        prev_radius = float(self._entry[-1][1])
        new_pos, new_radius = _rust.scene_sphere_move_by_entry(
            [float(prev[0]), float(prev[1]), float(prev[2])],
            prev_radius,
            [float(delta[0]), float(delta[1]), float(delta[2])],
        )
        self._entry.append((list(new_pos), new_radius, time))
        return self

    def move_to(self, pos: list[float], time: float) -> "Sphere":
        """Move the sphere to an absolute position at a specific time.

        Args:
            pos (list[float]): The target position of the sphere.
            time (float): The absolute time to move the sphere.

        Returns:
            Sphere: The sphere.

        Example:
            Drop a ball from above to y=0 by t=1::

                sphere = scene.add.invisible.sphere([0, 1, 0], 0.25)
                sphere.move_to([0, 0, 0], 1.0)
        """
        self._check_time(time)
        radius = self._entry[-1][1]
        self._entry.append((pos, radius, time))
        return self

    def radius(self, radius: float, time: float) -> "Sphere":
        """Change the radius of the sphere at a specific time.

        Args:
            radius (float): The target radius of the sphere.
            time (float): The absolute time to change the radius.

        Returns:
            Sphere: The sphere.

        Example:
            Inflate the sphere from 0.25 to 0.5 by t=1::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.25)
                sphere.radius(0.5, 1.0)
        """
        self._check_time(time)
        pos = self._entry[-1][0]
        self._entry.append((pos, radius, time))
        return self

    @property
    def entry(self) -> list[tuple[list[float], float, float]]:
        """Get the sphere entries.

        Example:
            Inspect the keyframed (position, radius, time) list::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.5)
                for pos, r, t in sphere.entry:
                    print(t, r, pos)
        """
        return self._entry

    @property
    def is_hemisphere(self) -> bool:
        """Get whether sphere is hemisphere.

        Example:
            Check the hemisphere flag on a collider::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.5)
                assert sphere.is_hemisphere in (True, False)
        """
        return self._hemisphere

    @property
    def is_inverted(self) -> bool:
        """Get whether sphere is inverted.

        Example:
            An inverted sphere acts as a containing bowl::

                sphere = scene.add.invisible.sphere([0, 0, 0], 1.0, invert=True)
                assert sphere.is_inverted
        """
        return self._invert

    @property
    def transition(self) -> str:
        """Get the sphere transition.

        Example:
            Read back the interpolation mode::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.5).interp("smooth")
                assert sphere.transition == "smooth"
        """
        return self._transition

    @property
    def param(self) -> SphereParam:
        """Get the sphere parameters.

        Example:
            Adjust a sphere parameter through the returned holder::

                sphere = scene.add.invisible.sphere([0, 0, 0], 0.5)
                sphere.param.set("friction", 0.2)
        """
        return self._param
