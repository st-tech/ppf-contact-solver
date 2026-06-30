"""Scene-level parameter proxies: ``_SceneProxy`` (scalar attrs) and
``_DynParamBuilder`` (keyframed dynamic parameters).

See :mod:`blender_addon.ops.api` for the package overview.
"""

import bpy  # pyright: ignore

from ...models.collection_utils import (
    safe_update_index,
    sort_keyframes_by_frame,
    validate_no_duplicate_frame,
)
from ...models.defaults import SCENE_PARAM_ALIASES
from ...models.groups import get_addon_data
from .._api_markers import blender_api


# ---------------------------------------------------------------------------
# Dynamic parameter builder
# ---------------------------------------------------------------------------

_DYN_PARAM_KEY_MAP = {
    "gravity": "GRAVITY",
    "wind": "WIND",
    "air_density": "AIR_DENSITY",
    "air_friction": "AIR_FRICTION",
    "vertex_air_damp": "VERTEX_AIR_DAMP",
}


@blender_api
class _DynParamBuilder:
    """Fluent builder for dynamic scene parameter keyframes.

    Mirrors the frontend ``session.param.dyn()`` API but uses **frames**
    instead of seconds.  Obtained from :meth:`SceneParam.dyn`.

    Valid parameter keys: ``"gravity"``, ``"wind"``, ``"air_density"``,
    ``"air_friction"``, ``"vertex_air_damp"``.

    Frames must be strictly increasing within a chain.  Every mutating
    method returns ``self`` so operations chain.

    Example::

        solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))
        solver.param.dyn("wind").time(30).hold().time(31).change((0, 1, 0), strength=5.0)
    """

    def __init__(self, key: str):
        param_type = _DYN_PARAM_KEY_MAP.get(key)
        if param_type is None:
            raise ValueError(
                f"Unknown dynamic parameter '{key}'. "
                f"Valid keys: {', '.join(_DYN_PARAM_KEY_MAP)}"
            )
        self._param_type = param_type
        self._frame = 1
        state = get_addon_data(bpy.context.scene).state
        # Find or create the DynParamItem
        self._item = None
        for item in state.dyn_params:
            if item.param_type == param_type:
                self._item = item
                break
        if self._item is None:
            self._item = state.dyn_params.add()
            self._item.param_type = param_type
            kf = self._item.keyframes.add()
            kf.frame = 1
            state.dyn_params_index = len(state.dyn_params) - 1

    @blender_api
    def time(self, frame: int) -> "_DynParamBuilder":
        """Advance the frame cursor.

        Args:
            frame: Target frame (must be strictly greater than the current
                cursor position).

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If *frame* is not strictly increasing.

        Example::

            solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))
        """
        frame = int(frame)
        if frame <= self._frame:
            raise ValueError(f"Frame must be increasing: {frame} <= {self._frame}")
        self._frame = frame
        return self

    @blender_api
    def hold(self) -> "_DynParamBuilder":
        """Hold the previous value at the current cursor frame (step function).

        Returns:
            ``self`` for chaining.

        Example::

            solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))
        """
        self._add_keyframe(use_hold=True)
        return self

    @blender_api
    def change(self, value, strength=None) -> "_DynParamBuilder":
        """Set a new value at the current cursor frame.

        Args:
            value: For ``"gravity"``, an ``(x, y, z)`` tuple.
                For ``"wind"``, an ``(x, y, z)`` direction tuple.
                For scalar keys (``"air_density"``, ``"air_friction"``,
                ``"vertex_air_damp"``), a ``float``.
            strength: Wind strength (only for ``"wind"``).

        Returns:
            ``self`` for chaining.

        Example::

            solver.param.dyn("wind").time(30).hold().time(31).change((0, 1, 0), strength=5.0)
        """
        self._add_keyframe(use_hold=False, value=value, strength=strength)
        return self

    @blender_api
    def clear(self) -> "_DynParamBuilder":
        """Remove this dynamic parameter entirely.

        Returns:
            ``self`` for chaining (though no further method on this
            builder will do anything meaningful after ``clear()``).

        Example::

            solver.param.dyn("wind").clear()
        """
        state = get_addon_data(bpy.context.scene).state
        removed = False
        for i, item in enumerate(state.dyn_params):
            if item.param_type == self._param_type:
                state.dyn_params.remove(i)
                state.dyn_params_index = safe_update_index(
                    state.dyn_params_index, len(state.dyn_params)
                )
                removed = True
                break
        self._item = None
        if removed:
            from ...models.groups import invalidate_overlays
            invalidate_overlays()
        return self

    def _add_keyframe(self, use_hold=False, value=None, strength=None):
        if self._item is None:
            raise ValueError("Dynamic parameter has been cleared")
        validate_no_duplicate_frame(self._item.keyframes, self._frame)
        kf = self._item.keyframes.add()
        kf.frame = self._frame
        kf.use_hold = use_hold
        if not use_hold and value is not None:
            if self._param_type == "GRAVITY":
                kf.gravity_value = tuple(value)
            elif self._param_type == "WIND":
                kf.wind_direction_value = tuple(value)
                if strength is not None:
                    kf.wind_strength_value = float(strength)
            else:
                kf.scalar_value = float(value)
        sort_keyframes_by_frame(self._item.keyframes)

# ---------------------------------------------------------------------------
# Scene proxy
# ---------------------------------------------------------------------------

@blender_api
class _SceneProxy:
    """Attribute proxy for scene and SSH/connection parameters.

    Accessed as :attr:`Solver.param`.  Supports both get and set via
    attribute access.  Writes go through the ``zozo_contact_solver.set``
    operator (with auto type coercion), reads fall through to the
    scene's addon state or SSH state.

    ``gravity`` is an alias for ``gravity_3d``.

    Example::

        solver.param.step_size = 0.004
        print(solver.param.gravity)

    Dynamic (keyframed) parameters are accessed via :meth:`dyn`::

        solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))
    """

    @blender_api
    def dyn(self, key: str) -> _DynParamBuilder:
        """Select a parameter for dynamic keyframing.

        Args:
            key: One of ``"gravity"``, ``"wind"``, ``"air_density"``,
                ``"air_friction"``, ``"vertex_air_damp"``.

        Returns:
            A chainable :class:`DynParam` builder.

        Raises:
            ValueError: If *key* is not one of the valid dynamic keys.

        Example::

            solver.param.dyn("gravity").time(60).hold().time(61).change((0, 0, 9.8))
        """
        return _DynParamBuilder(key)

    def __setattr__(self, key, value):
        bpy.ops.zozo_contact_solver.set(key=str(key), value=str(value))

    def __getattr__(self, key):
        key = SCENE_PARAM_ALIASES.get(key, key)
        scene = bpy.context.scene
        addon_data = get_addon_data(scene)
        if hasattr(addon_data.state, key):
            return getattr(addon_data.state, key)
        if hasattr(addon_data.ssh_state, key):
            return getattr(addon_data.ssh_state, key)
        raise AttributeError(f"No scene property '{key}'")
