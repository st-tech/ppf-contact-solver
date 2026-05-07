"""``_ParamProxy`` and ``_Group`` proxies for dynamics groups.

See :mod:`blender_addon.ops.api` for the package overview.
"""

import bpy  # pyright: ignore

from ...models.groups import get_addon_data
from .._api_markers import blender_api
from .pin import _Pin


# ---------------------------------------------------------------------------
# Group proxy
# ---------------------------------------------------------------------------

_PARAM_PROPS = {
    "solid_model", "shell_model",
    "shell_density", "solid_density", "rod_density",
    "shell_young_modulus", "solid_young_modulus", "rod_young_modulus",
    "shell_poisson_ratio", "solid_poisson_ratio",
    "friction",
    "contact_gap", "contact_gap_rat",
    "contact_offset", "contact_offset_rat",
    "enable_strain_limit", "strain_limit",
    "enable_inflate", "inflate_pressure",
    "enable_plasticity", "plasticity", "plasticity_threshold",
    "enable_bend_plasticity", "bend_plasticity", "bend_plasticity_threshold",
    "bend_rest_angle_source",
    "bend", "shrink", "shrink_x", "shrink_y",
    "stitch_stiffness",
}


@blender_api
class _ParamProxy:
    """Proxy for material and simulation parameters on a group.

    Accessed via :attr:`Group.param`.  Attribute access is whitelisted:
    reading or writing a name outside the whitelist raises
    :class:`AttributeError`.

    Whitelisted attributes:

    - **Solver model**: ``solid_model``, ``shell_model``
    - **Density**: ``solid_density``, ``shell_density``, ``rod_density``
    - **Young's modulus**: ``solid_young_modulus``, ``shell_young_modulus``,
      ``rod_young_modulus``
    - **Poisson ratio**: ``solid_poisson_ratio``, ``shell_poisson_ratio``
    - **Contact**: ``friction``, ``contact_gap``, ``contact_gap_rat``,
      ``contact_offset``, ``contact_offset_rat``
    - **Strain limit**: ``enable_strain_limit``, ``strain_limit``
    - **Inflation**: ``enable_inflate``, ``inflate_pressure``
    - **Plasticity**: ``enable_plasticity``, ``plasticity``,
      ``plasticity_threshold``
    - **Bend plasticity**: ``enable_bend_plasticity``, ``bend_plasticity``,
      ``bend_plasticity_threshold``, ``bend_rest_angle_source``
    - **Shell-specific**: ``bend``, ``shrink``, ``shrink_x``, ``shrink_y``,
      ``stitch_stiffness``

    Example::

        # Solver model
        group.param.solid_model = "ARAP"
        group.param.shell_model = "ARAP"

        # Density (kg/m^3 for solid/shell, kg/m for rod)
        group.param.solid_density = 1000.0
        group.param.shell_density = 0.3
        group.param.rod_density = 0.05

        # Young's modulus (Pa) and Poisson ratio
        group.param.solid_young_modulus = 1.0e6
        group.param.shell_young_modulus = 5.0e5
        group.param.rod_young_modulus = 1.0e7
        group.param.solid_poisson_ratio = 0.45
        group.param.shell_poisson_ratio = 0.30

        # Contact
        group.param.friction = 0.5
        group.param.contact_gap = 0.001
        group.param.contact_gap_rat = 0.1
        group.param.contact_offset = 0.002
        group.param.contact_offset_rat = 0.2

        # Strain limit
        group.param.enable_strain_limit = True
        group.param.strain_limit = 1.05

        # Inflation
        group.param.enable_inflate = True
        group.param.inflate_pressure = 100.0

        # Plasticity (shells and tets only)
        group.param.enable_plasticity = True
        group.param.plasticity = 0.3
        group.param.plasticity_threshold = 0.2
        group.param.enable_bend_plasticity = True
        group.param.bend_plasticity = 0.5
        group.param.bend_plasticity_threshold = 0.1
        group.param.bend_rest_angle_source = "REST"

        # Shell-specific
        group.param.bend = 1.0e-4
        group.param.shrink = 0.98
        group.param.shrink_x = 0.99
        group.param.shrink_y = 0.97
        group.param.stitch_stiffness = 5.0e4
    """

    def __init__(self, group_proxy: "_Group"):
        object.__setattr__(self, "_group_proxy", group_proxy)

    def __setattr__(self, key, value):
        if key not in _PARAM_PROPS:
            raise AttributeError(f"Unknown material parameter '{key}'")
        gp = object.__getattribute__(self, "_group_proxy")
        group = gp._get_group()
        setattr(group, key, value)
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()

    def __getattr__(self, key):
        if key not in _PARAM_PROPS:
            raise AttributeError(f"Unknown material parameter '{key}'")
        gp = object.__getattribute__(self, "_group_proxy")
        group = gp._get_group()
        return getattr(group, key)

@blender_api
class _Group:
    """A dynamics group proxy.

    Created via :meth:`Solver.create_group`.  Material parameters are
    accessed via :attr:`param`.  Every mutating method returns ``self``
    so operations chain.

    Example::

        group = solver.create_group("Shirt", type="SHELL")
        group.add("Shirt").set_overlay_color(0.9, 0.2, 0.1)
        group.param.friction = 0.5
        group.param.shell_density = 1.0
    """

    def __init__(self, uuid: str):
        object.__setattr__(self, "_uuid", uuid)
        object.__setattr__(self, "_param", _ParamProxy(self))

    @property
    @blender_api
    def uuid(self) -> str:
        """The UUID of this group.  Stable across renames.

        Example::

            group = solver.create_group("Shirt", type="SHELL")
            same_group = solver.get_group(group.uuid)
        """
        return object.__getattribute__(self, "_uuid")

    @property
    @blender_api
    def param(self) -> _ParamProxy:
        """Material and simulation parameter proxy.  See :class:`GroupParam`.

        Example::

            group.param.friction = 0.5
            group.param.shell_density = 1.0
        """
        return object.__getattribute__(self, "_param")

    def _get_group(self):
        from ...models.groups import get_group_by_uuid

        group = get_group_by_uuid(bpy.context.scene, self._uuid)
        if group is None:
            raise ValueError(f"Group '{self._uuid}' not found")
        return group

    # -- Overlay -------------------------------------------------------------

    @blender_api
    def set_overlay_color(self, r: float, g: float, b: float, a: float = 1.0) -> "_Group":
        """Set the viewport overlay color for this group and enable it.

        Args:
            r: Red channel in ``[0, 1]``.
            g: Green channel in ``[0, 1]``.
            b: Blue channel in ``[0, 1]``.
            a: Alpha in ``[0, 1]`` (default ``1.0``).

        Returns:
            ``self`` for chaining.

        Example::

            group.set_overlay_color(0.9, 0.2, 0.1)  # red overlay
        """
        from ...ui.dynamics.overlay import apply_object_overlays

        group = self._get_group()
        group.color = (r, g, b, a)
        group.show_overlay_color = True
        apply_object_overlays()
        return self

    # -- Object assignment ---------------------------------------------------

    @blender_api
    def add(self, *object_names: str) -> "_Group":
        """Add mesh objects to this group by name.

        Args:
            *object_names: One or more Blender object names.

        Returns:
            ``self`` for chaining.

        Example::

            group.add("Shirt", "Skirt", "Sleeve")
        """
        import json

        bpy.ops.zozo_contact_solver.add_objects_to_group(
            group_uuid=self._uuid, object_names=json.dumps(list(object_names))
        )
        return self

    @blender_api
    def remove(self, object_name: str) -> "_Group":
        """Remove an object from this group.

        Args:
            object_name: Name of the object to remove.

        Returns:
            ``self`` for chaining.

        Example::

            group.remove("Sleeve")
        """
        bpy.ops.zozo_contact_solver.remove_object_from_group(
            group_uuid=self._uuid, object_name=object_name,
        )
        return self

    # -- Pin management ------------------------------------------------------

    @blender_api
    def create_pin(self, object_name: str, vertex_group_name: str) -> _Pin:
        """Pin a vertex group so its vertices stay fixed during simulation.

        Args:
            object_name: Name of the mesh object.
            vertex_group_name: Name of the vertex group on that object.

        Returns:
            A :class:`Pin` proxy for the newly created pin.

        Raises:
            ValueError: If the object is missing, not a mesh, or the
                vertex group does not exist on it.

        Example::

            pin = group.create_pin("Cloth", "collar")
            pin.move(delta=(0, 0, 0.2), frame=60)
        """
        group = self._get_group()

        obj = bpy.data.objects.get(object_name)
        if obj is None:
            raise ValueError(f"Object '{object_name}' not found")
        if obj.type != "MESH":
            raise ValueError(f"Object '{object_name}' is not a mesh")
        if vertex_group_name not in [vg.name for vg in obj.vertex_groups]:
            raise ValueError(
                f"Vertex group '{vertex_group_name}' not found on '{object_name}'"
            )

        # Validation + locking + raw mutation all go through
        # core.mutation so scripts and MCP clients share one gate.
        from ...core import mutation
        try:
            mutation.create_pin(self._uuid, object_name, vertex_group_name)
        except mutation.MutationError as e:
            raise ValueError(str(e))
        return _Pin(self._uuid, object_name, vertex_group_name)

    @blender_api
    def get_pins(self) -> list[_Pin]:
        """Return all pins in this group as :class:`Pin` proxies.

        Example::

            for pin in group.get_pins():
                pin.clear_keyframes()
        """
        from ...models.groups import decode_vertex_group_identifier

        group = self._get_group()
        result = []
        for pin in group.pin_vertex_groups:
            obj_name, vg_name = decode_vertex_group_identifier(pin.name)
            if obj_name and vg_name:
                result.append(_Pin(self._uuid, obj_name, vg_name))
        return result

    # -- Keyframe animation (convenience, delegates to each Pin) ---------------

    @blender_api
    def clear_keyframes(self) -> "_Group":
        """Delete all keyframes for all pins in this group.

        Convenience method that calls :meth:`Pin.clear_keyframes` on every
        pin returned by :meth:`get_pins`.

        Returns:
            ``self`` for chaining.

        Example::

            group.clear_keyframes()
        """
        for pin in self.get_pins():
            pin.clear_keyframes()
        return self

    # -- Lifecycle -----------------------------------------------------------

    @blender_api
    def delete(self) -> None:
        """Delete this group and every pin it owns.

        Example::

            group.delete()
        """
        bpy.ops.zozo_contact_solver.delete_group(group_uuid=self._uuid)

    def __repr__(self):
        try:
            group = self._get_group()
            name = group.name or "unnamed"
            return f"Group({name!r}, uuid={self._uuid!r})"
        except ValueError:
            return f"Group(deleted, uuid={self._uuid!r})"
