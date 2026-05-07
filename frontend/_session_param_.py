# File: _session_param_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""ParamManager: simulation parameter holder used by Session.

Split out of ``_session_.py``. The Rust mirror handles validation; this
class keeps the dynamic-parameter time cursor + dyn_param dict.
"""

import copy
from typing import Any, Optional

from . import _rust  # type: ignore[attr-defined]

from ._param_ import ParamHolder, app_param
from ._utils_ import Utils


class ParamManager:
    """Class to manage simulation parameters.

    Example:
        Configure the standard simulation parameters for a session before
        building it::

            session = app.session.create(scene)
            (
                session.param.set("frames", 120)
                             .set("dt", 0.01)
                             .set("fps", 30)
            )
            session = session.build()
    """

    def __init__(self):
        """Initialize the ParamManager."""
        self._key = None
        self._param = ParamHolder(app_param())
        self._default_param = self._param.copy()
        self._time = 0.0
        self._dyn_param = {}

    def copy(self) -> "ParamManager":
        """Copy the ParamManager object.

        Returns:
            ParamManager: The copied ParamManager object.

        Example:
            Snapshot the current parameters before tweaking one of them::

                baseline = session.param.copy()
                session.param.set("dt", 0.005)
        """
        return copy.deepcopy(self)

    def set(self, key: str, value: Optional[Any] = None) -> "ParamManager":
        """Set a parameter value.

        If ``value`` is ``None``, the parameter is set to ``True``.

        Args:
            key (str): The parameter key. Must not contain an underscore;
                use ``-`` instead.
            value (Any, optional): The parameter value. Defaults to ``None``
                (interpreted as ``True``).

        Returns:
            ParamManager: The updated ParamManager object.

        Raises:
            ValueError: If ``key`` contains an underscore or does not exist.

        Example:
            Chain several ``.set`` calls to configure a session. Keys use
            hyphens, never underscores::

                (
                    session.param.set("frames", 250)
                                 .set("dt", 0.01)
                                 .set("fps", 30)
                                 .set("min-newton-steps", 32)
                                 .set("gravity", [0, 0, 0])
                )
        """
        _rust.scene_validate_param_key_no_underscore(key)
        if value is None:
            value = True
        # `_param.set` itself raises on missing key, so no separate
        # precheck is needed.
        self._param.set(key, value)
        return self

    def clear_all(self):
        """Clear all parameters to their default values.

        Example:
            Reset every parameter back to its default before configuring a
            new run::

                session.param.clear_all()
                session.param.set("frames", 60).set("dt", 0.01)
        """
        self._param = self._default_param.copy()
        self._dyn_param = {}

    def clear(self, key: str) -> "ParamManager":
        """Reset a parameter to its default value and drop any dynamic entries for it.

        Args:
            key (str): The parameter key.

        Returns:
            ParamManager: The updated ParamManager object.

        Example:
            Revert a single parameter back to its default after trying an
            override::

                session.param.set("dt", 0.001)
                session.param.clear("dt")
        """
        self._param.set(key, self._default_param.get(key))
        if key in self._dyn_param:
            del self._dyn_param[key]
        return self

    def dyn(self, key: str) -> "ParamManager":
        """Select the current dynamic parameter key and reset the internal time cursor.

        Args:
            key (str): The dynamic parameter key.

        Returns:
            ParamManager: The updated ParamManager object.

        Raises:
            ValueError: If ``key`` does not exist.

        Example:
            Flip gravity between t=1s and t=2s, then restore it::

                g = session.param.get("gravity")
                (session.param.dyn("gravity")
                              .time(1.0).hold()
                              .time(1.5).change([-x for x in g])
                              .time(2.0).change(g))
        """
        _rust.scene_validate_param_key_exists(key in self._param.key_list(), key)
        self._time = 0.0
        self._key = key
        return self

    def change(self, value: Any) -> "ParamManager":
        """Change the value of the dynamic parameter at the current time.

        Args:
            value (Any): The new value of the dynamic parameter. May be a
                scalar, bool, or list/tuple of floats, depending on the key.

        Returns:
            ParamManager: The updated ParamManager object.

        Raises:
            ValueError: If no dynamic key is currently selected.

        Example:
            Slow playback to 10% after the third second via a dynamic key::

                (
                    session.param.dyn("playback")
                                 .time(2.99).hold()
                                 .time(3.0).change(0.1)
                )
        """
        if self._key is None:
            raise ValueError("Key is not set")
        else:
            if self._key in self._dyn_param:
                self._dyn_param[self._key].append((self._time, value))
            else:
                initial_val = self._param.get(self._key)
                self._dyn_param[self._key] = [
                    (0.0, initial_val),
                    (self._time, value),
                ]
            return self

    def hold(self) -> "ParamManager":
        """Hold the current value of the dynamic parameter at the current time.

        Returns:
            ParamManager: The updated ParamManager object.

        Raises:
            ValueError: If no dynamic key is currently selected.

        Example:
            Keep playback steady until t=2.99s, then drop it at t=3.0s::

                (
                    session.param.dyn("playback")
                                 .time(2.99).hold()
                                 .time(3.0).change(0.1)
                )
        """
        if self._key is None:
            raise ValueError("Key is not set")
        else:
            if self._key in self._dyn_param:
                last_val = self._dyn_param[self._key][-1][1]
                self.change(last_val)
            else:
                val = self._param.get(self._key)
                self.change(val)
        return self

    def export(self, path: str):
        """Export the parameters to ``param.toml`` (and ``dyn_param.txt`` when present).

        In fast-check mode, ``frames`` is forced to ``1``.

        Args:
            path (str): The path to the export directory.

        Example:
            Write the parameters alongside a session directory for inspection
            or external launching::

                session.param.export(fixed_session.info.path)
        """
        _rust.param_export_to_disk(
            path,
            self._param,
            self._dyn_param,
            Utils.is_fast_check(),
        )

    def time(self, time: float) -> "ParamManager":
        """Advance the current time cursor for the dynamic parameter.

        Args:
            time (float): The new current time. Must be strictly greater than
                the previous value.

        Returns:
            ParamManager: The updated ParamManager object.

        Raises:
            ValueError: If ``time`` is not strictly increasing.

        Example:
            Advance the cursor between two dynamic-value updates::

                (
                    session.param.dyn("playback")
                                 .time(1.0).hold()
                                 .time(2.0).change(0.5)
                )
        """
        _rust.scene_validate_param_time_strictly_increasing(self._time, time)
        self._time = time
        return self

    def get(self, key: Optional[str] = None) -> Any:
        """Get the value of a parameter.

        Args:
            key (Optional[str]): The parameter key. Must be specified.

        Returns:
            Any: The value of the parameter.

        Raises:
            ValueError: If ``key`` is ``None``.

        Example:
            Read the current gravity vector so a dynamic override can flip
            its sign later::

                g = session.param.get("gravity")
                print(g)
        """
        if key is None:
            raise ValueError("Key must be specified")
        else:
            return self._param.get(key)

    def items(self):
        """Get all parameter items.

        Returns:
            ItemsView: The parameter items.

        Example:
            Inspect every parameter currently configured on the session::

                for key, value in session.param.items():
                    print(f"{key} = {value}")
        """
        return self._param.items()
