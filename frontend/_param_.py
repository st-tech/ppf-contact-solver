# File: _param_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from typing import Any

from . import _rust  # type: ignore[attr-defined]


# Default-parameter table accessors. ``app_param()`` returns the
# application-wide defaults; ``object_param(obj_type)`` returns the
# per-material defaults for ``"tri"``, ``"tet"``, or ``"rod"``. Each
# entry maps a parameter key to ``(default_value, display_name,
# description)``. The Rust definitions live in
# ``core::datamodel::params``.
app_param = _rust.app_param_dict
object_param = _rust.object_param_dict


class ParamHolder:
    def __init__(self, param: dict[str, tuple[Any, str, str]]):
        self._default_param = param.copy()
        self._rust = _rust.ParamHolder(param)

    def clear_all(self) -> "ParamHolder":
        self._rust.clear_all()
        return self

    def set(self, key: str, value: Any) -> "ParamHolder":
        # Rust raises KeyError on missing key with the matching
        # "Parameter '...' not found." message.
        self._rust.set(key, value)
        return self

    def get(self, key: str) -> Any:
        return self._rust.get(key)

    def get_desc(self, key: str) -> tuple[str, str]:
        return self._rust.get_desc(key)

    def key_list(self) -> list[str]:
        return self._rust.key_list()

    def items(self) -> list[tuple[str, Any]]:
        return self._rust.items()

    def copy(self) -> "ParamHolder":
        clone = ParamHolder.__new__(ParamHolder)
        clone._default_param = self._default_param.copy()
        clone._rust = self._rust.copy()
        return clone

    def __deepcopy__(self, memo: dict) -> "ParamHolder":
        return self.copy()

    def __getstate__(self) -> dict:
        # The Rust holder is not picklable. Reduce to the schema
        # (default tuples) plus current values; __setstate__ rebuilds
        # the Rust mirror from the schema and replays the values.
        return {
            "default_param": self._default_param,
            "values": dict(self._rust.items()),
        }

    def __setstate__(self, state: dict) -> None:
        self._default_param = state["default_param"]
        self._rust = _rust.ParamHolder(state["default_param"])
        for key, value in state["values"].items():
            self._rust.set(key, value)
