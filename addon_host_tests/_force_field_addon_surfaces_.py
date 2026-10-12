# File: addon_host_tests/_force_field_addon_surfaces_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The add-on surfaces that report force fields, on a plain interpreter:
#
#   * every MCP tool's inputSchema is closed (additionalProperties false)
#     exactly when the server refuses an argument the tool does not declare,
#     and every tool declares, and documents, every argument it takes; a tool
#     that needs a connection refuses such an argument before it asks for
#     one;
#   * the operators generated from the MCP tools (`solver.<tool>()` in the
#     Python API) hand the tool exactly the arguments the caller gave;
#   * the force field status module's pure parts: a Fluid Flow's size bound
#     in the solver's bricks, the sizes the panel prints, the label wrapping
#     that keeps a refusal's tail visible, and the metadata checks a Volume's
#     VDB header answers;
#   * the wire protocol's current version has a changelog entry.
#
# What reads real Blender data (the panel's rows, the MCP listing, the
# Python API's get_force_fields) runs in the Blender rig.

from __future__ import annotations

import ast
import inspect
import re
import sys
import types
from typing import Optional

import pytest

from conftest import ADDON_ROOT, _install_stub_bpy, load_addon_module


# --- MCP: closed input schemas -------------------------------------------------


@pytest.fixture(scope="module")
def decorators():
    return load_addon_module("mcp.decorators")


def _fixed(name: str, count: int = 1, flag: bool = False):
    """A tool with two declared arguments.

    Args:
        name: A name
        count: A count
        flag: A flag
    """


def _open(name: str, **extra):
    """A tool that takes arguments it does not name.

    Args:
        name: A name
    """


def test_a_fixed_signature_advertises_a_closed_schema(decorators):
    schema = decorators.generate_tool_schema(_fixed, "d", {})["inputSchema"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"name", "count", "flag"}
    assert schema["required"] == ["name"]


def test_a_kwargs_tool_stays_open_and_names_no_kwargs_property(decorators):
    schema = decorators.generate_tool_schema(_open, "d", {})["inputSchema"]
    assert "additionalProperties" not in schema
    assert set(schema["properties"]) == {"name"}


def test_the_call_refuses_what_the_schema_closes(decorators):
    with pytest.raises(decorators.ValidationError) as e:
        decorators.validate_and_convert_args(_fixed, {"name": "a", "spacing": 0.1})
    assert "takes no argument spacing" in str(e.value)
    assert "its arguments are: name, count, flag" in str(e.value)
    # A tool with **kwargs takes them, and hands them on.
    got = decorators.validate_and_convert_args(_open, {"name": "a", "other": 2})
    assert got == {"name": "a", "other": 2}


class _Communicator:
    """The one call a connection tool asks of ``core.client.communicator``."""

    def __init__(self):
        self.connected = False

    def is_connected(self):
        return self.connected


def _needs_server(name: str = ""):
    """A tool that needs a connection.

    Args:
        name: A name
    """
    return {"ran": name}


def test_a_connection_tool_refuses_an_argument_before_asking_for_a_server(
        decorators, monkeypatch):
    com = _Communicator()
    client = types.ModuleType("blender_addon.core.client")
    client.communicator = com
    monkeypatch.setitem(sys.modules, "blender_addon.core.client", client)
    tool = decorators.connection_handler(_needs_server)
    try:
        # The registry dispatches the guarded wrapper.
        assert decorators.get_handler_registry()["_needs_server"]["func"] is tool
        refused = tool({"name": "a", "spacing": 0.25})
        assert refused["status"] == "error"
        assert "takes no argument spacing" in refused["message"]
        assert "its arguments are: name" in refused["message"]
        # A well-formed call is told it needs a connection, then runs once
        # there is one.
        offline = tool({"name": "a"})
        assert offline["status"] == "error" and "Not connected" in offline["message"]
        com.connected = True
        assert tool({"name": "a"}) == {"status": "success", "ran": "a"}
        assert "takes no argument spacing" in tool({"spacing": 1})["message"]
    finally:
        decorators._handler_registry.pop("_needs_server", None)


# --- the operators generated from the MCP tools ---------------------------------


def _bridged_tool(name: str, collection: Optional[str] = None,
                  visualize: Optional[bool] = None, count: int = 2,
                  sizes: Optional[list[int]] = None):
    """A tool with a required, optional, defaulted and JSON argument."""


@pytest.fixture()
def bridge(monkeypatch):
    """``ops.zozo_contact_solver``'s class builder, with the operator base
    class and the property constructors it needs from Blender stood in.
    Returns ``build(handler)``: the operator class for a handler that
    records the arguments it receives in ``build.received``."""
    # The session's bpy stand-in, which `load_addon_module` installs too,
    # is extended here, so it has to exist first.
    _install_stub_bpy()
    props = types.ModuleType("bpy.props")
    for kind in ("Bool", "Float", "Int", "String"):
        setattr(props, f"{kind}Property", lambda _kind=kind, **kw: (_kind, kw))
    monkeypatch.setitem(sys.modules, "bpy.props", props)
    monkeypatch.setattr(sys.modules["bpy"].types, "Operator", object, raising=False)
    module = load_addon_module("ops.zozo_contact_solver")
    received = []

    def build(func):
        def wrapper(args):
            received.append(dict(args))
            return {"status": "success"}

        wrapper._original_func = func
        info = {"func": wrapper, "schema": {"description": "",
                                            "inputSchema": {"properties": {}}}}
        return module._build_operator_class(func.__name__, info)

    build.received = received
    return build


def _call(cls, **given):
    """Run ``cls.execute`` as Blender would after ``bpy.ops...(**given)``:
    the given properties are set, every other one reads its default."""
    defaults = {"name": "", "collection": "", "visualize": False, "count": 2, "sizes": ""}
    reports = []
    op = types.SimpleNamespace(
        properties=types.SimpleNamespace(is_property_set=lambda p: p in given),
        report=lambda kind, text: reports.append(text),
        **{**defaults, **given})
    return cls.execute(op, None), reports


def test_a_generated_operator_passes_only_the_arguments_given(bridge):
    cls = bridge(_bridged_tool)
    assert _call(cls, name="a", collection="Fields") == ({"FINISHED"}, [])
    # An Optional[bool] left out is not passed as False, and a defaulted
    # one keeps the handler's own default.
    assert bridge.received[-1] == {"name": "a", "collection": "Fields"}
    # An explicit empty string reaches the handler, which reads it as clear.
    _call(cls, name="a", collection="")
    assert bridge.received[-1] == {"name": "a", "collection": ""}
    _call(cls, name="a", visualize=False, count=0, sizes="[3, 4]")
    assert bridge.received[-1] == {"name": "a", "visualize": False, "count": 0,
                                   "sizes": [3, 4]}
    # A required argument left out is the handler's to refuse by name.
    _call(cls)
    assert bridge.received[-1] == {}
    # A list given as text that is not JSON is refused, not dropped.
    result, reports = _call(cls, name="a", sizes="")
    assert result == {"CANCELLED"} and "sizes" in reports[0]


def _decorated_handlers():
    """``(file, function node)`` for every function an MCP handler decorator
    registers, read from the source the way the decorator would see it."""
    found = []
    for path in sorted((ADDON_ROOT / "mcp").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
                if name.endswith("_handler"):
                    found.append((path, node))
                    break
    return found


def test_every_tool_schema_is_closed_and_complete(decorators):
    handlers = _decorated_handlers()
    # The add-on registers well over a hundred tools; a scan that finds a
    # handful has stopped seeing the decorators.
    assert len(handlers) > 100, len(handlers)
    open_ones, undocumented = [], []
    for path, node in handlers:
        args = node.args
        where = f"{path.relative_to(ADDON_ROOT)}:{node.lineno} {node.name}"
        if args.vararg or args.kwarg:
            open_ones.append(where)
        doc = types.SimpleNamespace(__doc__=ast.get_docstring(node))
        described = decorators.parse_docstring(doc)["parameters"]
        named = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs
                 if a.arg != "self"]
        missing = [a for a in named if a not in described]
        if missing:
            undocumented.append(f"{where}: {', '.join(missing)}")
    assert not open_ones, (
        "A tool taking *args or **kwargs advertises an open inputSchema while "
        f"every other tool's is closed: {open_ones}")
    assert not undocumented, (
        "A tool's inputSchema is its whole reference, so every argument it "
        f"takes needs an Args entry: {undocumented}")


# --- the force field status module ------------------------------------------------


@pytest.fixture(scope="module")
def status():
    return load_addon_module("ui.dynamics.force_field_status")


def test_a_frame_is_bounded_by_its_bricks(status):
    assert status.BRICK_BYTES == 8 * 8 * 8 * 4 * 4
    assert status.brick_count((8, 8, 8)) == 1
    assert status.brick_count((9, 8, 1)) == 2
    assert status.brick_count((64, 64, 128)) == 8 * 8 * 16
    # Every brick of the box: which ones a frame keeps (those holding a
    # non-zero value) is a property of its voxels, which the bound never
    # reads. A staggered grid and Apply Density's density both fill bricks
    # no active velocity voxel lies in, so no active voxel count bounds it.
    assert status.frame_bytes_bound((16, 16, 16)) == 8 * (status.TABLE_ENTRY_BYTES
                                                          + status.BRICK_BYTES)
    assert status.frame_bytes_bound((17, 17, 17)) == 27 * (status.TABLE_ENTRY_BYTES
                                                           + status.BRICK_BYTES)
    assert status.frame_bytes_bound((0, 0, 0)) == 0


def test_sizes_read_in_binary_units(status):
    assert status.format_size(512) == "512 B"
    assert status.format_size(1536) == "1.5 KB"
    assert status.format_size(34 * 1024 * 1024) == "34.0 MB"
    assert status.format_size(5 * 1024 ** 3) == "5.0 GB"


def test_wrapping_keeps_every_word_and_the_width(status):
    text = ("a Fluid Flow field with no Domain Object; pick a gas domain "
            "or a Volume")
    lines = status.wrap_lines(text, 24, len)
    assert all(len(line) <= 24 for line in lines)
    assert " ".join(lines) == text
    # The actionable tail survives, on its own row.
    assert lines[-1].endswith("Volume")


def test_wrapping_cuts_a_word_wider_than_the_line(status):
    path = "/a/very/long/path/to/a/velocity/file.vdb"
    lines = status.wrap_lines(f"cannot read {path}", 12, len)
    assert all(len(line) <= 12 for line in lines)
    assert "".join(lines[1:]) == path


def test_wrapping_a_language_without_spaces(status):
    text = "押すオブジェクトがありません"
    lines = status.wrap_lines(text, 5, len)
    assert "".join(lines) == text and all(len(line) <= 5 for line in lines)
    assert status.wrap_lines("", 10, len) == [""]


def _header(**grids):
    return {name: {"type": t, "class": "unknown", "bbox": bbox}
            for name, (t, bbox) in grids.items()}


def test_the_velocity_grids_follow_the_volumes_settings(status):
    vol = types.SimpleNamespace(velocity_grid="vel", velocity_x_grid="",
                                velocity_y_grid="", velocity_z_grid="")
    assert status.volume_velocity_grids(vol) == ["vel"]
    vol.velocity_x_grid, vol.velocity_y_grid = "vel.x", "vel.y"
    # Two of three components name nothing: the vector grid is read.
    assert status.volume_velocity_grids(vol) == ["vel"]
    vol.velocity_z_grid = "vel.z"
    assert status.volume_velocity_grids(vol) == ["vel.x", "vel.y", "vel.z"]
    vol.velocity_grid = vol.velocity_x_grid = ""
    assert status.volume_velocity_grids(vol) == []


def test_a_header_answers_what_transfer_would_refuse(status):
    def why(header, names, density=False):
        return status.header_problem(header, names, density, "V", "f.vdb")

    header = _header(vel=("vec3s", ((0, 0, 0), (9, 4, 1))),
                     density=("float", ((0, 0, 0), (9, 4, 1))))
    assert why(header, ["vel"], True) is None
    # Each reason is Transfer's own (`core.force_field.vdb_grid_problem`),
    # naming the Volume and its file.
    assert why(header, ["velocity"]) == (
        "its Volume 'V''s file f.vdb holds no grid named 'velocity' (it holds density, "
        "vel); set its Velocity Grid")
    assert why(header, [""]) == (
        "its Volume 'V' names no velocity grid (f.vdb holds density, vel); set Velocity "
        "Grid in its Volume properties")
    assert why(_header(vel=("float", None)), ["vel"]) == (
        "the velocity grid 'vel' in its Volume 'V''s file f.vdb is float, not vec3s or vec3d")
    assert why(_header(vel=("vec3s", None)), ["vel"], True) == (
        "Apply Density is on and its Volume 'V''s file f.vdb holds no density grid "
        "(it holds vel)")
    assert why(_header(vel=("vec3s", None), density=("vec3s", None)), ["vel"], True) == (
        "the density grid in its Volume 'V''s file f.vdb is vec3s, not float or double")
    # A grid Mantaflow wrote is in grid units whatever its type.
    cache = _header(vel=("vec3s", None))
    cache["vel"]["mantaflow"] = True
    assert "is a Mantaflow cache file" in why(cache, ["vel"])
    # Three component grids are scalars, and a vector among them is refused.
    comps = _header(**{c: ("float", ((0, 0, 0), (1, 1, 1))) for c in ("x", "y", "z")})
    assert why(comps, ["x", "y", "z"]) is None
    comps["y"]["type"] = "vec3s"
    assert why(comps, ["x", "y", "z"]) == (
        "the velocity grid 'y' in its Volume 'V''s file f.vdb is vec3s, not float or double")


def test_a_header_gives_the_box_of_the_velocity_grids(status):
    header = _header(vel=("vec3s", ((-3, 4, 7), (16, 13, 11))),
                     density=("float", ((-40, 4, 7), (16, 13, 11))))
    # The velocity grids' box, the one Transfer cuts into bricks; a density
    # grid reaching further is read only inside it.
    assert status.header_extent(header, ["vel"]) == (20, 10, 5)
    comps = _header(x=("float", ((0, 0, 0), (3, 3, 3))),
                    y=("float", ((-2, 0, 0), (3, 5, 3))),
                    z=("float", ((0, 0, 0), (3, 3, 3))))
    assert status.header_extent(comps, ["x", "y", "z"]) == (6, 6, 4)
    # A file that does not record its bounds gives no resolution rather
    # than a guess.
    assert status.header_extent(_header(vel=("vec3s", None)), ["vel"]) is None
    # An empty grid records an inverted box.
    empty = _header(vel=("vec3s", ((0, 0, 0), (-1, -1, -1))))
    assert status.header_extent(empty, ["vel"]) == (0, 0, 0)


# --- the wire protocol ---------------------------------------------------------


def test_the_protocol_version_has_a_changelog_entry():
    text = (ADDON_ROOT / "protocol_version.toml").read_text(encoding="utf-8")
    (version,) = re.findall(r'^protocol = "([0-9.]+)"$', text, flags=re.MULTILINE)
    assert re.search(rf"^#   {re.escape(version)}: \S", text, flags=re.MULTILINE), (
        f"protocol {version} has no changelog entry in protocol_version.toml")


def test_inspect_sees_no_variadic_tool_argument(decorators):
    # The decorator's own test for an open tool reads the signature, which
    # is what the schema and the call both rely on.
    assert decorators.accepts_undeclared_arguments(_open)
    assert not decorators.accepts_undeclared_arguments(_fixed)
    assert inspect.signature(_open).parameters["extra"].kind is inspect.Parameter.VAR_KEYWORD
