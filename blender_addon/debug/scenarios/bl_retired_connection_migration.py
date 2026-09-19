# File: scenarios/bl_retired_connection_migration.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# What a saved artifact naming the RETIRED Local connection opens as.
#
# WHY THIS NEEDS A SCENARIO OF ITS OWN. Blender stores an EnumProperty as its
# ITEM NUMBER, so retiring an item does not make files that hold it invalid: it
# makes them read the field's DEFAULT, silently. A .blend saved while connected
# to a solver on this machine therefore opens pointed at a host it was never
# given, with the directory it was pointed at gone from the file, and nothing
# reports it. The migration in core.migrate_renames is the only thing standing
# between the artist and that, and it reads the RAW stored number rather than
# the field, which is the one way to see a value whose item no longer exists.
# A unit test cannot reach that: the raw ID-property only exists on a real
# PropertyGroup, so the check has to run inside Blender.
#
# A PROFILE IS THE SAME QUESTION IN A FILE THE USER WROTE. Profiles are TOML the
# artist keeps and re-applies, so a retired type name and a retired key must
# land on today's fields rather than be refused.
#
# Subtests:
#   A. retired_id_moves_to_platform_native: a scene carrying the retired item
#      number opens on THIS platform's native connection.
#   B. retired_path_carries_over: the directory it was pointed at lands on that
#      connection's path field, and the legacy key is dropped once it has.
#   C. existing_native_path_wins: a scene that also carries a native path keeps
#      it, and the legacy key is LEFT IN PLACE, so the value is still
#      recoverable from the file rather than discarded.
#   D. current_scene_untouched: a scene saved by a current build reports nothing
#      and is not written to.
#   E. retired_identifier_unassignable: the retired identifier cannot be set on
#      the field any more, so no new file can be written carrying it.
#   F. profile_retired_type_applies: a profile naming the retired type applies
#      as this platform's native rather than being refused.
#   G. profile_retired_key_applies: the retired path key lands on the native
#      path field.
#   H. profile_explicit_key_wins: a profile carrying BOTH the retired key and
#      today's key keeps today's.
#
# It touches only scene properties and an in-memory dict, so it needs no server,
# no solver and no GPU, and runs on every platform.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True
# Pure saved-state migration; no solve, so the backend does not enter into it.
BACKENDS = ("real",)


_DRIVER_BODY = r'''
import sys
import traceback

result.setdefault("errors", [])
result.setdefault("checks", {})


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


try:
    migrate = __import__(pkg + ".core.migrate_renames",
                         fromlist=["migrate_retired_connection"])
    profile_mod = __import__(pkg + ".core.profile", fromlist=["apply_profile"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])

    # The connection Local became here, and the field that takes its directory.
    # Read from the module rather than restated, so the scenario cannot drift
    # from the mapping it is checking.
    NATIVE, PATH_FIELD = migrate._PLATFORM_NATIVE.get(
        sys.platform, migrate._DEFAULT_NATIVE)
    RETIRED_ID = migrate._RETIRED_LOCAL_ID
    RETIRED_KEY = migrate._RETIRED_LOCAL_PATH_KEY

    root = groups.get_addon_data(bpy.context.scene)
    root.state.project_name = "retired_connection_migration"
    props = root.ssh_state

    def reset():
        """Leave the group holding no stored connection at all.

        Deleting the raw keys is what makes the next case start from a file
        that never carried them, which `in props.keys()` is the only way to
        express: assigning a default would leave a stored value behind and the
        untouched case below would then be checking the wrong thing.
        """
        for key in ("server_type", RETIRED_KEY, PATH_FIELD):
            if key in props.keys():
                del props[key]

    # ---- A and B: the retired number, with a directory ----
    reset()
    props[RETIRED_KEY] = "/tmp/where-the-solver-was"
    props["server_type"] = RETIRED_ID
    moved = migrate.migrate_retired_connection(bpy.context.scene)
    record("retired_id_moves_to_platform_native",
           props.server_type == NATIVE,
           {"platform": sys.platform, "expected": NATIVE,
            "got": props.server_type, "moved": moved})
    record("retired_path_carries_over",
           getattr(props, PATH_FIELD, "") == "/tmp/where-the-solver-was"
           and RETIRED_KEY not in props.keys(),
           {"field": PATH_FIELD, "got": getattr(props, PATH_FIELD, ""),
            "legacy_key_still_present": RETIRED_KEY in props.keys()})

    # ---- C: a native path already set is the later answer of the two ----
    reset()
    props[RETIRED_KEY] = "/tmp/the-old-one"
    setattr(props, PATH_FIELD, "/tmp/the-one-set-since")
    props["server_type"] = RETIRED_ID
    migrate.migrate_retired_connection(bpy.context.scene)
    record("existing_native_path_wins",
           getattr(props, PATH_FIELD, "") == "/tmp/the-one-set-since"
           and props.server_type == NATIVE
           and props.get(RETIRED_KEY, "") == "/tmp/the-old-one",
           {"field": PATH_FIELD, "got": getattr(props, PATH_FIELD, ""),
            "legacy_kept": props.get(RETIRED_KEY, "")})

    # ---- D: a scene from a current build is not written to ----
    reset()
    props.server_type = "CUSTOM"
    said = migrate.migrate_retired_connection(bpy.context.scene)
    record("current_scene_untouched",
           said == "" and props.server_type == "CUSTOM",
           {"said": said, "server_type": props.server_type})

    # ---- E: the retired identifier is gone from the enum ----
    try:
        props.server_type = "LOCAL"
        assignable = True
    except TypeError:
        assignable = False
    record("retired_identifier_unassignable", assignable is False,
           {"assignable": assignable})

    # ---- F, G, H: the same question in a profile the user wrote ----
    reset()
    ok = profile_mod.apply_profile({"type": "Local"}, props)
    record("profile_retired_type_applies",
           ok is True and props.server_type == NATIVE,
           {"applied": ok, "got": props.server_type, "expected": NATIVE})

    reset()
    profile_mod.apply_profile(
        {"type": "Local", "local_path": "/tmp/from-the-profile"}, props)
    record("profile_retired_key_applies",
           getattr(props, PATH_FIELD, "") == "/tmp/from-the-profile",
           {"field": PATH_FIELD, "got": getattr(props, PATH_FIELD, "")})

    reset()
    profile_mod.apply_profile(
        {"type": "Local", "local_path": "/tmp/the-retired-key",
         PATH_FIELD: "/tmp/todays-key"}, props)
    record("profile_explicit_key_wins",
           getattr(props, PATH_FIELD, "") == "/tmp/todays-key",
           {"field": PATH_FIELD, "got": getattr(props, PATH_FIELD, "")})

    reset()

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
'''


def build_driver(ctx: r.ScenarioContext) -> str:
    """Return the Python source the bootstrap will exec inside Blender.

    No substitutions: the scenario reads the platform mapping out of the module
    it is checking and writes only to the scene's own property group.
    """
    return _DRIVER_BODY


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
