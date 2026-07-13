# File: scenarios/bl_i18n_no_leaks.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression guard against untranslated UI "leaks": English strings that
# reach the panel but are not catalog keys, so bpy.app.translations can
# never translate them and they render as English regardless of language.
#
# The motivating case was "Status: Disconnected". The panel builds the
# label as pgettext_iface("Status: {status}").format(status=status.value),
# where status.value is a RemoteStatus enum string. The "Status:" template
# was a catalog key, but the 23 RemoteStatus values were not, so the value
# always leaked as English. The fix wraps the value with iface_() AND adds
# every RemoteStatus value to i18n/en.json plus each locale catalog.
#
# This scenario freezes that contract so a future status (or any other
# UI-displayed enum value added to the driver's ENUM_SOURCES) cannot ship
# without a catalog key:
#
#   A. status_values_are_catalog_keys: every RemoteStatus value exists in
#      i18n/en.json. A missing value is a guaranteed leak.
#   B. status_values_translate_at_runtime: with the UI language set to a
#      locale that has the translation, bpy.app.translations resolves each
#      status value to a NON-English string (proves the registered path, not
#      just catalog membership). A value that still equals its English source
#      is the leak we guard against. We do not require an exact match with our
#      catalog value: Blender's own built-in catalog may translate a common
#      word (e.g. "Resumable") with a different term, which is still a valid
#      translation, not a leak. Untranslated entries (empty catalog value) are
#      skipped, since an empty value is a valid "falls back to English" state.
#
# Pure introspection scenario: no server, no solver, no transfer.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, os, json, time, traceback
result.setdefault("checks", {})
result.setdefault("errors", [])


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


# UI-displayed value enums: (submodule, class name). Their .value strings
# are shown in panels, so each must be a catalog key. Add new user-facing
# value enums here as they appear.
ENUM_SOURCES = [
    ("core.status", "RemoteStatus"),
]

# Locales to exercise for the runtime-resolution check. Any locale that
# ships full translations of the enum values works here.
RESOLVE_LOCALE = "ja_JP"


try:
    i18n_mod = __import__(pkg + ".i18n", fromlist=["register"])
    i18n_dir = os.path.dirname(i18n_mod.__file__)
    with open(os.path.join(i18n_dir, "en.json"), encoding="utf-8") as f:
        en = json.load(f)
    en_keys = set(k for k in en if k != "_meta")

    # Gather every UI-displayed enum value.
    values = []
    for submod, clsname in ENUM_SOURCES:
        mod = __import__(pkg + "." + submod, fromlist=[clsname])
        cls = getattr(mod, clsname)
        values.extend(member.value for member in cls)

    # ----- A: every displayed value is a catalog key (leak guard) -----
    missing = sorted(v for v in values if v not in en_keys)
    record(
        "A_status_values_are_catalog_keys",
        not missing,
        {"checked": len(values), "missing": missing},
    )

    # ----- B: runtime translation resolves for a filled locale --------
    locale_path = os.path.join(i18n_dir, RESOLVE_LOCALE + ".json")
    view = bpy.context.preferences.view
    saved = (view.language, view.use_translate_interface)
    ok_b = True
    detail_b = {"locale": RESOLVE_LOCALE, "resolved": 0, "skipped": 0, "leaked": []}
    try:
        with open(locale_path, encoding="utf-8") as f:
            loc = json.load(f)
        view.use_translate_interface = True
        view.language = RESOLVE_LOCALE
        for v in values:
            if not loc.get(v, ""):
                detail_b["skipped"] += 1
                continue  # empty translation is a valid English fallback
            got = bpy.app.translations.pgettext_iface(v)
            # A leak is when the value renders as its English self. Any
            # non-English result (ours or Blender's built-in) is a pass.
            if got != v:
                detail_b["resolved"] += 1
            else:
                ok_b = False
                detail_b["leaked"].append(v)
    finally:
        view.language, view.use_translate_interface = saved
    record("B_status_values_translate_at_runtime", ok_b, detail_b)

except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
