# File: test_i18n.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression tests for the localization layer (i18n/). Runs inside Blender.
# Data-driven: it validates whatever catalogs ship in i18n/, so it does not
# hardcode any translation and stays correct as catalogs grow.
#
# Usage from a host shell (against the loaded addon):
#
#     echo 'import ppf_contact_solver.tests.test_i18n as t; print(t.run_all())' | \
#         python blender_addon/debug/main.py exec -
#
# Or self-contained against THIS working tree (no symlink dependency):
#
#     Blender --background --factory-startup --python-expr \
#       "import sys; sys.path.insert(0, '<repo>'); \
#        import blender_addon.tests.test_i18n as t; \
#        import pprint; pprint.pprint(t.run_all())"
#
# Each test function begins with `test_`. `run_all()` discovers them, runs
# each, captures exceptions, and returns a summary dict.

import json
import os
import traceback

import bpy  # pyright: ignore

from .. import i18n

_TEST_MODULE = "ppf_cts_i18n_test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _i18n_dir():
    return os.path.dirname(i18n.__file__)


def _read_json(name):
    with open(os.path.join(_i18n_dir(), name), encoding="utf-8") as f:
        return json.load(f)


def _locale_files():
    """Every <locale>.json in i18n/ except the en.json master."""
    return sorted(
        f for f in os.listdir(_i18n_dir())
        if f.endswith(".json") and f != "en.json"
    )


def _messages(data):
    """Drop the optional _meta object; keep string entries only."""
    return {k: v for k, v in data.items() if k != "_meta" and isinstance(v, str)}


class _LangGuard:
    """Save and restore the user's language / translate toggles so running
    this test in a live session never changes their UI language."""

    _ATTRS = (
        "language",
        "use_translate_interface",
        "use_translate_tooltips",
        "use_translate_new_dataname",
    )

    def __enter__(self):
        view = bpy.context.preferences.view
        self._saved = {a: getattr(view, a) for a in self._ATTRS}
        view.use_translate_interface = True
        view.use_translate_tooltips = True
        view.use_translate_new_dataname = True
        return self

    def __exit__(self, *exc):
        view = bpy.context.preferences.view
        for a, v in self._saved.items():
            try:
                setattr(view, a, v)
            except Exception:
                pass
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_en_master_present_and_valid():
    """en.json exists, parses, and is a flat {string: string} identity map."""
    data = _read_json("en.json")
    msgs = _messages(data)
    assert msgs, "en.json master has no entries"
    for k, v in msgs.items():
        assert isinstance(k, str) and k, f"bad key {k!r}"
        assert v == k, f"en.json is not an identity map: {k!r} -> {v!r}"
    return {"keys": len(msgs)}


def test_locale_files_parse_and_are_known():
    """Every locale catalog parses, its values are strings, and every key is
    present in the en.json master (no orphan keys that translators can't see)."""
    master = set(_messages(_read_json("en.json")))
    report = {}
    orphans = {}
    for fname in _locale_files():
        msgs = _messages(_read_json(fname))
        for k, v in msgs.items():
            assert isinstance(v, str), f"{fname}: {k!r} value is not a string"
        extra = sorted(set(msgs) - master)
        if extra:
            orphans[fname] = extra[:10]
        filled = sum(1 for v in msgs.values() if v)
        report[fname] = {"keys": len(msgs), "translated": filled}
    assert not orphans, f"locale keys missing from en.json master: {orphans}"
    return report


def test_build_translations_shape():
    """_build_translations() returns {locale: {(ctx, msgid): msgstr}} with
    non-empty msgstr values and the expected context registration."""
    built = i18n._build_translations()
    op_ctx = bpy.app.translations.contexts.operator_default
    for locale, entries in built.items():
        assert entries, f"{locale}: empty entry map"
        for (ctx, msgid), msgstr in entries.items():
            assert ctx in ("*", op_ctx), f"{locale}: unexpected context {ctx!r}"
            assert msgstr, f"{locale}: empty msgstr for {msgid!r}"
    return {locale: len(e) for locale, e in built.items()}


def test_translations_resolve_end_to_end():
    """Register the catalogs and confirm Blender returns each non-empty
    translation via pgettext_iface for a sample of real keys per locale."""
    built = i18n._build_translations()
    if not built:
        return {"skipped": "no translations in any catalog"}

    pgettext_iface = bpy.app.translations.pgettext_iface
    checked = {}
    try:
        bpy.app.translations.unregister(_TEST_MODULE)
    except Exception:
        pass
    bpy.app.translations.register(_TEST_MODULE, built)
    try:
        with _LangGuard():
            view = bpy.context.preferences.view
            lang_codes = {
                item.identifier
                for item in view.bl_rna.properties["language"].enum_items
            }
            for fname in _locale_files():
                locale = fname[: -len(".json")]
                if locale not in lang_codes:
                    checked[locale] = {"skipped": "locale not available in this build"}
                    continue
                msgs = {k: v for k, v in _messages(_read_json(fname)).items() if v}
                if not msgs:
                    checked[locale] = {"skipped": "no translations filled"}
                    continue
                view.language = locale
                sample = list(msgs.items())[:25]
                for msgid, expected in sample:
                    got = pgettext_iface(msgid)
                    assert got == expected, (
                        f"[{locale}] pgettext_iface({msgid!r}) -> {got!r}, "
                        f"expected {expected!r}"
                    )
                checked[locale] = {"sampled": len(sample), "translated": len(msgs)}
    finally:
        try:
            bpy.app.translations.unregister(_TEST_MODULE)
        except Exception:
            pass
    return checked


def test_english_fallback_for_untranslated():
    """A key that is absent from a locale catalog (or empty) must fall back
    to the English source string, never to an empty string."""
    built = i18n._build_translations()
    if not built:
        return {"skipped": "no translations"}
    pgettext_iface = bpy.app.translations.pgettext_iface
    sentinel = "Zzx Untranslated Sentinel String 8471"
    try:
        bpy.app.translations.unregister(_TEST_MODULE)
    except Exception:
        pass
    bpy.app.translations.register(_TEST_MODULE, built)
    try:
        with _LangGuard():
            view = bpy.context.preferences.view
            locale = next(iter(built))
            lang_codes = {
                item.identifier
                for item in view.bl_rna.properties["language"].enum_items
            }
            if locale not in lang_codes:
                return {"skipped": "locale not available"}
            view.language = locale
            got = pgettext_iface(sentinel)
            assert got == sentinel, f"fallback failed: {got!r}"
    finally:
        try:
            bpy.app.translations.unregister(_TEST_MODULE)
        except Exception:
            pass
    return {"ok": True}


# Enums whose .value strings are shown directly in the addon UI (e.g.
# interpolated into "Status: {status}"). Their values must exist as catalog
# keys, otherwise they render as untranslated English ("leaks"). When you add
# a new user-facing value enum, add it here so this test guards it too.
_UI_VALUE_ENUMS = [
    ("..core.status", "RemoteStatus"),
]


def test_no_untranslated_ui_value_leaks():
    """Every value of a UI-displayed enum must be a catalog key, so it can be
    translated instead of leaking as English (regression guard for the
    'Status: Disconnected' class of bug)."""
    import importlib

    en = set(_messages(_read_json("en.json")))
    leaks = {}
    checked = {}
    for modpath, clsname in _UI_VALUE_ENUMS:
        mod = importlib.import_module(modpath, package=__package__)
        cls = getattr(mod, clsname)
        values = [member.value for member in cls]
        missing = [v for v in values if v not in en]
        checked[clsname] = len(values)
        if missing:
            leaks[clsname] = missing
    assert not leaks, (
        "UI-displayed enum values missing from en.json (untranslated leaks). "
        "Add them to i18n/en.json and every locale catalog: " + repr(leaks)
    )
    return {"checked": checked, "leaks": 0}


def test_register_unregister_public_api():
    """The public register()/unregister() are callable and idempotent."""
    # Do not disturb an addon-owned registration if one is already active.
    if getattr(i18n, "_registered", False):
        return {"skipped": "addon already owns the registration"}
    i18n.register()
    i18n.register()  # idempotent
    i18n.unregister()
    i18n.unregister()  # idempotent
    return {"ok": True}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    results = {}
    passed = 0
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            results[name] = {"status": "PASS", "detail": fn()}
            passed += 1
        except Exception as exc:
            results[name] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            failed += 1
    results["_summary"] = {"passed": passed, "failed": failed}
    return results
