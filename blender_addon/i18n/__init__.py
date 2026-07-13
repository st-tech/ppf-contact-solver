# File: i18n/__init__.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Centralized localization for the addon. One JSON catalog per locale lives
# in this directory, keyed by the English source string. Loading builds the
# {locale: {(context, msgid): msgstr}} dict Blender expects and hands it to
# bpy.app.translations from the addon's register().
#
# Translators edit ONLY the JSON files, so a language contribution stays
# language-only with no Python review. en.json is the master list of every
# translatable English string (an identity map); each locale file maps the
# same English keys to translations. An empty value means "not translated
# yet": the loader skips it, and Blender falls back to the English source
# (or to its own built-in translation for common words like "Delete").
#
# Adding a language is dropping in one <locale>.json file. No code change.

import json
import os

import bpy  # pyright: ignore

# Register each entry under both the default context and the operator
# context. Operator bl_label / bl_description are looked up by Blender under
# the "Operator" context, while panels, property names/descriptions, layout
# labels, and our runtime pgettext_iface() calls use the default "*" context.
# Registering under both guarantees a hit regardless of which surface the
# string came from, and costs only a second dict entry per message.
_CONTEXTS = ("*", bpy.app.translations.contexts.operator_default)

_HERE = os.path.dirname(__file__)
_MASTER = "en.json"       # source / key list, never registered as a translation
_META_KEY = "_meta"       # optional per-file metadata object, not a message
_registered = False


def _load_locale(path):
    """Read one locale JSON into {english: translation}, dropping the
    optional ``_meta`` object and any entry whose translation is empty."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for key, val in data.items():
        if key == _META_KEY:
            continue
        if isinstance(val, str) and val:
            out[key] = val
    return out


def _build_translations():
    """Assemble {locale: {(ctx, msgid): msgstr}} from every ``*.json`` in
    this directory except the ``en.json`` master."""
    translations = {}
    for fname in sorted(os.listdir(_HERE)):
        if not fname.endswith(".json") or fname == _MASTER:
            continue
        locale = fname[: -len(".json")]
        try:
            messages = _load_locale(os.path.join(_HERE, fname))
        except Exception as exc:
            # A malformed catalog must never break addon registration.
            print(f"[ppf-cts i18n] skipped {fname}: {exc}")
            continue
        if not messages:
            continue
        entries = {}
        for msgid, msgstr in messages.items():
            for ctx in _CONTEXTS:
                entries[(ctx, msgid)] = msgstr
        translations[locale] = entries
    return translations


def register():
    """Register the addon translation catalogs. Idempotent."""
    global _registered
    if _registered:
        return
    try:
        translations = _build_translations()
        if translations:
            bpy.app.translations.register(__package__, translations)
        _registered = True
    except Exception as exc:
        # Localization is advisory; a failure here must not stop the addon.
        print(f"[ppf-cts i18n] translation register failed: {exc}")


def unregister():
    global _registered
    if not _registered:
        return
    try:
        bpy.app.translations.unregister(__package__)
    except Exception:
        pass
    _registered = False
