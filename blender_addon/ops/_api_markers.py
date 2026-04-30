"""Markers for the public Blender Python API.

The ``docs/generate_blender_api_reference.py`` generator looks for these
decorator names *literally* in the AST of ``api.py``.  Import aliases are
rejected by the generator so the source of truth is always the bare name.

Both decorators are no-ops at runtime — they exist only to mark intent.
"""


def blender_api(obj):
    """Tag a class or function as part of the public Blender Python API."""
    return obj


def blender_api_hide(obj):
    """Opt a method out of documentation even when its owning class is marked."""
    return obj
