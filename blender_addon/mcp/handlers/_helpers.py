"""Internal helpers shared across handler modules.

Kept out of the ``from .<module> import *`` re-exports in __init__.py so
nothing here leaks into the public handler namespace.
"""


def check_vec3_validation(name: str, v):
    """Coerce ``v`` to a length-3 float tuple, raising ValidationError."""
    from ..decorators import ValidationError
    from ...core.utils import check_vec3
    return check_vec3(name, v, ValidationError)
