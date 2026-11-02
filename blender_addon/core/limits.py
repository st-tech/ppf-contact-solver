# File: limits.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
"""Limits the solver enforces, mirrored where the add-on needs them before a
connection exists.

The add-on never imports the frontend, so it cannot read these off the solver's
own registry when it registers its properties. Each value here is a COPY of one
defined in ``crates/ppf-cts-core/src/datamodel/params.rs``, and
``addon_host_tests`` hold the two equal, so the copy cannot drift.
"""

# The largest step size (seconds) the solver accepts, `MAX_STEP_SIZE` in the
# core's parameter registry, which says why the limit exists.
MAX_STEP_SIZE = 0.02


def step_size_refusal(dt, where="The Step Size"):
    """The sentence a refused step size is reported with, or None when it is
    within the limit. A NaN is refused as well."""
    if dt > 0.0 and dt <= MAX_STEP_SIZE:
        return None
    return (
        f"{where} is {dt:g} s; the solver accepts at most {MAX_STEP_SIZE:g} s "
        f"and refuses a larger step rather than taking it"
    )
