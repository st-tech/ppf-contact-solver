# File: scenarios/rig_launch_config.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Guard for how the rig LAUNCHES Blender, as opposed to what the addon
# does once it is up. Three of those decisions are invisible at run time
# and regress silently:
#
#   - the window stays small. A worker's Blender is never read back
#     pixel-wise, so an unbounded window costs a full-screen framebuffer
#     per worker and nothing else. The size a desktop hands out can be
#     very large (6720x3677 on an xrdp session), and multiplied by the
#     worker pool that is the whole cost.
#   - PPF_BLENDER_WINDOW rejects what it cannot parse, rather than
#     falling back to a default that the caller will then report as the
#     size in effect.
#   - a dead DISPLAY reads as dead. ensure_display() decides whether to
#     start its own Xvfb on that answer, and a false positive sends
#     Blender at a display that is not there, which fails inside GHOST
#     with nothing pointing at the cause.
#
# Server-only: this exercises the launcher's own helpers, so it must not
# need the Blender it configures.

from __future__ import annotations

import os

import blender_harness as bh

from . import _runner as r

# No solver and no Blender, so this holds on the real-GPU jobs too.
BACKENDS = ("emulated", "real")


def _check_window(violations: list[str]) -> None:
    saved = os.environ.get("PPF_BLENDER_WINDOW")
    try:
        os.environ.pop("PPF_BLENDER_WINDOW", None)
        default = bh.window_geometry()
        if default is None:
            violations.append(
                "window_geometry() defaults to None, so Blender sizes its "
                "own window and the framebuffer is whatever the display is"
            )
        else:
            _, _, w, h = default
            if w * h > 1920 * 1080:
                violations.append(
                    f"default rig window is {w}x{h}, larger than a 1080p "
                    f"framebuffer; the point of the default is to be small"
                )

        for raw, want in (
            ("1024x768", (0, 0, 1024, 768)),
            ("640,480", (0, 0, 640, 480)),
            ("10,20,640,480", (10, 20, 640, 480)),
        ):
            os.environ["PPF_BLENDER_WINDOW"] = raw
            got = bh.window_geometry()
            if got != want:
                violations.append(
                    f"PPF_BLENDER_WINDOW={raw!r} parsed to {got}, want {want}"
                )

        for raw in ("off", "0"):
            os.environ["PPF_BLENDER_WINDOW"] = raw
            if bh.window_geometry() is not None:
                violations.append(
                    f"PPF_BLENDER_WINDOW={raw!r} must opt out of geometry"
                )
            if bh.window_args():
                violations.append(
                    f"PPF_BLENDER_WINDOW={raw!r} must emit no window flags"
                )

        for raw in ("bogus", "12x", "1,2,3"):
            os.environ["PPF_BLENDER_WINDOW"] = raw
            try:
                bh.window_geometry()
            except ValueError:
                continue
            violations.append(
                f"PPF_BLENDER_WINDOW={raw!r} was accepted; an unparseable "
                f"override must raise, not be silently ignored"
            )
    finally:
        os.environ.pop("PPF_BLENDER_WINDOW", None)
        if saved is not None:
            os.environ["PPF_BLENDER_WINDOW"] = saved


def _check_window_flags(violations: list[str]) -> None:
    saved = os.environ.get("PPF_BLENDER_WINDOW")
    try:
        os.environ["PPF_BLENDER_WINDOW"] = "800x600"
        args = bh.window_args()
        if "--window-geometry" not in args:
            violations.append(f"window_args() lost --window-geometry: {args}")
        elif args[args.index("--window-geometry") + 1:][:4] != \
                ["0", "0", "800", "600"]:
            violations.append(f"window_args() geometry does not match: {args}")
        if "--no-window-focus" not in args:
            violations.append(
                f"window_args() must not let a worker steal focus: {args}"
            )
    finally:
        os.environ.pop("PPF_BLENDER_WINDOW", None)
        if saved is not None:
            os.environ["PPF_BLENDER_WINDOW"] = saved


def _check_display_probe(violations: list[str]) -> None:
    # :77 through :90 are not the range ensure_display() allocates from,
    # so whichever is free here is genuinely dead and must read that way.
    dead = None
    for index in range(77, 91):
        if not os.path.exists(f"/tmp/.X11-unix/X{index}"):
            dead = f":{index}"
            break
    if dead and bh._display_is_live(dead):
        violations.append(
            f"_display_is_live({dead}) is True with no server there; "
            f"ensure_display would skip starting its own Xvfb"
        )
    for garbage in ("", "not-a-display", ":abc"):
        if bh._display_is_live(garbage):
            violations.append(
                f"_display_is_live({garbage!r}) must be False"
            )


def run(ctx: r.ScenarioContext) -> dict:
    violations: list[str] = []
    _check_window(violations)
    _check_window_flags(violations)
    _check_display_probe(violations)

    if bh.find_blender() is None:
        violations.append(
            "find_blender() resolved nothing; the rig cannot launch Blender "
            "on this host"
        )
    return r.failed(violations) if violations else r.passed()
