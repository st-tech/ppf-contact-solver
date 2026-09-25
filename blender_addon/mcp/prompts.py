# File: mcp/prompts.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Prompt templates served over MCP.
#
# A prompt here is a short, user-selected starting point for a workflow the
# add-on supports end to end. Each one names the ordered steps and leaves the
# detail to the tools: every tool's description and input schema come from the
# handler that implements it, so the prompt carries no second copy of them
# that could drift.
#
# The `prompts` capability is only advertised because these exist: a server
# that declares the capability must answer `prompts/list` and `prompts/get`.

from __future__ import annotations

from typing import Any

from .protocol import INVALID_PARAMS, ProtocolError

# Each entry: the advertised descriptor, plus a body builder taking the
# validated arguments. `arguments` follows the PromptArgument schema, and the
# `title` is the human-readable label a client shows in a prompt picker.
_PROMPTS: dict[str, dict[str, Any]] = {
    "run_simulation": {
        "title": "Run a simulation end to end",
        "description": (
            "Take a Blender scene from an unconfigured state to fetched "
            "simulation results, in the order the add-on requires."
        ),
        "arguments": [
            {
                "name": "backend",
                "description": "Where the solver runs: local, ssh, docker, or windows_native.",
                "required": False,
            },
            {
                "name": "frames",
                "description": "How many frames to solve.",
                "required": False,
            },
        ],
    },
    "pin_and_constrain": {
        "title": "Pin and constrain an object",
        "description": (
            "Hold part of a mesh in place, or drive it along a path, using "
            "the add-on's two pin kinds and its collider and merge tools."
        ),
        "arguments": [
            {
                "name": "object_name",
                "description": "The Blender object to constrain.",
                "required": True,
            },
            {
                "name": "intent",
                "description": (
                    "What the constraint is for: hold, drive, stitch, or collide."
                ),
                "required": False,
            },
        ],
    },
    "tune_parameters": {
        "title": "Tune solver parameters",
        "description": (
            "Choose scene and material parameters for a stated goal, and "
            "read back what the solver actually received."
        ),
        "arguments": [
            {
                "name": "goal",
                "description": (
                    "What to change, for example 'softer cloth', 'faster "
                    "solve', or 'less stretch'."
                ),
                "required": True,
            },
        ],
    },
    "diagnose_failure": {
        "title": "Diagnose a failed or stalled solve",
        "description": (
            "Work through a solver failure from its reported symptom to the "
            "authoring or parameter cause."
        ),
        "arguments": [
            {
                "name": "symptom",
                "description": (
                    "What was observed: a crash kind, a hang, zero frames "
                    "written, or a visibly wrong result."
                ),
                "required": True,
            },
        ],
    },
}


def _text(role: str, body: str) -> dict[str, Any]:
    return {"role": role, "content": {"type": "text", "text": body}}


def _run_simulation(args: dict[str, str]) -> list[dict[str, Any]]:
    backend = args.get("backend", "local")
    frames = args.get("frames", "the scene's configured frame range")
    return [
        _text(
            "user",
            f"Set up and run a simulation in this Blender scene on the "
            f"{backend} backend, solving {frames}.\n\n"
            "Follow the add-on's required order and stop at the first step "
            "that reports an error rather than continuing:\n"
            "1. Establish a connection for the chosen backend and confirm it "
            "reports connected.\n"
            "2. Create an object group and assign the participating objects "
            "to it, setting each object's type.\n"
            "3. Apply constraints: pins, invisible colliders, merge pairs.\n"
            "4. Set scene parameters, then per-group material parameters.\n"
            "5. Build the scene, start the solve, and poll its status.\n"
            "6. Fetch the results back into Blender.\n\n"
            "Read each tool's description and input schema before calling "
            "it: they state its preconditions, units and refusals. Check the "
            "mesh resolution with the edge-length and bounding-box tools "
            "before building.",
        )
    ]


def _pin_and_constrain(args: dict[str, str]) -> list[dict[str, Any]]:
    intent = args.get("intent", "hold")
    return [
        _text(
            "user",
            f"Constrain the object '{args['object_name']}' with intent: "
            f"{intent}.\n\n"
            "The add-on has exactly two pin kinds. A pull pin is a soft "
            "spring and is the only compliant hold; a fix pin is an exact "
            "boundary condition whose degree of freedom is eliminated. There "
            "is no stiffness scalar to tune between them.\n\n"
            "Read the descriptions of the pin, invisible-collider and "
            "merge-pair tools before choosing one: they state which intent "
            "each serves. Confirm the vertex group a pin names already "
            "exists on the mesh before adding the pin.",
        )
    ]


def _tune_parameters(args: dict[str, str]) -> list[dict[str, Any]]:
    return [
        _text(
            "user",
            f"Adjust this scene's solver parameters for the goal: "
            f"{args['goal']}.\n\n"
            "Read the parameter tools' input schemas for units and ranges "
            "before choosing any value. Change one "
            "group of parameters at a time, then read the parameters back "
            "and report what the solver actually received, since a scene "
            "parameter and a per-group material parameter are set through "
            "different tools and are reported separately.",
        )
    ]


def _diagnose_failure(args: dict[str, str]) -> list[dict[str, Any]]:
    return [
        _text(
            "user",
            f"Diagnose this solver failure. Reported symptom: "
            f"{args['symptom']}.\n\n"
            "Establish the facts before proposing a cause: how many frames "
            "were written, what the solver's own output says, and whether "
            "the run reported a crash kind or simply stopped making "
            "progress.\n\n"
            "Read the solver status and console tools' output for what the "
            "run wrote and why it stopped. Several "
            "failures are authoring problems rather than solver problems, so "
            "check the scene's geometry and constraints before changing any "
            "parameter.",
        )
    ]


_BUILDERS = {
    "run_simulation": _run_simulation,
    "pin_and_constrain": _pin_and_constrain,
    "tune_parameters": _tune_parameters,
    "diagnose_failure": _diagnose_failure,
}


def list_prompts() -> list[dict[str, Any]]:
    """Every prompt this server offers, in a deterministic order."""
    return [
        {
            "name": name,
            "title": spec["title"],
            "description": spec["description"],
            "arguments": spec["arguments"],
        }
        for name, spec in sorted(_PROMPTS.items())
    ]


def get_prompt(name: Any, arguments: Any) -> dict[str, Any]:
    """Render the prompt *name* with *arguments*.

    Raises ProtocolError with INVALID_PARAMS when the name is unknown or a
    required argument is absent, so an unusable prompt fails at the call
    rather than producing a message with a hole in it.
    """
    if not isinstance(name, str) or name not in _PROMPTS:
        raise ProtocolError(INVALID_PARAMS, f"Unknown prompt: {name!r}")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ProtocolError(INVALID_PARAMS, "'arguments' must be an object")

    spec = _PROMPTS[name]
    supplied = {k: str(v) for k, v in arguments.items() if v is not None}
    missing = [
        arg["name"]
        for arg in spec["arguments"]
        if arg.get("required") and arg["name"] not in supplied
    ]
    if missing:
        raise ProtocolError(
            INVALID_PARAMS,
            f"Prompt {name!r} is missing required argument(s): {', '.join(missing)}",
        )

    return {
        "description": spec["description"],
        "messages": _BUILDERS[name](supplied),
    }
