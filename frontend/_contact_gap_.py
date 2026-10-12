# File: _contact_gap_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""The smallest contact gap a step size can solve, and the refusal below it.

The solver gives an engaged contact the dynamic stiffness ``mass / gap^2`` and
gives the same vertex the inertial stiffness ``mass / step^2``, both in single
precision, where a sum holds the smaller of two terms only while their ratio
stays under ``2^24``. A pair's gap once engaged is below the gap it engages
at, so when ``(step / gap)^2`` is past ``2^24`` at the ENGAGING gap, every
engaged state of that pair has a contact stiffness the inertia cannot be added
to. Measured runs on that side of the bound end at their first frame with
``cg failed``, and the bound moves with the step as the ratio says it should.

The engaging gap is the mean of the two participants' contact gaps for two
meshes, and an analytic collider's own gap for a vertex against a wall or a
sphere. The check takes the smallest one a scene can form and compares it with
``step / 2^12``. Both remedies are real: a larger gap, or a smaller step.
"""

from dataclasses import dataclass

# (step / gap)^2 at which single precision stops holding mass / step^2 beside
# mass / gap^2: the reciprocal of its unit round-off.
_PRECISION_RANGE = 2.0**24


@dataclass(frozen=True)
class GapParticipant:
    """One thing a contact pair can be formed with, and the gap it brings.

    Attributes:
        label: The name a refusal gives it.
        gap: Its smallest contact gap over the run, in solver units.
        role: ``"solved"`` for a deformable object, ``"rigid"`` for a PDRD
            body, ``"collider"`` for a static collision mesh, ``"analytic"``
            for an invisible wall or sphere.
        own_pairs: Whether it can be in contact with itself. False for a
            rigid body and a collider, which have no pair of their own, and
            for an object whose self-intersections are allowed.
        cross_pairs: Whether it can be in contact with another participant.
            False for an object an allowance takes out of contact with the
            others, since which others is the allowance rule's to say.
        limit: What holds ``gap`` when the object's contact gap limit does
            and its authored ``contact-gap`` does not, worded so a refusal
            can say the object "is held to" it
            (``frontend/_contact_gap_limit_.py``). Empty otherwise.
    """

    label: str
    gap: float
    role: str
    own_pairs: bool = False
    cross_pairs: bool = True
    limit: str = ""


def smallest_solvable_gap(step: float) -> float:
    """The contact gap below which a step of length ``step`` cannot be solved."""
    return step / _PRECISION_RANGE**0.5


def _smallest_pair(participants: list[GapParticipant]):
    """The pair with the smallest engaging gap, as ``(gap, participants)``."""
    moving = sorted(
        (p for p in participants if p.role in ("solved", "rigid")),
        key=lambda p: p.gap,
    )
    crossing = [p for p in moving if p.cross_pairs]
    candidates = []
    for p in moving:
        if p.own_pairs:
            candidates.append((p.gap, (p,)))
            break
    if len(crossing) >= 2:
        a, b = crossing[0], crossing[1]
        candidates.append((0.5 * (a.gap + b.gap), (a, b)))
    if crossing:
        meshes = [p for p in participants if p.role == "collider"]
        if meshes:
            c = min(meshes, key=lambda p: p.gap)
            a = crossing[0]
            candidates.append((0.5 * (a.gap + c.gap), (a, c)))
    if moving:
        analytic = [p for p in participants if p.role == "analytic"]
        if analytic:
            c = min(analytic, key=lambda p: p.gap)
            candidates.append((c.gap, (c,)))
    return min(candidates, key=lambda c: c[0]) if candidates else None


def refuse_unsolvable_gaps(participants: list[GapParticipant], step: float) -> None:
    """Refuse a scene whose smallest contact gap a step cannot solve.

    Args:
        participants: Everything in the scene a contact pair can be formed
            with.
        step: The longest step the solver takes, in seconds: ``dt`` capped at
            one frame and scaled by ``playback``.

    Raises:
        ValueError: The smallest engaging gap is below ``step / 2^12``. The
            message names the participants, both numbers, and both remedies.
            Where a contact gap limit set a participant's gap, a larger
            ``contact-gap`` is not a remedy, and the message names what the
            limit measured in its place.
    """
    pair = _smallest_pair(participants)
    if pair is None:
        return
    gap, members = pair
    bound = smallest_solvable_gap(step)
    if gap >= bound:
        return
    if len(members) == 2:
        where = f"between {members[0].label!r} and {members[1].label!r}"
    else:
        where = f"of {members[0].label!r}"
    shorter_step = f"lower dt to at most {gap * _PRECISION_RANGE**0.5:.3e}"
    held = [p for p in members if p.limit]
    if held:
        holds = " and ".join(f"{p.label!r} is held to {p.limit}" for p in held)
        remedy = (
            f"{holds} by its contact gap limit, which a larger contact gap "
            f"does not lift: {shorter_step}, coarsen the mesh there, or "
            "reduce the contact offset."
        )
    else:
        remedy = (
            f"Raise the contact gap to at least {bound:.3e}, or "
            f"{shorter_step}."
        )
    raise ValueError(
        f"The contact gap {where} is {gap:.3e}, below the {bound:.3e} that a "
        f"step of {step:g} s can solve. A contact is as stiff as mass / gap^2 "
        "and the body as mass / step^2, added in single precision, and past a "
        f"ratio of 2^24 the linear solve does not converge. {remedy}"
    )


def longest_step(param) -> float:
    """The longest step the solver takes under ``param``, in seconds.

    The solver steps by ``dt``, never by more than one frame, and scales the
    result by ``playback``. Any of the three can be scheduled to change during
    a run, so the largest ``dt`` and ``playback`` and the smallest ``fps`` are
    the ones that count.

    Args:
        param: The session's ``ParamManager``.
    """
    dt = max(float(v) for v in param.values("dt"))
    fps = min(float(v) for v in param.values("fps"))
    playback = max(float(v) for v in param.values("playback"))
    return min(dt, 1.0 / fps) * playback


def refuse_unsolvable_scene(scene, param) -> None:
    """Refuse ``scene`` under ``param`` when its smallest gap cannot be solved.

    Args:
        scene: A ``FixedScene``.
        param: The session's ``ParamManager``.

    Raises:
        ValueError: See :func:`refuse_unsolvable_gaps`. Nothing is checked when
            ``disable-contact`` is set, since no pair is formed.
    """
    if param.get("disable-contact"):
        return
    refuse_unsolvable_gaps(scene.contact_gap_participants, longest_step(param))
