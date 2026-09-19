# File: tools/cpu_contact_fixture.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# THE PENETRATION GUARANTEE, MEASURED. A free cloth sheet is dropped onto a
# corner-pinned one, CROSSED at 45 degrees so their edges and triangles genuinely
# interleave, and every committed frame is tested for a crossing with a
# segment-triangle intersection against the ACTUAL surface meshes.
#
# THE SAME SCENE IS RUN TWICE, and the second run is what makes the first one
# evidence. With contact on, no edge of either sheet may pass through a triangle
# of the other and no upper vertex may end up underneath the lower sheet, on any
# frame. With `disable-contact` set, the upper sheet falls straight through and
# the second of those goes non-zero. Without that control a fixture reporting
# zero proves only that the geometry never came close enough to cross, which is
# exactly how a contact test passes on a backend that computes no contact at all.
#
# TWO INSTRUMENTS, BECAUSE ONE OF THEM CANNOT SEE A FAST TUNNEL. The frames are
# 1/60 s apart and the falling sheet covers more than the sheet spacing in one of
# them, so a sheet that passes through leaves NO interleaved pose in the output:
# the segment-triangle count stays at zero through the whole control run. What
# records it is the side each surface ended on, measured by casting a ray up and
# down from every upper vertex against the lower sheet's actual triangles.
#
# WHY CROSSED AND NOT PARALLEL. Two sheets sliding through each other with
# parallel edges pierce no edge-triangle at any pose, so a parallel fixture can
# report zero crossings while the two are entirely on the wrong sides of one
# another. Rotating the upper sheet 45 degrees about the vertical makes its edges
# cross the lower sheet's triangles the moment it passes through, which is what
# the control run below confirms rather than assumes.
#
# WHY SEGMENT-TRIANGLE AND NOT A HULL. A convex hull false-positives on any drape
# over a corner: the cloth is correctly outside the surface and inside the hull.
# The test below is against the triangles the solver was given.
#
# WHAT THIS IS ACTUALLY TESTING. Non-penetration is enforced structurally by the
# ACCD CCD-filtered line search plus `check_intersection`, NOT by the barrier:
# the barrier is a cubic energy, finite at the surface, so a large enough step
# walks straight through it. Neuter the line search and the upper sheet passes
# through on the step that reaches the lower one, and the crossing count goes
# non-zero. This fixture reads the OUTPUT rather than the solver's own report, so
# it still catches that even if the intersection gate is the half that is broken.

import math
import os

import numpy as np

from frontend import App

# --------------------------------------------------------------------------
# The two sheets.
#
# The lower one is pinned at its four corners with a ZERO MOVE. Both halves
# matter: pinning only the corners keeps the sheet a dynamic SHELL rather than
# promoting it to a rest-pose STATIC collider (`Object.update_static` does that
# to a fully pinned object with no pin operations, and a static collision mesh is
# a separate capability this backend still refuses), and the operation is what
# keeps the object dynamic at all.
# --------------------------------------------------------------------------
LOWER_HALF = 0.20
UPPER_HALF = 0.12
LOWER_Y = 1.0
DROP = 0.03
UPPER_Y = LOWER_Y + DROP
RESOLUTION = 5
UPPER_TWIST = math.radians(45.0)

FRAMES = 12
FPS = 60.0
GRAVITY = -9.8
DT = 0.5 / FPS
ELAPSED = FRAMES / FPS

# Authoring, not solver tolerances.
YOUNG_MODULUS = 2000.0
BEND = 20.0

# A sheet released from rest and never stopped falls this far, which is derived
# rather than measured and is several times the gap.
FREE_FALL = 0.5 * abs(GRAVITY) * ELAPSED * ELAPSED
assert FREE_FALL > 3.0 * DROP, (
    "the run is too short for free fall to clear the gap, so a sheet that was "
    "stopped is not distinguishable from one that never arrived"
)


def sheet(resolution: int, half: float, height: float, twist: float):
    """A square grid in the horizontal plane, rotated `twist` about the vertical."""
    axis = np.linspace(-half, half, resolution)
    cos_t, sin_t = math.cos(twist), math.sin(twist)
    points = np.array(
        [
            [cos_t * x - sin_t * z, height, sin_t * x + cos_t * z]
            for x in axis
            for z in axis
        ],
        dtype=np.float64,
    )
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            a = i * resolution + j
            b = a + 1
            c = a + resolution
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])
    return points, np.array(faces, dtype=np.uint32)


lower_vertices, lower_faces = sheet(RESOLUTION, LOWER_HALF, LOWER_Y, 0.0)
upper_vertices, upper_faces = sheet(RESOLUTION, UPPER_HALF, UPPER_Y, UPPER_TWIST)
LOWER_COUNT = int(lower_vertices.shape[0])
UPPER_COUNT = int(upper_vertices.shape[0])
LOWER_CORNERS = [
    0,
    RESOLUTION - 1,
    RESOLUTION * (RESOLUTION - 1),
    RESOLUTION * RESOLUTION - 1,
]


def edges_of(faces: np.ndarray) -> np.ndarray:
    """Every undirected edge of a triangle set, once."""
    pairs = set()
    for face in faces:
        for k in range(3):
            a, b = int(face[k]), int(face[(k + 1) % 3])
            pairs.add((min(a, b), max(a, b)))
    return np.array(sorted(pairs), dtype=np.int64)


LOWER_EDGES = edges_of(lower_faces)
UPPER_EDGES = edges_of(upper_faces)


def below_count(points, triangle_points, triangles) -> int:
    """How many of `points` sit UNDERNEATH the surface `triangles` describes.

    A ray is cast straight down and straight up from each point against the
    ACTUAL triangles, and a point that the surface is above rather than below is
    counted. This is the instrument that catches a crossing which happened
    BETWEEN two output frames: the sheets move further per frame than they are
    apart, so a tunnel leaves no interleaved pose to find, and only the side each
    surface ended on records it.
    """
    count = 0
    for origin in points:
        hit_down = False
        hit_up = False
        for triangle in triangles:
            v0 = triangle_points[triangle[0]]
            v1 = triangle_points[triangle[1]]
            v2 = triangle_points[triangle[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            direction = np.array([0.0, -1.0, 0.0])
            pvec = np.cross(direction, e2)
            det = float(np.dot(e1, pvec))
            if abs(det) < 1e-16:
                continue
            inv = 1.0 / det
            tvec = origin - v0
            u = float(np.dot(tvec, pvec)) * inv
            if u < 0.0 or u > 1.0:
                continue
            qvec = np.cross(tvec, e1)
            v = float(np.dot(direction, qvec)) * inv
            if v < 0.0 or u + v > 1.0:
                continue
            t = float(np.dot(e2, qvec)) * inv
            if t > 0.0:
                hit_down = True
            elif t < 0.0:
                hit_up = True
        if hit_up and not hit_down:
            count += 1
    return count


def crossings(segment_points, segments, triangle_points, triangles) -> int:
    """How many segments pass through a triangle, by Moller-Trumbore.

    An INTERSECTION COUNT, not a distance: the question is whether the two
    surfaces are on the wrong sides of each other, and a distance answers a
    different one. The parameters are taken in the open interval so a segment
    that merely touches a triangle's plane at an endpoint is not reported, which
    is what a resting contact does.
    """
    count = 0
    for a, b in segments:
        origin = segment_points[a]
        direction = segment_points[b] - origin
        for triangle in triangles:
            v0 = triangle_points[triangle[0]]
            v1 = triangle_points[triangle[1]]
            v2 = triangle_points[triangle[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            pvec = np.cross(direction, e2)
            det = float(np.dot(e1, pvec))
            if abs(det) < 1e-16:
                continue
            inv = 1.0 / det
            tvec = origin - v0
            u = float(np.dot(tvec, pvec)) * inv
            if u <= 0.0 or u >= 1.0:
                continue
            qvec = np.cross(tvec, e1)
            v = float(np.dot(direction, qvec)) * inv
            if v <= 0.0 or u + v >= 1.0:
                continue
            t = float(np.dot(e2, qvec)) * inv
            if 0.0 < t < 1.0:
                count += 1
    return count


def run(name: str, disable_contact: bool) -> dict:
    """Simulate the pair once and measure the output."""
    print(f"\n=== {name}: disable-contact {disable_contact} ===")
    app = App.create(f"cpu-contact-{name}")
    app.asset.add.tri("lower", lower_vertices, lower_faces)
    app.asset.add.tri("upper", upper_vertices, upper_faces)

    scene = app.scene.create()
    lower = scene.add("lower")
    lower.param.set("young-mod", YOUNG_MODULUS).set("bend", BEND)
    lower.pin(LOWER_CORNERS).move_by([0.0, 0.0, 0.0], t_start=0.0, t_end=ELAPSED)
    upper = scene.add("upper")
    upper.param.set("young-mod", YOUNG_MODULUS).set("bend", BEND)
    fixed_scene = scene.build()

    session = app.session.create(fixed_scene)
    session.param.set("dt", DT).set("fps", FPS).set(
        "gravity", [0.0, GRAVITY, 0.0]
    ).set("precond", "block-jacobi").set("frames", FRAMES).set(
        "disable-contact", disable_contact
    )
    session = session.build()
    session.start(blocking=True)

    stdout = "\n".join(session.get.log.stdout())
    finished = session.finished()
    print(f"finished: {finished}")
    if not finished:
        for line in stdout.splitlines():
            if line.startswith("###") or "FATAL" in line or "refus" in line:
                print(f"  {line.strip()}")
    assert finished, (
        "the run did not complete its frames; the lines above name what the "
        "backend refused or where it stopped"
    )

    output_root = session.output.path
    present = sorted(
        int(entry[len("vert_") : -len(".bin")])
        for entry in os.listdir(output_root)
        if entry.startswith("vert_") and entry.endswith(".bin")
    )
    # FRAMES + 1 FILES: `vert_0.bin` is the rest pose, written before the first
    # step, and one file follows per simulated frame at exactly `f / fps`.
    assert present == list(range(FRAMES + 1)), (
        f"expected frames 0..{FRAMES} contiguous (frame 0 is the rest pose), "
        f"got {present}"
    )

    def read_frame(index: int) -> np.ndarray:
        # float32, as every vert_N.bin is.
        path = os.path.join(output_root, f"vert_{index}.bin")
        return np.fromfile(path, dtype=np.float32).reshape(-1, 3)

    rest = read_frame(0)
    assert rest.shape[0] == LOWER_COUNT + UPPER_COUNT, (
        f"expected {LOWER_COUNT + UPPER_COUNT} vertices, got {rest.shape[0]}"
    )

    # THE OUTPUT IS IN SOLVER ORDER, NOT AUTHORING ORDER. `index_map` orders the
    # solved namespace, so an authored index reads a different vertex. The rest
    # pose is that order's own geometry, so the map is recovered from it once and
    # every frame is indexed through it.
    def solver_rows(authored: np.ndarray) -> list[int]:
        rows = []
        for point in authored:
            target = point.astype(np.float32)
            distance = np.linalg.norm(rest - target, axis=1)
            nearest = int(np.argmin(distance))
            assert distance[nearest] < 1e-5, (
                f"authored vertex at {target.tolist()} has no match in the "
                f"solver's rest pose (nearest is {distance[nearest]:.3e} m away)"
            )
            rows.append(nearest)
        return rows

    lower_rows = solver_rows(lower_vertices)
    upper_rows = solver_rows(upper_vertices)
    assert sorted(lower_rows + upper_rows) == list(range(rest.shape[0])), (
        "the authored-to-solver map is not a bijection, so two authored vertices "
        "matched one solver row"
    )

    total = 0
    first_crossing = None
    wrong_side = 0
    first_wrong_side = None
    for index in range(FRAMES + 1):
        frame = read_frame(index)
        lower_pose = frame[lower_rows].astype(np.float64)
        upper_pose = frame[upper_rows].astype(np.float64)
        found = crossings(upper_pose, UPPER_EDGES, lower_pose, lower_faces)
        found += crossings(lower_pose, LOWER_EDGES, upper_pose, upper_faces)
        total += found
        if found and first_crossing is None:
            first_crossing = index
        under = below_count(upper_pose, lower_pose, lower_faces)
        wrong_side += under
        if under and first_wrong_side is None:
            first_wrong_side = index

    last = read_frame(FRAMES)
    descent = float(rest[upper_rows][:, 1].mean() - last[upper_rows][:, 1].mean())
    print(f"segment-triangle crossings over {FRAMES + 1} frames: {total}")
    print(f"upper vertices under the lower sheet, summed over frames: {wrong_side}")
    print(
        f"upper sheet mean descent: {descent * 1e3:.2f} mm "
        f"(gap {DROP * 1e3:.0f} mm, free fall {FREE_FALL * 1e3:.1f} mm)"
    )
    return {
        "crossings": total,
        "first_crossing": first_crossing,
        "wrong_side": wrong_side,
        "first_wrong_side": first_wrong_side,
        "descent": descent,
    }


held = run("held", disable_contact=False)
through = run("through", disable_contact=True)

# --------------------------------------------------------------------------
# 1. THE CONTROL: the geometry really does interleave, and the test can see it.
#
# This is not a check on the solver's physics. It is what makes the zero below
# mean something: it establishes that these two sheets, on this trajectory,
# produce hundreds of segment-triangle crossings when nothing stops them.
# --------------------------------------------------------------------------
assert through["wrong_side"] > 0, (
    "the control run with contact disabled left no upper vertex under the lower "
    "sheet, so the two never pass through each other on this trajectory and the "
    "zero below would be evidence of nothing. Re-author the geometry so the free "
    "sheet falls through the pinned one"
)
assert through["descent"] > 0.8 * FREE_FALL, (
    f"the control sheet descended {through['descent']:.4e} m against "
    f"{FREE_FALL:.4e} m of free fall, so something slowed it even with contact "
    "disabled and it is not the clean control this comparison needs"
)

# --------------------------------------------------------------------------
# 2. NO CROSSING, ON ANY COMMITTED FRAME, WITH CONTACT ON.
# --------------------------------------------------------------------------
assert held["crossings"] == 0, (
    f"the two sheets interleave at frame {held['first_crossing']} and after: "
    f"{held['crossings']} segment-triangle intersections in total. "
    "Non-penetration is enforced by the ACCD CCD-filtered line search plus "
    "check_intersection, not by the barrier, so a non-zero count here means one "
    "of those two is absent or is not bounding the step it was asked about"
)
assert held["wrong_side"] == 0, (
    f"an upper vertex is under the lower sheet from frame "
    f"{held['first_wrong_side']}: {held['wrong_side']} vertex-frames on the wrong "
    f"side, against {through['wrong_side']} in the contact-disabled control. The "
    "sheets moved further per frame than they are apart, so a tunnel leaves no "
    "interleaved pose to find and this is the measurement that records it"
)

# --------------------------------------------------------------------------
# 3. THE UPPER SHEET FELL, AND WAS STOPPED.
#
# A backend that froze the scene, or refused every step while still writing
# frames, passes (2) trivially; one whose barrier is assembled but whose line
# search does not bound the step gives up the whole gap and keeps going.
# --------------------------------------------------------------------------
assert held["descent"] > 0.25 * DROP, (
    f"the upper sheet descended only {held['descent']:.4e} m of a {DROP:.4e} m "
    "gap, so it never reached the lower sheet and this run tested nothing about "
    "contact"
)
assert held["descent"] < 0.6 * FREE_FALL, (
    f"the upper sheet descended {held['descent']:.4e} m against "
    f"{FREE_FALL:.4e} m of free fall and {through['descent']:.4e} m in the "
    "contact-disabled control, so nothing meaningfully stopped it"
)

print(
    f"\ncpu contact fixture: PASS (held {held['crossings']} crossings and "
    f"{held['wrong_side']} wrong-side vertex-frames, control "
    f"{through['crossings']} and {through['wrong_side']})"
)
