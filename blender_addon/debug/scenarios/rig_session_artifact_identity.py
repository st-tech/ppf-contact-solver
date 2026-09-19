# File: scenarios/rig_session_artifact_identity.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Guard for the identity the runner's cached session artifacts carry.
#
# The vertex map, the surface map and the statistics manifest are downloaded
# only while unset, so one fetch pays for them and every later frame reuses
# them. That is required: a live run fetches on every status poll, and
# re-reading three files per poll would put the download in the frame loop.
# The cost of the cache is that the artifacts describe ONE uploaded dataset
# and outlive it unless something drops them, which is how a scene whose
# groups changed came to decode its frames against the previous scene's
# manifest and drop every frame of the new run ("statistics frame object
# count does not match manifest").
#
# ``EffectRunner._drop_stale_session_artifacts`` binds them to the
# ``upload_id`` the server reports, and both fetch entry points call it.
# This scenario drives that method through the real ``_do_fetch_frames`` and
# ``_do_fetch_map`` against a fake backend, so it fails if either call site
# is removed, if the identity stops being recorded, if a reset path leaves
# an identity behind, and equally if the cache stops caching.
#
# ``bl_stale_statistics_manifest`` covers the same defect end to end through
# Blender and the solver. This one needs neither, so it runs on every host
# and in the server-only jobs, and it names the mechanism rather than the
# symptom.
#
# The probe runs in a SUBPROCESS. Loading addon modules means stubbing
# ``bpy``, and the orchestrator imports every scenario into one long-lived
# process, so installing that stub here would leave a fake ``bpy`` in
# ``sys.modules`` for everything else in the run. The stub loader is a
# trimmed copy of the one in ``addon_host_tests/conftest.py``, which is not
# imported because it is a pytest conftest and the rig must not require
# pytest to be installed.

from __future__ import annotations

import json
import subprocess
import sys

from . import REPO_ROOT_POSIX
from . import _runner as r


# No Blender and no solver, so this holds on the real-GPU jobs too.
BACKENDS = ("real",)


_PROBE = r'''
import json
import os
import pickle
import sys
import types

REPO_ROOT = sys.argv[1]
ADDON_ROOT = os.path.join(REPO_ROOT, "blender_addon")


def install_stubs():
    """Register the Blender modules the import chain reaches.

    Every level has to be a real ``sys.modules`` entry, not an attribute on
    a namespace object, because the add-on resolves ``from bpy.app.handlers
    import ...`` through the import system.
    """
    bpy = types.ModuleType("bpy")
    app = types.ModuleType("bpy.app")
    handlers = types.ModuleType("bpy.app.handlers")
    translations = types.ModuleType("bpy.app.translations")
    handlers.persistent = lambda fn: fn
    translations.pgettext_iface = lambda text, *a, **k: text
    translations.pgettext_tip = lambda text, *a, **k: text
    app.handlers = handlers
    app.translations = translations
    bpy.app = app
    bpy.data = types.SimpleNamespace(filepath="", texts={})
    bpy.types = types.SimpleNamespace()
    sys.modules.update({
        "bpy": bpy,
        "bpy.app": app,
        "bpy.app.handlers": handlers,
        "bpy.app.translations": translations,
    })
    mathutils = types.ModuleType("mathutils")
    for name in ("Vector", "Matrix", "Quaternion", "Euler"):
        setattr(mathutils, name, type(name, (list,), {}))
    sys.modules["mathutils"] = mathutils


def load(dotted):
    """Load ``blender_addon.<dotted>`` from source without running any
    package ``__init__``, whose first line imports bpy for real."""
    import importlib.util

    fqname = "blender_addon." + dotted
    if fqname in sys.modules:
        return sys.modules[fqname]
    parts = dotted.split(".")
    for depth in range(0, len(parts)):
        name = "blender_addon" + ("." + ".".join(parts[:depth]) if depth else "")
        if name not in sys.modules:
            shell = types.ModuleType(name)
            shell.__path__ = [os.path.join(ADDON_ROOT, *parts[:depth])]
            sys.modules[name] = shell
    source = os.path.join(ADDON_ROOT, *parts) + ".py"
    spec = importlib.util.spec_from_file_location(fqname, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[fqname] = module
    spec.loader.exec_module(module)
    return module


install_stubs()
runner_mod = load("core.effect_runner")
effects_mod = load("core.effects")

ROOT = "/probe/project"
VERT = b"\x00" * 12  # one float32 vertex triple


def artifacts(upload):
    # What the fake server holds for a given upload. The map is a pickle
    # because ``_ensure_anim_map`` sniffs the first byte to choose its
    # decoder, and the statistics blobs are opaque here: the runner moves
    # bytes and only the Blender side decodes them.
    import cbor2

    return {
        "map.pickle": pickle.dumps({upload: [[0, 1, 2]]}),
        "surface_map.pickle": pickle.dumps({"version": 2, "maps": {}}),
        # Every exported session carries one; this scene has no exact SOLID
        # pin, so it holds no blocks.
        "display_pin_map.pickle": cbor2.dumps({
            "version": 2,
            "kind": "DisplayPinMap",
            "payload": {"version": 1, "n_total": 0, "blocks": []},
        }),
        "statistics_manifest.cbor": ("manifest-" + upload).encode(),
        "vert_1.bin": VERT,
        "statistics_0.cbor": ("stats0-" + upload).encode(),
        "statistics_1.cbor": ("stats1-" + upload).encode(),
    }


class FakeBackend:
    backend_type = "local"

    def __init__(self):
        self.files = artifacts("a")
        self.reads = []

    def receive_data(self, path, project, chunk_size=None, progress_cb=None,
                     interrupt_cb=None):
        name = os.path.basename(path)
        self.reads.append(name)
        if name not in self.files:
            raise Exception("fake backend has no " + name)
        return self.files[name]

    def disconnect(self):
        pass


class FakeEngine:
    def __init__(self):
        self.state = None
        self.events = []

    def dispatch(self, event):
        self.events.append(type(event).__name__)


# The real constructor, so a field added to the runner later is initialized
# the way production initializes it rather than the way a probe guessed.
engine = FakeEngine()
runner = runner_mod.EffectRunner(engine)
backend = FakeBackend()
runner._backend = backend
runner._project_name = "probe"

out = {}


def poll(upload_id):
    runner._response_cache.record({"upload_id": upload_id})


def live_fetch():
    # ``_fetched`` must be passed by identity: the stale-context guard
    # compares the object, not its contents.
    del runner._fetched[:]
    runner._anim_frames.clear()
    before = len(backend.reads)
    runner._do_fetch_frames(ROOT, 1, runner._fetched, True)
    return backend.reads[before:]


def snapshot():
    return {
        "upload_id": runner._anim_upload_id,
        "manifest": (runner._anim_statistics_manifest or b"").decode() or None,
        "map_keys": sorted(runner._anim_map),
        "zero_fetched": runner._anim_statistics_zero_fetched,
    }


# 1. First live fetch under upload "a" caches the artifacts and records the
#    upload they came from.
poll("a")
out["first"] = {"reads": live_fetch(), "state": snapshot()}

# 2. Same upload, different bytes on the server: nothing is re-read. This is
#    the caching the live fetch depends on, so a fix that simply dropped the
#    cache would fail here rather than pass everything else.
backend.files = artifacts("b")
out["same_upload"] = {"reads": live_fetch(), "state": snapshot()}

# 3. A new upload invalidates: manifest, map and the zero-fetched flag all
#    come back from the server.
poll("b")
out["new_upload"] = {"reads": live_fetch(), "state": snapshot()}

# 4. The other entry point invalidates too.
poll("c")
backend.files = artifacts("c")
before = len(backend.reads)
runner._do_fetch_map(ROOT)
out["fetch_map"] = {"reads": backend.reads[before:], "state": snapshot()}

# 5. A frame queued under one upload must not survive into the next. The
#    map is dropped with the other artifacts, and a frame applied against
#    an empty map resolves to no object, so it writes no PC2 and raises
#    nothing: the run looks like it simply produced no output.
poll("e")
backend.files = artifacts("e")
del runner._fetched[:]
runner._anim_frames.clear()
runner._do_fetch_frames(ROOT, 1, runner._fetched, True)
queued_before = [frame[2][-1].decode() for frame in runner._anim_frames]
poll("f")
backend.files = artifacts("f")
del runner._fetched[:]
# The queue is deliberately NOT cleared here: dropping it is the runner's
# job, and this is the step that checks it does it.
runner._do_fetch_frames(ROOT, 1, runner._fetched, True)
out["queue_across_uploads"] = {
    "before": queued_before,
    "after": [frame[2][-1].decode() for frame in runner._anim_frames],
    "map_keys": sorted(runner._anim_map),
    "applied": runner._anim_applied,
}

# 6. A cleared cache must not keep claiming an identity, or the next fetch
#    would compare against an upload whose artifacts are already gone.
runner.execute(effects_mod.DoResetAnimationBuffer())
out["after_reset"] = snapshot()

poll("d")
live_fetch()
runner._do_disconnect()
out["after_disconnect"] = snapshot()

runner.stop()
print("PROBE_JSON " + json.dumps(out))
'''


def _violations(out: dict) -> list[str]:
    v: list[str] = []

    first = out["first"]
    if first["state"]["upload_id"] != "a":
        v.append(
            f"first fetch recorded upload_id {first['state']['upload_id']!r}, "
            f"expected 'a'; without an identity nothing can go stale"
        )
    if first["state"]["manifest"] != "manifest-a":
        v.append(f"first fetch cached {first['state']['manifest']!r}")
    if first["state"]["map_keys"] != ["a"]:
        v.append(f"first fetch cached map {first['state']['map_keys']}")

    same = out["same_upload"]
    if "statistics_manifest.cbor" in same["reads"] or "map.pickle" in same["reads"]:
        v.append(
            f"a second fetch under the same upload re-read {same['reads']}; "
            f"the artifacts are re-downloaded on every live poll"
        )
    if same["state"]["manifest"] != "manifest-a":
        v.append(
            f"the cache changed under an unchanged upload: "
            f"{same['state']['manifest']!r}"
        )

    new = out["new_upload"]
    if new["state"]["upload_id"] != "b":
        v.append(f"new upload recorded {new['state']['upload_id']!r}, expected 'b'")
    if new["state"]["manifest"] != "manifest-b":
        v.append(
            f"a new upload left the manifest at {new['state']['manifest']!r}; "
            f"frames of the new run decode against the previous manifest"
        )
    if new["state"]["map_keys"] != ["b"]:
        v.append(f"a new upload left the vertex map at {new['state']['map_keys']}")
    if new["state"]["zero_fetched"] is not True:
        v.append("the new upload's statistics_0 was never fetched")

    fetch_map = out["fetch_map"]
    if fetch_map["state"]["upload_id"] != "c" or fetch_map["state"]["map_keys"] != ["c"]:
        v.append(
            f"_do_fetch_map did not invalidate: {fetch_map['state']}; "
            f"the batch fetch entry point is not calling "
            f"_drop_stale_session_artifacts"
        )
    if fetch_map["state"]["manifest"] is not None:
        v.append(
            f"_do_fetch_map left a stale manifest "
            f"({fetch_map['state']['manifest']!r})"
        )

    queue = out["queue_across_uploads"]
    if queue["before"] != ["stats1-e"]:
        v.append(f"the probe queued {queue['before']}, expected ['stats1-e']")
    if "stats1-e" in queue["after"]:
        v.append(
            f"a frame fetched under the previous upload survived into the "
            f"next ({queue['after']}); it is applied against the new map, or "
            f"against an empty one, and writes no PC2 either way"
        )
    if queue["after"] != ["stats1-f"] or queue["map_keys"] != ["f"]:
        v.append(
            f"after the upload changed the queue held {queue['after']} with "
            f"map {queue['map_keys']}, expected the new upload's frame and map"
        )
    if queue["applied"] != 0:
        v.append(
            f"the applied counter survived the queue it counts "
            f"({queue['applied']}), so the fetch reports progress it lost"
        )

    for name in ("after_reset", "after_disconnect"):
        state = out[name]
        if state["upload_id"] is not None or state["manifest"] is not None:
            v.append(
                f"{name}: cleared artifacts still claim {state['upload_id']!r}; "
                f"the next fetch would match it and skip the download"
            )
    return v


def run(ctx: r.ScenarioContext) -> dict:
    probe = subprocess.run(
        [sys.executable, "-c", _PROBE, REPO_ROOT_POSIX],
        capture_output=True, text=True, timeout=max(60.0, ctx.timeout),
    )
    line = next(
        (l for l in probe.stdout.splitlines() if l.startswith("PROBE_JSON ")), None
    )
    if probe.returncode != 0 or line is None:
        tail = (probe.stdout + probe.stderr)[-1500:]
        return r.failed([f"probe did not report (rc={probe.returncode}):\n{tail}"])
    violations = _violations(json.loads(line[len("PROBE_JSON "):]))
    return r.failed(violations) if violations else r.passed()
