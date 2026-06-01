# File: frontend/build_worker.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Build pipeline subprocess driver. ``ppf-cts-server`` (Rust binary)
# spawns this script when handling ``DoSpawnBuild``: it cannot link
# against libpython without widening its dependency footprint, so the
# decode + tetrahedralize + FixedScene work runs here.
#
# Wire format (stdout, line-buffered):
#
#   PROGRESS percent=NN info=<text>\n   per-stage progress update
#   META frames=<int>\n                 total frames from param.pickle
#   ERROR <message>\n                   fatal error, followed by exit 1
#
# Side channel (file): on a scene-validation failure the structured
# violation payload (world-space geometry for the viewport overlay) is
# written to ``<root>/build_violations.json``. The stdout protocol only
# carries a flat ERROR string, so geometry travels via this file; the
# server reads it back on failure and forwards it to the add-on.
#
# Cancellation: the parent sends SIGTERM. The handler raises
# ``KeyboardInterrupt``, BlenderApp.populate().make() unwinds, and we
# exit with code 130 so the parent can distinguish cancel from crash.

import signal
import sys
import traceback


def _emit(line: str) -> None:
    """Write ``line`` to stdout, flush so the parent sees it promptly.

    The Rust side reads ``BufReader::lines()`` which only yields once a
    newline is seen; without ``flush()`` the kernel pipe buffer can hold
    progress updates indefinitely and the UI freezes mid-build.
    """
    sys.stdout.write(line)
    if not line.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def _progress(progress: float, info: str) -> None:
    # Strip newlines from ``info`` so the line-oriented protocol stays
    # parseable. The percent integer truncates to two decimals worth of
    # precision, which is plenty for a UI progress bar.
    safe = (info or "").replace("\n", " ").replace("\r", " ")
    pct = max(0.0, min(1.0, float(progress)))
    _emit(f"PROGRESS percent={pct:.4f} info={safe}")


def _on_sigterm(_signum, _frame):  # pragma: no cover - signal path
    # Translate SIGTERM into KeyboardInterrupt so the build's Python
    # frames unwind through their normal exception machinery (closing
    # files, releasing GPU handles) instead of dying mid-syscall.
    raise KeyboardInterrupt("SIGTERM received")


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        _emit("ERROR usage: build_worker.py <name> <root>")
        return 2
    name = argv[1]
    root = argv[2]

    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        # Patches the production ``frontend`` module when the test rig
        # invokes us with ``PPF_CTS_DATA_ROOT`` set. Production runs
        # without the env var and skip this entirely. Importing the
        # helper here (not at module top) keeps the ``ERROR`` path
        # clean if the frontend package fails to import.
        from frontend import _debug_runtime_
        _debug_runtime_.install_debug_patches()

        # Imported lazily so ``ERROR`` can still be reported if the
        # frontend package or its dependencies fail to import.
        import os
        from frontend import BlenderApp

        app = BlenderApp(name, progress_callback=_progress)
        # ppf-cts-server stores `data.pickle` and `param.pickle` under
        # the path the addon supplied (its remote `current_directory`
        # plus the project name) and passes that path back as `root`.
        # When that differs from the canonical `<data_dirpath>/<name>`
        # BlenderApp computed for itself, override `BlenderApp._root`
        # so the build worker reads from the same location the addon
        # wrote to. Skip when the test rig already steered us via
        # `PPF_CTS_DATA_ROOT`.
        if not os.environ.get("PPF_CTS_DATA_ROOT") and root != app._root:
            app._data_dirpath = os.path.dirname(root)
            app._root = root
            cache_root = os.path.join(root, ".cash")
            os.makedirs(cache_root, exist_ok=True)
            if hasattr(app, "_mesh_manager"):
                # `MeshManager` reads from `_cache_dir`, not `_cache_root`.
                # The previous typo silently no-op'd this override:
                # `MeshManager._cache_dir` stayed pointed at the canonical
                # path BlenderApp.__init__ computed via `blender_app_paths`,
                # so the per-project tet cache lived at one location while
                # the upload pickles lived at another. On a fresh project
                # name (or after the canonical cache was lost), every build
                # re-ran fTetWild from scratch and that's intrinsically
                # non-deterministic across runs. Two builds of the same
                # SOLID mesh produced different tet hulls (~1% size
                # difference); for a SHELL tucked inside the SOLID's
                # contact-gap zone, that difference flipped the kite
                # between "free" and "in contact" and the user saw the
                # kite locally wrinkle "for no reason."
                app._mesh_manager._cache_dir = cache_root
        app.populate().make()
        # Forward static build metadata the response builder needs to
        # publish solver progress (frame / total_frames). Pulled from
        # the FixedSession's resolved param set so static + dyn merges
        # are respected. Best-effort: a missing or non-int `frames` is
        # surfaced as a debug log on the parent side.
        try:
            total_frames = int(app.session._param.get("frames"))
            if total_frames > 0:
                _emit(f"META frames={total_frames}")
        except (AttributeError, TypeError, ValueError):
            pass
        # Drop a scene_info.json next to the project so ppf-cts-server's
        # response builder can splice it into every status response. The
        # build runs in this subprocess, so any field derived from the
        # parsed app must be persisted to disk before we exit. The
        # addon's panel renders this dict as the "Scene Info" box after
        # a successful build.
        try:
            import json
            info: dict[str, str] = {}
            fmt = lambda n: f"{n:,}"
            fs = app._fixed_scene
            if fs is not None:
                if fs._vert is not None and len(fs._vert) > 0:
                    info["Vertices"] = fmt(len(fs._vert[0]))
                if fs._tri is not None:
                    info["Triangles"] = fmt(len(fs._tri))
                if fs._tet is not None:
                    info["Tetrahedra"] = fmt(len(fs._tet))
                if fs._rod is not None and len(fs._rod) > 0:
                    info["Rod Edges"] = fmt(len(fs._rod))
            sd = getattr(app, "_scene_decoder", None)
            if sd is not None and getattr(sd, "_data", None) is not None:
                n_dynamic = 0
                n_static = 0
                ref_groups: dict[str, str] = {}
                canonical_refs: dict[str, set[str]] = {}
                for group in sd._data:
                    gt = group.get("type", "")
                    objs = group.get("object", [])
                    if gt == "STATIC":
                        n_static += len(objs)
                    else:
                        n_dynamic += len(objs)
                    for obj in objs:
                        mesh_ref = obj.get("mesh_ref")
                        if mesh_ref:
                            ref_groups.setdefault(mesh_ref, gt)
                            canonical_refs.setdefault(mesh_ref, set()).add(
                                obj.get("name", "")
                            )
                            canonical_refs[mesh_ref].add(mesh_ref)
                info["Dynamic Objects"] = fmt(n_dynamic)
                info["Static Objects"] = fmt(n_static)
                if canonical_refs:
                    by_type: dict[str, list[int]] = {}
                    for ref_name, names in canonical_refs.items():
                        gt = ref_groups.get(ref_name, "")
                        by_type.setdefault(gt, []).append(len(names))
                    for gt in sorted(by_type):
                        counts = sorted(by_type[gt], reverse=True)
                        label = f"Shared {gt.capitalize()}s"
                        info[label] = "(" + ",".join(str(c) for c in counts) + ")"
            # Static session params (resolved by `make()` from static +
            # dyn merges) populate the "Total Frames" / "FPS" rows.
            # Dynamic rows ("Simulated Frames" / "Last Saved") are added
            # by the response builder per poll, since they change as the
            # solver runs.
            try:
                fss = app.session
                if fss is not None and getattr(fss, "_param", None) is not None:
                    frames = fss._param.get("frames")
                    if frames is not None:
                        info["Total Frames"] = fmt(int(frames))
                    fps = fss._param.get("fps")
                    if fps is not None:
                        info["FPS"] = str(int(fps))
            except Exception:
                pass
            with open(os.path.join(root, "scene_info.json"), "w") as fp:
                json.dump(info, fp)
        except Exception as exc:
            sys.stderr.write(f"scene_info write failed: {exc}\n")
        # Final progress beat so the parent sees 100% before EOF.
        _progress(1.0, "Build complete.")
        return 0
    except KeyboardInterrupt:
        # Cancel path: stay silent on stdout (parent already knows it
        # asked for cancel) and exit with the conventional cancel code.
        return 130
    except SystemExit as exc:
        # Honor explicit exit codes, but route any error message through
        # the wire format so the parent can surface it.
        code = int(exc.code) if isinstance(exc.code, int) else 1
        if exc.code and not isinstance(exc.code, int):
            _emit(f"ERROR {exc.code}")
        return code
    except BaseException as exc:
        # When scene validation fails, ValidationError carries a
        # structured ``violations`` payload (self-intersecting triangles,
        # contact-offset pairs, wall/sphere hits) with world-space
        # geometry. Persist it to a sidecar the server reads back on
        # failure and forwards to the add-on, which highlights the
        # offending faces in the viewport. ``getattr`` keeps this a
        # no-op for ordinary errors that carry no violations.
        violations = getattr(exc, "violations", None)
        if violations:
            try:
                import json
                import os
                with open(os.path.join(root, "build_violations.json"), "w") as fp:
                    json.dump({"violations": violations}, fp)
            except Exception as werr:  # best-effort; never mask the build error
                sys.stderr.write(f"build_violations write failed: {werr}\n")
        msg = f"{type(exc).__name__}: {exc}"
        _emit(f"ERROR {msg}")
        # Send the traceback to stderr for the server log; stdout stays
        # parseable for the line-oriented protocol.
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
