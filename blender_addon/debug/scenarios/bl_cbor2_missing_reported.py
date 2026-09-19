# File: scenarios/bl_cbor2_missing_reported.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A missing cbor2 wheel is reported, not raised as a traceback.
#
# ``cbor2`` ships as a bundled wheel that Blender installs when it installs the
# extension. An add-on directory placed under ``extensions/user_default`` by
# hand, or copied from a checkout whose gitignored ``wheels/*.whl`` were never
# fetched, enables normally, registers every operator, and has no cbor2:
# measured on Blender 5.2, Blender does NOT refuse to enable such an extension.
# The failure then arrives from whichever button encodes first, which is the
# community report: a raw ``ModuleNotFoundError: No module named 'cbor2'``
# traceback out of a Transfer click.
#
# ``core.module.get_cbor2`` raises ``Cbor2NotInstalledError`` carrying an
# actionable message for exactly this reason. What this scenario pins is that
# EVERY entry point converts it into that message, that the panel the buttons
# live on says so before they are pressed, and that the type keeps the shape
# the handlers depend on:
#
#   * Cbor2NotInstalledError is a ModuleNotFoundError, so an uncaught raise
#     still reads as a missing module, and it is NOT a ValueError, so an
#     ``except ValueError`` handler does not silently cover it.
#   * get_cbor2 raises it, carrying the message naming both recoveries.
#   * the Transfer, Update Params, Run and Resume paths each report the message
#     and set the connection error the panel's repair branch reads, rather than
#     letting the exception escape.
#   * the debug Transfer-without-Build path reports it too.
#   * the Solver panel, which owns Transfer and Run, warns when cbor2 is
#     missing, so the condition is visible before a button is pressed.
#   * the generic staged-encode fallback records the failure on the connection
#     as well, so an unclassified encode failure cannot leave the panel looking
#     healthy after nothing was uploaded.
#   * the manifest carries a wheel for the interpreter actually running, and
#     every wheel it names is on disk. A Blender release that moves to a Python
#     ABI no manifest cell covers is the way this whole condition gets created,
#     and it is invisible in the source, so it is measured against the running
#     Blender on whatever host the rig runs on.
#
# The scenario substitutes ``get_cbor2`` and ``cbor2_available`` for the
# duration and restores them, so it neither needs nor produces an install
# whose wheel is really absent. It never contacts a server.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True
# Pure error-plumbing logic; no solver is involved.
BACKENDS = ("real",)


_DRIVER_BODY = r'''
import os
import traceback

result.setdefault("errors", [])
result.setdefault("checks", {})


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


class _FakeLayout:
    """Records label() calls so a panel draw can be inspected without a real
    Blender UILayout."""

    def __init__(self):
        self.labels = []
        self.props = []

    def label(self, text="", icon="", **kw):
        self.labels.append((text, icon))

    def prop(self, _data, name, **kw):
        self.props.append(name)

    def row(self, **kw):
        return self

    def column(self, **kw):
        return self

    def box(self):
        return self

    def operator(self, *a, **kw):
        return self

    def separator(self, **kw):
        return self


module_mod = None
solver_ui = None
saved = {}
try:
    module_mod = __import__(pkg + ".core.module",
                            fromlist=["Cbor2NotInstalledError"])
    cbor_encode = __import__(pkg + ".core.encoder.cbor_encode",
                             fromlist=["dumps_envelope"])
    solver_ui = __import__(pkg + ".ui.solver", fromlist=["SOLVER_PT_SolverPanel"])
    debug_ops = __import__(pkg + ".ui.debug_ops",
                           fromlist=["DEBUG_OT_TransferWithoutBuild"])
    client = __import__(pkg + ".core.client", fromlist=["communicator"])
    facade = __import__(pkg + ".core.facade", fromlist=["tick"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    com = client.communicator

    Cbor2NotInstalledError = module_mod.Cbor2NotInstalledError
    MESSAGE = module_mod.CBOR2_MISSING_MESSAGE

    # ---- the type keeps the shape every handler depends on ----
    record("error_is_a_module_not_found_error",
           issubclass(Cbor2NotInstalledError, ModuleNotFoundError), {})
    # An `except ValueError` cannot stand in for it, which is why each call
    # site names the type explicitly.
    record("error_is_not_a_value_error",
           not issubclass(Cbor2NotInstalledError, ValueError), {})
    record("message_names_both_recoveries",
           "Reinstall" in MESSAGE and "Install cbor2" in MESSAGE,
           {"message": MESSAGE})

    # ---- get_cbor2 raises it rather than letting the import error out ----
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def _no_cbor2(name, *a, **kw):
        if name == "cbor2":
            raise ModuleNotFoundError("No module named 'cbor2'")
        return real_import(name, *a, **kw)

    if isinstance(__builtins__, dict):
        __builtins__["__import__"] = _no_cbor2
    else:
        __builtins__.__import__ = _no_cbor2
    try:
        raised = None
        try:
            module_mod.get_cbor2()
        except BaseException as exc:
            raised = exc
        record("get_cbor2_raises_the_typed_error",
               isinstance(raised, Cbor2NotInstalledError),
               {"raised": type(raised).__name__ if raised else None})
        record("get_cbor2_carries_the_actionable_message",
               str(raised) == MESSAGE if raised else False,
               {"text": str(raised) if raised else None})
    finally:
        if isinstance(__builtins__, dict):
            __builtins__["__import__"] = real_import
        else:
            __builtins__.__import__ = real_import

    # ---- every encode entry point converts it into a report ----
    # Substitute the encoder helpers the operators call, so the whole handler
    # runs against a realistic failure without needing a wheel-less install.
    def _raise_missing(*a, **kw):
        raise Cbor2NotInstalledError(MESSAGE)

    saved["encode_obj_with_hash"] = solver_ui.encode_obj_with_hash
    saved["encode_param_with_hash"] = solver_ui.encode_param_with_hash
    saved["compute_data_hash"] = solver_ui.compute_data_hash
    saved["compute_param_hash"] = solver_ui.compute_param_hash
    solver_ui.encode_obj_with_hash = _raise_missing
    solver_ui.encode_param_with_hash = _raise_missing
    solver_ui.compute_data_hash = _raise_missing
    solver_ui.compute_param_hash = _raise_missing

    StageAbort = __import__(pkg + ".core.async_op",
                            fromlist=["StageAbort"]).StageAbort

    class _OperatorSelf:
        """Stand-in for the operator instance.

        A Blender Operator cannot be instantiated from Python
        (``bpy_struct.__new__`` demands its own argument), and the staged
        encode steps only read ``self._payload``, so an ordinary object
        carrying that attribute drives the real method unbound.
        """

        def __init__(self):
            self._payload = {}

    def _stage_reports(cls, method_name, label):
        """Run one staged encode step and report how it failed.

        ``com.set_error`` dispatches onto the engine queue, which Blender's
        main-thread timer drains, so the panel's error is read AFTER a tick.
        Reading it before the tick would report an empty string no matter what
        the handler did.
        """
        com.set_error("")
        facade.tick()
        op = _OperatorSelf()
        aborted = None
        try:
            getattr(cls, method_name)(op, bpy.context)
        except StageAbort as exc:
            aborted = exc
        except BaseException as exc:
            record(label, False,
                   {"escaped": f"{type(exc).__name__}: {exc}"})
            return
        facade.tick()
        record(
            label,
            aborted is not None and MESSAGE in str(aborted) and MESSAGE in com.error,
            {"abort": str(aborted) if aborted else None, "com_error": com.error},
        )

    _stage_reports(solver_ui.SOLVER_OT_Transfer, "_stage_encode_geometry",
                   "transfer_geometry_reports_missing_cbor2")
    _stage_reports(solver_ui.SOLVER_OT_Transfer, "_stage_encode_params",
                   "transfer_params_reports_missing_cbor2")
    _stage_reports(solver_ui.SOLVER_OT_Run, "_stage_check_geometry",
                   "run_geometry_reports_missing_cbor2")
    _stage_reports(solver_ui.SOLVER_OT_Run, "_stage_check_params",
                   "run_params_reports_missing_cbor2")

    # ---- Resume names the type its handler catches ----
    import inspect
    resume_src = inspect.getsource(solver_ui.SOLVER_OT_ResumeFrom.invoke)
    record("resume_catches_the_typed_error",
           resume_src.count("Cbor2NotInstalledError") >= 2,
           {"occurrences": resume_src.count("Cbor2NotInstalledError")})

    debug_src = inspect.getsource(debug_ops.DEBUG_OT_TransferWithoutBuild.execute)
    record("debug_transfer_catches_the_typed_error",
           "Cbor2NotInstalledError" in debug_src, {})

    # The generic staged-encode fallback must record the failure on the
    # connection too, so an unclassified encode error is visible in the panel.
    async_src = inspect.getsource(
        __import__(pkg + ".core.async_op", fromlist=["AsyncOperator"])
        .AsyncOperator._run_stage_tick
    )
    record("generic_encode_failure_sets_the_panel_error",
           "set_error" in async_src, {})

    # ---- the manifest carries a wheel for the Blender running this ----
    # The reason a wheel goes missing at all is that the manifest names one
    # per (Python ABI x platform) cell and a Blender release can move to an
    # ABI no cell covers. That is unobservable from the source alone, so the
    # check is made against the interpreter actually running: whatever host
    # the rig runs on, the add-on installed there must have had a wheel to
    # install. A Blender version bump that outruns the manifest fails here
    # rather than in a user's first Transfer.
    import platform as _platform
    import re
    import sys as _sys
    manifest_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(module_mod.__file__))),
        "blender_manifest.toml",
    )
    manifest = open(manifest_path, encoding="utf-8").read()
    wheels = re.findall(r'"\./wheels/([^"]+)"', manifest)
    abi = f"cp{_sys.version_info.major}{_sys.version_info.minor}"
    # The platform tag is keyed on the OS AND the machine. Keyed on the OS
    # alone, an arm64 host passes on the strength of the x86_64 wheel, which
    # Blender does not install there. An unlisted pair is named in the key
    # itself, so it fails with the pair in the record.
    os_key = "linux" if _sys.platform.startswith("linux") else _sys.platform
    machine = _platform.machine().lower()
    plat_key = {
        ("darwin", "arm64"): "macosx_11_0_arm64",
        ("linux", "x86_64"): "manylinux_2_28_x86_64",
        ("linux", "aarch64"): "manylinux_2_28_aarch64",
        ("win32", "amd64"): "win_amd64",
        ("win32", "arm64"): "win_arm64",
    }.get((os_key, machine), f"<no wheel platform for {os_key}/{machine}>")
    matching = [w for w in wheels if w.endswith(f"-{abi}-{abi}-{plat_key}.whl")]
    record("manifest_has_a_wheel_for_this_blender",
           bool(matching),
           {"abi": abi, "platform_key": plat_key, "wheels": wheels})
    record("manifest_declares_every_wheel_it_ships",
           all(os.path.exists(os.path.join(os.path.dirname(manifest_path),
                                           "wheels", w)) for w in wheels),
           {"missing": [w for w in wheels
                        if not os.path.exists(os.path.join(
                            os.path.dirname(manifest_path), "wheels", w))]})

    # ---- the panel the buttons live on says so before they are pressed ----
    class _PanelSelf:
        def __init__(self, layout):
            self.layout = layout

    saved["cbor2_available"] = solver_ui.cbor2_available
    solver_ui.cbor2_available = lambda: False
    fl_missing = _FakeLayout()
    solver_ui.SOLVER_PT_SolverPanel.draw(_PanelSelf(fl_missing), bpy.context)
    record(
        "solver_panel_warns_when_cbor2_is_missing",
        any(ic == "ERROR" and "cbor2" in t for t, ic in fl_missing.labels),
        {"labels": fl_missing.labels[:6]},
    )

    solver_ui.cbor2_available = lambda: True
    fl_present = _FakeLayout()
    solver_ui.SOLVER_PT_SolverPanel.draw(_PanelSelf(fl_present), bpy.context)
    record(
        "solver_panel_silent_when_cbor2_is_present",
        not any("cbor2" in t for t, _ in fl_present.labels),
        {"labels": fl_present.labels[:6]},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
finally:
    if solver_ui is not None:
        for name, fn in saved.items():
            setattr(solver_ui, name, fn)
'''


def build_driver(ctx: r.ScenarioContext) -> str:
    """Return the Python source the bootstrap will exec inside Blender.

    No substitutions are needed: the scenario substitutes the encoder helpers
    and the availability probe for the duration of its own checks and restores
    them, so it neither installs nor removes a wheel and never contacts a
    server. It runs on any host.
    """
    return _DRIVER_BODY


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
