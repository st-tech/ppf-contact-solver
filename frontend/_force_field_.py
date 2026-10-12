# File: _force_field_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""External force fields: the script compiler and the session writer.

A force field pushes every free vertex it reaches. Most sources add an
acceleration (m/s^2, like gravity), evaluated once per solver step at the
vertex's position at the start of that step; a Wind is an air velocity that
acts through the aerodynamic drag; and Drag, Harmonic, a Vortex's damping and
Fluid are solved implicitly, at the Newton iterate (a Fluid read at the
step's end). It comes from two sources, which a scene may mix:

* **Field objects**, Blender-style fields (Force, Wind, Vortex, Turbulence,
  Magnetic, Drag, Harmonic, Texture and Fluid Flow) placed by a position and
  axes, which the solver evaluates exactly at each vertex on the device. Every
  continuous setting may be animated as per-sample values. Each type computes
  what Blender's field of that type was measured to, and
  ``frontend._field_eval_`` computes the same in double precision.
* **Exact scripts**, each a Python function ``eval(x, y, z, t)`` returning
  ``(ax, ay, az)``, compiled here into a small bytecode that the solver runs
  per vertex on the device. A script has no resolution and no domain.

The script is a restricted subset of Python, refused by line when it steps
outside it: arithmetic (``+ - * / // % **``), comparisons, ``and`` / ``or`` /
``not`` (which yield 1.0 or 0.0), ``if`` / ``elif`` / ``else``, conditional
expressions, local assignment (including ``+=`` and tuple unpacking),
``for i in range(<constant>)`` (unrolled), ``abs``, ``min``, ``max``,
``float``, and the functions and constants of ``math`` listed in
``MATH_FUNCTIONS`` and ``MATH_CONSTANTS``. Every path must end in
``return (ax, ay, az)``. The solver evaluates it in single precision.

THIS MODULE IMPORTS NOTHING FROM THE SOLVER, on purpose: the server runs it in
a short worker to answer the add-on's Compile and Check without a build, and
the bytecode it emits is verified again by the solver's loader
(``crates/ppf-cts-solver/src/force_field.rs``), which refuses anything that did
not come from here.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import sys
import textwrap
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# A package import only: the server's Compile and Check worker runs this
# module as `python -m frontend._force_field_ check`.
from . import _noise_ as _noise
from . import _script_api_ as _script_api
from ._field_eval_ import FLUID_BRICK, FLUID_EMPTY, fluid_bricks

noise = _noise.noise
curl_noise = _noise.curl_noise

# The bytecode table. `external_field.kernel.cpp` (`FieldOp`) and
# `force_field.rs` (`op`) spell the same numbers.
SCRIPT_VERSION = 3
OP_CONST = 1
OP_LOAD = 2
OP_STORE = 3
OP_ADD = 4
OP_SUB = 5
OP_MUL = 6
OP_DIV = 7
OP_NEG = 8
OP_MOD = 9
OP_POW = 10
OP_LT = 11
OP_LE = 12
OP_GT = 13
OP_GE = 14
OP_EQ = 15
OP_NE = 16
OP_NOT = 17
OP_AND = 18
OP_OR = 19
OP_JUMP = 20
OP_JUMP_IF_FALSE = 21
OP_RETURN = 22
# Pop x, y, z, w, seed; the operand is the octave count. NOISE pushes one
# value, CURL pushes three. Both are `_noise_.py`'s algorithm, which the
# kernel implements in single precision. w is how far the pattern has evolved:
# the compiler makes it from a call's `time * frequency` and applies `decay`
# with ordinary instructions, so the instruction itself has no more inputs.
OP_NOISE = 52
OP_CURL = 53

# name -> (opcode, arity)
MATH_FUNCTIONS = {
    "sqrt": (32, 1),
    "sin": (33, 1),
    "cos": (34, 1),
    "tan": (35, 1),
    "exp": (36, 1),
    "log": (37, 1),
    "fabs": (38, 1),
    "floor": (39, 1),
    "ceil": (40, 1),
    "tanh": (41, 1),
    "asin": (42, 1),
    "acos": (43, 1),
    "atan": (44, 1),
    "sinh": (45, 1),
    "cosh": (46, 1),
    "atan2": (48, 2),
    "hypot": (51, 2),
    "pow": (OP_POW, 2),
}
OP_ABS = 38
OP_FLOOR = 39
OP_MIN = 49
OP_MAX = 50
MATH_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
# The list every surface shows (`_script_api_.py`) names exactly these, so a
# function added here without its entry there fails at import.
assert set(MATH_FUNCTIONS) == {n for n, _, _ in _script_api.MATH}, "math list drifted"
assert set(MATH_CONSTANTS) == {n for n, _, _ in _script_api.MATH_CONSTANTS}, "constants drifted"
# The noise builtins: `noise` is a scalar call like any other, `curl_noise`
# yields three values and is admitted only where three fit.
VECTOR_BUILTINS = ("curl_noise",)
MAX_OCTAVES = 8

STACK_CAPACITY = 32
VAR_CAPACITY = 64
N_PARAMS = 4
MAX_INSTRUCTIONS = 65536
UNROLL_LIMIT = 4096

_BINOPS = {
    ast.Add: OP_ADD,
    ast.Sub: OP_SUB,
    ast.Mult: OP_MUL,
    ast.Div: OP_DIV,
    ast.Mod: OP_MOD,
}
_COMPARE = {
    ast.Lt: OP_LT,
    ast.LtE: OP_LE,
    ast.Gt: OP_GT,
    ast.GtE: OP_GE,
    ast.Eq: OP_EQ,
    ast.NotEq: OP_NE,
}


class ForceFieldScriptError(ValueError):
    """A script the compiler refuses. ``lineno`` is 1-based in the source."""

    def __init__(self, message: str, lineno: Optional[int] = None):
        self.lineno = lineno
        self.detail = message
        where = f"line {lineno}: " if lineno else ""
        super().__init__(f"force field script, {where}{message}")


@dataclass
class CompiledScript:
    """A compiled script: the words, the constant pool and the source."""

    code: list
    constants: list
    source: str
    max_stack: int = 0

    def summary(self) -> str:
        return (
            f"{len(self.code)} instructions, {len(self.constants)} constants, "
            f"stack depth {self.max_stack}"
        )


def _word(opcode: int, arg: int = 0) -> int:
    return opcode | (arg << 8)


class _Compiler:
    def __init__(self, z_up: bool):
        self.code: list = []
        self.constants: list = []
        self.const_index: dict = {}
        self.slots: dict = {}
        self.z_up = z_up
        self.temp_counter = 0

    # --- emission -------------------------------------------------------
    def emit(self, opcode: int, arg: int = 0, node=None) -> int:
        if len(self.code) >= MAX_INSTRUCTIONS:
            raise ForceFieldScriptError(
                f"the program exceeds {MAX_INSTRUCTIONS} instructions; "
                "unrolled loops are the usual cause",
                getattr(node, "lineno", None),
            )
        self.code.append(_word(opcode, arg))
        return len(self.code) - 1

    def patch(self, at: int, target: int) -> None:
        opcode = self.code[at] & 0xFF
        self.code[at] = _word(opcode, target)

    def const(self, value: float, node=None) -> None:
        v = float(value)
        if not math.isfinite(v):
            raise ForceFieldScriptError(
                f"the constant {value!r} is not finite", getattr(node, "lineno", None)
            )
        key = struct_f32(v)
        if key not in self.const_index:
            self.const_index[key] = len(self.constants)
            self.constants.append(v)
        self.emit(OP_CONST, self.const_index[key], node)

    def slot(self, name: str, node) -> int:
        if name not in self.slots:
            if len(self.slots) >= VAR_CAPACITY:
                raise ForceFieldScriptError(
                    f"the program needs more than {VAR_CAPACITY} variable slots "
                    "(the four arguments, the locals and the compiler's "
                    "temporaries); reuse variables",
                    node.lineno,
                )
            self.slots[name] = len(self.slots)
        return self.slots[name]

    def temp(self, node) -> int:
        # A temporary is live only inside the statement that made it, so the
        # counter restarts at every statement and the slots are reused.
        name = f"$t{self.temp_counter}"
        self.temp_counter += 1
        return self.slot(name, node)

    # --- entry ----------------------------------------------------------
    def compile_function(self, fn: ast.FunctionDef) -> None:
        args = fn.args
        if (
            args.posonlyargs
            or args.vararg
            or args.kwonlyargs
            or args.kwarg
            or args.defaults
            or len(args.args) != N_PARAMS
        ):
            raise ForceFieldScriptError(
                f"'{fn.name}' must take exactly four plain arguments, "
                "(x, y, z, t)",
                fn.lineno,
            )
        if fn.decorator_list:
            raise ForceFieldScriptError("decorators are not supported", fn.lineno)
        names = [a.arg for a in args.args]
        if len(set(names)) != N_PARAMS:
            raise ForceFieldScriptError("the four arguments need distinct names", fn.lineno)
        # Slots 0..3 are x, y, z, t in the solver's Y-up axes and scene units.
        # A Z-up script (the Blender add-on's) sees Blender's axes instead:
        # x_b = x, y_b = -z, z_b = y.
        raw = ["$x", "$y", "$z", "$t"]
        for r in raw:
            self.slot(r, fn)
        if self.z_up:
            for axis, (raw_name, negate) in enumerate(
                (("$x", False), ("$z", True), ("$y", False))
            ):
                self.emit(OP_LOAD, self.slots[raw_name], fn)
                if negate:
                    self.emit(OP_NEG, 0, fn)
                self.emit(OP_STORE, self.slot(names[axis], fn), fn)
            self.emit(OP_LOAD, self.slots["$t"], fn)
            self.emit(OP_STORE, self.slot(names[3], fn), fn)
        else:
            # Alias the argument names onto the raw slots.
            for name, r in zip(names, raw):
                self.slots[name] = self.slots[r]
        assigned = set(names)
        body = list(fn.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(
            getattr(body[0], "value", None), ast.Constant
        ) and isinstance(body[0].value.value, str):
            body = body[1:]
        returns = self.block(body, assigned)
        if not returns:
            raise ForceFieldScriptError(
                f"'{fn.name}' can finish without returning (ax, ay, az); "
                "every path must end in a return",
                fn.lineno,
            )

    # --- statements -----------------------------------------------------
    def block(self, stmts, assigned: set) -> bool:
        """Compile statements; returns True when every path returned."""
        for index, stmt in enumerate(stmts):
            if self.stmt(stmt, assigned):
                if index + 1 < len(stmts):
                    raise ForceFieldScriptError(
                        "code after a return can never run", stmts[index + 1].lineno
                    )
                return True
        return False

    def stmt(self, node, assigned: set) -> bool:
        self.temp_counter = 0
        if isinstance(node, ast.Return):
            self.return_(node, assigned)
            return True
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                raise ForceFieldScriptError("chained assignment is not supported", node.lineno)
            self.assign(node.targets[0], node.value, assigned, node)
            return False
        if isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name):
                raise ForceFieldScriptError("only a plain name can be updated in place", node.lineno)
            self.read_name(node.target.id, assigned, node)
            self.binop(node.op, node, assigned)
            self.expr(node.value, assigned)
            self.emit_binop(node.op, node)
            self.emit(OP_STORE, self.slot(node.target.id, node), node)
            return False
        if isinstance(node, ast.AnnAssign):
            if node.value is None:
                raise ForceFieldScriptError("a declaration needs a value", node.lineno)
            self.assign(node.target, node.value, assigned, node)
            return False
        if isinstance(node, ast.If):
            self.expr(node.test, assigned)
            jf = self.emit(OP_JUMP_IF_FALSE, 0, node)
            then_assigned = set(assigned)
            then_returns = self.block(node.body, then_assigned)
            if node.orelse:
                jend = None
                if not then_returns:
                    jend = self.emit(OP_JUMP, 0, node)
                self.patch(jf, len(self.code))
                else_assigned = set(assigned)
                else_returns = self.block(node.orelse, else_assigned)
                if jend is not None:
                    self.patch(jend, len(self.code))
                if then_returns and else_returns:
                    return True
                if then_returns:
                    assigned |= else_assigned
                elif else_returns:
                    assigned |= then_assigned
                else:
                    assigned |= then_assigned & else_assigned
                return False
            self.patch(jf, len(self.code))
            if then_returns:
                # The fall-through path is the only one continuing.
                return False
            assigned |= then_assigned & assigned
            return False
        if isinstance(node, ast.For):
            self.for_(node, assigned)
            return False
        if isinstance(node, ast.Pass):
            return False
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            return False
        raise ForceFieldScriptError(
            f"'{type(node).__name__}' statements are not supported; see the "
            "force field script subset",
            node.lineno,
        )

    def assign(self, target, value, assigned: set, node) -> None:
        if isinstance(target, ast.Name):
            self.expr(value, assigned)
            self.emit(OP_STORE, self.slot(target.id, node), node)
            assigned.add(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)) and self.is_vector_call(value):
            if len(target.elts) != 3 or not all(isinstance(t, ast.Name) for t in target.elts):
                raise ForceFieldScriptError(
                    f"{value.func.id} returns three numbers; unpack them into three "
                    "names, like cx, cy, cz = curl_noise(x, y, z)",
                    node.lineno,
                )
            self.vector_call(value, assigned)
            for t_ast in reversed(target.elts):
                self.emit(OP_STORE, self.slot(t_ast.id, node), node)
                assigned.add(t_ast.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
            if len(target.elts) != len(value.elts):
                raise ForceFieldScriptError("tuple unpacking needs matching lengths", node.lineno)
            temps = []
            for v in value.elts:
                self.expr(v, assigned)
                t = self.temp(node)
                self.emit(OP_STORE, t, node)
                temps.append(t)
            for t_ast, t in zip(target.elts, temps):
                if not isinstance(t_ast, ast.Name):
                    raise ForceFieldScriptError("only plain names can be unpacked into", node.lineno)
                self.emit(OP_LOAD, t, node)
                self.emit(OP_STORE, self.slot(t_ast.id, node), node)
                assigned.add(t_ast.id)
            return
        raise ForceFieldScriptError(
            "only a plain name or a tuple of names can be assigned", node.lineno
        )

    def return_(self, node, assigned: set, value=None) -> None:
        value = node.value if value is None else value
        if isinstance(value, ast.IfExp):
            # `return A if c else B` with two tuples is an if / else of two
            # returns; a conditional inside a component is an ordinary
            # expression and is handled below.
            self.expr(value.test, assigned)
            jf = self.emit(OP_JUMP_IF_FALSE, 0, node)
            self.return_(node, assigned, value.body)
            self.patch(jf, len(self.code))
            self.return_(node, assigned, value.orelse)
            return
        if self.is_vector_call(value):
            # `return curl_noise(...)`: the three values it pushes, stored in
            # reverse since the last one is on top.
            self.vector_call(value, assigned)
            temps = [self.temp(node) for _ in range(3)]
            for t in reversed(temps):
                self.emit(OP_STORE, t, node)
        elif not isinstance(value, (ast.Tuple, ast.List)) or len(value.elts) != 3:
            raise ForceFieldScriptError(
                "return a tuple of three numbers, (ax, ay, az)", node.lineno
            )
        else:
            # Evaluate the three components into temporaries first, so the
            # output axis mapping below can reorder them.
            temps = []
            for v in value.elts:
                self.expr(v, assigned)
                t = self.temp(node)
                self.emit(OP_STORE, t, node)
                temps.append(t)
        if self.z_up:
            # Blender (ax, ay, az) -> solver (ax, az, -ay).
            self.emit(OP_LOAD, temps[0], node)
            self.emit(OP_LOAD, temps[2], node)
            self.emit(OP_LOAD, temps[1], node)
            self.emit(OP_NEG, 0, node)
        else:
            for t in temps:
                self.emit(OP_LOAD, t, node)
        self.emit(OP_RETURN, 0, node)

    def for_(self, node, assigned: set) -> None:
        if node.orelse:
            raise ForceFieldScriptError("for/else is not supported", node.lineno)
        if not isinstance(node.target, ast.Name):
            raise ForceFieldScriptError("the loop variable must be a plain name", node.lineno)
        it = node.iter
        if not (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range"
            and not it.keywords
            and 1 <= len(it.args) <= 3
        ):
            raise ForceFieldScriptError(
                "only 'for i in range(<constant>)' loops are supported, and they are unrolled",
                node.lineno,
            )
        bounds = []
        for a in it.args:
            v = _constant_int(a)
            if v is None:
                raise ForceFieldScriptError(
                    "range() bounds must be integer constants, so the loop can be unrolled",
                    node.lineno,
                )
            bounds.append(v)
        values = list(range(*bounds))
        if len(values) > UNROLL_LIMIT:
            raise ForceFieldScriptError(
                f"a loop of {len(values)} iterations exceeds the unroll limit of {UNROLL_LIMIT}",
                node.lineno,
            )
        name = node.target.id
        for v in values:
            self.const(float(v), node)
            self.emit(OP_STORE, self.slot(name, node), node)
            assigned.add(name)
            inner = set(assigned)
            if self.block(node.body, inner):
                raise ForceFieldScriptError(
                    "a return inside a loop is not supported", node.lineno
                )
            assigned |= inner

    # --- expressions ----------------------------------------------------
    def read_name(self, name: str, assigned: set, node) -> None:
        if name not in assigned:
            if name in self.slots:
                raise ForceFieldScriptError(
                    f"'{name}' may be read before it is assigned on some path",
                    node.lineno,
                )
            raise ForceFieldScriptError(f"unknown name '{name}'", node.lineno)
        self.emit(OP_LOAD, self.slots[name], node)

    def binop(self, op, node, assigned) -> None:
        if type(op) not in _BINOPS and not isinstance(op, (ast.Pow, ast.FloorDiv)):
            raise ForceFieldScriptError(
                f"the operator '{type(op).__name__}' is not supported", node.lineno
            )

    def emit_binop(self, op, node) -> None:
        if isinstance(op, ast.Pow):
            self.emit(OP_POW, 0, node)
        elif isinstance(op, ast.FloorDiv):
            self.emit(OP_DIV, 0, node)
            self.emit(OP_FLOOR, 0, node)
        else:
            self.emit(_BINOPS[type(op)], 0, node)

    def expr(self, node, assigned: set) -> None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                self.const(1.0 if node.value else 0.0, node)
                return
            if isinstance(node.value, (int, float)):
                self.const(float(node.value), node)
                return
            raise ForceFieldScriptError(
                f"the constant {node.value!r} is not a number", node.lineno
            )
        if isinstance(node, ast.Name):
            self.read_name(node.id, assigned, node)
            return
        if isinstance(node, ast.Attribute):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "math"
                and node.attr in MATH_CONSTANTS
            ):
                self.const(MATH_CONSTANTS[node.attr], node)
                return
            raise ForceFieldScriptError(
                f"'{ast.unparse(node)}' is not supported; only math.pi, math.e "
                "and math.tau are",
                node.lineno,
            )
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.UAdd):
                self.expr(node.operand, assigned)
            elif isinstance(node.op, ast.USub):
                self.expr(node.operand, assigned)
                self.emit(OP_NEG, 0, node)
            elif isinstance(node.op, ast.Not):
                self.expr(node.operand, assigned)
                self.emit(OP_NOT, 0, node)
            else:
                raise ForceFieldScriptError("bitwise operators are not supported", node.lineno)
            return
        if isinstance(node, ast.BinOp):
            self.binop(node.op, node, assigned)
            if isinstance(node.op, ast.Pow):
                n = _constant_int(node.right)
                if n is not None and -4 <= n <= 4:
                    self.small_pow(node.left, n, assigned, node)
                    return
            self.expr(node.left, assigned)
            self.expr(node.right, assigned)
            self.emit_binop(node.op, node)
            return
        if isinstance(node, ast.BoolOp):
            opcode = OP_AND if isinstance(node.op, ast.And) else OP_OR
            self.expr(node.values[0], assigned)
            for v in node.values[1:]:
                self.expr(v, assigned)
                self.emit(opcode, 0, node)
            return
        if isinstance(node, ast.Compare):
            left = node.left
            for i, (op, right) in enumerate(zip(node.ops, node.comparators)):
                if type(op) not in _COMPARE:
                    raise ForceFieldScriptError(
                        f"the comparison '{type(op).__name__}' is not supported", node.lineno
                    )
                self.expr(left, assigned)
                self.expr(right, assigned)
                self.emit(_COMPARE[type(op)], 0, node)
                if i > 0:
                    self.emit(OP_AND, 0, node)
                left = right
            return
        if isinstance(node, ast.IfExp):
            self.expr(node.test, assigned)
            jf = self.emit(OP_JUMP_IF_FALSE, 0, node)
            self.expr(node.body, assigned)
            jend = self.emit(OP_JUMP, 0, node)
            self.patch(jf, len(self.code))
            self.expr(node.orelse, assigned)
            self.patch(jend, len(self.code))
            return
        if isinstance(node, ast.Call):
            self.call(node, assigned)
            return
        raise ForceFieldScriptError(
            f"'{type(node).__name__}' expressions are not supported", node.lineno
        )

    def small_pow(self, base, n: int, assigned: set, node) -> None:
        if n == 0:
            self.const(1.0, node)
            return
        self.expr(base, assigned)
        if abs(n) > 1:
            t = self.temp(node)
            self.emit(OP_STORE, t, node)
            self.emit(OP_LOAD, t, node)
            for _ in range(abs(n) - 1):
                self.emit(OP_LOAD, t, node)
                self.emit(OP_MUL, 0, node)
        if n < 0:
            t2 = self.temp(node)
            self.emit(OP_STORE, t2, node)
            self.const(1.0, node)
            self.emit(OP_LOAD, t2, node)
            self.emit(OP_DIV, 0, node)

    # --- noise builtins -------------------------------------------------
    @staticmethod
    def is_vector_call(node) -> bool:
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in VECTOR_BUILTINS)

    NOISE_SIGNATURE = "(x, y, z, octaves=1, seed=0, time=0.0, frequency=1.0, decay=0.0)"

    def noise_args(self, node, assigned: set) -> tuple:
        """Push x, y, z, w and the seed of a noise call; return the octaves
        and the bound arguments, which :meth:`noise_fade` reads."""
        name = node.func.id
        params = ["x", "y", "z", "octaves", "seed", "time", "frequency", "decay"]
        if len(node.args) > len(params):
            raise ForceFieldScriptError(
                f"{name} takes {self.NOISE_SIGNATURE}", node.lineno)
        bound = dict(zip(params, node.args))
        for kw in node.keywords:
            if kw.arg not in params or kw.arg in bound:
                raise ForceFieldScriptError(
                    f"{name} takes {self.NOISE_SIGNATURE}; "
                    f"{kw.arg!r} is not one of them or is given twice", node.lineno)
            bound[kw.arg] = kw.value
        for p in ("x", "y", "z"):
            if p not in bound:
                raise ForceFieldScriptError(f"{name} needs {p}", node.lineno)
        for p in ("frequency", "decay"):
            if p in bound and "time" not in bound:
                raise ForceFieldScriptError(
                    f"{name}'s {p} acts over time, so it needs time too, "
                    f"usually time=t", node.lineno)
        octaves = 1
        if "octaves" in bound:
            octaves = _constant_int(bound["octaves"])
            if octaves is None or not 1 <= octaves <= MAX_OCTAVES:
                raise ForceFieldScriptError(
                    f"{name}'s octaves must be a whole number from 1 to "
                    f"{MAX_OCTAVES} written in the script", node.lineno)
        for p in ("x", "y", "z"):
            self.expr(bound[p], assigned)
        # w = time * frequency: how far the pattern has evolved.
        if "time" in bound:
            self.expr(bound["time"], assigned)
            if "frequency" in bound:
                self.expr(bound["frequency"], assigned)
                self.emit(OP_MUL, 0, node)
        else:
            self.const(0.0, node)
        if "seed" in bound:
            self.expr(bound["seed"], assigned)
        else:
            self.const(0.0, node)
        return octaves, bound

    def noise_fade(self, node, bound: dict, assigned: set) -> bool:
        """Push ``exp(-decay * time)`` when the call has a decay; say whether
        it did."""
        if "decay" not in bound:
            return False
        self.expr(bound["decay"], assigned)
        self.expr(bound["time"], assigned)
        self.emit(OP_MUL, 0, node)
        self.emit(OP_NEG, 0, node)
        self.emit(MATH_FUNCTIONS["exp"][0], 0, node)
        return True

    def vector_call(self, node, assigned: set) -> None:
        octaves, bound = self.noise_args(node, assigned)
        self.emit(OP_CURL, octaves, node)
        if self.noise_fade(node, bound, assigned):
            # Stack: cx, cy, cz, fade. Scale all three by the fade.
            fade = self.temp(node)
            self.emit(OP_STORE, fade, node)
            parts = [self.temp(node) for _ in range(3)]
            for t in reversed(parts):
                self.emit(OP_STORE, t, node)
            for t in parts:
                self.emit(OP_LOAD, t, node)
                self.emit(OP_LOAD, fade, node)
                self.emit(OP_MUL, 0, node)

    def call(self, node, assigned: set) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in VECTOR_BUILTINS:
            raise ForceFieldScriptError(
                f"{node.func.id} returns three numbers; return it, or unpack it "
                "into three names, like cx, cy, cz = curl_noise(x, y, z)",
                node.lineno,
            )
        if isinstance(node.func, ast.Name) and node.func.id == "noise":
            octaves, bound = self.noise_args(node, assigned)
            self.emit(OP_NOISE, octaves, node)
            if self.noise_fade(node, bound, assigned):
                self.emit(OP_MUL, 0, node)
            return
        if node.keywords:
            raise ForceFieldScriptError("keyword arguments are not supported", node.lineno)
        func = node.func
        args = node.args
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "math":
            spec = MATH_FUNCTIONS.get(func.attr)
            if spec is None:
                raise ForceFieldScriptError(
                    f"math.{func.attr} is not supported; the supported functions are "
                    + ", ".join(sorted(MATH_FUNCTIONS)),
                    node.lineno,
                )
            opcode, arity = spec
            if len(args) != arity:
                raise ForceFieldScriptError(
                    f"math.{func.attr} takes {arity} argument(s)", node.lineno
                )
            for a in args:
                self.expr(a, assigned)
            self.emit(opcode, 0, node)
            return
        if isinstance(func, ast.Name):
            if func.id == "abs" and len(args) == 1:
                self.expr(args[0], assigned)
                self.emit(OP_ABS, 0, node)
                return
            if func.id == "float" and len(args) == 1:
                self.expr(args[0], assigned)
                return
            if func.id in ("min", "max") and len(args) >= 2:
                opcode = OP_MIN if func.id == "min" else OP_MAX
                self.expr(args[0], assigned)
                for a in args[1:]:
                    self.expr(a, assigned)
                    self.emit(opcode, 0, node)
                return
        calls = [sig.split("(")[0] for _, entries in _script_api.SECTIONS
                 if entries is not _script_api.MATH_CONSTANTS
                 for _, sig, _ in entries if not sig.startswith("for ")]
        raise ForceFieldScriptError(
            f"the call '{ast.unparse(node)}' is not supported; the built-in "
            "functions are " + ", ".join(calls),
            node.lineno,
        )


def struct_f32(v: float) -> bytes:
    import struct

    return struct.pack("<f", v)


def _constant_int(node) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.Constant) and isinstance(node.value, float) and node.value.is_integer():
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _constant_int(node.operand)
        return -v if v is not None else None
    return None


def _max_stack(code: list) -> int:
    """The largest stack depth any path reaches (the solver re-proves it)."""
    depth = [None] * (len(code) + 1)
    depth[0] = 0
    peak = 0
    for pc, word in enumerate(code):
        here = depth[pc]
        if here is None:
            continue
        opcode, arg = word & 0xFF, word >> 8
        if opcode in (OP_CONST, OP_LOAD):
            after = here + 1
        elif opcode == OP_NOISE:
            after = here - 4
        elif opcode == OP_CURL:
            after = here - 2
        elif opcode in (OP_STORE, OP_JUMP_IF_FALSE):
            after = here - 1
        elif opcode == OP_JUMP:
            after = here
        elif opcode == OP_RETURN:
            continue
        elif opcode in (OP_NEG, OP_NOT) or 32 <= opcode <= 46:
            after = here
        else:
            after = here - 1
        peak = max(peak, after, here)
        if opcode in (OP_JUMP, OP_JUMP_IF_FALSE):
            depth[arg] = after
            if opcode == OP_JUMP_IF_FALSE:
                depth[pc + 1] = after
        else:
            depth[pc + 1] = after
    return peak


def _find_function(tree: ast.Module) -> ast.FunctionDef:
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            if isinstance(node, ast.ImportFrom) or names != ["math"]:
                raise ForceFieldScriptError("only 'import math' is allowed", node.lineno)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        else:
            raise ForceFieldScriptError(
                "the script holds one function and, optionally, 'import math'",
                node.lineno,
            )
    named = [f for f in functions if f.name == "eval"]
    if len(named) == 1:
        return named[0]
    if len(functions) == 1:
        return functions[0]
    raise ForceFieldScriptError(
        "define exactly one function, def eval(x, y, z, t)", None
    )


def compile_script(
    source,
    *,
    z_up: bool = False,
    check: bool = True,
    check_box: Optional[Sequence[Sequence[float]]] = None,
) -> CompiledScript:
    """Compile a force-field script into solver bytecode.

    Args:
        source: The script text, or a Python function (its source is read with
            ``inspect.getsource``).
        z_up: Read and return vectors in Blender's Z-up axes rather than the
            solver's Y-up ones. The Blender add-on sets this.
        check: Cross-check the bytecode against the original Python at 256
            seeded points, refusing a compile whose answers disagree.
        check_box: ``((xmin, ymin, zmin), (xmax, ymax, zmax))`` for the
            cross-check points, in the script's own axes. Defaults to the
            cube ``[-1, 1]^3``.

    Raises:
        ForceFieldScriptError: naming the line and the construct.
    """
    if callable(source):
        import inspect

        try:
            source = inspect.getsource(source)
        except (OSError, TypeError):
            raise ForceFieldScriptError(
                f"the source of {getattr(source, '__name__', source)!r} cannot be "
                "read (it was defined at an interactive prompt or through exec); "
                "pass the function's source text instead"
            ) from None
    source = textwrap.dedent(str(source))
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ForceFieldScriptError(f"syntax error: {e.msg}", e.lineno) from None
    fn = _find_function(tree)
    compiler = _Compiler(z_up=z_up)
    compiler.compile_function(fn)
    compiled = CompiledScript(
        code=compiler.code,
        constants=compiler.constants,
        source=source,
        max_stack=_max_stack(compiler.code),
    )
    if compiled.max_stack > STACK_CAPACITY:
        raise ForceFieldScriptError(
            f"the expression nesting needs a stack of {compiled.max_stack}, "
            f"past the solver's {STACK_CAPACITY}; split it with local variables",
            fn.lineno,
        )
    if check:
        _cross_check(compiled, source, fn.name, z_up, check_box)
    return compiled


# --- the reference interpreter --------------------------------------------


def run_bytecode(code: list, constants: list, x: float, y: float, z: float, t: float):
    """Run bytecode in float64, mirroring the kernel's semantics.

    Used by the cross-check and by the rig. The kernel runs the same program
    in float32, so it agrees to single precision.
    """
    stack: list = []
    vars_ = [0.0] * VAR_CAPACITY
    vars_[0], vars_[1], vars_[2], vars_[3] = x, y, z, t
    pc = 0
    n = len(code)
    while pc < n:
        word = code[pc]
        op, arg = word & 0xFF, word >> 8
        pc += 1
        if op == OP_CONST:
            stack.append(constants[arg])
        elif op == OP_LOAD:
            stack.append(vars_[arg])
        elif op == OP_STORE:
            vars_[arg] = stack.pop()
        elif op == OP_JUMP:
            pc = arg
        elif op == OP_JUMP_IF_FALSE:
            if stack.pop() == 0.0:
                pc = arg
        elif op == OP_RETURN:
            return (stack[-3], stack[-2], stack[-1])
        elif op in (OP_NOISE, OP_CURL):
            seed = int(stack.pop())
            nw = stack.pop()
            nz = stack.pop()
            ny = stack.pop()
            nx = stack.pop()
            if op == OP_NOISE:
                stack.append(_noise.noise(nx, ny, nz, arg, seed, time=nw))
            else:
                stack.extend(_noise.curl_noise(nx, ny, nz, arg, seed, time=nw))
        elif op in (OP_NEG, OP_NOT) or 32 <= op <= 46:
            a = stack.pop()
            stack.append(_unary(op, a))
        else:
            b = stack.pop()
            a = stack.pop()
            stack.append(_binary(op, a, b))
    raise RuntimeError("the program ran past its end")


def _floor(a):
    return float(math.floor(a)) if math.isfinite(a) else a


def _unary(op, a):
    try:
        if op == OP_NEG:
            return -a
        if op == OP_NOT:
            return 1.0 if a == 0.0 else 0.0
        f = {
            32: math.sqrt, 33: math.sin, 34: math.cos, 35: math.tan,
            36: math.exp, 37: math.log, 38: abs, 39: _floor,
            40: lambda v: -_floor(-v), 41: math.tanh, 42: math.asin,
            43: math.acos, 44: math.atan, 45: math.sinh, 46: math.cosh,
        }[op]
        return float(f(a))
    except (ValueError, OverflowError):
        return float("nan")


def _binary(op, a, b):
    try:
        if op == OP_ADD:
            return a + b
        if op == OP_SUB:
            return a - b
        if op == OP_MUL:
            return a * b
        if op == OP_DIV:
            if b == 0.0:
                return float("nan") if a == 0.0 else math.copysign(float("inf"), a) * math.copysign(1.0, b)
            return a / b
        if op == OP_MOD:
            return a - b * _floor(a / b) if b != 0.0 else float("nan")
        if op == OP_POW:
            return float(a**b) if not (a < 0 and not float(b).is_integer()) else float("nan")
        if op == OP_LT:
            return 1.0 if a < b else 0.0
        if op == OP_LE:
            return 1.0 if a <= b else 0.0
        if op == OP_GT:
            return 1.0 if a > b else 0.0
        if op == OP_GE:
            return 1.0 if a >= b else 0.0
        if op == OP_EQ:
            return 1.0 if a == b else 0.0
        if op == OP_NE:
            return 1.0 if a != b else 0.0
        if op == OP_AND:
            return 1.0 if (a != 0.0 and b != 0.0) else 0.0
        if op == OP_OR:
            return 1.0 if (a != 0.0 or b != 0.0) else 0.0
        if op == 48:
            return math.atan2(a, b)
        if op == OP_MIN:
            return min(a, b)
        if op == OP_MAX:
            return max(a, b)
        if op == 51:
            return math.hypot(a, b)
    except (ValueError, OverflowError, ZeroDivisionError):
        return float("nan")
    raise RuntimeError(f"opcode {op} is not in the table")


def _cross_check(compiled, source, name, z_up, check_box):
    namespace: dict = {"math": math, "noise": _noise.noise,
                       "curl_noise": _noise.curl_noise}
    try:
        exec(compile(source, "<force field script>", "exec"), namespace)
    except Exception as e:  # the script's own top level failed
        raise ForceFieldScriptError(f"the script does not run as Python: {e}") from None
    fn = namespace[name]
    lo, hi = (check_box if check_box is not None else ((-1.0,) * 3, (1.0,) * 3))
    rng = random.Random(0x5EED)
    for _ in range(256):
        p = [rng.uniform(float(lo[k]), float(hi[k])) for k in range(3)]
        t = rng.uniform(0.0, 10.0)
        try:
            expected = fn(p[0], p[1], p[2], t)
        except (ArithmeticError, ValueError):
            continue
        if not (isinstance(expected, (tuple, list)) and len(expected) == 3):
            raise ForceFieldScriptError(
                f"the function returned {expected!r} at {p}, t={t:.3f}; "
                "return a tuple of three numbers"
            )
        # The solver hands the program its own axes; convert the point the
        # script saw into them, and the answer back.
        if z_up:
            solver = (p[0], p[2], -p[1])
        else:
            solver = (p[0], p[1], p[2])
        got = run_bytecode(compiled.code, compiled.constants, *solver, t)
        if z_up:
            got = (got[0], -got[2], got[1])
        for k in range(3):
            e = float(expected[k])
            g = float(got[k])
            if math.isnan(e) and math.isnan(g):
                continue
            if not math.isclose(e, g, rel_tol=1e-9, abs_tol=1e-9):
                raise ForceFieldScriptError(
                    "the compiled program disagrees with the Python function "
                    f"at ({p[0]:.4g}, {p[1]:.4g}, {p[2]:.4g}), t={t:.4g}: "
                    f"component {k} is {g!r} against {e!r}. This is a compiler "
                    "defect, please report it with the script"
                )


# --- field objects --------------------------------------------------------

#: The field types the solver evaluates.
FIELD_TYPES = ("force", "wind", "vortex", "turbulence", "magnetic", "drag", "harmonic",
               "texture", "fluid")
#: Types with a law measured but no solver support yet, refused by name.
FIELD_TYPES_PENDING: dict = {}
TEXTURE_MODES = ("rgb", "gradient", "curl")
TEXTURE_EXTENSIONS = ("extend", "repeat", "clip", "clip_cube")
FIELD_SHAPES = ("point", "plane")
FIELD_FALLOFFS = ("sphere", "tube", "cone")
FIELD_Z_DIRECTIONS = ("both", "positive", "negative")
#: A new Blender field's frame in solver axes (Y up): local Z points up.
DEFAULT_AXES = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
#: The continuous settings, each one value per track sample.
FIELD_SCALARS = ("strength", "power", "min_distance", "max_distance", "radial_power",
                 "radial_min", "radial_max", "size", "noise")
#: The per-sample values after the noise scale, in the solver's track order.
FIELD_TAIL = ("wind_factor", "linear_drag", "quadratic_drag", "harmonic_damping",
              "rest_length", "inflow", "nabla")


#: The most cells a fluid domain may have along one axis, as the solver's
#: loader takes it.
FLUID_MAX_CELLS = 1 << 15
_BRICK_CELLS = FLUID_BRICK ** 3
_AXIS_NAMES = ("x", "y", "z")
# Marks an iterator that ran out.
_END = object()


def _frame_digest(table, bricks) -> str:
    """The digest that ties a fluid frame to the files written for it."""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(table, dtype="<u4"))
    h.update(np.ascontiguousarray(bricks, dtype="<f4"))
    return h.hexdigest()


def _under(path: str, directory: str) -> bool:
    """Whether ``path`` lies inside ``directory``."""
    root = os.path.realpath(directory)
    try:
        return os.path.commonpath([root, os.path.realpath(path)]) == root
    except ValueError:  # different drives
        return False


def _inside(place: tuple, directory: str) -> bool:
    """Whether either file of a frame's ``(table path, bricks path)`` lies
    inside ``directory``."""
    return _under(place[0], directory) or _under(place[1], directory)


def _same_place(a: tuple, b: tuple) -> bool:
    """Whether two ``(table path, bricks path)`` name the same files."""
    return all(os.path.realpath(p) == os.path.realpath(q) for p, q in zip(a, b, strict=True))


def _read_only(array):
    """``array`` with writing turned off: a frame never changes once a volume
    holds it, because a pickle of the volume may name the files written for
    the frame in place of its values."""
    array.setflags(write=False)
    return array


def _nothing_else_writes(array, given) -> bool:
    """Whether no object but ``array`` itself can change its memory.

    Decided from the array's own base chain, whatever ``given`` was: an array
    that owns its memory qualifies unless it is the caller's own ndarray (or
    shares its memory); a view qualifies only when every array along its
    chain is read-only and the chain ends in an immutable ``bytes`` (the
    decoder's frames). Anything else, a view of an ``array.array``, of a
    ``bytearray`` or of a writable ndarray, can still be written by whoever
    holds that buffer."""
    if array.base is None:
        return not (isinstance(given, np.ndarray) and np.may_share_memory(array, given))
    base = array
    while isinstance(base, np.ndarray):
        if base.flags.writeable:
            return False
        base = base.base
    return isinstance(base, bytes)


def _private(array, given):
    """``array``, made from the caller's ``given``, as a read-only array
    nothing else can write: itself when :func:`_nothing_else_writes`,
    otherwise a copy."""
    if not _nothing_else_writes(array, given):
        array = array.copy()
    return _read_only(array)


# Bytes read at a time when a file is compared with a frame.
_COMPARE_CHUNK = 1 << 24


def _file_holds(path: str, array) -> bool:
    """Whether the file at ``path`` holds exactly the bytes of ``array``, a
    contiguous array in the byte order it was written in."""
    expected = array.reshape(-1).view(np.uint8)
    try:
        if os.path.getsize(path) != expected.size:
            return False
        chunk = np.empty(min(expected.size, _COMPARE_CHUNK), dtype=np.uint8)
        at = 0
        with open(path, "rb") as f:
            while at < expected.size:
                n = f.readinto(chunk)
                if not n:
                    return False
                if not np.array_equal(chunk[:n], expected[at:at + n]):
                    return False
                at += n
    except OSError:
        return False
    return True


class _FluidFrames:
    """A fluid volume's frames: per frame the brick table and the kept bricks.

    A frame is READ-ONLY once the volume holds it. A frame cut in this
    process is held in memory. Every time the session writer writes it
    (:func:`write_session`) the frame records the files, so it knows every
    place a copy was written, in every session or export directory. A pickle
    of the volume (a saved scene, session or app state) carries those places
    in place of the bricks when one of them still holds the frame byte for
    byte, and carries the bricks when none does, so the sequence is on disk
    once. A frame read back that way is loaded when it is asked for, from the
    most recently written place that still holds it, checked against the
    sizes and the digest recorded when it was written; when no place does,
    it fails by name, listing them, never reading as anything else.
    """

    #: Frames read back from files that stay loaded: the two a step blends.
    CACHED = 2

    def __init__(self, tables: list, bricks: list):
        # The callers hand over arrays nothing else holds (`_private`).
        self._memory = [(_read_only(t), _read_only(b))
                        for t, b in zip(tables, bricks, strict=True)]
        n = len(self._memory)
        # Per frame, once it is written: (table bytes, bricks bytes, digest),
        # and every (table path, bricks path) it was written to, oldest first.
        self._written: list = [None] * n
        self._places: list = [[] for _ in range(n)]
        self._init_transient()

    def _init_transient(self) -> None:
        # Frames loaded only to survive the directory of their every copy
        # being cleared (`clear`), released again once they are written anew.
        self._held: set = set()
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self._memory)

    def frame(self, k: int):
        """Frame ``k`` as ``(table, bricks)``, both read-only."""
        k = range(len(self))[k]
        resident = self._memory[k]
        if resident is not None:
            return resident
        loaded = self._cache.get(k)
        if loaded is None:
            loaded = self._load(k)
            while len(self._cache) >= self.CACHED:
                self._cache.pop(next(iter(self._cache)))
            self._cache[k] = loaded
        return loaded

    def _read(self, k: int, place: tuple):
        """Frame ``k`` read from ``place``: ``(frame, None)``, or ``(None,
        why not)`` when the files are gone or hold anything else."""
        table_bytes, bricks_bytes, digest = self._written[k]
        for path, size in zip(place, (table_bytes, bricks_bytes), strict=True):
            try:
                actual = os.path.getsize(path)
            except OSError:
                return None, f"{path} is gone"
            if actual != size:
                return None, f"{path} holds {actual} bytes, not {size}"
        table = np.fromfile(place[0], dtype="<u4")
        bricks = np.fromfile(place[1], dtype="<f4").reshape(-1, _BRICK_CELLS, 4)
        if _frame_digest(table, bricks) != digest:
            return None, f"{place[0]} and {place[1]} hold other values"
        return (_read_only(table), _read_only(bricks)), None

    def _load(self, k: int, outside: Optional[str] = None):
        """Frame ``k`` from the most recently written place that still holds
        it. With ``outside``, only a place not inside that directory is
        read, and None is returned when none of those holds it."""
        reasons = []
        for place in reversed(self._places[k]):
            if outside is not None and _inside(place, outside):
                continue
            frame, reason = self._read(k, place)
            if frame is not None:
                return frame
            reasons.append(reason)
        if outside is not None:
            return None
        count = len(self._places[k])
        message = (
            f"fluid frame {k} was written to {count} place{'s' if count != 1 else ''}, "
            f"and none holds it any more: {'; '.join(reasons) or 'no place was recorded'}. "
            "A directory that held it was deleted, rebuilt or updated since, or its "
            "files were changed; build the FluidVolume again"
        )
        if reasons and all(r.endswith(" is gone") for r in reasons):
            raise FileNotFoundError(message)
        raise ValueError(message)

    def _holds(self, place: tuple, frame) -> bool:
        """Whether both files of ``place`` hold resident ``frame`` exactly."""
        table, bricks = frame
        return (_file_holds(place[0], np.ascontiguousarray(table, dtype="<u4"))
                and _file_holds(place[1], np.ascontiguousarray(bricks, dtype="<f4")))

    def bind(self, k: int, table_path: str, bricks_path: str, table, bricks) -> None:
        """Record that frame ``k`` was written as ``table`` and ``bricks``
        (the ``<u4`` and ``<f4`` arrays exactly as written) to these files."""
        if self._written[k] is None:
            # A frame never changes (it is read-only), so its digest is taken
            # once, at its first write.
            self._written[k] = (int(table.nbytes), int(bricks.nbytes),
                                _frame_digest(table, bricks))
        place = (os.path.abspath(table_path), os.path.abspath(bricks_path))
        self._places[k] = [p for p in self._places[k] if not _same_place(p, place)]
        self._places[k].append(place)
        if k in self._held:
            self._held.discard(k)
            self._memory[k] = None
        self._cache.pop(k, None)

    def has_copy_under(self, directory: str) -> bool:
        """Whether a frame was written to a place inside ``directory``."""
        return any(_inside(p, directory) for places in self._places for p in places)

    def plan_clear(self, directory: str) -> dict:
        """Check, before ``directory`` is cleared, that every frame stays
        readable without the copies inside it, changing nothing: a frame read
        back from files is loaded now from a copy outside the directory, or
        else from one inside it, and raises when no copy holds it, so a frame
        lost already fails before anything is cleared. Returns the frames to
        keep in memory, those with no copy outside, for :meth:`clear`."""
        keep = {}
        for k in range(len(self)):
            if self._memory[k] is not None:
                continue
            if any(_inside(p, directory) for p in self._places[k]):
                if self._load(k, outside=directory) is None:
                    keep[k] = self._load(k)
            else:
                self._load(k)
        return keep

    def clear(self, directory: str, keep: dict) -> None:
        """Forget the copies inside ``directory``, which is being cleared,
        and keep in memory the frames :meth:`plan_clear` found no other copy
        of, until they are written anew."""
        for k in range(len(self)):
            self._places[k] = [p for p in self._places[k] if not _inside(p, directory)]
        for k, frame in keep.items():
            self._memory[k] = frame
            self._held.add(k)
            self._cache.pop(k, None)

    def __getstate__(self):
        frames = []
        for k, resident in enumerate(self._memory):
            places = self._places[k]
            if resident is None:
                frames.append(("files", list(places), self._written[k]))
                continue
            # A resident frame refers to its files only when they hold it
            # byte for byte: another volume, or a frame of this one, may have
            # been written to the same paths since.
            holding = next((p for p in reversed(places) if self._holds(p, resident)), None)
            if holding is None:
                frames.append(("memory", resident[0], resident[1]))
            else:
                # The copy just compared is read first.
                ordered = [p for p in places if p is not holding] + [holding]
                frames.append(("files", ordered, self._written[k]))
        return {"frames": frames}

    def __setstate__(self, state):
        self._memory, self._written, self._places = [], [], []
        for frame in state["frames"]:
            if frame[0] == "files":
                self._memory.append(None)
                self._places.append([tuple(p) for p in frame[1]])
                self._written.append(tuple(frame[2]))
            else:
                self._memory.append((_read_only(frame[1]), _read_only(frame[2])))
                self._places.append([])
                self._written.append(None)
        self._init_transient()


class _FrameColumn:
    """The tables or the bricks of a volume's frames, read by frame index
    like the list they stand for."""

    def __init__(self, frames: _FluidFrames, index: int):
        self._frames = frames
        self._index = index

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, k: int):
        return self._frames.frame(k)[self._index]

    def __iter__(self):
        return (self[k] for k in range(len(self)))


def _check_resolution(k: int, r) -> tuple:
    """Frame ``k``'s resolution as three ints, ``(0, 0, 0)`` for an empty
    frame, or refused."""
    r = tuple(int(v) for v in r)
    if len(r) != 3 or min(r) < 0 or max(r) > FLUID_MAX_CELLS:
        raise ValueError(
            f"fluid frame {k}: a domain of {r} cells; the solver takes 0 to "
            f"{FLUID_MAX_CELLS} along each axis"
        )
    return (0, 0, 0) if min(r) == 0 else r


def _per_axis_cell(cell, n: int):
    """``cell=`` as ``(n, 3)``: one number, one per axis, one per frame, or one
    per axis and frame."""
    c = np.asarray(cell, dtype=np.float64)
    if c.shape == ():
        return np.full((n, 3), float(c))
    if c.shape == (3,) and n == 3:
        raise ValueError(
            "cell= of shape (3,) is ambiguous for a volume of 3 frames, one size per "
            "axis or one per frame; give (3, 3), one (cx, cy, cz) per frame"
        )
    if c.shape == (3,):
        return np.repeat(c[None], n, axis=0)
    if c.shape == (n,):
        return np.repeat(c[:, None], 3, axis=1)
    if c.shape == (n, 3):
        return c.copy()
    raise ValueError(
        f"fluid cell has shape {c.shape}; give a number, (cx, cy, cz), one per frame "
        f"({n},), or ({n}, 3)"
    )


class FluidVolume:
    """A fluid's velocity, and optionally its density, on a grid, frame by
    frame: what a ``"fluid"`` field reads.

    Each frame's domain is a box of ``resolution`` cells, ``cell`` scene units
    across along each of its axes, its corner (cell (0, 0, 0)'s outer corner)
    at ``corner`` and its edges along the columns of ``axes`` (a rotation,
    solver axes): array index ``(i, j, k)`` runs along those columns. Values
    sit at cell centers; between them the solver interpolates trilinearly,
    and outside the box the fluid is absent, so nothing acts there, Flow's
    damping included. Velocities are scene units per second of the scene
    clock the frames are placed on, in the DOMAIN's axes; the solver scales
    them by World Scaling (``world-scaling``) together with the domain's
    lengths, and divides them by the step's Playback Speed (``playback``):
    a step moves the vertices by ``dt * playback`` seconds of motion while
    the clock advances ``dt``, so a vertex the fluid carries keeps pace with
    the fluid's frames at any playback.

    Frames are ``step`` seconds apart from ``start``. The solver reads the
    fluid at each step's end and blends the two frames bracketing it (Blender
    instead holds the frame it steps to for every substep); the end frame
    holds outside the sequence. The transform and the resolution may differ
    per frame (a moving domain, or a VDB sequence whose active box changes),
    and a frame may be EMPTY, of zero extent: no domain, so nothing acts.

    Each frame is cut into the solver's sparse bricks as it is read, slab by
    slab in its own dtype, so only the occupied part of the domain is kept in
    memory. ``velocity`` and ``density`` may be lists of arrays, lists of
    callables each returning one (called once), or iterators such as
    generators; they are read one frame at a time, so a long sequence can
    come from disk with one dense frame alive at once. The cut frames are
    read-only: to change the fluid, build another FluidVolume.

    Once a session is built, a saved scene, session or app refers to the
    brick files the session writer wrote rather than carrying the bricks
    again. A frame remembers every session or export directory it was
    written to, and a save refers to them only when one still holds the
    frame byte for byte; otherwise it carries the bricks. A copy read back
    from a save reads each frame from the most recently written directory
    that still holds it, and fails naming every one when none does.

    Args:
        velocity: The frames' velocities, each ``(nx, ny, nz, 3)``.
        cell: The cell size: a number, ``(cx, cy, cz)`` along the domain's
            axes, one number per frame ``(F,)``, or ``(F, 3)``. With exactly
            3 frames, ``(3,)`` is refused as ambiguous.
        corner: Cell (0, 0, 0)'s outer corner, ``(3,)`` or one per frame
            ``(F, 3)``, solver axes, scene units. The solver refuses a corner
            with a coordinate beyond 1e6 solver units (scene units times World
            Scaling).
        axes: The domain's rotation, its columns the domain's axes in solver
            axes, ``(3, 3)`` or one per frame ``(F, 3, 3)``; None is the
            identity.
        density: None, or one density frame per velocity frame, each
            ``(nx, ny, nz)``. ``use_density`` on the field needs it.
        start: The time of the first frame, seconds.
        step: The seconds between frames, needed with more than one.

    Attributes:
        resolution (list): Per frame ``(nx, ny, nz)``, ``(0, 0, 0)`` for an
            empty frame; at most 32768 along an axis.
        cell (numpy.ndarray): ``(F, 3)``, the cell size along each axis.
        corner (numpy.ndarray): ``(F, 3)``.
        axes (numpy.ndarray): ``(F, 3, 3)``, the axes as columns.
        start (float): The first frame's time, seconds.
        step (float): The seconds between frames.
        has_density (bool): Whether the volume holds a density.

    Raises:
        ValueError: A frame's shape or size, a value that is not finite as
            the float32 the solver reads (naming the frame and the cell
            ``(i, j, k)``), a cell, corner or axes setting, or a frame count
            the solver would refuse.

    Example:
        A uniform breeze along X through a 2 m box, with smoke only in its
        lower half (solver Y is up)::

            n = 16
            u = np.zeros((n, n, n, 3)); u[..., 0] = 2.0
            rho = np.zeros((n, n, n)); rho[:, : n // 2, :] = 1.0
            volume = FluidVolume([u], cell=2.0 / n, corner=(-1, -1, -1),
                                 density=[rho])
            scene.force_field.add("fluid", fluid=volume, strength=5.0,
                                  flow=1.0, use_density=True)
    """

    def __init__(self, velocity, *, cell, corner=(0.0, 0.0, 0.0), axes=None,
                 density=None, start: float = 0.0, step: Optional[float] = None):
        for name, value, ndim in (("velocity", velocity, 4), ("density", density, 3)):
            if isinstance(value, np.ndarray) and value.ndim == ndim:
                raise ValueError(
                    f"{name}= takes a sequence of frames, and this is one frame; "
                    f"give [{name}]"
                )
        if (density is not None and hasattr(velocity, "__len__")
                and hasattr(density, "__len__") and len(density) != len(velocity)):
            raise ValueError(
                f"{len(density)} density frames for {len(velocity)} velocity frames"
            )
        frames = iter(velocity)
        densities = None if density is None else iter(density)
        tables, bricks, resolution = [], [], []
        # One frame at a time, and no reference to it kept past its cut, so a
        # generator's dense frames are alive one at a time.
        k = 0
        while True:
            frame = next(frames, _END)
            if frame is _END:
                break
            d = None
            if densities is not None:
                d = next(densities, _END)
                if d is _END:
                    raise ValueError(
                        f"the fluid's density frames end at frame {k}, before its "
                        "velocity frames; give one density frame per velocity frame"
                    )
            u = np.asarray(frame() if callable(frame) else frame)
            rho = None if d is None else np.asarray(d() if callable(d) else d)
            frame = d = None
            try:
                table, brick = fluid_bricks(u, rho)
            except (TypeError, ValueError) as e:
                raise type(e)(f"fluid frame {k}: {e}") from None
            resolution.append(u.shape[:3])
            u = rho = None
            tables.append(table)
            bricks.append(brick)
            k += 1
        if not tables:
            raise ValueError("a fluid volume needs at least one velocity frame")
        if densities is not None and next(densities, _END) is not _END:
            raise ValueError(
                f"the fluid has more density frames than its {len(tables)} velocity "
                "frames; give one density frame per velocity frame"
            )
        self._set(resolution, tables, bricks, cell, corner, axes, start, step,
                  has_density=density is not None)

    @classmethod
    def from_bricks(cls, resolution, tables, bricks, *, cell, has_density: bool,
                    corner=(0.0, 0.0, 0.0), axes=None, start: float = 0.0,
                    step: Optional[float] = None) -> "FluidVolume":
        """A volume from frames already cut into the solver's bricks, as
        ``frontend._field_eval_.fluid_bricks`` cuts them, validated as the
        solver's loader validates them, so a malformed frame is refused here.
        The volume keeps read-only arrays of its own: an array the caller can
        still write is copied, and a read-only view of ``bytes`` is kept.

        Args:
            resolution: Integers: one ``(nx, ny, nz)`` for every frame, or one
                per frame ``(F, 3)``. A zero extent is an empty frame.
            tables: Per frame, the brick table: one dimension of integers,
                each the slot of a kept brick or ``FLUID_EMPTY``
                (0xFFFFFFFF), one entry per brick of the domain, x fastest.
            bricks: Per frame, the kept bricks, ``(count, 512, 4)`` or as many
                floats flat, cells x fastest, channels velocity then density.
            cell: As for the constructor.
            has_density: Whether the bricks' density channel holds a density,
                which ``use_density`` needs; the channel is zero otherwise.
            corner: As for the constructor.
            axes: As for the constructor.
            start: As for the constructor.
            step: As for the constructor.

        Raises:
            TypeError: A table that is not integers, or bricks that are not
                real numbers.
            ValueError: A resolution, table, brick or setting the solver would
                refuse, naming the frame.
        """
        if not isinstance(has_density, (bool, np.bool_)):
            raise TypeError(f"has_density= takes a bool, got {has_density!r}")
        tables = list(tables)
        bricks = list(bricks)
        n = len(tables)
        if len(bricks) != n or not n:
            raise ValueError(f"{n} brick tables and {len(bricks)} brick sets")
        res = np.asarray(resolution)
        if res.dtype.kind not in "iu" or res.shape not in ((3,), (n, 3)):
            raise ValueError(
                f"resolution= has shape {res.shape} of {res.dtype}; give integers "
                f"(nx, ny, nz) for every frame, or one per frame ({n}, 3)"
            )
        if res.shape == (3,):
            res = np.repeat(res[None], n, axis=0)
        given_res = [tuple(int(v) for v in r) for r in res]
        res = [_check_resolution(k, r) for k, r in enumerate(res)]
        for k in range(n):
            given_table, given_bricks = tables[k], bricks[k]
            t = np.asarray(tables[k])
            if t.dtype.kind not in "iu" or t.ndim != 1:
                raise TypeError(
                    f"fluid frame {k}: a brick table of {t.dtype} with shape {t.shape}; "
                    "give one dimension of integers"
                )
            b = np.asarray(bricks[k])
            if b.dtype.kind not in "iuf":
                raise TypeError(f"fluid frame {k}: bricks of {b.dtype}; give real numbers")
            if not ((b.ndim == 1 and b.size % (_BRICK_CELLS * 4) == 0)
                    or (b.ndim == 3 and b.shape[1:] == (_BRICK_CELLS, 4))):
                raise ValueError(
                    f"fluid frame {k}: bricks of shape {b.shape}; give "
                    f"(count, {_BRICK_CELLS}, 4) or as many floats flat"
                )
            with np.errstate(over="ignore", invalid="ignore"):
                b = np.asarray(b, dtype=np.float32).reshape(-1, _BRICK_CELLS, 4)
            nb = [(r + FLUID_BRICK - 1) // FLUID_BRICK for r in res[k]]
            if len(t) != nb[0] * nb[1] * nb[2]:
                raise ValueError(
                    f"fluid frame {k}: {len(t)} table entries for the {nb} bricks of a "
                    f"{res[k]} domain"
                )
            if res[k] == (0, 0, 0) and len(b):
                raise ValueError(
                    f"fluid frame {k}: an empty frame (a {list(given_res[k])} domain) "
                    f"holds {len(b)} bricks; it holds none"
                )
            v = t.astype(np.int64)
            bad = np.flatnonzero((v != FLUID_EMPTY) & ((v < 0) | (v >= len(b))))
            if len(bad):
                raise ValueError(
                    f"fluid frame {k}: table entry {int(bad[0])} is {int(v[bad[0]])}, which "
                    f"names none of the {len(b)} bricks the frame holds"
                )
            finite = np.isfinite(b)
            if not finite.all():
                slot, c, ch = (int(x) for x in np.argwhere(~finite)[0])
                name = f"velocity component {ch}" if ch < 3 else "density"
                where = np.flatnonzero(v == slot)
                if len(where):
                    p = int(where[0])
                    brick = (p % nb[0], (p // nb[0]) % nb[1], p // (nb[0] * nb[1]))
                    local = (c % FLUID_BRICK, (c // FLUID_BRICK) % FLUID_BRICK,
                             c // FLUID_BRICK ** 2)
                    cell_at = tuple(FLUID_BRICK * brick[a] + local[a] for a in range(3))
                    at = f"cell {cell_at}"
                else:
                    at = f"brick {slot}, which no table entry names,"
                raise ValueError(
                    f"fluid frame {k}: the {name} at {at} is {float(b[slot, c, ch])}, "
                    "not finite as the float32 the solver reads"
                )
            # The volume's own read-only arrays: a caller's later edit of what
            # it passed cannot reach a frame.
            tables[k] = _private(t.astype(np.uint32, copy=False), given_table)
            bricks[k] = _private(b, given_bricks)
        self = cls.__new__(cls)
        self._set(res, tables, bricks, cell, corner, axes, start, step,
                  has_density=bool(has_density))
        return self

    def _set(self, resolution, tables, bricks, cell, corner, axes, start, step,
             has_density: bool):
        n = len(tables)
        resolution = [_check_resolution(k, r) for k, r in enumerate(resolution)]

        def per_frame(name, value, shape):
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape == shape:
                arr = np.repeat(arr[None], n, axis=0)
            if arr.shape != (n,) + shape:
                raise ValueError(f"fluid {name} has shape {arr.shape}; give {shape} or one per frame")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"fluid {name} is not finite")
            return arr

        cell = _per_axis_cell(cell, n)
        # The solver reads the cell, and divides by it, in float32: both the
        # size and its reciprocal must be normal floats, or every point reads
        # as inside the domain or as outside it.
        tiny = np.finfo(np.float32).tiny
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            c32 = cell.astype(np.float32)
            inverse = np.float32(1.0) / c32
        ok = (np.isfinite(cell) & (cell > 0.0) & np.isfinite(c32) & (c32 >= tiny)
              & np.isfinite(inverse) & (inverse >= tiny))
        if not ok.all():
            k, a = (int(v) for v in np.argwhere(~ok)[0])
            raise ValueError(
                f"fluid frame {k}: a cell size of {float(cell[k, a])!r} along local "
                f"{_AXIS_NAMES[a]}; the solver reads it and its reciprocal as normal "
                f"float32 values, so give a size from {float(tiny):.3g} to "
                f"{float(1.0 / tiny):.3g}"
            )
        self.cell = cell
        self.corner = per_frame("corner", corner, (3,))
        self.axes = per_frame("axes", np.eye(3) if axes is None else axes, (3, 3))
        for k, a in enumerate(self.axes):
            if not np.allclose(a.T @ a, np.eye(3), atol=1e-5) or np.linalg.det(a) < 0.0:
                raise ValueError(f"fluid axes at frame {k} are not a rotation")
        if n > 1 and (step is None or not (math.isfinite(float(step)) and float(step) > 0.0)):
            raise ValueError(f"{n} fluid frames need step=, the seconds between them, positive")
        if not math.isfinite(float(start)):
            raise ValueError(f"fluid start {start} is not finite")
        self.resolution = resolution
        self.start = float(start)
        self.step = float(step) if step is not None else 1.0
        self.has_density = bool(has_density)
        self._frames = _FluidFrames(tables, bricks)

    @property
    def frames(self) -> int:
        """The number of frames."""
        return len(self._frames)

    @property
    def tables(self) -> _FrameColumn:
        """Per frame, the brick table, indexed by frame: read-only uint32,
        one entry per brick of the domain, x fastest, each the slot of a kept
        brick or ``FLUID_EMPTY``."""
        return _FrameColumn(self._frames, 0)

    @property
    def bricks(self) -> _FrameColumn:
        """Per frame, the kept bricks, indexed by frame: read-only float32
        ``(count, 512, 4)``, cells x fastest, channels velocity then
        density."""
        return _FrameColumn(self._frames, 1)

    def as_dict(self) -> dict:
        """The volume as ``frontend._field_eval_`` reads it."""
        return {"resolution": self.resolution, "cell": self.cell, "corner": self.corner,
                "axes": self.axes, "start": self.start, "step": self.step,
                "has_density": self.has_density, "tables": self.tables,
                "bricks": self.bricks}


def _fluid_frames(fields) -> list:
    """The distinct frame stores of the fluids that ``fields`` (field
    records, or :func:`resolve_targets`'s ``(record, bit)`` pairs) read."""
    stores = {}
    for entry in fields:
        record = entry[0] if isinstance(entry, tuple) else entry
        volume = record.get("fluid") if isinstance(record, dict) else None
        column = volume.get("tables") if isinstance(volume, dict) else None
        if isinstance(column, _FrameColumn):
            stores[id(column._frames)] = column._frames
    return list(stores.values())


def fluid_copies_under(fields, directory: str) -> bool:
    """Whether a fluid frame that ``fields`` read was written to a place
    inside ``directory``, so a pickle of them may name files there."""
    return any(frames.has_copy_under(directory) for frames in _fluid_frames(fields))


def hold_fluid_frames(fields, directory: str) -> None:
    """Before ``directory`` is cleared, make every fluid frame that
    ``fields`` (field records, or :func:`resolve_targets`'s ``(record,
    bit)`` pairs) reads independent of the files inside it, so the session
    writer can write those frames again: a frame with no copy elsewhere is
    loaded into memory, and the copies inside are forgotten.

    Every frame read back from files is checked first, all of them before
    any is changed, so a frame that no copy holds any more raises here,
    before anything is cleared, rather than in the writer afterwards.

    Raises:
        FileNotFoundError: A frame's every copy is gone.
        ValueError: A frame's every copy is gone or holds other values.
    """
    plans = [(frames, frames.plan_clear(directory)) for frames in _fluid_frames(fields)]
    for frames, keep in plans:
        frames.clear(directory, keep)


def _groups(groups) -> Optional[tuple]:
    """Validate a ``groups=`` argument: None for every object, else labels."""
    if groups is None:
        return None
    if isinstance(groups, str):
        groups = (groups,)
    labels = tuple(groups)
    if not labels:
        raise ValueError(
            "groups=[] names no group, so the source would reach nothing; "
            "pass None to reach every object"
        )
    for label in labels:
        if not isinstance(label, str):
            raise TypeError(f"a group is named by its label (a str), got {label!r}")
    return labels


def _choice(name: str, value: str, allowed) -> str:
    value = str(value).lower()
    if value not in allowed:
        raise ValueError(f"field {name} {value!r} is not one of {', '.join(allowed)}")
    return value


def _per_sample(name: str, value, shape: tuple):
    """``value`` as an array of ``(T,) + shape``: one value repeated, or one
    per track sample. Returns the array and T (1 for a constant)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == shape:
        return arr[None, ...], 1
    if arr.ndim == len(shape) + 1 and arr.shape[1:] == shape and len(arr) >= 1:
        return arr, len(arr)
    raise ValueError(
        f"field {name} has shape {arr.shape}; give {shape or 'a number'} or one per "
        f"track sample, {('T',) + shape}"
    )


def make_field(kind: str, *, position=(0.0, 0.0, 0.0), axes=None, strength=1.0,
               shape: str = "point", falloff: str = "sphere", z_direction: str = "both",
               power=0.0, min_distance=None, max_distance=None, radial_power=0.0,
               radial_min=None, radial_max=None, gravitation: bool = False, size=1.0,
               noise=0.0, noise_scale=(1.0, 1.0, 1.0), wind_factor=0.0,
               linear_drag=1.0, quadratic_drag=1.0, harmonic_damping=1.0,
               rest_length=0.0, inflow=0.0, image=None, texture_mode: str = "rgb",
               extension: str = "extend", interpolation: bool = True,
               object_coordinates: bool = False, z_up: bool = False, nabla=0.025,
               seed: int = 0, global_coordinates: bool = False, start: float = 0.0,
               step: Optional[float] = None, groups=None, fluid=None, flow=0.0,
               use_density: bool = False) -> dict:
    """Validate one field object into the record the solver and
    ``frontend._field_eval_`` take. Arguments are those of
    :meth:`ForceField.add`."""
    kind = str(kind).lower()
    if kind in FIELD_TYPES_PENDING:
        raise ValueError(FIELD_TYPES_PENDING[kind])
    kind = _choice("type", kind, FIELD_TYPES)
    if kind == "fluid":
        # Blender stores a Fluid Flow's Flow and a Vortex's Inflow as one
        # value; the track's inflow slot carries it.
        if np.any(np.asarray(inflow) != 0.0):
            raise ValueError("a fluid field takes flow=, not inflow=")
        inflow = flow
        for name, value in (("noise", noise), ("wind_factor", wind_factor)):
            if np.any(np.asarray(value) != 0.0):
                raise ValueError(
                    f"a fluid field takes no {name}: it acts through the fluid's velocity alone"
                )
        if gravitation:
            raise ValueError(
                "a fluid field takes no gravitation: Blender applies none to a Fluid "
                "Flow field"
            )
    elif np.any(np.asarray(flow) != 0.0):
        raise ValueError(f"only a fluid field takes flow=, not a {kind} field")
    record: dict = {
        "type": kind,
        "shape": _choice("shape", shape, FIELD_SHAPES),
        "falloff": _choice("falloff", falloff, FIELD_FALLOFFS),
        "z_direction": _choice("z direction", z_direction, FIELD_Z_DIRECTIONS),
        "gravitation": bool(gravitation),
        "global_coordinates": bool(global_coordinates),
        "seed": int(seed),
        "use_min": min_distance is not None,
        "use_max": max_distance is not None,
        "use_radial_min": radial_min is not None,
        "use_radial_max": radial_max is not None,
        "groups": _groups(groups),
    }
    if not 0 <= record["seed"] < 2 ** 32:
        raise ValueError(f"field seed {seed} is not an unsigned 32-bit integer")
    values = {
        "position": (position, (3,)),
        "axes": (DEFAULT_AXES if axes is None else axes, (3, 3)),
        "noise_scale": (noise_scale, (3,)),
        "strength": (strength, ()),
        "power": (power, ()),
        "min_distance": (0.0 if min_distance is None else min_distance, ()),
        "max_distance": (0.0 if max_distance is None else max_distance, ()),
        "radial_power": (radial_power, ()),
        "radial_min": (0.0 if radial_min is None else radial_min, ()),
        "radial_max": (0.0 if radial_max is None else radial_max, ()),
        "size": (size, ()),
        "noise": (noise, ()),
        "wind_factor": (wind_factor, ()),
        "linear_drag": (linear_drag, ()),
        "quadratic_drag": (quadratic_drag, ()),
        "harmonic_damping": (harmonic_damping, ()),
        "rest_length": (rest_length, ()),
        "inflow": (inflow, ()),
        "nabla": (nabla, ()),
    }
    arrays = {}
    samples = 1
    for name, (value, shape_) in values.items():
        arr, n = _per_sample(name, value, shape_)
        if n > 1:
            if samples > 1 and n != samples:
                raise ValueError(
                    f"field {name} has {n} track samples and another setting has {samples}"
                )
            samples = n
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"field {name} is not finite")
        arrays[name] = arr
    for name, arr in arrays.items():
        if len(arr) == 1 and samples > 1:
            arrays[name] = np.repeat(arr, samples, axis=0)
    if samples > 1:
        if step is None or not (math.isfinite(float(step)) and float(step) > 0.0):
            raise ValueError(
                f"{samples} track samples need step=, the seconds between them, positive"
            )
    if not math.isfinite(float(start)):
        raise ValueError(f"field start {start} is not finite")
    for k, a in enumerate(arrays["axes"]):
        if not np.allclose(a.T @ a, np.eye(3), atol=1e-5) or np.linalg.det(a) < 0.0:
            raise ValueError(
                f"field axes at sample {k} are not a rotation (orthonormal columns, "
                "right-handed); object scale reaches a field only as noise_scale"
            )
    # The solver reads the noise scale as float32, divides by it (Noise
    # Amount's gusts, an object_coordinates texture's lookup) and refuses one
    # that is not a positive NORMAL float32, so it is refused here first.
    tiny, huge = np.finfo(np.float32).tiny, np.finfo(np.float32).max
    with np.errstate(over="ignore", under="ignore"):
        ns32 = arrays["noise_scale"].astype(np.float32)
    if not np.all((arrays["noise_scale"] > 0.0) & np.isfinite(ns32) & (ns32 >= tiny)):
        raise ValueError(
            "field noise_scale must be positive along every axis, from "
            f"{float(tiny):.3g} to {float(huge):.3g} (the solver reads it as a normal "
            "float32)"
        )
    if kind == "wind" and np.any(arrays["wind_factor"] != 0.0):
        raise ValueError(
            "a wind field takes no wind_factor: it acts through the drag, which "
            "weighs a surface by how it faces the flow already"
        )
    if kind == "texture":
        if image is None:
            raise ValueError("a texture field needs image=, an (H, W, 3) array of linear RGB")
        img = np.asarray(image, dtype=np.float64)
        if img.ndim != 3 or img.shape[2] != 3 or img.shape[0] < 1 or img.shape[1] < 1:
            raise ValueError(f"texture image has shape {img.shape}; give (H, W, 3)")
        if not np.all(np.isfinite(img)):
            raise ValueError("texture image is not finite")
        record["image"] = img
        record["texture_mode"] = _choice("texture mode", texture_mode, TEXTURE_MODES)
        record["extension"] = _choice("texture extension", extension, TEXTURE_EXTENSIONS)
        record["interpolation"] = bool(interpolation)
        record["object_coordinates"] = bool(object_coordinates)
        record["z_up"] = bool(z_up)
        if record["texture_mode"] != "rgb" and np.any(arrays["nabla"] <= 0.0):
            raise ValueError("a gradient or curl texture needs nabla above 0")
    elif image is not None:
        raise ValueError(f"only a texture field takes image=, not a {kind} field")
    elif z_up:
        raise ValueError(f"only a texture field takes z_up=, not a {kind} field")
    if kind == "fluid":
        if fluid is None:
            raise ValueError("a fluid field needs fluid=, a FluidVolume")
        if not isinstance(fluid, FluidVolume):
            raise TypeError(f"fluid= takes a FluidVolume, got {type(fluid).__name__}")
        if use_density and not fluid.has_density:
            raise ValueError(
                "use_density=True weighs the push by the fluid's density, and this "
                "FluidVolume was built without density=, so the field would act "
                "nowhere; give the volume density= or drop use_density"
            )
        record["fluid"] = fluid.as_dict()
        record["use_density"] = bool(use_density)
    elif fluid is not None or use_density:
        raise ValueError(f"only a fluid field takes fluid= and use_density=, not a {kind} field")
    if kind == "turbulence" and np.any(arrays["size"] <= 0.0):
        raise ValueError(
            "a turbulence field needs size above 0: its swirls are size across"
        )
    record.update(arrays)
    record["start"] = float(start)
    record["step"] = float(step) if step is not None else 1.0
    return record


# --- the public holder ----------------------------------------------------


class ForceField:
    """A scene's external force field, reached as ``scene.force_field``.

    Each source pushes every free vertex it reaches, independent of its
    mass. Most add an acceleration (m/s^2, like gravity), evaluated once per
    solver step at the vertex's position at the start of the step; a Wind is
    an air velocity that acts through the aerodynamic drag; and Drag,
    Harmonic, a Vortex's damping and a Fluid are solved implicitly, at the
    Newton iterate, a Fluid read at the step's end. A field object has no
    domain: it acts wherever its falloff reaches (a Fluid, only inside its
    volume). Fix-pinned vertices ignore every source. A source reaches every
    object, or with ``groups=[...]`` only the objects put in those groups
    with ``Object.group(label)``. Scale it per object with
    ``Object.param.set("force-field-weight", w)``; a weight of 0 opts the
    object out.

    Example:
        An exact swirl, compiled to run on the device, for one group only::

            def swirl(x, y, z, t):
                r = math.sqrt(x * x + z * z) + 1e-6
                return (-z / r, 0.0, x / r)

            scene.add("sheet").group("flags")
            scene.force_field.script(swirl, groups=["flags"])

        Built-in curl noise, a swirling field with no sources or sinks::

            def gusts(x, y, z, t):
                return curl_noise(x, y, z, octaves=3, seed=1, time=t, frequency=0.5)

            scene.force_field.script(gusts)

        A vortex about the vertical axis whose strength ramps up over two
        seconds, sampled ten times a second::

            scene.force_field.add(
                "vortex", position=(0, 0, 0),
                strength=np.linspace(0.0, 4.0, 21), step=0.1)
    """

    def __init__(self):
        self._fields: list = []
        self._scripts: list = []

    def add(self, kind: str, *, position=(0.0, 0.0, 0.0), axes=None, strength=1.0,
            shape: str = "point", falloff: str = "sphere", z_direction: str = "both",
            power=0.0, min_distance=None, max_distance=None, radial_power=0.0,
            radial_min=None, radial_max=None, gravitation: bool = False, size=1.0,
            noise=0.0, noise_scale=(1.0, 1.0, 1.0), wind_factor=0.0,
            linear_drag=1.0, quadratic_drag=1.0, harmonic_damping=1.0,
            rest_length=0.0, inflow=0.0, image=None, texture_mode: str = "rgb",
            extension: str = "extend", interpolation: bool = True,
            object_coordinates: bool = False, z_up: bool = False, nabla=0.025,
            seed: int = 0, global_coordinates: bool = False, start: float = 0.0,
            step: Optional[float] = None, groups=None, fluid=None, flow=0.0,
            use_density: bool = False) -> "ForceField":
        """Add a field object, evaluated exactly at every vertex it reaches.

        The laws are Blender's force fields, measured on Blender as a black
        box, in the field's local frame, with ``p`` the vertex relative to
        ``position`` and ``W`` the falloff weight:

        * ``"force"``: a push away from the origin (``shape="point"``) or from
          the XY plane (``shape="plane"``); a negative strength pulls.
        * ``"wind"``: an AIR velocity of ``strength`` along local Z, added to
          the scene wind inside the drag, so it acts only where
          ``air-density`` is positive and on a surface by how it faces it.
        * ``"vortex"``: a swirl about local Z growing with the distance from
          the origin; ``shape="plane"`` is a rotation about the axis with its
          centripetal pull.
        * ``"turbulence"``: smooth noise ``size`` across.
        * ``"magnetic"``: ``v x t`` with ``t`` circling local Z (or local Z
          itself on a plane), from the vertex's velocity.
        * ``"drag"``: ``-(linear_drag |v| + quadratic_drag) v``, as Blender's
          Drag measures (its Linear Drag multiplies ``|v| v``).
        * ``"harmonic"``: a spring to the origin of ``rest_length``
          (``shape="plane"``: to the XY plane), critically damped by
          ``harmonic_damping``.
        * ``"texture"``: an ``image`` (``(H, W, 3)`` linear RGB, row 0 at the
          bottom) spanning ``x, y`` in [-1, 1] of world coordinates (or of
          the field's own with ``object_coordinates``, divided by
          ``noise_scale`` along each local axis), read as
          ``strength * (0.5 - rgb)`` (``texture_mode="rgb"``), minus the
          intensity's gradient (``"gradient"``) or the color's curl
          (``"curl"``), forward differences ``nabla`` apart, a world-axes
          vector. World coordinates and that vector are solver axes (Y up),
          or Blender's (Z up) with ``z_up``. ``extension`` is ``"extend"``,
          ``"repeat"``, ``"clip"`` or ``"clip_cube"``; ``interpolation``
          reads bilinearly between pixel centers, else the nearest pixel.
        * ``"fluid"``: a fluid's velocity ``U`` from ``fluid`` (a
          :class:`FluidVolume`), as Blender's Fluid Flow:
          ``strength * W * rho * (U - flow * v)`` with ``v`` the vertex's
          velocity and ``rho`` the fluid's density if ``use_density``, else
          the domain's presence (1 inside the domain, 0 outside, blended
          between frames), so outside the domain nothing acts, the damping
          included. At ``flow=1`` a vertex relaxes toward the fluid at the
          rate ``strength`` (1/s); at ``flow=0`` the fluid pushes whatever
          the vertex does. Solved implicitly, and read at the end of each
          step. ``U`` is per second of the scene clock, so the solver
          divides it by the Playback Speed (``playback``), which sets how
          many seconds of motion a second of that clock holds (see
          :class:`FluidVolume`). A fluid field takes no ``noise``,
          ``wind_factor`` or ``gravitation``: it acts through the fluid's
          velocity alone. (Blender ignores Gravitation on a Fluid Flow, but
          applies Noise Amount and Wind Factor, which the add-on therefore
          refuses on one.)

        A Plane ``"vortex"`` also damps the velocity toward its swirl, and
        ``inflow`` adds to that damping (on a Point, it is the whole of it).
        Drag, Harmonic and these damping terms are solved implicitly, at the
        Newton iterate, so strong values stay stable.

        A field object has no domain: it acts wherever its falloff reaches,
        however far that is. The solver refuses an origin with a coordinate
        beyond 1e6 solver units (scene units times World Scaling).

        Args:
            kind: The field type, case-insensitive: ``"force"``, ``"wind"``,
                ``"vortex"``, ``"turbulence"``, ``"magnetic"``, ``"drag"``,
                ``"harmonic"``, ``"texture"`` or ``"fluid"``.
            position: The origin, solver axes (Y up), scene units.
            axes: A rotation, its columns the field's local X, Y, Z in solver
                axes; the default is a new Blender field's, local Z up.
            strength: The Strength, per unit mass: an acceleration in m/s^2 for a
                Force, Turbulence or Texture; an air speed in m/s for a Wind; a rate
                in 1/s for a Plane Vortex, a Magnetic field and a Fluid; and 1/s^2 for
                a Point Vortex and a Harmonic, whose negative strength pushes away. A
                Drag does not use it.
            shape, falloff, z_direction: As Blender's settings of those names. The
                shape defaults to ``"point"`` for every kind, where Blender gives a
                new Wind and a new Vortex the Plane.
            power, min_distance, max_distance: The falloff. A distance of
                None is off.
            radial_power, radial_min, radial_max: The Tube's radial falloff,
                or the Cone's, whose limits are angles in DEGREES.
            gravitation: Multiply by ``1 / d^2``, exactly and without a cap, with
                ``d`` the distance from the origin (on a plane shape, from its
                plane; on a Plane Vortex, from its axis). A vertex the falloff
                reaches at ``d = 0`` fails the run, naming the field by its index
                in ``add()`` order. A fluid field takes none.
            size: A Turbulence's swirl size.
            noise: Noise Amount, curl-noise gusts as a fraction of the
                strength; on a Wind they drift with the flow.
            noise_scale: The field object's scale along its local X, Y and Z,
                positive, as Blender's object scale reaches a field: it sizes
                Noise Amount's gusts (a Turbulence's are ``size`` across
                instead), and an ``object_coordinates`` Texture reads its
                image at the local point divided by it, as Blender reads it
                under the object's whole inverse world matrix. ``axes`` holds
                the rotation alone.
            linear_drag, quadratic_drag: A Drag's coefficients.
            harmonic_damping, rest_length: A Harmonic's damping ratio and
                spring rest length.
            inflow: A Vortex's added velocity damping.
            image, texture_mode, extension, interpolation: A Texture's image and how
                it is read (see above).
            object_coordinates, z_up, nabla: A Texture read in the field object's own
                frame, its scale included (see ``noise_scale``), rather than the
                world's; its image and vector in Blender's axes; and its difference
                step, in the coordinates the image is read at (animatable).
            wind_factor: How much the push is reduced where it runs along a
                surface: ``(1 - wf) a + wf (a . n) n`` with ``n`` the
                surface normal, as Blender's Wind Factor (not on a Wind, whose
                drag weighs the surface already).
            seed: The noise's seed.
            global_coordinates: A Turbulence pinned to the world rather than
                to the field.
            start, step: Every continuous setting may be one value or one per
                track sample; samples are ``step`` seconds apart from
                ``start``, linear between, and the end sample holds outside.
            groups: Group labels (``Object.group``) whose objects the field
                reaches; None, the default, reaches every object.
            fluid, flow, use_density: A Fluid field's volume, how much of the vertex's
                own velocity it subtracts, and whether the fluid's density weighs it
                (see above), which needs a volume built with density.

        Returns:
            ForceField: This field, for chaining.

        Raises:
            ValueError: A setting the solver would refuse or ignore, named.
            TypeError: ``fluid`` is not a :class:`FluidVolume`.
        """
        self._fields.append(make_field(
            kind, position=position, axes=axes, strength=strength, shape=shape,
            falloff=falloff, z_direction=z_direction, power=power,
            min_distance=min_distance, max_distance=max_distance,
            radial_power=radial_power, radial_min=radial_min, radial_max=radial_max,
            gravitation=gravitation, size=size, noise=noise, noise_scale=noise_scale,
            wind_factor=wind_factor, linear_drag=linear_drag,
            quadratic_drag=quadratic_drag, harmonic_damping=harmonic_damping,
            rest_length=rest_length, inflow=inflow, image=image,
            texture_mode=texture_mode, extension=extension,
            interpolation=interpolation, object_coordinates=object_coordinates,
            z_up=z_up, nabla=nabla, seed=seed, global_coordinates=global_coordinates,
            start=start, step=step,
            groups=groups, fluid=fluid, flow=flow, use_density=use_density))
        return self

    @staticmethod
    def builtins() -> None:
        """Print every function, constant and construct a script may use."""
        print(_script_api.reference_text())

    def script(self, source, *, z_up: bool = False, groups=None) -> "ForceField":
        """Add an exact script, a function ``eval(x, y, z, t)`` returning
        ``(ax, ay, az)``, given as source text or as the function itself.

        Besides ``math``, a script may call ``noise(x, y, z, octaves=1,
        seed=0, time=0.0, frequency=1.0, decay=0.0)``, smooth noise in about
        [-1, 1], and ``curl_noise(...)`` with the same arguments, a swirling
        vector field with no sources or sinks, returned as three values
        (``return curl_noise(...)`` or ``cx, cy, cz = curl_noise(...)``).
        ``octaves`` is a number from 1 to 8 written in the script; the rest
        may be any expression. Pass ``time=t`` to let the pattern evolve in
        place, ``frequency`` times per second, and fade as
        ``exp(-decay * t)``. The frontend exports both as ``frontend.noise``
        and ``frontend.curl_noise`` for evaluating the same noise with numpy.
        :meth:`builtins` prints the whole list.

        Compiled now, so a construct outside the supported subset is refused
        here with its line. Scripts add up; ``groups`` is as for :meth:`add`.
        """
        compiled = compile_script(source, z_up=z_up)
        print(f"[Info] Force field script compiled: {compiled.summary()}")
        self._scripts.append((compiled, _groups(groups)))
        return self

    def clear(self) -> "ForceField":
        """Remove every field object and every script."""
        self._fields = []
        self._scripts = []
        return self

    @property
    def empty(self) -> bool:
        return not self._fields and not self._scripts

    @property
    def fields(self) -> list:
        """The field object records, as ``frontend._field_eval_`` takes them."""
        return list(self._fields)

    @property
    def scripts(self) -> list:
        """``[(CompiledScript, groups or None), ...]``."""
        return list(self._scripts)


# --- target resolution ------------------------------------------------------

ALL_TARGETS = 32


def resolve_targets(field: ForceField, group_vertices: dict, n_vert: int):
    """Give each distinct target set a bit and build the per-vertex mask.

    ``group_vertices`` maps a group label to the indices of the dynamic
    vertices of the objects in that group. Returns ``(fields, scripts, mask)``:
    ``[(field record, bit)]``, ``[(CompiledScript, bit)]`` and a ``uint32`` mask
    per vertex, or None when every source reaches every vertex. ``bit`` is
    ``ALL_TARGETS`` for such a source.

    Raises:
        ValueError: A source names a group no object is in, or the sources
            name more than 32 distinct sets of groups.
    """
    sets: dict = {}

    def bit_of(groups):
        if groups is None:
            return ALL_TARGETS
        key = frozenset(groups)
        unknown = sorted(g for g in key if g not in group_vertices)
        if unknown:
            known = sorted(group_vertices)
            raise ValueError(
                f"a force field source targets group(s) {unknown}, which no object "
                f"is in; the groups are {known}. Put objects in a group with "
                "Object.group(label)"
            )
        if key not in sets:
            if len(sets) >= ALL_TARGETS:
                raise ValueError(
                    f"the force field sources name more than {ALL_TARGETS} different "
                    "sets of groups; merge sources that share targets"
                )
            sets[key] = len(sets)
        return sets[key]

    fields = [(f, bit_of(f["groups"])) for f in field.fields]
    scripts = [(c, bit_of(groups)) for c, groups in field.scripts]
    if not sets:
        return fields, scripts, None
    mask = np.zeros(n_vert, dtype=np.uint32)
    for key, bit in sets.items():
        for label in key:
            mask[np.asarray(group_vertices[label], dtype=np.int64)] |= np.uint32(1 << bit)
    return fields, scripts, mask


# --- the session writer ---------------------------------------------------

FIELD_DIR = os.path.join("bin", "force_field")


def track_array(field: dict):
    """The per-sample floats the solver's track file holds: the axes'
    columns (local X, Y, Z), the nine ``FIELD_SCALARS``, the noise scale,
    then the ``FIELD_TAIL`` values, as ``(T, 28)`` float32
    (``TRACK_FILE_FLOATS`` in the solver)."""
    axes = np.asarray(field["axes"], dtype=np.float64)
    columns = np.transpose(axes, (0, 2, 1)).reshape(len(axes), 9)
    scalars = np.stack([np.asarray(field[name], dtype=np.float64) for name in FIELD_SCALARS],
                       axis=1)
    tail = np.stack([np.asarray(field[name], dtype=np.float64) for name in FIELD_TAIL],
                    axis=1)
    return np.concatenate([columns, scalars, np.asarray(field["noise_scale"]), tail],
                          axis=1).astype("<f4")


def write_session(session_path: str, fields, scripts, weight=None, mask=None) -> None:
    """Write the field's files for the solver under ``<session>/bin/force_field``.

    ``fields`` and ``scripts`` are :func:`resolve_targets`'s resolved lists of
    ``(source, bit)`` pairs, and ``mask`` its per-vertex target mask. A fluid's
    frames are written one at a time, and a :class:`FluidVolume`'s frames
    then refer to these files, which is what a pickle of it carries.
    """
    directory = os.path.join(session_path, FIELD_DIR)
    if not fields and not scripts:
        return
    os.makedirs(directory, exist_ok=True)
    lines = []
    for i, (f, bit) in enumerate(fields):
        origin = f"field-{i}-origin.bin"
        track = f"field-{i}-track.bin"
        # The origin in DOUBLE: the solver rounds it once, after World
        # Scaling.
        np.ascontiguousarray(f["position"], dtype="<f8").tofile(os.path.join(directory, origin))
        track_array(f).tofile(os.path.join(directory, track))
        lines += [
            "[[field]]",
            f'type = "{f["type"]}"',
            f'shape = "{f["shape"]}"',
            f'falloff = "{f["falloff"]}"',
            f'z_direction = "{f["z_direction"]}"',
            f"use_min = {str(f['use_min']).lower()}",
            f"use_max = {str(f['use_max']).lower()}",
            f"use_radial_min = {str(f['use_radial_min']).lower()}",
            f"use_radial_max = {str(f['use_radial_max']).lower()}",
            f"gravitation = {str(f['gravitation']).lower()}",
            f"global_coordinates = {str(f['global_coordinates']).lower()}",
            f"seed = {int(f['seed'])}",
            f"start = {float(f['start'])!r}",
            f"step = {float(f['step'])!r}",
            f'origin = "{origin}"',
            f'track = "{track}"',
            f"targets = {int(bit)}",
            "",
        ]
        if f["type"] == "fluid":
            vol = f["fluid"]
            sub = f"field-{i}-fluid"
            os.makedirs(os.path.join(directory, sub), exist_ok=True)
            frames = len(vol["tables"])
            # Per frame: the corner, the axes as columns, and the cell along
            # each axis (FLUID_TRANSFORM_FLOATS in the solver).
            cell = np.broadcast_to(np.asarray(vol["cell"], dtype=np.float64).reshape(
                frames, -1), (frames, 3))
            transform = np.concatenate(
                [np.asarray(vol["corner"], dtype=np.float64).reshape(frames, 3),
                 np.transpose(np.asarray(vol["axes"], dtype=np.float64),
                              (0, 2, 1)).reshape(frames, 9),
                 cell], axis=1)
            np.ascontiguousarray(transform, dtype="<f8").tofile(
                os.path.join(directory, sub, "transform.bin"))
            source = vol["tables"]._frames if isinstance(vol["tables"], _FrameColumn) else None
            tables, bricks = [], []
            for k in range(frames):
                tables.append(f"{sub}/table-{k:05d}.bin")
                bricks.append(f"{sub}/bricks-{k:05d}.bin")
                table = np.ascontiguousarray(vol["tables"][k], dtype="<u4")
                brick = np.ascontiguousarray(vol["bricks"][k], dtype="<f4")
                table.tofile(os.path.join(directory, tables[-1]))
                brick.tofile(os.path.join(directory, bricks[-1]))
                if source is not None:
                    source.bind(k, os.path.join(directory, tables[-1]),
                                os.path.join(directory, bricks[-1]), table, brick)
                table = brick = None
            lines[-1:-1] = [
                "[field.fluid]",
                "resolution = [" + ", ".join(
                    "[" + ", ".join(str(int(v)) for v in r) + "]" for r in vol["resolution"]) + "]",
                f"use_density = {str(f['use_density']).lower()}",
                f"start = {float(vol['start'])!r}",
                f"step = {float(vol['step'])!r}",
                f'transform = "{sub}/transform.bin"',
                "tables = [" + ", ".join(f'"{t}"' for t in tables) + "]",
                "bricks = [" + ", ".join(f'"{b}"' for b in bricks) + "]",
            ]
        if f["type"] == "texture":
            image = f"field-{i}-image.bin"
            img = np.asarray(f["image"])
            np.ascontiguousarray(img, dtype="<f4").tofile(os.path.join(directory, image))
            lines[-1:-1] = [
                "[field.texture]",
                f'image = "{image}"',
                f"width = {img.shape[1]}",
                f"height = {img.shape[0]}",
                f'mode = "{f["texture_mode"]}"',
                f'extension = "{f["extension"]}"',
                f"interpolation = {str(f['interpolation']).lower()}",
                f"object_coordinates = {str(f['object_coordinates']).lower()}",
                f"z_up = {str(bool(f['z_up'])).lower()}",
            ]
    for i, (script, bit) in enumerate(scripts):
        np.asarray(script.code, dtype="<u4").tofile(os.path.join(directory, f"script-{i}-code.bin"))
        np.asarray(script.constants, dtype="<f4").tofile(
            os.path.join(directory, f"script-{i}-constants.bin")
        )
        with open(os.path.join(directory, f"script-{i}.py"), "w", encoding="utf-8") as f:
            f.write(script.source)
        lines += [
            "[[script]]",
            f"version = {SCRIPT_VERSION}",
            f'code = "script-{i}-code.bin"',
            f'constants = "script-{i}-constants.bin"',
            f'source = "script-{i}.py"',
            f"targets = {int(bit)}",
            "",
        ]
    if weight is not None:
        np.asarray(weight, dtype="<f4").tofile(os.path.join(directory, "weight.bin"))
        lines += ["[weight]", 'data = "weight.bin"', ""]
    if mask is not None:
        np.asarray(mask, dtype="<u4").tofile(os.path.join(directory, "targets.bin"))
        lines += ["[targets]", 'data = "targets.bin"', ""]
    with open(os.path.join(directory, "force_field.toml"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --- the server's Compile and Check worker --------------------------------


def check_main(argv: Sequence[str]) -> int:
    """``python -m frontend._force_field_ check`` reads a JSON request on
    stdin, ``{"source": ..., "z_up": bool}``, and
    writes one JSON answer on stdout: ``{"ok": true, "summary": ...}`` or
    ``{"ok": false, "error": ..., "line": n}``. Exit status 0 either way;
    a nonzero status means the worker itself failed."""
    request = json.loads(sys.stdin.read() or "{}")
    try:
        compiled = compile_script(
            request.get("source", ""),
            z_up=bool(request.get("z_up", False)),
        )
        answer = {"ok": True, "summary": compiled.summary()}
    except ForceFieldScriptError as e:
        answer = {"ok": False, "error": e.detail, "line": e.lineno}
    sys.stdout.write(json.dumps(answer))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "check":
        sys.exit(check_main(sys.argv[2:]))
    sys.exit("usage: python -m frontend._force_field_ check < request.json")
