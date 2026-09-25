# File: _force_field_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""External force fields: the script compiler and the session writer.

A force field adds a per-vertex acceleration ``a(x, t)`` (m/s^2, like gravity)
to every free vertex, evaluated once per solver step at the vertex's position
at the start of that step. It comes from two sources, which a scene may mix:

* **Sampled grids**, a world-space box sampled at ``W x H x D`` cell corners
  and ``T`` instants, trilinear in space and linear in time, zero outside the
  box. A grid's kind is ``"acceleration"`` or ``"air-velocity"``; the second
  adds to the scene wind inside the aerodynamic drag.
* **One exact script**, a Python function ``eval(x, y, z, t)`` returning
  ``(ax, ay, az)``, compiled here into a small bytecode that the solver runs
  per vertex on the GPU. It has no resolution and no domain.

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
import json
import math
import os
import random
import sys
import textwrap
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

try:  # a package import, and a plain one for the server's worker path
    from . import _noise_ as _noise
    from . import _script_api_ as _script_api
except ImportError:  # pragma: no cover
    import _noise_ as _noise  # type: ignore
    import _script_api_ as _script_api  # type: ignore

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


# --- grids ----------------------------------------------------------------


@dataclass
class FieldGrid:
    """One sampled grid, as the frontend holds it until export."""

    values: "object"  # numpy array (T, D, H, W, 3), float32
    box_min: tuple
    box_max: tuple
    times: tuple
    kind: str = "acceleration"
    groups: Optional[tuple] = None

    @property
    def dims(self):
        t, d, h, w, _ = self.values.shape
        return (w, h, d, t)

    @property
    def nbytes(self) -> int:
        return int(self.values.size) * 4


def estimate_bytes(width: int, height: int, depth: int, samples: int) -> int:
    """Bytes a W x H x D x T grid of 3-vectors holds on the solver."""
    return int(width) * int(height) * int(depth) * int(samples) * 3 * 4


def estimate_line(width: int, height: int, depth: int, samples: int) -> str:
    """The ``[Info]`` line every place a grid is authored prints."""
    mb = estimate_bytes(width, height, depth, samples) / 1.0e6
    return f"[Info] Force field {width}x{height}x{depth}x{samples}: {mb:.1f} MB estimated"


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


def make_grid(values, box_min, box_max, times=None, kind="acceleration",
              quiet: bool = False, groups=None) -> FieldGrid:
    """Validate and wrap a grid, printing the size estimate unless ``quiet``."""
    import numpy as np

    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[None, ...]
    if arr.ndim != 5 or arr.shape[-1] != 3:
        raise ValueError(
            f"force field values have shape {arr.shape}; give (T, D, H, W, 3) "
            "or (D, H, W, 3)"
        )
    t, d, h, w, _ = arr.shape
    if min(w, h, d) < 2:
        raise ValueError(
            f"force field grid {w}x{h}x{d}: every spatial extent needs at least 2 samples"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("force field values contain NaN or infinity")
    lo = tuple(float(v) for v in box_min)
    hi = tuple(float(v) for v in box_max)
    if len(lo) != 3 or len(hi) != 3 or not all(a < b for a, b in zip(lo, hi)):
        raise ValueError(f"force field box {lo} .. {hi} is not a box")
    if times is None:
        if t != 1:
            raise ValueError(f"{t} time samples need {t} instants in 'times'")
        ts: tuple = (0.0,)
    else:
        ts = tuple(float(v) for v in times)
        if len(ts) != t:
            raise ValueError(f"{len(ts)} instants for {t} time samples")
        if any(not math.isfinite(v) for v in ts) or any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError(f"force field instants {ts} must be finite and strictly increasing")
    if kind not in ("acceleration", "air-velocity"):
        raise ValueError(f"force field kind {kind!r} is neither 'acceleration' nor 'air-velocity'")
    if not quiet:
        print(estimate_line(w, h, d, t))
    return FieldGrid(values=np.ascontiguousarray(arr), box_min=lo, box_max=hi, times=ts,
                     kind=kind, groups=_groups(groups))


def grid_shape(box_min, box_max, spacing: float) -> tuple:
    """``(W, H, D)``: the fewest points along each axis of the box that are at
    most ``spacing`` apart (at least two)."""
    spacing = float(spacing)
    if not (math.isfinite(spacing) and spacing > 0.0):
        raise ValueError(f"force field spacing must be a positive length, got {spacing}")
    n = [max(2, int(math.ceil((float(b) - float(a)) / spacing - 1e-9)) + 1)
         for a, b in zip(box_min, box_max)]
    return n[0], n[1], n[2]


def sample_grid(fn: Callable, box_min, box_max, spacing, times, kind="acceleration",
                groups=None) -> FieldGrid:
    """Sample ``fn(x, y, z, t)`` (numpy arrays in, three arrays out) on a grid
    of points at most ``spacing`` apart."""
    import numpy as np

    w, h, d = grid_shape(box_min, box_max, spacing)
    ts = [float(v) for v in times] if times is not None else [0.0]
    print(estimate_line(w, h, d, len(ts)))
    xs = np.linspace(box_min[0], box_max[0], w)
    ys = np.linspace(box_min[1], box_max[1], h)
    zs = np.linspace(box_min[2], box_max[2], d)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    out = np.empty((len(ts), d, h, w, 3), dtype=np.float32)
    for k, t in enumerate(ts):
        fx, fy, fz = fn(X, Y, Z, t)
        out[k, ..., 0] = np.broadcast_to(fx, X.shape)
        out[k, ..., 1] = np.broadcast_to(fy, X.shape)
        out[k, ..., 2] = np.broadcast_to(fz, X.shape)
    return make_grid(out, box_min, box_max, ts if times is not None else None, kind,
                     quiet=True, groups=groups)


# --- the public holder ----------------------------------------------------


class ForceField:
    """A scene's external force field, reached as ``scene.force_field``.

    Every source adds an ACCELERATION (m/s^2, independent of mass, like
    gravity) to every free vertex it reaches, evaluated once per solver step
    at the vertex's position at the start of the step. Fix-pinned vertices
    ignore it. A source reaches every object, or with ``groups=[...]`` only
    the objects put in those groups with ``Object.group(label)``. Scale it
    per object with ``Object.param.set("force-field-weight", w)``; a weight
    of 0 opts the object out.

    Example:
        An exact swirl, compiled to run on the GPU, for one group only::

            def swirl(x, y, z, t):
                r = math.sqrt(x * x + z * z) + 1e-6
                return (-z / r, 0.0, x / r)

            scene.add("sheet").group("flags")
            scene.force_field.script(swirl, groups=["flags"])

        Built-in curl noise, a swirling field with no sources or sinks::

            def gusts(x, y, z, t):
                return curl_noise(x, y, z, octaves=3, seed=1, time=t, frequency=0.5)

            scene.force_field.script(gusts)

        A sampled grid, points every 0.125 m (17 along each axis) at 4
        instants::

            scene.force_field.sample(
                lambda x, y, z, t: (0 * x, 2.0 * np.sin(t) + 0 * y, 0 * z),
                box_min=(-1, 0, -1), box_max=(1, 2, 1),
                spacing=0.125, times=[0, 1, 2, 3])
    """

    def __init__(self):
        self._grids: list = []
        self._scripts: list = []

    def grid(self, values, box_min, box_max, times=None, kind: str = "acceleration",
             groups=None) -> "ForceField":
        """Add a sampled grid.

        Args:
            values: ``(T, D, H, W, 3)`` samples, or ``(D, H, W, 3)`` for one
                instant. ``values[k, iz, iy, ix]`` is the vector at
                ``box_min + (box_max - box_min) * (ix, iy, iz) / (W-1, H-1, D-1)``
                and time ``times[k]``.
            box_min, box_max: The world-space box, solver axes (Y up).
            times: ``T`` strictly increasing instants in seconds; omit for one.
            kind: ``"acceleration"`` (m/s^2, added like gravity) or
                ``"air-velocity"`` (m/s, added to the scene wind; acts only
                where ``air-density`` is positive).
            groups: Group labels (``Object.group``) whose objects the grid
                reaches; None, the default, reaches every object.

        The grid is zero outside its box, and time outside ``times`` holds
        the end sample. Prints the ``[Info] ... MB estimated`` line.
        """
        self._grids.append(make_grid(values, box_min, box_max, times, kind, groups=groups))
        return self

    def sample(self, fn: Callable, box_min, box_max, spacing, times=None,
               kind: str = "acceleration", groups=None) -> "ForceField":
        """Sample ``fn(X, Y, Z, t)`` (numpy arrays in, three arrays out) into a
        grid over the box with points at most ``spacing`` apart along every
        axis (the counts follow from the box), at each of ``times``.
        ``groups`` is as for :meth:`grid`."""
        self._grids.append(sample_grid(fn, box_min, box_max, spacing, times, kind,
                                       groups=groups))
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
        and ``frontend.curl_noise`` for sampling grids with numpy.
        :meth:`builtins` prints the whole list.

        Compiled now, so a construct outside the supported subset is refused
        here with its line. Scripts add up; ``groups`` is as for :meth:`grid`.
        """
        compiled = compile_script(source, z_up=z_up)
        print(f"[Info] Force field script compiled: {compiled.summary()}")
        self._scripts.append((compiled, _groups(groups)))
        return self

    def clear(self) -> "ForceField":
        """Remove every grid and every script."""
        self._grids = []
        self._scripts = []
        return self

    @property
    def empty(self) -> bool:
        return not self._grids and not self._scripts

    @property
    def grids(self) -> list:
        return list(self._grids)

    @property
    def scripts(self) -> list:
        """``[(CompiledScript, groups or None), ...]``."""
        return list(self._scripts)

    def estimated_bytes(self) -> int:
        return sum(g.nbytes for g in self._grids)


# --- target resolution ------------------------------------------------------

ALL_TARGETS = 32


def resolve_targets(field: ForceField, group_vertices: dict, n_vert: int):
    """Give each distinct target set a bit and build the per-vertex mask.

    ``group_vertices`` maps a group label to the indices of the dynamic
    vertices of the objects in that group. Returns ``(grids, scripts, mask)``:
    ``[(FieldGrid, bit)]``, ``[(CompiledScript, bit)]`` and a ``uint32`` mask
    per vertex, or None when every source reaches every vertex. ``bit`` is
    ``ALL_TARGETS`` for such a source.

    Raises:
        ValueError: A source names a group no object is in, or the sources
            name more than 32 distinct sets of groups.
    """
    import numpy as np

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

    grids = [(g, bit_of(g.groups)) for g in field.grids]
    scripts = [(c, bit_of(groups)) for c, groups in field.scripts]
    if not sets:
        return grids, scripts, None
    mask = np.zeros(n_vert, dtype=np.uint32)
    for key, bit in sets.items():
        for label in key:
            mask[np.asarray(group_vertices[label], dtype=np.int64)] |= np.uint32(1 << bit)
    return grids, scripts, mask


# --- the session writer ---------------------------------------------------

FIELD_DIR = os.path.join("bin", "force_field")


def write_session(session_path: str, grids, scripts, weight=None, mask=None) -> None:
    """Write the field's files for the solver under ``<session>/bin/force_field``.

    ``grids`` and ``scripts`` are :func:`resolve_targets`'s resolved lists of
    ``(source, bit)`` pairs, and ``mask`` its per-vertex target mask.
    """
    import numpy as np

    directory = os.path.join(session_path, FIELD_DIR)
    if not grids and not scripts:
        return
    os.makedirs(directory, exist_ok=True)
    lines = []
    for i, (g, bit) in enumerate(grids):
        name = f"grid-{i}.bin"
        np.ascontiguousarray(g.values, dtype="<f4").tofile(os.path.join(directory, name))
        w, h, d, t = g.dims
        lines += [
            "[[grid]]",
            f'kind = "{g.kind}"',
            f"dims = [{w}, {h}, {d}, {t}]",
            f"min = [{', '.join(repr(float(v)) for v in g.box_min)}]",
            f"max = [{', '.join(repr(float(v)) for v in g.box_max)}]",
            f"times = [{', '.join(repr(float(v)) for v in g.times)}]",
            f'data = "{name}"',
            f"targets = {int(bit)}",
            "",
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
