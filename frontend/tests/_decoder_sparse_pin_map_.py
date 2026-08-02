# File: _decoder_sparse_pin_map_.py
# Code: GitHub Copilot
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from frontend._decoder_ import (
        _SparseLinearMap,
        _apply_sparse_frame_map,
        _build_harmonic_interior_operator,
    )
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"frontend / _ppf_cts_py not importable in this environment: {exc}",
        allow_module_level=True,
    )


def test_sparse_linear_map_matches_dense_reference():
    matrix = np.array(
        [
            [4.0, -1.0, 0.0, 0.0],
            [-1.0, 4.0, -1.0, 0.0],
            [0.0, -1.0, 4.0, -1.0],
            [0.0, 0.0, -1.0, 3.0],
        ]
    )
    rhs_map = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 1.0],
            [0.0, 0.5, 0.0],
        ]
    )
    values = np.arange(27, dtype=np.float64).reshape(3, 3, 3) / 7.0

    linear_map = _SparseLinearMap(
        sp.csc_matrix(matrix),
        sp.csr_matrix(rhs_map),
    )
    actual = _apply_sparse_frame_map(linear_map, values)
    expected = np.stack(
        [np.linalg.solve(matrix, rhs_map @ frame) for frame in values]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_sparse_linear_map_roundtrips_without_pickling_superlu():
    matrix = sp.csc_matrix([[2.0, -1.0], [-1.0, 2.0]])
    rhs_map = sp.eye(2, format="csr")
    values = np.array([[3.0], [5.0]])

    restored = pickle.loads(
        pickle.dumps(_SparseLinearMap(matrix, rhs_map))
    )

    np.testing.assert_allclose(
        restored.apply(values),
        np.linalg.solve(matrix.toarray(), values),
        rtol=1e-12,
        atol=1e-12,
    )


def test_partial_pin_motion_matches_dense_maps_and_reuses_cache():
    from frontend._decoder_ import ParamDecoder

    surface_matrix = np.array([[3.0, -1.0], [-1.0, 2.0]])
    surface_rhs = np.array(
        [[1.0, 0.25, 0.0], [0.0, 0.5, 1.0]]
    )
    interior_matrix = np.array([[2.0]])
    interior_rhs = np.array([[0.75, 1.25]])
    surface_map = _SparseLinearMap(
        sp.csc_matrix(surface_matrix),
        sp.csr_matrix(surface_rhs),
    )
    interior_map = _SparseLinearMap(
        sp.csc_matrix(interior_matrix),
        sp.csr_matrix(interior_rhs),
    )
    times = [0.0, 0.5, 1.0]
    input_positions = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.1, 0.0], [0.0, 1.2, 0.1]],
            [[0.2, 0.1, 0.0], [1.2, 0.1, 0.1], [0.1, 1.3, 0.2]],
        ]
    )
    obj_cfg = {
        i: {
            "pin_anim": {
                i: {"time": times, "position": input_positions[:, i, :]}
            }
        }
        for i in range(3)
    }
    rest_full = np.array(
        [[0.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.3, 0.3, 0.4]]
    )
    motion_cache = {}
    solid_pin = {
        "surface_map": surface_map,
        "interior_map": interior_map,
        "motion_cache": motion_cache,
        "keep": np.ones(3, dtype=bool),
        "n_input": 3,
        "rest_full": rest_full,
    }

    ops = ParamDecoder._build_solid_poisson_move_ops(solid_pin, obj_cfg)

    dense_surface = np.linalg.solve(surface_matrix, surface_rhs)
    dense_interior = np.linalg.solve(interior_matrix, interior_rhs)
    residual = np.zeros_like(input_positions)
    rigid_positions = np.empty((3, 3, 3))
    p0 = input_positions[0]
    for frame in range(3):
        rotation, translation = ParamDecoder._rigid_fit(
            p0, input_positions[frame]
        )
        residual[frame] = (
            input_positions[frame] - (p0 @ rotation.T + translation)
        )
        rigid_positions[frame] = rest_full @ rotation.T + translation
    surface_residual = np.einsum(
        "si,fic->fsc", dense_surface, residual
    )
    interior_residual = np.einsum(
        "is,fsc->fic", dense_interior, surface_residual
    )
    expected_positions = rigid_positions + np.concatenate(
        [surface_residual, interior_residual], axis=1
    )
    for frame, op in enumerate(ops):
        np.testing.assert_allclose(
            op.delta,
            expected_positions[frame + 1] - expected_positions[frame],
            rtol=1e-12,
            atol=1e-12,
        )

    sliced = dict(solid_pin)
    sliced["keep"] = np.array([True, False, True])
    sliced_ops = ParamDecoder._build_solid_poisson_move_ops(sliced, obj_cfg)
    assert motion_cache.get("value") is not None
    for full_op, sliced_op in zip(ops, sliced_ops):
        np.testing.assert_allclose(
            sliced_op.delta,
            full_op.delta[[0, 2]],
            rtol=1e-12,
            atol=1e-12,
        )


def test_hard_pin_frame_map_preserves_normal_offset_under_rotation():
    from frontend._decoder_ import ParamDecoder

    times = [0.0, 1.0]
    rest_triangle = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = np.array([2.0, 3.0, 4.0])
    moved_triangle = rest_triangle @ rotation.T + translation
    obj_cfg = {
        vertex: {
            "pin_anim": {
                vertex: {
                    "time": times,
                    "position": np.stack(
                        [rest_triangle[vertex], moved_triangle[vertex]]
                    ),
                }
            }
        }
        for vertex in range(3)
    }
    frame_map = {
        "triangles": np.array([[0, 1, 2]], dtype=np.int64),
        "coefs": np.array([[0.25, 0.25, 0.2]], dtype=np.float64),
    }

    ops = ParamDecoder._build_solid_frame_move_ops(frame_map, obj_cfg)

    rest_point = np.array([0.25, 0.25, 0.2])
    expected_delta = rest_point @ rotation.T + translation - rest_point
    np.testing.assert_allclose(
        ops[0].delta[0], expected_delta, rtol=1e-12, atol=1e-12
    )


def test_hard_pin_frame_map_drops_normal_offset_when_triangle_degenerates():
    from frontend._decoder_ import ParamDecoder

    times = [0.0, 1.0]
    rest_triangle = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    degenerate_triangle = np.array(
        [[2.0, 3.0, 4.0], [3.0, 3.0, 4.0], [4.0, 3.0, 4.0]]
    )
    obj_cfg = {
        vertex: {
            "pin_anim": {
                vertex: {
                    "time": times,
                    "position": np.stack(
                        [rest_triangle[vertex], degenerate_triangle[vertex]]
                    ),
                }
            }
        }
        for vertex in range(3)
    }
    frame_map = {
        "triangles": np.array([[0, 1, 2]], dtype=np.int64),
        "coefs": np.array([[0.25, 0.25, 0.2]], dtype=np.float64),
    }

    ops = ParamDecoder._build_solid_frame_move_ops(frame_map, obj_cfg)

    expected_final = np.array([2.75, 3.0, 4.0])
    expected_rest = np.array([0.25, 0.25, 0.2])
    np.testing.assert_allclose(
        ops[0].delta[0],
        expected_final - expected_rest,
        rtol=1e-12,
        atol=1e-12,
    )


def test_harmonic_interior_map_reproduces_linear_boundary_values():
    # One tetrahedron with vertices 0..2 on the selected boundary and vertex
    # 3 in the interior. The binary tet graph is K4, so the harmonic value at
    # vertex 3 is the mean of the three boundary values.
    harmonic = _build_harmonic_interior_operator(
        4,
        np.array([[0, 1, 2, 3]], dtype=np.int64),
        [0, 1, 2],
        [3],
    )
    assert harmonic is not None

    boundary = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    actual = harmonic.apply(boundary)

    np.testing.assert_allclose(
        actual,
        boundary.mean(axis=0, keepdims=True),
        rtol=1e-12,
        atol=1e-12,
    )
