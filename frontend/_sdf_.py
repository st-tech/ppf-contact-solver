# File: _sdf_.py
# SDF (Signed Distance Field) implementation with numba JIT compilation

import numpy as np

from numba import njit, prange

# Marching cubes edge table - maps cube configuration to edges that intersect surface
EDGE_TABLE = np.array([
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
], dtype=np.int32)

# Triangle table - flattened with offsets for numba compatibility
# Each row is padded to 16 elements, -1 indicates end
TRI_TABLE_DATA = np.array([
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 8, 3, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 1, 9, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 8, 3, 9, 8, 1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 2, 10, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 8, 3, 1, 2, 10, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9, 2, 10, 0, 2, 9, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1,-1,-1,-1,-1,-1,-1],
    [3, 11, 2, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 11, 2, 8, 11, 0, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 9, 0, 2, 3, 11, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1,-1,-1,-1,-1,-1,-1],
    [3, 10, 1, 11, 10, 3, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1,-1,-1,-1,-1,-1,-1],
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1,-1,-1,-1,-1,-1,-1],
    [9, 8, 10, 10, 8, 11, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 7, 8, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 3, 0, 7, 3, 4, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 1, 9, 8, 4, 7, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1,-1,-1,-1,-1,-1,-1],
    [1, 2, 10, 8, 4, 7, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1,-1,-1,-1,-1,-1,-1],
    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1,-1,-1,-1,-1,-1,-1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1,-1,-1,-1],
    [8, 4, 7, 3, 11, 2, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1,-1,-1,-1,-1,-1,-1],
    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1,-1,-1,-1,-1,-1,-1],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1,-1,-1,-1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1,-1,-1,-1,-1,-1,-1],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1,-1,-1,-1],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1,-1,-1,-1],
    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1,-1,-1,-1,-1,-1,-1],
    [9, 5, 4, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9, 5, 4, 0, 8, 3, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 5, 4, 1, 5, 0, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1,-1,-1,-1,-1,-1,-1],
    [1, 2, 10, 9, 5, 4, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1,-1,-1,-1,-1,-1,-1],
    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1,-1,-1,-1,-1,-1,-1],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1,-1,-1,-1],
    [9, 5, 4, 2, 3, 11, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1,-1,-1,-1,-1,-1,-1],
    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1,-1,-1,-1,-1,-1,-1],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1,-1,-1,-1],
    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1,-1,-1,-1,-1,-1,-1],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1,-1,-1,-1],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1,-1,-1,-1],
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1,-1,-1,-1,-1,-1,-1],
    [9, 7, 8, 5, 7, 9, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1,-1,-1,-1,-1,-1,-1],
    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1,-1,-1,-1,-1,-1,-1],
    [1, 5, 3, 3, 5, 7, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1,-1,-1,-1,-1,-1,-1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1,-1,-1,-1],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1,-1,-1,-1],
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1,-1,-1,-1,-1,-1,-1],
    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1,-1,-1,-1,-1,-1,-1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1,-1,-1,-1],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1,-1,-1,-1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1,-1,-1,-1,-1,-1,-1],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1,-1,-1,-1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
    [11, 10, 5, 7, 11, 5, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 8, 3, 5, 10, 6, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9, 0, 1, 5, 10, 6, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1,-1,-1,-1,-1,-1,-1],
    [1, 6, 5, 2, 6, 1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1,-1,-1,-1,-1,-1,-1],
    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1,-1,-1,-1,-1,-1,-1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1,-1,-1,-1],
    [2, 3, 11, 10, 6, 5, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1,-1,-1,-1,-1,-1,-1],
    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1,-1,-1,-1,-1,-1,-1],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1,-1,-1,-1],
    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1,-1,-1,-1,-1,-1,-1],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1,-1,-1,-1],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1,-1,-1,-1],
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1,-1,-1,-1,-1,-1,-1],
    [5, 10, 6, 4, 7, 8, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1,-1,-1,-1,-1,-1,-1],
    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1,-1,-1,-1],
    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1,-1,-1,-1,-1,-1,-1],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1,-1,-1,-1],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1,-1,-1,-1],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1,-1,-1,-1,-1,-1,-1],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1,-1,-1,-1],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1,-1,-1,-1],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1,-1,-1,-1],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1,-1,-1,-1],
    [10, 4, 9, 6, 4, 10, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1,-1,-1,-1,-1,-1,-1],
    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1,-1,-1,-1,-1,-1,-1],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1,-1,-1,-1],
    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1,-1,-1,-1,-1,-1,-1],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1,-1,-1,-1],
    [0, 2, 4, 4, 2, 6, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1,-1,-1,-1,-1,-1,-1],
    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1,-1,-1,-1,-1,-1,-1],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1,-1,-1,-1],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1,-1,-1,-1],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1,-1,-1,-1],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1,-1,-1,-1,-1,-1,-1],
    [6, 4, 8, 11, 6, 8, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1,-1,-1,-1,-1,-1,-1],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1,-1,-1,-1],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1,-1,-1,-1],
    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1,-1,-1,-1,-1,-1,-1],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1,-1,-1,-1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1,-1,-1,-1,-1,-1,-1],
    [7, 3, 2, 6, 7, 2, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1,-1,-1,-1],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1,-1,-1,-1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
    [0, 9, 1, 11, 6, 7, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1,-1,-1,-1],
    [7, 11, 6, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7, 6, 11, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3, 0, 8, 11, 7, 6, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 1, 9, 11, 7, 6, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1,-1,-1,-1,-1,-1,-1],
    [10, 1, 2, 6, 11, 7, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1,-1,-1,-1,-1,-1,-1],
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1,-1,-1,-1,-1,-1,-1],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1,-1,-1,-1],
    [7, 2, 3, 6, 2, 7, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1,-1,-1,-1,-1,-1,-1],
    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1,-1,-1,-1,-1,-1,-1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1,-1,-1,-1],
    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1,-1,-1,-1,-1,-1,-1],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1,-1,-1,-1],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1,-1,-1,-1],
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1,-1,-1,-1,-1,-1,-1],
    [6, 8, 4, 11, 8, 6, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1,-1,-1,-1,-1,-1,-1],
    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1,-1,-1,-1,-1,-1,-1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1,-1,-1,-1],
    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1,-1,-1,-1,-1,-1,-1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1,-1,-1,-1],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1,-1,-1,-1],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
    [8, 2, 3, 8, 4, 2, 4, 6, 2, -1,-1,-1,-1,-1,-1,-1],
    [0, 4, 2, 4, 6, 2, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1,-1,-1,-1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1,-1,-1,-1,-1,-1,-1],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1,-1,-1,-1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1,-1,-1,-1,-1,-1,-1],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    [10, 9, 4, 6, 10, 4, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 9, 5, 7, 6, 11, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1,-1,-1,-1,-1,-1,-1],
    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1,-1,-1,-1,-1,-1,-1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1,-1,-1,-1],
    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1,-1,-1,-1,-1,-1,-1],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1,-1,-1,-1],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1,-1,-1,-1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1,-1,-1,-1,-1,-1,-1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1,-1,-1,-1],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1,-1,-1,-1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1,-1,-1,-1],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1,-1,-1,-1],
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1,-1,-1,-1,-1,-1,-1],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1,-1,-1,-1],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1,-1,-1,-1],
    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1,-1,-1,-1,-1,-1,-1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1,-1,-1,-1],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1,-1,-1,-1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1,-1,-1,-1],
    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1,-1,-1,-1,-1,-1,-1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
    [1, 5, 6, 2, 1, 6, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1,-1,-1,-1],
    [0, 3, 8, 5, 6, 10, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10, 5, 6, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 5, 10, 7, 5, 11, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1,-1,-1,-1,-1,-1,-1],
    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1,-1,-1,-1,-1,-1,-1],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1,-1,-1,-1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1,-1,-1,-1,-1,-1,-1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1,-1,-1,-1],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1,-1,-1,-1],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1,-1,-1,-1,-1,-1,-1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1,-1,-1,-1],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1,-1,-1,-1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
    [1, 3, 5, 3, 7, 5, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1,-1,-1,-1,-1,-1,-1],
    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1,-1,-1,-1,-1,-1,-1],
    [9, 8, 7, 5, 9, 7, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1,-1,-1,-1,-1,-1,-1],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1,-1,-1,-1],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1,-1,-1,-1],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1,-1,-1,-1],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    [9, 4, 5, 2, 11, 3, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1,-1,-1,-1],
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1,-1,-1,-1,-1,-1,-1],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1,-1,-1,-1],
    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1,-1,-1,-1,-1,-1,-1],
    [0, 4, 5, 1, 0, 5, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1,-1,-1,-1],
    [9, 4, 5, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1,-1,-1,-1,-1,-1,-1],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1,-1,-1,-1],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1,-1,-1,-1],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1,-1,-1,-1],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1,-1,-1,-1,-1,-1,-1],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1,-1,-1,-1],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1,-1,-1,-1],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
    [1, 10, 2, 8, 7, 4, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1,-1,-1,-1,-1,-1,-1],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1,-1,-1,-1],
    [4, 0, 3, 7, 4, 3, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4, 8, 7, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9, 10, 8, 10, 11, 8, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1,-1,-1,-1,-1,-1,-1],
    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1,-1,-1,-1,-1,-1,-1],
    [3, 1, 10, 11, 3, 10, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1,-1,-1,-1,-1,-1,-1],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1,-1,-1,-1],
    [0, 2, 11, 8, 0, 11, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3, 2, 11, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1,-1,-1,-1,-1,-1,-1],
    [9, 10, 2, 0, 9, 2, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1,-1,-1,-1],
    [1, 10, 2, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1, 3, 8, 9, 1, 8, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 9, 1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0, 3, 8, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
], dtype=np.int32)

# SDF type constants
SDF_SPHERE = 0
SDF_CAPSULE = 1


@njit(cache=True)
def sdf_sphere(x, y, z, cx, cy, cz, radius):
    """Compute sphere SDF at point."""
    dx = x - cx
    dy = y - cy
    dz = z - cz
    return (dx*dx + dy*dy + dz*dz) ** 0.5 - radius


@njit(cache=True)
def sdf_capsule(x, y, z, p0x, p0y, p0z, bax, bay, baz, ba_dot_ba, radius):
    """Compute capsule SDF at point."""
    pax = x - p0x
    pay = y - p0y
    paz = z - p0z
    pa_dot_ba = pax*bax + pay*bay + paz*baz
    h = pa_dot_ba / ba_dot_ba
    if h < 0.0:
        h = 0.0
    elif h > 1.0:
        h = 1.0
    dx = pax - bax * h
    dy = pay - bay * h
    dz = paz - baz * h
    return (dx*dx + dy*dy + dz*dz) ** 0.5 - radius


@njit(cache=True, parallel=True)
def eval_sdf_grid(xs, ys, zs, sdf_types, sdf_params):
    """Evaluate SDF on entire grid using numba parallel."""
    nx, ny, nz = len(xs), len(ys), len(zs)
    grid = np.empty((nx, ny, nz), dtype=np.float64)
    num_sdfs = len(sdf_types)

    for i in prange(nx):
        x = xs[i]
        for j in range(ny):
            y = ys[j]
            for k in range(nz):
                z = zs[k]
                # Evaluate all SDFs and take minimum (union)
                min_dist = np.inf
                for s in range(num_sdfs):
                    sdf_type = sdf_types[s]
                    params = sdf_params[s]
                    if sdf_type == SDF_SPHERE:
                        dist = sdf_sphere(x, y, z, params[0], params[1], params[2], params[3])
                    else:  # SDF_CAPSULE
                        dist = sdf_capsule(x, y, z, params[0], params[1], params[2],
                                          params[3], params[4], params[5], params[6], params[7])
                    if dist < min_dist:
                        min_dist = dist
                grid[i, j, k] = min_dist

    return grid


@njit(cache=True, parallel=True)
def eval_sdf_slice(x, ys, zs, sdf_types, sdf_params):
    """Evaluate SDF on a single x-slice using numba parallel."""
    ny, nz = len(ys), len(zs)
    slice_grid = np.empty((ny, nz), dtype=np.float64)
    num_sdfs = len(sdf_types)

    for j in prange(ny):
        y = ys[j]
        for k in range(nz):
            z = zs[k]
            min_dist = np.inf
            for s in range(num_sdfs):
                sdf_type = sdf_types[s]
                params = sdf_params[s]
                if sdf_type == SDF_SPHERE:
                    dist = sdf_sphere(x, y, z, params[0], params[1], params[2], params[3])
                else:  # SDF_CAPSULE
                    dist = sdf_capsule(x, y, z, params[0], params[1], params[2],
                                      params[3], params[4], params[5], params[6], params[7])
                if dist < min_dist:
                    min_dist = dist
            slice_grid[j, k] = min_dist

    return slice_grid


def eval_sdf_grid_with_progress(xs, ys, zs, sdf_types, sdf_params, progress=True):
    """Evaluate SDF on grid with optional progress bar."""
    if not progress:
        # Use faster bulk evaluation without progress
        return eval_sdf_grid(xs, ys, zs, sdf_types, sdf_params)

    # Slice-by-slice evaluation with progress bar
    nx, ny, nz = len(xs), len(ys), len(zs)
    grid = np.empty((nx, ny, nz), dtype=np.float64)

    try:
        from tqdm.auto import tqdm
        iterator = tqdm(range(nx), desc="SDF eval", unit="slice")
    except ImportError:
        # No tqdm, fall back to fast bulk evaluation
        return eval_sdf_grid(xs, ys, zs, sdf_types, sdf_params)

    for i in iterator:
        grid[i] = eval_sdf_slice(xs[i], ys, zs, sdf_types, sdf_params)

    return grid


def marching_cubes(sdf_func, bounds, step, progress=True):
    """
    Simple marching cubes implementation.

    Args:
        sdf_func: SDF object with get_numba_data() method
        bounds: ((min_x, min_y, min_z), (max_x, max_y, max_z))
        step: Grid step size
        progress: Show progress bar (default True)

    Returns:
        vertices, faces as numpy arrays
    """
    (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds

    # Create grid
    xs = np.arange(min_x, max_x + step, step)
    ys = np.arange(min_y, max_y + step, step)
    zs = np.arange(min_z, max_z + step, step)

    # Get numba-compatible SDF data
    sdf_types, sdf_params = sdf_func.get_numba_data()

    # Evaluate SDF on grid using numba
    grid = eval_sdf_grid_with_progress(xs, ys, zs, sdf_types, sdf_params, progress=progress)

    nx, ny, nz = len(xs), len(ys), len(zs)
    vertices = []
    faces = []
    vertex_map = {}

    # Edge vertex indices for each cube corner pair
    edge_corners = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    corner_offsets = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ])

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corner_vals = np.array([
                    grid[i, j, k], grid[i+1, j, k], grid[i+1, j+1, k], grid[i, j+1, k],
                    grid[i, j, k+1], grid[i+1, j, k+1], grid[i+1, j+1, k+1], grid[i, j+1, k+1],
                ])

                cube_index = 0
                for c in range(8):
                    if corner_vals[c] < 0:
                        cube_index |= (1 << c)

                if cube_index == 0 or cube_index == 255:
                    continue

                base = np.array([xs[i], ys[j], zs[k]])
                corner_pos = base + corner_offsets * step

                edge_verts = {}
                edges = EDGE_TABLE[cube_index]
                for e in range(12):
                    if edges & (1 << e):
                        c1, c2 = edge_corners[e]
                        edge_key = (i, j, k, e)
                        if edge_key not in vertex_map:
                            v1, v2 = corner_vals[c1], corner_vals[c2]
                            p1, p2 = corner_pos[c1], corner_pos[c2]
                            if abs(v1 - v2) < 1e-10:
                                v = p1
                            else:
                                t = -v1 / (v2 - v1)
                                v = p1 + t * (p2 - p1)
                            vertex_map[edge_key] = len(vertices)
                            vertices.append(v)
                        edge_verts[e] = vertex_map[edge_key]

                tri_list = TRI_TABLE_DATA[cube_index]
                idx = 0
                while idx < 16 and tri_list[idx] != -1:
                    faces.append([
                        edge_verts[tri_list[idx]],
                        edge_verts[tri_list[idx + 1]],
                        edge_verts[tri_list[idx + 2]],
                    ])
                    idx += 3

    if not vertices:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)

    return np.array(vertices), np.array(faces, dtype=np.int32)


class SDF:
    """Base class for signed distance functions."""

    def __or__(self, other):
        """Union of two SDFs."""
        return UnionSDF(self, other)

    def __and__(self, other):
        """Intersection of two SDFs."""
        return IntersectionSDF(self, other)

    def __sub__(self, other):
        """Difference of two SDFs."""
        return DifferenceSDF(self, other)

    def bounds(self):
        """Return bounding box as ((min_x, min_y, min_z), (max_x, max_y, max_z))."""
        raise NotImplementedError

    def get_numba_data(self):
        """Return (sdf_types, sdf_params) arrays for numba evaluation."""
        raise NotImplementedError

    def save(self, path, step=0.01, progress=True):
        """Save SDF as mesh file."""
        import trimesh

        b = self.bounds()
        verts, faces = marching_cubes(self, b, step, progress=progress)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(path)


class SphereSDF(SDF):
    def __init__(self, radius, center=(0, 0, 0)):
        self.radius = radius
        self.cx, self.cy, self.cz = float(center[0]), float(center[1]), float(center[2])

    def __call__(self, x, y, z):
        dx, dy, dz = x - self.cx, y - self.cy, z - self.cz
        return (dx*dx + dy*dy + dz*dz) ** 0.5 - self.radius

    def bounds(self):
        r = self.radius
        return ((self.cx - r - 0.1, self.cy - r - 0.1, self.cz - r - 0.1),
                (self.cx + r + 0.1, self.cy + r + 0.1, self.cz + r + 0.1))

    def get_numba_data(self):
        sdf_types = np.array([SDF_SPHERE], dtype=np.int32)
        sdf_params = np.array([[self.cx, self.cy, self.cz, self.radius, 0, 0, 0, 0]], dtype=np.float64)
        return sdf_types, sdf_params


class CapsuleSDF(SDF):
    def __init__(self, p0, p1, radius):
        self.p0x, self.p0y, self.p0z = float(p0[0]), float(p0[1]), float(p0[2])
        self.p1x, self.p1y, self.p1z = float(p1[0]), float(p1[1]), float(p1[2])
        self.bax = self.p1x - self.p0x
        self.bay = self.p1y - self.p0y
        self.baz = self.p1z - self.p0z
        self.ba_dot_ba = self.bax*self.bax + self.bay*self.bay + self.baz*self.baz
        self.radius = float(radius)

    def __call__(self, x, y, z):
        pax, pay, paz = x - self.p0x, y - self.p0y, z - self.p0z
        pa_dot_ba = pax*self.bax + pay*self.bay + paz*self.baz
        h = pa_dot_ba / self.ba_dot_ba
        h = 0.0 if h < 0.0 else (1.0 if h > 1.0 else h)
        dx = pax - self.bax * h
        dy = pay - self.bay * h
        dz = paz - self.baz * h
        return (dx*dx + dy*dy + dz*dz) ** 0.5 - self.radius

    def bounds(self):
        r = self.radius
        mins = (min(self.p0x, self.p1x) - r - 0.1,
                min(self.p0y, self.p1y) - r - 0.1,
                min(self.p0z, self.p1z) - r - 0.1)
        maxs = (max(self.p0x, self.p1x) + r + 0.1,
                max(self.p0y, self.p1y) + r + 0.1,
                max(self.p0z, self.p1z) + r + 0.1)
        return (mins, maxs)

    def get_numba_data(self):
        sdf_types = np.array([SDF_CAPSULE], dtype=np.int32)
        sdf_params = np.array([[self.p0x, self.p0y, self.p0z,
                                self.bax, self.bay, self.baz,
                                self.ba_dot_ba, self.radius]], dtype=np.float64)
        return sdf_types, sdf_params


class UnionSDF(SDF):
    def __init__(self, a, b):
        # Flatten nested unions
        self.children = []
        for s in (a, b):
            if isinstance(s, UnionSDF):
                self.children.extend(s.children)
            else:
                self.children.append(s)

    def __call__(self, x, y, z):
        return min(c(x, y, z) for c in self.children)

    def bounds(self):
        all_bounds = [c.bounds() for c in self.children]
        mins = np.minimum.reduce([b[0] for b in all_bounds])
        maxs = np.maximum.reduce([b[1] for b in all_bounds])
        return (tuple(mins), tuple(maxs))

    def get_numba_data(self):
        all_types = []
        all_params = []
        for c in self.children:
            types, params = c.get_numba_data()
            all_types.append(types)
            all_params.append(params)
        return np.concatenate(all_types), np.concatenate(all_params)


class IntersectionSDF(SDF):
    def __init__(self, a, b):
        self.children = []
        for s in (a, b):
            if isinstance(s, IntersectionSDF):
                self.children.extend(s.children)
            else:
                self.children.append(s)

    def __call__(self, x, y, z):
        return max(c(x, y, z) for c in self.children)

    def bounds(self):
        all_bounds = [c.bounds() for c in self.children]
        mins = np.maximum.reduce([b[0] for b in all_bounds])
        maxs = np.minimum.reduce([b[1] for b in all_bounds])
        return (tuple(mins), tuple(maxs))

    def get_numba_data(self):
        # For intersection, we'd need different logic - not implemented yet
        raise NotImplementedError("Intersection not yet supported with numba")


class DifferenceSDF(SDF):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x, y, z):
        return max(self.a(x, y, z), -self.b(x, y, z))

    def bounds(self):
        return self.a.bounds()

    def get_numba_data(self):
        # For difference, we'd need different logic - not implemented yet
        raise NotImplementedError("Difference not yet supported with numba")


# Convenience functions
def sphere(radius, center=(0, 0, 0)):
    """Create a sphere SDF."""
    return SphereSDF(radius, center)


def capsule(p0, p1, radius):
    """Create a capsule SDF."""
    return CapsuleSDF(p0, p1, radius)
