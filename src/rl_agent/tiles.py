"""Tile coding: 12 features -> the indices of the active tiles (a sparse binary feature vector).

Each GROUP tiles a few features together with n_tilings overlapping grids. For every
grid, a state falls into exactly one tile, so each group contributes n_tilings active
tiles; with 9 groups x 8 tilings + 1 bias tile, every state has exactly 73 active tiles.

    Q(s, a) = sum of w[a, i] over the active tiles i        (a linear function)

Direct indexing (no hashing): every tile has a fixed position in the weight vector,
    index = group_offset + tiling * cells_per_tiling + cell_within_the_tiling
so there are no collisions.

Grid offsets: tiling t shifts dimension j of a group by ((t * (2j + 1)) mod T) / T of a
tile width (asymmetric displacement, Sutton & Barto 2018, sec. 9.5.4). Shifting every
dimension by the same amount would line the tilings up along the diagonal.
"""
from dataclasses import dataclass

import numpy as np

from rl_agent.features import FEATURE_NAMES, feature_ranges, warp

# name -> list of (feature, tiles across its range, optional (low, high) override)
GROUPS = (
    ("travel",     (("e_x", 10, None), ("x_dot", 8, None))),
    ("swing",      (("d", 8, None), ("v_d", 8, None), ("L", 4, None))),
    ("anti_sway",  (("d", 8, None), ("v_d", 8, None), ("x_dot", 6, None))),
    ("hoist",      (("z_b", 8, None), ("l_dot", 6, None))),
    ("path_shape", (("e_x", 12, None), ("z_b", 8, None))),
    ("approach",   (("c_path", 8, None), ("g_next", 6, None), ("x_dot", 6, None))),
    ("under",      (("c_below", 6, None), ("l_dot", 6, None))),
    # fine grid for the last 0.5 m and the last 2 m of height: precise placement
    ("landing",    (("e_x", 10, (warp(-0.5), warp(0.5))), ("d", 6, None), ("z_b", 6, (0.0, 2.0)))),
    ("slot",       (("g_side", 4, None), ("h_side", 3, None), ("d", 6, None), ("z_b", 6, None))),
)


@dataclass
class _Column:
    """One dimension of one group."""
    feature: int       # index into the 12 features
    n: int             # tiles across the range
    lo: float
    hi: float


class TileCoder:
    def __init__(self, params, groups=GROUPS, n_tilings: int = 8):
        self.n_tilings = T = n_tilings
        ranges = feature_ranges(params)
        cols, group_of_col, offsets_base = [], [], []
        self.group_names, sizes = [], []
        for g, (name, dims) in enumerate(groups):
            self.group_names.append(name)
            cells = 1
            for j, (feat, n, rng) in enumerate(dims):
                f = FEATURE_NAMES.index(feat)
                lo, hi = rng if rng is not None else ranges[f]
                cols.append(_Column(f, n, lo, hi))
                group_of_col.append(g)
                cells *= n + 1               # +1: offsets can push a value into one extra cell
            sizes.append(cells)

        C = len(cols)
        self.feat = np.array([c.feature for c in cols])
        self.n = np.array([c.n for c in cols], dtype=float)
        self.lo = np.array([c.lo for c in cols])
        self.span = np.array([c.hi - c.lo for c in cols])

        # stride of each column inside its group's grid (row-major)
        self.stride = np.zeros(C, dtype=np.int64)
        for g in range(len(groups)):
            idx = [k for k in range(C) if group_of_col[k] == g]
            s = 1
            for k in reversed(idx):
                self.stride[k] = s
                s *= int(self.n[k]) + 1
        # first column of each group, for summing a group's columns with np.add.reduceat
        self.group_start = np.array([group_of_col.index(g) for g in range(len(groups))])

        # tiling offsets, in tile widths: (T, C)
        j_in_group = np.array([sum(1 for k in range(c) if group_of_col[k] == group_of_col[c])
                               for c in range(C)])
        t = np.arange(T)[:, None]
        self.offset = ((t * (2 * j_in_group[None, :] + 1)) % T) / T

        # where each (tiling, group) block starts in the weight vector: (T, G)
        sizes = np.array(sizes, dtype=np.int64)
        group_base = np.concatenate(([0], np.cumsum(sizes * T)[:-1]))
        self.base = group_base[None, :] + np.arange(T)[:, None] * sizes[None, :]
        self.bias_index = int((sizes * T).sum())
        self.n_features = self.bias_index + 1
        self.n_active = T * len(groups) + 1
        self.group_sizes = sizes * T

    def active(self, features: np.ndarray) -> np.ndarray:
        """Indices of the n_active tiles that are on for this feature vector."""
        s = (features[self.feat] - self.lo) / self.span * self.n        # 0..n across the range
        s = np.clip(s, 0.0, self.n - 1e-9)                              # clip out-of-range values
        coords = np.floor(s[None, :] + self.offset).astype(np.int64)    # (T, C)
        cell = np.add.reduceat(coords * self.stride, self.group_start, axis=1)   # (T, G)
        return np.append((self.base + cell).ravel(), self.bias_index)