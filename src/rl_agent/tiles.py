"""Tile coding: 16 features -> the indices of the active tiles (a sparse binary feature vector).

Each GROUP tiles a few features together with n_tilings overlapping grids. For every
grid, a state falls into exactly one tile, so each group contributes n_tilings active
tiles; with 12 groups x 8 tilings + 1 bias tile, every state has exactly 97 active tiles.

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
    # Precision groups (added after C3/C4 capped at ~85%/~80%). They come after the original
    # nine, so the original groups keep their positions in the weight vector.
    # settle: swing inside +-15 cm / +-25 cm/s. The coarse swing groups put the whole 5 cm
    #         tolerance band inside about one tile, so 3 cm and 7 cm of swing looked the same.
    ("settle",     (("d", 8, (warp(-0.15), warp(0.15))), ("v_d", 8, (warp(-0.25), warp(0.25))),
                    ("L", 3, None))),
    # creep: last 20 cm of placement and trolley speed within +-0.2 m/s (tolerance 5 cm/s; the
    #        coarse groups use 0.25 m/s-wide speed tiles), during the last metre of descent.
    ("creep",      (("e_x", 10, (warp(-0.2), warp(0.2))), ("x_dot", 8, (-0.2, 0.2)),
                    ("z_b", 4, (0.0, 1.0)))),
    # side: sideways clearance to a stack beside the container (tight slots, lift-off).
    # The coarse e_x column separates "beside a stack at the target" (lowering is right) from
    # "beside a stack at the start" (lowering is a floor collision); without it, a probe agent
    # carried "lower" from slot landings over to lift-off and set the container back down at A.
    ("side",       (("g_lat", 8, (0.0, 1.2)), ("c_lat", 4, (-1.5, 0.5)),
                    ("d_lat", 6, (warp(-0.4), warp(0.4))), ("xd_lat", 4, (-0.5, 0.5)),
                    ("e_x", 4, (warp(-1.5), warp(1.5))))),
)


@dataclass
class _Column:
    """One dimension of one group."""
    feature: int       # index into the features
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

    def check_agent(self, agent, source="") -> None:
        """Refuse weights made for a different tile layout (e.g. the 26,081-tile coder used
        before the precision groups were added) instead of indexing them wrongly."""
        if agent.w.shape[1] != self.n_features or agent.n_active != self.n_active:
            raise SystemExit(
                f"weights {source} have {agent.w.shape[1]:,} tiles / {agent.n_active} active, the "
                f"current tile coder has {self.n_features:,} / {self.n_active}: they come from an "
                f"older feature layout and cannot be used with this code")

    def active(self, features: np.ndarray) -> np.ndarray:
        """Indices of the n_active tiles that are on for this feature vector."""
        s = (features[self.feat] - self.lo) / self.span * self.n        # 0..n across the range
        s = np.clip(s, 0.0, self.n - 1e-9)                              # clip out-of-range values
        coords = np.floor(s[None, :] + self.offset).astype(np.int64)    # (T, C)
        cell = np.add.reduceat(coords * self.stride, self.group_start, axis=1)   # (T, G)
        return np.append((self.base + cell).ravel(), self.bias_index)