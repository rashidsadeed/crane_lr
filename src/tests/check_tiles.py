"""Checks for rl_agent/tiles.py."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # add src/ to the search path

import time                                                          # noqa: E402

import numpy as np                                                   # noqa: E402

from baseline import BaselineController                              # noqa: E402
from dynamics import CraneParams                                     # noqa: E402
from env import CraneEnv                                             # noqa: E402
from rl_agent.features import compute_features, warp, FEATURE_NAMES  # noqa: E402
from rl_agent.tiles import TileCoder                                 # noqa: E402

p = CraneParams()
tc = TileCoder(p)
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def shared(a, b, group=None):
    """Tiles two feature vectors have in common (in all groups, or in one group)."""
    ta, tb = tc.active(a), tc.active(b)
    if group is not None:
        g = tc.group_names.index(group)
        ta = ta[:-1].reshape(tc.n_tilings, -1)[:, g]
        tb = tb[:-1].reshape(tc.n_tilings, -1)[:, g]
    return len(set(ta) & set(tb))


E = FEATURE_NAMES.index("e_x")
print(f"weights per action: {tc.n_features:,}   active tiles per state: {tc.n_active}")
for name, size in zip(tc.group_names, tc.group_sizes):
    print(f"   {name:11s} {size:6,d}")

# ---------------------------------------------------------------- basic properties
f = compute_features(np.array([4.0, 0.8, 4.0, -0.2, 0.05, -0.1, 10, 6, 4, 0, 0, 0, 0]), p)
a = tc.active(f)
NA = tc.n_active
check("K1 97 active tiles, all different", NA == 97 and len(a) == NA and len(set(a)) == NA, f"{len(a)} tiles")
check("K2 all indices inside the weight vector", a.min() >= 0 and a.max() < tc.n_features,
      f"range {a.min()}..{a.max()} of {tc.n_features}")
check("K3 bias tile always on", tc.bias_index in a)
check("K4 deterministic", np.array_equal(a, tc.active(f.copy())))

# each group's tiles stay inside its own block of the weight vector
blocks = np.concatenate(([0], np.cumsum(tc.group_sizes)))
per_group = a[:-1].reshape(tc.n_tilings, -1)
ok = all(blocks[g] <= i < blocks[g + 1] for g in range(len(tc.group_names)) for i in per_group[:, g])
check("K5 every group stays in its own block", ok)

# ---------------------------------------------------------------- generalisation
# a tiny change shares almost everything
near = f.copy(); near[E] += 0.01
check("G1 tiny change: almost all tiles shared", shared(f, near) >= 0.9 * NA, f"{shared(f, near)} of {NA} shared")

# a big change in e_x and z_b: groups using them share nothing, the other groups are untouched
far = f.copy(); far[E] = warp(0.3); far[FEATURE_NAMES.index("z_b")] = 0.5
using = ("travel", "hoist", "path_shape", "landing", "slot", "creep")
other = ("swing", "anti_sway", "approach", "under", "settle")   # ("side" depends on z_b via c_lat)
s_using = sum(shared(f, far, g) for g in using)
s_other = sum(shared(f, far, g) for g in other)
check("G2 big change: groups using e_x or z_b share nothing", s_using == 0, f"{s_using} of {8 * len(using)} shared")
check("G2 ... groups not using them are unchanged", s_other == 8 * len(other),
      f"{s_other} of {8 * len(other)} shared")

# within one group, sharing falls off gradually with distance (coarse generalisation)
counts = []
for de in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
    g = f.copy(); g[E] += de
    counts.append(shared(f, g, "travel"))
check("G3 travel group: sharing falls off gradually with distance",
      all(x >= y for x, y in zip(counts, counts[1:])) and counts[0] == 8 and counts[-1] == 0,
      f"shared of 8 at warped e_x + (0, .05, .1, .2, .4, .8): {counts}")

# ---------------------------------------------------------------- landing resolution
# near the target, 3 cm of placement error changes every landing-group tile
on = compute_features(np.array([10.0, 0, 7.5, 0, 0, 0, 10.0, 0, 0, 0, 0, 0, 0]), p)
off = compute_features(np.array([10.03, 0, 7.5, 0, 0, 0, 10.0, 0, 0, 0, 0, 0, 0]), p)
n_land = shared(on, off, "landing")
n_travel = shared(on, off, "travel")
check("R1 landing group tells 0 cm from 3 cm apart", n_land <= 2, f"{n_land} of 8 landing tiles shared")
check("R2 ... while the coarse travel group mostly cannot", n_travel >= 5, f"{n_travel} of 8 travel tiles shared")

# ---------------------------------------------------------------- clipping and coverage
wild = f.copy(); wild[:] = 99.0
w = tc.active(wild)
check("C1 out-of-range values still give valid tiles", len(set(w)) == NA and w.max() < tc.n_features)

env = CraneEnv()
ctl = BaselineController(p)
used = set()
for seed in range(60):
    obs, _ = env.reset(seed=seed)
    ctl.reset(obs)
    while True:
        used.update(tc.active(compute_features(obs, p)).tolist())
        obs, r, term, trunc, _ = env.step_command(*ctl.act(obs))
        if term or trunc:
            break
check("C2 baseline episodes reach a good share of the tiles", len(used) > 1000,
      f"{len(used):,} of {tc.n_features:,} tiles visited in 60 baseline episodes")

# ---------------------------------------------------------------- speed
t0 = time.perf_counter()
for _ in range(20000):
    tc.active(f)
us = (time.perf_counter() - t0) / 20000 * 1e6
check("S1 fast enough (< 50 us per state)", us < 50, f"{us:.1f} us")

print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")