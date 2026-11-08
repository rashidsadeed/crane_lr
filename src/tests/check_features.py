"""Hand-checked cases for rl_agent/features.py (grows with each piece)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # add src/ to the search path

import math                                                          # noqa: E402

import numpy as np                                                   # noqa: E402

from dynamics import CraneParams                                     # noqa: E402
from rl_agent.features import (warp, unpack, motion_features, path_features,   # noqa: E402
                               target_features, side_features, compute_features,
                               feature_ranges, FEATURE_NAMES, G_LAT_OPEN, LAT_DZ)

p = CraneParams()
TOL = 1e-6
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def close(a, b):
    return math.isclose(a, b, abs_tol=TOL)


# ---------------------------------------------------------------- warp
for u, expected in ((4.0, 2.0), (-4.0, -2.0), (0.25, 0.5), (0.0, 0.0)):
    check(f"W warp({u})", close(warp(u), expected), f"-> {warp(u)} (expected {expected})")

# ---------------------------------------------------------------- unpack
# trolley at 4 m moving right at 0.8 m/s, hoisting at 0.2 m/s, swung forward 0.05 rad,
# target at 10, one 2-high stack (top 4 m) at 6, two empty slots
obs = np.array([4.0, 0.8, 4.0, -0.2, 0.05, -0.1,  10.0,  6.0, 4.0,  0, 0,  0, 0])
q = unpack(obs, p)
expected = {"x": 4.0, "x_dot": 0.8, "l_dot": -0.2, "L": 5.0, "x_B": 10.0,
            "x_p": 4.249896, "z_b": 4.0, "d": 0.249896, "v_d": -0.509371}
for key, value in expected.items():
    got = q.get(key)
    check(f"U {key}", got is not None and close(got, value), f"-> {got} (expected {value})")
check("U stacks: empty slots skipped", q.get("stacks") == [(6.0, 4.0)], f"-> {q.get('stacks')}")

# ---------------------------------------------------------------- motion_features
f = motion_features(q)
expected = (2.397937, 0.8, 4.0, -0.2, 0.499896, -0.713702, 5.0)
names = ("e_x", "x_dot", "z_b", "l_dot", "d", "v_d", "L")
check("M seven motion features", len(f) == 7, f"got {len(f)}")
for name, got, value in zip(names, f, expected):
    check(f"M {name}", close(got, value), f"-> {got:.6f} (expected {value})")

# container on the floor at the start, hanging straight: no swing, full distance
q0 = unpack(np.array([1.0, 0, 8.0, 0, 0, 0,  9.0,  0, 0, 0, 0, 0, 0]), p)
f0 = motion_features(q0)
check("M at rest at the start", close(f0[0], math.sqrt(8.0)) and close(f0[2], 0.0)
      and f0[4] == 0.0 and f0[5] == 0.0 and close(f0[6], 9.0),
      f"e_x={f0[0]:.4f} (sqrt 8 = 2.8284), z_b={f0[2]}, d={f0[4]}, v_d={f0[5]}, L={f0[6]}")
check("U no stacks at all", q0["stacks"] == [], f"-> {q0['stacks']}")

# overshoot: container 0.09 m past the target -> negative e_x
q1 = unpack(np.array([10.09, 0, 8.0, 0, 0, 0,  10.0,  0, 0, 0, 0, 0, 0]), p)
check("M overshoot gives negative e_x", close(motion_features(q1)[0], -0.3),
      f"-> {motion_features(q1)[0]:.4f} (expected -0.3 = -sqrt(0.09))")

# ---------------------------------------------------------------- path_features (piece 2)
def ob(x, l, theta, x_B, stacks):
    """Observation at rest-velocities with the given position, cable, angle, target, stacks."""
    o = np.zeros(13)
    o[0], o[2], o[4], o[6] = x, l, theta, x_B
    for i, (c, top) in enumerate(stacks):
        o[7 + 2 * i], o[8 + 2 * i] = c, top
    return o


path_cases = [
    # name,                          observation,                                  (c_path, g_next, c_below)
    ("approaching, below the top",   ob(2.0, 4.5, 0.05, 10, [(6, 4)]),             (-0.5, 1.32511, 3.5)),
    ("above the stack",              ob(6.0, 3.5, 0.0, 10, [(6, 4)]),              (0.5, 5.0, 0.5)),
    ("past the stack",               ob(8.5, 3.5, 0.0, 10, [(6, 4)]),              (4.5, 5.0, 4.5)),
    ("lifting at the start",         ob(1.0, 8.0, 0.0, 10, [(4.0, 4)]),            (-4.0, 0.6, 0.0)),
    ("moving left",                  ob(10.0, 7.0, 0.0, 5, [(7, 2)]),              (-1.0, 0.6, 1.0)),
    ("no stacks",                    ob(3.0, 6.0, 0.0, 9, []),                     (2.0, 5.0, 2.0)),
    ("two stacks, taller second",    ob(1.0, 5.0, 0.0, 11, [(4.5, 2), (8.0, 4)]),  (-1.0, 4.6, 3.0)),
]
for name, o, expected in path_cases:
    got = path_features(unpack(o, p), p)
    ok = len(got) == 3 and all(math.isclose(g, e, abs_tol=1e-5) for g, e in zip(got, expected))
    check(f"P {name}", ok, f"-> {tuple(round(g, 5) for g in got)} (expected {expected})")

# ---------------------------------------------------------------- target_features (piece 3)
target_cases = [
    # name,                               observation,                                 (g_side, h_side)
    ("slot: neighbours on both sides",    ob(2.0, 5.0, 0.0, 7, [(3.6, 2), (10.4, 4)]), (1.0, 4.0)),
    ("behind: 0.6 m left neighbour",      ob(2.0, 5.0, 0.0, 10, [(7.0, 4)]),           (0.6, 4.0)),
    ("open floor",                        ob(2.0, 5.0, 0.0, 9, []),                    (3.0, 0.0)),
    ("path stack 2.6 m away counts",      ob(1.0, 5.0, 0.0, 10, [(5.0, 4)]),           (2.6, 4.0)),
    ("path stack 3.6 m away ignored",     ob(1.0, 5.0, 0.0, 10, [(4.0, 4)]),           (3.0, 0.0)),
]
for name, o, expected in target_cases:
    got = target_features(unpack(o, p), p)
    ok = len(got) == 2 and all(math.isclose(g, e, abs_tol=1e-9) for g, e in zip(got, expected))
    check(f"T {name}", ok, f"-> {tuple(round(g, 5) for g in got)} (expected {expected})")

# ---------------------------------------------------------------- side features (piece 5)
W = p.w_c
def zb(l):
    return unpack(ob(0.0, l, 0.0, 10, []), p)["z_b"]
q0 = unpack(ob(2.0, 4.5, 0.0, 10, []), p)
th = 0.05
d_th = unpack(ob(2.0, 4.5, th, 10, []), p)["d"]          # > 0: container swung to the right
side_cases = [
    # name,                                     observation,                          expected
    ("no stacks: open",                         ob(2.0, 4.5, 0.0, 10, []),            (G_LAT_OPEN, LAT_DZ, 0.0, 0.0)),
    ("tall stack on the right",                 ob(2.0, 4.5, 0.0, 10, [(5, 4)]),
        (3.0 - W, zb(4.5) - 4, 0.0, 0.0)),
    ("stack far below the container ignored",   ob(2.0, 4.5, 0.0, 10, [(5, 1)]),      (G_LAT_OPEN, LAT_DZ, 0.0, 0.0)),
    ("top just below the container counts",     ob(2.0, 4.5, 0.0, 10, [(5, zb(4.5) - 0.3)]),
        (3.0 - W, 0.3, 0.0, 0.0)),
    ("nearest of two; left; swinging away",     ob(2.0, 4.5, th, 10, [(7, 4), (2.0 - W - 0.4 + d_th, 4)]),
        (0.4, zb(4.5) - 4, warp(-d_th), 0.0)),
    ("far gap clipped to G_LAT_OPEN",           ob(2.0, 4.5, 0.0, 10, [(9, 4)]),      (G_LAT_OPEN, zb(4.5) - 4, 0.0, 0.0)),
]
for name, o, expected in side_cases:
    got = side_features(unpack(o, p), p)
    ok = len(got) == 4 and all(math.isclose(g, e, abs_tol=1e-9) for g, e in zip(got, expected))
    check(f"S {name}", ok, f"-> {tuple(round(g, 5) for g in got)} (expected {tuple(round(e, 5) for e in expected)})")
qv = unpack(ob(2.0, 4.5, 0.0, 10, [(1.0 - W, 4)]), p); qv["x_dot"] = 0.3
check("S trolley speed is signed towards the stack (stack left, moving right -> negative)",
      math.isclose(side_features(qv, p)[3], -0.3), f"-> {side_features(qv, p)[3]}")

# ---------------------------------------------------------------- compute_features (piece 4)
o = ob(2.0, 4.5, 0.05, 10, [(6, 4)])
q = unpack(o, p)
f = compute_features(o, p)
parts = motion_features(q) + path_features(q, p) + target_features(q, p) + side_features(q, p)
NF = len(FEATURE_NAMES)
check("C 16 features, in FEATURE_NAMES order", f.shape == (16,) and NF == 16
      and np.allclose(f, parts), f"shape {f.shape}")

R = np.array(feature_ranges(p))
check("C ranges: one (low, high) per feature, low < high",
      R.shape == (NF, 2) and np.all(R[:, 0] < R[:, 1]), f"shape {R.shape}")

# every feature stays inside its range over complete baseline episodes, all layout types
from baseline import BaselineController                              # noqa: E402
from env import CraneEnv                                             # noqa: E402

env = CraneEnv()
ctl = BaselineController(p)
lo, hi = np.full(NF, np.inf), np.full(NF, -np.inf)
for seed in range(100):
    obs, _ = env.reset(seed=seed)
    ctl.reset(obs)
    while True:
        f = compute_features(obs, p)
        lo, hi = np.minimum(lo, f), np.maximum(hi, f)
        obs, r, term, trunc, _ = env.step_command(*ctl.act(obs))
        if term or trunc:
            break
outside = [n for n, a, b, (rl, rh) in zip(FEATURE_NAMES, lo, hi, R)
           if a < rl - 1e-6 or b > rh + 1e-6]
check("C baseline episodes stay inside the feature ranges", not outside,
      f"outside: {outside}" if outside else "100 episodes")

print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")