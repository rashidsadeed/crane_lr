"""Checks for CraneEnv._required_height and _potential."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np                                   # noqa: E402

import dynamics as dyn                               # noqa: E402
from env import CraneEnv                             # noqa: E402
from scenarios import Layout, Stack                  # noqa: E402

env = CraneEnv()
p = env.p
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def state(x, z_b):
    """Container hanging straight with its bottom at height z_b."""
    return (x, 0.0, p.H - z_b - p.h_c, 0.0, 0.0, 0.0)


# canonical: A = 0, B = 10, 2-high stack at 5 (x 3.8..6.2, top 4.0) -> h_req = 4.3
env.reset(options={"layout": Layout("S1", 0.0, 10.0, (Stack(5.0, 2),))})
cases = [
    ("at the start, on the floor",       state(0.0, 0.0), 18.6),   # 10 + 4.3 + 4.3
    ("lifted to 4.3 m at the start",     state(0.0, 4.3), 14.3),   # 10 + 0 + 4.3
    ("still over the stack edge, x=7.3", state(7.3, 4.3),  7.0),   # strip lo 6.1 < 6.2: counts
    ("just past the stack, x=7.6",       state(7.6, 4.3),  6.7),   # strip lo 6.4 > 6.2: gone
    ("lifted higher than needed, 5 m",   state(7.6, 5.0),  7.4),   # descent from 5 m
    ("placed at the target",             state(10.0, 0.0), 0.0),
]
for name, s, D in cases:
    got = -env._potential(s)
    check(f"P {name}", abs(got - D) < 1e-9, f"D = {got:.3f} (expected {D})")

# continuity: passing the stack edge at the required height does not make D jump
D_before = -env._potential(state(7.399, 4.3))
D_after = -env._potential(state(7.401, 4.3))
check("P no jump when the stack drops out of the strip", abs(D_before - D_after - 0.002) < 1e-9,
      f"{D_before:.4f} -> {D_after:.4f} (only the 2 mm of travel)")

# a stack beside the target (S3) is never 'in the way'
env.reset(options={"layout": Layout("S3", 0.0, 6.0, (Stack(9.0, 2),))})
check("P stack right of the target ignored", env._required_height(0.0) == 0.0,
      f"h_req = {env._required_height(0.0)}")

# telescoping: with Phi = 0 at the end, total discounted shaping = -Phi(s0), whatever the path
env.reset(options={"layout": Layout("S1", 0.0, 10.0, (Stack(5.0, 2),))})
rng = np.random.default_rng(0)
gamma = env.task.gamma
totals = []
for _ in range(5):
    s = state(0.0, 1.0)
    phi0 = env._potential(s)
    total, disc = 0.0, 1.0
    for t in range(int(rng.integers(20, 200))):          # random path, random length
        s_next, _ = dyn.step(s, rng.choice([-0.5, 0, 0.5]), rng.choice([-0.4, 0, 0.4]), p)
        total += disc * (gamma * env._potential(s_next) - env._potential(s))
        disc *= gamma
        s = s_next
    total += disc * (gamma * 0.0 - env._potential(s))    # terminal step: Phi = 0
    totals.append(float(total))
check("P shaping sums to -Phi(s0) on any terminating path",
      all(abs(T + phi0) < 1e-9 for T in totals),
      f"totals {[round(T, 6) for T in totals]}, -Phi(s0) = {-phi0}")

print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")