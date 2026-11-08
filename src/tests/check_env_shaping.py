"""Checks for CraneEnv._required_height and _potential."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np                                   # noqa: E402

import dynamics as dyn                               # noqa: E402
from env import CraneEnv, TaskConfig                 # noqa: E402
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

# funnel: the ramp after the stack edge is continuous (old rule: a 2.6 jump at the edge,
# here passing through it below the top, i.e. inside the stack's 10 cm collision margin)
xs = np.arange(7.2, 7.8, 0.001)
D = np.array([-env._potential(state(x, 3.0)) for x in xs])
check("P funnel: no jump in D while passing beside the stack below its top",
      np.abs(np.diff(D)).max() < 0.1, f"largest 1 mm change {np.abs(np.diff(D)).max():.3f}")
old_env = CraneEnv(task_cfg=TaskConfig(funnel=False))
old_env.reset(options={"layout": Layout("S1", 0.0, 10.0, (Stack(5.0, 2),))})
check("P funnel=False keeps the old rule (stack gone 1 cm past its edge)",
      old_env._required_height(7.41) == 0.0 and env._required_height(7.41) > 4.0,
      f"old {old_env._required_height(7.41)}, funnel {env._required_height(7.41):.2f}")

# funnel in a tight slot: target at 6, 2-high neighbour 0.4 m past the target's footprint
W = p.w_c
env.reset(options={"layout": Layout("S3", 0.0, 6.0, (Stack(6.0 + W + 0.4, 2),))})
slot = [("lined up",                          6.00, 0.0, 0.0),
        ("0.25 m towards it: half weight",     6.25, 0.0, 0.5 * 4.3),
        ("0.35 m towards it: inside margin",   6.35, 0.0, 4.3),
        ("0.15 m off, still: clear",           6.15, 0.0, 0.0),
        ("0.15 m off, 10 cm swing: funnel",    6.15, 0.1, 0.25 * 4.3)]
for name, x, A, h in slot:
    got = env._required_height(x, A)
    check(f"P slot: {name}", abs(got - h) < 1e-9, f"h_req {got:.3f} (expected {h:.3f})")

# lined up over the target, descent is always allowed (even at maximum funnel width)
from scenarios import ScenarioConfig, sample_layout, canonical_layouts   # noqa: E402
from rl_agent.train import STAGES                                         # noqa: E402
bad = []
for stage in ("C1", "C2", "C3", "C4"):
    scen, _ = STAGES[stage]
    e = CraneEnv(scen_cfg=scen)
    rng = np.random.default_rng(1)
    lays = [sample_layout(rng, scen, p) for _ in range(300)] + list(canonical_layouts(p).values())
    for L in lays:
        e.reset(options={"layout": L})
        if e._required_height(L.x_B, 5.0) != 0.0:
            bad.append((stage, L))
check("P funnel never blocks descent lined up over the target (C1-C4, 1200 layouts + canonical)",
      not bad, f"{len(bad)} blocked" if bad else "")
try:
    CraneEnv(scen_cfg=ScenarioConfig(g_min=0.25))
    check("P funnel wider than the narrowest slot is refused", False)
except AssertionError:
    check("P funnel wider than the narrowest slot is refused", True)

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