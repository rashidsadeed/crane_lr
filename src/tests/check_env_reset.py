"""Checks for CraneEnv.reset and _get_obs."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np                                   # noqa: E402

from env import CraneEnv                             # noqa: E402
from scenarios import Layout, Stack, canonical_layouts   # noqa: E402

env = CraneEnv()
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


# a given layout is used as-is
L = Layout("S4", 1.0, 7.0, (Stack(3.6, 1), Stack(10.4, 2)))
obs, info = env.reset(options={"layout": L})
expected = np.array([1.0, 0, 8.0, 0, 0, 0,  7.0,  3.6, 2.0,  10.4, 4.0,  0, 0])
check("R1 observation for a given layout", np.allclose(obs, expected), f"\n      {obs}")
check("R2 info carries the layout", info["layout"] == L)
check("R3 observation inside the observation space", env.observation_space.contains(obs))
check("R4 bookkeeping reset", env.t == 0 and env.on_start and env.prev_cmd == (0.0, 0.0))

# same seed -> same episode; different seed -> different episode
o1, _ = env.reset(seed=42)
o2, _ = env.reset(seed=42)
o3, _ = env.reset(seed=43)
check("R5 same seed -> same observation", np.array_equal(o1, o2))
check("R6 different seed -> different observation", not np.array_equal(o1, o3))

# many random resets stay inside the space; empty slots are zero
ok, ok_pad = True, True
for s in range(500):
    o, info = env.reset(seed=s)
    ok &= env.observation_space.contains(o)
    n = len(info["layout"].stacks)
    ok_pad &= np.all(o[7 + 2 * n:] == 0.0)
check("R7 500 random resets inside the space", ok)
check("R8 unused stack slots are zero", ok_pad)

# canonical layouts work through options
ok = all(env.reset(options={"layout": L})[0][6] == 10.0 for L in canonical_layouts(env.p).values())
check("R9 canonical layouts accepted", ok)

print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")