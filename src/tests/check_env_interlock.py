"""Checks for the load-on-ground interlock."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import dynamics as dyn                               # noqa: E402
import geometry as geo                               # noqa: E402
from env import CraneEnv                             # noqa: E402
from scenarios import Layout                         # noqa: E402

env = CraneEnv()
env.reset(options={"layout": Layout("S0", 1.0, 8.0, ())})
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


# while resting at A
cases = [((0.5, 0.4), (0.0, 0.0), "travel and lower blocked"),
         ((-0.2, -0.4), (0.0, -0.4), "travel blocked, hoisting allowed"),
         ((0.0, 0.0), (0.0, 0.0), "doing nothing allowed")]
for cmd, expected, what in cases:
    got = env._apply_interlock(*cmd)
    check(f"I1 on start: {what}", got == expected, f"{cmd} -> {got}")

# hoist from the floor until the container is clear: interlock must release
s = env.state
released_at = None
for k in range(1, 11):
    a_x, a_l = env._apply_interlock(0.5, -0.4)       # agent tries to travel and hoist
    s, _ = dyn.step(s, a_x, a_l, env.p)
    env._update_on_start(geo.container_corners(s, env.p))
    if not env.on_start and released_at is None:
        released_at = k
        x_at_release = s[0]
check("I2 interlock releases after lift-off", released_at == 3,
      f"released at step {released_at} (expected 3: 0.01 m needs ~0.22 s at 0.4 m/s^2)")
check("I3 trolley did not move before release", x_at_release == 1.0, f"x = {x_at_release}")

# after release, commands pass through
check("I4 after lift-off: commands unchanged", env._apply_interlock(0.5, 0.4) == (0.5, 0.4))

# one-way: lowering back to the floor does not re-engage it
env.state = (env.state[0], 0.0, env.p.l_max, 0.0, 0.0, 0.0)
env._update_on_start(geo.container_corners(env.state, env.p))
check("I5 interlock does not re-engage", env.on_start is False)

print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")