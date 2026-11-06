"""Checks for CraneEnv._detect_event."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import dynamics as dyn                               # noqa: E402
from env import CraneEnv                             # noqa: E402
from scenarios import Layout, Stack                  # noqa: E402

env = CraneEnv()
p = env.p
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def run_step(layout, s, a_x, a_l, lifted=True):
    """Put the env in state s, take one control step, return the detected event."""
    env.reset(options={"layout": layout})
    env.state = s
    env.on_start = not lifted
    _, info = dyn.step(s, a_x, a_l, p, return_substeps=True)
    return env._detect_event(s, info)


open_floor = Layout("S0", 1.0, 8.0, ())
one_stack = Layout("S1", 0.0, 10.0, (Stack(5.0, 2),))          # stack x 3.8..6.2, top 4.0

# E1: lifting off the floor at the start: floor contact is not a touchdown
ev, s, d = run_step(open_floor, (1.0, 0, 8.0, 0, 0, 0), 0.0, -0.4, lifted=False)
check("E1 lifting at the start: no event", ev is None, f"event={ev}")

# E2: moving right at 1 m/s, low enough to hit the stack; right edge 0.15 m from it
s0 = (3.8 - 0.15 - 1.2, 1.0, 6.0, 0.0, 0.0, 0.0)              # container bottom at z = 2
ev, s, d = run_step(one_stack, s0, 0.0, 0.0)
check("E2 collision with the stack", ev == "collision" and d == {"stack": 0},
      f"event={ev}, detail={d}, right edge at {s[0] + 1.2:.3f} (stack edge 3.8, margin 0.1)")

# E3: same, but high enough to pass over the stack (bottom at 4.2 > 4.0 + 0.1)
s0 = (3.8 - 0.15 - 1.2, 1.0, 10.0 - 4.2 - 2.0, 0.0, 0.0, 0.0)
ev, s, d = run_step(one_stack, s0, 0.0, 0.0)
check("E3 clear above the stack: no event", ev is None, f"event={ev}")

# E4: lowering at 0.2 m/s, 5 mm above the floor (overshoots 8.0 within a substep)
ev, s, d = run_step(open_floor, (8.0, 0, 7.995, 0.2, 0, 0), 0.0, 0.0)
check("E4 touchdown detected", ev == "touchdown", f"event={ev}")
check("E4 landing speed taken before the floor clamp", abs(d.get("v_land", 0) - 0.2) < 1e-9,
      f"v_land={d.get('v_land')} (state l_dot after clamp = {s[3]})")

# E5: tilted container touches with a corner before l reaches 8
ev, s, d = run_step(open_floor, (8.0, 0, 7.95, 0.2, 0.05, 0), 0.0, 0.0)
check("E5 tilted: corner touches first", ev == "touchdown" and s[2] < p.l_max,
      f"event={ev}, l at contact = {s[2]:.4f} (< 8.0)")

# E6: trolley runs into the right end buffer
ev, s, d = run_step(open_floor, (11.95, 1.0, 5.0, 0, 0, 0), 0.0, 0.0)
check("E6 end stop", ev == "end_stop", f"event={ev}")

# E7: hoisting at full speed into the upper stop: hard stop
ev, s, d = run_step(open_floor, (6.0, 0, 1.53, -0.5, 0, 0), 0.0, -0.4)
check("E7 fast into the upper stop: hard_top_stop", ev == "hard_top_stop",
      f"event={ev}, detail={d}")

# E8: creeping into the upper stop: allowed
ev, s, d = run_step(open_floor, (6.0, 0, 1.505, -0.1, 0, 0), 0.0, 0.0)
check("E8 slow into the upper stop: no event", ev is None and s[2] == p.l_min,
      f"event={ev}, l={s[2]}")

print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")