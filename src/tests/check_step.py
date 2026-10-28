"""Hand-checked tests for step(): velocity limit, trolley end stop, hoist top stop."""
import math
from dynamics import CraneParams, step

p = CraneParams()
TOL = 1e-9
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


# ---------------------------------------------------------------- Test 6
# From rest at x = 2, accelerate at +0.5 for 5 s (50 control steps).
# Speed reaches 1.0 m/s at t = 2 s (x = 3.0), then cruises: x = 6.0 at t = 5 s.
s = (2.0, 0.0, 4.0, 0.0, 0.0, 0.0)
max_speed = 0.0
for k in range(1, 51):
    s, info = step(s, 0.5, 0.0, p, return_substeps=True)
    max_speed = max(max_speed, max(sub[1] for sub in info["substeps"]))
    if k == 20:
        check("T6 speed at t=2s", math.isclose(s[1], 1.0, abs_tol=TOL), f"x_dot={s[1]:.12f}")
        check("T6 position at t=2s", math.isclose(s[0], 3.0, abs_tol=TOL), f"x={s[0]:.12f}")
check("T6 speed at t=5s", math.isclose(s[1], 1.0, abs_tol=TOL), f"x_dot={s[1]:.12f}")
check("T6 position at t=5s", math.isclose(s[0], 6.0, abs_tol=TOL), f"x={s[0]:.12f}")
check("T6 speed never above limit", max_speed <= p.vx_max + TOL, f"max={max_speed:.12f}")
check("T6 no stop flags", not info["x_stop_hit"] and not info["l_stop_hit"])

# ---------------------------------------------------------------- Test 7a
# Cruising right at 1 m/s from x = 11.5 with no command: reaches the
# right end stop (12 m) after about 0.5 s. Run 8 steps to be safely past it.
s = (11.5, 1.0, 4.0, 0.0, 0.0, 0.0)
x_flag_seen = False
for _ in range(8):
    s, info = step(s, 0.0, 0.0, p)
    x_flag_seen = x_flag_seen or info["x_stop_hit"]
check("T7a clamped at x_max", math.isclose(s[0], p.x_max, abs_tol=TOL), f"x={s[0]:.12f}")
check("T7a trolley stopped", s[1] == 0.0, f"x_dot={s[1]}")
check("T7a x_stop_hit flagged", x_flag_seen)

# ---------------------------------------------------------------- Test 7b
# Hoist from l = 2.0 at a_l = -0.4. Hoist speed saturates at -0.5 m/s after
# 1.25 s (0.3125 m), then covers the last 0.1875 m in 0.375 s:
# top stop reached at t = 1.625 s, i.e. during control step 17.
s = (5.0, 0.0, 2.0, 0.0, 0.0, 0.0)
first_hit = None
for k in range(1, 26):
    s, info = step(s, 0.0, -0.4, p)
    if info["l_stop_hit"] and first_hit is None:
        first_hit = k
check("T7b first hit in step 17", first_hit == 17, f"first hit at step {first_hit}")
check("T7b clamped at l_min", math.isclose(s[2], p.l_min, abs_tol=TOL), f"l={s[2]:.12f}")
check("T7b hoist stopped", s[3] == 0.0, f"l_dot={s[3]}")

print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")