"""Checks for CraneEnv._reward."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math                                          # noqa: E402

from env import CraneEnv                             # noqa: E402
from scenarios import Layout                         # noqa: E402

env = CraneEnv()
env.reset(options={"layout": Layout("S0", 0.0, 10.0, ())})    # open floor, B = 10
p, task = env.p, env.task
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def state(x, z_b, x_dot=0.0, theta_dot=0.0):
    return (x, x_dot, p.H - z_b - p.h_c, 0.0, 0.0, theta_dot)


def theta_dot_for(A, z_b):
    L = p.H - z_b - p.h_c + p.h_c / 2
    return A * math.sqrt(p.g / L) / L


s = state(5.0, 2.0)                                   # D = 5 + 2 = 7
env.prev_cmd = (0.0, 0.0)

# R1: standing still, no swing, same command: time cost + small positive shaping
r, t = env._reward(s, s, (0.0, 0.0), None)
expected = -0.1 + 2 * (task.gamma * -7.0 + 7.0)       # = -0.1 + 0.07
check("R1 idle step", abs(r - expected) < 1e-9, f"r = {r:.4f} (expected {expected:.4f}), {t}")

# R2: moving 0.1 m closer: shaping clearly positive
r, t = env._reward(s, state(5.1, 2.0), (0.0, 0.0), None)
check("R2 progress gives positive shaping", t["shaping"] > 0.2, f"shaping = {t['shaping']:.4f}")

# R3: swing penalty: A = A_ref costs w_swing, and it is capped at 4x
_, t1 = env._reward(s, state(5.0, 2.0, theta_dot=theta_dot_for(0.30, 2.0)), (0.0, 0.0), None)
_, t2 = env._reward(s, state(5.0, 2.0, theta_dot=theta_dot_for(1.00, 2.0)), (0.0, 0.0), None)
check("R3 swing penalty at A = 0.3 m", abs(t1["swing"] + 0.05) < 1e-9, f"{t1['swing']:.4f}")
check("R3 swing penalty capped", abs(t2["swing"] + 0.20) < 1e-9, f"{t2['swing']:.4f}")

# R4: jerk: full reversal on both axes costs 2 * w_jerk
env.prev_cmd = (-0.5, -0.4)
_, t = env._reward(s, s, (0.5, 0.4), None)
check("R4 full reversal on both axes", abs(t["jerk"] + 0.04) < 1e-9, f"{t['jerk']:.4f}")
env.prev_cmd = (0.0, 0.0)

# terminal outcomes, from a state 1 cm above the target (D = 0.01)
s_end = state(10.0, 0.01)
land = {"err_x": 0.0}
r, t = env._reward(s_end, state(10.0, 0.0), (0.0, 0.0), "success", land)
check("R5 perfect success: +120", abs(t["terminal"] - 120.0) < 1e-9, f"terminal = {t['terminal']}")

r, t = env._reward(s_end, state(10.0, 0.0), (0.0, 0.0), "success", {"err_x": 0.025})
check("R6 success 2.5 cm off: +110", abs(t["terminal"] - 110.0) < 1e-9, f"terminal = {t['terminal']}")

r, t = env._reward(s_end, state(10.0, 0.0), (0.0, 0.0), "bad_landing", land)
check("R7 bad landing: -50", t["terminal"] == -50.0)

for crash in ("collision", "floor_collision", "end_stop", "hard_top_stop"):
    r, t = env._reward(s, s, (0.0, 0.0), crash)
    check(f"R8 {crash}: -100", t["terminal"] == -100.0)

# R9: at a terminal state Phi(s_next) = 0, so shaping = w_shaping * D(s)
r, t = env._reward(s, s, (0.0, 0.0), "collision")
check("R9 terminal shaping = 2 * D(s)", abs(t["shaping"] - 14.0) < 1e-9, f"{t['shaping']:.4f}")

# R10: truncation is not terminal: no terminal reward, Phi(s_next) kept
r, t = env._reward(s, s, (0.0, 0.0), None)
check("R10 truncation keeps Phi(s_next)", t["terminal"] == 0.0 and abs(t["shaping"] - 0.07) < 1e-9)

print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")