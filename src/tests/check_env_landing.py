"""Checks for CraneEnv._judge_landing."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math                                          # noqa: E402

import dynamics as dyn                               # noqa: E402
from env import CraneEnv                             # noqa: E402
from scenarios import Layout                         # noqa: E402

env = CraneEnv()
env.reset(options={"layout": Layout("S0", 1.0, 8.0, ())})   # target at x_B = 8
p = env.p
L = p.l_max + p.h_c / 2                               # pendulum length on the floor: 9 m
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def landed(x=8.0, x_dot=0.0, theta_dot=0.0):
    """State on the floor at the target; swing set through theta_dot."""
    return (x, x_dot, p.l_max, 0.0, 0.0, theta_dot)


def theta_dot_for(A):
    """theta_dot giving swing amplitude A with theta = 0:  A = L*theta_dot / omega."""
    return A * math.sqrt(p.g / L) / L


cases = [
    # name,                          state,                               v_land, outcome,          failed
    ("perfect landing",              landed(),                            0.10, "success",         []),
    ("placement 4 cm off",           landed(x=8.04),                      0.10, "success",         []),
    ("placement 6 cm off",           landed(x=8.06),                      0.10, "bad_landing",     ["err_x"]),
    ("swing 4 cm",                   landed(theta_dot=theta_dot_for(0.04)), 0.10, "success",       []),
    ("swing 6 cm",                   landed(theta_dot=theta_dot_for(0.06)), 0.10, "bad_landing",   ["A"]),
    ("trolley still moving 0.06",    landed(x_dot=0.06),                  0.10, "bad_landing",     ["v_x"]),
    ("landing too fast 0.25",        landed(),                            0.25, "bad_landing",     ["v_land"]),
    ("several faults at once",       landed(x=8.10, x_dot=0.1),           0.30, "bad_landing",     ["err_x", "v_x", "v_land"]),
    ("set down 0.7 m from target",   landed(x=8.70),                      0.10, "floor_collision", ["err_x"]),
]

for name, s, v_land, outcome, failed in cases:
    got, info = env._judge_landing(s, v_land)
    ok = got == outcome and info["failed"] == failed
    check(f"J {name}", ok, f"-> {got}, failed={info['failed']}, "
          f"err_x={info['err_x']:.3f}, A={info['A']:.3f}")

print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")