"""Checks for baseline.py: profiles, input shaping, stability, and closed-loop results."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # add src/ to the search path

import math                                                          # noqa: E402

import numpy as np                                                   # noqa: E402

from baseline import (Trapezoid, ZVShaped, BaselineController,       # noqa: E402
                      BaselineGains, run_episode)
from env import CraneEnv                                             # noqa: E402
from scenarios import ScenarioConfig, SAMPLERS, canonical_layouts, make_eval_set   # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


g = 9.81

# ---------------------------------------------------------------- B1-B3: trapezoid profile
tp = Trapezoid(1.0, 9.0, 0.3, 1.0)                     # 8 m: reaches cruise speed
xs = [tp(t) for t in np.arange(0, tp.T + 1, 0.01)]
check("B1 trapezoid ends at the target, at rest",
      abs(xs[-1][0] - 9.0) < 1e-9 and xs[-1][1] == 0.0, f"x={xs[-1][0]:.6f}, T={tp.T:.2f} s")
check("B2 trapezoid respects speed and acceleration",
      max(abs(v) for _, v, _ in xs) <= 1.0 + 1e-9 and max(abs(a) for _, _, a in xs) <= 0.3 + 1e-9)
tri = Trapezoid(5.0, 3.0, 0.3, 1.0)                    # 2 m backwards: triangular
check("B3 short move: triangular, backwards", tri.v < 1.0 and abs(tri(tri.T)[0] - 3.0) < 1e-9,
      f"peak speed {tri.v:.3f} m/s")

# ---------------------------------------------------------------- B4: ZV shaping cancels swing
# open-loop pendulum (fixed L) driven by the shaped vs unshaped profile acceleration
def residual_swing(profile, L, t_end):
    th = thd = 0.0
    dt = 0.001
    for t in np.arange(0, t_end, dt):
        a = profile(t)[2]
        thd += dt * (-(g * math.sin(th) + a * math.cos(th)) / L)
        th += dt * thd
    return math.hypot(L * math.sin(th), L * thd / math.sqrt(g / L))

L = 7.0
raw = Trapezoid(0.0, 10.0, 0.3, 1.0)
zv = ZVShaped(raw, math.sqrt(g / L))
A_raw = residual_swing(raw, L, zv.T + 5)
A_zv = residual_swing(zv, L, zv.T + 5)
check("B4 ZV shaping removes the residual swing", A_zv < 0.02 and A_raw > 0.1,
      f"unshaped {A_raw:.3f} m -> shaped {A_zv:.4f} m (cost: +{zv.delay:.2f} s)")

# ---------------------------------------------------------------- B5: closed-loop stability
gains = BaselineGains()
worst = -1e9
for L in np.linspace(2.5, 9.0, 27):
    c = 2 * gains.zeta_s * math.sqrt(g * L)
    A = np.array([[0, 1, 0, 0],
                  [-gains.k_p, -gains.k_v, 0, c],
                  [0, 0, 0, 1],
                  [gains.k_p / L, gains.k_v / L, -g / L, -c / L]])
    worst = max(worst, max(np.linalg.eigvals(A).real))
check("B5 trolley loop stable for every cable length", worst < -0.1,
      f"slowest pole real part {worst:.3f} (time constant {-1 / worst:.1f} s)")

# ---------------------------------------------------------------- B6: canonical layouts
env = CraneEnv()
ctl = BaselineController(env.p)
for name, L in canonical_layouts(env.p).items():
    ep = run_episode(env, ctl, layout=L)
    check(f"B6 canonical {name}", ep["outcome"] == "success",
          f"{ep['outcome']}, {ep['time']:.1f} s, err {100 * ep['landing']['err_x']:.1f} cm")

# ---------------------------------------------------------------- B7: random layouts, all types
cfg = ScenarioConfig()
n = 30
for kind in SAMPLERS:
    eps = [run_episode(env, ctl, layout=L)
           for L in make_eval_set(kind, n, seed=77, cfg=cfg, params=env.p)]
    rate = sum(e["outcome"] == "success" for e in eps) / n
    check(f"B7 {kind}: success rate >= 95%", rate >= 0.95, f"{rate:.0%} of {n}")

print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")