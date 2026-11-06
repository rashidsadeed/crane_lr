"""Evaluate the baseline on N held-out layouts per type:  python -m tests.eval_baseline 200"""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # add src/ to the search path
from collections import Counter  # noqa: E402
import numpy as np
from env import CraneEnv
from scenarios import ScenarioConfig, SAMPLERS, make_eval_set
from baseline import BaselineController, run_episode
env = CraneEnv(); ctl = BaselineController(); cfg = ScenarioConfig()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
t0 = time.time()
print(f"{'type':4s} {'success':>8s} {'time [s]':>9s} {'err_x [cm]':>11s} {'swing [cm]':>11s} {'peak [m]':>9s}  failures")
for kind in SAMPLERS:
    eps = [run_episode(env, ctl, layout=L) for L in make_eval_set(kind, N, seed=1000 + list(SAMPLERS).index(kind), cfg=cfg, params=env.p)]
    ok = [e for e in eps if e["outcome"] == "success"]
    fails = Counter(e["outcome"] for e in eps if e["outcome"] != "success")
    print(f"{kind:4s} {len(ok)/N:8.1%} {np.mean([e['time'] for e in ok]):9.1f} "
          f"{100*np.mean([e['landing']['err_x'] for e in ok]):11.2f} {100*np.mean([e['landing']['A'] for e in ok]):11.2f} "
          f"{np.mean([e['peak_swing'] for e in ok]):9.3f}  {dict(fails)}")
print(f"({time.time()-t0:.0f} s)")