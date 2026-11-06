"""Validate trained agents against the baseline on layouts never used in training or evaluation.

Usage, from src/:
    python -m tests.validate_agent runs/seed1/best_C0.npz runs/seed2/best_C0.npz --stage C0
    python -m tests.validate_agent runs/seed*/best_C2.npz --stage C2 --n 200 --json c2_report.json
    python -m tests.validate_agent runs/seed1/best_C4.npz --stage C4 --canonical

Tests (the baseline runs on exactly the same episodes, except under action noise):
    T1 held-out layouts of the stage, the stage's tolerances      does it solve the stage?
    T2 same layouts, strict C4 tolerances                          how far from the final target?
    T3 same layouts, 2% random actions (each held 3 steps)          robust, or a memorised route?
    T4 same layouts, 5% random actions
    T5 same layouts, every episode starts mid-air between A and B   recovers from unseen states?
    T6 (--canonical) the fixed canonical layouts, strict tolerances

Per driver and test: success rate, delivery time, landing quality as median / 95th percentile
over all set-downs at the target (success or bad landing) next to its limit, effort, jerk, and how failures ended.
The held-out layouts use seed 123456; training evaluation uses 987654, so they never overlap.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # add src/ to the search path

import argparse                                                      # noqa: E402
import json                                                          # noqa: E402
from collections import Counter                                      # noqa: E402
from dataclasses import replace                                      # noqa: E402

import numpy as np                                                   # noqa: E402

from baseline import BaselineController                              # noqa: E402
from env import CraneEnv, TaskConfig, N_ACTIONS                      # noqa: E402
from rl_agent.features import compute_features                       # noqa: E402
from rl_agent.sarsa import SarsaLambda                               # noqa: E402
from rl_agent.tiles import TileCoder                                 # noqa: E402
from rl_agent.train import STAGES                                    # noqa: E402
from scenarios import canonical_layouts, sample_layout              # noqa: E402

HELD_OUT_SEED = 123_456
LANDING = ("err_x", "A", "v_x", "v_land")
LIMIT_NAME = {"err_x": "tol_x", "A": "tol_A", "v_x": "tol_v", "v_land": "tol_land"}


# ------------------------------------------------------------------ drivers
class AgentDriver:
    """Greedy agent, optionally with epsilon random actions held for `hold` steps."""

    def __init__(self, path, tc, p, eps=0.0, hold=3, seed=0):
        self.ag = SarsaLambda.load(path, seed=seed)
        self.tc, self.p, self.eps, self.hold = tc, p, eps, hold
        self.rng = np.random.default_rng(seed)

    def reset(self, obs):
        self.left = 0

    def act(self, obs):
        if self.left > 0:
            self.left -= 1
            return "action", self.a
        if self.eps and self.rng.random() < self.eps:
            self.a, self.left = int(self.rng.integers(N_ACTIONS)), self.hold - 1
            return "action", self.a
        return "action", self.ag.greedy(self.tc.active(compute_features(obs, self.p)))


class BaselineDriver:
    def __init__(self, p):
        self.c = BaselineController(p)

    def reset(self, obs):
        self.c.reset(obs)

    def act(self, obs):
        return "command", self.c.act(obs)


def run_episode(env, driver, layout, seed):
    obs, _ = env.reset(seed=seed, options={"layout": layout})
    driver.reset(obs)
    while True:
        kind, a = driver.act(obs)
        obs, r, term, trunc, info = env.step(a) if kind == "action" else env.step_command(*a)
        if term or trunc:
            return info["episode"]


# ------------------------------------------------------------------ summary
def summarise(eps, task):
    ok = [e for e in eps if e["outcome"] == "success"]
    # landing quality over set-downs at the target (success or bad landing); a floor collision
    # (set down more than 0.5 m away) is counted under failures instead
    landed = [e["landing"] for e in eps if e["outcome"] in ("success", "bad_landing")]
    out = {"n": len(eps), "success": len(ok) / len(eps),
           "time": float(np.mean([e["time"] for e in ok])) if ok else None,
           "effort": float(np.mean([e["effort"] for e in eps])),
           "jerk": float(np.mean([e["reward_terms"]["jerk"] for e in eps])),
           "peak_swing": float(np.mean([e["peak_swing"] for e in eps])),
           "failures": dict(Counter(e["outcome"] for e in eps if e["outcome"] != "success")),
           "landing": {}}
    for k in LANDING:
        v = np.array([l[k] for l in landed])
        out["landing"][k] = {"limit": getattr(task, LIMIT_NAME[k]),
                             "median": float(np.median(v)) if len(v) else None,
                             "p95": float(np.percentile(v, 95)) if len(v) else None,
                             "over_limit": float(np.mean(v > getattr(task, LIMIT_NAME[k]))) if len(v) else None}
    return out


def fmt(s):
    def lm(k, scale, unit, prec=1):
        d = s["landing"][k]
        if d["median"] is None:
            return f"{k} {'-':>11s}"
        return f"{k} {scale * d['median']:4.{prec}f}/{scale * d['p95']:4.{prec}f}{unit}"
    t = f"{s['time']:5.1f}s" if s["time"] is not None else "    - "
    return (f"{s['success']:5.0%} {t}  {lm('err_x', 100, 'cm')}  {lm('A', 100, 'cm')}  "
            f"{lm('v_land', 1, '', 2)}  effort {s['effort']:5.1f}  jerk {s['jerk']:6.2f}  {s['failures']}")


def limits_line(task):
    return (f"limits: err_x {100 * task.tol_x:.0f} cm, swing A {100 * task.tol_A:.0f} cm, "
            f"v_x {task.tol_v:.2f} m/s, v_land {task.tol_land:.2f} m/s   (landing columns: median/p95)")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", nargs="+", help="agent weight files (.npz)")
    ap.add_argument("--stage", default="C0", help="curriculum stage for layouts and tolerances")
    ap.add_argument("--n", type=int, default=100, help="held-out layouts per test")
    ap.add_argument("--canonical", action="store_true", help="also run the canonical layouts (T6)")
    ap.add_argument("--json", default=None, help="write all results to this file")
    args = ap.parse_args()

    scen, tol = STAGES[args.stage]
    stage_task, strict_task = TaskConfig(**tol), TaskConfig()
    p = CraneEnv().p
    tc = TileCoder(p)
    rng = np.random.default_rng(HELD_OUT_SEED)
    layouts = [sample_layout(rng, scen, p) for _ in range(args.n)]
    n_distinct = len(set(layouts))
    # label = the path without .npz (e.g. runs/seed1/best_C0), so agents from different run
    # folders never share a label (they would overwrite each other in the report)
    labels = [str(pathlib.Path(w).with_suffix("")) for w in args.weights]
    if len(set(labels)) != len(labels):
        raise SystemExit("the same weights file was given twice")
    width = max(8, *map(len, labels))

    tests = [
        ("T1 held-out layouts, stage tolerances", stage_task, 0.0, layouts),
        ("T2 held-out layouts, STRICT (C4) tolerances", strict_task, 0.0, layouts),
        ("T3 held-out layouts + 2% random actions", stage_task, 0.02, layouts),
        ("T4 held-out layouts + 5% random actions", stage_task, 0.05, layouts),
        ("T5 held-out layouts, mid-air starts", replace(stage_task, p_air_start=1.0), 0.0, layouts),
    ]
    if args.canonical:
        tests.append(("T6 canonical layouts, STRICT tolerances", strict_task, 0.0,
                      list(canonical_layouts(p).values())))

    print(f"stage {args.stage}: {args.n} held-out layouts ({n_distinct} distinct), seed {HELD_OUT_SEED}")
    report = {"stage": args.stage, "n": args.n, "distinct_layouts": n_distinct, "tests": {}}
    for name, task, eps, lays in tests:
        print(f"\n== {name}\n   {limits_line(task)}")
        env = CraneEnv(scen_cfg=scen, task_cfg=task)
        drivers = [] if eps > 0 else [("baseline", BaselineDriver(p))]
        drivers += [(lab, AgentDriver(w, tc, p, eps=eps)) for lab, w in zip(labels, args.weights)]
        report["tests"][name] = {}
        for label, drv in drivers:
            eps_list = [run_episode(env, drv, L, seed=10_000 + i) for i, L in enumerate(lays)]
            s = summarise(eps_list, task)
            report["tests"][name][label] = s
            print(f"   {label:{width}s} {fmt(s)}")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=1))
        print(f"\nsaved {args.json}")


if __name__ == "__main__":
    main()
