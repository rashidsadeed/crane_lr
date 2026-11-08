"""Train SARSA(lambda) through the whole curriculum, unattended and crash-safe.

    python -m rl_agent.train --stages C0,C1,C2,C3,C4 --seed 1          # new run
    python -m rl_agent.train --resume runs/seed1                       # continue after a crash

Per stage:
  * Reverse curriculum: episodes start with the container hanging near the target; every
    `adv_every` episodes a greedy check (`adv_n` episodes, current start distribution) runs, and
    when it reaches `advance_at_greedy` starts move further back, then the share of mid-air
    starts falls until every episode starts on the floor at A. (--adv-mode train: advance on
    training success >= `advance_at` instead, the original rule.)
  * Exploration starts at `eps0` (lower for the stages in STAGE_EPS0, or --eps0) and decays
    over `eps_decay_episodes` after the first curriculum advance.
  * Convergence: once all starts are at A, the greedy policy is evaluated every `eval_every`
    episodes on a fixed held-out set of the stage's layouts. The stage is converged after
    `patience` evaluations in a row at or above `target`. `max_episodes` is a safety cap.
  * The best checkpoint (greedy success, then delivery time) is kept and carried into the next
    stage.

Crash safety: every `checkpoint_minutes` the full training state (weights, curriculum, counters,
random-generator states) is written atomically to the run folder; --resume continues exactly.
"""
import argparse
import json
import os
import pathlib
import pickle
import time
from collections import Counter, deque
from dataclasses import dataclass, field, replace

import numpy as np

from env import CraneEnv, N_ACTIONS, TaskConfig
from scenarios import ScenarioConfig, sample_layout
from rl_agent.features import compute_features
from rl_agent.sarsa import SarsaLambda
from rl_agent.tiles import TileCoder

LOOSE = dict(tol_x=0.20, tol_A=0.20, tol_v=0.15, tol_land=0.40)
STAGES = {
    # name: (scenario config, task tolerances)
    # C0: open floor, random A and B. (A fixed A=0, B=10 made the 40 held-out evaluations one
    # identical deterministic episode, and the agent memorised that single route.)
    "C0": (ScenarioConfig(weights={"S0": 1.0}), LOOSE),
    "C1": (ScenarioConfig(weights={"S0": 0.5, "S1": 0.5}, max_high=1, g_min=1.0), LOOSE),
    "C2": (ScenarioConfig(weights={"S1": 0.4, "S2": 0.3, "S3": 0.3}, g_min=1.0), LOOSE),
    "C3": (ScenarioConfig(g_min=0.6), dict(tol_x=0.10, tol_A=0.10, tol_v=0.10, tol_land=0.30)),
    "C4": (ScenarioConfig(), {}),
    # probe, not part of the default curriculum: C0's open floor with the final (C4) tolerances.
    # Tests whether the agent can reach 5 cm swing / 5 cm placement at all, without stacks.
    "C0s": (ScenarioConfig(weights={"S0": 1.0}), {}),
}
# Lower starting exploration for stages that only tighten tolerances on layouts the agent
# already handles (a random 0.3 s action near touchdown ruins a 5 cm landing). Overridden by --eps0.
STAGE_EPS0 = {"C0s": 0.03, "C4": 0.03}


@dataclass
class Settings:
    alpha: float = 0.1            # step size at the start of every stage
    alpha_decay: float = 0.85     # convergence phase: alpha *= alpha_decay after each evaluation
    alpha_min: float = 0.005      # low floor so the weights settle once the agent is good (at 0.02
                                  # greedy success still swung 52-100% between evaluations)
    lam: float = 0.9
    eps0: float = 0.10            # exploration at the start of each stage (see STAGE_EPS0)
    eps0_override: float | None = None   # --eps0: use this in every stage instead
    eps_min: float = 0.02
    hold: int = 3                 # steps a random exploratory action is kept
    q_init: float = 50.0          # optimistic initial Q-value (exploration)
    max_episodes: int = 120_000   # per stage (safety cap; C3/C4 still improved at 60k)
    eps_decay_episodes: int = 24_000   # exploration decays to eps_min over this many episodes after
                                       # the first curriculum advance (independent of the cap)
    eval_every: int = 500
    eval_n: int = 100             # held-out layouts per evaluation (40 gave +-6 points of noise
                                  # and favoured lucky checkpoints when picking the best)
    target: float = 0.90          # greedy success needed for convergence
    patience: int = 3             # evaluations in a row at or above target
    checkpoint_minutes: float = 5.0
    snapshot_every: int = 2500    # keep a copy of the weights this often (to replay later)
    dist0: float = 0.5            # reverse curriculum: first start distance [m]
    grow: float = 1.6
    advance_at: float = 0.6       # adv_mode "train": training success (with exploration) needed
    window: int = 200
    adv_mode: str = "greedy"      # "greedy": advance on a greedy check under the current start
                                  # distribution; "train": on training success (exploration noise
                                  # hides progress at strict tolerances, so the curriculum can stall)
    adv_every: int = 200          # greedy check every this many episodes
    adv_n: int = 20               # episodes per greedy check
    advance_at_greedy: float = 0.8
    p_step: float = 0.25


@dataclass
class Curriculum:
    dist: float
    p_air: float = 1.0
    full: bool = False
    recent: deque = field(default_factory=lambda: deque(maxlen=200))

    def record(self, success: bool, s: Settings) -> str | None:
        """adv_mode "train": advance once the recent training success reaches advance_at."""
        self.recent.append(success)
        if len(self.recent) < self.recent.maxlen or np.mean(self.recent) < s.advance_at:
            return None
        return self.advance(s)

    def advance(self, s: Settings) -> str | None:
        """Move starts further back; once at full distance, reduce the share of mid-air starts."""
        self.recent.clear()
        if not self.full:
            self.dist *= s.grow
            self.full = self.dist >= 12.0      # longest possible route (rail length x_max)
            return f"starts within {min(self.dist, 12.0):.1f} m of the target"
        if self.p_air > 0:
            self.p_air = max(0.0, self.p_air - s.p_step)
            return f"mid-air starts {self.p_air:.0%}"
        return None

    @property
    def done(self) -> bool:
        return self.full and self.p_air == 0.0


def eval_layouts(stage: str, n: int, p) -> list:
    """Fixed held-out layouts for a stage (same every time, never used for training)."""
    scen, _ = STAGES[stage]
    rng = np.random.default_rng(987_654)
    return [sample_layout(rng, scen, p) for _ in range(n)]


def evaluate(env, agent, tc, layouts) -> dict:
    """Greedy episodes from the floor at A on the given layouts."""
    task = env.task
    env.task = replace(task, p_air_start=0.0)
    eps = []
    for L in layouts:
        obs, _ = env.reset(options={"layout": L})
        while True:
            obs, r, term, trunc, info = env.step(agent.greedy(tc.active(compute_features(obs, env.p))))
            if term or trunc:
                eps.append(info["episode"])
                break
    env.task = task
    ok = [e for e in eps if e["outcome"] == "success"]
    return {"success": len(ok) / len(eps),
            "time": float(np.mean([e["time"] for e in ok])) if ok else None,
            "outcomes": dict(Counter(e["outcome"] for e in eps))}


def greedy_check(env, agent, tc, n: int, seed0: int) -> float:
    """Greedy success under the env's current task (the curriculum's start distribution).
    Fixed seeds, so the result depends only on the weights; the agent's random generator is
    used only for tie-breaking, and its state is checkpointed, so resumes stay exact."""
    ok = 0
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        while True:
            obs, r, term, trunc, info = env.step(agent.greedy(tc.active(compute_features(obs, env.p))))
            if term or trunc:
                ok += info["episode"]["outcome"] == "success"
                break
    return ok / n


# ------------------------------------------------------------------ checkpoints
def save_state(run: pathlib.Path, st: dict, agent: SarsaLambda):
    """Atomic: write to temporary files, then rename (a crash mid-write leaves the old ones)."""
    agent.save(run / "weights.tmp.npz")
    os.replace(run / "weights.tmp.npz", run / "weights.npz")
    st = dict(st, agent_rng=agent.rng.bit_generator.state, agent_eps=agent.epsilon)
    with open(run / "state.tmp.pkl", "wb") as f:
        pickle.dump(st, f)
    os.replace(run / "state.tmp.pkl", run / "state.pkl")


def load_state(run: pathlib.Path):
    with open(run / "state.pkl", "rb") as f:
        st = pickle.load(f)
    agent = SarsaLambda.load(run / "weights.npz")
    agent.rng.bit_generator.state = st["agent_rng"]
    agent.epsilon = st["agent_eps"]
    return st, agent


def log(run: pathlib.Path, row: dict):
    with open(run / "log.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


# ------------------------------------------------------------------ training
def train(run: pathlib.Path, st: dict, agent: SarsaLambda):
    s: Settings = st["settings"]
    p = CraneEnv().p
    tc = TileCoder(p)
    tc.check_agent(agent, str(run))
    agent.explore_hold = s.hold
    last_ckpt = time.time()

    while st["stage_i"] < len(st["stages"]):
        stage = st["stages"][st["stage_i"]]
        scen, tol = STAGES[stage]
        base = TaskConfig(**tol)
        env = CraneEnv(scen_cfg=scen, task_cfg=base)
        held_out = eval_layouts(stage, s.eval_n, p)
        cur: Curriculum = st["curriculum"]
        print(f"=== stage {stage}  (episode {st['ep']} of at most {s.max_episodes}) ===", flush=True)
        if st["ep"] == 0:
            agent.alpha = s.alpha      # full step size again for a new stage (not on a resume)
        window, t0 = [], time.time()

        while True:
            ep = st["ep"]
            # exploration stays at eps0 until the first discovery (first curriculum advance),
            # then decays over the next eps_decay_episodes (old runs: 40% of the cap)
            eps0 = getattr(s, "eps0_override", None)
            if eps0 is None:
                eps0 = STAGE_EPS0.get(stage, s.eps0)
            if st.get("ep_adv") is None:
                agent.epsilon = eps0
            else:
                agent.epsilon = max(min(s.eps_min, eps0),
                                    eps0 * (1 - (ep - st["ep_adv"]) / getattr(
                                        s, "eps_decay_episodes", 0.4 * s.max_episodes)))
            env.task = replace(base, p_air_start=cur.p_air, air_max_dist=cur.dist)
            # deterministic episode seed (Python's hash() of strings changes between runs)
            obs, info = env.reset(seed=st["seed"] * 10_000_000 + st["stage_i"] * 1_000_000 + ep)
            a = agent.begin_episode(tc.active(compute_features(obs, p)))
            while True:
                obs, r, term, trunc, info = env.step(a)
                a = agent.step(r, tc.active(compute_features(obs, p)), terminal=term)
                if term or trunc:
                    break
            e = info["episode"]
            window.append(e)
            st["ep"] += 1
            st["total_ep"] += 1
            if s.adv_mode == "train":
                msg = cur.record(e["outcome"] == "success", s)
            else:
                msg = None
                if not cur.done and st["ep"] % s.adv_every == 0:
                    seed0 = 10**12 + st["seed"] * 10**10 + st["stage_i"] * 10**9 + st["ep"] * 100
                    g = greedy_check(env, agent, tc, s.adv_n, seed0)
                    if g >= s.advance_at_greedy:
                        msg = cur.advance(s)
                        if msg:
                            msg += f"  (greedy check {g:.0%})"
            if msg:
                if st.get("ep_adv") is None:
                    st["ep_adv"] = st["ep"]
                print(f"   {stage} ep {st['ep']}: curriculum -> {msg}", flush=True)

            # ---- periodic report, and convergence check once all starts are at A
            converged = False
            if st["ep"] % s.eval_every == 0:
                c = Counter(x["outcome"] for x in window)
                row = {"stage": stage, "ep": st["ep"], "total_ep": st["total_ep"],
                       "train_success": c["success"] / len(window),
                       "return": float(np.mean([x["return"] for x in window])),
                       "epsilon": agent.epsilon, "start_dist": min(cur.dist, 12.0),
                       "p_air": cur.p_air, "outcomes": dict(c),
                       "minutes": st["minutes"] + (time.time() - t0) / 60}
                text = ""
                if cur.done:
                    ev = evaluate(env, agent, tc, held_out)
                    row.update(greedy_success=ev["success"], greedy_time=ev["time"],
                               greedy_outcomes=ev["outcomes"])
                    key = (ev["success"], -(ev["time"] or 1e9))
                    if st["best"] is None or key >= st["best"]:
                        st["best"] = key
                        agent.save(run / f"best_{stage}.npz")
                    st["streak"] = st["streak"] + 1 if ev["success"] >= s.target else 0
                    converged = st["streak"] >= s.patience
                    # shrinking step size: lets the weights settle instead of chattering
                    # (constant-alpha SARSA with linear features keeps moving, Gordon 2000)
                    agent.alpha = max(s.alpha_min, agent.alpha * s.alpha_decay)
                    row["alpha"] = agent.alpha
                    text = (f"  alpha {agent.alpha:.3f}  greedy {ev['success']:5.1%}"
                            + (f" in {ev['time']:.1f}s" if ev["time"] else "")
                            + f"  streak {st['streak']}/{s.patience}")
                log(run, row)
                print(f"{stage} ep {st['ep']:6d}  train {row['train_success']:5.1%}  "
                      f"ret {row['return']:7.1f}  eps {agent.epsilon:.3f}  "
                      f"start<={row['start_dist']:4.1f} m air {cur.p_air:4.0%}{text}  "
                      f"{dict(c)}", flush=True)
                window = []

            # ---- snapshot for replaying how the behaviour developed (float32, compressed)
            if st["ep"] % s.snapshot_every == 0:
                snap = run / "snapshots"
                snap.mkdir(exist_ok=True)
                np.savez_compressed(snap / f"{stage}_ep{st['ep']:06d}.npz", w=agent.w.astype(np.float32),
                                    n_active=agent.n_active, gamma=agent.gamma, lam=agent.lam,
                                    alpha=agent.alpha, epsilon=agent.epsilon)

            # ---- crash-safe checkpoint
            if time.time() - last_ckpt > 60 * s.checkpoint_minutes:
                save_state(run, dict(st, minutes=st["minutes"] + (time.time() - t0) / 60), agent)
                last_ckpt = time.time()

            # ---- end of stage
            if converged or st["ep"] >= s.max_episodes:
                why = "converged" if converged else "reached the episode cap"
                print(f"=== {stage} {why} after {st['ep']} episodes; best greedy "
                      f"{st['best'][0] if st['best'] else 0:.0%} ===", flush=True)
                log(run, {"stage": stage, "end": why, "ep": st["ep"], "best": st["best"]})
                if (run / f"best_{stage}.npz").exists():   # carry the best weights forward
                    best = SarsaLambda.load(run / f"best_{stage}.npz")
                    agent.w = best.w
                st["minutes"] += (time.time() - t0) / 60
                st.update(stage_i=st["stage_i"] + 1, ep=0, best=None, streak=0, ep_adv=None,
                          curriculum=Curriculum(dist=s.dist0,
                                                recent=deque(maxlen=s.window)))
                save_state(run, st, agent)
                last_ckpt = time.time()
                break

    agent.save(run / "final.npz")
    print(f"done: {st['total_ep']} episodes in {st['minutes']:.0f} min. Weights: {run / 'final.npz'}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="C0,C1,C2,C3,C4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--resume", default=None, help="run folder to continue")
    ap.add_argument("--max-episodes", type=int, default=None, help="per-stage cap")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--init", default=None, help="start from these weights (e.g. an earlier stage)")
    ap.add_argument("--checkpoint-minutes", type=float, default=None)
    ap.add_argument("--q-init", type=float, default=None, help="optimistic initial Q-value")
    ap.add_argument("--eps0", type=float, default=None,
                    help="starting exploration in every stage (default: 0.10, or STAGE_EPS0)")
    ap.add_argument("--adv-mode", choices=("greedy", "train"), default=None,
                    help="curriculum advance on a greedy check (default) or on training success")
    args = ap.parse_args()

    if args.resume:
        run = pathlib.Path(args.resume)
        st, agent = load_state(run)
        print(f"resuming {run}: stage {st['stages'][st['stage_i']]}, episode {st['ep']}", flush=True)
    else:
        s = Settings()
        if args.max_episodes:
            s.max_episodes = args.max_episodes
        if args.eval_every:
            s.eval_every = args.eval_every
        if args.checkpoint_minutes:
            s.checkpoint_minutes = args.checkpoint_minutes
        if args.q_init is not None:
            s.q_init = args.q_init
        if args.eps0 is not None:
            s.eps0_override = args.eps0
        if args.adv_mode:
            s.adv_mode = args.adv_mode
        stages = args.stages.split(",")
        for name in stages:
            assert name in STAGES, f"unknown stage {name}"
        run = pathlib.Path(args.out) / f"seed{args.seed}"
        if (run / "log.jsonl").exists() or (run / "state.pkl").exists():
            raise SystemExit(f"{run} already holds a run. Use --resume {run} to continue it, "
                             f"or a different --out (or move the old folder away) to start fresh.")
        run.mkdir(parents=True, exist_ok=True)
        tc = TileCoder(CraneEnv().p)
        if args.init:
            agent = SarsaLambda.load(args.init, seed=args.seed)
        else:
            agent = SarsaLambda(tc.n_features, tc.n_active, n_actions=N_ACTIONS, alpha=s.alpha,
                                lam=s.lam, gamma=TaskConfig().gamma, epsilon=s.eps0,
                                explore_hold=s.hold, q_init=s.q_init, seed=args.seed)
        st = {"settings": s, "stages": stages, "stage_i": 0, "seed": args.seed,
              "ep": 0, "total_ep": 0, "minutes": 0.0, "best": None, "streak": 0,
              "curriculum": Curriculum(dist=s.dist0, recent=deque(maxlen=s.window))}
        (run / "config.json").write_text(json.dumps(
            {"stages": stages, "seed": args.seed, "settings": s.__dict__}, indent=1))
    train(run, st, agent)


if __name__ == "__main__":
    main()