"""Watch the crane in a window, or record it to a video file.

Who drives:
    python -m tests.watch                                         baseline, random layouts
    python -m tests.watch --agent runs/seed1/best_C2.npz          trained agent (greedy)
    python -m tests.watch --agent runs/seed1/snapshots/C1_ep005000.npz   agent part-way through training
    python -m tests.watch --progress runs/seed1 --stage C2        every snapshot of a stage in turn:
                                                                  watch the behaviour improve
    python -m tests.watch --random                                random actions

Which layouts:
    --layout S1_mid_2high     one canonical layout (S0_open, S1_mid_1high, S1_mid_2high,
                              S1_nearA_2high, S2_behind)
    --stage C2                random layouts of a curriculum stage (default C4: all types)
    --seed 5                  first layout seed

Output:
    (default)                 pygame window in real time; --speed 2 for twice as fast
    --record demo.mp4         write a video instead (25 frames/s; needs: pip install imageio[ffmpeg])
    --episodes 5              number of episodes (default: until the window is closed, or 3 when recording)

Close the window to stop.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # add src/ to the search path

import argparse                                                      # noqa: E402
import re                                                            # noqa: E402

from baseline import BaselineController                              # noqa: E402
from env import CraneEnv, TaskConfig                                 # noqa: E402
from scenarios import canonical_layouts                              # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
who = ap.add_mutually_exclusive_group()
who.add_argument("--agent", help="weights file (.npz) of a trained agent")
who.add_argument("--progress", help="run folder: replay its snapshots in training order")
who.add_argument("--random", action="store_true", help="random actions")
ap.add_argument("--stage", default=None, help="curriculum stage for layouts and tolerances")
ap.add_argument("--layout", default=None, help="canonical layout name")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time")
ap.add_argument("--record", default=None, help="video file to write (.mp4 or .gif)")
ap.add_argument("--episodes", type=int, default=None)
args = ap.parse_args()

# ---------------------------------------------------------------- environment for the chosen stage
from rl_agent.train import STAGES                                    # noqa: E402

stage = args.stage or "C4"
scen, tol = STAGES[stage]
mode = "rgb_array" if args.record else "human"
env = CraneEnv(scen_cfg=scen, task_cfg=TaskConfig(**tol), render_mode=mode, render_speed=args.speed)
p = env.p
canon = canonical_layouts(p)
if args.layout and args.layout not in canon:
    sys.exit(f"unknown layout '{args.layout}'. Choose one of: {', '.join(canon)}")

# ---------------------------------------------------------------- drivers
policies = []                       # list of (label, policy factory)
if args.agent or args.progress:
    from rl_agent.features import compute_features                   # noqa: E402
    from rl_agent.sarsa import SarsaLambda                           # noqa: E402
    from rl_agent.tiles import TileCoder                             # noqa: E402
    tc = TileCoder(p)

    def agent_policy(path):
        agent = SarsaLambda.load(path)
        tc.check_agent(agent, str(path))
        return lambda obs: ("action", agent.greedy(tc.active(compute_features(obs, p))))

    if args.agent:
        policies = [(f"agent {pathlib.Path(args.agent).name}", agent_policy(args.agent))]
    else:
        snaps = sorted((pathlib.Path(args.progress) / "snapshots").glob(f"{stage}_ep*.npz"))
        if not snaps:
            sys.exit(f"no snapshots for stage {stage} in {args.progress}/snapshots")
        for s in snaps:
            n = int(re.search(r"ep(\d+)", s.name).group(1))
            policies.append((f"agent, {stage} after {n:,} episodes", agent_policy(s)))
elif args.random:
    policies = [("random actions", lambda obs: ("action", env.action_space.sample()))]
else:
    ctl = BaselineController(p)
    policies = [("baseline controller", lambda obs: ("command", ctl.act(obs)))]

n_episodes = args.episodes or (len(policies) if args.progress else (3 if args.record else 10**9))

# ---------------------------------------------------------------- run
frames = []


def grab_frames():
    """Recording: draw every 4th substep of the latest step (100 Hz physics -> 25 frames/s)."""
    if env._renderer is None:
        env.render()                                  # creates the renderer
    sub = env._frames
    t_end = env.t * p.dt_ctrl
    for i in range(len(sub) - 1, -1, -4)[::-1]:
        t = t_end - (len(sub) - 1 - i) * p.dt_sim
        last = i == len(sub) - 1
        frames.append(env._renderer.frame(env, sub[i], t, env._last_outcome if last else None))


for k in range(n_episodes):
    if env.window_closed:
        break
    label, policy = policies[k % len(policies)]
    layout = canon.get(args.layout) if args.layout else None
    obs, _ = env.reset(seed=args.seed + (0 if args.progress else k),
                       options={"layout": layout} if layout else None)
    if label == "baseline controller":
        ctl.reset(obs)
    while True:
        kind, a = policy(obs)
        env.hud_text = label + (f"   phase: {ctl.phase}" if label == "baseline controller" else "")
        obs, r, term, trunc, info = env.step(a) if kind == "action" else env.step_command(*a)
        if args.record:
            grab_frames()
        if term or trunc or env.window_closed:
            break
    if term or trunc:
        ep = info["episode"]
        print(f"{label}: {ep['kind']}  {ep['outcome']}  {ep['time']:.1f} s")
        if args.record:                                # hold the last frame for 1.5 s
            frames.extend([frames[-1]] * 37)
env.close()

if args.record:
    import imageio.v2 as imageio                                     # noqa: E402
    imageio.mimsave(args.record, frames, fps=25) if args.record.endswith(".gif") else \
        imageio.mimsave(args.record, frames, fps=25, codec="libx264", quality=7, macro_block_size=8)
    print(f"saved {args.record} ({len(frames) / 25:.0f} s of video)")