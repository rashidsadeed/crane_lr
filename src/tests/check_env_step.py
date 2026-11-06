"""Whole-environment checks: Gymnasium's env checker and complete episodes."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from collections import Counter                      # noqa: E402

import numpy as np                                   # noqa: E402
from gymnasium.utils.env_checker import check_env    # noqa: E402

from env import CraneEnv                             # noqa: E402
from scenarios import Layout, Stack                  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


HOLD, UP, DOWN = 7, 6, 8                    # trolley 0 with hoist hold / up / down
RIGHT_UP, RIGHT, LEFT = 12, 13, 1           # trolley +0.5 & up, +0.5 & hold, -0.5 & hold

env = CraneEnv()
p = env.p

# S1: Gymnasium's own API checks (spaces, reset/step signatures, seeding, dtypes)
try:
    check_env(env, skip_render_check=True)
    check("S1 gymnasium check_env", True)
except Exception as e:                                # noqa: BLE001
    check("S1 gymnasium check_env", False, repr(e))

# S2: doing nothing -> timeout after 600 steps, not a failure
env.reset(options={"layout": Layout("S0", 1.0, 8.0, ())})
for k in range(1000):
    obs, r, term, trunc, info = env.step(HOLD)
    if term or trunc:
        break
ep = info["episode"]
check("S2 idle -> truncated at 60 s", trunc and not term and k + 1 == 600 and ep["outcome"] == "timeout",
      f"steps={k + 1}, outcome={ep['outcome']}, return={ep['return']:.2f}")

# S3: interlock: commanding travel while resting at A moves nothing
env.reset(options={"layout": Layout("S0", 1.0, 8.0, ())})
obs, *_ = env.step(RIGHT)
check("S3 travel blocked on the floor", obs[0] == 1.0 and obs[1] == 0.0, f"x={obs[0]}, x_dot={obs[1]}")


def run(layout, actions, setup=None):
    """Run actions (a list, or a function of the observation) until the episode ends."""
    obs, _ = env.reset(options={"layout": layout})
    if setup:
        setup()
        obs = env._get_obs()
    for t in range(2000):
        a = actions(obs) if callable(actions) else actions[min(t, len(actions) - 1)]
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            return info["episode"], r
    raise RuntimeError("episode did not end")


# S4: lift 1 m, then drive right at low height into a 2-high stack -> collision
lay = Layout("S1", 1.0, 10.0, (Stack(5.0, 2),))
def lift_then_drive(obs):
    z_b = p.H - obs[2] - p.h_c
    return UP if z_b < 1.0 else RIGHT
ep, r = run(lay, lift_then_drive)
check("S4 drive into a stack -> collision", ep["outcome"] == "collision" and ep["reward_terms"]["terminal"] == -100,
      f"outcome={ep['outcome']}, last reward={r:.1f}")

# S5: lift a little, travel, set it down 4 m short of the target -> floor collision
def short_drop(obs):
    z_b = p.H - obs[2] - p.h_c
    if obs[0] < 3.0:
        return UP if z_b < 0.5 else RIGHT
    return DOWN if abs(obs[1]) < 0.05 else LEFT        # brake, then lower
ep, r = run(Layout("S0", 1.0, 8.0, ()), short_drop)
check("S5 set down far from the target -> floor collision", ep["outcome"] == "floor_collision",
      f"outcome={ep['outcome']}")

# S6: hoist straight up at full speed into the upper stop -> hard top stop
ep, r = run(Layout("S0", 1.0, 8.0, ()), [UP])
check("S6 full-speed hoist into the top -> hard_top_stop", ep["outcome"] == "hard_top_stop",
      f"outcome={ep['outcome']}, time={ep['time']:.1f} s")


# S7/S8: placed directly over the target, lowering -> success or bad landing
def hover(l_dot):
    def setup():
        env.state = (8.0, 0.0, p.l_max - 0.05, l_dot, 0.0, 0.0)   # 5 cm above the floor
        env.on_start = False
    return setup
ep, r = run(Layout("S0", 1.0, 8.0, ()), [HOLD], setup=hover(0.1))
check("S7 gentle lowering at the target -> success", ep["outcome"] == "success",
      f"outcome={ep['outcome']}, landing={ep['landing']}, last reward={r:.1f}")
ep, r = run(Layout("S0", 1.0, 8.0, ()), [HOLD], setup=hover(0.3))
check("S8 lowering at 0.3 m/s -> bad landing", ep["outcome"] == "bad_landing"
      and ep["landing"]["failed"] == ["v_land"], f"failed={ep['landing']['failed']}")

# S9: random policy over random layouts: no crashes of the code, every episode ends cleanly
rng = np.random.default_rng(0)
outcomes = Counter()
ok_space = True
for i in range(200):
    obs, _ = env.reset(seed=i)
    while True:
        obs, r, term, trunc, info = env.step(int(rng.integers(15)))
        ok_space &= env.observation_space.contains(obs) or bool(term)
        if term or trunc:
            outcomes[info["episode"]["outcome"]] += 1
            break
check("S9 200 random episodes run cleanly", sum(outcomes.values()) == 200 and ok_space,
      f"outcomes {dict(outcomes)}")

# S10: determinism: same seed and same actions -> identical episode
def rollout(seed):
    obs, _ = env.reset(seed=seed)
    a_rng = np.random.default_rng(seed)
    traj = [obs]
    for _ in range(100):
        obs, r, term, trunc, _ = env.step(int(a_rng.integers(15)))
        traj.append(obs)
        if term or trunc:
            break
    return np.array(traj)
check("S10 same seed -> identical trajectory", np.array_equal(rollout(5), rollout(5)))

print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")