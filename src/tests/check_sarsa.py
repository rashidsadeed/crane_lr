"""Checks for rl_agent/sarsa.py: action choice, the update by hand, traces, and learning a toy task."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # add src/ to the search path

import math                                                          # noqa: E402
from collections import Counter                                      # noqa: E402

import numpy as np                                                   # noqa: E402

from rl_agent.sarsa import SarsaLambda                               # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


tiles_a = np.array([0, 5, 9])          # a tiny "state": 3 active tiles out of 20
tiles_b = np.array([1, 6, 12])          # no tile in common with tiles_a

# ---------------------------------------------------------------- action choice
ag = SarsaLambda(n_features=20, n_active=3, n_actions=4, seed=0)
picks = Counter(ag.greedy(tiles_a) for _ in range(4000))
check("A1 all-zero weights: ties broken at random", len(picks) == 4 and min(picks.values()) > 850,
      f"{dict(sorted(picks.items()))}")
ag.w[2, 5] = 1.0
check("A2 greedy picks the highest Q", all(ag.greedy(tiles_a) == 2 for _ in range(50)))
picks = Counter(ag.choose(tiles_a, epsilon=0.2) for _ in range(20000))
share = picks[2] / 20000
check("A3 epsilon = 0.2: best action 85% of the time", abs(share - 0.85) < 0.01,
      f"{share:.3f} (= 0.8 + 0.2/4)")

# ---------------------------------------------------------------- one update, by hand
# from S = tiles_a with A, reward -1, then S' = tiles_b with A' = 2 (made the greedy choice)
ag = SarsaLambda(n_features=20, n_active=3, n_actions=4, alpha=0.3, lam=0.5, gamma=0.9,
                 epsilon=0.0, seed=0)
ag.w[2, 1] = 2.0                        # Q(S', 2) = 2, so A' = 2
ag.begin_episode(tiles_a)
A = ag.action
a_next = ag.step(-1.0, tiles_b, terminal=False)
# delta = R + gamma Q(S',A') - Q(S,A) = -1 + 0.9 * 2 - 0 = 0.8 ; step per tile = 0.3 / 3 = 0.1
check("U1 next action is the greedy one", a_next == 2)
check("U2 each active tile of (S, A) moves by alpha/n * delta = 0.08",
      np.allclose(ag.w[A, tiles_a], 0.08),
      f"w[A, F(S)] = {ag.w[A, tiles_a]}")
check("U3 traces decayed by gamma * lambda = 0.45", np.allclose(ag.tr_val, 0.45), f"{ag.tr_val}")

# second step: terminal with reward +1 from S' with A' = 2
ag.step(1.0, tiles_b, terminal=True)
# delta = 1 - Q(S', 2) = 1 - 2 = -1 ; trace 1 on (S', 2), 0.45 on (S, A)
check("U4 terminal: current pair moves by -0.1, earlier pair by -0.045 (its trace)",
      np.allclose(ag.w[2, tiles_b], [2.0 - 0.1, -0.1, -0.1]) and np.allclose(ag.w[A, tiles_a], 0.08 - 0.045),
      f"w[2, F(S')] = {np.round(ag.w[2, tiles_b], 4)}, w[A, F(S)] = {np.round(ag.w[A, tiles_a], 4)}")

# ---------------------------------------------------------------- replacing traces
ag = SarsaLambda(n_features=20, n_active=3, n_actions=1, lam=0.9, gamma=1.0, epsilon=0.0, seed=0)
ag.begin_episode(tiles_a)
for _ in range(5):
    ag.step(0.0, tiles_a, terminal=False)      # visit the same state again and again
check("T1 revisiting a state resets its traces to 1, not above (replacing)",
      len(ag.tr_idx) == 3 and np.allclose(ag.tr_val, 0.9), f"{ag.tr_val}")
for _ in range(80):
    ag.step(0.0, tiles_b, terminal=False)
check("T2 old traces are dropped once below trace_min",
      set(ag.tr_idx.tolist()) == set(tiles_b.tolist()), f"{len(ag.tr_idx)} traces kept")

# ---------------------------------------------------------------- learning a toy task
# corridor of 10 cells; actions left/right; -1 per step; episode ends at the right end.
# features: one tile per cell + a shared bias tile (like the crane's bias).
N = 10
def tiles_of(cell):
    return np.array([cell, N])

ag = SarsaLambda(n_features=N + 1, n_active=2, n_actions=2, alpha=0.5, lam=0.9,
                 gamma=0.99, epsilon=0.1, seed=1)
lengths = []
for ep in range(300):
    cell, a = 0, ag.begin_episode(tiles_of(0))
    for t in range(500):
        cell = max(0, cell - 1) if a == 0 else cell + 1
        done = cell == N - 1
        a = ag.step(-1.0, tiles_of(cell), terminal=done)
        if done:
            break
    lengths.append(t + 1)
check("L1 learns the corridor: episodes shrink to the shortest path",
      np.mean(lengths[:10]) > np.mean(lengths[-50:]) + 2 and np.mean(lengths[-50:]) < 11,
      f"mean length first 10: {np.mean(lengths[:10]):.0f}, last 50: {np.mean(lengths[-50:]):.1f} (best 9)")
greedy = [ag.greedy(tiles_of(c)) for c in range(N - 1)]
check("L2 greedy policy goes right everywhere", greedy == [1] * (N - 1), f"{greedy}")
v0 = ag.q_values(tiles_of(0)).max()
v_true = -sum(0.99 ** k for k in range(9))
check("L3 value of the start close to the true -8.65", abs(v0 - v_true) < 1.0, f"{v0:.2f}")

# ---------------------------------------------------------------- save / load
path = pathlib.Path(__file__).resolve().parent / "_sarsa_test.npz"
ag.save(path)
ag2 = SarsaLambda.load(path)
path.unlink()
check("S1 save and load keep the weights", np.array_equal(ag.w, ag2.w) and ag2.lam == 0.9)

print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")