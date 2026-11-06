"""Linear SARSA(lambda) with binary (tile-coded) features and replacing traces.

Follows Sutton & Barto (2018), "Sarsa(lambda) with binary features and linear function
approximation", section 12.7:

    Q(s, a) = sum_{i in F(s)} w[a, i]                     F(s): the 73 active tiles of s

    each step, from (S, A) with reward R and next state S':
        z[A, i] = 1               for i in F(S)           replacing traces
        delta   = R - Q(S, A)
        if S' is terminal:        w += alpha * delta * z ;  episode ends
        A'      = epsilon-greedy(S')
        delta  += gamma * Q(S', A')
        w      += alpha * delta * z
        z      *= gamma * lambda

alpha is per active tile: alpha = alpha_0 / n_active, so one update moves Q(S, A)
by about alpha_0 * delta.

Traces are kept sparse: only the (action, tile) pairs touched recently, as two arrays.
Traces below `trace_min` are dropped (at gamma*lambda = 0.896 that is after ~65 steps),
so an update costs a few thousand operations instead of one per weight (391,215).
"""
import numpy as np


class SarsaLambda:
    def __init__(self, n_features: int, n_active: int, n_actions: int = 15,
                 alpha: float = 0.1, lam: float = 0.9, gamma: float = 0.995,
                 epsilon: float = 0.2, trace_min: float = 1e-3, explore_hold: int = 1,
                 q_init: float = 0.0, seed: int | None = None):
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_active = n_active
        self.alpha = alpha            # alpha_0: step size per update (divided by n_active below)
        self.lam = lam
        self.gamma = gamma
        self.epsilon = epsilon
        self.trace_min = trace_min
        self.explore_hold = explore_hold      # a random action is kept for this many steps
        self._hold, self._held = 0, 0
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros((n_actions, n_features))
        # optimistic initial values (Sutton & Barto 2018, sec. 2.6): every Q starts at q_init,
        # carried by the bias tile (the last feature, active in every state)
        self.w[:, -1] = q_init
        self._reset_traces()

    # ------------------------------------------------------------------ values and actions
    def q_values(self, tiles: np.ndarray) -> np.ndarray:
        """Q(s, a) for all actions: sum of each action's weights over the active tiles."""
        return self.w[:, tiles].sum(axis=1)

    def greedy(self, tiles: np.ndarray) -> int:
        """Best action; ties broken at random (all weights start at 0, so ties are common)."""
        q = self.q_values(tiles)
        best = np.flatnonzero(q == q.max())
        return int(best[0] if len(best) == 1 else self.rng.choice(best))

    def choose(self, tiles: np.ndarray, epsilon: float | None = None) -> int:
        """Epsilon-greedy with sticky exploration: with probability epsilon a random action
        is started and then kept for explore_hold steps (0.1 s steps are too short for a
        single random action to go anywhere); otherwise the greedy action."""
        eps = self.epsilon if epsilon is None else epsilon
        if self._hold > 0:
            self._hold -= 1
            return self._held
        if self.rng.random() < eps:
            self._held = int(self.rng.integers(self.n_actions))
            self._hold = self.explore_hold - 1
            return self._held
        return self.greedy(tiles)

    # ------------------------------------------------------------------ learning
    def _reset_traces(self):
        self.tr_idx = np.zeros(0, dtype=np.int64)     # flat indices a * n_features + i
        self.tr_val = np.zeros(0)

    def begin_episode(self, tiles: np.ndarray) -> int:
        """Clear the traces and pick the first action."""
        self._reset_traces()
        self._hold = 0
        self.tiles, self.action = tiles, self.choose(tiles)
        return self.action

    def step(self, reward: float, next_tiles: np.ndarray, terminal: bool) -> int | None:
        """One SARSA(lambda) update for the transition just taken.

        terminal=True  (success, crash, landing): no bootstrap, returns None.
        terminal=False (running, or truncated by the time limit): bootstraps from
                       Q(S', A') and returns A'. After a truncation the caller just resets.
        """
        flat = self.action * self.n_features + self.tiles
        # replacing traces: z = 1 on the active (S, A) pairs
        keep = ~np.isin(self.tr_idx, flat, assume_unique=True)
        self.tr_idx = np.concatenate((self.tr_idx[keep], flat))
        self.tr_val = np.concatenate((self.tr_val[keep], np.ones(len(flat))))

        delta = reward - self.w.ravel()[flat].sum()
        step_size = self.alpha / self.n_active

        if terminal:
            self.w.ravel()[self.tr_idx] += step_size * delta * self.tr_val
            return None

        next_action = self.choose(next_tiles)
        delta += self.gamma * self.w[next_action, next_tiles].sum()
        self.w.ravel()[self.tr_idx] += step_size * delta * self.tr_val

        # decay, and drop traces that no longer matter
        self.tr_val *= self.gamma * self.lam
        live = self.tr_val >= self.trace_min
        self.tr_idx, self.tr_val = self.tr_idx[live], self.tr_val[live]

        self.tiles, self.action = next_tiles, next_action
        self.last_delta = delta
        return next_action

    # ------------------------------------------------------------------ saving
    def save(self, path: str, **extra):
        np.savez_compressed(path, w=self.w, alpha=self.alpha, lam=self.lam, gamma=self.gamma,
                            epsilon=self.epsilon, n_active=self.n_active, **extra)

    @classmethod
    def load(cls, path: str, seed: int | None = None) -> "SarsaLambda":
        d = np.load(path)
        agent = cls(d["w"].shape[1], int(d["n_active"]), d["w"].shape[0], float(d["alpha"]),
                    float(d["lam"]), float(d["gamma"]), float(d["epsilon"]), seed=seed)
        agent.w = d["w"].copy()
        return agent