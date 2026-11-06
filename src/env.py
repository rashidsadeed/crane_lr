"""Gymnasium environment: 2D overhead gantry crane, pick-lift-travel-lower-place."""
from dataclasses import dataclass
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import dynamics as dyn
import geometry as geo
from scenarios import MAX_STACKS, ScenarioConfig, sample_layout, layout_rects

# Actions are SPEED REFERENCES, like a crane operator's joysticks: the drive accelerates toward
# the reference as fast as its limits allow (the physics still uses accelerations).
VX_LEVELS = (-1.0, -0.4, -0.1, 0.0, 0.1, 0.4, 1.0)    # trolley speed references [m/s]
VL_LEVELS = (-0.5, 0.0, 0.15, 0.5)                     # hoist speed references [m/s], + = lowering
N_ACTIONS = len(VX_LEVELS) * len(VL_LEVELS)            # 28
CRASHES = ("collision", "floor_collision", "end_stop", "hard_top_stop", "timeout")


@dataclass(frozen=True)
class TaskConfig:
    tol_x: float = 0.05
    tol_A: float = 0.05
    tol_v: float = 0.05
    tol_land: float = 0.20
    far_from_target: float = 0.5
    v_top_safe: float = 0.20
    v_end_safe: float = 0.20       # trolley may touch an end buffer below this speed [m/s]
    liftoff_eps: float = 0.01
    margin: float = 0.10
    clearance: float = 0.30
    # Reward balance (gamma = 0.995):
    #   stalling forever costs time_cost / (1 - gamma) = 100 = |r_crash|, so hovering never
    #   beats trying; and r_success > time_cost * 400 steps + |r_crash|, so even a 40 s delivery
    #   beats crashing at once. With time_cost = 0.1 the agent learned to hover (stall = -20).
    time_cost: float = 0.5
    w_swing: float = 0.05
    A_ref: float = 0.30
    w_jerk: float = 0.02
    w_shaping: float = 2.0
    gamma: float = 0.995
    r_success: float = 200.0
    r_precision: float = 20.0
    r_crash: float = -100.0
    r_bad_landing: float = -50.0   # full penalty for a landing far outside the limits
    graded_bad_landing: bool = True  # scale r_bad_landing by how far the worst criterion is past
                                     # its limit: x min(1, worst_ratio - 1). A near miss costs ~0,
                                     # twice the limit or more costs the full -50. False: flat -50
                                     # (no learning signal between 'just missed' and 'way off').
    max_steps: int = 600
    timeout_is_failure: bool = False  # False: max_steps truncates (no penalty, the agent bootstraps);
                                      # True: a -100 failure. The observation has no clock, so True
                                      # makes the reward non-Markov.
    shaping_gamma: float = 1.0     # 1.0: F = Phi(s') - Phi(s), idling costs the full time cost
                                   # gamma: F = gamma Phi(s') - Phi(s), exactly policy-invariant
    p_air_start: float = 0.0       # probability an episode starts with the container in the air
    air_max_dist: float = 99.0     # air starts at most this far (horizontally) from the target [m]


class CraneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, params=None, scen_cfg=None, task_cfg=None,
                 render_mode=None, render_speed=1.0):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_speed = render_speed      # "human" mode: 2.0 = twice real time
        self._renderer = None
        self.hud_text = ""                    # optional extra line shown in the window
        self.p = params if params is not None else dyn.CraneParams()
        self.scen_cfg = scen_cfg if scen_cfg is not None else ScenarioConfig()
        self.task = task_cfg if task_cfg is not None else TaskConfig()
        p = self.p
        assert self.scen_cfg.max_stacks <= MAX_STACKS, (
            f"max_stacks = {self.scen_cfg.max_stacks} > {MAX_STACKS} observation slots")
        # touchdown detection relies on the floor being exactly at the lower hoist stop:
        # at l = l_max the hanging container's bottom is at z = 0
        assert abs(dyn.container_bottom(p.l_max, p)) < 1e-9, (
            f"H - l_max - h_c = {dyn.container_bottom(p.l_max, p)}, must be 0")
        self.action_space = spaces.Discrete(N_ACTIONS)
        low = np.zeros(7 + 2 * MAX_STACKS)
        high = np.zeros(7 + 2 * MAX_STACKS)
        low[0], high[0] = p.x_min, p.x_max
        low[1], high[1] = -p.vx_max, p.vx_max
        low[2], high[2] = p.l_min, p.l_max
        low[3], high[3] = -p.vl_max, p.vl_max
        low[4], high[4] = -math.pi / 2, math.pi / 2
        low[5], high[5] = -3.0, 3.0
        low[6], high[6] = p.x_min, p.x_max
        max_top = self.scen_cfg.max_high * p.h_c
        for i in range(MAX_STACKS):
            low[7 + 2 * i], high[7 + 2 * i] = p.x_min, p.x_max
            low[8 + 2 * i], high[8 + 2 * i] = 0.0, max_top
        self.observation_space = spaces.Box(low, high, dtype=np.float64)
        self.state = None
        self.layout = None

    @staticmethod
    def _decode_action(action):
        i_x, i_l = divmod(int(action), len(VL_LEVELS))
        return VX_LEVELS[i_x], VL_LEVELS[i_l]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        layout = options.get("layout")
        if layout is None:
            layout = sample_layout(self.np_random, self.scen_cfg, self.p)
        self.layout = layout
        self.rects = layout_rects(layout, self.p)
        self.state = (layout.x_A, 0.0, self.p.l_max, 0.0, 0.0, 0.0)
        self.t = 0
        self.on_start = True
        if self.np_random.random() < self.task.p_air_start:
            self._air_start()
        self.prev_acc = (0.0, 0.0)            # accelerations the drives applied last step
        self._action = None                   # action index of the current step (None: step_command)
        self._prev_action = None
        self.ep_return = 0.0
        self.ep_terms = {"time": 0.0, "swing": 0.0, "jerk": 0.0, "shaping": 0.0, "terminal": 0.0}
        self.peak_A = 0.0
        self.effort = 0.0
        self.switches = 0
        self._last_outcome = None
        self._frames = [self.state]           # states to draw for the latest step
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {"layout": layout}

    def _air_start(self):
        """Exploring start: container hanging still somewhere between A and B, clear of stacks."""
        p, rng, lay = self.p, self.np_random, self.layout
        d_max = min(self.task.air_max_dist, abs(lay.x_B - lay.x_A))
        x = lay.x_B - float(np.sign(lay.x_B - lay.x_A)) * float(rng.uniform(0.0, d_max))
        under = [st.n_high * p.h_c for st in lay.stacks if abs(st.center - x) < p.w_c + self.task.margin]
        z_lo = max(under, default=0.0) + self.task.clearance
        # reverse curriculum also on height: close starts are also low starts
        z_hi = min(p.H - p.l_min - p.h_c, max(z_lo, 0.3) + 1.0 + self.task.air_max_dist)
        z_b = float(rng.uniform(max(z_lo, 0.3), z_hi))
        self.state = (x, 0.0, p.H - z_b - p.h_c, 0.0, 0.0, 0.0)   # inverse of dyn.container_bottom
        self.on_start = False

    def _get_obs(self):
        obs = np.zeros(7 + 2 * MAX_STACKS, dtype=np.float64)
        obs[0:6] = self.state
        obs[6] = self.layout.x_B
        for i, st in enumerate(self.layout.stacks):
            obs[7 + 2 * i] = st.center
            obs[8 + 2 * i] = st.n_high * self.p.h_c
        return obs

    def _apply_interlock(self, a_x_cmd, a_l_cmd):
        if self.on_start:
            return 0.0, min(a_l_cmd, 0.0)
        return a_x_cmd, a_l_cmd

    def _update_on_start(self, corners):
        if self.on_start and geo.lowest_z(corners) > self.task.liftoff_eps:
            self.on_start = False

    def _detect_event(self, s_start, step_info):
        p, task = self.p, self.task
        s_prev = s_start
        for s in step_info["substeps"]:
            corners = geo.container_corners(s, p)
            self._update_on_start(corners)
            idx = geo.first_collision(corners, self.rects, task.margin, s[4])
            if idx is not None:
                return "collision", s, {"stack": idx}
            if not self.on_start and geo.lowest_z(corners) <= 1e-9:
                return "touchdown", s, {"v_land": s_prev[3]}
            s_prev = s
        s_end = step_info["substeps"][-1]
        if step_info["x_stop_speed"] > task.v_end_safe:
            return "end_stop", s_end, {"v_end": step_info["x_stop_speed"]}
        if step_info["l_top_speed"] > task.v_top_safe:
            return "hard_top_stop", s_end, {"v_top": step_info["l_top_speed"]}
        return None, s_end, {}

    def _judge_landing(self, s, v_land):
        task = self.task
        x_p, _ = dyn.payload_center(s, self.p)
        _, _, A = dyn.swing(s, self.p)
        landing = {"err_x": abs(x_p - self.layout.x_B), "A": A, "v_x": abs(s[1]), "v_land": v_land}
        limits = {"err_x": task.tol_x, "A": task.tol_A, "v_x": task.tol_v, "v_land": task.tol_land}
        landing["failed"] = [k for k, lim in limits.items() if landing[k] > lim]
        landing["worst_ratio"] = max(landing[k] / lim for k, lim in limits.items())
        if landing["err_x"] > task.far_from_target:
            return "floor_collision", landing
        if not landing["failed"]:
            return "success", landing
        return "bad_landing", landing

    def _required_height(self, x_p):
        stacks = [(st.center, st.n_high * self.p.h_c) for st in self.layout.stacks]
        ahead = geo.stacks_in_way(x_p, self.layout.x_B, stacks, self.p.w_c)
        return max(top for _, top in ahead) + self.task.clearance if ahead else 0.0

    def _potential(self, s):
        x_p, _ = dyn.payload_center(s, self.p)
        z_b = dyn.container_bottom(s[2], self.p)
        h_req = self._required_height(x_p)
        return -(abs(self.layout.x_B - x_p) + max(0.0, h_req - z_b) + max(z_b, h_req))

    def _reward(self, s, s_next, acc, outcome, landing=None):
        """acc = (a_x, a_l) the drives actually applied (after saturation), so the jerk term is
        bounded by w_jerk per axis and is the same for env.step and env.step_command."""
        task = self.task
        terminal = outcome is not None
        _, _, A = dyn.swing(s_next, self.p)
        swing = -task.w_swing * min(A * A / task.A_ref ** 2, 4.0)
        jerk = -task.w_jerk * (abs(acc[0] - self.prev_acc[0]) / (2 * self.p.ax_max)
                               + abs(acc[1] - self.prev_acc[1]) / (2 * self.p.al_max))
        phi_next = 0.0 if terminal else self._potential(s_next)
        shaping = task.w_shaping * (task.shaping_gamma * phi_next - self._potential(s))
        if outcome == "success":
            r_term = task.r_success + task.r_precision * (1.0 - landing["err_x"] / task.tol_x)
        elif outcome == "bad_landing":
            if task.graded_bad_landing and "worst_ratio" in landing:
                r_term = task.r_bad_landing * min(1.0, landing["worst_ratio"] - 1.0)
            else:
                r_term = task.r_bad_landing
        elif outcome in CRASHES:
            r_term = task.r_crash
        else:
            r_term = 0.0
        terms = {"time": -task.time_cost, "swing": swing, "jerk": jerk,
                 "shaping": shaping, "terminal": r_term}
        return sum(terms.values()), terms

    def step(self, action):
        """Action = index of a (trolley, hoist) speed-reference pair. The drive turns each
        reference into the acceleration that reaches it within one 0.1 s step (deadbeat),
        and dynamics.saturate clips that to the drive limits."""
        v_x_ref, v_l_ref = self._decode_action(action)
        dt = self.p.dt_ctrl
        self._action = int(action)
        try:
            return self.step_command((v_x_ref - self.state[1]) / dt, (v_l_ref - self.state[3]) / dt)
        finally:
            self._action = None

    def step_command(self, a_x_cmd, a_l_cmd):
        cmd = self._apply_interlock(a_x_cmd, a_l_cmd)
        s = self.state
        _, dyn_info = dyn.step(s, cmd[0], cmd[1], self.p, return_substeps=True)
        event, s_next, detail = self._detect_event(s, dyn_info)
        outcome, landing = event, None
        if event == "touchdown":
            outcome, landing = self._judge_landing(s_next, detail["v_land"])
        self.t += 1
        if outcome is None and self.t >= self.task.max_steps and self.task.timeout_is_failure:
            outcome = "timeout"
        terminated = outcome is not None
        truncated = not terminated and self.t >= self.task.max_steps
        acc = (dyn_info["a_x"], dyn_info["a_l"])      # what the drives applied (saturated)
        reward, terms = self._reward(s, s_next, acc, outcome, landing)
        self.state = s_next
        # switches: a change of discrete action (env.step), or of the applied acceleration by
        # more than 10% of the drive limit (env.step_command, continuous controllers)
        if self._action is not None:
            self.switches += self._prev_action is not None and self._action != self._prev_action
        else:
            self.switches += (abs(acc[0] - self.prev_acc[0]) > 0.1 * self.p.ax_max
                              or abs(acc[1] - self.prev_acc[1]) > 0.1 * self.p.al_max)
        self._prev_action = self._action
        self.prev_acc = acc
        self.peak_A = max(self.peak_A, dyn.swing(s_next, self.p)[2])
        self.effort += (abs(dyn_info["a_x"]) + abs(dyn_info["a_l"])) * self.p.dt_ctrl
        self.ep_return += reward
        for k, v in terms.items():
            self.ep_terms[k] += v
        # substep states up to the event, for smooth rendering
        subs = dyn_info["substeps"]
        self._frames = subs[:next(i for i, st in enumerate(subs) if st is s_next) + 1]
        self._last_outcome = outcome if terminated else ("timeout" if truncated else None)
        if self.render_mode == "human":
            self.render()
        info = {"event": outcome, "cmd": cmd, "reward_terms": terms}
        if terminated or truncated:
            info["episode"] = {
                "kind": self.layout.kind,
                "outcome": outcome if terminated else "timeout",
                "time": self.t * self.p.dt_ctrl,
                "return": self.ep_return,
                "reward_terms": dict(self.ep_terms),
                "landing": landing,
                "peak_swing": self.peak_A,
                "effort": self.effort,
                "switches": self.switches,
            }
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ rendering
    def render(self):
        """'human': play the latest step in the window. 'rgb_array': return the current frame."""
        if self.render_mode is None:
            gym.logger.warn("render() called without a render_mode; "
                            "create the env with CraneEnv(render_mode='human').")
            return None
        if self._renderer is None:
            from render import CraneRenderer          # pygame only needed when rendering
            self._renderer = CraneRenderer(self.p, self.render_mode, speed=self.render_speed)
        if self.render_mode == "human":
            t_start = (self.t - 1) * self.p.dt_ctrl if self.t > 0 else -self.p.dt_sim
            self._renderer.show(self, self._frames, t_start, self._last_outcome)
            return None
        return self._renderer.frame(self, self.state, self.t * self.p.dt_ctrl, self._last_outcome)

    @property
    def window_closed(self) -> bool:
        """True once the user has closed the 'human' window."""
        return self._renderer is not None and self._renderer.closed

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None