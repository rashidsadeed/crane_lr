r"""Classical baseline: phase logic + trapezoidal travel profile + anti-sway feedback.

The controller reads only the raw observation (the same 13 values the agent gets)
and returns continuous acceleration commands for env.step_command(a_x, a_l).

Trolley (every step):
    a_x = a_ref + k_p (x_ref - x) + k_v (v_ref - x_dot) + c(L) * theta_dot
          \___/   \_______________________________/   \__________________/
       profile     track the trapezoidal profile       damp the swing
    c(L) = 2 * zeta_s * sqrt(g L): with this term the linearised swing equation
    theta_dd = -(g/L) theta - (c/L) theta_dot  is a damped pendulum, ratio zeta_s.

Hoist (every step):
    a_l = k_h (v_l_ref - l_dot)           l_dot > 0 = lowering
    v_l_ref drives the container bottom z_b to a commanded height z_cmd with a
    braking profile  |v| = min(vl_max, sqrt(2 a_h |z_b - z_cmd|)).

Height logic:
    z_cmd = h_req                 while a stack still lies between container and target
          = h_hover               once the path is clear (wait above the floor)
    set-down                      once trolley and swing have settled over the target:
                                  v_l_ref = sqrt(v_touch^2 + 2 a_h z_b)  (lands at ~v_touch)

Travel starts once the remaining lift will finish before the container's leading
edge reaches the first stack in its way (lift and travel overlap when there is room).
"""
from dataclasses import dataclass
import math

from dynamics import CraneParams, container_bottom, pendulum_length
from geometry import stacks_in_way
from scenarios import MAX_STACKS


@dataclass(frozen=True)
class BaselineGains:
    # travel profile
    a_ref: float = 0.30        # profile acceleration [m/s^2] (below 0.5: leaves room for feedback)
    v_ref: float = 1.00        # profile cruise speed [m/s]
    # trolley feedback
    k_p: float = 0.10          # position gain [1/s^2]
    k_v: float = 0.50          # velocity gain [1/s]
    zeta_s: float = 0.5        # damping ratio added to the swing
    shaping: bool = True       # zero-vibration (ZV) input shaping of the travel profile
    # hoist
    k_h: float = 4.0           # hoist velocity gain [1/s]
    a_h: float = 0.30          # hoist braking profile deceleration [m/s^2]
    h_lift_min: float = 0.5    # lift height when nothing is in the way [m]
    h_hover: float = 0.4       # wait height above the floor before set-down [m]
    v_touch: float = 0.10      # set-down speed at contact [m/s]
    # set-down permission (tighter than the success tolerances)
    settle_x: float = 0.025    # payload error [m]
    settle_A: float = 0.025    # swing amplitude [m]
    settle_v: float = 0.020    # trolley speed [m/s]
    t_predict: float = 2.0     # error must also be small after this much drift [s]
    abort_tol: float = 0.04    # abandon set-down if error, swing or speed exceed this
    ready_steps: int = 5       # conditions must hold this many steps in a row (0.5 s)
    # geometry
    clearance: float = 0.30    # height above a stack top [m]
    side_buffer: float = 0.25  # horizontal room past a stack before descending [m]


class Trapezoid:
    """Time-based trapezoidal (or triangular) profile from x0 to x1, starting at rest."""

    def __init__(self, x0: float, x1: float, a: float, v: float):
        self.x0, self.dir = x0, (1.0 if x1 >= x0 else -1.0)
        D = abs(x1 - x0)
        self.a = a
        if D >= v * v / a:                       # reaches cruise speed
            self.v = v
            self.t1 = v / a                      # end of acceleration
            self.t2 = self.t1 + (D - v * v / a) / v   # end of cruise
        else:                                    # triangular: never reaches v
            self.v = math.sqrt(D * a)
            self.t1 = self.t2 = self.v / a
        self.T = self.t2 + self.t1               # total duration
        self.D = D

    def __call__(self, t: float):
        """(x_ref, v_ref, a_ref) at time t after the start."""
        a, v, t1, t2 = self.a, self.v, self.t1, self.t2
        if t <= 0:
            s, sd, sdd = 0.0, 0.0, 0.0
        elif t < t1:
            s, sd, sdd = 0.5 * a * t * t, a * t, a
        elif t < t2:
            s, sd, sdd = 0.5 * a * t1 * t1 + v * (t - t1), v, 0.0
        elif t < self.T:
            tr = self.T - t
            s, sd, sdd = self.D - 0.5 * a * tr * tr, a * tr, -a
        else:
            s, sd, sdd = self.D, 0.0, 0.0
        return self.x0 + self.dir * s, self.dir * sd, self.dir * sdd


class ZVShaped:
    """Zero-vibration shaped profile: the average of the profile and a copy delayed by
    half a swing period, T_d = pi / omega.  The swing excited by the first half-size copy
    is cancelled by the second, so the reference ends with zero residual swing."""

    def __init__(self, profile: Trapezoid, omega: float):
        self.prof = profile
        self.delay = math.pi / omega
        self.T = profile.T + self.delay

    def __call__(self, t: float):
        a = self.prof(t)
        b = self.prof(t - self.delay)
        return tuple(0.5 * (u + v) for u, v in zip(a, b))


class BaselineController:
    def __init__(self, params: CraneParams | None = None, gains: BaselineGains | None = None):
        self.p = params if params is not None else CraneParams()
        self.g = gains if gains is not None else BaselineGains()

    # ------------------------------------------------------------------ episode
    def reset(self, obs):
        """Call once per episode with the first observation."""
        self.x_B = float(obs[6])
        self.stacks = [(float(obs[7 + 2 * i]), float(obs[8 + 2 * i]))   # (centre, top height)
                       for i in range(MAX_STACKS) if obs[8 + 2 * i] > 0]
        self.t = 0.0
        self.profile = None        # created when travel starts
        self.t_travel = 0.0
        self.th_r = self.thd_r = 0.0   # reference pendulum: the swing the profile should cause
        self.setting_down = False
        self.ready_count = 0
        self.phase = "lift"

    # ------------------------------------------------------------------ helpers
    def _stacks_in_way(self, x_p: float):
        """Stacks whose footprint overlaps the strip the container still has to sweep."""
        return stacks_in_way(x_p, self.x_B, self.stacks, self.p.w_c)

    def _z_cmd(self, x_p: float) -> float:
        """Height the container bottom should be at now."""
        ahead = self._stacks_in_way(x_p)
        if ahead:
            return max(top for _, top in ahead) + self.g.clearance
        return self.g.h_hover

    def _may_start_travel(self, x_p: float, z_b: float) -> bool:
        """True once the remaining lift finishes before the leading edge reaches the first stack."""
        g, half = self.g, self.p.w_c / 2
        ahead = self._stacks_in_way(x_p)
        h_need = max([top for _, top in ahead], default=0.0) + g.clearance
        h_need = max(h_need, g.h_lift_min)
        if z_b >= h_need:
            return True
        if not ahead:
            return z_b >= g.h_lift_min
        # distance from the leading edge to the nearest stack edge in the travel direction
        d = self.x_B - x_p
        edges = [(c - half) - (x_p + half) if d > 0 else (x_p - half) - (c + half) for c, _ in ahead]
        gap = min(edges) - g.side_buffer
        if gap <= 0:
            return False
        t_reach = math.sqrt(2 * gap / g.a_ref)                 # from rest, accelerating
        t_lift = (h_need - z_b) / self.p.vl_max + self.p.vl_max / g.a_h   # with braking
        return t_lift <= 0.8 * t_reach

    # ------------------------------------------------------------------ control
    def act(self, obs):
        """Return (a_x, a_l) acceleration commands for this control step."""
        p, g = self.p, self.g
        x, x_dot, l, l_dot, th, th_dot = (float(v) for v in obs[:6])
        L = pendulum_length(l, p)
        x_p = x + L * math.sin(th)
        z_b = container_bottom(l, p)

        # ---- trolley
        if self.profile is None and self._may_start_travel(x_p, z_b):
            prof = Trapezoid(x, self.x_B, g.a_ref, g.v_ref)
            self.profile = ZVShaped(prof, math.sqrt(p.g / L)) if g.shaping else prof
            self.t_travel = 0.0
            self.phase = "travel"
        if self.profile is None:
            x_ref, v_ref, a_ref = x, 0.0, 0.0                  # hold position while lifting
        else:
            x_ref, v_ref, a_ref = self.profile(self.t_travel)
            self.t_travel += p.dt_ctrl
        # reference pendulum, linearised, with the same hoisting (Coriolis) term:
        #     th_r'' = -(g th_r + a_ref + 2 l_dot th_r') / L
        # Once the profile has ended it is retired (set to rest): from then on the
        # feedback damps whatever swing is left, instead of copying a reference swing.
        if self.profile is not None and self.t_travel > self.profile.T:
            self.th_r = self.thd_r = 0.0
        else:
            n = 10
            h = p.dt_ctrl / n
            for _ in range(n):
                self.thd_r += h * (-(p.g * self.th_r + a_ref + 2 * l_dot * self.thd_r) / L)
                self.th_r += h * self.thd_r
        c = 2.0 * g.zeta_s * math.sqrt(p.g * L)
        a_x = (a_ref + g.k_p * (x_ref - x) + g.k_v * (v_ref - x_dot)
               + c * (th_dot - self.thd_r))

        # ---- hoist
        omega = math.sqrt(p.g / L)
        v_d = l_dot * math.sin(th) + L * th_dot * math.cos(th)
        A = math.hypot(L * math.sin(th), v_d / omega)
        err = x_p - self.x_B
        ready = (self.profile is not None and self.t_travel > self.profile.T
                 and not self._stacks_in_way(x_p)
                 and abs(err) < g.settle_x and abs(err + x_dot * g.t_predict) < g.settle_x
                 and A < g.settle_A and abs(x_dot) < g.settle_v)
        self.ready_count = self.ready_count + 1 if ready else 0
        if self.ready_count >= g.ready_steps:
            self.setting_down = True
        elif self.setting_down and max(abs(err), A, abs(x_dot)) > g.abort_tol:
            self.setting_down = False          # drifted off: back to hover height and retry
        if self.setting_down:
            self.phase = "set_down"
            v_prof = math.sqrt(g.v_touch ** 2 + 2 * g.a_h * max(z_b, 0.0))
        else:
            z_cmd = self._z_cmd(x_p) if self.profile is not None else max(
                self._z_cmd(x_p), g.h_lift_min)
            if self.profile is not None and not self._stacks_in_way(x_p):
                self.phase = "lower"
            dz = z_b - z_cmd                                   # > 0: too high, lower
            v_prof = math.copysign(math.sqrt(2 * g.a_h * abs(dz)), dz)
        # on the braking part of the profile, the reference itself decelerates:
        # d(v_ref)/dt = -a_h * l_dot / |v_ref|; feeding that forward removes the tracking lag
        if abs(v_prof) < p.vl_max:
            v_l_ref = v_prof
            a_ff = -g.a_h * l_dot / max(abs(v_prof), 0.05)
        else:
            v_l_ref, a_ff = math.copysign(p.vl_max, v_prof), 0.0
        a_l = a_ff + g.k_h * (v_l_ref - l_dot)

        self.t += p.dt_ctrl
        return a_x, a_l


def run_episode(env, controller, layout=None, seed=None, record=False):
    """Run one episode with the baseline. Returns info['episode'] (and the trajectory)."""
    opts = {"layout": layout} if layout is not None else None
    obs, _ = env.reset(seed=seed, options=opts)
    controller.reset(obs)
    traj = [(obs.copy(), controller.phase)] if record else None
    while True:
        a_x, a_l = controller.act(obs)
        obs, r, term, trunc, info = env.step_command(a_x, a_l)
        if record:
            traj.append((obs.copy(), controller.phase))
        if term or trunc:
            return (info["episode"], traj) if record else info["episode"]