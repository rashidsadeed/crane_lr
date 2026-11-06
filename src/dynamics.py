from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class CraneParams:
    dt_sim: float = 0.01
    dt_ctrl: float = 0.1
    g: float = 9.81
    H: float = 10.0
    h_c: float = 2.0
    w_c: float = 2.4
    l_min: float = 1.5
    l_max: float = 8.0
    vx_max: float = 1.0
    ax_max: float = 0.5
    vl_max: float = 0.5
    al_max: float = 0.4
    x_min: float = 0.0
    x_max: float = 12.0
    zeta: float = 0.0


def pendulum_length(l, params):
    return l + params.h_c / 2


def theta_accel(a_x, sin_th, cos_th, l_dot, theta_dot, L, params):
    theta_ddot = -(params.g * sin_th + a_x * cos_th + 2.0 * l_dot * theta_dot) / L
    if params.zeta > 0.0:
        theta_ddot -= 2.0 * params.zeta * math.sqrt(params.g / L) * theta_dot
    return theta_ddot


def payload_center(s, params):
    x, _, l, _, theta, _ = s
    L = pendulum_length(l, params)
    return (x + L * math.sin(theta), params.H - L * math.cos(theta))


def swing(s, params):
    _, _, l, l_dot, theta, theta_dot = s
    L = pendulum_length(l, params)
    sin_th, cos_th = math.sin(theta), math.cos(theta)
    d = L * sin_th
    v_d = l_dot * sin_th + L * theta_dot * cos_th
    omega = math.sqrt(params.g / L)
    return d, v_d, math.sqrt(d * d + (v_d / omega) ** 2)


def derivatives(s, a_x, a_l, params):
    x, x_dot, l, l_dot, theta, theta_dot = s
    L = pendulum_length(l, params)
    th_dd = theta_accel(a_x, math.sin(theta), math.cos(theta), l_dot, theta_dot, L, params)
    return (x_dot, a_x, l_dot, a_l, theta_dot, th_dd)


def hook_point(s, params):
    x, _, l, _, theta, _ = s
    return x + l * math.sin(theta), params.H - l * math.cos(theta)


def _add_scaled(s, k, h):
    return tuple(a + h * b for a, b in zip(s, k))


def rk4_step(s, a_x, a_l, dt, params):
    k1 = derivatives(s, a_x, a_l, params)
    k2 = derivatives(_add_scaled(s, k1, dt / 2), a_x, a_l, params)
    k3 = derivatives(_add_scaled(s, k2, dt / 2), a_x, a_l, params)
    k4 = derivatives(_add_scaled(s, k3, dt), a_x, a_l, params)
    return tuple(a + dt / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(s, k1, k2, k3, k4))


def pendulum_energy(s, params):
    _, _, l, _, theta, theta_dot = s
    L = pendulum_length(l, params)
    return 0.5 * (L * theta_dot) ** 2 + params.g * L * (1 - math.cos(theta))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _saturate_axis(v, a_cmd, v_max, a_max, dt):
    a = _clamp(a_cmd, -a_max, a_max)
    return _clamp(a, (-v_max - v) / dt, (v_max - v) / dt)


def saturate(s, a_x_cmd, a_l_cmd, params):
    return (_saturate_axis(s[1], a_x_cmd, params.vx_max, params.ax_max, params.dt_sim),
            _saturate_axis(s[3], a_l_cmd, params.vl_max, params.al_max, params.dt_sim))


def _enforce_stops(s, params):
    x, x_dot, l, l_dot, theta, theta_dot = s
    x_hit = l_top_hit = False
    l_top_speed = x_stop_speed = 0.0
    if x < params.x_min:
        x_stop_speed = abs(x_dot)
        x, x_dot, x_hit = params.x_min, 0.0, True
    elif x > params.x_max:
        x_stop_speed = abs(x_dot)
        x, x_dot, x_hit = params.x_max, 0.0, True
    if l < params.l_min:
        l_top_speed = abs(l_dot)
        l, l_dot, l_top_hit = params.l_min, 0.0, True
    elif l > params.l_max:
        l, l_dot = params.l_max, 0.0
    return (x, x_dot, l, l_dot, theta, theta_dot), x_hit, l_top_hit, l_top_speed, x_stop_speed


def step(s, a_x_cmd, a_l_cmd, params, return_substeps=False):
    n_sub = int(round(params.dt_ctrl / params.dt_sim))
    dt = params.dt_sim
    x_hit = l_hit = False
    top_speed = x_speed = 0.0
    sum_ax = sum_al = 0.0
    subs = [] if return_substeps else None
    for _ in range(n_sub):
        a_x, a_l = saturate(s, a_x_cmd, a_l_cmd, params)
        s = rk4_step(s, a_x, a_l, dt, params)
        s, hx, hl, v_top, v_x = _enforce_stops(s, params)
        x_speed = max(x_speed, v_x)
        x_hit = x_hit or hx
        l_hit = l_hit or hl
        top_speed = max(top_speed, v_top)
        sum_ax += a_x
        sum_al += a_l
        if return_substeps:
            subs.append(s)
    info = {"a_x": sum_ax / n_sub, "a_l": sum_al / n_sub, "x_stop_hit": x_hit,
            "l_stop_hit": l_hit, "l_top_speed": top_speed, "x_stop_speed": x_speed}
    if return_substeps:
        info["substeps"] = subs
    return s, info