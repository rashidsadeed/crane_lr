from dataclasses import dataclass
import math

@dataclass(frozen=True, slots=True)
class CraneParams:
    dt_sim: float = 0.01
    dt_ctrl: float = 0.1
    g: float = 9.81 #m/s^2
    H: float = 10.0 # crane high m
    #container dim
    h_c: float = 2.0 #m
    w_c: float = 2.4 #m
    #cable movement limits
    l_min: float = 1.5
    l_max: float = 8.0
    #trolley limits
    vx_max: float = 1.0 #m/s
    ax_max: float = 0.5 #m/s^2
    #hoist limits
    vl_max: float = 0.5
    al_max: float = 0.4
    #trolley end stops
    x_min: float = 0.0
    x_max: float = 12.0

    zeta : float = 0.0 #swing damping ratio (optional)


# s = [x, x_dot, l, l_dot, theta, theta_dot]
def pendulum_length(l: float, params: CraneParams) -> float:
    L = l + params.h_c/2
    return L

def theta_accl(a_x, sin_th, cos_th, l_dot, theta_dot, L, params:CraneParams):
    """theta_ddot = -( g*sin(theta) + a_x*cos(theta) + 2*l_dot*theta_dot ) / L
                     - 2*zeta*omega*theta_dot"""
    # gravity + trolley acc + coriolis (changing length)
    numerator = params.g * sin_th + a_x * cos_th + 2.0 * l_dot * theta_dot
    theta_ddot = -numerator / L

    #damping
    if params.zeta > 0.0:
        omega = math.sqrt(params.g / L)
        theta_ddot -= 2.0 * params.zeta * omega * theta_dot
    return theta_ddot

def payload_center(s: tuple, params: CraneParams) -> tuple:
    """(x_p, z_p): container centre, z up from the floor.
        x_p = x + L*sin(theta),   z_p = H - L*cos(theta),   L = l + h_c/2
    """
    x, _, l, _, theta, _ = s
    L = pendulum_length(l, params)
    return (x + L * math.sin(theta), params.H - L * math.cos(theta))


def swing(s: tuple, params: CraneParams) -> tuple:
    """(d, v_d, A): swing measured in metres at the container centre.
        d     = L*sin(theta)                              sideways offset
        v_d   = l_dot*sin(theta) + L*theta_dot*cos(theta) its rate (product rule)
        omega = sqrt(g/L)                                 natural frequency
        A     = sqrt(d^2 + (v_d/omega)^2)                 swing amplitude
    """
    _, _, l, l_dot, theta, theta_dot = s
    L = pendulum_length(l, params)
    sin_th, cos_th = math.sin(theta), math.cos(theta)
    d = L * sin_th
    v_d = l_dot * sin_th + L * theta_dot * cos_th
    omega = math.sqrt(params.g / L)
    A = math.sqrt(d * d + (v_d / omega) ** 2)
    return d, v_d, A

def derivatives(s: tuple, a_x:float, a_l:float, params: CraneParams):
    """return ds/dt as a 6-tuple for applied accelerations a_x, a_l
    
        derivatin using the euler-lagrange method for a point mass at
            (x + L sin(theta), -L cos(theta)):
        d/dt( L^2 * theta_dot ) + L*a_x*cos(theta) + g*L*sin(theta) = 0

        mass cancles out, so it doesnt appear.
    """
    x, x_dot, l, l_dot, theta, theta_dot = s
    L = pendulum_length(l, params)
    sin_th = math.sin(theta)
    cos_th = math.cos(theta)

    theta_ddot = theta_accl(a_x, sin_th, cos_th, l_dot, theta_dot, L, params)

    return (x_dot, a_x, l_dot, a_l, theta_dot, theta_ddot)


def hook_point(s: tuple, params: CraneParams) -> tuple:
    """
    Return the position of the hook (x_h, z_h) in the global coordinate system.
    x_h = x + l*sin(theta), z_h = H - l*cos(theta)
    """
    x, _, l, _, theta, _ = s
    hook = x + (l * math.sin(theta)), params.H - (l * math.cos(theta))
    return hook

def _add_scaled(s,k,h):
    """element-wise s + h*k for 6-tuples"""
    return tuple(s_i + h *k_i for s_i, k_i in zip(s,k))

def rk4_step(s: tuple, a_x:float, a_l: float, dt: float, params:CraneParams) -> tuple:
    """
    measures slopes at points across the step and averages them
    derivates ===  f()
    k1: at the start (current slope) --- f(s)
    k2: at the midpoint, reached by following k1 for dt/2 --- f(s + dt/2 * k1)
    k3: reached by following k2 for dt/2 --- f(s + dt/2 * k2)
    k4: at the end following k3 for the full dt --- f(s + dt * k3)
    s_next = s + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    k1 = derivatives(s, a_x, a_l, params)
    k2 = derivatives(_add_scaled(s, k1, dt/2), a_x, a_l, params)
    k3 = derivatives(_add_scaled(s, k2, dt/2), a_x, a_l, params)
    k4 = derivatives(_add_scaled(s, k3, dt), a_x, a_l, params)
    #s_next
    return tuple([s_i + dt/6 * (k1_i + 2 * k2_i + 2 * k3_i + k4_i) for s_i, k1_i, k2_i, k3_i, k4_i in zip(s, k1, k2, k3, k4)])

def pendulum_energy(s: tuple, params:CraneParams):
    """
    E = Kinetic + Potential
    """
    x, x_dot, l, l_dot, theta, theta_dot = s
    L = pendulum_length(l, params)

    k = 1/2 * (L*theta_dot)**2
    p = params.g * L * (1 - math.cos(theta))

    return k + p

def _clamp(value, lo, hi):
    return max(lo, min(hi, value))

def _saturate_axis(v: float, a_cmd: float, v_max: float, a_max: float, dt: float) -> float:
    """
    applied acceleration for one drive axis
        1. |a| <= a_max
        2. v + a*dt stays within [-v_max, v_max]
    """
    a_cmd = _clamp(a_cmd, -a_max, a_max)
    a_lo = (-v_max - v) / dt
    a_hi = (v_max - v) / dt
    return _clamp(a_cmd, a_lo, a_hi)

def saturate(s, a_x_cmd, a_l_cmd, params: CraneParams):
    """
    saturate the commanded accelerations for the trolley and hoist
    """
    x, x_dot, l, l_dot, theta, theta_dot = s
    a_x = _saturate_axis(x_dot, a_x_cmd, params.vx_max, params.ax_max, params.dt_sim)
    a_l = _saturate_axis(l_dot, a_l_cmd, params.vl_max, params.al_max, params.dt_sim)
    return a_x, a_l

def _enforce_stops(s: tuple, params: CraneParams):
    """Clamp trolley position and cable length to their ranges.

    Hitting a limit clamps the position and zeroes the matching velocity:
        x < x_min  -> x = x_min, x_dot = 0, x_hit = True      (left end buffer)
        x > x_max  -> x = x_max, x_dot = 0, x_hit = True      (right end buffer)
        l < l_min  -> l = l_min, l_dot = 0, l_top_hit = True  (upper limit switch)
        l > l_max  -> l = l_max, l_dot = 0                    (container on floor;
                                                               env.py judges the landing)
    theta and theta_dot are left unchanged.

    Returns (s_new, x_hit, l_top_hit).
    """
    x, x_dot, l, l_dot, theta, theta_dot = s
    x_hit = l_top_hit = False
    if x < params.x_min:
        x, x_dot, x_hit = params.x_min, 0.0, True
    elif x > params.x_max:
        x, x_dot, x_hit = params.x_max, 0.0, True

    if l < params.l_min:
        l, l_dot, l_top_hit = params.l_min, 0.0, True
    elif l > params.l_max:
        l, l_dot = params.l_max, 0.0

    return (x, x_dot, l, l_dot, theta, theta_dot), x_hit, l_top_hit

def step(s: tuple, a_x_cmd: float, a_l_cmd: float, params: CraneParams, return_substeps: bool =False):

    """Advance one control step (dt_ctrl) with the command held constant.

    Each of the n = dt_ctrl/dt_sim substeps:
        (a_x, a_l) = saturate(s, a_x_cmd, a_l_cmd, dt_sim)
        s          = rk4_step(s, a_x, a_l, dt_sim)
        s          = enforce_stops(s)

    Returns (s_next, info), info = {
        "a_x": mean applied trolley accel over the step,
        "a_l": mean applied hoist accel over the step,
        "x_stop_hit": bool, "l_stop_hit": bool,
        "substeps": [states after each substep]   (only if return_substeps)
    }
    """

    n_sub = int(round(params.dt_ctrl / params.dt_sim)) #10
    dt = params.dt_sim

    x_hit = l_hit = False
    sum_ax = sum_al = 0.0
    substeps = [] if return_substeps else None

    for _ in range(n_sub):
        a_x, a_l = saturate(s, a_x_cmd, a_l_cmd, params)
        s = rk4_step(s, a_x, a_l, dt, params)
        s, hx, hl = _enforce_stops(s, params)
        x_hit = x_hit or hx
        l_hit = l_hit or hl
        sum_ax += a_x
        sum_al += a_l
        if return_substeps:
            substeps.append(s)

    info = {
        "a_x": sum_ax / n_sub,
        "a_l": sum_al / n_sub,
        "x_stop_hit": x_hit,
        "l_stop_hit": l_hit,
    }

    if return_substeps:
        info["substeps"] = substeps

    return s, info



if __name__ == "__main__":
    print("start")
    p = CraneParams()
    s0 = (5.0, 0.0, 6.0, 0.0, math.radians(10), 0.0)
    dt, n = 0.01, 6000
    s_rk4 = s0
    s_eul = s0
    E0 = pendulum_energy(s0, p)


    #test the _enforce_stops function
    print(_enforce_stops((12.03, 1.0, 4.0, 0.0, 0.1, 0.2), p))
    # ((12.0, 0.0, 4.0, 0.0, 0.1, 0.2), True, False)
    print(_enforce_stops((5.0, 0.0, 1.48, -0.5, 0.0, 0.0), p))
    # ((5.0, 0.0, 1.5, 0.0, 0.0, 0.0), False, True)
    print(_enforce_stops((5.0, 0.0, 8.01, 0.3, 0.0, 0.0), p))
    # ((5.0, 0.0, 8.0, 0.0, 0.0, 0.0), False, False)

    # test the acceleration saturation function
    # (velocity, commanded accel, expected applied accel, reason)
    cases = [
        ( 0.0,    2.0,  0.5, "acceleration limit"),
        ( 0.98,   0.5,  0.5, "velocity limit looser (a_hi = 2.0)"),
        ( 0.998,  0.5,  0.2, "velocity limit kicks in (a_hi = 0.2)"),
        ( 1.0,    0.5,  0.0, "at speed limit, no further acceleration"),
        ( 1.0,   -0.5, -0.5, "braking always allowed"),
        (-1.0,   -0.5,  0.0, "mirrored speed limit"),
    ]
    all_passed = True
    for v, a_cmd, a_exp, reason in cases:
        a = _saturate_axis(v, a_cmd, p.vx_max, p.ax_max, dt)
        ok = math.isclose(a, a_exp, rel_tol=1e-9)
        all_passed &= ok
        print(f"{'PASS' if ok else 'FAIL'}  v={v:+.3f}  a_cmd={a_cmd:+.2f}  "
              f"-> a={a:+.6f}  (expected {a_exp:+.2f})  {reason}")

    ## test energy drift of rk4 vs euler
    # for i in range(n):
    #     s_rk4 = rk4_step(s_rk4, 0.0, 0.0, dt, p)
    #     s_eul = _add_scaled(s_eul, derivatives(s_eul, 0.0, 0.0, p), dt)

    # drift_rk4 = abs(pendulum_energy(s_rk4,p) - E0) / E0
    # drift_euler = abs(pendulum_energy(s_eul,p) - E0) / E0

    # print(f"rk4 drift: {drift_rk4:.02e}")
    # print(f"euler drift: {drift_euler:.02e}")