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

    z : float = 0.0 #swing damping ratio (optional)


# s = [x, x_dot, l, l_dot, theta, theta_dot]
def pendulum_length(l: float, params: CraneParams) -> float:
    L = l + params.h_c/2
    return L

def theta_rate(a_x, sin_th, cos_th, l_dot, theta_dot, L, params:CraneParams):
    """theta_ddot = -( g*sin(theta) + a_x*cos(theta) + 2*l_dot*theta_dot ) / L
                     - 2*zeta*omega*theta_dot"""
    # gravity + trolley acc + coriolis (changing length)
    numerator = params.g * sin_th + a_x * cos_th + 2.0 * l_dot * theta_dot
    theta_ddot = -numerator / L

    #damping
    if params.z > 0.0:
        omega = math.sqrt(params.g / L)
        theta_ddot -= 2.0 * params.z * omega * theta_dot
    return theta_ddot

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

    theta_ddot = swing(a_x, sin_th, cos_th, l_dot, theta_dot, L, params)

    return tuple(x_dot, a_x, l_dot, a_l, theta_dot, theta_ddot)


def hook_point(s: tuple, params: CraneParams) -> tuple:
    """x_h, z_h"""
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

