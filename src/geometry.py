from typing import NamedTuple
import math

from dynamics import CraneParams, hook_point

#NamedTuple is immutable and reachble by index or name, like a tuple or a dataclass
class Rect(NamedTuple):
    x_lo: float #left edge
    x_hi: float #right edge
    z_lo: float #bottom edge
    z_hi: float #top edge

def stack_rect(center: float, n_high: int ,params: CraneParams) -> Rect:
    """stack of n_high rectangles centered at x=center, with the crane's container dimensions"""
    w_c, h_c = params.w_c, params.h_c
    return Rect(center - w_c/2, center + w_c/2, 0.0, n_high * h_c)

def rect_corners(r: Rect) -> tuple:
    """Four corners of a rectangle, TL, TR, BR, BL"""
    return ((r.x_lo, r.z_hi), (r.x_hi, r.z_hi), (r.x_hi, r.z_lo), (r.x_lo, r.z_lo))

def container_corners(s: tuple, params: CraneParams) -> tuple:
    """
        Four corners of the hanging, rotated container (TL, TR, BR, BL)
        d = (sin th, -cos th)   down along the cable
        u = (cos th,  sin th)   across the container
        corner = hook + a*u + b*d,  a in {-w_c/2, +w_c/2},  b in {0, h_c}
    """
    x, x_dot, l, l_dot, theta, theta_dot = s
    d = (math.sin(theta), -math.cos(theta))  # down along the cable
    u = (math.cos(theta), math.sin(theta))     # across the container
    hook = hook_point(s, params)
    w2, h = params.w_c / 2, params.h_c
    return tuple((hook[0] + a * u[0] + b * d[0], hook[1] + a * u[1] + b * d[1])
                 for a, b in ((-w2, 0.0), (w2, 0.0), (w2, h), (-w2, h))) # TL, TR, BR, BL

def _project(points, axis) -> tuple:
    """
    (min, max) of the dot products of the points with the axis vector.
    """
    dot_products = [point[0] * axis[0] + point[1] * axis[1] for point in points]
    return min(dot_products), max(dot_products)


def container_hits_rect(corners, r, margin: float, theta: float) -> bool:
    """
    Separating axis test on x,z,u, d with a safety margin.
    project the corners of the container and the rectangle onto each axis, and check for overlap.
    """
    u = (math.cos(theta), math.sin(theta))  # across the container
    d = (math.sin(theta), -math.cos(theta))  # down along the cable
    x = (1.0, 0.0)  # horizontal
    z = (0.0, 1.0)  # vertical
    axes = [x, z, u, d]
    rectangle_corners = rect_corners(r)
    for a in axes:
        a_min, a_max = _project(corners, a)
        r_min, r_max = _project(rectangle_corners, a)
        if not (a_min - margin <= r_max and r_min <= a_max + margin):
            return False
    return True

def first_collision(corners, rects, margin: float, theta: float):
    """
    Index of the first stack the container hits, or None.
    """
    for i, r in enumerate(rects):
        if container_hits_rect(corners, r, margin, theta):
            return i
    return None


def lowest_z(corners) -> float:
    """
    Hight of the lowest container corner (0 = touching the ground).
    """
    return min(point[1] for point in corners)





