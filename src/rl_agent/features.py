"""feature extraction: raw observation (13 values) -> 12 relative features for tile coding"""
import math
import numpy as np
from dynamics import CraneParams, container_bottom, pendulum_length
import geometry as geo
from scenarios import MAX_STACKS, ScenarioConfig

G_NEXT_OPEN = 5.0      # g_next when no stack blocks the way [m]
G_SIDE_OPEN = 3.0      # g_side when the target has no neighbour within this distance [m]

FEATURE_NAMES = ("e_x", "x_dot", "z_b", "l_dot", "d", "v_d", "L",
                 "c_path", "g_next", "c_below", "g_side", "h_side")


def warp(u: float) -> float:
    """signed square root: streches values near zero, compresses large values"""
    return math.copysign(math.sqrt(abs(u)), u)

def unpack(obs, params: CraneParams) -> dict:
    """Raw observation -> named physical quantities used by the features"""
    x, x_dot, l, l_dot, theta, theta_dot = (float(v) for v in obs[:6])
    L = pendulum_length(l, params)
    sin_th, cos_th = math.sin(theta), math.cos(theta)
    stacks = [(float(obs[7 + 2 * i]), float(obs[8 + 2 * i]))   # (centre, top height)
              for i in range(MAX_STACKS) if obs[8 + 2 * i] > 0] # skip empty slots
    return {
        "x": x, "x_dot": x_dot, "l": l, "l_dot": l_dot, "L": L,
        "x_B": float(obs[6]),
        "x_p": x + L * sin_th,                                  # container centre
        "z_b": container_bottom(l, params),                     # container bottom
        "d": L * sin_th,                                        # swing displacement
        "v_d": l_dot * sin_th + L * theta_dot * cos_th,         # its rate
        "stacks": stacks,
    }

def motion_features(q: dict) -> tuple:
    """f1-f7: e_x, x_dot, z_b, l_dot, d, v_d, L  (e_x, d, v_d warped)."""
    e_x = q["x_B"] - q["x_p"]

    return (
        warp(e_x),
        q["x_dot"],
        q["z_b"],
        q["l_dot"],
        warp(q["d"]),
        warp(q["v_d"]),
        q["L"],
    )


def _overlaps(c: float, lo: float, hi: float, half: float) -> bool:
    """Does the footprint [c - half, c + half] overlap the interval [lo, hi]?"""
    return c - half < hi and c + half > lo


def stacks_in_way(q: dict, half: float) -> list:
    """Stacks whose footprint overlaps the strip the container still has to sweep:
        strip = [min(x_p, x_B) - half, max(x_p, x_B) + half]"""
    return geo.stacks_in_way(q["x_p"], q["x_B"], q["stacks"], 2 * half)


def path_features(q: dict, params: CraneParams) -> tuple:
    """f8-f10: c_path, g_next, c_below.

    c_path  = z_b - (tallest top among stacks in the way)            z_b if none
    g_next  = gap from the container's leading edge to the nearest stack in the way
              that is taller than z_b (would block at the current height),
              clipped to [0, G_NEXT_OPEN];                            G_NEXT_OPEN if none
    c_below = z_b - (tallest top among stacks overlapping the container's footprint)
                                                                      z_b if none
    """
    half = params.w_c / 2
    z_b, x_p = q["z_b"], q["x_p"]
    ahead = stacks_in_way(q, half)

    # f8: clearance over the tallest stack still ahead (the floor counts as height 0)
    c_path = z_b - max((top for _, top in ahead), default=0.0)

    # f9: leading edge to the nearest stack that blocks at the current height
    direction = 1.0 if q["x_B"] >= x_p else -1.0
    gaps = [direction * (c - x_p) - 2 * half for c, top in ahead if top > z_b]
    g_next = min(max(min(gaps), 0.0), G_NEXT_OPEN) if gaps else G_NEXT_OPEN

    # f10: clearance over whatever is directly under the container
    below = [top for c, top in q["stacks"] if _overlaps(c, x_p - half, x_p + half, half)]
    c_below = z_b - max(below, default=0.0)

    return c_path, g_next, c_below

def target_features(q: dict, params: CraneParams) -> tuple:
    """f11-f12: g_side, h_side.

    Neighbours of the target = stacks whose footprint is within G_SIDE_OPEN of the
    target footprint, on either side (edge gap = |c - x_B| - w_c).
    g_side = smallest edge gap to a neighbour, clipped to [0, G_SIDE_OPEN];  G_SIDE_OPEN if none
    h_side = top height of the tallest neighbour;                            0 if none
    """
    W = params.w_c
    near = [(abs(c - q["x_B"]) - W, top) for c, top in q["stacks"]
            if abs(c - q["x_B"]) - W <= G_SIDE_OPEN]
    if not near:
        return G_SIDE_OPEN, 0.0
    g_side = min(max(min(g for g, _ in near), 0.0), G_SIDE_OPEN)
    h_side = max(top for _, top in near)
    return g_side, h_side


def feature_ranges(params: CraneParams, max_top: float | None = None) -> tuple:
    """(low, high) per feature, in the same order as FEATURE_NAMES.
    The tile coder spreads its tiles over these ranges and clips values outside them.
    Warped features get warped bounds. max_top defaults to the tallest stack the default
    scenario config allows (max_high * h_c = 4 m)."""
    if max_top is None:
        max_top = ScenarioConfig().max_high * params.h_c
    z_max = params.H - params.l_min - params.h_c                 # highest bottom: 6.5 m
    return (
        (warp(-2.0), warp(params.x_max)),                         # e_x   (m, warped)
        (-params.vx_max, params.vx_max),                          # x_dot
        (0.0, z_max),                                             # z_b
        (-params.vl_max, params.vl_max),                          # l_dot
        (warp(-1.0), warp(1.0)),                                  # d     (warped)
        (warp(-1.5), warp(1.5)),                                  # v_d   (warped)
        (params.l_min + params.h_c / 2, params.l_max + params.h_c / 2),   # L
        (-max_top - 0.5, z_max),                                  # c_path
        (0.0, G_NEXT_OPEN),                                       # g_next
        (0.0, z_max),                                             # c_below
        (0.0, G_SIDE_OPEN),                                       # g_side
        (0.0, max_top),                                           # h_side
    )


def compute_features(obs, params: CraneParams) -> np.ndarray:
    """Raw observation (13) -> the 12 features, as a float array (order: FEATURE_NAMES)."""
    q = unpack(obs, params)
    return np.array(motion_features(q) + path_features(q, params) + target_features(q, params))