"""Scenario generator: start, target and obstacle stacks for each episode.

The floor is a line; every object occupies a footprint of width w_c:
    start  [x_A - w_c/2, x_A + w_c/2]
    target [x_B - w_c/2, x_B + w_c/2]   (reserved: nothing may stand there)
    stack  [c   - w_c/2, c   + w_c/2]
Neighbouring footprints must be at least g_min apart (edge to edge),
i.e. their centres at least w_c + g_min apart.

Layout types:
    S0  open floor            no stacks
    S1  over one              1 stack anywhere between A and B
    S2  behind obstacle       1 stack directly left of the target
    S3  before obstacle       1 stack directly right of the target
    S4  slot                  stacks directly on both sides of the target
    S5  off-path              1-2 stacks beyond B, at least 2 m clear of it
    S6  mixed                 a path stack plus an S2, S3 or S4 arrangement at B

All samplers build layouts constructively (gaps first, then positions) and
return None when a draw cannot fit; sample_layout retries the same type.
Randomness comes only from the numpy Generator passed in, so one seed
reproduces one layout.
"""
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from dynamics import CraneParams
from geometry import stack_rect


# ------------------------------------------------------------------ data types

class Stack(NamedTuple):
    center: float      # x of the stack's centre [m]
    n_high: int        # number of containers stacked (1 or 2)


class Layout(NamedTuple):
    kind: str          # "S0" ... "S6"
    x_A: float         # start position (container on the floor)
    x_B: float         # target position
    stacks: tuple      # tuple of Stack, 0-3 of them


DEFAULT_WEIGHTS = {"S0": 0.05, "S1": 0.20, "S2": 0.15, "S3": 0.15,
                   "S4": 0.15, "S5": 0.10, "S6": 0.20}


@dataclass(frozen=True)
class ScenarioConfig:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    g_min: float = 0.4               # smallest edge-to-edge gap [m]
    g_max: float = 1.5               # largest gap for adjacent placements [m]
    max_high: int = 2                # tallest stack allowed
    max_stacks: int = 3
    x_A_range: tuple = (0.0, 2.5)
    min_travel: float = 4.0          # x_B >= x_A + min_travel
    off_path_gap: tuple = (2.0, 3.0) # S5: free gap between target and first stack [m]
    fixed_AB: tuple | None = None    # e.g. (0.0, 10.0) for curriculum stage C0


def layout_rects(layout: Layout, params: CraneParams) -> tuple:
    """Stacks as Rects, ready for geometry.first_collision."""
    return tuple(stack_rect(st.center, st.n_high, params) for st in layout.stacks)


# ------------------------------------------------------------------ validation

def validate_layout(layout: Layout, cfg: ScenarioConfig, params: CraneParams) -> list:
    """Return a list of rule violations (strings); an empty list means valid."""
    violations = []
    eps = 1e-9                                   # tolerance for floating-point noise

    # 1. start and target
    lo, hi = cfg.x_A_range
    if not (lo - eps <= layout.x_A <= hi + eps):
        violations.append(f"x_A = {layout.x_A:.2f} outside [{lo}, {hi}]")
    if layout.x_B < layout.x_A + cfg.min_travel - eps:
        violations.append(f"x_B = {layout.x_B:.2f} < x_A + {cfg.min_travel} "
                          f"= {layout.x_A + cfg.min_travel:.2f}")
    if layout.x_B > params.x_max + eps:
        violations.append(f"x_B = {layout.x_B:.2f} > x_max = {params.x_max}")

    # 2. stacks
    if len(layout.stacks) > cfg.max_stacks:
        violations.append(f"{len(layout.stacks)} stacks > max {cfg.max_stacks}")
    for i, st in enumerate(layout.stacks):
        if not (1 <= st.n_high <= cfg.max_high):
            violations.append(f"stack{i} n_high = {st.n_high} outside [1, {cfg.max_high}]")

    # 3. workspace: every footprint centre in [x_min, x_max]
    footprints = [("A", layout.x_A), ("B", layout.x_B)]
    footprints += [(f"stack{i}", st.center) for i, st in enumerate(layout.stacks)]
    for name, c in footprints:
        if not (params.x_min - eps <= c <= params.x_max + eps):
            violations.append(f"{name} centre {c:.2f} outside [{params.x_min}, {params.x_max}]")

    # 4. gaps between neighbouring footprints, sorted left to right
    footprints.sort(key=lambda f: f[1])
    for (n1, c1), (n2, c2) in zip(footprints, footprints[1:]):
        gap = c2 - c1 - params.w_c               # edge-to-edge distance
        if gap < cfg.g_min - eps:
            violations.append(f"gap {n1}-{n2} = {gap:.2f} < {cfg.g_min}")

    return violations


# ------------------------------------------------------------------ helpers

def _gap(rng, cfg) -> float:
    """Edge-to-edge gap for an adjacent placement, U[g_min, g_max]."""
    return float(rng.uniform(cfg.g_min, cfg.g_max))


def _height(rng, cfg) -> int:
    """Stack height, 1 .. max_high (inclusive)."""
    return int(rng.integers(1, cfg.max_high + 1))


def _start_target(rng, cfg, params, x_B_lo_offset: float, x_B_hi: float):
    """Draw (x_A, x_B) with  x_A + x_B_lo_offset <= x_B <= x_B_hi.

    x_A is drawn only from the part of x_A_range that leaves room for x_B,
    so tight layout types do not waste draws. Returns None if nothing fits.
    Honours cfg.fixed_AB.

    Note: rng.uniform(lo, hi) silently accepts lo > hi, so every range is
    checked before drawing.
    """
    if cfg.fixed_AB is not None:
        x_A, x_B = cfg.fixed_AB
        return (x_A, x_B) if x_A + x_B_lo_offset <= x_B <= x_B_hi else None

    a_lo, a_hi = cfg.x_A_range
    a_hi = min(a_hi, x_B_hi - x_B_lo_offset)      # leave room for x_B
    if a_lo > a_hi:
        return None
    x_A = float(rng.uniform(a_lo, a_hi))
    return x_A, float(rng.uniform(x_A + x_B_lo_offset, x_B_hi))


# ------------------------------------------------------------------ samplers
# Each returns a Layout, or None if this draw cannot fit.

def _sample_s0(rng, cfg, params):
    """Open floor: no stacks."""
    ab = _start_target(rng, cfg, params, cfg.min_travel, params.x_max)
    if ab is None:
        return None
    return Layout("S0", ab[0], ab[1], ())


def _sample_s1(rng, cfg, params):
    """One stack anywhere in the path, at least g_min from A and from B."""
    W, g = params.w_c, cfg.g_min
    ab = _start_target(rng, cfg, params, max(cfg.min_travel, 2 * W + 2 * g), params.x_max)
    if ab is None:
        return None
    x_A, x_B = ab
    c = float(rng.uniform(x_A + W + g, x_B - W - g))
    return Layout("S1", x_A, x_B, (Stack(c, _height(rng, cfg)),))


def _sample_s2(rng, cfg, params):
    """One stack directly left of the target, gap U[g_min, g_max]."""
    W = params.w_c
    g = _gap(rng, cfg)
    ab = _start_target(rng, cfg, params,
                       max(cfg.min_travel, 2 * W + cfg.g_min + g), params.x_max)
    if ab is None:
        return None
    x_A, x_B = ab
    return Layout("S2", x_A, x_B, (Stack(x_B - W - g, _height(rng, cfg)),))


def _sample_s3(rng, cfg, params):
    """One stack directly right of the target, gap U[g_min, g_max]."""
    W = params.w_c
    g = _gap(rng, cfg)
    ab = _start_target(rng, cfg, params, cfg.min_travel, params.x_max - W - g)
    if ab is None:
        return None
    x_A, x_B = ab
    return Layout("S3", x_A, x_B, (Stack(x_B + W + g, _height(rng, cfg)),))


def _sample_s4(rng, cfg, params):
    """Slot: stacks directly left and right of the target."""
    W = params.w_c
    g_l, g_r = _gap(rng, cfg), _gap(rng, cfg)
    ab = _start_target(rng, cfg, params,
                       max(cfg.min_travel, 2 * W + cfg.g_min + g_l),
                       params.x_max - W - g_r)
    if ab is None:
        return None
    x_A, x_B = ab
    return Layout("S4", x_A, x_B, (Stack(x_B - W - g_l, _height(rng, cfg)),
                                   Stack(x_B + W + g_r, _height(rng, cfg))))


def _sample_s5(rng, cfg, params):
    """Off-path: 1-2 stacks beyond the target, the first at least 2 m clear of it.
    They never block the route, so the best policy is to not lift higher than needed."""
    W = params.w_c
    g1 = float(rng.uniform(*cfg.off_path_gap))
    two = cfg.max_stacks >= 2 and rng.random() < 0.5
    g2 = _gap(rng, cfg) if two else 0.0
    room = W + g1 + (W + g2 if two else 0.0)      # space needed right of x_B
    ab = _start_target(rng, cfg, params, cfg.min_travel, params.x_max - room)
    if ab is None:
        return None
    x_A, x_B = ab
    c1 = x_B + W + g1
    stacks = [Stack(c1, _height(rng, cfg))]
    if two:
        stacks.append(Stack(c1 + W + g2, _height(rng, cfg)))
    return Layout("S5", x_A, x_B, tuple(stacks))


def _sample_s6(rng, cfg, params):
    """Mixed: a stack in the path, plus an S2, S3 or S4 arrangement at the target."""
    W, g = params.w_c, cfg.g_min
    variant = str(rng.choice(["S2", "S3", "S4"]))
    has_left = variant in ("S2", "S4")
    has_right = variant in ("S3", "S4")
    if 1 + has_left + has_right > cfg.max_stacks:
        return None
    g_l = _gap(rng, cfg) if has_left else 0.0
    g_r = _gap(rng, cfg) if has_right else 0.0

    # the path stack must fit between A and the target's left neighbour (or B itself)
    left_block = (W + g_l) if has_left else 0.0   # distance from x_B to the left neighbour
    offset = max(cfg.min_travel, 2 * W + 2 * g + left_block)
    x_B_hi = params.x_max - ((W + g_r) if has_right else 0.0)
    ab = _start_target(rng, cfg, params, offset, x_B_hi)
    if ab is None:
        return None
    x_A, x_B = ab

    c_path = float(rng.uniform(x_A + W + g, x_B - left_block - W - g))
    stacks = [Stack(c_path, _height(rng, cfg))]
    if has_left:
        stacks.append(Stack(x_B - W - g_l, _height(rng, cfg)))
    if has_right:
        stacks.append(Stack(x_B + W + g_r, _height(rng, cfg)))
    return Layout("S6", x_A, x_B, tuple(stacks))


SAMPLERS = {"S0": _sample_s0, "S1": _sample_s1, "S2": _sample_s2, "S3": _sample_s3,
            "S4": _sample_s4, "S5": _sample_s5, "S6": _sample_s6}


# ------------------------------------------------------------------ public API

def sample_kind(kind: str, rng, cfg: ScenarioConfig, params: CraneParams,
                max_tries: int = 100, check: bool = True) -> Layout:
    """Draw one layout of a given type, retrying infeasible draws.
    Raises RuntimeError if the type cannot fit under this config
    (e.g. S3 with fixed_AB = (0, 10): no room right of the target)."""
    sampler = SAMPLERS[kind]
    for _ in range(max_tries):
        layout = sampler(rng, cfg, params)
        if layout is not None:
            if check:
                v = validate_layout(layout, cfg, params)
                assert not v, f"{kind} sampler produced an invalid layout {layout}: {v}"
            return layout
    raise RuntimeError(f"layout type {kind} did not fit in {max_tries} tries "
                       f"under this config (fixed_AB={cfg.fixed_AB})")


def sample_layout(rng, cfg: ScenarioConfig, params: CraneParams,
                  max_tries: int = 100, check: bool = True) -> Layout:
    """Pick a layout type by cfg.weights, then draw a layout of that type.
    Infeasible draws are retried within the same type, so the type
    frequencies match the weights exactly."""
    kinds = [k for k, w in cfg.weights.items() if w > 0]
    probs = np.array([cfg.weights[k] for k in kinds], dtype=float)
    kind = str(rng.choice(kinds, p=probs / probs.sum()))
    return sample_kind(kind, rng, cfg, params, max_tries, check)


def canonical_layouts(params: CraneParams) -> dict:
    """Fixed evaluation set: start 0 m, target 10 m.

    S5 (off-path) cannot exist with B = 10 (no room beyond the target,
    none before A = 0), so it is replaced by a 2-high stack right next to A,
    which tests lifting before travelling.
    """
    W = params.w_c
    A, B = 0.0, 10.0
    return {
        "S0_open":        Layout("S0", A, B, ()),
        "S1_mid_1high":   Layout("S1", A, B, (Stack(5.0, 1),)),
        "S1_mid_2high":   Layout("S1", A, B, (Stack(5.0, 2),)),
        "S1_nearA_2high": Layout("S1", A, B, (Stack(A + W + 0.6, 2),)),
        "S2_behind":      Layout("S2", A, B, (Stack(B - W - 1.0, 2),)),
    }


def make_eval_set(kind: str, n: int, seed: int, cfg: ScenarioConfig,
                  params: CraneParams) -> tuple:
    """n held-out layouts of one type, reproducible from the seed."""
    rng = np.random.default_rng(seed)
    return tuple(sample_kind(kind, rng, cfg, params) for _ in range(n))