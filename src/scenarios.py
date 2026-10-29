from typing import NamedTuple
from dataclasses import dataclass
import numpy as np

from geometry import stack_rect
from dynamics import CraneParams

class Stack(NamedTuple):
    """A stack of containers, represented as a rectangle in the x,z plane"""
    center: float  # x coordinate of the center of the stack
    n_high: int    # number of containers stacked vertically

class Layout(NamedTuple):
    """A layout of stacks, represented as a list of rectangles in the x,z plane"""
    kind: str # "S0" .... "S6"
    x_A: float # start: container on the floor
    x_B: float # targer: container must be placed here
    stacks: tuple

def layout_rects(layout: Layout, params: CraneParams) -> tuple:
    """Return a tuple of Rects representing the stacks in the layout"""
    return tuple(stack_rect(st.center, st.n_high, params) for st in layout.stacks)

DEFAULT_WEIGHTS = {"S0": 0.05, "S1": 0.20, "S2": 0.15, "S3": 0.15, "S4": 0.15, "S5": 0.10, "S6": 0.20}

@dataclass(frozen=True)
class ScenarioConfig:
    weights: dict # {"S0": 0.05, "S1": 0.20, ...}; kinds left out are never draws
    g_min: float = 0.4 # smallest gap between naightbouring footprints [m]
    g_max: float = 1.5 # largest gap for "adjacent" placements [m]
    max_high: int = 2 #tallest stack allowed
    max_stacks: int = 3 #maximum number of stacks allowed
    x_A_range: tuple = (0.0, 2.5)
    min_travel: float = 4.0 # x_B >= x_A + min_travel
    fixed_AB: tuple | None = None # e.g. (0.0, 10.0) for curriculum stage C0


def validate_layout(layout: Layout, cfg: ScenarioConfig, params: CraneParams) -> list:
    """
    return a list of rule violations (Strings); an empty list means the layout is valid.
    """
    violations = []
    #start and target
    if (layout.x_B >= layout.x_A + cfg.min_travel) or (layout.x_B <= params.x_max):
        violations.append("target out of range")
    elif cfg.x_A_range[0] <= layout.x_A <= cfg.x_A_range[1]:
        violations.append("start out of range")

    #stacks
    if len(layout.stacks) <= cfg.max_stacks:
        violations.append("number of stacks out of bounds")
    elif (i in layout.)