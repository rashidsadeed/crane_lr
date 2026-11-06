"""Tests for scenarios.py: validity, type properties, frequencies, determinism, config."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # add src/ to the search path

from collections import Counter                                       # noqa: E402

import numpy as np                                                    # noqa: E402

from dynamics import CraneParams                                      # noqa: E402
from scenarios import (ScenarioConfig, SAMPLERS, sample_kind,         # noqa: E402
                       sample_layout, validate_layout, canonical_layouts,
                       make_eval_set, layout_rects)

p = CraneParams()
cfg = ScenarioConfig()
W = p.w_c
EPS = 1e-9
N = 1000
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def draw(kind, n=N, seed=0, config=cfg):
    rng = np.random.default_rng(seed)
    return [sample_kind(kind, rng, config, p, check=False) for _ in range(n)]


def adjacent_gap(left_center, right_center):
    return right_center - left_center - W


# ---------------------------------------------------------------- validity, every type
samples = {}
for kind in SAMPLERS:
    samples[kind] = draw(kind)
    bad = [(L, validate_layout(L, cfg, p)) for L in samples[kind]
           if validate_layout(L, cfg, p)]
    check(f"V {kind}: {N} layouts all valid", not bad,
          f"first invalid: {bad[0]}" if bad else "")

# ---------------------------------------------------------------- type properties
check("P S0 has no stacks", all(len(L.stacks) == 0 for L in samples["S0"]))

check("P S1 stack strictly between A and B",
      all(len(L.stacks) == 1 and L.x_A < L.stacks[0].center < L.x_B for L in samples["S1"]))

check("P S2 stack is the target's left neighbour, gap in [g_min, g_max]",
      all(len(L.stacks) == 1 and
          cfg.g_min - EPS <= adjacent_gap(L.stacks[0].center, L.x_B) <= cfg.g_max + EPS
          for L in samples["S2"]))

check("P S3 stack is the target's right neighbour, gap in [g_min, g_max]",
      all(len(L.stacks) == 1 and
          cfg.g_min - EPS <= adjacent_gap(L.x_B, L.stacks[0].center) <= cfg.g_max + EPS
          for L in samples["S3"]))

check("P S4 stacks on both sides of the target, gaps in [g_min, g_max]",
      all(len(L.stacks) == 2 and
          cfg.g_min - EPS <= adjacent_gap(L.stacks[0].center, L.x_B) <= cfg.g_max + EPS and
          cfg.g_min - EPS <= adjacent_gap(L.x_B, L.stacks[1].center) <= cfg.g_max + EPS
          for L in samples["S4"]))

check("P S5 all stacks beyond B, first at least 2 m clear",
      all(1 <= len(L.stacks) <= 2 and all(st.center > L.x_B for st in L.stacks) and
          adjacent_gap(L.x_B, min(st.center for st in L.stacks)) >= cfg.off_path_gap[0] - EPS
          for L in samples["S5"]))

def target_neighbour(st, x_B):
    gap = adjacent_gap(st.center, x_B) if st.center < x_B else adjacent_gap(x_B, st.center)
    return cfg.g_min - EPS <= gap <= cfg.g_max + EPS

# the sampler puts the path stack first, then the target's neighbours
check("P S6 path stack between A and B, plus 1-2 target neighbours",
      all(2 <= len(L.stacks) <= 3 and
          L.x_A < L.stacks[0].center < L.x_B and
          all(target_neighbour(st, L.x_B) for st in L.stacks[1:])
          for L in samples["S6"]))

variants = Counter(len(L.stacks) for L in samples["S6"])
check("P S6 uses all variants (2 and 3 stacks)", variants[2] > 0 and variants[3] > 0,
      f"stack counts {dict(variants)}")

heights = Counter(st.n_high for L in samples["S1"] for st in L.stacks)
check("P heights 1 and 2 both drawn", set(heights) == {1, 2}, f"{dict(heights)}")

# ---------------------------------------------------------------- frequencies
rng = np.random.default_rng(1)
counts = Counter(sample_layout(rng, cfg, p, check=False).kind for _ in range(10_000))
worst = max(abs(counts[k] / 10_000 - w) for k, w in cfg.weights.items())
check("F type frequencies within 1.5 points of weights", worst < 0.015,
      f"worst deviation {worst * 100:.2f} points; counts {dict(sorted(counts.items()))}")

# ---------------------------------------------------------------- determinism
a = make_eval_set("S6", 50, seed=123, cfg=cfg, params=p)
b = make_eval_set("S6", 50, seed=123, cfg=cfg, params=p)
c = make_eval_set("S6", 50, seed=124, cfg=cfg, params=p)
check("D same seed -> identical layouts", a == b)
check("D different seed -> different layouts", a != c)

# ---------------------------------------------------------------- config respected
cfg_wide = ScenarioConfig(g_min=1.0)
ok = True
for kind in SAMPLERS:
    for L in draw(kind, n=300, config=cfg_wide):
        ok &= not validate_layout(L, cfg_wide, p)
check("C g_min = 1.0 respected by every type", ok)

cfg_low = ScenarioConfig(max_high=1)
check("C max_high = 1 gives only 1-high stacks",
      all(st.n_high == 1 for kind in SAMPLERS for L in draw(kind, n=200, config=cfg_low)
          for st in L.stacks))

cfg_c0 = ScenarioConfig(weights={"S0": 1.0}, fixed_AB=(0.0, 10.0))
rng = np.random.default_rng(0)
Ls = [sample_layout(rng, cfg_c0, p) for _ in range(20)]
check("C fixed_AB = (0, 10) gives exactly A = 0, B = 10",
      all(L.x_A == 0.0 and L.x_B == 10.0 and not L.stacks for L in Ls))

try:
    sample_kind("S3", np.random.default_rng(0), cfg_c0, p)
    raised = False
except RuntimeError:
    raised = True
check("C impossible type under fixed_AB raises a clear error", raised)

# ---------------------------------------------------------------- canonical set
canon = canonical_layouts(p)
bad = {k: validate_layout(L, cfg, p) for k, L in canon.items() if validate_layout(L, cfg, p)}
check("K canonical layouts all valid", not bad, f"{bad}" if bad else f"{len(canon)} layouts")
check("K layout_rects gives one Rect per stack",
      len(layout_rects(canon["S1_mid_2high"], p)) == 1 and
      layout_rects(canon["S1_mid_2high"], p)[0].z_hi == 4.0)

print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")