"""Draw 3 random layouts of each type (S0-S6) into layouts.png for a visual check."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # add src/ to the search path

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.patches import Rectangle                              # noqa: E402
import numpy as np                                                    # noqa: E402

from dynamics import CraneParams                                      # noqa: E402
from scenarios import ScenarioConfig, SAMPLERS, sample_kind, layout_rects  # noqa: E402

p, cfg = CraneParams(), ScenarioConfig()
W, h = p.w_c, p.h_c
rng = np.random.default_rng(7)
n_per_kind = 3

fig, axes = plt.subplots(len(SAMPLERS), n_per_kind, figsize=(15, 2.0 * len(SAMPLERS)),
                         sharex=True, sharey=True)
for row, kind in enumerate(SAMPLERS):
    for col in range(n_per_kind):
        ax = axes[row, col]
        L = sample_kind(kind, rng, cfg, p)
        ax.axhline(0, color="black", lw=1)
        ax.axvspan(p.x_min - W / 2, p.x_max + W / 2, color="0.95", zorder=0)  # floor workspace
        for r in layout_rects(L, p):                                        # obstacle stacks
            ax.add_patch(Rectangle((r.x_lo, r.z_lo), W, r.z_hi - r.z_lo,
                                   facecolor="#D85A30", edgecolor="black", lw=0.6))
        ax.add_patch(Rectangle((L.x_A - W / 2, 0), W, h,                    # container at A
                               facecolor="#7F77DD", edgecolor="black", lw=0.6))
        ax.add_patch(Rectangle((L.x_B - W / 2, 0), W, h, fill=False,        # drop zone at B
                               edgecolor="#1D9E75", lw=1.5, ls="--"))
        ax.text(L.x_A, h + 0.3, "A", ha="center", fontsize=8)
        ax.text(L.x_B, h + 0.3, "B", ha="center", fontsize=8, color="#0F6E56")
        ax.set_title(f"{kind}: A={L.x_A:.1f}, B={L.x_B:.1f}", fontsize=9)
        ax.set_xlim(-1.6, 13.6)
        ax.set_ylim(-0.3, 5.0)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

fig.suptitle("Random layouts per type: purple = container at A, dashed = target B, "
             "orange = stacks", fontsize=11)
fig.tight_layout()
out = pathlib.Path(__file__).resolve().parent / "layouts.png"
fig.savefig(out, dpi=110)
print(f"saved {out}")