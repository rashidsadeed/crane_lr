"""Plot one baseline episode for the report.

Draws the scene (stacks, target, container snapshots every 3 s and the path of the
container centre), then the container's x position, bottom height and swing amplitude
over time, with the controller phases shaded (lift, travel, lower, set-down).

Usage, from src/:
    python -m tests.plot_baseline [layout]

Layouts: the canonical evaluation set (start A = 0 m, target B = 10 m):
    S0_open          open floor, no stacks
    S1_mid_1high     1-high stack at 5 m, in the path
    S1_mid_2high     2-high stack at 5 m, in the path            (default)
    S1_nearA_2high   2-high stack right next to the start: lift before travelling
    S2_behind        2-high stack directly left of the target: lower only after passing it

The figure is saved next to this script as baseline_<layout>.png.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # add src/ to the search path

import math                                                          # noqa: E402

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.patches import Polygon, Rectangle                    # noqa: E402

import dynamics as dyn                                               # noqa: E402
import geometry as geo                                               # noqa: E402
from baseline import BaselineController, run_episode                 # noqa: E402
from env import CraneEnv                                             # noqa: E402
from scenarios import canonical_layouts, layout_rects                # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "S1_mid_2high"
env = CraneEnv()
p = env.p
layout = canonical_layouts(p)[name]
ep, traj = run_episode(env, BaselineController(p), layout=layout, record=True)

t = [k * p.dt_ctrl for k in range(len(traj))]
states = [tuple(float(v) for v in o[:6]) for o, _ in traj]
phases = [ph for _, ph in traj]
xp = [dyn.payload_center(s, p)[0] for s in states]
zb = [p.H - s[2] - p.h_c for s in states]
A = [dyn.swing(s, p)[2] for s in states]

PHASE_COLOURS = {"lift": "#EEEDFE", "travel": "#E1F5EE", "lower": "#FAEEDA", "set_down": "#FAECE7"}

fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1, 1, 1], hspace=0.45)

# ---- scene
ax = fig.add_subplot(gs[0])
ax.axhline(0, color="black", lw=1)
for r in layout_rects(layout, p):
    ax.add_patch(Rectangle((r.x_lo, r.z_lo), r.x_hi - r.x_lo, r.z_hi - r.z_lo,
                           facecolor="#D85A30", edgecolor="black", lw=0.6))
ax.add_patch(Rectangle((layout.x_B - p.w_c / 2, 0), p.w_c, p.h_c, fill=False,
                       edgecolor="#1D9E75", lw=1.5, ls="--"))
for k in range(0, len(states), 30):                     # container snapshots every 3 s
    ax.add_patch(Polygon(geo.container_corners(states[k], p), closed=True, fill=False,
                         edgecolor="#534AB7", lw=0.6, alpha=0.5))
ax.plot(xp, [z + p.h_c / 2 for z in zb], color="#534AB7", lw=1.5, label="container centre")
ax.set_xlim(-1.5, 13.5)
ax.set_ylim(-0.3, 8.5)
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.set_title(f"Baseline, layout {name}: {ep['outcome']} in {ep['time']:.1f} s "
             f"(placement error {100 * ep['landing']['err_x']:.1f} cm, "
             f"residual swing {100 * ep['landing']['A']:.1f} cm)", fontsize=11)
ax.legend(loc="upper right", fontsize=8)


def shade(ax):
    start = 0
    for k in range(1, len(phases) + 1):
        if k == len(phases) or phases[k] != phases[start]:
            ax.axvspan(t[start], t[min(k, len(t) - 1)], color=PHASE_COLOURS[phases[start]], lw=0)
            start = k


for i, (series, label, ref) in enumerate([
        (xp, "container x [m]", layout.x_B),
        (zb, "bottom height [m]", None),
        (A, "swing amplitude [m]", env.task.tol_A)]):
    ax = fig.add_subplot(gs[i + 1])
    shade(ax)
    ax.plot(t, series, color="#2C2C2A", lw=1.2)
    if ref is not None:
        ax.axhline(ref, color="#1D9E75", lw=1, ls="--")
    ax.set_ylabel(label, fontsize=9)
    ax.set_xlim(0, t[-1])
    ax.tick_params(labelsize=8)
    if i == 2:
        ax.set_xlabel("time [s]")
        handles = [Rectangle((0, 0), 1, 1, color=c) for c in PHASE_COLOURS.values()]
        ax.legend(handles, PHASE_COLOURS.keys(), loc="upper right", fontsize=8, ncol=4)

out = pathlib.Path(__file__).resolve().parent / f"baseline_{name}.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved {out}")