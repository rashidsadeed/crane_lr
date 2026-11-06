"""Plot training progress from one or more run folders (their log.jsonl).

Usage, from src/:
    python -m tests.plot_training runs/seed1                     one run
    python -m tests.plot_training runs/seed1 runs/seed2 runs/seed3   several seeds, overlaid

Four panels against total training episodes, with the curriculum stages shaded:
    1. success rate: during training (with exploration) and greedy on held-out layouts
    2. reverse curriculum: start distance from the target, and share of mid-air starts
    3. how episodes ended (stacked shares), for the first run
    4. greedy delivery time on held-out layouts
Saved as training_curves.png in the first run folder (or --out).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # add src/ to the search path

import argparse                                                      # noqa: E402
import json                                                          # noqa: E402

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402

OUTCOMES = ["success", "bad_landing", "floor_collision", "collision", "end_stop",
            "hard_top_stop", "timeout"]
OUTCOME_COLOURS = ["#1D9E75", "#EF9F27", "#D85A30", "#A32D2D", "#7F77DD", "#D4537E", "#888780"]
STAGE_SHADES = ["#F1EFE8", "#FFFFFF"]
SEED_COLOURS = ["#534AB7", "#0F6E56", "#993C1D", "#185FA5", "#993556"]


def read_log(run: pathlib.Path):
    rows = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines() if line.strip()]
    return [r for r in rows if "train_success" in r], [r for r in rows if "end" in r]


def stage_spans(rows):
    """[(stage, first total episode, last total episode)] in order."""
    spans = []
    for r in rows:
        x = r["total_ep"]
        if spans and spans[-1][0] == r["stage"]:
            spans[-1][2] = x
        else:
            spans.append([r["stage"], x - (rows[0]["ep"] if not spans else 0), x])
    return spans


def shade(ax, spans, label=False):
    for i, (stage, a, b) in enumerate(spans):
        ax.axvspan(a, b, color=STAGE_SHADES[i % 2], zorder=0, lw=0)
        if label:
            ax.text((a + b) / 2, 1.02, stage, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, color="#444441")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders, e.g. runs/seed1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = [pathlib.Path(r) for r in args.runs]
    logs = [read_log(r) for r in runs]
    if not logs[0][0]:
        sys.exit(f"no progress rows yet in {runs[0] / 'log.jsonl'}")

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1, 1, 1], "hspace": 0.25})
    spans = stage_spans(logs[0][0])
    for k, ax in enumerate(axes):
        shade(ax, spans, label=(k == 0))

    # 1. success rates
    ax = axes[0]
    for i, (rows, _) in enumerate(logs):
        c = SEED_COLOURS[i % len(SEED_COLOURS)]
        name = runs[i].name
        x = [r["total_ep"] for r in rows]
        ax.plot(x, [100 * r["train_success"] for r in rows], color=c, lw=1, alpha=0.45,
                label=f"{name}: training (with exploration)")
        g = [(r["total_ep"], 100 * r["greedy_success"]) for r in rows if "greedy_success" in r]
        if g:
            gx, gy = zip(*g)
            ax.plot(gx, gy, color=c, lw=0, marker="o", ms=4, label=f"{name}: greedy, held-out layouts")
    ax.axhline(90, color="#1D9E75", lw=1, ls="--")
    ax.text(1.0, 90, " target 90%", transform=ax.get_yaxis_transform(), va="center",
            fontsize=8, color="#0F6E56")
    ax.set_ylabel("success [%]")
    ax.set_ylim(-3, 103)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_title("SARSA(λ) training progress" + (f"  ({len(runs)} seeds)" if len(runs) > 1 else ""),
                 fontsize=12, pad=18)

    # 2. reverse curriculum
    ax = axes[1]
    ax2 = ax.twinx()
    for i, (rows, _) in enumerate(logs):
        c = SEED_COLOURS[i % len(SEED_COLOURS)]
        x = [r["total_ep"] for r in rows]
        ax.step(x, [r["start_dist"] for r in rows], where="post", color=c, lw=1.4)
        ax2.step(x, [100 * r["p_air"] for r in rows], where="post", color=c, lw=1.2, ls=":")
    ax.set_ylabel("start distance\nfrom target [m] (solid)")
    ax2.set_ylabel("mid-air starts [%]\n(dotted)")
    ax.set_ylim(0, 13)
    ax2.set_ylim(-3, 103)

    # 3. outcomes of the first run
    ax = axes[2]
    rows = logs[0][0]
    x = [r["total_ep"] for r in rows]
    shares = np.array([[r["outcomes"].get(o, 0) for o in OUTCOMES] for r in rows], dtype=float)
    shares = 100 * shares / shares.sum(axis=1, keepdims=True)
    ax.stackplot(x, shares.T, colors=OUTCOME_COLOURS, labels=[o.replace("_", " ") for o in OUTCOMES],
                 alpha=0.9)
    ax.set_ylabel(f"how episodes ended\n[%], {runs[0].name}")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5, loc="upper left", ncol=4, framealpha=0.9)

    # 4. delivery time
    ax = axes[3]
    for i, (rows, _) in enumerate(logs):
        g = [(r["total_ep"], r["greedy_time"]) for r in rows if r.get("greedy_time")]
        if g:
            gx, gy = zip(*g)
            ax.plot(gx, gy, color=SEED_COLOURS[i % len(SEED_COLOURS)], marker="o", ms=4, lw=1,
                    label=runs[i].name)
    ax.set_ylabel("greedy delivery\ntime [s]")
    ax.set_xlabel("training episodes (all stages)")
    if any(r.get("greedy_time") for rows, _ in logs for r in rows):
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "no greedy evaluations yet (they start once all episodes start at A)",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#888780")

    out = pathlib.Path(args.out) if args.out else runs[0] / "training_curves.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()