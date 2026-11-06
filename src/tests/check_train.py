"""Checks for rl_agent/train.py: stage chaining, checkpoints, and exact resume after a crash.

Runs two short trainings (about 1-2 minutes in total):
  A  uninterrupted:  C0 then C1, 300 episodes each
  B  same settings, killed part-way through, then resumed with --resume
and checks that B ends with exactly the same weights as A.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # add src/ to the search path

import json                                                          # noqa: E402
import shutil                                                        # noqa: E402
import subprocess                                                    # noqa: E402
import time                                                          # noqa: E402

import numpy as np                                                   # noqa: E402

SRC = pathlib.Path(__file__).resolve().parents[1]
TMP = SRC / "tests" / "_train_test"
shutil.rmtree(TMP, ignore_errors=True)
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def cmd(*extra):
    return [sys.executable, "-m", "rl_agent.train", "--stages", "C0,C1", "--max-episodes", "300",
            "--eval-every", "100", "--checkpoint-minutes", "0.002", *extra]


# A: uninterrupted
t0 = time.time()
a = subprocess.run(cmd("--seed", "7", "--out", str(TMP / "A")), cwd=SRC, capture_output=True, text=True)
check("T1 run finishes", a.returncode == 0 and "done:" in a.stdout,
      f"{time.time() - t0:.0f} s" if a.returncode == 0 else a.stderr[-500:])
runA = TMP / "A" / "seed7"
check("T2 both stages ran, in order", "=== stage C0" in a.stdout and "=== stage C1" in a.stdout
      and a.stdout.index("=== stage C0") < a.stdout.index("=== stage C1"))
rows = [json.loads(l) for l in (runA / "log.jsonl").read_text().splitlines()]
check("T3 log has periodic rows and a stage-end row per stage",
      sum("end" in r for r in rows) == 2 and sum("train_success" in r for r in rows) == 6,
      f"{len(rows)} rows")
check("T4 checkpoint files written", all((runA / f).exists() for f in ("state.pkl", "weights.npz", "final.npz")))

# B: same run, killed part-way through, then resumed
p = subprocess.Popen(cmd("--seed", "7", "--out", str(TMP / "B")), cwd=SRC,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
runB = TMP / "B" / "seed7"
while True:                                   # wait until C1 has started and a checkpoint exists
    line = p.stdout.readline()
    if "=== stage C1" in line or line == "":
        break
time.sleep(3)
p.kill()
p.wait()
check("T5 killed part-way through, with a checkpoint on disk", (runB / "state.pkl").exists()
      and not (runB / "final.npz").exists())

b = subprocess.run([sys.executable, "-m", "rl_agent.train", "--resume", str(runB)],
                   cwd=SRC, capture_output=True, text=True)
check("T6 resume finishes", b.returncode == 0 and "resuming" in b.stdout and "done:" in b.stdout,
      b.stdout.splitlines()[0] if b.returncode == 0 else b.stderr[-500:])
wA = np.load(runA / "final.npz")["w"]
wB = np.load(runB / "final.npz")["w"]
check("T7 resumed run ends with exactly the same weights", np.array_equal(wA, wB),
      f"max difference {np.abs(wA - wB).max():.2e}")

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")