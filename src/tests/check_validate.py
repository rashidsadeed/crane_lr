import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

"""Hand-checked cases for validate_layout."""
from dynamics import CraneParams
from scenarios import Layout, Stack, ScenarioConfig, validate_layout


p = CraneParams()
cfg = ScenarioConfig()

# (layout, number of violations expected, description)
cases = [
    (Layout("S1", 1.0, 10.0, (Stack(5.0, 2),)),  0, "valid S1"),
    (Layout("S2", 1.0, 10.0, (Stack(7.3, 1),)),  1, "stack too close to target"),
    (Layout("S1", 1.0, 10.0, (Stack(2.0, 1),)),  1, "stack overlaps start"),
    (Layout("S3", 1.0, 9.0,  (Stack(12.5, 1),)), 1, "stack outside workspace"),
    (Layout("S0", 2.0, 5.5,  ()),                1, "travel too short"),
    (Layout("S1", 1.0, 10.0, (Stack(5.0, 3),)),  1, "stack too tall"),
    (Layout("S1", 3.0, 10.0, ()),                1, "start out of range"),
]

all_ok = True
for layout, n_expected, description in cases:
    v = validate_layout(layout, cfg, p)
    ok = len(v) == n_expected
    all_ok &= ok
    print(f"{'PASS' if ok else 'FAIL'}  {description:28s} -> {v}")

print("\nALL PASSED" if all_ok else "\nSOME CASES FAILED")