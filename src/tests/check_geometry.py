"""Hand-checked tests for geometry.py."""
import math
from dynamics import CraneParams, payload_center
from geometry import (Rect, stack_rect, rect_corners, container_corners,
                      container_hits_rect, first_collision, lowest_z)

p = CraneParams()
M = 0.10                                   # safety margin [m]
TOL = 1e-9
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def close_pts(a, b, tol=TOL):
    return len(a) == len(b) and all(math.isclose(p1[0], p2[0], abs_tol=tol) and
                                    math.isclose(p1[1], p2[1], abs_tol=tol) for p1, p2 in zip(a, b))


def state(x, l, theta_deg=0.0):
    return (x, 0.0, l, 0.0, math.radians(theta_deg), 0.0)


def fmt(pts):
    return "[" + ", ".join(f"({x:.4f}, {z:.4f})" for x, z in pts) + "]"


stack = stack_rect(5.0, 2, p)              # 2-high stack: x 3.8..6.2, z 0..4

# ---------------------------------------------------------------- stack_rect / rect_corners
check("G1 stack_rect", stack == Rect(3.8, 6.2, 0.0, 4.0), f"{stack}")
expected = ((3.8, 4.0), (6.2, 4.0), (6.2, 0.0), (3.8, 0.0))
check("G2 rect_corners order TL,TR,BR,BL", close_pts(rect_corners(stack), expected),
      fmt(rect_corners(stack)))

# ---------------------------------------------------------------- container_corners
# theta = 0, hook at (5, 4)  ->  l = H - 4 = 6
c0 = container_corners(state(5.0, 6.0), p)
expected = ((3.8, 4.0), (6.2, 4.0), (6.2, 2.0), (3.8, 2.0))
check("G3 container corners at theta=0, order TL,TR,BR,BL", close_pts(c0, expected), fmt(c0))

# theta = 10 deg: edge lengths kept, centre of corners = payload_center
s10 = state(6.0, 4.0, 10.0)
c10 = container_corners(s10, p)
top = math.dist(c10[0], c10[1])
side = math.dist(c10[1], c10[2])
check("G4 shape preserved when tilted", math.isclose(top, 2.4, abs_tol=TOL) and
      math.isclose(side, 2.0, abs_tol=TOL), f"top {top:.6f} (2.4), side {side:.6f} (2.0)")
cx = sum(q[0] for q in c10) / 4
cz = sum(q[1] for q in c10) / 4
pc = payload_center(s10, p)
check("G5 mean of corners = payload_center", close_pts(((cx, cz),), (pc,)),
      f"({cx:.6f}, {cz:.6f}) vs ({pc[0]:.6f}, {pc[1]:.6f})")

# ---------------------------------------------------------------- vertical clearance
# container centred over the stack; bottom = H - l - h_c
for bottom, should_hit in ((4.11, False), (4.09, True)):
    c = container_corners(state(5.0, 10.0 - bottom - 2.0), p)
    hit = container_hits_rect(c, stack, M, 0.0)
    check(f"G6 above stack, bottom at {bottom} m", hit == should_hit,
          f"hit={hit} (expected {should_hit})")

# ---------------------------------------------------------------- side clearance
# container left edge = trolley x - 1.2; stack right edge = 6.2
for gap, should_hit in ((0.11, False), (0.09, True)):
    c = container_corners(state(6.2 + gap + 1.2, 5.0), p)
    hit = container_hits_rect(c, stack, M, 0.0)
    check(f"G7 beside stack, gap {gap} m", hit == should_hit,
          f"hit={hit} (expected {should_hit})")

# ---------------------------------------------------------------- tilted near miss
# theta = 10 deg, container right of and above the stack's top-right corner.
# Bounding boxes come within the margin, but the true gap is 0.149 m:
# only the container's own axes (u, d) can see it.
th = math.radians(10.0)
c_near = container_corners(state(6.5, 3.8, 10.0), p)
xs = [q[0] for q in c_near]
zs = [q[1] for q in c_near]
bbox_close = not (max(xs) + M < stack.x_lo or stack.x_hi + M < min(xs) or
                  max(zs) + M < stack.z_lo or stack.z_hi + M < min(zs))
hit = container_hits_rect(c_near, stack, M, th)
check("G8 tilted near miss: bounding boxes within margin", bbox_close,
      f"container bottom-left corner {fmt((c_near[3],))}")
check("G9 tilted near miss: rectangles clear", hit is False, f"hit={hit} (expected False)")

c_hit = container_corners(state(6.3, 3.8, 10.0), p)     # 0.2 m further left: corner over the stack
hit = container_hits_rect(c_hit, stack, M, th)
check("G10 tilted, moved 0.2 m left: hit", hit is True, f"hit={hit} (expected True)")

# ---------------------------------------------------------------- first_collision
stacks = [stack_rect(1.5, 1, p), stack]                 # index 0: 1-high at 1.5, index 1: 2-high at 5
c = container_corners(state(5.0, 4.1), p)               # bottom at 3.9, over stack 1
idx = first_collision(c, stacks, M, 0.0)
check("G11 first_collision finds stack 1", idx == 1, f"index={idx} (expected 1)")
c = container_corners(state(5.0, 3.5), p)               # bottom at 4.5, clear of both
idx = first_collision(c, stacks, M, 0.0)
check("G12 first_collision returns None when clear", idx is None, f"index={idx} (expected None)")

# ---------------------------------------------------------------- floor
z0 = lowest_z(container_corners(state(10.0, 8.0), p))
z5 = lowest_z(container_corners(state(10.0, 8.0, 5.0), p))
check("G13 flat on the floor at l=8", math.isclose(z0, 0.0, abs_tol=TOL), f"lowest z = {z0:.6f}")
check("G14 tilted corner dips below floor", z5 < 0.0, f"lowest z = {z5:.4f} (expected < 0)")

print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")