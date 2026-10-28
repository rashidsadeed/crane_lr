"""Physics tests for dynamics.py: period, steady angle, slow hoisting, mirror symmetry."""
import math
from dataclasses import replace
from dynamics import CraneParams, rk4_step, step, payload_center, swing

p = CraneParams()
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


# ---------------------------------------------------------------- Helpers
check("helper payload_center at rest",
      all(math.isclose(a, b, abs_tol=1e-12)
          for a, b in zip(payload_center((3.0, 0.0, 4.0, 0.0, 0.0, 0.0), p), (3.0, 5.0))),
      f"{payload_center((3.0, 0.0, 4.0, 0.0, 0.0, 0.0), p)} (expected (3.0, 5.0))")
d, v_d, A = swing((3.0, 0.0, 4.0, 0.0, 0.1, 0.0), p)
check("helper swing at turning point",
      math.isclose(d, 5 * math.sin(0.1), abs_tol=1e-12) and abs(v_d) < 1e-12
      and math.isclose(A, d, abs_tol=1e-12),
      f"d={d:.6f}, v_d={v_d:.2e}, A={A:.6f} (expected 0.499167, 0, 0.499167)")


# ---------------------------------------------------------------- Test 3: period
# Free swing from 1 deg at fixed length; measure the time between upward
# zero crossings of theta (linear interpolation between 0.01 s samples).
def measured_period(l, n_periods=5):
    s = (6.0, 0.0, l, 0.0, math.radians(1.0), 0.0)
    dt, t = p.dt_sim, 0.0
    crossings = []
    while len(crossings) < n_periods + 1:
        s_new = rk4_step(s, 0.0, 0.0, dt, p)
        if s[4] < 0.0 <= s_new[4]:                    # theta crosses zero upward
            frac = -s[4] / (s_new[4] - s[4])
            crossings.append(t + frac * dt)
        s, t = s_new, t + dt
    return (crossings[-1] - crossings[0]) / n_periods

for l in (1.5, 8.0):
    L = l + p.h_c / 2
    T_theory = 2 * math.pi * math.sqrt(L / p.g)
    T_meas = measured_period(l)
    err = abs(T_meas - T_theory) / T_theory
    check(f"T3 period at L={L:.1f} m", err < 0.005,
          f"measured {T_meas:.4f} s, theory {T_theory:.4f} s, error {err:.2e}")


# ---------------------------------------------------------------- Test 4: steady angle
# Constant trolley acceleration with light damping: the swing settles where
# g*sin(theta) + a*cos(theta) = 0, i.e. theta = -atan(a/g).
# rk4_step is called directly to bypass the 1 m/s speed limit (pure physics test).
p_damped = replace(p, zeta=0.05)
s = (0.0, 0.0, 4.0, 0.0, 0.0, 0.0)
for _ in range(20_000):                                # 200 s
    s = rk4_step(s, 0.5, 0.0, p.dt_sim, p_damped)
theta_expected = -math.atan(0.5 / p.g)
check("T4 steady angle under a_x = 0.5", math.isclose(s[4], theta_expected, abs_tol=1e-4),
      f"theta = {math.degrees(s[4]):.4f} deg, expected {math.degrees(theta_expected):.4f} deg")


# ---------------------------------------------------------------- Test 5: slow hoisting
# Hoist at a constant 0.05 m/s from l = 8 (L = 9) to l = 1.5 (L = 2.5).
# Adiabatic invariant for a slowly shortening pendulum: angle amplitude ~ L^(-3/4).
s = (6.0, 0.0, 8.0, -0.05, math.radians(2.0), 0.0)
dt = p.dt_sim
theta_peaks = []                                       # (L, |theta|) at each turning point
while s[2] > p.l_min:
    s_new = rk4_step(s, 0.0, 0.0, dt, p)
    if s[5] * s_new[5] < 0.0:                          # theta_dot changes sign: turning point
        theta_peaks.append((s_new[2] + p.h_c / 2, abs(s_new[4])))
    s = s_new
ratio = theta_peaks[-1][1] / theta_peaks[0][1]
L_ratio = theta_peaks[0][0] / theta_peaks[-1][0]       # use the actual L at the peaks
ratio_theory = L_ratio ** 0.75
err = abs(ratio - ratio_theory) / ratio_theory
check("T5 swing growth while hoisting", err < 0.05,
      f"amplitude ratio {ratio:.3f}, theory {ratio_theory:.3f}, error {err:.2%}")


# ---------------------------------------------------------------- Test 8: mirror symmetry
# Reflect about x = 6: x -> 12 - x, and negate x_dot, theta, theta_dot, a_x.
# The mirrored run must be the exact reflection of the original.
def mirror(s):
    x, x_dot, l, l_dot, th, th_dot = s
    return (12.0 - x, -x_dot, l, l_dot, -th, -th_dot)

commands = [(0.5, -0.4)] * 15 + [(0.2, 0.0)] * 10 + [(-0.5, 0.4)] * 20 + [(0.0, 0.0)] * 15
s_a = (4.0, 0.3, 5.0, 0.0, 0.05, -0.02)
s_b = mirror(s_a)
for a_x, a_l in commands:
    s_a, _ = step(s_a, a_x, a_l, p)
    s_b, _ = step(s_b, -a_x, a_l, p)
max_diff = max(abs(u - v) for u, v in zip(mirror(s_a), s_b))
check("T8 mirror symmetry", max_diff < 1e-9, f"max difference {max_diff:.2e}")

print("\nALL PASSED" if all(results) else f"\n{results.count(False)} CHECK(S) FAILED")