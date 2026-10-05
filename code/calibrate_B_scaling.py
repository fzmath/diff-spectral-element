"""Numerical calibration of mu_max(B) scaling in 1D/2D/3D (SEM, GLL mass).
Outputs results/calibrate_B_scaling.json (used by Theorem 2 / Assumption H4)."""
import json, os
import numpy as np
from sem_lib import build_1d, build_2d, build_3d

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def mumax_1d(p, E):
    K, M, nodes, Dg = build_1d(p, E)
    d = np.sqrt(np.diag(M))
    B = K / np.outer(d, d)
    return np.linalg.eigvalsh(B).max()


def mumax_2d(p, E):
    K, M, x, y, nodes, n1d, Dg = build_2d(p, E)
    d = np.sqrt(np.diag(M))
    B = K / np.outer(d, d)
    return np.linalg.eigvalsh(B).max()


def mumax_3d(p, E):
    K, M, x, y, z = build_3d(p, E)
    d = np.sqrt(np.diag(M))
    B = K / np.outer(d, d)
    return np.linalg.eigvalsh(B).max()


def slope(xs, ys):
    lx, ly = np.log(xs), np.log(ys)
    return (ly[-1] - ly[0]) / (lx[-1] - lx[0])


print("== 1D, E=2 (h=0.5), p-refinement ==")
ps = [2, 4, 8, 16, 32]
mm = [mumax_1d(p, 2) for p in ps]
for p, m in zip(ps, mm):
    print(f"  p={p:3d}  mu_max={m:12.4e}")
print(f"  fitted exponent a ~ {slope(ps, mm):.2f}  (mu_max ~ p^a)")

print("== 1D, p=4, h-refinement ==")
Es = [2, 4, 8, 16]
mm = [mumax_1d(4, E) for E in Es]
for E, m in zip(Es, mm):
    print(f"  E={E:3d} (h=1/E)  mu_max={m:12.4e}")
print(f"  fitted exponent b ~ {slope(Es, mm):.2f}  (mu_max ~ h^-b, h=1/E)")

print("== 2D, E=2 (h=0.5), p-refinement ==")
ps = [2, 4, 8, 16]
mm = [mumax_2d(p, 2) for p in ps]
for p, m in zip(ps, mm):
    print(f"  p={p:3d}  mu_max={m:12.4e}")
print(f"  fitted exponent a ~ {slope(ps, mm):.2f}")

print("== 2D, p=4, h-refinement ==")
Es = [2, 4, 8]
mm = [mumax_2d(4, E) for E in Es]
for E, m in zip(Es, mm):
    print(f"  E={E:3d}  mu_max={m:12.4e}")
print(f"  fitted exponent b ~ {slope(Es, mm):.2f}")

print("== 3D, E=2 (h=0.5), p-refinement ==")
ps = [2, 4, 6]
mm = [mumax_3d(p, 2) for p in ps]
for p, m in zip(ps, mm):
    print(f"  p={p:3d}  mu_max={m:12.4e}")
print(f"  fitted exponent a ~ {slope(ps, mm):.2f}")

print("== 3D, p=2, h-refinement ==")
Es = [2, 4]
mm = [mumax_3d(2, E) for E in Es]
for E, m in zip(Es, mm):
    print(f"  E={E:3d}  mu_max={m:12.4e}")
print(f"  fitted exponent b ~ {slope(Es, mm):.2f}")

# ---- save calibration for the manuscript (Theorem 2 / H4) ----
p1 = [2, 4, 8, 16, 32]
m1 = [mumax_1d(p, 2) for p in p1]
out = {
    "1d_p_ref": [{"p": p, "mu_max": m} for p, m in zip(p1, m1)],
    "1d_p_exponent": round(slope(p1, m1), 2),
    "1d_p16_to_32_exponent": round(np.log(m1[-1] / m1[-2]) / np.log(32 / 16), 2),
    "1d_h_ref": [{"E": E, "mu_max": m} for E, m in zip([2, 4, 8, 16],
                  [mumax_1d(4, E) for E in [2, 4, 8, 16]])],
    "1d_h_exponent": round(slope([2, 4, 8, 16],
                          [mumax_1d(4, E) for E in [2, 4, 8, 16]]), 2),
    "2d_p_exponent": round(slope([2, 4, 8, 16],
                          [mumax_2d(p, 2) for p in [2, 4, 8, 16]]), 2),
    "2d_h_exponent": round(slope([2, 4, 8],
                          [mumax_2d(4, E) for E in [2, 4, 8]]), 2),
    "3d_p_exponent": round(slope([2, 4, 6],
                          [mumax_3d(p, 2) for p in [2, 4, 6]]), 2),
    "3d_h_exponent": round(slope([2, 4], [mumax_3d(2, E) for E in [2, 4]]), 2),
}
with open(os.path.join(OUT, "calibrate_B_scaling.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("saved calibrate_B_scaling.json")
